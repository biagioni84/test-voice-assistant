"""Calibra rag.min_score y rag.ambiguity_threshold contra tests/calibration_questions.yaml de forma
principista (barrido de umbrales + precisión/recall), en vez de a mano contra casos puntuales -- ver
README, sección "parámetros calibrados vs. arquitectura estable". Escribe el resultado a
calibration.toml, que voice/config.py carga y superpone sobre los defaults de config.toml (ver
voice/config.py: _apply_calibration).

Atado a los documentos de docs/ (genéricos, de prueba, ver tests/calibration_questions.yaml) --
cuando lleguen documentos reales: reescribir tests/calibration_questions.yaml con preguntas sobre
ESOS documentos, y correr este script de nuevo. No tiene sentido calibrar contra los docs viejos.

Solo levanta el Retriever (embeddings + reranker), no el resto del pipeline (Whisper/LLM/TTS/wake
word) -- no hace falta y así corre en segundos, no minutos.

    python scripts/calibrate.py                    # calibra y escribe calibration.toml
    python scripts/calibrate.py --dry-run           # solo el reporte, no escribe nada
    python scripts/calibrate.py --questions otro.yaml
    python scripts/calibrate.py --check             # NO recalibra: valida que calibration.toml
                                                     # vigente siga sin casos peligrosos + corre el
                                                     # lint de contenido. Exit code != 0 si falla --
                                                     # pensado para scripts/verify.py (ver README).
"""
import argparse
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import voice  # noqa: F401,E402  (importa llama_cpp primero, ver voice/__init__.py)
from voice.canonical import CanonicalMatcher  # noqa: E402
from voice.config import load_config  # noqa: E402
from voice.rag import Hit, Retriever  # noqa: E402


def best_per_doc(hits: list[Hit]) -> dict[str, float]:
    out: dict[str, float] = {}
    for h in hits:
        out[h.source] = max(out.get(h.source, -1e9), h.score)
    return out


def _label_ok(label: str, docs_over: set[str]) -> bool:
    """Qué cuenta como "bien resuelto" para cada label a un umbral dado:
      - in_domain/ambiguous:     alcanza con 1+ doc sobre el umbral (hay contexto para responder;
                                  para "ambiguous" NO exigimos acá que crucen 2 docs distintos --
                                  ver más abajo por qué).
      - out_of_domain/no_answer: ningún doc debería cruzar (0 sobre el umbral) -- cualquiera que
                                  cruce es directamente el riesgo de alucinación que motiva min_score.

    Probado y DESCARTADO exigir 2+ docs distintos para "ambiguous" acá (para forzar que el gate de
    ambigüedad, voice/guardrails.py: ambiguous_docs, tenga candidatos): bajaba min_score muchísimo
    (de 0.0 a -1.6) para que un segundo doc débilmente relacionado cruzara -- y a ESE nivel, casi
    cualquier chunk del corpus (chico y genérico, ver docs/) cruza para casi cualquier pregunta, lo
    que rompió preguntas simples de un solo tema: "¿Cuál es el horario de la oficina los sábados?"
    empezaba a mezclarse con recursos_humanos.md y disparaba el gate de ambigüedad sin ninguna
    ambigüedad real (visto corriendo scripts/eval.py en pregunta_compuesta y varios casos de
    validación de descomposición). Cambia un bug (alucinar con 1 doc débil) por otro peor (pedir
    aclaración de más en preguntas normales). min_score se calibra acá SOLO para la decisión
    binaria "¿hay o no hay contexto razonable?"; que ese contexto alcance para 2 documentos
    genuinamente en conflicto (y no solo 1 débil) es un juicio más fino que se calibra aparte, en
    sweep_ambiguity() con el min_score YA elegido -- si con ese min_score no aparecen suficientes
    casos de 2 docs, es una señal honesta de que este corpus/reranker no separa bien ese caso
    todavía (ver README), no algo para forzar bajando min_score."""
    if label in ("in_domain", "ambiguous"):
        return len(docs_over) >= 1
    return len(docs_over) == 0  # out_of_domain, no_answer


def sweep_min_score(results: list[dict]) -> list[dict]:
    """Barre como umbrales candidatos todos los scores de TODOS los documentos candidatos
    observados (no solo el mejor por pregunta -- para el caso 'ambiguous' hace falta saber si el
    SEGUNDO mejor doc, de otra fuente, también cruza, ver _label_ok)."""
    all_scores = sorted({h.score for r in results for h in r["hits"]}, reverse=True)
    if not all_scores:
        return []
    candidates = all_scores + [all_scores[-1] - 1.0]  # extremo: "no aceptar nada"
    rows = []
    for thr in candidates:
        breakdown: dict[str, list[int]] = {}
        ok_total = 0
        for r in results:
            docs_over = {h.source for h in r["hits"] if h.score >= thr}
            good = _label_ok(r["label"], docs_over)
            counts = breakdown.setdefault(r["label"], [0, 0])
            counts[1] += 1
            if good:
                counts[0] += 1
                ok_total += 1
        rows.append({"threshold": thr, "ok": ok_total, "total": len(results), "breakdown": breakdown})
    return rows


def sweep_ambiguity(ambiguous_gaps: list[float], in_domain_gaps: list[float]) -> list[dict]:
    """gaps = diferencia entre el mejor score y el segundo mejor score DE OTRO documento, ya
    filtrados por el min_score elegido (misma cuenta que voice/guardrails.py: ambiguous_docs). Un
    umbral MAYOR agarra más casos ambiguos reales (recall) pero también más falsos positivos sobre
    preguntas in_domain que en realidad tienen una sola respuesta clara."""
    gaps = sorted(set(ambiguous_gaps + in_domain_gaps))
    if not gaps:
        return []
    candidates = gaps + [gaps[-1] + 0.5]
    rows = []
    for thr in candidates:
        tp = sum(1 for g in ambiguous_gaps if g < thr)
        fn = len(ambiguous_gaps) - tp
        fp = sum(1 for g in in_domain_gaps if g < thr)
        tn = len(in_domain_gaps) - fp
        recall = tp / len(ambiguous_gaps) if ambiguous_gaps else float("nan")
        fpr = fp / len(in_domain_gaps) if in_domain_gaps else float("nan")
        rows.append({"threshold": thr, "tp": tp, "fn": fn, "fp": fp, "tn": tn, "recall": recall, "fpr": fpr})
    return rows


def pick_best_min_score(rows: list[dict]) -> float:
    """Prioridad: primero que out_of_domain/no_answer NUNCA se resuelvan mal (0 documentos
    alucinados sobre el umbral en el 100% de esos casos) -- ese es el riesgo de más impacto (una
    respuesta confiada e inventada). Entre los umbrales que cumplen eso, maximiza el total de
    preguntas bien resueltas (incluye in_domain/ambiguous); empate a favor de un umbral MÁS ALTO
    (más estricto -- menos margen para que un match débil se cuele, ver _label_ok)."""

    def is_safe(row: dict) -> bool:
        for lbl in ("out_of_domain", "no_answer"):
            correct, total = row["breakdown"].get(lbl, [0, 0])
            if total and correct != total:
                return False
        return True

    safe_rows = [r for r in rows if is_safe(r)]
    candidates = safe_rows or rows  # si ninguno es 100% seguro, cae al mejor total nomás
    return max(candidates, key=lambda r: (r["ok"], r["threshold"]))["threshold"]


def pick_best_ambiguity(rows: list[dict]) -> float:
    # prioriza FPR=0 sobre in_domain (no pedir aclaración de más en preguntas con respuesta clara),
    # y entre esos, el mayor recall sobre los casos ambiguos reales
    zero_fpr = [r for r in rows if r["fpr"] == 0]
    candidates = zero_fpr or rows
    return max(candidates, key=lambda r: r["recall"])["threshold"]


def recall_at_n(retriever: Retriever, questions: list[dict], n: int) -> list[dict]:
    """Recall@n del primer filtro, DENSO puro vs. HÍBRIDO (denso+BM25 fusionado por RRF), antes de
    que el reranker reordene -- ver Retriever.dense_top_n / hybrid_top_n. Solo sobre preguntas con
    expected_doc/expected_docs (in_domain/ambiguous en tests/calibration_questions.yaml);
    out_of_domain/no_answer no tienen "doc correcto" que buscar. Esto mide si ese primer filtro (el
    que trae los `reranker_candidates` candidatos antes del cross-encoder) ya pierde el chunk
    correcto antes de que el reranker tenga la oportunidad de reordenarlo."""
    rows = []
    for q in questions:
        expected = q.get("expected_docs") or ([q["expected_doc"]] if q.get("expected_doc") else None)
        if not expected:
            continue
        dense_hits = retriever.dense_top_n(q["question"], n)
        hybrid_hits = retriever.hybrid_top_n(q["question"], n)
        dense_got = {h.source for h in dense_hits}
        hybrid_got = {h.source for h in hybrid_hits}
        rows.append({
            "question": q["question"],
            "expected": expected,
            "dense_got": sorted(dense_got),
            "hybrid_got": sorted(hybrid_got),
            "dense_hit": bool(dense_got & set(expected)),
            "hybrid_hit": bool(hybrid_got & set(expected)),
        })
    return rows


def sweep_canonical_threshold(rows: list[dict]) -> list[dict]:
    """rows: [{"label": "match"|"no_match", "expected_entry": str|None, "best_entry": str|None,
    "best_score": float|None}, ...]. Barre umbrales candidatos (todos los scores observados) y
    cuenta, por cada uno, cuántos "match" se resuelven bien (matchea Y es la entrada correcta) vs.
    cuántos casos PELIGROSOS produce: una pregunta 'no_match' que termina matcheando algo (recita
    texto que no correspondía a nada), o una 'match' que matchea la entrada EQUIVOCADA (recita el
    texto de otra entrada, con total confianza) -- ver pick_best_canonical_threshold, que prioriza
    cero casos peligrosos por sobre recall."""
    scores = sorted({r["best_score"] for r in rows if r["best_score"] is not None}, reverse=True)
    if not scores:
        return []
    candidates = scores + [scores[-1] - 1.0]
    out = []
    for thr in candidates:
        tp = fp_wrong_entry = fp_no_match = fn = tn = 0
        for r in rows:
            predicted = r["best_score"] is not None and r["best_score"] >= thr
            if r["label"] == "match":
                if predicted and r["best_entry"] == r["expected_entry"]:
                    tp += 1
                elif predicted:
                    fp_wrong_entry += 1
                else:
                    fn += 1
            else:
                if predicted:
                    fp_no_match += 1
                else:
                    tn += 1
        out.append({
            "threshold": thr, "tp": tp, "fn": fn, "tn": tn,
            "fp_wrong_entry": fp_wrong_entry, "fp_no_match": fp_no_match,
            "dangerous": fp_wrong_entry + fp_no_match,
        })
    return out


def pick_best_canonical_threshold(rows: list[dict]) -> float:
    """Precisión primero (ver README: si una canónica no matchea cae al RAG, sin drama; si matchea
    MAL, recita con confianza un texto equivocado -- ese es el error caro). Entre los umbrales con
    CERO casos peligrosos, el que maximiza cuántos 'match' se resuelven bien; empate a favor de más
    estricto. Si ninguno logra cero peligrosos, el de menos peligrosos (y entre esos, más estricto)."""
    safe = [r for r in rows if r["dangerous"] == 0]
    if safe:
        return max(safe, key=lambda r: (r["tp"], r["threshold"]))["threshold"]
    return min(rows, key=lambda r: (r["dangerous"], -r["threshold"]))["threshold"]


def run_check(cfg, questions_path: Path, canonical_questions_path: Path) -> bool:
    """Modo --check: NO recalibra (no barre umbrales, no escribe calibration.toml) -- valida que
    los umbrales VIGENTES (los que ya aplicó _apply_calibration) sigan sin producir casos
    peligrosos contra los sets de calibración, y corre el lint de contenido (doc_topics,
    clarify_reply, respuestas canónicas). Pensado para correr rápido antes de cada commit (ver
    scripts/verify.py) -- si algo cambió en el contenido y el umbral vigente ya no es seguro, hay
    que volver a correr `scripts/calibrate.py` (sin --check) para recalibrar, no ajustar a mano."""
    from voice import guardrails

    print("Cargando retriever + canonical (embeddings + reranker)…", flush=True)
    retriever = Retriever(cfg.rag)
    canonical = CanonicalMatcher(cfg.canonical) if cfg.canonical.enabled else None

    ok = True

    # ---- lint de contenido (doc_topics + clarify_reply + respuestas canónicas ya se lintean al
    # cargar CanonicalMatcher, arriba) -------------------------------------------------------
    print("\n## Lint de contenido\n")
    lint_warnings = 0
    for doc, topic in cfg.rag.doc_topics.items():
        for w in guardrails.lint_for_voice(topic, label=f"doc_topics[{doc!r}]"):
            print(f"  ⚠ {w}")
            lint_warnings += 1
    if len(cfg.rag.doc_topics) >= 2:
        sample_docs = list(cfg.rag.doc_topics)[:2]
        sample_msg = guardrails.clarify_reply(sample_docs[0], sample_docs[1], cfg.rag.doc_topics)
        for w in guardrails.lint_for_voice(sample_msg, label="clarify_reply (muestra)"):
            print(f"  ⚠ {w}")
            lint_warnings += 1
    print(f"({lint_warnings} warning(s) de lint -- no bloquean, revisar igual)")

    # ---- rag.min_score vigente, sin recalibrar ---------------------------------------------
    # Solo lo PELIGROSO: una pregunta out_of_domain/no_answer con algún doc cruzando el umbral
    # (riesgo real de alucinar). Un in_domain/ambiguous que NO cruza ningún doc es un falso
    # negativo aceptable (abstiene, no inventa) -- no cuenta como peligroso, mismo criterio que
    # canonical.threshold arriba y que sweep_min_score/pick_best_min_score (ver README).
    questions = yaml.safe_load(questions_path.read_text(encoding="utf-8"))
    dangerous_rag = []
    misses_rag = 0
    for q in questions:
        hits = retriever.score_candidates(q["question"])
        docs_over = {h.source for h in hits if h.score >= cfg.rag.min_score}
        if q["label"] in ("out_of_domain", "no_answer"):
            if docs_over:
                dangerous_rag.append((q["question"], q["label"], sorted(docs_over)))
        elif not docs_over:
            misses_rag += 1
    print(f"\n## rag.min_score vigente ({cfg.rag.min_score:.3f}) contra {questions_path.name}\n")
    if dangerous_rag:
        ok = False
        print(f"❌ {len(dangerous_rag)} caso(s) PELIGROSOS con el umbral vigente:")
        for q, label, docs in dangerous_rag:
            print(f"   - {q!r} ({label}): cruza min_score con {docs} (no debería)")
    else:
        print(f"✅ 0 casos peligrosos ({misses_rag} falso(s) negativo(s) -- aceptable, abstiene, ver README)")

    # ---- canonical.threshold vigente, sin recalibrar ---------------------------------------
    if cfg.canonical.enabled and canonical_questions_path.exists() and canonical is not None:
        canon_questions = yaml.safe_load(canonical_questions_path.read_text(encoding="utf-8")) or []
        dangerous_canon = []  # solo lo PELIGROSO (ver sweep_canonical_threshold): matchear la
        # entrada equivocada, o matchear algo en una pregunta 'no_match'. Un 'match' que NO
        # matcheó nada (falso negativo) es aceptable por diseño (precisión > recall, ver README)
        # -- NO cuenta como caso peligroso acá, sería un chequeo más estricto que lo que se calibró.
        misses = 0
        for q in canon_questions:
            best = canonical.best_score(q["question"])
            predicted = best is not None and best[1] >= cfg.canonical.threshold
            if q["label"] == "match":
                if predicted and best[0] != q.get("entry_id"):
                    dangerous_canon.append((q["question"], "match", f"matcheó '{best[0]}' en vez de '{q.get('entry_id')}'"))
                elif not predicted:
                    misses += 1
            else:
                if predicted:
                    dangerous_canon.append((q["question"], "no_match", f"matcheó '{best[0]}' (no debería matchear nada)"))
        print(f"\n## canonical.threshold vigente ({cfg.canonical.threshold:.3f}) contra {canonical_questions_path.name}\n")
        if dangerous_canon:
            ok = False
            print(f"❌ {len(dangerous_canon)} caso(s) PELIGROSOS con el umbral vigente:")
            for q, label, why in dangerous_canon:
                print(f"   - {q!r} ({label}): {why}")
        else:
            print(f"✅ 0 casos peligrosos ({misses} falso(s) negativo(s) -- aceptable, cae al RAG, ver README)")

    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="tests/calibration_questions.yaml")
    ap.add_argument("--canonical-questions", default="tests/calibration_canonical.yaml")
    ap.add_argument("--out", default="calibration.toml")
    ap.add_argument("--recall-n", type=int, default=5, help="tamaño del top-n denso para el reporte de recall@n")
    ap.add_argument("--dry-run", action="store_true", help="solo mostrar el reporte, no escribir el archivo")
    ap.add_argument("--check", action="store_true", help="no recalibra -- valida los umbrales vigentes + lint, exit code != 0 si falla")
    args = ap.parse_args()

    cfg = load_config()

    if args.check:
        ok = run_check(cfg, ROOT / args.questions, ROOT / args.canonical_questions)
        print(f"\n{'✅ calibrate.py --check: OK' if ok else '❌ calibrate.py --check: FALLÓ'}")
        sys.exit(0 if ok else 1)

    print("Cargando retriever (embeddings + reranker)…", flush=True)
    retriever = Retriever(cfg.rag)

    # fallback para cuando no hay suficientes datos para recalibrar algo: el default de
    # config.toml, NO cfg.rag (que ya trae calibration.toml de una corrida anterior superpuesto --
    # usar cfg.rag ahí perpetuaría un valor viejo/stale en vez de caer al default documentado).
    _raw_config = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    _raw_defaults = _raw_config.get("rag", {})
    fallback_min_score = _raw_defaults.get("min_score", cfg.rag.min_score)
    fallback_ambiguity = _raw_defaults.get("ambiguity_threshold", cfg.rag.ambiguity_threshold)
    fallback_canonical_threshold = _raw_config.get("canonical", {}).get("threshold", cfg.canonical.threshold)

    questions = yaml.safe_load((ROOT / args.questions).read_text(encoding="utf-8"))

    results = []
    for q in questions:
        hits = retriever.score_candidates(q["question"])
        best = max((h.score for h in hits), default=None)
        results.append({"question": q["question"], "label": q["label"], "best_score": best, "hits": hits})

    print(f"\n# Calibración — {args.questions} ({len(questions)} preguntas)\n")
    print(f"{'pregunta':<55} {'label':<14} {'mejor':>7}  top-3 docs (score)")
    for r in results:
        top = sorted(r["hits"], key=lambda h: -h.score)[:3]
        docs = ", ".join(f"{h.source}={h.score:.2f}" for h in top)
        best_s = f"{r['best_score']:.2f}" if r["best_score"] is not None else "—"
        print(f"{r['question'][:53]:<55} {r['label']:<14} {best_s:>7}  {docs}")

    # ---- recall@n: denso puro vs. híbrido (denso+BM25 por RRF), antes del reranker ---------
    recall_rows = recall_at_n(retriever, questions, args.recall_n)
    print(f"\n## Recall@{args.recall_n}: denso vs. híbrido (denso+BM25 con RRF), ANTES del reranker\n")
    print(
        f"(mide si el primer filtro, el que trae reranker_candidates={cfg.rag.reranker_candidates} "
        f"candidatos para el cross-encoder, ya pierde el chunk correcto antes de que el reranker "
        f"pueda reordenarlo -- hybrid_enabled={cfg.rag.hybrid_enabled}, "
        f"bm25_candidates={cfg.rag.bm25_candidates}, rrf_k={cfg.rag.rrf_k})\n"
    )
    if recall_rows:
        dense_hits = sum(1 for r in recall_rows if r["dense_hit"])
        hybrid_hits = sum(1 for r in recall_rows if r["hybrid_hit"])
        for r in recall_rows:
            dmark = "✓" if r["dense_hit"] else "✗"
            hmark = "✓" if r["hybrid_hit"] else "✗"
            print(f"  denso={dmark} híbrido={hmark}  {r['question'][:55]:<57} esperado={r['expected']}")
            if not r["dense_hit"] or not r["hybrid_hit"]:
                print(f"      denso top{args.recall_n}={r['dense_got']}  híbrido top{args.recall_n}={r['hybrid_got']}")
        print(
            f"\nRecall@{args.recall_n} denso:   {dense_hits}/{len(recall_rows)} ({dense_hits / len(recall_rows):.0%})"
        )
        print(
            f"Recall@{args.recall_n} híbrido: {hybrid_hits}/{len(recall_rows)} ({hybrid_hits / len(recall_rows):.0%})"
        )
        if hybrid_hits < dense_hits:
            print(
                "-> el híbrido pierde recall respecto del denso solo en este set: revisar bm25_candidates "
                "o si el corpus/preguntas tienen vocabulario que BM25 penaliza de más."
            )
        elif dense_hits < len(recall_rows) or hybrid_hits < len(recall_rows):
            print(
                f"-> hay preguntas donde el chunk correcto NO entra al top-{args.recall_n}: subir "
                f"reranker_candidates/bm25_candidates, o revisar el chunking/embedding para esas preguntas."
            )
    else:
        print("(ninguna pregunta del set tiene expected_doc/expected_docs)")

    # ---- rag.min_score --------------------------------------------------------------------
    min_score_rows = sweep_min_score(results)
    chosen_min_score = pick_best_min_score(min_score_rows) if min_score_rows else fallback_min_score
    labels = ["in_domain", "ambiguous", "out_of_domain", "no_answer"]
    print("\n## Barrido de rag.min_score (gate de abstención)\n")
    print("(por label: bien_resueltos/total a ese umbral -- ver _label_ok: in_domain/ambiguous "
          "necesitan 1+ doc sobre el umbral, out_of_domain/no_answer necesitan 0)")
    header = "  ".join(f"{lbl:>15}" for lbl in labels)
    print(f"{'umbral':>8}  {'OK total':>8}  {header}")
    for r in min_score_rows:
        cells = "  ".join(f"{r['breakdown'].get(lbl, [0,0])[0]:>6}/{r['breakdown'].get(lbl, [0,0])[1]:<7}" for lbl in labels)
        print(f"{r['threshold']:>8.2f}  {r['ok']:>3}/{r['total']:<4}  {cells}")
    print(
        f"\n-> elegido: min_score = {chosen_min_score:.3f} (máximo de preguntas bien resueltas ENTRE "
        f"los umbrales donde out_of_domain/no_answer no alucinan ni una vez; empate a favor de más "
        f"estricto)"
    )

    # ---- rag.ambiguity_threshold (usa el min_score YA elegido, como en producción) --------
    ambiguous_gaps, in_domain_gaps = [], []
    for r in results:
        hits_over = [h for h in r["hits"] if h.score >= chosen_min_score]
        ranked = sorted(best_per_doc(hits_over).items(), key=lambda kv: -kv[1])
        if len(ranked) < 2:
            continue
        gap = ranked[0][1] - ranked[1][1]
        if r["label"] == "ambiguous":
            ambiguous_gaps.append(gap)
        elif r["label"] == "in_domain":
            in_domain_gaps.append(gap)

    amb_rows = sweep_ambiguity(ambiguous_gaps, in_domain_gaps)
    print(f"\n## Barrido de rag.ambiguity_threshold (gate de ambigüedad, con min_score={chosen_min_score:.3f})\n")
    if not amb_rows:
        print(
            "(sin suficientes preguntas ambiguous/in_domain con 2+ documentos sobre min_score para "
            "calibrar esto -- se deja el valor actual sin cambios)"
        )
        chosen_ambiguity = fallback_ambiguity
    else:
        print(f"{'umbral':>8} {'TP amb':>7} {'FN amb':>7} {'recall amb':>11} {'FP in_dom':>10} {'FPR':>6}")
        for r in amb_rows:
            print(
                f"{r['threshold']:>8.2f} {r['tp']:>7} {r['fn']:>7} {r['recall']:>11.2f} "
                f"{r['fp']:>10} {r['fpr']:>6.2f}"
            )
        chosen_ambiguity = pick_best_ambiguity(amb_rows)
        print(
            f"\n-> elegido: ambiguity_threshold = {chosen_ambiguity:.3f} (recall máximo entre los "
            f"umbrales con FPR=0 sobre in_domain, si hay alguno; si no, el de mayor recall a secas)"
        )

    # ---- canonical.threshold ---------------------------------------------------------------
    chosen_canonical_threshold = fallback_canonical_threshold
    canonical_path = ROOT / args.canonical_questions
    if cfg.canonical.enabled and canonical_path.exists():
        print(f"\n## Calibración de canonical.threshold — {args.canonical_questions}\n")
        print("Cargando CanonicalMatcher (reranker sobre tests/canonical_answers.yaml)…", flush=True)
        matcher = CanonicalMatcher(cfg.canonical)
        canonical_questions = yaml.safe_load(canonical_path.read_text(encoding="utf-8")) or []

        canon_rows = []
        for q in canonical_questions:
            best = matcher.best_score(q["question"])  # no muta rotación, ver voice/canonical.py
            canon_rows.append({
                "question": q["question"],
                "label": q["label"],
                "expected_entry": q.get("entry_id"),
                "best_entry": best[0] if best else None,
                "best_score": best[1] if best else None,
            })

        print(f"{'pregunta':<50} {'label':<10} {'esperado':<22} {'mejor match':<22} {'score':>7}")
        for r in canon_rows:
            best_s = f"{r['best_score']:.2f}" if r["best_score"] is not None else "—"
            print(
                f"{r['question'][:48]:<50} {r['label']:<10} {str(r['expected_entry'] or '—'):<22} "
                f"{str(r['best_entry'] or '—'):<22} {best_s:>7}"
            )

        canon_sweep_rows = sweep_canonical_threshold(canon_rows)
        if not canon_sweep_rows:
            print("(sin scores para barrer -- ¿tests/canonical_answers.yaml está vacío?)")
        else:
            print(f"\n{'umbral':>8} {'TP':>4} {'FN':>4} {'TN':>4} {'FP entrada mala':>16} {'FP no_match':>12} {'peligrosos':>10}")
            for r in canon_sweep_rows:
                print(
                    f"{r['threshold']:>8.2f} {r['tp']:>4} {r['fn']:>4} {r['tn']:>4} "
                    f"{r['fp_wrong_entry']:>16} {r['fp_no_match']:>12} {r['dangerous']:>10}"
                )
            chosen_canonical_threshold = pick_best_canonical_threshold(canon_sweep_rows)
            print(
                f"\n-> elegido: canonical.threshold = {chosen_canonical_threshold:.3f} (cero casos "
                f"peligrosos -- ni matchear la entrada equivocada ni matchear algo que no debía "
                f"matchear nada -- y entre esos, el que resuelve más 'match' bien; empate a favor "
                f"de más estricto)"
            )
    elif cfg.canonical.enabled:
        print(f"\n(canonical.enabled pero no existe {args.canonical_questions} -- no se calibra canonical.threshold)")

    if args.dry_run:
        print("\n(--dry-run: no se escribió calibration.toml)")
        return

    # margen de seguridad: chosen_min_score/chosen_ambiguity vienen de un score REAL observado (el
    # umbral se eligió para incluir/excluir exactamente ese candidato). Truncar a pocos decimales al
    # escribir el archivo puede mover el umbral hacia el lado "menos permisivo" por un margen
    # mínimo -- alcanza para dar vuelta la comparación >= / < justo en ese caso límite (encontrado
    # en la práctica: un umbral escrito con .4f excluyó por error el mismo candidato que lo generó,
    # ver README). Restar/sumar un epsilon chico deja el resultado del lado seguro.
    EPS = 1e-4
    safe_min_score = chosen_min_score - EPS
    safe_ambiguity = chosen_ambiguity + EPS
    # mismo sentido de comparación que min_score (score >= umbral) -- restar EPS lo deja del lado
    # inclusivo del candidato que lo generó.
    safe_canonical_threshold = chosen_canonical_threshold - EPS

    out_path = ROOT / args.out
    out_path.write_text(
        "# Generado por scripts/calibrate.py -- NO editar a mano, volver a correr el script.\n"
        f"# Calibrado contra {args.questions} ({len(questions)} preguntas etiquetadas) y "
        f"{args.canonical_questions}.\n"
        "# Si cambian los documentos de docs/, el contenido de canonical_answers.yaml, o los\n"
        "# datasets de calibración:\n"
        "#   python scripts/calibrate.py\n"
        "# Ver README: parámetros calibrados (acá) vs. arquitectura estable (config.toml/código).\n"
        "\n[rag]\n"
        f"min_score = {safe_min_score:.6f}\n"
        f"ambiguity_threshold = {safe_ambiguity:.6f}\n"
        "\n[canonical]\n"
        f"threshold = {safe_canonical_threshold:.6f}\n",
        encoding="utf-8",
    )
    print(
        f"\nEscrito {out_path} (min_score={safe_min_score:.6f}, ambiguity_threshold={safe_ambiguity:.6f}, "
        f"canonical.threshold={safe_canonical_threshold:.6f}, con margen de {EPS} contra pérdida de precisión)"
    )


if __name__ == "__main__":
    main()
