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
"""
import argparse
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import voice  # noqa: F401,E402  (importa llama_cpp primero, ver voice/__init__.py)
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
    """Recall@n de la etapa DENSA (coseno), antes de que el reranker reordene -- ver
    Retriever.dense_top_n. Solo sobre preguntas con expected_doc/expected_docs (in_domain/ambiguous
    en tests/calibration_questions.yaml); out_of_domain/no_answer no tienen "doc correcto" que
    buscar. Importante: NO hay retrieval híbrido (denso + BM25/léxico) en este proyecto, solo denso
    -- esto mide si ESE primer filtro (el que trae los `reranker_candidates` candidatos antes del
    cross-encoder) ya pierde el chunk correcto antes de que el reranker tenga la oportunidad de
    reordenarlo, no una comparación contra un retrieval léxico que no existe acá."""
    rows = []
    for q in questions:
        expected = q.get("expected_docs") or ([q["expected_doc"]] if q.get("expected_doc") else None)
        if not expected:
            continue
        hits = retriever.dense_top_n(q["question"], n)
        got = {h.source for h in hits}
        hit_ok = bool(got & set(expected))
        rows.append({"question": q["question"], "expected": expected, "got": sorted(got), "hit": hit_ok})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="tests/calibration_questions.yaml")
    ap.add_argument("--out", default="calibration.toml")
    ap.add_argument("--recall-n", type=int, default=5, help="tamaño del top-n denso para el reporte de recall@n")
    ap.add_argument("--dry-run", action="store_true", help="solo mostrar el reporte, no escribir el archivo")
    args = ap.parse_args()

    cfg = load_config()
    print("Cargando retriever (embeddings + reranker)…", flush=True)
    retriever = Retriever(cfg.rag)

    # fallback para cuando no hay suficientes datos para recalibrar algo: el default de
    # config.toml, NO cfg.rag (que ya trae calibration.toml de una corrida anterior superpuesto --
    # usar cfg.rag ahí perpetuaría un valor viejo/stale en vez de caer al default documentado).
    _raw_defaults = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8")).get("rag", {})
    fallback_min_score = _raw_defaults.get("min_score", cfg.rag.min_score)
    fallback_ambiguity = _raw_defaults.get("ambiguity_threshold", cfg.rag.ambiguity_threshold)

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

    # ---- recall@n de la etapa densa (antes del reranker) -----------------------------------
    recall_rows = recall_at_n(retriever, questions, args.recall_n)
    print(f"\n## Recall@{args.recall_n} de la etapa densa (coseno, ANTES del reranker)\n")
    print(
        f"(no hay retrieval híbrido en este proyecto -- solo denso; esto mide si el primer filtro, "
        f"el que trae reranker_candidates={cfg.rag.reranker_candidates} candidatos para el "
        f"cross-encoder, ya pierde el chunk correcto antes de que el reranker pueda reordenarlo)\n"
    )
    if recall_rows:
        hits = sum(1 for r in recall_rows if r["hit"])
        for r in recall_rows:
            mark = "✓" if r["hit"] else "✗ PERDIDO"
            print(f"  {mark}  {r['question'][:60]:<62} esperado={r['expected']} top{args.recall_n}={r['got']}")
        print(f"\nRecall@{args.recall_n}: {hits}/{len(recall_rows)} ({hits / len(recall_rows):.0%})")
        if hits < len(recall_rows):
            print(
                f"-> hay preguntas donde el chunk correcto NO entra al top-{args.recall_n} denso: "
                f"subir reranker_candidates (actualmente {cfg.rag.reranker_candidates}) para darle "
                f"más candidatos al reranker, o revisar el chunking/embedding para esas preguntas."
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

    out_path = ROOT / args.out
    out_path.write_text(
        "# Generado por scripts/calibrate.py -- NO editar a mano, volver a correr el script.\n"
        f"# Calibrado contra {args.questions} ({len(questions)} preguntas etiquetadas).\n"
        "# Si cambian los documentos de docs/ o el dataset de calibración:\n"
        "#   python scripts/calibrate.py\n"
        "# Ver README: parámetros calibrados (acá) vs. arquitectura estable (config.toml/código).\n"
        "\n[rag]\n"
        f"min_score = {safe_min_score:.6f}\n"
        f"ambiguity_threshold = {safe_ambiguity:.6f}\n",
        encoding="utf-8",
    )
    print(f"\nEscrito {out_path} (min_score={safe_min_score:.6f}, ambiguity_threshold={safe_ambiguity:.6f}, con margen de {EPS} contra pérdida de precisión)")


if __name__ == "__main__":
    main()
