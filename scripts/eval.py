"""Modo "test": corre los casos de tests/eval_questions.yaml contra el LLM/RAG real (sin STT/mic) y
vuelca todo a un archivo Markdown. Herramienta de desarrollo, no de producción: no calcula pass/fail
solo -- el juicio de si cada caso está bien lo hace un humano (o Claude) leyendo el resultado.

Dos tipos de caso:
  - pipeline (default): corre la conversación completa (reescritura -> RAG -> LLM -> respuesta).
  - rewrite: prueba SOLO voice/rewrite.py en aislado (historial + pregunta -> pregunta autónoma),
    sin retrieval ni LLM de respuesta. Sirve para saber si un fallo es de la reescritura o de lo
    que pasa después, sin tener que adivinar.

    python scripts/eval.py                        # corre todos los casos de tests/eval_questions.yaml
    python scripts/eval.py --cases tests/otro.yaml
    python scripts/eval.py --out eval_results/mi_corrida.md
    python scripts/eval.py --only cambio_de_dia_en_followup
    python scripts/eval.py --show-chunk-text       # texto completo de cada chunk recuperado, no
                                                    # solo source@score
"""
import argparse
import datetime
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import voice  # noqa: F401,E402  (importa llama_cpp primero)
from voice import rewrite  # noqa: E402
from voice.config import load_config  # noqa: E402
from voice.pipeline import Assistant  # noqa: E402

rewrite_latencies_ms: list[float] = []  # se llena en run_case(), se reporta al final
subq_check_failures: list[str] = []  # casos con expect_n_subquestions que no dieron ese número
history_check_failures: list[str] = []  # turnos donde self.llm.history no creció (record_turn no se llamó)
canonical_check_failures: list[str] = []  # casos con expect_canonical_*/expect_exact_text/etc que no dieron lo esperado
gate_check_failures: list[str] = []  # casos con expect_clarify/expect_context_restricted_to que no dieron lo esperado


def case_variants(case: dict) -> list[tuple[str | None, list[str]]]:
    """Con temperature=0 (determinístico) repetir la MISMA pregunta no aporta nada -- por eso
    'variants' reemplazó a un viejo 'repeat': varias formulaciones distintas del mismo caso, para
    medir robustez a cómo se pregunta en vez de robustez al muestreo aleatorio."""
    if "variants" in case:
        n = len(case["variants"])
        return [(f"variante {i}/{n}", v) for i, v in enumerate(case["variants"], 1)]
    return [(None, case["turns"])]


def run_rewrite_case(bot: Assistant, case: dict) -> list[str]:
    """Caso 'type: rewrite': prueba voice/rewrite.py aislado. `history` es una lista de
    [pregunta_usuario, respuesta_asistente]; `question` es el turno a reescribir."""
    bot.reset_conversation()
    for u, a in case["history"]:
        bot.llm.record_turn(u, a)

    import time
    t0 = time.monotonic()
    subqs, was_rewritten = bot.llm.rewrite_query(case["question"])
    dt_ms = (time.monotonic() - t0) * 1000
    rewrite_latencies_ms.append(dt_ms)

    lines = [f"## {case['id']} (rewrite)"]
    if case.get("note"):
        lines.append(f"> **nota:** {case['note'].strip()}")
    if case.get("expect"):
        lines.append(f"> **se espera:** {case['expect'].strip()}")
    lines.append("")
    for u, a in case["history"]:
        lines.append(f"- 🗣 {u}")
        lines.append(f"  - 🤖 {a}")
    lines.append(f"- 🗣 **{case['question']}**")
    expect_n = case.get("expect_n_subquestions")
    if expect_n is not None:
        # chequeo AUTOMÁTICO (no juicio humano) -- pensado para casos que fijan un invariante
        # estructural puntual, como "el modo descomposición no debe truncarse" (ver
        # voice/llm.py: _decompose() NO debe recibir stop=["?", ...], si no el JSON de varias
        # sub-preguntas se corta en el primer "?" y esto cae a 1 sub-pregunta o menos).
        ok = len(subqs) == expect_n
        mark = "✓" if ok else "✗ FALLÓ"
        lines.append(f"  - {mark}: se esperaban {expect_n} sub-preguntas, se obtuvieron {len(subqs)}")
        if not ok:
            subq_check_failures.append(f"{case['id']}: esperaba {expect_n}, obtuvo {len(subqs)} -> {subqs!r}")
    if len(subqs) > 1:
        sub_txt = " | ".join(subqs)
        lines.append(f"  - ➜ descompuesta en {len(subqs)}: **{sub_txt}** (se_reescribió={was_rewritten}, {dt_ms:.0f}ms)")
    else:
        lines.append(f"  - ➜ reescrita: **{subqs[0]}** (se_reescribió={was_rewritten}, {dt_ms:.0f}ms)")
    lines.append("")
    return lines


def run_case(bot: Assistant, case: dict, label: str | None, turns: list[str], show_chunk_text: bool) -> list[str]:
    bot.reset_conversation()
    suffix = f" ({label})" if label else ""
    lines = [f"## {case['id']}{suffix}"]
    if label is None or label.startswith("variante 1/"):
        if case.get("note"):
            lines.append(f"> **nota:** {case['note'].strip()}")
        if case.get("expect"):
            lines.append(f"> **se espera:** {case['expect'].strip()}")
    lines.append("")
    turn_results: list = []
    for q in turns:
        # chequeo AUTOMÁTICO (no juicio humano): la pregunta del usuario tiene que guardarse en el
        # historial SIEMPRE, incluso en turnos que terminan en abstención/nota fija -- ver
        # voice/pipeline.py: Assistant._fixed_reply/answer, ambos llaman a record_turn(). Regresión
        # real encontrada en vivo (23/09, ver README): un turno se procesó "sin historial" sin que
        # nada lo hubiera reiniciado a propósito -- terminó siendo un timeout de followup_s
        # silencioso (arreglado aparte, ver Assistant.run), pero este chequeo queda como red de
        # seguridad genuina contra que record_turn deje de llamarse en algún camino nuevo.
        hist_before = len(bot.llm.history)
        turn = bot.answer(q)
        hist_after = len(bot.llm.history)
        if hist_after != hist_before + 2:
            history_check_failures.append(
                f"{case['id']}: tras {q!r}, self.llm.history pasó de {hist_before} a {hist_after} "
                f"elementos (se esperaban +2 -- record_turn no se llamó o llamó de más)"
            )
        if turn.t_rewrite:
            rewrite_latencies_ms.append(turn.t_rewrite * 1000)
        lines.append(f"- 🗣 **{q}**")
        if turn.was_rewritten and len(turn.subquestions) > 1:
            lines.append(f"  - ➜ descompuesta en {len(turn.subquestions)}: {turn.subquestions!r}")
        elif turn.was_rewritten:
            lines.append(f"  - ➜ reescrita: {turn.retrieval_query!r}")
        lines.append(f"  - 🤖 {turn.answer}")
        if not turn.context_hits:
            lines.append("  - RAG: *(sin contexto)*")
        elif show_chunk_text:
            for h in turn.context_hits:
                lines.append(f"  - RAG: {h.source}@{h.score:.2f}: {h.text!r}")
        else:
            hits = ", ".join(f"{h.source}@{h.score:.2f}" for h in turn.context_hits)
            lines.append(f"  - RAG: {hits}")
        # distingue "no se recuperó nada" de "se recuperó con score bajo" (ver voice/pipeline.py:
        # Turn.retrieval_trace) -- top1/top2 siempre (no solo sin hits): top2 es lo que compite con
        # el top-1 para el gate de ambigüedad/confianza (ver gate_docs), aunque no haya cruzado
        # min_score.
        for tr in turn.retrieval_trace:
            top1 = f"{tr['top1_source']}@{tr['top1_score']:.2f}" if tr["top1_source"] else "ningún candidato"
            top2 = f"{tr['top2_source']}@{tr['top2_score']:.2f}" if tr.get("top2_source") else "—"
            if not tr["had_hits"]:
                lines.append(f"  - 🔎 {tr['subq']!r}: top1={top1} top2={top2} (min_score={tr['min_score']:.2f})")
            elif show_chunk_text:
                lines.append(f"  - 🔎 {tr['subq']!r}: top1={top1} top2={top2}")
        for cm in turn.canonical_matches:
            lines.append(f"  - 📌 canonical:{cm['entry_id']}@{cm['score']:.2f}")
        turn_results.append(turn)
    lines += _check_canonical_expectations(case, turn_results)
    lines += _check_gate_expectations(case, turn_results)
    lines.append("")
    return lines


def _check_gate_expectations(case: dict, turn_results: list) -> list[str]:
    """Chequeos AUTOMÁTICOS sobre el gate de ambigüedad/confianza (voice/guardrails.py:
    gate_docs(), rediseñado 24/09 -- ver README). Campos opcionales del caso, aplicados al ÚLTIMO
    turno:
      expect_clarify: true/false -- si el último turno debería (o no) pedir aclaración cross-doc
        ("Tu pregunta toca dos temas distintos..." en la respuesta).
      expect_context_restricted_to: <doc> -- el CONTEXTO del último turno debe venir SOLO de ese
        documento (alta confianza en gate_docs restringiendo el CONTEXTO a un solo doc)."""
    out: list[str] = []
    cid = case["id"]
    last = turn_results[-1]

    expect_clarify = case.get("expect_clarify")
    if expect_clarify is not None:
        got_clarify = "toca dos temas distintos" in last.answer
        ok = got_clarify == expect_clarify
        out.append(f"  - {'✓' if ok else '✗ FALLÓ'}: expect_clarify={expect_clarify} (obtuvo {got_clarify})")
        if not ok:
            gate_check_failures.append(
                f"{cid}: expect_clarify={expect_clarify}, obtuvo clarify={got_clarify} (answer={last.answer!r})"
            )

    expect_doc = case.get("expect_context_restricted_to")
    if expect_doc is not None:
        docs = {h.source for h in last.context_hits}
        ok = docs == {expect_doc}
        out.append(f"  - {'✓' if ok else '✗ FALLÓ'}: expect_context_restricted_to={expect_doc!r} (obtuvo docs={sorted(docs)!r})")
        if not ok:
            gate_check_failures.append(
                f"{cid}: expect_context_restricted_to={expect_doc!r}, obtuvo docs={sorted(docs)!r}"
            )

    return out


def _check_canonical_expectations(case: dict, turn_results: list) -> list[str]:
    """Chequeos AUTOMÁTICOS (no juicio humano) sobre la capa de respuestas canónicas
    (voice/canonical.py) -- ver README. Campos opcionales del caso:
      expect_canonical_entries: list[str|None], un elemento por turno -- el entry_id que debería
        haber matcheado ESE turno (None = no debería matchear ninguna entrada canónica).
      expect_exact_text: el último turno debe responder con ESTE texto, carácter por carácter.
      expect_variant_rotation: N -- los primeros N turnos deben dar N textos DISTINTOS entre sí, y
        el turno N+1 (si existe) debe repetir el texto del primer turno (rotación completa).
      expect_repeat_of_previous: true -- el último turno debe repetir EXACTO el texto del anterior
        (was_repeat=True), sin rotar variantes.
    Cualquier fallo se junta en canonical_check_failures y termina el proceso con exit code != 0
    (ver main())."""
    out: list[str] = []
    cid = case["id"]

    expect_entries = case.get("expect_canonical_entries")
    if expect_entries is not None:
        for i, (expected, turn) in enumerate(zip(expect_entries, turn_results)):
            got_ids = [m["entry_id"] for m in turn.canonical_matches]
            # expected=None: no debería matchear NADA. expected=str: alcanza con que esa entrada
            # esté ENTRE los matches del turno (no necesariamente la única -- un turno compuesto
            # mixto puede matchear una sub-pregunta como canonical y resolver la otra por RAG).
            ok = (not got_ids) if expected is None else (expected in got_ids)
            got_repr = got_ids if got_ids else None
            out.append(f"  - {'✓' if ok else '✗ FALLÓ'}: turno {i + 1} esperaba canonical={expected!r}, obtuvo {got_repr!r}")
            if not ok:
                canonical_check_failures.append(f"{cid}: turno {i + 1} esperaba canonical={expected!r}, obtuvo {got_repr!r}")

    expect_text = case.get("expect_exact_text")
    if expect_text is not None:
        got = turn_results[-1].answer
        ok = got == expect_text
        out.append(f"  - {'✓' if ok else '✗ FALLÓ'}: expect_exact_text ({'coincide' if ok else f'obtuvo {got!r}'})")
        if not ok:
            canonical_check_failures.append(f"{cid}: expect_exact_text no coincide -- esperaba {expect_text!r}, obtuvo {got!r}")

    n = case.get("expect_variant_rotation")
    if n is not None:
        texts = [t.answer for t in turn_results[:n]]
        distinct_ok = len(set(texts)) == n
        out.append(f"  - {'✓' if distinct_ok else '✗ FALLÓ'}: primeros {n} turnos dan {len(set(texts))} textos distintos (se esperaban {n})")
        if not distinct_ok:
            canonical_check_failures.append(f"{cid}: rotación -- primeros {n} turnos no dan {n} textos distintos: {texts!r}")
        if len(turn_results) > n:
            wrap_ok = turn_results[n].answer == turn_results[0].answer
            out.append(f"  - {'✓' if wrap_ok else '✗ FALLÓ'}: turno {n + 1} repite la variante del turno 1 (rotación completa)")
            if not wrap_ok:
                canonical_check_failures.append(f"{cid}: rotación -- turno {n + 1} ({turn_results[n].answer!r}) no repite el turno 1 ({turn_results[0].answer!r})")

    if case.get("expect_repeat_of_previous"):
        last, prev = turn_results[-1], turn_results[-2]
        ok = last.answer == prev.answer and last.was_repeat
        out.append(f"  - {'✓' if ok else '✗ FALLÓ'}: último turno repite el anterior tal cual (was_repeat={last.was_repeat})")
        if not ok:
            canonical_check_failures.append(
                f"{cid}: expect_repeat_of_previous falló -- anterior={prev.answer!r} último={last.answer!r} was_repeat={last.was_repeat}"
            )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="tests/eval_questions.yaml")
    ap.add_argument("--out", default=None, help="default: eval_results/<timestamp>.md")
    ap.add_argument("--only", default=None, help="correr solo el caso con este id")
    ap.add_argument("--split", choices=["dev", "validation"], default=None, help="correr solo casos de ese split")
    ap.add_argument("--show-chunk-text", action="store_true", help="ver el texto completo de cada chunk recuperado")
    args = ap.parse_args()

    cases_path = ROOT / args.cases
    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
    if args.split:
        cases = [c for c in cases if c.get("split", "dev") == args.split]

    cfg = load_config()
    cfg.wakeword.enabled = False

    # lint de voz sobre los nombres de tema del gate de ambigüedad y el mensaje de aclaración que
    # arman (ver voice/guardrails.py: lint_for_voice/clarify_reply) -- se corre una sola vez acá,
    # no por caso, mismo espíritu que el lint de tests/canonical_answers.yaml al cargar.
    from voice import guardrails
    for doc, topic in cfg.rag.doc_topics.items():
        for w in guardrails.lint_for_voice(topic, label=f"doc_topics[{doc!r}]"):
            print(f"[lint] ⚠ {w}", flush=True)
    if len(cfg.rag.doc_topics) >= 2:
        sample_docs = list(cfg.rag.doc_topics)[:2]
        sample_msg = guardrails.clarify_reply(sample_docs[0], sample_docs[1], cfg.rag.doc_topics)
        for w in guardrails.lint_for_voice(sample_msg, label="clarify_reply (muestra)"):
            print(f"[lint] ⚠ {w}", flush=True)

    bot = Assistant(cfg, speak=False)

    out_path = Path(args.out) if args.out else ROOT / "eval_results" / f"{datetime.datetime.now():%Y%m%d_%H%M%S}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [f"# Eval run — {datetime.datetime.now():%Y-%m-%d %H:%M} — {cases_path.name}", ""]
    body: list[str] = []
    # separados en secciones: 'dev' se usó para ajustar el prompt/few-shot de voice/rewrite.py (no
    # mide generalización); 'validation' es holdout, formulaciones no vistas durante el ajuste.
    for split in ("dev", "validation"):
        split_cases = [c for c in cases if c.get("split", "dev") == split]
        if not split_cases:
            continue
        title = "DEV (usados para ajustar el prompt)" if split == "dev" else "VALIDACIÓN (holdout, no se ajustó nada contra estos)"
        body.append(f"# {title} — {len(split_cases)} casos")
        body.append("")
        for case in split_cases:
            if case.get("type") == "rewrite":
                body += run_rewrite_case(bot, case)
            else:
                for label, turns in case_variants(case):
                    body += run_case(bot, case, label, turns, args.show_chunk_text)

    # tasa de falsos positivos del pre-filtro barato (voice/rewrite.py: looks_compound) sobre
    # preguntas que NO son compuestas -- casos marcados "compound: true" se excluyen (ahí SÍ debe
    # disparar, es lo esperado). Un falso positivo acá solo cuesta latencia (ver looks_compound),
    # pero igual conviene medirlo para saber si sale caro en la práctica.
    total_q = fired_q = 0
    for case in cases:
        if case.get("compound"):
            continue
        if case.get("type") == "rewrite":
            texts = [case["question"]]
        elif "variants" in case:
            texts = [q for v in case["variants"] for q in v]
        else:
            texts = list(case["turns"])
        for text in texts:
            total_q += 1
            if rewrite.looks_compound(text):
                fired_q += 1
    if total_q:
        body.append(
            f"**Falsos positivos del pre-filtro de preguntas compuestas** (sobre preguntas NO "
            f"marcadas `compound: true`): {fired_q}/{total_q} ({fired_q / total_q:.0%}) dispararon "
            f"el pre-filtro sin ser compuestas -- cada una paga solo la latencia extra de una "
            f"llamada de reescritura en modo descomposición, que debería devolver 1 sola sub-pregunta."
        )
        body.append("")

    if rewrite_latencies_ms:
        def pct(xs, p):
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(len(xs) * p))]

        all_ms = rewrite_latencies_ms
        # el camino "sin historial" (rewrite_query hace early-return) tarda microsegundos, no 0ms
        # exacto -- 5ms es un umbral seguro por debajo de lo que tarda cualquier llamada real al LLM
        fired_ms = [m for m in all_ms if m > 5]
        body.append(
            f"**Latencia de reescritura** (todos los turnos, incluye primer-turno en 0ms): "
            f"n={len(all_ms)}, p50={statistics.median(all_ms):.0f}ms, p95={pct(all_ms, 0.95):.0f}ms"
        )
        if fired_ms:
            body.append(
                f"**Latencia cuando SÍ se ejecuta** (excluye primer-turno): n={len(fired_ms)}, "
                f"p50={statistics.median(fired_ms):.0f}ms, p95={pct(fired_ms, 0.95):.0f}ms, "
                f"min={min(fired_ms):.0f}ms, max={max(fired_ms):.0f}ms"
            )
        body.append("")

    text = "\n".join(header + body)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n(guardado en {out_path})")

    # chequeos AUTOMÁTICOS del harness (el resto lo juzga un humano/Claude leyendo el markdown) --
    # invariantes estructurales donde SÍ hay una respuesta correcta objetiva, a diferencia de
    # "¿la respuesta está bien?".
    all_failures = subq_check_failures + history_check_failures + canonical_check_failures + gate_check_failures
    if all_failures:
        if subq_check_failures:
            print(f"\n❌ {len(subq_check_failures)} chequeo(s) de expect_n_subquestions fallaron:")
            for f in subq_check_failures:
                print(f"   - {f}")
        if history_check_failures:
            print(f"\n❌ {len(history_check_failures)} chequeo(s) de persistencia de historial fallaron:")
            for f in history_check_failures:
                print(f"   - {f}")
        if canonical_check_failures:
            print(f"\n❌ {len(canonical_check_failures)} chequeo(s) de respuestas canónicas fallaron:")
            for f in canonical_check_failures:
                print(f"   - {f}")
        if gate_check_failures:
            print(f"\n❌ {len(gate_check_failures)} chequeo(s) del gate de ambigüedad/confianza fallaron:")
            for f in gate_check_failures:
                print(f"   - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
