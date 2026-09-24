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
from voice.guardrails import CJK_RE  # noqa: E402
from voice.pipeline import Assistant  # noqa: E402

rewrite_latencies_ms: list[float] = []  # se llena en run_case(), se reporta al final
subq_check_failures: list[str] = []  # casos con expect_n_subquestions que no dieron ese número
history_check_failures: list[str] = []  # turnos donde self.llm.history no creció (record_turn no se llamó)
canonical_check_failures: list[str] = []  # casos con expect_canonical_*/expect_exact_text/etc que no dieron lo esperado
gate_check_failures: list[str] = []  # casos con expect_clarify/expect_context_restricted_to que no dieron lo esperado
content_check_failures: list[str] = []  # casos con expect_contains/expect_not_contains/expect_abstain/etc
cjk_check_failures: list[str] = []  # CUALQUIER turno (o salida de reescritura) con caracteres CJK -- chequeo universal, sin campo
flaky_case_reports: list[str] = []  # veredicto de cada caso flaky: true (ver run_case_flaky) -- sección aparte, no cuenta para el exit code si pasa

# Plantillas fijas de voice/guardrails.py: abstain_reply() -- para expect_abstain. Comparación
# exacta a propósito: son texto FIJO, no generado, así que no hay variación legítima que tolerar.
_ABSTAIN_TEXTS = {
    "No lo encontré, ¿me lo preguntás de otra forma?",
    "No tengo información sobre esa persona.",
    "No tengo información sobre eso.",
}


def _check_cjk(label: str, text: str) -> None:
    """Chequeo AUTOMÁTICO y UNIVERSAL (no necesita campo expect_*, corre siempre): ningún texto que
    se muestra/habla debería tener caracteres CJK (ver voice/guardrails.py: CJK_RE, CJK_GRAMMAR).
    Antes esto se verificaba a mano contando caracteres en el .md generado; ahora es un assert real."""
    if CJK_RE.search(text):
        cjk_check_failures.append(f"{label}: {text!r}")


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

    out_text = " | ".join(subqs)
    _check_cjk(f"{case['id']} (rewrite)", out_text)

    expect_was_rewritten = case.get("expect_was_rewritten")
    if expect_was_rewritten is not None:
        ok = was_rewritten == expect_was_rewritten
        lines.append(f"  - {'✓' if ok else '✗ FALLÓ'}: expect_was_rewritten={expect_was_rewritten} (obtuvo {was_rewritten})")
        if not ok:
            content_check_failures.append(f"{case['id']}: expect_was_rewritten={expect_was_rewritten}, obtuvo {was_rewritten}")

    for needle in case.get("expect_contains", []):
        ok = needle.lower() in out_text.lower()
        lines.append(f"  - {'✓' if ok else '✗ FALLÓ'}: expect_contains {needle!r}")
        if not ok:
            content_check_failures.append(f"{case['id']}: no contiene {needle!r} -- salida: {out_text!r}")
    for needle in case.get("expect_not_contains", []):
        ok = needle.lower() not in out_text.lower()
        lines.append(f"  - {'✓' if ok else '✗ FALLÓ'}: expect_not_contains {needle!r}")
        if not ok:
            content_check_failures.append(f"{case['id']}: contiene {needle!r} (no debería) -- salida: {out_text!r}")

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
        _check_cjk(f"{case['id']}{suffix} ({q!r})", turn.answer)
        turn_results.append(turn)
    lines += _check_canonical_expectations(case, turn_results)
    lines += _check_gate_expectations(case, turn_results)
    lines += _check_content_expectations(case, turn_results)
    lines.append("")
    return lines


_FLAKY_RETRIES = 3
_FLAKY_MIN_PASS = 2


def run_case_flaky(bot: Assistant, case: dict, label: str | None, turns: list[str], show_chunk_text: bool) -> list[str]:
    """Para casos con `flaky: true` (25/09, ver README "reescritor: nodeterminismo de GPU"): NO es
    una licencia genérica para bajar el estándar de ningún caso -- solo para uno que YA se confirmó,
    con evidencia (comparación contra el código sin el cambio en cuestión, ver commit), que el mismo
    turno da un resultado distinto entre corridas por varianza de floating-point en el batching de
    GPU de llama-server, no por un bug de código. Reintenta el caso hasta _FLAKY_RETRIES veces
    (conversación nueva cada vez, bot.reset_conversation() ya lo hace run_case()); pasa si al menos
    _FLAKY_MIN_PASS de los intentos pasan TODOS sus chequeos de contenido/gate/canonical -- no exige
    unanimidad. Los chequeos que NO se tocan (CJK, historial, n_subquestions) siguen siendo
    invariantes duros: si alguno de esos falla en cualquier intento, no hay reintento que lo tape,
    porque esta función solo hace rollback de las 3 listas de "contenido" (ver abajo), nunca de las
    de seguridad/plumbing.

    Los intentos fallidos que terminan en un veredicto de PASE (>=_FLAKY_MIN_PASS) se sacan de las
    listas globales de fallos -- no deben contar para el exit code de la suite -- pero CADA intento
    (pase o no) se reporta igual en `flaky_case_reports`, una sección aparte del reporte final: la
    intermitencia queda visible, no se esconde detrás de un "todo verde" silencioso."""
    cid = case["id"]
    tracked = (content_check_failures, gate_check_failures, canonical_check_failures)
    attempts: list[tuple[bool, list[str]]] = []  # (pasó_este_intento, nuevas_fallas_de_este_intento)
    all_lines: list[str] = []
    for i in range(1, _FLAKY_RETRIES + 1):
        before = [len(lst) for lst in tracked]
        attempt_label = f"{label + ' -- ' if label else ''}intento {i}/{_FLAKY_RETRIES}"
        lines = run_case(bot, case, attempt_label, turns, show_chunk_text)
        new_failures: list[str] = []
        for lst, n_before in zip(tracked, before):
            new_failures += lst[n_before:]
            del lst[n_before:]  # rollback -- se re-agrega más abajo SOLO si el veredicto final es fallo
        passed = not new_failures
        attempts.append((passed, new_failures))
        all_lines += lines
        if sum(1 for ok, _ in attempts if ok) >= _FLAKY_MIN_PASS:
            break  # ya alcanzó el mínimo -- no hace falta gastar más intentos (ni más latencia)

    n_pass = sum(1 for ok, _ in attempts if ok)
    n_total = len(attempts)
    ok = n_pass >= _FLAKY_MIN_PASS
    verdict = f"{'✅ PASA' if ok else '❌ FALLA'} (marcado flaky: {n_pass}/{n_total} intentos, mínimo {_FLAKY_MIN_PASS})"
    flaky_case_reports.append(f"- {cid}{f' ({label})' if label else ''}: {verdict}")
    for i, (passed, new_failures) in enumerate(attempts, 1):
        if not passed:
            for f in new_failures:
                flaky_case_reports.append(f"    - intento {i}/{n_total} falló: {f}")
    if not ok:
        # el veredicto final es fallo real (menos de _FLAKY_MIN_PASS intentos pasaron) -- ESTO sí
        # cuenta para el exit code, con un resumen (no cada línea repetida de cada intento, eso ya
        # queda en flaky_case_reports arriba).
        content_check_failures.append(
            f"{cid}: marcado flaky pero solo {n_pass}/{n_total} intentos pasaron (mínimo {_FLAKY_MIN_PASS}) -- "
            f"ver sección 'casos inestables' del reporte para el detalle de cada intento"
        )
    all_lines.append(f"**Veredicto flaky para {cid}{f' ({label})' if label else ''}:** {verdict}")
    all_lines.append("")
    return all_lines


def _check_content_expectations(case: dict, turn_results: list) -> list[str]:
    """Chequeos AUTOMÁTICOS sobre el CONTENIDO de la respuesta -- para casos donde la salida del
    LLM no se puede comparar textual (varía turno a turno aunque temperature=0, ver README), se
    verifica lo que SÍ es estable: presencia/ausencia de datos concretos, si abstuvo, si coincide
    con un turno anterior, o el documento usado como CONTEXTO. Todos opcionales, aplicados al
    ÚLTIMO turno salvo que se diga lo contrario:
      expect_abstain: true -- el último turno debe ser exactamente una de las plantillas fijas de
        abstención (voice/guardrails.py: abstain_reply) -- comparación EXACTA, son texto fijo.
      expect_contains: [str, ...] -- TODAS estas substrings (case-insensitive) deben aparecer en
        la respuesta del último turno.
      expect_not_contains: [str, ...] -- NINGUNA de estas substrings debe aparecer en el último turno.
      expect_not_contains_any_turn: [str, ...] -- como expect_not_contains pero revisa TODOS los
        turnos de la conversación, no solo el último (para casos multi-turno donde el riesgo es
        que aparezca en CUALQUIER punto, no solo al final).
      expect_contains_each_turn: [[str,...], [str,...], ...] -- una lista de listas, un elemento
        por turno (en el mismo orden que `turns`); lista vacía = sin chequeo para ese turno
        (documenta explícitamente qué turno se deja sin aserción y por qué, en vez de omitirlo).
      expect_doc: <archivo> -- alias de expect_context_restricted_to para casos sin gate
        (RAG normal): el CONTEXTO del último turno debe venir de ese único documento.
      expect_same_as_previous: true -- el último turno debe responder EXACTAMENTE lo mismo que el
        anterior (consistencia). NOTA: temperature=0 debería ser determinístico, pero se observó
        alguna variación por efectos de batching de la GPU (ver README) -- si este chequeo falla
        de forma intermitente sin cambios de código, es señal de esa varianza, no necesariamente
        un bug nuevo."""
    out: list[str] = []
    cid = case["id"]
    last = turn_results[-1]

    if case.get("expect_abstain"):
        ok = last.answer in _ABSTAIN_TEXTS
        out.append(f"  - {'✓' if ok else '✗ FALLÓ'}: expect_abstain (obtuvo {last.answer!r})")
        if not ok:
            content_check_failures.append(f"{cid}: expect_abstain -- obtuvo {last.answer!r}, no es ninguna plantilla fija")

    for needle in case.get("expect_contains", []):
        ok = needle.lower() in last.answer.lower()
        out.append(f"  - {'✓' if ok else '✗ FALLÓ'}: expect_contains {needle!r}")
        if not ok:
            content_check_failures.append(f"{cid}: no contiene {needle!r} -- respuesta: {last.answer!r}")

    for needle in case.get("expect_not_contains", []):
        ok = needle.lower() not in last.answer.lower()
        out.append(f"  - {'✓' if ok else '✗ FALLÓ'}: expect_not_contains {needle!r}")
        if not ok:
            content_check_failures.append(f"{cid}: contiene {needle!r} (no debería) -- respuesta: {last.answer!r}")

    for needle in case.get("expect_not_contains_any_turn", []):
        offenders = [i for i, t in enumerate(turn_results, 1) if needle.lower() in t.answer.lower()]
        ok = not offenders
        out.append(f"  - {'✓' if ok else '✗ FALLÓ'}: expect_not_contains_any_turn {needle!r}")
        if not ok:
            content_check_failures.append(f"{cid}: {needle!r} aparece en el/los turno(s) {offenders} (no debería en ninguno)")

    each_turn = case.get("expect_contains_each_turn")
    if each_turn is not None:
        for i, (needles, turn) in enumerate(zip(each_turn, turn_results), 1):
            for needle in needles:
                ok = needle.lower() in turn.answer.lower()
                out.append(f"  - {'✓' if ok else '✗ FALLÓ'}: turno {i} expect_contains {needle!r}")
                if not ok:
                    content_check_failures.append(f"{cid}: turno {i} no contiene {needle!r} -- respuesta: {turn.answer!r}")

    expect_doc = case.get("expect_doc")
    if expect_doc is not None:
        docs = {h.source for h in last.context_hits}
        ok = docs == {expect_doc}
        out.append(f"  - {'✓' if ok else '✗ FALLÓ'}: expect_doc={expect_doc!r} (obtuvo docs={sorted(docs)!r})")
        if not ok:
            content_check_failures.append(f"{cid}: expect_doc={expect_doc!r}, obtuvo docs={sorted(docs)!r}")

    if case.get("expect_same_as_previous"):
        prev = turn_results[-2]
        ok = last.answer == prev.answer
        out.append(f"  - {'✓' if ok else '✗ FALLÓ'}: expect_same_as_previous (anterior={prev.answer!r}, último={last.answer!r})")
        if not ok:
            content_check_failures.append(f"{cid}: expect_same_as_previous -- anterior={prev.answer!r} != último={last.answer!r}")

    return out


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


# Campos que cuentan como "este caso tiene al menos un chequeo automático" -- todo lo que termina
# revisado por alguna de las funciones _check_*_expectations de arriba. El campo `expect` (prosa
# libre) NO cuenta: sigue siendo juicio humano/Claude, a propósito (ver docstring del módulo).
_AUTO_ASSERTION_FIELDS = [
    "expect_n_subquestions", "expect_was_rewritten", "expect_contains", "expect_not_contains",
    "expect_canonical_entries", "expect_exact_text", "expect_variant_rotation",
    "expect_repeat_of_previous", "expect_clarify", "expect_context_restricted_to",
    "expect_abstain", "expect_not_contains_any_turn", "expect_contains_each_turn",
    "expect_doc", "expect_same_as_previous",
]


def _count_assertions(cases: list[dict]) -> dict:
    """Cobertura de aserciones automáticas por caso y por split -- para el reporte final (ver
    main()) y para que quede a la vista cuáles casos siguen sin ninguna, con su propio `note` como
    explicación (en vez de tener que buscarlos)."""
    by_split: dict[str, list[str]] = {}
    without: list[tuple[str, str]] = []  # (id, note) de los que no tienen ningún expect_* automático
    for c in cases:
        split = c.get("split", "dev")
        has_check = any(f in c for f in _AUTO_ASSERTION_FIELDS)
        by_split.setdefault(split, [0, 0])
        by_split[split][0] += 1
        if has_check:
            by_split[split][1] += 1
        else:
            without.append((c["id"], (c.get("note") or "").strip().split("\n")[0][:100]))
    total = len(cases)
    with_check = sum(v[1] for v in by_split.values())
    print("\n**Cobertura de aserciones automáticas por split:**")
    for split, (tot, ok) in by_split.items():
        print(f"  {split}: {ok}/{tot} casos con al menos un chequeo automático")
    if without:
        print(f"\n**{len(without)} caso(s) SIN aserción automática** (solo `expect:` en prosa, juicio humano/Claude):")
        for cid, note in without:
            print(f"  - {cid}: {note or '(sin nota -- revisar por qué)'}")
    return {"with": with_check, "total": total, "without": total - with_check}


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
            elif case.get("flaky"):
                for label, turns in case_variants(case):
                    body += run_case_flaky(bot, case, label, turns, args.show_chunk_text)
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
    all_failures = (
        subq_check_failures + history_check_failures + canonical_check_failures + gate_check_failures
        + content_check_failures + cjk_check_failures
    )
    n_checks = _count_assertions(cases)
    print(
        f"\n**Cobertura de aserciones automáticas**: {n_checks['with']} de {n_checks['total']} casos "
        f"tienen al menos un chequeo automático ({n_checks['without']} sin ninguno -- ver el reporte "
        f"de más arriba o README para por qué en cada caso)."
    )
    if flaky_case_reports:
        # sección APARTE, informativa -- un caso flaky que terminó en PASE (>=2/3 intentos) ya
        # tuvo sus fallas intermedias sacadas de all_failures de arriba (ver run_case_flaky):
        # no cuentan para el exit code, pero quedan visibles acá, no escondidas.
        print("\n**Casos marcados `flaky: true`** (ver README \"reescritor: nodeterminismo de GPU\"):")
        for line in flaky_case_reports:
            print(f"   {line}")
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
        if content_check_failures:
            print(f"\n❌ {len(content_check_failures)} chequeo(s) de contenido fallaron:")
            for f in content_check_failures:
                print(f"   - {f}")
        if cjk_check_failures:
            print(f"\n❌ {len(cjk_check_failures)} caso(s) con caracteres CJK:")
            for f in cjk_check_failures:
                print(f"   - {f}")
        sys.exit(1)
    print("\n✅ todos los chequeos automáticos pasaron")


if __name__ == "__main__":
    main()
