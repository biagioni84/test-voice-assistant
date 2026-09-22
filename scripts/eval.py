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
from voice.config import load_config  # noqa: E402
from voice.pipeline import Assistant  # noqa: E402

rewrite_latencies_ms: list[float] = []  # se llena en run_case(), se reporta al final


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
    out, was_rewritten = bot.llm.rewrite_query(case["question"])
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
    lines.append(f"  - ➜ reescrita: **{out}** (se_reescribió={was_rewritten}, {dt_ms:.0f}ms)")
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
    for q in turns:
        turn = bot.answer(q)
        if turn.t_rewrite:
            rewrite_latencies_ms.append(turn.t_rewrite * 1000)
        lines.append(f"- 🗣 **{q}**")
        if turn.was_rewritten:
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
    lines.append("")
    return lines


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


if __name__ == "__main__":
    main()
