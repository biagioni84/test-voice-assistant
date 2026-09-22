"""Modo "test": corre los casos de tests/eval_questions.yaml contra el LLM/RAG real (sin STT/mic) y
vuelca todo a un archivo Markdown. Herramienta de desarrollo, no de producción: no calcula pass/fail
solo -- el juicio de si cada caso está bien lo hace un humano (o Claude) leyendo el resultado.

    python scripts/eval.py                        # corre todos los casos de tests/eval_questions.yaml
    python scripts/eval.py --cases tests/otro.yaml
    python scripts/eval.py --out eval_results/mi_corrida.md
    python scripts/eval.py --show-chunk-text       # además del source@score, el texto completo del
                                                    # chunk recuperado (para diagnosticar si el RAG
                                                    # trajo lo que debía, sin adivinar)
"""
import argparse
import datetime
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import voice  # noqa: F401,E402  (importa llama_cpp primero)
from voice.config import load_config  # noqa: E402
from voice.pipeline import Assistant  # noqa: E402


def case_variants(case: dict) -> list[tuple[str | None, list[str]]]:
    """Con temperature=0 (determinístico) repetir la MISMA pregunta no aporta nada -- por eso
    'variants' reemplaza al viejo 'repeat': varias formulaciones distintas del mismo caso, para
    medir robustez a cómo se pregunta en vez de robustez al muestreo aleatorio."""
    if "variants" in case:
        n = len(case["variants"])
        return [(f"variante {i}/{n}", v) for i, v in enumerate(case["variants"], 1)]
    return [(None, case["turns"])]


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
        lines.append(f"- 🗣 **{q}**")
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
    ap.add_argument("--show-chunk-text", action="store_true", help="ver el texto completo de cada chunk recuperado")
    args = ap.parse_args()

    cases_path = ROOT / args.cases
    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]

    cfg = load_config()
    cfg.wakeword.enabled = False
    bot = Assistant(cfg, speak=False)

    out_path = Path(args.out) if args.out else ROOT / "eval_results" / f"{datetime.datetime.now():%Y%m%d_%H%M%S}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [f"# Eval run — {datetime.datetime.now():%Y-%m-%d %H:%M} — {cases_path.name}", ""]
    body: list[str] = []
    for case in cases:
        for label, turns in case_variants(case):
            body += run_case(bot, case, label, turns, args.show_chunk_text)

    text = "\n".join(header + body)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n(guardado en {out_path})")


if __name__ == "__main__":
    main()
