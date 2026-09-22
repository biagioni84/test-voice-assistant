"""Modo "test": corre los casos de tests/eval_questions.yaml contra el LLM/RAG real (sin STT/mic) y
vuelca todo a un archivo Markdown. Herramienta de desarrollo, no de producción: no calcula pass/fail
solo -- el juicio de si cada caso está bien lo hace un humano (o Claude) leyendo el resultado.

    python scripts/eval.py                        # corre todos los casos de tests/eval_questions.yaml
    python scripts/eval.py --cases tests/otro.yaml
    python scripts/eval.py --out eval_results/mi_corrida.md
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


def run_case(bot: Assistant, case: dict, run_n: int | None = None) -> list[str]:
    bot.reset_conversation()
    suffix = f" (corrida {run_n[0]}/{run_n[1]})" if run_n else ""
    lines = [f"## {case['id']}{suffix}"]
    if run_n is None or run_n[0] == 1:
        if case.get("note"):
            lines.append(f"> **nota:** {case['note'].strip()}")
        if case.get("expect"):
            lines.append(f"> **se espera:** {case['expect'].strip()}")
    lines.append("")
    for q in case["turns"]:
        turn = bot.answer(q)
        hits = ", ".join(f"{h.source}@{h.score:.2f}" for h in turn.context_hits) or "*(sin contexto)*"
        lines.append(f"- 🗣 **{q}**")
        lines.append(f"  - 🤖 {turn.answer}")
        lines.append(f"  - RAG: {hits}")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="tests/eval_questions.yaml")
    ap.add_argument("--out", default=None, help="default: eval_results/<timestamp>.md")
    args = ap.parse_args()

    cases_path = ROOT / args.cases
    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))

    cfg = load_config()
    cfg.wakeword.enabled = False
    bot = Assistant(cfg, speak=False)

    out_path = Path(args.out) if args.out else ROOT / "eval_results" / f"{datetime.datetime.now():%Y%m%d_%H%M%S}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [f"# Eval run — {datetime.datetime.now():%Y-%m-%d %H:%M} — {cases_path.name}", ""]
    body: list[str] = []
    for case in cases:
        n = case.get("repeat", 1)
        for i in range(1, n + 1):
            body += run_case(bot, case, run_n=(i, n) if n > 1 else None)

    text = "\n".join(header + body)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n(guardado en {out_path})")


if __name__ == "__main__":
    main()
