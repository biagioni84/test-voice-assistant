"""Un solo comando de verificación, pensado para correr ANTES DE CADA COMMIT (ver README, sección
"Cierre de etapa"). Corre, en orden:
  1. scripts/calibrate.py --check  -- lint de contenido (doc_topics, clarify_reply, respuestas
     canónicas) + confirma que los umbrales VIGENTES (rag.min_score, canonical.threshold) siguen
     sin producir casos peligrosos. NO recalibra, NO escribe nada.
  2. scripts/eval.py                -- la suite completa (~52 casos, dev + validación), con sus
     chequeos automáticos (expect_*, CJK universal) + el lint de doc_topics/clarify_reply que
     corre al arrancar.

Exit code 0 solo si los dos pasos terminan en 0. Pensado para no tener que acordarse de correr cada
script por separado -- un solo comando, un solo resultado.

    python scripts/verify.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(label: str, cmd: list[str]) -> int:
    print(f"\n{'=' * 72}\n▶ {label}: {' '.join(cmd)}\n{'=' * 72}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    return r.returncode


def main() -> None:
    steps = [
        ("calibrate.py --check (lint + umbrales vigentes)", [sys.executable, "scripts/calibrate.py", "--check"]),
        ("eval.py (suite completa)", [sys.executable, "scripts/eval.py"]),
    ]
    results = [(label, run(label, cmd)) for label, cmd in steps]

    print(f"\n{'=' * 72}\nRESUMEN\n{'=' * 72}")
    for label, code in results:
        print(f"  {'✅' if code == 0 else '❌'} {label} (exit {code})")

    if any(code != 0 for _, code in results):
        print("\n❌ VERIFICACIÓN FALLÓ -- no commitear hasta que todo pase")
        sys.exit(1)
    print("\n✅ VERIFICACIÓN OK -- listo para commit")


if __name__ == "__main__":
    main()
