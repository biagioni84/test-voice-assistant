"""Benchmark end-to-end sin micrófono: Piper genera la pregunta hablada -> Whisper -> RAG -> LLM -> Piper.
Sirve también como test de humo de toda la cadena y para comparar LLMs / hilos / capas en GPU.

    python scripts/bench.py
    python scripts/bench.py --llm-file Qwen2.5-3B-Instruct-Q4_K_M.gguf --threads 8
    python scripts/bench.py --gpu-layers 6 --save-wav /tmp/out.wav
"""
import argparse
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import voice  # noqa: F401,E402
from voice.config import load_config  # noqa: E402
from voice.pipeline import Assistant  # noqa: E402

QUESTIONS = [
    "¿A qué hora abre la oficina los sábados?",
    "¿Cómo puedo restablecer la contraseña del correo?",
    "Contame un chiste corto.",  # sin contexto RAG relevante: prueba el camino sin documentos
]


def vram_mb() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    used, total = out.split(", ")
    return f"{used}/{total} MiB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-file")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--gpu-layers", type=int)
    ap.add_argument("--stt-device")
    ap.add_argument("--save-wav")
    args = ap.parse_args()

    cfg = load_config()
    cfg.wakeword.enabled = False
    if args.llm_file:
        cfg.llm.file = args.llm_file
    if args.threads:
        cfg.llm.n_threads = args.threads
    if args.gpu_layers is not None:
        cfg.llm.n_gpu_layers = args.gpu_layers
    if args.stt_device:
        cfg.stt.device = args.stt_device
        cfg.stt.compute_type = "int8" if args.stt_device == "cpu" else cfg.stt.compute_type

    print(f"VRAM antes de cargar: {vram_mb()}")
    bot = Assistant(cfg, speak=False)
    print(f"VRAM con todo cargado: {vram_mb()}")
    print(f"LLM={cfg.llm.file} threads={cfg.llm.n_threads} gpu_layers={cfg.llm.n_gpu_layers}\n")

    rows, last = [], None
    for q in QUESTIONS:
        bot.llm.reset()
        spoken = bot.tts.synth(q)
        audio16 = resample_poly(spoken, 16000 // 2, bot.tts.sample_rate // 2).astype(np.int16)
        text, t_stt = bot.transcribe(audio16)
        print(f"🎤 \"{q}\"\n🗣  Whisper entendió: \"{text}\"")
        turn = bot.answer(text or q)
        turn.t_stt = t_stt
        bot.print_metrics(turn)
        print(f"   RAG: {[(round(h.score, 2), h.source) for h in turn.context_hits]}\n")
        rows.append(turn)
        last = turn

    print(f"VRAM tras inferencia: {vram_mb()}")
    tps = [t.n_tokens / t.t_llm_total for t in rows if t.t_llm_total]
    print(f"Promedio: {np.mean(tps):.1f} tok/s | 1er token {np.mean([t.t_llm_first_token for t in rows]):.2f}s | "
          f"stt {np.mean([t.t_stt for t in rows]):.2f}s | fin de voz→1er audio {np.mean([t.t_first_audio for t in rows]):.2f}s")

    if args.save_wav and last and last.audio is not None:
        with wave.open(args.save_wav, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(last.sample_rate)
            w.writeframes(last.audio.tobytes())
        print(f"Última respuesta guardada en {args.save_wav}")


if __name__ == "__main__":
    main()
