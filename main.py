"""Asistente de voz local. Ejemplos:
    python main.py                      # loop completo: wake word -> VAD -> STT -> RAG -> LLM -> TTS
    python main.py --no-wake            # sin wake word (escucha directo)
    python main.py --text "¿a qué hora abre la oficina?"   # salta audio de entrada
    python main.py --wav pregunta.wav   # usa un WAV 16kHz mono como entrada
"""
import argparse
import wave

import numpy as np

import voice  # noqa: F401  (importa llama_cpp primero)
from voice.config import load_config
from voice.pipeline import Assistant


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, "se espera WAV 16kHz mono"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-wake", action="store_true")
    ap.add_argument("--text", help="pregunta por texto (sin micrófono ni STT)")
    ap.add_argument("--wav", help="pregunta desde WAV 16kHz mono")
    ap.add_argument("--no-speak", action="store_true", help="no reproducir audio")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_wake:
        cfg.wakeword.enabled = False
    bot = Assistant(cfg, speak=not args.no_speak)

    if args.text or args.wav:
        t_stt = 0.0
        text = args.text
        if args.wav:
            text, t_stt = bot.transcribe(read_wav(args.wav))
            print(f"🗣  {text}")
        turn = bot.answer(text)
        turn.t_stt = t_stt
        bot.print_metrics(turn)
        return

    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nchau")


if __name__ == "__main__":
    main()
