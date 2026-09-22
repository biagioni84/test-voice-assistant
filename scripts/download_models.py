"""Descarga todos los modelos necesarios (idempotente). Uso: python scripts/download_models.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import voice  # noqa: F401,E402  (importa llama_cpp primero)
from voice.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()

    print("[1/5] openWakeWord (melspectrogram + embedding + wake words)")
    import openwakeword
    openwakeword.utils.download_models()

    print("[2/5] Whisper turbo (faster-whisper / CTranslate2)")
    from faster_whisper.utils import download_model
    download_model(cfg.stt.model)

    print(f"[3/5] LLM GGUF: {cfg.llm.repo}/{cfg.llm.file}")
    from huggingface_hub import hf_hub_download
    hf_hub_download(cfg.llm.repo, cfg.llm.file, local_dir=ROOT / "models" / "llm")

    print(f"[4/5] Voz Piper: {cfg.tts.voice}")
    from piper.download_voices import download_voice
    voices_dir = ROOT / cfg.tts.voices_dir
    voices_dir.mkdir(parents=True, exist_ok=True)
    download_voice(cfg.tts.voice, voices_dir)

    print(f"[5/5] Embeddings RAG: {cfg.rag.embed_model}")
    from fastembed import TextEmbedding
    TextEmbedding(cfg.rag.embed_model, cache_dir=str(ROOT / "models" / "embeddings"))

    print("\nListo.")


if __name__ == "__main__":
    main()
