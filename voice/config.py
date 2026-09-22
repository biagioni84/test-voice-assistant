"""Carga config.toml en dataclasses tipadas."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AudioCfg:
    sample_rate: int = 16000
    mic_name: str = ""
    speaker_name: str = ""


@dataclass
class WakeCfg:
    enabled: bool = True
    model: str = "hey_jarvis"
    threshold: float = 0.5
    vad_threshold: float = 0.5     # 0 = desactivado; si >0, exige que Silero detecte voz para contar
                                    # el wake word (filtra falsos positivos por TV, música, ruido)
    noise_suppression: bool = True  # SpeexDSP; barato y ayuda con ruido de fondo estacionario (ventilador)


@dataclass
class VadCfg:
    speech_threshold: float = 0.5
    end_silence_ms: int = 900
    min_speech_ms: int = 250
    no_speech_timeout_s: float = 6.0
    max_utterance_s: float = 20.0
    followup_s: float = 5.0


@dataclass
class SttCfg:
    model: str = "turbo"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    language: str = "es"
    beam_size: int = 1


@dataclass
class RagCfg:
    enabled: bool = True
    docs_dir: str = "docs"
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 3
    min_score: float = 0.30
    chunk_chars: int = 500


@dataclass
class LlmCfg:
    repo: str = "bartowski/Qwen2.5-7B-Instruct-GGUF"
    file: str = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    n_ctx: int = 4096
    n_threads: int = 6
    n_gpu_layers: int = 0
    max_tokens: int = 200
    temperature: float = 0.0  # 0 = greedy/determinístico; con >0 la misma conversación puede dar
                               # resultados distintos entre corridas (ver README, tests/eval_questions.yaml)
    system_prompt: str = "Eres un asistente de voz. Responde en español, breve."
    # OJO: el LLM de respuesta ya NO recibe el historial de la charla (ver voice/rewrite.py) -- por
    # eso no hay un "history_turns" acá; el que le llega al reescritor está en [rewrite].


@dataclass
class RewriteCfg:
    enabled: bool = True
    history_turns: int = 2   # últimos N turnos (usuario+asistente) que ve el reescritor
    max_tokens: int = 40


@dataclass
class TtsCfg:
    voice: str = "es_AR-daniela-high"
    voices_dir: str = "models/piper"
    length_scale: float = 1.0


@dataclass
class Config:
    audio: AudioCfg
    wakeword: WakeCfg
    vad: VadCfg
    stt: SttCfg
    rag: RagCfg
    llm: LlmCfg
    rewrite: RewriteCfg
    tts: TtsCfg


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    parts = {}
    for f in fields(Config):
        section_cls = f.type if not isinstance(f.type, str) else globals()[f.type]
        parts[f.name] = section_cls(**raw.get(f.name, {}))
    return Config(**parts)


def resolve(p: str | Path) -> Path:
    """Rutas relativas de la config se resuelven contra la raíz del proyecto."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
