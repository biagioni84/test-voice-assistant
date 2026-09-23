"""TTS con Piper + habla en streaming: los tokens del LLM se cortan en frases y cada frase se
sintetiza y reproduce mientras el LLM sigue generando (esconde gran parte de la latencia)."""
from __future__ import annotations

import queue
import re
import threading
import time
from typing import Iterable, Iterator

import numpy as np
from piper import PiperVoice, SynthesisConfig

from .audio import Speaker
from .config import TtsCfg, resolve
from .guardrails import CJK_RE

_SENT_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def iter_sentences(tokens: Iterable[str]) -> Iterator[str]:
    """Agrupa tokens en frases completas. Emite en cuanto hay un fin de frase."""
    buf = ""
    for tok in tokens:
        buf += tok
        parts = _SENT_END.split(buf)
        if len(parts) > 1:
            *done, buf = parts
            for s in done:
                s = s.strip()
                if s:
                    yield s
    if buf.strip():
        yield buf.strip()


_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # símbolos, emoticones, transporte, símbolos suplementarios
    "\U00002600-\U000027BF"  # símbolos varios y dingbats
    "\U0001F1E6-\U0001F1FF"  # indicadores regionales (banderas)
    "\U0000FE0F"              # variation selector-16 (fuerza estilo emoji)
    "\U0000200D"              # zero-width joiner (emojis compuestos)
    "]+"
)


def clean_for_speech(text: str) -> str:
    """Quita markdown, emojis y caracteres CJK que Piper leería mal. Lo de CJK es una red de
    seguridad: el logit_bias en voice/llm.py ya debería prevenir que el LLM los genere (ver
    voice/guardrails.py y README), esto es por si algo igual se escapa."""
    text = _EMOJI.sub("", text)
    text = CJK_RE.sub("", text)
    text = re.sub(r"[*_#`>~|]+", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


class PiperTTS:
    def __init__(self, cfg: TtsCfg):
        model = resolve(cfg.voices_dir) / f"{cfg.voice}.onnx"
        self.voice = PiperVoice.load(model)
        self.sample_rate = self.voice.config.sample_rate
        self._syn = SynthesisConfig(length_scale=cfg.length_scale)

    def synth(self, text: str) -> np.ndarray:
        chunks = [c.audio_int16_array for c in self.voice.synthesize(text, self._syn)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


class StreamingSpeaker:
    """say(frase) encola; un hilo sintetiza con Piper y el Speaker reproduce. finish() espera el final."""

    def __init__(self, tts: PiperTTS, speaker: Speaker | None):
        self.tts, self.speaker = tts, speaker
        self._q: queue.Queue = queue.Queue()
        self.first_audio_at: float | None = None  # time.monotonic() del primer audio listo
        self.collected: list[np.ndarray] = []      # para guardar a WAV / tests
        self.synth_ms: list[float] = []            # tiempo de síntesis por frase, en orden -- para
                                                     # desglosar la latencia (ver voice/pipeline.py)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def say(self, sentence: str) -> None:
        self._q.put(sentence)

    def finish(self) -> None:
        self._q.join()
        if self.speaker:
            self.speaker.wait()

    def _run(self) -> None:
        while True:
            s = self._q.get()
            try:
                t0 = time.monotonic()
                audio = self.tts.synth(clean_for_speech(s))
                self.synth_ms.append((time.monotonic() - t0) * 1000)
                if self.first_audio_at is None:
                    self.first_audio_at = time.monotonic()
                self.collected.append(audio)
                if self.speaker:
                    self.speaker.play(audio, self.tts.sample_rate)
            finally:
                self._q.task_done()
