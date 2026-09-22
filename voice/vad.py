"""VAD con Silero (ONNX vía pysilero-vad) + segmentador de frases."""
from __future__ import annotations

import collections
import time

import numpy as np
from pysilero_vad import SileroVoiceActivityDetector

from .config import VadCfg

CHUNK = 512  # muestras (32 ms @ 16 kHz) que exige Silero


class UtteranceRecorder:
    def __init__(self, cfg: VadCfg, sample_rate: int = 16000):
        self.cfg = cfg
        self.sr = sample_rate
        self._vad = SileroVoiceActivityDetector()
        self._chunk_ms = 1000 * CHUNK / sample_rate

    def record(self, mic, timeout_s: float | None = None) -> np.ndarray | None:
        """Lee del mic hasta que termina una frase. None si nadie habló antes del timeout."""
        c = self.cfg
        timeout_s = c.no_speech_timeout_s if timeout_s is None else timeout_s
        self._vad.reset()
        preroll = collections.deque(maxlen=int(300 / self._chunk_ms))  # 300 ms previos al inicio de voz
        frames: list[np.ndarray] = []
        started = False
        speech_ms = silence_ms = 0.0
        t0 = time.monotonic()

        while True:
            chunk = mic.read(CHUNK)
            # histéresis: umbral menor para seguir "en voz" que para arrancar
            thr = c.speech_threshold - (0.15 if started else 0.0)
            is_speech = self._vad(chunk.tobytes()) >= thr

            if not started:
                preroll.append(chunk)
                if is_speech:
                    started = True
                    frames.extend(preroll)
                    speech_ms = self._chunk_ms
                elif time.monotonic() - t0 > timeout_s:
                    return None
                continue

            frames.append(chunk)
            if is_speech:
                speech_ms += self._chunk_ms
                silence_ms = 0.0
            else:
                silence_ms += self._chunk_ms
                if silence_ms >= c.end_silence_ms:
                    break
            if len(frames) * self._chunk_ms / 1000 >= c.max_utterance_s:
                break

        if speech_ms < c.min_speech_ms:
            return None
        return np.concatenate(frames)
