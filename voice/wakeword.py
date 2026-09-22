"""Wake word con openWakeWord (backend ONNX; tflite-runtime no tiene wheel para Python 3.12)."""
from __future__ import annotations

import numpy as np
from openwakeword.model import Model

from .config import WakeCfg

FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz, tamaño recomendado por openWakeWord


class WakeWord:
    def __init__(self, cfg: WakeCfg):
        self.threshold = cfg.threshold
        self._model = Model(
            wakeword_models=[cfg.model],
            inference_framework="onnx",
            vad_threshold=cfg.vad_threshold,
            enable_speex_noise_suppression=cfg.noise_suppression,
        )

    def detect(self, frame: np.ndarray) -> float:
        """Devuelve el score (0-1) para un frame int16 de FRAME_SAMPLES."""
        return max(self._model.predict(frame).values())

    def triggered(self, frame: np.ndarray) -> bool:
        return self.detect(frame) >= self.threshold

    def reset(self) -> None:
        self._model.reset()
