"""Speech-to-text con faster-whisper (large-v3-turbo) en GPU."""
from __future__ import annotations

import ctypes
import glob
import importlib.util
import os

import numpy as np

from .config import SttCfg


def _preload_cuda_libs() -> None:
    """CTranslate2 necesita cuBLAS 12 y cuDNN 9; los traen los wheels nvidia-* de pip pero fuera del
    loader path. Los cargamos a mano (RTLD_GLOBAL) para no depender de LD_LIBRARY_PATH."""
    for pkg, patterns in (
        ("nvidia.cublas", ["libcublasLt.so.12", "libcublas.so.12"]),
        ("nvidia.cudnn", ["libcudnn.so.9", "libcudnn_*.so.9"]),
    ):
        spec = importlib.util.find_spec(pkg)
        if not spec or not spec.submodule_search_locations:
            continue
        libdir = os.path.join(list(spec.submodule_search_locations)[0], "lib")
        for pat in patterns:
            for path in sorted(glob.glob(os.path.join(libdir, pat))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


class Transcriber:
    def __init__(self, cfg: SttCfg):
        self.cfg = cfg
        if cfg.device == "cuda":
            _preload_cuda_libs()
        from faster_whisper import WhisperModel

        try:
            self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
            self.device = cfg.device
        except Exception as e:  # sin CUDA / sin VRAM -> CPU
            print(f"[stt] fallo en {cfg.device} ({e}); usando CPU int8")
            self.model = WhisperModel(cfg.model, device="cpu", compute_type="int8")
            self.device = "cpu"

    def transcribe(self, audio_int16: np.ndarray) -> str:
        audio = audio_int16.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,  # ya segmentamos con Silero
        )
        return " ".join(s.text.strip() for s in segments).strip()
