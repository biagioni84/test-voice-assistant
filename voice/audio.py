"""Entrada (micrófono) y salida (parlantes) de audio vía PulseAudio/WSLg usando soundcard."""
from __future__ import annotations

import queue
import threading
import warnings
from contextlib import contextmanager

import numpy as np
import soundcard as sc

# soundcard avisa de "data discontinuity" cada vez que no leemos el mic a tiempo (p.ej. durante STT/LLM).
warnings.filterwarnings("ignore", message="data discontinuity in recording")


def _pick(devices, name: str):
    if not name:
        return None
    for d in devices:
        if name.lower() in d.name.lower():
            return d
    raise RuntimeError(f"No se encontró dispositivo '{name}'. Disponibles: {[d.name for d in devices]}")


class Mic:
    """Stream de micrófono mono int16. Uso: with Mic(sr).open() as stream: stream.read(512)."""

    def __init__(self, sample_rate: int = 16000, name: str = ""):
        self.sample_rate = sample_rate
        self._dev = _pick(sc.all_microphones(), name) or sc.default_microphone()

    @contextmanager
    def open(self):
        with self._dev.recorder(samplerate=self.sample_rate, channels=1, blocksize=self.sample_rate // 10) as rec:
            yield _MicStream(rec)


class _MicStream:
    def __init__(self, rec):
        self._rec = rec

    def read(self, n: int) -> np.ndarray:
        x = self._rec.record(numframes=n)[:, 0]
        return (np.clip(x, -1, 1) * 32767).astype(np.int16)


class Speaker:
    """Reproduce arrays int16 en un hilo aparte, en orden. `wait()` bloquea hasta vaciar la cola."""

    def __init__(self, name: str = ""):
        self._dev = _pick(sc.all_speakers(), name) or sc.default_speaker()
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        self._q.put((audio, sample_rate))

    def wait(self) -> None:
        self._q.join()

    def interrupt(self) -> None:
        """Descarta lo pendiente (el fragmento en curso termina solo)."""
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            pass
        self._stop.clear()

    def _run(self) -> None:
        player, rate = None, None
        while True:
            audio, sr = self._q.get()
            try:
                if not self._stop.is_set():
                    if sr != rate:
                        if player:
                            player.__exit__(None, None, None)
                        player = self._dev.player(samplerate=sr, channels=1)
                        player.__enter__()
                        rate = sr
                    player.play(audio.astype(np.float32) / 32768.0)
            finally:
                self._q.task_done()
