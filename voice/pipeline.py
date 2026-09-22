"""Orquestación: wake word → VAD → Whisper → reescritura de consulta → RAG → LLM → Piper."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .audio import Mic, Speaker
from .config import Config
from .guardrails import abstain_reply, is_chitchat
from .llm import LocalLLM
from .rag import Retriever
from .stt import Transcriber
from .tts import PiperTTS, StreamingSpeaker, iter_sentences
from .vad import UtteranceRecorder
from .wakeword import FRAME_SAMPLES, WakeWord

POST_WAKE_SKIP_MS = 200  # descarta la cola de la propia palabra de activación para que el VAD no la tome como frase


@dataclass
class Turn:
    """Resultado y métricas (segundos) de un turno de conversación."""
    question: str = ""             # lo que dijo/escribió el usuario, tal cual
    retrieval_query: str = ""      # pregunta usada para RAG (== question si no hubo reescritura)
    was_rewritten: bool = False
    answer: str = ""
    context_hits: list = field(default_factory=list)
    t_stt: float = 0.0
    t_rewrite: float = 0.0
    t_rag: float = 0.0
    t_llm_first_token: float = 0.0
    t_llm_total: float = 0.0
    n_tokens: int = 0
    t_first_audio: float = 0.0   # desde que terminó de hablar el usuario hasta el primer audio
    audio: np.ndarray | None = None
    sample_rate: int = 0


class Assistant:
    def __init__(self, cfg: Config, speak: bool = True):
        self.cfg = cfg
        log = lambda m: print(f"[init] {m}", flush=True)
        t = time.monotonic()
        log("cargando Whisper…")
        self.stt = Transcriber(cfg.stt)
        # la 1ra inferencia siempre tiene ~200ms extra (cuDNN/cuBLAS eligen algoritmo la primera vez);
        # se paga acá en vez de en la primera pregunta real del usuario.
        self.stt.transcribe(np.zeros(cfg.audio.sample_rate, dtype=np.int16))
        log(f"Whisper en {self.stt.device} ({time.monotonic() - t:.1f}s)")
        self.rag = Retriever(cfg.rag) if cfg.rag.enabled else None
        log(f"RAG: {len(self.rag.chunks) if self.rag else 0} chunks")
        t = time.monotonic()
        self.llm = LocalLLM(cfg.llm, cfg.rewrite)
        log(f"LLM listo ({time.monotonic() - t:.1f}s)")
        self.tts = PiperTTS(cfg.tts)
        self.speaker = Speaker(cfg.audio.speaker_name) if speak else None
        self.vad = UtteranceRecorder(cfg.vad, cfg.audio.sample_rate)
        self.wake = WakeWord(cfg.wakeword) if cfg.wakeword.enabled else None
        self.mic = Mic(cfg.audio.sample_rate, cfg.audio.mic_name)

    def reset_conversation(self) -> None:
        """Arranca una charla nueva: limpia el historial que usa voice/rewrite.py (usado por
        scripts/eval.py entre casos de prueba, para que uno no contamine al siguiente)."""
        self.llm.reset()

    # ---- un turno: audio -> texto -> respuesta hablada -------------------------------------
    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        t = time.monotonic()
        text = self.stt.transcribe(audio)
        return text, time.monotonic() - t

    def _abstain(self, question: str, turn: Turn, t0: float, has_history: bool) -> Turn:
        """Corta acá sin llamar al LLM de respuesta (ver voice/guardrails.py): sin esto, "responder
        como asistente general" era justo lo que llevaba a inventar identidades (bug 'Juan Carlos es
        el Rey de España', ver README)."""
        answer_text = abstain_reply(turn.retrieval_query, has_history)
        out = StreamingSpeaker(self.tts, self.speaker)
        print(f"🤖 {answer_text}", flush=True)
        out.say(answer_text)
        out.finish()
        self.llm.record_turn(question, answer_text)
        turn.answer = answer_text
        if out.first_audio_at:
            turn.t_first_audio = out.first_audio_at - t0
        if out.collected:
            turn.audio, turn.sample_rate = np.concatenate(out.collected), self.tts.sample_rate
        return turn

    def answer(self, question: str, t_user_done: float | None = None) -> Turn:
        turn = Turn(question=question)
        t0 = t_user_done or time.monotonic()
        has_history = bool(self.llm.history)

        t = time.monotonic()
        retrieval_query, was_rewritten = self.llm.rewrite_query(question)
        turn.t_rewrite = time.monotonic() - t
        turn.retrieval_query = retrieval_query
        turn.was_rewritten = was_rewritten
        if was_rewritten:
            print(f"   (reescrita: {retrieval_query!r})", flush=True)

        t = time.monotonic()
        hits = self.rag.retrieve(retrieval_query) if self.rag else []
        turn.t_rag = time.monotonic() - t
        turn.context_hits = hits

        if self.rag and not hits and not is_chitchat(retrieval_query):
            return self._abstain(question, turn, t0, has_history)

        context = Retriever.format_context(hits) if hits else None

        out = StreamingSpeaker(self.tts, self.speaker)
        t_llm = time.monotonic()
        first = None
        tokens = []

        def counted():
            nonlocal first
            # el LLM de respuesta recibe la pregunta YA reescrita/autónoma, sin el historial de la
            # charla (ver voice/llm.py: stream_answer es stateless a propósito)
            for tok in self.llm.stream_answer(retrieval_query, context):
                if first is None:
                    first = time.monotonic()
                tokens.append(tok)
                yield tok

        print("🤖 ", end="", flush=True)
        for sentence in iter_sentences(counted()):
            print(sentence, end=" ", flush=True)
            out.say(sentence)
        print(flush=True)
        turn.t_llm_total = time.monotonic() - t_llm
        turn.t_llm_first_token = (first or time.monotonic()) - t_llm
        turn.n_tokens = len(tokens)
        turn.answer = "".join(tokens).strip()
        out.finish()
        # se guarda la pregunta ORIGINAL (no la reescrita): así la próxima reescritura ve la charla
        # tal como pasó de verdad, no una versión ya "limpiada" de sí misma.
        self.llm.record_turn(question, turn.answer)
        if out.first_audio_at:
            turn.t_first_audio = out.first_audio_at - t0
        if out.collected:
            turn.audio, turn.sample_rate = np.concatenate(out.collected), self.tts.sample_rate
        return turn

    def print_metrics(self, t: Turn) -> None:
        tps = t.n_tokens / t.t_llm_total if t.t_llm_total else 0
        print(
            f"   ⏱ stt {t.t_stt:.2f}s | rewrite {t.t_rewrite * 1000:.0f}ms | rag {t.t_rag * 1000:.0f}ms | "
            f"llm 1er token {t.t_llm_first_token:.2f}s, {t.n_tokens} tok @ {tps:.1f} tok/s | "
            f"fin de voz → 1er audio {t.t_first_audio:.2f}s",
            flush=True,
        )

    # ---- loop interactivo ------------------------------------------------------------------
    def run(self) -> None:
        """El mic se cierra mientras se procesa y habla, así el asistente no se escucha a sí mismo."""
        sr, v = self.cfg.audio.sample_rate, self.cfg.vad
        wake_word = self.cfg.wakeword.model if self.wake else None
        print(f"\nListo. {'Di «' + wake_word + '»' if wake_word else 'Habla'} (Ctrl+C para salir)\n", flush=True)
        while True:
            # wake word y primera frase comparten el stream para no perder el inicio de la frase
            with self.mic.open() as mic:
                if self.wake:
                    self.wake.reset()
                    while not self.wake.triggered(mic.read(FRAME_SAMPLES)):
                        pass
                    print("👂 wake word detectado", flush=True)
                    mic.read(sr * POST_WAKE_SKIP_MS // 1000)
                audio = self.vad.record(mic)

            while audio is not None:
                t_done = time.monotonic()
                text, t_stt = self.transcribe(audio)
                if text:
                    print(f"🗣  {text}", flush=True)
                    turn = self.answer(text, t_done)
                    turn.t_stt = t_stt
                    self.print_metrics(turn)
                if v.followup_s <= 0:
                    break
                with self.mic.open() as mic:  # ventana de follow-up sin repetir wake word
                    audio = self.vad.record(mic, timeout_s=v.followup_s)
            print("   (esperando wake word…)\n" if self.wake else "", end="", flush=True)
