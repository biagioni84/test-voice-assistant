"""Orquestación: wake word → VAD → Whisper → reescritura de consulta → RAG → LLM → Piper."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .audio import Mic, Speaker
from .canonical import CanonicalMatcher
from .config import Config
from .guardrails import (
    abstain_partial_reply,
    abstain_reply,
    ambiguous_docs,
    clarify_reply,
    is_chitchat,
    is_repeat_request,
)
from .llm import LocalLLM
from .rag import Retriever
from .rewrite import strip_part_labels
from .stt import Transcriber
from .tts import PiperTTS, StreamingSpeaker, iter_sentences
from .vad import UtteranceRecorder
from .wakeword import FRAME_SAMPLES, WakeWord

POST_WAKE_SKIP_MS = 200  # descarta la cola de la propia palabra de activación para que el VAD no la tome como frase


@dataclass
class Turn:
    """Resultado y métricas (segundos) de un turno de conversación."""
    question: str = ""             # lo que dijo/escribió el usuario, tal cual
    retrieval_query: str = ""      # pregunta(s) usada(s) para RAG, para logging (ver subquestions)
    subquestions: list = field(default_factory=list)  # 1 elemento salvo que se haya descompuesto
    was_rewritten: bool = False
    answer: str = ""
    context_hits: list = field(default_factory=list)
    retrieval_trace: list = field(default_factory=list)  # 1 dict por sub-pregunta RAG, ver Assistant.answer
    canonical_matches: list = field(default_factory=list)  # 1 dict por sub-pregunta resuelta por canonical.py
    was_repeat: bool = False  # True si este turno fue un pedido de "repetí" (ver guardrails.is_repeat_request)
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
        self.canonical = CanonicalMatcher(cfg.canonical) if cfg.canonical.enabled else None
        log(f"Canonical: {len(self.canonical.entries) if self.canonical else 0} entradas")
        t = time.monotonic()
        self.llm = LocalLLM(cfg.llm, cfg.rewrite)
        log(f"LLM listo ({time.monotonic() - t:.1f}s)")
        self.tts = PiperTTS(cfg.tts)
        self.speaker = Speaker(cfg.audio.speaker_name) if speak else None
        self.vad = UtteranceRecorder(cfg.vad, cfg.audio.sample_rate)
        self.wake = WakeWord(cfg.wakeword) if cfg.wakeword.enabled else None
        self.mic = Mic(cfg.audio.sample_rate, cfg.audio.mic_name)
        self._last_answer: str | None = None  # para "repetí" (ver guardrails.is_repeat_request) -- cualquier respuesta, canónica o no

    def reset_conversation(self) -> None:
        """Arranca una charla nueva: limpia el historial que usa voice/rewrite.py, la rotación de
        variantes canónicas y el "repetí" (usado por scripts/eval.py entre casos de prueba, para
        que uno no contamine al siguiente -- y por Assistant.run() entre ventanas de conversación)."""
        self.llm.reset()
        if self.canonical:
            self.canonical.reset()
        self._last_answer = None

    # ---- un turno: audio -> texto -> respuesta hablada -------------------------------------
    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        t = time.monotonic()
        text = self.stt.transcribe(audio)
        return text, time.monotonic() - t

    def _fixed_reply(self, question: str, answer_text: str, turn: Turn, t0: float) -> Turn:
        """Corta acá sin llamar al LLM de respuesta (ver voice/guardrails.py): usado tanto para la
        abstención (sin CONTEXTO) como para el gate de ambigüedad cross-doc (dos documentos con
        evidencia comparable). Sin esto, "responder como asistente general" era justo lo que llevaba
        a inventar identidades (bug 'Juan Carlos es el Rey de España', ver README)."""
        out = StreamingSpeaker(self.tts, self.speaker)
        print(f"🤖 {answer_text}", flush=True)
        out.say(answer_text)
        out.finish()
        self.llm.record_turn(question, answer_text)
        self._last_answer = answer_text
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

        # "repetí" / "¿cómo?" / "no te escuché": repite LA ÚLTIMA respuesta (canónica o no) tal
        # cual -- chequeo determinista, corta ANTES de la reescritura/descomposición/canonical/RAG,
        # nada de eso aplica acá. Sin última respuesta (nada que repetir), sigue el flujo normal.
        if self._last_answer is not None and is_repeat_request(question):
            turn.was_repeat = True
            return self._fixed_reply(question, self._last_answer, turn, t0)

        t = time.monotonic()
        subquestions, was_rewritten = self.llm.rewrite_query(question)
        turn.t_rewrite = time.monotonic() - t
        turn.subquestions = subquestions
        turn.retrieval_query = subquestions[0] if len(subquestions) == 1 else " / ".join(subquestions)
        turn.was_rewritten = was_rewritten
        if was_rewritten and len(subquestions) > 1:
            print(f"   (descompuesta en {len(subquestions)} sub-preguntas: {subquestions!r})", flush=True)
        elif was_rewritten:
            print(f"   (reescrita: {subquestions[0]!r})", flush=True)

        # canonical (FAQ, texto fijo) -> RAG -> LLM, en ese orden de prioridad, independiente por
        # sub-pregunta (ver punto 4 y voice/canonical.py). Una sub-pregunta que matchea una entrada
        # canónica NUNCA llega al RAG ni al LLM -- su texto va directo a la respuesta final.
        canonical_parts: list[str] = []   # texto canónico, YA elegido (variante rotada) -- va tal cual
        resolved: list[tuple[str, str | None]] = []  # (sub-pregunta, contexto o None) -> al LLM
        unresolved_notes: list[str] = []               # notas FIJAS (no generadas) para el resto
        all_hits = []
        for subq in subquestions:
            if self.canonical:
                match = self.canonical.match(subq)
                if match:
                    canonical_parts.append(match.text)
                    turn.canonical_matches.append({
                        "subq": subq, "entry_id": match.entry_id, "score": match.score,
                        "close_second": match.close_second,
                    })
                    if match.close_second:
                        print(
                            f"   ⚠ canonical: {subq!r} matcheó '{match.entry_id}'@{match.score:.2f}, "
                            f"pero '{match.close_second[0]}'@{match.close_second[1]:.2f} está muy "
                            f"cerca -- revisar solapamiento de formulaciones",
                            flush=True,
                        )
                    continue

            t = time.monotonic()
            hits = self.rag.retrieve(subq) if self.rag else []
            turn.t_rag += time.monotonic() - t
            all_hits += hits

            if self.rag:
                # traza de retrieval: distingue "no se recuperó nada" (ni un candidato) de "se
                # recuperó con score bajo" (había un top-1, no cruzó min_score) -- antes ambos casos
                # se veían igual en el log ("sin contexto"), sin poder saber cuál pasó (ver README).
                top1 = hits[0] if hits else None
                if top1 is None:
                    cands = self.rag.score_candidates(subq)  # solo si no hubo hits: recalcula el
                    top1 = cands[0] if cands else None       # top-1 sin filtrar, nada más para el log
                turn.retrieval_trace.append({
                    "subq": subq,
                    "top1_source": top1.source if top1 else None,
                    "top1_score": top1.score if top1 else None,
                    "min_score": self.cfg.rag.min_score,
                    "had_hits": bool(hits),
                })

            if self.rag and not hits and not is_chitchat(subq):
                if len(subquestions) > 1:
                    unresolved_notes.append(abstain_partial_reply(subq))
                else:
                    unresolved_notes.append(abstain_reply(subq, has_history))
                continue

            if self.rag and hits:
                ambiguous = ambiguous_docs(hits, self.cfg.rag.ambiguity_threshold)
                if ambiguous:
                    unresolved_notes.append(clarify_reply(*ambiguous, self.cfg.rag.doc_topics))
                    continue

            context = Retriever.format_context(hits) if hits else None
            resolved.append((subq, context))

        turn.context_hits = all_hits

        if not resolved:
            # ninguna sub-pregunta necesitó al LLM de respuesta: o hay canónicas (texto fijo) y/o
            # notas fijas de abstención/ambigüedad, o ambas -- ninguna pasa por el LLM. Este es
            # también el camino "puro canonical" (match sin necesitar RAG en absoluto): latencia
            # mínima, sin decodificación de tokens (ver voice/canonical.py).
            answer_text = " ".join([*canonical_parts, *unresolved_notes]).strip()
            return self._fixed_reply(question, answer_text, turn, t0)

        out = StreamingSpeaker(self.tts, self.speaker)
        t_llm = time.monotonic()
        first = None
        tokens = []

        print("🤖 ", end="", flush=True)

        # las partes canónicas (texto fijo) se dicen PRIMERO, antes de que el LLM empiece a
        # generar -- son instantáneas (no hay decodificación) y el LLM no las toca en absoluto.
        for part in canonical_parts:
            print(part, end=" ", flush=True)
            out.say(part)

        def counted():
            nonlocal first
            # el LLM de respuesta recibe las sub-preguntas YA reescritas/autónomas, sin el
            # historial de la charla (ver voice/llm.py: stream_answer(_multi) es stateless a propósito)
            for tok in self.llm.stream_answer_multi(resolved):
                if first is None:
                    first = time.monotonic()
                tokens.append(tok)
                yield tok

        # en modo multi-parte (>1 sub-pregunta resuelta), el LLM etiqueta cada línea "Parte N: "
        # como andamiaje para no ignorar ninguna parte (ver rewrite.MULTI_PART_SUFFIX) -- eso no es
        # para el usuario, se saca acá antes de hablar/mostrar/guardar la respuesta.
        multi_part = len(resolved) > 1
        spoken: list[str] = []
        for sentence in iter_sentences(counted()):
            if multi_part:
                sentence = strip_part_labels(sentence)
                if not sentence:
                    continue
            print(sentence, end=" ", flush=True)
            out.say(sentence)
            spoken.append(sentence)
        turn.t_llm_total = time.monotonic() - t_llm
        turn.t_llm_first_token = (first or time.monotonic()) - t_llm
        turn.n_tokens = len(tokens)
        generated = " ".join(spoken).strip() if multi_part else "".join(tokens).strip()

        # las sub-preguntas que abstuvieron/dieron ambiguo se agregan al final, como notas fijas
        # (no generadas) -- no pasaron por el LLM en absoluto (ver punto 4 del README)
        for note in unresolved_notes:
            print(note, end=" ", flush=True)
            out.say(note)
        print(flush=True)
        turn.answer = " ".join([*canonical_parts, generated, *unresolved_notes]).strip()

        out.finish()
        # se guarda la pregunta ORIGINAL (no la reescrita): así la próxima reescritura ve la charla
        # tal como pasó de verdad, no una versión ya "limpiada" de sí misma.
        self.llm.record_turn(question, turn.answer)
        self._last_answer = turn.answer
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
        for cm in t.canonical_matches:
            print(f"   📌 {cm['subq']!r}: canonical:{cm['entry_id']}@{cm['score']:.2f}", flush=True)
        for tr in t.retrieval_trace:
            if tr["top1_source"] is None:
                print(f"   🔎 {tr['subq']!r}: sin ningún candidato (corpus vacío o embedding sin match)", flush=True)
            else:
                status = "✓ recuperado" if tr["had_hits"] else "✗ NO cruzó min_score"
                print(
                    f"   🔎 {tr['subq']!r}: top1={tr['top1_source']}@{tr['top1_score']:.2f} "
                    f"(min_score={tr['min_score']:.2f}) {status}",
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
            # termina la ventana de conversación: limpiar antes de la próxima (ver LocalLLM.reset)
            # para que una charla nueva no arrastre contexto de una completamente distinta.
            # BUG real (23/09): este print antes solo salía si había wake word configurado -- sin
            # wake word (--no-wake), el reset por timeout de followup_s pasaba en silencio, sin
            # ninguna señal visible; en una charla real esto se veía como "se perdió el historial
            # sin razón" (turno con rewrite=0ms, respuesta de abstención "sin historial" de la
            # nada). Ahora se avisa siempre, mencione o no el wake word.
            self.reset_conversation()
            msg = f"   (esperando «{wake_word}»…)\n" if self.wake else "   (charla reiniciada por inactividad -- nueva pregunta arranca sin historial)\n"
            print(msg, end="", flush=True)
