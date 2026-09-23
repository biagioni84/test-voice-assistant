"""LLM local vía llama-server (subproceso HTTP, no bindings embebidos).

Por qué: con una sola instancia embebida de llama_cpp.Llama compartiendo un contexto entre el
reescritor y el LLM de respuesta, cada llamada reprocesaba el prefijo entero del prompt (~800ms) --
alternar entre los dos prompts (muy distintos entre sí) invalidaba el cacheo de prefijo automático de
llama.cpp en cada turno. La solución no era cachear "a mano" (`save_state`/`load_state` resultó tener
más overhead que lo que ahorra, ver README) sino no compartir un único contexto: llama-server con
`--parallel 2` da un slot de KV cache independiente por rol (0=reescritor, 1=respuesta), y con
`cache_prompt: true` + `id_slot` fijo por rol, cada uno mantiene su propio prefijo cacheado sin que el
otro lo toque. Medido: prefill del reescritor bajó de ~800ms a ~80-300ms según el turno.
"""
from __future__ import annotations

import atexit
import json
import subprocess
import time
from typing import Iterator

import requests

from . import rewrite
from .config import LlmCfg, RewriteCfg, resolve
from .guardrails import CJK_GRAMMAR

SLOT_REWRITE = 0
SLOT_ANSWER = 1


class LocalLLM:
    def __init__(self, cfg: LlmCfg, rewrite_cfg: RewriteCfg):
        self.cfg = cfg
        self.rewrite_cfg = rewrite_cfg
        self.base_url = f"http://{cfg.server_host}:{cfg.server_port}"
        self.history: list[dict] = []  # lo llena record_turn(); lo lee rewrite_query()

        self._proc = self._start_server()
        atexit.register(self.close)
        self._wait_ready()

    def _start_server(self) -> subprocess.Popen:
        cmd = [
            self.cfg.server_bin,
            "-m", str(resolve("models/llm") / self.cfg.file),
            "--host", self.cfg.server_host,
            "--port", str(self.cfg.server_port),
            "--parallel", "2",
            "-c", str(self.cfg.n_ctx),
            "-ngl", str(self.cfg.n_gpu_layers),
            "-t", str(self.cfg.n_threads),
            "--slot-save-path", "/tmp/llama-server-slots",  # necesario para habilitar /slots?action=erase
            "--no-webui",
        ]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _wait_ready(self, timeout: float = 90.0) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server terminó solo (exit code {self._proc.returncode}) -- revisar {self.cfg.server_bin}")
            try:
                if requests.get(f"{self.base_url}/health", timeout=1).ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.3)
        raise RuntimeError("llama-server no respondió a tiempo en /health")

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def reset(self) -> None:
        """Arranca una charla nueva: limpia self.history nada más.

        NO se borra el KV cache de los slots. Se había agregado un borrado acá por una sospecha de
        contaminación entre charlas (el bug de "y ahí" parecía reaparecer sin relación con el
        contenido cacheado) -- pero un experimento controlado (correr una charla B (a) desde cero,
        (b) después de otra charla A con cache_prompt=true, (c) igual pero cache_prompt=false) dio
        el mismo resultado exacto en los tres casos. El matching de prefijo de llama-server compara
        tokens exactos y solo reusa lo que coincide byte a byte; no hay mecanismo por el que pueda
        "mezclar" contenido de una charla con otra. La causa real de "y ahí" era otra (ver
        voice/rewrite.py: is_empty_reference) y quedó resuelta aparte. Dejar el cache intacto entre
        charlas es seguro y más rápido: el prefijo estático del reescritor queda tibio incluso para
        la primera charla nueva."""
        self.history.clear()

    def record_turn(self, question: str, answer: str) -> None:
        """Guarda la pregunta ORIGINAL del usuario (no la reescrita) y la respuesta -- así el
        próximo rewrite_query() ve la charla tal como pasó de verdad."""
        self.history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

    def _post(
        self,
        messages: list[dict],
        id_slot: int,
        max_tokens: int,
        temperature: float,
        stream: bool,
        stop: list[str] | None = None,
    ):
        payload = {
            "model": "local",
            "messages": messages,
            "id_slot": id_slot,
            "cache_prompt": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "grammar": CJK_GRAMMAR,  # ver voice/guardrails.py: gramática en vez de logit_bias grande
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        return requests.post(f"{self.base_url}/v1/chat/completions", json=payload, stream=stream, timeout=60)

    def _recent_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        hist = self.history[-2 * self.rewrite_cfg.history_turns :]
        for i in range(0, len(hist) - 1, 2):
            pairs.append((hist[i]["content"], hist[i + 1]["content"]))
        return pairs

    def rewrite_query(self, question: str) -> tuple[list[str], bool]:
        """Devuelve (sub_preguntas, se_reescribió_o_descompuso). Normalmente sub_preguntas trae un
        solo elemento (comportamiento de siempre); trae más de uno solo si voice/rewrite.py:
        looks_compound() disparó Y el reescritor devolvió una descomposición válida.

        A diferencia de la reescritura de seguimiento (que no tiene sentido sin historial: no hay
        nada que resolver), la detección de preguntas compuestas se evalúa en CUALQUIER turno,
        incluido el primero -- "decime el horario de lunes a viernes y también el de los sábados"
        puede ser la primera frase de la charla."""
        if not self.rewrite_cfg.enabled:
            return [question], False

        pairs = self._recent_pairs()

        if rewrite.looks_compound(question):
            subqs = self._decompose(question, pairs)
            if subqs is not None:
                return subqs, True
            # no validó (JSON roto, etc.) -- sigue el camino normal de abajo con la pregunta tal
            # cual, que todavía puede beneficiarse de la reescritura de seguimiento si hay historial

        if not self.history:
            return [question], False

        if rewrite.is_empty_reference(question):
            resolved = rewrite.resolve_empty_reference(pairs)
            if resolved:
                return [resolved], True

        messages = [
            {"role": "system", "content": rewrite.SYSTEM_PROMPT},
            *rewrite.few_shot_messages(),
            {"role": "user", "content": rewrite.format_input(pairs, question)},
        ]
        # stop=["?", "\n"]: corta la decodificación apenas termina la pregunta reescrita en vez de
        # esperar el token EOS del modelo -- medido con scripts/.../test_stop.py: ahorra ~1 token y
        # ~15-35ms en este caso (el modelo ya paraba casi enseguida solo), pero además actúa como
        # red de seguridad si algún día empieza a divagar/explicar en vez de cortar limpio. OJO:
        # llama-server NO incluye el string de corte en la respuesta (confirmado empíricamente) --
        # si cortó por "?" hay que reponerlo a mano o looks_like_question() lo rechaza siempre.
        resp = self._post(
            messages, SLOT_REWRITE, self.rewrite_cfg.max_tokens, 0.0, stream=False, stop=["?", "\n"]
        ).json()
        choice = resp["choices"][0]
        out = choice["message"]["content"].strip()
        if choice.get("finish_reason") == "stop" and out and not out.endswith("?") and "\n" not in out:
            # el endpoint OpenAI-compatible de llama-server no distingue CUÁL de los dos stops se
            # disparó (no expone un "stopping_word" como el endpoint nativo /completion) -- si el
            # texto no tiene "\n" y le falta el "?" final, la explicación más probable con mucha
            # diferencia es que cortó justo por "?" (el few-shot condiciona fuerte a una sola línea
            # corta terminada en "?"); reponerlo acá evita descartar reescrituras buenas solo por
            # esta ambigüedad del endpoint. looks_like_question() igual filtra por longitud después.
            out = out + "?"
        if rewrite.looks_like_question(out):
            return [out], True
        return [question], False

    def _decompose(self, question: str, pairs: list[tuple[str, str]]) -> list[str] | None:
        """Modo descomposición del reescritor: intenta partir `question` en 1-3 sub-preguntas
        autónomas (ver voice/rewrite.py: DECOMPOSE_SYSTEM_PROMPT). None si el output no valida
        (rewrite.parse_subquestions) -- el llamador cae al camino normal."""
        messages = [
            {"role": "system", "content": rewrite.DECOMPOSE_SYSTEM_PROMPT},
            *rewrite.decompose_few_shot_messages(),
            {"role": "user", "content": rewrite.format_input(pairs, question)},
        ]
        # más presupuesto que una reescritura simple: hasta 3 sub-preguntas en un array JSON
        resp = self._post(
            messages, SLOT_REWRITE, self.rewrite_cfg.max_tokens * 3, 0.0, stream=False
        ).json()
        out = resp["choices"][0]["message"]["content"].strip()
        return rewrite.parse_subquestions(out)

    def _stream_from_messages(self, messages: list[dict]) -> Iterator[str]:
        r = self._post(messages, SLOT_ANSWER, self.cfg.max_tokens, self.cfg.temperature, stream=True)
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8")
            if not s.startswith("data: "):
                continue
            payload = s[6:]
            if payload.strip() == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0]["delta"].get("content")
            if delta:
                yield delta

    def stream_answer(self, question: str, context: str | None = None) -> Iterator[str]:
        """Genera la respuesta para `question` (ya debería venir autónoma/reescrita si corresponde)
        + `context` de este turno. Stateless: no lee ni actualiza self.history -- llamar a
        record_turn() aparte con lo que corresponda guardar."""
        user = f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}" if context else question
        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user},
        ]
        yield from self._stream_from_messages(messages)

    def stream_answer_multi(self, subquestions: list[tuple[str, str | None]]) -> Iterator[str]:
        """Una sola llamada al LLM de respuesta cubriendo varias sub-preguntas YA resueltas (con
        CONTEXTO válido, o chitchat sin CONTEXTO) -- ver voice/pipeline.py: Assistant.answer(). Las
        sub-preguntas que abstuvieron o dieron ambiguo NO llegan acá: se resuelven aparte con una
        nota fija NO generada (voice/guardrails.py: abstain_partial_reply/clarify_reply) que se
        concatena a lo que este método devuelve.

        Con una sola sub-pregunta (el caso de siempre, sin descomposición) es exactamente
        stream_answer -- no hay cambio de comportamiento ni de prompt para preguntas simples."""
        if len(subquestions) == 1:
            q, ctx = subquestions[0]
            yield from self.stream_answer(q, ctx)
            return

        parts = []
        for i, (q, ctx) in enumerate(subquestions, 1):
            block = f"PARTE {i}\nPREGUNTA: {q}"
            if ctx:
                block += f"\nCONTEXTO:\n{ctx}"
            parts.append(block)
        messages = [
            {"role": "system", "content": self.cfg.system_prompt + rewrite.MULTI_PART_SUFFIX},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        yield from self._stream_from_messages(messages)
