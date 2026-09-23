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

    def _post(self, messages: list[dict], id_slot: int, max_tokens: int, temperature: float, stream: bool):
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
        return requests.post(f"{self.base_url}/v1/chat/completions", json=payload, stream=stream, timeout=60)

    def rewrite_query(self, question: str) -> tuple[str, bool]:
        """Reescribe `question` como pregunta autónoma usando self.history. No hace nada (pasa la
        pregunta tal cual) si no hay historial todavía -- el primer turno de la charla no paga esta
        llamada extra. Devuelve (pregunta_a_usar, se_reescribió)."""
        if not self.history or not self.rewrite_cfg.enabled:
            return question, False

        pairs: list[tuple[str, str]] = []
        hist = self.history[-2 * self.rewrite_cfg.history_turns :]
        for i in range(0, len(hist) - 1, 2):
            pairs.append((hist[i]["content"], hist[i + 1]["content"]))

        if rewrite.is_empty_reference(question):
            resolved = rewrite.resolve_empty_reference(pairs)
            if resolved:
                return resolved, True

        messages = [
            {"role": "system", "content": rewrite.SYSTEM_PROMPT},
            *rewrite.few_shot_messages(),
            {"role": "user", "content": rewrite.format_input(pairs, question)},
        ]
        resp = self._post(messages, SLOT_REWRITE, self.rewrite_cfg.max_tokens, 0.0, stream=False).json()
        out = resp["choices"][0]["message"]["content"].strip()
        if rewrite.looks_like_question(out):
            return out, True
        return question, False

    def stream_answer(self, question: str, context: str | None = None) -> Iterator[str]:
        """Genera la respuesta para `question` (ya debería venir autónoma/reescrita si corresponde)
        + `context` de este turno. Stateless: no lee ni actualiza self.history -- llamar a
        record_turn() aparte con lo que corresponda guardar."""
        user = f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}" if context else question
        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user},
        ]
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
