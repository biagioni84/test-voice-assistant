"""LLM local con llama.cpp (GPU vía offload parcial, ver README). Dos roles distintos sobre el mismo
modelo cargado una vez:
  - rewrite_query(): reescribe follow-ups en preguntas autónomas usando el historial (voice/rewrite.py)
  - stream_answer(): genera la respuesta final -- stateless, NO ve el historial de la charla (ver
    voice/pipeline.py: quien orquesta decide qué se guarda en self.history con record_turn())
"""
from __future__ import annotations

from typing import Iterator

from llama_cpp import Llama

from . import rewrite
from .config import LlmCfg, RewriteCfg, resolve
from .guardrails import cjk_token_bias


class LocalLLM:
    def __init__(self, cfg: LlmCfg, rewrite_cfg: RewriteCfg):
        self.cfg = cfg
        self.rewrite_cfg = rewrite_cfg
        self.llm = Llama(
            model_path=str(resolve("models/llm") / cfg.file),
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
        )
        self.history: list[dict] = []  # lo llena record_turn(); lo lee rewrite_query()
        # penaliza tokens con caracteres CJK para prevenir la fuga de idioma (ver voice/guardrails.py
        # y README) en vez de detectarla después y reintentar. Se usa en ambos roles.
        self._cjk_bias = cjk_token_bias(self.llm)

    def reset(self) -> None:
        self.history.clear()

    def record_turn(self, question: str, answer: str) -> None:
        """Guarda la pregunta ORIGINAL del usuario (no la reescrita) y la respuesta -- así el
        próximo rewrite_query() ve la charla tal como pasó de verdad."""
        self.history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

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

        messages = [
            {"role": "system", "content": rewrite.SYSTEM_PROMPT},
            *rewrite.few_shot_messages(),
            {"role": "user", "content": rewrite.format_input(pairs, question)},
        ]
        resp = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.rewrite_cfg.max_tokens,
            temperature=0.0,
            logit_bias=self._cjk_bias,
            stream=False,
        )
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
        for part in self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            logit_bias=self._cjk_bias,
            stream=True,
        ):
            tok = part["choices"][0]["delta"].get("content")
            if tok:
                yield tok
