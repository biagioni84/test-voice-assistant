"""LLM local 7-8B con llama.cpp (CPU; el modelo Q4 no entra en 4GB de VRAM)."""
from __future__ import annotations

from typing import Iterator

from llama_cpp import Llama

from .config import LlmCfg, resolve


class LocalLLM:
    def __init__(self, cfg: LlmCfg):
        self.cfg = cfg
        self.llm = Llama(
            model_path=str(resolve("models/llm") / cfg.file),
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
        )
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history.clear()

    def stream_reply(self, question: str, context: str | None = None) -> Iterator[str]:
        """Genera tokens en streaming. El contexto RAG va solo en este turno, no en el historial."""
        user = f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}" if context else question
        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            *self.history[-2 * self.cfg.history_turns :],
            {"role": "user", "content": user},
        ]
        reply = []
        try:
            for part in self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                stream=True,
            ):
                tok = part["choices"][0]["delta"].get("content")
                if tok:
                    reply.append(tok)
                    yield tok
        finally:
            self.history += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "".join(reply)},
            ]
