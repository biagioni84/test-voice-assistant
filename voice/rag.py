"""RAG mínimo: chunking por párrafos + embeddings multilingües (fastembed/ONNX) + coseno en numpy.

Para un POC con pocos documentos esto alcanza y evita una vector DB. Si el corpus crece a decenas de
miles de chunks, reemplazar `_matrix` por FAISS/sqlite-vec manteniendo la misma interfaz `retrieve()`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .config import ROOT, RagCfg, resolve


@dataclass
class Hit:
    score: float
    source: str
    text: str


def _chunk(text: str, max_chars: int) -> list[str]:
    """Junta párrafos consecutivos hasta max_chars; párrafos gigantes se cortan por oraciones."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(p) > max_chars:
            pieces, buf = [], ""
            for sent in p.replace("\n", " ").split(". "):
                if buf and len(buf) + len(sent) > max_chars:
                    pieces.append(buf)
                    buf = ""
                buf += sent + ". "
            pieces.append(buf.strip())
        else:
            pieces = [p]
        for piece in pieces:
            if cur and len(cur) + len(piece) > max_chars:
                chunks.append(cur)
                cur = ""
            cur = f"{cur}\n{piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


class Retriever:
    def __init__(self, cfg: RagCfg):
        self.cfg = cfg
        from fastembed import TextEmbedding

        self._emb = TextEmbedding(cfg.embed_model, cache_dir=str(ROOT / "models" / "embeddings"))
        self.chunks: list[tuple[str, str]] = []  # (fuente, texto)
        self._matrix = np.zeros((0, 1), dtype=np.float32)
        self._build()

    def _embed(self, texts: list[str]) -> np.ndarray:
        v = np.array(list(self._emb.embed(texts)), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def _build(self) -> None:
        docs_dir = resolve(self.cfg.docs_dir)
        files = sorted(p for p in docs_dir.glob("**/*") if p.suffix in (".md", ".txt"))
        for f in files:
            for c in _chunk(f.read_text(encoding="utf-8"), self.cfg.chunk_chars):
                self.chunks.append((f.relative_to(docs_dir).as_posix(), c))
        if not self.chunks:
            return

        # caché del índice: se recalcula solo si cambian los docs, el modelo o el chunking
        key = hashlib.sha256(
            json.dumps([self.cfg.embed_model, self.cfg.chunk_chars, self.chunks]).encode()
        ).hexdigest()[:16]
        cache = ROOT / ".cache" / f"rag_{key}.npy"
        if cache.exists():
            self._matrix = np.load(cache)
        else:
            self._matrix = self._embed([f"{src}: {txt}" for src, txt in self.chunks])
            cache.parent.mkdir(exist_ok=True)
            np.save(cache, self._matrix)

    def retrieve(self, query: str) -> list[Hit]:
        if not self.chunks:
            return []
        scores = self._matrix @ self._embed([query])[0]
        top = np.argsort(-scores)[: self.cfg.top_k]
        return [
            Hit(float(scores[i]), *self.chunks[i]) for i in top if scores[i] >= self.cfg.min_score
        ]

    @staticmethod
    def format_context(hits: list[Hit]) -> str:
        return "\n---\n".join(h.text for h in hits)
