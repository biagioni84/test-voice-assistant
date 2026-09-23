"""RAG: chunking por párrafos + embeddings multilingües (fastembed/ONNX) + coseno para el primer
filtro, más un reranker cross-encoder sobre los candidatos (ver Retriever.retrieve).

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
    score: float           # score final que usan el gate de ambigüedad y la abstención: el del
                            # reranker si está activo (logit crudo, no 0-1), si no el coseno
    source: str
    text: str
    dense_score: float = 0.0  # coseno original, para debug (--show-chunk-text)


def _extract_title(text: str) -> tuple[str, str]:
    """Si la primera línea es un heading markdown ("# Título (lo que sea)"), la separa del resto y
    devuelve (título corto, texto sin el heading). Si no hay heading, título vacío."""
    first, _, rest = text.partition("\n")
    if first.startswith("# "):
        title = first[2:].split("(")[0].strip()
        return title, rest.strip()
    return "", text


def _chunk(text: str, max_chars: int) -> list[str]:
    """Un chunk por párrafo (separado por línea en blanco) -- a propósito NO se combinan párrafos
    distintos aunque entren juntos en max_chars, para no mezclar hechos sin relación en un mismo
    chunk (p.ej. horario de oficina + wifi, ver README). Un párrafo que por sí solo supere max_chars
    se corta por oraciones."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paras:
        if len(p) <= max_chars:
            chunks.append(p)
            continue
        pieces, buf = [], ""
        for sent in p.replace("\n", " ").split(". "):
            if buf and len(buf) + len(sent) > max_chars:
                pieces.append(buf.strip())
                buf = ""
            buf += sent + ". "
        if buf.strip():
            pieces.append(buf.strip())
        chunks.extend(pieces)
    return chunks


class Retriever:
    def __init__(self, cfg: RagCfg):
        self.cfg = cfg
        from fastembed import TextEmbedding

        self._emb = TextEmbedding(cfg.embed_model, cache_dir=str(ROOT / "models" / "embeddings"))
        self.chunks: list[tuple[str, str]] = []  # (fuente, texto)
        self._matrix = np.zeros((0, 1), dtype=np.float32)
        self._build()

        self._reranker = None
        if cfg.reranker_enabled:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(
                cfg.reranker_model, cache_dir=str(ROOT / "models" / "reranker")
            )

    def _embed(self, texts: list[str]) -> np.ndarray:
        v = np.array(list(self._emb.embed(texts)), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def _build(self) -> None:
        docs_dir = resolve(self.cfg.docs_dir)
        files = sorted(p for p in docs_dir.glob("**/*") if p.suffix in (".md", ".txt"))
        for f in files:
            title, body = _extract_title(f.read_text(encoding="utf-8"))
            for c in _chunk(body, self.cfg.chunk_chars):
                text = f"{title}: {c}" if title else c
                self.chunks.append((f.relative_to(docs_dir).as_posix(), text))
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
        """Primer filtro por coseno (barato, sobre todos los chunks) para traer
        `reranker_candidates` candidatos; si hay reranker, los reordena con el cross-encoder (juicio
        de relevancia semántica más fino que el coseno, ver README) y ESE score es el que se filtra
        contra `min_score` y el que usan el gate de ambigüedad y la abstención. Sin reranker, se
        queda con el coseno tal como antes."""
        if not self.chunks:
            return []
        dense = self._matrix @ self._embed([query])[0]
        n = self.cfg.reranker_candidates if self._reranker else self.cfg.top_k
        cand_idx = np.argsort(-dense)[:n]

        if self._reranker:
            texts = [self.chunks[i][1] for i in cand_idx]
            rerank_scores = list(self._reranker.rerank(query, texts))
            order = sorted(range(len(cand_idx)), key=lambda j: -rerank_scores[j])[: self.cfg.top_k]
            return [
                Hit(float(rerank_scores[j]), *self.chunks[cand_idx[j]], dense_score=float(dense[cand_idx[j]]))
                for j in order
                if rerank_scores[j] >= self.cfg.min_score
            ]

        return [
            Hit(float(dense[i]), *self.chunks[i], dense_score=float(dense[i]))
            for i in cand_idx
            if dense[i] >= self.cfg.min_score
        ]

    @staticmethod
    def format_context(hits: list[Hit]) -> str:
        return "\n---\n".join(h.text for h in hits)
