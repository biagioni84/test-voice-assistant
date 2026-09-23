"""RAG: chunking por párrafos + retrieval híbrido (embeddings densos + BM25 léxico, fusionados con
RRF) para el primer filtro, más un reranker cross-encoder sobre los candidatos fusionados (ver
Retriever.retrieve). Para un POC con pocos documentos esto alcanza y evita una vector DB. Si el
corpus crece a decenas de miles de chunks, reemplazar `_matrix` por FAISS/sqlite-vec y el BM25 en
memoria por un índice invertido persistente, manteniendo la misma interfaz `retrieve()`.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from .config import ROOT, RagCfg, resolve

# Stopwords en español (sin acentos -- ver _tokenize, que saca acentos ANTES de comparar) para el
# lado BM25 del híbrido. Lista chica y estándar (artículos, preposiciones, pronombres, formas
# comunes de ser/estar/haber/tener) -- no es específica de ningún documento, es la misma para
# cualquier corpus en español.
_SPANISH_STOPWORDS = frozenset(
    """
    el la los las un una unos unas lo al del de a ante bajo con contra desde en entre hacia hasta
    para por segun sin so sobre tras y o u e ni que qué cual cuál cuales cuáles quien quién quienes
    quiénes como cómo cuando cuándo donde dónde adonde adónde cuanto cuánto cuanta cuánta cuantos
    cuántos cuantas cuántas es son fue fui fuiste fuimos fueron sera será seran serán ser estar esta
    está esto estos estas ese esa esos esas aquel aquella aquellos aquellas eso esas soy eres somos
    sois hay habia había habran habrán ha he has han muy mas más pero si sí no se su sus mi mis tu
    tus le les te me nos os vos yo el ella ellos ellas puedo puede podemos pueden tengo tiene
    tenemos tienen
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _tokenize(text: str) -> list[str]:
    """Tokenizado para BM25: minúsculas, sin acentos (para que "días"/"dias" matcheen igual), solo
    letras/números, sin stopwords. Simple a propósito -- alcanza para español y no depende de
    librerías de NLP pesadas."""
    text = _strip_accents(text.lower())
    return [t for t in _TOKEN_RE.findall(text) if t not in _SPANISH_STOPWORDS]


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

        self._bm25 = None
        if cfg.hybrid_enabled and self.chunks:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([_tokenize(f"{src}: {txt}") for src, txt in self.chunks])

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

    def _rrf_fuse(self, rankings: list[list[int]]) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion (Cormack et al. 2009): cada ranking es una lista de índices de
        chunk ya ordenada (mejor primero, típicamente un top-N truncado, no el corpus entero).
        score(doc) = suma, sobre los rankings donde aparece, de 1/(rrf_k + rank). Un doc que no
        aparece en un ranking simplemente no suma nada de ESE lado -- no se le imputa un rank peor
        artificialmente. `rrf_k` es la constante estándar del paper (config, no se calibra contra
        docs puntuales, ver voice/config.py). Devuelve (índice, score) ordenado descendente."""
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.cfg.rrf_k + rank + 1)
        return sorted(scores.items(), key=lambda kv: -kv[1])

    def _candidate_pool(self, query: str, pool_size: int) -> tuple[list[int], np.ndarray, dict[int, float] | None]:
        """Índices candidatos para `pool_size` resultados, más el vector denso completo (para
        `dense_score` de debug) y, si el híbrido está activo, el score RRF por índice (None si no).
        Sin híbrido (o sin BM25 construido, p.ej. corpus vacío), es el top-`pool_size` denso de
        siempre."""
        dense = self._matrix @ self._embed([query])[0]
        if self.cfg.hybrid_enabled and self._bm25 is not None:
            fusion_n = max(pool_size, self.cfg.bm25_candidates)
            dense_rank = [int(i) for i in np.argsort(-dense)[:fusion_n]]
            bm25_scores = self._bm25.get_scores(_tokenize(query))
            bm25_rank = [int(i) for i in np.argsort(-bm25_scores)[: self.cfg.bm25_candidates]]
            fused = self._rrf_fuse([dense_rank, bm25_rank])
            cand_idx = [idx for idx, _ in fused[:pool_size]]
            return cand_idx, dense, dict(fused)
        cand_idx = [int(i) for i in np.argsort(-dense)[:pool_size]]
        return cand_idx, dense, None

    def score_candidates(self, query: str) -> list[Hit]:
        """Primer filtro (denso, o híbrido denso+BM25 fusionado por RRF si `hybrid_enabled`) para
        traer `reranker_candidates` candidatos; si hay reranker, los reordena con el cross-encoder
        (juicio de relevancia semántica más fino, ver README). A diferencia de retrieve(), NO filtra
        por `min_score` ni corta a `top_k` -- devuelve TODOS los candidatos, ordenados por score
        final descendente. Sirve para evaluar distintos umbrales sin volver a llamar al modelo (ver
        scripts/calibrate.py); retrieve() es un filtro sobre esto mismo."""
        if not self.chunks:
            return []
        n = self.cfg.reranker_candidates if self._reranker else self.cfg.top_k
        cand_idx, dense, rrf_scores = self._candidate_pool(query, n)

        if self._reranker:
            texts = [self.chunks[i][1] for i in cand_idx]
            rerank_scores = list(self._reranker.rerank(query, texts))
            order = sorted(range(len(cand_idx)), key=lambda j: -rerank_scores[j])
            return [
                Hit(float(rerank_scores[j]), *self.chunks[cand_idx[j]], dense_score=float(dense[cand_idx[j]]))
                for j in order
            ]

        score_of = rrf_scores if rrf_scores is not None else {i: float(dense[i]) for i in cand_idx}
        order = sorted(cand_idx, key=lambda i: -score_of[i])
        return [Hit(float(score_of[i]), *self.chunks[i], dense_score=float(dense[i])) for i in order]

    def dense_top_n(self, query: str, n: int) -> list[Hit]:
        """Top-n SOLO por coseno, SIN BM25 ni reranker -- vista de diagnóstico para medir recall@n
        de la etapa puramente densa (ver scripts/calibrate.py). No es lo que usa
        retrieve()/score_candidates() en producción; acá `n` puede ser cualquier valor, para
        chequear si el primer filtro ya pierde el chunk correcto antes del reranker."""
        if not self.chunks:
            return []
        dense = self._matrix @ self._embed([query])[0]
        idx = np.argsort(-dense)[:n]
        return [Hit(float(dense[i]), *self.chunks[i], dense_score=float(dense[i])) for i in idx]

    def hybrid_top_n(self, query: str, n: int) -> list[Hit]:
        """Como dense_top_n pero fusionando con BM25 vía RRF (si `hybrid_enabled`) -- para comparar
        recall@n denso vs. híbrido en scripts/calibrate.py, ANTES del reranker. Si el híbrido está
        desactivado, es idéntico a dense_top_n."""
        if not self.chunks:
            return []
        if not (self.cfg.hybrid_enabled and self._bm25 is not None):
            return self.dense_top_n(query, n)
        cand_idx, dense, rrf_scores = self._candidate_pool(query, max(n, self.cfg.bm25_candidates))
        order = sorted(cand_idx, key=lambda i: -rrf_scores[i])[:n]
        return [Hit(float(rrf_scores[i]), *self.chunks[i], dense_score=float(dense[i])) for i in order]

    def retrieve(self, query: str) -> list[Hit]:
        """score_candidates() filtrado por `min_score` y cortado a `top_k` -- lo que usa el
        pipeline en producción. (Equivalente a "top_k primero, filtrar después": como los
        candidatos ya vienen ordenados descendente, da exactamente el mismo resultado que filtrar
        primero y cortar después.)"""
        hits = [h for h in self.score_candidates(query) if h.score >= self.cfg.min_score]
        return hits[: self.cfg.top_k]

    @staticmethod
    def format_context(hits: list[Hit]) -> str:
        return "\n---\n".join(h.text for h in hits)
