"""Capa de respuestas canónicas (FAQ con texto fijo), evaluada ANTES del RAG -- ver README y
tests/canonical_answers.yaml.

Por qué: ciertas preguntas frecuentes se responden mejor con un texto fijo, redactado por la
empresa, que con algo generado por el LLM -- cero riesgo de alucinación en las preguntas más
comunes, control total del mensaje, y latencia mínima (no hay decodificación de tokens, ver
voice/pipeline.py: Assistant.answer). Convive con el RAG: si una (sub)pregunta matchea una entrada
canónica (por score de reranker sobre sus formulaciones), se responde con su texto tal cual; si no
matchea nada, sigue el pipeline normal (RAG -> LLM).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import ROOT, CanonicalCfg, resolve


class CanonicalValidationError(Exception):
    """El archivo de contenido no valida -- el mensaje nombra la entrada y el problema, pensado
    para que alguien no técnico (quien edita el contenido) entienda qué corregir."""


@dataclass
class CanonicalEntry:
    id: str
    formulaciones: list[str]
    respuestas: list[str]


# ============================================================================
# Lint para voz (warnings, NO bloquean la carga) -- el texto de cada respuesta lo lee Piper en voz
# alta, así que lo que es válido para texto escrito puede sonar mal hablado.
# ============================================================================

_DIGIT_RE = re.compile(r"\d")
_TIME_RANGE_RE = re.compile(r"\d+\s*[-–]\s*\d+|\bhs\.?\b", re.IGNORECASE)
_ABBREV_RE = re.compile(r"\b(av|ud|uds|sr|sra|dr|dra|etc|nro|n°|c/u|ej|pág|tel)\.", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"^\s*[-*•]\s|\*\*[^*]+\*\*|__[^_]+__|^\s*\d+\.\s", re.MULTILINE)
_MAX_WORDS_RESPONSE = 80
_LENGTH_VARIANCE_RATIO = 2.0  # si la variante más larga tiene más del doble de palabras que la más corta


def _lint_response(entry_id: str, idx: int, text: str) -> list[str]:
    label = f"{entry_id} (respuesta {idx + 1})"
    warnings = []
    if _DIGIT_RE.search(text):
        warnings.append(f"{label}: tiene dígitos -- Piper lee mejor los números escritos en palabras ('diez' en vez de '10')")
    if _TIME_RANGE_RE.search(text):
        warnings.append(f"{label}: parece tener un formato tipo '9-13hs' -- escribilo como se pronuncia ('de nueve a trece horas')")
    if _ABBREV_RE.search(text):
        warnings.append(f"{label}: tiene una abreviatura -- escribila completa, Piper no las expande (leería 'av' o 'sr' tal cual)")
    if _MARKDOWN_RE.search(text):
        warnings.append(f"{label}: tiene viñetas o markdown -- esto se lee en voz alta, no se muestra como texto")
    n_words = len(text.split())
    if n_words > _MAX_WORDS_RESPONSE:
        warnings.append(f"{label}: tiene {n_words} palabras (más de {_MAX_WORDS_RESPONSE}) -- una respuesta hablada debería ser más corta")
    return warnings


def lint_entry(entry: CanonicalEntry) -> list[str]:
    """Warnings de estilo para las respuestas de una entrada -- se imprimen al cargar
    (CanonicalMatcher.__init__), no impiden que la entrada se use."""
    warnings: list[str] = []
    for i, r in enumerate(entry.respuestas):
        warnings += _lint_response(entry.id, i, r)
    lens = [len(r.split()) for r in entry.respuestas]
    if len(lens) > 1 and max(lens) / max(min(lens), 1) > _LENGTH_VARIANCE_RATIO:
        warnings.append(
            f"{entry.id}: las variantes de respuesta tienen largos muy distintos entre sí "
            f"({lens} palabras) -- probablemente no dicen lo mismo, revisar"
        )
    return warnings


# ============================================================================
# Carga + validación (errores fatales -- el archivo no se usa hasta que se corrigen)
# ============================================================================


def _normalize_formulacion(s: str) -> str:
    """Para detectar duplicados ENTRE entradas: minúsculas, sin acentos, sin signos de puntuación
    de pregunta -- dos formulaciones que solo difieren en mayúsculas/tildes/"¿...?" cuentan como la
    misma frase."""
    s = unicodedata.normalize("NFKD", s.strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[¿?¡!.,]", "", s).strip()


def load_canonical_entries(path: Path) -> list[CanonicalEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise CanonicalValidationError(f"{path}: el archivo debe ser una lista de entradas")

    entries: list[CanonicalEntry] = []
    seen_ids: set[str] = set()
    seen_formulaciones: dict[str, str] = {}  # formulación normalizada -> id que la usó primero

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CanonicalValidationError(f"entrada #{i + 1}: no es un objeto válido")

        entry_id = item.get("id")
        if not entry_id or not isinstance(entry_id, str):
            raise CanonicalValidationError(f"entrada #{i + 1}: falta 'id' o no es texto")
        if entry_id in seen_ids:
            raise CanonicalValidationError(f"'{entry_id}': id duplicado -- cada entrada necesita un id único")
        seen_ids.add(entry_id)

        formulaciones = item.get("formulaciones")
        if not formulaciones or not isinstance(formulaciones, list):
            raise CanonicalValidationError(f"'{entry_id}': falta 'formulaciones' o está vacío")
        for f in formulaciones:
            if not isinstance(f, str) or not f.strip():
                raise CanonicalValidationError(f"'{entry_id}': hay una formulación vacía o inválida")

        respuestas = item.get("respuestas")
        if not respuestas or not isinstance(respuestas, list):
            raise CanonicalValidationError(f"'{entry_id}': falta 'respuestas' o está vacío")
        for r in respuestas:
            if not isinstance(r, str) or not r.strip():
                raise CanonicalValidationError(f"'{entry_id}': hay una respuesta vacía o inválida")

        for f in formulaciones:
            norm = _normalize_formulacion(f)
            prev_owner = seen_formulaciones.get(norm)
            if prev_owner and prev_owner != entry_id:
                raise CanonicalValidationError(
                    f"'{entry_id}': la formulación {f!r} es prácticamente igual a una ya usada en "
                    f"la entrada '{prev_owner}' -- una misma frase no puede matchear dos entradas a "
                    f"la vez, hay que sacarla de una de las dos"
                )
            seen_formulaciones[norm] = entry_id

        entries.append(CanonicalEntry(id=entry_id, formulaciones=list(formulaciones), respuestas=list(respuestas)))

    return entries


# ============================================================================
# Matching (reranker contra las formulaciones) + rotación de variantes de respuesta
# ============================================================================


@dataclass
class CanonicalMatch:
    entry_id: str
    score: float
    text: str
    close_second: tuple[str, float] | None = None  # (id, score) de otra entrada casi empatada


class CanonicalMatcher:
    def __init__(self, cfg: CanonicalCfg):
        self.cfg = cfg
        self.entries: dict[str, CanonicalEntry] = {}
        # listas paralelas: todas las formulaciones de todas las entradas, precalculadas UNA vez
        # al cargar (no por consulta) -- ver match().
        self._flat_formulaciones: list[str] = []
        self._flat_entry_ids: list[str] = []
        self._variant_idx: dict[str, int] = {}  # estado de rotación -- vive en la conversación, ver reset()
        self._reranker = None

        if not cfg.enabled:
            return

        for entry in load_canonical_entries(resolve(cfg.path)):
            self.entries[entry.id] = entry
            for w in lint_entry(entry):
                print(f"[canonical] ⚠ {w}", flush=True)
            for f in entry.formulaciones:
                self._flat_formulaciones.append(f)
                self._flat_entry_ids.append(entry.id)

        if self.entries:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(cfg.reranker_model, cache_dir=str(ROOT / "models" / "reranker"))

    def reset(self) -> None:
        """Arranca una charla nueva: la rotación de variantes vuelve a empezar por la primera de
        cada entrada (ver voice/pipeline.py: Assistant.reset_conversation)."""
        self._variant_idx.clear()

    def match(self, query: str) -> CanonicalMatch | None:
        """Busca la entrada cuya mejor formulación matchea `query` por score de reranker. None si
        el mejor score no supera cfg.threshold (sigue el pipeline normal: RAG). OJO: llamar a esto
        AVANZA el estado de rotación de la entrada que matchea -- no llamarlo dos veces para la
        misma pregunta si ya se va a usar el resultado."""
        if not self._reranker or not self._flat_formulaciones:
            return None

        scores = list(self._reranker.rerank(query, self._flat_formulaciones))
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]
        if best_score < self.cfg.threshold:
            return None
        entry_id = self._flat_entry_ids[best_idx]

        # dos entradas DISTINTAS muy cerca, ambas sobre el umbral -- se toma la mejor igual (no se
        # inventa un gate de ambigüedad nuevo acá), pero se loguea para poder revisar si hace falta
        # separar mejor las formulaciones de esas dos entradas (ver README).
        close_second: tuple[str, float] | None = None
        for i, s in enumerate(scores):
            other_id = self._flat_entry_ids[i]
            if other_id != entry_id and s >= self.cfg.threshold and (close_second is None or s > close_second[1]):
                close_second = (other_id, s)

        return CanonicalMatch(
            entry_id=entry_id, score=best_score, text=self._next_variant(entry_id), close_second=close_second
        )

    def best_score(self, query: str) -> tuple[str, float] | None:
        """Como match() pero SIN filtrar por cfg.threshold y SIN avanzar la rotación de variantes
        -- para calibración (ver scripts/calibrate.py, que necesita barrer distintos umbrales
        sobre el mismo score sin mutar estado). Devuelve (entry_id, score) del mejor candidato,
        cualquiera sea el score."""
        if not self._reranker or not self._flat_formulaciones:
            return None
        scores = list(self._reranker.rerank(query, self._flat_formulaciones))
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return self._flat_entry_ids[best_idx], scores[best_idx]

    def _next_variant(self, entry_id: str) -> str:
        """Conversación nueva (self._variant_idx recién reseteado) -> variante 0 (la principal).
        Si la misma entrada vuelve a matchear en la misma conversación -> la siguiente no usada;
        al agotarse, vuelve a la primera (módulo). Determinista, nada de random."""
        respuestas = self.entries[entry_id].respuestas
        idx = self._variant_idx.get(entry_id, 0) % len(respuestas)
        self._variant_idx[entry_id] = idx + 1
        return respuestas[idx]
