"""Reglas deterministas (no generadas por el LLM) para dos fallas encontradas con scripts/eval.py
(ver README): inventar identidades/hechos cuando no hay CONTEXTO de RAG, y fugas de idioma (el 3B
a veces cambia a chino a mitad de una respuesta, sin relación con el RAG)."""
import re

CJK_RE = re.compile(
    "[\U00003000-\U0000303F\U00003040-\U0000309F\U000030A0-\U000030FF"
    "\U00004E00-\U00009FFF\U0000AC00-\U0000D7A3\U0000FF00-\U0000FFEF]"
)

# Preguntas/comentarios sin riesgo real de "inventar un hecho de terceros": charla social, o sobre
# el propio asistente. Estas SÍ se dejan pasar al LLM aunque no haya CONTEXTO de RAG. Lista chica y
# ad-hoc (los patrones que hicieron falta hasta ahora en tests/eval_questions.yaml); sumar más acá
# a medida que aparezcan casos legítimos que el abstain esté bloqueando de más.
_CHITCHAT_PATTERNS = [
    r"\bhola\b", r"\bbuen[oa]s?\s+(d[ií]as?|tardes|noches)\b", r"\bqu[eé]\s+tal\b",
    r"\bchau\b", r"\badi[oó]s\b", r"\bnos\s+vemos\b", r"\bhasta\s+(luego|pronto)\b",
    r"\bgracias\b",
    r"\bchiste\b", r"\bbroma\b", r"\balgo\s+gracioso\b",
    r"\bc[oó]mo\s+est[aá]s\b", r"\bc[oó]mo\s+te\s+llam[aá]s\b", r"\bqui[eé]n\s+sos\b",
    r"\bpor\s+qu[eé]\s+(demoras|tardas)\b",       # meta-preguntas sobre el propio asistente
    r"\b(sos|eres)\s+un[ao]?\s+\w+\b",            # venteo/insultos dirigidos al asistente
]
_CHITCHAT_RE = re.compile("|".join(_CHITCHAT_PATTERNS), re.IGNORECASE)

_PERSON_RE = re.compile(r"\bqui[eé]n\s+es\b", re.IGNORECASE)


def is_chitchat(question: str) -> bool:
    """Charla social o sobre el propio asistente: se deja pasar al LLM aunque no haya CONTEXTO."""
    return bool(_CHITCHAT_RE.search(question))


def abstain_reply(question: str) -> str:
    """Respuesta fija para cuando no hay CONTEXTO y la pregunta no es charla social.

    Antes de esto, el system_prompt le decía al LLM "si no hay CONTEXTO, respondé como asistente
    general" -- y eso es justo lo que lo llevaba a inventar que "Juan Carlos" (sin ningún documento
    que lo mencione) es el Rey de España: el nombre coincide con una persona real y famosa, y el
    modelo la trae de su conocimiento general aunque no tenga nada que ver con este asistente de
    oficina. Cortar acá, sin llamar al LLM, elimina el riesgo por completo para este camino."""
    if _PERSON_RE.search(question):
        return "No tengo información sobre esa persona."
    return "No tengo información sobre eso."


def cjk_token_bias(llm, value: float = -100.0) -> dict[int, float]:
    """logit_bias que castiga cada token del vocabulario que contenga caracteres CJK, para que el
    modelo prácticamente nunca los elija -- prevenir en el muestreo en vez de detectar y reintentar
    después. ~0.3s sobre un vocab de 150k tokens, se calcula una sola vez al cargar el modelo."""
    bias = {}
    for tid in range(llm.n_vocab()):
        try:
            s = llm.detokenize([tid]).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if CJK_RE.search(s):
            bias[tid] = value
    return bias
