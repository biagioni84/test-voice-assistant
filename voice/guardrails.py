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

# "repetí" / "¿cómo?" / "no te escuché": pide que se repita LA ÚLTIMA respuesta dada (canónica o
# no) tal cual -- sin rotar variantes canónicas ni volver a generar con el LLM. Ver
# voice/pipeline.py: Assistant.answer (chequeo determinista, corta antes que nada más).
_REPEAT_PATTERNS = [
    r"\brepet[ií](\s|$)", r"\bpod[eé]s\s+repetir\b", r"\botra\s+vez\b",
    r"^[¿\s]*c[oó]mo\??[\s!]*$",          # "¿cómo?" solo, no "¿cómo es el wifi?"
    r"\bno\s+te\s+escuch[eé]\b", r"\bno\s+escuch[eé]\b", r"\bno\s+entend[ií]\b",
    r"\bqu[eé]\s+dijiste\b", r"\bperd[oó]n,?\s+qu[eé]\??\s*$",
]
_REPEAT_RE = re.compile("|".join(_REPEAT_PATTERNS), re.IGNORECASE)


def is_chitchat(question: str) -> bool:
    """Charla social o sobre el propio asistente: se deja pasar al LLM aunque no haya CONTEXTO."""
    return bool(_CHITCHAT_RE.search(question))


def is_repeat_request(question: str) -> bool:
    """"Repetí", "¿cómo?", "no te escuché" -- pide repetir la última respuesta, no una pregunta
    nueva. Sesgado a patrones cortos y específicos (a diferencia de is_chitchat) porque el costo de
    un falso positivo acá es más alto: interceptar una pregunta real como si fuera un pedido de
    repetición estaría mal, no solo caro en latencia."""
    return bool(_REPEAT_RE.search(question.strip()))


def abstain_reply(question: str, has_history: bool = False) -> str:
    """Respuesta fija para cuando no hay CONTEXTO y la pregunta no es charla social.

    Antes de esto, el system_prompt le decía al LLM "si no hay CONTEXTO, respondé como asistente
    general" -- y eso es justo lo que lo llevaba a inventar que "Juan Carlos" (sin ningún documento
    que lo mencione) es el Rey de España: el nombre coincide con una persona real y famosa, y el
    modelo la trae de su conocimiento general aunque no tenga nada que ver con este asistente de
    oficina. Cortar acá, sin llamar al LLM, elimina el riesgo por completo para este camino.

    Con historial, la pregunta ya pasó por voice/rewrite.py -- si aun así no encontró nada, puede
    ser que la reescritura no haya resuelto bien la referencia, así que se lo decimos al usuario en
    vez de la respuesta genérica (que sonaría rara después de varias vueltas de conversación)."""
    if has_history:
        return "No lo encontré, ¿me lo preguntás de otra forma?"
    if _PERSON_RE.search(question):
        return "No tengo información sobre esa persona."
    return "No tengo información sobre eso."


def abstain_partial_reply(subquestion: str) -> str:
    """Nota fija (NO generada) para UNA sub-pregunta sin CONTEXTO dentro de una pregunta compuesta
    (ver voice/llm.py: rewrite_query en modo descomposición). A diferencia de abstain_reply (que
    reemplaza TODA la respuesta), esto se concatena junto a la respuesta generada para las
    sub-preguntas que sí se resolvieron -- por eso nombra la sub-pregunta puntual, para que quede
    claro a cuál de las partes se refiere."""
    return f"No tengo información para responder esto: {subquestion}"


def ambiguous_docs(hits: list, threshold: float) -> tuple[str, str] | None:
    """Si los dos documentos DISTINTOS con mejor score (no necesariamente los hits en posición 1 y
    2 -- si el top-2 son del mismo doc, se ignoran entre sí y se compara contra el mejor de otro
    doc) están a menos de `threshold` de diferencia, devuelve (doc1, doc2) para pedir aclaración en
    vez de generar. Si no hay ambigüedad (o hay un solo doc representado), None.

    No cubre todos los casos de "el modelo mezcla mal el CONTEXTO" -- solo el patrón específico de
    dos documentos con evidencia comparable. Si un solo doc tiene un match débil-pero-por-encima-
    del-umbral y el modelo igual alucina con él, esto no lo agarra (ver README, caso
    retrieval_ambiguo_dos_docs variante 1)."""
    best_per_doc: dict[str, float] = {}
    for h in hits:
        best_per_doc[h.source] = max(best_per_doc.get(h.source, 0.0), h.score)
    ranked = sorted(best_per_doc.items(), key=lambda kv: -kv[1])
    if len(ranked) < 2:
        return None
    (doc1, s1), (doc2, s2) = ranked[0], ranked[1]
    if s1 - s2 < threshold:
        return doc1, doc2
    return None


def clarify_reply(doc1: str, doc2: str, doc_topics: dict[str, str]) -> str:
    """Pregunta de aclaración nombrando los dos temas en juego (ver ambiguous_docs)."""
    t1 = doc_topics.get(doc1, doc1)
    t2 = doc_topics.get(doc2, doc2)
    return f"Tu pregunta toca dos temas distintos: {t1} y {t2}. ¿Sobre cuál de los dos te referís?"


# Gramática GBNF para prevenir la fuga de idioma (Qwen a veces cambia a chino a mitad de una
# respuesta). Reemplaza a un intento anterior con logit_bias sobre ~31.000 tokens CJK del
# vocabulario: funcionaba con el modelo embebido, pero con llama-server el campo `logit_bias` del
# request no escala -- medido: pasar de 15.000 a 31.000 entradas cuadruplica la latencia (¬1000ms de
# overhead extra), casi seguro por una búsqueda no indexada en el servidor. Una gramática que
# restringe los caracteres válidos por posición no tiene ese problema (medido: mismo tok/s con o sin
# gramática) porque el compilador de gramáticas de llama.cpp arma un autómata, no una lista plana.
CJK_GRAMMAR = (
    r'root ::= [^　-〿぀-ゟ゠-ヿ'
    r'一-鿿가-힣＀-￯]*'
)
