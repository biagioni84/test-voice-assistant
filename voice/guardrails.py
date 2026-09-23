"""Reglas deterministas (no generadas por el LLM) para varias fallas encontradas con
scripts/eval.py y en vivo (ver README): inventar identidades/hechos cuando no hay CONTEXTO de RAG,
fugas de idioma (el 3B a veces cambia a chino a mitad de una respuesta, sin relación con el RAG), y
mezclar/pedir aclaración de más entre documentos que en realidad no compiten de verdad."""
from __future__ import annotations

import re
from dataclasses import dataclass

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


def _best_per_doc(hits: list) -> dict[str, float]:
    """Mejor score por documento distinto. BUG real (24/09, ver README): usar 0.0 como default en
    max(best_per_doc.get(source, 0.0), score) hacía que un doc cuyo ÚNICO hit tuviera score
    NEGATIVO quedara "flotando" en 0.0 en vez de su score real -- eso achicaba artificialmente el
    gap contra el top-1 y disparaba el gate de ambigüedad de más ("¿La oficina abre los martes?":
    oficina.md@0.38 vs. recursos_humanos.md@-0.15 con un solo hit débil, gap real 0.53, pero el
    bug lo veía como 0.38 - 0.0 = 0.38, por debajo del umbral). float("-inf") como default deja el
    score real, sea cual sea el signo."""
    best_per_doc: dict[str, float] = {}
    for h in hits:
        best_per_doc[h.source] = max(best_per_doc.get(h.source, float("-inf")), h.score)
    return best_per_doc


@dataclass
class DocGate:
    """Resultado de gate_docs() -- a lo sumo uno de los dos campos no es None (o ninguno, caso
    "seguir normal"). Ver voice/pipeline.py: Assistant.answer()."""
    clarify: tuple[str, str] | None = None   # (doc1, doc2) -- pedir aclaración, no generar
    restrict_to: str | None = None            # nombre de doc -- alta confianza: usar SOLO sus chunks


def gate_docs(hits: list, min_score: float, confidence_margin: float, ambiguity_threshold: float) -> DocGate:
    """Rediseño (24/09, ver README) del viejo `ambiguous_docs`: que dos documentos tengan scores
    parecidos NO significa que sean conflictivos -- pueden ser complementarios (uno relevante, el
    otro apenas "no descartable"), y forzar una aclaración ahí molesta al usuario sin necesidad. Al
    revés: la cercanía SÍ importa cuando ninguno de los dos domina con claridad.

    Tres salidas posibles, en este orden de prioridad:
      1. El top-1 supera min_score + confidence_margin ("confianza alta"): DocGate(restrict_to=doc1)
         -- el CONTEXTO se arma solo con chunks de ESE documento, ignorando cualquier otro doc que
         haya cruzado min_score de pura casualidad. Esto es lo que evita la mezcla de políticas
         (el bug original que motivó el viejo gate) SIN tener que preguntar nada.
      2. Si no hay confianza alta, pero los dos mejores docs DISTINTOS están a menos de
         `ambiguity_threshold` de diferencia: DocGate(clarify=(doc1, doc2)) -- ahí sí, ninguno
         domina con claridad, pedir aclaración es lo honesto.
      3. Ninguno de los dos casos: DocGate() vacío -- sigue el camino de siempre (CONTEXTO con los
         hits tal como vinieron, sin restringir ni aclarar)."""
    ranked = sorted(_best_per_doc(hits).items(), key=lambda kv: -kv[1])
    if not ranked:
        return DocGate()
    doc1, s1 = ranked[0]
    if s1 >= min_score + confidence_margin:
        return DocGate(restrict_to=doc1)
    if len(ranked) < 2:
        return DocGate()
    doc2, s2 = ranked[1]
    if s1 - s2 < ambiguity_threshold:
        return DocGate(clarify=(doc1, doc2))
    return DocGate()


def clarify_reply(doc1: str, doc2: str, doc_topics: dict[str, str]) -> str:
    """Pregunta de aclaración nombrando los dos temas en juego (ver gate_docs). Los nombres de tema
    salen de `doc_topics` (config, legibles para voz -- NO los nombres de archivo) y deberían pasar
    lint_for_voice() sin warnings (sin siglas, sin puntuación doble, ver README)."""
    t1 = doc_topics.get(doc1, doc1)
    t2 = doc_topics.get(doc2, doc2)
    return f"Tu pregunta toca dos temas distintos: {t1} y {t2}. ¿Sobre cuál de los dos te referís?"


# ============================================================================
# Lint para voz -- texto que va a pasar por Piper (respuestas canónicas, mensajes de aclaración).
# Compartido con voice/canonical.py (antes duplicado ahí).
# ============================================================================

_DIGIT_RE = re.compile(r"\d")
_TIME_RANGE_RE = re.compile(r"\d+\s*[-–]\s*\d+|\bhs\.?\b", re.IGNORECASE)
_ABBREV_RE = re.compile(r"\b(av|ud|uds|sr|sra|dr|dra|etc|nro|n°|c/u|ej|pág|tel)\.", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"^\s*[-*•]\s|\*\*[^*]+\*\*|__[^_]+__|^\s*\d+\.\s", re.MULTILINE)
# siglas ("RR.HH.", "S.A.") -- 2+ mayúsculas seguidas de un punto; se lee mal en voz alta letra por
# letra, y encadenadas ("RR.HH.") además dejan puntuación muy junta
_SIGLA_RE = re.compile(r"\b[A-ZÁÉÍÓÚÑ]{2,}\.")
# puntuación doble/repetida ("..", "R.H.", dos signos de cierre pegados)
_DOUBLE_PUNCT_RE = re.compile(r"[.,;:!?]{2,}|\.\s*[A-ZÁÉÍÓÚÑ]{1,3}\.")
_MAX_WORDS_VOICE = 80


def lint_for_voice(text: str, label: str = "") -> list[str]:
    """Warnings de estilo para texto que Piper va a leer en voz alta -- no bloquean nada, son para
    que alguien (técnico o no) revise antes de usar el texto. `label` es un prefijo libre para
    identificar de qué texto vienen los warnings (p.ej. el id de una entrada canónica)."""
    p = f"{label}: " if label else ""
    warnings = []
    if _DIGIT_RE.search(text):
        warnings.append(f"{p}tiene dígitos -- Piper lee mejor los números escritos en palabras ('diez' en vez de '10')")
    if _TIME_RANGE_RE.search(text):
        warnings.append(f"{p}parece tener un formato tipo '9-13hs' -- escribilo como se pronuncia ('de nueve a trece horas')")
    if _ABBREV_RE.search(text):
        warnings.append(f"{p}tiene una abreviatura -- escribila completa, Piper no las expande")
    if _SIGLA_RE.search(text):
        warnings.append(f"{p}tiene una sigla (mayúsculas + punto) -- se lee letra por letra, mejor escribirla completa")
    if _DOUBLE_PUNCT_RE.search(text):
        warnings.append(f"{p}tiene puntuación doble o muy junta (p.ej. 'RR.HH.') -- revisar")
    if _MARKDOWN_RE.search(text):
        warnings.append(f"{p}tiene viñetas o markdown -- esto se lee en voz alta, no se muestra como texto")
    n_words = len(text.split())
    if n_words > _MAX_WORDS_VOICE:
        warnings.append(f"{p}tiene {n_words} palabras (más de {_MAX_WORDS_VOICE}) -- una respuesta hablada debería ser más corta")
    return warnings


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
