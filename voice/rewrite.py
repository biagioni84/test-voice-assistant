"""Reescritura de preguntas de seguimiento en una pregunta autónoma, ANTES del retrieval de RAG.
Reemplaza al viejo fallback de "sticky hits" (heredar los chunks del turno anterior sin mirar la
pregunta -- ver README, bug "cambio de día en follow-up").

El diagnóstico con --show-chunk-text había mostrado que el chunk correcto casi siempre SE
RECUPERABA bien; el problema real es que la pregunta tal como la dice el usuario ("Te pregunté el
sábado a qué hora abre.") no es autónoma -- mezcla una frase meta ("te pregunté") con el dato real
(sábado), y esa mezcla también confundía al LLM de respuesta. Reescribirla a una pregunta limpia
("¿A qué hora abre la oficina el sábado?") antes de retrieval, y pasarle SOLO esa pregunta (sin
historial) al LLM de respuesta, ataca las dos cosas con el mismo mecanismo.

"Referencia vacía" ("y eso", "y ahí") se resuelve con una regla determinista (is_empty_reference +
resolve_empty_reference), no con el LLM: probado extensamente que el 3B es demasiado sensible a la
redacción EXACTA de la respuesta previa (varía turno a turno aunque temperature=0, porque el
CONTEXTO retornado por el RAG varía) -- para este patrón puntual, tan bien definido, alcanza con
mirar la última pregunta del usuario en el historial. Más rápido además (sin llamada al LLM)."""

import re

# "y eso", "eso", "y ahí", "¿y ahí?", "eso mismo", "lo mismo", con o sin "¿...?" -- referencias que
# no traen ningún dato nuevo, solo piden repetir/continuar sobre la última pregunta
_EMPTY_REF_RE = re.compile(r"^[¿\s]*(y\s+)?(eso( mismo)?|ah[ií]|lo mismo)[\.\?\s]*$", re.IGNORECASE)

SYSTEM_PROMPT = """Reescribí la última pregunta del usuario como una pregunta autónoma que se
entienda sin el historial de la charla. Resolvé referencias ("y el domingo", "eso", "ahí") usando
el historial: reemplazá la referencia por el sustantivo/tema real al que apunta, tomado del
historial. Si la pregunta actual es solo una referencia vacía sin ningún dato nuevo ("y eso",
"¿y ahí?", "decime de nuevo"), la reescritura es directamente la pregunta anterior completa, tal
cual, no una versión abreviada con "eso". Si el usuario corrige algo ("no, te pregunté por el
sábado"), la pregunta reescrita usa la corrección, no lo que se había hablado antes. No respondas
la pregunta. No agregues información que no esté en el historial ni en la pregunta actual. Devolvé
SOLO la pregunta reescrita, nada más -- sin explicaciones ni comillas."""
# (probado y descartado: una versión condensada de este prompt, ~130 tokens en vez de ~240, parecía
# preservar el significado pero era una instrucción más débil -- fallaba justo en el caso de "y eso"
# con texto de respuesta REAL del LLM (no el texto exacto de los ejemplos few-shot). El ahorro de
# tokens no vale la regresión; ver README.)

# (historial [(usuario, asistente), ...], pregunta actual, reescritura esperada)
# Nota: NO hay ejemplo para "y eso"/"y ahí" -- ver is_empty_reference(), se resuelve sin el LLM.
_EXAMPLES: list[tuple[list[tuple[str, str]], str, str]] = [
    (
        [("¿A qué hora abre la oficina el domingo?", "La oficina no abre el domingo, está cerrada.")],
        "y los sábados",
        "¿A qué hora abre la oficina los sábados?",
    ),
    (
        [("¿A qué hora abre la oficina el domingo?", "La oficina no abre el domingo, está cerrada.")],
        "Te pregunté el sábado, no el domingo.",
        "¿A qué hora abre la oficina el sábado?",
    ),
    (
        # cambio de tema total: la pregunta ya es autónoma, no depende del historial
        [("¿A qué hora abre la oficina los sábados?", "Los sábados abre de diez a una de la tarde.")],
        "¿Cómo puedo restablecer la contraseña del correo?",
        "¿Cómo puedo restablecer la contraseña del correo?",
    ),
    (
        # comentario/meta-pregunta, no es sobre los documentos: se deja prácticamente igual, no se
        # fuerza a que "hable de" el tema anterior
        [("¿A qué hora abre la oficina los sábados?", "Los sábados abre de diez a una de la tarde.")],
        "¿Por qué demoras tanto en responder?",
        "¿Por qué demoras tanto en responder?",
    ),
]

_MAX_WORDS = 20  # si la reescritura tiene más palabras que esto, algo salió mal (parafraseo largo,
                 # el modelo respondió la pregunta en vez de reescribirla, etc.)


def is_empty_reference(question: str) -> bool:
    """"y eso", "y ahí", etc: referencia sin ningún dato nuevo. Ver el docstring del módulo."""
    return bool(_EMPTY_REF_RE.match(question.strip()))


def resolve_empty_reference(history: list[tuple[str, str]]) -> str | None:
    """Para una referencia vacía, la pregunta autónoma es directamente la última pregunta del
    usuario en el historial. None si no hay historial (no debería pasar: rewrite_query ya filtra
    ese caso antes de llegar acá)."""
    return history[-1][0] if history else None


def format_input(history: list[tuple[str, str]], question: str) -> str:
    lines = ["Historial:"]
    for u, a in history:
        lines.append(f"Usuario: {u}")
        lines.append(f"Asistente: {a}")
    lines.append("")
    lines.append(f"Pregunta actual: {question}")
    return "\n".join(lines)


def few_shot_messages() -> list[dict]:
    msgs = []
    for history, question, rewritten in _EXAMPLES:
        msgs.append({"role": "user", "content": format_input(history, question)})
        msgs.append({"role": "assistant", "content": rewritten})
    return msgs


def looks_like_question(text: str) -> bool:
    """Validación barata: si esto falla, se usa la pregunta original tal cual (ver LocalLLM.rewrite_query)."""
    text = text.strip()
    if not text or not text.endswith("?"):
        return False
    if len(text.split()) > _MAX_WORDS:
        return False
    return True
