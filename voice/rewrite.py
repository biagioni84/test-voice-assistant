"""Reescritura de preguntas de seguimiento en una pregunta autónoma, ANTES del retrieval de RAG.
Reemplaza al viejo fallback de "sticky hits" (heredar los chunks del turno anterior sin mirar la
pregunta -- ver README, bug "cambio de día en follow-up").

El diagnóstico con --show-chunk-text había mostrado que el chunk correcto casi siempre SE
RECUPERABA bien; el problema real es que la pregunta tal como la dice el usuario ("Te pregunté el
sábado a qué hora abre.") no es autónoma -- mezcla una frase meta ("te pregunté") con el dato real
(sábado), y esa mezcla también confundía al LLM de respuesta. Reescribirla a una pregunta limpia
("¿A qué hora abre la oficina el sábado?") antes de retrieval, y pasarle SOLO esa pregunta (sin
historial) al LLM de respuesta, ataca las dos cosas con el mismo mecanismo.
"""

SYSTEM_PROMPT = """Reescribí la última pregunta del usuario como una pregunta autónoma que se
entienda sin el historial de la charla. Resolvé referencias ("y el domingo", "eso", "ahí") usando
el historial: reemplazá la referencia por el sustantivo/tema real al que apunta, tomado del
historial. Si la pregunta actual es solo una referencia vacía sin ningún dato nuevo ("y eso",
"¿y ahí?", "decime de nuevo"), la reescritura es directamente la pregunta anterior completa, tal
cual, no una versión abreviada con "eso". Si el usuario corrige algo ("no, te pregunté por el
sábado"), la pregunta reescrita usa la corrección, no lo que se había hablado antes. No respondas
la pregunta. No agregues información que no esté en el historial ni en la pregunta actual. Devolvé
SOLO la pregunta reescrita, nada más -- sin explicaciones ni comillas."""

# (historial [(usuario, asistente), ...], pregunta actual, reescritura esperada)
_EXAMPLES: list[tuple[list[tuple[str, str]], str, str]] = [
    (
        # "eso"/"ahí" sin ningún sustantivo nuevo: no hay nada que resolver más que repetir la
        # pregunta anterior (probado: sin este ejemplo, el modelo devolvía basura tipo "y ahí" que
        # no pasa la validación y termina sin recuperar nada del RAG)
        [("¿A qué hora abre la oficina el domingo?", "La oficina no abre el domingo, está cerrada.")],
        "y eso",
        "¿A qué hora abre la oficina el domingo?",
    ),
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
