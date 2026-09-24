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

import json
import re
import unicodedata

# "y eso", "eso", "y ahí", "¿y ahí?", "eso mismo", "lo mismo", con o sin "¿...?" -- referencias que
# no traen ningún dato nuevo, solo piden repetir/continuar sobre la última pregunta
_EMPTY_REF_RE = re.compile(r"^[¿\s]*(y\s+)?(eso( mismo)?|ah[ií]|lo mismo)[\.\?\s]*$", re.IGNORECASE)

SYSTEM_PROMPT = """Reescribí la última entrada del usuario como una pregunta autónoma que se
entienda sin el historial de la charla. Resolvé referencias ("y el domingo", "eso", "ahí") usando
el historial: reemplazá la referencia por el sustantivo/tema real al que apunta, tomado del
historial -- arrastrá TODAS las entidades relevantes del turno anterior (sujeto, lugar, persona) a
la pregunta reescrita, no solo el dato nuevo que aporta la entrada actual ("todos los días abre a
las diez" después de hablar de "la oficina" tiene que seguir mencionando "la oficina", no perderla).
Preservá el TIPO de pregunta del turno anterior: si era de sí o no ("¿abre la oficina?"), la
reescrita también tiene que ser de sí o no, no una de qué/cuándo/cómo, a menos que el usuario haya
cambiado explícitamente el tipo de pregunta.

Si la entrada actual es solo una referencia vacía sin ningún dato nuevo ("y eso", "¿y ahí?", "decime
de nuevo"), la reescritura es directamente la pregunta anterior completa, tal cual, no una versión
abreviada con "eso". Si el usuario corrige algo ("no, te pregunté por el sábado"), la pregunta
reescrita usa la corrección, no lo que se había hablado antes.

Si la pregunta actual NO menciona ningún sujeto explícito -- ni trae uno propio ("¿A qué hora abre
el martes?", sin decir qué abre) ni hay historial de dónde tomarlo -- completala con el sujeto por
defecto de este asistente: "{default_subject}". Esto NO aplica si la pregunta ya nombra un sujeto
explícito, sea el mismo tema u otro completamente distinto (una persona, un lugar, un tema no
relacionado): en ese caso dejá el sujeto tal cual lo dijo el usuario, no lo reemplaces por el
default ni lo agregues igual "por las dudas".

La entrada puede venir como una AFIRMACIÓN en vez de una pregunta: en español hablado, las
confirmaciones de sí o no suelen sonar como afirmaciones con entonación ("todos los días abre a las
diez") y llegan transcriptas sin signos de interrogación. Convertí esas afirmaciones en una pregunta
de confirmación de sí o no que combine el dato nuevo con el sujeto/tema del historial (si hay
historial) -- esto vale AUNQUE la afirmación ya suene como una oración completa y gramaticalmente
autónoma ("todos los días abre a las diez" se entiende sola, pero igual está omitiendo el sujeto
real del historial; agregalo lo mismo, no la dejes "como está" solo porque ya es una oración
entera). Esto aplica también sin historial: si la entrada es una afirmación aislada (primer turno de
la charla), convertila en pregunta igual, aunque no haya sujeto del historial para completarla.

No respondas la pregunta. No agregues información que no esté en el historial ni en la entrada
actual. Devolvé SOLO la pregunta reescrita, nada más -- sin explicaciones ni comillas."""
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
    (
        # sujeto elidido + preservar tipo sí/no (bug real: sin esto daba "¿Cuándo abre todos los
        # días?", perdiendo "la oficina" Y cambiando sí/no por qué/cuándo)
        [("¿A qué hora abre la oficina?", "Abre a las diez de la mañana.")],
        "todos los días",
        "¿La oficina abre todos los días?",
    ),
    (
        # sujeto omitido, SIN historial (primer turno) -- bug real (25/09, ver README "sujeto
        # omitido"): "¿A qué hora abre el martes?" ya viene con "?" así que ningún filtro barato
        # (looks_like_statement, looks_compound) la agarraba, y sin historial no había de dónde
        # completar "la oficina" -- el reescritor ahora corre siempre (ver LocalLLM.rewrite_query)
        # y esta instrucción/ejemplo es lo que le enseña a completar el sujeto genérico, no el
        # historial. A PROPÓSITO no va al final de la lista, ver comentario en el último few-shot.
        [],
        "¿A qué hora abre el martes?",
        "¿A qué hora abre {default_subject} el martes?",
    ),
    (
        # contraejemplo, agregado junto con el de arriba: sujeto YA explícito (una persona, no el
        # negocio) -- NO hay que reemplazarlo por el default ni agregarlo igual. Sin este ejemplo,
        # el riesgo real es que el modelo generalice de más "toda pregunta sin sujeto de negocio
        # completo con {default_subject}" en vez de "solo si el sujeto está genuinamente ausente".
        [],
        "¿A qué se dedica Juan?",
        "¿A qué se dedica Juan?",
    ),
    (
        # afirmación (sin "?") como confirmación de sí/no, CON historial -- arrastra el sujeto.
        # A PROPÓSITO usa un tema distinto (sala de reuniones) al de arriba (oficina/horario): 3
        # ejemplos seguidos casi idénticos en tema hacían que el 3B, ante una entrada nueva sin
        # relación, "completara el patrón" repitiendo el ejemplo más parecido en vez de generalizar
        # -- bug real encontrado (23/09, ver README) con "El estacionamiento tiene doce lugares."
        # (primer turno, sin relación con oficina/horarios) devolviendo textual la salida de OTRO
        # few-shot. Diversificar el tema de cada ejemplo evita que el modelo tenga un solo patrón
        # de superficie para copiar.
        [("¿Cuántas personas entran en la sala Norte?", "Tiene capacidad para diez personas.")],
        "Se puede reservar hasta por dos horas seguidas.",
        "¿La sala Norte se puede reservar hasta por dos horas seguidas?",
    ),
    (
        # variante con "Y" al principio y dos turnos de historial -- bug real (23/09): con una
        # oración COMPLETA (no un fragmento suelto como "todos los días") el modelo tendía a tratar
        # la afirmación como si ya se bastara sola y perdía el sujeto igual; el "Y" inicial
        # empeoraba todavía más la pérdida del sujeto. A propósito usa "la oficina" de nuevo (mismo
        # sustantivo genérico que el primer ejemplo, no un dato específico de los docs) pero con
        # datos y pregunta DISTINTOS a los de la conversación real que disparó el bug (ver
        # tests/eval_questions.yaml: validation_sujeto_elidido_y_afirmacion) -- sin esto, un tema
        # totalmente distinto (wifi) no alcanzaba para que el modelo generalizara el patrón
        # "Y + oración completa + mismo sujeto que el historial", ver README.
        [
            ("¿A qué hora cierra la oficina?", "Cierra a las seis de la tarde."),
            ("¿Y los feriados?", "No abre los feriados."),
        ],
        "Y abre normalmente los fines de semana.",
        "¿La oficina abre normalmente los fines de semana?",
    ),
    (
        # afirmación aislada, SIN historial (primer turno de la charla), tema distinto de todos
        # los anteriores (vacaciones) -- se convierte a pregunta igual, aunque no haya sujeto del
        # historial para completarla. A PROPÓSITO queda como el ÚLTIMO few-shot de la lista (ver
        # más abajo): recency bias real en un 3B (mismo motivo que el comentario de "sala Norte"
        # más arriba) -- este es el patrón que más importa tener "fresco" para una entrada de
        # primer turno sin relación con nada anterior. Los 2 ejemplos de sujeto omitido (agregados
        # 25/09) NO van al final por el mismo motivo: puestos ahí, un 3B copiaba literalmente la
        # respuesta del último few-shot ("¿A qué se dedica Juan?") para CUALQUIER entrada nueva sin
        # relación -- bug real encontrado al agregarlos (ver tests/eval_questions.yaml:
        # rewrite_afirmacion_primer_turno, que empezó a fallar exactamente así).
        [],
        "Las vacaciones se piden con dos semanas de anticipación.",
        "¿Las vacaciones se piden con dos semanas de anticipación?",
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


def build_system_prompt(default_subject: str) -> str:
    """SYSTEM_PROMPT trae un placeholder "{default_subject}" (ver instrucción de sujeto omitido,
    25/09) -- se completa acá con el valor de config (voice/config.py: RewriteCfg.default_subject),
    nunca hardcodeado en el prompt."""
    return SYSTEM_PROMPT.format(default_subject=default_subject)


def few_shot_messages(default_subject: str) -> list[dict]:
    """default_subject completa el placeholder "{default_subject}" en los ejemplos que lo traen
    (ver _EXAMPLES, caso de sujeto omitido) -- .format() en los que no lo traen es un no-op."""
    msgs = []
    for history, question, rewritten in _EXAMPLES:
        msgs.append({"role": "user", "content": format_input(history, question)})
        msgs.append({"role": "assistant", "content": rewritten.format(default_subject=default_subject)})
    return msgs


def looks_like_question(text: str) -> bool:
    """Validación barata: si esto falla, se usa la pregunta original tal cual (ver LocalLLM.rewrite_query)."""
    text = text.strip()
    if not text or not text.endswith("?"):
        return False
    if len(text.split()) > _MAX_WORDS:
        return False
    return True


# signos que terminan una oración -- ver TERMINAL_PUNCT_CHARS y el bug real documentado en llm.py:
# rewrite_query (reponer "?" sin chequear esto daba salidas como "...de la mañana.?")
TERMINAL_PUNCT_CHARS = ".?!…"

# arranca con una palabra interrogativa (con o sin "¿" adelante) -- si matchea, el texto YA tiene
# pinta de pregunta y no hace falta pasarlo por looks_like_statement
_INTERROGATIVE_START_RE = re.compile(
    r"^[¿\s]*(qu[eé]|cu[aá]les?|c[oó]mo|cu[aá]ndo|d[oó]nde|ad[oó]nde|qui[eé]nes?|"
    r"cu[aá]nt[oa]s?|por\s+qu[eé])\b",
    re.IGNORECASE,
)


def looks_like_statement(question: str) -> bool:
    """Verdadero si la entrada NO tiene pinta de pregunta ya formada -- ni termina en "?" ni arranca
    con una palabra interrogativa. En voz hablada, las confirmaciones de sí/no muchas veces salen
    como afirmaciones con entonación ("todos los días abre a las diez") y Whisper las transcribe sin
    signos -- esto dispara el reescritor para convertirlas en pregunta, incluso en el primer turno
    (sin historial, ver LocalLLM.rewrite_query). Sesgado a recall igual que looks_compound: un falso
    positivo sobre una entrada que en realidad no había que tocar (chitchat, comando imperativo)
    solo cuesta latencia -- is_near_identical() abajo actúa de red de seguridad si el modelo la deja
    prácticamente igual (señal de que no había nada que convertir)."""
    q = question.strip()
    if not q or q.endswith("?"):
        return False
    if _INTERROGATIVE_START_RE.match(q):
        return False
    return True


def _normalize_for_compare(s: str) -> str:
    # OJO: a propósito NO se saca el "¿" inicial -- agregarlo es justo la señal de que el modelo SÍ
    # convirtió la afirmación en pregunta (ver is_near_identical). Solo se saca puntuación de CIERRE
    # (".", "!", "…", "?"), que es cosmética y no indica si hubo una reescritura real.
    s = s.strip()
    s = s.rstrip(TERMINAL_PUNCT_CHARS).strip()
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def is_near_identical(rewritten: str, original: str) -> bool:
    """Compara normalizado (minúsculas, sin acentos, sin puntuación de cierre -- pero SÍ
    conservando un "¿" inicial si lo hay) -- si son prácticamente iguales, el modelo no reescribió
    de verdad, solo repitió la entrada tal cual (a lo sumo con un "?" pegado por la lógica de
    reposición del stop, ver LocalLLM.rewrite_query). Importante cuando `original` era una
    afirmación sin forma de pregunta (looks_like_statement): si el modelo SÍ la envolvió en "¿...?"
    -- aunque no haya cambiado ninguna otra palabra -- eso YA es una conversión válida (el punto de
    looks_like_statement es justo convertir "todos los días abre a las diez" en "¿todos los días
    abre a las diez?", no reformular más que eso) y no debe rechazarse; si en cambio el modelo
    devolvió el texto sin el "¿" inicial (no lo convirtió en absoluto), esto lo agarra."""
    return _normalize_for_compare(rewritten) == _normalize_for_compare(original)


# ============================================================================
# Descomposición de preguntas compuestas (punto 4). Ver LocalLLM.rewrite_query en voice/llm.py.
#
# A diferencia de la reescritura de seguimiento (que solo tiene sentido si hay historial: sin
# historial previo no hay nada que resolver), una pregunta compuesta puede llegar en el PRIMER
# turno de la charla ("decime el horario de lunes a viernes y también el de los sábados") -- por
# eso looks_compound() se evalúa siempre, independiente de self.history.
# ============================================================================

# Palabras interrogativas del español -- dos DISTINTAS en la misma pregunta son una señal de que
# hay dos pedidos distintos ("¿cuándo... y dónde...?"). A propósito exige la TILDE en todas: sin
# ella, "que"/"como"/"cuando"/"donde" son conjunciones/relativos comunísimos en cualquier oración
# ("dije QUE eran", "trabajo COMO profesor", "el lugar DONDE vivo") -- BUG real encontrado (23/09):
# "¿y cuántos días dije que eran?" (un solo pedido, de seguimiento) disparaba looks_compound porque
# "que" (el relativo, sin tilde) matcheaba como si fuera "qué" interrogativo, sumando 2 con
# "cuántos". La tilde es justo lo que en español distingue el uso interrogativo del relativo/
# conjunción -- Whisper la pone cuando transcribe una pregunta real.
_INTERROGATIVE_RE = re.compile(
    r"\b(qué|cuáles?|cómo|cuándo|dónde|adónde|quiénes?|cuánt[oa]s?|por\s+qué)\b",
    re.IGNORECASE,
)
# conectores típicos de enumerar dos pedidos en el mismo turno
_COMPOUND_CONNECTOR_RE = re.compile(r"\badem[aá]s\b|\btambi[eé]n\b|\sy\s", re.IGNORECASE)
# enumeraciones explícitas ("primero... segundo...", "1) ... 2) ...")
_ENUM_RE = re.compile(
    r"\bprimero\b|\bpor un lado\b|\bpor otro lado\b|(?:^|\s)[1-9][\).]\s|(?:^|\s)[a-c][\).]\s",
    re.IGNORECASE,
)

_MAX_SUBQUESTIONS = 3


def looks_compound(question: str) -> bool:
    """Filtro barato (sin llamar al LLM) para detectar preguntas PROBABLEMENTE compuestas. Si no
    dispara, el comportamiento es exactamente el de hoy (una sola pregunta). Si dispara, se invoca
    al reescritor en "modo descomposición" (ver DECOMPOSE_SYSTEM_PROMPT) -- un falso positivo ahí
    solo cuesta la latencia de esa llamada extra, porque en el peor caso el reescritor devuelve una
    lista con una sola sub-pregunta (ver los ejemplos few-shot de abajo, hay uno para ese caso).
    Por eso está pensado a propósito para priorizar recall (agarrar todo lo compuesto real) por
    encima de precisión -- scripts/eval.py reporta la tasa de falsos positivos sobre casos NO
    compuestos de tests/eval_questions.yaml, para poder ver si esto sale caro en la práctica."""
    q = question.strip()
    if q.count("?") >= 2:
        return True
    if _COMPOUND_CONNECTOR_RE.search(q):
        return True
    if len(_INTERROGATIVE_RE.findall(q)) >= 2:
        return True
    if _ENUM_RE.search(q):
        return True
    return False


DECOMPOSE_SYSTEM_PROMPT = """La pregunta actual del usuario puede combinar más de un pedido en el mismo turno. Separala en 1 a
3 sub-preguntas independientes: cada una debe entenderse sola (autónoma, sin las demás) y estar
formulada como pregunta ("¿Cuál es...?", "¿Cómo...?"), nunca como orden ("decime X", "contame Y" se
convierte en "¿Cuál es X?", "¿Qué es Y?"). Usá el historial para resolver referencias, igual que en
una reescritura normal. Si en realidad la pregunta es un solo pedido (no es compuesta), devolvé una
lista con esa única sub-pregunta ya reescrita como autónoma -- no inventes una segunda parte que no
esté ahí. No respondas ninguna pregunta. No agregues información que no esté en el historial ni en
la pregunta actual. Devolvé SOLO un array JSON de strings, nada más -- sin explicaciones, sin
markdown, sin comillas triples. Máximo 3 elementos."""

# (historial, pregunta actual, sub-preguntas esperadas) -- incluye a propósito un caso donde el
# pre-filtro barato dispararía pero en realidad es un solo pedido (ver looks_compound): el
# reescritor debe devolver igual una lista, con un solo elemento, no forzar una segunda parte.
_DECOMPOSE_EXAMPLES: list[tuple[list[tuple[str, str]], str, list[str]]] = [
    (
        [],
        "Decime el horario de lunes a viernes y también el de los sábados.",
        [
            "¿Cuál es el horario de la oficina de lunes a viernes?",
            "¿Cuál es el horario de la oficina los sábados?",
        ],
    ),
    (
        [("¿A qué hora abre la oficina los sábados?", "Los sábados abre de diez a una de la tarde.")],
        "¿Y los domingos, y además cómo restablezco la contraseña del correo?",
        [
            "¿A qué hora abre la oficina los domingos?",
            "¿Cómo puedo restablecer la contraseña del correo?",
        ],
    ),
    (
        [],
        "¿Cuántos días de vacaciones tengo y cuándo se pagan los sueldos?",
        [
            "¿Cuántos días de vacaciones tengo por año?",
            "¿Cuándo se pagan los sueldos?",
        ],
    ),
    (
        # falso positivo del pre-filtro (dispara por "y"): un solo pedido, no dos
        [],
        "¿Puedo trabajar desde casa los sábados y feriados?",
        ["¿Puedo trabajar desde casa los sábados y feriados?"],
    ),
]


# Se agrega al system_prompt del LLM de respuesta (voice/llm.py: stream_answer_multi) SOLO cuando
# hay más de una sub-pregunta resuelta en el mismo turno -- para preguntas simples el prompt de
# respuesta no cambia en nada respecto de antes de este punto 4.
#
# El formato "Parte N: <respuesta>" (una línea por parte) no es capricho -- se probó primero solo
# con la instrucción en prosa ("respondé cada parte por separado") y el 3B mezclaba: con dos partes
# que comparten CONTEXTO idéntico (pasa seguido con los docs de prueba actuales, donde un mismo
# párrafo trae varios datos, ver pregunta_compuesta) el modelo ignoraba la PARTE 1 por completo y
# solo contestaba con el dato más "saliente" del CONTEXTO para ambas partes. Pedirle una etiqueta
# "Parte N:" por línea le da un ancla estructural que si o si tiene que llenar para cada parte --
# con eso, 2/2 partes se responden bien en el caso de prueba. Las etiquetas se sacan antes de
# hablar/mostrar la respuesta (ver strip_part_labels) -- son un andamiaje interno, no algo para el
# usuario.
MULTI_PART_SUFFIX = """

Esta vez la pregunta viene dividida en PARTES numeradas, cada una con su propia PREGUNTA y (si
corresponde) su propio CONTEXTO -- el CONTEXTO puede repetirse igual entre partes, pero cada parte
tiene su propio dato puntual para extraer de ahí; no ignores una parte solo porque el CONTEXTO se
parezca al de otra. Respondé TODAS las partes, en el mismo orden, usando exactamente este formato,
una línea por parte:
Parte 1: <respuesta breve, una frase>
Parte 2: <respuesta breve, una frase>
(una línea "Parte N: ..." por cada parte que te dieron, ni una menos)"""

_PART_LABEL_RE = re.compile(r"(?im)^\s*parte\s*\d+\s*:\s*")


def strip_part_labels(text: str) -> str:
    """Saca las etiquetas "Parte N: " que pide MULTI_PART_SUFFIX -- son un andamiaje interno para
    que el modelo separe bien cada sub-respuesta, no algo que deba decirse en voz alta ni mostrarse
    al usuario (ver voice/pipeline.py: Assistant.answer())."""
    return _PART_LABEL_RE.sub("", text).strip()


def decompose_few_shot_messages() -> list[dict]:
    msgs = []
    for history, question, subqs in _DECOMPOSE_EXAMPLES:
        msgs.append({"role": "user", "content": format_input(history, question)})
        msgs.append({"role": "assistant", "content": json.dumps(subqs, ensure_ascii=False)})
    return msgs


def parse_subquestions(text: str) -> list[str] | None:
    """Valida el output JSON del reescritor en modo descomposición. None si no valida -- el
    llamador (LocalLLM._decompose) cae al camino normal (pregunta tal cual, o reescritura de
    seguimiento) en vez de usar esto."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list) or not (1 <= len(data) <= _MAX_SUBQUESTIONS):
        return None
    out = []
    for item in data:
        if not isinstance(item, str):
            return None
        q = item.strip()
        if not q:
            return None
        if not q.endswith("?"):
            q += "?"
        if len(q.split()) > _MAX_WORDS:
            return None
        out.append(q)
    return out
