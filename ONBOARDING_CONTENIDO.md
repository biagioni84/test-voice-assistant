# Incorporar contenido real

Este documento es la guía para reemplazar el contenido de prueba (`docs/`, `tests/canonical_answers.yaml`,
`tests/calibration_questions.yaml`, `tests/calibration_canonical.yaml`) por el contenido real de la
empresa. Nada de esto necesita tocar código -- es la razón de ser de esta guía: el código quedó
congelado (ver README, sección "Cierre de etapa"), lo que sigue es un trabajo de **contenido y
calibración**, no de desarrollo.

## 1. Documentos del RAG (`docs/`)

**Formato esperado**: Markdown (`.md`) o texto plano (`.txt`), uno o más archivos en `docs/`.

**Cómo estructurarlos** (ver `voice/rag.py: _chunk`, `README` sección "Chunking"):
- **Un hecho por párrafo.** El chunking parte los documentos por párrafo (línea en blanco = límite
  de chunk) -- si dos hechos sin relación quedan en el mismo párrafo, van a terminar en el mismo
  chunk, y el reranker/LLM van a tener que separar mentalmente dos cosas que no van juntas. Ejemplo
  MALO: un párrafo que mezcla horario de oficina y política de wifi. Ejemplo BUENO: un párrafo por
  horario, otro por wifi, otro por salas de reunión.
- **Título claro en la primera línea**, como encabezado Markdown: `# Nombre del documento (lo que
  sea entre paréntesis se ignora)`. Se usa para prefijar cada chunk (`"Título: contenido"`), ayuda
  al retrieval a saber de qué documento viene cada dato.
- Un párrafo muy largo se corta automáticamente por oraciones (`chunk_chars` en `config.toml`,
  actualmente 500) -- no hace falta partirlo a mano, pero si un párrafo mezcla temas, igual conviene
  partirlo en dos párrafos (dos hechos) en vez de confiar en el corte automático.
- Frases completas y autocontenidas: evitar "esto también aplica" o referencias a "lo anterior" --
  cada chunk se recupera y se le muestra al LLM aislado, sin el resto del documento alrededor.

**Dónde van**: `docs/` en la raíz del proyecto (configurable con `rag.docs_dir` en `config.toml`,
no hace falta tocarlo si se deja en `docs/`). Reemplazar los 3 archivos de ejemplo
(`oficina.md`, `recursos_humanos.md`, `soporte_tecnico.md`) por los reales, o agregar más.

**Cómo reindexar**: no hace falta ningún comando aparte -- `voice/rag.py: Retriever._build()` cachea
el índice de embeddings en `.cache/` con una clave que incluye el contenido de los documentos; si
los documentos cambian, el hash cambia y se recalcula solo la próxima vez que arranque el asistente
(o `scripts/eval.py`/`scripts/calibrate.py`). Si por algún motivo hace falta forzar un rebuild,
borrar `.cache/` a mano.

## 2. Respuestas canónicas (`tests/canonical_answers.yaml`)

Ver `voice/canonical.py` y la sección "Respuestas canónicas" del README para el mecanismo completo.
Acá solo la guía de contenido.

**Cómo redactar para voz** -- este texto lo lee Piper en voz alta, no se muestra como texto escrito:

| Malo (para voz) | Bueno (para voz) | Por qué |
|---|---|---|
| "Atendemos de 9 a 18hs." | "Atendemos de nueve a dieciocho horas." | Piper lee dígitos y abreviaturas mal o de forma robótica |
| "Somos una empresa de IT fundada en 2015." | "Somos una empresa de tecnología fundada hace diez años." | siglas ("IT") se leen letra por letra; preferir años relativos si no hace falta el año exacto |
| "- Desarrollo\n- Consultoría\n- Soporte" | "Nos dedicamos al desarrollo de software, la consultoría y el soporte técnico." | viñetas/markdown no existen en voz, hay que redactar la lista como oración |
| "Contactanos: info@empresa.com o al 011-4567-8901." | "Podés escribirnos a info arroba empresa punto com, o llamarnos al cero once, cuatro cinco seis siete, ocho nueve cero uno." | los símbolos ("@", "-") se leen mal si no se los escribe fonéticamente |
| Una respuesta de 150 palabras | Una respuesta de 2-4 oraciones | una respuesta hablada larga es difícil de seguir; si hace falta más detalle, dividir en dos entradas |

**Variantes con la misma información**: cada entrada puede (y debería) tener 2-3 variantes de
respuesta, para que sonar repetitivo en una conversación larga se sienta menos robótico. Pero
**todas las variantes tienen que decir exactamente lo mismo** -- son formas distintas de decir el
mismo dato, no datos distintos. Si se actualiza un dato (un teléfono, una dirección), hay que
actualizarlo en TODAS las variantes de esa entrada a la vez, o van a quedar contradiciéndose entre
sí según qué variante le toque en la rotación.

**Cómo correr el lint**: se corre solo, no hace falta un comando aparte -- `voice/canonical.py:
CanonicalMatcher.__init__` lo corre al cargar el contenido (al arrancar el asistente, o al correr
`scripts/eval.py`/`scripts/calibrate.py`) e imprime warnings con `[canonical] ⚠`. Para verlos sin
levantar todo el pipeline: `python scripts/calibrate.py --dry-run` (carga el `CanonicalMatcher`,
corre el lint, no escribe nada). Los warnings NO bloquean la carga -- revisarlos y corregir a mano
lo que corresponda.

**Validación** (si algo está mal, el asistente no arranca -- el error nombra la entrada):
- ids únicos.
- `formulaciones` y `respuestas` no vacías, ningún elemento vacío.
- ninguna formulación puede ser prácticamente igual a una de OTRA entrada (competirían por la misma
  pregunta).

## 3. `doc_topics` (`config.toml`, sección `[rag.doc_topics]`)

Un nombre de tema por documento, en español natural, para el mensaje de aclaración del gate de
ambigüedad ("Tu pregunta toca dos temas distintos: **X** y **Y**. ¿Sobre cuál de los dos te
referís?"). NO son los nombres de archivo -- son nombres pensados para decirse en voz alta.

- Bueno: `"políticas de recursos humanos"`, `"horarios y logística de oficina"`.
- Malo: `"recursos_humanos.md"` (nombre de archivo), `"políticas de RR.HH."` (sigla + puntuación
  doble -- ver el bug real que esto causó, README sección del gate de ambigüedad).

`scripts/eval.py` corre `lint_for_voice` sobre estos nombres al arrancar (mismo lint que las
respuestas canónicas) -- revisar los warnings `[lint]` antes de dar por buena la calibración.

## 4. Sets de preguntas de calibración y validación

**Quién las escribe**: idealmente, gente de la empresa que conozca el negocio pero que **no haya
mirado los documentos que se acaban de cargar** -- el objetivo es capturar cómo preguntaría alguien
real, no formulaciones optimizadas para que el sistema las entienda. Si la misma persona que redactó
los documentos también escribe las preguntas, hay riesgo de que use exactamente el vocabulario del
documento, lo cual no representa una pregunta real de un usuario.

**Categorías necesarias** (para `tests/calibration_questions.yaml`, `tests/calibration_canonical.yaml`,
y para nutrir casos nuevos en `tests/eval_questions.yaml`):

| Categoría | Qué es | Mínimo sugerido |
|---|---|---|
| En dominio | hay un documento real que responde -- no debería abstener | 6-10, cubriendo cada documento |
| Fuera de dominio | no tiene nada que ver con ningún documento | 4-6 |
| Sin respuesta en docs | el tema está pero el dato puntual no | 3-4 |
| Ambigua | calza genuinamente con 2+ documentos a la vez | 3-5, sobre temas que REALMENTE se solapen (ver limitación conocida: este set de prueba no tuvo ninguna con datos reales) |
| Compuesta | pide 2 datos en la misma pregunta | 3-4, incluida al menos una imperativa ("decime X y también Y") |
| Follow-up | depende del turno anterior para tener sentido | 4-6, con formulaciones vagas distintas ("y eso", "¿y el otro?", "lo mismo pero para X") |
| Afirmación-como-pregunta | dicha como afirmación, sin "?", pide confirmación | 3-4 |
| Match canónico (si aplica) | debería responder con una entrada canónica | 2-3 por entrada, incluida una imperativa y una como follow-up |
| Falso positivo canónico | se parece a una pregunta canónica pero NO debería matchear | 2-3 por entrada con riesgo de confusión |

**Formato**: seguir el de `tests/calibration_questions.yaml`/`tests/calibration_canonical.yaml`
existentes (campos `question`, `label`, y `expected_doc`/`entry_id` según corresponda) -- son
archivos YAML simples, no hace falta saber programar para editarlos.

## 5. Orden de pasos para incorporar contenido nuevo

1. **Cargar contenido**: reemplazar `docs/*.md`, `tests/canonical_answers.yaml`, `config.toml`
   (`[rag.doc_topics]`).
2. **Lint**: `python scripts/calibrate.py --dry-run` y revisar los warnings `[canonical]`/`[lint]`.
   Corregir el contenido hasta que no queden warnings importantes (dígitos, siglas, puntuación
   doble, respuestas muy largas o con largos muy dispares entre variantes).
3. **Reindexar**: no hace falta ningún paso manual (ver sección 1) -- pasa solo la primera vez que
   se levanta el asistente con los documentos nuevos.
4. **Escribir los sets de calibración** (`tests/calibration_questions.yaml`,
   `tests/calibration_canonical.yaml`) con las categorías de la sección 4, sobre el contenido real.
5. **`scripts/calibrate.py`**: corre el barrido y escribe `calibration.toml` con los umbrales
   nuevos.
6. **Revisar precision/recall** del reporte de `calibrate.py` -- no basta con que el script corra,
   hay que leer la tabla y confirmar que los umbrales elegidos tengan sentido (ver criterios
   mínimos más abajo). Si algo no cierra, generalmente es señal de que el set de calibración
   necesita más preguntas en alguna categoría, no de que haya que ajustar el script.
7. **Suite completa**: `python scripts/eval.py` (o el comando único de verificación, ver README) --
   tiene que terminar con exit code 0. Revisar también a mano el `.md` generado, sobre todo los
   casos que siguen siendo juicio humano (ver README, cobertura de aserciones).
8. **Sesión en vivo con usuarios que no conozcan el sistema**: la prueba real. Wake word o
   `--no-wake`, conversación natural, sin guionar preguntas de antemano.
9. **Registrar los fallos como casos nuevos**: cualquier bug o comportamiento raro de la sesión en
   vivo se agrega a `tests/eval_questions.yaml` -- **como `split: validation`** si es una
   formulación nueva que expone un patrón general (no se ajusta el prompt/umbral calcado a ese
   caso puntual, se generaliza o se documenta como límite conocido, ver README), o **como `split:
   dev`** (default) si es un caso de control que ya se usó para ajustar algo puntual. Mismo
   criterio que se usó durante todo este desarrollo (ver README, "Dev vs. validación").

## 6. Criterios mínimos para dar por buena la calibración

- **Precisión antes que recall**, siempre, en dos lugares:
  - Abstención (`rag.min_score`): preferible que una pregunta con respuesta real quede sin
    responder (cae a "no lo encontré", el usuario puede reformular) a que una pregunta sin
    respuesta real reciba una inventada con confianza.
  - Respuestas canónicas (`canonical.threshold`): preferible que una pregunta que debería matchear
    una entrada caiga al RAG (peor, pero no incorrecto) a que matchee la entrada EQUIVOCADA
    (recita, con total confianza, un texto que no corresponde).
- **Cero alucinaciones en el set de seguridad**: los casos de identidad inventada
  (`identidad_inventada*`), abstención fuera de dominio (`pregunta_sin_contexto`,
  `canonical_fuera_de_dominio_sigue_abstenido`) y 0 caracteres CJK tienen que pasar siempre, sin
  excepción, antes de considerar lista la calibración -- son los chequeos `expect_abstain`/CJK
  automáticos del harness, no hace falta juicio humano para esto.
- Si `scripts/calibrate.py` reporta casos "peligrosos" (una entrada canónica equivocada, o un doc
  cruzando el umbral cuando no debería) en el umbral elegido, **no se acepta esa calibración** --
  hay que agregar más preguntas al set para que el barrido tenga más señal, no forzar un umbral a
  mano.
