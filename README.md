# Asistente de voz local

Asistente de voz que corre **100% local** (sin llamadas a APIs externas): detecta una palabra de
activación, escucha una pregunta, busca la respuesta en documentos propios (RAG) o en un set de
respuestas fijas (FAQ), la genera con un LLM chico y la dice en voz alta, todo en la misma máquina.

Es un POC probado en una laptop con GPU de 4GB de VRAM (RTX 3050) — el presupuesto de hardware
ajustado explica varias decisiones de diseño (modelo de 3B en vez de uno más grande, cuantización,
reparto de capas GPU/CPU, etc.), documentadas en detalle en **[`BITACORA.md`](BITACORA.md)**.

> Este documento describe **cómo funciona el sistema hoy**. Para el historial de decisiones, bugs
> reales encontrados y resueltos, y experimentos descartados (con su razonamiento), ver
> **[`BITACORA.md`](BITACORA.md)**.

## Índice

- [Instalar](#instalar)
- [Usar](#usar)
- [El recorrido de un turno](#el-recorrido-de-un-turno)
- [Módulos](#módulos)
- [Configuración](#configuración)
- [Testing](#testing)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Estructura del repo](#estructura-del-repo)
- [Contenido real y despliegue](#contenido-real-y-despliegue)
- [Notas / gotchas de esta máquina](#notas--gotchas-de-esta-máquina)

## Instalar

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
uv pip install -p .venv/bin/python --no-deps openwakeword

.venv/bin/python scripts/download_models.py   # ~5 GB: Whisper turbo, LLM, voz Piper, embeddings
```

El LLM corre como un subproceso separado (`llama-server`), no como paquete de pip — hay que
compilarlo desde el repo de `llama.cpp` con soporte CUDA:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build-cuda -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target llama-server -j"$(nproc)"
```

(`86` es la compute capability de una RTX 30xx — ajustar según la GPU.) Después, `config.toml` →
`[llm].server_bin` tiene que apuntar al binario resultante.

Sin GPU (o para no esperar la compilación), se puede correr todo en CPU bajando `n_gpu_layers` a
`0` en `config.toml` — va a ser bastante más lento (ver benchmarks en `BITACORA.md`).

## Usar

```bash
# loop completo con wake word ("hey jarvis" por defecto)
.venv/bin/python main.py

# sin wake word, escucha directo
.venv/bin/python main.py --no-wake

# sin micrófono, para probar rápido
.venv/bin/python main.py --text "¿a qué hora abre la oficina los sábados?"

# benchmark end-to-end (usa Piper para generar la pregunta hablada, sin necesitar mic)
.venv/bin/python scripts/bench.py
```

Toda la configuración vive en `config.toml` (ver [Configuración](#configuración)).

## El recorrido de un turno

Desde que el usuario termina de hablar hasta que sale el primer audio de la respuesta, un turno
pasa por esta secuencia. Cada paso dice qué módulo lo implementa y qué lo puede desviar del camino
"normal".

```
usuario habla
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ CAPTURA: wake word (opcional) → VAD → STT                                    │
│   voice/wakeword.py · voice/vad.py · voice/stt.py                            │
└─────────────────────────────────────────────────────────────────────────────┘
     │  texto transcripto
     ▼
¿"repetí" / "¿cómo?" / "no te escuché"?  (guardrails.is_repeat_request)
     │ no                                    │ sí
     │                                       ▼
     │                          repite la ÚLTIMA respuesta tal cual → TTS
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ REESCRITURA / DESCOMPOSICIÓN                                                  │
│   voice/rewrite.py + voice/llm.py: LocalLLM.rewrite_query                    │
│                                                                                │
│   ¿pregunta compuesta? (looks_compound) ──sí──► LLM descompone en 1-3         │
│        │no                                       sub-preguntas independientes │
│        ▼                                                                      │
│   ¿charla social reconocida? (is_chitchat) ──sí──► se salta la reescritura    │
│        │no                                                                    │
│        ▼                                                                      │
│   LLM reescribe la pregunta como autónoma: resuelve referencias del           │
│   historial, completa el sujeto si está omitido, convierte afirmaciones      │
│   ("todos los días abre a las diez") en preguntas de confirmación            │
└─────────────────────────────────────────────────────────────────────────────┘
     │  1-3 sub-preguntas, cada una autónoma
     ▼
┌─────────────── por cada sub-pregunta, en orden de prioridad ────────────────┐
│                                                                               │
│  1. RESPUESTAS CANÓNICAS (FAQ, texto fijo)                                   │
│     voice/canonical.py: CanonicalMatcher.match()                            │
│     Reranker cross-encoder contra formulaciones guardadas. Si matchea por    │
│     encima de canonical.threshold → texto FIJO (con rotación de variante),  │
│     NUNCA llega al RAG ni al LLM.                                           │
│              │ no matchea                                                    │
│              ▼                                                               │
│  2. RETRIEVAL HÍBRIDO + RERANKER                                             │
│     voice/rag.py: Retriever.retrieve()                                      │
│     Denso (embeddings) + BM25 léxico, fusionados por RRF → reranker         │
│     cross-encoder reordena los candidatos por relevancia fina.              │
│              │                                                               │
│              ▼                                                               │
│  3. ¿SIN HITS Y NO ES CHARLA SOCIAL?  ──sí──► abstención (texto FIJO)        │
│              │no                             guardrails.abstain_reply()      │
│              ▼                                                               │
│  4. GATE DE AMBIGÜEDAD / CONFIANZA                                           │
│     voice/guardrails.py: gate_docs()                                        │
│     ¿el top-1 domina con confianza clara? ──sí──► CONTEXTO restringido a    │
│              │no                                   ESE documento solamente   │
│              ▼                                                               │
│     ¿2 documentos distintos, ninguno domina, muy cerca entre sí? ──sí──►    │
│              │no                                 aclaración (texto FIJO)     │
│              ▼                                                               │
│     CONTEXTO normal (todos los hits recuperados)                            │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────┘
     │  por sub-pregunta: texto canónico listo, nota fija, o (sub-pregunta + CONTEXTO)
     ▼
¿alguna sub-pregunta necesitó al LLM?  ──no──► solo texto fijo (canónicas + notas) ──► TTS
     │ sí
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ GENERACIÓN                                                                    │
│   voice/llm.py: LocalLLM.stream_answer_multi()                              │
│   Una sola llamada al LLM con todas las sub-preguntas resueltas juntas       │
│   (etiquetadas "Parte N:" internamente si son más de una). Streaming token   │
│   por token, restringido por una gramática GBNF que impide caracteres CJK.  │
└─────────────────────────────────────────────────────────────────────────────┘
     │  tokens en streaming
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FILTROS Y VOZ                                                                 │
│   voice/tts.py: iter_sentences() + clean_for_speech()                       │
│   Los tokens se agrupan en oraciones completas; cada oración se limpia      │
│   (markdown, emojis, CJK residual) y se sintetiza/reproduce SIN esperar a   │
│   que el LLM termine de generar el resto de la respuesta.                   │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
respuesta hablada
```

**Por qué "1-3 sub-preguntas" en vez de "1 pregunta"**: una pregunta compuesta ("decime el horario
de los sábados y también si puedo trabajar remoto") se separa en partes independientes que pasan
**cada una** por el camino completo (canónica → RAG → gate) antes de juntarse en una sola llamada al
LLM de respuesta. Una pregunta simple es el caso particular de "1 sub-pregunta".

## Módulos

Todo el código vive en `voice/`. Cada archivo tiene una responsabilidad concreta:

### `pipeline.py` — orquestación

`Assistant` es la clase central: en `__init__` carga todos los módulos de abajo (con warm-ups para
Whisper y Piper, para no pagar el costo de la primera inferencia en la primera pregunta real del
usuario). `run()` es el loop principal (wake word → VAD → transcribir → responder → repetir);
`answer()` implementa el recorrido de un turno completo descripto arriba. `Turn` es un dataclass que
acumula toda la traza y los tiempos de un turno (para métricas y para `scripts/eval.py`).

### `audio.py` — micrófono y parlantes

`Mic`/`Speaker` envuelven `soundcard` (que a su vez usa PulseAudio, disponible sin configuración
extra en WSLg). `Speaker` reproduce en un hilo aparte con una cola, así el resto del pipeline no
espera a que termine de sonar cada frase.

### `wakeword.py` — palabra de activación

Envuelve `openWakeWord` (backend ONNX). Procesa audio en frames de 80ms; `triggered()` compara el
score contra `wakeword.threshold`. Configurable/desactivable (`wakeword.enabled = false` en
`config.toml`, o `--no-wake` en `main.py`).

### `vad.py` — detección de fin de frase

`UtteranceRecorder.record()` usa Silero VAD para grabar del micrófono hasta que detecta silencio
sostenido (`vad.end_silence_ms`) o se agota un timeout de "nadie habló" (`vad.no_speech_timeout_s`).
Guarda 300ms de preroll antes de que se detecte el inicio de la voz, para no cortar la primera
sílaba. Usa dos umbrales distintos para "empezar a grabar" y "seguir grabando" (histéresis), así una
caída momentánea de energía en medio de una frase no la corta antes de tiempo.

### `stt.py` — transcripción

`Transcriber` envuelve `faster-whisper` (`large-v3-turbo`, cuantizado `int8_float16` en GPU, ~1GB de
VRAM). Si la carga en GPU falla (sin CUDA, sin VRAM), cae automáticamente a CPU con `int8`.

### `rewrite.py` + `llm.py: LocalLLM.rewrite_query()` — reescritura y descomposición

Antes de tocar el RAG, la pregunta del usuario se reescribe como una pregunta autónoma (sin
depender del historial de la charla) usando el mismo LLM chico, con few-shots. Esto reemplaza tener
que pasarle el historial completo al LLM de respuesta: el LLM de respuesta siempre recibe una
pregunta ya autosuficiente + el CONTEXTO de RAG de ese turno, nada más.

Disparadores concretos (funciones deterministas y baratas, sin LLM, que deciden SI hace falta
reescribir o descomponer):

- **`looks_compound(pregunta)`**: conectores ("y", "también"), dos signos de pregunta, dos palabras
  interrogativas con tilde, o enumeraciones explícitas → dispara la descomposición en 1-3
  sub-preguntas (`DECOMPOSE_SYSTEM_PROMPT`, devuelve un array JSON). Sesgado a falsos positivos a
  propósito: cuestan solo latencia, no corrección (si en realidad no era compuesta, el LLM devuelve
  una lista de un solo elemento).
- **`is_chitchat(pregunta)`** (de `guardrails.py`): frases de charla social reconocidas se saltan la
  reescritura por completo — no hay nada que resolver ni sujeto que completar.
- **Sujeto omitido**: si la pregunta no menciona ningún sujeto explícito (`"¿A qué hora abre el
  martes?"`, sin decir qué abre) y no hay historial de dónde tomarlo, el reescritor lo completa con
  `rewrite.default_subject` (configurable, no hardcodeado en el prompt).
- **`is_empty_reference(pregunta)`**: referencias vacías sin dato nuevo ("y eso", "¿y ahí?") se
  resuelven de forma determinista (la pregunta anterior del historial, tal cual) — sin llamar al
  LLM, porque el LLM resultó ser frágil a la redacción exacta de la respuesta anterior.

El resto de los casos (hay historial, o la entrada tiene forma de afirmación sin "?") pasan por el
LLM con few-shots (`SYSTEM_PROMPT`, `_EXAMPLES`). Salvo charla social reconocida, el reescritor
corre en **todo** turno, incluida una pregunta simple bien formada del primer turno.

### `canonical.py` — respuestas canónicas (FAQ)

`CanonicalMatcher.match()` compara la (sub-)pregunta contra una lista de "formulaciones" guardadas
por entrada (`tests/canonical_answers.yaml` en este POC — reemplazar por contenido real, ver
`ONBOARDING_CONTENIDO.md`) usando el mismo modelo de reranker que el RAG. Si el mejor score supera
`canonical.threshold`, se responde con texto FIJO, redactado de antemano — nunca se genera con el
LLM. Cada entrada puede tener varias redacciones equivalentes ("variantes"); `_next_variant()` las
rota de forma determinista dentro de una misma conversación (nunca al azar), así una pregunta
repetida ("¿y el wifi?" dos veces) no suena idéntica las dos veces.

### `rag.py` — retrieval

`Retriever` construye el índice al arrancar: cada archivo de `rag.docs_dir` se separa en chunks (un
párrafo = un chunk = un hecho, a propósito — nunca se combinan párrafos distintos en un mismo chunk,
para no mezclar hechos sin relación al armar el CONTEXTO). El resultado se cachea en `.cache/` por
hash de contenido — cambiar los documentos, el modelo de embeddings o el chunking invalida el caché
automáticamente.

`retrieve(query)` hace: (1) recall inicial — denso (coseno de embeddings) o híbrido (denso + BM25
léxico, fusionados por Reciprocal Rank Fusion) según `rag.hybrid_enabled`; (2) reranking — un
cross-encoder reordena los `rag.reranker_candidates` mejores candidatos por relevancia fina; (3)
filtra por `rag.min_score` (en la escala del reranker, no 0-1). Ese score final es el que usan tanto
la abstención como el gate de ambigüedad.

### `guardrails.py` — reglas deterministas

Un conjunto de funciones sin LLM que resuelven casos donde generar con el modelo es más riesgoso que
seguir una regla fija:

- **`abstain_reply()`**: texto fijo (3 variantes) para cuando no hay CONTEXTO relevante y la
  pregunta no es charla social — evita que el LLM invente una respuesta sin datos de respaldo.
- **`gate_docs()`**: decide, a partir de los hits del RAG, si (a) un documento domina con confianza
  clara (`score >= min_score + confidence_margin`) → restringe el CONTEXTO a sus chunks solamente,
  descartando cualquier otro documento que haya cruzado `min_score` de casualidad; (b) ningún
  documento domina Y dos de distintos documentos están muy cerca entre sí (`ambiguity_threshold`) →
  pide aclaración; (c) ninguno de los dos casos → CONTEXTO normal. Cercanía de scores por sí sola
  NO implica ambigüedad (pueden ser complementarios, no en conflicto) — de ahí la separación en dos
  condiciones distintas en vez de una sola.
- **`is_chitchat()` / `is_repeat_request()`**: listas de patrones para charla social y pedidos de
  repetición ("repetí", "¿cómo?"), respectivamente.
- **`CJK_GRAMMAR`** (usada en `llm.py`): gramática GBNF que restringe los caracteres válidos que
  puede generar el LLM, para que nunca emita texto en chino/japonés/coreano (un desvío observado en
  el modelo de 3B sin relación con el idioma del CONTEXTO).
- **`lint_for_voice()`**: chequea texto pensado para decirse en voz alta (dígitos sueltos, rangos de
  horario mal formateados, abreviaturas/siglas, puntuación doble, markdown, largo excesivo) — corre
  sobre respuestas canónicas y nombres de tema al cargar, como advertencia.

### `llm.py` — el LLM local

`LocalLLM` maneja el modelo chico (Qwen2.5-3B por defecto) como un subproceso `llama-server` (HTTP,
no bindings embebidos), con `--parallel 2`: un slot de KV cache fijo para el reescritor y otro para
la respuesta, cada uno con su propio prefijo cacheado. Expone `rewrite_query()` (reescritura/
descomposición, ver arriba) y `stream_answer_multi()` (streaming de la respuesta final, con
`CJK_GRAMMAR` aplicada). El LLM de respuesta es **stateless**: no ve el historial de la charla,
solo la pregunta ya reescrita + el CONTEXTO del turno actual.

### `tts.py` — texto a voz

`PiperTTS` envuelve Piper (ONNX). `iter_sentences()` agrupa los tokens que van llegando del LLM en
oraciones completas; `StreamingSpeaker` sintetiza y reproduce cada oración en cuanto está lista, sin
esperar a que el LLM termine de generar el resto — esto es lo que permite que el primer audio salga
mucho antes que si se esperara la respuesta completa. `clean_for_speech()` quita markdown, emojis y
caracteres CJK residuales (red de seguridad detrás de `CJK_GRAMMAR`) antes de sintetizar.

### `config.py` — configuración tipada

Carga `config.toml` en dataclasses tipadas (una por sección: `AudioCfg`, `VadCfg`, `RagCfg`, etc.).
Si existe `calibration.toml`, sus valores pisan los de `config.toml` para los campos calibrados
(ver [Configuración](#configuración)).

## Configuración

Dos archivos, con roles distintos:

- **`config.toml`**: arquitectura y defaults documentados. Todo lo que NO es un umbral calibrado
  contra datos (modelos, tamaños, flags de arquitectura como `rag.hybrid_enabled`) vive acá, con
  comentarios explicando qué hace cada campo.
- **`calibration.toml`**: generado por `scripts/calibrate.py`, nunca a mano. Pisa `rag.min_score`,
  `rag.ambiguity_threshold` y `canonical.threshold` — los tres umbrales que dependen del contenido
  real (documentos + respuestas canónicas), no de la arquitectura. Si no existe, se usan los
  defaults de `config.toml` (documentados como "sin calibrar todavía").

**Parámetros PROVISORIOS** (atados al contenido de prueba de este POC, hay que recalibrar/revisar
al incorporar contenido real — ver `ONBOARDING_CONTENIDO.md`): los tres de `calibration.toml` de
arriba, más `rag.confidence_margin` (no calibrado contra datos: este corpus de prueba no tiene
todavía un caso real de dos documentos genuinamente en conflicto).

**Arquitectura estable** (no debería cambiar al incorporar contenido real): el pipeline completo
descripto arriba, el diseño de tres vías del gate de ambigüedad, el retrieval híbrido con RRF
(`rrf_k=60` es la constante estándar del paper, no un valor ajustado), el reranker cross-encoder, la
gramática anti-CJK, TTS por oración.

## Testing

Un solo comando corre todo y devuelve pass/fail (exit code), pensado como paso obligatorio antes de
cada commit:

```bash
python scripts/verify.py
```

Internamente corre, en orden:

1. **`scripts/calibrate.py --check`**: valida que los umbrales VIGENTES (los de `calibration.toml`)
   sigan sin producir casos peligrosos contra los datasets de calibración — sin recalibrar ni
   escribir nada. Un falso negativo (abstiene de más, o no matchea una canónica) es aceptable; un
   falso positivo con riesgo de alucinación no lo es. También corre el lint de voz sobre
   `doc_topics` y las respuestas canónicas.
2. **`scripts/eval.py`**: corre la suite completa de casos de comportamiento
   (`tests/eval_questions.yaml`) contra el pipeline real (sin mic/STT, texto directo), con
   aserciones automáticas por caso (abstención exacta, documento recuperado, presencia/ausencia de
   datos concretos, etc. — ver el docstring de `scripts/eval.py` para la lista completa de campos
   `expect_*`).

Otros scripts:

- **`scripts/calibrate.py`** (sin `--check`): barre umbrales candidatos contra
  `tests/calibration_questions.yaml`/`tests/calibration_canonical.yaml` y escribe
  `calibration.toml`. Prioriza precisión sobre recall (cero casos peligrosos, aunque eso implique
  abstenerse en preguntas legítimas que quedan cerca del límite).
- **`scripts/bench.py`**: benchmark de latencia end-to-end sin necesitar micrófono (genera la
  pregunta hablada con Piper).
- **`scripts/download_models.py`**: descarga todos los modelos necesarios.

## Limitaciones conocidas

1. **La restricción a un documento (`gate_docs`) pierde información complementaria de un segundo
   documento.** Si la pregunta real necesita datos de dos documentos a la vez y uno domina con
   confianza, el otro se descarta aunque tuviera algo útil que agregar.
2. **El reescritor está cerca del límite de lo que los few-shots pueden lograr confiablemente para
   un modelo de 3B.** Se observaron casos donde omite una palabra clave de la entrada original,
   formatea números de forma inconsistente (dígitos vs. escritos), o resuelve un follow-up vago al
   día/dato equivocado. El **orden** de los few-shots importa tanto como su contenido: uno puesto al
   final de la lista puede actuar como el patrón "por defecto" que el modelo copia ante una entrada
   nueva sin relación clara (recency bias).
3. **`recall@5` no tiene todavía evidencia con documentos reales** — el corpus de prueba de este POC
   (3 archivos, 14 chunks) es mínimo a propósito, no representa la escala ni la ambigüedad léxica de
   contenido real.
4. **El gate de aclaración (`gate_docs`, camino "pedir aclaración") está verificado solo con datos
   sintéticos** — este corpus no tiene un caso real de dos documentos genuinamente en conflicto.
5. **Un modo de falla del LLM en preguntas compuestas**: cuando dos sub-preguntas comparten un
   CONTEXTO parecido, a veces subestima o no extrae un dato presente en una de las partes, aunque
   esté claramente ahí.
6. **Las latencias medidas en este repo son de una RTX 3050 4GB, no del hardware de destino final**
   (ver `DEPLOY_ORIN.md` para el plan de migración y comparación).

El detalle completo de cómo se encontró cada una (con trazas, reproducciones y lo que se descartó
en el camino) está en `BITACORA.md`.

## Estructura del repo

```
voice/
  audio.py      mic/parlantes (PulseAudio vía soundcard)
  wakeword.py   openWakeWord (ONNX)
  vad.py        Silero VAD + segmentador de frases
  stt.py        faster-whisper (GPU)
  rag.py        chunking + retrieval híbrido (denso + BM25/RRF) + reranker
  rewrite.py    reescritura de consulta + descomposición de preguntas compuestas
  canonical.py  respuestas canónicas (FAQ, texto fijo) -- matching + rotación de variantes
  llm.py        llama-server (subproceso HTTP), streaming
  guardrails.py reglas deterministas (abstención, ambigüedad, anti-CJK, "repetí")
  tts.py        Piper, streaming por frase
  pipeline.py   orquesta todo (clase Assistant)
  config.py     config.toml + calibration.toml -> dataclasses tipadas
scripts/
  download_models.py   descarga todos los modelos
  bench.py              benchmark end-to-end sin mic
  eval.py                corre tests/eval_questions.yaml contra el LLM/RAG real
  calibrate.py           calibra rag.min_score/ambiguity_threshold/canonical.threshold
  verify.py              un solo comando: calibrate.py --check + eval.py
docs/                     documentos de ejemplo para el RAG (reemplazar por los reales)
tests/
  eval_questions.yaml         casos de comportamiento (dev/validación) para scripts/eval.py
  calibration_questions.yaml  set etiquetado para calibrar rag.min_score/ambiguity_threshold
  canonical_answers.yaml      FAQ de respuestas canónicas -- contenido de PRUEBA, reemplazar
  calibration_canonical.yaml  set etiquetado para calibrar canonical.threshold
config.toml         arquitectura + defaults documentados
calibration.toml    generado por scripts/calibrate.py -- pisa los umbrales calibrados
BITACORA.md         historial de decisiones, bugs reales y experimentos descartados
ONBOARDING_CONTENIDO.md  checklist para incorporar documentos/respuestas/preguntas reales
DEPLOY_ORIN.md           checklist de migración a la Jetson Orin
```

## Contenido real y despliegue

- **Incorporar documentos, respuestas canónicas y preguntas de calibración reales**: ver
  [`ONBOARDING_CONTENIDO.md`](ONBOARDING_CONTENIDO.md) — formato esperado, cómo redactar para voz,
  cuántas preguntas de calibración por categoría, orden de pasos y criterios mínimos de aceptación.
- **Migrar a la Jetson Orin**: ver [`DEPLOY_ORIN.md`](DEPLOY_ORIN.md) — compilación con CUDA para
  Jetson, ajustes de memoria, primera corrida de comparación, experimento pendiente 3B vs 7-8B.

## Notas / gotchas de esta máquina

- `openwakeword>=0.6` depende de `tflite-runtime`, que no tiene wheel para `cp312` → se instala con
  `--no-deps` y se fuerza `inference_framework="onnx"`.
- Si `llama_cpp` se importa **después** de otro paquete que trae su propia `libggml*.so` (queda en
  `site-packages/lib64/`), falla con `undefined symbol: gguf_init_from_file_ptr`. Por eso
  `voice/__init__.py` importa `llama_cpp` primero, antes que nada más.
- `faster-whisper` en GPU necesita cuBLAS/cuDNN 12; se instalan como wheels de pip
  (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) pero no quedan en el loader path, así que
  `voice/stt.py` los precarga a mano con `ctypes.CDLL(..., RTLD_GLOBAL)`.
- `soundcard` 0.4.6 no expone `SoundcardRuntimeWarning` como clase; el filtro de warnings usa el
  mensaje en texto en vez de la clase.
