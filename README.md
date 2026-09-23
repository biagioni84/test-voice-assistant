# Asistente de voz local (POC)

Pipeline: **wake word → VAD → Whisper turbo → RAG → LLM → Piper**, todo local, sin llamadas a APIs externas.

## Hardware verificado (esta máquina)

- CPU: i5-12450H, 12 hilos, AVX2/FMA — suficiente para VAD, wake word, embeddings y el LLM en CPU.
- GPU: RTX 3050 **4 GB VRAM** (con ~1.3 GB ya tomados por el sistema/WSL, quedan ~2.7 GB libres). Es la
  restricción real del proyecto: alcanza holgado para Whisper, no para un LLM 7-8B en GPU.
- RAM: 31 GB — sobra para todo.
- Audio: WSLg expone mic/parlantes por PulseAudio (`soundcard` los usa sin configuración extra).

## Decisiones y por qué

- **STT**: `faster-whisper` (CTranslate2) con `large-v3-turbo` en GPU, `int8_float16` (~1 GB VRAM).
  Transcribe una frase de ~3s en <1s.
- **Wake word / VAD**: `openWakeWord` (backend ONNX, no tflite — no hay wheel de `tflite-runtime` para
  Python 3.12) + `pysilero-vad` (Silero VAD vía ONNX). Ambos corren en CPU, livianos.
- **LLM**: `llama-cpp-python` **compilado con soporte CUDA** (el wheel CPU-only por defecto daba una
  latencia horrible — ver abajo) con **Qwen2.5-3B-Instruct Q4_K_M**, no el 7-8B pedido originalmente.
  Medido en esta máquina con `scripts/bench.py` (tiempo "fin de voz → 1er audio" = lo que espera el
  usuario desde que deja de hablar hasta que arranca a sonar la respuesta):

  | Configuración | tok/s | Fin de voz → 1er audio | VRAM usada | Margen libre |
  |---|---|---|---|---|
  | 3B, CPU puro (wheel por defecto) | 5.2 | 5.7s | ~1GB (solo Whisper) | mucho |
  | 3B, 20/36 capas en GPU | 15.4 | 1.9s | ~2.9/4.0GB | ~1.1GB |
  | **3B, 28/36 capas en GPU (default)** | **20.9** | **2.0s** | **~3.3/4.0GB** | **~0.7GB** |
  | 3B, todas las capas en GPU (`-1`) | 42.5 | 1.1s | ~3.9/4.0GB | ~0.2GB, arriesgado |
  | 7B, offload seguro (8-12/28 capas) | 5.9-6.4 | 3.8-4.4s | ~3.4-3.8/4.0GB | ~0.3-0.7GB |
  | 7B, todas las capas en GPU (`-1`) | 2.7 | 8.1s | no entra en 4GB | — |
  | Qwen3-4B, offload seguro (20/36 capas) | 9.5 | 3.0s | ~3.4/4.0GB | ~0.7GB |

  Lo que se sentía "muy lento" era el LLM corriendo en CPU (wheel `llama-cpp-python` por defecto no
  trae CUDA). Compilarlo con `CMAKE_ARGS="-DGGML_CUDA=on"` (hay `nvcc` y el CUDA toolkit instalados) y
  offloadear capas del 3B a la GPU da ~4x más throughput y ~3x menos latencia. El 7B directamente no
  entra en 4GB de VRAM: al forzar todas las capas, el driver empieza a "spillear" a RAM compartida y
  **todo** se vuelve más lento, incluso Whisper (que comparte la misma GPU). Con offload parcial "seguro"
  el 7B sigue siendo 2-3x más lento que el 3B en esta GPU. Se probó también **Qwen3-4B-Instruct-2507**
  (generación más nueva, mejor calidad por parámetro que Qwen2.5): en su punto seguro de offload da
  mejores respuestas que el 3B pero a menos de la mitad de velocidad — mismo patrón, esta GPU de 4GB no
  da para más sin sacrificar la sensación de "conversación fluida". **Decisión: nos quedamos con el 3B**
  acá; el salto de calidad (7-8B, o Qwen3 en ese rango) se guarda para cuando corra en hardware con más
  VRAM/ancho de banda (ver sección de la Jetson AGX Orin más abajo).
- **RAG**: sin vector DB — `fastembed` (ONNX, embeddings multilingües) + coseno en numpy contra un
  índice cacheado en `.cache/`. Alcanza de sobra para decenas de miles de chunks; si el corpus crece
  mucho, cambiar `Retriever._matrix` por FAISS o sqlite-vec sin tocar la interfaz.
- **TTS**: Piper, voz `es_AR-daniela-high`. Las respuestas del LLM se cortan por frase (`iter_sentences`)
  y cada frase se sintetiza y reproduce mientras el LLM sigue generando, para no esperar la respuesta
  completa antes de hablar. `clean_for_speech` saca markdown y emojis antes de sintetizar (un modelo
  metió un 😄 en una respuesta durante las pruebas; sin este filtro Piper lo lee mal o lo salta feo).
- **Documentos del RAG**: en `docs/` hay 3 archivos **genéricos de ejemplo** (oficina, soporte técnico,
  RR.HH.) solo para poder probar la recuperación. Reemplazar por los documentos reales cuando estén.

## Instalar

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
uv pip install -p .venv/bin/python --no-deps openwakeword

# llama-cpp-python CON CUDA (el wheel normal de pip es CPU-only y va ~4x más lento). Tarda varios
# minutos en compilar. 86 = compute capability de RTX 30xx; para otra GPU, ajustar ese número.
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86" FORCE_CMAKE=1 \
    uv pip install -p .venv/bin/python --no-binary llama-cpp-python llama-cpp-python

.venv/bin/python scripts/download_models.py   # ~5 GB: Whisper turbo, LLM, voz Piper, embeddings
```

Si no hay GPU o no querés esperar la compilación, se puede usar el wheel CPU-only en su lugar
(`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary=llama-cpp-python`),
pero entonces conviene bajar `n_gpu_layers` a `0` en `config.toml` — con un valor >0 y sin CUDA
compilado, `llama-cpp-python` ignora el offload y corre todo en CPU igual, solo que sin avisar.

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
.venv/bin/python scripts/bench.py --llm-file Qwen2.5-7B-Instruct-Q4_K_M.gguf   # comparar con el 7B
```

Toda la config vive en `config.toml` (umbrales de VAD, modelo de wake word, voz de Piper, etc.).

## Plan para portar a Jetson AGX Orin

Estimación (sin hardware disponible para medir todavía); en resumen: los 64GB
unificados sacan la presión de VRAM que tenemos acá (el 7-8B pasa a ser viable), pero el ancho de banda
de memoria real de la Orin (~150-180GB/s medido, vs 204.8GB/s de spec) está en la misma liga que el de
esta RTX 3050 — no es un salto mágico de tok/s por sí solo. Mejoras concretas, de más a menos impacto:

**Velocidad**
1. `nvpmodel -m 0` + `jetson_clocks` al arrancar — sin esto, Jetson corre en un modo de energía reducido
   por defecto y se pierde rendimiento gratis.
2. `faster-whisper` **no tiene wheels aarch64+CUDA** (los de pip son CPU-only ahí) → portar `voice/stt.py`
   a **whisper.cpp compilado con CUDA** (soporte aarch64/Jetson maduro) o a TensorRT para el encoder.
   Sin esto, Whisper corre en CPU en la Orin y probablemente sea el cuello de botella nuevo.
3. LLM: recompilar `llama-cpp-python` con `CMAKE_CUDA_ARCHITECTURES=87` (Orin es compute capability 8.7,
   no 8.6) y usar `n_gpu_layers=-1` sin miedo — con 64GB no hace falta el balance fino que hicimos acá.
   Probar también `flash_attn=True` (más velocidad y memoria de KV cache más chica).
4. Si con eso no alcanza: migrar el LLM a **TensorRT-LLM** (soporte oficial para AGX Orin desde
   JetPack 6.1). Bien más rápido que llama.cpp — benchmarks 2026 muestran un 8B en Q4/FP8 a ~41 tok/s
   con concurrencia 1, contra los ~15-20 tok/s que estimamos para llama.cpp puro. Cuesta más setup
   (compilar desde la rama `-jetson` del repo).
5. Para no pelear wheels ARM faltantes a mano (nos pasó dos veces acá, con `tflite-runtime` y con
   `onnxruntime-gpu`+CUDA13): armar el stack sobre **jetson-containers** (dusty-nv), que ya tiene
   imágenes probadas de llama.cpp y whisper/faster-whisper para JetPack 6.x en AGX Orin 64GB.

**Precisión de detección**
1. **Wake word**: ya implementado en este repo (`voice/wakeword.py` + `config.toml` →
   `[wakeword].vad_threshold` y `.noise_suppression`) — openWakeWord exige que Silero VAD confirme voz
   real antes de contar una activación, y aplica supresión de ruido SpeexDSP. Corregir el `threshold` en
   el lugar real de instalación (el valor 0.5 por defecto es de fábrica, no calibrado a tu sala).
   Si los falsos positivos/negativos persisten, entrenar una wake word **propia** con
   [easy-oww](https://github.com/pjdoland/easy-oww) usando grabaciones del entorno real (incluye
   negativos adversariales para bajar falsos positivos).
2. **VAD**: recalibrar `speech_threshold` / `end_silence_ms` en `config.toml` en el sitio real de
   instalación. Una Jetson con cooler activo tiene un piso de ruido que la laptop (pasiva) no tiene.
3. **Hardware de micrófono — el cambio de mayor impacto**: en un dispositivo con parlantes reales (no
   auriculares como en esta prueba por WSLg), sin cancelación de eco el propio audio de Piper puede
   filtrarse al micrófono y disparar el wake word o cortar mal el VAD. La solución estándar en proyectos
   de Jetson es un **array de micrófonos far-field USB tipo ReSpeaker** (v2.0/v3.0/XVF3800): traen AEC,
   beamforming, supresión de ruido y de-reverberación en hardware, plug-and-play con Jetson. Mucho más
   robusto que resolverlo en software con el mic integrado.
4. **STT**: con el margen extra de la Orin, subir `beam_size` de 1 a ~5 (mejor precisión, costo
   modesto) y pasar un `initial_prompt` con vocabulario del dominio (nombres propios, términos de los
   documentos del RAG) para sesgar el reconocimiento. Evaluar volver de `turbo` a `large-v3` si el
   presupuesto de latencia lo permite — turbo sacrifica algo de precisión por velocidad.

**¿Y un modelo más grande que 8B en la Orin?** La generación de tokens (batch=1) está limitada por
ancho de banda de memoria, no por cómputo: tok/s ≈ ancho de banda / tamaño leído por token. Con eso:

| Modelo (Q4_K_M) | Tamaño en disco | tok/s (llama.cpp, estimado) | tok/s (TensorRT-LLM, medido/estimado) |
|---|---|---|---|
| Qwen3-8B (dense) | 5.0 GB | ~15-20 | **41** (benchmark real, concurrencia 1) |
| Qwen2.5-14B (dense) | 9.0 GB | ~9-12 | ~20-25 (estimado) |
| Qwen3-30B-A3B (**MoE**, 8/128 expertos activos) | 18.6 GB | ~20-30 (estimado) | **61** (benchmark real) |

Un modelo **dense** más grande paga el precio esperado: cada token lee todos los parámetros, así que
14B ya se siente perceptiblemente más lento que 8B, y un 32B dense (~5-6 tok/s con llama.cpp) rompe la
sensación de conversación fluida — aunque "entre" sobrado en 64GB, la latencia no da.

La jugada más inteligente es ir a **MoE** en vez de a "más denso": `Qwen3-30B-A3B` activa solo 8 de 128
expertos por token (~3B efectivos, de ahí el "A3B"), así que el costo de ancho de banda por token es
parecido a un modelo de ~3GB — no a 30GB. El benchmark real lo confirma: 61 tok/s con TensorRT-LLM,
**más rápido que el 8B dense** pese a tener 4x más parámetros totales. Con llama.cpp puro (sin
TensorRT-LLM) el soporte CUDA de MoE ya es maduro (Mixtral, DeepSeek-MoE), así que debería rendir
razonablemente aunque falta confirmar con hardware real.

**Recomendación**: 8B como default simple; si de todos modos se monta TensorRT-LLM, ir directo a
`Qwen3-30B-A3B` en vez de a un 14B dense (más capacidad, igual o mejor velocidad). No conviene pasar de
ahí para un asistente de *voz* — las respuestas están ancladas al RAG (contexto corto, 1-3 frases), y
ese caso de uso probablemente no necesita más "inteligencia" que la que ya da un 8B/30B-A3B; la latencia
sí se nota en cada intercambio hablado. Antes de fijar el default, correr `scripts/bench.py` con ambos
en la Orin real y comparar en preguntas típicas del RAG.

## Bug real: el RAG perdía contexto en follow-ups cortos

Probando con micrófono real (no con las preguntas fijas de `bench.py`) apareció esto en una
conversación real:

```
🗣 ¿A qué hora abre la oficina el domingo?  → "no abre el domingo" ✓
🗣 y eso                                     → "no abre el domingo" ✓ (pero por casualidad)
🗣 Te pregunté el sábado a qué hora abre.    → "La oficina no abre los sábados" ✗ (falso)
🗣 ¿Cómo que no abre los sábados?            → "abre de 10 a 1 los sábados" ✓ (se contradice a sí mismo)
```

**Causa raíz** (confirmada recuperando estas queries a mano contra el índice): un follow-up corto como
`"y eso"` o `"y los sábados"` embebido *solo* no trae señal suficiente — el coseno contra el chunk
correcto queda debajo de `min_score` y no se inyecta ningún CONTEXTO. Sin contexto fresco, el LLM
improvisa con lo que dijo antes en la charla (que puede ser incorrecto) en vez de admitir que no sabe.
Un segundo problema, más sutil: un doc apenas relacionado (`soporte_tecnico.md` a 0.33 para "¿por qué
demoras tanto en responder?") a veces pasaba el umbral y el modelo intentaba forzarlo igual.

**Fix** (`voice/pipeline.py: Assistant._retrieve_sticky`): se prueba primero la pregunta sola contra el
índice. Si no trae nada **y** es corta (≤4 palabras — heurístico para distinguir un follow-up tipo "y
los sábados" de una pregunta completa que simplemente no tiene doc relacionado), se reusan **los hits
tal cual del último turno que sí tuvo contexto** — sin reconstruir una query combinada ni volver a
embeder nada. Guardar los `Hit` ya resueltos (no la pregunta) en vez de "la pregunta anterior" importa:
si el turno inmediatamente anterior fue en sí mismo un comentario sin tema (p.ej. "¿por qué demoras
tanto?"), ese turno no pisa el "último contexto bueno" — así un comentario de paso en medio de la
conversación no corta la continuidad del tema real. También se subió `min_score` a 0.35 (filtra matches
débiles como el de soporte_tecnico.md) y se reforzó el `system_prompt` para que el CONTEXTO gane siempre
por sobre lo que el propio modelo dijo antes, y para que ignore CONTEXTO que no tenga que ver con la
pregunta en vez de forzarlo.

Con esto, la misma conversación:

```
🗣 ¿A qué hora abre la oficina el domingo?  → "no abre el domingo" ✓
🗣 y eso                                     → "no abre el domingo" ✓ (ahora por contexto heredado, no azar)
🗣 Te pregunté el sábado a qué hora abre.    → "abre a las 10 de la mañana los sábados" ✓
🗣 ¿Por qué demoras tanto en responder?      → sin contexto (correcto, no hay doc relacionado)
🗣 y los sábados                             → "los sábados abre de diez a una de la tarde" ✓ (recuperó
                                                el tema real, saltando el comentario sin tema del medio)
```

**Límite que queda, y no es un bug de RAG**: en el turno de "¿por qué demoras tanto en responder?", el
LLM (3B) a veces igual menciona el horario de la oficina aunque no se le haya dado ningún CONTEXTO ese
turno — lo arrastra de su propio historial de charla por pura continuidad conversacional, un sesgo
conocido de modelos chicos hacia "seguir el tema" en vez de notar que cambió. No hay mucho margen para
arreglar esto con prompting en un 3B; un modelo más grande (ver la sección de la Orin) maneja mejor la
atención multi-turno.

**SUPERADO (22/09)**: el fallback de sticky-hits de esta sección se eliminó por completo y se
reemplazó por reescritura de consulta -- ver la siguiente sección. El diagnóstico con
`--show-chunk-text` (ver más abajo) mostró que sticky-hits no era el problema real: el chunk
correcto casi siempre se recuperaba bien, el problema era que la pregunta tal como la decía el
usuario no era autónoma.

## Reescritura de consulta (reemplaza a sticky-hits)

Con las 5 variantes de `cambio_de_dia_en_followup` en `tests/eval_questions.yaml`, sticky-hits daba
2/5 bien, 1/5 seguía hablando de domingo, y 2/5 abstenían de más (esas formulaciones de más de 4
palabras no calificaban para el heurístico de "corto"). El diagnóstico con `--show-chunk-text`
confirmó que el chunk correcto (con sábados *y* domingos) casi siempre llegaba bien al LLM -- el
problema no era de retrieval, era que la pregunta tal como la dice el usuario ("Te pregunté el sábado
a qué hora abre.") mezcla una frase meta ("te pregunté") con el dato real, y esa mezcla confundía
tanto al retrieval como a la generación.

**Cómo funciona ahora** (`voice/rewrite.py`, `voice/llm.py`):

1. **Reescritura**: si hay historial (el primer turno de la charla no paga esto), una llamada extra
   al mismo Qwen2.5-3B (`temperature=0`, `max_tokens=40`, mismo `logit_bias` anti-CJK) reescribe la
   pregunta como autónoma, usando los últimos `rewrite.history_turns=2` turnos. Prompt + 5 ejemplos
   few-shot (referencia simple, corrección explícita, cambio de tema total, meta-pregunta sobre el
   asistente, y una continuación vacía tipo "y eso" que necesitó su propio ejemplo -- sin él, el
   modelo devolvía basura tipo "y ahí" que no pasaba la validación).
2. **Validación barata**: si la reescritura no termina en "?" o tiene más de 20 palabras, se usa la
   pregunta original tal cual (probablemente no era una pregunta para reescribir -- charla social,
   insultos, etc.).
3. **Retrieval**: se hace *solo* con la pregunta reescrita. No hay más fallback de sticky-hits, no se
   reusan chunks de turnos anteriores.
4. **Abstención**: si la reescrita no supera el umbral, corta igual que antes (`voice/guardrails.py`),
   pero con un mensaje distinto cuando hay historial: *"No lo encontré, ¿me lo preguntás de otra
   forma?"* en vez del genérico *"No tengo información sobre eso."*
5. **LLM de respuesta**: recibe *solo* la pregunta reescrita + el CONTEXTO de este turno -- ya **no
   ve el historial de la charla en absoluto**. Antes recibía los últimos turnos además del CONTEXTO;
   sacarlo evitó que el LLM se "ancle" en el tema de turnos anteriores (la causa original del bug).

**Resultado** (`tests/eval_questions.yaml`, casos `type: rewrite` prueban solo el reescritor,
aislado del resto):

| Caso | Antes (sticky-hits) | Después (reescritura) |
|---|---|---|
| `cambio_de_dia_en_followup` (5 variantes) | 2/5 bien, 1/5 mal, 2/5 abstenían de más | **5/5** |
| `followup_corto_mismo_tema`, `comentario_sin_tema_no_corta_continuidad` | pasaban | siguen pasando |
| `identidad_inventada`, `_nombre_comun`, `chiste_generico` | 5/5, 3/3, 5/5 | sin regresión |
| `retrieval_ambiguo_dos_docs` (bug de fidelidad al contexto, no tocado) | 0/3 | sigue 0/3, sin cambios (esperado) |

**Costo**: la reescritura agrega una llamada al LLM por turno con historial. Medido con
`scripts/eval.py`: quando corre de verdad, p50 ≈ 880ms, p95 ≈ 970ms (n=22 llamadas reales sobre 55
turnos evaluados). El primer turno de cualquier charla no la paga (0ms, sin historial que reescribir).

**Todavía no resuelto** (bug de "fidelidad al contexto", ver `retrieval_ambiguo_dos_docs` y
`pregunta_compuesta` en `tests/eval_questions.yaml`): cuando el CONTEXTO mezcla información de dos
documentos relacionados (horario de oficina + política de trabajo remoto), el modelo a veces inventa
una política que combina mal las dos cosas, incluso con el chunk correcto presente. Este bug no
depende del historial de la charla (pasa en preguntas de un solo turno), así que la reescritura no lo
toca -- es un problema de capacidad del modelo al leer el CONTEXTO, no de qué pregunta se le hace.
Próximos pasos, en orden: mejorar el chunking (separar hechos que hoy están mezclados en un mismo
chunk) → un lookup estructurado de horarios en vez de texto libre → si nada de eso alcanza, un modelo
más grande (7-8B o el MoE de la sección de la Orin).

## Latencia del reescritor: dos callejones sin salida, y una migración que sí funcionó

Antes de perseguir un número de latencia absoluto en esta laptop (RTX 3050, no es el hardware de
destino -- la Orin), se midieron métricas portables con `verbose=True` de llama.cpp:

- La llamada al reescritor (~616 tokens de prompt, ~10-30 de salida) se reparte en **793ms de
  prefill (616 tokens) vs 319ms de decode (~11 tokens)** -- el prefill domina, no la generación.
- **9 de 36 capas del LLM están en CPU** (`n_gpu_layers=28`) -- afecta tanto prefill como decode.
- **569 de esos 616 tokens (92%) son el prefijo estático** (system prompt + few-shot de
  `voice/rewrite.py`), idéntico en cada llamada.

Con ese 92% estático, cachearlo parecía el punto obvio. Se probaron dos caminos con el LLM **embebido**
(`llama_cpp.Llama`), y **ninguno dio una mejora neta**:

1. **`Llama.save_state()`/`load_state()`**: sí evita reprocesar el prefijo (prefill bajó de 616 a 54
   tokens), pero `load_state()` en sí mismo cuesta 350-760ms -- más de lo que ahorra. Confirmado
   también con el modelo 100% en GPU (`n_gpu_layers=-1`), mismo resultado: es una limitación de cómo
   `llama-cpp-python` serializa el estado, no del hardware. **Revertido.**
2. **Condensar el system prompt** (238→131 tokens): en un test aislado con texto escrito a mano
   parecía funcionar -- pero el test tenía trampa (coincidía exacto con el few-shot). Con texto real
   del LLM de respuesta, la versión condensada fallaba el caso "y eso". **Revertido.**

**La causa raíz real**: el reescritor y el LLM de respuesta compartían una sola instancia embebida con
un único contexto. Alternar entre sus dos prompts (muy distintos entre sí) invalidaba el cacheo de
prefijo automático de llama.cpp en cada turno -- no es algo que se arregle cacheando "a mano" con una
sola instancia, hace falta que cada rol tenga su propio contexto.

**La migración que funcionó: llama-server con `--parallel 2`.** En vez del binding embebido, el LLM
corre como subproceso (`voice/llm.py`, HTTP a `127.0.0.1:8811`), con 2 slots de KV cache
independientes -- `id_slot=0` fijo para el reescritor, `id_slot=1` para la respuesta -- y
`cache_prompt: true` en cada request. Cada slot mantiene su propio prefijo cacheado sin que el otro lo
toque. Compilado desde el repo de llama.cpp con CUDA (no es un paquete de pip, ver "Instalar" abajo).

Medido en aislado: la primera llamada a un slot procesa el prompt completo (~800ms); las siguientes
con el mismo prefijo estático solo reprocesan la cola dinámica (~25-300ms). Bien por debajo del
objetivo original de <100ms de prefill en la mayoría de los casos.

**Tres bugs reales aparecieron en el camino, cada uno con su propio diagnóstico:**

1. **`logit_bias` no escala en el server.** El enfoque anti-CJK del round anterior (`-100` sobre
   ~31.000 tokens del vocabulario) funcionaba bien embebido, pero vía HTTP el campo `logit_bias` del
   request no escala: medido, pasar de 15.000 a 31.000 entradas **cuadruplica** la latencia (~1000ms
   de overhead extra, casi seguro una búsqueda no indexada del lado del servidor). Reemplazado por una
   **gramática GBNF** (`voice/guardrails.py: CJK_GRAMMAR`) que restringe los caracteres válidos por
   posición -- medido: mismo tok/s con o sin gramática, porque el compilador de gramáticas arma un
   autómata en vez de recorrer una lista plana. Bonus: ya no hace falta cargar el modelo (ni siquiera
   en modo `vocab_only`) solo para calcular el bias.
   - *Bug de sintaxis en el camino*: la primera versión de la gramática usaba escapes `\x{XXXX}`
     (sintaxis PCRE/Python) en vez de `\uXXXX` (sintaxis real de GBNF) -- el parser los toleraba sin
     tirar error pero la gramática no restringía nada de verdad, así que una fuga de CJK pasó igual en
     una corrida completa antes de notarlo. Corregido y verificado con 0 caracteres CJK en 43 casos.
2. **"y eso" seguía siendo frágil incluso con el prefijo exacto del few-shot.** Después de arreglar 1,
   "y eso" todavía fallaba -- diagnosticado a fondo (aislando grammar, cache_prompt, id_slot uno por
   uno) hasta confirmar que era el modelo mismo, de forma estable y no por empate de logits (probado
   con distintas seeds y algo de temperature, mismo resultado siempre). El LLM de respuesta genera una
   frase ligeramente distinta a la del ejemplo few-shot turno a turno (aunque `temperature=0`, porque
   el CONTEXTO recuperado varía un poco), y el 3B no generalizaba de forma confiable a esa variación.
   **Fix**: en vez de seguir puliendo el prompt, "referencia vacía" ("y eso", "y ahí", "eso mismo", "lo
   mismo") se resuelve con una regla determinista (`voice/rewrite.py: is_empty_reference` +
   `resolve_empty_reference`) que devuelve directamente la última pregunta del historial, sin llamar al
   LLM -- más simple, 100% confiable, y de paso más rápido para el patrón de follow-up más común.

**Un "bug" que resultó no serlo -- vale la pena documentarlo por la metodología**: al perseguir el
punto 2, en el camino se sospechó (equivocadamente) que NO borrar el KV cache de los slots entre
conversaciones distintas causaba contaminación cruzada (el matching de prefijo "mezclando" contenido de
una charla vieja sin relación). `reset_conversation()` llegó a borrar los dos slots por las dudas. Pero
un **experimento controlado** lo descartó: correr una charla B (a) desde cero, (b) después de otra
charla A con `cache_prompt: true`, (c) igual pero `cache_prompt: false` -- **las tres dieron texto
idéntico**. El matching de prefijo de llama-server compara tokens exactos y solo reusa lo que coincide
byte a byte; no hay mecanismo por el que pueda mezclar contenido de charlas distintas. La causa real de
"y ahí" era la del punto 2. Sacado el borrado (no aporta nada demostrado y cuesta latencia real, ver
abajo).

**Timings por rol, frío vs tibio** (medido directo del campo `timings` de llama-server, no estimado):

| | prompt_n | prompt_ms (frío) | prompt_ms (tibio) | predicted_n | predicted_ms |
|---|---|---|---|---|---|
| Reescritor | 570 | 370ms | **27ms** (cache_n=569) | 13 | ~300-420ms |
| Respuesta | 266 | 173ms | **27ms** (cache_n=265) | 16 | ~400-420ms |

Hallazgo importante: con caché tibio el prefill cae ~93%, pero el **decode no se mueve** (sigue en
~300-420ms) -- el cacheo solo ahorra reprocesar el prompt, no generar tokens nuevos, que es secuencial
sí o sí (25-30ms/token, fijo por la velocidad del modelo). Por eso el objetivo original de "<100ms de
prefill" no se traduce en un total tan bajo: una vez tibio el cuello de botella pasa a ser el decode,
no el prefill, y eso no depende del cacheo de prompt.

**Resultado final** (43 casos, `tests/eval_questions.yaml`, sin borrado de slots): **p50≈446ms,
p95≈743ms** cuando el reescritor corre (bajó de ~870ms, mejoró más todavía al sacar el borrado
innecesario), sin regresiones en identidad/chiste/cambio-de-día, y "y eso" ahora resuelve en 0ms (no
llama al LLM). Un caso nuevo sigue fallando (`comentario_sin_tema_no_corta_continuidad`: el LLM de
respuesta ignora un chunk con score muy superior -0.81 vs 0.49- y responde con el menos relevante) --
es la misma familia de "bug A" (fidelidad al contexto) ya documentada como fuera de alcance, no una
regresión de esta migración.

**Bug real encontrado en el camino, no relacionado con la latencia**: `Assistant.run()` nunca
reseteaba `self.llm.history` entre ventanas de conversación separadas (solo `scripts/eval.py` lo
hacía). En uso real esto significa que una charla nueva podía arrastrar el historial de una charla
previa sin relación, horas antes. Arreglado: `run()` llama a `reset_conversation()` al cerrarse cada
ventana de conversación.

**Instalar llama-server con CUDA** (no es un paquete de pip, se compila desde el repo de llama.cpp):
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build-cuda -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target llama-server -j"$(nproc)"
```
(86 = compute capability de RTX 30xx; ajustar si es otra GPU.) `config.toml` → `[llm].server_bin`
apunta al binario resultante -- por defecto asume `/home/pbdev/llama.cpp/build-cuda/bin/llama-server`,
ajustar a donde se haya compilado.

## Dev vs. validación en scripts/eval.py

Los casos que se usaron para ajustar el prompt/few-shot de `voice/rewrite.py` (p.ej.
`cambio_de_dia_en_followup`) miden regresión, no generalización -- el prompt se ajustó *mirando
exactamente esos casos*. Se agregó el campo `split: validation` para un grupo separado de casos
holdout, con formulaciones que no se parecen a ningún ejemplo few-shot, agregados *después* de
terminar de ajustar el prompt y sin tocarlos más (correr `scripts/eval.py --split validation`).

**Resultado (4 casos nuevos, formulaciones nunca vistas en los few-shot):**

| Formulación nueva | Resultado |
|---|---|
| "¿y ahí?" (variante de "eso" con otra palabra) | No se resolvió sola, pero abstuvo con seguridad (no alucinó) y se auto-corrigió en el siguiente turno |
| "¿Y lo mismo pero para el sábado?" (patrón nuevo) | No generalizó -- abstuvo |
| "dale, ¿y el otro día que te dije?" | Caso mal diseñado (presupone un "otro día" que nunca se mencionó) -- no es señal limpia |
| "Uy perdón, quise decir el sábado." (corrección sin "no"/"te pregunté") | ✓ Generalizó bien |

**2/4 generalizan, 2/4 no** -- el prompt captura bien el *concepto* que está explícito en el texto de
la instrucción (`"eso", "ahí"` se mencionan literalmente) pero no generaliza a formas nuevas no vistas
ni en la instrucción ni en los ejemplos ("lo mismo pero para X"). Importante: en ningún caso de los que
falló, alucinó -- siempre cayó a la abstención segura ("No lo encontré, ¿me lo preguntás de otra
forma?"). Es una limitación de cobertura, no de seguridad.

## Chunking más fino + gate de ambigüedad cross-doc

**Chunking** (`voice/rag.py`): antes, `_chunk()` combinaba párrafos consecutivos de un mismo
documento hasta `chunk_chars=500`, así que el título, la intro y el horario de `oficina.md` quedaban
en un solo chunk (474 caracteres, entraban justos). Ahora **un chunk = un párrafo = un hecho**, sin
combinar entre sí (sí se sigue partiendo un párrafo individual que por sí solo supere `chunk_chars`).
Cada chunk además lleva el título del documento (`_extract_title()`, del primer heading markdown):
`oficina.md` pasó de 2 chunks a 5, cada uno con `"Oficina: <hecho>"`.

**Gate de ambigüedad** (`voice/guardrails.py: ambiguous_docs()`): si los dos documentos *distintos*
con mejor score están a menos de `rag.ambiguity_threshold` (0.10) de diferencia, se corta sin llamar
al LLM y se pide aclaración nombrando los dos temas (`clarify_reply()`, usa `rag.doc_topics` del
config para los nombres). Mismo mecanismo que la abstención (`_fixed_reply` en `voice/pipeline.py`,
refactorizado para servir a los dos casos).

**Resultado en `retrieval_ambiguo_dos_docs`** (el bug de "inventa una política mezclando dos docs"):
2 de 3 variantes ahora piden aclaración en vez de alucinar. La 3ª (`oficina.md` domina con
score-gap=0.21) no es ambigüedad real -- es el modelo malinterpretando un chunk claro y dominante, un
problema distinto que el gate no puede resolver (es "bug A" puro, ver más abajo).

**Límite real encontrado, no resuelto** (dos manifestaciones del mismo problema): el coseno de
similaridad de embeddings no distingue "la pregunta toca dos temas de verdad" de "un chunk sin
relación tiene vocabulario parecido por casualidad".
- `pregunta_compuesta` (una sola pregunta, sobre un solo doc) dispara el gate de ambigüedad por
  error: su score-gap real (`oficina.md` vs `recursos_humanos.md` = 0.04) es **idéntico en magnitud**
  al de una ambigüedad real (`retrieval_ambiguo_dos_docs` variante 2 = 0.04) -- no hay umbral que
  separe limpiamente ambos casos con esta señal sola.
- El chunking más fino además **destapó** un false-positive nuevo en `comentario_sin_tema_no_corta_continuidad`:
  el chunk aislado de "horario de atención" de `soporte_tecnico.md` (antes diluido junto a otro
  contenido del mismo doc) ahora matchea con más precisión y pasa `min_score` (0.36 vs el umbral de
  0.35) para una pregunta que solo comparte vocabulario incidental ("responder"/"atención"). Subir
  `min_score` no lo arregla sin romper matches legítimos de score igual de bajo (`oficina.md@0.37`
  en follow-ups cortos).

En ambos casos la solución de verdad era la que el pedido original ya anticipaba: un **reranker**.
Implementado en la siguiente ronda, ver abajo.

## Reranker cross-encoder

**Modelo**: `jinaai/jina-reranker-v2-base-multilingual` vía `fastembed.rerank.cross_encoder.TextCrossEncoder`
(mismo paquete que ya usábamos para los embeddings densos, sin sumar `sentence-transformers`/`torch`).
No es `bge-reranker-v2-m3` -- ese no está en el catálogo de fastembed; este es el cross-encoder
multilingüe disponible más cercano (1.1GB, ONNX). El reranker reordena candidatos del primer filtro
(denso, o híbrido denso+BM25 -- ver siguiente sección), no del corpus entero.

**Dónde corre**: en CPU. Con Whisper + los 2 slots de llama-server ya usando ~3.6GB de los 4GB de
VRAM, no entra (probado: solo quedan ~400-500MB libres). Igual que fastembed para los embeddings
densos, mismo patrón.

**Cómo se integra** (`voice/rag.py: Retriever.retrieve()`): el coseno de siempre trae los
`reranker_candidates` mejores candidatos (antes esto ERA el resultado final); el reranker los
reordena con un juicio de relevancia más fino, y ESE score (logit crudo del cross-encoder, no 0-1)
es el que se compara contra `min_score` y alimenta el gate de ambigüedad. `Hit` ahora guarda también
`dense_score` (el coseno original) para debug con `--show-chunk-text`.

**Latencia medida**: ~55-60ms por candidato reranqueado (lineal). Con `reranker_candidates=10`
(default inicial): ~600-870ms por retrieval -- una adición real y no trivial al "fin de voz → 1er
audio" de *cada* pregunta, no solo las que reescriben. Bajado a `reranker_candidates=5` (nuestro
corpus tiene 14 chunks en total, 5 candidatos ya cubre bien): ~270-340ms por retrieval, sin ninguna
regresión en los casos de prueba. Medido en `bench.py` con el pipeline completo: VRAM 3.6GB (entra
con margen), "fin de voz → 1er audio" pasó de ~1.40s a ~1.64s -- el reranker es ahora el segundo
costo más grande del pipeline después del LLM.

**Recalibración de umbrales** (antes en escala de coseno 0-1, ahora en escala del reranker: logit
crudo, puede ser negativo): medido contra los casos de `tests/eval_questions.yaml` --

| Caso | Score reranker | Antes (coseno) |
|---|---|---|
| Preguntas con respuesta real y clara (3 casos control) | **+0.41 a +1.52** | 0.57 a 0.81 |
| `pregunta_sin_contexto` (control, sin doc relacionado) | -1.02 | 0.27 |
| **Near-miss conocido** (`soporte_tecnico.md` para "¿por qué demoras?") | **-3.22** | 0.36 (¡apenas sobre el viejo umbral de 0.35!) |

La separación es mucho más limpia que con coseno -- con `min_score=0.0`, todo lo bueno queda arriba y
todo lo malo (incluido el near-miss que antes se colaba) queda abajo, sin zona gris. `ambiguity_threshold`
subió a 0.5 (en esta escala, un gap chico entre dos candidatos que *ya* pasaron `min_score` -- no
apareció ningún caso así en los datos disponibles, revisar si aparece en uso real).

*(Esta calibración de `min_score=0.0`/`ambiguity_threshold=0.5` fue manual, a mano contra los casos
de arriba -- quedó documentada acá como historia de cómo se llegó a la escala del reranker, pero
dejó de ser el mecanismo vigente: ver "Independencia de los documentos" más abajo. El valor que
efectivamente se usa hoy sale de `calibration.toml`, generado por `scripts/calibrate.py` contra un
set etiquetado, no de estos números.)*

**Resultado, antes → después:**

| Caso | Antes (coseno + gate) | Después (reranker) |
|---|---|---|
| `comentario_sin_tema_no_corta_continuidad` (near-miss) | ✗ alucinaba citando soporte_tecnico.md | ✓ **arreglado del todo** -- sin contexto, respuesta correcta |
| `retrieval_ambiguo_dos_docs` variantes 1 y 2 | pedían aclaración (gate viejo, con falsos positivos en otros casos) | ✓ **abstienen** ("No tengo información sobre eso") -- ningún doc puntúa positivo, más honesto que forzar una aclaración |
| `retrieval_ambiguo_dos_docs` variante 3 | alucinaba mezclando 2 docs | recupera 1 solo doc (correcto), pero el LLM **sigue malinterpretándolo** (invierte martes/jueves) -- ya no es mezcla de docs, es "bug A" puro |
| `pregunta_compuesta` | con el gate viejo daba falso positivo; sin gate, respondía bien | ✗ regresión: el chunk correcto (tiene las dos partes) puntúa **-0.66** para esta frase compuesta/imperativa -- por debajo de `min_score`, abstiene en vez de responder. No es un problema de umbral: la pregunta pide DOS datos y ningún chunk aislado los tiene juntos de una forma que el reranker reconozca bien para frases compuestas. **Arreglado más abajo** (sección "Descomposición de preguntas compuestas") separando en sub-preguntas antes del retrieval, en vez de tocar `min_score`. |

## Retrieval híbrido (denso + BM25)

El primer filtro (antes del reranker) ahora combina dos rankings en vez de uno solo:

- **Denso**: coseno de siempre, embeddings de `fastembed`.
- **Léxico (BM25)**: `rank_bm25.BM25Okapi` sobre los mismos chunks, indexado en memoria (el corpus
  es chico; con miles de chunks pasaría a un índice invertido persistente, igual que se documentó
  para el denso con FAISS/sqlite-vec). Tokenizado para español: minúsculas, sin acentos (para que
  "días"/"dias" matcheen igual), stopwords chicas y estándar (artículos, preposiciones, pronombres,
  formas de ser/estar/haber/tener) -- `voice/rag.py: _tokenize`.
- **Fusión**: Reciprocal Rank Fusion (RRF, Cormack et al. 2009) -- `score(doc) = Σ 1/(rrf_k + rank)`
  sobre los rankings donde aparece cada doc. Los candidatos fusionados (top `reranker_candidates`)
  son los que ve el reranker, igual que antes con el denso solo.

**Por qué RRF y no una suma pesada de scores**: coseno y BM25 viven en escalas totalmente distintas
(uno acotado 0-1, el otro sin cota, atado al vocabulario) -- normalizar y pesar esas dos escalas a
mano es exactamente el tipo de ajuste fino contra datos puntuales que se quiere evitar (ver
"Independencia de los documentos"). RRF fusiona por **posición** (rank), no por score, así que no
hace falta ninguna normalización ni peso relativo entre las dos señales.

**Parámetros, y por qué son arquitectura y no calibración**: `rrf_k=60` es la constante estándar del
paper de RRF -- prácticamente nunca se ajusta por dominio, así que vive en config por transparencia,
no porque haya que tocarla. `bm25_candidates` (cuántos trae BM25 para la fusión) sí puede necesitar
subirse con corpus más grandes, pero eso es una cuestión de tamaño de corpus, no de qué digan los
documentos -- no es el tipo de parámetro que calibra `scripts/calibrate.py`.

**Recall@5, denso vs. híbrido** (`scripts/calibrate.py`, 9 preguntas con `expected_doc(s)` en
`tests/calibration_questions.yaml`): **9/9 (100%) en ambos** -- con este corpus (14 chunks, 3
documentos, todos con vocabulario simple y sin ambigüedad léxica) no hay ninguna pregunta donde el
híbrido rescate algo que el denso ya no encontrara, así que hoy no se ve una mejora medible. Es
esperable: el caso donde BM25 aporta de verdad es vocabulario exacto que el embedding no captura
bien (códigos, nombres propios, siglas, términos técnicos específicos) -- con documentos genéricos y
preguntas parafraseadas, el denso ya cubre bien. Vale la pena tenerlo desde ya como arquitectura
para cuando lleguen documentos reales (con nombres de sistemas, códigos de ticket, siglas internas,
etc., donde el matching léxico exacto sí puede marcar diferencia) -- recalibrar/re-medir con
`scripts/calibrate.py` en ese momento, no ahora.

**Costo**: negligible. Medido: `BM25Okapi.get_scores()` ~0.03ms, la fusión completa (`_candidate_pool`)
~10-15ms -- comparado con el reranker (~270-580ms, con bastante varianza de corrida a corrida por el
costo de inferencia ONNX en CPU, no relacionado a este cambio) es ruido. Confirmado con
`hybrid_enabled=True` vs `False` en la misma máquina: la variación entre corridas es mayor que la
diferencia entre tener o no tener BM25 activado.

## Testear el LLM/RAG con scripts/eval.py

Herramienta de desarrollo (no de producción): corre una lista de casos de prueba contra el LLM/RAG real
(sin STT ni micrófono) y vuelca todo a un Markdown en `eval_results/` — no calcula pass/fail solo, el
juicio de si cada caso está bien lo hace un humano (o Claude) leyendo el resultado.

```bash
.venv/bin/python scripts/eval.py                       # corre tests/eval_questions.yaml completo
.venv/bin/python scripts/eval.py --out mi_corrida.md
```

Los casos viven en `tests/eval_questions.yaml` — se van agregando ahí a medida que aparecen bugs
nuevos (cada caso tiene `note`/`expect` explicando qué prueba y por qué). Soporta `variants:` por caso:
varias formulaciones distintas de la misma pregunta, para medir robustez a *cómo* se pregunta. Con
`temperature=0` (ver `[llm]` en `config.toml`) la generación es determinística, así que repetir
literalmente la misma pregunta ya no aporta nada — por eso `variants` reemplazó a un viejo `repeat: N`
que medía ruido de muestreo en vez de robustez real. También soporta `type: rewrite`: casos que prueban
*solo* `voice/rewrite.py` aislado (historial + pregunta → pregunta autónoma esperada), sin pasar por
RAG ni el LLM de respuesta — útil para saber si un fallo es de la reescritura o de lo que pasa después.
`--show-chunk-text` muestra el texto completo de cada chunk recuperado (no solo `source@score`), y al
final de la corrida se reporta la latencia de la reescritura (p50/p95, con y sin contar los turnos que
no la disparan por no tener historial todavía). `--split dev` / `--split validation` corre solo esos
casos (ver la sección "Dev vs. validación" más arriba); `--only <id>` corre un caso puntual.

**Guardrails deterministas** (`voice/guardrails.py`, no dependen de que el LLM "decida" seguir una
instrucción):
- **Abstención sin CONTEXTO**: antes, cuando el RAG no traía nada, el prompt le decía al LLM "respondé
  como asistente general" — y eso es lo que lo llevaba a inventar que "Juan Carlos" (sin ningún
  documento que lo mencione) era el Rey de España. Ahora, si no hay CONTEXTO y la pregunta no es charla
  social reconocible (lista blanca chica: saludos, chistes, preguntas sobre el propio asistente), se
  corta con una respuesta fija **sin llamar al LLM**.
- **logit_bias anti-CJK**: Qwen (entrenado por un lab chino) a veces cambiaba de idioma a mitad de
  respuesta. En vez de detectar y reintentar, se penalizan (`-100`) los ~31.000 tokens del vocabulario
  que contienen caracteres CJK (chino/japonés/coreano) *antes* de generar — ~0.3s calcularlo una sola
  vez al cargar el modelo. `clean_for_speech` en `voice/tts.py` sigue filtrando CJK como red de
  seguridad, por si algo igual se escapa.

**Resultado, antes → después de estos dos fixes** (33 ejecuciones, 22/09):

| Caso | Antes | Después |
|---|---|---|
| "¿Quién es Juan Carlos?" (nombre real y famoso, sin doc) | ✗ 0/5 — inventaba que era el Rey de España | ✓ **5/5** — respuesta fija, sin alucinar |
| "¿Quién es María Fernández?" (nombre común) | ✗ 1/3 inventaba una actriz | ✓ **3/3** |
| Conversación completa insistiendo sobre Juan Carlos | ✗ escalaba (inventó que "el jefe directo de Juan Carlos era Felipe VI") | ✓ **consistente** en las 3 vueltas |
| Chiste sin relación a los docs | ✗ 2/5 cambiaba a chino a mitad de frase | ✓ **5/5** en español limpio |
| Horarios, reset de contraseña, vacaciones, trabajo remoto, follow-ups cortos, pregunta repetida, grosería | ✓ ya pasaban | ✓ sin regresión |

*(La tabla de arriba es de la ronda del 22/09, cuando se agregaron los guardrails de identidad/CJK.
El "cambio de día en un follow-up" que quedaba roto ahí se arregló con la reescritura de consulta —
ver esa sección más arriba. El estado actual, más las rondas de latencia/dev-validación/chunking, está
resumido abajo.)*

**Estado actual (23/09/2026, `tests/eval_questions.yaml`, ~90 turnos entre dev y validación, con
reranker + híbrido + calibración + descomposición + respuestas canónicas + fixes de sujeto/afirmación):**

| Área | Estado |
|---|---|
| Horarios, contraseña, vacaciones, trabajo remoto, consistencia, grosería (control) | ✓ sin regresión |
| Cambio de día en follow-up (5 variantes, antes roto) | ✓ **5/5** con la reescritura de consulta |
| Identidad inventada, nombre común, multi-turno, chiste (guardrails deterministas) | ✓ sin regresión |
| Validación holdout de reescritura (formulaciones nuevas, no usadas para ajustar el prompt) | 2/4 generalizan bien; 2/4 no resuelven pero abstienen seguro (nunca alucinan) |
| Cross-doc ambiguo (`retrieval_ambiguo_dos_docs`) | 2/3 abstienen honestamente (ningún doc solo cruza claro); 1/3 recupera 1 doc correcto pero el LLM lo malinterpreta -- "bug A", no de retrieval |
| Descomposición de preguntas compuestas (dev + 4 casos de validación) | ✓ sin regresión, `pregunta_compuesta` arreglado |
| Falsos positivos del pre-filtro de preguntas compuestas sobre preguntas no-compuestas | 0/88 (0%) tras arreglar `_INTERROGATIVE_RE` (matcheaba "que" sin tilde como interrogativo) |
| Sujeto elidido + afirmación-como-pregunta (nuevo, ver sección de bugs en vivo) | ✓ patrón general confiable en varios temas; 1 caso conocido (oración ya completa, sin remisión al sujeto) degrada seguro en vez de alucinar |
| Respuestas canónicas (nuevo): matches, falsos positivos, compuesta mixta, texto exacto, rotación, "repetí" | ✓ **14/14**, todo como aserciones automáticas |
| Persistencia de historial (chequeo automático nuevo, todo turno) | ✓ sin fallos |
| `comentario_sin_tema_no_corta_continuidad` (turno del medio) | ✓ sin regresión |
| 0 caracteres CJK en toda la corrida | ✓ |

Todo documentado con el detalle completo en `tests/eval_questions.yaml` y en las secciones de arriba.
Tres chequeos del harness ya son **automáticos** (exit code != 0 si fallan, no juicio humano):
`expect_n_subquestions`, persistencia de historial, y las aserciones de respuestas canónicas -- el
resto sigue siendo juicio humano/Claude leyendo el Markdown, con una conversión progresiva a
aserciones automáticas pendiente (empezando por los casos de seguridad: abstención, identidad, CJK).

## Independencia de los documentos: calibración vs. arquitectura

Los documentos de `docs/` son genéricos, de prueba -- no son los definitivos. Ajustar umbrales,
few-shots o patrones a mano contra casos puntuales de ESOS documentos es sobreajustar: el trabajo no
generaliza a los documentos reales cuando lleguen. La regla desde acá en adelante:

- **Arquitectura estable** (no depende de qué documentos haya): el mecanismo del reescritor y la
  descomposición de preguntas compuestas (siguiente sección), el mecanismo del reranker (candidatos
  densos → cross-encoder → filtro), el diseño de `llama-server` con 2 slots, la categoría de
  guardrails deterministas (`voice/guardrails.py`) y su forma de intervenir sin llamar al LLM.
- **Parámetros provisorios, atados a `docs/`**: los valores puntuales de `rag.min_score` y
  `rag.ambiguity_threshold`, las entradas de `[rag.doc_topics]`, y el propio dataset de
  `tests/calibration_questions.yaml`. Estos SÍ hay que recalcular cuando cambien los documentos.

**`scripts/calibrate.py`**: en vez de mover `min_score`/`ambiguity_threshold` a mano hasta que un
caso puntual pase, el script barre umbrales candidatos contra un set etiquetado
(`tests/calibration_questions.yaml`: preguntas marcadas `in_domain` / `out_of_domain` / `no_answer` /
`ambiguous`) y elige el que resuelve mejor esas etiquetas, priorizando ante todo que
`out_of_domain`/`no_answer` nunca "crucen" el umbral (el riesgo de más impacto: una respuesta
confiada e inventada). Escribe el resultado a `calibration.toml`, que `voice/config.py` carga y
superpone sobre los defaults de `config.toml` (`_apply_calibration`) -- así **recalibrar para
documentos reales es correr el script de nuevo**, reescribiendo antes
`tests/calibration_questions.yaml` con preguntas sobre esos documentos, sin tocar `config.toml` a
mano.

```bash
.venv/bin/python scripts/calibrate.py              # calibra y escribe calibration.toml
.venv/bin/python scripts/calibrate.py --dry-run     # solo el reporte, no escribe nada
```

**Un intento descartado, documentado porque es una lección real**: la primera versión del barrido
exigía que una pregunta `ambiguous` tuviera **2 documentos distintos** sobre el umbral (para que el
gate de ambigüedad, que necesita 2 candidatos, tuviera con qué disparar). Eso bajó `min_score` de
0.0 a -1.6 -- y a ese nivel, con un corpus chico y genérico, casi cualquier chunk cruza para casi
cualquier pregunta: preguntas de un solo tema ("¿Cuál es el horario de la oficina los sábados?")
empezaban a mezclarse con documentos sin relación real y disparaban el gate de ambigüedad sin
ambigüedad de verdad (se vio corriendo `scripts/eval.py` sobre `pregunta_compuesta` y los nuevos
casos de descomposición). Cambiaba un bug (alucinar con 1 solo doc débil) por otro peor (pedir
aclaración de más en preguntas normales). Se volvió a la versión conservadora: `min_score` se
calibra solo para "¿hay o no hay contexto razonable?" (1+ doc alcanza también para `ambiguous`); que
ese contexto tenga genuinamente 2 documentos en conflicto es un juicio más fino, calibrado aparte y
por separado (`ambiguity_threshold`) -- si no hay suficientes casos para calibrarlo con los datos
disponibles, el script lo dice explícitamente en vez de forzar un valor. Con esto, el resultado
calibrado actual (`calibration.toml`) es `min_score ≈ -0.68` (recupera "¿Cuándo se pagan los
sueldos?", un near-miss real que con `min_score=0.0` quedaba apenas afuera) y `ambiguity_threshold`
sin datos suficientes para recalibrar, así que queda en el default de `config.toml` (0.5).

**Recall@5, antes del reranker**: `scripts/calibrate.py` reporta si el primer filtro (el que trae
`reranker_candidates` candidatos para el cross-encoder) ya pierde el chunk correcto antes de que el
reranker tenga la oportunidad de reordenarlo. *En esta ronda esto era solo denso*; el retrieval
híbrido (denso + BM25, con la comparación recall@5 denso vs. híbrido) se agregó después -- ver
sección "Retrieval híbrido (denso + BM25)" más abajo para el detalle completo y los resultados.
Esta medición sí importa una vez que lleguen documentos reales y más grandes -- si el recall baja,
la señal es subir `reranker_candidates` o revisar chunking/embeddings, no tocar `min_score`.

**Reescritor -- stop sequences**: se agregó `stop=["?", "\n"]` a la llamada de reescritura simple
(no a la de descomposición, que devuelve JSON) para cortar la decodificación apenas termina la
pregunta reescrita, en vez de esperar el token EOS del modelo. Medido en un caso real
(`rewrite_correccion_explicita`): sin stop, 12 tokens / 287.7ms; con `stop=["?","\n"]`, 11 tokens /
250-272ms -- ahorro real pero modesto (~15%, el modelo ya paraba casi enseguida solo en este caso);
más que nada es una red de seguridad contra divague si algún día el modelo empieza a explicar en vez
de cortar limpio. Detalle no obvio: `llama-server` **no incluye el string de corte en la
respuesta** (confirmado empíricamente) -- si cortó por `"?"`, el texto devuelto no lo tiene, y hay
que reponerlo a mano o `looks_like_question()` lo rechaza siempre y se pierde la reescritura entera.
El endpoint OpenAI-compatible tampoco distingue cuál de los dos stops se disparó (no expone
`stopping_word` como el endpoint nativo `/completion`) -- se repone el `"?"` cuando el texto no tiene
`"\n"` y el corte fue por `finish_reason="stop"`, una apuesta documentada en el código
(`voice/llm.py: rewrite_query`) en vez de escondida.

## Descomposición de preguntas compuestas (punto 4)

Motivador: `pregunta_compuesta` ("Decime el horario de lunes a viernes y también el de los
sábados.") -- el chunk correcto (tiene ambos datos) puntuaba negativo con el reranker para esta
frase compuesta/imperativa, así que abstenía en vez de responder. No es un problema de umbral: la
pregunta pide DOS datos y el reranker juzga la relevancia de la frase completa contra el chunk, no
de cada dato por separado. La solución es separar en sub-preguntas ANTES del retrieval, para que
cada una se juzgue contra el chunk que le corresponde.

**Distinción clave**: "compuesta" (dos datos distintos, un tema -- necesita descomponerse) vs.
"ambigua" (un dato, dos documentos en conflicto -- necesita el gate de `ambiguous_docs`). El
coseno/reranker solos no distinguen esto; hace falta resolverlo antes, con un filtro barato.

**`voice/rewrite.py: looks_compound()`** -- filtro determinista (sin LLM), evaluado en **cualquier
turno, incluido el primero** (una pregunta compuesta puede ser la primera frase de la charla, no
depende de que haya historial). Señales, cualquiera alcanza: dos signos de interrogación,
conectores ("y", "además", "también"), dos palabras interrogativas distintas, marcas de
enumeración. A propósito sesgado a favor de recall: un falso positivo solo cuesta la latencia de una
llamada extra al reescritor en "modo descomposición", que en el peor caso devuelve una sola
sub-pregunta. Medido contra los 63 turnos no-compuestos de `tests/eval_questions.yaml`: **1/63 (2%)
falsos positivos** -- el pre-filtro es barato en la práctica.

**Modo descomposición** (`voice/rewrite.py: DECOMPOSE_SYSTEM_PROMPT` + few-shot dedicado): el
reescritor devuelve un array JSON de 1 a 3 sub-preguntas autónomas, cada una en forma interrogativa
("¿Cuál es...?") -- normaliza fraseo imperativo ("decime X y también Y" → dos preguntas). Si el JSON
no valida (`parse_subquestions`), se usa la pregunta original tal cual, sin sub-preguntas.

**Por sub-pregunta, independiente** (`voice/pipeline.py: Assistant.answer()`): retrieval + reranker
+ el gate de abstención/ambigüedad de siempre corren por separado para cada una. Las que abstienen o
dan ambiguo NO llegan al LLM de respuesta -- se resuelven con una nota fija (NO generada,
`voice/guardrails.py: abstain_partial_reply`/`clarify_reply`) que se concatena al final. Las
resueltas (con CONTEXTO válido, o chitchat sin CONTEXTO) van todas juntas a **una sola llamada** al
LLM de respuesta (`LocalLLM.stream_answer_multi`).

**El detalle no obvio de esa llamada única**: pedirle en prosa "respondé cada parte por separado" no
alcanzaba -- con dos sub-preguntas que comparten el mismo CONTEXTO (pasa seguido con este corpus,
donde un párrafo trae varios datos juntos), el 3B ignoraba una de las dos partes por completo y solo
contestaba con el dato más "saliente" para ambas. Pedirle un formato con ancla estructural (`Parte 1:
...`, `Parte 2: ...`, una línea por parte, obligatorio) lo resuelve de forma confiable. Esas
etiquetas son un andamiaje interno -- se sacan (`strip_part_labels`) antes de hablar/mostrar/guardar
la respuesta, el usuario nunca las escucha.

**Verificado con `scripts/eval.py`** (agregado `compound: true` a los casos que deben disparar el
pre-filtro, para no contarlos como falsos positivos):
- `pregunta_compuesta` (dev): ✓ responde ambas partes.
- 4 casos nuevos en `split: validation` (holdout, formulaciones no parecidas a los few-shot):
  primer turno sin historial, seguimiento después de un tema sin relación, una parte fuera de
  dominio (responde la parte con datos + nota fija para la otra, sin alucinar ni abstener de más), y
  fraseo imperativo distinto al del few-shot ("Contame"/"de paso decime" en vez de "Decime... y
  también"). Los 4 pasan.
- Sin regresión en los casos no compuestos (dev + validación completos, 0 caracteres CJK en toda la
  corrida).

**El stop-sequence de la reescritura simple (`"?"`, `"\n"`) NO se aplica en modo descomposición**:
`_decompose()` (`voice/llm.py`) llama a `_post()` sin pasar `stop` -- si se aplicara, la
decodificación del array JSON de sub-preguntas cortaría en el primer `"?"` (a mitad de la primera
sub-pregunta), el JSON quedaría inválido, y `rewrite_query()` caería silenciosamente a "no
descompuesto". Confirmado por lectura de código y con una llamada real al servidor (JSON completo,
sin truncar). Se agregó `rewrite_decompose_no_trunca_json` (`type: rewrite`,
`expect_n_subquestions: 2`) a `tests/eval_questions.yaml` -- a diferencia del resto del harness
(juicio humano/Claude), este caso se verifica **automáticamente**: si el número de sub-preguntas no
coincide, `scripts/eval.py` termina con exit code != 0. Probado reintroduciendo el bug a propósito
(agregando el `stop` a `_decompose()`) -- el caso falla como se espera (1 sub-pregunta en vez de 2),
confirmando que el chequeo tiene dientes.

**Diferido a propósito** (per el pivot de documentos genéricos): lookup estructurado de horarios y
la comparación 3B vs. 7-8B quedan para cuando existan documentos reales.

## Bugs de una prueba en vivo (23/09): sujeto elidido, afirmaciones, historial silencioso

Una conversación real (`--no-wake`, sin mic) expuso varios bugs que el harness no cubría:

```
🗣  ¿Qué hora abre la oficina?          🤖 Diez de la mañana.
🗣  todos los días                      (reescrita: '¿Cuándo abre todos los días?') 🤖 No lo encontré...
🗣  Y todos los días abre a las 10...    (reescrita: '...mañana.?')                 🤖 No lo encontré...
🗣  Todos los días abre a las 10...      (sin historial, rewrite=0ms)                🤖 No tengo información...
```

**1) El reescritor perdía el sujeto y cambiaba el tipo de pregunta.** "todos los días" (sí/no,
sujeto "la oficina" elidido) se reescribía a "¿Cuándo abre todos los días?" -- perdía "la oficina"
Y cambiaba sí/no por qué/cuándo. `SYSTEM_PROMPT` y los few-shot ahora exigen explícitamente arrastrar
TODAS las entidades del turno anterior (no solo el dato nuevo) y preservar el tipo de pregunta.
Verificado con distintos temas (oficina, wifi, sala de reuniones, estacionamiento): el patrón
general de sujeto elidido generaliza bien; una oración YA gramaticalmente completa sin ninguna
palabra que remita al sujeto ("Y todos los días abre a las 10 de la mañana") sigue siendo un límite
real del 3B en algunos casos -- documentado como tal en vez de forzado con un few-shot calcado del
caso puntual (ver `validation_sujeto_elidido_y_afirmacion`); lo importante es que degrada seguro
(abstiene honesto) en vez de alucinar o corromper texto.

**Lección de few-shot -- overfitting de superficie**: la primera versión de los few-shot nuevos
usaba el mismo tema (oficina/horarios) en 3 ejemplos seguidos. Con una entrada nueva sin relación
("El estacionamiento tiene doce lugares.", primer turno), el 3B directamente devolvía **el texto de
otro few-shot**, textual -- había memorizado el patrón de superficie en vez de generalizar.
Diversificar el tema de cada ejemplo (oficina / sala de reuniones / wifi / vacaciones) lo arregló.
Documentado en el código porque es una lección reusable, no específica de este bug.

**2) Afirmaciones sin "?" no se reescribían.** En voz hablada, una confirmación de sí/no suena como
afirmación con entonación ("todos los días abre a las diez") y Whisper la transcribe sin signos de
pregunta -- el reescritor ni se ejecutaba (dependía de que hubiera historial). `voice/rewrite.py:
looks_like_statement()` (determinista, sin LLM) detecta entradas sin "?" ni palabra interrogativa al
inicio y dispara el reescritor **en cualquier turno, incluido el primero sin historial** -- excluye
`is_chitchat` para no pagar la llamada en comandos/insultos ("Contame un chiste.") que no son
afirmaciones a convertir.

**3) Dos bugs de validación.** (a) Reponer el "?" del stop-sequence (ver sección de arriba) no
chequeaba si el texto ya terminaba en OTRO signo de cierre -- daba `"...mañana.?"`. Ahora se compara
contra `rewrite.TERMINAL_PUNCT_CHARS` (`.?!…`), no solo `"?"`. (b) Si la entrada era una afirmación y
el modelo la devolvió prácticamente igual (no la convirtió), esa "reescritura" se rechaza
(`rewrite.is_near_identical`) y cae a la pregunta original -- con un cuidado no obvio: la comparación
NO saca el "¿" inicial antes de comparar (a diferencia de la puntuación de cierre), porque agregarlo
es justo la señal de que el modelo sí convirtió la afirmación; sacarlo daba falsos "no convirtió nada"
sobre conversiones válidas.

**4) Historial "perdido" en el 4º turno.** Investigado a fondo: `record_turn()` se llama
sin excepción en los dos caminos de salida de `Assistant.answer()` (confirmado por lectura de
código, y ahora por un chequeo automático en `scripts/eval.py` que verifica que
`self.llm.history` crezca en +2 en cada turno). La causa real era otra: `Assistant.run()` solo
imprimía el aviso de "charla reiniciada" cuando había wake word configurado -- sin wake word
(`--no-wake`), un reset legítimo por `followup_s` (silencio de más de 5s entre turnos) pasaba
**sin ningún aviso visible**, indistinguible de un bug. Arreglado: el aviso ahora sale siempre.

**5) Observabilidad**: `Turn.retrieval_trace` (nuevo) guarda, por sub-pregunta, el chunk top-1 y su
score AUNQUE no haya cruzado `min_score` -- antes "no se recuperó nada" y "se recuperó con score
bajo" se veían idénticos en el log (`RAG: *(sin contexto)*`). Ahora, cuando no hay hits,
`scripts/eval.py` y `Assistant.print_metrics` muestran `top1=<doc>@<score> (min_score=<umbral>)`.

**6) Corrección de premisa**: "¿La oficina abre todos los días?" contra un chunk que dice "lunes a
viernes" -- una vez arreglado el bug 1 (retrieval encuentra el chunk correcto), el LLM de respuesta
**ya corrige la premisa por sí solo** ("No, abre de lunes a viernes...", "No, solo los sábados..."),
gracias a la instrucción de honestidad que ya tenía el `system_prompt`. No hizo falta tocar nada
más -- era un efecto secundario del bug 1, no un problema aparte.

**Harness**: `rewrite_sujeto_elidido_sino`, `rewrite_afirmacion_como_confirmacion`,
`rewrite_afirmacion_primer_turno` (`type: rewrite`, temas distintos entre sí -- ver la lección de
arriba) y `correccion_de_premisa` en dev; `validation_sujeto_elidido_y_afirmacion` (la conversación
real completa) en validación.

## Respuestas canónicas (FAQ, texto fijo)

Ciertas preguntas frecuentes se responden mejor con un texto fijo, redactado por la empresa, que con
algo generado por el LLM: cero riesgo de alucinación en las preguntas más comunes, control total del
mensaje, y latencia mínima (no hay decodificación de tokens). Convive con el RAG: si una
(sub)pregunta matchea una entrada canónica, se responde con su texto tal cual; si no, sigue el
pipeline normal.

**Contenido de prueba, no definitivo**: `tests/canonical_answers.yaml` tiene 5 entradas inventadas
(a qué se dedica la empresa, ubicación, contacto, historia, horario de atención al público -- esta
última a propósito se solapa en TEMA con `docs/oficina.md`, para probar que la canónica gana
prioridad). Se prueba el **mecanismo**, no el contenido -- cuando exista contenido real, se
reemplaza ese archivo entero (mismo formato) y se recalibra (ver más abajo).

**Formato** (por entrada): `id` único, `formulaciones` (3-6 formas de preguntar lo mismo, para el
matching) y `respuestas` (1+ variantes de texto -- todas deben decir lo mismo; si se actualiza un
dato, hay que cambiarlo en TODAS). Validado al cargar (`voice/canonical.py:
load_canonical_entries`), rechaza el archivo con un error que nombra la entrada si: hay ids
duplicados, algún campo está vacío, o una formulación es prácticamente igual a una ya usada en OTRA
entrada (dos entradas no pueden competir por la misma frase).

**Lint para voz** (warnings al cargar, no bloquean): dígitos o formatos tipo "9-13hs" (Piper lee
mejor los números en palabras), abreviaturas ("Av.", "Sr.", "etc."), viñetas o markdown, más de ~80
palabras, y variantes de respuesta de largos muy distintos entre sí (probablemente no dicen lo
mismo). Todo lo que se lee en voz alta, no se muestra como texto.

**Dónde corre en el pipeline** (`voice/pipeline.py: Assistant.answer()`): **después** de
reescritura/descomposición, **antes** del RAG -- por cada sub-pregunta, se prueba primero contra
canonical; si matchea, esa sub-pregunta ya está resuelta y ni siquiera llega al RAG. El matching
reusa el mismo mecanismo de reranker cross-encoder que el RAG (`fastembed.rerank.cross_encoder`),
pero contra las formulaciones canónicas en vez de los chunks de documentos -- sin normalizar/pesar
contra el coseno, porque acá no hay coseno de por medio, es reranker solo. Una pregunta compuesta
mixta (una parte canónica, otra no) responde la parte canónica textual y la otra por RAG/LLM,
concatenadas -- el LLM nunca toca el texto canónico.

**Rotación de variantes, determinista** (`CanonicalMatcher._next_variant`): conversación nueva
siempre empieza por la variante 1; si la misma entrada vuelve a matchear en la misma conversación,
usa la siguiente no usada, y al agotarlas vuelve a la 1 (módulo). El estado vive en la conversación
(`Assistant.reset_conversation()` lo resetea junto con el historial). "Repetí" / "¿cómo?" / "no te
escuché" (`guardrails.is_repeat_request`, nuevo patrón determinista en la lista blanca de charla
social) repite la ÚLTIMA respuesta dada tal cual -- canónica o no -- sin rotar ni volver a generar.

**Calibración del umbral** (`canonical.threshold`, PROVISORIO): igual mecanismo que
`rag.min_score` -- `scripts/calibrate.py` barre umbrales contra `tests/calibration_canonical.yaml`
(preguntas etiquetadas `match`/`no_match`), priorizando **precisión** por sobre recall: si una
canónica no matchea, cae al RAG (sin drama); si matchea la entrada EQUIVOCADA, o matchea algo que no
debía matchear nada, recita con total confianza un texto que no corresponde -- ese es el error caro.
Con el threshold=2.0 puesto a mano inicialmente (una simple suposición), hasta una formulación
**idéntica** a una del propio archivo puntuaba apenas 1.68 -- confirmando que había que calibrar de
verdad, no adivinar. Calibrado (0.23): 12/14 casos `match` resueltos bien, 0 casos peligrosos.

**Verificado con `scripts/eval.py`**, todo como aserciones **automáticas** (`expect_canonical_entries`
/ `expect_exact_text` / `expect_variant_rotation` / `expect_repeat_of_previous` -- exit code != 0 si
fallan, no juicio humano): matches correctos por entrada (incluida una imperativa y una como
follow-up), 3 falsos positivos que NO deben matchear (persona con el mismo verbo, área puntual en
vez de la empresa entera, "baño" vs. "dirección"), fuera de dominio sigue abstenido, compuesta mixta,
texto exacto carácter por carácter, rotación de variantes (1→2→3→1) y "repetí" sin rotar. De las
formulaciones nuevas probadas, 3 cayeron por debajo del umbral calibrado (near-misses genuinos,
0.10-0.14 vs. 0.23) y se reemplazaron por formulaciones más claras en vez de bajar el umbral a mano
-- documentado en el propio harness, no escondido.

**Cómo agregar o editar contenido** (pensado para alguien NO técnico, editando
`tests/canonical_answers.yaml` -- o el archivo real que lo reemplace):
1. Copiar el formato de una entrada existente: `id`, `formulaciones` (3-6 formas de preguntar lo
   mismo) y `respuestas` (1 o más variantes de texto).
2. Al arrancar el asistente (o correr `scripts/eval.py`/`scripts/calibrate.py`), si algo está mal
   (id repetido, campo vacío, formulación que choca con otra entrada) sale un error con el nombre
   de la entrada -- corregir eso antes de seguir.
3. El lint (warnings, no bloquean) avisa de cosas que suenan raro en voz alta: números en dígitos,
   abreviaturas, viñetas/markdown, respuestas muy largas, o variantes de largo muy distinto entre sí
   dentro de la misma entrada. Conviene revisarlas igual aunque no bloqueen.
4. **Importante**: si se actualiza un dato (una dirección, un teléfono), hay que cambiarlo en
   **TODAS** las variantes de respuesta de esa entrada -- son formas distintas de decir lo mismo, no
   pueden quedar diciendo cosas distintas entre sí.
5. Después de cambiar contenido (agregar/sacar entradas, reformular preguntas o respuestas), correr
   `python scripts/calibrate.py` de nuevo -- el umbral (`canonical.threshold`) está calibrado contra
   el contenido ANTERIOR, y agregar o sacar entradas puede correr los scores de las demás.

## Rediseño del gate de ambigüedad (24/09): cercanía de scores no implica ambigüedad

Prueba en vivo: **"¿La oficina abre los martes?"** con `top1=oficina.md@0.38` (claramente por
encima de `min_score`) igual pedía aclaración entre "oficina" y "recursos humanos".

**Bug real, no solo de calibración**: `_best_per_doc()` (antes `ambiguous_docs`) usaba `0.0` como
score default en `max(best_per_doc.get(source, 0.0), score)` -- para un documento cuyo único hit
tiene score **negativo**, eso lo dejaba "flotando" en 0.0 en vez de su score real.
`recursos_humanos.md` tenía un solo hit en -0.15, pero el bug lo veía como 0.0 -- achicando el gap
real (0.38 - (-0.15) = 0.53) a uno falso (0.38 - 0.0 = 0.38), por debajo del umbral. Arreglado con
`float("-inf")` como default.

**Pero arreglar el bug de cálculo no alcanzaba** -- el diseño en sí asumía que cercanía de scores
entre dos documentos implica ambigüedad, y eso no es cierto: pueden ser complementarios (uno
genuinamente relevante, el otro apenas "no descartable"), no necesariamente en conflicto.
`voice/guardrails.py: gate_docs()` (reemplaza a `ambiguous_docs`) separa dos decisiones antes
mezcladas en una:

1. **Confianza alta** (`top1 >= min_score + confidence_margin`, nuevo parámetro PROVISORIO en
   config): el CONTEXTO se restringe a los chunks de **ese** documento solamente, descartando
   cualquier otro que haya cruzado `min_score` de casualidad -- sin preguntar nada. Esto es lo que
   previene la mezcla de políticas (el bug original que motivó el gate viejo) de verdad, en vez de
   solo evitar la pregunta de aclaración.
2. Solo si **ningún** documento tiene confianza alta y los dos mejores (de documentos DISTINTOS)
   están a menos de `ambiguity_threshold`: pedir aclaración -- ahí sí, ninguno domina con claridad.
3. Ninguno de los dos casos: sigue el camino de siempre (CONTEXTO con los hits tal cual).

`confidence_margin=0.8` es PROVISORIO, sin calibrar con `scripts/calibrate.py` -- este corpus no
tiene todavía un caso real de 2 documentos genuinamente en conflicto para calibrar el camino
"clarify" con datos (verificado con un test sintético de `gate_docs()`, ver
`tests/eval_questions.yaml`). Revisar cuando haya documentos reales con temas que sí se solapen.

**Traza ampliada**: `Turn.retrieval_trace` ahora guarda también el top-2 (documento y score), no
solo el top-1 -- para poder ver qué compite, aunque no haya cruzado `min_score`.

**Mensaje de aclaración**: los nombres de tema (`rag.doc_topics`, config) tenían una sigla con
puntuación doble ("políticas de RR.HH.") -- cambiado a "políticas de recursos humanos".
`voice/guardrails.py: lint_for_voice()` (compartido con el lint de respuestas canónicas, antes
duplicado) ahora también detecta siglas (mayúsculas + punto) y puntuación doble/muy junta;
`scripts/eval.py` lo corre contra `doc_topics` y una muestra de `clarify_reply()` al arrancar,
como warning (no bloquea).

**Antes/después, verificado con `scripts/eval.py`** (`expect_clarify` / `expect_context_restricted_to`,
aserciones automáticas):
- `ambiguedad_falso_positivo_confianza_alta` (nuevo, la conversación real): antes pedía aclaración
  de más; después responde directo, CONTEXTO restringido a `oficina.md` solamente.
- `retrieval_ambiguo_dos_docs` (las 3 variantes): sin cambio -- con la calibración actual, ningún
  segundo documento cruza `min_score` para estas preguntas puntuales, así que nunca llegaban (ni
  antes ni ahora) a la comparación de 2 documentos. La variante 3 sigue con "bug A" (el LLM
  malinterpreta el único chunk recuperado) -- no es un problema del gate ni de retrieval.

**Latencia de Piper, desglosada**: la traza mostraba ~2s sin explicar en un turno de aclaración
(stt 0.78s + rag 0.56s vs. 3.47s a primer audio, sin LLM de por medio). Causa: `_fixed_reply()`
(usado para abstención, aclaración, canónicas-solas y "repetí") pasaba el texto de respuesta
**completo** a `StreamingSpeaker.say()` de una vez -- Piper sintetizaba el mensaje entero antes de
que saliera el primer audio, a diferencia del camino de generación del LLM (que ya sintetizaba
frase por frase). `Assistant._say_text()` (nuevo, usa `tts.iter_sentences`) ahora parte el texto en
frases para los tres caminos (fijos, canónicas, notas de sub-preguntas no resueltas) igual que el
LLM. `StreamingSpeaker.synth_ms` (nuevo) guarda el tiempo de síntesis por frase; `Turn.t_tts_total`/
`t_tts_first` lo exponen en la traza y en `print_metrics`. Medido con el mensaje de aclaración real
(dos oraciones): con el split, el primer audio sale a los ~1.22s (tiempo de la PRIMERA oración
sola) en vez de ~1.58s (las dos oraciones juntas) -- mejora real, aunque la reducción total depende
del largo de cada oración individual, no es gratis para mensajes con una primera oración larga.

## Cierre de etapa (23/09): congelamos el desarrollo sobre documentos de prueba

A partir de acá, los próximos saltos de valor (documentos reales, preguntas de gente de la
empresa, sesiones con usuarios, corrida en la Orin) no dependen de más código. Esta sección
documenta el estado del proyecto al cerrar esta etapa: verificable automáticamente, con el
recorrido de un turno documentado de punta a punta, y con un procedimiento claro para incorporar
contenido real. No se agregó funcionalidad nueva ni se tocó ningún parámetro/umbral -- lo de acá
es instrumentación (para poder medir), aserciones (para poder verificar) y documentación.

### Latencia verificada

La prueba en vivo que motivó el rediseño del gate de ambigüedad (sección de arriba) reportaba
**3.47s** de fin de voz a primer audio para "¿La oficina abre los martes?", con ~2s sin explicar.
Ese ~2s ya se explicó y arregló en su momento (Piper sintetizando el mensaje completo antes de
hablar, en vez de por oración). Esta vuelta se verificó que **no queda nada más sin explicar**, en
las 4 rutas posibles de un turno, agregando instrumentación que faltaba (sin cambiar nada del
comportamiento):

- `Turn.t_canonical` -- antes invisible: la llamada a `canonical.match()` corre por CADA
  sub-pregunta, matchee o no (es un reranker cross-encoder, no es gratis: ~520-580ms medidos acá).
  Se paga siempre, incluso en preguntas que van a terminar en RAG puro.
- La llamada de traza a `score_candidates()` (para tener el top-2 cuando `hits` trae menos de 2)
  ahora se cuenta dentro de `turn.t_rag` -- antes quedaba fuera de cualquier timer, como si fuera
  gratis.
- `Turn.t_llm_first_sentence` -- distingue "1er TOKEN generado" (`t_llm_first_token`) de "1ra
  ORACIÓN completa" (`t_llm_first_sentence`): el TTS no puede arrancar hasta que
  `tts.iter_sentences()` junta una oración entera, así que para una primera oración larga el
  segundo puede ser bastante mayor que el primero. Antes esto se perdía en un salto sin explicar
  entre `t_llm_first_token` y `t_first_audio`.
- Warm-up de Piper agregado en `Assistant.__init__` (una síntesis descartable de "Prueba." al
  arrancar, igual que ya se hacía con Whisper): la primera síntesis real medía ~18.6ms por cada
  1000 muestras de audio contra ~12-15ms en las siguientes (ONNX Runtime elige algoritmo/arma
  buffers internos la primera vez) -- un efecto real pero chico, no explica una latencia de
  segundos por sí solo. Ahora se paga en el arranque, no en la primera respuesta del usuario.

Con esto, las 4 rutas quedan **100% explicadas** (diferencia entre la suma de los timers y
`t_first_audio` medida en 0-1ms, ruido de reloj):

| Ruta | canonical | rag | llm 1ra oración | llm total | tts 1ra frase | **fin de voz → 1er audio** |
|---|---|---|---|---|---|---|
| Canónica ("¿A qué se dedica la empresa?") | 521ms | 0ms | -- | -- | 1164ms | **1.68s** |
| RAG + LLM ("Olvidé mi contraseña") | 561ms | 878ms | 930ms | 1872ms | 1349ms | **3.72s** |
| Abstención (fuera de dominio) | 577ms | 805ms | -- | -- | 697ms | **2.08s** |
| Aclaración (sintético, ver nota) | -- | -- | -- | -- | 1230ms | **1.23s** |

Notas sobre la tabla:
- **Canónica** y **abstención** no pasan por el LLM de respuesta (texto fijo o nota fija --
  `_fixed_reply`), por eso no tienen columnas de LLM.
- **Aclaración**: la fila mide el mensaje de `clarify_reply()` synthesizado solo, para aislar el
  costo de TTS de un mensaje de 2 oraciones. En un turno real, `canonical.match()` y el RAG corren
  ANTES de llegar al gate (es el gate el que decide pedir aclaración), así que el costo real de un
  turno de aclaración de punta a punta es más parecido a canonical + rag + este número (~2.6s), no
  a 1.23s solo.
- El caso original ("¿La oficina abre los martes?" con 3.47s) **ya no reproduce el camino de
  aclaración** con la calibración actual: tras el rediseño del gate, ese top-1 domina con
  confianza y el turno sigue por RAG + LLM (fila 2), no por el mensaje fijo de aclaración. No fue
  posible remedir exactamente esa traza original por ese motivo -- lo que se verificó en su lugar
  es que las 4 rutas posibles hoy están completamente explicadas.
- `canonical.match()` corriendo siempre, aunque la pregunta termine yendo a RAG, es un costo
  evitable (podría saltarse con un filtro léxico barato antes del reranker, o correr en paralelo
  con el primer paso del RAG) -- **no se optimizó acá** (fuera de alcance de esta etapa: "no
  ajustar funcionalidad"), queda anotado como oportunidad futura en limitaciones.

### Harness 100% automático

Hasta ahora, varios casos de `tests/eval_questions.yaml` dependían de que una persona (o Claude)
leyera el reporte en Markdown y juzgara si la respuesta era razonable. Se convirtieron todos los
casos que se pueden verificar programáticamente a aserciones con exit code ≠ 0, priorizando
seguridad primero. Ver `scripts/eval.py` (funciones `_check_cjk`, `_check_content_expectations`,
`_check_canonical_expectations`, `_check_gate_expectations`) y los campos `expect_*` de cada caso
en `tests/eval_questions.yaml`.

Campos de aserción disponibles (todos opcionales, se combinan según lo que tenga sentido para el
caso):

| Campo | Qué verifica |
|---|---|
| `expect_abstain` | La respuesta es textualmente una de las 3 frases fijas de `abstain_reply()`/`abstain_partial_reply()` (SEGURIDAD: fuera de dominio, sin datos inventados) |
| `expect_contains` / `expect_not_contains` | Substring (case-insensitive) presente/ausente en la respuesta del último turno |
| `expect_not_contains_any_turn` | Substring ausente en TODOS los turnos de la conversación (identidad inventada, fuga de contexto) |
| `expect_contains_each_turn` | Lista de listas, una por turno; `[]` en un turno = sin chequeo ese turno (deja explícito qué no se verifica y por qué, en vez de omitir el campo en silencio) |
| `expect_doc` | El/los documento(s) en `context_hits` coinciden exactamente con lo esperado (retrieval acertó el documento, no solo "no abstuvo") |
| `expect_same_as_previous` | Igual texto que el turno anterior ("repetí"); nota: a temperature=0 puede haber no-determinismo chico por batching de GPU, documentado en el caso |
| `expect_clarify` / `expect_context_restricted_to` | Resultado del gate de ambigüedad: pidió aclaración, o restringió el CONTEXTO a un documento |
| `expect_canonical_entries` | Qué entrada canónica (o `null` = ninguna) matcheó cada sub-pregunta |
| `expect_was_rewritten` | La reescritura/descomposición disparó (o no) |
| `expect_n_subquestions` | Cantidad de sub-preguntas tras la descomposición |
| CJK (universal, sin campo) | 0 caracteres CJK en CUALQUIER respuesta o reescritura -- corre siempre, no es opt-in |

Cobertura actual (reportada por `_count_assertions()` al final de cada corrida de
`scripts/eval.py`): **51 de 52 casos** tienen al menos una aserción automática. El único caso sin
aserción es `rewrite_referencia_dos_turnos_atras`, documentado en su propio `note:`: la pregunta es
deliberadamente ambigua entre dos temas igual de válidos (ese es justamente el comportamiento que
prueba), así que forzar un `expect_contains` sobre uno de los dos elegiría arbitrariamente un
"ganador" y el chequeo terminaría probando algo que el caso no se propone probar.

Por categoría/split (aprox., ver la salida real de `scripts/eval.py` para el conteo exacto en cada
corrida):
- **Seguridad** (abstención fuera de dominio, identidad inventada, 0 CJK): 100% con aserción,
  todas con `expect_abstain` y/o `expect_not_contains_any_turn` y/o el chequeo CJK universal.
- **Validación** (holdout, nunca ajustado contra estos casos): 10/10 con aserción.
- **Desarrollo**: 41/42 (el 1 caso sin cobertura ya descrito arriba).

**Un solo comando** corre todo y devuelve pass/fail por exit code -- `scripts/verify.py`:

```
python scripts/verify.py
```

Corre, en orden: (1) `scripts/calibrate.py --check` -- valida los umbrales VIGENTES (`rag.min_score`,
`canonical.threshold`, ya calibrados en `calibration.toml`) contra los datasets de calibración SIN
recalibrar ni escribir nada, contando solo casos "peligrosos" (falso positivo con riesgo de
alucinación) como fallo -- un falso negativo (abstiene de más, o cae al RAG en vez de matchear
canónica) se reporta pero NO cuenta como fallo, es la filosofía de precisión-sobre-recall de
siempre, no relajada acá; (2) el lint de voz sobre `doc_topics` y `clarify_reply()`; (3)
`scripts/eval.py`, la suite completa (52 casos, dev + validación) con todas las aserciones de
arriba. Exit code 0 solo si los dos pasos terminan en 0. **Documentado como el paso obligatorio
antes de cada commit** (ver docstring de `scripts/verify.py`).

### Flujo de decisión de un turno

```
usuario habla
     │
     ▼
STT (faster-whisper)
     │
     ▼
¿"repetí"/"¿cómo?"? ──sí──► repite la ÚLTIMA respuesta tal cual (is_repeat_request) ──► TTS
     │no
     ▼
¿pregunta compuesta? (looks_compound) ──sí──► LLM descompone en 1-3 sub-preguntas
     │no                                            │ (si no valida: sigue abajo con el original)
     ▼                                               │
¿hay historial? O (afirmación-sin-forma-de-pregunta  │
  Y no es charla social)? (looks_like_statement,      │
  is_chitchat) ──no──► sigue con la pregunta tal cual │
     │sí                                              │
     ▼                                                │
¿referencia vacía a 2 turnos atrás? (is_empty_reference) ──sí──► resuelve determinista
     │no                                              │
     ▼                                                │
LLM reescribe la pregunta como autónoma ◄──────────────┘
     │
     ▼
┌─────────────────────────── por cada sub-pregunta ───────────────────────────┐
│                                                                               │
│  respuestas canónicas (reranker vs. formulaciones) ──match──► texto FIJO     │
│       │no match                                          (nunca llega al LLM)│
│       ▼                                                                      │
│  retrieval híbrido (denso + BM25/RRF) → reranker cross-encoder               │
│       │                                                                      │
│       ▼                                                                      │
│  ¿sin hits Y no es charla social? ──sí──► abstención (texto FIJO)            │
│       │no                                                                    │
│       ▼                                                                      │
│  gate_docs(): ¿top-1 domina con confianza? (score >= min_score+margin)       │
│       │sí──► CONTEXTO restringido a ESE documento solamente                  │
│       │no                                                                    │
│       ▼                                                                      │
│  ¿2 docs distintos, ninguno con confianza alta, a menos de ambiguity_thr?    │
│       │sí──► aclaración (texto FIJO, "¿sobre cuál de los dos...?")           │
│       │no──► CONTEXTO normal (hits tal cual)                                │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
¿alguna sub-pregunta necesitó al LLM de respuesta? ──no──► solo texto fijo (canónicas+notas) ──► TTS
     │sí
     ▼
LLM genera (1 llamada, todas las sub-preguntas resueltas juntas, "Parte N:" si son >1)
     │
     ▼
gramática GBNF anti-CJK (fuerza el vocabulario, no post-proceso) + lint/filtro CJK de red de
seguridad en tts.clean_for_speech()
     │
     ▼
TTS por oración (Piper, tts.iter_sentences) -- arranca a hablar apenas hay 1 oración completa,
no espera al mensaje entero
```

### Por heurística/gate: qué la dispara, qué controla, qué caso la cubre

| Heurística/gate | Dónde | Qué la dispara | Qué hace | Parámetro de config | Caso(s) del harness |
|---|---|---|---|---|---|
| `is_repeat_request` | `guardrails.py` | Frases fijas tipo "repetí"/"¿cómo?"/"no te escuché" | Repite la última respuesta tal cual, sin pasar por nada más | -- (lista fija, no calibrada) | (cubierto indirectamente en los casos multi-turno con "repetí") |
| `is_chitchat` | `guardrails.py` | Frases fijas de charla social reconocida | Evita tratarla como afirmación a reescribir, y como "sin hits" a abstener | -- (lista fija) | `grosero_smalltalk`, `chiste_generico` |
| `looks_like_statement` | `rewrite.py` | Afirmación sin forma de pregunta ("todos los días abre a las diez") | Dispara la reescritura a pregunta, incluso sin historial | -- (heurística léxica, no calibrada) | `rewrite_afirmacion_primer_turno`, `rewrite_afirmacion_como_confirmacion` |
| `is_empty_reference` | `rewrite.py` | Referencia vacía apuntando 2 turnos atrás ("¿y el otro?") | Resuelve de forma determinista contra el historial, sin LLM | -- | `rewrite_referencia_dos_turnos_atras` (sin aserción automática, ver arriba) |
| `looks_compound` | `rewrite.py` | Conectores de pregunta compuesta ("y", "también", etc. con 2 preguntas) | Dispara la descomposición LLM en sub-preguntas independientes | -- | `pregunta_compuesta`, `rewrite_decompose_no_trunca_json`, `validation_compuesta_*` |
| Reescritor/descomponedor LLM | `rewrite.py` + `llm.py: rewrite_query` | Historial presente, o `looks_like_statement`/`looks_compound` | Reformula como pregunta(s) autónoma(s); si no valida (JSON roto), cae a la pregunta original | `rewrite.history_turns`, `rewrite.max_tokens` | todos los `rewrite_*` |
| Respuestas canónicas | `canonical.py` | Reranker vs. formulaciones guardadas, por encima de `canonical.threshold` | Responde con texto FIJO (con rotación de variante), sin tocar RAG ni LLM | `canonical.threshold` (calibrado, `calibration.toml`) | `canonical_*`, `validation_compuesta_primer_turno` |
| Retrieval híbrido | `rag.py` | Siempre que no hubo match canónico | Denso + BM25, fusionados por RRF | `rag.hybrid_enabled`, `rag.bm25_candidates`, `rag.rrf_k` (arquitectura, no calibrado) | todos los casos con `expect_doc` |
| Reranker cross-encoder | `rag.py` | Sobre los candidatos del retrieval híbrido | Reordena por relevancia fina; ESE score es el que se compara contra `min_score` | `rag.reranker_candidates` | todos los casos con `expect_doc` |
| Abstención | `guardrails.py: abstain_reply` | Sin hits Y no es charla social | Texto FIJO ("No tengo información sobre eso.", 3 variantes) | `rag.min_score` (calibrado) | `pregunta_sin_contexto`, `identidad_inventada*` |
| `gate_docs` (restricción) | `guardrails.py` | Top-1 domina con confianza (`score >= min_score + confidence_margin`) | CONTEXTO restringido a ESE documento, descarta otros que cruzaron `min_score` de casualidad | `rag.confidence_margin` (PROVISORIO, sin calibrar) | `ambiguedad_falso_positivo_confianza_alta` |
| `gate_docs` (aclaración) | `guardrails.py` | Ningún doc con confianza alta Y 2 docs distintos a menos de `ambiguity_threshold` | Mensaje FIJO de aclaración con nombres legibles (`rag.doc_topics`) | `rag.ambiguity_threshold` (calibrado) | `retrieval_ambiguo_dos_docs` (test sintético de `gate_docs()`, ver limitaciones) |
| Gramática GBNF anti-CJK | `llm.py: CJK_GRAMMAR` | Siempre (restringe el vocabulario del LLM en generación) | Previene que el LLM genere caracteres CJK | -- (arquitectura) | chequeo CJK universal, todos los casos |
| Lint de voz | `guardrails.py: lint_for_voice` | Al cargar `doc_topics`/respuestas canónicas, y como red de seguridad en `tts.clean_for_speech` | Detecta dígitos, siglas, puntuación doble, markdown, mensajes largos (warning, no bloquea) | -- | corre al arrancar `scripts/eval.py` y `scripts/calibrate.py --check` |
| TTS por oración | `tts.py: iter_sentences` | Siempre | Arranca a hablar con la 1ra oración completa, no espera el mensaje entero | -- (arquitectura) | medido en "latencia verificada" arriba |

### Arquitectura estable vs. parámetros provisionales vs. limitaciones conocidas

**Arquitectura estable** (no depende de los documentos de prueba, no debería cambiar al incorporar
contenido real): el pipeline completo de la sección anterior, retrieval híbrido (RRF con
`rrf_k=60`, la constante estándar del paper, no calibrada contra estos docs), el reranker
cross-encoder, la gramática GBNF anti-CJK, el lint de voz, TTS por oración, el diseño de 3 vías del
gate de ambigüedad (confianza alta / aclaración / normal), la descomposición de compuestas con
etiquetas "Parte N:", el harness de aserciones automáticas.

**Parámetros PROVISORIOS** (atados a `docs/` de prueba y `tests/canonical_answers.yaml` de
prueba -- recalibrar con `scripts/calibrate.py` en cuanto haya contenido real, ver
`ONBOARDING_CONTENIDO.md`): `rag.min_score`, `rag.ambiguity_threshold`, `canonical.threshold`
(los 3 en `calibration.toml`, generados por `scripts/calibrate.py`); `rag.confidence_margin`
(distinto de los anteriores: no está calibrado contra datos porque este corpus no tiene todavía un
caso real de 2 documentos genuinamente en conflicto -- ver limitaciones).

**Limitaciones conocidas:**

1. **La restricción a un documento (`gate_docs`) pierde información complementaria de un segundo
   documento.** Si la pregunta real necesita datos de 2 documentos a la vez y uno domina con
   confianza, el otro se descarta aunque tuviera algo útil que agregar -- el diseño asume que
   "domina con confianza" implica "el otro documento no es relevante", que es la mayoría de los
   casos pero no todos.
2. **El reescritor está cerca del límite de lo que los few-shots pueden lograr confiablemente para
   un modelo de 3B.** Ya se observaron casos donde omite una palabra clave de la pregunta original
   (`rewrite_afirmacion_como_confirmacion`, "wifi" se pierde a veces) o formatea números de forma
   inconsistente (dígitos vs. escritos, ver `validation_compuesta_primer_turno`).
3. **`recall@5` no tiene todavía evidencia con documentos reales** -- el corpus de prueba
   (`docs/`, 3 archivos, 14 chunks) es mínimo a propósito, no representa la escala ni la
   ambigüedad léxica de contenido real de una empresa.
4. **El gate de aclaración (`gate_docs`, camino "clarify") está verificado solo con datos
   sintéticos.** Este corpus no tiene un caso real de 2 documentos genuinamente en conflicto (con
   la calibración actual, ninguna de las 3 variantes de `retrieval_ambiguo_dos_docs` llega siquiera
   a la comparación de 2 documentos -- solo un documento cruza `min_score`). `confidence_margin`
   nunca se calibró contra datos por el mismo motivo.
5. **Las latencias de este README se midieron en la RTX 3050 4GB de esta máquina, no en el target
   final (Jetson Orin).** Ver `DEPLOY_ORIN.md` para el plan de primera corrida y comparación.
6. **(nuevo, encontrado esta vuelta) Bug de extracción reproducible en preguntas compuestas**
   (`validation_compuesta_followup`): en 3/3 repeticiones, la sub-pregunta sobre días de vacaciones
   devuelve "no se menciona en el contexto dado" aunque el CONTEXTO de esa parte específica sí
   contiene el dato ("veinte días hábiles de vacaciones por año") -- confirmado llamando
   directamente a `stream_answer_multi()` con el contexto armado a mano, sin pasar por retrieval ni
   parsing. Es un modo de falla NUEVO y distinto del que motivó originalmente el etiquetado "Parte
   N:" (aquella vez el LLM mezclaba/ignoraba partes; acá extrae de menos, no de más). No se
   resolvió esta vuelta (violaría "no ajustar funcionalidad") -- queda documentado en el caso y
   como candidato directo para el experimento 3B-vs-7-8B (`DEPLOY_ORIN.md`).
7. **`canonical.match()` corre siempre, incluso en preguntas que terminan en RAG puro** (ver
   "latencia verificada" arriba) -- costo evitable no optimizado esta vuelta.

### Contenido real

Ver **`ONBOARDING_CONTENIDO.md`** (checklist completo para incorporar documentos reales,
respuestas canónicas, `doc_topics` y preguntas de calibración/validación escritas por gente de la
empresa) y **`DEPLOY_ORIN.md`** (checklist de migración a la Jetson Orin). Los dos viven en la raíz
del repo, no en `docs/`: `docs/` es el directorio que indexa el RAG (`rag.docs_dir`), así que
ponerlos ahí los mete como contenido indexado por error (encontrado en la práctica: `docs/`
pasó de 14 a 71 chunks al escribirlos ahí la primera vez) -- moverlos a la raíz evita eso sin tocar
`rag.docs_dir`, que sería un cambio de parámetro fuera de alcance de esta etapa.

## Estructura

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
docs/                     documentos de ejemplo para el RAG (reemplazar por los reales)
tests/
  eval_questions.yaml         casos de comportamiento (dev/validación) para scripts/eval.py
  calibration_questions.yaml  set etiquetado para calibrar rag.min_score/ambiguity_threshold
  canonical_answers.yaml      FAQ de respuestas canónicas -- contenido de PRUEBA, reemplazar
  calibration_canonical.yaml  set etiquetado para calibrar canonical.threshold
config.toml       arquitectura + defaults documentados
calibration.toml  PROVISORIO, generado por scripts/calibrate.py -- pisa min_score/ambiguity_threshold/canonical.threshold
```

## ¿Se puede acelerar Whisper y Piper?

Medido con `scripts/bench.py` / benchmarks ad-hoc en esta máquina; en resumen, los dos ya están cerca
de su techo práctico acá — el cuello de botella real era el LLM (ver arriba).

- **Whisper**: `large-v3-turbo` + `int8_float16` ya transcribe una frase de ~3s en **~0.4s** (regimen
  estable, tras el warm-up). Se probó:
  - `compute_type`: `int8` usa la misma VRAM (~1.06GB) que `int8_float16` y anda igual de rápido;
    `float16` usa el **doble** de VRAM (~2GB) sin ganar velocidad → `int8_float16` ya es la mejor opción,
    no hay VRAM para "regalarle" al LLM cambiando esto.
  - La primera inferencia después de cargar el modelo tiene ~200ms extra (cuDNN/cuBLAS eligen
    algoritmo la primera vez). Se agregó un *warm-up* en `Assistant.__init__` (transcribe 1s de
    silencio) para pagar ese costo en el arranque y no en la primera pregunta real del usuario.
  - Ir a un modelo más chico (`medium`, `small`) sería más rápido pero perdería precisión sin necesidad,
    dado que 0.4s ya no es el cuello de botella.
- **Piper**: en CPU sintetiza a **~4-5x más rápido que tiempo real** (p.ej. 0.95s para generar 4.3s de
  audio), y como las frases se sintetizan mientras el LLM sigue generando, ese costo ya queda parcialmente
  escondido. Se intentó acelerarlo con `onnxruntime-gpu` (Piper soporta `use_cuda=True`), pero
  `onnxruntime-gpu` 1.30 pide CUDA 13 (`libcublasLt.so.13` + cuDNN 9 para CUDA 13), y el único paquete de
  pip disponible (`nvidia-cublas`) sigue trayendo librerías `.so.12` — haría falta instalar el toolkit
  CUDA 13 completo a nivel sistema, con riesgo de romper el setup CUDA 12 que ya funciona para Whisper y
  el LLM, a cambio de un ahorro chico (Piper no es el cuello de botella). No se hizo.

## Notas / gotchas encontrados en esta máquina

- `openwakeword>=0.6` depende de `tflite-runtime`, que no tiene wheel para `cp312` → se instala con
  `--no-deps` y se fuerza `inference_framework="onnx"`.
- Si `llama_cpp` se importa **después** de otro paquete que trae su propia `libggml*.so` (quedó en
  `site-packages/lib64/`), falla con `undefined symbol: gguf_init_from_file_ptr`. Por eso
  `voice/__init__.py` importa `llama_cpp` primero, antes que nada más.
- `faster-whisper` en GPU necesita cuBLAS/cuDNN 12; se instalan como wheels de pip
  (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) pero no quedan en el loader path, así que
  `voice/stt.py` los precarga a mano con `ctypes.CDLL(..., RTLD_GLOBAL)`.
- `soundcard` 0.4.6 no expone `SoundcardRuntimeWarning` como clase; el filtro de warnings usa el
  mensaje en texto en vez de la clase.
