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

## Testear el LLM/RAG con scripts/eval.py

Herramienta de desarrollo (no de producción): corre una lista de casos de prueba contra el LLM/RAG real
(sin STT ni micrófono) y vuelca todo a un Markdown en `eval_results/` — no calcula pass/fail solo, el
juicio de si cada caso está bien lo hace un humano (o Claude) leyendo el resultado.

```bash
.venv/bin/python scripts/eval.py                       # corre tests/eval_questions.yaml completo
.venv/bin/python scripts/eval.py --out mi_corrida.md
```

Los casos viven en `tests/eval_questions.yaml` — se van agregando ahí a medida que aparecen bugs
nuevos (cada caso tiene `note`/`expect` explicando qué prueba y por qué). Soporta `repeat: N` por caso:
como `temperature=0.6` hace que las respuestas no sean determinísticas, un solo intento no alcanza para
saber si algo "se arregló" — repetir varias veces da una idea real de la tasa de fallo.

**Última corrida completa (31 ejecuciones, 22/09), resumen:**

| Caso | Resultado |
|---|---|
| Horarios, reset de contraseña, vacaciones, trabajo remoto (preguntas directas) | ✓ Pass |
| Pregunta sin ningún doc relacionado | ✓ Pass (admite que no sabe) |
| Follow-ups cortos y comentarios sin tema en medio de la charla | ✓ Pass (fix de sticky-hits sostiene) |
| Pregunta repetida dos veces seguidas | ✓ Pass (consistente) |
| Comentario grosero / small talk | ✓ Pass (no fuerza contenido de los docs) |
| Cambiar de día en un follow-up ("¿y el sábado?" después de hablar del domingo) | ✗ **5/5** corridas fallaron en dar una respuesta limpia sobre el día correcto |
| "¿Quién es Juan Carlos?" (nombre sin doc, pero real y famoso) | ✗ **5/5** inventó que es el Rey de España (con fechas de reinado distintas e incorrectas cada vez) |
| "¿Quién es María Fernández?" (nombre común, sin referente famoso obvio) | ✗ 1/3 inventó una actriz; 2/3 admitió que no sabía |
| Chiste sin relación a los docs | ✗ **2/5** cambió de idioma a mitad de frase (remate en chino) |
| Pregunta que roza dos documentos a la vez (horario de oficina + política de trabajo remoto) | ✗ inventó que "los sábados son de presencia obligatoria" (dato falso, mezcla dos políticas distintas) |
| Pregunta compuesta (dos horarios en una sola pregunta) | ✗ dio un horario de cierre de sábado incorrecto (14hs en vez de 13hs) aun con el contexto correcto |

Quedan documentados como casos de regresión en `tests/eval_questions.yaml`; los fixes se discuten aparte.

## Estructura

```
voice/
  audio.py      mic/parlantes (PulseAudio vía soundcard)
  wakeword.py   openWakeWord (ONNX)
  vad.py        Silero VAD + segmentador de frases
  stt.py        faster-whisper (GPU)
  rag.py        chunking + embeddings + retrieval
  llm.py        llama.cpp (CPU), streaming
  tts.py        Piper, streaming por frase
  pipeline.py   orquesta todo (clase Assistant)
scripts/
  download_models.py   descarga todos los modelos
  bench.py              benchmark end-to-end sin mic
docs/           documentos de ejemplo para el RAG (reemplazar por los reales)
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
