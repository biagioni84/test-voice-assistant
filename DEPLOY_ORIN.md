# Migración a Jetson AGX Orin

Checklist para portar el POC (desarrollado y verificado en una RTX 3050 4GB laptop) a la Orin.
Ninguno de estos pasos se hizo todavía en este repo -- es la guía para cuando llegue el hardware,
no un reporte de resultados.

## 1. Compilación de llama.cpp con CUDA en Jetson

- **JetPack**: usar la versión de JetPack que corresponda al modelo de Orin (AGX Orin 32/64GB) y
  confirmar la versión de CUDA que trae (JetPack 6.x trae CUDA 12.x, compatible con lo que ya usa
  este proyecto en la 3050 -- confirmar la versión exacta antes de compilar).
- **Compute capability**: la Orin usa Ampere (compute capability 8.7), distinto de la 3050 (8.6) --
  ajustar `CMAKE_CUDA_ARCHITECTURES` al compilar (`87` en vez de `86`, ver el flag documentado en
  el README para la compilación en esta máquina).
- **Memoria unificada**: la Orin tiene memoria UNIFICADA entre CPU y GPU (no VRAM dedicada como la
  3050) -- el presupuesto de "VRAM" que gobernó las decisiones de este POC (quantización,
  `n_gpu_layers` parcial, modelo 3B en vez de 7-8B) probablemente ya no aplica de la misma forma.
  Esto es la motivación central del experimento 3B vs. 7-8B de la sección 4.
- Confirmar que la build de llama.cpp para Jetson soporta los mismos flags que ya usa este proyecto
  (`--parallel 2`, `--slot-save-path`, grammar GBNF) -- no debería haber diferencias, pero
  verificarlo en la primera compilación en vez de asumirlo.

## 2. Modelos y cuantizaciones

- Empezar con el MISMO modelo/cuantización que ya está calibrado y verificado en este repo
  (`Qwen2.5-3B-Instruct-Q4_K_M`) para tener una comparación limpia antes de cambiar nada más.
- Con memoria unificada y más margen, evaluar cuantizaciones menos agresivas (Q5/Q6/Q8) del mismo
  3B antes de saltar a un modelo más grande -- aísla si una mejora viene de "menos cuantización" o
  de "más parámetros" (ver experimento 3B vs. 7-8B más abajo).

## 3. Reranker en GPU, y volver a `top-10`

- En la 3050, el reranker corre en CPU a propósito (no entraba en VRAM junto con Whisper + LLM) --
  ver README, sección "Reranker cross-encoder". En la Orin, con memoria unificada y más margen,
  evaluar correrlo en GPU (`fastembed`/ONNX Runtime con proveedor CUDA) -- debería bajar bastante
  la latencia del reranker, que en la 3050 es el segundo costo más grande del pipeline después del
  LLM.
- Si el reranker en GPU es notablemente más rápido, volver a subir `rag.reranker_candidates` (bajado
  de 10 a 5 en la 3050 por latencia, ver README) y volver a medir recall@5 con
  `scripts/calibrate.py` -- más candidatos para el reranker debería ayudar en corpus más grandes
  (los reales, no el placeholder de este repo).

## 4. Primera corrida: comparar calidad y fijar objetivo de latencia

1. `python scripts/eval.py` (la suite completa) en la Orin, **con el mismo `calibration.toml`** que
   se usa en la 3050 (mismos documentos placeholder, o los reales si ya se migraron -- ver
   `ONBOARDING_CONTENIDO.md`). Con `temperature=0`, las respuestas deberían **coincidir** con
   las de la 3050 (mismo modelo, mismos pesos, msmo muestreo determinístico) -- si no coinciden,
   investigar antes de seguir (podría ser una diferencia de build/cuantización real, no solo
   hardware).
2. `python scripts/bench.py` para tener los números de latencia de la Orin (tok/s, tiempo a primer
   token, fin de voz → primer audio) -- comparar contra los ya documentados de la 3050 (ver README).
3. Con esos dos resultados (calidad igual, latencia medida), **recién ahí** fijar un objetivo de
   latencia real para el target -- no antes, porque hasta no tener el número real de la Orin
   cualquier objetivo es una suposición.

## 5. Experimento pendiente: 3B vs. 7-8B

Diferido explícitamente durante todo el desarrollo en la 3050 (no entraba un 7-8B con margen
suficiente de VRAM, ver README, "Decisiones y por qué"). Con la memoria unificada de la Orin, este
es el momento de retomarlo -- empezando por el **reescritor**, no por el LLM de respuesta:

- El reescritor es el rol donde más de cerca se vio al 3B llegar a su límite durante este
  desarrollo (ver README, sección de bugs en vivo -- pérdida de sujeto, overfitting de pocos
  few-shots, "wifi" que a veces se pierde en la reescritura de confirmación). Es la hipótesis más
  concreta y medible de "¿un modelo más grande resuelve esto sin más prompt engineering?".
- **Medir con el harness de validación** (`--split validation` en `scripts/eval.py`), no con el de
  dev -- el split dev se ajustó mirando el comportamiento del 3B, así que no es una comparación
  limpia entre modelos. El split validation, al ser holdout, sí lo es.
- Si el 7-8B generaliza mejor en el reescritor con el MISMO prompt (sin ajustar nada), es la señal
  más clara de que vale la pena evaluarlo también para el LLM de respuesta. Si no mejora, es
  evidencia de que el límite no era el tamaño del modelo sino otra cosa (calidad del prompt,
  necesidad de más contexto, etc.) -- documentarlo de cualquier forma, es información real.
- Repetir la comparación para el LLM de respuesta usando los mismos criterios (split validation,
  mismo `calibration.toml`, mismos documentos).
