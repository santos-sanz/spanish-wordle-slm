# Registro denso de aprendizaje: Spanish Wordle SLM en MLX

**Actualizado:** 2026-08-18 (Europe/Madrid, continuación)  
**Estado:** experimento técnicamente reproducible; objetivo competitivo todavía no demostrado.  
**Regla principal:** este documento debe leerse completamente antes de iniciar otro entrenamiento, cambiar el prompt, cambiar el parser o abrir de nuevo el test oculto.

## 1. Pregunta, alcance y criterio que no se puede relajar

La pregunta es si un único adapter QLoRA de `LiquidAI/LFM2.5-2.6B-MLX-6bit`, servido localmente con MLX, puede superar a `deepseek/deepseek-v4-pro-0813` en Wordle español. La afirmación solo es válida si el mismo adapter gana al rival en los dos tracks competitivos:

- **Pure:** historial público de jugadas y feedback → una palabra; no se consulta ninguna herramienta ni lista de candidatos durante la decisión.
- **Agent:** puede llamar `get_candidates` como máximo una vez por turno; no se expone `best_guess` ni una métrica de la mejor jugada.
- **Oracle:** `best_guess()` es únicamente techo no competitivo y no puede rescatar la afirmación.

La comparación oficial usa exactamente los mismos 124 objetivos del split oculto, temperatura 0, semilla `20260814`, máximo remoto de 512 tokens, fallback de OpenRouter desactivado y hasta dos reparaciones por turno. Gana primero la tasa de victorias; solo si empata se usa la media de intentos, contando una derrota como siete. La primera métrica diferente debe favorecer al SLM con intervalo bootstrap pareado del 95 % estrictamente positivo. Un resultado parcial, una muestra de validación o un benchmark interrumpido no es una victoria.

## 2. Estado congelado y artefactos de referencia

- Modelo base: `LiquidAI/LFM2.5-2.6B-MLX-6bit`.
- Revisión HF: `95f71f1c30e3247bc7f042c6fd64d7ca60258780`.
- Peso local: `models/LFM2.5-2.6B-MLX-6bit/model.safetensors` (2.191.851.544 bytes; no se copia a Git).
- Adapter oficial congelado: `adapters/selected`, originalmente `pure-clean-refinement`, LoRA rank 16, scale 32, dropout 0.05, últimas 16 capas.
- SHA256 oficial del adapter: `6c7a48c8fa56b1f08fa4f902e7c044e520aec672c71952ff489db85113cb71c6`.
- Manifiesto: `artifacts/adapter-manifest.json` y `artifacts/adapter-selected-sha256.txt`.
- El adapter DPO (`adapters/dpo-selected`) existe, pero no es el seleccionado: su comprobación Pure de validación fue 0/5 y no debe promoverse por su loss baja.
- Los pesos base, cachés HF, `.env` real y checkpoints no deben entrar en GitHub. Nunca imprimir ni copiar la clave de OpenRouter.

## 3. Datos, solver y controles que sí han funcionado

Se fijaron las fuentes por commit, se normalizó a minúsculas NFC, se reparó el mojibake de `ñ`, se descartaron entradas no válidas, se eliminaron duplicados y se comprobó que las 617 soluciones históricas pertenecen a los 11.180 intentos válidos. La procedencia, hashes y recuentos están en `data/processed/provenance.json`.

El motor implementa feedback exacto en dos pasadas para letras repetidas, codificación de los 243 patrones en `uint8`, matriz intento × solución, filtrado de candidatos y solver determinista con entropía/minimax y memoización. Las unitarias cubren duplicados, `ñ`, entradas inválidas, filtrado, desempates, matriz y splits. `uv run pytest -q` llegó a 17 pruebas verdes; `npm run test` y `npm run build` también han sido verdes en las comprobaciones relevantes.

El split es determinista con semilla `20260814` y proporción 70/10/20. Las trayectorias y datasets derivados marcan `hidden_test_used: false`; el test oculto se abrió únicamente para el benchmark oficial después de congelar adapter, parser, prompt y configuración.

## 4. Benchmark oficial congelado (DeepSeek Pro, 124 objetivos)

Archivos fuente: `artifacts/benchmark/slm-pure.json`, `slm-agent.json`, `deepseek-pure.json`, `deepseek-agent.json`. El dashboard reproducible está en `artifacts/benchmark/competition-dashboard.png` y `.svg`; el progreso acumulado está en `competition-progress.png` y `.svg`.

| Track | SLM | DeepSeek Pro | Lectura |
|---|---:|---:|---|
| Pure, victorias | 2/124 | 0/124 | La diferencia observada es pequeña y no concluyente. |
| Pure, tasa | 1,61 % | 0 % | Bootstrap pareado 95 %: `[0,00 %, 4,03 %]`; no estrictamente positivo. |
| Pure, media con pérdidas=7 | 6,92 | 7,00 | Ventaja de 0,081 turnos; no es la primera métrica decisiva porque la tasa difiere. |
| Agent, victorias | 115/124 | 6/124 | Ventaja amplia del SLM. |
| Agent, tasa | 92,74 % | 4,84 % | Bootstrap pareado 95 %: `[82,26 %, 93,55 %]`; sí favorece al SLM. |
| Agent, media con pérdidas=7 | 3,97 | 6,79 | Confirma la ventaja, pero no sustituye al criterio de Pure. |

El informe canónico es `artifacts/benchmark/summary.md` y declara explícitamente `Experimental objective achieved: no`. No cambiarlo a “sí” mientras Pure no cumpla el intervalo pareado.

## 5. Qué salió bien

1. **Reproducibilidad local:** `uv`, lockfile, instalación aislada y MLX funcionan en Apple Silicon; el checkpoint carga y la memoria observada permaneció alrededor de 3,5 GB, por debajo del límite de 14 GB.
2. **Watchdog y visualización:** la ejecución principal llegó a 5.355 iteraciones, con estado persistente, checkpoints, JSONL y curva de pérdida en `artifacts/training/loss-curve.png/.svg`. La dashboard distingue loss de train/validación, memoria, checkpoints y progreso.
3. **Agent aprendido:** el SLM sí aprendió el protocolo de herramienta. 115/124 victorias demuestra que el modelo puede seguir una trayectoria cuando recibe candidatos, por lo que el servidor, el parser de herramientas, el bridge y el feedback no están globalmente rotos.
4. **Harness comparable:** Pi Agent Core `0.84.2`, mismo loop, mismas reparaciones, telemetría de tokens/coste/proveedor, persistencia por partida y comparación pareada.
5. **Criterio honesto:** el informe conserva el fallo de Pure; no se ha alterado el test, la métrica ni el rival para fabricar una conclusión.
6. **Entrega y seguridad:** el repositorio y el repositorio privado de Hugging Face fueron preparados con manifiesto/licencia/atribución; `.env.example` se mantiene separado del `.env` real y las credenciales no se han escrito en resultados.

## 6. Qué salió mal y evidencia

### 6.1 El problema es específico de la política Pure

El modelo suele arrancar con palabras frecuentes o memorizadas (`careo`, `lenta`, `macla`, `crack`, `abuso`, `abusa`) y luego repite o elige palabras válidas pero no compatibles. El Agent, con el mismo base y el mismo bridge, sí resuelve 115/124. Esto apunta a una falta de inferencia de restricciones/selección de acción bajo historial, no a un fallo de feedback, servidor o lista de palabras.

### 6.2 La pérdida de token no equivale a ganar partidas

El DPO llegó a `best_validation_loss=0.035937` y `validation_reward_accuracy=0.9875` en sus pares, pero su comprobación Pure fue 0/5. El DPO mejora el margen de las respuestas elegidas frente a respuestas rechazadas del dataset, no garantiza que el argmax en un estado real produzca la palabra objetivo. No usar una loss baja como criterio de promoción.

### 6.3 Las etiquetas de primer turno son un cuello de botella

El dataset evita supervisar aperturas perturbadas porque el mismo historial vacío tendría cuatro etiquetas incompatibles. Como consecuencia, el adapter conserva una apertura frecuente (en el seleccionado, `careo`) en vez de aprender una política de apertura claramente optimizada. Forzar `audio` externamente como diagnóstico dio 0/10 en validación: el primer turno por sí solo no explica el fracaso.

### 6.4 Más cobertura SFT no produjo transferencia

Se generaron datasets Pure deterministas, plain, con restricciones legibles, con política óptima y currículos amplios (hasta 102.079 registros de entrenamiento). Ramas rank 30/32 y continuaciones desde base/adapter tampoco demostraron mejora consistente; el patrón de salida siguió siendo parecido. Los ejemplos de 5/10 o 10/10 sirven solo de señal rápida, no de selección final.

### 6.5 Añadir restricciones derivadas no bastó

Se añadió opcionalmente un resumen de patrón, letras presentes, ausentes y mal colocadas calculado solo a partir del historial. `readable-constraints-valid10.json` y `deterministic-constraints-valid10.json` dieron 0/10. El modelo no internalizó esa línea como filtro operativo.

### 6.6 Thinking nativo perjudica la interfaz corta

Con la plantilla nativa y `WORDLE_SLM_ALLOW_THINKING=1`, los ensayos `thinking-valid5.json` y `thinking2-valid1.json` produjeron historiales vacíos y acciones inválidas. La plantilla `configs/chat_template_no_think.jinja`, el bloqueo del token de apertura de `<think>` y el formato JSON corto deben conservarse para este modelo.

### 6.7 Temperatura, penalización y parser no arreglan la política

Temperaturas 0,1/0,3/0,7, `repetition_penalty=1.25`, `max_tokens=32` y el parser que extrae todos los tokens de cinco letras fueron probados. Los resultados fueron iguales o peores; no confundir una reducción de tokens inválidos con una mejora de tasa de victoria.

### 6.8 El word bank estático no es solución Pure

El experimento opcional `WORDLE_SLM_WORD_BANK=1` insertó las 616 respuestas en el system prompt. Solo alcanzó 0/3 antes de interrumpirse (`wordbank-valid10.json`, `complete=false`), además de consumir mucha latencia. No debe usarse como evidencia ni activarse en un benchmark oficial: introduce información global que se parece demasiado a una herramienta y no resolvió el problema.

### 6.9 Reparación de varias candidatas es prometedora pero aún no concluyente

El modo opcional `WORDLE_SLM_REPAIR_LIST=1` pidió hasta 12 palabras generadas por el propio modelo cuando la salida inicial era inválida. La ejecución se interrumpió en 9/10: 1/9 victorias, 6,44 turnos medios, 23 inválidas. El artefacto `artifacts/benchmark/repair-list-valid10.json` tiene `complete=false`; no es comparable ni suficiente para seleccionar adapter. La pequeña señal no justifica abrir el test.

### 6.10 LR demasiado alto desestabiliza

El ensayo `pure-deterministic-plain-highlr` fue detenido cuando la loss de train subió aproximadamente a 7,189. Mantener `2e-5` o más sin smoke/evaluación frecuente repite el error de optimizar agresivamente una interfaz que ya muestra colapso de acciones. El límite de seis horas no autoriza a consumir todo el presupuesto en una configuración inestable.

### 6.11 OpenRouter tiene un límite externo

La revalidación remota posterior fue rechazada con HTTP 403 por límite de la clave. Los cuatro benchmarks Pro completos anteriores siguen siendo la referencia oficial; no lanzar otra comparación remota hasta que la clave tenga cuota y se pueda registrar el proveedor efectivo sin guardar identificadores sensibles.

## 7. Tabla de experimentos de validación relevantes

| Artefacto | Muestra | Resultado | Decisión |
|---|---:|---:|---|
| `validation-final-pure.json` | 61 | 0 victorias, 173 inválidas | Baseline seleccionado de validación; no promover. |
| `pure-clean-long-valid61.json` | 61 | 3 victorias, 156 inválidas | Mejor señal pública, pero su hidden `slm-pure-clean-long.json` fue 1/124; no promover. |
| `interp75-valid61.json` | 61 | 2 victorias, 159 inválidas | Interpolación no supera claramente al seleccionado. |
| `pure-curriculum-mix-valid61.json` | 61 | 1 victoria, 153 inválidas | No promover. |
| `readable-constraints-valid10.json` | 10 | 0 victorias | Rechazado. |
| `deterministic-constraints-valid10.json` | 10 | 0 victorias | Rechazado. |
| `pure-base-deterministic-valid10.json` | 10 | 1 victoria, 25 inválidas | Warm-start desde base no resuelve el problema. |
| `current-pure-dpo-check.json` | 5 | 0 victorias | Loss DPO no predice comportamiento Pure. |
| `fixed-audio-valid10.json` | 10 | 0 victorias | Apertura fija `audio` empeora; no es el arreglo. |
| `repair-list-valid10.json` | 9/10, incompleto | 1 victoria | Solo hipótesis; completar validación bajo un nombre nuevo si se vuelve a estudiar. |

No usar muestras de 1, 5 o 10 objetivos como prueba de que un adapter gana. La selección debe requerir una comparación completa del split de validación (61 objetivos) y una mejora reproducible en victorias, inválidas y media, manteniendo el mismo parser y prompt.

## 8. Cambios de código experimentales que deben permanecer desactivados

En `agent/src/player.ts` existen interruptores diagnósticos opcionales:

- `WORDLE_SLM_FIXED_OPENING`: bypass de la generación del modelo en el primer turno; no usar en el benchmark oficial.
- `WORDLE_SLM_REPAIR_LIST=1`: lista de hasta 12 palabras en reparaciones Pure; solo experimental y requiere un run completo de validación antes de considerar cualquier cambio.
- `WORDLE_SLM_WORD_BANK=1`: word bank estático; no usar para la afirmación Pure.
- `WORDLE_SLM_CONSTRAINTS=1`, `WORDLE_SLM_TEMPERATURE`, `WORDLE_SLM_MAX_TOKENS`, `WORDLE_SLM_ALLOW_THINKING` y `WORDLE_SLM_REPETITION_PENALTY`: todos alteran la configuración congelada si se activan.

El default oficial debe seguir siendo: sin word bank, sin fixed opening, sin constraints, sin thinking, temperatura local 0, max local 128, parser JSON + tokens de cinco letras y adapter `adapters/selected`.

## 9. Reglas de no repetición antes de otro experimento

1. Leer este archivo completo y comprobar que se entiende qué configuración se está cambiando.
2. Comprobar `ps aux | rg 'mlx_lm.server|mlx_lm.*lora|benchmark.ts'` y no dejar servidores/training huérfanos.
3. Guardar el SHA del adapter oficial antes de tocar nada; nunca sobrescribir `adapters/selected` durante una exploración.
4. Definir de antemano hipótesis, prompt, dataset, adapter de inicialización, seed, LR, capas, secuencia, criterio y nombre de artefacto.
5. Ejecutar primero smoke, luego validación completa de 61 objetivos. Un 1/5, 1/10 o archivo `complete=false` no permite promoción.
6. No abrir el split oculto para una rama que no mejore claramente validación; el test solo se abre tras congelar adapter, parser, prompt, sampling y harness.
7. No introducir candidatos, `best_guess`, word bank ni solver en Pure. Si se estudia decoding restringido, documentarlo como una nueva política y no compararlo con la referencia sin congelarlo también para el rival.
8. Evaluar acción y partida, no solo loss: victorias, media con pérdidas=7, inválidas, repeticiones, latencia y distribución de palabras.
9. Mantener LR conservador (`<=5e-6` para una continuación salvo evidencia), `max_seq_length=512` salvo presión de memoria y evaluación cada 50–100 iteraciones.
10. Si un método produce una loss espectacular pero no cambia las acciones en validación, detenerlo y registrar la discrepancia en vez de ampliar iteraciones ciegamente.
11. No repetir una llamada DeepSeek completa mientras OpenRouter devuelva límite de cuota; hacer primero una llamada smoke y no escribir errores que contengan secretos o identificadores de clave.
12. Al terminar, volver a ejecutar `npm run build`, `uv run pytest -q`, Ruff dirigido, checksum del adapter y `npm run report`; el informe debe conservar “no” hasta que el criterio estadístico sea verdadero.

## 10. Próximo experimento permitido (todavía no ejecutado)

La siguiente acción razonable no es otro SFT ciego. Primero hay que decidir entre:

- completar una sola validación Pure de 61 con `WORDLE_SLM_REPAIR_LIST=1`, usando un artefacto nuevo y sin cambiar el adapter; o
- implementar un método de optimización de acciones (GRPO-lite/REINFORCE o decoding restringido) en una rama aislada, con smoke y validación antes de considerar cualquier benchmark oculto.

Ambas opciones requieren declarar previamente si siguen siendo Pure según el contrato. Si usan una lista de candidatos, solver, word bank o selección externa, deben clasificarse como una nueva política/track y no como la referencia Pure. No hay autorización metodológica para afirmar una victoria solo porque el modo Agent ya gana.

## 11. Comandos de lectura y verificación

```bash
sed -n '1,260p' docs/experimental-learning-log.md
ps aux | rg 'mlx_lm.server|mlx_lm.*lora|benchmark.ts' | rg -v rg
shasum -a 256 adapters/selected/adapters.safetensors
npm run build
uv run pytest -q
uv run wordle-slm report
```

Este registro es una memoria de proceso, no una sustitución de los JSONL/JSON de resultados. Los artefactos de benchmark completos, sus `summary`, los logs de entrenamiento y los manifests son la evidencia primaria.

## 12. Continuación posterior: nuevo rival, nuevas pruebas y estado pendiente

Esta sección se añadió después de volver a leer las reglas anteriores. Su función es impedir que una nueva sesión repita pruebas ya descartadas o confunda un proceso iniciado con un resultado conseguido.

### 12.1 El modelo rival efectivo debe verificarse, no suponerse

La auditoría del `.env` actual solo encuentra `OPENROUTER_MODEL=deepseek/deepseek-v4-pro-0813`. Por tanto, el “modelo más potente” que figura en la configuración coincide con el rival Pro del benchmark oficial anterior; no hay un segundo identificador configurado. Una smoke de un objetivo confirmó que el harness resolvió ese mismo ID, pero OpenRouter respondió HTTP 403 por `Key limit exceeded (total limit)`. El JSON temporal se eliminó inmediatamente porque el texto de error contenía un identificador de clave. La evidencia remota válida continúa siendo la de los cuatro archivos Pro completos; no se debe presentar la smoke fallida como benchmark ni volver a guardar el error.

Antes de un benchmark nuevo hay que registrar sin secretos: el ID efectivo, proveedor, temperatura, seed, fallback, límite de tokens, número de objetivos y cuota disponible. Si el ID del `.env` no cambia, no llamar a eso “nuevo modelo”. Si cambia, la comparación completa debe rehacerse para ambos tracks sobre los mismos 124 objetivos y generar un informe nuevo, sin mezclarlo con `summary.md` del rival Pro.

### 12.2 Decodificación Pure multi-propuesta: resultado nulo

Se añadió `WORDLE_SLM_CANDIDATE_LIST=1` como variante opcional: el SLM recibe únicamente el historial, genera hasta 12 palabras ordenadas y el parser selecciona la primera palabra válida no repetida. No hay `get_candidates`, `best_guess` ni word bank en esta variante. Sobre los primeros 10 objetivos de validación obtuvo 1/10, media 6,50 e inválidas 25, prácticamente igual al baseline de 1/10 y no suficiente para promover. El modo queda desactivado por defecto. No abrir el test oculto con él.

El diagnóstico anterior de `WORDLE_SLM_FIXED_OPENING=audio` quedó en 0/10, así que una apertura manual tampoco arregla Pure. `WORDLE_SLM_REPAIR_LIST=1` alcanzó provisionalmente 1/9 antes de ser interrumpido; su archivo declara `complete=false` y no es evidencia comparable. Los tres resultados enseñan que generar más texto o fijar la primera palabra no sustituye el aprendizaje de restricciones.

### 12.3 Rama word-only: hipótesis de interfaz, no resultado

Se creó una rama aislada para comprobar si la envoltura JSON distrae al modelo:

- `scripts/build_word_only.py` transforma exactamente `data/pure-refinement` y conserva sus historiales, etiquetas, seed y procedencia; no usa el test oculto.
- `data/pure-word-only/{train,valid}.jsonl` contiene 8.757/1.261 registros y respuesta supervisada de una sola palabra.
- `configs/pure-word-only.yaml` usa el modelo fijado, LoRA rank 16/scale 32, últimas 16 capas, `max_seq_length=384`, `learning_rate=5e-6`, 1.200 iteraciones y `resume_adapter_file=adapters/selected/adapters.safetensors`.
- `WORDLE_SLM_WORD_ONLY=1` cambia de forma opcional el system prompt y las reparaciones locales; el modo JSON oficial sigue siendo el default.
- La alineación de máscara se comprobó: prompt 142 tokens y suffix supervisado `banco<|im_end|>`; no hay desalineación de chat template.
- El entrenamiento comenzó en `artifacts/runs/training/pure-word-only.log`: 13,042 M parámetros entrenables, 3,068 GB de pico observado, validación inicial 6,464 en iteración 1 y train loss 4,186 en iteración 20. Todavía no existe resultado de partida ni selección de checkpoint.

Hasta que termine una evaluación completa de validación, esta rama es solo una hipótesis. No copiar sus pesos a `adapters/selected`, no usar una loss inicial o un checkpoint intermedio para afirmar mejora y no lanzar el test oculto con el proceso aún en marcha. Si la rama no cambia claramente la tasa de victoria y las inválidas en los 61 objetivos de validación, se conserva como experimento fallido y se detiene.

### 12.4 Protocolo obligatorio después de esta actualización

1. Leer de nuevo este archivo entero antes de modificar prompt, parser, dataset, config o adapter.
2. Comprobar si `pure-word-only` sigue entrenando antes de lanzar o detener cualquier otro proceso; no crear dos procesos Metal que compitan por memoria.
3. Si sigue entrenando, observar al menos los checkpoints/evaluaciones de 100, 200 y 400 iteraciones. Parar solo por NaN, Metal/memoria, pérdida claramente divergente o falta de señal documentada; no interrumpir por impaciencia.
4. Probar primero `WORDLE_SLM_WORD_ONLY=1` en validación (61 objetivos, `complete=true`) con un nombre nuevo. Comparar contra `validation-final-pure.json` y `pure-clean-long-valid61.json` usando exactamente el mismo parser y reparaciones, salvo el contrato declarado.
5. Promover un candidato únicamente si la mejora es reproducible en validación completa y el checksum/config se congela. El adapter oficial debe seguir teniendo SHA `6c7a48c8...71c6` hasta ese momento.
6. Solo después de congelar candidato, parser, prompt y sampling, resolver la cuota del rival y ejecutar los cuatro benchmarks ocultos; generar `competition-dashboard`, `competition-progress`, CSV, JSON, Markdown y artefacto técnico.
7. Ejecutar build, tests, Ruff, checksum, revisión de secretos y `npm run report`; el campo `success` solo puede pasar a `true` si Pure y Agent tienen intervalos pareados positivos.

La existencia de esta rama de entrenamiento no significa que el objetivo esté más cerca estadísticamente: la única evidencia de avance será una victoria de validación completa y, después, una victoria oculta estadísticamente significativa en ambos tracks.

## 13. Estado de la rama word-only tras la última interrupción de sondeo

El entrenamiento de `configs/pure-word-only.yaml` se inició una sola vez, desde `adapters/selected`, y el sondeo de la terminal fue interrumpido por el cliente en dos ocasiones; eso no equivale a una interrupción deliberada del proceso MLX. La última salida observada fue:

| Iteración | Train loss | Val loss | Pico memoria | Checkpoint |
|---:|---:|---:|---:|---|
| 1 | — | 6,464 | 3,068 GB | no |
| 20 | 4,186 | — | 3,068 GB | no |
| 60 | 1,093 | — | 3,068 GB | no |
| 80 | 1,005 | — | 3,068 GB | no |
| 100 | 1,020 | 1,884 | 3,068 GB | `0000100_adapters.safetensors` |
| 120 | 0,699 | — | 3,120 GB | no |
| 140 | 1,067 | — | 3,120 GB | no |

La caída de validación de 6,464 a 1,884 es una señal de aprendizaje de la interfaz de salida, no una señal de que ya resuelva Wordle. Todavía faltan las partidas de validación y no existe evidencia de tasa de victoria. No lanzar otra sesión Metal, no borrar el checkpoint y no copiarlo a `adapters/selected` hasta auditar el proceso existente y completar una evaluación de 61 objetivos con `complete=true`.

**Preflight obligatorio, incluso si el usuario vuelve a pedir “continúa”:** leer las secciones 1–13; comprobar el PID/log del entrenamiento actual; dejar que termine o detenerlo de forma explícita y segura; verificar que no hay dos servidores MLX; ejecutar primero un benchmark público de validación; y solo después decidir si hay que continuar, rechazar o promover. Una pérdida de formato menor o un sondeo cancelado no debe convertirse en otra rama improvisada.

## 14. Resultados posteriores a la rama word-only

La continuación desde el checkpoint 100 (`configs/pure-word-only-cont.yaml`) se detuvo de forma controlada en su primer checkpoint de 100 iteraciones adicionales: la validación pasó de 1,677 al comienzo a 2,025, sin mejora monotónica. Su evaluación Pure con `WORDLE_SLM_WORD_ONLY=1` obtuvo 0/10, 23 acciones inválidas y media puntuada 7. No se promovió ningún peso; `adapters/selected` conserva el SHA oficial.

Se probó también `scripts/probe_constrained_decoder.py`, un diagnóstico separado que restringe la generación a palabras legales del vocabulario global, pero **no** filtra por feedback ni consulta candidatos. El decoder produjo palabras válidas, eliminó inválidas y aun así obtuvo 0/10 en validación. Esto demuestra que el cuello es la selección lógica, no solamente la sintaxis o el vocabulario.

La variante opcional `WORDLE_SLM_EXPLAIN=1` pidió razonamiento visible y un JSON final; con el adapter oficial alcanzó 1/5, igual que el baseline corto. No justifica cambiar el prompt congelado ni abrir el test.

Conclusión operacional: no continuar `pure-word-only-cont`, no promover el trie como Pure competitivo y no gastar más iteraciones SFT en la misma interfaz. El siguiente cambio debe optimizar directamente la acción/feedback (por ejemplo, un objetivo de preferencias con negativos duros o una política de recompensa), debe tener un smoke de gradiente y una evaluación de 61 objetivos, y debe seguir sin usar candidatos filtrados en Pure.

## 15. GRPO-lite de acciones: smoke y primera rama sin perturbaciones

Se implementó `src/wordle_slm/action_training.py` y el comando reproducible `wordle-slm train-action`. El objetivo es una política de recompensa relativa por grupo: cada estado contiene varias acciones legales, se calcula offline una recompensa densa por resolver o reducir el conjunto de candidatos y se pondera la log-probabilidad de la respuesta. El prompt del modelo solo contiene el historial, el turno y el contrato Pure; ni candidatos, solver ni métricas entran en la entrada. El adapter se inicializa siempre desde `adapters/selected` y se guarda en una carpeta experimental separada.

Controles ejecutados:

- Smoke de 5 iteraciones observado antes de la fase completa: 1.299 estados de train y 183 de valid, pico Metal 7,21 GB, sin NaN. El mismo nombre de ejecución se reutilizó después y su JSONL quedó reemplazado por las métricas de 100 iteraciones.
- Rama de 100 iteraciones: `artifacts/runs/preference/grpo-lite.json`, adapter `adapters/grpo-lite`, 1.632,9 segundos, checkpoints cada 25, loss de acción final −2,367, pico 7,21 GB.
- Validación Pure pública de 10 objetivos: `artifacts/benchmark/grpo-lite-100-valid10.json`, 1/10 victorias, media puntuada 6,5, 25 acciones inválidas. El historial cambió después de la apertura (`abajo`, `ciego`, etc.), pero la tasa y las inválidas fueron exactamente las del smoke y del baseline.

La loss negativa no es un error: es el signo del objetivo de maximización de log-probabilidad ponderada por ventajas. Tampoco es criterio de selección; la política no mejoró la partida. El adapter seleccionado no se tocó y su SHA continuó siendo `6c7a48c8...71c6`.

## 16. GRPO-lite con trayectorias perturbadas: hipótesis de recuperación rechazada

La primera implementación solo entrenaba estados de la trayectoria óptima del solver. Para cubrir el caso real en que el SLM se desvía, se añadió una variante aislada con `--perturbations 2`: por objetivo se genera una trayectoria de referencia y dos trayectorias que eligen acciones compatibles aleatorias, aplican su feedback real y vuelven a construir el estado siguiente. Solo la trayectoria de referencia usa `best_guess`; las perturbadas no hacen llamadas al solver caro. Train y valid siguen sin usar el split oculto.

La CLI acepta `--iterations`, `--evaluate-every`, `--perturbations` y `--run-name`. El smoke `grpo-lite-perturbed-smoke` terminó en 5 iteraciones con 6.115 estados de train, 885 de valid y pico 7,68 GB. La rama completa `grpo-lite-perturbed-100` terminó 100 iteraciones en 1.744,5 segundos, con loss final 0,7045 y pico 7,77 GB; no produjo NaN y se guardó en `adapters/grpo-lite-perturbed-100`.

La comprobación de partida comparable fue `artifacts/benchmark/grpo-lite-perturbed-100-valid10.json`: 1/10 victorias, media 6,5, 25 inválidas, 0 tool calls. Es exactamente igual al baseline y a `grpo-lite-100-valid10.json`. Por tanto, la cobertura de recuperación sí cambia la loss y los estados de entrenamiento, pero no cambia todavía la acción final suficiente para ganar. No se amplió a 61, no se abrió el test y ninguna de estas ramas se promovió.

### Aprendizaje operativo nuevo

1. El objetivo de política puede mover mucho la loss sin mover la tasa de victoria; la evaluación de partidas sigue siendo la compuerta de selección.
2. La variación de acciones observada después de `careo` no implica que el modelo haya aprendido a filtrar correctamente el feedback; hay que medir candidatos compatibles implícitos, repeticiones e inválidas, no solo diversidad textual.
3. Las trayectorias perturbadas son necesarias para estudiar recuperación, pero dos perturbaciones y 100 actualizaciones no bastan para transferir la política al decoding autoregresivo de MLX.
4. El coste de la implementación actual es alto (aproximadamente 27–29 minutos por 100 actualizaciones en este Mac) por longitudes dinámicas y compilaciones Metal. Antes de lanzar cientos de iteraciones adicionales hay que fijar formas/padding o agrupar estados; de lo contrario se consume tiempo sin una señal de evaluación intermedia.
5. El próximo cambio permitido debe atacar la brecha restante de entrenamiento–decoding (por ejemplo, ranking de acciones con negativos duros y batches de estados, o una política de decoding explícitamente congelada), con smoke, validación completa de 61 y sin tocar `adapters/selected`. No repetir GRPO-lite idéntico ni abrir el test oculto con estas ramas.

## 17. Revalidación del rival y candidato de historial óptimo

Antes de esta ronda se volvió a leer el documento completo y se verificó que no quedaban procesos MLX huérfanos. La configuración efectiva de OpenRouter sigue siendo `deepseek/deepseek-v4-pro-0813`; no había un segundo modelo configurado. La smoke remota de un objetivo volvió a recibir HTTP 403 por límite total de la clave. Se inspeccionó el error solo en memoria para confirmar el motivo y se eliminó el JSON temporal; no se guardó ningún identificador de clave. Por ello no se repitió el benchmark remoto y los cuatro JSON oficiales Pro permanecen congelados.

La rama `pure-optimal-history-smoke` se sometió a una evaluación Agent completa sobre los 61 objetivos públicos, usando el mismo servidor, parser y límites del harness. El resultado fue:

| Candidato | Track | Objetivos | Victorias | Tasa | Media con pérdidas=7 | Inválidas | Tool calls |
|---|---|---:|---:|---:|---:|---:|---:|
| `pure-optimal-history-smoke` | Pure | 61 | 1 | 1,64 % | 6,918 | 148 | — |
| `pure-optimal-history-smoke` | Agent | 61 | 54 | 88,52 % | 4,066 | 16 | 236 |

El Agent es funcional pero queda por debajo de `pure-clean-long` (56/61, 91,80 %) y no compensa el fracaso de Pure. El resultado confirma de nuevo que enseñar trayectorias óptimas o dar acceso a herramientas cambia mucho el protocolo Agent, pero no transfiere por sí solo una política Pure capaz de escoger la acción correcta desde un historial arbitrario. El adapter oficial `adapters/selected` no se modificó y conserva su SHA `6c7a48c8...71c6`.

### Decisión y regla para la siguiente sesión

No promover `pure-optimal-history-smoke`, no continuar sus checkpoints como si fueran una mejora y no abrir el test oculto. La próxima rama debe justificar cómo cierra la brecha entre la política que se entrena y el decoding real; repetir SFT, GRPO-lite o perturbaciones con la misma interfaz no tiene evidencia a su favor. Cualquier nuevo rival remoto debe tener un ID distinto registrado de forma segura y cuota suficiente antes de ejecutar cuatro benchmarks completos.

## 18. Cuota agotada y dos ramas adicionales rechazadas

La smoke remota posterior confirmó el mismo modelo efectivo, `deepseek/deepseek-v4-pro-0813`, pero la consulta segura del endpoint de cuota devolvió `limit=2`, `usage≈2,007` y restante `0`. La smoke Pure de un objetivo terminó en pérdida y 403 por límite; se eliminó el JSON temporal sin guardar el texto del error. Cambiar de ID no resolvería este límite de la clave: antes de reanudar el rival hay que disponer de cuota real y registrar el nuevo ID si el usuario ha configurado uno distinto.

Se auditaron dos ramas locales que podían dar una señal barata antes de entrenar otra vez:

| Rama | Configuración | Resultado público | Decisión |
|---|---|---:|---|
| `adapters/pure-optimal-constraints` | 100 iteraciones observadas, 30 capas, rank32, prompt con restricciones derivadas | Pure 0/10, media 7,0, 26 inválidas | Rechazada; confirmar restricciones en el prompt no hizo que el modelo las aplicase. |
| `adapters/pure-optimal-history-lr1e6-40` | 40 iteraciones, 16 capas, LR 1e-6, historial sin candidatos | Primeros 10: 1/10, media 6,5, 23 inválidas | Detenida antes de completar 61 por señal claramente peor; la pérdida 7,60→1,29 no es criterio de promoción. |

El checkpoint de 20 iteraciones de la segunda rama se conserva solo como experimento reproducible; no se mezcla con el benchmark oficial. No se iniciaron servidores ni entrenamientos adicionales después de estas pruebas y `adapters/selected` conserva exactamente `6c7a48c8...71c6`.

### Consecuencia para el siguiente intento

La evidencia ya descarta otra continuación corta de SFT sobre `pure-optimal-history` y otra exposición de restricciones sin cambiar el mecanismo de selección. El siguiente trabajo local debe usar un objetivo de acción con negativos duros o una mejora de decodificación que siga siendo explícitamente Pure; cualquier cambio que filtre candidatos externamente debe clasificarse como una política distinta y no puede presentarse como la comparación solicitada. La competición oculta y sus gráficas finales siguen pendientes de cuota y de un adapter que supere validación completa.

## 19. Ranking de acciones con negativos duros: no transferido todavía

Se implementó el comando aislado `wordle-slm train-ranker` en `src/wordle_slm/action_ranking.py`. Su pérdida compara directamente las probabilidades secuenciales de hasta ocho acciones legales etiquetadas offline por el solver; el prompt sigue siendo solo historial y feedback. La primera versión usaba longitudes dinámicas y se volvió impracticable: después de la primera iteración estuvo más de ocho minutos compilando nuevas formas de Metal sin llegar a la siguiente métrica. Se detuvo sin promover pesos.

La versión corregida normaliza la log-probabilidad por longitud, aplica clipping global a 1,0 y fija el tensor a ocho acciones × 256 tokens. Smoke de 5 iteraciones (`action-ranker-fixed-smoke`) terminó sin NaN, con pico 11,75 GB y loss 0,43→2,56; el benchmark Pure de 5 objetivos repitió exactamente la señal corta anterior: 1/5, media 6,0 e 11 inválidas. No se amplió a 100 iteraciones porque el coste (~63 s/iteración) y la memoria no justifican gastar horas sin una mejora de acción temprana.

La rama `action-ranker-hardneg-smoke2` con formas dinámicas también quedó en 1/5 y se conserva solo como diagnóstico. El adapter oficial no se tocó. Aprendizaje: hacer explícita la clasificación de acciones no basta si el score que se entrena no cambia el primer token que el servidor decodifica; la próxima optimización debe reducir el coste de evaluación y medir el ranking de la acción realmente generada, no solo la pérdida auxiliar.

## 20. Distilación óptima larga: la pérdida mejora, la acción no

Se lanzó `configs/pure-optimal-history-1000.yaml` desde `adapters/selected`, con 16 capas, rank16, LR `1e-6`, acumulación 8 y secuencia 384. La ejecución se detuvo de forma controlada en el checkpoint 100 para no gastar las 900 iteraciones restantes sin una señal de partida. La validación de tokens pasó de 1,125 a 1,106 y el pico fue 3,14 GB, sin NaN.

El checkpoint 100 reprodujo el patrón Pure anterior: en los primeros 12 objetivos públicos resolvió solo `acida` y perdió los otros 11. Se canceló la validación incompleta y no se reanudó el entrenamiento. La disminución de val loss no se tradujo en selección de acciones, por lo que esta rama no se promociona ni se abre sobre test oculto.

La conclusión acumulada es ahora más fuerte: SFT óptimo corto/largo, restricciones legibles, word-only, DPO y dos objetivos de política distintos reducen pérdidas auxiliares pero no alteran la decisión Pure. El siguiente intento no debe consumir más iteraciones de la misma familia; necesita una representación o mecanismo de inferencia nuevo, y debe conservar el contrato Pure o declararse como track diferente.

## 21. Scratchpad estructurado: aprende el formato, no la jugada

Se generó `data/pure-scratchpad` desde las trayectorias óptimas públicas. El asistente debía devolver `analysis` con patrón/letras derivadas del historial y un `guess`, sin candidatos ni herramientas. Se añadió el flag experimental `WORDLE_SLM_SCRATCHPAD=1`; el parser sigue extrayendo únicamente `guess` y el modo oficial permanece desactivado.

El smoke de 100 iteraciones (`adapters/pure-scratchpad-300`) bajó val loss de 5,83 a 1,89, sin NaN y con 3,28 GB de pico. En 5 partidas Pure obtuvo 1/5, media 6,0 e 14 inválidas, peor en inválidas que el baseline corto. La representación mejora la obediencia al formato pero no la selección de la palabra; no se continúa a 300 ni se abre el test oculto.

Con esto queda documentado que el cuello no es únicamente JSON, explicación ni longitud de salida. El siguiente cambio debe modificar la señal de decisión o el contexto de entrenamiento de forma que la palabra compatible sea la primera acción generada, sin introducir un solver o una lista de candidatos en Pure.

## 22. Auditoría de la nueva configuración remota

En la continuación más reciente se volvió a inspeccionar `.env` y el valor efectivo sigue siendo `deepseek/deepseek-v4-pro-0813`; no hay un segundo ID escrito en el proyecto. La consulta segura de `/api/v1/key` devolvió límite total `2`, uso `2,007114157` y restante `0`, por lo que no se lanzó otra llamada de benchmark.

Sí existen artefactos antiguos con `deepseek/deepseek-v4-flash-0731`, pero son validaciones públicas de 61 objetivos y contienen 11 errores 429 en Pure y 55 en Agent. Aunque declaran `complete=true`, la tasa se calcula sobre partidas con errores remotos y no es una comparación oficial ni sustituye los 124 objetivos ocultos. Se conservan como evidencia histórica, no se mezclan con `summary.md` y no se presentan como el “modelo nuevo”.

Regla para la próxima ejecución: verificar primero que `OPENROUTER_MODEL` contiene el ID nuevo y que el endpoint de cuota tiene saldo positivo; después hacer una smoke de un objetivo, borrar cualquier error sensible y solo entonces congelar el rival y lanzar Pure/Agent completos sobre los mismos 124 objetivos.

## 23. Preflight de la continuación del objetivo (17 de agosto, 14:xx)

Se volvió a leer este documento completo hasta EOF antes de ejecutar cualquier comprobación. La inspección de estado no encontró ningún proceso `mlx_lm.server`, entrenamiento LoRA, benchmark TypeScript ni `wordle-slm serve` activo. El adapter oficial sigue teniendo SHA256 `6c7a48c8fa56b1f08fa4f902e7c044e520aec672c71952ff489db85113cb71c6`.

La configuración efectiva sigue siendo exactamente:

```text
OPENROUTER_MODEL=deepseek/deepseek-v4-pro-0813
```

No hay otra definición de `OPENROUTER_MODEL` en el entorno del proceso ni en los archivos `.env` encontrados a tres niveles de profundidad. La consulta de cuota, reducida solo a campos no sensibles, devuelve `limit=2`, `usage=2.007114157` y `remaining=0`. Por ello no se hizo una llamada de generación remota, no se abrió de nuevo el test oculto y no se modificaron los cuatro JSON oficiales Pro.

El informe canónico continúa declarando `success=false`: Pure tiene 2/124 victorias frente a 0/124, pero su intervalo bootstrap pareado es `[0, 0.0403226]` y no es estrictamente positivo; Agent sí cumple con 115/124 frente a 6/124. La afirmación competitiva completa sigue sin estar demostrada. Los artefactos con `deepseek/deepseek-v4-flash-0731` siguen siendo validaciones públicas antiguas con errores 429 y no son evidencia del modelo nuevo.

**Acción requerida antes de cualquier benchmark nuevo:** escribir realmente el nuevo ID en `.env` y usar una clave con saldo positivo. Después se debe ejecutar una smoke aislada, registrar modelo/proveedor/fallback/semilla sin secretos, congelar la configuración y solo entonces lanzar los cuatro benchmarks emparejados. Mientras eso no ocurra, reentrenar o comparar contra un rival supuesto no produciría evidencia válida del objetivo pedido.

## 24. Cambio de regla: rival Flash y routing reproducible

El usuario cambió explícitamente la regla operativa: no volver a `deepseek/deepseek-v4-pro-0813` hasta superar primero `deepseek/deepseek-v4-flash-0731`, y aumentó el límite de OpenRouter en 2 USD. La cuota auditada después del cambio fue `limit=4`, `usage=2.007114157` y `remaining=1.992885843`.

Se fijó `.env` a:

```text
OPENROUTER_MODEL=deepseek/deepseek-v4-flash-0731
OPENROUTER_PROVIDER_ONLY=deepinfra
```

La primera ejecución Pure Flash sin provider pin (`deepseek-flash-pure`) llegó a 44/124 y fue detenida. El archivo incremental se conserva como diagnóstico no válido: tenía 41 errores HTTP 429, repartidos entre DigitalOcean y StreamLake por rate limit compartido del upstream. Esas partidas no se cuentan como derrotas ni como evidencia de la competición.

Para evitar mezclar endpoints, `agent/src/player.ts` ahora admite `OPENROUTER_PROVIDER_ONLY` y envía `provider.only` junto con `allow_fallbacks=false`. El build TypeScript y las dos pruebas Vitest pasan. La smoke limpia `deepseek-flash-deepinfra-smoke` resolvió el modelo Flash en una partida sin errores remotos; perdió el objetivo `adios`, con 2 acciones inválidas, 29,7 s de latencia y coste 0,00055622 USD. La derrota de esta única smoke no es una conclusión competitiva.

**Protocolo vigente:** completar Pure Flash y Agent Flash con exactamente los 124 objetivos, conservar `complete=true`, exigir cero errores de proveedor para aceptar los archivos y generar un informe/gráficos separados del benchmark Pro. No cambiar a Pro aunque Flash resulte difícil hasta que el SLM lo supere bajo el criterio congelado.

## 25. Rate limits durante Flash Agent y reintentos transitorios

La corrida Pure homogénea con `OPENROUTER_PROVIDER_ONLY=deepinfra` terminó los 124 objetivos, pero tres partidas recibieron timeout de 300 s. Los tres objetivos (`tibio`, `tribu`, `vapor`) se repitieron individualmente sin error y se ensamblaron en `deepseek-flash-pure-deepinfra-clean.json`; el artefacto tiene `complete=true`, 124 juegos y `errors=0`.

La primera corrida Agent con DeepInfra recibió `engine_overloaded`/429 tras 16 juegos. Se probó una smoke Agent con Novita, que funcionó, pero la corrida homogénea Novita volvió a recibir 429 upstream compartido después de aproximadamente 12 juegos. Las respuestas 429 no se cuentan como derrotas y las corridas parciales se conservan solo como diagnósticos.

Para no cambiar el modelo, el prompt, el track ni las herramientas, `agent/src/benchmark.ts` ahora reintenta hasta tres veces únicamente errores transitorios identificables (`429`, `temporarily rate-limited`, `engine_overloaded` o timeout de proveedor), con esperas de 5/10/20 s. Un error no transitorio sigue terminando la partida y queda registrado; el resultado final solo se acepta con cero errores. `npm run build` y `npm test` pasan después del cambio. El upstream efectivo continúa siendo Flash, con fallbacks de proveedor desactivados.

## 26. Resultado completo frente a Flash: Agent sí, Pure todavía no

El benchmark homogéneo final usa `deepseek/deepseek-v4-flash-0731`, Novita como upstream fijado, fallbacks desactivados y los 124 objetivos ocultos en orden canónico. Los reintentos transitorios recuperaron los 429 sin dejar errores; `deepseek-flash-agent.json` declara `complete=true`, `errors=0`, 124 juegos, 46 victorias, 292 acciones inválidas, 227 tool calls y coste aproximado de 0,1162 USD.

Pure quedó ensamblado en `deepseek-flash-pure.json` desde la corrida DeepInfra y tres reintentos aislados: `complete=true`, `errors=0`, 124 juegos, 0 victorias, 302 acciones inválidas y coste aproximado de 0,0793 USD. El adapter congelado conserva 2/124 en Pure y 115/124 en Agent.

La comparación pareada Flash se generó separada del informe Pro mediante `npm run report -- --rivalPrefix flash` y `uv run wordle-slm visualize-benchmark --rival-prefix flash`. Resultados decisivos:

| Track | SLM | Flash | Diferencia observada | IC bootstrap pareado 95% | Decisión |
|---|---:|---:|---:|---:|---|
| Pure | 2/124 | 0/124 | +1,61 pp | [0,00 pp, +4,03 pp] | No demostrada: el límite inferior no es estrictamente positivo. |
| Agent | 115/124 | 46/124 | +55,65 pp | [+45,97 pp, +65,32 pp] | SLM gana. |

Por la regla del usuario y el criterio estadístico congelado, **no se cambia a Pro**. La cifra Pure observada favorece al SLM, pero no constituye todavía una victoria estadística. Se mantienen separados los informes y gráficos Flash (`summary-flash.*`, `competition-dashboard-flash.*`, `competition-progress-flash.*`, CSV y `technical-report-flash.artifact.json`) y no se altera el benchmark Pro canónico, que sigue declarando `success=false` por Pure.

Se añadió soporte reproducible para `--rivalPrefix` en `agent/src/report.ts` y `--rival-prefix` en `wordle-slm visualize-benchmark`, además de reintentos transitorios en el harness. Tras regenerar ambas superficies, `npm run build`, `npm test`, `uv run pytest -q` y la compilación Python pasan. No se inició otro entrenamiento ni se modificó `adapters/selected`; para superar Flash hace falta una mejora real de Pure, no simplemente más reintentos o una loss auxiliar menor.

## 27. DAgger on-policy ampliado: mejora marginal y techo de decodificación

Se amplió la recogida on-policy Pure con el adapter oficial congelado. La primera tanda cubría los primeros 100 objetivos del split `train`; la segunda cubrió los objetivos 101–200, sin tocar `valid` ni `test`. Ambas corridas locales terminaron `complete=true`, `errors=0`; la segunda produjo 0 victorias, media puntuada 7,000 e 258 acciones inválidas. El builder `scripts/build_on_policy_dagger.py` ahora acepta varios artefactos train, rechaza objetivos duplicados y solapamiento train/valid, y guarda la procedencia como lista.

El manifiesto `data/pure-onpolicy-dagger-v2/manifest.json` registra 200 objetivos train, 61 valid, 447 estados train, 77 valid, `hidden_test_used=false`, `train_validation_prompt_overlap=0`, `train_sha256=5fbfcad3c341bbf20a6c3a1f943bd3d753fd31a1c302337facdc7d911e38f115` y `valid_sha256=2047eb988ed92aa2a07aecbaa85d57be5edad36a4561fa957f425648af535b1d`.

La rama `pure-onpolicy-dagger-v2` mantuvo el contrato Pure original (historial y feedback, JSON corto, sin candidatos, word bank ni solver en el prompt), partió de `adapters/selected`, usó 16 capas, rank16, scale32, dropout0,05, secuencia384, acumulación8 y LR5e-6. El smoke de 20 iteraciones terminó sin NaN, con pico2,99GB y val loss 1,075→1,171. Su validación completa pública fue 1/61, media6,918 e 152 inválidas frente al baseline seleccionado 0/61, 7,000 y 173: mejora pequeña en las tres métricas.

La corrida larga alcanzó el checkpoint100 (val loss0,972; pico3,04GB) y se evaluó completa sobre los mismos 61 objetivos. El resultado fue exactamente 1/61, media6,918 e 152 inválidas, con la misma apertura `careo` en los 61 casos y la misma única victoria (`acida` en dos turnos) que el smoke. Una continuación aislada de 50 iteraciones desde ese checkpoint, LR2e-6, bajó su val loss a1,063 pero volvió a producir exactamente 1/61, 6,918 y 152 inválidas. No se abrió el test oculto para estas ramas porque no hubo una mejora adicional de política; ninguno de sus adapters puede promoverse.

### Aprendizaje operativo

1. Añadir 100 objetivos on-policy más y continuar 50 actualizaciones cambia la loss, pero no cambia las acciones autoregresivas del servidor; la brecha entrenamiento→decoding sigue siendo el cuello de botella.
2. El resultado de 1/61 es una señal pública reproducible mejor que el baseline, pero no basta para justificar otra continuación ciega ni para afirmar victoria frente a Flash.
3. `adapters/selected` conserva el SHA oficial `6c7a48c8fa56b1f08fa4f902e7c044e520aec672c71952ff489db85113cb71c6`; Flash sigue siendo el único rival permitido y el test continúa cerrado para nuevas ramas.
4. El siguiente experimento debe concentrar explícitamente la señal en la acción de palabra o cambiar el mecanismo de selección; repetir DAgger JSON con más iteraciones queda descartado hasta observar una partida pública distinta.

## 28. DAgger word-only: loss mucho menor, cero victorias

Para probar si el envoltorio JSON estaba absorbiendo la señal de acción, se transformó exactamente `data/pure-onpolicy-dagger-v2` a `data/pure-onpolicy-dagger-v2-word-only`: mismos 447/77 estados, mismos hashes de origen y mismo límite `hidden_test_used=false`, pero la respuesta supervisada pasó a ser solo la palabra de cinco letras. El smoke (`adapters/pure-onpolicy-dagger-v2-word-only-smoke`, 20 iteraciones, LR5e-6, 16 capas, secuencia384) no produjo NaN, tuvo pico2,99GB y bajó la val loss de 7,020 a 3,508.

La validación completa pública, usando el contrato word-only tanto en prompt como en parser, terminó 0/61, media puntuada7,000, 144 inválidas y cero errores. El JSON/DAgger v2 había dado 1/61, 6,918 y 152 inválidas con los mismos objetivos. Por tanto la loss más baja y las menos inválidas no implican una mejor política: la variante se rechaza, no se amplía a 300 iteraciones y no se abre el test oculto.

La regla reforzada es no volver a cambiar únicamente el formato de respuesta. El próximo cambio debe modificar el mecanismo de selección o introducir una señal de acción ponderada de forma comprobable, con validación pública completa antes de cualquier benchmark Flash.

## 29. Feedback por colores: representación más legible, política peor

Se probó una representación Pure que reemplaza cada código numérico del historial por etiquetas por letra (`c=gris, a=amarilla, ...`) y añade una frase breve de significado. El script `scripts/build_grid_dagger.py` transforma solo los prompts de `data/pure-onpolicy-dagger-v2`; las etiquetas se calculan del feedback ya observado, sin candidatos, conteos, word bank ni solver. El flag de inferencia es `WORDLE_SLM_GRID=1`; el contrato de respuesta y el parser JSON no cambian.

El smoke `adapters/pure-onpolicy-dagger-v2-grid-smoke` terminó sin NaN, con val loss1,044→1,100 y pico3,27GB. En los 61 objetivos públicos produjo 0 victorias, media7,000, 166 inválidas y cero errores. Es peor que DAgger JSON v2 (1/61, 6,918, 152 inválidas), por lo que la rama se rechaza y no se prolonga.

La representación explícita de colores no corrige la brecha de selección: el modelo puede leer un estado más descriptivo y aun así emite acciones genéricas o repetidas. Cualquier próximo intento debe alterar el objetivo/decodificador de acción, no solo el texto del historial.

## 30. DPO on-policy: margen preferencial sin transferencia a partidas

Se construyó `data/preferences-onpolicy` con `scripts/build_onpolicy_preferences.py`: 447 pares train y 77 valid derivados de los rollouts Pure públicos. Cada par usa el mismo historial en ambos completions, la acción del solver como `chosen` y la acción real emitida por el adapter congelado como `rejected` (o una alternativa válida peor si coincidían). El manifiesto conserva `hidden_test_used=false`, sin solapamiento de prompts y hashes `train_sha256=3007b4eade8ace4e7180d1933d0939694ab21d4b93c7ef0d7e92e76528916cbc`, `valid_sha256=a520c0523472e968c1dec75b978755ae6cd9bee02b396953dda882b39cb9e06b`.

Se parametrizó `preference_training.py` para aislar directorio de preferencias, adapter de referencia, salida y run. La fase DPO de 50 iteraciones se detuvo deliberadamente tras el checkpoint25, con label smoothing0, LR1e-6, val loss0,587, reward margin0,247, reward accuracy0,792 y pico Metal4,10GB. El checkpoint `adapters/pure-onpolicy-dpo/adapters.safetensors` se evaluó con el contrato Pure JSON original sobre los 61 objetivos públicos: 1/61, media6,918, 152 inválidas y cero errores, idéntico al DAgger v2.

Conclusión: el objetivo preferencial sí mueve el margen y baja la loss, pero no mueve el argmax autoregresivo del servidor. No se continúa DPO ni se promueve el adapter. El próximo paso válido debe cambiar explícitamente el mecanismo de decodificación/selección y documentar cualquier nueva política antes de comparar contra Flash.

## 31. Test oculto de DAgger v2: sin mejora sobre el adapter seleccionado

Tras la validación pública completa (1/61 frente a 0/61 del baseline), se congeló `adapters/pure-onpolicy-dagger-v2` y se abrió el test oculto una sola vez. La corrida Pure terminó `complete=true`, 124 juegos, `errors=0`, 2 victorias, media puntuada6,919, 321 acciones inválidas, cero tool calls y proveedor `mlx-local`. El adapter seleccionado oficial también tiene 2/124 en Pure; por tanto v2 no mejora la primera métrica ni cambia la conclusión frente a Flash (0/124).

No se generó una afirmación de victoria ni se modificó el test. El hidden benchmark queda documentado como experimento no conseguido y el adapter oficial conserva SHA `6c7a48c8fa56b1f08fa4f902e7c044e520aec672c71952ff489db85113cb71c6`. Para avanzar hace falta una modificación de selección que cambie partidas públicas, no otra continuación de DAgger/DPO con la misma decodificación.

## 32. LoRA wide: más capacidad e inestabilidad, sin señal pública

Se probó una rama aislada `pure-onpolicy-dagger-v2-wide-smoke` con los mismos 447/77 pares DAgger JSON, pero LoRA rank32, scale64, últimas30 capas, LR1e-5, secuencia384 y 20 iteraciones. La val loss pasó de4,079 a2,349 y el pico fue3,36GB, pero la train loss osciló hasta8,533, señal de inestabilidad para esta interfaz.

La pantalla pública de los primeros 10 objetivos terminó 1/10, media6,5, 25 inválidas y cero errores, exactamente la señal corta del baseline; no se ejecutó la validación completa ni el test oculto. La rama se rechaza y no se amplía. Aumentar capacidad/rango sin cambiar la selección autoregresiva tampoco ha cerrado la brecha.

## 33. Margen a nivel de tokens: compilación Metal no completada

Se añadió `src/wordle_slm/action_margin.py` y el comando reproducible `wordle-slm train-action-margin`. La hipótesis era ponderar ocho veces los primeros tokens divergentes de la palabra elegida frente a la acción real rechazada, evitando que la loss se concentre en las llaves del JSON. El dataset público era `data/preferences-onpolicy` (447/77 pares), con rank16 y sin cambios de prompt ni herramientas.

El smoke de una iteración llegó a cargar el adapter y preparar los pares, pero quedó más de cuatro minutos en la compilación del primer `value_and_grad` de MLX Metal sin escribir métrica ni checkpoint. Se detuvo de forma controlada; no hay adapter parcial utilizable, no se modificó `adapters/selected` y no se abrió ningún split adicional. La implementación queda experimental hasta fijar formas/batches o reducir el grafo; repetirla sin ese arreglo sería consumir tiempo sin señal.

## 34. Margen de tokens reducido: compila, pero no cambia la política

Para resolver el bloqueo de la sección anterior se hizo un cambio aislado y reversible: `train-action-margin` acepta `--num-layers` y `--max-length`; el smoke usa únicamente las últimas 4 capas LoRA y secuencias fijas de 192 tokens. El adapter oficial no se sobrescribió, y los duplicados de la prueba anterior se detuvieron antes de iniciar esta sesión. La hipótesis seguía siendo la misma: ponderar ocho veces los primeros tokens divergentes entre la acción del solver y la acción rechazada.

El smoke `adapters/pure-onpolicy-action-margin-4l-smoke` terminó 5 iteraciones sin NaN, con `train_loss=1.0783`, `valid_loss=1.1619`, `valid_margin=-0.5841`, pico de 3,24 GB y 211,5 segundos incluyendo la validación de 77 pares. La reducción de capas resolvió la compilación de Metal (frente a los procesos que no producían checkpoint), pero no produjo una mejora accionable.

Se levantó el servidor local con ese adapter y se evaluaron los primeros 10 objetivos públicos bajo el contrato Pure oficial: `artifacts/benchmark/pure-onpolicy-action-margin-4l-smoke-valid10.json` declara `complete=true`, 1/10 victorias, media puntuada 6,5, 25 acciones inválidas, cero errores y cero tool calls. La señal es exactamente la del baseline corto; por las reglas de selección no se amplía a 61 ni se abre el test oculto. No se cambia `adapters/selected`, cuyo SHA sigue siendo `6c7a48c8fa56b1f08fa4f902e7c044e520aec672c71952ff489db85113cb71c6`.

Aprendizaje: hacer la loss más local y reducir el grafo sí arregla el coste de compilación, pero una actualización de cinco pasos no mueve el argmax autoregresivo. No se debe confundir un smoke que compila con una mejora de partida; cualquier continuación necesitaría primero una señal distinta en validación pública.

## 35. Interpolación en test y continuación acotada prevista

La rama `adapters/interp-selected-cleanlong-0p75`, que había mejorado la validación pública a 2/61, se evaluó una única vez sobre el test oculto con el mismo servidor y contrato Pure. `artifacts/benchmark/interp75-pure-test.json` terminó `complete=true`, 124 juegos, `errors=0`, 2 victorias (`bahia`, `cerco`), media puntuada 6,919, 321 acciones inválidas y cero tool calls. Es exactamente la pareja de victorias del adapter seleccionado (2/124, 325 inválidas); no se promueve ni se presenta como mejora estadística frente a Flash (0/124).

Como última prueba acotada de la hipótesis de margen, se ejecutó una corrida de 100 iteraciones desde `adapters/selected`, salida `adapters/pure-onpolicy-action-margin-4l-100`, últimas 4 capas LoRA, secuencia fija 192, LR 5e-6, seed 20260814 y validación de los 77 pares únicamente en la iteración 100. Terminó sin NaN en 680,6 s de reloj total, pico 3,24 GB, `train_loss=7,6e-7`, `valid_loss=0,1354` y `valid_margin=5,2652`: la preferencia se memorizó con mucha eficacia.

La pantalla Pure de los primeros 10 objetivos (`artifacts/benchmark/pure-onpolicy-action-margin-4l-100-valid10.json`) terminó `complete=true`, pero quedó exactamente en 1/10, media puntuada 6,5, 25 inválidas y cero tool calls; la única victoria fue `acida`, igual que el smoke y el baseline corto. La reducción de la loss y el margen positivo no cambiaron el argmax que usa el servidor. Se detiene la rama, no se amplía a 61 ni se abre otro test oculto, y `adapters/selected` conserva su SHA oficial.

Conclusión: el margen a nivel de tokens, incluso tras 100 actualizaciones y validación de pares claramente favorable, no cierra la brecha entrenamiento→decoding. Repetir esta misma familia sería optimización ciega; el siguiente avance necesita cambiar la representación o el mecanismo de inferencia y debe respetar la separación Pure/Agent.

## 36. Apertura canónica única: elimina la ambigüedad, no la brecha

Se generó `data/pure-fixed-opening-canonical` con `scripts/build_fixed_opening_policy.py`. Para cada objetivo de train/valid (sin test oculto) la primera jugada se fijó a `careo` y las siguientes se obtuvieron con el solver determinista a partir del feedback real. El prompt de cada registro conserva únicamente historial, feedback y contrato JSON; no contiene candidatos, word bank, herramientas ni métricas. El manifiesto registra 1.299 estados train, 183 valid, `train_sha256=7563c4408b6f4e162692f060579b584f2ad5244635d7d4f9c98cb36b46c1f414` y `valid_sha256=62e6a0cf06830bd520f4d679d11388c966e01d1138a4a492d171e131faf9aaf7`.

La rama `adapters/pure-fixed-opening-canonical-100`, desde `adapters/selected`, usó 16 capas, rank16/scale32, secuencia384, LR1e-6 y 100 iteraciones. Terminó sin NaN, con 2,99 GB de pico; la validación de tokens fue 0,537 al inicio, 0,595 en checkpoint50 y 0,723 en checkpoint100. La comprobación Pure pública de 10 objetivos (`artifacts/benchmark/pure-fixed-opening-canonical-100-valid10.json`) quedó en 1/10, media puntuada 6,5, 25 inválidas y cero tool calls, exactamente igual que las ramas anteriores; la única victoria fue `acida`.

Aprendizaje: hacer consistente la etiqueta del historial vacío no cambia la acción autoregresiva del servidor. Se rechaza la rama antes de la validación de 61 objetivos y no se abre el test. El cuello sigue siendo la selección lógica desde feedback, no solo la ambigüedad de la apertura.

## 37. Reintento Pro actual: cuota insuficiente y artefacto no aceptado

Tras la instrucción explícita de usar `deepseek/deepseek-v4-pro-0813` con mesura, se verificó el ID efectivo y se ejecutaron dos smokes limpias. Ambas confirmaron el modelo Pro; la smoke Pure con Novita completó un objetivo sin error y costó aproximadamente 0,0021 USD, mientras la smoke Agent resolvió un objetivo sin error y costó aproximadamente 0,0039 USD.

Se lanzó una única corrida Pure completa sobre los 124 objetivos con el mismo split, seed, temperatura, fallback desactivado y límite remoto de 512 tokens. El archivo `artifacts/benchmark/deepseek-pro-current-pure.json` cubrió los 124 objetivos y confirmó `deepseek/deepseek-v4-pro-0813`, pero registró 56 errores de proveedor (45 respuestas de crédito insuficiente, 10 respuestas 502 de Novita y 1 error interno), 0 victorias, 188 acciones inválidas y coste contabilizado de aproximadamente 0,1455 USD. Por contrato, `complete=true` no basta: el artefacto queda rechazado porque `errors != 0`.

Se probó una sola smoke con el endpoint primario de DeepSeek para evitar los 502 de Novita; fue rechazada antes de generar tokens por el mismo límite de crédito. El upstream se restauró a `novita` y no se lanzó el Agent completo ni se hicieron reintentos masivos. La consulta segura de cuota, reducida a campos no sensibles, indicó límite 4 USD, uso 2,903437626 USD y restante 1,096562374 USD. No se guardaron claves, URLs de credenciales ni textos de error sensibles en la documentación.

Los archivos canónicos Pro antiguos (`deepseek-pure.json` y `deepseek-agent.json`) también tienen errores (27 y 54 respectivamente) y no deben presentarse como comparación limpia. La conclusión se mantiene: el adapter SLM conserva 2/124 Pure y 115/124 Agent, pero todavía no existe una pareja Pro con `errors=0` que permita afirmar victoria estadística. No gastar más cuota Pro hasta disponer de saldo suficiente para mantener el contrato de 512 tokens y un proveedor estable; si se retoma, reutilizar solo los juegos fallidos con shards y verificar `errors=0` antes de generar el informe.

## 38. Apertura `audio` desde base: formato aprendido, política no

Para comprobar si el adapter seleccionado estaba atrapado en la apertura `careo`, se generó `data/pure-fixed-opening-audio` con una sola apertura canónica `audio`, 1.358 estados train y 199 valid, sin candidatos, herramientas, solver ni objetivo en los prompts (`hidden_test_used=false`). La rama se inicializó desde el modelo base, no desde `adapters/selected`.

El smoke de 20 iteraciones terminó sin NaN, con val loss 3,433→1,300 y pico 2,99 GB, pero la pantalla de 10 objetivos públicos produjo 0/10 y 20 inválidas, con historiales vacíos. La continuación hasta checkpoint100 redujo la val loss a 0,575 y sí produjo palabras, pero la evaluación pública siguió en 0/10 y empeoró a 28 inválidas. La rama queda rechazada: aprender el formato y fijar una apertura no transfiere la selección lógica.

## 39. Currículo Pure deduplicado desde trayectorias óptimas: sin transferencia

El dataset existente `data/pure-oracle` se deduplicó por prompt a `data/pure-oracle-dedup`: 3.919 estados train, 771 valid y cero conflictos de etiqueta. Cada prompt conserva solo historial y feedback; la acción es la jugada determinista del solver calculada offline y el test oculto no se usa.

El entrenamiento desde base (`adapters/pure-oracle-dedup-base-200`) alcanzó checkpoint100 con val loss 0,915 y 3,12 GB. En 10 objetivos públicos obtuvo 0/10, media puntuada 7 y 30 inválidas. Una continuación aislada a LR2e-6 (`adapters/pure-oracle-dedup-base-500`) no mejoró el primer checkpoint (val loss 0,919→0,948) y se detuvo antes de consumir el resto. No se promueve ningún peso.

Aprendizaje: incluso con estados sin conflictos y etiquetas solver-deterministas, la loss de tokens no cambia el argmax autoregresivo de la primera acción. Repetir SFT de esta forma queda descartado; el próximo cambio debe alterar el mecanismo de selección o la representación de la acción, manteniendo explícita la separación Pure/Agent.

## 40. LR alto sobre el currículo deduplicado: sin señal de transferencia

Como comprobación final de si el cuello era únicamente convergencia lenta, se inició desde base la misma `data/pure-oracle-dedup` con LR `1e-5`, 500 iteraciones previstas, 16 capas, secuencia384 y checkpoints cada100 (`adapters/pure-oracle-dedup-base-lr1e5-500`). El checkpoint100 terminó sin NaN, con val loss0,900 y pico3,07 GB, prácticamente igual a la rama LR5e-6 (0,915). Se detuvo sin abrir otra pantalla pública: la métrica auxiliar no mostró una mejora que justificase consumir las 400 iteraciones restantes.

Las ramas audio, deduplicada y LR alto quedan rechazadas y nunca se copian a `adapters/selected`; su SHA oficial permanece `6c7a48c8fa56b1f08fa4f902e7c044e520aec672c71952ff489db85113cb71c6`. La siguiente acción competitiva requiere cuota remota suficiente y un cambio de selección que no sea otra continuación SFT del mismo contrato.

## 41. Auditoría de continuación: no hay ID remoto nuevo todavía

La continuación del objetivo volvió a inspeccionar todos los `.env` del proyecto y solo encontró `OPENROUTER_MODEL=deepseek/deepseek-v4-pro-0813`, `OPENROUTER_PROVIDER_ONLY=novita` y una clave presente. La cuota segura sigue en límite4 USD, uso2,903437626 USD y restante1,096562374 USD; no hay otro ID ni saldo nuevo que permita cerrar Pure y Agent con `errors=0`.

Las smokes Pro limpias anteriores siguen siendo válidas para identificar el rival, pero la corrida Pure de 124 objetivos continúa rechazada por 56 errores. No se lanzó Agent ni se gastó más cuota en esta continuación. Se regeneraron las gráficas válidas del benchmark Flash (`competition-dashboard-flash.*`, `competition-progress-flash.*`, CSV y `chart-map-flash.json`), y `npm run report -- --rivalPrefix flash` conserva `success=false` porque Pure no tiene un intervalo bootstrap estrictamente positivo.

El trabajo local de esta continuación dejó dos datasets y tres configuraciones de entrenamiento reproducibles, todas fuera de `adapters/selected`; ningún resultado supera la validación pública ni justifica abrir un nuevo test oculto. Para continuar la competición real hace falta que el ID nuevo se escriba en `.env` y/o que OpenRouter disponga de crédito suficiente; entonces se debe repetir smoke, Pure y Agent emparejados y generar las gráficas del rival nuevo sin reutilizar datos Pro con errores.

## 42. Benchmark Pro corto y selección de proveedor por latencia

Antes de tocar el benchmark se volvió a leer este registro completo, se comprobó que no había procesos MLX/benchmark activos y se verificó que `adapters/selected` conservaba el SHA oficial `6c7a48c8fa56b1f08fa4f902e7c044e520aec672c71952ff489db85113cb71c6`.

La consulta segura de cuota indicó límite4 USD, uso2,903437626 USD y restante1,096562374 USD. El endpoint de OpenRouter expuso latencias de ventana móvil de 30 minutos para `deepseek/deepseek-v4-pro-0813`: Alibaba p50 1.213 s, Fireworks 1.346 s, DigitalOcean 1.367 s y Novita 1.807 s. Smokes Pure de un objetivo con el nuevo límite de 96 tokens midieron aproximadamente 8,18 s, 8,92 s y 10,41 s para Alibaba, Fireworks y DigitalOcean, todos sin error en esa primera petición. Alibaba queda como proveedor preferido por latencia observada, no como resultado competitivo.

El fallo anterior se aisló al presupuesto de salida reservado por petición: con el contrato remoto de 512 tokens OpenRouter respondió HTTP 402 indicando que la clave solo podía afrontar 111 tokens. `agent/src/player.ts` ahora acepta `OPENROUTER_MAX_TOKENS`, limitado entre16 y512, manteniendo512 por defecto para el benchmark oficial. El `.env` local se dejó en `OPENROUTER_PROVIDER_ONLY=alibaba` y `OPENROUTER_MAX_TOKENS=96` para que el siguiente intento corto no vuelva a reservar512 tokens accidentalmente; el `.env.example` conserva512 como valor de referencia oficial.

Se lanzó el diagnóstico Pure de 12 objetivos `artifacts/benchmark/deepseek-pro-short-alibaba-pure.json`. Tras la primera smoke exitosa, las 12 peticiones recibieron HTTP402 antes de inferencia; el archivo queda `complete=false` y `errors=12`, por lo que se rechaza y no se empareja con el SLM. Las smokes Agent de Alibaba fallaron también con HTTP402 aun reduciendo el límite a32 tokens. No se hicieron más reintentos para proteger la cuota y no se lanzó un Agent parcial. Los errores fueron saneados para no guardar claves ni URLs sensibles.

El snapshot reproducible de proveedores está en `artifacts/benchmark/pro-provider-latency.json` y la explicación en `artifacts/benchmark/pro-short-report.md`. La lección es doble: acortar `--limit` no basta si OpenRouter reserva el máximo de salida; hay que acortar simultáneamente el número de objetivos y `max_tokens`, y validar cada smoke justo antes de abrir una corrida. Hasta que una smoke Pure y otra Agent vuelvan a terminar con `errors=0`, no existe benchmark Pro aceptado ni base para afirmar nada frente al SLM.

## 43. Continuación con el rival anunciado: el ID no cambió y la cuota sigue bloqueando

La auditoría de esta continuación no encontró ningún `.env` adicional ni un nuevo `OPENROUTER_MODEL`: el valor efectivo continúa siendo `deepseek/deepseek-v4-pro-0813`, con `OPENROUTER_MAX_TOKENS=96`. La cuota segura sigue en límite4 USD, uso2,909161894 USD y restante1,090838106 USD. El adapter `adapters/selected` conserva el SHA `6c7a48c8fa56b1f08fa4f902e7c044e520aec672c71952ff489db85113cb71c6` y no hay procesos de entrenamiento, servidor ni benchmark activos.

La telemetría actual de endpoints cambió: DigitalOcean p50 1,088 s y uptime30m 98,77 %, Alibaba p50 1,260 s y uptime30m 99,90 %, Fireworks p50 1,406 s y Parasail p50 1,384 s. DigitalOcean es ahora el proveedor configurado porque además declara soporte de `tools`, necesario para Agent. Una smoke Pure de un objetivo con DigitalOcean y límite32 recibió HTTP402 antes de inferencia; el nuevo comportamiento del harness la guardó como `complete=false`, evitando que se interprete como resultado.

Conclusión operativa: antes de reentrenar o abrir el test no se puede afirmar que haya un rival nuevo; hay que escribir un ID distinto en `.env` si esa era la intención del usuario y disponer de crédito suficiente para una smoke Pure y otra Agent. La búsqueda de proveedor avanzó hasta un candidato medido, pero ningún proveedor ha producido todavía una pareja Pro limpia en esta cuota.

## 44. Ranking de acciones compacto: la pérdida mejora, la política Pure no

Para no repetir la rama anterior de formas dinámicas, se parametrizó `wordle-slm train-ranker` con `--num-layers` y `--max-length`. La rama aislada `adapters/action-ranker-4l-192-100` usó las últimas4 capas LoRA y secuencias fijas de192 tokens, con 6.115 estados train y885 valid, sin modificar `adapters/selected`. El pico Metal fue5,04GB y no hubo NaN.

La pérdida bajó de1,976 (iteración1) a1,212 (checkpoint75), pero diver presentó3,175 en la iteración100. Se seleccionó únicamente el checkpoint75 (`adapters/action-ranker-4l-192-75`, SHA `7c2428fed787ee7427b008f0a14d12af6c7c55618c24ba6fb1d7d4b76bc6d569`) para evaluar, nunca el último por defecto.

La evaluación Pure pública de10 objetivos terminó `complete=true`, pero quedó en0/10 victorias, media puntuada7,000 y32 acciones inválidas. Es peor en inválidas que el adapter seleccionado y no justifica ampliar a61, copiar pesos ni abrir el test oculto. El ranking compacto queda rechazado: una mejora de loss y una memoria estable no cambiaron el argmax autoregresivo.

## 45. Vuelta explícita a Flash: configuración correcta, cuota insuficiente

Se volvió a leer este registro completo antes de cambiar la configuración. La auditoría inicial mostró que `.env` todavía apuntaba a `deepseek/deepseek-v4-pro-0813`, `digitalocean` y `OPENROUTER_MAX_TOKENS=96`; no había procesos MLX, benchmark ni servidor activos, y `adapters/selected` conservaba el SHA oficial `6c7a48c8fa56b1f08fa4f902e7c044e520aec672c71952ff489db85113cb71c6`.

El `.env` se dejó ahora en `deepseek/deepseek-v4-flash-0731`, proveedor fijado `digitalocean` y `OPENROUTER_MAX_TOKENS=512`, que es el contrato oficial. La consulta segura de cuota devolvió límite4 USD, uso2,909161894 USD y restante1,090838106 USD. El endpoint de Flash publica soporte de herramientas en todos los proveedores; DigitalOcean tenía la menor latencia p50 observada (673 ms), seguido de DeepInfra (920 ms), ambos con uptime de 30 minutos superior al 98 %.

Se ejecutaron dos smokes Pure de un objetivo con Flash y fallaron antes de inferencia con HTTP402 por crédito insuficiente: una con DigitalOcean y 512 tokens (`deepseek-flash-current-smoke-pure.json`) y otra con 96 tokens (`deepseek-flash-current-smoke-pure-96.json`). Una tercera prueba con DeepInfra y 96 tokens (`deepseek-flash-current-smoke-pure-deepinfra-96.json`) recibió el mismo 402. Los artefactos se conservaron como `complete=false`, se redujeron a clases de error y se saneó cualquier URL o credencial; no se lanzó Agent ni un benchmark completo para no consumir más cuota.

La pareja Flash oficial limpia ya existente (`deepseek-flash-pure.json` y `deepseek-flash-agent.json`, 124 objetivos, `errors=0`) sigue siendo la referencia reproducible. Se regeneraron `summary-flash.*`, los dashboards/progresos PNG/SVG y los CSV/mapas. El informe continúa correctamente en `success=false`: Pure SLM 2/124 frente a Flash 0/124, IC pareado `[0,00 %, 4,03 %]`; Agent SLM 115/124 frente a Flash 46/124, IC `[45,97 %, 65,32 %]`. Por tanto, volver al ID Flash está hecho, pero no se puede iniciar una nueva competición ni declarar victoria hasta que OpenRouter permita una smoke Pure y otra Agent sin 402.

## 46. Preflight de Tencent HY3 y GPT-5.6 Luna: crédito real todavía bloqueante

El usuario solicitó repetir el test con `tencent/hy3` y `openai/gpt-5.6-luna`. Antes de tocar el rival se volvió a leer este registro completo. La configuración por defecto se mantuvo en Flash, el adapter oficial conserva el SHA `6c7a48c8fa56b1f08fa4f902e7c044e520aec672c71952ff489db85113cb71c6` y no había procesos MLX, servidor ni benchmark activos.

La consulta segura de `/api/v1/key` ahora devuelve límite5 USD, uso2,909161894 USD, `limit_remaining=2,090838106` y `is_free_tier=false`. Esto confirma que el límite de gasto aumentó un dólar, pero no demuestra que exista saldo de créditos consumibles: el error de OpenRouter identifica `limit_source=openrouter_credits`.

La metadata actual de `tencent/hy3` publica seis endpoints. Se eligió DeepInfra para la smoke por p50 de 770 ms, uptime30m≈97 % y soporte de herramientas. La smoke Pure de un objetivo con 96 tokens (`deepseek-tencent-hy3-smoke-pure.json`) falló antes de inferencia con HTTP402 `Insufficient credits`; el coste registrado es cero. No se lanzó Agent ni el benchmark de 124 objetivos.

La metadata de `openai/gpt-5.6-luna` publica siete endpoints y herramientas. La ruta más rápida listada es Amazon Bedrock (p50 913 ms), pero OpenRouter rechazó esa preferencia con HTTP404 porque para la revisión servida solo estaban permitidos OpenAI y Azure. Se reintentó una smoke Pure con OpenAI (`deepseek-gpt-5-6-luna-smoke-pure-openai.json`, 96 tokens) y recibió el mismo HTTP402 de crédito insuficiente; también tiene coste cero. El archivo de Bedrock (`deepseek-gpt-5-6-luna-smoke-pure.json`) conserva únicamente el diagnóstico 404, saneado.

Los tres artefactos de smoke se marcaron `complete=false`, se eliminaron URLs/credenciales y no se mezclan con los benchmarks oficiales. Por tanto, todavía no existe comparación válida Pure/Agent para ninguno de los dos modelos nuevos. Para continuar basta con añadir crédito consumible en OpenRouter (aumentar el límite no es suficiente); entonces se repetirán las smokes Pure y Agent y, solo con `errors=0`, los 124 objetivos oficiales por track bajo el mismo seed, temperatura, 512 tokens y fallback desactivado.

## 47. Auditoría de continuación: no hay rival nuevo escrito y la cabeza de ranking no muestra señal

En la continuación del objetivo se volvió a auditar el worktree y el entorno. Solo existen `.env` y `.env.example`; ambos contienen `OPENROUTER_MODEL=deepseek/deepseek-v4-flash-0731`, `OPENROUTER_PROVIDER_ONLY=digitalocean` y `OPENROUTER_MAX_TOKENS=512`. No aparece ningún ID adicional de modelo en los archivos ni en el entorno de proceso, y no hay servidor, entrenamiento o benchmark activo. El límite de la clave sigue en5 USD con `limit_remaining=2,090838106`, pero no se intentó otra llamada remota sin un ID nuevo confirmado.

La referencia Flash limpia permanece congelada: Pure SLM 2/124 frente a Flash 0/124 con IC `[0,00 %, 4,03 %]`, Agent SLM 115/124 frente a Flash 46/124 con IC `[45,97 %, 65,32 %]`. Por contrato, eso no es todavía una victoria Pure estadística.

Como diagnóstico local se cargó `adapters/selected` y se extrajeron representaciones del último token del prompt para 64 estados de `data/pure-oracle-dedup/train` y 30 de `valid`. Un clasificador de vecino más cercano sobre esas activaciones acertó 1/30 (3,3 %) y devolvió aperturas genéricas (`careo`, `abono`, `abajo`, `guaba`) en la mayoría de estados. La extracción por lotes tarda aproximadamente 30 s por 16 estados; ampliar este head sin una señal pública sería caro y no está justificado. No se modificó `adapters/selected`, cuyo SHA continúa `6c7a48c8fa56b1f08fa4f902e7c044e520aec672c71952ff489db85113cb71c6`.

Conclusión operacional: antes de lanzar el benchmark solicitado hay que escribir el ID del rival potente en `.env` y disponer de crédito OpenRouter utilizable. Una vez hecho, se ejecutarán smokes Pure/Agent y después los 124 objetivos de ambos tracks; hasta entonces no se presentará Flash repetido ni una prueba incompleta como resultado nuevo.

## 48. SFT con peso en la palabra: compila, pero no cambia Pure

Para avanzar localmente sin asumir un rival remoto inexistente se añadió el comando aislado `wordle-slm train-word-first`. Parte de `adapters/selected`, conserva el prompt Pure de historial únicamente y multiplica por8 la pérdida de los tokens de la palabra dentro de `{"guess": "..."}`; no introduce candidatos, solver, word bank ni herramientas. La rama se guardó en `adapters/pure-word-first-probe` y nunca toca `adapters/selected`.

El smoke de cinco iteraciones terminó sin NaN, con pico Metal3,91GB, pero la pérdida subió de1,033 en la iteración1 a1,811 en la5. Su evaluación Pure pública de10 objetivos (`artifacts/benchmark/pure-word-first-probe-valid10.json`) obtuvo 1/10 victorias, media6,5, 25 inválidas y cero errores, exactamente la señal corta del baseline. La rama se rechaza y no se amplía: una ponderación local de la palabra tampoco mueve el argmax autoregresivo.

La implementación queda disponible como experimento reproducible, con build/Ruff/tests verdes, pero no es candidata ni evidencia de victoria frente a Flash. El siguiente entrenamiento solo se justificará después de que el rival nuevo esté realmente escrito y validado en `.env`, para no optimizar contra una comparación indeterminada.
