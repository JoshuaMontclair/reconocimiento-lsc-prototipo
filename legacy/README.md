# legacy/ — archivo de la exploración preliminar (marzo de 2026)

Trabajo de grado 281-2026-1 · Joshua Emmanuel Méndez Ospina · Director: Felipe Palta.
Documentos hermanos: `ESTADO_PROYECTO_2026-09-13.md` (diagnóstico) y `DECISIONES_TESIS.txt` (decisiones vigentes).

## Qué es esta carpeta

`legacy/src/` contiene los scripts de la exploración preliminar que el anteproyecto declara en §1.3.4 y que
**no forman parte del plan de ejecución**. Se archivan con `git mv` (la historia de cada archivo se conserva) en
el commit inmediatamente posterior a la etiqueta `exploracion-marzo-2026`, que congela el estado tal cual quedó (salvo README.md y requirements.txt, reescritos en septiembre de 2026 y commiteados aparte)
en marzo. Nada se borra: son la evidencia de lo que se probó y de por qué no es reportable.

Estos scripts **no están pensados para ejecutarse desde aquí**. Calculan la raíz del proyecto como
`Path(__file__).resolve().parent.parent` (que ahora apunta a `legacy/`) e importan `src.train_lstm`,
`src.train_v3` y `src.train_confusion_fix` con esa raíz en `sys.path`, así que fallarían al importar.
Si algún día hiciera falta reproducir uno, se copia temporalmente a `src/` o se ajusta `BASE_DIR`.
Los modelos que produjeron siguen en `models/*.keras` (ignorados por git, en disco), con los JSON de resultados
trackeados en `models/`.

## Por qué se archivan

Tres razones, todas documentadas en `ESTADO_PROYECTO_2026-09-13.md` (hallazgo B y §5):

1. **Selección de modelos mirando el conjunto de prueba.** 13 de los 18 scripts de entrenamiento/evaluación
   eligieron arquitecturas, semillas, subconjuntos del ensemble o modelos individuales según su acierto sobre
   las 100 muestras de test (por ejemplo `train_targeted.py` borra los modelos que no aciertan dos muestras
   concretas del test; `eval_smart_ensemble.py` prueba todas las combinaciones de modelos hasta dar 100 %).
   Con eso el test deja de ser test y **ninguna cifra de marzo es reportable**.
2. **Semilla de partición cambiada para limpiar el test.** `train_final_100.py` pasa de la semilla 42 a la 0
   porque «gives clean test set»: saca del test la muestra `0011_0004_0000`, que ningún modelo acertaba. Por eso
   las semillas 42 y 0 quedan contaminadas y `data/dataset/` (sobrescrito con semilla 0) no sirve como partición.
3. **Scripts superados.** La arquitectura que sobrevivió (Conv1D×2 + BiLSTM + atención, entrada 30×450 con
   velocidades, ensemble de 5 semillas) vive en `src/train_final_100.py`; las versiones intermedias
   (`train_v2`, `train_v3`, `train_v4`) ya no aportan nada que no esté ahí. `extract_landmarks.py` usa una API
   de MediaPipe que la versión fijada del proyecto no incluye.

El anteproyecto exige seleccionar el modelo solo con validación, nunca con test, y evaluar además con LOSO de
5 pliegues (§2.2.4). Todo lo que hay aquí se rehace bajo esas reglas; no se corrige.

## Qué se queda en `src/` y por qué

| Script | Razón |
|---|---|
| `preprocess_lsc50.py` | Preprocesamiento correcto de los CSV publicados (normalización por hombros, remuestreo); verificado 1000/1000. |
| `prepare_dataset.py` | Construcción de particiones; se reescribirá para persistir índices explícitos (actividad 1.e). |
| `reextract_lsc50.py` | Re-extracción con MediaPipe Tasks API (Pose + Hand; falta Face). Es la base de D-01. |
| `train_final_100.py` | Contiene la arquitectura canónica (`build_model`, `add_velocity`, `refined_augmentation`). Su lógica de semilla 0 y sus comprobaciones sobre muestras del test se ignoran y se retirarán. |
| `train_lstm.py` | Superado como entrenador (LSTM simple de 225 rasgos, ~86 %), pero **es módulo de apoyo**: `train_final_100.py` importa de él `augment_mirror/noise/scale/temporal` y `N_TOTAL_FEATURES`, y `finetune_lstm.py` importa `apply_augmentation` y `evaluate_model`. Se queda por esa dependencia. |
| `finetune_lstm.py` | Ajuste fino sobre landmarks re-extraídos con Tasks API; único puente entre el corpus re-extraído y el modelo. |
| `grammar_mapper.py` | Léxico de las 50 señas verificado contra la Tabla 2 del artículo de LSC50; base del módulo de gramática (mes 3). |
| `translator.py`, `realtime.py`, `tts.py` | Cadena del prototipo en vivo (cámara → Tasks API → modelo → gramática → voz). |

Cambio pendiente (no aplicado): en `src/train_final_100.py` líneas 38-42 sobran los nombres `COORDS`,
`N_BODY_FEATURES`, `N_HAND_FEATURES`, `evaluate_model` y `plot_training_history` (importados y no usados);
la importación de `train_lstm` en sí **sí** se usa y debe mantenerse.

## Scripts archivados (uno por línea)

| Script | Qué hacía | Por qué se archiva |
|---|---|---|
| `train_v2.py` | Primer salto sobre `train_lstm`: velocidades (225→450), CNN+BiLSTM+atención, 15× aumento, 300 épocas, label smoothing. Produjo `best_model_v2.keras`. | Superado: su arquitectura sobrevive en `train_final_100.py`; evaluado sobre la partición aleatoria. |
| `train_v3.py` | Ensemble de 5 semillas + TTA (`ensemble_0..4.keras`). Define `add_velocity`, `resample_sequence`, `predict_with_tta` que los demás scripts muertos importan. | Superado por `train_final_100.py`; solo lo importan scripts archivados. |
| `train_v4.py` | «Ultimate pipeline» hacia el 100 %: posición+velocidad+aceleración+ángulos, 15 miembros (CNN, BiLSTM, Transformer), multi-escala 30/45/60, TTA de 30+ aumentos (`ult_cnn_bilstm_*.keras`). | Caza del 100 % sobre el test; corrida parcial. |
| `train_ultimate.py` | Entrena con train+val (900 muestras) y «valida» sobre aumentaciones del propio train; sobremuestreo extremo de las clases 11, 26, 40, 44 (`fix_multiscale_vel_s1`, `ult2_mscnn_s1`). | Fuga de datos en la validación; las clases se eligieron mirando `test[43]` y `test[52]`; corrida incompleta. |
| `train_targeted.py` | Entrena muchos modelos y **conserva solo los que aciertan al menos una de las 2 muestras del test que fallaban** (líneas 321-344 borran el resto): `targeted_deep_res_*`, `targeted_wide_cnn_*`, `targeted_gru_*`. | Selección directa por el conjunto de prueba. |
| `train_targeted_final.py` | Sobremuestreo y clasificadores binarios para `test[43]` (26 vs 44) y `test[52]` (11 vs 40), «for 100 %». | Selección por el conjunto de prueba. |
| `train_confusion_fix.py` | Sobremuestreo 10× y focal loss para los pares 11↔40 y 26↔44 (`targeted_conffix_s1..s6`, 11 h de cómputo). | Los pares se detectaron en el test; selección por test. |
| `train_sample52_fix.py` | Sobremuestreo 50-100× de la clase 11, pesos de clase y MixUp 11/40 para dar la vuelta a **una** muestra del test (`0011_0004_0000`); `targeted_s52_*`. | Ajuste a una muestra concreta del test (que además puede ser un error de etiqueta del corpus: va a validación lingüística, 1.d). |
| `train_binary_11v40.py` | Clasificador binario 11 vs 40 que sobreescribe la predicción del ensemble cuando dice 40. | Parche diseñado sobre el error observado en el test. |
| `train_extra_models.py` | Miembros 5-7 del ensemble (BiLSTM solo, CNN temporal, CNN+BiLSTM regularizado) para «diversidad». | Se juzgó por su efecto sobre el test; superado. |
| `eval_ensemble.py` | Evalúa el ensemble existente con TTA agresivo sobre el test. | Evaluación sobre partición aleatoria contaminada. |
| `eval_full.py` | Carga `ensemble_*` + `targeted_*` como un solo ensemble con TTA «targeting 100 %». | Ídem; combina modelos ya seleccionados por test. |
| `eval_smart_ensemble.py` | Prueba **todas las combinaciones** de subconjuntos de modelos y se queda con la que da 100 % en el test. | Selección por test en estado puro. |
| `eval_mega_ensemble.py` | Carga todos los modelos, TTA ultra, e identifica qué muestras del test fallan y con qué modelos (`mega_ensemble_results.json`). | Instrumento de la caza del 100 %; lee el test para decidir. |
| `final_100_eval.py` | Ensemble ponderado de todos los `.keras` + kNN como desempate para las muestras del test que sigan mal (`final_100_results.json`). | Ídem. |
| `extract_landmarks.py` | Extracción con `mp.solutions.holistic` (API antigua) de videos e imágenes; nunca se ejecutó sobre LSC50. | mediapipe 0.10.32 (versión fijada) no incluye `solutions.holistic` ni `HolisticLandmarker`: **no puede correr**. Sustituido por `reextract_lsc50.py` con Tasks API (D-01). |

## Cronología de la exploración (según fechas de modificación de los archivos)

Todo se produjo en un sprint de ~53 h. Commit inicial en git: 21-mar 17:09 (único commit hasta septiembre).

| Cuándo | Qué |
|---|---|
| 21-mar 12:22–12:48 | `extract_landmarks`, `prepare_dataset`, `train_lstm`: pipeline base (LSTM simple, 225 rasgos). |
| 21-mar 16:53–19:28 | `grammar_mapper`, `realtime`, `tts`, `translator`, `reextract_lsc50`: prototipo en vivo y re-extracción con Tasks API. |
| 21-mar 21:48–22:30 | `preprocess_lsc50`, `finetune_lstm`; `best_model_original.keras` (~86 %) y primeras curvas. |
| 21-mar 23:04–23:45 | `train_v2`: velocidades + CNN+BiLSTM+atención → `best_model_v2.keras`. |
| 22-mar 00:09–02:08 | `train_v3`: ensemble de 5 semillas + TTA → `ensemble_0..4.keras`, `ensemble_results.json`. |
| 22-mar 02:15–07:06 | `train_extra_models`, `eval_ensemble`, `train_targeted`: miembros extra y modelos «dirigidos» a las 2 muestras erradas. |
| 22-mar 08:42–20:48 | `train_confusion_fix` → `targeted_conffix_s1..s6` (una tarde entera de cómputo). |
| 22-mar 21:19–23:53 | `eval_full`, `eval_smart_ensemble`, `train_sample52_fix`, `train_binary_11v40`: búsqueda del subconjunto y parches para `test[52]`. |
| 23-mar 01:12–09:42 | `targeted_s52_*.keras` (seis modelos más para una muestra; ocho `targeted_s52_*` en total). |
| 23-mar 14:01–16:14 | `train_v4`, `eval_mega_ensemble`, `train_targeted_final`, `final_100_eval`, `train_ultimate`: última ronda de «100 %». |
| 23-mar 16:15–17:44 | `train_final_100` con **semilla 0** → `final_0..4.keras`, `best_model.keras`, `training_results.json`. Fin del sprint. |

Nada volvió a tocarse hasta el diagnóstico del 13 de septiembre de 2026.
