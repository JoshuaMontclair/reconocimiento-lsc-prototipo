# Modelo TCN + imputación + 252 para el prototipo (opción experimental)

Fecha: 3 de octubre de 2026.

> **Advertencia.** Es una opción **EXPERIMENTAL** del prototipo, pendiente de la decisión del director (N-63,
> reunión del lunes 5 de octubre). **El modelo por defecto sigue siendo el canónico**,
> `models/canonico_holistic_mascara_manos/modelo_canonico.keras`, con «mascara» y umbral 0,2. Nada de lo que
> carga `src/realtime.py` sin argumentos cambió. El modelo nuevo solo se usa si se pide con `--model-path`.

## 1. Cómo se entrenó

Script nuevo: `src/train_final_tcn.py`. Usa la configuración que recomienda N-63
(`docs/EXPERIMENTO_FACTORIAL_TCN.md`, `docs/EXPERIMENTO_CONFIRMACION_TCN.md`, brazo T1) y el **mismo
procedimiento** con que `src/train_final_100.py` entrenó el canónico.

| | Canónico vigente | Modelo nuevo |
|---|---|---|
| Datos | `data/processed_holistic_mascara_rostro`, «manos» | igual |
| Partición | `data/partitions/aleatoria_80_10_10.json` sin los 3 duplicados de D-06 (797 / 100 / 100) | igual |
| Mano ausente | «mascara» (D-07) | «mascara» + **imputación** por mano sobre los cuadros originales, antes de remuestrear (`mano_ausente_variantes.regla_imputacion`) |
| Entrada | 30 × (126 + 126 velocidades) = 252 | igual |
| Red | CNN + BiLSTM + atención (714.291 parámetros) | **TCN** `construir_tcn` con `CONFIG_T` (156.402 parámetros) |
| Receta | Adam, lr 1e-3 con CosineDecay, lote 64, *label smoothing* 0,1 | igual (`ENTRENAMIENTO_COMUN` de la TCN) |
| Aumentación, épocas, paciencia | ×10, 200, 25 sobre `val_accuracy` | igual |
| Semillas | 2026-2030 | igual |
| Selección | mayor `val_accuracy`; en empate, menor `val_loss`; luego el orden de las semillas | igual |
| Prueba | una sola vez al final: modelo elegido y ensemble; solo se reporta | igual |

**Cómo se reutiliza el código**

- Cada semilla se entrena con `train_final_100.entrenar_semilla` tal cual. Mientras dura el entrenamiento,
  `build_model` se sustituye por `construir_tcn` (`arquitectura_tcn()`).
- Antes de entrenar, `comprobar_receta()` verifica en el código fuente que la receta fija de `entrenar_semilla`
  es la de la TCN.
- La partición, las métricas, la evaluación en prueba y el entorno vienen de `train_final_100`.

**Verificación de la entrada** (`verificar_entrada`, se ejecuta antes de entrenar y queda en `resultados.json`)

- La `X_seq` es, bit a bit, la «imputacion» de `experimento_confirmacion_tcn.preparar`. Cambia 735 de los 1000
  videos respecto de «mascara».
- El video de control «hola» (0030_0000_0003) coincide con `regla_imputacion`.
- La aumentación ×10 coincide, con la misma semilla, con la del experimento (`train_loso.aumentar`).
- `add_velocity` es el mismo objeto que usó el brazo T1.
- La red tiene 156.402 parámetros.

Antes de entrenar se comprobó que no hubiera otros entrenamientos activos. La corrida duró unos 7 minutos en
CPU (`models/tcn_imputacion/corrida_2026-10-03.log`).

## 2. Semillas y modelo elegido

Validación (100 muestras) del checkpoint de cada semilla:

| Semilla | val top-1 | val_loss | val top-3 | val F1 macro | Épocas (mejor) |
|---:|---:|---:|---:|---:|:-:|
| 2026 | 0,96 | 0,9620 | 0,99 | 0,957 | 44 (19) |
| 2027 | 0,96 | 0,9477 | 0,98 | 0,957 | 62 (37) |
| 2028 | 0,95 | 0,9649 | 0,99 | 0,947 | 61 (36) |
| 2029 | 0,97 | 0,9289 | 0,99 | 0,969 | 61 (36) |
| **2030** | **0,97** | **0,8927** | 0,98 | 0,969 | 92 (67) |

**Elegido: semilla 2030.** Empata con la 2029 en 0,97 de validación y la desempata su menor `val_loss`, según la
regla del canónico. Se copió a `models/tcn_imputacion/modelo.keras`
(sha256 `6970722dd06e2672c812c0b7f74baf135d03912cc1bb79440faab8220b2c1679`).

## 3. Validación y prueba del esquema aleatorio, frente al canónico

La validación es la única cifra que decidió. La prueba es de 100 muestras y solo se reporta.

| Sistema | Val top-1 | Prueba top-1 | Top-3 | F1 macro | Errores |
|---|---:|---:|---:|---:|---:|
| Canónico, modelo elegido (semilla 2026) | 0,96 | 0,88 | 0,98 | 0,871 | 12 |
| **TCN + imputación, modelo elegido (semilla 2030)** | **0,97** | **0,94** | 0,98 | **0,937** | **6** |
| Canónico, ensemble de 5 | — | 0,91 | 1,00 | 0,909 | 9 |
| TCN + imputación, ensemble de 5 | — | 0,96 | 0,99 | 0,958 | 4 |

Cautelas:

- El esquema aleatorio es optimista: las mismas personas aparecen en entrenamiento y en prueba.
- Con 100 muestras, una muestra vale 0,01.
- La comparación independiente del señante es la de los experimentos LOSO. Allí esta configuración (T1) obtuvo
  0,949 en validación y 0,677 en prueba, frente a 0,920 y 0,589 del canónico.

## 4. Umbral de confianza (D-09)

**Umbral elegido: 0,00** (`models/tcn_imputacion/modelo.umbral.json`, con el sha256 del modelo).

**Por qué un script nuevo.** `src/umbral_confianza.py` no se puede usar tal cual con este modelo:

- recalcula las predicciones con la regla «mascara», y este modelo espera «imputacion»;
- espera `metricas.json` en cada pliegue, y los pliegues de T1 guardan `resultado.json` y `probabilidades.npz`.

Por eso `src/umbral_tcn_imputacion.py` **importa** el criterio sin cambiarlo (`REJILLA`, `utilidad`,
`elegir_umbral`, `efecto`). Solo cambian la entrada con que se recalculan las predicciones y las rutas.

**Datos de selección.** La validación de los 5 modelos LOSO de esta misma configuración
(`models/confirmacion_tcn_2026-10-03/T1_imputacion_252/semilla_2026`, 500 muestras). Las predicciones se
recalcularon cargando cada modelo:

- reproducen el `val_top1` guardado en los 5 pliegues;
- coinciden con `probabilidades.npz` (diferencia máxima 1,8·10⁻⁷).

La validación del modelo final también se reproduce (0,97).

**Criterio.** U(t) = aciertos aceptados − c · errores aceptados, con c = 1, sobre la rejilla 0,00-0,99. En
empate se elige el menor t.

**Por qué sale 0,00.** U(t) vale 450 (475 aciertos − 25 errores) y es constante de 0,00 a 0,15. Desde ahí ningún
corte la mejora: en 0,20 vale 448, en 0,32 vuelve a 450 y en 0,50 baja a 436. La regla de empate elige 0,00. En
la práctica, **el prototipo no rechaza nada por confianza** con este modelo. El estado `rechazada_confianza`
no se dará, pero los rechazos por manos y por ventana incompleta siguen funcionando.

**Sensibilidad.** c = 0,5 da 0,00 y c = 2 da 0,32.

**Efecto descriptivo de otros cortes** (en la prueba LOSO de señantes nuevos, 997 videos):

| Corte | Rechazo | Exactitud entre aceptadas |
|---|---:|---:|
| 0,00 | 0 % | 0,688 |
| 0,2 (el del canónico) | 1,6 % | 0,697 |
| 0,5 | 21 % | 0,795 |

No se cambió el criterio: se informa como anomalía. Si se quiere otro corte en vivo, se puede pasar
`--umbral` y queda registrado en el JSONL. Informe completo en `models/tcn_imputacion/umbral_D09_2026-10-03.{json,png}`.

## 5. Cambios en el código

Solo hay cambios **aditivos** en `src/translator.py` y `src/realtime.py`. Todo lo demás está en archivos nuevos.

**`src/translator.py`**

- `MANO_AUSENTE_OPCIONES` suma `"imputacion"` (las tres reglas previas siguen igual y en el mismo orden).
- Se importa `imputar_fotogramas` de `src/mano_ausente_variantes.py`, la misma función del entrenamiento.
- `_preprocesar` con «imputacion» hace, en orden:
  1. normalizar;
  2. enmascarar la mano ausente, igual que «mascara»;
  3. tomar las columnas «manos»;
  4. **imputar por mano sobre los cuadros de la ventana**;
  5. remuestrear a 30 pasos;
  6. agregar `add_velocity`.

  Una mano que no aparece en ningún cuadro queda en 0. Las otras reglas no tocan esa rama.
- «imputacion» con una representación distinta de «manos» lanza `ValueError`, en vez de imputar sobre
  columnas equivocadas.
- Con la procedencia del modelo nuevo (`mano_ausente: "imputacion"`), el traductor elige la imputación solo.
- El JSONL registra `mano_ausente` y `mano_ausente_modelo` como antes.
- `rechazada_manos` y `min_frames` no cambian y se evalúan antes de preprocesar.

**`src/realtime.py`**

- `--mano-ausente` acepta `imputacion`.
- El docstring tiene una línea de uso nueva.

**Archivos nuevos**

- `src/train_final_tcn.py`
- `src/umbral_tcn_imputacion.py`
- `tests/test_tcn_imputacion.py` (15 pruebas)
- `tests/datos/preprocesar_reglas_previas_2026-10-03.json`: huellas sha256 de la entrada de 39 casos con las
  reglas previas, calculadas con el `translator.py` del commit 1ba6f93
- `models/tcn_imputacion/`

**Pruebas**

- **Equivalencia con el entrenamiento** (tolerancia 1e-6): secuencias sintéticas con huecos interiores, al
  inicio y al final, un cuadro sin postura y una mano ausente en toda la ventana. También el video real «hola»:
  la entrada del prototipo coincide con la fila de `X_seq` del entrenamiento.
- **Reglas previas**: «ceros», «descartar» y «mascara» en las tres representaciones, y el modelo sin
  procedencia, dan una entrada **idéntica bit a bit** a la de antes del cambio.
- **Prototipo por defecto**: sigue cargando el canónico con «mascara» y umbral 0,2.
- **Rechazos**: se comprueban con «imputacion».
- **Carga del modelo nuevo**: carga con su procedencia y su umbral, sin avisos.

Suite completa: **651 pruebas OK**, en los 22 módulos que corren con venv312.
`tests/test_arquitecturas_2.py` no se pudo correr porque importa `pytest`, que venv312 no tiene. Ese archivo es
de otro experimento y el fallo es anterior a este cambio.

**Prueba sin cámara.** Se usó «hola» 0030_0000_0003, que está en la **prueba** del esquema aleatorio, así que
el modelo no lo vio al entrenar.

| Entrada | Resultado | Confianza | Rasgos registrados |
|---|---|---:|---|
| Puntos re-extraídos, inyectados con `ExtractorSecuencia` | aceptada, «hola» | 0,959 | `mano_ausente` y `mano_ausente_modelo` = «imputacion» |
| Los 69 fotogramas del `.avi` con el HolisticLandmarker real | aceptada, «hola» | 0,957 | igual |

- El segundo lugar quedó muy lejos: «¿por qué?» con 0,005.
- Preprocesamiento: 0,6 ms. Inferencia: 16-19 ms.
- Los fotogramas no se escriben a disco.

## 6. Uso en vivo

```bash
python src/realtime.py --model-path models/tcn_imputacion/modelo.keras --ancho 1280 --alto 720
```

La representación «manos», la regla «imputacion» y el umbral (0,00) se leen solos de
`modelo.procedencia.json` y `modelo.umbral.json`. Para volver al canónico basta omitir `--model-path`.

**Recordatorio:** opción experimental, pendiente de la decisión del director (N-63). Si se adopta, faltan:

- actualizar README y DECISIONES con las cifras;
- recalibrar el umbral anidado (N-47);
- volver a medir la latencia de punta a punta.
