# Modelo TCN + imputación + 252 para el prototipo (modelo final, D-14)

Fecha: 3 de octubre de 2026. Actualizado el 22 de noviembre de 2026 (sección 7).

> **Estado actual (22-nov-2026).** El director aceptó este modelo como modelo final (D-14, 5-oct-2026) y,
> desde el 22-nov-2026, **es el que carga `src/realtime.py` por defecto**, con el conversor `inventario`
> (D-15), captura a 1280 × 720 y el aviso de seña cortada en los bordes de la ventana (N-50, opción B). Su
> umbral de operación es **0,68** (N-75, validación anidada por señante; `models/tcn_imputacion/modelo.umbral.json`).
> La red heredada (`models/canonico_holistic_mascara_manos/modelo_canonico.keras`, «mascara», umbral 0,2)
> sigue disponible con `--model-path`. Ver la sección 7.
>
> Las secciones 1 a 6 se escribieron el 3-oct-2026, cuando era una opción EXPERIMENTAL pendiente de la
> decisión del director (N-63) y el canónico era el defecto; se conservan como registro de ese momento.

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

> **Actualización.** Desde el 22-nov-2026 (N-75) `modelo.umbral.json` dice **0,68**, el umbral global de la
> validación anidada por señante (N-74, `src/umbral_anidado_tcn.py`). El 0,00 de esta sección queda dentro del
> mismo archivo, en la clave `umbral_d09_validacion_vistos`. `--umbral 0` vuelve a ese corte.

**Umbral elegido (3-oct-2026): 0,00** (`models/tcn_imputacion/modelo.umbral.json`, con el sha256 del modelo).

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

Desde el 22-nov-2026 basta con:

```bash
python src/realtime.py
```

Es equivalente a la orden explícita que se usaba antes, que sigue siendo válida:

```bash
python src/realtime.py --model-path models/tcn_imputacion/modelo.keras --ancho 1280 --alto 720 --conversor inventario
```

La representación «manos», la regla «imputacion» y el umbral (hoy 0,68) se leen solos de
`modelo.procedencia.json` y `modelo.umbral.json`. Para volver a la red heredada:

```bash
python src/realtime.py --model-path models/canonico_holistic_mascara_manos/modelo_canonico.keras --conversor heredado
```

**Recordatorio:** opción experimental, pendiente de la decisión del director (N-63). Si se adopta, faltan:

- actualizar README y DECISIONES con las cifras;
- recalibrar el umbral anidado (N-47);
- volver a medir la latencia de punta a punta.

## 7. Predeterminados del prototipo y aviso de bordes (22-nov-2026)

Decisión del autor del 22-nov-2026, a partir de D-14 (modelo final), D-15 (conversor) y N-50 (opción B).

### 7.1 Predeterminados de `src/realtime.py`

| Opción | Antes | Ahora | Cómo volver a lo anterior |
|---|---|---|---|
| `--model-path` | `models/canonico_holistic_mascara_manos/modelo_canonico.keras` (red heredada) | `models/tcn_imputacion/modelo.keras` (modelo final) | `--model-path models/canonico_holistic_mascara_manos/modelo_canonico.keras` |
| umbral | 0,2 (`modelo_canonico.umbral.json`) | 0,68 (`modelo.umbral.json`, N-75) | se sigue leyendo de `<modelo>.umbral.json`; `--umbral` lo sustituye |
| `--conversor` | `heredado` (GrammarMapper) | `inventario` (D-15) | `--conversor heredado` |
| `--ancho` × `--alto` | 640 × 480 | 1280 × 720 | `--ancho 640 --alto 480` |
| `--bordes` | (no existía) | `aviso` | `--bordes off` |

En el código: `src/translator.py` define `MODELO_FINAL`, `MODELO_HEREDADO`, `MODELO_POR_DEFECTO`
(= `MODELO_FINAL`) y `CONVERSOR_POR_DEFECTO` (= `"inventario"`). `LSCTranslator` sin conversor inyectado
(`grammar_mapper=None`) sigue construyendo GrammarMapper: es la interfaz programática de las pruebas y de
`src/latencia_fuera_de_linea.py`, que pasan el conversor de forma explícita. Ese script toma la red heredada de
`MODELO_HEREDADO`, así que sus configuraciones no cambian.

**Por qué 1280 × 720.** No es un cambio de gusto; tiene tres razones documentadas:

1. **Forma de la imagen (N-57).** MediaPipe da x e y como fracciones del ancho y del alto, y la normalización
   divide las dos entre la distancia entre hombros, medida casi toda en x. Por eso la y normalizada escala con
   ancho/alto. El corpus es 1920 × 1080 (16:9). Con 640 × 480 (4:3) la y queda multiplicada por 0,75. Con la red
   heredada, la simulación de N-57 bajó el LOSO de 0,580 a 0,540. 1280 × 720 tiene la forma del corpus.
2. **Es la resolución con que se ha usado el modelo final.** La orden de uso de N-65 y el protocolo de la
   sesión de latencia (`docs/LATENCIA_4c_MODELO_FINAL.md`, sección 5) usan `--ancho 1280 --alto 720`.
3. **Cabe en el presupuesto de tiempo.** A 1280 × 720 el extractor tarda unos 11 ms por fotograma (mediana;
   0 de 20.058 fotogramas por encima de 33,3 ms; `docs/LATENCIA_4c_MODELO_FINAL.md`, sección 3).

Además, la altura que mide el aviso de bordes (7.2) está en y normalizada: sus umbrales salen del corpus 16:9
y solo son comparables con una captura 16:9.

Si la cámara no entrega 16:9 (OpenCV toma la resolución más cercana que admita), el arranque lo dice por consola
(`[AVISO] la camara entrega ...x..., que no es 16:9`). La consola muestra siempre la resolución pedida y la
efectiva. Límite: no se ha medido en vivo la tasa del bucle con la cámara del equipo a 1280 × 720; la medición
de la sección 3 de `LATENCIA_4c_MODELO_FINAL.md` es fuera de línea.

### 7.2 Aviso de seña cortada en los bordes (N-50, opción B)

Implementa la recomendación de `models/senas_cortadas_2026-09-22/recomendacion.md` (opción B):

- **Protocolo de reposo.** La persona está en reposo cuando se pulsa ESPACIO y vuelve al reposo antes de que
  se cierre la ventana. La pantalla lo recuerda: «REPOSO: desde el reposo, ESPACIO abre la ventana» y, con la
  ventana abierta, «VENTANA n/N (termine en reposo; X aborta)».
- **X aborta** si algo falla (ya existía: `rechazada_incompleta`).
- **Aviso en el registro, sin rechazar.** En cada ventana cerrada se mide si la seña pudo empezar antes de la
  ventana o terminar después. El resultado se escribe en el JSONL, en la consola (`[aviso borde: inicio]`) y en
  pantalla («AVISO: la sena pudo empezar antes de la ventana»). **No cambia el estado del ensayo ni impide la
  inferencia.** Al cerrar, la consola imprime cuántas ventanas tuvieron aviso.

**La medida.** Es la de `src/analisis_senas_cortadas.py`, reutilizada, no reescrita. `src/bordes_ventana.py`
importa `_velocidad_pose`, `_altura` y `_nanmean` y aplica los mismos tramos de borde que `rasgos_clip`:

1. Se toma la postura de la ventana. Los fotogramas sin postura quedan en ceros, como en el corpus. Se
   normaliza por hombros con `normalize_sequence`.
2. **Velocidad.** Por cada par de fotogramas consecutivos, el desplazamiento medio de las muñecas 15 y 16 de
   la postura (‖Δxy‖), dividido entre el Δt real del par (marcas c_i). La unidad es anchos de hombro por
   segundo. El análisis medía por fotograma a 24 fps. El prototipo no corre a 24 fps, así que mide por segundo,
   que es la unidad del archivo de umbrales (`umbrales_por_segundo` del análisis).
3. **Altura h.** Por fotograma, min(y de la muñeca izquierda, y de la derecha). La y crece hacia abajo y
   0 es la línea de los hombros.
4. **Borde.** K = max(3, round(0,125 s × fps de la ventana)), con K ≤ N; es la regla de `umbrales_a_fps`. A
   30 fot./s, K = 4.
5. `v_pose_ini_por_s` es la media de los pares dentro de los primeros K fotogramas; `h_ini`, la media de esos
   K fotogramas. `v_pose_fin_por_s` y `h_fin` salen igual de los últimos K.
6. **Regla.** Es la del rasgo «v_pose ∨ h» de `aplicar_regla`:
   - `aviso_inicio` = `v_pose_ini_por_s` > 1,7614 **o** `h_ini` < 0,9740;
   - `aviso_fin` = `v_pose_fin_por_s` > 0,2295 **o** `h_fin` < 1,2610.

   Las desigualdades son estrictas. Un rasgo sin datos no dispara. Si faltan los dos rasgos de un borde, la
   bandera es `null` («no se pudo medir»).

**Los umbrales.** Son los de `models/senas_cortadas_2026-09-22/umbral_bordes_candidato.json`, que el análisis
calculó con el percentil 95 de los clips íntegros de los cinco señantes (K = 3 a 24 fps, es decir, un borde de
0,125 s). El archivo dice «CANDIDATO, no vigente» y el arranque lo repite. `--umbral-bordes RUTA` permite usar
otros.

**Adaptación al modelo final: no hizo falta.** La medida solo usa la **postura** (muñecas y hombros). El
prototipo la extrae y la normaliza siempre, cualquiera que sea el modelo.

- La red heredada y el modelo final usan la misma representación de entrada, «manos» (126 + 126 velocidades),
  del mismo processed: `data/processed_holistic_mascara_rostro`, de donde salieron los umbrales.
- Ninguna de las dos recibe la postura como entrada.
- La regla de mano ausente («mascara» o «imputacion») solo toca los bloques de las manos.

Por eso los umbrales valen igual con las dos redes. Las pruebas lo comprueban: con «mascara», con
«imputacion» y con una representación con postura, las medidas son idénticas.

Lo que sí depende de la red es el **daño** que hace un recorte al acierto. N-50 lo midió con la red heredada:
top-1 0,580 → 0,410 sin el 35 % inicial. No se ha repetido con el modelo final. Como el aviso no rechaza, ese
dato no cambia lo que hace el prototipo; cambiaría solo si más adelante se adoptara la opción A, que sí rechaza.

**En el JSONL.** Clave nueva `bordes`, siempre la última del registro de ensayo. Las claves anteriores no
cambian ni de nombre ni de orden.

```json
"bordes": {"modo": "aviso", "umbrales": "models/senas_cortadas_2026-09-22/umbral_bordes_candidato.json",
           "aviso_inicio": true, "aviso_fin": false, "K": 4, "fps_ventana": 30.0,
           "v_pose_ini_por_s": 2.853, "v_pose_fin_por_s": 0.0, "h_ini": 0.5, "h_fin": 2.0,
           "pares_validos_ini": 3, "pares_validos_fin": 3, "costo_ns": 283375}
```

(Ventana sintética de `tests/test_bordes_ventana.py` que empieza a media seña; valores redondeados.)

- Se calcula para todo cierre con N ≥ 2, **también en los rechazos**.
- Con cierre `abortada`, medidas y banderas quedan en `null`.
- Con `--bordes off` el registro lleva `{"modo": "off"}` y no se calcula nada.
- Los avisos no entran en `contadores`, que sigue teniendo los cuatro estados de siempre.

**Dónde se calcula y cuánto cuesta.** En `LSCTranslator._cerrar_ventana`, **después** de fijar el estado y de
sellar t5. Por eso no cambia ninguna marca t0..t5 ni ninguna etapa de §2.2.7, ni L_resp. Las pruebas lo
comprueban: con el mismo reloj falso, los registros con `aviso` y con `off` son idénticos salvo `bordes`.

El tiempo del cálculo queda en `costo_ns` de cada registro. Se midió el 22-nov-2026 en el MacBook Pro M5 Max,
con 3.000 ventanas sintéticas por tamaño, tras 200 de calentamiento:

| Ventana | Mediana | p90 | p99 | Preprocesamiento de la misma ventana (mediana) |
|---|---:|---:|---:|---:|
| N = 75 (2,5 s a 30 fot./s) | 0,19 ms | 0,21 ms | 1,2 ms | 0,39 ms |
| N = 150 (tope de 5 s) | 0,25 ms | 0,29 ms | 6,8 ms | 0,50 ms |

La medición se hizo con el equipo muy cargado: carga media de 46 a 49, con un entrenamiento corriendo en
paralelo. Eso infla las colas (p99 y máximo); la mediana coincide con la estimación de N-50 (0,19 ms). Para
comparar: el extractor tarda unos 11-12 ms **por fotograma** y la inferencia unos 16-19 ms. El cálculo ocurre
una vez por ventana, después de t5, y retrasa como mucho el fotograma siguiente del bucle en esa fracción de
milisegundo.

**Pruebas.** `tests/test_bordes_ventana.py` comprueba que:

- con marcas a 24 fps, la medida reproduce `rasgos_clip` × 24, en clips sintéticos y en cuatro clips reales
  del processed;
- la velocidad por segundo no depende de la tasa;
- un valor exactamente igual al umbral no avisa, y la regla coincide con `aplicar_regla`;
- una ventana reposo-seña-reposo no avisa; si empieza a media seña, `aviso_inicio`; si acaba a media seña,
  `aviso_fin`;
- el estado, la clase, la confianza y las marcas no cambian con el aviso;
- `off` no calcula nada;
- los rechazos llevan el aviso y la ventana abortada lo deja en `null`;
- funciona con las dos redes reales cargadas desde disco;
- los predeterminados de `realtime.py`, la pantalla y la consola son los descritos.

**Límites.**

- Los umbrales son candidatos, medidos en clips del corpus y no en ventanas del prototipo abiertas con tecla
  (`recomendacion.md`, §6). Con una persona no vista, la regla marcó como cortados entre el 2,5 % y el 75,5 %
  de los clips íntegros, según el pliegue.
- El aviso informa, no decide. Cómo se usa en un análisis tiene que fijarse antes de la sesión.
- La altura depende de la postura de reposo de cada persona: unas manos en reposo más arriba, por ejemplo
  sobre una mesa, dispararían el aviso del final.

