# Experimento: Transformer pequeño, ResNet 1D y BiGRU frente a la TCN (LOSO)

Fecha: 3 de octubre de 2026. Antecedente: `docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md`, donde ganó la TCN
pequeña con GAP (brazo T). Motivación: `docs/REVISION_LITERATURA_ARQUITECTURAS.md`.

**Pregunta.** De las tres candidatas de la revisión que quedaron sin probar (Transformer pequeño, ResNet 1D o
InceptionTime pequeña, y BiGRU), ¿alguna acierta más que la TCN con personas nuevas?

Este documento no cambia el modelo canónico, el umbral, `translator.py` ni el alcance.

---

## 1. Diseño

Todos los brazos usan la misma entrada canónica, las mismas particiones y la receta de entrenamiento de T.
Lo único que cambia es el modelo.

- **Entrada:** HolisticLandmarker → normalización por hombros → regla «mascara» (D-07) → «manos» (126 rasgos)
  → 30 pasos → aumentación ×10 de `train_loso` → `add_velocity` → 30 × 252. Datos:
  `data/processed_holistic_mascara_rostro`.
- **Evaluación:** LOSO de 5 pliegues (D-06; `data/partitions/loso_5_pliegues.json`), con validación interna de
  2 muestras por seña.
- **Semillas:** 2026, 2027 y 2028 en todos los brazos.
- **Receta (la de T):** Adam con CosineDecay (α = 10⁻⁶), *label smoothing* 0,1, lote 64, `EarlyStopping` con
  paciencia 25 sobre `val_accuracy` (restaurando los mejores pesos), *checkpoint* sobre `val_accuracy`,
  200 épocas como máximo. lr 1e-3, salvo B (5e-4, ver abajo).
- **Código del entrenamiento:** no se reimplementa. `src/experimento_arquitecturas_2.py` llama a
  `experimento_arquitecturas.entrenar_corrida`, la misma función que entrenó T, y le amplía en memoria la
  función `constructor` con los brazos X, I y B. `src/experimento_arquitecturas.py` no se modifica.

| Brazo | Modelo | Parámetros | lr |
|---|---|---:|---:|
| **T** TCN + GAP (control) | `arquitecturas_literatura.construir_tcn` (`CONFIG_T`). **No se reentrena**: se reutilizan las 15 corridas de `models/arquitecturas_literatura_2026-10-02/T_tcn` (ver §1.2) | 156.402 | 1e-3 |
| **X** Transformer pequeño | `arquitecturas_2.construir_transformer` (`CONFIG_X`) | 187.890 | 1e-3 |
| **I** ResNet 1D | `arquitecturas_2.construir_resnet1d` (`CONFIG_I`) | 315.954 | 1e-3 |
| **B** BiGRU | `arquitecturas_2.construir_bigru` (`CONFIG_B`) | 341.171 | 5e-4 |

Los cuatro quedan entre 150 y 400 mil parámetros.

### 1.1 Configuración de cada brazo (fijada antes de entrenar; una por familia, sin variantes)

Se permitían hasta 2 variantes por familia, elegidas por validación. **No se usa ninguna.** La validación
está cerca del techo (0,95 con 100 muestras por pliegue: una muestra vale 0,01), así que elegir la mejor de
dos variantes inflaría la validación del brazo en un orden parecido al margen del criterio. Con una sola
configuración por familia, la comparación con T queda limpia.

**X · Transformer pequeño (encoder, pre-norm).** Versión reducida de `build_transformer`
(`legacy/src/train_v4.py`), que tenía d = 256, 4 bloques, 8 cabezas, post-norm y densas de 256.

- Proyección lineal `Dense(252 → 64)`.
- Codificación posicional **aprendida**: una matriz entrenable de 30 × 64 que se suma a la entrada (capa propia
  `PosicionAprendida`, registrada para serializar; hay que importar `src/arquitecturas_2.py` antes de cargar
  el modelo). Luego *dropout* 0,2.
- 2 bloques pre-norm: [LayerNorm → `MultiHeadAttention` (4 cabezas, `key_dim` 64, *dropout* 0,2) → *dropout*
  0,2 → + atajo] y [LayerNorm → Dense 128 (GELU) → *dropout* 0,2 → Dense 64 → *dropout* 0,2 → + atajo].
- LayerNorm final → GAP → *dropout* 0,3 (como T) → softmax de 50.
- **`key_dim` = 64 por cabeza.** Es el tamaño de cabeza del ejemplo de Keras para series de tiempo. Con el
  `key_dim` clásico (d / cabezas = 16) el modelo tendría unos 88 mil parámetros, por debajo del rango pedido.
  La dimensión del modelo sigue siendo d = 64.
- GAP en lugar de *token* de clase: es el mismo agregador de T.

**I · ResNet 1D (elegida en lugar de InceptionTime).** ResNet de Wang et al. (2017), la línea base de series de
tiempo que la revisión de Ismail Fawaz et al. (2019; [6] en la revisión) encontró como la mejor de 9 redes,
también en los conjuntos multivariados.

- 3 bloques residuales con 64 filtros cada uno. Cada bloque: Conv1D k = 8 → BN → ReLU → Conv1D k = 5 → BN →
  ReLU → Conv1D k = 3 → BN; suma con el atajo y ReLU.
- Atajo: Conv1D 1 × 1 + BN en el primer bloque (252 → 64 canales); BN en los otros dos, como la implementación
  de Ismail Fawaz et al.
- *Padding* «same». Campo receptivo de 40 pasos, que cubre los 30.
- GAP → *dropout* 0,3 (como T; la ResNet original no lleva *dropout*) → softmax de 50.
- **Por qué ResNet y no InceptionTime.** InceptionTime es un *ensemble* de 5 redes con ventanas 10/20/40
  pensadas para series largas (cientos de pasos). Con 30 pasos hay que reescalar las ventanas a mano (por
  ejemplo 3/5/9), lo que deja de ser la arquitectura publicada, y una sola red sin *ensemble* tampoco lo es.
  Las ventanas 8-5-3 de ResNet, en cambio, caben tal cual en 30 pasos, y ResNet es la referencia más
  establecida de la familia. La TCN ya cubre la variante con dilataciones.

**B · BiGRU.** La arquitectura canónica (Conv1D + recurrente bidireccional + atención + densas) con GRU en vez
de LSTM, en los tamaños de **c07**, la mejor versión de esa familia en la búsqueda de hiperparámetros
(`docs/BUSQUEDA_HIPERPARAMETROS.md`):

- Conv1D 1 × 64 con k = 5 → BN → ReLU → *dropout* 0,15;
- BiGRU 128 (`reset_after=True`, el valor por defecto de Keras) → BN → *dropout* 0,3;
- atención aditiva + GAP + GMP concatenados → densas 128 (*dropout* 0,3) y 64 (*dropout* 0,15) → softmax;
- **lr 5e-4**, el de c07 (lote 64 y *label smoothing* 0,1, como c07 y T). Es la única desviación de la receta
  de T, y está declarada.
- Parte de `build_gru_cnn` (`legacy/src/train_v4.py`) solo en la celda: la estructura es la de
  `busqueda_hiperparametros.construir_modelo`. La prueba unitaria comprueba que, con LSTM, el constructor de B
  reproduce `construir_modelo(c07)` capa por capa y con los mismos pesos iniciales (389.811 parámetros): B solo
  cambia la celda.

### 1.2 T reutilizado: verificación

`python src/experimento_arquitecturas_2.py verificar` comprueba, antes de entrenar:

- que las 15 corridas de `T_tcn` tienen `CONFIG_T` exacta, lr 1e-3, lote 64, *label smoothing* 0,1,
  200 épocas, paciencia 25, aumentación ×10, 30 pasos, `processed_holistic_mascara_rostro`, «manos»,
  «mascara», 252 rasgos y 156.402 parámetros (hecho a las 00:05: correcto);
- el número de parámetros de cada red (hecho: T 156.402, X 187.890, I 315.954, B 341.171);
- con `--dest`, que **este script** reproduce la corrida de T con la semilla 2026 y el pliegue 0 (mismas
  predicciones, top-3, épocas y validación). Se corre después de la espera (§3), porque entrena.

---

## 2. Criterio prerregistrado (escrito el 3-oct-2026 a las 00:08, hora de Colombia, antes de las corridas completas y de la prueba de humo)

1. **Solo decide la validación:** el top-1 medio de validación de los 5 pliegues (media de las 3 semillas por
   pliegue). Es el `val_accuracy` máximo del entrenamiento (el del *checkpoint*), igual que en los experimentos
   anteriores. La prueba LOSO (top-1, top-3 y F1 macro) solo describe.
2. **Margen** = máx(σ, 0,01). σ es la desviación estándar entre semillas, agrupada: para cada brazo de este
   experimento (T, X, I y B), la DE (ddof = 1) entre las 3 semillas de la validación media de 5 pliegues;
   luego, la raíz de la media de esas 4 varianzas.
3. **Un brazo supera a T solo si cumple las dos condiciones:**
   - (a) la media de Δk = val(brazo, k) − val(T, k) es mayor que el margen;
   - (b) Δk > 0 en al menos 4 de los 5 pliegues (las diferencias se redondean a 6 decimales antes de contar,
     por la coma flotante de Keras).
4. **Si varios brazos cumplen** y quedan a menos del margen del mejor, gana el de **menos parámetros**. Si
   ninguno cumple, **T se mantiene** como la mejor arquitectura probada.
5. **Se reporta además (descriptivo; no cambia la regla):**
   - la comparación de cada brazo con el canónico R0 (validación 0,920 / prueba 0,589; brazo A de
     `models/experimento_mano_ausente_2026-10-02`, ya verificado en el experimento anterior), con la misma
     regla aplicada frente a R0;
   - McNemar exacto frente a T sobre las 997 predicciones de prueba agregadas: el principal con la **semilla
     2026**; 2027 y 2028 como complemento; y el **voto de las 3 semillas** (clase por mayoría; si las tres
     semillas difieren, decide la más confiada, porque R0 solo guarda la probabilidad de la clase predicha);
   - pruebas de signos y de Wilcoxon por pliegue, en validación y en prueba. Con 5 pliegues, el p bilateral
     mínimo de ambas es 2/2⁵ = **0,0625**: ninguna puede llegar a 0,05;
   - la persona 0000 (pliegue 0) por separado;
   - parámetros, latencia con lote de 1 (`model.predict`, como `translator.py`, y llamada directa), medida
     sin entrenamientos activos, y tiempo de entrenamiento por pliegue.
6. **Prueba de humo:** 1 pliegue, 3 épocas, en el *carpeta_temporal*. Solo comprueba que el código corre y que los
   modelos se guardan y se recargan; sus cifras no se usan y las configuraciones no cambian por ella (salvo un
   error que impida correr, que se declararía).

---

## 3. Ejecución

- **Espera obligatoria.** Este experimento no entrenó nada mientras corría la confirmación de la TCN
  (`docs/EXPERIMENTO_CONFIRMACION_TCN.md`). Sus 3 procesos de entrenamiento corrieron de 00:00 a 00:21, y su
  medición de latencia terminó hacia las 00:31. A las 00:32 el informe ya tenía su veredicto. Hubo dos
  revisiones sin procesos de Python de entrenamiento, a las 00:31:49 y a las 00:42:11, y tampoco apareció
  ninguno entre ellas (sondeo cada 15 s). Espera total: de las 00:05 a las 00:42, unos 37 minutos. En ese
  tiempo solo se escribió código, se corrieron las pruebas unitarias (construyen modelos sin entrenar) y se
  escribió el criterio.
- **Qué recomendó la confirmación:** T2 (TCN + imputación de la mano ausente + 132 rasgos con 6 velocidades por
  coordenada), y T1 (imputación + 252) como alternativa respaldada por la prueba. **Este experimento no cambia
  su entrada:** sigue con la canónica (mascara + 252), para comparar contra T con lo ya medido.
- **Reproducción de T (00:42-00:43).** `verificar --dest` reentrenó T con la semilla 2026 y el pliegue 0 con
  este script. El historial y las probabilidades de validación y de prueba son idénticos (diferencia
  máxima 0), y `predicciones.json` es **idéntico byte a byte** al de `T_tcn` (55 épocas, mejor 30; validación 0,980; prueba 0,495). El
  *pipeline* de este experimento es, entonces, el mismo que produjo T.
- **Prueba de humo (00:42).** En el *carpeta_temporal*: X, I y B con el pliegue 0, la semilla 2026 y 3 épocas. Los
  tres modelos se guardan y se recargan desde disco (X con su capa `PosicionAprendida`), y la procedencia
  queda anotada. Sus cifras no se usan. Las configuraciones no cambiaron.
- **Lanzamiento de las corridas completas:** 00:43:34, con 3 procesos en paralelo, uno por semilla, cada uno
  en el orden X → I → B y con los pliegues 0 a 4.
- **Fin:** los 3 procesos terminaron a las 01:28, 01:30 y 01:34, sin errores. En total fueron 45 corridas
  (3 brazos × 3 semillas × 5 pliegues), en unos 50 minutos. No hubo recortes.
- **Latencia:** se midió de 01:34 a 01:35, sin procesos de entrenamiento y con la CPU un 89 % libre.
- **Comprobación:** la validación reevaluada sobre el *checkpoint* recargado coincide con la del entrenamiento
  en las 45 corridas (diferencia máxima 3·10⁻⁸, de coma flotante).
- **Datos crudos:** `docs/EXPERIMENTO_ARQUITECTURAS_2.json`. Corridas, modelos, predicciones y probabilidades:
  `models/arquitecturas_2_2026-10-03/`.

---

## 4. Resultados

### 4.1 Tabla resumen

Media de 3 semillas × 5 pliegues. La DE es entre los 5 pliegues (medias de semillas). La validación decide;
la prueba solo describe. R0 es la referencia del canónico (no compite en el criterio).

| Brazo | Parámetros | **Val top-1** (DE) | Δ val vs T | Pliegues mejor que T | Prueba top-1 (DE) | Top-3 | F1 macro | Persona 0000: val / prueba | ms `predict` lote 1 | ms llamada directa | s por pliegue (s/época; épocas) |
|---|---:|---:|---:|:-:|---:|---:|---:|:-:|---:|---:|---:|
| R0 canónico (ref.) | 714.291 | 0,920 (0,020) | −0,029 | 0 | 0,589 (0,094) | 0,764 | 0,544 | 0,923 / 0,452 | 13,3 | 61,3 | 255 (≈4,6; 54)¹ |
| **T TCN + GAP** (control) | **156.402** | **0,949** (0,019) | — | — | **0,651** (0,117) | **0,820** | **0,619** | 0,963 / **0,465** | 11,7 | 7,3 | 128 (2,15; 58)¹ |
| X Transformer pequeño | 187.890 | **0,953** (0,023) | +0,004 | 3 (1 empate) | 0,641 (0,158) | 0,811 | 0,610 | 0,960 / 0,392 | 11,8 | 7,6 | 253 (4,53; 56) |
| I ResNet 1D | 315.954 | 0,948 (0,020) | −0,001 | 2 | 0,648 (0,111) | 0,811 | 0,617 | 0,957 / 0,458 | 11,6 | 7,1 | 132 (2,14; 61) |
| B BiGRU (c07 con GRU) | 341.171 | 0,949 (0,027) | +0,001 | 3 | 0,633 (0,120) | 0,805 | 0,597 | **0,980** / 0,438 | 12,7 | 41,6 | 181 (2,56; 70) |

Notas:

1. **Tiempos de R0 y de T:** vienen de sus experimentos originales (`experimento_mano_ausente` y
   `arquitecturas_literatura`, con 3 o 4 procesos en paralelo). X, I y B corrieron aquí con 3 procesos en
   paralelo. Los segundos por época son lo más comparable.
2. **Latencia:** mediana de 3 modelos (semilla 2026, pliegues 0-2) × 100 llamadas, sobre la red sola (sin
   MediaPipe ni remuestreo). Con `predict` domina el costo fijo de Keras, de unos 11-12 ms. La llamada directa
   `model(x)` muestra el cómputo de la red: X, I y T están en 7-8 ms; B, con su recurrente, en 42 ms; R0, en
   61 ms.
3. **Tamaño en disco:** T 0,74 MB · X 0,83 · I 1,34 · B 1,38 · R0 8,3.

### 4.2 Por pliegue (media de las 3 semillas)

**Validación top-1 (decide):**

| Pliegue (señante) | R0 | T | X | I | B |
|---|---:|---:|---:|---:|---:|
| 0 (0000) | 0,923 | 0,963 | 0,960 | 0,957 | **0,980** |
| 1 (0001) | 0,893 | **0,940** | **0,940** | 0,933 | 0,927 |
| 2 (0002) | 0,917 | 0,937 | 0,940 | **0,947** | 0,923 |
| 3 (0003) | 0,950 | 0,973 | **0,990** | 0,977 | 0,977 |
| 4 (0004) | 0,917 | 0,930 | 0,933 | 0,927 | **0,940** |
| **Media** | 0,920 | 0,949 | **0,953** | 0,948 | 0,949 |

**Prueba top-1 (descriptiva):**

| Pliegue (señante) | R0 | T | X | I | B |
|---|---:|---:|---:|---:|---:|
| 0 (0000) | 0,452 | **0,465** | 0,392 | 0,458 | 0,438 |
| 1 (0001) | 0,536 | 0,628 | 0,586 | 0,643 | **0,653** |
| 2 (0002) | 0,631 | 0,663 | 0,689 | **0,697** | 0,640 |
| 3 (0003) | 0,640 | 0,753 | **0,755** | 0,715 | 0,665 |
| 4 (0004) | 0,687 | 0,745 | **0,782** | 0,725 | 0,768 |
| **Media** | 0,589 | **0,651** | 0,641 | 0,648 | 0,633 |

**F1 macro de prueba (descriptiva):**

| Pliegue | R0 | T | X | I | B |
|---|---:|---:|---:|---:|---:|
| 0 | 0,400 | **0,438** | 0,379 | 0,433 | 0,415 |
| 1 | 0,472 | 0,590 | 0,538 | 0,603 | **0,605** |
| 2 | 0,604 | 0,638 | 0,657 | **0,671** | 0,596 |
| 3 | 0,603 | 0,722 | **0,725** | 0,689 | 0,632 |
| 4 | 0,639 | 0,708 | **0,749** | 0,689 | 0,738 |
| **Media** | 0,544 | **0,619** | 0,610 | 0,617 | 0,597 |

**Por semilla** (media de 5 pliegues):

| Brazo | Val 2026 / 2027 / 2028 | Prueba top-1 2026 / 2027 / 2028 | F1 2026 / 2027 / 2028 |
|---|---|---|---|
| R0 | 0,932 / 0,918 / 0,910 | 0,580 / 0,592 / 0,596 | 0,537 / 0,539 / 0,554 |
| T | 0,948 / 0,946 / 0,952 | 0,657 / 0,645 / 0,651 | 0,624 / 0,619 / 0,614 |
| X | 0,954 / 0,958 / 0,946 | 0,652 / 0,641 / 0,629 | 0,628 / 0,605 / 0,596 |
| I | 0,954 / 0,942 / 0,948 | 0,659 / 0,635 / 0,649 | 0,638 / 0,599 / 0,614 |
| B | 0,948 / 0,948 / 0,952 | 0,612 / 0,649 / 0,638 | 0,569 / 0,621 / 0,601 |

### 4.3 Aplicación del criterio prerregistrado

**Margen.** σ entre semillas agrupada sobre T, X, I y B = 0,0047 (varianzas: T 9·10⁻⁶; X 3,7·10⁻⁵;
I 3,6·10⁻⁵; B 5·10⁻⁶). Como 0,0047 < 0,01, **margen = 0,010**.

| Brazo | Δk de validación frente a T, por pliegue | Media de Δ | (a) Δ > 0,010 | (b) mejor en ≥ 4 de 5 | **Supera a T** |
|---|---|---:|:-:|:-:|:-:|
| X | −0,003 / 0,000 / +0,003 / +0,017 / +0,003 | +0,0040 | no | no (3, 1 empate) | **no** |
| I | −0,007 / −0,007 / +0,010 / +0,003 / −0,003 | −0,0007 | no | no (2) | **no** |
| B | +0,017 / −0,013 / −0,013 / +0,003 / +0,010 | +0,0007 | no | no (3) | **no** |

**Resultado del criterio: ninguno supera a T. T (TCN pequeña con GAP) se mantiene.** Los tres quedan dentro
del margen de T, a ±0,4 puntos de validación media. Ninguno es peor que T según la regla simétrica: ninguno
pierde por más del margen en 4 de 5 pliegues. Las cuatro familias están empatadas en validación.

### 4.4 Comparación con el canónico R0 (validación 0,920, prueba 0,589)

Es la misma regla aplicada frente a R0, de forma descriptiva.

| Brazo | Δ val vs R0 (por pliegue) | Media | Pliegues mejor | Cumpliría el criterio frente a R0 | Δ prueba top-1 vs R0 (por pliegue) | Media | Pliegues mejor |
|---|---|---:|:-:|:-:|---|---:|:-:|
| T | +0,040 / +0,047 / +0,020 / +0,023 / +0,013 | +0,029 | 5 | sí | +0,013 / +0,092 / +0,032 / +0,113 / +0,058 | +0,062 | 5 |
| X | +0,037 / +0,047 / +0,023 / +0,040 / +0,017 | +0,033 | 5 | sí | −0,060 / +0,050 / +0,057 / +0,115 / +0,095 | +0,052 | 4 |
| I | +0,033 / +0,040 / +0,030 / +0,027 / +0,010 | +0,028 | 5 | sí | +0,007 / +0,107 / +0,066 / +0,075 / +0,038 | +0,059 | 5 |
| B | +0,057 / +0,033 / +0,007 / +0,027 / +0,023 | +0,029 | 5 | sí | −0,013 / +0,117 / +0,008 / +0,025 / +0,082 | +0,044 | 4 |

Las cuatro arquitecturas pequeñas superan al canónico en validación por casi 3 puntos, en los 5 pliegues. En
prueba, T e I lo superan en los 5 pliegues; X y B, en 4, y pierden justo con la persona 0000.

### 4.5 Pruebas estadísticas (descriptivas)

**Signos y Wilcoxon por pliegue (n = 5).** Con 5 pliegues, el p bilateral mínimo de ambas pruebas es
2 · (1/2)⁵ = **0,0625**. Solo se alcanza si los 5 pliegues van en el mismo sentido, y ninguna de las dos
puede llegar a 0,05.

| Comparación | Validación | Prueba top-1 | Prueba top-3 | Prueba F1 |
|---|---|---|---|---|
| X − T | 3 a favor, 1 en contra, 1 empate; signos p = 0,63; Wilcoxon p = 0,44 | 3 / 2 (−0,073 / −0,042 / +0,025 / +0,002 / +0,037); 1,0; 0,81 | 2 / 3; 1,0; 0,81 | 3 / 2; 1,0; 0,81 |
| I − T | 2 / 3; 1,0; 1,0 | 2 / 3 (−0,007 / +0,015 / +0,034 / −0,038 / −0,020); 1,0; 0,81 | 1 / 4; 0,38; 0,44 | 2 / 3; 1,0; 1,0 |
| B − T | 3 / 2; 1,0; 1,0 | 2 / 3 (−0,027 / +0,025 / −0,024 / −0,088 / +0,023); 1,0; 0,44 | 1 / 4; 0,38; 0,31 | 2 / 3; 1,0; 0,44 |
| T, X, I, B − R0 | 5 / 0 en los cuatro; 0,0625; 0,0625 | T e I 5 / 0 (0,0625); X y B 4 / 1 (signos 0,38) | — | — |

**McNemar exacto frente a T sobre las 997 predicciones de prueba agregadas.** El principal es el de la
semilla 2026, fijado en el criterio. 2027, 2028 y el voto de las 3 semillas se dan como complemento.

| Brazo | Aciertos brazo / T (s2026) | Solo acierta el brazo | Solo acierta T | p s2026 | p s2027 | p s2028 | Voto de 3 semillas: aciertos brazo / T; p |
|---|---|---:|---:|---:|---:|---:|---|
| X | 650 / 655 | 108 | 113 | 0,79 | 0,84 | 0,14 | 682 / 685; 0,88 |
| I | 657 / 655 | 110 | 108 | 0,95 | 0,54 | 0,94 | 694 / 685; 0,52 |
| B | 610 / 655 | 82 | 127 | **0,0023** (peor) | 0,84 | 0,40 | 666 / 685; 0,16 |
| R0 (ref.) | 578 / 655 | 71 | 148 | 2,1·10⁻⁷ (peor) | 6,3·10⁻⁴ | 2,6·10⁻⁴ | 607 / 685; 4,9·10⁻⁸ |

- **Voto:** la clase por mayoría de las 3 semillas; si las tres difieren (entre 121 y 162 de 997 muestras según
  el brazo), decide la semilla más confiada.
- **El voto sube la exactitud de todos los brazos** (T 0,687; I 0,696; X 0,684; B 0,668; R0 0,609). Estas
  exactitudes son sobre las 997 muestras juntas, no la media por pliegue de la tabla 4.1.
- **B con la semilla 2026 es peor que T (p = 0,002),** pero con 2027, 2028 y el voto no hay diferencia. Es la
  semilla 2026 de B la que queda baja en prueba (0,612 frente a 0,649 y 0,638): B es el brazo más sensible a la
  semilla en prueba (DE entre semillas 0,019, frente a 0,006 de T).
- **Agrupación por señante.** McNemar trata las 997 predicciones como independientes, pero están agrupadas en
  5 señantes. Su p es más optimista que el de las pruebas por pliegue.

### 4.6 Persona 0000 (pliegue 0), aparte

| Brazo | Val (pliegue 0) | Prueba top-1: media (2026 / 2027 / 2028) | Top-3 | F1 |
|---|---:|---|---:|---:|
| R0 | 0,923 | 0,452 (0,460 / 0,480 / 0,415) | 0,627 | 0,400 |
| **T** | 0,963 | **0,465** (0,495 / 0,425 / 0,475) | **0,652** | **0,438** |
| X | 0,960 | **0,392** (0,390 / 0,420 / 0,365) | 0,593 | 0,379 |
| I | 0,957 | 0,458 (0,540 / 0,390 / 0,445) | 0,620 | 0,433 |
| B | **0,980** | 0,438 (0,395 / 0,470 / 0,450) | 0,600 | 0,415 |

- **X repite el patrón de los experimentos anteriores** (como c07): con la persona 0000 cae 7,3 puntos frente a
  T y 6 frente a R0, aunque su validación del pliegue 0 es casi igual. Sus 3 semillas quedan por debajo de las
  3 de T.
- **B tiene la validación más alta del pliegue 0 (0,980) y aun así queda por debajo de T en prueba** (−2,7
  puntos), otra vez la disociación entre validación y persona nueva.
- **I queda en el nivel de T** (0,458 frente a 0,465), pero con mucha dispersión entre semillas (0,39-0,54).
- **0000 sigue siendo, con mucho, la persona más difícil para todas las arquitecturas.**

### 4.7 Costo

- **Parámetros:** T 156.402 < X 187.890 < I 315.954 < B 341.171. T sigue siendo la más pequeña.
- **Latencia con lote de 1** (`predict`, como `translator.py`): R0 13,3 ms · T 11,7 · X 11,8 · I 11,6 ·
  B 12,7. En llamada directa: R0 61,3 · T 7,3 · X 7,6 · I 7,1 · B 41,6. Las tres convolucionales o de atención
  cuestan lo mismo; la BiGRU es unas 6 veces más lenta que T en cómputo de red, aunque sigue siendo más rápida
  que R0.
- **Entrenamiento por pliegue** (3 procesos en paralelo): I 132 s (2,14 s/época) ≈ T 128 s (2,15) < B 181 s
  (2,56; es la que más épocas necesita, 70 de media) < X 253 s (4,53 s/época, el doble que T).

---

## 5. Veredicto

**Según el criterio prerregistrado, ninguna de las tres candidatas supera a la TCN. T (TCN pequeña con GAP)
se mantiene como la mejor arquitectura probada.**

- **Validación:** X 0,953, B 0,949, T 0,949, I 0,948. Las diferencias con T (+0,004, +0,001 y −0,001) están muy
  por debajo del margen de 0,010, y ninguna es mejor que T en 4 de 5 pliegues. Las cuatro familias quedan
  empatadas.
- **Prueba (descriptiva):** ninguna supera a T. Top-1: T 0,651, I 0,648, X 0,641, B 0,633; F1: T 0,619,
  I 0,617, X 0,610, B 0,597. McNemar no encuentra diferencias con T, salvo B con la semilla 2026 (en contra
  de B).
- **Persona 0000:** T sigue siendo la mejor (0,465). X la empeora claramente (0,392).
- **Costo:** T es la más pequeña de todas y, junto con I y X, la más rápida. B cuesta 6 veces más en cómputo
  de red.

**Lectura.** Con esta entrada y estos datos, la ganancia sobre el canónico no depende de la familia: las
cuatro redes pequeñas (convolucional dilatada, ResNet, Transformer y BiGRU) suben unos 3 puntos de validación
y 4-6 de prueba frente a R0. Entre ellas, la validación ya no las distingue, y la prueba y la persona 0000
favorecen ligeramente a T e I. Por parsimonia (menos parámetros, la misma latencia que la más rápida y la
mejor en la persona 0000), **T sigue siendo la candidata**. I (ResNet 1D) es un segundo equivalente en prueba,
con el doble de parámetros.

**Este informe no cambia el modelo canónico, el umbral ni `translator.py`.** Tampoco cambia la recomendación
de la confirmación sobre la entrada (T2, o T1 como alternativa; `docs/EXPERIMENTO_CONFIRMACION_TCN.md`): este
experimento usó la entrada canónica a propósito, y no dice nada sobre la imputación ni sobre las 6
velocidades.

---

## 6. Cautelas

1. **Alcance.** `08_marco.tex:57-61` (anteproyecto) declara las GRU, las TCN y los *transformers* como trabajo
   futuro: «estas no se implementarán en el proyecto, pero se mencionan como referencias y posibles líneas de
   comparación o de trabajo futuro». Este experimento las prueba como comparación, que el párrafo sí admite.
   **Adoptar cualquiera de ellas (T incluida) amplía el alcance, y la decisión es del director.**
2. **Solo 5 señantes.** Ninguna prueba por pliegue puede bajar de p = 0,0625, y el McNemar sobre 997
   predicciones ignora la agrupación por señante. «Empatadas» significa que con 5 personas no se distinguen,
   no que sean equivalentes en general.
3. **La validación está saturada.** Las cuatro rondan 0,95 con 100 muestras por pliegue, donde una muestra
   vale 0,01, y es el máximo de `val_accuracy` a lo largo de las épocas (el *checkpoint*), así que es optimista
   para todas por igual. A este nivel, el criterio no puede separar arquitecturas; por eso el resultado es
   «nadie supera a T» y no «T es mejor que las otras».
4. **Una configuración por familia, sin búsqueda.** El resultado vale para *estas* configuraciones:
   - **X:** `key_dim` 64 (no d / cabezas = 16) para entrar en el rango de 150-400 mil parámetros; posición
     aprendida; sin *warmup* del lr, que a veces se usa con Transformers. Una búsqueda podría mejorarlo, pero
     elegirla con una validación saturada inflaría la cifra (§1.1).
   - **I:** se eligió ResNet y no InceptionTime (§1.1). Se añadió un *dropout* final de 0,3, como T, que la
     ResNet original no tiene.
   - **B:** usa el lr de c07 (5e-4) y no el de T (1e-3). Es la única desviación de la receta y está declarada.
5. **Entrada canónica.** La confirmación de la TCN recomendó cambiar la entrada (imputación, y 132 rasgos en
   T2). Aquí no se aplicó, por diseño. No se sabe si el empate entre familias se mantiene con esa entrada.
6. **T reutilizado.** Sus 15 corridas vienen de otro experimento. Se verificó su configuración, y este script
   reproduce exactamente su semilla 2026, pliegue 0 (§3), así que la comparación es con el mismo *pipeline*.
7. **Tiempos de entrenamiento.** Se midieron con 3 procesos en paralelo, y los de T y R0 vienen de sus
   experimentos. Sirven como orden de magnitud. La latencia sí se midió sin entrenamientos activos.
8. **Capa propia en X.** La posición aprendida es una capa propia (`PosicionAprendida`). Para cargar el modelo
   hay que importar `src/arquitecturas_2.py`. T, I y B usan solo capas nativas de Keras.
9. **Pruebas sin pytest.** `venv312` no tiene pytest, y no se instaló. `tests/test_arquitecturas_2.py` está
   escrito para pytest, pero aquí se ejecutó con un corredor mínimo del *carpeta_temporal* que llama a cada
   `test_*`: pasaron las 9.

---

## 7. Archivos

- **Scripts nuevos:**
  - `src/arquitecturas_2.py`: modelos X, I y B, configuraciones, receta y capa `PosicionAprendida`.
  - `src/experimento_arquitecturas_2.py`: verificar, entrenar, inferencia y resumen.
  - `tests/test_arquitecturas_2.py`: parámetros, softmax, B igual a c07 salvo la celda, estructura de la
    ResNet y del Transformer, guardado y recarga de X, constructor ampliado, voto y McNemar.
- **Modelos y corridas:** `models/arquitecturas_2_2026-10-03/`, con las carpetas `X_transformer`,
  `I_resnet1d` y `B_bigru`. En cada carpeta hay `semilla_S/pliegue_K/` con `modelo.keras`, su procedencia,
  `resultado.json`, `predicciones.json`, `probabilidades.npz` e `historial.json`. Además: `verificacion.json`
  (con la reproducción de T), `humo.json`, `inferencia_redes.json`, `resultados_arquitecturas_2.json`,
  `pids.txt` y los *logs* `corrida_semilla_S.log`.
- **Datos crudos del informe:** `docs/EXPERIMENTO_ARQUITECTURAS_2.json`.
- **Solo lectura:** T en `models/arquitecturas_literatura_2026-10-02/T_tcn/` y R0 en
  `models/experimento_mano_ausente_2026-10-02/A_mascara/`.

Reproducción:

```
python src/experimento_arquitecturas_2.py verificar --dest <scratch>/repro_t
python src/experimento_arquitecturas_2.py entrenar --brazos X,I,B --semillas 2026   # una por proceso
python src/experimento_arquitecturas_2.py inferencia
python src/experimento_arquitecturas_2.py resumen --docs
```

---

## Explicación simple

Probamos tres modelos más que recomienda la literatura: un Transformer pequeño, una ResNet 1D y una red
recurrente GRU. Usamos la misma entrada y las mismas cinco personas, y los comparamos con la TCN, que había
ganado el experimento anterior. Ninguno la superó: en validación los cuatro aciertan casi lo mismo (alrededor
del 95 %), y con personas nuevas la TCN sigue igual o un poco mejor (65 %). Los cuatro le ganan al modelo
actual por unos 3 a 6 puntos, así que lo que ayuda es usar una red pequeña, más que la familia en particular.
El Transformer volvió a empeorar a la persona más difícil (0000) y la GRU es seis veces más lenta. Por eso la
TCN sigue siendo la mejor opción: es la más pequeña y la más rápida. Adoptarla, como cualquiera de estas
redes, le corresponde al director, porque el anteproyecto las dejó como trabajo futuro.
