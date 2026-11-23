# Experimento: arquitecturas recomendadas por la literatura frente al modelo canónico (LOSO)

Fecha: 2 de octubre de 2026. Motivación: `docs/REVISION_LITERATURA_ARQUITECTURAS.md` (§4 y §5).

**Pregunta.** ¿Alguna de las arquitecturas que recomienda la literatura acierta más que el modelo actual con
personas nuevas?

Este documento no cambia el modelo canónico, el umbral ni el alcance.

---

## 1. Diseño

Todos los brazos usan exactamente la misma entrada canónica y las mismas particiones. Lo único que cambia es
el modelo.

- **Entrada:** HolisticLandmarker → normalización por hombros → regla «mascara» (D-07) → representación «manos»
  (126 rasgos) → 30 pasos → aumentación ×10 de `train_loso` (espejo sobre las 126 posiciones) →
  `add_velocity` → 30 × 252.
- **Datos:** `data/processed_holistic_mascara_rostro`, columnas «manos».
- **Evaluación:** LOSO de 5 pliegues (D-06; `data/partitions/loso_5_pliegues.json`), con validación interna de
  2 muestras por seña tomadas de los cuatro señantes de entrenamiento.
- **Semillas:** 2026, 2027 y 2028 en todos los brazos.

| Brazo | Modelo | Entrenamiento |
|---|---|---|
| **R0** canónico (control) | `build_model` de `train_final_100.py` (Conv1D ×2 → BiLSTM ×2 → atención → densas) | Se **reutiliza** el brazo A de `models/experimento_mano_ausente_2026-10-02` (ver §1.1) |
| **R1** c07 | `construir_modelo(c07)` de `busqueda_hiperparametros.py` | Adam lr 5e-4 con CosineDecay, *label smoothing* 0,1, lote 64, paciencia 25, 200 épocas |
| **M** MiniRocket + *ridge* | `MiniRocketMultivariate` (sktime 1.2.0) + `StandardScaler` + `RidgeClassifier` | Sobre el mismo entrenamiento aumentado ×10 que ven las redes |
| **T** TCN pequeña con GAP | `construir_tcn` (`src/arquitecturas_literatura.py`) | El del canónico: Adam lr 1e-3 con CosineDecay, *label smoothing* 0,1, lote 64, paciencia 25 sobre `val_accuracy`, 200 épocas |
| **G** ST-GCN pequeño (opcional) | `construir_stgcn` (`src/arquitecturas_literatura.py`) | Igual que T |

### 1.1 R0 reutilizado: verificación

`python src/experimento_arquitecturas.py verificar` comprueba, antes de entrenar nada
(`models/arquitecturas_literatura_2026-10-02/verificacion.json`), lo siguiente:

- Las 15 corridas del brazo A de `experimento_mano_ausente` tienen la configuración canónica:
  `processed_holistic_mascara_rostro`, «manos», «mascara», aumentación ×10, 200 épocas, paciencia 25, lote 64,
  *dropout* 0,3, 30 pasos y 714.291 parámetros.
- Sus predicciones de prueba, su validación y sus épocas coinciden **exactamente** con las del brazo A de
  `experimento_velocidades_2026-09-30`, un experimento independiente con las mismas semillas.
- La semilla 2026 reproduce la referencia N-30 (0,460 / 0,533 / 0,606 / 0,615 / 0,685).
- La `X_seq` de este experimento es, bit a bit, la del brazo A.

Por eso R0 no se reentrena.

### 1.2 Configuración de cada brazo (fijada antes de ver resultados; una por familia, sin variantes)

**R1 · c07.** Es la configuración que recomendó el criterio de la búsqueda de hiperparámetros
(`docs/BUSQUEDA_HIPERPARAMETROS.md`): conv 1 × 64 con k = 5, BiLSTM 128, densas 128-64, *dropout* 0,3, lr 5e-4,
lote 64 y *label smoothing* 0,1. Tiene 389.811 parámetros. Allí se evaluó con las semillas 2031-2033; aquí se
corre con 2026-2028, para tenerla en las mismas condiciones que R0. Las corridas de la semilla 2026 se
cotejan con las de la fase 1 de la búsqueda, que usaron el mismo *pipeline* y la misma semilla.

**M · MiniRocket + *ridge*** (revisión, candidata A; [8], [9]).

- `MiniRocketMultivariate(num_kernels=10000, max_dilations_per_kernel=32, random_state=semilla)` sobre
  (n, 252 canales, 30 pasos). Los sesgos se ajustan con el entrenamiento aumentado. Son los valores por
  defecto del artículo y de sktime.
- `StandardScaler` ajustado solo con el entrenamiento.
- `RidgeClassifier` con α elegido por validación cruzada interna **agrupada por señante** (`GroupKFold` con los
  4 señantes de entrenamiento). Las copias aumentadas heredan el señante de su original, así que ninguna copia
  de un video queda a ambos lados. En cada partición interna, el escalado se reajusta con su parte de
  entrenamiento.
- Rejilla de α: 17 valores en `logspace(-3, 5)`. Es la de `RidgeClassifierCV` del artículo (`logspace(-3, 3)`)
  ampliada hacia arriba, porque los 10 000 rasgos están estandarizados. Gana la mayor exactitud interna media;
  en empate, el α mayor. Luego se reajusta con todo el entrenamiento.
- Por qué la validación cruzada no es la LOO por defecto de `RidgeClassifierCV`: sobre datos aumentados ×10,
  la LOO deja las otras 9 copias de cada video en el entrenamiento y elegiría un α demasiado pequeño.
- La validación del pliegue (2 por seña) **no** interviene en el ajuste: solo se mide.
- Entrenamiento aumentado ×10, el mismo de las redes: se exporta desde el entorno de TensorFlow con la misma
  semilla y la misma llamada a `train_loso.aumentar`. Cabe en memoria (6970 × 9996 rasgos; unos 6,5 GB de
  pico).
- 3 semillas: la de la aumentación y la de MiniRocket son la misma (2026, 2027, 2028).
- Entorno: sktime y numba no están en el entorno del proyecto y **no se instalaron en él**. Se creó un entorno
  aparte en el *carpeta_temporal* de la sesión (Python 3.12, sktime 1.2.0, numba 0.68.0, scikit-learn 1.8.0). Los
  tensores de cada pliegue se exportan a `.npz`. Script: `src/experimento_arquitecturas_minirocket.py`.

**T · TCN pequeña con GAP** (revisión, candidata B; [5], [6], [12]).

- 4 bloques residuales al estilo de Bai et al. [12], con dilataciones 1, 2, 4 y 8.
- Cada bloque tiene 2 × [Conv1D de 64 filtros, k = 3, *padding* «same» → BatchNorm → ReLU → *dropout* 0,2],
  y luego la suma con el atajo (1 × 1 si cambian los canales) y ReLU.
- Después: GlobalAveragePooling1D → *dropout* 0,3 → softmax de 50.
- **156.402 parámetros.** Campo receptivo de 61 pasos, que cubre los 30.
- *Padding* «same» (no causal), porque se clasifica la ventana completa, igual que el canónico.
- Parte de la idea de `build_temporal_cnn` (`legacy/src/train_extra_models.py`), pero sin densas grandes ni
  GMP, que es lo que la revisión recomienda reducir.

**G · ST-GCN pequeño** (revisión, candidata D; [2], [3]; Yan et al. 2018).

- Grafo de las dos manos: 42 nodos, las 21 aristas de la mano de MediaPipe
  (`HandLandmarksConnections.HAND_CONNECTIONS` del mediapipe instalado) en cada mano, y autoconexiones.
  Las dos manos son componentes separadas.
- Adyacencia fija y normalizada: Â = D^-½ (A + I) D^-½.
- Remodelación de la entrada: (30, 252) → (30, 42, 6). Los 6 canales son x, y, z, vx, vy, vz, y el nodo es
  21·mano + punto.
- Orden de columnas verificado: «manos» = `mano_izq` (99:162) + `mano_der` (162:225), con x, y, z intercaladas
  y las velocidades después, en el mismo orden.
- Después: BatchNorm por nodo y canal → 3 bloques [ConvGrafo → BN → ReLU → Conv2D temporal 9 × 1 → BN →
  *dropout* 0,2 → + atajo → ReLU], con canales 32, 64 y 64 y paso temporal 1, 2 y 1 → GAP sobre tiempo y
  nodos → *dropout* 0,3 → softmax.
- **97.858 parámetros.**
- Espejo: la aumentación aplica el espejo antes de `add_velocity`. `verificar` comprueba que, tras remodelar,
  el espejo equivale a intercambiar los nodos de las dos manos y negar x y vx. Así, la remodelación sigue
  siendo coherente después de aumentar. También comprueba que las capas Reshape/Permute del modelo hacen lo
  mismo que la versión numpy.

### 1.3 Prueba de humo (21:47-21:51)

En el *carpeta_temporal*, no en `models/`. Todos los brazos se corrieron con el pliegue 0 y la semilla 2026:

- R1, T y G con 3 épocas: los modelos se guardan y se recargan, incluido G con su capa propia.
- M con su configuración completa.

Otras comprobaciones:

- La *ridge* espectral de la validación cruzada interna coincide con `sklearn.RidgeClassifier` en 3 valores de α.
- La latencia de M se mide bien.

**Declaración:** la prueba de humo de M ya es una corrida completa del pliegue 0 con la semilla 2026, y su
resultado se vio antes de escribir este criterio. La configuración de M no se cambió por él; estaba fijada
en el código antes de correrla. Las cifras de la prueba de humo no se usan; la corrida oficial se repite en
`models/`.

---

## 2. Criterio prerregistrado (escrito el 2-oct-2026 a las 21:52, hora de Colombia, antes de las corridas completas)

1. **Solo decide la validación:** el top-1 medio de validación de los 5 pliegues (media de las 3 semillas por
   pliegue).
   - En las redes es el `val_accuracy` máximo del entrenamiento, el del *checkpoint*, igual que en todos los
     experimentos anteriores.
   - En M es la exactitud del modelo final sobre la misma validación, que no se usa para ajustar nada.
   - La prueba LOSO (top-1, top-3 y F1 macro) se reporta como descripción y **no decide nada**.
2. **Margen** = máx(σ, 0,01). σ es la desviación estándar entre semillas, agrupada: para cada brazo
   (R0, R1, M, T y G), la DE (ddof = 1) entre las 3 semillas de la validación media de 5 pliegues; luego, la
   raíz de la media de esas varianzas.
3. **Un brazo X supera al canónico solo si cumple las dos condiciones:**
   - (a) la media de Δk = val(X, k) − val(R0, k) es mayor que el margen;
   - (b) Δk > 0 en al menos 4 de los 5 pliegues.
4. **Si varios brazos cumplen** y quedan a menos del margen del mejor, gana el de **menos parámetros**. Si
   ninguno cumple, se mantiene el canónico.
5. **Pruebas que se reportan** (descriptivas; no cambian la regla):
   - McNemar exacto sobre las 997 predicciones de prueba agregadas de cada brazo frente a R0. El principal
     usa las de la **semilla 2026**; las de 2027 y 2028 se dan como complemento.
   - Pruebas de signos y de Wilcoxon por pliegue, en validación y en prueba. Con 5 pliegues, el p bilateral
     mínimo de ambas es 2/2⁵ = **0,0625**: ninguna puede llegar a 0,05.
6. **La persona 0000 (pliegue 0) se reporta aparte.** En todos los experimentos anteriores, las mejoras en
   validación la empeoraron en prueba.
7. **Costo.** Se reportan:
   - los parámetros;
   - la latencia de inferencia con lote de 1, como `translator.py`, medida sin otros entrenamientos activos;
   - el tiempo de entrenamiento por pliegue.
8. **Presupuesto** (unas 6 horas): si no alcanza, se recorta primero G y luego las semillas de los brazos más
   lentos, y se declara.

---

## 3. Ejecución

- **Lanzamiento:** 21:52. Las redes corrieron en 3 procesos en paralelo, uno por semilla, cada uno en el orden
  R1 → T → G y pliegues 0 a 4. MiniRocket corrió en un cuarto proceso, de 21:52 a 22:16. Las redes terminaron
  a las 23:45, unas 2 horas de cómputo en total.
- **Latencia:** se midió a las 23:46, con todos los entrenamientos terminados y la CPU con un 80 % libre.
- **Recortes:** ninguno. Entraron los cinco brazos, cada uno con 3 semillas y 5 pliegues.
- **Datos crudos:** `docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.json`. Corridas, modelos, predicciones y
  probabilidades: `models/arquitecturas_literatura_2026-10-02/`.
- **Reproducción de R1:** con la semilla 2026, R1 reproduce **exactamente** la fase 1 de la búsqueda de
  hiperparámetros (c07) en los 5 pliegues: misma validación, misma prueba y mismas épocas. Así, el
  entrenamiento de este script es el mismo *pipeline*.
- **Ajuste de código posterior a las corridas:** las exactitudes de Keras son float32, así que en el pliegue 2
  la diferencia R1 − R0 salió −2·10⁻⁸ cuando en realidad es 0. Las diferencias se redondean a 6 decimales
  antes de contar pliegues, para que eso cuente como empate. No cambia el criterio ni ninguna decisión: R1
  sigue mejor en 4 de 5.

---

## 4. Resultados

### 4.1 Tabla resumen

Media de 3 semillas × 5 pliegues. La DE es entre los 5 pliegues (medias de semillas). La validación decide;
la prueba solo describe.

| Brazo | Parámetros | **Val top-1** (DE) | Δ val vs R0 | Pliegues mejor | Prueba top-1 (DE) | Top-3 | F1 macro | Persona 0000: val / prueba | ms `predict` lote 1 | ms llamada directa | s por pliegue (s/época) |
|---|---:|---:|---:|:-:|---:|---:|---:|:-:|---:|---:|---:|
| R0 canónico | 714.291 | **0,920** (0,020) | — | — | 0,589 (0,094) | 0,764 | 0,544 | 0,923 / 0,452 | 14,0 | 62,3 | 255 (4,6)¹ |
| R1 c07 | 389.811 | **0,935** (0,021) | +0,015 | 4 (+1 empate) | 0,592 (0,141) | 0,764 | 0,551 | 0,943 / **0,363** | 13,3 | 33,1 | 163 (2,8) |
| M MiniRocket + *ridge* | 519.842² | **0,695** (0,039) | −0,225 | 0 | 0,429 (0,164) | 0,599 | 0,392 | 0,743 / 0,172 | 0,6³ | — | 94 (—) |
| **T TCN + GAP** | **156.402** | **0,949** (0,019) | **+0,029** | **5** | **0,651** (0,117) | **0,820** | **0,619** | **0,963 / 0,465** | **12,3** | **7,6** | 128 (2,2) |
| G ST-GCN | 97.858 | **0,679** (0,025) | −0,241 | 0 | 0,341 (0,131) | 0,526 | 0,302 | 0,710 / 0,135 | 13,1 | 6,9 | 1028 (17,0) |

Notas:

1. **Tiempos de R0:** vienen de su experimento original (`experimento_mano_ausente`, 3 semillas en paralelo).
   Los demás brazos corrieron aquí con 3 o 4 procesos en paralelo. Los segundos por época son más
   comparables que los segundos por pliegue.
2. **Parámetros de M:** coeficientes de la *ridge* (50 × 9996), interceptos, y media y escala del escalado.
   Los 84 núcleos de MiniRocket son fijos y no se aprenden.
3. **Latencia de M:** transformación de sktime, escalado y *ridge* con lote de 1, en su propio entorno
   (numba y numpy). No es comparable uno a uno con el `predict` de Keras, cuyo costo fijo ronda los 12 ms.
   Ninguna cifra incluye MediaPipe ni el remuestreo.

Tamaño en disco: R0 8,3 MB · R1 1,56 MB · T 0,74 MB · G 0,50 MB · M 4,1 MB.

Valores de α elegidos en M por la validación cruzada interna (por señante): 10³ (2 veces), 10^3,5 (9) y 10⁴
(4). Ninguno quedó en el borde de la rejilla.

### 4.2 Por pliegue (media de las 3 semillas)

**Validación top-1 (decide):**

| Pliegue (señante) | R0 | R1 | M | T | G |
|---|---:|---:|---:|---:|---:|
| 0 (0000) | 0,923 | 0,943 | 0,743 | **0,963** | 0,710 |
| 1 (0001) | 0,893 | 0,917 | 0,647 | **0,940** | 0,660 |
| 2 (0002) | 0,917 | 0,917 | 0,720 | **0,937** | 0,650 |
| 3 (0003) | 0,950 | 0,967 | 0,697 | **0,973** | 0,697 |
| 4 (0004) | 0,917 | **0,930** | 0,670 | **0,930** | 0,680 |
| **Media** | 0,920 | 0,935 | 0,695 | **0,949** | 0,679 |

**Prueba top-1 (descriptiva):**

| Pliegue (señante) | R0 | R1 | M | T | G |
|---|---:|---:|---:|---:|---:|
| 0 (0000) | 0,452 | 0,363 | 0,172 | **0,465** | 0,135 |
| 1 (0001) | 0,536 | 0,591 | 0,377 | **0,628** | 0,322 |
| 2 (0002) | 0,631 | 0,594 | 0,500 | **0,663** | 0,402 |
| 3 (0003) | 0,640 | 0,682 | 0,497 | **0,753** | 0,358 |
| 4 (0004) | 0,687 | 0,730 | 0,598 | **0,745** | 0,487 |
| **Media** | 0,589 | 0,592 | 0,429 | **0,651** | 0,341 |

**F1 macro de prueba (descriptiva):**

| Pliegue | R0 | R1 | M | T | G |
|---|---:|---:|---:|---:|---:|
| 0 | 0,400 | 0,320 | 0,150 | **0,438** | 0,120 |
| 1 | 0,472 | 0,534 | 0,327 | **0,590** | 0,278 |
| 2 | 0,604 | 0,558 | 0,462 | **0,638** | 0,363 |
| 3 | 0,603 | 0,661 | 0,470 | **0,722** | 0,321 |
| 4 | 0,639 | 0,680 | 0,551 | **0,708** | 0,427 |

**Por semilla.** Validación media de 5 pliegues y prueba top-1 media:

| Brazo | Val 2026 / 2027 / 2028 | Prueba 2026 / 2027 / 2028 |
|---|---|---|
| R0 | 0,932 / 0,918 / 0,910 | 0,580 / 0,592 / 0,596 |
| R1 | 0,932 / 0,936 / 0,936 | 0,599 / 0,590 / 0,588 |
| M | 0,696 / 0,704 / 0,686 | 0,431 / 0,432 / 0,422 |
| T | 0,948 / 0,946 / 0,952 | 0,657 / 0,645 / 0,651 |
| G | 0,672 / 0,686 / 0,680 | 0,333 / 0,349 / 0,340 |

T supera a R0 en las tres semillas, tanto en validación como en prueba.

### 4.3 Aplicación del criterio prerregistrado

**Margen.** σ entre semillas agrupada = 0,0073 (varianzas: R0 1,2·10⁻⁴; R1 5·10⁻⁶; M 8·10⁻⁵; T 9·10⁻⁶;
G 5·10⁻⁵). Como 0,0073 < 0,01, **margen = 0,010**.

| Brazo | Δk de validación por pliegue | Media de Δ | (a) Δ > 0,010 | (b) mejor en ≥ 4 de 5 | **Supera a R0** |
|---|---|---:|:-:|:-:|:-:|
| R1 | +0,020 / +0,023 / 0,000 / +0,017 / +0,013 | +0,0147 | sí | sí (4, 1 empate) | **sí** |
| M | −0,180 / −0,247 / −0,197 / −0,253 / −0,247 | −0,2247 | no | no (0) | no |
| T | +0,040 / +0,047 / +0,020 / +0,023 / +0,013 | **+0,0287** | sí | sí (5) | **sí** |
| G | −0,213 / −0,233 / −0,267 / −0,253 / −0,237 | −0,2407 | no | no (0) | no |

R1 y T cumplen las dos condiciones. Entre ellos, T queda por encima de R1 por 0,014 en validación (por
pliegue: +0,020 / +0,023 / +0,020 / +0,007 / 0,000), más que el margen. Por eso no hay empate que desempatar
por tamaño; de todos modos, T es también el de menos parámetros entre los dos.

**Resultado del criterio: T (TCN pequeña con GAP).** M y G quedan muy por debajo del canónico: −22 y −24
puntos de validación, peores en 5 de 5 pliegues.

### 4.4 Pruebas estadísticas (descriptivas)

**Signos y Wilcoxon por pliegue (n = 5).** Con 5 pliegues, el p bilateral mínimo de ambas pruebas es
2 · (1/2)⁵ = **0,0625**. Solo se alcanza si los 5 pliegues van en el mismo sentido, y ninguna de las dos
puede llegar a 0,05.

| Comparación | Validación | Prueba top-1 | Prueba top-3 | Prueba F1 |
|---|---|---|---|---|
| R1 − R0 | 4 a favor, 1 empate; signos p = 0,125; Wilcoxon p = 0,125 | 3 / 2; p = 1,0; 0,81 | 3 / 2; p = 1,0; 1,0 | 3 / 2; p = 1,0; 1,0 |
| M − R0 | 0 / 5; p = 0,0625; 0,0625 | 0 / 5; 0,0625; 0,0625 | 0 / 5 | 0 / 5 |
| **T − R0** | **5 / 0; p = 0,0625; 0,0625** | **5 / 0** (+0,013 / +0,092 / +0,032 / +0,113 / +0,058); 0,0625; 0,0625 | **5 / 0**; 0,0625 | **5 / 0** (+0,039 / +0,117 / +0,034 / +0,119 / +0,068); 0,0625 |
| G − R0 | 0 / 5; 0,0625; 0,0625 | 0 / 5 | 0 / 5 | 0 / 5 |

**McNemar exacto sobre las 997 predicciones de prueba agregadas, frente a R0.** El principal es el de la
semilla 2026, fijado en el criterio; 2027 y 2028 se dan como complemento.

| Brazo | Aciertos brazo / R0 (s2026) | Solo acierta el brazo | Solo acierta R0 | p exacto s2026 | p s2027 | p s2028 |
|---|---|---:|---:|---:|---:|---:|
| R1 | 597 / 578 | 111 | 92 | 0,21 | 0,95 | 0,62 |
| M | 430 / 578 | 80 | 228 | 1,3·10⁻¹⁷ (peor) | 2·10⁻¹⁹ | 9·10⁻²² |
| **T** | **655 / 578** | **148** | **71** | **2,1·10⁻⁷** | **6,3·10⁻⁴** | **2,6·10⁻⁴** |
| G | 332 / 578 | 63 | 309 | 4·10⁻⁴⁰ (peor) | 5·10⁻⁴¹ | 1·10⁻⁴¹ |

McNemar trata las 997 predicciones como independientes, pero están agrupadas por señante (5 grupos). Su p es
más optimista que el de las pruebas por pliegue. Aun así, la dirección de T es la misma en los 5 pliegues y en
las 3 semillas.

### 4.5 Persona 0000 (pliegue 0), aparte

| Brazo | Val (pliegue 0) | Prueba top-1: media (2026 / 2027 / 2028) | Top-3 | F1 |
|---|---:|---|---:|---:|
| R0 | 0,923 | 0,452 (0,460 / 0,480 / 0,415) | 0,627 | 0,400 |
| R1 | 0,943 | **0,363** (0,410 / 0,330 / 0,350) | 0,512 | 0,320 |
| M | 0,743 | 0,172 (0,170 / 0,170 / 0,175) | 0,313 | 0,150 |
| **T** | **0,963** | **0,465** (0,495 / 0,425 / 0,475) | 0,652 | 0,438 |
| G | 0,710 | 0,135 (0,125 / 0,145 / 0,135) | 0,302 | 0,120 |

- **R1 repite el patrón de los experimentos anteriores:** sube 2 puntos en la validación del pliegue 0 y baja
  9 puntos con la persona 0000.
- **T es la primera variante que no lo repite.** Sube 4 puntos en validación y en prueba queda igual o
  ligeramente mejor: +1,3 puntos de top-1, +2,5 de top-3 y +3,8 de F1. Pero la diferencia de top-1 está dentro
  del ruido entre semillas: los rangos 0,425–0,495 y 0,415–0,480 se solapan. Lo que se puede afirmar es que T
  **no empeora** a la persona 0000, no que la mejore.
- **0000 sigue siendo, con mucho, la persona más difícil para todos los brazos.**

### 4.6 Costo

- **Parámetros:** T tiene 156.402, un 78 % menos que el canónico y un 60 % menos que c07. Su archivo pesa
  0,74 MB, frente a 8,3 MB.
- **Latencia con lote de 1** (`predict`, como `translator.py`; mediana de 3 modelos × 100 llamadas):
  R0 14,0 ms · R1 13,3 · T 12,3 · G 13,1.
  - Con `predict` domina el costo fijo de Keras, de unos 12 ms, así que la diferencia real se ve en la llamada
    directa `model(x)`: R0 62,3 ms · R1 33,1 · **T 7,6** · G 6,9. T es unas 8 veces más rápida que el canónico
    en cómputo de red.
  - MiniRocket: 0,6 ms por ventana en su entorno.
- **Entrenamiento por pliegue,** con 3 o 4 procesos en paralelo:
  - R1: 163 s (2,8 s/época);
  - T: 128 s (2,2 s/época; 58 épocas de media);
  - G: 1028 s (17,0 s/época);
  - M: 94 s (90 de ellos en la validación cruzada de α; la transformación tarda menos de 2 s);
  - R0, en su experimento original: 255 s (unos 4,6 s/época).

---

## 5. Veredicto

**Según el criterio prerregistrado, la TCN pequeña con GAP (brazo T) supera al modelo canónico.**

- **Validación:** media de 0,949 frente a 0,920, es decir +2,9 puntos, casi tres veces el margen de 0,010. Es
  mejor en los 5 pliegues y en las 3 semillas.
- **c07 (R1)** también cumple (+1,5 puntos, 4 de 5 pliegues con 1 empate). Pero queda 1,4 puntos por debajo
  de T, más que el margen, así que no empata con ella.
- **MiniRocket (M) y el ST-GCN pequeño (G)** quedan muy por debajo: unos −22 a −24 puntos en validación y −16
  a −25 en prueba.

**Como descripción (no decide):**

- En prueba, T también mejora en los 5 pliegues: top-1 0,651 frente a 0,589 (+6,2 puntos), top-3 0,820 frente
  a 0,764 y F1 macro 0,619 frente a 0,544 (+7,5).
- McNemar con la semilla 2026: 148 frente a 71 aciertos exclusivos, p = 2·10⁻⁷.
- Es la primera mejora de validación en este proyecto que no empeora a la persona 0000.
- Además, T tiene 4,6 veces menos parámetros y su red corre unas 8 veces más rápido.
- c07, en cambio, no se traduce en prueba (0,592 frente a 0,589) y vuelve a empeorar a la persona 0000.

**Este informe no cambia el modelo canónico.** Adoptar T amplía el alcance del anteproyecto (ver §6.1) y es
decisión del director. Si se adopta, habría que hacer lo siguiente:

- reentrenar el modelo final con T;
- recalibrar el umbral de confianza (D-09) y el umbral anidado;
- volver a medir la latencia de punta a punta en `translator.py`. T solo usa capas nativas de Keras, así que se
  carga sin objetos propios.

---

## 6. Cautelas

1. **Alcance.** `08_marco.tex:57-61` declara la TCN (y las demás alternativas) como trabajo futuro: «estas no
   se implementarán en el proyecto». El ST-GCN tampoco está en el anteproyecto. Este experimento las prueba
   para responder la pregunta del director, pero **adoptar T o G amplía el alcance y la decisión es del
   director**. MiniRocket podría presentarse como línea base adicional (`09_metodologia.tex:131-134`), aunque
   añade una dependencia, sktime, que no está en `requirements.txt` y no se añadió.
2. **Solo 5 señantes.** Ninguna prueba por pliegue puede bajar de p = 0,0625, y el McNemar sobre 997
   predicciones ignora la agrupación por señante. El efecto de T es consistente (5 de 5 pliegues, 3 de 3
   semillas, validación y prueba), pero la población de señantes es mínima. Generalizar «T es mejor» a
   personas fuera de LSC50 sigue siendo una hipótesis.
3. **La validación de las redes es optimista.** Es el máximo de `val_accuracy` a lo largo de las épocas, el
   *checkpoint*. La de M no, porque se mide sin elegir nada con ella. El tamaño de ese sesgo no se midió, pero
   cabe esperar que esté muy por debajo de la distancia de 22 puntos que separa a M del resto, así que no
   cambiaría el veredicto. Entre las redes, el sesgo es el mismo para todas. Además, la
   validación está cerca del techo (0,95 con 100 muestras: una muestra vale 0,01).
4. **Una configuración por familia, sin búsqueda.** El resultado de M y de G vale para *esas*
   configuraciones, no para la familia.
   - **M:** los núcleos de MiniRocket suman cero (6 pesos de −1 y 3 de +2, según su definición [9]). Por eso
     la convolución no ve el nivel constante de cada canal, es decir, la posición y la configuración estática
     de la mano, salvo en los bordes con relleno. **Hipótesis, no verificada:** eso explicaría que se quede
     en 0,70 incluso con los señantes vistos. Una variante con rasgos de nivel (por ejemplo, medias por canal
     junto a los de MiniRocket) no se probó.
   - **G:** es un ST-GCN mínimo: 3 bloques, partición «uniforme» con una sola Â y GAP sobre los 42 nodos. Así
     pierde qué dedo hace qué. Además, se ajusta casi al 100 % en entrenamiento y se queda en 0,68 en
     validación. El ST-GCN original usa 10 bloques y particiones espaciales con pesos distintos, y en OpenHands
     tiene 2,3 M de parámetros [3]. Este resultado **no** es evidencia contra las redes de grafos en general.
   - El margen de mejora de T con ajuste tampoco se exploró.
5. **Prueba de humo de M.** Se corrió con la configuración completa en el pliegue 0, semilla 2026, antes de
   escribir el criterio (§1.3), y su resultado se vio. La configuración no se cambió.
6. **Semillas.** Se usaron 2026-2028, como pide el encargo, y no las 2031-2033 que sugiere la revisión (§5).
   R0 se reutilizó tras verificar su configuración (§1.1). R1 con 2026-2028 da 0,935 en validación, igual que
   con 2031-2033 en la búsqueda (0,935): el resultado de c07 es estable.
7. **Tiempos de entrenamiento.** Se midieron con varios procesos en paralelo, y los de R0 vienen de otro
   experimento. Sirven como orden de magnitud. La latencia sí se midió sin entrenamientos activos.
8. **Entorno de M.** sktime 1.2.0, numba 0.68.0 y scikit-learn 1.8.0, en un entorno aparte del *carpeta_temporal*
   de la sesión. El entorno del proyecto (`venv312`) no se modificó. Los tensores exportados (`.npz`, 3,1 GB)
   están en el *carpeta_temporal* y se regeneran con
   `python src/experimento_arquitecturas.py exportar --dest <carpeta>`.

---

## 7. Archivos

- **Scripts nuevos:**
  - `src/experimento_arquitecturas.py`: verificar, exportar, entrenar, inferencia y resumen.
  - `src/arquitecturas_literatura.py`: modelos T y G, grafo y capa `ConvGrafo`.
  - `src/experimento_arquitecturas_minirocket.py`: brazo M y su latencia.
  - `tests/test_arquitecturas_literatura.py`: grafo, orden de columnas y conmutación con el espejo.
- **Modelos y corridas:** `models/arquitecturas_literatura_2026-10-02/`, con las carpetas `R1_c07`,
  `M_minirocket`, `T_tcn` y `G_stgcn`. En cada carpeta hay `semilla_S/pliegue_K/` con el modelo, `resultado.json`,
  `predicciones.json` y las probabilidades o puntajes. Además: `verificacion.json`, `humo.json`,
  `inferencia_redes.json`, `inferencia_minirocket.json`, `resultados_arquitecturas.json` y los *logs*.
- **Datos crudos del informe:** `docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.json`.
- **R0:** `models/experimento_mano_ausente_2026-10-02/A_mascara/` (solo lectura).

Reproducción:

```
python src/experimento_arquitecturas.py verificar
python src/experimento_arquitecturas.py exportar --dest <scratch>/export
<scratch>/venv_minirocket/bin/python src/experimento_arquitecturas_minirocket.py correr --datos <scratch>/export
python src/experimento_arquitecturas.py entrenar --brazos R1,T,G --semillas 2026   # una por proceso
python src/experimento_arquitecturas.py inferencia
<scratch>/venv_minirocket/bin/python src/experimento_arquitecturas_minirocket.py latencia
python src/experimento_arquitecturas.py resumen --docs
```

---

## Explicación simple

Probamos cuatro modelos distintos con la misma entrada y las mismas cinco personas, y los comparamos con el
modelo actual. El ganador fue una red convolucional pequeña (TCN): solo mira ventanas cortas del movimiento y
luego promedia. Acertó más que el modelo actual con las cinco personas nuevas y con las tres semillas: pasó del
59 % al 65 %. Además, es casi cinco veces más pequeña y mucho más rápida. Por primera vez, la mejora no
perjudicó a la persona más difícil (0000). MiniRocket y la red de grafos pequeña quedaron muy por debajo, pero
eso vale para las versiones simples que se probaron, no para esas familias en general. Como la TCN estaba
declarada como trabajo futuro en el anteproyecto, adoptarla le corresponde al director.
