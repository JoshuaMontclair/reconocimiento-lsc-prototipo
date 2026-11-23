# Experimento de cierre: CNN 2D pequeña («esqueleto como imagen») frente a la TCN final (LOSO)

Fecha: 8 de noviembre de 2026. El director pidió «entrenar y evaluar al menos un modelo más, para que la
investigación quede más completa» (pedido adicional de D-14), y el estudiante eligió una **CNN 2D**, la «red
convolucional compleja» que mencionó el director.

**Es una comparación de cierre.** Hay una sola configuración, fijada antes de entrenar, y no hay búsqueda de
hiperparámetros. El resultado se reporta salga como salga. **No cambia por sí solo el modelo final (D-14)**, ni el
prototipo, ni el umbral, ni `translator.py`.

---

## 1. Diseño

Los dos brazos usan la entrada del modelo final:

- *processed* `data/processed_holistic_mascara_rostro`, representación «manos»;
- imputación lineal de la mano ausente (`mano_ausente_variantes.regla_imputacion`) sobre los cuadros originales y
  después el remuestreo a 30 pasos;
- aumentación ×10 de `train_loso` (espejo sobre las 126 posiciones) y después `add_velocity` (252 rasgos).

La receta de entrenamiento es la de la TCN: Adam lr 1e-3 con CosineDecay, *label smoothing* 0,1, lote 64, paciencia
25 sobre `val_accuracy` y 200 épocas máximas. LOSO de 5 pliegues (D-06) con validación interna de 2 por seña (100
muestras). Semillas 2026, 2027 y 2028.

| Brazo | Modelo | Parámetros | Origen |
|---|---|---:|---|
| **T1** TCN final (control) | `arquitecturas_literatura.construir_tcn` (`CONFIG_T`) | 156.402 | **Reutilizado**: `models/confirmacion_tcn_2026-10-03/T1_imputacion_252` (15 corridas) |
| **C2D** CNN 2D pequeña | `arquitectura_cnn2d.construir_cnn2d` (`CONFIG_C2D`) | 146.546 | **Nuevo**: `models/cnn2d_2026-11-08/C2D_cnn2d` |

El script nuevo `src/experimento_cnn2d.py` no reimplementa el entrenamiento. Entrena con
`experimento_arquitecturas.entrenar_corrida`, la misma función que entrenó T1. Para eso amplía en memoria su
`constructor` con el brazo C2D, como hizo `experimento_arquitecturas_2.py`. La `X_seq` «imputacion» y la función de
entrada de T1 vienen de `experimento_confirmacion_tcn.preparar` y `canon_de_brazo`. Ningún script existente se
modificó.

### 1.1 La entrada como imagen

La ventana de 30 × 252 se reorganiza como una imagen de **30 (tiempo) × 21 (puntos de la mano, en el orden de
MediaPipe) × 12 canales**:

- canales 0-5: (x, y, z, vx, vy, vz) de la mano izquierda **de la persona**;
- canales 6-11: los mismos 6 de la mano derecha.

**Orden real de columnas, verificado** (`src/representacion.py` y `add_velocity`):

- La representación «manos» toma del *processed* las columnas 99:162 (mano izquierda) y 162:225 (mano derecha).
  Son 21 puntos × (x, y, z) intercalados.
- `add_velocity` agrega después las 126 velocidades en el mismo orden.
- Por tanto, la columna de (mano m, posición o velocidad v, punto p, coordenada c) es
  126·v + 63·m + 3·p + c, y el canal de la imagen es 6·m + 3·v + c.
- La remodelación es (30, 252) → (30, 2 [pos/vel], 2 [mano], 21, 3) → permutación → (30, 21, 2 [mano],
  2 [pos/vel], 3) → (30, 21, 12).

**Dónde se aplica.** La remodelación está **dentro del modelo** (capas `Reshape` → `Permute` → `Reshape`). Por eso
se aplica siempre **después** de la aumentación (espejo incluido) y de `add_velocity`, y la entrada del modelo sigue
siendo 30 × 252, como la que arma `translator.py`.

**Qué se ganó con este orden:**

- Las dos manos no quedan una al lado de la otra en el eje espacial: son canales del mismo punto.
- Los vecinos del eje de puntos son puntos consecutivos de MediaPipe en la misma mano (0 muñeca, 1-4 pulgar, 5-8
  índice, 9-12 medio, 13-16 anular, 17-20 meñique). Solo los pasos 4→5, 8→9, 12→13 y 16→17 saltan de la punta de un
  dedo a la base del siguiente: por eso «casi siempre».

**Comprobaciones** (`verificar`, 8-nov, terminado a las 15:01:40 y repetido a las 15:03 para escribir el JSON en `models/`, sobre 64 videos reales y en `tests/test_cnn2d.py`):

1. Cada canal de la imagen es exactamente la columna esperada.
2. La capa del modelo da, bit a bit, la misma imagen que la versión numpy (`remodelar_a_imagen_np`).
3. **Tras el espejo de la aumentación** (que intercambia las manos y cambia el signo de x) **y `add_velocity`**, la
   imagen es la original con los canales 0-5 y 6-11 intercambiados y con x y vx negadas. La diferencia máxima es 0.
4. Una prueba con una remodelación errónea (manos lado a lado en el eje de puntos) falla la comprobación.

### 1.2 Arquitectura (fijada antes de entrenar; una sola, sin variantes)

`CONFIG_C2D` en `src/arquitectura_cnn2d.py`:

| Etapa | Capas | Salida |
|---|---|---|
| Entrada | 30 × 252 → imagen | 30 × 21 × 12 |
| Bloque 1 | [Conv2D 32, 3×3, *same* → BN → ReLU] × 2 → SpatialDropout2D 0,15 → MaxPooling 2×1 | 15 × 21 × 32 |
| Bloque 2 | [Conv2D 64, 3×3 → BN → ReLU] × 2 → SpatialDropout2D 0,15 → MaxPooling 2×1 | 7 × 21 × 64 |
| Bloque 3 | [Conv2D 64, 3×3 → BN → ReLU] × 2 → SpatialDropout2D 0,15 (sin *pooling*) | 7 × 21 × 64 |
| Cabeza | GlobalAveragePooling2D → Dropout 0,3 → Dense 50 softmax | 50 |

- **Parámetros: 146.546**, dentro del rango pedido (100-350 mil) y comparable con la TCN (156.402).
- Es la arquitectura sugerida (3 bloques 3×3, 32 → 64 → 64, BN, ReLU, *dropout* espacial, *pooling* 2×1 solo en el
  tiempo y solo en los dos primeros bloques, GAP, *dropout* 0,3). Se fijaron **dos convoluciones por bloque**
  porque con una sola la red tendría 62.802 parámetros, por debajo del rango. Así también queda paralela a la TCN, que
  usa dos convoluciones por bloque.
- El eje de puntos **no se reduce nunca** (21 en todo el camino). El tiempo pasa de 30 a 15 y a 7.
- Campo receptivo: unos 13 puntos en el eje de puntos y prácticamente los 30 pasos en el tiempo. El GAP final
  resume todo.
- *Dropout* espacial de 0,15, a mitad del rango sugerido (0,1-0,2), porque apaga mapas completos y es más fuerte que
  el *dropout* por elemento de la TCN (0,2).
- Sin conexiones residuales ni normalización de entrada, para que sea una CNN 2D plana y la diferencia con la TCN sea
  la convolución en dos ejes.

**Referencias de «esqueleto como imagen»** (verificadas el 8-nov en Crossref y en el texto de la segunda):

- Y. Du, Y. Fu y L. Wang, «Skeleton based action recognition with convolutional neural network», *3rd IAPR Asian
  Conference on Pattern Recognition (ACPR)*, 2015, pp. 579-583, doi:10.1109/ACPR.2015.7486569. Convierte la secuencia
  de esqueleto en una imagen y la clasifica con una CNN.
- C. Li, Q. Zhong, D. Xie y S. Pu, «Co-occurrence Feature Learning from Skeleton Data for Action Recognition and
  Detection with Hierarchical Aggregation», *IJCAI-18*, 2018, pp. 786-792, doi:10.24963/ijcai.2018/109. Representa la
  secuencia «as a tensor of shape frames×joints×3 (the last dimension as channel)», que es la misma disposición de
  este brazo (tiempo × puntos × coordenadas como canales). También integra las coordenadas y sus diferencias
  temporales, como aquí las posiciones y las velocidades.
- Q. Ke, M. Bennamoun, S. An, F. Sohel y F. Boussaid, «A New Representation of Skeleton Sequences for 3D Action
  Recognition», *CVPR*, 2017, pp. 4570-4579, doi:10.1109/CVPR.2017.486 (otra variante de la misma idea).

Advertencia: las tres son de reconocimiento de acciones con esqueleto del cuerpo, no de lengua de señas con puntos
de la mano. La configuración concreta (tamaños, *dropout*, *pooling*) **no** viene de ellas: es la sugerida para
este cierre, ajustada solo para quedar en el rango de parámetros.

### 1.3 T1 reutilizado: verificación (8-nov, terminada a las 15:01:40)

`python src/experimento_cnn2d.py verificar --dest <carpeta_temporal>/repro_t1`:

- **Configuración.** Las 15 corridas de T1 tienen `CONFIG_T` exacta, lr 1e-3, lote 64, *label smoothing* 0,1, 200
  épocas, paciencia 25, aumentación ×10, 30 pasos, el *processed* canónico, «manos», imputación, 252 rasgos y
  156.402 parámetros.
- **Reproducción (condición previa).** Se reentrenó T1 con la semilla 2026 y el pliegue 0 con este script (el
  constructor ampliado), en el *carpeta_temporal*:
  - `predicciones.json` **idéntico byte a byte**;
  - historial idéntico y probabilidades de validación y de prueba con diferencia máxima 0;
  - mismas 43 épocas, validación 0,980, `val_loss` 0,870123 y prueba 0,540.

  Por eso T1 se reutiliza sin reentrenar (`models/cnn2d_2026-11-08/verificacion.json`).

### 1.4 Prueba de humo (8-nov, 15:01:52-15:02:03)

En el *carpeta_temporal*: C2D con la semilla 2026, el pliegue 0 y 3 épocas, a 3,3 s por época. El modelo se guarda y se
recarga con entrada 30 × 252 y 146.546 parámetros, y la procedencia queda corregida (imputación, 252, script y
brazo). Las cifras de la prueba de humo no se usan (`models/cnn2d_2026-11-08/humo.json`).

---

## 2. Criterio prerregistrado (escrito el 8-nov-2026; archivo guardado a las 15:04:01, hora de Colombia, 9 s antes de lanzar las corridas completas)

No se ha visto ningún resultado completo de C2D, solo la prueba de humo de 3 épocas. Las cifras de T1 ya se
conocían (D-14). **Solo deciden métricas de validación. La prueba LOSO nunca decide.**

**Notación.** Para cada brazo V, semilla s y pliegue k:

- `val(V, s, k)` es el `val_accuracy` máximo (el del *checkpoint*);
- `loss(V, s, k)` es la `val_loss` del *checkpoint* recargado, la que guarda `entrenar_corrida` (entropía cruzada con
  *label smoothing* 0,1);
- `val(V, k)` y `loss(V, k)` son la media de las 3 semillas.

Las diferencias por pliegue se redondean a 6 decimales, y las comparaciones con un umbral usan una tolerancia de
1e-9, como en los experimentos anteriores.

1. **Regla 1: top-1 medio de validación.**
   - **Margen** = máx(σ, 0,01). σ es la DE entre semillas agrupada sobre los 2 brazos: para cada brazo, la DE
     (ddof = 1) entre las 3 semillas de la validación media de 5 pliegues; luego, la raíz de la media de las 2
     varianzas (`experimento_confirmacion_tcn.sigma_agrupada`).
   - Con Δk = val(C2D, k) − val(T1, k): **C2D gana** si la media de Δk es mayor que el margen **y** Δk > 0 en al menos
     4 de 5 pliegues (Δk = 0 no cuenta). **T1 gana** con la misma regla sobre −Δk.
   - Si ninguno gana, hay **empate en top-1** y se pasa a la regla 2.
2. **Regla 2: desempate con la `val_loss`** (solo si hay empate en top-1; menor es mejor).
   - **Margen de pérdida** = σ_loss, la DE entre semillas de la `val_loss` media de 5 pliegues, agrupada sobre los 2
     brazos igual que σ (sin piso).
   - Con dk = loss(T1, k) − loss(C2D, k): **C2D gana** si la media de dk es mayor que σ_loss **y** dk > 0 en al menos
     4 de 5 pliegues. **T1 gana** con la misma regla sobre −dk.
   - Si ninguno gana, el resultado es **empate**.
3. **Veredicto.**
   - **C2D solo «supera» a la TCN** si gana por la regla 1, o por la regla 2 tras un empate en la regla 1.
   - Si empata o pierde, **la TCN sigue siendo el modelo final (D-14)**.
   - Aunque C2D ganara, **el cambio de modelo final es decisión del director**. Este experimento no lo cambia.
4. **Descriptivo, sin valor de decisión:**
   - prueba top-1, top-3 y F1 macro;
   - la persona 0000 (pliegue 0) aparte;
   - McNemar exacto C2D frente a T1 sobre las 997 predicciones de prueba agregadas, con la semilla 2026 (y las otras
     dos) y con el voto de las 3 semillas (mayoría; si las tres discrepan, gana la semilla más confiada;
     `experimento_confirmacion_tcn.voto_semillas`);
   - signos y Wilcoxon por pliegue (validación, `val_loss`, prueba top-1, top-3 y F1). Con 5 pliegues, el p bilateral
     mínimo es 2/2⁵ = **0,0625**, así que ninguna de las dos puede llegar a 0,05;
   - parámetros, latencia con lote de 1 (`predict`, como `translator.py`, y llamada directa), medida sin
     entrenamientos activos después de las corridas, y tiempo de entrenamiento por pliegue;
   - una tabla final de las familias comparadas en la tesis, con validación y prueba top-1 LOSO, la entrada con que
     se midió cada una y su fuente.
5. **Plan de ejecución.** Tres procesos en paralelo, uno por semilla, cada uno con los pliegues 0 a 4. Son 15 corridas.
   Si no terminaran, el criterio se aplicaría con las semillas que tengan los 5 pliegues (mínimo 2), y se declararía.
6. **Condición previa** (ya cumplida, §1.3): T1 solo se reutiliza si su reproducción con este script da
   `predicciones.json` idéntico byte a byte.

---

## 3. Ejecución

- **Antes de lanzar** se comprobó que no hubiera entrenamientos activos (`otros_entrenamientos()` vacío).
- **Lanzamiento:** 8-nov a las 15:04:10, 9 s después de guardar el criterio. Fueron 3 procesos en paralelo, uno por
  semilla, con los pliegues 0 → 4 (`corrida_semilla_<s>.log`, PID en `pids.txt`). Terminaron a las 15:41:42 (2027),
  15:45:12 (2028) y 15:46:17 (2026). Son 15 corridas sin errores ni recortes, y el criterio se aplica con las 3
  semillas completas.
- **Coherencia.** En las 15 corridas, el `val_accuracy` máximo coincide con la exactitud del modelo recargado sobre la
  validación (`incoherencias_val_top1_vs_reevaluado` vacío).
- **Latencia:** se midió a las 15:46:45-15:47:03, sin ningún proceso de entrenamiento activo.
- **Criterio:** no se modificó después de ver resultados. Lo único que se corrigió del §2 fue su hora: el borrador
  decía 15:05 y el archivo se guardó a las 15:04:01, antes del lanzamiento (§6, anomalía 1).

---

## 4. Resultados

### 4.1 Tabla T1 frente a C2D

Media de 3 semillas × 5 pliegues. **Validación** (decide): top-1 y `val_loss`. **Prueba** (descriptiva): el resto.

| Brazo | Parámetros | **Val top-1** (DE pl.) | **`val_loss`** (DE sem.) | Prueba top-1 (DE pl.) | Top-3 | F1 macro | Voto 3 semillas | Persona 0000: val / prueba | ms `predict` lote 1 | ms `model(x)` | MB en disco | s por pliegue (s/época; épocas) |
|---|---:|---:|---:|---:|---:|---:|---:|:-:|---:|---:|---:|---:|
| **T1 TCN final** | 156.402 | **0,949** (0,017) | **0,978** (0,023) | **0,677** (0,127) | **0,847** | **0,647** | **0,716** | **0,960 / 0,488** | 14,6 | 8,5 | 0,74 | **123** (2,16; 56)¹ |
| C2D CNN 2D | 146.546 | 0,866 (0,013) | 1,218 (0,011) | 0,524 (0,117) | 0,728 | 0,486 | 0,558 | 0,873 / 0,340 | 14,7 | **6,0** | **0,66** | 482 (7,68; 63) |
| Diferencia C2D − T1 | −9.856 | **−0,083** | **+0,240** | **−0,154** | −0,119 | −0,161 | −0,158 | −0,087 / −0,148 | +0,1 | −2,5 | −0,08 | ×3,9 |

¹ Tiempos de T1 tomados de sus corridas del 3-oct, también con 3 procesos en paralelo. Su reproducción de hoy, sola,
tardó 55 s (1,24 s/época).

**Por pliegue** (media de 3 semillas; pliegues 0 / 1 / 2 / 3 / 4, señantes 0000 a 0004):

| Brazo | Val top-1 | `val_loss` | Prueba top-1 |
|---|---|---|---|
| T1 | 0,960 / 0,937 / 0,937 / 0,973 / 0,940 | 0,897 / 1,012 / 1,046 / 0,920 / 1,013 | 0,488 / 0,611 / 0,721 / 0,770 / 0,795 |
| C2D | 0,873 / 0,863 / 0,860 / 0,883 / 0,850 | 1,149 / 1,289 / 1,235 / 1,160 / 1,255 | 0,340 / 0,477 / 0,576 / 0,593 / 0,632 |

**Por semilla** (media de 5 pliegues; 2026 / 2027 / 2028):

| Brazo | Val top-1 | `val_loss` | Prueba top-1 |
|---|---|---|---|
| T1 | 0,950 / 0,954 / 0,944 | 0,975 / 0,956 / 1,001 | 0,688 / 0,658 / 0,685 |
| C2D | 0,858 / 0,878 / 0,862 | 1,224 / 1,205 / 1,224 | 0,536 / 0,503 / 0,533 |

La peor semilla de T1 en validación (0,944) queda 0,066 por encima de la mejor de C2D (0,878). En prueba, la peor de T1
(0,658) supera a la mejor de C2D (0,536) por 0,122.

### 4.2 Aplicación del criterio prerregistrado

- **σ entre semillas** (top-1, agrupada sobre los 2 brazos) = 0,0083 (varianzas: T1 2,5·10⁻⁵, C2D 1,1·10⁻⁴).
  **Margen = máx(0,0083; 0,01) = 0,010.**
- **Regla 1.** Δk = val(C2D, k) − val(T1, k) = −0,087 / −0,073 / −0,077 / −0,090 / −0,090; media **−0,0833**.

| Quién gana | Media a su favor | (a) > margen 0,010 | (b) a favor en ≥ 4 de 5 | **Gana** |
|---|---:|:-:|:-:|:-:|
| C2D | −0,0833 | no | no (0 de 5) | **no** |
| T1 | +0,0833 | sí | sí (5 de 5) | **sí** |

- **Regla 2** (`val_loss`): **no se aplica**, porque la regla 1 no terminó en empate. Como dato informativo, también
  la favorece a T1: σ_loss = 0,018 y dk = loss(T1) − loss(C2D) = −0,252 / −0,277 / −0,189 / −0,241 / −0,242 (media
  −0,240), es decir, T1 tiene menor pérdida en los 5 pliegues.
- **Resultado: C2D pierde frente a T1, decidido por el top-1 de validación.** La TCN sigue siendo el modelo final
  (D-14).

### 4.3 Pruebas pareadas por pliegue (descriptivas)

Diferencias C2D − T1 por pliegue (media de 3 semillas). Con 5 pliegues, el p bilateral mínimo de la prueba de signos y
de Wilcoxon es 2/2⁵ = **0,0625**: ninguna de las dos puede llegar a 0,05, ni siquiera con 5 de 5 pliegues en contra.

| Métrica | Δ por pliegue (0 / 1 / 2 / 3 / 4) | A favor / en contra / empates de C2D | Signos p | Wilcoxon p |
|---|---|:-:|---:|---:|
| Val top-1 | −0,087 / −0,073 / −0,077 / −0,090 / −0,090 | 0 / 5 / 0 | 0,0625 | 0,0625 |
| `val_loss` (+ = peor C2D) | +0,252 / +0,277 / +0,189 / +0,241 / +0,242 | 0 / 5 / 0 | 0,0625 | 0,0625 |
| Prueba top-1 | −0,148 / −0,134 / −0,145 / −0,177 / −0,163 | 0 / 5 / 0 | 0,0625 | 0,0625 |
| Prueba top-3 | −0,165 / −0,101 / −0,104 / −0,115 / −0,108 | 0 / 5 / 0 | 0,0625 | 0,0625 |
| Prueba F1 macro | −0,152 / −0,136 / −0,147 / −0,195 / −0,174 | 0 / 5 / 0 | 0,0625 | 0,0625 |

Es el resultado más extremo que admiten 5 pliegues: C2D es peor en todos, en todas las métricas.

### 4.4 McNemar sobre las 997 predicciones de prueba agregadas (descriptivo)

| Semilla | Aciertos C2D / T1 | Solo acierta C2D | Solo acierta T1 | Ambos | Ninguno | p exacto bilateral |
|---|---|---:|---:|---:|---:|---:|
| 2026 | 534 / 686 | 68 | 220 | 466 | 243 | 7,4·10⁻²⁰ |
| 2027 | 501 / 656 | 54 | 209 | 447 | 287 | 1,1·10⁻²² |
| 2028 | 531 / 683 | 68 | 220 | 463 | 246 | 7,4·10⁻²⁰ |
| **Voto de 3 semillas** | **556 / 714** | 49 | 207 | 507 | 234 | **2,7·10⁻²⁴** |

T1 acierta unas 3 veces más muestras que C2D falla de las que C2D acierta y T1 falla. (Las semillas 2026 y 2028 dan
por casualidad los mismos discordantes, 68 y 220; los aciertos y las predicciones son distintos, §6.)

### 4.5 Persona 0000 (pliegue 0), aparte

| Brazo | Val (pliegue 0) | `val_loss` | Prueba top-1: media (2026 / 2027 / 2028) | Top-3 | F1 |
|---|---:|---:|---|---:|---:|
| **T1** | **0,960** | **0,897** | **0,488** (0,540 / 0,390 / 0,535) | **0,660** | **0,469** |
| C2D | 0,873 | 1,149 | 0,340 (0,370 / 0,320 / 0,330) | 0,495 | 0,317 |

C2D pierde 0,148 con la persona más difícil, una caída del mismo tamaño que en los otros señantes (−0,134 a
−0,177). No hay un daño especial para 0000, pero tampoco ninguna mejora.

### 4.6 Costo

- **Latencia de la red** (`inferencia.json`; semilla 2026, pliegues 0-2, orden intercalado; mediana de 100 llamadas,
  media de 3 modelos, sin entrenamientos activos):
  - `predict` con lote de 1, como `translator.py`: **T1 14,6 ms, C2D 14,7 ms**. Es la misma cifra: domina el costo
    fijo de `predict`.
  - llamada directa `model(x, training=False)`: T1 8,5 ms y C2D 6,0 ms.
  - La entrada (armado de la ventana con imputación) es la misma para los dos brazos, así que no se volvió a medir.
- **Disco:** T1 0,74 MB y C2D 0,66 MB.
- **Entrenamiento:** C2D tarda 7,7 s por época frente a 2,2 s de T1 (×3,6) y 482 s por pliegue frente a 123 s (×3,9),
  en los dos casos con 3 procesos en paralelo. Cada convolución recorre 21 puntos en vez de 1 fila de 252 rasgos.

### 4.7 Lectura descriptiva (no decide nada)

- C2D **no se quedó corta de entrenamiento**: paró por paciencia (63 épocas de media, mejor época 38), igual que T1.
- En la mejor época, C2D acierta el 96,0 % del entrenamiento aumentado frente al 98,6 % de T1, y la brecha con la
  validación es mayor (0,094 frente a 0,037). Ajusta peor **y** generaliza peor.
- Una hipótesis, **no probada aquí**: la TCN mezcla los 252 rasgos de cada paso en su primera capa. La CNN 2D, en
  cambio, solo ve 3 puntos vecinos por capa y la relación entre puntos lejanos de la mano (por ejemplo, la punta del
  pulgar y la del meñique) solo aparece tras varias capas. El ST-GCN, el otro modelo que impone una estructura
  espacial sobre los puntos, también quedó muy por debajo (§5).

---

## 5. Tabla final: las 13 filas de familias comparadas (para el documento de tesis)

Validación y prueba top-1 LOSO de 5 pliegues (validación interna D-06 de 2 por seña). Las cifras son la media de
semillas × pliegues, leídas de los JSON de datos crudos de cada informe (`resumen` las recalcula en
`docs/EXPERIMENTO_CNN2D.json`, clave `tabla_familias`). **Solo T1 y C2D se midieron con imputación**; las demás, con
«mascara» + 252.

| # | Familia | Entrada | Semillas | Parámetros | **Val top-1** | **Prueba top-1** | Prueba F1 macro | Fuente |
|---:|---|---|:-:|---:|---:|---:|---:|---|
| 1 | Random Forest | «mascara» + 252 → 2.016 estadísticos² | 1 | — | 0,766³ | 0,418 | 0,373⁴ | D-10/N-39; `models/linea_base_4d_2026-09-15/resultados.json` |
| 2 | SVM RBF | «mascara» + 252 → 2.016 estadísticos² | 1 | — | 0,494³ | 0,264 | 0,229⁴ | ídem |
| 3 | MLP (1 capa de 256) | «mascara» + 252 → 2.016 estadísticos² | 1 | — | 0,488³ | 0,248 | 0,222⁴ | ídem |
| 4 | Conv1D + BiLSTM + atención (canónico, R0) | «mascara» + 252 | 3 | 714.291 | 0,920 | 0,589 | 0,544 | `docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md` §4.1 |
| 5 | c07 (Conv1D + BiLSTM de la búsqueda, R1) | «mascara» + 252 | 3 | 389.811 | 0,935 | 0,592 | 0,551 | ídem |
| 6 | MiniRocket + *ridge* | «mascara» + 252 | 3 | 519.842⁵ | 0,695 | 0,429 | 0,392 | ídem |
| 7 | TCN pequeña + GAP (T) | «mascara» + 252 | 3 | 156.402 | 0,949 | 0,651 | 0,619 | ídem |
| 8 | ST-GCN pequeño | «mascara» + 252 | 3 | 97.858 | 0,679 | 0,341 | 0,302 | ídem |
| 9 | Transformer pequeño | «mascara» + 252 | 3 | 187.890 | 0,953 | 0,641 | 0,610 | `docs/EXPERIMENTO_ARQUITECTURAS_2.md` §4.1 |
| 10 | ResNet 1D | «mascara» + 252 | 3 | 315.954 | 0,948 | 0,648 | 0,617 | ídem |
| 11 | BiGRU (c07 con GRU) | «mascara» + 252 | 3 | 341.171 | 0,949 | 0,633 | 0,597 | ídem |
| 12 | **TCN final (D-14)**: TCN pequeña + GAP | **imputación + 252** | 3 | 156.402 | **0,949** | **0,677** | **0,647** | `docs/EXPERIMENTO_FACTORIAL_TCN.md` §4.2 (brazo T1); este informe |
| 13 | **CNN 2D pequeña** (esqueleto como imagen) | **imputación + 252** | 3 | 146.546 | 0,866 | 0,524 | 0,486 | este informe |

Notas de la tabla:

- Todas usan el *processed* `data/processed_holistic_mascara_rostro`, la representación «manos», 30 pasos y las
  particiones LOSO de D-06. Las redes (filas 4-13) se entrenan con la aumentación ×10; MiniRocket se ajusta sobre el
  mismo entrenamiento aumentado.
- ² Línea base clásica (D-10): ocho estadísticos por cada una de las 252 columnas, con aumentación de espejo ×2 y una
  sola semilla (2026). Con la aumentación completa de las redes sube poco (Random Forest 0,442 en prueba; N-39).
- ³ Validación de la configuración elegida en cada pliegue por máximo de validación dentro de una rejilla pequeña:
  es optimista frente a las redes, que tienen una sola configuración.
- ⁴ Media de los 5 pliegues, como en el resto de la tabla. N-39 reporta la F1 sobre las 997 predicciones agregadas
  (Random Forest 0,393; SVM 0,282; MLP 0,257); el top-1 coincide en las dos formas.
- ⁵ Número de rasgos de MiniRocket más los pesos del *ridge*, tal como lo reporta su informe.
- La fila 7 y la fila 12 son la **misma arquitectura** con distinto tratamiento de la mano ausente. Por eso son 13
  filas y **12 familias distintas** (11 anteriores + la CNN 2D).
- Leer las filas 12-13 contra las filas 4-11 mezcla dos efectos: la arquitectura y la imputación. La imputación sola
  sube la TCN de 0,651 a 0,677 en prueba y no cambia la validación (filas 7 y 12).

---

## 6. Veredicto

**Según el criterio prerregistrado, la CNN 2D pequeña NO supera a la TCN final: pierde.** La regla 1 lo decide sola.
En validación, C2D acierta 0,083 menos de media (margen 0,010) y es peor en los 5 pliegues. La `val_loss` no hizo falta,
y también va en contra (+0,240, 5 de 5 pliegues).

- **La TCN pequeña + GAP con imputación y 252 rasgos sigue siendo el modelo final (D-14).** Este experimento no cambia
  el prototipo, el umbral ni `translator.py`.
- Lo descriptivo va en la misma dirección y con margen amplio: −0,154 en prueba top-1, −0,161 en F1, −0,148 con la
  persona 0000, McNemar con p ≈ 10⁻²⁰ en cada semilla y signos y Wilcoxon en su mínimo posible (0,0625).
- La CNN 2D queda por encima de la línea base clásica y de MiniRocket y el ST-GCN, pero por debajo del canónico
  (0,866 frente a 0,920 en validación; 0,524 frente a 0,589 en prueba) y de todas las redes 1D.
- **Si este cierre hubiera dado lo contrario, el cambio de modelo final seguiría siendo decisión del director.** Lo
  mismo vale ahora: el informe le entrega la evidencia, y es él quien da por atendido el pedido de D-14.

## 7. Cautelas y anomalías

1. **Hora del criterio.** El borrador del §2 decía «15:05», pero el archivo se guardó a las 15:04:01 y las corridas se
   lanzaron a las 15:04:10. Se corrigió la hora antes de ver ningún resultado. El texto del criterio no cambió.
2. **Commit durante las corridas.** A las 15:13:20, con las corridas en marcha, se hizo el commit 64fc9dc
   («Experimentos de octubre, modelo final TCN + imputación (D-14)…»). No se hizo dentro de este experimento. El commit incluyó el
   estado parcial de este experimento: los scripts, la prueba, este informe con el criterio y el §3 «PENDIENTE», los
   logs a medio escribir y el pliegue 0 de cada semilla. Los scripts y la prueba versionados son idénticos a los
   actuales; los logs, los pliegues 1-4, los JSON y este informe completo quedan sin versionar. Como efecto
   secundario, el commit deja constancia de que el criterio existía antes de los resultados de los pliegues 1-4.
3. **Una sola configuración.** El resultado es de esta CNN 2D, no de la familia. Por diseño no se probaron variantes
   (atajos residuales, núcleos separados para tiempo y puntos, la transposición de puntos a canales de HCN). Una
   búsqueda podría acercarla a la TCN, pero con una brecha de 0,083 en validación y 0,154 en prueba es poco probable
   que la supere.
4. **El eje de puntos es una línea artificial.** El 3×3 mezcla puntos consecutivos de MediaPipe, y 4 de esos 20 pasos
   no son vecinos anatómicos (punta de un dedo → base del siguiente). Además, el *padding* «same» rellena con ceros en
   la muñeca (punto 0) y en la punta del meñique (punto 20).
5. **Latencia más alta que en octubre.** T1 dio hoy 14,6 ms con `predict` (11,6 ms en
   `docs/EXPERIMENTO_FACTORIAL_TCN.md` §4.3). Es el estado del equipo, no el modelo. Se compara solo dentro de esta
   medición, que es intercalada y sin entrenamientos activos.
6. **Discordantes iguales en dos semillas.** McNemar de las semillas 2026 y 2028 da los mismos discordantes (68 y 220).
   Se comprobó que es casualidad: los aciertos son distintos (534/686 frente a 531/683) y las 15 corridas tienen
   predicciones diferentes.
7. **Procesos ajenos.** Había 11 procesos de Python `multiprocessing` de hace 7 días, con 0 % de CPU, que no son
   entrenamientos. No se tocaron.
8. **Recuento de familias.** El encargo habla de «12 familias ya comparadas» y de una tabla de «13 familias». La
   lista enumerada tiene 11 familias; la tabla tiene 13 filas porque la TCN aparece dos veces (con «mascara» y con
   imputación). Con la CNN 2D son 12 familias distintas.

## 8. Archivos

- `src/arquitectura_cnn2d.py`: `CONFIG_C2D`, `construir_cnn2d`, la remodelación en numpy y los índices de columna y
  canal.
- `src/experimento_cnn2d.py`: subcomandos `verificar`, `entrenar`, `inferencia` y `resumen`.
- `tests/test_cnn2d.py`: 16 pruebas de remodelación, espejo, modelo, constructor ampliado, criterio y McNemar
  (`python -m unittest tests.test_cnn2d -v`: 16 OK).
- `models/cnn2d_2026-11-08/`:
  - `C2D_cnn2d/semilla_<s>/pliegue_<k>/`: modelo, procedencia, resultado, historial, predicciones y probabilidades;
  - `verificacion.json`, `humo.json` e `inferencia.json`;
  - `resultados_cnn2d.json`, los registros `corrida_semilla_<s>.log` y `pids.txt`.
- `docs/EXPERIMENTO_CNN2D.json`: datos crudos (criterio, por brazo, pruebas, McNemar, persona 0000, tabla de familias,
  verificación, inferencia y las 30 corridas sin predicciones).
- La reproducción de T1 y la prueba de humo quedaron en el *carpeta_temporal* de la sesión, fuera del repositorio.

## Explicación simple

El director pidió probar un modelo más, y elegimos una red convolucional 2D. Esta red mira cada seña como una «imagen»:
el tiempo hacia abajo, los 21 puntos de la mano hacia el lado y, en cada punto, la posición y la velocidad de las dos
manos. Le dimos la misma entrada, las mismas cinco personas y el mismo entrenamiento que a la TCN, que es el modelo
final. Fijamos la regla para decidir antes de entrenar. La CNN 2D perdió con claridad: acierta el 87 % en validación
frente al 95 % de la TCN, y el 52 % con personas nuevas frente al 68 %. Fue peor con las cinco personas y entrena casi
cuatro veces más lento. Por eso la TCN sigue siendo el modelo final; el director decide si con esto queda atendido su
pedido.
