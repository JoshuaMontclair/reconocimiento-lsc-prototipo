# Experimento: 30 frente a 60 pasos temporales con el modelo final (TCN + imputación + 252)

Estado: **§1 escrita el 22-nov-2026 a las 23:55 (hora de Colombia), antes de cualquier entrenamiento con
60 pasos** (ni siquiera la prueba de humo). Lo que sigue a §1 se escribe después y no la modifica. Copia de §1
con su SHA-256: `models/pasos_tcn_2026-11-23/criterio_prerregistrado.md` y `.sha256`.

## 1. Pregunta, diseño y criterio prerregistrado

### 1.1 Pregunta

El anteproyecto fijaba 60 pasos temporales; el modelo final (D-14: TCN pequeña con GAP + imputación de la
mano ausente + 252 rasgos, brazo T1 de `models/confirmacion_tcn_2026-10-03/T1_imputacion_252`) usa 30. El
valor 30 salió del experimento de pasos de septiembre (`docs/EXPERIMENTO_PASOS_TEMPORALES.md`, N-58), hecho con
la red de partida (Conv1D + BiLSTM + atención). Allí la regla prerregistrada señaló 60 (+1,6 puntos de
validación, 5 de 5 pliegues), sin confirmación en prueba (+0,9 puntos, 2 de 5 pliegues), y el proyecto siguió con
30. La comparación nunca se repitió con la TCN. ¿Con la TCN, 60 pasos es mejor que 30?

### 1.2 Brazos (todo lo demás igual a T1)

| Brazo | Pasos | Origen |
|---|---:|---|
| **P30** | 30 | T1 de la confirmación, **reutilizado** si se reproduce (§1.4) |
| **P60** | 60 | **nuevo**: 15 corridas (5 pliegues × semillas 2026, 2027, 2028) |

- **Arquitectura:** `CONFIG_T` de `src/arquitecturas_literatura.py` sin ningún cambio (una sola configuración,
  fijada antes): 4 bloques residuales con dilataciones 1-2-4-8, 2 convoluciones por bloque, 64 filtros, k = 3,
  padding «same», dropout 0,2 por bloque y 0,3 final, GlobalAveragePooling1D, softmax de 50. Se construye con
  `arquitecturas_literatura.construir_tcn(60, 252, 50)`. **156.402 parámetros en los dos brazos**: con GAP, la
  longitud no cambia el tamaño de la red.
- **Campo receptivo:** 1 + 2·(k − 1)·(1 + 2 + 4 + 8) = **61 pasos** (cada salida ve ±30 pasos). Con 30 pasos,
  cada una de las 30 posiciones ve la ventana completa. Con 60, solo las 2 posiciones centrales ven los 60 pasos;
  las de los bordes ven 31. GAP promedia todas. No se añade un bloque ni se cambia la dilatación para
  «compensar»: eso sería otra arquitectura y la pregunta es solo la longitud.
- **Entrada:** `mano_ausente_variantes.remuestrear_todas_con_regla(X_padded, lengths, "imputacion", 60)` sobre
  `data/processed_holistic_mascara_rostro`, representación «manos» (126 posiciones), y después de la aumentación
  `add_velocity` (126 velocidades) → 60 × 252. Consecuencia declarada: con 60 pasos la velocidad es la diferencia
  entre pasos dos veces más cercanos, de modo que su magnitud es aproximadamente la mitad que con 30; no se
  reescala (sería cambiar la entrada). Con 60 pasos se estiran por interpolación los videos de menos de 60
  cuadros (261 de 1000 según N-58; se recuenta en `verificar`).
- **Receta:** la de T1, sin cambios, a través de `experimento_arquitecturas.entrenar_corrida` (constructor «T»):
  Adam lr 1e-3 con CosineDecay, *label smoothing* 0,1, lote 64, máx. 200 épocas, paciencia 25 sobre
  `val_accuracy`, checkpoint de mejor `val_accuracy`, aumentación ×10 (`train_loso.aumentar` con las
  transformaciones de `refined_augmentation`), CPU.
- **Particiones:** `data/partitions/loso_5_pliegues.json` (D-06), validación interna de 2 por seña (100 muestras
  por pliegue). Solo se leen.

### 1.3 Criterio de decisión (el mismo tipo de regla que el factorial de octubre, N-63)

**Solo deciden métricas de validación LOSO. La prueba LOSO (top-1, top-3, F1) y el señante 0000 se reportan
únicamente como descripción y no deciden nada.**

Notación: `val(V, s, k)` es el `val_accuracy` máximo (el del checkpoint) del brazo V, semilla s, pliegue k;
`loss(V, s, k)` es la `val_loss` de ese checkpoint (entropía cruzada con *label smoothing* 0,1, la que guarda
`entrenar_corrida`). `val(V, k)` y `loss(V, k)` son la media de las 3 semillas. Las diferencias por pliegue se
redondean a 6 decimales y toda comparación con un umbral usa una tolerancia de 1e-9 (como en la confirmación y el
factorial; funciones de `experimento_confirmacion_tcn` y `experimento_factorial_tcn`).

1. **Regla 1: top-1 de validación con margen.**
   - **Margen** = máx(σ, 0,01). σ = DE entre semillas agrupada sobre P30 y P60: para cada brazo, la DE
     (ddof = 1) entre las 3 semillas de la validación media de 5 pliegues; luego, la raíz de la media de las
     dos varianzas.
   - **A supera a B** si la media de Δk = val(A, k) − val(B, k) es mayor que el margen **y** Δk > 0 en al menos
     4 de los 5 pliegues (Δk = 0 no cuenta como mejora).
   - Se evalúan los dos sentidos. Si P60 supera a P30, **se recomienda 60**. Si P30 supera a P60, **se mantiene
     30**. Si ninguno supera al otro, se pasa a la regla 2.
2. **Regla 2: desempate con la `val_loss`** (menor es mejor), como el paso 2 del factorial.
   - σ_loss = DE entre semillas de la `val_loss` media de 5 pliegues, agrupada sobre P30 y P60 igual que σ (sin
     piso).
   - **A supera a B en pérdida** si dk = loss(B, k) − loss(A, k) > 0 en al menos 4 de los 5 pliegues **y** la
     media de dk es mayor que σ_loss.
   - Si P60 supera a P30 en pérdida, **se recomienda 60**. Si P30 supera a P60, **se mantiene 30**. Si ninguno,
     regla 3.
3. **Regla 3: simplicidad y costo → 30.** 30 es la longitud vigente del modelo final y del prototipo (umbral,
   reporte 4.e, latencia y análisis de errores se hicieron con 30), cuesta la mitad de cómputo convolucional por
   ventana y no estira ningún video del corpus. Ante un empate en validación, se mantiene 30 sin condiciones.
4. **Descriptivo, sin valor de decisión:** prueba top-1, top-3 y F1 macro (media de 3 semillas, por pliegue y por
   semilla); el señante 0000 (pliegue 0) aparte; McNemar exacto sobre las 997 predicciones de prueba agregadas,
   con cada semilla y con el voto de las 3 semillas (mayoría; si las tres discrepan, gana la semilla más
   confiada); signos y Wilcoxon por pliegue en validación, `val_loss`, prueba top-1, top-3 y F1 (con 5 pliegues el
   p bilateral mínimo es 2/2⁵ = 0,0625, así que ninguna llega a 0,05); épocas y tiempo de entrenamiento.
5. **Costo (descriptivo, después de las corridas y sin entrenamientos activos):** parámetros, tamaño en disco,
   latencia de la red con lote de 1 (`model.predict`, como `translator.py`, y llamada directa `model(x)`) y costo de
   armar la entrada de una ventana (imputación → remuestreo → velocidades) con 30 y con 60 pasos.
6. **Condición previa para reutilizar P30.** Antes de lanzar P60 se reentrena T1, semilla 2026, pliegue 0, con el
   código de este experimento (`src/experimento_pasos_tcn.py`, con L = 30), en una carpeta aparte. P30 se reutiliza
   si `predicciones.json` sale idéntico byte a byte o, en su defecto, dentro de la tolerancia que usó la
   confirmación (`experimento_confirmacion_tcn.comparar_corridas`: mismas predicciones y top-3, mismas épocas,
   misma validación y diferencia máxima de probabilidades de prueba < 1e-5). Si no, se entrenan también las 15
   corridas de P30 con este script y se usan esas.
7. **Plan de ejecución.** Tres procesos en paralelo, uno por semilla, cada uno con los pliegues 0 → 4, con
   `nohup` y registro en `models/pasos_tcn_2026-11-23/`. Si no terminaran, el criterio se aplica con las semillas
   que tengan los 5 pliegues en los dos brazos (mínimo 2) y se declara.
8. **Estimación de duración** (con los registros de T1): T1 tardó de 85 a 192 s por corrida (media 123 s; 43 a 85
   épocas, media 55,7; 2,15 s por época) con 3 procesos en paralelo. La TCN recorre el doble de pasos, así que se
   espera ≈ 2 veces el costo por época (≈ 4,3 s) y ≈ 240 s por corrida con épocas parecidas: ≈ 20 min por proceso
   (5 corridas) y **≈ 20-30 min de reloj** para las 15 corridas. La reproducción previa (§1.6) añade ≈ 2 min.

**Lo que no se hace:** no se prueban otras longitudes (45, 90…), ni otra arquitectura para 60 pasos, ni otra
receta. No se cambia el modelo final, el umbral ni `translator.py`.

---

*Todo lo que sigue se escribió después de la corrida (23-nov-2026, a partir de las 00:18). §1 no se tocó: su
copia `models/pasos_tcn_2026-11-23/criterio_prerregistrado.md` conserva el SHA-256
`2a500098e039c715cf6eafd20577a15e6e5bbf1aa9bd2b88a199257385af9401`, y `tests/test_experimento_pasos_tcn.py` comprueba
que §1 de este informe es idéntica a esa copia.*

## 2. Ejecución

- **Verificación** (22-nov, 23:55-23:57; `verificacion.json`):
  - las 15 corridas de T1 tienen `CONFIG_T`, la receta común, 30 pasos, imputación, 252 rasgos y 156.402
    parámetros;
  - la `X_seq` de 30 pasos de este script es, bit a bit, la de `experimento_confirmacion_tcn.preparar` (la que
    entrenó T1); la de 60 pasos sale de `remuestrear_todas_con_regla(·, "imputacion", 60)`;
  - 261 de los 1000 videos tienen menos de 60 cuadros y se estiran con 60 pasos; ninguno tiene menos de 30;
  - 156.402 parámetros con 30 y con 60 pasos; campo receptivo de 61 pasos: ventana completa en 30 de 30
    posiciones y en 2 de 60.
- **Condición previa (§1.3.6): cumplida, byte a byte.** T1 con la semilla 2026 y el pliegue 0, reentrenado con este
  script en `models/pasos_tcn_2026-11-23/reproduccion_T1/`: `predicciones.json` idéntico byte a byte,
  probabilidades de validación y de prueba con diferencia máxima 0, historial idéntico, 43 épocas, validación 0,980,
  `val_loss` 0,870123 y prueba 0,540 iguales (58,5 s, solo en la máquina). **P30 se reutiliza de T1.**
- **Prueba de humo** (23:58, en el *carpeta_temporal*): P60, semilla 2026, pliegue 0, 3 épocas. El modelo se guarda y
  se recarga con entrada 60 × 252 y 156.402 parámetros; la procedencia y `resultado.json` dicen `seq_length` 60
  (`humo.json`). Sus cifras no se usan.
- **Corrida:** 3 procesos en paralelo (uno por semilla) lanzados con `nohup` a las **23:58:45** del 22-nov; el
  último terminó a las **00:17:02** del 23-nov. **Duración real: 18 min 17 s** para las 15 corridas (estimación
  previa: 20-30 min). Sin errores ni recortes; el criterio se aplica con las 3 semillas completas.
- **Coherencia:** en las 15 corridas nuevas el `val_accuracy` máximo coincide con la exactitud del checkpoint
  recargado, y la `val_loss` guardada coincide con su recálculo desde `probabilidades.npz` (|dif| < 1e-4).
- **Latencia:** medida a las 00:18, sin ningún entrenamiento activo.

## 3. Resultados

Media de 3 semillas × 5 pliegues. «DE» es entre los 5 pliegues (medias de semillas). La validación decide; la
prueba y el señante 0000 solo describen.

| Brazo | Parámetros | **Val top-1** (DE) | **`val_loss`** (DE sem.) | Prueba top-1 (DE) | Top-3 | F1 macro | Voto 3 semillas | Señante 0000: val / prueba | Épocas (mejor) | s por corrida |
|---|---:|---:|---:|---:|---:|---:|---:|:-:|---:|---:|
| **P30** (T1, vigente) | 156.402 | 0,949 (0,017) | 0,978 (0,023) | 0,677 (0,127) | 0,847 | 0,647 | 0,716 | 0,960 / 0,488 | 55,8 (30,8) | 123 |
| **P60** | 156.402 | 0,953 (0,019) | 0,968 (0,029) | 0,690 (0,113) | 0,863 | 0,660 | 0,725 | 0,973 / 0,505 | 59,9 (34,9) | 215 |

**Por pliegue** (media de 3 semillas; pliegues 0 / 1 / 2 / 3 / 4):

| Brazo | Val top-1 | `val_loss` | Prueba top-1 | Prueba top-3 | Prueba F1 |
|---|---|---|---|---|---|
| P30 | 0,960 / 0,937 / 0,937 / 0,973 / 0,940 | 0,897 / 1,012 / 1,046 / 0,920 / 1,013 | 0,488 / 0,611 / 0,721 / 0,770 / 0,795 | 0,660 / 0,807 / 0,889 / 0,935 / 0,942 | 0,469 / 0,565 / 0,685 / 0,745 / 0,770 |
| P60 | 0,973 / 0,943 / 0,943 / 0,973 / 0,933 | 0,890 / 0,996 / 1,021 / 0,887 / 1,044 | 0,505 / 0,672 / 0,717 / 0,773 / 0,785 | 0,682 / 0,858 / 0,897 / 0,932 / 0,948 | 0,488 / 0,642 / 0,678 / 0,741 / 0,754 |

**Por semilla** (media de 5 pliegues; 2026 / 2027 / 2028):

| Brazo | Val top-1 | `val_loss` | Prueba top-1 | F1 |
|---|---|---|---|---|
| P30 | 0,950 / 0,954 / 0,944 | 0,975 / 0,956 / 1,001 | 0,688 / 0,658 / 0,685 | 0,658 / 0,630 / 0,653 |
| P60 | 0,960 / 0,946 / 0,954 | 0,945 / 1,000 / 0,957 | 0,707 / 0,690 / 0,674 | 0,677 / 0,660 / 0,643 |

## 4. Aplicación del criterio prerregistrado

- **σ y margen.** Varianzas entre semillas de la validación: P30 2,5·10⁻⁵ y P60 4,9·10⁻⁵; σ agrupada 0,0061 <
  0,01, así que **margen = 0,010**. σ_loss agrupada = **0,0261**.
- **Regla 1 (top-1 de validación).** Δk = P60 − P30 = +0,013 / +0,007 / +0,007 / 0,000 / −0,007; media
  **+0,004** (menor que el margen) y mejor en **3 de 5** pliegues (menos de 4). P60 no supera a P30, y P30 tampoco a
  P60 (media −0,004, 1 de 5). **Empate.**
- **Regla 2 (`val_loss`).** Reducción con 60 por pliegue: +0,008 / +0,016 / +0,025 / +0,033 / −0,031; menor en
  **4 de 5** pliegues, pero la reducción media (**0,0100**) no supera σ_loss (0,0261). P60 no supera a P30 en
  pérdida; P30 tampoco a P60. **Empate.**
- **Regla 3 (simplicidad y costo): se mantiene 30.**

**Decisión según el criterio: P30. El modelo final conserva 30 pasos.**

## 5. Comparación pareada y McNemar (descriptivas)

**Signos y Wilcoxon por pliegue (P60 − P30, n = 5; p bilateral mínimo posible 0,0625).** Formato: pliegues a
favor de 60 / en contra / empates; p de signos; p de Wilcoxon. En `val_loss`, «a favor» es menor pérdida.

| Métrica | Δ por pliegue | Resultado |
|---|---|---|
| Val top-1 | +0,013 / +0,007 / +0,007 / 0 / −0,007 | 3 / 1 / 1; 0,625; 0,44 |
| `val_loss` (reducción) | +0,008 / +0,016 / +0,025 / +0,033 / −0,031 | 4 / 1 / 0; 0,375; 0,44 |
| Prueba top-1 | +0,017 / +0,060 / −0,003 / +0,003 / −0,010 | 3 / 2 / 0; 1,0; 0,63 |
| Prueba top-3 | +0,022 / +0,050 / +0,008 / −0,003 / +0,007 | 4 / 1 / 0; 0,375; 0,125 |
| Prueba F1 | +0,019 / +0,077 / −0,008 / −0,004 / −0,016 | 2 / 3 / 0; 1,0; 0,81 |

**McNemar exacto sobre las 997 predicciones de prueba agregadas (P60 frente a P30).** Formato: aciertos P60 /
P30; aciertos exclusivos de P60 / de P30; p bilateral.

| Semilla 2026 | 2027 | 2028 | Voto de las 3 semillas |
|---|---|---|---|
| 705 / 686; 97 / 78; p = 0,17 | 688 / 656; 101 / 69; p = 0,017 | 672 / 683; 84 / 95; p = 0,45 | 723 / 714; 74 / 65; **p = 0,50** |

La ventaja de 60 en prueba (+1,3 puntos de top-1) depende de la semilla: es significativa por McNemar solo con
2027, se invierte con 2028 y desaparece en el voto de las 3 semillas. Casi toda viene del pliegue 1 (+6,0 puntos).
McNemar trata las 997 predicciones como independientes, pero están agrupadas en 5 señantes.

## 6. Señante 0000 (pliegue 0), aparte

| Brazo | Val | `val_loss` | Prueba top-1: media (2026 / 2027 / 2028) | Top-3 | F1 |
|---|---:|---:|---|---:|---:|
| P30 | 0,960 | 0,897 | 0,488 (0,540 / 0,390 / 0,535) | 0,660 | 0,469 |
| P60 | 0,973 | 0,890 | 0,505 (0,545 / 0,515 / 0,455) | 0,682 | 0,488 |

60 pasos no empeora al señante 0000 (+1,7 puntos de media), pero la diferencia está dentro de la dispersión entre
semillas (0,390 a 0,545) y cambia de signo según la semilla.

## 7. Costo (sin entrenamientos activos, 23-nov 00:18)

Red: mediana de 100 llamadas por modelo, media de 3 modelos (semilla 2026, pliegues 0-2), en orden intercalado.
Entrada: *processed* → imputación → L pasos → `add_velocity`, mediana de 5 videos.

| Brazo | Parámetros | MB en disco | ms `predict` lote 1 | ms `model(x)` | ms entrada | s por época (3 procesos) | s por corrida |
|---|---:|---:|---:|---:|---:|---:|---:|
| P30 | 156.402 | 0,738 | 13,19 | 7,95 | 0,28 | 2,16 | 123 |
| P60 | 156.402 | 0,738 | 13,01 | 8,19 | 0,27 | 3,52 | 215 |

- En el prototipo, el costo de 60 pasos es **despreciable**: la red es tan pequeña que `predict` con lote de 1 está
  dominado por el costo fijo de la llamada (13 ms en los dos), y la llamada directa sube solo 0,24 ms. Armar la
  entrada cuesta lo mismo: la imputación sobre los cuadros originales domina y el remuestreo a 30 o 60 pasos es
  marginal.
- El costo real está en **entrenar**: 1,6 veces más por época y 1,75 veces más por corrida (más épocas: 59,9 frente
  a 55,8).
- La regla 3 no dependía de una medición de costo (§1.3.3); estas cifras son descriptivas.

## 8. Decisión

**Según el criterio prerregistrado (§1.3), se mantiene 30 pasos.** 60 no supera a 30 en validación por más del
margen (+0,4 puntos, 3 de 5 pliegues) ni en `val_loss` por más de su variación entre semillas (−0,010 frente a
σ_loss 0,026, aunque sí en 4 de 5 pliegues). Con empate en validación, la regla de simplicidad y costo favorece la
longitud vigente.

Lectura descriptiva (no cambia la decisión):

1. **Con la TCN, la ventaja de 60 en validación es menor que con la red de partida.** En N-58 (CNN + BiLSTM), 60
   superaba a 30 por +1,6 puntos en 5 de 5 pliegues; aquí, por +0,4 puntos en 3 de 5. La TCN ya está cerca del techo
   de la validación (~0,95, donde 1 muestra = 0,01).
2. **En prueba, 60 va ligeramente por delante** (+1,3 puntos de top-1, +1,7 de top-3, +1,4 de F1), pero dentro de
   la variación entre semillas (DE 0,017), concentrado en el pliegue 1 y sin significancia en el voto de las 3
   semillas (McNemar p = 0,50). Es el mismo patrón de N-58: una mejora pequeña en validación que no se traslada con
   claridad a señantes nuevos.
3. **El anteproyecto queda respondido con evidencia**: la diferencia frente a los 60 pasos que fijaba se justifica
   con este experimento y con N-58 (con dos arquitecturas, 60 no gana por la regla con la TCN y su ventaja en prueba
   no es consistente). Mantener 30 evita reentrenar el modelo final y recalibrar el umbral (N-74), el reporte 4.e,
   la latencia y el análisis de errores, todos hechos con 30.

Este informe no cambia el modelo final, el umbral ni `translator.py`.

## 9. Cautelas

1. **Solo 5 señantes.** Ninguna prueba por pliegue puede bajar de p = 0,0625; McNemar ignora la agrupación por
   señante.
2. **La validación está casi saturada** y la regla 2 quedó cerca: 4 de 5 pliegues a favor de 60 en `val_loss`, con
   una reducción media de 0,010 frente a σ_loss = 0,026. Con más semillas o menos dispersión, la `val_loss` podría
   separar los brazos; con estas 3 semillas no lo hace.
3. **60 pasos estira 261 de los 1000 videos del corpus** (26 %) por interpolación lineal; en vivo, la ventana de
   2,5 s a ≈ 30 fps tiene unos 75 fotogramas y no estiraría. El efecto de esa diferencia entre corpus y prototipo no
   se midió.
4. **El campo receptivo (61) apenas cubre 60 pasos** (solo 2 posiciones ven la ventana entera). No se probó una TCN
   con más dilatación para 60: sería otra arquitectura (§1.2).
5. **Velocidades a la mitad de escala con 60 pasos** (diferencias entre pasos más cercanos); no se reescalaron.
6. **P30 reutilizado.** Se verificó con la reproducción byte a byte de la semilla 2026, pliegue 0 (§2); los otros
   14 pliegues no se reprodujeron, pero el entrenamiento es determinista en este equipo.
7. **Tiempos de entrenamiento:** T1 se entrenó con 3 procesos en paralelo que alternaban T1 y T2; P60, con 3
   procesos de P60. Las cargas no son idénticas, así que la razón de 1,6 por época es aproximada.

## 10. Explicación simple

El sistema convierte cada seña en una «película» de un número fijo de fotos antes de reconocerla. El anteproyecto
decía 60 fotos y el modelo final usa 30. Probamos el modelo final (la red TCN) con 60 fotos, entrenándolo 15 veces, y
lo comparamos con las 15 versiones que ya existían con 30. En las pruebas de práctica, que deciden, los dos
aciertan casi igual (95 de cada 100). Con personas nuevas, 60 fotos acierta un poco más (69 % frente a 68 %), pero
esa diferencia cambia según la repetición y puede ser azar. La regla fijada antes de empezar dice que, si empatan,
se queda la opción vigente y más sencilla: **30 fotos**. Usar 60 casi no haría más lento el prototipo, pero
obligaría a rehacer el modelo y el umbral sin una ganancia demostrada.

## 11. Archivos y cómo reproducir

- **Script nuevo:** `src/experimento_pasos_tcn.py` (verificar, entrenar, inferencia y resumen). Importa sin
  modificarlos `experimento_arquitecturas`, `experimento_confirmacion_tcn`, `experimento_factorial_tcn`,
  `arquitecturas_literatura`, `mano_ausente_variantes` y `train_loso`.
- **Pruebas:** `tests/test_experimento_pasos_tcn.py` (20 pruebas rápidas: brazos, campo receptivo, entrada de 60
  pasos, condición previa, las tres reglas del criterio y la integridad de §1).
- **Corridas:** `models/pasos_tcn_2026-11-23/P60_imputacion_252/semilla_S/pliegue_K/` (modelo, procedencia con
  `seq_length` 60, `resultado.json`, `predicciones.json`, `probabilidades.npz`, `historial.json`); además
  `criterio_prerregistrado.md` y `.sha256`, `verificacion.json` y `.log`, `reproduccion_T1/`, `humo.json`,
  `corrida_semilla_*.log`, `pids.txt`, `inferencia.json` y `.log`, `resultados_pasos_tcn.json` y `resumen.txt`.
- **Datos crudos del informe:** `docs/EXPERIMENTO_PASOS_TCN.json`.
- **Solo lectura:** P30 = `models/confirmacion_tcn_2026-10-03/T1_imputacion_252/`.

```
python src/experimento_pasos_tcn.py verificar
python src/experimento_pasos_tcn.py entrenar --semillas 2026      # una semilla por proceso (2026, 2027, 2028)
python src/experimento_pasos_tcn.py inferencia
python src/experimento_pasos_tcn.py resumen --docs
```
