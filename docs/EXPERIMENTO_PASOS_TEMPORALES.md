# Experimento: número de pasos temporales (`seq_length`)

Estado: **criterio escrito antes de correr nada** (30-sep-2026, 19:32, hora de Bogotá, antes de lanzar el primer entrenamiento). Las secciones
de resultados se añaden después de la corrida y no cambian este criterio.

## 1. Pregunta

Cada seña se remuestrea a un número fijo de pasos antes de entrar a la red (`resample_sequence`).
Hoy son 30. El valor se heredó de `prepare_dataset.py` (marzo de 2026) y coincide con la referencia
r41 (Rivera-Vásquez y González-Serrano 2025), pero nunca se comparó con otros. ¿Hay algún valor que
funcione mejor? ¿Hay evidencia para cambiar 30?

## 2. Criterio prerregistrado (fijado ANTES de ver resultados)

1. **Qué decide.** El número de pasos se elige SOLO por la exactitud de **validación** del LOSO de
   5 pliegues: `mejor_val_accuracy` de cada pliegue (el checkpoint de mejor val_accuracy sobre las
   100 muestras de validación, 2 por seña, de los cuatro señantes vistos; D-06). Es la misma regla
   que D-07 y D-08. El top-1 de **prueba** (el señante no visto) se reporta para todos los valores,
   pero solo como descripción: elegir con él sería selección sobre el conjunto de prueba
   (hallazgo B de marzo).
2. **Unidad de comparación.** Para cada valor L y cada pliegue k, la validación del pliegue es la
   media entre las semillas de entrenamiento corridas. La validación media de L es la media de
   esos 5 valores.
3. **Variación entre semillas (σ_semilla).** Para cada valor L se calcula la media de validación de
   los 5 pliegues con cada semilla por separado; σ_semilla es la desviación estándar muestral
   (ddof = 1) de esas medias entre semillas, agrupada (raíz de la media de las varianzas) sobre
   todos los valores L. Se agrupa porque con 2 o 3 semillas la desviación de un solo valor es muy
   inestable.
4. **Regla para dejar 30.** 30 se mantiene salvo que otro valor L cumpla LAS DOS condiciones:
   - (a) su validación media supera la de 30 por un margen **mayor que max(1 punto, σ_semilla)**
     (1 punto = 1 muestra de 100 por pliegue; es el empate que ya usaron D-07 y D-08), y
   - (b) su validación supera a la de 30 en **al menos 4 de los 5 pliegues** (un empate en un
     pliegue no cuenta como mejora).
5. **Si varios valores cumplen la regla**, gana el de mayor validación media; si dos de ellos quedan
   a menos de 1 punto entre sí, gana el de **menos pasos** (menos cómputo en el prototipo).
6. **Comparación pareada por pliegue** (complementaria, no decide sola): para cada L frente a 30, las
   5 diferencias de validación por pliegue, con prueba de signos (binomial exacta) y Wilcoxon de
   rangos con signo (exacta). Límite que se declara de antemano: con 5 pares el menor p bilateral
   posible es 2/32 = 0,0625 en las dos pruebas, así que **ninguna puede dar p < 0,05 bilateral**;
   sirven para describir la consistencia, no como prueba de significancia. Los empates (diferencia
   0) se descartan en la prueba de signos y se tratan con el método de scipy en Wilcoxon.
7. **Qué no decide.** El top-1, top-3 y F1 de prueba, las épocas y el tiempo de inferencia se
   reportan para todos los valores. El tiempo de inferencia solo podría inclinar la balanza en un
   empate según el punto 5.

## 3. Configuración (la canónica de N-30 / D-08, salvo la longitud)

- Datos: `data/processed_holistic_mascara_rostro` (HolisticLandmarker, normalización por hombros,
  regla «mascara» de D-07), representación «manos» (126 rasgos, D-08). Es el processed y la
  representación con que se obtuvo la referencia 0,580 (`models/loso_2026-09-15_holistic_mascara_manos/`).
- Partición: `data/partitions/loso_5_pliegues.json` (semilla 2026, D-06: 997 muestras, validación
  2 por seña). Solo se lee.
- Entrenamiento: `train_loso.entrenar_y_evaluar` importada tal cual: CNN+BiLSTM+atención
  (`build_model`), velocidades (`add_velocity`), aumentación ×10 (las transformaciones de
  `refined_augmentation`), 200 épocas máx., batch 64, paciencia 25 sobre val_accuracy, dropout 0,3,
  Adam con CosineDecay desde 0,001, label smoothing 0,1, en CPU.
- Lo único que cambia: `resample_sequence(…, L)` con L ∈ {15, 20, 30, 45, 60}.
- Semillas de entrenamiento: 2026 (la canónica) y las que quepan en el presupuesto de tiempo
  (ver §4).
- Advertencia sobre 45 y 60: el video más corto tiene 40 cuadros (media 67,3, máximo 135, a 24 fps).
  Con 45 pasos 7 videos se estiran (interpolación lineal entre cuadros); con 60 pasos se estiran
  261 de 1000 (26 %). Estirar no añade información: solo inserta puntos intermedios.

## 4. Presupuesto de tiempo y semillas (decidido antes de ver resultados)

- Medición previa: un pliegue (pliegue 0, L = 30, semilla 2026) tardó **165 s** de principio a fin
  (67 épocas). Reprodujo exactamente la referencia de N-30 para ese pliegue: top-1 0,460, F1
  0,4207, val 0,95, 67 épocas con la mejor en la 42 (la época 1 sale idéntica hasta el cuarto
  decimal de la pérdida), así que el entrenamiento es determinista en este equipo.
- Estimación, suponiendo que el costo crece aproximadamente en proporción a L (BiLSTM) y que las
  épocas se parecen: ≈ 90 / 115 / 165 / 240 / 320 s por pliegue para L = 15 / 20 / 30 / 45 / 60,
  ≈ 15,5 min por semilla y pliegue, ≈ **78 min por semilla** para los 5 valores × 5 pliegues.
- Con 3 semillas son ≈ 4 h, por debajo del techo de ~10 h. **Se corren 3 semillas (2026, 2027,
  2028) y los 5 valores; no se recorta nada.** Se corren en serie, en un solo proceso, para que los
  tiempos sean comparables y el resultado de la semilla 2026 siga siendo reproducible.
- Orden: semilla → L (30, 20, 45, 15, 60) → pliegue, para que, si hubiera que cortar, cada semilla
  quede completa en los valores prioritarios.
- El tiempo de inferencia se mide aparte al final, con la máquina sin entrenamientos.

---

*Todo lo que sigue se escribió después de la corrida (30-sep-2026, 19:33 a 23:28). Las secciones
1 a 4 no se tocaron, salvo la hora de la línea de estado, que se corrigió a la real (19:32).*

## 5. Qué se corrió

- 75 entrenamientos: 5 valores de L × 5 pliegues × 3 semillas (2026, 2027, 2028), sin recortes ni
  fallos. De 19:33 a 23:25, en serie y en un solo proceso. Otros dos experimentos del proyecto
  esperaron a que este terminara, así que ningún otro entrenamiento compartió la máquina.
- **Reproducción de la referencia.** Con L = 30 y la semilla 2026, los 5 pliegues reproducen N-30
  número por número: top-1 0,460 / 0,533 / 0,606 / 0,615 / 0,685, F1 0,4207 / 0,4629 / 0,5728 /
  0,5852 / 0,6455, validación 0,95 / 0,90 / 0,91 / 0,96 / 0,94 y las mismas épocas (67 / 46 / 42 /
  61 / 62). La comparación usa por tanto exactamente el código y la configuración de la referencia.
- Tiempo de inferencia: medido a las 23:26-23:28, con la máquina sin entrenamientos. Para cada uno
  de los 75 modelos, `model.predict` de un lote de 1, como lo hace `translator.py` (10 llamadas de
  calentamiento y 50 medidas, mediana), y el costo por muestra en un lote de 200. Es solo la red:
  no incluye MediaPipe ni el remuestreo.

## 6. Resultados

Cada cifra es la media de las 3 semillas en cada pliegue; «±» es la desviación estándar muestral
entre los 5 pliegues. La validación decide; la prueba es solo descripción.

| Pasos (L) | **Validación media** | Validación por pliegue (0 / 1 / 2 / 3 / 4) | Prueba top-1 LOSO | Prueba top-3 | Prueba F1 macro | Épocas (mejor) | Inferencia, lote de 1 (ms) | Por muestra en lote de 200 (ms) | Entrenamiento por pliegue (s) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0,885 ± 0,040 | 0,877 / 0,833 / 0,907 / 0,940 / 0,870 | 0,526 ± 0,133 | 0,708 | 0,482 ± 0,131 | 63,3 (38,3) | 12,1 | 0,16 | 105 |
| 20 | 0,898 ± 0,030 | 0,900 / 0,860 / 0,897 / 0,943 / 0,890 | 0,569 ± 0,129 | 0,746 | 0,522 ± 0,134 | 58,6 (33,6) | 12,3 | 0,18 | 122 |
| **30** (actual) | 0,920 ± 0,020 | 0,923 / 0,893 / 0,917 / 0,950 / 0,917 | 0,589 ± 0,094 | 0,764 | 0,544 ± 0,103 | 54,5 (29,5) | 12,9 | 0,22 | 150 |
| 45 | 0,931 ± 0,031 | 0,933 / 0,893 / 0,937 / 0,977 / 0,917 | 0,599 ± 0,110 | 0,769 | 0,554 ± 0,112 | 61,0 (36,0) | 13,8 | 0,29 | 235 |
| 60 | **0,936 ± 0,016** | 0,937 / 0,917 / 0,937 / 0,960 / 0,930 | 0,598 ± 0,104 | 0,768 | 0,559 ± 0,102 | 59,9 (34,9) | 14,4 | 0,35 | 306 |

Prueba top-1 por pliegue (media de 3 semillas):

| L | pliegue 0 | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|---:|
| 15 | 0,315 | 0,486 | 0,566 | 0,650 | 0,615 |
| 20 | 0,373 | 0,518 | 0,591 | 0,672 | 0,690 |
| 30 | 0,452 | 0,536 | 0,631 | 0,640 | 0,687 |
| 45 | 0,413 | 0,585 | 0,655 | 0,677 | 0,665 |
| 60 | 0,438 | 0,564 | 0,614 | 0,705 | 0,668 |

Por semilla (media de los 5 pliegues), validación / prueba top-1:

| L | semilla 2026 | semilla 2027 | semilla 2028 |
|---:|---|---|---|
| 15 | 0,888 / 0,521 | 0,880 / 0,521 | 0,888 / 0,538 |
| 20 | 0,898 / 0,575 | 0,896 / 0,583 | 0,900 / 0,549 |
| 30 | 0,932 / 0,580 | 0,918 / 0,592 | 0,910 / 0,596 |
| 45 | 0,938 / 0,610 | 0,930 / 0,597 | 0,926 / 0,590 |
| 60 | 0,948 / 0,601 | 0,936 / 0,611 | 0,924 / 0,583 |

Las tres redes tienen los mismos 714.291 parámetros: L no cambia el tamaño del modelo, solo cuántos
pasos recorre.

**Variación entre semillas.** σ_semilla agrupada (criterio §2.3) = **0,0081** (0,8 puntos). Por
valor: 0,005 / 0,002 / 0,011 / 0,006 / 0,012 para L = 15 / 20 / 30 / 45 / 60. Como es menor que
1 punto, el margen exigido es **1 punto**. En prueba la variación entre semillas es de 0,8 a 1,8
puntos de top-1.

## 7. Aplicación del criterio y comparación pareada (cada L frente a 30, por pliegue)

| L | Δ validación media | Pliegues mejor / igual / peor | (a) Δ > 1 punto | (b) ≥ 4 de 5 | **Cumple** | Signos p (bilat.) | Wilcoxon p (bilat.) | Δ prueba top-1 (solo descriptivo) | Pliegues de prueba mejor / peor |
|---:|---:|---|:-:|:-:|:-:|---:|---:|---:|---|
| 15 | −3,5 puntos | 0 / 0 / 5 | no | no | no | 0,062 | 0,062 | −6,3 puntos | 1 / 4 |
| 20 | −2,2 puntos | 0 / 0 / 5 | no | no | no | 0,062 | 0,062 | −2,0 puntos | 2 / 3 |
| 45 | +1,1 puntos | 3 / 2 / 0 | sí | **no** | no | 0,25 | 0,19 | +1,0 punto | 3 / 2 |
| 60 | +1,6 puntos | 5 / 0 / 0 | sí | sí | **sí** | 0,062 | 0,062 | +0,9 puntos | 2 / 3 |

- **Solo 60 cumple la regla.** 45 supera el margen pero empata con 30 en los pliegues 1 y 4, así
  que mejora en 3 de 5, no en 4. Por el punto 5 del criterio no hay desempate que hacer: queda un
  solo candidato.
- 60 supera a 30 en validación en los 5 pliegues y también con cada semilla por separado (0,948
  frente a 0,932; 0,936 frente a 0,918; 0,924 frente a 0,910). Es una mejora pequeña pero muy
  consistente: 1,6 puntos, es decir, en promedio 1,6 de las 100 muestras de validación por pliegue.
- 15 y 20 son peores que 30 en todos los pliegues, en validación y casi siempre en prueba (el
  pliegue 0, el señante más difícil, pierde 14 puntos con 15 pasos). Acortar no conviene.
- Las pruebas pareadas no pueden dar p < 0,05 bilateral con 5 pliegues (§2.6). El 0,062 de 60 es
  el menor valor posible: los 5 pliegues van en la misma dirección. Esto describe consistencia, no
  significancia.
- **En prueba, la ventaja de 60 no se confirma.** Su top-1 LOSO medio es 0,598 frente a 0,589
  (+0,9 puntos, dentro de la variación entre semillas). Gana en 2 pliegues y pierde en 3, con
  p = 0,81 en Wilcoxon. 45 da prácticamente lo mismo (0,599). El F1 de prueba sube un poco con
  L (0,544 → 0,554 → 0,559) y el top-3 queda igual (≈ 0,77). Esto no decide (§2.1), pero pesa en
  la lectura.

## 8. Recomendación

1. **Según el criterio prerregistrado, el valor recomendado es 60 pasos.** Es el único que mejora
   la validación por más del margen (1,6 puntos frente a 1) y en al menos 4 de los 5 pliegues
   (5 de 5).
2. **La evidencia para cambiar existe, pero es débil.** La mejora está en la validación, que se
   mide con señantes que el modelo ya vio, y no se traslada de forma clara a la prueba con el
   señante no visto (+0,9 puntos, 2 de 5 pliegues). Lo que sí queda claro es que bajar de 30 empeora
   y que subir no empeora nada medible.
3. **Costo de 60 en el prototipo: despreciable.** La red tarda 14,4 ms por ventana frente a
   12,9 ms con 30 (+1,5 ms), en una cadena cuyo extractor gasta ≈ 11-15 ms por fotograma (N-21,
   N-40). `translator.py` toma L de la forma de entrada del modelo y la coteja con la procedencia,
   así que no habría que tocar código. El costo real está en entrenar: unas 2 veces más tiempo por
   modelo (306 s frente a 150 s por pliegue).
4. **Si se adopta 60, hay que rehacer lo que depende del modelo**, y conviene registrarlo en una
   decisión nueva antes de entrenar:
   - el modelo canónico del esquema aleatorio, con `train_final_100.py --seq-length 60`, que ya
     acepta la longitud;
   - el LOSO de referencia: la cifra reportable pasaría de 0,580 (semilla 2026) a la de 60 con
     esa misma semilla, que es 0,601 de top-1;
   - el umbral de confianza: 0,20 (D-09) y la propuesta anidada 0,55 (N-47) se calibraron con
     modelos de 30 pasos;
   - el análisis de errores (N-35) y las cifras citadas en README y DECISIONES.

   `train_loso.py` tiene `SEQ_LENGTH = 30` fijo; para el LOSO oficial habría que darle un
   `--seq-length` opcional cuyo valor por defecto sea 30. Este experimento no lo necesitó, porque
   `entrenar_y_evaluar` toma la longitud de los datos que recibe.
5. **Alternativa razonable**: quedarse con 30 y declarar este experimento como la justificación
   empírica del valor. Ningún valor mejora de forma clara la generalización a señantes nuevos, y
   así se evita recalibrar todo antes de las sesiones del Bloque 1. La regla escrita
   antes de correr apunta a 60; apartarse de ella exige decirlo y dar el motivo (el costo de
   recalibrar), no buscar otro criterio a posteriori.

Esta decisión le corresponde al estudiante con el director; este informe no la toma.

## 9. Limitaciones

- **5 pliegues = 5 señantes.** Ninguna prueba pareada puede llegar a p < 0,05 bilateral. Las
  diferencias entre valores vecinos (30, 45, 60) son de 1 a 2 puntos, del mismo orden que la
  resolución de la validación (100 muestras por pliegue, 1 muestra = 1 punto).
- **3 semillas.** Con 3 semillas, σ_semilla es una estimación gruesa. Además, las semillas solo
  cambian la inicialización y la aumentación; la partición es fija (semilla 2026), así que no
  miden la variación por qué muestras caen en validación.
- **La validación está sesgada.** Sale de los cuatro señantes vistos (limitación ya declarada en
  N-30), y aquí la ventaja de 60 sobre 30 en validación no se reproduce con claridad en prueba. Una validación anidada por
  señante (como N-47) sería más fiel, a cambio de 2 a 4 veces más cómputo.
- **Estirar no añade información.** Con 60 pasos, 261 de los 1.000 videos (26 %) tienen menos de
  60 cuadros y se rellenan por interpolación lineal; con 45 son 7. En vivo, la ventana de 2,5 s a
  ≈ 30 fps tiene unos 75 fotogramas, así que 60 no estiraría.
- **El umbral 0,20 (D-09) y el 0,55 (N-47) se calibraron con modelos de 30 pasos.** Las
  probabilidades de un modelo de 60 pasos pueden distribuirse distinto, así que esos umbrales no
  se pueden trasladar sin recalibrar.
- **La inferencia se midió sin MediaPipe** y en este equipo (Apple Silicon, CPU, TensorFlow 2.21).
  Las diferencias absolutas son de 1-2 ms y pueden cambiar en otro equipo.
- **Los valores intermedios no se probaron** (por ejemplo 40 o 50), ni valores mayores que 60.
  Como 45 y 60 quedan a medio punto entre sí en validación y empatan en prueba, no hay indicio de
  que valga la pena buscar más fino.

## 10. Explicación simple

Antes de reconocerla, el sistema convierte cada seña en una «película» con un número fijo de
fotos. Hasta ahora eran 30 fotos, un número heredado que nunca se había comparado con otros.
Probamos 15, 20, 30, 45 y 60, entrenando el modelo 75 veces. Con menos fotos (15 o 20) el sistema
se equivoca más: pierde detalle del movimiento. Con más fotos (45 o 60) acierta un poco más en las
pruebas de práctica, que usan a las mismas personas con las que aprendió. Con una persona nueva la
mejora es tan pequeña que no se distingue del azar. Usar 60 apenas hace más lento el sistema (1,5
milésimas de segundo más por seña). La regla que fijamos antes de empezar señala 60, pero la
ganancia real es pequeña, y cambiar obliga a recalcular otras piezas: es una decisión para tomar
con el director.

## 11. Archivos y cómo reproducir

- Script: `src/experimento_pasos.py` (nuevo; importa `train_loso.py` y `train_final_100.py` sin
  modificarlos).
- Salidas: `models/experimento_pasos_2026-09-30/`
  - `resultados_pasos.json`: datos crudos y agregados (75 corridas, resumen por valor, criterio,
    pruebas pareadas, reproducción de N-30);
  - `inferencia.json`: tiempos por modelo;
  - `resumen.txt`: la tabla impresa;
  - `pasos_LL/semilla_S/pliegue_K/`: modelo, procedencia (con la L real), métricas, historial,
    curvas, matriz de confusión y predicciones de cada corrida;
  - registros: `medicion_pliegue0_L30.log`, `corrida_entrenar.log` y `corrida_inferencia.log`.
- Reproducir:
  `python src/experimento_pasos.py entrenar --pasos 30,20,45,15,60 --semillas 2026,2027,2028`, luego
  `python src/experimento_pasos.py inferencia` y `python src/experimento_pasos.py resumen`.
  El entrenamiento es reanudable: salta lo que ya existe.
- Entorno: `python` (Python 3.12.13, TF 2.21.0,
  Keras 3.13.2, numpy 2.4.3, scikit-learn 1.8.0, scipy 1.17.1), el mismo con que se corrieron N-30
  y N-47.

### Incidencias

- `venv/bin/python` en la raíz del repositorio no existe: `venv/` es el resto del entorno Windows
  de marzo (README:21), y `~/venvs/tesis312` tampoco está. Se usó el venv con que se corrieron
  N-30 y N-47, que tiene las mismas versiones que la referencia.
- La primera medición del pliegue no arrancó porque `--folds 0` significa «los primeros 0
  pliegues» en `train_loso.parsear_folds`. Se relanzó con `--folds 0,`.
- Error de redondeo corregido en el resumen antes de redactar: las exactitudes vienen en float32
  y dos medias iguales (0,9167 frente a 0,9167, pliegue 4 de L = 45) diferían en 2·10⁻⁸. La
  primera versión del script contó esa diferencia como mejora: daba 4 de 5 pliegues para 45 y
  recomendaba 45 por desempate. Ahora se redondea a 6 decimales antes de comparar, y el empate
  cuenta como empate, como dice el criterio (§2.4 b). Con eso, 45 mejora en 3 de 5 y no cumple.
  El criterio no se cambió: se corrigió la implementación para que dijera lo que el criterio ya
  decía.
- Antes de la corrida completa se calculó un resumen provisional solo con la semilla 2026, en una
  copia en el directorio temporal, para probar el código. Ya entonces apuntaba a 60 (+1,6 puntos,
  4 de 5 pliegues). No se cambió nada del criterio ni de la corrida después de verlo.
