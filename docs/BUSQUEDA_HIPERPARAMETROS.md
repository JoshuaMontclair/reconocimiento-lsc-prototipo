# Búsqueda de hiperparámetros de la red canónica (LOSO, actividad 2.d)

Fecha: 30-sep-2026. Script: `src/busqueda_hiperparametros.py`. Salidas: `models/busqueda_hiperparametros/`.
Datos crudos: `docs/BUSQUEDA_HIPERPARAMETROS.json`.

Esta búsqueda **no cambia el modelo canónico ni el umbral 0,20**. Produce evidencia y una recomendación;
adoptarla es decisión del estudiante y del director y exigiría recalibrar el umbral.

## 1. Pregunta

¿Hay una combinación de hiperparámetros claramente mejor que la canónica para generalizar a personas nuevas
(LOSO)? Todos los hiperparámetros de la red y del entrenamiento canónicos (filtros, ventana y capas Conv1D,
unidades BiLSTM, densas, dropout, tasa de aprendizaje, lote y suavizado de etiquetas) vienen de la exploración
de marzo de 2026 y nunca se optimizaron. El anteproyecto (actividad 2.d) pide seleccionar el modelo final «por
comparación de las configuraciones entrenadas … sobre el conjunto de validación»; esta búsqueda cabe ahí.

Entrada fija: `data/processed_holistic_mascara_rostro`, representación «manos», regla «mascara», 30 pasos
remuestreados + velocidades → 30 × 252. Partición: `data/partitions/loso_5_pliegues.json` (D-06). Referencia
(N-30, semilla 2026): top-1 LOSO de prueba 0,580 ± 0,086 (0,460 / 0,533 / 0,606 / 0,615 / 0,685), F1 macro 0,537.

## 2. Criterio prerregistrado

Escrito el 30-sep-2026 hacia las 21:35, **antes de lanzar ninguna corrida real** (solo se hizo una prueba de
humo con 1 época y sin aumentación, que no informa nada sobre las configuraciones).

**Métrica de selección.** Exactitud top-1 de **validación** interna del checkpoint (`val_top1` = mejor
`val_accuracy` del historial, la misma métrica que decidió D-07 y D-08), en los 5 pliegues LOSO
(100 muestras por pliegue, 2 por seña de los cuatro señantes de entrenamiento). La prueba LOSO (señante no
visto: top-1, top-3, F1 macro) se calcula y se reporta **solo como descripción**; ningún paso del script la lee
para decidir (`seleccionar` y el criterio de `resumen` usan únicamente campos de validación y tamaño).

**Fase 1 → finalistas.** Entre las configuraciones que completan los 5 pliegues (sin corte temprano), las
3 de mayor val top-1 media de los 5 pliegues; empate: menor val_loss media; luego menos parámetros.
La canónica pasa a la fase 2 siempre, como control.

**Corte temprano (fase 1).** Tras los pliegues 0 y 1, si la val top-1 media de una configuración es más de
5 puntos inferior a la de la canónica en esos mismos pliegues de esta corrida, se detiene y no puede ser
finalista.

**Fase 2 → veredicto.** Canónica + 3 finalistas, semillas nuevas 2031, 2032 y 2033, 5 pliegues.
Para cada configuración *C* y pliegue *k*, val(*C*, *k*) = media sobre las semillas; Δ*k* = val(*C*, *k*) −
val(canónica, *k*).

- *σ_semilla*: para cada una de las 4 configuraciones, desviación estándar muestral (ddof = 1) entre las
  3 semillas de la val top-1 media de 5 pliegues; se agrupa como la raíz de la media de las 4 varianzas.
  Margen = max(σ_semilla, 0,01). El piso de 0,01 equivale a una muestra de validación por pliegue y evita un
  margen degenerado con solo 3 semillas. Se reporta además, como referencia, la desviación de una sola
  corrida (celda configuración × pliegue).
- Una configuración **reemplaza a la canónica** solo si cumple las tres condiciones:
  (a) media de Δ*k* > 0; (b) Δ*k* > 0 en **al menos 4 de 5** pliegues (un Δ*k* = 0 no cuenta como mejora);
  (c) media de Δ*k* > margen.
- Si varias cumplen: se toma la de mayor val media; las que queden a menos del margen de ella se consideran
  empatadas en la práctica y entre ellas se prefiere la de **menos parámetros** (latencia del prototipo en CPU).
- Si ninguna cumple, **se mantiene la canónica**. Una configuración más pequeña que solo empate con la
  canónica no la reemplaza por este criterio (cambiar exige recalibrar el umbral); si ocurre, se señala como
  observación para el estudiante y el director, no como recomendación.
- Pruebas pareadas sobre las 5 Δ*k* (signos, binomial exacta, y Wilcoxon exacta, ambas bilaterales): se
  reportan, pero **no deciden**. Con 5 pliegues el p mínimo alcanzable en ambas es 0,0625 (5 de 5 en el mismo
  sentido), así que ninguna puede llegar a 0,05.

**Sobreajuste a la validación.** Con 21 configuraciones y una validación de 100 muestras por pliegue, la mejor
de la fase 1 parecerá mejor de lo que es (selección del máximo de varias medidas ruidosas). Por eso la fase 2
usa semillas nuevas y el reporte muestra la **caída** de cada finalista entre la búsqueda (semilla 2026) y la
confirmación (2031–2033): val media y ventaja sobre la canónica en ambas fases.

## 3. Diseño, espacio, recortes y tiempos

**Espacio de búsqueda.**

| Hiperparámetro | Valores | Canónica |
|---|---|---|
| Filtros Conv1D | 64, 128 | 128 |
| Ventana Conv1D | 3, 5 | 3 |
| Capas Conv1D | 1, 2 | 2 |
| BiLSTM | (128, 64), (64, 32), (128,) | (128, 64) |
| Densas | (256, 128), (128, 64) | (256, 128) |
| Dropout *d* | 0,2, 0,3, 0,5 | 0,3 |
| lr inicial (CosineDecay, alpha 1e-6) | 5e-4, 1e-3, 2e-3 | 1e-3 |
| Lote | 32, 64 | 64 |
| Suavizado de etiquetas | 0, 0,1 | 0,1 |

El dropout conserva la estructura de la canónica: *d*/2 tras la convolución y tras la segunda densa, y *d* en
los demás puntos.

**Fijos.**
- Paciencia 25 sobre val_accuracy con restore_best_weights; 200 épocas máximas.
- Aumentación ×10; 30 pasos; velocidades (30 × 252); Adam.
- Partición `data/partitions/loso_5_pliegues.json`.

**Arquitectura parametrizable.** `construir_modelo(cfg)` en el script nuevo. Con los valores canónicos
reproduce `build_model` exactamente, y el subcomando `verificar` lo comprueba de tres formas:
- 714.291 parámetros;
- 26 capas iguales en tipo y configuración;
- pesos iniciales idénticos bit a bit con la misma semilla.

Además, la canónica entrenada por este script con la semilla 2026 reproduce **exactamente** la referencia
N-30 en la prueba: 0,460 / 0,533 / 0,606 / 0,615 / 0,685. Coinciden también las épocas y la validación de
cada pliegue.

**Recortes y su motivo.**
- El espacio tiene 1.728 combinaciones y se probaron 20 aleatorias (1,2 %).
- La fase 1 usa 1 sola semilla.
- No se exploraron otros optimizadores, otras programaciones de la tasa de aprendizaje, la paciencia, la
  aumentación, los pasos ni las velocidades. Estos cuatro últimos se dejaron fijos por instrucción.
- Por azar, 13 de las 20 configuraciones muestreadas tienen una sola capa Conv1D.
- El presupuesto (unas 12 h) habría alcanzado para más configuraciones, porque las corridas fueron más rápidas
  de lo estimado. **No se amplió** el plan fijado antes de empezar, para no decidir el tamaño de la búsqueda
  después de ver resultados.

**Tiempos.**

| Etapa | Inicio → fin | Duración |
|---|---|---|
| Espera a que terminaran los experimentos de pasos y velocidades | 30-sep 21:36 → 1-oct 02:06 | 4 h 30 min |
| Fase 0 (tiempo por pliegue de la canónica) | ya medido | 159–165 s por pliegue; 11 min los 5 pliegues |
| Fase 1 (21 configuraciones, 5 de ellas cortadas tras 2 pliegues; 90 corridas) | 02:06 → 05:26 | 3 h 20 min |
| Fase 2 (4 configuraciones × 3 semillas × 5 pliegues; 60 corridas) | 05:26 → 07:57 | 2 h 31 min |
| Espera de confirmación + inferencia + resumen | 07:57 → 08:13 | 16 min (inferencia 1,5 min) |
| **Cómputo total** | | **≈ 5 h 53 min** (presupuesto: 12 h) |

En la fase 1 cada configuración tardó entre 2 y 18 min. Las de lote 32 son las más lentas.

## 4. Fase 1: búsqueda (semilla 2026, 1 corrida por pliegue)

Ordenada por la val top-1 media, que es la métrica de selección. La prueba es solo descripción. ✂ marca una
configuración cortada tras los pliegues 0 y 1; sus medias son de esos 2 pliegues.

| id | configuración | parámetros | **val top-1 media** | val por pliegue (0/1/2/3/4) | prueba top-1 media | prueba F1 | épocas | min |
|---|---|---:|---:|---|---:|---:|---:|---:|
| **c07** | conv 1×64 k5 · BiLSTM 128 · dens 128-64 · do 0,3 · lr 5e-4 · b 64 · ls 0,1 | 389.811 | **0,932** | 0,95 / 0,92 / 0,89 / 0,97 / 0,93 | 0,599 | 0,557 | 58 | 7 |
| c00 (canónica) | conv 2×128 k3 · BiLSTM 128-64 · dens 256-128 · do 0,3 · lr 1e-3 · b 64 · ls 0,1 | 714.291 | 0,932 | 0,95 / 0,90 / 0,91 / 0,96 / 0,94 | 0,580 | 0,537 | 56 | 11 |
| **c17** | conv 2×64 k3 · BiLSTM 128 · dens 128-64 · do 0,3 · lr 2e-3 · b 32 · ls 0 | 370.163 | 0,930 | 0,94 / 0,89 / 0,92 / 0,97 / 0,93 | 0,585 | 0,543 | 76 | 13 |
| **c05** | conv 1×64 k3 · BiLSTM 128-64 · dens 128-64 · do 0,5 · lr 5e-4 · b 32 · ls 0,1 | 473.139 | 0,930 | 0,95 / 0,90 / 0,93 / 0,97 / 0,90 | 0,606 | 0,559 | 75 | 18 |
| c14 | conv 1×64 k5 · BiLSTM 128 · dens 128-64 · do 0,3 · lr 1e-3 · b 64 · ls 0 | 389.811 | 0,924 | 0,94 / 0,90 / 0,92 / 0,96 / 0,90 | 0,594 | 0,551 | 65 | 8 |
| c04 | conv 2×64 k5 · BiLSTM 128 · dens 256-128 · do 0,2 · lr 5e-4 · b 64 · ls 0 | 536.883 | 0,920 | 0,93 / 0,89 / 0,93 / 0,95 / 0,90 | 0,578 | 0,532 | 63 | 9 |
| c12 | conv 2×64 k5 · BiLSTM 64-32 · dens 256-128 · do 0,3 · lr 1e-3 · b 64 · ls 0,1 | 298.611 | 0,916 | 0,94 / 0,87 / 0,92 / 0,94 / 0,91 | 0,580 | 0,540 | 80 | 9 |
| c13 | conv 1×64 k5 · BiLSTM 64-32 · dens 128-64 · do 0,5 · lr 5e-4 · b 64 · ls 0,1 | 225.267 | 0,916 | 0,94 / 0,89 / 0,92 / 0,96 / 0,87 | 0,609 | 0,571 | 93 | 9 |
| c09 | conv 1×128 k5 · BiLSTM 64-32 · dens 128-64 · do 0,5 · lr 5e-4 · b 64 · ls 0,1 | 338.995 | 0,914 | 0,93 / 0,88 / 0,94 / 0,92 / 0,90 | 0,577 | 0,537 | 85 | 10 |
| c20 | conv 1×128 k3 · BiLSTM 128-64 · dens 256-128 · do 0,3 · lr 5e-4 · b 64 · ls 0,1 | 664.499 | 0,912 | 0,90 / 0,90 / 0,89 / 0,98 / 0,89 | 0,551 | 0,509 | 51 | 11 |
| c06 | conv 1×128 k5 · BiLSTM 128-64 · dens 128-64 · do 0,5 · lr 5e-4 · b 64 · ls 0 | 651.891 | 0,910 | 0,94 / 0,87 / 0,92 / 0,92 / 0,90 | 0,589 | 0,550 | 81 | 17 |
| c18 | conv 2×128 k5 · BiLSTM 128 · dens 256-128 · do 0,2 · lr 1e-3 · b 64 · ls 0 | 745.139 | 0,910 | 0,91 / 0,89 / 0,90 / 0,94 / 0,91 | 0,567 | 0,535 | 62 | 10 |
| c03 | conv 2×128 k5 · BiLSTM 128-64 · dens 256-128 · do 0,5 · lr 1e-3 · b 64 · ls 0,1 | 811.571 | 0,904 | 0,93 / 0,89 / 0,89 / 0,94 / 0,87 | 0,560 | 0,524 | 74 | 17 |
| c11 | conv 2×64 k5 · BiLSTM 128-64 · dens 256-128 · do 0,3 · lr 1e-3 · b 32 · ls 0,1 | 603.315 | 0,902 | 0,90 / 0,88 / 0,90 / 0,96 / 0,87 | 0,586 | 0,534 | 54 | 14 |
| c10 | conv 1×64 k3 · BiLSTM 128-64 · dens 256-128 · do 0,3 · lr 2e-3 · b 32 · ls 0,1 | 550.259 | 0,892 | 0,90 / 0,86 / 0,89 / 0,95 / 0,86 | 0,530 | 0,490 | 60 | 15 |
| c16 | conv 1×64 k3 · BiLSTM 64-32 · dens 128-64 · do 0,3 · lr 2e-3 · b 64 · ls 0 | 193.011 | 0,884 | 0,92 / 0,85 / 0,86 / 0,93 / 0,86 | 0,497 | 0,454 | 56 | 6 |
| c01 ✂ | conv 1×128 k3 · BiLSTM 64-32 · dens 128-64 · do 0,2 · lr 2e-3 · b 32 · ls 0 | 274.483 | 0,860 | 0,87 / 0,85 / — / — / — | 0,419 | 0,371 | 45 | 3 |
| c08 ✂ | conv 1×64 k3 · BiLSTM 64-32 · dens 128-64 · do 0,2 · lr 2e-3 · b 64 · ls 0 | 193.011 | 0,860 | 0,90 / 0,82 / — / — / — | 0,441 | 0,392 | 72 | 3 |
| c19 ✂ | conv 1×128 k3 · BiLSTM 128-64 · dens 256-128 · do 0,5 · lr 2e-3 · b 64 · ls 0 | 664.499 | 0,855 | 0,88 / 0,83 / — / — / — | 0,414 | 0,369 | 60 | 5 |
| c02 ✂ | conv 1×64 k3 · BiLSTM 64-32 · dens 128-64 · do 0,5 · lr 2e-3 · b 64 · ls 0,1 | 193.011 | 0,845 | 0,87 / 0,82 / — / — / — | 0,414 | 0,371 | 66 | 3 |
| c15 ✂ | conv 1×64 k5 · BiLSTM 64-32 · dens 256-128 · do 0,2 · lr 2e-3 · b 64 · ls 0 | 277.811 | 0,840 | 0,88 / 0,80 / — / — / — | 0,376 | 0,326 | 48 | 2 |

**Finalistas (`models/busqueda_hiperparametros/finalistas.json`, solo con validación): c07, c17 y c05.**
- c17 y c05 empataron en 0,930 de validación. Pasaron por la menor val_loss, el desempate fijado de antemano;
  c14 quedó cuarta (0,924).
- La búsqueda con una semilla **no separa a las mejores de la canónica**: las cuatro primeras están dentro de
  0,2 puntos (0,932–0,930), menos de una muestra de validación por pliegue. Por eso la confirmación con
  semillas nuevas es imprescindible.

**Lo que sí se ve en la fase 1** (descriptivo, una semilla):
- **lr 2e-3 es la única dimensión claramente mala.** Las 5 configuraciones cortadas usan lr 2e-3, y de las
  3 con lr 2e-3 que completaron, dos son las últimas (c10, c16). Las de lr 5e-4 y 1e-3 quedan todas entre
  0,90 y 0,93.
- Las configuraciones más grandes (c03, 811.571 parámetros; c18, 745.139) no superan a la canónica.

**¿Ordena la validación como la prueba?** La correlación de Spearman entre la val media y la prueba top-1 media
de las 16 configuraciones completas es ρ = 0,61 (p = 0,013).
- La validación (señantes ya vistos) sí ordena las configuraciones en la dirección correcta.
- La correlación es moderada: en la parte alta de la tabla el orden se mezcla.

## 5. Fase 2: confirmación (semillas nuevas 2031, 2032 y 2033)

### 5.1 Resumen

Cada cifra es la media de las 3 semillas en cada pliegue; ± es la desviación estándar entre los 5 pliegues.
La DE entre semillas es la de la media de 5 pliegues de cada semilla.

| config | parámetros | **val top-1** | DE entre semillas (val) | prueba top-1 LOSO | DE entre semillas (prueba) | prueba top-3 | prueba F1 macro | épocas (mejor) | s por pliegue |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| c00 canónica | 714.291 | 0,921 ± 0,024 | 0,006 | 0,585 ± 0,118 | 0,016 | 0,747 | 0,542 ± 0,120 | 58,0 (33,0) | 150 |
| **c07** | **389.811** | **0,935** ± 0,022 | 0,010 | 0,614 ± 0,150 | 0,033 | 0,783 | 0,579 ± 0,144 | 60,3 (35,3) | 91 |
| c05 | 473.139 | 0,934 ± 0,024 | 0,009 | 0,611 ± 0,117 | 0,005 | 0,767 | 0,574 ± 0,115 | 80,7 (55,7) | 234 |
| c17 | 370.163 | 0,911 ± 0,029 | 0,013 | 0,576 ± 0,120 | 0,015 | 0,752 | 0,533 ± 0,121 | 63,9 (38,9) | 128 |

### 5.2 Por pliegue y semilla

Validación top-1 (pliegues 0 / 1 / 2 / 3 / 4 → media):

| config | semilla 2031 | semilla 2032 | semilla 2033 | media por pliegue (3 semillas) |
|---|---|---|---|---|
| c00 | 0,92 / 0,90 / 0,91 / 0,93 / 0,93 → 0,918 | 0,94 / 0,89 / 0,93 / 0,96 / 0,92 → 0,928 | 0,92 / 0,87 / 0,92 / 0,97 / 0,90 → 0,916 | 0,927 / 0,887 / 0,920 / 0,953 / 0,917 |
| c07 | 0,94 / 0,91 / 0,94 / 0,95 / 0,92 → 0,932 | 0,93 / 0,91 / 0,93 / 0,99 / 0,97 → 0,946 | 0,94 / 0,90 / 0,93 / 0,96 / 0,91 → 0,928 | 0,937 / 0,907 / 0,933 / 0,967 / 0,933 |
| c05 | 0,95 / 0,92 / 0,94 / 0,98 / 0,92 → 0,942 | 0,95 / 0,90 / 0,92 / 0,95 / 0,90 → 0,924 | 0,94 / 0,90 / 0,95 / 0,97 / 0,92 → 0,936 | 0,947 / 0,907 / 0,937 / 0,967 / 0,913 |
| c17 | 0,86 / 0,87 / 0,89 / 0,95 / 0,91 → 0,896 | 0,95 / 0,90 / 0,92 / 0,96 / 0,88 → 0,922 | 0,90 / 0,89 / 0,92 / 0,97 / 0,89 → 0,914 | 0,903 / 0,887 / 0,910 / 0,960 / 0,893 |

Prueba top-1 LOSO, solo descriptiva (pliegues 0 / 1 / 2 / 3 / 4 → media):

| config | semilla 2031 | semilla 2032 | semilla 2033 | media por pliegue (3 semillas) |
|---|---|---|---|---|
| c00 | 0,365 / 0,593 / 0,621 / 0,640 / 0,655 → 0,575 | 0,425 / 0,523 / 0,641 / 0,720 / 0,710 → 0,604 | 0,400 / 0,543 / 0,591 / 0,690 / 0,660 → 0,577 | 0,397 / 0,553 / 0,618 / 0,683 / 0,675 |
| c07 | 0,365 / 0,588 / 0,662 / 0,650 / 0,680 → 0,589 | 0,390 / 0,618 / 0,707 / 0,745 / 0,800 → 0,652 | 0,335 / 0,588 / 0,646 / 0,690 / 0,750 → 0,602 | 0,363 / 0,598 / 0,672 / 0,695 / 0,743 |
| c05 | 0,455 / 0,553 / 0,652 / 0,645 / 0,735 → 0,608 | 0,440 / 0,593 / 0,621 / 0,710 / 0,680 → 0,609 | 0,360 / 0,608 / 0,727 / 0,660 / 0,725 → 0,616 | 0,418 / 0,585 / 0,667 / 0,672 / 0,713 |
| c17 | 0,395 / 0,578 / 0,621 / 0,625 / 0,745 → 0,593 | 0,380 / 0,543 / 0,606 / 0,635 / 0,675 → 0,568 | 0,390 / 0,508 / 0,591 / 0,655 / 0,695 → 0,568 | 0,388 / 0,543 / 0,606 / 0,638 / 0,705 |

### 5.3 Aplicación del criterio y comparación pareada con la canónica

- **σ_semilla** agrupada (4 configuraciones) = **0,0099**. Como es menor que el piso, **margen = 0,010**.
- Como referencia, la desviación de una sola corrida (celda configuración × pliegue) es 0,018.

| config | Δ val por pliegue (0/1/2/3/4) | **Δ val media** | pliegues mejor / igual / peor | (a) Δ > 0 | (b) ≥ 4 de 5 | (c) Δ > margen | **Cumple** | signos p | Wilcoxon p |
|---|---|---:|---|:-:|:-:|:-:|:-:|---:|---:|
| c07 | +0,010 / +0,020 / +0,013 / +0,013 / +0,017 | **+0,0147** | 5 / 0 / 0 | sí | sí | sí | **sí** | 0,062 | 0,062 |
| c05 | +0,020 / +0,020 / +0,017 / +0,013 / −0,003 | **+0,0133** | 4 / 0 / 1 | sí | sí | sí | **sí** | 0,375 | 0,125 |
| c17 | −0,023 / 0,000 / −0,010 / +0,007 / −0,023 | −0,0100 | 1 / 1 / 3 | no | no | no | no | 0,625 | 0,312 |

Lo mismo en la prueba LOSO, **solo descriptivo** (no decide):

| config | Δ prueba top-1 por pliegue | Δ media | mejor / peor | signos p | Wilcoxon p | Δ F1 macro media |
|---|---|---:|---|---:|---:|---:|
| c07 | −0,033 / +0,045 / +0,054 / +0,012 / +0,068 | +0,029 | 4 / 1 | 0,375 | 0,188 | +0,037 |
| c05 | +0,022 / +0,032 / +0,049 / −0,012 / +0,038 | +0,026 | 4 / 1 | 0,375 | 0,125 | +0,032 |
| c17 | −0,008 / −0,010 / −0,012 / −0,045 / +0,030 | −0,009 | 1 / 4 | 0,375 | 0,438 | −0,009 |

**Límites de las pruebas con 5 pliegues.**
- Con 5 diferencias, el p bilateral más pequeño posible es 0,0625, tanto en la prueba de signos como en
  Wilcoxon. Solo se alcanza con 5 de 5 en el mismo sentido, y es lo que obtiene c07 en validación.
- Ninguna prueba puede llegar a 0,05. Describen consistencia, no significancia.
- Además, los 5 pliegues comparten 3 de cada 4 señantes de entrenamiento, así que no son del todo
  independientes.

### 5.4 Caída búsqueda → confirmación

| config | val fase 1 (semilla 2026) | val fase 2 (3 semillas nuevas) | caída | ventaja sobre la canónica en fase 1 | ventaja en fase 2 | prueba fase 1 → fase 2 |
|---|---:|---:|---:|---:|---:|---|
| c00 canónica | 0,932 | 0,921 | **−1,1 puntos** | — | — | 0,580 → 0,585 |
| c07 | 0,932 | 0,935 | +0,3 (no cae) | 0,0 | +1,5 | 0,599 → 0,614 |
| c05 | 0,930 | 0,934 | +0,4 (no cae) | −0,2 | +1,3 | 0,606 → 0,611 |
| c17 | 0,930 | 0,911 | **−1,9 puntos** | −0,2 | −1,0 | 0,585 → 0,576 |

**Lectura.**
- **c17** muestra el efecto previsto de elegir el máximo entre medidas ruidosas: entró como finalista con
  0,930, con lr 2e-3, que es la dimensión mala, y con semillas nuevas cayó 1,9 puntos hasta quedar debajo de
  la canónica.
- **c07 y c05 no cayeron.**
- La ventaja de c07 y c05 aparece en la fase 2 porque **la canónica sí cayó**: la semilla 2026 le fue
  favorable.
  - Lo confirman, de forma independiente, los experimentos de pasos y de velocidades. Con la misma canónica
    y las semillas 2026, 2027 y 2028 obtuvieron val 0,932 / 0,918 / 0,910 (media 0,920).
  - Con las 6 semillas disponibles, la val de la canónica es ≈ 0,920. Ninguna de las 3 semillas de c07
    (0,932 / 0,946 / 0,928) queda por debajo de 0,928.
- La comparación justa es la de la fase 2, todas con semillas nuevas. En la fase 1, con una sola semilla, la
  diferencia no se veía.

## 6. Tamaño y latencia de los finalistas

Medido con `model.predict` y un lote de 1, como en `translator.py`.
- 10 llamadas de calentamiento y 100 medidas por modelo, sobre 3 modelos por configuración; se reporta la
  mediana de las medianas.
- Solo la red: sin MediaPipe ni remuestreo.
- CPU, sin otros entrenamientos activos: `inferencia.json` registra una lista vacía de procesos.

La llamada directa `model(x, training=False)` es en modo *eager*, sin compilar. Se incluye solo como
indicador del costo de cómputo de la red.

| config | parámetros | archivo .keras (sin optimizador) | predict lote 1, mediana (ms) | p90 (ms) | llamada directa (ms) | entrenamiento por pliegue (s) |
|---|---:|---:|---:|---:|---:|---:|
| c00 canónica | 714.291 | 2,83 MB | 12,9 | 13,5 | 61,1 | 150 |
| **c07** | **389.811 (−45 %)** | 1,56 MB | **12,3** | 12,7 | 32,1 | **91** |
| c05 | 473.139 (−34 %) | 1,90 MB | 12,8 | 13,2 | 59,9 | 234 |
| c17 | 370.163 (−48 %) | 1,50 MB | 12,4 | 13,0 | 32,6 | 128 |

Con lote 1 domina el costo fijo de `predict`, de unos 12 ms. Por eso c07 solo ahorra 0,6 ms por ventana,
aunque su cómputo propio sea la mitad: 32 frente a 61 ms en modo eager, porque tiene una sola capa BiLSTM y
una sola convolución. Lo mismo midió el experimento de velocidades: unos 12 ms a lote 1, independientes del
tamaño. Donde c07 sí ahorra es en el entrenamiento, un 40 % menos de tiempo por pliegue.

## 7. Recomendación según el criterio

1. **Dos configuraciones cumplen el criterio prerregistrado: c07 y c05.**
   - Las dos superan a la canónica en la val media (+1,47 y +1,33 puntos), en al menos 4 de 5 pliegues
     (5 de 5 y 4 de 5) y por encima del margen (1,0 punto).
   - Entre ellas la diferencia es de 0,13 puntos, menor que el margen: empate práctico.
   - El desempate fijado de antemano es el menor tamaño, y lo gana **c07**.
2. **Configuración recomendada por el criterio: c07.**

   | Hiperparámetro | c07 | Canónica |
   |---|---|---|
   | Conv1D | 1 capa de 64 filtros, ventana 5 | 2 capas de 128, ventana 3 |
   | BiLSTM | una sola capa de 128 | 128 → 64 |
   | Densas | 128 → 64 | 256 → 128 |
   | Dropout | 0,3 | 0,3 |
   | lr inicial | 5e-4 | 1e-3 |
   | Lote | 64 | 64 |
   | Suavizado de etiquetas | 0,1 | 0,1 |
   | Parámetros | 389.811 | 714.291 |

   - Es casi la mitad de grande.
   - Entrena un 40 % más rápido.
   - Cuesta lo mismo en inferencia en el prototipo.
3. **Por el criterio, la canónica dejaría de ser la mejor opción. Pero la decisión no es de esta búsqueda, y
   este informe no cambia el modelo canónico ni el umbral.** Es del estudiante con el director. Adoptar c07
   exigiría rehacer:
   - el modelo canónico del esquema aleatorio y el LOSO de referencia. Hoy `train_final_100.py` y
     `train_loso.py` construyen siempre `build_model`, así que habría que añadirles una arquitectura
     configurable, por ejemplo importando `construir_modelo`;
   - el umbral de confianza 0,20 (D-09) y la propuesta anidada 0,55 (N-47), calibrados con la canónica;
   - el análisis de errores y las cifras citadas en README y DECISIONES.

   El prototipo (`translator.py`) carga la arquitectura desde el `.keras`, así que no necesitaría cambios.
4. **Qué tan fuerte es la evidencia: moderada, no concluyente.**
   - *A favor:*
     - c07 mejora la validación en los 5 pliegues.
     - Sus 3 semillas nuevas quedan en o por encima de la mejor de las 6 semillas de la canónica
       (0,932 / 0,946 / 0,928 frente a máximo 0,932).
     - No hubo caída entre búsqueda y confirmación.
     - c05, con otra arquitectura pero el mismo lr 5e-4 y densas 128-64, confirma la misma dirección.
     - En la prueba LOSO, sin valor de decisión, c07 da +2,9 puntos de top-1 y +3,7 de F1 en 4 de 5 pliegues.
   - *En contra:*
     - El margen se cumple gracias al piso de 0,01. σ_semilla salió en 0,0099, y frente a la desviación de
       una sola corrida (0,018) ninguna cumpliría la condición (c).
     - En la prueba, c07 **empeora en el pliegue 0**, el señante difícil (0,363 frente a 0,397).
     - Su ventaja de prueba depende mucho de una semilla (2032: 0,652; las otras, 0,589 y 0,602), con la
       mayor DE entre semillas de las cuatro (0,033).
     - c05 es más estable entre semillas (0,005), mejor en el pliegue 0 (0,418) y casi igual en la media.
       Si se prioriza la robustez ante el señante más difícil sobre el tamaño, c05 es la alternativa
       defendible, aunque entrena un 55 % más lento que la canónica.
5. **Alternativa razonable:** mantener la canónica y citar esta búsqueda en la tesis como evidencia de que sus
   hiperparámetros están cerca del óptimo del espacio explorado. La ganancia posible es de 1 a 1,5 puntos de
   validación y de unos 3 de prueba, del orden de la variación entre semillas. Así se evita recalibrar antes
   del Bloque 1. Como en el experimento de pasos, apartarse de la regla escrita antes de correr exige decirlo
   y dar el motivo (el costo de recalibrar), no buscar otro criterio a posteriori.

## 8. Relación con los otros dos experimentos (no integrados aquí)

- **Pasos temporales** (`docs/EXPERIMENTO_PASOS_TEMPORALES.md`).
  - Su regla prerregistrada elige **60 pasos** en vez de 30: +1,6 puntos de validación, en 5 de 5 pliegues.
  - Su propio informe califica la evidencia de **débil**: en la prueba LOSO la mejora es de +0,9 puntos, en
    2 de 5 pliegues.
  - Duplica el tiempo de entrenamiento. Su costo en inferencia es de +1,5 ms.
- **Velocidades** (`docs/EXPERIMENTO_VELOCIDADES.md`).
  - **No hay evidencia** de que las velocidades ayuden ni estorben en validación (Δ −0,3 puntos, 2 de
    5 pliegues).
  - Se mantienen (30 × 252).
- Esta búsqueda se hizo con 30 pasos y con velocidades, como pedía el diseño. **No se probó la combinación c07
  + 60 pasos.** Las dos mejoras son del mismo orden (1,5 puntos de validación) y no se puede suponer que se
  sumen. Si se decidiera cambiar ambas cosas, habría que confirmar la combinación con un LOSO de 3 semillas
  antes de adoptarla.
- Los tres experimentos coinciden en algo útil: con 6 semillas en total, la val de la canónica es ≈ 0,920 y su
  prueba LOSO ≈ 0,585–0,590. La cifra reportada de 0,580 (semilla 2026) está en la parte baja de su rango.

## 9. Limitaciones

1. **La validación no mide lo que importa.** Sale de los cuatro señantes de entrenamiento: mide reconocer a
   personas conocidas, no a personas nuevas.
   - Ordena las configuraciones en la dirección correcta (ρ = 0,61 con la prueba), pero de forma imperfecta.
   - Está cerca del techo (0,89–0,97): una diferencia de 1 punto es 1 muestra de cada 100 por pliegue.
2. **Búsqueda pequeña.** 20 de 1.728 combinaciones, una sola semilla en la fase 1 y las cuatro primeras dentro
   de 0,2 puntos.
   - Puede haber configuraciones mejores sin probar.
   - Con otra semilla de muestreo los finalistas podrían haber sido otros.
3. **Solo 5 señantes.** Ninguna prueba puede ser significativa al 5 %. La DE de la prueba entre pliegues
   (0,12–0,15) es diez veces mayor que las diferencias entre configuraciones.
4. **Sensibilidad al margen.** c07 cumple por 0,5 puntos sobre el piso de 0,01. Con un margen basado en la
   variación de una sola corrida no cumpliría.
5. **Dimensiones fijas.** Pasos, velocidades, aumentación, paciencia y optimizador quedaron fijos, y no se
   estudiaron interacciones, en particular con 60 pasos.
6. **El corte temprano usa solo 2 pliegues.** Las 5 configuraciones cortadas tenían lr 2e-3 y estaban entre
   6,5 y 8,5 puntos por debajo, así que es poco probable que se haya descartado una buena.
7. **Latencia solo de la red.** Medida en este portátil con lote 1. No incluye MediaPipe, el cuello de botella
   real del prototipo.
8. **Coordinación con los otros experimentos.** Las esperas de los otros experimentos no vigilaban este script. No
   hubo solapamiento: los dos habían terminado antes de empezar la fase 1, y la medición de latencia se hizo
   sin otros procesos activos.

## 10. Explicación simple

La red del prototipo se diseñó en marzo con valores elegidos a ojo: cuántos filtros, cuántas capas, qué tan
rápido aprende. Aquí se probaron 20 combinaciones distintas y se compararon con la original usando solo los
datos de validación, nunca los de prueba. En la primera ronda, con una sola semilla, varias empataron con la
original, así que se repitieron las tres mejores con tres semillas nuevas. Dos de ellas superaron a la original
en todos o casi todos los pliegues por algo más de 1 punto. Las dos aprenden más despacio (tasa 5e-4) y tienen
menos capas. La más pequeña (c07) tiene casi la mitad de parámetros, entrena un 40 % más rápido y responde igual
de rápido. También acierta unos 3 puntos más con personas nuevas, pero no con la persona más difícil, y la
ventaja es del tamaño del azar entre semillas. Cumple la regla fijada de antemano, aunque por poco. Cambiar o no
el modelo lo deciden el estudiante y el director, sabiendo que habría que recalibrar el umbral.

## 11. Archivos y cómo reproducir

**Script:** `src/busqueda_hiperparametros.py`, que reutiliza `train_loso.py` y `train_final_100.py` sin
modificarlos.

**Datos crudos:** `docs/BUSQUEDA_HIPERPARAMETROS.json`, que contiene:
- la fase 1 por configuración y pliegue;
- la fase 2 por configuración, semilla y pliegue;
- el criterio aplicado, las pruebas pareadas y la caída;
- la reproducción de N-30;
- la latencia y el entorno.

**Corridas:** `models/busqueda_hiperparametros/`, unos 296 MB.
- `plan_fase1.json`: las 21 configuraciones y las semillas.
- `finalistas.json`: el ranking de validación y los finalistas.
- `inferencia.json`.
- `espera.log` y los registros `corrida_*.log`.
- `fase1/<id>/semilla_2026/pliegue_k/` y `fase2/<id>/semilla_s/pliegue_k/`, cada una con:
  - `resultado.json`;
  - `historial.json`;
  - `predicciones_prueba.json`;
  - `modelo.keras` (sin estado del optimizador) y `modelo.procedencia.json`.
- `fase1/<id>/cortada.json` para las configuraciones cortadas.

**Reproducción** (intérprete `python`; TF 2.21.0, Keras 3.13.2,
CPU):

```
python src/busqueda_hiperparametros.py verificar
python src/busqueda_hiperparametros.py fase1
python src/busqueda_hiperparametros.py seleccionar
python src/busqueda_hiperparametros.py fase2
python src/busqueda_hiperparametros.py inferencia
python src/busqueda_hiperparametros.py resumen
```

Todo es determinista con las semillas fijadas: la canónica reproduce N-30 al dígito.
