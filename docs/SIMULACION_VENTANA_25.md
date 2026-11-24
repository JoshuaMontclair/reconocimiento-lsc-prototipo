# Ventana de 2,5 s: duración de los clips, duración de la seña y simulación del recorte

Generado por `src/simular_ventana_25.py` el 2026-11-23T20:11:01. Solo análisis: no cambia el prototipo,
los modelos, los umbrales ni las particiones.

## 1. Duración de los clips

| Fuente | n | media | mediana | p90 | máx | ≤ 2,5 s | ≤ 3,0 s |
|---|---|---|---|---|---|---|---|
| Video COLOR_BODY (fotogramas / 24), 997 válidos — **la correcta** | 997 | 2,81 | 2,79 | 3,38 | 5,62 | 29,1 % | 68,0 % |
| Filas de los CSV publicados / 24, 1.000 (base de N-11) | 1000 | 2,57 | 2,58 | 3,29 | 5,58 | 46,9 % | 79,4 % |

Las cifras de N-11 (media 2,57 s; mediana 2,58 s; p90 3,29 s; máx. 5,58 s; 47 % ≤ 2,5 s; 79 % ≤ 3,0 s)
salen de dividir entre 24 el número de FILAS de los CSV de puntos publicados, que en 749 de 1.000 muestras
tienen menos filas que fotogramas el video (mediana del déficit: 5; caracterizacion.md §12). El modelo
final se entrenó con la extracción propia sobre el video COLOR_BODY (24,0 fps en las 1.000 cabeceras),
cuyos clips tienen 40-135 fotogramas (mediana 67); 259 de 997 tienen menos de 60.

## 2. Duración de la seña dentro del clip

Criterio: reposo = muñecas quietas (velocidad media <= 1,0 anchos de hombro por segundo) y bajas (la más alta a 1,26 anchos de hombro o más bajo la línea de los hombros); seña = del primer al último fotograma que no está en reposo. Medida de `src/analisis_senas_cortadas.py` (_altura, _velocidad_pose).

- Clips con actividad: 997 (sin actividad: 0).
- Seña: mediana 1,46 s, media 1,55 s, p90 2,08 s, máx. 3,38 s; ≤ 2,5 s: 96,5 %; ≤ 2,25 s: 93,9 %; ≤ 3,0 s: 99,3 %.
- Reposo antes de la seña: mediana 0,33 s, p90 0,75 s, máx. 3,33 s; ≥ 0,5 s en 34,1 % de los clips, ≥ 1 s en 3,2 %.
- Reposo después de la seña: mediana 0,92 s, p90 1,18 s.
- La seña ocupa una mediana del 53,2 % del clip.
- La seña termina después de los primeros 2,5 s del clip en 10,3 % de los clips.

Por señante:

| señante | mediana seña (s) | seña ≤ 2,5 s | mediana reposo inicial (s) | mediana reposo final (s) |
|---|---|---|---|---|
| 0000 | 1,46 | 99,5 % | 0,58 | 0,94 |
| 0001 | 1,54 | 97,0 % | 0,29 | 1,00 |
| 0002 | 1,29 | 99,0 % | 0,46 | 0,96 |
| 0003 | 1,38 | 96,5 % | 0,04 | 0,83 |
| 0004 | 1,71 | 90,5 % | 0,42 | 0,83 |

Sensibilidad al criterio:

| h baja (anchos de hombro) | v (anchos/s) | mediana seña (s) | p90 (s) | ≤ 2,5 s |
|---|---|---|---|---|
| 1,00 | 0,5 | 1,67 | 2,25 | 95,8 % |
| 1,00 | 1,0 | 1,42 | 1,85 | 99,2 % |
| 1,00 | 2,0 | 1,33 | 1,71 | 99,2 % |
| 1,15 | 0,5 | 1,67 | 2,25 | 95,6 % |
| 1,15 | 1,0 | 1,42 | 1,88 | 98,9 % |
| 1,15 | 2,0 | 1,33 | 1,75 | 99,0 % |
| 1,26 | 0,5 | 1,67 | 2,39 | 93,2 % |
| 1,26 | 1,0 | 1,46 | 2,08 | 96,5 % |
| 1,26 | 2,0 | 1,38 | 1,96 | 96,8 % |

Cuánto de la seña deja fuera cada ventana (997 clips; tramo activo de la parte 2):

| variante | clips cortados | s fuera (mediana / p90) | fracción de la seña fuera (mediana / p90) |
|---|---|---|---|
| primeros_2_5s | 103 (10,3 %) | 0,25 / 0,71 | 11 % / 27 % |
| primeros_2_5s_sin_relleno | 103 (10,3 %) | 0,25 / 0,71 | 11 % / 27 % |
| centrados_2_5s | 147 (14,7 %) | 0,17 / 0,42 | 8 % / 19 % |
| ultimos_2_5s | 355 (35,6 %) | 0,21 / 0,69 | 12 % / 32 % |
| retraso_0_25s | 61 (6,1 %) | 0,29 / 0,79 | 11 % / 26 % |
| retraso_0_5s | 111 (11,1 %) | 0,29 / 0,88 | 13 % / 30 % |
| retraso_0_75s | 216 (21,7 %) | 0,29 / 0,96 | 14 % / 35 % |
| retraso_1_0s | 408 (40,9 %) | 0,29 / 0,93 | 16 % / 38 % |
| primeros_3_0s | 22 (2,2 %) | 0,21 / 0,57 | 7 % / 20 % |
| primeros_3_5s | 3 (0,3 %) | 0,08 / 0,88 | 3 % / 70 % |

## 3. Simulación con los modelos LOSO del modelo final

Modelos: `models/confirmacion_tcn_2026-10-03/T1_imputacion_252`, semillas [2026, 2027, 2028] × 5 pliegues. Ventana de 2,5 s = 60 fotogramas a 24 fps, aplicada antes de la imputación, el remuestreo a 30 y las velocidades (el orden del prototipo). Clips de menos de 60 fotogramas: se completan repitiendo el fotograma del borde. Control: el clip completo reproduce las probabilidades guardadas de las 15 corridas (sí; diferencia máxima 7.2e-07).

| variante | seña cortada | top-1 cortadas / no cortadas | top-1 (media de corridas) | por semilla | DE entre semillas | top-3 | rechazo con 0,68 | exactitud aceptadas | pliegues peor que completo |
|---|---|---|---|---|---|---|---|---|---|
| completo | 0,0 % | — / 0,677 | 0,677 | 0,688 / 0,658 / 0,685 | 0,014 | 0,847 | 35,8 % | 0,836 | 0 de 5 |
| primeros_2_5s | 10,3 % | 0,770 / 0,669 | 0,679 | 0,696 / 0,660 / 0,682 | 0,015 | 0,850 | 34,3 % | 0,831 | 2 de 5 |
| primeros_2_5s_sin_relleno | 10,3 % | 0,770 / 0,668 | 0,678 | 0,697 / 0,657 / 0,679 | 0,016 | 0,853 | 33,1 % | 0,828 | 3 de 5 |
| centrados_2_5s | 14,7 % | 0,722 / 0,676 | 0,682 | 0,699 / 0,662 / 0,685 | 0,015 | 0,857 | 33,4 % | 0,834 | 1 de 5 |
| ultimos_2_5s | 35,6 % | 0,672 / 0,644 | 0,668 | 0,674 / 0,653 / 0,676 | 0,010 | 0,848 | 35,9 % | 0,829 | 3 de 5 |
| retraso_0_25s | 6,1 % | 0,734 / 0,680 | 0,685 | 0,701 / 0,667 / 0,686 | 0,014 | 0,855 | 33,4 % | 0,834 | 2 de 5 |
| retraso_0_5s | 11,1 % | 0,653 / 0,681 | 0,685 | 0,703 / 0,663 / 0,690 | 0,017 | 0,854 | 33,0 % | 0,830 | 1 de 5 |
| retraso_0_75s | 21,6 % | 0,709 / 0,663 | 0,673 | 0,695 / 0,647 / 0,676 | 0,020 | 0,846 | 35,4 % | 0,832 | 3 de 5 |
| retraso_1_0s | 40,9 % | 0,691 / 0,615 | 0,650 | 0,678 / 0,628 / 0,643 | 0,021 | 0,834 | 40,5 % | 0,820 | 5 de 5 |
| primeros_3_0s | 2,2 % | 0,813 / 0,674 | 0,678 | 0,684 / 0,667 / 0,682 | 0,008 | 0,842 | 40,0 % | 0,856 | 2 de 5 |
| primeros_3_5s | 0,3 % | 0,667 / 0,658 | 0,658 | 0,672 / 0,641 / 0,662 | 0,013 | 0,828 | 47,4 % | 0,855 | 5 de 5 |

Top-1 por pliegue (media de las semillas):

| pliegue | señante | completo | primeros_2_5s | primeros_2_5s_sin_relleno | centrados_2_5s | ultimos_2_5s | retraso_0_25s | retraso_0_5s | retraso_0_75s | retraso_1_0s | primeros_3_0s | primeros_3_5s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0000 | 0,488 | 0,508 | 0,507 | 0,508 | 0,488 | 0,515 | 0,512 | 0,485 | 0,450 | 0,488 | 0,473 |
| 1 | 0001 | 0,611 | 0,593 | 0,593 | 0,588 | 0,588 | 0,595 | 0,596 | 0,593 | 0,575 | 0,613 | 0,603 |
| 2 | 0002 | 0,721 | 0,714 | 0,715 | 0,724 | 0,705 | 0,719 | 0,722 | 0,715 | 0,704 | 0,705 | 0,675 |
| 3 | 0003 | 0,770 | 0,773 | 0,767 | 0,778 | 0,775 | 0,783 | 0,780 | 0,775 | 0,752 | 0,760 | 0,752 |
| 4 | 0004 | 0,795 | 0,808 | 0,807 | 0,812 | 0,782 | 0,812 | 0,817 | 0,795 | 0,768 | 0,822 | 0,788 |

Según dónde termina la seña (todas las predicciones de prueba):

| grupo | clips | top-1 completo | top-1 primeros 2,5 s | top-1 retraso 0,25 s | top-1 retraso 1,0 s |
|---|---|---|---|---|---|
| clip_le_60 | 290 | 0,717 | 0,723 | 0,733 | 0,710 |
| clip_gt_60_sena_termina_antes_de_60 | 604 | 0,634 | 0,638 | 0,638 | 0,608 |
| clip_gt_60_sena_termina_despues_de_60 | 103 | 0,816 | 0,799 | 0,819 | 0,725 |

Clases que más pierden con los primeros 2,5 s:

| clase | seña | n | completo | primeros 2,5 s | pérdida | retraso 1,0 s | mediana clip (s) | mediana seña (s) | clips > 2,5 s |
|---|---|---|---|---|---|---|---|---|---|
| 26 | ¿cómo estás? | 60 | 0,433 | 0,350 | 0,083 | 0,417 | 2,67 | 1,54 | 70 % |
| 46 | ¿quién? | 60 | 0,767 | 0,733 | 0,033 | 0,750 | 2,77 | 1,42 | 75 % |
| 40 | ustedes | 60 | 0,517 | 0,483 | 0,033 | 0,567 | 2,96 | 1,48 | 80 % |
| 9 | poco | 60 | 0,550 | 0,517 | 0,033 | 0,483 | 2,50 | 1,31 | 45 % |
| 22 | triste | 60 | 0,700 | 0,667 | 0,033 | 0,550 | 2,73 | 1,58 | 80 % |
| 27 | más o menos | 60 | 0,467 | 0,433 | 0,033 | 0,417 | 2,81 | 1,42 | 75 % |
| 47 | diferente | 60 | 0,983 | 0,950 | 0,033 | 0,950 | 2,65 | 1,50 | 55 % |
| 11 | personas | 60 | 0,517 | 0,500 | 0,017 | 0,383 | 2,77 | 1,54 | 80 % |
| 15 | niña | 60 | 0,550 | 0,533 | 0,017 | 0,417 | 2,79 | 1,46 | 65 % |
| 19 | hambre | 60 | 0,967 | 0,950 | 0,017 | 0,867 | 2,67 | 1,23 | 65 % |
| 43 | ¿dónde? | 60 | 0,650 | 0,633 | 0,017 | 0,583 | 2,90 | 1,50 | 85 % |
| 41 | ¿qué? | 60 | 0,283 | 0,267 | 0,017 | 0,233 | 2,65 | 1,23 | 60 % |

## 4. Sesión en vivo del 23-nov (registro)

`registros/sesion_final_2026-11-23/latencia_2026-11-23.jsonl`: 87 ventanas en modo aislado con la clave «bordes»; aviso_fin en 69, aviso_inicio en 32.

## 5. Límites

- Recortar clips del corpus no es una ventana en vivo: en vivo la ventana empieza con la tecla y la persona arranca desde el reposo; aquí el inicio de la ventana es el del clip (con su reposo inicial, que varía).
- Los clips de menos de 60 fotogramas se completan con el fotograma del borde quieto; el reposo real tiene pequeños movimientos.
- El criterio de actividad es heurístico (altura y velocidad de las muñecas de la postura); la tabla de sensibilidad muestra cuánto cambia la duración de la seña con él.
- Cinco señantes y la cámara del corpus (24 fps); el prototipo captura a ~30 fps con otra cámara.
