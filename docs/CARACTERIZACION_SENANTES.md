# Condición del señante excluido (actividad 4.e)

Generado por `src/caracterizar_senantes.py` el 2026-11-23 (commit 5ecb7cc). Solo reporte: no entrena ni evalúa modelos. No usa imágenes de personas: solo estadísticas de puntos característicos.

## 1. Lo que documenta el corpus

- Methods: 2 voluntarios sordos y 3 oyentes («Two of them are deaf and three of them are not hearing impaired»).
- Data Records: 3 señantes nativos y 2 no nativos («5 volunteers: 3 native speakers and two non-native speakers»).
- Reclutados a través de la Asociación de Intérpretes de Lengua de Señas Colombiana-Español (Methods).
- Todos fluidos en LSC y mayores de 18 años en el momento de la captura (Methods).
- Los intérpretes que validaron las señas «are non-native speakers and are some of the participants that can be seen in the stored data» (Technical Validation).
- No publica sexo, edad individual, lateralidad ni qué voluntario (0000-0004) pertenece a cada grupo.
- Los dos criterios no se corresponden entre sí (2 + 3 frente a 3 + 2) y el artículo no publica cuál de los cinco voluntarios (0000-0004) pertenece a cada grupo. Se registran ambos SIN conciliarlos, como exige la actividad 1.b del anteproyecto.

Antropometría publicada en `IMU/INFO/Volunteer_k.txt` (cotejada con el archivo: Volunteer_1 sí, Volunteer_2 sí, Volunteer_3 sí, Volunteer_4 sí, Volunteer_5 sí):

| Archivo | Estatura | Húmero | Radio | Espalda (cóccix a C7) |
|---|---|---|---|---|
| Volunteer_1 | 176cm | 29cm | 27cm | 52cm |
| Volunteer_2 | 164cm | 28cm | 25cm | 46cm |
| Volunteer_3 | 156cm | 26cm | 22cm | 39cm |
| Volunteer_4 | 158cm | 26cm | 21cm | 38cm |
| Volunteer_5 | 165cm | 28cm | 26cm | 55cm |

La correspondencia entre Volunteer_k (IMU) y el campo VVVV = 000(k−1) de los nombres de archivo es la presumible, pero el material descargado no la declara: no se afirma.

Fuente: data/processed/caracterizacion/caracterizacion.json › senantes (actividad 1.b).

## 2. Propiedades medidas por señante (mediana [P25; P75] sobre sus videos)

| Propiedad | 0000 | 0001 | 0002 | 0003 | 0004 |
|---|---|---|---|---|---|
| Videos | 200 | 199 | 198 | 200 | 200 |
| Duración (cuadros) | 71,0 [67,0; 77,0] | 70,0 [63,0; 77,0] | 66,0 [60,0; 72,0] | 56,0 [52,0; 60,0] | 72,0 [64,0; 78,0] |
| Duración (s) | 2,96 [2,79; 3,21] | 2,92 [2,62; 3,21] | 2,75 [2,50; 3,00] | 2,33 [2,17; 2,50] | 3,00 [2,67; 3,25] |
| Ancho de hombros en la imagen (fracción del ancho) | 0,125 [0,119; 0,126] | 0,101 [0,098; 0,104] | 0,100 [0,099; 0,102] | 0,097 [0,096; 0,098] | 0,109 [0,108; 0,109] |
| Altura de los hombros en la imagen (fracción del alto (0 = arriba)) | 0,485 [0,481; 0,508] | 0,561 [0,552; 0,562] | 0,459 [0,457; 0,461] | 0,459 [0,457; 0,461] | 0,440 [0,438; 0,441] |
| Velocidad de la muñeca activa (anchos de hombro/s) | 1,82 [1,57; 2,09] | 1,67 [1,42; 2,00] | 1,73 [1,45; 2,24] | 1,55 [1,25; 1,83] | 1,47 [1,18; 1,80] |
| Altura máxima de la muñeca activa (anchos de hombro sobre el centro de los hombros) | -0,09 [-0,64; 0,43] | -0,09 [-0,39; 0,43] | 0,11 [-0,20; 0,44] | 0,03 [-0,41; 0,33] | 0,14 [-0,19; 0,54] |
| Fotogramas sin mano izquierda (%) | 6,7 | 1,3 | 14,3 | 5,9 | 3,0 |
| Fotogramas sin mano derecha (%) | 5,5 | 2,5 | 2,3 | 2,3 | 1,1 |
| Fotogramas sin alguna mano (%) | 11,3 | 3,7 | 16,3 | 7,9 | 3,7 |
| Videos con algún fotograma sin mano (%) | 88,5 | 57,8 | 94,4 | 81,5 | 54,5 |
| Videos con la derecha como mano activa (%) | 22,0 | 80,4 | 16,2 | 12,5 | 15,5 |
| Videos con las dos manos en movimiento (%) | 64,5 | 51,3 | 48,5 | 42,0 | 44,5 |
| De los de una mano, con la derecha (%) | 2,8 | 94,8 | 1,0 | 1,7 | 0,9 |
| Control: derecha como más activa según las muñecas de la postura (%) | 3,5 | 86,9 | 7,1 | 9,5 | 3,5 |
| Fotogramas sin alguna mano, media por video (%) (= análisis 4.d) | 11,6 | 3,7 | 16,8 | 7,8 | 3,6 |
| Distancia a las otras personas, misma seña (anchos de hombro) | 0,982 [0,816; 1,172] | 0,804 [0,559; 1,221] | 0,581 [0,486; 0,706] | 0,636 [0,554; 0,778] | 0,596 [0,502; 0,749] |
| Dispersión de sus propias repeticiones (anchos de hombro) | 0,373 [0,276; 0,528] | 0,200 [0,146; 0,274] | 0,189 [0,139; 0,254] | 0,158 [0,113; 0,213] | 0,172 [0,130; 0,227] |
| **Top-1 de prueba, modelo final (señante excluido)** | 0,488 | 0,611 | 0,721 | 0,770 | 0,795 |
| Videos con una sola mano en movimiento | 71 | 97 | 102 | 116 | 111 |
| Top-1 de prueba, modelo heredado | 0,452 | 0,536 | 0,631 | 0,640 | 0,687 |
| Top-1 de validación del final cuando es visto | 0,892 | 0,942 | 0,979 | 0,950 | 0,983 |

Definiciones:

- **identificador**: nombre de archivo SSSS_VVVV_RRRR: seña, señante (VVVV = 0000-0004), repetición; el pliegue k del LOSO deja fuera al señante 000k
- **muestras**: 997 válidas (D-06): 200 / 199 / 198 / 200 / 200 por señante
- **fotogramas**: lengths.npy (antes de remuestrear); segundos = fotogramas / 24
- **ancho_hombros**: mediana por video de la distancia x-y entre los hombros (postura 11 y 12), en fracción del ancho de la imagen; igual que c4a_estadisticas.py
- **altura_hombros_imagen**: mediana por video de la altura del centro de los hombros, fracción del alto de la imagen (0 = borde superior)
- **sin_mano**: fotogramas con la bandera de detección en 0 (METRICAS_FOTOGRAMA) / fotogramas de la persona (cociente de totales)
- **mano_activa**: bloque de mano cuya muñeca (punto 0) recorre más en x-y normalizado (pares consecutivos con la mano presente; analisis_errores._movimiento_munecas)
- **dos_manos**: recorrido de la muñeca que menos se mueve ≥ 0.5 × el de la que más se mueve (proxy de analisis_errores_tcn.py; no es una anotación lingüística)
- **velocidad_activa**: recorrido de la muñeca activa / (pares válidos / 24), en anchos de hombro por segundo
- **altura_max_activa**: −(mínimo de y de la muñeca activa) en el espacio normalizado: anchos de hombro por encima del centro de los hombros
- **distancia_a_otras_personas**: muñecas de la postura (15, 16) remuestreadas a 30 pasos; trayectoria media por persona y seña; distancia RMS (anchos de hombro) a la de cada una de las otras cuatro en la misma seña; promedio de las cuatro; resumen sobre las 50 señas
- **dispersion_repeticiones_propias**: distancia RMS de cada repetición a la trayectoria media de su persona y seña
- **resumen**: mediana, percentiles 25 y 75 (RIC) y media sobre los videos de la persona

## 3. Orden de las personas y posición del 0000

| Propiedad | de menor a mayor | posición del 0000 (1 = menor) | ¿0000 en un extremo? | Spearman con el top-1 final (descriptivo) |
|---|---|---|---|---|
| fotogramas_mediana | 0003 < 0002 < 0001 < 0000 < 0004 | 4 | no | 0,00 |
| ancho_hombros_mediana | 0003 < 0002 < 0001 < 0004 < 0000 | 5 | sí | -0,40 |
| altura_hombros_imagen_mediana | 0004 < 0003 < 0002 < 0000 < 0001 | 4 | no | -0,90 |
| velocidad_activa_mediana | 0004 < 0003 < 0001 < 0002 < 0000 | 5 | sí | -0,90 |
| altura_max_activa_mediana | 0000 < 0001 < 0003 < 0002 < 0004 | 1 | sí | 0,90 |
| sin_mano_izq_pct_fotogramas | 0001 < 0004 < 0003 < 0000 < 0002 | 4 | no | -0,20 |
| sin_mano_der_pct_fotogramas | 0004 < 0003 < 0002 < 0001 < 0000 | 5 | sí | -1,00 |
| sin_alguna_mano_pct_fotogramas | 0004 < 0001 < 0003 < 0000 < 0002 | 4 | no | -0,50 |
| mano_activa_derecha_pct_videos | 0003 < 0004 < 0002 < 0000 < 0001 | 4 | no | -0,80 |
| dos_manos_pct_videos | 0003 < 0004 < 0002 < 0001 < 0000 | 5 | sí | -0,90 |
| distancia_a_otras_personas_mediana | 0002 < 0004 < 0003 < 0001 < 0000 | 5 | sí | -0,70 |
| dispersion_repeticiones_propias_mediana | 0003 < 0004 < 0002 < 0001 < 0000 | 5 | sí | -0,90 |

Cinco personas no permiten establecer una relación entre una propiedad y el top-1 del pliegue: con cinco puntos, un orden idéntico (Spearman = ±1) aparece por azar en 2 de 120 órdenes posibles, y las propiedades están mezcladas entre sí y con la persona. Los coeficientes de Spearman son solo descriptivos.

Distancia media entre trayectorias de las personas (anchos de hombro; promedio de las 50 señas):

| | 0000 | 0001 | 0002 | 0003 | 0004 |
|---|---|---|---|---|---|
| 0000 | 0,000 | 1,084 | 0,908 | 1,046 | 0,932 |
| 0001 | 1,084 | 0,000 | 0,818 | 0,879 | 0,864 |
| 0002 | 0,908 | 0,818 | 0,000 | 0,358 | 0,321 |
| 0003 | 1,046 | 0,879 | 0,358 | 0,000 | 0,392 |
| 0004 | 0,932 | 0,864 | 0,321 | 0,392 | 0,000 |

## 4. Dentro de cada persona: aciertos frente a errores del modelo final (3 semillas juntas, descriptivo)

| Señante | duración (cuadros) acierto / error | velocidad activa acierto / error | altura máx. acierto / error | % sin alguna mano acierto / error | errores |
|---|---|---|---|---|---|
| 0000 | 72,0 / 70,0 | 1,86 / 1,79 | -0,16 / -0,03 | 12,0 / 11,3 | 307 de 600 |
| 0001 | 71,0 / 69,0 | 1,70 / 1,62 | -0,11 / -0,07 | 3,4 / 4,2 | 232 de 597 |
| 0002 | 66,0 / 65,5 | 1,73 / 1,81 | 0,07 / 0,19 | 16,4 / 17,7 | 166 de 594 |
| 0003 | 56,0 / 54,0 | 1,56 / 1,44 | 0,00 / 0,04 | 6,8 / 11,1 | 138 de 600 |
| 0004 | 73,0 / 68,0 | 1,44 / 1,66 | 0,04 / 0,33 | 3,6 / 3,8 | 123 de 600 |

Cotejo con cifras ya publicadas (c4a_estadisticas, ANALISIS_ERRORES_4d_TCN, tab:c5_senantes): 32 de 32 coinciden.


## 5. Resultado por pliegue (cotejado con lo guardado)

- final: media de 3 semillas 0000 0,488, 0001 0,611, 0002 0,721, 0003 0,770, 0004 0,795; rango entre semillas 0000 0,150, 0001 0,045, 0002 0,051, 0003 0,110, 0004 0,100.
- heredado: media de 3 semillas 0000 0,452, 0001 0,536, 0002 0,631, 0003 0,640, 0004 0,687; rango entre semillas 0000 0,065, 0001 0,090, 0002 0,076, 0003 0,040, 0004 0,085.

Fuentes: arreglos: `data/processed_holistic_mascara_rostro (X_padded.npy, lengths.npy, metadata.json)`; coordenadas_de_imagen: `data/extracted/LSC50_holistic/BODY_LANDMARKS`; banderas_de_deteccion: `data/extracted/LSC50_holistic/METRICAS_FOTOGRAMA`; particion_y_excluidas: `data/partitions/loso_5_pliegues.json`; lo_documentado: `data/processed/caracterizacion/caracterizacion.json; data/raw/LSC50/IMU/INFO (solo lectura)`; modelo_final: `models/confirmacion_tcn_2026-10-03/T1_imputacion_252`; modelo_heredado: `models/experimento_mano_ausente_2026-10-02/A_mascara`; validacion_como_visto: `docs/REPORTE_COMPARATIVO_4e_MODELO_FINAL.json › validacion_loso_final_por_senante`.
