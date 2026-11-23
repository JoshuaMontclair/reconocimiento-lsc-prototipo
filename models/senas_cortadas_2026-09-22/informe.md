# Señas cortadas en los bordes de la ventana: análisis, reglas y daño al modelo «manos»

Generado por `src/analisis_senas_cortadas.py` el 2026-09-22T16:36:55 (fin 2026-09-22T16:37:09). Solo reporte: no cambia el prototipo ni ningún modelo, umbral o partición.

Todas las cifras están en `resultados.json` (la clave de origen va entre corchetes).

## 0. Orden de los hechos (criterio antes que resultado)

- `criterio.md` modificado por última vez: 2026-09-22T16:16:33.
- Inicio de esta corrida: 2026-09-22T16:36:55. [corrida]

## 1. Lo que dice el anteproyecto y lo que hace el prototipo

- §2.2.6 (09_metodologia.tex:192-193): «Las secuencias incompletas (sin inicio o fin claros dentro de la ventana temporal) se descartarán.» Y 193-196: la inferencia opera «mediante una ventana temporal de longitud fija activada por la persona usuaria (botón o tecla) que delimita el inicio y el fin de la seña; la detección automática de inicio y fin (segmentación basada en movimiento) se deja como posible mejora futura».
- El prototipo (src/translator.py, `_cerrar_ventana`), como secuencia incompleta (`rechazada_incompleta`), solo descarta por X («abortada») o por menos de `min_frames` fotogramas; no mira si la seña empezó o terminó dentro de la ventana (N-37; protocolo 4.a, CL-6b y R6).

## 2. Cómo empiezan y terminan los clips del corpus

Corpus: 1000 clips, longitud 40-135 fotogramas (mediana 67.0) a 24 fps. Velocidades en anchos de hombro por fotograma; ×24 da anchos de hombro por segundo. Altura = y de la muñeca más alta de la postura respecto a la línea de los hombros (positivo = por debajo; 1 = un ancho de hombros). [corpus]

### 2.1 Bordes frente al centro (todos los clips íntegros)

| K | v_pose ini (mediana / p95) | v_pose fin (mediana / p95) | v_pose centro (mediana) | v_manos ini (mediana / p95) | v_manos fin (mediana / p95) | v_manos centro (mediana) | h ini (mediana) | h fin (mediana) | h centro (mediana) |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0,0157 / 0,0733 | 0,0026 / 0,0095 | 0,0544 | 0,0202 / 0,1194 | 0,0094 / 0,0322 | 0,0780 | 1,47 | 1,46 | 0,72 |
| 5 | 0,0132 / 0,1131 | 0,0027 / 0,0095 | 0,0544 | 0,0185 / 0,1635 | 0,0099 / 0,0312 | 0,0780 | 1,46 | 1,46 | 0,72 |
| 8 | 0,0119 / 0,1359 | 0,0032 / 0,0094 | 0,0544 | 0,0218 / 0,1776 | 0,0105 / 0,0318 | 0,0780 | 1,44 | 1,46 | 0,72 |

Cociente borde/centro de v_pose (mediana sobre clips, K = 5): inicio 0,229, fin 0,050. [corpus.cociente_borde_centro]

### 2.2 Fotogramas sin alguna mano, según dónde caen (K = 5)

Fracción media de fotogramas sin alguna mano: primeros 5 6,1 %, últimos 5 6,3 %, mitad central 10,1 %. Clips con v_manos_ini indefinido (ninguna mano detectada en dos fotogramas seguidos del borde): 0,0 %; v_manos_fin indefinido: 0,0 %. [corpus.sin_mano, corpus.v_manos_indefinido]

### 2.3 Por señante (clips íntegros, K = 5)

| señante | n | v_pose ini mediana / p95 | v_pose fin mediana / p95 | v_pose centro mediana | h ini mediana | h fin mediana | h centro mediana | sin mano ini | sin mano fin |
|---|---|---|---|---|---|---|---|---|---|
| 0000 | 200 | 0,0120 / 0,0481 | 0,0018 / 0,0050 | 0,0745 | 2,71 | 1,92 | 1,00 | 1,4 % | 7,0 % |
| 0001 | 200 | 0,0137 / 0,0833 | 0,0026 / 0,0107 | 0,0532 | 1,93 | 1,84 | 0,86 | 2,5 % | 0,6 % |
| 0002 | 200 | 0,0098 / 0,0621 | 0,0024 / 0,0059 | 0,0618 | 1,43 | 1,39 | 0,65 | 20,4 % | 23,3 % |
| 0003 | 200 | 0,0632 / 0,1398 | 0,0043 / 0,0106 | 0,0449 | 1,21 | 1,33 | 0,76 | 6,0 % | 0,1 % |
| 0004 | 200 | 0,0104 / 0,0892 | 0,0034 / 0,0115 | 0,0508 | 1,36 | 1,40 | 0,50 | 0,1 % | 0,4 % |

### 2.4 Por seña (K = 5): las 8 señas con mayor velocidad mediana en los bordes

| seña | v_pose ini mediana | v_pose fin mediana | h ini mediana | h fin mediana |
|---|---|---|---|---|
| 0018 | 0,0191 | 0,0030 | 1,37 | 1,41 |
| 0016 | 0,0167 | 0,0028 | 1,53 | 1,49 |
| 0014 | 0,0166 | 0,0030 | 1,52 | 1,50 |
| 0009 | 0,0165 | 0,0023 | 1,44 | 1,46 |
| 0012 | 0,0160 | 0,0021 | 1,56 | 1,50 |
| 0019 | 0,0157 | 0,0029 | 1,46 | 1,52 |
| 0004 | 0,0156 | 0,0027 | 1,46 | 1,49 |
| 0046 | 0,0155 | 0,0033 | 1,48 | 1,42 |

La tabla completa de las 50 señas y de los tres K está en `resultados.json` [corpus.por_sena].

### 2.5 POST HOC, descriptivo: ¿el inicio de los clips cumple el criterio de reposo de su final?

No forma parte del criterio ni de la selección. Umbral de reposo tomado del FINAL de los 1000 clips (incluidas las 3 duplicadas que D-06 excluye de las particiones; percentil 95 de v_pose_fin; percentil 5 de h_fin) y aplicado al INICIO de los mismos clips. [corpus.inicio_juzgado_con_reposo_del_final]

| K | v reposo (p95 fin) | h reposo (p5 fin) | inicio en movimiento, por señante 0000/0001/0002/0003/0004 | inicio con mano alta, por señante | alguno de los dos, por señante |
|---|---|---|---|---|---|
| 3 | 0,0095 | 1,26 | 81,0 % / 77,0 % / 68,0 % / 98,0 % / 69,5 % | 0,0 % / 2,0 % / 2,5 % / 45,5 % / 16,5 % | 81,0 % / 77,0 % / 68,0 % / 98,0 % / 71,5 % |
| 5 | 0,0095 | 1,26 | 71,5 % / 74,5 % / 52,0 % / 100,0 % / 53,5 % | 0,0 % / 2,5 % / 4,0 % / 60,0 % / 20,0 % | 71,5 % / 74,5 % / 52,0 % / 100,0 % / 58,0 % |
| 8 | 0,0094 | 1,27 | 48,5 % / 77,0 % / 36,0 % / 98,5 % / 50,5 % | 0,5 % / 4,5 % / 9,5 % / 84,5 % / 30,0 % | 48,5 % / 77,0 % / 36,0 % / 99,0 % / 56,0 % |

## 3. Reglas de detección: ajuste en entrenamiento, evaluación en el señante de prueba

Familia, umbrales (percentil 95 de los íntegros de ENTRENAMIENTO), selección de (K, rasgo) por detección media en los truncados de ENTRENAMIENTO y evaluación en el señante de prueba, tal como fija `criterio.md`. «detección» = marcada como cortada por cualquiera de los dos bordes (lo que rechazaría el prototipo); la detección por el borde correcto está en `resultados.json` [pliegues[].regla_elegida.test.deteccion_lado].

### 3.1 Configuración elegida por pliegue y resultado en prueba

| pliegue | prueba | K | rasgo | umbral ini | umbral fin | det. media train | FP íntegros prueba (ini / fin) | det. inicio 20/35/50 % | det. fin 20/35/50 % |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0000 | 3 | v_pose_o_h | 0,0760 (h: 0,90 / 1,25) | 0,0104 | 0,815 | 2,5 % (2,0 % / 0,5 %) | 67,5 % / 86,5 % / 80,0 % | 27,0 % / 95,0 % / 95,5 % |
| 1 | 0001 | 3 | v_pose_o_h | 0,0759 (h: 0,91 / 1,25) | 0,0091 | 0,816 | 9,5 % (3,0 % / 7,5 %) | 93,0 % / 88,9 % / 85,9 % | 30,7 % / 96,0 % / 98,5 % |
| 2 | 0002 | 3 | v_pose_o_h | 0,0760 (h: 0,92 / 1,25) | 0,0103 | 0,808 | 4,0 % (2,5 % / 1,5 %) | 70,2 % / 96,0 % / 90,4 % | 32,3 % / 93,4 % / 99,5 % |
| 3 | 0003 | 8 | v_pose_o_h | 0,1249 (h: 1,13 / 1,30) | 0,0087 | 0,875 | 75,5 % (66,5 % / 35,5 %) | 97,5 % / 98,0 % / 70,5 % | 88,5 % / 99,5 % / 100,0 % |
| 4 | 0004 | 3 | v_pose_o_h | 0,0775 (h: 0,93 / 1,27) | 0,0089 | 0,821 | 22,5 % (2,5 % / 20,0 %) | 70,5 % / 92,0 % / 83,5 % | 60,5 % / 98,5 % / 98,0 % |

Media entre pliegues (señantes de prueba): FP íntegros 22,8 % (rango 2,5 %-75,5 %); detección inicio 79,7 % / 92,3 % / 82,1 %; fin 47,8 % / 96,5 % / 98,3 %. Configuración mayoritaria: K = 3, rasgo v_pose_o_h (4 de 5 pliegues). [resumen_pliegues]

### 3.2 Rejilla completa en el señante de prueba (solo para leer; la elección no la usa)

| pliegue | K | rasgo | det. media train | FP prueba | det. media prueba | det. ini 35 % | det. fin 35 % |
|---|---|---|---|---|---|---|---|
| 0 | 3 | v_pose | 0,503 | 2,5 % | 0,464 | 26,5 % | 91,0 % |
| 0 | 3 | h | 0,652 | 0,0 % | 0,483 | 77,5 % | 38,5 % |
| 0 | 3 | v_pose_o_h | 0,815 | 2,5 % | 0,752 | 86,5 % | 95,0 % |
| 0 | 3 | v_manos | 0,420 | 19,5 % | 0,460 | 34,5 % | 86,5 % |
| 0 | 5 | v_pose | 0,468 | 3,0 % | 0,482 | 16,5 % | 91,0 % |
| 0 | 5 | h | 0,618 | 0,0 % | 0,451 | 64,0 % | 47,0 % |
| 0 | 5 | v_pose_o_h | 0,775 | 3,0 % | 0,736 | 72,5 % | 97,0 % |
| 0 | 5 | v_manos | 0,408 | 20,5 % | 0,474 | 27,0 % | 88,5 % |
| 0 | 8 | v_pose | 0,469 | 7,0 % | 0,520 | 10,0 % | 93,0 % |
| 0 | 8 | h | 0,569 | 0,0 % | 0,414 | 47,0 % | 60,0 % |
| 0 | 8 | v_pose_o_h | 0,723 | 7,0 % | 0,710 | 52,5 % | 98,5 % |
| 0 | 8 | v_manos | 0,422 | 27,0 % | 0,541 | 27,5 % | 91,5 % |
| 1 | 3 | v_pose | 0,527 | 9,0 % | 0,482 | 22,6 % | 92,5 % |
| 1 | 3 | h | 0,627 | 3,5 % | 0,602 | 85,9 % | 30,7 % |
| 1 | 3 | v_pose_o_h | 0,816 | 9,5 % | 0,822 | 88,9 % | 96,0 % |
| 1 | 3 | v_manos | 0,394 | 2,5 % | 0,348 | 10,6 % | 75,4 % |
| 1 | 5 | v_pose | 0,504 | 10,1 % | 0,457 | 12,6 % | 92,0 % |
| 1 | 5 | h | 0,610 | 3,5 % | 0,551 | 75,9 % | 37,7 % |
| 1 | 5 | v_pose_o_h | 0,792 | 10,6 % | 0,778 | 78,9 % | 96,5 % |
| 1 | 5 | v_manos | 0,384 | 5,5 % | 0,379 | 11,1 % | 84,4 % |
| 1 | 8 | v_pose | 0,508 | 14,6 % | 0,483 | 12,6 % | 94,0 % |
| 1 | 8 | h | 0,566 | 3,5 % | 0,486 | 55,3 % | 51,3 % |
| 1 | 8 | v_pose_o_h | 0,746 | 15,6 % | 0,721 | 60,3 % | 98,5 % |
| 1 | 8 | v_manos | 0,404 | 8,5 % | 0,393 | 8,5 % | 90,5 % |
| 2 | 3 | v_pose | 0,482 | 3,5 % | 0,560 | 37,4 % | 89,4 % |
| 2 | 3 | h | 0,620 | 2,5 % | 0,636 | 92,9 % | 50,5 % |
| 2 | 3 | v_pose_o_h | 0,808 | 4,0 % | 0,803 | 96,0 % | 93,4 % |
| 2 | 3 | v_manos | 0,389 | 10,1 % | 0,477 | 38,9 % | 69,2 % |
| 2 | 5 | v_pose | 0,462 | 2,0 % | 0,497 | 20,2 % | 94,9 % |
| 2 | 5 | h | 0,593 | 2,5 % | 0,644 | 90,9 % | 69,2 % |
| 2 | 5 | v_pose_o_h | 0,779 | 3,5 % | 0,783 | 94,4 % | 96,5 % |
| 2 | 5 | v_manos | 0,379 | 9,1 % | 0,445 | 25,8 % | 85,9 % |
| 2 | 8 | v_pose | 0,472 | 2,0 % | 0,468 | 8,1 % | 98,5 % |
| 2 | 8 | h | 0,538 | 2,0 % | 0,588 | 71,7 % | 87,4 % |
| 2 | 8 | v_pose_o_h | 0,726 | 3,5 % | 0,700 | 74,2 % | 99,0 % |
| 2 | 8 | v_manos | 0,393 | 7,1 % | 0,435 | 20,7 % | 93,4 % |
| 3 | 3 | v_pose | 0,611 | 50,5 % | 0,677 | 32,0 % | 94,0 % |
| 3 | 3 | h | 0,683 | 55,5 % | 0,833 | 99,5 % | 67,0 % |
| 3 | 3 | v_pose_o_h | 0,856 | 67,5 % | 0,942 | 100,0 % | 96,5 % |
| 3 | 3 | v_manos | 0,496 | 51,5 % | 0,583 | 32,0 % | 78,0 % |
| 3 | 5 | v_pose | 0,559 | 42,0 % | 0,593 | 23,0 % | 99,0 % |
| 3 | 5 | h | 0,705 | 63,0 % | 0,869 | 100,0 % | 79,5 % |
| 3 | 5 | v_pose_o_h | 0,865 | 70,0 % | 0,942 | 100,0 % | 99,5 % |
| 3 | 5 | v_manos | 0,489 | 51,5 % | 0,540 | 20,0 % | 87,5 % |
| 3 | 8 | v_pose | 0,526 | 23,5 % | 0,513 | 16,5 % | 98,5 % |
| 3 | 8 | h | 0,716 | 73,0 % | 0,879 | 98,0 % | 89,0 % |
| 3 | 8 | v_pose_o_h | 0,875 | 75,5 % | 0,923 | 98,0 % | 99,5 % |
| 3 | 8 | v_manos | 0,442 | 23,5 % | 0,380 | 6,5 % | 88,0 % |
| 4 | 3 | v_pose | 0,518 | 9,5 % | 0,526 | 32,5 % | 92,0 % |
| 4 | 3 | h | 0,616 | 14,5 % | 0,721 | 86,5 % | 92,0 % |
| 4 | 3 | v_pose_o_h | 0,821 | 22,5 % | 0,838 | 92,0 % | 98,5 % |
| 4 | 3 | v_manos | 0,387 | 2,0 % | 0,369 | 26,5 % | 83,5 % |
| 4 | 5 | v_pose | 0,502 | 13,0 % | 0,514 | 22,0 % | 95,0 % |
| 4 | 5 | h | 0,586 | 14,0 % | 0,718 | 81,5 % | 93,5 % |
| 4 | 5 | v_pose_o_h | 0,794 | 25,0 % | 0,828 | 85,5 % | 100,0 % |
| 4 | 5 | v_manos | 0,382 | 2,5 % | 0,334 | 11,5 % | 85,5 % |
| 4 | 8 | v_pose | 0,486 | 7,0 % | 0,480 | 12,0 % | 94,0 % |
| 4 | 8 | h | 0,540 | 16,0 % | 0,684 | 71,0 % | 93,5 % |
| 4 | 8 | v_pose_o_h | 0,727 | 21,5 % | 0,795 | 74,5 % | 100,0 % |
| 4 | 8 | v_manos | 0,405 | 3,5 % | 0,364 | 10,0 % | 84,5 % |

### 3.3 Sensibilidad al percentil del umbral (configuración elegida de cada pliegue; no cambia la elección)

| pliegue | percentil | FP íntegros prueba | det. inicio 20/35/50 % | det. fin 20/35/50 % |
|---|---|---|---|---|
| 0 | 90 | 3,5 % | 71,0 % / 94,5 % / 88,5 % | 37,5 % / 98,0 % / 98,0 % |
| 0 | 99 | 0,5 % | 54,0 % / 50,5 % / 47,0 % | 12,0 % / 88,5 % / 91,5 % |
| 1 | 90 | 13,1 % | 96,0 % / 96,0 % / 95,0 % | 42,2 % / 97,0 % / 99,0 % |
| 1 | 99 | 4,5 % | 53,3 % / 46,7 % / 57,3 % | 15,1 % / 83,4 % / 97,5 % |
| 2 | 90 | 6,1 % | 72,7 % / 97,0 % / 96,5 % | 43,4 % / 96,5 % / 100,0 % |
| 2 | 99 | 1,5 % | 53,5 % / 66,7 % / 52,5 % | 12,1 % / 82,8 % / 99,5 % |
| 3 | 90 | 90,0 % | 99,5 % / 100,0 % / 86,0 % | 98,0 % / 100,0 % / 100,0 % |
| 3 | 99 | 10,0 % | 52,5 % / 47,5 % / 9,5 % | 27,0 % / 90,0 % / 99,0 % |
| 4 | 90 | 39,5 % | 82,0 % / 96,0 % / 92,0 % | 75,0 % / 100,0 % / 98,5 % |
| 4 | 99 | 6,0 % | 49,5 % / 69,5 % / 56,0 % | 32,5 % / 96,0 % / 92,5 % |

### 3.4 Umbral candidato con los cinco señantes (configuración mayoritaria K = 3, rasgo v_pose_o_h; percentil 95)

v_pose ini 0,0734 / fin 0,0096 anchos de hombro por fotograma a 24 fps (= 1,761 / 0,229 anchos/s); h ini 0,974 / fin 1,261; v_manos ini 0,1196 / fin 0,0323. Es un CANDIDATO: se calcula con todos los señantes, así que no tiene evaluación limpia; la adopción la decide el director. [umbral_candidato_cinco_senantes]

En unidades independientes de la tasa (lo que usaría el prototipo): borde de 0,125 s; v_pose ini 1,761 / fin 0,229 anchos de hombro por segundo; h ini 0,974 / fin 1,261. Ejemplo de conversión a 30,4 fot./s (tasa medida en N-32): K = 4 fotogramas, v_pose ini 0,0579 / fin 0,0075 por fotograma. La conversión supone que la velocidad por segundo no depende de la tasa de captura; el ruido del extractor por fotograma sí puede depender de ella, y eso no se ha medido. [umbral_candidato_cinco_senantes.por_segundo]

## 4. Daño al modelo «manos» con clips truncados (señante de prueba de cada pliegue)

Control previo: predicciones recalculadas sobre los clips íntegros frente a `pliegue_k/predicciones.json` y `resultados_loso.json`: pliegue 0: 200/200 iguales, top-1 0,460 = 0,460; pliegue 1: 199/199 iguales, top-1 0,533 = 0,533; pliegue 2: 198/198 iguales, top-1 0,606 = 0,606; pliegue 3: 200/200 iguales, top-1 0,615 = 0,615; pliegue 4: 200/200 iguales, top-1 0,685 = 0,685. [modelo.control]

### 4.1 Agregado de los 997 clips de prueba, por versión

| versión | top-1 | top-3 | confianza media | rechazo < 0,20 | exactitud aceptadas | aceptadas ok / mal |
|---|---|---|---|---|---|---|
| integro | 0,580 | 0,770 | 0,543 | 16,0 % | 0,645 | 540 / 297 |
| inicio_0.20 | 0,554 | 0,755 | 0,535 | 18,4 % | 0,627 | 510 / 304 |
| inicio_0.35 | 0,410 | 0,631 | 0,455 | 24,9 % | 0,489 | 366 / 383 |
| inicio_0.50 | 0,200 | 0,331 | 0,391 | 34,6 % | 0,232 | 151 / 501 |
| fin_0.20 | 0,570 | 0,772 | 0,565 | 14,5 % | 0,637 | 543 / 309 |
| fin_0.35 | 0,573 | 0,752 | 0,539 | 17,3 % | 0,640 | 528 / 297 |
| fin_0.50 | 0,527 | 0,714 | 0,450 | 24,7 % | 0,613 | 460 / 291 |

### 4.2 Top-1 por pliegue y versión

| pliegue | integro | inicio_0.20 | inicio_0.35 | inicio_0.50 | fin_0.20 | fin_0.35 | fin_0.50 |
|---|---|---|---|---|---|---|---|
| 0 | 0,460 | 0,455 | 0,335 | 0,215 | 0,460 | 0,480 | 0,430 |
| 1 | 0,533 | 0,482 | 0,332 | 0,146 | 0,513 | 0,492 | 0,457 |
| 2 | 0,606 | 0,606 | 0,475 | 0,162 | 0,596 | 0,616 | 0,510 |
| 3 | 0,615 | 0,500 | 0,365 | 0,150 | 0,610 | 0,610 | 0,605 |
| 4 | 0,685 | 0,725 | 0,545 | 0,325 | 0,670 | 0,665 | 0,630 |

### 4.3 Regla elegida + umbral 0,20 sobre los clips truncados (agregado de los 5 señantes de prueba)

| versión | rechaza regla | rechaza umbral | rechaza alguno | pasan ambos | exactitud pasan ambos | errores en voz alta sin / con regla | aciertos en voz alta sin / con regla | U (c = 1) sin / con regla |
|---|---|---|---|---|---|---|---|---|
| integro | 22,9 % | 16,0 % | 36,3 % | 635 | 0,631 | 297 / 234 | 540 / 401 | 243 / 167 |
| inicio_0.20 | 79,7 % | 18,4 % | 82,5 % | 174 | 0,695 | 304 / 53 | 510 / 121 | 206 / 68 |
| inicio_0.35 | 92,3 % | 24,9 % | 94,3 % | 57 | 0,491 | 383 / 29 | 366 / 28 | -17 / -1 |
| inicio_0.50 | 82,0 % | 34,6 % | 87,0 % | 130 | 0,200 | 501 / 104 | 151 / 26 | -350 / -78 |
| fin_0.20 | 47,8 % | 14,5 % | 56,1 % | 438 | 0,594 | 309 / 178 | 543 / 260 | 234 / 82 |
| fin_0.35 | 96,5 % | 17,3 % | 96,8 % | 32 | 0,375 | 297 / 20 | 528 / 12 | 231 / -8 |
| fin_0.50 | 98,3 % | 24,7 % | 98,6 % | 14 | 0,786 | 291 / 3 | 460 / 11 | 169 / 8 |

U = aciertos − errores dichos en voz alta (función de D-09 con c = 1), solo como resumen; no decide nada aquí.

[modelo.agregado, modelo.pliegues, modelo.combinado_agregado]

### 4.4 POST HOC, descriptivo: la regla elegida aplicada solo a un borde

No forma parte del criterio. Misma regla y umbrales de cada pliegue, pero marcando solo por el borde inicial o solo por el final, para ver de dónde sale el beneficio y de dónde el costo. [modelo.post_hoc_por_borde]

| versión | solo inicio: rechaza | errores sin / con | aciertos sin / con | U sin / con | solo fin: rechaza | errores sin / con | aciertos sin / con | U sin / con |
|---|---|---|---|---|---|---|---|---|
| integro | 15,3 % | 297 / 250 | 540 / 449 | 243 / 199 | 13,0 % | 297 / 261 | 540 / 462 | 243 / 201 |
| inicio_0.20 | 78,5 % | 304 / 59 | 510 / 126 | 206 / 67 | 13,0 % | 304 / 269 | 510 / 438 | 206 / 169 |
| inicio_0.35 | 91,7 % | 383 / 33 | 366 / 29 | -17 / -4 | 13,0 % | 383 / 337 | 366 / 319 | -17 / -18 |
| inicio_0.50 | 79,3 % | 501 / 125 | 151 / 26 | -350 / -99 | 13,0 % | 501 / 432 | 151 / 134 | -350 / -298 |
| fin_0.20 | 15,3 % | 309 / 262 | 543 / 446 | 234 / 184 | 42,3 % | 309 / 200 | 543 / 290 | 234 / 90 |
| fin_0.35 | 15,3 % | 297 / 253 | 528 / 433 | 231 / 180 | 96,1 % | 297 / 20 | 528 / 15 | 231 / -5 |
| fin_0.50 | 15,3 % | 291 / 250 | 460 / 372 | 169 / 122 | 98,2 % | 291 / 3 | 460 / 12 | 169 / 9 |

## 5. Figuras

- `fig1_perfil_velocidad.png`
- `fig2_rasgos_borde_por_senante.png`
- `fig3_deteccion_prueba.png`
- `fig4_dano_modelo.png`

## 6. Opciones para el prototipo, recomendación y especificación

Están en `recomendacion.md` (misma carpeta), escrito a mano sobre estas cifras después de la corrida.


## Anexo: criterio fijado antes de correr (copia literal de `criterio.md`)

# Criterio de elección, fijado ANTES de medir — señas cortadas en los bordes de la ventana

Fecha y hora de escritura: 2026-09-22 16:17 (hora local, -05). Este archivo se escribe antes de
ejecutar `src/analisis_senas_cortadas.py`; el script lo lee, lo copia en `informe.md` y registra en
`resultados.json` la hora de modificación de este archivo y la hora de inicio de la corrida, para que
el verificador compruebe el orden. Nada de lo que sigue se cambia después de ver resultados.

## Qué se decide con datos

Una regla de detección de «seña cortada» al abrir y al cerrar la ventana, sus umbrales y su
parámetro K (número de fotogramas del borde). La adopción en el prototipo NO se decide aquí: se
propone al director (pieza 4, punto 4).

## Datos

- `data/processed_holistic_mascara_rostro/` (solo lectura): `X_padded.npy` (1000, 135, 405),
  `lengths.npy`, `metadata.json` (`sample_order`). Coordenadas ya normalizadas por hombros
  (centro = punto medio de los hombros, escala = distancia entre hombros en xy; y crece hacia abajo).
  Videos a 24 fps (N-02).
- Particiones: `data/partitions/loso_5_pliegues.json`. En el pliegue k, «entrenamiento» =
  `conjuntos.train` ∪ `conjuntos.val` (los cuatro señantes distintos de k) y «prueba» =
  `conjuntos.test` (todas las muestras no excluidas del señante k). Las tres muestras duplicadas
  excluidas por D-06 no entran en ningún conjunto.

## Rasgos por clip (todos en el espacio normalizado por hombros, plano xy)

Para un clip de T fotogramas y un borde de K fotogramas (K ∈ {3, 5, 8}):

1. `v_pose_ini(K)`: media, sobre los K−1 pares consecutivos de los primeros K fotogramas y sobre
   las dos muñecas de la POSTURA (puntos 15 y 16 de PoseLandmarker; columnas 45:47 y 48:50), del
   desplazamiento ‖Δxy‖ por fotograma. `v_pose_fin(K)`: lo mismo sobre los últimos K fotogramas.
   Un par en el que la postura falta en alguno de los dos fotogramas no cuenta; si no queda ningún
   par, el rasgo es NaN.
2. `v_manos_ini(K)`, `v_manos_fin(K)`: media, sobre los pares consecutivos del borde en que la
   mano está detectada en ambos fotogramas, y sobre las manos detectadas, del desplazamiento medio
   ‖Δxy‖ de los 21 puntos de la mano (bloques 99:162 y 162:225). NaN si no hay ningún par válido.
3. `h_ini(K)`, `h_fin(K)`: media, sobre los K fotogramas del borde, de min(y_muñeca_izq,
   y_muñeca_der) de la postura (la muñeca más alta; y menor = más arriba). NaN si no hay postura.
4. Referencia del tramo central: la misma velocidad sobre el tramo [⌊T/4⌋, T − ⌊T/4⌋) (mitad
   central del clip), para expresar los bordes como cociente borde/centro en el reporte descriptivo.

Se reporta además dónde caen los fotogramas sin mano (primeros K, últimos K, centro), porque si
la mano suele faltar en los bordes, el rasgo 2 queda indefinido justo donde se necesita.

## Clips truncados artificialmente

De cada clip íntegro de longitud T se generan seis versiones: sin los primeros round(p·T)
fotogramas («cortada al inicio») y sin los últimos round(p·T) («cortada al final»), p ∈ {0,20;
0,35; 0,50}. Se recorta ANTES de remuestrear a 30 pasos. Como la normalización y la máscara son por
fotograma, recortar la secuencia procesada equivale a recortar antes de normalizar.

## Familia de reglas y ajuste (solo con los señantes de entrenamiento del pliegue)

Regla(K, rasgo): «cortada al inicio si rasgo_ini(K) supera su umbral; cortada al final si
rasgo_fin(K) supera el suyo». Para la altura la comparación es al revés (cortada si h < h*).

- Rasgos candidatos: `v_pose`, `v_manos`, `h`, y la combinación `v_pose ∨ h` (cortada si
  cualquiera de las dos dispara).
- Umbral de cada rasgo y borde: percentil 95 del rasgo sobre los clips ÍNTEGROS de los señantes
  de entrenamiento del pliegue (percentil 5 para la altura). Es decir, el punto de operación fijado
  es «5 % de falsos positivos esperados sobre clips íntegros de señantes vistos». Un rasgo NaN no
  dispara (la ventana no se marca como cortada por ese rasgo).
- Sensibilidad que se reporta sin cambiar la elección: percentiles 90 y 99.
- Selección de (K, rasgo) por pliegue: entre las 12 combinaciones, la de mayor detección MEDIA sobre
  los clips truncados (los seis niveles p × lado) de los señantes de ENTRENAMIENTO, con el umbral del
  percentil 95 ajustado sobre esos mismos señantes. Empate (diferencia < 0,01): la de menor K y,
  después, el rasgo en el orden `v_pose`, `h`, `v_pose ∨ h`, `v_manos` (los rasgos de la postura
  antes que los de la mano, porque están definidos aunque la mano no se detecte).
- El señante de prueba del pliegue solo se usa para EVALUAR la configuración elegida: falsos
  positivos sobre sus clips íntegros, detección sobre sus clips truncados por nivel y lado.
- La regla que se propone al director es la configuración elegida en la mayoría de los pliegues
  (si no hay mayoría, la de menor K). Los umbrales que se citen como candidatos serán los cinco
  valores por pliegue con su media; el umbral que se implementaría se recalcularía con los cinco
  señantes y quedaría como archivo candidato, nunca sobrescribiendo nada.

## Daño al modelo «manos» (solo se reporta; no decide la regla)

Por pliegue k, con `models/loso_2026-09-15_holistic_mascara_manos/pliegue_k/modelo.keras` y el
preprocesamiento exacto del entrenador (columnas de la representación «manos» del processed,
`resample_sequence` a 30, `add_velocity`), sobre las muestras de prueba del pliegue: top-1, top-3,
confianza media, tasa de rechazo con el umbral vigente 0,20 (aceptada si confianza ≥ 0,20, como el
prototipo) y exactitud entre las aceptadas, para el clip íntegro y para cada nivel y lado de recorte.
CONTROL previo: las predicciones sobre los clips íntegros deben reproducir exactamente el top-1
de `resultados_loso.json` por pliegue; si no, el script se detiene.

Con la regla elegida por pliegue se reporta, sobre los clips truncados del señante de prueba, qué
fracción rechazaría la regla, qué fracción rechazaría el umbral 0,20, y la exactitud de las que pasan
ambos filtros, para cuantificar el daño que la regla evitaría.

## Lo que NO se hace

No se toca ningún modelo, umbral vigente ni partición. No se implementa nada en `translator.py`
ni `realtime.py`. Toda la escritura va a `models/senas_cortadas_2026-09-22/`, a
`src/analisis_senas_cortadas.py` (nuevo) y a `tests/test_analisis_senas_cortadas.py` (nuevo).

