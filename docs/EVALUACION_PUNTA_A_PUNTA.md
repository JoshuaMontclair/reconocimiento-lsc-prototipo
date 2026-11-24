# Evaluación punta a punta de la cadena integrada sobre los videos del corpus

Fecha: 2026-11-23. Script: `src/evaluacion_punta_a_punta.py` (ver su docstring). Videos procesados: 997 de 997 (señantes 0000, 0004, 0001, 0002, 0003). Sin personas en la sesión: la entrada son los videos COLOR_BODY del corpus.

La cadena es la del prototipo (`src/translator.py`, sin cambios): `LandmarkExtractor` sobre cada fotograma reducido en memoria a 1280×720, ventana de `LSCTranslator`, `_preprocesar` (normalización, imputación, 30 pasos, velocidades), una llamada a `predict`, umbral 0,68, conversor «inventario» y búsqueda del enunciado en el índice de voz pregrabada (sin reproducir). Ningún fotograma se escribe a disco.

## Modelos LOSO (cada video lo clasifica solo un modelo que no vio a su señante)

Media de 3 semillas por pliegue; global = media de los 5 pliegues (como el 0,677 de la tesis).

| fuente | ventana | top-1 | top-3 | rechazo (0,68) | de él: manos | exactitud aceptadas | voz encontrada | aviso de bordes |
|---|---|---|---|---|---|---|---|---|
| entrenamiento (probabilidades.npz) | clip | 0,677 | 0,847 | 35,8 % | 0,0 % | 0,836 | — | — |
| almacenados | clip | 0,677 | 0,847 | 35,8 % | 0,0 % | 0,836 | 100,0 % | 15,2 % |
| almacenados | v2_5s | 0,679 | 0,850 | 34,3 % | 0,0 % | 0,831 | 100,0 % | 32,9 % |
| almacenados | v2_5s_sin_relleno | 0,678 | 0,853 | 33,1 % | 0,0 % | 0,828 | 100,0 % | 34,0 % |
| video_1280x720 | clip | 0,673 | 0,845 | 36,1 % | 0,0 % | 0,839 | 100,0 % | 14,4 % |
| video_1280x720 | v2_5s | 0,677 | 0,852 | 35,5 % | 0,0 % | 0,835 | 100,0 % | 32,5 % |
| video_1280x720 | v2_5s_sin_relleno | 0,675 | 0,852 | 34,1 % | 0,0 % | 0,831 | 100,0 % | 33,3 % |
| video_1920x1080 | clip | 0,677 | 0,847 | 35,8 % | 0,0 % | 0,836 | 100,0 % | 15,2 % |
| video_1920x1080 | v2_5s | 0,679 | 0,850 | 34,3 % | 0,0 % | 0,831 | 100,0 % | 33,1 % |
| video_1920x1080 | v2_5s_sin_relleno | 0,678 | 0,853 | 33,1 % | 0,0 % | 0,828 | 100,0 % | 34,0 % |

Top-1 por pliegue (media de semillas):

| configuración | pliegue 0 (señante 0000) | pliegue 1 (señante 0001) | pliegue 2 (señante 0002) | pliegue 3 (señante 0003) | pliegue 4 (señante 0004) |
|---|---|---|---|---|---|
| entrenamiento (probabilidades.npz) · clip | 0,488 | 0,611 | 0,721 | 0,770 | 0,795 |
| almacenados · clip | 0,488 | 0,611 | 0,721 | 0,770 | 0,795 |
| almacenados · v2_5s | 0,508 | 0,593 | 0,714 | 0,773 | 0,808 |
| almacenados · v2_5s_sin_relleno | 0,507 | 0,593 | 0,716 | 0,767 | 0,807 |
| video_1280x720 · clip | 0,475 | 0,606 | 0,722 | 0,763 | 0,798 |
| video_1280x720 · v2_5s | 0,505 | 0,593 | 0,707 | 0,770 | 0,810 |
| video_1280x720 · v2_5s_sin_relleno | 0,505 | 0,591 | 0,710 | 0,765 | 0,805 |
| video_1920x1080 · clip | 0,488 | 0,611 | 0,721 | 0,770 | 0,795 |
| video_1920x1080 · v2_5s | 0,508 | 0,593 | 0,714 | 0,773 | 0,808 |
| video_1920x1080 · v2_5s_sin_relleno | 0,507 | 0,593 | 0,716 | 0,767 | 0,807 |

Rechazo con 0,68 y exactitud entre aceptadas por pliegue, punta a punta (clip · 2,5 s):

| pliegue | rechazo clip | exactitud aceptadas clip | rechazo 2,5 s | exactitud aceptadas 2,5 s | top-3 clip | top-3 2,5 s |
|---|---|---|---|---|---|---|
| 0 | 63,5 % | 0,790 | 57,3 % | 0,768 | 0,662 | 0,673 |
| 1 | 34,2 % | 0,759 | 33,5 % | 0,748 | 0,804 | 0,809 |
| 2 | 37,0 % | 0,909 | 38,0 % | 0,908 | 0,899 | 0,919 |
| 3 | 24,3 % | 0,860 | 27,5 % | 0,865 | 0,923 | 0,925 |
| 4 | 21,7 % | 0,878 | 21,3 % | 0,886 | 0,935 | 0,935 |

Por semilla (top-1, media de pliegues): referencia_npz_clip: 0,688 / 0,658 / 0,685; video_1280x720|clip: 0,688 / 0,651 / 0,680; video_1280x720|v2_5s: 0,695 / 0,664 / 0,672.

## Concordancia con la medida fuera de línea (mismo video, mismo modelo)

| comparación | n | misma top-1 | mismo acierto | misma decisión con 0,68 | acierta solo punta a punta | acierta solo la referencia | |Δp| clase ref. (mediana) | máx |Δp| (mediana) |
|---|---|---|---|---|---|---|---|---|
| loso: almacenados_vs_entrenamiento|clip | 2991 | 100,0 % | 100,0 % | 100,0 % | 0 | 0 | 0,000 | 0,000 |
| loso: video_1280x720_vs_entrenamiento|clip | 2991 | 95,8 % | 97,5 % | 95,4 % | 32 | 44 | 0,010 | 0,012 |
| loso: video_1280x720_vs_almacenados|clip | 2991 | 95,8 % | 97,5 % | 95,4 % | 32 | 44 | 0,010 | 0,012 |
| loso: video_1920x1080_vs_entrenamiento|clip | 2991 | 100,0 % | 100,0 % | 100,0 % | 0 | 0 | 0,000 | 0,000 |
| loso: video_1920x1080_vs_almacenados|clip | 2991 | 100,0 % | 100,0 % | 100,0 % | 0 | 0 | 0,000 | 0,000 |
| loso: video_1280x720_vs_video_1920x1080|clip | 2991 | 95,8 % | 97,5 % | 95,4 % | 32 | 44 | 0,010 | 0,012 |
| loso: video_1280x720_vs_almacenados|v2_5s | 2991 | 95,4 % | 97,2 % | 95,9 % | 38 | 45 | 0,010 | 0,012 |
| loso: video_1920x1080_vs_almacenados|v2_5s | 2991 | 100,0 % | 100,0 % | 100,0 % | 0 | 0 | 0,000 | 0,000 |
| loso: video_1280x720_vs_video_1920x1080|v2_5s | 2991 | 95,4 % | 97,2 % | 95,9 % | 38 | 45 | 0,010 | 0,012 |
| loso: video_1280x720_vs_almacenados|v2_5s_sin_relleno | 2991 | 95,7 % | 97,4 % | 96,2 % | 35 | 42 | 0,010 | 0,011 |
| loso: video_1920x1080_vs_almacenados|v2_5s_sin_relleno | 2991 | 100,0 % | 100,0 % | 100,0 % | 0 | 0 | 0,000 | 0,000 |
| loso: video_1280x720_vs_video_1920x1080|v2_5s_sin_relleno | 2991 | 95,7 % | 97,4 % | 96,2 % | 35 | 42 | 0,010 | 0,011 |
| desplegado: almacenados_vs_entrenamiento|clip | 100 | 100,0 % | 100,0 % | 100,0 % | 0 | 0 | 0,000 | — |
| desplegado: video_1280x720_vs_entrenamiento|clip | 100 | 96,0 % | 96,0 % | 99,0 % | 3 | 1 | 0,001 | — |
| desplegado: video_1280x720_vs_almacenados|clip | 100 | 96,0 % | 96,0 % | 99,0 % | 3 | 1 | 0,001 | 0,001 |
| desplegado: video_1920x1080_vs_entrenamiento|clip | 100 | 100,0 % | 100,0 % | 100,0 % | 0 | 0 | 0,000 | — |
| desplegado: video_1920x1080_vs_almacenados|clip | 100 | 100,0 % | 100,0 % | 100,0 % | 0 | 0 | 0,000 | 0,000 |
| desplegado: video_1280x720_vs_video_1920x1080|clip | 100 | 96,0 % | 96,0 % | 99,0 % | 3 | 1 | 0,001 | 0,001 |
| desplegado: video_1280x720_vs_almacenados|v2_5s | 100 | 98,0 % | 98,0 % | 98,0 % | 2 | 0 | 0,001 | 0,001 |
| desplegado: video_1920x1080_vs_almacenados|v2_5s | 100 | 100,0 % | 100,0 % | 100,0 % | 0 | 0 | 0,000 | 0,000 |
| desplegado: video_1280x720_vs_video_1920x1080|v2_5s | 100 | 98,0 % | 98,0 % | 98,0 % | 2 | 0 | 0,001 | 0,001 |
| desplegado: video_1280x720_vs_almacenados|v2_5s_sin_relleno | 100 | 98,0 % | 98,0 % | 98,0 % | 2 | 0 | 0,001 | 0,001 |
| desplegado: video_1920x1080_vs_almacenados|v2_5s_sin_relleno | 100 | 100,0 % | 100,0 % | 100,0 % | 0 | 0 | 0,000 | 0,000 |
| desplegado: video_1280x720_vs_video_1920x1080|v2_5s_sin_relleno | 100 | 98,0 % | 98,0 % | 98,0 % | 2 | 0 | 0,001 | 0,001 |

Punta a punta frente a entrenamiento, clip completo, por pliegue: pliegue 0: 94,8 %; pliegue 1: 95,6 %; pliegue 2: 96,3 %; pliegue 3: 95,3 %; pliegue 4: 97,0 %.

## Puntos característicos: extracción del video frente a la almacenada

| fuente | fotogramas | videos con T distinto | discrepa detección mano izq. | mano der. | postura | |Δx|+|Δy| muñeca izq. (mediana de videos) | der. |
|---|---|---|---|---|---|---|---|
| video_1280x720 | 67145 | 0 | 3,07 % | 1,75 % | 0,00 % | 0,00150 | 0,00112 |
| video_1920x1080 | 67145 | 0 | 0,00 % | 0,00 % | 0,00 % | 0,00000 | 0,00000 |

## Modelo desplegado (solo los videos de la prueba aleatoria, que no vio)

| fuente | ventana | n | top-1 | top-3 | rechazo (0,68) | exactitud aceptadas |
|---|---|---|---|---|---|---|
| entrenamiento (predicciones.json) | clip | 100 | 0,940 | 0,980 | 6,0 % | 0,968 |
| almacenados | clip | 100 | 0,940 | 0,980 | 6,0 % | 0,968 |
| almacenados | v2_5s | 100 | 0,940 | 0,990 | 5,0 % | 0,968 |
| almacenados | v2_5s_sin_relleno | 100 | 0,940 | 0,990 | 5,0 % | 0,968 |
| video_1280x720 | clip | 100 | 0,960 | 0,990 | 5,0 % | 0,968 |
| video_1280x720 | v2_5s | 100 | 0,960 | 1,000 | 5,0 % | 0,968 |
| video_1280x720 | v2_5s_sin_relleno | 100 | 0,960 | 1,000 | 5,0 % | 0,968 |
| video_1920x1080 | clip | 100 | 0,940 | 0,980 | 6,0 % | 0,968 |
| video_1920x1080 | v2_5s | 100 | 0,940 | 0,990 | 5,0 % | 0,968 |
| video_1920x1080 | v2_5s_sin_relleno | 100 | 0,940 | 0,990 | 5,0 % | 0,968 |

## Voz pregrabada y conversor (punta a punta, modelos LOSO y desplegado)

- clip: 2005 ventanas aceptadas; enunciado presente en 2005; WAV pregrabado encontrado en 2005; respaldo del conversor en 0; 50 enunciados distintos.
- v2_5s: 2023 ventanas aceptadas; enunciado presente en 2023; WAV pregrabado encontrado en 2023; respaldo del conversor en 0; 50 enunciados distintos.

## Tiempos

- video_1280x720: extracción (BGR→RGB + extract) mediana 10,0 ms, media 10,2 ms, p90 10,3 ms, p99 16,5 ms sobre 67145 fotogramas; lectura del video y reducción mediana 1,6 ms; total de la etapa 1164.8 s.
- video_1920x1080: extracción (BGR→RGB + extract) mediana 11,7 ms, media 11,7 ms, p90 11,8 ms, p99 16,7 ms sobre 67145 fotogramas; lectura del video y reducción mediana 0,3 ms; total de la etapa 1185.8 s.
- Cadena tras la extracción (todas las ventanas, todos los modelos y fuentes): 403.3 s.
- Tiempo total: 2763.9 s (46,1 min). Carga del equipo (1 min) al empezar 4.58, al terminar 3.96.

## Límites

- Son las personas, la cámara, la iluminación y el fondo del corpus: no es una persona nueva ni una cámara distinta. No sustituye la sesión con personas (Bloque 1).
- El fotograma de 1280×720 se obtiene reduciendo el video de 1920×1080; una cámara a 1280×720 tiene su propia óptica, compresión y ruido.
- Las marcas de tiempo del modo VIDEO son las del video (24 fps), no el reloj real; un HolisticLandmarker nuevo por video. La ventana de 2,5 s abre en el primer fotograma del clip, en reposo, como pide el protocolo del prototipo; no se simula a la persona que tarda en empezar.
- Con clips de menos de 60 fotogramas la ventana de 2,5 s se completa con el último fotograma (persona quieta).
