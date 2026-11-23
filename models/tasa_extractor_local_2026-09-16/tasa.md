# Tasa del extractor dentro y fuera de Google Drive (cabo de N-21, actividad 1.c)

Fecha de la corrida: 2026-09-16T10:49:18. Equipo: Apple M5 Max, 18 núcleos, 48.0 GB, macOS-26.6.2-arm64-arm-64bit. mediapipe 0.10.35, delegado CPU.

## Qué se midió

N-21 midió 15,0 ms por fotograma en la corrida completa frente a 11,3-11,6 ms en las pruebas cortas de N-20 y dejó como causa probable, sin verificar, la sincronización de Google Drive (el repositorio vive en CloudStorage y la corrida creó 5.000 archivos). Esta medición repite el extractor sobre una muestra leída desde Drive y desde una copia fuera de Drive, para separar el efecto del almacenamiento del efecto del extractor.

Extractor: el de D-05 sin cambios (`src/reextract_lsc50.py`: `crear_holistic`, `extraer_video`, `resumen_ms`), modelo `holistic_landmarker.task` SHA256 `e2dab61191e2dcd0…` verificado, las mismas `OPCIONES_HOLISTIC`, modo VIDEO, un landmarker nuevo por video. No se escribió ningún fotograma ni ninguna fila de puntos: los resultados se descartan en cuanto se cronometran.

Muestra: 60 videos, semilla 2026, 12 señas × 5 señantes (una repetición por par seña-señante); 3999 fotogramas y 149.6 MB por pasada.

## Resultados

| Condición | Orden | Almacenamiento | ms/fot. holistic (mediana / media ± DE) | ms/fot. decodificar (mediana) | ms/fot. total (media) | fot./s (media total) | fot./s (pared) |
|---|---|---|---|---|---|---|---|
| drive_1 | 1 | google_drive_cloudstorage | 11,32 / 11,35 ± 1,05 | 0,559 | 11,99 | 83,4 | 60,8 |
| local | 2 | disco_local_fuera_de_drive | 11,31 / 11,32 ± 0,79 | 0,550 | 11,95 | 83,7 | 62,1 |
| drive_2 | 3 | google_drive_cloudstorage | 11,38 / 11,39 ± 1,04 | 0,559 | 12,04 | 83,1 | 56,7 |

n por condición: 3999 fotogramas (60 videos). DE muestral (ddof=1).

### Media de `holistic` por bloques de 10 videos, en el orden de la pasada

N-21 vio esta serie subir de 12,4 a 16,0 ms a lo largo de sus 1.000 videos (primeros 50: 11,8; últimos 50: 16,1). Aquí, con 60 videos por pasada:

- `drive_1`: 11,39, 11,36, 11,39, 11,30, 11,34, 11,30 ms
- `local`: 11,32, 11,30, 11,37, 11,31, 11,31, 11,30 ms
- `drive_2`: 11,34, 11,39, 11,39, 11,48, 11,38, 11,35 ms

### Comparaciones (diferencia = primera condición − segunda)

| Comparación | Etapa | Δ media (ms) | Δ media (%) | Δ mediana (ms) | Δ pareada por video, media (ms) | videos con la 1.ª más lenta | p (signos) |
|---|---|---|---|---|---|---|---|
| drive_1 vs local | holistic | 0,031 | 0,27 | 0,007 | 0,029 | 39/60 | 0,0273 |
| drive_2 vs local | holistic | 0,071 | 0,63 | 0,062 | 0,071 | 42/60 | 0,0027 |
| drive_2 vs drive_1 | holistic | 0,040 | 0,35 | 0,055 | 0,042 | 33/60 | 0,5190 |
| drive_1 vs local | decodificar | 0,009 | 1,52 | 0,009 | 0,009 | 46/60 | < 0,0001 |
| drive_2 vs local | decodificar | 0,011 | 1,86 | 0,009 | 0,011 | 49/60 | < 0,0001 |
| drive_2 vs drive_1 | decodificar | 0,002 | 0,33 | 0,000 | 0,002 | 31/60 | 0,8974 |
| drive_1 vs local | total | 0,039 | 0,33 | 0,016 | 0,037 | 39/60 | 0,0273 |
| drive_2 vs local | total | 0,083 | 0,69 | 0,071 | 0,083 | 42/60 | 0,0027 |
| drive_2 vs drive_1 | total | 0,044 | 0,37 | 0,055 | 0,046 | 33/60 | 0,5190 |

## Contexto del sistema durante la medición

| Condición | Carga al empezar | Carga al terminar | CPU de Google Drive durante (s) | Núcleos medios de este proceso | Archivos ya en disco |
|---|---|---|---|---|---|
| drive_1 | 3.51 | 4.26 | 2,30 | 1,81 | 60/60 |
| local | 4.26 | 3.85 | 0,01 | 1,84 | 60/60 |
| drive_2 | 3.85 | 3.37 | 0,02 | 1,70 | 60/60 |

## Lectura

- Leer los videos desde Google Drive frente a leerlos desde disco local cambia la etapa `holistic` en 0.031 ms por fotograma (0.27 %, Drive más lento); la comparación pareada por video da p = 0.02734 en la prueba de los signos.
- La segunda pasada desde Drive (drive_2) frente a local da +0.071 ms por fotograma (42/60 videos más lentos desde Drive, p = 0.00267). Sale distinta de cero en la prueba de los signos, pero mide 0.071 ms, unas 49 veces menos que los 3,5 ms que N-21 tiene que explicar: detectable no es lo mismo que relevante, y parte de esa diferencia es deriva (punto siguiente).
- En la etapa donde de verdad se lee el archivo (`decodificar`) la diferencia es +0.009 ms por fotograma. Es ahí, y no en `holistic`, donde tendría que aparecer un efecto del almacenamiento.
- La diferencia total (+0.039 ms por fotograma) queda por debajo del umbral de 1.0 ms que se fijó como «relevante» antes de medir: con esta muestra y en estas condiciones NO se reproduce la brecha de 3,5 ms que separaba las pruebas cortas de N-20 (11,5 ms) de la corrida completa de N-21 (15,0 ms). La causa probable que N-21 dejó anotada —la sincronización de Drive— no queda confirmada por esta medición; tampoco queda descartada, porque aquí no se reproduce la escritura masiva de 5.000 archivos que la corrida de N-21 sí provocaba.
- Deriva: la repetición desde Drive al final (drive_2) difiere de la primera (drive_1) en +0.040 ms por fotograma en `holistic`. Esa cifra es la escala de lo que se mueve solo por el paso del tiempo y el calentamiento del equipo, y hay que compararla con la diferencia entre Drive y local antes de atribuirle nada al almacenamiento.
- Los procesos de Google Drive consumieron [2.3, 0.01, 0.02] segundos de CPU durante cada condición (en el orden ['drive_1', 'local', 'drive_2']). Con esos valores tan bajos, la sincronización estaba prácticamente quieta: la corrida de N-21 escribía 5.000 archivos y esta no escribe ninguno, así que la comparación de hoy aísla el efecto de LEER desde Drive, no el de sincronizar.
- Los videos de la muestra ya estaban descargados en el disco (100% de los bytes con bloques asignados; 0 archivos sin bloques). Leer «desde Drive» aquí significa leer a través del proveedor de archivos de DriveFS, no bajar nada de la red: ese es el escenario real del proyecto, pero conviene decirlo.
- Dentro de la primera pasada, la media de `holistic` por bloques de 10 videos fue [11.39, 11.36, 11.39, 11.3, 11.34, 11.3] ms (de 11.3 a 11.39). N-21 describió esa misma serie subiendo de 12,4 a 16,0 ms a lo largo de sus 1.000 videos; en 60 videos esa subida no aparece.
- El grueso de la distribución es el mismo en las tres condiciones (p90 de `holistic`: {'drive_1': 11.509, 'local': 11.481, 'drive_2': 11.621}). Los máximos sí difieren ({'drive_1': 54.78, 'local': 18.113, 'drive_2': 52.358}), pero con un solo máximo por condición y una sola repetición eso no distingue un efecto del almacenamiento de un pico cualquiera del sistema: una revisión independiente repitió la medición y el pico mayor le salió en la condición local. No se lee nada de esos máximos.
- Cifra del extractor fuera de Drive en esta corrida: 11.32 ms por fotograma en `holistic` (88.4 fot./s) y 11.95 ms en el bucle completo (83.7 fot./s).
- Sobre el tope de CPU: se pidieron 2 hilos a OMP/OpenBLAS/MKL, pero MediaPipe usa su propio grupo de hilos y no los obedece; el proceso ocupó [1.81, 1.84, 1.7] núcleos de media en cada condición. Es decir, el tope no estranguló al extractor y las cifras están en la misma escala que las de N-20 y N-21.

## Límites de esta medición

- Muestra de 60 videos (3999 fotogramas por pasada), no los 1.000 de N-21.
- Se pidieron 2 hilos a OMP/OPENBLAS/MKL porque el equipo podía estar grabando. MediaPipe no obedece esas variables (usa su propio grupo de hilos), así que el tope no estranguló al extractor; aun así, los ms absolutos de esta corrida se dan con esa salvedad y lo que de verdad se compara es la diferencia entre las tres condiciones, medidas seguidas y en las mismas condiciones.
- No se vacía la caché de disco entre condiciones (haría falta permiso de administrador y no se toca el equipo): tras la copia, los archivos de las dos condiciones están tibios en la caché del sistema, así que esta medición no dice qué pasaría con una lectura en frío.
- Los archivos de Drive ya estaban descargados en la caché local de DriveFS: 'desde Drive' significa a través del proveedor de archivos, no bajando de la red.
- La corrida de N-21 ESCRIBÍA 5.000 archivos dentro de Drive; esta solo lee. Si el coste de N-21 venía de la subida (CPU del proceso de sincronización), esta medición no lo reproduce y por tanto no puede confirmarlo ni descartarlo.
- Una sola repetición por condición: la deriva drive_1 → drive_2 es la única estimación de la variabilidad entre pasadas.
- Las dos rutas viven en el MISMO volumen APFS: no hay un disco ni un punto de montaje distinto para la carpeta de Drive, así que lo que se compara es el camino de lectura, no el soporte físico.
- La copia se hace con shutil.copy2 y el sistema de archivos es APFS: puede ser una copia por escritura diferida (clon), de modo que los bloques físicos se compartan con el original. Lo que sí cambia con seguridad es el CAMINO DE LECTURA (el proveedor de archivos de DriveFS frente al sistema de archivos normal), que es justo lo que se quería comparar; pero no se puede afirmar que sean bloques distintos en el disco.
- La prueba de los signos (dos colas, exacta) usa una decisión por video: si su media fue mayor o menor. Con 60 videos pareados detecta diferencias diminutas, así que un p bajo no significa que la diferencia importe; por eso se fijó ANTES de medir un umbral en ms por fotograma y la conclusión se lee contra ese umbral, no contra el p.

Cifras completas, muestra y entorno: `tasa_extractor_local_2026-09-16/tasa.json`.
