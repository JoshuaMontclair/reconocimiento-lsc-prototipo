# Caracterización del corpus LSC50 (actividad 1.b) y resumen de la bitácora de integridad (1.a)

Generado por `src/caracterizar_lsc50.py` el 2026-09-14T08:48:21-05:00 sobre 1000 de 1000 muestras esperadas. Cada cifra indica de qué archivo o cabecera sale. Los archivos de datos completos son `bitacora_integridad.json` (por muestra) y `caracterizacion.json`.

## 1. Identidad y versión del corpus

| Campo | Valor | Fuente |
|---|---|---|
| Nombre | LSC50: Colombian Sign Language Video and Inertial Measurement Dataset | anteproyecto 13_referencias.tex, \bibitem[19b]{ref19b}; README.md (Datasets) |
| Artículo descriptor | A. F. Flórez-Sierra, B. D. Solórzano, F. Segura-Quijano, Y. M. Cortés-Bello, L. Cubillos, L. F. Giraldo y C. Cifuentes-De la Portilla, Scientific Data, vol. 11, art. 1347, dic. 2024 | anteproyecto 13_referencias.tex, \bibitem{ref19}; README.md (Datasets) |
| DOI del artículo | 10.1038/s41597-024-04172-5 | ref19 |
| Licencia del artículo | CC BY-NC-ND 4.0 (rige sobre el texto y las figuras del artículo, no sobre los datos) | anteproyecto §1.4 |
| Depósito de datos | figshare, versión 1, publicado 2024-10-31 | ref19b |
| DOI del depósito | 10.6084/m9.figshare.27383016.v1 | ref19b |
| URL del depósito | https://figshare.com/articles/dataset/LSC50/27383016 | README.md |
| Código de los autores | https://github.com/BiomecanicaUniandes/LSC50 | README.md |
| Licencia de los datos | CC BY 4.0 | ref19b, anteproyecto §1.4 |
| Fecha de esta verificación | 2026-09-14T08:48:21-05:00 | reloj del sistema |
| Fecha de obtención | No registrada en el repositorio. Evidencia indirecta: los archivos conservan la fecha del depósito (2024-10-31); las carpetas se materializaron en disco en las fechas de «mtime_carpetas». PENDIENTE: que el estudiante consigne la fecha real de descarga desde figshare. | mtimes del sistema de archivos |
| Sumas de comprobación publicadas | figshare publica un MD5 por ZIP (IMU.zip, LANDMARKS.zip, VIDEOS.zip; API consultada el 2026-09-14), no por CSV ni por video. Los ZIP descargados NO están en disco (buscados en ['data'] hasta 3 niveles), así que el MD5 no se puede cotejar; lo que sí se verifica es que el contenido extraído está completo y es legible (esta bitácora). PENDIENTE: si el estudiante conserva los ZIP, correr con --zips RUTA --verificar-md5. Dentro del material descargado no viene ningún archivo de sumas (*.md5, *.sha1, *.sha256, *checksum*, MANIFEST*, *.sfv). | API de figshare; búsqueda en data/raw/LSC50 y de los ZIP en disco |

### 1.1 Sumas de comprobación que publica figshare (por ZIP) y cotejo con los ZIP en disco

Fuente: API pública de figshare, https://api.figshare.com/v2/articles/27383016, consultada el 2026-09-14. Nivel: un MD5 (computed_md5) por archivo ZIP del depósito (3 archivos); ninguna suma por CSV ni por video. Consulta en vivo: no solicitada (--consultar-figshare); se usa la instantánea del 14-sep-2026.

| ZIP | Bytes (figshare) | MD5 (figshare) | En disco | Bytes coinciden | MD5 coincide |
|---|---|---|---|---|---|
| IMU.zip | 1.608.738.643 | f94fc68d1ec5da41f9074e77d017a9c8 | no | — | — |
| LANDMARKS.zip | 2.770.068.538 | 996d3e9ee1d3fb39866ce395efd23214 | no | — | — |
| VIDEOS.zip | 6.247.307.413 | 26003107442257c7947352b18e3c2597 | no | — | — |

Archivos de sumas dentro del material descargado: ninguno.

## 2. Cotejo con lo que el corpus declara publicar y con lo que anticipa el anteproyecto

Fuente de lo anticipado: anteproyecto 05_justificacion.tex §1.3.4 (50 señas, 5 voluntarios y 4.000 videos publicados; 1.000 secuencias por modalidad y 4 repeticiones por derivación aritmética) y 09_metodologia.tex §2.2.2; lo declarado por el artículo, en «citas_del_articulo» → cifras. Fuente de lo verificado: nombres de archivo SSSS_VVVV_RRRR en LANDMARKS/BODY_LANDMARKS (cotejados con las otras 7 carpetas).

| Magnitud | Anticipado | Verificado |
|---|---|---|
| Señas (clases) | 50 | 50 |
| Señantes | 5 | 5 |
| Repeticiones por seña y señante | 4 | 4–4 |
| Secuencias | 1000 | 1000 |
| Videos .avi (declarado por el artículo) | 4000 | 4000 |
| Medidas inerciales (declarado por el artículo; fuera del alcance) | 250 | 250 filas en Timestamps.xlsx |

Archivos encontrados por carpeta (esperados 1000 en cada una):

| Carpeta | Archivos | Bytes |
|---|---|---|
| body | 1000 | 120.960.278 |
| lh | 1000 | 78.747.686 |
| rh | 1000 | 78.785.455 |
| face | 1000 | 5.902.603.160 |
| COLOR_BODY | 1000 | 2.617.149.802 |
| COLOR_FACE | 1000 | 1.134.657.230 |
| GRAY_DEPTH | 1000 | 2.752.336.822 |
| GRAY_IR | 1000 | 70.704.128 |

## 3. Vocabulario efectivo (50 señas con ID)

Fuente de la glosa del código: SIGN_VOCABULARY en src/grammar_mapper.py (verificado 50/50 contra la Tabla 2 del artículo, README 13-sep-2026). Fuente de la glosa publicada: IMU/INFO/Timestamps.xlsx del depósito figshare, columna «Sign», una hoja por voluntario (hojas: Volunteer 1, Volunteer 2, Volunteer 3, Volunteer 4, Volunteer 5).

| ID | Glosa (código) | Glosa publicada (Timestamps.xlsx) | Dinámica / estática |
|---|---|---|---|
| 0000 | gracias | Gracias | PENDIENTE (validación lingüística 1.d) |
| 0001 | buenos días | Buenos Días | PENDIENTE (validación lingüística 1.d) |
| 0002 | buenas tardes | Buenas Tardes | PENDIENTE (validación lingüística 1.d) |
| 0003 | buenas noches | Buenas Noches | PENDIENTE (validación lingüística 1.d) |
| 0004 | seña | Seña | PENDIENTE (validación lingüística 1.d) |
| 0005 | nombre | Nombre | PENDIENTE (validación lingüística 1.d) |
| 0006 | trabajar | Trabajar | PENDIENTE (validación lingüística 1.d) |
| 0007 | comer | Comer | PENDIENTE (validación lingüística 1.d) |
| 0008 | vivir | Vivir | PENDIENTE (validación lingüística 1.d) |
| 0009 | poco | Poco | PENDIENTE (validación lingüística 1.d) |
| 0010 | familia | Familia | PENDIENTE (validación lingüística 1.d) |
| 0011 | personas | Personas | PENDIENTE (validación lingüística 1.d) |
| 0012 | mujer | Mujer | PENDIENTE (validación lingüística 1.d) |
| 0013 | hombre | Hombre | PENDIENTE (validación lingüística 1.d) |
| 0014 | niño | Niño | PENDIENTE (validación lingüística 1.d) |
| 0015 | niña | Niña | PENDIENTE (validación lingüística 1.d) |
| 0016 | abuelo | Abuelo | PENDIENTE (validación lingüística 1.d) |
| 0017 | tío | Tío  | PENDIENTE (validación lingüística 1.d) |
| 0018 | hermano | Hermano | PENDIENTE (validación lingüística 1.d) |
| 0019 | hambre | Hambre | PENDIENTE (validación lingüística 1.d) |
| 0020 | feliz | Feliz | PENDIENTE (validación lingüística 1.d) |
| 0021 | contento | Contento | PENDIENTE (validación lingüística 1.d) |
| 0022 | triste | Triste | PENDIENTE (validación lingüística 1.d) |
| 0023 | aburrido | Aburrido | PENDIENTE (validación lingüística 1.d) |
| 0024 | bien | Bien | PENDIENTE (validación lingüística 1.d) |
| 0025 | mal | Mal | PENDIENTE (validación lingüística 1.d) |
| 0026 | ¿cómo estás? | ¿Cómo estas? | PENDIENTE (validación lingüística 1.d) |
| 0027 | más o menos | Más o menos | PENDIENTE (validación lingüística 1.d) |
| 0028 | sentir | Sentir | PENDIENTE (validación lingüística 1.d) |
| 0029 | juicioso | Jucioso | PENDIENTE (validación lingüística 1.d) |
| 0030 | hola | Hola | PENDIENTE (validación lingüística 1.d) |
| 0031 | adiós | Adiós | PENDIENTE (validación lingüística 1.d) |
| 0032 | por favor | Por favor | PENDIENTE (validación lingüística 1.d) |
| 0033 | con gusto | Con Gusto | PENDIENTE (validación lingüística 1.d) |
| 0034 | bienvenido | Bienvenido | PENDIENTE (validación lingüística 1.d) |
| 0035 | perdón | Perdón | PENDIENTE (validación lingüística 1.d) |
| 0036 | permiso | Permiso | PENDIENTE (validación lingüística 1.d) |
| 0037 | nunca | Nunca | PENDIENTE (validación lingüística 1.d) |
| 0038 | yo | Yo | PENDIENTE (validación lingüística 1.d) |
| 0039 | tú | Tú | PENDIENTE (validación lingüística 1.d) |
| 0040 | ustedes | Ustedes | PENDIENTE (validación lingüística 1.d) |
| 0041 | ¿qué? | ¿Qué? | PENDIENTE (validación lingüística 1.d) |
| 0042 | ¿cuándo? | ¿Cuándo? | PENDIENTE (validación lingüística 1.d) |
| 0043 | ¿dónde? | ¿Dónde? | PENDIENTE (validación lingüística 1.d) |
| 0044 | ¿cómo? | ¿Cómo? | PENDIENTE (validación lingüística 1.d) |
| 0045 | ¿por qué? | ¿Por qué? | PENDIENTE (validación lingüística 1.d) |
| 0046 | ¿quién? | ¿Quién? | PENDIENTE (validación lingüística 1.d) |
| 0047 | diferente | Diferente | PENDIENTE (validación lingüística 1.d) |
| 0048 | todos | Todos | PENDIENTE (validación lingüística 1.d) |
| 0049 | mucho | Mucho | PENDIENTE (validación lingüística 1.d) |

Discrepancias entre la glosa del código y la publicada (se reportan tal cual, para la validación 1.d):

| ID | Código | Publicado | Motivo |
|---|---|---|---|
| 0029 | juicioso | Jucioso | difiere tras normalizar (minúsculas, sin tildes, espacios) |

Nota sobre la Tabla 2 del artículo: Cotejo del revisor (14-sep-2026) con el texto completo del artículo en Europe PMC (PMC11655960): la Tabla 2 escribe la seña 0029 como «JUCIOSO», la misma grafía de Timestamps.xlsx; «juicioso» es la grafía del código. Las dos fuentes publicadas coinciden entre sí: la discrepancia es del código frente a ambas y va a 1.d.

Nota: Timestamps.xlsx trae además «Time Start (s)» y «Time End (s)» por seña y voluntario; su semántica (referencia temporal frente a la grabación IMU) no está documentada en el material descargado y no se usa aquí.

**Condición dinámica/estática por seña: PENDIENTE.** El anteproyecto (§1.4) declara el corpus como de señas dinámicas; la condición por seña se consigna tras la validación lingüística con la persona señante o intérprete (actividad 1.d). No se infiere aquí de los datos.

## 4. Muestras por clase, por señante y repeticiones

Fuente: nombres de archivo SSSS_VVVV_RRRR en LANDMARKS/BODY_LANDMARKS (cotejados con las otras 7 carpetas).

- Muestras por clase: 20–20 (balanceado).
- Muestras por señante: 0000: 200, 0001: 200, 0002: 200, 0003: 200, 0004: 200 (balanceado).
- Muestras por índice de repetición: 0000: 250, 0001: 250, 0002: 250, 0003: 250.
- Celdas clase × señante: 250 celdas con 4 muestras.

## 5. Fotogramas por modalidad

Fuente: filas de datos de cada CSV (sin cabecera) y avih.dwTotalFrames de cada AVI; DE = desviación estándar poblacional (numpy.std, ddof = 0), la misma convención que frame_stats de metadata.json.

| Modalidad | n | mín | máx | media | mediana | DE |
|---|---|---|---|---|---|---|
| CSV body | 1000 | 25 | 134 | 61.765 | 62 | 13.226 |
| CSV lh | 1000 | 24 | 134 | 61.72 | 62 | 13.289 |
| CSV rh | 1000 | 24 | 134 | 61.707 | 62 | 13.293 |
| CSV face | 1000 | 111 | 435 | 210.638 | 211 | 40.795 |
| CSV body/lh/rh truncados al mínimo (= preprocess_lsc50.py) | 1000 | 24 | 134 | 61.707 | 62 | 13.293 |
| Video COLOR_BODY | 1000 | 40 | 135 | 67.328 | 67 | 11.082 |
| Video COLOR_FACE | 1000 | 111 | 435 | 210.655 | 211 | 40.819 |
| Video GRAY_DEPTH | 1000 | 66 | 241 | 119.267 | 121 | 22.6 |
| Video GRAY_IR | 1000 | 66 | 241 | 119.444 | 121 | 22.672 |

Duración de los clips en segundos (fotogramas del video / fps medido de la cámara):

| Cámara | n | mín | máx | media | mediana | DE |
|---|---|---|---|---|---|---|
| COLOR_BODY | 1000 | 1.66667 | 5.625 | 2.805 | 2.79167 | 0.462 |
| COLOR_FACE | 1000 | 2.22 | 8.7 | 4.213 | 4.22 | 0.816 |
| GRAY_DEPTH | 1000 | 2.2 | 8.03333 | 3.976 | 4.03333 | 0.753 |
| GRAY_IR | 1000 | 2.2 | 8.03333 | 3.981 | 4.03333 | 0.756 |

## 6. Resolución, tasa de fotogramas y códec por cámara (medidos)

Fuente: cabeceras AVI (avih: dwWidth/dwHeight/dwMicroSecPerFrame; strh: dwRate/dwScale; strf: biCompression), una por archivo, 4 × N archivos.

| Cámara | Resolución (moda) | fps (moda) | Códec | Valores de fps observados | Resoluciones observadas |
|---|---|---|---|---|---|
| COLOR_BODY | 1920x1080 | 24.0 | XVID | {"24.0": 1000} | {"1920x1080": 1000} |
| COLOR_FACE | 640x424 | 50.0 | XVID | {"50.0": 1000} | {"640x424": 1000} |
| GRAY_DEPTH | 512x424 | 30.0 | XVID | {"30.0": 1000} | {"512x424": 1000} |
| GRAY_IR | 512x424 | 30.0 | XVID | {"30.0": 1000} | {"512x424": 1000} |

Cotejo con lo que declara el artículo (fuente: citas «registro» de ARTICULO_CITAS (Methods, Video data acquisition)):

| Cámara | Dispositivo (artículo) | fps declarado / medido | Resolución declarada / medida | Coincide | Todos los archivos en la moda |
|---|---|---|---|---|---|
| COLOR_BODY | Microsoft Kinect V2 (RGB) | 24 / 24.0 | 1920x1080 / 1920x1080 | sí | sí |
| COLOR_FACE | Nikon D5600 DSLR, objetivo AF-P DX NIKKOR 55 mm, f/3.5 | 50 / 50.0 | 640x424 / 640x424 | sí | sí |
| GRAY_DEPTH | Microsoft Kinect V2 (profundidad) | 30 / 30.0 | 512x424 / 512x424 | sí | sí |
| GRAY_IR | Microsoft Kinect V2 (infrarrojo) | 30 / 30.0 | 512x424 / 512x424 | sí | sí |

## 7. Desalineamiento rostro / cuerpo

Fuente: filas FACE_LANDMARKS / filas BODY_LANDMARKS por muestra; fotogramas COLOR_FACE / COLOR_BODY por muestra; fps medidos.

| Magnitud | n | mín | máx | media | mediana | DE |
|---|---|---|---|---|---|---|
| filas FACE / filas BODY | 1000 | 2.5319 | 9.88 | 3.478 | 3.3231 | 0.658 |
| fotogramas COLOR_FACE / COLOR_BODY | 1000 | 2.2609 | 4.6038 | 3.118 | 3.12235 | 0.203 |

Cociente de fps medidos COLOR_FACE / COLOR_BODY: 2.0833. FACE frente a su propio video (COLOR_FACE): 6 muestras con más fotogramas que filas, 994 iguales, 0 con menos.

Los landmarks faciales publicados proceden de otra cámara (COLOR_FACE) con otra tasa y otra resolución: no están alineados fotograma a fotograma con body/manos. Es la causa verificada de que el rostro quedara fuera del preprocesamiento de marzo (ESTADO_PROYECTO §3, RE1) y una razón más para la extracción propia sobre COLOR_BODY (D-01).

## 8. Señantes

Número de señantes: 5 (fuente: nombres de archivo SSSS_VVVV_RRRR en LANDMARKS/BODY_LANDMARKS (cotejados con las otras 7 carpetas)).

Condición tal como la publica el artículo descriptor, con sus dos criterios **sin conciliar**:

| Sección del artículo | Lo que publica |
|---|---|
| Methods | 2 voluntarios sordos y 3 oyentes («Two of them are deaf and three of them are not hearing impaired») |
| Data Records | 3 señantes nativos y 2 no nativos («5 volunteers: 3 native speakers and two non-native speakers») |

**Advertencia:** Los dos criterios no se corresponden entre sí (2 + 3 frente a 3 + 2) y el artículo no publica cuál de los cinco voluntarios (0000-0004) pertenece a cada grupo. Se registran ambos SIN conciliarlos, como exige la actividad 1.b del anteproyecto.

Otros datos que el artículo publica sobre ellos: Reclutados a través de la Asociación de Intérpretes de Lengua de Señas Colombiana-Español (Methods). Todos fluidos en LSC y mayores de 18 años en el momento de la captura (Methods). Los intérpretes que validaron las señas «are non-native speakers and are some of the participants that can be seen in the stored data» (Technical Validation). No publica sexo, edad individual, lateralidad ni qué voluntario (0000-0004) pertenece a cada grupo.

Fuente: Texto completo del artículo descriptor (ref19, doi 10.1038/s41597-024-04172-5) en Europe PMC, PMC11655960 (https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11655960/fullTextXML), consultado el 14-sep-2026. Citas literales en inglés; la sección se indica en cada una. Citas completas en «citas_del_articulo».

Citas literales del artículo sobre los señantes:

- [Methods] «Participants were recruited through the Association of Colombian-Spanish Sign Language Interpreters. Two of them are deaf and three of them are not hearing impaired.»
- [Methods] «All participants are fluent in Colombian Sign Language and were above 18 years old at the time of their participation in the data collection process.»
- [Data Records] «The total information in the dataset[14] is equally distributed in the number of classes (50 signs) and the number of participants involved (5 volunteers: 3 native speakers and two non-native speakers).»
- [Technical Validation (Sign validation)] «These LSC interpreters are non-native speakers and are some of the participants that can be seen in the stored data.»

Antropometría publicada por los autores (fuente: IMU/INFO/Volunteer_k.txt del depósito figshare (k = 1..5)):

| Archivo | back (from coccyx to cervical vertebrae C7) | height | humerus | radius | sampling frecuency |
|---|---|---|---|---|---|
| Volunteer_1 | 52cm | 176cm | 29cm | 27cm | 125Hz |
| Volunteer_2 | 46cm | 164cm | 28cm | 25cm | 125Hz |
| Volunteer_3 | 39cm | 156cm | 26cm | 22cm | 125Hz |
| Volunteer_4 | 38cm | 158cm | 26cm | 21cm | 125Hz |
| Volunteer_5 | 55cm | 165cm | 28cm | 26cm | 125Hz |

La correspondencia entre Volunteer_k (IMU) y el campo VVVV = 000(k−1) de los nombres de archivo es la presumible, pero el material descargado no la declara: no se afirma.

## 9. Condiciones de registro

Medidas aquí: resolucion_y_fps_por_camara: ver «camaras» (cabeceras AVI) y «camaras» → cotejo_declarado_vs_medido; duracion_de_los_clips: ver «fotogramas_por_modalidad».

Declaradas por el artículo (fuente: citas «escena» de ARTICULO_CITAS['registro'] (Methods, Video data acquisition); Texto completo del artículo descriptor (ref19, doi 10.1038/s41597-024-04172-5) en Europe PMC, PMC11655960 (https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11655960/fullTextXML), consultado el 14-sep-2026. Citas literales en inglés; la sección se indica en cada una.):

| Magnitud | Valor declarado |
|---|---|
| distancia_camaras_senante_cm | 200 |
| distancia_senante_fondo_cm | 10 |
| fondo | cortina azul de 3 m × 3 m |
| iluminacion | blanca y homogénea |
| montaje | DSLR y Kinect acoplados en una plataforma impresa en 3D sobre trípode con altura e inclinación ajustables |
| encuadre COLOR_FACE | rostro del señante; cámara a la altura de los ojos |
| encuadre COLOR_BODY | cuerpo desde las rodillas hasta la máxima extensión vertical de los brazos; Kinect 5 cm por debajo de la DSLR |
| encuadre GRAY_DEPTH | el del Kinect |
| encuadre GRAY_IR | el del Kinect |
| dispositivo COLOR_FACE | Nikon D5600 DSLR, objetivo AF-P DX NIKKOR 55 mm, f/3.5 |
| dispositivo COLOR_BODY | Microsoft Kinect V2 (RGB) |
| dispositivo GRAY_DEPTH | Microsoft Kinect V2 (profundidad) |
| dispositivo GRAY_IR | Microsoft Kinect V2 (infrarrojo) |

Citas literales sobre el registro:

- [Methods (Video data acquisition) · COLOR_FACE] «The first video corresponds to an RGB video of the face of the LSC speaker taken with a Nikon D5600 DSLR camera with an AF-P DX NIKKOR lens with a focal length of 55 millimeters. A diaphragm opening of 3.5 G, video that records at a rate of 50 frames per second (FPS) at a resolution of 640 by 424 pixels»
- [Methods (Video data acquisition) · COLOR_BODY] «The second type of video corresponds to an RGB capture of the LSC speaker’s body from the knees to a height of maximum vertical extension of the arms. This type of video is recorded at a rate of 24 FPS, and a resolution of 1920 by 1080 pixels»
- [Methods (Video data acquisition) · GRAY_DEPTH] «The third type corresponds to a grayscale video representing the scene’s depth. This video is recorded at a rate of 30 FPS at a resolution of 512 by 424 pixels»
- [Methods (Video data acquisition) · GRAY_IR] «This video is recorded at the same rate and resolution as the one representing the depth»
- [Methods (Video data acquisition) · COLOR_BODY, GRAY_DEPTH, GRAY_IR] «For RGB full body, depth, and IR videos, the Microsoft Kinect V2 is used, which has RGB sensors, infrared emission, and reception that work at the mentioned rates and resolutions.»
- [Methods (Video data acquisition) · escena] «Regarding the spatial distribution of the recording set, a group of cameras is located at a distance of 200 centimeters from the LSC speaker, which is 10 centimeters from a blue curtain of dimensions three by three meters.»
- [Methods (Video data acquisition) · escena] «The height of the Nikon d 5600 DSLR camera is adjusted to eye level for the LSC speaker, and the Microsoft Kinect V2 sits 5 centimeters below the DSLR camera, where both are docked on a 3D printed platform and mounted on a tripod that allows adjusting the height and inclination in the three axes.»
- [Methods (Video data acquisition) · escena] «The lighting must be white and homogeneous.»

No verificadas aquí: encuadre, distancia aparente, iluminación y fondo se toman del artículo y NO se verifican por inspección visual en esta corrida (no se decodifica ningún fotograma). PENDIENTE: inspección visual de una muestra por señante al hacer la extracción propia (1.c).

Lo que el artículo NO dice: (1) Qué versión de MediaPipe se usó. (2) Qué se hace con los fotogramas en los que MediaPipe no detecta (descartar o rellenar); tampoco por qué los CSV publicados tienen menos filas que fotogramas el video (N-02): la hipótesis de que fueran fotogramas sin detección quedó contradicha por la re-extracción propia (N-09) y la causa es desconocida. (3) Sumas de comprobación de los archivos del depósito (las MD5 están solo en la API de figshare, por ZIP). (4) Cuál de los cinco voluntarios (0000-0004) es sordo u oyente, nativo o no nativo; tampoco sexo, edad individual ni lateralidad. (5) Ni el número de repeticiones por seña y señante ni el de secuencias por modalidad: 4 y 1.000 son derivación aritmética (anteproyecto §1.3.4), que aquí se verifica contando archivos.

## 10. Diferencias entre landmarks publicados y video (N-02)

Fuente: DECISIONES_TESIS.txt N-02 (verificación manual del 13-sep-2026) reproducida por este script. Comparación: avih.dwTotalFrames de VIDEOS/COLOR_BODY/*.avi frente a filas de LANDMARKS/BODY_LANDMARKS/*.csv.

| Magnitud | Valor |
|---|---|
| Muestras con MÁS fotogramas en el video que filas en el CSV body | 749 |
| Muestras con el mismo número | 251 |
| Muestras con MENOS fotogramas que filas | 0 |
| Diferencia (sobre las positivas): máx / mediana / media | 51 / 5 / 7.427 |
| Muestras con body / mano izq. / mano der. de longitud distinta | 58 |
|   · con diferencia 1–2 (preprocess_lsc50.py las trunca al mínimo, sin registro) | 58 |
|   · con diferencia > 2 (preprocess_lsc50.py las descarta) | 0 |
| Filas todo-cero por modalidad | {"body": 0, "lh": 0, "rh": 0, "face": 0} |
| NaN por modalidad | {"body": 0, "lh": 0, "rh": 0, "face": 0} |
| Columna índice consecutiva 0..n−1 en todos los CSV | True |

Histograma de (fotogramas COLOR_BODY − filas body): {"0": 251, "1": 154, "2": 71, "3": 57, "4": 60, "5": 49, "6": 49, "7": 44, "8": 41, "9": 26, "10": 21, "11": 17, "12": 17, "13": 18, "14": 15, "15": 13, "16": 12, "17": 9, "18": 7, "19": 10, "20": 10, "21": 5, "22": 3, "23": 5, "24": 4, "25": 5, "26": 5, "27": 3, "28": 2, "29": 4, "30": 1, "31": 3, "33": 2, "34": 3, "36": 1, "38": 1, "40": 1, "51": 1}

Histograma de (máx − mín) entre body, lh y rh: {"0": 942, "1": 58}

**Hecho verificado:** Los CSV publicados no cubren todos los fotogramas del video y no contienen filas «vacías» (todo-cero); su columna índice está renumerada 0..n−1, así que no identifica qué fotogramas faltan.

**Causa del déficit: DESCONOCIDA.** Hipótesis de N-02: Que las filas que faltan sean los fotogramas sin detección de MediaPipe (DECISIONES_TESIS.txt, N-02, 13-sep-2026). No confirmada con los autores. Evidencia en contra (N-09): La re-extracción propia con Tasks API (DECISIONES_TESIS.txt, N-09, 14-sep-2026) detecta la postura en 61/61 fotogramas de 0025_0002_0001 y en 78/78 de 0049_0004_0003, con 0 y 2 fotogramas sin mano alguna, y aun así los CSV publicados tienen 4 filas menos en ambas: el déficit no se explica por fotogramas sin detección. Pendiente: Alineamiento monótono entre lo publicado y lo re-extraído con las METRICAS_FOTOGRAMA de la corrida completa (N-09). Hasta entonces se reporta: hecho verificado (déficit de filas) y causa desconocida. (Fuente: DECISIONES_TESIS.txt, entradas N-02 y N-09; este script solo reproduce las cifras de N-02.)

## 11. Resumen de la bitácora de integridad (1.a)

| Anomalía | Muestras |
|---|---|
| body_manos_longitud_distinta | 58 |
| duplicado:COLOR_BODY | 6 |
| duplicado:COLOR_FACE | 2 |
| duplicado:GRAY_DEPTH | 10 |
| duplicado:body | 6 |
| duplicado:face | 2 |
| duplicado:lh | 6 |
| duplicado:rh | 6 |
| truncada_por_preprocess | 58 |
| video_mas_fotogramas_que_filas_body | 749 |
| video_mas_fotogramas_que_filas_face | 6 |

- Muestras ausentes de la rejilla 50×5×4: 0
- Nombres fuera de la rejilla: 0
- Archivos ausentes: 0
- Muestras con algún archivo ilegible: 0
- Grupos de duplicados por carpeta (CSV: SHA-1 completo; video: criba por tamaño + SHA-1 de los primeros 65536 bytes y confirmación con SHA-1 completo): {"body": 3, "lh": 3, "rh": 3, "face": 1, "COLOR_BODY": 3, "COLOR_FACE": 1, "GRAY_DEPTH": 5, "GRAY_IR": 0}
- Candidatos de video refutados por el SHA-1 completo: ninguno

### 11.1 Duplicados exactos (archivos byte a byte idénticos bajo nombres distintos)

Fuente: SHA-1 del archivo completo: CSV de LANDMARKS y, para los videos, candidatos por tamaño + SHA-1 de los primeros 64 KiB confirmados con SHA-1 completo (bitacora_integridad.json → agregados).

| Muestras | Modalidades idénticas | Afecta al entrenamiento (body/lh/rh/COLOR_BODY) |
|---|---|---|
| 0000_0003_0000 = 0000_0003_0001 | face, COLOR_FACE, GRAY_DEPTH | no |
| 0002_0001_0000 = 0002_0001_0001 | body, lh, rh, COLOR_BODY | sí |
| 0008_0002_0000 = 0008_0002_0001 | body, lh, rh, COLOR_BODY | sí |
| 0016_0002_0000 = 0016_0002_0001 | body, lh, rh, COLOR_BODY | sí |
| 0017_0000_0000 = 0017_0000_0001 | GRAY_DEPTH | no |
| 0024_0002_0002 = 0024_0002_0003 | GRAY_DEPTH | no |
| 0041_0003_0000 = 0041_0003_0001 | GRAY_DEPTH | no |
| 0044_0004_0002 = 0044_0004_0003 | GRAY_DEPTH | no |

Muestras únicas para el entrenamiento (COLOR_BODY / body / lh / rh): 997 de 1000.

- Partición aleatoria_80_10_10 (1 bloque(s), 3 grupo(s)): 1 caso(s) repartidos entre conjuntos → única: {"0016_0002_0000": "train", "0016_0002_0001": "val"}
- Partición loso_5_pliegues (5 bloque(s), 3 grupo(s)): 1 caso(s) repartidos entre conjuntos → pliegue 4 (test = señante 0004): {"0016_0002_0000": "train", "0016_0002_0001": "val"}

**Hecho verificado:** Hay pares de muestras con nombre distinto (misma seña y señante, repeticiones consecutivas) cuyo archivo es byte a byte idéntico. La duplicación es por modalidad: un par puede tener idénticos COLOR_BODY y los tres CSV body/lh/rh y distintos COLOR_FACE y GRAY_IR, o al revés.

**Hipótesis no confirmada:** Error de copia al ensamblar el depósito (una toma publicada dos veces con dos números de repetición). No confirmado con los autores.

**Consecuencia:** Para el entrenamiento sobre COLOR_BODY/body/lh/rh el corpus tiene tantas secuencias únicas como indica «muestras_unicas_en_COLOR_BODY_body_lh_rh». Si un par queda repartido entre conjuntos de una partición, el modelo se evalúa sobre una copia exacta de una secuencia de entrenamiento: ver «cotejo_con_particiones». Qué hacer con ellos (excluir la copia o declararlo) es decisión para DECISIONES_TESIS.txt, no de este script.

## 12. Comprobaciones frente a la verificación manual del 13-sep-2026 (N-02)

| Comprobación | Obtenido | Esperado | OK |
|---|---|---|---|
| video_mas_fotogramas_que_filas | 749 | 749 | sí |
| video_igual_filas | 251 | 251 | sí |
| video_menos_fotogramas_que_filas | 0 | 0 | sí |
| diferencia_maxima | 51 | 51 | sí |
| diferencia_mediana | 5.0 | 5 | sí |
| body_manos_longitud_distinta | 58 | 58 | sí |

| Muestra | Fotogramas COLOR_BODY | Filas body | Filas todo-cero en manos | OK |
|---|---|---|---|---|
| 0000_0000_0000 | 59 | 59 | 0 | sí |
| 0025_0002_0001 | 61 | 57 | 0 | sí |
| 0049_0004_0003 | 78 | 74 | 0 | sí |

**Resultado global: todas las comprobaciones coinciden.**

## 13. Cotejo con data/processed/metadata.json

Fuente: data/processed/metadata.json (sample_order, frame_stats) frente a esta corrida.

- n_samples en metadata: 1000; sample_order igual a la rejilla ordenada: True.
- frame_stats en metadata: {"min": 24, "max": 134, "mean": 61.707, "median": 62.0, "std": 13.292522371619317}
- Recalculados aquí (mín de body/lh/rh por muestra): {"min": 24.0, "max": 134.0, "media": 61.707, "mediana": 62.0, "de": 13.293}
- Coinciden (mín, máx, media): True

## Entorno de la corrida

python: 3.12.13; numpy: 2.4.3; plataforma: macOS-26.6.2-arm64-arm-64bit; maquina: arm64; duracion_s: 56.7; con_hash: True; limite: None
