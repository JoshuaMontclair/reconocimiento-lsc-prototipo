# Protocolo experimental de la evaluación del prototipo (actividad 4.a)

**Estado: BORRADOR para discutir con el director. No está aprobado y no autoriza a ejecutar ninguna sesión.**
Trabajo de grado 281-2026-1 · Preparado el 15 de septiembre de 2026 · Sustituye a nada; es la primera versión.

Este documento concreta lo que el anteproyecto encarga a la actividad 4.a: el escenario de prueba, los
ejecutantes de cada bloque, las repeticiones, la presentación de la glosa y el registro de la variante, la
cronometría de la latencia, la caracterización del retardo de audio y las métricas que se reportarán
[MET 472-493 · §2.2.8, 4.a]. Donde el anteproyecto fija algo, aquí se cita; donde lo deja abierto, el
valor no se inventa: va a la sección 12, «POR DECIDIR CON EL DIRECTOR», con opciones.

---

## 0. Cómo leer este documento

### 0.1 Convención de citas

Cada afirmación normativa lleva entre corchetes el archivo y las líneas que la sostienen.

| Prefijo | Archivo | Sección del anteproyecto |
|---|---|---|
| `MET` | `Anteproyecto_FINAL_LaTeX/09_metodologia.tex` | §2.2 (2.2.1 tipo de estudio … 2.2.6 manejo de errores, 2.2.7 protocolo de evaluación, 2.2.8 actividades) |
| `ALC` | `Anteproyecto_FINAL_LaTeX/06_alcances.tex` | §1.4 |
| `ETI` | `Anteproyecto_FINAL_LaTeX/07_etica.tex` | §1.5 |
| `REC` | `Anteproyecto_FINAL_LaTeX/12_recursos.tex` | §2.5 (2.5.1 humanos, 2.5.2 técnicos, 2.5.3 presupuesto) |
| `OBJ` | `Anteproyecto_FINAL_LaTeX/04_objetivos.tex` | §1.2 (objetivos y tabla de trazabilidad) |
| `RES` | `Anteproyecto_FINAL_LaTeX/10_resultados.tex` | §2.3 (resultados esperados RE1-RE7) |
| `PRO` | `Anteproyecto_FINAL_LaTeX/03_problema.tex` | §1.1.2 (sistematización) |
| `CRO` | `Anteproyecto_FINAL_LaTeX/11_cronograma.tex` | §2.4 |
| `REG` | `Tesis/DECISIONES_TESIS.txt` (registro de decisiones) | entradas D-nn y N-nn |

Ejemplo: [MET 215-234 · §2.2.7] remite a las líneas 215 a 234 de `09_metodologia.tex`, dentro de §2.2.7.

Las citas de código remiten a `Tesis/src/` y nombran el archivo y la función, el argumento de línea de
órdenes o la clave del registro (por ejemplo, `translator.py`, `_registro_ensayo`), sin número de línea,
porque las líneas se mueven con cada cambio del código. Todas se comprobaron contra el código el 15-sep-2026.

### 0.2 Qué tipo de enunciado es cada cosa

- **Normativo**: lo fija el anteproyecto o una decisión registrada (D-nn). Lleva cita y no se discute aquí.
- **Propuesta del borrador**: procedimiento operativo que el anteproyecto no detalla y que este documento
  sugiere sin fijar cifras. Se marca como *Propuesta*.
- **POR DECIDIR (PD-nn)**: parámetro abierto; se decide con el director antes de la primera sesión.
- **Medido en esta revisión**: dato comprobado el 15-sep-2026 al preparar el borrador (comando y
  resultado en el informe de entrega). Todo lo demás procede de la fuente citada y no se volvió a medir.

---

## 1. Objetivo y preguntas

### 1.1 Objetivo

Definir, **antes** de ejecutar las pruebas, el procedimiento con el que se evalúa el sistema integrado en
calidad del reconocimiento, tiempo de respuesta y vocabulario efectivamente soportado. Responde a la
pregunta P6 [PRO 194-196 · §1.1.2], al objetivo específico OE4 [OBJ 60-61 · §1.2.2] y a los resultados
RE5 y RE6 [OBJ 167-179; RES 39-76 · §2.3]. El compromiso del trabajo es el rigor del procedimiento, no la
superación de una cifra [MET 11-15 · §2.2.1; MET 352-358 · §2.2.7].

### 1.2 Preguntas por resultado esperado

| Id | Resultado | Pregunta que el protocolo debe permitir responder | Nivel | Fuente normativa | Estado |
|---|---|---|---|---|---|
| Q1 | RE5 | ¿Qué top-1, top-3, precision, recall y F1, con matriz de confusión, alcanza el modelo con señantes **vistos** (partición aleatoria) y **no vistos** (LOSO)? | offline | [MET 276-280; MET 503-505; RES 40-43] | Ejecutada (sección 2.1) |
| Q2 | RE6 | ¿Qué diferencia hay entre ambos esquemas y cómo varía por pliegue según el señante excluido? | offline | [MET 522-525 · 4.e; RES 72-76] | Ejecutada; falta el reporte 4.e |
| Q3 | RE5 | ¿Qué exactitud en línea se obtiene cuando ejecuta las señas una persona usuaria fluida de la LSC ausente del corpus, sobre la totalidad de las ejecuciones y sobre las ejecuciones sin variante declarada, y con cuántas variantes y rechazos por umbral? | online, Bloque 1 | [MET 215-234; RES 43-47] | Pendiente de 0.c |
| Q4 | RE5 | ¿Cuánto valen L_total, sus seis etapas, L_resp y W, qué etapa domina, y cuánto valen el costo del extractor por fotograma y la tasa efectiva del bucle? | online, Bloques 1 y 2 | [MET 282-326; MET 505-513; RES 47-55] | Instrumentado (N-11, N-34); sin sesión válida |
| Q5 | RE5, RE3 | ¿Cómo se comporta el prototipo ante los casos límite de §2.2.6, y cuántos rechazos de cada tipo produce? | online, Bloque 2 | [MET 183-198; MET 247-262; RES 28-30; RES 55-57] | Pendiente |
| Q6 | RE5 | ¿Qué retardo constante añaden el controlador y el dispositivo de audio entre la orden de reproducción y el sonido? | caracterización única | [MET 326-331; MET 482-484; MET 513-517] | Pendiente |
| Q7 | RE5 | ¿Qué tamaño de vocabulario soporta efectivamente el prototipo? | ambos | [MET 517-518; RES 58-59] | Definición en PD-26 |
| Q8 | RE3 | ¿Qué tasa de procesamiento del extractor se observa sobre el flujo en vivo de la cámara? | online | [MET 116-123 · §2.2.3] | Se obtiene de Q4 |

### 1.3 Lo que este protocolo no hace

- No fija umbral de éxito en exactitud, latencia ni vocabulario [MET 352-358; ALC 98-101; OBJ 202-204].
- No valida el prototipo con usuarios finales en escenarios reales [ALC 152-153] ni opera en entornos no
  controlados [ALC 150-151].
- No convoca a ningún ejecutante distinto de la persona vinculada (Bloque 1) y del autor (Bloque 2) y no
  almacena video de ninguna prueba [ALC 76-77; ETI 48-53; MET 265].
- No compara la latencia total con cifras publicadas; solo el costo por fotograma y la tasa efectiva se
  sitúan frente a la literatura [RES 66-69; MET 321-324]. No compara la exactitud del Bloque 1 con la
  literatura [MET 268-269].
- No evalúa el modo oración del prototipo: el reconocimiento se limita a señas aisladas [ALC 7], y las
  etapas completas hasta t5 solo existen en el modo aislado (`translator.py`, `_inferir_y_emitir`: en modo
  oración la ventana aceptada solo acumula su glosa y no sella t4 ni t5).

---

## 2. Dos niveles de evaluación, declarados por separado

La evaluación distingue un nivel *offline*, sobre el conjunto de prueba ya extraído en puntos
característicos, y un nivel *online*, sobre el prototipo completo en funcionamiento [MET 202-204]. Los
resultados de cada nivel se reportan por separado y nunca se presentan como equivalentes [MET 485-493].

### 2.1 Evaluación offline (ya ejecutada)

**Quién ejecuta las señas.** Los cinco señantes del corpus LSC50, en su momento. Solo en este nivel tienen
sentido «señantes vistos» y «señantes no vistos» [MET 207-210].

**Esquemas** [MET 138-147 · §2.2.4]:
- *Partición aleatoria a nivel de muestra*: evaluación **dependiente del señante** (señantes vistos).
- *Partición independiente del señante*: validación cruzada dejando un señante fuera, cinco pliegues, con
  media y dispersión entre pliegues: evaluación **independiente del señante** (señantes no vistos).

**Condiciones vigentes** (registro): extractor único Holistic Landmarker de mediapipe 0.10.35 [REG
577-647 · D-05; REG 736-750 · N-19]; 997 muestras y validación de 2 por seña [REG 676-700 · D-06]; regla de
mano ausente «mascara» [REG 850-868 · D-07; REG 871-898 · N-23]; representación «manos» [REG 1003-1042 ·
D-08; REG 1104-1144 · N-30]. La selección de modelos se hace solo por validación [MET 147-149].

**Resultados vigentes** (recalculados en esta revisión desde
`models/loso_2026-09-15_holistic_mascara_manos/resultados_loso.json`; coinciden con N-30):

| Esquema | Condición | Prueba | top-1 | top-3 | F1 macro |
|---|---|---|---|---|---|
| LOSO, 5 pliegues | independiente del señante | 997 muestras (200 / 199 / 198 / 200 / 200) | 0,580 ± 0,086 | 0,770 ± 0,089 | 0,537 ± 0,093 |
| Aleatoria 797 / 100 / 100 | dependiente del señante | 100 muestras | 0,88 | 0,98 | 0,871 |

Desviación estándar muestral entre pliegues (ddof = 1). Top-1 por pliegue (señante excluido): 0000 0,460;
0001 0,533; 0002 0,606; 0003 0,615; 0004 0,685. Cada pliegue y la corrida aleatoria guardan
`matriz_confusion.json`, `classification_report.json` y `predicciones.json` (comprobado en esta revisión).
El modelo que usa el prototipo es el canónico entrenado con la partición aleatoria (semilla 2026,
seleccionado por validación): en su prueba única, top-1 0,88, top-3 0,98, F1 0,871 [REG 1171-1181 · N-31].

**Resultado por señante excluido.** El detalle de cada pliegue (top-1, top-3 y F1 macro, con errores, señas
con recall 0 y confianza) está en `models/analisis_errores_manos_2026-09-15/analisis.md`, sección 3 «Por
señante» (cifras en `analisis.json`, clave `por_senante`). Top-1 / top-3 / F1 macro: 0000 0,460 / 0,650 /
0,421; 0001 0,533 / 0,714 / 0,463; 0002 0,606 / 0,788 / 0,573; 0003 0,615 / 0,830 / 0,585; 0004 0,685 /
0,870 / 0,645 (comprobado en esta revisión en `analisis.json`).

**Efecto del umbral de confianza sobre las métricas offline** (exigido por [MET 187-189]; de
`models/umbral_D09_2026-09-15.json`, comprobado en esta revisión): con el umbral 0,20, en la prueba LOSO
se rechaza el 16,0 % de las 997 muestras y la exactitud entre las aceptadas es 0,645 (540 correctas y 297
incorrectas); con 0,50 se rechaza el 45,9 % y la exactitud entre aceptadas es 0,761 [REG 1230-1258 · N-33].
Las cifras de top-1 y top-3 no dependen del umbral [REG 1221-1222 · D-09]. Por señante excluido, con 0,20 el
rechazo va de 10,5 % (0003) a 29,0 % (0000); la tabla completa, con el error entre las aceptadas, está en
`analisis.md`, sección 5 «Calibración y umbral» (`analisis.json`, `calibracion.umbral.por_senante`;
comprobado en esta revisión).

**Lo que falta en el nivel offline, fuera de 4.a pero necesario para 4.d y 4.e**: la línea base clásica
(SVM, Random Forest o MLP) que exige [MET 131-136; MET 519-521] no existe todavía (búsqueda sin resultados
en `src/` y `scripts/`; pendiente anotado en [REG 431-432 · N-10]); y el reporte comparativo por pliegue con
la condición del señante excluido [MET 522-525].

### 2.2 Evaluación online (Bloques 1 y 2)

Se organiza en dos bloques con ejecutante, propósito y métricas distintos; esa decisión tiene su sede única
en §2.2.7 y fue adoptada por el director [MET 210-213].

| | Bloque 1: ejecución competente | Bloque 2: carga y casos límite |
|---|---|---|
| Ejecutante | Persona sorda señante o intérprete de LSC vinculada (actividad 0.c) [MET 215-217] | El autor del trabajo [MET 247-248] |
| Condición del ejecutante | Sexto señante, estrictamente no visto [MET 216-217] | Oyente, no usuario nativo de la LSC [MET 255-256; ALC 75-76] |
| Propósito | Única medición de exactitud en línea [MET 222-223] | Acumular ejecuciones para la latencia y ejercitar §2.2.6 [MET 248-251] |
| Exactitud | Sí, por duplicado [MET 229-233] | **Ninguna cifra** [MET 259-262; RES 55-58] |
| Latencia | Sí (aporta mediciones) [RES 55] | Sí [MET 248-249; RES 55] |
| Vocabulario | Las 50 señas [MET 223] | No fijado (PD-02, PD-17) |
| Requiere a la persona | Sí; sesiones presenciales y en tiempo real [REC 142-146] | No; ejecutable desde la integración [MET 262-263] |
| Video | No se almacena [MET 265; ETI 50-53] | No se almacena [MET 265] |

### 2.3 Regla de separación en el reporte

| Magnitud | Offline aleatoria | Offline LOSO | Online Bloque 1 | Online Bloque 2 |
|---|---|---|---|---|
| top-1, top-3, P, R, F1, matriz | Sí (dependiente) | Sí (independiente; media y dispersión) | Sí, por duplicado | **No** |
| Rechazos por umbral | Sí (efecto, N-33) | Sí (efecto, N-33) | Sí (número y proporción) | Sí (conteo) |
| Variantes declaradas | No aplica | No aplica | Sí (número y proporción) | No aplica |
| Latencias y etapas | No aplica | No aplica | Sí | Sí |
| Casos límite | No aplica | No aplica | Solo los que ocurran | Sí (ejercitados) |

Los resultados del Bloque 1 se presentan junto al LOSO como condición más exigente, sin presentarse como
equivalentes y declarando las diferencias de cámara, iluminación, encuadre y número de ejecuciones [MET
265-268]. Ninguna cifra obtenida con una partición que no separe señantes se presenta sin declarar esa
condición [MET 348-350].

---

## 3. Configuración fija del prototipo

Todas las sesiones de ambos bloques y la caracterización del retardo de audio se ejecutan con la misma
configuración. Un cambio en cualquiera de estas filas abre una decisión nueva en el registro y obliga a
declarar qué ensayos quedan antes y después del cambio.

| Elemento | Valor vigente | Fuente | Comprobado en esta revisión |
|---|---|---|---|
| Programa | `src/realtime.py`, modo aislado (sin `--modo-oracion`) | [ALC 7]; `realtime.py`, argumento `--modo-oracion` | — |
| Modelo | `models/canonico_holistic_mascara_manos/modelo_canonico.keras`, CNN+BiLSTM+atención, entrada (30, 252) | [REG 1147-1185 · N-31]; `translator.py`, `MODELO_POR_DEFECTO` | SHA256 `55e170fc…826fc1`, igual al declarado en `modelo_canonico.umbral.json` |
| Representación | «manos»: 126 rasgos por paso más velocidades | [REG 1003-1042 · D-08; REG 1104-1144 · N-30] | `modelo_canonico.procedencia.json`: `manos`, 126 / 252, 30 pasos |
| Normalización | centrado en el punto medio de los hombros y escalado por su distancia en xy | [REG 1150-1160 · N-31] | igual en la procedencia |
| Regla de mano ausente | «mascara» | [REG 850-868 · D-07; REG 893-896 · N-23] | `mano_ausente: mascara` en la procedencia |
| Extractor | `HolisticLandmarker`, mediapipe 0.10.35, delegado CPU, modo VIDEO, `OPCIONES_HOLISTIC` | [REG 577-647 · D-05; REG 736-750 · N-19]; `translator.py`, `OPCIONES_HOLISTIC` | SHA256 del `.task` `e2dab611…50c3f8`, igual al fijado en D-05 |
| Umbral de confianza | 0,20, tomado de `<modelo>.umbral.json`; **no se pasa `--umbral`** | [MET 183-189 · §2.2.6; REG 1201-1227 · D-09; REG 1230-1258 · N-33] | `"umbral": 0.2` |
| Ventana | fija, activada con ESPACIO; `--ventana-seg 2.5` → N = round(2,5 × tasa medida); tope 5 s; mínimo 15 fotogramas | [MET 193-196]; [REG 443-446 · N-11]; `translator.py`, `LSCTranslator.__init__` (`ventana_max_seg`) y `abrir_ventana`; `realtime.py`, argumentos `--ventana-frames`, `--ventana-seg`, `--ventana-max-seg` y `--min-frames` | valor a confirmar en PD-09 |
| Rechazo por manos | no se infiere si la proporción de fotogramas sin manos (o sin postura) supera 0,5 | [MET 191-192]; `realtime.py`, argumento `--max-sin-manos`; `translator.py`, `_cerrar_ventana` | valor a confirmar en PD-09 |
| Síntesis de voz | modo `offline`: `say` de macOS, voz Paulina (es-MX), 150 palabras por minuto; síntesis a archivo **antes** de t5 | [MET 172-179 · §2.2.5]; [REG 459-471 · N-12; REG 1261-1302 · N-34]; `tts.py`, `TTSModule.__init__` (`rate=150`) | voz Paulina es_MX presente; voz a confirmar en PD-29 |
| Registro | un JSONL por día en la carpeta indicada con `--registro-dir` | `translator.py`, `LSCTranslator.__init__` (`registro_path`); `realtime.py`, argumento `--registro-dir` | — |
| Resolución de captura | 640 × 480 (valor por defecto del programa cuando se escribió; desde el 22-nov-2026 el defecto es 1280 × 720, 16:9 como el corpus, N-57) | `realtime.py`, argumentos `--ancho` y `--alto` | a decidir en PD-08 |
| Equipo | Apple M5 Max, 18 núcleos, 48 GB de memoria, macOS 26.6.2, sin GPU en uso | [REG 792-793 · N-21]; requisito mínimo [REC 72-73] | `sysctl`: M5 Max, 18, 51 539 607 552 bytes; `sw_vers` 26.6.2 |
| Entorno de software | Python 3.12.13, TensorFlow 2.21.0, Keras 3.13.2, NumPy 2.4.3 (entorno de entrenamiento) | `resultados.json` del canónico | se registra el entorno real de cada sesión |
| Cámara | **no consta en el registro cuál se usó en N-26 y N-32** | [REC 72-73; REC 104-105, 119] | hoy el sistema lista la cámara integrada del portátil y una cámara de continuidad de un teléfono; no había cámara web externa conectada. Ver PD-08 |
| Salida de audio | parlantes integrados del portátil (dispositivo por defecto al inspeccionar) | — | `system_profiler SPAudioDataType`. Ver PD-24 |
| Lanzamiento | desde Terminal.app (desde el terminal del editor, macOS niega la cámara) | [REG 950-952 · N-26] | — |
| Almacenamiento de video | ninguno: el código no escribe fotogramas | [ETI 50-53]; [REG 437-439 · N-11] | `grep imwrite/VideoWriter` en `realtime.py` y `translator.py`: 0 coincidencias |

**Congelación.** *Propuesta*: al inicio de cada sesión se anotan en la hoja de sesión el SHA256 del
modelo y de `src/realtime.py`, `src/translator.py`, `src/tts.py`, `src/grammar_mapper.py` y
`src/representacion.py`, y el banner de arranque (`realtime.py`, `main`). Entre la primera y la última
sesión de un bloque no se modifica ninguno de esos archivos; si hubiera que hacerlo, se registra y los
ensayos se analizan por tramos. Advertencia: la conversión de glosas (3.a) sigue en desarrollo [CRO 50;
REG 968-970 · N-26], y un cambio en `grammar_mapper.py` altera la etapa de conversión (PD-25).

---

## 4. Ejecutantes

### 4.1 Bloque 1: persona señante o intérprete vinculada

**Perfil.** Se busca, en este orden, una persona sorda usuaria de la LSC como primera lengua y, en su
defecto, una persona intérprete certificada inscrita en el Registro Nacional de Intérpretes; el perfil
vinculado se declara en el reporte [REC 29-32]. La limitación del bloque depende de ese perfil: con una
persona sorda de primera lengua no aplica la de ejecución no nativa; con una intérprete certificada se
declara que las señas las ejecutó una usuaria fluida y certificada, no necesariamente nativa [MET 236-245].

**No participación en LSC50.** Se verifica y se deja constancia de que la persona no participó en la
grabación del corpus [MET 233-234; REC 32-33; MET 389-390]. El método de verificación no está fijado:
PD-21.

**Condiciones previas, todas obligatorias antes de la primera sesión del Bloque 1:**
1. Vinculación formalizada (actividad 0.c), condición para ejecutar el Bloque 1 [MET 386-390; REC 33-35].
   *No consta en el registro de decisiones a la fecha de este borrador.*
2. Consulta escrita al comité de ética de la Facultad radicada y respondida; lo que indique se cumple antes
   de ejecutar el bloque [ETI 69-75; CRO 106-108]. *No consta en el registro.*
3. Consentimiento informado previo y accesible (en LSC o con apoyo de una persona intérprete cuando la
   persona sea sorda), documentado por escrito en el único registro que la identifica y custodiado por el
   director separado de los registros experimentales; revocable en cualquier momento [ETI 55-61].
4. Reconocimiento económico cotizado por sesión u hora, incluida la función de ejecución [ETI 84-86;
   REC 140-148].

**Garantías durante y después** [ETI 50-69]: sin grabación ni almacenamiento de video; sin datos
identificativos en los registros; sin incorporación de material al conjunto de datos de trabajo ni a
ninguna publicación; los registros derivados son datos disociados y se conservan solo para reportar RE5; si
la persona revoca el consentimiento, se suprimen los registros de sus ejecuciones y el Bloque 1 se retira
del reporte con constancia de la razón.

**Presentación y variantes.** Antes de cada ejecución se presenta la glosa y, si la persona lo requiere, la
secuencia de referencia del corpus; la persona ejecuta la seña en su forma habitual y declara si difiere de
la que el corpus fija, declaración que se registra por ejecución [MET 225-228]. Medio, momento y categorías
de esa declaración: PD-11 y PD-13.

**Reporte por duplicado.** La exactitud se reporta sobre la totalidad de las ejecuciones y sobre las
ejecuciones sin variante declarada, junto con el número y la proporción de ejecuciones con variante y de
rechazos por umbral; reportar solo la cifra depurada sería un resultado con sesgo de selección [MET
229-233; RES 45-47].

**Contingencia.** Si la vinculación no se concreta, o si lo que indique el comité no puede cumplirse antes
del mes 4, no existe el Bloque 1: la evaluación en línea se reduce al Bloque 2 y se reporta sin ninguna
cifra de exactitud; el autor no sustituye a la persona para producir esa cifra [MET 269-271; REC 61-66].

### 4.2 Bloque 2: el autor

Solo latencia y casos límite. De este bloque se reportan el número de ejecuciones, el número de rechazos y
de casos límite ejercitados, y los tiempos; **no se reporta ninguna cifra de exactitud** [MET 259-262],
porque una exactitud medida sobre las ejecuciones del autor quedaría confundida con la brecha entre
señantes fluidos y aprendices [MET 255-259]. No requiere a la persona vinculada y puede ejecutarse desde
que el prototipo está integrado [MET 262-263].

Las pruebas en vivo del 14 y 15 de septiembre [REG 948-978 · N-26; REG 1188-1198 · N-32] **no forman parte
del Bloque 2**: son anteriores a este protocolo, usaron umbral 0,5 y su etapa de síntesis no incluía la
síntesis real (criterio E1, sección 10).

### 4.3 Nadie más

Ninguna otra persona ejecuta señas [ALC 76-77]. *Propuesta*: nadie más aparece en el encuadre durante las
sesiones; si alguien entra al cuadro con una ventana abierta, la ventana se aborta con X y se anota la
incidencia.

---

## 5. Procedimiento de una sesión

Una **ejecución** es cada ventana abierta con ESPACIO; produce exactamente un registro `"tipo": "ensayo"`
en el JSONL (`translator.py`, `_cerrar_ventana`). Una **presentación** es cada vez que se pide una glosa; puede
generar más de una ejecución si se permite repetir (PD-14).

### 5.1 Antes de la sesión

**Comunes a los dos bloques**
1. Confirmar que todos los parámetros de la sección 12 que afectan a la sesión están decididos y anotados.
2. Preparar el espacio según PD-06 (iluminación y fondo) y PD-07 (distancia y encuadre) y anotar las
   condiciones en la hoja de sesión (plantilla 9.2).
3. Reducir la carga del equipo: cerrar procesos ajenos y aplicar lo que se decida en PD-23 sobre la
   sincronización de Google Drive, a la que N-21 atribuye como causa probable el aumento del costo del
   extractor en la corrida completa de 1.c [REG 800-808 · N-21]. Anotar la carga (`uptime`).
4. *Propuesta*: ejecutar sin cámara la suite de pruebas del prototipo
   (`python -m unittest tests.test_prototipo tests.test_tts tests.test_umbral`) y anotar el resultado.
5. Anotar los SHA256 de la sección 3 («Congelación»).
6. Fijar la salida de audio y el volumen (PD-24); no usar audífonos inalámbricos.
7. Lanzar desde Terminal.app [REG 950-952] con una carpeta de registro propia del bloque, por ejemplo:
   ```
   python src/realtime.py --camera <índice decidido> --registro-dir registros/4b_bloque<k> \
          [--ancho <PD-08> --alto <PD-08>] [--ventana-frames <PD-09> | --ventana-seg <PD-09>]
   ```
   Sin `--umbral`, sin `--mano-ausente`, sin `--model-path` y sin `--modo-oracion`, para que rijan los
   valores de la sección 3.
8. Verificar en el banner (`realtime.py`, `main`): modelo canónico «manos», procedencia leída, umbral 0,20
   con origen en `modelo_canonico.umbral.json`, mano ausente «mascara», TTS `offline`, ruta del registro.
   Si alguna línea difiere, no se inicia la sesión.
9. Verificar el índice de cámara: con más de una cámara disponible, el índice 0 puede no ser la cámara
   decidida (PD-08). Comprobar la imagen en la ventana del programa antes de empezar.
10. Esperar a que la tasa del bucle mostrada en pantalla se estabilice (`realtime.py`, `dibujar_ui`) y anotarla.

**Solo Bloque 1**
11. Comprobar que existen el consentimiento firmado bajo custodia del director, la respuesta del comité
    cumplida y la constancia de no participación en LSC50 (sección 4.1). Sin cualquiera de ellos no hay
    sesión.
12. Asignar un código de participante sin datos identificativos (por ejemplo, `P01`); el nombre no aparece
    en ninguna hoja experimental [ETI 57-60].
13. Preparar el orden de presentación decidido en PD-03 y el medio de presentación de PD-11; si se usa una
    semilla, anotarla antes de conocer ningún resultado.
14. Explicar el procedimiento a la persona en el medio accesible acordado y resolver dudas; realizar, si se
    decide en PD-16, las ejecuciones de familiarización.

**Solo Bloque 2**
15. Preparar la lista de ejecuciones normales y el guion de casos límite (sección 5.3) con su orden.

### 5.2 Durante la sesión: Bloque 1, por presentación

1. **Presentar la glosa** k por el medio de PD-11. Si la persona lo pide, mostrar la secuencia de
   referencia del corpus [MET 225-226] y anotar que se mostró.
2. **Reposo.** La persona adopta la posición de reposo acordada (PD-10).
3. **Abrir la ventana.** Se pulsa ESPACIO (quién pulsa: PD-10). El manejador sella t0
   (`realtime.py`, `main`, tecla ESPACIO) y la barra de progreso avanza.
4. **Ejecutar** la seña en su forma habitual. La ventana se cierra sola al completar N fotogramas o al tope
   de tiempo (`translator.py`, `procesar_fotograma`). Si la ejecución se interrumpe, se pulsa X antes del
   cierre: cuenta como `rechazada_incompleta` (`realtime.py`, `main`, tecla X; `translator.py`,
   `abortar_ventana`).
5. **Declarar la variante** (sí / no / categorías de PD-13), en el momento decidido en PD-13 y sin que la
   persona haya visto el resultado si así se decide en PD-12.
6. **Anotar** en la hoja de ejecución (plantilla 9.3): número de presentación, glosa objetivo,
   `id_ensayo` impreso en la consola (`realtime.py`, `_resumen_consola`), variante, referencia mostrada y, como
   control, el estado mostrado.
7. **Rechazo por confianza.** El sistema no emite audio y pide repetir la seña [MET 184-186]. Si se repite,
   la repetición es una ejecución nueva vinculada a la misma presentación, conforme a PD-14.
8. **Esperar el reposo del sistema** antes de la siguiente presentación: que termine el audio (para no
   provocar `tts_ocupado_en_t5`, [REG 1296-1298 · N-34]) y que la pantalla vuelva a «REPOSO». Tras una
   ventana aceptada, el bucle queda detenido mientras se sintetiza [REG 1279-1281 · N-34]. *Propuesta*: si
   tras una ventana aceptada la voz no suena, o la consola muestra «[AVISO TTS] falló una locución en segundo
   plano», se anota en la hoja con el último `id_ensayo` y se sigue lo que decida PD-31.
9. **Pausas** según PD-05. Se pulsa Q al final de la sesión, no al final de cada pausa, salvo que PD-04
   disponga otra cosa.

### 5.3 Durante la sesión: Bloque 2

**Ejecuciones normales (latencia).** Mismo procedimiento que el Bloque 1 sin declaración de variante; en la
hoja se anota la seña que el autor intentó (dato operativo, no para exactitud). Número y reparto: PD-02.

**Casos límite de §2.2.6.** El anteproyecto exige ejercitarlos deliberadamente: rechazo por confianza
insuficiente, mano no detectada o fuera de cuadro, seña fuera del vocabulario y secuencia incompleta [MET
250-251]; §2.2.6 menciona además la oclusión [MET 191]. El guion propuesto describe cómo provocar cada caso
y qué hace el código vigente; el estado que efectivamente ocurra se registra tal cual, sin reetiquetar.

| Id | Caso | Cómo se provoca (*propuesta*) | Qué hace el código vigente | Qué se anota |
|---|---|---|---|---|
| CL-1 | Confianza insuficiente [MET 184-189] | No es provocable de forma determinista. Surge en ejecuciones normales; se añaden ejecuciones deliberadamente ambiguas (seña a medio ejecutar, movimiento sin forma de seña) | Si la confianza es menor que 0,20: `rechazada_confianza`, sin audio (`translator.py`, `_inferir_y_emitir`); si no, `aceptada` | Condición intentada y estado |
| CL-2 | Manos no detectadas o fuera de cuadro [MET 191-192] | Abrir la ventana con las dos manos fuera del encuadre durante toda la ventana; variante: fuera durante parte de la ventana | Indicación visual continua (`realtime.py`, `dibujar_ui`). Si la proporción de fotogramas **sin ninguna mano** supera 0,5: `rechazada_manos`, no se infiere (`translator.py`, `_cerrar_ventana`); si no la supera, se infiere | Proporción aproximada intentada; `proporcion_sin_manos` del JSONL |
| CL-3 | Oclusión [MET 191] | Ocultar una mano tras el propio cuerpo o tras un objeto, sin terceros | Un fotograma con una sola mano **no** cuenta como «sin manos»; se infiere con la mano ausente enmascarada (D-07). Solo hay rechazo si faltan las dos manos en más de la mitad | Mano ocluida y estado |
| CL-4 | Postura no detectada | Salir parcialmente del encuadre de modo que no se detecten los hombros | `rechazada_manos` con motivo «pose no detectada» (`translator.py`, `_cerrar_ventana`) | Estado y motivo |
| CL-5 | Seña fuera del vocabulario [MET 250-251] | Ejecutar señas o gestos que no están entre las 50 glosas (qué señas: PD-17) | No existe un estado específico: el vocabulario es cerrado [MET 29-31]; el resultado es `rechazada_confianza` o `aceptada` con una clase del vocabulario | Gesto ejecutado (descripción) y estado |
| CL-6a | Secuencia incompleta: abortada | Pulsar X con la ventana abierta | `rechazada_incompleta`, motivo «abortada» (`translator.py`, `_cerrar_ventana`) | Estado |
| CL-6b | Secuencia sin inicio o fin claros dentro de la ventana [MET 192-193] | Empezar la seña antes de ESPACIO o terminarla después del cierre de la ventana | **El prototipo no la descarta**: solo descarta por X o por menos de 15 fotogramas, así que infiere (riesgo R6). Desde el 22-nov-2026 (N-50, opción B) la marca como aviso en el JSONL (`bordes.aviso_inicio`, `bordes.aviso_fin`) sin rechazarla | Qué parte quedó fuera, estado y avisos de borde |
| CL-6c | Menos fotogramas que el mínimo | No provocable sin alterar el sistema: exige menos de 15 fotogramas en el tope de 5 s | `rechazada_incompleta` (`translator.py`, `_cerrar_ventana`) | Se declara no ejercitado si no ocurre |

### 5.4 Después de la sesión

1. Cerrar con Q y copiar a la hoja de sesión los contadores impresos (`realtime.py`, `main`, bloque
   «Contadores de la sesion») y, si aparece justo encima, la línea «[AVISO TTS] Voz de la sesión: …», que
   `translator.py`, `LSCTranslator.close`, imprime solo cuando alguna locución no sonó por un error o se
   descartó al cerrar.
2. Comprobar que el número de ejecuciones de la hoja coincide con el número de registros `ensayo` del
   JSONL de la sesión y que cada `id_ensayo` anotado existe. Una discrepancia se anota; no se corrige a
   mano el JSONL.
3. Ejecutar `python src/latencia_resumen.py registros/4b_bloque<k>/ --json <archivo del resumen>` y anotar el
   resumen de la sesión (dónde se guarda el JSON: PD-27). Comprobar en la tabla (`latencia_resumen.py`,
   `tabla`) y, entre paréntesis, en el JSON (`latencia_resumen.py`, `resumir`):
   - que todos los ensayos con audio tienen la síntesis terminada antes de t5: la línea «TTS: …» dice «solo
     orden emitida en 0» (`sintesis_antes_de_t5.no`) y las filas `sintesis`, `L_total` y `L_resp` marcadas
     `[sint. no incl.]` tienen n = 0 (`por_sintesis_antes_de_t5.sintesis_no_incluida`);
   - cuántos ensayos tenían otra locución sonando al emitir la orden: línea «TTS ocupado al emitir t5»
     (`tts_ocupado_en_t5.si`), que se copia a la hoja (E4);
   - que el modelo es el canónico (línea «Modelos:»);
   - que la sección «Umbral de confianza por ensayo» tiene una sola línea, con umbral 0,20 y origen en
     `modelo_canonico.umbral.json` (`umbrales[]`: un único elemento, con `umbral` 0.2 y ese
     `umbral_origen`);
   - la lista `advertencias`, que solo está en el JSON: se anota tal cual, y cada magnitud que una advertencia
     señala se toma de la clave que ella indica (`ventanas_inferidas`, `por_sintesis_antes_de_t5` o
     `umbrales`), no de `magnitudes_ms` ni de la `tasa_rechazo` global.
   El resumen agrupa todo lo que hay en la carpeta, sin separar por fecha ni por sesión (R15).
4. *Propuesta*: comprobar la voz. En el último registro de la sesión (mismo `sesion`) del JSONL,
   `tts_errores_reproduccion` debe valer 0, y al cerrar no debe haber aparecido la línea de aviso de voz del
   punto 1. Si vale más de 0 o apareció la línea, se registra la incidencia en la hoja (número de errores,
   `tts_ultimo_error_reproduccion` y primer `id_ensayo` en que el contador aumenta) y se aplica E7. Si vale
   null, la voz no es la de la sección 3 (se revisa `tts_modo`, E2). `latencia_resumen.py` no agrega estos
   campos.
5. Comprobar que no se creó ningún archivo de imagen o video durante la sesión (por ejemplo, listar los
   archivos modificados en el repositorio y en la carpeta temporal durante el intervalo de la sesión). Los
   audios sintetizados de la voz se borran solos (sección 9.1).
6. Guardar las hojas sin datos identificativos. Si hubo consentimiento, confirmar que su documento quedó
   bajo custodia del director, separado de las hojas [ETI 57-59].
7. Registrar en `DECISIONES_TESIS.txt` una nota N-nn con la sesión: bloque, fecha, número de ejecuciones,
   incidencias (también las de voz) y exclusiones aplicadas (sin cifras de exactitud del Bloque 2).

---

## 6. Variables, marcas de tiempo y su cálculo

### 6.1 Marcas

Todas se sellan con un único reloj monótono de alta resolución [MET 291-293], `time.perf_counter_ns`
[REG 447-449 · N-11]. El instante en que empieza la seña no es medible y no se usa como origen [MET 282-291;
ALC 94-95].

| Marca | Definición normativa [MET 293-300] | Dónde se sella |
|---|---|---|
| t0 | el manejador de eventos registra la activación de la ventana | `realtime.py`, `main`: `t0 = tr.ahora()` al leer ESPACIO con `cv2.waitKey` |
| c_i | la cámara entrega el fotograma i a la aplicación | `realtime.py`, `main`: `c_i = tr.ahora()` tras `cap.read()` |
| e_i | el extractor devuelve los puntos del fotograma i | `translator.py`, `procesar_fotograma`: `e_i`, al volver `extractor.extract` |
| t1 = c_N | llegada del último fotograma de la ventana | `translator.py`, `_cerrar_ventana`: `ens.t1 = ens.c[-1]` |
| t2 | fin del preprocesamiento | `translator.py`, `_inferir_y_emitir`: `ens.t2`, tras `_preprocesar` |
| t3 | el modelo devuelve clase y confianza | `translator.py`, `_inferir_y_emitir`: `ens.t3`, tras `modelo.predict` |
| t4 | se devuelve el enunciado en español | `translator.py`, `_inferir_y_emitir`: `ens.t4`, tras `grammar_mapper.translate` |
| t5 | el motor devuelve el audio y se emite la orden de reproducción | `translator.py`, `_inferir_y_emitir`: `ens.t5`, tras `_emitir_audio`, que sintetiza a archivo, espera la síntesis y emite la orden con `play_prepared_async` |

### 6.2 Magnitudes

| Magnitud | Fórmula | Carácter | Fuente |
|---|---|---|---|
| Captura | t1 − t0 | etapa; absorbe la extracción solapada fotograma a fotograma | [MET 302; MET 305-307] |
| Extracción | e_N − t1 | etapa; residuo del último fotograma | [MET 302-303; MET 306-307] |
| Preprocesamiento | t2 − e_N | etapa | [MET 303] |
| Inferencia | t3 − t2 | etapa | [MET 303] |
| Conversión | t4 − t3 | etapa | [MET 303-304] |
| Síntesis | t5 − t4 | etapa; incluye la síntesis real desde N-34 | [MET 304]; [REG 1267-1272 · N-34] |
| **L_total** | t5 − t0 = suma exacta de las seis etapas | **variable del estudio** | [MET 300-305] |
| **L_resp** | t5 − t1 | lectura conversacional | [MET 310-318] |
| W | t1 − t0 | **parámetro declarado, no resultado**; L_total = W + L_resp | [MET 318-321] |
| Costo del extractor | e_i − c_i por fotograma | indicador comparable | [MET 321-323] |
| Tasa efectiva | N / (c_N − c_1) por ensayo | indicador comparable | [MET 323-324] |

Nota de cálculo: la tasa N / (c_N − c_1) cuenta N fotogramas sobre N − 1 intervalos y excede la tasa por
intervalos en el factor N/(N − 1); el JSONL conserva todos los c_i, de modo que la variante por intervalos
puede recalcularse (`translator.py`, `_registro_ensayo`, comentario de la tasa efectiva). Se reporta la
fórmula literal de §2.2.7.

### 6.3 Qué ejecuciones aportan a cada magnitud (según el código vigente)

| Estado de la ejecución | Extracción, costo, tasa | Captura, W | Preprocesamiento, inferencia | Conversión, síntesis, L_total, L_resp |
|---|---|---|---|---|
| `aceptada` (modo aislado) | Sí | Sí | Sí | Sí |
| `rechazada_confianza` | Sí | Sí | Sí | No (no hay t4 ni t5) |
| `rechazada_manos` | Sí | Solo en la referencia `[todas]` del resumen | No | No |
| `rechazada_incompleta` | Sí, si hubo fotogramas | Solo en `[todas]`, si hubo fotogramas | No | No |

Fuente: `translator.py`, `_registro_ensayo` (qué campos tiene cada estado), y `latencia_resumen.py`, `resumir`
(cómo los agrupa el resumen). Consecuencias:
- L_total y L_resp solo se miden sobre ejecuciones aceptadas; con un umbral que rechace mucho, el n de esas
  magnitudes será menor que el número de ejecuciones.
- W y captura: el JSONL los tiene en toda ventana cerrada con fotogramas, pero el resumen los reporta sobre
  las ventanas **inferidas** (aceptadas y rechazadas por confianza), como pide [REG 566 · N-14]: filas
  `captura [inferidas]` y `W [inferidas]` (JSON `ventanas_inferidas`). El cálculo anterior, sobre todas las
  ventanas con fotogramas, sale como referencia (filas `captura [todas]` y `W [todas]`) y sigue en
  `magnitudes_ms`, con un aviso en `advertencias` cuando incluye ventanas no inferidas. Cuál se reporta: PD-19.
- Síntesis, L_total y L_resp: el resumen los separa por `tts_sintesis_antes_de_t5`, sin intervención: filas
  `[sint. incluida]` (true: la síntesis real está dentro de t5) y `[sint. no incl.]` (false: registros
  anteriores a N-34, motor sin síntesis a archivo o síntesis fallida o fuera del tope; o sin dato: sin audio),
  en JSON `por_sintesis_antes_de_t5.sintesis_incluida` y `por_sintesis_antes_de_t5.sintesis_no_incluida`.
- El resumen no separa por fecha, sesión ni carpeta: resume todos los registros que recibe. Los registros
  anteriores a este protocolo (E1) quedan fuera porque se le da solo la carpeta propia del bloque
  (`--registro-dir`, sección 5.1, punto 7).

### 6.4 Variables de reconocimiento (solo Bloque 1)

Por ejecución: glosa objetivo (hoja), clase predicha y top-3 (`id_clase`, `top3` del JSONL, presentes en
`aceptada` y `rechazada_confianza`), confianza, estado y variante declarada (hoja). Métricas: top-1, top-3,
precision, recall y F1 por clase y macro, y matriz de confusión [MET 276-280; MET 503-505], calculadas dos
veces: sobre la totalidad y sobre las ejecuciones sin variante declarada [MET 229-232]. Además: número y
proporción de ejecuciones con variante y de rechazos por umbral [MET 231-232]; el reporte distingue errores
de clasificación de rechazos deliberados [MET 197-198]. El denominador exacto de cada métrica cuando hay
rechazos por manos o por secuencia incompleta no está fijado: PD-15.

### 6.5 Variables de conteo (Bloque 2)

Número de ejecuciones; número de ejecuciones por estado; número de casos límite ejercitados por tipo
(CL-1 a CL-6) con el estado observado en cada uno [MET 261-262; RES 55-57].

### 6.6 Variables de condición (ambos bloques)

Cámara, resolución, tasa nominal y medida, computador (CPU, memoria), iluminación, fondo, distancia,
encuadre, número de repeticiones por seña y número de ejecuciones por sesión [MET 274-276; MET 498-502],
perfil de la persona del Bloque 1 [RES 59-62], carga del equipo y salida de audio.

---

## 7. Estadísticos que se reportan

1. **Latencias e indicadores** (L_total, seis etapas, L_resp, W, costo del extractor por fotograma, tasa
   efectiva): media, desviación estándar, mediana y percentil 90, con las condiciones de hardware [MET
   324-326; RES 52-55], más n por fila. `latencia_resumen.py` usa desviación muestral (ddof = 1) y añade
   mínimo y máximo (`latencia_resumen.py`, `estadisticas`). Síntesis, L_total y L_resp se toman de las filas
   `[sint. incluida]`; W y captura, del conjunto que decida PD-19 (sección 6.3). Se reportan por bloque y en
   conjunto, identificando la etapa que domina el total [MET 509-510].
2. **W** se reporta como parámetro declarado, junto a L_total y L_resp, de modo que ninguna lectura oculte
   a la otra [MET 318-321].
3. **Comparaciones permitidas**: costo por fotograma frente a los 44,92 ms por fotograma clave de [ref25] y
   tasa efectiva frente a los 25-30 fotogramas por segundo de [ref24] y los 14 de [ref42] [MET 321-324]. La
   latencia de respuesta se lee con marcos publicados que **no son umbrales** [MET 333-337].
4. **Exactitud del Bloque 1**: top-1, top-3, precision, recall y F1 (por clase y macro), matriz de
   confusión, por duplicado, con el número y la proporción de variantes y de rechazos por umbral [MET
   229-233; MET 503-505].
5. **Offline**: media y dispersión entre pliegues del LOSO, resultado por pliegue con la condición del
   señante excluido, y resultado de la partición aleatoria, cada uno con su carácter dependiente o
   independiente del señante [MET 144-147; MET 522-525; RES 59-63]. El resultado por señante excluido
   (top-1, top-3 y F1 macro) y el efecto del umbral por señante (con 0,20, rechazo de 10,5 % en 0003 a
   29,0 % en 0000) ya están en `models/analisis_errores_manos_2026-09-15/analisis.md`, secciones 3 y 5
   (`analisis.json`, claves `por_senante` y `calibracion.umbral.por_senante`). El F1 macro de la sección 2.1
   es la media de los F1 por pliegue (0,537); calculado sobre las 997 predicciones juntas es 0,574
   (`analisis.json`, `por_sena.f1_macro_agregado`): el reporte debe nombrar cuál usa.
6. **Retardo de audio**: estadísticos de la sección 8, reportados aparte como sesgo constante conocido que
   **no se resta** de ninguna latencia; si no se mide, se declara «no medido» [MET 513-517].
7. **Bloque 2**: conteos de la sección 6.5; ninguna proporción de aciertos.
8. **Efecto del umbral en línea**: tasa de rechazo por bloque y, en el Bloque 1, exactitud entre las
   aceptadas, en coherencia con lo reportado offline para D-09 [MET 187-189; REG 1218-1222 · D-09]. La tasa
   de rechazo se toma de la sección «Umbral de confianza por ensayo» del resumen (`umbrales[].tasa_rechazo`,
   con sus `contadores`), no de la `tasa_rechazo` global, que mezcla umbrales si hay más de uno. Esa tasa es
   1 − aceptadas / ensayos e incluye los rechazos por manos y las incompletas; los rechazos por umbral son
   `umbrales[].contadores.rechazada_confianza` (denominador de la exactitud: PD-15).
9. **No se reporta el cumplimiento de ningún umbral** [RES 69-70]. Intervalos de confianza u otros
   estadísticos no previstos por el anteproyecto: PD-20.

---

## 8. Caracterización del retardo de audio con grabación externa

### 8.1 Qué exige el anteproyecto

t5 es el instante en que se emite la orden de reproducción y no aquel en que suena la primera muestra; el
retardo del controlador y del dispositivo de audio queda fuera de la medición por software, se caracteriza
**una sola vez** con una grabación externa que capte la pulsación y el inicio del audio, y se reporta como
sesgo constante conocido [MET 326-331; MET 482-484]. No se resta de L_total ni de L_resp; si no llega a
realizarse, se declara «no medido» sin suponer ningún valor [MET 513-517].

### 8.2 Qué mide exactamente (y qué no separa)

Para cada ensayo de caracterización, la grabación da el intervalo externo I_ext entre la pulsación física
de ESPACIO y el inicio audible del audio; el JSONL da L_total = t5 − t0. La diferencia

  d = I_ext − L_total

suma dos retardos que este método **no separa**:
- entre la pulsación física y t0: t0 se sella al leer la tecla con `cv2.waitKey` al final de la iteración
  del bucle, hasta un periodo de bucle después de la pulsación [REG 567-568 · N-14]; con la tasa de
  30,4 fotogramas por segundo medida en N-26 [REG 955], ese periodo es de unos 33 ms (1/30,4 s);
- entre t5 y la primera muestra audible: el retardo que §2.2.7 quiere caracterizar.

*Propuesta*: reportar d como el sesgo constante conocido, declarando que incluye ambos componentes, y
declarar además que el instante acústico del clic de la tecla no coincide necesariamente con el cierre
eléctrico del interruptor.

### 8.3 Procedimiento (*propuesta*, con parámetros abiertos en PD-18)

1. **Ejecutante**: el autor, solo, en una sala silenciosa; ninguna persona aparece ni se oye en la
   grabación [ALC 76-77; ETI 48-53].
2. **Modalidad de grabación** (PD-18): se sugiere solo audio con un dispositivo externo al computador
   (por ejemplo, un teléfono o una grabadora), con el micrófono cerca del teclado y del parlante, para que
   no quede ningún video y para disponer de resolución de muestra de audio.
3. **Configuración**: la de la sección 3, con la misma salida de audio y el mismo volumen que las sesiones
   (PD-24). Para que cada ventana produzca audio, en esta corrida dedicada puede usarse `--umbral 0`
   (el retardo de audio no depende del umbral); la corrida va a una carpeta propia
   (`--registro-dir registros/4a_retardo_audio`) y sus ensayos no entran en ningún bloque.
4. **Ensayos**: separados por varios segundos de silencio, esperando que termine cada audio (verificar
   `tts_ocupado_en_t5 = false`, y que `tts_errores_reproduccion` no aumente respecto del registro anterior, ni en el del propio ensayo ni en los siguientes, ni
   aparezca el aviso de voz al cerrar: un ensayo cuyo audio no sonó no es emparejable). Número de ensayos:
   PD-18.
5. **Emparejamiento**: el ensayo k de la grabación se empareja con el `id_ensayo` k del JSONL; se anota en
   la plantilla 9.5. Ensayos no emparejables se excluyen y se cuentan.
6. **Detección de los instantes**: inicio del clic y inicio del audio sobre la forma de onda, con el método
   que se decida en PD-18 (lectura manual con un editor de audio o detección por umbral con un guion), el
   mismo para todos los ensayos.
7. **Estadísticos de d**: n, media, desviación estándar, mediana y percentil 90 (mismos que la sección 7).
8. **Conservación**: el destino del archivo de audio se decide en PD-18.

---

## 9. Instrumentos

### 9.1 Registro automático: JSONL del prototipo

Un registro por ejecución (`translator.py`, `_registro_ensayo`). Campos y uso en este protocolo:

| Campo | Uso |
|---|---|
| `tipo`, `sesion`, `id_ensayo`, `fecha_hora` | clave de unión con las hojas: (`sesion`, `id_ensayo`) |
| `modo` | control: debe ser `aislado` |
| `modelo`, `n_rasgos_modelo`, `usa_velocidades`, `pasos_modelo`, `representacion`, `procedencia_modelo` | control de configuración (criterio E2) |
| `extractor`, `mano_ausente`, `mano_ausente_modelo` | control de configuración |
| `tts_modo`, `tts_sintesis_antes_de_t5`, `tts_ocupado_en_t5` | control de la etapa de síntesis (E2, E4) |
| `tts_errores_reproduccion`, `tts_ultimo_error_reproduccion` | control de la salida de voz (E7): errores de reproducción acumulados en la sesión hasta ese registro (entero) y texto del último (hasta 200 caracteres); null si no hay voz o el motor no los informa (null es «no informado», no «cero errores»). Se leen después de sellar t5 y no alteran las marcas (`translator.py`, `_estado_voz`). También están en los registros `turno` (`fin_de_turno`) |
| `parametros` (`umbral`, `umbral_origen`, `ventana_frames_objetivo`, `ventana_frames_param`, `ventana_seg`, `ventana_max_seg`, `min_frames`, `max_sin_manos`, `fps_medida_antes`) | parámetros efectivos de cada ventana |
| `t0`, `c`, `e`, `t1` … `t5` | marcas de la sección 6.1 |
| `N`, `cierre` | fotogramas de la ventana y motivo de cierre (`frames`, `tiempo`, `abortada`) |
| `frames_con_deteccion`, `frames_sin_manos`, `frames_sin_pose`, `frames_con_dos_manos`, `proporcion_sin_manos` | casos límite CL-2 a CL-4 |
| `etapas_ns`, `W_ns`, `L_total_ns`, `L_resp_ns`, `costo_extractor_ns`, `tasa_efectiva_fps` | magnitudes de la sección 6.2 |
| `estado`, `motivo`, `id_clase`, `clase`, `confianza`, `top3`, `enunciado` | reconocimiento y estados de §2.2.6 |
| `contadores`, `glosas_turno` | control |

**Voz: errores y audios temporales.** El audio de cada ensayo suena en segundo plano después de t5, así que si
falla, el aumento de `tts_errores_reproduccion` suele aparecer en los registros siguientes (en el suyo solo si el hilo falla antes de que se lea el estado)
(R20). El JSONL no tiene registro de cierre: al salir, `translator.py`, `LSCTranslator.close`, imprime una
sola línea «[AVISO TTS] Voz de la sesión: N locución(es) no sonaron por error de reproducción (último: …); M
descartada(s) al cerrar.» si N o M son mayores que 0; sale justo antes de «Contadores de la sesion», y lo que
el hilo descarte después de ese instante no se cuenta. Durante la sesión, `tts.py` avisa en consola solo el
primer error de cada tipo de excepción (`TTSModule._registrar_error_reproduccion`); los siguientes solo se
cuentan. Los audios sintetizados ya no se acumulan: cada uno se borra de la carpeta temporal del módulo al
terminar de sonar (también si falla), si su preparación se reemplaza sin sonar o si la síntesis falla, y
`cleanup()` borra la carpeta al cerrar (`tts.py`, docstring del módulo, «Audios temporales y errores en
segundo plano»; `TTSModule._reproducir`, `_borrar_audio` y `cleanup`). Quedan así resueltos dos pendientes de
N-34, los archivos AIFF temporales hasta `close()` y las excepciones de reproducción no capturadas en el hilo
[REG 1299-1301 · N-34].

**Lo que el JSONL no registra** y por eso va a la hoja manual: bloque, código de ejecutante, glosa objetivo,
número de presentación, variante declarada, referencia mostrada, caso límite intentado y observaciones.
El JSONL no guarda imágenes ni puntos (`translator.py`, docstring del módulo, «SIN ALMACENAR VIDEO»). Hoy la
pantalla no muestra el `id_ensayo`; se lee de la consola (`realtime.py`, `_resumen_consola`). Registrar la
glosa objetivo y el bloque en el propio JSONL requeriría un cambio de código con revisión: PD-22.

### 9.2 Hoja de sesión (una por sesión)

| Campo | Valor |
|---|---|
| Bloque (1 / 2 / caracterización de audio) | |
| `sesion` del JSONL y carpeta de registro | |
| Fecha, hora de inicio y de fin | |
| Código de ejecutante (sin nombre) | |
| Perfil declarado (Bloque 1) | |
| Consentimiento bajo custodia del director (sí / no) · respuesta del comité cumplida (sí / no) · constancia de no participación en LSC50 (sí / no) | |
| Equipo (CPU, memoria, sistema) y carga al inicio (`uptime`) | |
| Estado de la sincronización de Google Drive (PD-23) | |
| Cámara (modelo e índice), resolución solicitada y efectiva, tasa nominal | |
| Tasa del bucle en pantalla antes de empezar | |
| Iluminación (PD-06), fondo, distancia y encuadre (PD-07) | |
| Salida de audio y volumen (PD-24) | |
| SHA256 del modelo y de los cinco archivos de código (sección 3) | |
| Banner verificado: umbral 0,20 y su origen, representación, mano ausente, TTS, voz | |
| Comando exacto de lanzamiento | |
| Orden de presentación (lista o semilla, PD-03) | |
| Pausas realizadas (hora y duración) | |
| Contadores al cerrar | |
| Voz: línea «[AVISO TTS] Voz de la sesión» al cerrar (texto o «no apareció») · `tts_errores_reproduccion` del último registro de la sesión | |
| Ejecuciones en hoja / registros `ensayo` en el JSONL | |
| Incidencias (cámara, cierre inesperado, persona en cuadro, otras) | |
| Exclusiones aplicadas y motivo (sección 10) | |

### 9.3 Hoja de ejecución del Bloque 1

| Presentación | Ronda | Glosa objetivo (id y glosa) | Intento | `id_ensayo` | Referencia mostrada | Variante declarada | Estado en pantalla (control) | Observaciones (sin datos identificativos) |
|---|---|---|---|---|---|---|---|---|
| 1 | | | 1 | | sí / no | sí / no / (PD-13) | | |
| | | | | | | | | |

### 9.4 Hoja de ejecución del Bloque 2

| N.º | `id_ensayo` | Tipo (normal / CL-1 … CL-6c) | Seña o gesto ejecutado | Condición provocada | Estado en pantalla | Observaciones |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| | | | | | | |

### 9.5 Hoja de caracterización del retardo de audio

| N.º | `id_ensayo` | L_total del JSONL (ms) | Instante de la pulsación en la grabación | Inicio del audio en la grabación | I_ext (ms) | d = I_ext − L_total (ms) | `tts_ocupado_en_t5` | `tts_errores_reproduccion` | Observaciones |
|---|---|---|---|---|---|---|---|---|---|
| 1 | | | | | | | | | |

### 9.6 Plantillas de reporte

**T1. Offline por pliegue** (RE5, RE6)

| Esquema | Pliegue / señante excluido | Condición | n prueba | top-1 | top-3 | P macro | R macro | F1 macro |
|---|---|---|---|---|---|---|---|---|
| LOSO | 0 / 0000 | independiente | | | | | | |
| … | | | | | | | | |
| LOSO | media ± DE | independiente | | | | | | |
| Aleatoria | — | dependiente | | | | | | |

**T2. Exactitud en línea del Bloque 1, por duplicado**

| Conjunto | n ejecuciones | n con predicción | top-1 | top-3 | P macro | R macro | F1 macro | Rechazos por umbral (n, %) |
|---|---|---|---|---|---|---|---|---|
| Totalidad de las ejecuciones | | | | | | | | |
| Ejecuciones sin variante declarada | | | | | | | | |
| Ejecuciones con variante declarada: n y proporción | | | — | — | — | — | — | — |

**T3. Estados por bloque**

| Bloque | Ejecuciones | `aceptada` | `rechazada_confianza` | `rechazada_manos` | `rechazada_incompleta` | Excluidas (motivo) |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |

**T4. Latencias** (una tabla por bloque y una conjunta; ms)

| Magnitud (fila de `latencia_resumen.py`) | n | media | DE | mediana | p90 |
|---|---|---|---|---|---|
| Captura (`[inferidas]` o `[todas]`, según PD-19) · Extracción · Preprocesamiento · Inferencia · Conversión | | | | | |
| Síntesis `[sint. incluida]` | | | | | |
| L_total `[sint. incluida]` · L_resp `[sint. incluida]` | | | | | |
| W (parámetro declarado; `[inferidas]` o `[todas]`, según PD-19) | | | | | |
| Costo del extractor por fotograma | | | | | |
| Tasa efectiva (fotogramas/s) | | | | | |

Debajo de la tabla: n de las filas `[sint. no incl.]` (debe ser 0 en los bloques; si no, esas ejecuciones se
cuentan aparte por E2 y no se mezclan), ensayos con `tts_ocupado_en_t5 = true` (JSON `tts_ocupado_en_t5.si`,
E4) e incidencias de voz (E7).

**T5. Casos límite del Bloque 2**

| Caso | Ejecuciones | Estado observado: aceptada / rech. confianza / rech. manos / rech. incompleta | Observaciones |
|---|---|---|---|
| CL-1 … CL-6c | | | |

**T6. Retardo de audio**: n, media, DE, mediana y p90 de d, con dispositivo de salida y modalidad de
grabación; o «no medido».

**T7. Condiciones experimentales**: filas de la hoja de sesión agregadas por bloque, con las diferencias
frente a las condiciones de registro del corpus (cámara COLOR_BODY 1920 × 1080 a 24 fotogramas por segundo
[REG 483-485 · N-13]) [MET 498-502].

---

## 10. Criterios de exclusión y de repetición

**Principio.** Ningún registro se borra ni se edita en el JSONL. Toda exclusión se decide con un criterio
fijado **antes** de la primera sesión, se aplica en el análisis, se cuenta y se reporta con su motivo. En
el Bloque 1 no se excluyen ejecuciones por el resultado obtenido: la única depuración que prevé el
anteproyecto es la de las variantes declaradas, y se reporta por duplicado [MET 229-233].

### 10.1 Criterios de exclusión (*propuestos*)

| Id | Criterio | Aplica a | Justificación |
|---|---|---|---|
| E1 | Registros anteriores a la aprobación de este protocolo: `registros/latencia_2026-09-14.jsonl` y `registros/latencia_2026-09-15.jsonl` | todo el reporte en línea | Umbral 0,5 y síntesis no incluida en t5 [REG 948-978 · N-26; REG 1188-1198 · N-32; REG 1263-1266 · N-34]. Comprobado en esta revisión: el primero tiene 62 ensayos en 3 sesiones (N-26 describe 59 en las dos primeras; la tercera, de 3 ensayos, es posterior) y el segundo 9; todos con umbral 0,5 y ningún ensayo con audio sintetizado antes de t5 (0 de 15 y 0 de 2) |
| E2 | Ensayo con configuración distinta de la sección 3 (modelo, umbral, representación, regla de mano ausente, modo, `tts_modo`, o `tts_sintesis_antes_de_t5 = false`) | latencias; en el Bloque 1 también exactitud | Las etapas no serían comparables; se cuenta aparte |
| E3 | Fallo técnico documentado en la hoja (congelamiento de la cámara, cierre del programa, persona ajena en cuadro con la ventana abierta) | según el caso | No es una ejecución válida del protocolo; la ventana se aborta con X cuando es posible y queda como `rechazada_incompleta` con la incidencia anotada |
| E4 | `tts_ocupado_en_t5 = true` | se incluye en L_total (t5 es la orden por definición) pero se cuenta y se reporta | La reproducción real se retrasa y L_total no lo ve [REG 1296-1298 · N-34]; el procedimiento 5.2.8 busca que no ocurra |
| E5 | Ejecuciones de familiarización (si PD-16 las admite) | todo | Se hacen con `--sin-registro` o se anotan como tales antes de empezar |
| E6 | Ensayos de la caracterización del retardo de audio | Bloques 1 y 2 | Van a una carpeta propia y usan una configuración dedicada (sección 8.3) |
| E7 | Error de reproducción de la voz: `tts_errores_reproduccion` vale más que en el registro anterior de la sesión (o más de 0 en el primero), o la línea «[AVISO TTS] Voz de la sesión» informa errores o locuciones descartadas. **D-13 fija el alcance: el «audio dudoso» de una sesión va desde el ensayo ANTERIOR al registro donde sube el contador**, porque la locución suena en segundo plano después de t5 y el fallo aparece en el registro siguiente; `latencia_resumen.py` lo reporta como `audio_dudoso` y conserva el conteo estrecho como `ensayos_desde_el_primer_error` | se conserva en las latencias (t5 es la orden de reproducción y el fallo ocurre después) y en la exactitud del Bloque 1, y se cuenta y reporta como incidencia; en la caracterización del retardo de audio el ensayo sin audio se excluye | El JSONL no identifica con certeza qué ensayo no sonó (R20); qué hacer con la presentación afectada: PD-31 |

### 10.2 Criterios de repetición

1. **Rechazo por confianza** en el Bloque 1: el sistema pide repetir [MET 184-186]. Cada repetición es una
   ejecución nueva, vinculada a la misma presentación y numerada como intento; todas cuentan en la
   totalidad. Número máximo de intentos y cómo entran al denominador: PD-14 y PD-15.
2. **Rechazo por manos o secuencia incompleta** en el Bloque 1: misma regla que el punto anterior (PD-14).
3. **Incidencia técnica (E3)**: la presentación puede repetirse; la ejecución fallida se conserva en el
   JSONL y se excluye con su motivo.
4. **Caso límite no logrado** en el Bloque 2 (por ejemplo, se quiso ocultar las manos y el extractor las
   detectó): la ejecución se registra con el estado observado, no se reetiqueta; puede repetirse la
   provocación como ejecución nueva.
5. **Sesión interrumpida**: las ejecuciones realizadas se conservan; la sesión se completa en otra fecha
   con las presentaciones pendientes, declarándolo (PD-04).
6. **Revocación del consentimiento**: no hay repetición; se suprimen los registros de la persona y el
   Bloque 1 se retira del reporte [ETI 67-69].
7. **Fallo de reproducción de la voz (E7)**: la ejecución se conserva en el JSONL. Si se repite la
   presentación y si la sesión se detiene ante errores repetidos: PD-31.

---

## 11. Riesgos y límites declarados

| Id | Riesgo o límite | Efecto sobre la interpretación | Fuente |
|---|---|---|---|
| R1 | **Umbral elegido sobre la validación de señantes vistos.** El criterio del anteproyecto favorece un umbral permisivo: en la prueba LOSO con señantes nuevos, 0,20 acepta 837 de 997 y 297 de ellas son erróneas (35,5 %) | En línea, con una persona nueva, cabe esperar más aceptaciones erróneas que en validación. No se reajusta mirando los bloques; otro umbral exige una decisión nueva y cambiar primero el método | [MET 187-189]; [REG 1225-1227 · D-09; REG 1250-1258 · N-33] |
| R2 | **Síntesis de ≈450 ms dentro de t5.** Desde N-34 la etapa de síntesis incluye la síntesis real (445-463 ms por frase en N-34; mediana 442 ms en la revisión independiente). En esta revisión, 15 síntesis a archivo con `say` (voz Paulina, 150 ppm), sin reproducir: mediana 427,9 ms (425,9-449,5), medidas fuera del prototipo | L_resp y L_total quedan ≈0,45 s por encima de las pruebas de N-26 y N-32 (L_resp 34 ms), que no son comparables | [REG 1275-1281 · N-34; REG 1294-1295] |
| R3 | **Pausa del bucle.** Tras cada ventana aceptada el bucle queda detenido mientras se sintetiza; no se ha verificado con cámara si después llegan fotogramas acumulados en ráfaga | Puede alterar la tasa medida antes de la siguiente ventana y, con ventana automática, su N | [REG 1279-1281; REG 1299-1300 · N-34]; `translator.py`, `fps_medida` y `_cerrar_ventana` (reinicio de la medición de la tasa) |
| R4 | **t0 tras la pulsación.** t0 se sella al leer la tecla, hasta un periodo de bucle después de la pulsación física | W y L_total empiezan algo después de la pulsación; el retardo queda dentro de d (sección 8.2) | [REG 567-568 · N-14] |
| R5 | **t5 no es el sonido.** El retardo del controlador y del dispositivo de audio no entra en la medición por software | Se caracteriza aparte y no se resta | [MET 326-331; MET 513-517] |
| R6 | **Secuencias sin inicio o fin claros.** El anteproyecto dice que se descartan; el prototipo solo descarta por X o por menos de 15 fotogramas y no detecta una seña truncada en los bordes de la ventana (desde el 22-nov-2026 la marca como aviso en el JSONL, sin descartarla: N-50, opción B) | Una ejecución truncada se infiere y cuenta como error o acierto; debe declararse y discutirse (CL-6b) | [MET 192-193]; `translator.py`, `_cerrar_ventana` |
| R7 | **Seña fuera del vocabulario sin estado propio** | Solo puede terminar en rechazo por confianza o en aceptación de una clase del vocabulario | [MET 29-31; MET 250-251]; `translator.py`, `_inferir_y_emitir` |
| R8 | **Carga del equipo.** En la corrida completa de 1.c el costo del extractor subió de ≈11,5 ms a ≈15 ms por fotograma; N-21 atribuye como causa probable la sincronización masiva de Google Drive | El costo por fotograma y la tasa efectiva dependen de la carga; hay que registrarla | [REG 800-808 · N-21] |
| R9 | **Condiciones distintas de las del corpus**: cámara, resolución (el corpus es 1920 × 1080 a 24 fps; el prototipo capturaba por defecto 640 × 480 y, desde el 22-nov-2026, 1280 × 720), iluminación, fondo, encuadre | Acotan la interpretación del Bloque 1 y deben documentarse | [MET 498-502; ALC 35-41]; [REG 483-485 · N-13]; `realtime.py`, argumentos `--ancho` y `--alto` |
| R10 | **Modelo entrenado con los cinco señantes del corpus** (partición aleatoria) y seleccionado por validación | En el Bloque 1 la persona es un sexto señante no visto; la referencia más cercana es el LOSO, cuyo top-1 varía entre 0,460 y 0,685 según el señante excluido | [MET 265-268]; [REG 1171-1181 · N-31]; sección 2.1 |
| R11 | **Pocas ejecuciones por clase.** 50 señas por r repeticiones, con r limitado por las horas de sesión | Precision, recall y F1 por clase tendrán alta incertidumbre; el percentil 90 con n pequeño depende de muy pocas observaciones | [MET 223-225; REC 142-146] |
| R12 | **Conversión de glosas en desarrollo** (3.a, meses 3-4) | La etapa de conversión y el enunciado cambian si cambia `grammar_mapper.py` durante los bloques | [CRO 50]; [REG 968-970 · N-26]; PD-25 |
| R13 | **Voz no colombiana.** No hay voz es-CO instalada; se usa Paulina (es-MX) | Afecta la naturalidad, no el reconocimiento; cambiar de voz puede cambiar el tiempo de síntesis | [REG 463-466 · N-12]; PD-29 |
| R14 | **Transcripción manual de la glosa objetivo.** El JSONL no la registra | Riesgo de desalinear hoja y registro; se mitiga con la comprobación 5.4.2 o con PD-22 | `translator.py`, `_registro_ensayo` |
| R15 | **Resumen de latencia sin separación por fecha, sesión ni carpeta.** Resume todo lo que recibe. Ya reporta W y captura sobre las ventanas inferidas y separa síntesis, L_total y L_resp por `tts_sintesis_antes_de_t5`, pero `magnitudes_ms` y la `tasa_rechazo` global del JSON conservan cálculos que mezclan grupos, y no agrega los campos de voz (`tts_errores_reproduccion`) | Hay que darle solo la carpeta del bloque (E1) y aplicar E2 antes de resumir; tomar cada magnitud de la clave que indiquen las `advertencias`; revisar la voz en el JSONL (5.4, punto 4) | `latencia_resumen.py`, `resumir` y `tabla`; [REG 566 · N-14; REG 1301 · N-34] |
| R16 | **Ruta absoluta en el campo `modelo`.** En el registro del 15-sep el campo guarda la ruta absoluta del modelo, que incluye el nombre de la cuenta del autor en la ruta de Google Drive (comprobado en esta revisión) | No identifica a la persona del Bloque 1, pero conviene considerarlo antes de publicar registros (RE7) | `realtime.py`, argumento `--model-path` (por defecto, `MODELO_POR_DEFECTO` como ruta absoluta); PD-27 |
| R17 | **Dependencia de 0.c y del comité.** A la fecha no consta en el registro la vinculación ni la consulta | Sin ellas no hay Bloque 1 ni cifra de exactitud en línea | [MET 269-271; REC 61-66; CRO 106-108] |
| R18 | **Sin línea base clásica** | No afecta a 4.b, pero sí a la comparación de 4.d | [MET 131-136; MET 519-521]; [REG 431-432 · N-10] |
| R19 | **Ausencia parcial de manos y oclusión.** §2.2.6 dice que, cuando no se detectan las manos, salen del cuadro o hay oclusión, el sistema no infiere; el prototipo solo deja de inferir si faltan las dos manos (o la postura) en más de la mitad de la ventana, e infiere con una mano ocluida enmascarándola (D-07). El registro interpreta §2.2.6 en ese sentido [REG 852-853 · D-07] | Una ejecución con oclusión parcial se infiere y cuenta como acierto o error; debe declararse junto a CL-2 y CL-3 y discutirse con el director (PD-09) | [MET 191-192; ALC 150-151]; `translator.py`, `_cerrar_ventana` |
| R20 | **Fallo de voz no atribuible con exactitud a un ensayo.** El audio suena en segundo plano después de t5; `tts_errores_reproduccion` es acumulado y el fallo puede aparecer en los registros siguientes; el del último ensayo, en general, solo lo informa la línea de cierre, y lo que el hilo descarte después del cierre no se cuenta | Un ensayo registrado como emitido pudo no sonar. No cambia L_total (t5 es la orden), pero sí la interacción y la caracterización de la sección 8 | `translator.py`, `_estado_voz` y `LSCTranslator.close`; `tts.py`, `TTSModule._en_hilo`; E7, PD-31 |

---

## 12. POR DECIDIR CON EL DIRECTOR

Cada punto es un parámetro que el anteproyecto deja abierto o que el código fija por defecto sin decisión
registrada. Ninguna opción trae una cifra inventada: donde hace falta un número, la opción dice cómo se
obtiene. Lo que se decida se registra como D-nn antes de la primera sesión.

### PD-01. Repeticiones por seña en el Bloque 1
El anteproyecto: cubre las 50 señas y el número de repeticiones lo fija 4.a en función de las horas de
sesión acordadas con la persona, y se reporta [MET 223-225; MET 475-476; REC 142-146].
Cálculo sugerido: r = ⌊(H · 3600 − T_prep − T_pausas) / (50 · T_ejec · (1 + p_rep))⌋, donde H son las horas
acordadas, T_ejec el tiempo medio por presentación (presentación, reposo, W, declaración, registro y
audio), p_rep la proporción esperada de repeticiones (PD-14) y T_prep y T_pausas los tiempos de
preparación y de pausa. T_ejec y p_rep no están medidos.
- **A. Medir T_ejec en una sesión piloto del Bloque 2** (el autor simula el procedimiento completo) y
  calcular r con las horas acordadas. *Pro*: r se deriva de datos propios. *Contra*: el ritmo de la persona
  vinculada puede diferir del del autor; p_rep no se conoce de antemano.
- **B. Fijar primero r y cotizar las horas que resulten.** *Pro*: el diseño manda sobre el presupuesto.
  *Contra*: el presupuesto de la persona está sin cotizar [REC 140-148] y puede no alcanzar.
- **C. Fijar las horas y ejecutar rondas completas de las 50 señas hasta agotarlas.** *Pro*: aprovecha todo
  el tiempo. *Contra*: r queda determinado durante la ejecución; debe declararse la regla de parada de
  antemano y detenerse solo al terminar una ronda completa, para que todas las señas tengan el mismo r.

### PD-02. Número de ejecuciones del Bloque 2
El anteproyecto: el bloque acumula «el número de ejecuciones que exige reportar la latencia con las
estadísticas» fijadas, sin dar un número [MET 248-249].
- **A. Igualar la estructura del Bloque 1** (50 señas por r). *Pro*: latencias comparables entre bloques.
  *Contra*: si el Bloque 1 no existe (contingencia), falta la referencia.
- **B. Fijar un n mínimo de ejecuciones aceptadas antes de empezar**, independiente del Bloque 1. *Pro*:
  garantiza n para L_total y L_resp, que solo existen en ejecuciones aceptadas (sección 6.3). *Contra*: el
  número de ejecuciones totales depende de la tasa de rechazo del autor.
- **C. Regla de parada por estabilidad de la media y del p90.** *Pro*: se detiene cuando los estadísticos
  dejan de moverse. *Contra*: parar mirando los datos introduce sesgo si la regla no se fija antes; más
  compleja de declarar.

### PD-03. Orden de presentación de las glosas (Bloque 1 y ejecuciones normales del Bloque 2)
- **A. Orden fijo (por identificador 0-49).** *Pro*: simple, fácil de seguir. *Contra*: confunde seña con
  posición en la sesión (fatiga, aprendizaje, cambio de iluminación).
- **B. Aleatorio sobre todas las presentaciones, con semilla anotada.** *Pro*: reparte los efectos de
  orden. *Contra*: una seña puede quedar concentrada al final de la sesión.
- **C. Rondas: cada ronda contiene las 50 señas en un orden aleatorio distinto, con semilla anotada.**
  *Pro*: equilibra repeticiones en el tiempo y permite cerrar sesiones al final de una ronda. *Contra*:
  exige preparar y seguir una lista por ronda.

### PD-04. Número de sesiones y ejecuciones por sesión
El anteproyecto exige fijar el número de ejecuciones por sesión en 4.a [MET 274-276].
- **A. Una sola sesión por bloque.** *Pro*: condiciones constantes. *Contra*: fatiga; menor tolerancia a
  incidencias.
- **B. Varias sesiones cortas, cada una con un número fijo de rondas completas (PD-03 C).** *Pro*: menos
  fatiga; cada sesión es analizable. *Contra*: cambian iluminación y montaje entre sesiones; exige
  reproducir el escenario.
- **C. Número de sesiones según la disponibilidad de la persona.** *Pro*: realista. *Contra*: el diseño
  queda sujeto a la agenda; hay que declarar la regla de cierre.

### PD-05. Pausas
- **A. Pausa fija cada cierto número de presentaciones.** *Pro*: reproducible. *Contra*: puede no
  coincidir con la fatiga real.
- **B. Pausa al final de cada ronda.** *Pro*: no parte rondas. *Contra*: las rondas de 50 pueden ser largas.
- **C. Pausa a demanda de la persona, anotada.** *Pro*: respeta a la persona (participación, no
  extracción [ETI 78-86]). *Contra*: menos homogéneo; debe anotarse cada pausa.

### PD-06. Iluminación y fondo
El anteproyecto: se especifican y se fijan en 4.a; se documentan las diferencias con el corpus [MET
274-276; MET 498-502].
- **A. Describir cualitativamente** (tipo de fuente, posición, ventanas cerradas o abiertas, fondo). *Pro*:
  sin instrumentos. *Contra*: poco reproducible.
- **B. Medir con un luxómetro o una aplicación de medición de luz** en posiciones fijas, con fuente
  artificial constante y fondo liso. *Pro*: reproducible y comparable entre sesiones. *Contra*: requiere el
  instrumento; una aplicación móvil es poco exacta.
- **C. Replicar en lo posible las condiciones descritas por LSC50** (según la caracterización 1.b). *Pro*:
  reduce la distancia con el corpus. *Contra*: puede no ser alcanzable con el espacio disponible; hay que
  documentar lo que no se logra.

### PD-07. Distancia a la cámara y encuadre
- **A. Distancia fija marcada en el piso y encuadre de cabeza a cintura con los codos dentro.** *Pro*:
  reproducible; mantiene hombros y manos en cuadro, necesarios para normalizar y para no rechazar por
  manos. *Contra*: el valor de la distancia depende de la cámara y de la resolución (PD-08).
- **B. Encuadre equivalente al del corpus COLOR_BODY** (según 1.b). *Pro*: más cerca del entrenamiento.
  *Contra*: exige ajustar cámara y distancia hasta reproducirlo.
- **C. Encuadre libre verificado en pantalla** (hombros y manos visibles en reposo y en el espacio de
  señado). *Pro*: rápido. *Contra*: varía entre sesiones; debe anotarse.

### PD-08. Cámara y resolución de captura
El anteproyecto prevé una cámara web RGB Full HD y presupuesta una Logitech C920s [REC 72-73; REC 104-105,
119]. No consta qué cámara usaron N-26 y N-32. El programa captura por defecto a 640 × 480
(`realtime.py`, argumentos `--ancho` y `--alto`). *Actualización del 22-nov-2026:* el defecto pasó a
1280 × 720 por la forma de la imagen (16:9 como el corpus; N-57) y por ser la resolución usada con el modelo
final; el arranque avisa si la cámara entrega otra forma (`docs/MODELO_TCN_IMPUTACION.md`, sección 7).
- **A. Cámara web externa Full HD a 1280 × 720 o 1920 × 1080.** *Pro*: coincide con el recurso del
  anteproyecto; más píxeles por mano. *Contra*: más costo de extracción por fotograma; hay que medir la tasa.
- **B. Cámara integrada del portátil a 640 × 480** (lo usado por defecto). *Pro*: sin montaje; comparable
  con las pruebas previas. *Contra*: no es el recurso declarado; menor resolución que el corpus.
- **C. Decidir tras una prueba corta del Bloque 2 con ambas** (tasa efectiva y detección de manos), antes de
  fijar. *Pro*: decisión con datos propios. *Contra*: una sesión más; esos ensayos no cuentan en el bloque.
En cualquier caso: índice de cámara verificado, resolución efectiva anotada y la misma en todos los bloques.

### PD-09. Parámetros de la ventana y del rechazo por manos
Vigentes: ventana automática de 2,5 s (N = round(2,5 × tasa medida)), tope 5 s, mínimo 15 fotogramas y
máximo de 0,5 de fotogramas sin manos [REG 443-446 · N-11]; `realtime.py`, argumentos de la ventana y
`--max-sin-manos`. El anteproyecto fija una ventana de longitud fija activada por la persona usuaria, sin
valor [MET 193-196], y no fija la proporción máxima sin manos [MET 191-192].
- **A. Mantener la ventana automática de 2,5 s.** *Pro*: justificada con la duración de los clips de LSC50
  [REG 443-446]; ya probada. *Contra*: N depende de la tasa medida justo antes de abrir (riesgo R3).
- **B. Fijar N con `--ventana-frames` igual al N observado con la cámara decidida.** *Pro*: la ventana es la
  misma en todos los ensayos, y W solo varía por la tasa. *Contra*: si la tasa cambia, W cambia.
- **C. Ensayar 3,0 s en el Bloque 2 antes de fijar** (alternativa anotada en `translator.py`, docstring de
  `LSCTranslator.__init__`, argumento `ventana_seg`). *Pro*: el 79 % de los clips dura ≤ 3,0 s frente al
  47 % ≤ 2,5 s [REG 445-446]. *Contra*: aumenta W y L_total; decidir sobre ensayos del autor no debe
  apoyarse en exactitud.
Mínimo de fotogramas y proporción sin manos: confirmar 15 y 0,5 o decidir otros antes de empezar.

### PD-10. Quién pulsa ESPACIO en el Bloque 1 y posición de reposo
La latencia total va «desde la activación de la ventana temporal por la persona usuaria» [MET 300-301].
- **A. La persona ejecutante pulsa.** *Pro*: literal respecto de la definición. *Contra*: parte de la
  ventana se consume en llevar la mano del teclado a la posición de reposo; la mano puede quedar fuera de
  cuadro al inicio.
- **B. El autor pulsa a una señal acordada de la persona.** *Pro*: la persona empieza en reposo y con las
  manos libres. *Contra*: introduce el tiempo de reacción del autor; hay que declararlo.
- **C. Pulsador o pedal.** *Pro*: manos libres para la persona y activación por ella. *Contra*: requiere
  hardware y comprobar que llega como la tecla ESPACIO.
Posición de reposo: definir si replica la del corpus, que inicia y termina en la misma posición [MET 285-286].

### PD-11. Medio de presentación de la glosa y de la secuencia de referencia
- **A. Tarjetas impresas con la glosa.** *Pro*: no interfiere con la pantalla ni con la cámara. *Contra*: no
  sirve para mostrar la secuencia de referencia.
- **B. Segunda pantalla con la glosa y, a demanda, el video de referencia del corpus.** *Pro*: cubre ambos
  requisitos [MET 225-226]. *Contra*: hay que mostrar el video sin copiarlo fuera del almacenamiento
  restringido [ETI 19-21]; la persona puede quedar mirando la pantalla y no a la cámara.
- **C. Presentación en LSC por una persona intérprete presente.** *Pro*: accesible. *Contra*: una tercera
  persona en la sala (fuera del cuadro), costo adicional [REC 140-148]; su ejecución podría servir de modelo
  y sesgar la forma de la seña.

### PD-12. Retroalimentación visible para la persona
- **A. La persona ve la pantalla del prototipo.** *Pro*: sabe cuándo abre y cierra la ventana. *Contra*: ver
  el resultado puede llevarla a adaptar la seña a lo que el sistema acepta (aprendizaje del sistema).
- **B. Pantalla orientada solo al autor; indicación de apertura y cierre por otra señal.** *Pro*: evita la
  adaptación. *Contra*: la persona no sabe si la ventana se cerró; hace falta una señal accesible (una luz
  o un gesto).
- **C. Mostrar solo la barra de progreso, ocultando la clase.** *Pro*: equilibrio. *Contra*: requiere un
  cambio de interfaz con revisión.

### PD-13. Momento, medio y categorías de la declaración de variante
- **Momento A: antes de conocer el resultado.** *Pro*: la declaración no depende de si el sistema acertó.
  *Contra*: la persona debe declarar enseguida de ejecutar.
- **Momento B: después del resultado.** *Pro*: más cómodo. *Contra*: riesgo de declarar variante cuando el
  sistema falla, que sesga la cifra depurada.
- **Categorías**: *sí / no* (lo que pide el anteproyecto [MET 226-228]) o *sí / no / no estoy segura*.
  *Pro de la tercera*: honestidad del dato. *Contra*: hay que decidir antes si «no estoy segura» cuenta como
  variante para el cálculo por duplicado.
- **Medio**: tarjetas o gesto acordado, respuesta escrita, o intérprete. Debe ser accesible [ETI 55-57].

### PD-14. Política ante un rechazo en el Bloque 1
- **A. No repetir: una ejecución por presentación.** *Pro*: denominador limpio; tiempo predecible. *Contra*:
  contradice en la práctica que el sistema pida repetir [MET 184-186].
- **B. Repetir hasta un número máximo fijo de intentos**, contando todos. *Pro*: refleja el uso real.
  *Contra*: varía el número de ejecuciones por seña; hay que decidir qué intento se usa para la exactitud.
- **C. Repetir solo tras rechazo por manos o secuencia incompleta, no tras rechazo por confianza.** *Pro*:
  separa fallos de captura de fallos del modelo. *Contra*: dos reglas distintas que declarar.

### PD-15. Denominador de la exactitud en línea
- **A. Top-1 y top-3 sobre las ejecuciones con predicción** (aceptadas y rechazadas por confianza), como en
  offline, y aparte tasa de rechazo y exactitud entre las aceptadas (como en D-09). *Pro*: separa el
  clasificador del umbral; coherente con N-33. *Contra*: excluye del top-1 las ejecuciones sin predicción,
  que deben reportarse aparte.
- **B. Sobre todas las ejecuciones, contando todo rechazo como error.** *Pro*: cifra del sistema completo.
  *Contra*: mezcla errores de clasificación con rechazos deliberados, que el anteproyecto pide distinguir
  [MET 197-198].
- **C. Reportar A y B.** *Pro*: completo. *Contra*: más tablas; hay que nombrar cada cifra sin ambigüedad.
Con PD-14 B: definir si cuenta el primer intento, el último o todos.

### PD-16. Ejecuciones de familiarización
- **A. Ninguna.** *Pro*: todas las ejecuciones cuentan. *Contra*: las primeras reflejan el aprendizaje del
  procedimiento, no la seña.
- **B. Unas pocas con señas fuera del orden de la sesión, sin registro (`--sin-registro`).** *Pro*: la
  persona conoce el procedimiento. *Contra*: hay que fijar cuántas y con qué señas.
- **C. Familiarización registrada y excluida por E5.** *Pro*: queda trazabilidad. *Contra*: registros que
  no se usan.

### PD-17. Casos límite del Bloque 2
- **Ejecuciones por caso**: número fijo por caso o proporcional al total de PD-02. *Pro del fijo*: todos los
  casos quedan ejercitados por igual. *Contra*: CL-1 y CL-6c no son provocables a voluntad.
- **Señas fuera del vocabulario (CL-5)**: (A) gestos cotidianos no lingüísticos; (B) señas de otros
  vocabularios de la LSC que el anteproyecto usa solo como referencia (LSC70, LSC-54) [MET 67-69]. *Pro de B*:
  más cercano al uso real. *Contra de B*: el autor no es usuario fluido y su ejecución puede no ser la seña.
- **Reporte de CL-5**: el conteo de ejecuciones fuera del vocabulario que terminan en `aceptada` es un
  conteo de estados, permitido [MET 261-262], pero se acerca a una cifra de error; decidir si se reporta
  como conteo o se omite.
- **CL-6b**: decidir si se declara como límite del prototipo (R6) o se abre una decisión para implementar el
  descarte. *Actualización del 22-nov-2026:* implementada la opción B de N-50 (protocolo de reposo, X y aviso
  en el JSONL sin rechazar; `docs/MODELO_TCN_IMPUTACION.md`, sección 7.2).

### PD-18. Caracterización del retardo de audio
- **Modalidad**: (A) solo audio con dispositivo externo; (B) video de alta velocidad del teclado y del
  parlante, sin personas. *Pro de A*: no hay video; resolución de muestra alta. *Contra de A*: el clic de la
  tecla puede ser débil. *Pro de B*: la pulsación se ve. *Contra de B*: es video (aunque sin personas) y su
  resolución temporal está limitada por los fotogramas.
- **Número de ensayos**: fijo antes de empezar. *Pro*: evita parar según el resultado. *Contra*: no hay dato
  previo de la dispersión.
- **Umbral de la corrida dedicada**: `--umbral 0` para que todo ensayo inferido emita audio, o el umbral de
  0,20 con señas que el modelo acepte. *Pro de 0*: todos los ensayos sirven. *Contra*: configuración
  distinta de la de los bloques (el retardo de audio no depende del umbral, pero hay que declararlo).
- **Detección de los instantes**: manual con un editor de audio o por umbral de amplitud con un guion.
  *Pro del guion*: reproducible. *Contra*: hay que escribirlo y revisarlo.
- **Conservación del archivo**: conservarlo como evidencia (solo contiene clic y voz sintética) o borrarlo
  tras extraer los instantes. Decidir con el criterio de §1.5.

### PD-19. Ventanas que entran en el promedio de W y de captura
Estado del código: `latencia_resumen.py` (`resumir` y `tabla`) ya imprime como filas principales
`captura [inferidas]` y `W [inferidas]` (JSON `ventanas_inferidas`) y, como referencia, `captura [todas]` y
`W [todas]`, que coinciden con `magnitudes_ms`; cuando estas incluyen ventanas no inferidas, `advertencias`
lo dice. Que B sea la fila principal del guion no decide qué se reporta: lo decide el director.
- **A. Todas las ventanas cerradas con fotogramas** (referencia `[todas]` y `magnitudes_ms`). *Pro*: más n.
  *Contra*: incluye ventanas abortadas (W más corto) y rechazadas por manos; en el JSON va con advertencia.
- **B. Solo ventanas inferidas** (aceptadas y rechazadas por confianza) [REG 566 · N-14] (filas
  `[inferidas]`). *Pro*: W describe las ventanas que llegan al modelo; el resumen ya la calcula. *Contra*:
  menos n que A; no es el mismo conjunto que L_total y L_resp.
- **C. Solo ventanas aceptadas**, igual que L_total. *Pro*: mismo conjunto que L_total y L_resp, de modo que
  L_total = W + L_resp también en los promedios. *Contra*: menos n; el resumen no la calcula: exige filtrar el
  JSONL o cambiar el guion con revisión.

### PD-20. Estadísticos adicionales
El anteproyecto fija media, desviación estándar, mediana y p90 para las magnitudes de latencia [MET
324-326] y enumera las métricas de exactitud que se reportan, sin fijar estadísticos de incertidumbre
[MET 276-280].
- **A. Solo lo que fija el anteproyecto.** *Pro*: fidelidad. *Contra*: sin medida de incertidumbre de las
  proporciones del Bloque 1.
- **B. Añadir intervalos de confianza para las proporciones del Bloque 1 y para la media de las latencias.**
  *Pro*: con pocas ejecuciones la incertidumbre es grande (R11). *Contra*: no está previsto; debe
  declararse como adición y no como umbral.

### PD-21. Verificación de la no participación en LSC50
[MET 233-234; REC 32-33]
- **A. Declaración escrita de la persona**, con la información que el artículo descriptor publica sobre la
  grabación. *Pro*: simple. *Contra*: depende de la memoria y del conocimiento de la persona.
- **B. Consulta a los autores del corpus.** *Pro*: más sólida. *Contra*: puede no responderse a tiempo.
- **C. A y B.** *Pro*: constancia doble. *Contra*: más trámite.

### PD-22. Registro de la glosa objetivo y del bloque
- **A. Hoja manual unida por (`sesion`, `id_ensayo`)**, sin tocar el código. *Pro*: sin cambios en el
  prototipo congelado. *Contra*: riesgo de transcripción (R14).
- **B. Cambio de código**: parámetros de sesión (`--bloque`, lista de presentación) que escriban la glosa
  objetivo y el bloque en el JSONL, con revisión independiente. *Pro*: unión automática y verificable.
  *Contra*: toca archivos del prototipo antes de congelarlo; exige pruebas nuevas.

### PD-23. Carga del equipo y sincronización de Google Drive
- **A. Pausar la sincronización durante las sesiones.** *Pro*: evita la carga observada en N-21. *Contra*:
  hay que recordar reanudarla.
- **B. Escribir los registros de sesión fuera de Google Drive y copiarlos después.** *Pro*: sin escrituras
  sincronizadas durante la sesión. *Contra*: paso manual adicional.
- **C. No intervenir y registrar la carga.** *Pro*: condiciones de uso reales. *Contra*: el costo del
  extractor puede quedar inflado (R8).

### PD-24. Salida de audio y volumen
- **A. Parlantes integrados del portátil.** *Pro*: sin montaje; dispositivo por defecto actual. *Contra*:
  volumen limitado.
- **B. Parlante externo con cable.** *Pro*: mejor audibilidad para la persona oyente de la interacción.
  *Contra*: otro dispositivo que caracterizar.
En cualquier caso, sin dispositivos inalámbricos y con la misma salida en la caracterización de la
sección 8 y en las sesiones.

### PD-25. Momento de los bloques respecto de la conversión de glosas (3.a)
- **A. Ejecutar los bloques con el `grammar_mapper.py` vigente y congelado.** *Pro*: no depende de 3.a.
  *Contra*: el enunciado y la etapa de conversión no serán los finales.
- **B. Esperar al cierre de 3.a** [CRO 50]. *Pro*: mide el sistema final. *Contra*: compromete el mes 4,
  que concentra 4.b a 4.e [CRO 55-59].
- **C. Bloque 2 ahora y Bloque 1 tras 3.a, remidiendo la conversión.** *Pro*: aprovecha el tiempo. *Contra*:
  latencias de los dos bloques con código distinto; hay que declararlo.

### PD-26. Definición operativa del «vocabulario efectivamente soportado»
[MET 517-518; RES 58-59]
- **A. Número de clases que el modelo puede emitir** (50). *Pro*: objetivo y verificable. *Contra*: no dice
  nada del uso en línea.
- **B. Además, en el Bloque 1, número de señas con al menos una ejecución reconocida correctamente.** *Pro*:
  informativo. *Contra*: depende de r y es una cifra nueva que habría que declarar como tal.

### PD-27. Custodia y publicación de los registros y hojas
El README deja abierto si se versionan los registros [README, sección «Registro de latencia (§2.2.7)»]. Para el
Bloque 1, los registros derivados se conservan solo para reportar RE5 y no se incorporan a ninguna
publicación [ETI 60-61; ETI 65-67].
- **A. No versionar ni publicar los registros del Bloque 1**; sí los del Bloque 2. *Pro*: coherente con §1.5.
  *Contra*: la reproducibilidad del Bloque 1 depende de la custodia local.
- **B. No versionar ningún registro de los bloques; publicar solo los resúmenes.** *Pro*: uniforme.
  *Contra*: se pierde la trazabilidad pública del Bloque 2.
- **C. Versionar todos sin publicar el repositorio hasta revisarlos.** *Pro*: trazabilidad. *Contra*: riesgo
  de publicarlos por error; revisar también la ruta absoluta del campo `modelo` (R16).

### PD-28. Confirmación del umbral 0,20
D-09 lo fijó con el criterio del anteproyecto y N-33 declaró su limitación [REG 1201-1258].
- **A. Mantener 0,20 durante los bloques.** *Pro*: decisión tomada antes de ver datos en línea. *Contra*:
  aceptará más errores con personas nuevas (R1).
- **B. Abrir una decisión nueva antes de los bloques** (validación anidada por señante y reentrenamiento,
  el camino que indica N-33 [REG 1256-1258]). *Pro*: umbral más fiel para personas nuevas. *Contra*: tiempo
  de reentrenamiento y nueva revisión; cambia el modelo de la sección 3.

### PD-29. Voz del sintetizador
- **A. Mantener Paulina (es-MX).** *Pro*: medida y probada (N-34). *Contra*: no es variante colombiana.
- **B. Instalar una voz es-CO del sistema antes de congelar** [REG 464-466 · N-12]. *Pro*: variante del país
  del proyecto. *Contra*: hay que volver a medir la síntesis y verificar la selección de voz.

### PD-30. Apoyo de intérprete durante las sesiones del Bloque 1
El anteproyecto exige consentimiento accesible, en LSC o con intérprete si la persona es sorda [ETI 55-57],
pero no fija cómo se comunica el autor con la persona durante la sesión.
- **A. Intérprete presente (fuera de cuadro).** *Pro*: comunicación plena. *Contra*: costo [REC 140-148];
  otra persona en la sala.
- **B. Comunicación escrita y señales acordadas.** *Pro*: sin costo adicional. *Contra*: más lento; posible
  malentendido en la declaración de variantes.
- **C. Depende del perfil**: si la persona vinculada es intérprete oyente, no hace falta. *Pro*: ajustado.
  *Contra*: la decisión queda condicionada a 0.c.

### PD-31. Fallos de reproducción de la voz durante una sesión
El anteproyecto define t5 como la emisión de la orden de reproducción y deja fuera el sonido [MET 326-331],
pero no fija qué hacer si el audio no llega a sonar. El JSONL registra los errores acumulados y el cierre
imprime un aviso (sección 9.1); `realtime.py` no avisa en pantalla cuando el contador sube, y la consola solo
muestra el primer error de cada tipo.
- **A. Conservar y contar (E7), sin repetir.** *Pro*: no altera el diseño; la latencia hasta t5 y el
  reconocimiento no dependen del sonido. *Contra*: en el Bloque 1 esa presentación queda sin voz.
- **B. Repetir la presentación cuando se note que no sonó.** *Pro*: la interacción queda completa. *Contra*:
  el JSONL no permite atribuir el fallo con certeza (R20); cambia el número de ejecuciones por seña (PD-14).
- **C. Detener la sesión ante un error, revisar la salida de audio (PD-24) y reanudar declarándolo
  (10.2, punto 5).** *Pro*: evita acumular ejecuciones sin voz. *Contra*: interrumpe a la persona; hay que
  fijar con cuántos errores se detiene.
Un aviso en pantalla cuando `tts_errores_reproduccion` sube sería un cambio de código con revisión antes de
congelar.

---

## Anexo A. Estado de los prerrequisitos a la fecha del borrador

| Prerrequisito | Estado | Fuente |
|---|---|---|
| Prototipo integrado con ventana por tecla, estados e instrumentación (3.c) | Hecho | [REG 435-456 · N-11; REG 753-785 · N-20] |
| Extractor único y modelo canónico «manos» con procedencia | Hecho | [REG 736-750 · N-19; REG 1147-1185 · N-31] |
| Umbral por validación | Hecho (0,20) | [REG 1230-1258 · N-33] |
| Síntesis antes de t5 | Hecho | [REG 1261-1302 · N-34] |
| Prueba en vivo con cámara del modelo «manos» | Breve (9 ensayos, umbral 0,5) | [REG 1188-1198 · N-32] |
| Vinculación de la persona (0.c) y consulta al comité | No consta en el registro | — |
| Conversión de glosas (3.a) | En desarrollo | [CRO 50]; [REG 968-970 · N-26] |
| Caracterización del retardo de audio | No realizada | — |
| Línea base clásica (para 4.d) | No existe | [REG 431-432 · N-10] |
| Guion de análisis de exactitud en línea (Bloque 1) | No existe: `latencia_resumen.py` resume latencias, estados y umbrales, no exactitud | `latencia_resumen.py`, docstring del módulo |
