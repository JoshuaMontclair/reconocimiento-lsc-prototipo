# Verificación de citas: párrafo de la arquitectura

Fecha: 30 de septiembre de 2026. Solo lectura del anteproyecto: no se tocaron `13_referencias.tex`, `08_marco.tex` ni ningún otro archivo.

## Fuentes consultadas

| ref | Fuente usada | Tipo de acceso |
|---|---|---|
| ref22 Dewanto et al. | `DOIs/Scoping_Review_Of_...Integration.pdf` (IEEE Xplore, descargado por la Javeriana) | texto completo |
| ref23 Srivastava et al. | `DOIs/f23.txt` y `DOIs/s11277-024-11356-0.pdf` | texto completo |
| ref24 Varshini y Rukmani | `DOIs/MP-GestLSTM_real_time_gesture_detection_using_Medi.pdf` | texto completo |
| ref25 Uddin et al. | página del editor (mdpi.com/2414-4088/9/3/23), acceso abierto | texto completo |
| ref26 Natarajan et al. | `DOIs/Development_of_an_End-to-End_...Generation.pdf` | texto completo |
| ref41 Rivera-Vásquez y González-Serrano | `DOIs/r41.txt` y `DOIs/ref41_...e18059.pdf` | texto completo |
| ref42 Muñoz Galindez y Vargas Cañas | `DOIs/f42.txt` y `DOIs/s00521-025-11326-y.pdf` | texto completo |
| ref10 Camgöz et al. | `DOIs/Sign_Language_Transformers_...Translation.pdf` | texto completo |
| ref6 Vaswani et al. | arXiv:1706.03762 (versión HTML, §6.3) | texto completo |

Metadatos (autores, título, volumen, número, páginas y fecha) cotejados con la API de Crossref. La resolución de los DOI se comprobó en doi.org.

Para las cifras del propio proyecto se cotejaron `models/linea_base_4d_2026-09-15/comparacion.md`, `models/senas_cortadas_2026-09-22/recomendacion.md` y `src/train_final_100.py` (`build_model`).

---

## 1. Verificación por afirmación

Estados: **CONFIRMADA** (vista en la fuente), **PARCIAL**, **INCORRECTA**, **NO VERIFICABLE**.

### 1.1 Afirmaciones del párrafo

| # | Afirmación | Fuente | Estado | Evidencia y comentario |
|---|---|---|---|---|
| P1 | «El esquema de puntos característicos de MediaPipe seguido de redes recurrentes es **el más extendido en la literatura**» [ref22–ref26] | ref22 | **PARCIAL** | La revisión concluye que la técnica más usada es «MediaPipe for feature extractor and Long Short Term Memory (LSTM) for temporal features» (resumen, p. 451). En la Fig. 6 (p. 455), LSTM es la técnica más frecuente. Pero el universo de la revisión son **solo los estudios que ya usan MediaPipe**: 209 artículos de Google Scholar vía Publish or Perish, filtrados por tener citas, y 38 retenidos por palabra clave en título y resumen (pp. 452–454). Dice que **dentro de MediaPipe + aprendizaje profundo** la LSTM es lo más frecuente. **No** dice que MediaPipe + RNN sea el esquema más extendido del reconocimiento de lengua de señas en general. |
| P1b | Ídem | ref23–ref26 | **INCORRECTA como respaldo de «el más extendido»** | Son estudios individuales: muestran que el esquema **se usa**, no que sea el más extendido. Además, ref26 no es «puntos de MediaPipe → RNN»: su reconocimiento usa VGG-19 sobre imágenes de 254 × 254 y luego Bi-LSTM (pp. 104362–104363, §III.A). MediaPipe aparece como extractor de pose. Ver la redacción propuesta en la §4. |
| P2 | El esquema «se ha aplicado a la LSC» [ref41, ref42] | ref41, ref42 | **CONFIRMADA** | ref41: tres capas LSTM sobre puntos de MediaPipe de manos y postura (Tabla 3, p. 5; resumen). ref42: «hand landmarks are extracted from images using MediaPipe … processed with a BILSTM network» (resumen, p. 18479). |
| P3 | La red supera a la mejor línea base clásica en 16 puntos de exactitud top-1 con validación independiente del señante (p = 2,9 × 10⁻¹⁸) | registros internos | **CONFIRMADA (interna)** | `comparacion.md` §6–7: LOSO agregado sobre 997 predicciones, red 0,580 frente a Random Forest 0,418 (= 16,2 puntos), McNemar χ² con corrección de continuidad p = 2,89 × 10⁻¹⁸. **Matiz que conviene escribir:** la línea base recibe la secuencia **resumida en estadísticos** y pierde el orden temporal (§8). Con la misma aumentación que la red, Random Forest sube a 0,442: la diferencia baja a 13,8 puntos, con p = 2,99 × 10⁻¹⁵ (§9). El resultado justifica modelar el orden temporal frente a un resumen. **No** justifica por separado cada componente (Conv1D, bidireccionalidad, atención): no hay ablación. |
| P4 | La atención pondera los instantes informativos «frente al **reposo inicial y final, común a todas las señas del corpus**» | registros internos (N-50) | **INCORRECTA en lo del reposo inicial** | `recomendacion.md` punto 1: «El inicio de los clips del corpus no está en reposo en la mayoría de los casos; el final sí». La velocidad mediana de las muñecas es 0,0132 al inicio y 0,0027 al final. El inicio está en movimiento en el 52–100 % de los clips, según el señante. Además, no hay ningún análisis de los pesos de atención que muestre que la red baja el peso del reposo. Esto es una hipótesis de diseño, no un hallazgo. |
| P5 | «la bidireccionalidad es admisible porque el sistema clasifica señas aisladas una vez cerrada la ventana de captura» | — (sin cita en el párrafo) | Sin cita; **respaldable** | Sincan y Keles (2020) lo dicen casi literalmente: en tiempo real, la pasada hacia atrás obliga a almacenar los fotogramas, y la evaluación solo puede empezar «after all the frames of an isolated sign is completed» (§5.2.2). Ver [refAtn1]. |
| P6 | «la LSTM modela el orden de la secuencia, que distingue señas con configuraciones similares» | — (sin cita) | Sin cita; **respaldable** | ref42 lo documenta para la LSC: los números del 6 al 9 son idénticos a los del 1 al 4 salvo por el movimiento de los dedos, y la N y la Ñ comparten forma (p. 18487). Sincan y Keles describen «doktor»/«dakika», que difieren solo en la repetición (§3). |
| P7 | «la convolución captura patrones de movimiento de corta duración» | — (sin cita) | Sin cita | Es una justificación de diseño razonable (núcleo de tamaño 3 en el eje temporal), pero ninguna fuente revisada aquí la sostiene. Puede quedar como razonamiento propio o citar el manual [ref4]. Esto último no se verificó en esta tarea. |

### 1.2 Afirmaciones que circularon en la conversación

| # | Afirmación | Estado | Evidencia |
|---|---|---|---|
| A1 | ref22: exactitudes «entre 86,7 % y 99 %» | **PARCIAL** | La revisión **no enuncia un rango**: es una reconstrucción a partir de las cifras que cita. Lo que se lee: 99 % ([34], [37]), 91 % ([35]), **100 %** ([36], p. 455), 94 % (LSTM) y 87 % (ResNet) ([40]), 92 % ([44]), «training accuracy ranging between 90% and 96%» ([45]) y 86,74 % (GRU, [47], p. 456). El mínimo 86,7 % es correcto (y corresponde a una **GRU**, no a una LSTM), pero el máximo citado es **100 %**, no 99 %. `08_marco.tex:206` ya añade «con un trabajo que reclama 100 %»; `08_marco.tex:52` y `03_problema.tex:112` no lo hacen. |
| A2 | ref23: 88,23 % sobre lengua de señas india | **CONFIRMADA** | «The system recognizes the signs and gestures in real-time with 88.23% accuracy» (resumen, p. 1455; también p. 1464). Contexto: 45 señas y gestos de un conjunto propio de ISL, partición 95:5 (p. 1463), tres capas LSTM y tres densas. El título dice «continuous», pero clasifica señas una a una. |
| A3 | ref24: MediaPipe + LSTM; 94,38 % sobre 20 clases de ASL | **CONFIRMADA** | «testing accuracy of 94.38%»; 800 videos, 20 clases de ASL a nivel de palabra, **dos señantes** (resumen, p. 1). Modelo: Masking, dos LSTM, Flatten y Dense de 20 unidades (pp. 10–11). |
| A4 | ref25: lengua de señas noruega, 11 clases, 95 %, partición aleatoria | **CONFIRMADA** | Números del 0 al 10, es decir, 11 clases. «testing accuracy of 95%» (resumen). «The split was performed randomly, with a fixed random seed» (§3), 70/30. Cinco participantes y 20 fotogramas clave por video. Es un artículo de tipo *brief report*. |
| A5 | ref26: MediaPipe + redes convolucionales-BiLSTM; reconocimiento y traducción | **CONFIRMADA** (con matiz) | «we use the MediaPipe library and a hybrid Convolutional Neural Network + Bi-directional Long Short Term Memory (CNN + Bi-LSTM) model» (resumen, p. 104358). Abarca reconocimiento, traducción **y generación de video** (NMT + MediaPipe + Dynamic GAN). Matiz: la CNN es VGG-19 sobre imágenes, no una red sobre puntos característicos. «above 95% classification accuracy» (resumen). |
| A6 | ref41: MediaPipe + LSTM sobre LSC; 30 fotogramas por video; 90 % letras / 82 % palabras | **CONFIRMADA** | «Extracción de 30 fotogramas secuenciales por cada video» (§2.1, p. 4). «precisión del 90 % en el reconocimiento de letras y del 82 % en palabras» (resumen; Tabla 4, p. 7). Matices: la «precisión» es la métrica `accuracy` de Keras sobre un 20 % de **validación** aleatoria (p. 5); la exactitud de entrenamiento es 1,00 en los tres experimentos. |
| A7 | ref42: CNN+BiLSTM sobre imagen y BiLSTM sobre coordenadas de MediaPipe; 87,6 % (mano) | **CONFIRMADA** (con una incoherencia interna de la fuente) | Resumen (p. 18479) y Tabla 5 (p. 18489): exactitud del 87,60 % para el modelo de coordenadas de la mano (BiLSTM(128) → Dense 1024 → Dense 37, entrada de 6 × 63). **Pero** la Conclusión (§5) escribe «achieving an accuracy of 85.7%». Conviene citar el 87,6 % de la Tabla 5 y anotar la discrepancia. Dato relevante para la tesis: la evaluación separa 60 voluntarios de entrenamiento y 10 de prueba (p. 18486), es decir, es **independiente del señante**. Las secuencias tienen solo 6 pasos. |
| A8 | ¿Alguna de ref22–ref26, ref41 o ref42 usa atención? | Respuesta verificada | **Ninguno de los modelos de reconocimiento la usa.** ref23, ref24, ref25, ref41 y ref42 no tienen capa de atención. ref26 usa la atención de Bahdanau **solo en el traductor NMT** (texto → glosas, §III.B, pp. 104364–104366), no en el reconocedor. ref22 (revisión) **menciona** la atención: «Attention mechanisms can assign higher weights to important features in sign language video sequences» (p. 454). También lista trabajos con atención temporal (Xu et al. [18], Res-GCN + red de atención temporal; Shin et al. [19], GCN con atención espaciotemporal) y la propone como línea futura (§V). ref42 menciona la autoatención solo como trabajo relacionado (p. 18480). Por tanto, la atención temporal de la red **necesita citas propias** (§3). |
| A9 | «Los transformers requieren conjuntos de datos extensos» [ref6, ref10] | ref6: **INCORRECTA**. ref10: **PARCIAL / no la sostiene** | **ref6 (Vaswani)** dice lo contrario en su único experimento con pocos datos (§6.3): el Transformer supera a BerkeleyParser «even when training only on the WSJ training set of 40K sentences», mientras que las RNN seq2seq no alcanzan el estado del arte «in small-data regimes». **ref10 (Camgöz)** observa sobreajuste al pasar de tres capas en PHOENIX14T (p. 10026, §5.3) y que las anotaciones disponibles son «too few to build models which would work on a large domain» (p. 10022). Ninguna de las dos es una afirmación general sobre los transformers. Fuente que sí lo dice (en visión): Dosovitskiy et al., ICLR 2021, arXiv:2010.11929, Introducción: los transformers «do not generalize well when trained on insufficient amounts of data» por carecer de sesgos inductivos. Afecta a `08_marco.tex:58-60`. |

---

## 2. Errores en las entradas bibliográficas de `13_referencias.tex`

**No hay errores de fondo** en ref6, ref10, ref22, ref23, ref24, ref25, ref26, ref41 ni ref42: autores, títulos, revistas, volúmenes, páginas y DOI coinciden con Crossref y con las portadas. Observaciones menores:

1. **ref41, DOI: la ficha es correcta; no la «corrijan» desde el PDF.** El PDF imprime `10.19053/01211129.v34.n71.2025.18059`, que **devuelve 404** en doi.org. El de la ficha, `10.19053/uptc.01211129.v34.n71.2025.18059`, está registrado en Crossref y resuelve (302 a revistas.uptc.edu.co/…/18059). Crossref escribe «Rivera-Vasquez» sin tilde; la portada lleva tilde: se mantiene «Rivera-Vásquez».
2. **ref23:** se puede añadir el mes del fascículo («ago. 2024»; publicado en línea el 16 de julio de 2024). El número 3 es correcto.
3. **ref24:** se puede añadir «nov. 2025» (publicado el 18 de noviembre de 2025). El número de artículo 2587853 es correcto.
4. **ref22:** se puede añadir «sep. 2024» (el seminario fue el 21–22 de septiembre de 2024). Páginas 451–458 correctas.
5. **ref10:** Crossref separa el nombre como «Necati / Cihan Camgoz»; la forma «N. C. Camgöz» de la ficha es la habitual del autor. Sin cambio.
6. **ref42:** correcta (vol. 37, n.º 22, pp. 18479–18496, ago. 2025). La fuente se contradice (87,6 % frente a 85,7 %; ver A7).

Error de **atribución** (no de ficha): el marco cita ref6 y ref10 para «requerir conjuntos de datos extensos» (A9).

---

## 3. Referencias propuestas para la atención temporal

Lo que hace nuestra red (`src/train_final_100.py`, `build_model`): sobre la salida de la segunda BiLSTM (B × 30 × 128), `Dense(1)` lineal por paso → `Softmax(axis=1)` sobre los 30 pasos → `Multiply` → `GlobalAveragePooling1D`. El resultado se concatena con el *average pooling* y el *max pooling* de la misma salida. Precisión técnica: `Multiply` + `GlobalAveragePooling1D` calcula (1/T)·Σ αₜhₜ, es decir, la suma ponderada dividida por T = 30. Es una constante, así que equivale a la suma ponderada salvo escala, pero conviene describirlo así o hablar de «media ponderada». La puntuación es lineal (sin `tanh`), un caso particular de las formulaciones de abajo.

### [refAtn1] Sincan y Keles, 2020: lengua de señas aislada, (B)LSTM + atención temporal, independiente del señante

O. M. Sincan y H. Y. Keles, ``AUTSL: A large scale multi-modal Turkish sign language dataset and baseline methods'', \emph{IEEE Access}, vol. 8, pp. 181340--181355, 2020. doi: 10.1109/ACCESS.2020.3028072.

- **Verificación:** Crossref (título, autores, vol. 8, pp. 181340–181355, 2020). Texto completo leído en arXiv:2008.00932 (versión HTML). Las citas de sección se refieren a esa versión; la versión de IEEE Access puede diferir en la paginación.
- **Qué hace su atención (§4.1, ecs. 2–4):** eᵢ = vᵀ tanh(W hᵢ + b), αᵢ = softmax(eᵢ) sobre todos los fotogramas, y c = Σ αᵢ hᵢ sobre los estados ocultos de la LSTM o BLSTM. El vector c sustituye al último estado oculto antes de la capa FC y el softmax. **Coincide** con la nuestra: puntaje por paso → softmax temporal → suma ponderada. Diferencias: la puntuación es una MLP con tanh, no lineal; no concatena average ni max pooling; la entrada son rasgos CNN de RGB o RGB-D, **no puntos característicos**.
- **Por qué sirve:** reconocimiento de lengua de señas **aislada**, 226 señas y 43 señantes, con protocolo **independiente del señante** (36 señantes de entrenamiento y validación, 7 de prueba). Añadir la atención temporal subió el top-1 en 18,49 puntos con RGB-D y en más de 19 con RGB (§5.2.2, Tablas 3–4). La BLSTM con atención fue el mejor modelo (62,02 %).
- **Respalda en el párrafo:** (a) usar atención temporal sobre la salida de una (Bi)LSTM para clasificar señas aisladas; (b) que la bidireccionalidad sea admisible en señas aisladas: los autores advierten que en tiempo real la pasada hacia atrás obliga a almacenar los fotogramas y solo puede evaluarse cuando la seña aislada ha terminado (§5.2.2).

### [refAtn2] Zhou et al., 2016: BiLSTM + atención (puntaje → softmax → suma ponderada) para clasificar

P. Zhou, W. Shi, J. Tian, Z. Qi, B. Li, H. Hao y B. Xu, ``Attention-based bidirectional long short-term memory networks for relation classification'', en \emph{Proc. 54th Annu. Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)}, Berlín, Alemania, ago. 2016, pp. 207--212. doi: 10.18653/v1/P16-2034.

- **Verificación:** Crossref y ACL Anthology (actas, Berlín, agosto de 2016, pp. 207–212). Texto completo leído.
- **Qué hace su atención (§3.3, p. 209, ecs. 9–11):** M = tanh(H), α = softmax(wᵀM) y r = Hαᵀ, donde H = [h₁ … h_T] son las salidas de la BiLSTM. Es exactamente «un puntaje escalar por paso → softmax sobre los pasos → suma ponderada» sobre una BiLSTM, para **clasificar** una secuencia. Es la formulación más cercana a la nuestra; nuestra `Dense(1)` lineal equivale a wᵀh + b sin el tanh. El dominio es texto (clasificación de relaciones), no señas.
- **Respalda:** la forma concreta de la capa de atención (BiLSTM → puntaje por paso → softmax → suma ponderada → clasificador). Es una revisión por pares de congreso de primer nivel en PLN.

### [refAtn3] Song et al., 2017: atención temporal sobre LSTM en reconocimiento de acciones con esqueleto

S. Song, C. Lan, J. Xing, W. Zeng y J. Liu, ``An end-to-end spatio-temporal attention model for human action recognition from skeleton data'', en \emph{Proc. AAAI Conf. on Artificial Intelligence}, vol. 31, n.º 1, feb. 2017. doi: 10.1609/aaai.v31i1.11212.

- **Verificación:** Crossref y la página de AAAI (vol. 31, n.º 1, 12 de febrero de 2017). Texto completo leído en arXiv:1611.06067v1. **No se pudieron confirmar las páginas** del volumen: la ficha va sin páginas.
- **Qué hace su atención (§4.2, ecs. 4 y 6):** una subred LSTM produce un peso βₜ = ReLU(·) por fotograma, y las puntuaciones de clase son o = Σₜ βₜ·zₜ, la suma ponderada de las salidas por fotograma de la LSTM principal. Su motivación es la nuestra: no todos los fotogramas aportan lo mismo, y solo algunos fotogramas clave contienen la información más discriminativa (§4.2). **Coincide en parte:** pondera pasos temporales sobre una LSTM, pero los pesos no se normalizan con softmax (ReLU), se aplican sobre las puntuaciones y no sobre los estados ocultos, y la calcula una LSTM auxiliar, no una capa densa. La entrada sí son **articulaciones de esqueleto** (SBU y NTU RGB+D).
- **Respalda:** que la atención «pondera los instantes informativos» de una secuencia de puntos del cuerpo. Es el precedente con esqueleto.

### [refAtn4] Raffel y Ellis, 2016: la formulación mínima de «attention pooling»

C. Raffel y D. P. W. Ellis, ``Feed-forward networks with attention can solve some long-term memory problems'', presentado en \emph{Int. Conf. on Learning Representations (ICLR), Workshop Track}, San Juan, Puerto Rico, 2016. arXiv:1512.08756.

- **Verificación:** arXiv (v5, 20 de septiembre de 2016). La cabecera del PDF dice «Workshop track - ICLR 2016». **No tiene DOI.** La ciudad (San Juan) **no se verificó** en esta tarea: si no se confirma, quítese de la ficha.
- **Qué hace (§1.2, ec. 1, p. 2):** eₜ = a(hₜ), αₜ = exp(eₜ)/Σₖ exp(eₖ) y c = Σₜ αₜhₜ, donde a es una función aprendible que depende solo de hₜ. Los autores lo describen como un *embedding* de longitud fija obtenido por una media ponderada adaptativa de la secuencia de estados. **Coincide exactamente** con nuestra capa: `Dense(1)` es una a(hₜ) lineal. En sus experimentos usan a(hₜ) = tanh(W hₜ + b) (§2). Además comparan con la media sin ponderar, que es lo que hace nuestro *average pooling* concatenado.
- **Matiz:** es un *workshop track*, con revisión más ligera que el congreso principal. Úsese como cita de la **definición** del mecanismo, junto a [refAtn2], no como única fuente.

### Opcionales (verificadas, de menor ajuste)

- **Bahdanau et al. (origen de la atención aditiva):** D. Bahdanau, K. Cho y Y. Bengio, ``Neural machine translation by jointly learning to align and translate'', presentado en \emph{Int. Conf. on Learning Representations (ICLR)}, San Diego, CA, EE. UU., 2015. arXiv:1409.0473. arXiv confirma «Accepted at ICLR 2015 as oral presentation»; la ciudad no se verificó. **No es nuestra atención:** el puntaje depende del estado del decodificador en cada paso de salida (traducción), no hay una única suma ponderada para clasificar. Sirve solo como antecedente histórico; Raffel y Ellis la simplifican a lo que hacemos.
- **Huang et al. (SLR aislada con atención temporal sobre BiLSTM):** J. Huang, W. Zhou, H. Li y W. Li, ``Attention-based 3D-CNNs for large-vocabulary sign language recognition'', \emph{IEEE Trans. Circuits and Systems for Video Technology}, vol. 29, n.º 9, pp. 2822--2832, sep. 2019. doi: 10.1109/TCSVT.2018.2870740. Existencia y metadatos verificados en Crossref. **Contenido verificado solo por el resumen y por la descripción de Sincan y Keles**: atención temporal para seleccionar los movimientos significativos sobre una BiLSTM, con entrada de video 3D-CNN. No se leyó el texto completo (muro de pago). No se recomienda citarla sin leerla.
- **Para «los transformers requieren datos extensos»:** A. Dosovitskiy et al., ``An image is worth 16x16 words: Transformers for image recognition at scale'', presentado en \emph{Int. Conf. on Learning Representations (ICLR)}, 2021. arXiv:2010.11929. La frase está en la Introducción (verificada). Advertencia: el dominio es imagen, no secuencias de puntos.

### Lo que no se encontró

No se halló, con verificación del texto, una publicación revisada por pares que combine **puntos característicos de MediaPipe o esqueleto + (Bi)LSTM + atención temporal con softmax** para lengua de señas **aislada**. Aparecieron preprints de arXiv sin revisión (p. ej., «SignVLA», 2026) y artículos de revistas de menor perfil cuyo texto no se pudo revisar; no se incluyen. Por eso se combinan [refAtn1] (señas aisladas, RGB) y [refAtn3] (esqueleto, acciones).

---

## 4. Párrafo corregido

> La arquitectura adoptada combina una red convolucional unidimensional, dos capas LSTM bidireccionales, un mecanismo de atención temporal y un clasificador denso. Cada componente responde a una propiedad del problema. La convolución, con núcleos de tres pasos, extrae patrones locales de movimiento de corta duración. La LSTM modela el orden de la secuencia, que es lo que separa señas con configuraciones manuales casi idénticas: en la LSC, los números del seis al nueve solo se distinguen de los del uno al cuatro por el movimiento de los dedos [ref42]. La bidireccionalidad es admisible porque el sistema clasifica señas aisladas una vez cerrada la ventana de captura, de modo que la pasada hacia atrás no añade espera [refAtn1]. La atención asigna a cada paso temporal un peso aprendido, normalizado con softmax, y resume la secuencia como su media ponderada [refAtn2, refAtn4], para que los pasos más informativos pesen más que, por ejemplo, el reposo con que terminan las capturas del corpus. Este recurso mejoró de forma sustancial el reconocimiento de señas aisladas con señantes no vistos [refAtn1] y el de acciones a partir de esqueleto [refAtn3]. En la literatura reciente, la combinación de puntos característicos de MediaPipe con redes recurrentes, en particular LSTM, es frecuente: en una revisión de alcance de estudios que integran MediaPipe con aprendizaje profundo, la LSTM fue la técnica más empleada [ref22]. Se ha aplicado a las lenguas de señas india, americana y noruega [ref23–ref25] y a la LSC [ref41, ref42]. Empíricamente, bajo validación independiente del señante, la red supera en 16 puntos de exactitud top-1 a la mejor línea base clásica, que recibe la secuencia resumida en estadísticos (prueba de McNemar, p = 2,9 × 10⁻¹⁸). Con la misma aumentación de datos para ambos sistemas, la ventaja es de 13,8 puntos (p = 3,0 × 10⁻¹⁵). Esta comparación respalda el modelado del orden temporal frente a un resumen de la secuencia, no cada componente por separado. La comparación con otras arquitecturas secuenciales (GRU, TCN, transformers) queda fuera del alcance y se propone como trabajo futuro.

Cambios respecto al borrador:

1. «el más extendido en la literatura» → «es frecuente … la LSTM fue la técnica más empleada [ref22]». Es lo que ref22 sostiene, dentro de su universo de estudios que ya usan MediaPipe.
2. Se retira ref26 de la lista de «MediaPipe → RNN»: su reconocedor es VGG-19 sobre imágenes + Bi-LSTM. Si se quiere conservar, debe ir aparte («o combinado con redes convolucionales sobre la imagen [ref26]»).
3. «reposo inicial y final, común a todas las señas» → «el reposo con que terminan las capturas». Los registros del proyecto (N-50) muestran que el inicio está en movimiento en la mayoría de los clips.
4. La atención se presenta como **propósito** («para que … pesen más»), no como efecto demostrado: no hay ablación ni análisis de pesos de atención. Si se hace la ablación (sin atención, solo average y max pooling), podrá afirmarse su efecto en este corpus.
5. Se añaden el matiz de la línea base (secuencia resumida) y el control de equidad de la aumentación (`comparacion.md` §9).
6. Opcional, si se desea precisión: «media ponderada» describe exactamente `Multiply` + `GlobalAveragePooling1D` (Σαₜhₜ/T); «suma ponderada» también es aceptable salvo una constante.

---

## 5. Lo que no se pudo verificar y por qué

| Elemento | Motivo |
|---|---|
| Páginas de Song et al. en las actas de AAAI-17 | Ni Crossref ni la página de AAAI muestran páginas; no se citan. |
| Ciudad de ICLR 2016 (Raffel y Ellis) y de ICLR 2015 (Bahdanau) | No aparece en arXiv ni en el PDF; no se comprobó en otra fuente. Quitar la ciudad si no se confirma. |
| Paginación de Sincan y Keles en IEEE Access | Se leyó la versión de arXiv (HTML). Las citas van por sección (§4.1, §5.2.2), no por página. Metadatos de IEEE Access confirmados en Crossref. |
| Contenido completo de Huang et al. (TCSVT 2019) | Muro de pago; solo se vio el resumen. Queda como opcional no recomendada. |
| Una referencia revisada por pares de «puntos/esqueleto + BiLSTM + atención con softmax» para SLR aislada | No se encontró una con texto verificable; ver el final de la §3. |
| Efecto de la atención en **nuestra** red | No existe ablación en el repositorio; el párrafo corregido no lo afirma. |
| Que «la convolución captura patrones de corta duración» tenga respaldo bibliográfico | Ninguna de las fuentes revisadas lo dice; queda como justificación de diseño sin cita. |
| ref6 (Vaswani), ficha | Se verificó el contenido de §6.3 en arXiv; la ficha de NeurIPS (sin páginas) no se cotejó con el registro del editor en esta tarea. |
