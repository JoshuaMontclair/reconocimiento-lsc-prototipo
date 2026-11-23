# Revisión de literatura: arquitecturas para patrones espacio-temporales

Fecha: 2 de octubre de 2026. Encargo: reunión con el director del 2-oct-2026. Hay que buscar, de preferencia en
artículos de revisión, cómo se aborda en general el reconocimiento de patrones espacio-temporales y decidir
qué otras arquitecturas probar. El director pregunta en particular si sería razonable «una estrategia más
sencilla» o «una red más robusta que el MLP».

Este documento solo informa. No cambia el modelo canónico, el umbral ni el alcance. No se entrenó nada y no se
modificó ningún otro archivo.

---

## 0. Resumen en cinco puntos

1. **Las revisiones agrupan las soluciones en cuatro familias:** recurrentes (LSTM y GRU), convolucionales
   temporales (CNN 1D, FCN, ResNet, TCN), de grafos (ST-GCN y variantes) y Transformers. En series de tiempo
   se añade una familia no profunda muy fuerte y barata, la de ROCKET y MiniRocket [1], [6]–[9].
2. **Sobre puntos del cuerpo y de las manos, las redes de grafos dan las mejores cifras publicadas.** En
   OpenHands, la exactitud media en 7 conjuntos fue 67,7 % con LSTM, 73,5 % con Transformer, 77,3 % con ST-GCN y
   81,1 % con SL-GCN [3]. Pero esos conjuntos son mucho más grandes que LSC50, la LSTM fue unas diez veces más
   rápida en CPU que SL-GCN [3] y ninguna de esas cifras usa un protocolo como nuestro LOSO con 5 señantes.
3. **El MLP queda por debajo de las CNN 1D con GAP en series de tiempo.** Ismail Fawaz et al. explican que con
   un MLP «la información temporal se pierde» y que las mejores son las CNN 1D con *global average pooling*
   (ResNet y FCN); el MLP queda a media tabla (4.º de 9 en el archivo univariado), no entre los peores [6]. Así, una «red más robusta que el MLP» y a la vez más sencilla que nuestra BiLSTM con atención
   existe y tiene respaldo: una **CNN 1D totalmente convolucional, tipo FCN o TCN** [6], [12], o su variante
   ligera para articulaciones de la mano, **DD-Net** [5].
4. **La «estrategia más sencilla» razonable es MiniRocket con un clasificador lineal.** Usa un conjunto
   pequeño y fijo de núcleos convolucionales (casi determinista, a diferencia de los núcleos aleatorios de
   ROCKET) y una regresión *ridge*. Iguala en exactitud a métodos mucho más caros
   [8], [9] y conserva el orden temporal, que es justo lo que pierde nuestra línea base de estadísticos.
5. **Recomendación:** probar primero, bajo LOSO, (1) **MiniRocket + *ridge*** y (2) **una CNN 1D/TCN pequeña con
   GAP**. Como tercera, opcional, (3) **un ST-GCN pequeño sobre el grafo de las dos manos**. GRU y Transformer
   quedan en segundo plano. Las candidatas (2) y (3) amplían el alcance del anteproyecto
   (`08_marco.tex:57-61`), así que la decisión es del director. La (1) puede presentarse como una línea base
   adicional, dentro del compromiso de `09_metodologia.tex:131-134`.

---

## 1. Método

- **Búsqueda.** Búsqueda web y bases abiertas (arXiv, CVF Open Access, ACL Anthology, PMC) sobre:
  (a) reconocimiento de acciones con esqueleto, (b) reconocimiento de señas aisladas con pose o puntos,
  (c) clasificación de series de tiempo multivariadas y (d) datos pequeños con evaluación independiente del
  sujeto. Se priorizaron las revisiones revisadas por pares publicadas desde 2019. Se añadieron estudios
  primarios cuando aportaban un dato que ninguna revisión daba: costo en CPU, modelos ligeros o el protocolo
  por sujeto.
- **Verificación de metadatos.** Autores, título, revista o actas, volumen, número, páginas y año de cada DOI
  se cotejaron con la API de Crossref el 2-oct-2026. Las discrepancias se anotan en la ficha.
- **Lectura.** Se descargó y leyó el texto completo de 11 fuentes (PDF de arXiv, CVF, ACL Anthology o PMC). Las
  citas textuales son de esas versiones. Cuando la versión leída no es la del editor, se indica, y en ese caso
  se cita por sección y no por página.
- **Estados.** **CONFIRMADA**: se leyó el texto y la afirmación está en él. **NO VERIFICABLE**: solo se vio el
  resumen.
- No se inventaron cifras. Las estimaciones propias (parámetros, costo) se marcan como tales.
- **Revisión independiente (2-oct-2026).** Un revisor volvió a resolver los 12 DOI en Crossref (todos
  coinciden en autores, título, año, revista o actas, volumen y páginas), descargó de nuevo los textos y buscó
  cada cita textual y cada cifra en ellos. Todas las citas textuales aparecen en la fuente. Se corrigieron
  errores de ficha y de atribución (marcados como «revisión independiente» donde cambian el estado), y
  Gholamiangonabadi et al. [11] pasó a CONFIRMADA al leerse su texto completo. Quedan 12 fuentes confirmadas.

---

## 2. Fuentes

### a) Reconocimiento de acciones con esqueleto

#### [1] Ren et al., 2024: revisión de reconocimiento de acciones con esqueleto 3D

B. Ren, M. Liu, R. Ding y H. Liu, "A survey on 3D skeleton-based action recognition using learning method,"
*Cyborg and Bionic Systems*, vol. 5, art. 0100, 2024. doi: 10.34133/cbsystems.0100.

- **Estado:** CONFIRMADA. Se leyó el texto completo en arXiv:2002.05907v2 (27-ene-2024). Las páginas citadas
  son las de esa versión.
- **Qué revisa:** las cuatro familias profundas sobre esqueleto: RNN, CNN, GCN y Transformers (§2.2–2.5).
- **Qué sostiene:**
  - Las RNN capturan bien la dinámica temporal, pero «are computationally inefficient due to sequential
    processing» (§2.1, p. 4).
  - Los Transformers son «computationally intensive due to attention mechanisms» (§2.1, p. 4).
  - Las GCN «can be sensitive to noisy or irregular connections among joints» (§2.1, p. 4).
  - En la discusión (§4, pp. 10–11), las GCN tienen el mejor potencial de exactitud, y «CNNs boast faster
    processing capabilities owing to their parallel processing nature, contrasting with RNNs' slower
    sequential processing».
  - Además: «Considering factors like dataset size and hardware resources, the choice becomes more
    adaptable». Es decir, la elección depende del tamaño de los datos y del hardware.
  - Cita DD-Net [5] como ejemplo de equilibrio entre tamaño del modelo y eficiencia (§2.3, p. 7).
- **Relevancia:** da el mapa general que pidió el director, con los pros y contras de cada familia. Respalda
  que, con datos pequeños y CPU, la elección no tiene por qué ser la arquitectura de mejor cifra en NTU RGB+D.

#### [5] Yang et al., 2019: DD-Net, una CNN 1D ligera para articulaciones de la mano

F. Yang, Y. Wu, S. Sakti y S. Nakamura, "Make skeleton-based action recognition model smaller, faster and
better," en *Proc. ACM Multimedia Asia (MMAsia '19)*, 2019, pp. 1–6. doi: 10.1145/3338533.3366569.

- **Estado:** CONFIRMADA. Se leyó arXiv:1907.09658v8. **Discrepancia:** Crossref ordena los autores como
  Yang, Wu, Sakti, Nakamura; el PDF de arXiv, como Yang, Sakti, Wu, Nakamura. *Revisión independiente:*
  OpenAlex y Semantic Scholar repiten el orden de Crossref (metadatos depositados por ACM), así que se cita en
  ese orden; la página de ACM no se pudo abrir (bloqueo automático).
- **Qué revisa o propone:** una red de **CNN 1D** sobre dos rasgos.
  - JCD (*Joint Collection Distances*): las distancias entre todos los pares de articulaciones en cada cuadro,
    que no cambian con la posición ni con el punto de vista.
  - Movimiento global a dos escalas (§III.A–B).
- **Qué sostiene:**
  - Justifica no usar RNN así: «we argue that it is relatively slow and difficult for parallel computing,
    compared with methods … that use Convolutional Neural Networks (CNNs)» (§II).
  - En SHREC (gestos de la mano, 22 articulaciones, 2800 muestras), con 0,15 M parámetros obtiene 91,8 % y
    90,0 % en 14 y 28 gestos, y con 1,82 M, 94,6 % y 91,9 % (Tabla II).
  - Supera a CNN+LSTM (8–9 M parámetros; 89,8 % y 86,3 %) y a STA-Res-TCN (5–6 M; 93,6 % y 90,7 %).
  - Velocidad: «2,000 FPS on one CPU (i.e., Intel E5-2620)» (§IV).
- **Relevancia:** es lo más cercano a nuestro problema: puntos de la mano, pocas muestras y prioridad en la
  velocidad. Muestra una **red más robusta que un MLP y más sencilla que una BiLSTM**. Las distancias JCD
  entre los 21 puntos de cada mano son, además, un rasgo barato de probar en cualquier arquitectura.
  Advertencia: la evaluación de SHREC usa su partición fija, no LOSO.

*Referencia complementaria, verificada solo en Crossref, no contada:* S. Yan, Y. Xiong y D. Lin, "Spatial
temporal graph convolutional networks for skeleton-based action recognition," *Proc. AAAI Conf. Artif.
Intell.*, vol. 32, n.º 1, 2018. doi: 10.1609/aaai.v32i1.12328. Es el artículo original de ST-GCN.

### b) Reconocimiento de señas aisladas con pose o puntos

#### [2] Sarhan y Frintrop, 2023: revisión de una década de reconocimiento de señas aisladas

N. Sarhan y S. Frintrop, "Unraveling a decade: A comprehensive survey on isolated sign language
recognition," en *Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW)*, 2023, pp. 3202–3211. doi:
10.1109/ICCVW60793.2023.00345.

- **Estado:** CONFIRMADA. Se leyó el PDF de CVF Open Access. Las páginas son las de Crossref; la versión de CVF
  numera distinto, así que se cita por sección.
- **Qué revisa:** el reconocimiento de señas aisladas por modalidad de entrada (RGB, profundidad, esqueleto),
  por parámetros lingüísticos modelados, por fusión y por aprendizaje por transferencia.
- **Qué sostiene:**
  - Los modelos sobre pose «have one or two orders of magnitude fewer parameters than those that process the
    video directly. However, skeleton data does not provide information about the hand shapes or the 3D
    structure of the hand and body» (§2.1).
  - Sobre el tamaño de los datos: «Datasets for SLR have always been small … they are still not large enough
    to train Deep CNNs from scratch. Therefore cross-domain transfer learning becomes inevitable» (§2.4).
  - Sobre la evaluación por señante: «a powerful model would pick up particularities about individual persons,
    and recognition scores would be overly optimistic» si el señante se repite entre entrenamiento y prueba
    (§3.1).
  - Conclusión (§4): «there is a noticeable trend towards skeleton-based methods, following the progress in
    human action recognition with spatial-temporal Graph Convolutional Networks (GCN)». Añade que el campo
    «remains in its early stages». Propone el aprendizaje autosupervisado y el ajuste fino multilingüe para
    escenarios de pocos datos.
- **Relevancia:** respalda dos cosas. La tendencia hacia los GCN sobre pose. Y que el LOSO del proyecto es el
  protocolo correcto, cuyas cifras son menores que las de una partición aleatoria.

#### [3] Selvaraj et al., 2022: OpenHands, cuatro modelos sobre pose de MediaPipe en siete lenguas de señas

P. Selvaraj, G. NC, P. Kumar y M. Khapra, "OpenHands: Making sign language recognition accessible with
pose-based pretrained models across languages," en *Proc. 60th Annu. Meeting Assoc. Comput. Linguistics
(Vol. 1: Long Papers)*, 2022, pp. 2114–2133. doi: 10.18653/v1/2022.acl-long.150.

- **Estado:** CONFIRMADA. Se leyó el PDF de ACL Anthology. **Ojo:** la versión de arXiv (2110.05877) tiene
  cifras distintas en la Tabla 2: los promedios 69,38 / 73,47 / 77,43 / 80,69 y el CSL con 100 clases. Hay que
  citar la versión de ACL, que es la que se usa aquí.
- **Qué revisa:** cuatro clasificadores sobre 27 puntos 2D de MediaPipe Holistic (parte superior del cuerpo,
  manos y cara), entrenados en 7 conjuntos de 6 lenguas de señas.
  - LSTM: 4 capas bidireccionales de 128 con atención temporal.
  - Transformer tipo BERT.
  - ST-GCN y SL-GCN.
- **Qué sostiene:**
  - Exactitud media: LSTM 67,7, BERT 73,5, ST-GCN 77,3 y SL-GCN 81,1 (Tabla 2, p. 2119). Según los autores,
    estos resultados «confirm that graph-based models on pose modality data define the SOTA».
  - Latencia en CPU (Apéndice, Tabla 7), con la LSTM de 1,6 M parámetros y el ST-GCN de 2,3 M:
    - LSTM: entre 5,4 y 12,9 ms por video;
    - Transformer (3,8 M): entre 23,8 y 76,4 ms;
    - ST-GCN: entre 23,0 y 99,4 ms;
    - SL-GCN (4,9 M): entre 52,8 y 225,3 ms.
  - Los autores concluyen que la LSTM «is an order of magnitude faster across all devices than the most
    accurate SL-GCN model, and is a good candidate when speed is essential at the cost of about 10% accuracy
    drop». También: «ST-GCN provides a good trade-off being about 2× faster than SL-GCN at the cost of only 3%
    lower average accuracy».
- **Relevancia:** es la comparación más directa entre familias sobre puntos de MediaPipe, con costo en CPU de
  portátil (Ryzen 7 3750H). Sostiene que un ST-GCN pequeño es el candidato con más evidencia de mejorar la
  exactitud y que la recurrente es la más barata.
- **Límites:** los conjuntos tienen entre 3200 (LSA64) y 125 000 (CSL) videos (Tabla 1) y el protocolo es el de cada conjunto, no LOSO.
  Las cifras no se trasladan a LSC50. El modelo preentrenado usa los 27 puntos de cuerpo, manos y cara, no
  nuestros 42 puntos de las manos.

#### [4] Boháček y Hrúz, 2022: SPOTER, un Transformer sobre pose para señas aisladas

M. Boháček y M. Hrúz, "Sign pose-based transformer for word-level sign language recognition," en *Proc.
IEEE/CVF Winter Conf. Appl. Comput. Vis. Workshops (WACVW)*, 2022, pp. 182–191. doi:
10.1109/WACVW54805.2022.00024.

- **Estado:** CONFIRMADA. Se leyó el PDF de CVF Open Access.
- **Qué propone:** un Transformer de 6 capas de codificador y 6 de decodificador, con 9 cabezas, sobre 54
  puntos 2D: 108 valores por cuadro (Tabla 3). Tiene 5,92 M de parámetros (§5.4).
- **Qué sostiene:**
  - 63,18 % en WLASL100 y 43,78 % en WLASL300 (Tabla 4). Mejora a Pose-TGCN (55,43 %) y a Pose-GRU (46,51 %).
  - En la ablación (Tabla 6), **la normalización** (el espacio de signado y cada mano en su propio marco local)
    sube la exactitud de 44,96 % a 58,97 %, más de 14 puntos. Las aumentaciones suman unos 4 puntos más.
  - En LSA64 informa 100 %, pero con «a single random split of 80% of the data for training and the rest 20%»
    (Tabla 5). Es una partición aleatoria, **no independiente del señante**.
- **Relevancia:**
  - Muestra que un Transformer sobre pose funciona con unos 2000 videos (WLASL100).
  - Muestra también que **el preprocesamiento puede pesar tanto como la arquitectura**: normalizar cada mano en
    su propio marco es una «estrategia sencilla» que no exige cambiar de red.
  - Es además un ejemplo de por qué las cifras con partición aleatoria (100 %) no se comparan con LOSO.

### c) Clasificación de series de tiempo multivariadas

#### [6] Ismail Fawaz et al., 2019: revisión del aprendizaje profundo en series de tiempo

H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar y P.-A. Muller, "Deep learning for time series
classification: a review," *Data Mining and Knowledge Discovery*, vol. 33, n.º 4, pp. 917–963, 2019. doi:
10.1007/s10618-019-00619-1.

- **Estado:** CONFIRMADA. Se leyó la versión de autor en arXiv:1809.04356, así que se cita por sección.
- **Qué revisa:** 9 redes de extremo a extremo (MLP, FCN, ResNet, Encoder, MCNN, t-LeNet, MCDCNN, Time-CNN y
  TWIESN) en 85 conjuntos univariados y 12 multivariados (8730 modelos entrenados).
- **Qué sostiene:**
  - Contra el MLP: «The problem with an MLP approach is that temporal information is lost and the features
    learned are no longer time-invariant. This is where CNNs are most useful» (§2.3.2). La CNN es la arquitectura
    más usada en series de tiempo, por su robustez y su menor tiempo de entrenamiento frente a las RNN y los
    MLP.
  - ResNet es la mejor y FCN la segunda (§5.1–5.2). El MLP, Time-CNN y MCDCNN rinden de forma parecida, y
    peor que ellas; los autores lo atribuyen, «probably», a que usan una capa densa en lugar del *global
    average pooling* (GAP) (§5.1). En el diagrama de rangos univariado (Fig. 7) el MLP queda 4.º de 9, por
    delante de Time-CNN, MCDCNN, MCNN y t-LeNet.
  - En los 12 conjuntos multivariados, ResNet, FCN y Encoder superan a las redes diseñadas para ese caso, pero
    la prueba no halla diferencias significativas por el bajo número de conjuntos (§5.3).
  - Datos pequeños: veinte conjuntos tienen 50 o menos ejemplos de entrenamiento, lo que «surprisingly was not
    an impediment for obtaining high accuracy when applying a very deep architecture such as ResNet» (§4.1.1).
- **Relevancia:** es la respuesta directa a lo de «una red más robusta que el MLP». Señala la **FCN o ResNet 1D
  con GAP** como la opción de referencia, sencilla y con evidencia.

#### [7] Mohammadi Foumani et al., 2024: revisión actualizada del aprendizaje profundo en series de tiempo

N. Mohammadi Foumani, L. Miller, C. W. Tan, G. I. Webb, G. Forestier y M. Salehi, "Deep learning for time
series classification and extrinsic regression: A current survey," *ACM Computing Surveys*, vol. 56, n.º 9,
pp. 1–45, 2024. doi: 10.1145/3649448.

- **Estado:** CONFIRMADA. Se leyó arXiv:2302.02515v2, así que se cita por sección.
- **Qué revisa:** MLP, CNN (incluidas ResNet, InceptionTime y LITE), RNN, modelos de atención y Transformers,
  aprendizaje autosupervisado, aumentación de datos y transferencia. Tiene una sección de aplicación al
  reconocimiento de actividad humana (HAR).
- **Qué sostiene:**
  - En la introducción (§1), InceptionTime «is competitive but did not outperform the state of the art on
    benchmarking archives». Explica que esos archivos «are relatively small, while deep learning often excels
    when data quantities are large».
  - ConvTran supera a ROCKET en velocidad y exactitud «when there are more than 10k training samples» (§1).
  - En HAR se usan CNN, RNN e híbridos CNN-RNN. Cita un estudio según el cual la BiLSTM fue mejor cuando
    importan los efectos de largo plazo y las CNN cuando importan los patrones cortos de movimiento (§7.1).
  - La aumentación es una herramienta clave contra el sobreajuste en series de tiempo (§5).
- **Relevancia:** cada pliegue LOSO tiene unos 800 videos de entrenamiento (unos 8000 con la aumentación
  ×10), por debajo de los 10 000 del ejemplo de ConvTran. La revisión no fija un umbral general, pero sí dice
  que en archivos pequeños el aprendizaje profundo no supera al estado del arte no profundo. Eso justifica
  probar MiniRocket como contraste.

#### [8] Pasos Ruiz et al., 2021: comparación a gran escala en series de tiempo multivariadas

A. Pasos Ruiz, M. Flynn, J. Large, M. Middlehurst y A. Bagnall, "The great multivariate time series
classification bake off: a review and experimental evaluation of recent algorithmic advances," *Data Mining
and Knowledge Discovery*, vol. 35, n.º 2, pp. 401–449, 2021. doi: 10.1007/s10618-020-00727-3.

- **Estado:** CONFIRMADA. Se leyeron el resumen y los resultados (§5.1) en PMC (PMC7897627). *Revisión
  independiente:* las conclusiones sí están en el texto de PMC y se leyeron (ver abajo).
- **Qué revisa:** tres algoritmos multivariados de aprendizaje profundo (ResNet, InceptionTime y TapNet), de
  *shapelets* y de diccionario, además de ROCKET, en 26 problemas del archivo UEA. (MLSTM-FCN no está entre
  los evaluados; se quitó de esta lista en la revisión independiente.)
- **Qué sostiene:**
  - El resumen dice que cuatro clasificadores son significativamente mejores que DTW y que ROCKET lo logra «in
    at least an order of magnitude less time than the other three».
  - ROCKET «is by far the fastest algorithm and remarkably can build a model for all 26 data sets in just over
    an hour».
  - ResNet frente a DTW: «better on 14 and worse on 12».
  - Conclusiones: «ROCKET is the best ranked and by far the fastest classifier and would be our
    recommendation as the default choice for MTSC problems». Sobre redes: InceptionTime «should be the
    starting point for future work with neural networks». Y una advertencia: «DTW, is still hard to beat».
- **Relevancia:** en series multivariadas, el método más barato (ROCKET) está entre los mejores y las redes
  profundas no ganan de forma consistente. Es la evidencia más directa a favor de una «estrategia más
  sencilla».

#### [9] Dempster et al., 2021: MiniRocket

A. Dempster, D. F. Schmidt y G. I. Webb, "MiniRocket: A very fast (almost) deterministic transform for time
series classification," en *Proc. 27th ACM SIGKDD Conf. Knowl. Discovery Data Mining (KDD '21)*, 2021,
pp. 248–257. doi: 10.1145/3447548.3467231.

- **Estado:** CONFIRMADA. Se leyó arXiv:2012.08791.
- **Qué propone:** convolucionar cada serie con un conjunto pequeño y fijo de núcleos (84 núcleos de longitud
  9 con varias dilataciones), resumir cada salida con la proporción de valores positivos (unos 10 000 rasgos) y
  entrenar un clasificador **lineal**. Los autores recomiendan una regresión *ridge* con validación cruzada si
  hay menos de unos 10 000 ejemplos de entrenamiento y regresión logística si hay más (§3.3).
- **Qué sostiene:**
  - Es hasta 75 veces más rápido que ROCKET en los conjuntos grandes, «while maintaining essentially the same
    accuracy» (resumen).
  - «MiniRocket is significantly faster than any other method of comparable accuracy (including Rocket), and
    significantly more accurate than any other method of even roughly-similar computational expense»
    (resumen).
- **Relevancia:** es una base fuerte, barata y casi determinista, y conserva el orden temporal. El artículo
  evalúa series univariadas (UCR); los autores solo ofrecen «a naïve facility» para multivariadas y dejan su
  estudio como trabajo futuro (conclusión). Para entradas multivariadas como la nuestra (252 canales × 30 pasos) existe
  la implementación `MiniRocketMultivariate` de sktime: se comprobó en su documentación (v1.2.0) que existe,
  con `num_kernels=10000` por defecto. Con series de 30 pasos, el propio artículo advierte que hay poco
  beneficio en calcular mucho más de *l* sesgos por serie de longitud *l* (§4.3.4), así que parte de los
  10 000 rasgos será redundante. sktime **no** está en `requirements.txt`, así que habría que añadirla como dependencia.

#### [10] Ismail Fawaz et al., 2020: InceptionTime

H. Ismail Fawaz, B. Lucas, G. Forestier, C. Pelletier, D. F. Schmidt, J. Weber, G. I. Webb, L. Idoumghar,
P.-A. Muller y F. Petitjean, "InceptionTime: Finding AlexNet for time series classification," *Data Mining
and Knowledge Discovery*, vol. 34, n.º 6, pp. 1936–1962, 2020. doi: 10.1007/s10618-020-00710-y.

- **Estado:** CONFIRMADA. Se leyó la versión de autor en arXiv:1909.04939.
- **Qué propone:** un conjunto (*ensemble*) de 5 redes CNN 1D tipo Inception con conexiones residuales.
- **Qué sostiene:**
  - Usan un conjunto de redes por el poco volumen de datos: «given the small training size of the UCR archive
    datasets which are not well suited to deep learning approaches, thus allowing us to control and leverage
    the variance of the error» (§3.2).
  - Una versión más profunda «was overfitting the majority of the datasets» (§6.3).
  - Sin las conexiones residuales, la red «suffered from a severe overfitting» en un conjunto (§6.2).
- **Relevancia:** muestra que más profundidad no ayuda con pocos datos y que promediar varias semillas o
  modelos reduce la varianza. Lo segundo es pertinente porque la varianza entre señantes del proyecto es
  alta (0,46 a 0,685).

### d) Datos pequeños y evaluación independiente del sujeto

#### [11] Gholamiangonabadi et al., 2020: validación dejando un sujeto fuera para elegir el modelo

D. Gholamiangonabadi, N. Kiselov y K. Grolinger, "Deep neural networks for human activity recognition with
wearable sensors: Leave-one-subject-out cross-validation for model selection," *IEEE Access*, vol. 8,
pp. 133982–133994, 2020. doi: 10.1109/ACCESS.2020.3010715.

- **Estado:** CONFIRMADA (actualizado en la revisión independiente). El autor original solo pudo leer el
  resumen; el revisor descargó el PDF de acceso abierto (CC BY) desde
  `ieeexplore.ieee.org/ielx7/6287639/8948470/09144538.pdf` y leyó el texto completo. Las cifras del resumen
  se repiten en la conclusión (§VI): con el escenario MG+W (*w* = 50) y la red CNN-2C-1D, en MHEALTH, «we
  could reach 85.1% accuracy with LOSOCV», frente a 99,85 % con validación cruzada de 10 pliegues, «which
  means that it is necessary to work on the design of architectures of methods and tune them based on
  LOSOCV».
- **Qué revisa:** seis redes, unas densas (*feed forward*) y otras convolucionales, con cuatro preprocesamientos
  para reconocer actividad con sensores, evaluadas con validación cruzada dejando un sujeto fuera (LOSOCV).
- **Qué sostiene (resumen y §VI):**
  - La CNN «with two convolutions and one-dimensional filter … generalizes better than other architectures».
  - La misma CNN baja de 99,85 % con validación cruzada de 10 pliegues a 85,1 % con LOSOCV.
- **Relevancia:** en otro dominio de series multivariadas por sujeto, una CNN 1D pequeña generaliza mejor que
  las redes densas a personas nuevas, y el protocolo cambia mucho la cifra. Además respalda elegir y ajustar
  la arquitectura con LOSO, como hace el proyecto. Es otro dominio (sensores de inercia), así que sirve como
  apoyo metodológico, no para predecir cifras aquí.

#### [12] Bai et al., 2018: comparación de redes convolucionales y recurrentes para secuencias (TCN)

S. Bai, J. Z. Kolter y V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for
sequence modeling," arXiv:1803.01271, 2018.

- **Estado:** CONFIRMADA (texto leído). **Es un preprint sin revisión por pares**, aunque muy citado.
- **Qué revisa:** compara una red convolucional temporal genérica (TCN: convoluciones causales dilatadas con
  conexiones residuales) con LSTM y GRU en tareas de secuencias.
- **Qué sostiene:** «a simple convolutional architecture is more effective across diverse sequence modeling
  tasks than recurrent architectures such as LSTMs … convolutional networks should be regarded as a natural
  starting point» (§6).
- **Relevancia:** es la referencia estándar de la TCN, que el anteproyecto menciona como trabajo futuro. Sus
  tareas (música, lenguaje, tareas sintéticas) no son de clasificación de gestos, así que solo sirve para
  justificar la arquitectura, no para predecir su exactitud aquí.

*Fuentes ya verificadas en `docs/VERIFICACION_CITAS_ARQUITECTURA.md` que conviene reutilizar aquí:* Sincan y
Keles, IEEE Access 2020 (AUTSL; BLSTM con atención e independencia del señante) y Dewanto et al., ref22 (en los
estudios con MediaPipe, la LSTM es la técnica más frecuente).

---

## 3. Qué dice la literatura, frente a las preguntas del director

**¿Cómo se abordan en general estos problemas?** Con cuatro familias profundas sobre la secuencia de puntos
[1], [2], [6], [7]:

- **Recurrentes (LSTM y GRU):** modelan bien el orden, pero son secuenciales y más lentas de entrenar [1]. Es
  el esquema actual del proyecto.
- **Convolucionales temporales (CNN 1D, FCN, ResNet, InceptionTime, TCN):** son las más usadas en series de
  tiempo, rápidas y paralelizables [1], [6]. Rinden bien en archivos de conjuntos pequeños [6], [10] y, en la
  comparación genérica de Bai et al., igualan o superan a LSTM y GRU [12]; pero ninguna de estas fuentes las
  compara con recurrentes en datos pequeños por señante.
- **Grafos (ST-GCN y SL-GCN):** tratan las articulaciones como nodos de un grafo y explotan la anatomía. Dan
  las mejores cifras sobre pose [2], [3], pero son más costosas y no hay evidencia con pocos señantes.
- **Transformers:** con atención global y costosos. Funcionan sobre pose con unos 2000 videos [4], pero en
  OpenHands quedaron por debajo de los GCN [3].

En series de tiempo hay además una familia **no profunda**, la de ROCKET y MiniRocket, que está entre las más
exactas y es la más barata [8], [9].

**¿Es razonable una estrategia más sencilla?** Sí. La evidencia más fuerte viene de las series de tiempo:
ROCKET y MiniRocket con un clasificador lineal igualan a métodos mucho más caros [8], [9], y el *bake off*
multivariado recomienda ROCKET «as the default choice for MTSC problems» [8]. La revisión de Foumani et al.
explica que los archivos de referencia son pequeños y que el aprendizaje profundo destaca con muchos datos; su
ejemplo es ConvTran, que supera a ROCKET con más de 10 000 ejemplos de entrenamiento [7]. (Ese umbral es de
un solo modelo, no una regla general.) Nuestro corpus tiene 997. Nuestra línea base
clásica (RF 0,42) **no** prueba esto, porque resume la secuencia en estadísticos y pierde el orden
(`comparacion.md` §8). MiniRocket conserva el orden y es el contraste justo.

**¿Hay una red «más robusta que el MLP»?** Sí: una CNN 1D totalmente convolucional con GAP (FCN, ResNet
pequeña o TCN) [6], [12], o una CNN 1D sobre rasgos de la mano como DD-Net [5]. Es la arquitectura más usada
en series de tiempo y, en una versión pequeña, debería tener menos parámetros que la red actual y ser más
rápida en CPU (estimación propia, sin medir). Además, nuestra red
canónica ya empieza con Conv1D, así que una CNN 1D sola es también la **ablación natural** de la BiLSTM y la
atención: dice si la recurrencia aporta algo.

**¿Qué pasa con pocos datos y personas nuevas?**

- Las cifras con partición aleatoria son optimistas: SPOTER 100 % en LSA64 con 80/20 aleatorio [4]; la CNN de
  [11] baja de 99,85 % a 85,1 % con LOSO.
- Más profundidad tiende a sobreajustar [10].
- Promediar semillas o modelos reduce la varianza [10].
- La aumentación es la defensa estándar [7].
- El preprocesamiento (normalizar cada mano en su marco) puede valer tanto como cambiar de red [4].
- Con 5 señantes, lo que limita es la variación entre personas, no la capacidad del modelo. Lo sugiere también
  que la versión pequeña del proyecto (390 mil parámetros) rinda igual o algo mejor que la canónica (714 mil).

---

## 4. Arquitecturas candidatas

Entrada de referencia: 30 pasos × 252 rasgos (42 puntos × 3 coordenadas, más sus velocidades), o 30 × 126 sin
velocidades. Los costos y parámetros son **estimaciones propias, no medidas**.

| # | Candidata | Idea en una línea | Por qué podría servir aquí | Costo en CPU (estimado) | Riesgo con 997 videos | Esfuerzo en Keras sobre 30 × 252 | Código reutilizable del repositorio |
|---|---|---|---|---|---|---|---|
| A | **MiniRocket + *ridge*** | Unos 10 000 rasgos de un conjunto pequeño y fijo de núcleos convolucionales (84 núcleos, varias dilataciones) y un clasificador lineal | La base más fuerte y barata de series de tiempo [8], [9]. Conserva el orden temporal, a diferencia de la línea base de estadísticos. Casi determinista | Muy bajo: una transformación y un producto lineal por seña; sin GPU | **Bajo**: solo se ajusta la *ridge* con su α por validación cruzada | Bajo: no es Keras. sktime `MiniRocketMultivariate` + `RidgeClassifierCV` de scikit-learn (ya instalado). Hay que **añadir sktime** a `requirements.txt` y adaptar a la forma (n, 252, 30) | Particiones D-06 y carga de datos de `src/train_loso.py`; estructura de `src/linea_base_clasica.py` |
| B | **CNN 1D / TCN pequeña con GAP** | 3–4 Conv1D (con dilataciones 1-2-4-8 si es TCN), BatchNorm, GAP y softmax | Es la «red más robusta que el MLP» con más respaldo [6], [12]. Es la variante ligera para manos de DD-Net [5] y la ablación de la BiLSTM | Bajo: paralelizable; del orden de 0,1–0,5 M parámetros según los filtros | **Bajo a medio**: pocos parámetros y GAP | Bajo: capas nativas. Con núcleo 3 y dilataciones 1-2-4-8, el campo receptivo es de 31 pasos y cubre los 30 | `build_temporal_cnn` (`legacy/src/train_extra_models.py`): 4 Conv1D, GAP + GMP y densas 256/128. Conviene reducir las densas |
| C | **ResNet 1D o InceptionTime pequeña** | Bloques convolucionales residuales y GAP; InceptionTime añade un conjunto de 5 redes | Fueron las mejores en series de tiempo [6], [10] | Medio: InceptionTime multiplica por 5 el costo de inferencia | **Medio**: más profundidad sobreajusta con pocos datos [10] | Bajo a medio | `build_deep_cnn` (`legacy/src/train_v4.py`), de estilo ResNet, pero con filtros de hasta 512 (unos 2,5 M de parámetros, estimado a mano para 252 rasgos y 50 clases): hay que reducirlo |
| D | **ST-GCN pequeño sobre el grafo de las dos manos** | Convolución de grafo sobre los 21 nodos de cada mano (aristas de MediaPipe) alternada con convolución temporal | Es la familia con las mejores cifras sobre pose [2], [3] y explota la anatomía de la mano | Medio: en OpenHands, 23–99 ms por video con 2,3 M parámetros en CPU [3]. Una versión de 3–4 capas y 64 canales debería costar bastante menos (sin medir) | **Medio a alto**: sin evidencia con 5 señantes; las GCN son sensibles a conexiones ruidosas [1] | **Medio a alto**: no hay capa de grafo en Keras. Hay que remodelar a (30, 42, 6), definir la adyacencia y escribir la capa (A·X·W con `einsum`) y sus pruebas | Ninguno |
| E | **Transformer pequeño** | Proyección a d = 64, 1–2 bloques de autoatención con codificación posicional y *pooling* | Ofrece atención global y supera a Pose-GRU y Pose-TGCN en WLASL100 [4] | Medio: con 30 pasos la atención es barata; en OpenHands, 24–76 ms con 3,8 M parámetros [3] | **Alto**: SPOTER usa 5,9 M parámetros y unos 2000 videos [4]; en OpenHands queda por debajo de ST-GCN [3] | Bajo: `MultiHeadAttention` nativa | `build_transformer` (`legacy/src/train_v4.py`): 4 bloques, d = 256 y 8 cabezas, unos 2,3 M parámetros (estimado a mano). Habría que reducirlo mucho |
| F | **GRU** (en lugar de la LSTM) | Cambiar la BiLSTM por una BiGRU, con el resto igual | Es la variante recurrente más ligera que nombra el anteproyecto | Bajo: unos ¾ de los parámetros recurrentes de la LSTM | **Bajo** | Muy bajo: un cambio de capa en `construir_modelo` | `build_gru_cnn` (`legacy/src/train_v4.py`): BiGRU 128/64 + rama CNN |

**Nota sobre el código heredado:** las funciones de `legacy/src/train_v4.py` y `legacy/src/train_extra_models.py`
solo sirven como punto de partida para **construir el modelo**. Nunca se evaluaron con particiones sin fuga ni
con LOSO, así que sus resultados de entonces no cuentan como evidencia. El entrenamiento debe pasar por el
pipeline canónico: representación, regla «mascara», aumentación ×10 y particiones D-06.

---

## 5. Recomendación: qué probar primero bajo LOSO

**1. MiniRocket + *ridge* (A).** Es lo primero que conviene probar.

- Es lo más barato de implementar y de correr, y responde directamente a lo de «una estrategia más sencilla».
- El *bake off* multivariado recomienda ROCKET como opción por defecto [8], MiniRocket lo iguala en
  exactitud a menor costo [9] y la revisión de Foumani et al. explica por qué las redes profundas no destacan
  en archivos pequeños [7].
- Si iguala a la red canónica (0,58–0,61), es un hallazgo útil para la tesis: con 997 videos y 5 señantes, la
  complejidad de la red no es lo que limita. Si queda claramente por debajo, refuerza la justificación de la
  red con un contraste más exigente que el Random Forest sobre estadísticos.
- **Encaje:** `09_metodologia.tex:131-134` compromete «al menos una línea base más simple», y MiniRocket puede
  presentarse como una línea base adicional que conserva el orden temporal. Aun así, añade una dependencia
  (sktime) y una comparación nueva, así que conviene que el director la apruebe.

**2. CNN 1D / TCN pequeña con GAP (B).** Es la segunda.

- Es la «red más robusta que el MLP» con más respaldo [5], [6], [12].
- Reducida, debería tener menos parámetros y ser más rápida en CPU que la BiLSTM con atención (sin medir;
  `build_temporal_cnn` tal cual tiene unos 0,65 M, casi lo mismo que la canónica), lo que importa para el prototipo
  en tiempo real.
- Hace las veces de **ablación** de la parte recurrente de la red actual. Con 4 capas dilatadas, el campo
  receptivo cubre los 30 pasos.
- Se parte de `build_temporal_cnn` reducido y, como variante barata, se puede añadir el rasgo JCD de DD-Net:
  distancias entre los 21 puntos de cada mano, es decir, 210 por mano y cuadro.
- **Encaje:** la TCN está declarada como trabajo futuro (`08_marco.tex:57-61`: «estas no se implementarán en el
  proyecto»). **Probarla amplía el alcance: decisión del director.**

**3. Opcional: ST-GCN pequeño (D).** Es la tercera.

- Es la familia que la literatura sobre pose señala como la de mejor exactitud [2], [3].
- El esfuerzo y el riesgo son mayores: hay que escribir la capa de grafo y no hay evidencia con pocos
  señantes.
- Solo conviene si (1) y (2) dejan tiempo en el cronograma y el director acepta ampliar el alcance.

**En segundo plano.** El **Transformer (E)** tiene el mayor riesgo con este tamaño de datos [3], [4]. La
**GRU (F)** es barata, pero no hay evidencia de que cambie mucho frente a la LSTM. Puede correrse como control
casi gratis si el director quiere cubrir las tres alternativas que nombra el anteproyecto (GRU, TCN y
Transformer), pero no es prioritaria.

**Cómo compararlas sin sesgo**, siguiendo lo ya hecho en el proyecto:

- Mismas particiones D-06, misma representación de 30 × 252, misma aumentación ×10 y semillas 2031–2033.
- **Una sola configuración por familia, fijada de antemano** con los valores por defecto de la literatura y sin
  nueva búsqueda, para no sobreajustar la validación.
- Criterio escrito antes de correr.
- McNemar sobre las 997 predicciones LOSO agregadas frente a la canónica, como en `comparacion.md`.
- Informar parámetros y latencia en CPU.

Con 5 pliegues, ninguna prueba pareada por pliegue llega a p < 0,05 (`BUSQUEDA_HIPERPARAMETROS.md` §2), así
que la decisión debe apoyarse en McNemar sobre las predicciones agregadas y en el tamaño del efecto.

**Ajustes que no cambian de arquitectura y la literatura respalda:**

- Normalizar cada mano en su propio marco local (+14 puntos en la ablación de SPOTER [4]).
- Promediar las predicciones de varias semillas (InceptionTime [10]).

Pueden valer tanto como cambiar de red y cuestan menos, pero no se evaluaron aquí.

---

## 6. Lo que no se pudo verificar

| Elemento | Motivo |
|---|---|
| ~~Texto completo de Gholamiangonabadi et al. [11]~~ | **Resuelto en la revisión independiente:** el PDF de acceso abierto sí se descargó y se leyó; pasa a CONFIRMADA |
| ~~Sección de conclusiones de Pasos Ruiz et al. [8]~~ | **Resuelto en la revisión independiente:** las conclusiones están en el texto de PMC y se citan en la ficha. Confirman que la versión de arXiv (2007.13156) es un borrador anterior sin ROCKET |
| Paginación del editor de [1], [6], [7], [10] y [12] | Se leyeron las versiones de arXiv o de autor; se cita por sección |
| Orden de autores de DD-Net [5] | Crossref, OpenAlex y Semantic Scholar dan Yang, Wu, Sakti, Nakamura; el PDF de arXiv, Yang, Sakti, Wu, Nakamura. Se cita el orden del editor; la página de ACM no se pudo abrir |
| Que MiniRocket rinda bien con 252 canales y solo 30 pasos | El artículo [9] evalúa series univariadas de UCR y [8] evalúa ROCKET, no MiniRocket, en multivariadas. Es una hipótesis que hay que probar |
| Costos y parámetros de la Tabla del §4 | Son estimaciones propias, no medidas en este equipo |
| Evidencia de ST-GCN o Transformer con ≤ 5 señantes | No se encontró ninguna revisión ni estudio con texto verificable |

---

## Explicación simple

Reconocer una seña a partir de los puntos de las manos es un problema de «patrones en el espacio y el tiempo»,
como reconocer gestos o acciones del cuerpo. Las revisiones muestran cuatro maneras de hacerlo con redes
(recurrentes, convolucionales en el tiempo, de grafos y Transformers) y una sin red profunda, MiniRocket, que
es muy barata y suele rendir igual que las redes cuando hay pocos datos. Con solo 997 videos y 5 personas, lo
que más pesa es la diferencia entre personas, no el tamaño de la red. Por eso conviene probar primero
MiniRocket y una red convolucional pequeña. Son más simples que la actual, corren rápido en un portátil y dicen
si la parte recurrente de la red de verdad aporta. Una red de grafos sobre la mano es la opción más prometedora
según la literatura, pero también la más cara de construir. Varias de estas pruebas amplían lo prometido en el
anteproyecto, así que decidir si se hacen le corresponde al director.
