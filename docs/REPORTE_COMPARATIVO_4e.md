# Reporte comparativo del desempeño bajo los dos esquemas de partición

**Actividad 4.e · Resultado esperado RE6 · BORRADOR del 16-sep-2026 · 2.ª versión**

> **Qué cambió respecto de la 1.ª versión.** Una revisión independiente cotejó cada cifra contra su
> fuente. A raíz de ella se corrigieron: siete citas de línea del anteproyecto (§2.2, §6.1, §7.1–7.4,
> §8.1), una frase invertida sobre el contraste entre representaciones (§5.6), una atribución
> equivocada de un límite del extractor (§8.4, punto 14), la cota del efecto de la aumentación sobre
> la línea base (§4.4, que confundía el salto entre dos modos con el efecto total) y el rotulado de
> dos listas de confusiones que estaban truncadas sin decirlo (§5.2). Ninguna cifra nueva: todas
> siguen saliendo de archivos ya guardados.
>
> Una segunda verificación independiente cotejó otra vez cada cifra, cada clave de JSON y cada cita
> de línea del anteproyecto contra su fuente y no halló ninguna cifra equivocada. Corrigió cuatro
> detalles de redacción, sin tocar ningún dato: la cota de unicidad de la cifra de `ref15` (§6.3),
> que era más ancha de lo que los `.tex` sostienen; la pareja «+0,031 ms / unas 90 veces» de §7.3,
> que mezclaba dos magnitudes de N-40; una remisión equivocada en §2.4; y una nota de método de
> §7.2 escrita en primera persona.

> **Qué es este documento.** Un borrador de redacción. Reúne cifras que **ya estaban medidas y
> guardadas** en el repositorio antes de escribirlo; no se ejecutó ningún experimento para
> producirlo y no se tomó ninguna decisión nueva. Cada cifra lleva su fuente: archivo y clave del
> JSON, o entrada de `DECISIONES_TESIS.txt`. Las cifras de la evaluación en línea y de la latencia
> con voz real **no están aquí porque no existen todavía** (§7).
>
> **Qué no es.** No es el capítulo de resultados de la tesis ni sustituye al informe final. No
> incorpora las cifras del Bloque 1 (§2.2.7 del anteproyecto), que no ha empezado.
>
> Las rutas relativas se refieren a la raíz del repositorio. Las rutas del anteproyecto se refieren a
> `Anteproyecto/Anteproyecto_FINAL_LaTeX/`. Se usa coma decimal.

---

## 1. Qué compara este reporte y con qué criterio

El anteproyecto pide este documento en tres sitios, y los tres dicen lo mismo con distinto grado de
detalle:

- **RE6** (`10_resultados.tex:72-76`): «Un reporte comparativo del desempeño bajo los dos esquemas de
  partición de los datos: la partición aleatoria a nivel de muestra y la partición independiente del
  señante. El reporte documenta la diferencia observada entre ambos esquemas y discute sus
  implicaciones sobre la capacidad de generalización del modelo».
- **Actividad 4.e** (`09_metodologia.tex:522-525`): el reporte incluye «la diferencia observada entre
  ambos, el reporte de cada pliegue de la validación cruzada con la condición del señante excluido, y
  su discusión en términos de capacidad de generalización del modelo».
- **§2.2.4** (`09_metodologia.tex:145-147`): «Los resultados obtenidos bajo cada esquema se reportan
  por separado y **nunca se presentan como equivalentes**; su contraste es el objeto del reporte
  comparativo de la actividad 4.e y del resultado esperado RE6».

El **criterio de lectura** lo fija el propio anteproyecto y conviene dejarlo escrito antes de
enseñar cualquier número:

1. **No hay umbral de aprobación.** `06_alcances.tex` («El proyecto no fija umbral de aprobación en
   ninguna de las tres variables») y `09_metodologia.tex:352-357` («el protocolo no define el éxito
   del prototipo en términos de un umbral mínimo de exactitud…»). Lo que se compromete es el rigor
   del procedimiento de medición, no superar una cifra. En consecuencia, **este documento no declara
   ningún resultado como suficiente o insuficiente**.
2. **Las cifras se interpretan por comparación**, y solo con dos referentes: la línea base propia
   (§2.2.4) y los valores publicados para la LSC (`09_metodologia.tex:355-357`,
   `08_marco.tex:206-212`).
3. **La comparación con lo publicado es de contexto, no un ranking** (`08_marco.tex:354-361`): «Cada
   trabajo emplea vocabularios de distinto tamaño y dificultad, conjuntos de datos con distinta
   variabilidad de señantes, particiones diferentes y métricas que no siempre coinciden […] las
   precisiones se usan como contexto y no como un ranking».
4. **Las métricas que se reportan** son top-1, top-3, precision, recall y F1-score, con la matriz de
   confusión, indicando en cada medición a cuál de los dos esquemas corresponde
   (`08_marco.tex:206-212`).

Conviene además recordar de dónde sale el diseño de doble partición: `09_metodologia.tex:340-348`
enuncia que «el desempeño medido sobre señantes ya vistos durante el entrenamiento puede descender de
forma apreciable al evaluarse sobre señantes no vistos», declara expresamente que **el anteproyecto
no cuantifica esa caída** porque sus fuentes no la sostienen con esa precisión, y remite la cifra a
la medición del propio proyecto. Este documento es donde esa cifra se reporta.

---

## 2. Los dos esquemas de partición, lado a lado

Todo lo de esta sección sale de la misma corrida:
`models/loso_2026-09-15_holistic_mascara_manos/resultados_loso.json` (modelo «manos», 15-sep-2026).
Configuración común declarada en la clave `configuracion` de ese archivo: arquitectura
`CNN+BiLSTM+atencion (train_final_100.build_model)`, `seq_length` 30, `n_features` 252, velocidades
sí, `aug` 10, `batch_size` 64, `patience` 25, `dropout` 0,3, `seed_entrenamiento` 2026, monitor
`val_accuracy`. Particiones de D-06 (`data/partitions/loso_5_pliegues.json` y
`data/partitions/aleatoria_80_10_10.json`, semilla 2026).

### 2.1 Resumen

| | **Partición aleatoria** 797/100/100 | **Partición independiente del señante** (LOSO, 5 pliegues) |
|---|---|---|
| Qué separa | muestras | señantes completos |
| Señantes en prueba | los 5, también presentes en entrenamiento | 1 por pliegue, ausente del entrenamiento |
| n entrenamiento | 797 (7.970 tras aumentación ×10) | 697–699 (6.970–6.990) |
| n validación | 100 (2 por seña, D-06) | 100 por pliegue (2 por seña, de los 4 señantes de entrenamiento) |
| n prueba | 100 (2 por seña) | 997 en total; 198–200 por pliegue |
| Modelos entrenados | 1 | 5 |
| **top-1** | **0,88** | **0,5797 ± 0,0860** (media ± DE entre pliegues) |
| top-3 | 0,98 | 0,7703 ± 0,0887 |
| precision macro | 0,8967 | 0,5574 (media entre pliegues) |
| recall macro | 0,88 | 0,5810 |
| F1 macro | 0,8707 | 0,5374 |
| Errores | 12 de 100 | 419 de 997 |
| Mejor val_accuracy | 0,96 | 0,932 ± 0,0259 |

Fuentes, en `resultados_loso.json`: columna aleatoria → clave `aleatoria_80_10_10`
(`top1`, `top3`, `precision_macro`, `recall_macro`, `f1_macro`, `n_errores`, `n_train_original`,
`n_val`, `n_test`, `mejor_val_accuracy`); columna LOSO → clave `agregado`
(`top1.media`, `top1.desviacion_estandar`, `top3.*`, `precision_macro.media`, `recall_macro.media`,
`f1_macro.media`, `mejor_val_accuracy.*`). El conteo de 419 errores es la suma de `n_errores` de los
cinco pliegues (108 + 93 + 78 + 77 + 63) y coincide con
`models/analisis_errores_manos_2026-09-15/analisis.json`, clave `confusiones.n_errores`.

**Diferencia observada en top-1: 0,88 − 0,580 = 0,30 (30 puntos).** Es una resta entre las dos filas
de la tabla, no una cifra medida por separado; léase con las advertencias de §2.3.

### 2.2 Cada pliegue, con el señante excluido (lo que pide la actividad 4.e)

| Pliegue | Señante fuera | n prueba | top-1 | top-3 | precision | recall | F1 | mejor val_accuracy | épocas |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0000 | 200 | 0,460 | 0,650 | 0,4386 | 0,460 | 0,4207 | 0,95 | 67 |
| 1 | 0001 | 199 | 0,533 | 0,714 | 0,4925 | 0,535 | 0,4629 | 0,90 | 46 |
| 2 | 0002 | 198 | 0,606 | 0,788 | 0,6022 | 0,610 | 0,5728 | 0,91 | 42 |
| 3 | 0003 | 200 | 0,615 | 0,830 | 0,6045 | 0,615 | 0,5852 | 0,96 | 61 |
| 4 | 0004 | 200 | 0,685 | 0,870 | 0,6492 | 0,685 | 0,6455 | 0,94 | 62 |
| **Media** | — | 997 | **0,5797** | 0,7703 | 0,5574 | 0,5810 | 0,5374 | 0,932 | 55,6 |
| **DE (ddof=1)** | — | — | 0,0860 | 0,0887 | 0,0880 | 0,0860 | 0,0927 | 0,0259 | 10,9 |

Fuente: `resultados_loso.json`, clave `pliegues[k]` (`senante_test`, `n_test`, `top1`, `top3`,
`precision_macro`, `recall_macro`, `f1_macro`, `mejor_val_accuracy`, `epocas_ejecutadas`) y clave
`agregado`. La nota `agregado._nota` del propio archivo declara que la desviación es la muestral
(ddof = 1) sobre los pliegues ejecutados.

Dos lecturas que las cifras sostienen:

- **El rango entre señantes es de 22,5 puntos de top-1** (0,460 en 0000 a 0,685 en 0004; resta entre
  las dos filas). Es del mismo orden que la diferencia entre algunas de las decisiones de diseño de
  §3, lo que obliga a leer con cautela cualquier diferencia de pocos puntos medida sobre 5 señantes.
- **Dentro del propio LOSO hay una segunda brecha, la de validación frente a prueba**: la validación
  media de los cinco pliegues es 0,932 y la prueba 0,580 (35 puntos). Esa validación sale de los
  cuatro señantes que sí están en el entrenamiento (D-06), así que mide otra cosa que la prueba. El
  registro ya lo declara como limitación en N-30 («la validación del LOSO sale de los señantes
  vistos, así que es un indicador sesgado de la generalización a señantes nuevos») y en D-09.

**Sobre «la condición del señante excluido».** La actividad 4.e pide reportar cada pliegue *con la
condición del señante excluido*. Hoy solo se puede identificar por su identificador del corpus
(0000–0004) y por propiedades medidas de sus clips (§5.3). **No consta en `DECISIONES_TESIS.txt`
ninguna caracterización por señante** —edad, condición auditiva, lateralidad, si es o no usuario
nativo— procedente de la actividad 1.b. Si el depósito LSC50 publica esa información, falta
consignarla y este apartado queda incompleto hasta entonces. Lo que sí está declarado en el
anteproyecto (`09_metodologia.tex:242-244`) es que LSC50 «declara señantes no nativos entre sus cinco
voluntarios e intérpretes no nativos entre los participantes visibles en los datos».

### 2.3 Por qué los dos esquemas no son comparables entre sí

Esta es la advertencia central del documento. **Las dos columnas de §2.1 no son dos mediciones de lo
mismo con distinto resultado: son dos mediciones de cosas distintas.** La resta de 30 puntos describe
la distancia entre dos protocolos, y atribuirla entera a «generalización a señantes nuevos» sería más
de lo que estos datos sostienen. Razones concretas, todas verificables en los archivos citados:

1. **Conjuntos de prueba de tamaño muy distinto.** 100 muestras frente a 997 (`resultados_loso.json`,
   `aleatoria_80_10_10.n_test` y suma de `pliegues[k].n_test`). En el esquema aleatorio **un punto de
   top-1 es una sola muestra**: la cifra 0,88 tiene una resolución de 0,01 y una incertidumbre de
   muestreo que el propio registro ya usó como argumento (N-17: «100 muestras: 4 puntos = 4
   muestras»).
2. **Conjuntos de entrenamiento de tamaño distinto.** 797 muestras frente a 697–699
   (`n_train_original`). El modelo del esquema aleatorio ve alrededor de un 14 % más de datos.
3. **Un modelo frente a cinco.** La columna aleatoria es un modelo y una partición; la columna LOSO
   es la media de cinco modelos entrenados por separado, cada uno con su parada temprana
   (`epocas_ejecutadas` va de 42 a 67). No son el mismo sistema evaluado dos veces.
4. **La prueba aleatoria no mide generalización a una persona nueva.** Los cinco señantes aparecen en
   entrenamiento y en prueba. El anteproyecto la conserva por otra razón explícita: «sirve de
   referencia y permite comparar con la literatura que reporta bajo ese esquema»
   (`09_metodologia.tex:139-142`).
5. **La elección del esquema cambia qué decisión se habría tomado.** En D-08 los tres candidatos de
   representación se ordenan de una manera bajo LOSO y **al revés** bajo el esquema aleatorio
   (§3.3). Es la prueba más directa de que los dos esquemas no son intercambiables: no solo dan
   cifras distintas, favorecen configuraciones distintas.
6. **El anteproyecto lo prohíbe expresamente**: «se reportan por separado y nunca se presentan como
   equivalentes» (`09_metodologia.tex:145-147`); y el proyecto se compromete «a no presentar como
   resultado —ni siquiera como preliminar— ninguna medición obtenida con una partición que no separe
   señantes sin declarar expresamente esa condición» (`09_metodologia.tex:348-351`).

**Cuál es la cifra principal.** El registro ya lo fijó: la cifra principal del proyecto es la del
LOSO, y la aleatoria se reporta «como contraste optimista, no como cifra principal» (N-25, repetido
en N-30 y N-31). Este reporte mantiene ese criterio.

### 2.4 El modelo canónico bajo el esquema aleatorio

El modelo que usa el prototipo es otro entrenamiento, no el del LOSO:
`models/canonico_holistic_mascara_manos/` (5 semillas, selección solo por validación). Sus cifras
—clave `prueba` de `models/canonico_holistic_mascara_manos/resultados.json`— son:

| Sistema | top-1 | top-3 | precision | recall | F1 | errores |
|---|---|---|---|---|---|---|
| Modelo seleccionado (semilla 2026) | 0,88 | 0,98 | 0,8967 | 0,88 | 0,8707 | 12 de 100 |
| Ensemble de 5 semillas (promedio de softmax) | 0,91 | 1,00 | 0,93 | 0,91 | 0,9087 | 9 de 100 |

Su validación (clave `seleccion.metricas_validacion_checkpoint`) es top-1 0,96, top-3 0,99, F1
0,9573. Procedencia declarada en `modelo_canonico.procedencia.json`: representación «manos», bloques
`mano_izq`/`mano_der`, 126 rasgos por paso, 252 de entrada con velocidades, 30 pasos, normalización
«centrado en el punto medio de los hombros (postura 11 y 12) y escalado por su distancia en xy»,
`mano_ausente` «mascara». Umbral de operación 0,20 en `modelo_canonico.umbral.json` (decisión D-09).

**Advertencia.** Estas cifras son del esquema aleatorio y **solo de él**. No existe hasta hoy ninguna
cifra de exactitud del prototipo medida en vivo: las dos sesiones registradas (N-26, N-32) las ejecutó
el autor, que sí está ausente del corpus pero es oyente y no usuario fluido de la LSC, y de ellas el
proyecto decidió no reportar exactitud (`09_metodologia.tex:247-262`; §8.1, punto 4). Además el
registro deja constancia de que el ensemble no es el sistema del prototipo:
`MODELO_POR_DEFECTO` apunta al modelo seleccionado (N-31).

---

## 3. Evolución de las decisiones que movieron el resultado

Tres decisiones cambiaron la cifra del LOSO entre el 13 y el 15 de septiembre. Las tres fijaron su
criterio **por escrito y antes de ver el resultado**, y las tres se resolvieron con el esquema
independiente del señante. Todas las cifras de esta sección son de LOSO salvo donde se indique.

### 3.0 Punto de partida

| Corrida | Rasgos | LOSO top-1 | top-3 | F1 | Aleatoria top-1 | Fuente |
|---|---|---|---|---|---|---|
| N-08 (13-sep, particiones D-02) | CSV publicados, 225 + velocidades | 0,524 ± 0,082 | 0,725 | 0,481 | 0,850 | N-08 |
| N-17 (14-sep, particiones D-06) | CSV publicados, 225 + velocidades | 0,515 ± 0,075 | 0,700 | 0,470 | 0,890 | N-17 |

N-08 es la primera vez que se midió la brecha que el anteproyecto anticipó sin cuantificar; su
lectura registrada es «la caída al pasar de señantes vistos a no vistos es de ~33 puntos».

### 3.1 Extractor propio (D-05): la cifra bajó antes de subir

D-05 sustituyó los CSV de puntos característicos publicados por el corpus por una extracción propia
con `HolisticLandmarker` de MediaPipe 0.10.35, para tener **un solo extractor** en el corpus y en el
prototipo (motivación de D-01, §1.4 del anteproyecto).

| Fuente de rasgos | LOSO top-1 | top-3 | F1 | Aleatoria top-1 | Fuente |
|---|---|---|---|---|---|
| CSV publicados (N-17) | 0,515 ± 0,075 | 0,700 | 0,470 | 0,890 | N-17 |
| Extractor propio, regla «ceros» (N-22) | 0,418 ± 0,117 | 0,620 | 0,384 | 0,840 | N-22; `models/comparacion_D07_2026-09-14.json`, clave `ceros` |

La caída fue de ~10 puntos y afectó a los dos esquemas, lo que el registro usa como argumento de que
no era dependencia del señante sino de los rasgos (N-22). El diagnóstico registrado: los CSV
publicados **omiten** los fotogramas en los que falta una mano, mientras que el preprocesamiento
propio los conservaba con la mano en ceros y luego los normalizaba por hombros, convirtiendo la mano
ausente en un vector constante grande (N-22, punto 1).

Cifras de la extracción propia sobre los 1.000 videos (N-21): 67.328 fotogramas, 0 fallidos;
detección de postura 100 %, rostro 99,97 %, mano izquierda 93,8 %, mano derecha 97,2 %, sin mano
alguna 0,44 %.

### 3.2 Regla de mano ausente (D-07): +11,5 puntos

D-07 fijó el criterio antes de correr: gana la variante con **mejor exactitud de validación media**
entre los cinco pliegues LOSO; el top-1 de prueba se reporta para las tres pero no decide; empate por
debajo de 1 punto → gana «descartar».

| Variante | Validación media | LOSO top-1 | top-3 | F1 | Aleatoria t1/t3/F1 |
|---|---|---|---|---|---|
| ceros | 0,838 | 0,418 ± 0,117 | 0,620 | 0,384 | 0,840 / 0,970 / 0,818 |
| descartar | 0,910 | 0,537 ± 0,125 | 0,724 | 0,486 | 0,930 / 0,970 / 0,926 |
| **mascara** (elegida) | **0,924** | 0,533 ± 0,176 | 0,694 | 0,498 | 0,930 / 0,980 / 0,926 |

Fuente: `models/comparacion_D07_2026-09-14.json`, claves `ceros`, `descartar`, `mascara`
(`val_media`, `loso_top1`, `loso_top1_de`, `loso_top3`, `loso_f1`, `aleatoria_top1`,
`aleatoria_top3`, `aleatoria_f1`); tabla reproducida en N-23.

Aplicación del criterio: 0,924 frente a 0,910 son 1,4 puntos, por encima del umbral de empate → se
adopta «mascara». **El registro deja escrita la lectura incómoda** (N-23): «en prueba las dos
variantes son indistinguibles en top-1 (0,533 frente a 0,537, con dispersiones de 0,13-0,18) y
“descartar” es mejor en top-3 (0,724 frente a 0,694); no se cambia la elección por eso, porque el
criterio era la validación y así se declaró». Este reporte no altera esa lectura.

### 3.3 Representación de entrada (D-08): +4,6 puntos, y el orden se invierte entre esquemas

D-08 resolvió la actividad 2.a (la pregunta P2 de la sistematización) con los tres candidatos que
fija `09_metodologia.tex:103-114`. Criterio, fijado antes de correr: mayor validación media del LOSO;
si la diferencia con el siguiente es menor de 1 punto, gana el de menos rasgos.

| Candidato | Rasgos/paso | Validación media | **LOSO top-1** | top-3 | F1 | **Aleatoria top-1** | s/pliegue |
|---|---|---|---|---|---|---|---|
| **C1 manos** (elegido) | 126 | **0,932** | **0,5797 ± 0,0860** | 0,7703 | 0,5374 | 0,88 | 270,9 |
| C2 manos + postura | 225 | 0,924 | 0,5334 ± 0,1757 | 0,6940 | 0,4976 | 0,93 | 331,7 |
| C3 manos + postura + cejas y labios | 405 | 0,918 | 0,5085 ± 0,1087 | 0,6901 | 0,4644 | 0,95 | 373,4 |

Fuente: `models/comparacion_D08_2026-09-15.json`, clave `candidatos[i]` (`nombre`, `nf`, `val`,
`t1`, `t1_de`, `t3`, `f1`, `al1`, `tiempo`); regla aplicada en la clave `regla_aplicada`
(«empate a menos de 1 punto entre ['C1 manos', 'C2 manos+postura'] (0,8 puntos entre los dos
primeros): gana el de menos rasgos») y candidato elegido en `elegido`.

**Este es el hallazgo más pertinente para el reporte de 4.e**: bajo LOSO el orden es C1 > C2 > C3;
bajo el esquema aleatorio es **C3 (0,95) > C2 (0,93) > C1 (0,88)**. La lectura que el registro fijó
(N-30) es que postura y rostro «ayudan a reconocer a las personas ya vistas y perjudican a las
nuevas», porque aportan rasgos propios de cada señante que el modelo aprende en lugar de la seña.
Es la dependencia del señante que el anteproyecto anticipa, aquí localizada en bloques concretos de
la entrada.

**Lo que este resultado no dice** (limitación ya escrita en N-30 y que conviene no ensanchar): no
dice que los rasgos no manuales carezcan de valor en la LSC —Oviedo 2001, `ref13`, documenta lo
contrario para el ceño fruncido y el abocinamiento de labios—, sino que *esta* representación de
ellos, con *este* corpus, *este* extractor y **cinco** señantes, no mejoró el reconocimiento de
señantes nuevos. Se declaran además: la cara mide ≈94 px en COLOR_BODY y MediaPipe mide mal las cejas
con la cabeza inclinada (N-28.4, N-30). Y: «manos» se normaliza con los hombros de la postura, de
modo que la posición de las manos respecto al cuerpo sí entra en la entrada aunque el bloque de
postura no (procedencia del modelo y N-30).

Recall LOSO agregado por grupo de señas, de N-30 y verificado contra
`analisis.json` (clave `controles.grupos_n30_vs_registro_coincide` = `true`):

| Grupo | n señas | C1 manos | C2 manos+postura | C3 +cejas/labios |
|---|---|---|---|---|
| interrogativos | 7 | 0,379 | 0,264 | 0,357 |
| emociones | 4 | 0,688 | 0,562 | 0,588 |
| bien / mal / más o menos | 3 | 0,283 | 0,350 | 0,167 |
| resto | 36 | 0,632 | 0,598 | 0,558 |

### 3.4 Trayectoria completa del LOSO

| Estado | LOSO top-1 | Qué lo fijó |
|---|---|---|
| CSV publicados, particiones D-02 | 0,524 ± 0,082 | N-08 |
| CSV publicados, particiones D-06 | 0,515 ± 0,075 | D-06 / N-17 |
| Extractor propio, «ceros» | 0,418 ± 0,117 | D-05 / N-22 |
| Extractor propio, «mascara», manos+postura | 0,533 ± 0,176 | D-07 / N-23 |
| Extractor propio, «mascara», **manos** | **0,5797 ± 0,0860** | D-08 / N-30 |

La trayectoria **no es monótona**: el cambio de extractor bajó la cifra 10 puntos y las dos
decisiones siguientes la recuperaron y la superaron. Leerla como una mejora progresiva sería
inexacto. Y la comparación entre la primera y la última fila mezcla dos cambios a la vez (fuente de
rasgos y representación), así que la diferencia de ≈6,5 puntos entre 0,515 y 0,580 no es atribuible a
ninguno de los dos por separado con estos datos.

---

## 4. La línea base clásica frente a la red (actividades 2.c y 2.d, D-10)

`09_metodologia.tex:131-137` obliga a implementar «al menos una línea base más simple […] un
clasificador clásico (SVM o Random Forest) o un MLP sobre características agregadas de la secuencia»
y a compararla con la red: «sin ella, el uso de un modelo temporal quedaría sin justificación
empírica».

D-10 fijó antes de correr: mismos datos y particiones que la red; entrada = ocho estadísticos
(media, desviación estándar, mínimo, máximo, mediana, primer paso, último paso y media del valor
absoluto de la diferencia entre pasos consecutivos) sobre cada una de las 252 columnas de la misma
secuencia que recibe la red → **2.016 rasgos**, sin PCA ni selección; escalado estándar ajustado solo
con el entrenamiento de cada pliegue; rejilla declarada; selección por top-1 de **validación**;
McNemar sobre las 997 predicciones agregadas con α = 0,05. Criterio de justificación: la red queda
justificada si su top-1 agregado es mayor **y** McNemar da p < 0,05.

**Control previo registrado** (`models/linea_base_4d_2026-09-15/resultados.json`, claves
`verificacion_tensor` y `red.comprobacion`): el tensor por muestra es (30, 252) e idéntico al del
entrenador de la red en `0000_0000_0000`, `0025_0000_0000` y `0049_0004_0003`
(`identico_al_entrenador` = `true`), y el script reproduce desde las predicciones guardadas el LOSO
de la red: esperado «0,580 ± 0,086», medido «0,5797 ± 0,0860», `reproduce` = `true`.

### 4.1 LOSO agregado (las 997 predicciones juntas), aumentación de espejo (D-10)

| Sistema | top-1 | top-3 | precision | recall | F1 macro | errores |
|---|---|---|---|---|---|---|
| **Red temporal (CNN+BiLSTM+atención)** | **0,5797** | 0,7703 | 0,5936 | 0,5801 | 0,5737 | 419 |
| Random Forest | 0,4183 | 0,6510 | 0,4189 | 0,4182 | 0,3933 | 580 |
| SVM RBF | 0,2638 | 0,4012 | 0,3404 | 0,2637 | 0,2824 | 734 |
| MLP | 0,2477 | 0,4203 | 0,3022 | 0,2478 | 0,2566 | 750 |

Fuente: `models/linea_base_4d_2026-09-15/resultados.json`, claves `red.metricas_agregadas` y
`loso.con_espejo.agregado_997.{random_forest,svm_rbf,mlp}`. Tabla también en
`models/linea_base_4d_2026-09-15/comparacion.md` §1 y en N-39.

### 4.2 Por pliegue (top-1) — la ventaja no viene de un solo señante

| Sistema | p0 (0000) | p1 (0001) | p2 (0002) | p3 (0003) | p4 (0004) | media ± DE |
|---|---|---|---|---|---|---|
| Red temporal | 0,460 | 0,533 | 0,606 | 0,615 | 0,685 | 0,580 ± 0,086 |
| Random Forest | 0,195 | 0,377 | 0,545 | 0,500 | 0,475 | 0,418 ± 0,139 |
| SVM RBF | 0,025 | 0,221 | 0,318 | 0,330 | 0,425 | 0,264 ± 0,152 |
| MLP | 0,035 | 0,221 | 0,278 | 0,325 | 0,380 | 0,248 ± 0,133 |

Fuente: `comparacion.md` §2, generada por `src/linea_base_clasica.py` desde `resultados.json`. La red
supera a Random Forest en los 5 de 5 pliegues (conclusión registrada en `resultados.json`, clave
`conclusion`).

### 4.3 McNemar frente a la red (997 predicciones LOSO agregadas, α = 0,05)

| Línea base | Aciertos red | Aciertos línea base | Solo red | Solo línea base | p (χ² con corrección) |
|---|---|---|---|---|---|
| Random Forest | 578 | 417 | 249 | 88 | 2,89 × 10⁻¹⁸ |
| SVM RBF | 578 | 263 | 351 | 36 | 2,37 × 10⁻⁵⁷ |
| MLP | 578 | 247 | 367 | 36 | 1,01 × 10⁻⁶⁰ |

Fuente: `resultados.json`, clave `comparacion_mcnemar.por_familia.{...}` (`tabla`, `p`,
`prueba_usada`, `aciertos_a`, `aciertos_b`). Concordancia con Random Forest: 329 aciertan las dos,
331 fallan las dos.

**Conclusión según D-10** (clave `conclusion` de `resultados.json`): la arquitectura temporal **queda
justificada empíricamente**: top-1 agregado 0,580 > 0,418 y p = 2,89 × 10⁻¹⁸ < 0,05.

### 4.4 Control de equidad: misma aumentación que la red

D-10 dio a la línea base solo la aumentación de espejo (factor 2), mientras la red se entrena con las
nueve transformaciones de `src/train_loso.py` (factor 10). Para que la comparación no se apoyara en
esa asimetría se repitió el LOSO dando a la línea base exactamente esas transformaciones. **No
cambió el criterio de D-10: es un control añadido.**

Hay **tres modos medidos**, los tres sobre las mismas particiones de D-06, con la misma semilla
(2026) y la misma rejilla (claves `particiones`, `semilla` y `rejilla`, idénticas en los dos
archivos citados abajo):

| Familia | sin aumentación | con espejo ×2 (D-10) | con la aumentación de la red ×10 | espejo → completa | sin aumentación → completa |
|---|---|---|---|---|---|
| Random Forest | 0,4052 | 0,4183 | **0,4423** | +0,024 | +0,037 |
| SVM RBF | 0,2357 | 0,2638 | 0,2728 | +0,009 | +0,037 |
| MLP | 0,2237 | 0,2477 | 0,2879 | +0,040 | **+0,064** |
| **Red temporal** | — | **0,5797** | **0,5797** | — | — |

Fuentes: `models/linea_base_4d_2026-09-15/resultados.json`, claves
`loso.sin_aumentacion.agregado_997` y `loso.con_espejo.agregado_997`; y
`models/linea_base_4d_2026-09-15/control_aumentacion_completa/resultados.json`, clave
`loso.con_aumentacion_completa.agregado_997`, cuya clave `aumentacion` declara «las mismas nueve
transformaciones de la red (src/train_loso.py, importar_pipeline_canonico), factor 10». Las dos
últimas columnas son **restas entre columnas de esta misma tabla**. La fila de la red es la misma en
los tres modos **por construcción**: la red no se reentrenó en ninguno de ellos, sus cifras son las
predicciones ya guardadas del LOSO «manos» (clave `red.resumen`, `dir` =
`models/loso_2026-09-15_holistic_mascara_manos`), y ese entrenamiento usa esas mismas nueve
transformaciones (`resultados_loso.json`, `configuracion.aug` = 10). **De la red, por tanto, esta
tabla no mide nada sobre el efecto de la aumentación.**

**Dos magnitudes distintas, que conviene no confundir:**

- **Paso de la aumentación de D-10 a la de la red** (espejo → completa): +0,024 (Random Forest),
  +0,009 (SVM RBF) y +0,040 (MLP); entre **+0,009 y +0,040** de top-1 agregado.
- **Efecto total de la aumentación** (sin aumentación → completa): +0,037 (Random Forest), +0,037
  (SVM RBF) y +0,064 (MLP); entre **+0,037 y +0,064**. El tope lo pone el MLP y **queda por encima
  del intervalo del paso anterior**: el intervalo de los saltos no acota el efecto total, y
  presentarlo como si lo hiciera sería una cota más estrecha que los datos.

McNemar de Random Forest con aumentación completa frente a la red (clave `comparacion_mcnemar` del
archivo del control): aciertan las dos 361, fallan las dos 339, solo la red 217, solo la línea base
80, p = 2,99 × 10⁻¹⁵.

**Lectura.** Igualar la aumentación sube a la mejor línea base 2,4 centésimas (0,4183 → 0,4423) y no
cambia la conclusión de D-10: la distancia con la red (0,5797 frente a 0,4423) no se explica por la
aumentación. Lo que estas cifras **no** dicen es cuánto subiría la línea base con otra aumentación,
con más datos o con otra entrada: son tres modos medidos sobre las mismas particiones, no una curva.

### 4.5 La línea base bajo el esquema aleatorio: la brecha la tienen los dos sistemas

| Sistema | top-1 aleatoria (100 muestras) | top-1 LOSO agregado (997) | Diferencia |
|---|---|---|---|
| Red temporal | 0,880 | 0,580 | 0,300 |
| Random Forest | 0,730 | 0,418 | 0,312 |
| MLP | 0,530 | 0,248 | 0,282 |
| SVM RBF | 0,510 | 0,264 | 0,246 |

Fuente: `comparacion.md` §5 y `resultados.json`, claves `aleatoria.con_espejo` y
`loso.con_espejo.agregado_997`; la columna «Diferencia» es la resta de las dos anteriores y arrastra
todas las advertencias de §2.3 (100 muestras frente a 997, un modelo frente a cinco).

Esto sostiene una afirmación que vale la pena dejar escrita: **la caída de señantes vistos a señantes
nuevos no es un defecto de la arquitectura temporal**; aparece con magnitud parecida en los cuatro
sistemas, incluidos los clásicos. Con este corpus y cinco señantes, generalizar a una persona nueva
es un problema de los datos y del protocolo antes que del clasificador.

### 4.6 Lo que la comparación con la línea base no dice

Reproducido de `comparacion.md` §7 y §8 y de N-39, porque es el tipo de afirmación que conviene no
ensanchar:

- **No dice que un clasificador clásico no sirva para la LSC.** Dice que, con esta entrada —ocho
  estadísticos por columna, que borran el orden temporal salvo lo que capturan el primer paso, el
  último y la media de las diferencias— y este corpus, resumir la secuencia cuesta caro.
- La línea base no recibe las nueve transformaciones en la corrida principal de D-10; §4.4 mide
  cuánto cambia eso, con los tres modos medidos y sin confundir el salto entre dos modos con el
  efecto total.
- McNemar compara aciertos pareados sobre las mismas muestras: no dice nada sobre latencia, coste de
  entrenamiento ni comportamiento del prototipo en vivo.
- Observación registrada en N-39 y **no corregida**: el MLP usa `max_iter=500` y el 10 % de validación
  interna de scikit-learn para la parada temprana, lo que puede hacer algo optimista su punto de
  parada. No toca la validación de D-06 ni la prueba.

---

## 5. Dónde falla el sistema (análisis de errores, actividad 4.d)

Fuente única de esta sección: `models/analisis_errores_manos_2026-09-15/analisis.json` (y su
`analisis.md`), sobre las **997 predicciones de prueba ya guardadas del LOSO «manos»**. Es un reporte:
no se reentrenó nada ni se decidió nada a partir de él (N-35). Sus controles internos
(clave `controles`) declaran `todo_coincide` = `true` y reproducen el top-1 por pliegue y la media
guardados.

### 5.1 Por seña

De las 50 señas, **1 tiene recall agregado 0** y ninguna queda sin ser predicha nunca
(`por_sena.senas_con_recall_0` = `[27]`, `por_sena.senas_nunca_predichas` = `[]`); 12 tienen recall
≥ 0,9 (`lectura.general`).

| Peores (recall agregado) | id | recall | F1 | soporte |
|---|---|---|---|---|
| más o menos | 27 | 0,00 | 0,000 | 20 |
| comer | 7 | 0,05 | 0,051 | 20 |
| permiso | 36 | 0,15 | 0,130 | 20 |
| niño | 14 | 0,15 | 0,136 | 20 |
| ¿cómo estás? | 26 | 0,15 | 0,150 | 20 |
| ¿cuándo? | 42 | 0,20 | 0,258 | 20 |
| mujer | 12 | 0,25 | 0,286 | 20 |
| bien | 24 | 0,30 | 0,400 | 20 |
| sentir | 28 | 0,30 | 0,444 | 20 |
| niña | 15 | 0,35 | 0,304 | 20 |

| Mejores | id | recall | F1 |
|---|---|---|---|
| todos | 48 | 1,00 | 1,000 |
| buenas noches | 3 | 1,00 | 1,000 |
| buenos días | 1 | 1,00 | 1,000 |
| juicioso | 29 | 1,00 | 0,952 |
| seña | 4 | 1,00 | 0,952 |
| diferente | 47 | 1,00 | 0,889 |
| buenas tardes | 2 | 1,00 | 0,809 |
| familia | 10 | 0,95 | 0,950 |
| nombre | 5 | 0,95 | 0,927 |
| feliz | 20 | 0,90 | 0,818 |

Fuente: `por_sena.peores_10`, `por_sena.mejores_10` y `por_sena.senas[i]`
(`nombre`, `recall`, `f1`, `soporte`). **Advertencia del propio análisis** (`lectura.general`): cada
seña tiene 4 muestras por señante (3 en tres casos) y 20 en el agregado (19 en las señas 2, 8 y 16),
así que diferencias de una o dos muestras no deben leerse como efectos.

### 5.2 Qué se confunde con qué

- Pares más confundidos **en un sentido** (`confusiones.pares`, ordenados por `conteo`; el archivo
  lista 198 pares distintos y aquí van **todos los de conteo ≥ 6**, el grupo siguiente baja a 5):
  «más o menos» → «poco» 8; «más o menos» → «yo» 8; «niña» → «niño» 6; «¿cómo estás?» → «¿cómo?» 6;
  «hola» → «hombre» 6; «permiso» → «gracias» 6; «¿qué?» → «¿dónde?» 6.
- Parejas más confundidas sumando **los dos sentidos** (`confusiones.parejas_no_ordenadas`, 163
  parejas; aquí van **las nueve con total ≥ 8**, la siguiente baja a 7): **niño/niña 11** (5 y 6);
  **¿cómo estás?/¿cómo? 11** (6 y 5); comer/abuelo 9 (4 y 5); «más o menos»/«yo» 9 (8 y 1);
  ¿qué?/¿dónde? 9 (6 y 3); «poco»/«más o menos» 8 (0 y 8, es decir un solo sentido); mujer/permiso 8
  (4 y 4); niño/tú 8 (3 y 5); permiso/¿cuándo? 8 (3 y 5).
- **El 29,6 % de los errores cae dentro de la misma categoría gramatical, frente al 13,6 % esperado
  si los errores fueran al azar** (`confusiones.por_categoria.proporcion_dentro` = 0,2959, 124 de los
  419 errores, y `proporcion_dentro_esperada_si_errores_al_azar` = 0,1356). El propio análisis
  advierte que las categorías de `grammar_mapper` son sintácticas, no de forma manual, **así que esa
  cifra no mide parecido articulatorio**.
- Señas que más aparecen como predicción errónea
  (`confusiones.predichas_mas_frecuentes_en_errores`, que es **una lista de diez entradas**, no el
  conteo completo): permiso 23 veces, niño 21, y **cinco empatadas en 19** (niña, abuelo, «por
  favor», «¿qué?» y «¿quién?»); las tres restantes de la lista quedan en 18 (comer, hombre, hola).
  Como el archivo solo guarda esas diez, no puede decirse desde aquí cuántas señas más llegan a 18.
- El 43,4 % de los errores cae en pares que se dan en los dos sentidos y el 29,4 % en pares
  clasificados como simétricos por la convención descriptiva del propio análisis
  (`confusiones.proporcion_errores_en_pares_con_ambos_sentidos`,
  `proporcion_errores_en_pares_simetricos`, y el criterio en `confusiones.criterio_simetria`).

### 5.3 Por señante, y por qué el 0000 es el difícil

| Señante | top-1 | top-3 | F1 | errores | rechazo con t = 0,20 | error entre aceptadas |
|---|---|---|---|---|---|---|
| 0000 | 0,460 | 0,650 | 0,4207 | 108 | 29,0 % | 43,7 % |
| 0001 | 0,533 | 0,714 | 0,4629 | 93 | 14,6 % | 40,6 % |
| 0002 | 0,606 | 0,788 | 0,5728 | 78 | 15,2 % | 31,0 % |
| 0003 | 0,615 | 0,830 | 0,5852 | 77 | 10,5 % | 34,6 % |
| 0004 | 0,685 | 0,870 | 0,6455 | 63 | 11,0 % | 29,2 % |

Fuente: `por_senante.{id}` (`top1`, `top3`, `f1_macro`, `n_errores`) y
`calibracion.umbral.por_senante.{id}` (`rechazo`, `error_aceptadas`).

**Por qué el 0000 es el más difícil: no se puede establecer con cinco señantes**, y el análisis lo
dice literalmente (`lectura.senante_0000`). Es extremo entre los cinco en longitud media de clip
(72,2 fotogramas; el siguiente 71,0) y en fotogramas sin mano derecha (5,5 % frente a 2,5 %), pero
**dentro** del 0000 ninguna de esas variables separa aciertos de errores con p < 0,05
(proporción sin alguna mano p = 0,6377; sin mano derecha p = 0,0759; longitud p = 0,0838). «Que un
señante sea extremo en una variable es compatible con que esa variable influya, no lo demuestra».

### 5.4 Longitud del clip y fotogramas sin mano: asociaciones, no causas

- **Longitud** (`lectura.longitud`): el signo cambia según con qué se compare. Dentro de señante los
  errores duran 1,37 fotogramas menos (p = 0,0253; Holm 0,2024); dentro de seña +0,55 (p = 0,5331);
  dentro de seña y señante +1,99 (p = 0,0574). **Ninguna diferencia sobrevive a la corrección de
  Holm.** Entre las 50 señas, la correlación entre longitud media y recall es 0,293 (p = 0,0405). La
  conclusión registrada: «con estos datos no hay una respuesta única».
- **Fotogramas sin mano** (`lectura.general`): los errores tienen algo más de fotogramas sin mano.
  De nueve comparaciones (tres variables × tres estratificaciones) **solo tres sobreviven a Holm**:
  sin mano izquierda dentro de señante (+2,06 pp, Holm 0,0056), sin mano derecha dentro de seña
  (+1,10 pp, Holm 0,0204) y sin alguna mano dentro de seña (+2,60 pp, Holm 0,0065). El propio
  análisis lo cierra así: «son asociaciones, no causas»; revisar D-07 con esto exigiría una decisión
  nueva fijada antes de medir.

### 5.5 Calibración y efecto del umbral de operación

`calibracion.global`: confianza media 0,5427 frente a exactitud 0,5797; **ECE 0,0664**, MCE 0,1365.
En 7 de los 10 intervalos con datos la confianza media queda por debajo de la exactitud.

Efecto del umbral t = 0,20 de D-09 sobre la prueba LOSO (`calibracion.umbral.global`): de 997
ensayos se aceptan 837 y se rechazan 160 (**rechazo 16,0 %**); entre las aceptadas, 540 correctas y
**297 erróneas** (exactitud entre aceptadas 0,6452, error 0,3548); se pierden por rechazo el 6,6 % de
los aciertos y se rechazan el 29,1 % de los errores.

**Esto describe el umbral ya elegido, no es un criterio para cambiarlo** (N-33, y la propia clave
`lectura.calibracion`). La limitación estaba declarada en D-09 antes de calcular: la validación sale
de señantes vistos, cuyas confianzas son más altas, así que el umbral puede quedar permisivo para
personas nuevas. N-33 confirma que se cumplió: con t = 0,20 el sistema diría en voz alta 297 señas
equivocadas de 837 aceptadas (35,5 %) frente a 129 de 539 (23,9 %) con t = 0,50, y la función
objetivo U es menor (243 frente a 281). **No se cambió la elección mirando la prueba**; el camino
correcto registrado sería cambiar primero el método (validación anidada por señante) en una decisión
nueva y reentrenar.

### 5.6 Contraste entre representaciones, seña a seña

`contraste.manos_postura` y `contraste.manos_postura_rostro` (mismas muestras,
`mismas_muestras` = `true`):

| Contraste | top-1 «manos» | top-1 del otro | Solo acierta «manos» | Solo acierta el otro | p McNemar global |
|---|---|---|---|---|---|
| manos vs. manos+postura | 0,5797 | 0,5336 | 134 | 88 | 0,0024 |
| manos vs. manos+postura+cejas/labios | 0,5797 | 0,5085 | 150 | 79 | 3,1 × 10⁻⁶ |

Frente a la representación con rostro, «manos» tiene mejor recall en 29 señas y peor en 12
(claves `ganan_con_manos` = 29 y `pierden_con_manos` = 12), y solo 2 señas tienen p < 0,05 por
separado. **Nota del propio archivo**: son 50
pruebas sin corrección por comparaciones múltiples y con 19–20 muestras por seña una sola muestra
mueve el recall 0,05; los p por seña son orientativos.

---

## 6. Comparación con los valores publicados para la LSC

`08_marco.tex` y `06_alcances.tex` fijan a la vez la obligación de comparar y el modo de hacerlo.
`08_marco.tex:499-500` señala a `ref21`, `ref41` y `ref42` como «punto de comparación obligado de los
resultados». `08_marco.tex:354-361` fija el criterio: «las precisiones se usan como contexto y no como
un ranking». `06_alcances.tex` añade que los vocabularios de los demás conjuntos públicos de LSC «se
emplean como referencia para situar los resultados frente a la literatura, no como fuente de datos de
entrenamiento».

**Regla que este borrador se impone:** solo se citan cifras que estén literalmente en los `.tex` del
anteproyecto. Donde haga falta una cifra que no está, se dice que falta y de qué referencia debería
salir.

### 6.1 Lo que los `.tex` publican

| Trabajo (según los `.tex`) | Datos | Técnica | Métrica que reporta el `.tex` | Esquema de partición según el `.tex` |
|---|---|---|---|---|
| Ortiz-Farfán y Camargo-Mendoza (2020), `ref15` | 22 clases; conjunto propio; 5 señantes en el repositorio, 6 en la evaluación al sumarse la intérprete | CNN | ≈68 % (15 de 22 clases) | Los `.tex` registran que reservaron a la persona intérprete para evaluar el prototipo (`09_metodologia.tex:219-221`); no dicen si el ≈68 % corresponde a esa condición |
| Gutiérrez Leguizamón et al. (2022), `ref16` | 39 señas estáticas, dominio hotelero, conjunto propio | CNN + MediaPipe Hands | 97,6 % | No consta en los `.tex` |
| Muñoz-Galindez y Vargas-Cañas (2023), `ref17` | LSC70W; 70 participantes | CNN + LSTM sobre fotogramas | 76 % | No consta en los `.tex` |
| Mora-Zárate et al. (2025), `ref21` | LSC-54: 54 señas / 22 participantes | Corpus de puntos característicos de MediaPipe Holistic | **«No aplica (conjunto de datos)»** | No aplica |
| Rivera-Vásquez y González-Serrano (2025), `ref41` | Conjunto propio; 27 letras y 19/28 palabras; n.º de señantes **no reportado** | MediaPipe (manos y postura, **sin rostro**) + LSTM | 90 % letras; 82 % palabras — **exactitudes de validación**, partición aleatoria del 20 %, exactitud de entrenamiento 1,00 en los tres experimentos | Partición aleatoria; **no aplica partición independiente del señante** (`08_marco.tex:318-324`, `:408-412`) |
| Muñoz Galindez y Vargas Cañas (2025), `ref42` | LSC70, subconjuntos LSC70AN y LSC70ANH; 37 clases / 70 voluntarios | CNN + BiLSTM sobre imagen y BiLSTM sobre coordenadas de MediaPipe | 75,9 % (imagen, mano); 75,5 % y 87,6 % (coordenadas, cuerpo completo y mano) | No consta en los `.tex` |

Para contexto internacional, los `.tex` también recogen (`08_marco.tex:53-55`, `:427-436`): 88,23 %
en lengua de señas india (`ref23`), 94,38 % sobre 20 clases de ASL con 2 señantes (`ref24`) y 95 %
sobre once clases de lengua de señas noruega con **partición no independiente del señante**
(`ref25`), además de los rangos de revisión 86,7–99 % (`ref22`) y 31,5–99,8 % (`ref11`). Los propios
`.tex` advierten que esas cifras «no son comparables entre sí» y «se citan únicamente para situar el
orden de magnitud».

### 6.2 Cómo se sitúan las cifras de este proyecto

| | Este proyecto | Comparabilidad con lo publicado |
|---|---|---|
| LOSO, señantes no vistos | top-1 **0,5797 ± 0,0860**; top-3 0,7703 | **Sin contraparte homogénea en los `.tex`.** De los tres trabajos que el anteproyecto fija como comparación obligada, ninguno declara en los `.tex` una partición independiente del señante, y de `ref41` los `.tex` declaran expresamente lo contrario |
| Aleatoria, señantes vistos | top-1 **0,88**; top-3 0,98 (100 muestras) | Es el esquema bajo el que reporta `ref41` (validación, 20 % aleatorio), pero con otro vocabulario (27 letras; 19 y 28 palabras frente a 50 señas), otro corpus y una cifra de **validación**, no de prueba |
| Vocabulario | 50 señas del corpus público LSC50 | `ref41` 27/19/28; `ref42` 37; `ref17` LSC70W; `ref16` 39; `ref15` 22 |
| Señantes | 5 (los del corpus) | `ref42` 70 voluntarios; `ref17` 70 participantes; `ref21` 22 participantes; `ref41` no reportado; `ref15` 5 en el repositorio |

**Por qué ninguna de esas comparaciones es directa**, con las tres razones que el propio anteproyecto
enumera (`08_marco.tex:354-361`):

1. **Corpus distinto.** Según los `.tex`, ninguno de los seis trabajos de LSC de la tabla §6.1 usa
   LSC50 como fuente de entrenamiento: `ref17` y `ref42` usan LSC70, `ref21` es el corpus LSC-54, y
   `ref15`, `ref16` y `ref41` usan conjuntos propios. La afirmación vale para esos seis trabajos y
   para lo que los `.tex` declaran de ellos, no para la literatura de LSC en general.
2. **Vocabulario distinto en tamaño y en naturaleza.** 50 señas dinámicas de vocabulario básico no es
   lo mismo que 39 señas estáticas de un dominio cerrado (`ref16`) ni que 27 letras del alfabeto
   dactilológico (`ref41`) ni que 37 señas alfanuméricas (`ref42`). El propio `.tex` lo dice: «un
   97,6 % sobre 39 señas estáticas en un dominio cerrado no es equivalente a un 76 % sobre señas
   dinámicas con 70 participantes».
3. **Protocolo distinto.** La cifra de `ref41` es de **validación** sobre una partición aleatoria del
   20 %, con exactitud de entrenamiento 1,00; la de este proyecto es de **prueba**, y la principal es
   además de señantes no vistos. Son magnitudes distintas aunque se llamen igual.

A eso se suma una asimetría de la que el proyecto es responsable y conviene declarar: **la cifra de
0,880 del esquema aleatorio se apoya en 100 muestras de prueba**, mientras las cifras publicadas se
apoyan en conjuntos cuyo tamaño los `.tex` no siempre consignan.

### 6.3 Cifras que faltan y de dónde deberían salir

- **Exactitud de `ref21` (LSC-54, Mora-Zárate et al. 2025).** Los `.tex` lo registran como conjunto de
  datos, con métrica «No aplica». El anteproyecto lo nombra punto de comparación obligado, pero **no
  hay ninguna cifra de exactitud suya en los `.tex`**: la comparación con `ref21` solo puede hacerse
  hoy sobre el corpus (54 señas, 22 participantes), no sobre desempeño. Si se quiere una cifra, debe
  salir de `ref21` a texto completo.
- **Esquema de partición de `ref42`.** Los `.tex` reportan sus tres exactitudes pero no dicen si
  separan señantes. Sin ese dato no se puede situar el 87,6 % frente a ninguna de las dos columnas de
  §2.1. Debe salir de `ref42` a texto completo.
- **Esquema de partición de `ref16` y `ref17`.** Tampoco consta en los `.tex`. Debe salir de esas
  referencias.
- **Si el ≈68 % de `ref15` corresponde a la evaluación con la intérprete reservada.** Los `.tex`
  registran ambos hechos por separado sin unirlos. De confirmarse, sería la única cifra publicada de
  LSC de la que los `.tex` dejaran constancia de haberse medido sobre un señante no visto —de
  `ref16`, `ref17` y `ref42` los `.tex` no declaran el esquema de partición, así que de sus cifras no
  puede afirmarse ni lo uno ni lo contrario— y el contraste más pertinente para el 0,5797 de este
  proyecto, aunque sobre 22 clases y un conjunto propio. Debe salir de `ref15` a texto completo.
- **Número de señantes de `ref41`.** Los `.tex` declaran expresamente que no lo reporta.

---

## 7. Lo que todavía no se puede reportar, y por qué

### 7.1 Evaluación en línea (Bloque 1): no existe

`09_metodologia.tex:206-231` define el Bloque 1 —las señas las ejecuta la persona sorda señante o
intérprete de LSC vinculada al proyecto, ausente del corpus— como **el único bloque del que se
reporta exactitud en línea**, por duplicado (todas las ejecuciones y las que no declararon variante
léxica). RE5 lo confirma (`10_resultados.tex:39-70`).

**Estado real, según el registro:** N-37 deja constancia de que «no consta en este registro la
vinculación de 0.c ni la consulta al comité de ética». `docs/PROTOCOLO_EXPERIMENTAL_4a.md` es un
borrador que «no autoriza a ejecutar sesiones» y deja 31 parámetros abiertos (PD-01 a PD-31). El
guion de cruce `src/exactitud_bloque1.py` existe desde N-41, pero el propio registro lo advierte:
**«el guion nunca ha corrido sobre una sesión real, porque el Bloque 1 no ha empezado. Todas las
cifras probadas son sintéticas y no dicen nada del desempeño del sistema»**.

Consecuencia para este reporte: **la fila «evaluación en línea» de la comparación está vacía**, y el
anteproyecto pide reportarla junto al resultado de la partición independiente del señante como
condición más exigente, sin presentarlos como equivalentes (`10_resultados.tex:65-67`).

### 7.2 Latencia con voz real: medida a medias, y se declara

N-34 cambió el prototipo para que la síntesis ocurra **antes** de sellar t5, como exige la definición
de `09_metodologia.tex:285-300`. Antes de ese cambio la etapa «síntesis» medía ≈0 ms y el registro lo
declaraba (`tts_sintesis_antes_de_t5` = `false`).

Lo que hay hoy, todo del registro (no se releyeron los archivos de `registros/` para escribir este
borrador: las cifras van tal como las consigna el registro de decisiones):

- N-36: sobre 73 registros y 71 ensayos, **17 ensayos llegaron hasta t5 y los 17 pertenecen al grupo
  «síntesis no incluida»**, con L_resp 34,3 ms; W 2.429,9 ms; umbral único 0,50. **«Todavía no hay
  ningún registro real con síntesis incluida: hace falta una sesión con cámara posterior a N-34»**.
- N-34: la síntesis a archivo mide 445–463 ms por frase (mediana 459 ms) en este Mac con la voz
  Paulina, medida sin reproducir. La consecuencia declarada: L_resp y L_total subirán ≈0,46 s
  respecto de N-26/N-32, «no es una regresión, es la parte de la latencia que antes quedaba fuera de
  la medición».
- N-26 (sesión del 14-sep, modelo manos+postura, umbral 0,50): tasa del bucle con cámara 30,4 fot./s
  (±0,2), ventana de 2,5 s = 75 fotogramas; extracción 12,0 ms por fotograma (n = 4.302; mediana
  12,1; p90 12,4); inferencia 19,7 ms; L_resp media 34,3 ms. N-32 (15-sep, modelo «manos»):
  extracción 12,3 ms, inferencia 20,4 ms, L_resp 34,3 ms en 2 ensayos aceptados. N-36 corrige el
  conteo de N-26: 62 ensayos en 3 sesiones, no 59.
- **El retardo del controlador y del dispositivo de audio no está medido.** `09_metodologia.tex:327-331`
  lo deja fuera de la medición por software y pide caracterizarlo una vez con una grabación externa;
  la actividad 4.c añade que «si esa caracterización no llega a realizarse, el reporte lo declara “no
  medido” y no supone ningún valor en su lugar». Hoy: **no medido**.

Además, N-32 advierte que su sesión fue «demasiado corta y sin registro de qué señas se intentaron:
no es una cifra de exactitud», y que las 9 ejecuciones dieron 7 rechazos por confianza con el umbral
0,50 entonces vigente.

### 7.3 Lo único comparable con lo publicado en materia de rendimiento

`09_metodologia.tex:321-324` restringe la comparación de rendimiento a dos indicadores: el costo del
extractor por fotograma y la tasa efectiva del bucle. Lo medido:

| Indicador | Valor medido | Fuente |
|---|---|---|
| Extractor, pruebas de 60 videos desde disco local | 11,317 ms/fotograma (88,36 fot./s) | `models/tasa_extractor_local_2026-09-16/tasa.json`, `comparaciones[0].media_b_ms` / `fps_b` |
| Extractor, los mismos 60 videos desde Google Drive | 11,348 y 11,388 ms (dos pasadas) | misma clave, `media_a_ms` de las dos comparaciones |
| Extractor, corrida completa de 1.000 videos con escritura masiva | 15,01 ms/fotograma; total 15,80; 63,3 fot./s en bucle | N-21 |
| Extractor, con cámara en vivo | 12,0 ms (N-26) y 12,3 ms (N-32) | N-26, N-32 |
| Tasa del bucle de captura y extracción con cámara | 30,4 fot./s (±0,2) | N-26, confirmada en N-32 |

Referentes que los `.tex` publican (`08_marco.tex:228-232`): 44,92 ms por fotograma clave y 950 ms
por video de prueba en CPU (`ref25`); 25–30 fot./s en un Intel i5 sin GPU (`ref24`); 14 fot./s de
adquisición por cámara (`ref42`). Los `.tex` advierten que ninguna de las tres incluye síntesis de
voz ni se mide desde un evento de la persona usuaria, **de modo que no son referente de la latencia
total ni de la de respuesta**; solo estos dos indicadores se sitúan frente a ellas.

Dos cautelas propias que conviene no omitir: (a) la tasa de 30,4 fot./s con cámara está acotada por
la cámara y por la ventana de 2,5 s, no solo por el extractor, así que compararla con los 25–30
fot./s de `ref24` mide cosas que no son idénticas; (b) la brecha entre los 11,3 ms de las pruebas y
los 15,0 ms de la corrida completa sigue **sin explicación cerrada**: N-40 descartó que leer desde
Google Drive la explique (diferencia de +0,031 ms en la etapa `holistic` y +0,039 ms en el bucle
completo, por debajo del umbral de relevancia de 1,0 ms fijado antes de medir; N-40 lo resume como
«unas 90 veces menor» que la brecha de 3,5 ms que N-21 dejó por explicar), pero deja vivas dos
hipótesis sin resolver —la sincronización durante la escritura de 5.000 archivos y la subida de la
carga del equipo de 2,2 a 7,0 durante los 23 minutos de N-21—.

### 7.4 Otras cosas que este reporte no puede cerrar

- **Matriz de confusión de la partición aleatoria.** Existe como archivo
  (`models/canonico_holistic_mascara_manos/evaluacion_modelo_seleccionado/matriz_confusion.json` y
  `.png`, declarados en `resultados.json`), pero con 100 muestras (2 por seña) su lectura por seña es
  de una o dos muestras; el análisis de errores de §5 se hizo deliberadamente sobre las 997 del LOSO.
- **Tamaño del vocabulario efectivamente soportado** (`10_resultados.tex:58`): el modelo tiene 50
  salidas y el corpus 50 señas, pero la cifra que el anteproyecto pide es la del **prototipo
  integrado en la evaluación en línea**, que no se ha hecho.
- **Validación lingüística de las glosas (actividad 1.d)**: depende también de la persona vinculada.
  Este reporte usa los nombres de seña tal como los trae `src/grammar_mapper.py`
  (`analisis.json`, clave `fuentes.vocabulario`), sin juicio lingüístico sobre ellos.

---

## 8. Límites y amenazas a la validez

### 8.1 Del corpus y del diseño

1. **Cinco señantes.** Es el límite que atraviesa todo el documento. Con 5 pliegues, cada punto de la
   media LOSO se apoya en una sola persona; el rango entre señantes es de 22,5 puntos (§2.2). El
   anteproyecto ya lo anticipa: «un conjunto de 50 clases es modesto para entrenar una red recurrente
   y […] un número mayor de muestras y de señantes sería deseable»
   (`09_metodologia.tex:55-58`). Varias preguntas del análisis de errores quedan sin respuesta
   explícitamente por esta razón (§5.3).
2. **997 muestras, con desbalance declarado.** 47 señas con 20 muestras y 3 («buenas tardes»,
   «vivir», «abuelo») con 19, tras excluir tres duplicados (D-06). 4 repeticiones por señante y seña,
   3 en tres casos (`analisis.json`, `lectura.general`).
3. **Las condiciones de registro no son variables de diseño.** Iluminación, encuadre, fondo,
   distancia, número de señantes y de repeticiones «son las condiciones que el corpus ofrece» y el
   anteproyecto las declara como limitación porque «acotan la variabilidad frente a la cual puede
   comprobarse la robustez del reconocedor» (`06_alcances.tex`).
4. **No hay exactitud del prototipo medida en vivo.** Las dos sesiones en vivo registradas (N-26,
   N-32) las ejecutó el autor, que está ausente del corpus pero es oyente y no usuario fluido de la
   LSC; de ellas **no se reporta ninguna cifra de exactitud** por decisión del propio proyecto
   (`09_metodologia.tex:247-262`). La exactitud en línea depende del Bloque 1, que no ha empezado
   (§7.1).

### 8.2 De los procedimientos de decisión

5. **Las tres decisiones de §3 se resolvieron por validación de señantes vistos.** D-07, D-08 y D-09
   fijaron como criterio la validación, que en el LOSO sale de los cuatro señantes que sí entran al
   entrenamiento. N-30 lo declara como indicador sesgado de la generalización a señantes nuevos y
   señala la validación anidada por señante como el camino correcto, dejado a trabajo futuro. En D-08
   la validación coincidió con el orden de la prueba; en D-09 **no**: el umbral 0,20 elegido sobre
   validación resulta permisivo para señantes nuevos (§5.5), y esa divergencia está documentada en
   N-33.
6. **Dos decisiones se tomaron con diferencias pequeñas frente a la dispersión.** En D-07, «mascara»
   ganó a «descartar» por 1,4 puntos de validación, mientras en prueba las dos son indistinguibles en
   top-1 (0,533 frente a 0,537, con DE de 0,13–0,18) y «descartar» es mejor en top-3. En D-08, C1 y
   C2 quedaron a 0,8 puntos de validación —dentro del margen de empate de la propia regla—. El
   criterio estaba fijado de antemano en ambos casos y no se cambió; pero la conclusión «C1 es mejor
   que C2» se apoya en el LOSO (4,6 puntos de top-1, y una DE que baja de 0,176 a 0,086), no en la
   validación.
7. **El modelo canónico no es el modelo del LOSO.** El LOSO entrena 5 modelos; el canónico es uno
   distinto, entrenado sobre la partición aleatoria y elegido entre 5 semillas por validación. Las
   cifras de §2.1 (LOSO) y §2.4 (canónico) no describen el mismo objeto.
8. **El umbral de operación se seleccionó sobre validación y se está usando en el prototipo.** El
   efecto medido sobre señantes nuevos (35,5 % de error entre las aceptadas) es un dato reportado, no
   un criterio de cambio: cambiarlo mirando la prueba sería reajustar sobre el conjunto de prueba
   (N-33).

### 8.3 De las comparaciones

9. **Las dos columnas de §2.1 no son comparables** por las seis razones de §2.3. La resta de 30
   puntos es descriptiva.
10. **Comparaciones múltiples.** El contraste por seña de §5.6 son 50 pruebas sin corrección, y el
    propio archivo lo declara. En el análisis de longitud y fotogramas sin mano sí se aplicó Holm
    sobre 14 pruebas, y **la mayoría de las asociaciones no sobrevive**.
11. **La línea base recibe una entrada deliberadamente empobrecida.** Ocho estadísticos por columna
    borran el orden temporal salvo lo que capturan el primer paso, el último y la media de las
    diferencias; es la comparación que pide el anteproyecto (modelo temporal frente a rasgos
    agregados), pero no permite concluir nada general sobre clasificadores clásicos (§4.6).
12. **La comparación con la literatura no es homogénea en ninguna de sus filas** (§6.2), y de tres de
    los seis trabajos de LSC los `.tex` no consignan el esquema de partición (§6.3).

### 8.4 De la instrumentación y de los datos

13. **Integridad del corpus verificada, con su límite.** N-43 cotejó los 8.000 archivos uno por uno
    (4.000 LANDMARKS y 4.000 VIDEOS) contra el CRC-32 y el tamaño del índice del depósito: 8.000 de
    8.000 coinciden, cero ausentes, cero sobras. **Límite declarado**: el CRC-32 son 32 bits, detecta
    corrupción accidental pero no es garantía frente a manipulación deliberada; y el cotejo es contra
    la versión que la API devuelve hoy.
14. **El extractor propio no es numéricamente idéntico al de los CSV publicados** (N-22: la
    convención de fotogramas sin mano y la coordenada z de la postura difieren; D-05 declara además
    que la salida de `HolisticLandmarker` no es numéricamente idéntica a la de las tres tareas
    sueltas de D-03/D-04), y el cambio de fuente de rasgos movió la cifra 10 puntos antes de las
    correcciones de D-07 (§3.1).
15. **Coordenada z de la postura**: el diagnóstico de N-22 midió que `HolisticLandmarker` estima la
    profundidad de la postura de otra manera que los CSV publicados (|dz| 0,07, correlación 0,915);
    se anotó como efecto secundario posible y **no cuantificado**.
16. **Umbrales por defecto del extractor sin ajustar** (`min_*_confidence`), declarado en D-05.
17. **Windows no verificado**; todo lo medido es macOS 26.6.2, Apple M5 Max, 18 núcleos, 48 GB, CPU
    sin GPU (N-21; y `resultados_loso.json`, clave `entorno`: `gpus_visibles` = `[]`, Python 3.12.13,
    TensorFlow 2.21.0, Keras 3.13.2, scikit-learn 1.8.0, numpy 2.4.3).

---

## 9. Dos cifras que se prestan a confusión, y cuál usar

Este apartado existe porque el propio registro pidió resolverlo antes de escribir la tesis.

1. **F1 macro del LOSO: 0,537 o 0,574.** Las dos son correctas y miden cosas distintas.
   - **0,5374** es la **media de los F1 macro de los cinco pliegues**
     (`resultados_loso.json`, `agregado.f1_macro.media`). Es la que reporta N-30 y la que va
     acompañada de su desviación entre pliegues (0,0927).
   - **0,5737** es el **F1 macro calculado sobre las 997 predicciones juntas**
     (`analisis.json`, `por_sena.f1_macro_agregado`; y
     `models/linea_base_4d_2026-09-15/resultados.json`, `red.metricas_agregadas.f1_macro`). Es la que
     usa la tabla de comparación con la línea base, porque la línea base se compara sobre el
     agregado.
   - N-35 ya lo señaló: «OJO: N-30 reporta la media del F1 por pliegue (0,537); en la tesis hay que
     decir cuál se usa». **Propuesta de este borrador**: usar la media entre pliegues con su
     desviación cuando se hable del esquema LOSO como tal (es lo que pide `09_metodologia.tex:143-145`,
     «se reportan la media y la dispersión entre pliegues»), y el agregado solo donde haga falta
     comparar predicción a predicción, declarándolo en la misma línea. **No mezclarlas en una misma
     tabla sin rótulo.**
2. **Precision macro del LOSO: 0,5574 o 0,5936.** Misma situación y misma causa: 0,5574 es la media
   entre pliegues (`resultados_loso.json`, `agregado.precision_macro.media`) y 0,5936 el agregado
   sobre 997 (`red.metricas_agregadas.precision_macro`). La tabla §1 de `comparacion.md` usa el
   agregado, y lo rotula como tal. **Recall y top-1 no tienen este problema**: top-1 agregado
   (0,57974) y media entre pliegues (0,57974) coinciden hasta la cuarta cifra porque los pliegues
   tienen casi el mismo tamaño (198–200).

3. **«C1 supera a C2 en 4,7 puntos» (N-30) frente a 4,63 puntos en el JSON.** N-30 escribe 4,7
   porque resta las cifras ya redondeadas a tres decimales (0,580 − 0,533 = 0,047); desde
   `comparacion_D08_2026-09-15.json` con precisión completa la diferencia es
   0,5797447 − 0,5334470 = 0,0462977, es decir **4,63 puntos**. La diferencia de top-3 sí coincide
   (0,7702893 − 0,6940319 = 0,0762574 → 7,6 puntos, como dice N-30). Este borrador usa 4,6 y deja
   señalada la discrepancia en lugar de elegir en silencio: es de redondeo, no de fondo, pero
   conviene fijar una convención de redondeo para la tesis antes de escribir el capítulo.

**No se ha encontrado ninguna discrepancia de fondo entre el registro y los JSON** en las cifras
revisadas para este documento: los valores de N-17, N-22, N-23, N-25, N-30, N-31, N-33, N-35 y N-39
que se citan aquí coinciden con `comparacion_D07_2026-09-14.json`, `comparacion_D08_2026-09-15.json`,
`resultados_loso.json`, `canonico_holistic_mascara_manos/resultados.json`,
`analisis_errores_manos_2026-09-15/analisis.json` y `linea_base_4d_2026-09-15/resultados.json`
respectivamente. Los casos 1 y 2 de §9 **no son discrepancias**: son dos definiciones distintas,
ambas declaradas en su archivo de origen. El caso 3 sí es una diferencia de redondeo entre el texto
del registro y el JSON, y queda señalada sin resolverla por cuenta propia. La revisión abarcó las
cifras citadas en este documento y no todo el registro.

---

## Apéndice · Índice de fuentes usadas

**Anteproyecto** (`Anteproyecto/Anteproyecto_FINAL_LaTeX/`, solo lectura):
`06_alcances.tex`; `08_marco.tex` (§2.1.2 tabla de variables, «Sobre la comparabilidad de las
precisiones» y Tabla de trabajos relacionados); `09_metodologia.tex` (§2.2.2 preparación del conjunto,
§2.2.3 representación, §2.2.4 modelo y línea base, §2.2.6 casos límite, §2.2.7 protocolo de
evaluación, §2.2.8 actividades 4.a–4.e); `10_resultados.tex` (RE1, RE2, RE3, RE5 y RE6).

**Registro de decisiones** (`DECISIONES_TESIS.txt`): D-05, D-06, D-07, D-08, D-09, D-10; N-08, N-17,
N-21, N-22, N-23, N-25, N-26, N-28, N-30, N-31, N-32, N-33, N-34, N-35, N-36, N-37, N-39, N-40, N-41,
N-43.

**Archivos de resultados del repositorio:**

| Ruta | Qué aporta |
|---|---|
| `models/loso_2026-09-15_holistic_mascara_manos/resultados_loso.json` | LOSO y esquema aleatorio del modelo «manos»; configuración y entorno |
| `models/comparacion_D08_2026-09-15.json` | Los tres candidatos de representación y la regla aplicada |
| `models/comparacion_D07_2026-09-14.json` | Las tres reglas de mano ausente |
| `models/canonico_holistic_mascara_manos/resultados.json` | Modelo canónico: 5 semillas, selección y prueba |
| `models/canonico_holistic_mascara_manos/modelo_canonico.procedencia.json` | Representación, normalización y regla del modelo del prototipo |
| `models/canonico_holistic_mascara_manos/modelo_canonico.umbral.json` | Umbral 0,20 (D-09) y SHA256 del modelo |
| `models/analisis_errores_manos_2026-09-15/analisis.json` (y `.md`) | Análisis de errores 2.d sobre las 997 predicciones |
| `models/linea_base_4d_2026-09-15/resultados.json` y `comparacion.md` | Línea base clásica, McNemar y controles |
| `models/linea_base_4d_2026-09-15/control_aumentacion_completa/resultados.json` | Control con la aumentación de la red |
| `models/tasa_extractor_local_2026-09-16/tasa.json` | Costo del extractor por fotograma, Drive frente a disco local |
| `models/integridad_corpus_2026-09-16/` | Cotejo de integridad archivo por archivo (vía N-43) |
| `docs/PROTOCOLO_EXPERIMENTAL_4a.md` | Estado del protocolo de la evaluación en línea |
