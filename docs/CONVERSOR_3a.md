# Procedimiento de conversión de glosas en enunciados en español (actividad 3.a, RE4)

**Fecha:** 8 de noviembre de 2026
**Versión del procedimiento:** `3a-2` (`src/conversor_glosas.py`, clase `ConversorGlosas`)
**Estado de la verificación:** la automática está hecha (§7). La humana (Actividad B) **no se ha
hecho**: ninguna regla, plantilla ni frase de este documento está revisada por la persona señante
vinculada (profesor de LSC, actividad 0.c).
**Documento normativo:** `docs/CONVERSOR_3a_ESPECIFICACION.md` («la especificación»). Este documento
la resume para la tesis. Si los dos difieren, manda la especificación.

**De dónde salen las cifras.** Si no se indica otra fuente, las cifras del conversor nuevo se
midieron el 8-nov con la implementación 3a-2 y con `python`.
Las del conversor heredado vienen de N-42 (`docs/estado_mapeo_gramatical.md`) y se volvieron a medir
el mismo día con `GrammarMapper`: dieron lo mismo.

---

## 1. Qué pide el anteproyecto

- **Actividad 3.a** (09_metodologia.tex): «Definición del procedimiento de conversión de glosas en
  enunciados en español: especificación de las reglas de conversión sobre el vocabulario soportado,
  tratamiento de las categorías que el español exige y la glosa no trae, documentación de los casos
  no cubiertos y verificación de los enunciados producidos con la persona señante o intérprete
  vinculada».
- **Método** (09_metodologia.tex, «Mapeo de Glosas a Español y Síntesis de Voz»): «La conversión de
  la glosa reconocida a español se hará mediante reglas explícitas y plantillas predefinidas sobre el
  vocabulario soportado: a cada glosa o a cada secuencia corta prevista se le asociará una palabra o
  un enunciado fijo en español».
- **RE4** (04_objetivos.tex): «Procedimiento documentado de conversión de glosas en enunciados en
  español, con sus reglas explícitas y sus límites».
- **Alcance** (06_alcances.tex): queda fuera «el mapeo gramatical automático abierto entre la LSC y
  el español: la conversión a español se limita a reglas y plantillas sobre el vocabulario
  soportado».

En consecuencia, el procedimiento no traduce de forma abierta y no usa modelos de lenguaje, redes,
estadística ni corpus paralelos. Son tablas y condiciones escritas a mano.

---

## 2. El problema que resuelve

### 2.1 La glosa es una etiqueta léxica

El reconocedor entrega glosas: las 50 etiquetas de LSC50 («tío», «juicioso», «¿dónde?»). Una glosa
no es una palabra del español y le faltan las categorías que el español exige. «La etiqueta con la
cual se glosa en español tiene un valor bastante limitado» (Oviedo, en INSOR 2006, p. xxix), y las
entradas del Diccionario básico «corresponden a señas, no a palabras del español» (INSOR 2006,
p. xxii). Para Auslan, el ID-gloss «is not intended as a translation» (Johnston 2008, p. 84). Una
secuencia lineal de glosas «is not an adequate representation of a signed utterance» (Müller et al.
2023, p. 683).

Lo que falta se ordena con los tres grupos de Moryossef et al. (2021, p. 4), que describen ASL y DGS
y aquí se usan solo como clasificación: **flexión** (persona, número, género, tiempo), **palabras
omitidas** (artículo, cópula, preposiciones) y **puntuación**. Para la LSC, Naranjo Orozco y Dieck
(2022) documentan dos de esas ausencias: «En la LSC no se expresa léxicamente la cópula» (p. 66,
nota 12), y los verbos «en sí mismos no contienen información temporal» (p. 61).

La justificación **no** se apoya en contrastar el orden de constituyentes de la LSC y del español.
La fuente disponible (ref13, Oviedo 2001, citada por el anteproyecto y no verificada en esta
actividad) describe el orden S-V-OD-OI en elicitación y un orden «muy flexible» en discurso, así que
la literatura no permite fijar un orden de la LSC.

### 2.2 El conversor heredado

El prototipo usaba `src/grammar_mapper.py` (`GrammarMapper`), descrito en N-42. Ese módulo no se
modificó (md5 `ce3ac6c8f7b841bbd02118eca0f2d320`, comprobado el 8-nov), y sus 89 pruebas fijan su
salida actual, incluidas las salidas agramaticales. Lo medido:

| hallazgo | cifra | ejemplo |
|---|---|---|
| pares sujeto × adjetivo con cópula en la salida | 0 de 91 | TÍO + JUICIOSO → «El tío juicioso» |
| pares sujeto × verbo con desacuerdo de número | 12 de 52 | USTEDES + COMER → «Ustedes come» |
| adjetivos que declaran género o número | 0 de 7 | NIÑA + CONTENTO → «La niña contento» |
| glosas solas que terminan en punto | 0 de 50 | «Gracias» |
| pares de glosas distintas sensibles al orden de entrada | 188 de 1.225 (15,3 %) | 158 dentro de una misma categoría y 30 entre saludo e interrogativo |
| tercias sensibles al orden de entrada | 6.669 de 19.600 (34,0 %) | — |
| plantillas declaradas / alcanzables | 7 / 5 | — |
| categorías usadas / nombradas en el código | 8 / 12 | — |

Además, «la hambre» (H4), glosas que se pierden en silencio (H5, H13) y un «¿» interno sin mayúscula
(H9). Todo eso está en `docs/estado_mapeo_gramatical.md`.

---

## 3. El procedimiento

### 3.1 Entrada, salida y cuatro vías

El conversor recibe la lista de glosas de un turno (o una sola glosa en modo aislado) y devuelve
**un** texto en español por una de cuatro vías (especificación, §1.3):

| vía | cuándo | qué sale |
|---|---|---|
| `lexico` | queda una sola glosa distinta | su forma de cita: «Comer.», «Hambre.», «¿Dónde?» |
| `plantilla` | el conjunto de glosas es la clave exacta de una de las 22 plantillas | el enunciado fijo de la plantilla |
| `reglas` | el conjunto encaja en un patrón admitido | un enunciado armado por reglas explícitas |
| `respaldo` | ninguna de las anteriores | las glosas en minúscula, separadas por comas, sin punto y marcadas como respaldo, con un motivo |

Sin ninguna glosa del vocabulario, la salida es `""` y no se dice nada.

Cuatro principios fijan el diseño:

1. **El orden de entrada no decide la frase** (R-04). La frase depende solo del conjunto de glosas
   distintas, y el orden de salida lo fija cada regla o plantilla. El Diccionario básico pone el
   interrogativo al final, al principio o en los dos sitios (INSOR 2006, pp. 480, 482 y 488), así
   que el orden no sirve como señal. El respaldo sí conserva el orden, porque es una transcripción.
2. **Ninguna glosa se pierde en silencio.** Si una regla no puede usar todas las glosas, la salida
   es un respaldo con su motivo (R-14, R-35).
3. **Nunca se presenta como frase algo que ninguna regla produjo.** Lo que no tiene regla va al
   respaldo (especificación, §5.1).
4. **Todo lo que se suple queda registrado** en la traza `explicar()`: qué categoría se añadió
   (`suplido`) y qué valor se eligió sin que la glosa lo diera (`supuestos`) (§3.3).

### 3.2 Reglas

Hay 35 reglas, de R-01 a R-35. Cada una tiene condición, acción, ejemplo, fundamento y la marca
«validar» (especificación, §3.3). En 26 la marca es **sí**: son decisiones de diseño del autor que
se le muestran a la persona señante. Las otras 9 son de forma o de código. El orden de comprobación
es normativo (especificación, §3.1, etapas A a F), porque decide qué motivo de respaldo se registra.

| etapa | reglas | qué hace |
|---|---|---|
| A. Entrada | R-01 a R-08 | normaliza cada elemento (ids, tildes, espacios, 13 alias); manda a respaldo los elementos fuera de vocabulario (R-02); colapsa las repeticiones (R-03); ignora el orden (R-04); da la forma de cita de la glosa sola (R-06); busca una plantilla (R-07); admite como máximo 4 glosas distintas (R-08) |
| B. Periferia | R-09 a R-13 | saludo inicial con coma (R-09); «Saludo a todos.» (R-10); vocativo sin artículo: «Hola, abuelo.», «Gracias, abuelo.» (R-11); «Hola, bienvenido.» y «Hola, ¿cómo estás?» (R-12); la cortesía solo entra por plantilla (R-13) |
| C. Núcleo declarativo | R-14 a R-28 | un sujeto, y a lo sumo un elemento de cada clase (R-14); artículo definido (R-15); pronombre sujeto explícito (R-16); persona, número y género (R-17); concordancia del verbo (R-18); presente de indicativo (R-19); cópula *ser* o *estar* según el adjetivo (R-20); concordancia del adjetivo (R-21); BIEN, MAL y MÁS O MENOS con «estar» o como modo del verbo (R-22); SENTIR como «sentirse» (R-23); «tener hambre» (R-24); MUCHO y POCO (R-25); NUNCA antes del verbo (R-26); VIVIR exige complemento (R-27); SEÑA y NOMBRE solo por plantilla (R-28) |
| D. Pregunta | R-29 a R-32 | solo hay pregunta si hay una glosa interrogativa, y una sola (R-29); pregunta parcial con el sujeto al final y una tabla cerrada de combinaciones (R-30); ¿QUIÉN? (R-31); ¿CÓMO ESTÁS? + sujeto (R-32) |
| E. Salida | R-33 a R-35 | mayúscula inicial (R-33); punto o «?», sin exclamación (R-34); forma del respaldo (R-35) |
| F. Modo estricto | — | con `estricto=True`, todo lo que no sea una glosa sola ni una de las 92 secuencias del inventario va a respaldo con el motivo `fuera_de_inventario` |

El léxico tiene 15 categorías, frente a las 8 que usa el heredado (especificación, §2.1). El
criterio es uno solo: una categoría reúne las glosas que cumplen la misma función en las reglas.
Las diferencias dentro de una categoría (por ejemplo, qué cópula lleva cada adjetivo) se declaran
como rasgos en la tabla del léxico (§2.3), no como excepciones en el código. Las formas verbales y
adjetivas están en tablas cerradas (§2.4); no se derivan por terminación.

**La glosa sola** (decisión D-L1, §2.2): forma de cita con mayúscula y punto, sin artículo y sin
conjugar («Comer.», no «Come», que además suena a orden); las preguntas van entre ¿? y no se usan
signos de exclamación. Las 50 salidas solas son distintas entre sí: 43 terminan en «.» y 7 en «?».

### 3.3 Qué categorías suple, y cómo queda registrado

Cada salida por reglas registra en `suplido` qué categoría se añadió y en `supuestos` qué valor se
eligió sin que la glosa lo diera (especificación, §3.2 y §6). Recuento en los 6.995 conjuntos de 2
a 4 glosas que dan frase:

| lo que el español exige | regla | conjuntos que lo suplen | supuesto que se registra |
|---|---|---|---|
| persona y número del verbo | R-17, R-18 | 6.838 | `tratamiento_tu` (528; incluye las plantillas con TÚ) |
| tiempo verbal | R-19 | 6.838 | `tiempo_presente` (6.838) |
| artículo | R-15 | 4.597 | `articulo_definido` (4.597) |
| cópula *ser*/*estar* | R-20, R-22 | 3.088 | — |
| género y número del adjetivo | R-21 | 2.776 | `genero_por_defecto` (212: YO o TÚ sin género configurado, o el destinatario de BIENVENIDO tras un saludo) |
| pronombre «me/te/se» | R-23 | 1.023 | — |
| «muy» | R-25 | 937 | — |
| verbo de soporte «tener» | R-24 | 420 | — |
| preposición «a» | R-10 | 8 | — |
| puntuación | R-09, R-30, R-34 | 6.973 | — |
| lectura elegida entre dos | R-32, R-20 | — | `lectura_tercera_persona` (48), `lectura_vocativa` (48) |

En el modo estricto (las 92 secuencias del inventario), los supuestos aparecen así:
`tiempo_presente` en 53, `articulo_definido` en 21, `tratamiento_tu` en 13, `genero_por_defecto`
en 4, y `lectura_vocativa` y `lectura_tercera_persona` en 1 cada uno.

---

## 4. Plantillas e inventario de secuencias previstas

### 4.1 Plantillas (22)

Una plantilla es un enunciado fijo para un conjunto exacto de glosas, y tiene precedencia sobre las
reglas (R-07). Cubre sobre todo la cortesía, que no se compone libremente (R-13). Las eligió el
autor y no tienen fuente de uso. T-18 a T-22 se añadieron en 3a-2 por simetría con plantillas que ya
existían. Reiter (1995) señala que el mantenimiento de las plantillas solo es un problema si son
muchas (p. 4, nota 1), y que un sistema de plantillas puede manejar concordancias sencillas (p. 6).

| id | glosas | enunciado fijo |
|---|---|---|
| T-01 | POR FAVOR + SEÑA | La seña, por favor. |
| T-02 | POR FAVOR + NOMBRE | El nombre, por favor. |
| T-03 | TÚ + NOMBRE + ¿QUÉ? | ¿Cuál es tu nombre? (admite saludo delante) |
| T-04 | TÚ + SEÑA + ¿QUÉ? | ¿Cuál es tu seña? (admite saludo delante) |
| T-05 | GRACIAS + MUCHO | Muchas gracias. |
| T-06 | CON GUSTO + MUCHO | Con mucho gusto. |
| T-07 | GRACIAS + TODOS | Gracias a todos. |
| T-08 | PERDÓN + PERMISO | Perdón, con permiso. |
| T-09 | POR FAVOR + PERMISO | Permiso, por favor. |
| T-10 | PERDÓN + ¿QUÉ? | Perdón, ¿qué? |
| T-11 | PERDÓN + ¿CÓMO? | Perdón, ¿cómo? |
| T-12 | BIEN + GRACIAS | Bien, gracias. |
| T-13 | MÁS O MENOS + GRACIAS | Más o menos, gracias. |
| T-14 | YO + BIEN + GRACIAS | Yo estoy bien, gracias. |
| T-15 | GRACIAS + ADIÓS | Gracias, adiós. |
| T-16 | GRACIAS + BUENAS NOCHES | Gracias, buenas noches. |
| T-17 | MUCHO + GRACIAS + TODOS | Muchas gracias a todos. |
| T-18 | GRACIAS + BUENOS DÍAS | Gracias, buenos días. |
| T-19 | GRACIAS + BUENAS TARDES | Gracias, buenas tardes. |
| T-20 | GRACIAS + USTEDES | Gracias a ustedes. |
| T-21 | GRACIAS + TÚ | Gracias a ti. |
| T-22 | MAL + GRACIAS | Mal, gracias. |

Las 22 se alcanzan; en el heredado se alcanzaban 5 de 7. En los 251.125 conjuntos de 2 a 4 glosas,
30 conjuntos salen por plantilla: 18 de 2 glosas, 4 de 3 y 8 de 4 (las de 3 y 4 incluyen T-03 y T-04
con un saludo delante).

### 4.2 Inventario de 92 secuencias previstas

El inventario (especificación, §4.3; constante `INVENTARIO` del módulo) asocia a cada «secuencia
corta prevista» su enunciado fijo, que es lo que pide el método del anteproyecto. Se formó en este
orden: (1) las 9 frases de ejemplo ya usadas con el autor (S-01 a S-09); (2) las fórmulas del
Diccionario básico que el léxico permite formar (INSOR 2006, pp. 485-490); (3) las combinaciones de
cortesía de las plantillas; (4) una secuencia por cada rama de las reglas que produce frase, para
que cada rama quede fijada por una salida exacta. 23 secuencias salen por plantilla y 69 por reglas.
Al importar el módulo se comprueba que las 92 dan su enunciado fijo; si alguna no lo da, el módulo
no carga.

### 4.3 Dos modos: qué se presenta como procedimiento

- **`inventario` (modo estricto):** solo dice las 50 glosas solas y las 92 secuencias del
  inventario, cada una con su enunciado fijo. Todo lo demás va a respaldo. Medido: en los 251.125
  conjuntos de 2 a 4 glosas, 92 dan frase y 6.903 que por reglas la tendrían salen con
  `fuera_de_inventario`. Es el modo que se ajusta al método del anteproyecto y el que se presenta
  en la tesis como **el procedimiento** (D-15).
- **`reglas`:** dice además las frases que arman las reglas fuera del inventario. Es una extensión
  cerrada y enumerable: 7.045 salidas posibles con frase (50 glosas solas + 6.995 conjuntos), todas
  listadas en `docs/CONVERSOR_3a_SALIDAS_NO_RESPALDO.csv`. No están verificadas una por una.

**Decisión del autor (8-nov-2026, D-15):** el procedimiento de la tesis es `inventario`; `reglas`
se reporta como extensión, sin verificar. Queda por informar al director. El prototipo sigue usando
por defecto el conversor heredado; para usar el procedimiento se pasa `--conversor inventario`.

---

## 5. Casos que cubre y casos que no cubre

### 5.1 Cobertura medida

| conjunto de entradas | con frase | por plantilla | por reglas | respaldo |
|---|---|---|---|---|
| 50 glosas solas | 50 (vía `lexico`) | — | — | 0 |
| 2.450 pares ordenados de glosas distintas | 570 (23,3 %) | 36 | 534 | 1.880 (76,7 %) |
| 251.125 conjuntos de 2 a 4 glosas distintas | 6.995 (2,8 %) | 30 | 6.965 | 244.130 |
| las mismas, en modo estricto | 92 | 23 secuencias del inventario | 69 secuencias del inventario | el resto |

Respaldo en los 2.450 pares, por motivo (17 motivos): cortesía fuera de plantilla 364, sin sujeto
306, saludo con glosa no admitida 282, pregunta sin sujeto 180, sustantivo abstracto 170, dos
sujetos 156, sin predicado 108, pregunta no admitida 80, despedida con contenido 74, dos
interrogativos 42, ¿cómo estás? no admitido 38, vivir sin complemento 28, sentir sin predicativo 28,
saludos no combinables 14, ¿quién? con sujeto no admitido 6, yo + bienvenido 2, todos con nunca 2. En
los conjuntos de 2 a 4 glosas aparecen 32 motivos distintos; el código defensivo `no_previsto` no
aparece nunca. La lista completa de los 35 motivos posibles está en la especificación, §5.2.

El respaldo alto es el precio de no inventar. No se midió cuántos respaldos son combinaciones sin
enunciado posible (GRACIAS + COMER) y cuántos son enunciados naturales que se dejaron fuera por
decisión de diseño. Entre estos últimos están el sujeto nulo (COMER + MUCHO, «Come mucho»), saludo
+ pronombre (HOLA + TÚ), saludo + cortesía (HOLA + GRACIAS), PERDÓN + vocativo y YO + SENTIR («Lo
siento»). Esos casos se le preguntan a la persona señante (Actividad B, bloque D).

### 5.2 Límites declarados

La especificación (§8) enumera 21 límites, L-01 a L-21. Los principales:

| límite | qué hace el conversor | consecuencia |
|---|---|---|
| L-13 errores del reconocedor | no los detecta | **una glosa equivocada puede dar una frase gramatical y falsa**, más creíble que un respaldo. Es el riesgo principal |
| L-14 negación con la cabeza o el rostro (Zeshan 2006, p. 42, en otras lenguas de señas; el léxico no tiene NO) | sale la afirmación | invierte el sentido |
| L-01 preguntas sí/no marcadas sin manos (Zeshan 2006, p. 40, en una muestra sin la LSC) | sale la afirmación | una pregunta se oye como afirmación |
| L-02 tiempo verbal | todo en presente (R-19) | un hecho pasado o futuro se dice en presente |
| L-03 plural por repetición en el espacio (Oviedo, en INSOR 2006, p. xxxix) | la repetición se colapsa (R-03) | «Niño.» en lugar de «niños» |
| L-04 espacio, concordancia espacial, clasificadores | no se ve | persona y referentes salen solo de las glosas de pronombre |
| L-05, L-06 sujeto nulo; coordinación, dos verbos o dos sujetos | respaldo | «El hermano y el tío comen» no se produce |
| L-07 posesivos | artículo definido | «el tío», aunque se quisiera decir «mi tío» |
| L-08 género de quien habla y de quien escucha | masculino por defecto, configurable | «Yo estoy contento», lo diga quien lo diga, si no se configura |
| L-09 tratamiento | *tú* | puede no ser el registro esperado (*usted*, *vos*) |
| L-12 polisemia de la glosa (SEÑA, PERMISO, BIEN) | una lectura por glosa | puede elegir la lectura que no era |
| L-17 prosodia de la voz con la puntuación nueva | no medida | queda para 3.b |
| L-21 salidas por reglas sin revisión individual | con `reglas`, se dicen todas | una frase rara pero gramatical («¿Quién es bienvenido?») puede decirse sin que nadie la revisara |

---

## 6. Comparación heredado → nuevo

### 6.1 Cifras

| medida | heredado (N-42, medido de nuevo el 8-nov) | nuevo 3a-2 |
|---|---|---|
| pares sujeto × adjetivo (13 × 7 = 91) con cópula | 0 de 91 | 84 de 91. Los otros 7: 6 de vocativo + BIENVENIDO salen como fórmula sin cópula («Bienvenida, familia.», R-20) y YO + BIENVENIDO va a respaldo (`yo_bienvenido`) |
| pares sujeto × verbo (13 × 4 = 52) con desacuerdo de número | 12 de 52 | 0. 26 dan frase, todos concordados («Ustedes comen.», «Las personas trabajan.»); los otros 26 van a respaldo: 13 con VIVIR sin complemento y 13 con SENTIR sin predicativo |
| adjetivos con formas de género y número | 0 de 7 | 7 de 7 (tabla de formas de la especificación, §2.4); el barrido 13 × 7 de la prueba 5 comprueba la concordancia |
| glosas solas que terminan en punto o «?» | 0 de 50 | 50 de 50 (43 «.» y 7 «?»), todas distintas |
| pares de glosas distintas cuya frase cambia al invertir el orden | 188 de 1.225 | 0 (los pares que van a respaldo tampoco cambian de estado al invertirlos; el respaldo conserva el orden de entrada por diseño, R-35) |
| tercias cuya frase cambia con el orden | 6.669 de 19.600 | 0 de 19.600 (las 6 permutaciones de cada tercia dan la misma frase o todas van a respaldo) |
| permutaciones de los conjuntos de 3 y 4 glosas con frase | — | 129.414, ninguna con frase distinta (prueba 4) |
| plantillas declaradas / alcanzables | 7 / 5 | 22 / 22 |
| categorías | 8 | 15 |
| glosas fuera de vocabulario | se ignoran en silencio | respaldo con motivo `fuera_de_vocabulario` (R-02) |
| tiempo medio por llamada (2.450 pares; el mínimo de 3 llamadas por par) | 1,45 µs (especificación, §5.1) | 4,26 µs; el peor par, 17,8 µs (requisito: menos de 1 ms) |

### 6.2 Las 9 frases de ejemplo

Medido el 8-nov con `GrammarMapper` y con `ConversorGlosas`, en las dos variantes (`reglas` e
`inventario`) y en todas las permutaciones de cada secuencia.

| # | glosas | heredado | nuevo (igual en `reglas` e `inventario`) | qué cambia |
|---|---|---|---|---|
| 1 | BUENOS DÍAS + TODOS | Buenos días todos | Buenos días a todos. | «a» de destinatario y punto (R-10, R-34) |
| 2 | BUENAS TARDES + TODOS | Buenas tardes todos | Buenas tardes a todos. | ídem |
| 3 | TÍO + JUICIOSO | El tío juicioso | El tío es juicioso. | cópula *ser* (R-20) |
| 4 | FAMILIA + FELIZ | La familia feliz | La familia está feliz. | cópula *estar*; colectivo en singular (R-17, R-20) |
| 5 | POR FAVOR + SEÑA | La seña por favor | La seña, por favor. | coma y punto (T-01) |
| 6 | HOLA + BIENVENIDO | Hola bienvenido | Hola, bienvenido. | coma tras el saludo (R-09, R-12) |
| 7 | HERMANO + TRABAJAR | El hermano trabaja | El hermano trabaja. | solo el punto |
| 8 | ¿DÓNDE? + HERMANO + VIVIR | ¿Dónde vive el hermano? | ¿Dónde vive el hermano? | igual; ahora por una regla declarada (R-30) |
| 9 | HOLA + ¿CÓMO ESTÁS? | Hola, ¿cómo estás? | Hola, ¿cómo estás? | igual |

Siete de las nueve cambian y dos quedan iguales. En las nueve, el heredado también da la misma frase
en todas las permutaciones; la sensibilidad al orden del heredado aparece en otros pares (§2.2).

---

## 7. Verificación automática

### 7.1 Pruebas

Ejecutadas el 8-nov con venv312:

| suite | resultado |
|---|---|
| `tests.test_conversor_glosas` (nueva) | 59 pruebas, OK |
| `tests.test_grammar_mapper` (heredado, sin cambios) | 89 pruebas, OK |
| `tests.test_prototipo` | 72 pruebas, OK. Incluye 8 nuevas de integración (`TestConversorGlosasEnElPrototipo`) |
| `tests.test_tts` | 86 pruebas, OK |
| suite completa (`unittest discover -s tests`) | 735 pruebas, **FAILED (errors=1)**. El único error es ajeno al conversor: `tests/test_arquitecturas_2.py` hace `import pytest`, y pytest no está instalado en venv312. Ese archivo no cambió desde 64fc9dc |

Qué comprueban las pruebas del conversor (especificación, §9.1):

1. Las 50 glosas solas, exactas; una glosa da lo mismo como texto, como id o repetida.
2. Las 92 secuencias del inventario dan su enunciado fijo, en su orden y en el inverso, con su vía y
   su plantilla.
3. **Los 2.450 pares ordenados, uno por uno:** ninguna excepción y ninguna salida vacía; la frase no
   depende del orden; mayúscula inicial; cierre en «.» o «?»; tantos «¿» como «?»; sin dobles
   espacios, « ,», «..» ni «.?»; el respaldo es exactamente la lista de glosas; los recuentos por vía
   (36 / 534 / 1.880) y por motivo (17 motivos), código por código.
4. **Los 19.600 conjuntos de 3 y los 230.300 de 4, uno por uno:** las mismas propiedades de forma,
   6.995 conjuntos con frase, los 32 motivos con su recuento exacto, nunca `no_previsto`, y las
   129.414 permutaciones de los conjuntos con frase.
5. Concordancia por barrido: 13 sujetos × verbos, × 7 adjetivos y × SENTIR + adjetivo; ninguna
   salida contiene «la hambre», «Personas come» ni un sujeto plural con el verbo en singular.
6. Bordes: `None`, ids fuera de rango, `True`, `numpy.int64`, `numpy.bool_`, cadenas sueltas,
   `bytes`, `set`, iterables sin fin (tope de 10.000 elementos), `__repr__` que lanza excepción y
   alias sin tildes. Modo estricto y constructor.
7. Determinismo: dos pasadas en orden inverso y el barrido repetido con otro `PYTHONHASHSEED`.
8. Rendimiento: el peor de los 2.450 pares tarda menos de 1 ms.
9. No regresión: md5 del heredado, `LSCTranslator()` sin argumentos sigue con `GrammarMapper`, y
   con el heredado el registro no gana claves nuevas.
10. Red de seguridad: mutaciones de una tabla a la vez (cópula, género, forma verbal, plantilla,
    orden del sujeto en R-30); cada mutación hace fallar alguna prueba.
11. Precedencia: una entrada por cada par de comprobaciones de §3.1 que pueden chocar con 3 o 4
    glosas, en todas sus permutaciones.

Mutaciones externas. La verificación de 3a-1 aplicó 12 mutaciones a una copia del módulo, fuera
del repositorio; la suite detectó 7. Según la especificación (§9.1, prueba 11), en 3a-2 se
añadieron pruebas y las cinco que sobrevivían (orden de P2/P3, las dos comprobaciones de D2, C4
antes que C3, TÚ sin `genero_por_defecto` y «¿Quién eres tú?» sin `copula`) ahora fallan. Esa
comprobación no se repitió al redactar este documento.

### 7.2 Gramaticalidad y fidelidad

**Gramaticalidad** (norma RAE/ASALE). Una verificación independiente leyó las salidas de la versión
3a-1 y las juzgó gramaticales: 50 de 50 solas, 17 de 17 plantillas, 548 de 548 pares que no van a
respaldo (274 conjuntos, todas las cadenas distintas) y 82 de 82 tríos sin respaldo de la muestra
del CSV. En total, el 100 % de lo revisado. En una muestra al azar de 150 pares (`random.seed(2026)`),
los 128 respaldos tenían un motivo coherente con la entrada. La verificación comprobó como correctas
algunas concordancias delicadas: «mucha/poca hambre», «La familia está contenta» (colectivo en
singular), «¿Quiénes son ustedes?» y «¿Cuál es tu nombre?».

En 3a-2 hay 570 pares que no van a respaldo (285 conjuntos). La verificación de gramaticalidad no
se repitió entera. Según la especificación, ninguna frase de 3a-1 cambió de texto salvo las 48 que
llevan `lectura_vocativa`. Las 17 de 2 glosas que son nuevas o cambiaron (T-18 a T-22, «Gracias,
x.» con 6 vocativos y «Bienvenido, x.» con 6), y las 42 de saludo + vocativo + BIENVENIDO, se
leyeron al redactar este documento y son gramaticales. Es el juicio del redactor, no una medición
independiente. La reverificación de 3a-2 señaló un problema de naturalidad, no de gramática: con
saludo + vocativo + BIENVENIDO, «Hola, bienvenido, hermano.» se oye en tres tramos, y el orden más
corriente sería «Hola, hermano, bienvenido.». Se pregunta en la Actividad B (E-15d).

**Fidelidad** (si la frase añade algo que las señas no dicen). La verificación no encontró ninguna
salida que invente contenido léxico ajeno a las glosas. Lo único que se añade son palabras
funcionales, todas registradas (§3.3), y la forma fija «con» de «Perdón, con permiso.». Hay dos
añadidos que no son neutros, y los dos están registrados como supuestos y se le preguntan a la
persona señante:

- ¿CÓMO ESTÁS? + un sustantivo se lee en 3.ª persona («¿Cómo está el hermano?») y no como vocativo
  («Hermano, ¿cómo estás?»). Son 48 conjuntos, con `lectura_tercera_persona`.
- BIENVENIDO con un sustantivo vocativo sale como fórmula («Bienvenida, familia.»), y con TÚ, USTEDES
  y TODOS, con cópula («Tú eres bienvenido.»). Son 48 conjuntos con `lectura_vocativa`.

---

## 8. Verificación humana pendiente (Actividad B)

**No realizada.** El material está listo:

- `docs/ACTIVIDAD_B_REVISION_FRASES.md`: instrucciones para el estudiante, las secuencias por
  bloque, las decisiones con opciones, las preguntas abiertas y qué hacer con cada respuesta.
- `docs/ACTIVIDAD_B_hoja_registro.csv`: 178 ítems. Contiene las 9 frases de ejemplo, las 22
  plantillas, 15 casos de reglas que dependen de decisiones por validar, 19 decisiones con opciones,
  5 respaldos que quizá deberían tener frase, el resto del inventario, las 50 glosas solas y 11
  preguntas abiertas. No tiene columnas para datos personales.

Por cada ítem se pregunta (especificación, §9.2): si la secuencia es natural en LSC y en qué orden
la señaría; si la frase dice lo mismo (**sí / en parte / no**, una versión reducida de la escala de
5 niveles de Koehn y Monz 2006, p. 107); y cómo la diría él. El precedente de verificar glosa y
traducción con una persona de la comunidad es el de Naranjo Orozco y Dieck (2022, p. 60).

**Cobertura.** Con `--conversor inventario`, si se completa la hoja, la persona señante habrá visto
todas las salidas que el prototipo puede decir: las 50 glosas solas y las 92 secuencias. Con
`--conversor reglas`, la hoja muestra 145 de 7.045 salidas posibles (2,1 %). Lo revisado no autoriza
a decir que el resto está validado.

**Qué se reportará.** Los juicios ítem por ítem, los recuentos de sí / en parte / no por bloque y la
diferencia literal con su versión. No se presentará BLEU como medida principal: BLEU se define
contra «a corpus of good quality human reference translations» (Papineni et al. 2002, p. 311), y
aquí habrá a lo sumo una referencia por ítem. Además, en juicios de adecuación «it is very difficult
to maintain consistent standards» (Koehn y Monz 2006, p. 107). Si se da alguna métrica automática,
será solo descriptiva, con SacreBLEU y su firma, como recomiendan Müller et al. (2023, pp. 685-686).

---

## 9. Uso en el prototipo

El conversor se elige con una opción de `src/realtime.py`. El valor por defecto no cambió.

```
python src/realtime.py --modo-oracion                          # heredado (GrammarMapper), como siempre
python src/realtime.py --modo-oracion --conversor inventario   # modo estricto: 50 solas + 92 secuencias
python src/realtime.py --modo-oracion --conversor reglas       # extensión: todas las salidas por reglas
python src/realtime.py --modo-oracion --conversor inventario --genero-primera f --genero-segunda m
```

- `--genero-primera` y `--genero-segunda` (`m` o `f`) dan el género de YO y de TÚ. Sin ellas se usa
  el masculino y se registra el supuesto `genero_por_defecto`. Con `--conversor heredado` no tienen
  efecto, y darlas termina en un error de argumentos.
- Con `reglas` o `inventario`, el registro JSONL del ensayo y del turno gana la clave `conversion`,
  con la traza de `explicar()`: vía, plantilla, reglas, `suplido`, `supuestos`, motivo y
  parámetros. Con el heredado, el registro queda igual.
- Desde Python: `construir_conversor("inventario")` en `src/translator.py` devuelve el conversor,
  que se pasa como `grammar_mapper=` a `LSCTranslator`. El módulo nuevo solo se importa si se pide.
- Desde la línea de órdenes: `python -m src.conversor_glosas` imprime las 92 secuencias (92/92 dan
  su enunciado fijo), y `--exportar RUTA.csv` escribe la tabla de salidas sin respaldo.
- Límite que ya existía: la ventana del prototipo dibuja con `FONT_HERSHEY_SIMPLEX`, que no muestra
  tildes, ñ ni ¿. La voz no se ve afectada.

---

## 10. Pendientes

1. **Actividad B** con la persona señante: ninguna regla está validada.
2. **Informar al director** de D-15: el procedimiento que se presenta es `inventario`, con `reglas`
   como extensión sin verificar.
3. Medir la prosodia de la voz con la puntuación nueva (L-17, actividad 3.b).
4. Fuentes que faltan (especificación, «Referencias»): norma sobre el artículo ante un sujeto
   nominal (R-15); uso vocativo de «todos» (R-10); la entrada «hambre» del DPD (R-24); «sentirse»
   con predicativo (R-23); variación regional de la LSC (L-19); *tú*, *usted* y *vos* en Cali (L-09);
   la Nueva gramática sobre el masculino genérico (R-17). Mientras falten, esas decisiones se
   presentan como juicio del autor.
5. La suite completa no pasa entera por `pytest` en `tests/test_arquitecturas_2.py`. Lo decide el
   autor: instalar pytest en venv312 o pasar ese archivo a unittest.

---

## 11. Archivos

| archivo | qué es |
|---|---|
| `docs/CONVERSOR_3a_ESPECIFICACION.md` | especificación normativa: léxico, 35 reglas, plantillas, inventario, motivos, trazabilidad, límites y verificación |
| `src/conversor_glosas.py` | implementación 3a-2 (`ConversorGlosas`) |
| `tests/test_conversor_glosas.py` | 59 pruebas |
| `src/translator.py`, `src/realtime.py`, `tests/test_prototipo.py` | integración aditiva (`--conversor`, `construir_conversor`, 8 pruebas) |
| `docs/CONVERSOR_3a_SALIDAS_NO_RESPALDO.csv` | las 7.045 salidas con frase (50 solas + 6.995 conjuntos), con su traza y el bloque de la Actividad B en que se mostrarán; `estado_revision = pendiente` en todas |
| `docs/CONVERSOR_3a_SALIDAS.csv` | tabla comparativa heredado → nuevo: 3.522 filas (50 solas, 22 plantillas, 2.450 pares y 1.000 tríos con semilla 2026). Se genera con `scripts/conversor_3a_salidas.py` |
| `docs/ACTIVIDAD_B_REVISION_FRASES.md`, `docs/ACTIVIDAD_B_hoja_registro.csv` | material de la Actividad B |
| `docs/estado_mapeo_gramatical.md` | estado medido del conversor heredado (N-42) |

---

## Referencias

Para cada una se indica si se verificó en el PDF o en la fuente, según la verificación del 8-nov.

**Verificadas (cita y página comprobadas):**

- INSOR, Instituto Caro y Cuervo y MEN (2006). *Diccionario básico de la Lengua de Señas
  Colombiana*. Bogotá [ref14]. Pp. xxii, xxix, xxxix, 480, 482, 486 y 488. Leído en una copia de
  web.archive.org, porque la URL de insor.gov.co da 404. Incluye a A. Oviedo, «Notas para orientar
  la lectura del Diccionario Básico…», pp. xxv y ss.
- Naranjo Orozco, A. y Dieck, M. (2022). «La expresión de la categoría gramatical de tiempo en la
  Lengua de Señas Colombiana». *Signo y Seña* 41, 48-78. doi:10.34096/sys.n41.10613. Pp. 60, 61 y
  66 (nota 12); p. 65, nota 11.
- Johnston, T. (2008). «Corpus linguistics and signed languages: no lemmata, no corpus». LREC 2008
  Workshop, 82-87. P. 84. Trata de Auslan.
- Müller, M. et al. (2023). «Considerations for meaningful sign language machine translation based
  on glosses». ACL 2023 (Short), 682-693. doi:10.18653/v1/2023.acl-short.60. Pp. 683 y 685-686.
- Moryossef, A. et al. (2021). «Data augmentation for sign language gloss translation». AT4SSL
  2021, 1-11. P. 4. Trata de ASL y DGS.
- San-Segundo, R. et al. (2010). «Advanced speech communication system for deaf people».
  Interspeech 2010, 250-253. doi:10.21437/Interspeech.2010-103. Pp. 251 y 253. Trata de la LSE, en
  la dirección español → LSE. En el PDF, el cuarto autor figura como «Syaheerah Lufti»: hay que
  corregir la grafía antes de citar.
- Reiter, E. (1995). «NLG vs. Templates». arXiv:cmp-lg/9504013. P. 4, nota 1, y p. 6.
- Koehn, P. y Monz, C. (2006). «Manual and automatic evaluation of machine translation between
  European languages». WMT 2006, 102-121. P. 107.
- Papineni, K. et al. (2002). «BLEU: a method for automatic evaluation of machine translation».
  ACL 2002, 311-318. P. 311.
- Zeshan, U. (2006). «Negative and interrogative constructions in sign languages». Ishara Press,
  cap. 2. doi:10.26530/OAPEN_453832. Pp. 40 y 42. La muestra no incluye la LSC.
- RAE y ASALE. *Diccionario panhispánico de dudas*, 2.ª ed. Entradas «coma» (§2.2.1, §2.2.3),
  «concordancia» (§4.7), «punto» (§2.3), «estar» (§2; versión provisional) y «signos de
  interrogación y exclamación» (§2.1, §2.2, §3.1). Leídas en copias de web.archive.org, porque rae.es
  responde 403. Conviene repetir la consulta desde un navegador antes de citar.

**No verificadas en esta actividad:**

- Oviedo, A. (2001). *Apuntes para una gramática de la LSC*. Univalle/INSOR [ref13]. Las páginas
  vienen del anteproyecto.
- DPD, entrada «usted»: no se pudo abrir (rae.es da 403 y web.archive.org no estaba accesible).
