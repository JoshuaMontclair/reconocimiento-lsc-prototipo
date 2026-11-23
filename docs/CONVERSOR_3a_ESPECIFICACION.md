# Especificación del procedimiento de conversión de glosas en enunciados en español (actividad 3.a)

**Fecha:** 8 de noviembre de 2026
**Estado:** especificación para implementar. **Nada de lo que sigue ha sido revisado por el profesor
el profesor** (persona señante vinculada, actividad 0.c, N-54). Toda regla marcada «validar: sí» es
una decisión de diseño del autor, con su fundamento, a la espera de la Actividad B (§9.2).
**Punto de partida, que no se toca:** `src/grammar_mapper.py` (md5
`ce3ac6c8f7b841bbd02118eca0f2d320`) y `tests/test_grammar_mapper.py` (89 pruebas), descritos en
`docs/estado_mapeo_gramatical.md` (N-42). En este documento se les llama **conversor heredado**.
**Qué se implementa:** un módulo nuevo, `src/conversor_glosas.py` (clase `ConversorGlosas`, versión
`3a-2`), que el prototipo puede elegir de dos formas. `--conversor inventario` (modo estricto: las 50
glosas solas y las 92 secuencias previstas de §4.3, cada una con su enunciado fijo) es la que se ajusta
al método del anteproyecto («a cada glosa o a cada secuencia corta prevista se le asociará […] un
enunciado fijo») y la que se presenta en la tesis como el procedimiento (decisión del autor del
8-nov-2026, D-15 en DECISIONES_TESIS.txt). `--conversor reglas`
es una **extensión** cerrada y enumerable que se puede auditar (§9.1, 7.045 salidas listadas), todavía
sin verificar. El valor por defecto era `heredado` (§7); desde el 22-nov-2026 `src/realtime.py` usa
`inventario` por defecto (decisión del autor; `docs/MODELO_TCN_IMPUTACION.md`, sección 7).

**De dónde salen las cifras.** Las del conversor heredado vienen de N-42 o se midieron hoy con
`python`. Las del conversor nuevo (§9.1) se midieron
hoy sobre un **borrador de referencia** de esta especificación, escrito fuera del repositorio solo
para comprobar que las reglas no son ambiguas y para contar. Ese borrador no es la implementación.
Sus cifras son las que la implementación debe reproducir. Si no las reproduce, hay que corregir el
código o este documento, de forma explícita.
**Revisión del 8-nov (tarde).** Tras una revisión adversarial se corrigió el texto y se rehízo el
borrador (versión 2, también fuera del repositorio). Cambios de regla que mueven cifras: ADIÓS +
vocativo se admite (R-11), YO + SENTIR + BIENVENIDO se admite (R-20) y el orden de comprobación de
`dos_<categoría>` es fijo (R-14). Todas las cifras de §9.1 son las del borrador 2, medidas de nuevo.
La respuesta punto por punto está al final («Respuesta a la revisión»). El borrador no calcula los
campos `reglas`, `suplido` ni `supuestos` (§6): esos campos se especifican aquí, pero no están
medidos.
**Segunda revisión del 8-nov (noche), versión 3a-2.** Una verificación de la implementación dio 26
hallazgos. Cambios de regla, primero en este documento y después en el código: 5 plantillas nuevas
(T-18 a T-22), GRACIAS + vocativo (R-11), BIENVENIDO con un vocativo como fórmula (R-20), BIENVENIDO
tras saludo concordado con `genero_segunda` (R-12), el supuesto `lectura_tercera_persona` en R-32 y el
tratamiento de `set`, `frozenset` e iterables sin fin (R-01). El inventario pasa de 85 a 92 secuencias.
**Desde esta versión, las cifras de §9.1 son las medidas con la implementación (3a-2)**; las del borrador
2 se conservan como historia donde se dice. La respuesta está en «Respuesta a la segunda revisión», al
final.

---

## 1. Propósito y alcance

### 1.1 Qué pide el anteproyecto

- Actividad 3.a (09_metodologia.tex): «Definición del procedimiento de conversión de glosas en
  enunciados en español: especificación de las reglas de conversión sobre el vocabulario soportado,
  tratamiento de las categorías que el español exige y la glosa no trae, documentación de los casos
  no cubiertos y verificación de los enunciados producidos con la persona señante o intérprete
  vinculada».
- Método (09_metodologia.tex, §«Mapeo de Glosas a Español»): «a cada glosa o a cada secuencia corta
  prevista se le asociará una palabra o un enunciado fijo en español».
- RE4 (04_objetivos.tex): «Procedimiento documentado de conversión de glosas en enunciados en
  español, con sus reglas explícitas y sus límites».
- Alcance (06_alcances.tex): queda fuera «el mapeo gramatical automático abierto entre la LSC y el
  español: la conversión a español se limita a reglas y plantillas sobre el vocabulario soportado».

### 1.2 Fundamento del componente

La justificación **no** se apoya en un contraste del orden de constituyentes entre la LSC y el
español. El anteproyecto lo descarta: la fuente disponible (ref13, Oviedo 2001) describe
S-V-OD-OI en elicitación y un orden «muy flexible» en discurso. La justificación es otra: la glosa
reconocida es una **etiqueta léxica**. Le faltan las categorías que el español exige, y el
procedimiento debe suplirlas de forma documentada y verificable. Lo respaldan estas fuentes:

- «la etiqueta con la cual se glosa en español tiene un valor bastante limitado» (Oviedo, en INSOR
  2006, p. xxix). Las entradas del Diccionario básico «corresponden a señas, no a palabras del
  español» (INSOR 2006, p. xxii).
- El ID-gloss «is not intended as a translation» (Johnston 2008, p. 84, sobre Auslan). Una secuencia
  lineal de glosas «is not an adequate representation of a signed utterance» (Müller et al. 2023,
  p. 683).
- Sobre plantillas: Reiter (1995) señala que el mantenimiento de las plantillas solo es un problema
  si son muchas, no si basta un conjunto pequeño (p. 4, nota 1), y que un sistema de plantillas
  puede manejar concordancias sencillas, como el plural del sustantivo o la concordancia
  sustantivo-verbo (p. 6). Es el caso de este procedimiento: un léxico de 50 glosas, 22 plantillas (17 en 3a-1) y
  concordancias de persona, número y género.
- Para ASL y DGS, no para la LSC, Moryossef et al. (2021, p. 4) resumen lo que le falta a la glosa
  frente a la lengua oral en tres grupos: flexión, palabras y puntuación omitidas, y diversidad
  sintáctica. Este documento usa esos tres grupos para ordenar lo que se suple (§3.2).
- Para la LSC: «En la LSC no se expresa léxicamente la cópula» (Naranjo Orozco y Dieck 2022, p. 66,
  nota 12). Los verbos «en sí mismos no contienen información temporal» (p. 61).

### 1.3 Qué hace el procedimiento

Recibe la lista de glosas de un turno (o una sola glosa en modo aislado) y devuelve **un** texto en
español, por una de cuatro vías:

| Vía | Cuándo | Qué sale |
|---|---|---|
| `lexico` | una sola glosa distinta | el enunciado fijo de esa glosa (§2) |
| `plantilla` | el conjunto de glosas está en el inventario cerrado de plantillas (§4) | el enunciado fijo de la plantilla |
| `reglas` | el conjunto encaja en uno de los patrones admitidos (§3) | un enunciado armado por reglas explícitas, que se puede enumerar de antemano (§9.1) |
| `respaldo` | ninguna de las anteriores | las glosas en minúscula, separadas por comas, marcado como respaldo (§5) |

Más un caso límite: sin ninguna glosa del vocabulario, la salida es `""` (vía `vacio`).

### 1.4 Qué NO hace

- No traduce de forma abierta. Toda salida que no es respaldo pertenece a un conjunto finito que se
  puede listar entero: **7.045 salidas** (50 glosas solas + 6.995 conjuntos de 2 a 4 glosas, §9.1,
  versión 3a-2).
- **No garantiza que esas salidas estén revisadas.** Que se puedan enumerar no las hace «previstas»
  en el sentido del anteproyecto ni verificadas. La vía `reglas` arma frase para 6.965 conjuntos, y
  nadie las revisa una por una; el inventario tiene 92. Hoy no hay ninguna verificada. Después de la
  Actividad B (§9.2), solo podrán darse por verificadas las que se le muestren a la persona señante:
  las 50 solas (B1), las 92 secuencias del inventario (B2) y una muestra de 30 conjuntos (B3), es
  decir, 172 de las 7.045 salidas (2,4 %); de los 6.995 conjuntos, 122 (1,7 %). Con `--conversor
  reglas`, el prototipo puede decir en voz alta frases que nadie revisó, algunas raras aunque
  gramaticales («¿Quién es bienvenido?», «¿Por qué estoy mal yo?»). Por eso **la implementación que se
  ajusta al método del anteproyecto es el modo estricto** (`--conversor inventario`, §7): solo dice las
  glosas solas y las secuencias del inventario, y todo lo demás va a respaldo (medido: 92 conjuntos con
  frase y 6.903 con `fuera_de_inventario`; L-21). La vía `reglas` se presenta como extensión, no como el
  procedimiento. Es una propuesta del autor que hay que decidir con el director (texto propuesto para
  DECISIONES_TESIS.txt al final, en «Respuesta a la segunda revisión»).
- No usa modelos de lenguaje, redes, estadística, corpus paralelos ni aprendizaje. Son tablas y
  condiciones escritas a mano.
- No infiere funciones sintácticas (sujeto, objeto) del orden de entrada (R-04).
- No detecta lo que no llega como glosa: rasgos no manuales, uso del espacio, repetición expresiva,
  tiempo por contexto (§8).
- No corrige errores del reconocedor. Si llega una glosa equivocada, puede producir una frase
  gramatical pero falsa (§8, L-13).
- No reemplaza al conversor heredado ni cambia el comportamiento por defecto del prototipo.

---

## 2. Léxico

### 2.1 Criterio de categorías

El conversor heredado usa 8 categorías (N-42). Este diseño usa **15**. El criterio es uno solo:
**una categoría es el conjunto de glosas que cumplen la misma función en las plantillas de núcleo**
(sujeto, verbo, predicado, modificador, interrogativo, periferia). Las diferencias de
comportamiento dentro de una categoría se declaran como **rasgos** en §2.3, no como excepciones en
el código. Por eso `ADJ` reúne adjetivos con cópula *ser* y con cópula *estar*: la cópula es un
rasgo (R-20, R-26). Hay 5 categorías de una sola glosa (`PREG_FIJA`, `CUANT_PRON`, `NOM_TENER`,
`VERBO_SENTIR`, `ADV_FREC`). Se separan porque su función en el núcleo es distinta de la de
cualquier otra categoría (por ejemplo, HAMBRE forma predicado con «tener» y no con cópula), no porque
una regla las nombre. Podrían modelarse como rasgos de otra categoría sin cambiar ninguna salida.
Es una decisión de presentación.

| Categoría | Glosas | n | Por qué es una categoría aparte |
|---|---|---|---|
| `SAL` saludo | hola, buenos días, buenas tardes, buenas noches, adiós | 5 | Abre el enunciado y se separa con coma (R-09). `adiós` lleva el rasgo `despedida`: por reglas solo combina con TODOS (R-10) y con un vocativo (R-11); por plantilla, con GRACIAS (T-15). BUENOS DÍAS, BUENAS TARDES y BUENAS NOCHES combinan con GRACIAS por plantilla (T-18, T-19, T-16) |
| `CORT` cortesía | gracias, por favor, con gusto, perdón, permiso | 5 | Solo entra en plantillas (R-13), salvo GRACIAS + vocativo (R-11, desde 3a-2). `permiso` era `objeto` en el heredado; se pasa aquí porque ninguno de los 4 verbos lo toma como complemento, y como fórmula («Permiso, por favor») sí forma enunciados |
| `PREG_FIJA` pregunta fija | ¿cómo estás? | 1 | Ya es un enunciado completo. Con un sujeto de 3.ª persona cambia de forma (R-32) |
| `INTERR` interrogativo | ¿qué?, ¿cuándo?, ¿dónde?, ¿cómo?, ¿por qué?, ¿quién? | 6 | Convierte el núcleo en pregunta parcial (R-29 a R-31) |
| `PRON` pronombre personal | yo, tú, ustedes | 3 | Sin artículo. Se conserva explícito (R-16). Persona gramatical propia |
| `CUANT_PRON` pronombre cuantificador | todos | 1 | Sin artículo. Tras un saludo es destinatario («a todos», R-10) |
| `NOM_PERS` sustantivo de persona | familia, personas, mujer, hombre, niño, niña, abuelo, tío, hermano | 9 | Puede ser sujeto y lleva artículo (R-15). Rasgos `vocativo` (R-11) y `quien` (R-31) |
| `NOM_ABS` sustantivo abstracto | seña, nombre | 2 | No es sujeto de ningún verbo del léxico. Solo entra en plantillas (R-28) |
| `NOM_TENER` sustantivo de estado | hambre | 1 | Forma predicado con «tener» (R-24) |
| `VERBO` verbo | trabajar, comer, vivir | 3 | Se conjuga (R-18). Rasgos `cant`, `modo`, `nunca` y `exige_complemento` |
| `VERBO_SENTIR` verbo pronominal | sentir | 1 | Exige predicativo y sale como «sentirse» (R-23) |
| `ADJ` adjetivo | feliz, contento, triste, aburrido, juicioso, diferente, bienvenido | 7 | Necesita cópula (R-20) y concuerda (R-21). Rasgos `copula` y `con_sentir` |
| `ADV_EST` adverbio de estado | bien, mal, más o menos | 3 | Predicado con «estar» o modo del verbo (R-22) |
| `ADV_CANT` cuantificador | mucho, poco | 2 | Cambia de forma según lo que modifica (R-25) |
| `ADV_FREC` frecuencia | nunca | 1 | Va antes del verbo (R-26) |

Total: 50. **Sujeto** (SUJ) designa a la unión `PRON` ∪ `CUANT_PRON` ∪ `NOM_PERS` (13 glosas,
igual que el heredado).

### 2.2 Criterio para la glosa sola (modo aislado y turno de una glosa)

**Decisión D-L1: la glosa sola se dice en su forma de cita, con mayúscula inicial y punto final. Las
preguntas van entre ¿…?, sin punto. No se conjuga, no se pone artículo y no se usan signos de
exclamación.**

1. **Verbos en infinitivo («Comer.»), no en 3.ª persona («Come»).** Conjugar exige una persona, y
   en una seña sola nadie la señó. Los verbos de la LSC marcan persona (anteproyecto, 08_marco.tex,
   l. 147-148, citando a Oviedo, ref13, no verificado aquí); la forma en que la marcan no llega al
   clasificador como glosa. Además, «Come», «Trabaja», «Vive» y
   «Siente» coinciden con el imperativo de *tú*: suenan a orden. El infinitivo es la forma de cita
   de la glosa de LSC50 y no afirma nada que no se haya señado. **Validar: sí.** Es la decisión
   que más cambia la salida del modo aislado.
2. **Sustantivos sin artículo («Seña.», «Hambre.»).** El artículo definido supone un referente
   conocido que no se señó. Así desaparece además el error heredado «La hambre» (H4).
3. **Adjetivos en masculino singular («Contento.»),** que es la forma de cita. Sin sujeto no hay
   con qué concordar. En la vía `lexico` no se registra `genero_por_defecto` (§6): la forma de cita
   no elige género para una persona, igual que el infinitivo no elige tiempo. Sí se registra cuando
   el masculino se aplica a una persona concreta: YO o TÚ (R-17) y el destinatario de BIENVENIDO tras
   un saludo (R-12).
4. **Fórmulas y preguntas tal cual («Buenos días.», «¿Cómo estás?»).** Ya son enunciados.
5. **Punto final.** El DPD (s. v. «punto», §2) asigna al punto el cierre de todo enunciado que no
   sea interrogativo ni exclamativo, y la salida se dice en voz alta como enunciado. El DPD (§2.3)
   permite omitirlo en carteles y rótulos, y se podría sostener que una palabra en pantalla es un
   rótulo. Se elige el punto por coherencia con el modo oración. Corrige el hallazgo de N-42: 0 de
   50 salidas terminaban en punto. Con este criterio terminan en punto 43 y en «?» 7 (borrador).
   **Validar: sí**, junto con el efecto en la prosodia de la voz (pregunta abierta 11 de N-42, no medida).
6. **Sin «¡…!».** La exclamación expresa una intensidad que en la seña va en rasgos que el
   prototipo no ve. Se prefiere el enunciado neutro. **Validar: sí.**

Con este criterio las 50 salidas solas siguen siendo distintas entre sí (50 de 50, borrador e
implementación), lo que mantiene el requisito 5(b) del contrato de interfaz (Apéndice A). El
precalentado de la voz (`LSCTranslator._precalentar_voz`, llamado desde `__init__`) llama a
`translate([g])` sobre el conversor inyectado, así que precalienta estas mismas 50 frases sin cambiar
nada.

### 2.3 Tabla del léxico (las 50 glosas)

Columnas: **forma** = forma de cita en español. **G/N/P** = género, número, persona (solo donde una
regla lo usa). **Rasgos** = los que consultan las reglas. **Sola** = salida exacta del conversor
nuevo para la glosa sola. **Heredado** = salida actual (inventario del 8-nov).

| id | glosa | forma | categoría (heredada) | G/N/P | rasgos | sola | heredado |
|---|---|---|---|---|---|---|---|
| 0 | gracias | gracias | CORT (cortesia) | — | — | Gracias. | Gracias |
| 1 | buenos días | buenos días | SAL (saludo) | — | — | Buenos días. | Buenos días |
| 2 | buenas tardes | buenas tardes | SAL (saludo) | — | — | Buenas tardes. | Buenas tardes |
| 3 | buenas noches | buenas noches | SAL (saludo) | — | — | Buenas noches. | Buenas noches |
| 4 | seña | seña | NOM_ABS (objeto) | f, sg | solo plantilla | Seña. | La seña |
| 5 | nombre | nombre | NOM_ABS (objeto) | m, sg | solo plantilla | Nombre. | El nombre |
| 6 | trabajar | trabajar | VERBO (verbo) | — | cant sí, modo sí, nunca sí | Trabajar. | Trabaja |
| 7 | comer | comer | VERBO (verbo) | — | cant sí, modo sí, nunca sí | Comer. | Come |
| 8 | vivir | vivir | VERBO (verbo) | — | cant no, modo sí, nunca no, exige_complemento | Vivir. | Vive |
| 9 | poco | poco | ADV_CANT (adverbio) | — | solo con verbo cant sí y con hambre | Poco. | Poco |
| 10 | familia | familia | NOM_PERS (sujeto) | f, sg, 3 | colectivo, vocativo sí, quien no | Familia. | La familia |
| 11 | personas | personas | NOM_PERS (sujeto) | f, pl, 3 | vocativo no, quien sí | Personas. | Personas |
| 12 | mujer | mujer | NOM_PERS (sujeto) | f, sg, 3 | vocativo no, quien sí | Mujer. | La mujer |
| 13 | hombre | hombre | NOM_PERS (sujeto) | m, sg, 3 | vocativo no, quien sí | Hombre. | El hombre |
| 14 | niño | niño | NOM_PERS (sujeto) | m, sg, 3 | vocativo sí, quien sí | Niño. | El niño |
| 15 | niña | niña | NOM_PERS (sujeto) | f, sg, 3 | vocativo sí, quien sí | Niña. | La niña |
| 16 | abuelo | abuelo | NOM_PERS (sujeto) | m, sg, 3 | vocativo sí, quien sí | Abuelo. | El abuelo |
| 17 | tío | tío | NOM_PERS (sujeto) | m, sg, 3 | vocativo sí, quien sí | Tío. | El tío |
| 18 | hermano | hermano | NOM_PERS (sujeto) | m, sg, 3 | vocativo sí, quien sí | Hermano. | El hermano |
| 19 | hambre | hambre | NOM_TENER (objeto) | f, sg | con «tener»; cuantificador mucha/poca | Hambre. | La hambre |
| 20 | feliz | feliz | ADJ (adjetivo) | inv. en género; pl felices | cópula estar, con_sentir sí | Feliz. | Feliz |
| 21 | contento | contento | ADJ (adjetivo) | -o/-a/-os/-as | cópula estar, con_sentir sí | Contento. | Contento |
| 22 | triste | triste | ADJ (adjetivo) | inv.; pl tristes | cópula estar, con_sentir sí | Triste. | Triste |
| 23 | aburrido | aburrido | ADJ (adjetivo) | -o/-a/-os/-as | cópula estar, con_sentir sí | Aburrido. | Aburrido |
| 24 | bien | bien | ADV_EST (adverbio) | — | admite «muy» | Bien. | Bien |
| 25 | mal | mal | ADV_EST (adverbio) | — | admite «muy» | Mal. | Mal |
| 26 | ¿cómo estás? | ¿cómo estás? | PREG_FIJA (pregunta) | — | — | ¿Cómo estás? | ¿Cómo estás? |
| 27 | más o menos | más o menos | ADV_EST (adverbio) | — | no admite «muy» ni «nunca» | Más o menos. | Más o menos |
| 28 | sentir | sentir(se) | VERBO_SENTIR (verbo) | — | exige predicativo | Sentir. | Siente |
| 29 | juicioso | juicioso | ADJ (adjetivo) | -o/-a/-os/-as | cópula ser, con_sentir no | Juicioso. | Juicioso |
| 30 | hola | hola | SAL (saludo) | — | — | Hola. | Hola |
| 31 | adiós | adiós | SAL (saludo) | — | despedida | Adiós. | Adiós |
| 32 | por favor | por favor | CORT (cortesia) | — | — | Por favor. | Por favor |
| 33 | con gusto | con gusto | CORT (cortesia) | — | — | Con gusto. | Con gusto |
| 34 | bienvenido | bienvenido | ADJ (adjetivo) | -o/-a/-os/-as | cópula ser, con_sentir sí; YO + cópula → respaldo (R-20) | Bienvenido. | Bienvenido |
| 35 | perdón | perdón | CORT (cortesia) | — | — | Perdón. | Perdón |
| 36 | permiso | permiso | CORT (objeto) | — | — | Permiso. | El permiso |
| 37 | nunca | nunca | ADV_FREC (adverbio) | — | — | Nunca. | Nunca |
| 38 | yo | yo | PRON (sujeto) | parám. (m), sg, 1 | — | Yo. | Yo |
| 39 | tú | tú | PRON (sujeto) | parám. (m), sg, 2 | — | Tú. | Tú |
| 40 | ustedes | ustedes | PRON (sujeto) | m, pl, 3 | — | Ustedes. | Ustedes |
| 41 | ¿qué? | qué | INTERR (pregunta) | — | — | ¿Qué? | ¿Qué? |
| 42 | ¿cuándo? | cuándo | INTERR (pregunta) | — | — | ¿Cuándo? | ¿Cuándo? |
| 43 | ¿dónde? | dónde | INTERR (pregunta) | — | — | ¿Dónde? | ¿Dónde? |
| 44 | ¿cómo? | cómo | INTERR (pregunta) | — | — | ¿Cómo? | ¿Cómo? |
| 45 | ¿por qué? | por qué | INTERR (pregunta) | — | — | ¿Por qué? | ¿Por qué? |
| 46 | ¿quién? | quién / quiénes | INTERR (pregunta) | — | — | ¿Quién? | ¿Quién? |
| 47 | diferente | diferente | ADJ (adjetivo) | inv.; pl diferentes | cópula ser, con_sentir sí | Diferente. | Diferente |
| 48 | todos | todos | CUANT_PRON (sujeto) | m, pl, 3 | no con «nunca»; destinatario tras saludo | Todos. | Todos |
| 49 | mucho | mucho | ADV_CANT (adverbio) | — | con adjetivo/bien/mal → «muy» | Mucho. | Mucho |

### 2.4 Tablas de formas (cerradas, sin derivar por terminación)

Presente de indicativo. Solo existen las personas que el léxico puede producir: 1.ª sg (yo),
2.ª sg (tú), 3.ª sg (sustantivos en singular) y 3.ª pl (ustedes, todos, personas).

| verbo | 1s | 2s | 3s | 3p |
|---|---|---|---|---|
| trabajar | trabajo | trabajas | trabaja | trabajan |
| comer | como | comes | come | comen |
| vivir | vivo | vives | vive | viven |
| sentir(se) | me siento | te sientes | se siente | se sienten |
| ser | soy | eres | es | son |
| estar | estoy | estás | está | están |
| tener | tengo | tienes | tiene | tienen |

Adjetivos: feliz/felices, triste/tristes, diferente/diferentes (invariables en género);
contento, aburrido, juicioso y bienvenido con -o / -a / -os / -as. Artículo definido: el, la, los,
las.

---

## 3. Reglas del modo oración

### 3.1 Algoritmo (orden de aplicación)

Este es el orden **real** de comprobación, el que usa el borrador 2 y la implementación. Es normativo: el motivo que se
registra es el de la primera comprobación que falla en este orden, y los recuentos de §9.1 dependen
de él. Las reglas de §3.3 están numeradas por tema, no por orden de ejecución; cuando el orden de
§3.3 y el de aquí no coinciden, manda este. Los nombres entre comillas invertidas son los códigos
exactos de `MOTIVOS` (§5.2).

```
explicar(signs):
 A. Entrada
  A1. R-01  leer el argumento (a lo sumo 10.000 elementos; un set o frozenset, en orden de id) y
            normalizar cada elemento → glosa del vocabulario o «fuera de vocabulario»
  A2. R-05  si no queda ninguna glosa del vocabulario → frase "" (vía vacio)
  A3. R-02  si hubo algún elemento fuera de vocabulario → `fuera_de_vocabulario`
  A4. R-03  colapsar repeticiones → glosas distintas, en orden de primera aparición
  A5. R-06  si queda una sola glosa → enunciado del léxico (§2.3), vía lexico
  A6. R-04  desde aquí solo cuenta el CONJUNTO de glosas distintas
  A7. R-07  si el conjunto es la clave exacta de una plantilla (§4.2) → enunciado fijo
  A8. R-08  si tiene más de 4 glosas → `demasiadas_glosas`
 B. Periferia (solo si hay glosas SAL; «resto» = el conjunto sin los saludos)
  B1. R-09  dos saludos que no son HOLA + {BUENOS DÍAS | BUENAS TARDES | BUENAS NOCHES}, o tres
            o más → `saludos_no_combinables`
  B2. R-09  resto vacío → «Hola, buenos días.»
  B3. R-10  resto = {TODOS} → «Saludo a todos.» (también con ADIÓS)
  B4. R-11  resto = {un NOM_PERS con vocativo sí} → «Saludo, x.» (también con ADIÓS)
  B5. R-09  ADIÓS con cualquier otro resto → `despedida_con_contenido`
  B6. R-07  resto = clave de T-03 o T-04 → «Saludo, » + plantilla (vía plantilla)
  B7. R-12  resto de una sola glosa: BIENVENIDO (concordado con genero_segunda) o ¿CÓMO ESTÁS? →
            «Saludo, …»; cualquier otra → `saludo_con_glosa_no_admitida`
  B8.       resto de 2 o 3 glosas → núcleo (C) con el resto; salida «Saludo, » + núcleo
 C. Núcleo (el conjunto, o el resto de B8)
  C1. R-28  hay NOM_ABS (SEÑA, NOMBRE) → `sustantivo_abstracto_fuera_de_plantilla`
  C2. R-13  hay CORT → si el núcleo es exactamente {GRACIAS, un NOM_PERS con vocativo sí} y no
            viene de B8 (sin saludo) → «Gracias, x.» (R-11); si no → `cortesia_fuera_de_plantilla`
  C3. R-14  más de un SUJ → `dos_sujetos`
  C4. R-29  más de una glosa INTERR o PREG_FIJA → `dos_interrogativos`
  C5. Si hay una glosa interrogativa (D); si no, declarativa (C6)
  C6. R-14  sin SUJ → `sin_sujeto`
  C7a. R-20 SUJ = NOM_PERS con vocativo sí y resto = {BIENVENIDO} → «Bienvenido, x.» (fórmula,
            sin artículo ni cópula; también tras un saludo, por B8)
  C7. predicado(resto sin el SUJ) → P1…P9; salida SUJ + predicado + «.»
 D. Interrogativa
  D1. ¿CÓMO ESTÁS?  R-32  con resto, sin SUJ o con YO → `como_estas_no_admitido`; si no,
                          «¿Cómo está…?» (con un SUJ vocativo, supuesto `lectura_tercera_persona`)
  D2. ¿QUIÉN?       R-31  sin SUJ → predicado(resto) en 3.ª singular (P1…P9)
                          con SUJ y resto → `quien_con_sujeto_y_predicado`
                          con SUJ = YO, TODOS o FAMILIA → `quien_sujeto_no_admitido`
  D3. otro INTERR   R-30  sin SUJ → `pregunta_sin_sujeto`
                          ¿DÓNDE?, ¿CUÁNDO?, ¿QUÉ?, ¿CÓMO?: con YO, o resto fuera de la tabla
                          de R-30 → `pregunta_no_admitida`
                          ¿POR QUÉ?: predicado(resto) (P1…P9)
 P. predicado(resto), en este orden
  P1. R-14  dos glosas de una misma categoría, comprobadas en este orden: VERBO, ADJ, ADV_EST,
            ADV_CANT → `dos_VERBO`, `dos_ADJ`, `dos_ADV_EST`, `dos_ADV_CANT`
  P2. R-26  NUNCA + MUCHO o POCO → `nunca_con_cantidad`
  P3. R-26  NUNCA con sujeto TODOS → `todos_con_nunca`
  P4. R-26  NUNCA + VERBO + cualquier predicado (ADJ, ADV_EST, HAMBRE) → `nunca_con_modo`
  P5. R-14  dos predicados de distinta categoría (ADJ, ADV_EST, NOM_TENER) → `dos_predicados`
  P6. R-14  VERBO + SENTIR → `dos_verbos`
  P7. con VERBO:
        R-14  + ADJ o HAMBRE → `verbo_con_predicado`
        R-26  + NUNCA con VIVIR → `nunca_no_admitido`
        R-25  + ADV_EST + MUCHO: con BIEN o MAL → «muy bien/mal»; con MÁS O MENOS →
              `cantidad_y_modo`. + ADV_EST + POCO → `cantidad_y_modo`
        R-25  VIVIR + MUCHO o POCO sin ADV_EST → `cantidad_no_admitida`
        R-27  VIVIR sin complemento → `vivir_sin_complemento`
  P8. con SENTIR (R-23):
        sin predicativo → `sentir_sin_predicativo`; con HAMBRE → `sentir_hambre`;
        con JUICIOSO → `adjetivo_no_admite_sentir`; POCO, o MUCHO + MÁS O MENOS →
        `cantidad_no_admitida`; NUNCA + MÁS O MENOS → `nunca_no_admitido`
  P9. sin verbo:
        sin predicado → `sin_predicado`
        ADJ: YO + BIENVENIDO → `yo_bienvenido` (R-20); POCO → `cantidad_no_admitida` (R-25)
        ADV_EST: un cuantificador que no sea MUCHO + BIEN/MAL → `cantidad_no_admitida`
        NUNCA con cópula ser o con MÁS O MENOS → `nunca_no_admitido` (R-26)
 E. Salida: R-33 mayúscula inicial, R-34 cierre («.» o «?»), R-35 forma del respaldo
 F. Modo estricto (solo con estricto=True, §7.1): si la vía resultante es `reglas` o `plantilla`
    y el conjunto no es una secuencia S-xx de §4.3 → `fuera_de_inventario`
```

Toda condición «→ respaldo» detiene el algoritmo y registra su código. Hay ramas defensivas que el
léxico actual no puede alcanzar (por ejemplo, un SUJ sin resto después de B8, o un verbo sin rasgo
`modo`). Si alguna se alcanzara, da respaldo con el código `no_previsto`, que no está en `MOTIVOS`.
El barrido exhaustivo de §9.1 (prueba 4) comprueba que no aparece nunca. Medido con el borrador 2:
0 veces en los 251.125 conjuntos de 2 a 4 glosas, en los que aparecen 32 motivos distintos. Medido
con la implementación 3a-2: 0 veces y los mismos 32 motivos; las pruebas fijan además cuántas veces
sale cada uno (§9.1) y comprueban por separado cada par de comprobaciones que este orden decide
(prueba 11). El motivo no depende del orden de iteración de Python: los 244.130 conjuntos de respaldo
de 3a-2 dan el mismo motivo con `PYTHONHASHSEED` 1 y 7 (prueba 7; con el borrador 2 eran 244.141). El borrador 1 recorría las categorías de P1 en el orden
de iteración de un `frozenset`. Con 4 glosas como máximo, el resto de un núcleo con sujeto tiene a lo
sumo 3 glosas y no puede repetir dos categorías, así que eso no cambiaba ninguna cifra. Aun así, el
orden de P1 se fija para que el motivo no dependa de un detalle de Python.

### 3.2 Qué categorías suple cada regla

Organizado con los tres grupos de Moryossef et al. (2021, p. 4), que describen ASL y DGS y aquí se
usan solo como clasificación:

| Lo que el español exige y la glosa no trae | Reglas | Código en `suplido` (§6) |
|---|---|---|
| **Flexión:** persona y número del verbo | R-17, R-18 | `persona_numero` |
| **Flexión:** género y número del adjetivo | R-21 | `concordancia_adjetivo` |
| **Flexión:** tiempo verbal | R-19 | `tiempo_verbal` |
| **Palabras omitidas:** artículo | R-15 | `articulo` |
| **Palabras omitidas:** cópula ser/estar | R-20, R-22 | `copula` |
| **Palabras omitidas:** verbo de soporte «tener», pronombre «me/te/se», preposición «a», «muy» | R-24, R-23, R-10, R-25 | `verbo_soporte`, `pronombre_reflexivo`, `preposicion`, `intensificador` |
| **Puntuación:** coma tras saludo, ¿…?, punto, mayúscula | R-09, R-30, R-33, R-34 | `puntuacion`, `mayuscula` |
| **Orden:** lo fija cada regla o plantilla, no la entrada | R-04, R-30 | `orden_fijo` |

`suplido` registra **qué categoría** se añadió. Cuando además hubo que elegir un **valor** que la
glosa no da, ese valor va en `supuestos` (§6): `articulo` → `articulo_definido` (R-15);
`persona_numero` con TÚ → `tratamiento_tu` (R-16, R-17); `concordancia_adjetivo` con YO o TÚ y un
adjetivo que varía en género, o con el destinatario de BIENVENIDO tras un saludo, cuando el género es
el de por defecto → `genero_por_defecto` (R-17, R-12); `tiempo_verbal` → `tiempo_presente` (R-19).
Desde 3a-2 se registran también dos **lecturas** elegidas entre dos posibles: `lectura_tercera_persona`
(¿CÓMO ESTÁS? + un sustantivo con vocativo sí, R-32) y `lectura_vocativa` (un sustantivo con vocativo
sí + BIENVENIDO, R-20). Un código no aparece en las dos listas.

### 3.3 Reglas

Formato de cada regla: **Condición**, **Acción**, **Ejemplo** (entrada → salida), **Fundamento** y
**Validar** (sí: decisión de diseño que se muestra al profesor de LSC vinculado; no: hecho de forma o de
código). Las glosas van en MAYÚSCULA, según la convención de INSOR (2006, Anexo 2, p. 567).

#### Etapa A — Entrada

**R-01 · Normalización de cada elemento**
- Condición: el argumento `signs` y cada uno de sus elementos.
- Acción sobre el argumento:
  - `str` suelto (por ejemplo, `translate("hola")`) → se trata como `["hola"]` y se registra el aviso
    `entrada_str_suelta`. Sin esto, se iteraría letra por letra y saldría `""` sin aviso.
  - `None`, `bytes`, `bytearray`, un `dict` u otro objeto que no sea iterable → frase `""` (vía
    `vacio`) con el aviso `entrada_no_lista`. `bytes` se excluye porque, al iterarlo, da enteros que
    se leerían como ids.
  - `set` o `frozenset` → sus elementos en orden de id de `SIGN_VOCABULARY` (lo que no es glosa,
    después, por `repr`), con el aviso `entrada_sin_orden`. Un conjunto no trae orden, y el de su
    iteración depende de `PYTHONHASHSEED`: sin esto, el orden del respaldo (R-35) cambiaría entre
    procesos (desde 3a-2).
  - Cualquier otro iterable (lista, tupla, generador) se recorre una vez, **hasta 10.000 elementos**
    (`MAX_ELEMENTOS_ENTRADA`). Si hay más, el resto no se lee y se registra el aviso
    `entrada_truncada` (§5.3). Sin el tope, un iterable sin fin (`itertools.count()`) dejaba
    `explicar` sin volver hasta agotar la memoria (medido en la verificación de 3a-1).
  - Un iterable que lanza al recorrerlo → frase `""` con `entrada_no_lista`, sin usar lo leído antes
    del fallo (§5.1, garantía 2).
- Acción sobre cada elemento:
  - entero: cualquier `numbers.Integral` que no sea `bool` (incluye `numpy.int64`, que numpy
    registra como `Integral`; medido con numpy 2.4.3) → `SIGN_VOCABULARY[int(x)]`. Un id fuera de
    0-49 queda fuera de vocabulario. Así el conversor no depende de que translator.py ya haga
    `int()` al leer la clase del modelo (`LSCTranslator._inferir_y_emitir`, leído el 8-nov). `numpy.bool_` no es `Integral` (medido) y queda fuera de vocabulario.
  - `str` → normalización Unicode NFC, `lower()`, `strip()` y espacios internos reducidos a uno.
    «Espacio» es cualquier carácter que `str.split()` trata como tal, incluidos el tabulador y el
    espacio duro (U+00A0): `["yo\xa0", "comer\t"]` → «Yo como.».
    Si el resultado es una de las 50 glosas, se acepta. Si no, se busca en una tabla de **13 alias**
    generada de forma automática: la misma glosa sin tildes (salvo ñ) y sin ¿? («adios» →
    adiós, «tu» → tú, «como estas» → ¿cómo estás?). Medido hoy: las 50 formas sin tildes ni ¿? son
    distintas entre sí y ninguna coincide con otra glosa. 13 de ellas difieren de la glosa y son los
    alias. El borrador no implementa los alias, y no afectan a ninguna cifra de §9.1.
  - Cualquier otro tipo (`None`, `float`, `bool`, listas) → fuera de vocabulario.
  - Registro de lo rechazado: en `fuera_de_vocabulario` se guarda `repr(x)` dentro de `try/except`
    (si `repr` lanza, se guarda `"<repr falló: NombreDelTipo>"`), truncado a 40 caracteres con «…», y
    como máximo los 10 primeros. El total va en `n_fuera_de_vocabulario`. Así un `__repr__` roto no
    rompe la garantía y 1.000 elementos rechazados no inflan el registro.
  - **Nunca lanza excepción** (la única excepción del módulo es la del constructor, §7.1).
- Ejemplo: `["  HOLA "]` → «Hola.»; `["tu","feliz"]` → «Tú estás feliz.»; `[True]` → `""`;
  `"hola"` → «Hola.» con aviso; `None` → `""` con aviso.
- Fundamento: corrige H14 (perder una tilde cambiaba la salida), H15 (`None` lanzaba
  `AttributeError`) y H15b (`True` se traducía como id 1) de N-42.
- Validar: no.

**R-02 · Elementos fuera de vocabulario**
- Condición: al menos un elemento quedó fuera de vocabulario y queda al menos una glosa válida.
- Acción: **respaldo** con las glosas válidas. Se registran los elementos rechazados en
  `fuera_de_vocabulario`. No se arma frase con lo que queda, porque omitir una pieza puede cambiar el
  sentido sin que nadie lo note (H13 de N-42).
- Ejemplo: `["yo","perro","comer"]` → `yo, comer` (respaldo, motivo `fuera_de_vocabulario`). El
  heredado da «Yo como».
- Fundamento: H13. En el prototipo no se alcanza, porque la entrada sale siempre de
  `SIGN_VOCABULARY` (contrato de interfaz, Apéndice A, punto 2).
- Validar: no.

**R-03 · Repetición de la misma glosa**
- Condición: una glosa aparece más de una vez en el turno, seguida o no.
- Acción: se conserva una sola aparición, la primera. Se registra cuántas se quitaron en
  `repeticiones`.
- Ejemplo: `["gracias","gracias"]` → «Gracias.» (heredado: «Gracias, gracias»).
  `["niño","niño"]` → «Niño.».
- Fundamento: el conversor no puede distinguir entre tres causas. (a) La misma seña reconocida en
  dos ventanas, como con las señas cortadas de N-50. (b) Énfasis. (c) Plural por repetición en el
  espacio, que existe para NIÑO (Oviedo, en INSOR 2006, p. xxxix), aunque el clasificador no ve el
  espacio. Colapsar es la opción que no inventa significado. La lectura (c) se declara no cubierta
  (L-03).
- Validar: sí. ¿Una seña repetida en el turno es casi siempre énfasis o plural?

**R-04 · Orden de entrada**
- Condición: siempre, salvo en el respaldo.
- Acción: la elección de plantilla o regla, y la frase resultante, dependen **solo del conjunto**
  de glosas distintas. El orden de salida lo fija cada regla o plantilla. El respaldo sí conserva el
  orden de primera aparición, porque es una transcripción y no una frase.
- Ejemplo: `["comer","yo"]` y `["yo","comer"]` → «Yo como.». `["¿dónde?","hermano","vivir"]` y
  sus 5 permutaciones → «¿Dónde vive el hermano?».
- Fundamento: (1) La literatura no permite leer funciones en el orden. Los ejemplos del Diccionario
  básico ponen el interrogativo al final, al principio o en los dos sitios (INSOR 2006, pp. 480, 482
  y 488). El anteproyecto, citando a Oviedo (ref13), describe el orden en discurso como «muy
  flexible». (2) Con este léxico y la restricción de un sujeto, un verbo y un predicado por turno
  (R-14), el orden no hace falta para asignar funciones. (3) Corrige un comportamiento incoherente:
  en el heredado, 188 de 1.225 pares eran sensibles al orden, sin que el código declarara cuál era
  el comportamiento pretendido (N-42, H7a y H7b). Con el borrador 2: 0 de 548 pares ordenados con
  frase, y 0 de 129.414 permutaciones de los conjuntos de 3 y 4 glosas con frase (§9.1).
- Validar: sí. Si el orden importara en algún caso, por ejemplo con tópico-comentario, la regla
  se mantiene y ese caso pasa a no cubierto (L-15).

**R-05 · Turno sin glosas del vocabulario**
- Condición: lista vacía, o ningún elemento quedó como glosa válida.
- Acción: frase `""`, vía `vacio`. El llamador no emite audio (`LSCTranslator._emitir_audio`;
  tts.py:645-660).
- Ejemplo: `[]`, `["xyz"]`, `[99]`, `[None]` → `""`.
- Fundamento: es el contrato actual (Apéndice A, punto 3). Decir en voz alta una cadena desconocida no es
  seguro. «Nunca vacío si hubo glosas» se entiende como «si hubo glosas **del vocabulario**».
- Validar: no.

**R-06 · Una sola glosa distinta**
- Condición: tras R-03 queda exactamente una glosa.
- Acción: enunciado de la columna «sola» de §2.3. Es idéntico en modo aislado y en un turno de una
  glosa.
- Ejemplo: `["comer"]` → «Comer.»; `["¿dónde?"]` → «¿Dónde?».
- Fundamento: §2.2 y el requisito 5(a) del contrato (Apéndice A): el texto que se precalienta en la voz es el
  mismo que se pide después.
- Validar: sí (D-L1).

**R-07 · Plantilla**
- Condición: el conjunto de glosas es exactamente la clave de una plantilla de §4.2 (paso A7). O
  bien, en las dos plantillas marcadas «admite saludo» (T-03 y T-04), el conjunto es esa clave más
  un solo saludo distinto de ADIÓS (paso B6). Con dos saludos serían 5 glosas, y R-08 lo manda antes
  a respaldo.
- Acción: enunciado fijo de la plantilla. Si se añadió un saludo: «Saludo, » + enunciado en
  minúscula.
- Ejemplo: POR FAVOR + SEÑA → «La seña, por favor.» (T-01, S-05). HOLA + TÚ + NOMBRE + ¿QUÉ? →
  «Hola, ¿cuál es tu nombre?» (T-03 con saludo, S-70).
- Fundamento: anteproyecto: «a cada glosa o a cada secuencia corta prevista se le asociará […] un
  enunciado fijo». Tiene precedencia sobre las reglas. Equivale, por analogía y con coincidencia
  exacta en lugar de por distancia, a la vía por ejemplos de San-Segundo et al. (2010, p. 251), un
  sistema para la LSE que busca la frase en un corpus de ejemplos y la acepta si la distancia al
  ejemplo más cercano está bajo un umbral. Esa vía está en la dirección español → LSE («4. Spanish
  into LSE translation»), la contraria a la de este conversor (glosa LSC → español): la analogía es
  solo de mecanismo. Un conjunto pequeño de plantillas no plantea problemas de
  mantenimiento (Reiter 1995, p. 4, nota 1).
- Validar: sí, cada plantilla por separado (§4.2).

**R-08 · Longitud máxima**
- Condición: más de **4** glosas distintas y sin plantilla.
- Acción: respaldo, motivo `demasiadas_glosas`.
- Ejemplo: HOLA + YO + TRABAJAR + MUCHO + BIEN → `hola, yo, trabajar, mucho, bien`.
- Fundamento: «secuencia corta prevista». Con 4 caben un saludo y un núcleo de 3 glosas
  («Hola, ¿dónde vive el hermano?»). Las 9 frases de ejemplo tienen 2 o 3 glosas.
- Validar: sí.

#### Etapa B — Periferia: saludos y cortesía

**R-09 · Saludo inicial**
- Condición: el conjunto contiene glosas `SAL`.
- Acción:
  - Saludos admitidos: uno solo, o HOLA + uno de {BUENOS DÍAS, BUENAS TARDES, BUENAS NOCHES}, que
    sale como «Hola, buenos días». Cualquier otra pareja de saludos, o tres, va a respaldo
    (`saludos_no_combinables`).
  - ADIÓS solo combina, por reglas, con TODOS (R-10) y con un vocativo (R-11). Con cualquier otro
    contenido va a respaldo (`despedida_con_contenido`). Por plantilla combina con GRACIAS (T-15).
  - Con contenido: «Saludo, » + núcleo. El núcleo empieza en minúscula, también una pregunta
    («Hola, ¿cómo estás?»).
- Ejemplo: HOLA + ¿CÓMO ESTÁS? → «Hola, ¿cómo estás?» (S-09). BUENOS DÍAS + FAMILIA + FELIZ →
  «Buenos días, la familia está feliz.» (S-69). HOLA + BUENOS DÍAS + ¿CÓMO? + USTEDES → «Hola,
  buenos días, ¿cómo están ustedes?» (S-71).
- Fundamento: DPD «coma» §2.2.1 (las interjecciones se aíslan con coma). El DLE clasifica «hola»
  como interjección, pero «buenos días» como expresión «para saludar». Que la coma tras «buenos
  días», «buenas tardes» y «buenas noches» siga la regla de la interjección es una **analogía del
  autor**, como en R-10. La apoya el uso del Diccionario básico, que traduce BUENAS-TARDES
  CÓMO-ESTÁ(?) como «Buenas tardes, ¿cómo estás?» (INSOR 2006, p. 488). DPD «signos de
  interrogación» §2.2: la pregunta empieza donde empieza y va en minúscula si no abre el enunciado.
  Con una oración declarativa detrás («Buenos días, la familia está feliz.») la alternativa es el
  punto («Buenos días. La familia está feliz.»), que se le muestra al profesor en B5.
- Validar: sí (qué saludos combinan, y coma o punto ante una declarativa).

**R-10 · Saludo + TODOS: destinatario con «a»**
- Condición: el conjunto es {saludo(s)} ∪ {TODOS}.
- Acción: «Saludo a todos.», sin coma.
- Ejemplo: BUENOS DÍAS + TODOS → «Buenos días a todos.». HOLA + TODOS → «Hola a todos.».
  ADIÓS + TODOS → «Adiós a todos.».
- Fundamento: DPD «coma» §2.2.1, ejemplo literal «¡Hola a todos!», sin coma entre la interjección y
  su complemento. Extenderlo a «buenos días», que el DLE clasifica como expresión, es una inferencia
  del autor. La alternativa vocativa sería «Buenos días, todos.» (DPD «coma» §2.2.3). Ninguna fuente
  consultada dice si «todos» funciona como vocativo.
- Validar: **sí**, «a todos» frente a «, todos».

**R-11 · Saludo o GRACIAS + sustantivo de persona: vocativo**
- Condición: el conjunto es {saludo(s)} ∪ {X}, con X de `NOM_PERS` y rasgo `vocativo` sí (familia,
  niño, niña, abuelo, tío, hermano). Incluye ADIÓS. Desde 3a-2, también {GRACIAS, X} sin saludo
  (paso C2).
- Acción: «Saludo, x.» o «Gracias, x.», sin artículo. HOLA + GRACIAS + X sigue en respaldo
  (`cortesia_fuera_de_plantilla`): no se compone un saludo con una cortesía.
- Ejemplo: HOLA + ABUELO → «Hola, abuelo.» (S-15; heredado: «Hola el abuelo»). ADIÓS + ABUELO →
  «Adiós, abuelo.» (S-85; heredado: «Adiós el abuelo»). GRACIAS + ABUELO → «Gracias, abuelo.» (S-91;
  heredado: «El abuelo gracias»).
- Fundamento: DPD «coma» §2.2.3: los vocativos van entre comas aun en enunciados muy breves. Que
  MUJER, HOMBRE y PERSONAS no se usen como vocativo («Hola, mujer») es una decisión del autor. Van a
  respaldo. ADIÓS + vocativo se admite desde el borrador 2: «Adiós, abuelo.» es tan natural como
  «Hola, abuelo.», y la versión anterior lo mandaba a respaldo sin razón declarada. GRACIAS + vocativo
  se admite desde 3a-2 por la misma razón: GRACIAS ya combinaba con TODOS (T-07) y con ADIÓS (T-15), y
  dejar «Gracias, abuelo.» en respaldo era una asimetría sin justificar en el texto (segunda revisión).
- Validar: sí (la lista de vocativos).

**R-12 · Saludo + fórmula suelta**
- Condición: el conjunto es {saludo} ∪ {BIENVENIDO} o {saludo} ∪ {¿CÓMO ESTÁS?}.
- Acción: «Saludo, bienvenido.» o «Saludo, ¿cómo estás?». Desde 3a-2, BIENVENIDO concuerda en singular
  con el género de quien escucha (`genero_segunda`, R-17): «Hola, bienvenida.» con
  `genero_segunda="f"`. Con el valor por defecto sale el masculino y se registra `genero_por_defecto`
  en `supuestos`, y `concordancia_adjetivo` en `suplido`.
- Ejemplo: HOLA + BIENVENIDO → «Hola, bienvenido.» (heredado: «Hola bienvenido»).
- Fundamento: el Diccionario básico pone BIENVENIDO entre las «Fórmulas de cortesía» (INSOR 2006,
  p. 486). La coma sigue la regla de R-09. El número del destinatario no se sabe y se elige el
  singular (L-08); el género sale del parámetro, que es el mismo que concuerda con TÚ.
- Validar: sí («bienvenido», «bienvenida» o «bienvenidos»).
- Cualquier otro {saludo} + una sola glosa → respaldo (`saludo_con_glosa_no_admitida`). Por
  ejemplo, HOLA + ¿DÓNDE? (el heredado da «Hola, ¿dónde?»), y también combinaciones naturales que se
  dejaron fuera por decisión de diseño: HOLA + TÚ, HOLA + USTEDES, HOLA + GRACIAS. Son candidatas de
  B4 (§9.2).

**R-13 · Cortesía solo por plantilla**
- Condición: el conjunto contiene `CORT` y no es una plantilla.
- Acción: respaldo, motivo `cortesia_fuera_de_plantilla`.
- Ejemplo: GRACIAS + YO + COMER → `gracias, yo, comer` (heredado: «Yo como gracias»).
  PERDÓN + YO + VIVIR → `perdón, yo, vivir` (heredado: «Yo vivo perdón»).
- Orden: va después de R-28 (paso C1 frente a C2). GRACIAS + SEÑA sale con
  `sustantivo_abstracto_fuera_de_plantilla`, no con este motivo.
- Fundamento: las fórmulas de cortesía no se componen libremente con una afirmación. El heredado
  las mandaba siempre al final, con resultados como «Yo vivo perdón» (N-42). Las combinaciones que
  el autor eligió están listadas como plantillas (T-01 a T-22) y, para GRACIAS + vocativo, en R-11.
  Criterio de 3a-2 para elegirlas: una combinación de cortesía entra si es **simétrica** de una
  que ya estaba (GRACIAS + BUENAS NOCHES → también BUENOS DÍAS y BUENAS TARDES, T-18 y T-19;
  GRACIAS + TODOS → también USTEDES y TÚ, T-20 y T-21; BIEN/MÁS O MENOS + GRACIAS → también MAL,
  T-22; ADIÓS/HOLA + vocativo → también GRACIAS, R-11). No son todas las naturales: «Hola, gracias» y
  «Perdón, abuelo» van a respaldo y se le muestran al profesor en B4.
- Validar: sí. ¿Falta alguna combinación de cortesía en §4.2?

#### Etapa C — Núcleo declarativo

**R-14 · Un sujeto, a lo sumo un elemento de cada clase**
- Condición: al conjunto sin la periferia lo llamamos núcleo. Cada comprobación va en el paso de
  §3.1 que se indica; el orden lo fija §3.1.
- Acción (cada línea: condición → código, ejemplo):
  - C3: más de un SUJ → `dos_sujetos`. HERMANO + TÍO + COMER → `hermano, tío, comer`. No se
    coordina ni se elige uno: el heredado descartaba el segundo en silencio (H5).
  - C6: declarativa sin SUJ → `sin_sujeto`. COMER + MUCHO → `comer, mucho`. No se inventa la
    persona (L-05). Como va antes que P1, FELIZ + TRISTE y COMER + TRABAJAR también salen con
    `sin_sujeto`. La pregunta sin sujeto es otro código (`pregunta_sin_sujeto`, R-30).
  - P1: dos glosas de la categoría VERBO → `dos_VERBO`. YO + COMER + TRABAJAR. Igual con dos ADJ
    (`dos_ADJ`, YO + FELIZ + TRISTE), dos ADV_EST (`dos_ADV_EST`, YO + BIEN + MAL) y dos ADV_CANT
    (`dos_ADV_CANT`, YO + MUCHO + POCO + COMER). Las demás categorías del predicado tienen una sola
    glosa. El heredado dejaba el segundo verbo en infinitivo suelto (H6).
  - P5: dos predicados de **distinta** categoría (ADJ, ADV_EST, NOM_TENER) → `dos_predicados`.
    YO + FELIZ + BIEN.
  - P6: un VERBO y SENTIR (`VERBO_SENTIR`) → `dos_verbos`. YO + SENTIR + TRABAJAR. Es decir:
    `dos_VERBO` = dos glosas de la categoría VERBO; `dos_verbos` = VERBO + VERBO_SENTIR.
  - P7: VERBO + (ADJ | NOM_TENER) → `verbo_con_predicado`. HERMANO + TRABAJAR + FELIZ. En cambio,
    VERBO + ADV_EST sí se admite (R-22): «La familia vive bien.».
  - P9: SUJ sin predicado ni verbo (por ejemplo, YO + NUNCA, o ¿POR QUÉ? + HERMANO) →
    `sin_predicado`.
- Fundamento: N-42, H5, H6 y H13; principio de que ninguna glosa se pierde en silencio.
- Validar: sí (sobre todo el sujeto nulo: COMER + MUCHO).

**R-15 · Artículo definido ante el sustantivo de persona**
- Condición: el sujeto es `NOM_PERS` y no es vocativo.
- Acción: se antepone el artículo concordado: el, la, los, las. Los pronombres y TODOS no llevan
  artículo. Registra `articulo` en `suplido` y `articulo_definido` en `supuestos` (se eligió el
  definido frente al posesivo o el indefinido, L-07).
- Ejemplo: PERSONAS + COMER → «Las personas comen.» (heredado: «Personas come»).
- Fundamento: concordancia nominal entre determinante y sustantivo (DPD «concordancia» §1a). Las
  traducciones del Diccionario básico añaden artículos que la glosa no trae (INSOR 2006, pp. 452 y
  490). **Falta la fuente normativa** sobre cuándo es obligatorio el artículo ante un sujeto
  nominal: la Nueva gramática no se consultó.
- Validar: **sí**, «el tío» frente a «mi tío».

**R-16 · Pronombre sujeto explícito**
- Condición: el sujeto es YO, TÚ o USTEDES.
- Acción: el pronombre se conserva. En una declarativa va delante del verbo y en una interrogativa
  parcial va al final (R-30). Con TÚ se registra `tratamiento_tu` en `supuestos` (L-09).
- Ejemplo: YO + TRABAJAR → «Yo trabajo.».
- Fundamento: Naranjo Orozco y Dieck (2022, p. 66) traducen YO TRABAJAR como «Yo trabajo». También
  se cumple que cada glosa señada quede visible en la frase. Costo conocido: el pronombre explícito,
  sobre todo pospuesto tras un complemento («Tú te sientes bien.», «¿Cómo te sientes tú?», «¿Por
  qué trabajas mucho tú?»), se puede oír como contraste, un significado que la seña no trae. Se
  mantiene hasta la Actividad B, donde se le muestra al profesor el par con y sin pronombre (B5).
  Es una **decisión de fidelidad**, no un descuido: en español el sujeto pronominal expreso es marcado
  y en voz puede sonar forzado («Yo trabajo.», «Tú estás bien.»), pero la seña YO o TÚ se hizo, y
  quitarla de la frase la haría invisible. Si el profesor no reconoce alguna de las preguntas
  pragmáticamente raras que resultan («¿Quién está más o menos?», «Buenos días, tú estás más o
  menos.»), esa combinación pasa a respaldo en el modo estricto (ya lo está: no es del inventario).
- Validar: sí («Yo trabajo» frente a «Trabajo»; «Tú te sientes bien.» frente a «Te sientes
  bien.»).

**R-17 · Persona, número y género del sujeto**
- Condición: siempre que hay sujeto.
- Acción: se toman de la tabla §2.3.
  - USTEDES: en el español de Colombia, *ustedes* concuerda en 3.ª plural y es la forma de 2.ª
    persona del plural; por eso no hay que elegir entre *ustedes* y *vosotros*. (La versión 3a-1
    decía «siempre, en todas las variedades», más de lo que se había consultado.)
  - TÚ: 2.ª singular de *tú*. La etiqueta «tú» de LSC50 no decide el tratamiento (§1.2); se elige
    *tú* por decisión del autor, pendiente de validar (L-09). Se registra `tratamiento_tu`.
  - YO y TÚ: el género no está en la glosa. Se toma de los parámetros `genero_primera` y
    `genero_segunda`, que valen `None` por defecto y entonces equivalen a `"m"` (§7.1). Cuando se usa
    el valor por defecto con un adjetivo que varía en género (contento, aburrido, juicioso,
    bienvenido), se registra `genero_por_defecto` en `supuestos`, con YO y con TÚ. Si el valor vino
    de un parámetro explícito (`"m"` o `"f"`), no se registra. Lo mismo vale para el destinatario de
    BIENVENIDO tras un saludo (R-12), que concuerda con `genero_segunda`.
  - USTEDES y TODOS: masculino plural. ¿QUIÉN? sin sujeto: masculino singular («¿Quién está
    contento?»). Es el masculino genérico para un grupo o un referente desconocido; no se registra
    `genero_por_defecto` porque no se elige el género de una persona identificada (la Nueva gramática,
    que es la fuente normativa del masculino genérico, no se consultó en esta actividad).
  - Glosa sola (vía `lexico`): la forma de cita es el masculino singular y no se registra (§2.2,
    punto 3).
  - FAMILIA: colectivo en singular.
- Ejemplo: YO + CONTENTO → «Yo estoy contento.» (supuesto `genero_por_defecto`). FAMILIA +
  CONTENTO → «La familia está contenta.».
- Fundamento: DPD «concordancia» §4.7: con un sujeto colectivo el verbo va en singular, y con
  atributo adjetivo el plural es incorrecto. El masculino por defecto es una decisión de diseño sin
  fuente consultada (L-08).
- Validar: sí (el género por defecto, y *tú* frente a *usted*/*vos*, L-09).

**R-18 · Concordancia del verbo con el sujeto**
- Condición: hay verbo, cópula o «tener».
- Acción: forma de la tabla §2.4 según persona y número.
- Ejemplo: USTEDES + COMER → «Ustedes comen.» (heredado: «Ustedes come»). TODOS + TRABAJAR + MUCHO
  → «Todos trabajan mucho.» (heredado: «Todos trabaja mucho»).
- Fundamento: DPD «concordancia» §1b. Corrige los 12 de 52 pares sujeto × verbo en desacuerdo
  (N-42, H1).
- Validar: no.

**R-19 · Presente de indicativo**
- Condición: toda forma verbal.
- Acción: siempre presente de indicativo. En toda salida por reglas con una forma verbal (verbo,
  cópula, «tener», «sentirse», «¿Cómo está…?») se registra `tiempo_verbal` en `suplido` y
  `tiempo_presente` en `supuestos`.
- Ejemplo: HERMANO + TRABAJAR → «El hermano trabaja.».
- Fundamento: sin marcas de anterioridad o posterioridad, el discurso en la LSC «tiende a
  interpretarse en el presente», y las autoras advierten que es «una tendencia general y no un
  principio absoluto» (Naranjo Orozco y Dieck 2022, p. 66). El léxico no tiene señas de tiempo:
  NUNCA es de frecuencia. Es un valor por defecto, no una traducción segura (L-02).
- Validar: no (es un límite declarado).

**R-20 · Cópula ser/estar ante adjetivo**
- Condición: SUJ (o ¿QUIÉN?) + ADJ, sin VERBO ni SENTIR.
- Acción: se inserta la cópula del rasgo `copula` del adjetivo, concordada (R-18), y el adjetivo
  concordado (R-21).
- Excepción `yo_bienvenido`: YO + BIENVENIDO **con cópula** → respaldo (paso P9). «Yo soy
  bienvenido» es gramatical, pero como autodeclaración resulta raro (juicio del autor). **Con SENTIR se admite**: YO + SENTIR +
  BIENVENIDO → «Yo me siento bienvenido.» (S-84), porque ahí no hay cópula (R-23). En el borrador 1
  la excepción se comprobaba antes de mirar si había SENTIR, y esa frase iba a respaldo.
- Excepción «fórmula» (desde 3a-2, paso C7a): un `NOM_PERS` con vocativo sí + BIENVENIDO, sin nada más
  en el núcleo → «Bienvenido, x.», con el adjetivo concordado con el sustantivo y sin artículo ni
  cópula: FAMILIA + BIENVENIDO → «Bienvenida, familia.» (S-92); con saludo, HOLA + FAMILIA + BIENVENIDO
  → «Hola, bienvenida, familia.». Registra `lectura_vocativa` en `supuestos`. Fundamento: BIENVENIDO es
  una fórmula de cortesía en el Diccionario básico (INSOR 2006, p. 486), como ya usa R-12, y esos mismos
  sustantivos van como vocativo tras un saludo (R-11). La lectura copulativa («La familia es
  bienvenida.») cambiaba el acto de habla: de dar la bienvenida a afirmar algo. La cópula se mantiene
  con MUJER, HOMBRE y PERSONAS (sin vocativo), con TÚ, USTEDES y TODOS («Tú eres bienvenido.», S-56;
  «Todos son bienvenidos.», S-55), con ¿QUIÉN? y con MUCHO («El tío es muy bienvenido.»). Con TÚ no se
  usa la fórmula («Bienvenido.») porque la seña TÚ desaparecería de la frase (R-16). Las dos lecturas
  se le muestran al profesor en B5 (15).

  | adjetivo | cópula | criterio (DPD «estar» §2) | ambigüedad a mostrar |
  |---|---|---|---|
  | feliz | estar | estado de ánimo, no permanente | «es feliz» = carácter o vida feliz; también es posible |
  | contento | estar | estado | «ser contento» no se usa en el sentido de estado (juicio del autor) |
  | triste | estar | estado | «es triste» = carácter, o algo que causa tristeza |
  | aburrido | estar | estado (sentir aburrimiento) | «es aburrido» = que aburre: el sentido cambia (inferencia del autor con el criterio del DPD) |
  | juicioso | ser | rasgo estable del carácter | en Colombia «está juicioso» = se porta bien ahora (no verificado en fuente) |
  | diferente | ser | clasifica al sujeto | «está diferente» = cambió |
  | bienvenido | ser | fórmula «ser bienvenido» | — |
- Ejemplo: TÍO + JUICIOSO → «El tío es juicioso.». FAMILIA + FELIZ → «La familia está feliz.».
  El heredado no ponía cópula en ninguno de los 91 pares sujeto × adjetivo (N-42, H3).
- Fundamento: la LSC no expresa léxicamente la cópula (Naranjo Orozco y Dieck 2022, p. 66, nota
  12), así que el español la tiene que suplir. El DPD (s. v. «estar», §2) da el criterio
  (permanente o inherente frente a resultado de un cambio o no permanente) y advierte que el sentido
  puede cambiar («es listo» / «está listo»). Ninguna fuente dice qué cópula toca a cada uno de
  estos 7 adjetivos: la tabla es del autor.
- Validar: **sí, adjetivo por adjetivo**.

**R-21 · Concordancia del adjetivo**
- Condición: hay un adjetivo predicativo, con cópula o con «sentirse».
- Acción: forma de §2.4 según el género y el número del sujeto (R-17).
- Ejemplo: NIÑA + CONTENTO → «La niña está contenta.» (heredado: «La niña contento», H2).
  PERSONAS + FELIZ → «Las personas están felices.».
- Fundamento: DPD «concordancia» §1a (sujeto con atributo).
- Validar: no.

**R-22 · Adverbio de estado: predicado con «estar» o modo del verbo**
- Condición: el núcleo tiene `ADV_EST` (bien, mal, más o menos).
- Acción:
  - Sin verbo: SUJ + «estar» + adverbio.
  - Con un verbo que tenga `modo` sí (trabajar, comer, vivir): el adverbio va pospuesto al verbo.
    Si además hay un cuantificador, decide R-25 (fila «verbo + ADV_EST»).
- Ejemplo: YO + BIEN → «Yo estoy bien.» (heredado: «Yo bien»). FAMILIA + VIVIR + BIEN → «La
  familia vive bien.».
- Fundamento: en la LSC no hay cópula léxica (R-20). La pregunta ¿CÓMO ESTÁS? del léxico se
  responde con «estar».
- Validar: sí («La familia está más o menos»).

**R-23 · SENTIR: «sentirse» + predicativo**
- Condición: el núcleo tiene SENTIR.
- Acción:
  - Exige un predicativo: un `ADJ` con `con_sentir` sí, o un `ADV_EST`. Sale «me/te/se siento…» +
    predicativo concordado. Registra `pronombre_reflexivo` en `suplido`.
  - Sin predicativo → respaldo (`sentir_sin_predicativo`).
  - Con HAMBRE → respaldo (`sentir_hambre`).
  - Con JUICIOSO → respaldo (`adjetivo_no_admite_sentir`).
  - El rasgo `copula` del adjetivo **no se aplica** con SENTIR, porque no hay cópula. Por eso se
    admiten DIFERENTE y BIENVENIDO, que con cópula llevan *ser*: YO + SENTIR + BIENVENIDO → «Yo me
    siento bienvenido.» (S-84); YO + NUNCA + SENTIR + DIFERENTE → «Yo nunca me siento diferente.»
    (S-75). Tampoco se aplica `yo_bienvenido` (R-20).
- Ejemplo: YO + SENTIR + TRISTE → «Yo me siento triste.» (S-51; heredado: «Yo siento triste»). YO +
  SENTIR → `yo, sentir`.
- Fundamento: es la construcción española con predicativo («sentirse» pronominal). No se consultó
  fuente para esta regla: es juicio del autor. El caso «YO SENTIR» solo es ambiguo: puede ser «lo
  siento», «siento algo» o «me siento…», y el conversor no puede elegir. «Lo siento» es un
  enunciado natural que queda en respaldo; es candidato de B4.
- Validar: sí.

**R-24 · HAMBRE: «tener hambre»**
- Condición: SUJ + HAMBRE, sin verbo.
- Acción: SUJ + «tener» concordado + «hambre». Con MUCHO o POCO: «mucha/poca hambre», concordado
  con el femenino de *hambre*.
- Ejemplo: YO + HAMBRE → «Yo tengo hambre.» (heredado: «Yo la hambre», H4). NIÑO + HAMBRE + MUCHO
  → «El niño tiene mucha hambre.».
- Registra `verbo_soporte` en `suplido`.
- Fundamento: en español el estado se expresa con «tener hambre». Como no hay artículo, desaparece
  el problema de «el/la hambre». La concordancia en femenino («mucha hambre», «poca hambre») es
  juicio del autor: no se consultó fuente. La entrada «hambre» del DPD es la fuente que corresponde
  y queda pendiente de consultar antes de citar.
- Validar: sí.

**R-25 · Cuantificadores MUCHO y POCO**
- Condición: el núcleo tiene `ADV_CANT`.
- Acción:

  | modifica a | MUCHO | POCO |
  |---|---|---|
  | verbo con `cant` sí (trabajar, comer), sin ADV_EST | pospuesto: «trabaja mucho» | pospuesto: «come poco» |
  | **verbo + ADV_EST** (cualquier verbo con `modo` sí) | con BIEN o MAL: verbo + «muy» + adverbio («trabaja muy bien»). Con MÁS O MENOS → `cantidad_y_modo` | → `cantidad_y_modo` |
  | VIVIR sin ADV_EST | → `cantidad_no_admitida` | → `cantidad_no_admitida` |
  | adjetivo (con cópula o con SENTIR) | «muy» + adjetivo | → `cantidad_no_admitida` |
  | BIEN, MAL sin verbo (con «estar» o con SENTIR) | «muy bien», «muy mal» | → `cantidad_no_admitida` |
  | MÁS O MENOS sin verbo | → `cantidad_no_admitida` | → `cantidad_no_admitida` |
  | HAMBRE | «mucha hambre» | «poca hambre» |
  | con NUNCA | → `nunca_con_cantidad` (R-26) | → `nunca_con_cantidad` |
- **Prioridad con verbo + ADV_EST + MUCHO:** manda la fila «verbo + ADV_EST». MUCHO intensifica al
  adverbio y no cuantifica al verbo: TÍO + TRABAJAR + BIEN + MUCHO → «El tío trabaja muy bien.»
  (S-83), nunca «trabaja bien mucho». Con POCO no hay lectura de intensificador y va a
  `cantidad_y_modo`: TÍO + TRABAJAR + BIEN + POCO → `tío, trabajar, bien, poco`.
- Registra `intensificador` en `suplido` cuando sale «muy».
- Ejemplo: YO + MUCHO + FELIZ → «Yo estoy muy feliz.» (S-50; heredado: «Yo feliz mucho»). HERMANO
  + COMER + POCO → «El hermano come poco.» (S-76). YO + HAMBRE + POCO → «Yo tengo poca hambre.»
  (S-77).
- Fundamento: es la forma española del intensificador ante un adjetivo o un adverbio. Que la seña
  MUCHO funcione como intensificador es una inferencia del autor. «Poco» con adjetivo («poco
  feliz») se excluyó porque en el borrador daba frases poco naturales («poco bienvenido»).
- Validar: sí.

**R-26 · NUNCA antes del verbo**
- Condición: el núcleo tiene NUNCA.
- Acción:
  - «nunca» va justo antes del verbo, la cópula o «me/te/se».
  - Solo se admite con: verbo con `nunca` sí (trabajar, comer) sin predicado ni cuantificador; ADJ
    con cópula *estar*; BIEN, MAL con «estar»; HAMBRE; SENTIR + predicativo (cualquier ADJ con
    `con_sentir` sí, BIEN o MAL).
  - La restricción de *ser* solo vale con cópula. Con SENTIR no hay cópula y se admiten DIFERENTE y
    BIENVENIDO (R-23).
  - Respaldo, cada caso con su código: NUNCA + MUCHO/POCO → `nunca_con_cantidad` (P2); NUNCA con
    sujeto TODOS → `todos_con_nunca` (P3); NUNCA + VERBO + predicado → `nunca_con_modo` (P4);
    NUNCA + VIVIR, NUNCA + ADJ con *ser* sin SENTIR, NUNCA + MÁS O MENOS → `nunca_no_admitido` (P7,
    P8, P9).
- Ejemplo: YO + NUNCA + COMER → «Yo nunca como.» (S-42). TÍO + ABURRIDO + NUNCA → «El tío nunca está
  aburrido.» (S-73). YO + NUNCA + HAMBRE → «Yo nunca tengo hambre.» (S-74). YO + NUNCA + SENTIR +
  DIFERENTE → «Yo nunca me siento diferente.» (S-75). TÍO + NUNCA + JUICIOSO → `tío, nunca,
  juicioso`.
- Fundamento: posición preverbal del adverbio de negación de frecuencia, la misma que da el
  heredado. Las exclusiones son del autor, sobre salidas del borrador como «Todos nunca son
  juiciosos».
- Validar: sí.

**R-27 · VIVIR exige complemento**
- Condición: VIVIR **sin adverbio de estado (ADV_EST) y sin ¿DÓNDE? ni ¿CÓMO?**. Vale con sujeto y
  sin él, en declarativa y en interrogativa.
- Acción: respaldo (`vivir_sin_complemento`, paso P7).
- Ejemplo: YO + VIVIR → `yo, vivir` (heredado: «Yo vivo»). ¿POR QUÉ? + HERMANO + VIVIR → `¿por
  qué?, hermano, vivir` (heredado: «¿Por qué vive el hermano?», que se entiende como «por qué sigue
  vivo»). ¿QUIÉN? + VIVIR → `¿quién?, vivir` (heredado: «¿Quién vive?», fórmula de centinela o
  pregunta existencial). En cambio, ¿DÓNDE? + HERMANO + VIVIR → «¿Dónde vive el hermano?» (S-08), y
  ¿CÓMO? + HERMANO + VIVIR → «¿Cómo vive el hermano?». ¿CUÁNDO? y ¿QUÉ? con VIVIR no están en la tabla
  de R-30 y salen con `pregunta_no_admitida`.
- Fundamento: «Yo vivo» es gramatical, pero significa «estoy vivo». Lo probable es «vivo en…», con
  el lugar sin señar o señado en el espacio. El léxico no tiene señas de lugar. El borrador ya
  mandaba a respaldo ¿POR QUÉ? + SUJ + VIVIR y ¿QUIÉN? + VIVIR. La condición escrita antes («sin
  interrogativo») no lo decía. Recuento en pares: 28 = 13 sujetos × 2 órdenes con VIVIR + 2 de
  ¿QUIÉN? + VIVIR.
- Validar: sí.

**R-28 · SEÑA y NOMBRE solo por plantilla**
- Condición: el núcleo tiene `NOM_ABS` y no es una plantilla.
- Acción: respaldo (`sustantivo_abstracto_fuera_de_plantilla`). Es la primera comprobación del
  núcleo (paso C1), antes que la cortesía (R-13), los sujetos y los interrogativos. Por eso
  GRACIAS + SEÑA, ¿QUÉ? + NOMBRE y COMER + NOMBRE salen con este código. Recuento en pares: 170 =
  85 conjuntos × 2 órdenes.
- Ejemplo: YO + NOMBRE → `yo, nombre` (heredado: «Yo el nombre»).
- Fundamento: ninguno de los 4 verbos los toma como complemento. «YO NOMBRE» suele seguir con un
  nombre deletreado, que el prototipo no reconoce (L-10).
- Validar: no.

#### Etapa D — Núcleo interrogativo

**R-29 · Solo hay pregunta si hay glosa interrogativa**
- Condición: siempre.
- Acción: el enunciado es interrogativo solo si contiene una glosa `INTERR` o `PREG_FIJA`. Con dos
  o más → respaldo (`dos_interrogativos`, paso C4). Una declarativa nunca se vuelve pregunta sí/no.
  Va antes que cualquier otra comprobación de la pregunta, y después de C1 a C3. Por eso ¿QUÉ? +
  ¿DÓNDE? sale con este código y no con `dos_INTERR`, que no existe. Recuento en pares: 42 = 7 × 6
  pares ordenados entre las 7 glosas interrogativas (6 INTERR + ¿CÓMO ESTÁS?).
- Ejemplo: ¿QUÉ? + ¿DÓNDE? + TÚ + VIVIR → `¿qué?, ¿dónde?, tú, vivir` (heredado: «¿Qué vives?»,
  H8).
- Fundamento: en otras lenguas de señas, las preguntas sí/no se marcan más a menudo solo con
  rasgos no manuales (Zeshan 2006, p. 40, en una muestra sin la LSC). El prototipo no los capta (L-01).
- Validar: no (es un límite declarado).

**R-30 · Estructura de la pregunta parcial**
- Condición: un `INTERR` distinto de ¿QUIÉN?, más un SUJ, más el resto.
- Acción:
  - Salida: «¿» + interrogativo + [nunca] + verbo concordado + complementos + **sujeto** + «?».
    El sujeto va al final, con artículo si es sustantivo y explícito si es pronombre.
  - Combinaciones admitidas (las demás → respaldo, `pregunta_no_admitida`):

    | interrogativo | resto admitido (exacto) | YO admitido |
    |---|---|---|
    | ¿DÓNDE? | TRABAJAR, COMER o VIVIR | no |
    | ¿CUÁNDO? | TRABAJAR o COMER | no |
    | ¿QUÉ? | COMER | no |
    | ¿CÓMO? | nada («¿Cómo está…?»), TRABAJAR, COMER, VIVIR o SENTIR («¿Cómo se siente…?») | no |
    | ¿POR QUÉ? | cualquier predicado válido para R-15 a R-27 (pasos P1 a P9) | sí |
  - Códigos: sin SUJ → `pregunta_sin_sujeto` (paso D3; va antes que el `sin_sujeto` de la
    declarativa). Con ¿DÓNDE?, ¿CUÁNDO?, ¿QUÉ? o ¿CÓMO?: YO, o un resto que no está en la tabla →
    `pregunta_no_admitida`. Con ¿POR QUÉ?: SUJ solo → `sin_predicado`; VIVIR sin ADV_EST →
    `vivir_sin_complemento` (R-27); el resto, los códigos de P1 a P9.
- Ejemplo: ¿DÓNDE? + HERMANO + VIVIR → «¿Dónde vive el hermano?» (S-08). ¿POR QUÉ? + NIÑO + TRISTE
  → «¿Por qué está triste el niño?» (S-66; heredado: «¿Por qué el niño triste?»). ¿POR QUÉ? +
  USTEDES + TRABAJAR → «¿Por qué trabajan ustedes?» (heredado: «¿Por qué trabaja ustedes?», H10).
  ¿POR QUÉ? + TÚ + SENTIR + TRISTE → «¿Por qué te sientes triste tú?» (S-80). ¿POR QUÉ? + NIÑO +
  HAMBRE → «¿Por qué tiene hambre el niño?» (S-81). ¿POR QUÉ? + HERMANO + VIVIR → respaldo (R-27).
- Posición del pronombre: con un pronombre, el sujeto al final da órdenes marcados («¿Por qué
  trabajas mucho tú?», S-67; «¿Por qué estoy mal yo?»). Las alternativas son el pronombre justo
  detrás del verbo («¿Por qué trabajas tú mucho?») u omitirlo («¿Por qué trabajas mucho?»); y, con
  ¿POR QUÉ?, «tanto» en lugar de «mucho» («¿Por qué trabajas tanto?»). No se cambia la regla antes
  de la Actividad B: se le muestran las variantes al profesor en B5.
- Fundamento: la posición del interrogativo en la seña no sirve como señal: en el Diccionario básico
  aparece al final, al principio o en los dos sitios (INSOR 2006, pp. 480, 482 y 488). Según Naranjo
  Orozco y Dieck (2022, p. 65, nota 11), «Las unidades interrogativas en la LSC merecen un estudio en
  profundidad, que no se ha emprendido aún» (cita tomada de la verificación del 8-nov, que leyó el PDF).
  Por eso el orden de salida lo fija la regla. Los signos ¿…? siguen DPD «signos de interrogación»
  §2.1: el de apertura no se suprime.
- Validar: **sí** (el sujeto al final, «¿Dónde vives tú?» frente a «¿Dónde vives?», y la tabla de
  combinaciones).

**R-31 · ¿QUIÉN?**
- Condición: el núcleo tiene ¿QUIÉN?.
- Acción:
  - Sin sujeto, con un predicado válido: «¿Quién» + predicado en 3.ª singular + «?». Un adjetivo
    va en masculino singular y no registra `genero_por_defecto` (R-17).
  - Con un solo SUJ y nada más, cuando el SUJ tiene el rasgo `quien` sí (tú, ustedes, personas,
    mujer, hombre, niño, niña, abuelo, tío, hermano): «¿Quién es…?», o «¿Quiénes son…?» si el SUJ
    es plural.
  - Con sujeto y predicado a la vez → `quien_con_sujeto_y_predicado`. Con YO, TODOS o FAMILIA como
    único SUJ → `quien_sujeto_no_admitido`.
  - Sin sujeto, el predicado pasa por P1 a P9. En particular, ¿QUIÉN? + VIVIR → `vivir_sin_complemento`
    (R-27), y ¿QUIÉN? + NUNCA, MUCHO o POCO → `sin_predicado`.
- Ejemplo: ¿QUIÉN? + HAMBRE → «¿Quién tiene hambre?» (S-61; heredado: «¿Quién la hambre?»).
  ¿QUIÉN? + USTEDES → «¿Quiénes son ustedes?» (S-63). ¿QUIÉN? + FELIZ → «¿Quién está feliz?»
  (S-78). ¿QUIÉN? + MAL → «¿Quién está mal?» (S-79). ¿QUIÉN? + VIVIR → `¿quién?, vivir`.
- Registra `copula` en `suplido` en «¿Quién es…?» / «¿Quiénes son…?».
- Fundamento: concordancia de número del interrogativo con el atributo plural.
- Validar: sí.

**R-32 · ¿CÓMO ESTÁS? + sujeto**
- Condición: {¿CÓMO ESTÁS?, SUJ} y nada más en el núcleo.
- Acción: «¿Cómo» + «estar» concordado + sujeto + «?». Sin sujeto, con YO, o con más glosas →
  respaldo (`como_estas_no_admitido`, paso D1; va antes que el `sin_sujeto` de la declarativa).
- No registra `copula`: «está» es la forma concordada del verbo que ya trae la glosa («estás»), no una
  cópula suplida. En cambio, ¿CÓMO? + SUJ (R-30) sí la registra, aunque la frase sea la misma.
- Lectura elegida (desde 3a-2): con un sujeto que tiene vocativo sí (familia, niño, niña, abuelo, tío,
  hermano) cabe también la lectura vocativa, «Hermano, ¿cómo estás?», que conserva la 2.ª persona de
  la glosa. Se mantiene la 3.ª persona y se registra `lectura_tercera_persona` en `supuestos` (48 de
  los 6.995 conjuntos con frase, medido). Razones: el Diccionario básico glosa CÓMO-ESTÁ(?) seguido de
  personas (INSOR 2006, p. 488), y con FAMILIA la lectura vocativa obligaría a cambiar el número
  («Familia, ¿cómo están?»), con lo que tampoco conservaría la glosa. Ninguna de las dos lecturas está
  validada: se le muestran al profesor en B5 (14).
- Ejemplo: ¿CÓMO ESTÁS? + FAMILIA → «¿Cómo está la familia?». ¿CÓMO ESTÁS? + USTEDES → «¿Cómo
  están ustedes?» (S-18). ¿CÓMO ESTÁS? + TÚ → «¿Cómo estás tú?» (S-82). El heredado da «¿Cómo
  estás?» en los tres casos. ¿CÓMO ESTÁS? + COMER → `¿cómo estás?, comer`.
- Fundamento: el Diccionario básico glosa CÓMO-ESTÁ(?) seguido de personas (INSOR 2006, p. 488).
  Consecuencia: ¿CÓMO? + SUJ y ¿CÓMO ESTÁS? + SUJ dan la misma frase. Son las 96 coincidencias
  medidas en §9.1. Se aceptan porque las dos secuencias preguntan lo mismo.
- Validar: sí.

#### Etapa E — Salida

**R-33 · Mayúscula inicial**
- Condición: toda salida por reglas.
- Acción: mayúscula en la primera letra, o en la que sigue a un «¿» inicial. Una pregunta que va
  detrás de un saludo queda en minúscula.
- Ejemplo: «¿Dónde vive el hermano?»; «Hola, ¿cómo está la familia?».
- Fundamento: DPD «mayúsculas» §3.1 y §3.3.1; DPD «signos de interrogación» §2.2. Corrige H9 de
  N-42 (el «¿» interno).
- Validar: no.

**R-34 · Cierre**
- Condición: toda salida por reglas.
- Acción: si el enunciado termina en «?», no se añade nada. Si no, punto final. No se usan «¡…!».
- Ejemplo: «El tío es juicioso.»; «¿Quién come?».
- Fundamento: DPD «punto» §2; DPD «signos de interrogación» §3.1 (no se pone punto tras el signo de
  cierre).
- Validar: sí (la ausencia de exclamación, D-L1).

**R-35 · Respaldo**
- Condición: se cumplió una condición de respaldo (§5.2).
- Acción: las glosas válidas distintas, en el orden de primera aparición y escritas como en
  `SIGN_VOCABULARY`, unidas por «, »: exactamente `", ".join(glosas[:8])`, y `omitidas ==
  glosas[8:]` (§5.3). **Sin mayúscula inicial y sin punto final**: es una transcripción, no una
  frase. Se marca `respaldo = True` con su `motivo`.
- Ejemplo: HERMANO + TÍO + COMER → `hermano, tío, comer`.
- Fundamento: es el requisito de seguridad de la tarea: no vacío si hubo glosas y nunca una frase
  agramatical presentada como frase.
- Validar: sí. ¿Se dice en voz alta? Lo propuesto es que sí, porque la persona oyente oye al menos
  las glosas.

---

## 4. Secuencias cortas previstas

### 4.1 Cómo se eligieron

El inventario es **cerrado**: lo que no aparece aquí sale por las reglas de §3 o por el respaldo.
Se formó con cuatro fuentes, en este orden:

1. Las **9 frases de ejemplo** ya usadas con el autor (S-01 a S-09).
2. Las fórmulas del Diccionario básico que el léxico permite formar: saludo + ¿CÓMO ESTÁS?
   (INSOR 2006, p. 488) y las «Fórmulas de cortesía» (pp. 485-490).
3. Las combinaciones de cortesía que forman enunciados habituales en español y que las reglas no
   cubren, porque la cortesía no se compone (R-13). Son las plantillas T-01 a T-17, más T-18 a T-22
   añadidas en 3a-2 por simetría con las anteriores (R-13). Las eligió el autor y no tienen fuente de
   uso: son las primeras que se le muestran al profesor.
4. Ejemplos de los patrones de reglas de §3, para que cada rama que produce frase quede fijada por
   una salida exacta. La primera versión (68 secuencias) no cubría todas las ramas. Se añadieron
   S-69 a S-85 para cubrir las que faltaban: saludo + núcleo declarativo, plantilla con saludo, dos
   saludos + núcleo, NUNCA con ADJ, HAMBRE y SENTIR, POCO con verbo y con HAMBRE, ¿QUIÉN? + ADJ y +
   ADV_EST, ¿POR QUÉ? + SENTIR y + HAMBRE, ¿CÓMO ESTÁS? + TÚ, verbo + BIEN + MUCHO, YO + SENTIR +
   BIENVENIDO y ADIÓS + vocativo. En 3a-2 se añadieron S-86 a S-92: una por plantilla nueva y una
   por cada rama nueva (GRACIAS + vocativo, BIENVENIDO con vocativo). No se afirma que estén cubiertas
   todas las combinaciones de ramas: el criterio es una secuencia por rama que produce frase, revisado
   a mano contra §3.1, no medido. Toda plantilla tiene su secuencia en el inventario (lo comprueban
   las pruebas), así que en el modo estricto todas dan frase.

Cada secuencia trae su enunciado fijo. En las que salen por reglas, la prueba exige que las reglas
den exactamente ese texto. El orden de las glosas en la tabla es solo de presentación (R-04).

### 4.2 Plantillas fijas (vía `plantilla`)

| id | glosas | enunciado fijo | admite saludo | nota para validar |
|---|---|---|---|---|
| T-01 | POR FAVOR + SEÑA | La seña, por favor. | no | ¿Pide que se haga la seña, o pregunta la seña personal? Sentido incierto |
| T-02 | POR FAVOR + NOMBRE | El nombre, por favor. | no | Suena a ventanilla. Alternativas: «Tu nombre, por favor.» o «Su nombre, por favor.» (B5) |
| T-03 | TÚ + NOMBRE + ¿QUÉ? | ¿Cuál es tu nombre? | sí | Alternativa: «¿Cómo te llamas?» |
| T-04 | TÚ + SEÑA + ¿QUÉ? | ¿Cuál es tu seña? | sí | Seña personal |
| T-05 | GRACIAS + MUCHO | Muchas gracias. | no | — |
| T-06 | CON GUSTO + MUCHO | Con mucho gusto. | no | — |
| T-07 | GRACIAS + TODOS | Gracias a todos. | no | Por analogía con «Hola a todos» (R-10) |
| T-08 | PERDÓN + PERMISO | Perdón, con permiso. | no | — |
| T-09 | POR FAVOR + PERMISO | Permiso, por favor. | no | — |
| T-10 | PERDÓN + ¿QUÉ? | Perdón, ¿qué? | no | Pedir que se repita |
| T-11 | PERDÓN + ¿CÓMO? | Perdón, ¿cómo? | no | Pedir que se repita |
| T-12 | BIEN + GRACIAS | Bien, gracias. | no | Respuesta a ¿CÓMO ESTÁS? |
| T-13 | MÁS O MENOS + GRACIAS | Más o menos, gracias. | no | — |
| T-14 | YO + BIEN + GRACIAS | Yo estoy bien, gracias. | no | — |
| T-15 | GRACIAS + ADIÓS | Gracias, adiós. | no | — |
| T-16 | GRACIAS + BUENAS NOCHES | Gracias, buenas noches. | no | Despedida |
| T-17 | MUCHO + GRACIAS + TODOS | Muchas gracias a todos. | no | — |
| T-18 | GRACIAS + BUENOS DÍAS | Gracias, buenos días. | no | 3a-2. Simetría con T-16. Despedida o saludo de cortesía |
| T-19 | GRACIAS + BUENAS TARDES | Gracias, buenas tardes. | no | 3a-2. Simetría con T-16 |
| T-20 | GRACIAS + USTEDES | Gracias a ustedes. | no | 3a-2. Simetría con T-07 |
| T-21 | GRACIAS + TÚ | Gracias a ti. | no | 3a-2. Simetría con T-07. Elige *tú* (registra `tratamiento_tu`); alternativas «Gracias a usted», «Gracias a vos» (L-09) |
| T-22 | MAL + GRACIAS | Mal, gracias. | no | 3a-2. Simetría con T-12 y T-13. Respuesta a ¿CÓMO ESTÁS? |

### 4.3 Inventario de secuencias previstas (92)

«Reglas» indica las que intervienen (las de entrada R-01 a R-06 y el cierre R-33/R-34 no se
listan). «Heredado» es la salida actual, medida hoy.

| S | glosas | enunciado fijo | vía | reglas / plantilla | heredado |
|---|---|---|---|---|---|
| S-01 | BUENOS DÍAS + TODOS | Buenos días a todos. | reglas | R-09, R-10 | Buenos días todos |
| S-02 | BUENAS TARDES + TODOS | Buenas tardes a todos. | reglas | R-09, R-10 | Buenas tardes todos |
| S-03 | TÍO + JUICIOSO | El tío es juicioso. | reglas | R-15, R-18, R-20, R-21 | El tío juicioso |
| S-04 | FAMILIA + FELIZ | La familia está feliz. | reglas | R-15, R-17, R-18, R-20, R-21 | La familia feliz |
| S-05 | POR FAVOR + SEÑA | La seña, por favor. | plantilla | T-01 | La seña por favor |
| S-06 | HOLA + BIENVENIDO | Hola, bienvenido. | reglas | R-09, R-12 | Hola bienvenido |
| S-07 | HERMANO + TRABAJAR | El hermano trabaja. | reglas | R-15, R-18, R-19 | El hermano trabaja |
| S-08 | ¿DÓNDE? + HERMANO + VIVIR | ¿Dónde vive el hermano? | reglas | R-15, R-18, R-30 | ¿Dónde vive el hermano? |
| S-09 | HOLA + ¿CÓMO ESTÁS? | Hola, ¿cómo estás? | reglas | R-09, R-12 | Hola, ¿cómo estás? |
| S-10 | HOLA + TODOS | Hola a todos. | reglas | R-09, R-10 | Hola todos |
| S-11 | BUENAS NOCHES + TODOS | Buenas noches a todos. | reglas | R-09, R-10 | Buenas noches todos |
| S-12 | ADIÓS + TODOS | Adiós a todos. | reglas | R-09, R-10 | Adiós todos |
| S-13 | HOLA + BUENOS DÍAS | Hola, buenos días. | reglas | R-09 | Hola, buenos días |
| S-14 | BUENAS TARDES + ¿CÓMO ESTÁS? | Buenas tardes, ¿cómo estás? | reglas | R-09, R-12 | Buenas tardes, ¿cómo estás? |
| S-15 | HOLA + ABUELO | Hola, abuelo. | reglas | R-09, R-11 | Hola el abuelo |
| S-16 | BUENOS DÍAS + FAMILIA | Buenos días, familia. | reglas | R-09, R-11 | Buenos días la familia |
| S-17 | HOLA + ¿CÓMO ESTÁS? + FAMILIA | Hola, ¿cómo está la familia? | reglas | R-09, R-15, R-18, R-32 | Hola, ¿cómo estás? |
| S-18 | ¿CÓMO ESTÁS? + USTEDES | ¿Cómo están ustedes? | reglas | R-18, R-32 | ¿Cómo estás? |
| S-19 | ¿CÓMO ESTÁS? + TODOS | ¿Cómo están todos? | reglas | R-18, R-32 | ¿Cómo estás? |
| S-20 | BUENOS DÍAS + BIENVENIDO | Buenos días, bienvenido. | reglas | R-09, R-12 | Buenos días bienvenido |
| S-21 | POR FAVOR + NOMBRE | El nombre, por favor. | plantilla | T-02 | El nombre por favor |
| S-22 | TÚ + NOMBRE + ¿QUÉ? | ¿Cuál es tu nombre? | plantilla | T-03 | ¿Qué el nombre? |
| S-23 | TÚ + SEÑA + ¿QUÉ? | ¿Cuál es tu seña? | plantilla | T-04 | ¿Qué la seña? |
| S-24 | GRACIAS + MUCHO | Muchas gracias. | plantilla | T-05 | Mucho gracias |
| S-25 | CON GUSTO + MUCHO | Con mucho gusto. | plantilla | T-06 | Mucho con gusto |
| S-26 | GRACIAS + TODOS | Gracias a todos. | plantilla | T-07 | Todos gracias |
| S-27 | PERDÓN + PERMISO | Perdón, con permiso. | plantilla | T-08 | El permiso perdón |
| S-28 | POR FAVOR + PERMISO | Permiso, por favor. | plantilla | T-09 | El permiso por favor |
| S-29 | PERDÓN + ¿QUÉ? | Perdón, ¿qué? | plantilla | T-10 | ¿Qué? |
| S-30 | PERDÓN + ¿CÓMO? | Perdón, ¿cómo? | plantilla | T-11 | ¿Cómo? |
| S-31 | BIEN + GRACIAS | Bien, gracias. | plantilla | T-12 | Bien gracias |
| S-32 | MÁS O MENOS + GRACIAS | Más o menos, gracias. | plantilla | T-13 | Más o menos gracias |
| S-33 | YO + BIEN + GRACIAS | Yo estoy bien, gracias. | plantilla | T-14 | Yo bien gracias |
| S-34 | GRACIAS + ADIÓS | Gracias, adiós. | plantilla | T-15 | Adiós gracias |
| S-35 | GRACIAS + BUENAS NOCHES | Gracias, buenas noches. | plantilla | T-16 | Buenas noches gracias |
| S-36 | MUCHO + GRACIAS + TODOS | Muchas gracias a todos. | plantilla | T-17 | Todos mucho gracias |
| S-37 | YO + TRABAJAR | Yo trabajo. | reglas | R-16, R-18, R-19 | Yo trabajo |
| S-38 | TÚ + COMER | Tú comes. | reglas | R-16, R-18, R-19 | Tú comes |
| S-39 | USTEDES + TRABAJAR | Ustedes trabajan. | reglas | R-16, R-17, R-18, R-19 | Ustedes trabaja |
| S-40 | PERSONAS + COMER | Las personas comen. | reglas | R-15, R-18, R-19 | Personas come |
| S-41 | TODOS + TRABAJAR + MUCHO | Todos trabajan mucho. | reglas | R-18, R-19, R-25 | Todos trabaja mucho |
| S-42 | YO + NUNCA + COMER | Yo nunca como. | reglas | R-16, R-18, R-26 | Yo nunca como |
| S-43 | FAMILIA + VIVIR + BIEN | La familia vive bien. | reglas | R-15, R-18, R-22 | La familia vive bien |
| S-44 | YO + FELIZ | Yo estoy feliz. | reglas | R-16, R-18, R-20, R-21 | Yo feliz |
| S-45 | NIÑA + CONTENTO | La niña está contenta. | reglas | R-15, R-18, R-20, R-21 | La niña contento |
| S-46 | USTEDES + ABURRIDO | Ustedes están aburridos. | reglas | R-17, R-18, R-20, R-21 | Ustedes aburrido |
| S-47 | HERMANO + DIFERENTE | El hermano es diferente. | reglas | R-15, R-18, R-20, R-21 | El hermano diferente |
| S-48 | YO + BIEN | Yo estoy bien. | reglas | R-16, R-18, R-22 | Yo bien |
| S-49 | ABUELO + MAL | El abuelo está mal. | reglas | R-15, R-18, R-22 | El abuelo mal |
| S-50 | YO + MUCHO + FELIZ | Yo estoy muy feliz. | reglas | R-18, R-20, R-21, R-25 | Yo feliz mucho |
| S-51 | YO + SENTIR + TRISTE | Yo me siento triste. | reglas | R-18, R-21, R-23 | Yo siento triste |
| S-52 | TÚ + SENTIR + BIEN | Tú te sientes bien. | reglas | R-18, R-23 | Tú sientes bien |
| S-53 | YO + HAMBRE | Yo tengo hambre. | reglas | R-18, R-24 | Yo la hambre |
| S-54 | NIÑO + HAMBRE + MUCHO | El niño tiene mucha hambre. | reglas | R-15, R-18, R-24, R-25 | El niño la hambre mucho |
| S-55 | TODOS + BIENVENIDO | Todos son bienvenidos. | reglas | R-17, R-18, R-20, R-21 | Todos bienvenido |
| S-56 | TÚ + BIENVENIDO | Tú eres bienvenido. | reglas | R-17, R-18, R-20, R-21 | Tú bienvenido |
| S-57 | ¿DÓNDE? + TÚ + VIVIR | ¿Dónde vives tú? | reglas | R-16, R-18, R-30 | ¿Dónde vives? |
| S-58 | ¿CUÁNDO? + HERMANO + TRABAJAR | ¿Cuándo trabaja el hermano? | reglas | R-15, R-18, R-30 | ¿Cuándo trabaja el hermano? |
| S-59 | ¿QUÉ? + NIÑO + COMER | ¿Qué come el niño? | reglas | R-15, R-18, R-30 | ¿Qué come el niño? |
| S-60 | ¿QUIÉN? + COMER | ¿Quién come? | reglas | R-18, R-31 | ¿Quién come? |
| S-61 | ¿QUIÉN? + HAMBRE | ¿Quién tiene hambre? | reglas | R-24, R-31 | ¿Quién la hambre? |
| S-62 | ¿QUIÉN? + TÚ | ¿Quién eres tú? | reglas | R-18, R-31 | ¿Quién? |
| S-63 | ¿QUIÉN? + USTEDES | ¿Quiénes son ustedes? | reglas | R-18, R-31 | ¿Quién ustedes? |
| S-64 | ¿CÓMO? + ABUELO | ¿Cómo está el abuelo? | reglas | R-15, R-18, R-30 | ¿Cómo el abuelo? |
| S-65 | ¿CÓMO? + TÚ + SENTIR | ¿Cómo te sientes tú? | reglas | R-18, R-23, R-30 | ¿Cómo sientes? |
| S-66 | ¿POR QUÉ? + NIÑO + TRISTE | ¿Por qué está triste el niño? | reglas | R-15, R-18, R-20, R-21, R-30 | ¿Por qué el niño triste? |
| S-67 | ¿POR QUÉ? + TÚ + TRABAJAR + MUCHO | ¿Por qué trabajas mucho tú? | reglas | R-18, R-25, R-30 | ¿Por qué trabajas mucho? |
| S-68 | HOLA + ¿DÓNDE? + HERMANO + VIVIR | Hola, ¿dónde vive el hermano? | reglas | R-09, R-15, R-18, R-30 | Hola ¿dónde vive el hermano? |
| S-69 | BUENOS DÍAS + FAMILIA + FELIZ | Buenos días, la familia está feliz. | reglas | R-09, R-15, R-17, R-18, R-20, R-21 | Buenos días la familia feliz |
| S-70 | HOLA + TÚ + NOMBRE + ¿QUÉ? | Hola, ¿cuál es tu nombre? | plantilla | T-03 con saludo (R-07, R-09) | Hola ¿qué el nombre? |
| S-71 | HOLA + BUENOS DÍAS + ¿CÓMO? + USTEDES | Hola, buenos días, ¿cómo están ustedes? | reglas | R-09, R-18, R-30 | Hola, buenos días ¿cómo ustedes? |
| S-72 | HOLA + BUENOS DÍAS + ¿CÓMO ESTÁS? | Hola, buenos días, ¿cómo estás? | reglas | R-09, R-12 | Hola, buenos días, ¿cómo estás? |
| S-73 | TÍO + ABURRIDO + NUNCA | El tío nunca está aburrido. | reglas | R-15, R-18, R-20, R-21, R-26 | El tío nunca aburrido |
| S-74 | YO + NUNCA + HAMBRE | Yo nunca tengo hambre. | reglas | R-16, R-18, R-24, R-26 | Yo nunca la hambre |
| S-75 | YO + NUNCA + SENTIR + DIFERENTE | Yo nunca me siento diferente. | reglas | R-16, R-18, R-21, R-23, R-26 | Yo nunca siento diferente |
| S-76 | HERMANO + COMER + POCO | El hermano come poco. | reglas | R-15, R-18, R-25 | El hermano come poco |
| S-77 | YO + HAMBRE + POCO | Yo tengo poca hambre. | reglas | R-16, R-18, R-24, R-25 | Yo la hambre poco |
| S-78 | ¿QUIÉN? + FELIZ | ¿Quién está feliz? | reglas | R-18, R-20, R-31 | ¿Quién feliz? |
| S-79 | ¿QUIÉN? + MAL | ¿Quién está mal? | reglas | R-18, R-22, R-31 | ¿Quién mal? |
| S-80 | ¿POR QUÉ? + TÚ + SENTIR + TRISTE | ¿Por qué te sientes triste tú? | reglas | R-18, R-21, R-23, R-30 | ¿Por qué sientes triste? |
| S-81 | ¿POR QUÉ? + NIÑO + HAMBRE | ¿Por qué tiene hambre el niño? | reglas | R-15, R-18, R-24, R-30 | ¿Por qué el niño la hambre? |
| S-82 | ¿CÓMO ESTÁS? + TÚ | ¿Cómo estás tú? | reglas | R-18, R-32 | ¿Cómo estás? |
| S-83 | TÍO + TRABAJAR + BIEN + MUCHO | El tío trabaja muy bien. | reglas | R-15, R-18, R-22, R-25 | El tío trabaja bien mucho |
| S-84 | YO + SENTIR + BIENVENIDO | Yo me siento bienvenido. | reglas | R-16, R-18, R-21, R-23 | Yo siento bienvenido |
| S-85 | ADIÓS + ABUELO | Adiós, abuelo. | reglas | R-09, R-11 | Adiós el abuelo |
| S-86 | GRACIAS + BUENOS DÍAS | Gracias, buenos días. | plantilla | T-18 | Buenos días gracias |
| S-87 | GRACIAS + BUENAS TARDES | Gracias, buenas tardes. | plantilla | T-19 | Buenas tardes gracias |
| S-88 | GRACIAS + USTEDES | Gracias a ustedes. | plantilla | T-20 | Ustedes gracias |
| S-89 | GRACIAS + TÚ | Gracias a ti. | plantilla | T-21 | Tú gracias |
| S-90 | MAL + GRACIAS | Mal, gracias. | plantilla | T-22 | Mal gracias |
| S-91 | GRACIAS + ABUELO | Gracias, abuelo. | reglas | R-11, R-13 | El abuelo gracias |
| S-92 | FAMILIA + BIENVENIDO | Bienvenida, familia. | reglas | R-11, R-12, R-20, R-21 | La familia bienvenido |

Cuenta: 92 secuencias. Salen por plantilla 23 (S-05, S-21 a S-36, S-70 y S-86 a S-90) y por reglas
69. De las 92 salidas heredadas, 14 coinciden con el texto nuevo si no se cuenta el punto final:
S-07, S-08, S-09, S-13, S-14, S-37, S-38, S-42, S-43, S-58, S-59, S-60, S-72 y S-76. Las salidas de
S-69 a S-85 y su columna «heredado» se midieron con el borrador 2 y con `GrammarMapper`; las de S-86 a
S-92, con la implementación 3a-2 y con `GrammarMapper`, en el orden de glosas de la tabla. El orden de
la tabla es de presentación y no siempre sigue `ORDEN_PRESENTACION` (§9.2).

---

## 5. Comportamiento fuera de lo previsto

### 5.1 Garantías para la voz (TTS) y el registro

1. **Nunca lanza excepción** con un argumento de cualquier tipo ni con elementos de cualquier tipo
   (R-01). Hoy la llamada no está en `try/except` (contrato, Apéndice A, 5(d)).
2. **Nunca devuelve `""` si hubo al menos una glosa del vocabulario** en un argumento que se pudo
   recorrer entero (hasta el tope de R-01). `""` solo sale por R-05, o por un iterable que lanza al
   recorrerlo, que se trata como «no es una lista» (R-01) aunque antes del fallo hubiera dado glosas
   válidas. En el prototipo no ocurre: siempre pasa listas.
3. **Nunca presenta como frase algo que ninguna regla ni plantilla produjo:** eso va al respaldo
   (R-35), en minúscula, sin punto y con `respaldo = True`.
4. **Determinista y sin estado:** la misma entrada da siempre la misma salida, en cualquier orden de
   llamadas y en cualquier proceso (contrato, Apéndice A, 5(a)). Con un `set` o un `frozenset` el
   orden lo pone el id de cada glosa, no la iteración del conjunto (R-01).
5. **Rápido** (Apéndice A, 5(c)): son búsquedas en tablas. El borrador 2 tarda 1,89 µs por llamada
   de media sobre los 2.450 pares ordenados, y 28,7 µs el peor par, frente a 1,8 µs de media del
   heredado (medido con el borrador 1), todo en esta máquina. **Medido con la implementación 3a-2**
   (8-nov, noche; `translate([a, b])` sobre los 2.450 pares, el mínimo de 3 llamadas por par con
   `time.perf_counter`): 4,0 µs de media y 7,5 µs el peor par; el heredado, con el mismo método,
   1,45 µs y 1,7 µs. Requisito: menos de 1 ms por llamada en el peor de los 2.450 pares; se cumple
   con holgura y lo comprueba la prueba 8.

### 5.2 Motivos de respaldo

`MOTIVOS` contiene **exactamente** estos 35 códigos, en este orden, que es el de §3.1. No se admiten
nombres más finos ni variantes: los recuentos de §9.1 se comparan código por código. 34 son
alcanzables con `estricto=False` (32 en conjuntos de 2 a 4 glosas, más `fuera_de_vocabulario` y
`demasiadas_glosas`, medido con el borrador 2) y `fuera_de_inventario` solo con `estricto=True`. El
código defensivo `no_previsto` (§3.1) no está en `MOTIVOS`. La columna «B4» marca los respaldos que
se le muestran al profesor (§9.2).

| # | motivo | paso (§3.1) | regla | ejemplo de entrada | salida | B4 |
|---|---|---|---|---|---|---|
| 1 | `fuera_de_vocabulario` | A3 | R-02 | YO + «perro» + COMER | `yo, comer` | no |
| 2 | `demasiadas_glosas` | A8 | R-08 | HOLA + YO + TRABAJAR + MUCHO + BIEN | `hola, yo, trabajar, mucho, bien` | no |
| 3 | `saludos_no_combinables` | B1 | R-09 | HOLA + ADIÓS | `hola, adiós` | sí |
| 4 | `despedida_con_contenido` | B5 | R-09 | ADIÓS + BIENVENIDO | `adiós, bienvenido` | sí |
| 5 | `saludo_con_glosa_no_admitida` | B7 | R-12 | HOLA + TÚ | `hola, tú` | sí |
| 6 | `sustantivo_abstracto_fuera_de_plantilla` | C1 | R-28 | YO + NOMBRE | `yo, nombre` | sí |
| 7 | `cortesia_fuera_de_plantilla` | C2 | R-13 | GRACIAS + YO + COMER | `gracias, yo, comer` | sí |
| 8 | `dos_sujetos` | C3 | R-14 | HERMANO + TÍO + COMER | `hermano, tío, comer` | sí |
| 9 | `dos_interrogativos` | C4 | R-29 | ¿QUÉ? + ¿DÓNDE? + TÚ + VIVIR | `¿qué?, ¿dónde?, tú, vivir` | sí |
| 10 | `como_estas_no_admitido` | D1 | R-32 | ¿CÓMO ESTÁS? + YO | `¿cómo estás?, yo` | sí |
| 11 | `quien_con_sujeto_y_predicado` | D2 | R-31 | ¿QUIÉN? + HERMANO + COMER | `¿quién?, hermano, comer` | sí |
| 12 | `quien_sujeto_no_admitido` | D2 | R-31 | ¿QUIÉN? + FAMILIA | `¿quién?, familia` | sí |
| 13 | `pregunta_sin_sujeto` | D3 | R-30 | ¿DÓNDE? + COMER | `¿dónde?, comer` | sí |
| 14 | `pregunta_no_admitida` | D3 | R-30 | ¿CUÁNDO? + HERMANO + VIVIR | `¿cuándo?, hermano, vivir` | sí |
| 15 | `sin_sujeto` | C6 | R-14 | COMER + MUCHO | `comer, mucho` | sí |
| 16 | `dos_VERBO` | P1 | R-14 | YO + COMER + TRABAJAR | `yo, comer, trabajar` | sí |
| 17 | `dos_ADJ` | P1 | R-14 | YO + FELIZ + TRISTE | `yo, feliz, triste` | sí |
| 18 | `dos_ADV_EST` | P1 | R-14 | YO + BIEN + MAL | `yo, bien, mal` | sí |
| 19 | `dos_ADV_CANT` | P1 | R-14 | YO + MUCHO + POCO + COMER | `yo, mucho, poco, comer` | sí |
| 20 | `nunca_con_cantidad` | P2 | R-26 | YO + NUNCA + MUCHO + COMER | `yo, nunca, mucho, comer` | sí |
| 21 | `todos_con_nunca` | P3 | R-26 | TODOS + NUNCA + COMER | `todos, nunca, comer` | sí |
| 22 | `nunca_con_modo` | P4 | R-26 | YO + NUNCA + TRABAJAR + BIEN | `yo, nunca, trabajar, bien` | sí |
| 23 | `dos_predicados` | P5 | R-14 | YO + FELIZ + BIEN | `yo, feliz, bien` | sí |
| 24 | `dos_verbos` | P6 | R-14 | YO + SENTIR + TRABAJAR | `yo, sentir, trabajar` | sí |
| 25 | `verbo_con_predicado` | P7 | R-14 | HERMANO + TRABAJAR + FELIZ | `hermano, trabajar, feliz` | sí |
| 26 | `nunca_no_admitido` | P7, P8, P9 | R-26 | TÍO + NUNCA + JUICIOSO | `tío, nunca, juicioso` | sí |
| 27 | `cantidad_y_modo` | P7 | R-25 | TÍO + TRABAJAR + BIEN + POCO | `tío, trabajar, bien, poco` | sí |
| 28 | `cantidad_no_admitida` | P7, P8, P9 | R-25 | YO + POCO + FELIZ | `yo, poco, feliz` | sí |
| 29 | `vivir_sin_complemento` | P7 | R-27 | ¿POR QUÉ? + HERMANO + VIVIR | `¿por qué?, hermano, vivir` | sí |
| 30 | `sentir_sin_predicativo` | P8 | R-23 | YO + SENTIR | `yo, sentir` | sí |
| 31 | `sentir_hambre` | P8 | R-23 | YO + SENTIR + HAMBRE | `yo, sentir, hambre` | sí |
| 32 | `adjetivo_no_admite_sentir` | P8 | R-23 | YO + SENTIR + JUICIOSO | `yo, sentir, juicioso` | sí |
| 33 | `sin_predicado` | P9 | R-14 | YO + NUNCA | `yo, nunca` | sí |
| 34 | `yo_bienvenido` | P9 | R-20 | YO + BIENVENIDO | `yo, bienvenido` | sí |
| 35 | `fuera_de_inventario` | F | §7.1 | ¿QUIÉN? + BIENVENIDO (con `estricto=True`) | `¿quién?, bienvenido` | no |

La salida de respaldo de la tabla supone ese orden de entrada (R-35). El código del motivo es el de
la primera comprobación que falla, en el orden de §3.1. La implementación declara `MOTIVOS` con
estos 35 códigos y prueba que los 34 primeros se alcanzan con `estricto=False` y el 35 con
`estricto=True`.

### 5.3 Turnos muy largos

R-08 corta en 4 glosas distintas. Si el respaldo trae más de 8 glosas distintas, la frase es
exactamente `", ".join(glosas[:8])` y `omitidas == glosas[8:]` (R-35). Así el TTS no lee una lista
sin fin. Con 8 o menos, `omitidas == []`. De la entrada se leen a lo sumo 10.000 elementos (R-01); lo
que pase de ahí no se lee, no cuenta para ninguna regla y deja el aviso `entrada_truncada`. Es un
límite declarado: con más de 10.000 elementos, una glosa distinta que viniera después no se ve. En
el prototipo un turno no se acerca a esa cifra. Hasta el tope, 1.000 o 10.000 elementos repetidos se
colapsan con R-03.

---

## 6. Trazabilidad

`explicar(signs)` devuelve un `dict` con estas claves, siempre presentes y en este orden:

| clave | tipo | contenido |
|---|---|---|
| `frase` | str | lo mismo que `translate(signs)` |
| `via` | str | `"vacio"`, `"lexico"`, `"plantilla"`, `"reglas"` o `"respaldo"` |
| `plantilla` | str o None | id T-xx si se usó plantilla |
| `reglas` | list[str] | ids R-xx que intervinieron, en orden de aplicación (las pruebas comparan el conjunto) |
| `suplido` | list[str] | categorías suplidas (§3.2), sin repetir y en orden de aparición: `articulo` (R-15), `copula` (R-20, R-22), `persona_numero` (R-17, R-18), `concordancia_adjetivo` (R-21), `tiempo_verbal` (R-19), `verbo_soporte` (R-24), `pronombre_reflexivo` (R-23), `preposicion` (R-10), `intensificador` (R-25), `puntuacion` (R-09, R-30, R-34), `mayuscula` (R-33), `orden_fijo` (R-04, en toda salida por reglas de 2 o más glosas) |
| `supuestos` | list[str] | valores tomados sin que la glosa los dé (§3.2): `articulo_definido` (R-15), `tratamiento_tu` (R-16/R-17, siempre que aparece TÚ en las vías `reglas` y `plantilla`, también en T-03, T-04, sus variantes con saludo y T-21; no en la vía `lexico`, que no registra supuestos), `genero_por_defecto` (R-17, R-12), `tiempo_presente` (R-19), `lectura_tercera_persona` (R-32), `lectura_vocativa` (R-20) |
| `respaldo` | bool | True solo en la vía respaldo |
| `motivo` | str o None | código de §5.2 (o `no_previsto`, §3.1) |
| `glosas` | list[str] | glosas válidas distintas, en orden de primera aparición |
| `fuera_de_vocabulario` | list[str] | `repr` de los elementos rechazados: a lo sumo 10, truncados a 40 caracteres (R-01) |
| `n_fuera_de_vocabulario` | int | número total de elementos rechazados |
| `alias` | list[str] | entradas aceptadas por alias (R-01) |
| `repeticiones` | int | apariciones quitadas por R-03 |
| `omitidas` | list[str] | glosas no escritas en el respaldo: `glosas[8:]` (§5.3) |
| `avisos` | list[str] | avisos sobre la forma de la entrada: `entrada_str_suelta`, `entrada_no_lista`, `entrada_sin_orden` (set o frozenset), `entrada_truncada` (más de 10.000 elementos) (R-01) |
| `parametros` | dict | `{"genero_primera": …, "genero_segunda": …, "estricto": …}`, los valores efectivos con que se construyó el conversor (`"m"` tanto con el valor por defecto como con `"m"` explícito; solo `supuestos` los distingue) |
| `version` | str | `"3a-2"` (era `"3a-1"` antes de la segunda revisión del 8-nov). Cambia con cualquier cambio de tabla o regla |

Qué se registra en cada vía:

- `vacio` y `respaldo`: `suplido` y `supuestos` vacíos.
- `lexico`: `suplido` = [`puntuacion`, `mayuscula`] (punto o «?» y mayúscula de la forma de cita,
  §2.2); `supuestos` vacío. La forma de cita no conjuga ni pone artículo, así que no hay más.
- `plantilla`: la plantilla se valida como unidad (§4.2), así que `suplido` solo recoge lo que
  añade R-09 al anteponer un saludo (`puntuacion`). `supuestos` registra `tratamiento_tu` en T-03,
  T-04 y T-21, porque el enunciado fijo elige «tu» o «ti».
- `reglas`: lo que registra cada regla, según la columna de arriba.

**En el registro del prototipo** (solo con `--conversor reglas` o `inventario`): el registro del ensayo y el del
turno ganan una clave `"conversion"` con el `dict` de `explicar` sin `frase`, porque `enunciado` ya
existe. Con el conversor heredado no se añade nada. Así siguen valiendo las claves fijadas en
test_prototipo.py:1200-1203 y test_tts.py:1550-1559. Es un cambio aditivo que requiere pruebas
nuevas.

**En la tesis:** hay dos tablas en `docs/`, y conviene citar cada una por su nombre.

- `docs/CONVERSOR_3a_SALIDAS_NO_RESPALDO.csv` es **la tabla de esta sección**: todas las salidas no
  respaldo, **7.045 filas** en 3a-2 (las 50 solas de la vía `lexico` + los 6.995 conjuntos de 2 a 4
  glosas con frase, §9.1). La escribe `python -m src.conversor_glosas --exportar RUTA.csv`. Columnas:
  `glosas` (en orden de id; es la clave), `presentacion` (las mismas glosas en el orden fijo de las
  hojas, §9.2), `frase`, `via`, `plantilla`, `reglas`, `suplido`, `supuestos`, `bloque_actividad_B` y
  `estado_revision`. `bloque_actividad_B` dice en qué bloque se le **mostrará** el ítem al profesor:
  `B1` (las 50 solas), el id S-xx (inventario, que va en B2), `B3` (muestra de 30, semilla 2026,
  §9.2) o vacío si no se le muestra. `estado_revision` vale `pendiente` en todas las filas hasta la
  sesión de la Actividad B. (En 3a-1 había una sola columna, `revisado`, con esos mismos valores; se
  podía leer como si las salidas ya estuvieran revisadas.) Esta tabla es el procedimiento «con sus
  reglas explícitas y sus límites» en forma revisable, y sus dos últimas columnas dejan ver qué parte
  se va a verificar y que todavía no lo está (§1.4).
- `docs/CONVERSOR_3a_SALIDAS.csv` es otra cosa: la **tabla comparativa heredado → nuevo**, 3.522 filas
  en 3a-2 (50 solas + 22 plantillas + 2.450 pares ordenados + una muestra de 1.000 tríos ordenados con
  semilla 2026, `muestra_trios`), con las columnas `tipo`, `entrada`, `salida_heredada`,
  `salida_nueva`, `via`, `reglas`, `plantilla`, `respaldo`, `motivo`, `suplido` y `supuestos`. La
  escribe `scripts/conversor_3a_salidas.py`. Incluye los respaldos.

---

## 7. Interfaz

### 7.1 Módulo `src/conversor_glosas.py`

```python
from src.grammar_mapper import SIGN_VOCABULARY   # se importa, no se redefine

class ConversorGlosas:
    VERSION = "3a-2"

    def __init__(self, genero_primera: str | None = None, genero_segunda: str | None = None,
                 estricto: bool = False) -> None:
        """genero_*: None (equivale a 'm' y registra genero_por_defecto, R-17), 'm' o 'f' explícitos
        (no lo registran); estricto: bool. Cualquier otro valor -> ValueError (en el constructor,
        nunca en translate)."""

    def translate(self, signs) -> str:
        """Compatible con GrammarMapper.translate: misma firma, devuelve solo la frase."""
        return self.explicar(signs)["frase"]

    def translate_ids(self, sign_ids) -> str:
        """Como GrammarMapper.translate_ids; translate ya acepta ids, así que es lo mismo."""
        return self.translate(sign_ids)

    def explicar(self, signs) -> dict:
        """Frase más trazabilidad (§6). Acepta cualquier argumento sin lanzar (R-01)."""
```

- Tablas como constantes del módulo: `LEXICO` (glosa → categoría y rasgos), `FORMAS_VERBALES`,
  `FORMAS_ADJETIVO`, `ARTICULOS`, `PLANTILLAS`, `PREGUNTAS_ADMITIDAS`, `INVENTARIO` (las 92
  secuencias de §4.3 con su enunciado fijo; en las de plantilla, `reglas` lista el id T-xx y, en S-70,
  también R-07 y R-09), `MOTIVOS` (los 35 códigos de §5.2, en su orden), `ALIAS` (generada),
  `ORDEN_PRESENTACION` (§9.2) y `MAX_ELEMENTOS_ENTRADA` (R-01).
- **Modo estricto** (`estricto=True`): solo dan frase la vía `lexico` y los conjuntos que son una
  secuencia S-xx de §4.3. Cualquier otro conjunto que por reglas o plantilla daría frase sale como
  respaldo con `fuera_de_inventario` (paso F de §3.1); un conjunto que ya era respaldo conserva su
  motivo. Así, lo que el prototipo dice en voz alta en ese modo es exactamente lo que el profesor
  revisa en B1 y B2.
- Al importar se comprueba `set(LEXICO) == set(SIGN_VOCABULARY.values())` con un `if` explícito que
  lanza `RuntimeError`, no con `assert`, que desaparece con `python -O`. Si cambia el vocabulario,
  el módulo falla al cargar y no en mitad de una sesión. Con el mismo `if` se comprueba que cada
  clave de `INVENTARIO` da su enunciado fijo.
- No usa `GRAMMAR_CATEGORIES` ni `GrammarMapper`. Así `src/analisis_errores.py`, que agrupa por
  `GRAMMAR_CATEGORIES["cat"]`, no cambia.
- CLI: `python -m src.conversor_glosas` imprime las 92 secuencias previstas con su salida y cuatro
  ejemplos de respaldo, y `--exportar RUTA.csv` escribe la tabla de §6 (7.045 filas; en el repositorio,
  `docs/CONVERSOR_3a_SALIDAS_NO_RESPALDO.csv`). Funciones de módulo para las tablas: `conjuntos`,
  `muestra_trios`, `muestra_b3`, `orden_presentacion`, `tabla_salidas` y `exportar_tabla`.

### 7.2 Integración en el prototipo (cambios aditivos)

- `src/realtime.py`: nueva opción `--conversor {heredado,reglas,inventario}`, por defecto
  `heredado` (desde el 22-nov-2026, `inventario`: `CONVERSOR_POR_DEFECTO` en translator.py). Con `reglas` se construye `ConversorGlosas()`; con `inventario`,
  `ConversorGlosas(estricto=True)` (`construir_conversor` en translator.py). Se pasa como
  `grammar_mapper=` a `LSCTranslator`; la inyección ya existía (parámetro `grammar_mapper` de
  `LSCTranslator.__init__`). Para las sesiones del protocolo antes de la Actividad B, y para la
  tesis, se propone `inventario`: es la variante que se ajusta al método del anteproyecto (§1.4).
  `reglas` queda como extensión (decisión del autor pendiente de acordar con el director, que se
  anotaría en DECISIONES_TESIS.txt).
- `src/realtime.py`: nuevas opciones `--genero-primera {m,f}` y `--genero-segunda {m,f}`, por
  defecto `m`, que se pasan al constructor. Con `--conversor heredado` no tienen efecto, y si se dan
  explícitamente el programa termina con un error de argumentos, para que nadie crea que se
  aplicaron. Sus valores quedan en el registro dentro de `conversion.parametros` (§6). Sin estas
  opciones, toda sesión diría «Yo estoy contento» la diga quien la diga (L-08).
- `src/translator.py`: si el conversor inyectado es un `ConversorGlosas` (`isinstance`, en
  `_es_conversor_glosas`), `LSCTranslator._convertir` llama a `explicar` en vez de a `translate` y se
  guarda `"conversion"` en el registro (§6). Si no, el código sigue como hoy. No se detecta con
  `getattr(…, "explicar")`: un doble de prueba como `MagicMock` tiene cualquier atributo y se tomaría
  por el conversor nuevo. `conversion_ns` mide la llamada que se haga.
- **Import perezoso** (desde 3a-2): translator.py no importa `src.conversor_glosas` al cargar. Lo
  importa `construir_conversor` solo con `reglas` o `inventario`, y `_es_conversor_glosas` mira
  `sys.modules` sin forzar el import. Así las comprobaciones de §7.1 (`RuntimeError` si el léxico o
  el inventario no cuadran) solo afectan a quien elige el conversor nuevo: con `heredado`, el
  prototipo arranca aunque el módulo nuevo falle al cargar (lo comprueba una prueba).
- Lo que no cambia: el valor por defecto (decisión del autor para esta actividad; cambió después, el
  22-nov-2026, a `inventario` en `realtime.py`, mientras `LSCTranslator` sin conversor inyectado sigue con
  GrammarMapper);
  `src/grammar_mapper.py`; `tests/test_grammar_mapper.py`; las aserciones de test_tts.py:1567-1573
  (precalentado con el heredado) y de test_prototipo.py:847 y :927 (comparan con
  `GrammarMapper()`).
- Límite que ya existe y no se corrige aquí: la ventana de `realtime.py` dibuja con
  `FONT_HERSHEY_SIMPLEX` (realtime.py:133), que no dibuja tildes, ñ ni ¿. La voz no se ve afectada.

---

## 8. Casos no cubiertos y límites

| # | Límite | Qué hace el conversor | Consecuencia |
|---|---|---|---|
| L-01 | **Preguntas sí/no.** En otras lenguas de señas se marcan sobre todo con rasgos no manuales (Zeshan 2006, p. 40; muestra sin la LSC), y el prototipo no los capta | sale la afirmación | una pregunta se oye como afirmación |
| L-02 | **Tiempo verbal.** La LSC lo marca con complementos o lo deja al contexto (Naranjo Orozco y Dieck 2022, pp. 61 y 66). El léxico no tiene señas de tiempo | todo en presente (R-19) | un hecho pasado o futuro se dice en presente |
| L-03 | **Plural por repetición en el espacio** (NIÑO, entre otras; Oviedo, en INSOR 2006, p. xxxix) y **numerales incorporados** (p. xxxviii) | la repetición se colapsa (R-03) | «Niño.» por «niños» |
| L-04 | **Espacio, verbos con concordancia espacial, INDEX y clasificadores.** Los verbos de la LSC marcan persona (anteproyecto, 08_marco.tex, l. 147-148, citando a Oviedo, ref13, no verificado); cómo la marcan no llega al clasificador. INSOR (2006, Anexo 2, p. 567) da la convención de glosa para los pronombres y el señalamiento | no se ve | persona y referentes salen solo de las glosas de pronombre |
| L-05 | **Sujeto nulo**, recuperable del contexto o del espacio | respaldo (R-14) | COMER + MUCHO no forma frase |
| L-06 | **Coordinación, subordinación, dos verbos o dos sujetos** | respaldo (R-14) | «El hermano y el tío comen» no se produce |
| L-07 | **Posesivos** («mi tío») | artículo definido (R-15) | puede no ser lo que se quiso decir |
| L-08 | **Género de quien habla y de quien escucha** | masculino por defecto, configurable (R-17) | «Yo estoy contento» la diga quien la diga |
| L-09 | **Tratamiento** (*tú*, *usted*, *vos*) | *tú*. La etiqueta «tú» de LSC50 no decide el tratamiento (§1.2): se elige *tú* por decisión del autor, pendiente de validar (B5). Se registra `tratamiento_tu` | en Cali *vos* y *usted* también son frecuentes (juicio del autor, sin fuente consultada); puede no ser el registro esperado |
| L-10 | **Nombres propios y deletreo** | no están en el léxico | YO + NOMBRE va a respaldo |
| L-11 | **Diálogo entre turnos** (GRACIAS → CON GUSTO) | cada turno es independiente | — |
| L-12 | **Polisemia de la glosa** (BIEN adjetivo/adverbio; PERMISO fórmula/sustantivo; SEÑA «seña personal» o «una seña») | una sola lectura por glosa | puede elegirse la lectura que no era |
| L-13 | **Errores del reconocedor** | no los detecta | **una glosa equivocada puede dar una frase gramatical y falsa**, más creíble que un respaldo. Es el riesgo principal: la frase bien formada no prueba que el reconocimiento acertara |
| L-14 | **Negación con la cabeza o con el rostro.** En otras lenguas de señas va sobre todo con movimientos de cabeza (Zeshan 2006, p. 42); en la LSC hay negación incorporada, gestual y léxica (INSOR 2006, pp. 167, 348 y 452). El léxico no tiene NO | sale la afirmación | **invierte el sentido**: HERMANO TRABAJAR negado con la cabeza → «El hermano trabaja.» |
| L-15 | **Orden como información** (tópico-comentario, énfasis) | se ignora (R-04) | — |
| L-16 | **Exclamación e intensidad** | enunciado neutro (R-34) | — |
| L-17 | **Prosodia de la voz** con la nueva puntuación | no medida | queda para 3.b (pregunta abierta 11 de N-42) |
| L-18 | **Respaldo frecuente.** 1.880 de 2.450 pares ordenados (76,7 %) van a respaldo en la implementación 3a-2 (§9.1; 1.902 en el borrador 2). Desglose medido por motivo: cortesía fuera de plantilla 364, sin sujeto 306, saludo con glosa no admitida 282, pregunta sin sujeto 180, sustantivo abstracto 170, dos sujetos 156, sin predicado 108, pregunta no admitida 80, despedida con contenido 74, dos interrogativos 42, ¿cómo estás? no admitido 38, vivir sin complemento 28, sentir sin predicativo 28, saludos no combinables 14, ¿quién? con sujeto no admitido 6, yo + bienvenido 2, todos con nunca 2. No se midió cuántos son combinaciones sin enunciado posible (como GRACIAS + COMER) y cuántos son enunciados naturales que se dejaron fuera **por decisión de diseño**: sujeto nulo (COMER + MUCHO, «Come mucho», lo normal en español), saludo + pronombre (HOLA + TÚ, HOLA + USTEDES), saludo + cortesía (HOLA + GRACIAS), PERDÓN + vocativo (PERDÓN + ABUELO), YO + SENTIR («Lo siento»). GRACIAS + vocativo dejó de estar en esta lista en 3a-2 (R-11) | lista de glosas | en uso libre, muchos turnos se oirán como lista. Es el precio de no inventar. Los casos de diseño son los candidatos de B4 (§9.2), que es donde se mide si debería haber frase |
| L-19 | **Variación regional** de las señas: una misma idea puede tener señas distintas según la región o la comunidad. No se consultó fuente para la LSC | una glosa, una entrada; el clasificador solo conoce las variantes de LSC50 | una variante no grabada en LSC50 no llega como glosa, o llega como otra |
| L-20 | **Notación de glosas no estandarizada.** Para la LSE, los usuarios del sistema de San-Segundo et al. señalaron que la notación en glosas «is not yet standardized enough» («one sign can be represented by two glosses») (2010, p. 253) | las 50 etiquetas de LSC50 se toman tal cual (R-01 solo normaliza tildes, mayúsculas y espacios) | si otra fuente glosa la misma seña con otra etiqueta, no se reconoce como alias |
| L-21 | **Salidas por reglas sin revisión individual.** 6.965 conjuntos dan frase por la vía `reglas` (3a-2); el profesor revisa 69 de ellos en el inventario (B2) y una muestra de 30 conjuntos fuera del inventario (B3) | con `--conversor reglas` dice todas; con `--conversor inventario` solo las del inventario (§7) | una frase rara pero gramatical («¿Quién es bienvenido?») puede decirse en voz alta sin que nadie la haya revisado. En la tesis se informa la cobertura de la verificación junto a los juicios (§9.2) |

---

## 9. Criterio de verificación

### 9.1 Verificación automática (pruebas nuevas: `tests/test_conversor_glosas.py`)

Las cifras se midieron primero con el borrador 2 (script de medida fuera del repositorio) y la
implementación 3a-1 las reprodujo todas. Desde la segunda revisión, las de la columna «3a-2» son las
que fijan las pruebas; se midieron con la implementación el 8-nov (noche), con
`python`. Donde las dos columnas difieren, la
diferencia la explican las 5 plantillas y las 2 ramas nuevas (párrafo de cambios, más abajo).

| Propiedad | Borrador 2 (= 3a-1) | 3a-2 (vigente) |
|---|---|---|
| Glosas solas con salida no vacía / distintas entre sí | 50 / 50 | 50 / 50 |
| Glosas solas que terminan en «.» / en «?» | 43 / 7 | 43 / 7 |
| Pares ordenados de glosas distintas | 2.450 | 2.450 |
| — por plantilla / por reglas / respaldo | 26 / 522 / 1.902 | 36 / 534 / 1.880 |
| — pares con frase cuya salida cambia al invertir el orden | 0 | 0 |
| — pares cuyo estado de respaldo cambia al invertir el orden | 0 | 0 |
| Conjuntos de 2 a 4 glosas con frase (no respaldo) | 6.984 de 251.125 (plantilla 25 = 13 + 4 + 8; reglas 6.959 = 261 + 1.753 + 4.945) | 6.995 de 251.125 (plantilla 30 = 18 + 4 + 8; reglas 6.965 = 267 + 1.753 + 4.945) |
| Salidas no respaldo en total (tabla de §6) | 7.034 = 50 + 6.984 | 7.045 = 50 + 6.995 |
| Frases distintas entre los conjuntos con frase | 6.888 | 6.899. En los dos casos, las 96 coincidencias son todas parejas ¿CÓMO? + X frente a ¿CÓMO ESTÁS? + X (R-32) |
| Permutaciones de los conjuntos de 3 y 4 glosas con frase / con frase distinta de la del conjunto | 129.414 / 0 | 129.414 / 0 |
| Motivos distintos en los conjuntos de 2 a 4 glosas / veces que aparece `no_previsto` | 32 / 0 | 32 / 0 |
| Conjuntos de respaldo cuyo motivo cambia entre `PYTHONHASHSEED` 1 y 7 | 0 de 244.141 | 0 de 244.130 |
| Modo estricto: conjuntos con frase / con `fuera_de_inventario` | 85 / 6.899 | 92 / 6.903 |

Motivos de respaldo en los 2.450 pares ordenados (3a-2), con los códigos exactos de §5.2; entre
paréntesis, el borrador 2 donde difiere:

| motivo | pares | | motivo | pares |
|---|---|---|---|---|
| `cortesia_fuera_de_plantilla` | 364 (382) | | `despedida_con_contenido` | 74 |
| `sin_sujeto` | 306 | | `dos_interrogativos` | 42 |
| `saludo_con_glosa_no_admitida` | 282 (286) | | `como_estas_no_admitido` | 38 |
| `pregunta_sin_sujeto` | 180 | | `vivir_sin_complemento` | 28 |
| `sustantivo_abstracto_fuera_de_plantilla` | 170 | | `sentir_sin_predicativo` | 28 |
| `dos_sujetos` | 156 | | `saludos_no_combinables` | 14 |
| `sin_predicado` | 108 | | `quien_sujeto_no_admitido` | 6 |
| `pregunta_no_admitida` | 80 | | `yo_bienvenido` | 2 |
| | | | `todos_con_nunca` | 2 |

Son 17 motivos y suman 1.880 (1.902 en el borrador 2). Cómo salen algunos (comprobado contra el código del borrador): 306 =
18 × 17 pares ordenados entre las 18 glosas de predicado (sin sujeto, C6 va antes que P1); 170 = 85
conjuntos con SEÑA o NOMBRE fuera de plantilla y de saludo × 2 (C1 va antes que C2); 42 = 7 × 6 entre
las 7 glosas interrogativas (C4); 108 = 13 SUJ × {NUNCA, MUCHO, POCO} × 2, menos TODOS + NUNCA (que
es `todos_con_nunca`), más ¿QUIÉN? + {NUNCA, MUCHO, POCO} × 2, más ¿POR QUÉ? + SUJ × 2; 28 de vivir =
13 SUJ × 2 + ¿QUIÉN? + VIVIR × 2.

Cambios de 3a-2 frente al borrador 2, todos explicados por los cambios de regla de la segunda
revisión: las 5 plantillas nuevas (T-18 a T-22) pasan 10 pares ordenados a la vía plantilla (4 de
`saludo_con_glosa_no_admitida` y 6 de `cortesia_fuera_de_plantilla`); GRACIAS + vocativo (R-11) pasa
12 pares de `cortesia_fuera_de_plantilla` a reglas y suma 6 conjuntos de 2 glosas. Total: 6.984 →
6.995. BIENVENIDO con vocativo (R-20) y BIENVENIDO tras saludo (R-12) cambian el texto o la traza de
salidas que ya tenían frase, no los recuentos. **Dato de regresión de 3a-2** (medido con la
implementación, no con el borrador): en los 251.125 conjuntos, cuántas veces sale cada uno de los 32
motivos; está en `MOTIVOS_CONJUNTOS` de las pruebas y fija el orden de §3.1 en las entradas de 3 y 4
glosas, donde dos comprobaciones pueden chocar y los pares no lo ven.

Cambios frente al borrador 1 (8-nov, mañana), todos explicados por las dos reglas que cambiaron:
ADIÓS + vocativo (R-11) pasa 12 pares ordenados de `despedida_con_contenido` (86 → 74) a reglas
(510 → 522) y suma 6 conjuntos de 2 glosas; YO + SENTIR + BIENVENIDO (R-20) suma 1 conjunto de 3 y 7
de 4. Total: 6.970 → 6.984. Ninguna salida del borrador 1 cambió de texto (medido: 0 de 6.970).

Pruebas que debe tener el módulo:

1. **Las 50 solas:** coinciden exactamente con la columna «sola» de §2.3. Además
   `translate([g]) == translate([id])` para todo id, y `translate([g, g, g]) == translate([g])`.
2. **Las 92 secuencias previstas** de §4.3 dan el enunciado fijo exacto, en el orden de la tabla y
   en su orden inverso. Se comprueban también la vía y la plantilla, y que el conjunto de reglas
   contenga las listadas (esta última parte no está medida: el borrador no calcula `reglas`).
3. **Los 2.450 pares ordenados**, de forma exhaustiva:
   - ninguna excepción y ninguna salida vacía;
   - invariancia al orden (tabla de arriba);
   - las salidas no respaldo empiezan por mayúscula, o por «¿» + mayúscula;
   - terminan en «.» o en «?», tienen el mismo número de «¿» que de «?», y no tienen dobles
     espacios, « ,», «..» ni «.?»;
   - el respaldo es exactamente `", ".join(glosas[:8])`, sin punto final, y `omitidas ==
     glosas[8:]` (en los pares, siempre `[]`);
   - los recuentos de vía coinciden con la tabla;
   - **los recuentos por motivo coinciden, código por código, con la tabla de 17 motivos de
     arriba** (y ningún otro motivo aparece en los pares).
4. **Conjuntos de 3 y 4 glosas:** recorrido exhaustivo de los 19.600 conjuntos de 3 y los 230.300
   de 4. Se comprueban las mismas propiedades de forma, el recuento de 6.995 (3a-2), que aparecen
   exactamente los 32 motivos medidos y nunca `no_previsto`, y cuántas veces sale cada motivo
   (`MOTIVOS_CONJUNTOS`). Las permutaciones se prueban solo sobre
   los conjuntos con frase.
5. **Concordancia, por barrido:**
   - para los 13 sujetos × 3 verbos con complemento admitido, la forma verbal es la de §2.4;
   - para los 13 sujetos × 7 adjetivos, la cópula es la de R-20 y el adjetivo concuerda con el
     sujeto (salvo YO + BIENVENIDO, que es `yo_bienvenido`);
   - para los 13 sujetos × los 6 adjetivos con `con_sentir` sí, SUJ + SENTIR + ADJ da «sentirse» +
     adjetivo concordado, también YO + SENTIR + BIENVENIDO;
   - ninguna salida por reglas contiene «la hambre», «Personas come» ni un sujeto plural con verbo
     en singular.
6. **Bordes:**
   - `[]`, `["xyz"]`, `[99]`, `[-1]`, `[None]`, `[True]`, `[1.5]`, `[numpy.bool_(True)]` → `""` sin
     excepción;
   - `[numpy.int64(7)]` → «Comer.» (igual que `[7]`);
   - `"hola"` (str suelto) → «Hola.» con aviso `entrada_str_suelta`; `None`, `b"hola"`, `{}` y `5`
     como argumento → `""` con aviso `entrada_no_lista`;
   - un elemento cuyo `__repr__` lanza excepción → sin excepción, con `"<repr falló: …>"` en
     `fuera_de_vocabulario`; 1.000 elementos rechazados → a lo sumo 10 entradas de 40 caracteres
     como máximo, y `n_fuera_de_vocabulario == 1000`;
   - `["xyz","comer"]` → respaldo `comer`, con `fuera_de_vocabulario`;
   - los 13 alias de R-01;
   - 5 glosas distintas → `demasiadas_glosas`;
   - 1.000 elementos válidos repetidos → sin excepción;
   - con `estricto=True`: las 50 solas y las 92 secuencias dan lo mismo que sin él; ¿QUIÉN? +
     BIENVENIDO → `fuera_de_inventario`; un respaldo conserva su motivo; ninguna otra frase sale en
     los 251.125 conjuntos de 2 a 4;
   - `ConversorGlosas(genero_primera="x")` y `estricto="sí"` → `ValueError`; con
     `genero_primera="f"`, YO + CONTENTO → «Yo estoy contenta.» y no registra `genero_por_defecto`;
     TÚ + CONTENTO sí lo registra con el valor por defecto y no con `genero_segunda="m"`; ¿QUIÉN? +
     CONTENTO → «¿Quién está contento?» sin registrarlo; ¿QUIÉN? + TÚ lleva `copula` en `suplido`;
   - (3a-2) un `set` o `frozenset` → orden de id y aviso `entrada_sin_orden`; `itertools.count()` →
     vuelve, con aviso `entrada_truncada`;
   - `explicar(x)["frase"] == translate(x)` para todo lo anterior;
   - todas las claves de §6 presentes y de su tipo.
7. **Determinismo y ausencia de estado:** dos pasadas de los 2.450 pares, la segunda en orden
   inverso, dan resultados idénticos (frase y motivo). El barrido de la prueba 4 se repite en un
   subproceso con otro `PYTHONHASHSEED` y da los mismos motivos.
8. **Rendimiento:** el peor de los 2.450 pares tarda menos de 1 ms.
9. **No regresión del punto de partida:**
   - md5 de `src/grammar_mapper.py` = `ce3ac6c8f7b841bbd02118eca0f2d320`;
   - `tests.test_grammar_mapper` sigue con 89 pruebas OK (comprobado hoy);
   - `LSCTranslator()` sin argumentos sigue usando `GrammarMapper`;
   - con el heredado, el registro no gana la clave `conversion`;
   - con un `MagicMock` como `grammar_mapper`, translator.py sigue llamando a `translate` (§7.2);
   - `python -O -c "import src.conversor_glosas"` sigue comprobando el léxico (§7.1).
10. **Red de seguridad:** como en N-42, se aplican mutaciones de una tabla a la vez (una cópula, un
    género, una forma verbal, una plantilla, el orden del sujeto en R-30) y se comprueba que alguna
    prueba falla en cada una.
11. **Precedencia (3a-2):** una entrada por cada par de comprobaciones que §3.1 ordena y que puede
    chocar con 3 o 4 glosas (C1 > C2, C2 > C3, C3 > C4, C4 > D, las dos de D2, D3 > P, P1 > P2,
    P2 > P3, P3 > P4, P4 > P7, P5 > P8, P6 > P7, P2 > P7, las dos de P9, B1 > B5 y B5 > B6), en todas
    sus permutaciones. Las tres mutaciones que la verificación de 3a-1 dejó vivas (P2/P3
    intercambiados, las dos comprobaciones de D2 intercambiadas, C4 antes que C3) y las tres de la
    traza (TÚ sin `genero_por_defecto`, «¿Quién eres tú?» sin `copula`, femenino tras ¿QUIÉN?) se
    aplicaron a una copia del módulo el 8-nov (noche): las seis hacen fallar alguna prueba. Dentro de
    P1 (VERBO > ADJ > ADV_EST > ADV_CANT) y entre P5 y P6 no hay choque posible con 4 glosas como
    máximo (§3.1).

### 9.2 Verificación humana: Actividad B con el profesor de LSC vinculado

**Estado: no realizada. Ningún enunciado ni regla de este documento está validado por él.**

**Qué se le muestra.** Una hoja por ítem, en papel o en pantalla, sin cámara y sin grabar
(07_etica.tex). Cada ítem lleva la secuencia de glosas en MAYÚSCULA y, si él lo pide, cada seña con
`scripts/mostrar_senas.py`, como el 28-sep (N-54), que solo lee los videos del corpus. Debajo va la
frase que produce el conversor. No se le muestra la salida heredada, para no sesgar su juicio.

**Orden en que se escribe la secuencia** (desde 3a-2). El conversor ignora el orden (R-04), pero la
pregunta 1 juzga la secuencia tal como se escribe. En 3a-1 la tabla de §6 daba las glosas en orden de
id, y en B3 salían secuencias como «BUENAS NOCHES + NIÑA + JUICIOSO + HOLA», con el saludo al final.
Criterio fijo: en B2 se usa el orden de la tabla de §4.3; en B1 y B3, el orden de
`ORDEN_PRESENTACION` (columna `presentacion` de la tabla de §6): saludo (HOLA primero), cortesía,
interrogativo, sujeto, NUNCA, verbo, predicado, cuantificador. Ese orden **no** es una hipótesis
sobre el orden de la LSC (§1.2): solo hace que todas las hojas sigan el mismo criterio. Al profesor
se le dice que el orden escrito no es significativo, y la pregunta 1 le pide que diga en qué orden
la señaría.
Bloques:

| bloque | ítems | contenido |
|---|---|---|
| B1 | 50 | las glosas solas (§2.3): ¿así se dice la seña sola en español? (D-L1) |
| B2 | 92 | las secuencias previstas S-01 a S-92 |
| B3 | 30 | conjuntos con frase elegidos al azar entre los 6.903 que no están en el inventario (6.995 − 92), para estimar cómo se comportan las reglas fuera del inventario. Desde 3a-2 la muestra es **estratificada**: 6 estratos (2, 3 o 4 glosas × con o sin saludo), 5 conjuntos por estrato, cada estrato con `random.Random(2026)` (`muestra_b3`). En 3a-1 era una muestra simple con semilla 2026, y salieron 23 de 30 ítems de 4 glosas y 24 con saludo, porque los conjuntos de 4 glosas con saludo son la mayoría de los candidatos. La estratificación no hace la muestra representativa de la población de 6.903: da 10 ítems por tamaño y 15 con saludo, para que cada tipo de secuencia tenga algún ítem. Lo que se estime con B3 debe decirse por estrato |
| B4 | 34 | casos de respaldo: uno por cada motivo de §5.2 marcado «B4: sí» (32), con el ejemplo de esa tabla, más 2 combinaciones de cortesía y saludo que probablemente son enunciados naturales (HOLA + GRACIAS, PERDÓN + ABUELO; GRACIAS + ABUELO salió de B4 en 3a-2: ahora da frase y es S-91). Pregunta: ¿debería existir una frase para esta secuencia? No se incluyen `fuera_de_vocabulario` (no se da en el prototipo, R-02), `demasiadas_glosas` (límite de diseño, R-08) ni `fuera_de_inventario` (solo modo estricto) |
| B5 | 15 | decisiones puntuales en forma de opciones: (1-4) ser/estar con JUICIOSO, ABURRIDO, DIFERENTE y FELIZ; (5) «a todos» frente a «, todos»; (6) «Comer.» frente a «Come.»; (7) «el tío» frente a «mi tío»; (8) «¿Dónde vives tú?» frente a «¿Dónde vives?»; (9) «Tú te sientes bien.» frente a «Te sientes bien.»; (10) S-67: «¿Por qué trabajas mucho tú?», «¿Por qué trabajas tú mucho?» o «¿Por qué trabajas tanto?»; (11) tratamiento: «¿Dónde vives tú?», «¿Dónde vive usted?» o «¿Dónde vivís vos?»; (12) «Buenos días, la familia está feliz.» frente a «Buenos días. La familia está feliz.»; (13) T-02: «El nombre, por favor.», «Tu nombre, por favor.» o «Su nombre, por favor.»; (14) R-32, ¿CÓMO ESTÁS? + HERMANO: «¿Cómo está el hermano?» (pregunta por un tercero) frente a «Hermano, ¿cómo estás?» (le pregunta al hermano), y lo mismo con FAMILIA («¿Cómo está la familia?» frente a «Familia, ¿cómo están?»); (15) R-20, BIENVENIDO: «¿dar la bienvenida a la familia, o afirmar que la familia es bienvenida?»: «Bienvenida, familia.» frente a «La familia es bienvenida.», y con TÚ y TODOS: «Tú eres bienvenido.» frente a «Bienvenido.», «Todos son bienvenidos.» frente a «Bienvenidos todos.» o «Hola a todos, bienvenidos.» |

Si una sesión no alcanza, el orden de prioridad es B2 (S-01 a S-36 primero), B5, B1, B4 y B3.

**Preguntas por ítem:**

1. «¿Esta secuencia de señas es algo que una persona diría en LSC?»: **natural / rara / no se
   dice**, y «¿en qué orden la señaría?» (el orden escrito no es significativo; ver arriba). Detecta
   secuencias del inventario que no existen en la lengua.
2. «¿La frase en español dice lo mismo que la secuencia?»: **sí / en parte / no**. Es un juicio de
   adecuación en 3 niveles, una versión reducida de la escala de 5 de Koehn y Monz (2006, p. 107).
3. «¿Cómo la diría usted en español?»: respuesta libre, anotada literal. Da la frase de referencia.
4. Solo en B5: «¿Cuál de las opciones?», más un comentario.

Se anota también su perfil declarado (persona sorda usuaria de la LSC o intérprete, 0.c). De él
depende cómo se lee la pregunta 3.

**Qué se hace con cada respuesta:**

| respuesta | acción |
|---|---|
| sí, y su versión coincide | el ítem queda «revisado por la persona señante (fecha), sin cambios» |
| sí, con otra versión | no se cambia nada solo por eso. Su versión se anota como alternativa aceptada |
| en parte | se corrige la regla o la plantilla con su versión, si el cambio es local, y se anota la decisión. Si no es local, la secuencia pasa a no cubierta |
| no | la plantilla se quita, o la regla se restringe para que esa entrada vaya a respaldo. Su versión, si la da, puede entrar como plantilla nueva atribuida a la sesión |
| pregunta 1 = no se dice | la secuencia sale del inventario aunque la frase sea buena |
| B4: «sí debería haber frase» | su frase entra como plantilla candidata, que se revisa en la sesión siguiente |
| B5 | gana su elección. Si duda, se mantiene la decisión del autor marcada como «sin acuerdo» |

Cada cambio sube `version` (3a-2, 3a-3…), actualiza las cifras de §9.1 y deja una entrada en
DECISIONES_TESIS.txt (propuesta por el estudiante).

**Qué se reporta en la tesis:** los juicios ítem por ítem y los recuentos de sí / en parte / no por
bloque, sin promedios con decimales. Se reporta también la diferencia literal entre cada frase y su
versión. **Junto a los juicios se informa la cobertura de la verificación:** de las 7.045 salidas
que el conversor puede dar con `reglas` (3a-2), se revisan 172 (50 solas + 92 del inventario + 30 de
muestra, 2,4 %); de los 6.995 conjuntos de 2 a 4 glosas, 122 (1,7 %). Con `inventario`, lo que el
prototipo puede decir son las 50 solas y las 92 secuencias, y B1 y B2 las revisan todas. Lo revisado no autoriza a decir que el resto está
validado: la muestra B3 solo da una estimación, con 30 ítems (5 por estrato) y un evaluador.

Con un solo evaluador y una sola referencia por ítem no se presenta BLEU como medida principal.
BLEU se define contra «a corpus of good quality human reference translations» (Papineni et al.
2002, p. 311), y aquí habrá, a lo sumo, una versión del profesor por ítem. Además, en juicios de adecuación es difícil
mantener un criterio constante sobre lo que significa cada nivel de la escala: «it is very difficult
to maintain consistent standards» (Koehn y Monz 2006, p. 107). Si se da alguna
métrica automática, va solo como descripción de la diferencia, calculada con SacreBLEU y con su
firma, como recomiendan Müller et al. (2023, pp. 685-686) para salidas en glosas; chrF es una
alternativa que ellos mismos mencionan (p. 685). Su advertencia de que BLEU «was never validated
empirically as an evaluation metric for gloss output» (p. 685) se refiere a salidas en glosas
(texto → glosa) y no se extiende aquí, donde la salida es español. El precedente de verificar glosa
y traducción con una persona de la comunidad es el de Naranjo Orozco y Dieck (2022, p. 60).
---

## 10. Comparación prevista heredado → nuevo (las 9 frases de ejemplo)

| # | glosas | heredado (medido) | nuevo (previsto) | qué cambia |
|---|---|---|---|---|
| 1 | BUENOS DÍAS + TODOS | Buenos días todos | Buenos días a todos. | «a» de destinatario y punto (R-10, R-34). Validar «a todos» |
| 2 | BUENAS TARDES + TODOS | Buenas tardes todos | Buenas tardes a todos. | ídem |
| 3 | TÍO + JUICIOSO | El tío juicioso | El tío es juicioso. | cópula *ser* (R-20). Validar *ser*/*estar* |
| 4 | FAMILIA + FELIZ | La familia feliz | La familia está feliz. | cópula *estar*, colectivo singular (R-17, R-20) |
| 5 | POR FAVOR + SEÑA | La seña por favor | La seña, por favor. | coma (T-01). Sentido por validar |
| 6 | HOLA + BIENVENIDO | Hola bienvenido | Hola, bienvenido. | coma tras el saludo (R-09, R-12) |
| 7 | HERMANO + TRABAJAR | El hermano trabaja | El hermano trabaja. | solo el punto |
| 8 | ¿DÓNDE? + HERMANO + VIVIR | ¿Dónde vive el hermano? | ¿Dónde vive el hermano? | igual. Ahora por regla declarada (R-30) y no por el orden fijo heredado |
| 9 | HOLA + ¿CÓMO ESTÁS? | Hola, ¿cómo estás? | Hola, ¿cómo estás? | igual. También en orden inverso, que en el heredado dependía de la rama `frase_completa` (H7b) |

Siete de las nueve cambian y dos quedan iguales. En las nueve, la salida nueva es la misma en
cualquier orden de entrada.

---

## Apéndice A. Contrato de interfaz con el prototipo

El texto cita este contrato por puntos (2, 3, 5(a) a 5(d)). El documento donde se redactó por
primera vez no está en el repositorio (no aparece con `grep`), así que aquí se **reconstruye a partir
del código**. Desde 3a-2 se cita por nombre de método y no por número de línea, porque la integración
de §7.2 movió las líneas (lectura del 8-nov, noche, sobre el árbol de trabajo). La numeración se conserva para que las referencias del
texto sigan valiendo. Lo que dice cada punto es comprobable en esas líneas; los motivos de 5(b) y
5(c) son del autor.

1. **Firma.** `translate(signs) -> str`, la misma de `GrammarMapper`. El prototipo la llama en tres
   sitios: modo aislado, con `[ens.clase]` (`LSCTranslator._inferir_y_emitir`, vía `_convertir`); modo oración, con las glosas del turno (`LSCTranslator.fin_de_turno`, vía
   `_convertir`); precalentado de la voz, `translate([glosa])` para las 50 glosas
   (`LSCTranslator._precalentar_voz`, llamado desde `__init__`).
2. **Entrada.** En el prototipo los elementos son glosas de `SIGN_VOCABULARY`: `ens.clase` sale de
   `SIGN_VOCABULARY.get(id_clase, f"[{id_clase}]")` (`_inferir_y_emitir`), y el modelo tiene 50 clases.
   La cadena `"[id]"` solo aparecería con un id que no está en el vocabulario y sería fuera de
   vocabulario (R-01, R-02).
3. **`""` significa «no hay nada que decir».** `LSCTranslator._emitir_audio` no emite audio con texto
   vacío, y `speak_async` y `prepare` vuelven sin hacer nada (tts.py:645-660, no releídas en 3a-2).
4. **Sin efectos externos.** No lee ni escribe archivos, no usa red y no guarda estado entre
   llamadas.
5. Requisitos:
   - (a) **Determinista y sin estado.** El texto que se precalienta (`_precalentar_voz`) tiene que ser
     el mismo que se pide después (modo aislado), o la cache de la voz no acierta.
   - (b) **Las 50 salidas solas son distintas entre sí,** para que cada seña se distinga por la voz.
   - (c) **Rápido.** La llamada del turno se cronometra en `conversion_ns` (`fin_de_turno`) y suma a
     la latencia del turno.
   - (d) **Nunca lanza excepción.** Las llamadas del modo aislado y del modo oración no están dentro
     de `try/except` (la del precalentado sí). Una excepción cortaría el ensayo o el turno.

---

## Referencias

Las claves y los estados vienen de la revisión de fuentes de esta actividad. «No verificada»
significa que la página no se reabrió.

- **INSOR 2006** — INSOR, Instituto Caro y Cuervo y MEN, *Diccionario básico de la Lengua de Señas
  Colombiana*, Bogotá, 2006 [ref14]. PDF leído en copia de web.archive.org (20240708010916),
  porque la URL de insor.gov.co da 404. Verificada.
- **Oviedo 2006** — A. Oviedo, «Notas para orientar la lectura del Diccionario Básico…», en INSOR
  2006, pp. xxv y ss. Verificada.
- **Oviedo 2001** — A. Oviedo, *Apuntes para una gramática de la LSC*, Univalle/INSOR, 2001 [ref13].
  **No verificada en esta actividad:** las páginas vienen del anteproyecto y de
  NOTA_ENTREGA_DIRECTOR.md.
- **Naranjo Orozco y Dieck 2022** — «La expresión de la categoría gramatical de tiempo en la Lengua
  de Señas Colombiana», *Signo y Seña* 41, pp. 48-78. doi:10.34096/sys.n41.10613. Verificada.
- **Johnston 2008** — «Corpus linguistics and signed languages: no lemmata, no corpus», LREC 2008
  Workshop, pp. 82-87. Verificada. Trata de Auslan.
- **Müller et al. 2023** — «Considerations for meaningful sign language machine translation based on
  glosses», ACL 2023 (Short), pp. 682-693. doi:10.18653/v1/2023.acl-short.60. Verificada.
- **Moryossef et al. 2021** — «Data augmentation for sign language gloss translation», AT4SSL 2021,
  pp. 1-11. Verificada. Trata de ASL y DGS.
- **San-Segundo et al. 2010** — «Advanced speech communication system for deaf people»,
  Interspeech 2010, pp. 250-253. doi:10.21437/Interspeech.2010-103. Verificada. Trata de la LSE.
  Vía por ejemplos (distancia y umbral): p. 251, en la sección «4. Spanish into LSE translation», es
  decir, en la dirección español → LSE. Comentario de los usuarios sobre la notación en
  glosas: p. 253 (comprobado hoy en el PDF; la revisión lo situaba en la p. 252). La Tabla 5 da
  2,56 % y el texto da 2,53 %. El cuarto autor figura como «Syaheerah Lufti» en el PDF:
  corregir la grafía antes de citar.
- **Reiter 1995** — «NLG vs. Templates», arXiv:cmp-lg/9504013. Verificada. Pocas plantillas y mantenimiento: p. 4, nota 1; concordancias sencillas con plantillas: p. 6 (páginas del PDF de arXiv, comprobadas hoy).
- **Koehn y Monz 2006** — «Manual and automatic evaluation of machine translation between European
  languages», WMT 2006, pp. 102-121. Verificada.
- **Papineni et al. 2002** — «BLEU…», ACL 2002, pp. 311-318. Verificada.
- **Zeshan 2006** — «Negative and interrogative constructions in sign languages», Ishara Press, cap. 2.
  doi:10.26530/OAPEN_453832. Verificada. La muestra no incluye la LSC.
- **DPD** — RAE y ASALE, *Diccionario panhispánico de dudas*, 2.ª ed., entradas «coma»,
  «concordancia», «punto», «mayúsculas» y «signos de interrogación y exclamación». La entrada
  «estar» es versión provisional. Leídas en copias de web.archive.org, porque rae.es responde 403.
  Conviene repetir la consulta desde un navegador antes de citar.
- **DLE** — RAE y ASALE, entradas «hola», «día» (buenos días) y «favor» (por favor), en copias de
  web.archive.org.

**Fuentes que faltan:** una norma sobre la obligatoriedad del artículo ante un sujeto nominal
(R-15; Nueva gramática, no consultada); una fuente sobre el uso vocativo de «todos» (R-10); la
entrada «hambre» del DPD, para «mucha/poca hambre» (R-24); una fuente sobre «sentirse» con
predicativo (R-23); una fuente sobre la variación regional de la LSC (L-19); una sobre el uso de
*tú*, *usted* y *vos* en Cali (L-09); la Nueva gramática sobre el masculino genérico (R-17, ¿QUIÉN?,
USTEDES, TODOS); y una fuente dialectal sobre la concordancia de *ustedes*, que la versión 3a-1
generalizaba a «todas las variedades» (R-17). Mientras falten, esas decisiones se presentan como juicio del
autor.

---

## Respuesta a la revisión

Revisión adversarial del 8-nov, 24 puntos. Se atendieron todos. Abajo, lo que se hizo con cada uno y
las discrepancias. Las cifras de esta tabla son las de 3a-1 (borrador 2); las vigentes están en §9.1, y
lo que cambió en 3a-2 está en la tabla siguiente.

| # | punto (gravedad) | qué se hizo |
|---|---|---|
| 1 | Precedencia de motivos (alta) | §3.1 reescrito con el orden real del borrador, paso a paso, y declarado normativo; `MOTIVOS` fijado con 35 códigos exactos (§5.2), sin «nombres más finos»; prueba de los 17 recuentos por motivo en §9.1. **Discrepancia parcial:** el orden propuesto en la revisión no es el del borrador. En el borrador, `dos_sujetos` (C3) va antes de `dos_interrogativos` (C4) y de toda la rama interrogativa, y `sin_sujeto` (C6) va después de ella. Se escribió el orden del código, que es el que produce las cifras |
| 2 | VIVIR con ¿POR QUÉ? y ¿QUIÉN? (media) | R-27 dice ahora «sin ADV_EST y sin ¿DÓNDE? ni ¿CÓMO?». El borrador ya mandaba los dos casos a `vivir_sin_complemento`; se comprobó con él. Ejemplos en R-27, R-30, R-31 y §5.2 |
| 3 | MUCHO con verbo + ADV_EST (media) | Se eligió «verbo + muy + adverbio» con BIEN/MAL, que es lo que ya hacía el borrador; POCO, y MUCHO con MÁS O MENOS, van a `cantidad_y_modo`. Fila nueva en R-25, ejemplo S-83 |
| 4 | Condiciones de R-14, R-20, R-23, R-26 (media) | R-14 reescrita con cada código, su paso y su ejemplo, incluidos `verbo_con_predicado` y la diferencia `dos_VERBO`/`dos_verbos`. **Cambio de regla:** `yo_bienvenido` solo con cópula; YO + SENTIR + BIENVENIDO da «Yo me siento bienvenido.» (S-84). R-23 y R-26: la restricción de *ser* no vale con SENTIR (S-75) |
| 5 | Inventario incompleto (media) | 17 secuencias nuevas (S-69 a S-85), medidas con el borrador 2. Corregida la frase de §4.1, que ya no dice «como mínimo, de cada patrón» |
| 6 | Alcance de la vía reglas (media) | Declarado en §1.4 y en L-21; cobertura (165 de 7.034, 115 de 6.984) en §1.4 y en §9.2; modo estricto `--conversor inventario` (§7) |
| 7 | L-18 sin medir (media) | Sustituida por el desglose medido por motivo y por la lista de enunciados naturales excluidos por diseño, sin estimar su proporción |
| 8 | Cita de Müller et al. (media) | El argumento contra BLEU se apoya ahora en Papineni et al. (p. 311) y Koehn y Monz (p. 107), con las frases comprobadas en los PDF. Müller se cita solo por SacreBLEU con firma y chrF, y se aclara que su advertencia es para salidas en glosas |
| 9 | L-19 y San-Segundo (baja) | Dividida en L-19 (variación regional, sin fuente) y L-20 (notación no estandarizada). La cita está en la **p. 253** del PDF, no en la 252 que daba la revisión |
| 10 | R-07 como «vía por ejemplos» (baja) | Reescrita como analogía, con coincidencia exacta frente a distancia y umbral (p. 251) |
| 11 | Pronombre pospuesto (baja) | Añadidos a B5 el par con y sin pronombre, S-67 con sus variantes y «tanto». **No se cambió R-30:** medido de nuevo con la implementación 3a-2 y con un criterio escrito, 431 conjuntos dan una pregunta parcial de R-30 (interrogativo distinto de ¿QUIÉN?) que termina en YO, TÚ o USTEDES, y en 279 de ellos hay al menos una palabra entre la forma verbal (con su «me/te/se») y el pronombre. (La versión anterior de esta fila decía 447 y 285: 447 sumaba las 16 de ¿CÓMO ESTÁS? + SUJ, que son R-32 y no preguntas parciales, y 285 contaba las 6 de ¿POR QUÉ? + PRON + NUNCA + VERBO, donde «nunca» va antes del verbo y no entre el verbo y el pronombre); cambiar la regla mueve todas esas salidas, y la alternativa («¿Por qué trabajas tú mucho?») también es un orden marcado. Se decide con el juicio del profesor, no antes |
| 12 | *ustedes* y tratamiento (baja) | R-17 y L-09 corregidas; par tú/usted/vos en B5 |
| 13 | Atribución del espacio (baja) | §2.2 y L-04 citan solo lo que dice el anteproyecto (08_marco.tex, l. 147-148, comprobado) |
| 14 | Criterio de categorías (baja) | Reformulado como propone la revisión; se explica por qué hay categorías de una sola glosa |
| 15 | Contrato de interfaz (baja) | Apéndice A, reconstruido desde el código con sus líneas, porque el original no está en el repositorio |
| 16 | 7.020 salidas (baja) | Con el borrador 2 son 7.034 (50 + 6.984); el CSV exporta también las 50 solas |
| 17 | Robustez (baja) | R-01: `numbers.Integral` sin `bool` (numpy comprobado), `repr` protegido y truncado, `str` suelto tratado como `[str]` con aviso, `bytes` y no iterables a `""`; `RuntimeError` en lugar de `assert`; `isinstance` en translator.py |
| 18 | Truncado del respaldo (baja) | §5.3, R-35 y la prueba 3 dicen lo mismo: `", ".join(glosas[:8])` y `omitidas == glosas[8:]` |
| 19 | `suplido` y `supuestos` (baja) | Cada regla dice qué registra; `tiempo_verbal` en `suplido` y `tiempo_presente` en `supuestos`, sin repetir códigos; qué se registra en cada vía (§6). No está medido: el borrador no calcula esos campos |
| 20 | Género configurable (baja) | `--genero-primera` y `--genero-segunda` en §7.2, con su valor en `conversion.parametros` |
| 21 | Cortesía y despedida (baja) | «Las combinaciones elegidas por el autor». **Cambio de regla:** ADIÓS + vocativo se admite por R-11 (S-85). «Hola, gracias», «Gracias, abuelo» y «Perdón, abuelo» no se añaden como plantillas: pasan a B4 para que decida el profesor. Nota de §2.1 corregida |
| 22 | Reiter, Camgöz, R-23, R-24 (baja) | Reiter citado en §1.2 y R-07 (pp. 4 y 6, comprobadas); Camgöz quitado de las referencias; R-23 y R-24 marcadas como juicio del autor, con la entrada «hambre» del DPD como fuente pendiente (no se consultó) |
| 23 | B4 (baja) | Sin `fuera_de_vocabulario` ni `demasiadas_glosas`; un ítem por motivo con ejemplos que pueden ser enunciados naturales (COMER + MUCHO, YO + SENTIR, HOLA + TÚ) y 3 combinaciones de cortesía más: 35 ítems |
| 24 | T-02 y R-09 (baja) | Alternativas de T-02 en la nota y en B5; la coma de R-09 se justifica como analogía con la interjección y con INSOR 2006, p. 488; el punto como alternativa, en B5 |

Efecto en las cifras: los cambios de regla de los puntos 4 y 21 suben los conjuntos con frase de
6.970 a 6.984 y los pares por reglas de 510 a 522; ninguna salida del borrador 1 cambió de texto.
Todas las cifras de §9.1 se volvieron a medir con el borrador 2.

---

## Respuesta a la segunda revisión (8-nov, noche; versión 3a-2)

Verificación de la implementación 3a-1, 26 hallazgos con tres lentes (gramaticalidad, conformidad con
este documento, honestidad). Cuando un hallazgo pedía cambiar una regla, se cambió primero este
documento, después el código y las pruebas, y se regeneraron los dos CSV. Gravedad entre paréntesis.
**Nada de lo que se cambió está revisado por el profesor de LSC vinculado.**

| # | hallazgo | qué se hizo |
|---|---|---|
| 1 | ¿CÓMO ESTÁS? + vocativo pasa a 3.ª persona (media) | **Discrepancia parcial: no se cambió la frase.** Se mantiene «¿Cómo está el hermano?» y se registra el supuesto `lectura_tercera_persona` (48 conjuntos con frase) con un sustantivo que tiene vocativo sí. Razones (R-32): el Diccionario básico glosa CÓMO-ESTÁ(?) seguido de personas (INSOR 2006, p. 488), y con FAMILIA la lectura vocativa exige cambiar el número («Familia, ¿cómo están?»), así que tampoco conservaría la glosa. Las dos lecturas van a B5 (14). Es la segunda opción que proponía el propio hallazgo |
| 2 | BIENVENIDO copulativo frente a fórmula (media) | **Aplicado en parte.** Con un sustantivo con vocativo sí: «Bienvenida, familia.» (R-20, excepción «fórmula», paso C7a; S-92; supuesto `lectura_vocativa`), también tras saludo («Hola, bienvenida, familia.»). **No aplicado** con TÚ, USTEDES y TODOS: «Bienvenido.» haría desaparecer la seña TÚ (R-16), y «Todos son bienvenidos.» / «Tú eres bienvenido.» también funcionan como bienvenida en español (juicio del autor). La cópula sigue con ¿QUIÉN?, MUCHO y los sustantivos sin vocativo. La pregunta «¿dar la bienvenida o afirmar que es bienvenido?» va a B5 (15) |
| 3 | Respaldos que deberían tener plantilla por simetría (media) | **Aplicada la primera opción:** T-18 «Gracias, buenos días.», T-19 «Gracias, buenas tardes.», T-20 «Gracias a ustedes.», T-21 «Gracias a ti.», T-22 «Mal, gracias.» y GRACIAS + vocativo por R-11 («Gracias, abuelo.»). Son los 11 conjuntos del hallazgo. Todos «validar: sí». Inventario S-86 a S-91. Criterio de simetría escrito en R-13. GRACIAS + ABUELO sale de B4 |
| 4 | Supuesto de género no registrado (baja) | BIENVENIDO tras saludo concuerda ahora con `genero_segunda` y registra `genero_por_defecto` con el valor por defecto (R-12). La glosa sola (vía `lexico`) no lo registra: es la forma de cita (§2.2, punto 3; R-17). ¿QUIÉN?, USTEDES y TODOS: masculino genérico, documentado en R-17 sin citar la Nueva gramática, que no se consultó |
| 5 | Pronombre sujeto explícito siempre (baja) | No es un error y no se cambia la regla. Se anota en R-16 como decisión de fidelidad; los pares con y sin pronombre ya están en B5 (8) y (9) |
| 6 | Las pruebas no fijan la precedencia de §3.1 en 3 y 4 glosas (media) | Prueba 11 nueva (`TestPrecedencia`, 18 entradas, en todas sus permutaciones) y recuento por motivo de los 251.125 conjuntos como dato de regresión de 3a-2 (`MOTIVOS_CONJUNTOS`). Las mutaciones M03, M04 y M10 del hallazgo se aplicaron a una copia del módulo: las tres fallan ahora |
| 7 | `suplido` y `supuestos` sin pruebas (baja) | Pruebas nuevas: TÚ + CONTENTO registra `genero_por_defecto` y con `genero_segunda="m"` no; ¿QUIÉN? + CONTENTO → «¿Quién está contento?» sin registrarlo; ¿QUIÉN? + TÚ lleva `copula`; R-32 no registra `copula` (anotado en R-32) y R-30 sí. M11, M08 y M05 fallan ahora |
| 8 y 22 | translator.py importa el módulo nuevo sin condición (baja) | Import perezoso en `construir_conversor` y detección por `sys.modules` en `_es_conversor_glosas` (§7.2). Prueba: `import src.translator` no carga `src.conversor_glosas`, y con el módulo nuevo roto al importar el heredado sigue cargando |
| 9 | Iterable infinito; generador roto da `""` (baja) | Tope de 10.000 elementos con aviso `entrada_truncada` (R-01, §5.3). El generador roto sigue dando `""`; se precisó la garantía 2 de §5.1 |
| 10 | `set`/`frozenset`: orden según `PYTHONHASHSEED` (baja) | Orden por id con aviso `entrada_sin_orden` (R-01, §5.1 garantía 4) |
| 11 y 25 | Firma del constructor y `tratamiento_tu` en la vía `lexico` (baja) | §7.1 con `None` = «m» y supuesto registrado; §6 precisa que `tratamiento_tu` no aplica a la vía `lexico` y que `parametros` da el valor efectivo |
| 12 y 16 | Ruta y filas de la tabla de §6 (baja y media) | §6 y §7.1 nombran las dos tablas: `CONVERSOR_3a_SALIDAS_NO_RESPALDO.csv` (tabla de §6, 7.045 filas) y `CONVERSOR_3a_SALIDAS.csv` (comparativa, 3.522 filas, tríos n=1000 con semilla 2026), con sus columnas y semillas |
| 13 | Comportamiento no pedido (baja) | Declarados en §7.1 y R-01: `translate_ids`, cualquier espacio Unicode, ejemplos de respaldo en la CLI. INVENTARIO da para S-70 las reglas T-03, R-07 y R-09 |
| 14 | La suite completa falla por `pytest` (baja) | **No aplicado:** es anterior a este trabajo (tests/test_arquitecturas_2.py, sin cambios desde 64fc9dc) y la decisión es del autor (instalar pytest en el venv o pasar ese archivo a unittest) |
| 15 | La columna `revisado` parece decir que ya se revisó (media) | Renombrada a `bloque_actividad_B`, más una columna `estado_revision` = `pendiente` en todas las filas (§6) |
| 17 | La vía `reglas` no es «un enunciado fijo por secuencia prevista» (media) | Cabecera, §1.4 y §7.2 presentan `inventario` como la implementación que se ajusta al método del anteproyecto y `reglas` como extensión cerrada, enumerable y sin verificar. Texto propuesto para DECISIONES_TESIS.txt abajo; no se escribió allí |
| 18 | «*ustedes* siempre, en todas las variedades» (baja) | Acotado al español de Colombia (R-17) |
| 19 | Paráfrasis de Naranjo Orozco y Dieck (baja) | Cita literal de la nota 11 en R-30. La frase la leyó en el PDF la verificación; no se volvió a abrir el PDF en esta corrección |
| 20 | Cifras 447 y 285 sin criterio (baja) | Fila 11 de la primera respuesta reescrita con el criterio y las cifras medidas con 3a-2: 431 y 279 |
| 21 | Referencias a líneas de translator.py desfasadas (baja) | Sustituidas por nombres de método en §2.2, R-01, R-05, §7.2 y el Apéndice A, con la fecha de lectura |
| 23 | B3 sesgada y glosas en orden de id (baja) | Orden fijo de presentación (`ORDEN_PRESENTACION`, columna `presentacion`) y B3 estratificada por tamaño y saludo, 5 por estrato, semilla 2026 (§9.2). La pregunta 1 pide además el orden en que él la señaría |
| 24 | La velocidad de §5.1 era del borrador (baja) | Medida con 3a-2: 4,0 µs de media y 7,5 µs el peor par; heredado 1,45 µs y 1,7 µs (método en §5.1) |
| 26 | San-Segundo et al.: dirección de traducción (baja) | Añadida en R-07 y en Referencias (español → LSE) |

Efecto en las cifras (§9.1): pares 26 / 522 / 1.902 → 36 / 534 / 1.880; conjuntos con frase 6.984 →
6.995; salidas no respaldo 7.034 → 7.045; inventario 85 → 92; cobertura de la Actividad B 165 → 172 de
7.045 (2,4 %) y 115 → 122 de 6.995 (1,7 %). Ningún conjunto que tenía frase en 3a-1 pasó a respaldo:
6.995 = 6.984 + los 11 conjuntos de las plantillas y la regla nuevas. Cambiaron de texto los 48
conjuntos que llevan el supuesto `lectura_vocativa` (vocativo + BIENVENIDO: 6 de 2 glosas y 42 con
saludo delante); con los parámetros por defecto, ninguna otra frase cambió de texto. Cambió la traza
(no la frase) de los 48 conjuntos con `lectura_tercera_persona` y de los que llevan BIENVENIDO tras un
saludo (R-12: `R-21`, `concordancia_adjetivo` y `genero_por_defecto`).

**Texto propuesto para DECISIONES_TESIS.txt** (no se escribió allí; lo decide el autor con el
director):

```
2026-11-08 · N-68 · Conversor de glosas 3.a: segunda revisión (versión 3a-2) y vía que se presenta
========================================================================================
  docs/CONVERSOR_3a_ESPECIFICACION.md (+ src/conversor_glosas.py, tests/test_conversor_glosas.py,
  docs/CONVERSOR_3a_SALIDAS.csv y docs/CONVERSOR_3a_SALIDAS_NO_RESPALDO.csv). Una verificación de la
  implementación 3a-1 dio 26 hallazgos; se corrigieron todos salvo uno (pytest, ajeno al conversor),
  con dos discrepancias parciales razonadas (¿CÓMO ESTÁS? + vocativo sigue en 3.ª persona, con supuesto
  registrado; BIENVENIDO con TÚ/USTEDES/TODOS sigue con cópula). Cambios de regla: 5 plantillas nuevas
  (T-18 a T-22), GRACIAS + vocativo, BIENVENIDO con vocativo como fórmula. Inventario: 92 secuencias.
  PROPUESTA (pendiente de acordar con el director): en la tesis se presenta como procedimiento de 3.a
  el modo estricto («--conversor inventario»: 50 glosas solas + 92 secuencias, cada una con su enunciado
  fijo), que es el método del anteproyecto (09_metodologia.tex: «a cada glosa o a cada secuencia corta
  prevista se le asociará […] un enunciado fijo»). La vía «reglas» (7.045 salidas enumeradas, 6.965 por
  reglas compositivas) se presenta como extensión cerrada y auditable, sin verificar. Las sesiones del
  protocolo antes de la Actividad B usarían «inventario». El valor por defecto del prototipo siguió siendo
  el heredado hasta el 22-nov-2026; desde entonces es «inventario».
  ADVERTENCIA: ninguna regla ni enunciado está revisado por el profesor de LSC vinculado (Actividad B pendiente).
```
