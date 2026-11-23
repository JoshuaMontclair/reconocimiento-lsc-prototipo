# Actividad B: revisión de las frases del conversor de glosas (actividad 3.a)

**Fecha de preparación:** 8 de noviembre de 2026
**Estado:** sesión **no realizada**. Ninguna frase ni regla está revisada todavía por la persona
señante vinculada (profesor de LSC, actividad 0.c, N-54).
**Versión del conversor que se muestra:** `3a-2` (`src/conversor_glosas.py`).
**Hoja de registro:** `docs/ACTIVIDAD_B_hoja_registro.csv` (178 filas, una por ítem).
**Procedimiento que se verifica:** `docs/CONVERSOR_3a.md` (resumen) y
`docs/CONVERSOR_3a_ESPECIFICACION.md` (§9.2, plan completo de la verificación humana).

Las frases de la hoja no se copiaron a mano: se generaron con el conversor 3a-2 el 8-nov. Cada
frase del inventario se comprobó contra la tabla `INVENTARIO` del módulo.

---

## 1. Para qué es esta sesión

Es la verificación humana que pide la actividad 3.a del anteproyecto: «verificación de los
enunciados producidos con la persona señante o intérprete vinculada». Se le muestra al profesor una
secuencia de señas (escrita como glosas en MAYÚSCULA) y la frase en español que produce el
conversor. Él dice si la frase dice lo mismo que las señas y cómo la diría él.

Lo que él responda decide qué reglas y plantillas se mantienen, cuáles se corrigen y cuáles se
quitan (§5). Hasta esta sesión, todas son decisiones del autor.

---

## 2. Instrucciones para el estudiante

**Antes de la sesión**

1. Imprimir la hoja o tenerla abierta en pantalla. No hace falta cámara. **No se graba ni se
   fotografía nada** (07_etica.tex).
2. Tener a mano `python scripts/mostrar_senas.py --solo <ids>` por si él pide ver alguna seña del
   corpus. Ese script solo reproduce los videos de LSC50, no abre la cámara.
3. Anotar en la cabecera de la hoja la fecha, la versión del conversor (`3a-2`) y el perfil que él
   declare: «persona sorda usuaria de la LSC» o «intérprete». **No se anotan nombre, edad ni otros
   datos personales.** La hoja no tiene columnas para eso.
4. **No mostrarle la salida del conversor heredado** (`GrammarMapper`), para no sesgar su juicio.

**Al empezar, decirle tres cosas**

- Las frases las produce un programa con reglas escritas por el estudiante. Todavía nadie las ha
  revisado, y se espera que algunas estén mal.
- El orden en que está escrita cada secuencia **no significa nada**: el conversor da la misma frase
  en cualquier orden. Por eso se le pregunta en qué orden la señaría él.
- Puede decir «no sé» o saltarse un ítem.

**Por cada ítem (bloques A, B, C, D, F y H)**

Leer la secuencia de señas, mostrar la frase y hacer estas preguntas. Anotar las respuestas
**literalmente**, sin corregirlas.

1. «¿Esta secuencia de señas es algo que una persona diría en LSC?» → **natural / rara / no se dice**.
   «¿En qué orden la señaría?» (columnas `secuencia_natural_en_LSC` y `orden_en_que_la_senaria`).
2. «¿La frase en español dice lo mismo que las señas?» → **sí / en parte / no**
   (columna `dice_lo_mismo`).
3. «¿Cómo la diría usted en español?» → texto libre (columna `como_lo_diria_usted`).
4. Lo demás que comente va en `observaciones`.

En el bloque D no hay frase: el conversor dice la lista de glosas. La pregunta 2 se cambia por
«¿Debería haber una frase para esta secuencia? ¿Cuál?».

**Bloque E (decisiones):** leer las opciones de la columna `opciones_bloque_E` y anotar cuál elige
en `eleccion_bloque_E`, con su comentario. Si duda, anotar «duda».

**Bloque G (preguntas abiertas):** preguntar y anotar la respuesta en `observaciones`.

**Prioridad si no alcanza el tiempo.** Seguir el orden de la hoja: A, B, C, E, D, F, H y G. El
orden sigue la prioridad de §9.2 de la especificación (primero las secuencias S-01 a S-36 y las
decisiones), con un cambio: las 15 secuencias del bloque C van antes que las decisiones (E) porque
varias decisiones se entienden mejor después de verlas. Si hace falta una segunda sesión, se retoma
donde quedó.

---

## 3. Qué hay en la hoja

| bloque | filas | ítems | contenido | equivale en §9.2 a |
|---|---|---|---|---|
| A | 1-9 | 9 | las 9 frases de ejemplo ya usadas con el autor (S-01 a S-09) | parte de B2 |
| B | 10-31 | 22 | las plantillas fijas T-02 a T-22, más T-03 con saludo (S-70). T-01 es la fila 5 (bloque A). Así están las 22 plantillas | parte de B2 |
| C | 32-46 | 15 | casos de reglas generales que dependen de decisiones por validar: ser/estar, BIENVENIDO, vocativos, preguntas, pronombre explícito. 14 son del inventario; la fila 39 no lo es | parte de B2, más 1 ítem fuera del inventario |
| E | 47-65 | 19 | decisiones de diseño con opciones (las 15 de §9.2, con la 14 y la 15 partidas en casos) | B5 |
| D | 66-70 | 5 | secuencias que hoy no dan frase y que quizá deberían | parte de B4 |
| F | 71-117 | 47 | el resto del inventario (S-10 a S-92 que no salen en A, B ni C) | resto de B2 |
| H | 118-167 | 50 | las 50 glosas solas, como las dice el modo aislado | B1 |
| G | 168-178 | 11 | preguntas abiertas sin secuencia | — |

La columna `dice_con_inventario` indica si el prototipo dice esa frase con
`--conversor inventario` (sí) o solo con `--conversor reglas` (no). Con `inventario` el prototipo
solo puede decir las 50 glosas solas y las 92 secuencias del inventario. Si se completan A, B, C, F
y H, el profesor habrá visto todas: las 92 secuencias y las 50 glosas solas.

**Lo que esta hoja no cubre** (está en §9.2 y se puede hacer en otra sesión):

- B3, una muestra estratificada de 30 conjuntos con frase fuera del inventario. Está en
  `docs/CONVERSOR_3a_SALIDAS_NO_RESPALDO.csv`, en las filas con `bloque_actividad_B = B3`.
- El resto de B4: un ítem por cada motivo de respaldo. Son 34 en §9.2, y esta hoja tiene 5.

---

## 4. Las secuencias

Las frases son las que produce hoy el conversor 3a-2. Una frase entre acentos graves (`así`) es un
respaldo: el conversor no arma frase y dice la lista de glosas.

### Bloque A: las 9 frases de ejemplo

| n | ref | señas | frase del conversor |
|---|---|---|---|
| 1 | S-01 | BUENOS DÍAS + TODOS | Buenos días a todos. |
| 2 | S-02 | BUENAS TARDES + TODOS | Buenas tardes a todos. |
| 3 | S-03 | TÍO + JUICIOSO | El tío es juicioso. |
| 4 | S-04 | FAMILIA + FELIZ | La familia está feliz. |
| 5 | S-05 / T-01 | POR FAVOR + SEÑA | La seña, por favor. |
| 6 | S-06 | HOLA + BIENVENIDO | Hola, bienvenido. |
| 7 | S-07 | HERMANO + TRABAJAR | El hermano trabaja. |
| 8 | S-08 | ¿DÓNDE? + HERMANO + VIVIR | ¿Dónde vive el hermano? |
| 9 | S-09 | HOLA + ¿CÓMO ESTÁS? | Hola, ¿cómo estás? |

### Bloque B: plantillas fijas

| n | ref | señas | frase del conversor |
|---|---|---|---|
| 10 | S-21 / T-02 | POR FAVOR + NOMBRE | El nombre, por favor. |
| 11 | S-22 / T-03 | TÚ + NOMBRE + ¿QUÉ? | ¿Cuál es tu nombre? |
| 12 | S-23 / T-04 | TÚ + SEÑA + ¿QUÉ? | ¿Cuál es tu seña? |
| 13 | S-24 / T-05 | GRACIAS + MUCHO | Muchas gracias. |
| 14 | S-25 / T-06 | CON GUSTO + MUCHO | Con mucho gusto. |
| 15 | S-26 / T-07 | GRACIAS + TODOS | Gracias a todos. |
| 16 | S-27 / T-08 | PERDÓN + PERMISO | Perdón, con permiso. |
| 17 | S-28 / T-09 | POR FAVOR + PERMISO | Permiso, por favor. |
| 18 | S-29 / T-10 | PERDÓN + ¿QUÉ? | Perdón, ¿qué? |
| 19 | S-30 / T-11 | PERDÓN + ¿CÓMO? | Perdón, ¿cómo? |
| 20 | S-31 / T-12 | BIEN + GRACIAS | Bien, gracias. |
| 21 | S-32 / T-13 | MÁS O MENOS + GRACIAS | Más o menos, gracias. |
| 22 | S-33 / T-14 | YO + BIEN + GRACIAS | Yo estoy bien, gracias. |
| 23 | S-34 / T-15 | GRACIAS + ADIÓS | Gracias, adiós. |
| 24 | S-35 / T-16 | GRACIAS + BUENAS NOCHES | Gracias, buenas noches. |
| 25 | S-36 / T-17 | MUCHO + GRACIAS + TODOS | Muchas gracias a todos. |
| 26 | S-70 / T-03 | HOLA + TÚ + NOMBRE + ¿QUÉ? | Hola, ¿cuál es tu nombre? |
| 27 | S-86 / T-18 | GRACIAS + BUENOS DÍAS | Gracias, buenos días. |
| 28 | S-87 / T-19 | GRACIAS + BUENAS TARDES | Gracias, buenas tardes. |
| 29 | S-88 / T-20 | GRACIAS + USTEDES | Gracias a ustedes. |
| 30 | S-89 / T-21 | GRACIAS + TÚ | Gracias a ti. |
| 31 | S-90 / T-22 | MAL + GRACIAS | Mal, gracias. |

### Bloque C: reglas generales que dependen de decisiones por validar

| n | ref | señas | frase del conversor | qué se pone a prueba |
|---|---|---|---|---|
| 32 | S-45 | NIÑA + CONTENTO | La niña está contenta. | «estar» y concordancia de género |
| 33 | S-47 | HERMANO + DIFERENTE | El hermano es diferente. | «ser» con DIFERENTE |
| 34 | S-73 | TÍO + ABURRIDO + NUNCA | El tío nunca está aburrido. | «estar» con ABURRIDO; NUNCA antes del verbo |
| 35 | S-53 | YO + HAMBRE | Yo tengo hambre. | verbo «tener» suplido |
| 36 | S-55 | TODOS + BIENVENIDO | Todos son bienvenidos. | BIENVENIDO con «ser» |
| 37 | S-56 | TÚ + BIENVENIDO | Tú eres bienvenido. | BIENVENIDO con TÚ; tratamiento de *tú* |
| 38 | S-92 | FAMILIA + BIENVENIDO | Bienvenida, familia. | BIENVENIDO como fórmula de bienvenida, sin «ser» |
| 39 | fuera de inventario | HOLA + HERMANO + BIENVENIDO | Hola, bienvenido, hermano. | lugar de la fórmula tras un saludo. Solo con `--conversor reglas` |
| 40 | S-15 | HOLA + ABUELO | Hola, abuelo. | vocativo tras saludo, sin artículo |
| 41 | S-91 | GRACIAS + ABUELO | Gracias, abuelo. | GRACIAS + vocativo |
| 42 | S-17 | HOLA + ¿CÓMO ESTÁS? + FAMILIA | Hola, ¿cómo está la familia? | ¿CÓMO ESTÁS? + persona: se lee como pregunta sobre un tercero |
| 43 | S-57 | ¿DÓNDE? + TÚ + VIVIR | ¿Dónde vives tú? | pronombre al final de la pregunta |
| 44 | S-67 | ¿POR QUÉ? + TÚ + TRABAJAR + MUCHO | ¿Por qué trabajas mucho tú? | pronombre al final, con MUCHO |
| 45 | S-63 | ¿QUIÉN? + USTEDES | ¿Quiénes son ustedes? | ¿QUIÉN? con plural |
| 46 | S-52 | TÚ + SENTIR + BIEN | Tú te sientes bien. | SENTIR como «sentirse»; pronombre explícito |

### Bloque E: decisiones de diseño (elegir una opción)

La primera opción es la que da hoy el conversor.

| n | ref | señas | opciones | qué se decide |
|---|---|---|---|---|
| 47 | E-01 | TÍO + JUICIOSO | «El tío es juicioso.» / «El tío está juicioso.» | «ser» o «estar» con JUICIOSO |
| 48 | E-02 | USTEDES + ABURRIDO | «Ustedes están aburridos.» / «Ustedes son aburridos.» | «ser» o «estar» con ABURRIDO |
| 49 | E-03 | HERMANO + DIFERENTE | «El hermano es diferente.» / «El hermano está diferente.» | «ser» o «estar» con DIFERENTE |
| 50 | E-04 | FAMILIA + FELIZ | «La familia está feliz.» / «La familia es feliz.» | «ser» o «estar» con FELIZ |
| 51 | E-05 | BUENOS DÍAS + TODOS | «Buenos días a todos.» / «Buenos días, todos.» | «a todos» o «, todos» |
| 52 | E-06 | COMER | «Comer.» / «Come.» | la seña sola: infinitivo o 3.ª persona |
| 53 | E-07 | TÍO + JUICIOSO | «El tío es juicioso.» / «Mi tío es juicioso.» | artículo o posesivo |
| 54 | E-08 | ¿DÓNDE? + TÚ + VIVIR | «¿Dónde vives tú?» / «¿Dónde vives?» | pronombre explícito en la pregunta |
| 55 | E-09 | TÚ + SENTIR + BIEN | «Tú te sientes bien.» / «Te sientes bien.» | pronombre explícito en la afirmación |
| 56 | E-10 | ¿POR QUÉ? + TÚ + TRABAJAR + MUCHO | «¿Por qué trabajas mucho tú?» / «¿Por qué trabajas tú mucho?» / «¿Por qué trabajas tanto?» | lugar del pronombre |
| 57 | E-11 | ¿DÓNDE? + TÚ + VIVIR | «¿Dónde vives tú?» / «¿Dónde vive usted?» / «¿Dónde vivís vos?» | tratamiento: *tú*, *usted* o *vos* |
| 58 | E-12 | BUENOS DÍAS + FAMILIA + FELIZ | «Buenos días, la familia está feliz.» / «Buenos días. La familia está feliz.» | coma o punto después del saludo |
| 59 | E-13 | POR FAVOR + NOMBRE | «El nombre, por favor.» / «Tu nombre, por favor.» / «Su nombre, por favor.» | redacción de T-02 |
| 60 | E-14a | ¿CÓMO ESTÁS? + HERMANO | «¿Cómo está el hermano?» / «Hermano, ¿cómo estás?» | ¿pregunta por el hermano o le pregunta al hermano? |
| 61 | E-14b | ¿CÓMO ESTÁS? + FAMILIA | «¿Cómo está la familia?» / «Familia, ¿cómo están?» | lo mismo con FAMILIA |
| 62 | E-15a | FAMILIA + BIENVENIDO | «Bienvenida, familia.» / «La familia es bienvenida.» | ¿se le da la bienvenida o se afirma que es bienvenida? |
| 63 | E-15b | TÚ + BIENVENIDO | «Tú eres bienvenido.» / «Bienvenido.» | BIENVENIDO con TÚ |
| 64 | E-15c | TODOS + BIENVENIDO | «Todos son bienvenidos.» / «Bienvenidos todos.» / «Hola a todos, bienvenidos.» | BIENVENIDO con TODOS |
| 65 | E-15d | HOLA + HERMANO + BIENVENIDO | «Hola, bienvenido, hermano.» / «Hola, hermano, bienvenido.» | lugar de la fórmula tras el saludo |

E-15d no está en §9.2 de la especificación. Se añadió porque la reverificación de 3a-2 señaló que,
en la voz, «Hola, bienvenido, hermano.» suena en tres tramos y que el orden más corriente sería
«Hola, hermano, bienvenido.». La regla no se cambió: decide el profesor.

### Bloque D: secuencias sin frase (¿debería haberla?)

| n | señas | lo que dice hoy | por qué no hay frase |
|---|---|---|---|
| 66 | COMER + MUCHO | `comer, mucho` | no hay sujeto, y el conversor no lo inventa («Come mucho») |
| 67 | HOLA + TÚ | `hola, tú` | saludo + pronombre no está previsto |
| 68 | HOLA + GRACIAS | `hola, gracias` | no se combina un saludo con una cortesía |
| 69 | PERDÓN + ABUELO | `perdón, abuelo` | cortesía + vocativo no está previsto (salvo GRACIAS) |
| 70 | YO + SENTIR | `yo, sentir` | ambiguo: «Lo siento», «Me siento…» |

### Bloques F y H

- **F (filas 71-117):** las 47 secuencias del inventario que no salen en A, B ni C. Se revisan
  con las mismas preguntas que A.
- **H (filas 118-167):** las 50 glosas solas, en el orden de id. Pregunta: «¿Así se dice esta seña
  sola en español?». Todas siguen el mismo criterio: forma de cita, con mayúscula y punto
  («Comer.», «Contento.», «Hambre.»), y las preguntas entre ¿?.

### Bloque G: preguntas abiertas sobre el diseño

168. **(R-03)** Si una misma seña se repite en el turno, ¿suele ser énfasis, plural (por ejemplo,
     NIÑO repetido) o un error? Hoy se dice una sola vez.
169. **(R-04)** ¿Hay casos en que el orden de las señas cambie lo que se quiere decir? Hoy el orden
     no cambia la frase.
170. **(R-08)** ¿Bastan cuatro señas por turno para las frases cortas de uso diario?
171. **(R-11)** ¿A quién se le habla así tras un saludo? Hoy sí: familia, niño, niña, abuelo, tío y
     hermano. Hoy no: mujer, hombre y personas.
172. **(R-13)** ¿Falta alguna combinación de cortesía de uso común, con GRACIAS, POR FAVOR, PERDÓN,
     PERMISO o CON GUSTO?
173. **(R-17)** Si no se configura, YO y TÚ concuerdan en masculino («Yo estoy contento»). ¿Es
     aceptable como valor por defecto?
174. **(R-25)** MUCHO junto a un adjetivo o a BIEN o MAL se dice «muy» («Yo estoy muy feliz»).
     ¿Es esa la lectura?
175. **(R-27)** YO + VIVIR no da frase, porque «Yo vivo» significa «estoy vivo». ¿Cómo se usa VIVIR
     sin un lugar?
176. **(R-35)** Cuando no hay frase, el prototipo dice las glosas en lista («hola, tú»). ¿Le sirve a
     quien escucha o lo confunde?
177. **(D-L1, R-34)** Las fórmulas se dicen sin exclamación: «Gracias.» y no «¡Gracias!». ¿Se
     pierde algo?
178. **(T-01, T-04)** SEÑA: ¿se entiende como «la seña personal» o como «una seña»?

---

## 5. Después de la sesión

**Qué se hace con cada respuesta** (§9.2 de la especificación):

| respuesta | acción |
|---|---|
| sí, y su versión coincide | el ítem queda «revisado por la persona señante (fecha), sin cambios» |
| sí, con otra versión | no se cambia nada solo por eso; su versión se anota como alternativa aceptada |
| en parte | se corrige la regla o la plantilla con su versión si el cambio es local, y se anota la decisión; si no es local, la secuencia pasa a «no cubierta» |
| no | se quita la plantilla, o se restringe la regla para que esa entrada vaya a respaldo; su versión puede entrar como plantilla nueva atribuida a la sesión |
| pregunta 1 = «no se dice» | la secuencia sale del inventario, aunque la frase sea buena |
| bloque D: «sí debería haber frase» | su frase entra como plantilla candidata, que se revisa en la sesión siguiente |
| bloque E | gana su elección; si duda, se mantiene la decisión del autor, marcada «sin acuerdo» |

**Registro.**

1. Guardar la hoja llena con un nombre que lleve la fecha, por ejemplo
   `docs/ACTIVIDAD_B_hoja_registro_2026-11-DD.csv`. No sobrescribir la hoja en blanco.
2. Cada cambio de regla sube la versión del conversor (`3a-3`…), se escribe primero en la
   especificación y después en el código y las pruebas, y se regeneran los dos CSV del conversor.
3. Proponer una entrada en DECISIONES_TESIS.txt con lo decidido.
4. Solo después de la sesión se puede escribir en la tesis «revisado por la persona señante», y
   solo para los ítems que vio.

**Qué se reporta en la tesis.**

- Los juicios ítem por ítem y los recuentos de sí, en parte y no por bloque, sin promedios con
  decimales.
- La diferencia literal entre cada frase y la versión del profesor.
- La cobertura de la verificación. Con `--conversor inventario`, si se completan A, B, C, F y H,
  quedan revisadas todas las salidas posibles: 50 glosas solas y 92 secuencias. Con
  `--conversor reglas`, el conversor puede dar 7.045 salidas con frase, y esta hoja muestra 145 de
  ellas (2,1 %): 50 glosas solas, 92 del inventario y 3 de fuera (filas 39, 60 y 61). Lo revisado
  no autoriza a decir que el resto está validado.
- No se presenta BLEU como medida principal: hay un solo evaluador y, a lo sumo, una referencia por
  ítem (§9.2 de la especificación).
