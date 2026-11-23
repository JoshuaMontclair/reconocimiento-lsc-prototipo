# Estado del mapeo gramatical (`src/grammar_mapper.py`)

**Fecha:** 16 de septiembre de 2026
**Alcance:** describir lo que el módulo hace HOY, con ejemplos ejecutados, y dejarlo clavado por
pruebas. **No** es una propuesta de rediseño: la rearquitectura es la actividad 3.a y está a la
espera de la persona señante (0.c).
**Pruebas que lo respaldan:** `tests/test_grammar_mapper.py` (89 pruebas).
**Cómo se obtuvieron los ejemplos:** ejecutando el módulo con
`python` desde la raíz del repositorio. Ninguna
salida de este documento fue supuesta; todas están además fijadas como salida esperada en el
archivo de pruebas.
**Archivos leídos, no modificados:** `src/grammar_mapper.py` (641 líneas, md5
`ce3ac6c8f7b841bbd02118eca0f2d320`) y `src/translator.py` (uso desde el prototipo).

---

## 1. Qué es y dónde entra en el sistema

Es la etapa que va del reconocimiento a la voz:

```
video → extractor → modelo CNN+BiLSTM → glosa(s) → GrammarMapper → enunciado → TTS
```

`src/translator.py` lo usa de dos maneras:

| Modo | Llamada | Cuándo |
|---|---|---|
| aislado | `grammar_mapper.translate([ens.clase])` — una glosa | por cada ventana aceptada (`_inferir_y_emitir`) |
| oración | `grammar_mapper.translate(glosas)` — la lista acumulada del turno | una vez, al pulsar fin de turno (`fin_de_turno`) |

(Líneas 816 y 833 de `src/translator.py` en la copia de trabajo del 16-sep-2026; el archivo lo están
modificando otros frentes, así que la referencia estable es el nombre del método.)

En los dos casos las glosas vienen de `SIGN_VOCABULARY`, nunca de texto libre, y el enunciado que
devuelve es exactamente lo que se manda a sintetizar y lo que queda escrito en el registro del
turno. El mapeador no guarda estado entre llamadas.

## 2. Qué hace hoy, paso a paso

1. **Normaliza la entrada.** Un `int` se resuelve con `SIGN_VOCABULARY`; un `str` pasa por
   `s.lower().strip()` y nada más. Lista vacía → `""`.
2. **Etiqueta cada glosa** con `GRAMMAR_CATEGORIES`: una categoría fija por seña y unos pocos
   rasgos (`persona`, `numero`, `genero`, `tipo`). Lo que no esté en esa tabla recibe la categoría
   `desconocido`.
3. **Prueba siete plantillas** (`PHRASE_TEMPLATES`) contra la tupla de categorías **en el orden de
   entrada**. La primera que coincide gana.
4. **Si no coincide ninguna, arma la frase con un orden de emisión fijo por categoría**, no por el
   orden en que llegaron las glosas:

   > saludos → primer sujeto (con artículo definido si declara género) → adverbios de frecuencia →
   > primer verbo conjugado en presente según la persona del sujeto → verbos siguientes en
   > infinitivo → objetos (con artículo) → adjetivos → demás adverbios → cortesía

5. **Si hay un interrogativo**, usa otro orden fijo y se salta el resto:
   `¿interrogativo + verbo + sujeto no pronominal + objetos + adjetivos + adverbios?`
6. **Capitaliza** la primera letra, o la que sigue a un `¿` inicial. No añade punto final.

Tamaño de las tablas, medido: 50 señas ↔ 50 entradas de categoría (cobertura 1:1), repartidas en
**sujeto 13, pregunta 7, adjetivo 7, adverbio 6, saludo 5, cortesía 4, objeto 4, verbo 4**.
`CONJUGATIONS` tiene 4 verbos × 3 personas (1, 2, 3), todas en singular.

## 3. Ejemplos reales de entrada y salida

Todas ejecutadas el 16-sep-2026.

### 3.1 Una glosa (lo que emite el modo aislado del prototipo)

| Categoría | Entrada | Salida |
|---|---|---|
| sujeto (pronombre) | `["yo"]` | `Yo` |
| sujeto (con género) | `["hermano"]` / `["niña"]` | `El hermano` / `La niña` |
| sujeto (plural sin género) | `["personas"]` | `Personas` |
| verbo | `["comer"]` / `["sentir"]` | `Come` / `Siente` |
| objeto | `["nombre"]` / `["seña"]` | `El nombre` / `La seña` |
| adjetivo | `["feliz"]` | `Feliz` |
| adverbio | `["bien"]` / `["nunca"]` | `Bien` / `Nunca` |
| saludo | `["hola"]` | `Hola` |
| cortesía | `["gracias"]` | `Gracias` |
| pregunta (frase completa) | `["¿cómo estás?"]` | `¿Cómo estás?` |
| pregunta (interrogativo) | `["¿dónde?"]` | `¿Dónde?` |

Las 50 señas presentadas solas dan salida no vacía (50/50). La tabla completa está en
`tests/test_grammar_mapper.py`, constante `SALIDA_DE_UNA_SOLA_GLOSA`.

### 3.2 Dos y tres glosas: en el caso general el orden de entrada no cambia la salida

En los casos de abajo, todas las permutaciones de la entrada dan una única salida. **Ojo: esto es
el caso general, no una regla sin excepciones.** Hay dos excepciones medidas, ambas en §5: dentro
de una misma categoría el orden se conserva (H7a) y, en turnos de exactamente dos glosas, el par
saludo + interrogativo cambia de salida al invertirse (H7b).

| Entrada | Salida |
|---|---|
| `["yo","comer"]` y `["comer","yo"]` | `Yo como` (las 2 permutaciones) |
| `["hermano","trabajar"]` y `["trabajar","hermano"]` | `El hermano trabaja` |
| `["yo","comer","mucho"]` | `Yo como mucho` (las 6 permutaciones) |
| `["familia","vivir","bien"]` | `La familia vive bien` (las 6) |
| `["yo","nunca","comer"]` | `Yo nunca como` (las 6) |
| `["perdón","yo","vivir"]` | `Yo vivo perdón` (las 6) |
| `["hola","yo","trabajar"]` | `Hola yo trabajo` (las 6) |
| `["¿dónde?","tú","vivir"]` | `¿Dónde vives?` (las 6) |

### 3.3 Preguntas

| Entrada | Salida |
|---|---|
| `["¿dónde?","tú","vivir"]` | `¿Dónde vives?` |
| `["¿quién?","comer"]` | `¿Quién come?` |
| `["¿dónde?","hermano","vivir"]` | `¿Dónde vive el hermano?` |
| `["¿cuándo?","yo","comer"]` | `¿Cuándo como?` |
| `["tú","¿cómo estás?"]` | `¿Cómo estás?` |
| `["hola","¿dónde?","tú","vivir"]` | `Hola ¿dónde vives?` |

### 3.4 Desde los IDs del modelo

| Entrada | Salida |
|---|---|
| `[30, 26]` | `Hola, ¿cómo estás?` |
| `[18, 6]` | `El hermano trabaja` |
| `[39, 43, 8]` | `¿Dónde vives?` |
| `[30, 38, 6, 0]` | `Hola yo trabajo gracias` |

### 3.5 Mayúsculas, espacios y acentos

| Entrada | Salida |
|---|---|
| `["HOLA"]`, `["Hola"]`, `["  hola  "]` | `Hola` |
| `["YO","COMER"]` | `Yo como` |
| `["TÚ","VIVIR"]` | `Tú vives` |
| `["¿DÓNDE?","TÚ","VIVIR"]` | `¿Dónde vives?` |
| `["Buenos Días"]` | `Buenos días` |
| `["adiós"]` vs `["adios"]` | `Adiós` vs `""` |
| `["tú","vivir"]` vs `["tu","vivir"]` | `Tú vives` vs `Vive` |

### 3.6 Fuera de vocabulario y lista vacía

| Entrada | Salida |
|---|---|
| `[]`, `[""]`, `["   "]` | `""` |
| `["perro"]`, `["casa"]` | `""` |
| `["yo","perro","comer"]` | `Yo como` (idéntica a `["yo","comer"]`) |
| `[999]`, `[50]`, `[-1]` | `""` |
| `[30, 999]` | `Hola` |

## 4. Qué NO hace

- **No concuerda en número.** `CONJUGATIONS` solo tiene formas de singular (personas 1, 2 y 3). Los
  13 sujetos incluyen 3 plurales (`ustedes`, `todos`, `personas`), así que las **12** combinaciones
  de sujeto plural × verbo (de 52 posibles) salen en desacuerdo:
  `["ustedes","comer"] → Ustedes come`, `["personas","trabajar"] → Personas trabaja`.
- **No concuerda en género ni número el adjetivo.** Ninguno de los 7 adjetivos declara `genero` ni
  `numero`: `["niña","contento"] → La niña contento`, `["ustedes","bienvenido"] → Ustedes bienvenido`.
- **No inserta cópula.** De los 91 pares sujeto × adjetivo, **0** salidas contienen `ser` o `estar`:
  `["yo","feliz"] → Yo feliz`, `["hermano","juicioso"] → El hermano juicioso`.
- **No respeta el orden de entrada** en el caso general: la salida se arma con el orden fijo de §2.
  `["perdón","yo","vivir"] → Yo vivo perdón`. Pero **no lo ignora del todo**, y conviene no
  enunciarlo de más: el orden de entrada llega a la salida por dos vías (H7a y H7b), una dentro de
  cada categoría y otra entre saludo e interrogativo en turnos de dos glosas.
- **No avisa de las glosas fuera de vocabulario**: no hay excepción, ni valor de retorno distinto,
  ni marca en la cadena. `["yo","perro","comer"]` da exactamente lo mismo que `["yo","comer"]`.
- **No marca tiempo**: todo sale en presente de indicativo. El corpus no tiene señas de tiempo y el
  módulo no tiene ninguna otra forma verbal.
- **No puntúa la oración**: no pone punto final (0 de las 50 salidas de una glosa terminan en punto)
  ni coma entre el saludo y el resto (`Hola yo trabajo`).
- **No acumula ni segmenta turnos**: eso lo hace `translator.py`; el mapeador recibe la lista ya
  cerrada y no guarda estado.

## 5. Hallazgos

Cosas que parecen errores. **No se corrigieron**: quedan clavadas como salida esperada en
`tests/test_grammar_mapper.py` (clase `TestHallazgos`, salvo donde se indique otra), con el ejemplo
que las muestra.

| # | Hallazgo | Ejemplo que lo muestra |
|---|---|---|
| H1 | Sin concordancia de número | `["ustedes","comer"] → Ustedes come` |
| H2 | Sin concordancia de género en el adjetivo | `["niña","contento"] → La niña contento` |
| H3 | Sin cópula | `["yo","feliz"] → Yo feliz` |
| H4 | Objeto sin verbo y artículo incorrecto | `["yo","hambre"] → Yo la hambre` (en español, «el hambre») |
| H5 | Solo cuenta el primer sujeto; los demás desaparecen | `["hermano","tío","comer"] → El hermano come` |
| H6 | Solo se conjuga el primer verbo; el resto sale en infinitivo suelto | `["yo","comer","trabajar"] → Yo como trabajar` |
| H7a | **Dentro de una misma categoría el orden de entrada SÍ manda.** Los tokens se recogen con comprensiones de lista sobre `tagged`, que conservan el orden de llegada | `["hola","buenos días"] → Hola, buenos días` pero `["buenos días","hola"] → Buenos días, hola`; `["yo","seña","nombre"] → Yo la seña el nombre` pero `["yo","nombre","seña"] → Yo el nombre la seña` |
| H7b | **Y también manda ENTRE categorías distintas**, en turnos de exactamente dos glosas: la plantilla `("saludo","pregunta")` casa solo en ese orden y une con coma; invertida no casa, la secuencia cae en las reglas generales y el prefijo se une con espacio. Son los 5 saludos × 6 interrogativos = 30 pares. Los 5 pares con `¿cómo estás?` son la excepción: su `tipo` es `frase_completa` y esa rama también une con coma | `["hola","¿dónde?"] → Hola, ¿dónde?` pero `["¿dónde?","hola"] → Hola ¿dónde?` (sin coma); en cambio `["hola","¿cómo estás?"]` y `["¿cómo estás?","hola"]` dan los dos `Hola, ¿cómo estás?` |
| H7c | La sensibilidad al orden **entre** categorías solo existe en turnos de dos glosas: `_try_templates` compara la tupla de categorías del turno entero, así que una tercera glosa basta para que la plantilla deje de casar. Medido: de las 12.572 tercias cuyas tres categorías son distintas entre sí, **0** son sensibles al orden | `["hola","¿dónde?"] → Hola, ¿dónde?` pero `["hola","¿dónde?","tú"] → Hola ¿dónde?` en las 6 permutaciones |
| H8 | Con interrogativo se pierden la cortesía y cualquier segundo interrogativo, porque `_build_question` se llama antes de añadir la cortesía y solo mira `preguntas[0]` | `["gracias","¿cómo estás?"] → ¿Cómo estás?`; `["¿qué?","¿dónde?","tú","vivir"] → ¿Qué vives?` |
| H9 | Puntuación incompleta: sin punto final, sin coma tras el saludo, y el `¿` interno no dispara mayúscula porque `_capitalize` solo mira el primer carácter | `["hola","¿dónde?","tú","vivir"] → Hola ¿dónde vives?` |
| H10 | En preguntas, un sujeto plural queda pospuesto y en desacuerdo | `["¿por qué?","ustedes","trabajar"] → ¿Por qué trabaja ustedes?` |
| H11 | Código inalcanzable con el vocabulario real: la rama de `gustar` (ese verbo no es una de las 50 señas); `PREPOSITIONS` (ningún objeto tiene `tipo: "lugar"`); la contracción `al` (ningún verbo tiene `tipo: "movimiento"`); el artículo indefinido de `ARTICLES` (`_get_article` solo lee `def`); la conjugación por terminación (los 4 verbos están en `CONJUGATIONS`); y las plantillas de `afirmacion`/`negacion` junto con las ramas «sí»/«no» (ninguna seña recibe esas categorías) | verificado en `TestHallazgos.test_h11_codigo_inalcanzable` y `TestPlantillas.test_plantillas_de_afirmacion_y_negacion_son_inalcanzables` |
| H12 | Rasgos declarados que no cambian ninguna salida: `numero` en los sujetos y `tipo` en verbos, objetos y adjetivos. Los `tipo` que SÍ tienen efecto son dos: `frecuencia` en adverbios y `frase_completa` en preguntas | `["hermano","vivir"] → El hermano vive` y `["hermano","comer"] → El hermano come`: `tipo` `estado` vs `accion` no cambia nada; en cambio `["¿cómo estás?","yo","comer"] → ¿Cómo estás?` frente a `["¿dónde?","yo","comer"] → ¿Dónde como?` |
| H13 | Una glosa perdida no deja rastro: la salida es idéntica a la de la secuencia sin ella, así que ni el registro del turno ni el usuario pueden saber que algo se descartó | `translate(["yo","perro","comer"]) == translate(["yo","comer"])` (`TestFueraDeVocabulario`) |
| H14 | Sin normalización de acentos: perder una tilde convierte la glosa en desconocida y puede cambiar la conjugación | `["tu","vivir"] → Vive` en vez de `Tú vives` (`TestMayusculasYAcentos`) |
| H15 | La entrada no se valida: un valor que no sea `int` ni `str` levanta `AttributeError` desde `.lower()` | `translate([None])` (`TestFueraDeVocabulario`) |
| H15b | Y un booleano ni siquiera falla: como `isinstance(True, int)` es verdadero en Python, entra por la rama de los IDs y se traduce en vez de rechazarse. Sin consecuencia práctica hoy (el prototipo pasa `str` o `int`), pero es un tipo que el rediseño 3.a querrá validar | `translate([True]) → Buenos días` (id 1) y `translate([False]) → Gracias` (id 0) |
| H16 | `_conjugate` con una persona fuera de 1/2/3 es incoherente entre sus dos ramas: la tabla devuelve el infinitivo y la terminación devuelve la 3.ª persona. Hoy inalcanzable (persona solo vale 1, 2 o 3) | `_conjugate("comer", 9) → "comer"` vs `_conjugate("bailar", 9) → "baila"` (`TestAuxiliaresInternos`) |

También quedan pendientes dos comentarios desactualizados dentro del archivo, ya anotados en el
registro (N-14): el comentario de las líneas 65-66 dice «Table 1 / Figure 1» donde va Tabla 2, y el
docstring de `_apply_grammar_rules` sigue diciendo «SOV → SVO», premisa retirada del anteproyecto
(§2.2.5). **No los toqué**: van con la rearquitectura de 3.a.

## 6. Cifras medidas

Todas calculadas por barrido exhaustivo sobre las 50 señas el 16-sep-2026.

| Cifra | Valor |
|---|---|
| Señas del vocabulario | 50 |
| Entradas en `GRAMMAR_CATEGORIES` | 50 (cobertura 1:1, sin sobrantes ni faltantes) |
| Categorías realmente usadas | 8 de las 12 nombradas en el comentario del código |
| Plantillas declaradas / alcanzables | 7 / 5 |
| Verbos / formas conjugadas | 4 verbos × 3 personas, todas de singular |
| Sujetos plurales | 3 de 13 |
| Adjetivos que declaran género o número | 0 de 7 |
| Pares sujeto × verbo con desacuerdo de número | 12 de 52 |
| Pares sujeto × adjetivo con cópula en la salida | 0 de 91 |
| Señas solas con salida no vacía | 50 de 50 |
| Salidas de una glosa que terminan en punto | 0 de 50 |
| Pares de señas distintas sensibles al orden de entrada | 188 de 1.225 (15,3 %) |
| — de ellos, pares de la MISMA categoría (H7a) | 158 de 188 |
| — de ellos, pares de categorías DISTINTAS (H7b) | 30 de 188 (5 saludos × 6 interrogativos) |
| Tercias de señas distintas sensibles al orden de entrada | 6.669 de 19.600 (34,0 %) |
| — de ellas, tercias cuyas 3 categorías son distintas entre sí (H7c) | 0 de 12.572 |
| Glosas que no aportan nada a la salida, en tercias | 7.374 de 58.800 (12,5 %) |
| Tercias con al menos una glosa perdida | 5.942 de 19.600 (30,3 %) |

Reparto de las glosas perdidas en tercias, por categoría: sujeto 4.292 · cortesía 1.140 ·
pregunta 898 · adjetivo 336 · adverbio 288 · verbo 228 · objeto 192.

Reparto de los 188 pares sensibles al orden, por par de categorías: sujeto-sujeto 78 ·
**pregunta-saludo 30** · adjetivo-adjetivo 21 · pregunta-pregunta 21 · saludo-saludo 10 ·
adverbio-adverbio 10 · cortesía-cortesía 6 · objeto-objeto 6 · verbo-verbo 6.

El renglón en negrita es el único que **cruza dos categorías**; los otros ocho son pares de la
misma categoría (78 + 21 + 21 + 10 + 10 + 6 + 6 + 6 = 158). De ahí el reparto 188 = 158 + 30, y de
ahí que no se pueda decir que el orden de entrada se ignora entre categorías distintas.

## 7. Preguntas abiertas para la rearquitectura (3.a)

Se listan sin proponer respuesta: son decisiones de la actividad 3.a, con la persona señante o
intérprete (0.c).

**Sobre el vocabulario y sus rasgos**
1. ¿Una seña puede tener más de una categoría? Hoy `juicioso` solo es adjetivo, `permiso` solo
   objeto, `bien` solo adverbio, y en LSC su uso depende del contexto.
2. ¿Qué rasgos hay que declarar por seña para que la concordancia sea posible (género y número del
   adjetivo, número del verbo, animacidad del sujeto), y quién los valida?
3. ¿Las 3 señas plurales del léxico (`ustedes`, `todos`, `personas`) son las únicas, o hay señas
   cuya pluralidad se marca con repetición o con espacio y no aparece en la glosa?

**Sobre la unidad de trabajo**
4. ¿Qué es un «turno»? Hoy lo delimita una tecla en `realtime.py`; ¿debe delimitarlo la persona
   señante, una pausa medida, o una marca no manual?
5. ¿Debe el orden de llegada de las glosas influir en el enunciado? Si la respuesta es sí, ¿cómo se
   distingue una secuencia que el reconocedor entregó desordenada de una intención de orden? Hoy la
   respuesta del módulo es **incoherente y no declarada**: el orden influye dentro de una misma
   categoría (H7a) y en el par saludo + interrogativo de dos glosas (H7b), y no influye en el resto
   —188 de los 1.225 pares sí, los otros 1.037 no—, sin que nada en el código diga cuál de los dos
   comportamientos es el pretendido.
6. ¿Qué se hace cuando en un turno llegan dos sujetos o dos verbos (H5, H6)?

**Sobre lo que no se reconoce**
7. Cuando llega una glosa fuera de vocabulario o de baja confianza, ¿qué debe ver y oír la persona:
   silencio, el enunciado incompleto, una marca explícita de hueco, o el rechazo del turno entero?
8. ¿El descarte silencioso de H13 debe quedar registrado en el log del turno aunque no se diga en
   voz alta?
9. ¿Cómo se relaciona esto con el umbral de confianza de D-09, que ya rechaza ventanas antes de que
   el mapeador vea nada?

**Sobre la salida**
10. ¿Qué registro de habla se espera: la frase mínima («Yo como») o la frase cortés completa?
11. ¿Cómo debe puntuarse el enunciado que va al TTS (H9), sabiendo que la puntuación cambia la
    prosodia de la voz sintética?
12. ¿Qué es un enunciado «aceptable» para la evaluación? Sin ese criterio no hay cómo medir si el
    rediseño mejora: hoy solo se puede afirmar que reproduce o no la salida de estas 89 pruebas.

**Sobre el método de validación**
13. ¿Con cuántos enunciados de referencia, producidos por quién y sobre qué guion, se va a juzgar el
    módulo rediseñado?
14. ¿Qué de lo que hoy hace debe conservarse a propósito? Cada prueba de `TestHallazgos` que el
    rediseño cambie tiene que cambiarse a conciencia, no por accidente.

## 8. Cómo reproducir

```bash
cd "<raíz del repositorio>"
python -m unittest tests.test_grammar_mapper -v     # 89 pruebas
python -m src.grammar_mapper                        # CLI: 25 casos + el vocabulario
```

La suite completa del proyecto, con este módulo incluido:

```bash
python -m unittest tests.test_prototipo tests.test_tts tests.test_reextract \
  tests.test_representacion tests.test_umbral tests.test_analisis_errores \
  tests.test_latencia_resumen tests.test_linea_base tests.test_grammar_mapper
```

Las pruebas no necesitan TensorFlow, mediapipe, corpus ni audio: el módulo es texto puro y la suite
corre en centésimas de segundo.

**La red de seguridad se comprobó.** Para verificar que estas pruebas de verdad atrapan un cambio de
comportamiento (y no solo pasan), el 16-sep-2026 se aplicaron seis mutaciones **en memoria**, sin
tocar el archivo, y se volvió a correr el módulo de pruebas. Las seis se detectaron:

| Mutación aplicada en memoria | Pruebas que fallan |
|---|---|
| dar a `comer` la forma plural `comen` (concordancia de número) | 13 |
| pasar `feliz` de adjetivo a verbo (cópula) | 2 |
| cambiar el género de `hambre` de `f` a `m` | 7 |
| eliminar la plantilla `("saludo","saludo")` | 1 |
| dar género femenino a `personas` (artículo nuevo) | 5 |
| quitar a `nunca` el tipo `frecuencia` | 5 |

Sin mutar: 89 pruebas, 0 fallos.

En la verificación independiente del 16-sep-2026 se aplicaron ocho mutaciones más, esta vez sobre el
archivo real y restaurándolo tras cada una (md5 `ce3ac6c8f7b841bbd02118eca0f2d320` antes y después).
Cuatro se detectaron: `_capitalize` sin capitalizar (187 fallos), la entrada sin `.strip()`
(1 fallo), `_build_question` mirando `preguntas[-1]` (2) y los saludos unidos con espacio en vez de
coma (2). Las otras cuatro, todas ellas permutaciones del **orden de emisión del tramo final** (objetos ↔ adjetivos ↔ adverbios pospuestos, en las reglas generales y dentro de
`_build_question`), NO se detectaban: ninguna prueba combinaba un objeto con un adjetivo o con un
adverbio pospuesto en el mismo turno, aunque el módulo sí produce salida distinta
(`["yo","hambre","feliz"] → Yo la hambre feliz`). El hueco se cerró añadiendo dos pruebas
(`test_orden_de_emision_entre_objeto_adjetivo_y_adverbio_pospuesto` y
`test_orden_de_emision_completo_dentro_de_la_pregunta`), que son las que llevan la cuenta de 84 a 86;
con ellas, las cuatro mutaciones de orden pasan a fallar.

**Corrección del 16-sep-2026 (tercera pasada).** La misma verificación independiente encontró que el
hallazgo H7, tal como estaba redactado, era **falso en su segunda mitad**: decía que la afirmación
«el orden de entrada se ignora» valía *entre categorías distintas*, y no es así. Recontado por
barrido exhaustivo de los 1.225 pares, los 188 sensibles al orden se reparten en 158 de la misma
categoría **más 30 que cruzan dos categorías** (los 5 saludos × 6 interrogativos, por la plantilla
`("saludo","pregunta")`, que casa solo en ese orden). El propio reparto por pares de categorías de
§6 ya lo decía —«pregunta-saludo 30»— y la redacción anterior no reparó en ello. El hallazgo se
partió en H7a (dentro de una categoría), H7b (entre saludo e interrogativo) y H7c (solo en turnos de
dos glosas, con las 12.572 tercias de categorías distintas midiendo 0 sensibles), y se añadieron tres
pruebas que fijan el contraejemplo cruzado y el rincón de los booleanos de H15b, llevando la cuenta
de 86 a 89:

| Prueba añadida | Qué clava |
|---|---|
| `test_h7b_el_orden_manda_entre_saludo_e_interrogativo` | `["hola","¿dónde?"] → Hola, ¿dónde?` frente a `["¿dónde?","hola"] → Hola ¿dónde?`, más el barrido de los 30 pares y la excepción de `¿cómo estás?` |
| `test_h7c_el_cruce_de_categorias_solo_afecta_a_turnos_de_dos` | que una tercera glosa devuelve el par a ser insensible al orden |
| `test_los_booleanos_no_rompen_sino_que_se_traducen_como_ids` | `translate([True]) → Buenos días`, `translate([False]) → Gracias` |

La lección para 3.a es de método, no solo de dato: la frase «el orden de entrada se ignora» del
docstring del módulo (y la equivalente del README) es **incompleta en los dos sentidos**, y la
formulación que sí sostienen los datos es que el orden de entrada llega a la salida por dos vías
—dentro de una categoría, y en la plantilla `("saludo","pregunta")`—.

**Verificación independiente del 16-sep-2026 (cuarta pasada).** Se repitió el barrido exhaustivo
desde cero, sin partir de las cifras anteriores, y **todas** las de §6 se reprodujeron exactamente
(1.225 pares → 188 sensibles = 158 + 30; 19.600 tercias → 6.669; 12.572 tercias de categorías
distintas → 0; 7.374 de 58.800 glosas sin aporte; 5.942 tercias con pérdida). Los 79 ejemplos de
entrada/salida de §3–§5 se volvieron a ejecutar uno por uno y los 79 coinciden. La red de seguridad
se volvió a medir con **22 mutaciones aplicadas al archivo real y restaurándolo tras cada una**
(md5 `ce3ac6c8f7b841bbd02118eca0f2d320` antes y después de cada una): 21 fallaron alguna prueba; la
única que no, la contracción `al` → `a el`, es una mutación de control sobre una rama que H11c ya
declara inalcanzable, así que no detectarla es el resultado correcto. Entre las atrapadas están las
dos ramas de prefijo de `_build_question` (unir con coma en vez de espacio y al revés), el orden de
emisión del tramo final en las reglas generales y en la pregunta, y el rincón de los booleanos.

Se corrigieron además tres afirmaciones equivocadas, todas de alcance excesivo y ninguna en una
aserción: dos en comentarios del archivo de pruebas y la tercera también en la fila H12 de §5 de este
documento. No se tocó el módulo.

| Dónde | Decía | Dice ahora |
|---|---|---|
| docstring de `TestPlantillas` | que las plantillas eran «las únicas reglas sensibles» al orden de entrada | que no lo son: 148 de los 188 pares sensibles al orden no casan con ninguna plantilla en ninguno de los dos órdenes (p. ej. `["gracias","por favor"] → Gracias, por favor` frente a `["por favor","gracias"] → Por favor, gracias`), y ninguna de las 6.669 tercias sensibles puede casar, porque no hay plantillas de 3 categorías |
| `test_turno_largo_pierde_casi_todo` | «las otras 43 glosas» | 44 (sobreviven 6 de 50) |
| `test_h12` y la fila H12 de §5 | que `frecuencia` en adverbios era el único `tipo` con efecto observable | que son dos: también `frase_completa` en preguntas, que es justo de lo que depende la excepción de `¿cómo estás?` en H7b — `["¿cómo estás?","yo","comer"] → ¿Cómo estás?` frente a `["¿dónde?","yo","comer"] → ¿Dónde como?` |

La primera y la tercera son el mismo error de enunciado que ya se había corregido en H7 —una regla
general enunciada de más—, esta vez sobrevivido en un docstring secundario y en una fila de la tabla
de hallazgos; queda anotado como aviso para 3.a de que
las afirmaciones de alcance («solo», «únicas», «siempre») sobre este módulo conviene cerrarlas con
el barrido que las contradiría.
