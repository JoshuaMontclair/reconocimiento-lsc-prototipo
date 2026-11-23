# Propuesta de valores para los parámetros abiertos del protocolo 4.a (PD-01 a PD-31)

**Estado: PROPUESTA para decidir con el director. No está aprobada, no cambia ningún archivo del prototipo,
ningún modelo, umbral ni partición, y no autoriza a ejecutar ninguna sesión.**
Trabajo de grado 281-2026-1 · 22 de septiembre de 2026 · Complementa a `docs/PROTOCOLO_EXPERIMENTAL_4a.md`
(borrador del 15-sep-2026), cuya sección 12 deja 31 parámetros abiertos con opciones, pro y contra. Aquí cada
uno recibe un valor recomendado con su justificación (sección 2), se deriva de esos valores un guion de sesión
completo con su duración estimada (sección 3), se listan los cambios de código que exigirían (sección 4) y se
reducen a ocho las preguntas que de verdad dependen del director (sección 5). La sección 6 recoge tres
hallazgos de esta revisión que el director debe conocer aunque no sean parámetros. Lo que se adopte se
registra como D-nn en `DECISIONES_TESIS.txt` antes de la primera sesión, como pide el protocolo.

---

## 0. Cómo leer este documento

### 0.1 Citas

Misma convención que el protocolo. `[MET a-b]` remite a las líneas a-b de
`Anteproyecto_FINAL_LaTeX/09_metodologia.tex`; `ALC` a `06_alcances.tex`; `ETI` a `07_etica.tex`; `RES` a
`10_resultados.tex`; `REC` a `12_recursos.tex`; `CRO` a `11_cronograma.tex`. `[REG a-b · X-nn]` remite a las
líneas a-b de `Tesis/DECISIONES_TESIS.txt` y a su entrada. `[PROT n]` remite a una sección del protocolo 4.a.
Las citas al anteproyecto se comprobaron el 22-sep-2026 con `grep` y lectura de las líneas (anexo A). Las citas
a código nombran archivo y función o argumento, sin número de línea.

### 0.2 Qué se midió en esta revisión y qué no

Ninguna cifra de este documento procede de una sesión de los bloques: no se abrió la cámara, no se reprodujo
audio y no se escribió nada en `registros/`, `data/` ni `models/`. Se trabajó sobre copias en la carpeta
temporal (SHA256 idénticos a los originales) y se sintetizó audio a archivos temporales que se borraron.
Comandos y salidas en el anexo A; los criterios se fijaron por escrito **antes** de calcular (anexo B).

| Dato medido hoy | Valor | Fuente | Para qué se usa |
|---|---|---|---|
| Ventanas del registro existente cuyo N objetivo no fue 75 | **16 de 71** (N entre 37 y 74); `fps_medida_antes` mediana 29,97 fot./s, mínimo 15,00; las 71 cerraron por fotogramas | copias de `registros/latencia_2026-09-14.jsonl` y `latencia_2026-09-15.jsonl` (73 registros, 71 ensayos, 4 sesiones) | PD-09 |
| \|W − 2,5 s\| con la ventana automática (como se grabó) frente a N fijo de 75 (estimado al ritmo real de cada ventana) | p90 **74,8 ms** frente a **17,9 ms** (diferencia 56,9 ms); 7 de 71 ventanas se apartaron más de 0,1 s; la peor duró 1,204 s (N = 37) | mismas copias; criterio C1 | PD-09 |
| Intervalo t0 → t0 entre ensayos consecutivos del autor, sin hoja ni declaración (excluidas 3 pausas ≥ 60 s) | n = 64: mediana 4,54 s, p90 27,25 s, mínimo 2,56 s | mismas copias; criterio C3 | Solo verosimilitud del ciclo (3.11) |
| Duración del audio de las 50 locuciones (los enunciados de `grammar_mapper.translate([glosa])`, voz Paulina, 150 ppm) | media **0,687 s**, mediana 0,683, p90 0,929, mín 0,325 («¿Qué?»), máx 1,214 («Buenas noches»); 1 canal, 22 050 Hz, 16 bits | síntesis con `say -o` a temporales, medidos con `afinfo` y borrados; criterio C2 | Ciclo (3.11) |
| Duración de los clips con que se entrenó el modelo vigente (`data/processed_holistic_mascara_rostro/lengths.npy`, que es el `processed_dir` de `modelo_canonico.procedencia.json`) | n = 1000; mediana 67 fotogramas = **2,792 s** a 24 fot./s; p90 3,375 s; **29,2 % ≤ 2,5 s y 68,0 % ≤ 3,0 s** | criterio C5 (solo descriptivo) | PD-09 y hallazgo 6.1 |
| Voces en español del sistema | solo `es_ES` y `es_MX`; **ninguna `es_CO`** | `say -v '?'` | PD-29 |
| Cámaras y salida de audio del sistema | cámara integrada del MacBook Pro y cámara de continuidad de un iPhone; **ninguna cámara web externa**; salida por defecto: parlantes integrados | `system_profiler SPCameraDataType SPAudioDataType` | PD-08, PD-24 |

Datos leídos del registro, no medidos hoy: tasa del bucle 30,4 fot./s con 75 fotogramas por ventana [REG 955 ·
N-26]; L_resp media 34,3 ms con la síntesis sin incluir [REG 960 · N-26; REG 1195 · N-32]; síntesis desde la
cache con mediana 0,144 ms [REG 1701 · N-45]; rechazo del 16,0 % con umbral 0,20 en la prueba LOSO de
señantes nuevos [REG 1244 · N-33]; rechazo del 62,7 % del autor con umbral 0,5 [REG 963-964 · N-26]. **L_resp con
la cache de voz no se ha medido todavía en vivo** [REG 1719 · N-45]; aquí se estima en ≈34 ms (N-26 más
0,144 ms) y se declara como estimación.

### 0.3 Regla de la columna «quién decide»

- **Estudiante (se declara)**: el anteproyecto ya fija el asunto o es de puro método (cómo se ejecuta algo que
  no cambia qué se mide ni para quién). Se adopta con el valor recomendado y se declara en el D-nn.
- **Director**: toca la validez de RE3 o RE5 (qué cuenta como acierto, qué entra en el denominador, qué hace
  el sistema ante un caso de §2.2.6), la ética (§1.5) o recursos no presupuestados u horas de la persona.

---

## 1. Resumen

1. **Bloque 1: r = 4 repeticiones por seña**, en rondas de las 50 señas con orden aleatorio por ronda y
   semilla anotada; 200 presentaciones, el mismo n que un pliegue LOSO. Dos sesiones de dos rondas,
   **≈ 58 min de presencia por sesión** en el escenario central (62-72 min en los pesimistas).
2. **Ventana de 2,5 s con N fijo** (`--ventana-frames 75`, o el N que dé el piloto con la cámara elegida),
   no automática: en el registro existente la automática se desvió de 2,5 s más de 0,1 s en 7 de 71
   ventanas, hasta 1,3 s. Se eleva al director una pregunta sobre 2,5 frente a 3,0 s (hallazgo 6.1).
3. **Cámara web externa** Full HD (el recurso presupuestado) en trípode a 200 cm, pantalla del Mac
   **orientada solo al operador**; la persona no ve el resultado.
4. **La persona activa la ventana con un pedal** que emite ESPACIO, si el director aprueba ese gasto; si no,
   el operador pulsa a la señal de la persona y se declara la desviación de [MET 300-301].
5. **Tarjetas** para la glosa, la declaración de variante (sí / no / no estoy segura) y la petición de
   repetir; video de referencia del corpus a demanda, desde la copia local restringida.
6. **Máximo dos intentos** por presentación; cifra principal: top-1 y top-3 del **primer intento** sobre
   las ejecuciones con predicción (PD-15 A), con la cifra del sistema completo y la exactitud entre
   aceptadas al lado. `src/exactitud_bloque1.py` ya calcula todas esas variantes.
7. **Seña cortada**: columna nueva `cortada` en la hoja; al inicio se aborta con X y se repite; al final se
   marca, no se excluye y se reporta aparte. Sin detección automática antes de los bloques.
8. **Audio dudoso** según D-13: se conserva, se cuenta y no se repite; solo se hace una pausa de revisión si el
   contador sube en tres registros seguidos.
9. **Retardo de audio**: 30 ensayos, solo audio con un teléfono, corrida dedicada con `--umbral 0`,
   detección por guion revisada a mano.
10. **Ocho preguntas para el director** (sección 5); las demás las fija el estudiante y se declaran.

---

## 2. Los 31 parámetros

| PD | Qué decide | Valor RECOMENDADO | Justificación | Costo o riesgo de la alternativa | Decide |
|---|---|---|---|---|---|
| 01 | Repeticiones por seña en el Bloque 1 | **r = 4** (200 presentaciones). Si las horas solo alcanzan una sesión: r = 2 y se declara | Lo fija 4.a según las horas acordadas [MET 223-225; MET 475-476]. 4 es el número de repeticiones por seña y señante de LSC50 [MET 51-53], verificado por conteo (`caracterizacion.md`, sección 4). Con r = 4 el Bloque 1 tiene 200 presentaciones, el tamaño de un pliegue LOSO (200 / 199 / 198 / 200 / 200 [PROT 2.1]), y se reporta «junto al» LOSO [MET 265-268] con el mismo n. Duración en 3.11 | r = 2: la mitad de horas, pero 2 ejecuciones por clase dan recall por clase en {0; 0,5; 1} [MET 279-280 pide P, R y F1 por clase]. r > 4: más horas de la persona que cotizar, porque el rubro depende de las repeticiones que fije 4.a [REC 140-148] | Director (horas de la persona) |
| 02 | Ejecuciones del Bloque 2 | **4 rondas de las 50 señas (200 normales) + 70 de casos límite** (PD-17). Regla de parada fijada antes: si tras 4 rondas hay menos de 100 `aceptada`, se añaden rondas completas hasta llegar a 100 o a 8 rondas | El bloque acumula las ejecuciones que exige reportar la latencia con media, DE, mediana y p90 [MET 248-249; MET 324-326]. L_total y L_resp solo existen en ejecuciones aceptadas [PROT 6.3]; con n ≥ 100 el p90 deja al menos 10 observaciones por encima. La parada depende de un conteo, no del valor de ninguna latencia | Parar por estabilidad de la media o del p90 (opción C del protocolo) es parar mirando el resultado. Igualar solo al Bloque 1 deja sin referencia si el Bloque 1 no existe [MET 269-271] | Estudiante |
| 03 | Orden de presentación | **Rondas**: cada ronda contiene las 50 señas en un orden aleatorio distinto; semilla anotada en la hoja antes de empezar; la lista se imprime en la hoja pre-llenada (cambio 4.4) | Reparte fatiga, aprendizaje y deriva de luz entre señas; permite cerrar cada sesión al final de una ronda, de modo que todas las señas tengan el mismo r en cada corte | Orden fijo: confunde seña con posición. Aleatorio sin rondas: una seña puede concentrarse al final | Estudiante |
| 04 | Sesiones y ejecuciones por sesión | **Bloque 1: 2 sesiones de 2 rondas** (100 presentaciones y ≈ 116 ejecuciones por sesión en el escenario central), en días distintos, mismo montaje. **Bloque 2: 1-2 sesiones** del autor | [MET 274-276] exige fijar las ejecuciones por sesión en 4.a. En el escenario central una sesión de dos rondas dura ≈ 58 min con la persona (3.11) | Una sola sesión de 4 rondas: ≈ 100 min, con más fatiga. Varias sesiones cortas: más montajes que reproducir | Director (horas) |
| 05 | Pausas | **5 min al final de cada ronda** y **pausa a demanda** en cualquier momento entre presentaciones (nunca con una ventana abierta); cada pausa se anota con hora y duración | Participación, no extracción [ETI 78-86]; la pausa entre rondas no parte rondas | Pausa fija cada k presentaciones: puede no coincidir con la fatiga real | Estudiante |
| 06 | Iluminación y fondo | **Luz artificial constante** de la sala, persianas cerradas; **fondo liso y uniforme**, sin personas ni pantallas detrás; iluminancia medida con una aplicación del teléfono a la altura del rostro al inicio y al final de cada sesión (se declara aproximada) | Se fijan en 4.a y se documentan las diferencias con el corpus [MET 274-276; MET 498-502]; el corpus declara luz «blanca y homogénea» y cortina azul a 10 cm (`caracterizacion.md`, condiciones de registro) | Solo descripción cualitativa: poco reproducible. Replicar la cortina azul: posible pero no presupuestado | Estudiante |
| 07 | Distancia y encuadre | **Cámara a 200 cm** de la persona, de pie, encuadre **de las rodillas a la máxima extensión vertical de los brazos**, como COLOR_BODY; marca en el piso; si la cámara no logra ese encuadre a 200 cm, se ajusta la distancia y se anota | Son las condiciones de registro del corpus (`caracterizacion.md`: distancia 200 cm, encuadre COLOR_BODY); acercarse a ellas reduce las diferencias que acotan el Bloque 1 [MET 498-502; ALC 35-41] | Encuadre libre: varía entre sesiones. Persona sentada: cambia el encuadre respecto del corpus (se declara si la persona lo prefiere) | Estudiante |
| 08 | Cámara y resolución | **Cámara web externa Full HD** con cable (la C920s presupuestada [REC 104-105; REC 119]) en trípode. Resolución: **1920 × 1080** si en el piloto del Bloque 2 la tasa efectiva mediana no cae más de un 10 % respecto de 640 × 480 con la misma cámara; si cae más, 1280 × 720 con la misma regla; si no, 640 × 480. Índice de cámara verificado en cada sesión | El anteproyecto declara una cámara web RGB Full HD [REC 72-73]. Hoy el sistema no lista ninguna cámara externa (medido). La cámara integrada está en la pantalla del portátil: con ella la persona vería el resultado (choca con PD-12). La regla del 10 % se fija ahora, sin base empírica previa, para no elegir mirando la exactitud | Cámara integrada a 640 × 480: sin montaje, pero no es el recurso declarado y obliga a que la persona mire la pantalla. Cámara de continuidad del iPhone: disponible sin compra, pero inalámbrica (posible fluctuación de c_i) y con encuadre automático que habría que apagar | Estudiante (recurso ya presupuestado) |
| 09 | Ventana y rechazo por manos | **N fijo**: `--ventana-frames N` con N = round(2,5 s × tasa efectiva mediana del piloto con la cámara y resolución elegidas) (75 con la tasa de N-26); tope 5 s (valor por defecto, 2 × 2,5); mínimo 15 fotogramas; máximo 0,5 sin manos. **Longitud nominal 2,5 s** (vigente), con la pregunta 3 al director | Criterio C1 fijado antes: N fijo gana si reduce el p90 de \|W − 2,5 s\| en 50 ms o más; lo redujo en 56,9 ms (74,8 → 17,9 ms). *Nota del verificador:* el margen es estrecho y depende del estimador; con el intervalo **mediano** entre fotogramas en lugar del medio, el p90 con N fijo sería 28,0 ms y la diferencia 46,8 ms, por debajo de 50. La recomendación se sostiene sobre todo por el mecanismo que sigue y por las ventanas de 1,2 s, no por el margen del umbral (ver anexo B). La automática toma N de la tasa medida justo antes de abrir (`translator.py`, `abrir_ventana`, `fps_medida`), que llegó a valer 15,0 fot./s con un bucle real de 30,7 (ensayo 35 de la sesión de las 16:28). Longitud fija [MET 193-196]; 2,5 s es el valor de N-11 [REG 443-446] y el que usa toda la instrumentación | Automática: W de 1,2 s en algunas ventanas, es decir, media seña. 3,0 s: +0,5 s en W y L_total (no en L_resp) y +≈ 2 min por sesión; ver hallazgo 6.1 | Estudiante el modo; **director la longitud** (pregunta 3) |
| 10 | Quién activa la ventana y reposo | **Pedal USB que emite la tecla ESPACIO, pulsado por la persona**; se comprueba antes que `realtime.py` lo lee como ESPACIO (código 32). Alternativa si no se aprueba: el operador pulsa a una señal acordada (asentir) y se declara. Reposo: **brazos relajados a los lados**, posición en que empieza y termina cada ejecución, como el corpus | La latencia total va «desde la activación de la ventana temporal por la persona usuaria» y la ventana la activa «la persona usuaria (botón o tecla)» [MET 194; MET 300-301]. El pedal deja las manos libres y no introduce el tiempo de reacción de un tercero. LSC50 inicia y termina cada captura en la misma posición de reposo [MET 285-286] | Operador pulsa: se aparta de la letra de [MET 300-301] y añade el tiempo de reacción de la persona dentro de W (más riesgo de seña cortada al final). Teclado del portátil: la mano viaja del teclado al reposo dentro de la ventana. Pedal: gasto no presupuestado | Director (definición de t0 y gasto) |
| 11 | Medio de presentación | **Tarjeta impresa** con la glosa en letra grande, sostenida junto a la cámara. **Video de referencia a demanda**: el COLOR_BODY de esa glosa de **un mismo señante del corpus, fijado antes y declarado**, reproducido en el Mac desde la copia local, con el programa de reproducción del sistema, **entre ventanas** y cerrado antes de la siguiente activación; se anota `referencia_mostrada` | Cubre los dos requisitos: glosa y, si se requiere, secuencia de referencia [MET 225-226]. Mostrar el video no lo transfiere ni lo deposita fuera del control del proyecto [ETI 19-21]. Reproducirlo con la ventana cerrada no toca las marcas de ninguna ejecución; con N fijo (PD-09) tampoco cambia N | Segunda pantalla o tableta: otro dispositivo con copia del corpus. Presentación por intérprete: tercera persona y costo; su ejecución podría servir de modelo | Estudiante |
| 12 | Retroalimentación visible | **La pantalla mira solo al operador**; la persona no ve clase, confianza ni estado. El operador comunica lo que el sistema pide (repetir) con una tarjeta | Evita que la persona adapte la seña a lo que el sistema acepta; la indicación visual de §2.2.6 [MET 191-192] sigue existiendo y la ve el operador, que la traslada | Pantalla visible: aprendizaje del sistema durante 200 presentaciones. Mostrar solo la barra: exige cambio de interfaz (4.8) | Estudiante |
| 13 | Declaración de variante | **Momento: justo después de ejecutar, antes de que el operador muestre «repetir» o «siguiente»** (la persona no conoce el resultado). **Medio: tres tarjetas** (sí / no / no estoy segura) que la persona señala. **Cifra depurada principal: conjunto estricto** (solo «no»); el amplio (no + no estoy segura) va como sensibilidad | [MET 226-228] pide declarar por ejecución si la forma difiere; declarar antes del resultado evita el sesgo de selección que [MET 229-233] quiere impedir. Medio accesible [ETI 55-57]. `exactitud_bloque1.py` ya calcula `sin_variante_estricto` y `sin_variante_amplio` | Declarar después del resultado: riesgo de declarar variante cuando falla. Solo sí/no: fuerza respuestas inciertas | Director (RE5) |
| 14 | Política ante un rechazo | **Máximo 2 intentos por presentación**: tras cualquier rechazo (confianza, manos, incompleta) o aborto por seña cortada al inicio, se repite una vez; si vuelve a fallar, se pasa a la siguiente. **Principal: `primer_intento`**; se reportan también `todos` y `ultimo_intento` | El sistema pide repetir [MET 184-186]. El primer intento es el análogo de la prueba LOSO (una ejecución por muestra) y no depende de cuántas veces se repitió. El guion ya calcula las tres políticas (`exactitud_bloque1.py`, `aplicar_politica`) si la hoja trae `presentacion` e `intento` | Sin repetir: contradice que el sistema pida repetir. Repetir sin tope: tiempo impredecible y más peso a las señas difíciles | Director (RE5) |
| 15 | Denominador de la exactitud | **Reportar las tres lecturas y nombrar como principal PD-15 A** (`con_prediccion`: top-1 y top-3 sobre aceptadas + rechazadas por confianza), con la cifra del sistema completo (`todas_rechazo_error`) y la exactitud entre aceptadas con su tasa de rechazo (`aceptadas`) al lado | La lectura A es la comparable con el LOSO, cuyo top-1 no depende del umbral [REG 1221-1222 · D-09], y separa errores de clasificación de rechazos deliberados [MET 197-198]. Además no cambia si se cambia el umbral (PD-28) | Solo la B: mezcla rechazos con errores. Solo la A: oculta el efecto del umbral, que [MET 187-189] manda reportar | Director (RE5) |
| 16 | Familiarización | **3 ejecuciones** en una corrida aparte con `--sin-registro`, con señas **fuera del vocabulario** que la persona elija, para practicar pedal, reposo y tarjetas | Las primeras ejecuciones reflejan el aprendizaje del procedimiento; con señas fuera del vocabulario no se practica ninguna glosa del bloque; sin registro no queda nada que excluir | Ninguna: las primeras presentaciones mezclan procedimiento y seña. Familiarización registrada (E5): registros que no se usan | Estudiante |
| 17 | Casos límite del Bloque 2 | **10 ejecuciones por cada caso provocable** (CL-1 ambiguas, CL-2 5 totales y 5 parciales, CL-3, CL-4, CL-5, CL-6a, CL-6b 5 de inicio adelantado y 5 de fin tardío) = 70; CL-6c se declara no ejercitado si no ocurre. **CL-5 con gestos cotidianos no lingüísticos**; se reporta como conteo de estados. **CL-6b se declara límite del prototipo** (R6), sin regla automática antes de los bloques | [MET 250-251] exige ejercitar los casos; el conteo de estados está permitido en el Bloque 2 [MET 261-262]. El autor no es usuario fluido [MET 255-256]: con señas de otros vocabularios no sabría si ejecuta la seña. Una regla automática de «cortada» exige su propia decisión y validación; otra pieza de este lote la estudia y sus cifras no están verificadas | Omitir el conteo de CL-5 aceptadas: oculta un comportamiento del vocabulario cerrado (R7). Implementar ya el descarte: cambia el prototipo antes de congelarlo, con una regla sin validar | Estudiante el número; **director CL-6b** (pregunta 7) |
| 18 | Retardo de audio | **Solo audio**, grabado con un teléfono a 48 kHz sin compresión si la aplicación lo permite; **30 ensayos** fijados antes; corrida dedicada con `--umbral 0` y `--registro-dir` propia; detección de clic e inicio del audio **por guion** (cambio 4.6) revisada a mano en los 30; archivo conservado fuera del repositorio hasta la entrega y luego borrado | Se caracteriza una sola vez con grabación externa que capte pulsación e inicio del audio [MET 326-331; MET 482-484]; no se resta y, si no se mide, «no medido» [MET 513-517]. Solo audio: sin video [ETI 50-53] y con resolución de muestra. `--umbral 0` hace que toda ventana inferida emita audio; el retardo no depende del umbral. 30 ensayos: sin dato previo de dispersión, deja 3 observaciones por encima del p90 y cuesta ≈ 4 min (3.11) | Video de alta velocidad: es video y su resolución la limitan los fotogramas. Detección manual sola: menos reproducible | Estudiante |
| 19 | Ventanas para W y captura | **B: ventanas inferidas** (filas `[inferidas]` de `latencia_resumen.py`), con las aceptadas como control de L_total = W + L_resp | Es lo que pidió N-14 [REG 566 · N-14] y lo que el resumen ya calcula; con N fijo (PD-09) W casi no cambia entre conjuntos | A: incluye abortadas y rechazadas por manos. C: exige filtrar el JSONL | Estudiante |
| 20 | Estadísticos adicionales | **Intervalos de Wilson al 95 %** para top-1 y top-3 del Bloque 1 (cifra principal y depurada), declarados como adición; latencias solo con lo que fija el anteproyecto | [MET 276-280] no fija incertidumbre; con 200 presentaciones y 4 por clase la incertidumbre es grande (R11). Wilson es cerrado y no requiere remuestreo | Sin intervalos: una proporción de 200 sin incertidumbre. Intervalos para todo: más tablas sin pregunta | Estudiante |
| 21 | No participación en LSC50 | **Declaración escrita de la persona** con la información pública de la grabación **y consulta a los autores del corpus**, solo con el consentimiento expreso de la persona para esa consulta (revela su identidad a terceros; [ETI 57-60] limita la identificación a un único registro custodiado por el director); si no responden antes de la primera sesión, rige la declaración y se anota | Se verifica y se deja constancia [MET 233-234; REC 32-33]; es la condición de «sexto señante» [MET 216-217] | Solo declaración: depende de la memoria de la persona. Esperar la respuesta: puede bloquear el bloque | Director (validez de RE5) |
| 22 | Glosa objetivo y bloque | **Hoja del operador pre-llenada** con ronda, presentación y glosa en el orden de la semilla (cambio 4.4), unida por (`sesion`, `id_ensayo`); `estado_pantalla` como control; **opcional**: campo `bloque` en el JSONL (cambio 4.1) | `exactitud_bloque1.py` ya cruza JSONL y hoja y lista todo lo que no empareja [REG 1568-1590 · N-41]; pre-llenar la glosa quita la transcripción que más riesgo tiene (R14) | Escribir la glosa en el JSONL desde el prototipo: más automático, pero toca `translator.py` y `realtime.py` antes de congelar | Estudiante |
| 23 | Carga y Google Drive | **`--registro-dir` en una carpeta local fuera de Drive** durante la sesión y copia a `registros/` al terminar; sincronización de Drive **pausada** durante la sesión; carga (`uptime`) anotada al inicio y al final | N-40 descartó la lectura desde Drive como causa de la caída de N-21, pero dejó viva la hipótesis de la sincronización con escritura [REG 1548-1553 · N-40] | No intervenir: el costo por fotograma puede quedar inflado (R8) | Estudiante |
| 24 | Salida de audio y volumen | **Parlantes integrados**, volumen fijo anotado en la hoja, sin dispositivos inalámbricos; la misma salida en la caracterización del retardo | Es la salida por defecto medida hoy; un único dispositivo que caracterizar | Parlante externo: otro dispositivo que caracterizar | Estudiante |
| 25 | Bloques y conversión de glosas | **Cada bloque con el `grammar_mapper.py` vigente al empezarlo, congelado** y con su SHA256 en la hoja; si 3.a lo cambia entre bloques, se declara | La conversión midió 0,0 ms [REG 958 · N-26] y no interviene en la exactitud del modo aislado (la clase se decide antes, en t3); la cache de voz se precalienta con los enunciados del mapeo cargado (`translator.py`, `_precalentar_voz`) | Esperar a 3.a: compromete el mes 4 [CRO 55-59]. Mezclar versiones dentro de un bloque: tramos no comparables | Estudiante |
| 26 | Vocabulario efectivamente soportado | **50**, las clases que el modelo puede emitir, y como complemento descriptivo del Bloque 1 el número de señas con al menos un acierto en el primer intento y la lista de señas con recall 0 | Se registra el tamaño del vocabulario soportado [MET 517-518; RES 58-59]; si fuera menor que 50 se documenta [ALC 23-24] | Solo el complemento como definición: depende de r y convertiría una muestra de 4 por seña en un «tamaño» | Director (RE5) |
| 27 | Custodia de registros y hojas | **Bloque 2 y retardo de audio: JSONL versionados** (D-12). **Bloque 1: JSONL fuera del repositorio público**, bajo la custodia que indique el director, con copia disociada solo de los resúmenes. Hojas del operador nunca versionadas (D-12) | D-12 decidió versionar los registros [REG 1772-1786 · D-12] antes de que existiera el Bloque 1, y consideró expresamente sus registros: «contendrán lo que el prototipo reconoció, no datos personales» [REG 1785-1786 · D-12]. Pero los registros del Bloque 1 «se conservan únicamente con el fin de reportar el resultado RE5», sin incorporarse «a ninguna publicación», y se suprimen si la persona revoca [ETI 59-61; ETI 65-69]; el repositorio se publica al cierre [ALC 161-162]. En un historial de git versionado, suprimir exige reescribirlo | Versionar el Bloque 1: publicación al cerrar y supresión costosa | Director (ética; acota D-12) |
| 28 | Umbral 0,20 | **0,20 (D-09) salvo que el director adopte, antes de la primera sesión del Bloque 1, el umbral de la validación anidada** (N-46); nunca se cambia entre sesiones de un bloque | D-09 y N-33 [REG 1201-1258]; N-46 propone el experimento [REG 1724-1751 · N-46]. La cifra principal (PD-15 A) no depende del umbral, así que la decisión cambia la tasa de rechazo y la exactitud entre aceptadas, no el top-1 principal | Mantener 0,20: 297 de 837 aceptadas eran erróneas en la prueba LOSO [REG 1251-1252 · N-33]. Cambiar a mitad de un bloque: tramos no comparables | Director |
| 29 | Voz | **Paulina (es_MX)**, 150 ppm | No hay voz `es_CO` en el sistema (medido); Paulina es la voz medida en N-34 y N-45 | Cambiar de voz: volver a medir y precalentar; no afecta al reconocimiento | Estudiante |
| 30 | Intérprete en las sesiones | **Según el perfil**: si la persona vinculada es sorda, **intérprete presente fuera de cuadro** durante el consentimiento y la explicación de cada sesión y disponible durante las rondas, sin ejecutar ninguna glosa ante la persona; si es intérprete oyente, no hace falta | Consentimiento en LSC o con intérprete si la persona es sorda [ETI 55-57]; perfil por prelación [REC 29-33] | Sin intérprete con persona sorda: consentimiento no accesible y riesgo de malentendidos en las tarjetas | Director (costo y ética) |
| 31 | Fallos de voz | **Conservar y contar (E7, D-13), sin repetir**. El operador anota «voz no sonó» si lo percibe. Si la consola avisa subida del contador en **tres registros seguidos**, pausa entre presentaciones, se revisa la salida (PD-24) y se reanuda declarándolo | D-13 fija el alcance del audio dudoso y deja PD-31 al director [REG 1789-1803 · D-13]; t5 es la orden de reproducción [MET 326-331], de modo que ni la latencia ni el reconocimiento dependen del sonido | Repetir la presentación: el JSONL no permite atribuir el fallo con certeza (R20) y cambia el número de intentos. Detener al primer error: interrumpe a la persona por algo que no afecta a la medida | Director (D-13 se lo asigna) |

---

## 3. Guion de sesión derivado de las recomendaciones

Roles: **operador** (el autor) y **persona** (la persona vinculada del Bloque 1). En el Bloque 2 el autor es a
la vez ejecutante y operador.

### 3.1 Antes de la sesión, sin la persona (≈ 30 min; no cuentan como horas de la persona)

1. Comprobar que están registrados como D-nn todos los PD de la sección 2 y, solo para el Bloque 1, que
   existen el consentimiento firmado bajo custodia del director, la respuesta del comité cumplida y la
   constancia de no participación en LSC50 [PROT 5.1, punto 11; ETI 55-61; ETI 69-75]. Sin cualquiera de
   ellos no hay sesión.
2. Pausar la sincronización de Google Drive; cerrar programas ajenos; anotar `uptime` (PD-23).
3. Anotar los SHA256 del modelo, de `realtime.py`, `translator.py`, `tts.py`, `grammar_mapper.py`,
   `representacion.py`, `latencia_resumen.py` y `exactitud_bloque1.py` (congelación [PROT 3]; N-37 dejó
   pendiente añadir el resumen a la lista).
4. Ejecutar sin cámara `python -m unittest tests.test_prototipo tests.test_tts tests.test_umbral` y anotar el
   resultado [PROT 5.1, punto 4].
5. Sala: luz artificial encendida, persianas cerradas, fondo liso detrás de la persona; marca en el piso a
   200 cm de la cámara (PD-06, PD-07).
6. Cámara externa en trípode, con cable, a la altura del pecho de la persona e inclinación ajustada para el
   encuadre rodillas–brazos extendidos; el Mac en una mesa al lado de la cámara, **pantalla hacia el
   operador**; el operador se sienta junto a la cámara, fuera del cuadro; pedal en el piso junto al pie
   dominante de la persona (PD-08, PD-10, PD-12).
7. Salida de audio: parlantes integrados, volumen fijo anotado (PD-24).
8. Preparar la hoja pre-llenada de la sesión (rondas, presentaciones y glosas en el orden de la semilla
   anotada), el mazo de tarjetas de glosa en ese orden, las tarjetas «sí / no / no estoy segura», «repetir» y
   «siguiente», y la carpeta local de registro (fuera de Drive).

### 3.2 Arranque y verificación (≈ 5 min)

1. Lanzar desde Terminal.app [REG 948-978 · N-26], con una carpeta de registro propia de la sesión:

   ```
   python src/realtime.py --camera <índice verificado> --ancho <PD-08> --alto <PD-08> \
          --ventana-frames <N del piloto; 75 con la tasa de N-26> \
          --registro-dir <carpeta local fuera de Drive>/4b_bloque1_s<k>
   ```

   Sin `--umbral` (salvo PD-28), sin `--mano-ausente`, sin `--model-path` y sin `--modo-oracion`.
2. Verificar el banner (`realtime.py`, `main`): modelo canónico «manos» con procedencia leída; umbral y su
   origen en `modelo_canonico.umbral.json`; mano ausente «mascara»; TTS `offline`; «Ventana: 75 fotogramas»;
   resolución efectiva; ruta del registro. Si una línea difiere, no se empieza.
3. Comprobar en la imagen que la cámara es la externa (con dos o más cámaras, el índice 0 puede ser otra).
4. Comprobar que el pedal abre la ventana (la barra avanza) y cerrarla con X: queda un
   `rechazada_incompleta` que se anota en la hoja como «prueba de pedal», válida = no (E3).
5. Anotar la tasa del bucle que muestra la pantalla (`dibujar_ui`) y la iluminancia.
6. Regla del operador: durante la sesión **no se pulsan M, R ni ENTER** (M cambia al modo oración;
   `realtime.py`, `main`).

### 3.3 Llegada de la persona (≈ 15 min de presencia, con el cierre de 3.8)

1. Con intérprete si corresponde (PD-30): recordar que no se graba video, que puede pausar o retirarse en
   cualquier momento y cómo se usan pedal y tarjetas [ETI 55-61].
2. Código de participante sin datos identificativos en la hoja (por ejemplo `P01`) [ETI 57-60].
3. Encuadre con la persona en la marca: en reposo y con los brazos extendidos arriba, hombros y manos dentro
   del cuadro; la pantalla debe decir «manos detectadas» en reposo, porque en reposo pueden quedar fuera las
   manos y eso se vería como rechazo por manos.
4. **Familiarización** (PD-16): cerrar el programa y lanzarlo aparte con `--sin-registro`; tres ejecuciones
   con señas fuera del vocabulario elegidas por la persona; volver a lanzar con el comando de 3.2 y repetir
   la verificación del banner.

### 3.4 Una presentación del Bloque 1 (ciclo)

| Paso | Operador | Persona |
|---|---|---|
| 1 | Muestra la tarjeta de la glosa k junto a la cámara | La lee; si lo requiere, pide la referencia (tarjeta «ver referencia») |
| 2 | Si se pidió: gira el portátil, reproduce el COLOR_BODY de esa glosa del señante fijado, lo cierra, vuelve la pantalla hacia sí; marca `referencia_mostrada = si` | Mira el video |
| 3 | Espera con la pantalla en «REPOSO» | Adopta el reposo |
| 4 | Mira la imagen de la persona en la pantalla | **Pulsa el pedal** (t0) y ejecuta la seña en su forma habitual; vuelve al reposo |
| 5 | Con la barra llena (N fotogramas), **antes de leer el resultado**, decide si hubo seña cortada (3.6) y anota `cortada` | — |
| 6 | Muestra las tres tarjetas de variante | Señala **sí / no / no estoy segura** |
| 7 | Anota `variante`; lee el `id_ensayo` en la consola y el estado en pantalla; anota `id_ensayo`, `intento`, `estado_pantalla` | — |
| 8 | Si el estado es un rechazo y es el intento 1: muestra «repetir» y se vuelve al paso 3 (intento 2 de la misma presentación). Si no: muestra «siguiente» | — |
| 9 | Si hubo audio (aceptada), espera a que termine antes de la siguiente presentación (evita `tts_ocupado_en_t5` [REG 1261-1302 · N-34]) | Puede pedir pausa |

Al final de cada ronda: pausa de 5 min (PD-05). La sesión se cierra siempre al final de una ronda.

### 3.5 Qué se marca en la hoja (una fila por ejecución)

Columnas de la plantilla vigente (`docs/plantilla_hoja_registro_bloque1.csv`): `sesion` (el valor del campo
`sesion` del JSONL, fecha y hora de arranque del programa, `translator.py`, `LSCTranslator.__init__`; hoy no se
imprime en consola, así que se copia del JSONL al terminar o se imprime con el cambio 4.1), `presentacion`, `ronda`, `intento`, `id_ensayo`,
`id_glosa_objetivo`, `glosa_objetivo` (pre-llenadas), `referencia_mostrada`, `variante`, `valida` (no solo
por E3 o E5, con el motivo en `observacion`), `estado_pantalla`, `observacion` (sin datos identificativos:
incidencias, «voz no sonó», pausa). **Columna nueva recomendada: `cortada`** (no / inicio / fin), que la
plantilla vigente no tiene (cambio 4.3); hasta que exista, se escribe en `observacion` con la forma
`cortada=fin`.

### 3.6 Incidentes

- **Intento fallido (rechazo por confianza, por manos o incompleta).** Se conserva en el JSONL y en la hoja
  como intento 1; se repite una vez (PD-14). El intento 2 es otra fila con el mismo `presentacion`. Si el
  intento 2 también se rechaza, se pasa a la siguiente presentación. Todo cuenta en `todos`.
- **Seña cortada al inicio** (la persona empezó antes de pulsar, o el pedal no respondió a tiempo): si el
  operador lo ve con la ventana abierta, **pulsa X** (`rechazada_incompleta`, motivo «abortada»), anota
  `cortada = inicio` y se repite como intento 2. Así el descarte lo hace el mecanismo que el prototipo ya
  tiene, antes de conocer el resultado.
- **Seña cortada al final** (la barra se llenó mientras la persona aún articulaba la seña, no el regreso al
  reposo): el prototipo ya infirió, porque no detecta el corte (R6). Se anota `cortada = fin` antes de leer
  el resultado; la ejecución **no se excluye**; se reporta su número y, como sensibilidad, la exactitud sin
  ellas. No se repite, salvo que además sea un rechazo en el intento 1 (regla general).
- **Audio dudoso (D-13).** Cuando la consola imprime «[AVISO TTS] … no sonaron» (`realtime.py`,
  `AvisoErroresVoz`), el operador anota en `observacion` el `id_ensayo` en que apareció. La presentación **no
  se repite** (PD-31). En el análisis, el audio dudoso va desde el ensayo **anterior** al primer registro con
  el contador en alto hasta el final de la sesión, como fija D-13 y calcula `latencia_resumen.py`
  (`audio_dudoso`); esos ensayos se conservan en latencia y exactitud y se cuentan como incidencia (E7). Si el
  contador sube en tres registros seguidos: pausa, revisión de la salida de audio y reanudación declarada.
- **Persona ajena en cuadro, congelamiento de la cámara o cierre del programa:** X si hay ventana abierta;
  `valida = no` con el motivo (E3); la presentación se repite. Si el programa se cierra, se relanza con el
  mismo comando (sesión nueva del JSONL) y se sigue desde la presentación pendiente, declarándolo.
- **Revocación del consentimiento:** se detiene la sesión y rige [ETI 67-69].

### 3.7 Bloque 2 (el autor)

Mismo montaje, comando y verificación, con `--registro-dir …/4b_bloque2_s<k>`, sin tarjetas de variante ni
familiarización. El autor sigue la hoja pre-llenada de su propio orden (4 rondas de 50, PD-02), pulsa el
pedal, ejecuta, anota `id_ensayo`, seña intentada y estado. Después, los 70 casos límite en el orden fijado
(PD-17), anotando el tipo, la condición provocada y el estado observado, sin reetiquetar [PROT 5.3]. El
piloto de PD-08 (dos resoluciones, al menos 50 ventanas cada una) se hace **antes** del bloque, en una carpeta
propia que no entra en él, y fija la resolución y el N de todas las sesiones.

### 3.8 Caracterización del retardo de audio (una sola vez)

1. Sala en silencio; ninguna persona habla ni aparece; solo el autor [ALC 75-77].
2. Misma salida y volumen que las sesiones (PD-24). Teléfono en modo avión, aplicación de grabación de voz en
   formato sin compresión si la ofrece, apoyado en la mesa entre el teclado y los parlantes, a distancias
   parecidas de ambos (la diferencia de recorrido del sonido se declara; a 343 m/s, 10 cm son ≈ 0,3 ms).
3. `python src/realtime.py <mismos argumentos de cámara y ventana> --umbral 0 --registro-dir
   <local>/4a_retardo_audio`. Se usa la **tecla ESPACIO del teclado** (el clic tiene que oírse); si en los
   bloques se usa pedal, se declara que el retardo entre la pulsación y t0 puede diferir con otro dispositivo.
4. Empezar la grabación, esperar 5 s de silencio; 30 ensayos: pulsar ESPACIO con firmeza, ejecutar cualquier
   seña con las manos en cuadro, esperar a que termine el audio y 5 s más. Anotar en la hoja 9.5 del protocolo
   el orden y el `id_ensayo` de cada uno; un ensayo sin audio (rechazo por manos, o contador de errores de voz
   en alto) no es emparejable y se excluye contando el motivo.
5. Parar la grabación; Q; copiar la carpeta. Detectar en el audio el inicio del clic y el inicio de la voz
   con el guion de 4.6 (umbral de amplitud sobre el ruido de fondo) y revisar los 30 sobre la forma de onda.
   d = I_ext − L_total por ensayo; n, media, DE, mediana y p90 de d [PROT 8.2], reportado como sesgo
   constante que no se resta y que incluye el retardo entre la pulsación física y t0 (hasta un periodo del
   bucle, ≈ 33 ms a 30,4 fot./s [REG 955 · N-26]).

### 3.9 Después de cada sesión (≈ 15 min, sin la persona)

Copiar el bloque «Contadores de la sesion» y, si aparece, la línea «[AVISO TTS] Voz de la sesión»; comprobar
que ejecuciones de la hoja = registros `ensayo` del JSONL; correr `python src/latencia_resumen.py <carpeta
de la sesión> --json <archivo>` y `python src/exactitud_bloque1.py --registros <carpeta> --hoja <hoja>
--salida <carpeta de análisis>` (este último solo en el Bloque 1); comprobar que no se creó ningún archivo de
imagen o video; copiar los registros a su destino según PD-27; nota N-nn en el registro [PROT 5.4].

### 3.10 Condiciones que se anotan una vez por sesión

Cámara (modelo e índice), resolución solicitada y efectiva, N fijo, tasa del bucle en pantalla, iluminancia
al inicio y al final, distancia, encuadre, salida y volumen, `uptime`, SHA256 de 3.1, semilla del orden,
versión de `grammar_mapper.py`, pausas, contadores y aviso de voz al cerrar [PROT 9.2].

### 3.11 Duración estimada

**Componentes del ciclo, fijados por el procedimiento antes de calcular** (criterio C3, anexo B): tarjeta de
glosa 4 s; reposo 2 s; W 2,5 s; L_resp ≈ 0,034 s (estimado, 0.2); audio 0,687 s de media, solo si la ejecución
es aceptada (medido, C2); declaración de variante 3 s; anotación 5 s. Un intento 2 no repite la tarjeta de
glosa.

| Ciclo | Duración |
|---|---|
| Primera ejecución de una presentación, aceptada | 17,22 s |
| Primera ejecución, rechazada (sin audio) | 16,53 s |
| Intento 2, aceptado | 13,22 s |

Supuestos de la sesión del Bloque 1: dos rondas por sesión; 15 min de llegada, recordatorio, encuadre y
verificación con la persona; 3 ejecuciones de familiarización; 5 min de pausa entre rondas; 5 min de cierre;
p_rep = proporción de presentaciones con intento 2 (a lo sumo una por presentación, supuesta aceptada).

| Escenario | p_rep | Referencias mostradas | Ejecuciones (4 rondas) | Señado por ronda | **Presencia por sesión (2 rondas)** | Total del bloque (2 sesiones) |
|---|---|---|---|---|---|---|
| Central: rechazo LOSO con 0,20 [REG 1244 · N-33] | 0,16 | 0 | 232 | 16,0 min | **57,9 min** | 115,8 min |
| Central con referencia en todas las glosas de una ronda (10 s cada una, supuesto) | 0,16 | 50 | 232 | 18,1 min | 62,1 min | 124,1 min |
| Intermedio | 0,35 | 0 | 270 | 18,0 min | 61,9 min | 123,7 min |
| Pesimista: rechazo del autor con 0,5 [REG 963-964 · N-26], no comparable | 0,63 | 50 | 326 | 23,0 min | 71,9 min | 143,8 min |
| Una sola sesión de 4 rondas (central) | 0,16 | 0 | 232 | 16,0 min | 99,9 min | 99,9 min |

Otros bloques: **Bloque 2**, ciclo de 10,22 s (lectura de la lista 2 s y anotación 3 s en lugar de tarjeta,
variante y anotación larga): 200 ejecuciones normales en 34,1 min más 70 casos límite a 12 s en 14,0 min.
**Retardo de audio**: ciclo de 8,22 s (con 3 s de silencio entre ensayos en el cálculo; el guion pide 5 s):
30 ensayos en ≈ 4 min de grabación. A esto se suma, sin la persona, ≈ 30 min de preparación y ≈ 15 min de
cierre por sesión (3.1 y 3.9).

**Límites de la estimación.** Los componentes del ciclo son supuestos del procedimiento, no mediciones: el
único dato real de ritmo es el del autor sin hoja ni tarjetas (mediana de 4,54 s entre activaciones), que
solo indica que el ciclo mecánico cabe holgado en 17 s. El piloto del Bloque 2, con la hoja y las tarjetas
completas, sustituye esta estimación antes de cotizar las horas de la persona. p_rep con una persona fluida
no se conoce: el 16 % procede de la prueba offline LOSO con cinco señantes.

---

## 4. Cambios de código que exigirían las recomendaciones (sin implementar)

Ninguno toca el modelo, el umbral, las particiones ni las marcas t0..t5. Todos se harían antes de congelar
(3.1, punto 3), con pruebas y revisión independiente, y quedarían en el registro.

| # | Archivo y función | Cambio | Lo exige | ¿Imprescindible? |
|---|---|---|---|---|
| 4.1 | `src/realtime.py`, `main` (argumento nuevo) y `src/translator.py`, `LSCTranslator.__init__` y `_registro_ensayo` | `--bloque {1,2,retardo,piloto}` que se escriba como clave `bloque` en cada registro `ensayo` y `turno`, sin cambiar las claves existentes; imprimir en el banner el valor de `sesion` para que el operador lo copie en la hoja | PD-22 (separación E1/E6 sin depender solo de la carpeta) | No; recomendable |
| 4.2 | `src/exactitud_bloque1.py`, `audio_dudoso`, `PENDIENTES` e `informe_md` | Alinear con D-13: la `convencion_b` actual coincide con la definición de D-13 (desde el anterior al primer aumento hasta el final); renombrarla o etiquetarla como D-13 y retirar de `PENDIENTES` la frase «no está decidido qué convención marca los ensayos con audio dudoso» | D-13 [REG 1789-1803] | Sí (el informe hoy contradice el registro) |
| 4.3 | `src/exactitud_bloque1.py`, `leer_hoja`, `bloque_conjunto`, `informe_md`; `docs/plantilla_hoja_registro_bloque1.csv` | Columna opcional `cortada` (no / inicio / fin); conteo por tipo y exactitud sin las `cortada = fin` como sensibilidad; leyenda de la plantilla | PD-14, 3.6 | Sí, si se adopta 3.6 |
| 4.4 | Guion nuevo, por ejemplo `src/orden_bloque1.py`, con prueba nueva | A partir de una semilla, genera el orden aleatorio por ronda y escribe la hoja pre-llenada (`ronda`, `presentacion`, `id_glosa_objetivo`, `glosa_objetivo`) en el formato de la plantilla | PD-03, PD-22 | Sí (quita la transcripción de la glosa) |
| 4.5 | `src/exactitud_bloque1.py`, `metricas` e `informe_md` | Intervalos de Wilson al 95 % para top-1 y top-3; marcar en el informe cuál es la cifra principal una vez decidida (`primer_intento`, `con_prediccion`, `sin_variante_estricto`) sin dejar de calcular las demás; complemento de PD-26 (número de señas con al menos un acierto y lista de señas con recall 0) sobre esa cifra principal, porque `por_sena` hoy se calcula sobre todos los intentos | PD-13, PD-14, PD-15, PD-20 | Sí, tras la decisión del director |
| 4.6 | Guion nuevo, por ejemplo `src/retardo_audio.py`, con prueba nueva | Lee la grabación (WAV) y la hoja 9.5; detecta inicio del clic e inicio de la voz por umbral de amplitud sobre el ruido; empareja por orden con el JSONL; calcula d y sus estadísticos; permite corregir a mano un instante | PD-18 | Sí, para la caracterización |
| 4.7 | `src/realtime.py`, ayuda de `--ventana-seg`; `src/translator.py`, docstring de `LSCTranslator.__init__`; README, sección de la ventana | La justificación de 2,5 s cita duraciones de los CSV publicados (media 2,57 s; 47 % ≤ 2,5 s); los clips con que se entrenó el modelo vigente, del video COLOR_BODY, tienen mediana 2,79 s y 29,2 % ≤ 2,5 s (hallazgo 6.1). Solo texto | Exactitud de la documentación | Sí (documental) |
| 4.8 | `src/realtime.py`, `dibujar_ui` | Solo si el director prefiere que la persona vea la pantalla: opción que oculte clase, confianza y top-3 y deje la barra y la indicación de manos | PD-12 C | No, con PD-12 B |
| 4.9 | `src/realtime.py`, `main` | Opcional: ignorar M, R y ENTER en modo evaluación (`--bloque` presente), para que una tecla equivocada no cambie al modo oración | 3.2, punto 6 | No |
| 4.10 | `docs/PROTOCOLO_EXPERIMENTAL_4a.md` | Incorporar las decisiones adoptadas, la columna `cortada`, `latencia_resumen.py` y `exactitud_bloque1.py` en la lista de congelación, y E7 con PD-31 resuelto | Todas | Sí (documental) |

`--ventana-frames`, `--sin-registro`, `--umbral 0`, `--registro-dir`, `--ancho` y `--alto` ya existen
(`realtime.py --help`), de modo que PD-09, PD-16, PD-18 y PD-23 no exigen código. El pedal no exige código si
emite la tecla ESPACIO (código 32 en `cv2.waitKey`); se comprueba en 3.2.

---

## 5. Preguntas para el director, con la opción recomendada

1. **Horas con la persona e intérprete (PD-01, PD-04, PD-30).** ¿Se cotizan dos sesiones de ≈ 1 h de
   presencia para el Bloque 1 (r = 4, 200 presentaciones) y, si la persona es sorda, intérprete durante el
   consentimiento y las explicaciones? *Recomendado: sí. Con una sola sesión de ≈ 1 h, r = 2 declarado.*
2. **Paquete de definición de RE5 (PD-13, PD-14, PD-15, PD-26).** Variante declarada antes del resultado con
   tres tarjetas, depurada estricta como principal; máximo dos intentos y primer intento como principal;
   cifra principal sobre ejecuciones con predicción, con la del sistema completo y la de aceptadas al lado;
   vocabulario soportado = 50 con el complemento descriptivo. *Recomendado: aprobar el paquete.*
3. **Longitud de la ventana (PD-09).** El valor vigente, 2,5 s, se justificó con los CSV publicados; los
   clips con que se entrenó el modelo vigente tienen mediana 2,79 s y solo el 29,2 % dura ≤ 2,5 s (68,0 %
   ≤ 3,0 s). ¿Se mantiene 2,5 s o se pasa a 3,0 s antes de congelar? *Recomendado: mantener 2,5 s, que es el
   valor con el que se ha medido todo, **salvo** que antes de los bloques se haga una prueba offline, con
   criterio fijado antes, del acierto de los modelos LOSO sobre clips recortados a 2,5 y a 3,0 s; esa
   prueba es el dato que falta para decidir.*
4. **Activación (PD-10).** ¿Se compra un pedal USB que emita ESPACIO para que la persona active la ventana?
   *Recomendado: sí (literal respecto de [MET 194; MET 300-301]); si no, el operador pulsa y se declara.*
5. **Umbral (PD-28).** ¿Se adopta, antes del Bloque 1, el umbral de la validación anidada de N-46? *Recomendado:
   decidirlo antes de la primera sesión del Bloque 1 y no cambiarlo después; la cifra principal (PD-15 A) no
   depende del umbral, así que esperar el experimento no bloquea el Bloque 2.*
6. **Registros del Bloque 1 (PD-27).** D-12 versiona los registros; §1.5 pide conservar los del Bloque 1 solo
   para RE5, fuera de toda publicación y suprimibles ante revocación. ¿Se excluyen del repositorio?
   *Recomendado: sí, bajo la custodia que indique el director; Bloque 2 y retardo de audio siguen D-12.*
7. **Secuencias cortadas (PD-17, CL-6b, R6).** §2.2.6 dice que se descartan; el prototipo no las detecta.
   *Recomendado: declararlo como límite del prototipo, abortar con X las cortadas al inicio y marcar a mano
   las cortadas al final (3.6), sin regla automática antes de los bloques.*
8. **Aprobación en bloque (PD-21, PD-31).** Verificación de no participación por declaración más consulta a
   los autores del corpus; fallos de voz conservados y contados sin repetir, con pausa ante tres subidas
   seguidas del contador. *Recomendado: aprobar ambos.*

Los otros 17 parámetros (PD-02, 03, 05, 06, 07, 08, 11, 12, 16, 18, 19, 20, 22, 23, 24, 25 y 29), más el modo
de la ventana de PD-09 y el número de casos límite de PD-17, los fija el estudiante con los valores de la
sección 2 y se declaran en el D-nn.

---

## 6. Hallazgos de esta revisión

### 6.1 La justificación de la ventana de 2,5 s no corresponde al corpus del modelo vigente

N-11 justificó 2,5 s con «clips de media 2,57 s, mediana 2,58, p90 3,29, máx 5,58; 47 % ≤ 2,5 s, 79 % ≤
3,0 s» [REG 443-446 · N-11], y la ayuda de `--ventana-seg` repite esas cifras. Son las filas de los CSV
publicados (`caracterizacion.md`, sección 5: CSV body, media 61,765 filas), que tienen menos filas que
fotogramas el video (N-02). El modelo vigente se entrenó con la extracción propia sobre los videos COLOR_BODY
(`processed_dir` de `modelo_canonico.procedencia.json`), cuyos clips tienen media 67,328 fotogramas, mediana
2,792 s, p90 3,375 s, **29,2 % ≤ 2,5 s y 68,0 % ≤ 3,0 s** (medido hoy, `lengths.npy`). Que un clip dure más de
2,5 s no implica que la seña quede cortada (los clips empiezan y terminan en reposo [MET 285-286]), pero la
cifra que sostenía la elección ya no es la del corpus de entrenamiento. Se eleva como pregunta 3 y como
cambio 4.7.

### 6.2 D-12 alcanza al Bloque 1 y choca con §1.5

D-12 decidió versionar `registros/*.jsonl` [REG 1772-1786 · D-12]; no excluye el Bloque 1, sino que lo
considera expresamente y concluye que sus registros no contienen datos personales, y declara que la hoja del
operador no se versiona [REG 1785-1786 · D-12]. §1.5 limita los registros derivados del Bloque 1 al reporte de RE5, sin
publicación y con supresión si la persona revoca [ETI 59-61; ETI 65-69], y el repositorio se publica al cierre
[ALC 161-162]. Se eleva como pregunta 6.

### 6.3 `exactitud_bloque1.py` quedó atrás de D-13

El guion (N-41, del 16-sep) ofrece dos convenciones de audio dudoso «ninguna elegida aquí» y lista E7 como
pendiente; D-13 fijó después la definición, que coincide con su `convencion_b`. Cambio 4.2.

---

## Anexo A. Comandos, salidas y comprobación de citas

Todo con `python` desde la raíz del repositorio; scripts y
salidas en la carpeta temporal `…/carpeta_temporal/wf_2026-09-22/pieza2/run2/`.

| Qué | Comando o guion | Salida |
|---|---|---|
| Copia de los JSONL | `cp registros/latencia_2026-09-1{4,5}.jsonl <run2>/`; `shasum -a 256` | SHA256 iguales (931eb41e… y ab70e6d8…) |
| C1 y referencia de C3 | `c1_c3_registros.py`; detalle de las 16 ventanas con un guion en línea (heredoc) | `c1_c3_registros.log`, `c1_detalle.log` |
| C2, audio de las 50 locuciones | `c2_audio.py` (say -r 150 -v Paulina -o, afinfo, borrado) | `c2_audio.log`, `c2_audio.json`; 0 archivos de audio restantes |
| C3, duración | `c3_duracion.py` | `c3_duracion.log` |
| C5, duración de los clips | lectura de `data/processed_holistic_mascara_rostro/lengths.npy` | `c5_duracion_clips.log` |
| Voces, cámaras, audio | `say -v '?'`; `system_profiler SPCameraDataType SPAudioDataType` | `sistema_camaras_audio.log` |
| Opciones del prototipo y de los guiones | `realtime.py --help`, `latencia_resumen.py --help`, `exactitud_bloque1.py --help` | `realtime_help.txt`, `latencia_help.txt`, `exactitud_help.txt` |
| Citas al anteproyecto | `grep -n` de una frase de cada pasaje citado en `09_metodologia.tex`, `07_etica.tex`, `06_alcances.tex`, `10_resultados.tex`, `12_recursos.tex`, `11_cronograma.tex`, más lectura con `cat -n` de los rangos | `grep_citas_anteproyecto.log`; `verificacion_citas_doc.log` (todas las citas de este documento, extraídas y con sus líneas impresas) |

## Anexo B. Criterios fijados antes de calcular

Archivo `…/pieza2/run2/00_CRITERIO_ANTES_DE_CALCULAR.txt`, escrito el 22-sep-2026 a las 16:33:18, antes de
correr ningún cálculo de esta revisión (C5 añadido después de C1-C4 y antes de calcularlo). Resumen:

- **C1** (PD-09): N fijo se recomienda si reduce el p90 de |W − 2,5 s| en 50 ms o más frente a la ventana
  automática; si no, se deja la automática. Resultado: 56,9 ms, se recomienda N fijo.
- **C2**: duración del audio de los 50 enunciados reales, sin reproducir; solo descriptivo.
- **C3**: fórmula del ciclo y de la sesión con componentes fijados por el procedimiento; el ritmo del autor
  solo como verosimilitud; p_rep 0,16 central y 0,35 y 0,63 de sensibilidad.
- **C4** (PD-01): r = repeticiones por seña y señante de LSC50 si la caracterización las confirma (4).
- **C5**: proporción de clips COLOR_BODY ≤ 2,5 y ≤ 3,0 s, solo descriptiva; no cambia la recomendación de
  2,5 s y, si contradice N-11, se eleva al director.

**Nota del verificador independiente (22-sep-2026).** (1) Los criterios de esta corrida son anteriores a sus
propios cálculos (mtimes: criterio 16:33:18; primer log 16:33:37). Pero una corrida anterior de esta misma
pieza (16:14-16:28) ya había visto la distribución de W de los mismos JSONL (16 de 71 ventanas con N ≠ 75, W
mínima 1,204 s) y ya recomendaba N fijo y r = 4 sin un criterio escrito para esas dos elecciones. C1 y C4, por
tanto, se escribieron conociendo la dirección del resultado; no son un criterio ciego. C1 no se ajustó a
posteriori de forma visible (el margen de 6,9 ms es estrecho y cambia de signo con el intervalo mediano), así que
la recomendación de N fijo debe leerse como apoyada en el mecanismo (`fps_medida_antes` desfasada tras la
pausa de síntesis) y en las ventanas de 1,2 s, no en haber superado un umbral fijado a ciegas. (2) C5 decía
leer `caracterizacion.json`; el cálculo usó `data/processed_holistic_mascara_rostro/lengths.npy`. Las dos
fuentes coinciden (mediana 2,79167 s en `caracterizacion.md`, sección 5, frente a 67 / 24 = 2,792 s).
