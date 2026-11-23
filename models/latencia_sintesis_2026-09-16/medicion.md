# Bajar el coste de la síntesis de voz del prototipo (pendiente de N-34)

**Fecha:** 2026-09-16 · **Equipo:** macOS 26.6.2 (arm64), Python 3.12.13 · **Voz:** Paulina
(`com.apple.voice.compact.es-MX.Paulina`), 150 palabras/min, modo `offline` con `say`.
**Sin reproducir audio:** en todas las mediciones la orden de reproducción se emite, pero el hilo que
reproduciría no llama a `afplay`. Ninguna cifra de este informe viene de memoria ni de otro registro:
todas se midieron hoy con los guiones que se listan al final.

## 1. Qué se mide y qué NO cambia

Se mide la **etapa «síntesis» del prototipo, `t5 - t4`**, tal como la ejecuta
`src/translator.py::_emitir_audio`: `prepare(texto)` + `esperar_preparado(tope)` +
`play_prepared_async()`. La reproducción queda fuera, igual que en N-34.

**`t5` sigue significando lo mismo** (§2.2.7: «la devolución del audio por el motor de síntesis y la
emisión de la orden de reproducción»). Con la cache, el audio está devuelto y disponible cuando se
emite la orden; lo que cambia es *cuándo* se sintetizó. Eso queda declarado en el registro (§6).

**Frases: las reales del prototipo.** Las 50 señas de `src/grammar_mapper.py` pasadas por
`GrammarMapper.translate([glosa])` —que es lo que el prototipo sintetiza en modo seña: «Gracias»,
«El hermano trabaja», «¿Dónde?»…— más 12 enunciados del modo oración («Hola, ¿cómo estás?», «Yo como
mucho», «¿Dónde vives?»…). 62 frases, 3 pases, 186 medidas por vía.

## 2. El criterio, fijado ANTES de medir

Escrito el **2026-09-16 a las 11:42:12**, antes de la primera medición de este frente
(`scratch_tesis/lote_4e_2026-09-16/latencia_sintesis/CRITERIO_ANTES_DE_MEDIR.txt`).
Se adopta una vía solo si cumple las seis:

| | Condición |
|---|---|
| C1 | La mediana de `t5 - t4` baja al menos un **50 %** y el **p90 de la vía queda por debajo de la mediana actual** (no basta mejorar el caso típico: no puede empeorar la cola). |
| C2 | No añade ninguna dependencia que no esté ya en `requirements.txt`. |
| C3 | Mismo audio: misma voz y mismo ritmo, comprobado por **sha256** del archivo sintetizado. |
| C4 | El JSONL puede declarar cuándo la síntesis no se hizo en ese instante. |
| C5 | Coste acotado: tope de entradas y de bytes, y todo desaparece con `cleanup()` (N-36). |
| C6 | No toca las marcas `t0..t5` ni los campos existentes del JSONL, ni reentrena, ni instala nada. |

## 3. El antes (situación de N-34), medido hoy

| magnitud | mediana | p25–p75 | p90 | mín–máx |
|---|---|---|---|---|
| `t5 - t4`, 186 medidas | **458,3 ms** | 456,0–460,4 | 462,6 | 443,1–953,3 |
| solo las 50 glosas (150) | 458,3 ms | 455,7–460,8 | 463,2 | 443,1–953,3 |
| solo modo oración (36) | 458,2 ms | 456,6–459,8 | 462,4 | 452,9–463,3 |

Coincide con N-34 (445–463 ms, mediana 459) y con su verificación independiente (442 ms, 428–490).
Sintetizar las 50 glosas de una vez con `say`: **23,0 s** y 1 720 222 bytes de audio.

**Determinismo (lo que hace posible la cache).** `say -o` con el mismo texto, la misma voz y el mismo
ritmo da **el mismo archivo byte a byte** (sha256 idéntico en las 4 frases comprobadas). Cambia si
cambia el ritmo (150 → 200 ppm) y si cambia la voz (Paulina → Mónica). Conclusión: se puede reutilizar
el audio del mismo texto, **pero la voz y el ritmo tienen que formar parte de la clave**.

## 4. Las tres vías

### (c) pyobjc — ¿está instalado?

**Sí.** `pyobjc 12.2.2` (incluidos `pyobjc-core` y los frameworks `AppKit` y `Foundation`) está en el
intérprete del proyecto, y `requirements.txt` lo trae explícitamente
(`pyobjc==12.2.2; sys_platform == "darwin"`) porque es la dependencia de pyttsx3 en macOS. **No se
instaló nada**: la vía (b) se apoya en lo que ya estaba.

### (b) Sintetizador vivo entre frases (NSSpeechSynthesizer)

| forma de usarlo | n | mediana | p25–p75 | p90 | máx |
|---|---|---|---|---|---|
| objeto vivo, hilo principal | 186 | **15,2 ms** | 14,5–17,4 | 18,8 | 519,6 |
| objeto vivo, un hilo nuevo por frase (como hace `prepare()`) | 186 | **14,2 ms** | 13,5–16,1 | 17,7 | 515,3 |
| objeto nuevo en cada frase | 20 | 120,7 ms | 119,5–122,2 | 122,5 | 124,3 |
| `say -o` (lo de hoy) | 186 | 458,3 ms | 456,0–460,4 | 462,6 | 953,3 |

- **El audio es el mismo que el de `say`**: sha256 idéntico en las 4 frases comparadas una a una, en
  las **186** sintetizadas en serie desde hilos nuevos (0 distintas) y también a 200 ppm.
- **Hace falta un candado.** Dos hilos usando el mismo objeto a la vez producen archivos truncados o
  mezclados (4 096 y 37 376 bytes, ninguno igual al de `say`). Con candado, cuatro peticiones
  simultáneas salen correctas y en fila (14,5 / 28,9 / 44,4 / 59,0 ms).
- Cumple C1 (−96,9 % en mediana; p90 17,7 ms ≪ 458 ms), C2, C3 y C6.

### (a) Cache de audio por texto

| | n | mediana | p90 | máx |
|---|---|---|---|---|
| acierto de cache (medido antes de implementar) | 186 | 0,137 ms | 0,174 | 3,04 |
| acierto de cache (ya implementado) | 150 | **0,144 ms** | 0,164 | 0,31 |

Cumple las seis condiciones. Un acierto cuesta ~0,14 ms: copiar el archivo y publicarlo.

## 5. El después (código ya cambiado)

| situación | n | mediana | p25–p75 | p90 | máx |
|---|---|---|---|---|---|
| control: cache y sintetizador persistente **apagados** | 186 | 457,7 ms | 455,4–460,0 | 462,3 | 947,4 |
| solo sintetizador persistente | 186 | 14,2 ms | 13,5–16,2 | 17,3 | 189,6 |
| **modo seña con la cache caliente** | 150 | **0,144 ms** | 0,135–0,151 | 0,164 | 0,31 |
| modo oración, frase nueva (fallo de cache + persistente) | 12 | 16,7 ms | 16,2–17,6 | 17,7 | 19,4 |
| modo oración, frase repetida | 24 | 0,161 ms | 0,150–0,179 | 0,18 | 0,21 |

- **Modo seña tras calentar: 458,3 → 0,144 ms (−99,97 %)**, con 150/150 aciertos de cache.
- **Modo oración, frase que nadie había dicho: 458,2 → 16,7 ms (−96,4 %)**, con 0/12 aciertos (es
  decir: la mejora ahí es del sintetizador persistente, no de la cache).
- El control reproduce la situación de N-34 (457,7 ms): apagando las dos cosas se vuelve exactamente
  a lo de ayer.
- **Calentar las 50 glosas cuesta ahora 795,7 ms** (50/50, 1 720 222 bytes), frente a los 23,0 s que
  costaba con `say`: el sintetizador persistente abarata también el precalentado.
- **El audio que se oye no cambia**: el de la cache tiene el mismo sha256 que `say -o` (3/3 frases) y
  el del sintetizador persistente también, a 150 y a 200 ppm.

**Qué significa para la latencia total.** N-34 declaró que `L_resp` y `L_total` subían ≈0,46 s al
meter la síntesis dentro de `t5`. Ese sumando vuelve a ser ≈0,0001 s en el modo seña y ≈0,017 s en una
frase nueva del modo oración, **sin cambiar qué significa `t5`**: el audio está hecho cuando se emite
la orden. Antes de N-34 la etapa valía ≈0 ms porque la síntesis **no se había hecho**; ahora vale ≈0 ms
porque **ya estaba hecha**. Son dos cosas distintas y por eso el registro tiene que distinguirlas.

## 6. Lo que declara el registro

Campo nuevo en el JSONL: **`tts_sintesis_desde_cache`**.

- `true`: el audio ya estaba hecho (la etapa «síntesis» no midió una síntesis).
- `false`: se sintetizó en ese instante.
- **No aparece** si no hubo audio o si el motor no informa de su cache (dobles de prueba, motores
  inyectados): no se afirma nada de un motor que no lo dice.
- Va en el registro de ensayo (después de los contadores de voz) y en el de turno (antes de ellos).
- Se lee **después** de sellar `t5` (o `t_c` en el turno): las 48 claves del ensayo y las 14 del turno
  siguen iguales, en el mismo orden y con los mismos valores con reloj falso.

Sin este campo, una sesión con la cache caliente mostraría «síntesis ≈ 0 ms» y parecería contradecir
los ~0,46 s de N-34 sin ningún aviso.

## 7. Límites y ciclo de vida de la cache (cierre de temporales de N-36)

| | |
|---|---|
| Tope de entradas | **64** (las 50 glosas con margen) |
| Tope de bytes | **8 MiB** (las 62 frases medidas ocupan 2,4 MiB: unas tres veces lo previsto) |
| Desalojo | por antigüedad de uso (LRU) al pasarse cualquiera de los dos topes; el audio desalojado se borra |
| Audio mayor que el tope | no entra (y no vacía la cache por su culpa) |
| Dónde vive | subcarpeta `cache` de la carpeta temporal del módulo (`lsc_tts_*/cache`) |
| Cuándo muere | `cleanup()` borra la carpeta temporal entera: nada sobrevive al proceso ni se escribe fuera de `TMPDIR` |
| Invalidación | la clave es el texto y la cache lleva firma (modo, voz, ritmo, ejecutor, extensión); si la firma cambia, se vacía entera |
| Nunca entrega el original | `prepare()` entrega una **copia**: el hilo que reproduce sigue siendo el dueño del archivo que borra al terminar |

**Consecuencia que se declara.** Si un proceso muere **sin** llamar a `cleanup()`, su carpeta
`lsc_tts_*` queda en `TMPDIR` —como las ~895 que contó N-36— pero ahora puede contener hasta el tope
de la cache (8 MiB) en vez de estar casi vacía. Medido tras todo el trabajo de hoy: de las 120
carpetas `lsc_tts_*` creadas en esta sesión de trabajo (pruebas y mediciones), 50 conservaban algún
archivo y las 120 juntas ocupaban **1,9 MB**; la mayor tenía 104 KB. Una cache llena con las 50 glosas
ocupa 1,72 MB. El tope sigue siendo por proceso y no hay ningún crecimiento sin límite dentro de una
sesión, pero conviene barrer `TMPDIR` cuando no corra ningún prototipo ni prueba (lo mismo que ya
pedía N-36).

## 8. Decisión

**Se adoptan las dos vías, (a) y (b)**, porque son complementarias y las dos cumplen C1–C6:

- la **cache** resuelve el modo seña, que tiene vocabulario cerrado (50 frases): 100 % de aciertos
  tras calentar, 0,14 ms;
- el **sintetizador persistente** resuelve lo que la cache no puede tener de antemano —las
  combinaciones del modo oración—, que pasan de 458 ms a 17 ms, y además abarata el precalentado
  (0,8 s en vez de 23 s).

Todo es reversible sin tocar código: `TTSModule(cache=False)`,
`TTSModule(sintetizador_persistente=False)` y `LSCTranslator(tts_precalentar=False)` devuelven el
comportamiento de N-34 (medido como control: 457,7 ms).

**Riesgos que se asumen y se declaran.**
1. Si el sintetizador persistente falla una vez (pyobjc ausente, voz que no carga, o más de 2 s sin
   terminar) se apaga para el resto de la sesión, avisa una vez y se vuelve a `say`: el audio sale
   igual, más lento.
2. La etapa «síntesis» deja de medir una síntesis cuando hay acierto de cache. El JSONL lo declara,
   pero `src/latencia_resumen.py` todavía no separa por ese campo (pendiente).
3. El máximo del sintetizador persistente fue 0,19–0,52 s en 186 medidas (un caso aislado), aún por
   debajo de la mediana de `say`.

## 9. Pruebas

**Suite completa: 472 pruebas OK** (las 436 de referencia más 36 nuevas en `tests/test_tts.py`), sin
audio y sin cámara. Cubren: acierto y fallo de cache, copia propia para quien reproduce, cambio de voz
y de ritmo, tope de entradas con desalojo LRU, tope de bytes, audio mayor que el tope, cache apagada,
cierre, audio de la cache que desaparece; precalentado (llenado, no repetir, esperar a la preparación
en curso, no arrancar dos veces, fallo de una frase, corte por `cleanup()`); sintetizador persistente
con dobles de pyobjc (uso, reutilización del objeto, una síntesis a la vez, caída a `say` en cinco
formas de fallo, apagado por parámetro); y el registro (el campo llega al ensayo y al turno, no
aparece si el motor no informa, no desordena las claves, no mueve las marcas, y el precalentado usa
los 50 enunciados y no las glosas).

**Mutantes: 12 de 12 detectados** —cache que ignora la firma, cache sin desalojo, cache que entrega su
propio archivo, acierto no declarado, precalentado que adelanta a una preparación en curso,
precalentado que arranca con una segunda preparación todavía viva, sintetizador persistente sin
candado, fallo que se reintenta, precalentado que repite, clave escrita siempre, prototipo que no
precalienta y precalentado con la glosa en vez del enunciado—.

## 10. Pendientes

1. `src/latencia_resumen.py` no separa todavía los ensayos con `tts_sintesis_desde_cache` `true`:
   mientras no lo haga, la etapa «síntesis» promediada mezcla aciertos de cache con síntesis reales.
2. No hay ninguna sesión real con cámara y audio desde este cambio: falta confirmar con audio que lo
   que suena es lo esperado (el sha256 dice que el archivo es el mismo, pero eso no sustituye a oírlo).
3. El precalentado ocupa el sintetizador ~0,8 s al arrancar; con cámara hay que ver si conviene
   lanzarlo antes de abrir la primera ventana.
4. Sigue sin medirse el efecto sobre los fotogramas acumulados tras la pausa de la síntesis (pendiente
   de N-34, necesita cámara). Ahora esa pausa es de ~0 ms en el modo seña, así que el problema
   probablemente desaparece, pero **no está comprobado**.
5. Un proceso que muere sin `cleanup()` deja ahora hasta 8 MiB de audio en su carpeta de `TMPDIR` (§7).
   El módulo no registra ningún cierre automático al terminar el proceso: si se quiere, es una decisión
   aparte.

## 11. Evidencia

Guiones y salidas en `scratch_tesis/lote_4e_2026-09-16/latencia_sintesis/`:
`CRITERIO_ANTES_DE_MEDIR.txt`, `frases.py`, `medir_antes.py` → `medicion_antes.json`,
`probe_nsspeech.py` → `medicion_nsspeech.json`, `probe_ns_hilos.py` → `medicion_ns_hilos.json`,
`medir_despues.py` → `medicion_despues.json`, `mutantes.py` → `mutantes.json`.
Todas las cifras de este informe están en `medicion.json`, en esta misma carpeta.
