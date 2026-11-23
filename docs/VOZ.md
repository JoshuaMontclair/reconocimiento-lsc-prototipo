# Voz del prototipo: audios pregrabados, respaldo XTTS-v2 y voz local

Fecha: 23 de noviembre de 2026. Equipo: MacBook Pro M5 Max, 18 núcleos, 48 GB, macOS 27.0.1. Commit base 4734147.

**Decisión del autor (23-nov-2026).**

1. **Voz principal: audios pregrabados.** En modo `--conversor inventario` el prototipo solo puede decir 142 enunciados
   fijos. Los 142 se generaron con ElevenLabs (voz «Robert - Colombian, Paisa») y se reproducen tal cual.
2. **Voz de respaldo: XTTS-v2**, hablante «Gilberto Mathias», idioma `es`, en vivo. Se usa para cualquier frase que
   no esté pregrabada.
3. **Último respaldo: la voz local de macOS «Mónica».** Hasta ahora la voz offline era Paulina.

| Archivo | Qué contiene |
|---|---|
| `models/voz_pregrabada/001.wav` … `142.wav` | los 142 enunciados (WAV PCM 16 bits, mono, 44,1 kHz) |
| `models/voz_pregrabada/indice.json` | índice: texto exacto, sha256 del texto y del audio, duración, origen, corte, verificación y revisión a oído |
| `src/voz_pregrabada.py` | enumeración de los enunciados desde el conversor, carga y validación del índice, búsqueda exacta, cobertura |
| `src/voz_xtts_servicio.py` | servicio XTTS-v2 en un proceso aparte (servidor) y su cliente |
| `src/tts.py` | modo `pregrabada` y voz local Mónica |
| `src/realtime.py`, `src/translator.py` | predeterminados del prototipo; `tts_fuente` y `tts_tiempo_sintesis_ns` en el JSONL |
| `src/latencia_fuera_de_linea.py` | opciones `--voz pregrabada` y `--solo-xtts` |
| `scripts/voz_pregrabada_cortar.py` | regenera los 142 WAV y el índice a partir de las dos tomas de ElevenLabs |
| `requirements-voz.txt` | entorno aparte del servicio XTTS (coqui-tts, torch, transformers…) |
| `docs/voz_escucha/ronda{1,2,3}_clave.json` | clave de cada ronda de escucha a ciegas (letra → voz) y tiempo de síntesis medido |
| `docs/LATENCIA_VOZ_PREGRABADA.json`, `models/latencia_voz_2026-11-23/` | latencia fuera de línea con la voz pregrabada y el control con Mónica |
| `docs/LATENCIA_VOZ_XTTS.json` | tiempo del respaldo XTTS-v2 por frase en MPS y en CPU |
| `tests/test_voz_pregrabada.py`, `tests/test_voz_xtts_servicio.py` | pruebas (sin PyTorch y sin reproducir audio) |

---

## 1. Cómo funciona

### 1.1 Cadena por frase

`src/tts.py`, modo `pregrabada`, decide qué voz dice cada frase en este orden:

1. **`pregrabada`.** Si el enunciado **exacto** está en el índice, se publica su WAV tal cual: no se copia ni se
   sintetiza.
   - La búsqueda solo normaliza a NFC. «Gracias» no encuentra «Gracias.».
   - El WAV del índice nunca se borra: los borrados del módulo solo actúan en su carpeta temporal.
2. **`xtts`.** Si no está pregrabada y el servicio XTTS está listo, se sintetiza a WAV.
   - Lo sintetizado queda en la cache de la sesión (la misma cache por texto que ya tenía el módulo): la segunda vez
     la frase sale de la cache.
3. **`local`.** En los demás casos se usa Mónica, por el mismo camino del modo offline (sintetizador persistente o
   `say`). Esos casos son: no hay servicio, el servicio sigue cargando, el servicio cayó o la frase falló en XTTS.
   - En este modo lo local **no** se guarda en la cache. Así la frase pasa a XTTS en cuanto el servicio esté listo.

La reproducción es la que ya tenía el proyecto: `afplay` en macOS, y si no hay, pygame o ffplay.

**Registro.** Cada ensayo y cada turno del JSONL llevan dos claves nuevas:

- `tts_fuente`: `pregrabada`, `xtts` o `local`;
- `tts_tiempo_sintesis_ns`: el tiempo desde `prepare()` hasta que el audio quedó listo.

Si el motor no informa, no se escriben. Esto sigue la regla de `tts_sintesis_desde_cache`. Van después de esa clave
y no mueven ninguna marca. Con la voz pregrabada, `tts_sintesis_desde_cache` no se escribe, porque la cache no
interviene. La consola del prototipo añade `voz <fuente>` a la línea de cada ensayo.

**Cobertura según el conversor.**

| Conversor | Glosas solas pregrabadas | Secuencias del inventario pregrabadas | El resto va a |
|---|---:|---:|---|
| `inventario` (por defecto), géneros por defecto | 50 / 50 | 92 / 92 | (el respaldo con lista de glosas) XTTS → Mónica |
| `inventario` con `--genero-primera f` o `--genero-segunda f` | 50 / 50 | 88-91 / 92 | XTTS → Mónica |
| `reglas` | 50 / 50 | 92 / 92 | XTTS → Mónica (frases fuera del inventario) |
| `heredado` (GrammarMapper) | 7 / 50 | 14 / 92 | XTTS → Mónica |

- **Variantes en femenino.** Hay 4 sin audio: «Hola, bienvenida.», «Buenos días, bienvenida.», «Tú eres
  bienvenida.» y «Yo me siento bienvenida.».
- **Conversor heredado.** GrammarMapper escribe sin punto final («Gracias»), así que casi nada coincide con el
  índice.

### 1.2 Servicio XTTS-v2 (proceso aparte)

**Por qué un proceso aparte.** El proyecto no puede importar PyTorch, porque TensorFlow y MediaPipe ya conviven en
su entorno.

- XTTS-v2 corre en un **proceso auxiliar persistente**, lanzado con **otro intérprete**: un entorno con coqui-tts
  (`requirements-voz.txt`).
- Ese proceso carga el modelo una sola vez y sintetiza a archivo bajo pedido.
- `src/voz_xtts_servicio.py` contiene el servidor y el cliente. Solo el servidor importa torch, y lo hace dentro de
  una función. Una prueba comprueba que importar la voz y el prototipo no carga `torch`, `TTS`, `torchaudio` ni
  `transformers`.

**Protocolo.** JSON por línea por stdin/stdout. Se eligió por ser lo más simple y robusto:

- no usa puertos;
- el servicio termina solo si el prototipo se cierra (fin de su stdin);
- si el servicio se cae, el cliente lo ve como fin de su stdout.

El servidor reserva el descriptor 1 para el protocolo y redirige todo lo que impriman coqui-tts o torch a stderr.
El WAV se escribe primero en `*.parcial` y luego se renombra.

**Estados del cliente.** `cargando` → `listo` → `caido`. El cliente da el servicio por **caído**, y no lo vuelve a
usar en la sesión, si ocurre cualquiera de estas cosas:

- el proceso termina;
- dice `error` al cargar;
- no está listo en 180 s;
- no responde a una frase en 30 s;
- se rompe el canal.

En los dos casos de tope vencido, el cliente termina el proceso que lanzó. Una frase que falla (`ok: false`) no tumba
el servicio: solo esa frase va a Mónica.

**Arranque.**

- **Intérprete.** Se indica con `--python-voz` en `realtime.py` o con la variable `TESIS_PYTHON_VOZ`. Sin ninguna de
  las dos no hay servicio y el respaldo es Mónica.
- **Dispositivo.** `--xtts-dispositivo auto` usa MPS si torch lo ofrece y si no la CPU.
- **Sin bloquear.** `realtime.py` construye la voz **antes** de cargar el modelo y MediaPipe, así que XTTS carga en
  paralelo. Después espera al servicio como mucho `--xtts-espera-seg` (20 s) antes de abrir la cámara. Si no llegó,
  la sesión empieza con Mónica para lo no pregrabado y pasa a XTTS sola cuando el servicio está listo. Lo avisa por
  consola: «[voz] servicio XTTS-v2 listo».
- **Precalentado.** El precalentado del prototipo, en modo pregrabada, solo sintetiza con XTTS las frases que no
  están en el índice. Con `inventario` no hay ninguna, y con `heredado` son 43.

**Salvaguarda contra el balbuceo.** Medido el 23-nov-2026 (sección 4.2).

- **El defecto.** En MPS, «Buenas tardes, tú eres bienvenido.» con la semilla 2026 dio 10,9 s de audio: la frase y
  ~8 s de sílabas sin sentido. Whisper transcribió «¡Paníme a me aníome, camízones y meón! …». En CPU, o con otras
  semillas en MPS, la misma frase dura 3,0-3,6 s.
- **Lo que hace el servidor.** Si el audio dura más de **1,5 s + 0,12 s × caracteres**, repite con la semilla
  siguiente, hasta 2 veces. Si sigue largo, responde con error y la frase sale con Mónica.
- **Por qué ese límite.** El máximo normal medido es 0,17 s por carácter, en frases de 19 caracteres (límite 3,8 s).

**Semilla.** Cada pedido fija la semilla (2026), así que la misma frase suena igual en cada sesión con el mismo
dispositivo.

### 1.3 Predeterminados y opciones

- `python src/realtime.py` usa el modo `pregrabada` si existe `models/voz_pregrabada/indice.json`. Si no existe, usa
  `offline` (Mónica) y avisa por consola.
- Si la voz pregrabada no se puede construir (índice inválido, sin reproductor), el prototipo sigue con `offline`. Si
  tampoco puede, sigue sin voz.
- En modo offline, sin `--tts-voice`, se elige la voz del sistema llamada «Mónica»:
  - `elegir_voz_local`, en `src/tts.py`;
  - macOS 26 la lista como «Mónica (Spanish (Spain))»;
  - las premium/enhanced van antes que la compact.
- Si no hay Mónica, sigue valiendo la regla de antes: es-CO > América Latina > es-ES, por calidad. En este Mac esa
  regla elegiría hoy **Carlos (es-CO)**, que está instalada.

```bash
python src/realtime.py                                        # pregrabada; respaldo Mónica (sin servicio)
python src/realtime.py --python-voz <python de un entorno con coqui-tts>   # pregrabada → XTTS → Mónica
TESIS_PYTHON_VOZ=<python con coqui-tts> python src/realtime.py              # lo mismo con la variable
python src/realtime.py --tts offline                          # solo Mónica
python src/realtime.py --tts offline --tts-voice com.apple.voice.compact.es-MX.Paulina   # otra voz local
python src/realtime.py --xtts-dispositivo cpu --xtts-espera-seg 0
python src/voz_pregrabada.py                                  # valida el índice y la cobertura
python src/voz_xtts_servicio.py --probar "Hola, tío." --python-voz <python con coqui-tts>
python src/tts.py --mode pregrabada --text "Gracias."
```

---

## 2. Escucha a ciegas (cuatro rondas y la voz final)

**Método.**

- **Oyente.** Un solo oyente: el autor.
- **Archivos.** En cada ronda cada voz dijo las mismas frases y quedó en un archivo con una letra (`voz_A.m4a`, …). El
  orden de las letras se barajó con una semilla fija, y el oyente escuchó sin saber qué voz era cada letra.
- **Tiempos.** Las claves (letra → voz) y el tiempo de síntesis medido (mediana por frase) están en
  `docs/voz_escucha/`.
- **Calificaciones.** Las hojas de calificación numérica que acompañaban cada ronda no se llenaron. El resultado de
  cada ronda es el que el autor comunicó.
- **Frases.** En las rondas 2 y 3 fueron «Hola, ¿cómo estás?», «Buenos días a todos.», «El hermano trabaja.»,
  «¿Dónde vive el hermano?» y «Muchas gracias.».
- **Comparabilidad.** Los tiempos de rondas distintas **no son comparables entre sí**: cambian el método (con o sin
  conversión a WAV) y la carga del equipo. Mónica, por ejemplo, salió 570 ms en la ronda 1, 1.108 ms en la 2 y
  643 ms en la 3.

**Ronda 1: 8 voces de macOS + gTTS.** Solo Mónica sonó natural.

| Letra | Voz | Motor | ms por frase (mediana) |
|---|---|---|---:|
| A | Rocko (es_MX) | `say` | 391,2 |
| B | Flo (es_MX) | `say` | 351,4 |
| C | Eddy (es_MX) | `say` | 345,7 |
| D | Google gTTS (en línea, es, tld com.mx) | gTTS | 297,6 |
| E | Paulina (es_MX, la de entonces) | `say` | 585,9 |
| F | Shelley (es_MX) | `say` | 407,9 |
| G | Reed (es_MX) | `say` | 390,1 |
| H | Sandy (es_MX) | `say` | 399,3 |
| **I** | **Mónica (es_ES)** | `say` | 569,6 |

**Ronda 2: Piper y Kokoro, con Mónica y Paulina de referencia.** Ninguna convenció, y Mónica ya no destacó.

| Letra | Voz | ms por frase (mediana) |
|---|---|---:|
| A | macOS Mónica (es_ES; ganó la ronda 1) | 1.107,5 |
| B | Kokoro ef_dora | 669,8 |
| C | Kokoro em_alex | 278,3 |
| D | Piper es_ES sharvard M (medium, hablante 1) | 62,8 |
| E | Piper es_MX ald (medium) | 68,1 |
| F | Piper es_AR daniela (high) | 955,5 |
| G | Piper es_ES davefx (medium) | 192,0 |
| H | Piper es_ES sharvard F (medium, hablante 0) | 148,5 |
| I | Kokoro em_santa | 1.927,5 |
| J | macOS Paulina (es_MX; la de entonces) | 1.169,8 |

**Voces «mejoradas» de macOS.** El autor las probó aparte y las descartó por robóticas. De esa prueba no hay
archivo de clave ni tiempos.

**Ronda 3: 8 hablantes de XTTS-v2 + Mónica.** XTTS se ejecutó en CPU, con la semilla 2026. El autor eligió
**Gilberto Mathias** (E).

| Letra | Voz | ms por frase (mediana, CPU) |
|---|---|---:|
| A | XTTS-v2 Claribel Dervla | 5.814,6 |
| B | XTTS-v2 Marcos Rudaski | 3.425,8 |
| C | XTTS-v2 Ana Florence | 2.668,3 |
| D | XTTS-v2 Sofia Hellen | 2.876,6 |
| **E** | **XTTS-v2 Gilberto Mathias** | 2.013,6 |
| F | XTTS-v2 Luis Moray | 697,3 |
| G | XTTS-v2 Alma María | 787,4 |
| H | macOS Mónica (referencia) | 643,1 |
| I | XTTS-v2 Dionisio Schuyler | 901,6 |

Los tiempos de XTTS de esta ronda incluyen el arranque en frío de cada hablante. Con el servicio ya cargado, Gilberto
Mathias tarda ~0,4 s por frase en MPS y ~1,1 s en CPU (sección 4.2).

**Voz final de los enunciados fijos: ElevenLabs.** Para los 142 enunciados que el prototipo puede decir en modo
inventario, el autor generó los audios con ElevenLabs, voz **«Robert - Colombian, Paisa»** de la biblioteca de
ElevenLabs (sección 3). XTTS-v2 con Gilberto Mathias queda como respaldo para lo demás, y Mónica como último
recurso.

---

## 3. Los 142 audios pregrabados

### 3.1 Qué enunciados

Son los que produce `ConversorGlosas(estricto=True)` (`src/conversor_glosas.py`, versión 3a-2) con los géneros por
defecto:

- las 50 glosas solas, en orden de id;
- las 92 secuencias de `INVENTARIO`, en su orden.

Son 142 textos distintos. `src/voz_pregrabada.py` los **enumera desde el código**. Las pruebas comprueban tres cosas:

- coinciden uno a uno, y en el mismo orden, con los textos del índice;
- la cobertura es del **100 %** (142/142, ninguno sobrante);
- las únicas variantes posibles sin audio son las 4 en femenino de la sección 1.1.

El número N de cada archivo es la fila N de la lista que se entregó para generar las voces.

### 3.2 Generación (el autor, 23-nov-2026)

- **Voz.** ElevenLabs, «Robert - Colombian, Paisa».
- **Tomas.** Dos, exportadas como MP3 de 128 kb/s, 44.100 Hz y mono:
  - frases 1-71: `ElevenLabs_2026-11-23T05_29_16_Robert - Colombian, Paisa_pvc_s50_sim75_v4.mp3`, 68,5 s;
  - frases 72-142: `ElevenLabs_2026-11-23T05_29_57_Robert - Colombian, Paisa_pvc_s50_sim75_v4.mp3`, 102,1 s.
- **Texto de entrada.** Las frases separadas por `<break time="1.5s" />`.
- **Ajustes.** Los de la cuenta del autor; el nombre del archivo exportado los resume (`pvc_s50_sim75_v4`).

### 3.3 Corte con alineación de Whisper

1. **Conversión.** MP3 → WAV PCM 16 bits a 44,1 kHz (para cortar) y a 16 kHz (para Whisper), con `afconvert`.
2. **Alineación.** **faster-whisper, modelo «base»** (CPU, int8, idioma `es`, beam 5, marcas de tiempo por palabra)
   transcribe cada toma. El `initial_prompt` son las 8 primeras frases de la toma. Las palabras reconocidas se
   alinean con el texto esperado (`difflib.SequenceMatcher` sobre palabras sin tildes ni puntuación).
3. **Corte.** En el punto medio entre la última palabra de una frase y la primera de la siguiente. Si falta una
   marca, se interpola en proporción a la longitud del texto.
4. **Recorte del silencio de los bordes.** Se conserva desde 60 ms antes de la primera ventana de 10 ms que supera
   (máximo − 40 dB) hasta 70 ms después de la última.
5. **Verificación automática.** Cada corte se re-transcribe con el mismo modelo y se compara con el texto esperado.
   - Dudoso = similitud de caracteres < 0,8, o duración por carácter fuera de [0,45 ; 2,2] × la mediana (mediana:
     0,079 s por carácter).
   - Salieron **16 dudosos**: 005, 015, 016, 017, 020, 023, 026, 035, 040, 041, 087, 088, 093, 098, 107 y 110.
6. **Revisión a oído.** El autor escuchó los 16:
   - **14 estaban bien.** Whisper «base» falla en palabras sueltas: oyó «¡Señor!» por «Seña.», «3» por «Triste.»,
     «2» por «Tú.».
   - **2 estaban cortados: 026 «Mal.» y 088 «Tú comes.»** Se rehicieron con la ventana corregida a mano (026: 21,38-
     22,08 s de la toma 1; 088: 22,95-23,98 s de la toma 2), con un fundido lineal de 10 ms en cada borde y sin
     recorte de silencio. El autor los aprobó.
   - La re-transcripción de los cortes nuevos dio «MAL» y «¡Tú comis!».

**El índice lo registra.** Para cada frase, `indice.json` guarda el corte (toma, inicio, fin y si se corrigió a mano),
la transcripción de verificación y su similitud, y además:

- `revisado_a_oido`: `true` en los 16 dudosos;
- `corte_rehecho`: `true` en 026 y 088.

Los otros 126 tienen solo la verificación automática.

### 3.4 Índice y tamaño

- **Contenido de `indice.json`.** Para cada frase guarda:
  - `numero`, `texto` exacto (NFC), `sha256_texto`, `archivo` y `duracion_s`;
  - `bytes`, `sha256_audio` y `origen` (p. ej. «seña sola: gracias» o «S-08: ¿dónde? + hermano + vivir»);
  - `corte`, `verificacion_whisper`, `revisado_a_oido` y `corte_rehecho`.
- **Procedencia, una vez para todo el índice.** Proveedor, voz, autor, fecha, licencia, atribución, archivos de
  origen y método de corte.
- **Validación al cargar.** Números consecutivos, sha256 del texto y del audio, archivo presente, WAV mono legible,
  duración coincidente y textos sin repetir. Un índice alterado no se carga, y el prototipo pasa a `offline` con
  aviso.
- **Tamaño.** **14.711.216 bytes de WAV (14,0 MiB)**, 166,7 s de audio. Duración por frase: de 0,34 s a 2,4 s.
  `indice.json` ocupa 88 KB más.

### 3.5 Cómo regenerarlos

`scripts/voz_pregrabada_cortar.py` repite todo el proceso de la sección 3.3 a partir de las dos tomas:

- toma los textos del conversor;
- aplica los cortes corregidos a mano que encuentre en el índice anterior;
- escribe los WAV, `cortes.json`, `verificacion.json` e `indice.json`;
- lista los dudosos para escucharlos.

Se ejecuta con un entorno que tenga faster-whisper, soundfile y numpy (la sección opcional de
`requirements-voz.txt`), en macOS por `afconvert`:

```bash
<python con faster-whisper> scripts/voz_pregrabada_cortar.py \
    --mp3 "ElevenLabs_..._05_29_16_....mp3" "ElevenLabs_..._05_29_57_....mp3" \
    --salida <carpeta nueva> --correcciones models/voz_pregrabada/indice.json
python src/voz_pregrabada.py --indice <carpeta nueva>/indice.json     # validar y medir cobertura
```

**Comprobado el 23-nov-2026.** Regenerados en una carpeta aparte, los **142 WAV salieron idénticos byte a byte** a los
del repositorio (mismo sha256). Además, el `indice.json` del repositorio es exactamente el que arma
`construir_indice` con la verificación original.

**Whisper no es del todo determinista.** Al regenerar, la re-transcripción de «Juicioso.» (030) dio «¡Güecioso!»
(similitud 0,75) y la marcó como dudosa; en el corte original había dado «cuícioso.» (0,875). Por eso el script **no**
marca como revisados a oído los dudosos nuevos. Solo conserva los que el índice anterior ya tenía marcados, y lista
los demás para escucharlos.

**Si cambia el conversor.** Si cambia el inventario o el léxico, cambian los textos. La prueba de cobertura falla y
hay que generar en ElevenLabs las frases nuevas.

---

## 4. Latencia

### 4.1 Voz pregrabada: L_resp fuera de línea con el modelo final + inventario

**Método.** El de `docs/LATENCIA_4c_MODELO_FINAL.md`, ahora con la voz pregrabada:

```bash
python src/latencia_fuera_de_linea.py --voz pregrabada --configuraciones final_inventario --sin-extractor \
    --salida docs/LATENCIA_VOZ_PREGRABADA.json --crudo models/latencia_voz_2026-11-23/crudo.json
```

- **Muestra.** 100 clips × 3 repeticiones.
- **Modelo.** El final, con umbral 0,68 (N-75). En 4.c el umbral era 0,00. Por eso 6 de 300 ensayos son
  `rechazada_confianza` y no tienen t5: las cifras de abajo son sobre 294.
- **Control.** En la misma sesión se repitió con la voz offline (Mónica, cache precalentada como en vivo), con
  `--voz offline`. Sus archivos son `models/latencia_voz_2026-11-23/control_offline*.json`.
- **Carga.** La carga de 1 min fue 1,6-1,8 en ambas. En 4.c fue de 4-7.

**Resultados.** ms; mediana / p90 / p99 / máx.

| Magnitud | **Pregrabada** | Control Mónica, desde la cache | Control Mónica, sin cache | 4.c (8-nov, Paulina, cache) |
|---|---|---|---|---|
| síntesis (t5 − t4) | **0,09 / 0,12 / 0,17 / 0,19** | 0,33 / 0,68 / 0,76 / 5,57 | 16,1 / 19,1 / 23,6 / 129,6 | 0,36 / 0,45 / 0,65 / 2,42 |
| inferencia | 12,3 / 13,9 / 22,4 / 34,1 | 11,9 / 12,8 / 14,3 / 15,7 | 12,8 / 13,7 / 14,5 / 14,9 | 14,4 / 17,9 / 22,5 / 24,2 |
| **post_cierre (≈ L_resp sin la última extracción)** | **12,8 / 14,5 / 23,4 / 34,6** | 12,8 / 13,6 / 15,2 / 20,5 | 29,4 / 32,6 / 37,4 / 143,4 | **15,2** / 18,8 / 23,7 / 25,2 |
| tiempo del motor (`tts_tiempo_sintesis_ns`) | 0,028 | — | — | — |
| fuente | `pregrabada` en los 294 | `local` (cache) | `local` | — |

Medianas de post_cierre por repetición: pregrabada 12,6 / 13,6 / 12,7 ms; control con cache 12,9 / 12,7 / 12,7 ms. La
condición `sin_cache` con voz pregrabada da lo mismo (13,0 ms de mediana), porque la cache no interviene.

**Turnos.** Fin de turno con frase nueva, sin cache. Mediana de cada repetición:

| | Glosas solas (50) | Secuencias del inventario (92) | Máximo, secuencias |
|---|---|---|---|
| Pregrabada | 0,009 / 0,008 / 0,008 ms | 0,012 / 0,008 / 0,008 ms | 0,04 ms |
| Mónica sin cache | 15,1 / 15,3 / 15,3 ms | 23,3 / 23,3 / 23,1 ms | 37-38 ms |

**Lectura.**

1. **Con la voz pregrabada, la etapa de síntesis baja a 0,09 ms de mediana** y deja de depender de la cache.
   - Mónica cuesta 0,33 ms desde la cache y 16 ms sin ella.
   - Una secuencia del inventario dicha por primera vez cuesta 0,01 ms en vez de 23 ms.
2. **El post_cierre es 12,8 ms**, igual que el control con Mónica desde la cache medido en la misma sesión (12,8 ms).
3. **La diferencia con los 15,2 ms de 4.c no es de la voz.** Viene de la inferencia: 12,3 ms hoy frente a 14,4 ms
   entonces, con el equipo menos cargado. Comparado en igualdad de condiciones, la voz pregrabada no cambia L_resp
   cuando la cache de Mónica está caliente, y lo mejora en ~16 ms cuando no lo está.
4. **Los máximos de 23-35 ms** con la voz pregrabada son de la inferencia (p99 22,4, máximo 34,1 ms en la misma fila),
   no de la voz.

### 4.2 Respaldo XTTS-v2 con el servicio real (MPS y CPU)

**Método.**

```bash
python src/latencia_fuera_de_linea.py --solo-xtts --python-voz <python con coqui-tts> \
    --salida docs/LATENCIA_VOZ_XTTS.json
```

- **Frases.** 20 que el prototipo puede decir y que **no** están pregrabadas, elegidas con la semilla 2026:
  - las 4 variantes en femenino;
  - 8 salidas del conversor `reglas` fuera del inventario (p. ej. «¿Por qué trabajan ustedes?»);
  - 8 respaldos del modo inventario con lista de glosas (p. ej. «hola, nunca, tú»).
- **Repeticiones.** 3, sin cache, a través de `TTSModule(mode="pregrabada")`: lo mismo que vería el prototipo.
- **Medidas.** El tiempo de `prepare()` hasta tener el WAV y el tiempo del servidor. No sonó nada.
- **Carga.** La carga de 1 min fue 1,3-2,2.
- **Fuente.** Las 120 frases salieron de XTTS.

| | MPS | CPU |
|---|---|---|
| Servicio listo desde el lanzamiento | 16,3 s (carga del modelo 15,4 s, con una síntesis de calentamiento de 0,93 s) | 15,3 s (14,8 s; 0,99 s) |
| **Por frase, mediana** | **417 ms** | **1.094 ms** |
| **Por frase, p90** | **554 ms** | **1.380 ms** |
| p99 / máximo | 2.521 / 2.524 ms | 1.546 / 1.551 ms |
| mínimo | 231 ms | 681 ms |
| Medianas por repetición | 427 / 408 / 410 ms | 1.087 / 1.097 / 1.096 ms |
| Duración del audio, mediana (máx.) | 2,62 s (3,62 s) | 2,57 s (3,62 s) |
| Factor de tiempo real (mediana) | 0,16 | 0,43 |

- **Ida y vuelta.** El cliente añade < 0,5 ms al tiempo del servidor.
- **Dato de partida.** Coincide en MPS (~400 ms). En CPU estas frases tardan 1,1 s, más que los ~760 ms del dato de
  partida.
- **El máximo en MPS.** Es «Buenas tardes, tú eres bienvenido.», en las tres repeticiones (2,5 s): el intento con la
  semilla 2026 balbucea (sección 1.2), se descarta y se repite. Sin esa frase, la mediana en MPS es 409 ms. En vivo
  ese costo se paga una vez, porque después la frase sale de la cache.
- **Antes de la salvaguarda.** Esa frase dejaba un WAV de 10,9 s con ~8 s de sílabas sin sentido, en las tres
  repeticiones.

**Qué significa para L_resp.** Una frase no pregrabada añade a L_resp la síntesis de XTTS:

| Caso | Se añade |
|---|---|
| XTTS en MPS | ~0,4 s (p90 0,55 s) |
| XTTS en CPU | ~1,1 s (p90 1,4 s) |
| Mónica sin cache, mientras XTTS carga o si cayó | ~16-23 ms |
| Desde la cache de la sesión | < 1 ms |

**Lo que no se midió.** Una sesión en vivo con cámara y servicio XTTS. Las cifras de esta sección son del servicio
aislado.

---

## 5. Licencias y publicación

- **Audios pregrabados (ElevenLabs).**
  - Se generaron con un **plan de pago de ElevenLabs que permite su uso y distribución**.
  - Por eso los 142 WAV y su índice **se incluyen en los dos repositorios**: este privado y el público.
  - **Atribución:** voz «Robert - Colombian, Paisa» de la biblioteca de ElevenLabs. Va en `indice.json` (`fuente`) y
    aquí.
- **XTTS-v2 (Coqui).**
  - Modelo con la **Coqui Public Model License 1.0.0**, que solo permite el uso **no comercial** del modelo y de sus
    salidas. Este prototipo es académico.
  - El modelo **no se redistribuye**: el servicio lo descarga del repositorio `tts-hub/XTTS-v2` de Hugging Face la
    primera vez que se instala y arranca (`model.pth`, ~1,9 GB).
  - Al lanzar el servicio se fija `COQUI_TOS_AGREED=1`, que declara que se aceptan esos términos.
  - Ningún audio de XTTS se guarda en el repositorio. Los de la sesión viven en la carpeta temporal y se borran al
    cerrar.
- **Whisper.** faster-whisper y el modelo «base» se usaron solo para cortar y verificar los audios. No forman parte del
  prototipo.
- **Voz local.** Mónica es una voz del sistema macOS. No se distribuye: la usa el sistema de quien ejecute el
  prototipo.

---

## 6. Límites

- **Evaluación.** Hubo **un solo oyente**, el autor, que además conoce las frases. Ninguna persona sorda ni ningún
  oyente ajeno al proyecto ha evaluado las voces. Las rondas no dejaron calificaciones numéricas.
- **Verificación de los cortes.** La verificación automática es Whisper «base», que se equivoca en palabras sueltas.
  Solo los 16 dudosos se escucharon uno a uno.
- **Cobertura.** Es del 100 % con los géneros por defecto. Con `--genero-primera f` o `--genero-segunda f`, 1 a 4
  enunciados no están pregrabados y suenan con otra voz (XTTS o Mónica). Lo mismo pasa con el conversor heredado,
  con el que la mayoría de las frases no están pregrabadas.
- **Mezcla de voces.** Dentro de una sesión pueden sonar hasta tres voces distintas: Robert, Gilberto Mathias y
  Mónica. El JSONL dice cuál sonó en cada ensayo.
- **XTTS.**
  - Necesita un entorno aparte de ~80 paquetes y la descarga del modelo.
  - Sin `--python-voz` ni `TESIS_PYTHON_VOZ`, el respaldo es Mónica.
  - La salvaguarda contra el balbuceo detecta audios **demasiado largos**, no frases mal dichas de duración normal.
  - El defecto se vio en 1 de 20 frases y solo en MPS.
- **Dispositivo.** MPS solo existe en Mac con Apple Silicon. En otro equipo, `auto` usa la CPU (~1,1 s por frase en
  este Mac).
