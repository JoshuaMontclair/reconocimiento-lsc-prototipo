# Actividad 4.c: latencia de punta a punta con el modelo final (D-14)

Fecha: 8 de noviembre de 2026. Equipo: MacBook Pro M5 Max, 18 núcleos, 48 GB, macOS 27.0.1, solo CPU
(TensorFlow 2.21.0, MediaPipe 0.10.35, OpenCV 4.13.0, Python 3.12.13; commit 73b1fbe).

**Qué hay y qué no hay en este informe.**

- **Medición en vivo con el modelo final y `--conversor inventario`: NO HAY.** No se abrió la cámara ni se ejecutó
  `realtime.py`. La sección 5 deja el protocolo para esa sesión.
- **Medición FUERA DE LÍNEA de lo que pasa después del cierre de la ventana** (preprocesamiento, inferencia,
  conversor, síntesis): sí. Usa el modelo final y la heredada con el mismo método (sección 2).
- **Tasa del extractor MediaPipe** sobre fotogramas del corpus leídos en memoria: sí (sección 3).
- **Caracterización de `registros/latencia_2026-10-03.jsonl`**, que no está en DECISIONES ni en la tesis: sí, pero
  sin decidir cómo se obtuvo (sección 4).

Archivos:

| Archivo | Qué contiene |
|---|---|
| `src/latencia_fuera_de_linea.py` | script de la medición (nuevo); `tests/test_latencia_fuera_de_linea.py`, 23 pruebas |
| `docs/LATENCIA_4c_MODELO_FINAL.json` | todos los resúmenes, la carga del equipo por bloque y la caracterización de los cinco JSONL |
| `models/latencia_4c_2026-11-08/crudo.json` | una fila por ensayo (1.800) y por frase de turno |

Comando: `python src/latencia_fuera_de_linea.py` (21:21-21:50).

---

## 1. Lo que la tesis cita hoy

**36 ms** entre el cierre de la ventana y la orden de reproducción: L_resp = t5 − t1, segunda sesión de N-52
(22-sep-2026).

- Configuración: red heredada (canónico «manos», «mascara», umbral 0,20), voz offline desde la cache.
- n = 3 ensayos: 36,4 ms de media (35,1-37,2).
- La propia N-52 dice que «no es una cifra para reportar en RE5».

Los otros registros en vivo de la heredada, resumidos con `src/latencia_resumen.py` (L_resp en ensayos aceptados en
modo aislado):

| Registro | Síntesis dentro de t5 | n | L_resp mediana (ms) | Inferencia mediana (ms) | Extractor por fotograma, mediana (ms) |
|---|---|---:|---:|---:|---:|
| 2026-09-14 | no (anterior a N-34) | 15 | 33,5 | 19,2 | 12,1 |
| 2026-09-15 | no | 2 | 34,3 | 19,7 | 12,3 |
| 2026-09-22 | sí, desde la cache | 3 | 36,8 | 21,7 | 12,3 |
| 2026-09-30 | sí, desde la cache | 14 | 47,4 | 28,4 | 15,7 |

El registro del 30-sep no figura en DECISIONES. Tiene la inferencia y el extractor más lentos que los demás, y en el
JSONL no hay nada que lo explique: no se registra la resolución ni la carga del equipo.

---

## 2. Medición fuera de línea tras el cierre de la ventana

### 2.1 Método

**Entrada.** Puntos ya extraídos del corpus (`data/extracted/LSC50_holistic`, solo lectura), cargados con
`preprocess_lsc50.load_single_sample`.

- Muestra estratificada de **100 clips**: 2 por seña.
- Los señantes quedan equilibrados (20 clips cada uno).
- Semilla 2026; la lista de videos está en el JSON.

**Cadena: la misma del prototipo.** Cada clip pasa fotograma a fotograma por `LSCTranslator` (`src/translator.py`),
con un extractor pregrabado que devuelve esos puntos al instante.

- La ventana es el clip completo: N mediana 68 fotogramas, frente a los 75 en vivo.
- Las marcas t2..t5 las sella el propio traductor, así que las etapas son las mismas del JSONL:
  - preprocesamiento = t2 − e_N;
  - inferencia = t3 − t2, una llamada a `model.predict`;
  - conversión = t4 − t3;
  - síntesis = t5 − t4.
- **post_cierre = t5 − e_N** es la suma de las cuatro etapas. Es L_resp sin la extracción del último fotograma, que
  se estima en la sección 3.

**Configuraciones.** El modelo se carga una vez por configuración. El traductor hace su inferencia de calentamiento
y además se descartan 10 ensayos antes de medir.

| Nombre | Modelo | Conversor | Umbral (de `<modelo>.umbral.json`) |
|---|---|---|---|
| `final_inventario` | `models/tcn_imputacion/modelo.keras` (TCN + imputación + 252) | `inventario` (D-15) | 0,00 |
| `heredada_heredado` | red heredada (`canonico_holistic_mascara_manos`; era la ruta por defecto de `realtime.py` hasta el 22-nov-2026) | heredado (GrammarMapper) | 0,20 |
| `final_heredado` | modelo final | heredado | 0,00 |

`final_heredado` es un control: es la configuración del registro del 3-oct y permite separar el efecto del modelo
del efecto del conversor.

**Voz.** `src/tts.py`, modo offline (voz Paulina, ejecutor `say` con el sintetizador persistente de macOS), con la
síntesis real a archivo. La reproducción se sustituyó por una función que solo borra el archivo: **no sonó nada**.
La etapa «síntesis» del prototipo termina en la orden de reproducción, no cuando suena, y la orden (el hilo de
`play_prepared_async`) sí se ejecuta. Hay dos condiciones:

- **cache**: como en vivo. El traductor precalienta las 50 glosas sueltas al arrancar y las 50 quedaron en cache.
- **sin_cache**: peor caso. La frase se sintetiza en ese momento.

**Repeticiones.** Tres. En cada una, todas las configuraciones en orden alterno (directo, inverso, directo).

**Control.** «hola» 0030_0000_0003 da «hola» 0,959 con el modelo final, igual que la prueba sin cámara de N-65
(0,959). Con la heredada da 0,876.

### 2.2 Carga del equipo durante la medición

Antes de cada bloque se esperó hasta 3 min a que la carga de 1 min bajara de 3. **No bajó en ningún bloque**, así
que se esperó el tope cada vez (6 × 180 s = 18 min) y **se midió igual**.

| Bloque | Carga 1 min al empezar | Carga 1 min al terminar | Duración |
|---|---:|---:|---:|
| post-cierre, repetición 1 | 5,93 | 6,74 | 27 s |
| post-cierre, repetición 2 | 4,69 | 6,00 | 30 s |
| post-cierre, repetición 3 | 5,43 | 5,97 | 29 s |
| extractor, repetición 1 | 4,37 | 7,02 | 174 s |
| extractor, repetición 2 | 4,22 | 6,84 | 175 s |
| extractor, repetición 3 | 3,32 | 4,59 | 168 s |

En todos los bloques había un proceso `node (vitest 1)` con ≈113 % de CPU. Llevaba 5 días corriendo y no es de esta
tarea ni de los cálculos de 4.d y 4.e. En algunos bloques había además otros procesos `Python` al 50-87 %, probablemente
esos cálculos. No se mató ningún proceso.

Con 18 núcleos, una carga de 4-7 deja la mayoría libres, pero la variación entre repeticiones (abajo) puede deberse
en parte a esto.

### 2.3 Resultados (ms; 300 ensayos por fila = 100 clips × 3 repeticiones)

**Condición «cache» (la de una sesión en vivo con las glosas sueltas).** Valores agregados: mediana / p90 / p99 /
máximo.

| Etapa | final + inventario | heredada + heredado | final + heredado |
|---|---|---|---|
| preprocesamiento | 0,42 / 0,56 / 0,67 / 0,78 | 0,27 / 0,32 / 0,44 / 0,52 | 0,42 / 0,55 / 0,66 / 1,01 |
| inferencia | 14,37 / 17,88 / 22,53 / 24,21 | 17,25 / 21,14 / 23,54 / 24,82 | 14,35 / 18,27 / 20,19 / 23,92 |
| conversión | 0,04 / 0,05 / 0,07 / 0,08 | 0,03 / 0,04 / 0,05 / 0,08 | 0,03 / 0,03 / 0,06 / 0,09 |
| síntesis (desde la cache) | 0,36 / 0,45 / 0,65 / 2,42 | 0,37 / 0,45 / 0,55 / 0,64 | 0,37 / 0,47 / 0,61 / 2,27 |
| **post_cierre** | **15,17 / 18,77 / 23,65 / 25,16** | **17,93 / 21,82 / 24,23 / 25,55** | 15,23 / 19,21 / 21,33 / 25,11 |
| n con la cadena completa | 300 | 297 (3 rechazadas por confianza < 0,20) | 300 |

Variación de la mediana de post_cierre entre las tres repeticiones:

| Configuración | Medianas por repetición | Rango |
|---|---|---:|
| final + inventario | 14,22 / 17,21 / 15,01 | 3,0 |
| heredada + heredado | 15,27 / 19,27 / 19,79 | 4,5 |
| final + heredado | 15,17 / 14,12 / 18,14 | 4,0 |

Casi toda la variación está en la inferencia; preprocesamiento, conversión y síntesis varían menos de 0,05 ms.

**Condición «sin_cache» (peor caso: la frase se sintetiza en el momento).**

| Configuración | Síntesis: mediana / p90 / p99 / máx. | **post_cierre: mediana / p90 / p99 / máx.** | Medianas de post_cierre por repetición |
|---|---|---|---|
| final + inventario | 20,61 / 25,51 / 30,04 / 151,76 | **38,89 / 45,31 / 55,32 / 164,75** | 33,46 / 41,13 / 40,82 |
| heredada + heredado | 21,10 / 25,93 / 32,12 / 143,84 | **41,03 / 46,91 / 55,42 / 159,19** | 34,75 / 43,06 / 42,51 |
| final + heredado | 19,91 / 24,32 / 32,59 / 144,45 | 36,17 / 42,18 / 51,96 / 159,22 | 34,92 / 36,44 / 36,74 |

Los máximos de ≈150 ms son una única síntesis lenta por configuración. Ningún p99 pasa de 33 ms.

**Turnos (modo oración).** Conversión de las 50 glosas solas y de las 92 secuencias del inventario, y síntesis sin
cache de cada frase. Valores: mediana de cada repetición.

| Configuración | Conversión, secuencias | Síntesis sin cache, glosas solas | Síntesis sin cache, secuencias | Máximo, secuencias |
|---|---|---|---|---|
| final + inventario (92/92 con frase) | 0,04-0,05 | 17,5 / 21,8 / 20,3 | 28,2 / 36,1 / 33,3 | 47-60 |
| heredada + heredado | 0,02 | 18,1 / 21,2 / 21,3 | 28,1 / 30,5 / 28,6 | 42-53 |

### 2.4 Lectura

1. **Con el modelo final y el conversor inventario, lo que pasa tras el cierre de la ventana cuesta 15 ms de mediana
   (p90 19, p99 24, máximo 25) fuera de línea en este equipo.** El 95 % es la inferencia.
2. **La heredada, medida igual, da 18 ms.** El final fue más rápido en las tres repeticiones (por 1,0, 2,1 y 4,8 ms),
   pero la diferencia es del mismo tamaño que la variación entre repeticiones. Lo que se sostiene es que **el modelo
   final no es más lento**, no que sea «X ms más rápido».
3. **El conversor no cambia la latencia.** final + inventario y final + heredado dan la misma mediana (15,17 y
   15,23 ms), y la conversión cuesta < 0,1 ms con cualquiera de los dos.
4. **La cache de voz cuenta.** Sin ella, una frase nueva añade ≈20 ms (glosa sola) o ≈30 ms (secuencia), con colas
   raras de hasta ≈150 ms.
5. **Fuera de línea se subestima la inferencia en vivo.** La inferencia del modelo final sale 14,4 ms aquí y 19,0 ms
   en el registro del 3-oct (misma configuración que `final_heredado`). La de la heredada sale 17,3 ms aquí, frente
   a 19-22 ms en los registros del 14 al 22-sep y 28,4 ms el 30-sep. Causa probable, no verificada: aquí los ensayos
   van seguidos, mientras que en vivo MediaPipe ocupa la CPU entre una ventana y otra. Por eso **estas cifras no
   sustituyen a la sesión en vivo**.

**Estimación de L_resp en vivo (no es una medición).** post_cierre más el costo de extraer el último fotograma
(mediana 10,8-11,2 ms a 1280×720, sección 3):

- **≈26 ms** con el modelo final y la cache;
- ≈29 ms con la heredada.

Los registros en vivo dieron más: 32,5 ms (3-oct, n = 8) y 36,8 ms (22-sep, n = 3). La diferencia es coherente con
el punto 5.

---

## 3. Tasa del extractor MediaPipe (fuera de línea)

**Método.** `LandmarkExtractor` del prototipo (HolisticLandmarker de D-05, CPU, modo VIDEO), un extractor por
resolución y repetición, con 10 fotogramas de calentamiento descartados.

- Entrada: los 100 videos de la muestra, 6.686 fotogramas por repetición, decodificados con OpenCV **solo en
  memoria**. Ningún fotograma se escribió a disco.
- Cronometraje como en vivo: c_i con el fotograma ya leído, luego BGR→RGB y `extract()`, luego e_i. Costo =
  e_i − c_i.
- Dos resoluciones: la nativa del corpus (1920×1080) y reducida en memoria a 1280×720, la de la sesión en vivo.

| Resolución | Mediana por repetición | p90 | p99 | Máximo | Fotogramas > 33,3 ms | Extractor / ventana de 2,5 s (mediana) |
|---|---|---|---|---|---|---|
| 1280×720 | 11,23 / 11,19 / 10,78 | 12,37 / 12,35 / 11,54 | 13,77 / 13,60 / 12,73 | 26,9 / 21,1 / 16,4 | 0 de 20.058 | 34 / 34 / 32 % |
| 1920×1080 | 12,28 / 12,64 / 12,23 | 13,73 / 13,80 / 12,83 | 15,42 / 15,18 / 13,79 | 80,9 / 19,4 / 67,2 | 2 de 20.058 | 37 / 38 / 37 % |

**Lectura.**

- A 1280×720 el extractor procesa un fotograma en ≈11 ms. Eso permitiría ≈89 fot./s con la mediana y ≈73 fot./s con
  el p99, frente a los 30 que entrega la cámara.
- En una ventana de 75 fotogramas consume ≈0,85 s de los 2,5 s.
- **El extractor sí mantiene el ritmo dentro de la ventana con holgura.** Esto coincide con los registros en vivo:
  tasa efectiva mediana 30,4 fot./s y costo del extractor 11,9 ms por fotograma el 3-oct (1.589 fotogramas), y
  12,1-12,3 ms en septiembre.
- Fuera de línea no se incluyen el dibujo ni `imshow`/`waitKey` del bucle en vivo, que también caben en los 33 ms
  (lo dice la tasa de 30 fot./s de los registros).

---

## 4. Qué es `registros/latencia_2026-10-03.jsonl`

### 4.1 Lo que contiene

Cifras de `latencia_resumen.py` y de `caracterizar_sesion` del script nuevo.

**Contenido general.**

- 27 líneas: 22 ensayos y 5 turnos, en dos sesiones.
  - 11:05:18: registros de 11:05:29 a 11:13:27; 17 ensayos (9 en modo aislado, 8 en oración) y 3 turnos.
  - 16:15:42: registros de 16:16:12 a 16:18:30; 5 ensayos en oración y 2 turnos.
- Entró en el repositorio con el commit 64fc9dc (8-nov). El archivo tiene fecha de modificación del 3-oct a las
  16:18.

**Configuración.**

- Modelo `models/tcn_imputacion/modelo.keras`, «imputacion», umbral 0,00 (de `modelo.umbral.json`).
- TTS offline.
- **Conversor heredado**: ningún registro tiene la clave `conversion`, que solo aparece con `reglas` o `inventario`.
  Coincide con un turno «Hola ¿dónde mucho?», que es salida de GrammarMapper.

**Estados.** aceptada 21, rechazada_manos 1 (56 % sin manos), ninguna rechazada por confianza (umbral 0,00).

**Cifras de latencia** (ms; mediana salvo indicación):

| Magnitud | Valor |
|---|---|
| Extracción del último fotograma | 11,9 |
| Costo del extractor por fotograma | 11,9 (n = 1.589; p90 12,3) |
| Preprocesamiento | 0,7 |
| Inferencia | 19,0 (p90 19,9) |
| Conversión | 0,0 |
| Síntesis desde la cache | 0,5 (n = 8) |
| **L_resp** | **32,5 (n = 8, de 31,7 a 33,2)** |
| W en ventanas inferidas | 2.482 |
| Síntesis de turno | 26,6 de media (frases nuevas de 30-35 ms; una desde la cache, 0,55) |

**Tasa efectiva.** Mediana 30,4 fot./s (de 26,2 a 30,5).

**Fotogramas por ventana.** Media 72,2.

- 9 de las 22 ventanas no llegaron a 75 (N = 51, 54, 67, 72, 74…).
- Todas cerraron por fotogramas.
- `fps_medida_antes` bajó a 20,5-29,6 en esas ventanas: es el mecanismo de la ventana automática descrito en PD-09.

**Glosas aceptadas.**

- Aislado: buenos días, aburrido (2), diferente, familia, hola, tío (2).
- Oración: «buenos días + todos» (3 veces), «¿dónde? + hermano + vivir», «hola + mucho + ¿dónde?», más o menos,
  buenos días.

**Comparación con las sesiones de la heredada.** El 3-oct la inferencia sale igual o hasta ≈3 ms menor que el
14-22 de septiembre (19,0 frente a 19,2-21,7) y bastante menor que el 30-sep (28,4). La extracción y la tasa son iguales a
las de septiembre, salvo el 30-sep.

### 4.2 Rasgos que ayudarían a decidir si fue una sesión en vivo con cámara (no se concluye aquí)

| Rasgo | Lo que muestra el registro | Por qué importa |
|---|---|---|
| Campo de fuente de video | **No existe.** El JSONL no guarda cámara, resolución ni archivo de origen (tampoco en las otras sesiones) | El registro por sí solo no puede probar la fuente |
| Programa que lo escribió | `realtime.py` solo lee de `cv2.VideoCapture(<índice de cámara>)`; no acepta archivos de video. En el repositorio no hay otro script que use `LSCTranslator` con registro activo en `registros/`; las pruebas usan carpetas temporales o `registro=False` | Si lo escribió `realtime.py`, la fuente fue un dispositivo de captura (cámara física o virtual). Un script ad hoc fuera del repositorio sí podría haberlo escrito |
| Ritmo de llegada de los fotogramas | Intervalo entre fotogramas: mediana 33,4 ms, DE 4,9, de 26,8 a 71,2 ms (sesión de las 11:05) | Es el ritmo de una fuente de 30 fot./s en tiempo real. Leer un archivo sin pausas daría intervalos ≈ costo del extractor (≈12 ms) |
| Tecla → primer fotograma | t0 → c_1 entre 1,8 y 22,9 ms | Compatible con un bucle interactivo que espera el siguiente fotograma de la cámara |
| Pausas entre ventanas | De 4,3 a 218 s, irregulares | Compatible con una persona pulsando teclas, pero un script también podría producirlas |
| Detección | Proporción sin manos: mediana 0,14 (de 0 a 0,56); 0 fotogramas sin pose; 4 ventanas sin ningún fotograma con dos manos (familia, tío, tío, más o menos) | Variabilidad propia de una persona frente a la cámara. Los clips del corpus casi siempre detectan manos: 6.658 de 6.686 fotogramas con manos en la muestra (repetición 1, 1920×1080) |
| N y duración | Ventanas de 51-75 fotogramas a ≈30 fot./s, W ≈ 2,48 s | Los clips del corpus son a 24 fot./s (mediana 62 fotogramas, máximo 134, según `translator.py`); no encajan con N = 75 a 30 fot./s sin tratamiento |
| Prueba sin cámara de N-65 | Ningún registro tiene N = 69 ni «hola» 0,959/0,957 | Esa prueba no quedó en este archivo |
| Glosas y frases | Secuencias como «buenos días + todos» repetidas y frases del conversor heredado | Compatibles con una prueba del prototipo; no dicen quién señó |

**Qué faltaría para cerrarlo.**

- Que el autor declare si fue él frente a la cámara, qué cámara usó y con qué opciones.
- O repetir una sesión con el protocolo de la sección 5, que sí deja constancia.

Hasta entonces, estas cifras no deben citarse en la tesis como medición en vivo.

---

## 5. Qué falta y protocolo de la sesión en vivo

**Falta una sesión en vivo del autor** con el modelo final **y** `--conversor inventario`, la configuración que va
a la tesis (D-14, D-15). Aunque el registro del 3-oct resultara ser en vivo, usó el conversor heredado. Por lo
medido en la sección 2, el conversor no debería cambiar la latencia, pero la cifra de la tesis debe salir de la
configuración que se presenta.

### Protocolo breve (≈20 min)

1. **Equipo.**
   - Cerrar programas pesados.
   - Anotar `uptime` y `ps -Ao pcpu,comm -r | head -6` antes y después: el JSONL no guarda la carga.
   - Cámara y distancia de PD-01 a PD-31, si se usan.
   - Anotar qué cámara se usa: el JSONL tampoco la guarda.
2. **Comando** (desde la raíz del repositorio):

   ```bash
   python src/realtime.py \
       --model-path models/tcn_imputacion/modelo.keras --ancho 1280 --alto 720 \
       --conversor inventario --ventana-frames 75 \
       --registro-dir registros/latencia_4c_AAAA-MM-DD
   ```

   - Umbral, regla «imputacion» y voz offline: los toma solos de la procedencia y de `modelo.umbral.json`. No
     se pasa `--umbral` ni `--tts`. El umbral era 0,00 cuando se escribió este protocolo; desde N-75
     (22-nov-2026) es 0,68.
   - Desde el 22-nov-2026, `--model-path models/tcn_imputacion/modelo.keras`, `--ancho 1280 --alto 720` y
     `--conversor inventario` son los valores por defecto de `realtime.py`; la orden explícita sigue valiendo
     y deja constancia de la configuración. El registro gana la clave `bordes` (aviso de seña cortada, N-50,
     opción B): se calcula después de t5 y no cambia ninguna magnitud de esta sección.
   - `--ventana-frames 75` fija N como recomienda PD-09 (N-48). Si el director no lo aprobó, puede omitirse: W
     variará, pero L_resp no depende de N.
   - La carpeta propia evita que el resumen mezcle sesiones (R15 del protocolo 4.a).
   - No usar `--sin-registro`.
   - Opcional: `--genero-primera m|f`.
3. **Esperar unos 10 s tras el arranque**, para que termine el precalentado de la voz (50 frases), antes de la
   primera ventana.
4. **Bloque A, modo aislado** (el de arranque; no pulsar M): **100 ventanas**, que son dos rondas de las 50 señas.
   - Con umbral 0,00 toda ventana inferida queda aceptada; solo las rechazadas por manos no dan L_resp.
   - Si al terminar hay menos de 100 aceptadas, añadir ventanas hasta llegar a 100, sin mirar las cifras (regla de
     PD-02).
   - Es lo que da L_resp con la cadena completa: n ≥ 100 para que el p90 tenga 10 observaciones por encima.
5. **Bloque B, modo oración** (pulsar M): **10 turnos** con secuencias del inventario, para medir conversión y
   síntesis de turno. Por ejemplo, «buenos días + todos» (S-01), «tío + juicioso» (S-03), «familia + feliz» (S-04),
   cada uno cerrado con ENTER.
6. **Salir con Q** y anotar los contadores que imprime la consola.
7. **Resumen:**

   ```bash
   python src/latencia_resumen.py registros/latencia_4c_AAAA-MM-DD/ \
       --json docs/LATENCIA_4c_EN_VIVO.json
   ```

   Filas que se reportan:
   - `L_resp [sint. incluida]`, y aparte `sintesis [desde cache]` frente a `[sintetizada]`;
   - `inferencia`, `extraccion`, `extractor/fotog.`, `tasa efectiva`;
   - `W [inferidas]`, y los turnos.
   - Errores de voz: 0 esperado.
8. **Declarar en DECISIONES** quién señó: el autor, sin LSC, así que no es cifra de exactitud. Declarar también la
   cámara, la resolución y la carga. Se reportan media, DE, mediana y p90, y aparte p99 y máximo.

---

## 6. Límites

- **Fuera de línea no es en vivo.** Los ensayos van seguidos y la inferencia sale 4-5 ms más rápida que en los
  registros en vivo (sección 2.4, punto 5).
- La carga del equipo estuvo entre 3,3 y 7,0 durante toda la medición (sección 2.2). No se consiguió una condición
  descargada.
- **Las predicciones sobre clips del corpus no son una medida de exactitud.** El modelo final se entrenó con la
  partición aleatoria, que incluye parte de estos clips. Aquí solo importan los tiempos.
- La ventana fuera de línea es el clip (mediana 68 fotogramas) y no 75. El preprocesamiento escala con N, pero cuesta
  menos de 1 ms.
- La voz se sintetizó de verdad, pero no se reprodujo. El reproductor no forma parte de t5.
- El costo del extractor es el de los videos del corpus (iluminación, encuadre y señantes del corpus), reducidos a
  1280×720. Con la cámara real puede variar.
