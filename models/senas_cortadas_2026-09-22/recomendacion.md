# Señas cortadas en los bordes de la ventana: opciones, recomendación y especificación

Escrito a mano el 2026-09-22 (16:40-17:00, hora local) DESPUÉS de la corrida de
`src/analisis_senas_cortadas.py` (criterio en `criterio.md`, fijado a las 16:16; corrida vigente iniciada
a las 16:36). Las cifras salen de `resultados.json` y de `informe.md` de esta carpeta; entre corchetes, la
clave o sección de origen. Nada de esto está implementado: es material para decidir con el director.

## 1. Qué exige el anteproyecto

- §2.2.6, `09_metodologia.tex:190-198`: «Las secuencias incompletas (sin inicio o fin claros dentro de la
  ventana temporal) se descartarán» (192-193); la ventana es «de longitud fija activada por la persona
  usuaria (botón o tecla)», y la detección automática de inicio y fin «se deja como posible mejora futura»
  (193-196); estos comportamientos «se ejercitarán durante la evaluación en línea, de modo que el reporte
  de resultados distinga entre errores de clasificación y rechazos deliberados del sistema» (196-198).
- Bloque 2, `09_metodologia.tex:247-251`: ejercitar deliberadamente el comportamiento de §2.2.6, incluida
  la «secuencia incompleta»; de ese bloque solo se reportan ejecuciones, rechazos, casos límite ejercitados
  y tiempos, ninguna exactitud (`09_metodologia.tex:259-262`; `06_alcances.tex:73-75`).
- RE3 incorpora ese comportamiento: `09_metodologia.tex:463-465` y `10_resultados.tex:28-30`
  («e incorpora el comportamiento definido para los errores y casos límite de §2.2.6»); RE3 y RE5 están
  en `04_objetivos.tex:163-164` y `04_objetivos.tex:178`.
- `09_metodologia.tex:285-286`: «LSC50 inicia y termina cada captura en la misma posición de reposo».
- Estado actual: como secuencia incompleta (`rechazada_incompleta`), el prototipo solo descarta por X («abortada») o por menos de `min_frames` fotogramas
  (`src/translator.py`, `_cerrar_ventana`); N-37 lo registró como hallazgo y el protocolo 4.a lo deja
  como CL-6b, riesgo R6 y decisión PD-17 (`docs/PROTOCOLO_EXPERIMENTAL_4a.md`).

## 2. Lo que dicen los datos (resumen)

1. **El inicio de los clips del corpus no está en reposo en la mayoría de los casos; el final sí.** Con
   K = 5, la velocidad mediana de las muñecas de la postura es 0,0132 anchos de hombro por fotograma en
   los primeros fotogramas y 0,0027 en los últimos (centro 0,0544) [corpus.global.5]. Juzgado con el
   criterio de reposo del FINAL (análisis post hoc, descriptivo), el inicio está «en movimiento» en el
   71,5 / 74,5 / 52,0 / 100,0 / 53,5 % de los clips de los señantes 0000 / 0001 / 0002 / 0003 / 0004
   [corpus.inicio_juzgado_con_reposo_del_final.por_K.5]. La altura mediana de la muñeca más alta sí es
   parecida al inicio y al final (1,46 y 1,46 anchos de hombro bajo la línea de los hombros, K = 5)
   [corpus.global.5]. Lectura: las muñecas están en la zona baja al empezar el clip, pero en la mayoría ya
   se mueven; «misma posición de reposo» (09:285-286) se cumple para la posición y no para la quietud.
   La figura `fig1_perfil_velocidad.png` lo muestra por señante; el 0003 empieza con la velocidad más alta.
2. **La regla elegida con los señantes de entrenamiento generaliza mal a un señante nuevo.** En 4 de 5
   pliegues el criterio eligió K = 3 y el rasgo `v_pose ∨ h`; en el pliegue 3, K = 8. Falsos positivos sobre
   clips ÍNTEGROS del señante de prueba: 2,5 / 9,5 / 4,0 / 75,5 / 22,5 % (media 22,8 %), frente al 5 %
   que fija el percentil 95 en los señantes vistos [pliegues[].regla_elegida.test.fp_integros;
   resumen_pliegues]. Con el percentil 99 los falsos positivos bajan a 0,5-10,0 %, pero la detección de un
   recorte del 20 % al inicio cae a 49,5-54,0 % [pliegues[].sensibilidad_percentil.99].
3. **Recortar el INICIO daña al modelo «manos»; recortar el FINAL casi no.** Top-1 sobre los 997 clips de
   prueba: íntegro 0,580; sin el 20 / 35 / 50 % inicial 0,554 / 0,410 / 0,200; sin el 20 / 35 / 50 % final
   0,570 / 0,573 / 0,527 [modelo.agregado]. El umbral de confianza vigente (0,20) NO protege: rechaza el
   24,9 % de los clips sin el 35 % inicial y el 34,6 % de los que no tienen la mitad inicial, frente al
   16,0 % de los íntegros [modelo.agregado].
4. **Dónde ayuda y dónde cuesta la regla** (regla elegida + umbral 0,20; U = aciertos − errores dichos en
   voz alta, la función de D-09 con c = 1) [modelo.combinado_agregado; modelo.post_hoc_por_borde]:
   - clips íntegros: la regla rechaza el 22,9 %; errores en voz alta 297 → 234, aciertos 540 → 401,
     U 243 → 167. En señantes nuevos, la regla cuesta más aciertos que errores evita;
   - sin el 35 % inicial: errores 383 → 29, aciertos 366 → 28 (U −17 → −1); sin la mitad inicial:
     errores 501 → 104, aciertos 151 → 26 (U −350 → −78). Aquí la regla evita un daño real;
   - sin el 35 % final: la regla rechaza el 96,5 %, errores 297 → 20 y aciertos 528 → 12 (U 231 → −8). Esos
     clips conservan casi todo su acierto, así que el borde final detecta «la mano aún no volvió a reposo»,
     no «la seña quedó incompleta»;
   - post hoc, solo con el borde inicial: rechaza el 15,3 % de los íntegros (U 243 → 199) y conserva casi
     todo el beneficio sobre los recortes del inicio (sin el 35 %: errores 383 → 33, aciertos 366 → 29).
5. **La altura depende de la postura de reposo de cada persona.** Mediana de h en los últimos 5
   fotogramas por señante: 1,92 / 1,84 / 1,39 / 1,33 / 1,40 [corpus.por_senante.5]; el umbral candidato de
   h al final es 1,261 [umbral_candidato_cinco_senantes]. Una persona que descanse las manos más arriba
   (p. ej. sobre una mesa) dispararía el borde final aunque haya terminado.
6. **El corpus no es una ventana del prototipo.** En el prototipo la ventana empieza al pulsar la tecla y
   dura 2,5 s por defecto (`LSCTranslator.__init__`, `ventana_seg`); los clips del corpus empiezan donde los
   cortó quien los grabó. Los umbrales del borde inicial aprendidos del corpus son laxos (1,761 anchos de
   hombro por segundo, frente a 0,229 al final) precisamente porque el corpus empieza en movimiento. Ni los
   falsos positivos ni la detección medidos aquí son los que tendría el prototipo; son la mejor aproximación
   disponible sin sesiones en vivo.

## 3. Opciones (ninguna implementada)

### A. Regla de movimiento/reposo en los bordes que RECHAZA → estado nuevo `rechazada_cortada`

- Dónde: función nueva que mide los bordes con los puntos de la ventana normalizados por hombros; se llama
  en `LSCTranslator._cerrar_ventana` después de las comprobaciones de fotogramas y de manos y antes de
  `_inferir_y_emitir`; si dispara, `ens.estado = "rechazada_cortada"` y no se infiere. `ESTADOS` gana un
  quinto estado, lo que cambia el diccionario `contadores` de TODOS los registros del JSONL; hay que
  tocar `realtime.py` (`MENSAJE_ESTADO`, `dibujar_ui`, `imprimir_contadores_sesion`) y
  `latencia_resumen.py` (`ESTADOS`, `_contar_estados`, `tabla`). Variante más barata: reutilizar
  `rechazada_incompleta` con un motivo distinto («borde inicial en movimiento»), que es literalmente la
  categoría de §2.2.6 y no cambia `ESTADOS`.
- Costo por ventana: calcular todos los rasgos de borde de una ventana de 75 fotogramas tarda 0,19 ms de
  mediana (p99 0,73 ms) en este equipo (cota superior; `carpeta_temporal/.../pieza4/costo_bordes.log`), frente
  a 12,3 ms de extracción por fotograma (N-32). Si se calcula antes de inferir, entra en la etapa
  «preprocesamiento» de §2.2.7.
- Pro: cumple a la letra «se descartarán» y RE3; el reporte distingue rechazo deliberado de error.
- Contra: con umbrales del corpus, falsos positivos del 2,5 al 75,5 % según la persona nueva (punto 2); el
  Bloque 1 tiene UNA persona nueva y no se puede saber de antemano en qué punto de ese rango caerá; el borde
  final rechaza ventanas con la seña completa (punto 4); depende de la postura de reposo (punto 5); hay que
  pasar los umbrales a segundos porque el bucle no corre a 24 fps (N-32 midió 30,4 fot./s).
- Costo de evaluación: los umbrales deberían re-estimarse en ventanas reales del prototipo antes del
  Bloque 1 (p. ej. con ejecuciones del Bloque 2), y fijarse en una D-nn antes de ver la exactitud del
  Bloque 1.
- Riesgo RE3: bajo (cumple). Riesgo RE5: ALTO: puede descartar una fracción grande e impredecible de
  ejecuciones válidas de la persona señante, lo que reduce las ejecuciones aceptadas del único bloque con
  exactitud y mezcla un artefacto del umbral con el desempeño.

### B. Protocolo de reposo al inicio y al final + la misma medida como AVISO (no rechaza)

- Protocolo: la persona está en reposo, con las manos en la posición acordada (PD-10), cuando se abre la
  ventana, y vuelve a ese reposo antes del cierre; quien pulsa la tecla lo hace con la persona ya en reposo
  (PD-10, opción B). Si la ejecución se sale de eso ANTES del cierre, se aborta con X (ya existe:
  `rechazada_incompleta`, CL-6a).
- Prototipo: la medida de bordes de A se calcula en cada ventana cerrada y se escribe en el JSONL con dos
  banderas (`aviso_inicio`, `aviso_fin`) y un aviso en pantalla, sin cambiar el estado ni impedir la
  inferencia.
- Pro: no pierde ejecuciones válidas; controla el caso dañino (inicio recortado) por la vía más barata,
  porque en el prototipo el inicio lo marca la tecla; deja datos de ventanas reales para calibrar A más
  adelante; el Bloque 2 puede ejercitar CL-6b y reportar CUÁNTAS ejecuciones provocadas dispararon el aviso
  (conteo, permitido por 09:259-262); el costo de código es el de A menos el estado nuevo.
- Contra: «se descartarán» queda cubierto por el protocolo y por X, no automáticamente para la ventana que
  se cierra con la seña a medias; hay que declararlo como desviación acotada en RE3. Un aviso que no decide
  nada puede ignorarse: el uso del aviso en el análisis del Bloque 1 tiene que fijarse antes de la sesión
  (p. ej. exactitud sobre todas las ejecuciones como cifra principal y, como secundaria declarada de
  antemano, sin las que tienen aviso).
- Costo de evaluación: bajo; en el Bloque 2, contar avisos en ejecuciones normales y en CL-6b provocados.
- Riesgo RE3: medio (desviación declarada, mitigada por X y por el aviso). Riesgo RE5: bajo.

### C. Declararlo como límite del prototipo y controlarlo solo por protocolo

- Pro: costo cero de código y de pruebas.
- Contra: RE3 (10_resultados.tex:28-30) promete incorporar el comportamiento de §2.2.6 y el Bloque 2 debe
  ejercitar la secuencia incompleta (09:247-251); CL-6b quedaría como «no detectado». Sin medida en el
  JSONL no hay forma de saber después cuántas ejecuciones del Bloque 1 empezaron fuera de la ventana, y el
  punto 3 muestra que esas son justamente las que el modelo falla sin que el umbral 0,20 las rechace.
- Riesgo RE3: alto (incumplimiento declarado). Riesgo RE5: medio (errores no separables de recortes).

## 4. Recomendación: B

1. El daño medible está en el borde INICIAL (top-1 0,580 → 0,410 con el 35 % inicial recortado, y el
   umbral 0,20 solo rechaza el 24,9 %), y en el prototipo ese borde lo fija la tecla: un protocolo de
   reposo antes de pulsar lo controla en origen, sin riesgo de rechazar ejecuciones válidas.
2. Una regla que RECHAZA, con umbrales del corpus, tiene falsos positivos de 2,5 a 75,5 % en una persona no
   vista (media 22,8 %) y, sobre los clips íntegros, baja U de 243 a 167: en el Bloque 1 costaría más
   aciertos de los errores que evita, y de forma impredecible.
3. El borde final, tal como lo mide la regla, marca «la mano no volvió a reposo» y no «la seña quedó
   incompleta» (96,5 % de rechazo en clips que conservan un top-1 de 0,573). Rechazar por él sería
   perjudicial; como aviso sí informa.
4. B conserva la medida en el JSONL, que es lo que falta para decidir A con datos del prototipo y no del
   corpus. Pasar de B a A más adelante es cambiar el efecto de una bandera, no reescribir.
5. Lo que B NO resuelve y hay que declarar en RE3: el descarte automático de una ventana que se cierra con la
   seña a medias. La mitigación es el protocolo (reposo al final, X antes del cierre) y el aviso registrado.

Si el director prefiere cumplir §2.2.6 a la letra, la alternativa menos dañina según estos datos es A
SOLO con el borde inicial (post hoc: rechaza el 15,3 % de los íntegros, U 243 → 199) reutilizando
`rechazada_incompleta`, con umbrales re-estimados en ventanas del Bloque 2 y fijados en una D-nn antes del
Bloque 1. No se recomienda A con los dos bordes.

## 5. Especificación de la implementación recomendada (para que otra pieza la programe)

**Archivos**: `src/bordes_ventana.py` (nuevo, sin TensorFlow ni MediaPipe); cambios en `src/translator.py` y
`src/realtime.py`; opcional `src/latencia_resumen.py`; pruebas nuevas en `tests/test_bordes_ventana.py` y
casos añadidos en `tests/test_prototipo.py`.

**Umbrales**: se leen de un JSON con el formato de `models/senas_cortadas_2026-09-22/umbral_bordes_candidato.json`
(CANDIDATO; no se copia a `models/canonico*` sin decisión). Valores actuales: `borde_s` 0,125;
`v_pose_por_s` ini 1,7614 / fin 0,2295 anchos de hombro por segundo; `h` ini 0,9740 / fin 1,2610.

**`src/bordes_ventana.py`**
- `cargar_umbrales_bordes(ruta) -> dict`: valida que existan `borde_s`, `v_pose_por_s.{ini,fin}`,
  `h.{ini,fin}`; error claro si falta algo.
- `medir_bordes(pose_xy_izq, pose_xy_der, pose_presente, c_ns, borde_s) -> dict`:
  entradas: coordenadas xy de las muñecas 15 y 16 de la postura YA normalizadas por hombros
  (`normalize_sequence` de `translator.py`, columnas 45:47 y 48:50), bandera de postura por fotograma y las
  marcas de llegada `c_i` (ns) de la ventana. `fps_ventana = (N − 1) / ((c_N − c_1)/1e9)`;
  `K = max(3, round(borde_s · fps_ventana))`, y `K = min(K, N)`. Velocidad por par consecutivo:
  media de ‖Δxy‖ de las dos muñecas dividida entre `(c_{i+1} − c_i)/1e9` (anchos de hombro por segundo);
  un par sin postura en alguno de los dos fotogramas no cuenta. `v_pose_ini_por_s` = media de los pares
  dentro de los primeros K fotogramas; `v_pose_fin_por_s`, de los últimos K. `h_ini` / `h_fin` = media sobre
  los K fotogramas del borde de min(y_izq, y_der), solo fotogramas con postura. Sin datos → `None`.
  Devuelve también `K`, `fps_ventana` y el número de pares válidos por borde.
- `evaluar_bordes(medidas, umbrales) -> dict`: `aviso_inicio = v_pose_ini_por_s > v.ini or h_ini < h.ini`,
  `aviso_fin` análogo con `fin`; un rasgo `None` no dispara; si los dos rasgos de un borde son `None`, la
  bandera es `None` (no se pudo medir), no `False`.

**`src/translator.py`**
- `LSCTranslator.__init__(..., bordes: str = "aviso", umbral_bordes: str | Path | None = None)`; `bordes` ∈
  {"off", "aviso"}; con "aviso" y sin ruta, se usa el candidato de esta carpeta y se avisa en consola de que
  es candidato. No se añade ningún estado a `ESTADOS`.
- Cálculo: en `_cerrar_ventana`, DESPUÉS de fijar el estado (y, si hubo inferencia, después de t5), y antes
  de `_registro_ensayo`; así ninguna marca t0..t5 ni etapa de §2.2.7 cambia. Se calcula para todo cierre
  con N ≥ 2, también en rechazos, porque también informa en ellos; con cierre «abortada» se registra `null`.
- JSONL: clave nueva `"bordes"` al final del registro de ensayo:
  `{"modo": "aviso"|"off", "umbrales": <ruta relativa>, "K": int, "fps_ventana": float,
  "v_pose_ini_por_s": float|null, "v_pose_fin_por_s": float|null, "h_ini": float|null, "h_fin": float|null,
  "aviso_inicio": bool|null, "aviso_fin": bool|null}`; con "off", `{"modo": "off"}`. Las claves previas del
  registro no cambian ni de nombre ni de orden.
- Contador en memoria `avisos_bordes = {"inicio": 0, "fin": 0}` (no dentro de `contadores`).

**`src/realtime.py`**
- Opciones `--bordes {off,aviso}` (por defecto `aviso`) y `--umbral-bordes RUTA`.
- `dibujar_ui`: tras el estado de un ensayo con aviso, una línea «AVISO: la seña pudo empezar antes de la
  ventana» / «... no terminar dentro de la ventana»; `_resumen_consola`: la bandera; `imprimir_contadores_sesion`:
  los dos contadores de aviso.

**Pruebas mínimas** (sintéticas, sin cámara): ventana con reposo en los bordes → sin avisos; ventana que
empieza a media seña → `aviso_inicio`; que acaba a media seña → `aviso_fin`; la velocidad usa `c_i`
(misma trayectoria a 15 y a 30 fot./s da la misma velocidad por segundo); sin postura en el borde → `None`;
con reloj falso, t0..t5, etapas y las claves previas del JSONL idénticos con y sin la medida; `off` no
calcula nada; estado y confianza no cambian por el aviso.

**Protocolo** (`docs/PROTOCOLO_EXPERIMENTAL_4a.md`, a decidir con el director): cerrar PD-17/CL-6b con B;
en PD-10, fijar la posición de reposo (replicar la del corpus) y que la tecla se pulse con la persona en
reposo; fijar antes del Bloque 1 cómo se reportan las ejecuciones con aviso.

## 6. Límites de este análisis

- Recortar un clip del corpus es una aproximación a una ventana del prototipo mal sincronizada, no una
  medición en vivo; la ventana real tiene 2,5 s fijos y reposo antes de la tecla.
- Cinco señantes; los falsos positivos por persona no vista varían de 2,5 % a 75,5 %: con cinco valores no
  se puede estimar esa distribución.
- La conversión a segundos supone que la velocidad por segundo no depende de la tasa de captura; el ruido
  del extractor por fotograma sí puede depender de ella y no se midió.
- Las lecturas del punto 4 que usan un solo borde son post hoc y descriptivas.
- Los clips de LSC50 no se grabaron con la cámara ni en las condiciones de las sesiones en vivo (encuadre,
  distancia, iluminación, persona). La normalización por hombros compensa la escala y la posición, pero no
  necesariamente el resto, y el efecto de esa diferencia sobre los umbrales no se midió. (Límite añadido por el verificador independiente
  el 2026-09-22.)
