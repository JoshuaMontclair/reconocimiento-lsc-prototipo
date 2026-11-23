# Umbral de confianza con validación anidada por señante: experimento (pieza 1, 22-sep-2026)

**Qué es este archivo.** El informe del experimento que N-46 propuso al director
(`docs/PROPUESTA_UMBRAL_ANIDADO.md`) y que el director pidió correr para decidir con datos. Se
escribe en dos tiempos: primero el criterio (esta sección, con hora, ANTES de lanzar nada) y después
los resultados. Ninguna cifra de resultados existe todavía al cerrar la sección 1.

**Qué NO hace.** No toca el umbral vigente (`models/canonico_holistic_mascara_manos/modelo_canonico.umbral.json`,
0,20, D-09/N-33), ni el modelo canónico, ni los 5 modelos LOSO de N-30, ni `data/partitions/`. Todo lo
nuevo queda en `models/umbral_anidado_2026-09-22/`, `src/umbral_anidado.py` (nuevo) y
`tests/test_umbral_anidado.py` (nuevo). Lo que sale de aquí es un **candidato**; la adopción la decide
el estudiante con el director.

---

## 1. Criterio fijado antes de correr

**Fecha y hora de cierre de esta sección: 2026-09-22 16:15 (hora local, -05).** Último commit del
repositorio al escribirla: `00a798d`. El script y las pruebas se escriben después de esta sección y
la corrida se lanza después de ambos; el verificador puede comprobar el orden por las horas de
modificación y por la cabecera de `corrida.log`.

### 1.1 Qué dice el anteproyecto

`09_metodologia.tex:184-189` (§2.2.6): «Cuando la confianza de la predicción sea inferior a un valor
de corte, el sistema no emitirá audio y devolverá un estado de “seña no reconocida” […] Ese valor de
corte es un parámetro de operación del prototipo, no un criterio de éxito del proyecto: no se fija en
este anteproyecto, sino que se determinará empíricamente sobre el conjunto de validación y se
reportará junto con el efecto que produce sobre la tasa de rechazo y sobre las métricas de
reconocimiento.» El experimento sigue cumpliendo eso: el umbral se elige sobre validación, nunca sobre
la prueba LOSO ni sobre el canónico; lo que cambia respecto de D-09 es **cómo se construye la
validación** (con una persona que el modelo no ha visto), como propone §4 de la propuesta.

### 1.2 Método exacto

1. **Modelos internos, uno por PAR de señantes.** Señantes `0000..0004` (índices 0..4). Para cada par
   `{a, b}` con `a < b` (C(5,2) = 10 pares) se entrena UN modelo con los otros tres señantes. De esos
   tres se apartan exactamente 2 muestras por seña (100) como validación de época (checkpoint y
   early stopping sobre `val_accuracy`), igual que en D-06; el resto (≈398-400) entrena. Las tres
   muestras duplicadas de D-06 quedan fuera de todo. El señante interno NO sirve de monitor de épocas
   (propuesta §2.2): se usa solo para el umbral.
   - Semilla de partición: 2026, generador `numpy.random.default_rng(2026*100 + 10*a + b)` por par
     (202601 … 202634), distinta de las del LOSO (20260…20264) y fijada aquí antes de correr.
   - Entrenamiento: `train_loso.entrenar_y_evaluar` importado tal cual (misma `build_model` de
     `train_final_100`, misma aumentación ×10 con el espejado de «manos», `add_velocity`,
     `resample_sequence` a 30 pasos, CosineDecay 0,001, label smoothing 0,1, batch 64, paciencia 25,
     épocas máximas 200, dropout 0,3, semilla de entrenamiento 2026, monitor `val_accuracy`), sobre
     `data/processed_holistic_mascara_rostro` con `--representacion manos` (el processed y la
     representación del canónico, `modelo_canonico.procedencia.json`). El modelo es el de N-30 salvo
     por los datos. En esa función el conjunto llamado `test` es aquí la unión de los dos señantes
     excluidos: **no es una prueba**, es el conjunto de calibración; sus `metricas.json` se guardan
     porque la función los escribe, y se leen como «exactitud sobre dos personas no vistas con un
     modelo de 3 señantes».
   - Cada par guarda además `predicciones_<a>.json` y `predicciones_<b>.json` con el softmax completo
     (50 probabilidades por muestra), la clase real, la predicha y la confianza (máximo del softmax,
     la misma que usa el prototipo). Control: la confianza debe coincidir con `prob_predicha` del
     `predicciones.json` que escribe `entrenar_y_evaluar`.
2. **Casillas.** Casilla `(k, j)` = pliegue externo `k` (señante de prueba `s_k`, como en
   `data/partitions/loso_5_pliegues.json`) y señante interno `j ≠ k`. Sus predicciones son las del
   modelo del par `{k, j}` sobre TODAS las muestras del señante `j` (198-200). 20 casillas, 10 modelos.
   Por construcción, ni `s_k` ni `s_j` están en el entrenamiento ni en la validación de época del
   modelo que predice la casilla; `tests/test_umbral_anidado.py` lo comprueba con dobles.
3. **Selector.** El de D-09, importado de `src/umbral_confianza.py` (`elegir_umbral`, `utilidad`,
   `efecto`, `REJILLA`), sin reimplementar: `U(t) = aciertos aceptados − c·errores aceptados`, aceptada
   si confianza ≥ t, `c = 1`, rejilla 0,00 … 0,99, con empate el menor `t`.
4. **Agregación.** El encargo de esta pieza trae una regla (mediana de los 4 umbrales de casilla por
   pliegue y mediana de los 5 umbrales de pliegue) y dice que, si la propuesta fija algo distinto, se
   sigue la propuesta. **La propuesta fija algo distinto** (§2.1 y §2.2 de
   `docs/PROPUESTA_UMBRAL_ANIDADO.md`): `t_k` se elige sobre las predicciones **juntas** de los 4
   señantes internos del pliegue `k` (≈797), y el umbral final sobre las predicciones **juntas** de las
   20 casillas (≈3.988). Como eso es lo que N-46 llevó al director, se adopta como regla PRINCIPAL:
   - `t_k` = `elegir_umbral` sobre la unión de las 4 casillas del pliegue `k` (c = 1).
   - `t_global` = `elegir_umbral` sobre la unión de las 20 casillas (c = 1). **Este es el candidato**
     que va en `candidato_umbral.json`.
   - La regla del encargo se calcula y se reporta como comprobación de robustez, NO como candidato:
     `t_(k,j)` por casilla, `t_k^med` = mediana de las 4 `t_(k,j)`, `t_global^med` = mediana de los 5
     `t_k^med`; y también la mediana de los 5 `t_k` principales. Si las dos reglas difieren se dice
     cuánto y no se cambia el candidato por ello.
   - La dispersión de los cinco `t_k` es la incertidumbre del umbral y se reporta siempre.
5. **Sensibilidad.** `c = 0,5` y `c = 2` para `t_k`, `t_global` y las medianas; se reporta y no cambia
   la elección.
6. **Efecto sobre la prueba LOSO original** (las 997 predicciones ya guardadas en
   `models/loso_2026-09-15_holistic_mascara_manos/pliegue_k/predicciones.json`; los 5 modelos
   externos NO se reentrenan). Para cada pliegue `k` y para el agregado de las 997:
   - con `t_k` (el estimado **limpio**: el señante `s_k` no intervino en nada de lo que eligió `t_k`);
   - con `t_global` (etiquetado **ligeramente optimista**: cada señante aparece como señante interno
     en los otros cuatro pliegues, ≈20 % de las predicciones que eligieron `t_global`);
   - con 0,20 (vigente) y con 0,50 (referencia), que deben reproducir EXACTAMENTE la tabla de N-33
     (0,20: rechazo 16,0 %, exactitud entre aceptadas 0,645, 540/297, U = 243; 0,50: 45,9 %, 0,761,
     410/129, U = 281). Si no la reproducen, el informe se detiene ahí.
   Métricas: tasa de rechazo, exactitud entre aceptadas, aciertos perdidos por rechazo, aceptadas
   ok/mal y `U(c = 1)`, con `efecto` y `utilidad` de `src/umbral_confianza.py`. El agregado con
   umbrales por pliegue suma las decisiones de cada pliegue con su propio `t_k`.
   Como reporte adicional (propuesta §2.2 c): el efecto de `t_global` sobre la prueba del canónico
   (100 predicciones guardadas en `evaluacion_modelo_seleccionado/predicciones.json`) y sobre su
   validación (leída de las curvas de `models/umbral_D09_2026-09-15.json`, que ya están calculadas
   para toda la rejilla; no se vuelve a inferir).
7. **Control de transferencia (propuesta §4).** Por señante: mediana de confianza y exactitud de los
   modelos internos (3 señantes) sobre esa persona no vista, frente a las del modelo externo (4
   señantes) sobre la misma persona como señante de prueba. Si las escalas de confianza se parecen,
   aplicar el umbral anidado al canónico (5 señantes) es un supuesto razonable; si no, se declara.
8. **Comparación con lo previo.** Con la tabla de N-33 (arriba) y con la simulación de N-46
   (mediana 0,47; rechazo 44,5 %; exactitud entre aceptadas 0,747; 413/140; U = 273), que usaba las
   predicciones de prueba de modelos de 4 señantes y por eso no era el método.
9. **Reglas de la corrida.** 10 entrenamientos seguidos en un proceso con `nohup`, log en
   `models/umbral_anidado_2026-09-22/corrida.log`; si un par falla se repite una vez y queda escrito;
   ningún archivo vigente se sobrescribe; el candidato se escribe en
   `models/umbral_anidado_2026-09-22/candidato_umbral.json` con el formato del `.umbral.json` vigente
   más el método, los umbrales por casilla y por pliegue y el sha256 del canónico
   (`55e170fc6935be3c23315f5f503849e27fa8da8252f5e72e1d0366187c826fc1`, medido hoy con `shasum`).
   Tiempo esperado según la propuesta §3: ≈32 min (rango 24-39; peor caso 1,9 h). Se registra el real.

### 1.3 Qué decidirá la recomendación (fijado antes de ver ningún resultado)

Se comparan dos **procedimientos** fijados de antemano, no dos números elegidos mirando la prueba:
(P0) el umbral vigente 0,20 de D-09 y (P1) el umbral anidado aplicado pliegue a pliegue (`t_k` sobre
la prueba de su propio pliegue, el estimado limpio). Sobre las mismas 997 predicciones:

- Se recomendará **adoptar** el umbral anidado si (i) el control de N-33 se reproduce, (ii) `U(c = 1)`
  agregado de P1 es mayor o igual que el de P0 (243) y (iii) las aceptadas incorrectas de P1 son menos
  que las de P0 (297). La condición (ii) es la que manda: el criterio del anteproyecto es U.
- Se recomendará **quedarse con 0,20** si (ii) falla, aunque (iii) se cumpla: bajar errores a costa
  de más aciertos perdidos de los que compensa no es lo que pide el criterio, y la evaluación en vivo
  se haría con un umbral peor según U.
- Si `U` queda igual o casi (±5 sobre 997), se dirá que los datos no distinguen y se recomendará por
  el argumento de método (validación con personas no vistas) y por el costo en rechazo para el
  protocolo 4.a, que se cuantifica.
- En cualquier caso se reportan los cinco `t_k` y su dispersión, y el rechazo esperado en vivo se
  discute con N-26/N-32 (con 0,5 se midió 62,7 % y 77,8 %).

Nada de lo anterior toca el `.umbral.json` vigente: la adopción es una decisión nueva del director.

---

## 2. Resultados

**Escrito el 2026-09-22 a partir de las 16:45**, después de que terminara la corrida (16:44:31). La
sección 1 no se modificó: su copia tal como estaba antes de los resultados quedó en la carpeta de
trabajo del flujo (`…/carpeta_temporal/wf_2026-09-22/pieza1_umbral_anidado/informe_seccion1_antes_de_resultados.md`,
sha256 `f5bddcc045a4d7d7…`, hora de modificación 16:16:33). Todas las cifras de esta sección salen de
`resultados_anidado.json`, `resultados_entrenamiento.json` y `corrida.log` de esta carpeta, salvo las
marcadas «cálculo complementario», que salen de dos guiones de la carpeta de trabajo (`extra.py`,
`bootstrap.py`) que solo leen esos archivos y las 997 predicciones guardadas.

**Nota de procedencia.** El flujo se relanzó a las ≈16:28. La corrida la había lanzado el primer
intento de esta pieza a las 16:24:14 (proceso único, `src/umbral_anidado.py` sin argumentos = construir,
entrenar y evaluar seguidos; cabecera de `corrida.log`). El relanzamiento la encontró en marcha
(4.º par), **no lanzó una segunda** (habría duplicado cómputo y escrito sobre las mismas carpetas),
revisó el código y las pruebas, esperó a que terminara y redactó esta sección. No hubo, por tanto,
ningún entrenamiento repetido ni ningún resultado de una corrida anterior descartado.

### 2.1 Corrida

- Comando (el de la regla de cómputo, desde la raíz del repositorio):
  `nohup python src/umbral_anidado.py > models/umbral_anidado_2026-09-22/corrida.log 2>&1 &`
  (el proceso visible era `Python src/umbral_anidado.py`, PID 14077, inicio 16:24:14; la línea exacta
  del `nohup` la escribió el primer intento y no la vi, pero el ejecutable, el guion y la ausencia de
  argumentos se leen del proceso y de la cabecera del log).
- **Tiempo real: 1.214,3 s (20,2 min) para los 10 entrenamientos** (16:24:16 → 16:44:31), frente a los
  ≈32 min estimados en la propuesta §3 (rango 24–39). Por par: de 70,7 s a 223,4 s.
- **Fallos: ninguno.** Los 10 pares terminaron en el primer intento (`intentos: 1`, `fallos: []`).
- Control del softmax guardado: en los 20 archivos `predicciones_<s>.json` la confianza difiere de
  `prob_predicha` del `predicciones.json` de `entrenar_y_evaluar` en ≤ 3,6·10⁻⁷ y la clase predicha
  coincide en todas las muestras.
- Particiones: `particiones_anidadas.json` (10 pares, 20 casillas, 997 muestras; verificadas sin fuga
  por el propio guion antes de entrenar).

### 2.2 Los 10 modelos internos (3 señantes cada uno)

| par | entrena con | train (orig.) | épocas (mejor) | val. de época | exactitud sobre a | exactitud sobre b | s |
|---|---|---:|---:|---:|---:|---:|---:|
| 0000·0001 | 0002, 0003, 0004 | 498 | 63 (38) | 0,93 | 0000: 0,300 | 0001: 0,407 | 119 |
| 0000·0002 | 0001, 0003, 0004 | 499 | 61 (36) | 0,95 | 0000: 0,315 | 0002: 0,561 | 127 |
| 0000·0003 | 0001, 0002, 0004 | 497 | 35 (10) | 0,93 | 0000: 0,285 | 0003: 0,550 | 71 |
| 0000·0004 | 0001, 0002, 0003 | 497 | 35 (10) | 0,90 | 0000: 0,310 | 0004: 0,520 | 72 |
| 0001·0002 | 0000, 0003, 0004 | 500 | 36 (11) | 0,92 | 0001: 0,447 | 0002: 0,480 | 75 |
| 0001·0003 | 0000, 0002, 0004 | 498 | 38 (13) | 0,90 | 0001: 0,447 | 0003: 0,505 | 79 |
| 0001·0004 | 0000, 0002, 0003 | 498 | 67 (42) | 0,88 | 0001: 0,513 | 0004: 0,515 | 140 |
| 0002·0003 | 0000, 0001, 0004 | 499 | 79 (54) | 0,90 | 0002: 0,636 | 0003: 0,570 | 160 |
| 0002·0004 | 0000, 0001, 0003 | 499 | 70 (45) | 0,90 | 0002: 0,444 | 0004: 0,490 | 142 |
| 0003·0004 | 0000, 0001, 0002 | 497 | 112 (87) | 0,94 | 0003: 0,565 | 0004: 0,625 | 223 |

Exactitud media de los modelos internos sobre personas no vistas: **0,474** (3.988 predicciones),
frente a **0,580** de los 5 modelos externos de 4 señantes sobre su señante de prueba (997). Cuatro
pares pararon con la mejor época entre la 10 y la 13: la validación de época (personas vistas) toca
techo pronto; es la misma regla de N-30 y no se tocó.

### 2.3 Umbrales por casilla, por pliegue y global (selector de D-09)

Casillas `(k, j)`: pliegue externo `k`, señante interno `j`; `t` elegido sobre las ≈200 predicciones
del modelo del par `{s_k, s_j}` sobre `s_j`.

| pliegue k (prueba) | j = 0000 | j = 0001 | j = 0002 | j = 0003 | j = 0004 | **t_k (unión, c = 1)** | mediana de las 4 casillas |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (0000) | — | 0,50 | 0,29 | 0,50 | 0,60 | **0,50** | 0,50 |
| 1 (0001) | 0,49 | — | 0,64 | 0,76 | 0,77 | **0,66** | 0,70 |
| 2 (0002) | 0,35 | 0,66 | — | 0,34 | 0,67 | **0,67** | 0,51 |
| 3 (0003) | 0,51 | 0,58 | 0,19 | — | 0,54 | **0,58** | 0,53 |
| 4 (0004) | 0,42 | 0,56 | 0,44 | 0,68 | — | **0,48** | 0,50 |

- **Umbral global (unión de las 20 casillas, 3.988 predicciones, c = 1): t = 0,55. Es el candidato**
  (regla principal fijada en §1.2, punto 4). En la selección: rechazo 57,2 %, exactitud entre
  aceptadas 0,742, 1.265 / 441, U = 824.
- Dispersión de los cinco `t_k`: **0,48 – 0,67** (0,50; 0,66; 0,67; 0,58; 0,48). Por casilla, de 0,19 a
  0,77: una sola persona de umbral da un valor muy inestable, como anticipó la propuesta (§7, variante B).
- Comprobación de robustez (regla del encargo, no es el candidato): mediana de las medianas de
  casilla = **0,51**; mediana de los cinco `t_k` = **0,58**. Las tres agregaciones quedan entre 0,51 y
  0,58, a ±0,04 del candidato; ninguna se acerca a 0,20.

### 2.4 Efecto sobre la prueba LOSO original (997 predicciones ya guardadas; los 5 modelos externos no se reentrenaron)

Control: con 0,20 y 0,50 se reproducen exactamente las cifras de N-33 (540 / 297, U = 243; 410 / 129,
U = 281; `control_N33.coincide = true` en ambos). Aceptada si confianza ≥ t.

**Agregado sobre las 997:**

| procedimiento | umbral(es) | rechazo | exactitud entre aceptadas | aciertos perdidos por rechazo | aceptadas ok / mal | U (c = 1) |
|---|---|---:|---:|---:|---:|---:|
| **P1: anidado por pliegue, `t_k` (estimado limpio)** | 0,50 / 0,66 / 0,67 / 0,58 / 0,48 | **52,3 %** | **0,782** | 35,6 % | **372 / 104** | **268** |
| anidado global 0,55 (candidato; ligeramente optimista) | 0,55 | 49,6 % | 0,787 | 31,7 % | 395 / 107 | 288 |
| mediana de casillas por pliegue (robustez) | 0,50 / 0,70 / 0,51 / 0,53 / 0,50 | 48,9 % | 0,770 | 32,2 % | 392 / 117 | 275 |
| mediana de medianas 0,51 (regla del encargo; cálculo complementario) | 0,51 | 46,6 % | 0,765 | 29,6 % | 407 / 125 | 282 |
| mediana de `t_k` 0,58 (cálculo complementario) | 0,58 | 51,7 % | 0,795 | 33,7 % | 383 / 99 | 284 |
| **P0: vigente 0,20 (D-09)** | 0,20 | **16,0 %** | **0,645** | 6,6 % | **540 / 297** | **243** |
| referencia 0,50 | 0,50 | 45,9 % | 0,761 | 29,1 % | 410 / 129 | 281 |

**Por pliegue** (rechazo · exactitud entre aceptadas · ok/mal · U):

| pliegue (prueba) | P1 `t_k` | 0,55 (candidato) | 0,20 (vigente) | 0,50 |
|---|---|---|---|---|
| 0 (0000), n = 200 | 0,50: 72,0 % · 0,786 · 44/12 · 32 | 75,0 % · 0,880 · 44/6 · 38 | 29,0 % · 0,563 · 80/62 · 18 | 72,0 % · 0,786 · 44/12 · 32 |
| 1 (0001), n = 199 | 0,66: 51,3 % · 0,691 · 67/30 · 37 | 45,2 % · 0,670 · 73/36 · 37 | 14,6 % · 0,594 · 101/69 · 32 | 41,7 % · 0,664 · 77/39 · 38 |
| 2 (0002), n = 198 | 0,67: 66,7 % · 0,864 · 57/9 · 48 | 52,0 % · 0,821 · 78/17 · 61 | 15,2 % · 0,690 · 116/52 · 64 | 48,5 % · 0,794 · 81/21 · 60 |
| 3 (0003), n = 200 | 0,58: 45,0 % · 0,791 · 87/23 · 64 | 43,0 % · 0,781 · 89/25 · 64 | 10,5 % · 0,654 · 117/62 · 55 | 40,0 % · 0,775 · 93/27 · 66 |
| 4 (0004), n = 200 | 0,48: 26,5 % · 0,796 · 117/30 · 87 | 33,0 % · 0,828 · 111/23 · 88 | 11,0 % · 0,708 · 126/52 · 74 | 27,5 % · 0,793 · 115/30 · 85 |

Lectura: frente al 0,20, el procedimiento limpio P1 dice en voz alta **193 señas equivocadas menos**
(297 → 104; −65 %) a cambio de **168 aciertos menos** (540 → 372; −31 %), y triplica el rechazo
(16,0 % → 52,3 %). U sube 25 (243 → 268). P1 gana en U en 4 de los 5 pliegues (diferencias +14, +5,
−16, +9, +13); pierde en el pliegue 2 (señante 0002), el mismo en el que la simulación de N-46 ya
había visto que un umbral alto no convenía.

**Incertidumbre de la diferencia en U (cálculo complementario, añadido después de ver los resultados
y que no cambia la regla de §1.3):** bootstrap pareado sobre las 997 muestras, 10.000 remuestreos,
semilla 2026: diferencia U(P1) − U(P0) = +25, intervalo del 95 % **[−12, +62]**; en el 9,8 % de los
remuestreos la diferencia es ≤ 0. Es decir: la reducción de errores dichos en voz alta es grande y
clara; la ganancia neta en U es positiva en el punto pero **no es concluyente** con 997 muestras de 5
personas.

### 2.5 Comparación con N-33 y con la simulación de N-46

| | umbral(es) | rechazo | exactitud entre aceptadas | ok / mal | U |
|---|---|---:|---:|---:|---:|
| N-33, vigente | 0,20 | 16,0 % | 0,645 | 540 / 297 | 243 |
| N-46, simulación (prueba de modelos de 4 señantes; no era el método) | 0,47 / 0,55 / 0,47 / 0,55 / 0,35 (mediana 0,47) | 44,5 % | 0,747 | 413 / 140 | 273 |
| **este experimento, P1 limpio** | 0,50 / 0,66 / 0,67 / 0,58 / 0,48 | **52,3 %** | **0,782** | **372 / 104** | **268** |
| este experimento, candidato global | 0,55 | 49,6 % | 0,787 | 395 / 107 | 288 |

- La simulación acertó el orden de magnitud y la dirección: el umbral para personas no vistas está
  en torno a 0,5, no en 0,20. Los `t_k` reales son **más altos que los simulados en los cinco
  pliegues** (+0,03, +0,11, +0,20, +0,03, +0,13), lo que es coherente con que los modelos internos de
  3 señantes son menos exactos (0,474) que los de 4 (0,580): con más errores entre las confianzas
  medias, el criterio pide un umbral más alto.
- Frente a la simulación, P1 rechaza más (52,3 % frente a 44,5 %), acierta más entre lo aceptado
  (0,782 frente a 0,747) y queda 5 puntos por debajo en U (268 frente a 273).
- El 0,50 «a ojo» (U = 281) queda por encima de P1 (268) sobre estas 997: ese contraste no es un
  argumento para elegir 0,50, porque se lee mirando la prueba (propuesta §5.3).

### 2.6 Sensibilidad a c (no cambia la elección)

| c | `t_k` por pliegue | `t` global | mediana de `t_k` | mediana de medianas | efecto en la prueba con `t_k`: rechazo · exactitud acept. · ok/mal · U(c=1) | con `t` global: rechazo · exactitud acept. · ok/mal · U(c=1) |
|---|---|---:|---:|---:|---|---|
| 0,5 | 0,24 / 0,41 / 0,22 / 0,43 / 0,38 | 0,39 | 0,38 | 0,36 | 28,2 % · 0,687 · 492/224 · 268 | 36,0 % · 0,718 · 458/180 · 278 |
| **1** | 0,50 / 0,66 / 0,67 / 0,58 / 0,48 | **0,55** | 0,58 | 0,51 | 52,3 % · 0,782 · 372/104 · 268 | 49,6 % · 0,787 · 395/107 · 288 |
| 2 | 0,64 / 0,77 / 0,78 / 0,58 / 0,67 | 0,71 | 0,67 | 0,71 | 61,5 % · 0,812 · 312/72 · 240 | 63,1 % · 0,826 · 304/64 · 240 |

Con cualquier `c` del rango, el umbral anidado queda muy por encima del 0,20 (en D-09, sobre
validación de señantes vistos, `c = 0,5` daba 0,11 y `c = 2` daba 0,20). La elección de `c` sigue
siendo una decisión de diseño del director (propuesta §7, pregunta 2). Cálculo complementario: si se
midiera con el costo de cada `c` en vez de con `c = 1`, con los `t_k` de `c = 1` el 0,20 ganaría con
`c = 0,5` (U = 391,5 frente a 320) y perdería por mucho con `c = 2` (−54 frente a 164).

### 2.7 Control de transferencia (propuesta §4)

| señante | modelos internos (3 señantes): n · exactitud · confianza mediana | modelo externo (4 señantes): n · exactitud · confianza mediana |
|---|---|---|
| 0000 | 800 · 0,303 · 0,230 | 200 · 0,460 · 0,332 |
| 0001 | 796 · 0,454 · 0,541 | 199 · 0,533 · 0,640 |
| 0002 | 792 · 0,530 · 0,440 | 198 · 0,606 · 0,513 |
| 0003 | 800 · 0,548 · 0,585 | 200 · 0,615 · 0,652 |
| 0004 | 800 · 0,538 · 0,596 | 200 · 0,685 · 0,747 |
| global | 3.988 · 0,474 · 0,460 | 997 · 0,580 · 0,555 |

**Las escalas no coinciden:** para las cinco personas, los modelos de 3 señantes son menos exactos y
menos confiados que el de 4 (mediana de confianza −0,095 en global). Por tanto el supuesto de la
propuesta («si se parecen, la transferencia se sostiene») **no se cumple limpiamente** y hay que
declararlo. Lo que sí está medido es la transferencia 3 → 4 señantes: los `t_k` elegidos con modelos
de 3 señantes, aplicados a los modelos externos de 4, bajan los errores aceptados de 297 a 104 y suben
U de 243 a 268 (§2.4). La transferencia 4 → 5 (canónico) no se puede medir con personas no vistas: el
canónico no tiene prueba de señante nuevo.

**Efecto del candidato 0,55 sobre el canónico** (personas vistas; solo como reporte):
prueba del canónico (100): rechazo 11,0 %, exactitud entre aceptadas 0,955, 85 / 4, U = 81 (con 0,20:
1,0 %, 0,889, 88 / 11, U = 77, N-33). Validación del canónico (100, leída de las curvas de
`models/umbral_D09_2026-09-15.json`): rechazo 14,0 %, exactitud entre aceptadas 0,988.

**En vivo.** No hay ninguna sesión con 0,55. En las dos sesiones registradas con 0,5 se rechazó el
62,7 % (N-26: 59 ensayos, 33 por confianza y 4 por falta de manos, con el modelo anterior
«manos_postura») y el 77,8 % (N-32: 9 ensayos, los 7 rechazos por confianza, con el canónico
«manos»); sobre esos mismos ensayos, cualquier umbral ≥ 0,5 habría rechazado al menos esa proporción,
porque lo rechazado por confianza tenía confianza < 0,5 y el rechazo por falta de manos no depende
del umbral. Son sesiones cortas del estudiante (oyente), no una estimación del rechazo del Bloque 1.

### 2.8 Aplicación de la regla de §1.3

- (i) El control de N-33 se reproduce: **sí**.
- (ii) U(c = 1) de P1 ≥ 243: **sí**, 268 (+25; fuera de la franja de empate ±5).
- (iii) Aceptadas incorrectas de P1 < 297: **sí**, 104.

Por la regla fijada antes de correr, **la recomendación es adoptar el umbral anidado**. Precisión que
corrige la redacción de §1.3 sin cambiar la regla: allí dice «el criterio del anteproyecto es U»;
lo exacto es que el anteproyecto (`09_metodologia.tex:184-189`) pide determinar el valor de corte
«empíricamente sobre el conjunto de validación» y reportar su efecto sobre el rechazo y las métricas
de reconocimiento, y que la función U es la que fijó D-09 para cumplirlo.

### 2.9 Límites

1. **Cinco señantes.** Cada `t_k` sale de 4 personas y el umbral global de 5; los `t_k` van de 0,48 a
   0,67 y los de casilla de 0,19 a 0,77. El umbral hereda esa incertidumbre.
2. **El modelo interno es más débil que el que usará el umbral.** Se entrena con 3 señantes (≈498
   muestras) y no con 4 (≈698) ni con 5 (canónico): exactitud 0,474 frente a 0,580, y confianzas más
   bajas en los cinco señantes (§2.7). La dirección del sesgo sobre el umbral no está medida.
3. **El candidato global es ligeramente optimista** sobre la prueba LOSO (cada señante es señante
   interno en los otros 4 pliegues, ≈20 % de las predicciones que lo eligieron); la cifra limpia es la
   de P1.
4. **La ganancia en U no es concluyente** (IC 95 % bootstrap [−12, +62]); la reducción de errores sí es
   grande. La decisión descansa en la regla fijada, no en una prueba de significación.
5. **Las 997 predicciones de prueba ya se habían mirado en N-33 y N-46.** Aquí no se usan para elegir
   (el umbral sale solo de las casillas internas), pero la comparación P1 frente a P0 se hace sobre el
   mismo conjunto que motivó el experimento.
6. **Reproducibilidad del entrenamiento.** `entrenar_y_evaluar` fija `tf.keras.utils.set_random_seed(2026)`
   pero no activa el determinismo de operaciones; en CPU multihilo una repetición podría diferir
   ligeramente. No se midió.
7. **Parada temprana.** Cuatro pares tuvieron su mejor época entre la 10 y la 13 con una validación de
   personas vistas; se mantuvo la regla de N-30 a propósito (§1.2).
8. **Nada de esto dice cuánto rechazará el prototipo en vivo con 0,55**: solo que será al menos tanto
   como con 0,5 sobre los mismos ensayos (§2.7).

### 2.10 Recomendación

**Adoptar el umbral anidado, con el valor candidato 0,55 (`candidato_umbral.json`), mediante una
decisión nueva del registro y antes de la primera sesión del Bloque 1**; no se ha tocado el
`modelo_canonico.umbral.json` vigente (0,20).

Justificación. (a) §2.2.6 (`09_metodologia.tex:184-189`) pide fijar el valor de corte sobre
validación y reportar su efecto; el método sigue cumpliéndolo y corrige el defecto que N-33 declaró
(una validación de personas vistas para un uso con personas nuevas). (b) Con la regla fijada antes de
correr, el procedimiento limpio mejora U (268 frente a 243) y reduce los errores dichos en voz alta de
297 a 104 sobre la prueba de señantes nuevos; la exactitud entre lo aceptado sube de 0,645 a 0,782.
(c) El costo es real y hay que decirlo: el rechazo pasa de 16,0 % a ≈50 % en la prueba, se pierden
168 de 540 aciertos (P1) y, sobre los ensayos de las dos sesiones registradas con 0,5 (62,7 % y
77,8 %; N-26 con otro modelo), 0,55 habría rechazado al menos esa proporción; el rechazo del Bloque 1
no está medido [frase corregida por el verificador el 2026-09-22: decía «en vivo el rechazo será al
menos el medido con 0,5», afirmación sobre sesiones futuras que los datos no sostienen]. Eso afecta al protocolo 4.a (repeticiones por seña, duración de las
sesiones y fatiga) y el anteproyecto pide reportar los rechazos por umbral del Bloque 1
(`10_resultados.tex:43-47`). (d) Si el director prefiere no asumir ese rechazo, la alternativa honesta
es quedarse con 0,20 declarando esta medición, no escoger un valor intermedio mirando la prueba.

Dos cosas para decidir con el director junto con la adopción: el valor de `c` (0,39 con `c = 0,5`;
0,55 con `c = 1`; 0,71 con `c = 2`) y si en las sesiones en vivo se guarda la confianza de cada
ensayo para poder reportar después el efecto de 0,20 y de 0,55 sobre los mismos intentos (propuesta §7,
pregunta 6).
