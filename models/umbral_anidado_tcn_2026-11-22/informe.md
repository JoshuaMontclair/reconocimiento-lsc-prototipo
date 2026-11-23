# Umbral de confianza con validación anidada por señante, modelo final (22-nov-2026)

**Qué es este archivo.** La repetición, con el modelo final de D-14 (TCN + imputación + 252 rasgos),
del experimento de N-46/N-47 (`src/umbral_anidado.py`, `models/umbral_anidado_2026-09-22/informe.md`).
D-14 la dejó como pendiente derivado («rehacer […] el umbral anidado (N-47)»). Se escribe en dos
tiempos: primero el criterio (sección 1, con hora, ANTES de escribir el guion y de entrenar nada) y
después los resultados (sección 2). Ninguna cifra de resultados de este experimento existe al cerrar
la sección 1.

**Qué NO hace.** No toca el umbral vigente del modelo final (`models/tcn_imputacion/modelo.umbral.json`,
0,00; N-65, aprobado por el director, N-66 (a)), ni `models/tcn_imputacion/modelo.keras`, ni los
modelos LOSO del brazo T1 (`models/confirmacion_tcn_2026-10-03/T1_imputacion_252/`), ni el experimento
de N-47, ni `data/partitions/`. Todo lo nuevo queda en `models/umbral_anidado_tcn_2026-11-22/`,
`src/umbral_anidado_tcn.py` (nuevo) y `tests/test_umbral_anidado_tcn.py` (nuevo). Lo que sale de aquí
es un **candidato**; la adopción la decide el estudiante con el director.

---

## 1. Criterio fijado antes de correr

**Fecha y hora de cierre de esta sección: 2026-11-22 23:00 (hora local, -05).** Último commit del
repositorio al escribirla: `055d7d7`. El guion y las pruebas se escriben después de esta sección y la
corrida se lanza después de ambos; el orden se puede comprobar por las horas de modificación y por la
cabecera de `corrida.log`.

### 1.0 Lo que ya se sabía al escribir este criterio (declarado)

Las 997 predicciones de prueba LOSO del brazo T1 (semilla 2026) ya se miraron en N-65: con 0,00
(vigente) dan 686 aciertos y 311 errores aceptados (rechazo 0 %, U = 375) y con 0,50 dan 626 / 161
(rechazo 21,1 %, U = 465) (`models/tcn_imputacion/umbral_D09_2026-10-03.json`,
`efecto.prueba_loso_senantes_nuevos`). Es decir: ya se sabe que sobre esa prueba un umbral más alto
que 0,00 sube U. Por eso la recomendación NO se apoya en mirar la prueba para elegir un número: el
umbral sale solo de las casillas internas (personas no vistas por los modelos internos) y la regla de
§1.3 compara dos **procedimientos** fijados de antemano. La prueba se usa solo para medir su efecto.

### 1.1 Qué dice el anteproyecto

Igual que en N-47 (`09_metodologia.tex:184-189`, §2.2.6): el valor de corte es un parámetro de
operación que «se determinará empíricamente sobre el conjunto de validación y se reportará junto con
el efecto que produce sobre la tasa de rechazo y sobre las métricas de reconocimiento». El experimento
lo sigue cumpliendo: el umbral se elige sobre validación (construida con una persona que el modelo no
ha visto), nunca sobre la prueba LOSO ni sobre el modelo final.

### 1.2 Método exacto (el de N-47; solo cambian modelo, entrada y receta)

1. **Modelos internos, uno por PAR de señantes.** Las MISMAS 10 particiones de N-47: se generan con
   `umbral_anidado.construir_particion` importada (semilla de partición 2026, generador
   `numpy.random.default_rng(2026*100 + 10*a + b)` por par, 2 muestras por seña de los tres señantes de
   entrenamiento como validación de época (D-06), los tres duplicados de D-06 fuera de todo) y, antes de
   entrenar, deben coincidir con `models/umbral_anidado_2026-09-22/particiones_anidadas.json` (mismos
   conjuntos train / val / umbral por par). Si no coinciden, no se entrena.
   - Entrenamiento: el del brazo T1 de `src/experimento_confirmacion_tcn.py`, sin reimplementar nada:
     `experimento_arquitecturas.entrenar_corrida("T", …)` (TCN `arquitecturas_literatura.construir_tcn`
     con `CONFIG_T`, 156.402 parámetros; `ENTRENAMIENTO_COMUN`: Adam con CosineDecay desde 0,001, lote
     64, label smoothing 0,1; épocas máximas 200, paciencia 25, parada temprana y checkpoint sobre
     `val_accuracy`), con la tupla `experimento_confirmacion_tcn.canon_de_brazo(canon, "T1")` y la
     entrada `X_seq["imputacion"]` de `experimento_confirmacion_tcn.preparar()`, que es
     `mano_ausente_variantes.remuestrear_todas_con_regla(·, "imputacion", 30)` sobre
     `data/processed_holistic_mascara_rostro`, representación «manos»; aumentación ×10 con
     `train_loso.aumentar` y las transformaciones de la representación; después `add_velocity` (126
     posiciones + 126 velocidades = 252). Semilla de entrenamiento 2026 (una sola, como N-47).
   - El conjunto que `entrenar_corrida` llama `test` es aquí la unión de los dos señantes excluidos:
     **no es una prueba**, es el conjunto de calibración.
   - Cada par guarda `predicciones_<a>.json` y `predicciones_<b>.json` con el softmax completo, la clase
     real, la predicha y la confianza (máximo del softmax), con `umbral_anidado.guardar_softmax`
     importada (recarga el modelo guardado y predice con la misma entrada). Control: la confianza debe
     coincidir con `prob_predicha` del `predicciones.json` que escribe `entrenar_corrida` y la clase
     predicha debe ser la misma en todas las muestras.
2. **Casillas.** Como N-47: casilla `(k, j)` = pliegue externo `k` (señante de prueba `s_k`, de
   `data/partitions/loso_5_pliegues.json`) y señante interno `j ≠ k`; sus predicciones son las del
   modelo del par `{k, j}` sobre TODAS las muestras de `j`. 20 casillas, 10 modelos.
3. **Selector.** El de D-09, importado de `src/umbral_confianza.py` (`elegir_umbral`, `utilidad`,
   `efecto`, `REJILLA`) a través de `src/umbral_anidado.py`, sin reimplementar: `U(t) = aciertos
   aceptados − c·errores aceptados`, aceptada si confianza ≥ t, `c = 1`, rejilla 0,00 … 0,99, con empate
   el menor `t`.
4. **Agregación** (la de N-47, `umbral_anidado.elegir_umbrales` importada):
   - `t_k` = `elegir_umbral` sobre la unión de las 4 casillas del pliegue `k` (c = 1).
   - `t_global` = `elegir_umbral` sobre la unión de las 20 casillas (c = 1). **Este es el candidato**
     que va en `candidato_umbral.json`.
   - Robustez (no es el candidato): `t_(k,j)` por casilla, mediana de las 4 por pliegue, mediana de
     esas 5 medianas y mediana de los 5 `t_k`. Si difieren del candidato se dice cuánto y no se cambia.
   - La dispersión de los cinco `t_k` se reporta siempre.
5. **Sensibilidad.** `c = 0,5` y `c = 2`; se reporta y no cambia la elección.
6. **Efecto sobre la prueba LOSO del modelo final** (las 997 predicciones ya guardadas del brazo T1,
   semilla 2026: `models/confirmacion_tcn_2026-10-03/T1_imputacion_252/semilla_2026/pliegue_k/predicciones.json`,
   los mismos 5 modelos sobre cuya validación N-65 eligió 0,00; NO se reentrenan). Por pliegue y
   agregado sobre las 997:
   - con `t_k` (estimado **limpio**: `s_k` no intervino en nada de lo que eligió `t_k`);
   - con `t_global` (**ligeramente optimista**: cada señante es señante interno en los otros 4 pliegues);
   - con las dos medianas de robustez;
   - con 0,00 (vigente) y 0,50 (referencia), que deben reproducir EXACTAMENTE las cifras guardadas en
     `models/tcn_imputacion/umbral_D09_2026-10-03.json` (0,00: 686 / 311; 0,50: 626 / 161; U = 375 y
     465). Si no las reproducen, el informe se detiene ahí.
   Métricas: top-1 sin umbral (no cambia con el umbral), tasa de rechazo, exactitud entre aceptadas,
   aciertos perdidos por rechazo, aceptadas ok / mal y `U(c = 1)`, con las funciones de
   `src/umbral_anidado.py` / `src/umbral_confianza.py`.
   Reportes adicionales, descriptivos, que no entran en la regla: (a) el mismo efecto sobre las
   predicciones de prueba de las semillas 2027 y 2028 del brazo T1; (b) bootstrap pareado de la
   diferencia U(P1) − U(P0) sobre las 997 muestras (10.000 remuestreos, semilla 2026, IC 95 % por
   percentiles); (c) efecto de `t_global` sobre la prueba del esquema aleatorio del modelo final (100
   predicciones guardadas en `models/tcn_imputacion/evaluacion_modelo_seleccionado/predicciones.json`) y
   sobre su validación (leída de las curvas de `umbral_D09_2026-10-03.json`; no se vuelve a inferir).
7. **Control de transferencia.** Por señante: exactitud y mediana de confianza de los modelos internos
   (3 señantes) sobre esa persona frente a las del modelo T1 externo (4 señantes, semilla 2026) sobre la
   misma persona como prueba (`umbral_anidado.control_transferencia` importada).
8. **Comparación con N-47** (modelo anterior): umbrales y efecto, solo como contexto.
9. **Reglas de la corrida.** Una prueba de humo (1 par, pocas épocas, `--aug 1`, salida en `/tmp`);
   después los 10 entrenamientos seguidos en un proceso con `nohup`, log en
   `models/umbral_anidado_tcn_2026-11-22/corrida.log`; si un par falla se repite una vez y queda
   escrito; ningún archivo vigente se sobrescribe; el candidato se escribe en
   `models/umbral_anidado_tcn_2026-11-22/candidato_umbral.json` (formato del `.umbral.json` vigente más
   el método y el sha256 del modelo final, `6970722dd06e2672c812c0b7f74baf135d03912cc1bb79440faab8220b2c1679`,
   medido hoy con `shasum`). Tiempo estimado antes de correr: se registra en §2 junto con el real.

### 1.3 Qué decidirá la recomendación (la regla de N-47, fijada antes de ver ningún resultado)

Se comparan dos **procedimientos**: (P0) el umbral vigente 0,00 (D-09 sobre validación de señantes
vistos, N-65) y (P1) el umbral anidado aplicado pliegue a pliegue (`t_k` sobre la prueba de su propio
pliegue, el estimado limpio). Sobre las mismas 997 predicciones (T1, semilla 2026):

- Se recomendará **adoptar** el umbral anidado (valor `t_global`) si (i) el control de §1.2.6 se
  reproduce, (ii) `U(c = 1)` agregado de P1 es mayor o igual que el de P0 (375) y (iii) las aceptadas
  incorrectas de P1 son menos que las de P0 (311). La condición (ii) es la que manda.
- Se recomendará **quedarse con 0,00** si (ii) falla, aunque (iii) se cumpla.
- Si `U` queda igual o casi (±5 sobre 997), se dirá que los datos no distinguen y se recomendará por el
  argumento de método (validación con personas no vistas) y por el costo en rechazo para el protocolo
  4.a, que se cuantifica.
- En cualquier caso se reportan los cinco `t_k`, su dispersión y el rechazo que implican.

Nada de lo anterior toca `models/tcn_imputacion/modelo.umbral.json`: la adopción es una decisión
nueva del director.

---

## 2. Resultados

**Escrito el 2026-11-22 a partir de las 23:15**, después de que terminara la corrida (23:14:01). La
sección 1 no se modificó: su copia tomada al cerrarla (23:00:36) tiene sha256
`08d703099ce7e63d…`, el mismo que la sección 1 de este archivo al empezar a escribir la 2 (el archivo
entero, antes de añadir esta sección). Horas de modificación: informe §1 23:00:36, guion 23:03:22,
pruebas 23:04:02, corrida 23:04:31 → 23:14:01. Todas las cifras salen de `resultados_anidado_tcn.json`,
`resultados_entrenamiento.json` y `corrida.log` de esta carpeta.

### 2.1 Corrida

- Prueba de humo antes: 1 par (`par_0000_0001`), 2 épocas, `--aug 1`, salida `/tmp/humo_umbral_anidado_tcn`;
  sin errores (3 s de entrenamiento).
- Comando, desde la raíz del repositorio:
  `nohup python src/umbral_anidado_tcn.py > models/umbral_anidado_tcn_2026-11-22/corrida.log 2>&1 &`
  (construir, entrenar y evaluar seguidos; un solo proceso, PID 27278).
- **Estimación previa: ≈15 min (rango 10-25)**, a partir de las corridas del brazo T1 (85-192 s por
  pliegue, media ≈120 s, con ≈6.980 muestras aumentadas y tres procesos en paralelo), escalada a ≈4.980
  muestras por par y un solo proceso. **Tiempo real: 563,9 s (9,4 min) para los 10 entrenamientos**
  (de 42,8 s a 67,0 s por par; 0,81-1,02 s por época).
- **Fallos: ninguno** (todos en el primer intento).
- Controles: (a) particiones regeneradas idénticas a las de N-47 (conjuntos y casillas);
  (b) control de entrada de punta a punta: el modelo T1 guardado (semilla 2026, pliegue 0) reproduce con
  la entrada de este guion sus probabilidades guardadas de validación y de prueba (máx. |dif| 1,8·10⁻⁷ y
  2,4·10⁻⁷, misma clase en todas); (c) en los 20 `predicciones_<s>.json` la confianza difiere de
  `prob_predicha` en ≤ 2,4·10⁻⁷ y la clase predicha coincide siempre.

### 2.2 Los 10 modelos internos (TCN + imputación + 252, 3 señantes cada uno)

| par | entrena con | train (orig.) | épocas (mejor) | val. de época | exactitud sobre a | exactitud sobre b | s |
|---|---|---:|---:|---:|---:|---:|---:|
| 0000·0001 | 0002, 0003, 0004 | 498 | 50 (25) | 0,95 | 0000: 0,385 | 0001: 0,437 | 45 |
| 0000·0002 | 0001, 0003, 0004 | 499 | 55 (30) | 1,00 | 0000: 0,410 | 0002: 0,727 | 49 |
| 0000·0003 | 0001, 0002, 0004 | 497 | 73 (48) | 0,97 | 0000: 0,475 | 0003: 0,715 | 66 |
| 0000·0004 | 0001, 0002, 0003 | 497 | 78 (53) | 0,92 | 0000: 0,435 | 0004: 0,650 | 67 |
| 0001·0002 | 0000, 0003, 0004 | 500 | 47 (22) | 0,98 | 0001: 0,593 | 0002: 0,641 | 43 |
| 0001·0003 | 0000, 0002, 0004 | 498 | 56 (31) | 0,96 | 0001: 0,643 | 0003: 0,745 | 52 |
| 0001·0004 | 0000, 0002, 0003 | 498 | 69 (44) | 0,91 | 0001: 0,598 | 0004: 0,705 | 66 |
| 0002·0003 | 0000, 0001, 0004 | 499 | 54 (29) | 0,91 | 0002: 0,682 | 0003: 0,705 | 49 |
| 0002·0004 | 0000, 0001, 0003 | 499 | 62 (37) | 0,95 | 0002: 0,652 | 0004: 0,750 | 64 |
| 0003·0004 | 0000, 0001, 0002 | 497 | 72 (47) | 0,98 | 0003: 0,685 | 0004: 0,695 | 60 |

Exactitud media de los modelos internos sobre personas no vistas: **0,616** (3.988 predicciones), frente
a **0,688** de los 5 modelos T1 externos de 4 señantes sobre su señante de prueba (997). En N-47, con el
modelo anterior, eran 0,474 y 0,580.

### 2.3 Umbrales por casilla, por pliegue y global (selector de D-09, c = 1)

| pliegue k (prueba) | j = 0000 | j = 0001 | j = 0002 | j = 0003 | j = 0004 | **t_k (unión)** | mediana de las 4 casillas |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (0000) | — | 0,85 | 0,40 | 0,49 | 0,83 | **0,54** | 0,66 |
| 1 (0001) | 0,87 | — | 0,42 | 0,53 | 0,70 | **0,62** | 0,61 |
| 2 (0002) | 0,68 | 0,67 | — | 0,57 | 0,36 | **0,65** | 0,62 |
| 3 (0003) | 0,74 | 0,66 | 0,62 | — | 0,28 | **0,67** | 0,64 |
| 4 (0004) | 0,74 | 0,65 | 0,45 | 0,74 | — | **0,74** | 0,70 |

- **Umbral global (unión de las 20 casillas, 3.988 predicciones): t_global = 0,68. Es el candidato.** En
  la selección: rechazo 41,2 %, exactitud entre aceptadas 0,794, 1.861 / 483, U = 1.378.
- Dispersión de los cinco `t_k`: **0,54 – 0,74** (0,54; 0,62; 0,65; 0,67; 0,74). Por casilla, de 0,28 a
  0,87: una sola persona de umbral sigue dando un valor muy inestable, como en N-47.
- Robustez (no es el candidato): mediana de las medianas de casilla = **0,64**; mediana de los cinco
  `t_k` = **0,65**. Las tres agregaciones quedan entre 0,64 y 0,68 (a ≤ 0,04 del candidato); ninguna se
  acerca a 0,00.

### 2.4 Efecto sobre la prueba LOSO del modelo final (997 predicciones guardadas de T1, semilla 2026)

Control: con 0,00 y 0,50 se reproducen exactamente las cifras de N-65 (686 / 311, U = 375; 626 / 161,
U = 465; `control_N65.coincide = true` en ambos). Aceptada si confianza ≥ t. El top-1 sin umbral es
**0,688** en todas las filas: el umbral no cambia la clase predicha, solo decide si se dice.

| procedimiento | umbral(es) | rechazo | exactitud entre aceptadas | aciertos perdidos por rechazo | aceptadas ok / mal | U (c = 1) |
|---|---|---:|---:|---:|---:|---:|
| **P1: anidado por pliegue, `t_k` (estimado limpio)** | 0,54 / 0,62 / 0,65 / 0,67 / 0,74 | **30,9 %** | **0,827** | 16,9 % | **570 / 119** | **451** |
| anidado global 0,68 (candidato; ligeramente optimista) | 0,68 | 35,0 % | 0,844 | 20,3 % | 547 / 101 | 446 |
| mediana de casillas por pliegue (robustez) | 0,66 / 0,61 / 0,62 / 0,64 / 0,70 | 32,0 % | 0,832 | 17,8 % | 564 / 114 | 450 |
| mediana de medianas 0,64 (robustez) | 0,64 | 31,8 % | 0,831 | 17,6 % | 565 / 115 | 450 |
| mediana de `t_k` 0,65 (robustez) | 0,65 | 32,7 % | 0,835 | 18,4 % | 560 / 111 | 449 |
| **P0: vigente 0,00 (N-65)** | 0,00 | **0,0 %** | **0,688** | 0,0 % | **686 / 311** | **375** |
| referencia 0,50 | 0,50 | 21,1 % | 0,795 | 8,7 % | 626 / 161 | 465 |

**Por pliegue** (rechazo · exactitud entre aceptadas · ok/mal · U):

| pliegue (prueba) | P1 `t_k` | 0,68 (candidato) | 0,00 (vigente) | 0,50 |
|---|---|---|---|---|
| 0 (0000), n = 200 | 0,54: 39,0 % · 0,705 · 86/36 · 50 | 58,5 % · 0,783 · 65/18 · 47 | 0 % · 0,540 · 108/92 · 16 | 35,5 % · 0,674 · 87/42 · 45 |
| 1 (0001), n = 199 | 0,62: 26,6 % · 0,747 · 109/37 · 72 | 29,1 % · 0,752 · 106/35 · 71 | 0 % · 0,613 · 122/77 · 45 | 19,1 % · 0,714 · 115/46 · 69 |
| 2 (0002), n = 198 | 0,65: 42,4 % · 0,921 · 105/9 · 96 | 43,4 % · 0,920 · 103/9 · 94 | 0 % · 0,717 · 142/56 · 86 | 27,8 % · 0,888 · 127/16 · 111 |
| 3 (0003), n = 200 | 0,67: 21,0 % · 0,835 · 132/26 · 106 | 22,0 % · 0,846 · 132/24 · 108 | 0 % · 0,740 · 148/52 · 96 | 12,0 % · 0,795 · 140/36 · 104 |
| 4 (0004), n = 200 | 0,74: 25,5 % · 0,926 · 138/11 · 127 | 22,0 % · 0,904 · 141/15 · 126 | 0 % · 0,830 · 166/34 · 132 | 11,0 % · 0,882 · 157/21 · 136 |

Lectura: frente al 0,00, el procedimiento limpio P1 dice en voz alta **192 señas equivocadas menos**
(311 → 119; −62 %) a cambio de **116 aciertos menos** (686 → 570; −17 %), con un rechazo del 30,9 %.
U sube 76 (375 → 451). P1 gana en U en 4 de los 5 pliegues (+34, +27, +10, +10) y pierde 5 en el
pliegue 4 (señante 0004, el más fácil).

**Incertidumbre de la diferencia en U** (fijada en §1.2.6 b, no entra en la regla): bootstrap pareado
sobre las 997 muestras, 10.000 remuestreos, semilla 2026: U(P1) − U(P0) = +76, **IC 95 % [+42, +110]**;
ningún remuestreo da una diferencia ≤ 0. Para el candidato global: +71, IC 95 % [+35, +108]. A
diferencia de N-47 (IC [−12, +62]), aquí la ganancia en U sí es clara.

**Otras semillas del brazo T1** (descriptivo, §1.2.6 a; los umbrales son los mismos, elegidos con los
modelos internos de semilla 2026):

| semilla T1 | top-1 | P1 `t_k`: rechazo · exact. acept. · ok/mal · U | 0,68: rechazo · exact. acept. · ok/mal · U | 0,00: ok/mal · U |
|---|---:|---|---|---|
| 2026 (principal) | 0,688 | 30,9 % · 0,827 · 570/119 · 451 | 35,0 % · 0,844 · 547/101 · 446 | 686/311 · 375 |
| 2027 | 0,658 | 33,1 % · 0,822 · 548/119 · 429 | 35,9 % · 0,833 · 532/107 · 425 | 656/341 · 315 |
| 2028 | 0,685 | 31,9 % · 0,835 · 567/112 · 455 | 36,5 % · 0,848 · 537/96 · 441 | 683/314 · 369 |

En las tres semillas P1 supera al 0,00 en U por 76, 114 y 86 y reduce los errores dichos en un 62-65 %.

### 2.5 Sensibilidad a c (no cambia la elección)

| c | `t_k` por pliegue | `t` global | mediana de `t_k` | mediana de medianas | prueba con `t_k`: rechazo · exact. acept. · ok/mal · U(c=1) | con `t` global: rechazo · exact. acept. · ok/mal · U(c=1) |
|---|---|---:|---:|---:|---|---|
| 0,5 | 0,40 / 0,33 / 0,30 / 0,28 / 0,36 | 0,36 | 0,33 | 0,38 | 8,3 % · 0,726 · 664/250 · 414 | 9,7 % · 0,736 · 662/238 · 424 |
| **1** | 0,54 / 0,62 / 0,65 / 0,67 / 0,74 | **0,68** | 0,65 | 0,64 | 30,9 % · 0,827 · 570/119 · 451 | 35,0 % · 0,844 · 547/101 · 446 |
| 2 | 0,85 / 0,77 / 0,72 / 0,82 / 0,79 | 0,76 | 0,79 | 0,74 | 44,5 % · 0,864 · 478/75 · 403 | 41,0 % · 0,859 · 505/83 · 422 |

Con cualquier `c` del rango el umbral anidado queda muy por encima de 0,00 (en N-65, sobre validación de
señantes vistos, c = 0,5 y c = 1 daban 0,00 y c = 2 daba 0,32).

### 2.6 Control de transferencia y efecto sobre el modelo final

| señante | modelos internos (3 señantes): n · exactitud · confianza mediana | modelo T1 externo (4 señantes): n · exactitud · confianza mediana |
|---|---|---|
| 0000 | 800 · 0,426 · 0,483 | 200 · 0,540 · 0,641 |
| 0001 | 796 · 0,568 · 0,777 | 199 · 0,613 · 0,881 |
| 0002 | 792 · 0,676 · 0,765 | 198 · 0,717 · 0,757 |
| 0003 | 800 · 0,713 · 0,855 | 200 · 0,740 · 0,896 |
| 0004 | 800 · 0,700 · 0,903 | 200 · 0,830 · 0,912 |
| global | 3.988 · 0,616 · 0,783 | 997 · 0,688 · 0,836 |

Las escalas **no coinciden del todo**: los modelos de 3 señantes son menos exactos en las cinco personas
y menos confiados en cuatro (mediana global −0,053; en N-47 era −0,095). La transferencia 3 → 4 señantes
está medida (§2.4); la 4 → 5 (modelo final del esquema aleatorio) sigue siendo un supuesto, porque ese
modelo no tiene prueba con personas nuevas.

**Efecto de 0,68 sobre el modelo final** (`models/tcn_imputacion/`, personas vistas; solo reporte):
prueba del esquema aleatorio (100): rechazo 6,0 %, exactitud entre aceptadas 0,968, 91 / 3, U = 88 (con
0,00: 0 %, 0,940, 94 / 6, U = 88: empate). Validación del modelo final (100, curvas de N-65): rechazo
5,0 %, exactitud entre aceptadas 0,979. Validación LOSO de selección de N-65 (500): rechazo 9,8 %,
exactitud entre aceptadas 0,965.

### 2.7 Comparación con N-47 (modelo anterior)

| | modelo | `t_k` por pliegue | `t` global | exactitud de los internos | P1 en la prueba LOSO: rechazo · ok/mal · U | vigente: ok/mal · U |
|---|---|---|---:|---:|---|---|
| N-47 | CNN+BiLSTM+atención, «mascara» | 0,50 / 0,66 / 0,67 / 0,58 / 0,48 | 0,55 | 0,474 | 52,3 % · 372/104 · 268 | 0,20: 540/297 · 243 |
| **este** | TCN + imputación + 252 (D-14) | 0,54 / 0,62 / 0,65 / 0,67 / 0,74 | **0,68** | 0,616 | **30,9 % · 570/119 · 451** | 0,00: 686/311 · 375 |

El umbral anidado sube (0,55 → 0,68) porque el modelo final es más confiado (mediana de confianza en la
prueba LOSO 0,836 frente a 0,555), pero el rechazo que implica en la prueba baja de ≈50 % a ≈31-35 %
porque las confianzas altas se concentran más en los aciertos.

### 2.8 Aplicación de la regla de §1.3

- (i) El control de N-65 se reproduce: **sí**.
- (ii) U(c = 1) de P1 ≥ 375: **sí**, 451 (+76; fuera de la franja de empate ±5).
- (iii) Aceptadas incorrectas de P1 < 311: **sí**, 119.

Por la regla fijada antes de correr, **la recomendación es adoptar el umbral anidado, con el valor
candidato 0,68**.

### 2.9 Límites

1. **Cinco señantes.** Los `t_k` van de 0,54 a 0,74 y los de casilla de 0,28 a 0,87.
2. **Una sola semilla de entrenamiento** de los modelos internos (2026, como N-47). No se midió cuánto
   cambiarían los umbrales con otra; sí se midió que el efecto se mantiene sobre las predicciones de
   prueba de las semillas 2027 y 2028 del brazo T1 (§2.4).
3. **El modelo interno es más débil que el que usará el umbral** (0,616 frente a 0,688, y menos confiado;
   §2.6). La dirección del sesgo sobre el umbral no está medida.
4. **El candidato global es ligeramente optimista** sobre la prueba LOSO; la cifra limpia es la de P1
   (U = 451, rechazo 30,9 %).
5. **La prueba ya se conocía** (§1.0). La referencia 0,50 da en ella U = 465, por encima de P1 (451) y
   del candidato (446). Ese contraste no es un argumento para elegir 0,50: se lee mirando la prueba, y
   en N-47 ocurrió lo mismo.
6. **Nada de esto dice cuánto rechazará el prototipo en vivo con 0,68.** No hay ninguna sesión con el
   modelo final y un umbral distinto de 0,00 cuyos ensayos se hayan reanalizado aquí.
7. **Reproducibilidad.** `entrenar_corrida` fija `set_random_seed(2026)` sin determinismo de
   operaciones; en CPU multihilo una repetición podría diferir ligeramente. No se midió.

### 2.10 Recomendación

**Adoptar el umbral anidado con el valor candidato 0,68 (`candidato_umbral.json`) para el modelo final,
mediante una decisión nueva del registro**; no se ha tocado `models/tcn_imputacion/modelo.umbral.json`
(sigue en 0,00).

Justificación. (a) §2.2.6 del anteproyecto pide fijar el valor de corte sobre validación y reportar su
efecto; el método lo cumple y corrige el defecto de N-65 (validación con personas vistas, donde la
utilidad es plana entre 0 y 0,15 y el empate lleva a 0,00). (b) Con la regla fijada antes de correr, el
procedimiento limpio sube U de 375 a 451 (IC 95 % bootstrap de la diferencia [+42, +110]) y baja los
errores dichos en voz alta de 311 a 119 sobre la prueba de señantes nuevos; la exactitud entre lo
aceptado sube de 0,688 a 0,827; el resultado se repite con las otras dos semillas de T1. (c) El costo:
el rechazo pasa de 0 % a ≈31-35 % en la prueba LOSO (se pierden 116 de 686 aciertos con P1) y a ≈6 % en
la prueba del esquema aleatorio, con la persona 0000 rechazada en un 39-59 %. Eso afecta al protocolo
4.a (repeticiones y duración de las sesiones) y el anteproyecto pide reportar los rechazos por umbral
del Bloque 1. (d) Si el director prefiere no asumir ese rechazo, la alternativa honesta es quedarse con
0,00 declarando esta medición, no escoger un valor intermedio mirando la prueba.

Para decidir con el director junto con la adopción: el valor de `c` (0,36 con c = 0,5; 0,68 con c = 1;
0,76 con c = 2) y que, si se adopta, la cifra que se cita es la limpia (P1: rechazo 30,9 %, 570 / 119,
U = 451).
