# Propuesta: elegir el umbral de confianza con validación anidada por señante

**Fecha:** 16 de septiembre de 2026.
**Para qué:** que el director decida el lunes si se hace el experimento, se toma una de las salidas
baratas o se deja el umbral como está con la limitación declarada.
**Qué NO hace este documento:** no cambia el umbral vigente (sigue **0,20**, D-09/N-33), no reentrena
nada, no toca ningún modelo, ningún `.umbral.json` ni ninguna partición. Todo lo que se calcula aquí
sale de archivos que ya estaban en el repositorio; cada cifra dice de cuál.
**Estado del repositorio al escribirlo:** último commit `74041a4`. Este documento es el único archivo
que se añade; no se modificó ningún archivo ya existente. La suite de referencia dio `436 pruebas OK`
al empezar (11:44); a las 11:51 daba 1 fallo y 19 errores, todos en `test_tts` y en la prueba de
`test_prototipo` que usa el módulo de voz real, por el trabajo en curso de otro frente sobre
`src/tts.py`. Era transitorio: en la verificación independiente la suite vuelve a pasar entera
(`OK`), ya con las pruebas nuevas que ese frente va añadiendo — por eso el total sube: 464 y luego
471 en dos corridas seguidas. `tests/test_umbral.py`, la que toca a este frente, pasa 9/9.

Fuentes usadas (sha256 abreviado, para que se pueda cotejar):

| Archivo | sha256 (16) | Qué aporta |
|---|---|---|
| `models/umbral_D09_2026-09-15.json` | `4f77c36c81e5a38e` | criterio, umbral elegido, efectos y curvas de D-09 |
| `models/canonico_holistic_mascara_manos/modelo_canonico.umbral.json` | `3c7eb69679a8a2dc` | el umbral que usa hoy el prototipo |
| `models/loso_2026-09-15_holistic_mascara_manos/resultados_loso.json` | `58ed66faee8a75d0` | tiempos y épocas medidos de los 5 pliegues |
| `models/loso_2026-09-15_holistic_mascara_manos/pliegue_{0..4}/predicciones.json` | `37544c61…`, `456fb575…`, `620987b5…`, `5b9195fd…`, `da2d253f…` | las 997 predicciones de prueba (señantes nuevos) |
| `models/canonico_holistic_mascara_manos/resultados.json` | `3333aafeddb41ac3` | tiempos y épocas medidos de las 5 semillas del canónico |
| `models/analisis_errores_manos_2026-09-15/analisis.json` | — | calibración (ECE y bandas de confianza) |
| `data/partitions/loso_5_pliegues.json` | — | conjuntos por nombre y señante de cada pliegue |

**Verificación independiente (16-sep, ronda 1).** Se rehízo por separado, desde los mismos
archivos guardados, el control de N-33, la tabla visto/no visto (con inferencia propia sobre los
cinco `modelo.keras`), §5.1 y la simulación de §6, y todas las cifras coinciden. De esa revisión
salieron cuatro correcciones menores, ya incorporadas: la fila `c = 0,5` de §6 (era la del criterio
«media por señante», no la del agregado), la distinción entre `t_k` y `t` final en §2.2, el
contraste del costo con el canónico en §3 y el estado de la suite en esta cabecera. Además, esa
revisión bajó el costo de la variante A: las 20 casillas se cubren con 10 entrenamientos (§3).

---

## 1. El problema, con las cifras comprobadas

**Lo que decidió D-09** (`DECISIONES_TESIS.txt`, entrada del 15-sep-2026, y campo `criterio` de
`models/umbral_D09_2026-09-15.json`): el umbral se elige **sobre validación**, como pide §2.2.6 del
anteproyecto, maximizando `U(t) = aciertos aceptados − c · errores aceptados` con `c = 1`, en la
rejilla 0,00–0,99 y, con empate, el menor `t`. Los datos de selección son las predicciones de
validación de los 5 modelos LOSO (100 por pliegue, 500 en total). El resultado fue **t = 0,20**
(N-33), y es el valor que hoy lee el prototipo desde
`models/canonico_holistic_mascara_manos/modelo_canonico.umbral.json`.

**Efecto del 0,20, tal como está guardado** (campo `efecto` de `models/umbral_D09_2026-09-15.json`):

| Conjunto | n | rechazo | exactitud entre aceptadas | aceptadas ok/mal | U(c=1) |
|---|---:|---:|---:|---:|---:|
| validación LOSO (selección) | 500 | 2,2 % | 0,945 | 462 / 27 | 435 |
| **prueba LOSO, señantes nuevos** | **997** | **16,0 %** | **0,645** | **540 / 297** | **243** |
| prueba LOSO con t = 0,50 (referencia) | 997 | 45,9 % | 0,761 | 410 / 129 | 281 |
| validación del canónico | 100 | 2,0 % | 0,969 | 95 / 3 | 92 |
| prueba del canónico | 100 | 1,0 % | 0,889 | 88 / 11 | 77 |

Es decir: con personas nuevas el prototipo rechaza el 16 % y, de lo que acepta y dice en voz alta,
**el 35,5 % está equivocado** (297 de 837). Esa es la limitación de N-33.

**Control que hice yo** (no cito de memoria): recalculé esa fila desde los cinco
`pliegue_k/predicciones.json` y coincide exactamente con lo guardado — n = 997, exactitud sin umbral
0,579739, rechazo 0,160481, exactitud entre aceptadas 0,645161, 540 / 297
(`sim_prueba.json → control_vs_N33.coincide = true`). También reproduje las predicciones de
validación cargando los cinco `modelo.keras` ya entrenados (inferencia, **sin reentrenar**): la
exactitud de validación sale 0,95 / 0,90 / 0,91 / 0,96 / 0,94, idéntica al `mejor_val_accuracy`
guardado de cada pliegue (`val_por_senante.json → control_reproduccion`, los cinco `coincide: true`).

**Por qué pasa, medido.** El umbral hereda la escala de confianza de la validación, y esa escala se
desploma cuando la persona es nueva. La misma persona, vista por la red (aparece en la validación de
los otros cuatro pliegues) y no vista (es el señante de prueba de su pliegue):

| Señante | validación (persona vista): n, exactitud, confianza mediana | prueba (persona nueva): n, exactitud, confianza mediana | caída de la mediana |
|---|---|---|---:|
| 0000 | 96 · 0,844 · 0,798 | 200 · 0,460 · 0,332 | −0,466 |
| 0001 | 115 · 0,948 · 0,866 | 199 · 0,533 · 0,640 | −0,227 |
| 0002 | 97 · 0,928 · 0,848 | 198 · 0,606 · 0,513 | −0,335 |
| 0003 | 94 · 0,957 · 0,876 | 200 · 0,615 · 0,652 | −0,223 |
| 0004 | 98 · 0,980 · 0,894 | 200 · 0,685 · 0,747 | −0,147 |
| **global** | **500 · 0,932 · 0,864** | **997 · 0,580 · 0,555** | **−0,309** |

(validación: inferencia mía sobre los modelos ya entrenados; prueba: predicciones ya guardadas.)

Con esa escala, exigir 0,20 deja fuera el 2,2 % de la validación y el 16,0 % de la prueba. Y la banda
que el 0,20 deja entrar es justo la mala: en la prueba LOSO, el intervalo de confianza [0,20 · 0,30)
tiene 121 muestras con exactitud **0,388** (`analisis.json → calibracion.global.intervalos`), o sea
seis de cada diez errores; el ECE global es 0,066.

**Conclusión del problema:** el criterio del anteproyecto no está mal aplicado; el dato de selección
es el que no representa la situación de uso. Arreglarlo es cambiar el dato, no el criterio.

---

## 2. El método propuesto: validación anidada por señante

La idea en una frase: **el umbral debe elegirse siempre sobre una persona que la red no ha visto**.

### 2.1 Esquema

Hoy (D-06/D-08), cada pliegue LOSO parte las 997 muestras así: señante *k* completo a **prueba**
(≈199), y de los cuatro señantes restantes se apartan 2 muestras por seña (100) para **validación**;
el resto (≈697) entrena. La validación son, por tanto, personas ya vistas.

La propuesta añade un bucle interno:

```
para cada pliegue externo k = 0..4          (señante k = PRUEBA, intacto, como hoy)
    para cada señante interno j de los 4 restantes
        entrenar con los 3 señantes que no son k ni j
        predecir sobre TODAS las muestras del señante j   (persona no vista)
    juntar las predicciones de los 4 señantes internos  (≈796 por pliegue externo)
    elegir t_k con el MISMO criterio de D-09 (U(t), c = 1, rejilla 0,00-0,99, empate -> menor t)
umbral final = el elegido sobre las predicciones internas juntas de los 5 pliegues (≈3.980)
la dispersión de los cinco t_k es la incertidumbre honesta del umbral
```

### 2.2 Qué se entrena, qué se elige y qué se reporta

- **Qué se entrena:** 20 casillas (pliegue externo × señante interno), que son 10 entrenamientos
  distintos —§3 explica por qué— con la arquitectura y los hiperparámetros exactos de D-08
  (CNN+BiLSTM+atención, `seq_length` 30, 252 rasgos con velocidades, aumentación ×10, `batch` 64,
  paciencia 25, dropout 0,3, semilla de entrenamiento 2026, monitor `val_accuracy`), tomados de
  `resultados_loso.json → configuracion`. **No** se reajusta nada más: si se toca la arquitectura, el
  experimento deja de ser comparable con D-08.
- **Cómo se para el entrenamiento interno:** recomiendo mantener el criterio de hoy (checkpoint y
  early stopping sobre 2 muestras por seña de los **3 señantes de entrenamiento**, 100 muestras) y
  usar al señante interno *j* **solo** para el umbral. Si el señante *j* sirviera además de monitor
  de épocas, volveríamos a usar el mismo dato para dos cosas y el umbral quedaría otra vez optimista,
  menos que hoy pero optimista.
- **Qué se elige:** únicamente `t`. Ni arquitectura, ni épocas, ni representación, ni modelo canónico.
- **Qué se reporta:** los cinco `t_k` y su dispersión; el `t` final; y su efecto (tasa de rechazo,
  exactitud entre aceptadas, aciertos perdidos por rechazo, aceptadas ok/mal y U) en (a) las
  predicciones internas —de donde se eligió—, (b) la prueba LOSO de los 5 señantes, **solo como
  reporte, sin volver a elegir mirándola**, y (c) la validación y la prueba del canónico. Igual que
  hoy, se reporta la sensibilidad a `c = 0,5` y `c = 2` sin cambiar la elección.
- **Cuál de esos números es el estimado honesto.** Solo `t_k` está libre del señante *k*: los cuatro
  modelos internos del pliegue *k* no lo vieron y los señantes internos tampoco son *k*. Por eso el
  estimado limpio del efecto es **aplicar cada `t_k` a la prueba de su propio pliegue** y agregar
  los cinco. El `t` final, en cambio, se elige juntando las predicciones internas de los cinco
  pliegues, y ahí cada señante aparece como señante interno de los otros cuatro pliegues (≈796 de
  las ≈3.980, un 20 %): su efecto sobre la prueba LOSO es por eso **ligeramente optimista** y debe
  reportarse etiquetado como tal, no como la medida de generalización.
- **Qué NO cambia:** las 997 predicciones de prueba y las cifras de D-08/N-30 (top-1 LOSO
  0,580 ± 0,086) siguen siendo las mismas: los modelos externos no se tocan.

### 2.3 Qué habría que escribir (no está escrito)

1. Un generador de la partición anidada en `src/train_loso.py` (bandera `--anidado`), que produzca
   `data/partitions/loso_anidado_5x4.json` con los 20 sub-pliegues por nombre de muestra, con la
   misma verificación que ya existe (`verificar_particion`: conjuntos disjuntos, 50 clases, el
   señante de prueba no aparece en train/val) más una nueva: **el señante interno no puede aparecer
   en el entrenamiento**.
2. La corrida de los 10 entrenamientos a `models/loso_anidado_<fecha>/par_ab/` (cada uno con las
   predicciones de sus **dos** señantes no vistos, que alimentan dos casillas), con sus
   `predicciones.json` como hoy.
3. Un modo `--anidado` en `src/umbral_confianza.py` que junte las predicciones internas, aplique el
   mismo `elegir_umbral`, escriba el informe y actualice el `.umbral.json` con la decisión nueva.
4. Pruebas nuevas en `tests/test_umbral.py`: hoy sus 9 pruebas fijan el criterio `U(t)` y el uso del
   umbral en el prototipo, y **siguen valiendo** (el criterio no cambia); faltarían las del armado de
   la partición anidada, la agregación por pliegue y los campos del `.umbral.json`.

---

## 3. El costo: cuántos entrenamientos y cuánto tardarían

**De dónde salen los tiempos.** No hay ninguna corrida nueva aquí: los segundos vienen de
`models/loso_2026-09-15_holistic_mascara_manos/resultados_loso.json` (campos
`tiempo_entrenamiento_s` y `epocas_ejecutadas` de cada pliegue) y de
`models/canonico_holistic_mascara_manos/resultados.json` (lo mismo por semilla). Medido, en este Mac,
solo CPU (`entorno.gpus_visibles: []`):

| Corrida | s | épocas | s/época | muestras aumentadas |
|---|---:|---:|---:|---:|
| LOSO pliegue 0 | 285,0 | 67 | 4,254 | 6.970 |
| LOSO pliegue 1 | 243,3 | 46 | 5,289 | 6.980 |
| LOSO pliegue 2 | 227,4 | 42 | 5,414 | 6.990 |
| LOSO pliegue 3 | 313,4 | 61 | 5,138 | 6.970 |
| LOSO pliegue 4 | 285,3 | 62 | 4,602 | 6.970 |
| **suma LOSO** | **1.354,4 (22,6 min)** | **278** | **4,872** | — |
| canónico, 5 semillas | 1.025,2 (17,1 min) | 262 | 3,913 | 7.970 |

De ahí sale el **costo unitario medido: 0,699 s por época y por cada 1.000 muestras aumentadas**
(1.354,4 s / 278 épocas / 6.970 muestras). Aplicada al caso del que salió (697 originales, 55,6
épocas) devuelve 270,9 s, el promedio medido por pliegue; eso es aritmética, no una comprobación.
El único contraste real disponible es el canónico, que corrió con la misma arquitectura y la misma
máquina pero con más datos: la regla le predice 0,699 × 7,970 = **5,57 s/época** y lo medido son
**3,913 s/época** (`resultados.json`), un 42 % menos. O sea que el tiempo por época **no** es
proporcional al número de muestras en este rango (entre pliegues varía de 4,25 a 5,41 s/época con
prácticamente los mismos datos), así que la regla se usa aquí solo como cota superior razonable:
los minutos estimados abajo tiran a largo, no a corto.

**Estimación.** Un modelo interno entrena con 3 señantes menos las 100 de validación de época ≈ 498
muestras originales (4.980 aumentadas). Con la media de épocas medida (55,6):

| Escenario de épocas | s por entrenamiento interno | 20 casillas, una corrida cada una | 5 internos (variante B) |
|---|---:|---:|---:|
| media medida, 55,6 | 193,5 | **64,5 min** | 16,1 min |
| mínimo medido, 42 | 146,2 | 48,7 min | 12,2 min |
| máximo medido, 67 | 233,2 | 77,7 min | 19,4 min |
| peor caso, tope de 200 épocas sin early stopping | 696,2 | 3,9 h | 1,0 h |

**Los 20 entrenamientos son en realidad 10** (lo vio la verificación independiente). El modelo
interno depende solo de *qué dos señantes quedan fuera*, no de cuál de los dos es el de prueba: el
par (k = 0, j = 1) y el par (k = 1, j = 0) entrenan los dos con los señantes {2, 3, 4} y con el
mismo conjunto de parada, así que son el mismo entrenamiento. Como los dos señantes excluidos son
**ambos** personas no vistas, una sola corrida sirve para las dos casillas: predice sobre el señante
0 (que usa el pliegue 1) y sobre el señante 1 (que usa el pliegue 0). Con cinco señantes hay
C(5,2) = 10 pares, o sea **10 entrenamientos** que cubren las 20 casillas. Eso deja la variante A en
≈**32,3 min** (rango 24,4–38,9 min; 1,9 h en el peor caso), la mitad de lo estimado arriba. El
generador de la partición (§2.3) tendría que indexar por par {k, j} y no por (k, j) ordenado.

**Resumen del costo:**

- **Variante A (completa, la que recomiendo): 20 casillas = 10 entrenamientos, ≈ 32 min** (≈ 65 min
  si se implementa ingenuamente una corrida por casilla; rango 24–39 min con las épocas realmente
  observadas; 1,9 h en el peor caso imaginable). **No hay que reentrenar los 5
  modelos externos ni el canónico**, así que ese es el costo total de cómputo.
- **Variante B (económica): 5 entrenamientos, ≈ 16 min**, un solo señante interno por pliegue.
- **Añadido opcional C:** si además se quisiera recalibrar el canónico sobre 4 señantes (§4),
  5 semillas × 698 muestras ≈ **22,6 min**, y ahí sí cambian las cifras de D-08.
- Si por alguna razón se decidiera reentrenar también los 5 externos: **22,6 min medidos**.

La incertidumbre grande no son los minutos, es el número de épocas: el early stopping tiene paciencia
25 y tope 200, y con menos datos podría tardar más en parar. Por eso doy el peor caso.

---

## 4. Qué pasa con los modelos y el prototipo si se adopta

- **El modelo canónico no cambia.** `modelo_canonico.keras` sigue siendo el de D-08 (semilla 2026,
  sha256 `55e170fc…`), y su `modelo_canonico.procedencia.json` tampoco cambia: describe la
  representación y los datos, no el umbral.
- **Cambia un solo archivo de operación:** `modelo_canonico.umbral.json`. Hoy dice
  `umbral: 0.2`, `decision: "D-09"`, `seleccionado_con: "validacion de models/loso_2026-09-15_…
  (5 pliegues, 500 muestras)"`. Pasaría a decir el nuevo valor, `decision: "D-11"` (la siguiente
  libre; el registro va por D-10 y N-43), `seleccionado_con: "validación anidada por señante,
  models/loso_anidado_<fecha>, 10 modelos internos → 20 casillas, ≈3.980 predicciones sobre
  personas no vistas"`,
  con su `informe` y el mismo `sha256_modelo`.
- **El prototipo no necesita ni una línea de código.** Ya lee el umbral del modelo
  (`src/representacion.py: leer_umbral`), avisa si el sha256 no corresponde al modelo cargado y
  registra `umbral` y `umbral_origen` en el JSONL (N-34). Solo cambia el número.
- **Lo que sí cambia en la práctica: se rechazará mucho más.** Con el umbral que sugiere la
  simulación (§6, ≈0,47), en la prueba LOSO el rechazo pasaría de 16,0 % a ≈44,5 %. En vivo será
  todavía mayor: con 0,5 las dos sesiones registradas rechazaron 62,7 % (N-26) y 77,8 % (N-32). Eso
  toca el protocolo de evaluación (`docs/PROTOCOLO_EXPERIMENTAL_4a.md`): más repeticiones por seña,
  sesiones más largas y más fatiga para la persona señante; hay que decidirlo antes de la primera
  sesión del Bloque 1, no después.
- **Supuesto que queda y que hay que declarar:** el umbral se estimaría con modelos entrenados con 3
  señantes y se aplicaría a un modelo entrenado con 5 (el canónico). Más datos suelen mover la escala
  de confianza, así que la transferencia no es gratis. Control barato que propongo dejar dentro del
  experimento: comparar la distribución de confianza de los modelos internos sobre su señante no
  visto con la de los modelos externos sobre su señante de prueba; si se parecen, la transferencia se
  sostiene; si no, hay que decirlo en la tesis y, si el director quiere cerrarlo del todo, ir a la
  variante C (canónico entrenado sobre 4 señantes y quinto señante reservado para el umbral), que
  cuesta 22,6 min más y **sí** cambia las cifras del canónico de D-08.
- **La tesis:** §2.2.6 pide elegir el umbral «sobre el conjunto de validación» y reportar su efecto.
  La propuesta lo sigue cumpliendo —sigue siendo validación, nunca prueba—; lo que cambia es **cómo
  se construye esa validación**. Conviene que quede escrito así en la decisión, para que no parezca
  que nos apartamos del anteproyecto.

---

## 5. Alternativas más baratas y qué pierde cada una

### 5.1 Reagrupar por señante la validación que ya existe (costo: 0 min) — **medido: no sirve**

Se puede elegir el umbral con las 500 predicciones de validación que ya existen, pero exigiendo que
sea bueno *persona a persona* en vez de en el montón. Fijé los tres criterios antes de calcular:
(a) el agregado de D-09; (b) máximo de la utilidad media por señante (normalizada por muestras);
(c) máximo de la utilidad del **peor** señante. Resultado medido (`val_por_senante.json`):

**los tres dan t = 0,20.** Y los óptimos persona a persona dentro de la validación son todavía más
permisivos: 0,20 (señante 0000), 0,15 (0001) y 0,00 (0002, 0003 y 0004).

La razón es evidente en cuanto se ve: agrupar por persona no cambia que **las cinco personas de la
validación son personas vistas**. El problema no es el reparto, es la escala de confianza. Esta
alternativa no cuesta nada y no arregla nada; su único valor es poder escribir en la tesis que se
probó y por qué no basta.

### 5.2 Dejar el umbral en 0,20 y declarar la limitación (costo: 0 min)

Es lo que hay hoy (N-33). Se pierde: el prototipo dice en voz alta una seña equivocada en el 35,5 %
de lo que acepta ante personas nuevas, y la evaluación en línea de los Bloques 1 y 2 se hará con un
umbral que sabemos permisivo, así que medirá una operación que probablemente no es la que se querría
entregar. Se gana: cero cómputo y cero cambios; y las cifras de latencia y exactitud en línea quedan
comparables con las pruebas del 14 y 15 de septiembre.

### 5.3 Subir el umbral a 0,50 «porque en la prueba va mejor» (costo: 0 min) — **no se debe hacer**

En la prueba LOSO, 0,50 da U = 281 frente a 243 (`models/umbral_D09_2026-09-15.json`). Es tentador y
es exactamente lo que el proyecto decidió no hacer: elegir mirando el conjunto de prueba. Contaminaría
la única medida limpia que tenemos de señantes nuevos y no se podría reportar honestamente. Lo
registro aquí para que quede constancia de que se consideró y se descartó.

### 5.4 Calibrar en vez de umbralizar (temperature scaling) (costo: bajo, pero mismo vicio)

Ajustar una temperatura sobre el softmax y luego umbralizar. El ECE de la prueba es 0,066
(`analisis.json`), o sea que el desajuste global no es enorme; y si la temperatura se ajusta sobre la
misma validación de personas vistas, se hereda el mismo sesgo. Solo tiene sentido **después** de
tener un conjunto de calibración de personas no vistas, es decir, después del experimento anidado. No
lo propongo ahora.

### 5.5 Fijar el umbral por lo que cuesta un error (elegir `c` en vez de `t`) (costo: 0 min)

Cambiar `c` mueve el umbral sin cambiar el método. Con los datos de validación, `c = 2` sigue dando
0,20 y `c = 0,5` da 0,11 (`sensibilidad_c` del informe D-09): la validación es tan optimista que ni
duplicando el costo del error se mueve el umbral. Sobre personas nuevas sí se movería mucho (§6). Es
una decisión legítima del director —cuánto cuesta decir una seña equivocada frente a acertar una—,
pero por sí sola no arregla el dato de selección.

---

## 6. Simulación sin reentrenar: qué umbral habría salido de un criterio por señante

> ### Aviso, y va en serio
> Lo que sigue **no es el resultado del método propuesto** y no puede citarse como tal. Es una
> aproximación hecha con las predicciones **de prueba** ya guardadas, y tiene dos defectos que el
> experimento real no tendría: (1) usa el conjunto de prueba para elegir un umbral, que es justo lo
> que el método prohíbe, y por eso aquí solo sirve para estimar *el orden de magnitud* del cambio;
> (2) las predicciones vienen de modelos entrenados con **4** señantes, mientras que en el método
> propuesto los umbrales saldrían de modelos entrenados con **3**, que son modelos distintos y con
> otra escala de confianza. Sirve para decidir si vale la pena gastar la hora de cómputo, no para
> poner un número en el `.umbral.json`.

**Diseño de la simulación** (fijado antes de calcular): con las 997 predicciones de prueba, cuyo
señante es el segundo campo del nombre `SSSS_VVVV_RRRR`, dejo fuera a un señante, elijo el umbral con
las predicciones de los **otros cuatro** —que también son personas no vistas por el modelo que las
predijo— y lo evalúo en el que dejé fuera. Cinco veces. Criterio y `c` idénticos a D-09.

**Umbral que habría elegido, y su efecto:**

| Señante evaluado | n | t elegido con los otros 4 | rechazo | exactitud entre aceptadas | aceptadas ok/mal | U(c=1) |
|---|---:|---:|---:|---:|---:|---:|
| 0000 | 200 | 0,47 | 70,0 % | 0,767 | 46 / 14 | 32 |
| 0001 | 199 | 0,55 | 45,2 % | 0,670 | 73 / 36 | 37 |
| 0002 | 198 | 0,47 | 45,5 % | 0,769 | 83 / 25 | 58 |
| 0003 | 200 | 0,55 | 43,0 % | 0,781 | 89 / 25 | 64 |
| 0004 | 200 | 0,35 | 19,0 % | 0,753 | 122 / 40 | 82 |
| **total** | **997** | (mediana 0,47) | **44,5 %** | **0,747** | **413 / 140** | **273** |

Comparado con el 0,20 vigente, sobre las mismas 997 predicciones:

| | rechazo | exactitud entre aceptadas | señas dichas bien | señas dichas **mal** | U(c=1) |
|---|---:|---:|---:|---:|---:|
| umbral vigente 0,20 (D-09) | 16,0 % | 0,645 | 540 | **297** | 243 |
| simulación por señante (≈0,47) | 44,5 % | 0,747 | 413 | **140** | 273 |
| referencia t = 0,50 | 45,9 % | 0,761 | 410 | 129 | 281 |

Lectura: el criterio por persona no vista pide un umbral **más del doble de alto** (0,35–0,55 frente
a 0,20) y ese cambio quitaría **157 de las 297 señas equivocadas** que hoy se dirían en voz alta
(−53 %), a cambio de perder 127 aciertos (−23,5 %) y casi triplicar el rechazo. Nótese también que el
umbral por señante queda muy cerca del 0,50 «puesto a ojo» que el prototipo traía antes de D-09: lo
que la simulación sugiere no es un valor exótico, es que el 0,20 es el que se sale de sitio.

**Dos matices honestos, medidos:**

- El umbral elegido con cuatro personas **no siempre** mejora a la quinta. Con el señante 0002,
  0,47 da U = 58 y el 0,20 da U = 64; con el 0000, 0,47 rechaza el 70 % de sus muestras. Cinco
  señantes son pocos y la dispersión persona a persona es grande (t óptimo individual entre 0,31 y
  0,54). El experimento real heredará esa incertidumbre: por eso el método debe **reportar los cinco
  `t_k`**, no solo el final.
- Todo esto depende de `c`. Repetí la simulación con la misma rejilla y el mismo criterio
  **agregado** de D-09, cambiando solo `c`:

  | c | t elegidos (LOFO) | mediana | rechazo | exactitud entre aceptadas | ok/mal |
  |---|---|---:|---:|---:|---:|
  | 0,5 | 0,17 · 0,16 · 0,27 · 0,20 · 0,17 | 0,17 | 15,4 % | 0,633 | 534 / 309 |
  | 1,0 | 0,47 · 0,55 · 0,47 · 0,55 · 0,35 | 0,47 | 44,5 % | 0,747 | 413 / 140 |
  | 2,0 | 0,63 · 0,60 · 0,63 · 0,60 · 0,60 | 0,60 | 54,2 % | 0,799 | 365 / 92 |

  (Con `c = 0,5` hay empate entre 0,16 y 0,17 al dejar fuera al señante 0001, y el criterio de
  D-09 rompe el empate por el **menor** `t`: 0,16. Si en vez del criterio agregado se usa el de
  utilidad media por señante, esa fila pasa a 0,17 en los cinco, 16,0 % de rechazo, 0,637 y
  533 / 304; con `c = 1` y `c = 2` los dos criterios coinciden.)

  Con `c = 0,5` el umbral se queda en ≈0,17 aun mirando personas nuevas. O sea: **parte de la
  discusión no es estadística, es de diseño de producto** — cuánto molesta una seña dicha al revés
  frente a que el sistema pida repetir.

**Cota superior, para dimensionar:** el mejor umbral posible mirando las 997 predicciones de una vez
(un oráculo, no un método) es 0,47 con U = 289 y 428 / 139. La simulación honesta llega a U = 273. El
margen que queda por optimizar eligiendo mejor el umbral es pequeño; el techo real lo pone la
exactitud del modelo con señantes nuevos (0,580), no el umbral.

---

## 7. Recomendación

**Hacer el experimento, variante A, y no tocar el umbral hasta tenerlo.** Razones:

1. **Cuesta poco:** ≈32 min de CPU estimados (10 entrenamientos; ≈65 min si se corre una vez por
   casilla), sin reentrenar el canónico ni los modelos externos, sin cambiar ninguna cifra ya
   publicada de D-08/N-30.
2. **La salida barata ya está descartada con datos:** reagrupar por señante la validación existente
   da 0,20 otra vez (§5.1). No hay atajo.
3. **El efecto es grande y va en la dirección del usuario:** la simulación apunta a menos de la mitad
   de señas equivocadas dichas en voz alta. Aunque el experimento real dé 0,40 o 0,55 en vez de 0,47,
   la conclusión «0,20 es demasiado permisivo para personas nuevas» ya es sólida.
4. **Llega a tiempo para los Bloques 1 y 2.** Si el umbral va a cambiar, tiene que cambiar **antes**
   de la primera sesión con la persona señante; cambiarlo después obliga a repetir sesiones o a
   reportar dos configuraciones.
5. **Es defendible en la sustentación.** «Elegimos el umbral sobre validación, y construimos la
   validación con una persona que el modelo no había visto» es una frase que resiste una pregunta del
   jurado; «el umbral se eligió con las mismas personas del entrenamiento y sabíamos que quedaba
   permisivo» obliga a una explicación larga.

Si el director prefiere no gastar la hora: la segunda mejor opción no es la variante B (5
entrenamientos), sino **dejar 0,20, declarar la limitación como está en N-33 y añadir a la tesis la
medición de §5.1** (que reagrupar por señante no cambia nada) más la simulación de §6 marcada como
aproximación. Es honesto y no cuesta cómputo, pero deja el prototipo operando en el punto malo
durante toda la evaluación en línea. La variante B la desaconsejo: estimar el umbral con una sola
persona, con la dispersión que se ve en §6 (0,35–0,55), es casi tan arbitrario como el 0,5 a ojo, y
ya cuesta 16 minutos.

### Preguntas concretas para el director

1. **¿Se hace el experimento anidado (variante A, ≈32 min de CPU, 10 entrenamientos que cubren las
   20 casillas) antes de la primera sesión del Bloque 1?** Es la pregunta principal; el resto
   depende de ella.
2. **¿Cuánto cuesta un error frente a un acierto (`c`)?** Hoy `c = 1` por D-09. De esto depende que el
   umbral quede en ≈0,17 (`c = 0,5`), ≈0,47 (`c = 1`) o ≈0,60 (`c = 2`). ¿Se mantiene `c = 1` o se
   pasa a `c = 2` argumentando que decir una seña equivocada es peor que pedir repetir?
3. **¿Se acepta rechazar cerca de la mitad de los intentos?** Con ≈0,47 el rechazo en prueba pasa de
   16 % a 44,5 %, y en vivo probablemente más (con 0,5 se midió 62,7 % y 77,8 % en N-26/N-32). Eso
   alarga las sesiones y cansa a la persona señante. ¿Se ajusta el protocolo 4.a (más repeticiones,
   más descansos) o se prefiere un umbral intermedio aunque no sea el óptimo del criterio?
4. **¿El umbral estimado con modelos de 3 señantes se aplica al canónico de 5 (barato, con un
   supuesto declarado) o se recalibra el canónico sobre 4 señantes (22,6 min más, pero cambian las
   cifras del canónico de D-08)?**
5. **¿Cómo se escribe esto respecto del anteproyecto?** §2.2.6 dice «sobre el conjunto de
   validación»: propongo declarar en la decisión que se sigue cumpliendo y que lo que cambia es la
   construcción de la validación. ¿Lo ve así o prefiere presentarlo como una desviación explícita?
6. **¿Los Bloques 1 y 2 se corren con un solo umbral o con dos?** Registrar cada ensayo con su
   confianza permitiría reportar después el efecto de cualquier umbral sin repetir sesiones (el JSONL
   ya guarda `umbral` y `umbral_origen`); podría ser la salida que evita tener que acertar hoy.

---

## Anexo: cómo reproducir lo de este documento

Guiones en `scratch_tesis/lote_4e_2026-09-16/umbral_anidado/` (fuera del repositorio; no
escriben nada dentro de él):

| Guion | Qué hace | Salida |
|---|---|---|
| `sim_prueba.py` | control contra N-33 y simulación por señante con las predicciones de prueba guardadas | `sim_prueba.json` |
| `sim_extra.py` | sensibilidad a `c`, criterio de peor caso y tabla visto/no visto | `sim_extra.json` |
| `val_por_senante.py` | inferencia de los 5 `modelo.keras` ya entrenados sobre sus conjuntos `val` (no reentrena) y criterios por señante en validación | `val_por_senante.json` |
| `costo2.py` | costo unitario y estimaciones a partir de los tiempos guardados | impresión |

```bash
cd "<raíz del repositorio>"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scratch_tesis/lote_4e_2026-09-16/umbral_anidado/sim_prueba.py
```

Ninguna de estas ejecuciones modifica `models/`, `data/`, `registros/` ni el umbral vigente.
