# Experimento: diseño factorial con la TCN ({mascara, pivote, imputación} × {252, 132})

Fecha: 3 de octubre de 2026. Objetivo: elegir la configuración de preprocesamiento del modelo final con la TCN
pequeña con GAP (`CONFIG_T`, la arquitectura elegida en `docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md` y
`docs/EXPERIMENTO_ARQUITECTURAS_2.md`).

**Pregunta.** La confirmación conjunta (`docs/EXPERIMENTO_CONFIRMACION_TCN.md`) midió 3 de las 6 combinaciones
de {tratamiento de la mano ausente} × {velocidades}. Encontró que la exactitud de validación está **saturada**
(unos 0,95 con 100 muestras por pliegue, donde cada muestra vale 0,01) y ya no distingue entre brazos. Su regla
eligió T2 por no inferioridad, aunque T1 fue claramente mejor en prueba y T2 empeoró a la persona 0000. Aquí
se completan las 3 celdas que faltan y se decide con un criterio de validación menos saturado.

Este documento no cambia el modelo canónico, el umbral ni `translator.py`.

---

## 1. Diseño

Los seis brazos usan la arquitectura T (`CONFIG_T` exacta de `src/arquitecturas_literatura.py`: 4 bloques
residuales dilatados 1-2-4-8, 64 filtros, k = 3, GAP) y su receta (Adam lr 1e-3 con CosineDecay,
*label smoothing* 0,1, lote 64, paciencia 25 sobre `val_accuracy`, máx. 200 épocas). Aumentación ×10 y 30 pasos.
Datos: `data/processed_holistic_mascara_rostro`, representación «manos». LOSO de 5 pliegues (D-06), validación
interna de 2 por seña (100 muestras). Semillas 2026, 2027 y 2028.

| Brazo | Mano ausente | Entrada | Parámetros | Origen |
|---|---|---|---:|---|
| **T0** | «mascara» (D-07) | 126 posiciones + 126 velocidades = 252 | 156.402 | reutilizado: `models/arquitecturas_literatura_2026-10-02/T_tcn` |
| **T1** | imputación | 252 | 156.402 | reutilizado: `models/confirmacion_tcn_2026-10-03/T1_imputacion_252` |
| **T2** | imputación | 126 + 6 velocidades por coordenada = 132 | 125.682 | reutilizado: `models/confirmacion_tcn_2026-10-03/T2_imputacion_132` |
| **T3** | «mascara» | 132 | 125.682 | **nuevo** |
| **T4** | pivote | 252 | 156.402 | **nuevo** |
| **T5** | pivote | 132 | 125.682 | **nuevo** |

- **Pivote** (`mano_ausente_variantes.regla_pivote`): cada uno de los 30 momentos toma un fotograma real en el
  que la mano sí se detectó (por mano, sin interpolar). **Imputación** (`regla_imputacion`): interpolación
  lineal por mano sobre los cuadros originales y después remuestreo. Ambas se aplican sobre el *processed*
  «mascara» antes de remuestrear (`remuestrear_todas_con_regla`), como en `docs/EXPERIMENTO_MANO_AUSENTE.md`.
- **132** = `velocidades_reducidas.funcion_entrada("por_coordenada")`, que sustituye a `add_velocity`; se calcula
  después de la aumentación y del remuestreo, como en `docs/EXPERIMENTO_VELOCIDADES_REDUCIDAS.md`.
- El script nuevo `src/experimento_factorial_tcn.py` no reimplementa nada: entrena con
  `experimento_arquitecturas.entrenar_corrida` (constructor «T») y reutiliza las funciones de
  `experimento_confirmacion_tcn` (comparación de corridas, σ agrupada, Δ por pliegue, regla de mejora, voto de
  semillas) y de `experimento_arquitecturas` (McNemar, pruebas pareadas).

### 1.1 `val_loss` de T0, T1 y T2, fijada antes de ver los brazos nuevos (3-oct, 01:42:04)

`python src/experimento_factorial_tcn.py fijar_val_loss` extrajo la `val_loss` de las 45 corridas reutilizadas y
la guardó en `models/factorial_tcn_2026-10-03/val_loss_T0_T1_T2_fijada.json`. No hubo que recalcularla: la guarda
`entrenar_corrida` (`model.evaluate` del *checkpoint* recargado, elegido por `val_accuracy`, sobre la validación
del pliegue; entropía cruzada con *label smoothing* 0,1, la misma pérdida del entrenamiento). Se comprobó que
coincide con el recálculo desde `probabilidades.npz` (diferencia máxima 4,4·10⁻⁵).

| Brazo | `val_loss` media | Por pliegue 0 / 1 / 2 / 3 / 4 (media de 3 semillas) | DE entre semillas |
|---|---:|---|---:|
| T0 | 0,9916 | 0,9203 / 1,0060 / 1,0564 / 0,8857 / 1,0897 | 0,0096 |
| T1 | 0,9776 | 0,8974 / 1,0118 / 1,0464 / 0,9198 / 1,0127 | 0,0228 |
| T2 | 0,9899 | 0,9138 / 1,0450 / 1,0423 / 0,8881 / 1,0602 | 0,0492 |

### 1.2 Verificación previa (3-oct, 01:42 a 01:46)

`python src/experimento_factorial_tcn.py verificar --dest <carpeta_temporal>/repro_t1`:

- **Configuración de los reutilizados.** Las 45 corridas de T0, T1 y T2 tienen `CONFIG_T`, la receta común,
  200 épocas, paciencia 25, aumentación ×10, 30 pasos, el *processed* canónico, «manos» y la regla y entrada de su
  brazo (T0 con `experimento_confirmacion_tcn.comprobar_config_t0`).
- **Entradas.** Las `X_seq` «mascara» e «imputación» de este script son, bit a bit, las de
  `experimento_confirmacion_tcn.preparar`. El pivote cambia 986 de los 1000 videos frente a «mascara» (la
  imputación, 735): el pivote no interpola ni siquiera cuando la mano está, así que casi todo video cambia.
  T0, T1 y T4 usan `add_velocity` (252); T2, T3 y T5, «por_coordenada» (132). Las posiciones no cambian.
- **Parámetros:** 156.402 con 252 rasgos y 125.682 con 132.
- **Reproducción de T1 (semilla 2026, pliegue 0):** PENDIENTE al escribir el criterio; se completa en §3.

### 1.3 Prueba de humo (3-oct, 01:43)

En el *carpeta_temporal*: T3, T4 y T5 con la semilla 2026, el pliegue 0 y 3 épocas. Los modelos se guardan, se recargan
con la entrada correcta (30 × 132, 30 × 252, 30 × 132), con los parámetros esperados y con la procedencia
corregida (regla y velocidades del brazo). Las cifras de la prueba de humo no se usan
(`models/factorial_tcn_2026-10-03/humo.json`).

---

## 2. Criterio prerregistrado (escrito el 3-oct-2026 a las 01:44, hora de Colombia, antes de las corridas completas)

No se ha visto ningún resultado completo de T3, T4 ni T5; solo la prueba de humo de 3 épocas. La exactitud
de validación de T0, T1 y T2 ya se conocía (confirmación) y su `val_loss` se fijó en §1.1. **Solo deciden
métricas de validación. La prueba LOSO nunca decide.**

**Notación.** Para cada brazo V, pliegue k y semilla s: `val(V, s, k)` es el `val_accuracy` máximo (el del
*checkpoint*) y `loss(V, s, k)` es la `val_loss` del *checkpoint* (§1.1). `val(V, k)` y `loss(V, k)` son la media
de las 3 semillas. Las diferencias por pliegue se redondean a 6 decimales y las comparaciones con un umbral usan
una tolerancia de 1e-9 (como en la confirmación).

1. **Regla 1: top-1 medio de validación.**
   - **Margen** = máx(σ, 0,01). σ es la DE entre semillas agrupada sobre **los 6 brazos**: para cada brazo, la
     DE (ddof = 1) entre las 3 semillas de la validación media de 5 pliegues; luego, la raíz de la media de las
     6 varianzas (`experimento_confirmacion_tcn.sigma_agrupada`).
   - **A supera a B** si la media de Δk = val(A, k) − val(B, k) es mayor que el margen **y** Δk > 0 en al menos
     4 de los 5 pliegues (Δk = 0 no cuenta).
   - Se evalúan los 30 pares ordenados. **Candidatos** = los brazos a los que ningún otro brazo supera.
   - Si queda **un solo candidato**, ese brazo es la configuración recomendada. Si hay un ciclo y no queda
     ninguno, los candidatos son los brazos cuya media está a ≤ margen de la mejor.
2. **Regla 2: desempate con la `val_loss`** (solo entre los candidatos de la regla 1; menor es mejor).
   - **σ_loss** = DE entre semillas de la `val_loss` media de 5 pliegues, agrupada sobre los 6 brazos igual que σ
     (sin piso).
   - **A supera a B en pérdida** si dk = loss(B, k) − loss(A, k) es > 0 (A menor) en al menos 4 de los 5
     pliegues **y** la media de dk es mayor que σ_loss.
   - Candidatos de la regla 2 = los candidatos de la regla 1 a los que ningún otro candidato supera en pérdida.
     Si queda uno, ese es la configuración recomendada. Si hay un ciclo, se pasa a la regla 3 con todos los
     candidatos de la regla 1.
3. **Regla 3: simplicidad.** Entre los candidatos que queden, se elige el primero en este orden, declarado aquí:
   - **Primero el preprocesamiento**, por costo y complejidad en vivo: **mascara < imputación < pivote**.
     «mascara» es lo que ya hace `translator.py`; la imputación añade una interpolación sobre la ventana; el
     pivote añade una búsqueda de fotogramas con reglas de ventana y tope.
   - **Después, las velocidades, dentro del mismo preprocesamiento: 252 antes que 132** si la diferencia de costo
     entre los dos es despreciable: menos de 1 ms de latencia total (red con `predict` de lote 1 + armado de la
     entrada) **y** menos de 0,5 MB de modelo en disco. 252 no cambia la representación vigente. Si la
     diferencia no es despreciable, 132 va antes que 252. El costo se mide con el subcomando `inferencia`
     (§2.5).
   - Con costos despreciables, el orden completo es **T0 > T3 > T1 > T2 > T4 > T5**.
4. **Descriptivo, sin valor de decisión:**
   - prueba top-1, top-3 y F1 macro;
   - la persona 0000 (pliegue 0) aparte;
   - la NLL de validación sin *label smoothing* (desde `probabilidades.npz`);
   - McNemar exacto sobre las 997 predicciones de prueba agregadas, con la semilla 2026 y con el voto de las 3
     semillas (mayoría; si las tres discrepan, gana la semilla más confiada). Se hace para los 15 pares de
     T0-T5 y para cada brazo frente a R0;
   - signos y Wilcoxon por pliegue (validación, `val_loss`, prueba top-1, top-3 y F1). Con 5 pliegues, el p
     bilateral mínimo es 2/2⁵ = **0,0625**, así que ninguna de las dos puede llegar a 0,05;
   - una tabla final 3 × 2 (preprocesamiento × velocidades) con validación, `val_loss` y prueba;
   - la comparación con el canónico actual R0 (CNN + BiLSTM + atención; validación 0,920, prueba 0,589).
5. **Costo.** Latencia con lote de 1 de la red (`predict`, como `translator.py`, y llamada directa) y del armado de
   la entrada de una ventana con cada regla (*processed* → regla → 30 pasos → velocidades). Se mide sin
   entrenamientos activos, después de las corridas.
6. **Plan de ejecución.** Tres procesos en paralelo, uno por semilla. Cada uno recorre los pliegues 0 a 4 y,
   dentro de cada pliegue, T3 → T4 → T5. Son 45 corridas. Si no terminaran, el criterio se aplicaría con las
   semillas que tengan los 5 pliegues en los 6 brazos (mínimo 2), y se declararía.
7. **Condición previa.** T0, T1 y T2 solo se reutilizan si la reproducción de T1 (semilla 2026, pliegue 0) con
   este script da `predicciones.json` idéntico byte a byte. Si no, se reentrenan con este script.

---

## 3. Ejecución

- **Reproducción de T1 (condición previa, §2.7).** Terminó a las 01:44, después de escribir el criterio y antes
  de lanzar las corridas. Se reentrenó T1 con la semilla 2026 y el pliegue 0 con este script, en el
  *carpeta_temporal*. Resultado:
  - `predicciones.json` **idéntico byte a byte**;
  - probabilidades de validación y de prueba con diferencia máxima 0;
  - historial idéntico;
  - mismas 43 épocas, validación 0,980, `val_loss` 0,870120 y prueba 0,540 iguales.

  Por eso T0, T1 y T2 se reutilizan (`models/factorial_tcn_2026-10-03/verificacion.json`).
- **Lanzamiento:** 3-oct a las 01:44:25, después de comprobar que no había entrenamientos activos. Fueron 3
  procesos en paralelo, uno por semilla, en el orden pliegue 0 → 4 y T3 → T4 → T5. Terminaron entre las 02:14 y
  las 02:19. Son 45 corridas sin errores ni recortes, y el criterio se aplica con las 3 semillas completas.
- **Coherencia.** En las 45 corridas nuevas, el `val_accuracy` máximo coincide con la exactitud del modelo
  recargado sobre la validación. La `val_loss` guardada coincide con su recálculo desde `probabilidades.npz`
  (diferencia máxima 1,4·10⁻⁷). La `val_loss` fijada de T0, T1 y T2 (§1.1) no cambió: `resumen` lo comprueba.
- **Latencia:** se midió a las 02:19, sin ningún proceso de entrenamiento activo.
- **Criterio:** no se modificó después de ver resultados.

---

## 4. Resultados

### 4.1 Tabla final 3 × 2 (preprocesamiento × velocidades)

Media de 3 semillas × 5 pliegues. **Validación** (decide): top-1 y `val_loss`. **Prueba** (descriptiva):
top-1 y F1 macro.

| Preprocesamiento | **252** (126 pos. + 126 vel.) | **132** (126 pos. + 6 vel.) |
|---|---|---|
| **mascara** | **T0** · val 0,949 · loss 0,992 · prueba 0,651 · F1 0,619 | **T3** · val 0,941 · loss 0,997 · prueba 0,654 · F1 0,624 |
| **imputación** | **T1** · val 0,949 · loss 0,978 · prueba **0,677** · F1 **0,647** | **T2** · val 0,945 · loss 0,990 · prueba 0,653 · F1 0,622 |
| **pivote** | **T4** · val **0,956** · loss **0,954** · prueba 0,670 · F1 0,639 | **T5** · val 0,949 · loss 0,977 · prueba 0,656 · F1 0,628 |

Referencia: el canónico R0 (CNN + BiLSTM + atención, «mascara», 252) tiene validación 0,920 y prueba 0,589
(F1 0,544).

**Lectura descriptiva del factorial:**

- **Velocidades.** Pasar de 252 a 132 nunca mejora la validación: −0,007 con mascara, −0,005 con imputación y
  −0,007 con pivote. Tampoco la `val_loss`: +0,006, +0,012 y +0,023. En prueba, el efecto es nulo con mascara
  (+0,003) y negativo con imputación (−0,024) y con pivote (−0,013).
- **Preprocesamiento.** En validación ordena **pivote > imputación ≈ mascara**, tanto con 252 como con 132. En
  prueba, con 252, el orden es **imputación (0,677) > pivote (0,670) > mascara (0,651)**.

### 4.2 Tabla completa

| Brazo | Parámetros | **Val top-1** (DE pl.) | **`val_loss`** (DE sem.) | NLL val sin suavizado | Prueba top-1 (DE pl.) | Top-3 | F1 macro | Voto 3 semillas | Persona 0000: val / prueba |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| R0 | 714.291 | 0,920 (0,020) | — | — | 0,589 (0,094) | 0,764 | 0,544 | 0,609 | 0,923 / 0,452 |
| T0 mascara 252 | 156.402 | 0,949 (0,019) | 0,992 (0,010) | 0,311 | 0,651 (0,117) | 0,820 | 0,619 | 0,687 | 0,963 / 0,465 |
| **T1 imputación 252** | 156.402 | 0,949 (0,017) | 0,978 (0,023) | 0,290 | **0,677** (0,127) | **0,847** | **0,647** | **0,716** | 0,960 / **0,488** |
| T2 imputación 132 | 125.682 | 0,945 (0,020) | 0,990 (0,049) | 0,297 | 0,653 (0,147) | 0,835 | 0,622 | 0,690 | 0,957 / 0,408 |
| T3 mascara 132 | 125.682 | 0,941 (0,016) | 0,997 (0,002) | 0,314 | 0,654 (0,118) | 0,822 | 0,624 | 0,694 | 0,957 / 0,480 |
| T4 pivote 252 | 156.402 | **0,956** (0,014) | **0,954** (0,024) | **0,272** | 0,670 (0,131) | 0,845 | 0,639 | 0,702 | 0,967 / 0,457 |
| T5 pivote 132 | 125.682 | 0,949 (0,017) | 0,977 (0,002) | 0,287 | 0,656 (0,147) | 0,819 | 0,628 | 0,687 | 0,970 / 0,427 |

**Por pliegue** (media de 3 semillas; pliegues 0 / 1 / 2 / 3 / 4):

| Brazo | Val top-1 | `val_loss` | Prueba top-1 |
|---|---|---|---|
| R0 | 0,923 / 0,893 / 0,917 / 0,950 / 0,917 | — | 0,452 / 0,536 / 0,631 / 0,640 / 0,687 |
| T0 | 0,963 / 0,940 / 0,937 / 0,973 / 0,930 | 0,920 / 1,006 / 1,056 / 0,886 / 1,090 | 0,465 / 0,628 / 0,663 / 0,753 / 0,745 |
| T1 | 0,960 / 0,937 / 0,937 / 0,973 / 0,940 | 0,897 / 1,012 / 1,046 / 0,920 / 1,013 | 0,488 / 0,611 / 0,721 / 0,770 / 0,795 |
| T2 | 0,957 / 0,923 / 0,933 / 0,973 / 0,937 | 0,914 / 1,045 / 1,042 / 0,888 / 1,060 | 0,408 / 0,630 / 0,717 / 0,775 / 0,733 |
| T3 | 0,957 / 0,927 / 0,927 / 0,960 / 0,937 | 0,934 / 1,026 / 1,053 / 0,928 / 1,047 | 0,480 / 0,598 / 0,680 / 0,755 / 0,758 |
| T4 | 0,967 / 0,963 / 0,950 / 0,967 / 0,933 | 0,915 / 0,950 / 0,945 / 0,902 / 1,057 | 0,457 / 0,633 / 0,731 / 0,758 / 0,770 |
| T5 | 0,970 / 0,940 / 0,937 / 0,963 / 0,933 | 0,886 / 0,991 / 1,036 / 0,901 / 1,069 | 0,427 / 0,606 / 0,690 / 0,792 / 0,767 |

**Por semilla, brazos nuevos** (media de 5 pliegues; 2026 / 2027 / 2028):

| Brazo | Val | `val_loss` | Prueba top-1 |
|---|---|---|---|
| T3 | 0,948 / 0,942 / 0,934 | 0,996 / 0,997 / 0,999 | 0,648 / 0,646 / 0,669 |
| T4 | 0,958 / 0,958 / 0,952 | 0,936 / 0,944 / 0,981 | 0,673 / 0,676 / 0,660 |
| T5 | 0,950 / 0,946 / 0,950 | 0,976 / 0,975 / 0,978 | 0,671 / 0,635 / 0,663 |

### 4.3 Costo (sin entrenamientos activos, 02:19)

Red con lote de 1: mediana de 100 llamadas por modelo, media de 3 modelos (semilla 2026, pliegues 0-2), en orden
intercalado. «Entrada»: armar una ventana desde el *processed* (regla → 30 pasos → velocidades), mediana de 5
videos.

| Brazo | ms `predict` lote 1 | ms `model(x)` | ms entrada | ms total (`predict` + entrada) | MB en disco |
|---|---:|---:|---:|---:|---:|
| R0 | 13,07 | 60,93 | 0,15 | 13,2 | 8,30 |
| T0 | 11,62 | 7,21 | 0,150 | 11,77 | 0,738 |
| T1 | 11,62 | 7,21 | 0,277 | 11,89 | 0,738 |
| T2 | 11,60 | 7,19 | 0,288 | 11,89 | 0,621 |
| T3 | 11,60 | 7,17 | 0,165 | 11,76 | 0,621 |
| T4 | 11,64 | 7,21 | 0,054 | 11,69 | 0,738 |
| T5 | 12,22* | 7,46* | 0,066 | 12,29* | 0,621 |

\* Uno de los 3 modelos de T5 midió 13,45 ms; los otros dos, 11,58 y 11,63 (mediana de T5 = 11,63). Es ruido de
medición, no un costo de la entrada de 132.

**Diferencia de costo 252 → 132** (para la regla 3):

| Preprocesamiento | Δ latencia total | Δ disco |
|---|---:|---:|
| mascara | −0,01 ms | −0,12 MB |
| imputación | −0,01 ms | −0,12 MB |
| pivote | +0,60 ms (por el valor atípico) | −0,12 MB |

En los tres casos es **despreciable** (< 1 ms y < 0,5 MB), así que la regla 3 pone 252 antes que 132.

Hallazgo no previsto: armar la entrada con el **pivote** es lo más barato (0,05 ms), porque solo indexa
fotogramas y no interpola. La imputación es lo más caro (0,28 ms). Todas las diferencias están por debajo de
0,3 ms frente a unos 12 ms de la red. El orden de simplicidad de la regla 3 se mantiene tal como se declaró: se
justificó por complejidad del código en vivo, no por milisegundos.

### 4.4 Aplicación del criterio prerregistrado

**Margen.** Las varianzas entre semillas de la validación son: T0 9,3·10⁻⁶, T1 2,5·10⁻⁵, T2 2,3·10⁻⁴,
T3 4,9·10⁻⁵, T4 1,2·10⁻⁵ y T5 5,3·10⁻⁶. La σ agrupada es 0,0074, menor que el piso, así que **margen = 0,010**.
La σ_loss agrupada es **0,0245**.

**Regla 1 (top-1 de validación; 30 pares ordenados).** Solo un par cumple las dos condiciones:

| Par | Δk por pliegue | Media de Δ | Pliegues mejor | ¿Cumple? |
|---|---|---:|:-:|:-:|
| **T4 > T3** | +0,010 / +0,037 / +0,023 / +0,007 / −0,003 | **+0,0147** | 4 | **sí** |
| T4 > T2 | | +0,0113 | 3 | no (solo 3 de 5) |
| T1 > T3 | | +0,0080 | 5 | no (Δ < margen) |
| T4 > T0 | | +0,0073 | 4 | no (Δ < margen) |
| T4 > T5 | | +0,0073 | 3 | no |
| T4 > T1 | | +0,0067 | 3 | no |
| T1 > T0 | | +0,0007 | 1 | no |

**Candidatos de la regla 1:** T0, T1, T2, T4 y T5. T3 queda fuera porque T4 lo supera. No hay ganador único:
**empate práctico**.

**Regla 2 (`val_loss` entre los 5 candidatos; σ_loss = 0,0245).** Solo un par cumple:

| Par (A supera en pérdida a B) | Reducción por pliegue (loss B − loss A) | Media | Pliegues con A menor | ¿Cumple? |
|---|---|---:|:-:|:-:|
| **T4 > T0** | +0,006 / +0,056 / +0,111 / −0,016 / +0,033 | **0,0379** | 4 | **sí** |
| T4 > T2 | | 0,0361 | 3 | no (3 de 5) |
| T4 > T1 | −0,017 / +0,062 / +0,101 / +0,018 / −0,044 | 0,0238 | 3 | no |
| T4 > T5 | | 0,0228 | 3 | no |
| T5 > T0 | | 0,0150 | 4 | no (media < σ_loss) |
| T1 > T0 | +0,023 / −0,006 / +0,010 / −0,034 / +0,077 | 0,0140 | 3 | no |
| T5 > T1 | | 0,0010 | 4 | no (media < σ_loss) |

**Candidatos de la regla 2:** T1, T2, T4 y T5. T0 queda fuera porque T4 lo supera en pérdida. Sigue sin haber
ganador único.

**Regla 3 (simplicidad).** Ningún candidato usa «mascara»; la imputación va antes que el pivote, y dentro de la
imputación va 252 antes que 132 (costo despreciable). El orden queda **T1 > T2 > T4 > T5**.

**Configuración recomendada según el criterio: T1 = TCN + imputación + 252** (decidida en el paso 3).

**Cómo se llega a T1, dicho con claridad:**

1. T1 no le gana directamente a ningún brazo, ni en validación ni en `val_loss`.
2. Queda elegido porque T4 (pivote + 252) elimina a T0 en la regla 2 sin superar a nadie más. Entonces T1 es el
   más simple de los que siguen en pie.
3. **Sensibilidad.** Si T4 no hubiera superado a T0 en `val_loss` (4 de 5 pliegues, reducción 0,038 frente a
   σ_loss = 0,0245), la regla 3 habría elegido **T0**. La diferencia entre T0 y T1 no la decide la validación
   por sí misma.
4. **Otra lectura de «los mejores brazos» da el mismo resultado.** Si se toman como candidatos el brazo con
   mejor media de validación (T4) y los que T4 no supera, salen T0, T1, T2, T4 y T5. En la regla 2, T4 solo
   elimina a T0. Queda otra vez T1.

### 4.5 Pruebas estadísticas (descriptivas)

**Signos y Wilcoxon por pliegue (n = 5).** Con 5 pliegues, el p bilateral mínimo de ambas pruebas es
2 · (1/2)⁵ = **0,0625**. Solo se alcanza si los 5 pliegues van en el mismo sentido, así que ninguna de las dos
puede llegar a 0,05. Formato: pliegues a favor / en contra / empates; p de signos; p de Wilcoxon. En
`val_loss`, «a favor» es menor pérdida.

| Comparación | Val top-1 | `val_loss` | Prueba top-1 (Δ medio) | Prueba F1 (Δ medio) |
|---|---|---|---|---|
| T1 − T0 | 1/2/2; 1,0; 1,0 | 3/2; 1,0; 0,63 | 4/1; 0,375; 0,19 (+0,026) | 4/1; 0,375; 0,19 (+0,028) |
| T4 − T0 | 4/1; 0,375; 0,31 | 4/1; 0,375; 0,19 | 4/1; 0,375; 0,31 (+0,019) | 4/1; 0,375; 0,19 (+0,020) |
| T3 − T0 | 1/4; 0,375; 0,19 | 2/3; 1,0; 0,81 | 4/1; 0,375; 0,63 (+0,003) | 3/2; 1,0; 0,63 (+0,005) |
| T4 − T1 | 3/2; 1,0; 0,44 | 3/2; 1,0; 0,44 | 2/3; 1,0; 0,44 (−0,007) | 2/3; 1,0; 0,63 (−0,008) |
| T2 − T1 | 0/4/1; 0,125; 0,125 | 2/3; 1,0; 0,44 | 2/3; 1,0; 0,63 (−0,024) | 2/3; 1,0; 0,63 (−0,025) |
| T3 − T1 | **0/5**; 0,0625; 0,0625 | **0/5**; 0,0625; 0,0625 | **0/5**; 0,0625; 0,0625 (−0,023) | 1/4; 0,375; 0,125 (−0,023) |
| T5 − T1 | 2/2/1; 1,0; 1,0 | 4/1; 0,375; 0,63 | 1/4; 0,375; 0,19 (−0,021) | 2/3; 1,0; 0,31 (−0,018) |
| T5 − T4 | 1/3/1; 0,625; 0,31 | 2/3; 1,0; 0,44 | 1/4; 0,375; 0,44 (−0,013) | 1/4; 0,375; 0,44 (−0,011) |
| T4 − T3 | 4/1; 0,375; 0,125 | 4/1; 0,375; 0,125 | 4/1; 0,375; 0,31 (+0,016) | 4/1; 0,375; 0,44 (+0,015) |
| T1 − R0 | 5/0; 0,0625; 0,0625 | — | **5/0**; 0,0625; 0,0625 (+0,088) | 5/0; 0,0625; 0,0625 (+0,103) |
| T4 − R0 | 5/0; 0,0625; 0,0625 | — | 5/0; 0,0625; 0,0625 (+0,081) | 5/0; 0,0625; 0,0625 (+0,095) |

Los 21 pares completos, con top-3, están en el JSON (`pruebas_pareadas_por_pliegue`).

**McNemar exacto sobre las 997 predicciones de prueba agregadas.** Formato: aciertos brazo / referencia;
aciertos exclusivos del brazo / de la referencia; p bilateral.

| Comparación | Semilla 2026 | Voto de las 3 semillas |
|---|---|---|
| T1 vs T0 | 686 / 655; 107 / 76; **p = 0,026** | 714 / 685; 88 / 59; **p = 0,021** |
| T2 vs T0 | 662 / 655; p = 0,67 | 688 / 685; p = 0,87 |
| T3 vs T0 | 646 / 655; p = 0,56 | 692 / 685; p = 0,58 |
| T4 vs T0 | 671 / 655; 99 / 83; p = 0,27 | 700 / 685; 92 / 77; p = 0,28 |
| T5 vs T0 | 669 / 655; p = 0,34 | 685 / 685; p = 1,0 |
| T2 vs T1 | 662 / 686; p = 0,11 | 688 / 714; 60 / 86; **p = 0,038** (T2 peor) |
| T3 vs T1 | 646 / 686; 94 / 134; **p = 0,0097** (T3 peor) | 692 / 714; p = 0,092 |
| T4 vs T1 | 671 / 686; 68 / 83; p = 0,26 | 700 / 714; 52 / 66; p = 0,23 |
| T5 vs T1 | 669 / 686; p = 0,23 | 685 / 714; 45 / 74; **p = 0,010** (T5 peor) |
| T3 vs T2 | 646 / 662; p = 0,30 | 692 / 688; p = 0,81 |
| T4 vs T2 | 671 / 662; p = 0,55 | 700 / 688; p = 0,36 |
| T5 vs T2 | 669 / 662; p = 0,63 | 685 / 688; p = 0,86 |
| T4 vs T3 | 671 / 646; 113 / 88; p = 0,090 | 700 / 692; p = 0,58 |
| T5 vs T3 | 669 / 646; p = 0,14 | 685 / 692; p = 0,63 |
| T5 vs T4 | 669 / 671; p = 0,94 | 685 / 700; p = 0,24 |
| T0 vs R0 | 655 / 578; p = 2·10⁻⁷ | 685 / 607; p = 5·10⁻⁸ |
| T1 vs R0 | 686 / 578; 177 / 69; p = 4·10⁻¹² | 714 / 607; p = 3·10⁻¹³ |
| T2 vs R0 | 662 / 578; p = 5·10⁻⁸ | 688 / 607; p = 1·10⁻⁸ |
| T3 vs R0 | 646 / 578; p = 2·10⁻⁵ | 692 / 607; p = 2·10⁻⁹ |
| T4 vs R0 | 671 / 578; 165 / 72; p = 1·10⁻⁹ | 700 / 607; p = 1·10⁻¹⁰ |
| T5 vs R0 | 669 / 578; p = 2·10⁻⁹ | 685 / 607; p = 3·10⁻⁸ |

El voto es por mayoría de la clase predicha; si las tres semillas discrepan, gana la semilla más confiada.
McNemar trata las 997 predicciones como independientes, pero están agrupadas en 5 señantes. Su p es más
optimista que el de las pruebas por pliegue.

### 4.6 Persona 0000 (pliegue 0), aparte

| Brazo | Val | `val_loss` | Prueba top-1: media (2026 / 2027 / 2028) | Top-3 | F1 |
|---|---:|---:|---|---:|---:|
| R0 | 0,923 | — | 0,452 (0,460 / 0,480 / 0,415) | 0,627 | 0,400 |
| T0 | 0,963 | 0,920 | 0,465 (0,495 / 0,425 / 0,475) | 0,652 | 0,438 |
| **T1** | 0,960 | 0,897 | **0,488** (0,540 / 0,390 / 0,535) | 0,660 | **0,469** |
| T2 | 0,957 | 0,914 | 0,408 (0,470 / 0,355 / 0,400) | 0,600 | 0,391 |
| T3 | 0,957 | 0,934 | 0,480 (0,535 / 0,460 / 0,445) | **0,667** | 0,453 |
| T4 | 0,967 | 0,915 | 0,457 (0,520 / 0,445 / 0,405) | 0,662 | 0,428 |
| T5 | **0,970** | **0,886** | 0,427 (0,470 / 0,360 / 0,450) | 0,592 | 0,421 |

- **T1 es el mejor con la persona 0000** (+2,3 puntos frente a T0 y +3,7 frente a R0), aunque con mucha
  dispersión entre semillas (0,390 a 0,540).
- **Las 6 velocidades empeoran a la 0000 cuando la mano ausente se rellena:**
  - con imputación, de 0,488 (T1) a 0,408 (T2);
  - con pivote, de 0,457 (T4) a 0,427 (T5).

  Con «mascara» no la empeoran: 0,465 (T0) frente a 0,480 (T3).
- **T5 tiene la mejor validación y la menor `val_loss` en el pliegue 0, pero queda 6 puntos por debajo de T1 en
  prueba.** Es el mismo patrón de los experimentos anteriores: en este pliegue, la validación no anticipa a la
  persona nueva.

### 4.7 Comparación con el canónico actual R0 (validación 0,920, prueba 0,589)

| Brazo | Δ val | Pliegues mejor en val | Δ prueba top-1 | Δ F1 | Pliegues mejor en prueba |
|---|---:|:-:|---:|---:|:-:|
| T0 | +0,029 | 5 | +0,062 | +0,075 | 5 |
| **T1** | +0,029 | 5 | **+0,088** | **+0,103** | 5 |
| T2 | +0,025 | 5 | +0,064 | +0,078 | 4 |
| T3 | +0,021 | 5 | +0,065 | +0,081 | 5 |
| T4 | **+0,036** | 5 | +0,081 | +0,095 | 5 |
| T5 | +0,029 | 5 | +0,067 | +0,085 | 4 |

Los seis brazos con TCN superan al canónico en validación en los 5 pliegues, con al menos el doble del margen.
T1 es el que más lo supera en prueba: +8,8 puntos de top-1 y +10,3 de F1, en los 5 pliegues; McNemar con la
semilla 2026 p = 4·10⁻¹².

---

## 5. Veredicto

**Según el criterio prerregistrado, la configuración recomendada para el modelo final es T1: TCN pequeña con
GAP + imputación de la mano ausente + 252 rasgos (126 posiciones + 126 velocidades).**

- **Paso 1 (top-1 de validación).** Empate práctico entre T0, T1, T2, T4 y T5. Solo T4 supera a T3.
- **Paso 2 (`val_loss`).** T4 supera a T0 (menor en 4 de 5 pliegues; reducción 0,038 > σ_loss = 0,0245). Nadie
  más supera a nadie. Quedan T1, T2, T4 y T5.
- **Paso 3 (simplicidad).** La imputación va antes que el pivote, y con costo despreciable va 252 antes que 132.
  Gana **T1**.

**¿Apunta la prueba a otra configuración? No.** La prueba LOSO, que es descriptiva, coincide con el criterio. T1
es el mejor de los seis en:

- prueba top-1 (0,677);
- top-3 (0,847);
- F1 macro (0,647);
- acierto del voto de las 3 semillas (0,716);
- la persona 0000 (0,488).

Además, es el único brazo con una ventaja de McNemar sobre T0 que se repite: p = 0,026 con la semilla 2026 y
p = 0,021 con el voto.

**Matices que el director debe conocer:**

1. **T1 no gana por mérito propio en validación.** Empata con T0 en exactitud (0,949 frente a 0,949) y su menor
   `val_loss` (0,978 frente a 0,992) no llega a la regla (3 de 5 pliegues). La regla lo elige porque T4 elimina a
   T0 en el paso 2. Sin ese resultado, el criterio habría elegido T0 por simplicidad.
2. **T4 (pivote + 252) es el mejor en validación:** top-1 0,956 y `val_loss` 0,954, ambos los mejores de los
   seis. Pero no supera a T1 por la regla (3 de 5 pliegues en ambas métricas) y en prueba queda segundo (0,670
   frente a 0,677; McNemar p = 0,26). Elegir T4 sería apartarse del criterio a favor de la validación pura.
   Eso implicaría implementar en `translator.py` la regla del pivote, más compleja: búsqueda de fotogramas con
   ventana y tope. A cambio, en prueba sería algo peor que T1 y le iría peor a la persona 0000 (0,457 frente a
   0,488).
3. **La conclusión de la confirmación queda corregida.** T2 (imputación + 132), que entonces se eligió por no
   inferioridad, queda por detrás de T1 en este criterio y en todos los indicadores de prueba. Las 6 velocidades
   no mejoran la validación con ningún preprocesamiento y empeoran a la persona 0000 cuando la mano se
   rellena. Su ahorro es de 0,12 MB y unos 0,01 ms.

**Este informe no cambia el modelo canónico, el umbral ni `translator.py`.** Adoptar T1 amplía el alcance del
anteproyecto (`EXPERIMENTO_ARQUITECTURAS_LITERATURA.md` §6.1), y la decisión es del director. Si se adopta, habría
que hacer lo siguiente:

- reentrenar el modelo final;
- implementar la imputación en `translator.py` sobre la ventana de cuadros, antes de remuestrear (0,28 ms por
  ventana);
- recalibrar el umbral de confianza (D-09) y el umbral anidado;
- volver a medir la latencia de punta a punta.

---

## 6. Cautelas

1. **Solo 5 señantes.** Ninguna prueba por pliegue puede bajar de p = 0,0625, y el McNemar ignora la agrupación
   por señante.
2. **La validación sigue casi saturada.** Las diferencias de top-1 entre brazos son de 1 a 1,5 muestras por
   pliegue. La `val_loss` separa algo más, pero su DE entre semillas (σ_loss = 0,0245) es del mismo orden que
   las diferencias entre brazos.
3. **La decisión depende de un solo par en la regla 2** (T4 > T0, §4.4). Se declaró de antemano cómo se
   eliminan candidatos, pero conviene saber que la frontera entre T0 y T1 es estrecha en validación. La prueba
   (descriptiva) sí separa a T1 de T0.
4. **La imputación en vivo no se probó.** En el corpus se imputa sobre el video completo. En vivo se imputaría
   sobre la ventana que abre la persona usuaria, y un hueco en el borde se rellena copiando el cuadro válido más
   cercano. Su efecto en vivo no se midió, y tampoco su interacción con la regla de `translator.py` que descarta
   ventanas con demasiadas manos ausentes.
5. **La validación es la del *checkpoint*** (máximo a lo largo de las épocas) y la `val_loss` es la de ese
   mismo *checkpoint*, elegido por exactitud y no por pérdida. El sesgo es el mismo para los seis brazos.
6. **T0, T1 y T2 reutilizados.** Se verificó con la reproducción byte a byte de T1 (semilla 2026, pliegue 0). La
   confirmación ya había reproducido T0 de la misma forma. Los demás pliegues no se reprodujeron, pero el
   entrenamiento es determinista en este equipo.
7. **Latencia.** Un modelo de T5 midió 13,45 ms (frente a unos 11,6 ms del resto). Es ruido de una sola
   medición; no cambia la regla 3, porque la diferencia sigue por debajo de 1 ms.

---

## 7. Archivos

- **Script nuevo:** `src/experimento_factorial_tcn.py`, con los subcomandos `fijar_val_loss`, `verificar`,
  `entrenar`, `inferencia` y `resumen`.
- **Pruebas:** `tests/test_factorial_tcn.py` (14 pruebas: criterio de 3 pasos, orden de simplicidad, regla de
  pérdida, entropía cruzada, entradas de los 6 brazos y pivote). Pasan junto con las 13 de
  `tests/test_confirmacion_tcn.py`.
- **Modelos y corridas:** `models/factorial_tcn_2026-10-03/`, con las carpetas `T3_mascara_132`,
  `T4_pivote_252` y `T5_pivote_132`. En cada una hay `semilla_S/pliegue_K/` con `modelo.keras`,
  `modelo.procedencia.json`, `resultado.json`, `predicciones.json`, `probabilidades.npz` e `historial.json`.
  Además: `val_loss_T0_T1_T2_fijada.json`, `verificacion.json`, `humo.json`, `inferencia.json`,
  `resultados_factorial_tcn.json`, `corrida_semilla_*.log` y `pids.txt`.
- **Datos crudos del informe:** `docs/EXPERIMENTO_FACTORIAL_TCN.json`.
- **Solo lectura:**
  - T0: `models/arquitecturas_literatura_2026-10-02/T_tcn/`;
  - T1 y T2: `models/confirmacion_tcn_2026-10-03/`;
  - R0: `models/experimento_mano_ausente_2026-10-02/A_mascara/`.

Reproducción:

```
python src/experimento_factorial_tcn.py fijar_val_loss
python src/experimento_factorial_tcn.py verificar --dest <scratch>/repro_t1
python src/experimento_factorial_tcn.py entrenar --semillas 2026      # una semilla por proceso
python src/experimento_factorial_tcn.py inferencia
python src/experimento_factorial_tcn.py resumen --docs
```

---

## Explicación simple

Probamos las 6 combinaciones posibles con la red pequeña (TCN): tres formas de tratar los cuadros donde no se
ve una mano (dejarlos en cero, rellenarlos o usar el cuadro real más cercano) por dos tamaños de entrada (126 o
6 velocidades). Como la exactitud de validación ya está pegada en unos 95 de cada 100, el desempate usó
también la pérdida de validación y, al final, la simplicidad. La regla fijada de antemano eligió **rellenar
los cuadros y mantener las 126 velocidades (T1)**. Esa misma versión es la que más acierta con personas
nuevas (68 % frente a 59 % del modelo actual) y también con la persona más difícil (0000). Reducir a 6
velocidades no ayuda en ninguna combinación. La decisión de adoptarla en el prototipo es del director.
