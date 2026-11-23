# Experimento: confirmación conjunta con la TCN (arquitectura + mano ausente + velocidades)

Fecha: 3 de octubre de 2026. Objetivo: decidir qué configuración llevar al modelo final, para discutirlo con el
director el lunes 5 de octubre a las 19:00.

**Pregunta.** Tres experimentos encontraron mejoras por separado, pero ninguna combinación se ha medido junta:

1. La **TCN pequeña con GAP** (brazo T de `docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md`) superó al canónico:
   validación 0,949, prueba top-1 0,651, F1 0,619.
2. La **imputación de la mano ausente** (brazo C de `docs/EXPERIMENTO_MANO_AUSENTE.md`), con la red canónica:
   validación 0,939, prueba 0,619.
3. Las **6 velocidades por coordenada** (brazo C de `docs/EXPERIMENTO_VELOCIDADES_REDUCIDAS.md`, 126 + 6 = 132),
   con la red canónica: no fueron peores (validación 0,923, prueba 0,589). Es una simplificación, no una mejora.

¿Se suman la imputación y la reducción de velocidades a la TCN, o alguna deja de servir con ella?

Este documento no cambia el modelo canónico, el umbral ni `translator.py`.

---

## 1. Diseño

Los tres brazos usan la arquitectura T (`CONFIG_T` exacta de `src/arquitecturas_literatura.py`: 4 bloques
residuales dilatados 1-2-4-8, 64 filtros, k = 3, GAP) y su receta de entrenamiento (Adam lr 1e-3 con
CosineDecay, *label smoothing* 0,1, lote 64, paciencia 25 sobre `val_accuracy`, máx. 200 épocas). Aumentación
×10 y 30 pasos. Datos: `data/processed_holistic_mascara_rostro`, representación «manos». LOSO de 5 pliegues
(D-06) con validación interna de 2 por seña. Semillas 2026, 2027 y 2028.

| Brazo | Mano ausente | Entrada | Parámetros |
|---|---|---|---:|
| **T0** (control) | «mascara» (D-07) | 126 posiciones + 126 velocidades = 252 | 156.402 |
| **T1** | imputación lineal por mano sobre los cuadros originales, antes de remuestrear | 252 | 156.402 |
| **T2** | imputación (igual que T1) | 126 posiciones + 6 velocidades por coordenada = 132 | 125.682 |

- **T0 no se reentrena.** Se reutilizan las 15 corridas de `models/arquitecturas_literatura_2026-10-02/T_tcn`.
- **T1 y T2** se entrenan con `src/experimento_confirmacion_tcn.py`, que no reimplementa nada:
  - entrenamiento: `experimento_arquitecturas.entrenar_corrida` con el constructor «T»;
  - imputación: `mano_ausente_variantes.remuestrear_todas_con_regla(..., "imputacion", 30)`;
  - velocidades reducidas: `velocidades_reducidas.funcion_entrada("por_coordenada")`. Sustituye a
    `add_velocity` como `canon[2]`, así que se calcula **después** de la aumentación y del remuestreo, como en su
    experimento.

### 1.1 Verificación previa (2-oct 23:57 a 3-oct 00:00)

`python src/experimento_confirmacion_tcn.py verificar --dest <carpeta_temporal>/repro_t0` comprobó lo siguiente:

- **Configuración de T0.** Las 15 corridas de `T_tcn` tienen exactamente `CONFIG_T`, lr 1e-3, lote 64,
  *label smoothing* 0,1, 200 épocas, paciencia 25, aumentación ×10, 30 pasos, `processed_holistic_mascara_rostro`,
  «manos», «mascara», 252 rasgos y 156.402 parámetros.
- **Reproducción de T0.** Se reentrenó la semilla 2026, pliegue 0, con el mismo código que entrena T1 y T2, en
  el *carpeta_temporal*. Resultado: `predicciones.json` **idéntico byte a byte**; probabilidades de validación y de
  prueba con diferencia máxima 0; historial idéntico; mismas 55 épocas (mejor en la 30); validación 0,980 y
  prueba 0,495 iguales. Por eso T0 se reutiliza y no se reentrena.
- **Entradas.** La `X_seq` «mascara» de este script es, bit a bit, la de `experimento_arquitecturas.preparar`. La
  imputación cambia la entrada de 735 de los 1000 videos. T0 y T1 usan `add_velocity` (252); T2 usa 132. Las
  posiciones no cambian.

### 1.2 Prueba de humo (3-oct, 00:00)

En el *carpeta_temporal*, no en `models/`: T1 y T2 con la semilla 2026, el pliegue 0 y 3 épocas. Los modelos se
guardan, se recargan con la entrada correcta (30 × 252 y 30 × 132) y su procedencia queda corregida (regla
«imputacion», velocidades «por_coordenada» en T2). Las cifras de la prueba de humo no se usan.

---

## 2. Criterio prerregistrado (escrito el 3-oct-2026 a las 00:01, hora de Colombia, antes de las corridas completas)

No se ha visto ningún resultado completo de T1 ni de T2. Solo la prueba de humo de 3 épocas.

1. **Solo decide la validación:** el top-1 medio de validación de los 5 pliegues. Para cada brazo V y pliegue
   k, val(V, k) es la media de las 3 semillas del `val_accuracy` máximo (el del *checkpoint*). La prueba LOSO
   (top-1, top-3, F1 macro) se reporta como descripción y **no decide nada**.
2. **Margen** = máx(σ, 0,01). σ es la desviación estándar entre semillas, agrupada sobre T0, T1 y T2: para
   cada brazo, la DE (ddof = 1) entre las 3 semillas de la validación media de 5 pliegues; luego, la raíz de
   la media de las tres varianzas.
   - Las Δk se redondean a 6 decimales (exactitudes float32 de Keras).
   - Las comparaciones con el margen usan una tolerancia de 1e-9. Un Δk de −0,0100 exacto no cuenta como
     «pierde por más del margen», y un Δ medio igual al margen no cuenta como «mayor que el margen».
3. **Paso 1: T1 frente a T0.** Δk = val(T1, k) − val(T0, k).
   - **T1 reemplaza a T0** si (a) la media de Δk es mayor que el margen **y** (b) Δk > 0 en al menos 4 de
     los 5 pliegues (Δk = 0 no cuenta como mejora).
   - **Si T1 no cumple, pero tampoco es peor,** se mantiene T0 y se informa que **la imputación es neutra con
     la TCN**. «No es peor» significa: media de Δk ≥ −margen, y Δk < −margen en a lo sumo 1 pliegue.
   - **Si T1 es peor** (no cumple lo anterior), se mantiene T0 y se informa que la imputación empeora con la
     TCN.
4. **Paso 2: T2 como no inferioridad.** La referencia es el brazo que retiene el paso 1: T1 si reemplazó a T0;
   si no, T0.
   - Δk = val(T2, k) − val(referencia, k).
   - **T2 se adopta** si la media de Δk ≥ −margen **y** Δk < −margen en a lo sumo 1 de los 5 pliegues. Es la
     regla de `EXPERIMENTO_VELOCIDADES_REDUCIDAS.md`.
   - Como información, sin valor de decisión, se reporta si T2 cumpliría además la regla de mejora (paso 1).
5. **Resultado final.** Si T2 se adopta, la configuración recomendada es T2. Si no, es la referencia del
   paso 2.
   - **Declaración de antemano:** si la referencia es T0 y T2 se adopta, la imputación entra al modelo final
     como parte de T2. Ese brazo no puede separar el efecto de la imputación del de la reducción de
     velocidades (no hay brazo «mascara + 132»). En ese caso se informa además, como descripción, T2 frente
     a T1.
6. **Pruebas que se reportan** (descriptivas; no cambian la regla):
   - **McNemar exacto** frente a T0 sobre las 997 predicciones de prueba agregadas de cada brazo: con la
     semilla 2026 (principal), con 2027 y 2028, y con el **voto de las 3 semillas**. El voto es por mayoría
     de la clase predicha; si las tres semillas discrepan, gana la predicción de la semilla con mayor
     probabilidad predicha.
   - **Signos y Wilcoxon** por pliegue, en validación y en prueba (top-1, top-3, F1). Con 5 pliegues, el p
     bilateral mínimo de ambas es 2/2⁵ = **0,0625**: ninguna puede llegar a 0,05.
7. **La persona 0000 (pliegue 0) se reporta aparte.**
8. **Comparación con el canónico actual R0** (CNN + BiLSTM + atención, «mascara», 252; validación 0,920,
   prueba 0,589; brazo A de `models/experimento_mano_ausente_2026-10-02`). Es descriptiva y usa las mismas
   pruebas: Δ por pliegue, signos, Wilcoxon y McNemar.
9. **Costo.** Se reportan los parámetros, la latencia con lote de 1 (como `translator.py`) medida sin
   entrenamientos activos y el costo de construir la entrada de una ventana con cada regla.
10. **Plan de ejecución.** Tres procesos en paralelo, uno por semilla; cada uno recorre los pliegues 0 a 4 y,
    dentro de cada pliegue, T1 → T2. Son 30 corridas. Si no terminaran, el criterio se aplicaría con las
    semillas que tengan los 5 pliegues en los tres brazos (mínimo 2), y se declararía.

---

## 3. Ejecución

- **Lanzamiento:** 3-oct a las 00:00:58, en 3 procesos en paralelo, uno por semilla, cada uno en el orden
  pliegue 0 → 4 y T1 → T2. Terminaron a las 00:20-00:21. Son 30 corridas sin errores ni recortes, y el
  criterio se aplica con las 3 semillas completas.
- **Latencia:** se midió a las 00:22, sin ningún proceso de entrenamiento activo y con la CPU un 93 % libre.
- **Datos crudos:** `docs/EXPERIMENTO_CONFIRMACION_TCN.json` (copia en
  `models/confirmacion_tcn_2026-10-03/resultados_confirmacion_tcn.json`).
- **Corridas:** en `models/confirmacion_tcn_2026-10-03/` están los modelos, predicciones, probabilidades,
  historiales y *logs*.
- **Validación del *checkpoint*:** en las 30 corridas nuevas, el `val_accuracy` máximo coincide con la
  exactitud del modelo recargado sobre la validación.
- **Criterio:** no se modificó después de ver resultados.

---

## 4. Resultados

### 4.1 Tabla resumen

Media de 3 semillas × 5 pliegues. La DE es entre los 5 pliegues (medias de semillas). La validación decide;
la prueba solo describe. R0 es el canónico actual (referencia, no entra en el criterio).

| Brazo | Parámetros | **Val top-1** (DE) | Prueba top-1 (DE) | Top-3 | F1 macro | Persona 0000: val / prueba | ms `predict` lote 1 | ms `model(x)` | ms entrada | Acierto voto 3 semillas |
|---|---:|---:|---:|---:|---:|:-:|---:|---:|---:|---:|
| R0 canónico | 714.291 | 0,920 (0,020) | 0,589 (0,094) | 0,764 | 0,544 | 0,923 / 0,452 | 13,4 | 61,3 | 0,15 | 0,609 |
| **T0** TCN + «mascara» + 252 | 156.402 | **0,949** (0,019) | 0,651 (0,117) | 0,820 | 0,619 | 0,963 / 0,465 | 11,9 | 7,3 | 0,15 | 0,687 |
| **T1** TCN + imputación + 252 | 156.402 | **0,949** (0,017) | **0,677** (0,127) | **0,847** | **0,647** | 0,960 / **0,488** | 11,8 | 7,3 | 0,37 | **0,716** |
| **T2** TCN + imputación + 132 | **125.682** | **0,945** (0,020) | 0,653 (0,147) | 0,835 | 0,622 | 0,957 / **0,408** | 11,6 | 7,2 | 0,39 | 0,690 |

Notas:

- **Latencia:** mediana de 3 modelos (semilla 2026, pliegues 0-2) × 100 llamadas, en orden intercalado R0,
  T0, T1, T2. Es solo la red, sin MediaPipe.
- **«ms entrada»:** costo de llevar una ventana del *processed* a la entrada de la red con la regla del brazo
  (video «hola» `0030_0000_0003`, 69 cuadros): remuestreo, más la imputación en T1 y T2, más las velocidades.
  La imputación añade unos 0,22 ms por ventana.
- **Tamaño en disco:** R0 8,3 MB · T0 y T1 0,74 MB · T2 0,62 MB.
- **Entrenamiento por pliegue** (3 procesos en paralelo): T0 128 s (2,15 s/época), T1 123 s (2,16 s/época),
  T2 113 s (2,10 s/época); R0 255 s en su experimento original.

### 4.2 Por pliegue (media de las 3 semillas)

**Validación top-1 (decide):**

| Pliegue (señante) | R0 | T0 | T1 | T2 |
|---|---:|---:|---:|---:|
| 0 (0000) | 0,923 | 0,963 | 0,960 | 0,957 |
| 1 (0001) | 0,893 | 0,940 | 0,937 | 0,923 |
| 2 (0002) | 0,917 | 0,937 | 0,937 | 0,933 |
| 3 (0003) | 0,950 | 0,973 | 0,973 | 0,973 |
| 4 (0004) | 0,917 | 0,930 | 0,940 | 0,937 |
| **Media** | 0,920 | 0,949 | 0,949 | 0,945 |

**Prueba top-1 (descriptiva):**

| Pliegue (señante) | R0 | T0 | T1 | T2 |
|---|---:|---:|---:|---:|
| 0 (0000) | 0,452 | 0,465 | **0,488** | 0,408 |
| 1 (0001) | 0,536 | 0,628 | 0,611 | **0,630** |
| 2 (0002) | 0,631 | 0,663 | **0,721** | 0,717 |
| 3 (0003) | 0,640 | 0,753 | 0,770 | **0,775** |
| 4 (0004) | 0,687 | 0,745 | **0,795** | 0,733 |
| **Media** | 0,589 | 0,651 | **0,677** | 0,653 |

**Top-3 y F1 macro de prueba por pliegue (descriptivas):**

| Pliegue | Top-3 R0 / T0 / T1 / T2 | F1 R0 / T0 / T1 / T2 |
|---|---|---|
| 0 | 0,627 / 0,652 / 0,660 / 0,600 | 0,400 / 0,438 / 0,469 / 0,391 |
| 1 | 0,715 / 0,817 / 0,807 / 0,802 | 0,472 / 0,590 / 0,565 / 0,586 |
| 2 | 0,786 / 0,840 / 0,889 / 0,919 | 0,604 / 0,638 / 0,685 / 0,695 |
| 3 | 0,827 / 0,890 / 0,935 / 0,923 | 0,603 / 0,722 / 0,745 / 0,736 |
| 4 | 0,863 / 0,900 / 0,942 / 0,928 | 0,639 / 0,708 / 0,770 / 0,700 |

**Por semilla** (media de 5 pliegues):

| Brazo | Val 2026 / 2027 / 2028 | Prueba top-1 2026 / 2027 / 2028 | F1 2026 / 2027 / 2028 |
|---|---|---|---|
| R0 | 0,932 / 0,918 / 0,910 | 0,580 / 0,592 / 0,596 | 0,537 / 0,539 / 0,554 |
| T0 | 0,948 / 0,946 / 0,952 | 0,657 / 0,645 / 0,651 | 0,624 / 0,619 / 0,614 |
| T1 | 0,950 / 0,954 / 0,944 | 0,688 / 0,658 / 0,685 | 0,658 / 0,630 / 0,653 |
| T2 | 0,962 / 0,934 / 0,938 | 0,664 / 0,652 / 0,642 | 0,630 / 0,627 / 0,608 |

### 4.3 Aplicación del criterio prerregistrado

**Margen.** Las varianzas entre semillas son: T0 9,3·10⁻⁶, T1 2,5·10⁻⁵ y T2 2,3·10⁻⁴. La σ agrupada es 0,0094.
Como es menor que 0,01, **margen = 0,010**.

**Paso 1 · T1 frente a T0.**

| Δk de validación por pliegue | Media de Δ | (a) Δ > 0,010 | (b) mejor en ≥ 4 de 5 | No es peor |
|---|---:|:-:|:-:|:-:|
| −0,003 / −0,003 / 0,000 / 0,000 / +0,010 | +0,0007 | no | no (1 mejor, 2 peor, 2 empates) | sí (ningún pliegue pierde por más del margen) |

**Resultado: se mantiene T0. La imputación es neutra con la TCN en validación.** La referencia para T2 es T0.

**Paso 2 · T2 frente a T0 (no inferioridad).**

| Δk de validación por pliegue | Media de Δ | (a) Δ ≥ −0,010 | Pliegues que pierden por más del margen | (b) ≤ 1 |
|---|---:|:-:|:-:|:-:|
| −0,007 / **−0,017** / −0,003 / 0,000 / +0,007 | −0,004 | sí | 1 (pliegue 1) | sí |

**Resultado: T2 se adopta (no es peor que T0).** Según la regla informativa de mejora, T2 no mejora a T0: es
mejor en 1 pliegue y peor en 3.

**Configuración final según el criterio: T2 (TCN + imputación + 126 posiciones + 6 velocidades por
coordenada).** Como se declaró de antemano (§2.5), la imputación entra como parte del paquete T2 y no por mérito
propio en validación.

### 4.4 Pruebas estadísticas (descriptivas)

**Signos y Wilcoxon por pliegue (n = 5).** Con 5 pliegues, el p bilateral mínimo de ambas pruebas es
2 · (1/2)⁵ = **0,0625**. Solo se alcanza si los 5 pliegues van en el mismo sentido, y ninguna de las dos
puede llegar a 0,05. Formato: pliegues a favor / en contra / empates; p de signos; p de Wilcoxon.

| Comparación | Validación | Prueba top-1 | Prueba top-3 | Prueba F1 |
|---|---|---|---|---|
| T1 − T0 | 1 / 2 / 2; 1,0; 1,0 | **4 / 1** (+0,023 / −0,017 / +0,057 / +0,017 / +0,050); 0,375; 0,19 | 4 / 1; 0,375; 0,19 | 4 / 1; 0,375; 0,19 |
| T2 − T0 | 1 / 3 / 1; 0,625; 0,44 | 3 / 2 (−0,057 / +0,002 / +0,054 / +0,022 / −0,012); 1,0; 1,0 | 3 / 2; 1,0; 0,63 | 2 / 3; 1,0; 1,0 |
| T2 − T1 | 0 / 4 / 1; 0,125; 0,125 | 2 / 3 (−0,080 / +0,018 / −0,003 / +0,005 / −0,062); 1,0; 0,63 | — | 2 / 3; 1,0; 0,63 |
| T0 − R0 | **5 / 0**; 0,0625; 0,0625 | **5 / 0**; 0,0625; 0,0625 | 5 / 0; 0,0625 | 5 / 0; 0,0625 |
| T1 − R0 | **5 / 0**; 0,0625; 0,0625 | **5 / 0** (+0,037 / +0,075 / +0,089 / +0,130 / +0,108); 0,0625 | 5 / 0; 0,0625 | 5 / 0; 0,0625 |
| T2 − R0 | **5 / 0**; 0,0625; 0,0625 | 4 / 1 (−0,043 en el pliegue 0); 0,375; 0,125 | 4 / 1; 0,375; 0,125 | 4 / 1; 0,375; 0,125 |

**McNemar exacto sobre las 997 predicciones de prueba agregadas.** Formato: aciertos brazo / referencia;
aciertos exclusivos del brazo / de la referencia; p bilateral.

| Comparación | Semilla 2026 (principal) | 2027 | 2028 | Voto de las 3 semillas |
|---|---|---|---|---|
| T1 vs T0 | 686 / 655; 107 / 76; **p = 0,026** | 656 / 643; p = 0,39 | 683 / 649; p = 0,013 | 714 / 685; 88 / 59; **p = 0,021** |
| T2 vs T0 | 662 / 655; 100 / 93; p = 0,67 | 650 / 643; p = 0,68 | 640 / 649; p = 0,53 | 688 / 685; 82 / 79; p = 0,87 |
| T2 vs T1 | 662 / 686; 89 / 113; p = 0,11 | — | — | 688 / 714; 60 / 86; **p = 0,038** (T2 peor) |
| T0 vs R0 | 655 / 578; 148 / 71; p = 2·10⁻⁷ | p = 6·10⁻⁴ | p = 3·10⁻⁴ | 685 / 607; p = 5·10⁻⁸ |
| T1 vs R0 | 686 / 578; 177 / 69; p = 4·10⁻¹² | p = 2·10⁻⁵ | p = 7·10⁻⁹ | 714 / 607; p = 3·10⁻¹³ |
| T2 vs R0 | 662 / 578; 161 / 77; p = 5·10⁻⁸ | p = 2·10⁻⁴ | p = 2·10⁻³ | 688 / 607; p = 1·10⁻⁸ |

El voto es por mayoría de la clase predicha; si las tres semillas discrepan, gana la semilla más confiada.
McNemar trata las 997 predicciones como independientes, pero están agrupadas en 5 señantes. Su p es más
optimista que el de las pruebas por pliegue.

### 4.5 Persona 0000 (pliegue 0), aparte

| Brazo | Val | Prueba top-1: media (2026 / 2027 / 2028) | Top-3 | F1 |
|---|---:|---|---:|---:|
| R0 | 0,923 | 0,452 (0,460 / 0,480 / 0,415) | 0,627 | 0,400 |
| T0 | 0,963 | 0,465 (0,495 / 0,425 / 0,475) | 0,652 | 0,438 |
| T1 | 0,960 | **0,488** (0,540 / 0,390 / 0,535) | 0,660 | 0,469 |
| T2 | 0,957 | **0,408** (0,470 / 0,355 / 0,400) | 0,600 | 0,391 |

- **T1 no empeora a la persona 0000:** +2,3 puntos frente a T0, aunque con mucha dispersión entre semillas
  (0,390 a 0,540).
- **T2 sí la empeora:** −5,7 puntos frente a T0 y −4,4 frente al canónico R0. Es el único pliegue en el que T2
  queda por debajo de R0 en prueba, y es peor que T0 en las 3 semillas (0,470 frente a 0,495; 0,355 frente a
  0,425; 0,400 frente a 0,475). Vuelve el patrón de los experimentos anteriores: igual validación en el
  pliegue 0 (0,957 frente a 0,963) y menos acierto con la persona 0000.

### 4.6 Comparación con el canónico actual R0 (validación 0,920, prueba 0,589)

| Brazo | Δ val | Pliegues mejor en val | Δ prueba top-1 | Δ F1 | Pliegues mejor en prueba |
|---|---:|:-:|---:|---:|:-:|
| T0 | +0,029 | 5 | +0,062 | +0,075 | 5 |
| T1 | +0,029 | 5 | **+0,088** | **+0,103** | 5 |
| T2 | +0,025 | 5 | +0,064 | +0,078 | 4 (pierde con 0000) |

Los tres brazos con TCN superan al canónico en validación en los 5 pliegues, con más del doble del margen.

---

## 5. Veredicto

**Según el criterio prerregistrado, la configuración recomendada es T2: TCN + imputación + 132 rasgos
(126 posiciones + 6 velocidades por coordenada).**

- **Paso 1:** la imputación es **neutra con la TCN** en validación (0,949 frente a 0,949; Δ = +0,0007). T1 no
  reemplaza a T0.
- **Paso 2:** T2 **no es peor** que T0 (Δ = −0,004 y solo pierde por más del margen en el pliegue 1). Por eso
  se adopta.
- **Justificación:** con la misma exactitud de validación dentro del margen, T2 usa una entrada de 132 en vez
  de 252 rasgos y tiene un 20 % menos de parámetros que T0 (125.682 frente a 156.402). Además, es 5,7 veces
  más pequeño que el canónico y su red corre unas 8 veces más rápido (7,2 frente a 61,3 ms).

**Lo que el director debe sopesar antes de adoptarla.** Es descriptivo y no cambia el criterio:

1. **La validación está saturada.** Los tres brazos rondan 0,95 con 100 muestras por pliegue, donde una muestra
   vale 0,01. A ese nivel, la validación ya casi no distingue entre brazos.
2. **En prueba, T1 es el mejor de los tres:**
   - top-1 0,677 frente a 0,651 de T0 (mejor en 4 de 5 pliegues);
   - McNemar frente a T0: p = 0,026 con la semilla 2026 y p = 0,021 con el voto de las 3 semillas;
   - frente al canónico: +8,8 puntos de top-1 y +10,3 de F1.
3. **T2 pierde parte de esa ganancia.** Frente a T1 es peor en validación en 4 de 5 pliegues (1 empate), y en
   prueba con el voto de las 3 semillas (McNemar p = 0,038).
4. **T2 empeora a la persona 0000:** 0,408 frente a 0,465 de T0 y 0,452 de R0.
5. **El ahorro de T2 es pequeño en la práctica:** 0,1 ms de red, 0,12 MB en disco y 30.720 parámetros.

Si el director considera que la prueba LOSO y la persona 0000 deben pesar, **T1** es la alternativa
respaldada por los datos descriptivos. Elegirla sería una decisión fuera del criterio prerregistrado, y así
habría que declararlo.

**Este informe no cambia el modelo canónico, el umbral ni `translator.py`.** Adoptar cualquier brazo con TCN
amplía el alcance del anteproyecto (`EXPERIMENTO_ARQUITECTURAS_LITERATURA.md` §6.1), y la decisión es del
director. Si se adopta T2 (o T1), habría que hacer lo siguiente:

- reentrenar el modelo final;
- implementar la imputación en `translator.py` sobre la ventana de cuadros antes de remuestrear (y, en T2, las 6
  velocidades);
- recalibrar el umbral de confianza (D-09) y el umbral anidado;
- volver a medir la latencia de punta a punta.

---

## 6. Cautelas

1. **Solo 5 señantes.** Ninguna prueba por pliegue puede bajar de p = 0,0625. El McNemar ignora la agrupación
   por señante.
2. **La imputación en vivo no se probó.** En el corpus se imputa sobre el video completo. `translator.py`
   infiere sobre una ventana fija que abre la persona usuaria, así que la imputación podría aplicarse igual,
   sobre la ventana completa y al cerrarse. Pero la ventana en vivo puede cortar la seña de otro modo que el
   video del corpus. Por ejemplo, un hueco en el borde se rellena copiando el cuadro válido más cercano. Su
   efecto en vivo no se midió, y tampoco su interacción con la regla de `translator.py` que descarta
   ventanas con demasiadas manos ausentes.
3. **El efecto de la imputación y el de las 6 velocidades no se pueden separar en T2.** No hubo brazo
   «mascara + 132» con la TCN (se declaró en §2.5). La comparación T2 − T1 sugiere que, con la TCN, quitar
   las 120 velocidades restantes cuesta algo en prueba.
4. **Dispersión de T2 entre semillas.** Su validación por semilla es 0,962 / 0,934 / 0,938 (DE 0,015,
   frente a 0,003 de T0). Esto no cambió el margen, que quedó en su piso de 0,010.
5. **La validación es la del *checkpoint***: el máximo a lo largo de las épocas. El sesgo optimista es el mismo
   para los tres brazos.
6. **T0 reutilizado.** Se verificó con la reproducción bit a bit de la semilla 2026, pliegue 0 (§1.1). Los
   otros 14 pliegues no se reprodujeron, pero el entrenamiento es determinista en este equipo.

---

## 7. Archivos

- **Script nuevo:** `src/experimento_confirmacion_tcn.py` (verificar, entrenar, inferencia y resumen).
- **Pruebas:** `tests/test_confirmacion_tcn.py` (criterio, voto, entradas de los brazos y orden de la
  imputación; 13 pruebas).
- **Modelos y corridas:** `models/confirmacion_tcn_2026-10-03/`, con las carpetas `T1_imputacion_252` y
  `T2_imputacion_132`. En cada una hay `semilla_S/pliegue_K/` con `modelo.keras`, `modelo.procedencia.json`,
  `resultado.json`, `predicciones.json`, `probabilidades.npz` e `historial.json`. Además: `verificacion.json`,
  `humo.json`, `inferencia.json`, `resultados_confirmacion_tcn.json`, `corrida_semilla_*.log` y `pids.txt`.
- **Datos crudos del informe:** `docs/EXPERIMENTO_CONFIRMACION_TCN.json`.
- **T0 (solo lectura):** `models/arquitecturas_literatura_2026-10-02/T_tcn/`.
- **R0 (solo lectura):** `models/experimento_mano_ausente_2026-10-02/A_mascara/`.

Reproducción:

```
python src/experimento_confirmacion_tcn.py verificar --dest <scratch>/repro_t0
python src/experimento_confirmacion_tcn.py entrenar --brazos T1,T2 --semillas 2026   # una por proceso
python src/experimento_confirmacion_tcn.py inferencia
python src/experimento_confirmacion_tcn.py resumen --docs
```

---

## Explicación simple

Juntamos las tres mejoras que habían salido por separado: la red pequeña (TCN), rellenar los cuadros donde no
se ve una mano y usar solo 6 velocidades en vez de 126. En la validación, que es la que decide, las tres
versiones de la TCN empataron (unos 95 de cada 100) y las tres le ganan al modelo actual. Según la regla
fijada de antemano, gana la versión más simple: la TCN con relleno y 6 velocidades (T2). Pero con las personas
nuevas, la que más acertó fue la TCN con relleno y las 126 velocidades (T1): 68 % frente a 59 % del modelo
actual. T2 bajó un poco y, además, le fue peor con la persona más difícil (0000). Elegir entre la regla
(T2) y lo que muestra la prueba (T1) es una decisión para el director.
