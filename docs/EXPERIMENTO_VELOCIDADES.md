# Experimento de ablación: ¿aportan las velocidades al modelo canónico?

Fecha: 30-sep-2026. Script: `src/experimento_velocidades.py`. Salidas: `models/experimento_velocidades_2026-09-30/`.
Datos crudos: `docs/EXPERIMENTO_VELOCIDADES.json`.

Este experimento **no cambia el modelo canónico**. Es una medición para el documento de tesis.

## 1. Pregunta

La entrada canónica de la red es 30 × 252: las 126 posiciones de la representación «manos» (D-08)
remuestreadas a 30 pasos, concatenadas con sus velocidades (`add_velocity` de `src/train_final_100.py`:
`vel[:, 1:] = X[:, 1:] − X[:, :-1]`, `vel[:, 0] = 0`). Las velocidades se heredaron del código de marzo de
2026 como práctica común y su aporte nunca se midió en el modelo vigente. Los modelos de marzo sin
velocidades no sirven para comparar (otra arquitectura y una partición contaminada).

Brazos, con todo lo demás idéntico:

| brazo | entrada | qué recibe la red |
|---|---|---|
| A | 30 × 252 | posiciones + velocidades (configuración canónica) |
| B | 30 × 126 | solo posiciones |
| C | 30 × 126 | solo velocidades (informativo: cuánto lleva cada parte) |

## 2. Criterio prerregistrado

Escrito el 30-sep-2026 hacia las 20:10, antes de lanzar las corridas completas y sin haber visto ningún
resultado de B ni de C (solo la prueba de humo de 2 épocas con aumentación ×2, que no informa nada).

**Unidad.** Para cada brazo y pliegue LOSO *k* se promedia sobre las semillas la exactitud top-1 de
**validación** del checkpoint (`mejor_val_accuracy`, la misma métrica que decidió D-07 y D-08).
Δ*k* = val(A, *k*) − val(B, *k*).

**Variación entre semillas.** Dentro de cada celda (brazo, pliegue) de A y de B se calcula la varianza
muestral (ddof = 1) de la validación entre semillas; la variación entre semillas es la raíz de la media de
esas 10 varianzas (desviación estándar combinada). Es la dispersión de una sola corrida debida solo a la
semilla.

**Veredicto.**
- «**Las velocidades ayudan**» si Δ*k* > 0 en al menos 4 de los 5 pliegues **y** la media de Δ*k* supera
  la variación entre semillas.
- «**Las velocidades estorban**» si Δ*k* < 0 en al menos 4 de los 5 pliegues **y** la media de −Δ*k*
  supera la variación entre semillas.
- «**No hay evidencia**» en cualquier otro caso.
- Un Δ*k* exactamente 0 no cuenta para ningún lado.

**Lo que no decide.** La prueba LOSO (top-1, top-3, F1 macro sobre el señante no visto) se reporta para
los tres brazos como descripción, con la diferencia pareada por pliegue, una prueba de signos (binomial
exacta, bilateral) y una de rangos con signo de Wilcoxon (exacta, bilateral) sobre las 5 diferencias por
pliegue. El brazo C no entra en el criterio: solo describe cuánta información llevan las velocidades por sí
solas.

**Plan de corridas fijado antes de ver resultados.** Semillas de entrenamiento 2026, 2027 y 2028; los
5 pliegues; fase 1: A y B (30 corridas); fase 2: C (15 corridas). El tiempo medido por pliegue en este
equipo con la configuración canónica es de unos 2 a 3 minutos en solitario, así que las 45 corridas caben
holgadamente en el límite de 10 horas. Si la fase 1 no terminara, el criterio se aplicaría con las semillas
que tengan los 5 pliegues completos en A y en B (mínimo 2).

**Comprobación de reproducción.** El brazo A con la semilla 2026 debe reproducir la referencia de N-30
(LOSO top-1 0,580 ± 0,086; por pliegue 0,460 / 0,533 / 0,606 / 0,615 / 0,685). Si no la reproduce, se
investiga antes de comparar.

## 3. Configuración exacta

Todo lo que sigue es idéntico entre brazos; solo cambia la función que construye la entrada de la red.

| elemento | valor |
|---|---|
| datos | `data/processed_holistic_mascara_rostro` (HolisticLandmarker, D-05), normalización por hombros, regla de mano ausente «mascara» (D-07) |
| representación | «manos», 126 rasgos por paso (D-08) |
| partición | `data/partitions/loso_5_pliegues.json` (semilla de partición 2026; D-06: 997 muestras, validación 2 por seña = 100) |
| remuestreo | `resample_sequence` a 30 pasos |
| aumentación | ×10, las 9 transformaciones de `refined_augmentation` en su orden (vía `train_loso.aumentar` con el espejo `espejar(X, "manos")`), aplicada **sobre las 126 posiciones, antes** de calcular velocidades |
| entrada | A: `add_velocity(X)` (252); B: `X` (126); C: `add_velocity(X)[:, :, 126:]` (126) |
| red | `build_model` (CNN + BiLSTM + atención), dropout 0,3 |
| optimización | Adam, CosineDecay desde 1e-3, suavizado de etiquetas 0,1, lote 64, máx. 200 épocas |
| selección | early stopping y checkpoint sobre `val_accuracy`, paciencia 25, `restore_best_weights` |
| semillas de entrenamiento | 2026, 2027, 2028 (`tf.keras.utils.set_random_seed`) |
| código | `train_loso.entrenar_y_evaluar` sin modificar; el brazo se elige sustituyendo `add_velocity` en la tupla `canon` |
| entorno | CPU, macOS arm64, Python 3.12.13, TensorFlow 2.21.0, Keras 3.13.2, NumPy 2.4.3 (las mismas versiones que la referencia N-30) |

Comprobaciones previas (`experimento_velocidades.py verificar`, en `verificacion.json`):
- la entrada de A es exactamente la concatenación [B | C], y la velocidad del primer paso es 0;
- con la misma semilla la aumentación produce las mismas posiciones, opera sobre 126 columnas y el
  espejo es una involución; como se aplica antes de `add_velocity`, no depende del ancho de la entrada del
  brazo y no rompe B ni C;
- la aumentación consume el generador antes de construir la red, así que los tres brazos ven los mismos
  datos aumentados y el mismo orden de lotes. Lo único que difiere en la inicialización es la primera
  convolución, que tiene otra forma (3 × 252 × 128 frente a 3 × 126 × 128).

**Reproducción de la referencia.** A con la semilla 2026 reproduce N-30 **bit a bit**: top-1 por pliegue
0,460 / 0,533 / 0,606 / 0,615 / 0,685 (media 0,580), las mismas validaciones y las mismas épocas
(67 / 46 / 42 / 61 / 62). El entrenamiento es determinista en este equipo.

Ejecución: 45 corridas (3 brazos × 3 semillas × 5 pliegues), 23:37 del 30-sep a 01:35 del 1-oct de 2026,
unos 2,5 min por pliegue (A 154 s, B 163 s y C 153 s de media). Se lanzaron después de esperar 205 min a
que terminara el experimento de pasos temporales. Inferencia medida al final, sin otros entrenamientos.

## 4. Resultados

### 4.1 Resumen por brazo

Media sobre los 5 pliegues de la media de las 3 semillas por pliegue; ± es la desviación estándar entre
pliegues.

| brazo | entrada | val top-1 | LOSO top-1 | LOSO top-3 | F1 macro | épocas ejecutadas | mejor época | parámetros | ms/muestra (lote 1) | ms/muestra (lote 200) |
|---|---|---|---|---|---|---|---|---|---|---|
| A con velocidades | 252 | 0,920 ± 0,020 | 0,589 ± 0,094 | 0,764 | 0,544 | 54,5 | 29,5 | 714 291 | 12,3 | 0,211 |
| B sin velocidades | 126 | 0,923 ± 0,027 | 0,574 ± 0,129 | 0,745 | 0,529 | 59,1 | 34,1 | 665 907 | 12,4 | 0,207 |
| C solo velocidades | 126 | 0,867 ± 0,036 | 0,567 ± 0,094 | 0,737 | 0,532 | 57,1 | 32,1 | 665 907 | 12,5 | 0,206 |

Inferencia: `model.predict` de un lote de 1, como en `translator.py`; mediana de 100 llamadas tras 20 de
calentamiento, sobre los 15 modelos de cada brazo (rango de medianas entre modelos: A 12,2-12,7 ms,
B 12,3-12,8 ms, C 12,4-12,7 ms). Construir la entrada cuesta 0,002 ms con `add_velocity` y nada sin ella.
A lote 1 manda el costo fijo de la llamada a Keras: los tres brazos cuestan lo mismo. Las velocidades
añaden 48 384 parámetros (+7,3 %), todos en la primera convolución.

LOSO por semilla (media de 5 pliegues, top-1): A 0,580 / 0,592 / 0,596; B 0,577 / 0,597 / 0,549;
C 0,574 / 0,556 / 0,571 (semillas 2026 / 2027 / 2028).

### 4.2 Por pliegue (media de las 3 semillas)

| pliegue (señante) | val A | val B | val C | top-1 A | top-1 B | top-1 C | top-3 A | top-3 B | top-3 C | F1 A | F1 B | F1 C | épocas A / B / C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (0000) | 0,923 | 0,927 | 0,897 | 0,452 | 0,350 | 0,442 | 0,627 | 0,508 | 0,615 | 0,400 | 0,327 | 0,404 | 56 / 48 / 55 |
| 1 (0001) | 0,893 | 0,890 | 0,817 | 0,536 | 0,591 | 0,506 | 0,715 | 0,765 | 0,695 | 0,472 | 0,518 | 0,472 | 46 / 53 / 67 |
| 2 (0002) | 0,917 | 0,927 | 0,850 | 0,631 | 0,611 | 0,598 | 0,786 | 0,778 | 0,749 | 0,604 | 0,585 | 0,559 | 54 / 57 / 48 |
| 3 (0003) | 0,950 | 0,963 | 0,907 | 0,640 | 0,655 | 0,607 | 0,827 | 0,832 | 0,785 | 0,603 | 0,614 | 0,576 | 58 / 67 / 54 |
| 4 (0004) | 0,917 | 0,907 | 0,867 | 0,687 | 0,663 | 0,682 | 0,863 | 0,840 | 0,842 | 0,639 | 0,602 | 0,646 | 58 / 70 / 60 |

Valores por semilla (validación; top-1 de prueba):

| pliegue | A val (2026/2027/2028) | B val | C val | A top-1 | B top-1 | C top-1 |
|---|---|---|---|---|---|---|
| 0 | 0,95 / 0,92 / 0,90 | 0,93 / 0,94 / 0,91 | 0,91 / 0,90 / 0,88 | 0,460 / 0,480 / 0,415 | 0,380 / 0,385 / 0,285 | 0,490 / 0,410 / 0,425 |
| 1 | 0,90 / 0,90 / 0,88 | 0,90 / 0,89 / 0,88 | 0,81 / 0,81 / 0,83 | 0,533 / 0,492 / 0,583 | 0,608 / 0,598 / 0,568 | 0,518 / 0,538 / 0,462 |
| 2 | 0,91 / 0,91 / 0,93 | 0,91 / 0,94 / 0,93 | 0,87 / 0,86 / 0,82 | 0,606 / 0,606 / 0,682 | 0,591 / 0,641 / 0,601 | 0,601 / 0,581 / 0,611 |
| 3 | 0,96 / 0,96 / 0,93 | 0,98 / 0,96 / 0,95 | 0,89 / 0,92 / 0,91 | 0,615 / 0,650 / 0,655 | 0,665 / 0,665 / 0,635 | 0,580 / 0,595 / 0,645 |
| 4 | 0,94 / 0,90 / 0,91 | 0,89 / 0,92 / 0,91 | 0,87 / 0,86 / 0,87 | 0,685 / 0,730 / 0,645 | 0,640 / 0,695 / 0,655 | 0,680 / 0,655 / 0,710 |

### 4.3 Comparación pareada

Diferencia por pliegue de las medias sobre semillas (puntos de exactitud). Signos: binomial exacta
bilateral, sin empates. Wilcoxon: rangos con signo, exacta, bilateral.

| comparación | métrica | Δ por pliegue 0 / 1 / 2 / 3 / 4 | Δ media | a favor del primero | signos p | Wilcoxon p |
|---|---|---|---|---|---|---|
| A − B | **val top-1** | −0,003 / +0,003 / −0,010 / −0,013 / +0,010 | **−0,003** | **2 / 5** | 1,00 | 0,63 |
| A − B | LOSO top-1 | +0,102 / −0,055 / +0,020 / −0,015 / +0,023 | +0,015 | 3 / 5 | 1,00 | 0,63 |
| A − B | LOSO top-3 | +0,118 / −0,050 / +0,008 / −0,005 / +0,023 | +0,019 | 3 / 5 | 1,00 | 0,63 |
| A − B | F1 macro | +0,073 / −0,046 / +0,019 / −0,011 / +0,037 | +0,014 | 3 / 5 | 1,00 | 0,63 |
| A − C | val top-1 | +0,027 / +0,077 / +0,067 / +0,043 / +0,050 | +0,053 | 5 / 5 | 0,0625 | 0,0625 |
| A − C | LOSO top-1 | +0,010 / +0,030 / +0,034 / +0,033 / +0,005 | +0,022 | 5 / 5 | 0,0625 | 0,0625 |
| A − C | F1 macro | −0,005 / +0,001 / +0,045 / +0,027 / −0,007 | +0,012 | 3 / 5 | 1,00 | 0,63 |
| B − C | val top-1 | +0,030 / +0,073 / +0,077 / +0,057 / +0,040 | +0,055 | 5 / 5 | 0,0625 | 0,0625 |
| B − C | LOSO top-1 | −0,092 / +0,085 / +0,014 / +0,048 / −0,018 | +0,007 | 3 / 5 | 1,00 | 1,00 |

En los 15 pares (pliegue, semilla) de validación A − B: 3 a favor de A, 6 a favor de B y 6 empates. Solo
es descriptivo, porque las semillas de un mismo pliegue no son independientes.

**Variación entre semillas** (validación, desviación estándar combinada de A y B): **0,016**.

## 5. Conclusión según el criterio prerregistrado

- A > B en validación en **2 de 5** pliegues (B > A en 3). No se llega a 4 de 5 en ninguna dirección.
- Diferencia media A − B en validación: **−0,003**, una sexta parte de la variación entre semillas (0,016).

**Veredicto: «No hay evidencia».** Con este modelo, este corpus y esta partición, las velocidades no
mejoran ni empeoran la exactitud de validación. El modelo canónico no cambia: sigue con velocidades
(30 × 252), como estaba.

Lo que dice la descripción, sin valor de decisión:
- **Prueba LOSO.** A queda 1,5 puntos por encima de B en top-1 (0,589 frente a 0,574), 1,9 en top-3 y 1,4
  en F1, pero gana solo en 3 de 5 pliegues y casi toda la ventaja sale del pliegue 0. Ese es el señante
  0000, el difícil: ahí B cae a 0,350 y A se queda en 0,452. En el pliegue 1, B supera a A por 5,5 puntos.
  A tiene menos dispersión entre señantes (0,094 frente a 0,129). Es compatible con que las velocidades
  den algo de robustez ante un señante atípico, pero con 5 señantes no se puede separar del azar.
- **Brazo C.** Las velocidades solas pierden 5,3 puntos de validación frente a A, en los 5 pliegues y en los
  15 pares, muy por encima de la variación entre semillas. Las posiciones llevan información que las
  velocidades no tienen, y el modelo no la recupera integrando las diferencias. En la prueba LOSO, en cambio,
  C pierde solo 2,2 puntos de top-1 frente a A y empata con B (0,567 frente a 0,574). La caída en
  validación es mayor porque la validación sale de los señantes ya vistos, y ahí las posiciones absolutas
  ayudan a reconocer a la persona. Con un señante nuevo, el movimiento solo generaliza casi igual que la
  posición sola. Es coherente con N-30, donde la postura y el rostro ayudaban con señantes vistos y
  perjudicaban con señantes nuevos.
- Posiciones y velocidades son, para esta red, casi redundantes: cualquiera de las dos llega a unos
  0,57-0,59 de LOSO, y juntas no suman más de lo que da la variación entre pliegues.
- **Costo.** Las velocidades añaden un 7 % de parámetros y ninguna diferencia medible en la latencia
  (unos 12 ms por muestra a lote 1, dominados por el costo fijo de `predict`).

## 6. Pruebas estadísticas y sus límites con 5 pliegues

- Con n = 5 diferencias, el p bilateral más pequeño posible es 2 / 2⁵ = **0,0625**, tanto en la prueba de
  signos como en la de Wilcoxon exacta, y se alcanza solo si los 5 pliegues van en la misma dirección.
  **Ninguna comparación puede llegar a p < 0,05**: el 0,0625 de A − C y B − C en validación es el máximo
  de evidencia que este diseño puede dar. Por eso el criterio no usa un p-valor, sino la regla de 4 de 5
  más el tamaño frente a la variación entre semillas.
- La prueba de signos ignora el tamaño de las diferencias. Wilcoxon lo usa en forma de rangos, pero con 5
  valores apenas tiene potencia: una diferencia real de 1 a 2 puntos sería indetectable.
- Los pliegues no son réplicas independientes de un mismo experimento: comparten 3 de los 4 señantes de
  entrenamiento. La desviación entre pliegues mezcla la dificultad de cada señante con el azar.
- Promediar 3 semillas reduce el ruido de inicialización, pero no el de los señantes. La variación entre
  semillas (0,016 en validación; en la prueba, hasta 10 puntos entre semillas de un mismo pliegue, por
  ejemplo B en el pliegue 0: 0,385 frente a 0,285) es del mismo orden que las diferencias A − B.

## 7. Limitaciones

1. **Cinco pliegues, cinco señantes.** El criterio solo puede detectar efectos grandes y consistentes. «No
   hay evidencia» no demuestra que las velocidades no sirvan: dice que su efecto, si existe, es menor que
   el ruido que este diseño puede resolver (del orden de 1 a 2 puntos).
2. **Tres semillas.** La variación entre semillas se estima con 3 valores por celda (10 celdas), así que el
   umbral de 0,016 es en sí mismo impreciso. La resolución de la validación es de 0,01 (100 muestras).
3. **La validación sale de los señantes vistos** (D-06). El brazo C muestra que el orden en validación y en
   prueba puede diferir: es un indicador sesgado de la generalización a señantes nuevos (ya señalado en N-30).
4. **Saltos falsos de la regla «mascara».** Cuando el extractor no detecta una mano, su bloque se pone en
   ceros tras normalizar (D-07). La velocidad del paso en que la mano desaparece o reaparece es un salto
   artificial hasta el origen y de vuelta, no un movimiento real. En las secuencias remuestreadas a 30
   pasos: la mano izquierda tiene transiciones presente↔ausente en 353 de 1000 secuencias (1095 pasos) y la
   derecha en 164 (430 pasos). La norma mediana de la velocidad de la mano en esos pasos es ≈ 4,9, frente a
   0,37-0,46 cuando la mano está presente en los dos pasos: unas 10 a 13 veces mayor. Además, el remuestreo
   lineal interpola entre un fotograma real y uno enmascarado, y crea bloques «encogidos» hacia el origen en
   los pasos vecinos. Las velocidades del brazo A y del C llevan, entonces, una señal mezclada: movimiento
   real más un indicador ruidoso de «se perdió la mano». Este experimento mide las velocidades tal como las
   produce el camino canónico; no mide velocidades limpias (por ejemplo, con la velocidad puesta en 0 cuando
   alguno de los dos pasos está enmascarado). Esa variante queda como trabajo futuro y exigiría su propio
   criterio previo.
5. **Velocidad de primer orden sin escalar.** Las velocidades no se normalizan aparte. Su escala es la de las
   posiciones, porque son diferencias en el mismo marco de hombros. No se probaron aceleraciones ni otras
   escalas.
6. **Una sola arquitectura.** La primera capa convolucional, con núcleo de 3 pasos, ya puede calcular
   diferencias temporales de las posiciones. Esto explica que B iguale a A. En una red sin convolución
   temporal, el resultado podría ser otro.
7. Los conteos de saltos falsos usan las 1000 muestras del processed, incluidos los 3 duplicados que D-06
   excluye de las particiones. La diferencia es despreciable.

## 8. Explicación simple

La red que reconoce las señas recibe, en cada instante, dónde están los puntos de las dos manos. Desde marzo
también recibe cuánto se movió cada punto respecto al instante anterior: la «velocidad». Nadie había
comprobado si esa segunda información sirve. Entrenamos la red tres veces de cada forma (con velocidades,
sin ellas y solo con ellas), probándola siempre con una persona que no vio al aprender. Con o sin
velocidades, acierta casi lo mismo (59 % frente a 57 %), y la diferencia es más pequeña que la que aparece
al repetir el mismo entrenamiento con otro punto de partida al azar. Conclusión: con estos datos no hay
pruebas de que las velocidades ayuden ni de que estorben. El modelo se deja como está.

## 9. Archivos

- `src/experimento_velocidades.py`: script (subcomandos `verificar`, `entrenar`, `inferencia`, `resumen`).
- `models/experimento_velocidades_2026-09-30/`: modelos y métricas por brazo, semilla y pliegue
  (`A_con_velocidades/`, `B_sin_velocidades/`, `C_solo_velocidades/`), con `verificacion.json`,
  `inferencia.json`, `resultados_velocidades.json`, los registros `corrida_*.log` y `espera.log` (espera al
  otro experimento). Cada `modelo.procedencia.json` declara su brazo y advierte que no es el modelo canónico.
- `docs/EXPERIMENTO_VELOCIDADES.json`: datos crudos (las 45 corridas, agregados, comparaciones, criterio,
  reproducción de N-30, inferencia y verificación).

Reproducir: `python src/experimento_velocidades.py entrenar --brazos A,B,C --semillas 2026,2027,2028`,
luego `inferencia` y `resumen --docs`. Se usó el intérprete `python`:
el `venv/` del repositorio es un entorno de Windows (`Scripts/`, sin `bin/python`) y no sirve en este equipo.
