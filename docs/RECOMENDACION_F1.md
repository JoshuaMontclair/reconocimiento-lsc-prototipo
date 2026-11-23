# Qué F1 se reporta en la tesis: análisis y recomendación (pieza 3, 2026-09-22)

> **Estado: propuesta para decidir con el director.** Aquí solo se analiza y se recomienda. No se
> cambió ningún modelo, umbral, partición, README ni registro. Todas las cifras salen de
> `models/f1_variantes_2026-09-22/f1_variantes.json` (script `src/f1_variantes.py`, corrida del
> 2026-09-22 16:33:09, 2.000 remuestreos, semilla 2026), que las recalcula desde los
> `predicciones.json` guardados. Cada tabla dice de qué clave sale.

## 0. El problema en una línea

El mismo LOSO de la red «manos» (N-30) da **F1 macro 0,537** si se calcula en cada pliegue y se
promedian los cinco, y **0,574** si se calcula sobre las 997 predicciones juntas (N-35). Hoy el
README usa las dos para el mismo modelo sin decir cuál es cuál: `README.md:436` pone «F1 0,537» en
la tabla de D-08 y `README.md:454` pone «F1 macro 0,574» en la tabla de la línea base. El borrador
`docs/REPORTE_COMPARATIVO_4e.md` (§9, líneas 880-900) ya lo señala y propone lo mismo que aquí se
recomienda. Esta pieza lo mide y lo fija por escrito.

## 1. Criterio, fijado antes de calcular

El criterio se escribió antes de correr nada, en
`…/carpeta_temporal/wf_2026-09-22/pieza3/00_CRITERIO_ANTES_DE_CALCULAR.txt` (2026-09-22 16:14:13; primera
corrida a las 16:20). En el relanzamiento se ratificó sin cambios en `01_CRITERIO_RELANZAMIENTO.txt`
(16:30:18), antes de la corrida de las 16:31. Dice, en resumen:

1. La cifra principal del LOSO se elige por la **unidad de réplica** del diseño (el señante) y por lo
   que pide el anteproyecto («media y dispersión entre pliegues»), **no por cuál lectura da el número
   más alto**.
2. El agregado sobre las 997 se reporta como complemento, siempre con apellido.
3. Esquema aleatorio (un solo conjunto de prueba): solo existe el agregado, sin «±».
4. Si el orden entre sistemas cambia según la lectura, se declara. McNemar se verifica sobre aciertos
   top-1.
5. No se afirma ninguna causa de la diferencia que no se haya medido.

## 2. Qué dice el anteproyecto (citas comprobadas con grep, 2026-09-22)

Carpeta `Anteproyecto/Anteproyecto_FINAL_LaTeX/`:

- `09_metodologia.tex:143-145`, en «Modelo de Reconocimiento y Línea Base» (la sección que
  `09_metodologia.tex:56` llama §2.2.4): «Con cinco señantes, la partición independiente se
  implementa como validación cruzada dejando un señante fuera en cada pliegue ---cinco pliegues---, y
  **se reportan la media y la dispersión entre pliegues**».
- `09_metodologia.tex:145-146`: los resultados de cada esquema «se reportan por separado y nunca se
  presentan como equivalentes».
- `09_metodologia.tex:147-149`: la selección del modelo se hace «con base en exactitud top-1 y top-3,
  precision, recall y F1-score sobre el conjunto de validación» (la validación no es la cifra que
  aquí se discute; D-06 ya lo dice).
- `09_metodologia.tex:276-281`: «la top-1 y la top-3 para el desempeño global del clasificador, y
  precision, recall y F1 por clase para detectar señas problemáticas y desbalances».
- `09_metodologia.tex:503-505` (actividad 4.c) y `10_resultados.tex:40-43`: exactitud «desagregada en
  top-1, top-3, precision, recall y F1-score, con su matriz de confusión», bajo los dos esquemas.
- `09_metodologia.tex:522-524` (actividad 4.e): «el reporte de cada pliegue de la validación cruzada
  con la condición del señante excluido».
- `08_marco.tex:203-204`: la tabla de variables pide «F1-score (macro y por clase)»; `08_marco.tex:208-211`:
  «Se reportan las cinco métricas de forma desagregada, junto con la partición empleada […] **No se
  fija umbral de aprobación.**»
- `06_alcances.tex:86-88` y `06_alcances.tex:95-97`: las mismas cinco métricas; señantes vistos y no
  vistos «se reportarán por separado».

Lo que el anteproyecto **no** dice: en ningún archivo se define si el F1 macro del LOSO se promedia
por pliegue o se calcula sobre las predicciones agregadas. La frase más cercana es
`09_metodologia.tex:143-145`, que habla de «media y dispersión entre pliegues» para el esquema LOSO
en general. (Nota de numeración: la pieza pedía mirar «§2.5»; según `04_objetivos.tex:6`, los
resultados esperados son §2.3. Aquí se cita siempre por archivo y línea.)

## 3. Las lecturas, sistema por sistema (LOSO, 997 predicciones, 50 señas)

Claves: `sistemas.<clave>.loso.por_pliegue.resumen.<métrica>.{media,de,por_pliegue}` y
`sistemas.<clave>.loso.agregado.<métrica>`. DE con ddof = 1. Macro sobre las 50 señas con 0/0 = 0
(la convención de `train_loso.py` y de sklearn).

| Sistema | (a) F1 macro, media ± DE entre pliegues | por pliegue (señante 0000…0004) | (b) F1 macro agregado (997) | (c) F1 ponderado agregado | (d) F1 micro agregado = top-1 agregada | top-1, media ± DE |
|---|---|---|---|---|---|---|
| Red, manos (C1, vigente) | 0,537 ± 0,093 | 0,421 / 0,463 / 0,573 / 0,585 / 0,645 | 0,574 | 0,574 | 0,580 | 0,580 ± 0,086 |
| Red, manos + postura (C2) | 0,498 ± 0,171 | 0,285 / 0,434 / 0,442 / 0,596 / 0,731 | 0,541 | 0,541 | 0,534 | 0,533 ± 0,176 |
| Red, manos + postura + rostro (C3) | 0,464 ± 0,111 | 0,296 / 0,445 / 0,453 / 0,551 / 0,576 | 0,508 | 0,508 | 0,509 | 0,509 ± 0,109 |
| Línea base RF | 0,373 ± 0,135 | 0,164 / 0,321 / 0,502 / 0,464 / 0,415 | 0,393 | 0,393 | 0,418 | 0,418 ± 0,139 |
| Línea base SVM | 0,229 ± 0,143 | 0,009 / 0,176 / 0,309 / 0,277 / 0,375 | 0,282 | 0,282 | 0,264 | 0,264 ± 0,152 |
| Línea base MLP | 0,222 ± 0,122 | 0,021 / 0,193 / 0,280 / 0,304 / 0,312 | 0,257 | 0,257 | 0,248 | 0,248 ± 0,133 |

(e) Precisión y recall macro en las dos lecturas:

| Sistema | P macro: media ± DE | P macro agregada | R macro: media ± DE | R macro agregado |
|---|---|---|---|---|
| Red, manos (C1, vigente) | 0,557 ± 0,088 | 0,594 | 0,581 ± 0,086 | 0,580 |
| Red, manos + postura (C2) | 0,526 ± 0,160 | 0,588 | 0,534 ± 0,176 | 0,534 |
| Red, manos + postura + rostro (C3) | 0,498 ± 0,111 | 0,550 | 0,508 ± 0,109 | 0,508 |
| Línea base RF | 0,404 ± 0,135 | 0,419 | 0,418 ± 0,139 | 0,418 |
| Línea base SVM | 0,265 ± 0,157 | 0,340 | 0,264 ± 0,152 | 0,264 |
| Línea base MLP | 0,262 ± 0,137 | 0,302 | 0,248 ± 0,133 | 0,248 |

(f) Intervalos al 95 % para el F1 macro (claves `sistemas.<clave>.loso.bootstrap.*`). Bootstrap
percentil, 2.000 remuestreos, semilla 2026. El del agregado remuestrea predicciones con reposición
dentro de cada seña (conserva 20 o 19 por seña). El de la media remuestrea los 5 pliegues con
reposición y, dentro de cada pliegue elegido, sus predicciones (también por seña). Se añaden el
remuestreo de solo pliegues y el intervalo t con 4 grados de libertad como contraste:

| Sistema | media ± DE | IC boot, pliegues + dentro | IC boot, solo pliegues | IC t (4 g. l.) | agregado | IC boot del agregado |
|---|---|---|---|---|---|---|
| Red, manos (C1, vigente) | 0,537 ± 0,093 | [0,455; 0,604] | [0,462; 0,607] | [0,422; 0,653] | 0,574 | [0,546; 0,598] |
| Red, manos + postura (C2) | 0,498 ± 0,171 | [0,360; 0,629] | [0,376; 0,624] | [0,286; 0,710] | 0,541 | [0,511; 0,566] |
| Red, manos + postura + rostro (C3) | 0,464 ± 0,111 | [0,363; 0,539] | [0,377; 0,542] | [0,327; 0,602] | 0,508 | [0,481; 0,533] |
| Línea base RF | 0,373 ± 0,135 | [0,253; 0,463] | [0,263; 0,469] | [0,205; 0,541] | 0,393 | [0,365; 0,419] |
| Línea base SVM | 0,229 ± 0,143 | [0,103; 0,327] | [0,102; 0,329] | [0,052; 0,406] | 0,282 | [0,256; 0,305] |
| Línea base MLP | 0,222 ± 0,122 | [0,109; 0,297] | [0,112; 0,302] | [0,071; 0,373] | 0,257 | [0,232; 0,277] |

En los 6 sistemas, el intervalo de la media (pliegues + dentro) es entre 2,9 y 4,9 veces más ancho
que el del agregado. La razón es de diseño: el remuestreo del agregado trata las 997 predicciones como
independientes, pero vienen de 5 personas. Con 5 pliegues solo existen 126 multiconjuntos distintos
de pliegues, así que el bootstrap de la media es tosco. El intervalo t es más ancho y supone que la
media entre pliegues sigue una distribución normal. Ninguno de los dos es exacto con n = 5, y así hay
que decirlo en la tesis.

Controles: los F1 por pliegue y su media reproducen `resultados_loso.json` de las tres redes
(tolerancia 1e-9). El agregado y la media de las tres líneas base reproducen
`models/linea_base_4d_2026-09-15/resultados.json` (`loso.con_espejo`). Clave:
`sistemas.<clave>.loso.controles.todo_coincide` = true en los 6. El «candidato con rostro» de N-30
es `models/loso_2026-09-15_holistic_mascara_rostro/`: su `resultados_loso.json` declara
`representacion` = `manos_postura_rostro`, y su F1 medio (0,4644) coincide con la clave `f1` de C3 en
`models/comparacion_D08_2026-09-15.json` (0,46437).

## 4. Por qué (a) y (b) difieren, y cuánto pesa cada causa

**Identidad exacta** (comprobada en cada sistema, clave `descomposicion.identidad_cierra`). Para la
seña c en el pliegue k, sea D_kc = 2·TP + FP + FN, que es igual a las muestras reales de la seña más
las veces que se predijo. Entonces:

- F1 agregado de la seña = Σ_k 2·TP_kc / Σ_k D_kc = Σ_k w_kc · F1_kc, con w_kc = D_kc / Σ_k D_kc;
- media por pliegue de la seña = Σ_k (1/5) · F1_kc.

Las dos lecturas promedian los mismos F1_kc, pero con pesos distintos. La media da el mismo peso a
cada señante. El agregado pesa cada pliegue según cuántas veces aparece y se predice allí la seña.
Un pliegue en el que la seña no se acierta nunca tiene F1_kc = 0 y pierde el término 2·TP de su D,
así que suele pesar menos de 1/5 en el agregado. No pasa siempre: depende de sus falsos positivos.
Es el mecanismo que Forman y Scholz (2010) describen para el F1 promediado por pliegues (§6).

**Cuánto aporta cada causa** (clave `sistemas.<clave>.loso.descomposicion`):

| Sistema | agregado − media | pares (pliegue, seña) con 0 aciertos (de 250) | señas con ≥ 1 pliegue a 0 | aporte de esas señas | aporte del resto | efecto del tamaño 198-200 | pares nunca predichos (de 250) |
|---|---|---|---|---|---|---|---|
| Red, manos (C1, vigente) | 0,0363 | 65 | 31 | 0,0412 | -0,0049 | 0,00000 | 44 |
| Red, manos + postura (C2) | 0,0437 | 77 | 36 | 0,0439 | -0,0002 | 0,00018 | 45 |
| Red, manos + postura + rostro (C3) | 0,0440 | 78 | 35 | 0,0428 | 0,0012 | 0,00004 | 45 |
| Línea base RF | 0,0205 | 94 | 40 | 0,0298 | -0,0094 | -0,00021 | 48 |
| Línea base SVM | 0,0532 | 141 | 49 | 0,0520 | 0,0012 | -0,00011 | 87 |
| Línea base MLP | 0,0347 | 142 | 49 | 0,0346 | 0,0001 | -0,00009 | 75 |

Lectura de lo medido:

1. **Las señas con algún pliegue sin aciertos explican la diferencia.** En la red «manos», esas 31
   señas aportan +0,0412 y las 19 restantes −0,0049; la suma, 0,0363, es la diferencia completa. En
   los 6 sistemas el aporte de esas señas es mayor, en valor absoluto, que el del resto.
2. **El tamaño desigual de los pliegues (198-200) no explica nada apreciable.** La media ponderada
   por tamaño se aparta de la simple como mucho 0,00021 (RF); en «manos», 0,00000 a cinco decimales.
3. **La convención 0/0 = 0 no mueve el F1, pero sí la precisión.** Cada (pliegue, seña) tiene al
   menos 3 muestras reales (clave `soporte_minimo_pliegue_sena` = 3; 0 pares sin muestras). Por eso,
   cuando una seña no tiene aciertos, FN > 0 y su F1 es 0 con cualquier convención. En cambio, 44 de
   los 250 pares (pliegue, seña) de «manos» nunca se predicen. Su precisión 0/0 se cuenta como 0, y
   eso baja la precisión macro media a 0,557, frente a 0,674 si se excluyen esos pares (clave
   `precision_macro_media_pliegues`). En el agregado, la precisión de una seña pondera cada pliegue
   por las veces que se predijo allí, así que esos pares pesan 0; en la media pesan 1/5 con valor 0.
   Ese es el mecanismo por el que la precisión difiere entre lecturas (0,557 frente a 0,594) y el
   recall casi no (0,581 frente a 0,580), pero no es la única diferencia de pesos: excluir sin más
   esos pares sobrecorrige (0,674, frente a 0,594 agregada). Con 3 o 4 muestras por seña y señante,
   el recall agregado de una seña es casi la media de sus recall por pliegue. La diferencia está del
   lado de las predicciones. [Redacción ajustada por el verificador, 2026-09-22.]
4. **Lo que no se afirma:** que el agregado sea «el verdadero» y la media «esté sesgada», o al revés.
   Estiman cosas distintas (§5).

## 5. Lo que responde cada lectura

- **Media ± DE entre pliegues:** «si el sistema se enfrenta a un señante que no vio, ¿qué F1 macro
  obtiene sobre sus ~200 señas, y cuánto varía de una persona a otra?». Cada pliegue es una persona, y
  la persona es la unidad de réplica del LOSO. Es lo que pide `09_metodologia.tex:143-145`, y la DE es
  la única de las dos lecturas que muestra la variabilidad entre personas: en «manos», el F1 va de
  0,421 (señante 0000) a 0,645 (señante 0004).
- **Agregado sobre las 997:** «juntando todas las presentaciones de prueba de los cinco señantes,
  ¿qué F1 macro tiene la tabla de confusión global?». Es la lectura natural para comparar predicción a
  predicción (matriz de confusión global, análisis de errores de N-35, McNemar) y la que da el
  software que concatena predicciones. Pero no tiene «±» entre personas, y su intervalo de bootstrap
  por predicción subestima la incertidumbre, porque ignora que las predicciones se agrupan en 5
  personas (§3, f).

## 6. Qué hacen otros: literatura verificada y práctica habitual

**Publicaciones sobre la LSC que cita el anteproyecto.** En la tabla de trabajos relacionados
(`08_marco.tex:387-419`), las cifras para la LSC son porcentajes sueltos: ≈68 % (`ref15`), 97,6 %
(`ref16`), 76 % (`ref17`), 90 % y 82 % (`ref41`), 75,9 %, 75,5 % y 87,6 % (`ref42`). Ninguna fila de
esa tabla menciona F1 ni una media ± DE entre señantes. En `ref41`, según `08_marco.tex:320-322`, «los
porcentajes son exactitudes de validación sobre una partición aleatoria del 20 %». Y `08_marco.tex:354-357`
advierte, sobre esas cifras de la LSC, que «no son directamente comparables entre sí» porque usan
«métricas que no siempre coinciden» [cita corregida por el verificador: la de `08_marco.tex:206`
se refiere a las exactitudes de MediaPipe con redes profundas y a una revisión general, no a la LSC]. Conclusión, limitada a lo que dice el
anteproyecto: no hay en esas referencias una convención de F1 para la LSC que la tesis deba seguir, y
la comparación con ellas solo puede hacerse en top-1 (exactitud) y en el esquema correspondiente.
Esta pieza no leyó los artículos originales, así que no afirma qué reportan fuera de lo que el
anteproyecto transcribe.

**Fuente externa verificada.** G. Forman y M. Scholz, «Apples-to-apples in cross-validation studies:
pitfalls in classifier performance measurement», *ACM SIGKDD Explorations* 12(1):49-57, 2010,
doi:10.1145/1882471.1882479 (PDF leído en https://www.kdd.org/exploration_files/v12-1-p49-forman-sigkdd.pdf).
Comparan F_avg (media de los F por pliegue) con F_tp,fp (F calculado sobre los conteos sumados) en
simulaciones binarias con validación cruzada de 10 pliegues, suponiendo predicciones independientes e
idénticamente distribuidas y el mismo clasificador en todos los pliegues. Dicen que F_tp,fp «is almost
perfectly unbiased […] and therefore it is the recommended way to compute F-measure» (p. 55), y que
F_avg, «which is in common use, penalizes methods that may occasionally predict zero positives for
some test folds» (p. 55). **Esto apunta al agregado**, y su mecanismo es justo el que se mide en §4.
Tres diferencias con este caso limitan cuánto se puede trasladar:

1. Allí los pliegues son particiones aleatorias de una misma población. Aquí cada pliegue es una
   persona distinta, y la pregunta del LOSO es cómo rinde el sistema con una persona nueva, no cuál es
   el F de la población juntada.
2. Allí el sesgo se mide frente a un F «verdadero» de la población. Aquí el estimando de la media por
   pliegue es el F1 de la sesión de un señante nuevo, que es justo lo que promedia la media.
3. F_tp,fp no da ninguna medida de la variabilidad entre personas, que el anteproyecto sí pide.

Hay una consecuencia honesta: el F1 de un pliegue se calcula con solo 3-4 muestras por seña, y por la
curvatura del F1 cerca de 0 es un estimador ruidoso y probablemente algo bajo del F1 que ese señante
tendría con muchas más muestras. Por eso se recomienda reportar también el agregado (§7).

**Práctica habitual, sin cita verificada en esta pieza:** en la evaluación dejando un sujeto fuera es
habitual reportar la métrica por sujeto y su media ± DE, porque el sujeto es la unidad de réplica. No
se buscó ni se verificó una referencia concreta que lo prescriba, así que en la tesis no debe
presentarse con cita salvo que se encuentre y se lea una.

## 7. El orden entre sistemas y las pruebas de McNemar no dependen de la lectura

**Orden** (clave `orden`). El ranking de los 6 sistemas es idéntico bajo las cuatro lecturas del F1
(media por pliegue, agregado, ponderado, micro) y también bajo la precisión y el recall macro en las
dos lecturas: manos > manos + postura > manos + postura + rostro > RF > SVM > MLP. Los 5 pares
esperados (manos > manos + postura; manos + postura > rostro; red > cada línea base) se cumplen en las
9 lecturas (claves `orden.<lectura>.pares`; la clave `orden._mismo_ranking_en_todas_las_lecturas_de_f1`
= true cubre solo las cuatro lecturas del F1, y el verificador comprobó aparte que el ranking de las
otras cinco es el mismo).

Una cautela que la lectura principal hace visible: el orden de las **medias** no es el orden de cada
**pliegue**. «Manos» supera a «manos + postura» en 3 de 5 pliegues (diferencias por pliegue +0,136,
+0,029, +0,130, −0,011, −0,086), y «manos + postura» supera a «rostro» en 2 de 5. «Manos» supera a
«rostro» y a cada línea base en los 5 pliegues. (Cálculo auxiliar sobre el mismo JSON:
`…/carpeta_temporal/wf_2026-09-22/pieza3/pliegues_por_par.py`, salida en `pliegues_por_par.log`.) Los
intervalos de la media de C1, C2 y C3 se solapan (§3, f).

**McNemar** (clave `mcnemar`). Compara, muestra a muestra, si cada sistema acertó la top-1. No lee
ningún F1, así que cambiar la lectura del F1 no cambia ninguna tabla ni ningún p. Se reproducen
N-35 y N-39:

| Comparación (997 pares) | solo acierta «manos» | solo acierta el otro | prueba | p |
|---|---|---|---|---|
| manos frente a manos + postura | 134 | 88 | χ² con corrección de continuidad | 2,53e-03 |
| manos frente a manos + postura + rostro | 150 | 79 | χ² con corrección | 3,73e-06 |
| manos frente a RF | 249 | 88 | χ² con corrección | 2,89e-18 |
| manos frente a SVM | 351 | 36 | χ² con corrección | 2,37e-57 |
| manos frente a MLP | 367 | 36 | χ² con corrección | 1,01e-60 |

N-35 dice 134/88 con p 0,0024 y 150/79 con p < 0,0001; N-39 dice 2,9e-18, 2,4e-57 y 1,0e-60.
Las tablas coinciden. Los p de N-39 coinciden; los de N-35 difieren en el tercer decimal porque
`src/analisis_errores.py` usa la prueba exacta binomial (134/88: p = 0,00245; 150/79: p = 3,1e-06) y
aquí se usa χ² con corrección (0,00253 y 3,7e-06). La conclusión es la misma. [Precisión añadida por
el verificador.]

## 8. Recomendación

1. **Cifra PRINCIPAL del LOSO:** el F1 macro calculado en cada pliegue y resumido como media ± DE
   (ddof = 1) sobre los 5 pliegues. Se nombra siempre **«F1 macro, media ± DE entre pliegues (LOSO,
   5 señantes)»**. Para «manos»: **0,537 ± 0,093**. Se acompaña de los cinco valores por pliegue con
   el señante excluido (lo pide `09_metodologia.tex:522-524`): 0,421 / 0,463 / 0,573 / 0,585 / 0,645.
   La misma regla vale para top-1, top-3, precisión y recall macro del LOSO: top-1 0,580 ± 0,086.
2. **Complemento, siempre con etiqueta:** **«F1 macro agregado sobre las 997 predicciones del LOSO»**
   (0,574 para «manos»). Se usa donde el análisis es predicción a predicción: la matriz de confusión
   global, el análisis de errores por seña y la comparación con la línea base por McNemar. Nunca va
   en la misma columna que la media, nunca lleva «±» y nunca aparece sin apellido. Si se quiere un
   intervalo para él, el de bootstrap por predicción debe declararse como condicional a esos 5
   señantes.
3. **Esquema aleatorio (un solo conjunto de prueba de 100):** solo existe el agregado. Se nombra
   **«F1 macro sobre el conjunto de prueba aleatorio (n = 100)»**, sin «±». Para «manos»: 0,871
   (IC bootstrap 95 % [0,805; 0,913]). La misma regla vale para cualquier evaluación con un único
   conjunto de prueba (por ejemplo, el Bloque 1 en línea), pero eso es una propuesta de esta pieza, no
   algo que diga el anteproyecto.
4. **Intervalos:** con 5 pliegues, se recomienda la DE como medida de dispersión principal, porque es
   lo que pide el anteproyecto. Si el director quiere un intervalo para la media, se da el de bootstrap
   de pliegues + dentro (para «manos», [0,455; 0,604]), declarando que con 5 réplicas es aproximado.
5. **Tablas existentes:** la tabla de la línea base (`README.md:452-457`, cabecera en la línea 452) está hecha sobre el agregado.
   Hay dos opciones: rotular su columna como «F1 macro agregado (997)», o reescribirla en media ± DE
   entre pliegues (red 0,537 ± 0,093; RF 0,373 ± 0,135; SVM 0,229 ± 0,143; MLP 0,222 ± 0,122). Lo que
   no puede quedar es una columna «F1» a secas que en una tabla vale 0,537 y en otra 0,574 para el
   mismo modelo. La decisión y la edición del README corresponden al estudiante.

¿Por qué no el agregado como principal, si la referencia verificada lo prefiere? Porque el criterio
se fijó antes de calcular y porque las tres diferencias de §6 lo mantienen: la pregunta del LOSO es
por persona, el anteproyecto pide dispersión entre pliegues y el agregado no la tiene. El orden entre
sistemas y las pruebas de significación no cambian con la lectura (§7), así que ninguna conclusión de
la tesis depende de esta elección. Solo depende de ella el número que se escribe.

## 9. Texto propuesto para la primera mención

**Tesis (capítulo de resultados, primera vez que aparece un F1 del LOSO):**

> En el esquema independiente del señante (validación cruzada dejando un señante fuera, cinco
> pliegues), cada métrica se calcula por separado en cada pliegue —es decir, sobre las
> aproximadamente 200 presentaciones del señante excluido— y se resume como media ± desviación
> estándar entre los cinco pliegues, porque la unidad de réplica de este esquema es el señante. El
> F1 macro es la media no ponderada del F1 de las 50 señas; cuando una seña no se predice nunca en un
> pliegue, su precisión se cuenta como 0. Con esta lectura, el modelo «manos» obtiene un F1 macro de
> 0,537 ± 0,093 (por señante excluido: 0,421, 0,463, 0,573, 0,585 y 0,645) y una exactitud top-1 de
> 0,580 ± 0,086. Donde el análisis es predicción a predicción (matriz de confusión global, análisis
> de errores y pruebas de McNemar) se usa además el F1 macro agregado sobre las 997 predicciones del
> LOSO, que vale 0,574. Las dos cifras no son intercambiables: el agregado pondera cada pliegue según
> cuántas veces aparece y se predice cada seña en él, de modo que los pliegues en los que una seña no
> se acierta nunca suelen pesar menos (en «manos», 60 de los 65 casos). En este corpus, el agregado supera a la media en los seis sistemas
> evaluados (entre 0,0205 y 0,0532), pero el orden entre sistemas es el mismo con las dos lecturas. En
> el esquema aleatorio hay un único conjunto de prueba de 100 muestras, así que solo existe el F1
> macro sobre ese conjunto (0,871) y no se le asocia dispersión entre pliegues.

**README (primera mención, antes de la primera tabla con F1 del LOSO):**

> **Cómo se lee el F1 en este README.** En el LOSO, «F1 macro» significa **media ± DE entre los 5
> pliegues** (una persona por pliegue), calculado en cada pliegue sobre sus ~200 predicciones: para
> el modelo vigente «manos», 0,537 ± 0,093. Cuando una tabla usa el **F1 macro agregado sobre las
> 997 predicciones** (0,574 para el mismo modelo), la columna lo dice así. En el esquema aleatorio
> hay un solo conjunto de prueba (n = 100) y el F1 es el de ese conjunto, sin «±». Detalle y
> justificación: `docs/RECOMENDACION_F1.md`; cifras: `models/f1_variantes_2026-09-22/`.

## 10. Límites

- Cinco señantes: cualquier media ± DE entre pliegues descansa en 5 números. Los intervalos son
  aproximados y se solapan entre C1, C2 y C3.
- El F1 de un pliegue se calcula con 3-4 muestras por seña. Es ruidoso y, por la curvatura del F1
  cerca de 0, probablemente algo bajo respecto de lo que ese señante daría con más muestras. No se
  midió cuánto.
- La descomposición de §4 es exacta, pero reparte la diferencia por grupos de señas. No es un
  experimento causal.
- Las predicciones de la línea base no guardan el pliegue: se dedujo del señante del nombre de cada
  muestra, que es la regla de D-06. Los controles contra `resultados.json` confirman que la
  asignación reproduce lo guardado.
- La revisión de la literatura sobre la LSC se limita a lo que transcribe el anteproyecto. Solo se
  verificó una fuente externa (Forman y Scholz 2010), que además apunta en la dirección contraria a
  la principal recomendada y por eso se discute en §6.
- `models/canonico_holistic_mascara_manos/evaluacion_modelo_seleccionado/predicciones.json` es
  idéntico byte a byte (mismo SHA-256) al `aleatoria_80_10_10/predicciones.json` del LOSO «manos», así
  que la fila «Canónico «manos»» del esquema aleatorio repite la de «Red, manos».

## 11. Reproducir

```bash
python src/f1_variantes.py --salida models/f1_variantes_2026-09-22 --n-boot 2000 --semilla 2026
python -m unittest tests.test_f1_variantes -v     # 26 pruebas
```

Tabla del esquema aleatorio (§8.3), por sistema (clave `sistemas.<clave>.aleatoria`):

| Sistema | n | top-1 | F1 macro (conjunto de prueba aleatorio) | IC 95 % bootstrap | F1 ponderado | P macro | R macro |
|---|---|---|---|---|---|---|---|
| Red, manos (C1, vigente) | 100 | 0,88 | 0,871 | [0,805; 0,913] | 0,871 | 0,897 | 0,880 |
| Red, manos + postura (C2) | 100 | 0,93 | 0,926 | [0,865; 0,968] | 0,926 | 0,950 | 0,930 |
| Red, manos + postura + rostro (C3) | 100 | 0,95 | 0,947 | [0,900; 0,979] | 0,947 | 0,963 | 0,950 |
| Línea base RF | 100 | 0,73 | 0,718 | [0,620; 0,769] | 0,718 | 0,757 | 0,730 |
| Línea base SVM | 100 | 0,51 | 0,496 | [0,392; 0,547] | 0,496 | 0,532 | 0,510 |
| Línea base MLP | 100 | 0,53 | 0,515 | [0,416; 0,565] | 0,515 | 0,560 | 0,530 |
| Canónico «manos» | 100 | 0,88 | 0,871 | [0,805; 0,913] | 0,871 | 0,897 | 0,880 |
