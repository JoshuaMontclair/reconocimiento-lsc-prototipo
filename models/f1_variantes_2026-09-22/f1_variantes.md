# Variantes del F1 sobre las predicciones guardadas (pieza 3, 2026-09-22)

Fecha: 2026-09-22T16:33:09. Script `src/f1_variantes.py`. Bootstrap: 2000 remuestreos, semilla 2026, IC percentil al 95 %. Macro sobre 50 señas, 0/0 = 0, DE entre pliegues con ddof = 1. Todas las cifras están en `f1_variantes.json` con la clave que se indica.

## 1. LOSO: las lecturas del F1, por sistema

| Sistema | F1 macro, media ± DE entre pliegues [IC boot pliegues+dentro] | F1 macro agregado (997) [IC boot] | F1 ponderado agregado | F1 micro agregado (= top-1) | top-1 media ± DE | P macro media / agregada | R macro media / agregada |
|---|---|---|---|---|---|---|---|
| Red temporal, manos (C1; N-30) | 0,5374 ± 0,0927 [0,455; 0,604] | 0,5737 [0,546; 0,598] | 0,5737 | 0,5797 | 0,5797 ± 0,0860 | 0,5574 / 0,5936 | 0,5810 / 0,5801 |
| Red temporal, manos + postura (C2; N-23/N-30) | 0,4976 ± 0,1708 [0,360; 0,629] | 0,5413 [0,511; 0,566] | 0,5412 | 0,5336 | 0,5334 ± 0,1757 | 0,5256 / 0,5885 | 0,5337 / 0,5338 |
| Red temporal, manos + postura + cejas/labios (C3; N-30) | 0,4644 ± 0,1105 [0,363; 0,539] | 0,5084 [0,481; 0,533] | 0,5084 | 0,5085 | 0,5085 ± 0,1087 | 0,4979 / 0,5497 | 0,5083 / 0,5084 |
| Línea base Random Forest (D-10, con espejo; N-39) | 0,3729 ± 0,1352 [0,253; 0,463] | 0,3933 [0,365; 0,419] | 0,3932 | 0,4183 | 0,4185 ± 0,1393 | 0,4040 / 0,4189 | 0,4183 / 0,4182 |
| Línea base SVM RBF (D-10, con espejo; N-39) | 0,2292 ± 0,1427 [0,103; 0,327] | 0,2824 [0,256; 0,305] | 0,2823 | 0,2638 | 0,2639 ± 0,1518 | 0,2651 / 0,3404 | 0,2640 / 0,2637 |
| Línea base MLP (D-10, con espejo; N-39) | 0,2219 ± 0,1219 [0,109; 0,297] | 0,2566 [0,232; 0,277] | 0,2566 | 0,2477 | 0,2478 ± 0,1326 | 0,2617 / 0,3022 | 0,2477 / 0,2478 |

Claves: `sistemas.<clave>.loso.por_pliegue.resumen.<métrica>.{media,de}`, `sistemas.<clave>.loso.agregado.<métrica>`, `sistemas.<clave>.loso.bootstrap.*`.

## 2. Intervalos para el F1 macro del LOSO (tres formas de leer la incertidumbre)

| Sistema | media ± DE | IC t (5 pliegues) | IC boot solo pliegues | IC boot pliegues + dentro | agregado | IC boot agregado |
|---|---|---|---|---|---|---|
| Red temporal, manos (C1; N-30) | 0,5374 ± 0,0927 | [0,422; 0,653] | [0,462; 0,607] | [0,455; 0,604] | 0,5737 | [0,546; 0,598] |
| Red temporal, manos + postura (C2; N-23/N-30) | 0,4976 ± 0,1708 | [0,286; 0,710] | [0,376; 0,624] | [0,360; 0,629] | 0,5413 | [0,511; 0,566] |
| Red temporal, manos + postura + cejas/labios (C3; N-30) | 0,4644 ± 0,1105 | [0,327; 0,602] | [0,377; 0,542] | [0,363; 0,539] | 0,5084 | [0,481; 0,533] |
| Línea base Random Forest (D-10, con espejo; N-39) | 0,3729 ± 0,1352 | [0,205; 0,541] | [0,263; 0,469] | [0,253; 0,463] | 0,3933 | [0,365; 0,419] |
| Línea base SVM RBF (D-10, con espejo; N-39) | 0,2292 ± 0,1427 | [0,052; 0,406] | [0,102; 0,329] | [0,103; 0,327] | 0,2824 | [0,256; 0,305] |
| Línea base MLP (D-10, con espejo; N-39) | 0,2219 ± 0,1219 | [0,071; 0,373] | [0,112; 0,302] | [0,109; 0,297] | 0,2566 | [0,232; 0,277] |

Anchura medida: el IC de la media entre pliegues (pliegues + dentro) es más ancho que el del agregado en 6 de 6 sistemas. El remuestreo del agregado trata las predicciones como independientes; el de la media remuestrea también los señantes, que son la unidad de réplica del LOSO.

## 3. Por qué difieren la media por pliegue y el agregado

| Sistema | media pliegues | agregado | diferencia | pares (pliegue, seña) con 0 aciertos / 250 | señas con algún pliegue a 0 | aporte de esas señas | aporte del resto | media ponderada por n − media simple |
|---|---|---|---|---|---|---|---|---|
| Red temporal, manos (C1; N-30) | 0,5374 | 0,5737 | 0,0363 | 65 | 31 | 0,0412 | -0,0049 | 0,00000 |
| Red temporal, manos + postura (C2; N-23/N-30) | 0,4976 | 0,5413 | 0,0437 | 77 | 36 | 0,0439 | -0,0002 | 0,00018 |
| Red temporal, manos + postura + cejas/labios (C3; N-30) | 0,4644 | 0,5084 | 0,0440 | 78 | 35 | 0,0428 | 0,0012 | 0,00004 |
| Línea base Random Forest (D-10, con espejo; N-39) | 0,3729 | 0,3933 | 0,0205 | 94 | 40 | 0,0298 | -0,0094 | -0,00021 |
| Línea base SVM RBF (D-10, con espejo; N-39) | 0,2292 | 0,2824 | 0,0532 | 141 | 49 | 0,0520 | 0,0012 | -0,00011 |
| Línea base MLP (D-10, con espejo; N-39) | 0,2219 | 0,2566 | 0,0347 | 142 | 49 | 0,0346 | 0,0001 | -0,00009 |

Identidad (comprobada numéricamente en cada sistema, clave `descomposicion.identidad_cierra`): el F1 agregado de una seña es Σ_k w_kc·F1_kc con w_kc = D_kc / Σ_k D_kc y D = 2·TP + FP + FN; la media por pliegue usa w = 1/5. Un pliegue en el que la seña no se acierta nunca tiene F1 = 0 y un D sin el término 2·TP, por lo que suele pesar menos de 1/5 en el agregado (no siempre: depende de sus falsos positivos) y pesa exactamente 1/5 en la media. Como D = 2·TP + FP + FN = (muestras reales de la seña) + (veces que se predijo la seña), el agregado pondera cada pliegue por cuántas veces aparece y se predice la seña allí.

| Sistema | soporte mínimo (pliegue, seña) | pares (pliegue, seña) sin muestras | pares nunca predichos / 250 | P macro media con 0/0 = 0 | P macro media sin las nunca predichas |
|---|---|---|---|---|---|
| Red temporal, manos (C1; N-30) | 3 | 0 | 44 | 0,5574 | 0,6742 |
| Red temporal, manos + postura (C2; N-23/N-30) | 3 | 0 | 45 | 0,5256 | 0,6306 |
| Red temporal, manos + postura + cejas/labios (C3; N-30) | 3 | 0 | 45 | 0,4979 | 0,6014 |
| Línea base Random Forest (D-10, con espejo; N-39) | 3 | 0 | 48 | 0,4040 | 0,4906 |
| Línea base SVM RBF (D-10, con espejo; N-39) | 3 | 0 | 87 | 0,2651 | 0,3606 |
| Línea base MLP (D-10, con espejo; N-39) | 3 | 0 | 75 | 0,2617 | 0,3415 |

Soporte mínimo medido en un (pliegue, seña): 3. Con soporte ≥ 1, el F1 de una seña sin aciertos es 0 bajo cualquier convención (FN > 0), así que la convención 0/0 = 0 no mueve el F1; sí mueve la precisión macro por pliegue (tabla anterior).

Medido: el agregado supera a la media por pliegue en 6 de 6 sistemas; el aporte de las señas con algún pliegue a 0 aciertos es mayor (en valor absoluto) que el del resto en 6 de 6; el efecto del tamaño desigual de los pliegues sobre la media es como mucho 0,00021 (última columna de la primera tabla de esta sección).

## 4. Orden entre sistemas bajo cada lectura

| Lectura | Ranking | Pares esperados que se cumplen |
|---|---|---|
| f1_macro_media_pliegues | red_manos > red_manos_postura > red_manos_postura_rostro > linea_base_random_forest > linea_base_svm_rbf > linea_base_mlp | 5 de 5 |
| f1_macro_agregado | red_manos > red_manos_postura > red_manos_postura_rostro > linea_base_random_forest > linea_base_svm_rbf > linea_base_mlp | 5 de 5 |
| f1_ponderado_agregado | red_manos > red_manos_postura > red_manos_postura_rostro > linea_base_random_forest > linea_base_svm_rbf > linea_base_mlp | 5 de 5 |
| f1_micro_agregado_top1 | red_manos > red_manos_postura > red_manos_postura_rostro > linea_base_random_forest > linea_base_svm_rbf > linea_base_mlp | 5 de 5 |
| top1_media_pliegues | red_manos > red_manos_postura > red_manos_postura_rostro > linea_base_random_forest > linea_base_svm_rbf > linea_base_mlp | 5 de 5 |
| precision_macro_media_pliegues | red_manos > red_manos_postura > red_manos_postura_rostro > linea_base_random_forest > linea_base_svm_rbf > linea_base_mlp | 5 de 5 |
| precision_macro_agregada | red_manos > red_manos_postura > red_manos_postura_rostro > linea_base_random_forest > linea_base_svm_rbf > linea_base_mlp | 5 de 5 |
| recall_macro_media_pliegues | red_manos > red_manos_postura > red_manos_postura_rostro > linea_base_random_forest > linea_base_svm_rbf > linea_base_mlp | 5 de 5 |
| recall_macro_agregado | red_manos > red_manos_postura > red_manos_postura_rostro > linea_base_random_forest > linea_base_svm_rbf > linea_base_mlp | 5 de 5 |

Mismo ranking bajo las cuatro lecturas del F1: **True** (clave `orden._mismo_ranking_en_todas_las_lecturas_de_f1`).

## 5. McNemar sobre aciertos top-1 (no usa F1)

| Comparación | solo acierta A | solo acierta B | discordantes | prueba | p | significativo (α = 0,05) |
|---|---|---|---|---|---|---|
| red_manos_vs_red_manos_postura | 134 | 88 | 222 | chi2 con corrección de continuidad | 2.53e-03 | True |
| red_manos_vs_red_manos_postura_rostro | 150 | 79 | 229 | chi2 con corrección de continuidad | 3.73e-06 | True |
| red_manos_vs_linea_base_random_forest | 249 | 88 | 337 | chi2 con corrección de continuidad | 2.89e-18 | True |
| red_manos_vs_linea_base_svm_rbf | 351 | 36 | 387 | chi2 con corrección de continuidad | 2.37e-57 | True |
| red_manos_vs_linea_base_mlp | 367 | 36 | 403 | chi2 con corrección de continuidad | 1.01e-60 | True |

McNemar compara, muestra a muestra, si cada sistema acertó la top-1. No lee ningún F1: cambiar la lectura del F1 (media o agregado) no altera ninguna tabla ni ningún p.

## 6. Esquema aleatorio (un solo conjunto de prueba de 100): solo agregado

| Sistema | n | top-1 | F1 macro agregado [IC boot] | F1 ponderado | P macro | R macro |
|---|---|---|---|---|---|---|
| Red temporal, manos (C1; N-30) | 100 | 0,880 | 0,8707 [0,805; 0,913] | 0,8707 | 0,8967 | 0,8800 |
| Red temporal, manos + postura (C2; N-23/N-30) | 100 | 0,930 | 0,9260 [0,865; 0,968] | 0,9260 | 0,9500 | 0,9300 |
| Red temporal, manos + postura + cejas/labios (C3; N-30) | 100 | 0,950 | 0,9473 [0,900; 0,979] | 0,9473 | 0,9633 | 0,9500 |
| Línea base Random Forest (D-10, con espejo; N-39) | 100 | 0,730 | 0,7180 [0,620; 0,769] | 0,7180 | 0,7567 | 0,7300 |
| Línea base SVM RBF (D-10, con espejo; N-39) | 100 | 0,510 | 0,4958 [0,392; 0,547] | 0,4958 | 0,5320 | 0,5100 |
| Línea base MLP (D-10, con espejo; N-39) | 100 | 0,530 | 0,5154 [0,416; 0,565] | 0,5154 | 0,5597 | 0,5300 |
| Modelo canónico «manos» (N-31; solo esquema aleatorio) | 100 | 0,880 | 0,8707 [0,805; 0,913] | 0,8707 | 0,8967 | 0,8800 |

Regla: con un solo pliegue no hay media ni DE entre pliegues; se reporta el agregado y se etiqueta como tal.

## 7. Pliegue a pliegue de la red «manos» (control frente a `resultados_loso.json`)

| Pliegue | señante | n | top-1 | P macro | R macro | F1 macro | F1 ponderado | señas con F1 = 0 | coincide con lo guardado |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0000 | 200 | 0,460 | 0,4386 | 0,4600 | 0,4207 | 0,4207 | 17 | True |
| 1 | 0001 | 199 | 0,533 | 0,4925 | 0,5350 | 0,4629 | 0,4615 | 16 | True |
| 2 | 0002 | 198 | 0,606 | 0,6022 | 0,6100 | 0,5728 | 0,5721 | 10 | True |
| 3 | 0003 | 200 | 0,615 | 0,6045 | 0,6150 | 0,5852 | 0,5852 | 11 | True |
| 4 | 0004 | 200 | 0,685 | 0,6492 | 0,6850 | 0,6455 | 0,6455 | 11 | True |

Tiempo de cálculo: 3.4 s.
