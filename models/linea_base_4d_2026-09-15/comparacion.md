# Línea base clásica frente a la red temporal (actividades 2.c y 2.d, D-10)

Generado por `src/linea_base_clasica.py` el 2026-09-16T08:44:53. Todo lo que sigue sale de esta corrida; el criterio estaba fijado en D-10 antes de correr.

## 0. Lo que se comprobó antes de comparar

- Tensor de entrada: [30, 252] por muestra (30 pasos x 252 rasgos: 126 coordenadas de las dos manos + 126 velocidades), idéntico valor a valor al que produce el entrenador de la red para 0000_0000_0000, 0025_0000_0000, 0049_0004_0003.
- Cifra de la red (N-30) reproducida desde sus `predicciones.json`: 0,5797 ± 0,0860 top-1 LOSO (esperado 0,580 ± 0,086). Agregado: 0,5797 (578/997), top-3 0,7703.
- Características: 2016 (8 estadísticos x 252 columnas), sin PCA ni selección; escalado estándar ajustado solo con el entrenamiento de cada pliegue.

## 1. LOSO agregado (las 997 predicciones de prueba juntas), con aumentación de espejo

| sistema | top-1 | top-3 | precisión | recall | F1 macro |
|---|---|---|---|---|---|
| Red temporal (CNN+BiLSTM+atención, N-30) | 0,580 | 0,770 | 0,594 | 0,580 | 0,574 |
| SVM RBF | 0,264 | 0,401 | 0,340 | 0,264 | 0,282 |
| Random Forest | 0,418 | 0,651 | 0,419 | 0,418 | 0,393 |
| MLP | 0,248 | 0,420 | 0,302 | 0,248 | 0,257 |

## 2. LOSO por pliegue (top-1) y media entre pliegues

| sistema | p0 (0000) | p1 (0001) | p2 (0002) | p3 (0003) | p4 (0004) | media ± DE |
|---|---|---|---|---|---|---|
| Red temporal | 0,460 | 0,533 | 0,606 | 0,615 | 0,685 | 0,580 ± 0,086 |
| SVM RBF | 0,025 | 0,221 | 0,318 | 0,330 | 0,425 | 0,264 ± 0,152 |
| Random Forest | 0,195 | 0,377 | 0,545 | 0,500 | 0,475 | 0,418 ± 0,139 |
| MLP | 0,035 | 0,221 | 0,278 | 0,325 | 0,380 | 0,248 ± 0,133 |

## 3. Control sin aumentación de espejo (LOSO agregado)

| familia | top-1 con espejo | top-1 sin espejo | diferencia |
|---|---|---|---|
| SVM RBF | 0,264 | 0,236 | +0,028 |
| Random Forest | 0,418 | 0,405 | +0,013 |
| MLP | 0,248 | 0,224 | +0,024 |

## 4. Configuración elegida por pliegue (top-1 de validación; la prueba no se miró)

| familia | p0 | p1 | p2 | p3 | p4 | aleatoria |
|---|---|---|---|---|---|---|
| SVM RBF | C=10,gamma=scale | C=10,gamma=scale | C=10,gamma=scale | C=100,gamma=scale | C=10,gamma=scale | C=10,gamma=scale |
| Random Forest | min_samples_leaf=1 | min_samples_leaf=2 | min_samples_leaf=1 | min_samples_leaf=1 | min_samples_leaf=2 | min_samples_leaf=1 |
| MLP | alpha=1e-2 | alpha=1e-2 | alpha=1e-4 | alpha=1e-2 | alpha=1e-2 | alpha=1e-2 |

## 5. Esquema aleatorio 797/100/100 (señantes vistos), con aumentación de espejo

| sistema | top-1 | top-3 | precisión | recall | F1 macro |
|---|---|---|---|---|---|
| Red temporal (N-30) | 0,880 | 0,980 | 0,897 | 0,880 | 0,871 |
| SVM RBF | 0,510 | 0,730 | 0,532 | 0,510 | 0,496 |
| Random Forest | 0,730 | 0,890 | 0,757 | 0,730 | 0,718 |
| MLP | 0,530 | 0,750 | 0,560 | 0,530 | 0,515 |

## 6. McNemar frente a la red (997 predicciones LOSO agregadas, α = 0,05)

| línea base | aciertos red | aciertos línea base | solo red | solo línea base | prueba | p |
|---|---|---|---|---|---|---|
| SVM RBF | 578 | 263 | 351 | 36 | chi2 con corrección de continuidad | 2,37e-57 |
| Random Forest | 578 | 417 | 249 | 88 | chi2 con corrección de continuidad | 2,89e-18 |
| MLP | 578 | 247 | 367 | 36 | chi2 con corrección de continuidad | 1,01e-60 |

## 7. Conclusión

- La arquitectura temporal QUEDA JUSTIFICADA empíricamente según el criterio de D-10: top-1 agregado de la red 0,580 > 0,418 de la mejor línea base (Random Forest) y McNemar p = 2,89e-18 < 0,05.
- Mejor línea base por top-1 LOSO agregado: Random Forest (0,418); red 0,580.
- Concordancia: 329 muestras las aciertan los dos, 331 las fallan los dos, 249 solo la red y 88 solo la línea base.
- La red supera a Random Forest en 5 de los 5 pliegues LOSO, así que la ventaja no viene de un solo señante.
- Orden de las tres familias por top-1 LOSO agregado: Random Forest 0,418, SVM RBF 0,264, MLP 0,248. Peor pliegue de cada sistema: red 00 (0,460); Random Forest 00 (0,195); SVM RBF 00 (0,025); MLP 00 (0,035).
- La aumentación de espejo aporta como mucho 0,028 de top-1 agregado: la distancia con la red no se explica por la aumentación.
- Lectura prudente: esto NO dice que un clasificador clásico no sirva para LSC; dice que, con esta entrada (ocho estadísticos por columna, que borran el orden temporal) y este corpus, resumir la secuencia cuesta caro. Sigue en pie la brecha entre señantes vistos y no vistos: en el esquema aleatorio la línea base también sube mucho, así que el problema de fondo (generalizar a un señante nuevo con 5 señantes) lo tienen los dos sistemas.

## 8. Límites de esta comparación

- Límite declarado en D-10: la línea base recibe la secuencia resumida en estadísticos y pierde el orden temporal salvo lo que capturan el primer paso, el último y la media de las diferencias entre pasos consecutivos.
- La línea base no se entrena sobre las nueve transformaciones de la red (factor 10), solo sobre el espejo, que es lo que dice D-10; el control sin espejo acota cuánto aporta.
- 997 muestras y 5 señantes: la dispersión entre pliegues (señantes) es grande en los dos sistemas y domina cualquier diferencia de décimas.
- McNemar compara aciertos pareados sobre las mismas muestras; no dice nada sobre latencia, coste de entrenamiento ni comportamiento en el prototipo en vivo.

## 9. Control de equidad: misma aumentación que la red

*Sección escrita a mano el 16-sep-2026 con las cifras de `control_aumentacion_completa/` (corrida aparte). El script no la genera: si se vuelve a ejecutar `src/linea_base_clasica.py` sobre este directorio, esta sección se pierde y hay que volver a añadirla.*

D-10 mandó entrenar la línea base con la aumentación de espejo (factor 2), mientras que la red se entrena con las nueve transformaciones de `src/train_loso.py` (factor 10). Para que la comparación no favorezca a la red por esa asimetría, se repitió el LOSO dando a la línea base exactamente esas mismas transformaciones (`--aumentacion completa`). El criterio de D-10 no se cambió: es un control añadido, no una elección nueva.

| familia | top-1 con espejo (D-10) | top-1 con la aumentación de la red | diferencia |
|---|---|---|---|
| Random Forest | 0,418 | 0,442 | +0,024 |
| Perceptrón multicapa | 0,248 | 0,288 | +0,040 |
| Máquina de vectores de soporte | 0,264 | 0,273 | +0,009 |

Red temporal: 0,580 (sin cambios; no se reentrenó).

McNemar de la mejor línea base con aumentación completa (Random Forest, top-1 0,442) frente a la red: solo la red acierta 217 muestras, solo la línea base 80, p = 2.99e-15 < 0,05.

Lectura: igualar la aumentación sube a la línea base entre 1 y 4 centésimas y no cambia la conclusión. La distancia con la red (0,580 frente a 0,442) no se explica por la aumentación, sino por lo que dice el límite de D-10: resumir la secuencia en estadísticos borra el orden temporal.
