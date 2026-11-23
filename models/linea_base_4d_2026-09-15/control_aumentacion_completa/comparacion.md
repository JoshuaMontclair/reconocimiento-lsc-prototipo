# Línea base clásica frente a la red temporal (actividad 4.d, D-10)

Generado por `src/linea_base_clasica.py` el 2026-09-16T07:29:38. Todo lo que sigue sale de esta corrida; el criterio estaba fijado en D-10 antes de correr.

## 0. Lo que se comprobó antes de comparar

- Tensor de entrada: [30, 252] por muestra (30 pasos x 252 rasgos: 126 coordenadas de las dos manos + 126 velocidades), idéntico valor a valor al que produce el entrenador de la red para 0000_0000_0000, 0025_0000_0000, 0049_0004_0003.
- Cifra de la red (N-30) reproducida desde sus `predicciones.json`: 0,5797 ± 0,0860 top-1 LOSO (esperado 0,580 ± 0,086). Agregado: 0,5797 (578/997), top-3 0,7703.
- Características: 2016 (8 estadísticos x 252 columnas), sin PCA ni selección; escalado estándar ajustado solo con el entrenamiento de cada pliegue.

## 1. LOSO agregado (las 997 predicciones de prueba juntas), con la aumentación completa de la red (factor 10)

| sistema | top-1 | top-3 | precisión | recall | F1 macro |
|---|---|---|---|---|---|
| Red temporal (CNN+BiLSTM+atención, N-30) | 0,580 | 0,770 | 0,594 | 0,580 | 0,574 |
| SVM RBF | 0,273 | 0,430 | 0,368 | 0,273 | 0,296 |
| Random Forest | 0,442 | 0,658 | 0,449 | 0,442 | 0,422 |
| MLP | 0,288 | 0,468 | 0,322 | 0,288 | 0,285 |

## 2. LOSO por pliegue (top-1) y media entre pliegues

| sistema | p0 (0000) | p1 (0001) | p2 (0002) | p3 (0003) | p4 (0004) | media ± DE |
|---|---|---|---|---|---|---|
| Red temporal | 0,460 | 0,533 | 0,606 | 0,615 | 0,685 | 0,580 ± 0,086 |
| SVM RBF | 0,025 | 0,221 | 0,303 | 0,355 | 0,460 | 0,273 ± 0,163 |
| Random Forest | 0,215 | 0,337 | 0,556 | 0,525 | 0,580 | 0,442 ± 0,159 |
| MLP | 0,095 | 0,256 | 0,303 | 0,350 | 0,435 | 0,288 ± 0,126 |

## 3. Control sin aumentación de espejo (LOSO agregado)

No ejecutado en esta corrida.

## 4. Configuración elegida por pliegue (top-1 de validación; la prueba no se miró)

| familia | p0 | p1 | p2 | p3 | p4 | aleatoria |
|---|---|---|---|---|---|---|
| SVM RBF | C=10,gamma=scale | C=10,gamma=scale | C=1,gamma=scale | C=1,gamma=scale | C=10,gamma=scale | — |
| Random Forest | min_samples_leaf=2 | min_samples_leaf=1 | min_samples_leaf=1 | min_samples_leaf=1 | min_samples_leaf=2 | — |
| MLP | alpha=1e-2 | alpha=1e-4 | alpha=1e-2 | alpha=1e-2 | alpha=1e-4 | — |

## 6. McNemar frente a la red (997 predicciones LOSO agregadas, α = 0,05)

| línea base | aciertos red | aciertos línea base | solo red | solo línea base | prueba | p |
|---|---|---|---|---|---|---|
| SVM RBF | 578 | 272 | 345 | 39 | chi2 con corrección de continuidad | 1,27e-54 |
| Random Forest | 578 | 441 | 217 | 80 | chi2 con corrección de continuidad | 2,99e-15 |
| MLP | 578 | 287 | 333 | 42 | chi2 con corrección de continuidad | 1,06e-50 |

## 7. Conclusión

- La arquitectura temporal QUEDA JUSTIFICADA empíricamente según el criterio de D-10: top-1 agregado de la red 0,580 > 0,442 de la mejor línea base (Random Forest) y McNemar p = 2,99e-15 < 0,05.
- Mejor línea base por top-1 LOSO agregado: Random Forest (0,442); red 0,580.
- Concordancia: 361 muestras las aciertan los dos, 339 las fallan los dos, 217 solo la red y 80 solo la línea base.
- La red supera a Random Forest en 5 de los 5 pliegues LOSO, así que la ventaja no viene de un solo señante.
- Orden de las tres familias por top-1 LOSO agregado: Random Forest 0,442, MLP 0,288, SVM RBF 0,273. Los árboles resisten mejor los 2.016 rasgos que el SVM y el MLP; los tres pierden lo mismo en el señante 0000, el difícil desde N-23.
- Esta corrida no incluye el control de aumentación, así que no mide cuánto aporta la aumentación; esa cifra está en la corrida principal.
- Lectura prudente: esto NO dice que un clasificador clásico no sirva para LSC; dice que, con esta entrada (ocho estadísticos por columna, que borran el orden temporal) y este corpus, resumir la secuencia cuesta caro. Esta corrida no ejecutó el esquema aleatorio, así que no dice nada sobre la brecha entre señantes vistos y nuevos; esa comparación está en la corrida principal. (Frase corregida a mano el 16-sep tras la verificación independiente: el texto anterior afirmaba algo que las cifras de esta corrida no sostienen.)

## 8. Límites de esta comparación

- Límite declarado en D-10: la línea base recibe la secuencia resumida en estadísticos y pierde el orden temporal salvo lo que capturan el primer paso, el último y la media de las diferencias entre pasos consecutivos.
- Esta corrida SÍ entrena la línea base con las nueve transformaciones de la red (factor 10), que es lo que la hace comparable en aumentación; D-10 solo pedía el espejo, así que la corrida de referencia de D-10 sigue siendo la de espejo.
- 997 muestras y 5 señantes: la dispersión entre pliegues (señantes) es grande en los dos sistemas y domina cualquier diferencia de décimas.
- McNemar compara aciertos pareados sobre las mismas muestras; no dice nada sobre latencia, coste de entrenamiento ni comportamiento en el prototipo en vivo.

