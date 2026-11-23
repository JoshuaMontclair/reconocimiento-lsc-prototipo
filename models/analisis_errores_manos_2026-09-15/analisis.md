# Análisis de errores del modelo «manos» (actividad 4.d)

Generado por `src/analisis_errores.py` el 2026-09-16T18:57:03. SOLO REPORTE: no toma decisiones ni reentrena.
Datos: predicciones de prueba ya guardadas en `models/loso_2026-09-15_holistic_mascara_manos/pliegue_k/predicciones.json` (el pliegue k deja fuera al señante k), 997 predicciones. Todas las cifras están en `analisis.json`; cada tabla indica la clave de donde sale. Decimales con coma.

## 0. Controles (antes de reportar)

Fuente: `analisis.json › controles`.

| pliegue | señante | n | top-1 calculado | top-1 guardado | top-3 | F1 macro | matriz y recall por clase |
|---|---|---|---|---|---|---|---|
| 0 | 0000 | 200 | 0,460 | 0,460 | 0,650 (coincide) | 0,421 (coincide) | coinciden |
| 1 | 0001 | 199 | 0,533 | 0,533 | 0,714 (coincide) | 0,463 (coincide) | coinciden |
| 2 | 0002 | 198 | 0,606 | 0,606 | 0,788 (coincide) | 0,573 (coincide) | coinciden |
| 3 | 0003 | 200 | 0,615 | 0,615 | 0,830 (coincide) | 0,585 (coincide) | coinciden |
| 4 | 0004 | 200 | 0,685 | 0,685 | 0,870 (coincide) | 0,645 (coincide) | coinciden |

Top-1 media ± DE de los pliegues: 0,580 ± 0,086 (guardado en resultados_loso.json: 0,580 ± 0,086; coincide). 
Umbral sobre la prueba LOSO (t = 0,20; frente a N-33, 540 / 297: coincide; fuente `calibracion.umbral.global`): rechazo 16,0 %, exactitud entre aceptadas 0,645, aceptadas 540 correctas / 297 erróneas, aciertos perdidos 6,6 %.
Control de los grupos de N-30 (recall agregado, 3 decimales; coincide): C1 0,379 / 0,688 / 0,283 / 0,632; C2 0,264 / 0,562 / 0,350 / 0,598; C3 0,357 / 0,588 / 0,167 / 0,558 (interrogativos / emociones / valoración / resto; N-30 registra C1 0,379 / 0,688 / 0,283 / 0,632; C2 0,264 / 0,562 / 0,350 / 0,598; C3 0,357 / 0,588 / 0,167 / 0,558).
Control de fotogramas sin mano: METRICAS_FOTOGRAMA frente a bloques en ceros de X_padded en las 997 muestras: coinciden; filas por muestra frente a lengths.npy: coinciden.

Agregado de las 997 predicciones (no es la media de pliegues): top-1 0,580, top-3 0,770, F1 macro 0,574, recall macro 0,580. Fuente: `global`, `por_sena`.

## 1. Métricas por seña

Fuente: `por_sena.senas` (agregado sobre los pliegues; soporte 20 por seña, salvo 19 en la seña 2, 19 en la seña 8, 19 en la seña 16). DE = desviación estándar muestral entre pliegues (4 muestras por seña en cada pliegue, salvo pliegue 1: 3 en la seña 2; pliegue 2: 3 en la seña 8, 3 en la seña 16: cada muestra mueve el recall del pliegue en 0,25 o más). Precisión indefinida (seña nunca predicha) = «—».

**Peores 10** (recall agregado ascendente; empate: F1 agregado ascendente, luego id):

| seña | categoría | recall | recall por pliegue (0..4) | DE recall | precisión | DE precisión (pliegues definidos) | F1 | DE F1 | veces predicha |
|---|---|---|---|---|---|---|---|---|---|
| más o menos (27) | adverbio | 0,000 | 0,00 / 0,00 / 0,00 / 0,00 / 0,00 | 0,000 | 0,000 | — (1) | 0,000 | 0,000 | 4 |
| comer (7) | verbo | 0,050 | 0,00 / 0,00 / 0,00 / 0,25 / 0,00 | 0,112 | 0,053 | 0,033 (4) | 0,051 | 0,047 | 19 |
| permiso (36) | objeto | 0,150 | 0,00 / 0,00 / 0,75 / 0,00 / 0,00 | 0,335 | 0,115 | 0,500 (4) | 0,130 | 0,383 | 26 |
| niño (14) | sujeto | 0,150 | 0,50 / 0,00 / 0,00 / 0,25 / 0,00 | 0,224 | 0,125 | 0,120 (4) | 0,136 | 0,124 | 24 |
| ¿cómo estás? (26) | pregunta | 0,150 | 0,25 / 0,00 / 0,25 / 0,25 / 0,00 | 0,137 | 0,150 | 0,144 (4) | 0,150 | 0,129 | 20 |
| ¿cuándo? (42) | pregunta | 0,200 | 0,25 / 0,75 / 0,00 / 0,00 / 0,00 | 0,326 | 0,364 | 0,388 (3) | 0,258 | 0,325 | 11 |
| mujer (12) | sujeto | 0,250 | 0,00 / 1,00 / 0,00 / 0,25 / 0,00 | 0,433 | 0,333 | 0,495 (2) | 0,286 | 0,387 | 15 |
| bien (24) | adverbio | 0,300 | 0,25 / 0,25 / 0,25 / 0,50 / 0,25 | 0,112 | 0,600 | 0,303 (5) | 0,400 | 0,040 | 10 |
| sentir (28) | verbo | 0,300 | 0,00 / 0,00 / 0,50 / 0,00 / 1,00 | 0,447 | 0,857 | 0,141 (2) | 0,444 | 0,433 | 7 |
| niña (15) | sujeto | 0,350 | 0,25 / 0,00 / 0,50 / 0,00 / 1,00 | 0,418 | 0,269 | 0,213 (5) | 0,304 | 0,282 | 26 |

**Mejores 10:**

| seña | categoría | recall | recall por pliegue (0..4) | DE recall | precisión | DE precisión (pliegues definidos) | F1 | DE F1 | veces predicha |
|---|---|---|---|---|---|---|---|---|---|
| todos (48) | sujeto | 1,000 | 1,00 / 1,00 / 1,00 / 1,00 / 1,00 | 0,000 | 1,000 | 0,000 (5) | 1,000 | 0,000 | 20 |
| buenas noches (3) | saludo | 1,000 | 1,00 / 1,00 / 1,00 / 1,00 / 1,00 | 0,000 | 1,000 | 0,000 (5) | 1,000 | 0,000 | 20 |
| buenos días (1) | saludo | 1,000 | 1,00 / 1,00 / 1,00 / 1,00 / 1,00 | 0,000 | 1,000 | 0,000 (5) | 1,000 | 0,000 | 20 |
| juicioso (29) | adjetivo | 1,000 | 1,00 / 1,00 / 1,00 / 1,00 / 1,00 | 0,000 | 0,909 | 0,149 (5) | 0,952 | 0,089 | 22 |
| seña (4) | objeto | 1,000 | 1,00 / 1,00 / 1,00 / 1,00 / 1,00 | 0,000 | 0,909 | 0,110 (5) | 0,952 | 0,061 | 22 |
| diferente (47) | adjetivo | 1,000 | 1,00 / 1,00 / 1,00 / 1,00 / 1,00 | 0,000 | 0,800 | 0,178 (5) | 0,889 | 0,112 | 25 |
| buenas tardes (2) | saludo | 1,000 | 1,00 / 1,00 / 1,00 / 1,00 / 1,00 | 0,000 | 0,679 | 0,296 (5) | 0,809 | 0,211 | 28 |
| familia (10) | sujeto | 0,950 | 1,00 / 1,00 / 0,75 / 1,00 / 1,00 | 0,112 | 0,950 | 0,089 (5) | 0,950 | 0,070 | 20 |
| nombre (5) | objeto | 0,950 | 0,75 / 1,00 / 1,00 / 1,00 / 1,00 | 0,112 | 0,905 | 0,110 (5) | 0,927 | 0,068 | 21 |
| feliz (20) | adjetivo | 0,900 | 0,75 / 1,00 / 0,75 / 1,00 / 1,00 | 0,137 | 0,750 | 0,235 (5) | 0,818 | 0,113 | 24 |

Señas con recall agregado 0: más o menos (27). Señas nunca predichas en las 997: ninguna.
La tabla completa de las 50 señas está en `analisis.json › por_sena.senas`; la figura `recall_por_sena.png` muestra el agregado y los 5 pliegues.

## 2. Pares de confusión

Fuente: `confusiones`. 419 errores repartidos en 198 pares real → predicha distintos; 60 pares aparecen 3 o más veces. Criterio: simétrica si los dos sentidos tienen al menos 2 casos y el cociente menor / mayor es al menos 0,50 (convención descriptiva de este análisis). Proporción de errores en pares con al menos un caso en el sentido inverso: 43,4 %; en pares simétricos según el criterio: 29,4 % (18 parejas).

**Pares dirigidos más frecuentes** (`confusiones.pares`, los 20 primeros):

| real → predicha | conteo | inverso | menor / mayor | simétrica | misma categoría | señantes en que ocurre |
|---|---|---|---|---|---|---|
| más o menos (27) → poco (9) | 8 | 0 | 0,00 | no | sí (adverbio) | 0001, 0003 |
| más o menos (27) → yo (38) | 8 | 1 | 0,12 | no | adverbio → sujeto | 0002, 0004 |
| niña (15) → niño (14) | 6 | 5 | 0,83 | sí | sí (sujeto) | 0000, 0003 |
| ¿cómo estás? (26) → ¿cómo? (44) | 6 | 5 | 0,83 | sí | sí (pregunta) | 0002, 0003 |
| hola (30) → hombre (13) | 6 | 1 | 0,17 | no | saludo → sujeto | 0000, 0002, 0003 |
| permiso (36) → gracias (0) | 6 | 1 | 0,17 | no | objeto → cortesia | 0001, 0002, 0004 |
| ¿qué? (41) → ¿dónde? (43) | 6 | 3 | 0,50 | sí | sí (pregunta) | 0001, 0003, 0004 |
| niño (14) → niña (15) | 5 | 6 | 0,83 | sí | sí (sujeto) | 0002, 0004 |
| abuelo (16) → comer (7) | 5 | 4 | 0,80 | sí | sujeto → verbo | 0003, 0004 |
| tú (39) → niño (14) | 5 | 3 | 0,60 | sí | sí (sujeto) | 0000, 0004 |
| ¿cuándo? (42) → permiso (36) | 5 | 3 | 0,60 | sí | pregunta → objeto | 0001, 0004 |
| ¿cómo? (44) → ¿cómo estás? (26) | 5 | 6 | 0,83 | sí | sí (pregunta) | 0000, 0002, 0003 |
| gracias (0) → vivir (8) | 4 | 0 | 0,00 | no | cortesia → verbo | 0001 |
| comer (7) → abuelo (16) | 4 | 5 | 0,80 | sí | verbo → sujeto | 0000, 0002 |
| comer (7) → permiso (36) | 4 | 0 | 0,00 | no | verbo → objeto | 0004 |
| mujer (12) → permiso (36) | 4 | 4 | 1,00 | sí | sujeto → objeto | 0004 |
| hermano (18) → tío (17) | 4 | 2 | 0,50 | sí | sí (sujeto) | 0001, 0004 |
| mal (25) → adiós (31) | 4 | 1 | 0,25 | no | adverbio → saludo | 0000, 0002 |
| más o menos (27) → ¿qué? (41) | 4 | 0 | 0,00 | no | adverbio → pregunta | 0000 |
| sentir (28) → por favor (32) | 4 | 0 | 0,00 | no | verbo → cortesia | 0001 |

**Parejas no ordenadas con más errores** (`confusiones.parejas_no_ordenadas`, 10 primeras):

| pareja {a, b} | a → b | b → a | total | simétrica |
|---|---|---|---|---|
| niño (14) / niña (15) | 5 | 6 | 11 | sí |
| ¿cómo estás? (26) / ¿cómo? (44) | 6 | 5 | 11 | sí |
| comer (7) / abuelo (16) | 4 | 5 | 9 | sí |
| más o menos (27) / yo (38) | 8 | 1 | 9 | no |
| ¿qué? (41) / ¿dónde? (43) | 6 | 3 | 9 | sí |
| poco (9) / más o menos (27) | 0 | 8 | 8 | no |
| mujer (12) / permiso (36) | 4 | 4 | 8 | sí |
| niño (14) / tú (39) | 3 | 5 | 8 | sí |
| permiso (36) / ¿cuándo? (42) | 3 | 5 | 8 | sí |
| gracias (0) / perdón (35) | 3 | 4 | 7 | sí |

**Por categoría gramatical** (`confusiones.por_categoria`): 124 de 419 errores (29,6 %) caen en otra seña de la misma categoría; si cada error fuera a una de las otras 49 señas al azar se esperaría 13,6 %.

Matriz de errores categoría real (fila) → categoría predicha (columna):

| real \ predicha | saludo | cortesia | pregunta | sujeto | verbo | objeto | adjetivo | adverbio | errores | % dentro |
|---|---|---|---|---|---|---|---|---|---|---|
| saludo (5) | 1 | 0 | 0 | 20 | 0 | 0 | 0 | 1 | 22 | 4,5 % |
| cortesia (4) | 0 | 11 | 5 | 1 | 6 | 2 | 0 | 0 | 25 | 44,0 % |
| pregunta (7) | 8 | 0 | 42 | 7 | 9 | 11 | 9 | 1 | 87 | 48,3 % |
| sujeto (13) | 9 | 5 | 13 | 52 | 9 | 7 | 5 | 11 | 111 | 46,8 % |
| verbo (4) | 5 | 7 | 8 | 7 | 0 | 9 | 4 | 7 | 47 | 0,0 % |
| objeto (4) | 2 | 7 | 4 | 7 | 0 | 0 | 0 | 0 | 20 | 0,0 % |
| adjetivo (7) | 3 | 0 | 9 | 8 | 5 | 0 | 3 | 3 | 31 | 9,7 % |
| adverbio (6) | 5 | 8 | 6 | 28 | 5 | 4 | 5 | 15 | 76 | 19,7 % |

Señas que más reciben predicciones erróneas (`confusiones.predichas_mas_frecuentes_en_errores`): permiso (36) 23; niño (14) 21; niña (15) 19; abuelo (16) 19; por favor (32) 19; ¿qué? (41) 19.

## 3. Por señante

Fuente: `por_senante` (pliegue k = señante k fuera del entrenamiento).

| señante | n | top-1 | top-3 | F1 macro | errores | señas con recall 0 (de 4) | seña más predicha (veces) | confianza media aciertos / errores | errores con confianza ≥ 0,5 | errores con la real en top-3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0000 | 200 | 0,460 | 0,650 | 0,421 | 108 | 17 | hola (30) (19) | 0,534 / 0,271 | 12 | 38 |
| 0001 | 199 | 0,533 | 0,714 | 0,463 | 93 | 16 | por favor (32) (23) | 0,675 / 0,462 | 39 | 36 |
| 0002 | 198 | 0,606 | 0,788 | 0,573 | 78 | 10 | mal (25) (11) | 0,609 / 0,360 | 21 | 36 |
| 0003 | 200 | 0,615 | 0,830 | 0,585 | 77 | 11 | comer (7) (15) | 0,696 / 0,429 | 27 | 43 |
| 0004 | 200 | 0,685 | 0,870 | 0,645 | 63 | 11 | permiso (36) (21) | 0,715 / 0,483 | 30 | 37 |

Señas con recall 0 y pares más frecuentes de cada señante:

- **0000**: recall 0 en gracias (0), comer (7), poco (9), mujer (12), abuelo (16), triste (22), aburrido (23), mal (25), más o menos (27), sentir (28), adiós (31), permiso (36), nunca (37), yo (38), tú (39), ¿dónde? (43), mucho (49). Pares: más o menos (27) → ¿qué? (41) ×4; gracias (0) → perdón (35) ×3; poco (9) → niño (14) ×3; niña (15) → niño (14) ×3; aburrido (23) → niño (14) ×3.
- **0001**: recall 0 en gracias (0), comer (7), niño (14), niña (15), aburrido (23), mal (25), ¿cómo estás? (26), más o menos (27), sentir (28), hola (30), adiós (31), con gusto (33), perdón (35), permiso (36), nunca (37), yo (38). Pares: gracias (0) → vivir (8) ×4; más o menos (27) → poco (9) ×4; sentir (28) → por favor (32) ×4; hola (30) → abuelo (16) ×4; con gusto (33) → por favor (32) ×4.
- **0002**: recall 0 en trabajar (6), comer (7), poco (9), mujer (12), niño (14), más o menos (27), bienvenido (34), ¿qué? (41), ¿cuándo? (42), ¿quién? (46). Pares: más o menos (27) → yo (38) ×4; ¿cuándo? (42) → ¿cómo estás? (26) ×4; comer (7) → abuelo (16) ×3; poco (9) → mucho (49) ×3; mujer (12) → niña (15) ×3.
- **0003**: recall 0 en niña (15), abuelo (16), más o menos (27), sentir (28), hola (30), permiso (36), tú (39), ustedes (40), ¿qué? (41), ¿cuándo? (42), ¿quién? (46). Pares: más o menos (27) → poco (9) ×4; hola (30) → hombre (13) ×4; permiso (36) → mujer (12) ×4; ustedes (40) → nunca (37) ×4; ¿quién? (46) → comer (7) ×4.
- **0004**: recall 0 en comer (7), poco (9), personas (11), mujer (12), niño (14), abuelo (16), ¿cómo estás? (26), más o menos (27), permiso (36), ¿cuándo? (42), ¿por qué? (45). Pares: comer (7) → permiso (36) ×4; mujer (12) → permiso (36) ×4; niño (14) → niña (15) ×4; más o menos (27) → yo (38) ×4; permiso (36) → gracias (0) ×4.

### Variables medidas por señante (¿qué distingue al 0000?)

Fuente: `evidencia_senantes`. Longitudes: `lengths.npy` (fotogramas antes de remuestrear). Fotogramas sin mano: `data/extracted/LSC50_holistic/METRICAS_FOTOGRAMA/*.csv` (columnas mano_izq_detectada y mano_der_detectada del extractor; izq/der son los bloques del extractor). Movimiento: desplazamiento medio por fotograma de la muñeca en el espacio normalizado por hombros (`X_padded.npy`, solo fotogramas consecutivos con la mano presente). Solo muestras de prueba de cada señante.

| variable | 0000 | 0001 | 0002 | 0003 | 0004 | menor | mayor |
|---|---|---|---|---|---|---|---|
| top-1 | 0,460 | 0,533 | 0,606 | 0,615 | 0,685 | 0000 | 0004 |
| longitud media (fotogramas) | 72,2 | 70,4 | 66,7 | 56,4 | 71,0 | 0003 | 0000 |
| longitud mediana (fotogramas) | 71,0 | 70,0 | 66,0 | 56,0 | 72,0 | 0003 | 0004 |
| DE de la longitud | 9,3 | 9,6 | 10,0 | 6,6 | 11,3 | 0003 | 0004 |
| longitud / media de los otros 4 en la misma seña | 1,096 | 1,059 | 0,989 | 0,806 | 1,070 | 0003 | 0000 |
| fotogramas sin mano izq. | 6,7 % | 1,3 % | 14,3 % | 5,9 % | 3,0 % | 0001 | 0002 |
| fotogramas sin mano der. | 5,5 % | 2,5 % | 2,3 % | 2,3 % | 1,1 % | 0004 | 0000 |
| fotogramas sin alguna mano | 11,3 % | 3,7 % | 16,3 % | 7,9 % | 3,7 % | 0004 | 0002 |
| fotogramas sin ambas manos | 0,8 % | 0,1 % | 0,4 % | 0,4 % | 0,5 % | 0001 | 0000 |
| desplaz. muñeca izq. por fotograma | 0,074 | 0,040 | 0,075 | 0,064 | 0,060 | 0001 | 0002 |
| desplaz. muñeca der. por fotograma | 0,049 | 0,069 | 0,034 | 0,029 | 0,030 | 0003 | 0001 |
| proporción del recorrido en el bloque der. | 39,9 % | 63,1 % | 35,1 % | 32,9 % | 33,6 % | 0003 | 0001 |
| confianza media (salida del modelo) | 0,392 | 0,575 | 0,511 | 0,593 | 0,642 | 0000 | 0004 |

**Nivel de muestra, dentro de cada señante por separado** (`evidencia_senantes.nivel_muestra.<medida>.por_senante`; p bilateral por permutación de acierto/error entre las muestras de ese señante; orientativo):

| medida | señante | media aciertos | media errores | muestras con valor > 0 | exactitud si 0 | exactitud si > 0 | p |
|---|---|---|---|---|---|---|---|
| proporcion_sin_alguna_mano | 0000 | 11,3 % | 11,9 % | 177 | 0,391 | 0,469 | 0,6377 |
| proporcion_sin_alguna_mano | 0001 | 3,8 % | 3,6 % | 115 | 0,464 | 0,583 | 0,8424 |
| proporcion_sin_alguna_mano | 0002 | 14,8 % | 19,8 % | 187 | 0,636 | 0,604 | 0,0129 |
| proporcion_sin_alguna_mano | 0003 | 7,1 % | 8,9 % | 163 | 0,541 | 0,632 | 0,1720 |
| proporcion_sin_alguna_mano | 0004 | 3,4 % | 4,1 % | 109 | 0,692 | 0,679 | 0,4957 |
| proporcion_sin_mano_der | 0000 | 6,4 % | 4,9 % | 149 | 0,333 | 0,503 | 0,0759 |
| proporcion_sin_mano_der | 0001 | 2,1 % | 3,1 % | 92 | 0,533 | 0,533 | 0,2091 |
| proporcion_sin_mano_der | 0002 | 2,8 % | 1,5 % | 65 | 0,564 | 0,692 | 0,0519 |
| proporcion_sin_mano_der | 0003 | 2,9 % | 1,3 % | 77 | 0,537 | 0,740 | 0,0205 |
| proporcion_sin_mano_der | 0004 | 1,0 % | 1,3 % | 45 | 0,677 | 0,711 | 0,6622 |
| proporcion_sin_mano_izq | 0000 | 6,0 % | 7,6 % | 171 | 0,414 | 0,468 | 0,1507 |
| proporcion_sin_mano_izq | 0001 | 2,0 % | 0,6 % | 63 | 0,456 | 0,698 | 0,0007 |
| proporcion_sin_mano_izq | 0002 | 12,5 % | 18,5 % | 181 | 0,706 | 0,597 | 0,0027 |
| proporcion_sin_mano_izq | 0003 | 4,7 % | 7,8 % | 139 | 0,639 | 0,604 | 0,0081 |
| proporcion_sin_mano_izq | 0004 | 2,6 % | 3,7 % | 99 | 0,723 | 0,646 | 0,2199 |
| longitud | 0000 | 73,4 | 71,2 | — | — | — | 0,0838 |
| longitud | 0001 | 71,1 | 69,6 | — | — | — | 0,2715 |
| longitud | 0002 | 66,8 | 66,5 | — | — | — | 0,8558 |
| longitud | 0003 | 56,7 | 56,1 | — | — | — | 0,5716 |
| longitud | 0004 | 71,8 | 69,4 | — | — | — | 0,1673 |

**Nivel de muestra, todas las muestras con estratos** (`evidencia_senantes.nivel_muestra.<medida>.estratificada_por_*`). Diferencia dentro de estrato = media ponderada de (media de errores − media de aciertos) de cada estrato, con peso errores·aciertos/n del estrato (igual al coeficiente de una regresión con efectos fijos del estrato); p bilateral por permutación de acierto/error dentro de cada estrato. La diferencia bruta (todas las muestras juntas) se muestra solo como referencia: no es la que se prueba. «Dentro de seña» compara clips de la misma seña; «dentro de seña y señante» compara las repeticiones de la misma seña del mismo señante. p Holm: corrección de Holm sobre las 14 pruebas estratificadas de §3 y §6.

| medida | estratificación | estratos con errores y aciertos | diferencia dentro de estrato (errores − aciertos) | diferencia bruta | p | p Holm |
|---|---|---|---|---|---|---|
| proporcion_sin_alguna_mano | dentro de señante | 5 de 5 | +1,57 pp | +1,92 pp | 0,0108 | 0,1188 |
| proporcion_sin_alguna_mano | dentro de seña | 42 de 50 | +2,60 pp | +1,92 pp | 0,0005 | 0,0065 |
| proporcion_sin_alguna_mano | dentro de seña y señante | 84 de 250 | +2,56 pp | +1,92 pp | 0,0125 | 0,1188 |
| proporcion_sin_mano_der | dentro de señante | 5 de 5 | -0,63 pp | -0,18 pp | 0,0554 | 0,3324 |
| proporcion_sin_mano_der | dentro de seña | 42 de 50 | +1,10 pp | -0,18 pp | 0,0017 | 0,0204 |
| proporcion_sin_mano_der | dentro de seña y señante | 84 de 250 | +0,23 pp | -0,18 pp | 0,6334 | 1,0000 |
| proporcion_sin_mano_izq | dentro de señante | 5 de 5 | +2,06 pp | +1,99 pp | 0,0004 | 0,0056 |
| proporcion_sin_mano_izq | dentro de seña | 42 de 50 | +1,61 pp | +1,99 pp | 0,0280 | 0,2024 |
| proporcion_sin_mano_izq | dentro de seña y señante | 84 de 250 | +2,26 pp | +1,99 pp | 0,0116 | 0,1188 |
| longitud | dentro de señante | 5 de 5 | -1,37 fotogramas | -0,73 fotogramas | 0,0253 | 0,2024 |
| longitud | dentro de seña | 42 de 50 | +0,55 fotogramas | -0,73 fotogramas | 0,5331 | 1,0000 |
| longitud | dentro de seña y señante | 84 de 250 | +1,99 fotogramas | -0,73 fotogramas | 0,0574 | 0,3324 |

**Lectura (señante con menor top-1).** El señante con menor top-1 es 0000 (0,460). Entre las variables medidas de sus clips, 0000 ocupa un extremo de los 5 en: longitud media (fotogramas) (mayor: 72,2; el siguiente 71,0); longitud / media de los otros 4 en la misma seña (mayor: 1,096; el siguiente 1,070); fotogramas sin mano der. (mayor: 5,5 %; el siguiente 2,5 %); fotogramas sin ambas manos (mayor: 0,8 %; el siguiente 0,5 %); no es extremo en: longitud mediana (fotogramas); DE de la longitud; fotogramas sin mano izq.; fotogramas sin alguna mano; desplaz. muñeca izq. por fotograma; desplaz. muñeca der. por fotograma; proporción del recorrido en el bloque der. Con 5 señantes no se puede estimar si alguna de estas variables explica el top-1; que un señante sea extremo en una variable es compatible con que esa variable influya, no lo demuestra. A nivel de muestra, dentro de 0000: proporcion_sin_alguna_mano: aciertos 11,3 %, errores 11,9 % (p = 0,6377); proporcion_sin_mano_der: aciertos 6,4 %, errores 4,9 % (p = 0,0759); longitud: aciertos 73,4, errores 71,2 (p = 0,0838). La seña que más predice el modelo con 0000 es hola (30) (19 de 200 predicciones, 4 reales). Salida del modelo (no es una propiedad de los clips): confianza media 0,392 frente a una exactitud de 0,460 (ECE 0,109); 12 de sus 108 errores tienen confianza ≥ 0,5 y 38 tienen la seña real en el top-3.

## 4. Por grupo de señas

Fuente: `por_grupo`. Recall = aciertos / muestras del grupo; «dentro del grupo» = la predicción cae en alguna seña del grupo (acierto o confusión interna); precisión del grupo = aciertos entre las predicciones que caen en el grupo. DE entre señantes con ddof = 1.

**Grupos de N-30** (interrogativos 41-46 y 26; emociones 20-23; valoración 24, 25, 27):

| grupo | señas | n | recall | dentro del grupo | precisión del grupo | F1 medio de sus señas | recall por señante 0000 / 0001 / 0002 / 0003 / 0004 | DE |
|---|---|---|---|---|---|---|---|---|
| interrogativos | 7 | 140 | 0,379 | 67,9 % | 0,379 | 0,380 | 0,43 / 0,50 / 0,25 / 0,29 / 0,43 | 0,106 |
| emociones | 4 | 80 | 0,688 | 72,5 % | 0,743 | 0,710 | 0,31 / 0,75 / 0,81 / 0,75 / 0,81 | 0,212 |
| valoracion | 3 | 60 | 0,283 | 28,3 % | 0,515 | 0,321 | 0,08 / 0,08 / 0,33 / 0,50 / 0,42 | 0,192 |
| resto | 36 | 717 | 0,632 | 92,2 % | 0,604 | 0,617 | 0,51 / 0,55 / 0,68 / 0,67 / 0,74 | 0,095 |

**Categorías de GRAMMAR_CATEGORIES** (src/grammar_mapper.py):

| grupo | señas | n | recall | dentro del grupo | precisión del grupo | F1 medio de sus señas | recall por señante 0000 / 0001 / 0002 / 0003 / 0004 | DE |
|---|---|---|---|---|---|---|---|---|
| saludo | 5 | 99 | 0,778 | 78,8 % | 0,700 | 0,739 | 0,75 / 0,58 / 0,80 / 0,75 / 1,00 | 0,151 |
| cortesia | 4 | 80 | 0,688 | 82,5 % | 0,591 | 0,640 | 0,44 / 0,25 / 0,88 / 0,94 / 0,94 | 0,322 |
| pregunta | 7 | 140 | 0,379 | 67,9 % | 0,379 | 0,380 | 0,43 / 0,50 / 0,25 / 0,29 / 0,43 | 0,106 |
| sujeto | 13 | 259 | 0,571 | 77,2 % | 0,532 | 0,553 | 0,46 / 0,58 / 0,69 / 0,50 / 0,63 | 0,093 |
| verbo | 4 | 79 | 0,405 | 40,5 % | 0,485 | 0,450 | 0,44 / 0,25 / 0,33 / 0,38 / 0,62 | 0,141 |
| objeto | 4 | 80 | 0,750 | 75,0 % | 0,645 | 0,707 | 0,62 / 0,75 / 0,94 / 0,69 / 0,75 | 0,117 |
| adjetivo | 7 | 140 | 0,779 | 80,0 % | 0,807 | 0,786 | 0,57 / 0,82 / 0,75 / 0,86 / 0,89 | 0,127 |
| adverbio | 6 | 120 | 0,367 | 49,2 % | 0,537 | 0,408 | 0,04 / 0,33 / 0,29 / 0,75 / 0,42 | 0,256 |

## 5. Calibración y umbral

Fuente: `calibracion`. Confianza = `prob_predicha` (máximo del softmax). 10 intervalos iguales; ECE = Σ (n_b/N)·|exactitud_b − confianza_b|; MCE = máximo de |exactitud_b − confianza_b|. Figura: `fiabilidad.png`.

| intervalo | n | confianza media | exactitud | confianza − exactitud |
|---|---|---|---|---|
| [0,0; 0,1) | 36 | 0,075 | 0,167 | -0,091 |
| [0,1; 0,2) | 124 | 0,151 | 0,258 | -0,107 |
| [0,2; 0,3) | 121 | 0,252 | 0,388 | -0,136 |
| [0,3; 0,4) | 86 | 0,346 | 0,465 | -0,119 |
| [0,4; 0,5) | 91 | 0,451 | 0,473 | -0,022 |
| [0,5; 0,6) | 70 | 0,549 | 0,486 | 0,063 |
| [0,6; 0,7) | 92 | 0,653 | 0,717 | -0,064 |
| [0,7; 0,8) | 97 | 0,760 | 0,763 | -0,003 |
| [0,8; 0,9) | 179 | 0,854 | 0,804 | 0,050 |
| [0,9; 1,0) | 101 | 0,924 | 0,911 | 0,013 |

Agregado: confianza media 0,543, exactitud 0,580, ECE 0,066, MCE 0,136.

**Umbral D-09 (t = 0,20, `models/canonico_holistic_mascara_manos/modelo_canonico.umbral.json`) y referencia 0,5, por señante** (`calibracion.umbral`, `calibracion.referencia_0_5`; aceptada si confianza ≥ t):

| señante | ECE | confianza media | exactitud | rechazo t | error entre aceptadas t | aciertos perdidos t | rechazo 0,5 | error entre aceptadas 0,5 | aciertos perdidos 0,5 |
|---|---|---|---|---|---|---|---|---|---|
| 0000 | 0,109 | 0,392 | 0,460 | 29,0 % | 43,7 % (62/142) | 13,0 % | 72,0 % | 21,4 % (12/56) | 52,2 % |
| 0001 | 0,114 | 0,575 | 0,533 | 14,6 % | 40,6 % (69/170) | 4,7 % | 41,7 % | 33,6 % (39/116) | 27,4 % |
| 0002 | 0,131 | 0,511 | 0,606 | 15,2 % | 31,0 % (52/168) | 3,3 % | 48,5 % | 20,6 % (21/102) | 32,5 % |
| 0003 | 0,081 | 0,593 | 0,615 | 10,5 % | 34,6 % (62/179) | 4,9 % | 40,0 % | 22,5 % (27/120) | 24,4 % |
| 0004 | 0,105 | 0,642 | 0,685 | 11,0 % | 29,2 % (52/178) | 8,0 % | 27,5 % | 20,7 % (30/145) | 16,1 % |
| agregado | 0,066 | 0,543 | 0,580 | 16,0 % | 35,5 % (297/837) | 6,6 % | 45,9 % | 23,9 % (129/539) | 29,1 % |

**Lectura.** En la prueba con señantes nuevos la confianza media (0,543) queda por debajo de la exactitud (0,580); ECE agregado 0,066. En 7 de 10 intervalos con datos la confianza media es menor que la exactitud (intervalos que empiezan en 0,0, 0,1, 0,2, 0,3, 0,4, 0,6, 0,7). Con t = 0,20 el rechazo va de 10,5 % (0003) a 29,0 % (0000) y el error entre aceptadas de 29,2 % (0004) a 43,7 % (0000). Esto describe el umbral ya elegido (D-09) sobre la prueba; no es un criterio para cambiarlo (N-33).

## 6. Error y longitud del clip

Fuente: `longitud`. longitud = fotogramas del clip antes de remuestrear a 30 (lengths.npy). Longitud media de aciertos 67,7 y de errores 66,9 fotogramas; correlación de Pearson longitud–acierto 0,033.

Cortes por cuantiles (np.quantile); cada intervalo incluye su borde superior y el primero también el inferior; con valores repetidos en un borde los grupos no quedan iguales.

**Longitud absoluta** (mezcla el efecto del señante: cada señante tiene su ritmo):

| quintil | fotogramas desde | hasta | n | exactitud |
|---|---|---|---|---|
| 1 | 40,0 | 57,0 | 206 | 0,607 |
| 2 | 57,0 | 64,0 | 211 | 0,531 |
| 3 | 64,0 | 70,0 | 205 | 0,512 |
| 4 | 70,0 | 76,8 | 175 | 0,629 |
| 5 | 76,8 | 135,0 | 200 | 0,630 |

**Longitud relativa a la mediana del propio señante:**

| quintil | cociente desde | hasta | n | exactitud |
|---|---|---|---|---|
| 1 | 0,60 | 0,89 | 202 | 0,550 |
| 2 | 0,89 | 0,97 | 202 | 0,475 |
| 3 | 0,97 | 1,03 | 199 | 0,628 |
| 4 | 1,03 | 1,11 | 202 | 0,624 |
| 5 | 1,11 | 1,90 | 192 | 0,625 |

**Desvío |log(longitud / media de la misma seña en los otros 4 señantes)|** (cuánto se aparta el ritmo del clip del de los señantes con que se entrenó ese pliegue):

| quintil | desvío desde | hasta | n | exactitud |
|---|---|---|---|---|
| 1 | 0,00 | 0,05 | 200 | 0,535 |
| 2 | 0,05 | 0,10 | 199 | 0,588 |
| 3 | 0,10 | 0,16 | 199 | 0,558 |
| 4 | 0,16 | 0,23 | 199 | 0,623 |
| 5 | 0,23 | 0,80 | 200 | 0,595 |

**Pruebas dentro de estrato** (`longitud.pruebas_longitud`, `longitud.pruebas_desvio_vs_misma_sena`; mismo estadístico que en §3: diferencia errores − aciertos dentro de cada estrato, combinada con peso errores·aciertos/n, y permutación dentro del estrato; diferencia bruta sin estratos: -0,73 fotogramas; p Holm sobre las 14 pruebas estratificadas):

| medida | estratificación | estratos con errores y aciertos | diferencia dentro de estrato | p | p Holm | permutaciones (semilla) |
|---|---|---|---|---|---|---|
| longitud | dentro de señante | 5 de 5 | -1,37 fotogramas | 0,0253 | 0,2024 | 10000 (2026) |
| longitud | dentro de seña | 42 de 50 | +0,55 fotogramas | 0,5331 | 1,0000 | 10000 (2027) |
| longitud | dentro de seña y señante | 84 de 250 | +1,99 fotogramas | 0,0574 | 0,3324 | 10000 (2028) |
| desvío |log| | dentro de señante | 5 de 5 | -0,001 | 0,9385 | 1,0000 | 10000 (2066) |
| desvío |log| | dentro de seña | 42 de 50 | -0,015 | 0,0683 | 0,3324 | 10000 (2067) |

Entre señas (`longitud.entre_senas`, 50 señas): correlación de Pearson entre la longitud media de la seña y su recall agregado 0,293, p = 0,0405 (permutación del recall entre señas).

| señante | longitud media aciertos | longitud media errores | errores − aciertos | exactitud por tercil de longitud (corto / medio / largo) |
|---|---|---|---|---|
| 0000 | 73,4 | 71,2 | -2,22 fotogramas | 0,29 / 0,60 / 0,52 |
| 0001 | 71,1 | 69,6 | -1,49 fotogramas | 0,44 / 0,61 / 0,54 |
| 0002 | 66,8 | 66,5 | -0,27 fotogramas | 0,58 / 0,63 / 0,61 |
| 0003 | 56,7 | 56,1 | -0,57 fotogramas | 0,54 / 0,70 / 0,61 |
| 0004 | 71,8 | 69,4 | -2,36 fotogramas | 0,66 / 0,65 / 0,75 |

**Lectura.** Errores − aciertos en longitud del clip: dentro de señante -1,37 fotogramas (p = 0,0253; Holm 0,2024: p < 0,05 sin corregir, pero no tras Holm); dentro de seña +0,55 fotogramas (p = 0,5331; Holm 1,0000: no distinguible del azar con α = 0,05); dentro de seña y señante +1,99 fotogramas (p = 0,0574; Holm 0,3324: no distinguible del azar con α = 0,05). La diferencia bruta, sin estratos, es -0,73 fotogramas. Por señante, la diferencia es negativa (errores más cortos) en 5 de 5: 0000 -2,22 fotogramas, 0001 -1,49 fotogramas, 0002 -0,27 fotogramas, 0003 -0,57 fotogramas, 0004 -2,36 fotogramas. Entre las 50 señas, la correlación entre la longitud media de la seña y su recall es 0,293 (p = 0,0405). Desvío frente al ritmo de la misma seña en los otros señantes: dentro de señante -0,001 (p = 0,9385; Holm 1,0000: no distinguible del azar con α = 0,05); dentro de seña -0,015 (p = 0,0683; Holm 0,3324: no distinguible del azar con α = 0,05). La diferencia dentro de señante no aparece al comparar clips de la misma seña, y el sentido de la correlación entre señas coincide con ella: es compatible con que se fallen más las señas de clips más cortos (qué seña es), no muestra que un clip más corto que otros de su misma seña falle más. Respuesta a «¿fallan más los clips cortos o los largos?»: con estos datos no hay una respuesta única; el signo cambia según con qué clips se compare y ninguna diferencia de longitud queda con p Holm < 0,05. Son asociaciones, no causas.

## 7. Contraste por seña con otras representaciones

Fuente: `contraste`. Mismas 997 muestras de prueba y mismas particiones (comprobado muestra a muestra). Diferencia = recall «manos» − recall de la otra representación (20 muestras por seña, salvo 19 en las señas 2, 8, 16: una muestra mueve el recall 0,050 o 0,053). McNemar exacto sobre las muestras de la seña; 50 pruebas sin corrección por comparaciones múltiples, orientativo.

### manos frente a manos_postura (`models/loso_2026-09-14_holistic_mascara`)

Top-1 agregado: manos 0,580, manos_postura 0,534. Muestras que solo acierta «manos»: 134; solo manos_postura: 88; McNemar global p = 0,0024.

| seña | categoría | recall manos | recall manos_postura | diferencia | solo manos / solo otro | p McNemar |
|---|---|---|---|---|---|---|
| vivir (8) | verbo | 0,68 | 0,37 | +0,32 | 8 / 2 | 0,109 |
| ustedes (40) | sujeto | 0,50 | 0,20 | +0,30 | 6 / 0 | 0,031 |
| personas (11) | sujeto | 0,55 | 0,30 | +0,25 | 5 / 0 | 0,062 |
| ¿dónde? (43) | pregunta | 0,55 | 0,30 | +0,25 | 8 / 3 | 0,227 |
| feliz (20) | adjetivo | 0,90 | 0,65 | +0,25 | 5 / 0 | 0,062 |
| por favor (32) | cortesia | 0,85 | 0,60 | +0,25 | 5 / 0 | 0,062 |
| mal (25) | adverbio | 0,55 | 0,35 | +0,20 | 6 / 2 | 0,289 |
| ¿cómo? (44) | pregunta | 0,40 | 0,20 | +0,20 | 5 / 1 | 0,219 |
| mucho (49) | adverbio | 0,50 | 0,30 | +0,20 | 6 / 2 | 0,289 |
| diferente (47) | adjetivo | 1,00 | 0,80 | +0,20 | 4 / 0 | 0,125 |
| permiso (36) | objeto | 0,15 | 0,55 | -0,40 | 0 / 8 | 0,008 |
| comer (7) | verbo | 0,05 | 0,40 | -0,35 | 0 / 7 | 0,016 |
| niño (14) | sujeto | 0,15 | 0,45 | -0,30 | 1 / 7 | 0,070 |
| más o menos (27) | adverbio | 0,00 | 0,20 | -0,20 | 0 / 4 | 0,125 |
| bien (24) | adverbio | 0,30 | 0,50 | -0,20 | 1 / 5 | 0,219 |
| adiós (31) | saludo | 0,40 | 0,60 | -0,20 | 1 / 5 | 0,219 |
| abuelo (16) | sujeto | 0,37 | 0,53 | -0,16 | 2 / 5 | 0,453 |
| mujer (12) | sujeto | 0,25 | 0,40 | -0,15 | 1 / 4 | 0,375 |
| trabajar (6) | verbo | 0,60 | 0,75 | -0,15 | 2 / 5 | 0,453 |
| ¿cómo estás? (26) | pregunta | 0,15 | 0,30 | -0,15 | 0 / 3 | 0,250 |

(Arriba: hasta 10 señas que más ganan con «manos» y hasta 10 que más pierden.) Ganan con «manos» 31 señas, pierden 15, sin cambio 4. Señas con p < 0,05: comer (7), permiso (36), ustedes (40).
Recall por grupo de N-30 (manos / manos_postura): interrogativos 0,379 / 0,264; emociones 0,688 / 0,562; valoracion 0,283 / 0,350; resto 0,632 / 0,598.

### manos frente a manos_postura_rostro (`models/loso_2026-09-15_holistic_mascara_rostro`)

Top-1 agregado: manos 0,580, manos_postura_rostro 0,509. Muestras que solo acierta «manos»: 150; solo manos_postura_rostro: 79; McNemar global p < 0,0001.

| seña | categoría | recall manos | recall manos_postura_rostro | diferencia | solo manos / solo otro | p McNemar |
|---|---|---|---|---|---|---|
| vivir (8) | verbo | 0,68 | 0,16 | +0,53 | 13 / 3 | 0,021 |
| personas (11) | sujeto | 0,55 | 0,25 | +0,30 | 6 / 0 | 0,031 |
| mal (25) | adverbio | 0,55 | 0,25 | +0,30 | 8 / 2 | 0,109 |
| hombre (13) | sujeto | 0,90 | 0,65 | +0,25 | 5 / 0 | 0,062 |
| tío (17) | sujeto | 0,85 | 0,60 | +0,25 | 6 / 1 | 0,125 |
| feliz (20) | adjetivo | 0,90 | 0,65 | +0,25 | 5 / 0 | 0,062 |
| adiós (31) | saludo | 0,40 | 0,15 | +0,25 | 6 / 1 | 0,125 |
| ¿dónde? (43) | pregunta | 0,55 | 0,35 | +0,20 | 6 / 2 | 0,289 |
| triste (22) | adjetivo | 0,65 | 0,45 | +0,20 | 5 / 1 | 0,219 |
| mucho (49) | adverbio | 0,50 | 0,30 | +0,20 | 4 / 0 | 0,125 |
| ¿cómo estás? (26) | pregunta | 0,15 | 0,40 | -0,25 | 2 / 7 | 0,180 |
| contento (21) | adjetivo | 0,65 | 0,80 | -0,15 | 1 / 4 | 0,375 |
| comer (7) | verbo | 0,05 | 0,20 | -0,15 | 0 / 3 | 0,250 |
| permiso (36) | objeto | 0,15 | 0,25 | -0,10 | 2 / 4 | 0,688 |
| niño (14) | sujeto | 0,15 | 0,25 | -0,10 | 1 / 3 | 0,625 |
| nunca (37) | adverbio | 0,50 | 0,60 | -0,10 | 0 / 2 | 0,500 |
| hola (30) | saludo | 0,50 | 0,60 | -0,10 | 4 / 6 | 0,754 |
| ¿por qué? (45) | pregunta | 0,50 | 0,55 | -0,05 | 4 / 5 | 1,000 |
| ustedes (40) | sujeto | 0,50 | 0,55 | -0,05 | 1 / 2 | 1,000 |
| niña (15) | sujeto | 0,35 | 0,40 | -0,05 | 1 / 2 | 1,000 |

(Arriba: hasta 10 señas que más ganan con «manos» y hasta 10 que más pierden.) Ganan con «manos» 29 señas, pierden 12, sin cambio 9. Señas con p < 0,05: vivir (8), personas (11).
Recall por grupo de N-30 (manos / manos_postura_rostro): interrogativos 0,379 / 0,357; emociones 0,688 / 0,588; valoracion 0,283 / 0,167; resto 0,632 / 0,558.

## Lectura prudente

- Top-1 agregado 0,580 (media de pliegues 0,580 ± 0,086); señas con recall agregado 0: 1; con recall ≥ 0,9: 12.
- La pareja más confundida es niño (14) / niña (15) con 11 errores (5 y 6 en cada sentido).
- 29,6 % de los errores quedan dentro de la misma categoría gramatical, frente a 13,6 % esperado al azar: las categorías de grammar_mapper son sintácticas, no de forma manual, así que esta cifra no mide parecido articulatorio.
- Proporción de fotogramas sin el bloque de mano izquierda, errores − aciertos: dentro de señante +2,06 pp (p = 0,0004; Holm 0,0056: p < 0,05 también tras Holm); dentro de seña +1,61 pp (p = 0,0280; Holm 0,2024: p < 0,05 sin corregir, pero no tras Holm); dentro de seña y señante +2,26 pp (p = 0,0116; Holm 0,1188: p < 0,05 sin corregir, pero no tras Holm). El signo es el mismo en las tres estratificaciones; comparaciones con p Holm < 0,05: 1 de 3.
- Proporción de fotogramas sin el bloque de mano derecha, errores − aciertos: dentro de señante -0,63 pp (p = 0,0554; Holm 0,3324: no distinguible del azar con α = 0,05); dentro de seña +1,10 pp (p = 0,0017; Holm 0,0204: p < 0,05 también tras Holm); dentro de seña y señante +0,23 pp (p = 0,6334; Holm 1,0000: no distinguible del azar con α = 0,05). El signo cambia según la estratificación; comparaciones con p Holm < 0,05: 1 de 3.
- Proporción de fotogramas sin alguna mano, errores − aciertos: dentro de señante +1,57 pp (p = 0,0108; Holm 0,1188: p < 0,05 sin corregir, pero no tras Holm); dentro de seña +2,60 pp (p = 0,0005; Holm 0,0065: p < 0,05 también tras Holm); dentro de seña y señante +2,56 pp (p = 0,0125; Holm 0,1188: p < 0,05 sin corregir, pero no tras Holm). El signo es el mismo en las tres estratificaciones; comparaciones con p Holm < 0,05: 1 de 3.
- Las diferencias de fotogramas sin mano son asociaciones a nivel de muestra: no dicen que la ausencia de mano cause el error ni explican por sí solas las diferencias entre señantes.
- Cada seña tiene 4 muestras por señante (3 en 3 casos) y 20 en el agregado (19 en la seña 2, 19 en la seña 8, 19 en la seña 16): las diferencias por seña de una o dos muestras no deben leerse como efectos.
- No se toma ninguna decisión a partir de este análisis; las hipótesis que sugiera requieren una decisión nueva fijada antes de medir (como D-07, D-08 y D-09).

## Figuras

- `matriz_confusion_normalizada.png`
- `recall_por_sena.png`
- `por_senante.png`
- `fiabilidad.png`
