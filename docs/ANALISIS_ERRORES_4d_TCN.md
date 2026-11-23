# Análisis de errores del modelo final: TCN + imputación + 252 rasgos (actividad 4.d)

Generado por `src/analisis_errores_tcn.py` el 2026-11-08T21:26:15. SOLO REPORTE: no decide, no reentrena y no ejecuta ningún modelo; lee predicciones ya guardadas. Todas las cifras están en `docs/ANALISIS_ERRORES_4d_TCN.json`; cada sección indica la clave de origen. Decimales con coma. Figuras en `docs/figuras_2026-11-08/` (prefijo `4d_`).

Modelo final (D-14): TCN de 4 bloques residuales Conv1D (dilatación 1-2-4-8, 64 filtros, 156.402 parámetros), regla de mano ausente «imputacion», 252 rasgos (126 posiciones + 126 velocidades), 30 pasos, umbral 0,00. «Heredada» = la red canónica anterior (Conv1D + BiLSTM + atención, 714.291 parámetros, regla «mascara»), la misma que analizó el informe previo `models/analisis_errores_manos_2026-09-15/analisis.md`.

## 0. Método, fuentes y controles

**Fuentes** (`fuentes`):

- TCN final, LOSO: `models/confirmacion_tcn_2026-10-03/T1_imputacion_252/semilla_S/pliegue_k/predicciones.json` (3 semillas 2026-2028 × 5 pliegues; el pliegue k deja fuera al señante k; 997 videos por semilla).
- Heredada, LOSO: `models/experimento_mano_ausente_2026-10-02/A_mascara/…` (el brazo R0 que reutilizan los experimentos de arquitecturas y de confirmación; mismas semillas y pliegues).
- Partición aleatoria 797/100/100 (señantes vistos): `models/tcn_imputacion/evaluacion_modelo_seleccionado` (semilla 2030) y `models/canonico_holistic_mascara_manos/evaluacion_modelo_seleccionado` (semilla 2026); las mismas 100 muestras de prueba.
- Propiedades de cada video: `data/processed_holistic_mascara_rostro` (lengths.npy; X_padded.npy con la mano ausente en ceros, es decir, antes de imputar) y `data/extracted/LSC50_holistic/METRICAS_FOTOGRAMA` (solo lectura).
- TCN con «mascara» (T0, para aislar la imputación, §6.2): `models/arquitecturas_literatura_2026-10-02/T_tcn`.
- No se usaron los `probabilidades.npz` (ignorados por git): bastan `prob_predicha` y `top3` de los predicciones.json. No se regeneró nada.

**Unidades.** «3 semillas» = las 2.991 predicciones (997 videos × 3 modelos entrenados con semillas distintas) juntas: es la exactitud media de los tres modelos y da conteos más estables que una sola semilla. «Voto» = una predicción por video, la clase que predicen al menos 2 de las 3 semillas (si las tres discrepan, la de mayor probabilidad; la regla de `experimento_confirmacion_tcn.voto_semillas`). El voto es un recurso de análisis (un conjunto de 3 modelos), no el modelo desplegado: su exactitud es mayor que la de cada semilla.

**Controles** (`controles`; el script se detiene si alguno falla):

| modelo | top-1 (media de 15 corridas) | top-3 | F1 macro | señante 0000 | coincide con el registro (3 decimales) | aleatoria: top-1 / errores |
|---|---|---|---|---|---|---|
| heredada | 0,589 | 0,764 | 0,544 | 0,452 | sí | 0,88 / 12 (coincide con metricas.json) |
| TCN final | 0,677 | 0,847 | 0,647 | 0,488 | sí | 0,94 / 6 (coincide con metricas.json) |

- Las 30 corridas (2 modelos × 3 semillas × 5 pliegues) reproducen top-1, top-3 y F1 de `resultados_confirmacion_tcn.json › corridas` (tolerancia 10⁻⁶).
- McNemar con el voto, TCN frente a heredada: solo acierta la TCN 163, solo la heredada 56, p = 2,6·10^-13 (guardado: 163 / 56; coincide).
- La heredada con semilla 2026 da exactamente las mismas 997 de 997 predicciones que el LOSO del informe previo (`models/loso_2026-09-15_holistic_mascara_manos`): el informe previo es una de las tres semillas de la heredada que se usa aquí.
- Fotogramas sin mano: METRICAS_FOTOGRAMA y los bloques en ceros de X_padded coinciden video a video (discrepancias: 0).

## 1. Cifras globales

Fuente: `controles.resumen` (media de pliegues) y `global` (3 semillas juntas y voto).

| modelo | top-1 media de pliegues | top-3 | F1 macro | señante 0000 | errores (3 semillas, de 2.991) | top-1 voto | errores voto (de 997) | aleatoria |
|---|---|---|---|---|---|---|---|---|
| heredada | 0,589 | 0,764 | 0,544 | 0,452 | 1229 | 0,609 | 390 | 0,88 (12 errores) |
| TCN final | 0,677 | 0,847 | 0,647 | 0,488 | 966 | 0,716 | 283 | 0,94 (6 errores) |

## 2. Pares más confundidos

Fuente: `pares`. Pareja = los dos sentidos de una confusión {a, b} juntos; par = un sentido real → predicha.

**TCN final, parejas con más errores** (3 semillas; `pares.tcn.parejas_3_semillas`). 966 errores en 292 pares dirigidos distintos; las 10 primeras parejas reúnen 22,3 % de los errores (heredada: 20,8 %).

| pareja {a, b} | a → b | b → a | total 3 semillas | por semilla (media) |
|---|---|---|---|---|
| ¿cómo estás? (26) / ¿cómo? (44) | 19 | 14 | 33 | 11,0 |
| niño (14) / niña (15) | 16 | 15 | 31 | 10,3 |
| ¿qué? (41) / ¿dónde? (43) | 16 | 10 | 26 | 8,7 |
| ¿qué? (41) / ¿cómo? (44) | 11 | 13 | 24 | 8,0 |
| mujer (12) / ¿cuándo? (42) | 4 | 16 | 20 | 6,7 |
| gracias (0) / permiso (36) | 12 | 5 | 17 | 5,7 |
| hola (30) / ¿por qué? (45) | 0 | 17 | 17 | 5,7 |
| comer (7) / mujer (12) | 10 | 6 | 16 | 5,3 |
| feliz (20) / contento (21) | 6 | 10 | 16 | 5,3 |
| poco (9) / más o menos (27) | 3 | 12 | 15 | 5,0 |
| niño (14) / aburrido (23) | 2 | 13 | 15 | 5,0 |
| tío (17) / hermano (18) | 2 | 13 | 15 | 5,0 |

**TCN final, pares dirigidos con el voto** (una predicción por video; `pares.tcn.pares_voto`):

| real → predicha | conteo (de 4 videos por señante) | inverso | señantes en que ocurre |
|---|---|---|---|
| ¿cómo? (44) → ¿qué? (41) | 7 | 4 | 0000, 0001, 0004 |
| aburrido (23) → niño (14) | 6 | 0 | 0000, 0001 |
| ¿cómo estás? (26) → ¿cómo? (44) | 6 | 4 | 0002, 0003, 0004 |
| ¿qué? (41) → ¿dónde? (43) | 6 | 4 | 0001, 0004 |
| ¿cuándo? (42) → mujer (12) | 6 | 1 | 0003, 0004 |
| ¿por qué? (45) → hola (30) | 6 | 0 | 0002, 0004 |
| niño (14) → niña (15) | 5 | 5 | 0000, 0002 |
| niña (15) → niño (14) | 5 | 5 | 0003, 0004 |
| hermano (18) → tío (17) | 5 | 1 | 0001, 0004 |
| sentir (28) → permiso (36) | 5 | 2 | 0000, 0001, 0003 |
| comer (7) → mujer (12) | 4 | 2 | 0000, 0002, 0004 |
| abuelo (16) → mujer (12) | 4 | 0 | 0000, 0003 |

**Heredada, parejas con más errores** (3 semillas; `pares.heredada.parejas_3_semillas`), para comparar:

| pareja {a, b} | a → b | b → a | total 3 semillas | TCN, misma pareja |
|---|---|---|---|---|
| niño (14) / niña (15) | 18 | 20 | 38 | 31 |
| ¿cómo estás? (26) / ¿cómo? (44) | 18 | 19 | 37 | 33 |
| permiso (36) / ¿cuándo? (42) | 11 | 14 | 25 | 5 |
| poco (9) / más o menos (27) | 5 | 19 | 24 | 15 |
| ¿qué? (41) / ¿dónde? (43) | 17 | 7 | 24 | 26 |
| hombre (13) / hola (30) | 4 | 19 | 23 | 3 |
| tío (17) / hermano (18) | 7 | 15 | 22 | 15 |
| mujer (12) / permiso (36) | 10 | 11 | 21 | 4 |
| abuelo (16) / hola (30) | 9 | 12 | 21 | 14 |
| ¿qué? (41) / ¿cómo? (44) | 12 | 9 | 21 | 24 |
| niño (14) / tú (39) | 5 | 14 | 19 | 2 |
| más o menos (27) / sentir (28) | 11 | 7 | 18 | 10 |

De las 10 parejas principales de cada red, 5 son comunes: ¿cómo estás? / ¿cómo?; niño / niña; ¿qué? / ¿dónde?; ¿qué? / ¿cómo?; poco / más o menos. Figura: `4d_parejas_confundidas_heredada_vs_tcn.png`; matriz completa: `4d_matriz_confusion_loso_tcn.png`.

## 3. Clases con peor recall y peor precisión

Fuente: `por_clase.tcn` (3 semillas: 60 predicciones por seña, 57 en las señas 2, 8 y 16). Precisión = aciertos / veces que se predijo la seña.

**Peores 10 por recall (TCN):**

| seña | categoría (conversor) | recall | recall por semilla | precisión | veces predicha | recall heredada |
|---|---|---|---|---|---|---|
| ¿qué? (41) | INTERR | 0,28 | 0,20 / 0,45 / 0,20 | 0,18 | 92 | 0,37 |
| ¿cuándo? (42) | INTERR | 0,33 | 0,40 / 0,30 / 0,30 | 0,50 | 40 | 0,25 |
| comer (7) | VERBO | 0,35 | 0,30 / 0,35 / 0,40 | 0,58 | 36 | 0,02 |
| permiso (36) | CORT | 0,42 | 0,50 / 0,25 / 0,50 | 0,42 | 59 | 0,20 |
| ¿cómo estás? (26) | PREG_FIJA | 0,43 | 0,35 / 0,45 / 0,50 | 0,41 | 64 | 0,30 |
| ¿cómo? (44) | INTERR | 0,45 | 0,60 / 0,50 / 0,25 | 0,42 | 65 | 0,33 |
| más o menos (27) | ADV_EST | 0,47 | 0,50 / 0,40 / 0,50 | 0,45 | 62 | 0,12 |
| bien (24) | ADV_EST | 0,47 | 0,40 / 0,45 / 0,55 | 0,90 | 31 | 0,42 |
| ¿por qué? (45) | INTERR | 0,47 | 0,40 / 0,40 / 0,60 | 0,93 | 30 | 0,53 |
| mujer (12) | NOM_PERS | 0,48 | 0,60 / 0,40 / 0,45 | 0,39 | 75 | 0,25 |

**Peores 8 por precisión (TCN)** (señas que «atraen» predicciones de otras):

| seña | precisión | veces predicha (de 2.991) | recall |
|---|---|---|---|
| ¿qué? (41) | 0,18 | 92 | 0,28 |
| niña (15) | 0,35 | 93 | 0,55 |
| abuelo (16) | 0,37 | 90 | 0,58 |
| mujer (12) | 0,39 | 75 | 0,48 |
| niño (14) | 0,40 | 87 | 0,58 |
| ¿cómo estás? (26) | 0,41 | 64 | 0,43 |
| ¿cómo? (44) | 0,42 | 65 | 0,45 |
| permiso (36) | 0,42 | 59 | 0,42 |

Recall 0 con el voto: TCN ninguna; heredada comer (7). Frente a la heredada (3 semillas), la TCN sube el recall en 33 señas y lo baja en 11. Mayores subidas: más o menos +0,35, comer +0,33, ¿quién? +0,32, hola +0,28, adiós +0,27, con gusto +0,25. Bajadas: ¿qué? -0,08, seña -0,07, ¿por qué? -0,07, juicioso -0,05, familia -0,03, feliz -0,03. Figura: `4d_recall_por_clase_heredada_vs_tcn.png`.

## 4. Por señante

Fuente: `por_senante` (media de las 3 semillas del pliegue que deja fuera a ese señante; «voto»: una predicción por video).

| señante | top-1 heredada | top-1 TCN | top-1 TCN por semilla | top-3 TCN | F1 TCN | top-1 voto TCN | señas con recall 0 (voto TCN) |
|---|---|---|---|---|---|---|---|
| 0000 | 0,452 | 0,488 | 0,540 / 0,390 / 0,535 | 0,660 | 0,469 | 0,580 | 10 |
| 0001 | 0,536 | 0,611 | 0,613 / 0,633 / 0,588 | 0,807 | 0,565 | 0,643 | 7 |
| 0002 | 0,631 | 0,721 | 0,717 / 0,697 / 0,747 | 0,889 | 0,685 | 0,747 | 5 |
| 0003 | 0,640 | 0,770 | 0,740 / 0,840 / 0,730 | 0,935 | 0,745 | 0,795 | 5 |
| 0004 | 0,687 | 0,795 | 0,830 / 0,730 / 0,825 | 0,942 | 0,770 | 0,815 | 3 |

**Señante 0000.** Sigue siendo el peor con las dos redes (TCN 0,488, heredada 0,452). Con el voto, la TCN deja 10 señas con recall 0 (comer (7), mujer (12), abuelo (16), sentir (28), nunca (37), yo (38), tú (39), ¿cuándo? (42), ¿cómo? (44), ¿por qué? (45)) frente a 17 de la heredada. Pares más frecuentes de 0000 con la TCN (3 semillas): tú (39) → niña (15) ×8; ¿cómo? (44) → ¿qué? (41) ×8; ¿dónde? (43) → ¿qué? (41) ×7; más o menos (27) → ¿qué? (41) ×6; poco (9) → niña (15) ×5. Confianza media de la TCN en 0000: aciertos 0,703, errores 0,455. Es también el señante en que más discrepan las semillas (§7): su top-1 por semilla va de 0,390 a 0,540 y el voto sube a 0,580.

Pares principales de los otros señantes (TCN, 3 semillas): **0001**: hola → abuelo ×11; adiós → abuelo ×11; más o menos → poco ×10 · **0002**: niño → niña ×8; ¿qué? → ¿cómo? ×8; ¿cuándo? → ¿cómo estás? ×8 · **0003**: niña → niño ×12; ¿cómo estás? → ¿cómo? ×12; tú → poco ×9 · **0004**: ¿por qué? → hola ×11; contento → feliz ×10; ¿cuándo? → mujer ×10.

## 5. Qué errores de la heredada corrige la TCN y cuáles introduce

Fuente: `transicion` (voto de 3 semillas de cada red; un video = una fila). «Neto» = solo TCN − solo heredada; p = McNemar exacto bilateral.

| grupo | ambos aciertan | solo TCN | solo heredada | ambos fallan | neto | p |
|---|---|---|---|---|---|---|
| todos (997) | 551 | 163 | 56 | 227 | +107 | 2,6·10^-13 |
| señante 0000 | 75 | 41 | 19 | 65 | +22 | 0,0062 |
| señante 0001 | 96 | 32 | 11 | 60 | +21 | 0,0019 |
| señante 0002 | 119 | 29 | 13 | 37 | +16 | 0,0195 |
| señante 0003 | 126 | 33 | 5 | 36 | +28 | 4,3·10^-6 |
| señante 0004 | 135 | 28 | 8 | 29 | +20 | 0,0012 |
| categoría saludo | 76 | 13 | 0 | 10 | +13 | 0,0002 |
| categoría cortesia | 52 | 16 | 5 | 7 | +11 | 0,0266 |
| categoría pregunta | 40 | 27 | 20 | 53 | +7 | 0,3817 |
| categoría sujeto | 145 | 40 | 12 | 62 | +28 | 0,0001 |
| categoría verbo | 32 | 16 | 4 | 27 | +12 | 0,0118 |
| categoría objeto | 58 | 9 | 4 | 9 | +5 | 0,2668 |
| categoría adjetivo | 104 | 14 | 7 | 15 | +7 | 0,1892 |
| categoría adverbio | 44 | 28 | 4 | 44 | +24 | 1,9·10^-5 |

- De los 390 videos que la heredada falla, la TCN corrige 163 (41,8 % de los errores de la heredada) e introduce 56 errores nuevos (9,2 % de los aciertos de la heredada). En 227 videos fallan las dos; en 92 de ellos (40,5 %) predicen la misma seña equivocada.
- El neto es positivo en 5 de 5 señantes y en 8 de 8 categorías. Por seña: ganan 32, pierden 8, igual 10 (4 videos por señante y seña: diferencias de 1-2 videos son ruido).
- Señas que más ganan (solo TCN / solo heredada): más o menos 9/0; ¿quién? 9/0; niño 9/1; comer 7/0; hola 7/0; mujer 6/0; adiós 6/0; bienvenido 6/0.
- Señas con neto negativo: ¿qué? 2/6; seña 0/1; vivir 2/3; familia 0/1; hermano 0/1; juicioso 0/1; ¿dónde? 2/3; ¿por qué? 5/6.
- Errores de la heredada que la TCN corrige, por par: hola → hombre ×7; niño → abuelo ×4; niño → niña ×4; más o menos → poco ×4; con gusto → por favor ×4; permiso → ¿cuándo? ×4; gracias → perdón ×3; comer → permiso ×3.
- Errores que introduce la TCN, por par: ¿qué? → ¿dónde? ×4; ¿por qué? → ¿cuándo? ×3; gracias → ¿cuándo? ×2; vivir → ¿cómo estás? ×2; personas → más o menos ×2; contento → bienvenido ×2; con gusto → permiso ×2; permiso → sentir ×2.

## 6. Error y propiedades medibles del video

Fuente: `propiedades`. Definiciones (`propiedades.definiciones`):

- **longitud**: fotogramas del video antes de remuestrear a 30 (lengths.npy).
- **prop_sin_alguna_mano**: fotogramas en que falta alguna de las dos manos / fotogramas (METRICAS_FOTOGRAMA; es lo que la regla «imputacion» rellena; la heredada lo deja en ceros).
- **prop_sin_mano_activa**: fotogramas sin el bloque de mano que más recorre / fotogramas (X_padded en ceros).
- **manos**: proxy: «dos» si el recorrido de la muñeca que menos se mueve es ≥ 0,5 veces el de la que más se mueve; no es una anotación lingüística.

Exactitudes con las 3 semillas juntas (cada video cuenta 3 veces); pruebas sobre el voto TCN con el estadístico del informe previo (`analisis_errores.permutacion_estratificada`: diferencia errores − aciertos dentro de estrato, permutación dentro del estrato, 10000 permutaciones; Holm sobre las 9 pruebas de cada red).

**Descriptivo por señante** (`propiedades.descriptivo_por_senante`):

| señante | longitud media | fotogramas sin alguna mano | sin la mano que más se mueve | videos con alguna mano ausente |
|---|---|---|---|---|
| 0000 | 72,2 | 11,6 % | 5,9 % | 177 de 200 |
| 0001 | 70,4 | 3,7 % | 2,5 % | 115 de 199 |
| 0002 | 66,7 | 16,8 % | 12,4 % | 187 de 198 |
| 0003 | 56,4 | 7,8 % | 5,4 % | 163 de 200 |
| 0004 | 71,0 | 3,6 % | 2,4 % | 109 de 200 |

### 6.1 Mano ausente (antes de imputar)

| tramo de fotogramas sin alguna mano | videos | exactitud heredada | exactitud TCN | solo TCN / solo heredada (voto) |
|---|---|---|---|---|
| 0 % | 246 | 0,575 | 0,665 | 44 / 16 |
| (0; 5 %] | 261 | 0,645 | 0,722 | 37 / 13 |
| (5; 15 %] | 294 | 0,600 | 0,669 | 43 / 16 |
| > 15 % | 196 | 0,517 | 0,645 | 39 / 11 |

| tramo de fotogramas sin la mano que más se mueve | videos | exactitud heredada | exactitud TCN |
|---|---|---|---|
| 0 % | 344 | 0,629 | 0,717 |
| (0; 5 %] | 323 | 0,598 | 0,668 |
| (5; 15 %] | 222 | 0,610 | 0,694 |
| > 15 % | 108 | 0,395 | 0,543 |

**Pruebas dentro de estrato** (`propiedades.modelos.<red>.pruebas_voto`; diferencia = media de errores − media de aciertos, en puntos porcentuales o fotogramas):

| medida | estratificación | heredada: diferencia (p; p Holm) | TCN: diferencia (p; p Holm) |
|---|---|---|---|
| prop_sin_alguna_mano | dentro de señante | +1,77 pp (0,0051; 0,0357) | +1,52 pp (0,0250; 0,1750) |
| prop_sin_alguna_mano | dentro de seña | +2,52 pp (0,0010; 0,0080) | +1,59 pp (0,0554; 0,3324) |
| prop_sin_alguna_mano | dentro de seña y señante | +2,54 pp (0,0161; 0,0805) | +0,48 pp (0,6148; 1,0000) |
| prop_sin_mano_activa | dentro de señante | +2,77 pp (< 0,0001; 0,0009) | +2,52 pp (< 0,0001; 0,0009) |
| prop_sin_mano_activa | dentro de seña | +1,46 pp (0,0341; 0,1364) | +1,21 pp (0,0968; 0,4840) |
| prop_sin_mano_activa | dentro de seña y señante | +1,72 pp (0,0460; 0,1380) | +0,29 pp (0,7151; 1,0000) |
| longitud | dentro de señante | -1,73 fotogramas (0,0057; 0,0357) | -2,01 fotogramas (0,0036; 0,0288) |
| longitud | dentro de seña | +0,55 fotogramas (0,5208; 0,5208) | +0,45 fotogramas (0,6053; 1,0000) |
| longitud | dentro de seña y señante | +1,54 fotogramas (0,1481; 0,2962) | -0,22 fotogramas (0,8354; 1,0000) |

**Lectura.** Dentro de cada señante, los videos fallados tienen más fotogramas sin la mano que más se mueve, con las dos redes (p Holm 0,0009 y 0,0009), y son unos 2 fotogramas más cortos (p Holm 0,0357 y 0,0288). Al comparar videos de la misma seña, la duración no se asocia con el error en ninguna red; la mano ausente sí con la heredada (sin alguna mano, dentro de seña: p Holm 0,0080) pero no con la TCN (p Holm 0,3324). Lo que sí distingue a los videos fallados es sobre todo qué seña y qué señante son; la diferencia entre las dos redes es compatible con que la imputación quite parte del efecto de la mano ausente, pero esta comparación no lo aísla (§6.2).

### 6.2 Aislar la imputación: TCN con «mascara» (T0) frente a la TCN final (T1)

Fuente: `imputacion_t1_vs_t0`. Misma arquitectura, rasgos, semillas y pliegues; solo cambia la regla de mano ausente. Global (voto): solo T1 88, solo T0 59, p = 0,0206 (reproduce `resultados_confirmacion_tcn.json › mcnemar.T1_vs_T0`).

| tramo | medida | videos | exactitud T0 (mascara) | exactitud T1 (imputación) | solo T1 / solo T0 (voto) | p |
|---|---|---|---|---|---|---|
| 0 % | sin alguna mano | 246 | 0,640 | 0,665 | 16 / 10 | 0,3269 |
| (0; 5 %] | sin alguna mano | 261 | 0,715 | 0,722 | 17 / 18 | 1,0000 |
| (5; 15 %] | sin alguna mano | 294 | 0,627 | 0,669 | 27 / 14 | 0,0596 |
| > 15 % | sin alguna mano | 196 | 0,616 | 0,645 | 28 / 17 | 0,1352 |
| 0 % | sin la mano que más se mueve | 344 | 0,690 | 0,717 | 21 / 12 | 0,1628 |
| (0; 5 %] | sin la mano que más se mueve | 323 | 0,678 | 0,668 | 19 / 23 | 0,6440 |
| (5; 15 %] | sin la mano que más se mueve | 222 | 0,616 | 0,694 | 30 / 10 | 0,0022 |
| > 15 % | sin la mano que más se mueve | 108 | 0,519 | 0,543 | 18 / 14 | 0,5966 |

Del neto de +29 videos a favor de la imputación, +24 están en los 490 videos con más de 5 % de fotogramas sin alguna mano y +5 en los 507 restantes. Es el patrón esperable si la imputación ayuda donde hay manos que rellenar, pero ninguna celda por separado llega a p < 0,05 salvo «sin la mano que más se mueve, (5; 15 %]», no se probó la interacción y las celdas se miraron después de ver los datos: es descriptivo.

### 6.3 Duración del video

| quintil | fotogramas | videos | exactitud heredada | exactitud TCN |
|---|---|---|---|---|
| 1 | 40–57,0 | 206 | 0,623 | 0,720 |
| 2 | 57–64,0 | 211 | 0,528 | 0,651 |
| 3 | 64–70,0 | 205 | 0,530 | 0,626 |
| 4 | 70–76,8 | 175 | 0,617 | 0,650 |
| 5 | 77–135,0 | 200 | 0,655 | 0,737 |

### 6.4 Manos en movimiento (una o dos)

Proxy por video con umbral 0,5 sobre el cociente de recorridos de las muñecas (fijado mirando solo la distribución del cociente, sin aciertos). 497 videos con una mano en movimiento y 500 con dos; por mayoría de sus videos, 27 señas son «de una mano» y 23 «de dos» (116 videos difieren de la etiqueta de su seña). Señas «de dos manos»: gracias, buenos días, buenas tardes, buenas noches, seña, nombre, trabajar, vivir, familia, tío, hermano, hambre, feliz, contento, ¿cómo estás?, juicioso, bienvenido, perdón, ¿qué?, ¿dónde?, ¿cómo?, diferente, mucho. Es una medida del movimiento de las muñecas, no una anotación lingüística: una seña de una mano con la otra en apoyo que se desplaza puede salir «de dos».

| grupo | videos | exactitud heredada | exactitud TCN | solo TCN / solo heredada (voto) |
|---|---|---|---|---|
| una mano se mueve (por video) | 497 | 0,500 | 0,633 | 105 / 27 |
| se mueven las dos (por video) | 500 | 0,678 | 0,721 | 58 / 29 |
| seña «de una mano» (por seña) | 539 | 0,462 | 0,596 | — |
| seña «de dos manos» (por seña) | 458 | 0,739 | 0,773 | — |

Por señante (TCN, por video, una / dos): 0000 0,38 / 0,55; 0001 0,53 / 0,69; 0002 0,65 / 0,80; 0003 0,73 / 0,83; 0004 0,77 / 0,82. Entre las 50 señas, correlación entre «de dos manos» y el recall de la TCN: 0,445 (p por permutación 0,0014); heredada 0,530 (p < 0,0001). Como la etiqueta es casi constante dentro de cada seña, esto compara señas, no videos de la misma seña.

### 6.5 Confianza de la predicción y top-3

| tramo de confianza | heredada: predicciones (%) | heredada: exactitud | TCN: predicciones (%) | TCN: exactitud |
|---|---|---|---|---|
| [0,00; 0,25) | 660 (22,1 %) | 0,267 | 148 (4,9 %) | 0,291 |
| [0,25; 0,50) | 702 (23,5 %) | 0,439 | 514 (17,2 %) | 0,292 |
| [0,50; 0,75) | 542 (18,1 %) | 0,620 | 562 (18,8 %) | 0,555 |
| [0,75; 1,00] | 1087 (36,3 %) | 0,867 | 1767 (59,1 %) | 0,860 |

- Confianza media frente a exactitud (3 semillas): TCN 0,733 frente a 0,677 (ECE 0,062; por semilla 0,063 / 0,076 / 0,057); heredada 0,550 frente a 0,589 (ECE 0,046). La TCN es, en promedio, más confiada que acertada; la heredada, al revés.
- Errores con confianza ≥ 0,75: TCN 247 de 966 (25,6 %); heredada 145 de 1229 (11,8 %). Con el umbral 0,00 aprobado ninguno se rechaza; esto describe el umbral, no es un criterio para cambiarlo.
- La seña correcta está en el top-3 en 507 de los 966 errores de la TCN (52,5 %; en segundo lugar en 357); heredada 522 de 1229 (42,5 %).
- Figura: `4d_acierto_vs_propiedades.png` (a: mano ausente; b: duración; c: confianza; d: manos).

## 7. Consistencia entre semillas

Fuente: `consistencia`. Para cada video, cuántas de las 3 semillas lo fallan.

| red | 0 de 3 | 1 de 3 | 2 de 3 | 3 de 3 | de los que fallan las 3, misma seña errónea | fallos estables (3 de 3) / videos con algún fallo |
|---|---|---|---|---|---|---|
| heredada | 451 | 139 | 131 | 276 | 91 | 50,5 % |
| TCN final | 537 | 143 | 128 | 189 | 55 | 41,1 % |

| señante | TCN: 0 / 1 / 2 / 3 semillas fallan | heredada: 0 / 1 / 2 / 3 |
|---|---|---|
| 0000 | 54 / 45 / 41 / 60 | 60 / 27 / 37 / 76 |
| 0001 | 95 / 27 / 26 / 51 | 83 / 19 / 33 / 64 |
| 0002 | 118 / 26 / 22 / 32 | 92 / 38 / 23 / 45 |
| 0003 | 133 / 22 / 19 / 26 | 106 / 23 / 20 / 51 |
| 0004 | 137 / 23 / 20 / 20 | 110 / 32 / 18 / 40 |

Señas con más videos fallados por las 3 semillas (TCN): comer 10; ¿cuándo? 10; niña 8; bien 8; más o menos 8; ¿qué? 8; personas 7; mujer 7; sentir 7; permiso 7; mucho 7; poco 6. Pares que las 3 semillas repiten exactamente (TCN): niña → niño ×4; ¿cómo estás? → ¿cómo? ×4; niña → abuelo ×3; hermano → tío ×3; más o menos → poco ×3; hola → abuelo ×3; adiós → abuelo ×3; ¿cuándo? → mujer ×3.

## 8. Los errores vistos por el conversor de glosas (modo «inventario», D-15)

Fuente: `conversor`. inventario (ConversorGlosas(estricto=True)); una seña aislada → su enunciado fijo de glosa sola. En el modo inventario cada seña aislada da su enunciado fijo («Niño.», «¿Cómo?»), así que un error de reconocimiento produce un enunciado bien formado con otro contenido; el conversor no tiene cómo notarlo. Categorías: las 15 del conversor (`conversor_glosas.LEXICO`). Solo descriptivo.

| red (voto) | errores | misma categoría | cambia de categoría | esperado misma categoría al azar | cambia de modalidad (pregunta ↔ enunciado) | misma categoría y distinto género | misma categoría en grammar_mapper (8 categorías) |
|---|---|---|---|---|---|---|---|
| heredada | 390 | 84/390 (21,5 %) | 306/390 (78,5 %) | 8,0 % | 82/390 (21,0 %) | 20 | 133/390 (34,1 %) (azar 13,5 %) |
| TCN final | 283 | 75/283 (26,5 %) | 208/283 (73,5 %) | 8,1 % | 52/283 (18,4 %) | 19 | 103/283 (36,4 %) (azar 13,7 %) |

Con las 3 semillas (TCN): 238 de 966 errores (24,6 %) quedan en la misma categoría y 186 (19,3 %) cambian de modalidad.
- Cambios de categoría más frecuentes (TCN, voto): ADJ → NOM_PERS ×11; PREG_FIJA → INTERR ×9; SAL → NOM_PERS ×9; INTERR → NOM_PERS ×9; INTERR → PREG_FIJA ×9; PRON → CORT ×8; VERBO → NOM_PERS ×7; VERBO_SENTIR → CORT ×7.
- Errores dentro de la misma categoría (TCN, voto), por categoría: NOM_PERS 29; INTERR 28; ADJ 9; CORT 7; ADV_EST 2.
- Enunciados más frecuentes (dicho → mostrado; TCN, voto): «¿Cómo?» → «¿Qué?» ×7; «Aburrido.» → «Niño.» ×6; «¿Cómo estás?» → «¿Cómo?» ×6; «¿Qué?» → «¿Dónde?» ×6; «¿Cuándo?» → «Mujer.» ×6; «¿Por qué?» → «Hola.» ×6; «Niño.» → «Niña.» ×5; «Niña.» → «Niño.» ×5.

## 9. Partición aleatoria (señantes vistos)

Fuente: `aleatoria`. 100 videos de prueba (señantes 0000: 16, 0001: 17, 0002: 20, 0003: 21, 0004: 26). TCN 94 aciertos, heredada 88. Tabla por video: ambos aciertan 86, solo TCN 8, solo heredada 2, ambos fallan 4 (McNemar p = 0,1094: con 100 videos la diferencia no es distinguible del azar). Confianza media de la TCN: aciertos 0,939, errores 0,632.

**Los 6 errores de la TCN** (`aleatoria.errores.tcn`; enunciado del conversor en modo inventario):

| video | señante | real → predicha | confianza | posición de la real en el top-3 | fotogramas | sin alguna mano | manos | heredada predice | enunciado dicho → mostrado |
|---|---|---|---|---|---|---|---|---|---|
| 0004_0001_0000 | 0001 | seña (4) → personas (11) | 0,11 | fuera | 73 | 28,8 % | dos | juicioso (29) | «Seña.» → «Personas.» |
| 0007_0000_0000 | 0000 | comer (7) → mujer (12) | 0,43 | 3 | 81 | 2,5 % | una | mujer (12) | «Comer.» → «Mujer.» |
| 0012_0000_0003 | 0000 | mujer (12) → yo (38) | 0,95 | fuera | 84 | 32,1 % | una | ¿cuándo? (42) | «Mujer.» → «Yo.» |
| 0015_0003_0001 | 0003 | niña (15) → niño (14) | 0,87 | 2 | 51 | 9,8 % | una | niño (14) | «Niña.» → «Niño.» |
| 0026_0003_0003 | 0003 | ¿cómo estás? (26) → ¿cómo? (44) | 0,95 | 3 | 59 | 6,8 % | dos | ¿cómo estás? (26) | «¿Cómo estás?» → «¿Cómo?» |
| 0047_0002_0000 | 0002 | diferente (47) → ¿cómo estás? (26) | 0,48 | 2 | 62 | 12,9 % | dos | diferente (47) | «Diferente.» → «¿Cómo estás?» |

Errores de la heredada en la misma partición: 0004_0001_0000 seña → juicioso; 0007_0000_0000 comer → mujer; 0009_0004_0000 poco → más o menos; 0012_0000_0003 mujer → ¿cuándo?; 0015_0003_0001 niña → niño; 0023_0002_0000 aburrido → tú; 0025_0000_0001 mal → poco; 0038_0001_0000 yo → por favor; 0041_0002_0002 ¿qué? → personas; 0042_0004_0002 ¿cuándo? → mujer; 0044_0000_0001 ¿cómo? → ¿cómo estás?; 0044_0003_0000 ¿cómo? → ¿cómo estás?.

## 10. Contraste con el informe de la heredada

Fuente: `contraste_informe_previo` (informe previo: `models/analisis_errores_manos_2026-09-15/analisis.json`, una sola semilla, la 2026). Parejas que el informe previo puso arriba, contadas ahora con 3 semillas (dos sentidos):

| pareja | informe previo (1 semilla) | heredada (3 semillas) | TCN (3 semillas) |
|---|---|---|---|
| niño (14) / niña (15) | 11 | 38 | 31 |
| ¿cómo estás? (26) / ¿cómo? (44) | 11 | 37 | 33 |
| comer (7) / abuelo (16) | 9 | 17 | 9 |
| más o menos (27) / yo (38) | 9 | 18 | 10 |
| ¿qué? (41) / ¿dónde? (43) | 9 | 24 | 26 |
| poco (9) / más o menos (27) | 8 | 24 | 15 |
| mujer (12) / permiso (36) | 8 | 21 | 4 |
| niño (14) / tú (39) | 8 | 19 | 2 |
| permiso (36) / ¿cuándo? (42) | 8 | 25 | 5 |
| gracias (0) / perdón (35) | 7 | 16 | 3 |

Peores 10 señas del informe previo y su recall ahora:

| seña | recall informe previo | heredada (3 semillas) | TCN (3 semillas) | puesto en la TCN (1 = peor) |
|---|---|---|---|---|
| más o menos (27) | 0,00 | 0,12 | 0,47 | 7 |
| comer (7) | 0,05 | 0,02 | 0,35 | 3 |
| permiso (36) | 0,15 | 0,20 | 0,42 | 4 |
| niño (14) | 0,15 | 0,35 | 0,58 | 20 |
| ¿cómo estás? (26) | 0,15 | 0,30 | 0,43 | 5 |
| ¿cuándo? (42) | 0,20 | 0,25 | 0,33 | 2 |
| mujer (12) | 0,25 | 0,25 | 0,48 | 10 |
| bien (24) | 0,30 | 0,42 | 0,47 | 8 |
| sentir (28) | 0,30 | 0,47 | 0,52 | 13 |
| niña (15) | 0,35 | 0,40 | 0,55 | 17 |

**Lectura.** Siguen difíciles con la TCN (≥ 15 errores en 3 semillas) las parejas niño / niña (31); ¿cómo estás? / ¿cómo? (33); ¿qué? / ¿dónde? (26); poco / más o menos (15): todas dentro de la misma categoría de grammar_mapper (sustantivos de persona, interrogativos, adverbios). Bajan a la mitad o menos: mujer / permiso (21 → 4); niño / tú (19 → 2); permiso / ¿cuándo? (25 → 5); gracias / perdón (16 → 3). Aparecen arriba con la TCN parejas que el informe previo no destacaba (p. ej. mujer / ¿cuándo? (20); gracias / permiso (17); hola / ¿por qué? (17); comer / mujer (16)). Errores dentro de la misma categoría de grammar_mapper: 29,6 % en el informe previo (1 semilla); con el voto, 34,1 % la heredada y 36,4 % la TCN: casi igual, siempre muy por encima del azar (≈ 13,6 %). El señante 0000 sigue siendo el peor. La asociación con la mano ausente que encontró el informe previo se mantiene con la heredada (voto) también dentro de la misma seña, y con la TCN solo dentro de señante (§6.1).

## 11. Qué se puede y qué no se puede concluir

- Hay 5 señantes. Todo lo que se diga «por señante» son 5 casos: no se puede estimar qué propiedad del señante explica su exactitud, ni generalizar a señantes nuevos con un intervalo útil.
- Cada seña tiene 4 videos por señante (3 en tres casos): diferencias de recall por seña de 1-2 videos, o de 0,05-0,10 en el agregado, no deben leerse como efectos.
- Las relaciones con la duración, la mano ausente y las manos en movimiento son asociaciones observadas, no causas. Las tres variables cambian con la seña y con el señante; por eso se comparan dentro de estrato, y aun así un tercer factor (la forma de la seña, la cámara, el ritmo del señante) puede explicar ambas cosas.
- La comparación TCN frente a heredada cambia a la vez la arquitectura y la regla de mano ausente. Solo T1 frente a T0 (§6.2) aísla la imputación, y sus celdas por tramo son pequeñas y se miraron después de ver los datos.
- El voto de 3 semillas es un recurso de análisis; el prototipo usa un solo modelo (semilla 2030, partición aleatoria). Las cifras del modelo desplegado son las de la partición aleatoria (§9) y, para señantes nuevos, la media de pliegues.
- El proxy «una / dos manos» mide el recorrido de las muñecas, no la estructura lingüística de la seña; no fue revisado por una persona señante.
- La lectura del conversor es descriptiva: dice qué enunciado vería el usuario, no cuánto molesta cada error.
- Nada de esto se usó para elegir el modelo final (D-14) ni el umbral; no se toma ninguna decisión a partir de este análisis.

## Para la tesis

1. La TCN corrige 163 videos que la heredada fallaba e introduce 56 errores nuevos (voto de 3 semillas, McNemar p = 2,6·10^-13); el saldo es positivo en 5 de 5 señantes y en 8 de 8 categorías gramaticales.
2. Los errores que quedan se concentran en parejas de la misma clase (interrogativos; niño / niña): ¿cómo estás? / ¿cómo? (33 errores en 3 semillas), niño / niña (31) y ¿qué? / ¿dónde? (26); 3 de esas 3 parejas están también entre las 10 más confundidas por la heredada.
3. El señante 0000 sigue siendo el peor (0,488 frente a 0,452 de la heredada) y es donde más discrepan las semillas (45 + 41 de sus 200 videos los fallan 1 o 2 de las 3 semillas).
4. Las señas en que se mueve una sola mano se aciertan menos que las de dos manos (0,63 frente a 0,72 con la TCN; menor en 5 de 5 señantes), y es allí donde la TCN más mejora a la heredada (corrige 105 videos e introduce 27 errores; con dos manos, 58 y 29).
5. La mano ausente se asocia con el error dentro de cada señante (+2,5 pp de fotogramas sin la mano que más se mueve en los errores, p Holm 0,0009), pero no al comparar videos de la misma seña: con la TCN no hay evidencia de que un video con más fotogramas perdidos falle más que otro de la misma seña.
6. La TCN es más confiada que acertada (confianza media 0,733 frente a exactitud 0,677; la heredada era menos confiada que acertada): 25,6 % de sus errores llegan con confianza ≥ 0,75 (heredada 11,8 %).
7. En 52,5 % de los errores la seña correcta está entre las 3 primeras (heredada 42,5 %).
8. Vistos por el conversor (modo inventario), 73,5 % de los errores cambian de categoría (p. ej. adjetivo por sustantivo de persona; los que la mantienen, 26,5 %, triplican lo esperable al azar, 8,1 %) y 18,4 % convierten una pregunta en enunciado o al revés; el enunciado sale bien formado, de modo que el usuario no ve señal de error.

## Figuras

- `docs/figuras_2026-11-08/4d_matriz_confusion_loso_tcn.png`
- `docs/figuras_2026-11-08/4d_parejas_confundidas_heredada_vs_tcn.png`
- `docs/figuras_2026-11-08/4d_recall_por_clase_heredada_vs_tcn.png`
- `docs/figuras_2026-11-08/4d_acierto_vs_propiedades.png`

## Reproducir

`python src/analisis_errores_tcn.py` (intérprete de venv312; lectura de `data/` en modo solo lectura; segundos de cómputo). Pruebas: `python -m unittest tests.test_analisis_errores_tcn -v`.
