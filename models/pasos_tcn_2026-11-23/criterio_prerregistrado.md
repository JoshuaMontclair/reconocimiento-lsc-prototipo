# Experimento: 30 frente a 60 pasos temporales con el modelo final (TCN + imputación + 252)

Estado: **§1 escrita el 22-nov-2026 a las 23:55 (hora de Colombia), antes de cualquier entrenamiento con
60 pasos** (ni siquiera la prueba de humo). Lo que sigue a §1 se escribe después y no la modifica. Copia de §1
con su SHA-256: `models/pasos_tcn_2026-11-23/criterio_prerregistrado.md` y `.sha256`.

## 1. Pregunta, diseño y criterio prerregistrado

### 1.1 Pregunta

El anteproyecto fijaba 60 pasos temporales; el modelo final (D-14: TCN pequeña con GAP + imputación de la
mano ausente + 252 rasgos, brazo T1 de `models/confirmacion_tcn_2026-10-03/T1_imputacion_252`) usa 30. El
valor 30 salió del experimento de pasos de septiembre (`docs/EXPERIMENTO_PASOS_TEMPORALES.md`, N-58), hecho con
la red de partida (Conv1D + BiLSTM + atención). Allí la regla prerregistrada señaló 60 (+1,6 puntos de
validación, 5 de 5 pliegues), sin confirmación en prueba (+0,9 puntos, 2 de 5 pliegues), y el proyecto siguió con
30. La comparación nunca se repitió con la TCN. ¿Con la TCN, 60 pasos es mejor que 30?

### 1.2 Brazos (todo lo demás igual a T1)

| Brazo | Pasos | Origen |
|---|---:|---|
| **P30** | 30 | T1 de la confirmación, **reutilizado** si se reproduce (§1.4) |
| **P60** | 60 | **nuevo**: 15 corridas (5 pliegues × semillas 2026, 2027, 2028) |

- **Arquitectura:** `CONFIG_T` de `src/arquitecturas_literatura.py` sin ningún cambio (una sola configuración,
  fijada antes): 4 bloques residuales con dilataciones 1-2-4-8, 2 convoluciones por bloque, 64 filtros, k = 3,
  padding «same», dropout 0,2 por bloque y 0,3 final, GlobalAveragePooling1D, softmax de 50. Se construye con
  `arquitecturas_literatura.construir_tcn(60, 252, 50)`. **156.402 parámetros en los dos brazos**: con GAP, la
  longitud no cambia el tamaño de la red.
- **Campo receptivo:** 1 + 2·(k − 1)·(1 + 2 + 4 + 8) = **61 pasos** (cada salida ve ±30 pasos). Con 30 pasos,
  cada una de las 30 posiciones ve la ventana completa. Con 60, solo las 2 posiciones centrales ven los 60 pasos;
  las de los bordes ven 31. GAP promedia todas. No se añade un bloque ni se cambia la dilatación para
  «compensar»: eso sería otra arquitectura y la pregunta es solo la longitud.
- **Entrada:** `mano_ausente_variantes.remuestrear_todas_con_regla(X_padded, lengths, "imputacion", 60)` sobre
  `data/processed_holistic_mascara_rostro`, representación «manos» (126 posiciones), y después de la aumentación
  `add_velocity` (126 velocidades) → 60 × 252. Consecuencia declarada: con 60 pasos la velocidad es la diferencia
  entre pasos dos veces más cercanos, de modo que su magnitud es aproximadamente la mitad que con 30; no se
  reescala (sería cambiar la entrada). Con 60 pasos se estiran por interpolación los videos de menos de 60
  cuadros (261 de 1000 según N-58; se recuenta en `verificar`).
- **Receta:** la de T1, sin cambios, a través de `experimento_arquitecturas.entrenar_corrida` (constructor «T»):
  Adam lr 1e-3 con CosineDecay, *label smoothing* 0,1, lote 64, máx. 200 épocas, paciencia 25 sobre
  `val_accuracy`, checkpoint de mejor `val_accuracy`, aumentación ×10 (`train_loso.aumentar` con las
  transformaciones de `refined_augmentation`), CPU.
- **Particiones:** `data/partitions/loso_5_pliegues.json` (D-06), validación interna de 2 por seña (100 muestras
  por pliegue). Solo se leen.

### 1.3 Criterio de decisión (el mismo tipo de regla que el factorial de octubre, N-63)

**Solo deciden métricas de validación LOSO. La prueba LOSO (top-1, top-3, F1) y el señante 0000 se reportan
únicamente como descripción y no deciden nada.**

Notación: `val(V, s, k)` es el `val_accuracy` máximo (el del checkpoint) del brazo V, semilla s, pliegue k;
`loss(V, s, k)` es la `val_loss` de ese checkpoint (entropía cruzada con *label smoothing* 0,1, la que guarda
`entrenar_corrida`). `val(V, k)` y `loss(V, k)` son la media de las 3 semillas. Las diferencias por pliegue se
redondean a 6 decimales y toda comparación con un umbral usa una tolerancia de 1e-9 (como en la confirmación y el
factorial; funciones de `experimento_confirmacion_tcn` y `experimento_factorial_tcn`).

1. **Regla 1: top-1 de validación con margen.**
   - **Margen** = máx(σ, 0,01). σ = DE entre semillas agrupada sobre P30 y P60: para cada brazo, la DE
     (ddof = 1) entre las 3 semillas de la validación media de 5 pliegues; luego, la raíz de la media de las
     dos varianzas.
   - **A supera a B** si la media de Δk = val(A, k) − val(B, k) es mayor que el margen **y** Δk > 0 en al menos
     4 de los 5 pliegues (Δk = 0 no cuenta como mejora).
   - Se evalúan los dos sentidos. Si P60 supera a P30, **se recomienda 60**. Si P30 supera a P60, **se mantiene
     30**. Si ninguno supera al otro, se pasa a la regla 2.
2. **Regla 2: desempate con la `val_loss`** (menor es mejor), como el paso 2 del factorial.
   - σ_loss = DE entre semillas de la `val_loss` media de 5 pliegues, agrupada sobre P30 y P60 igual que σ (sin
     piso).
   - **A supera a B en pérdida** si dk = loss(B, k) − loss(A, k) > 0 en al menos 4 de los 5 pliegues **y** la
     media de dk es mayor que σ_loss.
   - Si P60 supera a P30 en pérdida, **se recomienda 60**. Si P30 supera a P60, **se mantiene 30**. Si ninguno,
     regla 3.
3. **Regla 3: simplicidad y costo → 30.** 30 es la longitud vigente del modelo final y del prototipo (umbral,
   reporte 4.e, latencia y análisis de errores se hicieron con 30), cuesta la mitad de cómputo convolucional por
   ventana y no estira ningún video del corpus. Ante un empate en validación, se mantiene 30 sin condiciones.
4. **Descriptivo, sin valor de decisión:** prueba top-1, top-3 y F1 macro (media de 3 semillas, por pliegue y por
   semilla); el señante 0000 (pliegue 0) aparte; McNemar exacto sobre las 997 predicciones de prueba agregadas,
   con cada semilla y con el voto de las 3 semillas (mayoría; si las tres discrepan, gana la semilla más
   confiada); signos y Wilcoxon por pliegue en validación, `val_loss`, prueba top-1, top-3 y F1 (con 5 pliegues el
   p bilateral mínimo es 2/2⁵ = 0,0625, así que ninguna llega a 0,05); épocas y tiempo de entrenamiento.
5. **Costo (descriptivo, después de las corridas y sin entrenamientos activos):** parámetros, tamaño en disco,
   latencia de la red con lote de 1 (`model.predict`, como `translator.py`, y llamada directa `model(x)`) y costo de
   armar la entrada de una ventana (imputación → remuestreo → velocidades) con 30 y con 60 pasos.
6. **Condición previa para reutilizar P30.** Antes de lanzar P60 se reentrena T1, semilla 2026, pliegue 0, con el
   código de este experimento (`src/experimento_pasos_tcn.py`, con L = 30), en una carpeta aparte. P30 se reutiliza
   si `predicciones.json` sale idéntico byte a byte o, en su defecto, dentro de la tolerancia que usó la
   confirmación (`experimento_confirmacion_tcn.comparar_corridas`: mismas predicciones y top-3, mismas épocas,
   misma validación y diferencia máxima de probabilidades de prueba < 1e-5). Si no, se entrenan también las 15
   corridas de P30 con este script y se usan esas.
7. **Plan de ejecución.** Tres procesos en paralelo, uno por semilla, cada uno con los pliegues 0 → 4, con
   `nohup` y registro en `models/pasos_tcn_2026-11-23/`. Si no terminaran, el criterio se aplica con las semillas
   que tengan los 5 pliegues en los dos brazos (mínimo 2) y se declara.
8. **Estimación de duración** (con los registros de T1): T1 tardó de 85 a 192 s por corrida (media 123 s; 43 a 85
   épocas, media 55,7; 2,15 s por época) con 3 procesos en paralelo. La TCN recorre el doble de pasos, así que se
   espera ≈ 2 veces el costo por época (≈ 4,3 s) y ≈ 240 s por corrida con épocas parecidas: ≈ 20 min por proceso
   (5 corridas) y **≈ 20-30 min de reloj** para las 15 corridas. La reproducción previa (§1.6) añade ≈ 2 min.

**Lo que no se hace:** no se prueban otras longitudes (45, 90…), ni otra arquitectura para 60 pasos, ni otra
receta. No se cambia el modelo final, el umbral ni `translator.py`.
