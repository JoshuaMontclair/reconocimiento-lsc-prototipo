# Criterio de elección, fijado ANTES de medir — señas cortadas en los bordes de la ventana

Fecha y hora de escritura: 2026-09-22 16:17 (hora local, -05). Este archivo se escribe antes de
ejecutar `src/analisis_senas_cortadas.py`; el script lo lee, lo copia en `informe.md` y registra en
`resultados.json` la hora de modificación de este archivo y la hora de inicio de la corrida, para que
el verificador compruebe el orden. Nada de lo que sigue se cambia después de ver resultados.

## Qué se decide con datos

Una regla de detección de «seña cortada» al abrir y al cerrar la ventana, sus umbrales y su
parámetro K (número de fotogramas del borde). La adopción en el prototipo NO se decide aquí: se
propone al director (pieza 4, punto 4).

## Datos

- `data/processed_holistic_mascara_rostro/` (solo lectura): `X_padded.npy` (1000, 135, 405),
  `lengths.npy`, `metadata.json` (`sample_order`). Coordenadas ya normalizadas por hombros
  (centro = punto medio de los hombros, escala = distancia entre hombros en xy; y crece hacia abajo).
  Videos a 24 fps (N-02).
- Particiones: `data/partitions/loso_5_pliegues.json`. En el pliegue k, «entrenamiento» =
  `conjuntos.train` ∪ `conjuntos.val` (los cuatro señantes distintos de k) y «prueba» =
  `conjuntos.test` (todas las muestras no excluidas del señante k). Las tres muestras duplicadas
  excluidas por D-06 no entran en ningún conjunto.

## Rasgos por clip (todos en el espacio normalizado por hombros, plano xy)

Para un clip de T fotogramas y un borde de K fotogramas (K ∈ {3, 5, 8}):

1. `v_pose_ini(K)`: media, sobre los K−1 pares consecutivos de los primeros K fotogramas y sobre
   las dos muñecas de la POSTURA (puntos 15 y 16 de PoseLandmarker; columnas 45:47 y 48:50), del
   desplazamiento ‖Δxy‖ por fotograma. `v_pose_fin(K)`: lo mismo sobre los últimos K fotogramas.
   Un par en el que la postura falta en alguno de los dos fotogramas no cuenta; si no queda ningún
   par, el rasgo es NaN.
2. `v_manos_ini(K)`, `v_manos_fin(K)`: media, sobre los pares consecutivos del borde en que la
   mano está detectada en ambos fotogramas, y sobre las manos detectadas, del desplazamiento medio
   ‖Δxy‖ de los 21 puntos de la mano (bloques 99:162 y 162:225). NaN si no hay ningún par válido.
3. `h_ini(K)`, `h_fin(K)`: media, sobre los K fotogramas del borde, de min(y_muñeca_izq,
   y_muñeca_der) de la postura (la muñeca más alta; y menor = más arriba). NaN si no hay postura.
4. Referencia del tramo central: la misma velocidad sobre el tramo [⌊T/4⌋, T − ⌊T/4⌋) (mitad
   central del clip), para expresar los bordes como cociente borde/centro en el reporte descriptivo.

Se reporta además dónde caen los fotogramas sin mano (primeros K, últimos K, centro), porque si
la mano suele faltar en los bordes, el rasgo 2 queda indefinido justo donde se necesita.

## Clips truncados artificialmente

De cada clip íntegro de longitud T se generan seis versiones: sin los primeros round(p·T)
fotogramas («cortada al inicio») y sin los últimos round(p·T) («cortada al final»), p ∈ {0,20;
0,35; 0,50}. Se recorta ANTES de remuestrear a 30 pasos. Como la normalización y la máscara son por
fotograma, recortar la secuencia procesada equivale a recortar antes de normalizar.

## Familia de reglas y ajuste (solo con los señantes de entrenamiento del pliegue)

Regla(K, rasgo): «cortada al inicio si rasgo_ini(K) supera su umbral; cortada al final si
rasgo_fin(K) supera el suyo». Para la altura la comparación es al revés (cortada si h < h*).

- Rasgos candidatos: `v_pose`, `v_manos`, `h`, y la combinación `v_pose ∨ h` (cortada si
  cualquiera de las dos dispara).
- Umbral de cada rasgo y borde: percentil 95 del rasgo sobre los clips ÍNTEGROS de los señantes
  de entrenamiento del pliegue (percentil 5 para la altura). Es decir, el punto de operación fijado
  es «5 % de falsos positivos esperados sobre clips íntegros de señantes vistos». Un rasgo NaN no
  dispara (la ventana no se marca como cortada por ese rasgo).
- Sensibilidad que se reporta sin cambiar la elección: percentiles 90 y 99.
- Selección de (K, rasgo) por pliegue: entre las 12 combinaciones, la de mayor detección MEDIA sobre
  los clips truncados (los seis niveles p × lado) de los señantes de ENTRENAMIENTO, con el umbral del
  percentil 95 ajustado sobre esos mismos señantes. Empate (diferencia < 0,01): la de menor K y,
  después, el rasgo en el orden `v_pose`, `h`, `v_pose ∨ h`, `v_manos` (los rasgos de la postura
  antes que los de la mano, porque están definidos aunque la mano no se detecte).
- El señante de prueba del pliegue solo se usa para EVALUAR la configuración elegida: falsos
  positivos sobre sus clips íntegros, detección sobre sus clips truncados por nivel y lado.
- La regla que se propone al director es la configuración elegida en la mayoría de los pliegues
  (si no hay mayoría, la de menor K). Los umbrales que se citen como candidatos serán los cinco
  valores por pliegue con su media; el umbral que se implementaría se recalcularía con los cinco
  señantes y quedaría como archivo candidato, nunca sobrescribiendo nada.

## Daño al modelo «manos» (solo se reporta; no decide la regla)

Por pliegue k, con `models/loso_2026-09-15_holistic_mascara_manos/pliegue_k/modelo.keras` y el
preprocesamiento exacto del entrenador (columnas de la representación «manos» del processed,
`resample_sequence` a 30, `add_velocity`), sobre las muestras de prueba del pliegue: top-1, top-3,
confianza media, tasa de rechazo con el umbral vigente 0,20 (aceptada si confianza ≥ 0,20, como el
prototipo) y exactitud entre las aceptadas, para el clip íntegro y para cada nivel y lado de recorte.
CONTROL previo: las predicciones sobre los clips íntegros deben reproducir exactamente el top-1
de `resultados_loso.json` por pliegue; si no, el script se detiene.

Con la regla elegida por pliegue se reporta, sobre los clips truncados del señante de prueba, qué
fracción rechazaría la regla, qué fracción rechazaría el umbral 0,20, y la exactitud de las que pasan
ambos filtros, para cuantificar el daño que la regla evitaría.

## Lo que NO se hace

No se toca ningún modelo, umbral vigente ni partición. No se implementa nada en `translator.py`
ni `realtime.py`. Toda la escritura va a `models/senas_cortadas_2026-09-22/`, a
`src/analisis_senas_cortadas.py` (nuevo) y a `tests/test_analisis_senas_cortadas.py` (nuevo).
