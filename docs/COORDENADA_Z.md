# La coordenada z de las manos tras la normalización por hombros

Generado por `src/coordenada_z.py` el 2026-11-23T22:58:43. Solo análisis; fuente `data/processed_holistic_mascara_rostro` (997 videos válidos, cuadros con la mano detectada).

La normalización resta a todos los puntos el centro de los hombros también en z, y la z de la mano es relativa a la muñeca, así que la z normalizada de la muñeca es −z_hombros / d (la z de la postura, que la documentación pide descartar). Comprobación en los CSV crudos: |z| de la muñeca, mediana 6.4e-08, máx. 2.4e-07 (6713 cuadros).

| mano | cuadros | z' muñeca: mediana (p10–p90) | |z' − z'muñeca| mediana | varianza z' (media de 20 columnas) | de ella, desplazamiento común | varianza x', y' |
|---|---|---|---|---|---|---|
| izquierda | 63012 | 0,88 (0,35 a 1,24) | 0,075 | 0,138 | 97,6 % | 0,607 |
| derecha | 65267 | 0,88 (0,34 a 1,24) | 0,070 | 0,133 | 98,1 % | 0,456 |

Lectura: casi toda la variación de las columnas z de la mano es el desplazamiento común que aporta la z de los hombros (la misma cantidad en los 21 puntos de una mano y un cuadro); la forma de la mano en profundidad es una parte pequeña. No hay en el repositorio una ablación que entrene sin z; su efecto en la exactitud no está medido.
