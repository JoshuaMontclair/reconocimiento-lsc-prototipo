# Integridad del corpus LSC50 archivo por archivo (actividad 1.a)

Generado por `src/cotejo_integridad_corpus.py` el 2026-09-16T11:33:13. Depósito: figshare, artículo 27383016.

## Qué se comparó

El índice central de cada ZIP publicado guarda, por cada archivo, su tamaño sin comprimir y su CRC-32. Ese índice se leyó por rangos HTTP, sin descargar los ZIP (3.816.655 bytes transferidos en 2 peticiones). Para cada entrada se calculó el CRC-32 y el tamaño del archivo local y se compararon.

| ZIP | entradas | coinciden | difieren | ausentes en local | sobran en local |
|---|---|---|---|---|---|
| LANDMARKS.zip | 4.000 | 4.000 | 0 | 0 | 0 |
| VIDEOS.zip | 4.000 | 4.000 | 0 | 0 | 0 |
| **total** | **8.000** | **8.000** | **0** | **0** | **0** |

## Resultado

**La copia local es fiel al depósito.** Los 8.000 archivos cotejados tienen el mismo tamaño y el mismo CRC-32 que las entradas publicadas. Ninguno falta y ninguno difiere.

## Archivos locales que no vienen del depósito

- Creados por el sistema operativo (1): `VIDEOS/.DS_Store`. No son parte del corpus.

## Límites de esta comprobación

- El CRC-32 son 32 bits: detecta corrupción accidental (truncamientos, bytes cambiados, copias a medias), que es justo el riesgo que preocupaba. No es una garantía frente a una manipulación deliberada.
- No se comprobó el MD5 de los ZIP publicados, que exige descargarlos enteros (9,0 GB los dos cotejados; 10,6 GB los tres que publica el depósito) y que validaría el archivo comprimido como objeto, no cada archivo del corpus. El MD5 publicado queda anotado en el JSON para quien quiera hacerlo algún día.
- El depósito publica también IMU.zip, que la tesis no usa (D-01): no se cotejó.
- Se compara contra la versión del depósito que la API devuelve hoy. Si los autores publicaran una versión nueva, habría que repetir el cotejo.
