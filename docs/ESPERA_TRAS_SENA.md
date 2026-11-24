# Espera tras terminar la seña con la ventana fija de 2,5 s

Generado por `src/espera_tras_sena.py` el 2026-11-23T23:11:14. Solo análisis.

Definición: espera tras la seña = (cierre de la ventana − fin de la seña) + L_resp, con L_resp = t5 − t1 medido desde el cierre de la ventana.

Supuesto: la ventana abre al inicio del clip (la persona arranca desde el reposo tras pulsar ESPACIO) y cierra a los 2,5 s (60 fotogramas a 24 fps); fin de la seña = último fotograma activo de src/simular_ventana_25.py (tramo_sena).

- L_resp en vivo (23-nov, 64 ensayos con audio): mediana 32,1 ms, p10 29,4 ms, p90 33,6 ms.
- Clips: 997; la seña termina dentro de la ventana en 894 (89,7 %); en 103 la ventana la corta.
- Cierre − fin de la seña: mediana 0,67 s, p10 0,21 s, p90 1,17 s.
- **Espera tras la seña** (con la mediana de L_resp): mediana 0,70 s, p10 0,24 s, p90 1,20 s, máx. 1,74 s.
- Con todos los L_resp medidos (convolución): mediana 0,70 s, p10 0,24 s, p90 1,20 s.
- Espera ≤ 0,2 s: 9,2 %; ≤ 0,5 s: 31,8 %; ≥ 1 s: 22,7 %.

Por señante:

| señante | clips dentro / total | mediana (s) | p10 (s) | p90 (s) |
|---|---|---|---|---|
| 0000 | 187 / 200 | 0,53 | 0,22 | 0,87 |
| 0001 | 181 / 199 | 0,70 | 0,28 | 1,03 |
| 0002 | 186 / 198 | 0,78 | 0,32 | 1,09 |
| 0003 | 192 / 200 | 1,07 | 0,37 | 1,36 |
| 0004 | 148 / 200 | 0,53 | 0,16 | 1,03 |

Sensibilidad: la ventana abre D s antes del inicio de la seña:

| D (s) | la seña cabe | mediana (s) | p10 (s) | p90 (s) |
|---|---|---|---|---|
| 0,25 | 93,9 % | 0,87 | 0,37 | 1,12 |
| 0,50 | 88,9 % | 0,62 | 0,24 | 0,87 |
| 1,00 | 59,1 % | 0,20 | 0,07 | 0,41 |

## Avisos de seña cortada: corpus frente a en vivo

Corpus: cadena integrada sobre 997 videos (`models/evaluacion_punta_a_punta_2026-11-23/crudo.json`, puntos `video_1280x720`, modelos LOSO; los avisos no dependen del modelo). En vivo: `registros/sesion_final_2026-11-23/latencia_2026-11-23.jsonl`, ventanas en modo aislado.

| fuente | ventana | aviso de fin | aviso de inicio | inicio o fin |
|---|---|---|---|---|
| corpus | clip completo | 82 (8,2 %) | 75 (7,5 %) | 144 (14,4 %) |
| corpus | primeros 2,5 s | 259 (26,0 %) | 75 (7,5 %) | 324 (32,5 %) |
| en vivo 23-nov | 2,5 s | 69 de 87 (79,3 %) | 32 (36,8 %) | — |

Límites: el inicio de la ventana en vivo depende de cuándo pulsa la persona y cuándo arranca; aquí se toma el del clip del corpus. El fin de la seña sale de un criterio heurístico de reposo (ver docs/SIMULACION_VENTANA_25.md, tabla de sensibilidad). La espera está dominada por la ventana fija, que es una decisión de diseño, no por el cómputo.
