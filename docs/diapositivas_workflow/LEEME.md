# Diapositivas: del cuadro de video a la voz

Presentación Beamer (16:9) para el director (Felipe Palta) y los jurados. Explica el flujo completo del
prototipo, desde la captura hasta la voz, y responde las preguntas de la reunión del 28-sep-2026.

- Fuente: `workflow_lsc.tex`
- PDF: `workflow_lsc.pdf` (48 páginas: 36 del cuerpo y 12 del apéndice, A1 a A11)
- Figuras: `figuras/` (PDF, más los JSON con las cifras que citan las diapositivas)
- Guiones que regeneran las figuras: `guiones/`

Ninguna figura contiene imágenes ni fotogramas de las personas del corpus (07_etica.tex del
anteproyecto). Para ilustrar un cuadro se dibuja solo el esqueleto de puntos sobre fondo liso.

## Compilar

Desde esta carpeta (pdflatex, TeX Live 2026):

```sh
latexmk -pdf workflow_lsc.tex     # dos o tres pasadas: resuelve los números de página del índice
latexmk -c                        # borra los auxiliares y deja el PDF
```

Paquetes usados: beamer, babel (spanish), FiraSans, lmodern, icomma, tikz, tcolorbox, booktabs y
tabularx. La coma decimal va escrita como texto (`0,580`) o, dentro de fórmulas, con `icomma`.
Cada cifra lleva su fuente al pie de la diapositiva (macro `\fuente{...}`).

## Qué guion genera cada figura

Todos se corren desde la raíz del repositorio con el intérprete del proyecto
(`python`). Solo leen `data/` y `models/` y escriben
en `docs/diapositivas_workflow/figuras/`. Ninguno abre la cámara ni reproduce audio.

| Guion | Figuras y datos que escribe | Diapositivas |
|---|---|---|
| `guiones/figuras_datos.py` | `fig_muestras_por_clase.pdf`, `fig_cuadro_vs_video.pdf`, `fig_duracion_clips.pdf`, `fig_duracion_por_senante.pdf`, `fig_particiones.pdf` | 7, 9, 12, 18, A7 |
| (a mano) | `diagrama_eda_datos.tex` (TikZ, se incluye con `\input`) | 6 |
| `guiones/prueba_escritorio.py` | `escritorio_a_…` a `escritorio_e_…` y `prueba_escritorio.json` | 8, 10, 11, 13-15, 21, 22, 24 |
| `guiones/figura_esqueleto.py` | solo `escritorio_a_esqueleto_<muestra>.pdf` (misma función de dibujo; no carga el modelo) | 10 |
| `guiones/tabla_capas_modelo.py` | `tabla_capas_modelo.json` (debe correr antes que el siguiente) | 16 (fuente), A1 |
| `guiones/diagrama_arquitectura.py` | `arquitectura_modelo.pdf` (diagrama con parámetros por bloque; la diapositiva 16 usa un diagrama TikZ más legible con las mismas formas) | 16 (fuente) |
| `guiones/curvas_aprendizaje.py` | `curvas_aprendizaje_semilla2026.pdf` y `.json` | 20 |
| `guiones/pasada_modelo_un_video.py` | `pasada_modelo_0000_0003_0002.pdf` y `.json` | 23 |
| `guiones/comparacion_red_y_representacion.py` | `comparacion_red_y_representacion.pdf` | A3 |
| `guiones/resultados_metricas.py` | `res_metricas.json` (recalcula las métricas y las coteja con lo guardado) | 25-33 |
| `guiones/figuras_resultados.py` | `res_esquemas_particion.pdf`, `res_top1_por_persona.pdf`, `res_red_vs_linea_base.pdf`, `res_recall_por_sena.pdf`, `res_matriz_confusion_15.pdf`, `res_matriz_confusion_50.pdf`, `res_umbral.pdf` | 25, 27, 29-31, 33, A6 |
| `guiones/tabla_por_sena.py` | `tabla_por_sena_1.tex`, `tabla_por_sena_2.tex` (desde `res_metricas.json`) | A4, A5 |

Orden para regenerar todo:

```sh
PY=python
G=docs/diapositivas_workflow/guiones
F=docs/diapositivas_workflow/figuras
$PY $G/figuras_datos.py
$PY $G/tabla_capas_modelo.py && $PY $G/diagrama_arquitectura.py
$PY $G/curvas_aprendizaje.py
$PY $G/pasada_modelo_un_video.py 0000_0003_0002
$PY $G/comparacion_red_y_representacion.py
$PY $G/resultados_metricas.py $F/res_metricas.json && $PY $G/figuras_resultados.py $F/res_metricas.json
$PY $G/tabla_por_sena.py
$PY $G/prueba_escritorio.py 0030_0000_0000 0030_0000_0003 0033_0000_0001 \
    --figuras $F --json $F/prueba_escritorio.json          # 1-2 min
```

`corridas_reloj_real/` guarda los registros de las 3 corridas del 30-sep-2026 que cita la diapositiva 24
(confianza del prototipo con el reloj real). Esa parte no es determinista: cada corrida de
`prueba_escritorio.py` reescribe `prueba_escritorio.json` con valores algo distintos en `prototipo.reloj_real`
(y en los tiempos); todo lo demás del JSON se repite exacto.

`res_latencia_2026-09-22.json` se obtuvo con `src/latencia_resumen.py` sobre
`registros/latencia_2026-09-22.jsonl` (N-52).

## Correr la prueba de escritorio

Sigue un video del corpus por el camino real del sistema y muestra en cada etapa qué entra y qué
sale, con la forma de los datos. No reimplementa nada: importa las funciones del repositorio.

```sh
# «hola», señante 0000 (el video más largo del corpus: 135 cuadros)
$PY docs/diapositivas_workflow/guiones/prueba_escritorio.py 0030_0000_0000

# otra muestra, sin el cotejo con el prototipo (más rápido para mostrarlo en vivo)
$PY docs/diapositivas_workflow/guiones/prueba_escritorio.py 0033_0000_0001 --sin-prototipo

# la red por dentro con un video del conjunto de PRUEBA («gracias», señante 0003)
$PY docs/diapositivas_workflow/guiones/pasada_modelo_un_video.py 0000_0003_0002
```

Aviso al mostrarla: 0030_0000_0000 se usó para entrenar el modelo canónico, así que su predicción
(«hola», 0,902) prueba que el camino funciona, no que el modelo acierte. La exactitud con personas
nuevas es la del LOSO (0,580 ± 0,086, N-30).

El prototipo en vivo se lanza con `python src/realtime.py` desde Terminal.app (desde el terminal del editor,
macOS niega la cámara; N-26). La tecla ESPACIO abre la ventana de 2,5 s.
