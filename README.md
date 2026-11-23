# Reconocimiento de 50 señas de la Lengua de Señas Colombiana con salida en voz

Código del trabajo de grado *«Prototipo experimental para el reconocimiento de un vocabulario limitado de
la Lengua de Señas Colombiana con salida en audio en español, basado en MediaPipe Holistic y redes
convolucionales temporales, bajo condiciones controladas»*, de Joshua Emmanuel Montclair Ospina
(Ingeniería de Sistemas y Computación, Pontificia Universidad Javeriana Cali, 2026). Director: Felipe Palta.

El prototipo mira a una persona con la cámara del computador, reconoce cuál de 50 señas aisladas de la
LSC hizo y la dice en voz alta en español:

```
cámara → MediaPipe Holistic (puntos del cuerpo y las manos) → normalización respecto a los hombros
       → imputación de la mano no detectada → 30 pasos × 252 rasgos → red convolucional temporal (TCN)
       → umbral de confianza → conversión de glosas en un enunciado en español → voz
```

Todo corre en el computador, sin GPU dedicada y sin conexión a internet.

## Resultados principales

Corpus público LSC50: 997 videos válidos de 5 señantes.

| Evaluación | Modelo final (TCN + imputación) | Red de partida (Conv1D + BiLSTM + atención) |
|---|---|---|
| Parámetros | 156.402 | 714.291 |
| LOSO, señante no visto: top-1 | **67,7 %** (IC 95 % por señantes: 57,1–77,0 %) | 58,9 % |
| LOSO: top-3 | 84,7 % | 76,4 % |
| LOSO: F1 macro | 0,647 | 0,544 |
| Partición aleatoria, señantes vistos: top-1 | 94 % | 88 % |
| Latencia de respuesta en vivo (mediana) | 32,5 ms después de cerrar la ventana de 2,5 s | ≈ 36 ms |

LOSO deja fuera a un señante en cada uno de los 5 pliegues. Las cifras son la media de 3 semillas por
pliegue. Todas las decisiones de modelo se tomaron solo con la validación, con criterios fijados antes de
cada experimento; la prueba es descriptiva. El detalle está en `docs/` y el registro de decisiones en
`DECISIONES_TESIS.txt`.

## Estructura

| Carpeta | Contenido |
|---|---|
| `src/` | Extracción, preprocesamiento, entrenamiento, evaluación, experimentos, prototipo en vivo, conversor de glosas y voz |
| `tests/` | Pruebas automáticas (`unittest`; `tests/test_arquitecturas_2.py` usa `pytest`) |
| `scripts/` | Utilidades: visor de las señas del corpus, tablas de salidas del conversor |
| `data/partitions/` | Índices de las particiones: aleatoria 80/10/10 y LOSO de 5 pliegues |
| `data/` (resto) | Metadatos y caracterización del corpus; **sin videos ni secuencias de puntos** |
| `models/` | Resultados de cada experimento (JSON, registros, figuras) y los pesos publicados |
| `models/tcn_imputacion/modelo.keras` | **Pesos del modelo final** (TCN + imputación, D-14) |
| `models/canonico_holistic_mascara_manos/modelo_canonico.keras` | Pesos de la red de partida |
| `models/voz_pregrabada/` | Los 142 enunciados del prototipo, pregrabados (WAV) con su índice |
| `registros/` | Registros de las sesiones en vivo del autor (tiempos por etapa; sin imágenes) |
| `docs/` | Informes de cada experimento, protocolo de evaluación, especificación del conversor |
| `DECISIONES_TESIS.txt` | Registro de decisiones (D-xx) y notas (N-xx) del proyecto |
| `legacy/` | Código de la exploración preliminar de marzo de 2026 (solo referencia) |

## Instalación

Probado en macOS (Apple Silicon) con Python 3.12. MediaPipe 0.10 admite Python 3.9 a 3.12.

```bash
python3.12 -m venv ~/venvs/lsc
source ~/venvs/lsc/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Opcional, para la voz de respaldo (XTTS-v2), en un entorno aparte:

```bash
python3.12 -m venv ~/venvs/lsc-voz
~/venvs/lsc-voz/bin/pip install -r requirements-voz.txt
```

El modelo de MediaPipe Holistic (`holistic_landmarker.task`) no se incluye: el extractor lo descarga
de los servidores de MediaPipe la primera vez y lo guarda en `models/mediapipe/`
(ver `src/reextract_lsc50.py`).

## Datos: LSC50

Este repositorio **no redistribuye el corpus**: no contiene videos, imágenes ni secuencias de puntos.
Para reproducir el entrenamiento hay que descargar LSC50 y extraerlo en `data/raw/LSC50/`.

- Datos (figshare): https://doi.org/10.6084/m9.figshare.27383016.v1 — licencia **CC BY 4.0**.
- Artículo: A. F. Flórez-Sierra et al., «LSC50: Colombian Sign Language video and inertial measurement
  dataset», *Scientific Data* 11, 1347 (2024). https://doi.org/10.1038/s41597-024-04172-5

## Uso

**Prototipo en vivo.** Por defecto usa el modelo final (umbral 0,68), el conversor por inventario,
1280 × 720 y la voz pregrabada:

```bash
python src/realtime.py                       # una seña por ventana
python src/realtime.py --modo-oracion        # varias señas por turno → un enunciado
python src/realtime.py --python-voz ~/venvs/lsc-voz/bin/python   # con XTTS-v2 como respaldo
```

En macOS la cámara debe autorizarse para la aplicación de terminal desde la que se lanza.

**Reproducir los datos y el modelo** (con LSC50 en `data/raw/LSC50/`):

```bash
python src/reextract_lsc50.py                  # puntos característicos de los 1.000 videos
python src/preprocess_lsc50.py --landmarks data/extracted/LSC50_holistic \
       --salida data/processed_holistic_mascara_rostro
python src/train_loso.py --dry-run             # comprueba datos y particiones
python src/experimento_factorial_tcn.py entrenar --semillas 2026   # evaluación LOSO de la TCN
python src/train_final_tcn.py                  # modelo final sobre la partición aleatoria
```

Cada informe de `docs/` termina con los comandos exactos con que se obtuvieron sus cifras.

**Pruebas:**

```bash
python -m unittest discover -s tests
python -m pytest tests/test_arquitecturas_2.py
```

## Voz

Los 142 enunciados que dice el prototipo en modo inventario están pregrabados en
`models/voz_pregrabada/` con la voz «Robert - Colombian, Paisa» de ElevenLabs (generados con un plan de
pago que permite usarlos y distribuirlos) y se reproducen desde el disco. Cualquier otra frase la dice
XTTS-v2 (hablante «Gilberto Mathias») en un proceso aparte, si se indica `--python-voz`; si no, la voz
local de macOS. Detalles, comparación de voces y latencias en `docs/VOZ.md`.

## Licencia

El código se publica bajo la licencia MIT (`LICENSE`). El corpus LSC50 pertenece a sus autores y se
rige por su propia licencia (CC BY 4.0); este repositorio no lo incluye. Los audios de
`models/voz_pregrabada/` se generaron con ElevenLabs bajo un plan de pago. El modelo XTTS-v2 se rige por
la Coqui Public Model License (uso no comercial) y no se incluye: se descarga al instalar
`requirements-voz.txt`.

## Cómo citar

J. E. Montclair Ospina, *Prototipo experimental para el reconocimiento de un vocabulario limitado de la
Lengua de Señas Colombiana con salida en audio en español, basado en MediaPipe Holistic y redes
convolucionales temporales, bajo condiciones controladas*, trabajo de grado, Ingeniería de Sistemas y
Computación, Pontificia Universidad Javeriana Cali, 2026.
