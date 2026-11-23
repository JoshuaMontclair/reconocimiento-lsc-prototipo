"""
Representaciones de entrada del modelo (actividad 2.a; decisión D-08 en DECISIONES_TESIS.txt).

Un único lugar define:
  · el diseño de columnas del processed (preprocess_lsc50.py): postura, mano izquierda, mano
    derecha y, si se pidió --rostro cejas_labios, un bloque de 60 puntos faciales;
  · los tres candidatos que compara D-08 (manos; manos + postura; manos + postura + cejas y labios)
    y qué columnas toma cada uno;
  · el espejado horizontal de la aumentación, generalizado a cada candidato.

Diseño de columnas del processed (x, y, z por punto, normalizado por hombros):
    postura     0:99     33 puntos de PoseLandmarker
    mano_izq    99:162   21 puntos (mano izquierda DE LA PERSONA)
    mano_der    162:225  21 puntos
    rostro      225:405  60 puntos de la malla facial (solo si el processed se generó con rostro)

Subconjunto facial «cejas_labios» (D-08): cejas (MediaPipe FACE_LANDMARKS_RIGHT/LEFT_EYEBROW, 10 + 10)
y labios (FACE_LANDMARKS_LIPS, 40: 18 + 18 simétricos y 4 sobre la línea media). El orden de las
listas es el del bloque. Los pares simétricos se verificaron contra el modelo canónico de la malla
facial de MediaPipe (canonical_face_model.obj): cada índice de CEJA_A es el reflejo exacto x → −x del
índice en la misma posición de CEJA_B, e igual LABIO_A/LABIO_B; LABIO_CENTRO tiene x = 0.
tests/test_representacion.py lo comprueba con las coordenadas canónicas guardadas en tests/datos/.

Espejado: niega todas las x (el processed está centrado en el punto medio de los hombros, así que
x = 0 es el eje del cuerpo), intercambia los pares simétricos de la postura (los mismos
BODY_SYMMETRIC_PAIRS de train_lstm.augment_mirror), intercambia mano izquierda y derecha, e
intercambia los pares simétricos del rostro. Para «manos_postura» coincide exactamente con
train_lstm.augment_mirror (se prueba); para los otros candidatos es la misma operación sobre sus
bloques.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

COORDS = 3
N_POSTURA = 33
N_MANO = 21

# Cejas: CEJA_A = FACE_LANDMARKS_RIGHT_EYEBROW, CEJA_B = FACE_LANDMARKS_LEFT_EYEBROW (MediaPipe 0.10.35).
CEJA_A = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
CEJA_B = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336]
# Labios: LABIO_A[i] y LABIO_B[i] son simétricos; LABIO_CENTRO, sobre la línea media.
LABIO_A = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
LABIO_B = [314, 405, 321, 375, 291, 409, 270, 269, 267, 317, 402, 318, 324, 308, 415, 310, 311, 312]
LABIO_CENTRO = [0, 13, 14, 17]

ROSTRO_SUBCONJUNTOS: dict[str, list[int]] = {
    "cejas_labios": CEJA_A + CEJA_B + LABIO_A + LABIO_B + LABIO_CENTRO,   # 60 puntos
}
ROSTRO_OPCIONES = ("ninguno",) + tuple(ROSTRO_SUBCONJUNTOS)
PARES_ROSTRO = list(zip(CEJA_A, CEJA_B)) + list(zip(LABIO_A, LABIO_B))

# Pares simétricos de la postura: los de train_lstm.augment_mirror (misma lista, copiada para no
# importar TensorFlow desde aquí; test_representacion comprueba que son idénticas).
PARES_POSTURA = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
                 (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

REPRESENTACIONES: dict[str, tuple[str, ...]] = {
    "manos": ("mano_izq", "mano_der"),
    "manos_postura": ("postura", "mano_izq", "mano_der"),
    "manos_postura_rostro": ("postura", "mano_izq", "mano_der", "rostro"),
}
REPRESENTACION_POR_DEFECTO = "manos_postura"


def puntos_del_bloque(bloque: str, rostro: str = "cejas_labios") -> list[int]:
    """Identificadores de punto de un bloque, en el orden de sus columnas."""
    if bloque == "postura":
        return list(range(N_POSTURA))
    if bloque in ("mano_izq", "mano_der"):
        return list(range(N_MANO))
    if bloque == "rostro":
        return list(ROSTRO_SUBCONJUNTOS[rostro])
    raise ValueError(f"bloque desconocido: {bloque}")


def diseno_processed(rostro: str = "ninguno") -> list[tuple[str, int, int]]:
    """Bloques (nombre, columna inicial, columna final exclusiva) del X_padded que escribe
    preprocess_lsc50.py con --rostro <rostro>."""
    bloques, col = [], 0
    for nombre in ("postura", "mano_izq", "mano_der") + (("rostro",) if rostro != "ninguno" else ()):
        n = len(puntos_del_bloque(nombre, rostro if rostro != "ninguno" else "cejas_labios")) * COORDS
        bloques.append((nombre, col, col + n))
        col += n
    return bloques


def columnas(representacion: str, rostro_processed: str = "ninguno") -> np.ndarray:
    """Índices de columna del processed que toma una representación. Falla si la representación
    pide el rostro y el processed no lo tiene."""
    if representacion not in REPRESENTACIONES:
        raise ValueError(f"representación desconocida: {representacion!r}; opciones {list(REPRESENTACIONES)}")
    bloques = {n: (a, b) for n, a, b in diseno_processed(rostro_processed)}
    faltan = [b for b in REPRESENTACIONES[representacion] if b not in bloques]
    if faltan:
        raise ValueError(f"la representación {representacion!r} necesita {faltan} y el processed no los "
                         f"tiene (generarlo con preprocess_lsc50.py --rostro cejas_labios)")
    return np.concatenate([np.arange(*bloques[b]) for b in REPRESENTACIONES[representacion]])


def n_rasgos(representacion: str) -> int:
    """Rasgos por paso de una representación (sin velocidades)."""
    return sum(len(puntos_del_bloque(b)) * COORDS for b in REPRESENTACIONES[representacion])


def permutacion_espejo(representacion: str, rostro: str = "cejas_labios") -> np.ndarray:
    """Permutación de columnas del espejado: X_espejo[..., c] = X[..., perm[c]] (antes de negar x)."""
    puntos = [(b, p) for b in REPRESENTACIONES[representacion] for p in puntos_del_bloque(b, rostro)]
    posicion = {pt: i for i, pt in enumerate(puntos)}
    par_postura = {**dict(PARES_POSTURA), **{r: l for l, r in PARES_POSTURA}}
    par_rostro = {**dict(PARES_ROSTRO), **{r: l for l, r in PARES_ROSTRO}}
    perm = np.empty(len(puntos) * COORDS, dtype=np.int64)
    for i, (b, p) in enumerate(puntos):
        if b == "postura":
            espejo = (b, par_postura.get(p, p))
        elif b == "mano_izq":
            espejo = ("mano_der", p)
        elif b == "mano_der":
            espejo = ("mano_izq", p)
        else:
            espejo = (b, par_rostro.get(p, p))
        j = posicion[espejo]
        perm[i * COORDS:(i + 1) * COORDS] = np.arange(j * COORDS, (j + 1) * COORDS)
    return perm


def espejar(X: np.ndarray, representacion: str) -> np.ndarray:
    """Espejado horizontal de X (N, T, F) con F = n_rasgos(representacion)."""
    perm = permutacion_espejo(representacion)
    if X.shape[-1] != perm.size:
        raise ValueError(f"X tiene {X.shape[-1]} rasgos y la representación {representacion!r} tiene {perm.size}")
    Xm = X[..., perm].copy()
    Xm[..., 0::COORDS] *= -1
    return Xm


# ---------------------------------------------------------------------------
# Procedencia del modelo (N-24, N-30): qué representación y qué preprocesamiento espera cada .keras
# ---------------------------------------------------------------------------

def ruta_procedencia(ruta_modelo) -> Path:
    """modelo.keras -> modelo.procedencia.json (junto al modelo)."""
    ruta_modelo = Path(ruta_modelo)
    return ruta_modelo.with_name(ruta_modelo.stem + ".procedencia.json")


def escribir_procedencia(ruta_modelo, processed_dir, representacion: str, seq_length: int,
                         script: str, raiz=None) -> Path:
    """Escribe junto al modelo lo que el prototipo necesita para reproducir su entrada: representación,
    rasgos, velocidades, regla de mano ausente y subconjunto facial del processed con que se entrenó."""
    processed_dir = Path(processed_dir)
    meta = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    rel = (lambda p: str(Path(p).resolve().relative_to(Path(raiz).resolve()))
           if raiz and Path(p).resolve().is_relative_to(Path(raiz).resolve()) else str(p))
    n = n_rasgos(representacion)
    datos = {
        "modelo": Path(ruta_modelo).name,
        "representacion": representacion,
        "bloques": list(REPRESENTACIONES[representacion]),
        "n_rasgos_por_paso": n,
        "velocidades": True,
        "n_rasgos_entrada": 2 * n,
        "seq_length": int(seq_length),
        "normalizacion": "centrado en el punto medio de los hombros (postura 11 y 12) y escalado por su distancia en xy",
        "mano_ausente": meta.get("mano_ausente", {}).get("modo", "no_aplica"),
        "rostro_del_processed": meta.get("rostro", {}).get("subconjunto", "ninguno"),
        "processed_dir": rel(processed_dir),
        "landmarks_dir": meta.get("landmarks_dir", "data/raw/LSC50/LANDMARKS (CSV publicados)"),
        "script": script,
        "fecha": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    destino = ruta_procedencia(ruta_modelo)
    destino.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")
    return destino


def leer_procedencia(ruta_modelo) -> dict | None:
    """Procedencia declarada junto al modelo, o None si el modelo no la tiene (modelos anteriores al 15-sep)."""
    destino = ruta_procedencia(ruta_modelo)
    if not destino.exists():
        return None
    return json.loads(destino.read_text(encoding="utf-8"))


def leer_umbral(ruta_modelo) -> dict | None:
    """Umbral de confianza elegido para el modelo (D-09): <modelo>.umbral.json, o None si no lo tiene."""
    ruta_modelo = Path(ruta_modelo)
    destino = ruta_modelo.with_name(ruta_modelo.stem + ".umbral.json")
    if not destino.exists():
        return None
    datos = json.loads(destino.read_text(encoding="utf-8"))
    datos["ruta"] = str(destino)
    return datos
