"""
Velocidades reducidas (pedido del director, reunión del 2-oct-2026, N-60 encargo 2).
Ver docs/EXPERIMENTO_VELOCIDADES_REDUCIDAS.md.

Entrada canónica de la red: 30 × 252 = 126 posiciones («manos», D-08) + 126 velocidades
(add_velocity de train_final_100: vel[t] = x[t] − x[t−1], vel[0] = 0).

Disposición de las 126 columnas de posición (src/representacion.py, representación «manos»):
  mano izquierda = columnas 0-62, mano derecha = 63-125; dentro de cada mano, 21 puntos × (x, y, z)
  intercalados: la columna 3·p + c es la coordenada c (0 = x, 1 = y, 2 = z) del punto p.

Formas de la entrada de la red (todas calculadas a partir de las posiciones YA remuestreadas y
aumentadas, como add_velocity, para que el espejo de la aumentación siga siendo correcto):
  «actual»          126 posiciones + 126 velocidades                                   = 252
  «director»        126 posiciones + 2  (por mano, media de sus 63 columnas de velocidad) = 128
                    orden: [izq, der]
  «por_coordenada»  126 posiciones + 6  (por mano, media de las 21 vx, de las 21 vy y de las 21 vz)
                    = 132; orden: [izq_vx, izq_vy, izq_vz, der_vx, der_vy, der_vz]

Solo NumPy: no importa TensorFlow.
"""

from __future__ import annotations

import numpy as np

COORDS = 3
N_PUNTOS_MANO = 21
ANCHO_MANO = N_PUNTOS_MANO * COORDS          # 63
N_POS = 2 * ANCHO_MANO                       # 126
MANOS = ("izq", "der")
EJES = ("x", "y", "z")

FORMAS = {
    "actual": {"n_vel": 126, "descripcion": "126 posiciones + 126 velocidades (canonica)"},
    "director": {"n_vel": 2, "descripcion": "126 posiciones + 2 velocidades (media de las 63 por mano)"},
    "por_coordenada": {"n_vel": 6, "descripcion": "126 posiciones + 6 velocidades (media de vx, vy, vz por mano)"},
}


def velocidades(X: np.ndarray) -> np.ndarray:
    """Velocidades completas, idénticas a la mitad derecha de train_final_100.add_velocity:
    vel[:, 0] = 0, vel[:, t] = X[:, t] − X[:, t−1]. X: (N, T, F) o (T, F)."""
    X = np.asarray(X)
    vel = np.zeros_like(X)
    vel[..., 1:, :] = X[..., 1:, :] - X[..., :-1, :]
    return vel


def _comprobar_ancho(V: np.ndarray):
    if V.shape[-1] != N_POS:
        raise ValueError(f"se esperaban {N_POS} columnas (representacion «manos»); hay {V.shape[-1]}")


def reducir_director(V: np.ndarray) -> np.ndarray:
    """(…, 126) → (…, 2): por cada mano, la media de sus 63 columnas de velocidad (x, y y z mezcladas)."""
    _comprobar_ancho(V)
    izq = V[..., :ANCHO_MANO].mean(axis=-1)
    der = V[..., ANCHO_MANO:].mean(axis=-1)
    return np.stack([izq, der], axis=-1).astype(V.dtype, copy=False)


def reducir_por_coordenada(V: np.ndarray) -> np.ndarray:
    """(…, 126) → (…, 6): por cada mano, la media de las 21 vx, de las 21 vy y de las 21 vz.
    Orden [izq_vx, izq_vy, izq_vz, der_vx, der_vy, der_vz]."""
    _comprobar_ancho(V)
    partes = []
    for m in range(2):
        bloque = V[..., m * ANCHO_MANO:(m + 1) * ANCHO_MANO]
        for c in range(COORDS):
            partes.append(bloque[..., c::COORDS].mean(axis=-1))
    return np.stack(partes, axis=-1).astype(V.dtype, copy=False)


REDUCTORES = {"director": reducir_director, "por_coordenada": reducir_por_coordenada}


def nombres_columnas_velocidad(forma: str) -> list[str]:
    if forma == "actual":
        return [f"v_{m}_p{p}_{e}" for m in MANOS for p in range(N_PUNTOS_MANO) for e in EJES]
    if forma == "director":
        return [f"v_{m}_media63" for m in MANOS]
    if forma == "por_coordenada":
        return [f"v_{m}_{e}_media21" for m in MANOS for e in EJES]
    raise ValueError(f"forma desconocida: {forma}")


def entrada_red(X: np.ndarray, forma: str) -> np.ndarray:
    """Entrada de la red a partir de las posiciones remuestreadas y aumentadas X (N, T, 126).
    «actual» es exactamente add_velocity(X)."""
    _comprobar_ancho(X)
    V = velocidades(X)
    if forma == "actual":
        return np.concatenate([X, V], axis=-1)
    if forma in REDUCTORES:
        return np.concatenate([X, REDUCTORES[forma](V)], axis=-1)
    raise ValueError(f"forma desconocida: {forma}")


def funcion_entrada(forma: str):
    """Función X -> entrada de la red, para sustituir add_velocity en la tupla canon de train_loso."""
    if forma not in FORMAS:
        raise ValueError(f"forma desconocida: {forma}")
    return lambda X: entrada_red(X, forma)


def espejo_esperado_reducido(R: np.ndarray, forma: str) -> np.ndarray | None:
    """Lo que debe dar reducir(vel(espejo(X))) expresado a partir de reducir(vel(X)), cuando existe:
      por_coordenada: se intercambian las manos y vx cambia de signo (vy, vz iguales).
      director: no existe como función de las 2 columnas (la media mezcla x con y y z: tras el espejo
                es (−Σvx + Σvy + Σvz)/63 de la otra mano); devuelve None."""
    if forma == "por_coordenada":
        izq, der = R[..., 0:3].copy(), R[..., 3:6].copy()
        izq[..., 0] *= -1
        der[..., 0] *= -1
        return np.concatenate([der, izq], axis=-1)
    if forma == "director":
        return None
    raise ValueError(f"forma desconocida: {forma}")
