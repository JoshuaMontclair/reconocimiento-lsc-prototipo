"""Pruebas del grafo y de la remodelación del brazo G (src/arquitecturas_literatura.py). Solo numpy."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import arquitecturas_literatura as al  # noqa: E402
from src.representacion import columnas, espejar  # noqa: E402


def _add_velocity(X):
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, vel], axis=-1)


def test_adyacencia_dos_manos():
    A = al.adyacencia_dos_manos()
    assert A.shape == (42, 42)
    assert np.array_equal(A, A.T)
    assert np.all(np.diag(A) == 1)
    assert A[:21, 21:].sum() == 0 and A[21:, :21].sum() == 0          # manos separadas
    assert (A[:21, :21].sum() - 21) / 2 == 21                           # 21 aristas de MediaPipe
    assert np.array_equal(A[:21, :21], A[21:, 21:])                     # misma topología en las dos
    An = al.adyacencia_normalizada()
    assert np.allclose(An, An.T)
    assert np.all(np.linalg.eigvalsh(An) <= 1 + 1e-5)


def test_orden_columnas_manos():
    assert np.array_equal(columnas("manos"), np.arange(99, 225))


def test_remodelacion_nodos_y_canales():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 30, 126)).astype(np.float32)
    N = al.remodelar_a_nodos_np(_add_velocity(X))
    assert N.shape == (3, 30, 42, 6)
    # nodo 21*mano + punto, canales x, y, z
    assert np.array_equal(N[:, :, 5, 0:3], X[:, :, 15:18])            # mano izquierda, punto 5
    assert np.array_equal(N[:, :, 21 + 8, 0:3], X[:, :, 63 + 24:63 + 27])   # mano derecha, punto 8
    v = X[:, 1:, 15:18] - X[:, :-1, 15:18]
    assert np.allclose(N[:, 1:, 5, 3:6], v)


def test_espejo_conmuta_con_la_remodelacion():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 30, 126)).astype(np.float32)
    N = al.remodelar_a_nodos_np(_add_velocity(X))
    Nm = al.remodelar_a_nodos_np(_add_velocity(espejar(X, "manos")))
    swap = np.r_[np.arange(21, 42), np.arange(0, 21)]
    signo = np.array([-1, 1, 1, -1, 1, 1], dtype=np.float32)
    assert np.allclose(Nm, N[:, :, swap, :] * signo, atol=1e-6)
