"""
Pruebas de las velocidades reducidas (src/velocidades_reducidas.py; docs/EXPERIMENTO_VELOCIDADES_REDUCIDAS.md).

Comprueban que «actual» es exactamente add_velocity, que «director» (2) y «por_coordenada» (6) son las
medias pedidas, columna por columna, que las posiciones no cambian, y que espejar y LUEGO reducir da lo
esperado: en «por_coordenada» se intercambian las manos y vx cambia de signo; en «director» la media de
la otra mano con las x negadas. También el ejemplo de la cancelación (movimiento a la derecha y hacia
arriba de igual tamaño) que motivó la variante por coordenada.
Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_velocidades_reducidas -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import velocidades_reducidas as vr  # noqa: E402
from src.representacion import espejar  # noqa: E402

PROCESSED = ROOT_DIR / "data" / "processed_holistic_mascara_rostro"


def add_velocity_canonica(X):
    """Copia literal de train_final_100.add_velocity (para no importar TensorFlow en las pruebas)."""
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, vel], axis=-1)


def lote(n=4, T=30, semilla=0, huecos=True):
    rng = np.random.default_rng(semilla)
    X = rng.normal(0, 1, (n, T, 126)).astype(np.float32)
    if huecos:  # bloques de mano ausente en 0 (regla «mascara»)
        X[0, 5:8, 0:63] = 0.0
        X[1, 20:, 63:126] = 0.0
    return X


class PruebasFormas(unittest.TestCase):
    def test_actual_es_add_velocity(self):
        X = lote()
        np.testing.assert_array_equal(vr.entrada_red(X, "actual"), add_velocity_canonica(X))

    def test_add_velocity_del_repositorio(self):
        """Si TensorFlow está disponible, compara también con la función importada de train_final_100."""
        try:
            from src.train_final_100 import add_velocity
        except Exception as e:  # pragma: no cover
            self.skipTest(f"no se pudo importar train_final_100: {e}")
        X = lote(semilla=3)
        np.testing.assert_array_equal(vr.entrada_red(X, "actual"), add_velocity(X))

    def test_anchos_y_posiciones_intactas(self):
        X = lote()
        for forma, ancho in (("actual", 252), ("director", 128), ("por_coordenada", 132)):
            E = vr.entrada_red(X, forma)
            self.assertEqual(E.shape, (4, 30, ancho))
            np.testing.assert_array_equal(E[:, :, :126], X)
            np.testing.assert_array_equal(E[:, 0, 126:], 0.0)  # paso 1: velocidad 0
            self.assertEqual(len(vr.nombres_columnas_velocidad(forma)), ancho - 126)
            self.assertEqual(E.dtype, X.dtype)

    def test_director_columna_por_columna(self):
        X = lote(semilla=1)
        V = add_velocity_canonica(X)[:, :, 126:].astype(np.float64)
        E = vr.entrada_red(X, "director")
        for n in range(X.shape[0]):
            for t in range(X.shape[1]):
                self.assertAlmostEqual(E[n, t, 126], V[n, t, 0:63].sum() / 63, places=5)
                self.assertAlmostEqual(E[n, t, 127], V[n, t, 63:126].sum() / 63, places=5)

    def test_por_coordenada_columna_por_columna(self):
        X = lote(semilla=2)
        V = add_velocity_canonica(X)[:, :, 126:].astype(np.float64)
        E = vr.entrada_red(X, "por_coordenada")
        for n in range(X.shape[0]):
            for t in range(X.shape[1]):
                for m, base in ((0, 0), (1, 63)):
                    for c in range(3):
                        cols = [base + 3 * p + c for p in range(21)]
                        self.assertAlmostEqual(E[n, t, 126 + 3 * m + c], V[n, t, cols].mean(), places=5)

    def test_director_es_media_de_por_coordenada(self):
        X = lote(semilla=4)
        D = vr.entrada_red(X, "director")[:, :, 126:]
        C = vr.entrada_red(X, "por_coordenada")[:, :, 126:]
        np.testing.assert_allclose(D[..., 0], C[..., 0:3].mean(axis=-1), atol=1e-6)
        np.testing.assert_allclose(D[..., 1], C[..., 3:6].mean(axis=-1), atol=1e-6)

    def test_mano_que_se_traslada(self):
        """Si los 21 puntos se mueven igual (dx, dy, dz), C devuelve (dx, dy, dz) y B (dx+dy+dz)/3."""
        T = 5
        X = np.zeros((1, T, 126), dtype=np.float64)
        rng = np.random.default_rng(7)
        forma_mano = rng.normal(0, 0.3, 63)
        d = np.array([0.10, -0.04, 0.02])
        for t in range(T):
            X[0, t, 0:63] = forma_mano + np.tile(d * t, 21)
            X[0, t, 63:126] = forma_mano  # mano derecha quieta
        C = vr.entrada_red(X, "por_coordenada")[0, 1:, 126:]
        B = vr.entrada_red(X, "director")[0, 1:, 126:]
        np.testing.assert_allclose(C[:, 0:3], np.tile(d, (T - 1, 1)), atol=1e-12)
        np.testing.assert_allclose(C[:, 3:6], 0.0, atol=1e-12)
        np.testing.assert_allclose(B[:, 0], d.sum() / 3, atol=1e-12)
        np.testing.assert_allclose(B[:, 1], 0.0, atol=1e-12)

    def test_cancelacion_en_director(self):
        """La advertencia: +x y −y del mismo tamaño (en coordenadas de imagen de MediaPipe, y crece hacia abajo:
        hacia la derecha de la imagen y hacia arriba) se cancelan en B y no en C."""
        X = np.zeros((1, 3, 126))
        for t in range(3):
            X[0, t, 0:63:3] = 0.1 * t    # x de los 21 puntos de la mano izquierda
            X[0, t, 1:63:3] = -0.1 * t   # y
        B = vr.entrada_red(X, "director")[0, :, 126:]
        C = vr.entrada_red(X, "por_coordenada")[0, :, 126:]
        np.testing.assert_allclose(B, 0.0, atol=1e-12)
        np.testing.assert_allclose(C[1:, 0], 0.1, atol=1e-12)
        np.testing.assert_allclose(C[1:, 1], -0.1, atol=1e-12)

    def test_ancho_incorrecto(self):
        with self.assertRaises(ValueError):
            vr.entrada_red(np.zeros((1, 30, 252)), "director")
        with self.assertRaises(ValueError):
            vr.entrada_red(np.zeros((1, 30, 126)), "otra")


class PruebasEspejo(unittest.TestCase):
    """La aumentación espeja las 126 posiciones ANTES de calcular las velocidades (train_loso.aumentar y
    luego canon[2]). Se comprueba que espejar y luego reducir da lo esperado."""

    def test_por_coordenada_intercambia_manos_y_niega_vx(self):
        X = lote(semilla=5)
        R = vr.entrada_red(X, "por_coordenada")[:, :, 126:]
        Rm = vr.entrada_red(espejar(X, "manos"), "por_coordenada")[:, :, 126:]
        np.testing.assert_allclose(Rm, vr.espejo_esperado_reducido(R, "por_coordenada"), atol=1e-6)
        # explícito: izq tras el espejo = der antes, con vx negada
        np.testing.assert_allclose(Rm[..., 0], -R[..., 3], atol=1e-6)
        np.testing.assert_allclose(Rm[..., 1], R[..., 4], atol=1e-6)
        np.testing.assert_allclose(Rm[..., 2], R[..., 5], atol=1e-6)
        np.testing.assert_allclose(Rm[..., 3], -R[..., 0], atol=1e-6)
        np.testing.assert_allclose(Rm[..., 4], R[..., 1], atol=1e-6)
        np.testing.assert_allclose(Rm[..., 5], R[..., 2], atol=1e-6)

    def test_director_tras_espejo(self):
        X = lote(semilla=6).astype(np.float64)
        V = add_velocity_canonica(X)[:, :, 126:]
        Bm = vr.entrada_red(espejar(X, "manos"), "director")[:, :, 126:]
        der = V[:, :, 63:126]
        izq = V[:, :, 0:63]
        esperado_izq = (-der[..., 0::3].sum(-1) + der[..., 1::3].sum(-1) + der[..., 2::3].sum(-1)) / 63
        esperado_der = (-izq[..., 0::3].sum(-1) + izq[..., 1::3].sum(-1) + izq[..., 2::3].sum(-1)) / 63
        np.testing.assert_allclose(Bm[..., 0], esperado_izq, atol=1e-10)
        np.testing.assert_allclose(Bm[..., 1], esperado_der, atol=1e-10)
        self.assertIsNone(vr.espejo_esperado_reducido(Bm, "director"))

    def test_director_no_es_solo_intercambio(self):
        """Prueba de que reducir ANTES del espejo y luego intercambiar las 2 columnas sería incorrecto."""
        X = lote(semilla=8).astype(np.float64)
        B = vr.entrada_red(X, "director")[:, :, 126:]
        Bm = vr.entrada_red(espejar(X, "manos"), "director")[:, :, 126:]
        self.assertFalse(np.allclose(Bm, B[..., ::-1]))

    def test_espejo_conmuta_con_entrada_actual(self):
        """Para la forma «actual»: espejar posiciones y luego add_velocity = add_velocity y luego espejar
        cada mitad (la velocidad es lineal)."""
        X = lote(semilla=9)
        E = vr.entrada_red(espejar(X, "manos"), "actual")
        A = vr.entrada_red(X, "actual")
        np.testing.assert_allclose(E[:, :, :126], espejar(A[:, :, :126], "manos"), atol=1e-6)
        np.testing.assert_allclose(E[:, :, 126:], espejar(A[:, :, 126:], "manos"), atol=1e-6)


@unittest.skipUnless((PROCESSED / "X_padded.npy").exists(), "processed canonico no disponible")
class PruebasCorpus(unittest.TestCase):
    def test_hola_y_corpus(self):
        from src import train_loso as tl
        from src.train_final_100 import resample_sequence
        X_padded, y, L, nombres, _ = tl.cargar_datos(PROCESSED, "manos")
        X = tl.remuestrear_todas(X_padded, L, resample_sequence, 30)
        for forma, ancho in (("director", 128), ("por_coordenada", 132)):
            E = vr.entrada_red(X, forma)
            self.assertEqual(E.shape, (X.shape[0], 30, ancho))
            self.assertTrue(np.all(np.isfinite(E)))
            Em = vr.entrada_red(espejar(X, "manos"), forma)
            if forma == "por_coordenada":
                np.testing.assert_allclose(Em[:, :, 126:], vr.espejo_esperado_reducido(E[:, :, 126:], forma),
                                           atol=1e-5)
        i = nombres.index("0030_0000_0003")
        self.assertEqual(vr.entrada_red(X[i:i + 1], "director").shape, (1, 30, 128))


if __name__ == "__main__":
    unittest.main()
