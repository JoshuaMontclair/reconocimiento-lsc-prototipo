"""
Pruebas de las tres reglas de mano ausente (src/mano_ausente_variantes.py; docs/EXPERIMENTO_MANO_AUSENTE.md).

Casos sintéticos: hueco interior de 1 y de varios fotogramas, hueco al inicio, al final, mano nunca
vista; que «mascara» reproduce exactamente el preprocesamiento actual (train_loso.remuestrear_todas) y,
si el processed canónico está disponible, también sobre el corpus; y que las reglas conmutan con el
espejo de la aumentación.
Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_mano_ausente_variantes -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import mano_ausente_variantes as mav  # noqa: E402
from src.train_final_100 import resample_sequence  # noqa: E402

PROCESSED = ROOT_DIR / "data" / "processed_holistic_mascara_rostro"


def secuencia(T, huecos_izq=(), huecos_der=(), semilla=0):
    """Secuencia (T, 126) con trayectorias lineales en t (valor = base + pendiente·t) y huecos en 0."""
    rng = np.random.default_rng(semilla)
    base = rng.uniform(-2, 2, 126)
    pend = rng.uniform(-0.1, 0.1, 126)
    t = np.arange(T)[:, None]
    seq = (base + pend * t).astype(np.float32)
    seq[list(huecos_izq), 0:63] = 0.0
    seq[list(huecos_der), 63:126] = 0.0
    return seq, base, pend


class TestImputacion(unittest.TestCase):

    def test_hueco_de_un_fotograma_es_el_promedio(self):
        seq, _, _ = secuencia(40, huecos_izq=[10])
        out = mav.imputar_fotogramas(seq)
        np.testing.assert_allclose(out[10, :63], (seq[9, :63] + seq[11, :63]) / 2, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(out[:, 63:], seq[:, 63:])     # la otra mano no se toca
        mask = np.ones(40, bool); mask[10] = False
        np.testing.assert_array_equal(out[mask], seq[mask])         # los válidos no cambian

    def test_hueco_de_varios_fotogramas_es_lineal(self):
        seq, base, pend = secuencia(50, huecos_izq=[20, 21, 22, 23])
        out = mav.imputar_fotogramas(seq)
        t = np.arange(50)[:, None]
        esperado = (base + pend * t).astype(np.float32)
        # la trayectoria sintética es lineal: la imputación la recupera exactamente
        np.testing.assert_allclose(out[20:24, :63], esperado[20:24, :63], rtol=1e-5, atol=1e-5)
        w = 1 / 5
        np.testing.assert_allclose(out[20, :63], (1 - w) * seq[19, :63] + w * seq[24, :63], rtol=1e-5, atol=1e-5)

    def test_hueco_al_inicio_copia_el_primer_valido(self):
        seq, _, _ = secuencia(40, huecos_der=[0, 1, 2])
        out = mav.imputar_fotogramas(seq)
        for t in range(3):
            np.testing.assert_array_equal(out[t, 63:], seq[3, 63:])

    def test_hueco_al_final_copia_el_ultimo_valido(self):
        seq, _, _ = secuencia(40, huecos_izq=[37, 38, 39])
        out = mav.imputar_fotogramas(seq)
        for t in (37, 38, 39):
            np.testing.assert_array_equal(out[t, :63], seq[36, :63])

    def test_mano_nunca_vista_queda_en_ceros(self):
        seq, _, _ = secuencia(40, huecos_izq=range(40))
        out = mav.regla_imputacion(seq, 30)
        self.assertTrue(np.all(out[:, :63] == 0))
        self.assertTrue(np.all(out[:, 63:] != 0))

    def test_sin_huecos_es_identica_a_mascara(self):
        seq, _, _ = secuencia(45)
        np.testing.assert_array_equal(mav.regla_imputacion(seq, 30), mav.regla_mascara(seq, 30))

    def test_no_quedan_ceros_si_la_mano_se_vio(self):
        seq, _, _ = secuencia(60, huecos_izq=[0, 5, 6, 30, 59], huecos_der=[10, 11, 12, 13, 14])
        out = mav.regla_imputacion(seq, 30)
        self.assertFalse(np.any(np.all(out[:, :63] == 0, axis=1)))
        self.assertFalse(np.any(np.all(out[:, 63:] == 0, axis=1)))


class TestPivote(unittest.TestCase):

    def test_sin_huecos_usa_el_fotograma_nominal(self):
        T = 67
        idx = mav.indices_pivote(np.ones(T, bool), 30)
        p = np.linspace(0, T - 1, 30)
        np.testing.assert_array_equal(idx, np.floor(p + 0.5).astype(int))

    def test_hueco_de_un_fotograma_usa_un_vecino_real_no_usado(self):
        T = 67
        p = np.linspace(0, T - 1, 30)
        c = np.floor(p + 0.5).astype(int)
        det = np.ones(T, bool)
        det[c[10]] = False
        idx = mav.indices_pivote(det, 30)
        self.assertTrue(det[idx[10]])
        self.assertEqual(abs(idx[10] - c[10]), 1)
        self.assertNotIn(idx[10], set(np.delete(idx, 10)))         # no repite
        self.assertTrue(np.all(np.diff(idx) >= 0))                  # orden temporal

    def test_alterna_adelante_y_atras(self):
        # momento nominal 20 sin mano, 21 también, 19 libre y detectado: debe ir al más cercano a p
        T = 59   # p_j = 2 j exacto: c_j = 2j, los impares quedan libres
        det = np.ones(T, bool)
        det[20] = False
        idx = mav.indices_pivote(det, 30)
        self.assertEqual(idx[10], 21)                                # empate 19/21: primero adelante
        det[21] = False
        idx = mav.indices_pivote(det, 30)
        self.assertEqual(idx[10], 19)                                # adelante no detectado: atrás

    def test_hueco_largo_repite_extremos_y_conserva_orden(self):
        T = 40
        det = np.ones(T, bool)
        det[10:25] = False
        idx = mav.indices_pivote(det, 30)
        self.assertTrue(np.all(det[idx]))
        self.assertTrue(np.all(np.diff(idx) >= 0))
        self.assertTrue(set(idx) <= set(np.flatnonzero(det)))

    def test_hueco_largo_no_salta_al_otro_lado(self):
        # como «niña» 0015_0001_0000: hueco de 28 fotogramas (14-41, base 0) en un video de 74
        T = 74
        det = np.ones(T, bool)
        det[14:42] = False
        idx = mav.indices_pivote(det, 30)
        p = np.linspace(0, T - 1, 30)
        s = (T - 1) / 29
        validos = np.flatnonzero(det)
        for j in range(30):
            d_min = np.min(np.abs(validos - p[j]))
            self.assertLessEqual(abs(idx[j] - p[j]), d_min + s + 1e-9)
        dentro = (p > 14) & (p < 41)
        self.assertTrue(set(idx[dentro]) <= {13, 42})                # vecino más cercano a cada lado
        self.assertIn(13, idx[dentro])
        self.assertIn(42, idx[dentro])
        self.assertTrue(np.all(np.diff(idx) >= 0))

    def test_hueco_al_inicio_y_al_final(self):
        T = 50
        det = np.ones(T, bool)
        det[:6] = False
        det[-5:] = False
        idx = mav.indices_pivote(det, 30)
        self.assertTrue(np.all(det[idx]))
        self.assertEqual(idx[0], 6)
        self.assertEqual(idx[-1], 44)
        self.assertTrue(np.all(np.diff(idx) >= 0))

    def test_mano_nunca_vista(self):
        self.assertIsNone(mav.indices_pivote(np.zeros(40, bool), 30))
        seq, _, _ = secuencia(40, huecos_der=range(40))
        out = mav.regla_pivote(seq, 30)
        self.assertTrue(np.all(out[:, 63:] == 0))
        self.assertFalse(np.any(np.all(out[:, :63] == 0, axis=1)))

    def test_valores_son_fotogramas_reales(self):
        seq, _, _ = secuencia(55, huecos_izq=[3, 4, 17, 30, 31, 32], huecos_der=[0, 54])
        out, ind = mav.regla_pivote(seq, 30, devolver_indices=True)
        for j in range(30):
            np.testing.assert_array_equal(out[j, :63], seq[ind["mano_izq"][j], :63])
            np.testing.assert_array_equal(out[j, 63:], seq[ind["mano_der"][j], 63:])
        self.assertFalse(np.any(np.all(out[:, :63] == 0, axis=1)))


class TestMascaraYEspejo(unittest.TestCase):

    def test_mascara_es_resample_sequence(self):
        seq, _, _ = secuencia(70, huecos_izq=[21, 22, 25, 26])
        np.testing.assert_array_equal(mav.regla_mascara(seq, 30), resample_sequence(seq, 30))

    def test_reglas_conmutan_con_el_espejo(self):
        from src.representacion import espejar
        seq, _, _ = secuencia(63, huecos_izq=[1, 2, 40], huecos_der=[10, 11, 12, 62])
        espejo_fot = espejar(seq[None], "manos")[0]
        for regla in mav.REGLAS:
            a = espejar(mav.aplicar_regla(seq, regla, 30)[None], "manos")[0]
            b = mav.aplicar_regla(espejo_fot, regla, 30)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6, err_msg=regla)

    def test_regla_desconocida(self):
        with self.assertRaises(ValueError):
            mav.aplicar_regla(np.zeros((40, 126), np.float32), "otra")

    @unittest.skipUnless((PROCESSED / "X_padded.npy").exists(), "processed canónico no disponible")
    def test_mascara_reproduce_el_preprocesamiento_actual_del_corpus(self):
        from src import train_loso as tl
        X, y, L, nombres, _ = tl.cargar_datos(PROCESSED, "manos")
        ref = tl.remuestrear_todas(X, L, resample_sequence, 30)
        nuevo = mav.remuestrear_todas_con_regla(X, L, "mascara", 30)
        np.testing.assert_array_equal(nuevo, ref)
        # «hola» 0030_0000_0003: fotogramas 21, 22, 25, 26 (base 0) sin mano izquierda; pasos 9 y 11 en 0
        i = nombres.index("0030_0000_0003")
        sin_izq = np.flatnonzero(~mav.mano_detectada(X[i, :L[i]], mav.BLOQUES_MANOS["mano_izq"]))
        np.testing.assert_array_equal(sin_izq, [21, 22, 25, 26])
        np.testing.assert_array_equal(np.flatnonzero(np.all(ref[i, :, :63] == 0, axis=1)), [9, 11])
        for regla in ("pivote", "imputacion"):
            r = mav.aplicar_regla(X[i, :L[i]], regla, 30)
            self.assertFalse(np.any(np.all(r[:, :63] == 0, axis=1)), regla)


if __name__ == "__main__":
    unittest.main()
