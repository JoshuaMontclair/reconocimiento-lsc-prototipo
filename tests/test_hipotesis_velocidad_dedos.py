"""
Pruebas de la descomposición g / r (src/velocidad_dedos.py) y del criterio prerregistrado del experimento
explicativo (src/experimento_hipotesis_velocidad_dedos.py; docs/EXPERIMENTO_HIPOTESIS_VELOCIDAD_DEDOS.md).
Solo NumPy: no entrena ni importa TensorFlow.
Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_hipotesis_velocidad_dedos -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import velocidad_dedos as vd  # noqa: E402
from src import velocidades_reducidas as vr  # noqa: E402
from src import experimento_hipotesis_velocidad_dedos as eh  # noqa: E402
from src.representacion import espejar  # noqa: E402


def add_velocity_canonica(X):
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, vel], axis=-1)


def lote(n=4, T=30, semilla=0):
    rng = np.random.default_rng(semilla)
    X = rng.normal(0, 1, (n, T, 126)).astype(np.float32)
    X[0, 5:8, 0:63] = 0.0
    X[1, 20:, 63:126] = 0.0
    return X


class PruebasDescomposicion(unittest.TestCase):
    def test_actual_es_add_velocity(self):
        X = lote()
        np.testing.assert_array_equal(vd.entrada_red(X, "actual"), add_velocity_canonica(X))

    def test_anchos_posiciones_y_nombres(self):
        X = lote()
        for forma, ancho in (("actual", 252), ("g_r", 258), ("r", 252)):
            E = vd.entrada_red(X, forma)
            self.assertEqual(E.shape, (4, 30, ancho))
            np.testing.assert_array_equal(E[..., :126], X)
            np.testing.assert_array_equal(E[:, 0, 126:], 0.0)
            self.assertEqual(len(vd.nombres_columnas(forma)), ancho)
            self.assertEqual(E.dtype, X.dtype)
            self.assertEqual(vd.FORMAS[forma]["n_rasgos"], ancho)

    def test_g_es_por_coordenada_y_r_columna_por_columna(self):
        X = lote(semilla=1)
        V = add_velocity_canonica(X)[..., 126:].astype(np.float64)
        E = vd.entrada_red(X, "g_r")
        np.testing.assert_allclose(E[..., 126:132], vr.entrada_red(X, "por_coordenada")[..., 126:], atol=1e-6)
        for m in range(2):
            for c in range(3):
                g = V[..., m * 63 + c:(m + 1) * 63:3].mean(axis=-1)
                np.testing.assert_allclose(E[..., 126 + 3 * m + c], g, atol=1e-6)
                for p in range(21):
                    col = m * 63 + 3 * p + c
                    np.testing.assert_allclose(E[..., 132 + col], V[..., col] - g, atol=1e-6)

    def test_r_suma_cero_y_reconstruccion(self):
        X = lote(semilla=2)
        V = vd.velocidades(X)
        g, r = vd.global_y_relativa(V)
        np.testing.assert_allclose(vd.por_mano(r).sum(axis=-2), 0, atol=1e-5)
        np.testing.assert_allclose(vd.reconstruir_v(g, r), V, atol=1e-6)
        np.testing.assert_array_equal(vd.entrada_red(X, "r")[..., 126:], vd.entrada_red(X, "g_r")[..., 132:])

    def test_energia_pitagoras(self):
        V = vd.velocidades(lote(semilla=3))
        et, eg, er = vd.energias_por_paso(V)
        np.testing.assert_allclose(et, eg + er, rtol=1e-9, atol=1e-9)
        self.assertEqual(et.shape, (4, 30, 2))

    def test_solo_traslacion_no_tiene_r(self):
        """Mano rígida que se traslada: toda la energía es global."""
        X = np.zeros((1, 30, 126), np.float32)
        forma = np.random.default_rng(4).normal(0, 1, 63).astype(np.float32)
        for t in range(30):
            X[0, t, :63] = forma + np.tile([0.1 * t, -0.05 * t, 0.0], 21)
        g, r = vd.global_y_relativa(vd.velocidades(X))
        np.testing.assert_allclose(r, 0, atol=1e-5)
        np.testing.assert_allclose(g[0, 1:, 0:3], np.tile([0.1, -0.05, 0.0], (29, 1)), atol=1e-5)

    def test_abrir_la_mano_es_r(self):
        """Mano que se abre alrededor de su centro (escala): g = 0 y toda la energía es relativa."""
        X = np.zeros((1, 30, 126), np.float32)
        forma = np.random.default_rng(5).normal(0, 1, (21, 3))
        forma -= forma.mean(axis=0)
        for t in range(30):
            X[0, t, 63:] = ((1 + 0.02 * t) * forma).ravel()
        et, eg, er = vd.energias_por_paso(vd.velocidades(X))
        self.assertLess(eg.sum(), 1e-8)
        self.assertGreater(er.sum(), 0)


class PruebasRigida(unittest.TestCase):
    def test_giro_y_escala_sin_deformacion(self):
        rng = np.random.default_rng(9)
        forma = rng.normal(0, 1, (21, 3))
        X = np.zeros((1, 30, 126))
        for t in range(30):
            a = 0.01 * t
            Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            X[0, t, :63] = ((1 + 0.005 * t) * forma @ Rz.T + [0.1 * t, 0, 0]).ravel()
        e_r, e_gs, e_def = vd.descomponer_r_rigida(X)
        self.assertGreater(e_r[0, 1:, 0].sum(), 0)
        self.assertLess(e_def[0, 1:, 0].sum() / e_r[0, 1:, 0].sum(), 1e-3)

    def test_deformacion_pura(self):
        rng = np.random.default_rng(10)
        X = np.repeat(rng.normal(0, 1, (1, 1, 126)), 30, axis=1)
        X[0, 15:, 3 * 8 + 1] += 0.5          # un solo punto (la punta del índice) se mueve en y
        e_r, e_gs, e_def = vd.descomponer_r_rigida(X)
        self.assertGreater(e_def[0, 15, 0] / e_r[0, 15, 0], 0.5)
        np.testing.assert_allclose(e_r, e_gs + e_def, atol=1e-9)


class PruebasEspejo(unittest.TestCase):
    def test_espejo_g_y_r(self):
        X = lote(semilla=6)
        for forma in ("actual", "g_r", "r"):
            c = eh.comprobar_espejo(X, forma)
            self.assertLess(c["max_abs_dif"], 1e-6, forma)

    def test_gx_cambia_de_signo_y_manos_intercambiadas(self):
        X = lote(semilla=7)
        E = vd.entrada_red(X, "g_r")
        Em = vd.entrada_red(espejar(X, "manos"), "g_r")
        np.testing.assert_allclose(Em[..., 126], -E[..., 129], atol=1e-6)   # izq_vx espejo = −der_vx
        np.testing.assert_allclose(Em[..., 127], E[..., 130], atol=1e-6)    # izq_vy espejo = der_vy
        np.testing.assert_allclose(Em[..., 129], -E[..., 126], atol=1e-6)
        np.testing.assert_allclose(Em[..., 132 + 0], -E[..., 132 + 63], atol=1e-6)   # r izq p0 x = −r der p0 x


class PruebasTransicion(unittest.TestCase):
    def test_sin_huecos_o_nunca_detectada(self):
        for det in (np.ones(60, bool), np.zeros(60, bool)):
            tr, ar = vd.pasos_de_transicion(det, 30)
            self.assertFalse(tr.any())
            self.assertFalse(ar.any())

    def test_identidad_T30(self):
        det = np.ones(30, bool)
        det[10:13] = False
        tr, ar = vd.pasos_de_transicion(det, 30)
        # Con T = 30 cada paso es un fotograma: transiciones en los pasos 10 (sale) y 13 (vuelve).
        self.assertEqual(np.flatnonzero(tr).tolist(), [10, 13])
        self.assertEqual(np.flatnonzero(ar).tolist(), [10, 11, 12, 13])

    def test_remuestreo(self):
        det = np.ones(59, bool)
        det[20:24] = False          # p_j = 2 j: j = 10, 11 usan los fotogramas 20 y 22 (ausentes)
        tr, ar = vd.pasos_de_transicion(det, 30)
        self.assertEqual(np.flatnonzero(tr).tolist(), [10, 12])
        self.assertFalse(tr[0])

    def test_mascaras_por_video(self):
        Xp = np.ones((2, 40, 126), np.float32)
        Xp[0, 10:15, :63] = 0
        Xp[1, :, 63:] = 0
        tr, ar = vd.mascaras_transicion(Xp, [30, 40], 30)
        self.assertTrue(tr[0, :, 0].any())
        self.assertFalse(tr[0, :, 1].any())
        self.assertFalse(tr[1].any())     # mano derecha nunca detectada: sin transición


class PruebasSonda(unittest.TestCase):
    def test_rasgos(self):
        R = vd.rasgos_sonda(lote(semilla=8))
        self.assertEqual({k: v.shape[1] for k, v in R.items()}, {"g": 12, "r": 252, "r_12": 12, "g_r": 264, "v": 252})


BASE = [0.95, 0.94, 0.94, 0.96, 0.93]
LOSS = [0.90, 1.00, 1.05, 0.89, 1.06]


def _m(por_pliegue, ruido=(0.0, 0.01, -0.01)):
    base = np.asarray(por_pliegue, float)
    return np.stack([base + r for r in ruido])


def _datos(val_mod=None, loss_mod=None):
    val = {b: _m(BASE) for b in eh.BRAZOS}
    loss = {b: _m(LOSS) for b in eh.BRAZOS}
    for b, v in (val_mod or {}).items():
        val[b] = _m(v)
    for b, v in (loss_mod or {}).items():
        loss[b] = _m(v)
    return val, loss


SONDA_OK = {"r": {"test_top1_media": 0.30}, "g": {"test_top1_media": 0.40}, "r_12": {"test_top1_media": 0.2}}
SONDA_MAL = {"r": {"test_top1_media": 0.06}, "g": {"test_top1_media": 0.40}, "r_12": {"test_top1_media": 0.05}}


class PruebasCriterio(unittest.TestCase):
    def test_margenes(self):
        val, loss = _datos()
        m = eh.margenes(val, loss)
        self.assertAlmostEqual(m["sigma_top1"], 0.01, places=9)
        self.assertAlmostEqual(m["margen_top1"], 0.01, places=9)
        self.assertAlmostEqual(m["margen_val_loss"], 0.01, places=9)
        self.assertEqual(m["brazos"], list(eh.BRAZOS_SIGMA))

    def test_h1_confirmada(self):
        # H2 = T1 en top-1 y val_loss; T2 con val_loss mayor en los 5 pliegues.
        val, loss = _datos(loss_mod={"T2": [x + 0.02 for x in LOSS]})
        m = eh.margenes(val, loss)
        v = eh.veredicto_h1(val, loss, m, SONDA_OK)
        self.assertTrue(v["a_H2_no_peor_que_T1"] and v["b_H2_menor_val_loss_que_T2_en_4_de_5"])
        self.assertEqual(v["veredicto"], "CONFIRMADA")

    def test_h1_no_concluyente_por_la_sonda_o_por_b(self):
        val, loss = _datos(loss_mod={"T2": [x + 0.02 for x in LOSS]})
        m = eh.margenes(val, loss)
        self.assertEqual(eh.veredicto_h1(val, loss, m, SONDA_MAL)["veredicto"], "NO CONCLUYENTE")
        val, loss = _datos()     # H2 = T2 en todos los pliegues: (b) falla, nada es peor
        self.assertEqual(eh.veredicto_h1(val, loss, eh.margenes(val, loss), SONDA_OK)["veredicto"], "NO CONCLUYENTE")

    def test_h1_refutada(self):
        # H2 peor que T1 por más del margen en val_loss.
        val, loss = _datos(loss_mod={"H2": [x + 0.05 for x in LOSS]})
        v = eh.veredicto_h1(val, loss, eh.margenes(val, loss), SONDA_OK)
        self.assertTrue(v["H2_peor_que_T1_por_mas_del_margen"])
        self.assertEqual(v["veredicto"], "REFUTADA")
        # H2 peor que T2 en top-1.
        val, loss = _datos(val_mod={"H2": [x - 0.03 for x in BASE]})
        self.assertEqual(eh.veredicto_h1(val, loss, eh.margenes(val, loss), SONDA_OK)["veredicto"], "REFUTADA")

    def test_control_h1(self):
        val, loss = _datos()
        self.assertTrue(eh.control_h1(val, loss, eh.margenes(val, loss))["equivalente_a_T1"])
        val, loss = _datos(val_mod={"H1": [x - 0.03 for x in BASE]})
        self.assertFalse(eh.control_h1(val, loss, eh.margenes(val, loss))["equivalente_a_T1"])

    def test_h2(self):
        en = lambda fm, fi: {"mascara": {"ambas": {"fraccion_de_r_en_pasos_de_transicion": fm}},  # noqa: E731
                             "imputacion": {"ambas": {"fraccion_de_r_en_pasos_de_transicion": fi}}}
        val, loss = _datos()
        m = eh.margenes(val, loss)
        self.assertEqual(eh.veredicto_h2(en(0.7, 0.1), val, loss, m)["veredicto"], "CONFIRMADA")
        self.assertEqual(eh.veredicto_h2(en(0.3, 0.1), val, loss, m)["veredicto"], "REFUTADA")
        self.assertEqual(eh.veredicto_h2(en(0.7, 0.5), val, loss, m)["veredicto"], "NO CONCLUYENTE")
        val, loss = _datos(loss_mod={"H3": [x - 0.05 for x in LOSS]})
        v = eh.veredicto_h2(en(0.7, 0.1), val, loss, eh.margenes(val, loss))
        self.assertTrue(v["H3_supera_a_T3_por_mas_del_margen"])
        self.assertEqual(v["veredicto"], "REFUTADA")

    def test_rutas(self):
        self.assertIn("confirmacion_tcn_2026-10-03/T1_imputacion_252", str(eh.dir_corrida("x", "T1", 2026, 0)))
        self.assertIn("factorial_tcn_2026-10-03/T3_mascara_132", str(eh.dir_corrida("x", "T3", 2026, 0)))
        self.assertIn("arquitecturas_literatura_2026-10-02/T_tcn", str(eh.dir_corrida("x", "T0", 2026, 0)))
        self.assertEqual(str(eh.dir_corrida("/o", "H2", 2027, 3)), "/o/H2_imputacion_r_252/semilla_2027/pliegue_3")


if __name__ == "__main__":
    unittest.main()
