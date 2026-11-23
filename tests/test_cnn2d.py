"""
Pruebas de src/arquitectura_cnn2d.py y del criterio de src/experimento_cnn2d.py (docs/EXPERIMENTO_CNN2D.md).
Construyen modelos (TensorFlow) pero no entrenan nada.
Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_cnn2d -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf  # noqa: E402

from src import arquitectura_cnn2d as ac  # noqa: E402
from src import arquitecturas_literatura as al  # noqa: E402
from src import experimento_arquitecturas as ea  # noqa: E402
from src import experimento_cnn2d as ec  # noqa: E402
from src.representacion import espejar  # noqa: E402
from src.train_final_100 import add_velocity  # noqa: E402

VAL = [0.960, 0.937, 0.937, 0.973, 0.940]
LOSS = [0.90, 1.01, 1.05, 0.92, 1.01]


def _m(por_pliegue, ruido=(0.0, 0.0, 0.0)):
    base = np.asarray(por_pliegue, float)
    return np.stack([base + r for r in ruido])


def _criterio(val_c2d, loss_c2d=LOSS, val_t1=VAL, loss_t1=LOSS, ruido=(0.0, 0.0, 0.0), ruido_loss=(0.0, 0.0, 0.0)):
    val = {"T1": _m(val_t1, ruido), "C2D": _m(val_c2d, ruido)}
    loss = {"T1": _m(loss_t1, ruido_loss), "C2D": _m(loss_c2d, ruido_loss)}
    return ec.aplicar_criterio(val, loss)


class TestRemodelacion(unittest.TestCase):
    def setUp(self):
        self.X126 = np.random.default_rng(1).standard_normal((5, 30, 126))
        self.X = add_velocity(self.X126)

    def test_canales_y_columnas(self):
        im = ac.remodelar_a_imagen_np(self.X)
        self.assertEqual(im.shape, (5, 30, 21, 12))
        # Muñeca (punto 0) de la mano izquierda: x, y, z y vx, vy, vz en los canales 0-5.
        for c in range(3):
            np.testing.assert_array_equal(im[:, :, 0, c], self.X[:, :, c])
            np.testing.assert_array_equal(im[:, :, 0, 3 + c], self.X[:, :, 126 + c])
        # Punta del meñique (20) de la mano derecha: columnas 63 + 60 + c y 126 + 123 + c, canales 6-11.
        for c in range(3):
            np.testing.assert_array_equal(im[:, :, 20, 6 + c], self.X[:, :, 123 + c])
            np.testing.assert_array_equal(im[:, :, 20, 9 + c], self.X[:, :, 249 + c])
        # Cada una de las 252 columnas aparece exactamente una vez.
        Xid = np.broadcast_to(np.arange(252, dtype=float), (1, 30, 252))
        self.assertEqual(sorted(ac.remodelar_a_imagen_np(Xid)[0, 0].ravel().astype(int).tolist()), list(range(252)))

    def test_modelo_remodela_igual_que_numpy(self):
        m = ac.construir_cnn2d()
        im = ac.submodelo_imagen(m)(self.X.astype(np.float32)).numpy()
        np.testing.assert_array_equal(im, ac.remodelar_a_imagen_np(self.X).astype(np.float32))
        tf.keras.backend.clear_session()

    def test_espejo_intercambia_manos_y_niega_x(self):
        im = ac.remodelar_a_imagen_np(self.X)
        im_e = ac.remodelar_a_imagen_np(add_velocity(espejar(self.X126, "manos")))
        for mano in range(2):
            for v in range(2):
                for c in range(3):
                    signo = -1.0 if c == 0 else 1.0
                    np.testing.assert_allclose(im_e[..., ac.canal(mano, v, c)],
                                               signo * im[..., ac.canal(1 - mano, v, c)], atol=1e-12)

    def test_comprobar_remodelacion_detecta_orden_errado(self):
        """Una remodelación que pusiera las manos lado a lado en el eje de puntos no pasa la comprobación."""
        r = ec.comprobar_remodelacion(self.X126, add_velocity, lambda X: espejar(X, "manos"))
        self.assertTrue(r["espejo_intercambia_manos_y_niega_x_vx"])
        original = ac.remodelar_a_imagen_np
        try:
            ac.remodelar_a_imagen_np = lambda X: X.reshape(X.shape[0], 30, 2, 42, 3).transpose(
                0, 1, 3, 2, 4).reshape(X.shape[0], 30, 21, 12)
            with self.assertRaises(AssertionError):
                ec.comprobar_remodelacion(self.X126, add_velocity, lambda X: espejar(X, "manos"))
        finally:
            ac.remodelar_a_imagen_np = original
        tf.keras.backend.clear_session()


class TestModelo(unittest.TestCase):
    def test_parametros_forma_y_softmax(self):
        m = ac.construir_cnn2d(30, 252, 50)
        self.assertEqual(m.count_params(), ac.N_PARAM_C2D)
        self.assertTrue(100_000 <= m.count_params() <= 350_000)
        self.assertEqual(m.input_shape, (None, 30, 252))
        self.assertEqual(m.output_shape, (None, 50))
        p = m(np.random.default_rng(0).standard_normal((3, 30, 252)).astype(np.float32), training=False).numpy()
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-5)
        convs = [l for l in m.layers if isinstance(l, tf.keras.layers.Conv2D)]
        self.assertEqual([l.filters for l in convs], [32, 32, 64, 64, 64, 64])
        self.assertTrue(all(l.kernel_size == (3, 3) for l in convs))
        pools = [l.pool_size for l in m.layers if isinstance(l, tf.keras.layers.MaxPooling2D)]
        self.assertEqual(pools, [(2, 1), (2, 1)])           # solo tiempo, solo en los dos primeros bloques
        gap = [l for l in m.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)][0]
        self.assertEqual(tuple(gap.input.shape[1:]), (7, 21, 64))   # eje de puntos sin reducir
        tf.keras.backend.clear_session()

    def test_constructor_ampliado(self):
        self.assertIs(ea.constructor, ec.constructor)
        fn, hp = ec.constructor("C2D")
        self.assertEqual(hp, dict(ac.ENTRENAMIENTO_C2D, arquitectura=ac.CONFIG_C2D))
        self.assertEqual((hp["lr"], hp["batch"], hp["label_smoothing"]), (1e-3, 64, 0.1))
        fn_t, hp_t = ec.constructor("T")
        self.assertEqual(hp_t, dict(al.ENTRENAMIENTO_COMUN, arquitectura=al.CONFIG_T))
        self.assertEqual(fn_t(30, 252, 50).count_params(), 156_402)
        tf.keras.backend.clear_session()

    def test_guardar_y_cargar(self):
        import tempfile
        m = ac.construir_cnn2d()
        x = np.random.default_rng(2).standard_normal((2, 30, 252)).astype(np.float32)
        p = m(x, training=False).numpy()
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "m.keras"
            m.save(str(ruta))
            m2 = tf.keras.models.load_model(str(ruta))
            np.testing.assert_allclose(m2(x, training=False).numpy(), p, atol=1e-6)
        tf.keras.backend.clear_session()


class TestCriterio(unittest.TestCase):
    def test_empate_exacto(self):
        r = _criterio(VAL)
        self.assertEqual(r["resultado"], "empate")
        self.assertFalse(r["c2d_supera"])
        self.assertEqual(r["margen_top1"], 0.01)          # σ = 0 -> piso

    def test_c2d_gana_top1(self):
        r = _criterio([v + 0.02 for v in VAL])
        self.assertEqual((r["resultado"], r["decidido_por"]), ("C2D supera a T1", "top-1 de validacion"))
        self.assertTrue(r["c2d_supera"])
        self.assertIn("director", r["modelo_final"])

    def test_c2d_gana_media_pero_solo_en_3_pliegues(self):
        r = _criterio([VAL[0] + 0.05, VAL[1] + 0.05, VAL[2] + 0.05, VAL[3] - 0.01, VAL[4] - 0.01])
        self.assertFalse(r["regla_1_top1"]["C2D_gana"]["cumple"])
        self.assertTrue(r["regla_1_top1"]["empate"])

    def test_margen_exacto_no_basta(self):
        r = _criterio([v + 0.01 for v in VAL])            # Δ = 0,01 = margen: no es «mayor que»
        self.assertFalse(r["regla_1_top1"]["C2D_gana"]["cumple"])

    def test_c2d_pierde_top1(self):
        r = _criterio([v - 0.03 for v in VAL])
        self.assertEqual((r["resultado"], r["decidido_por"]), ("C2D pierde frente a T1", "top-1 de validacion"))
        self.assertFalse(r["regla_2_val_loss"]["aplica"])

    def test_desempate_por_val_loss(self):
        ruido_loss = (0.0, 0.01, -0.01)                    # σ_loss = 0,01
        r = _criterio(VAL, loss_c2d=[x - 0.03 for x in LOSS], ruido_loss=ruido_loss)
        self.assertAlmostEqual(r["margen_val_loss"], 0.01, places=9)
        self.assertEqual((r["resultado"], r["decidido_por"]),
                         ("C2D supera a T1", "empate en top-1; desempata la val_loss"))
        r = _criterio(VAL, loss_c2d=[x + 0.03 for x in LOSS], ruido_loss=ruido_loss)
        self.assertEqual(r["resultado"], "C2D pierde frente a T1")
        r = _criterio(VAL, loss_c2d=[x - 0.005 for x in LOSS], ruido_loss=ruido_loss)   # menor que σ_loss
        self.assertEqual(r["resultado"], "empate")

    def test_val_loss_no_decide_si_top1_decide(self):
        r = _criterio([v - 0.03 for v in VAL], loss_c2d=[x - 0.5 for x in LOSS])
        self.assertEqual(r["resultado"], "C2D pierde frente a T1")

    def test_sigma_agrupada_sobre_los_dos_brazos(self):
        val = {"T1": _m(VAL, (0.0, 0.02, -0.02)), "C2D": _m(VAL, (0.0, 0.0, 0.0))}
        r = ec.aplicar_criterio(val, {"T1": _m(LOSS), "C2D": _m(LOSS)})
        self.assertAlmostEqual(r["sigma_top1_entre_semillas_agrupada"], np.sqrt(0.02 ** 2 / 2), places=9)
        self.assertGreater(r["margen_top1"], 0.0141)                # sqrt(0,0002) = 0,0141 > piso de 0,01


class TestMcNemar(unittest.TestCase):
    def test_nombres(self):
        pa = {"a": (0, 0, .9), "b": (1, 1, .9), "c": (2, 0, .9), "d": (3, 3, .9)}   # C2D
        pb = {"a": (0, 0, .9), "b": (1, 0, .9), "c": (2, 2, .9), "d": (3, 0, .9)}   # T1
        r = ec.mcnemar_t1(pa, pb)
        self.assertEqual((r["aciertos_C2D"], r["aciertos_T1"]), (3, 2))
        self.assertEqual((r["solo_C2D_acierta"], r["solo_T1_acierta"], r["ambos_aciertan"]), (2, 1, 1))
        self.assertAlmostEqual(r["p_exacto_bilateral"], 1.0)


if __name__ == "__main__":
    unittest.main()
