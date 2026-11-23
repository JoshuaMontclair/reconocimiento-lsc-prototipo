"""
Pruebas del criterio y de las entradas del experimento de confirmación con la TCN
(src/experimento_confirmacion_tcn.py; docs/EXPERIMENTO_CONFIRMACION_TCN.md). Solo NumPy: no entrena ni
importa TensorFlow.
Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_confirmacion_tcn -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import experimento_confirmacion_tcn as ec  # noqa: E402
from src import velocidades_reducidas as vr  # noqa: E402
from src import mano_ausente_variantes as mav  # noqa: E402


def add_velocity_canonica(X):
    """Copia literal de train_final_100.add_velocity (para no importar TensorFlow)."""
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, vel], axis=-1)


def _val(por_pliegue, ruido=(0.0, 0.0, 0.0)):
    """(3 semillas, 5 pliegues) con la media por pliegue dada y un desplazamiento por semilla."""
    base = np.asarray(por_pliegue, float)
    return np.stack([base + r for r in ruido])


class TestCriterio(unittest.TestCase):
    T0 = [0.96, 0.94, 0.94, 0.97, 0.93]

    def test_margen_tiene_piso_de_001(self):
        c = ec.aplicar_criterio({"T0": _val(self.T0), "T1": _val(self.T0), "T2": _val(self.T0)})
        self.assertAlmostEqual(c["sigma_semilla_agrupada"], 0.0)
        self.assertEqual(c["margen"], 0.01)

    def test_sigma_agrupada_es_raiz_de_la_media_de_varianzas(self):
        r = (-0.01, 0.0, 0.01)
        c = ec.aplicar_criterio({"T0": _val(self.T0, r), "T1": _val(self.T0), "T2": _val(self.T0)})
        # varianza de T0 = 1e-4, las otras 0 -> sigma = sqrt(1e-4 / 3)
        self.assertAlmostEqual(c["sigma_semilla_agrupada"], np.sqrt(1e-4 / 3), places=9)

    def test_t1_reemplaza_si_mejora_mas_que_el_margen_en_4_de_5(self):
        t1 = np.array(self.T0) + [0.02, 0.02, 0.02, 0.02, -0.005]
        c = ec.aplicar_criterio({"T0": _val(self.T0), "T1": _val(t1), "T2": _val(t1)})
        self.assertTrue(c["paso1_T1_vs_T0"]["regla_mejora"]["cumple"])
        self.assertEqual(c["referencia_para_T2"], "T1")
        self.assertEqual(c["configuracion_final"], "T2")

    def test_t1_no_reemplaza_con_solo_3_pliegues_mejores(self):
        t1 = np.array(self.T0) + [0.05, 0.05, 0.05, -0.005, 0.0]
        c = ec.aplicar_criterio({"T0": _val(self.T0), "T1": _val(t1), "T2": _val(self.T0) - 0.05})
        self.assertFalse(c["paso1_T1_vs_T0"]["regla_mejora"]["cumple"])
        self.assertIn("neutra", c["paso1_T1_vs_T0"]["resultado"])
        self.assertEqual(c["referencia_para_T2"], "T0")
        self.assertEqual(c["configuracion_final"], "T0")

    def test_imputacion_peor(self):
        t1 = np.array(self.T0) - 0.03
        c = ec.aplicar_criterio({"T0": _val(self.T0), "T1": _val(t1), "T2": _val(self.T0)})
        self.assertIn("empeora", c["paso1_T1_vs_T0"]["resultado"])
        self.assertEqual(c["configuracion_final"], "T2")   # T2 no es peor que T0

    def test_no_inferioridad_tolera_un_pliegue_que_pierde(self):
        t2 = np.array(self.T0) + [-0.02, 0.0, 0.0, 0.0, 0.0]
        n = ec.regla_no_inferioridad(ec.deltas_por_pliegue(_val(t2), _val(self.T0)), 0.01)
        self.assertTrue(n["cumple"])
        t2 = np.array(self.T0) + [-0.02, -0.02, 0.01, 0.01, 0.01]
        n = ec.regla_no_inferioridad(ec.deltas_por_pliegue(_val(t2), _val(self.T0)), 0.01)
        self.assertFalse(n["cumple"])     # pierde por más del margen en 2 pliegues

    def test_delta_de_menos_margen_exacto_no_cuenta_como_perder(self):
        t2 = np.array(self.T0) - 0.01
        n = ec.regla_no_inferioridad(ec.deltas_por_pliegue(_val(t2), _val(self.T0)), 0.01)
        self.assertTrue(n["cumple"])
        self.assertEqual(n["pliegues_que_pierden_por_mas_del_margen"], 0)

    def test_delta_igual_al_margen_no_es_mejora(self):
        t1 = np.array(self.T0) + 0.01
        m = ec.regla_mejora(ec.deltas_por_pliegue(_val(t1), _val(self.T0)), 0.01)
        self.assertFalse(m["cumple"])


class TestVoto(unittest.TestCase):
    def test_mayoria_y_desempate_por_probabilidad(self):
        p = [{"a": (1, 1, 0.9), "b": (2, 3, 0.4), "c": (0, 5, 0.5)},
             {"a": (1, 1, 0.8), "b": (2, 2, 0.6), "c": (0, 6, 0.7)},
             {"a": (1, 4, 0.99), "b": (2, 2, 0.3), "c": (0, 7, 0.2)}]
        v = ec.voto_semillas(p)
        self.assertEqual(v["a"], (1, 1))
        self.assertEqual(v["b"], (2, 2))
        self.assertEqual(v["c"], (0, 6))      # las tres discrepan: la más confiada


class TestEntradas(unittest.TestCase):
    def setUp(self):
        self.canon = ("tf", "build", add_velocity_canonica, None, [])
        rng = np.random.default_rng(0)
        self.X = rng.normal(0, 1, (3, 30, 126)).astype(np.float32)

    def test_t0_y_t1_usan_add_velocity(self):
        for b in ("T0", "T1"):
            self.assertIs(ec.canon_de_brazo(self.canon, b)[2], add_velocity_canonica)

    def test_t2_usa_6_velocidades_por_coordenada(self):
        E = ec.canon_de_brazo(self.canon, "T2")[2](self.X)
        self.assertEqual(E.shape, (3, 30, 132))
        np.testing.assert_array_equal(E, vr.entrada_red(self.X, "por_coordenada"))
        V = add_velocity_canonica(self.X)[:, :, 126:]
        np.testing.assert_allclose(E[:, :, 126 + 4], V[:, :, 64::3].mean(-1), atol=1e-6)   # der_vy

    def test_reglas_de_los_brazos(self):
        self.assertEqual([ec.BRAZOS[b]["regla"] for b in ("T0", "T1", "T2")], ["mascara", "imputacion", "imputacion"])
        self.assertEqual([ec.BRAZOS[b]["n_rasgos"] for b in ("T0", "T1", "T2")], [252, 252, 132])

    def test_imputacion_antes_de_remuestrear(self):
        # Mano izquierda ausente en los cuadros 10-12 de 40: con imputación ningún paso queda en 0.
        rng = np.random.default_rng(1)
        seq = rng.normal(0, 1, (40, 126)).astype(np.float32)
        seq[10:13, :63] = 0
        Xi = mav.remuestrear_todas_con_regla(seq[None], [40], "imputacion", 30)[0]
        self.assertFalse(np.any(np.all(Xi[:, :63] == 0, axis=1)))
        np.testing.assert_array_equal(Xi, mav.regla_imputacion(seq, 30))


if __name__ == "__main__":
    unittest.main()
