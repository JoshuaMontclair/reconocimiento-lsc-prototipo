"""
Pruebas del criterio y de las entradas del diseño factorial con la TCN (src/experimento_factorial_tcn.py;
docs/EXPERIMENTO_FACTORIAL_TCN.md). Solo NumPy: no entrena ni importa TensorFlow.
Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_factorial_tcn -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import experimento_factorial_tcn as ef  # noqa: E402
from src import mano_ausente_variantes as mav  # noqa: E402
from src import velocidades_reducidas as vr  # noqa: E402

BASE = [0.96, 0.94, 0.94, 0.97, 0.93]
LOSS = [0.92, 1.00, 1.05, 0.89, 1.09]


def _m(por_pliegue, ruido=(0.0, 0.0, 0.0)):
    base = np.asarray(por_pliegue, float)
    return np.stack([base + r for r in ruido])


def _todos(val_mod=None, loss_mod=None):
    val = {b: _m(BASE) for b in ef.TODOS}
    loss = {b: _m(LOSS) for b in ef.TODOS}
    for b, v in (val_mod or {}).items():
        val[b] = _m(v)
    for b, v in (loss_mod or {}).items():
        loss[b] = _m(v)
    return val, loss


class TestBrazos(unittest.TestCase):
    def test_diseno_factorial_completo(self):
        celdas = {(i["regla"], i["n_rasgos"]) for i in ef.BRAZOS.values()}
        self.assertEqual(celdas, {(r, n) for r in ("mascara", "imputacion", "pivote") for n in (252, 132)})
        self.assertEqual(ef.NUEVOS, ("T3", "T4", "T5"))

    def test_reutilizados_apuntan_a_sus_carpetas(self):
        self.assertIn("arquitecturas_literatura_2026-10-02/T_tcn", str(ef.dir_corrida("x", "T0", 2026, 0)))
        self.assertIn("confirmacion_tcn_2026-10-03/T1_imputacion_252", str(ef.dir_corrida("x", "T1", 2026, 0)))
        self.assertIn("confirmacion_tcn_2026-10-03/T2_imputacion_132", str(ef.dir_corrida("x", "T2", 2026, 0)))
        self.assertEqual(str(ef.dir_corrida("/o", "T4", 2027, 3)), "/o/T4_pivote_252/semilla_2027/pliegue_3")

    def test_entradas_por_brazo(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 30, 126)).astype(np.float32)

        def add_velocity(X):
            vel = np.zeros_like(X)
            vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
            return np.concatenate([X, vel], axis=-1)
        for b, info in ef.BRAZOS.items():
            f = ef.funcion_entrada_brazo(b, add_velocity)
            E = f(X)
            self.assertEqual(E.shape[2], info["n_rasgos"], b)
            np.testing.assert_array_equal(E[:, :, :126], X)
            if info["n_rasgos"] == 132:
                np.testing.assert_array_equal(E, vr.entrada_red(X, "por_coordenada"))
            else:
                self.assertIs(f, add_velocity)

    def test_pivote_no_interpola(self):
        """Cada paso del pivote es un fotograma real de la secuencia (por mano)."""
        rng = np.random.default_rng(1)
        seq = rng.standard_normal((67, 126)).astype(np.float32)
        seq[10:25, 0:63] = 0.0
        out = mav.regla_pivote(seq, 30)
        for j in range(30):
            self.assertTrue(np.any(np.all(seq[:, 0:63] == out[j, 0:63], axis=1)))
        self.assertFalse(np.any(np.all(out[:, 0:63] == 0, axis=1)))


class TestPerdida(unittest.TestCase):
    def test_entropia_cruzada_con_y_sin_suavizado(self):
        p = np.full((2, 50), 0.5 / 49)
        p[0, 3] = p[1, 7] = 0.5
        y = [3, 7]
        self.assertAlmostEqual(ef.entropia_cruzada(p, y), -np.log(0.5), places=9)
        esperado = -(0.9 + 0.1 / 50) * np.log(0.5) - 49 * (0.1 / 50) * np.log(0.5 / 49)
        self.assertAlmostEqual(ef.entropia_cruzada(p, y, suavizado=0.1), esperado, places=9)

    def test_supera_en_perdida_exige_4_de_5_y_sigma(self):
        a = _m(np.array(LOSS) - [0.05, 0.05, 0.05, 0.05, -0.01])
        r = ef.supera_en_perdida(a, _m(LOSS), 0.02)
        self.assertTrue(r["cumple"])
        self.assertEqual(r["pliegues_menor"], 4)
        self.assertFalse(ef.supera_en_perdida(a, _m(LOSS), 0.05)["cumple"])        # media 0,038 < 0,05
        b = _m(np.array(LOSS) - [0.10, 0.10, 0.10, -0.01, -0.01])
        self.assertFalse(ef.supera_en_perdida(b, _m(LOSS), 0.01)["cumple"])        # solo 3 de 5


class TestCriterio(unittest.TestCase):
    def test_paso1_un_brazo_supera_a_todos(self):
        val, loss = _todos({"T4": np.array(BASE) + 0.02})
        c = ef.aplicar_criterio(val, loss)
        self.assertEqual(c["decide_en_paso"], 1)
        self.assertEqual(c["configuracion_final"], "T4")
        self.assertEqual(c["margen"], 0.01)

    def test_paso2_desempata_con_val_loss(self):
        val, loss = _todos(loss_mod={"T5": np.array(LOSS) - 0.05})
        val = {b: v + np.array([[0.0], [0.001], [-0.001]]) for b, v in val.items()}   # σ > 0
        loss = {b: v + np.array([[0.0], [0.002], [-0.002]]) for b, v in loss.items()}
        c = ef.aplicar_criterio(val, loss)
        self.assertEqual(c["paso1"]["no_superados"], list(ef.TODOS))
        self.assertEqual(c["decide_en_paso"], 2)
        self.assertEqual(c["configuracion_final"], "T5")

    def test_paso2_solo_entre_candidatos_del_paso1(self):
        # T1 queda superado en validación por T4; aunque tenga la menor val_loss, no compite en el paso 2.
        val, loss = _todos({"T4": np.array(BASE) + 0.005, "T1": np.array(BASE) - 0.02},
                           {"T1": np.array(LOSS) - 0.2})
        c = ef.aplicar_criterio(val, loss)
        self.assertNotIn("T1", c["paso1"]["no_superados"])
        self.assertNotEqual(c["configuracion_final"], "T1")

    def test_paso3_orden_de_simplicidad(self):
        val, loss = _todos()
        c = ef.aplicar_criterio(val, loss)
        self.assertEqual(c["decide_en_paso"], 3)
        self.assertEqual(c["paso3"]["orden_simplicidad"], ["T0", "T3", "T1", "T2", "T4", "T5"])
        self.assertEqual(c["configuracion_final"], "T0")

    def test_paso3_132_primero_si_el_costo_no_es_despreciable(self):
        costo = {"mascara": {"dif_ms": -1.5, "dif_mb": 0.1}, "imputacion": {"dif_ms": -0.2, "dif_mb": 0.6},
                 "pivote": {"dif_ms": -0.1, "dif_mb": 0.1}}
        orden = ef.orden_simplicidad(list(ef.TODOS), costo)
        self.assertEqual(orden, ["T3", "T0", "T2", "T1", "T4", "T5"])

    def test_paso3_preprocesamiento_antes_que_velocidades(self):
        self.assertEqual(ef.orden_simplicidad(["T5", "T1", "T3"], None), ["T3", "T1", "T5"])

    def test_sigma_val_agrupada_sobre_seis_brazos(self):
        val, loss = _todos()
        val["T3"] = _m(BASE, (-0.03, 0.0, 0.03))
        c = ef.aplicar_criterio(val, loss)
        self.assertAlmostEqual(c["sigma_val_agrupada"], np.sqrt(9e-4 / 6), places=9)
        self.assertAlmostEqual(c["margen"], max(np.sqrt(9e-4 / 6), 0.01), places=12)

    def test_costo_desde_inferencia(self):
        inf = {"costo_por_brazo": {b: {"ms_total": 10.0 + (0.3 if ef.BRAZOS[b]["n_rasgos"] == 252 else 0.0),
                                       "mb_archivo": 0.74 if ef.BRAZOS[b]["n_rasgos"] == 252 else 0.62}
                                   for b in ef.TODOS}}
        c = ef.costo_velocidades_desde_inferencia(inf)
        self.assertAlmostEqual(c["mascara"]["dif_ms"], -0.3)
        self.assertAlmostEqual(c["pivote"]["dif_mb"], 0.12)


if __name__ == "__main__":
    unittest.main()
