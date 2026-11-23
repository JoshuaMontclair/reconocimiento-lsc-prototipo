"""
Pruebas del reporte comparativo 4.e con el modelo final (src/reporte_comparativo_4e_final.py).
Funciones puras con datos sintéticos y cálculos a mano; y, si las predicciones guardadas están en models/,
una corrida corta (pocas réplicas, sin figuras, a una carpeta temporal) que debe reproducir las cifras
guardadas. Sin TensorFlow y sin escribir en docs/, models/ ni data/.
    python -m unittest tests.test_reporte_comparativo_4e_final -v
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import reporte_comparativo_4e_final as rc  # noqa: E402

HAY_DATOS = all((rc.LOSO_DIR[m] / "semilla_2028" / "pliegue_4" / "predicciones.json").exists()
                for m in rc.MODELOS) and all((rc.ALEATORIA[m] / "predicciones.json").exists() for m in rc.MODELOS)


class TestUtilidades(unittest.TestCase):

    def test_parsear_muestra(self):
        self.assertEqual(rc.parsear_muestra("0027_0003_0002"), (27, 3, 2))
        with self.assertRaises(ValueError):
            rc.parsear_muestra("27_3_2")

    def test_metricas_macro_a_mano(self):
        # clases 0,1,2: y = [0,0,1,1], p = [0,1,1,2]
        y, p = [0, 0, 1, 1], [0, 1, 1, 2]
        m = rc.metricas_macro(y, p, n=3)
        # precision: c0 1/1, c1 1/2, c2 0/1 → 0,5; recall: c0 1/2, c1 1/2, c2 (no real) 0 → 1/3
        self.assertAlmostEqual(m["precision_macro"], 0.5)
        self.assertAlmostEqual(m["recall_macro"], 1 / 3)
        # F1: c0 2·1/(2+1)=2/3, c1 2·1/(2+2)=1/2, c2 0 → (2/3+1/2)/3
        self.assertAlmostEqual(m["f1_macro"], (2 / 3 + 0.5) / 3)

    def test_metricas_macro_coincide_con_sklearn(self):
        from sklearn.metrics import f1_score, precision_score, recall_score
        rng = np.random.default_rng(0)
        y = rng.integers(0, 50, 300)
        p = np.where(rng.random(300) < 0.6, y, rng.integers(0, 50, 300))
        m = rc.metricas_macro(y, p)
        self.assertAlmostEqual(m["f1_macro"], f1_score(y, p, average="macro", zero_division=0))
        self.assertAlmostEqual(m["precision_macro"], precision_score(y, p, average="macro", zero_division=0))
        self.assertAlmostEqual(m["recall_macro"], recall_score(y, p, average="macro", zero_division=0))

    def test_recall_por_clase(self):
        r = rc.recall_por_clase([0, 0, 1], [0, 1, 1], n=3)
        self.assertEqual(r[0], 0.5)
        self.assertEqual(r[1], 1.0)
        self.assertTrue(np.isnan(r[2]))

    def test_voto_semillas(self):
        a = {"m1": (0, 1, 0.9), "m2": (0, 0, 0.5)}
        b = {"m1": (0, 1, 0.4), "m2": (0, 2, 0.6)}
        c = {"m1": (0, 3, 0.8), "m2": (0, 3, 0.7)}
        v = rc.voto_semillas([a, b, c])
        self.assertEqual(v["m1"], (0, 1))          # mayoría 2 de 3
        self.assertEqual(v["m2"], (0, 3))          # todas distintas: gana la de mayor probabilidad (0,7)

    def test_mcnemar(self):
        a = np.array([1, 1, 1, 0, 1, 0], bool)
        b = np.array([1, 0, 0, 0, 1, 1], bool)
        r = rc.mcnemar(a, b)
        self.assertEqual((r["solo_a"], r["solo_b"], r["ambos"], r["ninguno"]), (2, 1, 2, 1))
        self.assertAlmostEqual(r["p_exacto_bilateral"], 1.0)

    def test_stats_bloques_con_pesos_unitarios(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 50, 100)
        g = np.repeat(np.arange(5), 20)
        pred = np.where(rng.random((2, 100)) < 0.7, y, rng.integers(0, 50, (2, 100)))
        top3 = (pred == y) | (rng.random((2, 100)) < 0.3)
        bl = [(np.flatnonzero(g == k), np.ones(20)) for k in range(5)]
        t1, t3, f1 = rc.stats_bloques(pred, top3, y, bl)
        esperado_t1 = np.mean([[np.mean(pred[s, g == k] == y[g == k]) for k in range(5)] for s in range(2)])
        esperado_f1 = np.mean([[rc.metricas_macro(y[g == k], pred[s, g == k])["f1_macro"] for k in range(5)]
                               for s in range(2)])
        self.assertAlmostEqual(t1, esperado_t1)
        self.assertAlmostEqual(t3, top3.mean())
        self.assertAlmostEqual(f1, esperado_f1)

    def test_stats_bloques_pesos_equivalen_a_repetir(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 50, 40)
        pred = np.where(rng.random((1, 40)) < 0.5, y, rng.integers(0, 50, (1, 40)))
        top3 = pred == y
        idx = np.arange(40)
        w = rng.integers(0, 3, 40).astype(float)
        con_pesos = rc.stats_bloques(pred, top3, y, [(idx, w), (idx, np.ones(40))])
        rep = np.repeat(idx, w.astype(int))
        y2 = np.r_[y[rep], y]
        p2 = np.r_[pred[0, rep], pred[0]][None, :]
        bl2 = [(np.arange(len(rep)), np.ones(len(rep))), (np.arange(len(rep), len(rep) + 40), np.ones(40))]
        explicito = rc.stats_bloques(p2, p2 == y2, y2, bl2)
        np.testing.assert_allclose(con_pesos, explicito)

    def test_stats_pesos(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0, 1, 1, 1])
        ac = (y == p).astype(float)
        r = rc.stats_pesos(ac, ac, y, p, np.ones(4))
        self.assertAlmostEqual(r[0], 0.75)
        self.assertAlmostEqual(r[2], rc.metricas_macro(y, p)["f1_macro"])

    def test_pesos_estratificados_conservan_tamano_por_grupo(self):
        rng = np.random.default_rng(3)
        grupos = [np.array([0, 1, 2]), np.array([3, 4])]
        w = rc.pesos_estratificados(rng, grupos, 5)
        self.assertEqual(w[:3].sum(), 3)
        self.assertEqual(w[3:].sum(), 2)

    def test_percentil_e_ic_bc(self):
        v = np.arange(1001) / 1000
        lo, hi = rc.percentil(v)
        self.assertAlmostEqual(lo, 0.025)
        self.assertAlmostEqual(hi, 0.975)
        bc = rc.ic_bc(v, 0.5)
        self.assertAlmostEqual(bc[0], 0.025, places=2)


@unittest.skipUnless(HAY_DATOS, "faltan las predicciones guardadas en models/")
class TestDatosReales(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        salida = Path(cls.tmp.name) / "r.json"
        with contextlib.redirect_stdout(io.StringIO()):
            rc.main(["--replicas", "60", "--salida", str(salida), "--sin-figuras"])
        cls.r = rc._leer(salida)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_controles_contra_lo_guardado(self):
        self.assertTrue(self.r["controles_todo_coincide"])

    def test_cifras_puntuales_conocidas(self):
        lo, al = self.r["loso"], self.r["aleatoria"]
        self.assertAlmostEqual(lo["final"]["top1"]["media"], 0.6771, places=4)
        self.assertAlmostEqual(lo["heredado"]["top1"]["media"], 0.5891, places=4)
        self.assertAlmostEqual(lo["final"]["f1_macro"]["media"], 0.6467, places=4)
        self.assertAlmostEqual(lo["final"]["top1"]["por_pliegue_media_semillas"][0], 0.4883, places=4)
        self.assertEqual(al["final"]["top1"], 0.94)
        self.assertEqual(al["heredado"]["top1"], 0.88)

    def test_mcnemar_voto(self):
        v = self.r["mcnemar"]["loso_voto_3_semillas"]
        self.assertEqual((v["solo_a"], v["solo_b"]), (163, 56))

    def test_brecha_es_resta(self):
        ic = self.r["intervalos"]["estadisticos"]
        for m in rc.MODELOS:
            self.assertAlmostEqual(ic[f"{m}|brecha|top1"]["punto"],
                                   ic[f"{m}|aleatoria|top1"]["punto"] - ic[f"{m}|loso|top1"]["punto"])

    def test_validacion_no_contiene_al_senante_de_prueba(self):
        # validacion_por_senante_y_clase lanza AssertionError si lo contuviera; aquí basta con que exista
        if not self.r["validacion_loso_final_por_senante"]:
            self.skipTest("requiere el corpus procesado (data/processed_*), que no se distribuye")
        self.assertIn("top1_por_senante_media_semillas", self.r["validacion_loso_final_por_senante"])

    def test_por_clase_tiene_50_senas(self):
        self.assertEqual(len(self.r["por_clase"]["senas"]), 50)


if __name__ == "__main__":
    unittest.main()
