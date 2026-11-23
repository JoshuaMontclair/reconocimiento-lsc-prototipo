"""
Pruebas del análisis de errores del modelo final (actividad 4.d, src/analisis_errores_tcn.py): voto entre semillas,
tabla de transición, parejas no ordenadas, consistencia entre semillas, proxy de manos en movimiento, tramos,
lectura del conversor y quitado de claves internas. Datos sintéticos y cálculos a mano; sin TensorFlow ni modelos.
Una prueba final corre el análisis completo sobre las predicciones guardadas si están en disco (solo lectura).
    python -m unittest tests.test_analisis_errores_tcn -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import analisis_errores_tcn as aet  # noqa: E402
from src.conversor_glosas import ConversorGlosas  # noqa: E402
from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402

ID = {g: i for i, g in SIGN_VOCABULARY.items()}


class TestVoto(unittest.TestCase):

    def test_mayoria(self):
        preds = [{"a": (1, 2, 0.9), "b": (3, 3, 0.4)}, {"a": (1, 1, 0.8), "b": (3, 4, 0.9)},
                 {"a": (1, 2, 0.1), "b": (3, 3, 0.2)}]
        self.assertEqual(aet.voto_semillas(preds), {"a": (1, 2), "b": (3, 3)})

    def test_tres_distintas_gana_mayor_probabilidad(self):
        preds = [{"a": (0, 5, 0.3)}, {"a": (0, 6, 0.7)}, {"a": (0, 7, 0.5)}]
        self.assertEqual(aet.voto_semillas(preds), {"a": (0, 6)})

    def test_igual_a_la_regla_de_confirmacion(self):
        """Copia literal de experimento_confirmacion_tcn.voto_semillas: se compara el código fuente."""
        ruta = ROOT_DIR / "src" / "experimento_confirmacion_tcn.py"
        if not ruta.exists():
            self.skipTest("experimento_confirmacion_tcn.py no está")
        import inspect
        fuente = ruta.read_text(encoding="utf-8")
        ini = fuente.index("def voto_semillas")
        fin = fuente.index("\n\n\n", ini)
        original = fuente[ini:fin].split('"""')[2]          # cuerpo sin docstring
        propia = inspect.getsource(aet.voto_semillas).split('"""')[2]
        self.assertEqual(original.strip(), propia.strip())


class TestTransicion(unittest.TestCase):

    def test_tabla(self):
        t = aet.tabla_transicion([1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 0, 0])
        self.assertEqual((t["ambos_aciertan"], t["solo_tcn"], t["solo_heredada"], t["ambos_fallan"]), (1, 2, 1, 2))
        self.assertEqual(t["neto_tcn"], 1)
        self.assertEqual(t["n"], 6)
        self.assertAlmostEqual(t["p_mcnemar_exacto"], 1.0)   # 2 frente a 1: binomial bilateral = 1

    def test_mcnemar_conocido(self):
        t = aet.tabla_transicion([1] * 9 + [0], [0] * 9 + [1])
        # 9 frente a 1: p = 2·(1 + 10)/2^10
        self.assertAlmostEqual(t["p_mcnemar_exacto"], 2 * 11 / 1024)


class TestParejas(unittest.TestCase):

    def test_parejas_no_ordenadas(self):
        reales = [14, 14, 15, 26, 44, 44, 3]
        pred = [15, 15, 14, 44, 26, 26, 3]
        p = aet.parejas_no_ordenadas(reales, pred)
        self.assertEqual(p[0], {"a": 14, "b": 15, "a_a_b": 2, "b_a_a": 1, "total": 3})
        self.assertEqual(p[1], {"a": 26, "b": 44, "a_a_b": 1, "b_a_a": 2, "total": 3})
        self.assertEqual(len(p), 2)   # el acierto 3 → 3 no cuenta


class TestConsistencia(unittest.TestCase):

    def test_distribucion(self):
        preds = [{"a": (0, 0), "b": (1, 2), "c": (2, 3), "d": (3, 4)},
                 {"a": (0, 0), "b": (1, 1), "c": (2, 3), "d": (3, 5)},
                 {"a": (0, 0), "b": (1, 1), "c": (2, 3), "d": (3, 4)}]
        c = aet.consistencia_semillas(preds)
        self.assertEqual(c["distribucion_fallos"], {"0": 1, "1": 1, "2": 0, "3": 2})
        self.assertEqual(c["fallan_todas_misma_clase"], 1)        # c: siempre 3; d: 4, 5, 4
        self.assertEqual(c["por_muestra"]["d"]["clases_erroneas_distintas"], 2)


class TestPropiedades(unittest.TestCase):

    def test_manos_en_movimiento(self):
        self.assertEqual(aet.manos_en_movimiento(10.0, 2.0), (0.2, "una"))
        self.assertEqual(aet.manos_en_movimiento(4.0, 8.0), (0.5, "dos"))   # en el umbral cuenta como «dos»
        self.assertEqual(aet.manos_en_movimiento(0.0, 0.0), (None, None))

    def test_tramo_de(self):
        b = (0.0, 0.05, 0.15)
        self.assertEqual([aet.tramo_de(x, b) for x in (0.0, 0.01, 0.05, 0.06, 0.15, 0.4)], [0, 1, 1, 2, 2, 3])

    def test_tramo_confianza(self):
        self.assertEqual([aet.tramo_confianza(x) for x in (0.0, 0.249, 0.25, 0.5, 0.749, 0.75, 1.0)],
                         [0, 0, 1, 2, 2, 3, 3])

    def test_exactitud_por_grupo(self):
        g = aet.exactitud_por_grupo(["una", "dos", "una", "una"], [True, False, False, True])
        self.assertEqual(g["una"]["n"], 3)
        self.assertAlmostEqual(g["una"]["exactitud"], 2 / 3)
        self.assertEqual(g["dos"]["exactitud"], 0.0)

    def test_manos_por_sena(self):
        props = {"0001_0000_0000": {"manos": "dos"}, "0001_0001_0000": {"manos": "una"},
                 "0002_0000_0000": {"manos": "una"}, "0002_0001_0000": {"manos": "una"},
                 "0003_0000_0000": {"manos": None}}
        self.assertEqual(aet.manos_por_sena(props), {1: "dos", 2: "una"})   # empate → «dos»


class TestConversor(unittest.TestCase):

    def setUp(self):
        self.conv = ConversorGlosas(estricto=True)

    def test_nino_nina(self):
        e = aet.clasificar_error_conversor(ID["niña"], ID["niño"], self.conv)
        self.assertTrue(e["misma_categoria"])
        self.assertTrue(e["misma_categoria_distinto_genero"])
        self.assertFalse(e["cambia_modalidad"])
        self.assertEqual((e["enunciado_real"], e["enunciado_predicho"]), ("Niña.", "Niño."))

    def test_pregunta_por_saludo(self):
        e = aet.clasificar_error_conversor(ID["¿por qué?"], ID["hola"], self.conv)
        self.assertFalse(e["misma_categoria"])
        self.assertTrue(e["cambia_modalidad"])
        self.assertEqual((e["categoria_real"], e["categoria_predicha"]), ("INTERR", "SAL"))

    def test_esperado_al_azar(self):
        # Una sola seña real de SAL (5 saludos): 4 de las otras 49 son de su categoría
        self.assertAlmostEqual(aet.esperado_misma_categoria([ID["hola"]], aet.categoria_conversor), 4 / 49)


class TestPublico(unittest.TestCase):

    def test_quita_claves_internas_y_convierte_numpy(self):
        x = {"a": 1, "_b": [1, 2], "c": {"_d": 1, "e": (np.int64(2), np.float32(0.5), np.bool_(True))}}
        self.assertEqual(aet._publico(x), {"a": 1, "c": {"e": [2, 0.5, True]}})


@unittest.skipUnless((aet.LOSO_DIRS["tcn"] / "semilla_2026" / "pliegue_0" / "predicciones.json").exists()
                     and (aet.PROCESSED / "X_padded.npy").exists()
                     and aet.T0_DIR.exists(), "predicciones o datos procesados no disponibles")
class TestDatosReales(unittest.TestCase):
    """Corre el análisis completo (solo lectura, sin figuras ni archivos) y comprueba las cifras registradas."""

    @classmethod
    def setUpClass(cls):
        cls.a, _ = aet.analizar(n_perm=200, semilla=2026)

    def test_controles(self):
        c = self.a["controles"]
        self.assertTrue(c["todo_coincide"])
        self.assertAlmostEqual(c["resumen"]["tcn"]["top1"], 0.677, places=3)
        self.assertAlmostEqual(c["resumen"]["heredada"]["top1"], 0.589, places=3)
        self.assertEqual((c["mcnemar_voto"]["calculado"]["solo_tcn"], c["mcnemar_voto"]["calculado"]["solo_heredada"]),
                         (163, 56))

    def test_transicion_suma_997(self):
        t = self.a["transicion"]["global"]
        self.assertEqual(t["ambos_aciertan"] + t["solo_tcn"] + t["solo_heredada"] + t["ambos_fallan"], 997)
        self.assertEqual(sum(v["n"] for v in self.a["transicion"]["por_senante"].values()), 997)

    def test_aleatoria(self):
        self.assertEqual(len(self.a["aleatoria"]["errores"]["tcn"]), 6)
        self.assertEqual(len(self.a["aleatoria"]["errores"]["heredada"]), 12)

    def test_consistencia_suma(self):
        for m in aet.MODELOS:
            self.assertEqual(sum(self.a["consistencia"][m]["distribucion_fallos"].values()), 997)

    def test_imputacion_reproduce_guardado(self):
        g = self.a["imputacion_t1_vs_t0"]["global_voto"]
        self.assertEqual((g["solo_tcn"], g["solo_heredada"]), (88, 59))


if __name__ == "__main__":
    unittest.main()
