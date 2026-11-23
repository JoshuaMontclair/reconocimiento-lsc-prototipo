"""
Pruebas del análisis de errores (actividad 2.d, src/analisis_errores.py) con datos sintéticos pequeños y
cálculos a mano: recall/precisión/F1 por seña, pares de confusión, métricas por señante, ECE, efecto del
umbral, McNemar, permutación estratificada, controles contra lo guardado y flujo completo en una carpeta
temporal. Sin TensorFlow, sin modelos y sin tocar models/ ni data/.
    python -m unittest tests.test_analisis_errores -v
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import analisis_errores as ae  # noqa: E402
from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402


def _reg(sena: int, senante: str, rep: int, pred: int, top3=None, prob: float = 0.5, pliegue: int = 0) -> dict:
    return {"muestra": f"{sena:04d}_{senante}_{rep:04d}", "real": sena, "predicha": pred,
            "top3": top3 if top3 is not None else [pred, (pred + 1) % 50, (pred + 2) % 50],
            "prob_predicha": prob, "pliegue": pliegue, "senante": senante, "repeticion": rep}


class TestMetricasPorSena(unittest.TestCase):

    def test_parsear_muestra(self):
        self.assertEqual(ae.parsear_muestra("0027_0003_0002"), (27, "0003", 2))
        with self.assertRaises(ValueError):
            ae.parsear_muestra("27_3_2")

    def test_recall_precision_f1_a_mano(self):
        reales = [0, 0, 0, 1, 1, 2]
        preds = [0, 1, 0, 1, 0, 1]
        m = ae.metricas_por_clase(reales, preds, n=3)
        # clase 0: 2 aciertos de 3 reales; predicha 3 veces (muestras 0, 2, 4) → P = R = F1 = 2/3
        self.assertAlmostEqual(m["recall"][0], 2 / 3)
        self.assertAlmostEqual(m["precision"][0], 2 / 3)
        self.assertAlmostEqual(m["f1"][0], 2 / 3)
        # clase 1: 1 acierto de 2 reales; predicha 3 veces → R = 1/2, P = 1/3, F1 = 2·(1/6)/(5/6) = 0,4
        self.assertAlmostEqual(m["recall"][1], 0.5)
        self.assertAlmostEqual(m["precision"][1], 1 / 3)
        self.assertAlmostEqual(m["f1"][1], 0.4)
        # clase 2: nunca predicha → precisión indefinida, recall 0, F1 0
        self.assertIsNone(m["precision"][2])
        self.assertEqual(m["recall"][2], 0.0)
        self.assertEqual(m["f1"][2], 0.0)
        self.assertEqual(m["soporte"], [3, 2, 1])
        self.assertEqual(m["n_predichas"], [3, 3, 0])
        self.assertAlmostEqual(ae.f1_macro(reales, preds, n=3), (2 / 3 + 0.4 + 0.0) / 3)

    def test_f1_macro_igual_que_sklearn_con_las_50_etiquetas(self):
        try:
            from sklearn.metrics import precision_recall_fscore_support
        except ImportError:
            self.skipTest("sklearn no disponible")
        rng = np.random.default_rng(0)
        reales = rng.integers(0, 50, 300)
        preds = np.where(rng.random(300) < 0.5, reales, rng.integers(0, 50, 300))
        _, _, f1, _ = precision_recall_fscore_support(reales, preds, labels=list(range(50)), average="macro",
                                                      zero_division=0)
        self.assertAlmostEqual(ae.f1_macro(reales, preds), f1, places=12)

    def test_top_k(self):
        self.assertAlmostEqual(ae.top_k([1, 2, 3], [[1, 0, 0], [0, 2, 0], [0, 0, 9]], 3), 2 / 3)
        self.assertAlmostEqual(ae.top_k([1, 2, 3], [[1, 0, 0], [0, 2, 0], [0, 0, 9]], 1), 1 / 3)

    def test_media_y_de_muestral_ignorando_indefinidos(self):
        m, de, n = ae.media_de([0.25, None, 0.75])
        self.assertEqual((m, n), (0.5, 2))
        self.assertAlmostEqual(de, math.sqrt(((0.25 - 0.5) ** 2 + (0.75 - 0.5) ** 2) / 1))
        self.assertEqual(ae.media_de([None]), (None, None, 0))

    def test_seccion_por_sena_valores_por_pliegue_y_ranking(self):
        # seña 0: pliegue 0 acierta 2/2, pliegue 1 acierta 0/2; seña 1: acierta todo
        regs = [_reg(0, "0000", 0, 0, pliegue=0), _reg(0, "0000", 1, 0, pliegue=0),
                _reg(1, "0000", 0, 1, pliegue=0), _reg(1, "0000", 1, 1, pliegue=0),
                _reg(0, "0001", 0, 1, pliegue=1), _reg(0, "0001", 1, 1, pliegue=1),
                _reg(1, "0001", 0, 1, pliegue=1), _reg(1, "0001", 1, 1, pliegue=1)]
        s = ae.seccion_por_sena(regs)
        f0, f1 = s["senas"][0], s["senas"][1]
        self.assertEqual(f0["por_pliegue"]["recall"], [1.0, 0.0])
        self.assertAlmostEqual(f0["recall"], 0.5)
        self.assertAlmostEqual(f0["recall_de_pliegues"], math.sqrt(0.5))
        # seña 1 en el agregado: 4 aciertos de 4; predicha 6 veces → P = 4/6
        self.assertAlmostEqual(f1["recall"], 1.0)
        self.assertAlmostEqual(f1["precision"], 4 / 6)
        # pliegue 1: la seña 0 nunca se predice → precisión indefinida en ese pliegue
        self.assertEqual(f0["por_pliegue"]["precision"], [1.0, None])
        self.assertEqual(s["mejores_10"][0], 1)
        # señas sin soporte: recall indefinido, van primero entre las peores y no cuentan como recall 0
        self.assertIsNone(s["senas"][2]["recall"])
        self.assertEqual(s["peores_10"][0], 2)
        self.assertEqual(s["senas_con_recall_0"], [])


class TestConfusiones(unittest.TestCase):

    def test_pares_y_simetria(self):
        reales = [0, 0, 1, 1, 1, 2]
        preds = [1, 1, 0, 2, 1, 2]
        pares = ae.pares_confusion(reales, preds)
        # (0→1) 2 casos con 1 inverso: hay confusión en ambos sentidos, pero no es simétrica (el menor sentido < 2)
        self.assertEqual([(p["real"], p["predicha"], p["conteo"], p["conteo_inverso"], p["en_ambos_sentidos"],
                           p["simetrica"]) for p in pares],
                         [(0, 1, 2, 1, True, False), (1, 0, 1, 2, True, False), (1, 2, 1, 0, False, False)])
        self.assertAlmostEqual(pares[0]["cociente_menor_mayor"], 0.5)
        self.assertEqual(pares[2]["cociente_menor_mayor"], 0.0)

    def test_criterio_de_simetria(self):
        # 6/5 y 3/5 simétricas; 8/1 y 6/1 (un caso inverso), 2/5 (cociente 0,4) y 1/1 (menos de 2) no
        casos = {(6, 5): True, (5, 6): True, (3, 5): True, (2, 2): True, (8, 1): False, (6, 1): False,
                 (2, 5): False, (1, 1): False, (4, 0): False}
        for (c, inv), esperado in casos.items():
            self.assertEqual(ae.es_simetrica(c, inv), esperado, (c, inv))
        regs = [_reg(27, "0000", i, 38) for i in range(8)] + [_reg(38, "0000", 0, 27)] \
            + [_reg(14, "0001", i, 15, pliegue=1) for i in range(6)] + [_reg(15, "0001", i, 14, pliegue=1) for i in range(5)]
        c = ae.seccion_confusiones(regs)
        # 20 errores: 11 en la pareja niño/niña (simétrica) y 9 en más o menos/yo (8/1, no simétrica)
        self.assertAlmostEqual(c["proporcion_errores_en_pares_simetricos"], 11 / 20)
        self.assertAlmostEqual(c["proporcion_errores_en_pares_con_ambos_sentidos"], 1.0)
        self.assertEqual(c["n_parejas_simetricas"], 1)

    def test_seccion_confusiones_por_categoria(self):
        # niño(14) → niña(15) ×2 (misma categoría: sujeto); hola(30) → hombre(13) ×1 (saludo → sujeto)
        regs = [_reg(14, "0000", 0, 15), _reg(14, "0000", 1, 15), _reg(30, "0001", 0, 13), _reg(1, "0001", 0, 1)]
        c = ae.seccion_confusiones(regs)
        self.assertEqual(c["n_errores"], 3)
        pc = c["por_categoria"]
        self.assertEqual(pc["errores_dentro_de_la_misma_categoria"], 2)
        self.assertEqual(pc["matriz_errores_categoria_real_a_predicha"]["saludo"]["sujeto"], 1)
        # esperado al azar: niño es sujeto (13 señas) → 2·12/49; hola es saludo (5 señas) → 1·4/49
        self.assertAlmostEqual(pc["proporcion_dentro_esperada_si_errores_al_azar"], (2 * 12 / 49 + 4 / 49) / 3)
        self.assertEqual(c["parejas_no_ordenadas"][0]["total"], 2)
        self.assertEqual(c["pares"][0]["senantes"], ["0000"])


class TestPorSenante(unittest.TestCase):

    def test_top1_top3_f1_macro_a_mano(self):
        regs = [
            _reg(0, "0000", 0, 0, top3=[0, 5, 6], prob=0.9),
            _reg(0, "0000", 1, 1, top3=[1, 6, 0], prob=0.6),   # error con la real en la 3.ª posición del top-3
            _reg(1, "0000", 0, 1, top3=[1, 5, 6], prob=0.8),
            _reg(1, "0000", 1, 2, top3=[2, 5, 6], prob=0.3),   # error sin la real en el top-3
            _reg(0, "0001", 0, 0, top3=[0, 1, 2], prob=0.7, pliegue=1),
            _reg(0, "0002", 0, 3, top3=[3, 1, 2], prob=0.5, pliegue=2),   # verificador r2: error con confianza 0,5 exacta
        ]
        ps = ae.seccion_por_senante(regs)
        d = ps["0000"]
        self.assertEqual(d["n"], 4)
        self.assertAlmostEqual(d["top1"], 0.5)
        self.assertAlmostEqual(d["top3"], 0.75)   # con top-2 sería 0,5
        # clase 0: R 1/2, P 1/1 → F1 2/3; clase 1: R 1/2, P 1/2 → F1 1/2; clase 2: 0; resto de las 50: 0
        self.assertAlmostEqual(d["f1_macro"], (2 / 3 + 1 / 2) / 50)
        self.assertEqual(d["n_errores"], 2)
        self.assertEqual(d["errores_con_real_en_top3"], 1)
        self.assertEqual(d["errores_con_confianza_ge_0_5"], 1)
        self.assertAlmostEqual(d["confianza_media_aciertos"], (0.9 + 0.8) / 2)
        self.assertAlmostEqual(d["confianza_media_errores"], (0.6 + 0.3) / 2)
        self.assertEqual(d["senas_con_recall_0"], [])
        self.assertEqual(ps["0001"]["top1"], 1.0)
        self.assertEqual(ps["0002"]["errores_con_confianza_ge_0_5"], 1)   # ≥ 0,5 es inclusivo

    def test_grupos_n30_y_categorias(self):
        regs = [_reg(41, "0000", 0, 41), _reg(42, "0000", 0, 41),   # interrogativos: 1/2, ambas dentro del grupo
                _reg(20, "0000", 0, 0),                               # emociones: 0/1, fuera
                _reg(6, "0000", 0, 43),                               # trabajar (fuera) predicha como ¿dónde? (dentro)
                _reg(1, "0000", 0, 1)]
        por_sena = ae.seccion_por_sena(regs)
        g = ae.seccion_grupos(regs, por_sena)
        gi = g["grupos_n30"]["interrogativos"]
        self.assertAlmostEqual(gi["recall"], 0.5)
        self.assertAlmostEqual(gi["proporcion_predicha_dentro_del_grupo"], 1.0)
        # tres predicciones caen en el grupo (41, 41, 43) y una es acierto: la precisión no es el recall
        self.assertAlmostEqual(gi["precision_del_grupo"], 1 / 3)
        self.assertEqual(g["grupos_n30"]["emociones"]["recall"], 0.0)
        # verificador r2: feliz (20) predicha como gracias (0) cae fuera del grupo; el recall por señante del grupo
        # usa solo las muestras del grupo (0000 acierta 1 de 2 interrogativos; su top-1 total sería 2/5)
        self.assertEqual(g["grupos_n30"]["emociones"]["proporcion_predicha_dentro_del_grupo"], 0.0)
        self.assertAlmostEqual(gi["recall_por_senante"]["0000"], 0.5)
        self.assertEqual(g["categorias_gramaticales"]["saludo"]["recall"], 1.0)   # buenos días (1) es saludo


class TestCalibracionYUmbral(unittest.TestCase):

    def test_ece_a_mano(self):
        conf = [0.15, 0.15, 0.85, 0.95]
        ok = [True, False, True, False]
        f = ae.fiabilidad(conf, ok, n_bins=10)
        # intervalo 1: n 2, exactitud 0,5, confianza 0,15 → 0,35; intervalo 8: 1 − 0,85 = 0,15; intervalo 9: 0,95
        self.assertAlmostEqual(f["ece"], 2 / 4 * 0.35 + 1 / 4 * 0.15 + 1 / 4 * 0.95)
        self.assertAlmostEqual(f["mce"], 0.95)
        self.assertEqual([b["n"] for b in f["intervalos"]], [0, 2, 0, 0, 0, 0, 0, 0, 1, 1])
        self.assertIsNone(f["intervalos"][0]["exactitud"])

    def test_bordes_0_y_1(self):
        f = ae.fiabilidad([0.0, 1.0], [False, True], n_bins=10)
        self.assertEqual(f["intervalos"][0]["n"], 1)
        self.assertEqual(f["intervalos"][9]["n"], 1)
        self.assertAlmostEqual(f["ece"], 0.0)

    def test_efecto_umbral_a_mano_y_aceptacion_inclusiva(self):
        conf = [0.10, 0.20, 0.30, 0.90, 0.19]
        ok = [True, True, False, True, False]
        e = ae.efecto_umbral(conf, ok, 0.20)
        # aceptadas 0,20 (>=), 0,30 y 0,90: 2 correctas, 1 errónea
        self.assertEqual((e["aceptadas"], e["aceptadas_correctas"], e["aceptadas_erroneas"]), (3, 2, 1))
        self.assertAlmostEqual(e["rechazo"], 2 / 5)
        self.assertAlmostEqual(e["exactitud_aceptadas"], 2 / 3)
        self.assertAlmostEqual(e["error_aceptadas"], 1 / 3)
        self.assertAlmostEqual(e["aciertos_perdidos"], 1 / 3)
        self.assertAlmostEqual(e["errores_rechazados"], 1 / 2)
        self.assertIsNone(ae.efecto_umbral([0.1], [True], 0.5)["exactitud_aceptadas"])

    def test_umbral_por_senante(self):
        regs = [_reg(0, "0000", 0, 0, prob=0.1), _reg(1, "0000", 0, 2, prob=0.5),
                _reg(0, "0001", 0, 0, prob=0.9, pliegue=1), _reg(1, "0001", 0, 1, prob=0.25, pliegue=1)]
        cal = ae.seccion_calibracion(regs, 0.2, 10)
        u0, u1 = cal["umbral"]["por_senante"]["0000"], cal["umbral"]["por_senante"]["0001"]
        self.assertAlmostEqual(u0["rechazo"], 0.5)
        self.assertAlmostEqual(u0["error_aceptadas"], 1.0)
        self.assertAlmostEqual(u1["rechazo"], 0.0)
        self.assertAlmostEqual(u1["error_aceptadas"], 0.0)
        self.assertEqual(cal["umbral"]["global"]["aceptadas"], 3)


class TestPruebasEstadisticas(unittest.TestCase):

    def test_mcnemar_exacto(self):
        self.assertAlmostEqual(ae.mcnemar_exacto(0, 5), 2 / 32)
        self.assertAlmostEqual(ae.mcnemar_exacto(1, 5), 2 * (1 + 6) / 64)
        self.assertEqual(ae.mcnemar_exacto(3, 3), 1.0)
        self.assertEqual(ae.mcnemar_exacto(0, 0), 1.0)

    def test_diferencia_dentro_de_estratos_a_mano_e_igual_a_efectos_fijos(self):
        # estrato A: aciertos [0], errores [2] → d_A = 2, peso 1·1/2; estrato B: aciertos [1, 1], errores [4] → d_B = 3,
        # peso 1·2/3. D = (0,5·2 + (2/3)·3) / (0,5 + 2/3) = 18/7
        valores = np.array([0.0, 2.0, 1.0, 1.0, 4.0])
        ok = np.array([True, False, True, True, False])
        estr = np.array(["A", "A", "B", "B", "B"])
        d = ae.diferencia_dentro_de_estratos(valores, ok, estr)
        self.assertAlmostEqual(d["diferencia"], 18 / 7)
        self.assertEqual(d["por_estrato"]["A"]["diferencia"], 2.0)
        self.assertEqual(d["por_estrato"]["B"]["diferencia"], 3.0)
        self.assertAlmostEqual(d["diferencia_bruta"], 3.0 - 2 / 3)
        # regresión valores ~ error + efectos fijos del estrato (sin intercepto común)
        X = np.column_stack([(~ok).astype(float), (estr == "A").astype(float), (estr == "B").astype(float)])
        coef = np.linalg.lstsq(X, valores, rcond=None)[0][0]
        self.assertAlmostEqual(d["diferencia"], coef)

    def test_estratos_con_tasas_y_medias_distintas_y_efecto_dentro_del_estrato(self):
        # caso del verificador: A con 10 % de error y media 0, B con 90 % de error y media 10; dentro de cada
        # estrato los errores valen 1 menos. La diferencia bruta es +7 (errores 8, aciertos 1); dentro de estrato −1.
        rng = np.random.default_rng(11)

        def ruido(n):
            r = rng.normal(0, 0.3, n)
            return r - r.mean()

        valores = np.concatenate([ruido(90), -1 + ruido(10), 10 + ruido(10), 9 + ruido(90)])
        ok = np.array([True] * 90 + [False] * 10 + [True] * 10 + [False] * 90)
        estr = np.array(["A"] * 100 + ["B"] * 100)
        r = ae.permutacion_estratificada(valores, ok, estr, 4000, 5)
        self.assertAlmostEqual(r["diferencia_bruta_sin_estratos"], 7.0)
        self.assertAlmostEqual(r["diferencia_dentro_de_estratos"], -1.0)
        # ninguna permutación dentro de estrato llega a |−1|: p = 1 / (n_perm + 1)
        self.assertEqual(r["p_bilateral"], 1 / 4001)
        self.assertLess(abs(r["media_distribucion_nula"]), 0.05)   # nula centrada en 0, no en +7

    def test_sin_efecto_dentro_del_estrato_p_1_aunque_la_diferencia_bruta_no_sea_0(self):
        # A: aciertos {0, 1}, errores {0, 1} → d_A = 0; B: aciertos {10, 12}, errores {11, 11, 10, 12} → d_B = 0.
        # Bruta: errores 45/6 = 7,5, aciertos 23/4 = 5,75 → +1,75 (B tiene más errores y valores más altos)
        valores = np.array([0, 1, 0, 1, 10, 12, 11, 11, 10, 12], dtype=float)
        ok = np.array([True, True, False, False, True, True, False, False, False, False])
        estr = np.array(["A"] * 4 + ["B"] * 6)
        r = ae.permutacion_estratificada(valores, ok, estr, 500, 3)
        self.assertAlmostEqual(r["diferencia_dentro_de_estratos"], 0.0)
        self.assertAlmostEqual(r["diferencia_bruta_sin_estratos"], 1.75)
        self.assertEqual(r["p_bilateral"], 1.0)

    def test_p_exacto_por_enumeracion_con_dos_estratos(self):
        # A: valores [0, 1, 2], error en 2; B: valores [0, 3], error en 3. Centrados: A [−1, 0, 1], B [−1,5; 1,5].
        # Estadístico ∝ c_errA + c_errB; observado 2,5. Las 6 asignaciones dan ±2,5, ±1,5, ±0,5 → p exacto 2/6.
        valores = np.array([0, 1, 2, 0, 3], dtype=float)
        ok = np.array([True, True, False, True, False])
        estr = np.array(["A", "A", "A", "B", "B"])
        r = ae.permutacion_estratificada(valores, ok, estr, 30000, 9)
        self.assertAlmostEqual(r["diferencia_dentro_de_estratos"], 2.5 / (2 / 3 + 1 / 2))
        self.assertAlmostEqual(r["p_bilateral"], 1 / 3, delta=0.015)
        self.assertAlmostEqual(r["media_distribucion_nula"], 0.0, delta=0.03)

    def test_estratos_sin_errores_o_sin_aciertos_no_dan_comparacion(self):
        # todo el estrato A acierta con valor 0 y todo el B falla con valor 10: diferencia bruta 10, sin comparación
        valores = [0, 0, 0, 10, 10, 10]
        ok = [True, True, True, False, False, False]
        r = ae.permutacion_estratificada(valores, ok, ["A"] * 3 + ["B"] * 3, 200, 1)
        self.assertAlmostEqual(r["diferencia_bruta_sin_estratos"], 10.0)
        self.assertIsNone(r["diferencia_dentro_de_estratos"])
        self.assertIsNone(r["p_bilateral"])
        self.assertEqual(r["estratos_con_errores_y_aciertos"], 0)

    def test_permutacion_detecta_un_efecto_dentro_del_estrato(self):
        rng = np.random.default_rng(3)
        ok = np.array([True] * 40 + [False] * 40)
        valores = np.where(ok, 0.0, 5.0) + rng.normal(0, 0.1, 80)
        r = ae.permutacion_estratificada(valores, ok, np.array(["A", "B"] * 40), 500, 7)
        self.assertLess(r["p_bilateral"], 0.01)

    def test_holm_a_mano(self):
        h = ae.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
        # orden a 0,01·3 = 0,03; c 0,03·2 = 0,06; b 0,04·1 = 0,04 → máx. acumulado 0,06
        self.assertAlmostEqual(h["a"], 0.03)
        self.assertAlmostEqual(h["c"], 0.06)
        self.assertAlmostEqual(h["b"], 0.06)
        self.assertIsNone(h["d"])
        self.assertEqual(ae.holm({"x": 0.9, "y": 0.8})["x"], 1.0)

    def test_correlacion_permutacion(self):
        # r = 1; solo la identidad y el orden inverso dan |r| = 1 entre las 24 permutaciones → p exacto 1/12
        r = ae.correlacion_permutacion([1, 2, 3, 4], [1, 2, 3, 4], 20000, 4)
        self.assertAlmostEqual(r["pearson"], 1.0)
        self.assertAlmostEqual(r["p_bilateral"], 1 / 12, delta=0.01)
        self.assertIsNone(ae.correlacion_permutacion([1, 2], [3, 3], 10, 0)["pearson"])

    def test_tabla_por_cuantiles(self):
        filas = ae.tabla_por_cuantiles([1, 2, 3, 4], [True, True, False, False], 2)
        self.assertEqual([f["n"] for f in filas], [2, 2])
        self.assertEqual([f["exactitud"] for f in filas], [1.0, 0.0])


# ---------------------------------------------------------------------------------------------------
# Carpeta LOSO sintética con los mismos archivos que escribe train_loso.py
# ---------------------------------------------------------------------------------------------------

def _escribir_loso(dir_: Path, prediccion) -> dict:
    """5 pliegues × 50 señas × 1 repetición; prediccion(sena, k) → (predicha, prob). Métricas guardadas
    calculadas aquí con bucles propios (independientes del módulo)."""
    nombres = [f"{c:02d}_{SIGN_VOCABULARY[c]}" for c in range(50)]
    pliegues = []
    for k in range(5):
        preds = []
        for c in range(50):
            p, prob = prediccion(c, k)
            top3 = [p] + [x for x in (c, (c + 7) % 50, (c + 8) % 50) if x != p][:2]
            preds.append({"muestra": f"{c:04d}_{k:04d}_0000", "real": c, "predicha": p, "top3": top3, "prob_predicha": prob})
        d = dir_ / f"pliegue_{k}"
        d.mkdir(parents=True)
        (d / "predicciones.json").write_text(json.dumps(preds), encoding="utf-8")
        mat = [[0] * 50 for _ in range(50)]
        for r in preds:
            mat[r["real"]][r["predicha"]] += 1
        f1s, cr = [], {}
        for c in range(50):
            tp = mat[c][c]
            sop = sum(mat[c])
            npred = sum(mat[i][c] for i in range(50))
            rec = tp / sop
            pre = tp / npred if npred else 0.0
            f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
            f1s.append(f1)
            cr[nombres[c]] = {"precision": pre, "recall": rec, "f1-score": f1, "support": float(sop)}
        top1 = sum(r["real"] == r["predicha"] for r in preds) / 50
        top3 = sum(r["real"] in r["top3"] for r in preds) / 50
        (d / "metricas.json").write_text(json.dumps({"top1": top1, "top3": top3, "f1_macro": sum(f1s) / 50}), encoding="utf-8")
        (d / "classification_report.json").write_text(json.dumps(cr), encoding="utf-8")
        (d / "matriz_confusion.json").write_text(json.dumps({"etiquetas": list(range(50)), "nombres": nombres, "matriz": mat}),
                                                 encoding="utf-8")
        pliegues.append({"pliegue": k, "senante_test": f"{k:04d}", "top1": top1, "top3": top3,
                         "n_errores": sum(r["real"] != r["predicha"] for r in preds)})
    vals = [p["top1"] for p in pliegues]
    media = sum(vals) / 5
    de = math.sqrt(sum((v - media) ** 2 for v in vals) / 4)
    res = {"representacion": "sintetica", "pliegues": pliegues,
           "agregado": {"top1": {"media": media, "desviacion_estandar": de, "valores": vals}}}
    (dir_ / "resultados_loso.json").write_text(json.dumps(res), encoding="utf-8")
    return res


def _pred_base(c: int, k: int):
    return (c if (c + k) % 3 else (c + 1) % 50), 0.05 + 0.9 * ((c * 7 + k) % 10) / 10


def _pred_contraste(c: int, k: int):
    return (c if (c + k) % 4 else (c + 2) % 50), 0.5


class TestControlesYFlujoCompleto(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.loso = self.base / "loso"
        self.res = _escribir_loso(self.loso, _pred_base)
        self.contraste = self.base / "contraste"
        _escribir_loso(self.contraste, _pred_contraste)
        self.umbral = self.base / "umbral.json"
        self.umbral.write_text(json.dumps({"umbral": 0.2}), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_controles_reproducen_lo_guardado(self):
        regs, res = ae.cargar_loso(self.loso)
        self.assertEqual(len(regs), 250)
        ctl = ae.controles_loso(self.loso, regs, res)
        self.assertTrue(ctl["todo_coincide"])
        self.assertAlmostEqual(ctl["top1_de_calculada"], self.res["agregado"]["top1"]["desviacion_estandar"])

    def test_top1_distinto_detiene_el_analisis(self):
        res = json.loads((self.loso / "resultados_loso.json").read_text(encoding="utf-8"))
        res["pliegues"][2]["top1"] += 0.02
        (self.loso / "resultados_loso.json").write_text(json.dumps(res), encoding="utf-8")
        regs, res = ae.cargar_loso(self.loso)
        with self.assertRaises(RuntimeError):
            ae.controles_loso(self.loso, regs, res)

    def test_senante_que_no_corresponde_al_pliegue(self):
        ruta = self.loso / "pliegue_1" / "predicciones.json"
        preds = json.loads(ruta.read_text(encoding="utf-8"))
        preds[0]["muestra"] = "0000_0003_0000"
        ruta.write_text(json.dumps(preds), encoding="utf-8")
        with self.assertRaises(ValueError):
            ae.cargar_loso(self.loso)

    def _main(self, *extra):
        salida = self.base / "salida"
        argv = ["--loso-dir", str(self.loso), "--salida", str(salida), "--contraste", f"otra={self.contraste}",
                "--processed", str(self.base / "no_existe"), "--metricas-fotograma", str(self.base / "no_existe"),
                "--umbral-json", str(self.umbral), "--permutaciones", "50", *extra]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ae.main(argv), 0)
        return salida, json.loads((salida / "analisis.json").read_text(encoding="utf-8"))

    def test_flujo_completo_sin_figuras_y_contraste_a_mano(self):
        salida, a = self._main("--sin-figuras")
        self.assertTrue((salida / "analisis.md").exists())
        self.assertTrue(a["controles"]["todo_coincide"])
        self.assertNotIn("umbral_vs_n33", a["controles"])   # solo con el LOSO «manos» por defecto
        self.assertIsNone(a["longitud"])
        # top-1 agregado a mano: aciertos si (c + k) % 3 != 0
        aciertos = sum(1 for k in range(5) for c in range(50) if (c + k) % 3)
        self.assertAlmostEqual(a["global"]["top1"], aciertos / 250)
        # McNemar global a mano
        solo_base = sum(1 for k in range(5) for c in range(50) if (c + k) % 3 and not (c + k) % 4)
        solo_otro = sum(1 for k in range(5) for c in range(50) if not (c + k) % 3 and (c + k) % 4)
        d = a["contraste"]["otra"]
        self.assertTrue(d["mismas_muestras"])
        self.assertEqual((d["solo_acierta_manos"], d["solo_acierta_otro"]), (solo_base, solo_otro))
        self.assertAlmostEqual(d["p_mcnemar_global"], ae.mcnemar_exacto(solo_base, solo_otro))
        # por señante coincide con el pliegue guardado
        for p in self.res["pliegues"]:
            self.assertAlmostEqual(a["por_senante"][p["senante_test"]]["top1"], p["top1"])
        # umbral: aceptadas = prob >= 0,2
        n_acept = sum(1 for k in range(5) for c in range(50) if _pred_base(c, k)[1] >= 0.2)
        self.assertEqual(a["calibracion"]["umbral"]["global"]["aceptadas"], n_acept)

    def test_flujo_completo_con_figuras(self):
        salida, a = self._main()
        for f in ("matriz_confusion_normalizada.png", "recall_por_sena.png", "por_senante.png", "fiabilidad.png"):
            self.assertTrue((salida / f).stat().st_size > 0, f)
        self.assertEqual(len(a["figuras"]), 4)

    def _processed_sintetico(self):
        """lengths.npy, metadata.json y X_padded.npy (bloques de mano en ceros) y METRICAS_FOTOGRAMA coherentes."""
        proc, met = self.base / "processed", self.base / "metricas"
        proc.mkdir()
        met.mkdir()
        orden, longitudes = [], []
        T, D = 12, 126
        X = np.zeros((250, T, D), dtype=np.float32)
        for k in range(5):
            for c in range(50):
                m = f"{c:04d}_{k:04d}_0000"
                n = 6 + (c * 5 + k * 3) % 7
                i = len(orden)
                orden.append(m)
                longitudes.append(n)
                X[i, :n, :] = 0.1 + 0.01 * np.arange(n)[:, None]
                sin_izq = (c + 2 * k) % 3            # primeros fotogramas sin mano izquierda
                sin_der = (c * k) % 2                # último fotograma sin mano derecha
                X[i, :sin_izq, 0:63] = 0
                if sin_der:
                    X[i, n - 1, 63:126] = 0
                with open(met / f"{m}.csv", "w", encoding="utf-8") as fh:
                    fh.write("fotograma,mano_izq_detectada,mano_der_detectada\n")
                    for t in range(n):
                        fh.write(f"{t},{0 if t < sin_izq else 1},{0 if (sin_der and t == n - 1) else 1}\n")
        np.save(proc / "lengths.npy", np.array(longitudes))
        np.save(proc / "X_padded.npy", X)
        (proc / "metadata.json").write_text(json.dumps({
            "sample_order": orden,
            "landmark_groups": {"left_hand": {"start_col": 0}, "right_hand": {"start_col": 63}}}), encoding="utf-8")
        return proc, met, dict(zip(orden, longitudes))

    def test_flujo_con_longitudes_y_fotogramas_pruebas_dentro_de_estrato_y_holm(self):
        proc, met, longitudes = self._processed_sintetico()
        salida = self.base / "salida_proc"
        argv = ["--loso-dir", str(self.loso), "--salida", str(salida), "--contraste", f"otra={self.contraste}",
                "--processed", str(proc), "--metricas-fotograma", str(met), "--umbral-json", str(self.umbral),
                "--permutaciones", "300", "--sin-figuras"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ae.main(argv), 0)
        a = json.loads((salida / "analisis.json").read_text(encoding="utf-8"))
        self.assertTrue(a["controles"]["fotogramas_vs_lengths"])
        self.assertTrue(a["controles"]["deteccion_vs_x_padded"])
        lo, nm = a["longitud"], a["evidencia_senantes"]["nivel_muestra"]
        # la misma prueba de longitud en §3 y §6: mismo resultado y p Holm
        for e in ("senante", "sena", "sena_y_senante"):
            t6, t3 = lo["pruebas_longitud"][f"estratificada_por_{e}"], nm["longitud"][f"estratificada_por_{e}"]
            self.assertEqual((t6["diferencia_dentro_de_estratos"], t6["p_bilateral"], t6["p_holm"]),
                             (t3["diferencia_dentro_de_estratos"], t3["p_bilateral"], t3["p_holm"]))
        # a mano: acierto si (c + k) % 3 != 0; diferencia dentro de señante = coeficiente con efectos fijos
        L = np.array([longitudes[f"{c:04d}_{k:04d}_0000"] for k in range(5) for c in range(50)], dtype=float)
        err = np.array([(c + k) % 3 == 0 for k in range(5) for c in range(50)])
        sen = np.repeat(np.arange(5), 50)
        X = np.column_stack([err.astype(float)] + [(sen == k).astype(float) for k in range(5)])
        coef = np.linalg.lstsq(X, L, rcond=None)[0][0]
        self.assertAlmostEqual(lo["pruebas_longitud"]["estratificada_por_senante"]["diferencia_dentro_de_estratos"], coef)
        sena = np.tile(np.arange(50), 5)
        X_sena = np.column_stack([err.astype(float)] + [(sena == c).astype(float) for c in range(50)])
        coef_sena = np.linalg.lstsq(X_sena, L, rcond=None)[0][0]
        t_sena = lo["pruebas_longitud"]["estratificada_por_sena"]
        self.assertAlmostEqual(t_sena["diferencia_dentro_de_estratos"], coef_sena)
        self.assertEqual((t_sena["estratos"], len(t_sena["por_estrato"])), (50, 50))
        self.assertEqual(lo["pruebas_longitud"]["estratificada_por_senante"]["estratos"], 5)
        self.assertAlmostEqual(lo["diferencia_bruta_errores_menos_aciertos"], L[err].mean() - L[~err].mean())
        # verificador r2: desvío |log(L / media de la misma seña en los OTROS señantes)| a mano, dentro de señante
        sena_i = np.tile(np.arange(50), 5)
        desvio = np.array([abs(math.log(L[i] / L[(sena_i == sena_i[i]) & (sen != sen[i])].mean())) for i in range(250)])
        coef_desvio = np.linalg.lstsq(X, desvio, rcond=None)[0][0]
        self.assertAlmostEqual(lo["pruebas_desvio_vs_misma_sena"]["estratificada_por_senante"]["diferencia_dentro_de_estratos"],
                               coef_desvio)
        for k in range(5):
            m = sen == k
            self.assertAlmostEqual(lo["por_senante"][f"{k:04d}"]["diferencia_errores_menos_aciertos"],
                                   L[m & err].mean() - L[m & ~err].mean())
        # una repetición por seña y señante: las celdas no tienen errores y aciertos a la vez → sin prueba ni Holm
        celda = lo["pruebas_longitud"]["estratificada_por_sena_y_senante"]
        self.assertIsNone(celda["p_bilateral"])
        self.assertIsNone(celda["p_holm"])
        # familia de Holm: 4 medidas × (señante, seña) + desvío × (señante, seña) = 10 pruebas con p
        fam = a["pruebas_nivel_muestra"]
        self.assertEqual(fam["n_pruebas"], 10)
        ps = sorted(p["p_bilateral"] for p in fam["pruebas"] if p["p_bilateral"] is not None)
        holm_min = min(p["p_holm"] for p in fam["pruebas"] if p["p_holm"] is not None)
        self.assertAlmostEqual(holm_min, min(1.0, 10 * ps[0]))
        # correlación entre señas a mano
        rec = np.array([np.mean([(c + k) % 3 != 0 for k in range(5)]) for c in range(50)])
        lm = np.array([L[np.arange(250) % 50 == c].mean() for c in range(50)])
        self.assertAlmostEqual(lo["entre_senas"]["pearson"], float(np.corrcoef(lm, rec)[0, 1]))
        md = (salida / "analisis.md").read_text(encoding="utf-8")
        self.assertIn("p Holm", md)
        self.assertIn("diferencia dentro de estrato", md)

    def test_salida_dentro_de_data_se_rechaza(self):
        destino = ROOT_DIR / "data" / "analisis_no_debe_crearse"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ae.main(["--loso-dir", str(self.loso), "--salida", str(destino)])
        self.assertFalse(destino.exists())


if __name__ == "__main__":
    unittest.main()
