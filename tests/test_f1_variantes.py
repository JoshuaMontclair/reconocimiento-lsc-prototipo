"""
Pruebas de src/f1_variantes.py con predicciones sintéticas pequeñas y cálculos a mano.

Lo que se comprueba:
  · métricas macro / ponderadas / micro contra un cálculo a mano y contra sklearn (si está instalado);
  · un caso en el que la media por pliegue y el agregado difieren de forma CONOCIDA (1/3 frente a 1/2) por
    clases con 0 aciertos en un pliegue, y un caso en el que coinciden;
  · la descomposición exacta cierra y reparte la diferencia entre señas con y sin pliegues a cero;
  · el F1 micro es la top-1; el ponderado y el macro coinciden con soportes iguales y difieren con desiguales;
  · el bootstrap es reproducible con la semilla, contiene la estimación puntual y degenera a [1, 1]
    cuando todo se acierta; el remuestreo estratificado conserva el soporte de cada clase;
  · McNemar contra una tabla a mano, y es idéntico para dos sistemas con F1 distintos y los mismos aciertos;
  · el orden entre sistemas bajo una lectura; los cargadores (nombre → seña, pliegue = señante) y sus errores;
  · con los archivos reales presentes (se omite si no están): 0,5374 ± 0,0927 y 0,5737 de N-30 / N-35.

Ejecutar:  python -m unittest tests.test_f1_variantes -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import f1_variantes as fv  # noqa: E402


def _reg(sena: int, senante: int, rep: int, pred: int, pliegue: int | None = None) -> dict:
    return {"muestra": f"{sena:04d}_{senante:04d}_{rep:04d}", "real": sena, "predicha": pred,
            "top3": [pred, (pred + 1) % 50, (pred + 2) % 50], "prob_predicha": 0.5,
            "pliegue": senante if pliegue is None else pliegue, "senante": f"{senante:04d}", "repeticion": rep}


def _caso_difieren() -> list[dict]:
    """Dos pliegues, dos clases (0 y 1), dos muestras por clase y pliegue.
    Pliegue 0: clase 0 acierta ambas, clase 1 falla ambas (predice 0).
    Pliegue 1: clase 0 falla ambas (predice 1), clase 1 acierta ambas.
    Por pliegue: clase acertada P = 0,5, R = 1 → F1 = 2/3; clase fallada F1 = 0 → macro (sobre 2 clases) 1/3.
    Agregado: cada clase TP = 2, FP = 2, FN = 2 → F1 = 4/8 = 0,5 → macro 0,5."""
    return [_reg(0, 0, 0, 0), _reg(0, 0, 1, 0), _reg(1, 0, 0, 0), _reg(1, 0, 1, 0),
            _reg(0, 1, 0, 1), _reg(0, 1, 1, 1), _reg(1, 1, 0, 1), _reg(1, 1, 1, 1)]


def _caso_coinciden() -> list[dict]:
    """Dos pliegues con el MISMO patrón de confusión: clase 0 acierta 1 de 2 (la otra va a 1), clase 1
    acierta ambas. Por pliegue: clase 0 TP1 FP0 FN1 → F1 2/3; clase 1 TP2 FP1 FN0 → F1 4/5; macro 11/15.
    Agregado: conteos dobles, mismas razones → 11/15."""
    return [_reg(0, 0, 0, 0), _reg(0, 0, 1, 1), _reg(1, 0, 0, 1), _reg(1, 0, 1, 1),
            _reg(0, 1, 0, 0), _reg(0, 1, 1, 1), _reg(1, 1, 0, 1), _reg(1, 1, 1, 1)]


class PruebaMetricas(unittest.TestCase):
    def test_conteos_y_prf_a_mano(self):
        reales = [0, 0, 1, 1, 2]
        pred = [0, 1, 1, 1, 0]
        tp, fp, fn = fv.conteos(reales, pred, n=3)
        np.testing.assert_array_equal(tp, [1, 2, 0])
        np.testing.assert_array_equal(fp, [1, 1, 0])
        np.testing.assert_array_equal(fn, [1, 0, 1])
        pc = fv.prf_por_clase(tp, fp, fn)
        self.assertAlmostEqual(pc["precision"][0], 0.5)
        self.assertAlmostEqual(pc["recall"][0], 0.5)
        self.assertAlmostEqual(pc["f1"][0], 0.5)
        self.assertAlmostEqual(pc["precision"][1], 2 / 3)
        self.assertAlmostEqual(pc["f1"][1], 0.8)
        self.assertEqual(pc["precision"][2], 0.0)   # nunca predicha: 0/0 → 0
        self.assertEqual(pc["f1"][2], 0.0)

    def test_metricas_macro_ponderado_micro(self):
        reales = [0, 0, 0, 1, 2, 2]      # soportes 3, 1, 2
        pred = [0, 0, 1, 1, 2, 0]
        m = fv.metricas(reales, pred, n=3)
        # clase 0: TP2 FP1 FN1 → P 2/3 R 2/3 F1 2/3; clase 1: TP1 FP1 FN0 → P 1/2 R 1 F1 2/3;
        # clase 2: TP1 FP0 FN1 → P 1 R 1/2 F1 2/3
        self.assertAlmostEqual(m["f1_macro"], 2 / 3)
        self.assertAlmostEqual(m["f1_ponderado"], (3 * 2 / 3 + 1 * 2 / 3 + 2 * 2 / 3) / 6)
        self.assertAlmostEqual(m["precision_macro"], (2 / 3 + 1 / 2 + 1) / 3)
        self.assertAlmostEqual(m["recall_macro"], (2 / 3 + 1 + 1 / 2) / 3)
        self.assertAlmostEqual(m["precision_ponderada"], (3 * 2 / 3 + 1 * 1 / 2 + 2 * 1) / 6)
        self.assertAlmostEqual(m["f1_micro"], 4 / 6)
        self.assertAlmostEqual(m["top1"], 4 / 6)
        self.assertEqual(m["n"], 6)

    def test_micro_es_top1_y_ponderado_difiere_del_macro_con_soportes_desiguales(self):
        rng = np.random.default_rng(3)
        reales = np.repeat(np.arange(4), [10, 5, 2, 1])
        pred = np.where(rng.random(len(reales)) < 0.6, reales, rng.integers(0, 4, len(reales)))
        m = fv.metricas(reales, pred, n=4)
        self.assertAlmostEqual(m["f1_micro"], float(np.mean(reales == pred)))
        self.assertAlmostEqual(m["precision_micro"], m["f1_micro"])
        self.assertNotAlmostEqual(m["f1_macro"], m["f1_ponderado"], places=3)

    def test_ponderado_igual_al_macro_con_soportes_iguales(self):
        reales = [0, 0, 1, 1, 2, 2]
        pred = [0, 1, 1, 1, 2, 0]
        m = fv.metricas(reales, pred, n=3)
        self.assertAlmostEqual(m["f1_macro"], m["f1_ponderado"])

    def test_coincide_con_sklearn(self):
        try:
            from sklearn.metrics import precision_recall_fscore_support
        except ImportError:
            self.skipTest("sklearn no instalado")
        rng = np.random.default_rng(11)
        reales = rng.integers(0, 50, 997)
        pred = np.where(rng.random(997) < 0.5, reales, rng.integers(0, 50, 997))
        m = fv.metricas(reales, pred)
        for avg, claves in (("macro", ("precision_macro", "recall_macro", "f1_macro")),
                            ("weighted", ("precision_ponderada", "recall_ponderado", "f1_ponderado")),
                            ("micro", ("precision_micro", "recall_micro", "f1_micro"))):
            p, r, f, _ = precision_recall_fscore_support(reales, pred, labels=list(range(50)), average=avg, zero_division=0)
            self.assertAlmostEqual(m[claves[0]], float(p), places=12)
            self.assertAlmostEqual(m[claves[1]], float(r), places=12)
            self.assertAlmostEqual(m[claves[2]], float(f), places=12)

    def test_etiqueta_fuera_de_rango(self):
        with self.assertRaises(ValueError):
            fv.conteos([0, 3], [0, 0], n=3)


class PruebaLecturas(unittest.TestCase):
    def test_caso_en_que_difieren_de_forma_conocida(self):
        regs = _caso_difieren()
        lp = fv.lectura_por_pliegue(regs, n=2)
        ag = fv.lectura_agregada(regs, n=2)
        self.assertEqual(lp["n_pliegues"], 2)
        self.assertEqual(lp["n_por_pliegue"], [4, 4])
        for m in lp["pliegues"]:
            self.assertAlmostEqual(m["f1_macro"], 1 / 3)
            self.assertEqual(m["clases_con_f1_0"], 1)
        self.assertAlmostEqual(lp["resumen"]["f1_macro"]["media"], 1 / 3)
        self.assertAlmostEqual(lp["resumen"]["f1_macro"]["de"], 0.0)
        self.assertAlmostEqual(ag["f1_macro"], 0.5)
        self.assertAlmostEqual(ag["top1"], 0.5)
        self.assertGreater(ag["f1_macro"], lp["resumen"]["f1_macro"]["media"])

    def test_caso_en_que_coinciden(self):
        regs = _caso_coinciden()
        lp = fv.lectura_por_pliegue(regs, n=2)
        ag = fv.lectura_agregada(regs, n=2)
        self.assertAlmostEqual(lp["resumen"]["f1_macro"]["media"], 11 / 15)
        self.assertAlmostEqual(ag["f1_macro"], 11 / 15)
        d = fv.descomposicion(regs, n=2)
        self.assertAlmostEqual(d["diferencia_agregado_menos_media"], 0.0)
        self.assertEqual(d["pares_pliegue_sena_con_0_aciertos"], 0)

    def test_descomposicion_cierra_y_atribuye_a_las_senas_con_pliegue_a_cero(self):
        d = fv.descomposicion(_caso_difieren(), n=2)
        self.assertTrue(d["identidad_cierra"])
        self.assertAlmostEqual(d["media_por_pliegue"], 1 / 3)
        self.assertAlmostEqual(d["agregado"], 0.5)
        self.assertAlmostEqual(d["diferencia_agregado_menos_media"], 1 / 6)
        self.assertEqual(d["pares_pliegue_sena_con_0_aciertos"], 2)
        self.assertEqual(d["senas_con_algun_pliegue_sin_aciertos"], 2)
        self.assertAlmostEqual(d["aporte_senas_con_algun_pliegue_sin_aciertos"], 1 / 6)
        self.assertAlmostEqual(d["aporte_senas_sin_pliegues_sin_aciertos"], 0.0)
        # Pesos del agregado: pliegue con aciertos D = 2·2+2+0 = 6, pliegue a cero D = 0+0+2 = 2 → 3/4 y 1/4.
        s0 = d["por_sena"][0]
        self.assertAlmostEqual(sorted(s0["pesos_agregado"])[0], 0.25)
        self.assertAlmostEqual(sorted(s0["pesos_agregado"])[1], 0.75)
        self.assertAlmostEqual(sum(s["aporte"] for s in d["por_sena"]), 1 / 6)

    def test_convencion_0_0_mueve_la_precision_y_no_el_f1(self):
        # Pliegue 0 nunca predice la seña 1 y el pliegue 1 nunca predice la 0: dos pares con precisión 0/0.
        d = fv.descomposicion(_caso_difieren(), n=2)
        self.assertEqual(d["soporte_minimo_pliegue_sena"], 2)
        self.assertEqual(d["pares_pliegue_sena_sin_muestras"], 0)
        self.assertEqual(d["pares_pliegue_sena_nunca_predichos"], 2)
        pm = d["precision_macro_media_pliegues"]
        self.assertAlmostEqual(pm["con_0_0_igual_a_0"], 0.25)          # (0,5 + 0) / 2 en cada pliegue
        self.assertAlmostEqual(pm["sin_las_senas_nunca_predichas"], 0.5)
        lp = fv.lectura_por_pliegue(_caso_difieren(), n=2)
        self.assertAlmostEqual(lp["resumen"]["precision_macro"]["media"], pm["con_0_0_igual_a_0"])
        # El F1 de la seña fallada es 0 con cualquier convención porque FN > 0.
        for s in d["por_sena"]:
            self.assertEqual(min(s["f1_por_pliegue"]), 0.0)
        d2 = fv.descomposicion(_caso_coinciden(), n=2)
        self.assertEqual(d2["pares_pliegue_sena_nunca_predichos"], 0)
        self.assertAlmostEqual(d2["precision_macro_media_pliegues"]["diferencia"], 0.0)

    def test_descomposicion_cierra_en_datos_aleatorios_con_50_clases(self):
        rng = np.random.default_rng(2026)
        regs = []
        for senante in range(5):
            for sena in range(50):
                for rep in range(4):
                    pred = sena if rng.random() < 0.5 else int(rng.integers(0, 50))
                    regs.append(_reg(sena, senante, rep, pred))
        d = fv.descomposicion(regs)
        lp = fv.lectura_por_pliegue(regs)
        ag = fv.lectura_agregada(regs)
        self.assertAlmostEqual(d["media_por_pliegue"], lp["resumen"]["f1_macro"]["media"], places=12)
        self.assertAlmostEqual(d["agregado"], ag["f1_macro"], places=12)
        self.assertAlmostEqual(d["aporte_senas_con_algun_pliegue_sin_aciertos"] + d["aporte_senas_sin_pliegues_sin_aciertos"],
                               d["diferencia_agregado_menos_media"], places=12)
        self.assertGreater(d["pares_pliegue_sena_con_0_aciertos"], 0)

    def test_media_ponderada_por_n(self):
        regs = _caso_difieren() + [_reg(0, 1, 2, 0)]   # el pliegue 1 tiene 5 muestras, el 0 tiene 4
        lp = fv.lectura_por_pliegue(regs, n=2)
        f = lp["resumen"]["f1_macro"]
        esperada = (4 * f["por_pliegue"][0] + 5 * f["por_pliegue"][1]) / 9
        self.assertAlmostEqual(f["media_ponderada_por_n"], esperada)


class PruebaBootstrap(unittest.TestCase):
    def _regs(self, semilla=5, p=0.6):
        rng = np.random.default_rng(semilla)
        regs = []
        for senante in range(5):
            for sena in range(50):
                for rep in range(4):
                    pred = sena if rng.random() < p else int(rng.integers(0, 50))
                    regs.append(_reg(sena, senante, rep, pred))
        return regs

    def test_remuestreo_estratificado_conserva_soportes(self):
        reales = np.repeat(np.arange(5), [4, 3, 4, 4, 2])
        rng = np.random.default_rng(0)
        for _ in range(20):
            idx = fv._indices_estratificados(reales, rng)
            self.assertEqual(len(idx), len(reales))
            np.testing.assert_array_equal(np.bincount(reales[idx], minlength=5), [4, 3, 4, 4, 2])

    def test_reproducible_y_contiene_la_estimacion(self):
        regs = self._regs()
        a = fv.bootstrap_agregado(regs, n_boot=50, semilla=2026)
        b = fv.bootstrap_agregado(regs, n_boot=50, semilla=2026)
        self.assertEqual(a["ic_inferior"], b["ic_inferior"])
        self.assertEqual(a["ic_superior"], b["ic_superior"])
        self.assertLessEqual(a["ic_inferior"], a["estimacion_puntual"])
        self.assertGreaterEqual(a["ic_superior"], a["estimacion_puntual"])
        c = fv.bootstrap_agregado(regs, n_boot=50, semilla=1)
        self.assertNotEqual(a["ic_inferior"], c["ic_inferior"])

    def test_media_pliegues_tres_intervalos(self):
        regs = self._regs()
        r = fv.bootstrap_media_pliegues(regs, n_boot=40, semilla=2026)
        self.assertEqual(r["n_pliegues"], 5)
        for clave in ("pliegues_y_dentro", "solo_pliegues", "intervalo_t"):
            self.assertLessEqual(r[clave]["ic_inferior"], r["estimacion_puntual"])
            self.assertGreaterEqual(r[clave]["ic_superior"], r["estimacion_puntual"])
        self.assertAlmostEqual(r["intervalo_t"]["t_critico"], 2.776, places=3)   # t_{4; 0,975}
        lp = fv.lectura_por_pliegue(regs)
        self.assertAlmostEqual(r["estimacion_puntual"], lp["resumen"]["f1_macro"]["media"])
        self.assertAlmostEqual(r["de_entre_pliegues"], lp["resumen"]["f1_macro"]["de"])

    def test_todo_correcto_da_intervalo_degenerado(self):
        regs = self._regs(p=1.0)
        a = fv.bootstrap_agregado(regs, n_boot=10, semilla=2026)
        self.assertEqual((a["ic_inferior"], a["ic_superior"]), (1.0, 1.0))
        r = fv.bootstrap_media_pliegues(regs, n_boot=10, semilla=2026)
        self.assertEqual((r["pliegues_y_dentro"]["ic_inferior"], r["pliegues_y_dentro"]["ic_superior"]), (1.0, 1.0))
        self.assertAlmostEqual(r["intervalo_t"]["ic_inferior"], 1.0)


class PruebaMcNemarYOrden(unittest.TestCase):
    def test_mcnemar_tabla_a_mano(self):
        # posiciones: ambos aciertan 0, 6, 8, 9; solo A 1, 2; solo B 4, 7; fallan ambos 3, 5
        a = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1], dtype=bool)
        b = np.array([1, 0, 0, 0, 1, 0, 1, 1, 1, 1], dtype=bool)
        m = fv.mcnemar(a, b)
        self.assertEqual(m["tabla"], {"aciertan_ambos": 4, "solo_a": 2, "solo_b": 2, "fallan_ambos": 2})
        self.assertEqual(m["prueba_usada"], "exacta binomial")
        self.assertAlmostEqual(m["p"], 1.0)
        self.assertFalse(m["significativo"])
        # Con n10 + n01 ≥ 25 usa chi2: n10 = 30, n01 = 5 → (|30−5|−1)² / 35 = 16,457
        a2 = np.array([1] * 30 + [0] * 5 + [1] * 10, dtype=bool)
        b2 = np.array([0] * 30 + [1] * 5 + [1] * 10, dtype=bool)
        m2 = fv.mcnemar(a2, b2)
        self.assertEqual(m2["prueba_usada"], "chi2 con corrección de continuidad")
        self.assertAlmostEqual(m2["chi2_con_correccion_de_continuidad"], 24 ** 2 / 35)
        self.assertTrue(m2["significativo"])

    def test_mcnemar_no_depende_del_f1(self):
        """Dos sistemas B y B' con exactamente los mismos aciertos pero errores repartidos de otra forma:
        F1 macro distinto, McNemar contra A idéntico."""
        regs_a = _caso_difieren()
        # B: acierta las mismas muestras que A' pero, cuando falla, predice la clase 1 en vez de la 0.
        regs_b = [dict(r) for r in regs_a]
        regs_b2 = [dict(r) for r in regs_a]
        for r in regs_b2:
            if r["real"] != r["predicha"]:
                r["predicha"] = 1 - r["predicha"] if r["real"] != 1 - r["predicha"] else r["predicha"]
        # sistema de referencia: acierta todo salvo la primera muestra
        regs_ref = [dict(r, predicha=r["real"]) for r in regs_a]
        regs_ref[0]["predicha"] = 1 - regs_ref[0]["real"]
        f1_b = fv.lectura_agregada(regs_b, n=2)["f1_macro"]
        f1_b2 = fv.lectura_agregada(regs_b2, n=2)["f1_macro"]
        aciertos_b = [r["real"] == r["predicha"] for r in regs_b]
        aciertos_b2 = [r["real"] == r["predicha"] for r in regs_b2]
        self.assertEqual(aciertos_b, aciertos_b2)
        m1 = fv.mcnemar(*fv.alinear(regs_ref, regs_b))
        m2 = fv.mcnemar(*fv.alinear(regs_ref, regs_b2))
        self.assertEqual(m1["tabla"], m2["tabla"])
        self.assertEqual(m1["p"], m2["p"])
        # y ninguna clave del resultado de McNemar contiene un F1
        self.assertFalse(any("f1" in k.lower() for k in m1))
        # (con dos clases, redirigir un error a la otra clase lo convierte en acierto o lo deja igual;
        # la prueba de fondo es que la tabla solo lee aciertos: se refuerza con 50 clases abajo)
        del f1_b, f1_b2

    def test_mcnemar_no_depende_del_f1_con_50_clases(self):
        rng = np.random.default_rng(9)
        regs_ref, regs_b, regs_b2 = [], [], []
        for senante in range(5):
            for sena in range(50):
                for rep in range(4):
                    acierta_ref = rng.random() < 0.6
                    acierta_b = rng.random() < 0.5
                    otra = int((sena + 1 + rng.integers(0, 48)) % 50)   # cualquier clase distinta de la real
                    otra2 = int((sena + 25) % 50)                         # error concentrado en una sola clase
                    regs_ref.append(_reg(sena, senante, rep, sena if acierta_ref else otra))
                    regs_b.append(_reg(sena, senante, rep, sena if acierta_b else otra))
                    regs_b2.append(_reg(sena, senante, rep, sena if acierta_b else otra2))
        f1_b = fv.lectura_agregada(regs_b)["f1_macro"]
        f1_b2 = fv.lectura_agregada(regs_b2)["f1_macro"]
        self.assertNotAlmostEqual(f1_b, f1_b2, places=3)
        m1 = fv.mcnemar(*fv.alinear(regs_ref, regs_b))
        m2 = fv.mcnemar(*fv.alinear(regs_ref, regs_b2))
        self.assertEqual(m1["tabla"], m2["tabla"])
        self.assertEqual(m1["p"], m2["p"])

    def test_alinear_exige_las_mismas_muestras(self):
        with self.assertRaises(ValueError):
            fv.alinear(_caso_difieren(), _caso_difieren()[:-1])

    def test_orden_entre_sistemas(self):
        v = {"red_manos": 0.5, "red_manos_postura": 0.4, "red_manos_postura_rostro": 0.45, "linea_base_random_forest": 0.3}
        o = fv.orden_entre_sistemas(v)
        self.assertEqual(o["ranking"], ["red_manos", "red_manos_postura_rostro", "red_manos_postura", "linea_base_random_forest"])
        self.assertFalse(o["todos_se_cumplen"])
        fallido = [p for p in o["pares"] if not p["se_cumple"]]
        self.assertEqual(len(fallido), 1)
        self.assertEqual((fallido[0]["mayor"], fallido[0]["menor"]), ("red_manos_postura", "red_manos_postura_rostro"))
        v["red_manos_postura"] = 0.46
        self.assertTrue(fv.orden_entre_sistemas(v)["todos_se_cumplen"])


class PruebaCarga(unittest.TestCase):
    def test_parsear_muestra(self):
        self.assertEqual(fv.parsear_muestra("0007_0003_0002"), (7, "0003", 2))
        with self.assertRaises(ValueError):
            fv.parsear_muestra("7_3_2")

    def test_cargar_loso_plano_asigna_pliegue_por_senante_y_valida(self):
        filas = [{"muestra": f"{s:04d}_{v:04d}_0000", "real": s, "predicha": s, "top3": [s, 0, 1], "prob_predicha": 0.9}
                 for v in range(5) for s in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "p.json"
            ruta.write_text(json.dumps(filas), encoding="utf-8")
            regs = fv.cargar_loso_plano(ruta)
            self.assertEqual(sorted({r["pliegue"] for r in regs}), [0, 1, 2, 3, 4])
            self.assertTrue(all(r["pliegue"] == int(r["senante"]) for r in regs))
            filas[0]["real"] = 2      # la seña del nombre ya no coincide
            ruta.write_text(json.dumps(filas), encoding="utf-8")
            with self.assertRaises(ValueError):
                fv.cargar_loso_plano(ruta)
            ruta.write_text(json.dumps([f for f in filas[1:] if f["muestra"][5:9] != "0004"]), encoding="utf-8")
            with self.assertRaises(ValueError):   # falta un señante
                fv.cargar_loso_plano(ruta)

    def test_cargar_loso_pliegues_comprueba_senante_fuera(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "resultados_loso.json").write_text(json.dumps({"pliegues": [{"pliegue": 0, "senante_test": "0000"},
                                                                              {"pliegue": 1, "senante_test": "0001"}]}), encoding="utf-8")
            for k in range(2):
                (d / f"pliegue_{k}").mkdir()
                filas = [{"muestra": f"{s:04d}_{k:04d}_0000", "real": s, "predicha": s, "top3": [s, 0, 1], "prob_predicha": 0.9}
                         for s in range(3)]
                (d / f"pliegue_{k}" / "predicciones.json").write_text(json.dumps(filas), encoding="utf-8")
            regs = fv.cargar_loso_pliegues(d)
            self.assertEqual(len(regs), 6)
            self.assertEqual({r["pliegue"] for r in regs}, {0, 1})
            malo = [{"muestra": "0000_0001_0000", "real": 0, "predicha": 0, "top3": [0, 1, 2], "prob_predicha": 0.9}]
            (d / "pliegue_0" / "predicciones.json").write_text(json.dumps(malo), encoding="utf-8")
            with self.assertRaises(ValueError):
                fv.cargar_loso_pliegues(d)

    def test_ejecutar_de_extremo_a_extremo_con_sistemas_sinteticos(self):
        """Dos sistemas planos sintéticos en una carpeta temporal; el flujo completo escribe JSON y MD."""
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "m").mkdir()
            sistemas = []
            for clave, p in (("red_manos", 0.7), ("linea_base_random_forest", 0.4)):
                filas = []
                for v in range(5):
                    for s in range(50):
                        for rep in range(4):
                            pred = s if rng.random() < p else int(rng.integers(0, 50))
                            filas.append({"muestra": f"{s:04d}_{v:04d}_{rep:04d}", "real": s, "predicha": pred,
                                          "top3": [pred, 0, 1], "prob_predicha": 0.5})
                (base / "m" / f"{clave}.json").write_text(json.dumps(filas), encoding="utf-8")
                sistemas.append({"clave": clave, "nombre": clave, "loso_plano": f"m/{clave}.json"})
            # sin resultados.json de la línea base, controles_linea_base devuelve «no disponible»
            res = fv.ejecutar(sistemas, n_boot=15, semilla=2026, base=base)
            self.assertIn("red_manos_vs_linea_base_random_forest", res["mcnemar"])
            self.assertTrue(res["orden"]["f1_macro_media_pliegues"]["todos_se_cumplen"])
            self.assertTrue(res["orden"]["f1_macro_agregado"]["todos_se_cumplen"])
            self.assertFalse(res["sistemas"]["linea_base_random_forest"]["loso"]["controles"]["disponible"])
            fv.escribir_md(res, base / "salida.md")
            texto = (base / "salida.md").read_text(encoding="utf-8")
            self.assertIn("F1 macro, media ± DE entre pliegues", texto)
            json.dumps(res)   # serializable


@unittest.skipUnless((ROOT_DIR / "models" / "loso_2026-09-15_holistic_mascara_manos" / "pliegue_4" / "predicciones.json").exists(),
                     "predicciones reales del LOSO «manos» no disponibles")
class PruebaDatosReales(unittest.TestCase):
    """Reproduce las dos cifras de N-30 y N-35 desde las predicciones guardadas (solo lectura)."""

    def test_reproduce_n30_y_n35(self):
        regs = fv.cargar_loso_pliegues(ROOT_DIR / "models" / "loso_2026-09-15_holistic_mascara_manos")
        self.assertEqual(len(regs), 997)
        lp = fv.lectura_por_pliegue(regs)
        ag = fv.lectura_agregada(regs)
        self.assertAlmostEqual(lp["resumen"]["f1_macro"]["media"], 0.5374, places=3)   # N-30
        self.assertAlmostEqual(lp["resumen"]["f1_macro"]["de"], 0.0927, places=3)
        self.assertAlmostEqual(ag["f1_macro"], 0.5737, places=3)                        # N-35
        self.assertAlmostEqual(lp["resumen"]["top1"]["media"], 0.5797, places=3)
        self.assertAlmostEqual(ag["f1_micro"], 0.5797, places=3)
        d = fv.descomposicion(regs)
        self.assertTrue(d["identidad_cierra"])
        self.assertAlmostEqual(d["diferencia_agregado_menos_media"], ag["f1_macro"] - lp["resumen"]["f1_macro"]["media"], places=12)


if __name__ == "__main__":
    unittest.main()


class TestHuecosDeLaVerificacion(unittest.TestCase):
    """Cierra las tres mutaciones que la verificación de N-49 dejó sin detectar."""

    def test_bootstrap_agregado_es_estratificado_por_sena(self):
        # Cada seña se predice siempre igual dentro de sí misma: con remuestreo estratificado cada réplica
        # conserva el soporte y el F1 no varía (IC degenerado). Sin estratificar, el soporte cambia y varía.
        reales = [0, 0, 0, 1, 1, 2]
        predichas = [0, 0, 0, 2, 2, 2]
        regs = [{"real": r, "predicha": p} for r, p in zip(reales, predichas)]
        out = fv.bootstrap_agregado(regs, n_boot=200, semilla=7, n=3)
        self.assertAlmostEqual(out["de_bootstrap"], 0.0, places=12)
        self.assertAlmostEqual(out["ic_inferior"], out["estimacion_puntual"], places=12)
        self.assertAlmostEqual(out["ic_superior"], out["estimacion_puntual"], places=12)

    def test_percentiles_son_del_95(self):
        out = fv._percentiles(np.arange(1001, dtype=float))
        self.assertAlmostEqual(out["ic_inferior"], 25.0)
        self.assertAlmostEqual(out["ic_superior"], 975.0)

    def test_empate_no_cumple_el_orden(self):
        esperados = [("a", "b")]
        res = fv.orden_entre_sistemas({"a": 0.5, "b": 0.5}, esperados)
        self.assertFalse(res["pares"][0]["se_cumple"])
        self.assertFalse(res["todos_se_cumplen"])
