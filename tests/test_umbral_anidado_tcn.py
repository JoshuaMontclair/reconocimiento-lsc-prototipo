"""
Pruebas de la validación anidada del umbral con el modelo final (src/umbral_anidado_tcn.py; D-14).
Rápidas y sin TensorFlow: no se entrena nada. Las que necesitan el corpus procesado o los archivos de
N-47 / N-65 / T1 se omiten si no están.
    python -m unittest tests.test_umbral_anidado_tcn -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import umbral_anidado_tcn as uat  # noqa: E402
from src import umbral_anidado as ua  # noqa: E402
from src import umbral_confianza as uc  # noqa: E402
from src import experimento_confirmacion_tcn as ect  # noqa: E402
from src import experimento_arquitecturas as ea  # noqa: E402
from src import arquitecturas_literatura as al  # noqa: E402
from src import mano_ausente_variantes as mav  # noqa: E402
from src import train_loso  # noqa: E402

PROCESSED = ROOT_DIR / "data" / "processed_holistic_mascara_rostro"
HAY_CORPUS = (PROCESSED / "metadata.json").exists() and (PROCESSED / "y_labels.npy").exists()
HAY_N47 = (ROOT_DIR / uat.PARTICION_N47).exists()
HAY_T1 = (ROOT_DIR / uat.T1_DIR / "semilla_2026" / "pliegue_0" / "predicciones.json").exists()
HAY_N65 = (ROOT_DIR / uat.INFORME_N65).exists()
SENANTES = ["0000", "0001", "0002", "0003", "0004"]


def add_velocity_copia(X):
    """Copia literal de train_final_100.add_velocity (para no importar TensorFlow)."""
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, vel], axis=-1)


class TestMismoSelectorYMaquinaria(unittest.TestCase):

    def test_selector_es_el_mismo_objeto_que_D09(self):
        self.assertIs(uat.elegir_umbral, uc.elegir_umbral)
        self.assertIs(uat.utilidad, uc.utilidad)
        self.assertIs(uat.efecto, uc.efecto)
        self.assertIs(uat.REJILLA, uc.REJILLA)
        self.assertIs(uat.elegir_umbral, ua.elegir_umbral)

    def test_particion_agregacion_y_efecto_son_los_de_N47(self):
        self.assertIs(uat.construir_particion, ua.construir_particion)
        self.assertIs(uat.elegir_umbrales, ua.elegir_umbrales)
        self.assertIs(uat.efecto_prueba_loso, ua.efecto_prueba_loso)
        self.assertIs(uat.resumen, ua.resumen)
        self.assertEqual(uat.SEMILLA_PARTICION, 2026)
        self.assertEqual(uat.C_PRINCIPAL, 1.0)
        self.assertEqual(tuple(uat.C_SENSIBILIDAD), (0.5, 1.0, 2.0))


class TestEntradaYRecetaDeT1(unittest.TestCase):

    def test_brazo_regla_y_rasgos(self):
        self.assertEqual(uat.BRAZO, "T1")
        self.assertEqual(uat.REGLA, "imputacion")
        self.assertEqual(uat.N_RASGOS, 252)
        self.assertEqual(uat.SEQ_LENGTH, 30)
        self.assertEqual(ect.BRAZOS["T1"]["forma"], "actual")

    def test_constructor_es_la_tcn_con_config_t(self):
        self.assertEqual(uat.CONSTRUCTOR, "T")
        _, hp = ea.constructor(uat.CONSTRUCTOR)
        self.assertEqual(hp, dict(al.ENTRENAMIENTO_COMUN, arquitectura=al.CONFIG_T))
        self.assertEqual(hp, ect.hiperparametros_t())

    def test_funcion_de_entrada_es_la_de_canon_de_brazo_T1(self):
        sentinela = object()
        canon = ("tf", "build", sentinela, "resample", "transf")
        t = uat.canon_t1(canon)
        self.assertIs(t[2], sentinela)                       # add_velocity del canon, sin envolver
        self.assertEqual(t, ect.canon_de_brazo(canon, "T1"))

    def test_preparar_t1_toma_la_entrada_imputacion_de_la_confirmacion(self):
        rng = np.random.default_rng(0)
        X_padded = rng.standard_normal((40, 50, 126)).astype(np.float32)
        X_padded[:, :, :63][:, 10:20] = 0.0                  # mano izquierda ausente en un tramo
        lengths = np.full(40, 50)
        X_imp = mav.remuestrear_todas_con_regla(X_padded, lengths, "imputacion", 30)
        canon = ("tf", "build", "addv", "resample", "transf")
        falso = ({"mascara": np.zeros_like(X_imp), "imputacion": X_imp}, "y", "nombres", "sen", "loso", "pos",
                 canon, (X_padded, lengths))
        with mock.patch.object(ect, "preparar", return_value=falso) as m:
            X_seq, *_rest, canon_b = uat.preparar_t1(Path("/no/importa"))
        m.assert_called_once()
        self.assertIs(X_seq, X_imp)
        self.assertEqual(canon_b, ect.canon_de_brazo(canon, "T1"))

    def test_preparar_t1_se_detiene_si_la_entrada_no_es_imputacion(self):
        rng = np.random.default_rng(1)
        X_padded = rng.standard_normal((40, 50, 126)).astype(np.float32)
        lengths = np.full(40, 50)
        falso = ({"imputacion": np.zeros((40, 30, 126), np.float32)}, "y", "n", "s", "l", "p",
                 ("tf", "b", "a", "r", "t"), (X_padded, lengths))
        with mock.patch.object(ect, "preparar", return_value=falso):
            with self.assertRaises(SystemExit):
                uat.preparar_t1(Path("/no/importa"))

    @unittest.skipUnless(HAY_CORPUS, "sin el processed canonico")
    def test_entrada_bit_a_bit_igual_a_la_de_T1_en_una_muestra(self):
        X_padded, y, lengths, nombres, _ = train_loso.cargar_datos(PROCESSED, "manos")
        muestra = [nombres.index(n) for n in ("0030_0000_0003", "0015_0001_0000") if n in nombres] + [0, 501, 996]
        X = uat.entrada_posiciones(X_padded[muestra], lengths[muestra])
        for j, i in enumerate(muestra):
            # Lo que hace experimento_confirmacion_tcn.preparar para T1 (remuestrear_todas_con_regla) video a video.
            esperado = mav.regla_imputacion(X_padded[i, :int(lengths[i])], 30)
            self.assertTrue(np.array_equal(X[j], esperado), nombres[i])
        E = add_velocity_copia(X)
        self.assertEqual(E.shape[1:], (30, 252))
        self.assertTrue(np.array_equal(E[:, :, :126], X))

    def test_pliegue_de_par_calibra_con_los_dos_excluidos(self):
        par = {"par": "par_0000_0001", "a": "0000", "b": "0001",
               "conjuntos": {"train": ["0000_0002_0000"], "val": ["0000_0003_0000"],
                             "umbral": {"0000": ["0000_0000_0000"], "0001": ["0000_0001_0000"]}}}
        p = uat.pliegue_de_par(par, 3)
        self.assertEqual(p["pliegue"], 3)
        self.assertEqual(p["conjuntos"]["test"], ["0000_0000_0000", "0000_0001_0000"])
        self.assertEqual(p["conjuntos"]["train"], ["0000_0002_0000"])
        self.assertEqual(p["conjuntos"]["val"], ["0000_0003_0000"])


class TestParticionesIgualesANuevoYN47(unittest.TestCase):

    @unittest.skipUnless(HAY_CORPUS and HAY_N47, "sin corpus o sin particiones de N-47")
    def test_particiones_regeneradas_coinciden_con_las_de_N47(self):
        loso = json.loads(train_loso.ARCHIVO_LOSO.read_text(encoding="utf-8"))
        nombres, y, senantes = ua.cargar_nombres(PROCESSED)
        part = uat.construir_particion(nombres, y, senantes, loso, uat.SEMILLA_PARTICION)
        comp = uat.comparar_con_n47(part)
        self.assertTrue(comp["coincide"], comp)
        previa = json.loads((ROOT_DIR / uat.PARTICION_N47).read_text(encoding="utf-8"))
        for p_new, p_old in zip(part["pares"], previa["pares"]):
            self.assertEqual(p_new["conjuntos"], p_old["conjuntos"])
            self.assertEqual(p_new["semilla_validacion"], p_old["semilla_validacion"])

    @unittest.skipUnless(HAY_N47, "sin particiones de N-47")
    def test_comparar_detecta_una_particion_distinta(self):
        previa = json.loads((ROOT_DIR / uat.PARTICION_N47).read_text(encoding="utf-8"))
        cambiada = json.loads(json.dumps(previa))
        a, b = cambiada["pares"][0]["conjuntos"]["train"][0], cambiada["pares"][0]["conjuntos"]["val"][0]
        cambiada["pares"][0]["conjuntos"]["train"][0], cambiada["pares"][0]["conjuntos"]["val"][0] = b, a
        self.assertFalse(uat.comparar_con_n47(cambiada)["coincide"])
        self.assertTrue(uat.comparar_con_n47(previa)["coincide"])


class TestReglaYBootstrap(unittest.TestCase):

    CONTROL_OK = {"0.00": {"coincide": True}, "0.50": {"coincide": True}}

    def test_adoptar_si_U_sube_y_bajan_los_errores(self):
        r = uat.aplicar_regla(self.CONTROL_OK, {"utilidad": 400, "aceptadas_incorrectas": 200},
                              {"utilidad": 375, "aceptadas_incorrectas": 311})
        self.assertTrue(r["ii_U_P1_mayor_o_igual"] and r["iii_menos_errores_aceptados"])
        self.assertTrue(r["recomendacion"].startswith("adoptar"))

    def test_quedarse_si_U_baja_aunque_bajen_los_errores(self):
        r = uat.aplicar_regla(self.CONTROL_OK, {"utilidad": 360, "aceptadas_incorrectas": 100},
                              {"utilidad": 375, "aceptadas_incorrectas": 311})
        self.assertTrue(r["recomendacion"].startswith("quedarse"))

    def test_empate_dentro_de_5(self):
        r = uat.aplicar_regla(self.CONTROL_OK, {"utilidad": 380, "aceptadas_incorrectas": 250},
                              {"utilidad": 375, "aceptadas_incorrectas": 311})
        self.assertTrue(r["empate_dentro_de_5"])
        self.assertIn("no distinguen", r["recomendacion"])

    def test_control_fallido_detiene(self):
        r = uat.aplicar_regla({"0.00": {"coincide": False}}, {"utilidad": 500, "aceptadas_incorrectas": 1},
                              {"utilidad": 375, "aceptadas_incorrectas": 311})
        self.assertTrue(r["recomendacion"].startswith("detenerse"))

    def test_bootstrap_observado_es_la_diferencia_de_U(self):
        rng = np.random.default_rng(3)
        ok = rng.random(300) < 0.6
        conf = rng.random(300)
        a1, a0 = conf >= 0.5, conf >= 0.0
        b = uat.bootstrap_diferencia_u(ok, a1, a0, n=500)
        esperado = uc.utilidad(conf, ok, 0.5, 1.0) - uc.utilidad(conf, ok, 0.0, 1.0)
        self.assertAlmostEqual(b["diferencia_observada"], esperado)
        self.assertLessEqual(b["ic95"][0], b["ic95"][1])
        self.assertEqual(b, uat.bootstrap_diferencia_u(ok, a1, a0, n=500))   # reproducible con la semilla


class TestControlN65(unittest.TestCase):

    @unittest.skipUnless(HAY_T1 and HAY_N65, "sin predicciones de T1 o sin informe de N-65")
    def test_prueba_T1_reproduce_las_cifras_de_N65(self):
        loso = json.loads(train_loso.ARCHIVO_LOSO.read_text(encoding="utf-8"))
        prueba = uat.cargar_prueba_t1(ROOT_DIR / uat.T1_DIR, loso, 2026)
        self.assertEqual(sum(len(v[0]) for v in prueba.values()), 997)
        c = uat.control_n65(prueba)
        self.assertTrue(all(v["coincide"] for v in c.values()), c)
        self.assertEqual(c["0.00"]["obtenido"], {"aceptadas_correctas": 686, "aceptadas_incorrectas": 311, "utilidad": 375.0})

    @unittest.skipUnless(HAY_N65, "sin informe de N-65")
    def test_control_falla_con_otras_predicciones(self):
        rng = np.random.default_rng(5)
        prueba = {k: (rng.random(200), rng.random(200) < 0.5) for k in range(5)}
        c = uat.control_n65(prueba)
        self.assertFalse(all(v["coincide"] for v in c.values()))


class TestSalidaProtegida(unittest.TestCase):

    def test_no_escribe_en_carpetas_vigentes(self):
        for prohibida in (uat.T1_DIR, "models/tcn_imputacion", "models/umbral_anidado_2026-09-22/x", "data/tmp"):
            with self.assertRaises(SystemExit):
                uat.main(["--construir", "--out", prohibida])


if __name__ == "__main__":
    unittest.main()
