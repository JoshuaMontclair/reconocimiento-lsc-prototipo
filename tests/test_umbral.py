"""
Pruebas del umbral de confianza (D-09): criterio de elección en src/umbral_confianza.py y uso por
defecto en el prototipo (<modelo>.umbral.json). Sin TensorFlow ni modelos reales.
    python -m unittest tests.test_umbral -v
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

from src import umbral_confianza as uc  # noqa: E402


class TestCriterio(unittest.TestCase):

    def test_utilidad_y_aceptacion_inclusiva(self):
        conf = np.array([0.1, 0.2, 0.3, 0.9])
        ok = np.array([False, True, False, True])
        # t = 0,2 acepta 0,2 (>=), 0,3 y 0,9: 2 aciertos y 1 error
        self.assertEqual(uc.utilidad(conf, ok, 0.2, 1.0), 1.0)
        self.assertEqual(uc.utilidad(conf, ok, 0.2, 2.0), 0.0)
        self.assertEqual(uc.utilidad(conf, ok, 0.0, 1.0), 0.0)

    def test_elige_el_menor_t_con_utilidad_maxima(self):
        conf = np.array([0.05, 0.40, 0.45, 0.80, 0.90])
        ok = np.array([False, True, True, True, True])
        t, curva = uc.elegir_umbral(conf, ok, 1.0)
        # quitar el error de 0,05 da U = 4 para todo t en (0,05; 0,40]: el menor es 0,06
        self.assertAlmostEqual(t, 0.06)
        self.assertEqual(max(curva), 4.0)
        self.assertEqual(len(curva), len(uc.REJILLA))

    def test_costo_mayor_no_baja_el_umbral(self):
        rng = np.random.default_rng(0)
        conf = rng.uniform(size=400)
        ok = rng.uniform(size=400) < conf          # más confianza, más probabilidad de acierto
        t1, _ = uc.elegir_umbral(conf, ok, 1.0)
        t2, _ = uc.elegir_umbral(conf, ok, 2.0)
        self.assertGreaterEqual(t2, t1)

    def test_efecto(self):
        conf = np.array([0.1, 0.3, 0.6, 0.9])
        ok = np.array([True, False, True, True])
        e = uc.efecto(conf, ok, 0.5)
        self.assertEqual(e["tasa_rechazo"], 0.5)
        self.assertEqual(e["exactitud_entre_aceptadas"], 1.0)
        self.assertAlmostEqual(e["aciertos_perdidos_por_rechazo"], 1 / 3)
        self.assertEqual(e["errores_evitados_por_rechazo"], 1.0)
        self.assertEqual((e["aceptadas_correctas"], e["aceptadas_incorrectas"]), (2, 0))


class TestUmbralEnElPrototipo(unittest.TestCase):

    def _traductor(self, model_path, **kw):
        sys.path.insert(0, str(ROOT_DIR / "tests"))
        from tests.test_prototipo import ExtractorFalso, ModeloFalso
        from src.translator import LSCTranslator
        with contextlib.redirect_stdout(io.StringIO()):
            return LSCTranslator(modelo=ModeloFalso(450), model_path=model_path, extractor=ExtractorFalso(),
                                 tts=None, tts_mode="none", registro=False, calentar=False, **kw)

    def test_umbral_del_modelo_por_defecto_y_sobrescritura(self):
        with tempfile.TemporaryDirectory() as tmp:
            modelo = Path(tmp) / "m.keras"
            (Path(tmp) / "m.umbral.json").write_text(json.dumps({"umbral": 0.2, "decision": "D-09"}), encoding="utf-8")
            tr = self._traductor(str(modelo))
            self.assertEqual(tr.umbral, 0.2)
            self.assertTrue(tr.umbral_origen.endswith("m.umbral.json"))
            tr = self._traductor(str(modelo), umbral=0.7)
            self.assertEqual((tr.umbral, tr.umbral_origen), (0.7, "parametro"))

    def test_huella_de_modelo_distinta_avisa(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            modelo = Path(tmp) / "m.keras"
            modelo.write_bytes(b"modelo-nuevo")
            (Path(tmp) / "m.umbral.json").write_text(json.dumps(
                {"umbral": 0.2, "sha256_modelo": hashlib.sha256(b"modelo-viejo").hexdigest()}), encoding="utf-8")
            salida = io.StringIO()
            with contextlib.redirect_stdout(salida):
                tr = self._traductor(str(modelo))
            self.assertEqual(tr.umbral, 0.2)
            self.assertIn("sha256", tr.umbral_origen)
            (Path(tmp) / "m.umbral.json").write_text(json.dumps(
                {"umbral": 0.2, "sha256_modelo": hashlib.sha256(b"modelo-nuevo").hexdigest()}), encoding="utf-8")
            tr = self._traductor(str(modelo))
            self.assertNotIn("sha256", tr.umbral_origen)

    def test_sin_archivo_de_umbral_usa_0_5(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = self._traductor(str(Path(tmp) / "sin_umbral.keras"))
            self.assertEqual((tr.umbral, tr.umbral_origen), (0.5, "defecto_0.5"))

    def test_confianza_igual_al_umbral_se_acepta_como_en_el_criterio(self):
        # Revisión: el prototipo acepta con confianza >= umbral, la misma comparación que utilidad();
        # 0,5 y 0,25 son exactos en float32 y float32(0,2) queda por encima de 0,2 en float64.
        sys.path.insert(0, str(ROOT_DIR / "tests"))
        from tests.test_prototipo import construir, correr_ventana
        with tempfile.TemporaryDirectory() as tmp:
            for p in (0.5, 0.25, 0.2):
                reg = correr_ventana(construir(tmp, n_rasgos=450, confianza=p, umbral=p, registro=False))
                self.assertEqual(reg["estado"], "aceptada", p)
                t = uc.REJILLA[int(round(p * 100))]
                self.assertEqual(uc.utilidad(np.array([p], dtype=np.float32), np.array([True]), t, 1.0), 1.0)
            reg = correr_ventana(construir(tmp, n_rasgos=450, confianza=0.25, umbral=0.26, registro=False))
            self.assertEqual(reg["estado"], "rechazada_confianza")

    def test_umbral_real_del_canonico(self):
        from src.representacion import leer_umbral
        ruta = ROOT_DIR / "models" / "canonico_holistic_mascara_manos" / "modelo_canonico.keras"
        info = leer_umbral(ruta)
        if info is None:
            self.skipTest("el canónico «manos» no tiene umbral calculado")
        self.assertTrue(0.0 <= info["umbral"] < 1.0)
        self.assertEqual(info["decision"], "D-09")


if __name__ == "__main__":
    unittest.main(verbosity=2)
