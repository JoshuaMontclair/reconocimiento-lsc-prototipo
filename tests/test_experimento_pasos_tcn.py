"""
Pruebas del experimento de 30 frente a 60 pasos con la TCN (src/experimento_pasos_tcn.py;
docs/EXPERIMENTO_PASOS_TCN.md). Rápidas: no entrenan ni importan TensorFlow.
Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_experimento_pasos_tcn -v
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import experimento_pasos_tcn as ep  # noqa: E402
from src import arquitecturas_literatura as al  # noqa: E402
from src import mano_ausente_variantes as mav  # noqa: E402

BASE = [0.96, 0.94, 0.94, 0.97, 0.94]
LOSS = [0.90, 1.01, 1.05, 0.92, 1.01]


def _m(por_pliegue, ruido=(0.0, 0.0, 0.0)):
    base = np.asarray(por_pliegue, float)
    return np.stack([base + r for r in ruido])


def _crit(val60=BASE, loss60=LOSS, val30=BASE, loss30=LOSS, ruido=(0.0, 0.0, 0.0)):
    return ep.aplicar_criterio({"P30": _m(val30, ruido), "P60": _m(val60, ruido)},
                               {"P30": _m(loss30), "P60": _m(loss60)})


class TestDiseno(unittest.TestCase):
    def test_brazos(self):
        self.assertEqual({b: i["pasos"] for b, i in ep.BRAZOS.items()}, {"P30": 30, "P60": 60})
        self.assertEqual(ep.REGLA, "imputacion")
        self.assertEqual(ep.ORDEN_SIMPLICIDAD[0], "P30")

    def test_p30_reutiliza_t1(self):
        d = ep.dir_corrida("/o", "P30", 2027, 3)
        self.assertIn("confirmacion_tcn_2026-10-03/T1_imputacion_252/semilla_2027/pliegue_3", str(d))
        self.assertEqual(str(ep.dir_corrida("/o", "P30", 2027, 3, reutilizar_p30=False)),
                         "/o/P30_imputacion_252/semilla_2027/pliegue_3")
        self.assertEqual(str(ep.dir_corrida("/o", "P60", 2026, 0)), "/o/P60_imputacion_252/semilla_2026/pliegue_0")

    def test_config_t_sin_cambios(self):
        self.assertEqual(al.CONFIG_T, {"filtros": 64, "ventana": 3, "dilataciones": [1, 2, 4, 8],
                                       "convs_por_bloque": 2, "dropout_bloque": 0.2, "dropout_final": 0.3,
                                       "padding": "same"})

    def test_campo_receptivo(self):
        self.assertEqual(ep.campo_receptivo(), 61)
        self.assertEqual(ep.posiciones_con_ventana_completa(30), 30)
        self.assertEqual(ep.posiciones_con_ventana_completa(60), 2)
        self.assertEqual(ep.posiciones_con_ventana_completa(61), 1)
        self.assertEqual(ep.posiciones_con_ventana_completa(62), 0)

    def test_leer_reutilizar_p30(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertTrue(ep.leer_reutilizar_p30(t))
            (Path(t) / "verificacion.json").write_text(json.dumps({"reutilizar_P30": False}), encoding="utf-8")
            self.assertFalse(ep.leer_reutilizar_p30(t))


class TestEntrada(unittest.TestCase):
    def test_imputacion_a_60_pasos(self):
        rng = np.random.default_rng(0)
        seq = rng.standard_normal((45, 126)).astype(np.float32)
        seq[5:12, 0:63] = 0.0
        X = mav.remuestrear_todas_con_regla(seq[None], [45], "imputacion", 60)
        self.assertEqual(X.shape, (1, 60, 126))
        np.testing.assert_array_equal(X[0], mav.regla_imputacion(seq, 60))
        self.assertFalse(np.any(np.all(X[0, :, 0:63] == 0, axis=1)))   # la mano rellenada no queda en cero
        np.testing.assert_allclose(X[0, 0], mav.imputar_fotogramas(seq)[0])
        np.testing.assert_allclose(X[0, -1], mav.imputar_fotogramas(seq)[-1])

    def test_misma_regla_con_30_y_60(self):
        """Los dos brazos salen de la misma secuencia imputada; solo cambia la rejilla del remuestreo."""
        rng = np.random.default_rng(1)
        seq = rng.standard_normal((61, 126)).astype(np.float32)
        seq[20:30, 63:126] = 0.0
        x30 = mav.regla_imputacion(seq, 30)
        x60 = mav.regla_imputacion(seq, 60)
        # Los extremos de las dos rejillas caen en el primer y el último cuadro de la misma secuencia imputada.
        np.testing.assert_allclose(x30[0], x60[0])
        np.testing.assert_allclose(x30[-1], x60[-1])
        self.assertEqual(x60.shape, (60, 126))


class TestReproduccion(unittest.TestCase):
    def _comp(self, byte, coincide):
        return {"predicciones_json_identico_byte_a_byte": byte, "coincide": coincide}

    def test_condicion_previa(self):
        self.assertTrue(ep.coincide_reproduccion(self._comp(True, True))["reutilizar_P30"])
        self.assertTrue(ep.coincide_reproduccion(self._comp(False, True))["reutilizar_P30"])
        self.assertFalse(ep.coincide_reproduccion(self._comp(False, False))["reutilizar_P30"])


class TestCriterio(unittest.TestCase):
    def test_empate_total_queda_30_por_simplicidad(self):
        r = _crit()
        self.assertEqual(r["configuracion_final"], "P30")
        self.assertEqual(r["decide_en_regla"], 3)
        self.assertAlmostEqual(r["margen"], 0.01)

    def test_60_supera_en_validacion(self):
        r = _crit(val60=[b + 0.02 for b in BASE])
        self.assertEqual((r["configuracion_final"], r["decide_en_regla"]), ("P60", 1))

    def test_30_supera_en_validacion(self):
        r = _crit(val60=[b - 0.02 for b in BASE])
        self.assertEqual((r["configuracion_final"], r["decide_en_regla"]), ("P30", 1))

    def test_margen_exacto_no_basta(self):
        """Δ medio = margen (0,01) no es «mayor que el margen»: pasa a la regla 2 y luego a la 3."""
        r = _crit(val60=[b + 0.01 for b in BASE])
        self.assertFalse(r["regla1"]["P60_supera_a_P30"]["cumple"])
        self.assertEqual((r["configuracion_final"], r["decide_en_regla"]), ("P30", 3))

    def test_solo_3_de_5_pliegues_no_basta(self):
        r = _crit(val60=[BASE[0] + 0.05, BASE[1] + 0.05, BASE[2] + 0.05, BASE[3], BASE[4]])
        self.assertTrue(r["regla1"]["P60_supera_a_P30"]["a_delta_mayor_que_margen"])
        self.assertFalse(r["regla1"]["P60_supera_a_P30"]["b_mejor_en_4_de_5"])
        self.assertEqual(r["configuracion_final"], "P30")

    def test_desempate_por_val_loss_a_favor_de_60(self):
        r = _crit(loss60=[x - 0.05 for x in LOSS])
        self.assertEqual((r["configuracion_final"], r["decide_en_regla"]), ("P60", 2))

    def test_desempate_por_val_loss_a_favor_de_30(self):
        r = _crit(loss60=[x + 0.05 for x in LOSS])
        self.assertEqual((r["configuracion_final"], r["decide_en_regla"]), ("P30", 2))

    def test_val_loss_dentro_de_sigma_no_desempata(self):
        """Con mucha dispersión de val_loss entre semillas, una reducción pequeña no supera σ_loss."""
        val = {"P30": _m(BASE), "P60": _m(BASE)}
        loss = {"P30": _m(LOSS, (-0.1, 0.0, 0.1)), "P60": _m([x - 0.02 for x in LOSS], (-0.1, 0.0, 0.1))}
        r = ep.aplicar_criterio(val, loss)
        self.assertGreater(r["sigma_val_loss_agrupada"], 0.02)
        self.assertEqual((r["configuracion_final"], r["decide_en_regla"]), ("P30", 3))

    def test_margen_sube_con_sigma(self):
        ruido = (-0.03, 0.0, 0.03)
        r = _crit(val60=[b + 0.02 for b in BASE], ruido=ruido)
        self.assertAlmostEqual(r["margen"], 0.03)
        self.assertFalse(r["regla1"]["P60_supera_a_P30"]["cumple"])

    def test_tolerancia_float32(self):
        """Diferencias de ~1e-8 (float32 de Keras) cuentan como empate, no como pliegue mejor."""
        r = _crit(val60=[b + 1e-8 for b in BASE])
        self.assertEqual(r["regla1"]["P60_supera_a_P30"]["pliegues_mejor"], 0)


class TestCriterioCongelado(unittest.TestCase):
    def test_copia_del_criterio_intacta(self):
        d = ROOT_DIR / "models" / "pasos_tcn_2026-11-23"
        copia, firma = d / "criterio_prerregistrado.md", d / "criterio_prerregistrado.sha256"
        if not (copia.exists() and firma.exists()):
            self.skipTest("sin la carpeta del experimento")
        esperado = firma.read_text(encoding="utf-8").split()[0]
        self.assertEqual(hashlib.sha256(copia.read_bytes()).hexdigest(), esperado)

    def test_seccion_1_del_informe_igual_a_la_copia(self):
        d = ROOT_DIR / "models" / "pasos_tcn_2026-11-23" / "criterio_prerregistrado.md"
        informe = ROOT_DIR / "docs" / "EXPERIMENTO_PASOS_TCN.md"
        if not (d.exists() and informe.exists()):
            self.skipTest("sin la carpeta del experimento")
        copia = d.read_text(encoding="utf-8")
        inicio = copia.index("## 1. Pregunta")
        self.assertIn(copia[inicio:].rstrip("\n"), informe.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
