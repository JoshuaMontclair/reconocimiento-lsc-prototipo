"""
Pruebas del análisis de señas cortadas (src/analisis_senas_cortadas.py) con clips sintéticos:
la regla detecta un clip truncado y no marca uno que empieza y termina en reposo; el recorte artificial
respeta las proporciones; los umbrales se ajustan solo con el entrenamiento; la selección de (K, rasgo)
no mira la prueba. Sin TensorFlow, sin modelos y sin tocar data/ ni models/.
    python -m unittest tests.test_analisis_senas_cortadas -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import analisis_senas_cortadas as sc  # noqa: E402


def clip_sintetico(T: int = 60, reposo_ini: int = 12, reposo_fin: int = 12, amplitud: float = 0.08,
                   ruido: float = 0.0, semilla: int = 0, y_reposo: float = 2.8, y_sena: float = 0.6) -> np.ndarray:
    """Clip (T, 225) en el espacio normalizado por hombros: postura con hombros en (±0,5; 0), muñecas de
    la postura y bloques de mano que descansan quietos en y_reposo, suben a y_sena y se mueven `amplitud`
    por fotograma durante la seña, y vuelven a reposo. Las manos están detectadas en todos los fotogramas."""
    rng = np.random.default_rng(semilla)
    seq = np.zeros((T, 225), dtype=np.float64)
    seq[:, 11 * 3:11 * 3 + 2] = (0.5, 0.0)      # hombro izq
    seq[:, 12 * 3:12 * 3 + 2] = (-0.5, 0.0)     # hombro der
    seq[:, 0:2] = (0.0, -1.5)                   # nariz: la postura nunca es toda ceros
    x_izq, x_der = 0.6, -0.6
    fase = 0.0
    for t in range(T):
        en_sena = reposo_ini <= t < T - reposo_fin
        if en_sena:
            fase += amplitud
            y = y_sena
            dx = 0.3 * np.sin(fase / 0.3 * 2)
        else:
            y = y_reposo
            dx = 0.0
        for lado, x0, c_pose, c_mano in (("izq", x_izq, 15 * 3, 99), ("der", x_der, 16 * 3, 162)):
            px, py = x0 + dx, y + rng.normal(0, ruido)
            seq[t, c_pose:c_pose + 2] = (px, py)
            pts = np.stack([np.full(21, px) + np.linspace(0, 0.2, 21), np.full(21, py) - np.linspace(0, 0.2, 21),
                            np.zeros(21)], axis=1)
            seq[t, c_mano:c_mano + 63] = pts.reshape(-1)
    return seq


class TestRecorte(unittest.TestCase):

    def test_respeta_las_proporciones(self):
        seq = np.arange(60 * 225, dtype=np.float64).reshape(60, 225)
        for p in sc.PROPORCIONES:
            n = int(round(p * 60))
            ini = sc.recortar(seq, p, "inicio")
            fin = sc.recortar(seq, p, "fin")
            self.assertEqual(ini.shape[0], 60 - n)
            self.assertEqual(fin.shape[0], 60 - n)
            np.testing.assert_array_equal(ini, seq[n:])          # se va el inicio
            np.testing.assert_array_equal(fin, seq[:60 - n])     # se va el final

    def test_redondeo_y_minimo(self):
        seq = np.zeros((41, 225))
        self.assertEqual(sc.recortar(seq, 0.35, "inicio").shape[0], 41 - round(0.35 * 41))
        self.assertEqual(sc.recortar(np.zeros((3, 225)), 0.9, "fin").shape[0], 2)   # nunca menos de 2
        with self.assertRaises(ValueError):
            sc.recortar(seq, 0.2, "centro")

    def test_versiones(self):
        seq = np.zeros((40, 225))
        v = sc.versiones_de(seq)
        self.assertEqual(list(v), ["integro", "inicio_0.20", "inicio_0.35", "inicio_0.50", "fin_0.20", "fin_0.35", "fin_0.50"])
        self.assertEqual(v["inicio_0.50"].shape[0], 20)


class TestRasgos(unittest.TestCase):

    def test_reposo_en_los_bordes_y_movimiento_en_el_centro(self):
        seq = clip_sintetico()
        r = sc.rasgos_clip(seq, 5)
        self.assertAlmostEqual(r["v_pose_ini"], 0.0)
        self.assertAlmostEqual(r["v_pose_fin"], 0.0)
        self.assertGreater(r["v_pose_centro"], 0.05)
        self.assertGreater(r["v_manos_centro"], 0.05)
        self.assertAlmostEqual(r["h_ini"], 2.8)       # muñecas abajo (y positivo) en reposo
        self.assertAlmostEqual(r["h_fin"], 2.8)
        self.assertLess(r["h_centro"], 1.0)           # más arriba durante la seña
        self.assertEqual(r["sin_mano_ini"], 0.0)
        self.assertEqual(r["T"], 60)

    def test_clip_truncado_tiene_bordes_en_movimiento(self):
        seq = clip_sintetico()
        ini = sc.rasgos_clip(sc.recortar(seq, 0.35, "inicio"), 5)
        fin = sc.rasgos_clip(sc.recortar(seq, 0.35, "fin"), 5)
        self.assertGreater(ini["v_pose_ini"], 0.05)
        self.assertAlmostEqual(ini["v_pose_fin"], 0.0)
        self.assertGreater(fin["v_pose_fin"], 0.05)
        self.assertAlmostEqual(fin["v_pose_ini"], 0.0)
        self.assertLess(ini["h_ini"], 1.0)

    def test_mano_ausente_no_cuenta_en_v_manos_y_no_rompe_v_pose(self):
        seq = clip_sintetico()
        seq[:8, 99:162] = 0.0      # mano izquierda ausente al inicio
        seq[:8, 162:225] = 0.0     # y la derecha también
        r = sc.rasgos_clip(seq, 5)
        self.assertTrue(np.isnan(r["v_manos_ini"]))      # ningún par con mano: indefinido
        self.assertAlmostEqual(r["v_pose_ini"], 0.0)     # la postura sigue definida
        self.assertEqual(r["sin_mano_ini"], 1.0)

    def test_sin_postura_da_nan(self):
        seq = clip_sintetico()
        seq[:3, :] = 0.0
        r = sc.rasgos_clip(seq, 3)
        self.assertTrue(np.isnan(r["v_pose_ini"]))
        self.assertTrue(np.isnan(r["h_ini"]))


def _rasgos_de(clips: list[np.ndarray], Ks=sc.KS) -> dict:
    out = {v: {K: {r: np.full(len(clips), np.nan) for r in sc.NOMBRES_RASGOS} for K in Ks} for v in sc.VERSIONES}
    for i, seq in enumerate(clips):
        for v, s in sc.versiones_de(seq).items():
            for K in Ks:
                r = sc.rasgos_clip(s, K)
                for nombre in sc.NOMBRES_RASGOS:
                    out[v][K][nombre][i] = r[nombre]
    return out


class TestReglas(unittest.TestCase):

    def setUp(self):
        # 20 clips íntegros con reposo en los bordes y un temblor leve en reposo; los 12 primeros son el
        # «entrenamiento». El temblor de la prueba (0,001) es menor que el del entrenamiento (0,003), de
        # modo que el percentil 95 del entrenamiento cubre los valores de reposo de la prueba: un reposo
        # normal no debe marcarse, y lo que se comprueba es que la regla separa reposo de movimiento.
        self.clips = [clip_sintetico(T=50 + i, ruido=0.003 if i < 12 else 0.001, semilla=i,
                                     reposo_ini=10 + i % 3, reposo_fin=10 + i % 2) for i in range(20)]
        self.rasgos = _rasgos_de(self.clips)
        self.idx_train = np.arange(12)
        self.idx_test = np.arange(12, 20)

    def test_detecta_truncado_y_no_marca_integro(self):
        u = sc.ajustar_umbrales(self.rasgos["integro"][5], self.idx_train, 95)
        e = sc.evaluar_regla(self.rasgos, 5, "v_pose", u, self.idx_test)
        self.assertEqual(e["fp_integros"], 0.0)                       # ningún íntegro de prueba marcado
        for v in ("inicio_0.35", "inicio_0.50", "fin_0.35", "fin_0.50"):
            self.assertEqual(e["deteccion"][v], 1.0, v)               # todos los truncados detectados
            self.assertEqual(e["deteccion_lado"][v], 1.0, v)          # y por el borde correcto
        # el 20 % de estos clips sintéticos quita todo el reposo (10-12 fot. de 50-69): también se detecta
        self.assertEqual(e["deteccion"]["inicio_0.20"], 1.0)
        m = sc.aplicar_regla(self.rasgos["inicio_0.35"][5], u, "v_pose")
        self.assertTrue(m["cortada_ini"].all())
        # el final sigue en reposo: no se marca en la prueba (en el entrenamiento el percentil 95 deja,
        # por construcción, algún clip por encima de su propio umbral)
        self.assertFalse(m["cortada_fin"][self.idx_test].any())

    def test_altura_y_combinacion(self):
        u = sc.ajustar_umbrales(self.rasgos["integro"][5], self.idx_train, 95)
        for rasgo in ("h", "v_pose_o_h", "v_manos"):
            e = sc.evaluar_regla(self.rasgos, 5, rasgo, u, self.idx_test)
            self.assertEqual(e["fp_integros"], 0.0, rasgo)
            self.assertEqual(e["deteccion"]["inicio_0.50"], 1.0, rasgo)
            self.assertEqual(e["deteccion"]["fin_0.50"], 1.0, rasgo)

    def test_nan_no_dispara(self):
        u = sc.ajustar_umbrales(self.rasgos["integro"][5], self.idx_train, 95)
        R = {k: v.copy() for k, v in self.rasgos["integro"][5].items()}
        R["v_pose_ini"][:] = np.nan
        R["v_pose_fin"][:] = np.nan
        m = sc.aplicar_regla(R, u, "v_pose")
        self.assertFalse(m["cortada"].any())

    def test_umbrales_solo_con_entrenamiento(self):
        R = self.rasgos["integro"][5]
        u1 = sc.ajustar_umbrales(R, self.idx_train, 95)
        # Alterar de forma grosera a los clips de PRUEBA no cambia ningún umbral.
        R2 = {k: v.copy() for k, v in R.items()}
        R2["v_pose_ini"][self.idx_test] = 99.0
        R2["v_pose_fin"][self.idx_test] = 99.0
        R2["h_ini"][self.idx_test] = -99.0
        R2["v_manos_ini"][self.idx_test] = 99.0
        u2 = sc.ajustar_umbrales(R2, self.idx_train, 95)
        for rasgo in ("v_pose", "v_manos", "h"):
            for borde in ("ini", "fin"):
                self.assertEqual(u1[rasgo][borde], u2[rasgo][borde], (rasgo, borde))
        # y alterar al entrenamiento sí los cambia
        R3 = {k: v.copy() for k, v in R.items()}
        R3["v_pose_ini"][self.idx_train] = 99.0
        self.assertNotEqual(sc.ajustar_umbrales(R3, self.idx_train, 95)["v_pose"]["ini"], u1["v_pose"]["ini"])
        self.assertEqual(u1["n_train"], 12)
        self.assertEqual(u1["percentil"], 95)
        with self.assertRaises(ValueError):
            sc.ajustar_umbrales(R, np.array([], dtype=int), 95)

    def test_percentil_es_el_declarado(self):
        R = self.rasgos["integro"][5]
        u = sc.ajustar_umbrales(R, self.idx_train, 95)
        self.assertAlmostEqual(u["v_pose"]["ini"], float(np.nanpercentile(R["v_pose_ini"][self.idx_train], 95)))
        self.assertAlmostEqual(u["h"]["fin"], float(np.nanpercentile(R["h_fin"][self.idx_train], 5)))
        u99 = sc.ajustar_umbrales(R, self.idx_train, 99)
        self.assertGreaterEqual(u99["v_pose"]["ini"], u["v_pose"]["ini"])

    def test_seleccion_no_mira_la_prueba(self):
        elegida, rejilla = sc.seleccionar_config(self.rasgos, self.idx_train)
        self.assertEqual(len(rejilla), len(sc.KS) * len(sc.RASGOS))
        self.assertIn("train", rejilla[0])
        self.assertNotIn("test", rejilla[0])
        # Con la prueba alterada la elección y la rejilla de entrenamiento son idénticas.
        alterado = {v: {K: {r: a.copy() for r, a in d.items()} for K, d in dK.items()} for v, dK in self.rasgos.items()}
        for v in sc.VERSIONES:
            for K in sc.KS:
                for r in ("v_pose_ini", "v_pose_fin", "v_manos_ini", "v_manos_fin"):
                    alterado[v][K][r][self.idx_test] = 0.0
                for r in ("h_ini", "h_fin"):
                    alterado[v][K][r][self.idx_test] = 5.0
        elegida2, rejilla2 = sc.seleccionar_config(alterado, self.idx_train)
        self.assertEqual((elegida["K"], elegida["rasgo"]), (elegida2["K"], elegida2["rasgo"]))
        for a, b in zip(rejilla, rejilla2):
            self.assertEqual(a["train"]["deteccion_media"], b["train"]["deteccion_media"])
            self.assertEqual(a["umbrales"]["v_pose"], b["umbrales"]["v_pose"])

    def test_desempate_menor_K_y_orden_de_rasgos(self):
        # En estos clips sintéticos todas las configuraciones detectan el 100 %: el desempate elige K = 3, v_pose.
        elegida, _ = sc.seleccionar_config(self.rasgos, self.idx_train)
        self.assertEqual((elegida["K"], elegida["rasgo"]), (3, "v_pose"))


class TestParticion(unittest.TestCase):

    def test_conjuntos_pliegue_separa_al_senante(self):
        particion = {"pliegues": [{"senante_test": "0001",
                                   "conjuntos": {"train": ["0000_0000_0000", "0000_0002_0001"], "val": ["0001_0000_0000"],
                                                 "test": ["0000_0001_0000", "0003_0001_0002"]}}]}
        entren, prueba, s = sc.conjuntos_pliegue(particion, 0)
        self.assertEqual(s, "0001")
        self.assertEqual(entren, ["0000_0000_0000", "0000_0002_0001", "0001_0000_0000"])
        self.assertEqual(prueba, ["0000_0001_0000", "0003_0001_0002"])
        malo = {"pliegues": [{"senante_test": "0001",
                              "conjuntos": {"train": ["0000_0001_0000"], "val": [], "test": ["0003_0001_0002"]}}]}
        with self.assertRaises(ValueError):
            sc.conjuntos_pliegue(malo, 0)


class TestMetricas(unittest.TestCase):

    def test_metricas_prediccion_a_mano(self):
        prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.15, 0.75], [0.4, 0.35, 0.25], [0.19, 0.01, 0.8]])
        y = np.array([0, 2, 1, 0])
        m = sc.metricas_prediccion(prob, y, umbral=0.5)
        self.assertEqual(m["top1"], 0.5)                 # aciertan la 0 y la 1
        self.assertEqual(m["rechazo_umbral"], 0.25)      # la 2 (0,4) queda bajo 0,5
        self.assertAlmostEqual(m["exactitud_aceptadas"], 2 / 3)
        self.assertEqual((m["aceptadas_ok"], m["aceptadas_mal"]), (2, 1))
        self.assertEqual(m["top3"], 1.0)

    def test_entrada_modelo_columnas_manos_y_velocidades(self):
        def resample(seq, target):
            idx = np.linspace(0, seq.shape[0] - 1, target)
            return np.stack([np.interp(idx, np.arange(seq.shape[0]), seq[:, f]) for f in range(seq.shape[1])], axis=1).astype(np.float32)

        def add_velocity(X):
            vel = np.zeros_like(X)
            vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
            return np.concatenate([X, vel], axis=-1)

        seq = np.zeros((10, 225), dtype=np.float32)
        seq[:, 99] = np.arange(10)          # primera columna de la mano izquierda
        seq[:, 0] = 100.0                   # la postura no debe entrar
        X = sc.entrada_modelo([seq], resample, add_velocity, seq_length=5)
        self.assertEqual(X.shape, (1, 5, 252))
        np.testing.assert_allclose(X[0, :, 0], np.linspace(0, 9, 5))
        self.assertFalse((X == 100.0).any())
        np.testing.assert_allclose(X[0, 1:, 126], np.diff(np.linspace(0, 9, 5)))

    def test_resumen(self):
        r = sc.resumen(np.array([1.0, np.nan, 3.0, 2.0]))
        self.assertEqual(r["n"], 3)
        self.assertEqual(r["mediana"], 2.0)
        self.assertEqual(sc.resumen(np.array([np.nan]))["n"], 0)


class TestPorBordeYConversion(unittest.TestCase):

    def test_combinar_por_borde(self):
        # 2 clips de prueba: el 0 cortado al inicio (mueve en el borde inicial), el 1 íntegro.
        integro = clip_sintetico(semilla=1)
        cortado = sc.recortar(integro, 0.35, "inicio")
        rasgos = {v: {3: {r: np.zeros(2) for r in sc.NOMBRES_RASGOS}} for v in sc.VERSIONES}
        for v in sc.VERSIONES:
            for i, s in enumerate((cortado, integro)):
                r = sc.rasgos_clip(s, 3)
                for nombre in sc.NOMBRES_RASGOS:
                    rasgos[v][3][nombre][i] = r[nombre]
        u = {"v_pose": {"ini": 0.01, "fin": 0.01}, "v_manos": {"ini": 0.01, "fin": 0.01}, "h": {"ini": 1.0, "fin": 1.0}}
        modelo_res = {"confianza": {v: np.array([0.9, 0.9]) for v in sc.VERSIONES},
                      "acierto": {v: np.array([False, True]) for v in sc.VERSIONES}}
        ambos = sc.combinar_regla_y_umbral(modelo_res, rasgos, 3, "v_pose_o_h", u, np.array([0, 1]))
        ini = sc.combinar_regla_y_umbral(modelo_res, rasgos, 3, "v_pose_o_h", u, np.array([0, 1]), borde="ini")
        fin = sc.combinar_regla_y_umbral(modelo_res, rasgos, 3, "v_pose_o_h", u, np.array([0, 1]), borde="fin")
        self.assertEqual(ambos["integro"]["rechaza_regla"], 0.5)      # solo el cortado
        self.assertEqual(ini["integro"]["rechaza_regla"], 0.5)
        self.assertEqual(fin["integro"]["rechaza_regla"], 0.0)        # su final está en reposo
        self.assertEqual(ini["integro"]["errores_en_voz_alta_con_regla"], 0)
        self.assertEqual(ini["integro"]["aciertos_en_voz_alta_con_regla"], 1)
        with self.assertRaises(ValueError):
            sc.combinar_regla_y_umbral(modelo_res, rasgos, 3, "v_pose_o_h", u, np.array([0, 1]), borde="medio")

    def test_utilidad(self):
        self.assertEqual(sc.utilidad(540, 297), 243.0)                # cifras de N-33 con t = 0,20
        self.assertEqual(sc.utilidad(10, 4, c=2), 2.0)

    def test_conversion_a_segundos_y_a_otra_tasa(self):
        u = {"v_pose": {"ini": 0.0734, "fin": 0.0096}, "v_manos": {"ini": 0.12, "fin": 0.03}, "h": {"ini": 0.97, "fin": 1.26}}
        ps = sc.umbrales_por_segundo(u, 3, fps_corpus=24.0)
        self.assertAlmostEqual(ps["v_pose_por_s"]["ini"], 0.0734 * 24)
        self.assertAlmostEqual(ps["borde_s"], 0.125)
        self.assertEqual(ps["h"], u["h"])
        a24 = sc.umbrales_a_fps(ps, 24.0)                             # ida y vuelta a la misma tasa
        self.assertEqual(a24["K"], 3)
        self.assertAlmostEqual(a24["v_pose"]["fin"], 0.0096)
        a30 = sc.umbrales_a_fps(ps, 30.4)
        self.assertEqual(a30["K"], 4)                                 # round(0,125 · 30,4) = 4
        self.assertAlmostEqual(a30["v_pose"]["ini"], 0.0734 * 24 / 30.4)
        self.assertEqual(sc.umbrales_a_fps(ps, 10.0)["K"], 3)         # nunca menos de 3
        with self.assertRaises(ValueError):
            sc.umbrales_a_fps(ps, 0)


if __name__ == "__main__":
    unittest.main()


class TestBordeDelUmbral(unittest.TestCase):
    """La confianza exactamente igual al umbral se ACEPTA (>=), como en el prototipo y en D-09."""

    def test_confianza_igual_al_umbral_se_acepta(self):
        n = 50
        fila = np.full(n, 0.8 / (n - 1))
        fila[3] = 0.2
        prob = np.vstack([fila])
        m = sc.metricas_prediccion(prob, np.array([3]), umbral=0.2)
        self.assertEqual(m["rechazo_umbral"], 0.0)
        self.assertEqual(m["aceptadas_ok"], 1)
