"""
Pruebas de src/linea_base_clasica.py (linea base clasica, actividad 4.d, decision D-10).

Lo que se comprueba, en el orden de D-10:
  · los ocho estadisticos contra un calculo a mano (valor por valor, sin reutilizar la funcion);
  · el espejado de la aumentacion no cambia la etiqueta y es una involucion;
  · la seleccion de configuracion usa la VALIDACION y no toca la prueba: la firma de
    elegir_configuracion no admite la prueba, un doble que falla al leerla demuestra que la
    seleccion ya habia terminado, y dos corridas con pruebas distintas eligen lo mismo;
  · McNemar contra una tabla calculada a mano (binomial exacta y chi2 con correccion);
  · una corrida minima de extremo a extremo con datos sinteticos pequenos.

Ejecutar:  python -m unittest tests.test_linea_base
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import linea_base_clasica as lb  # noqa: E402

RAIZ2 = math.sqrt(5.0)
N_MANOS = 126          # representacion «manos»: 2 manos x 21 puntos x 3 coordenadas


def _rejilla_minima():
    """Rejilla reducida (dos configuraciones por familia) para que las pruebas sean rapidas.
    No sustituye a la de D-10: solo comprueba el mecanismo de seleccion."""
    return {
        "svm_rbf": [("C=1,gamma=scale", {"C": 1.0, "gamma": "scale"}),
                    ("C=100,gamma=0.01", {"C": 100.0, "gamma": 0.01})],
        "random_forest": [("min_samples_leaf=1", {"min_samples_leaf": 1}),
                          ("min_samples_leaf=2", {"min_samples_leaf": 2})],
    }


def _datos_sinteticos(n_por_clase=6, n_clases=3, seq=30, semilla=7):
    """Secuencias (N, 30, 126) separables por clase, como las que entran a la linea base."""
    rng = np.random.default_rng(semilla)
    centros = rng.normal(0, 1.0, size=(n_clases, N_MANOS))
    X, y = [], []
    t = np.linspace(0, 1, seq)[:, None]
    for c in range(n_clases):
        for _ in range(n_por_clase):
            base = centros[c] * (0.5 + t)          # trayectoria dependiente de la clase
            X.append(base + rng.normal(0, 0.05, size=(seq, N_MANOS)))
            y.append(c)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


class PruebaTocada(Exception):
    """La prueba se leyo antes de tiempo."""


class DobleDePruebaQueFalla:
    """Doble del acceso al conjunto de prueba: falla si alguien lo llama."""

    def __init__(self):
        self.llamadas = 0

    def __call__(self):
        self.llamadas += 1
        raise PruebaTocada("se intento leer el conjunto de prueba")


# ---------------------------------------------------------------------------
# 1. Los ocho estadisticos contra un calculo a mano
# ---------------------------------------------------------------------------

class TestEstadisticos(unittest.TestCase):

    def test_ocho_estadisticos_contra_calculo_a_mano(self):
        # Muestra 0: columna A = 0, 2, 4, 6 ; columna B = 1, 5, 3, 7. Muestra 1: todo unos.
        X = np.array([[[0.0, 1.0], [2.0, 5.0], [4.0, 3.0], [6.0, 7.0]],
                      [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]])
        F = lb.estadisticos(X)
        self.assertEqual(F.shape, (2, 16))          # 8 estadisticos x 2 columnas

        # A mano, columna A de la muestra 0: media 12/4 = 3; desviaciones -3,-1,1,3 -> suma de
        # cuadrados 20; varianza (ddof=0) 20/4 = 5 -> de = sqrt(5); min 0; max 6; mediana (2+4)/2 = 3;
        # primer paso 0; ultimo paso 6; diferencias 2,2,2 -> media de |dif| = 2.
        # Columna B: media 16/4 = 4; desviaciones -3,1,-1,3 -> 20/4 = 5 -> sqrt(5); min 1; max 7;
        # mediana (3+5)/2 = 4; primero 1; ultimo 7; diferencias 4,-2,4 -> (4+2+4)/3 = 10/3.
        esperada_0 = [3.0, 4.0,            # media
                      RAIZ2, RAIZ2,        # desviacion estandar (ddof = 0)
                      0.0, 1.0,            # minimo
                      6.0, 7.0,            # maximo
                      3.0, 4.0,            # mediana
                      0.0, 1.0,            # primer paso
                      6.0, 7.0,            # ultimo paso
                      2.0, 10.0 / 3.0]     # media de |diferencia entre pasos consecutivos|
        np.testing.assert_allclose(F[0], esperada_0, rtol=0, atol=1e-12)

        esperada_1 = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        np.testing.assert_allclose(F[1], esperada_1, rtol=0, atol=1e-12)

    def test_son_ocho_y_en_el_orden_declarado(self):
        self.assertEqual(len(lb.ESTADISTICOS), 8)
        self.assertEqual(lb.ESTADISTICOS,
                         ("media", "desviacion_estandar", "minimo", "maximo", "mediana",
                          "primer_paso", "ultimo_paso", "media_abs_diferencia"))
        X = np.random.default_rng(0).normal(size=(4, 30, 7))
        F = lb.estadisticos(X)
        self.assertEqual(F.shape, (4, 8 * 7))
        for i, estadistico in enumerate(lb.ESTADISTICOS):
            bloque = F[:, i * 7:(i + 1) * 7]
            referencia = {
                "media": X.mean(axis=1), "desviacion_estandar": X.std(axis=1),
                "minimo": X.min(axis=1), "maximo": X.max(axis=1),
                "mediana": np.median(X, axis=1), "primer_paso": X[:, 0, :],
                "ultimo_paso": X[:, -1, :],
                "media_abs_diferencia": np.abs(np.diff(X, axis=1)).mean(axis=1),
            }[estadistico]
            np.testing.assert_allclose(bloque, referencia, atol=1e-12)

    def test_2016_rasgos_para_el_tensor_de_la_red(self):
        # D-10: 252 columnas x 8 estadisticos = 2.016 rasgos.
        X = np.zeros((2, 30, 252))
        self.assertEqual(lb.estadisticos(X).shape, (2, 2016))
        self.assertEqual(len(lb.nombres_de_rasgos(252)), 2016)
        self.assertEqual(lb.nombres_de_rasgos(252)[0], "media_col000")
        self.assertEqual(lb.nombres_de_rasgos(252)[-1], "media_abs_diferencia_col251")

    def test_exige_tres_dimensiones_y_dos_pasos(self):
        with self.assertRaises(ValueError):
            lb.estadisticos(np.zeros((30, 252)))
        with self.assertRaises(ValueError):
            lb.estadisticos(np.zeros((2, 1, 252)))


# ---------------------------------------------------------------------------
# 2. El espejado no cambia la etiqueta
# ---------------------------------------------------------------------------

class TestEspejo(unittest.TestCase):

    def test_el_espejo_no_cambia_la_etiqueta(self):
        X, y = _datos_sinteticos(n_por_clase=2, n_clases=3)
        Xa, ya = lb.aumentar_espejo(X, y)
        self.assertEqual(len(Xa), 2 * len(X))
        self.assertEqual(len(ya), 2 * len(y))
        # La segunda mitad son las mismas muestras espejadas y con la MISMA etiqueta.
        np.testing.assert_array_equal(ya[:len(y)], y)
        np.testing.assert_array_equal(ya[len(y):], y)
        for i in range(len(y)):
            self.assertEqual(int(ya[i]), int(ya[len(y) + i]))

    def test_el_original_se_conserva_y_el_espejo_es_el_de_la_red(self):
        from src.representacion import espejar
        X, y = _datos_sinteticos(n_por_clase=2, n_clases=2)
        Xa, _ = lb.aumentar_espejo(X, y)
        np.testing.assert_array_equal(Xa[:len(X)], X)          # el original, intacto
        np.testing.assert_allclose(Xa[len(X):], espejar(X, "manos"), atol=0)

    def test_espejar_dos_veces_devuelve_el_original(self):
        from src.representacion import espejar
        X, _ = _datos_sinteticos(n_por_clase=1, n_clases=2)
        np.testing.assert_allclose(espejar(espejar(X, "manos"), "manos"), X, atol=0)


# ---------------------------------------------------------------------------
# 3. La seleccion usa la validacion y nunca la prueba (D-10, punto 4)
# ---------------------------------------------------------------------------

class TestSeleccionNoMiraLaPrueba(unittest.TestCase):

    def setUp(self):
        X, y = _datos_sinteticos(n_por_clase=8, n_clases=3, semilla=11)
        # 6 por clase a entrenamiento, 1 a validacion, 1 a prueba.
        idx_tr, idx_val, idx_te = [], [], []
        for c in range(3):
            i = np.where(y == c)[0]
            idx_tr += list(i[:6]); idx_val.append(i[6]); idx_te.append(i[7])
        self.X, self.y = X, y
        self.tr = (X[idx_tr], y[idx_tr])
        self.val = (X[idx_val], y[idx_val])
        self.te = (X[idx_te], y[idx_te], [f"m{j}" for j in idx_te])

    def test_la_firma_de_elegir_configuracion_no_admite_la_prueba(self):
        import inspect
        parametros = set(inspect.signature(lb.elegir_configuracion).parameters)
        for prohibido in ("X_te", "F_te", "y_te", "prueba", "test", "obtener_prueba"):
            self.assertNotIn(prohibido, parametros)
        self.assertTrue({"F_tr", "y_tr", "F_val", "y_val"} <= parametros)

    def test_un_doble_que_falla_demuestra_que_la_seleccion_ya_habia_terminado(self):
        doble = DobleDePruebaQueFalla()
        traza = []
        with self.assertRaises(PruebaTocada):
            lb.ejecutar_corrida("humo", self.tr[0], self.tr[1], self.val[0], self.val[1], doble,
                                familias=("svm_rbf", "random_forest"), espejo=True,
                                rejilla=_rejilla_minima(), n_clases=3, n_jobs=1, traza=traza)
        # El doble se llamo una sola vez y ya habia una eleccion registrada para cada familia.
        self.assertEqual(doble.llamadas, 1)
        elegidas = [t for t in traza if t["paso"] == "seleccion"]
        self.assertEqual({t["familia"] for t in elegidas}, {"svm_rbf", "random_forest"})
        for t in elegidas:
            self.assertIn(t["elegida"], [n for n, _ in _rejilla_minima()[t["familia"]]])
        # Y la seleccion ocurrio ANTES del unico acceso a la prueba.
        pasos = [t["paso"] for t in traza]
        self.assertNotIn("prueba", pasos)

    def test_dos_pruebas_distintas_dan_la_misma_eleccion(self):
        rng = np.random.default_rng(3)
        prueba_real = lambda: self.te                                      # noqa: E731
        prueba_basura = lambda: (rng.normal(size=self.te[0].shape).astype(np.float32),  # noqa: E731
                                 np.array([2, 0, 1]), ["a", "b", "c"])
        elecciones = []
        for obtener in (prueba_real, prueba_basura):
            r = lb.ejecutar_corrida("humo", self.tr[0], self.tr[1], self.val[0], self.val[1],
                                    obtener, familias=("svm_rbf", "random_forest"), espejo=True,
                                    rejilla=_rejilla_minima(), n_clases=3, n_jobs=1)
            elecciones.append({f: r["familias"][f]["configuracion_elegida"]
                               for f in ("svm_rbf", "random_forest")})
        self.assertEqual(elecciones[0], elecciones[1])

    def test_gana_la_de_mayor_top1_de_validacion_este_donde_este(self):
        # Anadida por la verificacion independiente de 4.d (16-sep-2026): la prueba anterior usaba
        # una rejilla en la que las dos configuraciones empataban a 1,0 de validacion, asi que no
        # distinguia «elegir la mejor» de «elegir la peor» (un mutante que cambia > por < sobrevivia).
        # Aqui la rejilla separa de verdad (gamma = 1e6 hunde el SVM) y se prueba en los dos ordenes,
        # para que ni el criterio del maximo ni la regla de empate «gana la primera» pasen por azar.
        buena = ("C=10,gamma=scale", {"C": 10.0, "gamma": "scale"})
        mala = ("C=1,gamma=1e6", {"C": 1.0, "gamma": 1e6})
        for rejilla in ({"svm_rbf": [buena, mala]}, {"svm_rbf": [mala, buena]}):
            r = lb.ejecutar_corrida("humo", self.tr[0], self.tr[1], self.val[0], self.val[1],
                                    lambda: self.te, familias=("svm_rbf",), espejo=True,
                                    rejilla=rejilla, n_clases=3, n_jobs=1)
            detalle = r["familias"]["svm_rbf"]["seleccion"]
            top1 = {c["configuracion"]: c["validacion"]["top1"] for c in detalle["rejilla"]}
            self.assertGreater(top1["C=10,gamma=scale"], top1["C=1,gamma=1e6"])   # separa de verdad
            self.assertEqual(detalle["elegida"], "C=10,gamma=scale")
            self.assertEqual(detalle["validacion_elegida"]["top1"], max(top1.values()))

    def test_el_escalador_se_ajusta_solo_con_el_entrenamiento(self):
        # Anadida por la verificacion independiente de 4.d (16-sep-2026): D-10 exige «escalado
        # estandar ajustado solo con el entrenamiento de cada pliegue» y ninguna prueba lo sujetaba
        # (un mutante que ajustaba el escalador con entrenamiento + validacion sobrevivia).
        import sklearn.preprocessing as preprocesamiento
        real = preprocesamiento.StandardScaler
        formas = []

        class EscaladorEspia(real):
            def fit(self, X, y=None):
                formas.append(X.shape)
                return super().fit(X, y)

        preprocesamiento.StandardScaler = EscaladorEspia
        try:
            r = lb.ejecutar_corrida(
                "humo", self.tr[0], self.tr[1], self.val[0], self.val[1], lambda: self.te,
                familias=("random_forest",), espejo=True,
                rejilla={"random_forest": [("min_samples_leaf=1", {"min_samples_leaf": 1})]},
                n_clases=3, n_jobs=1)
        finally:
            preprocesamiento.StandardScaler = real
        self.assertEqual(len(formas), 1)                      # un unico ajuste por corrida
        self.assertEqual(formas[0][0], r["n_train_usado"])    # ni validacion ni prueba entran
        self.assertNotEqual(formas[0][0], r["n_train_usado"] + r["n_val"])
        self.assertNotEqual(formas[0][0], r["n_train_usado"] + r["n_test"])
        self.assertEqual(formas[0][1], r["n_rasgos"])

    def test_la_elegida_es_la_de_mayor_top1_de_validacion(self):
        r = lb.ejecutar_corrida("humo", self.tr[0], self.tr[1], self.val[0], self.val[1],
                                lambda: self.te, familias=("random_forest",), espejo=False,
                                rejilla=_rejilla_minima(), n_clases=3, n_jobs=1)
        detalle = r["familias"]["random_forest"]["seleccion"]
        top1 = [c["validacion"]["top1"] for c in detalle["rejilla"]]
        nombres = [c["configuracion"] for c in detalle["rejilla"]]
        self.assertEqual(detalle["elegida"], nombres[int(np.argmax(top1))])   # empate: la primera


# ---------------------------------------------------------------------------
# 4. McNemar contra un caso calculado a mano
# ---------------------------------------------------------------------------

class TestMcNemar(unittest.TestCase):

    def test_caso_calculado_a_mano(self):
        # 20 muestras: 5 las aciertan los dos, 10 solo A, 2 solo B, 3 no las acierta ninguno.
        a = np.array([True] * 5 + [True] * 10 + [False] * 2 + [False] * 3)
        b = np.array([True] * 5 + [False] * 10 + [True] * 2 + [False] * 3)
        r = lb.mcnemar(a, b)
        self.assertEqual(r["tabla"], {"aciertan_ambos": 5, "solo_a": 10, "solo_b": 2,
                                      "fallan_ambos": 3})
        self.assertEqual(r["n_discordantes"], 12)
        self.assertEqual(r["n_muestras"], 20)
        self.assertEqual(r["aciertos_a"], 15)
        self.assertEqual(r["aciertos_b"], 7)

        # chi2 con correccion de continuidad = (|10 - 2| - 1)^2 / 12 = 49/12 = 4,0833...
        self.assertAlmostEqual(r["chi2_con_correccion_de_continuidad"], 49 / 12, places=12)
        # Con 1 grado de libertad, P(X > x) = erfc(sqrt(x/2)).
        p_chi2_a_mano = math.erfc(math.sqrt((49 / 12) / 2))
        self.assertAlmostEqual(r["p_chi2"], p_chi2_a_mano, places=12)
        # Exacta: binomial de dos colas con n = 12, k = 10, p = 0,5
        # 2 * [C(12,10) + C(12,11) + C(12,12)] / 2^12 = 2 * (66 + 12 + 1) / 4096 = 158/4096.
        self.assertAlmostEqual(r["p_exacto_binomial"], 158 / 4096, places=12)
        # 12 discordantes (< 25): manda la exacta, y a alfa = 0,05 es significativa.
        self.assertEqual(r["prueba_usada"], "exacta binomial")
        self.assertAlmostEqual(r["p"], 158 / 4096, places=12)
        self.assertTrue(r["significativo"])
        self.assertEqual(r["alfa"], 0.05)

    def test_muchos_discordantes_usa_chi2(self):
        a = np.array([True] * 40 + [False] * 20)
        b = np.array([False] * 40 + [True] * 20)
        r = lb.mcnemar(a, b)
        self.assertEqual(r["n_discordantes"], 60)
        self.assertEqual(r["prueba_usada"], "chi2 con corrección de continuidad")
        self.assertAlmostEqual(r["chi2_con_correccion_de_continuidad"], (20 - 1) ** 2 / 60, places=12)

    def test_sin_discordancias_no_hay_diferencia(self):
        a = np.array([True, False, True, False])
        r = lb.mcnemar(a, a.copy())
        self.assertEqual(r["n_discordantes"], 0)
        self.assertEqual(r["p"], 1.0)
        self.assertFalse(r["significativo"])

    def test_se_puede_forzar_la_prueba_y_exige_el_mismo_numero_de_muestras(self):
        a = np.array([True] * 5 + [True] * 10 + [False] * 2 + [False] * 3)
        b = np.array([True] * 5 + [False] * 10 + [True] * 2 + [False] * 3)
        forzada = lb.mcnemar(a, b, exacto=False)
        self.assertEqual(forzada["prueba_usada"], "chi2 con corrección de continuidad")
        self.assertAlmostEqual(forzada["p"], math.erfc(math.sqrt((49 / 12) / 2)), places=12)
        with self.assertRaises(ValueError):
            lb.mcnemar(a, b[:-1])


# ---------------------------------------------------------------------------
# 5. Metricas, formato de predicciones y corrida minima de extremo a extremo
# ---------------------------------------------------------------------------

class TestMetricasYFormato(unittest.TestCase):

    def test_top1_y_top3_a_mano(self):
        probs = np.array([[0.5, 0.3, 0.1, 0.1],    # predicha 0, real 0 -> acierta top-1
                          [0.1, 0.2, 0.4, 0.3],    # predicha 2, real 3 -> top-3 (2, 3, 1)
                          [0.7, 0.1, 0.1, 0.1]])   # predicha 0, real 1 -> top-3 (0, ...)
        y = np.array([0, 3, 1])
        m = lb.metricas(y, probs, n_clases=4)
        self.assertAlmostEqual(m["top1"], 1 / 3)
        self.assertAlmostEqual(m["top3"], 1.0)
        self.assertEqual(m["n_errores"], 2)
        self.assertEqual(m["n"], 3)

    def test_formato_identico_al_de_la_red(self):
        probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
        filas = lb.predicciones_json(["a", "b"], np.array([0, 2]), probs)
        self.assertEqual(set(filas[0]), {"muestra", "real", "predicha", "top3", "prob_predicha"})
        self.assertEqual(filas[0], {"muestra": "a", "real": 0, "predicha": 0, "top3": [0, 1, 2],
                                    "prob_predicha": 0.5})
        self.assertEqual(filas[1]["predicha"], 1)
        self.assertEqual(filas[1]["top3"][0], filas[1]["predicha"])   # como en la red

    def test_con_empates_top3_empieza_por_la_clase_predicha(self):
        # Random Forest empata a menudo: la primera del top-3 debe seguir siendo la predicha.
        probs = np.array([[0.4, 0.4, 0.2], [0.0, 0.0, 1.0], [0.25, 0.25, 0.5]])
        filas = lb.predicciones_json(["a", "b", "c"], np.array([0, 2, 1]), probs)
        for f in filas:
            self.assertEqual(f["top3"][0], f["predicha"])
        self.assertEqual([f["predicha"] for f in filas], [0, 2, 2])
        m = lb.metricas(np.array([0, 2, 1]), probs, n_clases=3)
        self.assertAlmostEqual(m["top1"], 2 / 3)


class TestCorridaExtremoAExtremo(unittest.TestCase):

    def test_corrida_minima_con_datos_sinteticos(self):
        X, y = _datos_sinteticos(n_por_clase=8, n_clases=3, semilla=23)
        idx_tr, idx_val, idx_te = [], [], []
        for c in range(3):
            i = np.where(y == c)[0]
            idx_tr += list(i[:6]); idx_val.append(i[6]); idx_te.append(i[7])
        nombres_te = [f"muestra_{j:03d}" for j in idx_te]
        traza = []
        r = lb.ejecutar_corrida(
            "sintetica", X[idx_tr], y[idx_tr], X[idx_val], y[idx_val],
            lambda: (X[idx_te], y[idx_te], nombres_te),
            familias=("svm_rbf", "random_forest"), espejo=True, rejilla=_rejilla_minima(),
            n_clases=3, n_jobs=1, traza=traza)

        self.assertEqual(r["corrida"], "sintetica")
        self.assertTrue(r["espejo"])
        self.assertEqual(r["n_train_original"], 18)
        self.assertEqual(r["n_train_usado"], 36)        # espejo: factor 2
        self.assertEqual(r["n_val"], 3)
        self.assertEqual(r["n_test"], 3)
        self.assertEqual(r["n_rasgos"], 8 * 2 * N_MANOS)   # 8 estadisticos x (126 + 126 velocidades)
        for familia in ("svm_rbf", "random_forest"):
            f = r["familias"][familia]
            self.assertIn(f["configuracion_elegida"], [n for n, _ in _rejilla_minima()[familia]])
            self.assertEqual(len(f["seleccion"]["rejilla"]), 2)
            for clave in ("top1", "top3", "precision_macro", "recall_macro", "f1_macro"):
                self.assertGreaterEqual(f["prueba"][clave], 0.0)
                self.assertLessEqual(f["prueba"][clave], 1.0)
            self.assertEqual([p["muestra"] for p in f["predicciones"]], nombres_te)
            self.assertEqual(set(f["predicciones"][0]),
                             {"muestra", "real", "predicha", "top3", "prob_predicha"})
        # El orden de la traza: primero los rasgos, luego las selecciones, y la prueba al final.
        pasos = [t["paso"] for t in traza]
        self.assertEqual(pasos, ["rasgos", "seleccion", "seleccion", "prueba"])
        self.assertLess(max(i for i, p in enumerate(pasos) if p == "seleccion"),
                        pasos.index("prueba"))

    def test_el_tensor_por_muestra_es_30x252(self):
        X, _ = _datos_sinteticos(n_por_clase=1, n_clases=2)
        T = lb.tensor_de_red(X)
        self.assertEqual(T.shape[1:], (30, 252))
        np.testing.assert_allclose(T[:, :, :N_MANOS], X, atol=0)      # coordenadas intactas
        np.testing.assert_allclose(T[:, 0, N_MANOS:], 0.0, atol=0)    # primera velocidad, cero
        np.testing.assert_allclose(T[:, 1:, N_MANOS:], X[:, 1:, :] - X[:, :-1, :], atol=1e-6)
        with self.assertRaises(AssertionError):
            lb.tensor_de_red(X[:, :10, :])                            # no son 30 pasos

    def test_rasgos_agregados_devuelve_2016_columnas(self):
        X, _ = _datos_sinteticos(n_por_clase=2, n_clases=2)
        F = lb.rasgos_agregados(X)
        self.assertEqual(F.shape, (4, 2016))


# ---------------------------------------------------------------------------
# 6. Que la rejilla en disco sea la de D-10
# ---------------------------------------------------------------------------

class TestRejillaDeD10(unittest.TestCase):

    def test_la_rejilla_es_la_declarada_en_d10(self):
        self.assertEqual(set(lb.REJILLA), {"svm_rbf", "random_forest", "mlp"})
        svm = dict(lb.REJILLA["svm_rbf"])
        self.assertEqual(len(svm), 6)
        self.assertEqual({p["C"] for p in svm.values()}, {1.0, 10.0, 100.0})
        self.assertEqual({str(p["gamma"]) for p in svm.values()}, {"scale", "0.01"})
        rf = dict(lb.REJILLA["random_forest"])
        self.assertEqual({p["min_samples_leaf"] for p in rf.values()}, {1, 2})
        mlp = dict(lb.REJILLA["mlp"])
        self.assertEqual({p["alpha"] for p in mlp.values()}, {1e-4, 1e-2})

    def test_los_estimadores_llevan_los_hiperparametros_de_d10(self):
        svm = lb.crear_modelo("svm_rbf", {"C": 10.0, "gamma": "scale"})
        self.assertEqual((svm.kernel, svm.C, svm.gamma, svm.probability), ("rbf", 10.0, "scale", True))
        rf = lb.crear_modelo("random_forest", {"min_samples_leaf": 2}, n_jobs=1)
        self.assertEqual((rf.n_estimators, rf.max_features, rf.min_samples_leaf), (400, "sqrt", 2))
        mlp = lb.crear_modelo("mlp", {"alpha": 1e-2})
        self.assertEqual((mlp.hidden_layer_sizes, mlp.alpha, mlp.early_stopping),
                         ((256,), 1e-2, True))
        for modelo in (svm, rf, mlp):
            self.assertEqual(modelo.random_state, lb.SEMILLA)
        with self.assertRaises(ValueError):
            lb.crear_modelo("familia_inventada", {})


class TestAumentacionCompleta(unittest.TestCase):
    """Control de equidad: la línea base puede entrenarse con la misma aumentación de la red
    (factor 10). El modo por defecto sigue siendo «espejo», que es lo que fija D-10."""

    def _secuencias(self, n=6):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(n, lb.SEQ_LENGTH, 126)).astype(np.float32)
        y = np.arange(n) % 3
        return X, y

    def test_factor_por_modo(self):
        self.assertEqual(lb.FACTOR_AUMENTACION, {"ninguna": 1, "espejo": 2, "completa": 10})
        X, y = self._secuencias()
        for modo, factor in lb.FACTOR_AUMENTACION.items():
            Xa, ya = lb.aumentar_muestras(X, y, modo)
            self.assertEqual(len(Xa), factor * len(X), modo)
            self.assertEqual(len(ya), factor * len(y), modo)
            # el original va primero y sin tocar, y las etiquetas se repiten en el mismo orden
            np.testing.assert_array_equal(Xa[:len(X)], X)
            np.testing.assert_array_equal(ya, np.tile(y, factor))

    def test_el_modo_espejo_es_el_de_d10_y_sigue_siendo_el_de_por_defecto(self):
        X, y = self._secuencias()
        np.testing.assert_array_equal(lb.aumentar_muestras(X, y, "espejo")[0],
                                      lb.aumentar_espejo(X, y)[0])
        np.testing.assert_array_equal(lb.aumentar_muestras(X, y)[0], lb.aumentar_espejo(X, y)[0])

    def test_la_completa_usa_las_nueve_transformaciones_de_la_red(self):
        nombres = [n for n, _ in lb.transformaciones_de_la_red()]
        self.assertEqual(len(nombres), 9)
        self.assertEqual(nombres[0], "espejo")
        for esperado in ("ruido_0.005", "temporal_0.85-0.95", "escala_0.85-1.15"):
            self.assertIn(esperado, nombres)
        X, y = self._secuencias()
        Xa, _ = lb.aumentar_muestras(X, y, "completa")
        # la primera transformación es el espejo de la representación, igual que en la red
        np.testing.assert_allclose(Xa[len(X):2 * len(X)], lb.aumentar_espejo(X, y)[0][len(X):],
                                   rtol=0, atol=0)

    def test_modo_desconocido_falla(self):
        X, y = self._secuencias()
        with self.assertRaises(ValueError):
            lb.aumentar_muestras(X, y, "inventada")

    def test_la_linea_de_comandos_ofrece_los_tres_modos(self):
        import subprocess
        ayuda = subprocess.run([sys.executable, str(BASE_DIR / "src" / "linea_base_clasica.py"),
                                "--help"], capture_output=True, text=True, timeout=120)
        self.assertEqual(ayuda.returncode, 0)
        self.assertIn("--aumentacion", ayuda.stdout)
        for modo in lb.AUMENTACIONES:
            self.assertIn(modo, ayuda.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
