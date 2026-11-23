"""
Pruebas de la representación de entrada (D-08): subconjunto facial, columnas por candidato, espejado
generalizado de la aumentación y preprocesamiento con --rostro cejas_labios.

Sin mediapipe ni corpus: el preprocesamiento se prueba sobre un árbol plano sintético en un tmpdir.
Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_representacion -v
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import representacion as rep  # noqa: E402


class TestSubconjuntoFacial(unittest.TestCase):

    def test_tamanos_y_unicidad(self):
        sub = rep.ROSTRO_SUBCONJUNTOS["cejas_labios"]
        self.assertEqual(len(sub), 60)
        self.assertEqual(len(set(sub)), 60)
        self.assertTrue(all(0 <= i < 468 for i in sub))
        self.assertEqual(len(rep.CEJA_A) + len(rep.CEJA_B), 20)
        self.assertEqual(len(rep.LABIO_A) + len(rep.LABIO_B) + len(rep.LABIO_CENTRO), 40)

    def test_labios_y_cejas_son_las_regiones_oficiales_de_mediapipe(self):
        # FACE_LANDMARKS_LIPS / LEFT_EYEBROW / RIGHT_EYEBROW de mediapipe 0.10.35 (copiadas el 15-sep-2026)
        lips = {0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191,
                267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415}
        self.assertEqual(set(rep.LABIO_A + rep.LABIO_B + rep.LABIO_CENTRO), lips)
        self.assertEqual(set(rep.CEJA_A), {46, 52, 53, 55, 63, 65, 66, 70, 105, 107})
        self.assertEqual(set(rep.CEJA_B), {276, 282, 283, 285, 293, 295, 296, 300, 334, 336})

    def test_pares_son_reflejos_exactos_en_el_modelo_canonico(self):
        datos = json.loads((ROOT_DIR / "tests" / "datos" / "canonico_cejas_labios.json").read_text(encoding="utf-8"))
        V = {int(k): np.array(v) for k, v in datos["vertices"].items()}
        for a, b in rep.PARES_ROSTRO:
            np.testing.assert_allclose(V[a] * [-1, 1, 1], V[b], atol=1e-9, err_msg=f"par {a}-{b}")
        for c in rep.LABIO_CENTRO:
            self.assertAlmostEqual(V[c][0], 0.0, places=9)

    def test_pares_de_postura_iguales_a_train_lstm(self):
        # Leído del código fuente para no importar TensorFlow en esta prueba.
        import ast
        arbol = ast.parse((ROOT_DIR / "src" / "train_lstm.py").read_text(encoding="utf-8"))
        pares = next(ast.literal_eval(n.value) for n in arbol.body
                     if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "BODY_SYMMETRIC_PAIRS")
        self.assertEqual([tuple(p) for p in pares], rep.PARES_POSTURA)


class TestColumnas(unittest.TestCase):

    def test_columnas_por_representacion(self):
        np.testing.assert_array_equal(rep.columnas("manos"), np.arange(99, 225))
        np.testing.assert_array_equal(rep.columnas("manos_postura"), np.arange(0, 225))
        np.testing.assert_array_equal(rep.columnas("manos_postura_rostro", "cejas_labios"), np.arange(0, 405))
        self.assertEqual([rep.n_rasgos(r) for r in ("manos", "manos_postura", "manos_postura_rostro")], [126, 225, 405])

    def test_rostro_sin_processed_con_rostro_falla(self):
        with self.assertRaises(ValueError):
            rep.columnas("manos_postura_rostro", "ninguno")
        with self.assertRaises(ValueError):
            rep.columnas("otra")


class TestEspejado(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_manos_postura_igual_a_augment_mirror(self):
        from src.train_lstm import augment_mirror
        X = self.rng.normal(size=(4, 6, 225)).astype(np.float32)
        np.testing.assert_array_equal(rep.espejar(X, "manos_postura"), augment_mirror(X))

    def test_involucion_en_todas_las_representaciones(self):
        for r in rep.REPRESENTACIONES:
            X = self.rng.normal(size=(3, 5, rep.n_rasgos(r))).astype(np.float32)
            np.testing.assert_array_equal(rep.espejar(rep.espejar(X, r), r), X, err_msg=r)

    def test_manos_intercambia_y_niega_x(self):
        X = self.rng.normal(size=(2, 4, 126)).astype(np.float32)
        M = rep.espejar(X, "manos")
        izq, der = X[..., :63], X[..., 63:]
        np.testing.assert_array_equal(M[..., :63][..., 1::3], der[..., 1::3])       # y sin cambio
        np.testing.assert_array_equal(M[..., :63][..., 0::3], -der[..., 0::3])      # x negada
        np.testing.assert_array_equal(M[..., 63:][..., 2::3], izq[..., 2::3])       # z sin cambio

    def test_rostro_intercambia_pares_y_centro_queda(self):
        r = "manos_postura_rostro"
        X = self.rng.normal(size=(1, 2, 405)).astype(np.float32)
        M = rep.espejar(X, r)
        sub = rep.ROSTRO_SUBCONJUNTOS["cejas_labios"]
        col = lambda idx: 225 + 3 * sub.index(idx)  # noqa: E731
        for a, b in rep.PARES_ROSTRO[:3] + rep.PARES_ROSTRO[-2:]:
            np.testing.assert_array_equal(M[..., col(a)], -X[..., col(b)])
            np.testing.assert_array_equal(M[..., col(a) + 1:col(a) + 3], X[..., col(b) + 1:col(b) + 3])
        for c in rep.LABIO_CENTRO:
            np.testing.assert_array_equal(M[..., col(c)], -X[..., col(c)])
            np.testing.assert_array_equal(M[..., col(c) + 1], X[..., col(c) + 1])
        # Los bloques de postura y manos se espejan igual que sin rostro.
        np.testing.assert_array_equal(M[..., :225], rep.espejar(X[..., :225], "manos_postura"))

    def test_dimension_equivocada_falla(self):
        with self.assertRaises(ValueError):
            rep.espejar(np.zeros((1, 2, 225), dtype=np.float32), "manos")


class TestAumentacionGeneralizada(unittest.TestCase):
    """La ruta genérica de aumentar (otras representaciones) aplica las mismas transformaciones, en
    el mismo orden y con el mismo consumo de np.random que refined_augmentation."""

    def test_ruta_generica_equivale_a_refined_augmentation(self):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import src.train_final_100 as tf100
        X = np.random.default_rng(3).normal(size=(6, 30, 225)).astype(np.float32)
        y = np.arange(6)
        np.random.seed(123)
        Xa, ya = tf100.refined_augmentation(X, y)
        # Forzar la ruta genérica con el espejado de 225 rasgos (idéntico a augment_mirror).
        with mock.patch.object(tf100, "espejar", lambda X_, r: rep.espejar(X_, "manos_postura")):
            np.random.seed(123)
            Xb, yb = tf100.aumentar(X, y, 10, "manos")
        np.testing.assert_array_equal(Xa, Xb)
        np.testing.assert_array_equal(ya, yb)

    def test_manos_postura_usa_refined_augmentation(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import src.train_final_100 as tf100
        self.assertIs(tf100.transformaciones_para("manos_postura"), tf100.TRANSFORMACIONES_AUMENTO)

    def test_train_loso_aumenta_igual_que_train_final_100(self):
        # Revisión D-08 (15-sep-2026): las corridas LOSO usan importar_pipeline_canonico/aumentar de
        # train_loso, no la ruta de train_final_100; ambas deben dar lo mismo con cada representación.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import src.train_final_100 as tf100
            import src.train_loso as tl
        y = np.arange(4)
        for r in rep.REPRESENTACIONES:
            X = np.random.default_rng(5).normal(size=(4, 12, rep.n_rasgos(r))).astype(np.float32)
            transf = tl.importar_pipeline_canonico(r)[4]
            for k in (2, 10):
                np.random.seed(99)
                Xa, ya = tf100.aumentar(X, y, k, r)
                np.random.seed(99)
                Xb, yb = tl.aumentar(X, y, k, transf)
                np.testing.assert_array_equal(Xa, Xb, err_msg=f"{r} k={k}")
                np.testing.assert_array_equal(ya, yb, err_msg=f"{r} k={k}")
            np.testing.assert_array_equal(transf[0][1](X), rep.espejar(X, r), err_msg=r)


def _escribir_csv(ruta: Path, filas: np.ndarray):
    n = filas.shape[1] // 3
    with open(ruta, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow([""] + [f"landmark_{i}_{c}" for i in range(n) for c in "xyz"])
        for t, fila in enumerate(filas):
            w.writerow([t] + [f"{v:.6f}" for v in fila])


class TestPreprocesamientoConRostro(unittest.TestCase):

    def _arbol(self, raiz: Path, T=5, sin_rostro=(), con_face=True):
        rng = np.random.default_rng(11)
        carpetas = ["BODY_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"] + (["FACE_LANDMARKS"] if con_face else [])
        for c in carpetas:
            (raiz / c).mkdir(parents=True)
        for m in ("0000_0000_0000", "0001_0000_0000", "0002_0001_0000"):
            body = rng.uniform(0.3, 0.7, size=(T, 99))
            body[:, 33:36] = [0.6, 0.4, 0.0]; body[:, 36:39] = [0.4, 0.4, 0.0]   # hombros
            _escribir_csv(raiz / "BODY_LANDMARKS" / f"{m}.csv", body)
            _escribir_csv(raiz / "LEFT_HAND_LANDMARKS" / f"{m}.csv", rng.uniform(0.3, 0.7, size=(T, 63)))
            _escribir_csv(raiz / "RIGHT_HAND_LANDMARKS" / f"{m}.csv", rng.uniform(0.3, 0.7, size=(T, 63)))
            if con_face:
                face = rng.uniform(0.45, 0.55, size=(T, 1404))
                if m == "0001_0000_0000":
                    face[list(sin_rostro)] = 0.0
                _escribir_csv(raiz / "FACE_LANDMARKS" / f"{m}.csv", face)

    def _correr(self, argv):
        import matplotlib
        matplotlib.use("Agg")
        from src import preprocess_lsc50 as pp
        with contextlib.redirect_stdout(io.StringIO()):
            pp.main(argv)

    def test_bloque_facial_normalizado_y_enmascarado(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._arbol(tmp / "plano", sin_rostro=(2,))
            self._correr(["--landmarks", str(tmp / "plano"), "--salida", str(tmp / "sin"), "--mano-ausente", "mascara"])
            self._correr(["--landmarks", str(tmp / "plano"), "--salida", str(tmp / "con"), "--mano-ausente", "mascara",
                          "--rostro", "cejas_labios"])
            A = np.load(tmp / "sin" / "X_padded.npy"); B = np.load(tmp / "con" / "X_padded.npy")
            meta = json.loads((tmp / "con" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(A.shape[2], 225); self.assertEqual(B.shape[2], 405)
            np.testing.assert_array_equal(A, B[:, :, :225])
            self.assertEqual(meta["n_features"], 405)
            self.assertEqual(meta["rostro"]["columnas"], [225, 405])
            self.assertEqual(meta["mano_ausente"]["fotogramas_sin_rostro"], 1)
            # punto facial normalizado por hombros: (p - punto medio) / distancia entre hombros (0,2)
            filas = list(csv.DictReader(io.StringIO((tmp / "plano" / "FACE_LANDMARKS" / "0000_0000_0000.csv").read_text(), newline="")))
            k = rep.ROSTRO_SUBCONJUNTOS["cejas_labios"][7]
            p = np.array([float(filas[1][f"landmark_{k}_{c}"]) for c in "xyz"])
            i = meta["sample_order"].index("0000_0000_0000")
            np.testing.assert_allclose(B[i, 1, 225 + 21:225 + 24], (p - [0.5, 0.4, 0.0]) / 0.2, atol=1e-4)
            j = meta["sample_order"].index("0001_0000_0000")
            self.assertTrue(np.all(B[j, 2, 225:] == 0.0))
            self.assertTrue(np.any(B[j, 1, 225:] != 0.0))
            self.assertTrue(np.any(B[j, 2, :225] != 0.0))

    def test_descartar_omite_fotograma_sin_rostro(self):
        # Revisión D-08 (15-sep-2026): en modo descartar el fotograma sin rostro se omite y lengths lo refleja.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._arbol(tmp / "plano", sin_rostro=(2,))
            self._correr(["--landmarks", str(tmp / "plano"), "--salida", str(tmp / "d"), "--mano-ausente", "descartar",
                          "--rostro", "cejas_labios"])
            meta = json.loads((tmp / "d" / "metadata.json").read_text(encoding="utf-8"))
            lengths = np.load(tmp / "d" / "lengths.npy")
            self.assertEqual(dict(zip(meta["sample_order"], lengths.tolist())),
                             {"0000_0000_0000": 5, "0001_0000_0000": 4, "0002_0001_0000": 5})
            self.assertEqual(meta["mano_ausente"]["fotogramas_descartados"], 1)

    def test_rostro_requiere_arbol_plano_con_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._arbol(tmp / "plano", con_face=False)
            with self.assertRaises(SystemExit):
                self._correr(["--landmarks", str(tmp / "plano"), "--salida", str(tmp / "o"), "--rostro", "cejas_labios"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
