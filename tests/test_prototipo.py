"""
Pruebas sin cámara del prototipo (src/translator.py, src/latencia_resumen.py).

Se inyectan un extractor falso, un modelo falso (y, si existen, los modelos reales de
225 y 450 rasgos), un TTS falso y un reloj falso; los fotogramas son sintéticos.
El extractor real (LandmarkExtractor, D-05) se prueba con un HolisticLandmarker falso
inyectado por constructor y, si el modelo .task y un video del corpus están en disco,
con el HolisticLandmarker real sobre 10 fotogramas (solo lectura; nada se escribe).

Ejecutar:
    python -m unittest tests/test_prototipo.py
"""

import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.translator import (  # noqa: E402
    ESTADOS, ETAPAS, HOLISTIC_TASK, IDX_MANO_DER, IDX_MANO_IZQ, IDX_POSE, MODEL_SHA256,
    N_FACE, N_FACE_FEATURES, N_FEATURES, N_HAND, N_POSE, OPCIONES_HOLISTIC, Extraccion,
    LandmarkExtractor, LSCTranslator, normalize_sequence,
)
from src.grammar_mapper import GrammarMapper  # noqa: E402
from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402
from src.conversor_glosas import ConversorGlosas  # noqa: E402
from src.translator import CONVERSORES, construir_conversor  # noqa: E402
from src import latencia_resumen  # noqa: E402

EXT_PROHIBIDAS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".avi", ".mp4", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Dobles de prueba
# ---------------------------------------------------------------------------

class RelojFalso:
    """Reloj monótono determinista: avanza `paso` ns en cada lectura."""

    def __init__(self, paso_ns: int = 1_000_000):
        self.t = 0
        self.paso = paso_ns

    def __call__(self) -> int:
        self.t += self.paso
        return self.t

    def avanzar(self, ns: int) -> None:
        self.t += ns


class ExtractorFalso:
    """Devuelve puntos sintéticos con la misma interfaz que LandmarkExtractor (Extraccion
    con keypoints de 225, banderas, rostro y holistic_result); la detección se controla con
    `patron`, una función (índice de fotograma) -> (pose_detectada, n_manos). El rostro
    acompaña a la pose (1404 valores) y es None sin pose."""

    descripcion = "extractor_falso"

    def __init__(self, patron=None):
        self.patron = patron or (lambda i: (True, 2))
        self.llamadas = 0
        self.cerrado = False

    def extract(self, frame_rgb):
        i = self.llamadas
        self.llamadas += 1
        pose, manos = self.patron(i)
        kp = np.zeros(N_FEATURES, dtype=np.float32)
        rostro = None
        if pose:
            # 33 puntos de pose con hombros separados (normalización bien definida)
            pose_xyz = np.random.default_rng(i).uniform(0.2, 0.8, size=(33, 3)).astype(np.float32)
            pose_xyz[11] = [0.6, 0.4, 0.0]   # hombro izquierdo
            pose_xyz[12] = [0.4, 0.4, 0.0]   # hombro derecho
            kp[IDX_POSE] = pose_xyz.reshape(-1)
            rostro = np.random.default_rng(3000 + i).uniform(0.4, 0.6, size=N_FACE_FEATURES).astype(np.float32)
        if manos >= 1:
            kp[IDX_MANO_IZQ] = np.random.default_rng(1000 + i).uniform(0.3, 0.7, size=63).astype(np.float32)
        if manos == 2:
            kp[IDX_MANO_DER] = np.random.default_rng(2000 + i).uniform(0.3, 0.7, size=63).astype(np.float32)
        return Extraccion(keypoints=kp, pose_detectada=pose, n_manos=manos, rostro=rostro,
                          holistic_result=None)

    def close(self):
        self.cerrado = True


# --- Dobles del HolisticLandmarker de MediaPipe (para probar LandmarkExtractor sin modelo) ---

class PuntoFalso:
    """Imita NormalizedLandmark (solo x, y, z)."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


def puntos_falsos(n: int, semilla: int) -> list:
    xyz = np.random.default_rng(semilla).uniform(0.0, 1.0, size=(n, 3)).astype(np.float32)
    return [PuntoFalso(*fila) for fila in xyz]


def aplanado(puntos) -> np.ndarray:
    return np.asarray([[p.x, p.y, p.z] for p in puntos], dtype=np.float32).reshape(-1)


class ResultadoHolisticoFalso:
    """Imita HolisticLandmarkerResult: listas planas, vacías cuando no hay detección."""

    def __init__(self, pose=None, izq=None, der=None, rostro=None):
        self.pose_landmarks = list(pose or [])
        self.left_hand_landmarks = list(izq or [])
        self.right_hand_landmarks = list(der or [])
        self.face_landmarks = list(rostro or [])


class HolisticFalso:
    """Imita HolisticLandmarker: detect_for_video devuelve los resultados programados en
    orden y guarda cada marca de tiempo recibida."""

    def __init__(self, resultados):
        self.resultados = list(resultados)
        self.timestamps: list[int] = []
        self.cerrado = False

    def detect_for_video(self, image, ts_ms):
        self.timestamps.append(int(ts_ms))
        return self.resultados.pop(0) if self.resultados else ResultadoHolisticoFalso()

    def close(self):
        self.cerrado = True


def resultado_completo(semilla: int = 0) -> ResultadoHolisticoFalso:
    """Pose + dos manos + rostro de 478 puntos (los 10 últimos son iris)."""
    return ResultadoHolisticoFalso(pose=puntos_falsos(N_POSE, semilla),
                                   izq=puntos_falsos(N_HAND, 100 + semilla),
                                   der=puntos_falsos(N_HAND, 200 + semilla),
                                   rostro=puntos_falsos(N_FACE + 10, 300 + semilla))


def extractor_con_falso(resultados, reloj=None) -> tuple:
    """LandmarkExtractor real con un HolisticLandmarker falso inyectado por constructor."""
    falso = HolisticFalso(resultados)
    ext = LandmarkExtractor(model_paths={HOLISTIC_TASK: "(no se carga)"},
                            reloj=reloj or RelojFalso(), landmarker=falso)
    return ext, falso


class ModeloFalso:
    """Imita model.input_shape y model.predict; devuelve una distribución fija."""

    def __init__(self, n_rasgos: int, id_clase: int = 7, confianza: float = 0.9, pasos: int = 30):
        self.input_shape = (None, pasos, n_rasgos)
        self.id_clase = id_clase
        self.confianza = confianza
        self.entradas: list[np.ndarray] = []

    def predict(self, X, verbose=0):
        self.entradas.append(np.array(X, copy=True))
        prob = np.full(50, (1.0 - self.confianza) / 49, dtype=np.float32)
        prob[self.id_clase] = self.confianza
        return np.repeat(prob[np.newaxis], X.shape[0], axis=0)


class TestManoAusenteD07(unittest.TestCase):
    """D-07: el prototipo trata el fotograma sin una mano igual que preprocess_lsc50.py
    (--mano-ausente). Ventana de 30 fotogramas (el remuestreo 30 -> 30 deja los pasos tal cual);
    el fotograma 5 llega con la mano derecha ausente (ExtractorFalso: manos=1 => solo izquierda)."""

    def _entrada(self, mano_ausente):
        modelo = ModeloFalso(N_FEATURES)
        patron = lambda i: (True, 1 if i == 5 else 2)
        tr = LSCTranslator(modelo=modelo, extractor=ExtractorFalso(patron), tts=None, tts_mode="none",
                           registro=False, ventana_frames=30, calentar=False, mano_ausente=mano_ausente)
        reg = correr_ventana(tr, n=30)
        self.assertIsNotNone(reg)
        self.assertEqual(reg["mano_ausente"], mano_ausente)
        tr.close()
        return modelo.entradas[0]

    def test_mascara_pone_en_ceros_el_bloque_tras_normalizar(self):
        X = self._entrada("mascara")
        self.assertEqual(X.shape, (1, 30, N_FEATURES))
        der = X[0, :, IDX_MANO_DER]
        self.assertTrue(np.all(der[5] == 0.0))                       # la mano ausente queda en ceros
        self.assertTrue(np.all(np.any(der[[4, 6]] != 0.0, axis=1)))  # las presentes, no
        self.assertTrue(np.any(X[0, 5, IDX_POSE] != 0.0))            # la postura del fotograma se conserva
        self.assertTrue(np.any(X[0, 5, IDX_MANO_IZQ] != 0.0))

    def test_ceros_deja_el_vector_artificial(self):
        X = self._entrada("ceros")
        der = X[0, 5, IDX_MANO_DER].reshape(21, 3)
        # -ref/escala: x e y quedan como constantes no nulas (z puede ser 0 porque los hombros
        # del extractor falso tienen z = 0, y (0 - 0)/escala = 0)
        self.assertFalse(np.any(der[:, :2] == 0.0))
        self.assertEqual(len(np.unique(der[:, 0])), 1)
        self.assertGreater(float(np.abs(der[:, :2]).max()), 1.0)

    def test_descartar_quita_el_fotograma(self):
        X = self._entrada("descartar")
        self.assertEqual(X.shape, (1, 30, N_FEATURES))               # 29 fotogramas remuestreados a 30
        der = X[0, :, IDX_MANO_DER]
        self.assertFalse(np.any(np.all(der == 0.0, axis=1)))         # ningún bloque de mano en ceros
        # Revisión D-07: y, a diferencia de «ceros», ningún paso conserva el vector artificial
        # -ref/escala (x = -2.5 con los hombros del extractor falso; las manos presentes quedan
        # en |x| <= 1): el fotograma 5 se quitó, no se normalizó.
        self.assertLess(float(np.abs(der.reshape(30, 21, 3)[:, :, 0]).max()), 1.5)

    def test_valor_invalido(self):
        with self.assertRaises(ValueError):
            LSCTranslator(modelo=ModeloFalso(N_FEATURES), extractor=ExtractorFalso(), tts=None,
                          tts_mode="none", registro=False, calentar=False, mano_ausente="otra")


class TTSFalso:
    mode = "falso"

    def __init__(self):
        self.hablado: list[str] = []
        self.limpiado = False

    def speak_async(self, texto: str):
        self.hablado.append(texto)

    def cleanup(self):
        self.limpiado = True


class TTSFalsoPorArchivo:
    """Imita la interfaz de src/tts.py en modo edge/gtts: prepare() sintetiza a archivo y
    play_prepared_async() emite la orden. La síntesis es inmediata en la prueba."""
    mode = "edge"

    def __init__(self):
        self.preparado: list[str] = []
        self.ordenes = 0
        self._listo = False

    def prepare(self, texto: str):
        self.preparado.append(texto)
        self._listo = True

    @property
    def is_prepared(self) -> bool:
        return self._listo

    def play_prepared_async(self):
        self.ordenes += 1
        self._listo = False

    def speak_async(self, texto: str):
        raise AssertionError("con motor por archivo debe usarse prepare/play_prepared_async")


class TTSFalsoOffline:
    """Imita src/tts.py en modo offline desde el 15-sep-2026: prepare() sintetiza a archivo,
    esperar_preparado() espera el resultado y play_prepared_async() emite la orden."""
    mode = "offline"
    soporta_preparar = True

    def __init__(self, sintetiza=True):
        self.sintetiza = sintetiza
        self.eventos: list[str] = []

    def prepare(self, texto: str):
        self.eventos.append(f"prepare:{texto}")

    def esperar_preparado(self, tope):
        self.eventos.append("esperar")
        return self.sintetiza

    @property
    def is_prepared(self) -> bool:
        return self.sintetiza

    def play_prepared_async(self):
        self.eventos.append("orden")

    def speak_async(self, texto: str):
        raise AssertionError("offline debe sintetizar antes de la orden")


def frame_sintetico(i: int = 0) -> np.ndarray:
    return np.full((48, 64, 3), (i * 7) % 255, dtype=np.uint8)


def construir(tmp, n_rasgos=450, confianza=0.9, patron=None, reloj=None, **kw):
    """Traductor con dobles inyectados; el registro va a una carpeta temporal."""
    params = dict(
        modelo=ModeloFalso(n_rasgos, confianza=confianza),
        extractor=ExtractorFalso(patron),
        tts=TTSFalso(),
        reloj=reloj or RelojFalso(),
        registro_dir=Path(tmp) / "registros",
        ventana_frames=10,
        min_frames=5,
        umbral=0.5,
        max_sin_manos=0.5,
        calentar=False,
    )
    params.update(kw)
    with redirect_stdout(io.StringIO()):
        return LSCTranslator(**params)


def correr_ventana(tr, n=None, c_fn=None):
    """Abre una ventana y alimenta fotogramas hasta que se cierre; devuelve el registro."""
    tr.abrir_ventana()
    n = n if n is not None else tr.ensayo.ventana_frames + 5
    for i in range(n):
        res = tr.procesar_fotograma(frame_sintetico(i), c_i=c_fn(i) if c_fn else None)
        if res.registro is not None:
            return res.registro
    return None


def archivos_prohibidos(*carpetas) -> list[str]:
    malos = []
    for carpeta in carpetas:
        for raiz, _dirs, archivos in os.walk(carpeta):
            for a in archivos:
                if Path(a).suffix.lower() in EXT_PROHIBIDAS:
                    malos.append(os.path.join(raiz, a))
    return malos


# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------

class TestInstrumentacion(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_proto_")

    def test_marcas_monotonas_y_etapas_suman_exacto(self):
        tr = construir(self.tmp)
        reg = correr_ventana(tr)
        self.assertEqual(reg["estado"], "aceptada")
        # Marcas: t0 < c_1 <= e_1 < c_2 ... c_N = t1 <= e_N < t2 < t3 < t4 < t5
        secuencia = [reg["t0"]]
        for c, e in zip(reg["c"], reg["e"]):
            secuencia.extend([c, e])
        secuencia.extend([reg["t2"], reg["t3"], reg["t4"], reg["t5"]])
        self.assertTrue(all(a < b for a, b in zip(secuencia, secuencia[1:])),
                        "las marcas del reloj monótono deben ser crecientes")
        self.assertEqual(reg["t1"], reg["c"][-1], "t1 = c_N por definición")
        self.assertTrue(all(isinstance(v, int) for v in [reg["t0"], reg["t1"], reg["t5"]] + reg["c"] + reg["e"]))
        # Las seis etapas suman exacto L_total (enteros en ns)
        etapas = reg["etapas_ns"]
        self.assertEqual(set(etapas), set(ETAPAS))
        self.assertEqual(sum(etapas[k] for k in ETAPAS), reg["L_total_ns"])
        self.assertEqual(reg["L_total_ns"], reg["t5"] - reg["t0"])
        # L_total = W + L_resp
        self.assertEqual(reg["W_ns"], reg["t1"] - reg["t0"])
        self.assertEqual(reg["L_resp_ns"], reg["t5"] - reg["t1"])
        self.assertEqual(reg["L_total_ns"], reg["W_ns"] + reg["L_resp_ns"])
        # Costo del extractor por fotograma y tasa efectiva
        costos = [e - c for c, e in zip(reg["c"], reg["e"])]
        self.assertEqual(reg["costo_extractor_ns"]["n"], reg["N"])
        self.assertAlmostEqual(reg["costo_extractor_ns"]["media"], float(np.mean(costos)))
        self.assertAlmostEqual(reg["costo_extractor_ns"]["p90"], float(np.percentile(costos, 90)))
        # Tasa efectiva con la fórmula literal de §2.2.7: N/(c_N - c_1)
        self.assertAlmostEqual(reg["tasa_efectiva_fps"], reg["N"] * 1e9 / (reg["c"][-1] - reg["c"][0]))
        self.assertTrue(all(v > 0 for v in etapas.values()))

    def test_velocidades_solo_si_el_modelo_espera_450(self):
        tr450 = construir(self.tmp, n_rasgos=450)
        self.assertTrue(tr450.usa_velocidades)
        reg = correr_ventana(tr450)
        self.assertEqual(reg["estado"], "aceptada")
        X = tr450.modelo.entradas[-1]
        self.assertEqual(X.shape, (1, 30, 450))
        pos, vel = X[..., :225], X[..., 225:]
        np.testing.assert_array_equal(vel[:, 0, :], 0.0)
        np.testing.assert_allclose(vel[:, 1:, :], pos[:, 1:, :] - pos[:, :-1, :], rtol=1e-6, atol=1e-6)
        self.assertTrue(reg["usa_velocidades"])
        self.assertEqual(reg["n_rasgos_modelo"], 450)

        tr225 = construir(self.tmp, n_rasgos=225)
        self.assertFalse(tr225.usa_velocidades)
        reg = correr_ventana(tr225)
        self.assertEqual(tr225.modelo.entradas[-1].shape, (1, 30, 225))
        self.assertFalse(reg["usa_velocidades"])

        with self.assertRaises(ValueError):
            construir(self.tmp, n_rasgos=300)

    def test_preprocesamiento_remuestrea_con_la_funcion_del_entrenamiento(self):
        from src.train_final_100 import resample_sequence
        tr = construir(self.tmp, n_rasgos=225, ventana_frames=12)
        tr.abrir_ventana()
        for i in range(12):
            res = tr.procesar_fotograma(frame_sintetico(i))
            if res.registro is not None:
                break
        X = tr.modelo.entradas[-1][0]
        # Reconstruir el esperado con las mismas funciones: normalizar → resample
        ext = ExtractorFalso()
        seq = np.stack([ext.extract(frame_sintetico(i)).keypoints for i in range(12)])
        esperado = resample_sequence(normalize_sequence(seq), 30)
        np.testing.assert_allclose(X, esperado, rtol=1e-6, atol=1e-6)


class TestCoherenciaConElCorpus(unittest.TestCase):
    """D-05: el prototipo usa exactamente el mismo extractor (HolisticLandmarker), el mismo
    modelo .task y las mismas opciones que la re-extracción del corpus (src/reextract_lsc50.py)."""

    def test_opciones_del_extractor_iguales_a_reextract(self):
        from src import translator, reextract_lsc50
        if not hasattr(reextract_lsc50, "OPCIONES_HOLISTIC"):
            self.skipTest("src/reextract_lsc50.py aún no define OPCIONES_HOLISTIC "
                          "(la pieza A de D-05 no ha terminado su reescritura)")
        self.assertEqual(translator.OPCIONES_HOLISTIC, reextract_lsc50.OPCIONES_HOLISTIC)
        if hasattr(reextract_lsc50, "MODEL_URLS"):
            self.assertEqual(translator.MODEL_URLS[HOLISTIC_TASK],
                             reextract_lsc50.MODEL_URLS.get(HOLISTIC_TASK), "URL del modelo holistic")
        if hasattr(reextract_lsc50, "MODELS_DIR"):
            self.assertEqual(translator.MODELS_DIR, reextract_lsc50.MODELS_DIR)
        codigo = (ROOT_DIR / "src" / "translator.py").read_text(encoding="utf-8")
        self.assertIn("Delegate.CPU", codigo, "delegado CPU explícito como en reextract_lsc50.py")

    def test_opciones_holistic_son_las_aprobadas(self):
        """El diccionario aprobado en D-05, literal; MODEL_URLS solo contiene el modelo holistic."""
        self.assertEqual(OPCIONES_HOLISTIC, dict(
            min_face_detection_confidence=0.5, min_face_suppression_threshold=0.5,
            min_face_landmarks_confidence=0.5, min_pose_detection_confidence=0.5,
            min_pose_suppression_threshold=0.5, min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5, output_face_blendshapes=False,
            output_segmentation_mask=False))
        from src import translator
        self.assertEqual(set(translator.MODEL_URLS), {HOLISTIC_TASK})
        self.assertTrue(translator.MODEL_URLS[HOLISTIC_TASK].startswith(
            "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"))
        self.assertEqual(N_FEATURES, 225)
        self.assertEqual(N_FACE_FEATURES, 1404)


class TestExtractorHolistic(unittest.TestCase):
    """LandmarkExtractor (D-05) con un HolisticLandmarker FALSO inyectado por constructor:
    distribución de los 225 rasgos, ceros y banderas sin detección, rostro y marcas de tiempo."""

    @classmethod
    def setUpClass(cls):
        try:
            import mediapipe  # noqa: F401  (LandmarkExtractor lo importa para construir mp.Image)
        except ImportError as exc:
            raise unittest.SkipTest(f"mediapipe no disponible: {exc}")

    def test_distribucion_de_los_225_rasgos_y_rostro(self):
        res = resultado_completo(semilla=1)
        ext, falso = extractor_con_falso([res])
        e = ext.extract(frame_sintetico(0))
        self.assertIsInstance(e, Extraccion)
        self.assertEqual(e.keypoints.shape, (225,))
        self.assertEqual(e.keypoints.dtype, np.float32)
        # Orden fijo: pose (0:99), mano izquierda (99:162), mano derecha (162:225)
        np.testing.assert_array_equal(e.keypoints[IDX_POSE], aplanado(res.pose_landmarks))
        np.testing.assert_array_equal(e.keypoints[IDX_MANO_IZQ], aplanado(res.left_hand_landmarks))
        np.testing.assert_array_equal(e.keypoints[IDX_MANO_DER], aplanado(res.right_hand_landmarks))
        self.assertEqual((IDX_POSE.start, IDX_POSE.stop), (0, 99))
        self.assertEqual((IDX_MANO_IZQ.start, IDX_MANO_IZQ.stop), (99, 162))
        self.assertEqual((IDX_MANO_DER.start, IDX_MANO_DER.stop), (162, 225))
        self.assertTrue(e.pose_detectada)
        self.assertEqual(e.n_manos, 2)
        # Rostro: 468 x 3 = 1404 valores, los 468 PRIMEROS puntos (sin los 10 de iris)
        self.assertIsNotNone(e.rostro)
        self.assertEqual(e.rostro.shape, (1404,))
        self.assertEqual(e.rostro.dtype, np.float32)
        np.testing.assert_array_equal(e.rostro, aplanado(res.face_landmarks[:N_FACE]))
        self.assertEqual(len(res.face_landmarks), 478)
        # El objeto crudo se conserva para dibujar y no hay campos del extractor antiguo
        self.assertIs(e.holistic_result, res)
        self.assertFalse(hasattr(e, "pose_result")); self.assertFalse(hasattr(e, "hand_result"))
        self.assertIn("HolisticLandmarker", ext.descripcion)
        self.assertIn("225 rasgos + rostro", ext.descripcion)
        ext.close()
        self.assertTrue(falso.cerrado)

    def test_ceros_y_banderas_cuando_falta_una_mano_o_la_pose(self):
        s = 5
        solo_derecha = ResultadoHolisticoFalso(pose=puntos_falsos(N_POSE, s), der=puntos_falsos(N_HAND, s + 1))
        solo_izquierda = ResultadoHolisticoFalso(pose=puntos_falsos(N_POSE, s), izq=puntos_falsos(N_HAND, s + 2))
        sin_manos = ResultadoHolisticoFalso(pose=puntos_falsos(N_POSE, s), rostro=puntos_falsos(478, s + 3))
        sin_pose = ResultadoHolisticoFalso(izq=puntos_falsos(N_HAND, s + 4), der=puntos_falsos(N_HAND, s + 5))
        nada = ResultadoHolisticoFalso()
        ext, _ = extractor_con_falso([solo_derecha, solo_izquierda, sin_manos, sin_pose, nada])

        e = ext.extract(frame_sintetico(0))
        self.assertTrue(e.pose_detectada); self.assertEqual(e.n_manos, 1)
        np.testing.assert_array_equal(e.keypoints[IDX_MANO_IZQ], 0.0)
        np.testing.assert_array_equal(e.keypoints[IDX_MANO_DER], aplanado(solo_derecha.right_hand_landmarks))

        e = ext.extract(frame_sintetico(1))
        self.assertEqual(e.n_manos, 1)
        np.testing.assert_array_equal(e.keypoints[IDX_MANO_DER], 0.0)
        np.testing.assert_array_equal(e.keypoints[IDX_MANO_IZQ], aplanado(solo_izquierda.left_hand_landmarks))

        e = ext.extract(frame_sintetico(2))
        self.assertTrue(e.pose_detectada); self.assertEqual(e.n_manos, 0)
        np.testing.assert_array_equal(e.keypoints[99:], 0.0)
        self.assertIsNotNone(e.rostro, "el rostro no depende de las manos")

        e = ext.extract(frame_sintetico(3))
        self.assertFalse(e.pose_detectada); self.assertEqual(e.n_manos, 2)
        np.testing.assert_array_equal(e.keypoints[IDX_POSE], 0.0)
        self.assertTrue(np.any(e.keypoints[IDX_MANO_IZQ] != 0) and np.any(e.keypoints[IDX_MANO_DER] != 0))

        e = ext.extract(frame_sintetico(4))
        self.assertFalse(e.pose_detectada); self.assertEqual(e.n_manos, 0)
        np.testing.assert_array_equal(e.keypoints, np.zeros(225, np.float32))
        self.assertEqual(e.keypoints.shape, (225,))

    def test_rostro_es_none_sin_rostro(self):
        con_pose_sin_rostro = ResultadoHolisticoFalso(pose=puntos_falsos(N_POSE, 9),
                                                     izq=puntos_falsos(N_HAND, 10), der=puntos_falsos(N_HAND, 11))
        ext, _ = extractor_con_falso([con_pose_sin_rostro, ResultadoHolisticoFalso()])
        e = ext.extract(frame_sintetico(0))
        self.assertIsNone(e.rostro)
        self.assertTrue(e.pose_detectada); self.assertEqual(e.n_manos, 2)   # los 225 no dependen del rostro
        self.assertIsNone(ext.extract(frame_sintetico(1)).rostro)

    def test_marca_de_tiempo_estrictamente_creciente(self):
        # Reloj congelado (paso 0): el modo VIDEO exige ts crecientes, así que se fuerza +1
        ext, falso = extractor_con_falso([resultado_completo(i) for i in range(6)], reloj=RelojFalso(paso_ns=0))
        for i in range(6):
            ext.extract(frame_sintetico(i))
        self.assertEqual(falso.timestamps, [0, 1, 2, 3, 4, 5])
        # Reloj a 0,5 ms por lectura: varios fotogramas caen en el mismo ms y aun así crece
        ext, falso = extractor_con_falso([resultado_completo(i) for i in range(8)], reloj=RelojFalso(paso_ns=500_000))
        for i in range(8):
            ext.extract(frame_sintetico(i))
        self.assertTrue(all(a < b for a, b in zip(falso.timestamps, falso.timestamps[1:])), falso.timestamps)
        self.assertTrue(all(isinstance(t, int) and t >= 0 for t in falso.timestamps))
        # Reloj de cámara a 40 ms: las marcas siguen al reloj real (ms desde el arranque)
        ext, falso = extractor_con_falso([resultado_completo(i) for i in range(4)], reloj=RelojFalso(paso_ns=40_000_000))
        for i in range(4):
            ext.extract(frame_sintetico(i))
        self.assertEqual(falso.timestamps, [40, 80, 120, 160])

    def test_create_from_options_recibe_opciones_holistic_cpu_y_video(self):
        """Sin inyección, el extractor construye el HolisticLandmarker real con OPCIONES_HOLISTIC,
        el modelo holistic_landmarker.task, delegado CPU y modo VIDEO (se intercepta la fábrica)."""
        import mediapipe as mp
        capturadas = []

        def fabrica(options):
            capturadas.append(options)
            return HolisticFalso([])

        with mock.patch.object(mp.tasks.vision.HolisticLandmarker, "create_from_options", side_effect=fabrica):
            ext = LandmarkExtractor({HOLISTIC_TASK: "/ruta/falsa/holistic_landmarker.task"}, reloj=RelojFalso())
        self.assertEqual(len(capturadas), 1)
        o = capturadas[0]
        self.assertIsInstance(o, mp.tasks.vision.HolisticLandmarkerOptions)
        self.assertEqual(o.running_mode, mp.tasks.vision.RunningMode.VIDEO)
        self.assertEqual(o.base_options.delegate, mp.tasks.BaseOptions.Delegate.CPU)
        self.assertEqual(o.base_options.model_asset_path, "/ruta/falsa/holistic_landmarker.task")
        for clave, valor in OPCIONES_HOLISTIC.items():
            self.assertEqual(getattr(o, clave), valor, clave)
        self.assertIsInstance(ext.landmarker, HolisticFalso)
        self.assertEqual(ext.descripcion, f"HolisticLandmarker {mp.__version__}, CPU, 225 rasgos + rostro")

    def test_pipeline_completo_con_extractor_real_y_holistic_falso(self):
        """LSCTranslator con el LandmarkExtractor real (HolisticLandmarker falso): la ventana se
        cierra, los 225 rasgos llegan al modelo y el rostro NO entra en la inferencia."""
        resultados = []
        for i in range(12):
            r = resultado_completo(i)
            # hombros separados para que la normalización esté bien definida
            r.pose_landmarks[11] = PuntoFalso(0.6, 0.4, 0.0); r.pose_landmarks[12] = PuntoFalso(0.4, 0.4, 0.0)
            resultados.append(r)
        ext, _ = extractor_con_falso(resultados)
        tmp = tempfile.mkdtemp(prefix="tesis_proto_")
        tr = construir(tmp, n_rasgos=225, extractor=ext, ventana_frames=10)
        reg = correr_ventana(tr)
        self.assertEqual(reg["estado"], "aceptada")
        self.assertEqual(reg["frames_con_dos_manos"], 10)
        self.assertEqual(reg["frames_sin_pose"], 0)
        self.assertEqual(tr.modelo.entradas[-1].shape, (1, 30, 225))
        self.assertIn("HolisticLandmarker", reg["extractor"])


@unittest.skipUnless((ROOT_DIR / "models" / "mediapipe" / HOLISTIC_TASK).exists()
                     and (ROOT_DIR / "data" / "raw" / "LSC50" / "VIDEOS" / "COLOR_BODY" / "0000_0000_0000.avi").exists(),
                     "modelo holistic_landmarker.task o video del corpus no disponibles")
class TestExtractorHolisticReal(unittest.TestCase):
    """HolisticLandmarker REAL sobre los 10 primeros fotogramas de un video del corpus (solo
    lectura): el mismo modelo (SHA256) que la re-extracción, detecciones y forma de la salida."""

    VIDEO = ROOT_DIR / "data" / "raw" / "LSC50" / "VIDEOS" / "COLOR_BODY" / "0000_0000_0000.avi"

    def test_modelo_en_disco_es_el_aprobado(self):
        ruta = ROOT_DIR / "models" / "mediapipe" / HOLISTIC_TASK
        self.assertEqual(hashlib.sha256(ruta.read_bytes()).hexdigest(), MODEL_SHA256[HOLISTIC_TASK])

    def test_ensure_models_avisa_si_el_sha256_no_es_el_de_d05(self):
        """Añadida en la revisión de la pieza B: el arranque real (extractor no inyectado) pasa por
        _ensure_models_downloaded, que debe comprobar el SHA256 del .task y avisar si difiere."""
        from src import translator
        with redirect_stdout(io.StringIO()) as buf:
            rutas = translator._ensure_models_downloaded()
        self.assertEqual(Path(rutas[HOLISTIC_TASK]), ROOT_DIR / "models" / "mediapipe" / HOLISTIC_TASK)
        self.assertNotIn("[AVISO]", buf.getvalue())
        with mock.patch.dict(translator.MODEL_SHA256, {HOLISTIC_TASK: "0" * 64}), \
             redirect_stdout(io.StringIO()) as buf:
            translator._ensure_models_downloaded()
        self.assertIn("[AVISO]", buf.getvalue())
        self.assertIn("D-05", buf.getvalue())

    def test_diez_fotogramas_del_corpus(self):
        import cv2
        with redirect_stdout(io.StringIO()):
            ext = LandmarkExtractor({HOLISTIC_TASK: str(ROOT_DIR / "models" / "mediapipe" / HOLISTIC_TASK)})
        cap = cv2.VideoCapture(str(self.VIDEO))
        self.assertTrue(cap.isOpened())
        salidas = []
        try:
            for _ in range(10):
                ok, bgr = cap.read()
                self.assertTrue(ok)
                salidas.append(ext.extract(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()
            ext.close()
        self.assertEqual(len(salidas), 10)
        for e in salidas:
            self.assertEqual(e.keypoints.shape, (225,)); self.assertEqual(e.keypoints.dtype, np.float32)
            self.assertTrue(e.pose_detectada)
            self.assertTrue(np.any(e.keypoints[IDX_POSE] != 0))
            self.assertIsNotNone(e.rostro); self.assertEqual(e.rostro.shape, (1404,))
            self.assertEqual(len(e.holistic_result.face_landmarks), 478)
        # En el reposo inicial del clip las dos manos son visibles (la persona las tiene al frente)
        self.assertGreaterEqual(sum(1 for e in salidas if e.n_manos == 2), 8)
        # Coordenadas normalizadas al cuadro (x, y de los puntos detectados dentro de [-0.5, 1.5])
        puntos = np.stack([e.keypoints for e in salidas]).reshape(10, -1, 3)
        detectados = puntos[puntos.any(axis=2)]
        self.assertGreater(len(detectados), 0)
        self.assertTrue(np.all((detectados[:, :2] >= -0.5) & (detectados[:, :2] <= 1.5)))


class TestSintesisAntesDeT5(unittest.TestCase):
    """§2.2.7: t5 es la emisión de la orden de reproducción con el audio ya sintetizado.
    El registro declara si eso se cumplió (motor por archivo) o no (pyttsx3)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_proto_")

    def test_speak_async_declara_sintesis_no_incluida(self):
        tr = construir(self.tmp)                      # TTSFalso: solo speak_async
        reg = correr_ventana(tr)
        self.assertEqual(reg["estado"], "aceptada")
        self.assertIs(reg["tts_sintesis_antes_de_t5"], False)
        self.assertIsNotNone(reg["etapas_ns"]["sintesis"])

    def test_motor_por_archivo_sintetiza_antes_de_la_orden(self):
        tts = TTSFalsoPorArchivo()
        tr = construir(self.tmp, tts=tts)
        reg = correr_ventana(tr)
        self.assertEqual(reg["estado"], "aceptada")
        self.assertIs(reg["tts_sintesis_antes_de_t5"], True)
        self.assertEqual(tts.preparado, [reg["enunciado"]])
        self.assertEqual(tts.ordenes, 1)
        # t5 se sella DESPUÉS de emitir la orden y la síntesis queda dentro de t5 - t4
        self.assertGreater(reg["etapas_ns"]["sintesis"], 0)
        # En modo oración, el turno también lo declara
        tr2 = construir(self.tmp, tts=TTSFalsoPorArchivo(), modo_oracion=True)
        correr_ventana(tr2)
        turno = tr2.fin_de_turno()
        self.assertIs(turno["tts_sintesis_antes_de_orden"], True)

    def test_offline_sintetiza_antes_de_la_orden(self):
        tts = TTSFalsoOffline()
        tr = construir(self.tmp, tts=tts)
        reg = correr_ventana(tr)
        self.assertEqual(reg["estado"], "aceptada")
        self.assertIs(reg["tts_sintesis_antes_de_t5"], True)
        self.assertEqual(tts.eventos, [f"prepare:{reg['enunciado']}", "esperar", "orden"])
        self.assertIs(reg["tts_ocupado_en_t5"], False)
        tts.is_speaking = True                      # otra locución sonando: se registra
        self.assertIs(correr_ventana(construir(self.tmp, tts=tts))["tts_ocupado_en_t5"], True)

    def test_offline_si_la_sintesis_falla_se_declara_y_se_emite_igual(self):
        tts = TTSFalsoOffline(sintetiza=False)
        reg = correr_ventana(construir(self.tmp, tts=tts))
        self.assertIs(reg["tts_sintesis_antes_de_t5"], False)
        self.assertEqual(tts.eventos[-1], "orden")

    def test_sin_audio_es_none_y_rechazo_no_lo_marca(self):
        tr = construir(self.tmp, tts=None)
        reg = correr_ventana(tr)
        self.assertEqual(reg["estado"], "aceptada")
        self.assertIsNone(reg["tts_sintesis_antes_de_t5"])
        tr2 = construir(self.tmp, confianza=0.3)
        self.assertIsNone(correr_ventana(tr2)["tts_sintesis_antes_de_t5"])


class TestEstadosDeError(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_proto_")

    def test_confianza_bajo_umbral(self):
        tr = construir(self.tmp, confianza=0.3, umbral=0.5)
        reg = correr_ventana(tr)
        self.assertEqual(reg["estado"], "rechazada_confianza")
        self.assertIn("repita", reg["motivo"])
        self.assertEqual(tr.tts.hablado, [], "sin audio cuando la seña no se reconoce")
        self.assertIsNone(reg["t4"]); self.assertIsNone(reg["t5"])
        self.assertIsNotNone(reg["t3"]); self.assertIsNotNone(reg["confianza"])
        self.assertIsNone(reg["L_total_ns"]); self.assertIsNone(reg["enunciado"])
        self.assertEqual(tr.contadores["rechazada_confianza"], 1)
        self.assertEqual(reg["contadores"]["rechazada_confianza"], 1)
        # El umbral es un parámetro: con 0.2 la misma distribución se acepta
        tr2 = construir(self.tmp, confianza=0.3, umbral=0.2)
        self.assertEqual(correr_ventana(tr2)["estado"], "aceptada")

    def test_manos_no_detectadas_no_infiere(self):
        # 6 de 10 fotogramas sin manos (60 % > 50 %)
        tr = construir(self.tmp, patron=lambda i: (True, 0 if i % 10 < 6 else 2))
        reg = correr_ventana(tr)
        self.assertEqual(reg["estado"], "rechazada_manos")
        self.assertEqual(len(tr.modelo.entradas), 0, "no debe inferirse sin manos")
        self.assertEqual(tr.tts.hablado, [])
        self.assertEqual(reg["frames_sin_manos"], 6)
        self.assertAlmostEqual(reg["proporcion_sin_manos"], 0.6)
        self.assertIsNone(reg["t2"]); self.assertIsNone(reg["clase"])
        self.assertIsNotNone(reg["W_ns"]); self.assertIsNotNone(reg["etapas_ns"]["captura"])
        self.assertEqual(tr.contadores["rechazada_manos"], 1)
        # 4 de 10 sin manos (40 % <= 50 %): sí se infiere
        tr2 = construir(self.tmp, patron=lambda i: (True, 0 if i % 10 < 4 else 2))
        self.assertEqual(correr_ventana(tr2)["estado"], "aceptada")

    def test_pose_no_detectada_no_infiere(self):
        tr = construir(self.tmp, patron=lambda i: (i % 10 >= 7, 2))   # 7 de 10 sin pose
        reg = correr_ventana(tr)
        self.assertEqual(reg["estado"], "rechazada_manos")
        self.assertIn("pose", reg["motivo"])
        self.assertEqual(reg["frames_sin_pose"], 7)
        self.assertEqual(len(tr.modelo.entradas), 0)

    def test_ventana_abortada_es_incompleta(self):
        tr = construir(self.tmp)
        tr.abrir_ventana()
        for i in range(3):
            tr.procesar_fotograma(frame_sintetico(i))
        self.assertTrue(tr.ventana_abierta)
        reg = tr.abortar_ventana()
        self.assertFalse(tr.ventana_abierta)
        self.assertEqual(reg["estado"], "rechazada_incompleta")
        self.assertEqual(reg["cierre"], "abortada")
        self.assertEqual(reg["N"], 3)
        self.assertEqual(len(tr.modelo.entradas), 0)
        self.assertEqual(tr.contadores["rechazada_incompleta"], 1)
        self.assertIsNone(tr.abortar_ventana(), "sin ventana abierta no hay nada que abortar")

    def test_tope_de_tiempo_con_pocos_fotogramas_es_incompleta(self):
        reloj = RelojFalso(paso_ns=1_000_000)
        tr = construir(self.tmp, reloj=reloj, ventana_frames=50, min_frames=15, ventana_max_seg=0.010)
        tr.abrir_ventana()
        reg = None
        for i in range(50):
            reloj.avanzar(4_000_000)   # la cámara entrega cada 4 ms → tope de 10 ms a los 3 fotogramas
            res = tr.procesar_fotograma(frame_sintetico(i))
            if res.registro is not None:
                reg = res.registro
                break
        self.assertIsNotNone(reg)
        self.assertEqual(reg["cierre"], "tiempo")
        self.assertEqual(reg["estado"], "rechazada_incompleta")
        self.assertLess(reg["N"], 15)

    def test_nada_se_infiere_fuera_de_ventana(self):
        tr = construir(self.tmp)
        for i in range(50):
            res = tr.procesar_fotograma(frame_sintetico(i))
            self.assertFalse(res.en_ventana)
            self.assertIsNone(res.registro)
        self.assertEqual(len(tr.modelo.entradas), 0)
        self.assertEqual(tr.n_ensayos, 0)
        self.assertFalse(tr.registro_path.exists())
        self.assertEqual(tr.extractor.llamadas, 50, "el extractor sí corre en reposo (para la vista)")

    def test_no_se_abre_una_segunda_ventana(self):
        tr = construir(self.tmp)
        self.assertIsNotNone(tr.abrir_ventana())
        self.assertIsNone(tr.abrir_ventana())
        self.assertEqual(tr.n_ensayos, 1)


class TestRegistroYArchivos(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_proto_")

    def _sesion_completa(self):
        """Un ensayo por estado; devuelve el traductor."""
        patrones = {
            "aceptada": (0.9, lambda i: (True, 2)),
            "rechazada_confianza": (0.3, lambda i: (True, 2)),
            "rechazada_manos": (0.9, lambda i: (True, 0)),
        }
        regs = []
        tr = None
        for estado, (conf, patron) in patrones.items():
            tr = construir(self.tmp, confianza=conf, patron=patron)
            regs.append(correr_ventana(tr))
        tr.abrir_ventana(); tr.procesar_fotograma(frame_sintetico(0)); regs.append(tr.abortar_ventana())
        return tr, regs

    def test_jsonl_con_todos_los_campos(self):
        tr, regs = self._sesion_completa()
        self.assertEqual([r["estado"] for r in regs], list(ESTADOS))
        self.assertTrue(tr.registro_path.exists(), "debe crear la carpeta y el archivo")
        self.assertRegex(tr.registro_path.name, r"^latencia_\d{4}-\d{2}-\d{2}\.jsonl$")
        with open(tr.registro_path, encoding="utf-8") as f:
            lineas = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lineas), 4)
        campos = {"tipo", "sesion", "id_ensayo", "fecha_hora", "modo", "modelo", "n_rasgos_modelo",
                  "usa_velocidades", "extractor", "tts_modo", "parametros", "t0", "c", "e", "t1", "t2",
                  "t3", "t4", "t5", "N", "cierre", "frames_con_deteccion", "frames_sin_manos",
                  "frames_sin_pose", "proporcion_sin_manos", "etapas_ns", "W_ns", "L_total_ns",
                  "L_resp_ns", "costo_extractor_ns", "tasa_efectiva_fps", "estado", "motivo",
                  "id_clase", "clase", "confianza", "top3", "enunciado", "contadores",
                  "tts_sintesis_antes_de_t5"}
        for l in lineas:
            self.assertTrue(campos <= set(l), f"faltan campos: {campos - set(l)}")
            self.assertEqual(set(l["etapas_ns"]), set(ETAPAS))
            self.assertEqual(set(l["contadores"]), set(ESTADOS))
            self.assertIn(l["estado"], ESTADOS)
        aceptada = lineas[0]
        self.assertEqual(len(aceptada["top3"]), 3)
        self.assertEqual(aceptada["top3"][0]["id"], 7)
        self.assertEqual(aceptada["clase"], "comer")   # SIGN_VOCABULARY[7] = "comer"
        self.assertEqual(aceptada["enunciado"], GrammarMapper().translate(["comer"]))
        self.assertEqual(aceptada["frames_con_deteccion"], aceptada["N"])

    def test_no_escribe_imagenes_ni_video(self):
        cwd = os.getcwd()
        trabajo = tempfile.mkdtemp(prefix="tesis_cwd_")
        try:
            os.chdir(trabajo)
            tr, _ = self._sesion_completa()
            tr.close()
        finally:
            os.chdir(cwd)
        self.assertEqual(archivos_prohibidos(self.tmp, trabajo), [])
        # Solo el JSONL en la carpeta de registros
        creados = [p.name for p in (Path(self.tmp) / "registros").iterdir()]
        self.assertTrue(all(n.endswith(".jsonl") for n in creados), creados)
        # Y en el código del prototipo no hay ninguna escritura de imagen o video
        for archivo in ("src/translator.py", "src/realtime.py"):
            codigo = (ROOT_DIR / archivo).read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"imwrite|VideoWriter|imencode", codigo), archivo)

    def test_registro_desactivado_no_escribe(self):
        tr = construir(self.tmp, registro=False)
        correr_ventana(tr)
        self.assertFalse(tr.registro_dir.exists())
        self.assertIsNotNone(tr.ultimo)

    def test_resumen_de_latencia(self):
        tr, _ = self._sesion_completa()
        tr2 = construir(self.tmp, modo_oracion=True)
        correr_ventana(tr2); correr_ventana(tr2); tr2.fin_de_turno()
        registros = latencia_resumen.leer_registros([str(tr.registro_dir)])
        self.assertEqual(len(registros), 7)   # 4 + 2 ensayos + 1 turno
        r = latencia_resumen.resumir(registros)
        self.assertEqual(r["n_ensayos"], 6)
        self.assertEqual(r["n_turnos"], 1)
        self.assertEqual(r["contadores"], {"aceptada": 3, "rechazada_confianza": 1,
                                           "rechazada_manos": 1, "rechazada_incompleta": 1})
        m = r["magnitudes_ms"]
        self.assertEqual(m["L_total"]["n"], 1)        # solo el aceptado en modo aislado llega a t5
        self.assertEqual(m["inferencia"]["n"], 4)     # aceptada aislada + confianza + 2 oración
        self.assertEqual(m["captura"]["n"], 6)
        self.assertEqual(m["W"]["n"], 6)
        for s in (m["L_total"], m["captura"], r["costo_extractor_por_fotograma_ms"], r["tasa_efectiva_fps"]):
            self.assertIsNotNone(s)
            for k in s:
                self.assertFalse(isinstance(s[k], float) and np.isnan(s[k]))
        self.assertEqual(r["turnos"]["n"], 1)
        # Bandera de síntesis: solo el aceptado en modo aislado emitió audio (TTSFalso: speak_async)
        self.assertEqual(r["sintesis_antes_de_t5"], {"con_audio": 1, "si": 0, "no": 1})
        texto = latencia_resumen.tabla(r)
        for etiqueta in ETAPAS + ("W", "L_total", "L_resp", "extractor/fotog.", "tasa efectiva", "Turnos"):
            self.assertIn(etiqueta, texto)
        json.dumps(r)   # serializable


class TestModoOracion(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_proto_")

    def test_acumula_glosas_y_fin_de_turno_explicito(self):
        tr = construir(self.tmp, modo_oracion=True)
        r1 = correr_ventana(tr)
        r2 = correr_ventana(tr)
        self.assertEqual(r1["estado"], "aceptada")
        self.assertEqual(tr.glosas_turno, ["comer", "comer"])
        self.assertEqual(tr.tts.hablado, [], "en modo oración no se sintetiza por ventana")
        self.assertIsNone(r1["t4"]); self.assertIsNone(r1["t5"]); self.assertIsNone(r1["L_total_ns"])
        self.assertEqual(r1["modo"], "oracion")
        self.assertEqual(r2["glosas_turno"], ["comer", "comer"])
        # Ventana rechazada no añade glosa
        tr.modelo.confianza = 0.2
        r3 = correr_ventana(tr)
        self.assertEqual(r3["estado"], "rechazada_confianza")
        self.assertEqual(len(tr.glosas_turno), 2)
        # Fin de turno: convierte con GrammarMapper y sintetiza UNA vez
        turno = tr.fin_de_turno()
        self.assertEqual(turno["tipo"], "turno")
        self.assertEqual(turno["glosas"], ["comer", "comer"])
        self.assertEqual(turno["enunciado"], GrammarMapper().translate(["comer", "comer"]))
        self.assertEqual(tr.tts.hablado, [turno["enunciado"]])
        self.assertGreater(turno["conversion_ns"], 0); self.assertGreater(turno["sintesis_ns"], 0)
        self.assertEqual(tr.glosas_turno, [])
        self.assertEqual(tr.enunciado_turno, turno["enunciado"])
        self.assertIsNone(tr.fin_de_turno(), "sin glosas no hay turno")
        # Reiniciar descarta sin sintetizar
        tr.modelo.confianza = 0.9
        correr_ventana(tr)
        tr.reiniciar_turno()
        self.assertEqual(tr.glosas_turno, [])
        self.assertEqual(len(tr.tts.hablado), 1)

    def test_modo_aislado_sintetiza_por_ventana(self):
        tr = construir(self.tmp, modo_oracion=False)
        reg = correr_ventana(tr)
        self.assertEqual(tr.tts.hablado, [reg["enunciado"]])
        self.assertEqual(tr.glosas_turno, [])


class TestVentanaAutomatica(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_proto_")

    def test_ventana_frames_auto_desde_tasa_medida(self):
        tr = construir(self.tmp, ventana_frames=0, ventana_seg=2.5, min_frames=5)
        # Reposo a 30 fps exactos (c_i cada 33 333 333 ns)
        for i in range(40):
            tr.procesar_fotograma(frame_sintetico(i), c_i=1 + i * 33_333_333)
        self.assertAlmostEqual(tr.fps_medida(), 30.0, places=3)
        ens = tr.abrir_ventana()
        self.assertEqual(ens.ventana_frames, 75)          # round(2.5 × 30)
        self.assertAlmostEqual(ens.fps_medida, 30.0, places=3)
        tr.abortar_ventana()
        # A 12 fps (bucle lento): 30 fotogramas
        for i in range(40):
            tr.procesar_fotograma(frame_sintetico(i), c_i=10**12 + i * 83_333_333)
        ens = tr.abrir_ventana()
        self.assertEqual(ens.ventana_frames, 30)
        tr.abortar_ventana()
        # Sin tasa medible se usa la última conocida o, si no hay, la supuesta (30 fps)
        tr2 = construir(self.tmp, ventana_frames=0, ventana_seg=2.0, min_frames=5)
        self.assertEqual(tr2.abrir_ventana().ventana_frames, 60)
        # --ventana-frames explícito manda
        tr3 = construir(self.tmp, ventana_frames=42)
        self.assertEqual(tr3.abrir_ventana().ventana_frames, 42)

    def test_registro_conserva_los_parametros(self):
        tr = construir(self.tmp, ventana_frames=10, ventana_seg=2.5, min_frames=5, umbral=0.6,
                       max_sin_manos=0.4)
        reg = correr_ventana(tr)
        p = reg["parametros"]
        self.assertEqual(p["ventana_frames_objetivo"], 10)
        self.assertEqual(p["umbral"], 0.6)
        self.assertEqual(p["max_sin_manos"], 0.4)
        self.assertEqual(p["ventana_max_seg"], 5.0)
        self.assertEqual(reg["cierre"], "frames")


@unittest.skipUnless((ROOT_DIR / "models" / "best_model.keras").exists()
                     and (ROOT_DIR / "models" / "best_model_original.keras").exists(),
                     "modelos reales no disponibles")
class TestModelosReales(unittest.TestCase):
    """Arranque con los dos modelos reales (450 y 225 rasgos) con extractor falso."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_proto_")

    def _arrancar(self, nombre, rasgos):
        import tensorflow as tf
        with redirect_stdout(io.StringIO()):
            modelo = tf.keras.models.load_model(str(ROOT_DIR / "models" / nombre))
            tr = LSCTranslator(model_path=str(ROOT_DIR / "models" / nombre), modelo=modelo,
                               extractor=ExtractorFalso(), tts=TTSFalso(),
                               registro_dir=Path(self.tmp) / "registros",
                               ventana_frames=20, min_frames=5)
        self.assertEqual(tr.n_rasgos, rasgos)
        self.assertEqual(tr.usa_velocidades, rasgos == 450)
        reg = correr_ventana(tr)
        self.assertIn(reg["estado"], ("aceptada", "rechazada_confianza"))
        self.assertEqual(len(reg["top3"]), 3)
        self.assertTrue(0.0 <= reg["confianza"] <= 1.0)
        self.assertGreater(reg["etapas_ns"]["inferencia"], 0)
        return tr, reg

    def test_best_model_450(self):
        tr, reg = self._arrancar("best_model.keras", 450)
        self.assertTrue(reg["usa_velocidades"])

    def test_best_model_original_225(self):
        tr, reg = self._arrancar("best_model_original.keras", 225)
        self.assertFalse(reg["usa_velocidades"])


# --- D-08 (15-sep-2026): representación declarada junto al modelo -------------------------------------

class ExtractorSecuencia:
    """Devuelve una secuencia fija de fotogramas (keypoints de 225, rostro de 1404 o None, pose, manos)."""

    descripcion = "extractor_secuencia"

    def __init__(self, fotogramas):
        self.fotogramas = fotogramas
        self.i = 0

    def extract(self, frame_rgb):
        kp, rostro, pose, manos = self.fotogramas[self.i % len(self.fotogramas)]
        self.i += 1
        return Extraccion(keypoints=kp.copy(), pose_detectada=pose, n_manos=manos,
                          rostro=None if rostro is None else rostro.copy(), holistic_result=None)

    def close(self):
        pass


def secuencia_sintetica(T=12, sin_der=(3, 7), sin_izq=(9,), sin_rostro=(5,), semilla=0, sin_pose=()):
    rng = np.random.default_rng(semilla)
    fotogramas = []
    for t in range(T):
        kp = rng.uniform(0.3, 0.7, size=N_FEATURES).astype(np.float32)
        kp[33:36] = [0.6 + 0.01 * t, 0.4, 0.0]; kp[36:39] = [0.4, 0.41, 0.0]      # hombros
        manos = 2
        if t in sin_der:
            kp[IDX_MANO_DER] = 0.0; manos -= 1
        if t in sin_izq:
            kp[IDX_MANO_IZQ] = 0.0; manos -= 1
        rostro = None if t in sin_rostro else rng.uniform(0.45, 0.55, size=N_FACE_FEATURES).astype(np.float32)
        if t in sin_pose:     # revisión D-08: sin postura, con manos y rostro presentes
            kp[:99] = 0.0
        fotogramas.append((kp, rostro, t not in sin_pose, manos))
    return fotogramas


def procedencia_para(representacion, mano_ausente="mascara", rostro_del_processed="cejas_labios", pasos=30):
    from src.representacion import n_rasgos
    n = n_rasgos(representacion)
    return {"representacion": representacion, "n_rasgos_entrada": 2 * n, "seq_length": pasos,
            "velocidades": True, "mano_ausente": mano_ausente, "rostro_del_processed": rostro_del_processed}


def camino_del_corpus(fotogramas, representacion, modo):
    """La misma secuencia por el preprocesamiento del corpus (preprocess_lsc50 con --rostro cejas_labios)."""
    from src import preprocess_lsc50 as pp
    from src.representacion import ROSTRO_SUBCONJUNTOS, columnas
    cols_r = np.array([i * 3 + c for i in ROSTRO_SUBCONJUNTOS["cejas_labios"] for c in range(3)])
    filas = [np.concatenate([kp, np.zeros(180, np.float32) if r is None else r[cols_r]]) for kp, r, _, _ in fotogramas]
    data = np.stack(filas).astype(np.float32)
    muestras = {"m": {"data": data, "sign_id": 0, "volunteer_id": 0, "repetition_id": 0, "n_frames": len(data)}}
    with redirect_stdout(io.StringIO()):
        pp.aplicar_regla_sin_postura(muestras)
        pp.aplicar_regla_mano_ausente(muestras, modo)
        pp.normalize_all_samples(muestras)
        if modo == "mascara":
            pp.aplicar_mascara_mano_ausente(muestras)
    return muestras["m"]["data"][:, columnas(representacion, "cejas_labios")]


class TestRepresentacionD08(unittest.TestCase):

    def _prototipo(self, fotogramas, procedencia, n_pasos, **kw):
        from src.train_final_100 import add_velocity  # noqa: F401  (mismo módulo que usa el prototipo)
        n = procedencia["n_rasgos_entrada"]
        modelo = ModeloFalso(n, pasos=n_pasos)
        tr = LSCTranslator(modelo=modelo, extractor=ExtractorSecuencia(fotogramas), tts=None, tts_mode="none",
                           registro=False, ventana_frames=len(fotogramas), min_frames=1, calentar=False,
                           procedencia=dict(procedencia, seq_length=n_pasos), **kw)
        with redirect_stdout(io.StringIO()):
            reg = correr_ventana(tr, n=len(fotogramas))
        tr.close()
        self.assertIsNotNone(reg)
        return modelo.entradas[0], reg, tr

    def test_equivalencia_numerica_con_el_corpus(self):
        from src.train_final_100 import add_velocity
        fot = secuencia_sintetica(sin_pose=(10,))
        for representacion in ("manos", "manos_postura", "manos_postura_rostro"):
            for modo in ("mascara", "descartar", "ceros"):
                with self.subTest(representacion=representacion, modo=modo):
                    corpus = camino_del_corpus(fot, representacion, modo)
                    X, reg, tr = self._prototipo(fot, procedencia_para(representacion, modo), n_pasos=len(corpus))
                    esperado = add_velocity(corpus[np.newaxis, ...].astype(np.float32))
                    self.assertEqual(X.shape, esperado.shape)
                    np.testing.assert_allclose(X, esperado, atol=1e-6)
                    self.assertEqual(reg["representacion"], representacion)
                    self.assertEqual(reg["mano_ausente"], modo)

    def test_manos_descartar_con_processed_con_rostro_omite_fotograma_sin_rostro(self):
        fot = secuencia_sintetica(sin_der=(), sin_izq=(), sin_rostro=(4,))
        corpus = camino_del_corpus(fot, "manos", "descartar")
        self.assertEqual(len(corpus), 11)
        X, _, tr = self._prototipo(fot, procedencia_para("manos", "descartar"), n_pasos=11)
        self.assertTrue(tr.arma_rostro)
        self.assertEqual(X.shape, (1, 11, 252))

    def test_mascara_manos_no_arma_rostro(self):
        _, _, tr = self._prototipo(secuencia_sintetica(), procedencia_para("manos", "mascara"), n_pasos=30)
        self.assertFalse(tr.arma_rostro)

    def test_sin_procedencia_se_deduce_del_ancho(self):
        for ancho, rep_esperada in ((252, "manos"), (450, "manos_postura"), (810, "manos_postura_rostro"),
                                    (225, "manos_postura")):
            with self.subTest(ancho=ancho), redirect_stdout(io.StringIO()) as salida:
                tr = LSCTranslator(modelo=ModeloFalso(ancho), extractor=ExtractorFalso(), tts=None,
                                   tts_mode="none", registro=False, calentar=False)
                self.assertEqual(tr.representacion, rep_esperada)
                self.assertIsNone(tr.procedencia_ruta)
                self.assertIn("AVISO", salida.getvalue())
        with self.assertRaises(ValueError):
            LSCTranslator(modelo=ModeloFalso(300), extractor=ExtractorFalso(), tts=None, tts_mode="none",
                          registro=False, calentar=False)

    def test_incoherencias_se_rechazan(self):
        with self.assertRaises(ValueError):   # la procedencia declara otro ancho
            LSCTranslator(modelo=ModeloFalso(450), extractor=ExtractorFalso(), tts=None, tts_mode="none",
                          registro=False, calentar=False, procedencia=procedencia_para("manos"))
        with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):   # representación pedida distinta
            LSCTranslator(modelo=ModeloFalso(252), extractor=ExtractorFalso(), tts=None, tts_mode="none",
                          registro=False, calentar=False, representacion="manos_postura")

    def test_regla_del_modelo_por_defecto_y_sobrescritura_avisada(self):
        _, reg, tr = self._prototipo(secuencia_sintetica(), procedencia_para("manos", "descartar"), n_pasos=30)
        self.assertEqual((tr.mano_ausente, tr.mano_ausente_modelo), ("descartar", "descartar"))
        with redirect_stdout(io.StringIO()) as salida:
            tr = LSCTranslator(modelo=ModeloFalso(252), extractor=ExtractorFalso(), tts=None, tts_mode="none",
                               registro=False, calentar=False, procedencia=procedencia_para("manos", "mascara"),
                               mano_ausente="ceros")
        self.assertEqual((tr.mano_ausente, tr.mano_ausente_modelo), ("ceros", "mascara"))
        self.assertIn("difiere", salida.getvalue())
        with redirect_stdout(io.StringIO()):
            tr = LSCTranslator(modelo=ModeloFalso(450), extractor=ExtractorFalso(), tts=None, tts_mode="none",
                               registro=False, calentar=False,
                               procedencia=procedencia_para("manos_postura", "no_aplica", "ninguno"))
        self.assertEqual(tr.mano_ausente, "descartar")
        self.assertEqual(tr.procedencia_ruta, "inyectada")
        with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):   # regla declarada inválida
            LSCTranslator(modelo=ModeloFalso(252), extractor=ExtractorFalso(), tts=None, tts_mode="none",
                          registro=False, calentar=False, procedencia=procedencia_para("manos", "xyz"))

    def test_modelo_por_defecto_inexistente_falla_claro(self):
        import src.translator as trmod
        with mock.patch.object(trmod, "MODELO_POR_DEFECTO", ROOT_DIR / "models" / "no_existe.keras"):
            with self.assertRaises(FileNotFoundError):
                LSCTranslator(extractor=ExtractorFalso(), tts=None, tts_mode="none", registro=False, calentar=False)
            with self.assertRaises(FileNotFoundError):   # realtime.py pasa la ruta por defecto explícitamente
                LSCTranslator(model_path=str(ROOT_DIR / "models" / "no_existe.keras"), extractor=ExtractorFalso(),
                              tts=None, tts_mode="none", registro=False, calentar=False)

    def test_procedencia_se_lee_junto_al_modelo_real(self):
        from src.representacion import leer_procedencia
        ruta = ROOT_DIR / "models" / "canonico_holistic_mascara" / "modelo_canonico.keras"
        if not ruta.exists():
            self.skipTest("modelo canónico manos_postura no disponible")
        proc = leer_procedencia(ruta)
        self.assertIsNotNone(proc)
        self.assertEqual((proc["representacion"], proc["n_rasgos_entrada"], proc["mano_ausente"]),
                         ("manos_postura", 450, "mascara"))



# ---------------------------------------------------------------------------
# Errores de la voz en el JSONL (pendiente de N-34, 15-sep-2026)
# ---------------------------------------------------------------------------

# Claves y orden del registro de ensayo y del turno ANTES de añadir los contadores de la voz
# (salida de src/translator.py sin cambios del 15-sep-2026 con construir() y TTSFalso).
CLAVES_ENSAYO_ANTES = [
    "tipo", "sesion", "id_ensayo", "fecha_hora", "modo", "modelo", "n_rasgos_modelo", "usa_velocidades",
    "pasos_modelo", "extractor", "representacion", "procedencia_modelo", "mano_ausente", "mano_ausente_modelo",
    "tts_modo", "tts_sintesis_antes_de_t5", "tts_ocupado_en_t5", "parametros", "t0", "c", "e", "t1", "t2",
    "t3", "t4", "t5", "N", "cierre", "frames_con_deteccion", "frames_sin_manos", "frames_sin_pose",
    "frames_con_dos_manos", "proporcion_sin_manos", "etapas_ns", "W_ns", "L_total_ns", "L_resp_ns",
    "costo_extractor_ns", "tasa_efectiva_fps", "estado", "motivo", "id_clase", "clase", "confianza", "top3",
    "enunciado", "glosas_turno", "contadores",
]
CLAVES_TURNO_ANTES = [
    "tipo", "sesion", "id_turno", "fecha_hora", "glosas", "n_glosas", "enunciado", "t_tecla", "t_conversion",
    "t_sintesis", "conversion_ns", "sintesis_ns", "tts_modo", "tts_sintesis_antes_de_orden",
]
CLAVES_VOZ = ["tts_errores_reproduccion", "tts_ultimo_error_reproduccion"]


class TTSFalsoConErrores(TTSFalso):
    """Doble con la interfaz de errores de src/tts.py: estado_reproduccion() y los atributos."""
    mode = "falso"

    def __init__(self):
        super().__init__()
        self.errores_reproduccion = 0
        self.ultimo_error_reproduccion = None
        self.locuciones_descartadas_al_cerrar = 0

    def fallar(self, aviso="CalledProcessError: Command '['afplay', 'x.aiff']' returned non-zero exit status 1."):
        """Simula que una locución en segundo plano falló (lo que hace _registrar_error_reproduccion)."""
        self.errores_reproduccion += 1
        self.ultimo_error_reproduccion = aviso

    def estado_reproduccion(self):
        return {"errores_reproduccion": self.errores_reproduccion,
                "ultimo_error_reproduccion": self.ultimo_error_reproduccion,
                "locuciones_descartadas_al_cerrar": self.locuciones_descartadas_al_cerrar,
                "cerrado": self.limpiado}


class TTSFalsoSoloAtributos(TTSFalso):
    """Doble con los contadores como atributos pero sin estado_reproduccion()."""
    mode = "falso"

    def __init__(self):
        super().__init__()
        self.errores_reproduccion = 3
        self.ultimo_error_reproduccion = "FileNotFoundError: el audio sintetizado ya no existe"
        self.locuciones_descartadas_al_cerrar = 0


def leer_jsonl(ruta) -> list[dict]:
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class TestErroresDeVozEnElRegistro(unittest.TestCase):
    """Un fallo de reproducción en segundo plano no debe pasar inadvertido en una sesión con cámara:
    cada registro del JSONL lleva los errores ACUMULADOS hasta ese instante y close() avisa por consola.
    Diseño: sin TTS o con un motor que no expone los contadores, los dos campos van presentes con null
    («no informado», que no es «cero errores»)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_proto_")

    def test_doble_con_errores_los_lleva_al_jsonl(self):
        tts = TTSFalsoConErrores()
        tr = construir(self.tmp, tts=tts)
        r1 = correr_ventana(tr)
        self.assertEqual((r1["tts_errores_reproduccion"], r1["tts_ultimo_error_reproduccion"]), (0, None))
        tts.fallar()                                   # la locución del ensayo 1 falló en su hilo
        r2 = correr_ventana(tr)
        tts.fallar("RuntimeError: " + "x" * 500)       # texto largo: se recorta
        tr.modelo.confianza = 0.2
        r3 = correr_ventana(tr)                        # también un rechazo lleva el acumulado
        self.assertEqual(r3["estado"], "rechazada_confianza")
        lineas = leer_jsonl(tr.registro_path)
        self.assertEqual([l["tts_errores_reproduccion"] for l in lineas], [0, 1, 2])
        self.assertIn("afplay", lineas[1]["tts_ultimo_error_reproduccion"])
        self.assertEqual(len(lineas[2]["tts_ultimo_error_reproduccion"]), 200)
        self.assertTrue(lineas[2]["tts_ultimo_error_reproduccion"].startswith("RuntimeError: xxx"))
        self.assertEqual([r1["tts_errores_reproduccion"], r2["tts_errores_reproduccion"]], [0, 1])

    def test_modo_oracion_el_turno_tambien_lo_lleva(self):
        tts = TTSFalsoConErrores()
        tr = construir(self.tmp, tts=tts, modo_oracion=True)
        correr_ventana(tr)
        tts.fallar()
        turno = tr.fin_de_turno()
        self.assertEqual(turno["tts_errores_reproduccion"], 1)
        self.assertEqual(leer_jsonl(tr.registro_path)[-1]["tts_errores_reproduccion"], 1)

    def test_doble_solo_con_atributos_se_lee_igual(self):
        reg = correr_ventana(construir(self.tmp, tts=TTSFalsoSoloAtributos()))
        self.assertEqual(reg["tts_errores_reproduccion"], 3)
        self.assertIn("FileNotFoundError", reg["tts_ultimo_error_reproduccion"])

    def test_doble_sin_atributos_no_falla_y_deja_null(self):
        for tts in (TTSFalso(), TTSFalsoPorArchivo(), TTSFalsoOffline(), None):
            tr = construir(self.tmp, tts=tts)
            reg = correr_ventana(tr)
            self.assertEqual(reg["estado"], "aceptada")
            for clave in CLAVES_VOZ:
                self.assertIn(clave, reg)
                self.assertIsNone(reg[clave], (type(tts).__name__, clave))
            salida = io.StringIO()
            with redirect_stdout(salida):
                tr.close()
            self.assertNotIn("[AVISO TTS]", salida.getvalue())

    def test_interfaz_rota_o_valores_raros_no_tumban_el_registro(self):
        class Rompe(TTSFalso):
            def estado_reproduccion(self):
                raise RuntimeError("roto")

        class Raro(TTSFalso):
            errores_reproduccion = mock.MagicMock()     # ni entero ni serializable
            ultimo_error_reproduccion = ValueError("excepción en vez de texto")

        reg = correr_ventana(construir(self.tmp, tts=Rompe()))
        self.assertEqual((reg["tts_errores_reproduccion"], reg["tts_ultimo_error_reproduccion"]), (None, None))
        reg = correr_ventana(construir(self.tmp, tts=Raro()))
        self.assertIsNone(reg["tts_errores_reproduccion"])
        self.assertEqual(reg["tts_ultimo_error_reproduccion"], "excepción en vez de texto")
        json.dumps(reg)

    def test_campos_existentes_no_cambian(self):
        # Mismas claves y mismo orden que antes, salvo las dos nuevas tras tts_ocupado_en_t5.
        reg_sin = correr_ventana(construir(self.tmp, tts=TTSFalso()))
        tts = TTSFalsoConErrores()
        tts.fallar()
        reg_con = correr_ventana(construir(self.tmp, tts=tts))
        for reg in (reg_sin, reg_con):
            self.assertEqual([k for k in reg if k not in CLAVES_VOZ], CLAVES_ENSAYO_ANTES)
            i = list(reg).index("tts_ocupado_en_t5")
            self.assertEqual(list(reg)[i + 1:i + 3], CLAVES_VOZ)
        # Y mismos valores con el mismo reloj falso: leer el estado de la voz no altera t0..t5 ni nada más.
        distintos = {k for k in CLAVES_ENSAYO_ANTES if reg_sin[k] != reg_con[k]}
        self.assertEqual(distintos - {"sesion", "fecha_hora"}, set())
        self.assertEqual([reg_con[k] for k in ("t0", "t1", "t2", "t3", "t4", "t5")],
                         sorted(reg_con[k] for k in ("t0", "t1", "t2", "t3", "t4", "t5")))
        # Turno
        tr = construir(self.tmp, tts=TTSFalso(), modo_oracion=True)
        correr_ventana(tr)
        turno = tr.fin_de_turno()
        self.assertEqual([k for k in turno if k not in CLAVES_VOZ], CLAVES_TURNO_ANTES)
        self.assertEqual(list(turno)[-2:], CLAVES_VOZ)

    def test_close_imprime_resumen_solo_si_hubo_errores(self):
        tts = TTSFalsoConErrores()
        tr = construir(self.tmp, tts=tts)
        correr_ventana(tr)
        tts.fallar(); tts.fallar()
        tts.locuciones_descartadas_al_cerrar = 1
        salida = io.StringIO()
        with redirect_stdout(salida):
            tr.close()
        texto = salida.getvalue()
        self.assertEqual(texto.count("[AVISO TTS]"), 1, texto)
        self.assertIn("2 locución(es) no sonaron", texto)
        self.assertIn("1 descartada(s) al cerrar", texto)
        self.assertIn("afplay", texto)
        self.assertTrue(tts.limpiado)
        self.assertEqual(tr.estado_voz_al_cerrar["errores_reproduccion"], 2)
        # Sin errores ni descartes: silencio
        tr2 = construir(self.tmp, tts=TTSFalsoConErrores())
        correr_ventana(tr2)
        salida = io.StringIO()
        with redirect_stdout(salida):
            tr2.close()
        self.assertEqual(salida.getvalue(), "")
        self.assertEqual(tr2.estado_voz_al_cerrar["errores_reproduccion"], 0)

    def test_con_ttsmodule_real_y_reproductor_que_falla(self):
        """src/tts.py real en modo gtts con gTTS y subprocess.run sustituidos: no hay red ni audio. La
        reproducción del ensayo 1 falla en su hilo; el registro del ensayo 2 lo lleva."""
        import subprocess
        from types import SimpleNamespace
        from src import tts as ttsmod

        class GTTSFalso:
            def __init__(self, text, lang, tld):
                self.text = text

            def save(self, ruta):
                with open(ruta, "wb") as f:
                    f.write(self.text.encode("utf-8"))

        import threading
        ordenes = []
        permiso = threading.Semaphore(0)   # cada reproducción falla solo cuando la prueba lo permite

        def run_falso(orden, **kw):
            ordenes.append(orden[0])
            permiso.acquire(timeout=2.0)
            raise subprocess.CalledProcessError(1, orden)

        salida = io.StringIO()
        with mock.patch.dict(sys.modules, {"gtts": SimpleNamespace(gTTS=GTTSFalso)}), \
                mock.patch.object(ttsmod, "_detectar_reproductor", return_value="afplay"), \
                mock.patch.object(ttsmod.shutil, "which", return_value=None), \
                mock.patch.object(ttsmod.subprocess, "run", side_effect=run_falso), \
                redirect_stdout(salida):
            voz = ttsmod.TTSModule(mode="gtts", verbose=False)
            tr = construir(self.tmp, tts=voz, tts_espera_max_seg=2.0)

            def esperar_errores(k):
                limite = time.monotonic() + 3.0
                while voz.estado_reproduccion()["errores_reproduccion"] < k and time.monotonic() < limite:
                    time.sleep(0.01)

            r1 = correr_ventana(tr)                    # su reproducción queda detenida hasta release()
            permiso.release()
            esperar_errores(1)
            r2 = correr_ventana(tr)
            # Se libera y espera DENTRO de los parches a que el hilo del ensayo 2 termine: ningún hilo llega
            # a subprocess.run real después de salir del bloque.
            permiso.release()
            esperar_errores(2)
            tr.close()
        self.assertEqual(ordenes, ["afplay", "afplay"])
        self.assertIs(r1["tts_sintesis_antes_de_t5"], True)
        self.assertEqual(r1["tts_errores_reproduccion"], 0)
        self.assertEqual(r2["tts_errores_reproduccion"], 1)
        self.assertEqual(tr.estado_voz_al_cerrar["errores_reproduccion"], 2)
        self.assertIn("CalledProcessError", r2["tts_ultimo_error_reproduccion"])
        self.assertEqual([l["tts_errores_reproduccion"] for l in leer_jsonl(tr.registro_path)][:1], [0])
        self.assertIn("Voz de la sesión", salida.getvalue())


class TraductorDoble:
    """Lo mínimo que lee src/realtime.py para imprimir el bloque final de la sesión."""

    def __init__(self, voz="omitir", registro_activo=True):
        self.contadores = {"aceptada": 2, "rechazada_confianza": 1,
                           "rechazada_manos": 0, "rechazada_incompleta": 0}
        self.n_ensayos = 3
        self.n_turnos = 1
        self.registro_activo = registro_activo
        self.registro_path = Path("/tmp/reg/latencia_2026-09-15.jsonl")
        if voz != "omitir":                     # "omitir" = traductor sin el atributo (getattr defensivo)
            self.estado_voz_al_cerrar = voz


BLOQUE_SIN_VOZ = [
    "",
    "Contadores de la sesion:",
    "  aceptada               2",
    "  rechazada_confianza    1",
    "  rechazada_manos        0",
    "  rechazada_incompleta   0",
    "  ensayos 3 | turnos 1",
    "Registro: /tmp/reg/latencia_2026-09-15.jsonl",
    'Resumen: python src/latencia_resumen.py "/tmp/reg/latencia_2026-09-15.jsonl"',
]


class TestSalidaDeConsolaDeRealtime(unittest.TestCase):
    """src/realtime.py sin cámara: el bloque «Contadores de la sesion» y el aviso por consola cuando
    sube el contador de errores de voz (N-37). No se abre la cámara ni se reproduce audio: se usan un
    traductor doble y, en la última prueba, el LSCTranslator real con dobles."""

    @classmethod
    def setUpClass(cls):
        from src import realtime              # importa cv2; ninguna prueba abre la cámara
        cls.rt = realtime

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_proto_")

    def _imprimir(self, tr) -> list[str]:
        salida = io.StringIO()
        with redirect_stdout(salida):
            self.rt.imprimir_contadores_sesion(tr)
        return salida.getvalue().split("\n")[:-1]

    def test_sin_estado_de_voz_el_bloque_no_cambia(self):
        """getattr defensivo: un traductor que no deja estado_voz_al_cerrar imprime lo de siempre."""
        self.assertEqual(self._imprimir(TraductorDoble()), BLOQUE_SIN_VOZ)

    def test_muestra_errores_y_locuciones_descartadas(self):
        voz = {"errores_reproduccion": 2, "ultimo_error_reproduccion": "CalledProcessError: afplay",
               "locuciones_descartadas_al_cerrar": 1}
        lineas = self._imprimir(TraductorDoble(voz))
        self.assertEqual(lineas[:7], BLOQUE_SIN_VOZ[:7])          # nada de lo anterior cambia de orden
        self.assertEqual(lineas[7:10], ["  errores de voz         2",
                                        "  locuciones descartadas 1",
                                        "  ultimo error de voz: CalledProcessError: afplay"])
        self.assertEqual(lineas[10:], BLOQUE_SIN_VOZ[7:])
        self.assertEqual(len(lineas), len(BLOQUE_SIN_VOZ) + 3)

    def test_sin_errores_tambien_se_reporta_y_sin_mensaje_no_hay_linea(self):
        voz = {"errores_reproduccion": 0, "ultimo_error_reproduccion": None,
               "locuciones_descartadas_al_cerrar": 0}
        lineas = self._imprimir(TraductorDoble(voz))
        self.assertEqual(lineas[7:9], ["  errores de voz         0", "  locuciones descartadas 0"])
        self.assertEqual(len(lineas), len(BLOQUE_SIN_VOZ) + 2)

    def test_null_se_dice_sin_dato_y_no_cero(self):
        """None = el motor no informa; decir «0» sería afirmar que no hubo errores."""
        voz = {"errores_reproduccion": None, "ultimo_error_reproduccion": None,
               "locuciones_descartadas_al_cerrar": None}
        lineas = self._imprimir(TraductorDoble(voz))
        self.assertEqual(lineas[7:9], ["  errores de voz         sin dato",
                                       "  locuciones descartadas sin dato"])

    def test_estado_de_voz_roto_o_registro_desactivado(self):
        for voz in (None, "no soy un dict", 7):
            self.assertEqual(self._imprimir(TraductorDoble(voz)), BLOQUE_SIN_VOZ, voz)
        lineas = self._imprimir(TraductorDoble(registro_activo=False))
        self.assertEqual(lineas, BLOQUE_SIN_VOZ[:-2])

    def test_aviso_solo_cuando_el_contador_sube(self):
        avisos = self.rt.AvisoErroresVoz()
        secuencia = [
            ({"tts_errores_reproduccion": 0}, False),                       # sesión sin errores aún
            ({"tts_errores_reproduccion": 0}, False),
            ({"tts_errores_reproduccion": 1, "tts_ultimo_error_reproduccion": "afplay 1"}, True),
            ({"tts_errores_reproduccion": 1, "tts_ultimo_error_reproduccion": "afplay 1"}, False),
            ({"tts_errores_reproduccion": None}, False),                    # el motor no informa
            ({"tts_errores_reproduccion": 3, "tts_ultimo_error_reproduccion": "afplay 3"}, True),
            ({"tts_errores_reproduccion": 3}, False),
        ]
        obtenidas = []
        for reg, sube in secuencia:
            salida = io.StringIO()
            with redirect_stdout(salida):
                self.rt._avisar_voz(avisos, reg)
            texto = salida.getvalue()
            self.assertEqual(bool(texto), sube, reg)
            if texto:
                obtenidas.append(texto.rstrip("\n"))
                self.assertEqual(texto.count("[AVISO TTS]"), 1, texto)      # una vez por subida
        self.assertEqual(obtenidas, [
            "  [AVISO TTS] 1 locucion(es) no sonaron por error de reproduccion (total 1 en la sesion): afplay 1",
            "  [AVISO TTS] 2 locucion(es) no sonaron por error de reproduccion (total 3 en la sesion): afplay 3",
        ])
        self.assertEqual(avisos.ultimo, 3)

    def test_valores_que_no_son_contador_no_avisan(self):
        avisos = self.rt.AvisoErroresVoz()
        for reg in (None, "texto", {}, {"tts_errores_reproduccion": True},
                    {"tts_errores_reproduccion": "2"}, {"tts_errores_reproduccion": 2.5},
                    {"tts_errores_reproduccion": -1}):
            self.assertIsNone(avisos.revisar(reg), reg)
        self.assertEqual(avisos.ultimo, 0)
        # Un mensaje ausente no impide el aviso.
        self.assertIn("total 1 en la sesion", avisos.revisar({"tts_errores_reproduccion": 1}))

    def test_con_el_traductor_real_y_dobles(self):
        """Sin cámara: se corren ventanas con el LSCTranslator real y sus registros pasan por el aviso,
        como hace el bucle de realtime.py. El fallo de la locución del ensayo 1 aparece en el ensayo 2."""
        tts = TTSFalsoConErrores()
        tr = construir(self.tmp, tts=tts)
        avisos = self.rt.AvisoErroresVoz()
        salida = io.StringIO()
        with redirect_stdout(salida):
            self.rt._avisar_voz(avisos, correr_ventana(tr))       # ensayo 1: todavía sin errores
            self.assertEqual(salida.getvalue(), "")
            tts.fallar()                                          # su locución no sonó
            self.rt._avisar_voz(avisos, correr_ventana(tr))       # ensayo 2: el registro lo delata
            tr.modo_oracion = True
            correr_ventana(tr)
            tts.fallar("RuntimeError: el reproductor murio")
            self.rt._avisar_voz(avisos, tr.fin_de_turno())        # el turno también lleva el contador
            tts.locuciones_descartadas_al_cerrar = 2
            tr.close()
            self.rt.imprimir_contadores_sesion(tr)
        texto = salida.getvalue()
        self.assertEqual(texto.count("[AVISO TTS] 1 locucion(es)"), 2, texto)
        self.assertIn("(total 1 en la sesion): CalledProcessError", texto)
        self.assertIn("(total 2 en la sesion): RuntimeError: el reproductor murio", texto)
        lineas = texto.split("\n")
        self.assertIn("  errores de voz         2", lineas)
        self.assertIn("  locuciones descartadas 2", lineas)
        self.assertIn("  ultimo error de voz: RuntimeError: el reproductor murio", lineas)
        self.assertEqual(tr.estado_voz_al_cerrar["errores_reproduccion"], 2)



# ---------------------------------------------------------------------------
# Conversor de glosas por reglas (actividad 3.a, docs/CONVERSOR_3a_ESPECIFICACION.md §6 y §7.2).
# Pruebas añadidas el 8-nov-2026; no cambian ninguna de las anteriores.
# ---------------------------------------------------------------------------

class TTSFalsoPrecalentable(TTSFalsoOffline):
    """Motor con cache: guarda las frases que el prototipo le pide precalentar."""

    def __init__(self):
        super().__init__()
        self.precalentadas: list[str] = []

    def precalentar(self, frases):
        self.precalentadas = list(frases)
        return True


class TestConversorGlosasEnElPrototipo(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_proto_")

    def test_por_defecto_sigue_el_heredado_sin_clave_conversion(self):
        tr = construir(self.tmp, modo_oracion=False)
        self.assertIs(type(tr.grammar_mapper), GrammarMapper)
        reg = correr_ventana(tr)
        self.assertEqual(reg["enunciado"], GrammarMapper().translate(["comer"]))
        self.assertNotIn("conversion", reg)
        tr.modo_oracion = True
        correr_ventana(tr)
        turno = tr.fin_de_turno()
        self.assertNotIn("conversion", turno)
        self.assertTrue(all("conversion" not in l for l in leer_jsonl(tr.registro_path)))

    def test_construir_conversor(self):
        self.assertEqual(CONVERSORES, ("heredado", "reglas", "inventario"))
        self.assertIsNone(construir_conversor("heredado"))
        self.assertIsNone(construir_conversor())
        reglas = construir_conversor("reglas")
        self.assertIsInstance(reglas, ConversorGlosas)
        self.assertFalse(reglas.estricto)
        self.assertTrue(construir_conversor("inventario").estricto)
        self.assertEqual(construir_conversor("reglas", genero_primera="f").genero_primera, "f")
        with self.assertRaises(ValueError):
            construir_conversor("heredado", genero_primera="f")
        with self.assertRaises(ValueError):
            construir_conversor("otro")

    def test_reglas_en_modo_aislado_guarda_la_traza(self):
        tr = construir(self.tmp, grammar_mapper=ConversorGlosas())
        reg = correr_ventana(tr)
        self.assertEqual(reg["estado"], "aceptada")
        self.assertEqual(reg["enunciado"], "Comer.")
        esperado = ConversorGlosas().explicar(["comer"])
        del esperado["frase"]
        self.assertEqual(reg["conversion"], esperado)
        self.assertEqual(tr.tts.hablado, ["Comer."])
        self.assertIsNotNone(reg["etapas_ns"]["conversion"])
        linea = leer_jsonl(tr.registro_path)[-1]           # serializable y con la clave nueva
        self.assertEqual(linea["conversion"]["via"], "lexico")
        # Un ensayo rechazado no convierte: no lleva traza
        tr.modelo.confianza = 0.2
        self.assertNotIn("conversion", correr_ventana(tr))

    def test_reglas_en_modo_oracion_guarda_la_traza_del_turno(self):
        tr = construir(self.tmp, modo_oracion=True, grammar_mapper=ConversorGlosas())
        r1 = correr_ventana(tr)
        correr_ventana(tr)
        self.assertNotIn("conversion", r1)                  # la ventana no convierte en modo oración
        turno = tr.fin_de_turno()
        self.assertEqual(turno["enunciado"], "Comer.")
        self.assertEqual(turno["conversion"]["repeticiones"], 1)
        self.assertNotIn("frase", turno["conversion"])
        self.assertEqual(leer_jsonl(tr.registro_path)[-1]["conversion"]["glosas"], ["comer"])

    def test_inventario_manda_lo_no_previsto_a_respaldo(self):
        tr = construir(self.tmp, modo_oracion=True, grammar_mapper=construir_conversor("inventario"))
        tr.glosas_turno = ["¿quién?", "bienvenido"]
        turno = tr.fin_de_turno()
        self.assertEqual(turno["enunciado"], "¿quién?, bienvenido")
        self.assertEqual(turno["conversion"]["motivo"], "fuera_de_inventario")
        self.assertTrue(turno["conversion"]["parametros"]["estricto"])
        tr.glosas_turno = ["hermano", "trabajar"]
        self.assertEqual(tr.fin_de_turno()["enunciado"], "El hermano trabaja.")

    def test_doble_magicmock_sigue_usando_translate(self):
        doble = mock.MagicMock()
        doble.translate.return_value = "texto del doble"
        tr = construir(self.tmp, grammar_mapper=doble, registro=False)
        reg = correr_ventana(tr)
        self.assertEqual(reg["enunciado"], "texto del doble")
        self.assertNotIn("conversion", reg)
        doble.translate.assert_called_with(["comer"])
        doble.explicar.assert_not_called()

    def test_precalentado_con_el_conversor_nuevo(self):
        tts = TTSFalsoPrecalentable()
        construir(self.tmp, tts=tts, grammar_mapper=ConversorGlosas())
        conv = ConversorGlosas()
        self.assertEqual(tts.precalentadas, [conv.translate([g]) for g in SIGN_VOCABULARY.values()])

    def test_opciones_de_realtime(self):
        from src import realtime              # importa cv2; no se abre la cámara
        parser = realtime.crear_parser()
        args = parser.parse_args([])
        self.assertEqual((args.conversor, args.genero_primera, args.genero_segunda), ("heredado", None, None))
        self.assertIsNone(realtime.conversor_desde_args(args, parser))
        args = parser.parse_args(["--conversor", "reglas", "--genero-primera", "f"])
        conv = realtime.conversor_desde_args(args, parser)
        self.assertIsInstance(conv, ConversorGlosas)
        self.assertEqual((conv.genero_primera, conv.genero_segunda, conv.estricto), ("f", "m", False))
        args = parser.parse_args(["--conversor", "inventario"])
        self.assertTrue(realtime.conversor_desde_args(args, parser).estricto)
        args = parser.parse_args(["--genero-segunda", "f"])          # heredado + género → error
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            realtime.conversor_desde_args(args, parser)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--conversor", "otro"])


if __name__ == "__main__":
    unittest.main()
