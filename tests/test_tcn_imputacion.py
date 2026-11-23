"""
Pruebas sin cámara de la opción EXPERIMENTAL «imputacion» del prototipo y del modelo TCN + imputación + 252
(N-63; src/train_final_tcn.py, src/umbral_tcn_imputacion.py, docs/MODELO_TCN_IMPUTACION.md).

  · Equivalencia numérica: la entrada que arma LSCTranslator._preprocesar con «imputacion» es la del
    entrenamiento (processed «mascara» -> mano_ausente_variantes.regla_imputacion -> add_velocity) para
    secuencias sintéticas con huecos (interiores, al inicio, al final, sin postura, mano ausente en toda la
    ventana) y para un video real del corpus («hola», 0030_0000_0003).
  · Reglas viejas: ceros, descartar y mascara (y el modelo sin procedencia) dan EXACTAMENTE lo mismo que la
    versión de src/translator.py anterior a este cambio (tests/datos/preprocesar_reglas_previas_2026-10-03.json:
    forma y sha256 de cada entrada, generado con `git show HEAD:src/translator.py` el 3-oct-2026, commit 1ba6f93), y el prototipo por defecto
    sigue cargando el canónico con «mascara» y umbral 0,2.
  · Rechazos: rechazada_manos y rechazada_incompleta siguen valiendo con «imputacion».
  · El modelo nuevo carga con su procedencia y su umbral, elige la imputación solo y, con los puntos y con
    los fotogramas reales del video «hola» (HolisticLandmarker real, sin cámara, nada se escribe a disco
    salvo el JSONL en una carpeta temporal), predice «hola» y registra «imputacion».

Ejecutar:
    python -m unittest tests.test_tcn_imputacion -v
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.translator import (  # noqa: E402
    IDX_MANO_DER, IDX_MANO_IZQ, MANO_AUSENTE_OPCIONES, MODELO_POR_DEFECTO, N_FEATURES, Extraccion, LSCTranslator,
)
from src import mano_ausente_variantes as mav  # noqa: E402
from src.train_final_100 import add_velocity  # noqa: E402
# Dobles de prueba existentes (no se importan clases TestCase, para no duplicar pruebas).
from tests.test_prototipo import (  # noqa: E402
    ExtractorFalso, ExtractorSecuencia, ModeloFalso, TTSFalso, archivos_prohibidos, camino_del_corpus,
    correr_ventana, procedencia_para, secuencia_sintetica,
)

MODELO_TCN = ROOT_DIR / "models" / "tcn_imputacion" / "modelo.keras"
PROCESSED = ROOT_DIR / "data" / "processed_holistic_mascara_rostro"
EXTRAIDOS = ROOT_DIR / "data" / "extracted" / "LSC50_holistic"
VIDEO_HOLA = "0030_0000_0003"
VIDEO_HOLA_AVI = ROOT_DIR / "data" / "raw" / "LSC50" / "VIDEOS" / "COLOR_BODY" / f"{VIDEO_HOLA}.avi"
REFERENCIA_REGLAS_PREVIAS = ROOT_DIR / "tests" / "datos" / "preprocesar_reglas_previas_2026-10-03.json"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def entrada_del_prototipo(fotogramas, procedencia, n_pasos, clase=LSCTranslator, mano_ausente=None,
                          n_rasgos=None, **kw):
    """X (1, n_pasos, F) que el prototipo entrega al modelo para la ventana `fotogramas` (dobles inyectados)."""
    n = n_rasgos if n_rasgos is not None else procedencia["n_rasgos_entrada"]
    modelo = ModeloFalso(n, pasos=n_pasos)
    params = dict(modelo=modelo, extractor=ExtractorSecuencia(fotogramas), tts=None, tts_mode="none",
                  registro=False, ventana_frames=len(fotogramas), min_frames=1, calentar=False,
                  mano_ausente=mano_ausente)
    if procedencia is not None:
        params["procedencia"] = dict(procedencia, seq_length=n_pasos)
    params.update(kw)
    with redirect_stdout(io.StringIO()):
        tr = clase(**params)
        reg = correr_ventana(tr, n=len(fotogramas))
    tr.close()
    return (modelo.entradas[0] if modelo.entradas else None), reg, tr


def secuencia_con_huecos():
    """45 fotogramas: mano derecha ausente al inicio (0-2), en un hueco interior (20-21) y al final (44);
    mano izquierda ausente en 7 y en 30-33; un fotograma sin postura (25)."""
    return secuencia_sintetica(T=45, sin_der=(0, 1, 2, 20, 21, 44), sin_izq=(7, 30, 31, 32, 33), sin_rostro=(3,),
                               semilla=5, sin_pose=(25,))


def secuencia_sin_mano_izquierda():
    """40 fotogramas con la mano izquierda ausente en TODA la ventana y la derecha con dos huecos."""
    return secuencia_sintetica(T=40, sin_der=(5, 6, 39), sin_izq=tuple(range(40)), sin_rostro=(), semilla=7)


def entrada_del_entrenamiento(fotogramas, n_pasos=30):
    """Camino del entrenamiento: preprocess_lsc50 («mascara», representación «manos») -> regla_imputacion
    (mano_ausente_variantes, la de train_final_tcn.construir_X_seq) -> add_velocity."""
    processed = camino_del_corpus(fotogramas, "manos", "mascara")
    return add_velocity(mav.regla_imputacion(processed, n_pasos)[np.newaxis].astype(np.float32))


def fotogramas_de_puntos_extraidos(nombre):
    """Los puntos del corpus re-extraído (data/extracted/LSC50_holistic, 225 por fotograma) como fotogramas del
    doble ExtractorSecuencia; pose y número de manos se deducen de los bloques en ceros, como el extractor."""
    from src import preprocess_lsc50 as pp
    _, dirs = pp.resolver_directorios(EXTRAIDOS)
    data = pp.load_single_sample(nombre, dirs, "ninguno")
    fot = []
    for kp in data.astype(np.float32):
        pose = bool(np.any(kp[:99] != 0))
        manos = int(np.any(kp[IDX_MANO_IZQ] != 0)) + int(np.any(kp[IDX_MANO_DER] != 0))
        fot.append((kp.copy(), None, pose, manos))
    return fot


def casos_reglas_previas():
    """Casos con las reglas anteriores a «imputacion»: (nombre, fotogramas, procedencia, n_rasgos, n_pasos,
    mano_ausente indicada). Los mismos casos generaron la referencia con la versión anterior del traductor."""
    secuencias = {"s12": secuencia_sintetica(sin_pose=(10,)), "s45": secuencia_con_huecos(),
                  "s40": secuencia_sin_mano_izquierda()}
    casos = []
    for nombre_s, fot in secuencias.items():
        for rep in ("manos", "manos_postura", "manos_postura_rostro"):
            for modo in ("ceros", "descartar", "mascara"):
                proc = procedencia_para(rep, modo)
                casos.append((f"{nombre_s}_{rep}_{modo}", fot, proc, proc["n_rasgos_entrada"], 30, None))
        for ancho in (252, 450, 225):      # sin procedencia: representación deducida del ancho, «mascara»
            casos.append((f"{nombre_s}_sin_procedencia_{ancho}", fot, None, ancho, 30, None))
        proc = procedencia_para("manos", "mascara")   # regla indicada distinta de la del modelo
        casos.append((f"{nombre_s}_manos_mascara_indicada_ceros", fot, proc, 252, 30, "ceros"))
    return casos


def huella(X) -> dict:
    """Forma y sha256 de los bytes float32 de una entrada: igualdad de huella = igualdad bit a bit."""
    import hashlib
    X = np.ascontiguousarray(X, dtype=np.float32)
    return {"forma": list(X.shape), "sha256": hashlib.sha256(X.tobytes()).hexdigest()}


def entradas_reglas_previas(clase=LSCTranslator) -> dict:
    salida = {}
    for nombre, fot, proc, n, pasos, indicada in casos_reglas_previas():
        X, reg, _ = entrada_del_prototipo(fot, proc, pasos, clase=clase, mano_ausente=indicada, n_rasgos=n)
        salida[nombre] = X
    return salida


# ---------------------------------------------------------------------------
# Equivalencia numérica con el entrenamiento
# ---------------------------------------------------------------------------

class TestEquivalenciaImputacion(unittest.TestCase):

    def test_sintetica_con_huecos_igual_al_entrenamiento(self):
        for nombre, fot in (("huecos", secuencia_con_huecos()), ("doce", secuencia_sintetica(sin_pose=(10,))),
                            ("sin_izquierda", secuencia_sin_mano_izquierda())):
            with self.subTest(secuencia=nombre):
                esperado = entrada_del_entrenamiento(fot)
                X, reg, tr = entrada_del_prototipo(fot, procedencia_para("manos", "imputacion"), 30)
                self.assertEqual(X.shape, (1, 30, 252))
                self.assertEqual(X.dtype, np.float32)
                np.testing.assert_allclose(X, esperado, rtol=0, atol=1e-6)
                self.assertEqual((reg["mano_ausente"], reg["mano_ausente_modelo"]), ("imputacion", "imputacion"))
                self.assertEqual(tr.mano_ausente, "imputacion")

    def test_la_imputacion_cambia_la_entrada_respecto_de_mascara(self):
        fot = secuencia_con_huecos()
        X_imp, _, _ = entrada_del_prototipo(fot, procedencia_para("manos", "imputacion"), 30)
        X_mas, _, _ = entrada_del_prototipo(fot, procedencia_para("manos", "mascara"), 30)
        self.assertFalse(np.array_equal(X_imp, X_mas))
        # Con imputación ningún paso queda con un bloque de mano en ceros (las dos manos aparecen en la ventana).
        pos = X_imp[0, :, :126]
        self.assertFalse(np.any(np.all(pos[:, :63] == 0, axis=1)))
        self.assertFalse(np.any(np.all(pos[:, 63:] == 0, axis=1)))

    def test_mano_ausente_en_toda_la_ventana_queda_en_cero(self):
        X, _, _ = entrada_del_prototipo(secuencia_sin_mano_izquierda(), procedencia_para("manos", "imputacion"), 30)
        self.assertTrue(np.all(X[0, :, 0:63] == 0.0))              # posiciones de la mano izquierda
        self.assertTrue(np.all(X[0, :, 126:189] == 0.0))           # y sus velocidades
        self.assertTrue(np.all(np.any(X[0, :, 63:126] != 0.0, axis=1)))   # la derecha, imputada en sus huecos

    @unittest.skipUnless((PROCESSED / "X_padded.npy").exists() and (EXTRAIDOS / "BODY_LANDMARKS").is_dir(),
                         "processed canónico o corpus re-extraído no disponibles")
    def test_video_real_del_corpus_igual_al_entrenamiento(self):
        """«hola» 0030_0000_0003: los puntos re-extraídos por el prototipo = la fila del processed canónico
        (data/processed_holistic_mascara_rostro) con la regla de train_final_tcn.construir_X_seq."""
        from src.train_loso import cargar_datos
        X_padded, y, lengths, nombres, _ = cargar_datos(PROCESSED, "manos")
        i = nombres.index(VIDEO_HOLA)
        T = int(lengths[i])
        esperado = add_velocity(mav.regla_imputacion(X_padded[i, :T], 30)[np.newaxis])
        fot = fotogramas_de_puntos_extraidos(VIDEO_HOLA)
        self.assertEqual(len(fot), T)
        X, reg, _ = entrada_del_prototipo(fot, procedencia_para("manos", "imputacion"), 30)
        np.testing.assert_allclose(X, esperado, rtol=0, atol=1e-6)
        # Y es exactamente la fila de X_seq con que entrenó src/train_final_tcn.py.
        X_seq = mav.remuestrear_todas_con_regla(X_padded[i:i + 1], lengths[i:i + 1], "imputacion", 30)
        np.testing.assert_allclose(X, add_velocity(X_seq), rtol=0, atol=1e-6)
        self.assertEqual(reg["mano_ausente"], "imputacion")


# ---------------------------------------------------------------------------
# Las reglas viejas no cambian
# ---------------------------------------------------------------------------

class TestReglasPreviasSinCambios(unittest.TestCase):

    def test_opciones_previas_conservadas(self):
        self.assertEqual(MANO_AUSENTE_OPCIONES[:3], ("ceros", "descartar", "mascara"))
        self.assertEqual(MANO_AUSENTE_OPCIONES, ("ceros", "descartar", "mascara", "imputacion"))
        self.assertEqual(MODELO_POR_DEFECTO,
                         ROOT_DIR / "models" / "canonico_holistic_mascara_manos" / "modelo_canonico.keras")

    @unittest.skipUnless(REFERENCIA_REGLAS_PREVIAS.exists(), "falta la referencia de las reglas previas")
    def test_identicas_a_la_version_anterior_del_traductor(self):
        ref = json.loads(REFERENCIA_REGLAS_PREVIAS.read_text(encoding="utf-8"))["casos"]
        actual = entradas_reglas_previas()
        self.assertEqual(sorted(actual), sorted(ref))
        for nombre, X in actual.items():
            with self.subTest(caso=nombre):
                self.assertEqual(huella(X), ref[nombre])          # bit a bit, no solo a tolerancia

    def test_imputacion_solo_con_la_representacion_manos(self):
        for rep in ("manos_postura", "manos_postura_rostro"):
            with self.subTest(rep=rep), self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                entrada_del_prototipo(secuencia_sintetica(), procedencia_para(rep, "imputacion"), 30)
        with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):   # sin procedencia, ancho 450
            LSCTranslator(modelo=ModeloFalso(450), extractor=ExtractorFalso(), tts=None, tts_mode="none",
                          registro=False, calentar=False, mano_ausente="imputacion")

    def test_imputacion_indicada_sobre_un_modelo_mascara_se_avisa(self):
        with redirect_stdout(io.StringIO()) as salida:
            tr = LSCTranslator(modelo=ModeloFalso(252), extractor=ExtractorFalso(), tts=None, tts_mode="none",
                               registro=False, calentar=False, procedencia=procedencia_para("manos", "mascara"),
                               mano_ausente="imputacion")
        self.assertEqual((tr.mano_ausente, tr.mano_ausente_modelo), ("imputacion", "mascara"))
        self.assertIn("difiere", salida.getvalue())

    @unittest.skipUnless(MODELO_POR_DEFECTO.exists(), "modelo canónico no disponible")
    def test_prototipo_por_defecto_sigue_siendo_el_canonico(self):
        with redirect_stdout(io.StringIO()):
            tr = LSCTranslator(extractor=ExtractorFalso(), tts=None, tts_mode="none", registro=False,
                               calentar=False)
        self.assertEqual(Path(tr.model_path), MODELO_POR_DEFECTO)
        self.assertEqual((tr.mano_ausente, tr.mano_ausente_modelo, tr.representacion), ("mascara", "mascara", "manos"))
        self.assertEqual(tr.umbral, 0.2)
        self.assertTrue(tr.umbral_origen.endswith("modelo_canonico.umbral.json"))


# ---------------------------------------------------------------------------
# Rechazos con «imputacion»
# ---------------------------------------------------------------------------

class TestRechazosConImputacion(unittest.TestCase):

    def _tr(self, patron, tmp, **kw):
        params = dict(modelo=ModeloFalso(252), extractor=ExtractorFalso(patron), tts=TTSFalso(),
                      registro_dir=Path(tmp) / "registros", ventana_frames=20, min_frames=15, calentar=False,
                      procedencia=procedencia_para("manos", "imputacion"), umbral=0.5)
        params.update(kw)
        with redirect_stdout(io.StringIO()):
            return LSCTranslator(**params)

    def test_manos_no_detectadas_no_infiere(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = self._tr(lambda i: (True, 0 if i % 3 else 2), tmp)     # 2/3 de la ventana sin manos
            reg = correr_ventana(tr)
            self.assertEqual(reg["estado"], "rechazada_manos")
            self.assertEqual(tr.modelo.entradas, [])
            self.assertEqual(reg["mano_ausente"], "imputacion")
            linea = json.loads(next((Path(tmp) / "registros").glob("*.jsonl")).read_text().splitlines()[-1])
            self.assertEqual((linea["mano_ausente"], linea["mano_ausente_modelo"]), ("imputacion", "imputacion"))

    def test_menos_fotogramas_que_el_minimo_es_incompleta(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = self._tr(None, tmp)
            tr.abrir_ventana()
            for i in range(5):
                tr.procesar_fotograma(np.zeros((8, 8, 3), np.uint8))
            reg = tr.abortar_ventana()
            self.assertEqual(reg["estado"], "rechazada_incompleta")
            self.assertEqual(tr.modelo.entradas, [])

    def test_ventana_valida_infiere_y_registra(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = self._tr(lambda i: (True, 1 if i in (3, 4, 9) else 2), tmp)
            reg = correr_ventana(tr)
            self.assertEqual(reg["estado"], "aceptada")
            self.assertEqual(tr.modelo.entradas[0].shape, (1, 30, 252))
            self.assertEqual(reg["mano_ausente"], "imputacion")


# ---------------------------------------------------------------------------
# Modelo nuevo: procedencia, umbral y prueba sin cámara con «hola»
# ---------------------------------------------------------------------------

@unittest.skipUnless(MODELO_TCN.exists(), "models/tcn_imputacion/modelo.keras no disponible")
class TestModeloTCNImputacion(unittest.TestCase):

    def _traductor(self, extractor, tmp, **kw):
        """Como realtime.py: solo la ruta del modelo (sin modelo inyectado), así que la procedencia y el umbral se
        leen de los archivos junto al .keras. Con un modelo inyectado el traductor no lee la procedencia."""
        with redirect_stdout(io.StringIO()):
            return LSCTranslator(model_path=str(MODELO_TCN), extractor=extractor, tts=TTSFalso(),
                                 registro_dir=Path(tmp) / "registros", **kw)

    def test_carga_con_su_procedencia_y_su_umbral(self):
        from src.representacion import leer_procedencia, leer_umbral
        import hashlib
        proc = leer_procedencia(MODELO_TCN)
        umb = leer_umbral(MODELO_TCN)
        self.assertEqual((proc["representacion"], proc["n_rasgos_entrada"], proc["seq_length"], proc["mano_ausente"]),
                         ("manos", 252, 30, "imputacion"))
        self.assertEqual(proc["arquitectura"]["config_T"]["dilataciones"], [1, 2, 4, 8])
        self.assertEqual(umb["sha256_modelo"], hashlib.sha256(MODELO_TCN.read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            # Sin modelo inyectado: el camino real de realtime.py (carga del .keras, procedencia y umbral).
            with redirect_stdout(io.StringIO()) as salida:
                tr = LSCTranslator(model_path=str(MODELO_TCN), extractor=ExtractorFalso(), tts=None,
                                   tts_mode="none", registro_dir=Path(tmp) / "registros", calentar=True)
            self.assertNotIn("AVISO", salida.getvalue())
            self.assertEqual((tr.mano_ausente, tr.mano_ausente_modelo), ("imputacion", "imputacion"))
            self.assertEqual((tr.representacion, tr.n_rasgos, tr.seq_len, tr.usa_velocidades), ("manos", 252, 30, True))
            self.assertEqual(tr.umbral, float(umb["umbral"]))
            self.assertTrue(tr.umbral_origen.endswith("modelo.umbral.json"))
            self.assertTrue(tr.procedencia_ruta.endswith("modelo.procedencia.json"))
            tr.close()

    @unittest.skipUnless((EXTRAIDOS / "BODY_LANDMARKS").is_dir(), "corpus re-extraído no disponible")
    def test_sin_camara_puntos_de_hola(self):
        fot = fotogramas_de_puntos_extraidos(VIDEO_HOLA)
        with tempfile.TemporaryDirectory() as tmp:
            tr = self._traductor(ExtractorSecuencia(fot), tmp, ventana_frames=len(fot))
            reg = correr_ventana(tr, n=len(fot))
            self.assertEqual(reg["estado"], "aceptada", reg["motivo"])
            self.assertEqual(reg["clase"], "hola")
            self.assertEqual(reg["mano_ausente"], "imputacion")
            linea = json.loads(next((Path(tmp) / "registros").glob("*.jsonl")).read_text().splitlines()[-1])
            self.assertEqual((linea["mano_ausente"], linea["clase"]), ("imputacion", "hola"))
            self.assertTrue(linea["modelo"].endswith("tcn_imputacion/modelo.keras"))
            tr.close()

    @unittest.skipUnless(VIDEO_HOLA_AVI.exists() and (ROOT_DIR / "models" / "mediapipe" / "holistic_landmarker.task").exists(),
                         "video del corpus o modelo de MediaPipe no disponibles")
    def test_sin_camara_fotogramas_reales_de_hola(self):
        """Fotogramas del .avi (solo lectura, en memoria) -> HolisticLandmarker REAL -> modelo nuevo."""
        import cv2
        from src.translator import HOLISTIC_TASK, LandmarkExtractor
        with redirect_stdout(io.StringIO()):
            ext = LandmarkExtractor({HOLISTIC_TASK: str(ROOT_DIR / "models" / "mediapipe" / HOLISTIC_TASK)})
        cap = cv2.VideoCapture(str(VIDEO_HOLA_AVI))
        fotogramas = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            fotogramas.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        self.assertGreater(len(fotogramas), 30)
        with tempfile.TemporaryDirectory() as tmp:
            tr = self._traductor(ext, tmp, ventana_frames=len(fotogramas))
            tr.abrir_ventana()
            reg = None
            for f in fotogramas:
                reg = tr.procesar_fotograma(f).registro or reg
            self.assertIsNotNone(reg)
            self.assertEqual(reg["estado"], "aceptada", reg["motivo"])
            self.assertEqual(reg["clase"], "hola")
            self.assertEqual(reg["mano_ausente"], "imputacion")
            self.assertEqual(archivos_prohibidos(tmp), [])
            tr.close()


if __name__ == "__main__":
    unittest.main()
