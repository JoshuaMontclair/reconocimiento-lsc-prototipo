"""
Pruebas de src/reextract_lsc50.py (D-05: HolisticLandmarker) SIN mediapipe real ni videos.

El resultado de HolisticLandmarker se sustituye por objetos falsos con atributos x, y, z; el
video, por un cv2.VideoCapture falso que entrega fotogramas sintéticos. No se escribe nada
fuera de un directorio temporal. Cubren:
  · fila de ceros cuando falta un componente (postura, mano, rostro);
  · recorte del rostro a los 468 primeros puntos (sin los 10 de iris);
  · ensamblado de las filas BODY / LEFT / RIGHT / FACE a partir de un resultado falso;
  · banderas de detección y n_manos;
  · cálculo de la tasa (media / mediana / p90 / DE) y de ts_ms;
  · que OPCIONES_HOLISTIC es exactamente el diccionario fijado en D-05;
  · convención de los CSV (cabecera con índice sin nombre, CRLF) y columnas de METRICAS_FOTOGRAMA;
  · el bucle extraer_video con un landmarker falso (marcas de tiempo crecientes, una fila por fotograma)
    y la recuperación de métricas con --reanudar.
  · main() de punta a punta con dobles (revisión D-05): un landmarker por video y cerrado, video
    ilegible en `fallidos`, --reanudar recupera métricas de esta versión y REPROCESA los CSV de la
    versión anterior (tres tareas) en vez de omitirlos.

Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_reextract -v
"""

import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import reextract_lsc50 as rx  # noqa: E402

# Diccionario fijado en D-05 (otra pieza lo importa; debe ser idéntico al del script).
OPCIONES_HOLISTIC_D05 = dict(min_face_detection_confidence=0.5, min_face_suppression_threshold=0.5,
                             min_face_landmarks_confidence=0.5, min_pose_detection_confidence=0.5,
                             min_pose_suppression_threshold=0.5, min_pose_landmarks_confidence=0.5,
                             min_hand_landmarks_confidence=0.5, output_face_blendshapes=False,
                             output_segmentation_mask=False)


# ---------------------------------------------------------------------------
# Dobles de prueba
# ---------------------------------------------------------------------------

def puntos(n: int, base: float = 0.0):
    """n landmarks falsos con x, y, z distinguibles: punto i -> (base + i, base + i + 0.1, base + i + 0.2)."""
    return [SimpleNamespace(x=base + i, y=base + i + 0.1, z=base + i + 0.2) for i in range(n)]


def resultado(pose=True, izq=True, der=True, rostro=True):
    """HolisticLandmarkerResult falso: listas vacías cuando un componente no se detecta
    (así lo entrega mediapipe 0.10.35)."""
    return SimpleNamespace(
        pose_landmarks=puntos(rx.N_POSE, 100.0) if pose else [],
        left_hand_landmarks=puntos(rx.N_HAND, 200.0) if izq else [],
        right_hand_landmarks=puntos(rx.N_HAND, 300.0) if der else [],
        face_landmarks=puntos(rx.N_FACE_HOLISTIC, 400.0) if rostro else [],
        pose_world_landmarks=[], left_hand_world_landmarks=[], right_hand_world_landmarks=[],
        face_blendshapes=[], segmentation_mask=None,
    )


class HolisticFalso:
    """Registra las marcas de tiempo recibidas y devuelve resultados según `patron(i)`."""

    def __init__(self, patron=None):
        self.patron = patron or (lambda i: resultado())
        self.timestamps = []
        self.cerrado = False

    def detect_for_video(self, imagen, ts_ms):
        self.timestamps.append(ts_ms)
        return self.patron(len(self.timestamps) - 1)

    def close(self):
        self.cerrado = True


class CapturaFalsa:
    """cv2.VideoCapture falso: n fotogramas BGR sintéticos, cabecera fija."""

    def __init__(self, n=5, fps=24.0, ancho=64, alto=48):
        self.n, self.fps, self.ancho, self.alto = n, fps, ancho, alto
        self.i = 0

    def isOpened(self):
        return True

    def get(self, prop):
        import cv2
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: self.ancho,
                cv2.CAP_PROP_FRAME_HEIGHT: self.alto, cv2.CAP_PROP_FRAME_COUNT: self.n,
                cv2.CAP_PROP_FOURCC: float(int.from_bytes(b"MJPG", "little"))}.get(prop, 0.0)

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.zeros((self.alto, self.ancho, 3), dtype=np.uint8)

    def release(self):
        pass


# ---------------------------------------------------------------------------
# Constantes y opciones
# ---------------------------------------------------------------------------

class TestOpciones(unittest.TestCase):

    def test_opciones_holistic_exactas(self):
        self.assertEqual(rx.OPCIONES_HOLISTIC, OPCIONES_HOLISTIC_D05)
        self.assertEqual(list(rx.OPCIONES_HOLISTIC), list(OPCIONES_HOLISTIC_D05))   # mismo orden de claves

    def test_constantes_del_modelo(self):
        self.assertEqual(rx.MODELO_HOLISTIC, "holistic_landmarker.task")
        self.assertEqual(rx.SHA256_HOLISTIC, "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8")
        self.assertEqual(rx.MODEL_URLS[rx.MODELO_HOLISTIC],
                         "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/"
                         "float16/latest/holistic_landmarker.task")
        self.assertEqual((rx.N_POSE, rx.N_HAND, rx.N_FACE, rx.N_FACE_HOLISTIC), (33, 21, 468, 478))
        self.assertEqual(rx.OUTPUT_DIR.name, "LSC50_holistic", "la salida por defecto no es la de marzo")

    def test_columnas_metricas_sin_tres_tareas_ni_d04(self):
        self.assertEqual(rx.COLUMNAS_METRICAS, [
            "fotograma", "ts_ms", "pose_detectada", "mano_izq_detectada", "mano_der_detectada",
            "n_manos", "rostro_detectado", "decodificar_ms", "holistic_ms", "total_ms"])
        for antigua in ("pose_ms", "manos_ms", "rostro_ms", "manos_asignacion",
                        "manos_etiqueta_duplicada", "manos_contra_etiqueta", "rostro_via", "rostro_recorte_px"):
            self.assertNotIn(antigua, rx.COLUMNAS_METRICAS)

    def test_flags_eliminados_en_d05(self):
        args = rx.parsear_argumentos(["--sin-rostro", "--reanudar", "--limite", "3", "--nota", "x",
                                      "--muestras", "0000_0000_0000"])
        self.assertTrue(args.sin_rostro and args.reanudar)
        self.assertEqual((args.limite, args.nota, args.muestras), (3, "x", ["0000_0000_0000"]))
        for flag in ("--factor-recorte", "--rostro-cuadro-completo"):
            with self.assertRaises(SystemExit):
                with mock.patch("sys.stderr"):
                    rx.parsear_argumentos([flag, "1"])


# ---------------------------------------------------------------------------
# Filas y banderas
# ---------------------------------------------------------------------------

class TestFilas(unittest.TestCase):

    def test_fila_de_ceros_si_falta_el_componente(self):
        self.assertEqual(rx.fila_puntos([], rx.N_POSE), [0.0] * 99)
        self.assertEqual(rx.fila_puntos(None, rx.N_HAND), [0.0] * 63)
        self.assertEqual(rx.fila_puntos([], rx.N_FACE), [0.0] * 1404)

    def test_fila_aplanada_en_orden_x_y_z(self):
        fila = rx.fila_puntos(puntos(3), 3)
        self.assertEqual(fila, [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2])
        self.assertTrue(all(isinstance(v, float) for v in fila))

    def test_rostro_recortado_a_468_sin_iris(self):
        fila = rx.fila_puntos(puntos(rx.N_FACE_HOLISTIC), rx.N_FACE)
        self.assertEqual(len(fila), 468 * 3)
        self.assertEqual(fila[-3:], [467.0, 467.1, 467.2])      # último punto escrito: el 467
        self.assertNotIn(468.0, fila)                            # el primer punto de iris no entra

    def test_fila_corta_se_completa_con_ceros(self):
        fila = rx.fila_puntos(puntos(2), 4)
        self.assertEqual(fila, [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_ensamblado_desde_resultado_falso(self):
        filas = rx.filas_desde_resultado(resultado())
        self.assertEqual((len(filas["body"]), len(filas["lh"]), len(filas["rh"]), len(filas["face"])),
                         (99, 63, 63, 1404))
        self.assertEqual(filas["body"][:3], [100.0, 100.1, 100.2])
        self.assertEqual(filas["lh"][:3], [200.0, 200.1, 200.2])     # left_hand_landmarks -> LEFT
        self.assertEqual(filas["rh"][:3], [300.0, 300.1, 300.2])     # right_hand_landmarks -> RIGHT
        self.assertEqual(filas["face"][:3], [400.0, 400.1, 400.2])
        self.assertEqual(filas["face"][-3:], [867.0, 867.1, 867.2])  # 400 + 467
        self.assertTrue(filas["pose_detectada"] and filas["mano_izq_detectada"]
                        and filas["mano_der_detectada"] and filas["rostro_detectado"])
        self.assertEqual(filas["n_manos"], 2)

    def test_componente_ausente_da_ceros_y_bandera_en_cero(self):
        filas = rx.filas_desde_resultado(resultado(izq=False, rostro=False))
        self.assertEqual(filas["lh"], [0.0] * 63)
        self.assertEqual(filas["face"], [0.0] * 1404)
        self.assertNotEqual(filas["rh"], [0.0] * 63)
        self.assertFalse(filas["mano_izq_detectada"])
        self.assertFalse(filas["rostro_detectado"])
        self.assertTrue(filas["pose_detectada"] and filas["mano_der_detectada"])
        self.assertEqual(filas["n_manos"], 1)

    def test_banderas_y_n_manos_en_todas_las_combinaciones(self):
        for izq in (False, True):
            for der in (False, True):
                filas = rx.filas_desde_resultado(resultado(izq=izq, der=der))
                self.assertEqual(filas["mano_izq_detectada"], izq)
                self.assertEqual(filas["mano_der_detectada"], der)
                self.assertEqual(filas["n_manos"], int(izq) + int(der))
        filas = rx.filas_desde_resultado(resultado(pose=False, izq=False, der=False, rostro=False))
        self.assertEqual(filas["body"], [0.0] * 99)
        self.assertEqual(filas["n_manos"], 0)
        self.assertFalse(filas["pose_detectada"] or filas["rostro_detectado"])

    def test_sin_rostro_no_arma_la_fila_pero_conserva_la_bandera(self):
        filas = rx.filas_desde_resultado(resultado(), con_rostro=False)
        self.assertIsNone(filas["face"])
        self.assertTrue(filas["rostro_detectado"])


# ---------------------------------------------------------------------------
# Tasa y marcas de tiempo
# ---------------------------------------------------------------------------

class TestTasaYTiempos(unittest.TestCase):

    def test_resumen_ms(self):
        r = rx.resumen_ms([1.0, 2.0, 3.0, 4.0, 10.0])
        self.assertEqual(r["n"], 5)
        self.assertAlmostEqual(r["media"], 4.0)
        self.assertAlmostEqual(r["mediana"], 3.0)
        self.assertAlmostEqual(r["p90"], round(float(np.percentile([1, 2, 3, 4, 10], 90)), 3))   # 7.6
        self.assertAlmostEqual(r["p90"], 7.6)
        self.assertAlmostEqual(r["de"], round(float(np.std([1, 2, 3, 4, 10], ddof=1)), 3))
        self.assertEqual((r["min"], r["max"]), (1.0, 10.0))
        self.assertEqual(rx.resumen_ms([]), {"n": 0})
        self.assertEqual(rx.resumen_ms([5.0])["de"], 0.0)

    def test_ts_ms_redondeado_y_creciente(self):
        self.assertEqual(rx.ts_fotograma(0, 24.0), 0)
        self.assertEqual(rx.ts_fotograma(1, 24.0), 42)      # 41.67 -> 42
        self.assertEqual(rx.ts_fotograma(24, 24.0), 1000)
        self.assertEqual(rx.ts_fotograma(1, 50.0), 20)
        ts = [rx.ts_fotograma(i, 24.0) for i in range(500)]
        self.assertTrue(all(b > a for a, b in zip(ts, ts[1:])), "el modo VIDEO exige marcas estrictamente crecientes")
        self.assertTrue(all(isinstance(t, int) for t in ts))


# ---------------------------------------------------------------------------
# CSV, bucle de extracción y reanudación (todo en un directorio temporal)
# ---------------------------------------------------------------------------

class TestCsvYBucle(unittest.TestCase):

    def test_csv_con_la_convencion_de_los_publicados(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "x.csv"
            rx.escribir_csv(ruta, rx.make_headers("landmark", 2), [[0.5, 0.25, -0.125, 1.0, 0.0, 2.0]])
            crudo = ruta.read_bytes()
        lineas = crudo.split(b"\r\n")
        self.assertEqual(lineas[0], b",landmark_0_x,landmark_0_y,landmark_0_z,landmark_1_x,landmark_1_y,landmark_1_z")
        self.assertEqual(lineas[1], b"0,0.5,0.25,-0.125,1.0,0.0,2.0")
        self.assertEqual(lineas[2], b"")                       # termina en CRLF
        self.assertNotIn(b"\n", crudo.replace(b"\r\n", b""))   # ningún LF suelto
        self.assertEqual(len(rx.make_headers("landmark", rx.N_FACE)), 1404)
        self.assertEqual(rx.make_headers("landmark", rx.N_POSE)[-1], "landmark_32_z")

    def _extraer(self, patron=None, n=5, con_rostro=True):
        hol = HolisticFalso(patron)
        with mock.patch.object(rx.cv2, "VideoCapture", lambda ruta: CapturaFalsa(n=n)), \
             mock.patch.object(rx.mp, "Image", lambda **kw: kw["data"]):
            res = rx.extraer_video(Path("falso.avi"), hol, con_rostro)
        return res, hol

    def test_extraer_video_una_fila_por_fotograma_y_ts_crecientes(self):
        patron = lambda i: resultado(izq=(i % 2 == 0), der=(i != 3), rostro=(i != 4))
        res, hol = self._extraer(patron, n=5)
        self.assertEqual(hol.timestamps, [0, 42, 83, 125, 167])
        self.assertEqual([len(res[k]) for k in ("body", "lh", "rh", "face", "registros")], [5] * 5)
        self.assertEqual(res["entrada"]["fotogramas_decodificados"], 5)
        self.assertEqual((res["entrada"]["fps"], res["entrada"]["codec"]), (24.0, "MJPG"))
        reg = res["registros"]
        self.assertEqual([r["mano_izq_detectada"] for r in reg], [1, 0, 1, 0, 1])
        self.assertEqual([r["mano_der_detectada"] for r in reg], [1, 1, 1, 0, 1])
        self.assertEqual([r["n_manos"] for r in reg], [2, 1, 2, 0, 2])
        self.assertEqual([r["rostro_detectado"] for r in reg], [1, 1, 1, 1, 0])
        self.assertEqual(res["lh"][1], [0.0] * 63)
        self.assertEqual(res["face"][4], [0.0] * 1404)
        self.assertEqual([r["ts_ms"] for r in reg], hol.timestamps)
        self.assertEqual(list(reg[0]), rx.COLUMNAS_METRICAS)
        for r in reg:
            self.assertGreaterEqual(r["total_ms"], r["decodificar_ms"] + r["holistic_ms"] - 1e-3)

    def test_extraer_video_sin_rostro(self):
        res, _ = self._extraer(n=3, con_rostro=False)
        self.assertEqual(res["face"], [])
        self.assertEqual([r["rostro_detectado"] for r in res["registros"]], [1, 1, 1])

    def test_resumen_y_recuperacion_desde_metricas(self):
        patron = lambda i: resultado(izq=(i != 1), der=(i != 1), rostro=(i != 2))
        res, _ = self._extraer(patron, n=4)
        with mock.patch.object(rx, "filas_csv_publicado", lambda stem, carpeta=None: None):
            resumen = rx.resumir_video("falso", res, 0.5)
        self.assertEqual((resumen["fotogramas"], resumen["sin_mano_alguna"], resumen["sin_rostro"]), (4, 1, 1))
        self.assertEqual((resumen["mano_izq_detectada"], resumen["mano_der_detectada"]), (3, 3))
        self.assertEqual(set(resumen["ms_medio"]), set(rx.ETAPAS_MS))
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "falso.csv"
            rx.escribir_metricas(ruta, res["registros"])
            with open(ruta, newline="") as f:
                self.assertEqual(next(csv.reader(f)), rx.COLUMNAS_METRICAS)
            with mock.patch.object(rx.cv2, "VideoCapture", lambda ruta: CapturaFalsa(n=4)):
                rec = rx.recuperar_desde_metricas(Path("falso.avi"), ruta)
            self.assertEqual(rec["registros"], res["registros"])
            self.assertAlmostEqual(rec["segundos"], sum(r["total_ms"] for r in res["registros"]) / 1000.0)
            # un CSV de métricas de la versión anterior (tres tareas) no se recupera
            ruta.write_text("fotograma,ts_ms,pose_ms,manos_ms,rostro_ms,total_ms\n0,0,1,1,1,3\n")
            with mock.patch.object(rx.cv2, "VideoCapture", lambda ruta: CapturaFalsa(n=1)):
                self.assertIsNone(rx.recuperar_desde_metricas(Path("falso.avi"), ruta))


# ---------------------------------------------------------------------------
# main() de punta a punta con dobles (añadido en la revisión de D-05)
# ---------------------------------------------------------------------------

class TestPrincipalConDobles(unittest.TestCase):
    """main() con ensure_modelo, crear_holistic, cv2.VideoCapture y mp.Image sustituidos:
    ningún modelo ni video real; todo en un directorio temporal."""

    def _correr(self, d: Path, extra: list[str], fotogramas: dict, ilegibles=()):
        videos = d / "videos"
        videos.mkdir(exist_ok=True)
        for stem in fotogramas:
            (videos / f"{stem}.avi").touch()          # solo el nombre: la captura es falsa
        modelo = d / "holistic_landmarker.task"
        modelo.write_bytes(b"modelo falso")
        creados = []

        def crear(ruta):
            self.assertEqual(ruta, str(modelo))
            h = HolisticFalso()
            creados.append(h)
            return h

        class CapturaIlegible(CapturaFalsa):
            def isOpened(self):
                return False

        def captura(ruta):
            stem = Path(ruta).stem
            return CapturaIlegible() if stem in ilegibles else CapturaFalsa(n=fotogramas[stem])

        with mock.patch.object(rx, "ensure_modelo", lambda *a, **k: (str(modelo), "sha-falso")), \
             mock.patch.object(rx, "crear_holistic", crear), \
             mock.patch.object(rx.cv2, "VideoCapture", captura), \
             mock.patch.object(rx.mp, "Image", lambda **kw: kw["data"]), \
             mock.patch.object(rx, "filas_csv_publicado", lambda stem, carpeta=None: None), \
             mock.patch("sys.stdout", new_callable=io.StringIO) as salida:
            datos = rx.main(["--videos", str(videos), "--salida", str(d / "salida"), *extra])
        return datos, creados, salida.getvalue()

    def test_un_landmarker_por_video_cerrado_y_video_ilegible_en_fallidos(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            datos, creados, texto = self._correr(d, [], {"aaaa": 3, "bbbb": 2, "cccc": 1}, ilegibles={"cccc"})
            self.assertEqual(len(creados), 3, "un HolisticLandmarker nuevo por video (también el que falla)")
            self.assertTrue(all(h.cerrado for h in creados), "todos cerrados al terminar cada video")
            self.assertEqual([h.timestamps for h in creados[:2]], [[0, 42, 83], [0, 42]])
            self.assertEqual(datos["estado"], "completado")
            self.assertEqual([f["muestra"] for f in datos["fallidos"]], ["cccc"])
            self.assertIn("No se pudo abrir", datos["fallidos"][0]["error"])
            self.assertEqual([(v["muestra"], v["fotogramas"]) for v in datos["por_video"]], [("aaaa", 3), ("bbbb", 2)])
            self.assertEqual(datos["deteccion"]["fotogramas"], 5)
            self.assertEqual(datos["omitidos_por_reanudar"], [])
            for carpeta in ("BODY_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS", "FACE_LANDMARKS", "METRICAS_FOTOGRAMA"):
                self.assertTrue((d / "salida" / carpeta / "aaaa.csv").exists(), carpeta)
                self.assertFalse((d / "salida" / carpeta / "cccc.csv").exists(), carpeta)
            self.assertEqual(len((d / "salida/BODY_LANDMARKS/aaaa.csv").read_bytes().split(b"\r\n")), 3 + 2)   # cabecera + 3 filas + final
            self.assertTrue((d / "salida" / "tasa_procesamiento.json").exists())
            self.assertEqual(list(datos["modelos"]), [rx.MODELO_HOLISTIC])

    def test_reanudar_recupera_esta_version_y_reprocesa_la_anterior(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._correr(d, [], {"aaaa": 3, "bbbb": 2})
            # aaaa queda con METRICAS de la versión anterior (tres tareas); bbbb con las de esta versión
            met_a = d / "salida/METRICAS_FOTOGRAMA/aaaa.csv"
            met_a.write_text("fotograma,ts_ms,pose_ms,manos_ms,rostro_ms,total_ms\n0,0,1,1,1,3\n")
            datos, creados, texto = self._correr(d, ["--reanudar"], {"aaaa": 3, "bbbb": 2})
            self.assertEqual(datos["omitidos_por_reanudar"], ["bbbb"])
            self.assertEqual(len(creados), 1, "solo aaaa se reprocesa")
            self.assertIn("[AVISO] aaaa", texto)
            resumenes = {v["muestra"]: v for v in datos["por_video"]}
            self.assertEqual(set(resumenes), {"aaaa", "bbbb"})
            self.assertTrue(resumenes["bbbb"]["recuperado_de_metricas_previas"])
            self.assertNotIn("recuperado_de_metricas_previas", resumenes["aaaa"])
            self.assertEqual(datos["deteccion"]["fotogramas"], 5)
            self.assertEqual(datos["tasa_efectiva"]["fotogramas_esta_corrida"], 3)
            with open(met_a, newline="") as f:
                self.assertEqual(next(csv.reader(f)), rx.COLUMNAS_METRICAS, "METRICAS de aaaa ya es de esta versión")
            # tercera corrida: ya no hay nada que reprocesar
            datos, creados, _ = self._correr(d, ["--reanudar"], {"aaaa": 3, "bbbb": 2})
            self.assertEqual((datos["omitidos_por_reanudar"], len(creados)), (["aaaa", "bbbb"], 0))
            self.assertIsNone(datos["tasa_efectiva"]["fotogramas_por_segundo_pared"])


if __name__ == "__main__":
    unittest.main()
