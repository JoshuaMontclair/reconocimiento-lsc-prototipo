"""
Pruebas de src/tasa_extractor_local.py (cabo de N-21: tasa del extractor dentro y fuera de
Google Drive) SIN mediapipe real, SIN videos reales y SIN tocar data/.

Todo lo que necesita el módulo para medir se sustituye por dobles: el HolisticLandmarker por un
objeto que no hace nada, extraer_video por una función que devuelve registros por fotograma
inventados pero con la forma exacta que produce src/reextract_lsc50.py, y las llamadas a `ps`
por una lista fija de procesos. No se escribe nada fuera de un directorio temporal.

Cubren:
  · selección determinista de la muestra: misma semilla -> misma lista; equilibrio por señante y
    por seña; una repetición por par (seña, señante); semilla distinta -> lista distinta;
    señas incompletas descartadas; n = 0 y n mayor que el corpus;
  · descripción de la muestra (conteos por seña, señante y repetición);
  · estadísticas: resumen por etapa (media, DE, mediana, p90, n), fotogramas por segundo,
    comparación pareada por video y prueba de los signos exacta;
  · lectura del tiempo de CPU de `ps` y materialización de archivos (st_blocks);
  · formato del JSON: claves obligatorias, las tres condiciones, la muestra declarada con su
    semilla, serializable a JSON, y el informe .md con su tabla;
  · que el módulo NO reimplementa el extractor: usa las OPCIONES_HOLISTIC y el modelo de D-05.

Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_tasa_extractor -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import reextract_lsc50 as rx        # noqa: E402
from src import tasa_extractor_local as te   # noqa: E402


# ---------------------------------------------------------------------------
# Dobles
# ---------------------------------------------------------------------------

def nombres_corpus(n_senas=50, n_senantes=5, n_repes=4):
    """Nombres con la forma del corpus LSC50: <seña>_<señante>_<repetición>."""
    return [f"{s:04d}_{v:04d}_{r:04d}"
            for s in range(n_senas) for v in range(n_senantes) for r in range(n_repes)]


def registros_falsos(n_fotogramas: int, holistic_ms: float, decodificar_ms: float = 0.7):
    """Registros por fotograma con las MISMAS columnas que produce reextract_lsc50.extraer_video."""
    return [{
        "fotograma": i, "ts_ms": int(round(i * 1000 / 24)),
        "pose_detectada": 1, "mano_izq_detectada": 1, "mano_der_detectada": 1,
        "n_manos": 2, "rostro_detectado": 1,
        "decodificar_ms": decodificar_ms,
        "holistic_ms": holistic_ms,
        "total_ms": round(holistic_ms + decodificar_ms + 0.05, 3),
    } for i in range(n_fotogramas)]


class LandmarkerFalso:
    """Sustituye al HolisticLandmarker: solo cuenta que se creó y se cerró."""
    creados = 0
    cerrados = 0

    def __init__(self):
        LandmarkerFalso.creados += 1

    def close(self):
        LandmarkerFalso.cerrados += 1


def extraer_video_falso(ms_por_muestra: dict, fotogramas: int = 8):
    """Devuelve una función con la firma de rx.extraer_video que no abre ningún video."""
    def _f(video_path, holistic, con_rostro=True):
        stem = Path(video_path).stem
        ms = ms_por_muestra[stem]
        return {"registros": registros_falsos(fotogramas, ms),
                "entrada": {"fps": 24.0, "fotogramas_decodificados": fotogramas},
                "body": [], "lh": [], "rh": [], "face": []}
    return _f


# ---------------------------------------------------------------------------
# Muestra determinista
# ---------------------------------------------------------------------------

class TestSeleccionMuestra(unittest.TestCase):

    def test_determinista_con_la_misma_semilla(self):
        c = nombres_corpus()
        a = te.seleccionar_muestra(c, n=60, semilla=2026)
        b = te.seleccionar_muestra(c, n=60, semilla=2026)
        self.assertEqual(a, b)
        # el orden de entrada no debe importar
        revuelto = list(reversed(c))
        self.assertEqual(te.seleccionar_muestra(revuelto, n=60, semilla=2026), a)

    def test_otra_semilla_da_otra_muestra(self):
        c = nombres_corpus()
        a = te.seleccionar_muestra(c, n=60, semilla=2026)
        b = te.seleccionar_muestra(c, n=60, semilla=7)
        self.assertNotEqual(a, b)

    def test_tamano_y_equilibrio(self):
        muestra = te.seleccionar_muestra(nombres_corpus(), n=60, semilla=2026)
        self.assertEqual(len(muestra), 60)
        self.assertEqual(len(set(muestra)), 60, "no debe repetir videos")
        comp = te.describir_muestra(muestra)
        self.assertEqual(sorted(comp["videos_por_senante"].values()), [12] * 5)
        self.assertEqual(sorted(comp["videos_por_sena"].values()), [5] * 12)

    def test_una_repeticion_por_pareja_sena_senante(self):
        muestra = te.seleccionar_muestra(nombres_corpus(), n=60, semilla=2026)
        parejas = [te.partes_muestra(s)[:2] for s in muestra]
        self.assertEqual(len(parejas), len(set(parejas)))

    def test_devuelve_nombres_del_corpus_y_ordenados(self):
        c = nombres_corpus()
        muestra = te.seleccionar_muestra(c, n=20, semilla=2026)
        self.assertTrue(set(muestra) <= set(c))
        self.assertEqual(muestra, sorted(muestra))

    def test_descarta_senas_sin_todos_los_senantes(self):
        c = [n for n in nombres_corpus(n_senas=3) if not (n.startswith("0001_") and n.split("_")[1] == "0004")]
        muestra = te.seleccionar_muestra(c, n=10, semilla=2026)
        self.assertTrue(all(not s.startswith("0001_") for s in muestra),
                        "la seña 1 no tiene los 5 señantes: no debe entrar")

    def test_casos_limite(self):
        self.assertEqual(te.seleccionar_muestra(nombres_corpus(), n=0), [])
        # más de lo que hay: devuelve todo lo que se puede formar, sin romperse
        muestra = te.seleccionar_muestra(nombres_corpus(n_senas=2), n=60, semilla=2026)
        self.assertEqual(len(muestra), 10)   # 2 señas x 5 señantes
        self.assertEqual(te.seleccionar_muestra([], n=10), [])
        self.assertEqual(te.seleccionar_muestra(["basura", "otra_cosa"], n=4), [])

    def test_n_no_multiplo_de_senantes(self):
        muestra = te.seleccionar_muestra(nombres_corpus(), n=7, semilla=2026)
        self.assertEqual(len(muestra), 7)

    def test_describir_muestra(self):
        comp = te.describir_muestra(["0001_0000_0000", "0001_0001_0002", "0002_0000_0001"])
        self.assertEqual(comp["n"], 3)
        self.assertEqual(comp["senas"], [1, 2])
        self.assertEqual(comp["videos_por_sena"], {"1": 2, "2": 1})
        self.assertEqual(comp["videos_por_senante"], {"0": 2, "1": 1})
        self.assertEqual(comp["videos_por_repeticion"], {"0": 1, "1": 1, "2": 1})


# ---------------------------------------------------------------------------
# Estadísticas
# ---------------------------------------------------------------------------

class TestEstadisticas(unittest.TestCase):

    def test_resumen_por_etapa_es_el_de_reextract(self):
        """Se reutiliza resumen_ms de N-21: mismas claves y mismos valores."""
        tiempos = {"decodificar": [1.0, 1.0, 1.0, 1.0],
                   "holistic": [10.0, 12.0, 14.0, 16.0],
                   "total": [11.0, 13.0, 15.0, 17.0]}
        est = te.estadisticas_condicion(tiempos, n_fotogramas=4, segundos_pared=1.0)
        h = est["ms_por_fotograma"]["holistic"]
        self.assertEqual(h, rx.resumen_ms(tiempos["holistic"]))
        self.assertEqual(h["n"], 4)
        self.assertAlmostEqual(h["media"], 13.0)
        self.assertAlmostEqual(h["mediana"], 13.0)
        self.assertAlmostEqual(h["de"], 2.582, places=3)   # muestral, ddof=1
        # 4 fotogramas de 11, 13, 15 y 17 ms = 56 ms; el JSON guarda segundos con 2 decimales
        self.assertEqual(est["segundos_bucle"], 0.06)

    def test_fotogramas_por_segundo(self):
        self.assertEqual(te.fps(10.0), 100.0)
        self.assertEqual(te.fps(12.5), 80.0)
        self.assertIsNone(te.fps(0))
        self.assertIsNone(te.fps(None))
        tiempos = {"decodificar": [1.0] * 4, "holistic": [9.0] * 4, "total": [10.0] * 4}
        est = te.estadisticas_condicion(tiempos, n_fotogramas=4, segundos_pared=0.08)
        self.assertEqual(est["fotogramas_por_segundo"]["bucle_por_media"], 100.0)
        self.assertEqual(est["fotogramas_por_segundo"]["bucle_por_mediana"], 100.0)
        self.assertEqual(est["fotogramas_por_segundo"]["pared"], 50.0)

    def test_formato_de_p_nunca_dice_cero(self):
        self.assertEqual(te._fmt_p(0.00000031), "< 0,0001")
        self.assertEqual(te._fmt_p(0.0273), "0,0273")
        self.assertEqual(te._fmt_p(None), "-")

    def test_prueba_de_signos_exacta(self):
        # 5 positivas de 5: p = 2 * (1/32) = 0,0625
        self.assertAlmostEqual(te.prueba_signos([1, 2, 3, 4, 5])["p_valor"], 0.0625, places=6)
        r = te.prueba_signos([1, -1, 1, -1])
        self.assertEqual(r["p_valor"], 1.0)
        self.assertEqual(r["veces_positiva"], 2)
        self.assertEqual(r["veces_negativa"], 2)
        # los empates se descartan
        self.assertEqual(te.prueba_signos([0, 0, 1, 1, 1, 1, 1])["n_pares"], 5)
        self.assertEqual(te.prueba_signos([0, 0, 0])["n_pares"], 0)
        self.assertIsNone(te.prueba_signos([])["p_valor"])
        # un p diminuto no puede guardarse como 0,0: sería un valor imposible en una prueba exacta
        r = te.prueba_signos([1] * 49 + [-1] * 10)
        self.assertGreater(r["p_valor"], 0.0)
        self.assertAlmostEqual(r["p_valor"], 2.7e-07, places=9)

    def test_medias_por_bloque(self):
        por_video = [{"ms_medio": {"holistic": float(i)}} for i in range(1, 26)]
        self.assertEqual(te.medias_por_bloque(por_video, 10, "holistic"), [5.5, 15.5, 23.0])
        self.assertEqual(te.medias_por_bloque([], 10, "holistic"), [])
        self.assertEqual(te.medias_por_bloque([{"ms_medio": {"holistic": None}}], 10, "holistic"), [])

    def test_comparacion_pareada(self):
        def cond(nombre, ms):
            tiempos = {"decodificar": [0.7] * 4, "holistic": [ms] * 4, "total": [ms + 0.75] * 4}
            d = {"nombre": nombre,
                 "por_video": [{"muestra": f"000{i}_0000_0000",
                                "ms_medio": {"decodificar": 0.7, "holistic": ms + i,
                                             "total": ms + i + 0.75}} for i in range(4)]}
            d.update(te.estadisticas_condicion(tiempos, 4, 1.0))
            return d
        a, b = cond("drive_1", 15.0), cond("local", 12.0)
        c = te.comparar(a, b, "drive_1", "local", "holistic")
        self.assertEqual(c["diferencia_media_ms"], 3.0)
        self.assertEqual(c["diferencia_media_pct"], 25.0)
        self.assertEqual(c["pareada_por_video"]["n_videos"], 4)
        self.assertEqual(c["pareada_por_video"]["diferencia_media_ms"], 3.0)
        self.assertEqual(c["pareada_por_video"]["videos_con_a_mas_lento"], 4)
        self.assertEqual(c["fps_a"], 66.67)
        self.assertEqual(c["fps_b"], 83.33)


# ---------------------------------------------------------------------------
# Entorno (ps y st_blocks) con dobles
# ---------------------------------------------------------------------------

class TestEntorno(unittest.TestCase):

    def test_cputime_a_segundos(self):
        self.assertAlmostEqual(te._cputime_a_segundos("0:00.79"), 0.79)
        self.assertAlmostEqual(te._cputime_a_segundos("125:59.67"), 125 * 60 + 59.67)
        self.assertAlmostEqual(te._cputime_a_segundos("1-02:00:00"), 86400 + 7200)
        self.assertEqual(te._cputime_a_segundos(""), 0.0)
        self.assertEqual(te._cputime_a_segundos("no es un tiempo"), 0.0)

    def test_procesos_drive_filtra_y_suma(self):
        salida = ("  1628   125:59.67   231488 /Applications/Google Drive.app/Contents/MacOS/Google Drive"
                  " --single_process\n"
                  "  1775     0:36.13    20224 /Applications/Google Drive.app/Contents/PlugIns/"
                  "DFSFileProviderExtension.appex/Contents/MacOS/DFSFileProviderExtension\n"
                  "  9999     3:00.00     1000 /usr/bin/algo-que-no-es-drive\n")
        with mock.patch.object(te.subprocess, "run",
                               return_value=SimpleNamespace(stdout=salida, returncode=0)):
            procs = te.procesos_drive()
        self.assertEqual([p["pid"] for p in procs], [1628, 1775])
        self.assertAlmostEqual(te.cpu_drive_total(procs), 125 * 60 + 59.67 + 36.13, places=2)

    def test_procesos_drive_no_confunde_la_carpeta_con_la_aplicacion(self):
        """El repositorio vive en una carpeta llamada «Google Drive»: un proceso cualquiera que
        trabaje ahí NO es un proceso de sincronización y no debe contarse."""
        salida = ("  1628     0:10.00   231488 /Applications/Google Drive.app/Contents/MacOS/Google Drive\n"
                  " 32239  1000:00.00  100000 ~/.local/bin/otra-cosa --cwd "
                  "~/Google Drive/My Drive/Tesis\n")
        with mock.patch.object(te.subprocess, "run",
                               return_value=SimpleNamespace(stdout=salida, returncode=0)):
            procs = te.procesos_drive()
        self.assertEqual([p["pid"] for p in procs], [1628])
        self.assertAlmostEqual(te.cpu_drive_total(procs), 10.0)

    def test_cpu_drive_delta_suma_por_pid(self):
        antes = [{"pid": 1, "cpu_segundos_acumulados": 100.0},
                 {"pid": 2, "cpu_segundos_acumulados": 5.0}]
        despues = [{"pid": 1, "cpu_segundos_acumulados": 101.5},
                   {"pid": 3, "cpu_segundos_acumulados": 900.0}]   # 2 murió, 3 nació
        d = te.cpu_drive_delta(antes, despues)
        self.assertEqual(d["segundos"], 1.5, "un proceso que nace no puede inflar el delta")
        self.assertEqual(d["pids_comparados"], [1])
        self.assertEqual(d["pids_que_nacieron"], [3])
        self.assertEqual(d["pids_que_murieron"], [2])

    def test_procesos_drive_tolera_fallo_de_ps(self):
        with mock.patch.object(te.subprocess, "run", side_effect=OSError("no hay ps")):
            self.assertEqual(te.procesos_drive(), [])

    def test_materializacion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1, p2 = Path(tmp) / "a.avi", Path(tmp) / "b.avi"
            p1.write_bytes(b"x" * 4096)
            p2.write_bytes(b"y" * 4096)
            m = te.materializacion([p1, p2, Path(tmp) / "no_existe.avi"])
        self.assertEqual(m["n_archivos_medidos"], 3)
        self.assertEqual(m["bytes_logicos"], 8192)
        self.assertGreater(m["bytes_en_bloques"], 0)
        self.assertEqual(m["archivos_sin_bloques_asignados"], 0)

    def test_copia_local_no_borra_el_origen(self):
        with tempfile.TemporaryDirectory() as tmp:
            origen, destino = Path(tmp) / "raw", Path(tmp) / "local"
            origen.mkdir()
            for stem in ("0001_0000_0000", "0002_0000_0000"):
                (origen / f"{stem}.avi").write_bytes(b"v" * 100)
            r = te.copiar_a_local(["0001_0000_0000", "0002_0000_0000"], origen, destino)
            self.assertEqual(r["copiados"], 2)
            self.assertEqual(r["bytes_copiados"], 200)
            self.assertTrue((origen / "0001_0000_0000.avi").exists(), "el original no se toca")
            # segunda vez: no vuelve a copiar
            r2 = te.copiar_a_local(["0001_0000_0000", "0002_0000_0000"], origen, destino)
            self.assertEqual((r2["copiados"], r2["reutilizados"]), (0, 2))


# ---------------------------------------------------------------------------
# Medición con dobles y formato del JSON
# ---------------------------------------------------------------------------

class TestMedicionYJson(unittest.TestCase):

    def setUp(self):
        self.stems = ["0001_0000_0000", "0001_0001_0000", "0002_0000_0000", "0002_0001_0000"]
        LandmarkerFalso.creados = LandmarkerFalso.cerrados = 0

    def _medir(self, nombre, carpeta, ms, orden=1, almacen="disco_local_fuera_de_drive"):
        ms_por_muestra = {s: ms for s in self.stems}
        with mock.patch.object(te.rx, "crear_holistic", lambda ruta: LandmarkerFalso()), \
             mock.patch.object(te.rx, "extraer_video", extraer_video_falso(ms_por_muestra)), \
             mock.patch.object(te, "procesos_drive", lambda *a, **k: [
                 {"pid": 1628, "cpu_segundos_acumulados": 10.0, "rss_kb": 1, "comando": "Google Drive"}]):
            return te.medir_condicion(nombre, almacen, carpeta, self.stems,
                                      "modelo.task", orden, verboso=False)

    def test_medir_condicion_con_dobles(self):
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            for s in self.stems:
                (carpeta / f"{s}.avi").write_bytes(b"v" * 1000)
            cond = self._medir("local", carpeta, 12.0)
        self.assertEqual(cond["n_videos"], 4)
        self.assertEqual(cond["n_fotogramas"], 32)          # 4 videos x 8 fotogramas
        self.assertEqual(cond["bytes_leidos"], 4000)
        self.assertAlmostEqual(cond["ms_por_fotograma"]["holistic"]["media"], 12.0)
        self.assertEqual(len(cond["por_video"]), 4)
        self.assertEqual(cond["fallidos"], [])
        self.assertEqual(LandmarkerFalso.creados, 4, "un landmarker nuevo por video, como en D-05")
        self.assertEqual(LandmarkerFalso.cerrados, 4, "y se cierra siempre")
        self.assertEqual(cond["google_drive"]["cpu_segundos_consumidos_durante_la_condicion"], 0.0)

    def test_un_video_que_falla_no_tumba_la_condicion(self):
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            for s in self.stems:
                (carpeta / f"{s}.avi").write_bytes(b"v" * 1000)
            ms = {s: 12.0 for s in self.stems}

            def revienta(video_path, holistic, con_rostro=True):
                if Path(video_path).stem == self.stems[1]:
                    raise RuntimeError("no se pudo abrir")
                return extraer_video_falso(ms)(video_path, holistic, con_rostro)

            with mock.patch.object(te.rx, "crear_holistic", lambda ruta: LandmarkerFalso()), \
                 mock.patch.object(te.rx, "extraer_video", revienta), \
                 mock.patch.object(te, "procesos_drive", lambda *a, **k: []):
                cond = te.medir_condicion("local", "disco_local_fuera_de_drive", carpeta,
                                          self.stems, "modelo.task", 1, verboso=False)
        self.assertEqual(cond["n_videos"], 3)
        self.assertEqual([f["muestra"] for f in cond["fallidos"]], [self.stems[1]])
        self.assertEqual(LandmarkerFalso.cerrados, 4, "también se cierra el del video que falló")

    def test_json_e_informe_completos(self):
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp) / "videos"
            carpeta.mkdir()
            for s in self.stems:
                (carpeta / f"{s}.avi").write_bytes(b"v" * 1000)
            condiciones = {
                "drive_1": self._medir("drive_1", carpeta, 15.0, 1, "google_drive_cloudstorage"),
                "local": self._medir("local", carpeta, 12.0, 2),
                "drive_2": self._medir("drive_2", carpeta, 15.2, 3, "google_drive_cloudstorage"),
            }
            comparaciones = []
            for etapa in ("holistic", "decodificar", "total"):
                comparaciones.append(te.comparar(condiciones["drive_1"], condiciones["local"],
                                                 "drive_1", "local", etapa))
                comparaciones.append(te.comparar(condiciones["drive_2"], condiciones["local"],
                                                 "drive_2", "local", etapa))
                comparaciones.append(te.comparar(condiciones["drive_2"], condiciones["drive_1"],
                                                 "drive_2", "drive_1", etapa))
            datos = {
                "descripcion": "prueba", "pregunta": "prueba", "estado": "completado",
                "fecha_hora": "2026-09-16T00:00:00", "comando": ["x"], "git_commit": None,
                "hardware": {"cpu": "X", "nucleos_logicos": 2, "ram_gb": 1.0,
                             "sistema_operativo": "macOS"},
                "software": {"mediapipe": "0.10.35"},
                "modelo": {"sha256": "a" * 64},
                "semilla": 2026, "umbral_declarado_ms": 1.0,
                "muestra": te.describir_muestra(self.stems),
                "orden_condiciones": list(te.ORDEN_CONDICIONES),
                "totales": {"fotogramas_por_pasada": condiciones["local"]["n_fotogramas"],
                            "megabytes_muestra": 0.1},
                "condiciones": condiciones, "comparaciones": comparaciones,
            }
            datos["limitaciones"] = te.redactar_limitaciones(datos)
            datos["lectura"] = te.redactar_lectura(datos)
            destino = Path(tmp) / "salida"
            destino.mkdir()
            ruta_json = destino / "tasa.json"
            ruta_json.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
            texto_md = te.escribir_md(datos, destino / "tasa.md")
            leido = json.loads(ruta_json.read_text(encoding="utf-8"))

        # forma del JSON
        for clave in ("descripcion", "pregunta", "semilla", "muestra", "condiciones",
                      "comparaciones", "limitaciones", "lectura", "orden_condiciones", "modelo"):
            self.assertIn(clave, leido)
        self.assertEqual(sorted(leido["condiciones"]), ["drive_1", "drive_2", "local"])
        self.assertEqual(leido["muestra"]["muestras"], self.stems)
        self.assertEqual(leido["semilla"], 2026)
        for nombre, cond in leido["condiciones"].items():
            for clave in ("almacenamiento", "orden", "n_videos", "n_fotogramas", "bytes_leidos",
                          "ms_por_fotograma", "fotogramas_por_segundo", "materializacion",
                          "google_drive", "cpu_de_este_proceso", "por_video",
                          "carga_sistema_1_5_15_min_inicio",
                          "media_holistic_por_bloques_de_10_videos"):
                self.assertIn(clave, cond, f"falta {clave} en la condición {nombre}")
            for etapa in ("decodificar", "holistic", "total"):
                self.assertIn(etapa, cond["ms_por_fotograma"])
                self.assertIn("mediana", cond["ms_por_fotograma"][etapa])
                self.assertIn("de", cond["ms_por_fotograma"][etapa])
                self.assertIn("n", cond["ms_por_fotograma"][etapa])
        # las comparaciones cubren las tres etapas y las tres parejas
        self.assertEqual(len(leido["comparaciones"]), 9)
        dl = [c for c in leido["comparaciones"]
              if c["a"] == "drive_1" and c["b"] == "local" and c["etapa"] == "holistic"][0]
        self.assertAlmostEqual(dl["diferencia_media_ms"], 3.0)
        self.assertEqual(dl["pareada_por_video"]["n_videos"], 4)
        # el informe trae las tres condiciones y las secciones esperadas
        for trozo in ("drive_1", "local", "drive_2", "## Resultados", "## Lectura",
                      "## Límites de esta medición", "Comparaciones",
                      "bloques de 10 videos"):
            self.assertIn(trozo, texto_md)
        self.assertNotIn("| 0,0000 |", texto_md, "un valor p nunca se escribe como cero exacto")

    def test_limitaciones_se_redactan_desde_el_json(self):
        """--rehacer-informe debe poder rehacerlas sin volver a medir: solo dependen del JSON."""
        datos = {"muestra": {"n": 60}, "condiciones": {"local": {"n_fotogramas": 3999}}}
        lims = te.redactar_limitaciones(datos)
        self.assertTrue(any("60 videos (3999 fotogramas por pasada)" in x for x in lims))
        self.assertTrue(any("5.000 archivos" in x for x in lims),
                        "debe recordar que N-21 escribía y esta corrida solo lee")
        self.assertTrue(any("caché" in x for x in lims))

    def test_lectura_no_fuerza_la_conclusion_de_n21(self):
        """Si Drive y local salen iguales, el texto lo dice en vez de confirmar N-21."""
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(tmp)
            for s in self.stems:
                (carpeta / f"{s}.avi").write_bytes(b"v" * 10)
            condiciones = {
                "drive_1": self._medir("drive_1", carpeta, 12.0, 1, "google_drive_cloudstorage"),
                "local": self._medir("local", carpeta, 12.0, 2),
                "drive_2": self._medir("drive_2", carpeta, 12.0, 3, "google_drive_cloudstorage"),
            }
            comparaciones = [te.comparar(condiciones[a], condiciones[b], a, b, etapa)
                             for etapa in ("holistic", "decodificar", "total")
                             for a, b in (("drive_1", "local"), ("drive_2", "local"),
                                          ("drive_2", "drive_1"))]
            datos = {"condiciones": condiciones, "comparaciones": comparaciones,
                     "orden_condiciones": list(te.ORDEN_CONDICIONES), "umbral_declarado_ms": 1.0}
            lectura = te.redactar_lectura(datos)
        texto = " ".join(lectura)
        self.assertIn("no queda confirmada", texto)
        self.assertIn("tampoco queda descartada", texto)


# ---------------------------------------------------------------------------
# El extractor es el de D-05, sin reimplementar
# ---------------------------------------------------------------------------

class TestExtractorEsElDeD05(unittest.TestCase):

    def test_usa_las_opciones_y_el_modelo_de_d05(self):
        self.assertIs(te.rx, rx)
        self.assertEqual(rx.OPCIONES_HOLISTIC["min_pose_detection_confidence"], 0.5)
        self.assertEqual(rx.SHA256_HOLISTIC,
                         "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8")
        self.assertEqual(te.ETAPAS_MS, rx.ETAPAS_MS)
        # el módulo no define su propio extractor ni su propio resumen de tiempos
        for nombre in ("crear_holistic", "extraer_video", "resumen_ms", "ensure_modelo"):
            self.assertFalse(nombre in te.__dict__,
                             f"{nombre} debe importarse de reextract_lsc50, no redefinirse")

    def test_topes_de_cpu_declarados(self):
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            self.assertIsNotNone(os.environ.get(v))


if __name__ == "__main__":
    unittest.main(verbosity=2)
