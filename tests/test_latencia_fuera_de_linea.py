"""
Pruebas de src/latencia_fuera_de_linea.py (actividad 4.c) SIN cámara, SIN MediaPipe real, SIN modelos
reales, SIN voz y SIN tocar data/ ni registros/.

Cubren:
  · muestra estratificada: determinista, 2 por seña, equilibrada por señante, nombres inválidos;
  · puntos → Extraccion (banderas de pose y manos como las daría el extractor) y extractor pregrabado;
  · lectura de puntos con el cargador del corpus desde un árbol «plano» temporal;
  · estadísticas (mediana, p90, p99, máx.) y variación entre repeticiones;
  · espera por carga: baja a tiempo, no baja (se mide igual), tope total agotado;
  · voz silenciada: no reproduce, borra el archivo y cuenta;
  · un ensayo completo por LSCTranslator con dobles: la ventana es el clip, las etapas son las del
    registro del prototipo y post_cierre = suma de las cuatro etapas;
  · turnos: las 92 secuencias del inventario dan frase con «inventario»;
  · extractor: costo e_i − c_i con OpenCV simulado en memoria y resumen frente a 33,3 ms;
  · caracterización de un registro sintético (pausas, intervalos, N ≠ 75, campo de fuente ausente);
  · separación informe / crudo y agregado.

Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_latencia_fuera_de_linea -v
"""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import latencia_fuera_de_linea as lf  # noqa: E402
from src import translator as trmod  # noqa: E402
from src.translator import IDX_MANO_DER, IDX_MANO_IZQ, N_FEATURES, LSCTranslator  # noqa: E402


def nombres_corpus(n_senas=50, n_senantes=5, n_repes=4):
    return [f"{s:04d}_{v:04d}_{r:04d}" for s in range(n_senas) for v in range(n_senantes) for r in range(n_repes)]


def puntos_sinteticos(T=20, sin_izq=(), sin_der=(), sin_pose=(), semilla=0):
    rng = np.random.default_rng(semilla)
    kp = rng.uniform(0.3, 0.7, size=(T, N_FEATURES)).astype(np.float32)
    kp[:, 33:36] = [0.6, 0.4, 0.0]
    kp[:, 36:39] = [0.4, 0.41, 0.0]
    for t in sin_izq:
        kp[t, IDX_MANO_IZQ] = 0.0
    for t in sin_der:
        kp[t, IDX_MANO_DER] = 0.0
    for t in sin_pose:
        kp[t, :99] = 0.0
    return kp


class ModeloFalso:
    def __init__(self, id_clase=30, confianza=0.9):
        self.input_shape = (None, 30, 252)
        self.id_clase, self.confianza = id_clase, confianza
        self.llamadas = 0

    def predict(self, X, verbose=0):
        self.llamadas += 1
        assert X.shape == (1, 30, 252)
        p = np.full(50, (1 - self.confianza) / 49, dtype=np.float32)
        p[self.id_clase] = self.confianza
        return p[np.newaxis]


class TTSFalsoOffline:
    mode = "offline"
    soporta_preparar = True

    def __init__(self):
        self.eventos = []
        self.is_prepared = True

    def prepare(self, texto):
        self.eventos.append(("prepare", texto))

    def esperar_preparado(self, tope):
        return True

    def play_prepared_async(self):
        self.eventos.append(("orden", None))

    def speak_async(self, texto):
        raise AssertionError("no debe llegar aquí")


def traductor_con_dobles(conversor="inventario", confianza=0.9):
    procedencia = {"representacion": "manos", "n_rasgos_entrada": 252, "seq_length": 30, "velocidades": True,
                   "mano_ausente": "imputacion", "rostro_del_processed": "ninguno"}
    ext = lf.ExtractorPregrabado()
    with redirect_stdout(io.StringIO()):
        tr = LSCTranslator(modelo=ModeloFalso(confianza=confianza), procedencia=procedencia, extractor=ext,
                           tts=TTSFalsoOffline(), umbral=0.0, registro=False, calentar=False, min_frames=5,
                           grammar_mapper=trmod.construir_conversor(conversor), tts_precalentar=False)
    return tr, ext


class TestMuestra(unittest.TestCase):
    def test_determinista_y_equilibrada(self):
        a = lf.seleccionar_muestra(nombres_corpus(), 2, 2026)
        b = lf.seleccionar_muestra(list(reversed(nombres_corpus())), 2, 2026)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 100)
        d = lf.describir_muestra(a)
        self.assertEqual(set(d["por_senante"].values()), {20})
        self.assertEqual((d["senas"], d["por_sena_min"], d["por_sena_max"]), (50, 2, 2))
        # Dos señantes distintos por seña.
        for s in range(50):
            senantes = {lf.partes(x)[1] for x in a if lf.partes(x)[0] == s}
            self.assertEqual(len(senantes), 2)

    def test_semilla_cambia_repeticiones_no_senantes(self):
        a = lf.seleccionar_muestra(nombres_corpus(), 2, 2026)
        b = lf.seleccionar_muestra(nombres_corpus(), 2, 7)
        self.assertNotEqual(a, b)
        self.assertEqual([lf.partes(x)[:2] for x in a], [lf.partes(x)[:2] for x in b])

    def test_pocos_senantes_y_nombres_invalidos(self):
        nombres = ["0000_0003_0001", "0000_0003_0002", "basura.csv", "0001_0000_0000", "0001_0001_0000"]
        m = lf.seleccionar_muestra(nombres, 2, 1)
        self.assertEqual(sum(1 for x in m if x.startswith("0000_")), 1)   # un solo señante disponible
        self.assertEqual(sum(1 for x in m if x.startswith("0001_")), 2)
        self.assertNotIn("basura", " ".join(m))


class TestPuntos(unittest.TestCase):
    def test_banderas_como_el_extractor(self):
        kp = puntos_sinteticos(6, sin_izq=(1, 2), sin_der=(2,), sin_pose=(4,))
        ex = lf.extracciones_desde_puntos(kp)
        self.assertEqual([e.n_manos for e in ex], [2, 1, 0, 2, 2, 2])
        self.assertEqual([e.pose_detectada for e in ex], [True, True, True, True, False, True])
        self.assertTrue(all(e.rostro is None for e in ex))
        ex[0].keypoints[0] = 99.0
        self.assertNotEqual(kp[0, 0], 99.0)                 # copia, no vista

    def test_extractor_pregrabado(self):
        ex = lf.ExtractorPregrabado()
        e = lf.extracciones_desde_puntos(puntos_sinteticos(3))
        ex.cargar(e)
        self.assertIs(ex.extract(None), e[0])
        self.assertIs(ex.extract("ignorado"), e[1])
        ex.extract(0)
        with self.assertRaises(IndexError):
            ex.extract(0)

    def test_cargar_puntos_arbol_plano_temporal(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            kp = puntos_sinteticos(4, sin_der=(1,))
            for carpeta, cols in (("BODY_LANDMARKS", slice(0, 99)), ("LEFT_HAND_LANDMARKS", slice(99, 162)),
                                  ("RIGHT_HAND_LANDMARKS", slice(162, 225))):
                (raiz / carpeta).mkdir()
                n = (cols.stop - cols.start) // 3
                nombres = [f"landmark_{i}_{c}" for i in range(n) for c in "xyz"]
                pd.DataFrame(kp[:, cols], columns=nombres).to_csv(raiz / carpeta / "0001_0002_0003.csv")
            with redirect_stdout(io.StringIO()):
                leido = lf.cargar_puntos("0001_0002_0003", raiz)
            self.assertEqual(leido.shape, (4, 225))
            np.testing.assert_allclose(leido, kp, rtol=1e-6)
            with self.assertRaises(RuntimeError), redirect_stdout(io.StringIO()):
                lf.cargar_puntos("9999_0000_0000", raiz)


class TestEstadisticas(unittest.TestCase):
    def test_resumen(self):
        r = lf.resumen_ms(list(range(1, 101)) + [None])
        self.assertEqual(r["n"], 100)
        self.assertAlmostEqual(r["mediana"], 50.5)
        self.assertAlmostEqual(r["p90"], np.percentile(range(1, 101), 90))
        self.assertAlmostEqual(r["p99"], np.percentile(range(1, 101), 99))
        self.assertEqual((r["min"], r["max"]), (1, 100))
        self.assertIsNone(lf.resumen_ms([None]))
        self.assertEqual(lf.resumen_ms([5.0])["de"], 0.0)

    def test_variacion(self):
        v = lf.variacion([{"mediana": 10.0}, {"mediana": 12.0}, None, {"mediana": 11.0}])
        self.assertEqual(v["valores"], [10.0, 12.0, 11.0])
        self.assertAlmostEqual(v["rango"], 2.0)
        self.assertAlmostEqual(v["rango_relativo"], 2.0 / 11.0)
        self.assertIsNone(lf.variacion([None]))


class TestCarga(unittest.TestCase):
    def _reloj(self):
        t = {"v": 0.0}

        def reloj():
            return t["v"]

        def dormir(s):
            t["v"] += s
        return reloj, dormir

    def test_baja_a_tiempo(self):
        cargas = iter([(4.0, 4, 4), (3.5, 4, 4), (2.5, 3, 3)])
        reloj, dormir = self._reloj()
        p = lf.PresupuestoEspera(600)
        with mock.patch.object(lf, "procesos_top", return_value=[]):
            r = lf.esperar_carga(3.0, 120, p, intervalo_s=15, leer=lambda: next(cargas), dormir=dormir, reloj=reloj)
        self.assertTrue(r["se_alcanzo"])
        self.assertEqual(r["esperado_s"], 30.0)
        self.assertEqual(r["carga_al_empezar"]["1min"], 2.5)
        self.assertEqual(p.usado_s, 30.0)

    def test_no_baja_se_mide_igual(self):
        reloj, dormir = self._reloj()
        p = lf.PresupuestoEspera(600)
        with mock.patch.object(lf, "procesos_top", return_value=[]):
            r = lf.esperar_carga(3.0, 60, p, intervalo_s=15, leer=lambda: (5.0, 5, 5), dormir=dormir, reloj=reloj)
        self.assertFalse(r["se_alcanzo"])
        self.assertEqual(r["esperado_s"], 60.0)

    def test_presupuesto_total_agotado(self):
        reloj, dormir = self._reloj()
        p = lf.PresupuestoEspera(0)
        with mock.patch.object(lf, "procesos_top", return_value=[]):
            r = lf.esperar_carga(3.0, 60, p, leer=lambda: (5.0, 5, 5), dormir=dormir, reloj=reloj)
        self.assertEqual(r["esperado_s"], 0.0)
        self.assertFalse(r["se_alcanzo"])


class TestVozSilenciada(unittest.TestCase):
    def test_no_reproduce_y_borra(self):
        class TTS:
            def __init__(self):
                self.borrados = []

            def _borrar_audio(self, ruta):
                self.borrados.append(ruta)

            def _reproducir(self, archivo, texto):
                raise AssertionError("sonaría")

            def speak(self, texto):
                raise AssertionError("sonaría")

            def _play_audio(self, ruta):
                raise AssertionError("sonaría")
        t = TTS()
        lf.silenciar_tts(t)
        t._reproducir("/tmp/a.aiff", "hola")
        t.speak("hola")
        self.assertTrue(t._play_audio("/tmp/b.aiff"))
        self.assertEqual(t.borrados, ["/tmp/a.aiff"])
        self.assertEqual(t.locuciones_silenciadas, 2)


class TestEnsayo(unittest.TestCase):
    def test_ventana_es_el_clip_y_etapas_del_prototipo(self):
        tr, ext = traductor_con_dobles()
        clip = lf.extracciones_desde_puntos(puntos_sinteticos(23, sin_der=(3, 4)))
        reg = lf.correr_ensayo(tr, ext, clip)
        self.assertEqual(reg["N"], 23)
        self.assertEqual(reg["cierre"], "frames")
        self.assertEqual(reg["estado"], "aceptada")
        self.assertEqual(reg["clase"], "hola")
        self.assertEqual(tr.modelo.llamadas, 1)                     # UNA inferencia por ventana
        f = lf.fila_ensayo("0030_0000_0003", reg)
        suma = sum(f[k] for k in lf.ETAPAS_POST)
        self.assertAlmostEqual(f["post_cierre"], suma, places=6)
        self.assertEqual(f["post_cierre"], (reg["t5"] - reg["e"][-1]) / 1e6)
        self.assertEqual(f["L_resp_fuera_de_linea"], reg["L_resp_ns"] / 1e6)
        self.assertEqual(f["enunciado"], reg["enunciado"])
        self.assertIn(("orden", None), tr.tts.eventos)
        # Un segundo ensayo de otra longitud usa su propia ventana.
        reg2 = lf.correr_ensayo(tr, ext, lf.extracciones_desde_puntos(puntos_sinteticos(11)))
        self.assertEqual(reg2["N"], 11)

    def test_rechazada_por_manos_no_tiene_post_cierre(self):
        tr, ext = traductor_con_dobles()
        clip = lf.extracciones_desde_puntos(puntos_sinteticos(10, sin_izq=range(8), sin_der=range(8)))
        f = lf.fila_ensayo("x", lf.correr_ensayo(tr, ext, clip))
        self.assertEqual(f["estado"], "rechazada_manos")
        self.assertIsNone(f["post_cierre"])
        self.assertIsNone(f["inferencia"])
        r = lf.resumir_filas([f])
        self.assertIsNone(r["post_cierre"])

    def test_sin_registro_en_disco(self):
        tr, ext = traductor_con_dobles()
        self.assertFalse(tr.registro_activo)


class TestTurnos(unittest.TestCase):
    def test_inventario_da_frase_a_las_92_secuencias(self):
        secs = lf.secuencias_inventario()
        self.assertEqual(len(secs), 92)
        tr, _ = traductor_con_dobles("inventario")
        tts = TTSFalsoOffline()
        r = lf.medir_turnos(tr, tts, ["hola", "gracias"], secs)
        self.assertEqual(r["resumen"]["secuencia"]["con_frase"], 92)
        self.assertEqual(r["resumen"]["sola"]["con_frase"], 2)
        self.assertEqual(r["resumen"]["secuencia"]["fallos_sintesis"], 0)
        self.assertEqual(sum(1 for e in tts.eventos if e[0] == "orden"), 94)
        self.assertEqual(len(r["filas"]), 94)


class TestExtractor(unittest.TestCase):
    def test_costo_por_fotograma_en_memoria(self):
        frames = [np.zeros((1080, 1920, 3), np.uint8) for _ in range(4)]

        class Cap:
            def __init__(self, ruta):
                self.i = 0

            def isOpened(self):
                return True

            def read(self):
                if self.i >= len(frames):
                    return False, None
                self.i += 1
                return True, frames[self.i - 1]

            def release(self):
                pass

        class Ext:
            def __init__(self):
                self.formas = []

            def extract(self, rgb):
                self.formas.append(rgb.shape)
                return trmod.Extraccion(keypoints=np.zeros(225, np.float32), pose_detectada=True, n_manos=1)

        tiempos = iter(range(0, 10_000_000_000, 5_000_000))       # cada lectura avanza 5 ms
        ext = Ext()
        with mock.patch("cv2.VideoCapture", Cap):
            v = lf.medir_extractor_video(Path("x.avi"), ext, (1280, 720), reloj=lambda: next(tiempos))
        self.assertEqual(v["fotogramas"], 4)
        self.assertEqual(v["con_manos"], 4)
        self.assertEqual(v["costos_ms"], [5.0] * 4)
        self.assertEqual(ext.formas[0], (720, 1280, 3))

    def test_resumen_frente_al_presupuesto(self):
        por_video = [{"video": "a", "con_manos": 2, "costos_ms": [10.0, 40.0]},
                     {"video": "b", "con_manos": 1, "costos_ms": [12.0, 14.0]}]
        r = lf.resumen_extractor(por_video, fps_camara=30.0, ventana_seg=2.5)
        self.assertAlmostEqual(r["fraccion_sobre_presupuesto"], 0.25)
        self.assertEqual(r["fotogramas"], 4)
        self.assertAlmostEqual(r["tasa_max_mediana_fps"], 1000.0 / 13.0)
        self.assertEqual(r["ventana"]["fotogramas"], 75)
        self.assertAlmostEqual(r["ventana"]["fraccion_de_la_ventana_mediana"], np.median([75 * 25.0, 75 * 13.0]) / 2500)


class TestCaracterizacion(unittest.TestCase):
    def test_sesion_sintetica(self):
        base = {"tipo": "ensayo", "sesion": "S", "modelo": "m", "mano_ausente": "imputacion", "tts_modo": "offline",
                "parametros": {"umbral": 0.0, "fps_medida_antes": 30.0}, "cierre": "frames",
                "frames_sin_pose": 0, "costo_extractor_ns": {"mediana": 12_000_000}}
        e1 = dict(base, fecha_hora="2026-10-03T11:00:00.000", t0=0, c=[0, 33_000_000, 66_000_000], N=75,
                  estado="aceptada", clase="hola", confianza=0.9, enunciado="Hola", modo="aislado",
                  proporcion_sin_manos=0.1, frames_con_dos_manos=0, tasa_efectiva_fps=30.0)
        e2 = dict(base, fecha_hora="2026-10-03T11:00:10.000", t0=10_000_000_000, c=[0, 30_000_000], N=51,
                  estado="rechazada_manos", clase=None, confianza=None, enunciado=None, modo="oracion",
                  proporcion_sin_manos=0.6, frames_con_dos_manos=10, tasa_efectiva_fps=29.0)
        t1 = {"tipo": "turno", "sesion": "S", "fecha_hora": "2026-10-03T11:00:20.000", "glosas": ["hola"],
              "enunciado": "Hola", "sintesis_ns": 30_000_000, "tts_sintesis_desde_cache": False}
        c = lf.caracterizar_sesion([e1, e2, t1])
        self.assertEqual(c["duracion_s"], 20.0)
        self.assertEqual((c["ensayos"], c["turnos"]), (2, 1))
        self.assertEqual(c["ventanas_con_N_distinto_de_75"], 1)
        self.assertEqual(c["ventanas_sin_ninguna_con_dos_manos"], 1)
        self.assertEqual(c["pausa_entre_ventanas_s"]["mediana"], 10.0)
        self.assertEqual(c["intervalo_entre_fotogramas_ms"]["n"], 3)
        self.assertFalse(c["campo_fuente_de_video"])
        self.assertFalse(c["conversor_3a_en_registro"])
        self.assertEqual(c["glosas_aceptadas"], {"hola": 1})
        self.assertEqual(c["costo_extractor_mediana_ms"]["mediana"], 12.0)
        self.assertEqual(c["turnos_detalle"][0]["sintesis_ms"], 30.0)

    def test_registros_desde_archivo_temporal(self):
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "latencia_2026-01-01.jsonl"
            reg = {"tipo": "ensayo", "sesion": "S", "fecha_hora": "2026-01-01T10:00:00.000", "N": 75, "t0": 0,
                   "c": [0, 33_000_000], "estado": "aceptada", "clase": "hola", "parametros": {"umbral": 0.0},
                   "etapas_ns": {"inferencia": 20_000_000}}
            ruta.write_text(json.dumps(reg) + "\n", encoding="utf-8")
            c = lf.caracterizar_registros([ruta])
            self.assertIn("latencia_2026-01-01.jsonl", c)
            self.assertEqual(c["latencia_2026-01-01.jsonl"]["latencia_resumen"]["n_ensayos"], 1)
            self.assertIn("S", c["latencia_2026-01-01.jsonl"]["sesiones"])


class TestSalida(unittest.TestCase):
    def _datos(self):
        fila = {"video": "v", "N": 60, "estado": "aceptada", "clase": "hola", "confianza": 0.9, "enunciado": "Hola",
                "sintesis_desde_cache": True, "sintesis_antes_de_t5": True, "preprocesamiento": 0.5,
                "inferencia": 15.0, "conversion": 0.03, "sintesis": 0.4, "post_cierre": 15.93,
                "L_resp_fuera_de_linea": 15.93}
        reps = [{"repeticion": k, "filas": [dict(fila, inferencia=15.0 + k)],
                 "resumen": lf.resumir_filas([dict(fila, inferencia=15.0 + k)])} for k in (1, 2, 3)]
        tur = {"repeticion": 1, "filas": [{"x": 1}],
               "resumen": {t: {"n": 1, "con_frase": 1, "conversion": lf.resumen_ms([0.02]),
                               "sintesis_sin_cache": lf.resumen_ms([20.0]), "orden": lf.resumen_ms([0.1]),
                               "fallos_sintesis": 0} for t in ("sola", "secuencia")}}
        return {"inicio": "x", "post_cierre": {"final_inventario": {"repeticiones": {"cache": reps, "sin_cache": reps},
                                                                   "turnos": [tur]}}, "extractor": None}

    def test_agregado_y_variacion(self):
        agg = lf.agregar(self._datos())
        a = agg["final_inventario"]["cache"]
        self.assertEqual(a["magnitudes"]["inferencia"]["n"], 3)
        self.assertEqual(a["variacion_mediana"]["inferencia"]["valores"], [16.0, 17.0, 18.0])
        self.assertEqual(a["estados"], {"aceptada": 3})
        self.assertEqual(agg["final_inventario"]["turnos"]["secuencia"]["n"], 1)

    def test_separar_crudo(self):
        informe, crudo = lf.separar_crudo(self._datos())
        reps = informe["post_cierre"]["final_inventario"]["repeticiones"]["cache"]
        self.assertTrue(all("filas" not in r and "resumen" in r for r in reps))
        self.assertNotIn("filas", informe["post_cierre"]["final_inventario"]["turnos"][0])
        self.assertEqual(len(crudo["post_cierre"]["final_inventario"]["repeticiones"]["cache"][0]["filas"]), 1)
        self.assertEqual(crudo["post_cierre"]["final_inventario"]["turnos"][0]["filas"], [{"x": 1}])


class TestConfiguracion(unittest.TestCase):
    def test_configuraciones(self):
        self.assertEqual(lf.CONFIGURACIONES["final_inventario"]["conversor"], "inventario")
        self.assertEqual(Path(lf.CONFIGURACIONES["final_inventario"]["modelo"]).relative_to(ROOT_DIR).as_posix(),
                         "models/tcn_imputacion/modelo.keras")
        self.assertEqual(Path(lf.CONFIGURACIONES["heredada_heredado"]["modelo"]), Path(trmod.MODELO_POR_DEFECTO))
        self.assertEqual(lf.CONFIGURACIONES["heredada_heredado"]["conversor"], "heredado")
        self.assertEqual(lf.RESOLUCION_EN_VIVO, (1280, 720))
        a = lf.parsear_argumentos([])
        self.assertEqual((a.por_clase, a.repeticiones, a.semilla, a.umbral_carga), (2, 3, 2026, 3.0))


if __name__ == "__main__":
    unittest.main()
