"""
Pruebas del servicio auxiliar de voz de respaldo (src/voz_xtts_servicio.py) SIN PyTorch.

El servicio falso es el MISMO servidor con el motor «prueba» (escribe un tono; no importa torch), lanzado como
proceso aparte con el intérprete de las pruebas: así se prueba el protocolo real (JSON por línea por stdin/
stdout), el arranque, las caídas y los topes. Cubren:
  1. arranque y síntesis a WAV; peticiones seguidas y desde varios hilos; detener idempotente;
  2. una frase que falla no tumba el servicio; un proceso que muere, que no arranca, que no dice «listo» o que no
     responde a tiempo queda «caido» (y el cliente termina el proceso que lanzó); un intérprete inexistente;
  3. el canal queda limpio aunque el motor imprima en stdout; pedidos inválidos reciben error;
  4. src/tts.py con el servicio real de prueba: la frase no pregrabada usa XTTS, y si el servicio cae a mitad de
     sesión el prototipo (LSCTranslator) sigue y registra «local».
    python -m unittest tests/test_voz_xtts_servicio.py -v
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import wave
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import voz_xtts_servicio as sx  # noqa: E402
from tests.test_voz_pregrabada import DoblesVozLocal  # noqa: E402

SCRIPT = RAIZ / "src" / "voz_xtts_servicio.py"


class _Base(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lsc_xtts_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def cliente(self, *extra, **kw):
        kw.setdefault("motor", "prueba")
        kw.setdefault("tope_arranque_seg", 20.0)
        kw.setdefault("tope_sintesis_seg", 10.0)
        c = sx.ServicioXTTS(sys.executable, argumentos_extra=list(extra), **kw)
        self.addCleanup(c.detener, 1.0)
        return c

    @staticmethod
    def esperar(condicion, tope=5.0):
        limite = time.monotonic() + tope
        while not condicion() and time.monotonic() < limite:
            time.sleep(0.01)
        return condicion()


class TestProtocolo(_Base):

    def test_arranca_sintetiza_y_se_detiene(self):
        c = self.cliente()
        self.assertEqual(c.estado_actual, "sin_iniciar")
        self.assertFalse(c.esperar_listo(0.1), "sin iniciar no está disponible")
        c.iniciar()
        self.assertTrue(c.esperar_listo(20))
        self.assertTrue(c.disponible)
        self.assertEqual(c.info_listo["motor"], "prueba")
        self.assertEqual(c.info_listo["hablante"], "Gilberto Mathias")
        self.assertEqual(c.info_listo["idioma"], "es")
        salida = self.tmp / "a.wav"
        r = c.sintetizar("Hola, tío.", salida)
        self.assertTrue(r["ok"])
        self.assertEqual(r["id"], 1)
        self.assertGreaterEqual(r["ms_cliente"], 0)
        with wave.open(str(salida), "rb") as w:
            self.assertEqual((w.getnchannels(), w.getframerate(), w.getsampwidth()), (1, 24000, 2))
            self.assertEqual(w.getnframes(), 6000)
        self.assertFalse((self.tmp / "a.wav.parcial").exists())
        self.assertEqual(c.sintetizar("Otra.", self.tmp / "b.wav")["id"], 2)
        e = c.estado()
        self.assertEqual((e["estado"], e["peticiones"], e["fallos"]), ("listo", 2, 0))
        proc = c._proc
        c.detener()
        self.assertEqual(proc.wait(timeout=5), 0, "con «salir» termina limpio")
        self.assertTrue(c.caido)
        c.detener()                                   # idempotente
        with self.assertRaises(sx.ErrorServicioVoz):
            c.sintetizar("tras detener", self.tmp / "c.wav")

    def test_peticiones_desde_varios_hilos(self):
        c = self.cliente("--retraso-ms", "20")
        c.iniciar()
        self.assertTrue(c.esperar_listo(20))
        resultados, errores = [], []

        def pedir(i):
            try:
                resultados.append(c.sintetizar(f"frase {i}", self.tmp / f"{i}.wav"))
            except Exception as exc:  # noqa: BLE001
                errores.append(exc)

        hilos = [threading.Thread(target=pedir, args=(i,)) for i in range(6)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join(10)
        self.assertEqual(errores, [])
        self.assertEqual(sorted(r["id"] for r in resultados), list(range(1, 7)))
        self.assertTrue(all((self.tmp / f"{i}.wav").exists() for i in range(6)))

    def test_avisa_los_cambios_de_estado(self):
        cambios = []
        c = self.cliente(al_cambiar=lambda estado, detalle: cambios.append(estado))
        c.iniciar()
        self.assertTrue(c.esperar_listo(20))
        c.detener()
        self.assertEqual(cambios[:2], ["cargando", "listo"])


class TestFallosYCaidas(_Base):

    def test_una_frase_que_falla_no_tumba_el_servicio(self):
        c = self.cliente("--fallar-con", "MALO")
        c.iniciar()
        self.assertTrue(c.esperar_listo(20))
        with self.assertRaisesRegex(sx.ErrorServicioVoz, "fallo simulado"):
            c.sintetizar("esto es MALO", self.tmp / "x.wav")
        self.assertTrue(c.disponible)
        self.assertTrue(c.sintetizar("esto no", self.tmp / "y.wav")["ok"])
        self.assertEqual(c.estado()["fallos"], 1)

    def test_balbuceo_se_reintenta_con_otra_semilla(self):
        # XTTS a veces sigue generando sílabas tras la frase (10,9 s para 34 caracteres en MPS, docs/VOZ.md).
        c = self.cliente("--alargar-con", "LARGA", "--alargar-siempre-con", "SIEMPRE")
        c.iniciar()
        self.assertTrue(c.esperar_listo(20))
        r = c.sintetizar("frase LARGA", self.tmp / "x.wav")
        self.assertEqual((r["semilla"], r["descartadas_s"], r["duracion_s"]), (sx.SEMILLA + 1, [30.0], 0.25))
        self.assertEqual(c.sintetizar("normal", self.tmp / "n.wav")["descartadas_s"], [])
        with self.assertRaisesRegex(sx.ErrorServicioVoz, "balbuceo"):
            c.sintetizar("SIEMPRE larga", self.tmp / "y.wav")
        self.assertFalse((self.tmp / "y.wav").exists(), "no se escribe el audio sospechoso")
        self.assertTrue(c.disponible, "una frase sospechosa no tumba el servicio")

    def test_limite_de_duracion(self):
        # Por encima de lo normal medido (máx. 0,17 s/carácter con 19 caracteres) y por debajo del balbuceo.
        self.assertGreater(sx.duracion_maxima_s("x" * 19), 3.24)
        self.assertLess(sx.duracion_maxima_s("Buenas tardes, tú eres bienvenido."), 10.86)
        self.assertGreater(sx.duracion_maxima_s("Buenas tardes, tú eres bienvenido."), 3.62)
        self.assertEqual(sx.duracion_maxima_s("  hola  "), sx.duracion_maxima_s("hola"))

    def test_proceso_que_muere_a_mitad(self):
        c = self.cliente("--morir-con", "MUERE")
        c.iniciar()
        self.assertTrue(c.esperar_listo(20))
        with self.assertRaises(sx.ErrorServicioVoz):
            c.sintetizar("aquí MUERE", self.tmp / "x.wav")
        self.assertTrue(self.esperar(lambda: c.caido))
        self.assertIn("código 3", c.motivo_caida)
        t0 = time.monotonic()
        with self.assertRaises(sx.ErrorServicioVoz):
            c.sintetizar("después", self.tmp / "y.wav")
        self.assertLess(time.monotonic() - t0, 0.5, "caído, falla en el acto")

    def test_no_arranca(self):
        c = self.cliente("--no-arrancar")
        c.iniciar()
        self.assertFalse(c.esperar_listo(20))
        self.assertTrue(c.caido)
        self.assertIn("terminó", c.motivo_caida)

    def test_error_al_cargar_el_motor(self):
        # Motor xtts con un intérprete sin torch: el servidor dice {"evento": "error"} (o muere) y queda caído.
        c = self.cliente(motor="xtts")
        c.iniciar()
        self.assertFalse(c.esperar_listo(60))
        self.assertTrue(c.caido)
        self.assertTrue("error al cargar" in c.motivo_caida or "terminó" in c.motivo_caida, c.motivo_caida)

    def test_no_dice_listo_a_tiempo(self):
        c = self.cliente("--colgar-arranque", tope_arranque_seg=1.0)
        c.iniciar()
        t0 = time.monotonic()
        self.assertFalse(c.esperar_listo(10))
        self.assertLess(time.monotonic() - t0, 8)
        self.assertTrue(c.caido)
        self.assertIn("no estuvo listo", c.motivo_caida)
        self.assertTrue(self.esperar(lambda: c._proc.poll() is not None), "el cliente termina el proceso que lanzó")

    def test_no_responde_a_tiempo(self):
        c = self.cliente("--retraso-ms", "5000", tope_sintesis_seg=0.5)
        c.iniciar()
        self.assertTrue(c.esperar_listo(20))
        t0 = time.monotonic()
        with self.assertRaisesRegex(sx.ErrorServicioVoz, "no respondió"):
            c.sintetizar("lenta", self.tmp / "x.wav")
        self.assertLess(time.monotonic() - t0, 4)
        self.assertTrue(c.caido)
        self.assertTrue(self.esperar(lambda: c._proc.poll() is not None))

    def test_interprete_inexistente(self):
        c = sx.ServicioXTTS(str(self.tmp / "no-hay-python"))
        with self.assertRaises(sx.ErrorServicioVoz):
            c.iniciar()
        self.assertTrue(c.caido)
        self.assertFalse(c.esperar_listo(0.1))

    def test_opciones_invalidas(self):
        with self.assertRaises(ValueError):
            sx.ServicioXTTS(sys.executable, dispositivo="gpu")
        with self.assertRaises(ValueError):
            sx.ServicioXTTS(sys.executable, motor="otro")


class TestCanal(_Base):

    def test_el_canal_queda_limpio_aunque_el_motor_imprima(self):
        ayudante = self.tmp / "ruidoso.py"
        ayudante.write_text(textwrap.dedent(f"""
            import os, sys
            sys.path.insert(0, {str(RAIZ / 'src')!r})
            import voz_xtts_servicio as s
            original = s._MotorPrueba.sintetizar
            def ruidosa(self, texto, semilla=s.SEMILLA):
                print("RUIDO por print")
                os.write(1, b"RUIDO por el descriptor 1\\n")
                return original(self, texto, semilla)
            s._MotorPrueba.sintetizar = ruidosa
            sys.exit(s.main(sys.argv[1:]))
        """), encoding="utf-8")
        c = self.cliente(script=ayudante)
        c.iniciar()
        self.assertTrue(c.esperar_listo(20))
        self.assertTrue(c.sintetizar("hola", self.tmp / "x.wav")["ok"])
        self.assertTrue(self.esperar(lambda: any("RUIDO" in x for x in c.ultimas_lineas_stderr(10))))
        self.assertFalse(any("no JSON" in x for x in c.ultimas_lineas_stderr(10)))

    def test_pedidos_invalidos_reciben_error(self):
        proc = subprocess.Popen([sys.executable, "-I", str(SCRIPT), "--motor", "prueba"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        listo = json.loads(proc.stdout.readline())
        self.assertEqual(listo["evento"], "listo")
        pedidos = ["no es json", json.dumps([1, 2]), json.dumps({"id": 1, "texto": "hola"}),
                   json.dumps({"id": 2, "texto": "hola", "salida": "relativa.wav"}),
                   json.dumps({"id": 3, "texto": "  ", "salida": str(self.tmp / "v.wav")}),
                   json.dumps({"id": 4, "texto": "bien", "salida": str(self.tmp / "ok.wav")})]
        for p in pedidos:
            proc.stdin.write(p + "\n")
        proc.stdin.write(json.dumps({"orden": "salir"}) + "\n")
        proc.stdin.flush()
        respuestas = [json.loads(proc.stdout.readline()) for _ in pedidos]
        self.assertEqual([r["ok"] for r in respuestas], [False] * 5 + [True])
        self.assertEqual([r["id"] for r in respuestas], [None, None, 1, 2, 3, 4])
        self.assertEqual(proc.wait(timeout=5), 0)
        self.assertTrue((self.tmp / "ok.wav").exists())
        self.assertFalse((RAIZ / "relativa.wav").exists())

    def test_eof_en_stdin_termina_el_servicio(self):
        proc = subprocess.Popen([sys.executable, "-I", str(SCRIPT), "--motor", "prueba"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.assertIn(b"listo", proc.stdout.readline())
        proc.stdin.close()
        self.assertEqual(proc.wait(timeout=5), 0)

    def test_python_voz_desde_el_entorno(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {sx.VARIABLE_PYTHON_VOZ: "  /opt/py  "}):
            self.assertEqual(sx.python_voz_desde_entorno(), "/opt/py")
        with mock.patch.dict(os.environ, {sx.VARIABLE_PYTHON_VOZ: ""}):
            self.assertIsNone(sx.python_voz_desde_entorno())


class TestConElPrototipo(DoblesVozLocal, unittest.TestCase):
    """src/tts.py y LSCTranslator con el servicio de prueba real (proceso aparte) y la voz local falsa."""

    def setUp(self):
        self.poner_dobles()
        self.tmp = tempfile.mkdtemp(prefix="lsc_xtts_proto_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_xtts_y_caida_a_mitad_de_sesion_sin_romper_el_prototipo(self):
        from src.conversor_glosas import ConversorGlosas
        from tests.test_prototipo import construir, correr_ventana
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            t = self.construir_pregrabada(None, python_voz=sys.executable, xtts_motor="prueba")
            self.assertIsNotNone(t.xtts)
            self.assertTrue(t.esperar_servicio_voz(20))
            t._reproducir = lambda archivo, texto: None          # sin audio
            tr = construir(self.tmp, tts=t, grammar_mapper=ConversorGlosas(estricto=True), modo_oracion=True)
            tr.glosas_turno = ["hermano", "tío", "comer"]
            turno = tr.fin_de_turno()
            self.assertEqual(turno["tts_fuente"], "xtts")
            # El servicio cae (se mata el proceso que lanzó el módulo) y la sesión sigue con la voz local.
            t.xtts._proc.kill()
            self.assertTrue(self.esperar(lambda: t.xtts.caido))
            tr.glosas_turno = ["yo", "comer", "tío"]
            turno = tr.fin_de_turno()
            self.assertEqual(turno["tts_fuente"], "local")
            self.assertIs(turno["tts_sintesis_antes_de_orden"], True)
            reg = correr_ventana(tr)                              # un ensayo normal (glosa sola: pregrabada)
            self.assertEqual(reg["estado"], "aceptada")
            tr.glosas_turno = ["hermano", "tío", "comer"]
            self.assertEqual(tr.fin_de_turno()["tts_fuente"], "xtts", "lo ya sintetizado sigue en la cache")
            tr.close()
        self.assertIn("listo", salida.getvalue())
        self.assertIn("no está disponible", salida.getvalue())

    def test_servicio_que_no_arranca_no_bloquea_el_constructor(self):
        t0 = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            t = self.construir_pregrabada(None, python_voz=sys.executable, xtts_motor="prueba",
                                          xtts_argumentos=["--colgar-arranque"], xtts_tope_arranque_seg=30)
        self.assertLess(time.monotonic() - t0, 2.0, "el constructor no espera la carga")
        self.assertTrue(t.xtts.cargando)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(t.esperar_servicio_voz(0.3))
            t.prepare("hola, tú")
            self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual(t.ultima_fuente, "local")
        proc = t.xtts._proc
        t.cleanup()
        self.assertTrue(self.esperar(lambda: proc.poll() is not None), "cleanup termina el proceso")

    def test_interprete_inexistente_deja_la_voz_local(self):
        from src import tts as tts_mod
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            t = tts_mod.TTSModule(mode="pregrabada", verbose=False, sintetizador_persistente=False,
                                  python_voz=str(Path(self.tmp) / "no-hay-python"))
            self.addCleanup(t.cleanup)
            t.prepare("hola, tú")
            self.assertTrue(t.esperar_preparado(2.0))
        self.assertIn("no se pudo lanzar", salida.getvalue())
        self.assertEqual(t.ultima_fuente, "local")


if __name__ == "__main__":
    unittest.main()
