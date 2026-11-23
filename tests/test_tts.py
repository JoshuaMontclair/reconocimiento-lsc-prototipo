"""
Pruebas de src/tts.py SIN reproducir audio y sin depender de las voces del equipo.

Cubren los tres arreglos del 14-sep-2026 (§2.2.5 del anteproyecto):
  1. modo por defecto «offline»;
  2. selección de voz en español por `languages`, por código es-XX en el identificador o por
     nombre conocido —nunca por «"es" in id», que en macOS casaba con «synthesis» y elegía
     la voz inglesa Albert—, con preferencia es-CO > Latinoamérica > es-ES;
  3. los modos edge y gtts fallan con TTSDependenciaFaltante y mensaje claro si faltan sus
     paquetes, sin que la importación del módulo falle.
Desde el 15-sep-2026 cubren además la síntesis a archivo antes de t5 (N-34) y sus pendientes: borrado
de los audios temporales, captura de las excepciones de reproducción en los hilos y cleanup() con
locuciones en cola. Desde el 16-sep-2026, la cache de audio por texto (acierto, fallo, cambio de voz o
de ritmo, topes de entradas y de bytes, copia propia para quien reproduce, precalentado y cierre), el
sintetizador persistente de macOS con dobles de pyobjc (sin audio y sin pyobjc real) y la llegada del
campo tts_sintesis_desde_cache al registro JSONL del prototipo.
Las voces son objetos falsos; pyttsx3, `say` y `afplay` se sustituyen por dobles que registran las
órdenes: ninguna prueba reproduce audio. Ejecutar desde la raíz del repositorio:
    python -m unittest tests/test_tts.py -v
"""

from __future__ import annotations

import contextlib
import io
import os
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RAIZ, "src"))
sys.path.insert(0, _RAIZ)

import tts  # noqa: E402


def voz(id_, name, languages=None):
    """Voz falsa con la misma forma que las de pyttsx3 (id, name, languages)."""
    return SimpleNamespace(id=id_, name=name, languages=languages if languages is not None else [])


# Voces reales de macOS (nombres e identificadores tal como los lista pyttsx3/nsss el 14-sep-2026).
ALBERT = voz("com.apple.speech.synthesis.voice.Albert", "Albert", ["en_US"])
PAULINA = voz("com.apple.voice.compact.es-MX.Paulina", "Paulina", ["es_MX"])
MONICA = voz("com.apple.voice.compact.es-ES.Monica", "Mónica", ["es_ES"])
EDDY_MX = voz("com.apple.eloquence.es-MX.Eddy", "Eddy (Spanish (Mexico))", ["es_MX"])
SAMANTHA = voz("com.apple.voice.compact.en-US.Samantha", "Samantha", ["en_US"])


class MotorFalso:
    """Imita pyttsx3.Engine: entrega voces y registra setProperty/say/runAndWait."""

    driver_name = "falso"

    def __init__(self, voces):
        self._voces = voces
        self.propiedades = {}
        self.dichos = []
        self.bucles = 0
        self.proxy = SimpleNamespace(ocupado=None)
        self.proxy.setBusy = lambda valor: setattr(self.proxy, "ocupado", valor)

    def getProperty(self, nombre):
        return self._voces if nombre == "voices" else self.propiedades.get(nombre)

    def setProperty(self, nombre, valor):
        self.propiedades[nombre] = valor

    def say(self, texto):
        self.dichos.append(texto)

    def runAndWait(self):
        self.bucles += 1

    def save_to_file(self, texto, ruta):
        self.dichos.append(("archivo", texto))
        with open(ruta, "wb") as f:
            f.write(b"RIFF-falso")


def pyttsx3_falso(voces):
    motor = MotorFalso(voces)
    return SimpleNamespace(init=lambda: motor), motor


class TestModoPorDefecto(unittest.TestCase):
    def test_modo_por_defecto_es_offline(self):
        self.assertEqual(tts.MODO_POR_DEFECTO, "offline")
        self.assertEqual(tts.MODOS[0], "offline")

    def test_modo_invalido(self):
        with self.assertRaises(ValueError):
            tts.TTSModule(mode="otro")


class TestClasificacionDeVoces(unittest.TestCase):
    def test_filtro_de_marzo_casaba_con_synthesis(self):
        # Documenta el defecto: «"es" in id» es verdadero para Albert (voz inglesa).
        self.assertIn("es", ALBERT.id.lower())
        es_espanol, _, criterio = tts.clasificar_voz(ALBERT)
        self.assertFalse(es_espanol)
        self.assertEqual(criterio, "languages")

    def test_languages_macos(self):
        self.assertEqual(tts.clasificar_voz(PAULINA), (True, "es-mx", "languages"))
        self.assertEqual(tts.clasificar_voz(MONICA), (True, "es-es", "languages"))

    def test_languages_espeak_bytes(self):
        v = voz("spanish-latin-am", "Spanish (Latin America)", [b"\x05es-419"])
        self.assertEqual(tts.clasificar_voz(v), (True, "es-419", "languages"))

    def test_languages_lcid_windows(self):
        # LCID hexadecimal de SAPI5: idioma primario 0x0A = español; sub-idioma = región.
        self.assertEqual(tts.clasificar_voz(voz("x", "Helena", ["40A"])), (True, "es-es", "languages"))
        self.assertEqual(tts.clasificar_voz(voz("x", "Sabina", ["80A"])), (True, "es-mx", "languages"))
        self.assertEqual(tts.clasificar_voz(voz("x", "Bogota", ["240A"])), (True, "es-co", "languages"))
        self.assertEqual(tts.clasificar_voz(voz("x", "Zira", ["409"]))[0], False)

    def test_codigo_en_identificador(self):
        v = voz(r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-ES_HELENA_11.0", "Helena")
        self.assertEqual(tts.clasificar_voz(v), (True, "es-es", "id"))
        w = voz(r"...\Tokens\TTS_MS_EN-US_ZIRA_11.0", "Zira")
        self.assertFalse(tts.clasificar_voz(w)[0])

    def test_nombre_conocido(self):
        self.assertEqual(tts.clasificar_voz(voz("voz-7", "Mónica")), (True, None, "nombre"))
        self.assertEqual(tts.clasificar_voz(voz("voz-8", "Spanish Female")), (True, None, "nombre"))
        # Laura es eslovaca en macOS: no está en la lista y no debe pasar.
        self.assertEqual(tts.clasificar_voz(voz("voz-9", "Laura")), (False, None, "ninguno"))


class TestEleccionDeVoz(unittest.TestCase):
    def test_no_elige_albert(self):
        elegida, motivo = tts.elegir_voz_espanol([ALBERT, SAMANTHA, PAULINA])
        self.assertIs(elegida, PAULINA)
        self.assertIn("languages", motivo)

    def test_prefiere_latinoamerica_sobre_es_es_y_compact_sobre_eloquence(self):
        # Paulina (es-MX compact) gana a Mónica (es-ES) por variante y a Eddy (es-MX eloquence) por calidad.
        elegida, _ = tts.elegir_voz_espanol([MONICA, EDDY_MX, PAULINA])
        self.assertIs(elegida, PAULINA)
        # En empate de variante y calidad decide el orden del sistema.
        eddy2 = voz("com.apple.eloquence.es-MX.Flo", "Flo (Spanish (Mexico))", ["es_MX"])
        elegida, _ = tts.elegir_voz_espanol([EDDY_MX, eddy2])
        self.assertIs(elegida, EDDY_MX)

    def test_variante_manda_sobre_calidad(self):
        # es-MX eloquence gana a es-ES compact porque la variante se compara antes que la calidad.
        elegida, _ = tts.elegir_voz_espanol([MONICA, EDDY_MX])
        self.assertIs(elegida, EDDY_MX)

    def test_prefiere_es_co(self):
        carlos = voz("com.apple.voice.compact.es-CO.Carlos", "Carlos", ["es_CO"])
        elegida, _ = tts.elegir_voz_espanol([PAULINA, carlos, MONICA])
        self.assertIs(elegida, carlos)

    def test_sin_voz_en_espanol(self):
        elegida, motivo = tts.elegir_voz_espanol([ALBERT, SAMANTHA])
        self.assertIsNone(elegida)
        self.assertIn("ninguna voz en español", motivo)


class TestTTSModuleOffline(unittest.TestCase):
    """Instancia el modo offline con un pyttsx3 falso: sin audio y sin voces del equipo."""

    def test_elige_monica_y_la_fija_en_el_motor(self):
        # Desde el 23-nov-2026 la voz local es Mónica (elegida por el autor en la escucha a ciegas, docs/VOZ.md):
        # gana aunque la regla general prefiera Paulina (es-MX) a Mónica (es-ES).
        modulo, motor = pyttsx3_falso([ALBERT, PAULINA, MONICA])
        with mock.patch.object(tts, "_importar_pyttsx3", return_value=modulo), \
                mock.patch.object(tts, "nombres_voces_say", return_value={"Paulina", "Mónica"}):
            t = tts.TTSModule(verbose=False)
        self.assertEqual(t.mode, "offline")
        self.assertEqual(t.voice, MONICA.id)
        self.assertEqual(t.voice_name, "Mónica")
        self.assertIn("elegida por el autor", t.voice_reason)
        self.assertEqual(motor.propiedades["voice"], MONICA.id)
        self.assertEqual(motor.propiedades["rate"], 150)
        self.assertIn(t.ejecutor_offline, ("say", "pyttsx3"))
        t.cleanup()

    def test_sin_monica_o_sin_preferida_vale_la_regla_general_y_elige_paulina(self):
        # Sin Mónica en el sistema, la regla de siempre (variante, luego calidad) elige Paulina y lo dice.
        modulo, motor = pyttsx3_falso([ALBERT, EDDY_MX, PAULINA])
        with mock.patch.object(tts, "_importar_pyttsx3", return_value=modulo), \
                mock.patch.object(tts, "nombres_voces_say", return_value={"Paulina"}):
            t = tts.TTSModule(verbose=False)
        self.assertEqual((t.voice, t.voice_name), (PAULINA.id, "Paulina"))
        self.assertEqual(motor.propiedades["voice"], PAULINA.id)
        self.assertIn("no hay voz «Mónica»", t.voice_reason)
        t.cleanup()
        # Con voz_preferida=None se vuelve a la regla sola aunque Mónica exista.
        modulo, motor = pyttsx3_falso([ALBERT, MONICA, PAULINA])
        with mock.patch.object(tts, "_importar_pyttsx3", return_value=modulo), \
                mock.patch.object(tts, "nombres_voces_say", return_value={"Paulina", "Mónica"}):
            t = tts.TTSModule(verbose=False, voz_preferida=None)
        self.assertEqual((t.voice, t.voice_name), (PAULINA.id, "Paulina"))
        self.assertEqual(motor.propiedades["voice"], PAULINA.id)
        t.cleanup()

    def test_elegir_voz_local_reconoce_el_nombre_de_macos_y_prefiere_calidad(self):
        # macOS 26 lista «Mónica (Spanish (Spain))»; dos entradas → la de mejor calidad.
        compacta = voz("com.apple.voice.compact.es-ES.Monica", "Mónica (Spanish (Spain))", ["es_ES"])
        mejorada = voz("com.apple.voice.enhanced.es-ES.Monica", "Mónica (Spanish (Spain))", ["es_ES"])
        carlos = voz("com.apple.voice.compact.es-CO.Carlos", "Carlos", ["es_CO"])
        elegida, motivo = tts.elegir_voz_local([carlos, compacta, mejorada])
        self.assertIs(elegida, mejorada)
        self.assertIn("Mónica", motivo)
        # «Monicaca» o una voz cuyo nombre solo contiene «monica» más adelante no cuentan.
        rara = voz("x.Otra", "Otra Mónica", ["es_ES"])
        self.assertIs(tts.elegir_voz_local([carlos, rara])[0], carlos)

    def test_sin_voz_en_espanol_avisa_y_lista(self):
        modulo, motor = pyttsx3_falso([ALBERT, SAMANTHA])
        salida = io.StringIO()
        with mock.patch.object(tts, "_importar_pyttsx3", return_value=modulo), \
                contextlib.redirect_stdout(salida):
            t = tts.TTSModule(verbose=False)
        self.assertIsNone(t.voice)
        self.assertNotIn("voice", motor.propiedades, "no debe fijar una voz que no es en español")
        self.assertIn("AVISO TTS", salida.getvalue())
        self.assertIn("Albert", salida.getvalue())
        self.assertIn("Samantha", salida.getvalue())
        t.cleanup()

    def test_voz_indicada_inexistente(self):
        modulo, _ = pyttsx3_falso([PAULINA])
        with mock.patch.object(tts, "_importar_pyttsx3", return_value=modulo), \
                mock.patch.object(tts.tempfile, "mkdtemp", wraps=tts.tempfile.mkdtemp) as mkdtemp:
            with self.assertRaises(ValueError):
                tts.TTSModule(voice="no.existe", verbose=False)
        mkdtemp.assert_not_called()   # un constructor que falla no deja carpeta temporal

    def test_locucion_con_say_pasa_voz_por_nombre_y_texto_por_stdin(self):
        modulo, motor = pyttsx3_falso([PAULINA])
        ordenes = []

        def run_falso(orden, **kw):
            ordenes.append((orden, kw.get("input")))
            return SimpleNamespace(returncode=0)

        with mock.patch.object(tts, "_importar_pyttsx3", return_value=modulo), \
                mock.patch.object(tts, "ejecutor_offline_por_defecto", return_value="say"), \
                mock.patch.object(tts, "nombres_voces_say", return_value={"Paulina"}), \
                mock.patch.object(tts.subprocess, "run", side_effect=run_falso):
            t = tts.TTSModule(verbose=False)
            t.speak("Buenos días")
        self.assertEqual(ordenes, [(["say", "-r", "150", "-v", "Paulina"], "Buenos días".encode("utf-8"))])
        self.assertEqual(motor.dichos, [], "con say no se usa engine.say")
        self.assertFalse(t.is_speaking)
        t.cleanup()

    def test_say_no_lista_la_voz_avisa(self):
        # `say -v NombreInexistente` termina con código 0 y habla con la voz por defecto
        # (verificado el 14-sep-2026 escribiendo a archivo): el módulo debe avisar al instanciar.
        modulo, _ = pyttsx3_falso([PAULINA])
        salida = io.StringIO()
        with mock.patch.object(tts, "_importar_pyttsx3", return_value=modulo), \
                mock.patch.object(tts, "ejecutor_offline_por_defecto", return_value="say"), \
                mock.patch.object(tts, "nombres_voces_say", return_value={"Alex", "Samantha"}), \
                contextlib.redirect_stdout(salida):
            t = tts.TTSModule(verbose=False)
        self.assertEqual(t.voice_name, "Paulina")
        self.assertIn("AVISO TTS", salida.getvalue())
        self.assertIn("no lista la voz «Paulina»", salida.getvalue())
        t.cleanup()

    def test_parsear_salida_de_say(self):
        texto = (
            "Eddy (Spanish (Mexico)) es_MX    # ¡Hola! Me llamo Eddy.\n"
            "Mónica              es_ES    # ¡Hola! Me llamo Mónica.\n"
            "Paulina             es_MX    # ¡Hola! Me llamo Paulina.\n"
            "Albert              en_US    # Hello! My name is Albert.\n"
            "linea sin formato\n"
        )
        self.assertEqual(tts._parsear_salida_say(texto), {
            "Eddy (Spanish (Mexico))": "es_MX", "Mónica": "es_ES", "Paulina": "es_MX", "Albert": "en_US",
        })
        self.assertEqual(tts._parsear_salida_say(""), {})

    def test_locucion_con_pyttsx3_encola_antes_del_bucle(self):
        modulo, motor = pyttsx3_falso([PAULINA])
        with mock.patch.object(tts, "_importar_pyttsx3", return_value=modulo), \
                mock.patch.object(tts, "ejecutor_offline_por_defecto", return_value="pyttsx3"):
            t = tts.TTSModule(verbose=False)
            t.speak("Gracias")
            t.speak("Con gusto")
        self.assertEqual(motor.dichos, ["Gracias", "Con gusto"])
        self.assertEqual(motor.bucles, 2)
        self.assertIs(motor.proxy.ocupado, True)
        t.cleanup()


class TestModosOpcionales(unittest.TestCase):
    """edge y gtts sin sus paquetes: la importación de tts ya ocurrió; el constructor debe fallar claro."""

    def test_edge_sin_edge_tts(self):
        with mock.patch.dict(sys.modules, {"edge_tts": None}):
            with self.assertRaises(tts.TTSDependenciaFaltante) as ctx:
                tts.TTSModule(mode="edge")
        self.assertIn("edge-tts", str(ctx.exception))
        self.assertIn("offline", str(ctx.exception))

    def test_gtts_sin_gtts(self):
        with mock.patch.dict(sys.modules, {"gtts": None}):
            with self.assertRaises(tts.TTSDependenciaFaltante) as ctx:
                tts.TTSModule(mode="gtts")
        self.assertIn("gTTS", str(ctx.exception))

    def test_sin_reproductor(self):
        with mock.patch.dict(sys.modules, {"pygame": None}), \
                mock.patch.object(tts.shutil, "which", return_value=None), \
                mock.patch.object(tts.sys, "platform", "linux"):
            self.assertIsNone(tts._detectar_reproductor())

    def test_la_dependencia_faltante_es_runtime_error(self):
        self.assertTrue(issubclass(tts.TTSDependenciaFaltante, RuntimeError))


def audios_sueltos(t) -> list[str]:
    """Audios de locución que quedan en la carpeta temporal del módulo (N-36), sin contar la
    SUBCARPETA de la cache, que tiene su propio ciclo de vida y sus propias pruebas."""
    return sorted(n for n in os.listdir(t._temp_dir)
                  if os.path.isfile(os.path.join(t._temp_dir, n)))


class _DoblesSay:
    """TTSModule offline con `say`, con subprocess.run sustituido: no se ejecuta `say` ni `afplay`.

    Se construye con sintetizador_persistente=False para que estas pruebas sigan mirando el camino
    de `say` (el sintetizador persistente de macOS tiene sus propias pruebas, con dobles de pyobjc)."""

    def _modulo_say(self, run_falso, **kw):
        modulo, motor = pyttsx3_falso([PAULINA])
        parches = [mock.patch.object(tts, "_importar_pyttsx3", return_value=modulo),
                   mock.patch.object(tts, "ejecutor_offline_por_defecto", return_value="say"),
                   mock.patch.object(tts, "nombres_voces_say", return_value={"Paulina"}),
                   mock.patch.object(tts.subprocess, "run", side_effect=run_falso),
                   mock.patch.object(tts.shutil, "which", side_effect=lambda p: f"/usr/bin/{p}")]
        for p in parches:
            p.start()
            self.addCleanup(p.stop)
        kw.setdefault("sintetizador_persistente", False)
        return tts.TTSModule(verbose=False, **kw), motor

    @staticmethod
    def _esperar(condicion, tope=2.0):
        limite = time.monotonic() + tope
        while not condicion() and time.monotonic() < limite:
            time.sleep(0.005)
        return condicion()


class TestSintesisAArchivo(_DoblesSay, unittest.TestCase):
    """§2.2.7 (15-sep-2026): offline sintetiza a archivo antes de la orden de reproducción, y las
    locuciones se reproducen en orden sin descartarse."""

    def test_say_sintetiza_a_aiff_y_luego_afplay(self):
        ordenes = []

        def run_falso(orden, **kw):
            ordenes.append((list(orden), kw.get("input")))
            if orden[0] == "say":
                with open(orden[orden.index("-o") + 1], "wb") as f:
                    f.write(b"FORM-falso")
            return SimpleNamespace(returncode=0)

        t, motor = self._modulo_say(run_falso)
        self.assertTrue(t.soporta_preparar)
        t.prepare("Buenos días")
        self.assertTrue(t.esperar_preparado(2.0))
        say = ordenes[0][0]
        self.assertEqual(say[:6], ["say", "-r", "150", "-v", "Paulina", "-o"])
        self.assertTrue(say[6].endswith(".aiff"))
        self.assertEqual(ordenes[0][1], "Buenos días".encode("utf-8"))
        self.assertEqual(len(ordenes), 1, "prepare no debe reproducir")
        t.play_prepared_async()
        self.assertTrue(self._esperar(lambda: len(ordenes) == 2))
        self.assertEqual(ordenes[1][0], ["afplay", say[6]])
        self.assertEqual(motor.dichos, [])
        t.cleanup()

    def test_fallo_de_sintesis_no_espera_hasta_el_tope(self):
        def run_falso(orden, **kw):
            raise subprocess.CalledProcessError(1, orden)

        t, _ = self._modulo_say(run_falso)
        inicio = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            t.prepare("Hola")
            listo = t.esperar_preparado(5.0)
        self.assertFalse(listo)
        self.assertLess(time.monotonic() - inicio, 1.0)
        self.assertIn("CalledProcessError", t.error_preparacion)
        t.cleanup()

    def test_la_orden_toma_el_archivo_del_momento(self):
        ordenes = []

        def run_falso(orden, **kw):
            ordenes.append(list(orden))
            if orden[0] == "say":
                with open(orden[orden.index("-o") + 1], "wb") as f:
                    f.write(b"x")
            return SimpleNamespace(returncode=0)

        t, _ = self._modulo_say(run_falso)
        t.prepare("uno"); self.assertTrue(t.esperar_preparado(2.0))
        archivo_uno = ordenes[-1][-1]
        t.play_prepared_async()
        t.prepare("dos"); self.assertTrue(t.esperar_preparado(2.0))
        self.assertTrue(self._esperar(lambda: any(o[0] == "afplay" for o in ordenes)))
        self.assertEqual([o for o in ordenes if o[0] == "afplay"][0], ["afplay", archivo_uno])
        t.cleanup()

    def test_locuciones_en_orden_sin_descartes(self):
        habladas, en_curso, solapes = [], [0], []
        candado = threading.Lock()

        def run_falso(orden, **kw):
            with candado:
                en_curso[0] += 1
                solapes.append(en_curso[0])
            time.sleep(0.05)
            habladas.append(kw.get("input").decode("utf-8"))
            with candado:
                en_curso[0] -= 1
            return SimpleNamespace(returncode=0)

        t, _ = self._modulo_say(run_falso)
        t.speak_async("primera")
        time.sleep(0.01)
        t.speak_async("segunda")          # antes se descartaba porque la primera seguía sonando
        t.speak_async("tercera")
        self.assertTrue(self._esperar(lambda: len(habladas) == 3, tope=3.0))
        self.assertEqual(habladas[0], "primera")
        self.assertEqual(sorted(habladas[1:]), ["segunda", "tercera"])
        self.assertEqual(max(solapes), 1, "nunca dos locuciones a la vez")
        t.cleanup()

    def test_pyttsx3_sintetiza_con_save_to_file(self):
        modulo, motor = pyttsx3_falso([PAULINA])
        with mock.patch.object(tts, "_importar_pyttsx3", return_value=modulo), \
                mock.patch.object(tts, "ejecutor_offline_por_defecto", return_value="pyttsx3"):
            t = tts.TTSModule(verbose=False)
            t.prepare("Gracias")
            self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual(motor.dichos, [("archivo", "Gracias")])
        self.assertTrue(t._prepared_file.endswith(".wav"))
        self.assertIs(motor.proxy.ocupado, True)
        t.cleanup()

    # --- Revisión independiente (15-sep-2026) ---

    def _preparacion_reemplazada(self, vieja_falla):
        """prepare("vieja") queda detenida; prepare("nueva") termina; luego la vieja termina."""
        compuerta = threading.Event()

        def run_falso(orden, **kw):
            texto = kw.get("input").decode("utf-8")
            if texto == "vieja":
                compuerta.wait(2.0)
                if vieja_falla:
                    raise subprocess.CalledProcessError(1, orden)
            with open(orden[orden.index("-o") + 1], "wb") as f:
                f.write(texto.encode("utf-8"))
            return SimpleNamespace(returncode=0)

        t, _ = self._modulo_say(run_falso)
        t.prepare("vieja")
        evento_viejo = t._evento_preparado
        t.prepare("nueva")
        self.assertTrue(t.esperar_preparado(2.0))
        with contextlib.redirect_stdout(io.StringIO()):
            compuerta.set()
            self.assertTrue(evento_viejo.wait(2.0))
        self.assertTrue(t.is_prepared, "la preparación vieja no debe borrar el archivo de la nueva")
        with open(t._prepared_file, "rb") as f:
            self.assertEqual(f.read(), b"nueva")
        self.assertIsNone(t.error_preparacion)
        # Pendiente de N-34: el audio de la preparación vieja (si lo hubo) no queda en la carpeta temporal.
        self.assertEqual(audios_sueltos(t), [os.path.basename(t._prepared_file)])
        t.cleanup()

    def test_preparacion_vieja_exitosa_no_reemplaza_la_nueva(self):
        self._preparacion_reemplazada(vieja_falla=False)

    def test_preparacion_vieja_fallida_no_borra_la_nueva(self):
        self._preparacion_reemplazada(vieja_falla=True)

    def test_reproducciones_preparadas_no_se_solapan(self):
        en_curso, maximo, reproducidas = [0], [0], []
        candado = threading.Lock()

        def run_falso(orden, **kw):
            if orden[0] == "say":
                with open(orden[orden.index("-o") + 1], "wb") as f:
                    f.write(b"x")
                return SimpleNamespace(returncode=0)
            with candado:
                en_curso[0] += 1
                maximo[0] = max(maximo[0], en_curso[0])
            time.sleep(0.05)
            reproducidas.append(orden[1])
            with candado:
                en_curso[0] -= 1
            return SimpleNamespace(returncode=0)

        t, _ = self._modulo_say(run_falso)
        for texto in ("uno", "dos", "tres"):
            t.prepare(texto)
            self.assertTrue(t.esperar_preparado(2.0))
            t.play_prepared_async()
        self.assertTrue(self._esperar(lambda: len(reproducidas) == 3, tope=3.0))
        self.assertEqual(maximo[0], 1, "nunca dos reproducciones a la vez")
        t.cleanup()

    def test_la_orden_toma_el_archivo_aunque_el_hilo_arranque_tarde(self):
        # Determinista: el hilo de reproducción no corre hasta después del prepare siguiente.
        reproducidas, pendientes = [], []
        hilo_real = threading.Thread

        class HiloDiferido:
            def __init__(self, target, args=(), daemon=None):
                self.target, self.args, self.daemon = target, args, daemon

            def start(self):
                if getattr(self.target, "__name__", "") == "_prepare_audio":
                    hilo_real(target=self.target, args=self.args, daemon=True).start()
                else:
                    pendientes.append(self)

        def run_falso(orden, **kw):
            if orden[0] == "say":
                with open(orden[orden.index("-o") + 1], "wb") as f:
                    f.write(kw.get("input"))
            else:
                with open(orden[1], "rb") as f:
                    reproducidas.append(f.read().decode("utf-8"))
            return SimpleNamespace(returncode=0)

        t, _ = self._modulo_say(run_falso)
        with mock.patch.object(tts.threading, "Thread", HiloDiferido):
            t.prepare("uno")
            self.assertTrue(t.esperar_preparado(2.0))
            t.play_prepared_async()
            t.prepare("dos")
            self.assertTrue(t.esperar_preparado(2.0))
        for hilo in pendientes:
            hilo.target(*hilo.args)
        self.assertEqual(reproducidas, ["uno"])
        t.cleanup()

    def test_pyttsx3_sin_reproductor_de_wav_no_declara_preparar(self):
        modulo, _ = pyttsx3_falso([PAULINA])
        with mock.patch.object(tts, "_importar_pyttsx3", return_value=modulo), \
                mock.patch.object(tts, "ejecutor_offline_por_defecto", return_value="pyttsx3"):
            t = tts.TTSModule(verbose=False)
        for reproductor, esperado in ((None, False), ("start", False), ("ffplay", True), ("pygame", True)):
            with mock.patch.object(tts, "_detectar_reproductor", return_value=reproductor):
                self.assertIs(t.soporta_preparar, esperado, reproductor)
        t.cleanup()


def _say_escribe(orden, kw, contenido=None):
    """Doble de `say -o archivo`: escribe el archivo que pide la orden."""
    with open(orden[orden.index("-o") + 1], "wb") as f:
        f.write(contenido if contenido is not None else kw.get("input") or b"x")


class _SinTrazaDeHilo:
    """Registra las excepciones que escaparían de un hilo (threading.excepthook) durante la prueba."""

    def setUp(self):
        super().setUp()
        self.trazas = []
        parche = mock.patch.object(threading, "excepthook", side_effect=lambda args: self.trazas.append(args))
        parche.start()
        self.addCleanup(parche.stop)


class TestAudiosTemporales(_DoblesSay, unittest.TestCase):
    """Pendiente de N-34: los audios sintetizados no se acumulan hasta cleanup()."""

    def test_audio_se_borra_tras_reproducirlo(self):
        existia_al_reproducir = []

        def run_falso(orden, **kw):
            if orden[0] == "say":
                _say_escribe(orden, kw)
            else:
                existia_al_reproducir.append(os.path.exists(orden[1]))
            return SimpleNamespace(returncode=0)

        t, _ = self._modulo_say(run_falso)
        for texto in ("uno", "dos"):
            t.prepare(texto)
            self.assertTrue(t.esperar_preparado(2.0))
            archivo = t._prepared_file
            t.play_prepared_async()
            self.assertTrue(self._esperar(lambda: not os.path.exists(archivo)), "el audio debe borrarse al sonar")
        self.assertEqual(existia_al_reproducir, [True, True], "no se borra antes de reproducir")
        self.assertEqual(audios_sueltos(t), [])
        self.assertEqual(t.errores_reproduccion, 0)
        t.cleanup()

    def test_preparacion_reemplazada_sin_reproducir_se_borra(self):
        reproducidos = []

        def run_falso(orden, **kw):
            if orden[0] == "say":
                _say_escribe(orden, kw)
            else:
                with open(orden[1], "rb") as f:
                    reproducidos.append(f.read())
            return SimpleNamespace(returncode=0)

        t, _ = self._modulo_say(run_falso)
        t.prepare("uno"); self.assertTrue(t.esperar_preparado(2.0))
        archivo_uno = t._prepared_file
        t.prepare("dos"); self.assertTrue(t.esperar_preparado(2.0))
        archivo_dos = t._prepared_file
        self.assertFalse(os.path.exists(archivo_uno), "«uno» se reemplazó sin sonar: su audio sobra")
        self.assertEqual(audios_sueltos(t), [os.path.basename(archivo_dos)])
        t.play_prepared_async()
        self.assertTrue(self._esperar(lambda: not os.path.exists(archivo_dos)))
        self.assertEqual(reproducidos, [b"dos"])
        t.cleanup()

    def test_la_orden_conserva_su_archivo_aunque_se_prepare_otro(self):
        # play_prepared_async toma el archivo en la llamada: el prepare siguiente no debe borrarlo.
        compuerta, reproducidos = threading.Event(), []

        def run_falso(orden, **kw):
            if orden[0] == "say":
                _say_escribe(orden, kw)
            else:
                compuerta.wait(2.0)
                with open(orden[1], "rb") as f:
                    reproducidos.append(f.read())
            return SimpleNamespace(returncode=0)

        t, _ = self._modulo_say(run_falso)
        t.prepare("uno"); self.assertTrue(t.esperar_preparado(2.0))
        t.play_prepared_async()
        t.prepare("dos"); self.assertTrue(t.esperar_preparado(2.0))
        t.prepare("tres"); self.assertTrue(t.esperar_preparado(2.0))   # «dos» se reemplaza sin sonar
        compuerta.set()
        self.assertTrue(self._esperar(lambda: reproducidos == [b"uno"]))
        self.assertTrue(self._esperar(lambda: len(audios_sueltos(t)) == 1))
        self.assertEqual(audios_sueltos(t), [os.path.basename(t._prepared_file)])
        t.cleanup()

    def test_sintesis_fallida_borra_el_audio_parcial(self):
        def run_falso(orden, **kw):
            _say_escribe(orden, kw, b"parcial")
            raise subprocess.CalledProcessError(1, orden)

        t, _ = self._modulo_say(run_falso)
        with contextlib.redirect_stdout(io.StringIO()):
            t.prepare("Hola")
            self.assertFalse(t.esperar_preparado(2.0))
        self.assertEqual(audios_sueltos(t), [])
        t.cleanup()

    def test_gtts_borra_el_mp3_tras_hablar(self):
        existia_al_reproducir = []

        class GTTSFalso:
            def __init__(self, text, lang, tld):
                self.text = text

            def save(self, ruta):
                with open(ruta, "wb") as f:
                    f.write(self.text.encode("utf-8"))

        def run_falso(orden, **kw):
            existia_al_reproducir.append((orden[0], os.path.exists(orden[1])))
            return SimpleNamespace(returncode=0)

        with mock.patch.dict(sys.modules, {"gtts": SimpleNamespace(gTTS=GTTSFalso)}), \
                mock.patch.object(tts, "_detectar_reproductor", return_value="afplay"), \
                mock.patch.object(tts.shutil, "which", return_value=None), \
                mock.patch.object(tts.subprocess, "run", side_effect=run_falso):
            t = tts.TTSModule(mode="gtts", verbose=False)
            t.speak("Gracias")
        self.assertEqual(existia_al_reproducir, [("afplay", True)])
        self.assertEqual(audios_sueltos(t), [])
        t.cleanup()

    def test_reproductor_externo_de_windows_deja_el_archivo_para_cleanup(self):
        t, _ = self._modulo_say(lambda orden, **kw: SimpleNamespace(returncode=0))
        ruta = t._next_audio_file(".mp3")
        with open(ruta, "wb") as f:
            f.write(b"x")
        with mock.patch.object(tts, "_detectar_reproductor", return_value="start"), \
                mock.patch.object(tts.shutil, "which", return_value=None), \
                mock.patch.object(tts.os, "startfile", create=True) as startfile, \
                mock.patch.object(tts.time, "sleep"):
            t._reproducir(ruta, "texto")
        startfile.assert_called_once_with(ruta)
        self.assertTrue(os.path.exists(ruta), "el reproductor externo puede tenerlo abierto")
        t.cleanup()
        self.assertFalse(os.path.exists(t._temp_dir))

    def test_borrar_audio_no_sale_de_la_carpeta_temporal(self):
        t, _ = self._modulo_say(lambda orden, **kw: SimpleNamespace(returncode=0))
        ajeno = os.path.join(os.path.dirname(t._temp_dir), f"ajeno_{os.getpid()}.aiff")
        with open(ajeno, "wb") as f:
            f.write(b"x")
        self.addCleanup(lambda: os.path.exists(ajeno) and os.remove(ajeno))
        t._borrar_audio(ajeno)
        self.assertTrue(os.path.exists(ajeno))
        t.cleanup()


class TestErroresDeReproduccion(_SinTrazaDeHilo, _DoblesSay, unittest.TestCase):
    """Pendiente de N-34: las excepciones de los hilos de locución se capturan, se avisan una vez y se exponen."""

    def test_reproductor_que_falla_se_captura_y_se_avisa_una_vez(self):
        def run_falso(orden, **kw):
            if orden[0] == "say":
                _say_escribe(orden, kw)
                return SimpleNamespace(returncode=0)
            raise subprocess.CalledProcessError(1, orden)

        t, _ = self._modulo_say(run_falso)
        salida = io.StringIO()
        archivos = []
        with contextlib.redirect_stdout(salida):
            for texto in ("uno", "dos"):
                t.prepare(texto)
                self.assertTrue(t.esperar_preparado(2.0))
                archivos.append(t._prepared_file)
                t.play_prepared_async()
                self.assertTrue(self._esperar(lambda: not t.is_speaking and not os.path.exists(archivos[-1])))
            self.assertTrue(self._esperar(lambda: t.errores_reproduccion == 2))
        self.assertEqual(self.trazas, [], "ninguna excepción debe salir del hilo")
        self.assertEqual(salida.getvalue().count("[AVISO TTS]"), 1, salida.getvalue())
        self.assertIn("CalledProcessError", salida.getvalue())
        self.assertIn("CalledProcessError", t.ultimo_error_reproduccion)
        self.assertIsInstance(t.ultima_excepcion_reproduccion, subprocess.CalledProcessError)
        self.assertEqual(audios_sueltos(t), [], "el audio se borra también si la reproducción falla")
        estado = t.estado_reproduccion()
        self.assertEqual((estado["errores_reproduccion"], estado["locuciones_descartadas_al_cerrar"]), (2, 0))
        self.assertFalse(estado["cerrado"])
        t.cleanup()

    def test_un_tipo_nuevo_de_error_vuelve_a_avisar(self):
        fallo = [subprocess.CalledProcessError(1, ["afplay"])]

        def run_falso(orden, **kw):
            if orden[0] == "say":
                _say_escribe(orden, kw)
                return SimpleNamespace(returncode=0)
            raise fallo[0]

        t, _ = self._modulo_say(run_falso)
        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            for n, error in enumerate((fallo[0], subprocess.TimeoutExpired(["afplay"], 60), fallo[0]), 1):
                fallo[0] = error
                t.prepare("x"); self.assertTrue(t.esperar_preparado(2.0))
                t.play_prepared_async()
                self.assertTrue(self._esperar(lambda: t.errores_reproduccion == n))
        self.assertEqual(salida.getvalue().count("[AVISO TTS]"), 2, salida.getvalue())
        self.assertIn("CalledProcessError", t.ultimo_error_reproduccion)
        self.assertEqual(self.trazas, [])
        t.cleanup()

    def test_speak_async_con_motor_que_falla(self):
        modulo, motor = pyttsx3_falso([PAULINA])

        def falla():
            raise RuntimeError("motor caído")

        motor.runAndWait = falla
        with mock.patch.object(tts, "_importar_pyttsx3", return_value=modulo), \
                mock.patch.object(tts, "ejecutor_offline_por_defecto", return_value="pyttsx3"):
            t = tts.TTSModule(verbose=False)
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            t.speak_async("Hola")
            self.assertTrue(self._esperar(lambda: t.errores_reproduccion == 1))
        self.assertEqual(t.ultimo_error_reproduccion, "RuntimeError: motor caído")
        self.assertIn("[AVISO TTS]", salida.getvalue())
        self.assertFalse(t.is_speaking)
        self.assertEqual(self.trazas, [])
        with self.assertRaises(RuntimeError):     # speak() síncrono sí propaga a quien llama
            t.speak("Hola")
        t.cleanup()

    def test_archivo_borrado_mientras_espera_turno(self):
        reproducidos = []

        def run_falso(orden, **kw):
            if orden[0] == "say":
                _say_escribe(orden, kw)
            else:
                reproducidos.append(orden[1])
            return SimpleNamespace(returncode=0)

        t, _ = self._modulo_say(run_falso)
        t.prepare("uno"); self.assertTrue(t.esperar_preparado(2.0))
        archivo = t._prepared_file
        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            with t._candado_reproduccion:            # otra locución «suena»: esta espera turno
                t.play_prepared_async()
                os.remove(archivo)                   # el audio desaparece mientras espera
            self.assertTrue(self._esperar(lambda: t.errores_reproduccion == 1))
        self.assertEqual(reproducidos, [])
        self.assertIn("FileNotFoundError", t.ultimo_error_reproduccion)
        self.assertIn("[AVISO TTS]", salida.getvalue())
        self.assertEqual(self.trazas, [])
        t.cleanup()

    def test_cleanup_con_locuciones_en_cola_termina_limpio(self):
        sonando, soltar, reproducidos = threading.Event(), threading.Event(), []

        def run_falso(orden, **kw):
            if orden[0] == "say":
                _say_escribe(orden, kw)
                return SimpleNamespace(returncode=0)
            reproducidos.append(orden[1])
            sonando.set()
            soltar.wait(3.0)
            return SimpleNamespace(returncode=0)

        t, _ = self._modulo_say(run_falso)
        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            t.prepare("uno"); self.assertTrue(t.esperar_preparado(2.0))
            t.play_prepared_async()
            self.assertTrue(sonando.wait(2.0))
            for texto in ("dos", "tres"):            # quedan en cola detrás de «uno»
                t.prepare(texto); self.assertTrue(t.esperar_preparado(2.0))
                t.play_prepared_async()
            t.speak_async("cuatro")                  # también en cola (locución directa)
            time.sleep(0.05)
            inicio = time.monotonic()
            t.cleanup()
            self.assertLess(time.monotonic() - inicio, 0.5, "cleanup no espera a la cola")
            self.assertFalse(os.path.exists(t._temp_dir))
            soltar.set()
            self.assertTrue(self._esperar(lambda: t.locuciones_descartadas_al_cerrar == 3))
            self.assertTrue(self._esperar(lambda: not t.is_speaking))
            t.cleanup()                              # idempotente
            # Tras cerrar: prepare falla de inmediato y speak no habla.
            t.prepare("cinco")
            inicio = time.monotonic()
            self.assertFalse(t.esperar_preparado(5.0))
            self.assertLess(time.monotonic() - inicio, 0.5)
            self.assertIn("cerrado", t.error_preparacion)
            t.speak("seis")
        self.assertEqual(len(reproducidos), 1, "solo sonó la que ya estaba sonando")
        self.assertEqual(t.errores_reproduccion, 0)
        self.assertEqual(t.locuciones_descartadas_al_cerrar, 4)
        self.assertEqual(self.trazas, [])
        self.assertNotIn("[AVISO TTS]", salida.getvalue())
        self.assertTrue(t.estado_reproduccion()["cerrado"])
        self.assertFalse(os.path.exists(t._temp_dir), "nada vuelve a crear la carpeta temporal")

    def test_error_provocado_por_el_cierre_no_cuenta_como_error(self):
        # El reproductor falla porque cleanup() borró el audio mientras sonaba: es el cierre, no un error.
        sonando, soltar = threading.Event(), threading.Event()

        def run_falso(orden, **kw):
            if orden[0] == "say":
                _say_escribe(orden, kw)
                return SimpleNamespace(returncode=0)
            sonando.set()
            soltar.wait(3.0)
            raise subprocess.CalledProcessError(1, orden)

        t, _ = self._modulo_say(run_falso)
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            t.prepare("uno"); self.assertTrue(t.esperar_preparado(2.0))
            t.play_prepared_async()
            self.assertTrue(sonando.wait(2.0))
            t.cleanup()
            soltar.set()
            self.assertTrue(self._esperar(lambda: t.locuciones_descartadas_al_cerrar == 1))
        self.assertEqual(t.errores_reproduccion, 0)
        self.assertIsNone(t.ultimo_error_reproduccion)
        self.assertNotIn("[AVISO TTS]", salida.getvalue())
        self.assertEqual(self.trazas, [])

    def test_cola_tras_cleanup_no_suena_aunque_el_audio_siga_en_disco(self):
        # Si la carpeta no se pudo borrar (p. ej. un archivo en uso en Windows), la cola igual no suena.
        sonando, soltar, reproducidos = threading.Event(), threading.Event(), []

        def run_falso(orden, **kw):
            if orden[0] == "say":
                _say_escribe(orden, kw)
                return SimpleNamespace(returncode=0)
            reproducidos.append(orden[1])
            sonando.set()
            soltar.wait(3.0)
            return SimpleNamespace(returncode=0)

        t, _ = self._modulo_say(run_falso)
        t.prepare("uno"); self.assertTrue(t.esperar_preparado(2.0))
        t.play_prepared_async()
        self.assertTrue(sonando.wait(2.0))
        t.prepare("dos"); self.assertTrue(t.esperar_preparado(2.0))
        archivo_dos = t._prepared_file
        t.play_prepared_async()
        with mock.patch.object(tts.shutil, "rmtree"):
            t.cleanup()
        self.assertTrue(os.path.exists(archivo_dos))
        soltar.set()
        self.assertTrue(self._esperar(lambda: t.locuciones_descartadas_al_cerrar == 1))
        self.assertEqual(len(reproducidos), 1)
        self.assertFalse(os.path.exists(archivo_dos), "el audio descartado también se borra")
        self.assertEqual((t.errores_reproduccion, self.trazas), (0, []))
        tts.shutil.rmtree(t._temp_dir, ignore_errors=True)



# ---------------------------------------------------------------------------
# Cache de audio por texto (16-sep-2026, pendiente de N-34)
# ---------------------------------------------------------------------------

class _CuentaSay:
    """run_falso que escribe el archivo de `say -o` y cuenta cuántas veces se sintetizó cada texto."""

    def __init__(self, contenido=None, falla=(), reproducidos=None):
        self.textos: list[str] = []
        self.ordenes: list[list[str]] = []
        self.reproducidos = reproducidos if reproducidos is not None else []
        self.contenido = contenido
        self.falla = set(falla)

    def __call__(self, orden, **kw):
        if orden[0] == "say":
            texto = (kw.get("input") or b"").decode("utf-8")
            self.textos.append(texto)
            self.ordenes.append(list(orden))
            if texto in self.falla:
                raise subprocess.CalledProcessError(1, orden)
            _say_escribe(orden, kw, self.contenido)
        else:                                    # afplay: anota qué sonaría (no reproduce nada)
            with open(orden[1], "rb") as f:
                self.reproducidos.append(f.read())
        return SimpleNamespace(returncode=0)


class TestCacheDeAudio(_DoblesSay, unittest.TestCase):
    """La cache deja la síntesis en casi cero cuando vuelve el mismo texto (medido: 0,14 ms frente a
    458 ms), sin cambiar la voz ni el ritmo y sin crecer sin control."""

    def _preparar(self, t, texto):
        t.prepare(texto)
        self.assertTrue(t.esperar_preparado(2.0), texto)
        return t._prepared_file

    def test_la_clave_de_la_cache_es_el_texto_exacto(self):
        # Añadida tras la verificación del 16-sep: un mutante que normalizaba la clave (minúsculas o
        # sin espacios de los extremos) sobrevivía a la suite, y habría servido «Hola» cuando el
        # prototipo pidió «hola»: audio equivocado, no solo una estadística mal.
        say = _CuentaSay()
        t, _ = self._modulo_say(say)
        self._preparar(t, "Hola")
        self.assertIs(t.ultima_preparacion_desde_cache, False)
        for distinto in ("hola", "HOLA", "Hola ", " Hola", "Hola."):
            self._preparar(t, distinto)
            self.assertIs(t.ultima_preparacion_desde_cache, False,
                          f"{distinto!r} no es el mismo texto que 'Hola': no puede ser un acierto")
            with open(t._prepared_file, "rb") as f:
                self.assertEqual(f.read(), distinto.encode("utf-8"))
        self.assertEqual(say.textos, ["Hola", "hola", "HOLA", "Hola ", " Hola", "Hola."])
        self.assertEqual(t.estado_cache()["aciertos"], 0)
        self._preparar(t, "Hola")            # el texto exacto sí acierta
        self.assertIs(t.ultima_preparacion_desde_cache, True)

    def test_acierto_de_cache_no_vuelve_a_sintetizar(self):
        say = _CuentaSay()
        t, _ = self._modulo_say(say)
        uno = self._preparar(t, "Gracias")
        self.assertIs(t.ultima_preparacion_desde_cache, False)
        self.assertEqual(say.textos, ["Gracias"])
        dos = self._preparar(t, "Gracias")
        self.assertEqual(say.textos, ["Gracias"], "el segundo prepare debe salir de la cache")
        self.assertIs(t.ultima_preparacion_desde_cache, True)
        self.assertEqual(t.ultimo_sintetizador, "cache")
        self.assertNotEqual(uno, dos, "cada preparación tiene su propio archivo")
        with open(dos, "rb") as f:
            self.assertEqual(f.read(), b"Gracias")
        estado = t.estado_cache()
        self.assertEqual((estado["aciertos"], estado["fallos"], estado["entradas"]), (1, 1, 1))
        self.assertGreater(estado["bytes"], 0)
        t.cleanup()

    def test_fallo_de_cache_sintetiza_y_lo_declara(self):
        say = _CuentaSay()
        t, _ = self._modulo_say(say)
        self._preparar(t, "Gracias")
        self._preparar(t, "Hola")
        self.assertEqual(say.textos, ["Gracias", "Hola"])
        self.assertIs(t.ultima_preparacion_desde_cache, False)
        self.assertEqual(t.estado_cache()["entradas"], 2)
        t.cleanup()

    def test_la_cache_entrega_una_copia_que_el_reproductor_puede_borrar(self):
        say = _CuentaSay()
        t, _ = self._modulo_say(say)
        self._preparar(t, "Gracias")
        archivo = self._preparar(t, "Gracias")           # acierto: copia nueva
        t.play_prepared_async()
        self.assertTrue(self._esperar(lambda: not os.path.exists(archivo)))
        self.assertEqual(say.reproducidos, [b"Gracias"])
        # El audio guardado sigue en la cache: el tercer prepare vuelve a acertar.
        self._preparar(t, "Gracias")
        self.assertEqual(say.textos, ["Gracias"])
        self.assertEqual(t.estado_cache()["aciertos"], 2)
        t.cleanup()

    def test_cambiar_el_ritmo_invalida_la_cache(self):
        say = _CuentaSay()
        t, _ = self._modulo_say(say)
        self._preparar(t, "Gracias")
        t.rate = 200                                     # otra velocidad: el audio guardado ya no vale
        self._preparar(t, "Gracias")
        self.assertEqual(say.textos, ["Gracias", "Gracias"])
        self.assertIs(t.ultima_preparacion_desde_cache, False)
        self.assertEqual([o[o.index("-r") + 1] for o in say.ordenes], ["150", "200"])
        self.assertEqual(t.estado_cache()["entradas"], 1, "la cache vieja se vacía, no se acumula")
        t.cleanup()

    def test_cambiar_la_voz_invalida_la_cache(self):
        say = _CuentaSay()
        t, _ = self._modulo_say(say)
        self._preparar(t, "Gracias")
        t.voice_name = "Mónica"
        t.voice = MONICA.id
        self._preparar(t, "Gracias")
        self.assertEqual(say.textos, ["Gracias", "Gracias"])
        self.assertIs(t.ultima_preparacion_desde_cache, False)
        self.assertEqual([o[o.index("-v") + 1] for o in say.ordenes], ["Paulina", "Mónica"])
        self.assertEqual(t.estado_cache()["entradas"], 1)
        t.cleanup()

    def test_tope_de_entradas_desaloja_la_usada_hace_mas_tiempo(self):
        say = _CuentaSay()
        t, _ = self._modulo_say(say, cache_max_entradas=2)
        for texto in ("uno", "dos"):
            self._preparar(t, texto)
        self._preparar(t, "uno")                         # acierto: «uno» pasa a ser la más reciente
        guardados = os.listdir(t._cache_dir)
        self._preparar(t, "tres")                        # entra: debe salir «dos», no «uno»
        self.assertEqual(t.estado_cache()["entradas"], 2)
        self.assertEqual(t.estado_cache()["desalojos"], 1)
        self.assertEqual(len(os.listdir(t._cache_dir)), 2, "el audio desalojado se borra")
        self._preparar(t, "uno")
        self.assertEqual(say.textos, ["uno", "dos", "tres"], "«uno» seguía en la cache")
        self._preparar(t, "dos")
        self.assertEqual(say.textos, ["uno", "dos", "tres", "dos"], "«dos» fue la desalojada")
        self.assertEqual(len(guardados), 2)
        t.cleanup()

    def test_tope_de_bytes_desaloja(self):
        say = _CuentaSay(contenido=b"x" * 100)
        t, _ = self._modulo_say(say, cache_max_bytes=250)
        for texto in ("uno", "dos", "tres"):
            self._preparar(t, texto)
        estado = t.estado_cache()
        self.assertEqual(estado["entradas"], 2)
        self.assertLessEqual(estado["bytes"], 250)
        self.assertEqual(estado["desalojos"], 1)
        t.cleanup()

    def test_un_audio_mayor_que_el_tope_no_entra_ni_vacia_la_cache(self):
        say = _CuentaSay(contenido=b"x" * 40)
        t, _ = self._modulo_say(say, cache_max_bytes=100)
        self._preparar(t, "corta")
        say.contenido = b"x" * 400
        self._preparar(t, "larguísima")
        self.assertEqual(t.estado_cache()["entradas"], 1)
        self._preparar(t, "larguísima")
        self.assertEqual(say.textos.count("larguísima"), 2, "sin cache, se sintetiza cada vez")
        self._preparar(t, "corta")
        self.assertEqual(say.textos.count("corta"), 1, "la entrada que sí cabía no se perdió")
        t.cleanup()

    def test_cache_desactivada_sintetiza_siempre(self):
        say = _CuentaSay()
        t, _ = self._modulo_say(say, cache=False)
        self._preparar(t, "Gracias")
        self._preparar(t, "Gracias")
        self.assertEqual(say.textos, ["Gracias", "Gracias"])
        self.assertIs(t.ultima_preparacion_desde_cache, False)
        estado = t.estado_cache()
        self.assertEqual((estado["activa"], estado["entradas"], estado["aciertos"]), (False, 0, 0))
        self.assertEqual(os.listdir(t._cache_dir), [], "con la cache apagada no se guarda nada")
        t.cleanup()

    def test_cleanup_borra_la_cache_y_lo_declara(self):
        say = _CuentaSay()
        t, _ = self._modulo_say(say)
        self._preparar(t, "Gracias")
        carpeta = t._cache_dir
        self.assertTrue(os.path.isdir(carpeta))
        t.cleanup()
        self.assertFalse(os.path.exists(carpeta), "la cache vive dentro de la carpeta temporal")
        estado = t.estado_cache()
        self.assertEqual((estado["entradas"], estado["bytes"]), (0, 0))
        t.prepare("Gracias")
        self.assertTrue(t.esperar_preparado(1.0) is False)
        self.assertIsNone(t.ultima_preparacion_desde_cache, "cerrado: no hay preparación que declarar")

    def test_audio_de_la_cache_que_desaparece_no_rompe_la_preparacion(self):
        say = _CuentaSay()
        t, _ = self._modulo_say(say)
        self._preparar(t, "Gracias")
        for nombre in os.listdir(t._cache_dir):          # alguien borró el audio guardado
            os.remove(os.path.join(t._cache_dir, nombre))
        self._preparar(t, "Gracias")
        self.assertEqual(say.textos, ["Gracias", "Gracias"], "se sintetiza de nuevo")
        self.assertIs(t.ultima_preparacion_desde_cache, False)
        self.assertGreaterEqual(t.estado_cache()["errores"], 1)
        t.cleanup()


class TestPrecalentado(_DoblesSay, unittest.TestCase):
    """precalentar() llena la cache con las frases más probables sin estorbar a una locución en curso."""

    def test_precalentar_llena_la_cache_y_luego_se_acierta(self):
        say = _CuentaSay()
        t, _ = self._modulo_say(say)
        self.assertTrue(t.precalentar(["uno", "dos", "uno"]))
        self.assertTrue(t.esperar_precalentado(3.0))
        self.assertEqual(say.textos, ["uno", "dos"], "sin repetir la misma frase")
        self.assertEqual(t.estado_cache()["entradas"], 2)
        self.assertEqual(t.precalentadas, 2)
        t.prepare("dos")
        self.assertTrue(t.esperar_preparado(2.0))
        self.assertIs(t.ultima_preparacion_desde_cache, True)
        self.assertEqual(say.textos, ["uno", "dos"])
        self.assertEqual(audios_sueltos(t), [os.path.basename(t._prepared_file)],
                         "el precalentado no deja audios sueltos")
        t.cleanup()

    def test_precalentar_no_repite_lo_que_ya_esta_en_la_cache(self):
        say = _CuentaSay()
        t, _ = self._modulo_say(say)
        t.prepare("uno"); self.assertTrue(t.esperar_preparado(2.0))
        t.precalentar(["uno", "dos"])
        self.assertTrue(t.esperar_precalentado(3.0))
        self.assertEqual(say.textos, ["uno", "dos"])
        self.assertEqual(t.precalentadas, 1)
        t.cleanup()

    def _con_compuerta(self, texto_lento):
        """Módulo con `say` detenido en una frase concreta hasta abrir la compuerta."""
        compuerta, arranco = threading.Event(), threading.Event()
        say = _CuentaSay()

        def run_falso(orden, **kw):
            if orden[0] == "say" and (kw.get("input") or b"").decode("utf-8") == texto_lento:
                arranco.set()
                compuerta.wait(3.0)
            return say(orden, **kw)

        t, _ = self._modulo_say(run_falso)
        return t, say, compuerta, arranco

    def test_precalentar_espera_a_la_preparacion_en_curso(self):
        t, say, compuerta, arranco = self._con_compuerta("lenta")
        t.prepare("lenta")
        self.assertTrue(arranco.wait(2.0))
        t.precalentar(["otra"])
        time.sleep(0.15)
        self.assertEqual(say.textos, [], "el precalentado no adelanta a una preparación en curso")
        compuerta.set()
        self.assertTrue(t.esperar_preparado(2.0))
        self.assertTrue(t.esperar_precalentado(3.0))
        self.assertEqual(say.textos, ["lenta", "otra"])
        t.cleanup()

    def test_el_precalentado_espera_a_las_dos_preparaciones_solapadas(self):
        # Dos preparaciones pueden estar vivas a la vez (la vieja sigue sintetizando cuando llega la
        # nueva): mientras quede alguna, el precalentado no se cuela.
        say = _CuentaSay()
        puertas = {t: threading.Event() for t in ("vieja", "nueva")}
        arrancadas = {t: threading.Event() for t in ("vieja", "nueva")}

        def run_falso(orden, **kw):
            texto = (kw.get("input") or b"").decode("utf-8")
            if texto in puertas:
                arrancadas[texto].set()
                puertas[texto].wait(3.0)
            return say(orden, **kw)

        t, _ = self._modulo_say(run_falso)
        t.prepare("vieja")
        self.assertTrue(arrancadas["vieja"].wait(2.0))
        t.prepare("nueva")
        self.assertTrue(arrancadas["nueva"].wait(2.0))
        t.precalentar(["otra"])
        puertas["nueva"].set()
        self.assertTrue(t.esperar_preparado(2.0))
        time.sleep(0.15)
        self.assertEqual(say.textos, ["nueva"], "la preparación vieja sigue viva: el precalentado espera")
        puertas["vieja"].set()
        self.assertTrue(t.esperar_precalentado(3.0))
        self.assertEqual(say.textos[-1], "otra")
        t.cleanup()

    def test_precalentar_no_arranca_dos_veces_ni_sin_frases_ni_sin_cache(self):
        t, _say, compuerta, _arranco = self._con_compuerta("lenta")
        self.assertFalse(t.precalentar([]))
        self.assertFalse(t.precalentar([""]))
        self.assertTrue(t.precalentar(["lenta", "otra"]))
        self.assertFalse(t.precalentar(["tercera"]), "ya hay un precalentado en marcha")
        compuerta.set()
        self.assertTrue(t.esperar_precalentado(3.0))
        t.cleanup()
        self.assertFalse(t.precalentar(["cuarta"]), "tras cleanup no se precalienta")
        sin_cache, _ = self._modulo_say(_CuentaSay(), cache=False)
        self.assertFalse(sin_cache.precalentar(["uno"]))
        sin_cache.cleanup()

    def test_una_frase_que_falla_al_precalentar_no_detiene_el_resto(self):
        say = _CuentaSay(falla={"mala"})
        t, _ = self._modulo_say(say)
        t.precalentar(["mala", "buena"])
        self.assertTrue(t.esperar_precalentado(3.0))
        self.assertEqual((t.precalentadas, t.precalentado_fallos), (1, 1))
        self.assertIn("precalentado:", t.ultimo_error_sintetizador)
        self.assertEqual(t.estado_cache()["entradas"], 1)
        self.assertEqual(audios_sueltos(t), [], "ni siquiera el audio a medias queda suelto")
        t.cleanup()

    def test_cleanup_corta_el_precalentado(self):
        t, say, compuerta, arranco = self._con_compuerta("primera")
        t.precalentar(["primera", "segunda", "tercera"])
        self.assertTrue(arranco.wait(2.0))
        with contextlib.redirect_stdout(io.StringIO()):
            t.cleanup()
        compuerta.set()
        # cleanup() no espera a los hilos (N-36): se espera a que la frase en curso termine y ahí para.
        self.assertTrue(self._esperar(lambda: say.textos == ["primera"]))
        time.sleep(0.15)
        self.assertEqual(say.textos, ["primera"], "tras cerrar no se sintetiza nada más")
        self.assertEqual(t.estado_cache()["entradas"], 0)



# ---------------------------------------------------------------------------
# Sintetizador persistente de macOS (NSSpeechSynthesizer) con dobles de pyobjc
# ---------------------------------------------------------------------------

class _RegistroNS:
    def __init__(self):
        self.inits = 0
        self.voces: list[str] = []
        self.ritmos: list[float] = []
        self.textos: list[str] = []
        self.archivos: list[str] = []
        self.stops = 0
        self.vueltas = 0
        self.simultaneos = 0
        self.max_simultaneos = 0
        self.candado = threading.Lock()


class _SintetizadorFalso:
    """Doble de NSSpeechSynthesizer: escribe el archivo y anota voz, ritmo, texto y solapes."""

    def __init__(self, reg, c):
        self.reg, self.c = reg, c
        self._hablando = False
        self._restantes = 0

    def setRate_(self, ritmo):
        self.reg.ritmos.append(ritmo)

    def startSpeakingString_toURL_(self, texto, url):
        self.reg.textos.append(texto)
        self.reg.archivos.append(url)
        if not self.c.get("arranca", True):
            return False
        contenido = self.c.get("contenido", b"ns")
        if contenido is not None:
            with open(url, "wb") as f:
                f.write(contenido)
        with self.reg.candado:
            self.reg.simultaneos += 1
            self.reg.max_simultaneos = max(self.reg.max_simultaneos, self.reg.simultaneos)
        self._restantes = int(self.c.get("vueltas", 1))
        self._hablando = True
        return True

    def isSpeaking(self):
        if not self._hablando:
            return False
        if self.c.get("nunca_termina"):
            return True
        if self._restantes > 0:
            self._restantes -= 1
            return True
        self._fin()
        return False

    def stopSpeaking(self):
        self.reg.stops += 1
        if self._hablando:
            self._fin()

    def _fin(self):
        self._hablando = False
        with self.reg.candado:
            self.reg.simultaneos -= 1


def nsspeech_falso(**c):
    """Sustituto de tts._importar_nsspeech: pyobjc de mentira, sin audio. Devuelve (importador, reg)."""
    reg = _RegistroNS()

    class _Alloc:
        def initWithVoice_(self, voz):
            reg.inits += 1
            reg.voces.append(voz)
            return None if c.get("init_nulo") else _SintetizadorFalso(reg, c)

    class _Clase:
        @staticmethod
        def alloc():
            return _Alloc()

    class _RunLoop:
        @staticmethod
        def currentRunLoop():
            return _RunLoop

        @staticmethod
        def runUntilDate_(_fecha):
            reg.vueltas += 1
            if c.get("espera"):
                time.sleep(c["espera"])

    class _Date:
        @staticmethod
        def dateWithTimeIntervalSinceNow_(segundos):
            return segundos

    class _URL:
        @staticmethod
        def fileURLWithPath_(ruta):
            return ruta

    def importador():
        if c.get("sin_pyobjc"):
            raise ImportError("No module named 'AppKit'")
        return _Clase, _Date, _RunLoop, _URL

    return importador, reg


class TestSintetizadorPersistente(_DoblesSay, unittest.TestCase):
    """Vía (b) de la medición del 16-sep-2026: un NSSpeechSynthesizer vivo en vez de arrancar `say`
    en cada frase. Aquí con dobles: ni pyobjc real ni audio."""

    def _modulo_ns(self, say=None, **c):
        say = say if say is not None else _CuentaSay()
        importador, reg = nsspeech_falso(**c)
        parche = mock.patch.object(tts, "_importar_nsspeech", importador)
        parche.start()
        self.addCleanup(parche.stop)
        t, motor = self._modulo_say(say, sintetizador_persistente=True)
        return t, say, reg, motor

    def _preparar(self, t, texto):
        t.prepare(texto)
        self.assertTrue(t.esperar_preparado(2.0), texto)
        return t._prepared_file

    def test_sintetiza_con_el_objeto_vivo_y_no_arranca_say(self):
        t, say, reg, _ = self._modulo_ns()
        archivo = self._preparar(t, "Gracias")
        with open(archivo, "rb") as f:
            self.assertEqual(f.read(), b"ns")
        self.assertEqual(say.textos, [], "con sintetizador persistente no se arranca `say`")
        self.assertEqual((reg.inits, reg.voces, reg.textos), (1, [PAULINA.id], ["Gracias"]))
        self.assertEqual(reg.ritmos, [150.0], "el ritmo del módulo llega al sintetizador")
        self.assertGreaterEqual(reg.vueltas, 1, "se hace correr el bucle mientras sintetiza")
        self.assertEqual(t.ultimo_sintetizador, "nsspeech")
        self.assertIs(t.estado_cache()["sintetizador_persistente"], True)
        t.cleanup()

    def test_el_mismo_objeto_sirve_para_todas_las_frases(self):
        t, _say, reg, _ = self._modulo_ns()
        for texto in ("uno", "dos", "tres"):
            self._preparar(t, texto)
        self.assertEqual((reg.inits, reg.textos), (1, ["uno", "dos", "tres"]))
        t.cleanup()

    def test_cambiar_la_voz_rehace_el_sintetizador_persistente(self):
        """Verificación independiente (16-sep-2026): el objeto vivo guarda la voz con la que nació.

        Antes de esta corrección, cambiar la voz del módulo vaciaba la cache (bien) pero el
        NSSpeechSynthesizer seguía siendo el de la voz anterior, así que el audio salía con la voz
        vieja: comprobado con `say` real —el AIFF de «Gracias» tras pasar a Mónica seguía teniendo el
        sha256 de Paulina—. El ritmo no tiene ese problema: se aplica con setRate_ en cada síntesis.
        """
        t, say, reg, _ = self._modulo_ns()
        self._preparar(t, "Gracias")
        t.voice, t.voice_name = MONICA.id, "Mónica"
        self._preparar(t, "Gracias")
        self.assertEqual(reg.voces, [PAULINA.id, MONICA.id], "el objeto se rehace con la voz nueva")
        self.assertEqual((reg.inits, reg.textos), (2, ["Gracias", "Gracias"]))
        self.assertIs(t.ultima_preparacion_desde_cache, False, "la cache tampoco reutiliza el audio")
        self.assertEqual(say.textos, [], "sigue sin arrancar `say`")
        t.cleanup()

    def test_una_sintesis_a_la_vez_aunque_haya_varios_hilos(self):
        # Sin candado, dos hilos sobre el mismo objeto dan audios truncados (medido el 16-sep-2026).
        t, _say, reg, _ = self._modulo_ns(vueltas=3, espera=0.01)
        hilos = [threading.Thread(target=t._sintetizar_offline_a_archivo,
                                  args=(f"frase {i}", os.path.join(t._temp_dir, f"h{i}.aiff")))
                 for i in range(4)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join(5)
        self.assertEqual([h.is_alive() for h in hilos], [False] * 4)
        self.assertEqual(reg.max_simultaneos, 1, "el candado serializa las síntesis")
        self.assertEqual(len(reg.textos), 4)
        t.cleanup()

    def test_si_no_arranca_cae_a_say_y_no_lo_reintenta(self):
        t, say, reg, _ = self._modulo_ns(arranca=False)
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            archivo = self._preparar(t, "Gracias")
            with open(archivo, "rb") as f:
                self.assertEqual(f.read(), b"Gracias", "el audio lo rehace `say`")
            self._preparar(t, "Hola")      # reemplaza la preparación anterior y borra su audio
        self.assertEqual(say.textos, ["Gracias", "Hola"])
        self.assertEqual(len(reg.textos), 1, "tras el primer fallo no se vuelve a intentar")
        self.assertEqual(t.ultimo_sintetizador, "say")
        self.assertIn("startSpeakingString_toURL_", t.ultimo_error_sintetizador)
        self.assertEqual(salida.getvalue().count("[AVISO TTS]"), 1, "se avisa una sola vez")
        self.assertIs(t.estado_cache()["sintetizador_persistente"], False)
        t.cleanup()

    def test_si_no_termina_dentro_del_tope_lo_para_y_usa_say(self):
        t, say, reg, _ = self._modulo_ns(nunca_termina=True, espera=0.005, contenido=None)
        with mock.patch.object(tts, "NS_TOPE_SINTESIS_SEG", 0.05), \
                contextlib.redirect_stdout(io.StringIO()):
            archivo = self._preparar(t, "Gracias")
        with open(archivo, "rb") as f:
            self.assertEqual(f.read(), b"Gracias")
        self.assertEqual((reg.stops, say.textos), (1, ["Gracias"]))
        self.assertIn("TimeoutError", t.ultimo_error_sintetizador)
        t.cleanup()

    def test_sin_pyobjc_se_usa_say_sin_romper_nada(self):
        t, say, reg, _ = self._modulo_ns(sin_pyobjc=True)
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            self._preparar(t, "Gracias")
        self.assertEqual((say.textos, reg.inits), (["Gracias"], 0))
        self.assertIn("ImportError", t.ultimo_error_sintetizador)
        self.assertIn("[AVISO TTS]", salida.getvalue())
        t.cleanup()

    def test_si_la_voz_no_sirve_se_usa_say(self):
        t, say, reg, _ = self._modulo_ns(init_nulo=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self._preparar(t, "Gracias")
        self.assertEqual((say.textos, reg.inits), (["Gracias"], 1))
        self.assertIn("initWithVoice_", t.ultimo_error_sintetizador)
        t.cleanup()

    def test_audio_vacio_se_borra_y_say_lo_rehace(self):
        t, say, reg, _ = self._modulo_ns(contenido=b"")
        with contextlib.redirect_stdout(io.StringIO()):
            archivo = self._preparar(t, "Gracias")
        with open(archivo, "rb") as f:
            self.assertEqual(f.read(), b"Gracias")
        self.assertEqual((say.textos, len(reg.textos)), (["Gracias"], 1))
        self.assertIn("no escribió audio", t.ultimo_error_sintetizador)
        t.cleanup()

    def test_fuera_del_camino_de_say_no_se_usa(self):
        t, _say, reg, motor = self._modulo_ns()
        t.ejecutor_offline = "pyttsx3"               # p. ej. Windows o Linux
        t.prepare("Gracias")
        self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual(reg.inits, 0, "el sintetizador persistente es solo del camino de macOS")
        self.assertEqual(motor.dichos, [("archivo", "Gracias")])
        t.cleanup()

    def test_apagado_por_parametro_usa_say(self):
        importador, reg = nsspeech_falso()
        with mock.patch.object(tts, "_importar_nsspeech", importador):
            say = _CuentaSay()
            t, _ = self._modulo_say(say, sintetizador_persistente=False)
            self._preparar(t, "Gracias")
        self.assertEqual((say.textos, reg.inits), (["Gracias"], 0))
        self.assertIs(t.estado_cache()["sintetizador_persistente"], False)
        t.cleanup()

    def test_la_cache_evita_incluso_el_sintetizador_persistente(self):
        t, say, reg, _ = self._modulo_ns()
        self._preparar(t, "Gracias")
        self._preparar(t, "Gracias")
        self.assertEqual((len(reg.textos), say.textos), (1, []))
        self.assertIs(t.ultima_preparacion_desde_cache, True)
        t.cleanup()



# ---------------------------------------------------------------------------
# El campo nuevo llega al registro del prototipo (JSONL)
# ---------------------------------------------------------------------------

class TestCampoDesdeCacheEnElRegistro(unittest.TestCase):
    """tts_sintesis_desde_cache: si la síntesis salió de la cache, el registro tiene que decirlo; si el
    motor no informa, la clave NO se escribe (no se afirma nada de un motor que no lo dice)."""

    @classmethod
    def setUpClass(cls):
        # Se reutilizan los dobles del prototipo (modelo, extractor y reloj falsos): sin cámara,
        # sin modelo real y sin audio.
        from tests.test_prototipo import (CLAVES_ENSAYO_ANTES, CLAVES_TURNO_ANTES, CLAVES_VOZ,
                                          TTSFalsoOffline, construir, correr_ventana)
        cls.construir, cls.correr_ventana, cls.TTSFalsoOffline = (
            staticmethod(construir), staticmethod(correr_ventana), TTSFalsoOffline)
        cls.CLAVES_ENSAYO_ANTES, cls.CLAVES_TURNO_ANTES = CLAVES_ENSAYO_ANTES, CLAVES_TURNO_ANTES
        cls.CLAVES_VOZ = CLAVES_VOZ

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="lsc_test_campo_")
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def _motor(self, valor, con_precalentar=False):
        base = self.TTSFalsoOffline

        class Motor(base):
            def __init__(self):
                super().__init__()
                self.ultima_preparacion_desde_cache = valor
                self.precalentadas: list = []

            if con_precalentar:
                def precalentar(self, frases, bloquear=False, tope_seg=None):
                    self.precalentadas = list(frases)
                    return True

        return Motor()

    def test_la_cache_se_declara_en_el_ensayo_y_en_el_turno(self):
        for valor in (True, False):
            with self.subTest(valor=valor):
                tr = self.construir(self.tmp, tts=self._motor(valor))
                reg = self.correr_ventana(tr)
                self.assertIs(reg["tts_sintesis_desde_cache"], valor)
                json.dumps(reg)
                tr2 = self.construir(self.tmp, tts=self._motor(valor), modo_oracion=True)
                self.correr_ventana(tr2)
                turno = tr2.fin_de_turno()
                self.assertIs(turno["tts_sintesis_desde_cache"], valor)
                json.dumps(turno)

    def test_un_motor_que_no_informa_no_escribe_la_clave(self):
        for motor in (self.TTSFalsoOffline(), self._motor("sí"), self._motor(None)):
            with self.subTest(motor=type(motor).__name__):
                tr = self.construir(self.tmp, tts=motor)
                reg = self.correr_ventana(tr)
                self.assertNotIn("tts_sintesis_desde_cache", reg)

    def test_la_clave_nueva_no_desordena_las_anteriores(self):
        # El orden que fija tests/test_prototipo.py (N-37) se mantiene: la clave nueva va DESPUÉS de
        # los dos contadores de voz en el ensayo y ANTES de ellos en el turno.
        tr = self.construir(self.tmp, tts=self._motor(True), modo_oracion=False)
        reg = self.correr_ventana(tr)
        # «bordes» (aviso de N-50, opción B, 22-nov-2026) va siempre al final del registro de ensayo.
        claves = [k for k in reg if k not in self.CLAVES_VOZ and k not in ("tts_sintesis_desde_cache", "bordes")]
        self.assertEqual(claves, self.CLAVES_ENSAYO_ANTES)
        self.assertEqual(list(reg)[-1], "bordes")
        i = list(reg).index("tts_ocupado_en_t5")
        self.assertEqual(list(reg)[i + 1:i + 4], self.CLAVES_VOZ + ["tts_sintesis_desde_cache"])
        tr2 = self.construir(self.tmp, tts=self._motor(True), modo_oracion=True)
        self.correr_ventana(tr2)
        turno = tr2.fin_de_turno()
        claves = [k for k in turno if k not in self.CLAVES_VOZ and k != "tts_sintesis_desde_cache"]
        self.assertEqual(claves, self.CLAVES_TURNO_ANTES)
        self.assertEqual(list(turno)[-3:], ["tts_sintesis_desde_cache"] + self.CLAVES_VOZ)

    def test_leer_la_cache_no_mueve_las_marcas(self):
        reg_sin = self.correr_ventana(self.construir(self.tmp, tts=self.TTSFalsoOffline()))
        reg_con = self.correr_ventana(self.construir(self.tmp, tts=self._motor(True)))
        marcas = ("t0", "t1", "t2", "t3", "t4", "t5", "L_total_ns", "L_resp_ns", "etapas_ns")
        self.assertEqual([reg_con[k] for k in marcas], [reg_sin[k] for k in marcas])

    def test_el_prototipo_precalienta_con_las_50_glosas_convertidas(self):
        from src.grammar_mapper import SIGN_VOCABULARY
        motor = self._motor(False, con_precalentar=True)
        tr = self.construir(self.tmp, tts=motor)
        self.assertTrue(tr.tts_precalentando)
        self.assertEqual(len(motor.precalentadas), len(SIGN_VOCABULARY))
        self.assertEqual(motor.precalentadas[:3], ["Gracias", "Buenos días", "Buenas tardes"])
        self.assertNotIn("gracias", motor.precalentadas, "se precalienta el enunciado, no la glosa")
        # Y se puede apagar sin tocar nada más.
        otro = self._motor(False, con_precalentar=True)
        tr2 = self.construir(self.tmp, tts=otro, tts_precalentar=False)
        self.assertFalse(tr2.tts_precalentando)
        self.assertEqual(otro.precalentadas, [])

    def test_no_se_precalienta_lo_que_nunca_se_va_a_consultar(self):
        # Un motor que no puede preparar a archivo habla con speak_async: su cache no se consultaría.
        motor = self._motor(False, con_precalentar=True)
        motor.soporta_preparar = False
        motor.mode = "offline"
        tr = self.construir(self.tmp, tts=motor)
        self.assertFalse(tr.tts_precalentando)
        self.assertEqual(motor.precalentadas, [])

    def test_un_motor_sin_precalentar_no_estorba(self):
        tr = self.construir(self.tmp, tts=self.TTSFalsoOffline())
        self.assertFalse(tr.tts_precalentando)
        self.assertIsNotNone(self.correr_ventana(tr))


if __name__ == "__main__":
    unittest.main()
