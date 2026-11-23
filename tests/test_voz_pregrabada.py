"""
Pruebas de la voz pregrabada (decisión del autor del 23-nov-2026; docs/VOZ.md): src/voz_pregrabada.py y el modo
«pregrabada» de src/tts.py, SIN reproducir audio, sin la voz del equipo y sin PyTorch.

Cubren:
  1. el índice real (models/voz_pregrabada/indice.json): válido, 142 WAV mono de 44,1 kHz con su sha256, la
     procedencia (ElevenLabs «Robert - Colombian, Paisa»), el método de corte y la revisión a oído;
  2. la cobertura: los textos del índice son EXACTAMENTE los que el conversor en modo inventario puede producir,
     enumerados desde el código (100 %), y las variantes en femenino quedan fuera (van al respaldo);
  3. la validación detecta índices alterados y la búsqueda es exacta;
  4. la cadena de respaldo pregrabada → xtts → local con un servicio XTTS falso (objeto en memoria) y una voz
     local falsa (pyttsx3 y `say` sustituidos), la cache solo de XTTS y que nunca se borra un WAV del índice;
  5. que el registro JSONL del prototipo dice qué fuente se usó y cuánto tardó, sin desordenar las claves;
  6. los predeterminados de realtime.py y que el proyecto no importa torch.
El servicio XTTS como proceso (protocolo, caídas, topes) se prueba en tests/test_voz_xtts_servicio.py.
    python -m unittest tests/test_voz_pregrabada.py -v
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import tts as tts_mod  # noqa: E402
from src import voz_pregrabada as vp  # noqa: E402
from src.conversor_glosas import INVENTARIO, ConversorGlosas  # noqa: E402
from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402

INDICE = RAIZ / "models" / "voz_pregrabada" / "indice.json"
MONICA = SimpleNamespace(id="com.apple.voice.compact.es-ES.Monica", name="Mónica (Spanish (Spain))",
                         languages=["es_ES"])
VARIANTES_FEMENINO = {"Hola, bienvenida.", "Buenos días, bienvenida.", "Tú eres bienvenida.",
                      "Yo me siento bienvenida."}


def escribir_wav(ruta: Path, segundos: float = 0.1, sr: int = 44100) -> None:
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x01\x00" * int(segundos * sr))


# ---------------------------------------------------------------------------
# Dobles (también los usa tests/test_voz_xtts_servicio.py)
# ---------------------------------------------------------------------------

class MotorPyttsx3Falso:
    driver_name = "falso"

    def __init__(self, voces):
        self._voces, self.propiedades = voces, {}
        self.proxy = SimpleNamespace(setBusy=lambda v: None)

    def getProperty(self, nombre):
        return self._voces if nombre == "voices" else self.propiedades.get(nombre)

    def setProperty(self, nombre, valor):
        self.propiedades[nombre] = valor


class ServicioFalso:
    """Imita ServicioXTTS en memoria: estados, sintetizar a WAV, fallos y caída a demanda."""

    def __init__(self, estado="listo", falla=False, cae=False):
        self.estado_actual = estado
        self.falla, self.cae = falla, cae
        self.pedidos: list[str] = []
        self.detenido = False
        self.tope_arranque_seg = 0.2

    disponible = property(lambda self: self.estado_actual == "listo")
    caido = property(lambda self: self.estado_actual == "caido")
    cargando = property(lambda self: self.estado_actual == "cargando")

    def esperar_listo(self, tope=None):
        return self.disponible

    def sintetizar(self, texto, salida, tope_seg=None):
        from src.voz_xtts_servicio import ErrorServicioVoz
        self.pedidos.append(texto)
        if self.cae:
            self.estado_actual = "caido"
            raise ErrorServicioVoz("el proceso terminó (código 3)")
        if self.falla:
            raise ErrorServicioVoz("XTTS no pudo sintetizar: fallo simulado")
        escribir_wav(Path(salida), 0.05, 24000)
        return {"ok": True, "ms": 1.0}

    def estado(self):
        return {"estado": self.estado_actual, "peticiones": len(self.pedidos)}

    def detener(self, tope_seg=2.0):
        self.detenido = True
        self.estado_actual = "caido"


class DoblesVozLocal:
    """Parches para construir TTSModule sin voz real: pyttsx3 falso con Mónica, `say` y `afplay` sustituidos
    por un subprocess.run que registra las órdenes (say escribe un AIFF falso; afplay no suena)."""

    def poner_dobles(self, local_ok: bool = True):
        self.ordenes: list[list[str]] = []
        run_real = subprocess.run

        def run_falso(orden, **kw):
            if not orden or orden[0] not in ("say", "afplay"):
                return run_real(orden, **kw)          # p. ej. platform.processor(): no es de la voz
            self.ordenes.append(list(orden))
            if orden[0] == "say" and "-o" in orden:
                with open(orden[orden.index("-o") + 1], "wb") as f:
                    f.write(b"FORM-falso")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        modulo = SimpleNamespace(init=lambda: MotorPyttsx3Falso([MONICA]))
        if local_ok:
            imp = mock.patch.object(tts_mod, "_importar_pyttsx3", return_value=modulo)
        else:
            imp = mock.patch.object(tts_mod, "_importar_pyttsx3",
                                    side_effect=tts_mod.TTSDependenciaFaltante("sin pyttsx3"))
        parches = [imp,
                   mock.patch.object(tts_mod, "ejecutor_offline_por_defecto", return_value="say"),
                   mock.patch.object(tts_mod, "nombres_voces_say", return_value={MONICA.name}),
                   mock.patch.object(tts_mod.subprocess, "run", side_effect=run_falso),
                   mock.patch.object(tts_mod.shutil, "which", side_effect=lambda p: f"/usr/bin/{p}")]
        for p in parches:
            p.start()
            self.addCleanup(p.stop)

    def construir_pregrabada(self, servicio=None, **kw):
        kw.setdefault("sintetizador_persistente", False)
        kw.setdefault("python_voz", "")
        with contextlib.redirect_stdout(io.StringIO()):
            t = tts_mod.TTSModule(mode="pregrabada", verbose=False, servicio_xtts=servicio, **kw)
        self.addCleanup(t.cleanup)
        return t

    def says(self):
        return [o for o in self.ordenes if o[0] == "say"]

    @staticmethod
    def esperar(condicion, tope=3.0):
        limite = time.monotonic() + tope
        while not condicion() and time.monotonic() < limite:
            time.sleep(0.005)
        return condicion()


# ---------------------------------------------------------------------------
# 1-2. Índice real y cobertura
# ---------------------------------------------------------------------------

class TestIndiceReal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datos = json.loads(INDICE.read_text(encoding="utf-8"))
        cls.indice = vp.IndiceVozPregrabada.cargar(INDICE)

    def test_indice_valido_con_142_wav_mono_44100(self):
        self.assertEqual(vp.validar_indice(self.datos, INDICE.parent), [])
        self.assertEqual(len(self.indice), 142)
        frases = self.datos["frases"]
        self.assertEqual([f["numero"] for f in frases], list(range(1, 143)))
        self.assertEqual([f["archivo"] for f in frases], [f"{n:03d}.wav" for n in range(1, 143)])
        total = 0
        for f in frases:
            info = vp.info_wav(INDICE.parent / f["archivo"])
            self.assertEqual((info["canales"], info["frecuencia_hz"], info["bits"]), (1, 44100, 16), f["archivo"])
            self.assertAlmostEqual(f["duracion_s"], info["duracion_s"], places=2)
            self.assertGreater(f["duracion_s"], 0.3)
            self.assertLess(f["duracion_s"], 4.0)
            self.assertEqual(f["sha256_texto"], vp.sha256_texto(f["texto"]))
            self.assertEqual(f["sha256_audio"], vp.sha256_archivo(INDICE.parent / f["archivo"]))
            total += (INDICE.parent / f["archivo"]).stat().st_size
        self.assertEqual(self.datos["audio"]["bytes_total"], total)
        # Ningún WAV sobrante en la carpeta.
        self.assertEqual(sorted(p.name for p in INDICE.parent.glob("*.wav")), [f["archivo"] for f in frases])

    def test_procedencia_metodo_y_revision_a_oido(self):
        fuente = self.datos["fuente"]
        self.assertEqual((fuente["proveedor"], fuente["voz"]), ("ElevenLabs", "Robert - Colombian, Paisa"))
        self.assertEqual(fuente["fecha"], "2026-11-23")
        self.assertIn("el autor", fuente["generado_por"])
        self.assertIn("Robert - Colombian, Paisa", fuente["atribucion"])
        self.assertIn("faster-whisper", self.datos["corte"]["herramienta"])
        revisados = [f["numero"] for f in self.datos["frases"] if f["revisado_a_oido"]]
        self.assertEqual(revisados, [5, 15, 16, 17, 20, 23, 26, 35, 40, 41, 87, 88, 93, 98, 107, 110])
        self.assertEqual(self.datos["n_revisados_a_oido"], 16)
        rehechos = [f["numero"] for f in self.datos["frases"] if f["corte_rehecho"]]
        self.assertEqual(rehechos, [26, 88])
        for n in (26, 88):
            f = self.datos["frases"][n - 1]
            self.assertTrue(f["corte"]["corregido_a_mano"])
            self.assertIn("rehecho", f["verificacion_whisper"]["nota"])
        # Todos tienen su verificación automática.
        self.assertTrue(all("verificacion_whisper" in f for f in self.datos["frases"]))

    def test_sin_rutas_de_la_maquina_en_el_indice(self):
        texto = INDICE.read_text(encoding="utf-8")
        for prohibido in ("/Users/", "/private/", "/tmp/", "." + "cla" + "ude"):   # rutas de una máquina
            self.assertNotIn(prohibido, texto)


class TestCobertura(unittest.TestCase):

    def test_textos_enumerados_desde_el_codigo(self):
        # Enumeración independiente: 50 glosas solas + 92 secuencias con el conversor estricto.
        conv = ConversorGlosas(estricto=True)
        esperados = [conv.translate([SIGN_VOCABULARY[i]]) for i in sorted(SIGN_VOCABULARY)]
        esperados += [conv.translate(list(s["glosas"])) for s in INVENTARIO]
        self.assertEqual(len(INVENTARIO), 92)
        self.assertEqual(len(esperados), 142)
        self.assertEqual(len(set(esperados)), 142)
        self.assertEqual(vp.textos_inventario(), [unicodedata.normalize("NFC", t) for t in esperados])
        for g in SIGN_VOCABULARY.values():
            self.assertFalse(conv.explicar([g])["respaldo"])

    def test_cobertura_del_100_por_ciento_y_en_el_mismo_orden(self):
        indice = vp.IndiceVozPregrabada.cargar(INDICE)
        c = vp.cobertura(indice, vp.textos_inventario())
        self.assertEqual((c["n"], c["cubiertos"], c["porcentaje"], c["faltan"], c["sobran"]),
                         (142, 142, 100.0, [], []))
        self.assertEqual(indice.textos, vp.textos_inventario())
        # Con géneros explícitos «m» la salida es la misma.
        self.assertEqual(vp.textos_inventario("m", "m"), vp.textos_inventario())

    def test_variantes_en_femenino_no_estan_pregrabadas(self):
        indice = vp.IndiceVozPregrabada.cargar(INDICE)
        variantes = vp.variantes_de_genero()
        todas = set().union(*map(set, variantes.values()))
        self.assertEqual(todas, VARIANTES_FEMENINO)
        self.assertFalse(any(t in indice for t in todas))
        self.assertEqual(set(variantes["yo=f, tú=m"]), {"Yo me siento bienvenida."})

    def test_cli_comprobar(self):
        r = vp.comprobar(INDICE)
        self.assertEqual(r["cobertura_generos_por_defecto"]["porcentaje"], 100.0)
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            self.assertEqual(vp.main(["--indice", str(INDICE)]), 0)
        self.assertIn("142/142", salida.getvalue())


# ---------------------------------------------------------------------------
# 3. Validación y búsqueda exacta
# ---------------------------------------------------------------------------

class TestValidacionYBusqueda(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lsc_voz_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        escribir_wav(self.tmp / "001.wav")
        escribir_wav(self.tmp / "002.wav", 0.2)
        self.datos = {"version": 1, "frases": [
            {"numero": 1, "texto": "Gracias.", "sha256_texto": vp.sha256_texto("Gracias."), "archivo": "001.wav",
             "duracion_s": 0.1},
            {"numero": 2, "texto": "Buenos días.", "sha256_texto": vp.sha256_texto("Buenos días."),
             "archivo": "002.wav", "duracion_s": 0.2}]}

    def guardar(self, datos):
        (self.tmp / "indice.json").write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
        return self.tmp / "indice.json"

    def test_valido(self):
        indice = vp.IndiceVozPregrabada.cargar(self.guardar(self.datos))
        self.assertEqual(len(indice), 2)
        self.assertEqual(vp.IndiceVozPregrabada.cargar(self.tmp).textos, ["Gracias.", "Buenos días."])

    def test_detecta_alteraciones(self):
        casos = {
            "sha256_texto": lambda d: d["frases"][0].update(texto="Gracias"),
            "repetido": lambda d: d["frases"][1].update(texto="Gracias.", sha256_texto=vp.sha256_texto("Gracias.")),
            "falta": lambda d: d["frases"][1].update(archivo="009.wav"),
            "consecutivos": lambda d: d["frases"][1].update(numero=3),
            "duracion_s": lambda d: d["frases"][0].update(duracion_s=0.5),
            "versión": lambda d: d.update(version=2),
            "carpeta": lambda d: d["frases"][0].update(archivo="../001.wav"),
            "sha256_audio": lambda d: d["frases"][0].update(sha256_audio="0" * 64),
        }
        for clave, alterar in casos.items():
            with self.subTest(clave=clave):
                datos = json.loads(json.dumps(self.datos))
                alterar(datos)
                errores = vp.validar_indice(datos, self.tmp)
                self.assertTrue(any(clave in e for e in errores), errores)
                with self.assertRaises(vp.IndiceInvalido):
                    vp.IndiceVozPregrabada.cargar(self.guardar(datos))

    def test_wav_estereo_o_roto(self):
        with wave.open(str(self.tmp / "002.wav"), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(b"\x00\x00" * 2 * 8820)
        self.assertTrue(any("mono" in e for e in vp.validar_indice(self.datos, self.tmp)))
        (self.tmp / "001.wav").write_bytes(b"no es un wav")
        self.assertTrue(any("WAV legible" in e for e in vp.validar_indice(self.datos, self.tmp)))

    def test_indice_inexistente_o_ilegible(self):
        with self.assertRaises(vp.IndiceInvalido):
            vp.IndiceVozPregrabada.cargar(self.tmp / "no.json")
        (self.tmp / "indice.json").write_text("{roto", encoding="utf-8")
        with self.assertRaises(vp.IndiceInvalido):
            vp.IndiceVozPregrabada.cargar(self.tmp)
        self.assertFalse(vp.existe_indice(self.tmp / "otra"))
        self.assertTrue(vp.existe_indice(self.tmp))

    def test_busqueda_exacta(self):
        indice = vp.IndiceVozPregrabada.cargar(self.guardar(self.datos))
        self.assertEqual(indice.buscar("Gracias."), self.tmp / "001.wav")
        # NFD y NFC son el mismo texto: se encuentra.
        self.assertEqual(indice.buscar(unicodedata.normalize("NFD", "Buenos días.")), self.tmp / "002.wav")
        for otro in ("Gracias", "gracias.", " Gracias.", "Gracias. ", "GRACIAS.", "Buenos dias.", "", None, 3,
                     ["Gracias."]):
            with self.subTest(otro=otro):
                self.assertIsNone(indice.buscar(otro))
                self.assertNotIn(otro, indice)

    def test_busqueda_en_el_indice_real(self):
        indice = vp.IndiceVozPregrabada.cargar(INDICE)
        self.assertEqual(indice.buscar("Gracias.").name, "001.wav")
        self.assertEqual(indice.buscar("¿Dónde vive el hermano?").name, "058.wav")   # S-08 → 50 + 8
        self.assertIsNone(indice.buscar("Gracias"))                 # lo que daría GrammarMapper
        self.assertIsNone(indice.buscar("hermano, tío, comer"))     # un respaldo del conversor


# ---------------------------------------------------------------------------
# 4. Cadena de respaldo en src/tts.py
# ---------------------------------------------------------------------------

class TestCadenaDeRespaldo(DoblesVozLocal, unittest.TestCase):

    def setUp(self):
        self.poner_dobles()

    def test_pregrabada_no_sintetiza_ni_borra_el_wav_del_indice(self):
        xtts = ServicioFalso()
        t = self.construir_pregrabada(xtts)
        self.assertEqual(t.mode, "pregrabada")
        self.assertTrue(t.soporta_preparar)
        self.assertEqual(t.voice_name, MONICA.name)              # voz local de respaldo
        t.prepare("Gracias.")
        self.assertTrue(t.is_prepared, "la preparación pregrabada queda lista antes de retornar")
        self.assertTrue(t.esperar_preparado(0.0))
        self.assertEqual(t.ultima_fuente, "pregrabada")
        self.assertIsInstance(t.ultima_sintesis_ns, int)
        self.assertIsNone(t.ultima_preparacion_desde_cache)
        archivo = INDICE.parent / "001.wav"
        self.assertEqual(Path(t._prepared_file), archivo)
        t.play_prepared_async()
        self.assertTrue(self.esperar(lambda: any(o[0] == "afplay" for o in self.ordenes)))
        self.assertEqual([o for o in self.ordenes if o[0] == "afplay"], [["afplay", str(archivo)]])
        time.sleep(0.05)
        self.assertTrue(archivo.exists(), "el WAV del índice nunca se borra")
        self.assertEqual(self.says(), [])
        self.assertEqual(xtts.pedidos, [])
        # Reemplazar una preparación pregrabada sin reproducirla tampoco la borra.
        t.prepare("Buenos días.")
        t.prepare("Gracias.")
        self.assertTrue((INDICE.parent / "002.wav").exists())
        self.assertEqual(t.estado_voz()["fuentes"]["pregrabada"], 3)

    def test_no_pregrabada_va_a_xtts_y_la_segunda_vez_sale_de_la_cache(self):
        xtts = ServicioFalso()
        t = self.construir_pregrabada(xtts)
        t.prepare("hermano, tío, comer")
        self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual((t.ultima_fuente, t.ultima_preparacion_desde_cache), ("xtts", False))
        self.assertTrue(t._prepared_file.endswith(".wav"))
        self.assertEqual(xtts.pedidos, ["hermano, tío, comer"])
        t.prepare("hermano, tío, comer")
        self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual((t.ultima_fuente, t.ultima_preparacion_desde_cache), ("xtts", True))
        self.assertEqual(xtts.pedidos, ["hermano, tío, comer"], "la cache evita un segundo pedido")
        self.assertEqual(self.says(), [])

    def test_xtts_cargando_usa_la_voz_local_sin_guardarla_y_luego_xtts(self):
        xtts = ServicioFalso(estado="cargando")
        t = self.construir_pregrabada(xtts)
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            t.prepare("yo, sentir")
            self.assertTrue(t.esperar_preparado(2.0))
            t.prepare("tú, comer")
            self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual(t.ultima_fuente, "local")
        self.assertTrue(t._prepared_file.endswith(".aiff"))
        self.assertEqual(len(self.says()), 2)
        self.assertEqual(self.says()[0][:6], ["say", "-r", "150", "-v", MONICA.name, "-o"])
        self.assertEqual(salida.getvalue().count("aún está cargando"), 1, "se avisa una vez")
        self.assertEqual(t.estado_cache()["entradas"], 0, "lo local no se guarda en la cache")
        xtts.estado_actual = "listo"
        t.prepare("yo, sentir")
        self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual(t.ultima_fuente, "xtts")

    def test_xtts_que_falla_una_frase_cae_a_local_solo_para_esa_frase(self):
        xtts = ServicioFalso(falla=True)
        t = self.construir_pregrabada(xtts)
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            t.prepare("hola, tú")
            self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual(t.ultima_fuente, "local")
        self.assertIn("no pudo decir", salida.getvalue())
        self.assertTrue(xtts.disponible)
        xtts.falla = False
        t.prepare("hola, tú")
        self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual(t.ultima_fuente, "xtts")

    def test_xtts_que_cae_deja_la_sesion_con_la_voz_local(self):
        xtts = ServicioFalso(cae=True)
        t = self.construir_pregrabada(xtts)
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            for frase in ("hola, tú", "yo, comer", "Gracias."):
                t.prepare(frase)
                self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual(xtts.pedidos, ["hola, tú"], "caído, no se le vuelve a pedir")
        self.assertEqual(t.ultima_fuente, "pregrabada", "lo pregrabado sigue sonando")
        self.assertEqual(t.estado_voz()["fuentes"], {"local": 2, "pregrabada": 1})
        self.assertEqual(salida.getvalue().count("no está disponible"), 1)

    def test_sin_servicio_y_sin_voz_local_la_frase_no_se_prepara_pero_lo_pregrabado_si(self):
        self.poner_dobles(local_ok=False)
        with contextlib.redirect_stdout(io.StringIO()):
            t = self.construir_pregrabada(None)
            self.assertFalse(t.local_disponible)
            t.prepare("hola, tú")
            self.assertFalse(t.esperar_preparado(2.0))
        self.assertIn("sin voz local", t.error_preparacion)
        self.assertIsNone(t.ultima_fuente)
        t.prepare("Hola.")
        self.assertTrue(t.esperar_preparado(0.0))
        self.assertEqual(t.ultima_fuente, "pregrabada")

    def test_fuente_desconocida_se_borra_al_preparar_otra(self):
        # ultima_fuente describe la preparación VIGENTE: mientras la nueva no está lista vale None.
        xtts = ServicioFalso()
        t = self.construir_pregrabada(xtts)
        t.prepare("Gracias.")
        evento = threading.Event()
        original = xtts.sintetizar
        xtts.sintetizar = lambda texto, salida, tope_seg=None: (evento.wait(2.0), original(texto, salida))[1]
        t.prepare("hola, tú")
        self.assertIsNone(t.ultima_fuente)
        self.assertIsNone(t.ultima_sintesis_ns)
        evento.set()
        self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual(t.ultima_fuente, "xtts")

    def test_speak_sigue_la_misma_cadena(self):
        xtts = ServicioFalso(estado="caido")
        t = self.construir_pregrabada(xtts)
        t.speak("Gracias.")
        t.speak("hola, tú")
        afplay = [o for o in self.ordenes if o[0] == "afplay"]
        self.assertEqual(afplay[0], ["afplay", str(INDICE.parent / "001.wav")])
        self.assertTrue(afplay[1][1].endswith(".aiff"))
        self.assertFalse(os.path.exists(afplay[1][1]), "el audio local temporal se borra")
        self.assertTrue((INDICE.parent / "001.wav").exists())
        self.assertEqual(t.estado_voz()["fuentes"], {"pregrabada": 1, "local": 1})

    def test_precalentar_solo_lo_no_pregrabado_y_solo_con_xtts(self):
        xtts = ServicioFalso()
        t = self.construir_pregrabada(xtts)
        self.assertTrue(t.precalentar(["Gracias.", "Gracias", "hola, tú", "Buenos días."], bloquear=True, tope_seg=3))
        self.assertEqual(xtts.pedidos, ["Gracias", "hola, tú"])
        self.assertEqual(t.estado_cache()["entradas"], 2)
        # Todo pregrabado: no hay nada que precalentar.
        self.assertFalse(t.precalentar(["Gracias.", "Hola."]))
        # Sin servicio (o caído) no se precalienta con la voz local.
        t2 = self.construir_pregrabada(None)
        self.assertFalse(t2.precalentar(["hola, tú"]))
        t3 = self.construir_pregrabada(ServicioFalso(estado="caido"))
        self.assertFalse(t3.precalentar(["hola, tú"]))
        self.assertEqual(self.says(), [])

    def test_cleanup_detiene_el_servicio(self):
        xtts = ServicioFalso()
        t = self.construir_pregrabada(xtts)
        t.cleanup()
        self.assertTrue(xtts.detenido)

    def test_sin_indice_es_dependencia_faltante(self):
        with self.assertRaises(tts_mod.TTSDependenciaFaltante):
            tts_mod.TTSModule(mode="pregrabada", verbose=False, indice_pregrabada="/no/existe/indice.json",
                              python_voz="")

    def test_python_voz_desde_el_entorno_y_vacio_lo_desactiva(self):
        from src import voz_xtts_servicio as sx
        lanzados = []

        class ClienteFalso(ServicioFalso):
            def __init__(self, python, **kw):
                super().__init__(estado="cargando")
                lanzados.append((python, kw))

            def iniciar(self):
                pass

        with mock.patch.object(sx, "ServicioXTTS", ClienteFalso), \
                mock.patch.dict(os.environ, {"TESIS_PYTHON_VOZ": "/opt/voz/python"}):
            t = self.construir_pregrabada(None, python_voz=None)
            self.assertEqual(lanzados[0][0], "/opt/voz/python")
            self.assertEqual(lanzados[0][1]["hablante"], "Gilberto Mathias")
            self.assertEqual(lanzados[0][1]["dispositivo"], "auto")
            self.assertIsInstance(t.xtts, ClienteFalso)
            t2 = self.construir_pregrabada(None, python_voz="")
            self.assertIsNone(t2.xtts)
        self.assertEqual(len(lanzados), 1)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(self.construir_pregrabada(None, python_voz=None).xtts)


class TestModoOfflineInformaLaFuente(DoblesVozLocal, unittest.TestCase):

    def setUp(self):
        self.poner_dobles()

    def test_offline_dice_local_y_la_cache_tambien(self):
        with contextlib.redirect_stdout(io.StringIO()):
            t = tts_mod.TTSModule(mode="offline", verbose=False, sintetizador_persistente=False)
        self.addCleanup(t.cleanup)
        self.assertEqual(t.voice_name, MONICA.name)
        t.prepare("Hola")
        self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual((t.ultima_fuente, t.ultima_preparacion_desde_cache), ("local", False))
        t.prepare("Hola")
        self.assertTrue(t.esperar_preparado(2.0))
        self.assertEqual((t.ultima_fuente, t.ultima_preparacion_desde_cache), ("local", True))


# ---------------------------------------------------------------------------
# 5. Registro JSONL del prototipo
# ---------------------------------------------------------------------------

class TestRegistroDiceLaFuente(DoblesVozLocal, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from tests.test_prototipo import (CLAVES_ENSAYO_ANTES, CLAVES_TURNO_ANTES, CLAVES_VOZ, TTSFalsoOffline,
                                          construir, correr_ventana)
        cls.construir_tr, cls.correr_ventana = staticmethod(construir), staticmethod(correr_ventana)
        cls.TTSFalsoOffline = TTSFalsoOffline
        cls.CLAVES_ENSAYO_ANTES, cls.CLAVES_TURNO_ANTES, cls.CLAVES_VOZ = (CLAVES_ENSAYO_ANTES, CLAVES_TURNO_ANTES,
                                                                           CLAVES_VOZ)

    def setUp(self):
        self.poner_dobles()
        self.tmp = tempfile.mkdtemp(prefix="lsc_voz_reg_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _motor(self, fuente, ns=1234, desde_cache=None):
        base = self.TTSFalsoOffline

        class Motor(base):
            def __init__(self):
                super().__init__()
                self.ultima_fuente, self.ultima_sintesis_ns = fuente, ns
                self.ultima_preparacion_desde_cache = desde_cache
        return Motor()

    def test_con_el_motor_real_en_pregrabada(self):
        xtts = ServicioFalso()
        t = self.construir_pregrabada(xtts)
        t._reproducir = lambda archivo, texto: None          # sin audio
        tr = self.construir_tr(self.tmp, tts=t, grammar_mapper=ConversorGlosas(estricto=True))
        reg = self.correr_ventana(tr)
        self.assertEqual(reg["estado"], "aceptada")
        self.assertEqual(reg["tts_modo"], "pregrabada")
        self.assertEqual(reg["tts_fuente"], "pregrabada")
        self.assertIsInstance(reg["tts_tiempo_sintesis_ns"], int)
        self.assertLess(reg["tts_tiempo_sintesis_ns"], 50_000_000)
        self.assertNotIn("tts_sintesis_desde_cache", reg)
        self.assertIs(reg["tts_sintesis_antes_de_t5"], True)
        # Turno con una secuencia fuera del inventario: respaldo del conversor → XTTS.
        tr2 = self.construir_tr(self.tmp, tts=t, grammar_mapper=ConversorGlosas(estricto=True), modo_oracion=True)
        tr2.glosas_turno = ["hermano", "tío", "comer"]
        turno = tr2.fin_de_turno()
        self.assertEqual(turno["enunciado"], "hermano, tío, comer")
        self.assertEqual(turno["tts_fuente"], "xtts")
        self.assertIsInstance(turno["tts_tiempo_sintesis_ns"], int)
        json.dumps(reg), json.dumps(turno)
        # El JSONL escrito también lo lleva.
        lineas = [json.loads(x) for x in Path(tr.registro_path).read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(r.get("tts_fuente") == "pregrabada" for r in lineas))

    def test_tres_fuentes_en_el_registro(self):
        for fuente in ("pregrabada", "xtts", "local"):
            with self.subTest(fuente=fuente):
                reg = self.correr_ventana(self.construir_tr(self.tmp, tts=self._motor(fuente, 77)))
                self.assertEqual((reg["tts_fuente"], reg["tts_tiempo_sintesis_ns"]), (fuente, 77))

    def test_un_motor_que_no_informa_no_escribe_las_claves(self):
        for motor in (self.TTSFalsoOffline(), self._motor("otra"), self._motor(None), self._motor(3),
                      self._motor(mock.MagicMock())):
            reg = self.correr_ventana(self.construir_tr(self.tmp, tts=motor))
            self.assertNotIn("tts_fuente", reg)
            self.assertNotIn("tts_tiempo_sintesis_ns", reg)
        reg = self.correr_ventana(self.construir_tr(self.tmp, tts=self._motor("xtts", ns=-1)))
        self.assertEqual(reg["tts_fuente"], "xtts")
        self.assertNotIn("tts_tiempo_sintesis_ns", reg, "un tiempo inválido no se escribe")
        reg = self.correr_ventana(self.construir_tr(self.tmp, tts=self._motor("xtts", ns=True)))
        self.assertNotIn("tts_tiempo_sintesis_ns", reg)

    def test_las_claves_nuevas_no_desordenan_las_anteriores(self):
        reg = self.correr_ventana(self.construir_tr(self.tmp, tts=self._motor("xtts", 5, desde_cache=True)))
        nuevas = ("tts_sintesis_desde_cache", "tts_fuente", "tts_tiempo_sintesis_ns", "bordes")
        self.assertEqual([k for k in reg if k not in self.CLAVES_VOZ and k not in nuevas], self.CLAVES_ENSAYO_ANTES)
        i = list(reg).index("tts_ocupado_en_t5")
        self.assertEqual(list(reg)[i + 1:i + 6], self.CLAVES_VOZ + ["tts_sintesis_desde_cache", "tts_fuente",
                                                                    "tts_tiempo_sintesis_ns"])
        tr = self.construir_tr(self.tmp, tts=self._motor("local", 5, desde_cache=False), modo_oracion=True)
        self.correr_ventana(tr)
        turno = tr.fin_de_turno()
        self.assertEqual([k for k in turno if k not in self.CLAVES_VOZ and k not in nuevas], self.CLAVES_TURNO_ANTES)
        self.assertEqual(list(turno)[-5:], ["tts_sintesis_desde_cache", "tts_fuente", "tts_tiempo_sintesis_ns"]
                         + self.CLAVES_VOZ)

    def test_leer_la_fuente_no_mueve_las_marcas(self):
        reg_sin = self.correr_ventana(self.construir_tr(self.tmp, tts=self.TTSFalsoOffline()))
        reg_con = self.correr_ventana(self.construir_tr(self.tmp, tts=self._motor("pregrabada")))
        marcas = ("t0", "t1", "t2", "t3", "t4", "t5", "L_total_ns", "L_resp_ns", "etapas_ns")
        self.assertEqual([reg_con[k] for k in marcas], [reg_sin[k] for k in marcas])

    def test_el_prototipo_con_inventario_no_precalienta_nada_en_pregrabada(self):
        xtts = ServicioFalso()
        t = self.construir_pregrabada(xtts)
        tr = self.construir_tr(self.tmp, tts=t, grammar_mapper=ConversorGlosas(estricto=True))
        self.assertFalse(tr.tts_precalentando, "las 50 glosas solas ya están pregrabadas")
        self.assertEqual(xtts.pedidos, [])


# ---------------------------------------------------------------------------
# 6. Predeterminados de realtime.py y aislamiento de torch
# ---------------------------------------------------------------------------

class TestPredeterminadosRealtime(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src import realtime              # importa cv2; no se abre la cámara
        cls.rt = realtime

    def test_opciones(self):
        args = self.rt.crear_parser().parse_args([])
        self.assertIsNone(args.tts)
        self.assertEqual((args.python_voz, args.xtts_hablante, args.xtts_dispositivo, args.voz_indice),
                         (None, "Gilberto Mathias", "auto", None))
        self.assertGreater(args.xtts_espera_seg, 0)
        args = self.rt.crear_parser().parse_args(["--tts", "offline", "--tts-voice", "x", "--python-voz", "/p",
                                                  "--xtts-dispositivo", "cpu"])
        self.assertEqual((args.tts, args.tts_voice, args.python_voz, args.xtts_dispositivo),
                         ("offline", "x", "/p", "cpu"))
        for modo in ("pregrabada", "offline", "edge", "gtts", "none"):
            self.assertEqual(self.rt.crear_parser().parse_args(["--tts", modo]).tts, modo)

    def test_pregrabada_si_existe_el_indice_y_si_no_offline_con_aviso(self):
        args = self.rt.crear_parser().parse_args([])
        self.assertEqual(self.rt.resolver_modo_tts(args), ("pregrabada", None))
        args = self.rt.crear_parser().parse_args(["--voz-indice", "/no/existe/indice.json"])
        modo, aviso = self.rt.resolver_modo_tts(args)
        self.assertEqual(modo, "offline")
        self.assertIn("Mónica", aviso)
        self.assertIn("AVISO", aviso)
        args = self.rt.crear_parser().parse_args(["--tts", "offline"])
        self.assertEqual(self.rt.resolver_modo_tts(args), ("offline", None))

    def test_construir_tts_cae_en_cascada_sin_tumbar_el_prototipo(self):
        args = self.rt.crear_parser().parse_args(["--python-voz", "/p", "--xtts-dispositivo", "cpu"])
        llamadas = []

        def constructor(mode, voice, **kw):
            llamadas.append((mode, kw))
            if mode == "pregrabada":
                raise tts_mod.TTSDependenciaFaltante("sin índice")
            return SimpleNamespace(mode=mode)

        with contextlib.redirect_stdout(io.StringIO()) as salida:
            motor, modo = self.rt.construir_tts(args, "pregrabada", constructor)
        self.assertEqual((motor.mode, modo), ("offline", "offline"))
        self.assertEqual(llamadas[0], ("pregrabada", {"indice_pregrabada": None, "python_voz": "/p",
                                                      "xtts_hablante": "Gilberto Mathias", "xtts_dispositivo": "cpu"}))
        self.assertIn("voz pregrabada no disponible", salida.getvalue())

        def todo_falla(mode, voice, **kw):
            raise RuntimeError("nada")

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.rt.construir_tts(args, "pregrabada", todo_falla), (None, "none"))
        self.assertEqual(self.rt.construir_tts(args, "none", todo_falla), (None, "none"))

    def test_espera_del_servicio(self):
        self.assertIsNone(self.rt.esperar_servicio_voz(None, 1))
        self.assertIsNone(self.rt.esperar_servicio_voz(SimpleNamespace(xtts=None), 1))
        for estado, esperado in (("listo", "listo"), ("caido", "NO disponible"), ("cargando", "aun cargando")):
            xtts = ServicioFalso(estado=estado)
            motor = SimpleNamespace(xtts=xtts, esperar_servicio_voz=lambda tope, x=xtts: x.esperar_listo(tope))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIn(esperado, self.rt.esperar_servicio_voz(motor, 0.01))


class TestLatenciaVoz(DoblesVozLocal, unittest.TestCase):
    """Opciones de src/latencia_fuera_de_linea.py para la voz pregrabada y el respaldo XTTS."""

    @classmethod
    def setUpClass(cls):
        from src import latencia_fuera_de_linea as lat
        cls.lat = lat

    def test_frases_fuera_del_inventario(self):
        frases = self.lat.frases_fuera_de_inventario()
        self.assertEqual(len(frases), 20)
        self.assertEqual(frases, self.lat.frases_fuera_de_inventario(), "misma semilla, misma lista")
        indice = vp.IndiceVozPregrabada.cargar(INDICE)
        self.assertFalse(any(f["frase"] in indice for f in frases))
        self.assertEqual({f["frase"] for f in frases if f["tipo"] == "variante_femenino"}, VARIANTES_FEMENINO)
        tipos = [f["tipo"] for f in frases]
        self.assertEqual((tipos.count("reglas"), tipos.count("respaldo_inventario")), (8, 8))
        self.assertEqual(len({f["frase"] for f in frases}), 20)

    def test_crear_tts_pregrabada_sin_servicio_y_silenciada(self):
        self.poner_dobles()
        with contextlib.redirect_stdout(io.StringIO()):
            t = self.lat.crear_tts(cache=False, voz="pregrabada")
        self.addCleanup(t.cleanup)
        self.assertEqual(t.mode, "pregrabada")
        self.assertIsNone(t.xtts)
        t.prepare("Gracias.")
        t.play_prepared_async()
        time.sleep(0.05)
        self.assertEqual(t.locuciones_silenciadas, 1)
        self.assertTrue((INDICE.parent / "001.wav").exists())
        self.assertEqual([o for o in self.ordenes if o[0] == "afplay"], [], "silenciada: no suena")

    def test_opciones(self):
        a = self.lat.parsear_argumentos([])
        self.assertEqual((a.voz, a.configuraciones, a.solo_xtts, a.dispositivos), ("offline", None, False,
                                                                                    ["mps", "cpu"]))
        a = self.lat.parsear_argumentos(["--voz", "pregrabada", "--configuraciones", "final_inventario",
                                         "--solo-xtts", "--dispositivos", "cpu", "--frases-xtts", "5"])
        self.assertEqual((a.voz, a.configuraciones, a.solo_xtts, a.dispositivos, a.frases_xtts),
                         ("pregrabada", ["final_inventario"], True, ["cpu"], 5))

    def test_fila_lleva_la_fuente(self):
        reg = {"etapas_ns": {"preprocesamiento": 1, "inferencia": 2, "conversion": 3, "sintesis": 4}, "e": [10],
               "t5": 20, "N": 1, "estado": "aceptada", "clase": "hola", "confianza": 0.9, "enunciado": "Hola.",
               "L_resp_ns": 30, "tts_fuente": "pregrabada", "tts_tiempo_sintesis_ns": 25_000}
        f = self.lat.fila_ensayo("v", reg)
        self.assertEqual((f["fuente"], f["tiempo_sintesis_motor"]), ("pregrabada", 0.025))

    def test_resultados_guardados(self):
        # Los JSON de la medición del 23-nov-2026 existen y dicen lo que docs/VOZ.md cita.
        pre = json.loads((RAIZ / "docs" / "LATENCIA_VOZ_PREGRABADA.json").read_text(encoding="utf-8"))
        a = pre["agregado"]["final_inventario"]["cache"]
        self.assertEqual(set(a["fuentes"]) - {"None"}, {"pregrabada"})
        self.assertEqual(pre["parametros"]["voz"], "pregrabada")
        x = json.loads((RAIZ / "docs" / "LATENCIA_VOZ_XTTS.json").read_text(encoding="utf-8"))
        for disp in ("mps", "cpu"):
            d = x["xtts"]["dispositivos"][disp]
            self.assertTrue(d["listo"])
            self.assertEqual(d["info_servicio"]["dispositivo"], disp)
            self.assertEqual(d["fuentes"], {"xtts": 60})


class TestSinTorchEnElProyecto(unittest.TestCase):

    def test_importar_la_voz_y_el_prototipo_no_importa_torch(self):
        codigo = ("import sys; sys.path.insert(0, '.');"
                  "import src.tts, src.voz_pregrabada, src.voz_xtts_servicio, src.translator, src.realtime;"
                  "t = src.tts.modo_tts_prototipo();"
                  "print(sorted(m for m in ('torch', 'TTS', 'torchaudio', 'transformers') if m in sys.modules))")
        r = subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(r.stdout.strip().splitlines()[-1], "[]")

    def test_torch_solo_dentro_del_servidor(self):
        for ruta in sorted((RAIZ / "src").glob("*.py")):
            arbol = ast.parse(ruta.read_text(encoding="utf-8"))
            for nodo in arbol.body:          # importaciones de nivel de módulo
                nombres = []
                if isinstance(nodo, ast.Import):
                    nombres = [a.name for a in nodo.names]
                elif isinstance(nodo, ast.ImportFrom):
                    nombres = [nodo.module or ""]
                for n in nombres:
                    self.assertFalse(n.split(".")[0] in ("torch", "TTS", "torchaudio"), f"{ruta.name}: {n}")
        texto = (RAIZ / "src" / "voz_xtts_servicio.py").read_text(encoding="utf-8")
        self.assertIn("import torch", texto)          # sí lo importa, dentro de _MotorXTTS
        requisitos = (RAIZ / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertNotIn("torch", requisitos)
        self.assertNotIn("coqui", requisitos)
        voz = (RAIZ / "requirements-voz.txt").read_text(encoding="utf-8")
        for paquete in ("coqui-tts==0.28.0", "transformers==4.57.3", "torch==", "torchaudio==", "six=="):
            self.assertIn(paquete, voz)


if __name__ == "__main__":
    unittest.main()
