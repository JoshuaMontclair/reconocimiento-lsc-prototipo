"""
Servicio auxiliar de voz de respaldo con XTTS-v2 (decisión del autor del 23-nov-2026; ver docs/VOZ.md).

POR QUÉ UN PROCESO APARTE. La voz de respaldo para las frases que NO están pregrabadas (respaldo del conversor con
la lista de glosas, conversor «reglas» o «heredado») es XTTS-v2 de Coqui, hablante
«Gilberto Mathias», idioma "es". XTTS necesita PyTorch y coqui-tts, que el entorno del proyecto NO tiene ni debe
tener (TensorFlow y MediaPipe ya conviven en él). Por eso XTTS corre en un proceso auxiliar persistente, lanzado
con OTRO intérprete de Python (un entorno con coqui-tts; ver requirements-voz.txt), que carga el modelo UNA vez y
sintetiza a archivo bajo pedido. Este archivo tiene las dos mitades:
  · el SERVIDOR (función servir; se ejecuta con el intérprete que tiene coqui-tts). Solo él importa torch, y lo
    hace dentro de la función: importar este módulo desde el proyecto no importa torch;
  · el CLIENTE (clase ServicioXTTS), que usa src/tts.py en modo «pregrabada». Solo biblioteca estándar.

PROTOCOLO: JSON por línea (UTF-8) por stdin/stdout del proceso auxiliar. Es lo más simple y robusto aquí: no hay
puertos, el proceso muere solo si el prototipo se cierra (EOF en su stdin) y un fallo del auxiliar se ve como fin
de su stdout. Todo lo que coqui-tts o torch impriman va a stderr: el servidor duplica el descriptor 1 para el
canal y redirige el 1 al 2 ANTES de importar nada pesado, así que ninguna impresión ajena ensucia el protocolo.
    servidor → cliente, al arrancar:  {"evento": "listo", "motor", "dispositivo", "hablante", "idioma",
                                        "frecuencia_hz", "carga_s", "calentamiento_ms", "pid", "versiones"}
                                o:   {"evento": "error", "error": "..."} y termina con código 2.
    cliente → servidor:              {"id": 7, "texto": "Hola, tío.", "salida": "/ruta/absoluta.wav"}
    servidor → cliente:              {"id": 7, "ok": true, "salida", "ms", "duracion_s", "frecuencia_hz",
                                      "semilla", "descartadas_s"}
                                o:   {"id": 7, "ok": false, "error": "...", "ms"}
    cliente → servidor:              {"orden": "salir"}  (o cerrar su stdin) → termina con código 0.
El WAV (PCM 16 bits, mono, 24 kHz) se escribe primero en «salida.parcial» y se renombra al terminar: quien lo lee
nunca ve un archivo a medias.

DISPOSITIVO: «auto» usa MPS (la GPU de Apple) si torch lo ofrece y si no la CPU; «mps» y «cpu» lo fuerzan (si MPS
falla al cargar, se usa la CPU y el evento «listo» lo dice). Medido en este Mac (docs/VOZ.md): ~0,4 s por frase en
MPS y ~0,8 s en CPU. Cada petición fija la semilla (SEMILLA) para que la misma frase suene igual dentro de una
sesión y entre sesiones del mismo equipo y dispositivo.

MOTOR «prueba»: el mismo servidor y el mismo protocolo, pero sin torch: escribe un tono corto. Sirve para las
pruebas (tests/test_voz_xtts_servicio.py) y para comprobar la tubería con el intérprete del proyecto. Opciones
solo de prueba: --retraso-ms, --fallar-con TEXTO (responde ok=false), --morir-con TEXTO (el proceso termina sin
responder), --alargar-con TEXTO / --alargar-siempre-con TEXTO (audio de 30 s en el primer intento o en todos),
--no-arrancar (termina sin decir «listo»), --colgar-arranque (nunca dice «listo»).

SALVAGUARDA CONTRA EL BALBUCEO. XTTS a veces sigue generando sílabas sin sentido después de la frase (medido en MPS
con una de 20 frases; ver SEMILLA y DURACION_*). Si el audio dura más de lo creíble para el texto se repite con otra
semilla y, si no se arregla, se responde con error: el prototipo dice esa frase con la voz local.

USO (normalmente lo lanza src/tts.py; ver --python-voz en src/realtime.py o la variable TESIS_PYTHON_VOZ):
    <python con coqui-tts> -I src/voz_xtts_servicio.py --dispositivo auto
    python src/voz_xtts_servicio.py --motor prueba          # sin torch
    python src/voz_xtts_servicio.py --probar "Hola, tío." --python-voz <python con coqui-tts>
                                                            # arranca el servicio, sintetiza y mide
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HABLANTE_POR_DEFECTO = "Gilberto Mathias"
IDIOMA_POR_DEFECTO = "es"
MODELO_XTTS = "tts_models/multilingual/multi-dataset/xtts_v2"
DISPOSITIVOS = ("auto", "mps", "cpu")
MOTORES = ("xtts", "prueba")
SEMILLA = 2026
# Salvaguarda contra el «balbuceo» de XTTS (medido el 23-nov-2026, docs/VOZ.md): en MPS, «Buenas tardes, tú eres
# bienvenido.» con la semilla 2026 da 10,9 s de audio (la frase y ~8 s de sílabas sin sentido, según Whisper); con
# otras semillas, o en CPU, 2-3,6 s. En 120 síntesis normales la duración máxima fue 0,17 s por carácter (frases de
# 19 caracteres; las cortas llevan más silencio relativo). Si el audio pasa de DURACION_BASE_S +
# DURACION_POR_CARACTER_S × caracteres se repite con la semilla siguiente, hasta REINTENTOS_DURACION veces; si
# sigue largo, la frase se responde con error y el prototipo usa la voz local para ella.
DURACION_BASE_S = 1.5
DURACION_POR_CARACTER_S = 0.12
REINTENTOS_DURACION = 2
FRASE_CALENTAMIENTO = "Hola."
TOPE_ARRANQUE_SEG = 180.0      # carga del modelo (medida: ~10-25 s con el modelo ya descargado)
TOPE_SINTESIS_SEG = 30.0       # una frase (medida: ~0,4 s en MPS, ~0,8 s en CPU)
VARIABLE_PYTHON_VOZ = "TESIS_PYTHON_VOZ"
SCRIPT = Path(__file__).resolve()


class ErrorServicioVoz(RuntimeError):
    """El servicio de voz no está disponible, no respondió a tiempo o no pudo sintetizar."""


# ---------------------------------------------------------------------------
# Servidor (se ejecuta con el intérprete que tiene coqui-tts; torch solo aquí dentro)
# ---------------------------------------------------------------------------

def _escribir_wav_pcm16(ruta: str, muestras, frecuencia: int) -> float:
    """Escribe floats en [-1, 1] como WAV PCM 16 bits mono, de forma atómica. Devuelve la duración en s."""
    import wave

    try:                                  # el entorno de coqui-tts tiene numpy: conversión vectorizada
        import numpy as np
        bruto = (np.clip(np.asarray(muestras, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2").tobytes()
    except ImportError:                   # motor «prueba» con un intérprete sin numpy
        import array
        datos = array.array("h", (int(max(-1.0, min(1.0, float(x))) * 32767) for x in muestras))
        if sys.byteorder != "little":
            datos.byteswap()
        bruto = datos.tobytes()
    parcial = ruta + ".parcial"
    with wave.open(parcial, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(frecuencia))
        w.writeframes(bruto)
    os.replace(parcial, ruta)
    return (len(bruto) // 2) / float(frecuencia)


class _MotorPrueba:
    """Motor sin torch: un tono de 0,25 s a 24 kHz. Mismo protocolo que XTTS."""
    frecuencia_hz = 24000

    def __init__(self, args):
        self.args = args
        self.dispositivo = "ninguno"
        self.versiones = {"motor": "prueba"}

    def sintetizar(self, texto: str, semilla: int = SEMILLA):
        import math
        a = self.args
        if a.morir_con and a.morir_con in texto:
            os._exit(3)                      # simula una caída del proceso a mitad de una petición
        if a.fallar_con and a.fallar_con in texto:
            raise RuntimeError(f"fallo simulado con {texto!r}")
        if a.retraso_ms:
            time.sleep(a.retraso_ms / 1000.0)
        segundos = 0.25
        if a.alargar_siempre_con and a.alargar_siempre_con in texto:
            segundos = 30.0                  # simula el balbuceo de XTTS en todos los intentos
        elif a.alargar_con and a.alargar_con in texto and semilla == SEMILLA:
            segundos = 30.0                  # solo con la primera semilla: el reintento sale bien
        n = int(segundos * self.frecuencia_hz)
        return [0.2 * math.sin(2 * math.pi * 440 * i / self.frecuencia_hz) for i in range(n)]


class _MotorXTTS:
    """XTTS-v2 de coqui-tts, cargado una sola vez."""
    frecuencia_hz = 24000

    def __init__(self, args):
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        from TTS.api import TTS

        self._torch = torch
        self.hablante, self.idioma = args.hablante, args.idioma
        pedido = args.dispositivo
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        self.dispositivo = ("mps" if mps else "cpu") if pedido == "auto" else pedido
        self.nota_dispositivo = None
        tts = TTS(MODELO_XTTS)
        try:
            tts = tts.to(self.dispositivo)
        except Exception as exc:  # noqa: BLE001 — MPS que falla al cargar: se sigue en CPU y se dice
            self.nota_dispositivo = f"{self.dispositivo} falló al cargar ({type(exc).__name__}: {exc}); se usa cpu"
            self.dispositivo = "cpu"
            tts = tts.to("cpu")
        self.tts = tts
        try:
            sr = int(tts.synthesizer.output_sample_rate)
            self.frecuencia_hz = sr or self.frecuencia_hz
        except Exception:  # noqa: BLE001
            pass
        import importlib.metadata as md
        self.versiones = {p: _version(md, p) for p in ("coqui-tts", "torch", "torchaudio", "transformers")}

    def sintetizar(self, texto: str, semilla: int = SEMILLA):
        self._torch.manual_seed(semilla)
        return self.tts.tts(text=texto, speaker=self.hablante, language=self.idioma)


def duracion_maxima_s(texto: str) -> float:
    """Duración máxima creíble del audio de un texto (salvaguarda contra el balbuceo de XTTS)."""
    return DURACION_BASE_S + DURACION_POR_CARACTER_S * len(texto.strip())


def _version(md, paquete: str) -> str | None:
    try:
        return md.version(paquete)
    except Exception:  # noqa: BLE001
        return None


def servir(args) -> int:
    """Bucle del servidor: carga el motor, dice «listo» y atiende peticiones hasta «salir» o EOF."""
    # Canal limpio: el descriptor 1 original queda solo para el protocolo; todo lo demás va a stderr.
    canal = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def enviar(obj: dict) -> None:
        canal.write(json.dumps(obj, ensure_ascii=False) + "\n")
        canal.flush()

    if args.no_arrancar:
        return 2
    t0 = time.perf_counter()
    try:
        motor = _MotorPrueba(args) if args.motor == "prueba" else _MotorXTTS(args)
        t1 = time.perf_counter()
        motor.sintetizar(FRASE_CALENTAMIENTO)            # la primera síntesis es más lenta: no la paga el usuario
        calentamiento_ms = (time.perf_counter() - t1) * 1000
    except Exception as exc:  # noqa: BLE001 — el cliente lo ve como evento «error» y cae a la voz local
        enviar({"evento": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 2
    if args.colgar_arranque:
        time.sleep(3600)
    enviar({"evento": "listo", "motor": args.motor, "dispositivo": motor.dispositivo,
            "nota_dispositivo": getattr(motor, "nota_dispositivo", None),
            "hablante": args.hablante, "idioma": args.idioma, "frecuencia_hz": motor.frecuencia_hz,
            "carga_s": round(time.perf_counter() - t0, 2), "calentamiento_ms": round(calentamiento_ms, 1),
            "pid": os.getpid(), "versiones": motor.versiones})
    for linea in sys.stdin:
        linea = linea.strip()
        if not linea:
            continue
        try:
            pedido = json.loads(linea)
        except ValueError:
            enviar({"id": None, "ok": False, "error": "línea no es JSON"})
            continue
        if not isinstance(pedido, dict):
            enviar({"id": None, "ok": False, "error": "el pedido no es un objeto"})
            continue
        if pedido.get("orden") == "salir":
            break
        ident, texto, salida = pedido.get("id"), pedido.get("texto"), pedido.get("salida")
        if not isinstance(texto, str) or not texto.strip() or not isinstance(salida, str) or not os.path.isabs(salida):
            enviar({"id": ident, "ok": False, "error": "faltan «texto» o «salida» (ruta absoluta)"})
            continue
        t = time.perf_counter()
        limite = duracion_maxima_s(texto)
        descartadas = []
        try:
            for intento in range(REINTENTOS_DURACION + 1):
                semilla = SEMILLA + intento
                muestras = motor.sintetizar(texto, semilla)
                dur = len(muestras) / float(motor.frecuencia_hz)
                if dur <= limite:
                    break
                descartadas.append(round(dur, 2))
            else:
                raise RuntimeError(f"audio sospechoso (balbuceo): {descartadas} s para {len(texto.strip())} "
                                   f"caracteres, límite {limite:.1f} s")
            dur = _escribir_wav_pcm16(salida, muestras, motor.frecuencia_hz)
        except Exception as exc:  # noqa: BLE001 — una frase que falla no tumba el servicio
            enviar({"id": ident, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "ms": round((time.perf_counter() - t) * 1000, 1)})
            continue
        enviar({"id": ident, "ok": True, "salida": salida, "ms": round((time.perf_counter() - t) * 1000, 1),
                "duracion_s": round(dur, 3), "frecuencia_hz": motor.frecuencia_hz, "semilla": semilla,
                "descartadas_s": descartadas})
    return 0


# ---------------------------------------------------------------------------
# Cliente (lo usa src/tts.py; solo biblioteca estándar)
# ---------------------------------------------------------------------------

def python_voz_desde_entorno() -> str | None:
    """Intérprete con coqui-tts indicado en TESIS_PYTHON_VOZ, o None (sin servicio: el respaldo es la voz local)."""
    valor = os.environ.get(VARIABLE_PYTHON_VOZ, "").strip()
    return valor or None


class ServicioXTTS:
    """Lanza y habla con el proceso auxiliar. Seguro entre hilos; nunca bloquea más que los topes dados.

    Estados: «sin_iniciar» → «cargando» (proceso vivo, sin «listo») → «listo» → «caido» (el proceso terminó, dijo
    «error», no estuvo listo a tiempo, no respondió a tiempo o se rompió el canal). Una vez «caido» no se vuelve a
    usar en la sesión: quien lo usa cae a la voz local. Un pedido con ok=false (una frase que falla) NO lo tumba.
    """

    def __init__(self, python: str, hablante: str = HABLANTE_POR_DEFECTO, idioma: str = IDIOMA_POR_DEFECTO,
                 dispositivo: str = "auto", motor: str = "xtts", tope_arranque_seg: float = TOPE_ARRANQUE_SEG,
                 tope_sintesis_seg: float = TOPE_SINTESIS_SEG, argumentos_extra=(), script: str | Path | None = None,
                 al_cambiar=None):
        if dispositivo not in DISPOSITIVOS:
            raise ValueError(f"dispositivo debe ser uno de {DISPOSITIVOS}, no {dispositivo!r}")
        if motor not in MOTORES:
            raise ValueError(f"motor debe ser uno de {MOTORES}, no {motor!r}")
        self.python = str(python)
        self.hablante, self.idioma, self.dispositivo_pedido, self.motor = hablante, idioma, dispositivo, motor
        self.tope_arranque_seg = float(tope_arranque_seg)
        self.tope_sintesis_seg = float(tope_sintesis_seg)
        self.argumentos_extra = list(argumentos_extra)
        self.script = str(script or SCRIPT)
        self.al_cambiar = al_cambiar               # función(estado, detalle) para avisar (la pone src/tts.py)
        self.estado_actual = "sin_iniciar"
        self.motivo_caida: str | None = None
        self.info_listo: dict | None = None
        self.peticiones = self.fallos = 0
        self.ultima_respuesta: dict | None = None   # la del último pedido con éxito (ms del servidor, etc.)
        self.t_inicio: float | None = None
        self.t_listo: float | None = None
        self._proc: subprocess.Popen | None = None
        self._evento_arranque = threading.Event()   # se activa con «listo» o con la caída
        self._candado = threading.Lock()            # estado y pendientes
        self._candado_pedido = threading.Lock()     # una petición a la vez (el servidor es secuencial)
        self._pendientes: dict[int, list] = {}
        self._siguiente_id = 0
        self._stderr = collections.deque(maxlen=40)

    # -- ciclo de vida --------------------------------------------------------------------------------------
    def orden(self) -> list[str]:
        return ([self.python, "-I", self.script, "--motor", self.motor, "--hablante", self.hablante,
                 "--idioma", self.idioma, "--dispositivo", self.dispositivo_pedido] + self.argumentos_extra)

    def iniciar(self) -> None:
        """Lanza el proceso y retorna en el acto (la carga sigue en segundo plano). ErrorServicioVoz si no
        se puede lanzar (intérprete inexistente, por ejemplo)."""
        if self._proc is not None:
            return
        entorno = dict(os.environ)
        entorno.update({"COQUI_TOS_AGREED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONIOENCODING": "utf-8",
                        "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1"})
        self.t_inicio = time.monotonic()
        try:
            self._proc = subprocess.Popen(self.orden(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, env=entorno, bufsize=0)
        except OSError as exc:
            self._caer(f"no se pudo lanzar {self.python} ({type(exc).__name__}: {exc})")
            raise ErrorServicioVoz(self.motivo_caida) from exc
        self._cambiar("cargando", f"pid {self._proc.pid}")
        for destino in (self._leer_stdout, self._leer_stderr, self._vigilar_arranque):
            threading.Thread(target=destino, daemon=True).start()

    def esperar_listo(self, tope_seg: float | None = None) -> bool:
        """Espera a que el servicio esté listo (o caiga) como mucho tope_seg. True si está disponible."""
        if self._proc is None:
            return False
        self._evento_arranque.wait(timeout=tope_seg)
        return self.disponible

    @property
    def disponible(self) -> bool:
        return self.estado_actual == "listo"

    @property
    def caido(self) -> bool:
        return self.estado_actual == "caido"

    @property
    def cargando(self) -> bool:
        return self.estado_actual == "cargando"

    def detener(self, tope_seg: float = 2.0) -> None:
        """Pide salir, espera un poco y, si sigue vivo, lo termina. Idempotente. Solo actúa sobre el proceso que
        lanzó este cliente."""
        proc = self._proc
        if proc is None:
            return
        with self._candado:
            ya = self.estado_actual == "caido"
            self.estado_actual = "caido"
            self.motivo_caida = self.motivo_caida or "detenido al cerrar"
            pendientes = list(self._pendientes.values())
        for p in pendientes:
            p[0].set()
        self._evento_arranque.set()
        if proc.poll() is None:
            try:
                if proc.stdin and not ya:
                    proc.stdin.write(b'{"orden": "salir"}\n')
                    proc.stdin.flush()
            except (OSError, ValueError):
                pass
            try:
                proc.stdin.close()
            except (OSError, ValueError, AttributeError):
                pass
            try:
                proc.wait(timeout=tope_seg)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=tope_seg)
                except subprocess.TimeoutExpired:
                    proc.kill()

    # -- síntesis -------------------------------------------------------------------------------------------
    def sintetizar(self, texto: str, salida: str | Path, tope_seg: float | None = None) -> dict:
        """Sintetiza texto en el WAV `salida` (ruta absoluta). Devuelve la respuesta del servidor más «ms_cliente»
        (ida y vuelta medida aquí). ErrorServicioVoz si no está disponible, si la frase falla o si no responde a
        tiempo; en este último caso el servicio se da por caído y se termina el proceso."""
        if not self.disponible:
            raise ErrorServicioVoz(f"servicio XTTS {self.estado_actual}"
                                   + (f": {self.motivo_caida}" if self.motivo_caida else ""))
        tope = self.tope_sintesis_seg if tope_seg is None else float(tope_seg)
        salida = str(Path(salida).resolve())
        with self._candado_pedido:
            with self._candado:
                self._siguiente_id += 1
                ident = self._siguiente_id
                hueco = [threading.Event(), None]
                self._pendientes[ident] = hueco
                self.peticiones += 1
            t0 = time.perf_counter()
            try:
                linea = json.dumps({"id": ident, "texto": texto, "salida": salida}, ensure_ascii=False) + "\n"
                self._proc.stdin.write(linea.encode("utf-8"))
                self._proc.stdin.flush()
            except (OSError, ValueError, AttributeError) as exc:
                self._olvidar(ident)
                self._caer(f"canal roto al pedir ({type(exc).__name__}: {exc})")
                raise ErrorServicioVoz(self.motivo_caida) from exc
            respondio = hueco[0].wait(timeout=tope)
            self._olvidar(ident)
            resp = hueco[1]
            if not respondio:
                self._caer(f"no respondió en {tope:.0f} s")
                self.detener(tope_seg=1.0)
                raise ErrorServicioVoz(self.motivo_caida)
            if resp is None:
                raise ErrorServicioVoz(f"servicio XTTS caído: {self.motivo_caida}")
            if not resp.get("ok"):
                with self._candado:
                    self.fallos += 1
                raise ErrorServicioVoz(f"XTTS no pudo sintetizar: {resp.get('error')}")
            resp = dict(resp)
            resp["ms_cliente"] = round((time.perf_counter() - t0) * 1000, 1)
            self.ultima_respuesta = resp
            if not (os.path.exists(salida) and os.path.getsize(salida) > 44):
                with self._candado:
                    self.fallos += 1
                raise ErrorServicioVoz(f"XTTS respondió ok pero no escribió {salida}")
            return resp

    def estado(self) -> dict:
        with self._candado:
            info = dict(self.info_listo or {})
            return {"estado": self.estado_actual, "motivo_caida": self.motivo_caida,
                    "dispositivo": info.get("dispositivo"), "hablante": self.hablante, "idioma": self.idioma,
                    "carga_s": info.get("carga_s"), "peticiones": self.peticiones, "fallos": self.fallos,
                    "pid": self._proc.pid if self._proc else None}

    def ultimas_lineas_stderr(self, n: int = 5) -> list[str]:
        return list(self._stderr)[-n:]

    # -- hilos internos -------------------------------------------------------------------------------------
    def _olvidar(self, ident: int) -> None:
        with self._candado:
            self._pendientes.pop(ident, None)

    def _cambiar(self, estado: str, detalle: str | None = None) -> None:
        with self._candado:
            if self.estado_actual == "caido":
                return
            self.estado_actual = estado
        if self.al_cambiar is not None:
            try:
                self.al_cambiar(estado, detalle)
            except Exception:  # noqa: BLE001 — un aviso nunca tumba el servicio
                pass

    def _caer(self, motivo: str) -> None:
        with self._candado:
            if self.estado_actual == "caido":
                return
            self.estado_actual = "caido"
            self.motivo_caida = motivo
            pendientes = list(self._pendientes.values())
        for p in pendientes:
            p[0].set()
        self._evento_arranque.set()
        if self.al_cambiar is not None:
            try:
                cola = self.ultimas_lineas_stderr(3)
                self.al_cambiar("caido", motivo + (f" | stderr: {' / '.join(cola)}" if cola else ""))
            except Exception:  # noqa: BLE001
                pass

    def _leer_stdout(self) -> None:
        proc = self._proc
        try:
            for bruto in proc.stdout:
                try:
                    msg = json.loads(bruto.decode("utf-8", "replace"))
                except ValueError:
                    self._stderr.append("(stdout no JSON) " + bruto.decode("utf-8", "replace").strip()[:200])
                    continue
                if not isinstance(msg, dict):
                    continue
                evento = msg.get("evento")
                if evento == "listo":
                    with self._candado:
                        self.info_listo = msg
                    self.t_listo = time.monotonic()
                    self._cambiar("listo", f"{msg.get('dispositivo')}, carga {msg.get('carga_s')} s")
                    self._evento_arranque.set()
                elif evento == "error":
                    self._caer(f"error al cargar: {msg.get('error')}")
                elif "id" in msg:
                    with self._candado:
                        hueco = self._pendientes.get(msg.get("id"))
                    if hueco is not None:
                        hueco[1] = msg
                        hueco[0].set()
        except (OSError, ValueError):
            pass
        codigo = proc.wait()
        self._caer(f"el proceso terminó (código {codigo})")

    def _leer_stderr(self) -> None:
        try:
            for bruto in self._proc.stderr:
                texto = bruto.decode("utf-8", "replace").strip()
                if texto:
                    self._stderr.append(texto[:300])
        except (OSError, ValueError):
            pass

    def _vigilar_arranque(self) -> None:
        if not self._evento_arranque.wait(timeout=self.tope_arranque_seg):
            self._caer(f"no estuvo listo en {self.tope_arranque_seg:.0f} s")
            self.detener(tope_seg=1.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Servicio XTTS-v2 de respaldo (JSON por línea por stdin/stdout)")
    p.add_argument("--motor", choices=MOTORES, default="xtts")
    p.add_argument("--hablante", default=HABLANTE_POR_DEFECTO)
    p.add_argument("--idioma", default=IDIOMA_POR_DEFECTO)
    p.add_argument("--dispositivo", choices=DISPOSITIVOS, default="auto")
    p.add_argument("--probar", metavar="TEXTO", default=None,
                   help="modo cliente: lanza el servicio con --python-voz, sintetiza TEXTO a un WAV temporal y mide")
    p.add_argument("--python-voz", default=None, help=f"intérprete con coqui-tts (default ${VARIABLE_PYTHON_VOZ})")
    # Solo para el motor «prueba» (pruebas automáticas).
    p.add_argument("--retraso-ms", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--fallar-con", default=None, help=argparse.SUPPRESS)
    p.add_argument("--morir-con", default=None, help=argparse.SUPPRESS)
    p.add_argument("--alargar-con", default=None, help=argparse.SUPPRESS)
    p.add_argument("--alargar-siempre-con", default=None, help=argparse.SUPPRESS)
    p.add_argument("--no-arrancar", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--colgar-arranque", action="store_true", help=argparse.SUPPRESS)
    return p


def _probar(args) -> int:
    import tempfile
    python = args.python_voz or python_voz_desde_entorno() or sys.executable
    cli = ServicioXTTS(python, args.hablante, args.idioma, args.dispositivo, args.motor)
    cli.iniciar()
    print(f"cargando con {python} ...", flush=True)
    if not cli.esperar_listo(TOPE_ARRANQUE_SEG):
        print(f"ERROR: {cli.estado()} {cli.ultimas_lineas_stderr()}")
        return 1
    print("listo:", json.dumps(cli.info_listo, ensure_ascii=False))
    carpeta = tempfile.mkdtemp(prefix="lsc_xtts_")
    salida = os.path.join(carpeta, "prueba.wav")
    print("respuesta:", json.dumps(cli.sintetizar(args.probar, salida), ensure_ascii=False))
    cli.detener()
    os.remove(salida)
    os.rmdir(carpeta)
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.probar is not None:
        return _probar(args)
    return servir(args)


if __name__ == "__main__":
    sys.exit(main())
