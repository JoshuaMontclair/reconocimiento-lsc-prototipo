"""
Latencia FUERA DE LÍNEA del prototipo tras el cierre de la ventana (actividad 4.c, modelo final D-14).

QUÉ PREGUNTA RESUELVE. La tesis cita 36 ms entre el cierre de la ventana y la orden de reproducción
(L_resp, N-52), medidos EN VIVO con la red heredada (canónico «manos», «mascara», umbral 0,20). D-14
cambió el modelo (TCN + imputación + 252, umbral 0,00) y D-15 el conversor de la tesis (--conversor
inventario). Este script mide, en este equipo y SIN CÁMARA, la parte de la cadena que va después del
cierre de la ventana con el modelo final y con la heredada, con el MISMO método para ambas, y aparte la
tasa del extractor de MediaPipe sobre fotogramas del corpus. No sustituye a una sesión en vivo.

QUÉ HACE.
  1. Post-cierre (bloque «post_cierre»). Toma secuencias de puntos YA EXTRAÍDAS del corpus
     (data/extracted/LSC50_holistic, solo lectura, con preprocess_lsc50.load_single_sample) y las pasa
     fotograma a fotograma por LSCTranslator (src/translator.py) con un extractor pregrabado que devuelve
     esos puntos al instante. La ventana es el clip completo (N = fotogramas del clip). Así las marcas
     t1..t5 las sella el propio traductor y las etapas son las del JSONL del prototipo:
       preprocesamiento = t2 − e_N   (normalización, mano ausente, columnas, 30 pasos, velocidades)
       inferencia       = t3 − t2    (UNA llamada a model.predict, como en vivo)
       conversion       = t4 − t3    (conversor de glosas)
       sintesis         = t5 − t4    (src/tts.py offline: prepare → esperar_preparado → orden)
       post_cierre      = t5 − e_N   (suma de las cuatro; equivale a L_resp sin la extracción del
                                      último fotograma, que en vivo vale ≈12 ms)
     El modelo se carga UNA vez por configuración; el traductor hace su inferencia de calentamiento y,
     además, se descartan CALENTAMIENTO ensayos antes de medir. Cada configuración se mide con la voz en
     dos condiciones: «cache» (como en vivo: el traductor precalienta las 50 glosas sueltas al arrancar)
     y «sin_cache» (peor caso: la frase se sintetiza en el momento con el sintetizador persistente).
  2. Turnos (bloque «turnos»). Conversión de las 50 glosas solas y de las 92 secuencias del inventario
     (docs/CONVERSOR_3a_ESPECIFICACION.md) con el conversor de cada configuración, y síntesis sin cache de
     las frases que salen (lo que costaría un fin de turno con una frase nueva).
  3. Extractor (bloque «extractor»). LandmarkExtractor del prototipo (HolisticLandmarker de D-05, CPU,
     modo VIDEO) sobre los fotogramas de los videos de la muestra leídos EN MEMORIA con OpenCV, en su
     resolución nativa (1920×1080) y reducidos en memoria a 1280×720 (la que usa la sesión en vivo).
     Se cronometra como en vivo: c_i después de leer el fotograma, BGR→RGB, extract(), e_i; costo =
     e_i − c_i. Ningún fotograma se escribe a disco (§1.5).

VOZ SIN ALTAVOCES. La síntesis es la real (src/tts.py, modo offline, sintetizador de Apple a archivo
AIFF en la carpeta temporal del módulo), pero la reproducción se sustituye por una función que solo
borra el archivo: no suena nada. La etapa «síntesis» del prototipo termina en la ORDEN de reproducción
(t5), no cuando suena, así que lo medido es lo mismo salvo el arranque del hilo de reproducción, que sí
se mide (play_prepared_async sigue creando el hilo).

CONFIGURACIONES.
  final_inventario   models/tcn_imputacion/modelo.keras + --conversor inventario (lo que va a la tesis)
  heredada_heredado  ruta por defecto de realtime.py (canónico «manos») + conversor heredado (N-52)
  final_heredado     modelo final + conversor heredado (control: es la configuración del registro del
                     3-oct y separa el efecto del modelo del del conversor)
  El umbral es el de <modelo>.umbral.json (0,00 y 0,20), como en el prototipo.

CARGA DEL EQUIPO. Antes de cada bloque se lee el promedio de carga de 1 min (os.getloadavg) y, si pasa
de --umbral-carga, se espera en pasos cortos hasta que baje o hasta agotar el tope por bloque y el tope
total; si no baja se mide igual y queda anotado. Cada bloque guarda la carga al empezar y al terminar
y los procesos con más CPU. Todo se repite --repeticiones veces (3) y se reporta la variación.

MUESTRA. Estratificada: --por-clase videos por seña (2 → 100 videos). Para la seña s se toman los
señantes en las posiciones (por_clase·s + j) mód k de la lista ordenada de señantes disponibles, de modo
que los 5 señantes quedan equilibrados (20 videos cada uno con 2 por clase), y la repetición con
random.Random(semilla) sobre la lista ordenada. Misma semilla → misma lista.

LÍMITES QUE SE DECLARAN. Los ensayos van seguidos (sin los 2,5 s de captura entre uno y otro): la
inferencia puede salir algo más rápida que en vivo, donde MediaPipe ocupa la CPU entre ventanas. Las
predicciones sobre clips del corpus NO son una medida de exactitud (el modelo final se entrenó con la
partición aleatoria, que incluye parte de estos clips). El extractor fuera de línea no incluye el
dibujo ni la ventana de OpenCV del bucle en vivo.

VOZ PREGRABADA Y RESPALDO XTTS (23-nov-2026; docs/VOZ.md).
  --voz pregrabada   el mismo post-cierre y los mismos turnos con src/tts.py en modo «pregrabada» (WAV del índice
                     models/voz_pregrabada; sin servicio XTTS: lo que no está pregrabado iría a la voz local). Cada
                     fila guarda la fuente (tts_fuente) y el tiempo del motor (tts_tiempo_sintesis_ns). Con
                     --configuraciones final_inventario se mide solo la configuración de la tesis, comparable con
                     la mediana de post_cierre de docs/LATENCIA_4c_MODELO_FINAL.md (15,2 ms con la voz offline).
  --solo-xtts        mide aparte el respaldo XTTS-v2 con el servicio REAL (src/voz_xtts_servicio.py, lanzado con
                     --python-voz) en cada --dispositivos (mps, cpu): carga, y por frase el tiempo de prepare()
                     hasta tener el WAV (lo que vería el prototipo) y el del servidor, sobre FRASES_XTTS frases
                     que NO están pregrabadas (variantes en femenino, salidas de «reglas» fuera del inventario y
                     respaldos con lista de glosas), sin cache, --repeticiones veces. No suena nada.

Uso:
    python src/latencia_fuera_de_linea.py      # medición completa → docs/LATENCIA_4c_MODELO_FINAL.json
                                               # (resúmenes) y models/latencia_4c_2026-11-08/crudo.json (filas)
    python src/latencia_fuera_de_linea.py --por-clase 1 --repeticiones 1 --sin-extractor \
        --salida /tmp/x.json --crudo /tmp/x_crudo.json
    python src/latencia_fuera_de_linea.py --solo-registros         # solo caracteriza registros/*.jsonl
    python src/latencia_fuera_de_linea.py --voz pregrabada --configuraciones final_inventario --sin-extractor \
        --salida docs/LATENCIA_VOZ_PREGRABADA.json --crudo models/latencia_voz_2026-11-23/crudo.json
    python src/latencia_fuera_de_linea.py --solo-xtts --python-voz <python con coqui-tts> \
        --salida docs/LATENCIA_VOZ_XTTS.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")      # preprocess_lsc50 importa pyplot; nunca abre ventanas
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import translator as trmod  # noqa: E402
from src.translator import Extraccion, IDX_MANO_DER, IDX_MANO_IZQ, IDX_POSE, LSCTranslator  # noqa: E402
from src import latencia_resumen as lr  # noqa: E402

EXTRAIDOS = ROOT_DIR / "data" / "extracted" / "LSC50_holistic"
VIDEOS = ROOT_DIR / "data" / "raw" / "LSC50" / "VIDEOS" / "COLOR_BODY"
REGISTROS = ROOT_DIR / "registros"
SALIDA = ROOT_DIR / "docs" / "LATENCIA_4c_MODELO_FINAL.json"
CRUDO = ROOT_DIR / "models" / "latencia_4c_2026-11-08" / "crudo.json"    # filas por ensayo

MODELO_FINAL = ROOT_DIR / "models" / "tcn_imputacion" / "modelo.keras"
MODELO_HEREDADO = Path(trmod.MODELO_HEREDADO)        # red heredada (defecto de realtime.py hasta el 22-nov-2026)
CONFIGURACIONES = {
    "final_inventario": {"modelo": MODELO_FINAL, "conversor": "inventario"},
    "heredada_heredado": {"modelo": MODELO_HEREDADO, "conversor": "heredado"},
    "final_heredado": {"modelo": MODELO_FINAL, "conversor": "heredado"},
}
CONDICIONES_VOZ = ("cache", "sin_cache")
ETAPAS_POST = ("preprocesamiento", "inferencia", "conversion", "sintesis")
MAGNITUDES_POST = ETAPAS_POST + ("post_cierre", "L_resp_fuera_de_linea")

SEMILLA = 2026
POR_CLASE = 2
REPETICIONES = 3
CALENTAMIENTO = 10
UMBRAL_CARGA = 3.0
RESOLUCION_EN_VIVO = (1280, 720)                    # --ancho 1280 --alto 720 (N-65)
FPS_CAMARA = 30.0
VENTANA_SEG = 2.5
CLIP_CONTROL_N65 = "0030_0000_0003"                 # N-65: «hola», 0,96 con el modelo final


# ---------------------------------------------------------------------------
# Muestra
# ---------------------------------------------------------------------------

def partes(stem: str) -> tuple[int, int, int]:
    """(seña, señante, repetición) desde '0005_0002_0001'."""
    a, b, c = Path(stem).stem.split("_")[:3]
    return int(a), int(b), int(c)


def seleccionar_muestra(nombres, por_clase: int = POR_CLASE, semilla: int = SEMILLA) -> list[str]:
    """Muestra estratificada por seña y equilibrada por señante (ver docstring del módulo)."""
    por_sena: dict[int, dict[int, list[str]]] = {}
    for nombre in nombres:
        stem = Path(nombre).stem
        try:
            s, v, _ = partes(stem)
        except (ValueError, IndexError):
            continue
        por_sena.setdefault(s, {}).setdefault(v, []).append(stem)
    rng = random.Random(semilla)
    elegidos = []
    for s in sorted(por_sena):
        senantes = sorted(por_sena[s])
        k = len(senantes)
        tomados = []
        for j in range(min(por_clase, k)):
            v = senantes[(por_clase * s + j) % k]
            if v in tomados:                      # con menos señantes que por_clase no se repite
                continue
            tomados.append(v)
            elegidos.append(rng.choice(sorted(por_sena[s][v])))
    return sorted(elegidos)


def describir_muestra(stems: list[str]) -> dict:
    sen = Counter(partes(s)[1] for s in stems)
    cla = Counter(partes(s)[0] for s in stems)
    return {"n": len(stems), "por_senante": {f"{k:04d}": v for k, v in sorted(sen.items())},
            "senas": len(cla), "por_sena_min": min(cla.values()) if cla else 0,
            "por_sena_max": max(cla.values()) if cla else 0, "videos": list(stems)}


# ---------------------------------------------------------------------------
# Puntos ya extraídos → Extraccion por fotograma
# ---------------------------------------------------------------------------

def cargar_puntos(stem: str, carpeta: Path = EXTRAIDOS) -> np.ndarray:
    """(T, 225) float32 del corpus re-extraído (pose, mano izq., mano der.; ceros sin detección). Usa el
    mismo cargador que el preprocesamiento del corpus; solo lee."""
    from src import preprocess_lsc50 as pp
    _, dirs = pp.resolver_directorios(Path(carpeta))
    datos = pp.load_single_sample(stem, dirs=dirs, rostro="ninguno")
    if datos is None:
        raise RuntimeError(f"no se pudieron cargar los puntos de {stem} en {carpeta}")
    return np.asarray(datos, dtype=np.float32)


def extracciones_desde_puntos(puntos: np.ndarray) -> list[Extraccion]:
    """Una Extraccion por fotograma con las banderas que daría el extractor: pose detectada si el bloque
    de pose no es todo cero; cada mano cuenta si su bloque no es todo cero (así llegan del extractor)."""
    salida = []
    for fila in np.asarray(puntos, dtype=np.float32):
        pose = bool(np.any(fila[IDX_POSE] != 0))
        manos = int(np.any(fila[IDX_MANO_IZQ] != 0)) + int(np.any(fila[IDX_MANO_DER] != 0))
        salida.append(Extraccion(keypoints=fila.copy(), pose_detectada=pose, n_manos=manos,
                                 rostro=None, holistic_result=None))
    return salida


class ExtractorPregrabado:
    """Sustituye a LandmarkExtractor: devuelve, en orden, las Extraccion cargadas (el argumento del
    fotograma se ignora). Su costo es de microsegundos: la etapa «extracción» no se mide aquí."""

    descripcion = "pregrabado (data/extracted/LSC50_holistic; sin cámara)"

    def __init__(self):
        self._cola: list[Extraccion] = []
        self._i = 0

    def cargar(self, extracciones: list[Extraccion]) -> None:
        self._cola = list(extracciones)
        self._i = 0

    def extract(self, _frame):
        if self._i >= len(self._cola):
            raise IndexError("ExtractorPregrabado sin fotogramas: la ventana pidió más de los cargados")
        ext = self._cola[self._i]
        self._i += 1
        return ext

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Voz sin altavoces
# ---------------------------------------------------------------------------

def silenciar_tts(tts) -> None:
    """La reproducción de un TTSModule pasa a ser «borrar el archivo»: la síntesis y la orden (el hilo de
    play_prepared_async) siguen siendo las reales, pero no suena nada. speak() y _play_audio() también se
    anulan por si algún camino las alcanzara. Cuenta las locuciones silenciadas."""
    tts.locuciones_silenciadas = 0

    def _reproducir_sin_sonido(archivo, texto):
        tts.locuciones_silenciadas += 1
        tts._borrar_audio(archivo)

    def _speak_sin_sonido(texto):
        tts.locuciones_silenciadas += 1

    tts._reproducir = _reproducir_sin_sonido
    tts.speak = _speak_sin_sonido
    tts._play_audio = lambda _ruta: True


VOCES = ("offline", "pregrabada")
FRASES_XTTS = 20
DISPOSITIVOS_XTTS = ("mps", "cpu")


def crear_tts(cache: bool, voz: str = "offline"):
    """TTSModule silenciado. «pregrabada» sin servicio XTTS (python_voz=""): se mide la voz pregrabada sola."""
    from src.tts import TTSModule
    if voz == "pregrabada":
        tts = TTSModule(mode="pregrabada", verbose=False, cache=cache, python_voz="")
    else:
        tts = TTSModule(mode="offline", verbose=False, cache=cache)
    silenciar_tts(tts)
    return tts


def frases_fuera_de_inventario(n: int = FRASES_XTTS, semilla: int = SEMILLA) -> list[dict]:
    """n frases que el prototipo puede decir y que NO están pregrabadas: las 4 variantes en femenino del
    inventario, salidas del conversor «reglas» fuera del inventario y respaldos del modo inventario (lista de
    glosas). Mismo n y semilla → misma lista."""
    from src.conversor_glosas import ConversorGlosas, conjuntos
    from src import voz_pregrabada as vp
    pregrabadas = set(vp.textos_inventario())
    filas = [{"tipo": "variante_femenino", "frase": t} for v in vp.variantes_de_genero().values() for t in v]
    filas = list({f["frase"]: f for f in filas}.values())
    reglas, inv = ConversorGlosas(), ConversorGlosas(estricto=True)
    cand_reglas, cand_resp = [], []
    for glosas in conjuntos((2, 3)):
        e = reglas.explicar(glosas)
        if not e["respaldo"] and e["frase"] not in pregrabadas:
            cand_reglas.append({"tipo": "reglas", "frase": e["frase"], "glosas": glosas})
        r = inv.explicar(glosas)
        if r["respaldo"]:
            cand_resp.append({"tipo": "respaldo_inventario", "frase": r["frase"], "glosas": glosas})
    rng = random.Random(semilla)
    resto = n - len(filas)
    filas += rng.sample(cand_reglas, (resto + 1) // 2) + rng.sample(cand_resp, resto // 2)
    assert not any(f["frase"] in pregrabadas for f in filas)
    return filas[:n]


def medir_xtts(python: str, dispositivos=DISPOSITIVOS_XTTS, n_frases: int = FRASES_XTTS, repeticiones: int = 3,
               semilla: int = SEMILLA, tope_arranque_seg: float = 300.0) -> dict:
    """Respaldo XTTS-v2 con el servicio real, por dispositivo: carga y tiempo por frase (sin cache)."""
    from src.tts import TTSModule
    frases = frases_fuera_de_inventario(n_frases, semilla)
    out = {"frases": frases, "dispositivos": {}}
    for disp in dispositivos:
        espera = leer_carga()
        _imprimir(f"XTTS en {disp}: cargando (carga 1 min {espera['1min']})")
        t0 = time.perf_counter()
        tts = TTSModule(mode="pregrabada", verbose=False, cache=False, python_voz=python, xtts_dispositivo=disp,
                        xtts_tope_arranque_seg=tope_arranque_seg)
        silenciar_tts(tts)
        listo = tts.esperar_servicio_voz(tope_arranque_seg)
        res = {"listo": listo, "espera_hasta_listo_s": round(time.perf_counter() - t0, 2),
               "info_servicio": dict(tts.xtts.info_listo or {}) if tts.xtts else None,
               "carga_al_empezar": espera, "filas": []}
        if listo:
            for rep in range(repeticiones):
                for f in frases:
                    tts.prepare(f["frase"])
                    preparado = bool(tts.esperar_preparado(60.0))
                    resp = tts.xtts.ultima_respuesta or {}
                    res["filas"].append({"repeticion": rep + 1, "frase": f["frase"], "preparado": preparado,
                                         "fuente": tts.ultima_fuente,
                                         "ms_prepare": (tts.ultima_sintesis_ns or 0) / 1e6 if preparado else None,
                                         "ms_servidor": resp.get("ms"), "ms_cliente": resp.get("ms_cliente"),
                                         "duracion_audio_s": resp.get("duracion_s")})
                    tts.play_prepared_async()            # silenciada: solo borra el archivo
            filas_x = [r for r in res["filas"] if r["fuente"] == "xtts"]
            res["fuentes"] = dict(Counter(str(r["fuente"]) for r in res["filas"]))
            res["ms_prepare"] = resumen_ms([r["ms_prepare"] for r in filas_x])
            res["ms_servidor"] = resumen_ms([r["ms_servidor"] for r in filas_x])
            res["duracion_audio_s"] = resumen_ms([r["duracion_audio_s"] for r in filas_x])
            res["factor_tiempo_real_mediana"] = (round(res["ms_servidor"]["mediana"] / 1000 / res["duracion_audio_s"]["mediana"], 3)
                                                 if res["ms_servidor"] and res["duracion_audio_s"] else None)
            res["por_repeticion_mediana_ms_prepare"] = [
                resumen_ms([r["ms_prepare"] for r in filas_x if r["repeticion"] == k + 1])["mediana"]
                for k in range(repeticiones)]
        res["estado_servicio"] = tts.xtts.estado() if tts.xtts else None
        res["carga_al_terminar"] = leer_carga()
        tts.cleanup()
        out["dispositivos"][disp] = res
    return out


# ---------------------------------------------------------------------------
# Estadísticas
# ---------------------------------------------------------------------------

def resumen_ms(valores) -> dict | None:
    """n, media, DE (ddof=1), mediana, p90, p99, mín y máx en ms (entrada en ms). None si no hay datos."""
    a = np.asarray([v for v in valores if v is not None], dtype=np.float64)
    if a.size == 0:
        return None
    return {"n": int(a.size), "media": float(a.mean()), "de": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "mediana": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "min": float(a.min()), "max": float(a.max())}


def variacion(resumenes: list[dict | None], clave: str = "mediana") -> dict | None:
    """Valores de `clave` en cada repetición, su rango absoluto y relativo (rango / media de los valores)."""
    vals = [r[clave] for r in resumenes if r is not None and r.get(clave) is not None]
    if not vals:
        return None
    media = float(np.mean(vals))
    rango = float(max(vals) - min(vals))
    return {"valores": [float(v) for v in vals], "rango": rango,
            "rango_relativo": (rango / media) if media else None}


# ---------------------------------------------------------------------------
# Carga del equipo
# ---------------------------------------------------------------------------

def leer_carga(leer=os.getloadavg) -> dict:
    uno, cinco, quince = leer()
    return {"1min": round(float(uno), 2), "5min": round(float(cinco), 2), "15min": round(float(quince), 2)}


def procesos_top(n: int = 6) -> list[dict]:
    """Procesos con más CPU (nombre del ejecutable, sin la ruta). Lista vacía si ps falla."""
    try:
        salida = subprocess.run(["ps", "-Ao", "pcpu=,comm=", "-r"], capture_output=True, text=True,
                                timeout=10).stdout
    except Exception:  # noqa: BLE001
        return []
    filas = []
    for linea in salida.splitlines():
        linea = linea.strip()
        if not linea:
            continue
        cpu, _, comm = linea.partition(" ")
        try:
            filas.append({"cpu_pct": float(cpu), "proceso": Path(comm.strip()).name})
        except ValueError:
            continue
        if len(filas) >= n:
            break
    return filas


class PresupuestoEspera:
    """Tope total de espera por carga (s) compartido por todos los bloques."""

    def __init__(self, total_s: float):
        self.total_s = float(total_s)
        self.usado_s = 0.0

    @property
    def restante_s(self) -> float:
        return max(0.0, self.total_s - self.usado_s)


def esperar_carga(umbral: float, max_bloque_s: float, presupuesto: PresupuestoEspera,
                  intervalo_s: float = 15.0, leer=os.getloadavg, dormir=time.sleep,
                  reloj=time.monotonic) -> dict:
    """Espera a que la carga de 1 min baje de `umbral`, con tope por bloque y tope total. Nunca falla:
    si no baja, devuelve se_alcanzo=False y el bloque se mide igual."""
    inicial = leer_carga(leer)
    t_ini = reloj()
    tope = min(max_bloque_s, presupuesto.restante_s)
    actual = inicial
    while actual["1min"] >= umbral and (reloj() - t_ini) < tope:
        dormir(min(intervalo_s, max(0.0, tope - (reloj() - t_ini))))
        actual = leer_carga(leer)
    esperado = reloj() - t_ini
    presupuesto.usado_s += esperado
    return {"umbral": umbral, "carga_al_pedir": inicial, "carga_al_empezar": actual,
            "esperado_s": round(esperado, 1), "se_alcanzo": actual["1min"] < umbral,
            "procesos_top": procesos_top()}


# ---------------------------------------------------------------------------
# Bloque post-cierre
# ---------------------------------------------------------------------------

def crear_traductor(modelo, conversor: str, tts, extractor) -> LSCTranslator:
    """LSCTranslator como lo arma realtime.py (umbral del modelo, ventana, mínimos, regla de mano ausente
    de la procedencia), con el extractor pregrabado, sin JSONL y con la voz silenciada."""
    return LSCTranslator(model_path=str(modelo), tts=tts, umbral=None, ventana_frames=0, ventana_seg=VENTANA_SEG,
                         min_frames=15, max_sin_manos=0.5, modo_oracion=False, registro=False,
                         extractor=extractor, grammar_mapper=trmod.construir_conversor(conversor),
                         calentar=True, tts_precalentar=True)


def correr_ensayo(tr: LSCTranslator, extractor: ExtractorPregrabado, extracciones: list[Extraccion]) -> dict:
    """Una ventana = el clip completo. Devuelve el registro del traductor (mismo formato que el JSONL)."""
    extractor.cargar(extracciones)
    tr.ventana_frames = len(extracciones)
    if tr.abrir_ventana() is None:
        raise RuntimeError("había una ventana abierta")
    registro = None
    for i in range(len(extracciones)):
        res = tr.procesar_fotograma(i)
        if res.registro is not None:
            registro = res.registro
            break
    if registro is None:
        raise RuntimeError("la ventana no se cerró con los fotogramas del clip")
    return registro


def fila_ensayo(stem: str, reg: dict) -> dict:
    """Lo que se guarda de cada ensayo, en ms."""
    et = reg["etapas_ns"]
    e_n = reg["e"][-1] if reg.get("e") else None
    ms = lambda ns: None if ns is None else ns / 1e6  # noqa: E731
    return {
        "video": stem, "N": reg["N"], "estado": reg["estado"], "clase": reg["clase"],
        "confianza": reg["confianza"], "enunciado": reg["enunciado"],
        "sintesis_desde_cache": reg.get("tts_sintesis_desde_cache"),
        "sintesis_antes_de_t5": reg.get("tts_sintesis_antes_de_t5"),
        "fuente": reg.get("tts_fuente"),
        "tiempo_sintesis_motor": ms(reg.get("tts_tiempo_sintesis_ns")),
        "preprocesamiento": ms(et.get("preprocesamiento")), "inferencia": ms(et.get("inferencia")),
        "conversion": ms(et.get("conversion")), "sintesis": ms(et.get("sintesis")),
        "post_cierre": ms(reg["t5"] - e_n) if (reg.get("t5") is not None and e_n is not None) else None,
        "L_resp_fuera_de_linea": ms(reg.get("L_resp_ns")),
    }


def resumir_filas(filas: list[dict]) -> dict:
    return {m: resumen_ms([f[m] for f in filas]) for m in MAGNITUDES_POST}


# ---------------------------------------------------------------------------
# Bloque turnos: conversión y síntesis sin cache de glosas solas y secuencias del inventario
# ---------------------------------------------------------------------------

def secuencias_inventario() -> list[list[str]]:
    from src.conversor_glosas import INVENTARIO, orden_presentacion
    return [orden_presentacion(list(s["glosas"])) for s in INVENTARIO]


def medir_turnos(tr: LSCTranslator, tts_sin_cache, glosas_solas: list[str], secuencias: list[list[str]],
                 reloj=time.perf_counter_ns) -> dict:
    """Para cada lista de glosas: conversión con el conversor del traductor (tr._convertir, la misma
    llamada que fin_de_turno) y síntesis sin cache (prepare → esperar_preparado) + orden."""
    filas = []
    for tipo, listas in (("sola", [[g] for g in glosas_solas]), ("secuencia", secuencias)):
        for glosas in listas:
            t_a = reloj()
            frase, _ = tr._convertir(list(glosas))
            t_b = reloj()
            sint = orden = None
            listo = None
            if frase:
                tts_sin_cache.prepare(frase)
                listo = bool(tts_sin_cache.esperar_preparado(15.0))
                t_c = reloj()
                tts_sin_cache.play_prepared_async()
                t_d = reloj()
                sint, orden = (t_c - t_b) / 1e6, (t_d - t_c) / 1e6
            filas.append({"tipo": tipo, "glosas": list(glosas), "frase": frase, "conversion": (t_b - t_a) / 1e6,
                          "sintesis": sint, "orden": orden, "sintetizada": listo})
    out = {}
    for tipo in ("sola", "secuencia"):
        fs = [f for f in filas if f["tipo"] == tipo]
        out[tipo] = {"n": len(fs), "con_frase": sum(1 for f in fs if f["frase"]),
                     "conversion": resumen_ms([f["conversion"] for f in fs]),
                     "sintesis_sin_cache": resumen_ms([f["sintesis"] for f in fs]),
                     "orden": resumen_ms([f["orden"] for f in fs]),
                     "fallos_sintesis": sum(1 for f in fs if f["frase"] and not f["sintetizada"])}
    return {"resumen": out, "filas": filas}


# ---------------------------------------------------------------------------
# Bloque extractor
# ---------------------------------------------------------------------------

def medir_extractor_video(ruta: Path, extractor, tam: tuple[int, int] | None = None,
                          reloj=time.perf_counter_ns, max_fotogramas: int | None = None) -> dict:
    """Costo por fotograma e_i − c_i como en vivo: c_i tras tener el fotograma, BGR→RGB y extract().
    Los fotogramas solo existen en memoria."""
    import cv2
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise RuntimeError(f"no se pudo abrir {ruta.name}")
    costos, con_manos, n = [], 0, 0
    try:
        while max_fotogramas is None or n < max_fotogramas:
            ok, frame = cap.read()
            if not ok:
                break
            if tam is not None:
                frame = cv2.resize(frame, tam, interpolation=cv2.INTER_AREA)
            c = reloj()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ext = extractor.extract(rgb)
            e = reloj()
            costos.append((e - c) / 1e6)
            con_manos += int(ext.n_manos > 0)
            n += 1
    finally:
        cap.release()
    return {"video": ruta.stem, "fotogramas": n, "con_manos": con_manos, "costos_ms": costos}


def resumen_extractor(por_video: list[dict], fps_camara: float = FPS_CAMARA, ventana_seg: float = VENTANA_SEG) -> dict:
    costos = [c for v in por_video for c in v["costos_ms"]]
    r = resumen_ms(costos)
    presupuesto = 1000.0 / fps_camara
    n_ventana = int(round(ventana_seg * fps_camara))
    out = {"costo_ms": r, "fotogramas": len(costos),
           "presupuesto_por_fotograma_ms": presupuesto,
           "fraccion_sobre_presupuesto": (sum(1 for c in costos if c > presupuesto) / len(costos)) if costos else None,
           "tasa_max_mediana_fps": (1000.0 / r["mediana"]) if r else None,
           "tasa_max_p99_fps": (1000.0 / r["p99"]) if r else None,
           "fotogramas_con_manos": sum(v["con_manos"] for v in por_video)}
    if r:
        # Tiempo de extractor que consumiría una ventana de n_ventana fotogramas: media por video × N.
        por_ventana = [n_ventana * float(np.mean(v["costos_ms"])) for v in por_video if v["costos_ms"]]
        out["ventana"] = {"fotogramas": n_ventana, "duracion_ms": ventana_seg * 1000.0,
                          "extractor_ms_por_ventana": resumen_ms(por_ventana),
                          "fraccion_de_la_ventana_mediana": float(np.median(por_ventana)) / (ventana_seg * 1000.0)}
    return out


# ---------------------------------------------------------------------------
# Caracterización de los registros en vivo (sin concluir de dónde vienen)
# ---------------------------------------------------------------------------

def _resumen_simple(valores) -> dict | None:
    a = np.asarray([v for v in valores if v is not None], dtype=np.float64)
    if a.size == 0:
        return None
    return {"n": int(a.size), "mediana": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max()),
            "de": float(a.std(ddof=1)) if a.size > 1 else 0.0}


def caracterizar_sesion(registros: list[dict]) -> dict:
    """Rasgos de una sesión del JSONL que ayudan a juzgar cómo se obtuvo (no lo deciden)."""
    ensayos = [r for r in registros if r.get("tipo") == "ensayo"]
    turnos = [r for r in registros if r.get("tipo") == "turno"]
    horas = sorted(r["fecha_hora"] for r in registros if r.get("fecha_hora"))
    dur = None
    if len(horas) >= 2:
        dur = (dt.datetime.fromisoformat(horas[-1]) - dt.datetime.fromisoformat(horas[0])).total_seconds()
    intervalos = []
    for r in ensayos:
        c = r.get("c") or []
        intervalos += [(b - a) / 1e6 for a, b in zip(c, c[1:])]
    t0 = sorted(r["t0"] for r in ensayos if r.get("t0") is not None)
    pausas = [(b - a) / 1e9 for a, b in zip(t0, t0[1:])]
    n_ok = [r for r in ensayos if r.get("N")]
    aceptadas = [r for r in ensayos if r.get("estado") == "aceptada"]
    claves = set().union(*(r.keys() for r in registros)) if registros else set()
    return {
        "sesion": registros[0].get("sesion") if registros else None,
        "primer_registro": horas[0] if horas else None, "ultimo_registro": horas[-1] if horas else None,
        "duracion_s": dur, "ensayos": len(ensayos), "turnos": len(turnos),
        "modelo": sorted({str(r.get("modelo")) for r in ensayos}),
        "mano_ausente": sorted({str(r.get("mano_ausente")) for r in ensayos}),
        "umbral": sorted({(r.get("parametros") or {}).get("umbral") for r in ensayos}, key=str),
        "tts_modo": sorted({str(r.get("tts_modo")) for r in ensayos}),
        "conversor_3a_en_registro": any("conversion" in r for r in registros),
        "campo_fuente_de_video": any(k in claves for k in ("fuente", "camara", "video", "fuente_video")),
        "modos": dict(Counter(r.get("modo") for r in ensayos)),
        "estados": dict(Counter(r.get("estado") for r in ensayos)),
        "cierre": dict(Counter(r.get("cierre") for r in ensayos)),
        "N": _resumen_simple([r.get("N") for r in ensayos]),
        "ventanas_con_N_distinto_de_75": sum(1 for r in ensayos if r.get("N") not in (None, 75)),
        "fps_medida_antes": _resumen_simple([(r.get("parametros") or {}).get("fps_medida_antes") for r in ensayos]),
        "tasa_efectiva_fps": _resumen_simple([r.get("tasa_efectiva_fps") for r in ensayos]),
        "intervalo_entre_fotogramas_ms": _resumen_simple(intervalos),
        "costo_extractor_mediana_ms": _resumen_simple([r["costo_extractor_ns"]["mediana"] / 1e6 for r in ensayos
                                                       if (r.get("costo_extractor_ns") or {}).get("mediana") is not None]),
        "proporcion_sin_manos": _resumen_simple([r.get("proporcion_sin_manos") for r in ensayos]),
        "proporcion_dos_manos": _resumen_simple([r["frames_con_dos_manos"] / r["N"] for r in n_ok
                                                 if r.get("frames_con_dos_manos") is not None]),
        "ventanas_sin_ninguna_con_dos_manos": sum(1 for r in n_ok if r.get("frames_con_dos_manos") == 0),
        "frames_sin_pose_total": sum(int(r.get("frames_sin_pose") or 0) for r in ensayos),
        "pausa_entre_ventanas_s": _resumen_simple(pausas),
        "confianza_aceptadas": _resumen_simple([r.get("confianza") for r in aceptadas]),
        "glosas_aceptadas": dict(Counter(r.get("clase") for r in aceptadas)),
        "enunciados_aislado": [r.get("enunciado") for r in aceptadas if r.get("enunciado")],
        "turnos_detalle": [{"glosas": t.get("glosas"), "enunciado": t.get("enunciado"),
                            "sintesis_ms": (t.get("sintesis_ns") or 0) / 1e6,
                            "desde_cache": t.get("tts_sintesis_desde_cache")} for t in turnos],
    }


def caracterizar_registros(rutas: list[Path]) -> dict:
    """Por archivo: resumen de latencia_resumen.resumir (las mismas cifras de su tabla) y rasgos por sesión."""
    out = {}
    for ruta in rutas:
        regs = lr.leer_registros([str(ruta)])
        res = lr.resumir(regs)
        por_sesion = {}
        for s in res["sesiones"]:
            por_sesion[s] = caracterizar_sesion([r for r in regs if r.get("sesion") == s])
        out[ruta.name] = {
            "latencia_resumen": {k: res.get(k) for k in (
                "n_registros", "n_ensayos", "n_turnos", "sesiones", "modelos", "contadores", "tasa_rechazo",
                "por_modo", "N_fotogramas", "magnitudes_ms", "ventanas_inferidas", "por_sintesis_antes_de_t5",
                "por_sintesis_desde_cache", "costo_extractor_por_fotograma_ms", "tasa_efectiva_fps", "turnos",
                "umbrales", "advertencias")},
            "sesiones": por_sesion,
        }
    return out


# ---------------------------------------------------------------------------
# Contexto
# ---------------------------------------------------------------------------

def sha256(ruta: Path) -> str | None:
    try:
        return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()
    except OSError:
        return None


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT_DIR))
    except (ValueError, OSError):
        return str(p)


def contexto_equipo() -> dict:
    def sysctl(clave):
        try:
            return subprocess.run(["sysctl", "-n", clave], capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    versiones = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("tensorflow", "mediapipe", "cv2"):
        try:
            versiones[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versiones[mod] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    mem = sysctl("hw.memsize")
    return {"plataforma": platform.platform(), "cpu": sysctl("machdep.cpu.brand_string"),
            "nucleos": os.cpu_count(), "memoria_gb": round(int(mem) / 2**30, 1) if mem and mem.isdigit() else None,
            "versiones": versiones, "git_commit": commit, "solo_cpu": True}


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def _imprimir(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def medir(args) -> dict:
    nombres = sorted(p.stem for p in (EXTRAIDOS / "BODY_LANDMARKS").glob("*.csv"))
    muestra = seleccionar_muestra(nombres, args.por_clase, args.semilla)
    _imprimir(f"muestra: {len(muestra)} videos")
    clips = {s: extracciones_desde_puntos(cargar_puntos(s)) for s in muestra}
    control = extracciones_desde_puntos(cargar_puntos(CLIP_CONTROL_N65))
    presupuesto = PresupuestoEspera(args.max_espera_total_min * 60)
    max_bloque = args.max_espera_bloque_min * 60

    # Traductores: el modelo se carga UNA vez por configuración.
    extractor = ExtractorPregrabado()
    configs = {}
    voz = getattr(args, "voz", "offline")
    elegidas = getattr(args, "configuraciones", None) or list(CONFIGURACIONES)
    for nombre, cfg in CONFIGURACIONES.items():
        if nombre not in elegidas:
            continue
        tts_c, tts_s = crear_tts(cache=True, voz=voz), crear_tts(cache=False, voz=voz)
        _imprimir(f"cargando {nombre}")
        tr = crear_traductor(cfg["modelo"], cfg["conversor"], tts_c, extractor)
        tts_c.esperar_precalentado(120)
        configs[nombre] = {"tr": tr, "tts": {"cache": tts_c, "sin_cache": tts_s}, "cfg": cfg}

    resultados = {n: {"modelo": _rel(c["cfg"]["modelo"]), "sha256_modelo": sha256(c["cfg"]["modelo"]),
                      "conversor": c["cfg"]["conversor"], "umbral": c["tr"].umbral,
                      "umbral_origen": c["tr"].umbral_origen, "mano_ausente": c["tr"].mano_ausente,
                      "representacion": c["tr"].representacion, "n_rasgos": c["tr"].n_rasgos,
                      "voz": {"modo": c["tts"]["cache"].mode, "nombre": c["tts"]["cache"].voice_name,
                              "ejecutor": c["tts"]["cache"].ejecutor_offline,
                              "precalentadas": c["tts"]["cache"].precalentadas,
                              "indice": (_rel(c["tts"]["cache"].indice.ruta)
                                         if getattr(c["tts"]["cache"], "indice", None) is not None else None)},
                      "calentamiento_descartado": [], "control_n65": None,
                      "repeticiones": {cond: [] for cond in CONDICIONES_VOZ}, "turnos": []}
                  for n, c in configs.items()}

    # Calentamiento descartado y control de N-65 (fuera de las repeticiones).
    for nombre, c in configs.items():
        tr = c["tr"]
        for stem in muestra[:args.calentamiento]:
            reg = correr_ensayo(tr, extractor, clips[stem])
            resultados[nombre]["calentamiento_descartado"].append(fila_ensayo(stem, reg)["inferencia"])
        reg = correr_ensayo(tr, extractor, control)
        resultados[nombre]["control_n65"] = {"video": CLIP_CONTROL_N65, "estado": reg["estado"],
                                             "clase": reg["clase"], "confianza": reg["confianza"]}

    bloques = []
    secuencias = secuencias_inventario()
    from src.grammar_mapper import SIGN_VOCABULARY
    glosas_solas = [SIGN_VOCABULARY[i] for i in sorted(SIGN_VOCABULARY)]
    for rep in range(args.repeticiones):
        orden = list(configs) if rep % 2 == 0 else list(reversed(list(configs)))
        espera = esperar_carga(args.umbral_carga, max_bloque, presupuesto)
        _imprimir(f"rep {rep + 1} post-cierre; carga {espera['carga_al_empezar']} (esperó {espera['esperado_s']} s)")
        t_ini = time.time()
        for nombre in orden:
            c = configs[nombre]
            for cond in CONDICIONES_VOZ:
                c["tr"].tts = c["tts"][cond]
                filas = [fila_ensayo(s, correr_ensayo(c["tr"], extractor, clips[s])) for s in muestra]
                resultados[nombre]["repeticiones"][cond].append({"repeticion": rep + 1, "filas": filas,
                                                                 "resumen": resumir_filas(filas)})
            c["tr"].tts = c["tts"]["cache"]
            turnos = medir_turnos(c["tr"], c["tts"]["sin_cache"], glosas_solas, secuencias)
            resultados[nombre]["turnos"].append({"repeticion": rep + 1, **turnos})
        bloques.append({"bloque": "post_cierre_y_turnos", "repeticion": rep + 1, "orden": orden, **espera,
                        "carga_al_terminar": leer_carga(), "duracion_s": round(time.time() - t_ini, 1)})

    for c in configs.values():
        c["tr"].close()
        c["tts"]["sin_cache"].cleanup()

    extractor_res = None
    if not args.sin_extractor:
        extractor_res = {"resoluciones": {}, "videos": list(muestra)}
        mp_paths = trmod._ensure_models_downloaded()
        for rep in range(args.repeticiones):
            espera = esperar_carga(args.umbral_carga, max_bloque, presupuesto)
            _imprimir(f"rep {rep + 1} extractor; carga {espera['carga_al_empezar']} (esperó {espera['esperado_s']} s)")
            t_ini = time.time()
            for etiqueta, tam in (("nativa_1920x1080", None), ("en_vivo_1280x720", RESOLUCION_EN_VIVO)):
                ext = trmod.LandmarkExtractor(mp_paths)
                medir_extractor_video(VIDEOS / f"{muestra[0]}.avi", ext, tam, max_fotogramas=10)   # calentamiento
                por_video = [medir_extractor_video(VIDEOS / f"{s}.avi", ext, tam) for s in muestra]
                ext.close()
                extractor_res["resoluciones"].setdefault(etiqueta, []).append(
                    {"repeticion": rep + 1, "resumen": resumen_extractor(por_video),
                     "mediana_por_video_ms": {v["video"]: float(np.median(v["costos_ms"])) for v in por_video
                                              if v["costos_ms"]}})
            bloques.append({"bloque": "extractor", "repeticion": rep + 1, **espera,
                            "carga_al_terminar": leer_carga(), "duracion_s": round(time.time() - t_ini, 1)})

    return {"muestra": describir_muestra(muestra), "post_cierre": resultados, "extractor": extractor_res,
            "bloques": bloques, "espera_total_s": round(presupuesto.usado_s, 1)}


def agregar(datos: dict) -> dict:
    """Resumen agregado: por configuración y condición de voz, cada magnitud sobre las repeticiones juntas
    y su variación entre repeticiones (medianas y p90)."""
    agg = {}
    for nombre, r in datos["post_cierre"].items():
        agg[nombre] = {}
        for cond, reps in r["repeticiones"].items():
            filas = [f for rep in reps for f in rep["filas"]]
            agg[nombre][cond] = {
                "estados": dict(Counter(f["estado"] for f in filas)),
                "desde_cache": dict(Counter(str(f["sintesis_desde_cache"]) for f in filas)),
                "fuentes": dict(Counter(str(f.get("fuente")) for f in filas)),
                "tiempo_sintesis_motor": resumen_ms([f.get("tiempo_sintesis_motor") for f in filas]),
                "glosas_distintas_emitidas": len({f["clase"] for f in filas if f["estado"] == "aceptada"}),
                "N": resumen_ms([f["N"] for f in filas]),
                "magnitudes": resumir_filas(filas),
                "variacion_mediana": {m: variacion([rep["resumen"][m] for rep in reps], "mediana") for m in MAGNITUDES_POST},
                "variacion_p90": {m: variacion([rep["resumen"][m] for rep in reps], "p90") for m in MAGNITUDES_POST},
            }
        tur = r["turnos"]
        agg[nombre]["turnos"] = {
            tipo: {"conversion_mediana": variacion([t["resumen"][tipo]["conversion"] for t in tur], "mediana"),
                   "sintesis_sin_cache_mediana": variacion([t["resumen"][tipo]["sintesis_sin_cache"] for t in tur], "mediana"),
                   "sintesis_sin_cache_p90": variacion([t["resumen"][tipo]["sintesis_sin_cache"] for t in tur], "p90"),
                   "sintesis_sin_cache_max": variacion([t["resumen"][tipo]["sintesis_sin_cache"] for t in tur], "max"),
                   "con_frase": tur[0]["resumen"][tipo]["con_frase"] if tur else None,
                   "n": tur[0]["resumen"][tipo]["n"] if tur else None}
            for tipo in ("sola", "secuencia")}
    if datos.get("extractor"):
        agg["extractor"] = {}
        for etiqueta, reps in datos["extractor"]["resoluciones"].items():
            agg["extractor"][etiqueta] = {
                "mediana_ms": variacion([r["resumen"]["costo_ms"] for r in reps], "mediana"),
                "p90_ms": variacion([r["resumen"]["costo_ms"] for r in reps], "p90"),
                "p99_ms": variacion([r["resumen"]["costo_ms"] for r in reps], "p99"),
                "max_ms": variacion([r["resumen"]["costo_ms"] for r in reps], "max"),
                "fraccion_sobre_33ms": [r["resumen"]["fraccion_sobre_presupuesto"] for r in reps],
                "fraccion_de_la_ventana_mediana": [r["resumen"]["ventana"]["fraccion_de_la_ventana_mediana"] for r in reps],
            }
    return agg


def separar_crudo(datos: dict) -> tuple[dict, dict]:
    """(informe, crudo): el informe lleva resúmenes; las filas por ensayo y por frase de turno van al crudo."""
    informe = json.loads(json.dumps(datos, default=str))
    crudo = {"inicio": datos.get("inicio"), "post_cierre": {}}
    for nombre, r in informe.get("post_cierre", {}).items():
        crudo["post_cierre"][nombre] = {"repeticiones": {}, "turnos": []}
        for cond, reps in r.get("repeticiones", {}).items():
            crudo["post_cierre"][nombre]["repeticiones"][cond] = [
                {"repeticion": rep["repeticion"], "filas": rep.pop("filas", None)} for rep in reps]
        for t in r.get("turnos", []):
            crudo["post_cierre"][nombre]["turnos"].append({"repeticion": t["repeticion"], "filas": t.pop("filas", None)})
    return informe, crudo


def parsear_argumentos(argv=None):
    p = argparse.ArgumentParser(description="Latencia fuera de línea tras el cierre de la ventana (4.c)")
    p.add_argument("--salida", type=Path, default=SALIDA)
    p.add_argument("--crudo", type=Path, default=CRUDO, help="JSON con las filas por ensayo y por frase")
    p.add_argument("--por-clase", type=int, default=POR_CLASE)
    p.add_argument("--semilla", type=int, default=SEMILLA)
    p.add_argument("--repeticiones", type=int, default=REPETICIONES)
    p.add_argument("--calentamiento", type=int, default=CALENTAMIENTO)
    p.add_argument("--umbral-carga", type=float, default=UMBRAL_CARGA)
    p.add_argument("--max-espera-bloque-min", type=float, default=3.0)
    p.add_argument("--max-espera-total-min", type=float, default=60.0)
    p.add_argument("--sin-extractor", action="store_true")
    p.add_argument("--solo-registros", action="store_true", help="solo caracteriza registros/*.jsonl (sin medir)")
    p.add_argument("--voz", choices=VOCES, default="offline",
                   help="voz del post-cierre y los turnos: offline (la de 4.c) o pregrabada (23-nov-2026)")
    p.add_argument("--configuraciones", nargs="+", choices=list(CONFIGURACIONES), default=None,
                   help="medir solo estas configuraciones (default: todas)")
    p.add_argument("--solo-xtts", action="store_true",
                   help="solo mide el respaldo XTTS-v2 con el servicio real (necesita --python-voz)")
    p.add_argument("--python-voz", default=None, help="intérprete con coqui-tts (default $TESIS_PYTHON_VOZ)")
    p.add_argument("--dispositivos", nargs="+", choices=list(DISPOSITIVOS_XTTS), default=list(DISPOSITIVOS_XTTS))
    p.add_argument("--frases-xtts", type=int, default=FRASES_XTTS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parsear_argumentos(argv)
    inicio = dt.datetime.now().isoformat(timespec="seconds")
    datos = {"descripcion": "Latencia fuera de línea tras el cierre de la ventana (actividad 4.c); "
                            "ver el docstring de src/latencia_fuera_de_linea.py",
             "inicio": inicio, "comando": "python " + " ".join([_rel(sys.argv[0])] + sys.argv[1:]),
             "equipo": contexto_equipo(), "carga_al_inicio": leer_carga(), "procesos_al_inicio": procesos_top(),
             "parametros": {"por_clase": args.por_clase, "semilla": args.semilla, "repeticiones": args.repeticiones,
                            "calentamiento": args.calentamiento, "umbral_carga": args.umbral_carga,
                            "max_espera_bloque_min": args.max_espera_bloque_min,
                            "max_espera_total_min": args.max_espera_total_min,
                            "resolucion_en_vivo": RESOLUCION_EN_VIVO, "fps_camara": FPS_CAMARA,
                            "ventana_seg": VENTANA_SEG}}
    datos["parametros"].update({"voz": args.voz, "configuraciones": args.configuraciones or list(CONFIGURACIONES)})
    if args.solo_xtts:
        from src.voz_xtts_servicio import python_voz_desde_entorno
        python = args.python_voz or python_voz_desde_entorno()
        if not python:
            print("ERROR: --solo-xtts necesita --python-voz o TESIS_PYTHON_VOZ")
            return 2
        datos["descripcion"] = ("Respaldo XTTS-v2 de la voz del prototipo con el servicio real (src/voz_xtts_servicio.py); "
                                "ver el docstring de src/latencia_fuera_de_linea.py y docs/VOZ.md")
        datos["parametros"].update({"dispositivos": args.dispositivos, "frases_xtts": args.frases_xtts})
        datos["xtts"] = medir_xtts(python, args.dispositivos, args.frases_xtts, args.repeticiones, args.semilla)
        datos["fin"] = dt.datetime.now().isoformat(timespec="seconds")
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        with open(args.salida, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=1)
        _imprimir(f"escrito {args.salida}")
        return 0
    datos["registros_en_vivo"] = caracterizar_registros(sorted(REGISTROS.glob("latencia_*.jsonl")))
    if not args.solo_registros:
        datos.update(medir(args))
        datos["agregado"] = agregar(datos)
    datos["fin"] = dt.datetime.now().isoformat(timespec="seconds")
    informe, crudo = separar_crudo(datos)
    informe["crudo"] = _rel(args.crudo) if not args.solo_registros else None
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(informe, f, ensure_ascii=False, indent=1)
    if not args.solo_registros:
        args.crudo.parent.mkdir(parents=True, exist_ok=True)
        with open(args.crudo, "w", encoding="utf-8") as f:
            json.dump(crudo, f, ensure_ascii=False)
    _imprimir(f"escrito {args.salida}" + ("" if args.solo_registros else f" y {args.crudo}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
