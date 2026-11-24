"""
Evaluación PUNTA A PUNTA de la cadena integrada del prototipo sobre los videos crudos del corpus, sin
personas en la sesión (aporte parcial al objetivo específico OE4, «evaluar el sistema integrado»).

QUÉ PREGUNTA RESUELVE. Las exactitudes de la tesis (LOSO 0,677 del modelo final, TCN + imputación + 252)
se calcularon con puntos característicos YA EXTRAÍDOS del corpus (data/extracted/LSC50_holistic, a
1920×1080) y con la cadena de entrenamiento. El prototipo en vivo hace otra cosa: recibe fotogramas de la
cámara a 1280×720, los pasa por LandmarkExtractor (src/translator.py), arma la ventana de 2,5 s, aplica
LSCTranslator._preprocesar (normalización, regla de mano ausente, columnas, imputación, 30 pasos,
velocidades), llama UNA vez a model.predict, aplica el umbral 0,68, convierte la glosa con el conversor
«inventario» y busca el enunciado en el índice de voz pregrabada. Este script ejecuta ESA cadena, con las
clases y funciones del prototipo, alimentada con los fotogramas de los videos COLOR_BODY del corpus en lugar
de la cámara, y mide cuánto acierta y en qué se aparta de la medida fuera de línea.

QUÉ ES IGUAL AL PROTOTIPO (src/realtime.py + src/translator.py, sin cambios en esos archivos).
  · Extractor: src.translator.LandmarkExtractor (HolisticLandmarker de D-05, delegado CPU, modo VIDEO,
    OPCIONES_HOLISTIC), con el .task de _ensure_models_downloaded (SHA256 comprobado).
  · Fotograma: cv2.VideoCapture.read → reducción en memoria a 1280×720 (la resolución que realtime.py pide a
    la cámara, --ancho 1280 --alto 720; INTER_AREA, como src/latencia_fuera_de_linea.py) → cv2.cvtColor
    BGR→RGB → LSCTranslator.procesar_fotograma. La cámara entrega 1280×720 de forma nativa; aquí se obtiene
    reduciendo el video de 1920×1080, que es lo más cercano posible con el corpus.
  · Ventana, estados y cadena: LSCTranslator.abrir_ventana / procesar_fotograma / _cerrar_ventana (mínimo de
    15 fotogramas, máximo 50 % sin manos o sin pose, aviso de bordes «aviso»), _preprocesar, model.predict,
    umbral, _convertir con construir_conversor("inventario") (D-15, el defecto de realtime.py).
  · Umbral: 0,68 (models/tcn_imputacion/modelo.umbral.json, N-75). El modelo desplegado lo lee solo, como en
    el prototipo; a los modelos LOSO (que no tienen ese archivo) se les pasa el mismo valor.
  · Voz: se busca el enunciado en models/voz_pregrabada/indice.json con IndiceVozPregrabada.buscar, la misma
    búsqueda que hace src/tts.py en modo «pregrabada». No se reproduce ningún audio.

QUÉ CAMBIA RESPECTO DEL PROTOTIPO EN VIVO (y por qué).
  · Fuente de fotogramas: el archivo de video en vez de la cámara. Ningún fotograma se escribe a disco: se
    decodifican, reducen y extraen en memoria y se descartan (§1.5).
  · Marcas de tiempo del modo VIDEO: LandmarkExtractor recibe un reloj inyectado (su parámetro «reloj») que
    devuelve el instante del fotograma en el video, round(i·1000/fps) ms, la misma convención que
    src/reextract_lsc50.py; en vivo el reloj es el real, que a la tasa de la cámara da lo mismo salvo la
    fluctuación. Un HolisticLandmarker NUEVO por video, como en la re-extracción del corpus (en vivo el
    extractor corre sin parar; el clip empieza en reposo, que hace las veces de los fotogramas previos).
  · El extractor del prototipo se ejecuta UNA vez por video; los puntos de cada fotograma se reproducen
    después para cada modelo (ExtractorPregrabado de src/latencia_fuera_de_linea.py), de modo que todos los
    modelos ven exactamente la misma extracción.
  · Tamaño de la ventana: el prototipo fija N = round(2,5 s × tasa medida del bucle). A 24 fps (la tasa del
    corpus) eso es N = 60; se fija ventana_frames = 60 en lugar de medir la tasa. Se eligen 60 fotogramas a
    24 fps (y no 75 fotogramas remuestreados a 30 fps) porque lo que importa es que la ventana cubra los
    mismos 2,5 s de movimiento: la red remuestrea cualquier ventana a 30 pasos, y fabricar fotogramas
    intermedios sería inventar imagen que la cámara no vio.

VENTANAS.
  clip     el clip completo (N = fotogramas del video): comparable con el LOSO fuera de línea.
  v2_5s    los primeros 2,5 s (60 fotogramas). Si el clip dura menos, se sigue alimentando el extractor con
           el último fotograma (la persona quieta en reposo, como seguiría viéndola la cámara hasta el cierre).
           Comparable con la variante «primeros_2_5s» de src/simular_ventana_25.py (relleno con el borde).
  v2_5s_sin_relleno  los primeros min(T, 60) fotogramas, sin relleno (solo en el JSON).

MODELOS.
  · LOSO de la configuración final: models/confirmacion_tcn_2026-10-03/T1_imputacion_252/semilla_{2026,2027,
    2028}/pliegue_{0..4}. Cada video se clasifica SOLO con los modelos del pliegue cuyo señante de prueba es
    el suyo (el pliegue k excluye al señante 000k; se verifica con data/partitions/loso_5_pliegues.json y con
    los nombres de prueba de probabilidades.npz). Su procedencia declara la regla de mano ausente con una
    frase larga («imputacion (...)»), que LSCTranslator no acepta; se le inyecta la misma procedencia con
    mano_ausente = "imputacion" y velocidades = True, que es lo que la frase dice.
  · Desplegado: models/tcn_imputacion/modelo.keras, SOLO sobre los 100 videos de prueba de la partición
    aleatoria (data/partitions/aleatoria_80_10_10.json, conjuntos.test), los únicos que no vio.

COMPARACIONES (para separar la cadena del extractor).
  almacenados   los puntos YA EXTRAÍDOS del corpus pasados por la MISMA cadena del prototipo (LSCTranslator
                con ExtractorPregrabado). Contra probabilidades.npz de cada corrida LOSO (clip completo) mide
                si _preprocesar + predict del prototipo reproducen la cadena de entrenamiento. Con la ventana
                de 2,5 s es la referencia fuera de línea de esa ventana.
  video_1280x720  la cadena punta a punta (lo que se reporta).
  video_1920x1080 (diagnóstico, --sin-nativa lo omite) el mismo recorrido sin reducir el fotograma: separa
                el efecto de la resolución del de la decodificación y el extractor.
  Concordancia = misma clase top-1 que la referencia fuera de línea para el mismo video y el mismo modelo.

DETERMINISMO. Sin aleatoriedad: orden fijo de videos (señantes 0000, 0004, 0001, 0002, 0003; dentro, orden
alfabético), extractor y TensorFlow en CPU. La caché opcional (--cache) guarda SOLO puntos característicos
y tiempos (nunca imágenes) para poder reanudar.

Uso:
    python src/evaluacion_punta_a_punta.py                       # 997 videos, las dos resoluciones
    python src/evaluacion_punta_a_punta.py --senantes 0000 0004  # solo esos señantes
    python src/evaluacion_punta_a_punta.py --limite 5 --salida /tmp/x.json   # prueba de humo
    python src/evaluacion_punta_a_punta.py --cache <carpeta>     # reanudable (puntos, no imágenes)
Salida: docs/EVALUACION_PUNTA_A_PUNTA.json y docs/EVALUACION_PUNTA_A_PUNTA.md (resúmenes) y
models/evaluacion_punta_a_punta_2026-11-23/crudo.json (filas por video, modelo, fuente y ventana; sin imágenes)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("GLOG_minloglevel", "2")

import numpy as np  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import translator as trmod  # noqa: E402
from src.translator import Extraccion, IDX_MANO_DER, IDX_MANO_IZQ, IDX_POSE, LSCTranslator  # noqa: E402
from src.latencia_fuera_de_linea import (  # noqa: E402
    ExtractorPregrabado, cargar_puntos, extracciones_desde_puntos, leer_carga, procesos_top, contexto_equipo,
)

VIDEOS = ROOT_DIR / "data" / "raw" / "LSC50" / "VIDEOS" / "COLOR_BODY"
CONFIRMACION = ROOT_DIR / "models" / "confirmacion_tcn_2026-10-03" / "T1_imputacion_252"
MODELO_FINAL = ROOT_DIR / "models" / "tcn_imputacion" / "modelo.keras"
PRED_FINAL = ROOT_DIR / "models" / "tcn_imputacion" / "evaluacion_modelo_seleccionado" / "predicciones.json"
UMBRAL_JSON = ROOT_DIR / "models" / "tcn_imputacion" / "modelo.umbral.json"
PART_LOSO = ROOT_DIR / "data" / "partitions" / "loso_5_pliegues.json"
PART_ALEATORIA = ROOT_DIR / "data" / "partitions" / "aleatoria_80_10_10.json"
SIMULACION_25 = ROOT_DIR / "docs" / "SIMULACION_VENTANA_25.json"
SALIDA = ROOT_DIR / "docs" / "EVALUACION_PUNTA_A_PUNTA.json"
CRUDO = ROOT_DIR / "models" / "evaluacion_punta_a_punta_2026-11-23" / "crudo.json"   # filas por video y modelo

SEMILLAS = (2026, 2027, 2028)
PLIEGUES = (0, 1, 2, 3, 4)
ORDEN_SENANTES = ("0000", "0004", "0001", "0002", "0003")
RESOLUCION_EN_VIVO = (1280, 720)       # realtime.py --ancho 1280 --alto 720 (defecto)
VENTANA_SEG = 2.5
FPS_CORPUS = 24.0
N_VENTANA = int(round(VENTANA_SEG * FPS_CORPUS))   # 60
MIN_FRAMES = 15                         # defectos de realtime.py
MAX_SIN_MANOS = 0.5
CONVERSOR = "inventario"
VENTANAS = ("clip", "v2_5s", "v2_5s_sin_relleno")
FUENTE_E2E = "video_1280x720"
FUENTE_NATIVA = "video_1920x1080"
FUENTE_ALMACENADA = "almacenados"


def _imprimir(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT_DIR))
    except (ValueError, OSError):
        return str(p)


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def _r(x, n: int = 4):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------------------
# Particiones y referencias fuera de línea
# ---------------------------------------------------------------------------

def cargar_particiones() -> dict:
    loso = json.loads(PART_LOSO.read_text(encoding="utf-8"))
    ale = json.loads(PART_ALEATORIA.read_text(encoding="utf-8"))
    excluidas = sorted(loso["excluidas"])
    assert sorted(ale["excluidas"]) == excluidas, "las dos particiones deben excluir los mismos duplicados"
    pliegue_de = {}
    for p in loso["pliegues"]:
        k, sen = int(p["pliegue"]), p["senante_test"]
        assert sen == f"{k:04d}", f"el pliegue {k} no excluye al señante 000{k}"
        for n in p["conjuntos"]["test"]:
            assert n.split("_")[1] == sen
            pliegue_de[n] = k
    return {"excluidas": excluidas, "pliegue_de": pliegue_de, "test_aleatoria": sorted(ale["conjuntos"]["test"])}


def cargar_referencias_loso() -> dict:
    """{(semilla, pliegue): {video: vector de 50 probabilidades}} desde probabilidades.npz (prueba)."""
    ref = {}
    for s in SEMILLAS:
        for k in PLIEGUES:
            z = np.load(CONFIRMACION / f"semilla_{s}" / f"pliegue_{k}" / "probabilidades.npz")
            nombres = [str(n) for n in z["nombres_test"]]
            assert all(n.split("_")[1] == f"{k:04d}" for n in nombres), "probabilidades.npz con otro señante"
            ref[(s, k)] = {n: z["test"][i].astype(np.float64) for i, n in enumerate(nombres)}
    return ref


def cargar_referencia_final() -> dict:
    """{video: {"predicha", "top3", "prob_predicha"}} del modelo desplegado sobre la prueba aleatoria."""
    filas = json.loads(PRED_FINAL.read_text(encoding="utf-8"))
    return {f["muestra"]: f for f in filas}


def procedencia_loso(ruta_modelo: Path) -> dict:
    """Procedencia del modelo LOSO normalizada a los valores que acepta LSCTranslator (ver docstring)."""
    p = json.loads((ruta_modelo.parent / "modelo.procedencia.json").read_text(encoding="utf-8"))
    assert str(p["mano_ausente"]).startswith("imputacion"), p["mano_ausente"]
    assert p["representacion"] == "manos" and int(p["n_rasgos_entrada"]) == 252 and int(p["seq_length"]) == 30
    p = dict(p)
    p["mano_ausente_original"] = p["mano_ausente"]
    p["mano_ausente"] = "imputacion"
    p["velocidades"] = True
    return p


# ---------------------------------------------------------------------------
# Traductores del prototipo
# ---------------------------------------------------------------------------

class ModeloConRegistro:
    """Envuelve el modelo ya cargado por LSCTranslator y guarda el vector de probabilidades de la ÚNICA llamada
    a predict de cada ventana (la del propio _inferir_y_emitir). No cambia la llamada ni su resultado."""

    def __init__(self, modelo):
        self._modelo = modelo
        self.ultima = None

    @property
    def input_shape(self):
        return self._modelo.input_shape

    def predict(self, X, verbose=0):
        p = self._modelo.predict(X, verbose=verbose)
        self.ultima = np.asarray(p, dtype=np.float64)[0].copy()
        return p


def crear_traductor(ruta_modelo: Path, extractor, umbral: float | None, procedencia: dict | None) -> LSCTranslator:
    """LSCTranslator con los parámetros por defecto de realtime.py (ventana 2,5 s, mínimo 15, máximo 50 % sin
    manos, aviso de bordes, conversor «inventario»), sin JSONL, sin voz y con el extractor reproducido."""
    with contextlib.redirect_stdout(io.StringIO()):
        tr = LSCTranslator(model_path=str(ruta_modelo), tts=None, tts_mode="none", umbral=umbral,
                           ventana_frames=0, ventana_seg=VENTANA_SEG, min_frames=MIN_FRAMES,
                           max_sin_manos=MAX_SIN_MANOS, modo_oracion=False, registro=False, extractor=extractor,
                           grammar_mapper=trmod.construir_conversor(CONVERSOR), calentar=True,
                           tts_precalentar=False, procedencia=procedencia, bordes="aviso")
    assert tr.mano_ausente == "imputacion" and tr.representacion == "manos" and tr.usa_velocidades
    assert not tr.arma_rostro, "la representación «manos» con «imputacion» no usa el rostro"
    tr.modelo = ModeloConRegistro(tr.modelo)
    return tr


def correr_ventana(tr: LSCTranslator, extractor: ExtractorPregrabado, extracciones: list[Extraccion],
                   n: int, fps: float = FPS_CORPUS, tope_seg: float | None = None) -> dict:
    """Una ventana de n fotogramas por la cadena del prototipo: abrir_ventana → procesar_fotograma × n →
    cierre por fotogramas → registro del ensayo (mismo diccionario que el JSONL) + probabilidades.

    La marca de llegada c_i de cada fotograma es la de la cámara a la tasa del video (t0 + i/fps), no el reloj
    del bucle (que aquí reproduce puntos en microsegundos): así el aviso de bordes, que mide velocidades con
    c_i, ve el tiempo real de la seña. El tope de tiempo de la ventana es el del prototipo (2 × 2,5 s) salvo
    que se dé tope_seg (la ventana «clip» de clips largos no debe cerrarse por tiempo)."""
    extractor.cargar(extracciones[:n])
    tr.modelo.ultima = None
    tr.ventana_frames = n
    tope_original = tr.ventana_max_seg
    if tope_seg is not None:
        tr.ventana_max_seg = max(tope_original, tope_seg)
    try:
        t0 = tr.ahora()
        if tr.abrir_ventana(t0) is None:
            raise RuntimeError("había una ventana abierta")
    finally:
        tr.ventana_max_seg = tope_original
    registro = None
    for i in range(n):
        res = tr.procesar_fotograma(None, c_i=t0 + int(round(i * 1e9 / fps)))
        if res.registro is not None:
            registro = res.registro
            break
    if registro is None:
        raise RuntimeError("la ventana no se cerró")
    if registro["cierre"] != "frames" or registro["N"] != n:
        raise RuntimeError(f"cierre inesperado {registro['cierre']} con N={registro['N']} de {n}")
    return {"registro": registro, "prob": tr.modelo.ultima}


# ---------------------------------------------------------------------------
# Extracción desde el video (en memoria)
# ---------------------------------------------------------------------------

class RelojDeVideo:
    """Reloj inyectado en LandmarkExtractor: devuelve el instante del fotograma actual en el video (ns)."""

    def __init__(self):
        self.ns = 0

    def __call__(self) -> int:
        return self.ns


def extraer_video(ruta: Path, mp_paths: dict, tam: tuple[int, int] | None, n_min: int) -> dict:
    """Decodifica el video, reduce cada fotograma a `tam` (None = nativo), BGR→RGB y LandmarkExtractor.extract,
    todo en memoria. Si el clip tiene menos de n_min fotogramas, sigue con el último fotograma (relleno).
    Devuelve los puntos por fotograma (225), las banderas y los tiempos; nunca la imagen."""
    import cv2
    reloj = RelojDeVideo()
    ext = trmod.LandmarkExtractor(mp_paths, reloj=reloj)       # t_inicio = 0
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise RuntimeError(f"no se pudo abrir {ruta.name}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or FPS_CORPUS)
    ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kps, pose, manos, t_lectura, t_extr = [], [], [], [], []
    ultimo = None
    i = 0
    try:
        while True:
            a = time.perf_counter_ns()
            ok, frame = cap.read()
            if not ok:
                break
            if tam is not None:
                frame = cv2.resize(frame, tam, interpolation=cv2.INTER_AREA)
            c = time.perf_counter_ns()                    # c_i: fotograma listo, como tras cap.read() en vivo
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            reloj.ns = int(round(i * 1000.0 / fps)) * 1_000_000
            e = ext.extract(rgb)
            fin = time.perf_counter_ns()                  # e_i
            kps.append(e.keypoints.copy()); pose.append(bool(e.pose_detectada)); manos.append(int(e.n_manos))
            t_lectura.append((c - a) / 1e6); t_extr.append((fin - c) / 1e6)
            ultimo = rgb
            del frame
            i += 1
        T = i
        while ultimo is not None and i < n_min:            # relleno: la persona sigue quieta ante la cámara
            reloj.ns = int(round(i * 1000.0 / fps)) * 1_000_000
            e = ext.extract(ultimo)
            kps.append(e.keypoints.copy()); pose.append(bool(e.pose_detectada)); manos.append(int(e.n_manos))
            i += 1
    finally:
        cap.release()
        ext.close()
    del ultimo
    return {"T": T, "fps": fps, "resolucion_video": [ancho, alto],
            "keypoints": np.asarray(kps, dtype=np.float32).reshape(-1, 225),
            "pose": np.asarray(pose, dtype=bool), "manos": np.asarray(manos, dtype=np.int8),
            "t_lectura_ms": np.asarray(t_lectura), "t_extraccion_ms": np.asarray(t_extr)}


def extracciones_de(d: dict) -> list[Extraccion]:
    return [Extraccion(keypoints=d["keypoints"][j], pose_detectada=bool(d["pose"][j]), n_manos=int(d["manos"][j]),
                       rostro=None, holistic_result=None) for j in range(len(d["keypoints"]))]


def extraer_con_cache(stem: str, mp_paths: dict, tam, etiqueta: str, cache: Path | None) -> tuple[dict, bool]:
    if cache is not None:
        ruta = cache / etiqueta / f"{stem}.npz"
        if ruta.exists():
            z = np.load(ruta)
            d = {k: z[k] for k in z.files}
            d["T"] = int(d["T"]); d["fps"] = float(d["fps"]); d["resolucion_video"] = d["resolucion_video"].tolist()
            return d, True
    d = extraer_video(VIDEOS / f"{stem}.avi", mp_paths, tam, N_VENTANA)
    if cache is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ruta, **{k: np.asarray(v) for k, v in d.items()})   # solo puntos y tiempos
    return d, False


def relleno_almacenado(puntos: np.ndarray, n: int) -> np.ndarray:
    """Primeros n fotogramas de los puntos almacenados, repitiendo el último si faltan (simular_ventana_25)."""
    idx = np.clip(np.arange(n), 0, len(puntos) - 1)
    return puntos[idx]


# ---------------------------------------------------------------------------
# Comparación de puntos (extracción en vivo frente a la almacenada)
# ---------------------------------------------------------------------------

def comparar_puntos(nuevos: np.ndarray, almacenados: np.ndarray) -> dict:
    """Sobre los fotogramas comunes: concordancia de detección por mano y diferencia media |Δx|+|Δy| (en
    coordenadas de imagen normalizadas 0-1) de las muñecas cuando ambas extracciones detectaron la mano."""
    T = min(len(nuevos), len(almacenados))
    a, b = nuevos[:T], almacenados[:T]
    out = {"T_comun": int(T)}
    for nombre, sl in (("izq", IDX_MANO_IZQ), ("der", IDX_MANO_DER), ("pose", IDX_POSE)):
        da = np.any(a[:, sl] != 0, axis=1)
        db = np.any(b[:, sl] != 0, axis=1)
        ambos = da & db
        out[f"det_{nombre}_nuevo"] = int(da.sum())
        out[f"det_{nombre}_almacenado"] = int(db.sum())
        out[f"det_{nombre}_discrepa"] = int((da != db).sum())
        if ambos.any():
            x0 = sl.start
            dif = np.abs(a[ambos, x0] - b[ambos, x0]) + np.abs(a[ambos, x0 + 1] - b[ambos, x0 + 1])
            out[f"dif_{nombre}_media"] = float(dif.mean())
        else:
            out[f"dif_{nombre}_media"] = None
    return out


# ---------------------------------------------------------------------------
# Filas y métricas
# ---------------------------------------------------------------------------

def fila(stem: str, modelo: str, fuente: str, ventana: str, res: dict, indice_voz) -> dict:
    reg, prob = res["registro"], res["prob"]
    verdad = int(stem[:4])
    top3 = [t["id"] for t in (reg["top3"] or [])]
    enunciado = reg.get("enunciado")
    bordes = reg.get("bordes") or {}
    return {
        "video": stem, "senante": stem.split("_")[1], "verdad": verdad, "modelo": modelo, "fuente": fuente,
        "ventana": ventana, "N": reg["N"], "estado": reg["estado"], "pred": reg["id_clase"],
        "confianza": reg["confianza"], "top3": top3, "enunciado": enunciado,
        "voz_pregrabada": (indice_voz.buscar(enunciado) is not None) if enunciado else None,
        "conversion_respaldo": (reg.get("conversion") or {}).get("respaldo"),
        "aviso_inicio": bordes.get("aviso_inicio"), "aviso_fin": bordes.get("aviso_fin"),
        "proporcion_sin_manos": reg.get("proporcion_sin_manos"),
        "prob": prob,
    }


def metricas(filas: list[dict]) -> dict | None:
    """top-1 / top-3 (inferida y correcta; una ventana rechazada por manos o incompleta cuenta como fallo),
    rechazo con el umbral (por motivo), exactitud entre aceptadas y cobertura de la voz pregrabada."""
    n = len(filas)
    if n == 0:
        return None
    est = Counter(f["estado"] for f in filas)
    top1 = sum(1 for f in filas if f["pred"] is not None and f["pred"] == f["verdad"]) / n
    top3 = sum(1 for f in filas if f["verdad"] in f["top3"]) / n
    acept = [f for f in filas if f["estado"] == "aceptada"]
    voz = [f["voz_pregrabada"] for f in acept]
    return {"n": n, "top1": top1, "top3": top3,
            "rechazo": 1 - len(acept) / n,
            "rechazo_confianza": est["rechazada_confianza"] / n,
            "rechazo_manos": est["rechazada_manos"] / n,
            "rechazo_incompleta": est["rechazada_incompleta"] / n,
            "aceptadas": len(acept),
            "exactitud_aceptadas": (sum(1 for f in acept if f["pred"] == f["verdad"]) / len(acept)) if acept else None,
            "voz_encontrada": (sum(1 for v in voz if v) / len(voz)) if any(v is not None for v in voz) else None,
            "voz_no_encontrada": sum(1 for v in voz if v is False),
            "aviso_bordes": (sum(1 for f in filas if f["aviso_inicio"] or f["aviso_fin"]) / n
                             if any(f["aviso_inicio"] is not None for f in filas) else None)}


def metricas_referencia_loso(ref: dict, nombres_por_pliegue: dict) -> dict:
    """Las mismas métricas desde probabilidades.npz (clip completo, cadena de entrenamiento), restringidas a
    los videos procesados. Sin rechazo por manos: la referencia siempre infiere."""
    out = {}
    for (s, k), probs in ref.items():
        if not nombres_por_pliegue.get(k):
            continue
        filas = []
        for n in nombres_por_pliegue.get(k, []):
            p = probs[n]
            o = np.argsort(p)[::-1]
            conf = float(p[o[0]])
            filas.append({"verdad": int(n[:4]), "pred": int(o[0]), "top3": [int(x) for x in o[:3]],
                          "estado": "aceptada" if conf >= UMBRAL else "rechazada_confianza",
                          "voz_pregrabada": None, "aviso_inicio": None, "aviso_fin": None})
        out[(s, k)] = metricas(filas)
    return out


def agregar_loso(por_corrida: dict) -> dict:
    """por_corrida[(s, k)] = métricas → por pliegue (media de semillas), por semilla (media de pliegues, sin
    ponderar, como docs/SIMULACION_VENTANA_25.md) y global (media de corridas)."""
    claves = ("top1", "top3", "rechazo", "rechazo_confianza", "rechazo_manos", "rechazo_incompleta",
              "exactitud_aceptadas", "voz_encontrada", "aviso_bordes")
    pliegues = sorted({k for _, k in por_corrida})
    semillas = sorted({s for s, _ in por_corrida})

    def media(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    out = {"por_pliegue": {}, "por_semilla": {}, "global": {}}
    for k in pliegues:
        out["por_pliegue"][f"{k}"] = {c: _r(media([por_corrida[(s, k)].get(c) for s in semillas if (s, k) in por_corrida
                                                    and por_corrida[(s, k)]])) for c in claves}
        out["por_pliegue"][f"{k}"]["n_videos"] = por_corrida[(semillas[0], k)]["n"] if por_corrida.get((semillas[0], k)) else 0
    for s in semillas:
        out["por_semilla"][f"{s}"] = {c: _r(media([por_corrida[(s, k)].get(c) for k in pliegues
                                                   if por_corrida.get((s, k))])) for c in claves}
    out["global"] = {c: _r(media([out["por_pliegue"][f"{k}"][c] for k in pliegues])) for c in claves}
    out["global"]["de_entre_semillas_top1"] = _r(np.std([out["por_semilla"][f"{s}"]["top1"] for s in semillas], ddof=1)) \
        if len(semillas) > 1 else None
    return out


def concordancia(filas_a: list[dict], ref_pred: dict, ref_prob: dict | None = None) -> dict:
    """Concordancia de la fila (a) con la referencia por (video, modelo): misma top-1; mismo acierto; misma
    decisión de aceptar; |Δ confianza| de la clase de la referencia; diferencia máxima de probabilidades."""
    n = iguales = mismo_acierto = misma_decision = 0
    solo_a = solo_ref = sin_inferencia = 0
    d_conf, d_max = [], []
    for f in filas_a:
        clave = (f["video"], f["modelo"])
        if clave not in ref_pred:
            continue
        r = ref_pred[clave]
        n += 1
        if f["pred"] is None:
            sin_inferencia += 1
        iguales += int(f["pred"] == r["pred"])
        a_ok, r_ok = f["pred"] == f["verdad"], r["pred"] == f["verdad"]
        mismo_acierto += int(a_ok == r_ok)
        solo_a += int(a_ok and not r_ok)
        solo_ref += int(r_ok and not a_ok)
        misma_decision += int((f["estado"] == "aceptada") == (r["confianza"] >= UMBRAL))
        if f["prob"] is not None and ref_prob is not None and clave in ref_prob:
            d_conf.append(abs(f["prob"][r["pred"]] - ref_prob[clave][r["pred"]]))
            d_max.append(float(np.max(np.abs(f["prob"] - ref_prob[clave]))))
        elif f["prob"] is not None and r.get("confianza") is not None:
            d_conf.append(abs(f["prob"][r["pred"]] - r["confianza"]))
    if n == 0:
        return {"n": 0}
    return {"n": n, "misma_top1": _r(iguales / n), "mismo_acierto": _r(mismo_acierto / n),
            "misma_decision_umbral": _r(misma_decision / n), "acierta_solo_punta_a_punta": solo_a,
            "acierta_solo_referencia": solo_ref, "sin_inferencia": sin_inferencia,
            "dif_conf_clase_ref_mediana": _r(np.median(d_conf)) if d_conf else None,
            "dif_conf_clase_ref_media": _r(np.mean(d_conf)) if d_conf else None,
            "dif_prob_max_mediana": _r(np.median(d_max)) if d_max else None,
            "dif_prob_max_max": _r(np.max(d_max)) if d_max else None}


def resumen_ms(a) -> dict | None:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return None
    return {"n": int(a.size), "media": _r(a.mean(), 3), "mediana": _r(np.median(a), 3),
            "p90": _r(np.percentile(a, 90), 3), "p99": _r(np.percentile(a, 99), 3), "max": _r(a.max(), 3)}


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

UMBRAL = float(json.loads(UMBRAL_JSON.read_text(encoding="utf-8"))["umbral"])


def ordenar_videos(nombres: list[str], senantes: list[str]) -> list[str]:
    orden = {s: i for i, s in enumerate(ORDEN_SENANTES)}
    elegidos = [n for n in nombres if n.split("_")[1] in senantes]
    return sorted(elegidos, key=lambda n: (orden[n.split("_")[1]], n))


def correr(args) -> dict:
    t_inicio = time.time()
    part = cargar_particiones()
    todos = sorted(p.stem for p in VIDEOS.glob("*.avi"))
    validos = [n for n in todos if n not in part["excluidas"]]
    assert len(todos) == 1000 and len(validos) == 997, (len(todos), len(validos))
    assert set(validos) == set(part["pliegue_de"]), "los 997 deben ser la prueba de exactamente un pliegue"
    videos = ordenar_videos(validos, args.senantes)
    if args.videos:
        videos = [v for v in videos if v in set(args.videos)]
    if args.limite:
        videos = videos[:args.limite]
    test_ale = set(part["test_aleatoria"])
    _imprimir(f"{len(videos)} videos ({', '.join(args.senantes)}); {sum(1 for v in videos if v in test_ale)} "
              f"de la prueba aleatoria")

    ref_loso = cargar_referencias_loso()
    ref_final = cargar_referencia_final()
    from src.voz_pregrabada import IndiceVozPregrabada
    indice_voz = IndiceVozPregrabada.cargar()

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")        # CPU únicamente, como el prototipo (§1.4)

    extractor = ExtractorPregrabado()
    traductores = {}
    _imprimir("cargando 15 modelos LOSO y el modelo desplegado en LSCTranslator")
    for s in SEMILLAS:
        for k in PLIEGUES:
            ruta = CONFIRMACION / f"semilla_{s}" / f"pliegue_{k}" / "modelo.keras"
            traductores[(s, k)] = crear_traductor(ruta, extractor, UMBRAL, procedencia_loso(ruta))
    tr_final = crear_traductor(MODELO_FINAL, extractor, None, None)   # umbral y procedencia leídos solos
    assert abs(tr_final.umbral - UMBRAL) < 1e-12, tr_final.umbral
    info_modelos = {
        "loso": {f"{s}_{k}": {"ruta": _rel(CONFIRMACION / f"semilla_{s}" / f"pliegue_{k}" / "modelo.keras"),
                              "senante_excluido": f"{k:04d}", "umbral": traductores[(s, k)].umbral,
                              "umbral_origen": "parametro (mismo valor que modelo.umbral.json del desplegado)"}
                 for s in SEMILLAS for k in PLIEGUES},
        "desplegado": {"ruta": _rel(MODELO_FINAL), "sha256": _sha256(MODELO_FINAL), "umbral": tr_final.umbral,
                       "umbral_origen": _rel(tr_final.umbral_origen.split(" ")[0])},
    }
    mp_paths = trmod._ensure_models_downloaded()
    sha_task = _sha256(Path(mp_paths[trmod.HOLISTIC_TASK]))

    fuentes = [FUENTE_E2E] + ([] if args.sin_nativa else [FUENTE_NATIVA])
    tamanos = {FUENTE_E2E: RESOLUCION_EN_VIVO, FUENTE_NATIVA: None}
    filas: list[dict] = []
    comparacion_puntos = {f: [] for f in fuentes}
    tiempos = {f: {"lectura": [], "extraccion": []} for f in fuentes}
    por_video = []
    t_extraccion_total = {f: 0.0 for f in fuentes}
    t_inferencia_total = 0.0
    cargas = [leer_carga()]

    for iv, stem in enumerate(videos):
        k = part["pliegue_de"][stem]
        modelos = [(f"loso_{s}_{k}", traductores[(s, k)]) for s in SEMILLAS]
        if stem in test_ale:
            modelos.append(("desplegado", tr_final))
        info_v = {"video": stem, "pliegue": k, "en_prueba_aleatoria": stem in test_ale}

        # Referencia: puntos almacenados por la misma cadena del prototipo.
        almac = cargar_puntos(stem)
        T_alm = len(almac)
        info_v["T_almacenado"] = T_alm
        ext_alm = {"clip": extracciones_desde_puntos(almac),
                   "v2_5s": extracciones_desde_puntos(relleno_almacenado(almac, N_VENTANA)),
                   "v2_5s_sin_relleno": extracciones_desde_puntos(almac[:N_VENTANA])}
        t_a = time.perf_counter()
        for nombre, tr in modelos:
            for ventana, exts in ext_alm.items():
                filas.append(fila(stem, nombre, FUENTE_ALMACENADA, ventana,
                                  correr_ventana(tr, extractor, exts, len(exts), FPS_CORPUS,
                                                 len(exts) / FPS_CORPUS + 1.0), indice_voz))
        t_inferencia_total += time.perf_counter() - t_a

        for fuente in fuentes:
            t_e = time.perf_counter()
            d, de_cache = extraer_con_cache(stem, mp_paths, tamanos[fuente], fuente, args.cache)
            t_extraccion_total[fuente] += time.perf_counter() - t_e
            tiempos[fuente]["lectura"].append(d["t_lectura_ms"])
            tiempos[fuente]["extraccion"].append(d["t_extraccion_ms"])
            T = d["T"]
            info_v[f"T_{fuente}"] = T
            info_v[f"resolucion_video"] = d["resolucion_video"]
            info_v[f"desde_cache_{fuente}"] = de_cache
            cp = comparar_puntos(d["keypoints"][:T], almac)
            cp["video"] = stem
            comparacion_puntos[fuente].append(cp)
            exts = extracciones_de(d)
            ventanas = {"clip": T, "v2_5s": N_VENTANA, "v2_5s_sin_relleno": min(T, N_VENTANA)}
            t_a = time.perf_counter()
            for nombre, tr in modelos:
                for ventana, n in ventanas.items():
                    filas.append(fila(stem, nombre, fuente, ventana,
                                      correr_ventana(tr, extractor, exts, n, d["fps"], n / d["fps"] + 1.0),
                                      indice_voz))
            t_inferencia_total += time.perf_counter() - t_a
        por_video.append(info_v)
        if (iv + 1) % 25 == 0 or iv + 1 == len(videos):
            transcurrido = time.time() - t_inicio
            _imprimir(f"{iv + 1}/{len(videos)} videos; {transcurrido / 60:.1f} min; carga {leer_carga()['1min']}")
            cargas.append(leer_carga())

    for tr in list(traductores.values()) + [tr_final]:
        tr.close()

    datos = {"videos": videos, "filas": filas, "por_video": por_video, "comparacion_puntos": comparacion_puntos,
             "tiempos": tiempos, "ref_loso": ref_loso, "ref_final": ref_final, "test_aleatoria": test_ale,
             "info_modelos": info_modelos, "sha_task": sha_task, "fuentes": fuentes,
             "t_total_s": time.time() - t_inicio, "t_extraccion_total_s": t_extraccion_total,
             "t_inferencia_total_s": t_inferencia_total, "cargas": cargas, "pliegue_de": part["pliegue_de"],
             "excluidas": part["excluidas"]}
    return datos


def resumir(datos: dict) -> dict:
    filas = datos["filas"]
    ref_loso = datos["ref_loso"]
    videos = datos["videos"]
    pliegue_de = datos["pliegue_de"]
    nombres_por_pliegue: dict[int, list[str]] = {}
    for v in videos:
        nombres_por_pliegue.setdefault(pliegue_de[v], []).append(v)

    # Referencias por (video, modelo)
    ref_npz_pred, ref_npz_prob = {}, {}
    for (s, k), probs in ref_loso.items():
        for v in nombres_por_pliegue.get(k, []):
            p = probs[v]
            ref_npz_pred[(v, f"loso_{s}_{k}")] = {"pred": int(np.argmax(p)), "confianza": float(np.max(p))}
            ref_npz_prob[(v, f"loso_{s}_{k}")] = p
    for v, r in datos["ref_final"].items():
        ref_npz_pred[(v, "desplegado")] = {"pred": int(r["predicha"]), "confianza": float(r["prob_predicha"])}

    def sel(fuente, ventana, loso=True):
        return [f for f in filas if f["fuente"] == fuente and f["ventana"] == ventana
                and (f["modelo"].startswith("loso_") if loso else f["modelo"] == "desplegado")]

    def por_corrida(fs):
        grupos: dict = {}
        for f in fs:
            _, s, k = f["modelo"].split("_")
            grupos.setdefault((int(s), int(k)), []).append(f)
        return {c: metricas(g) for c, g in grupos.items()}

    def ref_de(fs):
        return {(f["video"], f["modelo"]): {"pred": f["pred"], "confianza": f["confianza"] or 0.0} for f in fs}

    def prob_de(fs):
        return {(f["video"], f["modelo"]): f["prob"] for f in fs if f["prob"] is not None}

    out = {"loso": {}, "desplegado": {}, "concordancia": {}}
    # LOSO fuera de línea (probabilidades.npz) sobre los mismos videos
    out["loso"]["referencia_npz_clip"] = agregar_loso(metricas_referencia_loso(ref_loso, nombres_por_pliegue))
    for fuente in [FUENTE_ALMACENADA] + datos["fuentes"]:
        for ventana in VENTANAS:
            out["loso"][f"{fuente}|{ventana}"] = agregar_loso(por_corrida(sel(fuente, ventana)))
            fs = sel(fuente, ventana, loso=False)
            out["desplegado"][f"{fuente}|{ventana}"] = metricas(fs)
            if fs:
                out["desplegado"][f"{fuente}|{ventana}"]["exactitud_aceptadas"] = _r(
                    out["desplegado"][f"{fuente}|{ventana}"]["exactitud_aceptadas"])
    # Desplegado fuera de línea (predicciones.json) sobre los videos procesados
    filas_ref_final = []
    for v, r in datos["ref_final"].items():
        if v in set(videos):
            filas_ref_final.append({"verdad": int(v[:4]), "pred": int(r["predicha"]), "top3": list(r["top3"]),
                                    "estado": "aceptada" if r["prob_predicha"] >= UMBRAL else "rechazada_confianza",
                                    "voz_pregrabada": None, "aviso_inicio": None, "aviso_fin": None})
    out["desplegado"]["referencia_predicciones_json"] = metricas(filas_ref_final)

    # Concordancias
    conc = out["concordancia"]
    for grupo, es_loso in (("loso", True), ("desplegado", False)):
        for ventana in VENTANAS:
            alm = sel(FUENTE_ALMACENADA, ventana, es_loso)
            if ventana == "clip":
                conc[f"{grupo}: almacenados_vs_entrenamiento|clip"] = concordancia(alm, ref_npz_pred, ref_npz_prob)
            for fuente in datos["fuentes"]:
                fs = sel(fuente, ventana, es_loso)
                if ventana == "clip":
                    conc[f"{grupo}: {fuente}_vs_entrenamiento|clip"] = concordancia(fs, ref_npz_pred, ref_npz_prob)
                conc[f"{grupo}: {fuente}_vs_almacenados|{ventana}"] = concordancia(fs, ref_de(alm), prob_de(alm))
            if FUENTE_NATIVA in datos["fuentes"]:
                nat = sel(FUENTE_NATIVA, ventana, es_loso)
                fs = sel(FUENTE_E2E, ventana, es_loso)
                conc[f"{grupo}: {FUENTE_E2E}_vs_{FUENTE_NATIVA}|{ventana}"] = concordancia(fs, ref_de(nat), prob_de(nat))
    # Concordancia por pliegue (clip, punta a punta vs entrenamiento)
    conc["por_pliegue_e2e_vs_entrenamiento_clip"] = {
        str(k): concordancia([f for f in sel(FUENTE_E2E, "clip") if pliegue_de[f["video"]] == k],
                             ref_npz_pred, ref_npz_prob) for k in sorted(nombres_por_pliegue)}

    # Puntos: detección y diferencias
    pts = {}
    for fuente, lista in datos["comparacion_puntos"].items():
        tot = lambda c: int(sum(x[c] for x in lista))  # noqa: E731
        T_tot = tot("T_comun")
        pts[fuente] = {
            "videos": len(lista), "fotogramas_comunes": T_tot,
            "videos_con_T_distinto": sum(1 for pv in datos["por_video"] if pv.get(f"T_{fuente}") != pv["T_almacenado"]),
            **{f"det_{m}_{q}": tot(f"det_{m}_{q}") for m in ("izq", "der", "pose")
               for q in ("nuevo", "almacenado", "discrepa")},
            **{f"frac_det_{m}_discrepa": _r(tot(f"det_{m}_discrepa") / T_tot) if T_tot else None
               for m in ("izq", "der", "pose")},
            **{f"dif_{m}_media_mediana_videos": _r(np.median([x[f"dif_{m}_media"] for x in lista
                                                              if x[f"dif_{m}_media"] is not None]), 5)
               for m in ("izq", "der", "pose")},
        }
    out["puntos"] = pts

    # Tiempos
    tiempos = {}
    for fuente, t in datos["tiempos"].items():
        lect = np.concatenate(t["lectura"]) if t["lectura"] else np.array([])
        extr = np.concatenate(t["extraccion"]) if t["extraccion"] else np.array([])
        tiempos[fuente] = {"lectura_y_reduccion_ms": resumen_ms(lect),
                           "extraccion_ms_bgr2rgb_mas_extract": resumen_ms(extr),
                           "tiempo_total_extraccion_s": _r(datos["t_extraccion_total_s"][fuente], 1)}
    tiempos["inferencia_y_cadena_total_s"] = _r(datos["t_inferencia_total_s"], 1)
    tiempos["total_s"] = _r(datos["t_total_s"], 1)
    out["tiempos"] = tiempos

    # Resumen de voz y conversor (punta a punta, clip y 2,5 s)
    voz = {}
    for ventana in ("clip", "v2_5s"):
        acept = [f for f in filas if f["fuente"] == FUENTE_E2E and f["ventana"] == ventana and f["estado"] == "aceptada"]
        voz[ventana] = {"aceptadas": len(acept), "enunciado_presente": sum(1 for f in acept if f["enunciado"]),
                        "voz_encontrada": sum(1 for f in acept if f["voz_pregrabada"]),
                        "respaldo_conversor": sum(1 for f in acept if f["conversion_respaldo"]),
                        "enunciados_distintos": len({f["enunciado"] for f in acept})}
    out["voz"] = voz
    return out


def filas_compactas(filas: list[dict]) -> list[dict]:
    return [{k: (v if k != "prob" else None) for k, v in f.items() if k not in ("prob",)} for f in filas]


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _p(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f} %".replace(".", ",")


def _n(x, d=3):
    return "—" if x is None else f"{x:.{d}f}".replace(".", ",")


def escribir_md(inf: dict) -> str:
    r = inf["resumen"]
    L = r["loso"]
    fuentes = inf["parametros"]["fuentes"]
    lin = []
    lin.append("# Evaluación punta a punta de la cadena integrada sobre los videos del corpus\n")
    lin.append(f"Fecha: {inf['fin'][:10]}. Script: `src/evaluacion_punta_a_punta.py` (ver su docstring). "
               f"Videos procesados: {len(inf['videos'])} de 997 (señantes {', '.join(inf['parametros']['senantes'])}). "
               "Sin personas en la sesión: la entrada son los videos COLOR_BODY del corpus.\n")
    lin.append("La cadena es la del prototipo (`src/translator.py`, sin cambios): `LandmarkExtractor` sobre cada "
               "fotograma reducido en memoria a 1280×720, ventana de `LSCTranslator`, `_preprocesar` (normalización, "
               "imputación, 30 pasos, velocidades), una llamada a `predict`, umbral 0,68, conversor «inventario» y "
               "búsqueda del enunciado en el índice de voz pregrabada (sin reproducir). Ningún fotograma se escribe "
               "a disco.\n")
    lin.append("## Modelos LOSO (cada video lo clasifica solo un modelo que no vio a su señante)\n")
    lin.append("Media de 3 semillas por pliegue; global = media de los 5 pliegues (como el 0,677 de la tesis).\n")
    lin.append("| fuente | ventana | top-1 | top-3 | rechazo (0,68) | de él: manos | exactitud aceptadas | voz encontrada | aviso de bordes |")
    lin.append("|---|---|---|---|---|---|---|---|---|")
    filas_md = [("referencia_npz_clip", "entrenamiento (probabilidades.npz)", "clip")]
    for f in [FUENTE_ALMACENADA] + fuentes:
        for v in VENTANAS:
            filas_md.append((f"{f}|{v}", f, v))
    for clave, fuente, ventana in filas_md:
        g = L[clave]["global"]
        lin.append(f"| {fuente} | {ventana} | {_n(g['top1'])} | {_n(g['top3'])} | {_p(g['rechazo'])} | "
                   f"{_p(g['rechazo_manos'])} | {_n(g['exactitud_aceptadas'])} | {_p(g['voz_encontrada'])} | "
                   f"{_p(g['aviso_bordes'])} |")
    lin.append("\nTop-1 por pliegue (media de semillas):\n")
    pl = sorted(L["referencia_npz_clip"]["por_pliegue"])
    lin.append("| configuración | " + " | ".join(f"pliegue {k} (señante 000{k})" for k in pl) + " |")
    lin.append("|---|" + "---|" * len(pl))
    for clave, fuente, ventana in filas_md:
        lin.append(f"| {fuente} · {ventana} | " + " | ".join(_n(L[clave]["por_pliegue"][k]["top1"]) for k in pl) + " |")
    lin.append("\nRechazo con 0,68 y exactitud entre aceptadas por pliegue, punta a punta (clip · 2,5 s):\n")
    lin.append("| pliegue | rechazo clip | exactitud aceptadas clip | rechazo 2,5 s | exactitud aceptadas 2,5 s | top-3 clip | top-3 2,5 s |")
    lin.append("|---|---|---|---|---|---|---|")
    for k in pl:
        a, b = L[f"{FUENTE_E2E}|clip"]["por_pliegue"][k], L[f"{FUENTE_E2E}|v2_5s"]["por_pliegue"][k]
        lin.append(f"| {k} | {_p(a['rechazo'])} | {_n(a['exactitud_aceptadas'])} | {_p(b['rechazo'])} | "
                   f"{_n(b['exactitud_aceptadas'])} | {_n(a['top3'])} | {_n(b['top3'])} |")
    lin.append("\nPor semilla (top-1, media de pliegues): " + "; ".join(
        f"{c}: " + " / ".join(_n(L[c]["por_semilla"][s]["top1"]) for s in sorted(L[c]["por_semilla"]))
        for c in ("referencia_npz_clip", f"{FUENTE_E2E}|clip", f"{FUENTE_E2E}|v2_5s")) + ".\n")

    lin.append("## Concordancia con la medida fuera de línea (mismo video, mismo modelo)\n")
    lin.append("| comparación | n | misma top-1 | mismo acierto | misma decisión con 0,68 | acierta solo punta a punta | acierta solo la referencia | |Δp| clase ref. (mediana) | máx |Δp| (mediana) |")
    lin.append("|---|---|---|---|---|---|---|---|---|")
    for c, v in r["concordancia"].items():
        if c.startswith("por_pliegue") or not isinstance(v, dict) or "n" not in v or v["n"] == 0:
            continue
        lin.append(f"| {c} | {v['n']} | {_p(v['misma_top1'])} | {_p(v['mismo_acierto'])} | "
                   f"{_p(v['misma_decision_umbral'])} | {v['acierta_solo_punta_a_punta']} | "
                   f"{v['acierta_solo_referencia']} | {_n(v.get('dif_conf_clase_ref_mediana'))} | "
                   f"{_n(v.get('dif_prob_max_mediana'))} |")
    lin.append("\nPunta a punta frente a entrenamiento, clip completo, por pliegue: " + "; ".join(
        f"pliegue {k}: {_p(v['misma_top1'])}" for k, v in r["concordancia"]["por_pliegue_e2e_vs_entrenamiento_clip"].items()) + ".\n")

    lin.append("## Puntos característicos: extracción del video frente a la almacenada\n")
    lin.append("| fuente | fotogramas | videos con T distinto | discrepa detección mano izq. | mano der. | postura | |Δx|+|Δy| muñeca izq. (mediana de videos) | der. |")
    lin.append("|---|---|---|---|---|---|---|---|")
    for f, p in r["puntos"].items():
        lin.append(f"| {f} | {p['fotogramas_comunes']} | {p['videos_con_T_distinto']} | {_p(p['frac_det_izq_discrepa'], 2)} | "
                   f"{_p(p['frac_det_der_discrepa'], 2)} | {_p(p['frac_det_pose_discrepa'], 2)} | "
                   f"{_n(p['dif_izq_media_mediana_videos'], 5)} | {_n(p['dif_der_media_mediana_videos'], 5)} |")

    lin.append("\n## Modelo desplegado (solo los videos de la prueba aleatoria, que no vio)\n")
    D = r["desplegado"]
    lin.append("| fuente | ventana | n | top-1 | top-3 | rechazo (0,68) | exactitud aceptadas |")
    lin.append("|---|---|---|---|---|---|---|")
    m = D.get("referencia_predicciones_json")
    if m:
        lin.append(f"| entrenamiento (predicciones.json) | clip | {m['n']} | {_n(m['top1'])} | {_n(m['top3'])} | "
                   f"{_p(m['rechazo'])} | {_n(m['exactitud_aceptadas'])} |")
    for f in [FUENTE_ALMACENADA] + fuentes:
        for v in VENTANAS:
            m = D.get(f"{f}|{v}")
            if m:
                lin.append(f"| {f} | {v} | {m['n']} | {_n(m['top1'])} | {_n(m['top3'])} | {_p(m['rechazo'])} | "
                           f"{_n(m['exactitud_aceptadas'])} |")

    lin.append("\n## Voz pregrabada y conversor (punta a punta, modelos LOSO y desplegado)\n")
    for v, d in r["voz"].items():
        lin.append(f"- {v}: {d['aceptadas']} ventanas aceptadas; enunciado presente en {d['enunciado_presente']}; "
                   f"WAV pregrabado encontrado en {d['voz_encontrada']}; respaldo del conversor en "
                   f"{d['respaldo_conversor']}; {d['enunciados_distintos']} enunciados distintos.")
    t = r["tiempos"]
    lin.append("\n## Tiempos\n")
    for f in fuentes:
        e = t[f]["extraccion_ms_bgr2rgb_mas_extract"]
        l2 = t[f]["lectura_y_reduccion_ms"]
        lin.append(f"- {f}: extracción (BGR→RGB + extract) mediana {_n(e['mediana'], 1)} ms, media {_n(e['media'], 1)} ms, "
                   f"p90 {_n(e['p90'], 1)} ms, p99 {_n(e['p99'], 1)} ms sobre {e['n']} fotogramas; lectura del video y "
                   f"reducción mediana {_n(l2['mediana'], 1)} ms; total de la etapa {t[f]['tiempo_total_extraccion_s']} s.")
    lin.append(f"- Cadena tras la extracción (todas las ventanas, todos los modelos y fuentes): {t['inferencia_y_cadena_total_s']} s.")
    lin.append(f"- Tiempo total: {t['total_s']} s ({_n(t['total_s'] / 60, 1)} min). Carga del equipo (1 min) al "
               f"empezar {inf['cargas'][0]['1min']}, al terminar {inf['cargas'][-1]['1min']}.")
    lin.append("\n## Límites\n")
    lin.append("- Son las personas, la cámara, la iluminación y el fondo del corpus: no es una persona nueva ni una "
               "cámara distinta. No sustituye la sesión con personas (Bloque 1).")
    lin.append("- El fotograma de 1280×720 se obtiene reduciendo el video de 1920×1080; una cámara a 1280×720 tiene "
               "su propia óptica, compresión y ruido.")
    lin.append("- Las marcas de tiempo del modo VIDEO son las del video (24 fps), no el reloj real; un "
               "HolisticLandmarker nuevo por video. La ventana de 2,5 s abre en el primer fotograma del clip, en "
               "reposo, como pide el protocolo del prototipo; no se simula a la persona que tarda en empezar.")
    lin.append("- Con clips de menos de 60 fotogramas la ventana de 2,5 s se completa con el último fotograma "
               "(persona quieta).")
    return "\n".join(lin) + "\n"


def parsear_argumentos(argv=None):
    p = argparse.ArgumentParser(description="Evaluación punta a punta del prototipo sobre los videos del corpus")
    p.add_argument("--senantes", nargs="+", default=list(ORDEN_SENANTES), choices=list(ORDEN_SENANTES))
    p.add_argument("--limite", type=int, default=0, help="solo los primeros N videos (prueba de humo)")
    p.add_argument("--videos", nargs="+", default=None, help="solo estos videos (prueba de humo)")
    p.add_argument("--sin-nativa", action="store_true", help="omitir el diagnóstico a 1920×1080")
    p.add_argument("--cache", type=Path, default=None,
                   help="carpeta para guardar/reusar puntos y tiempos por video (nunca imágenes)")
    p.add_argument("--salida", type=Path, default=SALIDA)
    p.add_argument("--crudo", type=Path, default=CRUDO, help="JSON con las filas por video, modelo y ventana")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parsear_argumentos(argv)
    inicio = dt.datetime.now().isoformat(timespec="seconds")
    datos = correr(args)
    resumen = resumir(datos)
    sim = json.loads(SIMULACION_25.read_text(encoding="utf-8"))["simulacion"]["variantes"] if SIMULACION_25.exists() else {}
    informe = {
        "descripcion": "Evaluación punta a punta de la cadena del prototipo sobre los videos crudos del corpus "
                       "(aporte parcial a OE4); ver el docstring de src/evaluacion_punta_a_punta.py",
        "inicio": inicio, "fin": dt.datetime.now().isoformat(timespec="seconds"),
        "comando": "python " + " ".join([_rel(sys.argv[0])] + sys.argv[1:]),
        "equipo": contexto_equipo(), "procesos_al_terminar": procesos_top(),
        "parametros": {"senantes": args.senantes, "limite": args.limite, "fuentes": datos["fuentes"],
                       "resolucion_en_vivo": list(RESOLUCION_EN_VIVO), "ventana_seg": VENTANA_SEG,
                       "fps_corpus": FPS_CORPUS, "n_ventana": N_VENTANA, "min_frames": MIN_FRAMES,
                       "max_sin_manos": MAX_SIN_MANOS, "umbral": UMBRAL, "umbral_fuente": _rel(UMBRAL_JSON),
                       "conversor": CONVERSOR, "voz": "models/voz_pregrabada/indice.json (búsqueda, sin audio)",
                       "sha256_holistic_task": datos["sha_task"],
                       "sha256_holistic_task_fijado_D05": trmod.MODEL_SHA256[trmod.HOLISTIC_TASK],
                       "interpolacion_reduccion": "cv2.INTER_AREA",
                       "marcas_de_tiempo": "round(i*1000/fps) ms (reloj inyectado en LandmarkExtractor)",
                       "excluidas": datos["excluidas"]},
        "modelos": datos["info_modelos"],
        "referencias_citadas": {
            "loso_clip_tesis": "docs/EXPERIMENTO_CONFIRMACION_TCN.md, T1 (models/confirmacion_tcn_2026-10-03/T1_imputacion_252)",
            "simulacion_2_5s": {"fuente": "docs/SIMULACION_VENTANA_25.json", "primeros_2_5s": sim.get("primeros_2_5s"),
                                "completo": sim.get("completo")},
        },
        "cargas": datos["cargas"],
        "videos": datos["videos"],
        "resumen": resumen,
        "crudo": _rel(args.crudo),
    }
    crudo = {"inicio": inicio, "por_video": datos["por_video"], "comparacion_puntos": datos["comparacion_puntos"],
             "filas": filas_compactas(datos["filas"])}
    a_json = lambda o: o.tolist() if hasattr(o, "tolist") else str(o)  # noqa: E731
    informe = json.loads(json.dumps(informe, default=a_json))
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(informe, ensure_ascii=False, indent=1), encoding="utf-8")
    args.salida.with_suffix(".md").write_text(escribir_md(informe), encoding="utf-8")
    args.crudo.parent.mkdir(parents=True, exist_ok=True)
    args.crudo.write_text(json.dumps(crudo, ensure_ascii=False, default=a_json), encoding="utf-8")
    _imprimir(f"escrito {_rel(args.salida)}, {_rel(args.salida.with_suffix('.md'))} y {_rel(args.crudo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
