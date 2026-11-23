"""
Re-extracción de puntos característicos del corpus LSC50 con MediaPipe HolisticLandmarker.

Un solo extractor para el corpus y para el prototipo en vivo (decisión D-05 en
DECISIONES_TESIS.txt, con conformidad del director en N-19): mp.tasks.vision.HolisticLandmarker,
sucesor oficial de MediaPipe Holistic en la Tasks API (mediapipe 0.10.35). Entrega en una sola
llamada los 543 puntos de §1.4 y §2.2.3 del anteproyecto: postura (33) + rostro (468, sin iris)
+ mano izquierda (21) + mano derecha (21). D-05 corrige la premisa de D-01 (sí existe Holistic
en la API vigente) y deja SIN EFECTO para este extractor las decisiones D-03 (recorte manual del
rostro a partir de la postura) y D-04 (asignación manual de manos por muñeca): las dos cosas las
hace por dentro el grafo de Holistic.

Además cronometra el extractor (actividad 1.c: «tasa de procesamiento del extractor en el
hardware del propio proyecto», §2.2.3) y registra por fotograma qué componente detectó
(insumo de la actividad 1.b y de la hipótesis de N-02 sobre las filas que faltan en los
CSV publicados).

Salida (mismo árbol que producía este script con las tres tareas y misma convención que los CSV
publicados por LSC50: columna índice sin nombre + landmark_i_x/y/z, finales de línea CRLF):
    <salida>/BODY_LANDMARKS/<muestra>.csv          33 x 3 =   99 columnas de datos
    <salida>/LEFT_HAND_LANDMARKS/<muestra>.csv     21 x 3 =   63
    <salida>/RIGHT_HAND_LANDMARKS/<muestra>.csv    21 x 3 =   63
    <salida>/FACE_LANDMARKS/<muestra>.csv         468 x 3 = 1404   (salvo --sin-rostro)
    <salida>/METRICAS_FOTOGRAMA/<muestra>.csv      por fotograma: banderas de detección y ms por etapa
    <salida>/tasa_procesamiento.json               resumen global: hardware, modelo (SHA256) y opciones,
                                                   entrada, detección, ms por etapa (media/DE/mediana/p90)
                                                   y tasa efectiva en fotogramas/s
La salida por defecto es data/extracted/LSC50_holistic; la de marzo (data/extracted/LSC50,
Pose + Hand) no se toca.

Convenciones declaradas (también quedan escritas en tasa_procesamiento.json):
  · Una fila por fotograma decodificado; ningún fotograma se descarta. Si un componente
    (postura, mano izquierda, mano derecha, rostro) no se detecta en un fotograma, su fila se
    escribe con ceros y su bandera queda en 0 en METRICAS_FOTOGRAMA.
  · Un HolisticLandmarker NUEVO por video, en modo VIDEO (exige marcas de tiempo estrictamente
    crecientes): ts_ms = round(i * 1000 / fps_del_video). Delegado CPU.
  · Manos: LEFT_HAND_LANDMARKS = left_hand_landmarks del resultado = mano izquierda DE LA
    PERSONA; la asignación es interna del grafo de Holistic. Verificado el 14-sep-2026 (D-05)
    en 0000_0000_0000 contra el LEFT_HAND publicado por los autores: |dx|+|dy| medio de la
    muñeca 0,0013.
  · Rostro: lo detecta el grafo de Holistic sobre el cuadro completo (1920x1080), sin recorte
    manual. El resultado trae 478 puntos (468 de la malla + 10 de iris, índices 468-477); se
    escriben SOLO los 468 primeros para conservar los 543 puntos de §1.4 y la compatibilidad
    con los CSV publicados. Con --sin-rostro el grafo lo calcula igual, pero no se escribe
    FACE_LANDMARKS.
  · Los FACE_LANDMARKS publicados por LSC50 provienen de la cámara COLOR_FACE (50 fps, 640x424)
    y por eso tienen más filas que BODY; los de este script salen del MISMO video COLOR_BODY
    (24 fps), así que sus filas coinciden con BODY y manos, fotograma a fotograma.
  · Reloj: time.perf_counter_ns para todas las marcas.
  · Flags eliminados en D-05: --factor-recorte y --rostro-cuadro-completo (el recorte lo hace
    Holistic).

Uso:
    python src/reextract_lsc50.py                                   # los 1000 videos, salida por defecto
    python src/reextract_lsc50.py --muestras 0000_0000_0000 0025_0002_0001 --salida /tmp/prueba
    python src/reextract_lsc50.py --lista muestras.txt --limite 10 --sin-rostro
    python src/reextract_lsc50.py --reanudar                        # omite muestras ya extraídas
    python src/reextract_lsc50.py --nota "sin otra carga"            # texto libre en tasa_procesamiento.json
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent

# Rutas por defecto (se pueden cambiar por línea de comandos)
VIDEO_DIR = ROOT_DIR / "data" / "raw" / "LSC50" / "VIDEOS" / "COLOR_BODY"
OUTPUT_DIR = ROOT_DIR / "data" / "extracted" / "LSC50_holistic"   # D-05; la de marzo (LSC50) no se toca
MODELS_DIR = ROOT_DIR / "models" / "mediapipe"
# CSV publicados por los autores (solo lectura, para contrastar el número de filas)
PUBLICADOS_BODY = ROOT_DIR / "data" / "raw" / "LSC50" / "LANDMARKS" / "BODY_LANDMARKS"
PUBLICADOS_FACE = ROOT_DIR / "data" / "raw" / "LSC50" / "LANDMARKS" / "FACE_LANDMARKS"   # cámara COLOR_FACE, 50 fps

MODELO_HOLISTIC = "holistic_landmarker.task"
# SHA256 del modelo fijado en D-05 (float16/latest, 13.683.609 bytes, fechado 21-dic-2023).
SHA256_HOLISTIC = "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8"

# Solo el modelo de D-05: las tres tareas sueltas de D-01 (pose, hand, face) ya no se usan.
MODEL_URLS = {
    MODELO_HOLISTIC: "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
}

N_POSE = 33
N_HAND = 21
N_FACE = 468            # puntos que escriben los CSV (convención del Holistic antiguo / CSV publicados)
N_FACE_HOLISTIC = 478   # puntos que devuelve HolisticLandmarker (468 de la malla + 10 de iris)

# Opciones de HolisticLandmarker (mediapipe 0.10.35). Los umbrales 0.5 son los valores por
# defecto de la tarea; se pasan explícitos para que queden registrados en el JSON. El prototipo
# (src/translator.py) usa este mismo diccionario y tests/test_prototipo.py comprueba que ambos son
# idénticos: D-05 exige que corpus y prototipo usen exactamente la misma configuración.
OPCIONES_HOLISTIC = dict(min_face_detection_confidence=0.5, min_face_suppression_threshold=0.5,
                         min_face_landmarks_confidence=0.5, min_pose_detection_confidence=0.5,
                         min_pose_suppression_threshold=0.5, min_pose_landmarks_confidence=0.5,
                         min_hand_landmarks_confidence=0.5, output_face_blendshapes=False,
                         output_segmentation_mask=False)

COLUMNAS_METRICAS = [
    "fotograma", "ts_ms",
    "pose_detectada", "mano_izq_detectada", "mano_der_detectada", "n_manos", "rostro_detectado",
    "decodificar_ms", "holistic_ms", "total_ms",
]
ETAPAS_MS = ("decodificar", "holistic", "total")


# ----------------------------------------------------------------------------------------
# Modelo y utilidades
# ----------------------------------------------------------------------------------------

def sha256_archivo(ruta) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def ensure_modelo(nombre: str = MODELO_HOLISTIC, sha256_esperado: str | None = SHA256_HOLISTIC) -> tuple[str, str]:
    """Devuelve (ruta, sha256) del .task en models/mediapipe/; lo descarga desde MODEL_URLS si
    falta. Si se conoce el SHA256 esperado, lo verifica y aborta cuando no coincide: el modelo
    fijado en D-05 es parte de la configuración del extractor y un archivo distinto haría
    incomparables las extracciones."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = MODELS_DIR / nombre
    if not p.exists():
        url = MODEL_URLS[nombre]
        print(f"  Descargando {nombre} desde {url} ...")
        urllib.request.urlretrieve(url, str(p))
    sha = sha256_archivo(p)
    if sha256_esperado and sha != sha256_esperado:
        raise SystemExit(f"[ERROR] {p} no es el modelo fijado en D-05: SHA256 {sha} (esperado "
                         f"{sha256_esperado}). Sustituya el archivo o, si cambiar de modelo es una "
                         f"decisión documentada, actualice SHA256_HOLISTIC en este script.")
    return str(p), sha


def make_headers(prefix: str, n_landmarks: int) -> list[str]:
    """Cabeceras de datos compatibles con los CSV publicados y con preprocess_lsc50.py."""
    headers = []
    for i in range(n_landmarks):
        for c in ("x", "y", "z"):
            headers.append(f"{prefix}_{i}_{c}")
    return headers


def escribir_csv(ruta: Path, cabeceras: list[str], filas: list[list[float]]) -> None:
    """Escribe con la convención exacta de los CSV publicados por LSC50 (generados con
    pandas en Windows): primera columna índice SIN nombre, flotantes en repr más corto,
    finales de línea CRLF. Se usa el módulo csv estándar: no hace falta pandas."""
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow([""] + cabeceras)
        for i, fila in enumerate(filas):
            w.writerow([str(i)] + [repr(float(v)) for v in fila])


def escribir_metricas(ruta: Path, registros: list[dict]) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_METRICAS, lineterminator="\n")
        w.writeheader()
        for r in registros:
            w.writerow(r)


def resumen_ms(valores) -> dict:
    """media, DE (muestral, ddof=1), mediana, p90, mín, máx y n; en ms por fotograma."""
    a = np.asarray(valores, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "media": round(float(a.mean()), 3),
        "de": round(float(a.std(ddof=1)), 3) if a.size > 1 else 0.0,
        "mediana": round(float(np.median(a)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "min": round(float(a.min()), 3),
        "max": round(float(a.max()), 3),
    }


def info_hardware() -> dict:
    info = {
        "sistema_operativo": platform.platform(),
        "arquitectura": platform.machine(),
        "cpu": platform.processor() or None,
        "nucleos_fisicos": None,
        "nucleos_logicos": os.cpu_count(),
        "ram_gb": None,
        "gpu": "no se usa: delegado CPU en HolisticLandmarker",
    }
    if sys.platform == "darwin":
        def sysctl(clave: str) -> str:
            try:
                return subprocess.run(["sysctl", "-n", clave], capture_output=True,
                                      text=True, timeout=5).stdout.strip()
            except Exception:
                return ""
        info["cpu"] = sysctl("machdep.cpu.brand_string") or info["cpu"]
        info["nucleos_fisicos"] = int(sysctl("hw.physicalcpu") or 0) or None
        info["nucleos_logicos"] = int(sysctl("hw.logicalcpu") or 0) or info["nucleos_logicos"]
        mem = sysctl("hw.memsize")
        if mem:
            info["ram_gb"] = round(int(mem) / 2 ** 30, 1)
    elif sys.platform.startswith("linux"):
        try:
            info["ram_gb"] = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30, 1)
        except (ValueError, OSError):
            pass
    return info


def commit_git() -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(ROOT_DIR), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def carga_sistema():
    """Promedio de carga a 1, 5 y 15 min (documenta si había otros procesos, p. ej. un entrenamiento)."""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def filas_csv_publicado(stem: str, carpeta: Path = PUBLICADOS_BODY) -> int | None:
    """Filas de datos del CSV publicado por los autores en `carpeta` (solo lectura), o None.
    Por defecto BODY (cámara COLOR_BODY, 24 fps); con PUBLICADOS_FACE, el rostro publicado
    (cámara COLOR_FACE, 50 fps: no comparable fotograma a fotograma con COLOR_BODY)."""
    p = carpeta / f"{stem}.csv"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


def leer_cabecera_video(cap) -> dict:
    """Datos de entrada desde la cabecera de un cv2.VideoCapture (sin decodificar fotogramas).
    Si el contenedor no declara fps, se asume 30 y fps_valido queda en False."""
    if not cap.isOpened():
        return {"fps": 30.0, "fps_valido": False, "ancho": 0, "alto": 0,
                "fotogramas_cabecera": 0, "codec": ""}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    return {
        "fps": fps if fps > 0 else 30.0,
        "fps_valido": fps > 0,
        "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fotogramas_cabecera": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec": "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip(),
    }


# ----------------------------------------------------------------------------------------
# HolisticLandmarker: creación, marcas de tiempo y conversión del resultado a filas
# ----------------------------------------------------------------------------------------

def crear_holistic(ruta_modelo: str):
    """Un HolisticLandmarker en modo VIDEO con OPCIONES_HOLISTIC y delegado CPU. Se crea uno
    NUEVO por video: el modo VIDEO exige marcas de tiempo estrictamente crecientes."""
    vision = mp.tasks.vision
    return vision.HolisticLandmarker.create_from_options(vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=ruta_modelo,
                                          delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, **OPCIONES_HOLISTIC))


def ts_fotograma(indice: int, fps: float) -> int:
    """Marca de tiempo en ms del fotograma `indice` a `fps` fotogramas/s: round(i * 1000 / fps)."""
    return int(round(indice * 1000.0 / fps))


def fila_puntos(landmarks, n_puntos: int) -> list[float]:
    """Aplana los `n_puntos` primeros landmarks (objetos con x, y, z) en [x0, y0, z0, x1, ...].
    Fila de ceros (n_puntos * 3) si la lista está vacía o es None (componente no detectado).
    Para el rostro, n_puntos = N_FACE recorta los 478 puntos de Holistic a los 468 de la malla
    (sin iris). Si llegaran menos puntos que n_puntos, se completa con ceros para que la fila
    tenga siempre la anchura de la cabecera."""
    if not landmarks:
        return [0.0] * (n_puntos * 3)
    fila = [float(c) for lm in landmarks[:n_puntos] for c in (lm.x, lm.y, lm.z)]
    if len(fila) < n_puntos * 3:
        fila.extend([0.0] * (n_puntos * 3 - len(fila)))
    return fila


def filas_desde_resultado(res, con_rostro: bool = True) -> dict:
    """Convierte un HolisticLandmarkerResult (o un doble con los mismos atributos) en las
    cuatro filas de los CSV y las banderas de detección. left_hand_landmarks = mano izquierda
    DE LA PERSONA, tal como lo entrega Holistic y como lo escriben los CSV publicados.
    Las listas vacías (componente no detectado) producen filas de ceros. Con con_rostro=False
    no se arma la fila del rostro (None), aunque la bandera rostro_detectado sí se calcula."""
    pose_det = bool(res.pose_landmarks)
    lh_det = bool(res.left_hand_landmarks)
    rh_det = bool(res.right_hand_landmarks)
    rostro_det = bool(res.face_landmarks)
    return {
        "body": fila_puntos(res.pose_landmarks, N_POSE),
        "lh": fila_puntos(res.left_hand_landmarks, N_HAND),
        "rh": fila_puntos(res.right_hand_landmarks, N_HAND),
        "face": fila_puntos(res.face_landmarks, N_FACE) if con_rostro else None,
        "pose_detectada": pose_det,
        "mano_izq_detectada": lh_det,
        "mano_der_detectada": rh_det,
        "n_manos": int(lh_det) + int(rh_det),
        "rostro_detectado": rostro_det,
    }


# ----------------------------------------------------------------------------------------
# Extracción de un video
# ----------------------------------------------------------------------------------------

def extraer_video(video_path: Path, holistic, con_rostro: bool = True) -> dict:
    """Extrae los landmarks de un video con cronómetro por fotograma y etapa (decodificar,
    holistic, total). Devuelve filas por componente, registros por fotograma y datos de
    entrada. `holistic` debe ser un HolisticLandmarker recién creado para este video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {video_path.name}")
    entrada = leer_cabecera_video(cap)
    fps = entrada["fps"]

    body_rows, lh_rows, rh_rows, face_rows, registros = [], [], [], [], []
    idx = 0
    while True:
        t_ini = time.perf_counter_ns()
        ret, frame = cap.read()
        if not ret:
            break
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = ts_fotograma(idx, fps)
        t_dec = time.perf_counter_ns()

        res = holistic.detect_for_video(mp_img, ts)   # postura + rostro + manos en un solo grafo
        t_hol = time.perf_counter_ns()

        filas = filas_desde_resultado(res, con_rostro)
        body_rows.append(filas["body"])
        lh_rows.append(filas["lh"])
        rh_rows.append(filas["rh"])
        if con_rostro:
            face_rows.append(filas["face"])
        t_fin = time.perf_counter_ns()

        registros.append({
            "fotograma": idx, "ts_ms": ts,
            "pose_detectada": int(filas["pose_detectada"]),
            "mano_izq_detectada": int(filas["mano_izq_detectada"]),
            "mano_der_detectada": int(filas["mano_der_detectada"]),
            "n_manos": filas["n_manos"],
            "rostro_detectado": int(filas["rostro_detectado"]),
            "decodificar_ms": round((t_dec - t_ini) / 1e6, 3),
            "holistic_ms": round((t_hol - t_dec) / 1e6, 3),
            "total_ms": round((t_fin - t_ini) / 1e6, 3),
        })
        idx += 1

    cap.release()
    entrada["fotogramas_decodificados"] = idx
    return {"body": body_rows, "lh": lh_rows, "rh": rh_rows, "face": face_rows,
            "registros": registros, "entrada": entrada}


def recuperar_desde_metricas(video_path: Path, ruta_metricas: Path) -> dict | None:
    """Al reanudar (--reanudar), reconstruye los registros por fotograma de una muestra ya
    extraída a partir de su METRICAS_FOTOGRAMA/<muestra>.csv (banderas y ms por etapa de la
    corrida anterior) y los datos de entrada desde la cabecera del video (sin decodificarlo),
    para que tasa_procesamiento.json cubra también las muestras omitidas. None si el CSV de
    métricas no se puede leer, está vacío o es de la versión anterior del script (tres tareas)."""
    try:
        with open(ruta_metricas, newline="", encoding="utf-8") as f:
            lector = csv.DictReader(f)
            filas = list(lector)
            columnas = lector.fieldnames or []
    except (OSError, csv.Error):
        return None
    if not filas or set(COLUMNAS_METRICAS) - set(columnas):
        return None
    registros = []
    for r in filas:
        reg = {}
        for k in COLUMNAS_METRICAS:
            v = r.get(k, "")
            reg[k] = float(v or 0.0) if k.endswith("_ms") else int(float(v or 0))
        registros.append(reg)
    cap = cv2.VideoCapture(str(video_path))
    entrada = leer_cabecera_video(cap)
    cap.release()
    entrada["fotogramas_decodificados"] = len(registros)
    return {"registros": registros, "entrada": entrada,
            "segundos": sum(r["total_ms"] for r in registros) / 1000.0}


def resumir_video(stem: str, res: dict, segundos: float) -> dict:
    """Resumen por video para el JSON global y la consola."""
    reg = res["registros"]
    n = len(reg)
    ent = res["entrada"]
    publicadas = filas_csv_publicado(stem)

    def cuenta(clave):
        return int(sum(r[clave] for r in reg))

    def media(clave):
        return round(float(np.mean([r[clave] for r in reg])), 3) if n else None

    sin_mano_alguna = int(sum(1 for r in reg if not r["mano_izq_detectada"] and not r["mano_der_detectada"]))
    return {
        "muestra": stem,
        "fotogramas": n,
        "fotogramas_cabecera": ent["fotogramas_cabecera"],
        "fps": ent["fps"],
        "segundos_clip": round(n / ent["fps"], 3) if ent["fps"] else None,
        "resolucion": f"{ent['ancho']}x{ent['alto']}",
        "codec": ent["codec"],
        "filas_csv_publicado": publicadas,
        "video_menos_publicado": (n - publicadas) if publicadas is not None else None,
        "filas_csv_publicado_rostro_otra_camara": filas_csv_publicado(stem, PUBLICADOS_FACE),
        "pose_detectada": cuenta("pose_detectada"),
        "mano_izq_detectada": cuenta("mano_izq_detectada"),
        "mano_der_detectada": cuenta("mano_der_detectada"),
        "rostro_detectado": cuenta("rostro_detectado"),
        "sin_pose": n - cuenta("pose_detectada"),
        "sin_mano_izq": n - cuenta("mano_izq_detectada"),
        "sin_mano_der": n - cuenta("mano_der_detectada"),
        "sin_mano_alguna": sin_mano_alguna,
        "sin_rostro": n - cuenta("rostro_detectado"),
        "ms_medio": {k: media(f"{k}_ms") for k in ETAPAS_MS},
        "segundos_video": round(segundos, 3),
    }


# ----------------------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------------------

def parsear_argumentos(argv=None):
    p = argparse.ArgumentParser(
        description="Re-extracción de LSC50 con MediaPipe HolisticLandmarker (D-05) con cronómetro. "
                    "Eliminados en D-05: --factor-recorte y --rostro-cuadro-completo (el recorte lo hace Holistic).")
    p.add_argument("--videos", type=Path, default=VIDEO_DIR,
                   help=f"carpeta con los .avi (por defecto {VIDEO_DIR.relative_to(ROOT_DIR)})")
    p.add_argument("--salida", type=Path, default=OUTPUT_DIR,
                   help=f"carpeta de salida (por defecto {OUTPUT_DIR.relative_to(ROOT_DIR)})")
    p.add_argument("--muestras", nargs="+", metavar="SSSS_VVVV_RRRR",
                   help="procesar solo estas muestras (nombres sin extensión)")
    p.add_argument("--lista", type=Path, help="archivo de texto con una muestra por línea")
    p.add_argument("--limite", type=int, help="procesar como máximo N videos (tras ordenar)")
    p.add_argument("--sin-rostro", action="store_true",
                   help="no escribir FACE_LANDMARKS (el grafo de Holistic calcula el rostro igual)")
    p.add_argument("--reanudar", action="store_true",
                   help="omitir las muestras cuyos CSV ya existen en la salida (para reanudar corridas largas)")
    p.add_argument("--nota", default="",
                   help="condiciones de la prueba (texto libre) que se guardan en tasa_procesamiento.json")
    return p.parse_args(argv)


def main(argv=None):
    args = parsear_argumentos(argv)
    con_rostro = not args.sin_rostro
    t_pared_ini = time.perf_counter_ns()
    carga_inicio = carga_sistema()

    print("=" * 72)
    print("  Re-extracción de LSC50 con MediaPipe HolisticLandmarker (D-05)"
          + ("" if con_rostro else "  [--sin-rostro: no se escribe FACE_LANDMARKS]"))
    print("=" * 72)

    # Carpetas de salida
    salida = args.salida
    out_body, out_lh, out_rh = salida / "BODY_LANDMARKS", salida / "LEFT_HAND_LANDMARKS", salida / "RIGHT_HAND_LANDMARKS"
    out_face, out_met = salida / "FACE_LANDMARKS", salida / "METRICAS_FOTOGRAMA"
    carpetas = [out_body, out_lh, out_rh, out_met] + ([out_face] if con_rostro else [])
    for d in carpetas:
        d.mkdir(parents=True, exist_ok=True)

    # Modelo (descarga si falta; verificación del SHA256 fijado en D-05)
    ruta_modelo, sha_modelo = ensure_modelo()
    modelos_json = {
        MODELO_HOLISTIC: {
            "ruta": (str(Path(ruta_modelo).relative_to(ROOT_DIR))
                     if ruta_modelo.startswith(str(ROOT_DIR)) else ruta_modelo),
            "url_origen": MODEL_URLS[MODELO_HOLISTIC],
            "bytes": os.path.getsize(ruta_modelo),
            "sha256": sha_modelo,
            "sha256_esperado_d05": SHA256_HOLISTIC,
            "opciones": {"tarea": "HolisticLandmarker", "running_mode": "VIDEO",
                         **OPCIONES_HOLISTIC, "delegate": "CPU"},
        }
    }

    # Videos a procesar
    videos = sorted(args.videos.glob("*.avi"))
    if not videos:
        print("[ERROR] No se encontraron videos en", args.videos)
        sys.exit(1)
    pedidas = set(args.muestras or [])
    if args.lista:
        pedidas |= {l.strip() for l in args.lista.read_text().splitlines() if l.strip()}
    if pedidas:
        disponibles = {v.stem: v for v in videos}
        faltan = sorted(pedidas - disponibles.keys())
        if faltan:
            print("[ERROR] Muestras no encontradas en", args.videos, ":", ", ".join(faltan))
            sys.exit(1)
        videos = [disponibles[s] for s in sorted(pedidas)]
    if args.limite:
        videos = videos[: args.limite]
    total = len(videos)
    print(f"\n  Videos a procesar: {total}   Salida: {salida}")

    body_headers = make_headers("landmark", N_POSE)
    hand_headers = make_headers("landmark", N_HAND)
    face_headers = make_headers("landmark", N_FACE)

    por_video, fallidos, omitidos = [], [], []
    tiempos = {k: [] for k in ETAPAS_MS}
    ruta_json = salida / "tasa_procesamiento.json"

    def escribir_json(estado: str):
        n_fot = int(sum(v["fotogramas"] for v in por_video))
        # fotogramas realmente procesados en ESTA corrida (excluye los recuperados con --reanudar)
        n_fot_corrida = int(sum(v["fotogramas"] for v in por_video if not v.get("recuperado_de_metricas_previas")))
        seg_bucle = sum(tiempos["total"]) / 1000.0
        seg_pared = (time.perf_counter_ns() - t_pared_ini) / 1e9
        resoluciones = sorted({v["resolucion"] for v in por_video})
        fps_in = sorted({v["fps"] for v in por_video})
        fot_por_video = [v["fotogramas"] for v in por_video]
        seg_por_video = [v["segundos_clip"] for v in por_video if v["segundos_clip"] is not None]
        ms = {k: resumen_ms(v) for k, v in tiempos.items()}
        suma_etapas = ms["decodificar"].get("media", 0) + ms["holistic"].get("media", 0)
        datos = {
            "descripcion": "Tasa de procesamiento del extractor de puntos característicos (MediaPipe "
                           "HolisticLandmarker, Tasks API) en el hardware del proyecto: actividad 1.c / "
                           "§2.2.3 del anteproyecto, decisión D-05.",
            "estado": estado,
            "fecha_hora": dt.datetime.now().isoformat(timespec="seconds"),
            "comando": [sys.argv[0]] + sys.argv[1:],
            "git_commit": commit_git(),
            "condiciones_de_prueba": {
                "nota": args.nota,
                "carga_sistema_1_5_15_min_inicio": carga_inicio,
                "carga_sistema_1_5_15_min_fin": carga_sistema(),
                "explicacion": "promedio de carga (os.getloadavg) al iniciar y al terminar: si supera con holgura "
                               "1.0 había otros procesos compitiendo por la CPU y los tiempos son un techo",
            },
            "hardware": info_hardware(),
            "software": {"python": platform.python_version(), "mediapipe": mp.__version__,
                         "opencv": cv2.__version__, "numpy": np.__version__},
            "reloj": "time.perf_counter_ns (monótono) para todas las marcas; ms = ns / 1e6",
            "modelos": modelos_json,
            "entrada": {
                "directorio": str(args.videos),
                "resoluciones": resoluciones,
                "fps": fps_in,
                "codec": sorted({v.get("codec", "") for v in por_video}) if por_video else [],
                "n_videos": len(por_video),
                "n_fotogramas": n_fot,
                "n_fotogramas_segun_cabecera": int(sum(v["fotogramas_cabecera"] for v in por_video)),
                "muestras_pedidas": sorted(pedidas) if pedidas else None,
                "limite": args.limite,
                "ventana_temporal_por_sena": {
                    "nota": "tamaño de la ventana temporal que pide §2.2.3: fotogramas por seña (clip completo) "
                            "y su duración a la tasa de entrada",
                    "fotogramas": resumen_ms(fot_por_video) if fot_por_video else {"n": 0},
                    "segundos": resumen_ms(seg_por_video) if seg_por_video else {"n": 0},
                },
            },
            "salida": {
                "directorio": str(salida),
                "carpetas": [d.name for d in carpetas],
                "columnas_datos": {"BODY_LANDMARKS": N_POSE * 3, "LEFT_HAND_LANDMARKS": N_HAND * 3,
                                   "RIGHT_HAND_LANDMARKS": N_HAND * 3,
                                   "FACE_LANDMARKS": N_FACE * 3 if con_rostro else None},
                "convencion_csv": "columna índice sin nombre + landmark_i_x/y/z (i desde 0), una fila por fotograma "
                                  "decodificado, flotantes en repr más corto, finales de línea CRLF: la misma de los "
                                  "CSV publicados por LSC50",
                "relleno_sin_deteccion": "ningún fotograma se descarta; si un componente de Holistic (postura, mano "
                                         "izquierda, mano derecha, rostro) no se detecta en un fotograma, su fila se "
                                         "escribe con ceros (99, 63, 63 o 1404 columnas). Las banderas por fotograma "
                                         "están en METRICAS_FOTOGRAMA/<muestra>.csv (pose_detectada, "
                                         "mano_izq_detectada, mano_der_detectada, rostro_detectado)",
                "marca_de_tiempo_video": "ts_ms = round(i * 1000 / fps) por fotograma; un HolisticLandmarker nuevo "
                                         "por video (modo VIDEO: marcas de tiempo estrictamente crecientes)",
                "asignacion_manos": "interna de HolisticLandmarker; left_hand_landmarks = mano izquierda de la persona",
                "lateralidad_manos": "LEFT_HAND_LANDMARKS = mano izquierda DE LA PERSONA (cámara en tercera persona sin "
                                     "espejo), la misma convención de los CSV publicados. Verificado 14-sep-2026 (D-05) "
                                     "en 0000_0000_0000: la muñeca del LEFT_HAND publicado coincide con "
                                     "left_hand_landmarks con |dx|+|dy| medio 0,0013",
                "rostro": ("detectado por el grafo de Holistic sobre el cuadro completo; se escriben 468 de 478 puntos"
                           if con_rostro else
                           "desactivado (--sin-rostro): el grafo de Holistic lo calcula igual, pero no se escribe "
                           "FACE_LANDMARKS"),
                "rostro_puntos": f"HolisticLandmarker devuelve {N_FACE_HOLISTIC} puntos (468 de la malla + 10 de iris, "
                                 f"índices 468-477); se escriben los {N_FACE} primeros ({N_FACE * 3} columnas) para "
                                 f"conservar los 543 puntos de §1.4 y la convención de los CSV publicados",
                "rostro_publicado_no_comparable": "los FACE_LANDMARKS publicados por LSC50 salen de la cámara COLOR_FACE "
                                                  "(50 fps, 640x424) y tienen más filas que BODY; los de este script salen "
                                                  "del mismo video COLOR_BODY, con una fila por fotograma como BODY y manos "
                                                  "(por_video[].filas_csv_publicado_rostro_otra_camara guarda las filas "
                                                  "publicadas, solo como referencia)",
                "decisiones": "D-05 (HolisticLandmarker como extractor único; corrige D-01). D-03 (recorte manual del "
                              "rostro desde la postura) y D-04 (asignación manual de manos por muñeca) quedan sin "
                              "efecto en este extractor: las hace el grafo de Holistic por dentro",
            },
            "deteccion": {
                "fotogramas": n_fot,
                "pose_detectada": int(sum(v["pose_detectada"] for v in por_video)),
                "mano_izq_detectada": int(sum(v["mano_izq_detectada"] for v in por_video)),
                "mano_der_detectada": int(sum(v["mano_der_detectada"] for v in por_video)),
                "rostro_detectado": int(sum(v["rostro_detectado"] for v in por_video)),
                "sin_pose": int(sum(v["sin_pose"] for v in por_video)),
                "sin_mano_izq": int(sum(v["sin_mano_izq"] for v in por_video)),
                "sin_mano_der": int(sum(v["sin_mano_der"] for v in por_video)),
                "sin_mano_alguna": int(sum(v["sin_mano_alguna"] for v in por_video)),
                "sin_rostro": int(sum(v["sin_rostro"] for v in por_video)),
            },
            "tiempos_ms_por_fotograma": {
                "nota": "por fotograma, sobre todos los fotogramas de todos los videos procesados; DE muestral (ddof=1). "
                        "'decodificar' = cap.read + BGR->RGB + mp.Image; 'holistic' = detect_for_video (postura, rostro "
                        "y manos en un solo grafo); 'total' = bucle completo por fotograma (las dos etapas más el "
                        "ensamblado de filas)",
                **ms,
                "suma_medias_etapas": round(suma_etapas, 3),
            },
            "tasa_efectiva": {
                "fotogramas_por_segundo_bucle": round(n_fot / seg_bucle, 2) if seg_bucle > 0 else None,
                "fotogramas_por_segundo_pared": (round(n_fot_corrida / seg_pared, 2)
                                                 if (seg_pared > 0 and n_fot_corrida > 0) else None),
                "fotogramas_esta_corrida": n_fot_corrida,
                "segundos_bucle": round(seg_bucle, 2),
                "segundos_pared": round(seg_pared, 2),
                "nota": "'bucle' = suma de los tiempos por fotograma (incluye los recuperados con --reanudar); "
                        "'pared' = corrida completa (carga del modelo, creación del detector por video, escritura "
                        "de CSV) sobre los fotogramas procesados en esta corrida; None si no se procesó ninguno",
            },
            "por_video": por_video,
            "fallidos": fallidos,
            "omitidos_por_reanudar": omitidos,
            "nota_reanudar": ("con --reanudar, las muestras omitidas no se reprocesan; su resumen y sus tiempos "
                              "por fotograma se recuperan de METRICAS_FOTOGRAMA/<muestra>.csv de la corrida "
                              "anterior (por_video[].recuperado_de_metricas_previas = true) y entran en "
                              "'deteccion', 'tiempos_ms_por_fotograma' y 'tasa_efectiva' (bucle); "
                              "'fotogramas_por_segundo_pared' solo cubre esta corrida"
                              if omitidos else None),
        }
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)
        return datos

    def acumular(resumen: dict, registros: list[dict]) -> None:
        por_video.append(resumen)
        for r in registros:
            for k in ETAPAS_MS:
                tiempos[k].append(r[f"{k}_ms"])

    try:
        for k, video_path in enumerate(videos, 1):
            stem = video_path.stem
            salidas = [out_body / f"{stem}.csv", out_lh / f"{stem}.csv", out_rh / f"{stem}.csv", out_met / f"{stem}.csv"]
            if con_rostro:
                salidas.append(out_face / f"{stem}.csv")
            if args.reanudar and all(s.exists() for s in salidas):
                # Ya extraída: no se vuelve a procesar, pero su resumen entra al JSON global
                # a partir de las métricas que dejó la corrida anterior (marcado como tal).
                # Revisión D-05: si METRICAS_FOTOGRAMA no es de esta versión (p. ej. CSV de la
                # versión de tres tareas) o no se puede leer, los CSV no son de este extractor:
                # se avisa y se reprocesa en vez de omitirlos en silencio.
                rec = recuperar_desde_metricas(video_path, out_met / f"{stem}.csv")
                if rec is not None:
                    omitidos.append(stem)
                    resumen = resumir_video(stem, rec, rec["segundos"])
                    resumen["recuperado_de_metricas_previas"] = True
                    acumular(resumen, rec["registros"])
                    continue
                print(f"  [AVISO] {stem}: los CSV existentes no son de esta versión del extractor "
                      f"(METRICAS_FOTOGRAMA sin las columnas de D-05); se reprocesa y se sobrescriben.")

            t_v0 = time.perf_counter_ns()
            holistic = crear_holistic(ruta_modelo)
            try:
                res = extraer_video(video_path, holistic, con_rostro)
            except Exception as e:  # el video falla, la corrida sigue
                print(f"  [ERROR] {stem}: {e}")
                fallidos.append({"muestra": stem, "error": str(e)})
                continue
            finally:
                holistic.close()

            escribir_csv(out_body / f"{stem}.csv", body_headers, res["body"])
            escribir_csv(out_lh / f"{stem}.csv", hand_headers, res["lh"])
            escribir_csv(out_rh / f"{stem}.csv", hand_headers, res["rh"])
            if con_rostro:
                escribir_csv(out_face / f"{stem}.csv", face_headers, res["face"])
            escribir_metricas(out_met / f"{stem}.csv", res["registros"])

            seg = (time.perf_counter_ns() - t_v0) / 1e9
            resumen = resumir_video(stem, res, seg)
            acumular(resumen, res["registros"])

            n = resumen["fotogramas"]
            print(f"  [{k:4d}/{total}] {stem}  {n:3d} fot. ({resumen['resolucion']} @ {resumen['fps']:.0f} fps)"
                  f"  pose {resumen['pose_detectada']}/{n}"
                  f"  mI {resumen['mano_izq_detectada']}/{n}  mD {resumen['mano_der_detectada']}/{n}"
                  f"  rostro {resumen['rostro_detectado']}/{n}"
                  f"  {resumen['ms_medio']['total']:.1f} ms/fot.  {seg:.1f} s"
                  + (f"  publicado {resumen['filas_csv_publicado']} filas" if resumen["filas_csv_publicado"] is not None else ""))
            if k % 25 == 0:
                escribir_json("en curso")
    except KeyboardInterrupt:
        print("\n  Interrumpido por la persona usuaria; se guarda el resumen parcial.")
        escribir_json("interrumpido")
        raise

    datos = escribir_json("completado")

    recuperados = sum(1 for v in por_video if v.get("recuperado_de_metricas_previas"))
    print(f"\n  Completado: {len(por_video) - recuperados} videos procesados, {len(fallidos)} fallidos, "
          f"{len(omitidos)} omitidos"
          + (f" ({recuperados} con resumen recuperado de métricas previas)" if recuperados else ""))
    d = datos["deteccion"]
    print(f"  Fotogramas: {d['fotogramas']}   sin pose {d['sin_pose']}   sin mano izq {d['sin_mano_izq']}"
          f"   sin mano der {d['sin_mano_der']}   sin mano alguna {d['sin_mano_alguna']}   sin rostro {d['sin_rostro']}")
    print("  ms por fotograma (media / DE / mediana / p90):")
    for etapa in ETAPAS_MS:
        m = datos["tiempos_ms_por_fotograma"][etapa]
        if m.get("n"):
            print(f"    {etapa:12s} {m['media']:8.3f} / {m['de']:7.3f} / {m['mediana']:8.3f} / {m['p90']:8.3f}")
    te = datos["tasa_efectiva"]
    print(f"  Tasa efectiva: {te['fotogramas_por_segundo_bucle']} fot./s (bucle)  "
          f"{te['fotogramas_por_segundo_pared']} fot./s (pared, {te['segundos_pared']} s)")
    print(f"  Hardware: {datos['hardware']['cpu']}, {datos['hardware']['nucleos_logicos']} núcleos, "
          f"{datos['hardware']['ram_gb']} GB RAM")
    print(f"  Resumen global: {ruta_json}")
    return datos


if __name__ == "__main__":
    main()
