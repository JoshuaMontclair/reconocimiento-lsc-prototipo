"""
Bitácora de integridad (actividad 1.a) y caracterización (actividad 1.b) del corpus LSC50.

Lee data/raw/LSC50 (LANDMARKS, VIDEOS, IMU/INFO) y data/processed/metadata.json, y escribe en
data/processed/caracterizacion/:

  bitacora_integridad.json  Por muestra: presencia y tamaño de cada archivo (4 CSV + 4 videos),
                            filas / columnas / NaN / filas todo-cero / índice de cada CSV,
                            fotogramas / fps / resolución / códec de cada video (cabecera AVI leída
                            con struct: NO se decodifica ningún fotograma), diferencias entre
                            modalidades y lista de anomalías. Agregados: recuento por anomalía,
                            duplicados, ausentes, ilegibles, identidad y versión del corpus, sumas de
                            comprobación, fecha de la verificación y comprobaciones frente a N-02.
  bitacora_integridad.csv   La misma tabla por muestra, plana, para hoja de cálculo.
  caracterizacion.json      Vocabulario con ID, muestras por clase / señante / repetición,
                            estadísticas de fotogramas por modalidad, resolución y fps MEDIDOS por
                            cámara, desalineamiento rostro/cuerpo, señantes tal como los publica el
                            artículo (dos criterios sin conciliar), condición dinámica/estática por
                            seña (PENDIENTE 1.d) y diferencias entre landmarks publicados y video.
  caracterizacion.md        Lo mismo en prosa y tablas, para el informe. Cada cifra dice su fuente.

Además:
  · Duplicados: CSV por SHA-1 completo; videos por tamaño + SHA-1 de los primeros 64 KiB y, para los
    candidatos, SHA-1 del archivo completo (confirmados o refutados). Los pares confirmados se cotejan
    con data/partitions/*.json para saber si quedan repartidos entre conjuntos (fuga por duplicado).
  · Sumas de comprobación: figshare publica un MD5 por ZIP (instantánea de su API del 14-sep-2026
    incluida; --consultar-figshare la refresca en vivo). Si los ZIP descargados están en disco
    (--zips RUTA), se cotejan por tamaño y, con --verificar-md5, por MD5.
  · Lo que el artículo descriptor declara (señantes, cámaras, escena, cifras) va como citas literales
    tomadas de su texto completo (Europe PMC, PMC11655960, consultado el 14-sep-2026) y se contrasta
    con lo medido en disco.

Solo numpy y biblioteca estándar: pandas no está instalado en el entorno del proyecto.
No escribe nada en data/raw ni en data/extracted. No decodifica ningún fotograma.

Uso:
    python src/caracterizar_lsc50.py
    python src/caracterizar_lsc50.py --limite 20 --salida /ruta/de/prueba     (prueba de humo)
    python src/caracterizar_lsc50.py --sin-hash                                (más rápido, sin duplicados)
    python src/caracterizar_lsc50.py --zips ~/Downloads --verificar-md5        (cotejo MD5 de los ZIP)
    python src/caracterizar_lsc50.py --autoprueba                              (unittest con datos sintéticos)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import platform
import re
import struct
import sys
import tempfile
import time
import unicodedata
import unittest
import urllib.error
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
RAW_DIR = BASE_DIR / "data" / "raw" / "LSC50"
METADATA_PATH = BASE_DIR / "data" / "processed" / "metadata.json"
SALIDA_DIR = BASE_DIR / "data" / "processed" / "caracterizacion"
PARTICIONES_DIR = BASE_DIR / "data" / "partitions"      # D-02: índices de las dos particiones
DATA_DIR = BASE_DIR / "data"                             # donde se buscan los ZIP originales del depósito

# Rejilla de nombres que el corpus declara publicar: 50 señas × 5 voluntarios × 4 repeticiones
N_SENAS, N_SENANTES, N_REPETICIONES = 50, 5, 4

# Modalidades de landmarks publicados (clave interna → subruta y nº de puntos esperado)
MODALIDADES_CSV = {
    "body": ("LANDMARKS/BODY_LANDMARKS", 33),
    "lh": ("LANDMARKS/HANDS_LANDMARKS/LEFT_HAND_LANDMARKS", 21),
    "rh": ("LANDMARKS/HANDS_LANDMARKS/RIGHT_HAND_LANDMARKS", 21),
    "face": ("LANDMARKS/FACE_LANDMARKS", 468),
}
# Cámaras (clave → subruta)
CAMARAS = {
    "COLOR_BODY": "VIDEOS/COLOR_BODY",
    "COLOR_FACE": "VIDEOS/COLOR_FACE",
    "GRAY_DEPTH": "VIDEOS/GRAY_DEPTH",
    "GRAY_IR": "VIDEOS/GRAY_IR",
}
# Nº de bytes de cabecera que se leen de cada AVI (los chunks avih/strh/strf caben de sobra)
BYTES_CABECERA_AVI = 65536
# Bytes que se hashean de cada video para la heurística de duplicados (no se decodifica)
BYTES_HASH_VIDEO = 65536

# Identidad del corpus. Fuentes: README.md (enlaces) y anteproyecto 13_referencias.tex ref19 / ref19b.
IDENTIDAD_CORPUS = {
    "nombre": "LSC50: Colombian Sign Language Video and Inertial Measurement Dataset",
    "articulo_descriptor": {
        "cita": "A. F. Flórez-Sierra, B. D. Solórzano, F. Segura-Quijano, Y. M. Cortés-Bello, L. Cubillos, "
                "L. F. Giraldo y C. Cifuentes-De la Portilla, Scientific Data, vol. 11, art. 1347, dic. 2024",
        "doi": "10.1038/s41597-024-04172-5",
        "licencia": "CC BY-NC-ND 4.0 (rige sobre el texto y las figuras del artículo, no sobre los datos)",
        "fuente": "anteproyecto 13_referencias.tex, \\bibitem{ref19}; README.md (Datasets)",
    },
    "deposito_de_datos": {
        "repositorio": "figshare",
        "doi": "10.6084/m9.figshare.27383016.v1",
        "version": "1",
        "fecha_publicacion": "2024-10-31",
        "url": "https://figshare.com/articles/dataset/LSC50/27383016",
        "codigo_autores": "https://github.com/BiomecanicaUniandes/LSC50",
        "licencia": "CC BY 4.0",
        "fuente": "anteproyecto 13_referencias.tex, \\bibitem[19b]{ref19b}; README.md (Datasets)",
    },
}

# Instantánea de la API pública de figshare para el depósito, consultada el 14-sep-2026
# (GET https://api.figshare.com/v2/articles/27383016). Es lo ÚNICO que la fuente publica como suma de
# comprobación: un MD5 por archivo ZIP, no por CSV ni por video. --consultar-figshare la refresca en vivo.
FIGSHARE_INSTANTANEA = {
    "api": "https://api.figshare.com/v2/articles/27383016",
    "fecha_consulta": "2026-09-14",
    "titulo": "LSC50: Colombian Sign Language Video and Inertial Measurement Dataset",
    "doi": "10.6084/m9.figshare.27383016.v1",
    "version": 1,
    "published_date": "2024-10-31T18:04:32Z",
    "modified_date": "2024-10-31T18:04:32Z",
    "licencia": "CC BY 4.0",
    "autores": ["Christian Cifuentes-De la Portilla", "Luis Felipe Giraldo", "Brayan David Solórzano Quevedo"],
    "archivos": [
        {"name": "IMU.zip", "size": 1608738643, "computed_md5": "f94fc68d1ec5da41f9074e77d017a9c8",
         "download_url": "https://ndownloader.figshare.com/files/50115549"},
        {"name": "LANDMARKS.zip", "size": 2770068538, "computed_md5": "996d3e9ee1d3fb39866ce395efd23214",
         "download_url": "https://ndownloader.figshare.com/files/50117670"},
        {"name": "VIDEOS.zip", "size": 6247307413, "computed_md5": "26003107442257c7947352b18e3c2597",
         "download_url": "https://ndownloader.figshare.com/files/50118645"},
    ],
}

# Citas LITERALES (en inglés) del artículo descriptor, tomadas de su texto completo en Europe PMC.
# Se transcriben tal cual; «[14]» es la llamada bibliográfica del propio artículo a su depósito.
ARTICULO_FUENTE = (
    "Texto completo del artículo descriptor (ref19, doi 10.1038/s41597-024-04172-5) en Europe PMC, "
    "PMC11655960 (https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11655960/fullTextXML), consultado "
    "el 14-sep-2026. Citas literales en inglés; la sección se indica en cada una."
)
ARTICULO_CITAS = {
    "senantes": [
        {"seccion": "Methods",
         "cita": "Participants were recruited through the Association of Colombian-Spanish Sign Language "
                 "Interpreters. Two of them are deaf and three of them are not hearing impaired."},
        {"seccion": "Methods",
         "cita": "All participants are fluent in Colombian Sign Language and were above 18 years old at the "
                 "time of their participation in the data collection process."},
        {"seccion": "Data Records",
         "cita": "The total information in the dataset[14] is equally distributed in the number of classes "
                 "(50 signs) and the number of participants involved (5 volunteers: 3 native speakers and two "
                 "non-native speakers)."},
        {"seccion": "Technical Validation (Sign validation)",
         "cita": "These LSC interpreters are non-native speakers and are some of the participants that can "
                 "be seen in the stored data."},
    ],
    "registro": [
        {"seccion": "Methods (Video data acquisition)", "camara": "COLOR_FACE",
         "cita": "The first video corresponds to an RGB video of the face of the LSC speaker taken with a "
                 "Nikon D5600 DSLR camera with an AF-P DX NIKKOR lens with a focal length of 55 millimeters. "
                 "A diaphragm opening of 3.5 G, video that records at a rate of 50 frames per second (FPS) at "
                 "a resolution of 640 by 424 pixels"},
        {"seccion": "Methods (Video data acquisition)", "camara": "COLOR_BODY",
         "cita": "The second type of video corresponds to an RGB capture of the LSC speaker’s body from the "
                 "knees to a height of maximum vertical extension of the arms. This type of video is recorded "
                 "at a rate of 24 FPS, and a resolution of 1920 by 1080 pixels"},
        {"seccion": "Methods (Video data acquisition)", "camara": "GRAY_DEPTH",
         "cita": "The third type corresponds to a grayscale video representing the scene’s depth. This video "
                 "is recorded at a rate of 30 FPS at a resolution of 512 by 424 pixels"},
        {"seccion": "Methods (Video data acquisition)", "camara": "GRAY_IR",
         "cita": "This video is recorded at the same rate and resolution as the one representing the depth"},
        {"seccion": "Methods (Video data acquisition)", "camara": "COLOR_BODY, GRAY_DEPTH, GRAY_IR",
         "cita": "For RGB full body, depth, and IR videos, the Microsoft Kinect V2 is used, which has RGB "
                 "sensors, infrared emission, and reception that work at the mentioned rates and resolutions."},
        {"seccion": "Methods (Video data acquisition)", "camara": "escena",
         "cita": "Regarding the spatial distribution of the recording set, a group of cameras is located at a "
                 "distance of 200 centimeters from the LSC speaker, which is 10 centimeters from a blue curtain "
                 "of dimensions three by three meters."},
        {"seccion": "Methods (Video data acquisition)", "camara": "escena",
         "cita": "The height of the Nikon d 5600 DSLR camera is adjusted to eye level for the LSC speaker, and "
                 "the Microsoft Kinect V2 sits 5 centimeters below the DSLR camera, where both are docked on a "
                 "3D printed platform and mounted on a tripod that allows adjusting the height and inclination "
                 "in the three axes."},
        {"seccion": "Methods (Video data acquisition)", "camara": "escena",
         "cita": "The lighting must be white and homogeneous."},
    ],
    "landmarks": [
        {"seccion": "Methods (Video landmarks acquisition)",
         "cita": "For this, the holistic models provided by the MediaPipe framework are used."},
        {"seccion": "Methods (Video landmarks acquisition)",
         "cita": "MediaPipe Holistic uses the pose, face, and hand cue point models in MediaPipe Pose, "
                 "MediaPipe Face Mesh, and MediaPipe Hands, respectively, to generate a total of 543 cue "
                 "points; 33 pose landmarks, 468 face landmarks, and 21 reference landmarks in each hand."},
    ],
    "cifras": [
        {"seccion": "Data Records",
         "cita": "The dataset[14] has 4000 videos (.avi format) distributed in RGB, depth, and IR."},
        {"seccion": "Data Records (Videos)",
         "cita": "The first four digits symbolize the class number (Sign N presented in the Table 2), the "
                 "second four digits correspond to the number of the volunteer, and the last four digits "
                 "represent the number of the repetition of the sign made."},
        {"seccion": "Data Records (IMU)",
         "cita": "The IMU folder contains information on 250 inertial measurements associated with the 50 "
                 "signs considered."},
    ],
    "lo_que_el_articulo_no_dice": [
        "Qué versión de MediaPipe se usó.",
        "Qué se hace con los fotogramas en los que MediaPipe no detecta (descartar o rellenar); tampoco por qué "
        "los CSV publicados tienen menos filas que fotogramas el video (N-02): la hipótesis de que fueran "
        "fotogramas sin detección quedó contradicha por la re-extracción propia (N-09) y la causa es desconocida.",
        "Sumas de comprobación de los archivos del depósito (las MD5 están solo en la API de figshare, por ZIP).",
        "Cuál de los cinco voluntarios (0000-0004) es sordo u oyente, nativo o no nativo; tampoco sexo, edad "
        "individual ni lateralidad.",
        "Ni el número de repeticiones por seña y señante ni el de secuencias por modalidad: 4 y 1.000 son "
        "derivación aritmética (anteproyecto §1.3.4), que aquí se verifica contando archivos.",
    ],
}

# Condiciones de registro DECLARADAS por el artículo (derivadas de las citas anteriores), para
# contrastarlas con las MEDIDAS en las cabeceras AVI. Nada de esto se verifica visualmente aquí.
CAMARAS_DECLARADAS = {
    "COLOR_FACE": {"dispositivo": "Nikon D5600 DSLR, objetivo AF-P DX NIKKOR 55 mm, f/3.5", "fps": 50.0,
                   "resolucion": "640x424", "encuadre": "rostro del señante; cámara a la altura de los ojos"},
    "COLOR_BODY": {"dispositivo": "Microsoft Kinect V2 (RGB)", "fps": 24.0, "resolucion": "1920x1080",
                   "encuadre": "cuerpo desde las rodillas hasta la máxima extensión vertical de los brazos; "
                               "Kinect 5 cm por debajo de la DSLR"},
    "GRAY_DEPTH": {"dispositivo": "Microsoft Kinect V2 (profundidad)", "fps": 30.0, "resolucion": "512x424",
                   "encuadre": "el del Kinect"},
    "GRAY_IR": {"dispositivo": "Microsoft Kinect V2 (infrarrojo)", "fps": 30.0, "resolucion": "512x424",
                "encuadre": "el del Kinect"},
}
ESCENA_DECLARADA = {
    "distancia_camaras_senante_cm": 200,
    "distancia_senante_fondo_cm": 10,
    "fondo": "cortina azul de 3 m × 3 m",
    "iluminacion": "blanca y homogénea",
    "montaje": "DSLR y Kinect acoplados en una plataforma impresa en 3D sobre trípode con altura e inclinación ajustables",
    "fuente": "citas «escena» de ARTICULO_CITAS['registro'] (Methods, Video data acquisition)",
}

# Señantes tal como los publica el artículo. Los dos criterios NO se concilian (actividad 1.b).
SENANTES_ARTICULO = {
    "criterio_methods": "2 voluntarios sordos y 3 oyentes («Two of them are deaf and three of them are not hearing impaired»)",
    "criterio_data_records": "3 señantes nativos y 2 no nativos («5 volunteers: 3 native speakers and two non-native speakers»)",
    "otros_datos_publicados": [
        "Reclutados a través de la Asociación de Intérpretes de Lengua de Señas Colombiana-Español (Methods).",
        "Todos fluidos en LSC y mayores de 18 años en el momento de la captura (Methods).",
        "Los intérpretes que validaron las señas «are non-native speakers and are some of the participants "
        "that can be seen in the stored data» (Technical Validation).",
        "No publica sexo, edad individual, lateralidad ni qué voluntario (0000-0004) pertenece a cada grupo.",
    ],
    "advertencia": (
        "Los dos criterios no se corresponden entre sí (2 + 3 frente a 3 + 2) y el artículo no publica "
        "cuál de los cinco voluntarios (0000-0004) pertenece a cada grupo. Se registran ambos SIN "
        "conciliarlos, como exige la actividad 1.b del anteproyecto."
    ),
    "fuente": ARTICULO_FUENTE + " Citas completas en «citas_del_articulo».",
}

# Valores verificados a mano el 13-sep-2026 (DECISIONES_TESIS.txt, N-02) que esta corrida debe reproducir
ESPERADO_N02 = {
    "video_mas_fotogramas_que_filas": 749,
    "video_igual_filas": 251,
    "video_menos_fotogramas_que_filas": 0,
    "diferencia_maxima": 51,
    "diferencia_mediana": 5,
    "body_manos_longitud_distinta": 58,   # ESTADO_PROYECTO §3 RE1 («58 muestras truncadas en silencio»)
    "muestras_detalle": {
        "0000_0000_0000": {"fotogramas": 59, "filas_body": 59, "filas_todo_cero_manos": 0},
        "0025_0002_0001": {"fotogramas": 61, "filas_body": 57, "filas_todo_cero_manos": 0},
        "0049_0004_0003": {"fotogramas": 78, "filas_body": 74, "filas_todo_cero_manos": 0},
    },
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def nombres_esperados() -> list[str]:
    """Rejilla SSSS_VVVV_RRRR que el corpus declara publicar (1000 nombres)."""
    return [f"{s:04d}_{v:04d}_{r:04d}"
            for s in range(N_SENAS) for v in range(N_SENANTES) for r in range(N_REPETICIONES)]


def parse_nombre(nombre: str) -> tuple[int, int, int] | None:
    """SSSS_VVVV_RRRR → (seña, señante, repetición); None si no cumple el patrón."""
    m = re.fullmatch(r"(\d{4})_(\d{4})_(\d{4})", nombre)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def estad(valores) -> dict | None:
    """min / max / media / mediana / DE / p90 de una lista numérica (None si está vacía)."""
    v = np.asarray([x for x in valores if x is not None], dtype=np.float64)
    if v.size == 0:
        return None
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "max": float(v.max()),
        "media": round(float(v.mean()), 3),
        "mediana": float(np.median(v)),
        "de": round(float(v.std()), 3),
        "p90": float(np.percentile(v, 90)),
    }


def iso_mtime(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def normalizar_glosa(s: str) -> str:
    """Minúsculas, sin tildes, sin espacios repetidos: para comparar glosas de fuentes distintas."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


# ---------------------------------------------------------------------------
# Lectura de cabeceras AVI (struct, sin decodificar)
# ---------------------------------------------------------------------------

def leer_cabecera_avi(path: Path) -> dict:
    """Lee los chunks RIFF/avih/strh/strf de un AVI y devuelve fotogramas, fps, resolución y códec.

    Fuente de cada campo: 'avih' (MainAVIHeader: dwMicroSecPerFrame, dwTotalFrames, dwWidth,
    dwHeight), 'strh' del primer flujo 'vids' (dwScale, dwRate, dwLength) y 'strf'
    (BITMAPINFOHEADER: biWidth, biHeight, biBitCount, biCompression). Si el archivo trae índice
    OpenDML ('dmlh'), se registra también su dwTotalFrames.
    """
    info = {"legible": False, "bytes": None, "error": None}
    try:
        info["bytes"] = path.stat().st_size
        with open(path, "rb") as f:
            cab = f.read(BYTES_CABECERA_AVI)
    except OSError as e:
        info["error"] = f"no se pudo abrir: {e}"
        return info

    if len(cab) < 12 or cab[0:4] != b"RIFF" or cab[8:12] != b"AVI ":
        info["error"] = "no es RIFF/AVI"
        return info

    riff_size = struct.unpack_from("<I", cab, 4)[0]
    info["riff_size_mas_8"] = riff_size + 8
    # Un AVI clásico (< 1 GiB) cumple riff_size + 8 == tamaño de archivo; si no, o está truncado o es OpenDML
    info["riff_coherente_con_tamano"] = (riff_size + 8 == info["bytes"])

    i = cab.find(b"avih")
    if i < 0 or i + 8 + 40 > len(cab):
        info["error"] = "sin chunk avih"
        return info
    campos = struct.unpack_from("<10I", cab, i + 8)
    us_por_fotograma, _maxbps, _pad, _flags, total_frames, _init, streams, _buf, w, h = campos
    info.update({
        "fotogramas": int(total_frames),               # avih.dwTotalFrames
        "us_por_fotograma": int(us_por_fotograma),     # avih.dwMicroSecPerFrame
        "fps_avih": round(1e6 / us_por_fotograma, 4) if us_por_fotograma else None,
        "flujos": int(streams),
        "ancho": int(w),                               # avih.dwWidth
        "alto": int(h),                                # avih.dwHeight
    })

    j = cab.find(b"strh")
    if j >= 0 and j + 8 + 36 <= len(cab):
        fcc_type = cab[j + 8:j + 12]
        fcc_handler = cab[j + 12:j + 16]
        _flags, _prio, _lang, _init, scale, rate, _start, length = struct.unpack_from("<IHHIIIII", cab, j + 16)
        info.update({
            "strh_tipo": fcc_type.decode("latin-1"),
            "codec_strh": fcc_handler.decode("latin-1").strip("\x00 "),
            "strh_scale": int(scale),
            "strh_rate": int(rate),
            "fotogramas_strh": int(length),            # strh.dwLength
            "fps": round(rate / scale, 4) if scale else None,   # strh.dwRate / strh.dwScale
        })
    else:
        info["fps"] = info["fps_avih"]

    k = cab.find(b"strf")
    if k >= 0 and k + 8 + 20 <= len(cab):
        _bsz, bw, bh, _planes, bitcount = struct.unpack_from("<IiiHH", cab, k + 8)
        comp = cab[k + 8 + 16:k + 8 + 20]
        info.update({
            "strf_ancho": int(bw),
            "strf_alto": int(abs(bh)),
            "bits_por_pixel": int(bitcount),
            "codec_strf": comp.decode("latin-1").strip("\x00 "),
        })

    m = cab.find(b"dmlh")
    if m >= 0 and m + 12 <= len(cab):
        info["fotogramas_dmlh"] = int(struct.unpack_from("<I", cab, m + 8)[0])

    info["legible"] = True
    return info


def hash_parcial(path: Path, n_bytes: int) -> str:
    """SHA-1 de los primeros n_bytes de un archivo (heurística de duplicados sin leer 6 GB)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Lectura de CSV de landmarks (numpy, en un solo paso)
# ---------------------------------------------------------------------------

def leer_csv_landmarks(path: Path, n_puntos_esperado: int, con_hash: bool = True) -> dict:
    """Cuenta filas y columnas, NaN, filas todo-cero y comprueba cabecera e índice de un CSV.

    El formato publicado es: primera columna índice sin nombre, luego landmark_i_x/y/z.
    """
    info = {"legible": False, "bytes": None, "error": None}
    try:
        info["bytes"] = path.stat().st_size
        with open(path, "rb") as f:
            datos = f.read()
    except OSError as e:
        info["error"] = f"no se pudo abrir: {e}"
        return info

    if con_hash:
        info["sha1"] = hashlib.sha1(datos).hexdigest()
    info["nan_en_texto"] = datos.lower().count(b"nan")   # ocurrencias literales de 'nan' en el archivo
    info["campos_vacios_en_texto"] = datos.count(b",,") + datos.count(b",\n") + datos.count(b",\r\n")

    texto = datos.decode("utf-8", errors="replace")
    primera_linea, _, _resto = texto.partition("\n")
    cabecera = primera_linea.rstrip("\r").split(",")
    esperada = [""] + [f"landmark_{i}_{c}" for i in range(n_puntos_esperado) for c in ("x", "y", "z")]
    info["columnas_cabecera"] = len(cabecera) - 1          # sin la columna índice
    info["cabecera_ok"] = (cabecera == esperada)

    parse_relajado = False
    try:
        arr = np.loadtxt(io.StringIO(texto), delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    except ValueError:
        # Campos vacíos u otro texto no numérico: genfromtxt los convierte en NaN
        arr = np.genfromtxt(io.StringIO(texto), delimiter=",", skip_header=1, dtype=np.float64,
                            filling_values=np.nan)
        arr = np.atleast_2d(arr)
        parse_relajado = True
    info["parse_relajado"] = parse_relajado

    if arr.size == 0:
        info.update({"filas": 0, "columnas": info["columnas_cabecera"], "nan": 0, "filas_todo_cero": 0,
                     "indice_consecutivo": True, "indice_max": None, "legible": True})
        return info

    indice = arr[:, 0]
    valores = arr[:, 1:]
    info.update({
        "filas": int(valores.shape[0]),
        "columnas": int(valores.shape[1]),
        "columnas_ok": (valores.shape[1] == 3 * n_puntos_esperado),
        "nan": int(np.isnan(valores).sum()),
        "filas_con_nan": int(np.isnan(valores).any(axis=1).sum()),
        "filas_todo_cero": int((valores == 0).all(axis=1).sum()),
        "indice_consecutivo": bool(np.array_equal(indice, np.arange(indice.size))),
        "indice_max": int(np.nanmax(indice)) if indice.size else None,
        "legible": True,
    })
    return info


# ---------------------------------------------------------------------------
# Material publicado junto al corpus: Timestamps.xlsx y Volunteer_k.txt (IMU/INFO)
# ---------------------------------------------------------------------------

def leer_timestamps_xlsx(path: Path) -> dict | None:
    """Lee IMU/INFO/Timestamps.xlsx con zipfile + ElementTree (sin openpyxl).

    Devuelve {hoja: [{"N": ..., "Sign": ..., "Time Start (s)": ..., "Time End (s)": ...}, ...]}.
    """
    if not path.exists():
        return None
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    try:
        z = zipfile.ZipFile(path)
        ss = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            ss = ["".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")) for si in root.findall("m:si", ns)]
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        hojas = [(s.get("name"), s.get("sheetId")) for s in wb.find("m:sheets", ns)]
        salida = {}
        for nombre, sid in hojas:
            ruta = f"xl/worksheets/sheet{sid}.xml"
            if ruta not in z.namelist():
                continue
            root = ET.fromstring(z.read(ruta))
            filas = []
            for fila in root.findall(".//m:row", ns):
                vals = []
                for c in fila.findall("m:c", ns):
                    v = c.find("m:v", ns)
                    if v is None:
                        vals.append("")
                    elif c.get("t") == "s":
                        vals.append(ss[int(v.text)])
                    else:
                        vals.append(v.text)
                filas.append(vals)
            if not filas:
                continue
            enc = filas[0]
            regs = []
            for vals in filas[1:]:
                if len(vals) < 2 or vals[0] == "":
                    continue
                reg = {enc[i] if i < len(enc) else f"col{i}": vals[i] for i in range(len(vals))}
                regs.append(reg)
            salida[re.sub(r"\s+", " ", nombre).strip()] = regs
        return salida
    except (zipfile.BadZipFile, ET.ParseError, KeyError, ValueError) as e:
        return {"error": f"no se pudo leer: {e}"}


def leer_volunteer_txt(dir_info: Path) -> dict:
    """Lee IMU/INFO/Volunteer_k.txt (antropometría publicada por los autores)."""
    salida = {}
    for p in sorted(dir_info.glob("Volunteer_*.txt")):
        campos = {}
        for linea in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in linea:
                k, v = linea.split(":", 1)
                campos[k.strip()] = v.strip()
        salida[p.stem] = campos
    return salida


def buscar_sumas_de_comprobacion(raw: Path) -> list[str]:
    """Busca archivos de sumas de comprobación publicados junto al corpus."""
    patrones = ("*.md5", "*.sha1", "*.sha256", "*.sha512", "*checksum*", "*CHECKSUM*", "MANIFEST*", "*.sfv")
    encontrados = set()
    for pat in patrones:
        for p in raw.rglob(pat):
            encontrados.add(str(p.relative_to(raw)))
    return sorted(encontrados)


def consultar_figshare(timeout: float = 20.0) -> dict:
    """Consulta en vivo la API pública de figshare (requiere internet). Mismo esquema que FIGSHARE_INSTANTANEA."""
    try:
        with urllib.request.urlopen(FIGSHARE_INSTANTANEA["api"], timeout=timeout) as r:
            d = json.load(r)
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"error": f"no se pudo consultar {FIGSHARE_INSTANTANEA['api']}: {e}"}
    return {
        "api": FIGSHARE_INSTANTANEA["api"],
        "fecha_consulta": dt.date.today().isoformat(),
        "titulo": d.get("title"),
        "doi": d.get("doi"),
        "version": d.get("version"),
        "published_date": d.get("published_date"),
        "modified_date": d.get("modified_date"),
        "licencia": (d.get("license") or {}).get("name"),
        "autores": [a.get("full_name") for a in d.get("authors", [])],
        "archivos": [{"name": f.get("name"), "size": f.get("size"), "computed_md5": f.get("computed_md5"),
                      "download_url": f.get("download_url")} for f in d.get("files", [])],
    }


def _hash_completo(path: Path, algoritmo: str) -> str:
    h = hashlib.new(algoritmo)
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def md5_archivo(path: Path) -> str:
    """MD5 del archivo completo (para cotejar los ZIP con figshare)."""
    return _hash_completo(path, "md5")


def sha1_archivo(path: Path) -> str:
    """SHA-1 del archivo completo (para confirmar duplicados de video)."""
    return _hash_completo(path, "sha1")


def _archivos_por_nombre(base: Path, nombre: str, profundidad: int) -> list[Path]:
    """Busca `nombre` bajo `base` hasta `profundidad` niveles (sin recorrer los 68 GB de data/raw entero)."""
    hallados = []
    patron = nombre
    for _ in range(profundidad + 1):
        hallados += [p for p in base.glob(patron) if p.is_file()]
        patron = "*/" + patron
    return hallados


def buscar_zips_originales(carpetas: list[Path], verificar_md5: bool, profundidad: int = 3) -> dict:
    """Busca los ZIP del depósito (IMU.zip, LANDMARKS.zip, VIDEOS.zip) y los coteja con figshare.

    Por tamaño siempre; por MD5 solo con verificar_md5 (leer ~10 GB tarda minutos).
    """
    res = {}
    for esperado in FIGSHARE_INSTANTANEA["archivos"]:
        nombre = esperado["name"]
        hallados = []
        for base in carpetas:
            if base.exists():
                hallados += _archivos_por_nombre(base, nombre, profundidad)
        entrada = {"encontrado": bool(hallados), "rutas": [str(p) for p in hallados],
                   "md5_figshare": esperado["computed_md5"], "bytes_figshare": esperado["size"]}
        if hallados:
            p = hallados[0]
            entrada["bytes"] = p.stat().st_size
            entrada["bytes_coinciden_con_figshare"] = (entrada["bytes"] == esperado["size"])
            if verificar_md5:
                entrada["md5"] = md5_archivo(p)
                entrada["md5_coincide_con_figshare"] = (entrada["md5"] == esperado["computed_md5"])
            else:
                entrada["md5"] = "no calculado (use --verificar-md5)"
        res[nombre] = entrada
    return res


def confirmar_duplicados_video(candidatos: dict[str, list[list[str]]], raw: Path) -> tuple[dict, dict]:
    """Recalcula con SHA-1 del archivo COMPLETO los grupos candidatos (tamaño + primeros bytes iguales).

    Devuelve (confirmados, refutados) por cámara: confirmados = grupos con SHA-1 completo idéntico;
    refutados = candidatos en los que ningún par resultó idéntico.
    """
    confirmados, refutados = {}, {}
    for cam, grupos in candidatos.items():
        confirmados[cam], refutados[cam] = [], []
        for grupo in grupos:
            por_hash = defaultdict(list)
            for nombre in grupo:
                por_hash[sha1_archivo(raw / CAMARAS[cam] / f"{nombre}.avi")].append(nombre)
            iguales = [sorted(m) for m in por_hash.values() if len(m) > 1]
            if iguales:
                confirmados[cam] += [{"muestras": m, "sha1": h} for h, m in por_hash.items() if len(m) > 1]
            else:
                refutados[cam].append(sorted(grupo))
    return confirmados, refutados


def cotejar_particiones(grupos: list[list[str]], dir_part: Path) -> dict:
    """¿Algún grupo de duplicados queda repartido entre conjuntos distintos de una partición (D-02)?

    Un duplicado con un miembro en train y otro en val o test es una fuga: el modelo «ve» en prueba
    una secuencia idéntica a una de entrenamiento.
    """
    res = {}
    for esquema in ("aleatoria_80_10_10", "loso_5_pliegues"):
        p = dir_part / f"{esquema}.json"
        if not p.exists():
            res[esquema] = {"archivo": str(p), "encontrado": False}
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        bloques = []   # (etiqueta, {conjunto: set(nombres)})
        if isinstance(d.get("conjuntos"), dict):
            bloques.append(("única", {k: set(v) for k, v in d["conjuntos"].items()}))
        for pl in d.get("pliegues", []):
            if isinstance(pl.get("conjuntos"), dict):
                bloques.append((f"pliegue {pl.get('pliegue')} (test = señante {pl.get('senante_test')})",
                                {k: set(v) for k, v in pl["conjuntos"].items()}))
        hallazgos = []
        for etiqueta, conj in bloques:
            for grupo in grupos:
                donde = {n: next((c for c, s in conj.items() if n in s), None) for n in grupo}
                if len(set(donde.values())) > 1:
                    hallazgos.append({"bloque": etiqueta, "muestras": donde})
        res[esquema] = {"archivo": str(p), "encontrado": True, "bloques_revisados": len(bloques),
                        "grupos_revisados": len(grupos), "grupos_repartidos_entre_conjuntos": hallazgos}
    return res


# ---------------------------------------------------------------------------
# Verificación por muestra
# ---------------------------------------------------------------------------

def verificar_muestra(nombre: str, raw: Path, con_hash: bool) -> dict:
    ids = parse_nombre(nombre)
    reg = {
        "muestra": nombre,
        "sena_id": ids[0] if ids else None,
        "senante_id": ids[1] if ids else None,
        "repeticion": ids[2] if ids else None,
        "archivos": {},
        "csv": {},
        "video": {},
        "diferencias": {},
        "anomalias": [],
    }
    anom = reg["anomalias"]

    # --- CSV de landmarks ---
    for clave, (sub, n_puntos) in MODALIDADES_CSV.items():
        p = raw / sub / f"{nombre}.csv"
        existe = p.exists()
        reg["archivos"][clave] = {"ruta": str(p.relative_to(raw)), "existe": existe,
                                  "bytes": p.stat().st_size if existe else None,
                                  "mtime": iso_mtime(p) if existe else None}
        if not existe:
            anom.append(f"ausente:{clave}")
            continue
        info = leer_csv_landmarks(p, n_puntos, con_hash)
        reg["csv"][clave] = info
        if not info["legible"]:
            anom.append(f"ilegible:{clave}")
            continue
        if not info.get("cabecera_ok", False):
            anom.append(f"cabecera_inesperada:{clave}")
        if not info.get("columnas_ok", False):
            anom.append(f"columnas_inesperadas:{clave}")
        if info.get("nan", 0) > 0:
            anom.append(f"nan:{clave}")
        if info.get("filas_todo_cero", 0) > 0:
            anom.append(f"filas_todo_cero:{clave}")
        if not info.get("indice_consecutivo", True):
            anom.append(f"indice_no_consecutivo:{clave}")
        if info.get("parse_relajado"):
            anom.append(f"parse_relajado:{clave}")
        if info.get("filas", 0) == 0:
            anom.append(f"csv_vacio:{clave}")

    # --- Videos ---
    for cam, sub in CAMARAS.items():
        p = raw / sub / f"{nombre}.avi"
        existe = p.exists()
        reg["archivos"][cam] = {"ruta": str(p.relative_to(raw)), "existe": existe,
                                "bytes": p.stat().st_size if existe else None,
                                "mtime": iso_mtime(p) if existe else None}
        if not existe:
            anom.append(f"ausente:{cam}")
            continue
        info = leer_cabecera_avi(p)
        if con_hash and info["legible"]:
            info["sha1_primeros_bytes"] = hash_parcial(p, BYTES_HASH_VIDEO)
        reg["video"][cam] = info
        if not info["legible"]:
            anom.append(f"ilegible:{cam}")
            continue
        if not info.get("riff_coherente_con_tamano", True):
            anom.append(f"riff_tamano_incoherente:{cam}")
        if info.get("fotogramas_strh") is not None and info["fotogramas_strh"] != info["fotogramas"]:
            anom.append(f"avih_strh_fotogramas_distintos:{cam}")
        if info.get("fotogramas", 0) == 0:
            anom.append(f"video_sin_fotogramas:{cam}")

    # --- Diferencias entre modalidades ---
    filas = {k: reg["csv"][k]["filas"] for k in MODALIDADES_CSV if k in reg["csv"] and reg["csv"][k]["legible"]}
    fot = {c: reg["video"][c]["fotogramas"] for c in CAMARAS if c in reg["video"] and reg["video"][c]["legible"]}
    d = reg["diferencias"]
    if all(k in filas for k in ("body", "lh", "rh")):
        d["body_menos_lh"] = filas["body"] - filas["lh"]
        d["body_menos_rh"] = filas["body"] - filas["rh"]
        d["lh_menos_rh"] = filas["lh"] - filas["rh"]
        trio = (filas["body"], filas["lh"], filas["rh"])
        d["max_menos_min_body_manos"] = max(trio) - min(trio)
        d["filas_tras_truncar_al_minimo"] = min(trio)   # lo que hizo preprocess_lsc50.py
        if d["max_menos_min_body_manos"] > 0:
            anom.append("body_manos_longitud_distinta")
            anom.append("truncada_por_preprocess" if d["max_menos_min_body_manos"] <= 2
                        else "descartada_por_preprocess")
    if "body" in filas and "COLOR_BODY" in fot:
        d["fotogramas_COLOR_BODY_menos_filas_body"] = fot["COLOR_BODY"] - filas["body"]
        if d["fotogramas_COLOR_BODY_menos_filas_body"] > 0:
            anom.append("video_mas_fotogramas_que_filas_body")
        elif d["fotogramas_COLOR_BODY_menos_filas_body"] < 0:
            anom.append("video_menos_fotogramas_que_filas_body")
    if "face" in filas and "COLOR_FACE" in fot:
        d["fotogramas_COLOR_FACE_menos_filas_face"] = fot["COLOR_FACE"] - filas["face"]
        if d["fotogramas_COLOR_FACE_menos_filas_face"] > 0:
            anom.append("video_mas_fotogramas_que_filas_face")
        elif d["fotogramas_COLOR_FACE_menos_filas_face"] < 0:
            anom.append("video_menos_fotogramas_que_filas_face")
    if "face" in filas and "body" in filas and filas["body"]:
        d["ratio_filas_face_body"] = round(filas["face"] / filas["body"], 4)
    if "COLOR_FACE" in fot and "COLOR_BODY" in fot and fot["COLOR_BODY"]:
        d["ratio_fotogramas_COLOR_FACE_COLOR_BODY"] = round(fot["COLOR_FACE"] / fot["COLOR_BODY"], 4)
    if "GRAY_DEPTH" in fot and "GRAY_IR" in fot:
        d["fotogramas_GRAY_DEPTH_menos_GRAY_IR"] = fot["GRAY_DEPTH"] - fot["GRAY_IR"]
    return reg


# ---------------------------------------------------------------------------
# Agregados
# ---------------------------------------------------------------------------

def moda(valores):
    c = Counter(v for v in valores if v is not None)
    return c.most_common(1)[0][0] if c else None


def agregar(registros: list[dict], esperados: list[str], encontrados_por_carpeta: dict,
            con_hash: bool, raw: Path) -> dict:
    """Segunda pasada: modas por cámara, duplicados (confirmados con hash completo) y recuentos de anomalías."""
    presentes = {r["muestra"] for r in registros}
    ausentes = sorted(set(esperados) - presentes)

    # Moda de fps / resolución / códec por cámara; muestras que se apartan de la moda → anomalía
    modas = {}
    for cam in CAMARAS:
        fps = [r["video"].get(cam, {}).get("fps") for r in registros]
        res = [(r["video"].get(cam, {}).get("ancho"), r["video"].get(cam, {}).get("alto"))
               for r in registros if r["video"].get(cam, {}).get("legible")]
        cod = [r["video"].get(cam, {}).get("codec_strf") for r in registros]
        modas[cam] = {"fps": moda(fps), "resolucion": moda(res), "codec": moda(cod),
                      "fps_valores": {str(k): v for k, v in Counter(f for f in fps if f is not None).items()},
                      "resolucion_valores": {f"{k[0]}x{k[1]}": v for k, v in Counter(res).items()},
                      "codec_valores": {str(k): v for k, v in Counter(c for c in cod if c is not None).items()}}
        for r in registros:
            v = r["video"].get(cam)
            if not v or not v.get("legible"):
                continue
            if v.get("fps") != modas[cam]["fps"]:
                r["anomalias"].append(f"fps_distinto_de_la_moda:{cam}")
            if (v.get("ancho"), v.get("alto")) != modas[cam]["resolucion"]:
                r["anomalias"].append(f"resolucion_distinta_de_la_moda:{cam}")

    # Duplicados: CSV por SHA-1 completo; videos por (tamaño, SHA-1 de los primeros bytes) como criba y
    # SHA-1 del archivo completo sobre los candidatos para confirmarlos o refutarlos
    duplicados = {}
    duplicados_video = {"candidatos": {}, "confirmados": {}, "refutados": {}}
    if con_hash:
        for clave in MODALIDADES_CSV:
            grupos = defaultdict(list)
            for r in registros:
                h = r["csv"].get(clave, {}).get("sha1")
                if h:
                    grupos[h].append(r["muestra"])
            dup = [sorted(v) for v in grupos.values() if len(v) > 1]
            duplicados[clave] = dup
            for grupo in dup:
                for r in registros:
                    if r["muestra"] in grupo:
                        r["anomalias"].append(f"duplicado:{clave}")
        candidatos = {}
        for cam in CAMARAS:
            grupos = defaultdict(list)
            for r in registros:
                v = r["video"].get(cam, {})
                if v.get("sha1_primeros_bytes"):
                    grupos[(v["bytes"], v["sha1_primeros_bytes"])].append(r["muestra"])
            candidatos[cam] = [sorted(v) for v in grupos.values() if len(v) > 1]
        confirmados, refutados = confirmar_duplicados_video(candidatos, raw)
        duplicados_video = {"candidatos": candidatos, "confirmados": confirmados, "refutados": refutados}
        for cam in CAMARAS:
            duplicados[cam] = [g["muestras"] for g in confirmados[cam]]
            for grupo in duplicados[cam]:
                for r in registros:
                    if r["muestra"] in grupo:
                        r["anomalias"].append(f"duplicado:{cam}")
    # Grupos de muestras con algún archivo idéntico, y en qué modalidades
    pares = defaultdict(list)
    for clave, grupos in duplicados.items():
        for g in grupos:
            pares["|".join(g)].append(clave)
    grupos_duplicados = [{"muestras": k.split("|"), "modalidades": v} for k, v in sorted(pares.items())]

    # Recuento de anomalías (cada muestra cuenta una vez por etiqueta)
    recuento = Counter()
    for r in registros:
        r["anomalias"] = sorted(set(r["anomalias"]))
        for a in r["anomalias"]:
            recuento[a] += 1

    # Diferencias fotogramas − filas (N-02)
    dif_body = [r["diferencias"].get("fotogramas_COLOR_BODY_menos_filas_body") for r in registros]
    dif_body = [x for x in dif_body if x is not None]
    positivas = [x for x in dif_body if x > 0]
    dif_face = [r["diferencias"].get("fotogramas_COLOR_FACE_menos_filas_face") for r in registros]
    dif_face = [x for x in dif_face if x is not None]
    dif_bm = [r["diferencias"].get("max_menos_min_body_manos") for r in registros]
    dif_bm = [x for x in dif_bm if x is not None]

    ilegibles = sorted({r["muestra"] for r in registros if any(a.startswith("ilegible:") for a in r["anomalias"])})
    archivos_ausentes = sorted({f"{r['muestra']}:{a.split(':', 1)[1]}" for r in registros
                                for a in r["anomalias"] if a.startswith("ausente:")})
    nombres_inesperados = sorted(set(presentes) - set(esperados))

    return {
        "muestras_esperadas": len(esperados),
        "muestras_verificadas": len(registros),
        "archivos_por_carpeta": encontrados_por_carpeta,
        "muestras_ausentes_de_la_rejilla": ausentes,
        "nombres_fuera_de_la_rejilla": nombres_inesperados,
        "archivos_ausentes": archivos_ausentes,
        "muestras_con_archivo_ilegible": ilegibles,
        "duplicados": duplicados if con_hash else "no calculado (--sin-hash)",
        "duplicados_video_detalle": duplicados_video if con_hash else "no calculado (--sin-hash)",
        "grupos_duplicados": grupos_duplicados,
        "recuento_anomalias": dict(sorted(recuento.items())),
        "modas_por_camara": {cam: {"fps": m["fps"], "resolucion": (f"{m['resolucion'][0]}x{m['resolucion'][1]}"
                                                                    if m["resolucion"] else None),
                                   "codec": m["codec"], "fps_valores": m["fps_valores"],
                                   "resolucion_valores": m["resolucion_valores"], "codec_valores": m["codec_valores"]}
                             for cam, m in modas.items()},
        "fotogramas_COLOR_BODY_frente_a_filas_body": {
            "fuente": "avih.dwTotalFrames de VIDEOS/COLOR_BODY/*.avi frente a filas de LANDMARKS/BODY_LANDMARKS/*.csv",
            "video_mas_fotogramas_que_filas": sum(1 for x in dif_body if x > 0),
            "video_igual_filas": sum(1 for x in dif_body if x == 0),
            "video_menos_fotogramas_que_filas": sum(1 for x in dif_body if x < 0),
            "diferencia_sobre_las_positivas": estad(positivas),
            "diferencia_sobre_todas": estad(dif_body),
            "histograma_diferencia": {str(k): v for k, v in sorted(Counter(dif_body).items())},
        },
        "fotogramas_COLOR_FACE_frente_a_filas_face": {
            "fuente": "avih.dwTotalFrames de VIDEOS/COLOR_FACE/*.avi frente a filas de LANDMARKS/FACE_LANDMARKS/*.csv",
            "video_mas_fotogramas_que_filas": sum(1 for x in dif_face if x > 0),
            "video_igual_filas": sum(1 for x in dif_face if x == 0),
            "video_menos_fotogramas_que_filas": sum(1 for x in dif_face if x < 0),
            "diferencia_sobre_todas": estad(dif_face),
        },
        "body_frente_a_manos": {
            "fuente": "filas de BODY_LANDMARKS frente a LEFT_HAND_LANDMARKS y RIGHT_HAND_LANDMARKS",
            "muestras_con_longitud_distinta": sum(1 for x in dif_bm if x > 0),
            "muestras_con_diferencia_1_o_2_truncadas_por_preprocess": sum(1 for x in dif_bm if 0 < x <= 2),
            "muestras_con_diferencia_mayor_que_2_descartadas_por_preprocess": sum(1 for x in dif_bm if x > 2),
            "histograma_max_menos_min": {str(k): v for k, v in sorted(Counter(dif_bm).items())},
        },
        "csv_indice_consecutivo_en_todas": all(r["csv"][k]["indice_consecutivo"] for r in registros
                                              for k in r["csv"] if r["csv"][k].get("legible")),
        "totales_por_modalidad": {
            clave: {
                "filas": int(sum(r["csv"].get(clave, {}).get("filas", 0) for r in registros)),
                "nan": int(sum(r["csv"].get(clave, {}).get("nan", 0) for r in registros)),
                "filas_todo_cero": int(sum(r["csv"].get(clave, {}).get("filas_todo_cero", 0) for r in registros)),
                "bytes": int(sum(r["archivos"].get(clave, {}).get("bytes") or 0 for r in registros)),
            } for clave in MODALIDADES_CSV
        },
        "bytes_por_camara": {cam: int(sum(r["archivos"].get(cam, {}).get("bytes") or 0 for r in registros))
                             for cam in CAMARAS},
    }


def comprobar_n02(registros: list[dict], agregados: dict, completo: bool) -> dict:
    """Compara con los valores verificados a mano el 13-sep-2026 (N-02) y con las 58 truncadas."""
    if not completo:
        return {"aplicable": False, "motivo": "corrida parcial (--limite): las comprobaciones solo valen con las 1000 muestras"}
    fb = agregados["fotogramas_COLOR_BODY_frente_a_filas_body"]
    pos = fb["diferencia_sobre_las_positivas"] or {}
    res = {
        "aplicable": True,
        "video_mas_fotogramas_que_filas": (fb["video_mas_fotogramas_que_filas"], ESPERADO_N02["video_mas_fotogramas_que_filas"]),
        "video_igual_filas": (fb["video_igual_filas"], ESPERADO_N02["video_igual_filas"]),
        "video_menos_fotogramas_que_filas": (fb["video_menos_fotogramas_que_filas"], ESPERADO_N02["video_menos_fotogramas_que_filas"]),
        "diferencia_maxima": (int(pos.get("max", -1)), ESPERADO_N02["diferencia_maxima"]),
        "diferencia_mediana": (pos.get("mediana"), ESPERADO_N02["diferencia_mediana"]),
        "body_manos_longitud_distinta": (agregados["body_frente_a_manos"]["muestras_con_longitud_distinta"],
                                         ESPERADO_N02["body_manos_longitud_distinta"]),
    }
    por_muestra = {}
    idx = {r["muestra"]: r for r in registros}
    for nombre, esp in ESPERADO_N02["muestras_detalle"].items():
        r = idx.get(nombre)
        if r is None:
            por_muestra[nombre] = {"encontrada": False}
            continue
        obt = {
            "fotogramas": r["video"].get("COLOR_BODY", {}).get("fotogramas"),
            "filas_body": r["csv"].get("body", {}).get("filas"),
            "filas_todo_cero_manos": (r["csv"].get("lh", {}).get("filas_todo_cero", 0)
                                      + r["csv"].get("rh", {}).get("filas_todo_cero", 0)),
        }
        por_muestra[nombre] = {"encontrada": True, "obtenido": obt, "esperado": esp, "ok": obt == esp}
    res["muestras_detalle"] = por_muestra
    res["todo_ok"] = all(v[0] == v[1] for k, v in res.items() if isinstance(v, tuple)) \
        and all(m.get("ok") for m in por_muestra.values())
    return res


# ---------------------------------------------------------------------------
# Caracterización (1.b)
# ---------------------------------------------------------------------------

def cargar_vocabulario() -> dict[int, str]:
    sys.path.insert(0, str(SRC_DIR))
    from grammar_mapper import SIGN_VOCABULARY  # noqa: E402  (tabla de 50 señas, sin efectos al importar)
    return dict(SIGN_VOCABULARY)


def caracterizar(registros: list[dict], agregados: dict, metadata: dict | None, raw: Path,
                 completo: bool, dir_part: Path = PARTICIONES_DIR) -> dict:
    vocab = cargar_vocabulario()
    fuente_nombres = "nombres de archivo SSSS_VVVV_RRRR en LANDMARKS/BODY_LANDMARKS (cotejados con las otras 7 carpetas)"

    # --- Vocabulario publicado en Timestamps.xlsx (IMU/INFO) frente a SIGN_VOCABULARY ---
    ts = leer_timestamps_xlsx(raw / "IMU" / "INFO" / "Timestamps.xlsx")
    vocab_publicado = {}
    discrepancias = []
    hojas = []
    if ts and "error" not in ts:
        hojas = list(ts.keys())
        for hoja, regs in ts.items():
            for reg in regs:
                try:
                    n = int(reg.get("N", ""))
                except ValueError:
                    continue
                vocab_publicado.setdefault(n, set()).add(reg.get("Sign", ""))
        for sid, nombre in vocab.items():
            pub = vocab_publicado.get(sid, set())
            if not pub:
                discrepancias.append({"id": sid, "codigo": nombre, "publicado": None, "motivo": "ID ausente en Timestamps.xlsx"})
                continue
            if len(pub) > 1:
                discrepancias.append({"id": sid, "codigo": nombre, "publicado": sorted(pub),
                                      "motivo": "nombres distintos entre hojas"})
            for p in pub:
                if normalizar_glosa(p) != normalizar_glosa(nombre):
                    discrepancias.append({"id": sid, "codigo": nombre, "publicado": p,
                                          "motivo": "difiere tras normalizar (minúsculas, sin tildes, espacios)"})
    n_filas_timestamps = (sum(len(regs) for regs in ts.values()) if ts and "error" not in ts else None)
    vocabulario = [{
        "id": sid,
        "glosa_codigo": nombre,
        "glosa_publicada_Timestamps_xlsx": sorted(vocab_publicado.get(sid, [])) or None,
        "condicion_dinamica_estatica": "PENDIENTE (validación lingüística 1.d)",
    } for sid, nombre in sorted(vocab.items())]

    # --- Distribuciones ---
    por_clase = Counter(r["sena_id"] for r in registros)
    por_senante = Counter(r["senante_id"] for r in registros)
    por_clase_senante = Counter((r["sena_id"], r["senante_id"]) for r in registros)
    reps = Counter(r["repeticion"] for r in registros)
    tabla_clase_senante = {f"{s:04d}": {f"{v:04d}": por_clase_senante.get((s, v), 0) for v in range(N_SENANTES)}
                           for s in range(N_SENAS)}

    # --- Fotogramas por modalidad ---
    filas_mod = {k: [r["csv"][k]["filas"] for r in registros if k in r["csv"] and r["csv"][k]["legible"]]
                 for k in MODALIDADES_CSV}
    fot_cam = {c: [r["video"][c]["fotogramas"] for r in registros if c in r["video"] and r["video"][c]["legible"]]
               for c in CAMARAS}
    filas_trunc = [r["diferencias"]["filas_tras_truncar_al_minimo"] for r in registros
                   if "filas_tras_truncar_al_minimo" in r["diferencias"]]
    modas = agregados["modas_por_camara"]
    duracion_cam = {}
    for c in CAMARAS:
        fps = modas[c]["fps"]
        duracion_cam[c] = estad([f / fps for f in fot_cam[c]]) if fps else None

    # --- Rostro frente a cuerpo ---
    ratio_filas = [r["diferencias"]["ratio_filas_face_body"] for r in registros if "ratio_filas_face_body" in r["diferencias"]]
    ratio_fot = [r["diferencias"]["ratio_fotogramas_COLOR_FACE_COLOR_BODY"] for r in registros
                 if "ratio_fotogramas_COLOR_FACE_COLOR_BODY" in r["diferencias"]]

    # --- Cotejo con metadata.json (salida de preprocess_lsc50.py) ---
    cotejo_metadata = None
    if metadata:
        orden = metadata.get("sample_order", [])
        fs = metadata.get("frame_stats", {})
        est_trunc = estad(filas_trunc) or {}
        cotejo_metadata = {
            "fuente": "data/processed/metadata.json (sample_order, frame_stats) frente a esta corrida",
            "n_samples_metadata": metadata.get("n_samples"),
            "sample_order_igual_a_rejilla": (orden == sorted(r["muestra"] for r in registros)) if completo else "no aplica (corrida parcial)",
            "frame_stats_metadata": fs,
            "frame_stats_recalculados_min_body_lh_rh": {k: est_trunc.get(k) for k in ("min", "max", "media", "mediana", "de")},
            "coinciden": (completo and fs and est_trunc and int(fs.get("min", -1)) == int(est_trunc["min"])
                          and int(fs.get("max", -1)) == int(est_trunc["max"])
                          and abs(float(fs.get("mean", -1)) - est_trunc["media"]) < 1e-2),
        }

    # --- Señantes: artículo + antropometría publicada en IMU/INFO ---
    antropometria = leer_volunteer_txt(raw / "IMU" / "INFO")

    # --- Cámaras: declarado por el artículo frente a medido en las cabeceras ---
    cotejo_camaras = {}
    for c in CAMARAS:
        dec, med = CAMARAS_DECLARADAS[c], modas[c]
        cotejo_camaras[c] = {
            "fps_declarado": dec["fps"], "fps_medido_moda": med["fps"],
            "fps_coincide": (med["fps"] is not None and abs(med["fps"] - dec["fps"]) < 1e-6),
            "resolucion_declarada": dec["resolucion"], "resolucion_medida_moda": med["resolucion"],
            "resolucion_coincide": (med["resolucion"] == dec["resolucion"]),
            "todos_los_archivos_en_la_moda": (med["fps_valores"] == {str(dec["fps"]): len(fot_cam[c])}
                                              and med["resolucion_valores"] == {dec["resolucion"]: len(fot_cam[c])}),
        }

    # --- Duplicados exactos y su efecto sobre las particiones (D-02) ---
    grupos_dup = agregados.get("grupos_duplicados", [])
    modalidades_entrenamiento = {"body", "lh", "rh", "COLOR_BODY"}
    grupos_entrenamiento = [g["muestras"] for g in grupos_dup if set(g["modalidades"]) & modalidades_entrenamiento]
    cotejo_part = cotejar_particiones(grupos_entrenamiento, dir_part) if grupos_entrenamiento else {}
    n_unicas_body = len(registros) - sum(len(g) - 1 for g in grupos_entrenamiento)

    return {
        "identidad_corpus": IDENTIDAD_CORPUS,
        "cifras_anticipadas_por_el_anteproyecto": {
            "fuente": ("anteproyecto 05_justificacion.tex §1.3.4 (50 señas, 5 voluntarios y 4.000 videos publicados; 1.000 "
                       "secuencias por modalidad y 4 repeticiones por derivación aritmética) y 09_metodologia.tex §2.2.2; "
                       "lo declarado por el artículo, en «citas_del_articulo» → cifras"),
            "videos_avi": {"declarado_por_el_articulo": 4000,
                           "verificado": sum(agregados["archivos_por_carpeta"].get(c, 0) for c in CAMARAS),
                           "fuente": "conteo de *.avi en las cuatro carpetas de VIDEOS"},
            "medidas_inerciales": {"declarado_por_el_articulo": 250,
                                   "verificado_filas_en_Timestamps_xlsx": n_filas_timestamps,
                                   "fuente": ("filas con N y Sign en IMU/INFO/Timestamps.xlsx (5 hojas); IMU/RAW trae 5 libros "
                                              "con 7 hojas (una por sensor). Fuera del alcance del proyecto: se cuenta solo "
                                              "para el cotejo con lo declarado")},
            "senas": {"anticipado": N_SENAS, "verificado": len(por_clase), "fuente": fuente_nombres},
            "senantes": {"anticipado": N_SENANTES, "verificado": len(por_senante), "fuente": fuente_nombres},
            "repeticiones_por_sena_y_senante": {"anticipado": N_REPETICIONES,
                                                "verificado_min": min(por_clase_senante.values()) if por_clase_senante else None,
                                                "verificado_max": max(por_clase_senante.values()) if por_clase_senante else None,
                                                "fuente": fuente_nombres},
            "secuencias": {"anticipado": N_SENAS * N_SENANTES * N_REPETICIONES, "verificado": len(registros),
                           "fuente": fuente_nombres},
        },
        "vocabulario": {
            "fuente_codigo": "SIGN_VOCABULARY en src/grammar_mapper.py (verificado 50/50 contra la Tabla 2 del artículo, README 13-sep-2026)",
            "fuente_publicada": ("IMU/INFO/Timestamps.xlsx del depósito figshare, columna «Sign», una hoja por voluntario"
                                 + (f" (hojas: {', '.join(hojas)})" if hojas else " (NO ENCONTRADO)")),
            "nota_tabla_2_del_articulo": ("Cotejo del revisor (14-sep-2026) con el texto completo del artículo en Europe PMC "
                                          "(PMC11655960): la Tabla 2 escribe la seña 0029 como «JUCIOSO», la misma grafía de "
                                          "Timestamps.xlsx; «juicioso» es la grafía del código. Las dos fuentes publicadas "
                                          "coinciden entre sí: la discrepancia es del código frente a ambas y va a 1.d."),
            "nota_timestamps": ("Timestamps.xlsx trae además «Time Start (s)» y «Time End (s)» por seña y voluntario; su "
                                "semántica (referencia temporal frente a la grabación IMU) no está documentada en el "
                                "material descargado y no se usa aquí."),
            "n_senas": len(vocabulario),
            "discrepancias_codigo_vs_publicado": discrepancias,
            "senas": vocabulario,
        },
        "citas_del_articulo": {"fuente": ARTICULO_FUENTE, **ARTICULO_CITAS},
        "condicion_dinamica_estatica": {
            "estado": "PENDIENTE",
            "nota": ("El anteproyecto (§1.4) declara el corpus como de señas dinámicas; la condición por seña "
                     "se consigna tras la validación lingüística con la persona señante o intérprete (actividad 1.d). "
                     "No se infiere aquí de los datos."),
        },
        "distribucion": {
            "fuente": fuente_nombres,
            "muestras_por_clase": {f"{k:04d}": v for k, v in sorted(por_clase.items())},
            "muestras_por_senante": {f"{k:04d}": v for k, v in sorted(por_senante.items())},
            "muestras_por_repeticion": {f"{k:04d}": v for k, v in sorted(reps.items())},
            "muestras_por_clase_y_senante": tabla_clase_senante,
            "balanceado_por_clase": len(set(por_clase.values())) == 1,
            "balanceado_por_senante": len(set(por_senante.values())) == 1,
        },
        "fotogramas_por_modalidad": {
            "fuente": ("filas de datos de cada CSV (sin cabecera) y avih.dwTotalFrames de cada AVI; DE = desviación estándar "
                       "poblacional (numpy.std, ddof = 0), la misma convención que frame_stats de metadata.json"),
            "landmarks_publicados": {k: estad(v) for k, v in filas_mod.items()},
            "landmarks_body_lh_rh_truncados_al_minimo_como_preprocess": estad(filas_trunc),
            "videos": {c: estad(v) for c, v in fot_cam.items()},
            "duracion_s_por_camara_con_fps_medido": duracion_cam,
        },
        "camaras": {
            "fuente": "cabeceras AVI (avih: dwWidth/dwHeight/dwMicroSecPerFrame; strh: dwRate/dwScale; strf: biCompression), una por archivo, 4 × N archivos",
            "por_camara": modas,
            "declarado_por_el_articulo": {"fuente": "citas «registro» de ARTICULO_CITAS (Methods, Video data acquisition)",
                                          "por_camara": CAMARAS_DECLARADAS},
            "cotejo_declarado_vs_medido": cotejo_camaras,
        },
        "desalineamiento_rostro_cuerpo": {
            "fuente": "filas FACE_LANDMARKS / filas BODY_LANDMARKS por muestra; fotogramas COLOR_FACE / COLOR_BODY por muestra; fps medidos",
            "fps_COLOR_FACE_sobre_fps_COLOR_BODY": (round(modas["COLOR_FACE"]["fps"] / modas["COLOR_BODY"]["fps"], 4)
                                                    if modas["COLOR_FACE"]["fps"] and modas["COLOR_BODY"]["fps"] else None),
            "ratio_filas_face_body": estad(ratio_filas),
            "ratio_fotogramas_COLOR_FACE_COLOR_BODY": estad(ratio_fot),
            "face_frente_a_su_video": agregados["fotogramas_COLOR_FACE_frente_a_filas_face"],
            "consecuencia": ("Los landmarks faciales publicados proceden de otra cámara (COLOR_FACE) con otra tasa y otra "
                             "resolución: no están alineados fotograma a fotograma con body/manos. Es la causa verificada de "
                             "que el rostro quedara fuera del preprocesamiento de marzo (ESTADO_PROYECTO §3, RE1) y una razón "
                             "más para la extracción propia sobre COLOR_BODY (D-01)."),
        },
        "senantes": {
            "numero": len(por_senante),
            "fuente_numero": fuente_nombres,
            "condicion_segun_el_articulo": SENANTES_ARTICULO,
            "antropometria_publicada": {
                "fuente": "IMU/INFO/Volunteer_k.txt del depósito figshare (k = 1..5)",
                "advertencia": ("La correspondencia entre Volunteer_k (IMU) y el campo VVVV = 000(k−1) de los nombres de archivo "
                                "es la presumible, pero el material descargado no la declara: no se afirma."),
                "datos": antropometria,
            },
        },
        "condiciones_de_registro": {
            "medidas_aqui": {
                "resolucion_y_fps_por_camara": "ver «camaras» (cabeceras AVI) y «camaras» → cotejo_declarado_vs_medido",
                "duracion_de_los_clips": "ver «fotogramas_por_modalidad»",
            },
            "declaradas_por_el_articulo": {
                "fuente": ESCENA_DECLARADA["fuente"] + "; " + ARTICULO_FUENTE,
                "escena": {k: v for k, v in ESCENA_DECLARADA.items() if k != "fuente"},
                "encuadre_por_camara": {c: d["encuadre"] for c, d in CAMARAS_DECLARADAS.items()},
                "dispositivo_por_camara": {c: d["dispositivo"] for c, d in CAMARAS_DECLARADAS.items()},
            },
            "no_verificadas_aqui": ("encuadre, distancia aparente, iluminación y fondo se toman del artículo y NO se verifican "
                                    "por inspección visual en esta corrida (no se decodifica ningún fotograma). PENDIENTE: "
                                    "inspección visual de una muestra por señante al hacer la extracción propia (1.c)."),
        },
        "diferencias_landmarks_publicados_vs_video": {
            "fuente": "DECISIONES_TESIS.txt N-02 (verificación manual del 13-sep-2026) reproducida por este script",
            "COLOR_BODY_vs_body": agregados["fotogramas_COLOR_BODY_frente_a_filas_body"],
            "COLOR_FACE_vs_face": agregados["fotogramas_COLOR_FACE_frente_a_filas_face"],
            "body_vs_manos": agregados["body_frente_a_manos"],
            "filas_todo_cero_por_modalidad": {k: agregados["totales_por_modalidad"][k]["filas_todo_cero"] for k in MODALIDADES_CSV},
            "nan_por_modalidad": {k: agregados["totales_por_modalidad"][k]["nan"] for k in MODALIDADES_CSV},
            "indice_csv_consecutivo_en_todas": agregados["csv_indice_consecutivo_en_todas"],
            "hecho_verificado": ("Los CSV publicados no cubren todos los fotogramas del video y no contienen filas «vacías» "
                                 "(todo-cero); su columna índice está renumerada 0..n−1, así que no identifica qué fotogramas faltan."),
            "causa_del_deficit": {
                "estado": "DESCONOCIDA",
                "hipotesis_N02": ("Que las filas que faltan sean los fotogramas sin detección de MediaPipe "
                                  "(DECISIONES_TESIS.txt, N-02, 13-sep-2026). No confirmada con los autores."),
                "evidencia_en_contra_N09": ("La re-extracción propia con Tasks API (DECISIONES_TESIS.txt, N-09, 14-sep-2026) "
                                            "detecta la postura en 61/61 fotogramas de 0025_0002_0001 y en 78/78 de "
                                            "0049_0004_0003, con 0 y 2 fotogramas sin mano alguna, y aun así los CSV "
                                            "publicados tienen 4 filas menos en ambas: el déficit no se explica por "
                                            "fotogramas sin detección."),
                "pendiente": ("Alineamiento monótono entre lo publicado y lo re-extraído con las METRICAS_FOTOGRAMA de la "
                              "corrida completa (N-09). Hasta entonces se reporta: hecho verificado (déficit de filas) y "
                              "causa desconocida."),
                "fuente": "DECISIONES_TESIS.txt, entradas N-02 y N-09; este script solo reproduce las cifras de N-02",
            },
        },
        "duplicados_exactos": {
            "fuente": ("SHA-1 del archivo completo: CSV de LANDMARKS y, para los videos, candidatos por tamaño + SHA-1 de "
                       "los primeros 64 KiB confirmados con SHA-1 completo (bitacora_integridad.json → agregados)"),
            "grupos": grupos_dup,
            "grupos_que_afectan_al_entrenamiento": grupos_entrenamiento,
            "modalidades_que_afectan_al_entrenamiento": sorted(modalidades_entrenamiento),
            "muestras_unicas_en_COLOR_BODY_body_lh_rh": n_unicas_body,
            "cotejo_con_particiones": cotejo_part,
            "hecho_verificado": ("Hay pares de muestras con nombre distinto (misma seña y señante, repeticiones consecutivas) "
                                 "cuyo archivo es byte a byte idéntico. La duplicación es por modalidad: un par puede tener "
                                 "idénticos COLOR_BODY y los tres CSV body/lh/rh y distintos COLOR_FACE y GRAY_IR, o al revés."),
            "hipotesis_no_confirmada": ("Error de copia al ensamblar el depósito (una toma publicada dos veces con dos números "
                                        "de repetición). No confirmado con los autores."),
            "consecuencia": ("Para el entrenamiento sobre COLOR_BODY/body/lh/rh el corpus tiene tantas secuencias únicas como "
                             "indica «muestras_unicas_en_COLOR_BODY_body_lh_rh». Si un par queda repartido entre conjuntos de una "
                             "partición, el modelo se evalúa sobre una copia exacta de una secuencia de entrenamiento: ver "
                             "«cotejo_con_particiones». Qué hacer con ellos (excluir la copia o declararlo) es decisión para "
                             "DECISIONES_TESIS.txt, no de este script."),
        },
        "cotejo_con_metadata_json": cotejo_metadata,
    }


# ---------------------------------------------------------------------------
# Informe legible (Markdown)
# ---------------------------------------------------------------------------

def _tabla(encabezados: list[str], filas: list[list]) -> str:
    out = ["| " + " | ".join(encabezados) + " |", "|" + "|".join("---" for _ in encabezados) + "|"]
    for f in filas:
        out.append("| " + " | ".join("" if x is None else str(x) for x in f) + " |")
    return "\n".join(out)


def _fila_estad(nombre: str, e: dict | None) -> list:
    if not e:
        return [nombre, "—", "—", "—", "—", "—", "—"]
    return [nombre, e["n"], f"{e['min']:g}", f"{e['max']:g}", f"{e['media']:g}", f"{e['mediana']:g}", f"{e['de']:g}"]


def escribir_md(car: dict, bit: dict, path: Path) -> None:
    ag = bit["agregados"]
    L = []
    L.append("# Caracterización del corpus LSC50 (actividad 1.b) y resumen de la bitácora de integridad (1.a)\n")
    L.append(f"Generado por `src/caracterizar_lsc50.py` el {bit['fecha_verificacion']} sobre "
             f"{ag['muestras_verificadas']} de {ag['muestras_esperadas']} muestras esperadas. "
             "Cada cifra indica de qué archivo o cabecera sale. Los archivos de datos completos son "
             "`bitacora_integridad.json` (por muestra) y `caracterizacion.json`.\n")

    # 1. Identidad
    idc = car["identidad_corpus"]
    L.append("## 1. Identidad y versión del corpus\n")
    L.append(_tabla(["Campo", "Valor", "Fuente"], [
        ["Nombre", idc["nombre"], idc["deposito_de_datos"]["fuente"]],
        ["Artículo descriptor", idc["articulo_descriptor"]["cita"], idc["articulo_descriptor"]["fuente"]],
        ["DOI del artículo", idc["articulo_descriptor"]["doi"], "ref19"],
        ["Licencia del artículo", idc["articulo_descriptor"]["licencia"], "anteproyecto §1.4"],
        ["Depósito de datos", f"{idc['deposito_de_datos']['repositorio']}, versión {idc['deposito_de_datos']['version']}, "
                              f"publicado {idc['deposito_de_datos']['fecha_publicacion']}", "ref19b"],
        ["DOI del depósito", idc["deposito_de_datos"]["doi"], "ref19b"],
        ["URL del depósito", idc["deposito_de_datos"]["url"], "README.md"],
        ["Código de los autores", idc["deposito_de_datos"]["codigo_autores"], "README.md"],
        ["Licencia de los datos", idc["deposito_de_datos"]["licencia"], "ref19b, anteproyecto §1.4"],
        ["Fecha de esta verificación", bit["fecha_verificacion"], "reloj del sistema"],
        ["Fecha de obtención", bit["obtencion"]["nota"], "mtimes del sistema de archivos"],
        ["Sumas de comprobación publicadas", bit["sumas_de_comprobacion"]["resultado"], "API de figshare; búsqueda en data/raw/LSC50 y de los ZIP en disco"],
    ]))
    L.append("")
    sc = bit["sumas_de_comprobacion"]
    fs = sc["publicadas_por_figshare"]
    zips = sc["zips_originales_en_disco"]
    L.append("### 1.1 Sumas de comprobación que publica figshare (por ZIP) y cotejo con los ZIP en disco\n")
    L.append(f"Fuente: {fs['fuente']}. Nivel: {fs['nivel']}. Consulta en vivo: {fs['consulta_en_vivo_resumen']}.\n")
    L.append(_tabla(["ZIP", "Bytes (figshare)", "MD5 (figshare)", "En disco", "Bytes coinciden", "MD5 coincide"], [
        [a["name"], f"{a['size']:,}".replace(",", "."), a["computed_md5"],
         "sí" if zips.get(a["name"], {}).get("encontrado") else "no",
         zips.get(a["name"], {}).get("bytes_coinciden_con_figshare", "—"),
         zips.get(a["name"], {}).get("md5_coincide_con_figshare", zips.get(a["name"], {}).get("md5", "—"))]
        for a in fs["instantanea"]["archivos"]]))
    L.append(f"\nArchivos de sumas dentro del material descargado: {sc['archivos_de_sumas_en_el_material_descargado'] or 'ninguno'}.\n")

    # 2. Cotejo con lo anticipado
    ant = car["cifras_anticipadas_por_el_anteproyecto"]
    L.append("## 2. Cotejo con lo que el corpus declara publicar y con lo que anticipa el anteproyecto\n")
    L.append(f"Fuente de lo anticipado: {ant['fuente']}. Fuente de lo verificado: {ant['senas']['fuente']}.\n")
    L.append(_tabla(["Magnitud", "Anticipado", "Verificado"], [
        ["Señas (clases)", ant["senas"]["anticipado"], ant["senas"]["verificado"]],
        ["Señantes", ant["senantes"]["anticipado"], ant["senantes"]["verificado"]],
        ["Repeticiones por seña y señante", ant["repeticiones_por_sena_y_senante"]["anticipado"],
         f"{ant['repeticiones_por_sena_y_senante']['verificado_min']}–{ant['repeticiones_por_sena_y_senante']['verificado_max']}"],
        ["Secuencias", ant["secuencias"]["anticipado"], ant["secuencias"]["verificado"]],
        ["Videos .avi (declarado por el artículo)", ant["videos_avi"]["declarado_por_el_articulo"], ant["videos_avi"]["verificado"]],
        ["Medidas inerciales (declarado por el artículo; fuera del alcance)", ant["medidas_inerciales"]["declarado_por_el_articulo"],
         f"{ant['medidas_inerciales']['verificado_filas_en_Timestamps_xlsx']} filas en Timestamps.xlsx"],
    ]))
    L.append("")
    L.append("Archivos encontrados por carpeta (esperados 1000 en cada una):\n")
    L.append(_tabla(["Carpeta", "Archivos", "Bytes"], [
        [k, v, f"{ag['bytes_por_carpeta'].get(k, 0):,}".replace(",", ".")] for k, v in ag["archivos_por_carpeta"].items()]))
    L.append("")

    # 3. Vocabulario
    voc = car["vocabulario"]
    L.append("## 3. Vocabulario efectivo (50 señas con ID)\n")
    L.append(f"Fuente de la glosa del código: {voc['fuente_codigo']}. Fuente de la glosa publicada: {voc['fuente_publicada']}.\n")
    L.append(_tabla(["ID", "Glosa (código)", "Glosa publicada (Timestamps.xlsx)", "Dinámica / estática"], [
        [f"{s['id']:04d}", s["glosa_codigo"], " / ".join(s["glosa_publicada_Timestamps_xlsx"] or ["—"]),
         s["condicion_dinamica_estatica"]] for s in voc["senas"]]))
    L.append("")
    if voc["discrepancias_codigo_vs_publicado"]:
        L.append("Discrepancias entre la glosa del código y la publicada (se reportan tal cual, para la validación 1.d):\n")
        L.append(_tabla(["ID", "Código", "Publicado", "Motivo"], [
            [f"{d['id']:04d}", d["codigo"], d["publicado"], d["motivo"]] for d in voc["discrepancias_codigo_vs_publicado"]]))
    else:
        L.append("Sin discrepancias entre la glosa del código y la publicada tras normalizar.")
    L.append("")
    L.append(f"Nota sobre la Tabla 2 del artículo: {voc['nota_tabla_2_del_articulo']}\n")
    L.append(f"Nota: {voc['nota_timestamps']}\n")
    L.append(f"**Condición dinámica/estática por seña: {car['condicion_dinamica_estatica']['estado']}.** "
             f"{car['condicion_dinamica_estatica']['nota']}\n")

    # 4. Distribución
    dist = car["distribucion"]
    L.append("## 4. Muestras por clase, por señante y repeticiones\n")
    L.append(f"Fuente: {dist['fuente']}.\n")
    L.append(f"- Muestras por clase: {min(dist['muestras_por_clase'].values())}–{max(dist['muestras_por_clase'].values())} "
             f"({'balanceado' if dist['balanceado_por_clase'] else 'NO balanceado'}).")
    L.append(f"- Muestras por señante: " + ", ".join(f"{k}: {v}" for k, v in dist["muestras_por_senante"].items())
             + f" ({'balanceado' if dist['balanceado_por_senante'] else 'NO balanceado'}).")
    L.append(f"- Muestras por índice de repetición: " + ", ".join(f"{k}: {v}" for k, v in dist["muestras_por_repeticion"].items()) + ".")
    celdas = Counter(v for fila in dist["muestras_por_clase_y_senante"].values() for v in fila.values())
    L.append(f"- Celdas clase × señante: " + ", ".join(f"{n} celdas con {k} muestras" for k, n in sorted(celdas.items())) + ".\n")

    # 5. Fotogramas
    fpm = car["fotogramas_por_modalidad"]
    L.append("## 5. Fotogramas por modalidad\n")
    L.append(f"Fuente: {fpm['fuente']}.\n")
    filas = [_fila_estad(f"CSV {k}", e) for k, e in fpm["landmarks_publicados"].items()]
    filas.append(_fila_estad("CSV body/lh/rh truncados al mínimo (= preprocess_lsc50.py)", fpm["landmarks_body_lh_rh_truncados_al_minimo_como_preprocess"]))
    filas += [_fila_estad(f"Video {c}", e) for c, e in fpm["videos"].items()]
    L.append(_tabla(["Modalidad", "n", "mín", "máx", "media", "mediana", "DE"], filas))
    L.append("")
    L.append("Duración de los clips en segundos (fotogramas del video / fps medido de la cámara):\n")
    L.append(_tabla(["Cámara", "n", "mín", "máx", "media", "mediana", "DE"],
                    [_fila_estad(c, e) for c, e in fpm["duracion_s_por_camara_con_fps_medido"].items()]))
    L.append("")

    # 6. Cámaras
    cam = car["camaras"]
    L.append("## 6. Resolución, tasa de fotogramas y códec por cámara (medidos)\n")
    L.append(f"Fuente: {cam['fuente']}.\n")
    L.append(_tabla(["Cámara", "Resolución (moda)", "fps (moda)", "Códec", "Valores de fps observados", "Resoluciones observadas"], [
        [c, m["resolucion"], m["fps"], m["codec"], json.dumps(m["fps_valores"]), json.dumps(m["resolucion_valores"])]
        for c, m in cam["por_camara"].items()]))
    L.append("")
    L.append(f"Cotejo con lo que declara el artículo (fuente: {cam['declarado_por_el_articulo']['fuente']}):\n")
    L.append(_tabla(["Cámara", "Dispositivo (artículo)", "fps declarado / medido", "Resolución declarada / medida", "Coincide", "Todos los archivos en la moda"], [
        [c, CAMARAS_DECLARADAS[c]["dispositivo"], f"{k['fps_declarado']:g} / {k['fps_medido_moda']}",
         f"{k['resolucion_declarada']} / {k['resolucion_medida_moda']}",
         "sí" if k["fps_coincide"] and k["resolucion_coincide"] else "NO", "sí" if k["todos_los_archivos_en_la_moda"] else "NO"]
        for c, k in cam["cotejo_declarado_vs_medido"].items()]))
    L.append("")

    # 7. Rostro/cuerpo
    dr = car["desalineamiento_rostro_cuerpo"]
    L.append("## 7. Desalineamiento rostro / cuerpo\n")
    L.append(f"Fuente: {dr['fuente']}.\n")
    L.append(_tabla(["Magnitud", "n", "mín", "máx", "media", "mediana", "DE"], [
        _fila_estad("filas FACE / filas BODY", dr["ratio_filas_face_body"]),
        _fila_estad("fotogramas COLOR_FACE / COLOR_BODY", dr["ratio_fotogramas_COLOR_FACE_COLOR_BODY"]),
    ]))
    L.append(f"\nCociente de fps medidos COLOR_FACE / COLOR_BODY: {dr['fps_COLOR_FACE_sobre_fps_COLOR_BODY']}. "
             f"FACE frente a su propio video (COLOR_FACE): {dr['face_frente_a_su_video']['video_mas_fotogramas_que_filas']} muestras con más "
             f"fotogramas que filas, {dr['face_frente_a_su_video']['video_igual_filas']} iguales, "
             f"{dr['face_frente_a_su_video']['video_menos_fotogramas_que_filas']} con menos.\n")
    L.append(dr["consecuencia"] + "\n")

    # 8. Señantes
    se = car["senantes"]
    L.append("## 8. Señantes\n")
    L.append(f"Número de señantes: {se['numero']} (fuente: {se['fuente_numero']}).\n")
    L.append("Condición tal como la publica el artículo descriptor, con sus dos criterios **sin conciliar**:\n")
    L.append(_tabla(["Sección del artículo", "Lo que publica"], [
        ["Methods", se["condicion_segun_el_articulo"]["criterio_methods"]],
        ["Data Records", se["condicion_segun_el_articulo"]["criterio_data_records"]],
    ]))
    L.append(f"\n**Advertencia:** {se['condicion_segun_el_articulo']['advertencia']}\n")
    L.append("Otros datos que el artículo publica sobre ellos: " + " ".join(se["condicion_segun_el_articulo"]["otros_datos_publicados"]) + "\n")
    L.append(f"Fuente: {se['condicion_segun_el_articulo']['fuente']}\n")
    L.append("Citas literales del artículo sobre los señantes:\n")
    for c in car["citas_del_articulo"]["senantes"]:
        L.append(f"- [{c['seccion']}] «{c['cita']}»")
    L.append("")
    an = se["antropometria_publicada"]
    if an["datos"]:
        claves = sorted({k for d in an["datos"].values() for k in d})
        L.append(f"Antropometría publicada por los autores (fuente: {an['fuente']}):\n")
        L.append(_tabla(["Archivo"] + claves, [[v] + [d.get(k, "") for k in claves] for v, d in an["datos"].items()]))
        L.append(f"\n{an['advertencia']}\n")

    # 9. Condiciones de registro
    cr = car["condiciones_de_registro"]
    L.append("## 9. Condiciones de registro\n")
    L.append("Medidas aquí: " + "; ".join(f"{k}: {v}" for k, v in cr["medidas_aqui"].items()) + ".\n")
    dec = cr["declaradas_por_el_articulo"]
    L.append(f"Declaradas por el artículo (fuente: {dec['fuente']}):\n")
    L.append(_tabla(["Magnitud", "Valor declarado"], [[k, v] for k, v in dec["escena"].items()]
                    + [[f"encuadre {c}", v] for c, v in dec["encuadre_por_camara"].items()]
                    + [[f"dispositivo {c}", v] for c, v in dec["dispositivo_por_camara"].items()]))
    L.append("")
    L.append("Citas literales sobre el registro:\n")
    for c in car["citas_del_articulo"]["registro"]:
        L.append(f"- [{c['seccion']} · {c['camara']}] «{c['cita']}»")
    L.append("")
    L.append(f"No verificadas aquí: {cr['no_verificadas_aqui']}\n")
    L.append("Lo que el artículo NO dice: " + " ".join(f"({i + 1}) {t}" for i, t in enumerate(car["citas_del_articulo"]["lo_que_el_articulo_no_dice"])) + "\n")

    # 10. Diferencias landmarks vs video
    dv = car["diferencias_landmarks_publicados_vs_video"]
    cb = dv["COLOR_BODY_vs_body"]
    bm = dv["body_vs_manos"]
    L.append("## 10. Diferencias entre landmarks publicados y video (N-02)\n")
    L.append(f"Fuente: {dv['fuente']}. Comparación: {cb['fuente']}.\n")
    pos = cb["diferencia_sobre_las_positivas"] or {}
    L.append(_tabla(["Magnitud", "Valor"], [
        ["Muestras con MÁS fotogramas en el video que filas en el CSV body", cb["video_mas_fotogramas_que_filas"]],
        ["Muestras con el mismo número", cb["video_igual_filas"]],
        ["Muestras con MENOS fotogramas que filas", cb["video_menos_fotogramas_que_filas"]],
        ["Diferencia (sobre las positivas): máx / mediana / media",
         f"{pos.get('max', '—'):g} / {pos.get('mediana', '—'):g} / {pos.get('media', '—'):g}" if pos else "—"],
        ["Muestras con body / mano izq. / mano der. de longitud distinta", bm["muestras_con_longitud_distinta"]],
        ["  · con diferencia 1–2 (preprocess_lsc50.py las trunca al mínimo, sin registro)", bm["muestras_con_diferencia_1_o_2_truncadas_por_preprocess"]],
        ["  · con diferencia > 2 (preprocess_lsc50.py las descarta)", bm["muestras_con_diferencia_mayor_que_2_descartadas_por_preprocess"]],
        ["Filas todo-cero por modalidad", json.dumps(dv["filas_todo_cero_por_modalidad"])],
        ["NaN por modalidad", json.dumps(dv["nan_por_modalidad"])],
        ["Columna índice consecutiva 0..n−1 en todos los CSV", dv["indice_csv_consecutivo_en_todas"]],
    ]))
    L.append(f"\nHistograma de (fotogramas COLOR_BODY − filas body): {json.dumps(cb['histograma_diferencia'])}\n")
    L.append(f"Histograma de (máx − mín) entre body, lh y rh: {json.dumps(bm['histograma_max_menos_min'])}\n")
    L.append(f"**Hecho verificado:** {dv['hecho_verificado']}\n")
    cd = dv["causa_del_deficit"]
    L.append(f"**Causa del déficit: {cd['estado']}.** Hipótesis de N-02: {cd['hipotesis_N02']} "
             f"Evidencia en contra (N-09): {cd['evidencia_en_contra_N09']} Pendiente: {cd['pendiente']} "
             f"(Fuente: {cd['fuente']}.)\n")

    # 11. Bitácora: anomalías
    L.append("## 11. Resumen de la bitácora de integridad (1.a)\n")
    L.append(_tabla(["Anomalía", "Muestras"], [[k, v] for k, v in ag["recuento_anomalias"].items()]) if ag["recuento_anomalias"]
             else "Sin anomalías registradas.")
    L.append("")
    L.append(f"- Muestras ausentes de la rejilla 50×5×4: {len(ag['muestras_ausentes_de_la_rejilla'])}"
             + (f" → {ag['muestras_ausentes_de_la_rejilla'][:20]}" if ag["muestras_ausentes_de_la_rejilla"] else ""))
    L.append(f"- Nombres fuera de la rejilla: {len(ag['nombres_fuera_de_la_rejilla'])}"
             + (f" → {ag['nombres_fuera_de_la_rejilla'][:20]}" if ag["nombres_fuera_de_la_rejilla"] else ""))
    L.append(f"- Archivos ausentes: {len(ag['archivos_ausentes'])}" + (f" → {ag['archivos_ausentes'][:20]}" if ag["archivos_ausentes"] else ""))
    L.append(f"- Muestras con algún archivo ilegible: {len(ag['muestras_con_archivo_ilegible'])}"
             + (f" → {ag['muestras_con_archivo_ilegible'][:20]}" if ag["muestras_con_archivo_ilegible"] else ""))
    if isinstance(ag["duplicados"], dict):
        n_dup = {k: len(v) for k, v in ag["duplicados"].items()}
        L.append(f"- Grupos de duplicados por carpeta (CSV: SHA-1 completo; video: criba por tamaño + SHA-1 de los primeros "
                 f"{BYTES_HASH_VIDEO} bytes y confirmación con SHA-1 completo): {json.dumps(n_dup)}")
        dv_ = ag["duplicados_video_detalle"]
        refutados = {k: v for k, v in dv_["refutados"].items() if v}
        L.append(f"- Candidatos de video refutados por el SHA-1 completo: {json.dumps(refutados) if refutados else 'ninguno'}")
    else:
        L.append(f"- Duplicados: {ag['duplicados']}")
    L.append("")
    de = car["duplicados_exactos"]
    L.append("### 11.1 Duplicados exactos (archivos byte a byte idénticos bajo nombres distintos)\n")
    L.append(f"Fuente: {de['fuente']}.\n")
    if de["grupos"]:
        L.append(_tabla(["Muestras", "Modalidades idénticas", "Afecta al entrenamiento (body/lh/rh/COLOR_BODY)"], [
            [" = ".join(g["muestras"]), ", ".join(g["modalidades"]),
             "sí" if g["muestras"] in de["grupos_que_afectan_al_entrenamiento"] else "no"]
            for g in de["grupos"]]))
        L.append(f"\nMuestras únicas para el entrenamiento (COLOR_BODY / body / lh / rh): "
                 f"{de['muestras_unicas_en_COLOR_BODY_body_lh_rh']} de {ag['muestras_verificadas']}.\n")
        for esquema, r in de["cotejo_con_particiones"].items():
            if not r.get("encontrado"):
                L.append(f"- Partición {esquema}: archivo no encontrado ({r['archivo']}).")
                continue
            h = r["grupos_repartidos_entre_conjuntos"]
            L.append(f"- Partición {esquema} ({r['bloques_revisados']} bloque(s), {r['grupos_revisados']} grupo(s)): "
                     + ("ningún grupo repartido entre conjuntos." if not h else
                        f"{len(h)} caso(s) repartidos entre conjuntos → "
                        + "; ".join(f"{x['bloque']}: {json.dumps(x['muestras'])}" for x in h)))
        L.append("")
        L.append(f"**Hecho verificado:** {de['hecho_verificado']}\n")
        L.append(f"**Hipótesis no confirmada:** {de['hipotesis_no_confirmada']}\n")
        L.append(f"**Consecuencia:** {de['consecuencia']}\n")
    else:
        L.append("Sin duplicados exactos (o no calculados con --sin-hash).\n")

    # 12. Comprobaciones
    comp = bit["comprobaciones_N02"]
    L.append("## 12. Comprobaciones frente a la verificación manual del 13-sep-2026 (N-02)\n")
    if not comp.get("aplicable"):
        L.append(f"No aplicables: {comp.get('motivo')}\n")
    else:
        L.append(_tabla(["Comprobación", "Obtenido", "Esperado", "OK"], [
            [k, v[0], v[1], "sí" if v[0] == v[1] else "NO"] for k, v in comp.items() if isinstance(v, tuple)]))
        L.append("")
        L.append(_tabla(["Muestra", "Fotogramas COLOR_BODY", "Filas body", "Filas todo-cero en manos", "OK"], [
            [m, d["obtenido"]["fotogramas"], d["obtenido"]["filas_body"], d["obtenido"]["filas_todo_cero_manos"], "sí" if d["ok"] else "NO"]
            if d.get("encontrada") else [m, "—", "—", "—", "NO ENCONTRADA"] for m, d in comp["muestras_detalle"].items()]))
        L.append(f"\n**Resultado global: {'todas las comprobaciones coinciden' if comp['todo_ok'] else 'HAY DISCREPANCIAS'}.**\n")

    # 13. Cotejo con metadata.json
    cm = car["cotejo_con_metadata_json"]
    if cm:
        L.append("## 13. Cotejo con data/processed/metadata.json\n")
        L.append(f"Fuente: {cm['fuente']}.\n")
        L.append(f"- n_samples en metadata: {cm['n_samples_metadata']}; sample_order igual a la rejilla ordenada: {cm['sample_order_igual_a_rejilla']}.")
        L.append(f"- frame_stats en metadata: {json.dumps(cm['frame_stats_metadata'])}")
        L.append(f"- Recalculados aquí (mín de body/lh/rh por muestra): {json.dumps(cm['frame_stats_recalculados_min_body_lh_rh'])}")
        L.append(f"- Coinciden (mín, máx, media): {cm['coinciden']}\n")

    L.append("## Entorno de la corrida\n")
    L.append("; ".join(f"{k}: {v}" for k, v in bit["entorno"].items()) + "\n")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# CSV plano por muestra
# ---------------------------------------------------------------------------

def escribir_csv_plano(registros: list[dict], path: Path) -> None:
    cols = ["muestra", "sena_id", "senante_id", "repeticion"]
    for k in MODALIDADES_CSV:
        cols += [f"{k}_existe", f"{k}_bytes", f"{k}_filas", f"{k}_columnas", f"{k}_nan", f"{k}_filas_todo_cero", f"{k}_indice_consecutivo"]
    for c in CAMARAS:
        cols += [f"{c}_existe", f"{c}_bytes", f"{c}_fotogramas", f"{c}_fps", f"{c}_ancho", f"{c}_alto", f"{c}_codec"]
    cols += ["body_menos_lh", "body_menos_rh", "max_menos_min_body_manos", "fotogramas_COLOR_BODY_menos_filas_body",
             "fotogramas_COLOR_FACE_menos_filas_face", "ratio_filas_face_body", "anomalias"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in registros:
            fila = [r["muestra"], r["sena_id"], r["senante_id"], r["repeticion"]]
            for k in MODALIDADES_CSV:
                a = r["archivos"].get(k, {})
                c = r["csv"].get(k, {})
                fila += [a.get("existe"), a.get("bytes"), c.get("filas"), c.get("columnas"), c.get("nan"),
                         c.get("filas_todo_cero"), c.get("indice_consecutivo")]
            for cam in CAMARAS:
                a = r["archivos"].get(cam, {})
                v = r["video"].get(cam, {})
                fila += [a.get("existe"), a.get("bytes"), v.get("fotogramas"), v.get("fps"), v.get("ancho"), v.get("alto"),
                         v.get("codec_strf")]
            d = r["diferencias"]
            fila += [d.get("body_menos_lh"), d.get("body_menos_rh"), d.get("max_menos_min_body_manos"),
                     d.get("fotogramas_COLOR_BODY_menos_filas_body"), d.get("fotogramas_COLOR_FACE_menos_filas_face"),
                     d.get("ratio_filas_face_body"), ";".join(r["anomalias"])]
            w.writerow(fila)


# ---------------------------------------------------------------------------
# Autoprueba (unittest con datos sintéticos en un directorio temporal; no toca el corpus)
# ---------------------------------------------------------------------------

def _avi_sintetico(fotogramas: int, fps: int, ancho: int, alto: int, codec: bytes = b"XVID",
                   relleno: int = 16, carga: bytes | None = None) -> bytes:
    """AVI mínimo pero bien formado (RIFF/hdrl/avih/strl/strh/strf/movi) para probar leer_cabecera_avi.

    `carga` es el contenido del chunk de datos; si es None se usan `relleno` bytes a cero.
    """
    def chunk(ident: bytes, datos: bytes) -> bytes:
        return ident + struct.pack("<I", len(datos)) + datos + (b"\x00" if len(datos) % 2 else b"")

    def lista(tipo: bytes, datos: bytes) -> bytes:
        return chunk(b"LIST", tipo + datos)

    avih = struct.pack("<14I", int(round(1e6 / fps)), 0, 0, 0x10, fotogramas, 0, 1, 0, ancho, alto, 0, 0, 0, 0)
    strh = (b"vids" + codec + struct.pack("<IHHIIIIIIII", 0, 0, 0, 0, 1, fps, 0, fotogramas, 0, 0, 0)
            + struct.pack("<4H", 0, 0, ancho, alto))
    strf = struct.pack("<IiiHH", 40, ancho, alto, 1, 24) + codec + struct.pack("<IiiII", ancho * alto * 3, 0, 0, 0, 0)
    hdrl = lista(b"hdrl", chunk(b"avih", avih) + lista(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf)))
    movi = lista(b"movi", chunk(b"00dc", carga if carga is not None else bytes(relleno)))
    cuerpo = b"AVI " + hdrl + movi
    return b"RIFF" + struct.pack("<I", len(cuerpo)) + cuerpo


def _csv_sintetico(path: Path, n_puntos: int, filas: list[list[float]]) -> None:
    """CSV con el formato publicado: columna índice sin nombre + landmark_i_x/y/z."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"landmark_{i}_{c}" for i in range(n_puntos) for c in ("x", "y", "z")])
        for i, fila in enumerate(filas):
            w.writerow([i] + fila)


class PruebasSinteticas(unittest.TestCase):
    """Pruebas de las piezas de lectura y de la lógica de anomalías sobre un corpus sintético en un tmpdir."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.raw = Path(cls.tmp.name) / "LSC50"
        for sub, _ in MODALIDADES_CSV.values():
            (cls.raw / sub).mkdir(parents=True)
        for sub in CAMARAS.values():
            (cls.raw / sub).mkdir(parents=True)
        rng = np.random.default_rng(0)

        def filas(n, puntos):
            return [list(np.round(rng.random(3 * puntos), 6)) for _ in range(n)]

        cls.A, cls.B, cls.C, cls.D = "0000_0000_0000", "0000_0000_0001", "0000_0000_0002", "0000_0000_0003"
        # A: limpia; COLOR_BODY con 2 fotogramas más que filas (como en N-02); COLOR_FACE > 64 KiB
        cls.body_A = filas(10, 33)
        cls._escribir(cls.A, body=cls.body_A, lh=filas(10, 21), rh=filas(10, 21), face=filas(30, 468),
                      videos={"COLOR_BODY": (12, 24, 1920, 1080), "COLOR_FACE": (30, 50, 640, 424),
                              "GRAY_DEPTH": (15, 30, 512, 424), "GRAY_IR": (15, 30, 512, 424)},
                      relleno_face=70_000)
        # B: lh dos filas más corta (truncada por preprocess), NaN en rh, una fila todo-cero en body,
        #    video con tantos fotogramas como filas, sin GRAY_IR
        body_B = filas(10, 33)
        body_B[3] = [0.0] * 99
        rh_B = filas(10, 21)
        rh_B[2][5] = float("nan")
        cls._escribir(cls.B, body=body_B, lh=filas(8, 21), rh=rh_B, face=filas(30, 468),
                      videos={"COLOR_BODY": (10, 24, 1920, 1080), "COLOR_FACE": (30, 50, 640, 424),
                              "GRAY_DEPTH": (15, 30, 512, 424)})
        # C: body y COLOR_BODY idénticos a A (duplicado exacto: el AVI se copia byte a byte); lo demás distinto
        cls._escribir(cls.C, body=cls.body_A, lh=filas(10, 21), rh=filas(10, 21), face=filas(30, 468),
                      videos={"COLOR_BODY": (12, 24, 1920, 1080), "COLOR_FACE": (31, 50, 640, 424),
                              "GRAY_DEPTH": (15, 30, 512, 424), "GRAY_IR": (16, 30, 512, 424)})
        cuerpo_A = (cls.raw / CAMARAS["COLOR_BODY"] / f"{cls.A}.avi").read_bytes()
        (cls.raw / CAMARAS["COLOR_BODY"] / f"{cls.C}.avi").write_bytes(cuerpo_A)
        # D: COLOR_FACE con el mismo tamaño y los mismos primeros 64 KiB que A pero distinto en el último
        #    byte: candidato que el SHA-1 completo debe refutar
        cls._escribir(cls.D, body=filas(10, 33), lh=filas(10, 21), rh=filas(10, 21), face=filas(30, 468),
                      videos={"COLOR_BODY": (11, 24, 1920, 1080), "GRAY_DEPTH": (15, 30, 512, 424),
                              "GRAY_IR": (15, 30, 512, 424)})
        cara_A = (cls.raw / CAMARAS["COLOR_FACE"] / f"{cls.A}.avi").read_bytes()
        (cls.raw / CAMARAS["COLOR_FACE"] / f"{cls.D}.avi").write_bytes(cara_A[:-1] + bytes([cara_A[-1] ^ 0xFF]))
        cls.nombres = [cls.A, cls.B, cls.C, cls.D]

    @classmethod
    def _escribir(cls, nombre, body, lh, rh, face, videos, relleno_face=16):
        for clave, datos in (("body", body), ("lh", lh), ("rh", rh), ("face", face)):
            sub, n_puntos = MODALIDADES_CSV[clave]
            _csv_sintetico(cls.raw / sub / f"{nombre}.csv", n_puntos, datos)
        for cam, (fot, fps, w, h) in videos.items():
            relleno = relleno_face if cam == "COLOR_FACE" else 16
            # Carga útil distinta por muestra y cámara: dos AVI sintéticos con la misma cabecera y el mismo
            # relleno serían byte a byte idénticos y el detector de duplicados los marcaría (con razón)
            carga = (f"{nombre}:{cam}:".encode("ascii") * (relleno // 8 + 1))[:relleno]
            (cls.raw / CAMARAS[cam] / f"{nombre}.avi").write_bytes(_avi_sintetico(fot, fps, w, h, carga=carga))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rejilla_y_nombres(self):
        self.assertEqual(len(nombres_esperados()), 1000)
        self.assertEqual(parse_nombre("0049_0004_0003"), (49, 4, 3))
        self.assertIsNone(parse_nombre("49_4_3"))

    def test_cabecera_avi(self):
        info = leer_cabecera_avi(self.raw / CAMARAS["COLOR_BODY"] / f"{self.A}.avi")
        self.assertTrue(info["legible"])
        self.assertEqual((info["fotogramas"], info["fotogramas_strh"]), (12, 12))
        self.assertEqual((info["fps"], info["ancho"], info["alto"]), (24.0, 1920, 1080))
        self.assertEqual(info["codec_strf"], "XVID")
        self.assertTrue(info["riff_coherente_con_tamano"])
        basura = self.raw / "no_es_avi.avi"
        basura.write_bytes(b"esto no es un AVI")
        self.assertFalse(leer_cabecera_avi(basura)["legible"])

    def test_csv_landmarks(self):
        info = leer_csv_landmarks(self.raw / MODALIDADES_CSV["rh"][0] / f"{self.B}.csv", 21)
        self.assertTrue(info["legible"] and info["cabecera_ok"] and info["columnas_ok"])
        self.assertEqual((info["filas"], info["columnas"], info["nan"], info["filas_con_nan"]), (10, 63, 1, 1))
        info = leer_csv_landmarks(self.raw / MODALIDADES_CSV["body"][0] / f"{self.B}.csv", 33)
        self.assertEqual((info["filas"], info["filas_todo_cero"], info["indice_consecutivo"]), (10, 1, True))

    def test_muestras_y_agregados(self):
        registros = [verificar_muestra(n, self.raw, con_hash=True) for n in self.nombres]
        por = {r["muestra"]: r for r in registros}
        self.assertEqual(por[self.A]["diferencias"]["fotogramas_COLOR_BODY_menos_filas_body"], 2)
        self.assertIn("video_mas_fotogramas_que_filas_body", por[self.A]["anomalias"])
        self.assertEqual(por[self.B]["diferencias"]["max_menos_min_body_manos"], 2)
        self.assertEqual(por[self.B]["diferencias"]["filas_tras_truncar_al_minimo"], 8)
        for etiqueta in ("body_manos_longitud_distinta", "truncada_por_preprocess", "nan:rh",
                         "filas_todo_cero:body", "ausente:GRAY_IR"):
            self.assertIn(etiqueta, por[self.B]["anomalias"])
        self.assertNotIn("video_mas_fotogramas_que_filas_body", por[self.B]["anomalias"])
        self.assertFalse(por[self.B]["archivos"]["GRAY_IR"]["existe"])

        encontrados = {k: 4 for k in list(MODALIDADES_CSV) + list(CAMARAS)}
        ag = agregar(registros, self.nombres + ["0000_0001_0000"], encontrados, True, self.raw)
        self.assertEqual(ag["muestras_ausentes_de_la_rejilla"], ["0000_0001_0000"])
        rec = ag["recuento_anomalias"]
        self.assertEqual(rec["video_mas_fotogramas_que_filas_body"], 3)   # A, C y D
        self.assertEqual(rec["body_manos_longitud_distinta"], 1)
        self.assertEqual(rec["duplicado:body"], 2)
        self.assertEqual(rec["duplicado:COLOR_BODY"], 2)
        self.assertNotIn("duplicado:COLOR_FACE", rec)
        self.assertNotIn("posible_duplicado:COLOR_FACE", rec)
        self.assertEqual(ag["duplicados"]["body"], [[self.A, self.C]])
        self.assertEqual(ag["duplicados"]["COLOR_BODY"], [[self.A, self.C]])
        self.assertEqual(ag["duplicados_video_detalle"]["candidatos"]["COLOR_FACE"], [[self.A, self.D]])
        self.assertEqual(ag["duplicados_video_detalle"]["refutados"]["COLOR_FACE"], [[self.A, self.D]])
        self.assertEqual(ag["grupos_duplicados"], [{"muestras": [self.A, self.C], "modalidades": ["body", "COLOR_BODY"]}])
        self.assertEqual(ag["fotogramas_COLOR_BODY_frente_a_filas_body"]["video_igual_filas"], 1)
        self.assertEqual(ag["modas_por_camara"]["COLOR_BODY"]["resolucion"], "1920x1080")
        self.assertEqual(ag["modas_por_camara"]["COLOR_FACE"]["fps"], 50.0)
        self.assertFalse(comprobar_n02(registros, ag, completo=False)["aplicable"])

    def test_cotejo_particiones(self):
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            (dp / "aleatoria_80_10_10.json").write_text(json.dumps(
                {"conjuntos": {"train": [self.A, self.B], "val": [], "test": [self.C]}}), encoding="utf-8")
            (dp / "loso_5_pliegues.json").write_text(json.dumps(
                {"pliegues": [{"pliegue": 0, "senante_test": "0001",
                               "conjuntos": {"train": [self.A, self.C], "val": [self.B], "test": []}}]}), encoding="utf-8")
            res = cotejar_particiones([[self.A, self.C]], dp)
            self.assertEqual(len(res["aleatoria_80_10_10"]["grupos_repartidos_entre_conjuntos"]), 1)
            self.assertEqual(res["aleatoria_80_10_10"]["grupos_repartidos_entre_conjuntos"][0]["muestras"],
                             {self.A: "train", self.C: "test"})
            self.assertEqual(res["loso_5_pliegues"]["grupos_repartidos_entre_conjuntos"], [])
            self.assertFalse(cotejar_particiones([[self.A, self.C]], dp / "no_existe")["aleatoria_80_10_10"]["encontrado"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bitácora de integridad (1.a) y caracterización (1.b) del corpus LSC50.")
    ap.add_argument("--raw", type=Path, default=RAW_DIR, help="carpeta data/raw/LSC50")
    ap.add_argument("--metadata", type=Path, default=METADATA_PATH, help="data/processed/metadata.json")
    ap.add_argument("--salida", type=Path, default=SALIDA_DIR, help="carpeta de salida")
    ap.add_argument("--limite", type=int, default=None, help="verificar solo las N primeras muestras (prueba de humo)")
    ap.add_argument("--sin-hash", action="store_true", help="no calcular SHA-1 (sin detección de duplicados)")
    ap.add_argument("--particiones", type=Path, default=PARTICIONES_DIR, help="carpeta data/partitions (cotejo de duplicados)")
    ap.add_argument("--zips", type=Path, action="append", default=None,
                    help="carpeta(s) donde buscar IMU.zip, LANDMARKS.zip y VIDEOS.zip (por defecto data/, 3 niveles)")
    ap.add_argument("--verificar-md5", action="store_true", help="calcular el MD5 de los ZIP encontrados (lento: ~10 GB)")
    ap.add_argument("--consultar-figshare", action="store_true", help="refrescar en vivo la instantánea de la API de figshare")
    ap.add_argument("--autoprueba", action="store_true", help="ejecutar las pruebas unitarias con datos sintéticos y salir")
    args = ap.parse_args(argv)

    if args.autoprueba:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PruebasSinteticas)
        resultado = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if resultado.wasSuccessful() else 1

    raw: Path = args.raw
    if not raw.exists():
        print(f"[ERROR] no existe {raw}")
        return 1
    args.salida.mkdir(parents=True, exist_ok=True)
    con_hash = not args.sin_hash
    t0 = time.perf_counter()

    # Inventario de archivos por carpeta y nombres presentes en cualquiera de ellas
    carpetas = {k: raw / sub for k, (sub, _) in MODALIDADES_CSV.items()}
    carpetas.update({c: raw / sub for c, sub in CAMARAS.items()})
    encontrados = {}
    nombres_presentes = set()
    for k, d in carpetas.items():
        ext = ".csv" if k in MODALIDADES_CSV else ".avi"
        archivos = sorted(p.stem for p in d.glob(f"*{ext}")) if d.exists() else []
        encontrados[k] = len(archivos)
        nombres_presentes.update(archivos)
    esperados = nombres_esperados()
    nombres = sorted(set(esperados) | nombres_presentes)
    if args.limite:
        nombres = nombres[:args.limite]
    completo = (args.limite is None)
    print(f"Corpus: {raw}")
    print(f"Archivos por carpeta: {encontrados}")
    print(f"Muestras a verificar: {len(nombres)} (esperadas por la rejilla 50×5×4: {len(esperados)})")

    registros = []
    for i, nombre in enumerate(nombres, 1):
        registros.append(verificar_muestra(nombre, raw, con_hash))
        if i % 100 == 0 or i == len(nombres):
            print(f"  {i}/{len(nombres)}  ({time.perf_counter() - t0:.0f} s)")

    agregados = agregar(registros, esperados, encontrados, con_hash, raw)
    agregados["bytes_por_carpeta"] = {**{k: agregados["totales_por_modalidad"][k]["bytes"] for k in MODALIDADES_CSV},
                                      **agregados["bytes_por_camara"]}
    comprob = comprobar_n02(registros, agregados, completo)

    metadata = None
    if args.metadata.exists():
        with open(args.metadata, encoding="utf-8") as f:
            metadata = json.load(f)

    # Fecha de obtención: el repositorio no la registra; se deja la evidencia indirecta de los mtimes
    mtimes_archivo = [p.stat().st_mtime for k, d in carpetas.items() if d.exists() for p in list(d.iterdir())[:5]]
    mtimes_dir = {k: iso_mtime(d) for k, d in carpetas.items() if d.exists()}
    obtencion = {
        "nota": ("No registrada en el repositorio. Evidencia indirecta: los archivos conservan la fecha del depósito "
                 f"({dt.datetime.fromtimestamp(min(mtimes_archivo)).date().isoformat() if mtimes_archivo else '—'}); "
                 "las carpetas se materializaron en disco en las fechas de «mtime_carpetas». PENDIENTE: que el estudiante "
                 "consigne la fecha real de descarga desde figshare."),
        "mtime_carpetas": mtimes_dir,
    }
    sumas = buscar_sumas_de_comprobacion(raw)
    zips = buscar_zips_originales(args.zips or [DATA_DIR], args.verificar_md5)
    consulta = consultar_figshare() if args.consultar_figshare else None
    if consulta is None:
        consulta_resumen = "no solicitada (--consultar-figshare); se usa la instantánea del 14-sep-2026"
    elif "error" in consulta:
        consulta_resumen = consulta["error"]
    else:
        iguales = (consulta["archivos"] == FIGSHARE_INSTANTANEA["archivos"]
                   and consulta["doi"] == FIGSHARE_INSTANTANEA["doi"]
                   and consulta["published_date"] == FIGSHARE_INSTANTANEA["published_date"])
        consulta_resumen = (f"consultada el {consulta['fecha_consulta']}: "
                            + ("coincide con la instantánea (DOI, fecha y MD5 de los 3 ZIP)" if iguales
                               else "DIFIERE de la instantánea: revisar «consulta_en_vivo»"))
    hallados = [n for n, z in zips.items() if z["encontrado"]]
    if hallados:
        cotejos = []
        for n in hallados:
            z = zips[n]
            md5_txt = ("coincide" if z.get("md5_coincide_con_figshare") else
                       ("NO coincide" if "md5_coincide_con_figshare" in z else "no calculado"))
            cotejos.append(f"{n}: bytes {'coinciden' if z.get('bytes_coinciden_con_figshare') else 'NO coinciden'}, MD5 {md5_txt}")
        resultado_sumas = ("figshare publica un MD5 por ZIP (IMU.zip, LANDMARKS.zip, VIDEOS.zip). ZIP en disco: "
                           + "; ".join(cotejos) + ".")
    else:
        resultado_sumas = ("figshare publica un MD5 por ZIP (IMU.zip, LANDMARKS.zip, VIDEOS.zip; API consultada el "
                           f"{FIGSHARE_INSTANTANEA['fecha_consulta']}), no por CSV ni por video. Los ZIP descargados NO están en "
                           f"disco (buscados en {[str(p) for p in (args.zips or [DATA_DIR])]} hasta 3 niveles), así que el MD5 no se "
                           "puede cotejar; lo que sí se verifica es que el contenido extraído está completo y es legible (esta "
                           "bitácora). PENDIENTE: si el estudiante conserva los ZIP, correr con --zips RUTA --verificar-md5.")
    bitacora = {
        "titulo": "Bitácora de integridad del corpus LSC50 (actividad 1.a)",
        "corpus": IDENTIDAD_CORPUS,
        "fecha_verificacion": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "obtencion": obtencion,
        "sumas_de_comprobacion": {
            "publicadas_por_figshare": {
                "fuente": f"API pública de figshare, {FIGSHARE_INSTANTANEA['api']}, consultada el {FIGSHARE_INSTANTANEA['fecha_consulta']}",
                "nivel": "un MD5 (computed_md5) por archivo ZIP del depósito (3 archivos); ninguna suma por CSV ni por video",
                "instantanea": FIGSHARE_INSTANTANEA,
                "consulta_en_vivo": consulta if consulta is not None else "no solicitada (--consultar-figshare)",
                "consulta_en_vivo_resumen": consulta_resumen,
            },
            "archivos_de_sumas_en_el_material_descargado": sumas,
            "zips_originales_en_disco": zips,
            "resultado": resultado_sumas + (f" Además, {len(sumas)} archivo(s) de sumas dentro del material descargado: {sumas}."
                                            if sumas else " Dentro del material descargado no viene ningún archivo de sumas "
                                            "(*.md5, *.sha1, *.sha256, *checksum*, MANIFEST*, *.sfv)."),
        },
        "metodo": {
            "csv": "lectura completa con numpy.loadtxt (C); filas = líneas de datos sin cabecera; NaN y filas todo-cero sobre las columnas de landmarks (sin la columna índice); SHA-1 del archivo completo",
            "video": f"cabecera AVI leída con struct (primeros {BYTES_CABECERA_AVI} bytes: RIFF, avih, strh, strf, dmlh); ningún fotograma decodificado; SHA-1 de los primeros {BYTES_HASH_VIDEO} bytes para la heurística de duplicados",
            "rejilla": "SSSS_VVVV_RRRR con S∈[0,49], V∈[0,4], R∈[0,3] (1000 nombres); ausentes = nombres de la rejilla sin archivo; fuera de rejilla = archivos cuyo nombre no está en ella",
        },
        "entorno": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "plataforma": platform.platform(),
            "maquina": platform.machine(),
            "duracion_s": round(time.perf_counter() - t0, 1),
            "con_hash": con_hash,
            "limite": args.limite,
        },
        "agregados": agregados,
        "comprobaciones_N02": comprob,
        "muestras": registros,
    }
    with open(args.salida / "bitacora_integridad.json", "w", encoding="utf-8") as f:
        json.dump(bitacora, f, indent=1, ensure_ascii=False)
    escribir_csv_plano(registros, args.salida / "bitacora_integridad.csv")

    car = caracterizar(registros, agregados, metadata, raw, completo, args.particiones)
    car["fecha_generacion"] = bitacora["fecha_verificacion"]
    car["muestras_verificadas"] = len(registros)
    with open(args.salida / "caracterizacion.json", "w", encoding="utf-8") as f:
        json.dump(car, f, indent=1, ensure_ascii=False)
    escribir_md(car, bitacora, args.salida / "caracterizacion.md")

    # Resumen por consola
    ag = agregados
    fb = ag["fotogramas_COLOR_BODY_frente_a_filas_body"]
    print("\n=== RESUMEN ===")
    print(f"Muestras verificadas: {len(registros)} | ausentes de la rejilla: {len(ag['muestras_ausentes_de_la_rejilla'])} | "
          f"fuera de rejilla: {len(ag['nombres_fuera_de_la_rejilla'])} | archivos ausentes: {len(ag['archivos_ausentes'])} | "
          f"ilegibles: {len(ag['muestras_con_archivo_ilegible'])}")
    print(f"Anomalías: {ag['recuento_anomalias']}")
    print(f"COLOR_BODY vs body: más={fb['video_mas_fotogramas_que_filas']} igual={fb['video_igual_filas']} "
          f"menos={fb['video_menos_fotogramas_que_filas']} | dif positivas: {fb['diferencia_sobre_las_positivas']}")
    print(f"body vs manos distinta: {ag['body_frente_a_manos']['muestras_con_longitud_distinta']} "
          f"(histograma {ag['body_frente_a_manos']['histograma_max_menos_min']})")
    print(f"Filas todo-cero / NaN por modalidad: "
          + ", ".join(f"{k}: {v['filas_todo_cero']}/{v['nan']}" for k, v in ag["totales_por_modalidad"].items()))
    print(f"Índice consecutivo en todos los CSV: {ag['csv_indice_consecutivo_en_todas']}")
    for cam, m in ag["modas_por_camara"].items():
        print(f"{cam}: {m['resolucion']} @ {m['fps']} fps, códec {m['codec']} | fps observados {m['fps_valores']} | resoluciones {m['resolucion_valores']}")
    if isinstance(ag["duplicados"], dict):
        print(f"Duplicados confirmados (grupos por carpeta): { {k: len(v) for k, v in ag['duplicados'].items()} }")
        de = car["duplicados_exactos"]
        for g in de["grupos"]:
            print(f"  {' = '.join(g['muestras'])}: {', '.join(g['modalidades'])}")
        print(f"  Muestras únicas para el entrenamiento (COLOR_BODY/body/lh/rh): {de['muestras_unicas_en_COLOR_BODY_body_lh_rh']}")
        for esquema, r in de["cotejo_con_particiones"].items():
            h = r.get("grupos_repartidos_entre_conjuntos", [])
            print(f"  Partición {esquema}: {len(h)} grupo(s) repartidos entre conjuntos" + (f" → {h}" if h else ""))
    for c, k in car["camaras"]["cotejo_declarado_vs_medido"].items():
        print(f"Cámara {c}: declarado {k['fps_declarado']:g} fps {k['resolucion_declarada']} | medido {k['fps_medido_moda']} fps "
              f"{k['resolucion_medida_moda']} → {'coincide' if k['fps_coincide'] and k['resolucion_coincide'] else 'NO coincide'}")
    print(f"Sumas de comprobación: {bitacora['sumas_de_comprobacion']['resultado']}")
    if comprob.get("aplicable"):
        print(f"Comprobaciones N-02: {'TODAS OK' if comprob['todo_ok'] else 'DISCREPANCIAS'} → "
              + ", ".join(f"{k}={v[0]} (esp. {v[1]})" for k, v in comprob.items() if isinstance(v, tuple)))
        for m, d in comprob["muestras_detalle"].items():
            print(f"  {m}: {d.get('obtenido')} ok={d.get('ok')}")
    else:
        print(f"Comprobaciones N-02: {comprob.get('motivo')}")
    print(f"Salida: {args.salida}  ({bitacora['entorno']['duracion_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
