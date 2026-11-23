"""
Núcleo del prototipo LSC → español hablado.

Integra la cadena completa que describe el anteproyecto (§2.2.5-§2.2.7):

  cámara → LandmarkExtractor (MediaPipe Tasks HolisticLandmarker: pose + manos + rostro;
                              225 rasgos + rostro aparte)
         → normalización por hombros → columnas de la representación del modelo (D-08)
         → remuestreo a 30 pasos (resample_sequence)
         → velocidades si el modelo las espera (add_velocity)
         → modelo (por defecto el final, TCN + imputación + 252, D-14; la red heredada CNN+BiLSTM
           sigue disponible con model_path) → conversor de glosas → TTS

Principios de diseño (todos exigidos por el anteproyecto):

  * UN SOLO EXTRACTOR PARA CORPUS Y PROTOTIPO (D-05). El extractor del proyecto es
    mp.tasks.vision.HolisticLandmarker (mediapipe 0.10.35, modelo
    models/mediapipe/holistic_landmarker.task, delegado CPU, modo VIDEO) con las
    opciones OPCIONES_HOLISTIC definidas abajo. La re-extracción del corpus
    (src/reextract_lsc50.py) usa exactamente el mismo modelo, las mismas opciones y el
    mismo delegado; tests/test_prototipo.py compara ambos diccionarios. Los 225 rasgos
    que esperan los modelos actuales conservan el orden pose (33x3), mano izquierda
    (21x3), mano derecha (21x3), con ceros sin detección. El rostro (468x3, sin los 10
    puntos de iris) se devuelve aparte en Extraccion.rostro; solo entra en la inferencia
    (cejas y labios) si la representación del modelo lo usa o si su regla es «descartar»
    sobre un processed con rostro (D-08).

  * VENTANA FIJA ACTIVADA POR LA PERSONA USUARIA (§2.2.6). La inferencia no es
    continua: se abre una ventana con una tecla (eso sella t0), la ventana captura
    N fotogramas (o se cierra por tope de tiempo), y al cerrarse se infiere UNA vez.
    Fuera de una ventana no se infiere nunca.

  * RASGOS COHERENTES CON EL MODELO (D-08, N-30). Junto a cada modelo entrenado desde el
    15-sep-2026 hay un archivo <modelo>.procedencia.json (src/representacion.py) que declara
    su representación de entrada («manos», «manos_postura» o «manos_postura_rostro»), si usa
    velocidades y la regla de mano ausente del corpus con que se entrenó. El prototipo lo lee
    y construye exactamente esa entrada: fila completa (postura + manos [+ cejas y labios]),
    regla sin postura, regla de ausencia, normalización por hombros, máscara, columnas de la
    representación, resample_sequence y add_velocity del entrenamiento. Sin procedencia
    (modelos anteriores) la representación se deduce del ancho de model.input_shape y se avisa.

  * INSTRUMENTACIÓN (§2.2.7). Un único reloj monótono (time.perf_counter_ns). Por
    ensayo se sellan t0 (tecla), c_i (llegada de cada fotograma), e_i (retorno del
    extractor), t1 = c_N, t2 (fin del preprocesamiento), t3 (retorno del modelo),
    t4 (retorno de GrammarMapper) y t5 (emisión de la orden de reproducción del
    audio, no el instante en que suena). Seis etapas consecutivas que suman exacto
    L_total = t5 - t0; L_resp = t5 - t1; W = t1 - t0. Un registro por ensayo en JSONL.

  * ESTADOS DE ERROR (§2.2.6): confianza bajo el corte → «seña no reconocida», sin
    audio; manos (o pose) no detectadas en una proporción de la ventana mayor que el
    máximo → se indica y NO se infiere; ventana abortada o con menos fotogramas que
    el mínimo → descartada. Cada estado se cuenta y queda en el JSONL.

  * SEÑA CORTADA EN LOS BORDES (N-50, opción B): protocolo de reposo al abrir y al cerrar la
    ventana, X para abortar si algo falla y, en cada ventana cerrada, un AVISO en el JSONL
    (clave «bordes»: aviso_inicio, aviso_fin y los valores medidos) que NO rechaza la seña ni
    cambia su estado. Medida y umbrales de src/analisis_senas_cortadas.py (src/bordes_ventana.py).

  * SIN ALMACENAR VIDEO (§1.5): ningún fotograma se escribe a disco. El JSONL guarda
    marcas de tiempo, clase, confianza y contadores; no guarda imágenes ni puntos.

Uso programático (sin cámara; ver tests/test_prototipo.py):

    from src.translator import LSCTranslator
    tr = LSCTranslator(model_path="models/best_model.keras", tts_mode="none")
    tr.abrir_ventana()                       # sella t0
    for frame_rgb in fotogramas:
        res = tr.procesar_fotograma(frame_rgb)
        if res.registro is not None:         # la ventana se cerró: hay un ensayo
            print(res.registro["estado"], res.registro["clase"])
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.grammar_mapper import GrammarMapper, SIGN_VOCABULARY
# El conversor por reglas de la actividad 3.a (src/conversor_glosas.py, docs/CONVERSOR_3a_ESPECIFICACION.md)
# NO se importa aquí: se importa solo en construir_conversor() con «reglas» o «inventario». Así el
# conversor por defecto (GrammarMapper, «heredado») no depende de que ese módulo cargue: sus
# comprobaciones al importar (RuntimeError por léxico o inventario) solo afectan a quien lo elige.
# Las MISMAS funciones que usa el entrenamiento (D-02: arquitectura canónica).
from src.train_final_100 import add_velocity, resample_sequence
# Regla «imputacion» (N-63): la MISMA función con que se construyó la entrada del entrenamiento.
from src.mano_ausente_variantes import imputar_fotogramas
from src.representacion import (  # D-08: representación de entrada y procedencia del modelo
    REPRESENTACIONES, ROSTRO_SUBCONJUNTOS, columnas as columnas_representacion, leer_procedencia,
    n_rasgos as n_rasgos_representacion, ruta_procedencia, leer_umbral,
)
# Aviso de seña cortada en los bordes de la ventana (N-50, opción B): la medida de
# src/analisis_senas_cortadas.py y sus umbrales candidatos; solo avisa, no rechaza.
from src.bordes_ventana import (  # noqa: E402
    MODOS_BORDES, UMBRAL_BORDES_CANDIDATO, cargar_umbrales_bordes, evaluar_bordes, medir_bordes,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

N_POSE = 33
N_HAND = 21
N_FACE = 468          # puntos del rostro que se conservan: HolisticLandmarker entrega 478 y los
                      # 10 últimos (468-477) son iris; se descartan para igualar la malla de 468.
COORDS = 3
N_FEATURES = (N_POSE + 2 * N_HAND) * COORDS   # 225 = pose(99) + mano izq(63) + mano der(63)
N_FACE_FEATURES = N_FACE * COORDS             # 1404
LEFT_SHOULDER_IDX = 11 * COORDS               # 33
RIGHT_SHOULDER_IDX = 12 * COORDS              # 36

# Posición de cada bloque dentro de los 225 rasgos (orden fijo: pose, izquierda, derecha).
IDX_POSE = slice(0, N_POSE * COORDS)                                        # 0:99
IDX_MANO_IZQ = slice(N_POSE * COORDS, (N_POSE + N_HAND) * COORDS)           # 99:162
IDX_MANO_DER = slice((N_POSE + N_HAND) * COORDS, N_FEATURES)                # 162:225

# D-07 (14-sep-2026): tratamiento del fotograma en el que el extractor no detectó una mano (su bloque
# llega en ceros). Debe ser EXACTAMENTE el mismo que aplicó preprocess_lsc50.py --mano-ausente al corpus
# con el que se entrenó el modelo. «mascara» es la variante elegida por validación (N-23):
#   ceros      conservar y normalizar (la mano ausente queda como un vector constante -ref/escala);
#   descartar  quitar el fotograma de la ventana si falta cualquiera de las dos manos;
#   mascara    normalizar y DESPUÉS poner en ceros el bloque de la mano ausente.
#   imputacion (3-oct-2026, opción EXPERIMENTAL, N-63; pendiente de la decisión del director) «mascara» y,
#              sobre los fotogramas de la ventana ya en la representación «manos», rellenar por mano la
#              mano ausente interpolando entre los fotogramas válidos (mano_ausente_variantes.
#              imputar_fotogramas) ANTES de remuestrear; mano ausente en toda la ventana = 0. Es la
#              regla con que src/train_final_tcn.py entrenó models/tcn_imputacion/modelo.keras.
MANO_AUSENTE_OPCIONES = ("ceros", "descartar", "mascara", "imputacion")

# Representación deducida del ancho de entrada cuando el modelo no trae procedencia (modelos anteriores
# al 15-sep-2026). Los anchos de las tres representaciones no coinciden entre sí, con o sin velocidades.
REPRESENTACION_POR_ANCHO = {n_rasgos_representacion(r) * k: (r, k == 2)
                            for r in REPRESENTACIONES for k in (1, 2)}

# Modelo por defecto del prototipo (decisión del autor del 22-nov-2026, a partir de D-14): el modelo final,
# TCN + imputación + 252 (N-65; representación «manos», regla «imputacion» y umbral en modelo.umbral.json).
# La red heredada (canónico CNN+BiLSTM «manos» con «mascara», N-30/N-31) sigue disponible con --model-path.
MODELO_FINAL = ROOT_DIR / "models" / "tcn_imputacion" / "modelo.keras"
MODELO_HEREDADO = ROOT_DIR / "models" / "canonico_holistic_mascara_manos" / "modelo_canonico.keras"
MODELO_POR_DEFECTO = MODELO_FINAL


def _ruta_para_el_registro(ruta) -> str:
    """Ruta del modelo tal como se guarda en el JSONL. D-12 versiona los registros, así que se escribe
    relativa a la raíz del repositorio cuando está dentro: la ruta absoluta lleva el nombre de la
    cuenta del equipo (en este Mac, la carpeta de Google Drive) y no aporta nada al reporte. Si el
    modelo está fuera del repositorio se guarda tal cual, porque ahí la ruta sí informa."""
    if ruta is None:
        return None
    try:
        return str(Path(ruta).resolve().relative_to(ROOT_DIR))
    except (ValueError, OSError):
        return str(ruta)

FPS_SUPUESTA = 30.0   # solo si aún no se ha podido medir la tasa real del bucle

# Conversores de glosas que el prototipo puede usar (3.a, §7.2 de docs/CONVERSOR_3a_ESPECIFICACION.md):
#   heredado    GrammarMapper (src/grammar_mapper.py), el de siempre;
#   reglas      ConversorGlosas(): reglas explícitas y plantillas, con respaldo marcado;
#   inventario  ConversorGlosas(estricto=True): solo dice las glosas solas y las secuencias
#               previstas; todo lo demás va a respaldo («fuera_de_inventario»). Es el procedimiento de
#               conversión de la tesis (D-15) y el valor por defecto de src/realtime.py (--conversor)
#               desde el 22-nov-2026 (CONVERSOR_POR_DEFECTO).
# LSCTranslator, cuando no se le inyecta ningún conversor (grammar_mapper=None), sigue construyendo
# GrammarMapper: es la interfaz programática que usan las pruebas y los scripts fuera de línea, que pasan el
# conversor de forma explícita. El defecto del prototipo en vivo lo fija realtime.py con CONVERSOR_POR_DEFECTO.
CONVERSORES = ("heredado", "reglas", "inventario")
# Fuentes de audio que el registro acepta de un motor de voz (src/tts.py, FUENTES).
FUENTES_VOZ = ("pregrabada", "xtts", "local", "edge", "gtts")
CONVERSOR_POR_DEFECTO = "inventario"


def construir_conversor(nombre: str = "heredado", genero_primera: str | None = None,
                        genero_segunda: str | None = None):
    """Conversor de glosas para pasar como grammar_mapper= a LSCTranslator.

    «heredado» devuelve None, y LSCTranslator construye GrammarMapper() como siempre. Los géneros solo
    tienen sentido con «reglas» o «inventario»: con «heredado» se rechazan (ValueError) para que nadie
    crea que se aplicaron."""
    if nombre not in CONVERSORES:
        raise ValueError(f"conversor debe ser uno de {CONVERSORES}, no {nombre!r}")
    if nombre == "heredado":
        if genero_primera is not None or genero_segunda is not None:
            raise ValueError("--genero-primera y --genero-segunda no tienen efecto con el conversor heredado")
        return None
    from src.conversor_glosas import ConversorGlosas   # import perezoso: solo con reglas/inventario
    return ConversorGlosas(genero_primera=genero_primera, genero_segunda=genero_segunda,
                           estricto=(nombre == "inventario"))


def _es_conversor_glosas(conversor) -> bool:
    """True si «conversor» es un ConversorGlosas (3.a), sin forzar el import de src.conversor_glosas.

    Si el módulo no está cargado, nadie ha podido construir un ConversorGlosas y la respuesta es False.
    Se usa isinstance y no getattr(…, "explicar"): un MagicMock tiene cualquier atributo."""
    modulo = sys.modules.get("src.conversor_glosas")
    clase = getattr(modulo, "ConversorGlosas", None) if modulo is not None else None
    return clase is not None and isinstance(conversor, clase)

ESTADOS = ("aceptada", "rechazada_confianza", "rechazada_manos", "rechazada_incompleta")
ETAPAS = ("captura", "extraccion", "preprocesamiento", "inferencia", "conversion", "sintesis")

MODELS_DIR = ROOT_DIR / "models" / "mediapipe"
# D-05: el extractor único del proyecto es HolisticLandmarker. Mismo archivo .task, MISMAS
# opciones (OPCIONES_HOLISTIC) y mismo delegado CPU que src/reextract_lsc50.py, que re-extrae
# el corpus; tests/test_prototipo.py comprueba que ambos diccionarios son idénticos.
HOLISTIC_TASK = "holistic_landmarker.task"
MODEL_URLS = {
    HOLISTIC_TASK: "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
}
# SHA256 del archivo verificado el 2026-09-14 (13 683 609 bytes); sirve para comprobar que el
# corpus y el prototipo usan exactamente el mismo modelo.
MODEL_SHA256 = {
    HOLISTIC_TASK: "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8",
}
OPCIONES_HOLISTIC = dict(min_face_detection_confidence=0.5, min_face_suppression_threshold=0.5,
                         min_face_landmarks_confidence=0.5, min_pose_detection_confidence=0.5,
                         min_pose_suppression_threshold=0.5, min_pose_landmarks_confidence=0.5,
                         min_hand_landmarks_confidence=0.5, output_face_blendshapes=False,
                         output_segmentation_mask=False)


# ---------------------------------------------------------------------------
# Extracción de puntos característicos
# ---------------------------------------------------------------------------

@dataclass
class Extraccion:
    """Resultado del extractor para UN fotograma."""
    keypoints: np.ndarray            # (225,) float32: pose, mano izq, mano der; ceros sin detección
    pose_detectada: bool
    n_manos: int                     # 0, 1 o 2 manos detectadas
    # (1404,) float32 = 468 puntos x 3 (sin los 10 de iris); None si no hubo rostro. Solo entra en la
    # inferencia (cejas y labios, D-08) si la representación del modelo o su regla «descartar» lo piden.
    rostro: np.ndarray | None = None
    holistic_result: object = None   # HolisticLandmarkerResult crudo (solo para dibujar)


def _ensure_models_downloaded() -> dict[str, str]:
    """Descarga los .task de MediaPipe si no existen y devuelve sus rutas. Comprueba el
    SHA256 contra MODEL_SHA256 (el archivo fijado en D-05) y AVISA si no coincide: el corpus
    se re-extrae con ese archivo exacto y el prototipo debe usar el mismo. La URL apunta a
    «latest», así que una descarga nueva podría no ser ese archivo; por eso se comprueba."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for filename, url in MODEL_URLS.items():
        filepath = MODELS_DIR / filename
        if not filepath.exists():
            print(f"  Descargando {filename}...")
            urllib.request.urlretrieve(url, str(filepath))
        esperado = MODEL_SHA256.get(filename)
        if esperado:
            sha = hashlib.sha256(filepath.read_bytes()).hexdigest()
            if sha != esperado:
                print(f"  [AVISO] {filename}: SHA256 {sha[:12]}… no es el fijado en D-05 ({esperado[:12]}…); "
                      f"el corpus se re-extrae con ese modelo exacto (src/reextract_lsc50.py)")
        paths[filename] = str(filepath)
    return paths


def _aplanar(landmarks, n: int, nombre: str) -> np.ndarray:
    """(x, y, z) de los primeros n puntos como vector float32 de n*3. Exige que haya al
    menos n puntos: una lista más corta delataría un cambio en el modelo, no un fotograma
    sin detección (esos llegan como lista vacía y se tratan antes de llamar aquí)."""
    if len(landmarks) < n:
        raise ValueError(f"HolisticLandmarker devolvió {len(landmarks)} puntos de {nombre}; se esperaban {n}")
    out = np.empty((n, COORDS), dtype=np.float32)
    for k in range(n):
        lm = landmarks[k]
        out[k, 0] = lm.x
        out[k, 1] = lm.y
        out[k, 2] = lm.z
    return out.reshape(-1)


class LandmarkExtractor:
    """Extractor único del proyecto (D-05): mp.tasks.vision.HolisticLandmarker en modo
    VIDEO con delegado CPU y OPCIONES_HOLISTIC, idéntico al de src/reextract_lsc50.py.

    De cada fotograma produce los 225 rasgos que esperan los modelos actuales, en el
    orden pose (33x3), mano izquierda (21x3), mano derecha (21x3), con ceros donde no
    hubo detección, y aparte el rostro (468x3) o None. `left_hand_landmarks` es la mano
    izquierda de la persona (verificado contra los CSV publicados del corpus).
    """

    descripcion = f"HolisticLandmarker 0.10.35, CPU, {N_FEATURES} rasgos + rostro"

    def __init__(self, model_paths: dict[str, str] | None = None, reloj=time.perf_counter_ns,
                 landmarker=None):
        """
        Args:
            model_paths: rutas de los .task ({HOLISTIC_TASK: ruta}); None → descarga si falta.
            reloj: reloj monótono en ns del que salen las marcas de tiempo del modo VIDEO.
            landmarker: objeto con detect_for_video(image, ts_ms) y close(); las pruebas
                inyectan aquí un HolisticLandmarker falso. None → se crea el real.
        """
        import mediapipe as mp   # importación perezosa: las pruebas pueden inyectar un extractor falso
        self._mp = mp
        # La descripción declara la versión realmente cargada (queda en cada registro JSONL).
        self.descripcion = f"HolisticLandmarker {mp.__version__}, CPU, {N_FEATURES} rasgos + rostro"

        if landmarker is None:
            if model_paths is None:
                model_paths = _ensure_models_downloaded()
            BaseOptions = mp.tasks.BaseOptions
            landmarker = mp.tasks.vision.HolisticLandmarker.create_from_options(
                mp.tasks.vision.HolisticLandmarkerOptions(
                    # Delegado CPU explícito, como en reextract_lsc50.py (§1.4: íntegramente en CPU).
                    base_options=BaseOptions(model_asset_path=model_paths[HOLISTIC_TASK],
                                             delegate=BaseOptions.Delegate.CPU),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    **OPCIONES_HOLISTIC))
        self.landmarker = landmarker

        # El modo VIDEO exige marcas de tiempo estrictamente crecientes: se toman del reloj
        # real y, si dos fotogramas caen en el mismo ms, se fuerza +1.
        self._reloj = reloj
        self._t_inicio = reloj()
        self._ultimo_ts_ms = -1

    def _siguiente_timestamp_ms(self) -> int:
        ts = int((self._reloj() - self._t_inicio) // 1_000_000)
        if ts <= self._ultimo_ts_ms:
            ts = self._ultimo_ts_ms + 1
        self._ultimo_ts_ms = ts
        return ts

    def extract(self, frame_rgb: np.ndarray) -> Extraccion:
        """Extrae 225 rasgos (+ rostro aparte) de un fotograma RGB (uint8, HxWx3)."""
        mp = self._mp
        ts_ms = self._siguiente_timestamp_ms()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        res = self.landmarker.detect_for_video(mp_image, ts_ms)

        # HolisticLandmarkerResult: listas planas de 33 / 21 / 21 / 478 NormalizedLandmark,
        # vacías cuando no hay detección.
        keypoints = np.zeros(N_FEATURES, dtype=np.float32)
        pose_detectada = bool(res.pose_landmarks)
        if pose_detectada:
            keypoints[IDX_POSE] = _aplanar(res.pose_landmarks, N_POSE, "pose")
        n_manos = 0
        if res.left_hand_landmarks:
            keypoints[IDX_MANO_IZQ] = _aplanar(res.left_hand_landmarks, N_HAND, "mano izquierda")
            n_manos += 1
        if res.right_hand_landmarks:
            keypoints[IDX_MANO_DER] = _aplanar(res.right_hand_landmarks, N_HAND, "mano derecha")
            n_manos += 1
        rostro = _aplanar(res.face_landmarks, N_FACE, "rostro") if res.face_landmarks else None

        return Extraccion(
            keypoints=keypoints,
            pose_detectada=pose_detectada,
            n_manos=n_manos,
            rostro=rostro,
            holistic_result=res,
        )

    def close(self):
        self.landmarker.close()


# ---------------------------------------------------------------------------
# Normalización (idéntica a preprocess_lsc50.normalize_sample_vectorized)
# ---------------------------------------------------------------------------

def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Centra cada fotograma en el punto medio de los hombros y lo escala por la
    distancia entre hombros en el plano x-y. Misma fórmula que el preprocesamiento
    del corpus (preprocess_lsc50.py:308-341)."""
    T = sequence.shape[0]
    normalized = sequence.astype(np.float32, copy=True)

    ls_x = normalized[:, LEFT_SHOULDER_IDX]
    ls_y = normalized[:, LEFT_SHOULDER_IDX + 1]
    ls_z = normalized[:, LEFT_SHOULDER_IDX + 2]
    rs_x = normalized[:, RIGHT_SHOULDER_IDX]
    rs_y = normalized[:, RIGHT_SHOULDER_IDX + 1]
    rs_z = normalized[:, RIGHT_SHOULDER_IDX + 2]

    ref_x = (ls_x + rs_x) / 2.0
    ref_y = (ls_y + rs_y) / 2.0
    ref_z = (ls_z + rs_z) / 2.0

    scale = np.sqrt((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2)
    scale = np.maximum(scale, 1e-6)

    reshaped = normalized.reshape(T, -1, COORDS)
    ref = np.stack([ref_x, ref_y, ref_z], axis=1)[:, np.newaxis, :]
    reshaped -= ref
    reshaped /= scale[:, np.newaxis, np.newaxis]
    return reshaped.reshape(T, -1)


# ---------------------------------------------------------------------------
# Ensayo: todas las marcas de un intento de reconocimiento
# ---------------------------------------------------------------------------

@dataclass
class Ensayo:
    id_ensayo: int
    modo: str                       # "aislado" u "oracion"
    t0: int                         # tecla que abre la ventana
    ventana_frames: int             # N objetivo
    ventana_max_ns: int             # tope de seguridad por tiempo
    fps_medida: float | None        # tasa del bucle medida antes de abrir
    c: list[int] = field(default_factory=list)       # llegada de cada fotograma
    e: list[int] = field(default_factory=list)       # retorno del extractor
    keypoints: list[np.ndarray] = field(default_factory=list)
    rostros: list[np.ndarray | None] = field(default_factory=list)   # 1404 valores o None por fotograma
    pose_flags: list[bool] = field(default_factory=list)
    manos_counts: list[int] = field(default_factory=list)
    t1: int | None = None
    t2: int | None = None
    t3: int | None = None
    t4: int | None = None
    t5: int | None = None
    cierre: str | None = None       # "frames" | "tiempo" | "abortada"
    estado: str | None = None
    motivo: str | None = None
    id_clase: int | None = None
    clase: str | None = None
    confianza: float | None = None
    top3: list[dict] | None = None
    enunciado: str | None = None
    # True si el audio estaba sintetizado cuando se emitió la orden de reproducción (t5), como
    # exige §2.2.7; False si el motor sintetiza y reproduce en el mismo acto (pyttsx3), en cuyo
    # caso la etapa «síntesis» solo mide la emisión de la orden. None si no hubo audio.
    tts_sintesis_antes_de_t5: bool | None = None
    tts_ocupado_en_t5: bool | None = None   # True: otra locución sonaba y la reproducción espera su turno
    # True si el audio salió de la cache del motor (no se sintetizó en este ensayo, así que la etapa
    # «síntesis» vale casi cero); False si se sintetizó ahora; None si no hubo audio o el motor no
    # tiene cache ni informa de ella (16-sep-2026, pendiente de N-34).
    tts_sintesis_desde_cache: bool | None = None
    # Qué produjo el audio (23-nov-2026; src/tts.py, docs/VOZ.md): «pregrabada», «xtts» o «local» (o «edge»/
    # «gtts»), y cuánto tardó el motor en tenerlo listo desde prepare() (ns). None si no hubo audio o si el motor
    # no lo informa; entonces las claves tts_fuente / tts_tiempo_sintesis_ns NO se escriben en el JSONL.
    tts_fuente: str | None = None
    tts_tiempo_sintesis_ns: int | None = None
    # Traza de ConversorGlosas.explicar sin «frase» (3.a, §6). Solo con el conversor por reglas; con el
    # heredado queda en None y la clave «conversion» no se escribe en el JSONL.
    conversion: dict | None = None
    # Aviso de bordes (N-50, opción B): medidas, banderas aviso_inicio / aviso_fin y costo del cálculo. Se
    # calcula DESPUÉS de fijar el estado (y de t5): no cambia el estado, ni la inferencia, ni ninguna marca.
    bordes: dict | None = None

    @property
    def n(self) -> int:
        return len(self.c)


@dataclass
class ResultadoFotograma:
    """Lo que devuelve procesar_fotograma para que la interfaz dibuje y decida."""
    extraccion: Extraccion
    c: int
    e: int
    en_ventana: bool
    progreso: float                 # n/N de la ventana (0 si no hay ventana)
    registro: dict | None = None    # registro del ensayo si la ventana se cerró en este fotograma


# ---------------------------------------------------------------------------
# Traductor
# ---------------------------------------------------------------------------

def _estadisticas_ns(valores: list[int]) -> dict | None:
    if not valores:
        return None
    a = np.asarray(valores, dtype=np.float64)
    return {
        "n": int(a.size),
        "media": float(a.mean()),
        "mediana": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


class LSCTranslator:
    """Cadena completa con ventana fija por tecla, estados de error e instrumentación.

    Todos los componentes son inyectables (modelo, extractor, grammar_mapper, tts,
    reloj) para poder probar el pipeline sin cámara, sin MediaPipe y sin TensorFlow
    real. Con los valores por defecto se cargan los componentes reales.
    """

    def __init__(
        self,
        model_path: str | None = None,
        tts_mode: str = "none",
        tts_voice: str | None = None,
        umbral: float | None = None,
        ventana_frames: int = 0,
        ventana_seg: float = 2.5,
        ventana_max_seg: float | None = None,
        min_frames: int = 15,
        max_sin_manos: float = 0.5,
        modo_oracion: bool = False,
        registro_dir: str | Path | None = None,
        registro: bool = True,
        modelo=None,
        extractor=None,
        grammar_mapper=None,
        tts=None,
        reloj=time.perf_counter_ns,
        calentar: bool = True,
        tts_espera_max_seg: float = 15.0,
        tts_precalentar: bool = True,
        mano_ausente: str | None = None,
        representacion: str | None = None,
        procedencia: dict | None = None,
        bordes: str = "aviso",
        umbral_bordes: str | Path | None = None,
    ):
        """
        Args:
            model_path: ruta al .keras. None → MODELO_POR_DEFECTO (el modelo final, TCN + imputación;
                la red heredada es MODELO_HEREDADO). Si junto al
                modelo hay <modelo>.procedencia.json, de ahí salen la representación y la regla de
                mano ausente.
            tts_mode: "offline" (local, prioridad del anteproyecto), "edge", "gtts" o "none".
            umbral: confianza mínima para aceptar (parámetro de operación, no meta; §2.2.6). None → el
                elegido por validación para el modelo (<modelo>.umbral.json, D-09) o, sin él, 0,5.
            ventana_frames: N fotogramas que cierran la ventana. 0 → automático:
                round(ventana_seg × tasa medida del bucle) al abrir la ventana.
            ventana_seg: duración objetivo de la ventana (s). Por defecto 2,5 s: la
                duración de los clips COMPLETOS de LSC50 (data/processed/lengths.npy, 1000
                muestras a 24 fps, con reposo al inicio y al final) es media 61,7 fotogramas
                = 2,57 s, mediana 62 = 2,58 s, p90 79 = 3,29 s, máximo 134 = 5,58 s; el 47 %
                de los clips dura ≤ 2,5 s y el 79 % ≤ 3,0 s. La ventana se remuestrea a 30
                pasos igual que el clip completo en el entrenamiento, así que su duración
                debe ser comparable a la del clip. --ventana-seg 3.0 es la alternativa
                natural a ensayar en el Bloque 2; el valor efectivo W se reporta siempre.
            ventana_max_seg: tope de seguridad; la ventana se cierra por tiempo aunque no
                hayan llegado N fotogramas. None → 2 × ventana_seg.
            min_frames: por debajo de este número la ventana se descarta (incompleta).
            max_sin_manos: proporción máxima de fotogramas de la ventana sin manos (o sin
                pose) detectadas; si se supera, se indica y NO se infiere.
            modo_oracion: True → cada ventana aceptada acumula su glosa; la conversión y
                la síntesis ocurren en fin_de_turno().
            registro_dir: carpeta del JSONL. None → <repo>/registros/.
            registro: False → no escribe JSONL (solo memoria).
            modelo / extractor / grammar_mapper / tts / reloj: inyección para pruebas.
            calentar: hace una inferencia de calentamiento al arrancar para que el primer
                ensayo no pague el trazado de TensorFlow.
            tts_precalentar: al arrancar, sintetizar en segundo plano las 50 glosas sueltas ya
                convertidas a enunciado para que su síntesis no cueste nada en la sesión (solo si el
                motor tiene cache; ver src/tts.py). False mide siempre la síntesis completa.
            tts_espera_max_seg: tope de espera a que el motor en línea sintetice antes de
                emitir la orden de reproducción.
            mano_ausente: regla de D-07. None → la del corpus del modelo (procedencia); si se da y
                difiere, se usa la dada, se avisa y ambas quedan en el JSONL.
            representacion: None → la de la procedencia o, sin ella, la deducida del ancho. Si se da,
                debe coincidir con el ancho del modelo.
            procedencia: inyección para pruebas (en uso normal se lee junto al model_path).
            bordes: "aviso" (por defecto; N-50, opción B) mide en cada ventana cerrada si la seña pudo
                empezar antes o terminar después de la ventana (src/bordes_ventana.py) y lo escribe en el
                JSONL (clave «bordes»: aviso_inicio, aviso_fin, valores medidos y costo del cálculo) SIN
                cambiar el estado ni impedir la inferencia; "off" no calcula nada y registra {"modo": "off"}.
            umbral_bordes: archivo de umbrales del aviso. None → el candidato de N-50
                (models/senas_cortadas_2026-09-22/umbral_bordes_candidato.json).
        """
        if bordes not in MODOS_BORDES:
            raise ValueError(f"bordes debe ser uno de {MODOS_BORDES}, no {bordes!r}")
        self.bordes_modo = bordes
        self.umbrales_bordes = (cargar_umbrales_bordes(umbral_bordes if umbral_bordes is not None
                                                       else UMBRAL_BORDES_CANDIDATO)
                                if bordes == "aviso" else None)
        self._umbrales_bordes_registro = (_ruta_para_el_registro(self.umbrales_bordes["ruta"])
                                          if self.umbrales_bordes is not None else None)
        self.avisos_bordes = {"inicio": 0, "fin": 0}   # en memoria; no entra en «contadores» (ESTADOS)
        self.reloj = reloj
        umbral_param = umbral
        self.ventana_frames = int(ventana_frames)
        self.ventana_seg = float(ventana_seg)
        self.ventana_max_seg = float(ventana_max_seg) if ventana_max_seg is not None else 2.0 * self.ventana_seg
        self.min_frames = int(min_frames)
        self.max_sin_manos = float(max_sin_manos)
        if mano_ausente is not None and mano_ausente not in MANO_AUSENTE_OPCIONES:
            raise ValueError(f"mano_ausente debe ser uno de {MANO_AUSENTE_OPCIONES}, no {mano_ausente!r}")
        self.modo_oracion = bool(modo_oracion)
        self.tts_espera_max_seg = float(tts_espera_max_seg)

        # 1. Modelo: se lee input_shape para decidir si se calculan velocidades.
        procedencia_de_archivo = False
        if modelo is None:
            if model_path is None:
                model_path = str(MODELO_POR_DEFECTO)
            # Revisión D-08: la comprobación vale para cualquier ruta; realtime.py pasa MODELO_POR_DEFECTO
            # de forma explícita y antes solo se comprobaba con model_path=None.
            if not Path(model_path).exists():
                por_defecto = Path(model_path).resolve() == Path(MODELO_POR_DEFECTO).resolve()
                raise FileNotFoundError(f"No existe el modelo {'por defecto ' if por_defecto else ''}{model_path}; "
                                        f"entrénelo (train_final_tcn.py el final, train_final_100.py el heredado) "
                                        f"o indique --model-path")
            print("Cargando modelo...")
            import tensorflow as tf
            modelo = tf.keras.models.load_model(model_path)
            if procedencia is None:
                procedencia = leer_procedencia(model_path)
                procedencia_de_archivo = procedencia is not None
        self.modelo = modelo
        self.model_path = str(model_path) if model_path else getattr(modelo, "name", "inyectado")
        forma = tuple(self.modelo.input_shape)
        self.seq_len = int(forma[1])
        self.n_rasgos = int(forma[-1])
        self.procedencia = procedencia
        # Revisión D-08: la ruta solo se declara si la procedencia se leyó de ese archivo.
        self.procedencia_ruta = (str(ruta_procedencia(model_path)) if procedencia_de_archivo
                                 else "inyectada" if procedencia is not None else None)

        # Representación de entrada (D-08): declarada junto al modelo o deducida del ancho.
        if procedencia is not None:
            rep_modelo = procedencia["representacion"]
            if rep_modelo not in REPRESENTACIONES:
                raise ValueError(f"La procedencia declara la representación {rep_modelo!r}; opciones {list(REPRESENTACIONES)}")
            if int(procedencia["n_rasgos_entrada"]) != self.n_rasgos or int(procedencia["seq_length"]) != self.seq_len:
                raise ValueError(f"La procedencia declara entrada ({procedencia['seq_length']}, "
                                 f"{procedencia['n_rasgos_entrada']}) y el modelo espera {forma}")
            self.usa_velocidades = bool(procedencia.get("velocidades", True))
        else:
            if self.n_rasgos not in REPRESENTACION_POR_ANCHO:
                raise ValueError(f"El modelo espera {self.n_rasgos} rasgos y ninguna representación "
                                 f"({sorted(REPRESENTACION_POR_ANCHO)}) tiene ese ancho. input_shape={forma}")
            rep_modelo, self.usa_velocidades = REPRESENTACION_POR_ANCHO[self.n_rasgos]
            print(f"  [AVISO] el modelo no trae procedencia: representación «{rep_modelo}» deducida del ancho "
                  f"{self.n_rasgos}; la regla de mano ausente no está declarada")
        if representacion is not None and representacion != rep_modelo:
            raise ValueError(f"representacion={representacion!r} no coincide con la del modelo ({rep_modelo!r})")
        self.representacion = rep_modelo
        if n_rasgos_representacion(rep_modelo) * (2 if self.usa_velocidades else 1) != self.n_rasgos:
            raise ValueError(f"La representación {rep_modelo!r} no produce {self.n_rasgos} rasgos")

        # Regla de mano ausente (D-07): la del corpus del modelo salvo que se indique otra.
        regla_modelo = procedencia.get("mano_ausente") if procedencia else None
        if regla_modelo == "no_aplica":
            # Corpus sin fotogramas con mano ausente (CSV publicados): lo más parecido es omitirlos.
            regla_modelo = "descartar"
        if regla_modelo is not None and regla_modelo not in MANO_AUSENTE_OPCIONES:
            raise ValueError(f"La procedencia declara mano_ausente={regla_modelo!r}; opciones {MANO_AUSENTE_OPCIONES}")
        self.mano_ausente_modelo = regla_modelo
        if mano_ausente is None:
            mano_ausente = regla_modelo or "mascara"
        elif regla_modelo is not None and mano_ausente != regla_modelo:
            print(f"  [AVISO] mano_ausente={mano_ausente!r} difiere de la regla del corpus del modelo "
                  f"({regla_modelo!r}); se usa la indicada")
        self.mano_ausente = mano_ausente
        # «imputacion» (N-63) interpola los bloques de 63 valores de cada mano en el diseño de 126 columnas
        # de la representación «manos» (mano_ausente_variantes.BLOQUES_MANOS); con otra representación las
        # columnas no corresponden, así que se rechaza en vez de imputar sobre columnas equivocadas.
        if self.mano_ausente == "imputacion" and rep_modelo != "manos":
            raise ValueError(f"mano_ausente='imputacion' solo está definida para la representación «manos»; "
                             f"el modelo usa {rep_modelo!r}")
        # El bloque facial se arma si la representación lo usa o si, en «descartar», el corpus del
        # modelo omitió también los fotogramas sin rostro (su processed tenía rostro).
        self.rostro_processed = procedencia.get("rostro_del_processed", "ninguno") if procedencia else "ninguno"
        if procedencia is not None and "rostro_del_processed" not in procedencia and self.mano_ausente == "descartar":
            print("  [AVISO] la procedencia no declara rostro_del_processed: se supone «ninguno» (si el processed "
                  "tenía rostro, en «descartar» también se omitían los fotogramas sin rostro)")
        self.arma_rostro = ("rostro" in REPRESENTACIONES[rep_modelo]
                            or (self.mano_ausente == "descartar" and self.rostro_processed != "ninguno"))
        sub = ROSTRO_SUBCONJUNTOS["cejas_labios"]
        self._cols_rostro = np.array([i * COORDS + c for i in sub for c in range(COORDS)], dtype=np.int64)
        self._columnas = columnas_representacion(rep_modelo, "cejas_labios" if self.arma_rostro else "ninguno")

        # Umbral de confianza (D-09): el indicado, el elegido para el modelo o 0,5.
        info_umbral = leer_umbral(model_path) if (umbral_param is None and model_path) else None
        if umbral_param is not None:
            self.umbral, self.umbral_origen = float(umbral_param), "parametro"
        elif info_umbral is not None:
            self.umbral, self.umbral_origen = float(info_umbral["umbral"]), info_umbral["ruta"]
            huella = info_umbral.get("sha256_modelo")
            if huella and Path(model_path).exists():
                import hashlib
                if hashlib.sha256(Path(model_path).read_bytes()).hexdigest() != huella:
                    print(f"  [AVISO] {info_umbral['ruta']} se calculó para otro archivo de modelo "
                          f"(sha256 distinto): el umbral puede no corresponder; recalcúlelo con src/umbral_confianza.py")
                    self.umbral_origen += " (sha256 del modelo no coincide)"
        else:
            self.umbral, self.umbral_origen = 0.5, "defecto_0.5"

        print(f"  Modelo: {self.model_path}")
        print(f"  Umbral de confianza: {self.umbral:.2f} ({self.umbral_origen})")
        print(f"  input_shape={forma} → {self.n_rasgos} rasgos, representación «{self.representacion}», "
              f"velocidades={'sí' if self.usa_velocidades else 'no'}, pasos={self.seq_len}, "
              f"mano ausente «{self.mano_ausente}»"
              + ("" if procedencia is not None else " (sin procedencia)"))
        if self.umbrales_bordes is not None:
            candidato = "CANDIDATO" in str(self.umbrales_bordes.get("estado") or "")
            print(f"  Aviso de bordes (N-50, opción B; avisa, no rechaza): umbrales "
                  f"{'CANDIDATOS ' if candidato else ''}de {self._umbrales_bordes_registro}")
        else:
            print("  Aviso de bordes: desactivado")
        if calentar:
            self.modelo.predict(np.zeros((1, self.seq_len, self.n_rasgos), dtype=np.float32), verbose=0)

        # 2. Extractor
        if extractor is None:
            print("Verificando el modelo de MediaPipe (holistic_landmarker.task)...")
            mp_paths = _ensure_models_downloaded()
            print("Inicializando MediaPipe Tasks (HolisticLandmarker, CPU)...")
            extractor = LandmarkExtractor(mp_paths, reloj=reloj)
        self.extractor = extractor
        self.extractor_desc = getattr(extractor, "descripcion", type(extractor).__name__)

        # 3. Gramática
        self.grammar_mapper = grammar_mapper if grammar_mapper is not None else GrammarMapper()

        # 4. TTS
        self.tts = tts
        self.tts_modo = getattr(tts, "mode", "inyectado") if tts is not None else "none"
        if tts is None and tts_mode != "none":
            try:
                from src.tts import TTSModule
                self.tts = TTSModule(mode=tts_mode, voice=tts_voice)
                self.tts_modo = tts_mode
                print(f"  TTS: modo={tts_mode}")
            except Exception as exc:   # sin motor no se cae el prototipo: se registra
                print(f"  [AVISO] TTS no disponible ({exc}); se continúa sin audio")
                self.tts = None
                self.tts_modo = "none"

        # 4.b Cache de voz caliente (16-sep-2026, pendiente de N-34): las 50 glosas sueltas, ya pasadas
        # por GrammarMapper, son las frases más probables del modo seña. Sintetizarlas en segundo plano
        # al arrancar deja la etapa «síntesis» en ~0 ms cuando llegan, sin cambiar qué mide t5.
        self.tts_precalentar = bool(tts_precalentar)
        self.tts_precalentando = self._precalentar_voz() if self.tts_precalentar else False

        # 5. Registro JSONL (un archivo por día; cada registro lleva el id de sesión)
        self.sesion = _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.registro_activo = bool(registro)
        self.registro_dir = Path(registro_dir) if registro_dir is not None else ROOT_DIR / "registros"
        self.registro_path = self.registro_dir / f"latencia_{self.sesion[:10]}.jsonl"

        # 6. Estado
        self.ensayo: Ensayo | None = None
        self.n_ensayos = 0
        self.contadores = {estado: 0 for estado in ESTADOS}
        self.ultimo: dict | None = None
        self.glosas_turno: list[str] = []
        self.enunciado_turno = ""
        self.n_turnos = 0
        self._c_recientes: deque[int] = deque(maxlen=60)   # para medir la tasa del bucle
        self._fps_ultima: float | None = None

    # ------------------------------------------------------------------
    # Reloj y tasa
    # ------------------------------------------------------------------

    def ahora(self) -> int:
        """Marca del único reloj monótono del prototipo (ns)."""
        return int(self.reloj())

    def fps_medida(self) -> float | None:
        """Tasa efectiva del bucle captura+extracción sobre los últimos fotogramas."""
        if len(self._c_recientes) >= 10:
            c = self._c_recientes
            dur = c[-1] - c[0]
            if dur > 0:
                self._fps_ultima = (len(c) - 1) * 1e9 / dur
        return self._fps_ultima

    @property
    def ventana_abierta(self) -> bool:
        return self.ensayo is not None

    # ------------------------------------------------------------------
    # Ventana
    # ------------------------------------------------------------------

    def abrir_ventana(self, t0: int | None = None) -> Ensayo | None:
        """Sella t0 y abre la ventana. Devuelve None si ya hay una abierta."""
        if self.ensayo is not None:
            return None
        t0 = int(t0) if t0 is not None else self.ahora()
        fps = self.fps_medida()
        if self.ventana_frames > 0:
            n_objetivo = self.ventana_frames
        else:
            # N cubre ventana_seg a la tasa REAL del bucle (captura + extracción),
            # medida sobre los últimos fotogramas antes de la tecla.
            n_objetivo = int(round(self.ventana_seg * (fps if fps else FPS_SUPUESTA)))
        n_objetivo = max(n_objetivo, self.min_frames)
        self.n_ensayos += 1
        self.ensayo = Ensayo(
            id_ensayo=self.n_ensayos,
            modo="oracion" if self.modo_oracion else "aislado",
            t0=t0,
            ventana_frames=n_objetivo,
            ventana_max_ns=int(self.ventana_max_seg * 1e9),
            fps_medida=fps,
        )
        return self.ensayo

    def abortar_ventana(self) -> dict | None:
        """Cierra la ventana abierta como descartada (secuencia incompleta)."""
        if self.ensayo is None:
            return None
        return self._cerrar_ventana("abortada")

    def procesar_fotograma(self, frame_rgb: np.ndarray, c_i: int | None = None) -> ResultadoFotograma:
        """Extrae puntos de un fotograma; si hay ventana abierta lo acumula y, si con
        él se completa, cierra la ventana y ejecuta el resto de la cadena UNA vez.

        c_i: marca de llegada del fotograma. La interfaz la sella justo después de
        cap.read() y la pasa aquí; si es None se sella al entrar.
        """
        c_i = int(c_i) if c_i is not None else self.ahora()
        ext = self.extractor.extract(frame_rgb)
        e_i = self.ahora()
        self._c_recientes.append(c_i)

        ens = self.ensayo
        if ens is None:
            return ResultadoFotograma(extraccion=ext, c=c_i, e=e_i, en_ventana=False, progreso=0.0)

        ens.c.append(c_i)
        ens.e.append(e_i)
        ens.keypoints.append(ext.keypoints)
        ens.rostros.append(getattr(ext, "rostro", None))
        ens.pose_flags.append(bool(ext.pose_detectada))
        ens.manos_counts.append(int(ext.n_manos))

        cierre = None
        if ens.n >= ens.ventana_frames:
            cierre = "frames"
        elif c_i - ens.t0 >= ens.ventana_max_ns:
            cierre = "tiempo"

        registro = self._cerrar_ventana(cierre) if cierre else None
        progreso = 1.0 if cierre else min(ens.n / ens.ventana_frames, 1.0)
        return ResultadoFotograma(extraccion=ext, c=c_i, e=e_i, en_ventana=True,
                                  progreso=progreso, registro=registro)

    # ------------------------------------------------------------------
    # Cadena tras el cierre de la ventana
    # ------------------------------------------------------------------

    def _cerrar_ventana(self, cierre: str) -> dict:
        ens = self.ensayo
        self.ensayo = None
        ens.cierre = cierre
        # La tasa se vuelve a medir desde cero tras cada ventana (el resto de la cadena
        # detiene el bucle unos ms y contaminaría la estimación).
        self._c_recientes.clear()

        n = ens.n
        if n > 0:
            ens.t1 = ens.c[-1]           # t1 = c_N por definición (§2.2.7)

        sin_manos = sum(1 for m in ens.manos_counts if m == 0)
        sin_pose = sum(1 for p in ens.pose_flags if not p)
        prop_sin_manos = sin_manos / n if n else 1.0
        prop_sin_pose = sin_pose / n if n else 1.0

        # (c) Secuencia incompleta: abortada por la persona o con menos fotogramas que el mínimo.
        if cierre == "abortada" or n < self.min_frames:
            ens.estado = "rechazada_incompleta"
            ens.motivo = "abortada" if cierre == "abortada" else f"{n} fotogramas < mínimo {self.min_frames}"
        # (b) Manos (o pose) no detectadas: se indica y NO se infiere.
        elif prop_sin_manos > self.max_sin_manos:
            ens.estado = "rechazada_manos"
            ens.motivo = f"manos no detectadas en {prop_sin_manos:.0%} de la ventana (> {self.max_sin_manos:.0%})"
        elif prop_sin_pose > self.max_sin_manos:
            ens.estado = "rechazada_manos"
            ens.motivo = f"pose no detectada en {prop_sin_pose:.0%} de la ventana (> {self.max_sin_manos:.0%})"
        else:
            self._inferir_y_emitir(ens)

        # Aviso de bordes (N-50, opción B): con el estado ya fijado (y t5 ya sellado), así que no cambia el
        # estado, ni la inferencia, ni ninguna marca t0..t5 ni etapa de §2.2.7.
        ens.bordes = self._aviso_bordes(ens)

        self.contadores[ens.estado] += 1
        registro = self._registro_ensayo(ens, sin_manos=sin_manos, sin_pose=sin_pose)
        self._escribir(registro)
        self.ultimo = registro
        return registro

    def _aviso_bordes(self, ens: Ensayo) -> dict:
        """Aviso de seña posiblemente cortada en los bordes (N-50, opción B; src/bordes_ventana.py).

        Postura de la ventana (fotogramas sin postura en ceros, como en el corpus) → normalize_sequence →
        medir_bordes con las marcas c_i → evaluar_bordes con los umbrales. Es independiente de la
        representación y de la regla de mano ausente del modelo: solo usa la postura, que se extrae siempre.
        Se calcula para todo cierre con N ≥ 2, también en rechazos; con cierre «abortada» las medidas y
        las banderas quedan en null. costo_ns es el tiempo del cálculo con time.perf_counter_ns (no con el
        reloj inyectado, para no mover las marcas de los ensayos siguientes en las pruebas)."""
        if self.bordes_modo == "off":
            return {"modo": "off"}
        u = self.umbrales_bordes
        inicio = time.perf_counter_ns()
        if ens.cierre == "abortada" or ens.n < 2:
            medidas = medir_bordes(np.zeros((0, N_POSE * COORDS), np.float32), np.zeros(0, np.int64), u["borde_s"])
        else:
            pose = np.stack([kp[IDX_POSE] for kp in ens.keypoints]).astype(np.float32)   # (N, 99)
            pose[~np.asarray(ens.pose_flags, dtype=bool)] = 0.0
            medidas = medir_bordes(normalize_sequence(pose), ens.c, u["borde_s"])
        avisos = evaluar_bordes(medidas, u)
        costo_ns = time.perf_counter_ns() - inicio
        if avisos["aviso_inicio"]:
            self.avisos_bordes["inicio"] += 1
        if avisos["aviso_fin"]:
            self.avisos_bordes["fin"] += 1
        return {"modo": "aviso", "umbrales": self._umbrales_bordes_registro, **avisos, **medidas,
                "costo_ns": int(costo_ns)}

    def _preprocesar(self, ens: Ensayo) -> np.ndarray:
        """Construye la entrada del modelo con el mismo orden que el entrenamiento: fila completa
        (postura + manos [+ cejas y labios], como preprocess_lsc50), regla sin postura, regla de
        ausencia, normalización por hombros, máscara, columnas de la representación (D-08),
        resample_sequence y add_velocity (train_final_100). Con «imputacion» (N-63), entre las columnas
        y el remuestreo se imputa la mano ausente por mano sobre los fotogramas de la ventana
        (mano_ausente_variantes.imputar_fotogramas), igual que src/train_final_tcn.py."""
        seq = np.stack(ens.keypoints).astype(np.float32)          # (N, 225)
        if self.arma_rostro:
            bloque = np.zeros((seq.shape[0], self._cols_rostro.size), dtype=np.float32)
            for t, r in enumerate(ens.rostros):
                if r is not None:
                    bloque[t] = np.asarray(r, dtype=np.float32)[self._cols_rostro]
            seq = np.concatenate([seq, bloque], axis=1)           # (N, 405)
        # Un fotograma sin pose no tiene hombros: sin referencia de normalización se
        # deja entero en ceros (misma convención de relleno que D-01) en vez de dividir
        # por la escala mínima y disparar los valores.
        sin_pose = ~np.asarray(ens.pose_flags, dtype=bool)
        if sin_pose.any():
            seq[sin_pose] = 0.0
        # D-07: la mano no detectada llega como bloque en ceros ANTES de normalizar.
        sin_izq = (seq[:, IDX_MANO_IZQ] == 0).all(axis=1)
        sin_der = (seq[:, IDX_MANO_DER] == 0).all(axis=1)
        sin_rostro = (seq[:, N_FEATURES:] == 0).all(axis=1) if self.arma_rostro else np.zeros(len(seq), bool)
        if self.mano_ausente == "descartar":
            conservar = ~(sin_izq | sin_der | sin_rostro)
            if conservar.any():                                   # nunca dejar la ventana vacía
                seq = seq[conservar]
                sin_izq, sin_der, sin_rostro = sin_izq[conservar], sin_der[conservar], sin_rostro[conservar]
        seq = normalize_sequence(seq)
        if self.mano_ausente in ("mascara", "imputacion"):       # «imputacion» parte del processed «mascara»
            seq[sin_izq, IDX_MANO_IZQ] = 0.0
            seq[sin_der, IDX_MANO_DER] = 0.0
            seq[sin_rostro, N_FEATURES:] = 0.0
        seq = seq[:, self._columnas]                              # representación (D-08)
        if self.mano_ausente == "imputacion":
            seq = imputar_fotogramas(seq)                         # (N, 126) antes de remuestrear (N-63)
        seq = resample_sequence(seq, self.seq_len)                # (30, F)
        X = seq[np.newaxis, ...].astype(np.float32)               # (1, 30, F)
        if self.usa_velocidades:
            X = add_velocity(X)                                   # (1, 30, 2F)
        return X

    def _inferir_y_emitir(self, ens: Ensayo) -> None:
        X = self._preprocesar(ens)
        ens.t2 = self.ahora()                                     # fin del preprocesamiento

        prob = np.asarray(self.modelo.predict(X, verbose=0))[0]
        ens.t3 = self.ahora()                                     # retorno del modelo

        orden = np.argsort(prob)[::-1][:3]
        ens.top3 = [{"id": int(i), "glosa": SIGN_VOCABULARY.get(int(i), f"[{int(i)}]"),
                     "p": float(prob[i])} for i in orden]
        ens.id_clase = int(orden[0])
        ens.clase = SIGN_VOCABULARY.get(ens.id_clase, f"[{ens.id_clase}]")
        ens.confianza = float(prob[ens.id_clase])

        # (a) Confianza bajo el corte: «seña no reconocida», sin audio.
        if ens.confianza < self.umbral:
            ens.estado = "rechazada_confianza"
            ens.motivo = f"confianza {ens.confianza:.2f} < umbral {self.umbral:.2f}: repita la seña"
            return

        ens.estado = "aceptada"
        if self.modo_oracion:
            # La conversión y la síntesis se difieren al fin de turno (gancho del mes 3).
            self.glosas_turno.append(ens.clase)
            self.enunciado_turno = ""
            return

        ens.enunciado, ens.conversion = self._convertir([ens.clase])
        ens.t4 = self.ahora()                                     # retorno de GrammarMapper
        ens.tts_ocupado_en_t5 = bool(getattr(self.tts, "is_speaking", False)) if self.tts is not None else None
        ens.tts_sintesis_antes_de_t5 = self._emitir_audio(ens.enunciado)
        ens.t5 = self.ahora()                                     # orden de reproducción emitida
        ens.tts_sintesis_desde_cache = self._sintesis_desde_cache()   # después de t5: no altera marcas
        ens.tts_fuente, ens.tts_tiempo_sintesis_ns = self._fuente_voz()

    # ------------------------------------------------------------------
    # Modo oración: fin de turno explícito
    # ------------------------------------------------------------------

    def fin_de_turno(self, t_tecla: int | None = None) -> dict | None:
        """Convierte las glosas acumuladas con GrammarMapper y sintetiza. Devuelve el
        registro del turno (tipo "turno") o None si no había glosas."""
        if not self.glosas_turno:
            return None
        t_a = int(t_tecla) if t_tecla is not None else self.ahora()
        glosas = list(self.glosas_turno)
        enunciado, conversion = self._convertir(glosas)
        t_b = self.ahora()
        sintesis_antes = self._emitir_audio(enunciado)
        t_c = self.ahora()
        desde_cache = self._sintesis_desde_cache()     # después de t_c: no altera marcas
        fuente, tiempo_fuente = self._fuente_voz()
        self.n_turnos += 1
        self.enunciado_turno = enunciado
        self.glosas_turno.clear()
        registro = {
            "tipo": "turno",
            "sesion": self.sesion,
            "id_turno": self.n_turnos,
            "fecha_hora": _dt.datetime.now().isoformat(timespec="milliseconds"),
            "glosas": glosas,
            "n_glosas": len(glosas),
            "enunciado": enunciado,
            "t_tecla": t_a,
            "t_conversion": t_b,
            "t_sintesis": t_c,
            "conversion_ns": t_b - t_a,
            "sintesis_ns": t_c - t_b,
            "tts_modo": self.tts_modo,
            "tts_sintesis_antes_de_orden": sintesis_antes,
            **({"tts_sintesis_desde_cache": desde_cache} if desde_cache is not None else {}),
            **({"tts_fuente": fuente} if fuente is not None else {}),
            **({"tts_tiempo_sintesis_ns": tiempo_fuente} if tiempo_fuente is not None else {}),
            **({"conversion": conversion} if conversion is not None else {}),
        }
        voz = self._estado_voz()                       # después de t_c: no altera las marcas
        registro["tts_errores_reproduccion"] = voz["errores_reproduccion"]
        registro["tts_ultimo_error_reproduccion"] = voz["ultimo_error_reproduccion"]
        self._escribir(registro)
        return registro

    def _convertir(self, glosas: list[str]) -> tuple[str, dict | None]:
        """Llama al conversor y devuelve (enunciado, traza o None).

        Con ConversorGlosas (3.a) se llama a explicar y la traza (§6, sin «frase», que ya va como
        «enunciado») queda en el registro bajo «conversion». Con cualquier otro conversor (GrammarMapper
        o un doble de prueba) se llama a translate como siempre y no hay traza. Se detecta con
        isinstance (en _es_conversor_glosas, sin importar el módulo) y no con getattr(…, "explicar"):
        un MagicMock tiene cualquier atributo."""
        if _es_conversor_glosas(self.grammar_mapper):
            traza = self.grammar_mapper.explicar(glosas)
            return traza["frase"], {k: v for k, v in traza.items() if k != "frase"}
        return self.grammar_mapper.translate(glosas), None

    def reiniciar_turno(self) -> None:
        """Descarta las glosas acumuladas sin convertir ni sintetizar."""
        self.glosas_turno.clear()
        self.enunciado_turno = ""

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _emitir_audio(self, texto: str) -> bool | None:
        """Sintetiza y emite la orden de reproducción; retorna en el instante t5.

        Devuelve True si el audio ya estaba sintetizado al emitir la orden (la etapa «síntesis»
        t5 - t4 incluye la síntesis y excluye la reproducción, como exige §2.2.7), False si no, None
        si no hubo audio. Ese valor queda en el JSONL.

        Con src/tts.py los tres modos (offline incluido, desde el 15-sep-2026) sintetizan a archivo:
        prepare() → espera a que termine la síntesis (esperar_preparado, con tope
        tts_espera_max_seg) → play_prepared_async() emite la orden. Un motor sin esa interfaz (dobles
        de prueba) usa speak_async y declara la síntesis como no incluida.
        """
        tts = self.tts
        if tts is None or not texto:
            return None
        prepara = (hasattr(tts, "prepare") and hasattr(tts, "play_prepared_async")
                   and (getattr(tts, "soporta_preparar", False) or getattr(tts, "mode", None) in ("edge", "gtts")))
        if prepara:
            tts.prepare(texto)
            if hasattr(tts, "esperar_preparado"):
                listo = bool(tts.esperar_preparado(self.tts_espera_max_seg))
            else:
                limite = self.ahora() + int(self.tts_espera_max_seg * 1e9)
                while not tts.is_prepared and self.ahora() < limite:
                    time.sleep(0.002)
                listo = bool(tts.is_prepared)
            tts.play_prepared_async()
            return listo
        tts.speak_async(texto)
        return False

    def _precalentar_voz(self) -> bool:
        """Pide al motor que llene su cache con las 50 glosas sueltas convertidas a enunciado.

        No bloquea: el motor sintetiza en segundo plano y espera a que no haya ninguna preparación en
        curso antes de cada frase. Devuelve False si no hay motor, si no tiene cache (dobles de prueba
        y motores inyectados sin esa interfaz) o si el precalentado no pudo arrancar."""
        tts = self.tts
        precalentar = getattr(tts, "precalentar", None) if tts is not None else None
        if not callable(precalentar):
            return False
        # Si el motor no puede preparar el audio a archivo, el prototipo habla con speak_async y nunca
        # consultaría la cache: precalentarla sería trabajo perdido (misma condición que _emitir_audio).
        if not (getattr(tts, "soporta_preparar", False) or getattr(tts, "mode", None) in ("edge", "gtts")):
            return False
        try:
            frases = [self.grammar_mapper.translate([glosa]) for glosa in SIGN_VOCABULARY.values()]
            return bool(precalentar(frases))
        except Exception as exc:   # noqa: BLE001 — sin cache caliente se sintetiza como siempre
            print(f"  [AVISO TTS] no se pudo precalentar la voz ({exc}); se sintetiza frase a frase")
            return False

    def _sintesis_desde_cache(self) -> bool | None:
        """¿El audio de la locución recién emitida salió de la cache del motor (src/tts.py, 16-sep-2026)?

        True: no se sintetizó en este ensayo —la etapa «síntesis» vale casi cero porque el audio ya
        estaba hecho—. False: se sintetizó ahora. None: no hubo audio, o el motor no tiene cache ni
        informa de ella; en ese caso la clave tts_sintesis_desde_cache NO se escribe en el JSONL, para
        no afirmar nada de un motor que no lo dice. Se lee DESPUÉS de sellar t5 (o t_c en el turno), así
        que no altera ninguna marca. Sin esta declaración, una sesión con la cache caliente mostraría
        síntesis ≈ 0 ms y parecería contradecir los ~0,46 s que midió N-34.
        """
        tts = self.tts
        if tts is None:
            return None
        try:
            valor = getattr(tts, "ultima_preparacion_desde_cache", None)
        except Exception:   # noqa: BLE001 — un diagnóstico nunca debe tumbar el registro del ensayo
            return None
        return valor if isinstance(valor, bool) else None

    def _fuente_voz(self) -> tuple[str | None, int | None]:
        """(fuente, ns) de la locución recién preparada según el motor (src/tts.py: ultima_fuente y
        ultima_sintesis_ns, 23-nov-2026). Fuente ∈ {"pregrabada", "xtts", "local", "edge", "gtts"}; cualquier
        otro valor (motor que no informa, doble de prueba) → None y la clave no se escribe. Se lee DESPUÉS de
        sellar t5 (o t_c), así que no altera ninguna marca."""
        tts = self.tts
        if tts is None:
            return None, None
        try:
            fuente = getattr(tts, "ultima_fuente", None)
            ns = getattr(tts, "ultima_sintesis_ns", None)
        except Exception:   # noqa: BLE001 — un diagnóstico nunca debe tumbar el registro
            return None, None
        if not isinstance(fuente, str) or fuente not in FUENTES_VOZ:
            return None, None
        ns = int(ns) if isinstance(ns, int) and not isinstance(ns, bool) and ns >= 0 else None
        return fuente, ns

    def _estado_voz(self) -> dict:
        """Errores de reproducción ACUMULADOS en la sesión hasta este instante (src/tts.py, N-34).

        Una locución emitida con play_prepared_async/speak_async suena en un hilo: si falla, la excepción
        no llega aquí y el ensayo ya quedó registrado como emitido. Para que el fallo no pase inadvertido,
        cada registro del JSONL lleva estos contadores. Se leen de tts.estado_reproduccion() o, si no
        existe, de los atributos del TTSModule, con lectura defensiva para motores o dobles sin esa
        interfaz. Valores: entero y texto corto (≤ 200 caracteres) si el motor los expone; None si no hay
        TTS o el motor no los expone (None = «no informado», no «cero errores»). Se llama DESPUÉS de sellar
        t5 (o t_c en el turno), así que no altera las marcas; la locución del propio ensayo suena después,
        su fallo suele verse en los registros SIGUIENTES (en el mismo si el hilo falla antes) y en close().
        """
        estado = {"errores_reproduccion": None, "ultimo_error_reproduccion": None,
                  "locuciones_descartadas_al_cerrar": None}
        tts = self.tts
        if tts is None:
            return estado
        fuente = None
        try:
            consulta = getattr(tts, "estado_reproduccion", None)
            if callable(consulta):
                fuente = consulta()
            if not isinstance(fuente, dict):
                fuente = {clave: getattr(tts, clave, None) for clave in estado}
        except Exception:   # noqa: BLE001 — un diagnóstico nunca debe tumbar el registro del ensayo
            return estado
        for clave in ("errores_reproduccion", "locuciones_descartadas_al_cerrar"):
            valor = fuente.get(clave)
            estado[clave] = int(valor) if isinstance(valor, int) and not isinstance(valor, bool) else None
        ultimo = fuente.get("ultimo_error_reproduccion")
        if ultimo is not None:
            ultimo = str(ultimo)
            estado["ultimo_error_reproduccion"] = ultimo if len(ultimo) <= 200 else ultimo[:199] + "…"
        return estado

    # ------------------------------------------------------------------
    # Registro
    # ------------------------------------------------------------------

    def _registro_ensayo(self, ens: Ensayo, sin_manos: int, sin_pose: int) -> dict:
        n = ens.n
        voz = self._estado_voz()   # acumulado hasta ahora; se lee tras t5 (ver _estado_voz)
        etapas: dict[str, int | None] = {k: None for k in ETAPAS}
        if n > 0:
            etapas["captura"] = ens.t1 - ens.t0
            etapas["extraccion"] = ens.e[-1] - ens.t1
        if ens.t2 is not None:
            etapas["preprocesamiento"] = ens.t2 - ens.e[-1]
        if ens.t3 is not None:
            etapas["inferencia"] = ens.t3 - ens.t2
        if ens.t4 is not None:
            etapas["conversion"] = ens.t4 - ens.t3
        if ens.t5 is not None:
            etapas["sintesis"] = ens.t5 - ens.t4

        completo = ens.t5 is not None
        costos = [e - c for c, e in zip(ens.c, ens.e)]
        # Tasa efectiva del bucle de captura y extracción, N/(c_N - c_1), tal como la declara
        # §2.2.7. Cuenta N fotogramas sobre N-1 intervalos, así que excede la tasa por
        # intervalos (N-1)/(c_N - c_1) en el factor N/(N-1) (1,4 % con N = 75). El JSONL
        # conserva c, de modo que la variante por intervalos se recomputa sin volver a medir.
        tasa = n * 1e9 / (ens.c[-1] - ens.c[0]) if n >= 2 and ens.c[-1] > ens.c[0] else None

        return {
            "tipo": "ensayo",
            "sesion": self.sesion,
            "id_ensayo": ens.id_ensayo,
            "fecha_hora": _dt.datetime.now().isoformat(timespec="milliseconds"),
            "modo": ens.modo,
            "modelo": _ruta_para_el_registro(self.model_path),
            "n_rasgos_modelo": self.n_rasgos,
            "usa_velocidades": self.usa_velocidades,
            "pasos_modelo": self.seq_len,
            "extractor": self.extractor_desc,
            "representacion": self.representacion,
            "procedencia_modelo": self.procedencia_ruta,
            "mano_ausente": self.mano_ausente,
            "mano_ausente_modelo": self.mano_ausente_modelo,
            "tts_modo": self.tts_modo,
            "tts_sintesis_antes_de_t5": ens.tts_sintesis_antes_de_t5,
            "tts_ocupado_en_t5": ens.tts_ocupado_en_t5,
            "tts_errores_reproduccion": voz["errores_reproduccion"],
            "tts_ultimo_error_reproduccion": voz["ultimo_error_reproduccion"],
            **({"tts_sintesis_desde_cache": ens.tts_sintesis_desde_cache}
               if ens.tts_sintesis_desde_cache is not None else {}),
            **({"tts_fuente": ens.tts_fuente} if ens.tts_fuente is not None else {}),
            **({"tts_tiempo_sintesis_ns": ens.tts_tiempo_sintesis_ns} if ens.tts_tiempo_sintesis_ns is not None else {}),
            "parametros": {
                "umbral": self.umbral,
                "umbral_origen": self.umbral_origen,
                "ventana_frames_objetivo": ens.ventana_frames,
                "ventana_frames_param": self.ventana_frames,
                "ventana_seg": self.ventana_seg,
                "ventana_max_seg": self.ventana_max_seg,
                "min_frames": self.min_frames,
                "max_sin_manos": self.max_sin_manos,
                "fps_medida_antes": ens.fps_medida,
            },
            "t0": ens.t0,
            "c": list(ens.c),
            "e": list(ens.e),
            "t1": ens.t1, "t2": ens.t2, "t3": ens.t3, "t4": ens.t4, "t5": ens.t5,
            "N": n,
            "cierre": ens.cierre,
            "frames_con_deteccion": sum(1 for p, m in zip(ens.pose_flags, ens.manos_counts) if p and m > 0),
            "frames_sin_manos": sin_manos,
            "frames_sin_pose": sin_pose,
            "frames_con_dos_manos": sum(1 for m in ens.manos_counts if m == 2),
            "proporcion_sin_manos": (sin_manos / n) if n else None,
            "etapas_ns": etapas,
            "W_ns": (ens.t1 - ens.t0) if n > 0 else None,
            "L_total_ns": (ens.t5 - ens.t0) if completo else None,
            "L_resp_ns": (ens.t5 - ens.t1) if completo else None,
            "costo_extractor_ns": _estadisticas_ns(costos),
            "tasa_efectiva_fps": tasa,
            "estado": ens.estado,
            "motivo": ens.motivo,
            "id_clase": ens.id_clase,
            "clase": ens.clase,
            "confianza": ens.confianza,
            "top3": ens.top3,
            "enunciado": ens.enunciado,
            "glosas_turno": list(self.glosas_turno),
            "contadores": dict(self.contadores),
            **({"conversion": ens.conversion} if ens.conversion is not None else {}),
            # Siempre la última clave (N-50, opción B); las anteriores no cambian ni de nombre ni de orden.
            "bordes": ens.bordes if ens.bordes is not None else {"modo": self.bordes_modo},
        }

    def _escribir(self, registro: dict) -> None:
        if not self.registro_activo:
            return
        self.registro_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registro_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Cierre
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Cierra extractor y TTS. El JSONL no tiene registro de cierre: los errores de la voz van en cada
        registro (tts_errores_reproduccion) y, si hubo alguno o se descartaron locuciones al cerrar, se
        imprime UNA línea de resumen. El estado final queda en self.estado_voz_al_cerrar. Las locuciones
        que el hilo descarta después de este instante no alcanzan a contarse aquí."""
        if hasattr(self.extractor, "close"):
            self.extractor.close()
        if self.tts is not None and hasattr(self.tts, "cleanup"):
            self.tts.cleanup()
        voz = self._estado_voz()
        self.estado_voz_al_cerrar = voz
        errores = voz["errores_reproduccion"] or 0
        descartadas = voz["locuciones_descartadas_al_cerrar"] or 0
        if errores or descartadas:
            ultimo = f" (último: {voz['ultimo_error_reproduccion']})" if voz["ultimo_error_reproduccion"] else ""
            print(f"  [AVISO TTS] Voz de la sesión: {errores} locución(es) no sonaron por error de "
                  f"reproducción{ultimo}; {descartadas} descartada(s) al cerrar.")
