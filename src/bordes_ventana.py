"""
Aviso de seña posiblemente cortada en los bordes de la ventana del prototipo (§2.2.6; N-50, opción B).

Qué es y qué no es:
  · Es un AVISO: se escribe en el registro JSONL (aviso_inicio, aviso_fin y los valores medidos) y en
    pantalla. NUNCA cambia el estado del ensayo ni impide la inferencia; la seña no se rechaza por él.
  · La protección contra la seña cortada es el protocolo de reposo (la persona está en reposo cuando se
    pulsa ESPACIO y vuelve al reposo antes del cierre) y la tecla X para abortar (rechazada_incompleta).
  · El aviso deja datos de ventanas reales para calibrar, más adelante, una regla que sí rechace
    (opción A de models/senas_cortadas_2026-09-22/recomendacion.md).

La MEDIDA es la de src/analisis_senas_cortadas.py, reutilizada sin cambios: _velocidad_pose (por par
consecutivo, media sobre las muñecas 15 y 16 de la postura de ‖Δxy‖, NaN si falta la postura en alguno de
los dos fotogramas) y _altura (por fotograma, min(y_muñeca_izq, y_muñeca_der), NaN sin postura), sobre la
postura normalizada por hombros, y los mismos tramos de borde que rasgos_clip (pares dentro de los
primeros / últimos K fotogramas; alturas de esos K fotogramas). Lo único que cambia es la unidad: el
análisis mide en anchos de hombro POR FOTOGRAMA a 24 fps y el prototipo, cuyo bucle no corre a 24 fps,
divide cada desplazamiento entre su Δt real (marcas c_i de llegada de los fotogramas), es decir, mide en
anchos de hombro POR SEGUNDO, que es la unidad del archivo de umbrales (umbrales_por_segundo del análisis).
K = max(3, round(borde_s · fps_ventana)) como en umbrales_a_fps del análisis, con K ≤ N.

Los UMBRALES son los del archivo CANDIDATO que escribe el análisis
(models/senas_cortadas_2026-09-22/umbral_bordes_candidato.json; percentil 95 de los clips íntegros de los
cinco señantes, K = 3 a 24 fps → borde de 0,125 s): v_pose_por_s ini 1,7614 / fin 0,2295 anchos de
hombro por segundo y h ini 0,9740 / fin 1,2610 anchos de hombro bajo la línea de los hombros.

Independencia de la red: la medida solo usa la POSTURA (muñecas y hombros), que el prototipo extrae y
normaliza siempre, sea cual sea la representación de entrada del modelo. La red heredada y el modelo final
(TCN + imputación + 252) usan la misma representación «manos» del mismo processed
(data/processed_holistic_mascara_rostro, del que salieron los umbrales), y ninguna de las dos recibe la
postura como entrada; la regla de mano ausente («mascara» o «imputacion») solo toca los bloques de las
manos. Por eso los umbrales valen igual con las dos redes y no hubo que adaptarlos. Lo que sí se midió con
la red heredada, y no se ha repetido con el modelo final, es el DAÑO de un recorte sobre el acierto
(N-50, punto 3 de recomendacion.md); como el aviso no rechaza, ese dato no cambia lo que hace el prototipo.

La altura h está en coordenadas y normalizadas, que dependen de la forma de la imagen (N-57: con 4:3 la y
normalizada queda multiplicada por 0,75 respecto del corpus, que es 16:9). Los umbrales de h solo son
comparables con una captura 16:9 (por eso realtime.py captura por defecto a 1280 × 720).

Sin TensorFlow ni MediaPipe: solo numpy y las funciones del análisis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# La MISMA medida del análisis de N-50 (no se reimplementa).
from src.analisis_senas_cortadas import _altura, _nanmean, _velocidad_pose  # noqa: E402

UMBRAL_BORDES_CANDIDATO = ROOT_DIR / "models" / "senas_cortadas_2026-09-22" / "umbral_bordes_candidato.json"
K_MINIMO = 3                         # el mismo mínimo que umbrales_a_fps del análisis
MODOS_BORDES = ("off", "aviso")
COLUMNAS_POSTURA = 99                # 33 puntos × 3 de la postura, al inicio de los 225 rasgos
MEDIDAS = ("v_pose_ini_por_s", "v_pose_fin_por_s", "h_ini", "h_fin")


def cargar_umbrales_bordes(ruta: str | Path = UMBRAL_BORDES_CANDIDATO) -> dict:
    """Lee y valida el archivo de umbrales (formato de umbral_bordes_candidato.json).

    Exige borde_s > 0 y los cuatro umbrales numéricos v_pose_por_s.{ini,fin} y h.{ini,fin}. Devuelve un
    dict con esos valores, la ruta y el estado declarado en el archivo."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de umbrales de bordes {ruta}")
    datos = json.loads(ruta.read_text(encoding="utf-8"))

    def numero(valor, nombre):
        if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not np.isfinite(valor):
            raise ValueError(f"{ruta.name}: {nombre} debe ser un número finito, no {valor!r}")
        return float(valor)

    faltan = [c for c in ("borde_s", "v_pose_por_s", "h") if c not in datos]
    if faltan:
        raise ValueError(f"{ruta.name}: faltan las claves {faltan}")
    umbrales = {"borde_s": numero(datos["borde_s"], "borde_s"), "ruta": str(ruta),
                "estado": datos.get("estado")}
    if umbrales["borde_s"] <= 0:
        raise ValueError(f"{ruta.name}: borde_s debe ser positivo")
    for rasgo in ("v_pose_por_s", "h"):
        if not isinstance(datos[rasgo], dict):
            raise ValueError(f"{ruta.name}: {rasgo} debe tener las claves ini y fin")
        umbrales[rasgo] = {}
        for borde in ("ini", "fin"):
            if borde not in datos[rasgo]:
                raise ValueError(f"{ruta.name}: falta {rasgo}.{borde}")
            umbrales[rasgo][borde] = numero(datos[rasgo][borde], f"{rasgo}.{borde}")
    return umbrales


def _o_none(x: float) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def medir_bordes(pose_normalizada: np.ndarray, c_ns, borde_s: float) -> dict:
    """Rasgos de borde de una ventana del prototipo.

    Args:
        pose_normalizada: (N, ≥99) con la postura en las 99 primeras columnas, YA normalizada por hombros
            (normalize_sequence de translator.py) y con los fotogramas sin postura en ceros, como en el
            processed del corpus.
        c_ns: marcas de llegada c_i de los N fotogramas (ns, reloj monótono).
        borde_s: duración del borde en segundos (borde_s del archivo de umbrales).

    Devuelve K, fps_ventana, v_pose_ini_por_s, v_pose_fin_por_s (anchos de hombro por segundo), h_ini,
    h_fin (anchos de hombro; y crece hacia abajo, 0 = línea de los hombros) y los pares válidos por borde.
    Un rasgo sin datos (sin postura en el borde, o ventana de menos de 2 fotogramas) vale None."""
    seq = np.asarray(pose_normalizada, dtype=np.float64)[:, :COLUMNAS_POSTURA]
    c = np.asarray(c_ns, dtype=np.int64)
    N = seq.shape[0]
    if c.shape[0] != N:
        raise ValueError(f"{c.shape[0]} marcas c_i para {N} fotogramas")
    vacio = {"K": None, "fps_ventana": None, **{m: None for m in MEDIDAS},
             "pares_validos_ini": 0, "pares_validos_fin": 0}
    if N < 2 or c[-1] <= c[0]:
        return vacio
    fps = (N - 1) / ((c[-1] - c[0]) / 1e9)
    K = int(min(max(K_MINIMO, round(borde_s * fps)), N))

    # Desplazamiento por par (anchos de hombro) del análisis, dividido entre el Δt real del par.
    desplazamiento = _velocidad_pose(seq)                       # (N-1,), NaN sin postura
    dt = np.diff(c).astype(np.float64) / 1e9
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(dt > 0, desplazamiento / dt, np.nan)
    h = _altura(seq)                                            # (N,), NaN sin postura

    # Mismos tramos que rasgos_clip: pares dentro de los primeros / últimos K fotogramas.
    v_ini, v_fin = v[:K - 1], v[N - K:]
    return {
        "K": K,
        "fps_ventana": float(fps),
        "v_pose_ini_por_s": _o_none(_nanmean(v_ini)),
        "v_pose_fin_por_s": _o_none(_nanmean(v_fin)),
        "h_ini": _o_none(_nanmean(h[:K])),
        "h_fin": _o_none(_nanmean(h[N - K:])),
        "pares_validos_ini": int(np.sum(~np.isnan(v_ini))),
        "pares_validos_fin": int(np.sum(~np.isnan(v_fin))),
    }


def evaluar_bordes(medidas: dict, umbrales: dict) -> dict:
    """Banderas de aviso con la regla del archivo de umbrales (la de aplicar_regla del análisis, rasgo
    «v_pose ∨ h»): aviso_inicio = v_pose_ini_por_s > v.ini o h_ini < h.ini; aviso_fin igual con «fin».
    Desigualdades estrictas: un valor igual al umbral no avisa. Un rasgo None no dispara; si los dos
    rasgos de un borde son None, la bandera es None («no se pudo medir»), no False."""
    out = {}
    for borde, nombre in (("ini", "aviso_inicio"), ("fin", "aviso_fin")):
        v = medidas.get(f"v_pose_{borde}_por_s")
        h = medidas.get(f"h_{borde}")
        if v is None and h is None:
            out[nombre] = None
            continue
        out[nombre] = bool((v is not None and v > umbrales["v_pose_por_s"][borde])
                           or (h is not None and h < umbrales["h"][borde]))
    return out
