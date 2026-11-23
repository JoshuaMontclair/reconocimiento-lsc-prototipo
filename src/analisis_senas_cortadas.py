"""
Señas cortadas en los bordes de la ventana (§2.2.6 del anteproyecto, 09_metodologia.tex:192-193;
riesgo R6 y caso CL-6b de docs/PROTOCOLO_EXPERIMENTAL_4a.md; hallazgo de N-37). SOLO ANÁLISIS Y
REPORTE: no cambia el prototipo, no toca modelos, umbrales ni particiones.

Qué hace (criterio fijado antes de correr en models/senas_cortadas_2026-09-22/criterio.md):
  1. Mide en el corpus procesado (data/processed_holistic_mascara_rostro, solo lectura) cómo empiezan
     y terminan los clips: velocidad de las muñecas de la postura y de los puntos de las manos en los
     primeros y últimos K fotogramas (K = 3, 5, 8) frente al tramo central, altura de las muñecas
     respecto a la línea de los hombros, y dónde caen los fotogramas sin mano. Distribuciones por
     señante y por seña.
  2. Genera clips truncados artificialmente (20 %, 35 % y 50 % del inicio o del final, antes de
     remuestrear) y evalúa una familia de reglas «cortada al inicio / al final»: umbrales ajustados
     SOLO con los señantes de entrenamiento de cada pliegue LOSO (data/partitions/loso_5_pliegues.json)
     y evaluados en el señante de prueba (falsos positivos sobre íntegros, detección sobre truncados).
  3. Mide cuánto cae el acierto del modelo «manos» de cada pliegue
     (models/loso_2026-09-15_holistic_mascara_manos/pliegue_k/modelo.keras) con los clips truncados, con
     el preprocesamiento exacto del entrenador (columnas «manos», resample_sequence a 30, add_velocity),
     y qué hace el umbral de confianza vigente (0,20) con ellos. Control: las predicciones sobre los
     clips íntegros deben reproducir las guardadas en pliegue_k/predicciones.json.

Salida (todo en --salida, por defecto models/senas_cortadas_2026-09-22/): resultados.json, informe.md
y figuras PNG. Corre en CPU; las predicciones son unas 7 000 y tardan segundos.

Uso:
    python src/analisis_senas_cortadas.py
    python src/analisis_senas_cortadas.py --sin-modelo      # solo corpus y reglas (sin TensorFlow)
    python src/analisis_senas_cortadas.py --sin-figuras
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

PROCESSED = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
PARTICION = BASE_DIR / "data" / "partitions" / "loso_5_pliegues.json"
LOSO_DIR = BASE_DIR / "models" / "loso_2026-09-15_holistic_mascara_manos"
SALIDA = BASE_DIR / "models" / "senas_cortadas_2026-09-22"
CRITERIO = SALIDA / "criterio.md"

FPS_CORPUS = 24.0                     # N-02: todos los videos de LSC50 a 24 fps
KS = (3, 5, 8)
PROPORCIONES = (0.20, 0.35, 0.50)
LADOS = ("inicio", "fin")
PERCENTIL = 95                        # punto de operación fijado: 5 % de falsos positivos esperados
PERCENTILES_SENSIBILIDAD = (90, 99)
UMBRAL_CONFIANZA = 0.20               # D-09 / N-33, vigente
RASGOS = ("v_pose", "h", "v_pose_o_h", "v_manos")   # orden de desempate (criterio.md)
EMPATE = 0.01
SEQ_LENGTH = 30

COORDS = 3
COL_POSTURA = slice(0, 99)
COL_MUNECA_POSE = {"izq": 15 * COORDS, "der": 16 * COORDS}    # PoseLandmarker 15 y 16
COL_MANO = {"izq": 99, "der": 162}                             # bloques de 63 columnas
N_MANO = 21
COLS_MANOS = np.arange(99, 225)                                 # representación «manos» (D-08)


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def cargar_corpus(processed: Path = PROCESSED) -> dict:
    """X_padded (memmap), lengths, nombres (sample_order), y (clase) y señante por muestra."""
    meta = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
    X = np.load(processed / "X_padded.npy", mmap_mode="r")
    lengths = np.load(processed / "lengths.npy").astype(np.int64)
    nombres = list(meta["sample_order"])
    y = np.load(processed / "y_labels.npy").astype(np.int64)
    if not (len(nombres) == len(lengths) == X.shape[0] == len(y)):
        raise SystemExit("processed incoherente: sample_order, lengths, X_padded e y_labels difieren en tamaño")
    senantes = [n.split("_")[1] for n in nombres]
    return {"X": X, "lengths": lengths, "nombres": nombres, "y": y, "senantes": senantes,
            "metadata": {k: meta[k] for k in ("n_samples", "n_features", "frame_stats", "landmarks_dir",
                                                "fecha_procesamiento") if k in meta}}


def cargar_particion(ruta: Path = PARTICION) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def conjuntos_pliegue(particion: dict, k: int) -> tuple[list[str], list[str], str]:
    """(nombres de entrenamiento = train ∪ val, nombres de prueba, señante de prueba) del pliegue k."""
    p = particion["pliegues"][k]
    c = p["conjuntos"]
    entren = list(c["train"]) + list(c["val"])
    prueba = list(c["test"])
    if set(entren) & set(prueba):
        raise ValueError(f"pliegue {k}: entrenamiento y prueba comparten muestras")
    s_test = p["senante_test"]
    if any(n.split("_")[1] == s_test for n in entren) or any(n.split("_")[1] != s_test for n in prueba):
        raise ValueError(f"pliegue {k}: el señante de prueba {s_test} no está separado")
    return entren, prueba, s_test


# ---------------------------------------------------------------------------
# Recorte artificial y rasgos de borde
# ---------------------------------------------------------------------------

def recortar(seq: np.ndarray, proporcion: float, lado: str) -> np.ndarray:
    """Quita round(p·T) fotogramas del inicio («cortada al inicio») o del final («cortada al final»).
    Se aplica ANTES de remuestrear. Nunca deja menos de 2 fotogramas."""
    T = seq.shape[0]
    n = int(round(proporcion * T))
    n = min(n, max(T - 2, 0))
    if lado == "inicio":
        return seq[n:]
    if lado == "fin":
        return seq[:T - n]
    raise ValueError(f"lado desconocido: {lado!r}")


def _velocidad_pose(seq: np.ndarray) -> np.ndarray:
    """Por par consecutivo (T-1,): media sobre las dos muñecas de la postura de ‖Δxy‖; NaN si la postura
    falta en alguno de los dos fotogramas."""
    presente = np.any(seq[:, COL_POSTURA] != 0, axis=1)
    v = []
    for c0 in COL_MUNECA_POSE.values():
        xy = seq[:, c0:c0 + 2]
        v.append(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    v = np.mean(v, axis=0)
    v[~(presente[1:] & presente[:-1])] = np.nan
    return v


def _velocidad_manos(seq: np.ndarray) -> np.ndarray:
    """Por par consecutivo (T-1,): media, sobre las manos detectadas en ambos fotogramas, del
    desplazamiento medio ‖Δxy‖ de sus 21 puntos; NaN si ninguna mano está en ambos."""
    T = seq.shape[0]
    suma = np.zeros(T - 1)
    cuenta = np.zeros(T - 1)
    for c0 in COL_MANO.values():
        bloque = seq[:, c0:c0 + N_MANO * COORDS]
        presente = np.any(bloque != 0, axis=1)
        pts = bloque.reshape(T, N_MANO, COORDS)[:, :, :2]
        d = np.linalg.norm(np.diff(pts, axis=0), axis=2).mean(axis=1)
        ambos = presente[1:] & presente[:-1]
        suma[ambos] += d[ambos]
        cuenta[ambos] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        v = suma / cuenta
    v[cuenta == 0] = np.nan
    return v


def _altura(seq: np.ndarray) -> np.ndarray:
    """Por fotograma (T,): min(y_muñeca_izq, y_muñeca_der) de la postura (la más alta; y crece hacia
    abajo, 0 = línea de los hombros); NaN sin postura."""
    presente = np.any(seq[:, COL_POSTURA] != 0, axis=1)
    y_izq = seq[:, COL_MUNECA_POSE["izq"] + 1]
    y_der = seq[:, COL_MUNECA_POSE["der"] + 1]
    h = np.minimum(y_izq, y_der).astype(np.float64)
    h[~presente] = np.nan
    return h


def _sin_alguna_mano(seq: np.ndarray) -> np.ndarray:
    izq = ~np.any(seq[:, COL_MANO["izq"]:COL_MANO["izq"] + 63] != 0, axis=1)
    der = ~np.any(seq[:, COL_MANO["der"]:COL_MANO["der"] + 63] != 0, axis=1)
    return izq | der


def _nanmean(a) -> float:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0 or np.all(np.isnan(a)):
        return float("nan")
    return float(np.nanmean(a))


def rasgos_clip(seq: np.ndarray, K: int) -> dict[str, float]:
    """Rasgos de borde de un clip (T, ≥225) en el espacio normalizado por hombros, para un borde de
    K fotogramas. Velocidades en anchos de hombro por fotograma; alturas en anchos de hombro."""
    T = seq.shape[0]
    if T < 2:
        raise ValueError("un clip necesita al menos 2 fotogramas")
    K = min(K, T)
    vp, vm, h, sin = _velocidad_pose(seq), _velocidad_manos(seq), _altura(seq), _sin_alguna_mano(seq)
    c0, c1 = T // 4, T - T // 4              # mitad central del clip
    if c1 - c0 < 2:
        c0, c1 = 0, T
    return {
        "v_pose_ini": _nanmean(vp[:K - 1]), "v_pose_fin": _nanmean(vp[T - K:]),
        "v_pose_centro": _nanmean(vp[c0:c1 - 1]), "v_pose_max": _nanmean([np.nanmax(vp)]) if not np.all(np.isnan(vp)) else float("nan"),
        "v_manos_ini": _nanmean(vm[:K - 1]), "v_manos_fin": _nanmean(vm[T - K:]),
        "v_manos_centro": _nanmean(vm[c0:c1 - 1]),
        "h_ini": _nanmean(h[:K]), "h_fin": _nanmean(h[T - K:]), "h_centro": _nanmean(h[c0:c1]),
        "h_min": float(np.nanmin(h)) if not np.all(np.isnan(h)) else float("nan"),
        "sin_mano_ini": float(sin[:K].mean()), "sin_mano_fin": float(sin[T - K:].mean()),
        "sin_mano_centro": float(sin[c0:c1].mean()),
        "T": int(T),
    }


NOMBRES_RASGOS = ("v_pose_ini", "v_pose_fin", "v_pose_centro", "v_pose_max", "v_manos_ini", "v_manos_fin",
                  "v_manos_centro", "h_ini", "h_fin", "h_centro", "h_min", "sin_mano_ini", "sin_mano_fin",
                  "sin_mano_centro", "T")


def versiones_de(seq: np.ndarray) -> dict[str, np.ndarray]:
    """Clip íntegro y sus seis recortes, con nombre «lado_p»."""
    out = {"integro": seq}
    for lado in LADOS:
        for p in PROPORCIONES:
            out[f"{lado}_{p:.2f}"] = recortar(seq, p, lado)
    return out


VERSIONES = list(versiones_de(np.zeros((40, 225), dtype=np.float32)).keys())


def rasgos_corpus(corpus: dict, Ks=KS) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """rasgos[version][K][rasgo] = vector (N,) alineado con corpus['nombres']."""
    X, L = corpus["X"], corpus["lengths"]
    N = len(L)
    out = {v: {K: {r: np.full(N, np.nan) for r in NOMBRES_RASGOS} for K in Ks} for v in VERSIONES}
    for i in range(N):
        seq = np.asarray(X[i, :L[i], :225], dtype=np.float64)
        for v, s in versiones_de(seq).items():
            for K in Ks:
                r = rasgos_clip(s, K)
                for nombre in NOMBRES_RASGOS:
                    out[v][K][nombre][i] = r[nombre]
    return out


def perfil_velocidad(corpus: dict, n_bins: int = 20) -> dict[str, list[float]]:
    """Velocidad media de las muñecas de la postura por décimas de tiempo normalizado, por señante."""
    X, L, sen = corpus["X"], corpus["lengths"], corpus["senantes"]
    acum: dict[str, list] = {s: [np.zeros(n_bins), np.zeros(n_bins)] for s in sorted(set(sen))}
    for i in range(len(L)):
        seq = np.asarray(X[i, :L[i], :225], dtype=np.float64)
        v = _velocidad_pose(seq)
        t = (np.arange(len(v)) + 0.5) / len(v)
        b = np.minimum((t * n_bins).astype(int), n_bins - 1)
        for j in range(len(v)):
            if not np.isnan(v[j]):
                acum[sen[i]][0][b[j]] += v[j]
                acum[sen[i]][1][b[j]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return {s: (a / c).tolist() for s, (a, c) in acum.items()}


# ---------------------------------------------------------------------------
# Reglas: ajuste solo con entrenamiento, evaluación en prueba
# ---------------------------------------------------------------------------

def ajustar_umbrales(rasgos_integro_K: dict[str, np.ndarray], idx_train: np.ndarray, percentil: int = PERCENTIL) -> dict:
    """Umbrales de cada rasgo y borde a partir de los clips ÍNTEGROS de los índices de entrenamiento
    únicamente (percentil `percentil` para velocidades; 100 − percentil para la altura, que dispara
    por debajo). Solo recibe índices: nada del resto del corpus entra aquí."""
    idx_train = np.asarray(idx_train, dtype=np.int64)
    if idx_train.size == 0:
        raise ValueError("sin índices de entrenamiento no se puede ajustar")
    u = {}
    for rasgo in ("v_pose", "v_manos"):
        u[rasgo] = {b: float(np.nanpercentile(rasgos_integro_K[f"{rasgo}_{b}"][idx_train], percentil)) for b in ("ini", "fin")}
    u["h"] = {b: float(np.nanpercentile(rasgos_integro_K[f"h_{b}"][idx_train], 100 - percentil)) for b in ("ini", "fin")}
    u["percentil"] = int(percentil)
    u["n_train"] = int(idx_train.size)
    return u


def aplicar_regla(rasgos_K: dict[str, np.ndarray], umbrales: dict, rasgo: str) -> dict[str, np.ndarray]:
    """Marca cada clip: cortada_ini, cortada_fin y cortada (cualquiera). Un rasgo NaN no dispara."""
    def dispara(nombre: str, borde: str) -> np.ndarray:
        x = rasgos_K[f"{nombre}_{borde}"]
        t = umbrales[nombre][borde]
        with np.errstate(invalid="ignore"):
            return (x < t) if nombre == "h" else (x > t)
    out = {}
    for borde in ("ini", "fin"):
        if rasgo == "v_pose_o_h":
            m = dispara("v_pose", borde) | dispara("h", borde)
        elif rasgo in ("v_pose", "v_manos", "h"):
            m = dispara(rasgo, borde)
        else:
            raise ValueError(f"rasgo desconocido: {rasgo!r}")
        out[f"cortada_{borde}"] = np.nan_to_num(m.astype(float), nan=0.0).astype(bool)
    out["cortada"] = out["cortada_ini"] | out["cortada_fin"]
    return out


def evaluar_regla(rasgos: dict, K: int, rasgo: str, umbrales: dict, idx: np.ndarray) -> dict:
    """Sobre los índices dados: falsos positivos en íntegros y detección en cada versión truncada.
    «deteccion» = marcada como cortada por cualquiera de los dos bordes (lo que rechazaría el
    prototipo); «deteccion_lado» = marcada por el borde que efectivamente se recortó."""
    idx = np.asarray(idx, dtype=np.int64)
    m = aplicar_regla(rasgos["integro"][K], umbrales, rasgo)
    out = {"n": int(idx.size),
           "fp_integros": float(m["cortada"][idx].mean()),
           "fp_integros_ini": float(m["cortada_ini"][idx].mean()),
           "fp_integros_fin": float(m["cortada_fin"][idx].mean()),
           "deteccion": {}, "deteccion_lado": {}}
    for v in VERSIONES:
        if v == "integro":
            continue
        lado = v.split("_")[0]
        mv = aplicar_regla(rasgos[v][K], umbrales, rasgo)
        out["deteccion"][v] = float(mv["cortada"][idx].mean())
        out["deteccion_lado"][v] = float(mv["cortada_ini" if lado == "inicio" else "cortada_fin"][idx].mean())
    out["deteccion_media"] = float(np.mean(list(out["deteccion"].values())))
    return out


def seleccionar_config(rasgos: dict, idx_train: np.ndarray, Ks=KS, rasgos_cand=RASGOS,
                       percentil: int = PERCENTIL, empate: float = EMPATE) -> tuple[dict, list[dict]]:
    """Elige (K, rasgo) SOLO con entrenamiento: mayor detección media sobre los truncados de
    entrenamiento con umbrales ajustados en entrenamiento. Empate (< `empate`): menor K y luego el
    orden de `rasgos_cand`. Devuelve (elegida, rejilla)."""
    rejilla = []
    for K in Ks:
        u = ajustar_umbrales(rasgos["integro"][K], idx_train, percentil)
        for rasgo in rasgos_cand:
            e = evaluar_regla(rasgos, K, rasgo, u, idx_train)
            rejilla.append({"K": int(K), "rasgo": rasgo, "umbrales": u, "train": e})
    mejor = max(r["train"]["deteccion_media"] for r in rejilla)
    candidatas = [r for r in rejilla if r["train"]["deteccion_media"] >= mejor - empate]
    orden = {r: i for i, r in enumerate(rasgos_cand)}
    elegida = min(candidatas, key=lambda r: (r["K"], orden[r["rasgo"]]))
    return elegida, rejilla


# ---------------------------------------------------------------------------
# Modelo «manos»: daño con clips truncados
# ---------------------------------------------------------------------------

def _cargar_modelo(tf, ruta):
    try:
        return tf.keras.models.load_model(str(ruta))
    except ValueError:
        return tf.keras.models.load_model(str(ruta), safe_mode=False)


def entrada_modelo(seqs: list[np.ndarray], resample_sequence, add_velocity, seq_length: int = SEQ_LENGTH) -> np.ndarray:
    """Secuencias (T_i, 225) → (N, 30, 252): columnas «manos», remuestreo y velocidades, como el entrenador."""
    X = np.zeros((len(seqs), seq_length, COLS_MANOS.size), dtype=np.float32)
    for i, s in enumerate(seqs):
        X[i] = resample_sequence(np.asarray(s[:, COLS_MANOS], dtype=np.float32), seq_length)
    return add_velocity(X)


def metricas_prediccion(prob: np.ndarray, y: np.ndarray, umbral: float = UMBRAL_CONFIANZA) -> dict:
    pred = prob.argmax(axis=1)
    conf = prob.max(axis=1)
    ok = pred == y
    top3 = np.argsort(prob, axis=1)[:, ::-1][:, :3]
    acept = conf >= umbral
    return {
        "n": int(len(y)),
        "top1": float(ok.mean()),
        "top3": float(np.mean([y[i] in top3[i] for i in range(len(y))])),
        "confianza_media": float(conf.mean()),
        "confianza_mediana": float(np.median(conf)),
        "rechazo_umbral": float((~acept).mean()),
        "exactitud_aceptadas": float(ok[acept].mean()) if acept.any() else None,
        "aceptadas_ok": int((ok & acept).sum()),
        "aceptadas_mal": int((~ok & acept).sum()),
        "umbral": float(umbral),
    }


def evaluar_modelo_pliegue(tf, ruta_modelo: Path, corpus: dict, idx_test: np.ndarray, resample_sequence, add_velocity) -> dict:
    """Predicciones del modelo del pliegue sobre las muestras de prueba, íntegras y truncadas."""
    modelo = _cargar_modelo(tf, ruta_modelo)
    X, L, y = corpus["X"], corpus["lengths"], corpus["y"]
    seqs = [np.asarray(X[i, :L[i], :225], dtype=np.float32) for i in idx_test]
    salida = {"por_version": {}, "predichas_integro": None, "confianza": {}, "acierto": {}}
    for v in VERSIONES:
        lado_p = None if v == "integro" else (v.split("_")[0], float(v.split("_")[1]))
        entradas = seqs if lado_p is None else [recortar(s, lado_p[1], lado_p[0]) for s in seqs]
        prob = np.asarray(modelo.predict(entrada_modelo(entradas, resample_sequence, add_velocity), verbose=0, batch_size=256))
        salida["por_version"][v] = metricas_prediccion(prob, y[idx_test])
        salida["confianza"][v] = prob.max(axis=1)
        salida["acierto"][v] = prob.argmax(axis=1) == y[idx_test]
        if v == "integro":
            salida["predichas_integro"] = prob.argmax(axis=1)
    tf.keras.backend.clear_session()
    return salida


def control_predicciones(loso_dir: Path, k: int, nombres_test: list[str], predichas: np.ndarray) -> dict:
    """Las predicciones recalculadas sobre los clips íntegros deben coincidir muestra a muestra con
    pliegue_k/predicciones.json y el top-1 con resultados_loso.json."""
    guardadas = json.loads((loso_dir / f"pliegue_{k}" / "predicciones.json").read_text(encoding="utf-8"))
    por_nombre = {r["muestra"]: int(r["predicha"]) for r in guardadas}
    faltan = [n for n in nombres_test if n not in por_nombre]
    if faltan:
        raise SystemExit(f"pliegue {k}: {len(faltan)} muestras de prueba sin predicción guardada (p. ej. {faltan[0]})")
    coinciden = int(sum(int(por_nombre[n] == int(p)) for n, p in zip(nombres_test, predichas)))
    res = json.loads((loso_dir / "resultados_loso.json").read_text(encoding="utf-8"))
    top1_guardado = float(next(p["top1"] for p in res["pliegues"] if p["pliegue"] == k))
    reales = np.array([int(n.split("_")[0]) for n in nombres_test])
    top1_recalc = float((predichas == reales).mean())
    return {"muestras": len(nombres_test), "predicciones_coinciden": coinciden,
            "top1_guardado": top1_guardado, "top1_recalculado": top1_recalc,
            "ok": coinciden == len(nombres_test) and abs(top1_guardado - top1_recalc) < 1e-9}


BORDES_REGLA = {"ambos": "cortada", "ini": "cortada_ini", "fin": "cortada_fin"}


def combinar_regla_y_umbral(modelo_res: dict, rasgos: dict, K: int, rasgo: str, umbrales: dict, idx_test: np.ndarray,
                            borde: str = "ambos") -> dict:
    """Sobre las versiones truncadas del señante de prueba: qué rechaza la regla, qué rechaza el umbral
    de confianza y cómo quedan las que pasan ambos filtros (lo que el prototipo diría en voz alta).
    borde: «ambos» (la regla del criterio), «ini» o «fin» (solo ese borde; análisis POST HOC)."""
    if borde not in BORDES_REGLA:
        raise ValueError(f"borde desconocido: {borde!r}")
    out = {}
    for v in VERSIONES:
        marca = aplicar_regla(rasgos[v][K], umbrales, rasgo)[BORDES_REGLA[borde]][idx_test]
        conf = modelo_res["confianza"][v]
        ok = modelo_res["acierto"][v]
        acept = conf >= UMBRAL_CONFIANZA
        pasan = acept & ~marca
        out[v] = {
            "n": int(len(ok)),
            "rechaza_regla": float(marca.mean()),
            "rechaza_umbral": float((~acept).mean()),
            "rechaza_alguno": float((marca | ~acept).mean()),
            "pasan_ambos": int(pasan.sum()),
            "exactitud_pasan_ambos": float(ok[pasan].mean()) if pasan.any() else None,
            "errores_en_voz_alta_sin_regla": int((~ok & acept).sum()),
            "errores_en_voz_alta_con_regla": int((~ok & pasan).sum()),
            "aciertos_en_voz_alta_sin_regla": int((ok & acept).sum()),
            "aciertos_en_voz_alta_con_regla": int((ok & pasan).sum()),
        }
    return out


def utilidad(aciertos: int, errores: int, c: float = 1.0) -> float:
    """U = aciertos dichos en voz alta − c · errores dichos en voz alta (la función de D-09, c = 1)."""
    return float(aciertos - c * errores)


def umbrales_por_segundo(umbrales: dict, K: int, fps_corpus: float = FPS_CORPUS) -> dict:
    """Pasa los umbrales del corpus (por fotograma a 24 fps) a unidades independientes de la tasa:
    velocidades en anchos de hombro por SEGUNDO, borde en SEGUNDOS (K fotogramas = K/fps s). La altura
    no depende de la tasa. Es lo que necesitaría el prototipo, cuyo bucle no corre a 24 fps."""
    return {
        "v_pose_por_s": {b: umbrales["v_pose"][b] * fps_corpus for b in ("ini", "fin")},
        "v_manos_por_s": {b: umbrales["v_manos"][b] * fps_corpus for b in ("ini", "fin")},
        "h": dict(umbrales["h"]),
        "borde_s": K / fps_corpus,
        "K_corpus": int(K),
        "fps_corpus": float(fps_corpus),
    }


def umbrales_a_fps(por_segundo: dict, fps: float, K_min: int = 3) -> dict:
    """Umbrales por segundo → por fotograma a la tasa `fps` del prototipo, y K = max(K_min,
    round(borde_s · fps)) fotogramas de borde."""
    if fps <= 0:
        raise ValueError("fps debe ser positiva")
    return {
        "fps": float(fps),
        "K": int(max(K_min, round(por_segundo["borde_s"] * fps))),
        "v_pose": {b: por_segundo["v_pose_por_s"][b] / fps for b in ("ini", "fin")},
        "v_manos": {b: por_segundo["v_manos_por_s"][b] / fps for b in ("ini", "fin")},
        "h": dict(por_segundo["h"]),
    }


# ---------------------------------------------------------------------------
# Resúmenes descriptivos
# ---------------------------------------------------------------------------

def resumen(valores: np.ndarray) -> dict:
    v = np.asarray(valores, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "media": float(v.mean()), "p5": float(np.percentile(v, 5)),
            "p25": float(np.percentile(v, 25)), "mediana": float(np.median(v)),
            "p75": float(np.percentile(v, 75)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def resumen_por_grupo(valores: np.ndarray, grupos: list[str]) -> dict[str, dict]:
    g = np.asarray(grupos)
    return {str(x): resumen(valores[g == x]) for x in sorted(set(grupos))}


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # orden fijo: señantes 0000..0004


def _figuras(salida: Path, corpus: dict, perfil: dict, rasgos: dict, res: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984"})
    hechas = []
    senantes = sorted(set(corpus["senantes"]))
    sen = np.asarray(corpus["senantes"])
    color = {s: PALETA[i] for i, s in enumerate(senantes)}

    # Fig. 1: perfil de velocidad en tiempo normalizado
    fig, ax = plt.subplots(figsize=(7, 3.4))
    n_bins = len(next(iter(perfil.values())))
    x = (np.arange(n_bins) + 0.5) / n_bins
    for s in senantes:
        ax.plot(x, np.asarray(perfil[s]) * FPS_CORPUS, color=color[s], lw=2, label=f"señante {s}")
    ax.set_xlabel("tiempo normalizado del clip (0 = primer fotograma, 1 = último)")
    ax.set_ylabel("velocidad media de las muñecas\n(anchos de hombro / s)")
    ax.set_title("Perfil de velocidad de las muñecas de la postura, LSC50 re-extraído (24 fps)")
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    f = salida / "fig1_perfil_velocidad.png"
    fig.savefig(f, dpi=150); plt.close(fig); hechas.append(f.name)

    # Fig. 2: distribuciones de los rasgos de borde (K = 5) por señante
    K = 5
    R = rasgos["integro"][K]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6))
    paneles = [("v_pose_ini", "velocidad muñecas, primeros 5 fot. (anchos/s)", FPS_CORPUS),
               ("v_pose_fin", "velocidad muñecas, últimos 5 fot. (anchos/s)", FPS_CORPUS),
               ("h_ini", "altura muñeca más alta, primeros 5 fot.\n(anchos de hombro bajo la línea de hombros)", 1.0),
               ("h_fin", "altura muñeca más alta, últimos 5 fot.\n(anchos de hombro bajo la línea de hombros)", 1.0)]
    for ax, (nombre, etiqueta, esc) in zip(axes.ravel(), paneles):
        datos = [R[nombre][sen == s] * esc for s in senantes]
        datos = [d[~np.isnan(d)] for d in datos]
        bp = ax.boxplot(datos, tick_labels=senantes, widths=0.55, patch_artist=True, showfliers=True,
                        flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})
        for patch, s in zip(bp["boxes"], senantes):
            patch.set_facecolor(color[s]); patch.set_alpha(0.55); patch.set_edgecolor("#52514e")
        for med in bp["medians"]:
            med.set_color("#0b0b0b")
        ax.set_ylabel(etiqueta, fontsize=8)
        ax.set_xlabel("señante")
        if nombre.startswith("h_"):
            ax.invert_yaxis()
    fig.suptitle("Rasgos de borde en los clips íntegros (K = 5), por señante", fontsize=10)
    fig.tight_layout()
    f = salida / "fig2_rasgos_borde_por_senante.png"
    fig.savefig(f, dpi=150); plt.close(fig); hechas.append(f.name)

    # Fig. 3: detección en el señante de prueba por nivel de recorte, por pliegue
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, lado in zip(axes, LADOS):
        for i, pl in enumerate(res["pliegues"]):
            e = pl["regla_elegida"]["test"]
            ys = [e["deteccion"][f"{lado}_{p:.2f}"] for p in PROPORCIONES]
            ax.plot([p * 100 for p in PROPORCIONES], ys, "-o", color=PALETA[i], lw=2, ms=6,
                    label=f"pliegue {pl['pliegue']} (prueba {pl['senante_test']}; FP {e['fp_integros']:.2f})")
        ax.set_xlabel(f"% del clip recortado al {lado}")
        ax.set_title(f"cortada al {lado}")
        ax.set_ylim(0, 1.02)
        ax.set_xticks([p * 100 for p in PROPORCIONES])
    axes[0].set_ylabel("fracción detectada como cortada\n(señante de prueba, regla elegida en entrenamiento)")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    f = salida / "fig3_deteccion_prueba.png"
    fig.savefig(f, dpi=150); plt.close(fig); hechas.append(f.name)

    # Fig. 4: daño al modelo
    if res.get("modelo"):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
        etiquetas = ["íntegro"] + [f"{l[:3]} {int(p * 100)}%" for l in LADOS for p in PROPORCIONES]
        xs = np.arange(len(VERSIONES))
        for i, pl in enumerate(res["modelo"]["pliegues"]):
            t1 = [pl["por_version"][v]["top1"] for v in VERSIONES]
            rz = [pl["por_version"][v]["rechazo_umbral"] for v in VERSIONES]
            axes[0].plot(xs, t1, "-o", color=PALETA[i], lw=2, ms=5, label=f"pliegue {pl['pliegue']}")
            axes[1].plot(xs, rz, "-o", color=PALETA[i], lw=2, ms=5, label=f"pliegue {pl['pliegue']}")
        for ax, tit in zip(axes, ("top-1 del modelo «manos» (señante de prueba)", "rechazo por confianza < 0,20")):
            ax.set_xticks(xs); ax.set_xticklabels(etiquetas, rotation=35, ha="right", fontsize=8)
            ax.set_ylim(0, 1.02); ax.set_title(tit, fontsize=9)
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        f = salida / "fig4_dano_modelo.png"
        fig.savefig(f, dpi=150); plt.close(fig); hechas.append(f.name)
    return hechas


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f} %".replace(".", ",")


def escribir_informe(salida: Path, res: dict, criterio_texto: str) -> Path:
    L = []
    L.append("# Señas cortadas en los bordes de la ventana: análisis, reglas y daño al modelo «manos»\n")
    L.append(f"Generado por `src/analisis_senas_cortadas.py` el {res['corrida']['inicio']} "
             f"(fin {res['corrida']['fin']}). Solo reporte: no cambia el prototipo ni ningún modelo, umbral o partición.\n")
    L.append("Todas las cifras están en `resultados.json` (la clave de origen va entre corchetes).\n")
    L.append("## 0. Orden de los hechos (criterio antes que resultado)\n")
    L.append(f"- `criterio.md` modificado por última vez: {res['corrida']['criterio_mtime']}.")
    L.append(f"- Inicio de esta corrida: {res['corrida']['inicio']}. [corrida]\n")
    L.append("## 1. Lo que dice el anteproyecto y lo que hace el prototipo\n")
    L.append("- §2.2.6 (09_metodologia.tex:192-193): «Las secuencias incompletas (sin inicio o fin claros dentro de la "
             "ventana temporal) se descartarán.» Y 193-196: la inferencia opera «mediante una ventana temporal de longitud "
             "fija activada por la persona usuaria (botón o tecla) que delimita el inicio y el fin de la seña; la detección "
             "automática de inicio y fin (segmentación basada en movimiento) se deja como posible mejora futura».")
    L.append("- El prototipo (src/translator.py, `_cerrar_ventana`), como secuencia incompleta (`rechazada_incompleta`), solo descarta por X («abortada») o por menos de "
             "`min_frames` fotogramas; no mira si la seña empezó o terminó dentro de la ventana (N-37; protocolo 4.a, "
             "CL-6b y R6).\n")
    L.append("## 2. Cómo empiezan y terminan los clips del corpus\n")
    d = res["corpus"]
    L.append(f"Corpus: {d['n_clips']} clips, longitud {d['longitud']['min']}-{d['longitud']['max']} fotogramas "
             f"(mediana {d['longitud']['mediana']}) a {FPS_CORPUS:g} fps. Velocidades en anchos de hombro por fotograma; "
             f"×24 da anchos de hombro por segundo. Altura = y de la muñeca más alta de la postura respecto a la línea de "
             f"los hombros (positivo = por debajo; 1 = un ancho de hombros). [corpus]\n")
    L.append("### 2.1 Bordes frente al centro (todos los clips íntegros)\n")
    L.append("| K | v_pose ini (mediana / p95) | v_pose fin (mediana / p95) | v_pose centro (mediana) | v_manos ini (mediana / p95) | v_manos fin (mediana / p95) | v_manos centro (mediana) | h ini (mediana) | h fin (mediana) | h centro (mediana) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for K in KS:
        g = d["global"][str(K)]
        L.append(f"| {K} | {_f(g['v_pose_ini']['mediana'], 4)} / {_f(g['v_pose_ini']['p95'], 4)} | "
                 f"{_f(g['v_pose_fin']['mediana'], 4)} / {_f(g['v_pose_fin']['p95'], 4)} | {_f(g['v_pose_centro']['mediana'], 4)} | "
                 f"{_f(g['v_manos_ini']['mediana'], 4)} / {_f(g['v_manos_ini']['p95'], 4)} | "
                 f"{_f(g['v_manos_fin']['mediana'], 4)} / {_f(g['v_manos_fin']['p95'], 4)} | {_f(g['v_manos_centro']['mediana'], 4)} | "
                 f"{_f(g['h_ini']['mediana'], 2)} | {_f(g['h_fin']['mediana'], 2)} | {_f(g['h_centro']['mediana'], 2)} |")
    L.append("")
    L.append(f"Cociente borde/centro de v_pose (mediana sobre clips, K = 5): inicio {_f(d['cociente_borde_centro']['5']['v_pose_ini'], 3)}, "
             f"fin {_f(d['cociente_borde_centro']['5']['v_pose_fin'], 3)}. [corpus.cociente_borde_centro]\n")
    L.append("### 2.2 Fotogramas sin alguna mano, según dónde caen (K = 5)\n")
    sm = d["sin_mano"]["5"]
    L.append(f"Fracción media de fotogramas sin alguna mano: primeros 5 {_pct(sm['ini']['media'])}, últimos 5 "
             f"{_pct(sm['fin']['media'])}, mitad central {_pct(sm['centro']['media'])}. Clips con v_manos_ini indefinido "
             f"(ninguna mano detectada en dos fotogramas seguidos del borde): {_pct(d['v_manos_indefinido']['5']['ini'])}; "
             f"v_manos_fin indefinido: {_pct(d['v_manos_indefinido']['5']['fin'])}. [corpus.sin_mano, corpus.v_manos_indefinido]\n")
    L.append("### 2.3 Por señante (clips íntegros, K = 5)\n")
    L.append("| señante | n | v_pose ini mediana / p95 | v_pose fin mediana / p95 | v_pose centro mediana | h ini mediana | h fin mediana | h centro mediana | sin mano ini | sin mano fin |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s, r in d["por_senante"]["5"].items():
        L.append(f"| {s} | {r['v_pose_ini']['n']} | {_f(r['v_pose_ini']['mediana'], 4)} / {_f(r['v_pose_ini']['p95'], 4)} | "
                 f"{_f(r['v_pose_fin']['mediana'], 4)} / {_f(r['v_pose_fin']['p95'], 4)} | {_f(r['v_pose_centro']['mediana'], 4)} | "
                 f"{_f(r['h_ini']['mediana'], 2)} | {_f(r['h_fin']['mediana'], 2)} | {_f(r['h_centro']['mediana'], 2)} | "
                 f"{_pct(r['sin_mano_ini']['media'])} | {_pct(r['sin_mano_fin']['media'])} |")
    L.append("")
    L.append("### 2.4 Por seña (K = 5): las 8 señas con mayor velocidad mediana en los bordes\n")
    L.append("| seña | v_pose ini mediana | v_pose fin mediana | h ini mediana | h fin mediana |")
    L.append("|---|---|---|---|---|")
    for s in d["senas_bordes_mas_rapidos"]["5"]:
        r = d["por_sena"]["5"][s]
        L.append(f"| {s} | {_f(r['v_pose_ini']['mediana'], 4)} | {_f(r['v_pose_fin']['mediana'], 4)} | {_f(r['h_ini']['mediana'], 2)} | {_f(r['h_fin']['mediana'], 2)} |")
    L.append("\nLa tabla completa de las 50 señas y de los tres K está en `resultados.json` [corpus.por_sena].\n")
    L.append("### 2.5 POST HOC, descriptivo: ¿el inicio de los clips cumple el criterio de reposo de su final?\n")
    L.append("No forma parte del criterio ni de la selección. Umbral de reposo tomado del FINAL de los 1000 clips (incluidas las 3 "
             "duplicadas que D-06 excluye de las particiones; percentil 95 de v_pose_fin; percentil 5 de h_fin) y aplicado al INICIO de los mismos clips. [corpus.inicio_juzgado_con_reposo_del_final]\n")
    L.append("| K | v reposo (p95 fin) | h reposo (p5 fin) | inicio en movimiento, por señante 0000/0001/0002/0003/0004 | inicio con mano alta, por señante | alguno de los dos, por señante |")
    L.append("|---|---|---|---|---|---|")
    for K in KS:
        q = d["inicio_juzgado_con_reposo_del_final"]["por_K"][str(K)]
        L.append(f"| {K} | {_f(q['v_reposo_fin_p95'], 4)} | {_f(q['h_reposo_fin_p5'], 2)} | "
                 f"{' / '.join(_pct(x) for x in q['inicio_en_movimiento'].values())} | "
                 f"{' / '.join(_pct(x) for x in q['inicio_con_mano_alta'].values())} | "
                 f"{' / '.join(_pct(x) for x in q['inicio_en_movimiento_o_mano_alta'].values())} |")
    L.append("")

    L.append("## 3. Reglas de detección: ajuste en entrenamiento, evaluación en el señante de prueba\n")
    L.append("Familia, umbrales (percentil 95 de los íntegros de ENTRENAMIENTO), selección de (K, rasgo) por detección "
             "media en los truncados de ENTRENAMIENTO y evaluación en el señante de prueba, tal como fija `criterio.md`. "
             "«detección» = marcada como cortada por cualquiera de los dos bordes (lo que rechazaría el prototipo); la "
             "detección por el borde correcto está en `resultados.json` [pliegues[].regla_elegida.test.deteccion_lado].\n")
    L.append("### 3.1 Configuración elegida por pliegue y resultado en prueba\n")
    L.append("| pliegue | prueba | K | rasgo | umbral ini | umbral fin | det. media train | FP íntegros prueba (ini / fin) | det. inicio 20/35/50 % | det. fin 20/35/50 % |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for pl in res["pliegues"]:
        e = pl["regla_elegida"]
        u = e["umbrales"]
        r = e["rasgo"]
        u_ini = u["v_pose"]["ini"] if r in ("v_pose", "v_pose_o_h") else u["v_manos"]["ini"] if r == "v_manos" else u["h"]["ini"]
        u_fin = u["v_pose"]["fin"] if r in ("v_pose", "v_pose_o_h") else u["v_manos"]["fin"] if r == "v_manos" else u["h"]["fin"]
        extra = f" (h: {_f(u['h']['ini'], 2)} / {_f(u['h']['fin'], 2)})" if r == "v_pose_o_h" else ""
        t = e["test"]
        L.append(f"| {pl['pliegue']} | {pl['senante_test']} | {e['K']} | {r} | {_f(u_ini, 4)}{extra} | {_f(u_fin, 4)} | "
                 f"{_f(e['train']['deteccion_media'])} | {_pct(t['fp_integros'])} ({_pct(t['fp_integros_ini'])} / {_pct(t['fp_integros_fin'])}) | "
                 f"{' / '.join(_pct(t['deteccion'][f'inicio_{p:.2f}']) for p in PROPORCIONES)} | "
                 f"{' / '.join(_pct(t['deteccion'][f'fin_{p:.2f}']) for p in PROPORCIONES)} |")
    L.append("")
    m = res["resumen_pliegues"]
    L.append(f"Media entre pliegues (señantes de prueba): FP íntegros {_pct(m['fp_integros_media'])} (rango "
             f"{_pct(m['fp_integros_min'])}-{_pct(m['fp_integros_max'])}); detección inicio "
             f"{' / '.join(_pct(m['deteccion_media'][f'inicio_{p:.2f}']) for p in PROPORCIONES)}; fin "
             f"{' / '.join(_pct(m['deteccion_media'][f'fin_{p:.2f}']) for p in PROPORCIONES)}. "
             f"Configuración mayoritaria: K = {m['config_mayoritaria']['K']}, rasgo {m['config_mayoritaria']['rasgo']} "
             f"({m['config_mayoritaria']['votos']} de {len(res['pliegues'])} pliegues). [resumen_pliegues]\n")
    L.append("### 3.2 Rejilla completa en el señante de prueba (solo para leer; la elección no la usa)\n")
    L.append("| pliegue | K | rasgo | det. media train | FP prueba | det. media prueba | det. ini 35 % | det. fin 35 % |")
    L.append("|---|---|---|---|---|---|---|---|")
    for pl in res["pliegues"]:
        for r in pl["rejilla"]:
            L.append(f"| {pl['pliegue']} | {r['K']} | {r['rasgo']} | {_f(r['train']['deteccion_media'])} | {_pct(r['test']['fp_integros'])} | "
                     f"{_f(r['test']['deteccion_media'])} | {_pct(r['test']['deteccion']['inicio_0.35'])} | {_pct(r['test']['deteccion']['fin_0.35'])} |")
    L.append("")
    L.append("### 3.3 Sensibilidad al percentil del umbral (configuración elegida de cada pliegue; no cambia la elección)\n")
    L.append("| pliegue | percentil | FP íntegros prueba | det. inicio 20/35/50 % | det. fin 20/35/50 % |")
    L.append("|---|---|---|---|---|")
    for pl in res["pliegues"]:
        for pc, e in pl["sensibilidad_percentil"].items():
            L.append(f"| {pl['pliegue']} | {pc} | {_pct(e['fp_integros'])} | "
                     f"{' / '.join(_pct(e['deteccion'][f'inicio_{p:.2f}']) for p in PROPORCIONES)} | "
                     f"{' / '.join(_pct(e['deteccion'][f'fin_{p:.2f}']) for p in PROPORCIONES)} |")
    L.append("")
    c = res["umbral_candidato_cinco_senantes"]
    L.append(f"### 3.4 Umbral candidato con los cinco señantes (configuración mayoritaria K = {c['K']}, rasgo {c['rasgo']}; "
             f"percentil {c['umbrales']['percentil']})\n")
    L.append(f"v_pose ini {_f(c['umbrales']['v_pose']['ini'], 4)} / fin {_f(c['umbrales']['v_pose']['fin'], 4)} anchos de hombro por "
             f"fotograma a 24 fps (= {_f(c['umbrales']['v_pose']['ini'] * FPS_CORPUS, 3)} / {_f(c['umbrales']['v_pose']['fin'] * FPS_CORPUS, 3)} "
             f"anchos/s); h ini {_f(c['umbrales']['h']['ini'], 3)} / fin {_f(c['umbrales']['h']['fin'], 3)}; v_manos ini "
             f"{_f(c['umbrales']['v_manos']['ini'], 4)} / fin {_f(c['umbrales']['v_manos']['fin'], 4)}. Es un CANDIDATO: se "
             f"calcula con todos los señantes, así que no tiene evaluación limpia; la adopción la decide el director. "
             f"[umbral_candidato_cinco_senantes]\n")
    ps = c["por_segundo"]; ej = c["ejemplo_30_4_fps"]
    L.append(f"En unidades independientes de la tasa (lo que usaría el prototipo): borde de {_f(ps['borde_s'], 3)} s; v_pose ini "
             f"{_f(ps['v_pose_por_s']['ini'], 3)} / fin {_f(ps['v_pose_por_s']['fin'], 3)} anchos de hombro por segundo; h ini "
             f"{_f(ps['h']['ini'], 3)} / fin {_f(ps['h']['fin'], 3)}. Ejemplo de conversión a {_f(ej['fps'], 1)} fot./s (tasa medida en "
             f"N-32): K = {ej['K']} fotogramas, v_pose ini {_f(ej['v_pose']['ini'], 4)} / fin {_f(ej['v_pose']['fin'], 4)} por fotograma. "
             f"La conversión supone que la velocidad por segundo no depende de la tasa de captura; el ruido del "
             f"extractor por fotograma sí puede depender de ella, y eso no se ha medido. [umbral_candidato_cinco_senantes.por_segundo]\n")

    if res.get("modelo"):
        L.append("## 4. Daño al modelo «manos» con clips truncados (señante de prueba de cada pliegue)\n")
        ctl = res["modelo"]["control"]
        L.append("Control previo: predicciones recalculadas sobre los clips íntegros frente a `pliegue_k/predicciones.json` "
                 "y `resultados_loso.json`: " + "; ".join(f"pliegue {k}: {v['predicciones_coinciden']}/{v['muestras']} iguales, "
                 f"top-1 {_f(v['top1_recalculado'])} = {_f(v['top1_guardado'])}" for k, v in ctl.items()) + ". [modelo.control]\n")
        L.append("### 4.1 Agregado de los 997 clips de prueba, por versión\n")
        L.append("| versión | top-1 | top-3 | confianza media | rechazo < 0,20 | exactitud aceptadas | aceptadas ok / mal |")
        L.append("|---|---|---|---|---|---|---|")
        for v in VERSIONES:
            a = res["modelo"]["agregado"][v]
            L.append(f"| {v} | {_f(a['top1'])} | {_f(a['top3'])} | {_f(a['confianza_media'])} | {_pct(a['rechazo_umbral'])} | "
                     f"{_f(a['exactitud_aceptadas'])} | {a['aceptadas_ok']} / {a['aceptadas_mal']} |")
        L.append("")
        L.append("### 4.2 Top-1 por pliegue y versión\n")
        L.append("| pliegue | " + " | ".join(VERSIONES) + " |")
        L.append("|---|" + "---|" * len(VERSIONES))
        for pl in res["modelo"]["pliegues"]:
            L.append(f"| {pl['pliegue']} | " + " | ".join(_f(pl["por_version"][v]["top1"]) for v in VERSIONES) + " |")
        L.append("")
        L.append("### 4.3 Regla elegida + umbral 0,20 sobre los clips truncados (agregado de los 5 señantes de prueba)\n")
        L.append("| versión | rechaza regla | rechaza umbral | rechaza alguno | pasan ambos | exactitud pasan ambos | errores en voz alta sin / con regla | aciertos en voz alta sin / con regla | U (c = 1) sin / con regla |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for v in VERSIONES:
            a = res["modelo"]["combinado_agregado"][v]
            L.append(f"| {v} | {_pct(a['rechaza_regla'])} | {_pct(a['rechaza_umbral'])} | {_pct(a['rechaza_alguno'])} | {a['pasan_ambos']} | "
                     f"{_f(a['exactitud_pasan_ambos'])} | {a['errores_en_voz_alta_sin_regla']} / {a['errores_en_voz_alta_con_regla']} | "
                     f"{a['aciertos_en_voz_alta_sin_regla']} / {a['aciertos_en_voz_alta_con_regla']} | "
                     f"{a['U_sin_regla']:.0f} / {a['U_con_regla']:.0f} |")
        L.append("\nU = aciertos − errores dichos en voz alta (función de D-09 con c = 1), solo como resumen; no decide nada aquí.")
        L.append("\n[modelo.agregado, modelo.pliegues, modelo.combinado_agregado]\n")
        L.append("### 4.4 POST HOC, descriptivo: la regla elegida aplicada solo a un borde\n")
        L.append("No forma parte del criterio. Misma regla y umbrales de cada pliegue, pero marcando solo por el borde "
                 "inicial o solo por el final, para ver de dónde sale el beneficio y de dónde el costo. [modelo.post_hoc_por_borde]\n")
        L.append("| versión | solo inicio: rechaza | errores sin / con | aciertos sin / con | U sin / con | solo fin: rechaza | errores sin / con | aciertos sin / con | U sin / con |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        ph = res["modelo"]["post_hoc_por_borde"]["agregado"]
        for v in VERSIONES:
            a, b = ph["ini"][v], ph["fin"][v]
            L.append(f"| {v} | {_pct(a['rechaza_regla'])} | {a['errores_en_voz_alta_sin_regla']} / {a['errores_en_voz_alta_con_regla']} | "
                     f"{a['aciertos_en_voz_alta_sin_regla']} / {a['aciertos_en_voz_alta_con_regla']} | {a['U_sin_regla']:.0f} / {a['U_con_regla']:.0f} | "
                     f"{_pct(b['rechaza_regla'])} | {b['errores_en_voz_alta_sin_regla']} / {b['errores_en_voz_alta_con_regla']} | "
                     f"{b['aciertos_en_voz_alta_sin_regla']} / {b['aciertos_en_voz_alta_con_regla']} | {b['U_sin_regla']:.0f} / {b['U_con_regla']:.0f} |")
        L.append("")

    L.append("## 5. Figuras\n")
    for f in res.get("figuras", []):
        L.append(f"- `{f}`")
    L.append("\n## 6. Opciones para el prototipo, recomendación y especificación\n")
    L.append("Están en `recomendacion.md` (misma carpeta), escrito a mano sobre estas cifras después de la corrida.\n")
    L.append("\n## Anexo: criterio fijado antes de correr (copia literal de `criterio.md`)\n")
    L.append(criterio_texto)
    destino = salida / "informe.md"
    destino.write_text("\n".join(L) + "\n", encoding="utf-8")
    return destino


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def _json_seguro(o):
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"no serializable: {type(o)}")


def limpiar(o):
    """NaN → None en toda la estructura (json.dumps escribiría el literal NaN, que no es JSON válido)."""
    if isinstance(o, dict):
        return {str(k): limpiar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpiar(v) for v in o]
    if isinstance(o, np.ndarray):
        return limpiar(o.tolist())
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--particion", type=Path, default=PARTICION)
    ap.add_argument("--loso-dir", type=Path, default=LOSO_DIR)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--criterio", type=Path, default=CRITERIO)
    ap.add_argument("--sin-modelo", action="store_true", help="no cargar TensorFlow ni los modelos")
    ap.add_argument("--sin-figuras", action="store_true")
    args = ap.parse_args(argv)

    inicio = dt.datetime.now()
    t_ini = time.perf_counter()
    if not args.criterio.exists():
        raise SystemExit(f"falta {_rel(args.criterio)}: el criterio se escribe ANTES de correr")
    criterio_mtime = dt.datetime.fromtimestamp(args.criterio.stat().st_mtime).isoformat(timespec="seconds")
    if args.criterio.stat().st_mtime > inicio.timestamp():
        raise SystemExit("criterio.md tiene una fecha posterior al inicio de la corrida")
    criterio_texto = args.criterio.read_text(encoding="utf-8")
    args.salida.mkdir(parents=True, exist_ok=True)

    print(f"[1] Corpus {_rel(args.processed)}")
    corpus = cargar_corpus(args.processed)
    particion = cargar_particion(args.particion)
    N = len(corpus["nombres"])
    pos = {n: i for i, n in enumerate(corpus["nombres"])}
    sen = np.asarray(corpus["senantes"])
    clase = np.asarray([n.split("_")[0] for n in corpus["nombres"]])
    print(f"    {N} clips, {corpus['X'].shape}, longitudes {corpus['lengths'].min()}-{corpus['lengths'].max()}")

    print("[2] Rasgos de borde (7 versiones × 3 K)")
    rasgos = rasgos_corpus(corpus)
    perfil = perfil_velocidad(corpus)
    Ri = rasgos["integro"]
    if not np.all(np.nanmedian(Ri[5]["h_ini"]) > np.nanmedian(Ri[5]["h_centro"])):
        print("    [AVISO] la altura de las muñecas en el borde inicial no es mayor (más abajo) que en el centro: revisar el eje y")
    corpus_res = {
        "n_clips": N,
        "longitud": {"min": int(corpus["lengths"].min()), "max": int(corpus["lengths"].max()),
                     "mediana": float(np.median(corpus["lengths"])), "media": float(corpus["lengths"].mean())},
        "fps": FPS_CORPUS,
        "unidades": "velocidad: anchos de hombro por fotograma (×24 = por segundo); altura: anchos de hombro por debajo de la línea de los hombros",
        "global": {str(K): {r: resumen(Ri[K][r]) for r in NOMBRES_RASGOS} for K in KS},
        "por_senante": {str(K): {s: {r: resumen(Ri[K][r][sen == s]) for r in NOMBRES_RASGOS} for s in sorted(set(sen))} for K in KS},
        "por_sena": {str(K): {c: {r: resumen(Ri[K][r][clase == c]) for r in ("v_pose_ini", "v_pose_fin", "v_pose_centro", "h_ini", "h_fin", "h_centro", "sin_mano_ini", "sin_mano_fin")}
                              for c in sorted(set(clase))} for K in KS},
        "sin_mano": {str(K): {b: resumen(Ri[K][f"sin_mano_{b}"]) for b in ("ini", "fin", "centro")} for K in KS},
        "v_manos_indefinido": {str(K): {b: float(np.isnan(Ri[K][f"v_manos_{b}"]).mean()) for b in ("ini", "fin")} for K in KS},
        "cociente_borde_centro": {},
        "senas_bordes_mas_rapidos": {},
        "perfil_velocidad_por_senante": {"bins": 20, "unidad": "anchos de hombro por fotograma", "valores": perfil},
        "excluidas_D06": particion.get("excluidas", {}),
    }
    # POST HOC, DESCRIPTIVO (no forma parte del criterio ni de la selección): ¿el inicio de los clips
    # cumple el mismo criterio de reposo que su final? Umbral de reposo = percentil 95 de v_pose_fin y
    # percentil 5 de h_fin sobre los 1000 clips (incluidas las 3 duplicadas de D-06); se aplica al INICIO de los mismos clips, por señante.
    corpus_res["inicio_juzgado_con_reposo_del_final"] = {"nota": "post hoc y descriptivo; no interviene en la elección de la regla", "por_K": {}}
    for K in KS:
        with np.errstate(invalid="ignore", divide="ignore"):
            ci = Ri[K]["v_pose_ini"] / Ri[K]["v_pose_centro"]
            cf = Ri[K]["v_pose_fin"] / Ri[K]["v_pose_centro"]
        corpus_res["cociente_borde_centro"][str(K)] = {"v_pose_ini": float(np.nanmedian(ci)), "v_pose_fin": float(np.nanmedian(cf))}
        v_rep = float(np.nanpercentile(Ri[K]["v_pose_fin"], PERCENTIL))
        h_rep = float(np.nanpercentile(Ri[K]["h_fin"], 100 - PERCENTIL))
        with np.errstate(invalid="ignore"):
            ini_mov = Ri[K]["v_pose_ini"] > v_rep
            ini_alto = Ri[K]["h_ini"] < h_rep
        corpus_res["inicio_juzgado_con_reposo_del_final"]["por_K"][str(K)] = {
            "v_reposo_fin_p95": v_rep, "h_reposo_fin_p5": h_rep,
            "inicio_en_movimiento": {s: float(ini_mov[sen == s].mean()) for s in sorted(set(sen))},
            "inicio_con_mano_alta": {s: float(ini_alto[sen == s].mean()) for s in sorted(set(sen))},
            "inicio_en_movimiento_o_mano_alta": {s: float((ini_mov | ini_alto)[sen == s].mean()) for s in sorted(set(sen))},
            "fin_en_movimiento_por_construccion": float((Ri[K]["v_pose_fin"] > v_rep).mean()),
        }
        med = {c: max(corpus_res["por_sena"][str(K)][c]["v_pose_ini"]["mediana"], corpus_res["por_sena"][str(K)][c]["v_pose_fin"]["mediana"])
               for c in corpus_res["por_sena"][str(K)]}
        corpus_res["senas_bordes_mas_rapidos"][str(K)] = sorted(med, key=lambda c: -med[c])[:8]

    print("[3] Reglas por pliegue (ajuste solo con entrenamiento)")
    pliegues_res = []
    for k, p in enumerate(particion["pliegues"]):
        entren, prueba, s_test = conjuntos_pliegue(particion, k)
        idx_tr = np.array([pos[n] for n in entren])
        idx_te = np.array([pos[n] for n in prueba])
        elegida, rejilla = seleccionar_config(rasgos, idx_tr)
        for r in rejilla:
            r["test"] = evaluar_regla(rasgos, r["K"], r["rasgo"], r["umbrales"], idx_te)
        elegida = next(r for r in rejilla if r["K"] == elegida["K"] and r["rasgo"] == elegida["rasgo"])
        sens = {}
        for pc in PERCENTILES_SENSIBILIDAD:
            u = ajustar_umbrales(rasgos["integro"][elegida["K"]], idx_tr, pc)
            sens[str(pc)] = evaluar_regla(rasgos, elegida["K"], elegida["rasgo"], u, idx_te)
            sens[str(pc)]["umbrales"] = u
        pliegues_res.append({"pliegue": k, "senante_test": s_test, "n_train": int(idx_tr.size), "n_test": int(idx_te.size),
                             "regla_elegida": elegida, "rejilla": rejilla, "sensibilidad_percentil": sens})
        print(f"    pliegue {k} (prueba {s_test}): K={elegida['K']} rasgo={elegida['rasgo']} det. train {elegida['train']['deteccion_media']:.3f} "
              f"| prueba FP {elegida['test']['fp_integros']:.3f} det. media {elegida['test']['deteccion_media']:.3f}")

    votos = Counter((pl["regla_elegida"]["K"], pl["regla_elegida"]["rasgo"]) for pl in pliegues_res)
    (K_may, r_may), n_votos = min(votos.items(), key=lambda kv: (-kv[1], kv[0][0], RASGOS.index(kv[0][1])))
    resumen_pl = {
        "fp_integros_media": float(np.mean([pl["regla_elegida"]["test"]["fp_integros"] for pl in pliegues_res])),
        "fp_integros_min": float(min(pl["regla_elegida"]["test"]["fp_integros"] for pl in pliegues_res)),
        "fp_integros_max": float(max(pl["regla_elegida"]["test"]["fp_integros"] for pl in pliegues_res)),
        "deteccion_media": {v: float(np.mean([pl["regla_elegida"]["test"]["deteccion"][v] for pl in pliegues_res])) for v in VERSIONES if v != "integro"},
        "deteccion_lado_media": {v: float(np.mean([pl["regla_elegida"]["test"]["deteccion_lado"][v] for pl in pliegues_res])) for v in VERSIONES if v != "integro"},
        "config_mayoritaria": {"K": int(K_may), "rasgo": r_may, "votos": int(n_votos),
                               "regla_desempate": "más votos; luego menor K; luego orden de RASGOS"},
        "umbrales_por_pliegue": [{"pliegue": pl["pliegue"], "K": pl["regla_elegida"]["K"], "rasgo": pl["regla_elegida"]["rasgo"],
                                  "umbrales": pl["regla_elegida"]["umbrales"]} for pl in pliegues_res],
    }
    todos = np.arange(N)
    excluidas = set(particion.get("excluidas", {}))
    todos = np.array([i for i in todos if corpus["nombres"][i] not in excluidas])
    candidato = {"K": int(K_may), "rasgo": r_may, "n_clips": int(todos.size),
                 "umbrales": ajustar_umbrales(rasgos["integro"][K_may], todos, PERCENTIL),
                 "nota": "calculado con los cinco señantes (sin las 3 duplicadas de D-06); sin evaluación limpia; candidato, no vigente"}
    candidato["por_segundo"] = umbrales_por_segundo(candidato["umbrales"], K_may)
    candidato["ejemplo_30_4_fps"] = umbrales_a_fps(candidato["por_segundo"], 30.4)
    candidato["ejemplo_30_4_fps"]["fuente_fps"] = "tasa del bucle medida en N-32 (30,4 fot./s); solo ilustra la conversión"
    archivo_candidato = {
        "estado": "CANDIDATO, no vigente: la adopción la decide el estudiante con el director (D-nn pendiente)",
        "uso_previsto": "aviso de seña posiblemente cortada en los bordes de la ventana (opción B de recomendacion.md)",
        "regla": "aviso_inicio si v_pose_ini_por_s > v_pose_por_s.ini o h_ini < h.ini; aviso_fin si v_pose_fin_por_s > "
                 "v_pose_por_s.fin o h_fin < h.fin; un rasgo indefinido (sin postura) no dispara",
        "definiciones": {"v_pose": "media sobre los pares consecutivos del borde y sobre las muñecas 15 y 16 de la postura "
                                   "de ‖Δxy‖/Δt, en el espacio normalizado por hombros (anchos de hombro por segundo)",
                         "h": "media sobre los fotogramas del borde de min(y_muñeca_izq, y_muñeca_der) de la postura, "
                              "normalizada por hombros (y crece hacia abajo; 0 = línea de los hombros)",
                         "borde": "los primeros / últimos max(3, round(borde_s · fps)) fotogramas de la ventana"},
        "borde_s": candidato["por_segundo"]["borde_s"],
        "v_pose_por_s": candidato["por_segundo"]["v_pose_por_s"],
        "h": candidato["por_segundo"]["h"],
        "percentil": PERCENTIL,
        "origen": {"script": "src/analisis_senas_cortadas.py", "processed": _rel(args.processed),
                   "n_clips": candidato["n_clips"], "K_corpus": int(K_may), "fps_corpus": FPS_CORPUS,
                   "resultados": _rel(args.salida / "resultados.json") + " [umbral_candidato_cinco_senantes]"},
        "limites": ["umbrales medidos en los clips del corpus LSC50, no en ventanas del prototipo abiertas con tecla",
                    f"FP de la regla elegida en cada pliegue sobre clips íntegros de señantes no vistos entre {_pct(resumen_pl['fp_integros_min'])} y {_pct(resumen_pl['fp_integros_max'])} "
                    f"según el pliegue (resultados.json, pliegues[].regla_elegida.test)",
                    "la altura depende de la postura de reposo; el protocolo debe fijarla (PD-10)"],
    }

    res = {
        "corrida": {"inicio": inicio.isoformat(timespec="seconds"), "fin": None, "criterio": _rel(args.criterio),
                    "criterio_mtime": criterio_mtime, "script": "src/analisis_senas_cortadas.py",
                    "processed": _rel(args.processed), "particion": _rel(args.particion), "loso_dir": _rel(args.loso_dir),
                    "parametros": {"K": list(KS), "proporciones": list(PROPORCIONES), "percentil": PERCENTIL,
                                   "percentiles_sensibilidad": list(PERCENTILES_SENSIBILIDAD), "umbral_confianza": UMBRAL_CONFIANZA,
                                   "rasgos": list(RASGOS), "empate": EMPATE, "seq_length": SEQ_LENGTH},
                    "definiciones": {"deteccion": "clip truncado marcado como cortada por cualquiera de los dos bordes",
                                     "deteccion_lado": "marcado por el borde que efectivamente se recortó",
                                     "fp_integros": "clip íntegro marcado como cortada por cualquiera de los dos bordes",
                                     "entrenamiento_del_pliegue": "train ∪ val de loso_5_pliegues.json (cuatro señantes)",
                                     "prueba_del_pliegue": "test de loso_5_pliegues.json (el quinto señante)"}},
        "corpus": corpus_res,
        "pliegues": pliegues_res,
        "resumen_pliegues": resumen_pl,
        "umbral_candidato_cinco_senantes": candidato,
        "modelo": None,
        "figuras": [],
    }

    if not args.sin_modelo:
        print("[4] Modelo «manos» por pliegue con clips truncados")
        import tensorflow as tf
        from src.train_final_100 import add_velocity, resample_sequence
        modelo_res = {"pliegues": [], "control": {}, "agregado": {}, "combinado_agregado": {},
                      "post_hoc_por_borde": {"nota": "POST HOC, descriptivo: la regla elegida de cada pliegue aplicada "
                                                     "solo a un borde; no interviene en ninguna elección", "agregado": {}}}
        conf_all = {v: [] for v in VERSIONES}
        ok_all = {v: [] for v in VERSIONES}
        comb_all = {v: [] for v in VERSIONES}
        borde_all = {b: {v: [] for v in VERSIONES} for b in ("ini", "fin")}
        for k, pl in enumerate(pliegues_res):
            entren, prueba, s_test = conjuntos_pliegue(particion, k)
            idx_te = np.array([pos[n] for n in prueba])
            ruta = args.loso_dir / f"pliegue_{k}" / "modelo.keras"
            r = evaluar_modelo_pliegue(tf, ruta, corpus, idx_te, resample_sequence, add_velocity)
            ctl = control_predicciones(args.loso_dir, k, prueba, r["predichas_integro"])
            modelo_res["control"][str(k)] = ctl
            if not ctl["ok"]:
                raise SystemExit(f"pliegue {k}: las predicciones recalculadas no reproducen las guardadas ({ctl}); "
                                 f"el preprocesamiento no es el del entrenador")
            e = pl["regla_elegida"]
            comb = combinar_regla_y_umbral(r, rasgos, e["K"], e["rasgo"], e["umbrales"], idx_te)
            for b in ("ini", "fin"):
                cb = combinar_regla_y_umbral(r, rasgos, e["K"], e["rasgo"], e["umbrales"], idx_te, borde=b)
                for v in VERSIONES:
                    borde_all[b][v].append(cb[v])
            modelo_res["pliegues"].append({"pliegue": k, "senante_test": s_test, "modelo": _rel(ruta),
                                           "por_version": r["por_version"], "combinado": comb})
            for v in VERSIONES:
                conf_all[v].append(r["confianza"][v]); ok_all[v].append(r["acierto"][v]); comb_all[v].append(comb[v])
            print(f"    pliegue {k}: control {ctl['predicciones_coinciden']}/{ctl['muestras']} | top-1 íntegro {r['por_version']['integro']['top1']:.3f} "
                  + " ".join(f"{v}:{r['por_version'][v]['top1']:.3f}" for v in VERSIONES if v != "integro"))
        for v in VERSIONES:
            conf = np.concatenate(conf_all[v]); ok = np.concatenate(ok_all[v]); acept = conf >= UMBRAL_CONFIANZA
            modelo_res["agregado"][v] = {"n": int(ok.size), "top1": float(ok.mean()),
                                         "top3": float(sum(pl["por_version"][v]["top3"] * pl["por_version"][v]["n"] for pl in modelo_res["pliegues"]) / ok.size),
                                         "confianza_media": float(conf.mean()), "confianza_mediana": float(np.median(conf)),
                                         "rechazo_umbral": float((~acept).mean()),
                                         "exactitud_aceptadas": float(ok[acept].mean()) if acept.any() else None,
                                         "aceptadas_ok": int((ok & acept).sum()), "aceptadas_mal": int((~ok & acept).sum())}
            n_tot = sum(c["n"] for c in comb_all[v])
            modelo_res["combinado_agregado"][v] = {
                "n": n_tot,
                "rechaza_regla": float(sum(c["rechaza_regla"] * c["n"] for c in comb_all[v]) / n_tot),
                "rechaza_umbral": float(sum(c["rechaza_umbral"] * c["n"] for c in comb_all[v]) / n_tot),
                "rechaza_alguno": float(sum(c["rechaza_alguno"] * c["n"] for c in comb_all[v]) / n_tot),
                "pasan_ambos": int(sum(c["pasan_ambos"] for c in comb_all[v])),
                "exactitud_pasan_ambos": (float(sum(c["aciertos_en_voz_alta_con_regla"] for c in comb_all[v]) / sum(c["pasan_ambos"] for c in comb_all[v]))
                                          if sum(c["pasan_ambos"] for c in comb_all[v]) else None),
                "errores_en_voz_alta_sin_regla": int(sum(c["errores_en_voz_alta_sin_regla"] for c in comb_all[v])),
                "errores_en_voz_alta_con_regla": int(sum(c["errores_en_voz_alta_con_regla"] for c in comb_all[v])),
                "aciertos_en_voz_alta_sin_regla": int(sum(c["aciertos_en_voz_alta_sin_regla"] for c in comb_all[v])),
                "aciertos_en_voz_alta_con_regla": int(sum(c["aciertos_en_voz_alta_con_regla"] for c in comb_all[v])),
            }
            modelo_res["combinado_agregado"][v]["U_sin_regla"] = utilidad(
                modelo_res["combinado_agregado"][v]["aciertos_en_voz_alta_sin_regla"], modelo_res["combinado_agregado"][v]["errores_en_voz_alta_sin_regla"])
            modelo_res["combinado_agregado"][v]["U_con_regla"] = utilidad(
                modelo_res["combinado_agregado"][v]["aciertos_en_voz_alta_con_regla"], modelo_res["combinado_agregado"][v]["errores_en_voz_alta_con_regla"])
            for b in ("ini", "fin"):
                lst = borde_all[b][v]
                n_b = sum(c["n"] for c in lst)
                a_sin = int(sum(c["aciertos_en_voz_alta_sin_regla"] for c in lst)); e_sin = int(sum(c["errores_en_voz_alta_sin_regla"] for c in lst))
                a_con = int(sum(c["aciertos_en_voz_alta_con_regla"] for c in lst)); e_con = int(sum(c["errores_en_voz_alta_con_regla"] for c in lst))
                modelo_res["post_hoc_por_borde"]["agregado"].setdefault(b, {})[v] = {
                    "n": n_b, "rechaza_regla": float(sum(c["rechaza_regla"] * c["n"] for c in lst) / n_b),
                    "aciertos_en_voz_alta_sin_regla": a_sin, "errores_en_voz_alta_sin_regla": e_sin,
                    "aciertos_en_voz_alta_con_regla": a_con, "errores_en_voz_alta_con_regla": e_con,
                    "U_sin_regla": utilidad(a_sin, e_sin), "U_con_regla": utilidad(a_con, e_con)}
        res["modelo"] = modelo_res

    if not args.sin_figuras:
        print("[5] Figuras")
        res["figuras"] = _figuras(args.salida, corpus, perfil, rasgos, res)

    res["corrida"]["fin"] = dt.datetime.now().isoformat(timespec="seconds")
    res["corrida"]["duracion_s"] = round(time.perf_counter() - t_ini, 1)
    (args.salida / "umbral_bordes_candidato.json").write_text(json.dumps(limpiar(archivo_candidato), indent=1, ensure_ascii=False), encoding="utf-8")
    (args.salida / "resultados.json").write_text(json.dumps(limpiar(res), indent=1, ensure_ascii=False, default=_json_seguro), encoding="utf-8")
    escribir_informe(args.salida, json.loads((args.salida / "resultados.json").read_text(encoding="utf-8")), criterio_texto)
    print(f"[6] Escrito {_rel(args.salida / 'resultados.json')} e informe.md ({res['corrida']['duracion_s']} s)")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sys.exit(main())
