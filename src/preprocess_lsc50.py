"""
Pipeline de preprocesamiento para el dataset LSC50.

Carga los CSVs de landmarks (body + manos), verifica la estructura,
analiza distribución de clases, detecta valores faltantes, normaliza
coordenadas y guarda los datos procesados.

Uso:
    python src/preprocess_lsc50.py
    python src/preprocess_lsc50.py --landmarks data/extracted/LSC50_holistic --salida data/processed_holistic

Entradas (D-05, 14-sep-2026). --landmarks acepta los dos árboles de CSV y detecta cuál es; el
resultado queda en metadata.json (`landmarks_dir`, `layout`):
  - «anidado»: <DIR>/BODY_LANDMARKS/, <DIR>/HANDS_LANDMARKS/LEFT_HAND_LANDMARKS/,
    <DIR>/HANDS_LANDMARKS/RIGHT_HAND_LANDMARKS/   (CSV publicados por LSC50, data/raw)
  - «plano»:   <DIR>/BODY_LANDMARKS/, <DIR>/LEFT_HAND_LANDMARKS/, <DIR>/RIGHT_HAND_LANDMARKS/
    (re-extracción propia con src/reextract_lsc50.py; METRICAS_FOTOGRAMA/ no se lee; FACE_LANDMARKS/
    solo con --rostro)
--rostro cejas_labios (D-08, 15-sep-2026): añade tras las manos un bloque de 60 puntos faciales
(cejas y labios; lista y orden en src/representacion.py), normalizado por hombros como el resto.
Solo con el árbol plano: los FACE_LANDMARKS publicados vienen de otra cámara y no están alineados
con cuerpo y manos. El bloque del rostro sigue la misma regla --mano-ausente que las manos.
Los CSV re-extraídos traen una fila de CEROS cuando un componente no se detectó (en los
publicados esa fila simplemente no existe). Un fotograma con la postura en ceros no tiene
hombros y no se puede normalizar: se aplica la misma regla del prototipo
(aplicar_regla_sin_postura) y se registra en metadata.json (`fotogramas_sin_postura`). Con los
CSV publicados (sin filas de ceros) la salida es idéntica a la de marzo. --salida es el
directorio de salida (por defecto data/processed).
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.representacion import ROSTRO_OPCIONES, ROSTRO_SUBCONJUNTOS  # noqa: E402

# ---------------------------------------------------------------------------
# Configuración de rutas y constantes
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_BASE = BASE_DIR / "data" / "raw" / "LSC50" / "LANDMARKS"
BODY_DIR = RAW_BASE / "BODY_LANDMARKS"
LEFT_HAND_DIR = RAW_BASE / "HANDS_LANDMARKS" / "LEFT_HAND_LANDMARKS"
RIGHT_HAND_DIR = RAW_BASE / "HANDS_LANDMARKS" / "RIGHT_HAND_LANDMARKS"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Landmarks por tipo
N_BODY = 33        # landmarks 0-32
N_HAND = 21        # landmarks 0-20 (por mano)
COORDS = 3         # x, y, z
N_BODY_FEATURES = N_BODY * COORDS    # 99
N_HAND_FEATURES = N_HAND * COORDS    # 63
N_TOTAL_FEATURES = N_BODY_FEATURES + 2 * N_HAND_FEATURES  # 225

# Índices de columna en el array concatenado (body + lh + rh)
# Body landmark 11 (hombro izquierdo): cols 33, 34, 35
# Body landmark 12 (hombro derecho):   cols 36, 37, 38
LEFT_SHOULDER_IDX = 11 * COORDS   # 33
RIGHT_SHOULDER_IDX = 12 * COORDS  # 36


def _rel(ruta: Path) -> str:
    """Ruta relativa al repositorio para imprimir y registrar; absoluta si está fuera de él."""
    ruta = Path(ruta)
    return str(ruta.relative_to(BASE_DIR)) if ruta.is_relative_to(BASE_DIR) else str(ruta)


def resolver_directorios(landmarks_dir: Path) -> tuple[str, dict[str, Path]]:
    """Detecta el árbol de --landmarks y devuelve (layout, {"body", "lh", "rh"} -> Path).

    "anidado": <DIR>/BODY_LANDMARKS, <DIR>/HANDS_LANDMARKS/{LEFT,RIGHT}_HAND_LANDMARKS
               (CSV publicados por LSC50 en data/raw).
    "plano":   <DIR>/BODY_LANDMARKS, <DIR>/{LEFT,RIGHT}_HAND_LANDMARKS
               (re-extracción propia, D-05: data/extracted/LSC50_holistic).
    """
    landmarks_dir = Path(landmarks_dir)
    arboles = {
        "anidado": {"body": landmarks_dir / "BODY_LANDMARKS",
                    "lh": landmarks_dir / "HANDS_LANDMARKS" / "LEFT_HAND_LANDMARKS",
                    "rh": landmarks_dir / "HANDS_LANDMARKS" / "RIGHT_HAND_LANDMARKS"},
        "plano": {"body": landmarks_dir / "BODY_LANDMARKS",
                  "lh": landmarks_dir / "LEFT_HAND_LANDMARKS",
                  "rh": landmarks_dir / "RIGHT_HAND_LANDMARKS"},
    }
    presentes = [nombre for nombre, d in arboles.items() if d["lh"].is_dir() or d["rh"].is_dir()]
    if len(presentes) > 1:
        raise SystemExit(f"[ERROR] {landmarks_dir}: contiene HANDS_LANDMARKS/ y "
                         f"LEFT/RIGHT_HAND_LANDMARKS/ a la vez; el arbol es ambiguo")
    if not presentes:
        raise SystemExit(f"[ERROR] {landmarks_dir}: no se encontro ni HANDS_LANDMARKS/"
                         f"LEFT_HAND_LANDMARKS/ (arbol anidado) ni LEFT_HAND_LANDMARKS/ (arbol plano)")
    layout = presentes[0]
    for clave, d in arboles[layout].items():
        if not d.is_dir():
            raise SystemExit(f"[ERROR] Directorio no encontrado ({clave}, arbol {layout}): {d}")
    dirs = dict(arboles[layout])
    # D-08: el rostro solo se ofrece en el árbol plano (extracción propia, misma cámara que cuerpo y
    # manos). Los FACE_LANDMARKS publicados (árbol anidado) vienen de COLOR_FACE y no están alineados.
    if layout == "plano" and (landmarks_dir / "FACE_LANDMARKS").is_dir():
        dirs["face"] = landmarks_dir / "FACE_LANDMARKS"
    return layout, dirs


# ---------------------------------------------------------------------------
# Etapa 1: Carga y verificación de estructura
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> tuple[int, int, int]:
    """Extrae sign_id, volunteer_id, repetition_id del nombre del archivo."""
    stem = Path(filename).stem  # '0005_0002_0001'
    parts = stem.split("_")
    return int(parts[0]), int(parts[1]), int(parts[2])


def load_single_sample(sample_name: str, dirs: dict[str, Path] | None = None,
                       rostro: str = "ninguno") -> np.ndarray | None:
    """Carga y merge de body + left_hand + right_hand (+ subconjunto facial) para una muestra.

    dirs: {"body", "lh", "rh"[, "face"]} -> Path (ver resolver_directorios); por defecto, los CSV publicados.
    rostro: "ninguno" o un subconjunto de src/representacion.ROSTRO_SUBCONJUNTOS (D-08).
    Retorna array float32 de shape (T, 225) o (T, 225 + 3 * puntos del rostro), o None si hay error.
    """
    if dirs is None:
        dirs = {"body": BODY_DIR, "lh": LEFT_HAND_DIR, "rh": RIGHT_HAND_DIR}
    body_path = dirs["body"] / f"{sample_name}.csv"
    lh_path = dirs["lh"] / f"{sample_name}.csv"
    rh_path = dirs["rh"] / f"{sample_name}.csv"

    face_path = dirs["face"] / f"{sample_name}.csv" if rostro != "ninguno" else None

    for p in (body_path, lh_path, rh_path) + ((face_path,) if face_path else ()):
        if not p.exists():
            print(f"  [WARN] Archivo faltante: {p}")
            return None

    body = pd.read_csv(body_path, index_col=0, dtype=np.float32)
    lh = pd.read_csv(lh_path, index_col=0, dtype=np.float32)
    rh = pd.read_csv(rh_path, index_col=0, dtype=np.float32)
    face = None
    if face_path:
        cols = [f"landmark_{i}_{c}" for i in ROSTRO_SUBCONJUNTOS[rostro] for c in ("x", "y", "z")]
        face = pd.read_csv(face_path, index_col=0, usecols=lambda c: c in cols or c.startswith("Unnamed") or c == "",
                           dtype=np.float32)
        faltan = [c for c in cols if c not in face.columns]
        if faltan:
            print(f"  [WARN] {sample_name}: FACE_LANDMARKS sin las columnas {faltan[:3]}")
            return None
        face = face[cols]

    # Verificar columnas
    if body.shape[1] != N_BODY_FEATURES:
        print(f"  [WARN] {sample_name}: body tiene {body.shape[1]} cols (esperado {N_BODY_FEATURES})")
        return None
    if lh.shape[1] != N_HAND_FEATURES:
        print(f"  [WARN] {sample_name}: left_hand tiene {lh.shape[1]} cols (esperado {N_HAND_FEATURES})")
        return None
    if rh.shape[1] != N_HAND_FEATURES:
        print(f"  [WARN] {sample_name}: right_hand tiene {rh.shape[1]} cols (esperado {N_HAND_FEATURES})")
        return None

    # Truncar al mínimo de frames si hay diferencia pequeña (<=2 frames)
    filas = [body.shape[0], lh.shape[0], rh.shape[0]] + ([face.shape[0]] if face is not None else [])
    min_frames = min(filas)
    max_frames = max(filas)
    if max_frames - min_frames > 2:
        print(f"  [WARN] {sample_name}: diferencia de frames muy grande — "
              f"body={body.shape[0]}, lh={lh.shape[0]}, rh={rh.shape[0]}")
        return None

    body = body.iloc[:min_frames]
    lh = lh.iloc[:min_frames]
    rh = rh.iloc[:min_frames]

    # Concatenar: body(99) + left_hand(63) + right_hand(63) = 225 [+ rostro(3 x puntos), D-08]
    partes = [body.values, lh.values, rh.values]
    if face is not None:
        partes.append(face.iloc[:min_frames].values)
    merged = np.concatenate(partes, axis=1)
    return merged


def load_all_samples(dirs: dict[str, Path] | None = None, rostro: str = "ninguno") -> dict:
    """Carga todas las muestras. Retorna dict con datos y metadatos."""
    if dirs is None:
        dirs = {"body": BODY_DIR, "lh": LEFT_HAND_DIR, "rh": RIGHT_HAND_DIR}
    csv_files = sorted(dirs["body"].glob("*.csv"))
    sample_names = [f.stem for f in csv_files]

    samples = {}
    anomalies = []
    total = len(sample_names)

    print(f"  Cargando {total} muestras...")
    for i, name in enumerate(sample_names):
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  Progreso: {i + 1}/{total}")

        data = load_single_sample(name, dirs, rostro)
        if data is None:
            anomalies.append(name)
            continue

        sign_id, vol_id, rep_id = parse_filename(name)
        samples[name] = {
            "data": data,
            "sign_id": sign_id,
            "volunteer_id": vol_id,
            "repetition_id": rep_id,
            "n_frames": data.shape[0],
        }

    if anomalies:
        print(f"\n  [!] {len(anomalies)} muestras con anomalias: {anomalies[:10]}...")
    else:
        print("  Sin anomalias detectadas.")

    return samples


def verify_structure(samples: dict) -> None:
    """Imprime resumen de la estructura de datos cargada."""
    n = len(samples)
    frames = [s["n_frames"] for s in samples.values()]
    features = {s["data"].shape[1] for s in samples.values()}

    print(f"\n  Total muestras cargadas: {n}")
    print(f"  Features por frame: {features}")
    print(f"  Frames — min: {min(frames)}, max: {max(frames)}, "
          f"media: {np.mean(frames):.1f}, mediana: {np.median(frames):.1f}, "
          f"std: {np.std(frames):.1f}")


# ---------------------------------------------------------------------------
# Etapa 2: Distribución de clases
# ---------------------------------------------------------------------------

def compute_class_distribution(samples: dict) -> pd.DataFrame:
    """Cuenta muestras por clase (sign_id)."""
    records = [{"sign_id": s["sign_id"], "volunteer_id": s["volunteer_id"]}
               for s in samples.values()]
    df = pd.DataFrame(records)
    dist = df.groupby("sign_id").agg(
        count=("sign_id", "size"),
        volunteers=("volunteer_id", "nunique"),
    ).reset_index()
    return dist


def plot_class_distribution(dist: pd.DataFrame, output_path: Path) -> None:
    """Genera gráfico de barras de distribución de clases."""
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2ecc71" if c == 20 else "#e74c3c" for c in dist["count"]]
    ax.bar(dist["sign_id"], dist["count"], color=colors, edgecolor="gray", linewidth=0.5)
    ax.axhline(y=20, color="black", linestyle="--", linewidth=1, label="Esperado (20)")
    ax.set_xlabel("Clase (sign_id)")
    ax.set_ylabel("Muestras")
    ax.set_title("Distribucion de clases — LSC50")
    ax.set_xticks(range(0, 50, 5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Grafico guardado: {output_path}")


# ---------------------------------------------------------------------------
# Etapa 3: Valores faltantes
# ---------------------------------------------------------------------------

def detect_missing_values(samples: dict) -> pd.DataFrame:
    """Detecta NaN y filas completamente en cero por muestra."""
    records = []
    for name, s in samples.items():
        data = s["data"]
        n_nan = int(np.isnan(data).sum())
        n_zero_rows = int((data == 0).all(axis=1).sum())

        # Por grupo de landmarks
        body = data[:, :N_BODY_FEATURES]
        lh = data[:, N_BODY_FEATURES:N_BODY_FEATURES + N_HAND_FEATURES]
        rh = data[:, N_BODY_FEATURES + N_HAND_FEATURES:N_TOTAL_FEATURES]
        face = data[:, N_TOTAL_FEATURES:]

        records.append({
            "sample": name,
            "sign_id": s["sign_id"],
            "n_frames": s["n_frames"],
            "nan_total": n_nan,
            "zero_rows_total": n_zero_rows,
            "nan_body": int(np.isnan(body).sum()),
            "nan_lh": int(np.isnan(lh).sum()),
            "nan_rh": int(np.isnan(rh).sum()),
            "zero_rows_body": int((body == 0).all(axis=1).sum()),
            "zero_rows_lh": int((lh == 0).all(axis=1).sum()),
            "zero_rows_rh": int((rh == 0).all(axis=1).sum()),
            "zero_rows_face": int((face == 0).all(axis=1).sum()) if face.shape[1] else 0,
        })

    return pd.DataFrame(records)


def treat_missing_values(samples: dict, report: pd.DataFrame) -> int:
    """Aplica interpolación lineal a frames con NaN. Retorna cantidad de muestras tratadas."""
    treated = 0
    problem_samples = report[report["nan_total"] > 0]["sample"].tolist()

    for name in problem_samples:
        data = samples[name]["data"]
        df = pd.DataFrame(data)
        df = df.interpolate(method="linear", axis=0, limit_direction="both")
        samples[name]["data"] = df.values.astype(np.float32)
        treated += 1

    return treated


def plot_missing_value_report(report: pd.DataFrame, output_path: Path) -> None:
    """Genera visualización del reporte de valores faltantes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # NaN por grupo
    nan_cols = ["nan_body", "nan_lh", "nan_rh"]
    nan_sums = report[nan_cols].sum()
    axes[0].bar(["Body", "Left Hand", "Right Hand"], nan_sums.values,
                color=["#3498db", "#e67e22", "#9b59b6"])
    axes[0].set_title("Total NaN por grupo de landmarks")
    axes[0].set_ylabel("Cantidad de NaN")

    # Zero rows por grupo
    zero_cols = ["zero_rows_body", "zero_rows_lh", "zero_rows_rh"]
    zero_sums = report[zero_cols].sum()
    axes[1].bar(["Body", "Left Hand", "Right Hand"], zero_sums.values,
                color=["#3498db", "#e67e22", "#9b59b6"])
    axes[1].set_title("Filas completamente en cero por grupo")
    axes[1].set_ylabel("Cantidad de filas")

    fig.suptitle("Reporte de valores faltantes — LSC50", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Grafico guardado: {output_path}")


# ---------------------------------------------------------------------------
# Etapa 4: Normalización de coordenadas
# ---------------------------------------------------------------------------

def aplicar_regla_sin_postura(samples: dict) -> dict:
    """Regla para los fotogramas SIN POSTURA (fila de BODY_LANDMARKS toda en ceros: el
    extractor no detectó la postura). Sin hombros no hay punto de referencia ni escala, y
    normalizar dividiría por la escala mínima (1e-6) disparando los valores.

    Regla (la MISMA del prototipo, src/translator.py, SignTranslator._preprocesar, comentario
    «Un fotograma sin pose no tiene hombros: sin referencia de normalización se deja entero en
    ceros» y `seq[sin_pose] = 0.0` justo antes de normalize_sequence): el fotograma COMPLETO
    (todas sus columnas: postura, manos y, si lo hay, rostro) se pone en ceros antes de normalizar.
    normalize_sample_vectorized lo conserva en ceros ((0 - 0) / max(0, 1e-6) = 0), sin NaN ni
    inf. El fotograma no se elimina ni se interpola: la secuencia conserva su longitud, como en
    el prototipo (que después remuestrea a 30 pasos).

    Los CSV publicados (marzo) no traen filas de ceros, así que con ellos no cambia nada.
    Modifica samples in-place y devuelve el registro que va a metadata.json."""
    por_muestra = {}
    for name, s in samples.items():
        data = s["data"]
        sin_postura = (data[:, :N_BODY_FEATURES] == 0).all(axis=1)
        n = int(sin_postura.sum())
        if n:
            data[sin_postura, :] = 0.0
            por_muestra[name] = n
    total = int(sum(por_muestra.values()))
    return {
        "criterio": "fila de postura (BODY_LANDMARKS, 99 columnas) toda en ceros",
        "regla": "el fotograma completo (todas las columnas: postura, manos y rostro si lo hay) se "
                 "pone en ceros antes de normalizar y la normalizacion lo conserva en ceros; no se "
                 "elimina ni se interpola. Misma regla que el prototipo: src/translator.py, "
                 "SignTranslator._preprocesar ('Un fotograma sin pose no tiene hombros'), "
                 "seq[sin_pose] = 0.0 antes de normalize_sequence",
        "filas_postura_en_ceros": total,
        "fotogramas_puestos_en_ceros": total,
        "muestras_afectadas": len(por_muestra),
        "por_muestra": por_muestra,
    }


MANO_AUSENTE_OPCIONES = ("ceros", "descartar", "mascara")


def _manos_ausentes(data: np.ndarray):
    """Máscaras booleanas (T,) de fotogramas con la mano izquierda / derecha toda en ceros."""
    lh = data[:, N_BODY_FEATURES:N_BODY_FEATURES + N_HAND_FEATURES]
    rh = data[:, N_BODY_FEATURES + N_HAND_FEATURES:N_TOTAL_FEATURES]
    return (lh == 0).all(axis=1), (rh == 0).all(axis=1)


def aplicar_regla_mano_ausente(samples: dict, modo: str) -> dict:
    """D-07 (14-sep-2026): tratamiento de los fotogramas en los que el extractor NO detectó una
    mano (fila de esa mano toda en ceros en la extracción propia; los CSV publicados no traen
    filas así porque los autores omitieron esos fotogramas, ver N-22).

      ceros      no hace nada: el fotograma se conserva con la mano en ceros y se normaliza como
                 cualquier punto (comportamiento de N-22; una mano ausente queda como el vector
                 constante -ref/escala).
      descartar  elimina el fotograma si falta CUALQUIERA de las dos manos (convención observada
                 en los CSV publicados: filas publicadas ≈ fotogramas con ambas manos). Se aplica
                 ANTES de normalizar. Una muestra nunca queda con menos de 1 fotograma: si todos
                 tuvieran una mano ausente, se conserva entera y se anota.
      mascara    conserva el fotograma; la normalización se hace igual y DESPUÉS el bloque de la
                 mano ausente se vuelve a poner en ceros (ver aplicar_mascara_mano_ausente). El
                 cero pasa a significar «sin dato» en el espacio normalizado.

    Devuelve el registro para metadata.json; en modo «mascara» guarda las máscaras en cada muestra
    (clave "mascara_manos") para aplicarlas tras normalizar."""
    if modo not in MANO_AUSENTE_OPCIONES:
        raise ValueError(f"modo desconocido: {modo}")
    total_izq = total_der = total_alguna = total_rostro = descartados = 0
    hay_rostro = False
    por_muestra = {}
    muestras_sin_recorte = []
    for name, s in samples.items():
        data = s["data"]
        izq, der = _manos_ausentes(data)
        alguna = izq | der
        total_izq += int(izq.sum()); total_der += int(der.sum()); total_alguna += int(alguna.sum())
        # D-08: el bloque del rostro (si lo hay) sigue la misma regla que las manos.
        sin_rostro = None
        if data.shape[1] > N_TOTAL_FEATURES:
            hay_rostro = True
            sin_rostro = (data[:, N_TOTAL_FEATURES:] == 0).all(axis=1)
            total_rostro += int(sin_rostro.sum())
            alguna = alguna | sin_rostro
        if not alguna.any():
            continue
        por_muestra[name] = int(alguna.sum())
        if modo == "descartar":
            if alguna.all():
                muestras_sin_recorte.append(name)
                continue
            s["data"] = data[~alguna]
            s["n_frames"] = int(s["data"].shape[0])   # lengths.npy y el padding salen de n_frames
            descartados += int(alguna.sum())
        elif modo == "mascara":
            s["mascara_manos"] = (izq.copy(), der.copy())
            if sin_rostro is not None:
                s["mascara_rostro"] = sin_rostro.copy()
    registro = {
        "modo": modo,
        "criterio": "fila de la mano (63 columnas) toda en ceros = mano no detectada",
        "fotogramas_sin_mano_izq": total_izq,
        "fotogramas_sin_mano_der": total_der,
        "fotogramas_con_alguna_mano_ausente": total_alguna,
        "fotogramas_descartados": descartados,
        "muestras_afectadas": len(por_muestra),
        "muestras_conservadas_enteras_por_no_tener_ningun_fotograma_con_ambas_manos": muestras_sin_recorte,
        "por_muestra": por_muestra,
    }
    if hay_rostro:
        registro["nota_rostro"] = ("con rostro, muestras_afectadas, por_muestra, muestras_conservadas_enteras_... y "
                                   "bloques_de_mano_enmascarados_tras_normalizar cuentan tambien los fotogramas sin rostro")
        registro["fotogramas_sin_rostro"] = total_rostro
        registro["criterio_rostro"] = ("bloque del rostro toda en ceros = rostro no detectado; misma regla que las manos "
                                       "(descartar: el fotograma se omite si falta una mano o el rostro)")
    return registro


def aplicar_mascara_mano_ausente(samples: dict) -> int:
    """Modo «mascara» de D-07: tras normalizar, vuelve a poner en ceros el bloque de cada mano que
    no se detectó. Devuelve cuántos bloques de mano se enmascararon."""
    n = 0
    for s in samples.values():
        if "mascara_manos" not in s:
            continue
        izq, der = s.pop("mascara_manos")
        d = s["data"]
        d[izq, N_BODY_FEATURES:N_BODY_FEATURES + N_HAND_FEATURES] = 0.0
        d[der, N_BODY_FEATURES + N_HAND_FEATURES:N_TOTAL_FEATURES] = 0.0
        n += int(izq.sum()) + int(der.sum())
    for s in samples.values():
        if "mascara_rostro" in s:
            sin_rostro = s.pop("mascara_rostro")
            s["data"][sin_rostro, N_TOTAL_FEATURES:] = 0.0
            n += int(sin_rostro.sum())
    return n


def normalize_sample(data: np.ndarray) -> np.ndarray:
    """Normaliza una muestra: centrado en hombros + escalado por ancho de hombros.

    Para cada frame:
      1. ref = punto medio entre hombro izq (body lm 11) y hombro der (body lm 12)
      2. scale = distancia euclidiana entre hombros en plano x-y
      3. Restar ref de todas las coordenadas
      4. Dividir por scale
    """
    normalized = data.copy()
    n_landmarks = N_TOTAL_FEATURES // COORDS  # 75

    for t in range(data.shape[0]):
        # Hombro izquierdo (x, y, z)
        ls_x = data[t, LEFT_SHOULDER_IDX]
        ls_y = data[t, LEFT_SHOULDER_IDX + 1]
        ls_z = data[t, LEFT_SHOULDER_IDX + 2]

        # Hombro derecho (x, y, z)
        rs_x = data[t, RIGHT_SHOULDER_IDX]
        rs_y = data[t, RIGHT_SHOULDER_IDX + 1]
        rs_z = data[t, RIGHT_SHOULDER_IDX + 2]

        # Punto de referencia (midpoint)
        ref_x = (ls_x + rs_x) / 2.0
        ref_y = (ls_y + rs_y) / 2.0
        ref_z = (ls_z + rs_z) / 2.0

        # Factor de escala (distancia entre hombros en x-y)
        scale = np.sqrt((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2)
        scale = max(scale, 1e-6)  # evitar división por cero

        # Aplicar a todos los landmarks
        for i in range(n_landmarks):
            idx = i * COORDS
            normalized[t, idx] = (data[t, idx] - ref_x) / scale
            normalized[t, idx + 1] = (data[t, idx + 1] - ref_y) / scale
            normalized[t, idx + 2] = (data[t, idx + 2] - ref_z) / scale

    return normalized


def normalize_sample_vectorized(data: np.ndarray) -> np.ndarray:
    """Versión vectorizada de normalize_sample (más rápida)."""
    T = data.shape[0]
    normalized = data.copy()

    # Extraer hombros: shape (T,)
    ls_x = data[:, LEFT_SHOULDER_IDX]
    ls_y = data[:, LEFT_SHOULDER_IDX + 1]
    ls_z = data[:, LEFT_SHOULDER_IDX + 2]
    rs_x = data[:, RIGHT_SHOULDER_IDX]
    rs_y = data[:, RIGHT_SHOULDER_IDX + 1]
    rs_z = data[:, RIGHT_SHOULDER_IDX + 2]

    # Punto de referencia: shape (T,)
    ref_x = (ls_x + rs_x) / 2.0
    ref_y = (ls_y + rs_y) / 2.0
    ref_z = (ls_z + rs_z) / 2.0

    # Factor de escala: shape (T,)
    scale = np.sqrt((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2)
    scale = np.maximum(scale, 1e-6)

    # Reshape data a (T, 75, 3)
    reshaped = normalized.reshape(T, -1, COORDS)

    # Restar referencia: broadcast (T, 1, 3)
    ref = np.stack([ref_x, ref_y, ref_z], axis=1)[:, np.newaxis, :]  # (T, 1, 3)
    reshaped -= ref

    # Dividir por escala: broadcast (T, 1, 1)
    reshaped /= scale[:, np.newaxis, np.newaxis]

    return reshaped.reshape(T, -1)


def normalize_all_samples(samples: dict) -> None:
    """Normaliza todas las muestras in-place."""
    scales = []
    for name, s in samples.items():
        data = s["data"]
        ls_x = data[:, LEFT_SHOULDER_IDX]
        ls_y = data[:, LEFT_SHOULDER_IDX + 1]
        rs_x = data[:, RIGHT_SHOULDER_IDX]
        rs_y = data[:, RIGHT_SHOULDER_IDX + 1]
        scale = np.sqrt((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2)
        scales.extend(scale.tolist())

        s["data"] = normalize_sample_vectorized(data)

    scales = np.array(scales)
    print(f"  Escala (ancho hombros) — min: {scales.min():.4f}, max: {scales.max():.4f}, "
          f"media: {scales.mean():.4f}, std: {scales.std():.4f}")


# ---------------------------------------------------------------------------
# Etapa 5: Guardar datos procesados
# ---------------------------------------------------------------------------

def save_processed_data(samples: dict, output_dir: Path, metadata_extra: dict | None = None) -> None:
    """Guarda datos procesados en formatos .npy y .json.

    metadata_extra: claves adicionales para metadata.json (origen de los CSV, layout, regla de
    fotogramas sin postura, valores faltantes). Van al final; las claves de marzo no cambian.
    """
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Ordenar muestras por nombre para consistencia
    sorted_names = sorted(samples.keys())
    n = len(sorted_names)

    # Recopilar datos
    all_data = []
    labels = np.zeros(n, dtype=np.int32)
    lengths = np.zeros(n, dtype=np.int32)

    for i, name in enumerate(sorted_names):
        s = samples[name]
        data = s["data"]
        all_data.append(data)
        labels[i] = s["sign_id"]
        lengths[i] = s["n_frames"]

        # Guardar muestra individual
        np.save(samples_dir / f"{name}.npy", data)

    # Crear array con padding
    max_frames = int(lengths.max())
    n_features = int(all_data[0].shape[1])
    X_padded = np.zeros((n, max_frames, n_features), dtype=np.float32)
    for i, data in enumerate(all_data):
        T = data.shape[0]
        X_padded[i, :T, :] = data

    # Guardar arrays
    np.save(output_dir / "X_padded.npy", X_padded)
    np.save(output_dir / "y_labels.npy", labels)
    np.save(output_dir / "lengths.npy", lengths)

    print(f"  X_padded.npy: shape {X_padded.shape}")
    print(f"  y_labels.npy: shape {labels.shape}, clases {len(set(labels))}")
    print(f"  lengths.npy: shape {lengths.shape}")
    print(f"  Muestras individuales: {n} archivos en {samples_dir}")

    # Generar nombres de features
    feature_names = []
    for i in range(N_BODY):
        for c in ("x", "y", "z"):
            feature_names.append(f"body_landmark_{i}_{c}")
    for i in range(N_HAND):
        for c in ("x", "y", "z"):
            feature_names.append(f"lh_landmark_{i}_{c}")
    for i in range(N_HAND):
        for c in ("x", "y", "z"):
            feature_names.append(f"rh_landmark_{i}_{c}")
    rostro = (metadata_extra or {}).get("rostro", {}).get("subconjunto", "ninguno")
    if rostro != "ninguno":
        for i in ROSTRO_SUBCONJUNTOS[rostro]:
            for c in ("x", "y", "z"):
                feature_names.append(f"face_landmark_{i}_{c}")
    assert len(feature_names) == n_features, (len(feature_names), n_features)

    # Metadata
    metadata = {
        "n_samples": n,
        "n_classes": 50,
        "n_features": n_features,
        "n_landmarks": n_features // 3,
        "max_frames": max_frames,
        "frame_stats": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
        },
        "landmark_groups": {
            "body": {"start_col": 0, "end_col": N_BODY_FEATURES - 1, "n_landmarks": N_BODY},
            "left_hand": {"start_col": N_BODY_FEATURES, "end_col": N_BODY_FEATURES + N_HAND_FEATURES - 1, "n_landmarks": N_HAND},
            "right_hand": {"start_col": N_BODY_FEATURES + N_HAND_FEATURES, "end_col": N_TOTAL_FEATURES - 1, "n_landmarks": N_HAND},
            **({"face": {"start_col": N_TOTAL_FEATURES, "end_col": n_features - 1,
                         "n_landmarks": (n_features - N_TOTAL_FEATURES) // 3}} if rostro != "ninguno" else {}),
        },
        "normalization": {
            "method": "shoulder_centered_scaled",
            "reference_point": "midpoint(body_landmark_11, body_landmark_12)",
            "scale_factor": "euclidean_distance(shoulders, xy_plane)",
        },
        "feature_names": feature_names,
        "sample_order": sorted_names,
    }
    for clave, valor in (metadata_extra or {}).items():
        if clave in metadata:
            raise ValueError(f"metadata_extra no puede redefinir la clave {clave!r}")
        metadata[clave] = valor

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  metadata.json guardado: {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parsear_argumentos(argv=None):
    parser = argparse.ArgumentParser(
        description="Preprocesamiento LSC50: CSV de puntos (body + manos) -> X_padded.npy, "
                    "y_labels.npy, lengths.npy, metadata.json y samples/.")
    parser.add_argument("--landmarks", type=str, default=str(RAW_BASE),
                        help="directorio con los CSV de puntos, en arbol anidado (BODY_LANDMARKS/, "
                             "HANDS_LANDMARKS/LEFT_HAND_LANDMARKS/, ...; CSV publicados) o plano "
                             "(BODY_LANDMARKS/, LEFT_HAND_LANDMARKS/, RIGHT_HAND_LANDMARKS/; "
                             f"re-extraccion D-05); se detecta solo (defecto {_rel(RAW_BASE)})")
    parser.add_argument("--salida", type=str, default=str(OUTPUT_DIR),
                        help=f"directorio de salida (defecto {_rel(OUTPUT_DIR)})")
    parser.add_argument("--rostro", choices=ROSTRO_OPCIONES, default="ninguno",
                        help="D-08: anadir un subconjunto de puntos faciales tras las manos (solo arbol plano de la "
                             "extraccion propia); cejas_labios = 20 de cejas + 40 de labios (defecto ninguno)")
    parser.add_argument("--mano-ausente", choices=MANO_AUSENTE_OPCIONES, default="ceros",
                    help="D-07: que hacer con los fotogramas en los que falta una mano (solo aparecen en la "
                         "extraccion propia): ceros = conservar y normalizar (defecto, N-22); descartar = omitir "
                         "el fotograma si falta cualquiera de las dos manos (convencion de los CSV publicados); "
                         "mascara = normalizar y luego poner en ceros el bloque de la mano ausente")
    return parser.parse_args(argv)


def main(argv=None):
    args = parsear_argumentos(argv)
    landmarks_dir = Path(args.landmarks).expanduser().resolve()
    output_dir = Path(args.salida).expanduser().resolve()

    print("=" * 60)
    print("  LSC50 — Pipeline de Preprocesamiento")
    print("=" * 60)

    # Verificar rutas y detectar el arbol de entrada (anidado = CSV publicados; plano = D-05)
    if not landmarks_dir.is_dir():
        print(f"[ERROR] Directorio no encontrado: {landmarks_dir}")
        sys.exit(1)
    layout, dirs = resolver_directorios(landmarks_dir)
    if args.rostro != "ninguno" and "face" not in dirs:
        sys.exit(f"[ERROR] --rostro {args.rostro} requiere el arbol plano de la extraccion propia con "
                 f"FACE_LANDMARKS/ (D-08); {landmarks_dir} es arbol {layout}"
                 + (" y sus FACE_LANDMARKS vienen de otra camara (COLOR_FACE, no alineada)" if layout == "anidado" else ""))
    print(f"  Entrada: {landmarks_dir} (arbol {layout})")
    if args.rostro != "ninguno":
        print(f"  Rostro (D-08): {args.rostro}, {len(ROSTRO_SUBCONJUNTOS[args.rostro])} puntos")
    print(f"  Salida:  {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Etapa 1 ---
    print("\n[Etapa 1] Carga y verificacion de estructura")
    print("-" * 60)
    samples = load_all_samples(dirs, args.rostro)
    if not samples:
        # Revisión D-05 (14-sep-2026): sin muestras, verify_structure fallaba con un traceback
        # (min() de una lista vacía); se sale con un mensaje claro.
        sys.exit(f"[ERROR] No se cargo ninguna muestra de {landmarks_dir} (arbol {layout}): "
                 f"BODY_LANDMARKS/ sin CSV o cada muestra falta en alguna de las tres carpetas")
    verify_structure(samples)

    # --- Etapa 2 ---
    print("\n[Etapa 2] Distribucion de clases")
    print("-" * 60)
    dist = compute_class_distribution(samples)
    balanced = (dist["count"] == 20).all()
    print(f"  Clases: {len(dist)}")
    print(f"  Muestras por clase: min={dist['count'].min()}, max={dist['count'].max()}")
    print(f"  Balanceado: {'SI' if balanced else 'NO'}")
    if not balanced:
        unbalanced = dist[dist["count"] != 20]
        print(f"  Clases desbalanceadas:\n{unbalanced.to_string(index=False)}")
    plot_class_distribution(dist, output_dir / "class_distribution.png")

    # --- Etapa 3 ---
    print("\n[Etapa 3] Deteccion de valores faltantes")
    print("-" * 60)
    report = detect_missing_values(samples)
    total_nan = report["nan_total"].sum()
    total_zero_rows = report["zero_rows_total"].sum()
    print(f"  NaN totales: {total_nan}")
    print(f"  Filas completamente en cero: {total_zero_rows}")
    print(f"  Filas en cero por grupo — postura: {int(report['zero_rows_body'].sum())}, "
          f"mano izq: {int(report['zero_rows_lh'].sum())}, "
          f"mano der: {int(report['zero_rows_rh'].sum())}"
          + (f", rostro: {int(report['zero_rows_face'].sum())}" if args.rostro != "ninguno" else ""))

    treated = 0
    if total_nan > 0:
        print(f"  Aplicando interpolacion lineal...")
        treated = treat_missing_values(samples, report)
        print(f"  Muestras tratadas: {treated}")
    else:
        print("  Datos limpios — no se requiere tratamiento.")

    plot_missing_value_report(report, output_dir / "missing_values.png")
    valores_faltantes = {
        "nan_totales": int(total_nan),
        "muestras_con_nan_interpoladas": int(treated),
        "filas_todo_en_ceros": int(total_zero_rows),
        "filas_postura_en_ceros": int(report["zero_rows_body"].sum()),
        "filas_mano_izquierda_en_ceros": int(report["zero_rows_lh"].sum()),
        "filas_mano_derecha_en_ceros": int(report["zero_rows_rh"].sum()),
        **({"filas_rostro_en_ceros": int(report["zero_rows_face"].sum())} if args.rostro != "ninguno" else {}),
        "nota": "postura ausente: fotograma completo en ceros (fotogramas_sin_postura). Mano ausente: ver la clave mano_ausente de este mismo metadata (D-07): ceros = se normaliza como cualquier punto (-referencia/escala); descartar = el fotograma se omite; mascara = el bloque se pone en ceros tras normalizar. "
                "Solo la postura en ceros activa fotogramas_sin_postura",
    }

    # --- Etapa 4 ---
    print("\n[Etapa 4] Normalizacion de coordenadas")
    print("-" * 60)
    print("  Metodo: centrado en torso + escalado por ancho de hombros")
    sin_postura = aplicar_regla_sin_postura(samples)
    print(f"  Fotogramas sin postura (BODY en ceros): {sin_postura['filas_postura_en_ceros']} "
          f"en {sin_postura['muestras_afectadas']} muestras -> fotograma completo en ceros "
          f"(regla del prototipo, src/translator.py)")
    mano_ausente = aplicar_regla_mano_ausente(samples, args.mano_ausente)
    print(f"  Fotogramas con alguna mano ausente: {mano_ausente['fotogramas_con_alguna_mano_ausente']} "
          f"(izq {mano_ausente['fotogramas_sin_mano_izq']}, der {mano_ausente['fotogramas_sin_mano_der']}) "
          f"en {mano_ausente['muestras_afectadas']} muestras -> modo D-07 '{args.mano_ausente}'"
          + (f": {mano_ausente['fotogramas_descartados']} fotogramas descartados" if args.mano_ausente == "descartar" else ""))
    normalize_all_samples(samples)
    if args.mano_ausente == "mascara":
        mano_ausente["bloques_de_mano_enmascarados_tras_normalizar"] = aplicar_mascara_mano_ausente(samples)
        print(f"  Bloques enmascarados tras normalizar (manos y rostro): {mano_ausente['bloques_de_mano_enmascarados_tras_normalizar']}")

    # Verificación rápida: hombros post-normalización
    sample_name = next(iter(samples))
    s = samples[sample_name]
    mid_x = (s["data"][0, LEFT_SHOULDER_IDX] + s["data"][0, RIGHT_SHOULDER_IDX]) / 2
    mid_y = (s["data"][0, LEFT_SHOULDER_IDX + 1] + s["data"][0, RIGHT_SHOULDER_IDX + 1]) / 2
    print(f"  Verificacion ({sample_name}, frame 0): midpoint hombros = ({mid_x:.6f}, {mid_y:.6f}) [esperado ~0]")

    # --- Etapa 5 ---
    print("\n[Etapa 5] Guardando datos procesados")
    print("-" * 60)
    metadata_extra = {
        "landmarks_dir": _rel(landmarks_dir),
        "layout": layout,
        "fecha_procesamiento": datetime.now().isoformat(timespec="seconds"),
        "valores_faltantes": valores_faltantes,
        "fotogramas_sin_postura": sin_postura,
        "mano_ausente": mano_ausente,
        **({"rostro": {"subconjunto": args.rostro, "indices_malla_facial": ROSTRO_SUBCONJUNTOS[args.rostro],
                       "columnas": [N_TOTAL_FEATURES, N_TOTAL_FEATURES + 3 * len(ROSTRO_SUBCONJUNTOS[args.rostro])],
                       "decision": "D-08; orden y pares simetricos en src/representacion.py"}}
           if args.rostro != "ninguno" else {}),
    }
    save_processed_data(samples, output_dir, metadata_extra)

    print("\n" + "=" * 60)
    print("  Pipeline completado exitosamente!")
    print(f"  Datos en: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
