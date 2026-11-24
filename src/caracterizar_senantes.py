"""
Condición del señante excluido (actividad 4.e): caracterización objetiva de las cinco personas de LSC50
y su relación con el resultado de cada pliegue LOSO. SOLO REPORTE: no entrena, no evalúa ningún modelo y
no decide nada; lee datos y predicciones ya guardados.

Qué documenta el corpus sobre las personas (no se inventa nada más):
  · data/processed/caracterizacion/caracterizacion.json › senantes (actividad 1.b): los dos criterios sin
    conciliar del artículo descriptor (ref19) y la antropometría de IMU/INFO/Volunteer_k.txt, cuya
    correspondencia con los identificadores 0000-0004 el depósito no declara.
  · data/raw/LSC50/IMU/INFO/Volunteer_k.txt (solo lectura) para cotejar esa antropometría.

Qué se mide, por video (las 997 muestras válidas, D-06; los 3 duplicados se excluyen):
  · duración: fotogramas antes de remuestrear (data/processed_holistic_mascara_rostro/lengths.npy) y
    segundos (÷ 24 cuadros por segundo de COLOR_BODY).
  · encuadre: ancho de hombros d (mediana por video, fracción del ancho de la imagen) y altura del centro de
    los hombros en la imagen (fracción del alto, 0 = borde superior), desde
    data/extracted/LSC50_holistic/BODY_LANDMARKS (coordenadas de imagen). Mismo d que
    Documento_Tesis/scripts_figuras/c4a_estadisticas.py (fig:c4a_ancho_hombros).
  · detección de manos: % de fotogramas sin la mano izquierda / derecha (METRICAS_FOTOGRAMA, banderas de
    la extracción; cociente de totales, como fig:c4a_deteccion) y % de videos con algún fotograma sin mano.
  · mano activa: el bloque de mano cuya muñeca (punto 0) más recorre en el espacio normalizado por hombros
    (X_padded, mano ausente en ceros, solo pares de fotogramas consecutivos con la mano presente): la
    misma definición de src/analisis_errores.py::_movimiento_munecas que usa el análisis 4.d. Se reporta
    el % de videos cuya mano activa es la derecha y el % de videos con las dos manos en movimiento
    (cociente de recorridos ≥ 0,5, proxy de src/analisis_errores_tcn.py; no es una anotación lingüística).
  · velocidad de la mano activa: recorrido de su muñeca ÷ tiempo con la mano presente, en anchos de hombro
    por segundo.
  · espacio de señas: altura máxima que alcanza la muñeca activa, en anchos de hombro por encima del centro
    de los hombros (positivo = por encima).
  · distancia a las otras personas: con las muñecas de la postura (puntos 15 y 16, presentes en todos los
    fotogramas), cada video remuestreado a 30 pasos (misma interpolación que train_final_100.resample_sequence),
    trayectoria media de cada persona en cada seña; distancia RMS (anchos de hombro) entre la trayectoria
    media de la persona y la de cada una de las otras cuatro en la misma seña; mediana sobre las 50 señas
    del promedio de esas cuatro distancias. Como referencia, la dispersión de sus propias repeticiones
    (distancia RMS de cada repetición a la trayectoria media de su persona y seña).

Resultado de cada pliegue (señante excluido = señante de prueba), solo para ponerlo al lado:
  · modelo final (D-14): models/confirmacion_tcn_2026-10-03/T1_imputacion_252/semilla_S/pliegue_k
    (predicciones.json, cotejado con resultado.json › test_top1).
  · modelo heredado: models/experimento_mano_ausente_2026-10-02/A_mascara/semilla_S/pliegue_k
    (predicciones.json, cotejado con metricas.json › top1).
  · validación del modelo final por persona cuando es vista: docs/REPORTE_COMPARATIVO_4e_MODELO_FINAL.json
    › validacion_loso_final_por_senante.

Con cinco personas no se puede establecer ninguna relación entre una propiedad y el resultado del pliegue:
el reporte solo ordena y compara. Como control dentro de cada persona se compara, con las 3 semillas
juntas, la mediana de cada propiedad en los videos acertados y en los fallados (descriptivo).

No abre videos ni escribe imágenes de personas: solo estadísticas de puntos característicos.
No escribe nada en data/.

Uso:
    python src/caracterizar_senantes.py        # escribe docs/CARACTERIZACION_SENANTES.{json,md}
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
EXTRAIDO = BASE_DIR / "data" / "extracted" / "LSC50_holistic"
PARTICION = BASE_DIR / "data" / "partitions" / "loso_5_pliegues.json"
CARACTERIZACION = BASE_DIR / "data" / "processed" / "caracterizacion" / "caracterizacion.json"
IMU_INFO = BASE_DIR / "data" / "raw" / "LSC50" / "IMU" / "INFO"
REPORTE_4E = BASE_DIR / "docs" / "REPORTE_COMPARATIVO_4e_MODELO_FINAL.json"
BRAZOS = {
    "final": (BASE_DIR / "models" / "confirmacion_tcn_2026-10-03" / "T1_imputacion_252", "resultado.json", "test_top1"),
    "heredado": (BASE_DIR / "models" / "experimento_mano_ausente_2026-10-02" / "A_mascara", "metricas.json", "top1"),
}
SEMILLAS = (2026, 2027, 2028)
SENANTES = ("0000", "0001", "0002", "0003", "0004")
FPS = 24.0
PASOS = 30                 # longitud de entrada del modelo final (N-75)
UMBRAL_MANOS = 0.5         # proxy de «dos manos en movimiento» de analisis_errores_tcn.py
SALIDA_JSON = BASE_DIR / "docs" / "CARACTERIZACION_SENANTES.json"
SALIDA_MD = BASE_DIR / "docs" / "CARACTERIZACION_SENANTES.md"

# Columnas en X_padded (225 primeras: postura 33×3, mano izquierda 21×3, mano derecha 21×3)
MUNECA_POSTURA = {"izq": 15 * 3, "der": 16 * 3}
BLOQUE_MANO = {"izq": 99, "der": 162}


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(p)


def leer_csv_numerico(ruta: Path) -> np.ndarray:
    with open(ruta, newline="", encoding="utf-8") as f:
        filas = list(csv.reader(f))
    return np.array([[float(v) for v in fila[1:]] for fila in filas[1:]], dtype=np.float64)


def resample(seq: np.ndarray, target: int) -> np.ndarray:
    """Igual que src/train_final_100.resample_sequence (interpolación lineal por columna)."""
    T, F = seq.shape
    if T == target:
        return seq.astype(np.float64)
    idx = np.linspace(0, T - 1, target)
    return np.stack([np.interp(idx, np.arange(T), seq[:, f]) for f in range(F)], axis=1)


def resumen(v) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return {"mediana": float(np.median(v)), "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)), "media": float(v.mean()), "n": int(v.size)}


# ---------------------------------------------------------------------------------------------------
# Propiedades por video
# ---------------------------------------------------------------------------------------------------

def propiedades_por_video(muestras: list[str]) -> tuple[dict, dict]:
    meta = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
    orden = {m: i for i, m in enumerate(meta["sample_order"])}
    longitudes = np.load(PROCESSED / "lengths.npy")
    X = np.load(PROCESSED / "X_padded.npy", mmap_mode="r")
    props, trayectorias = {}, {}
    for m in muestras:
        i = orden[m]
        n = int(longitudes[i])
        fila = np.asarray(X[i, :n, :225], dtype=np.float64)
        d = {"senante": m.split("_")[1], "sena": int(m.split("_")[0]), "fotogramas": n, "segundos": n / FPS}

        # Encuadre (coordenadas de imagen)
        b = leer_csv_numerico(EXTRAIDO / "BODY_LANDMARKS" / f"{m}.csv")
        if len(b) != n:
            raise ValueError(f"{m}: BODY_LANDMARKS tiene {len(b)} filas y lengths.npy {n}")
        hi, hd = b[:, 33:36], b[:, 36:39]
        ancho = np.sqrt((hi[:, 0] - hd[:, 0]) ** 2 + (hi[:, 1] - hd[:, 1]) ** 2)
        d["ancho_hombros"] = float(np.median(ancho))
        d["altura_hombros_imagen"] = float(np.median((hi[:, 1] + hd[:, 1]) / 2))

        # Detección de manos (banderas de la extracción)
        with open(EXTRAIDO / "METRICAS_FOTOGRAMA" / f"{m}.csv", newline="", encoding="utf-8") as f:
            met = list(csv.DictReader(f))
        if len(met) != n:
            raise ValueError(f"{m}: METRICAS_FOTOGRAMA tiene {len(met)} filas y lengths.npy {n}")
        sin_izq = np.array([int(r["mano_izq_detectada"]) == 0 for r in met])
        sin_der = np.array([int(r["mano_der_detectada"]) == 0 for r in met])
        d["fotogramas_sin_izq"] = int(sin_izq.sum())
        d["fotogramas_sin_der"] = int(sin_der.sum())
        d["fotogramas_sin_alguna"] = int((sin_izq | sin_der).sum())

        # Mano activa, velocidad y altura (muñeca de cada bloque de mano, espacio normalizado)
        mov = {}
        for lado, c0 in BLOQUE_MANO.items():
            bloque = fila[:, c0:c0 + 63]
            presente = np.any(bloque != 0, axis=1)
            if int((~presente).sum()) != (d["fotogramas_sin_izq"] if lado == "izq" else d["fotogramas_sin_der"]):
                raise ValueError(f"{m}: bloques en ceros de X_padded ({lado}) no coinciden con METRICAS_FOTOGRAMA")
            xy = bloque[:, 0:2]
            ambos = presente[1:] & presente[:-1]
            pasos = np.linalg.norm(np.diff(xy, axis=0), axis=1)[ambos]
            mov[lado] = {"recorrido": float(pasos.sum()), "pasos": int(ambos.sum()),
                         "y_min": float(xy[presente, 1].min()) if presente.any() else np.nan}
        activa = "izq" if mov["izq"]["recorrido"] >= mov["der"]["recorrido"] else "der"
        otra = "der" if activa == "izq" else "izq"
        d["mano_activa"] = activa
        mx, mn = mov[activa]["recorrido"], mov[otra]["recorrido"]
        d["cociente_recorrido"] = mn / mx if mx > 0 else 0.0
        d["dos_manos"] = bool(d["cociente_recorrido"] >= UMBRAL_MANOS)
        pa = mov[activa]["pasos"]
        d["velocidad_activa"] = mov[activa]["recorrido"] / (pa / FPS) if pa > 0 else np.nan
        d["altura_max_activa"] = -mov[activa]["y_min"]
        # Control: mano activa según las muñecas de la postura (15 izquierda, 16 derecha), siempre presentes
        rp = {lado: float(np.linalg.norm(np.diff(fila[:, c0:c0 + 2], axis=0), axis=1).sum())
              for lado, c0 in MUNECA_POSTURA.items()}
        d["mano_activa_postura"] = "izq" if rp["izq"] >= rp["der"] else "der"
        props[m] = d

        # Trayectoria de las muñecas de la postura (siempre presentes), 30 pasos
        cols = [MUNECA_POSTURA["izq"], MUNECA_POSTURA["izq"] + 1, MUNECA_POSTURA["der"], MUNECA_POSTURA["der"] + 1]
        if np.any(np.all(fila[:, 0:99] == 0, axis=1)):
            raise ValueError(f"{m}: hay fotogramas sin postura")
        trayectorias[m] = resample(fila[:, cols], PASOS)
    return props, trayectorias


def distancia_rms(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia RMS entre dos trayectorias 30×4 (dos muñecas, x-y): raíz de la media, sobre pasos y
    muñecas, de la distancia euclídea al cuadrado. En anchos de hombro."""
    da = (a - b).reshape(PASOS, 2, 2)
    return float(np.sqrt(np.mean(np.sum(da ** 2, axis=2))))


def distancias_entre_personas(props: dict, tray: dict) -> dict:
    medias, dispersion = {}, {s: [] for s in SENANTES}
    for s in SENANTES:
        for c in range(50):
            ms = [m for m, p in props.items() if p["senante"] == s and p["sena"] == c]
            M = np.mean([tray[m] for m in ms], axis=0)
            medias[(s, c)] = M
            if len(ms) > 1:
                dispersion[s] += [distancia_rms(tray[m], M) for m in ms]
    a_otras = {s: [] for s in SENANTES}
    matriz = np.zeros((5, 5))
    for c in range(50):
        for i, s in enumerate(SENANTES):
            ds = []
            for j, q in enumerate(SENANTES):
                if q != s:
                    dd = distancia_rms(medias[(s, c)], medias[(q, c)])
                    ds.append(dd)
                    matriz[i, j] += dd / 50
            a_otras[s].append(float(np.mean(ds)))
    return {
        "a_las_otras_por_sena": a_otras,
        "repeticiones_propias": dispersion,
        "matriz_media_entre_personas": [[round(float(x), 4) for x in fila] for fila in matriz],
    }


# ---------------------------------------------------------------------------------------------------
# Resultados por pliegue (solo lectura)
# ---------------------------------------------------------------------------------------------------

def resultados_por_pliegue() -> tuple[dict, dict]:
    top1, aciertos = {}, {b: {} for b in BRAZOS}
    for brazo, (carpeta, archivo, clave) in BRAZOS.items():
        m = np.zeros((len(SEMILLAS), 5))
        for i, sem in enumerate(SEMILLAS):
            for k, s in enumerate(SENANTES):
                d = carpeta / f"semilla_{sem}" / f"pliegue_{k}"
                preds = json.loads((d / "predicciones.json").read_text(encoding="utf-8"))
                ok = [p["real"] == p["predicha"] for p in preds]
                for p in preds:
                    if p["muestra"].split("_")[1] != s:
                        raise ValueError(f"{p['muestra']} en el pliegue {k}, que deja fuera a {s}")
                    aciertos[brazo].setdefault(p["muestra"], []).append(bool(p["real"] == p["predicha"]))
                m[i, k] = float(np.mean(ok))
                guardado = float(json.loads((d / archivo).read_text(encoding="utf-8"))[clave])
                if abs(guardado - m[i, k]) > 1e-9:
                    raise ValueError(f"{brazo} {sem} pliegue {k}: {m[i, k]} recalculado frente a {guardado} guardado")
        top1[brazo] = {"por_semilla": m.round(4).tolist(), "media": dict(zip(SENANTES, m.mean(0).round(4).tolist())),
                       "rango_semillas": dict(zip(SENANTES, np.ptp(m, axis=0).round(4).tolist()))}
    val = json.loads(REPORTE_4E.read_text(encoding="utf-8"))["validacion_loso_final_por_senante"]
    top1["final_validacion_como_visto"] = {s: round(float(v), 4) for s, v in val["top1_por_senante_media_semillas"].items()}
    return top1, aciertos


# ---------------------------------------------------------------------------------------------------
# Cotejo con cifras ya publicadas en el documento (si no coinciden, el script se detiene)
# ---------------------------------------------------------------------------------------------------

CIFRAS_CONOCIDAS = {
    # c4a_estadisticas.json › por_senante (Cap. 4A, fig:c4a_duracion_senante y fig:c4a_ancho_hombros)
    "fotogramas_media": ({"0000": 72.19, "0001": 70.40, "0002": 66.69, "0003": 56.44, "0004": 71.03}, 0.005),
    "ancho_hombros_mediana": ({"0000": 0.1247, "0001": 0.1007, "0002": 0.1003, "0003": 0.0974, "0004": 0.1086}, 0.00005),
    # c4a: 0002 sin mano izquierda 14,3 %; 0000 sin mano derecha 5,5 % (N-35)
    "sin_mano_izq_pct_fotogramas": ({"0002": 14.3}, 0.05),
    "sin_mano_der_pct_fotogramas": ({"0000": 5.5}, 0.05),
    # ANALISIS_ERRORES_4d_TCN.md §6, descriptivo por señante
    "sin_alguna_mano_pct_media_por_video": ({"0000": 11.6, "0001": 3.7, "0002": 16.8, "0003": 7.8, "0004": 3.6}, 0.05),
    "videos_con_alguna_mano_ausente": ({"0000": 177, "0001": 115, "0002": 187, "0003": 163, "0004": 109}, 0),
    # tab:c5_senantes (media de 3 semillas, 3 decimales)
    "top1_prueba_final": ({"0000": 0.488, "0001": 0.611, "0002": 0.721, "0003": 0.770, "0004": 0.795}, 0.0005),
    "top1_prueba_heredado": ({"0000": 0.452, "0001": 0.536, "0002": 0.631, "0003": 0.640, "0004": 0.687}, 0.0005),
}


def cotejar(por_senante: dict) -> dict:
    salida = {}
    for clave, (esperado, tol) in CIFRAS_CONOCIDAS.items():
        for s, v in esperado.items():
            r = por_senante[s]
            x = r["fotogramas"]["media"] if clave == "fotogramas_media" else (
                r["ancho_hombros"]["mediana"] if clave == "ancho_hombros_mediana" else r[clave])
            if abs(x - v) > tol + 1e-12:
                raise ValueError(f"cotejo {clave} {s}: {x} frente a {v} publicado")
            salida[f"{clave}.{s}"] = {"publicado": v, "recalculado": round(float(x), 4)}
    return salida


# ---------------------------------------------------------------------------------------------------
# Lo documentado
# ---------------------------------------------------------------------------------------------------

def documentado() -> dict:
    car = json.loads(CARACTERIZACION.read_text(encoding="utf-8"))["senantes"]
    cotejo = {}
    for k in range(1, 6):
        ruta = IMU_INFO / f"Volunteer_{k}.txt"
        if ruta.exists():
            leido = {}
            for linea in ruta.read_text(encoding="utf-8", errors="replace").splitlines():
                if ":" in linea:
                    a, b = linea.split(":", 1)
                    leido[a.strip()] = b.strip()
            pub = car["antropometria_publicada"]["datos"][f"Volunteer_{k}"]
            cotejo[f"Volunteer_{k}"] = all(leido.get(c) == v for c, v in pub.items())
    return {"fuente": rel(CARACTERIZACION) + " › senantes (actividad 1.b)",
            "condicion_segun_el_articulo": car["condicion_segun_el_articulo"],
            "antropometria_publicada": car["antropometria_publicada"],
            "cotejo_con_IMU_INFO": cotejo}


# ---------------------------------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------------------------------

PROPIEDADES = [
    # clave, etiqueta, unidad, decimales
    ("fotogramas", "Duración", "cuadros", 1),
    ("segundos", "Duración", "s", 2),
    ("ancho_hombros", "Ancho de hombros en la imagen", "fracción del ancho", 3),
    ("altura_hombros_imagen", "Altura de los hombros en la imagen", "fracción del alto (0 = arriba)", 3),
    ("velocidad_activa", "Velocidad de la muñeca activa", "anchos de hombro/s", 2),
    ("altura_max_activa", "Altura máxima de la muñeca activa", "anchos de hombro sobre el centro de los hombros", 2),
]


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def analizar() -> dict:
    part = json.loads(PARTICION.read_text(encoding="utf-8"))
    excluidas = set(part["excluidas"])
    meta = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
    muestras = [m for m in meta["sample_order"] if m not in excluidas]
    if len(muestras) != 997:
        raise ValueError(f"se esperaban 997 muestras válidas, hay {len(muestras)}")
    props, tray = propiedades_por_video(muestras)
    dist = distancias_entre_personas(props, tray)
    top1, aciertos = resultados_por_pliegue()

    por_senante = {}
    for s in SENANTES:
        ms = [m for m in muestras if props[m]["senante"] == s]
        P = [props[m] for m in ms]
        fot = sum(p["fotogramas"] for p in P)
        r = {"videos": len(ms)}
        for clave, *_ in PROPIEDADES:
            r[clave] = resumen([p[clave] for p in P])
        r["sin_mano_izq_pct_fotogramas"] = 100 * sum(p["fotogramas_sin_izq"] for p in P) / fot
        r["sin_mano_der_pct_fotogramas"] = 100 * sum(p["fotogramas_sin_der"] for p in P) / fot
        r["sin_alguna_mano_pct_fotogramas"] = 100 * sum(p["fotogramas_sin_alguna"] for p in P) / fot
        r["sin_alguna_mano_pct_media_por_video"] = 100 * float(np.mean([p["fotogramas_sin_alguna"] / p["fotogramas"] for p in P]))
        r["videos_con_alguna_mano_ausente"] = int(sum(p["fotogramas_sin_alguna"] > 0 for p in P))
        r["videos_con_alguna_mano_ausente_pct"] = 100 * float(np.mean([p["fotogramas_sin_alguna"] > 0 for p in P]))
        r["mano_activa_derecha_pct_videos"] = 100 * float(np.mean([p["mano_activa"] == "der" for p in P]))
        r["control_mano_activa_postura_derecha_pct_videos"] = 100 * float(np.mean([p["mano_activa_postura"] == "der" for p in P]))
        r["control_acuerdo_mano_activa_postura_pct"] = 100 * float(np.mean([p["mano_activa_postura"] == p["mano_activa"] for p in P]))
        r["dos_manos_pct_videos"] = 100 * float(np.mean([p["dos_manos"] for p in P]))
        # Una mano en movimiento: qué mano usa
        una = [p for p in P if not p["dos_manos"]]
        r["una_mano_videos"] = len(una)
        r["una_mano_derecha_pct"] = 100 * float(np.mean([p["mano_activa"] == "der" for p in una])) if una else None
        r["distancia_a_otras_personas"] = resumen(dist["a_las_otras_por_sena"][s])
        r["dispersion_repeticiones_propias"] = resumen(dist["repeticiones_propias"][s])
        r["top1_prueba_final"] = top1["final"]["media"][s]
        r["top1_prueba_heredado"] = top1["heredado"]["media"][s]
        r["top1_validacion_final_como_visto"] = top1["final_validacion_como_visto"][s]
        por_senante[s] = r

    # Orden de las personas en cada propiedad y posición del 0000
    escalares = {
        "fotogramas_mediana": lambda r: r["fotogramas"]["mediana"],
        "ancho_hombros_mediana": lambda r: r["ancho_hombros"]["mediana"],
        "altura_hombros_imagen_mediana": lambda r: r["altura_hombros_imagen"]["mediana"],
        "velocidad_activa_mediana": lambda r: r["velocidad_activa"]["mediana"],
        "altura_max_activa_mediana": lambda r: r["altura_max_activa"]["mediana"],
        "sin_mano_izq_pct_fotogramas": lambda r: r["sin_mano_izq_pct_fotogramas"],
        "sin_mano_der_pct_fotogramas": lambda r: r["sin_mano_der_pct_fotogramas"],
        "sin_alguna_mano_pct_fotogramas": lambda r: r["sin_alguna_mano_pct_fotogramas"],
        "mano_activa_derecha_pct_videos": lambda r: r["mano_activa_derecha_pct_videos"],
        "dos_manos_pct_videos": lambda r: r["dos_manos_pct_videos"],
        "distancia_a_otras_personas_mediana": lambda r: r["distancia_a_otras_personas"]["mediana"],
        "dispersion_repeticiones_propias_mediana": lambda r: r["dispersion_repeticiones_propias"]["mediana"],
    }
    t_final = [por_senante[s]["top1_prueba_final"] for s in SENANTES]
    orden = {}
    for nombre, f in escalares.items():
        v = [f(por_senante[s]) for s in SENANTES]
        idx = np.argsort(v)
        orden[nombre] = {
            "valores": dict(zip(SENANTES, [round(float(x), 4) for x in v])),
            "de_menor_a_mayor": [SENANTES[i] for i in idx],
            "posicion_0000_de_menor_a_mayor": int(np.where(idx == 0)[0][0]) + 1,
            "0000_es_extremo": bool(idx[0] == 0 or idx[-1] == 0),
            "spearman_con_top1_final_descriptivo": round(spearman(v, t_final), 3),
        }

    # Dentro de cada persona: aciertos frente a errores del modelo final (3 semillas juntas; descriptivo)
    dentro = {}
    for s in SENANTES:
        ms = [m for m in muestras if props[m]["senante"] == s]
        fila = {}
        for clave in ("fotogramas", "ancho_hombros", "velocidad_activa", "altura_max_activa"):
            ok, err = [], []
            for m in ms:
                for a in aciertos["final"][m]:
                    (ok if a else err).append(props[m][clave])
            fila[clave] = {"mediana_aciertos": float(np.median(ok)), "mediana_errores": float(np.median(err)),
                           "n_aciertos": len(ok), "n_errores": len(err)}
        sin = {True: [], False: []}
        for m in ms:
            for a in aciertos["final"][m]:
                sin[a].append(props[m]["fotogramas_sin_alguna"] / props[m]["fotogramas"])
        fila["pct_fotogramas_sin_alguna_mano"] = {"media_aciertos": 100 * float(np.mean(sin[True])),
                                                  "media_errores": 100 * float(np.mean(sin[False]))}
        dentro[s] = fila

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    return {
        "script": rel(Path(__file__)),
        "fecha": dt.date.today().isoformat(),
        "git_commit": commit,
        "solo_reporte": True,
        "actividad": "4.e (anteproyecto 09_metodologia.tex): reporte de cada pliegue con la condición del señante excluido",
        "fuentes": {
            "arreglos": rel(PROCESSED) + " (X_padded.npy, lengths.npy, metadata.json)",
            "coordenadas_de_imagen": rel(EXTRAIDO / "BODY_LANDMARKS"),
            "banderas_de_deteccion": rel(EXTRAIDO / "METRICAS_FOTOGRAMA"),
            "particion_y_excluidas": rel(PARTICION),
            "lo_documentado": rel(CARACTERIZACION) + "; " + rel(IMU_INFO) + " (solo lectura)",
            "modelo_final": rel(BRAZOS["final"][0]),
            "modelo_heredado": rel(BRAZOS["heredado"][0]),
            "validacion_como_visto": rel(REPORTE_4E) + " › validacion_loso_final_por_senante",
        },
        "definiciones": {
            "identificador": "nombre de archivo SSSS_VVVV_RRRR: seña, señante (VVVV = 0000-0004), repetición; el pliegue k del LOSO deja fuera al señante 000k",
            "muestras": "997 válidas (D-06): 200 / 199 / 198 / 200 / 200 por señante",
            "fotogramas": "lengths.npy (antes de remuestrear); segundos = fotogramas / 24",
            "ancho_hombros": "mediana por video de la distancia x-y entre los hombros (postura 11 y 12), en fracción del ancho de la imagen; igual que c4a_estadisticas.py",
            "altura_hombros_imagen": "mediana por video de la altura del centro de los hombros, fracción del alto de la imagen (0 = borde superior)",
            "sin_mano": "fotogramas con la bandera de detección en 0 (METRICAS_FOTOGRAMA) / fotogramas de la persona (cociente de totales)",
            "mano_activa": "bloque de mano cuya muñeca (punto 0) recorre más en x-y normalizado (pares consecutivos con la mano presente; analisis_errores._movimiento_munecas)",
            "dos_manos": f"recorrido de la muñeca que menos se mueve ≥ {UMBRAL_MANOS} × el de la que más se mueve (proxy de analisis_errores_tcn.py; no es una anotación lingüística)",
            "velocidad_activa": "recorrido de la muñeca activa / (pares válidos / 24), en anchos de hombro por segundo",
            "altura_max_activa": "−(mínimo de y de la muñeca activa) en el espacio normalizado: anchos de hombro por encima del centro de los hombros",
            "distancia_a_otras_personas": "muñecas de la postura (15, 16) remuestreadas a 30 pasos; trayectoria media por persona y seña; distancia RMS (anchos de hombro) a la de cada una de las otras cuatro en la misma seña; promedio de las cuatro; resumen sobre las 50 señas",
            "dispersion_repeticiones_propias": "distancia RMS de cada repetición a la trayectoria media de su persona y seña",
            "resumen": "mediana, percentiles 25 y 75 (RIC) y media sobre los videos de la persona",
        },
        "documentado_por_el_corpus": documentado(),
        "por_senante": por_senante,
        "resultado_por_pliegue": top1,
        "orden_entre_personas": orden,
        "distancias_entre_personas": {"matriz_media": dist["matriz_media_entre_personas"], "orden": list(SENANTES)},
        "dentro_de_cada_persona_final_3_semillas": dentro,
        "cotejo_con_cifras_publicadas": cotejar(por_senante),
        "advertencia": ("Cinco personas no permiten establecer una relación entre una propiedad y el top-1 del pliegue: "
                        "con cinco puntos, un orden idéntico (Spearman = ±1) aparece por azar en 2 de 120 órdenes posibles, "
                        "y las propiedades están mezcladas entre sí y con la persona. Los coeficientes de Spearman son solo descriptivos."),
    }


# ---------------------------------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------------------------------

def c(x, d=2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def escribir_md(a: dict) -> str:
    ps = a["por_senante"]
    L = ["# Condición del señante excluido (actividad 4.e)", "",
         f"Generado por `{a['script']}` el {a['fecha']} (commit {a['git_commit']}). Solo reporte: no entrena ni evalúa modelos. "
         "No usa imágenes de personas: solo estadísticas de puntos característicos.", "",
         "## 1. Lo que documenta el corpus", ""]
    doc = a["documentado_por_el_corpus"]
    cond = doc["condicion_segun_el_articulo"]
    L += [f"- Methods: {cond['criterio_methods']}.", f"- Data Records: {cond['criterio_data_records']}."]
    L += [f"- {x}" for x in cond["otros_datos_publicados"]]
    L += [f"- {cond['advertencia']}", "",
          "Antropometría publicada en `IMU/INFO/Volunteer_k.txt` (cotejada con el archivo: "
          + ", ".join(f"{k} {'sí' if v else 'NO'}" for k, v in doc["cotejo_con_IMU_INFO"].items()) + "):", "",
          "| Archivo | Estatura | Húmero | Radio | Espalda (cóccix a C7) |", "|---|---|---|---|---|"]
    for k, v in doc["antropometria_publicada"]["datos"].items():
        L.append(f"| {k} | {v['height']} | {v['humerus']} | {v['radius']} | {v['back (from coccyx to cervical vertebrae C7)']} |")
    L += ["", doc["antropometria_publicada"]["advertencia"], "",
          f"Fuente: {doc['fuente']}.", "",
          "## 2. Propiedades medidas por señante (mediana [P25; P75] sobre sus videos)", "",
          "| Propiedad | " + " | ".join(SENANTES) + " |", "|---|" + "---|" * 5,
          "| Videos | " + " | ".join(str(ps[s]["videos"]) for s in SENANTES) + " |"]
    for clave, etiqueta, unidad, dec in PROPIEDADES:
        L.append(f"| {etiqueta} ({unidad}) | " + " | ".join(
            f"{c(ps[s][clave]['mediana'], dec)} [{c(ps[s][clave]['p25'], dec)}; {c(ps[s][clave]['p75'], dec)}]" for s in SENANTES) + " |")
    for clave, etiqueta in (("sin_mano_izq_pct_fotogramas", "Fotogramas sin mano izquierda (%)"),
                            ("sin_mano_der_pct_fotogramas", "Fotogramas sin mano derecha (%)"),
                            ("sin_alguna_mano_pct_fotogramas", "Fotogramas sin alguna mano (%)"),
                            ("videos_con_alguna_mano_ausente_pct", "Videos con algún fotograma sin mano (%)"),
                            ("mano_activa_derecha_pct_videos", "Videos con la derecha como mano activa (%)"),
                            ("dos_manos_pct_videos", "Videos con las dos manos en movimiento (%)"),
                            ("una_mano_derecha_pct", "De los de una mano, con la derecha (%)"),
                            ("control_mano_activa_postura_derecha_pct_videos", "Control: derecha como más activa según las muñecas de la postura (%)"),
                            ("sin_alguna_mano_pct_media_por_video", "Fotogramas sin alguna mano, media por video (%) (= análisis 4.d)")):
        L.append(f"| {etiqueta} | " + " | ".join(c(ps[s][clave], 1) for s in SENANTES) + " |")
    for clave, etiqueta in (("distancia_a_otras_personas", "Distancia a las otras personas, misma seña (anchos de hombro)"),
                            ("dispersion_repeticiones_propias", "Dispersión de sus propias repeticiones (anchos de hombro)")):
        L.append(f"| {etiqueta} | " + " | ".join(
            f"{c(ps[s][clave]['mediana'], 3)} [{c(ps[s][clave]['p25'], 3)}; {c(ps[s][clave]['p75'], 3)}]" for s in SENANTES) + " |")
    L += ["| **Top-1 de prueba, modelo final (señante excluido)** | " + " | ".join(c(ps[s]["top1_prueba_final"], 3) for s in SENANTES) + " |",
          "| Videos con una sola mano en movimiento | " + " | ".join(str(ps[s]["una_mano_videos"]) for s in SENANTES) + " |",
          "| Top-1 de prueba, modelo heredado | " + " | ".join(c(ps[s]["top1_prueba_heredado"], 3) for s in SENANTES) + " |",
          "| Top-1 de validación del final cuando es visto | " + " | ".join(c(ps[s]["top1_validacion_final_como_visto"], 3) for s in SENANTES) + " |",
          "", "Definiciones:", ""]
    L += [f"- **{k}**: {v}" for k, v in a["definiciones"].items()]
    L += ["", "## 3. Orden de las personas y posición del 0000", "",
          "| Propiedad | de menor a mayor | posición del 0000 (1 = menor) | ¿0000 en un extremo? | Spearman con el top-1 final (descriptivo) |",
          "|---|---|---|---|---|"]
    for k, v in a["orden_entre_personas"].items():
        L.append(f"| {k} | {' < '.join(v['de_menor_a_mayor'])} | {v['posicion_0000_de_menor_a_mayor']} | "
                 f"{'sí' if v['0000_es_extremo'] else 'no'} | {c(v['spearman_con_top1_final_descriptivo'], 2)} |")
    L += ["", a["advertencia"], "",
          "Distancia media entre trayectorias de las personas (anchos de hombro; promedio de las 50 señas):", "",
          "| | " + " | ".join(SENANTES) + " |", "|---|" + "---|" * 5]
    for s, fila in zip(SENANTES, a["distancias_entre_personas"]["matriz_media"]):
        L.append(f"| {s} | " + " | ".join(c(x, 3) for x in fila) + " |")
    L += ["", "## 4. Dentro de cada persona: aciertos frente a errores del modelo final (3 semillas juntas, descriptivo)", "",
          "| Señante | duración (cuadros) acierto / error | velocidad activa acierto / error | altura máx. acierto / error | % sin alguna mano acierto / error | errores |",
          "|---|---|---|---|---|---|"]
    for s, f in a["dentro_de_cada_persona_final_3_semillas"].items():
        L.append(f"| {s} | {c(f['fotogramas']['mediana_aciertos'], 1)} / {c(f['fotogramas']['mediana_errores'], 1)} | "
                 f"{c(f['velocidad_activa']['mediana_aciertos'])} / {c(f['velocidad_activa']['mediana_errores'])} | "
                 f"{c(f['altura_max_activa']['mediana_aciertos'])} / {c(f['altura_max_activa']['mediana_errores'])} | "
                 f"{c(f['pct_fotogramas_sin_alguna_mano']['media_aciertos'], 1)} / {c(f['pct_fotogramas_sin_alguna_mano']['media_errores'], 1)} | "
                 f"{f['fotogramas']['n_errores']} de {f['fotogramas']['n_aciertos'] + f['fotogramas']['n_errores']} |")
    L += ["", "Cotejo con cifras ya publicadas (c4a_estadisticas, ANALISIS_ERRORES_4d_TCN, tab:c5_senantes): "
          f"{len(a['cotejo_con_cifras_publicadas'])} de {len(a['cotejo_con_cifras_publicadas'])} coinciden.", ""]
    L += ["", "## 5. Resultado por pliegue (cotejado con lo guardado)", ""]
    for b in ("final", "heredado"):
        L.append(f"- {b}: media de 3 semillas " + ", ".join(f"{s} {c(v, 3)}" for s, v in a["resultado_por_pliegue"][b]["media"].items())
                 + "; rango entre semillas " + ", ".join(f"{s} {c(v, 3)}" for s, v in a["resultado_por_pliegue"][b]["rango_semillas"].items()) + ".")
    L += ["", "Fuentes: " + "; ".join(f"{k}: `{v}`" for k, v in a["fuentes"].items()) + ".", ""]
    return "\n".join(L)


def main() -> None:
    a = analizar()
    SALIDA_JSON.write_text(json.dumps(a, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    SALIDA_MD.write_text(escribir_md(a), encoding="utf-8")
    print(SALIDA_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
