"""
Duración de los clips frente a la ventana fija de 2,5 s del prototipo, y simulación fuera de línea del
riesgo de cortar la seña con el modelo final (TCN + imputación + 252 rasgos, 30 pasos).

SOLO ANÁLISIS: no cambia el prototipo, ni modelos, ni umbrales, ni particiones. No escribe en data/.

Partes:
  1. Duración de los clips. (a) La de los videos COLOR_BODY (fotogramas de data/processed_holistic_
     mascara_rostro/lengths.npy, el processed del modelo final, a 24 fps medidos en las cabeceras AVI;
     data/processed/caracterizacion/caracterizacion.md §5-§6), sobre las 997 muestras válidas (D-06).
     (b) La que cita N-11 (media 2,57 s; mediana 2,58 s; p90 3,29 s; 47 % <= 2,5 s): se reproduce con
     las FILAS de los CSV publicados (data/processed/lengths.npy, mínimo de body/lh/rh por muestra), que
     tienen menos filas que fotogramas el video en 749 de 1.000 muestras (caracterizacion.md §12).
  2. Duración de la SEÑA dentro del clip. Con la postura normalizada por hombros (la medida de
     src/analisis_senas_cortadas.py: _velocidad_pose y _altura, sin cambios) un fotograma está en
     «reposo» si las dos muñecas están quietas (velocidad media <= V_UMBRAL anchos de hombro por segundo)
     y bajas (la más alta, a H_BAJA anchos de hombro o más por debajo de la línea de los hombros; H_BAJA =
     1,26 es el umbral candidato de altura del borde final de N-50). Sirve para los dos reposos del corpus:
     brazos a los lados (h ≈ 2,7) y manos juntas a la altura de la cintura (h ≈ 1,5-1,9). La seña va del
     primer al último fotograma que no está en reposo. Se reporta una rejilla de sensibilidad.
  3. Simulación con los 15 modelos LOSO del brazo T1 de la confirmación con la TCN
     (models/confirmacion_tcn_2026-10-03/T1_imputacion_252/semilla_{2026,2027,2028}/pliegue_{0..4}), que
     son los que dan el 67,7 % de prueba del modelo final. A cada clip de prueba se le aplica una ventana
     de 2,5 s = 60 fotogramas a 24 fps ANTES de la cadena del entrenamiento, en el mismo orden que el
     prototipo (recorte -> imputación de la mano ausente dentro de la ventana -> remuestreo a 30 ->
     add_velocity). Si el clip tiene menos de 60 fotogramas, la ventana se completa repitiendo el
     fotograma del borde (la persona quieta), salvo en la variante «sin relleno». Control: el clip
     completo debe reproducir probabilidades.npz[test] de cada corrida.
     Variantes: primeros 2,5 s (con y sin relleno), centrados, últimos; «retraso_D»: la ventana abre D s
     (0,25 / 0,5 / 0,75 / 1,0) antes del inicio de la seña medido en la parte 2 (si el clip no tiene tanto
     reposo inicial se antepone el primer fotograma quieto), que imita a la persona que tarda D s en
     arrancar tras pulsar ESPACIO; y primeros 3,0 / 3,5 s para ver si alargar la ventana ayuda. Para cada
     variante se cuenta en cuántos clips la ventana deja fuera parte del tramo activo («seña cortada»).

Determinista: sin aleatoriedad; inferencia en CPU (tl.importar_pipeline_canonico oculta la GPU).

Uso:
    python src/simular_ventana_25.py                 # todo (15 modelos)
    python src/simular_ventana_25.py --semillas 2026  # una semilla
    python src/simular_ventana_25.py --sin-modelo     # solo duraciones (sin TensorFlow)
Salida: docs/SIMULACION_VENTANA_25.json y docs/SIMULACION_VENTANA_25.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import train_loso as tl  # noqa: E402
from src import mano_ausente_variantes as mav  # noqa: E402
from src.analisis_senas_cortadas import _altura, _velocidad_pose  # noqa: E402

PROCESSED = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
PROCESSED_CSV = BASE_DIR / "data" / "processed"            # filas de los CSV publicados (N-11)
CONFIRMACION = BASE_DIR / "models" / "confirmacion_tcn_2026-10-03" / "T1_imputacion_252"
REGISTRO_VIVO = BASE_DIR / "registros" / "sesion_final_2026-11-23" / "latencia_2026-11-23.jsonl"
JSON_SALIDA = BASE_DIR / "docs" / "SIMULACION_VENTANA_25.json"
MD_SALIDA = BASE_DIR / "docs" / "SIMULACION_VENTANA_25.md"

FPS_CORPUS = 24.0                     # COLOR_BODY: 24,0 fps en las 1.000 cabeceras AVI (caracterizacion.md §6)
VENTANA_S = 2.5
N_VENTANA = int(round(VENTANA_S * FPS_CORPUS))   # 60 fotogramas
SEQ_LENGTH = 30
REPRESENTACION = "manos"
REGLA = "imputacion"
SEMILLAS = (2026, 2027, 2028)
UMBRAL_PROTOTIPO = 0.68               # N-75
H_BAJA = 1.261                        # anchos de hombro bajo la línea de los hombros (umbral h fin de N-50)
V_UMBRAL = 1.0                        # anchos de hombro por segundo
REJILLA_H_BAJA = (1.0, 1.15, 1.261)      # por encima de ~1,3 la línea pasa el reposo a la cintura de 0002-0004
REJILLA_V = (0.5, 1.0, 2.0)
COL_POSTURA = 99


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def resumen(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {"n": int(a.size), "min": _r(a.min()), "p10": _r(np.percentile(a, 10)), "media": _r(a.mean()),
            "mediana": _r(np.median(a)), "p90": _r(np.percentile(a, 90)), "max": _r(a.max()),
            "de": _r(a.std())}


# ---------------------------------------------------------------------------
# 1. Duración de los clips
# ---------------------------------------------------------------------------

def mascara_validas(nombres) -> np.ndarray:
    return np.array([n not in tl.DUPLICADOS_EXCLUIDOS for n in nombres])


def duraciones_clips() -> dict:
    out = {}
    meta = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
    nombres = meta["sample_order"]
    L = np.load(PROCESSED / "lengths.npy").astype(np.int64)
    v = mascara_validas(nombres)
    s = L[v] / FPS_CORPUS
    out["video_color_body_997"] = {
        "fuente": "data/processed_holistic_mascara_rostro/lengths.npy (fotogramas del video COLOR_BODY), "
                  "sin los 3 duplicados de D-06; 24 fps",
        "fotogramas": resumen(L[v]), "segundos": resumen(s),
        "frac_le_2_5s": _r((s <= 2.5).mean()), "frac_le_3_0s": _r((s <= 3.0).mean()),
        "n_menos_de_60_fotogramas": int((L[v] < N_VENTANA).sum()),
        "por_senante_media_s": {sn: _r(s[np.array([n.split('_')[1] == sn for n in np.array(nombres)[v]])].mean(), 3)
                                for sn in sorted({n.split('_')[1] for n in nombres})},
    }
    meta_c = json.loads((PROCESSED_CSV / "metadata.json").read_text(encoding="utf-8"))
    Lc = np.load(PROCESSED_CSV / "lengths.npy").astype(np.int64)
    for etiqueta, m in (("filas_csv_publicados_1000", np.ones(len(Lc), bool)),
                        ("filas_csv_publicados_997", mascara_validas(meta_c["sample_order"]))):
        sc = Lc[m] / FPS_CORPUS
        out[etiqueta] = {
            "fuente": "data/processed/lengths.npy (filas de los CSV publicados, mínimo de body/lh/rh), 24 fps; "
                      "es la base de las cifras de N-11",
            "fotogramas": resumen(Lc[m]), "segundos": resumen(sc),
            "frac_le_2_5s": _r((sc <= 2.5).mean()), "frac_le_3_0s": _r((sc <= 3.0).mean()),
        }
    return out


# ---------------------------------------------------------------------------
# 2. Duración de la seña dentro del clip
# ---------------------------------------------------------------------------

def tramo_sena(seq_pose: np.ndarray, h_baja=H_BAJA, v_umbral=V_UMBRAL) -> tuple[int, int] | None:
    """(primer, último) fotograma activo de un clip (T, >=99 columnas de postura normalizada).
    Reposo: velocidad media de las muñecas (media de los dos pares vecinos) <= v_umbral anchos de hombro
    por segundo Y muñeca más alta a h_baja o más por debajo de la línea de los hombros. Activo: lo demás
    (un fotograma sin postura no es activo). None si ningún fotograma es activo."""
    seq = np.asarray(seq_pose, dtype=np.float64)
    h = _altura(seq)
    vp = _velocidad_pose(seq) * FPS_CORPUS
    T = seq.shape[0]
    vf = np.full(T, np.nan)
    with np.errstate(invalid="ignore"):
        izq = np.concatenate([[np.nan], vp])
        der = np.concatenate([vp, [np.nan]])
        both = np.vstack([izq, der])
        cuenta = np.sum(~np.isnan(both), axis=0)
        vf = np.where(cuenta > 0, np.nansum(both, axis=0) / np.maximum(cuenta, 1), np.nan)
    if np.all(np.isnan(h)):
        return None
    with np.errstate(invalid="ignore"):
        activo = (h < h_baja) | (vf > v_umbral)
    activo &= ~np.isnan(h)
    idx = np.flatnonzero(activo)
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def duraciones_sena(Xpose, lengths, nombres) -> tuple[dict, dict]:
    v = mascara_validas(nombres)
    tramos = {}
    for i in np.flatnonzero(v):
        tramos[nombres[i]] = tramo_sena(Xpose[i, :lengths[i], :COL_POSTURA])
    sin = [n for n, t in tramos.items() if t is None]
    ok = [n for n, t in tramos.items() if t is not None]
    pos = {n: i for i, n in enumerate(nombres)}
    dur = np.array([(tramos[n][1] - tramos[n][0] + 1) / FPS_CORPUS for n in ok])
    rep_ini = np.array([tramos[n][0] / FPS_CORPUS for n in ok])
    rep_fin = np.array([(lengths[pos[n]] - 1 - tramos[n][1]) / FPS_CORPUS for n in ok])
    fin_sena = np.array([tramos[n][1] + 1 for n in ok])          # fotogramas hasta el fin de la seña
    principal = {
        "criterio": f"reposo = muñecas quietas (velocidad media <= {num(V_UMBRAL, 1)} anchos de hombro por segundo) y "
                    f"bajas (la más alta a {num(H_BAJA)} anchos de hombro o más bajo la línea de los hombros); seña = "
                    f"del primer al último fotograma que no está en reposo",
        "n_clips": len(ok), "clips_sin_actividad": sin,
        "duracion_sena_s": resumen(dur),
        "frac_sena_le_2_5s": _r((dur <= 2.5).mean()), "frac_sena_le_2_25s": _r((dur <= 2.25).mean()),
        "frac_sena_le_3_0s": _r((dur <= 3.0).mean()),
        "reposo_inicial_s": resumen(rep_ini), "reposo_final_s": resumen(rep_fin),
        "frac_reposo_inicial_ge_0_5s": _r((rep_ini >= 0.5).mean()),
        "frac_reposo_inicial_ge_1_0s": _r((rep_ini >= 1.0).mean()),
        "frac_reposo_final_ge_0_25s": _r((rep_fin >= 0.25).mean()),
        "frac_sena_termina_despues_de_2_5s_del_inicio_del_clip": _r((fin_sena > N_VENTANA).mean()),
        "fraccion_del_clip_ocupada_por_la_sena": resumen(dur / np.array([lengths[pos[n]] / FPS_CORPUS for n in ok])),
        "por_senante": {},
    }
    for sn in sorted({n.split("_")[1] for n in ok}):
        m = np.array([n.split("_")[1] == sn for n in ok])
        principal["por_senante"][sn] = {"mediana_sena_s": _r(np.median(dur[m]), 3),
                                        "frac_sena_le_2_5s": _r((dur[m] <= 2.5).mean(), 3),
                                        "mediana_reposo_inicial_s": _r(np.median(rep_ini[m]), 3),
                                        "mediana_reposo_final_s": _r(np.median(rep_fin[m]), 3)}
    rejilla = []
    for dh in REJILLA_H_BAJA:
        for vu in REJILLA_V:
            d = []
            for i in np.flatnonzero(v):
                t = tramo_sena(Xpose[i, :lengths[i], :COL_POSTURA], dh, vu)
                if t is not None:
                    d.append((t[1] - t[0] + 1) / FPS_CORPUS)
            d = np.array(d)
            rejilla.append({"h_baja": dh, "v_umbral": vu, "n": int(d.size), "mediana_s": _r(np.median(d), 3),
                            "p90_s": _r(np.percentile(d, 90), 3), "frac_le_2_5s": _r((d <= 2.5).mean(), 3)})
    principal["sensibilidad"] = rejilla
    return principal, tramos


# ---------------------------------------------------------------------------
# 3. Ventanas
# ---------------------------------------------------------------------------

def limites(T: int, variante: str, tramo: tuple[int, int] | None) -> tuple[int, int] | None:
    """(a, n): la ventana cubre los fotogramas a, a+1, ..., a+n-1 del clip. Fuera de [0, T-1] se repite el
    fotograma del borde (la persona quieta). None = el clip completo, sin ventana."""
    if variante == "completo":
        return None
    if variante.startswith("primeros_"):
        seg = float(variante.split("_")[1] + "." + variante.split("_")[2].rstrip("s"))
        return 0, int(round(seg * FPS_CORPUS))
    n = N_VENTANA
    if variante == "ultimos_2_5s":
        return T - n, n
    if variante == "centrados_2_5s":
        return (T - n) // 2, n
    if variante.startswith("retraso_"):
        # la seña empieza D s después de abrir la ventana (D = 0,25 / 0,5 / 0,75 / 1,0)
        d = float(variante.split("_")[1] + "." + variante.split("_")[2].rstrip("s"))
        inicio = 0 if tramo is None else tramo[0]
        return inicio - int(round(d * FPS_CORPUS)), n
    raise ValueError(variante)


def ventana(seq: np.ndarray, variante: str, tramo: tuple[int, int] | None) -> np.ndarray:
    T = seq.shape[0]
    if variante == "primeros_2_5s_sin_relleno":
        return seq[:N_VENTANA]
    lim = limites(T, variante, tramo)
    if lim is None:
        return seq
    a, n = lim
    idx = np.clip(np.arange(a, a + n), 0, T - 1)
    return seq[idx]


def sena_cortada(T: int, variante: str, tramo: tuple[int, int] | None) -> bool:
    """True si la ventana no contiene todo el tramo activo (la seña) del clip."""
    if tramo is None or variante == "completo":
        return False
    if variante == "primeros_2_5s_sin_relleno":
        return tramo[1] >= N_VENTANA
    a, n = limites(T, variante, tramo)
    return tramo[0] < a or tramo[1] > a + n - 1


VARIANTES = ("completo", "primeros_2_5s", "primeros_2_5s_sin_relleno", "centrados_2_5s", "ultimos_2_5s",
             "retraso_0_25s", "retraso_0_5s", "retraso_0_75s", "retraso_1_0s", "primeros_3_0s", "primeros_3_5s")


def cortes_por_variante(lengths, nombres, tramos) -> dict:
    """Sobre los 997 clips: en cuántos la ventana deja fuera parte de la seña y cuánto (s y fracción)."""
    pos = {n: i for i, n in enumerate(nombres)}
    out = {}
    for var in VARIANTES:
        if var in ("completo",):
            continue
        fuera_s, fuera_f, n_tot = [], [], 0
        for n, t in tramos.items():
            n_tot += 1
            T = int(lengths[pos[n]])
            if not sena_cortada(T, var, t):
                continue
            if var == "primeros_2_5s_sin_relleno":
                a, m = 0, N_VENTANA
            else:
                a, m = limites(T, var, t)
            fuera = max(0, a - t[0]) + max(0, t[1] - (a + m - 1))
            fuera_s.append(fuera / FPS_CORPUS)
            fuera_f.append(fuera / (t[1] - t[0] + 1))
        out[var] = {"n_clips": n_tot, "n_cortadas": len(fuera_s), "frac_cortadas": _r(len(fuera_s) / n_tot),
                    "s_fuera_mediana": _r(np.median(fuera_s)) if fuera_s else None,
                    "s_fuera_p90": _r(np.percentile(fuera_s, 90)) if fuera_s else None,
                    "frac_sena_fuera_mediana": _r(np.median(fuera_f)) if fuera_f else None,
                    "frac_sena_fuera_p90": _r(np.percentile(fuera_f, 90)) if fuera_f else None}
    return out


def entrada_red(seq_manos: np.ndarray, add_velocity) -> np.ndarray:
    """Misma cadena que el entrenamiento y el prototipo: imputación dentro de la ventana, 30 pasos, velocidades."""
    x = mav.aplicar_regla(seq_manos, REGLA, SEQ_LENGTH)
    return add_velocity(x[None, ...])[0]


def simular(args, tramos) -> dict:
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    tf, _, add_velocity, _, _ = canon
    X_padded, y, lengths, nombres, _ = tl.cargar_datos(PROCESSED, REPRESENTACION)
    loso = json.loads(tl.ARCHIVO_LOSO.read_text(encoding="utf-8"))
    pos = {n: i for i, n in enumerate(nombres)}
    nombres_senas = tl.cargar_nombres_senas()

    # Entradas por pliegue y variante (no dependen de la semilla).
    entradas = {}
    for p in loso["pliegues"]:
        k = int(p["pliegue"])
        test = list(p["conjuntos"]["test"])
        entradas[k] = {"nombres": test, "y": np.array([y[pos[n]] for n in test])}
        for var in VARIANTES:
            entradas[k][var] = np.stack([
                entrada_red(ventana(X_padded[pos[n], :lengths[pos[n]], :], var, tramos.get(n)), add_velocity)
                for n in test]).astype(np.float32)

    corridas = []
    for s in args.semillas:
        for k in sorted(entradas):
            d = CONFIRMACION / f"semilla_{s}" / f"pliegue_{k}"
            model = tf.keras.models.load_model(str(d / "modelo.keras"))
            guardado = np.load(d / "probabilidades.npz")
            assert list(guardado["nombres_test"]) == entradas[k]["nombres"], (s, k)
            fila = {"semilla": s, "pliegue": k, "senante_test": loso["pliegues"][k]["senante_test"],
                    "n_test": len(entradas[k]["nombres"]), "variantes": {}}
            for var in VARIANTES:
                probs = model.predict(entradas[k][var], verbose=0, batch_size=256)
                if var == "completo":
                    dif = float(np.max(np.abs(probs - guardado["test"])))
                    iguales = int(np.sum(np.argmax(probs, 1) == np.argmax(guardado["test"], 1)))
                    fila["control"] = {"max_dif_prob": dif, "pred_iguales": iguales,
                                       "reproduce": bool(iguales == len(probs) and dif < 1e-4)}
                    if not fila["control"]["reproduce"]:
                        raise SystemExit(f"control fallido s={s} k={k}: {fila['control']}")
                yt = entradas[k]["y"]
                pred = probs.argmax(1)
                conf = probs.max(1)
                top3 = np.argsort(-probs, axis=1)[:, :3]
                acept = conf >= UMBRAL_PROTOTIPO
                cort = np.array([sena_cortada(int(lengths[pos[n]]), var, tramos.get(n)) for n in entradas[k]["nombres"]])
                fila["variantes"][var] = {
                    "frac_sena_cortada": float(cort.mean()),
                    "top1_sena_cortada": float((pred[cort] == yt[cort]).mean()) if cort.any() else None,
                    "top1_sena_completa": float((pred[~cort] == yt[~cort]).mean()) if (~cort).any() else None,
                    "top1": float((pred == yt).mean()), "top3": float(np.mean([yt[i] in top3[i] for i in range(len(yt))])),
                    "confianza_media": float(conf.mean()), "rechazo_068": float(1 - acept.mean()),
                    "exactitud_aceptadas_068": float((pred[acept] == yt[acept]).mean()) if acept.any() else None,
                    "aceptadas_mal_068": int(np.sum(acept & (pred != yt))),
                    "acierto": (pred == yt).astype(int).tolist(),
                }
            corridas.append(fila)
            print(f"  s={s} k={k}: " + " | ".join(f"{v} {fila['variantes'][v]['top1']:.3f}" for v in VARIANTES),
                  flush=True)
            del model
            tf.keras.backend.clear_session()

    # Agregados
    res = {"variantes": {}, "por_pliegue": {}, "por_clase": [], "por_tramo": {}}
    for var in VARIANTES:
        por_corrida = np.array([c["variantes"][var]["top1"] for c in corridas])
        por_semilla = {s: float(np.mean([c["variantes"][var]["top1"] for c in corridas if c["semilla"] == s]))
                       for s in args.semillas}
        res["variantes"][var] = {
            "top1_media_corridas": float(por_corrida.mean()),
            "top1_por_semilla": por_semilla,
            "top1_de_entre_semillas": float(np.std(list(por_semilla.values()))),
            "top3_media": float(np.mean([c["variantes"][var]["top3"] for c in corridas])),
            "frac_sena_cortada": float(np.mean([c["variantes"][var]["frac_sena_cortada"] for c in corridas
                                                if c["semilla"] == args.semillas[0]])),
            "top1_sena_cortada_media": (float(np.mean(_v)) if (_v := [c["variantes"][var]["top1_sena_cortada"]
                                        for c in corridas if c["variantes"][var]["top1_sena_cortada"] is not None])
                                        else None),
            "top1_sena_completa_media": float(np.mean([c["variantes"][var]["top1_sena_completa"] for c in corridas
                                                       if c["variantes"][var]["top1_sena_completa"] is not None])),
            "rechazo_068_media": float(np.mean([c["variantes"][var]["rechazo_068"] for c in corridas])),
            "exactitud_aceptadas_068_media": float(np.mean([c["variantes"][var]["exactitud_aceptadas_068"]
                                                            for c in corridas])),
            "aceptadas_mal_068_por_corrida": float(np.mean([c["variantes"][var]["aceptadas_mal_068"] for c in corridas])),
            "confianza_media": float(np.mean([c["variantes"][var]["confianza_media"] for c in corridas])),
            "pliegues_peor_que_completo": int(sum(
                np.mean([c["variantes"][var]["top1"] for c in corridas if c["pliegue"] == k])
                < np.mean([c["variantes"]["completo"]["top1"] for c in corridas if c["pliegue"] == k]) - 1e-12
                for k in sorted(entradas))),
        }
    for k in sorted(entradas):
        res["por_pliegue"][k] = {"senante_test": loso["pliegues"][k]["senante_test"]}
        for var in VARIANTES:
            res["por_pliegue"][k][var] = float(np.mean([c["variantes"][var]["top1"] for c in corridas
                                                        if c["pliegue"] == k]))
    # Por clase (todas las corridas juntas)
    aciertos = {var: {} for var in VARIANTES}
    for c in corridas:
        e = entradas[c["pliegue"]]
        for var in VARIANTES:
            for yy, a in zip(e["y"], c["variantes"][var]["acierto"]):
                aciertos[var].setdefault(int(yy), []).append(a)
    for cl in sorted(aciertos["completo"]):
        fila = {"clase": cl, "sena": nombres_senas.get(cl, str(cl)), "n": len(aciertos["completo"][cl])}
        for var in VARIANTES:
            fila[var] = float(np.mean(aciertos[var][cl]))
        fila["perdida_primeros_2_5s"] = fila["completo"] - fila["primeros_2_5s"]
        fila["perdida_retraso_1_0s"] = fila["completo"] - fila["retraso_1_0s"]
        # duración mediana de clip y de seña de la clase
        ns = [n for n in nombres if int(n.split("_")[0]) == cl and n not in tl.DUPLICADOS_EXCLUIDOS]
        fila["mediana_clip_s"] = float(np.median([lengths[pos[n]] for n in ns]) / FPS_CORPUS)
        dd = [(tramos[n][1] - tramos[n][0] + 1) / FPS_CORPUS for n in ns if tramos.get(n)]
        fila["mediana_sena_s"] = float(np.median(dd)) if dd else None
        fila["frac_clips_gt_2_5s"] = float(np.mean([lengths[pos[n]] > N_VENTANA for n in ns]))
        res["por_clase"].append(fila)
    res["por_clase"].sort(key=lambda f: -f["perdida_primeros_2_5s"])
    # Según si la seña termina dentro de los primeros 60 fotogramas
    grupos = {"clip_le_60": [], "clip_gt_60_sena_termina_antes_de_60": [], "clip_gt_60_sena_termina_despues_de_60": []}
    for c in corridas:
        e = entradas[c["pliegue"]]
        for j, n in enumerate(e["nombres"]):
            T = lengths[pos[n]]
            t = tramos.get(n)
            if T <= N_VENTANA:
                g = "clip_le_60"
            elif t is not None and t[1] + 1 <= N_VENTANA:
                g = "clip_gt_60_sena_termina_antes_de_60"
            else:
                g = "clip_gt_60_sena_termina_despues_de_60"
            grupos[g].append((c["variantes"]["completo"]["acierto"][j], c["variantes"]["primeros_2_5s"]["acierto"][j],
                              c["variantes"]["retraso_0_25s"]["acierto"][j], c["variantes"]["retraso_1_0s"]["acierto"][j]))
    for g, v in grupos.items():
        a = np.array(v) if v else np.zeros((0, 4))
        res["por_tramo"][g] = {"n_predicciones": int(len(a)), "n_clips": int(len(a) // max(len(args.semillas), 1)),
                               "top1_completo": _r(a[:, 0].mean()) if len(a) else None,
                               "top1_primeros_2_5s": _r(a[:, 1].mean()) if len(a) else None,
                               "top1_retraso_0_25s": _r(a[:, 2].mean()) if len(a) else None,
                               "top1_retraso_1_0s": _r(a[:, 3].mean()) if len(a) else None}
    for c in corridas:
        for var in VARIANTES:
            c["variantes"][var].pop("acierto")
    res["corridas"] = corridas
    return res


# ---------------------------------------------------------------------------
# Registro en vivo
# ---------------------------------------------------------------------------

def avisos_en_vivo() -> dict:
    if not REGISTRO_VIVO.exists():
        return {"disponible": False}
    n = fin = ini = 0
    for linea in REGISTRO_VIVO.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        r = json.loads(linea)
        if r.get("tipo") != "ensayo" or r.get("modo") != "aislado" or not isinstance(r.get("bordes"), dict):
            continue
        n += 1
        fin += bool(r["bordes"].get("aviso_fin"))
        ini += bool(r["bordes"].get("aviso_inicio"))
    return {"disponible": True, "fuente": str(REGISTRO_VIVO.relative_to(BASE_DIR)), "ventanas_aislado_con_bordes": n,
            "aviso_fin": fin, "aviso_inicio": ini}


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}".replace(".", ",")


def num(x, d=2):
    return "—" if x is None else f"{x:.{d}f}".replace(".", ",")


def escribir_md(datos: dict) -> str:
    dc = datos["duracion_clips"]
    v, c = dc["video_color_body_997"], dc["filas_csv_publicados_1000"]
    ds = datos["duracion_sena"]
    L = ["# Ventana de 2,5 s: duración de los clips, duración de la seña y simulación del recorte", "",
         f"Generado por `src/simular_ventana_25.py` el {datos['fecha']}. Solo análisis: no cambia el prototipo,",
         "los modelos, los umbrales ni las particiones.", "",
         "## 1. Duración de los clips", "",
         "| Fuente | n | media | mediana | p90 | máx | ≤ 2,5 s | ≤ 3,0 s |", "|---|---|---|---|---|---|---|---|",
         f"| Video COLOR_BODY (fotogramas / 24), 997 válidos — **la correcta** | {v['segundos']['n']} | "
         f"{num(v['segundos']['media'])} | {num(v['segundos']['mediana'])} | {num(v['segundos']['p90'])} | "
         f"{num(v['segundos']['max'])} | {pct(v['frac_le_2_5s'])} % | {pct(v['frac_le_3_0s'])} % |",
         f"| Filas de los CSV publicados / 24, 1.000 (base de N-11) | {c['segundos']['n']} | "
         f"{num(c['segundos']['media'])} | {num(c['segundos']['mediana'])} | {num(c['segundos']['p90'])} | "
         f"{num(c['segundos']['max'])} | {pct(c['frac_le_2_5s'])} % | {pct(c['frac_le_3_0s'])} % |", "",
         "Las cifras de N-11 (media 2,57 s; mediana 2,58 s; p90 3,29 s; máx. 5,58 s; 47 % ≤ 2,5 s; 79 % ≤ 3,0 s)",
         "salen de dividir entre 24 el número de FILAS de los CSV de puntos publicados, que en 749 de 1.000 muestras",
         "tienen menos filas que fotogramas el video (mediana del déficit: 5; caracterizacion.md §12). El modelo",
         "final se entrenó con la extracción propia sobre el video COLOR_BODY (24,0 fps en las 1.000 cabeceras),",
         f"cuyos clips tienen {v['fotogramas']['min']:.0f}-{v['fotogramas']['max']:.0f} fotogramas (mediana "
         f"{v['fotogramas']['mediana']:.0f}); {v['n_menos_de_60_fotogramas']} de 997 tienen menos de 60.", "",
         "## 2. Duración de la seña dentro del clip", "",
         f"Criterio: {ds['criterio']}. Medida de `src/analisis_senas_cortadas.py` (_altura, _velocidad_pose).", "",
         f"- Clips con actividad: {ds['n_clips']} (sin actividad: {len(ds['clips_sin_actividad'])}).",
         f"- Seña: mediana {num(ds['duracion_sena_s']['mediana'])} s, media {num(ds['duracion_sena_s']['media'])} s, "
         f"p90 {num(ds['duracion_sena_s']['p90'])} s, máx. {num(ds['duracion_sena_s']['max'])} s; "
         f"≤ 2,5 s: {pct(ds['frac_sena_le_2_5s'])} %; ≤ 2,25 s: {pct(ds['frac_sena_le_2_25s'])} %; "
         f"≤ 3,0 s: {pct(ds['frac_sena_le_3_0s'])} %.",
         f"- Reposo antes de la seña: mediana {num(ds['reposo_inicial_s']['mediana'])} s, p90 "
         f"{num(ds['reposo_inicial_s']['p90'])} s, máx. {num(ds['reposo_inicial_s']['max'])} s; ≥ 0,5 s en "
         f"{pct(ds['frac_reposo_inicial_ge_0_5s'])} % de los clips, ≥ 1 s en {pct(ds['frac_reposo_inicial_ge_1_0s'])} %.",
         f"- Reposo después de la seña: mediana {num(ds['reposo_final_s']['mediana'])} s, p90 "
         f"{num(ds['reposo_final_s']['p90'])} s.",
         f"- La seña ocupa una mediana del {pct(ds['fraccion_del_clip_ocupada_por_la_sena']['mediana'])} % del clip.",
         f"- La seña termina después de los primeros 2,5 s del clip en {pct(ds['frac_sena_termina_despues_de_2_5s_del_inicio_del_clip'])} % de los clips.",
         "", "Por señante:", "", "| señante | mediana seña (s) | seña ≤ 2,5 s | mediana reposo inicial (s) | mediana reposo final (s) |",
         "|---|---|---|---|---|"]
    for sn, f in ds["por_senante"].items():
        L.append(f"| {sn} | {num(f['mediana_sena_s'])} | {pct(f['frac_sena_le_2_5s'])} % | "
                 f"{num(f['mediana_reposo_inicial_s'])} | {num(f['mediana_reposo_final_s'])} |")
    L += ["", "Sensibilidad al criterio:", "", "| h baja (anchos de hombro) | v (anchos/s) | mediana seña (s) | p90 (s) | ≤ 2,5 s |",
          "|---|---|---|---|---|"]
    for f in ds["sensibilidad"]:
        L.append(f"| {num(f['h_baja'])} | {num(f['v_umbral'], 1)} | {num(f['mediana_s'])} | {num(f['p90_s'])} | "
                 f"{pct(f['frac_le_2_5s'])} % |")
    L += ["", "Cuánto de la seña deja fuera cada ventana (997 clips; tramo activo de la parte 2):", "",
          "| variante | clips cortados | s fuera (mediana / p90) | fracción de la seña fuera (mediana / p90) |",
          "|---|---|---|---|"]
    for var, f in datos["cortes"].items():
        L.append(f"| {var} | {f['n_cortadas']} ({pct(f['frac_cortadas'])} %) | {num(f['s_fuera_mediana'])} / "
                 f"{num(f['s_fuera_p90'])} | {pct(f['frac_sena_fuera_mediana'], 0)} % / {pct(f['frac_sena_fuera_p90'], 0)} % |")
    sim = datos.get("simulacion")
    if sim:
        L += ["", "## 3. Simulación con los modelos LOSO del modelo final", "",
              f"Modelos: `{datos['modelos']}`, semillas {datos['semillas']} × 5 pliegues. Ventana de 2,5 s = 60 "
              "fotogramas a 24 fps, aplicada antes de la imputación, el remuestreo a 30 y las velocidades (el orden del "
              "prototipo). Clips de menos de 60 fotogramas: se completan repitiendo el fotograma del borde. Control: el "
              "clip completo reproduce las probabilidades guardadas de las 15 corridas "
              f"({'sí' if all(c['control']['reproduce'] for c in sim['corridas']) else 'NO'}; diferencia máxima "
              f"{max(c['control']['max_dif_prob'] for c in sim['corridas']):.1e}).", "",
              "| variante | seña cortada | top-1 cortadas / no cortadas | top-1 (media de corridas) | por semilla | DE entre semillas | top-3 | rechazo con 0,68 | exactitud aceptadas | pliegues peor que completo |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for var, f in sim["variantes"].items():
            L.append(f"| {var} | {pct(f['frac_sena_cortada'])} % | {num(f['top1_sena_cortada_media'], 3)} / "
                     f"{num(f['top1_sena_completa_media'], 3)} | {num(f['top1_media_corridas'], 3)} | "
                     + " / ".join(num(x, 3) for x in f["top1_por_semilla"].values())
                     + f" | {num(f['top1_de_entre_semillas'], 3)} | {num(f['top3_media'], 3)} | {pct(f['rechazo_068_media'])} % | "
                       f"{num(f['exactitud_aceptadas_068_media'], 3)} | {f['pliegues_peor_que_completo']} de 5 |")
        L += ["", "Top-1 por pliegue (media de las semillas):", "",
              "| pliegue | señante | " + " | ".join(VARIANTES) + " |", "|---|---|" + "---|" * len(VARIANTES)]
        for k, f in sim["por_pliegue"].items():
            L.append(f"| {k} | {f['senante_test']} | " + " | ".join(num(f[v], 3) for v in VARIANTES) + " |")
        L += ["", "Según dónde termina la seña (todas las predicciones de prueba):", "",
              "| grupo | clips | top-1 completo | top-1 primeros 2,5 s | top-1 retraso 0,25 s | top-1 retraso 1,0 s |", "|---|---|---|---|---|---|"]
        for g, f in sim["por_tramo"].items():
            L.append(f"| {g} | {f['n_clips']} | {num(f['top1_completo'], 3)} | {num(f['top1_primeros_2_5s'], 3)} | "
                     f"{num(f['top1_retraso_0_25s'], 3)} | {num(f['top1_retraso_1_0s'], 3)} |")
        L += ["", "Clases que más pierden con los primeros 2,5 s:", "",
              "| clase | seña | n | completo | primeros 2,5 s | pérdida | retraso 1,0 s | mediana clip (s) | mediana seña (s) | clips > 2,5 s |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for f in sim["por_clase"][:12]:
            L.append(f"| {f['clase']} | {f['sena']} | {f['n']} | {num(f['completo'], 3)} | {num(f['primeros_2_5s'], 3)} | "
                     f"{num(f['perdida_primeros_2_5s'], 3)} | {num(f['retraso_1_0s'], 3)} | "
                     f"{num(f['mediana_clip_s'])} | {num(f['mediana_sena_s'])} | {pct(f['frac_clips_gt_2_5s'], 0)} % |")
    vivo = datos["vivo"]
    if vivo.get("disponible"):
        L += ["", "## 4. Sesión en vivo del 23-nov (registro)", "",
              f"`{vivo['fuente']}`: {vivo['ventanas_aislado_con_bordes']} ventanas en modo aislado con la clave «bordes»; "
              f"aviso_fin en {vivo['aviso_fin']}, aviso_inicio en {vivo['aviso_inicio']}."]
    L += ["", "## 5. Límites", "",
          "- Recortar clips del corpus no es una ventana en vivo: en vivo la ventana empieza con la tecla y la persona "
          "arranca desde el reposo; aquí el inicio de la ventana es el del clip (con su reposo inicial, que varía).",
          "- Los clips de menos de 60 fotogramas se completan con el fotograma del borde quieto; el reposo real tiene "
          "pequeños movimientos.",
          "- El criterio de actividad es heurístico (altura y velocidad de las muñecas de la postura); la tabla de "
          "sensibilidad muestra cuánto cambia la duración de la seña con él.",
          "- Cinco señantes y la cámara del corpus (24 fps); el prototipo captura a ~30 fps con otra cámara.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semillas", default=",".join(map(str, SEMILLAS)))
    ap.add_argument("--sin-modelo", action="store_true")
    args = ap.parse_args()
    args.semillas = [int(s) for s in args.semillas.split(",") if s.strip()]
    t0 = time.perf_counter()

    meta = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
    nombres = list(meta["sample_order"])
    lengths = np.load(PROCESSED / "lengths.npy").astype(np.int64)
    Xfull = np.load(PROCESSED / "X_padded.npy", mmap_mode="r")
    datos = {"fecha": dt.datetime.now().isoformat(timespec="seconds"), "script": "src/simular_ventana_25.py",
             "fps_corpus": FPS_CORPUS, "ventana_s": VENTANA_S, "n_ventana_fotogramas": N_VENTANA,
             "modelos": str(CONFIRMACION.relative_to(BASE_DIR)), "semillas": args.semillas,
             "umbral_prototipo": UMBRAL_PROTOTIPO}
    datos["duracion_clips"] = duraciones_clips()
    datos["duracion_sena"], tramos = duraciones_sena(Xfull, lengths, nombres)
    datos["cortes"] = cortes_por_variante(lengths, nombres, tramos)
    datos["vivo"] = avisos_en_vivo()
    if not args.sin_modelo:
        datos["simulacion"] = simular(args, tramos)
    datos["duracion_s"] = round(time.perf_counter() - t0, 1)
    JSON_SALIDA.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")
    MD_SALIDA.write_text(escribir_md(datos), encoding="utf-8")
    print(f"escrito {JSON_SALIDA.relative_to(BASE_DIR)} y {MD_SALIDA.relative_to(BASE_DIR)} "
          f"({datos['duracion_s']} s)")


if __name__ == "__main__":
    main()
