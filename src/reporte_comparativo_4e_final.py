"""
Reporte comparativo de los dos esquemas de partición con el modelo final (actividad 4.e, RE6).

SOLO REPORTE: lee predicciones ya guardadas; no entrena, no reevalúa modelos y no decide nada. Genera las
cifras de docs/REPORTE_COMPARATIVO_4e_MODELO_FINAL.md en docs/REPORTE_COMPARATIVO_4e_MODELO_FINAL.json y
tres figuras sin personas en docs/figuras_2026-11-08/ (prefijo 4e_).

MODELOS Y FUENTES
=================
  final     TCN + imputación + 252 rasgos (D-14), 156.402 parámetros.
            LOSO: models/confirmacion_tcn_2026-10-03/T1_imputacion_252/semilla_{2026,2027,2028}/pliegue_k/
                  (predicciones.json, resultado.json, probabilidades.npz con la validación).
            Aleatoria: models/tcn_imputacion/evaluacion_modelo_seleccionado/predicciones.json (semilla 2030,
                  elegida por validación entre 2026-2030).
  heredado  Conv1D + BiLSTM + atención, «mascara», 252 rasgos, 714.291 parámetros.
            LOSO: models/experimento_mano_ausente_2026-10-02/A_mascara/semilla_{2026,2027,2028}/pliegue_k/
                  (predicciones.json, metricas.json). Es el brazo R0 de N-62/N-63.
            Aleatoria: models/canonico_holistic_mascara_manos/evaluacion_modelo_seleccionado/predicciones.json
                  (semilla 2026, elegida por validación entre 2026-2030).

ESTADÍSTICOS (los mismos de N-62 a N-67 y del capítulo 5)
==========================================================
  LOSO: métrica de cada pliegue (= señante) → media de los 5 pliegues → media de las 3 semillas. top-1 y
        top-3 = fracción de aciertos; F1, precision y recall macro = media sobre las señas presentes en el
        pliegue (reales o predichas), como sklearn, con precision 0 para una seña nunca predicha.
  Aleatoria: un solo modelo sobre los 100 videos de prueba (2 por seña).
  Brecha = aleatoria − LOSO, del mismo modelo. Diferencia = final − heredado, en el mismo esquema.

INTERVALOS (método del script del capítulo 5, c5_intervalos_confianza.py, reimplementado aquí)
=============================================================================================
  10.000 réplicas, percentil 2,5-97,5, semilla 20261108.
  (a) Por videos, estratificado por seña: en cada esquema se remuestrean con reemplazo, dentro de cada
      seña, tantos videos como tiene. Los videos LOSO conservan su señante. LOSO y aleatoria se remuestrean
      por separado (son conjuntos de prueba distintos).
  (b) Jerárquico por señantes: se sacan 5 señantes con reemplazo y, dentro de cada uno, sus videos con
      reemplazo. LOSO: cada señante sacado es un bloque y el estadístico es la media de los bloques.
      Aleatoria: exactitud sobre todos los videos sacados. A DIFERENCIA del capítulo 5, la misma extracción
      de señantes se usa para LOSO y para la aleatoria en cada réplica, de modo que la brecha queda pareada
      por persona; los intervalos marginales estiman lo mismo que los del capítulo 5.
  Diferencias final − heredado: pareadas (mismos pesos para los dos modelos en cada réplica).
  Se guarda también el sesgo medio de las réplicas y el intervalo con corrección de sesgo (BC, Efron) como
  comprobación: el F1 macro es no lineal y baja en las réplicas que repiten videos.

Uso:
    python src/reporte_comparativo_4e_final.py
    python src/reporte_comparativo_4e_final.py --replicas 200 --salida /tmp/x.json --sin-figuras
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402

C = 50
SEMILLAS = (2026, 2027, 2028)
SENANTES = ("0000", "0001", "0002", "0003", "0004")
SEMILLA_BOOTSTRAP = 20261108
MODELOS = ("heredado", "final")

LOSO_DIR = {"heredado": BASE_DIR / "models/experimento_mano_ausente_2026-10-02/A_mascara",
            "final": BASE_DIR / "models/confirmacion_tcn_2026-10-03/T1_imputacion_252"}
ALEATORIA = {"heredado": BASE_DIR / "models/canonico_holistic_mascara_manos/evaluacion_modelo_seleccionado",
             "final": BASE_DIR / "models/tcn_imputacion/evaluacion_modelo_seleccionado"}
RESUMEN_CONFIRMACION = BASE_DIR / "models/confirmacion_tcn_2026-10-03/resultados_confirmacion_tcn.json"
N_PARAMETROS = {"heredado": 714291, "final": 156402}
DESCRIPCION = {"heredado": "Conv1D + BiLSTM + atención, «mascara», 252 rasgos (red heredada; R0 de N-62)",
               "final": "TCN + imputación + 252 rasgos (D-14)"}
SEMILLA_ALEATORIA = {"heredado": 2026, "final": 2030}
# Grupos de señas de N-30 (copiados de src/analisis_errores.GRUPOS_N30; el resto es «resto»).
GRUPOS_N30 = {"interrogativos": [41, 42, 43, 44, 45, 46, 26], "emociones": [20, 21, 22, 23],
              "valoracion": [24, 25, 27]}

C5_JSON = BASE_DIR.parent / "Documento_Tesis" / "scripts_figuras" / "c5_intervalos_confianza.json"
SALIDA_JSON = BASE_DIR / "docs/REPORTE_COMPARATIVO_4e_MODELO_FINAL.json"
DIR_FIGURAS = BASE_DIR / "docs/figuras_2026-11-08"

# Cifras que ya circulan (encargo del 8-nov, N-62/N-63/N-65, capítulo 5). Solo se COMPARAN; nada se ajusta.
CIFRAS_CONOCIDAS = {
    "final|loso|top1": 0.677, "final|loso|top3": 0.847, "final|loso|f1_macro": 0.647,
    "final|loso|senante_0000_top1": 0.488, "final|aleatoria|top1": 0.94, "final|aleatoria|f1_macro": 0.937,
    "heredado|loso|top1": 0.589, "heredado|loso|top3": 0.764, "heredado|loso|f1_macro": 0.544,
    "heredado|loso|senante_0000_top1": 0.452, "heredado|aleatoria|top1": 0.88,
    "ic_b|final|loso|top1": (0.571, 0.770), "ic_b|heredado|loso|top1": (0.509, 0.661),
    "ic_b|diferencia|loso|top1": (0.053, 0.121), "ic_b|diferencia|aleatoria|top1": (-0.010, 0.132),
    "mcnemar_voto|solo_final": 163, "mcnemar_voto|solo_heredado": 56, "mcnemar_voto|p": 2.6e-13,
}
TOL_CONOCIDAS = 0.0015      # medio punto de redondeo a 3 decimales, más holgura de Monte Carlo en los IC
TOL_IC = 0.006


# ---------------------------------------------------------------------------------------------------
# Utilidades puras (probadas en tests/test_reporte_comparativo_4e_final.py)
# ---------------------------------------------------------------------------------------------------

def parsear_muestra(nombre: str) -> tuple[int, int, int]:
    """'SSSS_VVVV_RRRR' → (seña, señante, repetición) como enteros."""
    partes = nombre.split("_")
    if len(partes) != 3 or not all(p.isdigit() and len(p) == 4 for p in partes):
        raise ValueError(f"nombre de muestra no reconocido: {nombre!r}")
    return int(partes[0]), int(partes[1]), int(partes[2])


def metricas_macro(y, p, n: int = C) -> dict:
    """precision, recall y F1 macro sobre las señas presentes (reales o predichas), convención sklearn:
    precision 0 si la seña nunca se predijo; recall 0 si nunca aparece como real (no ocurre aquí)."""
    y, p = np.asarray(y), np.asarray(p)
    tp = np.bincount(y, weights=(y == p).astype(float), minlength=n)
    nr = np.bincount(y, minlength=n).astype(float)
    npred = np.bincount(p, minlength=n).astype(float)
    pres = (nr + npred) > 0
    prec = np.divide(tp, npred, out=np.zeros(n), where=npred > 0)
    rec = np.divide(tp, nr, out=np.zeros(n), where=nr > 0)
    f1 = np.divide(2 * tp, nr + npred, out=np.zeros(n), where=pres)
    return {"precision_macro": float(prec[pres].mean()), "recall_macro": float(rec[pres].mean()),
            "f1_macro": float(f1[pres].mean())}


def recall_por_clase(y, p, n: int = C) -> np.ndarray:
    """Recall de cada seña; NaN si la seña no aparece como real."""
    y, p = np.asarray(y), np.asarray(p)
    tp = np.bincount(y, weights=(y == p).astype(float), minlength=n)
    nr = np.bincount(y, minlength=n).astype(float)
    return np.divide(tp, nr, out=np.full(n, np.nan), where=nr > 0)


def voto_semillas(preds_por_semilla: list) -> dict:
    """Voto de mayoría entre semillas; si todas discrepan gana la de mayor probabilidad predicha.
    preds: lista de {muestra: (real, predicha, prob)}. Mismo criterio que
    src/experimento_confirmacion_tcn.voto_semillas (no se importa para no cargar TensorFlow)."""
    claves = sorted(preds_por_semilla[0])
    out = {}
    for c in claves:
        votos = [p[c] for p in preds_por_semilla]
        clases = [v[1] for v in votos]
        cuenta = {cl: clases.count(cl) for cl in set(clases)}
        mx = max(cuenta.values())
        elegida = [cl for cl in clases if cuenta[cl] == mx][0] if mx > 1 else max(votos, key=lambda v: v[2])[1]
        out[c] = (votos[0][0], int(elegida))
    return out


def mcnemar(acierto_a, acierto_b) -> dict:
    """McNemar exacto (binomial bilateral sobre los discordantes) y χ² con corrección."""
    from scipy.stats import binomtest, chi2
    a, b = np.asarray(acierto_a, bool), np.asarray(acierto_b, bool)
    solo_a, solo_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = solo_a + solo_b
    chi = (abs(solo_a - solo_b) - 1) ** 2 / n if n else 0.0
    return {"n": int(len(a)), "aciertos_a": int(a.sum()), "aciertos_b": int(b.sum()), "solo_a": solo_a,
            "solo_b": solo_b, "ambos": int(np.sum(a & b)), "ninguno": int(np.sum(~a & ~b)),
            "p_exacto_bilateral": float(binomtest(solo_a, n, 0.5).pvalue) if n else 1.0,
            "chi2_con_correccion": float(chi), "p_chi2": float(chi2.sf(chi, 1)) if n else 1.0}


def stats_bloques(pred, top3, y, bloques) -> np.ndarray:
    """Estadístico LOSO con pesos. pred, top3: (S, N) clase predicha y acierto top-3 por semilla; y: (N,).
    bloques: lista de (índices, pesos), un bloque por señante (puede repetirse). Devuelve
    [top-1, top-3, F1 macro] = media de bloques de la métrica del bloque, media de semillas."""
    S = pred.shape[0]
    nb = len(bloques)
    idx = np.concatenate([b[0] for b in bloques])
    w = np.concatenate([np.asarray(b[1], float) for b in bloques])
    blk = np.repeat(np.arange(nb), [len(b[0]) for b in bloques])
    yy, P, T3 = y[idx], pred[:, idx], top3[:, idx].astype(float)
    A = (P == yy).astype(float)
    W = np.broadcast_to(w, P.shape)
    kb = np.arange(S)[:, None] * nb + blk[None, :]
    sw = np.bincount(blk, weights=w, minlength=nb)
    t1 = np.bincount(kb.ravel(), weights=(W * A).ravel(), minlength=S * nb).reshape(S, nb) / sw
    t3 = np.bincount(kb.ravel(), weights=(W * T3).ravel(), minlength=S * nb).reshape(S, nb) / sw
    kt = (kb * C + yy[None, :]).ravel()
    kp = (kb * C + P).ravel()
    m = S * nb * C
    tp = np.bincount(kt, weights=(W * A).ravel(), minlength=m)
    den = np.bincount(kt, weights=W.ravel(), minlength=m) + np.bincount(kp, weights=W.ravel(), minlength=m)
    f1c = np.divide(2 * tp, den, out=np.full(m, np.nan), where=den > 0).reshape(S, nb, C)
    f1 = np.nanmean(f1c, axis=2)
    return np.array([t1.mean(), t3.mean(), f1.mean()])


def stats_pesos(acierto, top3, y, pred, w) -> np.ndarray:
    """Esquema aleatorio (un modelo): [top-1, top-3, F1 macro] ponderados por w."""
    w = np.asarray(w, float)
    sw = w.sum()
    tp = np.bincount(y, weights=w * acierto, minlength=C)
    den = np.bincount(y, weights=w, minlength=C) + np.bincount(pred, weights=w, minlength=C)
    f1 = np.divide(2 * tp, den, out=np.full(C, np.nan), where=den > 0)
    return np.array([(acierto @ w) / sw, (top3 @ w) / sw, np.nanmean(f1)])


def pesos_estratificados(rng, grupos, n: int) -> np.ndarray:
    """Remuestreo con reemplazo dentro de cada grupo (lista de arrays de índices); devuelve multiplicidades."""
    w = np.zeros(n)
    for idx in grupos:
        np.add.at(w, rng.choice(idx, size=len(idx), replace=True), 1)
    return w


def multiplicidades(rng, k: int) -> np.ndarray:
    return np.bincount(rng.integers(0, k, size=k), minlength=k).astype(float)


def percentil(v, q=(2.5, 97.5)) -> list:
    return [float(x) for x in np.percentile(v, q)]


def ic_bc(v, t) -> list:
    """Percentil con corrección de sesgo (BC, Efron)."""
    from scipy.stats import norm
    z0 = norm.ppf(np.clip(np.mean(np.asarray(v) < t), 1e-4, 1 - 1e-4))
    lo, hi = norm.cdf(2 * z0 - 1.959964), norm.cdf(2 * z0 + 1.959964)
    return [float(x) for x in np.percentile(v, [100 * lo, 100 * hi])]


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


# ---------------------------------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------------------------------

def _leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(p)


def cargar_loso(modelo: str, raiz: Path | None = None) -> dict:
    """Predicciones de prueba de las 3 semillas × 5 pliegues, en el mismo orden de muestras."""
    raiz = raiz or LOSO_DIR[modelo]
    muestras, y = None, None
    pred, top3, prob, guardado, val = [], [], [], [], []
    val_pred = {}
    for s in SEMILLAS:
        regs = {}
        g_s, v_s = [], []
        for k in range(5):
            d = raiz / f"semilla_{s}" / f"pliegue_{k}"
            for r in _leer(d / "predicciones.json"):
                if parsear_muestra(r["muestra"])[1] != k:
                    raise AssertionError(f"{d}: {r['muestra']} no es del señante {k}")
                regs[r["muestra"]] = r
            if (d / "resultado.json").exists():
                m = _leer(d / "resultado.json")
                g_s.append({"top1": m["test_top1"], "top3": m["test_top3"], "f1_macro": m["test_f1_macro"],
                            "precision_macro": m["test_precision_macro"],
                            "recall_macro": m["test_recall_macro"], "n_test": m["n_test"]})
                v_s.append(m["val_top1"])
            else:
                m = _leer(d / "metricas.json")
                g_s.append({"top1": m["top1"], "top3": m["top3"], "f1_macro": m["f1_macro"],
                            "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"],
                            "n_test": m["n_test"]})
                v_s.append(m["mejor_val_accuracy"])
            npz = d / "probabilidades.npz"
            if npz.exists():
                z = np.load(npz)
                val_pred[(s, k)] = (list(z["nombres_val"]), np.argmax(z["val"], axis=1))
        orden = sorted(regs)
        if muestras is None:
            muestras = orden
            y = np.array([regs[m]["real"] for m in orden])
        if orden != muestras or not np.array_equal(y, [regs[m]["real"] for m in orden]):
            raise AssertionError(f"{modelo}: las semillas no tienen las mismas muestras de prueba")
        pred.append([regs[m]["predicha"] for m in orden])
        top3.append([regs[m]["real"] in regs[m]["top3"] for m in orden])
        prob.append([regs[m]["prob_predicha"] for m in orden])
        guardado.append(g_s)
        val.append(v_s)
    return {"modelo": modelo, "dir": _rel(raiz), "muestras": muestras, "y": y,
            "g": np.array([parsear_muestra(m)[1] for m in muestras]),
            "pred": np.array(pred), "top3": np.array(top3, bool), "prob": np.array(prob),
            "guardado": guardado, "val_top1": np.array(val), "val_pred": val_pred}


def cargar_aleatoria(modelo: str, d: Path | None = None) -> dict:
    d = d or ALEATORIA[modelo]
    regs = sorted(_leer(d / "predicciones.json"), key=lambda r: r["muestra"])
    met = _leer(d / "metricas.json")
    y = np.array([r["real"] for r in regs])
    p = np.array([r["predicha"] for r in regs])
    return {"modelo": modelo, "dir": _rel(d), "muestras": [r["muestra"] for r in regs], "y": y, "pred": p,
            "acierto": (y == p).astype(float), "top3": np.array([r["real"] in r["top3"] for r in regs], float),
            "g": np.array([parsear_muestra(r["muestra"])[1] for r in regs]), "guardado": met}


# ---------------------------------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------------------------------

def resumen_loso(L: dict) -> dict:
    """Métricas por semilla y pliegue, recalculadas desde las predicciones, y su comparación con lo guardado."""
    S = len(SEMILLAS)
    nombres = ["top1", "top3", "precision_macro", "recall_macro", "f1_macro"]
    M = {n: np.zeros((S, 5)) for n in nombres}
    max_dif = 0.0
    for i in range(S):
        for k in range(5):
            idx = np.flatnonzero(L["g"] == k)
            y, p = L["y"][idx], L["pred"][i, idx]
            M["top1"][i, k] = np.mean(y == p)
            M["top3"][i, k] = L["top3"][i, idx].mean()
            for n, v in metricas_macro(y, p).items():
                M[n][i, k] = v
            for n in nombres:
                max_dif = max(max_dif, abs(M[n][i, k] - L["guardado"][i][k][n]))
    out = {}
    for n in nombres:
        pp = M[n].mean(0)
        out[n] = {"media": float(M[n].mean()), "por_pliegue_media_semillas": pp.tolist(),
                  "de_entre_pliegues": float(pp.std(ddof=1)),
                  "por_semilla_media_pliegues": M[n].mean(1).tolist(),
                  "de_entre_semillas": float(M[n].mean(1).std(ddof=1)),
                  "por_semilla_por_pliegue": M[n].tolist()}
    agregado = {"top1_997_media_semillas": float(np.mean(L["pred"] == L["y"])),
                "f1_macro_997_media_semillas": float(np.mean([metricas_macro(L["y"], L["pred"][i])["f1_macro"]
                                                              for i in range(S)]))}
    v = L["val_top1"]
    vpp = v.mean(0)
    out["validacion_top1"] = {"media": float(v.mean()), "por_pliegue_media_semillas": vpp.tolist(),
                              "de_entre_pliegues": float(vpp.std(ddof=1)), "min": float(v.min()),
                              "max": float(v.max()), "por_semilla_por_pliegue": v.tolist(),
                              "valores_distintos": sorted({round(float(x), 2) for x in v.ravel()})}
    brecha_vt = v - M["top1"]
    out["validacion_menos_prueba_top1"] = {"media": float(brecha_vt.mean()),
                                           "por_pliegue_media_semillas": brecha_vt.mean(0).tolist()}
    out["agregado_997"] = agregado
    out["control_contra_guardado"] = {"max_diferencia_absoluta": max_dif, "coincide": bool(max_dif < 1e-6)}
    return out


def resumen_aleatoria(A: dict) -> dict:
    mm = metricas_macro(A["y"], A["pred"])
    out = {"n_test": int(len(A["y"])), "top1": float(A["acierto"].mean()), "top3": float(A["top3"].mean()),
           **mm, "n_errores": int(len(A["y"]) - A["acierto"].sum())}
    dif = max(abs(out[k] - A["guardado"][k]) for k in ("top1", "top3", "precision_macro", "recall_macro",
                                                         "f1_macro"))
    out["control_contra_guardado"] = {"max_diferencia_absoluta": float(dif), "coincide": bool(dif < 1e-6),
                                      "fuente": A["dir"] + "/metricas.json"}
    out["por_senante"] = {SENANTES[k]: {"n": int(np.sum(A["g"] == k)),
                                        "aciertos": int(A["acierto"][A["g"] == k].sum()),
                                        "top1": float(A["acierto"][A["g"] == k].mean())} for k in range(5)}
    return out


def validacion_por_senante_y_clase(L: dict) -> dict | None:
    """Exactitud de la validación LOSO (señantes VISTOS) por señante y recall por seña, desde
    probabilidades.npz. Solo existe para el modelo final."""
    if not L["val_pred"]:
        return None
    S = len(SEMILLAS)
    acc_sen = np.zeros((S, 5))
    n_sen = np.zeros((S, 5), int)
    rec = np.zeros((S, C))
    reproduce = []
    for i, s in enumerate(SEMILLAS):
        ys, ps, gs = [], [], []
        for k in range(5):
            nombres, p = L["val_pred"][(s, k)]
            y = np.array([parsear_muestra(m)[0] for m in nombres])
            g = np.array([parsear_muestra(m)[1] for m in nombres])
            if k in g:
                raise AssertionError(f"la validación del pliegue {k} contiene al señante de prueba")
            reproduce.append(abs(float(np.mean(y == p)) - float(L["val_top1"][i, k])))
            ys.append(y), ps.append(p), gs.append(g)
        y, p, g = np.concatenate(ys), np.concatenate(ps), np.concatenate(gs)
        for k in range(5):
            acc_sen[i, k] = np.mean(y[g == k] == p[g == k])
            n_sen[i, k] = int(np.sum(g == k))
        rec[i] = recall_por_clase(y, p)
    return {"top1_por_senante_media_semillas": dict(zip(SENANTES, acc_sen.mean(0).tolist())),
            "n_por_senante_por_semilla": dict(zip(SENANTES, n_sen[0].tolist())),
            "recall_por_clase_media_semillas": rec.mean(0).tolist(),
            "control_reproduce_val_top1": {"max_diferencia_absoluta": float(max(reproduce)),
                                           "coincide": bool(max(reproduce) < 1e-6)}}


def bootstrap(Ls: dict, As: dict, B: int, semilla: int = SEMILLA_BOOTSTRAP) -> dict:
    """Réplicas de [t1L, t3L, f1L, t1A, t3A, f1A] por modelo, métodos (a) y (b)."""
    rng = np.random.default_rng(semilla)
    L0, A0 = Ls["final"], As["final"]
    for m in MODELOS:
        assert np.array_equal(Ls[m]["y"], L0["y"]) and Ls[m]["muestras"] == L0["muestras"]
        assert np.array_equal(As[m]["y"], A0["y"]) and As[m]["muestras"] == A0["muestras"]
    y, g, ya, ga = L0["y"], L0["g"], A0["y"], A0["g"]
    N, NA = len(y), len(ya)
    idx_sen = [np.flatnonzero(g == k) for k in range(5)]
    idx_sena = [np.flatnonzero(y == c) for c in range(C)]
    idxa_sen = [np.flatnonzero(ga == k) for k in range(5)]
    idxa_sena = [np.flatnonzero(ya == c) for c in range(C)]
    R = {met: {m: np.zeros((B, 6)) for m in MODELOS} for met in ("a", "b")}
    for r in range(B):
        # (a) por videos, estratificado por seña, cada esquema por separado
        w = pesos_estratificados(rng, idx_sena, N)
        bl = [(idx, w[idx]) for idx in idx_sen]
        wa = pesos_estratificados(rng, idxa_sena, NA)
        for m in MODELOS:
            R["a"][m][r, :3] = stats_bloques(Ls[m]["pred"], Ls[m]["top3"], y, bl)
            R["a"][m][r, 3:] = stats_pesos(As[m]["acierto"], As[m]["top3"], ya, As[m]["pred"], wa)
        # (b) jerárquico por señantes, la misma extracción para los dos esquemas
        ks = rng.integers(0, 5, size=5)
        bl = [(idx_sen[k], multiplicidades(rng, len(idx_sen[k]))) for k in ks]
        wa = np.zeros(NA)
        for k in ks:
            np.add.at(wa, idxa_sen[k], multiplicidades(rng, len(idxa_sen[k])))
        for m in MODELOS:
            R["b"][m][r, :3] = stats_bloques(Ls[m]["pred"], Ls[m]["top3"], y, bl)
            R["b"][m][r, 3:] = stats_pesos(As[m]["acierto"], As[m]["top3"], ya, As[m]["pred"], wa)
    return R


ETIQUETAS = ["loso|top1", "loso|top3", "loso|f1_macro", "aleatoria|top1", "aleatoria|top3", "aleatoria|f1_macro"]


def intervalos(Ls: dict, As: dict, B: int, semilla: int = SEMILLA_BOOTSTRAP) -> dict:
    y, ya = Ls["final"]["y"], As["final"]["y"]
    bl0 = [(np.flatnonzero(Ls["final"]["g"] == k), np.ones(int(np.sum(Ls["final"]["g"] == k))))
           for k in range(5)]
    punto = {m: np.r_[stats_bloques(Ls[m]["pred"], Ls[m]["top3"], y, bl0),
                      stats_pesos(As[m]["acierto"], As[m]["top3"], ya, As[m]["pred"], np.ones(len(ya)))]
             for m in MODELOS}
    R = bootstrap(Ls, As, B, semilla)

    def derivar(v):           # añade las brechas aleatoria − LOSO (top-1, top-3, F1)
        return np.c_[v, v[..., 3:6] - v[..., 0:3]]

    etq = ETIQUETAS + ["brecha|top1", "brecha|top3", "brecha|f1_macro"]
    P = {m: derivar(punto[m][None, :])[0] for m in MODELOS}
    P["diferencia"] = P["final"] - P["heredado"]
    out = {"replicas": B, "semilla": semilla, "nivel": 0.95, "metodo": "percentil", "estadisticos": {}}
    for met in ("a", "b"):
        V = {m: derivar(R[met][m]) for m in MODELOS}
        V["diferencia"] = V["final"] - V["heredado"]
        for quien in ("heredado", "final", "diferencia"):
            for j, e in enumerate(etq):
                d = out["estadisticos"].setdefault(f"{quien}|{e}", {"punto": float(P[quien][j])})
                d[f"ic_{met}"] = percentil(V[quien][:, j])
                d[f"sesgo_{met}"] = float(V[quien][:, j].mean() - P[quien][j])
                d[f"ic_bc_{met}"] = ic_bc(V[quien][:, j], P[quien][j])
                d[f"fraccion_replicas_mayor_que_0_{met}"] = float(np.mean(V[quien][:, j] > 0))
    return out


def brecha_por_senante(Ls: dict, As: dict, B: int, semilla: int = SEMILLA_BOOTSTRAP + 1) -> dict:
    """Top-1 aleatoria (señante visto, ~20 videos) − top-1 LOSO (no visto, ~200 videos × 3 semillas),
    por señante; IC percentil remuestreando videos dentro del señante (independiente en cada esquema)."""
    rng = np.random.default_rng(semilla)
    out = {}
    for m in MODELOS:
        L, A = Ls[m], As[m]
        acL = (L["pred"] == L["y"]).astype(float)
        out[m] = {}
        for k in range(5):
            il, ia = np.flatnonzero(L["g"] == k), np.flatnonzero(A["g"] == k)
            loso = float(acL[:, il].mean())
            ale = float(A["acierto"][ia].mean())
            rl = rng.integers(0, len(il), size=(B, len(il)))
            ra = rng.integers(0, len(ia), size=(B, len(ia)))
            bl = acL[:, il][:, rl].mean(axis=(0, 2))
            ba = A["acierto"][ia][ra].mean(1)
            out[m][SENANTES[k]] = {"loso_top1": loso, "n_loso": int(len(il)), "aleatoria_top1": ale,
                                   "n_aleatoria": int(len(ia)), "brecha": ale - loso,
                                   "ic_brecha_por_videos": percentil(ba - bl),
                                   "ic_loso_por_videos": percentil(bl), "ic_aleatoria_por_videos": percentil(ba)}
    return out


def por_clase(Ls: dict, As: dict, val_final: dict | None) -> dict:
    rec = {}
    for m in MODELOS:
        L, A = Ls[m], As[m]
        rl = np.mean([recall_por_clase(L["y"], L["pred"][i]) for i in range(len(SEMILLAS))], axis=0)
        ra = recall_por_clase(A["y"], A["pred"])
        rec[m] = {"loso": rl, "aleatoria": ra, "caida": ra - rl}
    filas = []
    for c in range(C):
        f = {"id": c, "sena": SIGN_VOCABULARY[c], "n_loso": int(np.sum(Ls["final"]["y"] == c)),
             "n_aleatoria": int(np.sum(As["final"]["y"] == c))}
        for m in MODELOS:
            f[f"{m}_recall_aleatoria"] = float(rec[m]["aleatoria"][c])
            f[f"{m}_recall_loso"] = float(rec[m]["loso"][c])
            f[f"{m}_caida"] = float(rec[m]["caida"][c])
        if val_final:
            f["final_recall_validacion_loso"] = float(val_final["recall_por_clase_media_semillas"][c])
            f["final_caida_validacion_menos_prueba"] = f["final_recall_validacion_loso"] - f["final_recall_loso"]
        filas.append(f)
    orden = sorted(filas, key=lambda f: (-f["final_caida"], f["final_recall_loso"], f["id"]))
    out = {"senas": filas,
           "mayor_caida_final_10": [f["id"] for f in orden[:10]],
           "peor_recall_loso_final_10": [f["id"] for f in sorted(filas, key=lambda f: (f["final_recall_loso"],
                                                                                        f["id"]))[:10]],
           "peor_recall_loso_heredado_10": [f["id"] for f in sorted(filas, key=lambda f: (f["heredado_recall_loso"],
                                                                                            f["id"]))[:10]]}
    for m in MODELOS:
        r = rec[m]
        out[f"resumen_{m}"] = {
            "caida_media": float(np.mean(r["caida"])), "caida_mediana": float(np.median(r["caida"])),
            "senas_recall_loso_menor_0_5": int(np.sum(r["loso"] < 0.5)),
            "senas_recall_loso_mayor_igual_0_9": int(np.sum(r["loso"] >= 0.9)),
            "senas_recall_aleatoria_1": int(np.sum(r["aleatoria"] == 1.0)),
            "senas_recall_aleatoria_0_5": int(np.sum(r["aleatoria"] == 0.5)),
            "senas_recall_aleatoria_0": int(np.sum(r["aleatoria"] == 0.0)),
            "senas_con_caida_mayor_0_5": int(np.sum(r["caida"] > 0.5)),
            "senas_con_caida_negativa": int(np.sum(r["caida"] < 0))}
    grupos = dict(GRUPOS_N30)
    grupos["resto"] = [c for c in range(C) if not any(c in v for v in GRUPOS_N30.values())]
    out["por_grupo_n30"] = {g: {"n_senas": len(ids), **{f"{m}_{q}": float(np.mean(rec[m][q][ids]))
                                                       for m in MODELOS for q in ("aleatoria", "loso", "caida")}}
                            for g, ids in grupos.items()}
    d = rec["final"]["loso"] - rec["heredado"]["loso"]
    out["final_vs_heredado_loso"] = {"spearman_recall_loso": spearman(rec["final"]["loso"], rec["heredado"]["loso"]),
                                     "senas_mejoran": int(np.sum(d > 1e-9)), "senas_empeoran": int(np.sum(d < -1e-9)),
                                     "senas_igual": int(np.sum(np.abs(d) <= 1e-9)),
                                     "comunes_peores_10": sorted(set(out["peor_recall_loso_final_10"])
                                                                 & set(out["peor_recall_loso_heredado_10"]))}
    if val_final:
        vr = np.array(val_final["recall_por_clase_media_semillas"])
        out["control_caida_con_validacion_final"] = {
            "spearman_caida_aleatoria_vs_caida_validacion": spearman(rec["final"]["caida"], vr - rec["final"]["loso"]),
            "recall_validacion_media": float(vr.mean()),
            "mayor_caida_validacion_10": [int(i) for i in np.argsort(-(vr - rec["final"]["loso"]), kind="stable")[:10]]}
    return out


def mcnemar_final_vs_heredado(Ls: dict, As: dict) -> dict:
    out = {"loso_por_semilla": {}}
    for i, s in enumerate(SEMILLAS):
        out["loso_por_semilla"][str(s)] = mcnemar(Ls["final"]["pred"][i] == Ls["final"]["y"],
                                                  Ls["heredado"]["pred"][i] == Ls["heredado"]["y"])
    voto = {}
    for m in MODELOS:
        L = Ls[m]
        preds = [{mu: (int(L["y"][j]), int(L["pred"][i, j]), float(L["prob"][i, j]))
                  for j, mu in enumerate(L["muestras"])} for i in range(len(SEMILLAS))]
        voto[m] = voto_semillas(preds)
    cl = sorted(voto["final"])
    af = np.array([voto["final"][c][0] == voto["final"][c][1] for c in cl])
    ah = np.array([voto["heredado"][c][0] == voto["heredado"][c][1] for c in cl])
    out["loso_voto_3_semillas"] = mcnemar(af, ah)
    out["loso_voto_3_semillas"]["exactitud_voto"] = {"final": float(af.mean()), "heredado": float(ah.mean())}
    out["aleatoria"] = mcnemar(As["final"]["acierto"], As["heredado"]["acierto"])
    out["nota"] = ("a = final, b = heredado. Descriptivo: trata como independientes videos de 5 personas "
                   "(no lo son); la incertidumbre por persona está en los intervalos (b).")
    if RESUMEN_CONFIRMACION.exists():
        g = _leer(RESUMEN_CONFIRMACION)["mcnemar"]["T1_vs_R0"]["voto_3_semillas"]
        v = out["loso_voto_3_semillas"]
        out["control_contra_guardado"] = {
            "fuente": _rel(RESUMEN_CONFIRMACION) + " → mcnemar.T1_vs_R0.voto_3_semillas",
            "coincide": bool(g["solo_brazo_acierta"] == v["solo_a"] and g["solo_R0_acierta"] == v["solo_b"]
                             and abs(g["p_exacto_bilateral"] - v["p_exacto_bilateral"]) < 1e-15)}
    return out


def comparar_conocidas(res: dict) -> dict:
    lo, al, ic = res["loso"], res["aleatoria"], res["intervalos"]["estadisticos"]
    mc = res["mcnemar"]["loso_voto_3_semillas"]
    medido = {
        "final|loso|top1": lo["final"]["top1"]["media"], "final|loso|top3": lo["final"]["top3"]["media"],
        "final|loso|f1_macro": lo["final"]["f1_macro"]["media"],
        "final|loso|senante_0000_top1": lo["final"]["top1"]["por_pliegue_media_semillas"][0],
        "final|aleatoria|top1": al["final"]["top1"], "final|aleatoria|f1_macro": al["final"]["f1_macro"],
        "heredado|loso|top1": lo["heredado"]["top1"]["media"], "heredado|loso|top3": lo["heredado"]["top3"]["media"],
        "heredado|loso|f1_macro": lo["heredado"]["f1_macro"]["media"],
        "heredado|loso|senante_0000_top1": lo["heredado"]["top1"]["por_pliegue_media_semillas"][0],
        "heredado|aleatoria|top1": al["heredado"]["top1"],
        "ic_b|final|loso|top1": ic["final|loso|top1"]["ic_b"], "ic_b|heredado|loso|top1": ic["heredado|loso|top1"]["ic_b"],
        "ic_b|diferencia|loso|top1": ic["diferencia|loso|top1"]["ic_b"],
        "ic_b|diferencia|aleatoria|top1": ic["diferencia|aleatoria|top1"]["ic_b"],
        "mcnemar_voto|solo_final": mc["solo_a"], "mcnemar_voto|solo_heredado": mc["solo_b"],
        "mcnemar_voto|p": mc["p_exacto_bilateral"]}
    out = {}
    for k, conocido in CIFRAS_CONOCIDAS.items():
        m = medido[k]
        if isinstance(conocido, tuple):
            ok = all(abs(a - b) <= TOL_IC for a, b in zip(m, conocido))
        elif k.endswith("|p"):
            ok = abs(m - conocido) / conocido < 0.05
        elif isinstance(conocido, int):
            ok = m == conocido
        else:
            ok = abs(m - conocido) <= TOL_CONOCIDAS
        out[k] = {"conocida": list(conocido) if isinstance(conocido, tuple) else conocido,
                  "medida": m, "coincide": bool(ok)}
    return out


def comparar_c5(ic: dict, ruta: Path = C5_JSON) -> dict | None:
    """Cotejo (solo lectura) con los intervalos del capítulo 5, si el archivo existe."""
    if not ruta.exists():
        return None
    c5 = _leer(ruta)
    mapa = {"top-1 LOSO": "loso|top1", "top-3 LOSO": "loso|top3", "F1 macro LOSO": "loso|f1_macro",
            "top-1 aleatoria": "aleatoria|top1"}
    out = {}
    for clave, v in c5.items():
        quien, nom = clave.split("|")
        nuestro = ic[f"{quien}|{mapa[nom]}"]
        out[clave] = {"punto_c5": v["punto"], "punto": nuestro["punto"], "ic_a_c5": v["a"], "ic_a": nuestro["ic_a"],
                      "ic_b_c5": v["b"], "ic_b": nuestro["ic_b"],
                      "max_dif_extremos": float(max(abs(np.r_[v["a"], v["b"]] - np.r_[nuestro["ic_a"],
                                                                                       nuestro["ic_b"]])))}
    return {"fuente": str(ruta), "estadisticos": out,
            "max_dif_punto": float(max(abs(o["punto_c5"] - o["punto"]) for o in out.values())),
            "max_dif_extremos": float(max(o["max_dif_extremos"] for o in out.values()))}


# ---------------------------------------------------------------------------------------------------
# Figuras (sin personas: solo cifras)
# ---------------------------------------------------------------------------------------------------

COLOR = {"heredado": "#eb6834", "final": "#2a78d6"}
MARCA = {"heredado": "o", "final": "s"}
NOMBRE = {"heredado": "Red heredada (Conv1D+BiLSTM+atención)", "final": "Modelo final (TCN + imputación)"}
TINTA, TINTA2, REJILLA = "#1f1f1f", "#52514e", "#e6e6e3"


def _coma(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def _estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
                         "savefig.bbox": "tight", "savefig.dpi": 200, "figure.facecolor": "white"})
    return plt


def figura_barras(res: dict, ruta: Path):
    plt = _estilo()
    ic = res["intervalos"]["estadisticos"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    metr = [("top1", "top-1"), ("top3", "top-3"), ("f1_macro", "F1 macro")]
    for ax, esq, tit in [(axes[0], "aleatoria", "(a) Partición aleatoria · señantes vistos\n100 videos, 1 modelo"),
                         (axes[1], "loso", "(b) LOSO · señante no visto\n997 videos, 3 semillas × 5 pliegues")]:
        for j, (mk, et) in enumerate(metr):
            for i, m in enumerate(MODELOS):
                d = ic[f"{m}|{esq}|{mk}"]
                x = j + (i - 0.5) * 0.36
                v = d["punto"] * 100
                lo, hi = (np.array(d["ic_b"]) * 100)
                ax.bar(x, v, width=0.34, color=COLOR[m], edgecolor="white", linewidth=1.5, zorder=2)
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=TINTA, elinewidth=1, capsize=2.5,
                            zorder=3)
                ax.text(x, 3, _coma(v), ha="center", va="bottom", fontsize=7.5, color="white", fontweight="bold",
                        zorder=4)
        ax.set_xticks(range(3))
        ax.set_xticklabels([e for _, e in metr])
        ax.set_ylim(0, 105)
        ax.set_ylabel("%  (F1 × 100)")
        ax.set_title(tit, fontsize=9, loc="left")
        ax.yaxis.grid(True, color=REJILLA, lw=0.6)
        ax.set_axisbelow(True)
    ax = axes[2]
    for j, (mk, et) in enumerate(metr):
        for i, m in enumerate(MODELOS):
            d = ic[f"{m}|brecha|{mk}"]
            yy = -j - (i - 0.5) * 0.36
            v = d["punto"] * 100
            lo, hi = np.array(d["ic_b"]) * 100
            ax.plot([lo, hi], [yy, yy], color=COLOR[m], lw=2, solid_capstyle="butt")
            ax.plot(v, yy, MARCA[m], color=COLOR[m], ms=7, mec="white", mew=1, zorder=3)
            ax.text(hi + 1, yy, _coma(v), va="center", fontsize=7.5, color=TINTA2)
    ax.axvline(0, color=TINTA2, lw=0.8, ls="--")
    ax.set_yticks([0, -1, -2])
    ax.set_yticklabels([e for _, e in metr])
    ax.set_xlabel("brecha aleatoria − LOSO (puntos)")
    ax.set_title("(c) Brecha de cada modelo\nIC 95 % jerárquico por señantes", fontsize=9, loc="left")
    ax.xaxis.grid(True, color=REJILLA, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-5, 62)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=COLOR[m], label=NOMBRE[m]) for m in MODELOS], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.5, -0.12, "Barras de error y segmentos: intervalo percentil al 95 %, bootstrap jerárquico por "
             "señantes (método b, 10.000 réplicas).", ha="center", fontsize=7.5, color=TINTA2)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_senantes(res: dict, ruta: Path):
    plt = _estilo()
    bs = res["brecha_por_senante"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, m in enumerate(MODELOS):
        for k, s in enumerate(SENANTES):
            d = bs[m][s]
            x = k + (i - 0.5) * 0.32
            ax.plot([x, x], [d["loso_top1"] * 100, d["aleatoria_top1"] * 100], color=COLOR[m], lw=1.6, zorder=2)
            ax.plot(x, d["loso_top1"] * 100, MARCA[m], color=COLOR[m], ms=7, mec="white", mew=1, zorder=3)
            ax.plot(x, d["aleatoria_top1"] * 100, MARCA[m], mfc="white", mec=COLOR[m], mew=1.6, ms=7, zorder=3)
            ax.text(x, d["loso_top1"] * 100 - 4.5, _coma(d["brecha"] * 100, 0), ha="center", va="top",
                    fontsize=7, color=TINTA2)
        ax.plot([], [], MARCA[m], color=COLOR[m], label=NOMBRE[m])
    ax.plot([], [], "o", mfc="white", mec=TINTA2, label="hueco: aleatoria (visto, 16-26 videos)")
    ax.plot([], [], "o", color=TINTA2, label="lleno: LOSO (no visto, 198-200 videos × 3 semillas)")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"señante {s}\n(n aleatoria = {bs['final'][s]['n_aleatoria']})" for s in SENANTES],
                       fontsize=7.5)
    ax.set_ylim(25, 104)
    ax.set_ylabel("top-1 (%)")
    ax.set_title("Brecha por señante: mismo señante visto (aleatoria) y no visto (LOSO)\n"
                 "número bajo cada par: brecha en puntos", fontsize=9, loc="left")
    ax.yaxis.grid(True, color=REJILLA, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_clases(res: dict, ruta: Path):
    plt = _estilo()
    filas = sorted(res["por_clase"]["senas"], key=lambda f: (f["final_recall_loso"], -f["id"]))
    fig, ax = plt.subplots(figsize=(6.8, 10.5))
    for yy, f in enumerate(filas):
        a, l, h = f["final_recall_aleatoria"] * 100, f["final_recall_loso"] * 100, f["heredado_recall_loso"] * 100
        ax.plot([l, a], [yy, yy], color=COLOR["final"], lw=1.2, alpha=0.6, zorder=1)
        ax.plot(h, yy, "|", color=COLOR["heredado"], ms=9, mew=2, zorder=2)
        ax.plot(l, yy, "s", color=COLOR["final"], ms=6, mec="white", mew=0.8, zorder=3)
        ax.plot(a, yy, "s", mfc="white", mec=COLOR["final"], mew=1.3, ms=6, zorder=3)
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels([f"{f['sena']} ({f['id']})" for f in filas], fontsize=7)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-1, len(filas))
    ax.set_xlabel("recall por seña (%)")
    ax.xaxis.grid(True, color=REJILLA, lw=0.6)
    ax.set_axisbelow(True)
    ax.plot([], [], "s", mfc="white", mec=COLOR["final"], label="final · aleatoria (2 videos por seña: 0, 50 o 100)")
    ax.plot([], [], "s", color=COLOR["final"], label="final · LOSO (≈20 videos × 3 semillas)")
    ax.plot([], [], "|", color=COLOR["heredado"], ms=9, mew=2, label="heredada · LOSO")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.035), fontsize=7.5, frameon=False)
    ax.set_title("Recall por seña con señantes vistos (aleatoria) y no vistos (LOSO)\n"
                 "ordenado por el recall LOSO del modelo final", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------------------------------

def calcular(B: int) -> dict:
    Ls = {m: cargar_loso(m) for m in MODELOS}
    As = {m: cargar_aleatoria(m) for m in MODELOS}
    val_final = validacion_por_senante_y_clase(Ls["final"])
    res = {
        "script": "src/reporte_comparativo_4e_final.py",
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "solo_reporte": "lee predicciones guardadas; no entrena ni reevalúa modelos",
        "modelos": {m: {"descripcion": DESCRIPCION[m], "n_parametros": N_PARAMETROS[m],
                        "loso_dir": Ls[m]["dir"], "loso_semillas": list(SEMILLAS),
                        "aleatoria_dir": As[m]["dir"], "aleatoria_semilla_seleccionada": SEMILLA_ALEATORIA[m]}
                    for m in MODELOS},
        "loso": {m: resumen_loso(Ls[m]) for m in MODELOS},
        "aleatoria": {m: resumen_aleatoria(As[m]) for m in MODELOS},
        "validacion_loso_final_por_senante": {k: v for k, v in (val_final or {}).items()
                                              if k != "recall_por_clase_media_semillas"},
    }
    res["intervalos"] = intervalos(Ls, As, B)
    res["brecha_por_senante"] = brecha_por_senante(Ls, As, B)
    res["por_clase"] = por_clase(Ls, As, val_final)
    res["mcnemar"] = mcnemar_final_vs_heredado(Ls, As)
    res["cifras_conocidas"] = comparar_conocidas(res)
    res["cotejo_capitulo_5"] = comparar_c5(res["intervalos"]["estadisticos"])
    ctrl = [res["loso"][m]["control_contra_guardado"]["coincide"] for m in MODELOS]
    ctrl += [res["aleatoria"][m]["control_contra_guardado"]["coincide"] for m in MODELOS]
    if val_final:
        ctrl.append(val_final["control_reproduce_val_top1"]["coincide"])
    if "control_contra_guardado" in res["mcnemar"]:
        ctrl.append(res["mcnemar"]["control_contra_guardado"]["coincide"])
    res["controles_todo_coincide"] = bool(all(ctrl))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--replicas", type=int, default=10_000)
    ap.add_argument("--salida", default=str(SALIDA_JSON))
    ap.add_argument("--dir-figuras", default=str(DIR_FIGURAS))
    ap.add_argument("--sin-figuras", action="store_true")
    a = ap.parse_args(argv)
    res = calcular(a.replicas)
    if not res["controles_todo_coincide"]:
        print("AVISO: algún control contra lo guardado NO coincide; revisar el JSON.")
    salida = Path(a.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    if not a.sin_figuras:
        d = Path(a.dir_figuras)
        d.mkdir(parents=True, exist_ok=True)
        figs = {"barras": d / "4e_barras_esquemas_modelos.png", "senantes": d / "4e_brecha_por_senante.png",
                "clases": d / "4e_recall_por_clase_aleatoria_vs_loso.png"}
        figura_barras(res, figs["barras"])
        figura_senantes(res, figs["senantes"])
        figura_clases(res, figs["clases"])
        res["figuras"] = {k: _rel(v) for k, v in figs.items()}
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    ic = res["intervalos"]["estadisticos"]
    for m in ("heredado", "final", "diferencia"):
        for e in ("loso|top1", "aleatoria|top1", "brecha|top1"):
            d = ic[f"{m}|{e}"]
            print(f"{m:10s} {e:15s} {d['punto']*100:6.1f}  (a) [{d['ic_a'][0]*100:6.1f};{d['ic_a'][1]*100:6.1f}]"
                  f"  (b) [{d['ic_b'][0]*100:6.1f};{d['ic_b'][1]*100:6.1f}]")
    malas = [k for k, v in res["cifras_conocidas"].items() if not v["coincide"]]
    print("cifras conocidas que no coinciden:", malas or "ninguna")
    print("controles:", res["controles_todo_coincide"], "→", _rel(salida))
    return res


if __name__ == "__main__":
    main()
