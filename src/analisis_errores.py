"""
Análisis de errores del modelo adoptado «manos» (actividad 4.d). SOLO REPORTE: no decide ni reentrena.

Fuente principal: las predicciones de PRUEBA ya guardadas del LOSO de 5 pliegues de la representación
adoptada (D-08, N-30): <loso-dir>/pliegue_k/predicciones.json, donde el pliegue k deja fuera al señante k
(nombre de muestra SSSS_VVVV_RRRR: seña, señante, repetición). Se agregan las 997 predicciones.

Antes de calcular nada se comprueba que las predicciones reproducen lo guardado: top-1, top-3 y F1 macro
por pliegue (resultados_loso.json y metricas.json), la matriz de confusión (matriz_confusion.json) y el
recall por clase (classification_report.json). Si el top-1 no coincide, el script se detiene.

Contenido (todas las cifras van a analisis.json; analisis.md las presenta con su clave de origen):
  1. Recall, precisión y F1 por seña (agregado y por pliegue, con desviación estándar muestral entre
     pliegues); peores y mejores 10.
  2. Pares de confusión real → predicha, simetría, y agrupación por categoría de GRAMMAR_CATEGORIES.
  3. Por señante: top-1, top-3, F1 macro, señas con recall 0 y confusiones propias; variables medidas
     que distinguen al señante (longitud de clip, fotogramas sin mano, confianzas, movimiento de las
     muñecas), sin atribuir causas.
  4. Por grupo (interrogativos, emociones, valoración, resto; N-30) y por categoría gramatical.
  5. Calibración: diagrama de fiabilidad y ECE (B intervalos iguales) y efecto del umbral D-09 por
     señante (rechazo y error entre aceptadas; aceptada si confianza >= t, como el prototipo).
  6. Relación error-longitud del clip (fotogramas antes de remuestrear a 30): diferencia errores − aciertos
     DENTRO de estrato (señante; seña; seña y señante) con permutación dentro del estrato, correlación entre
     señas y corrección de Holm sobre la familia de pruebas estratificadas de §3 y §6.
  7. Contraste por seña con manos_postura (N-23) y con manos_postura_rostro (N-30): diferencia de recall
     y prueba exacta de McNemar sobre las mismas muestras.

Convenciones:
  · Precisión de una seña que nunca se predijo: indefinida (None) en las tablas; para el F1 macro se
    toma 0, como classification_report de sklearn con el que se generaron los metricas.json.
  · Desviación estándar entre pliegues con ddof = 1 (la misma de resultados_loso.json).
  · ECE = Σ_b (n_b / N) · |exactitud_b − confianza media_b|, intervalo b = mín(⌊conf·B⌋, B − 1).
  · Pruebas a nivel de muestra: D = Σ_g w_g·(media errores − media aciertos)_g / Σ_g w_g, w_g = e_g·a_g/n_g
    (coeficiente con efectos fijos del estrato); nula por permutación de acierto/error dentro de cada estrato.
  · Confusión «simétrica»: ambos sentidos con ≥ 2 casos y menor/mayor ≥ 0,5 (convención descriptiva).

Uso:
    python src/analisis_errores.py
    python src/analisis_errores.py --loso-dir models/loso_2026-09-15_holistic_mascara_manos \\
        --salida models/analisis_errores_manos_2026-09-15
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.grammar_mapper import GRAMMAR_CATEGORIES, SIGN_VOCABULARY  # noqa: E402

N_CLASES = 50
LOSO_DIR = BASE_DIR / "models" / "loso_2026-09-15_holistic_mascara_manos"
SALIDA = BASE_DIR / "models" / "analisis_errores_manos_2026-09-15"
CONTRASTES = {
    "manos_postura": BASE_DIR / "models" / "loso_2026-09-14_holistic_mascara",
    "manos_postura_rostro": BASE_DIR / "models" / "loso_2026-09-15_holistic_mascara_rostro",
}
PROCESSED = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
METRICAS_FOTOGRAMA = BASE_DIR / "data" / "extracted" / "LSC50_holistic" / "METRICAS_FOTOGRAMA"
UMBRAL_JSON = BASE_DIR / "models" / "canonico_holistic_mascara_manos" / "modelo_canonico.umbral.json"

# Grupos de N-30 (recall agregado por grupo de señas)
GRUPOS_N30 = {
    "interrogativos": [41, 42, 43, 44, 45, 46, 26],
    "emociones": [20, 21, 22, 23],
    "valoracion": [24, 25, 27],
}
# Cifras registradas con las que se contrasta (controles): N-30 (recall por grupo, 3 decimales) y N-33
REGISTRO_N30 = {
    "manos": [0.379, 0.688, 0.283, 0.632],
    "manos_postura": [0.264, 0.562, 0.350, 0.598],
    "manos_postura_rostro": [0.357, 0.588, 0.167, 0.558],
}
REGISTRO_N33_T020 = {"aceptadas_correctas": 540, "aceptadas_erroneas": 297}
ORDEN_CATEGORIAS = ["saludo", "cortesia", "pregunta", "sujeto", "verbo", "objeto", "adjetivo", "adverbio"]
TOL = 1e-9


# ---------------------------------------------------------------------------------------------------
# Utilidades puras (probadas en tests/test_analisis_errores.py)
# ---------------------------------------------------------------------------------------------------

def parsear_muestra(nombre: str) -> tuple[int, str, int]:
    """'SSSS_VVVV_RRRR' → (seña, señante como texto de 4 dígitos, repetición)."""
    partes = nombre.split("_")
    if len(partes) != 3 or not all(p.isdigit() and len(p) == 4 for p in partes):
        raise ValueError(f"nombre de muestra no reconocido: {nombre!r}")
    return int(partes[0]), partes[1], int(partes[2])


def categoria_de(sena: int) -> str:
    return GRAMMAR_CATEGORIES[SIGN_VOCABULARY[sena]]["cat"]


def grupo_n30(sena: int) -> str:
    for g, ids in GRUPOS_N30.items():
        if sena in ids:
            return g
    return "resto"


def matriz_confusion(reales, predichas, n: int = N_CLASES) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.int64)
    for r, p in zip(reales, predichas):
        m[int(r), int(p)] += 1
    return m


def metricas_por_clase(reales, predichas, n: int = N_CLASES) -> dict:
    """Recall, precisión y F1 por clase. Precisión None si la clase nunca se predijo; recall None si no
    tiene soporte. F1 = 0 si no hay aciertos (convención de sklearn con zero_division = 0)."""
    m = matriz_confusion(reales, predichas, n)
    tp = np.diag(m).astype(float)
    soporte = m.sum(axis=1)
    n_pred = m.sum(axis=0)
    recall, precision, f1 = [], [], []
    for c in range(n):
        rc = tp[c] / soporte[c] if soporte[c] > 0 else None
        pr = tp[c] / n_pred[c] if n_pred[c] > 0 else None
        if rc is None or pr is None or (rc + pr) == 0:
            f = 0.0
        else:
            f = 2 * pr * rc / (pr + rc)
        recall.append(rc)
        precision.append(pr)
        f1.append(f)
    return {"recall": recall, "precision": precision, "f1": f1,
            "soporte": soporte.tolist(), "n_predichas": n_pred.tolist(), "aciertos": tp.astype(int).tolist()}


def f1_macro(reales, predichas, n: int = N_CLASES) -> float:
    """Media del F1 sobre las n clases (precisión indefinida = 0), como classification_report."""
    return float(np.mean(metricas_por_clase(reales, predichas, n)["f1"]))


def top_k(reales, top3s, k: int) -> float:
    return float(np.mean([int(r) in [int(x) for x in t[:k]] for r, t in zip(reales, top3s)]))


SIMETRIA_MIN_CONTEO = 2
SIMETRIA_MIN_COCIENTE = 0.5
CRITERIO_SIMETRIA = (f"simétrica si los dos sentidos tienen al menos {SIMETRIA_MIN_CONTEO} casos y el cociente "
                     f"menor / mayor es al menos {SIMETRIA_MIN_COCIENTE:.2f}".replace(".", ",")
                     + " (convención descriptiva de este análisis)")


def es_simetrica(conteo: int, inverso: int) -> bool:
    """Criterio descriptivo: ambos sentidos con ≥ SIMETRIA_MIN_CONTEO casos y menor/mayor ≥ SIMETRIA_MIN_COCIENTE.
    Así 6/5 es simétrica y 8/1, 6/1 o 2/5 no."""
    if min(conteo, inverso) < SIMETRIA_MIN_CONTEO:
        return False
    return min(conteo, inverso) / max(conteo, inverso) >= SIMETRIA_MIN_COCIENTE


def pares_confusion(reales, predichas) -> list[dict]:
    """Pares real → predicha con real != predicha, ordenados por conteo descendente (empate: real,
    predicha). Cada par lleva el conteo inverso (predicha → real), si hay confusión en ambos sentidos
    (inverso > 0), el cociente menor/mayor de los dos sentidos y si la confusión es simétrica según
    `es_simetrica`."""
    cont: dict[tuple[int, int], int] = {}
    for r, p in zip(reales, predichas):
        r, p = int(r), int(p)
        if r != p:
            cont[(r, p)] = cont.get((r, p), 0) + 1
    salida = []
    for (r, p), c in cont.items():
        inv = cont.get((p, r), 0)
        salida.append({"real": r, "predicha": p, "conteo": c, "conteo_inverso": inv, "en_ambos_sentidos": inv > 0,
                       "cociente_menor_mayor": min(c, inv) / max(c, inv), "simetrica": es_simetrica(c, inv)})
    salida.sort(key=lambda d: (-d["conteo"], d["real"], d["predicha"]))
    return salida


def media_de(valores) -> tuple[float | None, float | None, int]:
    """Media y desviación estándar muestral (ddof = 1) ignorando None; y cuántos valores hubo."""
    v = [float(x) for x in valores if x is not None]
    if not v:
        return None, None, 0
    de = float(np.std(v, ddof=1)) if len(v) > 1 else None
    return float(np.mean(v)), de, len(v)


def fiabilidad(conf, ok, n_bins: int = 10) -> dict:
    """Diagrama de fiabilidad y ECE con n_bins intervalos iguales en [0, 1]."""
    conf = np.asarray(conf, dtype=float)
    ok = np.asarray(ok, dtype=bool)
    idx = np.minimum(np.floor(conf * n_bins).astype(int), n_bins - 1)
    idx = np.maximum(idx, 0)
    n = len(conf)
    bins, ece, mce = [], 0.0, 0.0
    for b in range(n_bins):
        sel = idx == b
        nb = int(sel.sum())
        fila = {"desde": b / n_bins, "hasta": (b + 1) / n_bins, "n": nb,
                "exactitud": None, "confianza_media": None, "brecha": None}
        if nb:
            acc = float(ok[sel].mean())
            cm = float(conf[sel].mean())
            fila.update(exactitud=acc, confianza_media=cm, brecha=cm - acc)
            ece += nb / n * abs(acc - cm)
            mce = max(mce, abs(acc - cm))
        bins.append(fila)
    return {"n": n, "n_bins": n_bins, "ece": float(ece) if n else None, "mce": float(mce) if n else None,
            "confianza_media": float(conf.mean()) if n else None,
            "exactitud": float(ok.mean()) if n else None, "intervalos": bins}


def efecto_umbral(conf, ok, t: float) -> dict:
    """Aceptada si conf >= t. Rechazo, exactitud y error entre aceptadas, aciertos perdidos por rechazo."""
    conf = np.asarray(conf, dtype=float)
    ok = np.asarray(ok, dtype=bool)
    acept = conf >= t
    n, na = len(conf), int(acept.sum())
    ok_a, mal_a = int((ok & acept).sum()), int((~ok & acept).sum())
    aciertos = int(ok.sum())
    return {
        "umbral": float(t), "n": n, "aceptadas": na, "rechazadas": n - na,
        "rechazo": (n - na) / n if n else None,
        "aceptadas_correctas": ok_a, "aceptadas_erroneas": mal_a,
        "exactitud_aceptadas": ok_a / na if na else None,
        "error_aceptadas": mal_a / na if na else None,
        "aciertos_perdidos": (aciertos - ok_a) / aciertos if aciertos else None,
        "errores_rechazados": (int((~ok).sum()) - mal_a) / int((~ok).sum()) if int((~ok).sum()) else None,
    }


def mcnemar_exacto(b: int, c: int) -> float:
    """p bilateral exacto de McNemar: binomial(b + c, 1/2) sobre los pares discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cola = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * cola))


def diferencia_dentro_de_estratos(valores, ok, estratos) -> dict:
    """Diferencia errores − aciertos de `valores` DENTRO de cada estrato, combinada entre estratos.

    Para el estrato g (n_g muestras, e_g errores, a_g = n_g − e_g aciertos) d_g = media(errores) − media(aciertos).
    Estimador: D = Σ_g w_g·d_g / Σ_g w_g con w_g = e_g·a_g / n_g. Es igual al coeficiente de «error» en una
    regresión de la variable sobre error + efectos fijos del estrato, y equivale a
    D = Σ_{i error} (v_i − media del estrato de i) / Σ_g w_g. Los estratos sin errores o sin aciertos tienen
    w_g = 0 y no aportan. Si ningún estrato tiene ambos, D es None (no hay comparación dentro de estrato)."""
    valores = np.asarray(valores, dtype=float)
    ok = np.asarray(ok, dtype=bool)
    estratos = np.asarray(estratos)
    codigos, inv = np.unique(estratos, return_inverse=True)
    n_g = np.bincount(inv, minlength=len(codigos)).astype(float)
    e_g = np.bincount(inv, weights=(~ok).astype(float), minlength=len(codigos))
    media_g = np.bincount(inv, weights=valores, minlength=len(codigos)) / n_g
    w_g = e_g * (n_g - e_g) / n_g
    W = float(w_g.sum())
    centrados = valores - media_g[inv]
    por_estrato = {}
    for j, cod in enumerate(codigos):
        sel = inv == j
        err, aci = sel & ~ok, sel & ok
        por_estrato[str(cod)] = {
            "n": int(n_g[j]), "errores": int(e_g[j]), "peso": float(w_g[j]),
            "diferencia": float(valores[err].mean() - valores[aci].mean()) if err.any() and aci.any() else None}
    return {
        "diferencia": float(centrados[~ok].sum() / W) if W > 0 else None,
        "suma_pesos": W, "centrados": centrados, "inv": inv,
        "estratos": len(codigos), "estratos_con_errores_y_aciertos": int((w_g > 0).sum()),
        "por_estrato": por_estrato,
        "diferencia_bruta": (float(valores[~ok].mean() - valores[ok].mean())
                             if (~ok).any() and ok.any() else None),
    }


def permutacion_estratificada(valores, ok, estratos, n_perm: int, semilla: int, por_estrato: bool = True) -> dict:
    """Prueba de permutación de la diferencia errores − aciertos DENTRO de estratos.

    Estadístico: `diferencia_dentro_de_estratos` (media ponderada de las diferencias de cada estrato). Nula:
    las etiquetas acierto/error se permutan dentro de cada estrato, con lo que cada d_g tiene media 0 y la
    distribución nula del estadístico queda centrada en 0 aunque los estratos tengan tasas de error y medias
    distintas. p bilateral = (#{|D_perm| ≥ |D_obs|} + 1) / (n_perm + 1). También se informa la diferencia
    bruta (todas las muestras juntas, sin estratos), que NO es la que se prueba."""
    valores = np.asarray(valores, dtype=float)
    ok = np.asarray(ok, dtype=bool)
    d = diferencia_dentro_de_estratos(valores, ok, estratos)
    salida = {
        "estadistico": "diferencia errores − aciertos dentro de estrato: Σ w_g·d_g / Σ w_g, w_g = errores·aciertos/n "
                       "(= coeficiente con efectos fijos del estrato); nula por permutación dentro de cada estrato",
        "diferencia_dentro_de_estratos": d["diferencia"],
        "diferencia_bruta_sin_estratos": d["diferencia_bruta"],
        "estratos": d["estratos"], "estratos_con_errores_y_aciertos": d["estratos_con_errores_y_aciertos"],
        "p_bilateral": None, "media_distribucion_nula": None,
        "permutaciones": n_perm, "semilla": semilla,
    }
    if por_estrato:
        salida["por_estrato"] = d["por_estrato"]
    if d["diferencia"] is None or n_perm <= 0:
        return salida
    obs, W = d["diferencia"], d["suma_pesos"]
    orden = np.argsort(d["inv"], kind="stable")
    inv_o = d["inv"][orden].astype(float)
    err_o = (~ok)[orden]
    c_o = d["centrados"][orden]
    rng = np.random.default_rng(semilla)
    tol = 1e-9 * max(1.0, abs(obs))
    extremos, suma = 0, 0.0
    for inicio in range(0, n_perm, 1000):
        m = min(1000, n_perm - inicio)
        # clave = código de estrato + U(0, 1): ordenar deja cada bloque de estrato en su sitio y baraja dentro
        idx = np.argsort(inv_o[None, :] + rng.random((m, len(valores))), axis=1)
        T = c_o[idx][:, err_o].sum(axis=1) / W
        extremos += int((np.abs(T) >= abs(obs) - tol).sum())
        suma += float(T.sum())
    salida["p_bilateral"] = (extremos + 1) / (n_perm + 1)
    salida["media_distribucion_nula"] = suma / n_perm
    return salida


def holm(pvalores: dict[str, float | None]) -> dict[str, float | None]:
    """p ajustados por Holm-Bonferroni (FWER) dentro de una familia; los None quedan fuera de la familia."""
    validos = sorted(((p, k) for k, p in pvalores.items() if p is not None), key=lambda t: (t[0], t[1]))
    m = len(validos)
    salida: dict[str, float | None] = {k: None for k in pvalores}
    acumulado = 0.0
    for i, (p, k) in enumerate(validos):
        acumulado = max(acumulado, min(1.0, (m - i) * p))
        salida[k] = acumulado
    return salida


def correlacion_permutacion(x, y, n_perm: int, semilla: int) -> dict:
    """Correlación de Pearson y p bilateral por permutación de y (#{|r_perm| ≥ |r_obs|} + 1) / (n_perm + 1)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    den = float(np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
    if den == 0:
        return {"pearson": None, "p_bilateral": None, "n": len(x), "permutaciones": n_perm, "semilla": semilla}
    r = float((xc * yc).sum() / den)
    rng = np.random.default_rng(semilla)
    extremos = 0
    for inicio in range(0, n_perm, 1000):
        m = min(1000, n_perm - inicio)
        idx = np.argsort(rng.random((m, len(y))), axis=1)
        rp = (yc[idx] * xc[None, :]).sum(axis=1) / den
        extremos += int((np.abs(rp) >= abs(r) - 1e-12).sum())
    return {"pearson": r, "p_bilateral": (extremos + 1) / (n_perm + 1) if n_perm > 0 else None, "n": len(x),
            "permutaciones": n_perm, "semilla": semilla}


def tabla_por_cuantiles(valores, ok, n_cuantiles: int = 5) -> list[dict]:
    """Exactitud por cuantil de `valores` (cortes por np.quantile; cada valor va al primer intervalo
    cuyo borde superior lo contiene)."""
    valores = np.asarray(valores, dtype=float)
    ok = np.asarray(ok, dtype=bool)
    bordes = np.quantile(valores, np.linspace(0, 1, n_cuantiles + 1))
    idx = np.clip(np.searchsorted(bordes[1:-1], valores, side="left"), 0, n_cuantiles - 1)
    filas = []
    for q in range(n_cuantiles):
        sel = idx == q
        filas.append({"cuantil": q + 1, "desde": float(bordes[q]), "hasta": float(bordes[q + 1]),
                      "n": int(sel.sum()), "exactitud": float(ok[sel].mean()) if sel.any() else None,
                      "media": float(valores[sel].mean()) if sel.any() else None})
    return filas


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(p)


def nombre_sena(i: int) -> str:
    return SIGN_VOCABULARY[i]


# ---------------------------------------------------------------------------------------------------
# Carga y controles
# ---------------------------------------------------------------------------------------------------

def cargar_loso(loso_dir: Path) -> tuple[list[dict], dict]:
    """Lee las predicciones de los 5 pliegues y añade pliegue, señante, repetición. Comprueba que la seña
    del nombre coincide con «real» y que el señante del nombre es el que el pliegue deja fuera."""
    resultados = json.loads((loso_dir / "resultados_loso.json").read_text(encoding="utf-8"))
    registros = []
    for info in resultados["pliegues"]:
        k = int(info["pliegue"])
        preds = json.loads((loso_dir / f"pliegue_{k}" / "predicciones.json").read_text(encoding="utf-8"))
        for r in preds:
            sena, senante, rep = parsear_muestra(r["muestra"])
            if sena != int(r["real"]):
                raise ValueError(f"{r['muestra']}: la seña del nombre no coincide con real={r['real']}")
            if senante != info["senante_test"]:
                raise ValueError(f"{r['muestra']} está en pliegue_{k}, que deja fuera a {info['senante_test']}")
            registros.append({**r, "pliegue": k, "senante": senante, "repeticion": rep})
    return registros, resultados


def controles_loso(loso_dir: Path, registros: list[dict], resultados: dict) -> dict:
    """Reproduce lo guardado. Levanta RuntimeError si el top-1 (agregado o por pliegue) no coincide."""
    ctl = {"pliegues": [], "todo_coincide": True}
    for info in resultados["pliegues"]:
        k = int(info["pliegue"])
        rs = [r for r in registros if r["pliegue"] == k]
        reales = [r["real"] for r in rs]
        preds = [r["predicha"] for r in rs]
        t1 = float(np.mean([a == b for a, b in zip(reales, preds)]))
        t3 = top_k(reales, [r["top3"] for r in rs], 3)
        f1 = f1_macro(reales, preds)
        met = json.loads((loso_dir / f"pliegue_{k}" / "metricas.json").read_text(encoding="utf-8"))
        mc = json.loads((loso_dir / f"pliegue_{k}" / "matriz_confusion.json").read_text(encoding="utf-8"))
        cr = json.loads((loso_dir / f"pliegue_{k}" / "classification_report.json").read_text(encoding="utf-8"))
        pc = metricas_por_clase(reales, preds)
        recall_cr = [cr[n]["recall"] for n in mc["nombres"]]
        recall_ok = all(abs((a if a is not None else 0.0) - b) < TOL for a, b in zip(pc["recall"], recall_cr))
        fila = {
            "pliegue": k, "senante": info["senante_test"], "n": len(rs),
            "top1_calculado": t1, "top1_guardado": info["top1"], "top1_coincide": abs(t1 - info["top1"]) < TOL,
            "top3_calculado": t3, "top3_guardado": info["top3"], "top3_coincide": abs(t3 - info["top3"]) < TOL,
            "f1_macro_calculado": f1, "f1_macro_guardado": met["f1_macro"],
            "f1_macro_coincide": abs(f1 - met["f1_macro"]) < TOL,
            "n_errores_coincide": sum(a != b for a, b in zip(reales, preds)) == info["n_errores"],
            "matriz_coincide": np.array_equal(matriz_confusion(reales, preds), np.array(mc["matriz"])),
            "recall_por_clase_coincide": recall_ok,
        }
        fila["todo_coincide"] = all(v for kk, v in fila.items() if kk.endswith("coincide"))
        ctl["pliegues"].append(fila)
        ctl["todo_coincide"] &= fila["todo_coincide"]
    valores = [f["top1_calculado"] for f in ctl["pliegues"]]
    ag = resultados["agregado"]["top1"]
    ctl["top1_media_calculada"] = float(np.mean(valores))
    ctl["top1_de_calculada"] = float(np.std(valores, ddof=1))
    ctl["top1_media_guardada"] = ag["media"]
    ctl["top1_de_guardada"] = ag["desviacion_estandar"]
    ctl["top1_agregado_coincide"] = (abs(ctl["top1_media_calculada"] - ag["media"]) < TOL
                                     and abs(ctl["top1_de_calculada"] - ag["desviacion_estandar"]) < TOL)
    ctl["todo_coincide"] &= ctl["top1_agregado_coincide"]
    if not ctl["top1_agregado_coincide"] or not all(f["top1_coincide"] for f in ctl["pliegues"]):
        raise RuntimeError(f"las predicciones de {loso_dir} no reproducen el top-1 de resultados_loso.json: {ctl}")
    return ctl


# ---------------------------------------------------------------------------------------------------
# Secciones del análisis
# ---------------------------------------------------------------------------------------------------

def _resumen_soporte(soporte: list[int]) -> dict:
    """Soporte más frecuente y señas que se apartan de él."""
    moda = int(np.bincount(soporte).argmax())
    return {"mas_frecuente": moda, "otras": {str(i): int(v) for i, v in enumerate(soporte) if v != moda}}


def seccion_por_sena(registros: list[dict]) -> dict:
    reales = [r["real"] for r in registros]
    preds = [r["predicha"] for r in registros]
    ag = metricas_por_clase(reales, preds)
    pliegues = sorted({r["pliegue"] for r in registros})
    por_pl = {k: metricas_por_clase([r["real"] for r in registros if r["pliegue"] == k],
                                    [r["predicha"] for r in registros if r["pliegue"] == k]) for k in pliegues}
    filas = []
    for c in range(N_CLASES):
        rec_pl = [por_pl[k]["recall"][c] for k in pliegues]
        pre_pl = [por_pl[k]["precision"][c] for k in pliegues]
        f1_pl = [por_pl[k]["f1"][c] for k in pliegues]
        mr, dr, _ = media_de(rec_pl)
        mp, dp, np_def = media_de(pre_pl)
        mf, df, _ = media_de(f1_pl)
        filas.append({
            "id": c, "nombre": nombre_sena(c), "categoria": categoria_de(c), "grupo_n30": grupo_n30(c),
            "soporte": ag["soporte"][c], "aciertos": ag["aciertos"][c], "n_predichas": ag["n_predichas"][c],
            "recall": ag["recall"][c], "precision": ag["precision"][c], "f1": ag["f1"][c],
            "por_pliegue": {"recall": rec_pl, "precision": pre_pl, "f1": f1_pl},
            "recall_media_pliegues": mr, "recall_de_pliegues": dr,
            "precision_media_pliegues": mp, "precision_de_pliegues": dp, "precision_pliegues_definidos": np_def,
            "f1_media_pliegues": mf, "f1_de_pliegues": df,
        })
    orden = sorted(filas, key=lambda f: (-1.0 if f["recall"] is None else f["recall"], f["f1"], f["id"]))
    return {
        "criterio_ranking": "recall agregado ascendente; empate: F1 agregado ascendente, luego id",
        "senas": filas,
        "peores_10": [f["id"] for f in orden[:10]],
        "mejores_10": [f["id"] for f in orden[::-1][:10]],
        "senas_con_recall_0": [f["id"] for f in filas if f["recall"] == 0],
        "senas_nunca_predichas": [f["id"] for f in filas if f["n_predichas"] == 0],
        "f1_macro_agregado": float(np.mean(ag["f1"])),
        "recall_macro_agregado": float(np.mean([x for x in ag["recall"] if x is not None])),
        "soporte_por_sena": _resumen_soporte(ag["soporte"]),
        "soporte_por_pliegue_mas_frecuente": int(np.bincount(
            [x for k in pliegues for x in por_pl[k]["soporte"]]).argmax()),
        "soporte_por_pliegue_min": int(min(x for k in pliegues for x in por_pl[k]["soporte"])),
        "soporte_por_pliegue_max": int(max(x for k in pliegues for x in por_pl[k]["soporte"])),
        "soporte_por_pliegue_otras": {str(k): _resumen_soporte(por_pl[k]["soporte"])["otras"] for k in pliegues
                                      if _resumen_soporte(por_pl[k]["soporte"])["otras"]},
    }


def seccion_confusiones(registros: list[dict]) -> dict:
    reales = [r["real"] for r in registros]
    preds = [r["predicha"] for r in registros]
    pares = pares_confusion(reales, preds)
    for p in pares:
        p.update(nombre_real=nombre_sena(p["real"]), nombre_predicha=nombre_sena(p["predicha"]),
                 categoria_real=categoria_de(p["real"]), categoria_predicha=categoria_de(p["predicha"]),
                 misma_categoria=categoria_de(p["real"]) == categoria_de(p["predicha"]),
                 senantes=sorted({r["senante"] for r in registros
                                  if r["real"] == p["real"] and r["predicha"] == p["predicha"]}))
    n_err = sum(p["conteo"] for p in pares)
    # parejas no ordenadas {a, b}: total en ambos sentidos
    no_ord: dict[tuple[int, int], dict] = {}
    for p in pares:
        a, b = sorted((p["real"], p["predicha"]))
        d = no_ord.setdefault((a, b), {"a": a, "b": b, "a_a_b": 0, "b_a_a": 0})
        if p["real"] == a:
            d["a_a_b"] += p["conteo"]
        else:
            d["b_a_a"] += p["conteo"]
    parejas = sorted(({**d, "total": d["a_a_b"] + d["b_a_a"], "nombre_a": nombre_sena(d["a"]),
                       "nombre_b": nombre_sena(d["b"]), "en_ambos_sentidos": d["a_a_b"] > 0 and d["b_a_a"] > 0,
                       "simetrica": es_simetrica(d["a_a_b"], d["b_a_a"])}
                      for d in no_ord.values()), key=lambda d: (-d["total"], d["a"], d["b"]))
    # por categoría
    cats = ORDEN_CATEGORIAS
    mcat = {cr: {cp: 0 for cp in cats} for cr in cats}
    for p in pares:
        mcat[p["categoria_real"]][p["categoria_predicha"]] += p["conteo"]
    dentro = sum(mcat[c][c] for c in cats)
    # referencia: si cada error de la seña s fuera a una de las otras 49 al azar, P(misma categoría) =
    # (n_cat(s) − 1) / 49; se pondera por los errores de cada seña
    tam = {c: sum(1 for i in range(N_CLASES) if categoria_de(i) == c) for c in cats}
    err_por_sena = {c: 0 for c in range(N_CLASES)}
    for p in pares:
        err_por_sena[p["real"]] += p["conteo"]
    esperado = sum(e * (tam[categoria_de(s)] - 1) / (N_CLASES - 1) for s, e in err_por_sena.items())
    por_cat = {}
    for c in cats:
        err_c = sum(mcat[c].values())
        pares_c = [p for p in pares if p["categoria_real"] == c][:5]
        por_cat[c] = {"n_senas": tam[c], "errores": err_c, "dentro_de_la_categoria": mcat[c][c],
                      "proporcion_dentro": mcat[c][c] / err_c if err_c else None,
                      "pares_principales": [(p["real"], p["predicha"], p["conteo"]) for p in pares_c]}
    return {
        "n_errores": n_err, "n_pares_distintos": len(pares),
        "pares": pares,
        "parejas_no_ordenadas": parejas,
        "n_pares_con_conteo_ge_3": sum(1 for p in pares if p["conteo"] >= 3),
        "criterio_simetria": CRITERIO_SIMETRIA,
        "proporcion_errores_en_pares_con_ambos_sentidos": (sum(p["conteo"] for p in pares if p["en_ambos_sentidos"]) / n_err
                                                           if n_err else None),
        "proporcion_errores_en_pares_simetricos": sum(p["conteo"] for p in pares if p["simetrica"]) / n_err if n_err else None,
        "n_parejas_simetricas": sum(1 for p in parejas if p["simetrica"]),
        "por_categoria": {
            "matriz_errores_categoria_real_a_predicha": mcat,
            "errores_dentro_de_la_misma_categoria": dentro,
            "proporcion_dentro": dentro / n_err if n_err else None,
            "proporcion_dentro_esperada_si_errores_al_azar": esperado / n_err if n_err else None,
            "detalle": por_cat,
        },
        "predichas_mas_frecuentes_en_errores": sorted(
            ({"id": c, "nombre": nombre_sena(c), "veces_predicha_por_error": sum(p["conteo"] for p in pares if p["predicha"] == c)}
             for c in range(N_CLASES)), key=lambda d: (-d["veces_predicha_por_error"], d["id"]))[:10],
    }


def seccion_por_senante(registros: list[dict]) -> dict:
    salida = {}
    for s in sorted({r["senante"] for r in registros}):
        rs = [r for r in registros if r["senante"] == s]
        reales = [r["real"] for r in rs]
        preds = [r["predicha"] for r in rs]
        pc = metricas_por_clase(reales, preds)
        ok = np.array([a == b for a, b in zip(reales, preds)])
        conf = np.array([r["prob_predicha"] for r in rs])
        pares = pares_confusion(reales, preds)
        errores = [r for r in rs if r["real"] != r["predicha"]]
        salida[s] = {
            "pliegue": rs[0]["pliegue"], "n": len(rs),
            "top1": float(ok.mean()), "top3": top_k(reales, [r["top3"] for r in rs], 3),
            "f1_macro": float(np.mean(pc["f1"])),
            "n_errores": int((~ok).sum()),
            "senas_con_recall_0": [c for c in range(N_CLASES) if pc["recall"][c] == 0],
            "senas_con_recall_le_0_25": [c for c in range(N_CLASES) if pc["recall"][c] is not None and pc["recall"][c] <= 0.25],
            "senas_nunca_predichas": [c for c in range(N_CLASES) if pc["n_predichas"][c] == 0],
            "predicha_mas_frecuente": max(range(N_CLASES), key=lambda c: (pc["n_predichas"][c], -c)),
            "n_predichas_de_la_mas_frecuente": max(pc["n_predichas"]),
            "pares_principales": [(p["real"], p["predicha"], p["conteo"]) for p in pares[:8]],
            "confianza_media": float(conf.mean()),
            "confianza_mediana": float(np.median(conf)),
            "confianza_media_aciertos": float(conf[ok].mean()) if ok.any() else None,
            "confianza_media_errores": float(conf[~ok].mean()) if (~ok).any() else None,
            "errores_con_confianza_ge_0_5": int(((~ok) & (conf >= 0.5)).sum()),
            "errores_con_real_en_top3": sum(1 for r in errores if r["real"] in r["top3"]),
            "recall_por_sena": pc["recall"],
        }
    return salida


def _longitudes(processed: Path) -> dict[str, int] | None:
    meta_p, len_p = processed / "metadata.json", processed / "lengths.npy"
    if not (meta_p.exists() and len_p.exists()):
        return None
    orden = json.loads(meta_p.read_text(encoding="utf-8"))["sample_order"]
    lengths = np.load(len_p)
    if len(orden) != len(lengths):
        raise ValueError("metadata.json sample_order y lengths.npy no tienen el mismo largo")
    return {m: int(l) for m, l in zip(orden, lengths)}


def _deteccion_por_muestra(dir_metricas: Path, muestras: list[str]) -> dict[str, dict] | None:
    if not dir_metricas.exists():
        return None
    salida = {}
    for m in muestras:
        ruta = dir_metricas / f"{m}.csv"
        if not ruta.exists():
            return None
        with open(ruta, newline="", encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        izq = np.array([int(x["mano_izq_detectada"]) for x in filas])
        der = np.array([int(x["mano_der_detectada"]) for x in filas])
        salida[m] = {"fotogramas": len(filas), "sin_izq": int((izq == 0).sum()), "sin_der": int((der == 0).sum()),
                     "sin_alguna": int(((izq == 0) | (der == 0)).sum()), "sin_ambas": int(((izq == 0) & (der == 0)).sum())}
    return salida


def _movimiento_munecas(processed: Path, muestras: list[str], longitudes: dict[str, int]) -> dict[str, dict] | None:
    """Desplazamiento medio por fotograma (xy, espacio normalizado por hombros) de la muñeca de cada
    bloque de mano, contando solo pares de fotogramas consecutivos con la mano presente; y el conteo de
    bloques de mano en ceros (control frente a METRICAS_FOTOGRAMA)."""
    meta_p, x_p = processed / "metadata.json", processed / "X_padded.npy"
    if not (meta_p.exists() and x_p.exists()):
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    orden = {m: i for i, m in enumerate(meta["sample_order"])}
    g = meta["landmark_groups"]
    X = np.load(x_p, mmap_mode="r")
    bloques = {"izq": g["left_hand"]["start_col"], "der": g["right_hand"]["start_col"]}
    salida = {}
    for m in muestras:
        n = longitudes[m]
        fila = X[orden[m], :n, :]
        d = {}
        for lado, c0 in bloques.items():
            bloque = np.asarray(fila[:, c0:c0 + 63])
            presente = np.any(bloque != 0, axis=1)
            xy = bloque[:, 0:2]
            ambos = presente[1:] & presente[:-1]
            pasos = np.linalg.norm(np.diff(xy, axis=0), axis=1)[ambos]
            d[f"bloques_en_ceros_{lado}"] = int((~presente).sum())
            d[f"recorrido_{lado}"] = float(pasos.sum())
            d[f"pasos_validos_{lado}"] = int(ambos.sum())
        salida[m] = d
    return salida


VARIABLES_EVIDENCIA = [
    # (clave, etiqueta, formato: "n3" | "n1" | "pct")
    ("top1", "top-1", "n3"),
    ("longitud_media", "longitud media (fotogramas)", "n1"),
    ("longitud_mediana", "longitud mediana (fotogramas)", "n1"),
    ("longitud_de", "DE de la longitud", "n1"),
    ("cociente_longitud_vs_otros_misma_sena_media", "longitud / media de los otros 4 en la misma seña", "n3"),
    ("proporcion_fotogramas_sin_mano_izq", "fotogramas sin mano izq.", "pct"),
    ("proporcion_fotogramas_sin_mano_der", "fotogramas sin mano der.", "pct"),
    ("proporcion_fotogramas_sin_alguna_mano", "fotogramas sin alguna mano", "pct"),
    ("proporcion_fotogramas_sin_ambas_manos", "fotogramas sin ambas manos", "pct"),
    ("desplazamiento_medio_muneca_izq", "desplaz. muñeca izq. por fotograma", "n3"),
    ("desplazamiento_medio_muneca_der", "desplaz. muñeca der. por fotograma", "n3"),
    ("proporcion_recorrido_bloque_der", "proporción del recorrido en el bloque der.", "pct"),
    ("confianza_media", "confianza media (salida del modelo)", "n3"),
]


ESTRATIFICACIONES = {
    # clave: (etiqueta, función registro → estrato)
    "senante": ("dentro de señante", lambda r: r["senante"]),
    "sena": ("dentro de seña", lambda r: f"{r['real']:04d}"),
    "sena_y_senante": ("dentro de seña y señante", lambda r: f"{r['real']:04d}_{r['senante']}"),
}
_OFFSET_MEDIDA = {"longitud": 0, "proporcion_sin_alguna_mano": 10, "proporcion_sin_mano_der": 20,
                  "proporcion_sin_mano_izq": 30, "desvio_log_vs_misma_sena": 40}
_OFFSET_ESTRATO = {"senante": 0, "sena": 1, "sena_y_senante": 2, "un_senante": 100}


def _semilla(semilla: int, medida: str, estrato: str) -> int:
    """Semilla fija por (medida, estratificación): la misma prueba da el mismo p en §3 y §6."""
    return semilla + _OFFSET_MEDIDA[medida] + _OFFSET_ESTRATO[estrato]


def pruebas_estratificadas(v, ok, registros: list[dict], medida: str, n_perm: int, semilla: int,
                           estratos: tuple[str, ...] = tuple(ESTRATIFICACIONES)) -> dict:
    """Permutación dentro de estrato para cada estratificación (señante; seña; celda seña × señante)."""
    return {f"estratificada_por_{e}": permutacion_estratificada(
                v, ok, np.array([ESTRATIFICACIONES[e][1](r) for r in registros]), n_perm, _semilla(semilla, medida, e),
                por_estrato=e != "sena_y_senante")   # 250 celdas: el detalle por celda no se guarda
            for e in estratos}


def seccion_evidencia_senantes(registros: list[dict], longitudes, deteccion, movimiento,
                               n_perm: int = 10000, semilla: int = 2026) -> dict:
    """Variables medidas por señante sobre sus muestras de prueba y su relación con el error a nivel de
    muestra (dentro de cada señante y estratificada por señante). No atribuye causas."""
    senantes = sorted({r["senante"] for r in registros})
    por = {}
    for s in senantes:
        rs = [r for r in registros if r["senante"] == s]
        d: dict = {"n": len(rs), "top1": float(np.mean([r["real"] == r["predicha"] for r in rs]))}
        if longitudes:
            L = np.array([longitudes[r["muestra"]] for r in rs])
            d.update(longitud_media=float(L.mean()), longitud_mediana=float(np.median(L)),
                     longitud_min=int(L.min()), longitud_max=int(L.max()), longitud_de=float(L.std(ddof=1)))
            # misma seña, otros señantes: cociente de longitud
            cocientes = []
            for r in rs:
                otros = [longitudes[x["muestra"]] for x in registros if x["real"] == r["real"] and x["senante"] != s]
                cocientes.append(longitudes[r["muestra"]] / float(np.mean(otros)))
            d["cociente_longitud_vs_otros_misma_sena_media"] = float(np.mean(cocientes))
        if deteccion:
            tot = sum(deteccion[r["muestra"]]["fotogramas"] for r in rs)
            d.update(
                fotogramas_totales=tot,
                proporcion_fotogramas_sin_mano_izq=sum(deteccion[r["muestra"]]["sin_izq"] for r in rs) / tot,
                proporcion_fotogramas_sin_mano_der=sum(deteccion[r["muestra"]]["sin_der"] for r in rs) / tot,
                proporcion_fotogramas_sin_alguna_mano=sum(deteccion[r["muestra"]]["sin_alguna"] for r in rs) / tot,
                proporcion_fotogramas_sin_ambas_manos=sum(deteccion[r["muestra"]]["sin_ambas"] for r in rs) / tot,
                muestras_con_algun_fotograma_sin_mano=sum(1 for r in rs if deteccion[r["muestra"]]["sin_alguna"] > 0),
            )
        if movimiento:
            ri = sum(movimiento[r["muestra"]]["recorrido_izq"] for r in rs)
            rd = sum(movimiento[r["muestra"]]["recorrido_der"] for r in rs)
            pi = sum(movimiento[r["muestra"]]["pasos_validos_izq"] for r in rs)
            pd_ = sum(movimiento[r["muestra"]]["pasos_validos_der"] for r in rs)
            d.update(desplazamiento_medio_muneca_izq=ri / pi if pi else None,
                     desplazamiento_medio_muneca_der=rd / pd_ if pd_ else None,
                     proporcion_recorrido_bloque_der=rd / (ri + rd) if (ri + rd) else None)
        conf = np.array([r["prob_predicha"] for r in rs])
        d["confianza_media"] = float(conf.mean())
        por[s] = d
    variables = [k for k, _, _ in VARIABLES_EVIDENCIA if por[senantes[0]].get(k) is not None]
    rangos = {}
    for v in variables:
        orden = sorted(senantes, key=lambda s: por[s][v])
        rangos[v] = {"menor": orden[0], "mayor": orden[-1], "orden_ascendente": orden,
                     "valor_menor": por[orden[0]][v], "valor_segundo_menor": por[orden[1]][v] if len(orden) > 1 else None,
                     "valor_mayor": por[orden[-1]][v], "valor_segundo_mayor": por[orden[-2]][v] if len(orden) > 1 else None}

    # nivel de muestra: ¿los errores tienen más fotogramas sin mano o clips más largos/cortos?
    ok = np.array([r["real"] == r["predicha"] for r in registros])
    estr = np.array([r["senante"] for r in registros])
    muestra = {}
    medidas = {}
    if deteccion:
        medidas["proporcion_sin_alguna_mano"] = np.array(
            [deteccion[r["muestra"]]["sin_alguna"] / deteccion[r["muestra"]]["fotogramas"] for r in registros])
        medidas["proporcion_sin_mano_der"] = np.array(
            [deteccion[r["muestra"]]["sin_der"] / deteccion[r["muestra"]]["fotogramas"] for r in registros])
        medidas["proporcion_sin_mano_izq"] = np.array(
            [deteccion[r["muestra"]]["sin_izq"] / deteccion[r["muestra"]]["fotogramas"] for r in registros])
    if longitudes:
        medidas["longitud"] = np.array([longitudes[r["muestra"]] for r in registros], dtype=float)
    for nombre, v in medidas.items():
        d = pruebas_estratificadas(v, ok, registros, nombre, n_perm, semilla)
        d["por_senante"] = {}
        for i, s in enumerate(senantes):
            m = estr == s
            d["por_senante"][s] = {
                "media_aciertos": float(v[m & ok].mean()) if (m & ok).any() else None,
                "media_errores": float(v[m & ~ok].mean()) if (m & ~ok).any() else None,
                "muestras_con_valor_mayor_que_0": int((v[m] > 0).sum()),
                "exactitud_si_valor_0": float(ok[m & (v == 0)].mean()) if (m & (v == 0)).any() else None,
                "exactitud_si_valor_mayor_que_0": float(ok[m & (v > 0)].mean()) if (m & (v > 0)).any() else None,
                # un solo estrato: la diferencia dentro de estrato es la diferencia de medias del señante
                "permutacion": permutacion_estratificada(v[m], ok[m], estr[m], n_perm,
                                                         _semilla(semilla, nombre, "un_senante") + i, por_estrato=False),
            }
        muestra[nombre] = d
    return {"por_senante": por, "extremos": rangos, "nivel_muestra": muestra,
            "nota": "descriptivo, 5 señantes: no permite estimar relaciones entre las variables por señante y el "
                    "top-1; nivel_muestra compara errores y aciertos dentro de estratos (señante, seña, seña y "
                    "señante) con permutación dentro del estrato; p Holm sobre la familia de pruebas estratificadas"}


def seccion_grupos(registros: list[dict], por_sena: dict) -> dict:
    reales = np.array([r["real"] for r in registros])
    preds = np.array([r["predicha"] for r in registros])
    senantes = np.array([r["senante"] for r in registros])

    def resumen(ids: list[int]) -> dict:
        ids_set = set(ids)
        sel = np.isin(reales, ids)
        ok = preds[sel] == reales[sel]
        en_grupo = np.isin(preds[sel], ids)
        pred_sel = np.isin(preds, ids)
        f1s = [por_sena["senas"][i]["f1"] for i in ids]
        por_s = {}
        for s in sorted(set(senantes)):
            m = sel & (senantes == s)
            por_s[s] = float((preds[m] == reales[m]).mean()) if m.any() else None
        _, de, _ = media_de(list(por_s.values()))
        return {"senas": sorted(ids_set), "n_senas": len(ids_set), "n_muestras": int(sel.sum()),
                "recall": float(ok.mean()) if sel.any() else None,
                "proporcion_predicha_dentro_del_grupo": float(en_grupo.mean()) if sel.any() else None,
                "precision_del_grupo": float((reales[pred_sel] == preds[pred_sel]).mean()) if pred_sel.any() else None,
                "f1_medio_senas": float(np.mean(f1s)),
                "recall_por_senante": por_s, "recall_de_senantes": de}

    grupos = {g: resumen(ids) for g, ids in GRUPOS_N30.items()}
    usados = {i for ids in GRUPOS_N30.values() for i in ids}
    grupos["resto"] = resumen([i for i in range(N_CLASES) if i not in usados])
    cats = {c: resumen([i for i in range(N_CLASES) if categoria_de(i) == c]) for c in ORDEN_CATEGORIAS}
    return {"grupos_n30": grupos, "categorias_gramaticales": cats,
            "nota": "recall = aciertos / muestras del grupo; proporcion_predicha_dentro_del_grupo = la predicción cae "
                    "en alguna seña del grupo (acierto o confusión interna); precision_del_grupo = aciertos entre las "
                    "predicciones que caen en el grupo"}


def seccion_calibracion(registros: list[dict], umbral: float, n_bins: int) -> dict:
    conf = np.array([r["prob_predicha"] for r in registros], dtype=float)
    ok = np.array([r["real"] == r["predicha"] for r in registros])
    senantes = np.array([r["senante"] for r in registros])
    salida = {"global": fiabilidad(conf, ok, n_bins), "por_senante": {},
              "umbral": {"t": umbral, "global": efecto_umbral(conf, ok, umbral), "por_senante": {}},
              "referencia_0_5": {"global": efecto_umbral(conf, ok, 0.5), "por_senante": {}}}
    for s in sorted(set(senantes)):
        m = senantes == s
        salida["por_senante"][s] = fiabilidad(conf[m], ok[m], n_bins)
        salida["umbral"]["por_senante"][s] = efecto_umbral(conf[m], ok[m], umbral)
        salida["referencia_0_5"]["por_senante"][s] = efecto_umbral(conf[m], ok[m], 0.5)
    return salida


def seccion_longitud(registros: list[dict], longitudes: dict[str, int] | None, n_perm: int, semilla: int) -> dict | None:
    if not longitudes:
        return None
    L = np.array([longitudes[r["muestra"]] for r in registros], dtype=float)
    ok = np.array([r["real"] == r["predicha"] for r in registros])
    senantes = np.array([r["senante"] for r in registros])
    # longitud relativa a la mediana del propio señante
    rel = np.empty_like(L)
    for s in np.unique(senantes):
        m = senantes == s
        rel[m] = L[m] / np.median(L[m])
    # desvío respecto de la misma seña en los otros señantes (los del entrenamiento del pliegue)
    desvio = np.empty_like(L)
    for i, r in enumerate(registros):
        otros = [longitudes[x["muestra"]] for x in registros if x["real"] == r["real"] and x["senante"] != r["senante"]]
        desvio[i] = abs(math.log(L[i] / float(np.mean(otros))))
    por_s = {}
    for s in np.unique(senantes):
        m = senantes == s
        la = float(L[m & ok].mean()) if (m & ok).any() else None
        le = float(L[m & ~ok].mean()) if (m & ~ok).any() else None
        por_s[str(s)] = {"n": int(m.sum()), "longitud_media_aciertos": la, "longitud_media_errores": le,
                         "diferencia_errores_menos_aciertos": le - la if la is not None and le is not None else None,
                         "cuantiles_longitud": tabla_por_cuantiles(L[m], ok[m], 3)}
    reales = np.array([r["real"] for r in registros])
    senas = sorted(set(reales.tolist()))
    long_sena = np.array([L[reales == c].mean() for c in senas])
    recall_sena = np.array([ok[reales == c].mean() for c in senas])
    return {
        "definicion": "longitud = fotogramas del clip antes de remuestrear a 30 (lengths.npy)",
        "longitud_media_aciertos": float(L[ok].mean()), "longitud_media_errores": float(L[~ok].mean()),
        "diferencia_bruta_errores_menos_aciertos": float(L[~ok].mean() - L[ok].mean()),
        "cuantiles_longitud_absoluta": tabla_por_cuantiles(L, ok, 5),
        "cuantiles_longitud_relativa_al_senante": tabla_por_cuantiles(rel, ok, 5),
        "cuantiles_desvio_log_vs_misma_sena_otros_senantes": tabla_por_cuantiles(desvio, ok, 5),
        "pruebas_longitud": pruebas_estratificadas(L, ok, registros, "longitud", n_perm, semilla),
        "pruebas_desvio_vs_misma_sena": pruebas_estratificadas(desvio, ok, registros, "desvio_log_vs_misma_sena",
                                                               n_perm, semilla, estratos=("senante", "sena")),
        "por_senante": por_s,
        "correlacion_pearson_longitud_acierto": float(np.corrcoef(L, ok.astype(float))[0, 1]),
        "entre_senas": {
            "descripcion": "una fila por seña: longitud media de sus clips y recall agregado (50 señas)",
            **correlacion_permutacion(long_sena, recall_sena, n_perm, semilla + 7),
        },
    }


def seccion_contraste(registros: list[dict], contrastes: dict[str, Path]) -> dict:
    base = {r["muestra"]: r for r in registros}
    salida = {}
    for nombre, d in contrastes.items():
        if not (d / "resultados_loso.json").exists():
            salida[nombre] = {"disponible": False, "dir": _rel(d)}
            continue
        otros, res = cargar_loso(d)
        otro = {r["muestra"]: r for r in otros}
        mismas = set(otro) == set(base) and all(otro[m]["real"] == base[m]["real"] for m in base)
        if not mismas:
            salida[nombre] = {"disponible": True, "dir": _rel(d), "mismas_muestras": False}
            continue
        muestras = sorted(base)
        ok_b = np.array([base[m]["real"] == base[m]["predicha"] for m in muestras])
        ok_o = np.array([otro[m]["real"] == otro[m]["predicha"] for m in muestras])
        reales = np.array([base[m]["real"] for m in muestras])
        pc_o = metricas_por_clase(reales, [otro[m]["predicha"] for m in muestras])
        pc_b = metricas_por_clase(reales, [base[m]["predicha"] for m in muestras])
        filas = []
        for c in range(N_CLASES):
            sel = reales == c
            b = int((ok_b[sel] & ~ok_o[sel]).sum())   # acierta manos, falla el otro
            cc = int((~ok_b[sel] & ok_o[sel]).sum())  # falla manos, acierta el otro
            filas.append({"id": c, "nombre": nombre_sena(c), "categoria": categoria_de(c),
                          "recall_manos": pc_b["recall"][c], f"recall_{nombre}": pc_o["recall"][c],
                          "diferencia_manos_menos_otro": pc_b["recall"][c] - pc_o["recall"][c],
                          "f1_manos": pc_b["f1"][c], f"f1_{nombre}": pc_o["f1"][c],
                          "solo_acierta_manos": b, "solo_acierta_otro": cc, "p_mcnemar": mcnemar_exacto(b, cc)})
        b_tot = int((ok_b & ~ok_o).sum())
        c_tot = int((~ok_b & ok_o).sum())
        orden = sorted(filas, key=lambda f: (-f["diferencia_manos_menos_otro"], f["id"]))
        grupos = {}
        for g, ids in {**GRUPOS_N30, "resto": [i for i in range(N_CLASES) if all(i not in v for v in GRUPOS_N30.values())]}.items():
            sel = np.isin(reales, ids)
            grupos[g] = {"recall_manos": float(ok_b[sel].mean()), f"recall_{nombre}": float(ok_o[sel].mean())}
        salida[nombre] = {
            "disponible": True, "dir": _rel(d), "mismas_muestras": True, "representacion_declarada": res.get("representacion"),
            "top1_manos": float(ok_b.mean()), f"top1_{nombre}": float(ok_o.mean()),
            "solo_acierta_manos": b_tot, "solo_acierta_otro": c_tot, "p_mcnemar_global": mcnemar_exacto(b_tot, c_tot),
            "senas": filas,
            "ganan_con_manos": [f["id"] for f in orden if f["diferencia_manos_menos_otro"] > 0],
            "pierden_con_manos": [f["id"] for f in orden[::-1] if f["diferencia_manos_menos_otro"] < 0],
            "sin_cambio": [f["id"] for f in filas if f["diferencia_manos_menos_otro"] == 0],
            "senas_con_p_mcnemar_lt_0_05": [f["id"] for f in filas if f["p_mcnemar"] < 0.05],
            "recall_por_grupo_n30": grupos,
            "nota": "una prueba por seña (50) sin corrección por comparaciones múltiples; con 19 o 20 muestras por "
                    "seña una muestra mueve el recall 0,05 o 0,053; los p por seña son orientativos",
        }
    return salida


# ---------------------------------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------------------------------

PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TINTA, TINTA_2, REJILLA_COLOR = "#0b0b0b", "#52514e", "#e4e3df"


def _estilo(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "savefig.facecolor": "#fcfcfb"})


def figuras(analisis: dict, registros: list[dict], salida: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _estilo(plt)
    hechas = []
    col_cat = {c: PALETA[i] for i, c in enumerate(ORDEN_CATEGORIAS)}

    # 1. matriz de confusión normalizada por fila, ordenada por categoría
    orden = [i for c in ORDEN_CATEGORIAS for i in range(N_CLASES) if categoria_de(i) == c]
    m = matriz_confusion([r["real"] for r in registros], [r["predicha"] for r in registros]).astype(float)
    mn = m / m.sum(axis=1, keepdims=True)
    mn = mn[np.ix_(orden, orden)]
    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(mn, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    etiquetas = [f"{nombre_sena(i)} ({i})" for i in orden]
    ax.set_xticks(range(N_CLASES), etiquetas, rotation=90, fontsize=7)
    ax.set_yticks(range(N_CLASES), etiquetas, fontsize=7)
    pos = 0
    for c in ORDEN_CATEGORIAS:
        n = sum(1 for i in range(N_CLASES) if categoria_de(i) == c)
        ax.axhline(pos - 0.5, color=TINTA_2, lw=0.6)
        ax.axvline(pos - 0.5, color=TINTA_2, lw=0.6)
        ax.text(N_CLASES + 0.8, pos + n / 2 - 0.5, c, va="center", fontsize=8, color=TINTA)
        pos += n
    ax.set_xlabel("seña predicha")
    ax.set_ylabel("seña real")
    ax.set_title("Matriz de confusión LOSO agregada (997 predicciones), normalizada por fila, ordenada por categoría",
                 fontsize=10, color=TINTA)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.09, label="proporción de la fila")
    for sp in ax.spines.values():
        sp.set_visible(True)
    fig.tight_layout()
    fig.savefig(salida / "matriz_confusion_normalizada.png", dpi=150)
    plt.close(fig)
    hechas.append("matriz_confusion_normalizada.png")

    # 2. recall por seña (agregado) con valores por pliegue
    senas = sorted(analisis["por_sena"]["senas"], key=lambda f: (f["recall"], f["f1"], f["id"]))
    fig, ax = plt.subplots(figsize=(8, 11))
    y = np.arange(len(senas))
    ax.barh(y, [f["recall"] for f in senas], color=[col_cat[f["categoria"]] for f in senas], height=0.72)
    for yi, f in zip(y, senas):
        ax.scatter(f["por_pliegue"]["recall"], [yi] * len(f["por_pliegue"]["recall"]), s=9, color=TINTA,
                   zorder=3, linewidths=0, alpha=0.55)
    ax.set_yticks(y, [f"{f['nombre']} ({f['id']})" for f in senas], fontsize=7)
    ax.set_xlim(0, 1.02)
    s1 = analisis["por_sena"]
    rango = (f"{s1['soporte_por_pliegue_min']}" if s1["soporte_por_pliegue_min"] == s1["soporte_por_pliegue_max"]
             else f"{s1['soporte_por_pliegue_min']} o {s1['soporte_por_pliegue_max']}")
    ax.set_xlabel(f"recall (barra: agregado de 5 pliegues; puntos: cada pliegue, {rango} muestras)")
    ax.grid(axis="x", color=REJILLA_COLOR, lw=0.6)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col_cat[c], label=c) for c in ORDEN_CATEGORIAS], loc="lower right",
              fontsize=7, frameon=False, title="categoría")
    ax.set_title("Recall por seña, LOSO «manos»", fontsize=10, color=TINTA)
    fig.tight_layout()
    fig.savefig(salida / "recall_por_sena.png", dpi=150)
    plt.close(fig)
    hechas.append("recall_por_sena.png")

    # 3. por señante: top-1, top-3, F1 macro
    ps = analisis["por_senante"]
    sen = sorted(ps)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(sen))
    ancho = 0.26
    for j, (clave, etq) in enumerate([("top1", "top-1"), ("top3", "top-3"), ("f1_macro", "F1 macro")]):
        vals = [ps[s][clave] for s in sen]
        barras = ax.bar(x + (j - 1) * ancho, vals, width=ancho - 0.02, color=PALETA[j], label=etq)
        for b, v in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}".replace(".", ","), ha="center",
                    fontsize=7, color=TINTA_2)
    ax.set_xticks(x, [f"señante {s}\n(pliegue {ps[s]['pliegue']})" for s in sen])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=REJILLA_COLOR, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("Prueba LOSO por señante (el pliegue k deja fuera al señante k)", fontsize=10, color=TINTA)
    fig.tight_layout()
    fig.savefig(salida / "por_senante.png", dpi=150)
    plt.close(fig)
    hechas.append("por_senante.png")

    # 4. fiabilidad
    cal = analisis["calibracion"]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    g = cal["global"]
    centros = [(b["desde"] + b["hasta"]) / 2 for b in g["intervalos"]]
    anchob = 1 / g["n_bins"]
    ax = axs[0, 0]
    ax.plot([0, 1], [0, 1], color=TINTA_2, lw=1, ls="--", label="calibración perfecta")
    ax.bar(centros, [b["exactitud"] or 0 for b in g["intervalos"]], width=anchob * 0.92, color=PALETA[0],
           label="exactitud del intervalo")
    ax.scatter([b["confianza_media"] for b in g["intervalos"] if b["n"]],
               [b["exactitud"] for b in g["intervalos"] if b["n"]], color=TINTA, s=12, zorder=3,
               label="(confianza media, exactitud)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("exactitud")
    ax.set_title(f"Agregado (997): ECE {g['ece']:.3f}".replace(".", ","), fontsize=10, color=TINTA)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    axs[1, 0].bar(centros, [b["n"] for b in g["intervalos"]], width=anchob * 0.92, color=TINTA_2)
    axs[1, 0].set_ylabel("n")
    axs[1, 0].set_xlabel("confianza (probabilidad máxima del softmax)")
    ax = axs[0, 1]
    ax.plot([0, 1], [0, 1], color=TINTA_2, lw=1, ls="--")
    marcadores = ["o", "s", "^", "D", "v"]
    for j, s in enumerate(sorted(cal["por_senante"])):
        f = cal["por_senante"][s]
        pts = [(b["confianza_media"], b["exactitud"]) for b in f["intervalos"] if b["n"] >= 5]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marcadores[j], ms=5, lw=1.6,
                color=PALETA[j], label=f"{s} (ECE {f['ece']:.3f})".replace(".", ","))
        axs[1, 1].plot(centros, [b["n"] for b in f["intervalos"]], marker=marcadores[j], ms=3, lw=1.2, color=PALETA[j])
    ax.set_ylim(0, 1)
    ax.set_title("Por señante (intervalos con n ≥ 5)", fontsize=10, color=TINTA)
    ax.legend(frameon=False, fontsize=7, loc="upper left", title="señante")
    axs[1, 1].set_xlabel("confianza (probabilidad máxima del softmax)")
    axs[1, 1].set_ylabel("n")
    for a in axs.flat:
        a.grid(color=REJILLA_COLOR, lw=0.6)
        a.set_axisbelow(True)
        a.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(salida / "fiabilidad.png", dpi=150)
    plt.close(fig)
    hechas.append("fiabilidad.png")
    return hechas


# ---------------------------------------------------------------------------------------------------
# Informe en Markdown
# ---------------------------------------------------------------------------------------------------

def _n(x, d: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{d}f}".replace(".", ",")


def _pct(x, d: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{d}f} %".replace(".", ",")


def _p(x) -> str:
    if x is None:
        return "—"
    return "< 0,0001" if x < 1e-4 else _n(x, 4)


def _pp(x) -> str:
    """«= 0,xxxx» o «< 0,0001», para escribir «p {_pp(x)}»."""
    t = _p(x)
    return t if t.startswith("<") else "= " + t


def _dif(x, medida: str) -> str:
    """Diferencia con signo: fotogramas para la longitud, puntos porcentuales para proporciones, 3 decimales si no."""
    if x is None:
        return "—"
    signo = "+" if x > 0 else ""
    if medida == "longitud":
        return f"{signo}{_n(x, 2)} fotogramas"
    if medida.startswith("proporcion"):
        return f"{signo}{_n(100 * x, 2)} pp"
    return f"{signo}{_n(x, 3)}"


def _fmt(x, fmt: str) -> str:
    return {"n3": lambda v: _n(v, 3), "n1": lambda v: _n(v, 1), "pct": _pct}[fmt](x)


def _s(i: int) -> str:
    return f"{nombre_sena(i)} ({i})"


def informe_md(a: dict) -> str:
    L: list[str] = []
    w = L.append
    ctl = a["controles"]
    ps = a["por_senante"]
    sen = sorted(ps)
    w("# Análisis de errores del modelo «manos» (actividad 4.d)\n")
    w(f"Generado por `src/analisis_errores.py` el {a['fecha']}. SOLO REPORTE: no toma decisiones ni reentrena.")
    w(f"Datos: predicciones de prueba ya guardadas en `{a['fuentes']['loso_dir']}/pliegue_k/predicciones.json` "
      f"(el pliegue k deja fuera al señante k), {a['global']['n']} predicciones. Todas las cifras están en "
      "`analisis.json`; cada tabla indica la clave de donde sale. Decimales con coma.\n")

    w("## 0. Controles (antes de reportar)\n")
    w("Fuente: `analisis.json › controles`.\n")
    w("| pliegue | señante | n | top-1 calculado | top-1 guardado | top-3 | F1 macro | matriz y recall por clase |")
    w("|---|---|---|---|---|---|---|---|")
    for f in ctl["pliegues"]:
        w(f"| {f['pliegue']} | {f['senante']} | {f['n']} | {_n(f['top1_calculado'])} | {_n(f['top1_guardado'])} | "
          f"{_n(f['top3_calculado'])} ({'coincide' if f['top3_coincide'] else 'NO coincide'}) | "
          f"{_n(f['f1_macro_calculado'])} ({'coincide' if f['f1_macro_coincide'] else 'NO coincide'}) | "
          f"{'coinciden' if f['matriz_coincide'] and f['recall_por_clase_coincide'] else 'NO coinciden'} |")
    w(f"\nTop-1 media ± DE de los pliegues: {_n(ctl['top1_media_calculada'])} ± {_n(ctl['top1_de_calculada'])} "
      f"(guardado en resultados_loso.json: {_n(ctl['top1_media_guardada'])} ± {_n(ctl['top1_de_guardada'])}; "
      f"{'coincide' if ctl['top1_agregado_coincide'] else 'NO coincide'}). ")
    u = a["calibracion"]["umbral"]["global"]
    n33 = ("" if "umbral_vs_n33" not in ctl else
           f"frente a N-33, 540 / 297: {'coincide' if ctl['umbral_vs_n33'] else 'NO coincide'}; ")
    w(f"Umbral sobre la prueba LOSO (t = {_n(u['umbral'], 2)}; {n33}fuente `calibracion.umbral.global`): rechazo {_pct(u['rechazo'])}, "
      f"exactitud entre aceptadas {_n(u['exactitud_aceptadas'])}, aceptadas {u['aceptadas_correctas']} correctas / "
      f"{u['aceptadas_erroneas']} erróneas, aciertos perdidos {_pct(u['aciertos_perdidos'])}.")
    if "grupos_n30_vs_registro" in ctl:
        w(f"Control de los grupos de N-30 (recall agregado, 3 decimales; "
          f"{'coincide' if ctl.get('grupos_n30_vs_registro_coincide') else 'NO coincide'}): {ctl['grupos_n30_vs_registro']}.")
    if ctl.get("deteccion_vs_x_padded") is not None:
        w(f"Control de fotogramas sin mano: METRICAS_FOTOGRAMA frente a bloques en ceros de X_padded en las 997 "
          f"muestras: {'coinciden' if ctl['deteccion_vs_x_padded'] else 'NO coinciden'}; filas por muestra frente a "
          f"lengths.npy: {'coinciden' if ctl['fotogramas_vs_lengths'] else 'NO coinciden'}.")
    g = a["global"]
    w(f"\nAgregado de las {g['n']} predicciones (no es la media de pliegues): top-1 {_n(g['top1'])}, top-3 {_n(g['top3'])}, "
      f"F1 macro {_n(a['por_sena']['f1_macro_agregado'])}, recall macro {_n(a['por_sena']['recall_macro_agregado'])}. "
      "Fuente: `global`, `por_sena`.\n")

    # 1
    s1 = a["por_sena"]
    w("## 1. Métricas por seña\n")
    sop = s1["soporte_por_sena"]
    otras = ", ".join(f"{v} en la seña {k}" for k, v in sop["otras"].items())
    spp = s1["soporte_por_pliegue_mas_frecuente"]
    otras_pl = "; ".join(f"pliegue {k}: " + ", ".join(f"{v} en la seña {i}" for i, v in d.items())
                         for k, d in s1["soporte_por_pliegue_otras"].items())
    w(f"Fuente: `por_sena.senas` (agregado sobre los pliegues; soporte {sop['mas_frecuente']} por seña"
      f"{', salvo ' + otras if otras else ''}). DE = desviación estándar muestral entre pliegues ({spp} muestras por seña "
      f"en cada pliegue{', salvo ' + otras_pl if otras_pl else ''}: cada muestra mueve el recall del pliegue en "
      f"{_n(1 / spp, 2) if spp else '—'} o más). Precisión indefinida (seña nunca predicha) = «—».\n")
    idx = {f["id"]: f for f in s1["senas"]}

    def tabla_senas(ids):
        w("| seña | categoría | recall | recall por pliegue (0..4) | DE recall | precisión | DE precisión (pliegues definidos) | F1 | DE F1 | veces predicha |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for i in ids:
            f = idx[i]
            w(f"| {_s(i)} | {f['categoria']} | {_n(f['recall'])} | {' / '.join(_n(x, 2) for x in f['por_pliegue']['recall'])} | "
              f"{_n(f['recall_de_pliegues'])} | {_n(f['precision'])} | {_n(f['precision_de_pliegues'])} ({f['precision_pliegues_definidos']}) | "
              f"{_n(f['f1'])} | {_n(f['f1_de_pliegues'])} | {f['n_predichas']} |")

    w(f"**Peores 10** ({s1['criterio_ranking']}):\n")
    tabla_senas(s1["peores_10"])
    w("\n**Mejores 10:**\n")
    tabla_senas(s1["mejores_10"])
    w(f"\nSeñas con recall agregado 0: {', '.join(_s(i) for i in s1['senas_con_recall_0']) or 'ninguna'}. "
      f"Señas nunca predichas en las 997: {', '.join(_s(i) for i in s1['senas_nunca_predichas']) or 'ninguna'}.")
    w("La tabla completa de las 50 señas está en `analisis.json › por_sena.senas`; la figura `recall_por_sena.png` "
      "muestra el agregado y los 5 pliegues.\n")

    # 2
    c = a["confusiones"]
    w("## 2. Pares de confusión\n")
    w(f"Fuente: `confusiones`. {c['n_errores']} errores repartidos en {c['n_pares_distintos']} pares real → predicha "
      f"distintos; {c['n_pares_con_conteo_ge_3']} pares aparecen 3 o más veces. Criterio: {c['criterio_simetria']}. "
      f"Proporción de errores en pares con al menos un caso en el sentido inverso: "
      f"{_pct(c['proporcion_errores_en_pares_con_ambos_sentidos'])}; en pares simétricos según el criterio: "
      f"{_pct(c['proporcion_errores_en_pares_simetricos'])} ({c['n_parejas_simetricas']} parejas).\n")
    w("**Pares dirigidos más frecuentes** (`confusiones.pares`, los 20 primeros):\n")
    w("| real → predicha | conteo | inverso | menor / mayor | simétrica | misma categoría | señantes en que ocurre |")
    w("|---|---|---|---|---|---|---|")
    for p in c["pares"][:20]:
        w(f"| {_s(p['real'])} → {_s(p['predicha'])} | {p['conteo']} | {p['conteo_inverso']} | {_n(p['cociente_menor_mayor'], 2)} | "
          f"{'sí' if p['simetrica'] else 'no'} | "
          f"{'sí (' + p['categoria_real'] + ')' if p['misma_categoria'] else p['categoria_real'] + ' → ' + p['categoria_predicha']} | "
          f"{', '.join(p['senantes'])} |")
    w("\n**Parejas no ordenadas con más errores** (`confusiones.parejas_no_ordenadas`, 10 primeras):\n")
    w("| pareja {a, b} | a → b | b → a | total | simétrica |")
    w("|---|---|---|---|---|")
    for p in c["parejas_no_ordenadas"][:10]:
        w(f"| {_s(p['a'])} / {_s(p['b'])} | {p['a_a_b']} | {p['b_a_a']} | {p['total']} | {'sí' if p['simetrica'] else 'no'} |")
    pc = c["por_categoria"]
    w(f"\n**Por categoría gramatical** (`confusiones.por_categoria`): {pc['errores_dentro_de_la_misma_categoria']} de "
      f"{c['n_errores']} errores ({_pct(pc['proporcion_dentro'])}) caen en otra seña de la misma categoría; si cada error "
      f"fuera a una de las otras 49 señas al azar se esperaría {_pct(pc['proporcion_dentro_esperada_si_errores_al_azar'])}.\n")
    w("Matriz de errores categoría real (fila) → categoría predicha (columna):\n")
    w("| real \\ predicha | " + " | ".join(ORDEN_CATEGORIAS) + " | errores | % dentro |")
    w("|---|" + "---|" * (len(ORDEN_CATEGORIAS) + 2))
    for cr in ORDEN_CATEGORIAS:
        det = pc["detalle"][cr]
        w(f"| {cr} ({det['n_senas']}) | " + " | ".join(str(pc["matriz_errores_categoria_real_a_predicha"][cr][cp]) for cp in ORDEN_CATEGORIAS)
          + f" | {det['errores']} | {_pct(det['proporcion_dentro'])} |")
    w("\nSeñas que más reciben predicciones erróneas (`confusiones.predichas_mas_frecuentes_en_errores`): "
      + "; ".join(f"{_s(d['id'])} {d['veces_predicha_por_error']}" for d in c["predichas_mas_frecuentes_en_errores"][:6]) + ".\n")

    # 3
    w("## 3. Por señante\n")
    w("Fuente: `por_senante` (pliegue k = señante k fuera del entrenamiento).\n")
    w("| señante | n | top-1 | top-3 | F1 macro | errores | señas con recall 0 (de 4) | seña más predicha (veces) | confianza media aciertos / errores | errores con confianza ≥ 0,5 | errores con la real en top-3 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in sen:
        d = ps[s]
        w(f"| {s} | {d['n']} | {_n(d['top1'])} | {_n(d['top3'])} | {_n(d['f1_macro'])} | {d['n_errores']} | "
          f"{len(d['senas_con_recall_0'])} | {_s(d['predicha_mas_frecuente'])} ({d['n_predichas_de_la_mas_frecuente']}) | "
          f"{_n(d['confianza_media_aciertos'])} / {_n(d['confianza_media_errores'])} | {d['errores_con_confianza_ge_0_5']} | "
          f"{d['errores_con_real_en_top3']} |")
    w("\nSeñas con recall 0 y pares más frecuentes de cada señante:\n")
    for s in sen:
        d = ps[s]
        w(f"- **{s}**: recall 0 en {', '.join(_s(i) for i in d['senas_con_recall_0']) or 'ninguna'}. "
          f"Pares: {'; '.join(f'{_s(r)} → {_s(p)} ×{k}' for r, p, k in d['pares_principales'][:5])}.")
    ev = a["evidencia_senantes"]
    if ev:
        w("\n### Variables medidas por señante (¿qué distingue al 0000?)\n")
        w("Fuente: `evidencia_senantes`. Longitudes: `lengths.npy` (fotogramas antes de remuestrear). Fotogramas sin "
          "mano: `data/extracted/LSC50_holistic/METRICAS_FOTOGRAMA/*.csv` (columnas mano_izq_detectada y "
          "mano_der_detectada del extractor; izq/der son los bloques del extractor). Movimiento: desplazamiento medio "
          "por fotograma de la muñeca en el espacio normalizado por hombros (`X_padded.npy`, solo fotogramas "
          "consecutivos con la mano presente). Solo muestras de prueba de cada señante.\n")
        w("| variable | " + " | ".join(sen) + " | menor | mayor |")
        w("|---|" + "---|" * (len(sen) + 2))
        for clave, etq, fmt in VARIABLES_EVIDENCIA:
            if ev["por_senante"][sen[0]].get(clave) is None:
                continue
            ex = ev["extremos"].get(clave, {})
            w(f"| {etq} | " + " | ".join(_fmt(ev["por_senante"][s][clave], fmt) for s in sen)
              + f" | {ex.get('menor', '—')} | {ex.get('mayor', '—')} |")
        if ev.get("nivel_muestra"):
            w("\n**Nivel de muestra, dentro de cada señante por separado** (`evidencia_senantes.nivel_muestra.<medida>.por_senante`; "
              "p bilateral por permutación de acierto/error entre las muestras de ese señante; orientativo):\n")
            w("| medida | señante | media aciertos | media errores | muestras con valor > 0 | exactitud si 0 | exactitud si > 0 | p |")
            w("|---|---|---|---|---|---|---|---|")
            for nombre, d in ev["nivel_muestra"].items():
                fmt = "n1" if nombre == "longitud" else "pct"
                for s in sen:
                    x = d["por_senante"][s]
                    w(f"| {nombre} | {s} | {_fmt(x['media_aciertos'], fmt)} | {_fmt(x['media_errores'], fmt)} | "
                      f"{x['muestras_con_valor_mayor_que_0'] if nombre != 'longitud' else '—'} | "
                      f"{_n(x['exactitud_si_valor_0']) if nombre != 'longitud' else '—'} | "
                      f"{_n(x['exactitud_si_valor_mayor_que_0']) if nombre != 'longitud' else '—'} | {_p(x['permutacion']['p_bilateral'])} |")
            fam = a.get("pruebas_nivel_muestra", {})
            w("\n**Nivel de muestra, todas las muestras con estratos** (`evidencia_senantes.nivel_muestra.<medida>.estratificada_por_*`). "
              "Diferencia dentro de estrato = media ponderada de (media de errores − media de aciertos) de cada estrato, con "
              "peso errores·aciertos/n del estrato (igual al coeficiente de una regresión con efectos fijos del estrato); "
              "p bilateral por permutación de acierto/error dentro de cada estrato. La diferencia bruta (todas las muestras "
              "juntas) se muestra solo como referencia: no es la que se prueba. «Dentro de seña» compara clips de la misma "
              "seña; «dentro de seña y señante» compara las repeticiones de la misma seña del mismo señante. "
              f"p Holm: corrección de Holm sobre las {fam.get('n_pruebas', '—')} pruebas estratificadas de §3 y §6.\n")
            w("| medida | estratificación | estratos con errores y aciertos | diferencia dentro de estrato (errores − aciertos) | diferencia bruta | p | p Holm |")
            w("|---|---|---|---|---|---|---|")
            for nombre, d in ev["nivel_muestra"].items():
                for e, (etq_e, _) in ESTRATIFICACIONES.items():
                    t = d[f"estratificada_por_{e}"]
                    w(f"| {nombre} | {etq_e} | {t['estratos_con_errores_y_aciertos']} de {t['estratos']} | "
                      f"{_dif(t['diferencia_dentro_de_estratos'], nombre)} | {_dif(t['diferencia_bruta_sin_estratos'], nombre)} | "
                      f"{_p(t['p_bilateral'])} | {_p(t.get('p_holm'))} |")
        w("\n" + a["lectura"]["senante_0000"] + "\n")

    # 4
    gr = a["por_grupo"]
    w("## 4. Por grupo de señas\n")
    w("Fuente: `por_grupo`. Recall = aciertos / muestras del grupo; «dentro del grupo» = la predicción cae en alguna "
      "seña del grupo (acierto o confusión interna); precisión del grupo = aciertos entre las predicciones que caen en "
      "el grupo. DE entre señantes con ddof = 1.\n")

    def tabla_grupos(dic):
        w("| grupo | señas | n | recall | dentro del grupo | precisión del grupo | F1 medio de sus señas | recall por señante " + " / ".join(sen) + " | DE |")
        w("|---|---|---|---|---|---|---|---|---|")
        for nombre, d in dic.items():
            w(f"| {nombre} | {d['n_senas']} | {d['n_muestras']} | {_n(d['recall'])} | {_pct(d['proporcion_predicha_dentro_del_grupo'])} | "
              f"{_n(d['precision_del_grupo'])} | {_n(d['f1_medio_senas'])} | "
              + " / ".join(_n(d["recall_por_senante"][s], 2) for s in sen) + f" | {_n(d['recall_de_senantes'])} |")

    w("**Grupos de N-30** (interrogativos 41-46 y 26; emociones 20-23; valoración 24, 25, 27):\n")
    tabla_grupos(gr["grupos_n30"])
    w("\n**Categorías de GRAMMAR_CATEGORIES** (src/grammar_mapper.py):\n")
    tabla_grupos(gr["categorias_gramaticales"])
    w("")

    # 5
    cal = a["calibracion"]
    w("## 5. Calibración y umbral\n")
    w(f"Fuente: `calibracion`. Confianza = `prob_predicha` (máximo del softmax). {cal['global']['n_bins']} intervalos "
      "iguales; ECE = Σ (n_b/N)·|exactitud_b − confianza_b|; MCE = máximo de |exactitud_b − confianza_b|. "
      "Figura: `fiabilidad.png`.\n")
    w("| intervalo | n | confianza media | exactitud | confianza − exactitud |")
    w("|---|---|---|---|---|")
    for b in cal["global"]["intervalos"]:
        w(f"| [{_n(b['desde'], 1)}; {_n(b['hasta'], 1)}) | {b['n']} | {_n(b['confianza_media'])} | {_n(b['exactitud'])} | {_n(b['brecha'])} |")
    gg = cal["global"]
    w(f"\nAgregado: confianza media {_n(gg['confianza_media'])}, exactitud {_n(gg['exactitud'])}, ECE {_n(gg['ece'])}, "
      f"MCE {_n(gg['mce'])}.\n")
    t = cal["umbral"]["t"]
    w(f"**Umbral D-09 (t = {_n(t, 2)}, `{a['fuentes']['umbral_json']}`) y referencia 0,5, por señante** "
      "(`calibracion.umbral`, `calibracion.referencia_0_5`; aceptada si confianza ≥ t):\n")
    w("| señante | ECE | confianza media | exactitud | rechazo t | error entre aceptadas t | aciertos perdidos t | rechazo 0,5 | error entre aceptadas 0,5 | aciertos perdidos 0,5 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for s in sen + ["agregado"]:
        f = cal["por_senante"][s] if s != "agregado" else cal["global"]
        uu = cal["umbral"]["por_senante"][s] if s != "agregado" else cal["umbral"]["global"]
        rr = cal["referencia_0_5"]["por_senante"][s] if s != "agregado" else cal["referencia_0_5"]["global"]
        w(f"| {s} | {_n(f['ece'])} | {_n(f['confianza_media'])} | {_n(f['exactitud'])} | {_pct(uu['rechazo'])} | "
          f"{_pct(uu['error_aceptadas'])} ({uu['aceptadas_erroneas']}/{uu['aceptadas']}) | {_pct(uu['aciertos_perdidos'])} | "
          f"{_pct(rr['rechazo'])} | {_pct(rr['error_aceptadas'])} ({rr['aceptadas_erroneas']}/{rr['aceptadas']}) | {_pct(rr['aciertos_perdidos'])} |")
    w("\n" + a["lectura"]["calibracion"] + "\n")

    # 6
    lo = a["longitud"]
    w("## 6. Error y longitud del clip\n")
    if lo is None:
        w("No disponible: faltan lengths.npy o metadata.json.\n")
    else:
        w(f"Fuente: `longitud`. {lo['definicion']}. Longitud media de aciertos {_n(lo['longitud_media_aciertos'], 1)} y de "
          f"errores {_n(lo['longitud_media_errores'], 1)} fotogramas; correlación de Pearson longitud–acierto "
          f"{_n(lo['correlacion_pearson_longitud_acierto'])}.\n")

        def tabla_q(filas, etq, d=1):
            w(f"| quintil | {etq} desde | hasta | n | exactitud |")
            w("|---|---|---|---|---|")
            for q in filas:
                w(f"| {q['cuantil']} | {_n(q['desde'], d)} | {_n(q['hasta'], d)} | {q['n']} | {_n(q['exactitud'])} |")

        w("Cortes por cuantiles (np.quantile); cada intervalo incluye su borde superior y el primero también el inferior; "
          "con valores repetidos en un borde los grupos no quedan iguales.\n")
        w("**Longitud absoluta** (mezcla el efecto del señante: cada señante tiene su ritmo):\n")
        tabla_q(lo["cuantiles_longitud_absoluta"], "fotogramas", 1)
        w("\n**Longitud relativa a la mediana del propio señante:**\n")
        tabla_q(lo["cuantiles_longitud_relativa_al_senante"], "cociente", 2)
        w("\n**Desvío |log(longitud / media de la misma seña en los otros 4 señantes)|** (cuánto se aparta el ritmo del "
          "clip del de los señantes con que se entrenó ese pliegue):\n")
        tabla_q(lo["cuantiles_desvio_log_vs_misma_sena_otros_senantes"], "desvío", 2)
        fam = a.get("pruebas_nivel_muestra", {})
        w("\n**Pruebas dentro de estrato** (`longitud.pruebas_longitud`, `longitud.pruebas_desvio_vs_misma_sena`; mismo "
          "estadístico que en §3: diferencia errores − aciertos dentro de cada estrato, combinada con peso "
          "errores·aciertos/n, y permutación dentro del estrato; "
          f"diferencia bruta sin estratos: {_dif(lo['diferencia_bruta_errores_menos_aciertos'], 'longitud')}; p Holm sobre "
          f"las {fam.get('n_pruebas', '—')} pruebas estratificadas):\n")
        w("| medida | estratificación | estratos con errores y aciertos | diferencia dentro de estrato | p | p Holm | permutaciones (semilla) |")
        w("|---|---|---|---|---|---|---|")
        for medida, clave in (("longitud", "pruebas_longitud"), ("desvío |log|", "pruebas_desvio_vs_misma_sena")):
            for e, t in lo[clave].items():
                etq_e = ESTRATIFICACIONES[e.removeprefix("estratificada_por_")][0]
                w(f"| {medida} | {etq_e} | {t['estratos_con_errores_y_aciertos']} de {t['estratos']} | "
                  f"{_dif(t['diferencia_dentro_de_estratos'], 'longitud' if medida == 'longitud' else 'desvio')} | "
                  f"{_p(t['p_bilateral'])} | {_p(t.get('p_holm'))} | {t['permutaciones']} ({t['semilla']}) |")
        es = lo["entre_senas"]
        w(f"\nEntre señas (`longitud.entre_senas`, {es['n']} señas): correlación de Pearson entre la longitud media de la "
          f"seña y su recall agregado {_n(es['pearson'])}, p {_pp(es['p_bilateral'])} (permutación del recall entre señas).\n")
        w("| señante | longitud media aciertos | longitud media errores | errores − aciertos | exactitud por tercil de longitud (corto / medio / largo) |")
        w("|---|---|---|---|---|")
        for s in sen:
            d = lo["por_senante"][s]
            w(f"| {s} | {_n(d['longitud_media_aciertos'], 1)} | {_n(d['longitud_media_errores'], 1)} | "
              f"{_dif(d['diferencia_errores_menos_aciertos'], 'longitud')} | "
              + " / ".join(_n(q["exactitud"], 2) for q in d["cuantiles_longitud"]) + " |")
        w("\n" + a["lectura"]["longitud"] + "\n")

    # 7
    w("## 7. Contraste por seña con otras representaciones\n")
    sop = a["por_sena"]["soporte_por_sena"]
    _m = sop["mas_frecuente"]
    _otras = sorted(set(sop["otras"].values()))
    _txt_sop = (f"{_m} muestras por seña" + (", salvo " + ", ".join(f"{v} en las señas {', '.join(k for k, x in sop['otras'].items() if x == v)}"
                                                                    for v in _otras) if _otras else "")
                + ": una muestra mueve el recall " + " o ".join(_n(1 / v, 3) for v in [_m] + _otras))
    w(f"Fuente: `contraste`. Mismas {a['global']['n']} muestras de prueba y mismas particiones (comprobado muestra a muestra). "
      f"Diferencia = recall «manos» − recall de la otra representación ({_txt_sop}). "
      "McNemar exacto sobre las muestras de la seña; 50 pruebas sin corrección por comparaciones múltiples, "
      "orientativo.\n")
    for nombre, d in a["contraste"].items():
        if not d.get("disponible") or not d.get("mismas_muestras"):
            w(f"- {nombre}: no disponible o con muestras distintas ({d.get('dir')}).")
            continue
        w(f"### manos frente a {nombre} (`{d['dir']}`)\n")
        w(f"Top-1 agregado: manos {_n(d['top1_manos'])}, {nombre} {_n(d['top1_' + nombre])}. Muestras que solo acierta "
          f"«manos»: {d['solo_acierta_manos']}; solo {nombre}: {d['solo_acierta_otro']}; McNemar global p "
          f"{_pp(d['p_mcnemar_global'])}.\n")
        filas = {f["id"]: f for f in d["senas"]}
        w("| seña | categoría | recall manos | recall " + nombre + " | diferencia | solo manos / solo otro | p McNemar |")
        w("|---|---|---|---|---|---|---|")
        seleccion = d["ganan_con_manos"][:10] + d["pierden_con_manos"][:10]
        for i in seleccion:
            f = filas[i]
            w(f"| {_s(i)} | {f['categoria']} | {_n(f['recall_manos'], 2)} | {_n(f['recall_' + nombre], 2)} | "
              f"{'+' if f['diferencia_manos_menos_otro'] > 0 else ''}{_n(f['diferencia_manos_menos_otro'], 2)} | "
              f"{f['solo_acierta_manos']} / {f['solo_acierta_otro']} | {_n(f['p_mcnemar'], 3)} |")
        w(f"\n(Arriba: hasta 10 señas que más ganan con «manos» y hasta 10 que más pierden.) Ganan con «manos» "
          f"{len(d['ganan_con_manos'])} señas, pierden {len(d['pierden_con_manos'])}, sin cambio {len(d['sin_cambio'])}. "
          f"Señas con p < 0,05: {', '.join(_s(i) for i in d['senas_con_p_mcnemar_lt_0_05']) or 'ninguna'}.")
        gg = d["recall_por_grupo_n30"]
        w("Recall por grupo de N-30 (manos / " + nombre + "): "
          + "; ".join(f"{g} {_n(v['recall_manos'])} / {_n(v['recall_' + nombre])}" for g, v in gg.items()) + ".\n")
    w("## Lectura prudente\n")
    for parrafo in a["lectura"]["general"]:
        w(f"- {parrafo}")
    w("\n## Figuras\n")
    for f in a["figuras"]:
        w(f"- `{f}`")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------
# Lectura (frases generadas a partir de las cifras, sin atribuir causas)
# ---------------------------------------------------------------------------------------------------

def _veredicto(t: dict) -> str:
    p, ph = t["p_bilateral"], t.get("p_holm")
    if p is None:
        return "sin comparación posible"
    if p >= 0.05:
        return "no distinguible del azar con α = 0,05"
    if ph is not None and ph >= 0.05:
        return "p < 0,05 sin corregir, pero no tras Holm"
    return "p < 0,05 también tras Holm"


def _frase_estratos(d: dict, medida: str, estratos=tuple(ESTRATIFICACIONES)) -> str:
    partes = []
    for e in estratos:
        t = d.get(f"estratificada_por_{e}")
        if t is None:
            continue
        partes.append(f"{ESTRATIFICACIONES[e][0]} {_dif(t['diferencia_dentro_de_estratos'], medida)} "
                      f"(p {_pp(t['p_bilateral'])}; Holm {_p(t.get('p_holm'))}: {_veredicto(t)})")
    return "; ".join(partes)


def _frase_consistencia(d: dict) -> str:
    ts = [d[f"estratificada_por_{e}"] for e in ESTRATIFICACIONES if f"estratificada_por_{e}" in d]
    difs = [t["diferencia_dentro_de_estratos"] for t in ts if t["diferencia_dentro_de_estratos"] is not None]
    holm_ok = sum(1 for t in ts if t.get("p_holm") is not None and t["p_holm"] < 0.05)
    signo = ("El signo es el mismo en las tres estratificaciones" if difs and (all(x > 0 for x in difs) or all(x < 0 for x in difs))
             else "El signo cambia según la estratificación")
    return f"{signo}; comparaciones con p Holm < 0,05: {holm_ok} de {len(ts)}."


def _lectura_longitud(lo: dict) -> str:
    pl, pdv = lo["pruebas_longitud"], lo["pruebas_desvio_vs_misma_sena"]
    t_sen, t_sena = pl["estratificada_por_senante"], pl["estratificada_por_sena"]
    difs_s = {s: d["diferencia_errores_menos_aciertos"] for s, d in lo["por_senante"].items()}
    negativos = sum(1 for x in difs_s.values() if x is not None and x < 0)
    es = lo["entre_senas"]
    frases = [
        f"**Lectura.** Errores − aciertos en longitud del clip: {_frase_estratos(pl, 'longitud')}. La diferencia bruta, "
        f"sin estratos, es {_dif(lo['diferencia_bruta_errores_menos_aciertos'], 'longitud')}.",
        f"Por señante, la diferencia es negativa (errores más cortos) en {negativos} de {len(difs_s)}: "
        + ", ".join(f"{s} {_dif(x, 'longitud')}" for s, x in sorted(difs_s.items())) + ".",
        f"Entre las {es['n']} señas, la correlación entre la longitud media de la seña y su recall es {_n(es['pearson'])} "
        f"(p {_pp(es['p_bilateral'])}).",
        f"Desvío frente al ritmo de la misma seña en los otros señantes: {_frase_estratos(pdv, 'desvio', ('senante', 'sena'))}.",
    ]
    difs = [t["diferencia_dentro_de_estratos"] for t in pl.values() if t["diferencia_dentro_de_estratos"] is not None]
    cambia = not (all(x > 0 for x in difs) or all(x < 0 for x in difs))
    alguna_holm = any(t.get("p_holm") is not None and t["p_holm"] < 0.05 for t in pl.values())
    if (t_sen["p_bilateral"] is not None and t_sen["p_bilateral"] < 0.05 and t_sena["p_bilateral"] is not None
            and t_sena["p_bilateral"] >= 0.05):
        frase = ("La diferencia dentro de señante no aparece al comparar clips de la misma seña")
        if es["pearson"] is not None and (es["pearson"] > 0) == (t_sen["diferencia_dentro_de_estratos"] < 0):
            frase += (", y el sentido de la correlación entre señas coincide con ella: es compatible con que se fallen "
                      "más las señas de clips más cortos (qué seña es), no muestra que un clip más corto que otros de su "
                      "misma seña falle más")
        frases.append(frase + ".")
    conclusion = []
    if cambia:
        conclusion.append("el signo cambia según con qué clips se compare")
    if not alguna_holm:
        conclusion.append("ninguna diferencia de longitud queda con p Holm < 0,05")
    if conclusion:
        frases.append("Respuesta a «¿fallan más los clips cortos o los largos?»: con estos datos no hay una respuesta única; "
                      + " y ".join(conclusion) + ". Son asociaciones, no causas.")
    else:
        frases.append("Son asociaciones a nivel de muestra, no causas.")
    return " ".join(frases)


def lecturas(a: dict) -> dict:
    ps, ev = a["por_senante"], a["evidencia_senantes"]
    sen = sorted(ps)
    peor = min(sen, key=lambda s: ps[s]["top1"])
    out = {}
    # señante difícil
    frases = []
    if ev:
        e = ev["extremos"]
        etq = {k: t for k, t, _ in VARIABLES_EVIDENCIA}
        entradas = [v for v in e if v not in ("top1", "confianza_media")]
        fmt_de = {k: f for k, _, f in VARIABLES_EVIDENCIA}

        def _extremo(v):
            lado = "mayor" if e[v]["mayor"] == peor else "menor"
            segundo = e[v]["valor_segundo_mayor"] if lado == "mayor" else e[v]["valor_segundo_menor"]
            return (f"{etq[v]} ({lado}: {_fmt(e[v]['valor_' + lado], fmt_de[v])}; el siguiente "
                    f"{_fmt(segundo, fmt_de[v])})")

        extremos = [_extremo(v) for v in entradas if e[v]["menor"] == peor or e[v]["mayor"] == peor]
        no_ext = [etq[v] for v in entradas if e[v]["menor"] != peor and e[v]["mayor"] != peor]
        frases.append(f"El señante con menor top-1 es {peor} ({_n(ps[peor]['top1'])}). Entre las variables medidas de "
                      f"sus clips, {peor} ocupa un extremo de los 5 en: {'; '.join(extremos) or 'ninguna'}; no es extremo en: "
                      f"{'; '.join(no_ext) or 'ninguna'}.")
        frases.append("Con 5 señantes no se puede estimar si alguna de estas variables explica el top-1; que un señante "
                      "sea extremo en una variable es compatible con que esa variable influya, no lo demuestra.")
        nm = ev.get("nivel_muestra", {})
        partes = []
        for nombre in ("proporcion_sin_alguna_mano", "proporcion_sin_mano_der", "longitud"):
            if nombre in nm:
                x = nm[nombre]["por_senante"][peor]
                fmt = "n1" if nombre == "longitud" else "pct"
                partes.append(f"{nombre}: aciertos {_fmt(x['media_aciertos'], fmt)}, errores {_fmt(x['media_errores'], fmt)} "
                              f"(p {_pp(x['permutacion']['p_bilateral'])})")
        if partes:
            frases.append(f"A nivel de muestra, dentro de {peor}: " + "; ".join(partes) + ".")
    c = a["calibracion"]["por_senante"][peor]
    frases.append(f"La seña que más predice el modelo con {peor} es {_s(ps[peor]['predicha_mas_frecuente'])} "
                  f"({ps[peor]['n_predichas_de_la_mas_frecuente']} de {ps[peor]['n']} predicciones, 4 reales).")
    frases.append(f"Salida del modelo (no es una propiedad de los clips): confianza media {_n(c['confianza_media'])} frente a una exactitud de {_n(c['exactitud'])} "
                  f"(ECE {_n(c['ece'])}); {ps[peor]['errores_con_confianza_ge_0_5']} de sus {ps[peor]['n_errores']} errores "
                  f"tienen confianza ≥ 0,5 y {ps[peor]['errores_con_real_en_top3']} tienen la seña real en el top-3.")
    out["senante_0000"] = ("**Lectura (señante con menor top-1).** " + " ".join(frases)).replace("..", ".")

    cal = a["calibracion"]
    g = cal["global"]
    signo = "por encima" if g["confianza_media"] > g["exactitud"] else "por debajo"
    u = cal["umbral"]
    rech = {s: u["por_senante"][s]["rechazo"] for s in sen}
    err = {s: u["por_senante"][s]["error_aceptadas"] for s in sen}
    con_n = [b for b in g["intervalos"] if b["n"]]
    bajo = [b for b in con_n if b["confianza_media"] < b["exactitud"]]
    out["calibracion"] = (
        f"**Lectura.** En la prueba con señantes nuevos la confianza media ({_n(g['confianza_media'])}) queda {signo} de la "
        f"exactitud ({_n(g['exactitud'])}); ECE agregado {_n(g['ece'])}. En {len(bajo)} de {len(con_n)} intervalos con "
        f"datos la confianza media es menor que la exactitud (intervalos que empiezan en "
        f"{', '.join(_n(b['desde'], 1) for b in bajo) or 'ninguno'}). Con t = {_n(u['t'], 2)} el rechazo va de "
        f"{_pct(min(rech.values()))} ({min(rech, key=rech.get)}) a {_pct(max(rech.values()))} ({max(rech, key=rech.get)}) y el "
        f"error entre aceptadas de {_pct(min(err.values()))} ({min(err, key=err.get)}) a {_pct(max(err.values()))} "
        f"({max(err, key=err.get)}). Esto describe el umbral ya elegido (D-09) sobre la prueba; no es un criterio para "
        "cambiarlo (N-33).")
    lo = a["longitud"]
    if lo:
        out["longitud"] = _lectura_longitud(lo)
    else:
        out["longitud"] = ""
    s1 = a["por_sena"]
    cf = a["confusiones"]
    top = cf["parejas_no_ordenadas"][0] if cf["parejas_no_ordenadas"] else None
    gen = [
        f"Top-1 agregado {_n(a['global']['top1'])} (media de pliegues {_n(a['controles']['top1_media_calculada'])} ± "
        f"{_n(a['controles']['top1_de_calculada'])}); señas con recall agregado 0: {len(s1['senas_con_recall_0'])}; "
        f"con recall ≥ 0,9: {sum(1 for f in s1['senas'] if f['recall'] is not None and f['recall'] >= 0.9)}.",
    ]
    if top:
        gen.append(f"La pareja más confundida es {_s(top['a'])} / {_s(top['b'])} con {top['total']} errores "
                   f"({top['a_a_b']} y {top['b_a_a']} en cada sentido).")
    pcat = cf["por_categoria"]
    gen.append(f"{_pct(pcat['proporcion_dentro'])} de los errores quedan dentro de la misma categoría gramatical, frente a "
               f"{_pct(pcat['proporcion_dentro_esperada_si_errores_al_azar'])} esperado al azar: las categorías de "
               "grammar_mapper son sintácticas, no de forma manual, así que esta cifra no mide parecido articulatorio.")
    nm = (ev or {}).get("nivel_muestra", {})
    etiquetas_mano = {"proporcion_sin_mano_izq": "fotogramas sin el bloque de mano izquierda",
                      "proporcion_sin_mano_der": "fotogramas sin el bloque de mano derecha",
                      "proporcion_sin_alguna_mano": "fotogramas sin alguna mano"}
    for nombre, etq_m in etiquetas_mano.items():
        if nombre in nm:
            gen.append(f"Proporción de {etq_m}, errores − aciertos: " + _frase_estratos(nm[nombre], nombre) + ". "
                       + _frase_consistencia(nm[nombre]))
    if any(n in nm for n in etiquetas_mano):
        gen.append("Las diferencias de fotogramas sin mano son asociaciones a nivel de muestra: no dicen que la ausencia "
                   "de mano cause el error ni explican por sí solas las diferencias entre señantes.")
    otras_pl = s1["soporte_por_pliegue_otras"]
    n_exc = sum(len(d) for d in otras_pl.values())
    gen.append(f"Cada seña tiene {s1['soporte_por_pliegue_mas_frecuente']} muestras por señante"
               + (f" ({s1['soporte_por_pliegue_min']} en {n_exc} casos)" if n_exc else "")
               + f" y {s1['soporte_por_sena']['mas_frecuente']} en el agregado"
               + (f" ({', '.join(str(v) + ' en la seña ' + k for k, v in s1['soporte_por_sena']['otras'].items())})"
                  if s1["soporte_por_sena"]["otras"] else "")
               + ": las diferencias por seña de una o dos muestras no deben leerse como efectos.")
    gen.append("No se toma ninguna decisión a partir de este análisis; las hipótesis que sugiera requieren una decisión "
               "nueva fijada antes de medir (como D-07, D-08 y D-09).")
    out["general"] = gen
    return out


# ---------------------------------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------------------------------

def aplicar_holm(a: dict) -> dict:
    """Familia de pruebas estratificadas a nivel de muestra (§3 y §6). Añade `p_holm` a cada prueba; la misma
    prueba repetida en §3 y §6 (longitud) cuenta una vez y debe tener el mismo p."""
    copias: dict[str, list[dict]] = {}
    for medida, d in ((a.get("evidencia_senantes") or {}).get("nivel_muestra") or {}).items():
        for clave, t in d.items():
            if clave.startswith("estratificada_por_"):
                copias.setdefault(f"{medida} / {clave.removeprefix('estratificada_por_')}", []).append(t)
    lo = a.get("longitud")
    if lo:
        for medida, grupo in (("longitud", lo["pruebas_longitud"]), ("desvio_log_vs_misma_sena", lo["pruebas_desvio_vs_misma_sena"])):
            for clave, t in grupo.items():
                copias.setdefault(f"{medida} / {clave.removeprefix('estratificada_por_')}", []).append(t)
    for clave, ts in copias.items():
        if len({(t["p_bilateral"], t["diferencia_dentro_de_estratos"]) for t in ts}) != 1:
            raise RuntimeError(f"la prueba {clave} dio resultados distintos en §3 y §6")
    ajustados = holm({k: ts[0]["p_bilateral"] for k, ts in copias.items()})
    for clave, ts in copias.items():
        for t in ts:
            t["p_holm"] = ajustados[clave]
    return {"metodo": "Holm-Bonferroni sobre las pruebas estratificadas a nivel de muestra de §3 y §6",
            "n_pruebas": sum(1 for v in ajustados.values() if v is not None),
            "pruebas": [{"prueba": k, "diferencia_dentro_de_estratos": ts[0]["diferencia_dentro_de_estratos"],
                         "p_bilateral": ts[0]["p_bilateral"], "p_holm": ajustados[k]}
                        for k, ts in sorted(copias.items(), key=lambda kv: (kv[1][0]["p_bilateral"] is None,
                                                                            kv[1][0]["p_bilateral"] or 0, kv[0]))]}


def analizar(loso_dir: Path, contrastes: dict[str, Path], processed: Path, dir_metricas: Path,
             umbral_json: Path, n_bins: int, n_perm: int, semilla: int) -> tuple[dict, list[dict]]:
    registros, resultados = cargar_loso(loso_dir)
    ctl = controles_loso(loso_dir, registros, resultados)
    muestras = [r["muestra"] for r in registros]
    umbral = float(json.loads(umbral_json.read_text(encoding="utf-8"))["umbral"])

    longitudes = _longitudes(processed)
    deteccion = _deteccion_por_muestra(dir_metricas, muestras)
    movimiento = _movimiento_munecas(processed, muestras, longitudes) if longitudes else None
    if deteccion and longitudes:
        ctl["fotogramas_vs_lengths"] = all(deteccion[m]["fotogramas"] == longitudes[m] for m in muestras)
    if deteccion and movimiento:
        ctl["deteccion_vs_x_padded"] = all(deteccion[m]["sin_izq"] == movimiento[m]["bloques_en_ceros_izq"]
                                           and deteccion[m]["sin_der"] == movimiento[m]["bloques_en_ceros_der"]
                                           for m in muestras)

    reales = [r["real"] for r in registros]
    a: dict = {
        "descripcion": "Análisis de errores del modelo «manos» (actividad 4.d) sobre la prueba LOSO agregada; solo reporte",
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "src/analisis_errores.py",
        "fuentes": {
            "loso_dir": _rel(loso_dir),
            "predicciones_sha256": {f"pliegue_{k}": sha256(loso_dir / f"pliegue_{k}" / "predicciones.json")
                                    for k in sorted({r['pliegue'] for r in registros})},
            "contrastes": {k: _rel(v) for k, v in contrastes.items()},
            "processed": _rel(processed) if longitudes else None,
            "metricas_fotograma": _rel(dir_metricas) if deteccion else None,
            "umbral_json": _rel(umbral_json),
            "vocabulario": "src/grammar_mapper.py (SIGN_VOCABULARY, GRAMMAR_CATEGORIES)",
        },
        "parametros": {"n_bins": n_bins, "permutaciones": n_perm, "semilla": semilla, "umbral": umbral},
        "controles": ctl,
        "global": {"n": len(registros), "top1": float(np.mean([r["real"] == r["predicha"] for r in registros])),
                   "top3": top_k(reales, [r["top3"] for r in registros], 3)},
    }
    a["por_sena"] = seccion_por_sena(registros)
    a["confusiones"] = seccion_confusiones(registros)
    a["por_senante"] = seccion_por_senante(registros)
    a["evidencia_senantes"] = seccion_evidencia_senantes(registros, longitudes, deteccion, movimiento, n_perm, semilla)
    a["por_grupo"] = seccion_grupos(registros, a["por_sena"])
    a["calibracion"] = seccion_calibracion(registros, umbral, n_bins)
    a["longitud"] = seccion_longitud(registros, longitudes, n_perm, semilla)
    a["contraste"] = seccion_contraste(registros, contrastes)
    a["pruebas_nivel_muestra"] = aplicar_holm(a)
    # controles frente al registro: solo tienen sentido con el LOSO «manos» por defecto
    es_manos = Path(loso_dir).resolve() == LOSO_DIR.resolve()
    ug = a["calibracion"]["umbral"]["global"]
    if es_manos and abs(umbral - 0.20) < 1e-12:
        ctl["umbral_vs_n33"] = all(ug[k] == v for k, v in REGISTRO_N33_T020.items())
    try:
        if not es_manos:
            raise KeyError("sin registro con que comparar")
        calc = {"manos": [round(a["por_grupo"]["grupos_n30"][g]["recall"], 3)
                          for g in ("interrogativos", "emociones", "valoracion", "resto")]}
        for nombre in ("manos_postura", "manos_postura_rostro"):
            d = a["contraste"].get(nombre, {})
            if d.get("mismas_muestras"):
                calc[nombre] = [round(d["recall_por_grupo_n30"][g]["recall_" + nombre], 3)
                                for g in ("interrogativos", "emociones", "valoracion", "resto")]
        ctl["grupos_n30_calculados"] = calc
        ctl["grupos_n30_vs_registro_coincide"] = all(
            all(abs(x - y) < 1e-9 for x, y in zip(v, REGISTRO_N30[k])) for k, v in calc.items())
        c1 = [a["por_grupo"]["grupos_n30"][g]["recall"] for g in ("interrogativos", "emociones", "valoracion", "resto")]
        partes = ["C1 " + " / ".join(f"{x:.3f}".replace(".", ",") for x in c1)]
        for nombre, etq in (("manos_postura", "C2"), ("manos_postura_rostro", "C3")):
            d = a["contraste"].get(nombre, {})
            if d.get("mismas_muestras"):
                partes.append(etq + " " + " / ".join(f"{d['recall_por_grupo_n30'][g]['recall_' + nombre]:.3f}".replace(".", ",")
                                                     for g in ("interrogativos", "emociones", "valoracion", "resto")))
        ctl["grupos_n30_vs_registro"] = ("; ".join(partes) + " (interrogativos / emociones / valoración / resto; "
                                         "N-30 registra C1 0,379 / 0,688 / 0,283 / 0,632; C2 0,264 / 0,562 / 0,350 / 0,598; "
                                         "C3 0,357 / 0,588 / 0,167 / 0,558)")
    except KeyError:
        pass
    return a, registros


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loso-dir", type=Path, default=LOSO_DIR)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--contraste", action="append", default=None, metavar="NOMBRE=DIR",
                    help="representación de contraste (repetible); por defecto manos_postura y manos_postura_rostro")
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--metricas-fotograma", type=Path, default=METRICAS_FOTOGRAMA)
    ap.add_argument("--umbral-json", type=Path, default=UMBRAL_JSON)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--permutaciones", type=int, default=10000)
    ap.add_argument("--semilla", type=int, default=2026)
    ap.add_argument("--sin-figuras", action="store_true")
    args = ap.parse_args(argv)

    salida = args.salida.resolve()
    datos = (BASE_DIR / "data").resolve()
    if salida == datos or datos in salida.parents:
        ap.error("la salida no puede estar dentro de data/")
    contrastes = CONTRASTES if args.contraste is None else {
        k: Path(v) for k, v in (c.split("=", 1) for c in args.contraste)}

    a, registros = analizar(args.loso_dir, contrastes, args.processed, args.metricas_fotograma, args.umbral_json,
                            args.bins, args.permutaciones, args.semilla)
    a["comando"] = " ".join(["python", "src/analisis_errores.py", *(sys.argv[1:] if argv is None else argv)])
    salida.mkdir(parents=True, exist_ok=True)
    a["figuras"] = [] if args.sin_figuras else figuras(a, registros, salida)
    a["lectura"] = lecturas(a)
    (salida / "analisis.json").write_text(json.dumps(a, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    (salida / "analisis.md").write_text(informe_md(a), encoding="utf-8")
    ctl = a["controles"]
    print(f"controles: {'todo coincide' if ctl['todo_coincide'] else 'HAY DIFERENCIAS (ver analisis.json › controles)'}")
    print(f"top-1 por pliegue: {' / '.join(f'{f['top1_calculado']:.3f}' for f in ctl['pliegues'])}; "
          f"media {ctl['top1_media_calculada']:.3f} ± {ctl['top1_de_calculada']:.3f}")
    print(f"escrito en {_rel(salida)}: analisis.json, analisis.md, {len(a['figuras'])} figuras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
