"""
Variantes del F1 (y de precisión y recall) sobre las predicciones de prueba ya guardadas. SOLO REPORTE:
no entrena, no decide y no toca ningún modelo, umbral ni partición.

Motivo (N-35): el mismo LOSO «manos» da F1 macro 0,537 si se promedian los cinco pliegues y 0,574 si se
calcula sobre las 997 predicciones juntas. Este script calcula, para cada sistema con predicciones
guardadas, las lecturas posibles y mide de dónde sale la diferencia, para que la tesis fije una.

Lecturas que calcula por sistema (LOSO de 5 pliegues, el pliegue k deja fuera al señante k):
  (a) F1 macro POR PLIEGUE y su media ± DE (ddof = 1) entre pliegues; también mín, máx y la media
      ponderada por el tamaño del pliegue (198-200).
  (b) F1 macro AGREGADO sobre las 997 predicciones (conteos TP/FP/FN sumados sobre los pliegues).
  (c) F1 ponderado por soporte, agregado.
  (d) F1 micro agregado, que en clasificación de una etiqueta por muestra es igual a la top-1.
  (e) Precisión y recall macro en las dos lecturas (media entre pliegues y agregado).
  (f) Intervalos por bootstrap percentil (N_BOOT remuestreos, semilla SEMILLA):
      · agregado: remuestreo de predicciones estratificado por seña (conserva el soporte 20/19 de cada
        seña, así ninguna seña desaparece del remuestreo);
      · media por pliegue: remuestreo de los 5 pliegues con reposición y, dentro de cada pliegue elegido,
        remuestreo de predicciones estratificado por seña; se reporta también el remuestreo de solo
        pliegues y el intervalo t clásico (media ± t_{4; 0,975} · DE / √5).
  (g) Descomposición exacta de la diferencia agregado − media por pliegue. Para cada seña c y pliegue k,
      D_kc = 2·TP + FP + FN. El F1 agregado de la seña es Σ_k 2TP_kc / Σ_k D_kc = Σ_k w_kc · F1_kc con
      w_kc = D_kc / Σ_k D_kc; la media por pliegue usa w = 1/5. La diferencia es
      (1/50) Σ_c Σ_k (w_kc − 1/5) · F1_kc, y se reparte entre las señas que tienen algún pliegue con
      0 aciertos y las que no. Se comprueba numéricamente que la identidad cierra.
  (h) Orden entre sistemas bajo cada lectura, y McNemar pareado sobre los aciertos top-1 (que por
      construcción no usa el F1): se reproducen las tablas de N-35 y N-39.
Para el esquema aleatorio (un solo conjunto de prueba de 100) solo existe la lectura agregada.

Convenciones (las mismas de train_loso.py, linea_base_clasica.py y analisis_errores.py):
  · macro sobre las 50 etiquetas, precisión o recall indefinidos (0/0) valen 0 (zero_division = 0);
  · desviación estándar entre pliegues con ddof = 1;
  · la línea base guarda las 997 predicciones en un solo archivo: el pliegue se deduce del señante del
    nombre de la muestra (SSSS_VVVV_RRRR), que es la regla del LOSO D-06.

Uso:
    python src/f1_variantes.py
    python src/f1_variantes.py --salida models/f1_variantes_2026-09-22 --n-boot 2000 --semilla 2026
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
N_CLASES = 50
N_PLIEGUES = 5
SEMILLA = 2026
N_BOOT = 2000
ALFA = 0.05
TOL = 1e-9
SALIDA_POR_DEFECTO = BASE_DIR / "models" / "f1_variantes_2026-09-22"

# Sistemas con predicciones guardadas. `loso_dir` → pliegue_k/predicciones.json (red);
# `loso_plano` → un solo archivo con las 997 (línea base); `aleatoria` → 100 predicciones del esquema
# aleatorio. El orden de la lista es el orden esperado por N-30 y N-39 (manos > manos+postura > rostro;
# red > línea base), que la sección de orden comprueba.
SISTEMAS_POR_DEFECTO = [
    {"clave": "red_manos", "nombre": "Red temporal, manos (C1; N-30)",
     "loso_dir": "models/loso_2026-09-15_holistic_mascara_manos",
     "aleatoria": "models/loso_2026-09-15_holistic_mascara_manos/aleatoria_80_10_10/predicciones.json"},
    {"clave": "red_manos_postura", "nombre": "Red temporal, manos + postura (C2; N-23/N-30)",
     "loso_dir": "models/loso_2026-09-14_holistic_mascara",
     "aleatoria": "models/loso_2026-09-14_holistic_mascara/aleatoria_80_10_10/predicciones.json"},
    {"clave": "red_manos_postura_rostro", "nombre": "Red temporal, manos + postura + cejas/labios (C3; N-30)",
     "loso_dir": "models/loso_2026-09-15_holistic_mascara_rostro",
     "aleatoria": "models/loso_2026-09-15_holistic_mascara_rostro/aleatoria_80_10_10/predicciones.json"},
    {"clave": "linea_base_random_forest", "nombre": "Línea base Random Forest (D-10, con espejo; N-39)",
     "loso_plano": "models/linea_base_4d_2026-09-15/predicciones_random_forest.json",
     "aleatoria": "models/linea_base_4d_2026-09-15/predicciones_random_forest_aleatoria.json"},
    {"clave": "linea_base_svm_rbf", "nombre": "Línea base SVM RBF (D-10, con espejo; N-39)",
     "loso_plano": "models/linea_base_4d_2026-09-15/predicciones_svm_rbf.json",
     "aleatoria": "models/linea_base_4d_2026-09-15/predicciones_svm_rbf_aleatoria.json"},
    {"clave": "linea_base_mlp", "nombre": "Línea base MLP (D-10, con espejo; N-39)",
     "loso_plano": "models/linea_base_4d_2026-09-15/predicciones_mlp.json",
     "aleatoria": "models/linea_base_4d_2026-09-15/predicciones_mlp_aleatoria.json"},
    {"clave": "canonico_manos", "nombre": "Modelo canónico «manos» (N-31; solo esquema aleatorio)",
     "aleatoria": "models/canonico_holistic_mascara_manos/evaluacion_modelo_seleccionado/predicciones.json"},
]

# Órdenes que N-30 y N-39 afirman y que aquí se comprueban bajo cada lectura.
ORDENES_ESPERADOS = [
    ("red_manos", "red_manos_postura"),
    ("red_manos_postura", "red_manos_postura_rostro"),
    ("red_manos", "linea_base_random_forest"),
    ("red_manos", "linea_base_svm_rbf"),
    ("red_manos", "linea_base_mlp"),
]
COMPARACIONES_MCNEMAR = [
    ("red_manos", "red_manos_postura"),
    ("red_manos", "red_manos_postura_rostro"),
    ("red_manos", "linea_base_random_forest"),
    ("red_manos", "linea_base_svm_rbf"),
    ("red_manos", "linea_base_mlp"),
]


# ---------------------------------------------------------------------------------------------------
# Carga


def parsear_muestra(nombre: str) -> tuple[int, str, int]:
    """'SSSS_VVVV_RRRR' → (seña, señante como texto de 4 dígitos, repetición)."""
    partes = nombre.split("_")
    if len(partes) != 3 or not all(p.isdigit() and len(p) == 4 for p in partes):
        raise ValueError(f"nombre de muestra no reconocido: {nombre!r}")
    return int(partes[0]), partes[1], int(partes[2])


def _leer_json(ruta: Path):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def _validar_filas(filas: list[dict], origen: str) -> None:
    for r in filas:
        for k in ("muestra", "real", "predicha"):
            if k not in r:
                raise ValueError(f"{origen}: falta la clave {k!r} en {r}")
        sena, _, _ = parsear_muestra(r["muestra"])
        if sena != int(r["real"]):
            raise ValueError(f"{origen}: {r['muestra']} tiene real={r['real']} y la seña del nombre es {sena}")
        if not (0 <= int(r["predicha"]) < N_CLASES):
            raise ValueError(f"{origen}: predicha fuera de rango en {r}")


def cargar_loso_pliegues(loso_dir: Path) -> list[dict]:
    """Lee pliegue_k/predicciones.json (k = 0..4) y comprueba que el señante de cada muestra es el que
    resultados_loso.json declara fuera en ese pliegue."""
    loso_dir = Path(loso_dir)
    resultados = _leer_json(loso_dir / "resultados_loso.json")
    registros = []
    for info in resultados["pliegues"]:
        k = int(info["pliegue"])
        ruta = loso_dir / f"pliegue_{k}" / "predicciones.json"
        filas = _leer_json(ruta)
        _validar_filas(filas, str(ruta))
        for r in filas:
            _, senante, rep = parsear_muestra(r["muestra"])
            if senante != info["senante_test"]:
                raise ValueError(f"{r['muestra']} está en pliegue_{k}, que deja fuera a {info['senante_test']}")
            registros.append({**r, "real": int(r["real"]), "predicha": int(r["predicha"]),
                              "pliegue": k, "senante": senante, "repeticion": rep})
    return registros


def cargar_loso_plano(ruta: Path) -> list[dict]:
    """Un solo archivo con las 997 predicciones (línea base). El pliegue es el señante del nombre
    (regla D-06: el pliegue k deja fuera al señante k). Exige exactamente los 5 señantes 0000..0004."""
    filas = _leer_json(ruta)
    _validar_filas(filas, str(ruta))
    registros = []
    for r in filas:
        _, senante, rep = parsear_muestra(r["muestra"])
        registros.append({**r, "real": int(r["real"]), "predicha": int(r["predicha"]),
                          "pliegue": int(senante), "senante": senante, "repeticion": rep})
    senantes = sorted({r["senante"] for r in registros})
    esperados = [f"{k:04d}" for k in range(N_PLIEGUES)]
    if senantes != esperados:
        raise ValueError(f"{ruta}: señantes {senantes}, se esperaban {esperados}")
    return registros


def cargar_aleatoria(ruta: Path) -> list[dict]:
    filas = _leer_json(ruta)
    _validar_filas(filas, str(ruta))
    return [{**r, "real": int(r["real"]), "predicha": int(r["predicha"])} for r in filas]


def sha256(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------
# Métricas desde conteos (sin sklearn; las pruebas las cotejan con sklearn)


def conteos(reales, predichas, n: int = N_CLASES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TP, FP, FN por clase a partir de la matriz de confusión (filas = real, columnas = predicha)."""
    reales = np.asarray(reales, dtype=np.int64)
    predichas = np.asarray(predichas, dtype=np.int64)
    if reales.shape != predichas.shape:
        raise ValueError("reales y predichas deben tener la misma longitud")
    if reales.size and (reales.min() < 0 or reales.max() >= n or predichas.min() < 0 or predichas.max() >= n):
        raise ValueError("etiqueta fuera de rango")
    mc = np.bincount(reales * n + predichas, minlength=n * n).reshape(n, n)
    tp = np.diag(mc).astype(np.float64)
    fp = mc.sum(axis=0) - tp
    fn = mc.sum(axis=1) - tp
    return tp, fp, fn


def _div0(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den con 0 donde den = 0 (zero_division = 0 de sklearn)."""
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def prf_por_clase(tp, fp, fn) -> dict:
    tp, fp, fn = (np.asarray(x, dtype=np.float64) for x in (tp, fp, fn))
    precision = _div0(tp, tp + fp)
    recall = _div0(tp, tp + fn)
    f1 = _div0(2.0 * tp, 2.0 * tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "soporte": tp + fn}


def metricas(reales, predichas, n: int = N_CLASES) -> dict:
    """Macro (sobre las n etiquetas), ponderado por soporte y micro; más top-1 y n."""
    tp, fp, fn = conteos(reales, predichas, n)
    pc = prf_por_clase(tp, fp, fn)
    soporte = pc["soporte"]
    total = float(soporte.sum())
    pesos = soporte / total if total > 0 else np.zeros_like(soporte)
    micro_p = _div0(np.array([tp.sum()]), np.array([tp.sum() + fp.sum()]))[0]
    micro_r = _div0(np.array([tp.sum()]), np.array([tp.sum() + fn.sum()]))[0]
    micro_f1 = _div0(np.array([2 * tp.sum()]), np.array([2 * tp.sum() + fp.sum() + fn.sum()]))[0]
    return {
        "n": int(total),
        "top1": float(tp.sum() / total) if total > 0 else 0.0,
        "precision_macro": float(pc["precision"].mean()),
        "recall_macro": float(pc["recall"].mean()),
        "f1_macro": float(pc["f1"].mean()),
        "precision_ponderada": float((pc["precision"] * pesos).sum()),
        "recall_ponderado": float((pc["recall"] * pesos).sum()),
        "f1_ponderado": float((pc["f1"] * pesos).sum()),
        "precision_micro": float(micro_p),
        "recall_micro": float(micro_r),
        "f1_micro": float(micro_f1),
        "clases_con_f1_0": int(np.sum(pc["f1"] == 0)),
        "clases_nunca_predichas": int(np.sum(tp + fp == 0)),
        "clases_sin_aciertos": int(np.sum(tp == 0)),
    }


def f1_macro(reales, predichas, n: int = N_CLASES) -> float:
    tp, fp, fn = conteos(reales, predichas, n)
    return float(prf_por_clase(tp, fp, fn)["f1"].mean())


# ---------------------------------------------------------------------------------------------------
# Lecturas


def _media_de(valores: list[float]) -> dict:
    v = np.asarray(valores, dtype=np.float64)
    return {"media": float(v.mean()), "de": float(v.std(ddof=1)) if v.size > 1 else None,
            "min": float(v.min()), "max": float(v.max()), "por_pliegue": [float(x) for x in v]}


def lectura_por_pliegue(registros: list[dict], n: int = N_CLASES) -> dict:
    """(a) y (e): métricas por pliegue, media ± DE entre pliegues y media ponderada por n_k."""
    pliegues = sorted({r["pliegue"] for r in registros})
    por_pliegue = []
    for k in pliegues:
        rs = [r for r in registros if r["pliegue"] == k]
        m = metricas([r["real"] for r in rs], [r["predicha"] for r in rs], n)
        m["pliegue"] = int(k)
        m["senante"] = rs[0].get("senante")
        por_pliegue.append(m)
    claves = ("top1", "precision_macro", "recall_macro", "f1_macro", "f1_ponderado", "f1_micro")
    resumen = {c: _media_de([m[c] for m in por_pliegue]) for c in claves}
    ns = np.asarray([m["n"] for m in por_pliegue], dtype=np.float64)
    for c in claves:
        vals = np.asarray([m[c] for m in por_pliegue])
        resumen[c]["media_ponderada_por_n"] = float((vals * ns).sum() / ns.sum())
    return {"n_pliegues": len(pliegues), "n_por_pliegue": [int(x) for x in ns], "pliegues": por_pliegue,
            "resumen": resumen}


def lectura_agregada(registros: list[dict], n: int = N_CLASES) -> dict:
    """(b), (c), (d) y (e): todo sobre las predicciones juntas."""
    return metricas([r["real"] for r in registros], [r["predicha"] for r in registros], n)


def descomposicion(registros: list[dict], n: int = N_CLASES) -> dict:
    """(g): por qué el agregado y la media por pliegue difieren, con identidad exacta."""
    pliegues = sorted({r["pliegue"] for r in registros})
    K = len(pliegues)
    tp = np.zeros((K, n)); fp = np.zeros((K, n)); fn = np.zeros((K, n))
    for i, k in enumerate(pliegues):
        rs = [r for r in registros if r["pliegue"] == k]
        tp[i], fp[i], fn[i] = conteos([r["real"] for r in rs], [r["predicha"] for r in rs], n)
    D = 2 * tp + fp + fn                                   # (K, n)
    f1_kc = _div0(2 * tp, D)                               # F1 de la seña c en el pliegue k
    f1_medio_c = f1_kc.mean(axis=0)                        # media por pliegue, por seña
    sumaD = D.sum(axis=0)
    f1_agregado_c = _div0(2 * tp.sum(axis=0), sumaD)       # F1 agregado por seña
    w = np.zeros_like(D)
    np.divide(D, sumaD[None, :], out=w, where=sumaD[None, :] > 0)
    contribucion_c = ((w - 1.0 / K) * f1_kc).sum(axis=0) / n   # aporte de cada seña a (agregado − media)
    media_pliegues = float(f1_kc.mean(axis=1).mean())
    agregado = float(f1_agregado_c.mean())
    diferencia = agregado - media_pliegues
    if abs(contribucion_c.sum() - diferencia) > 1e-9:
        raise AssertionError("la descomposición no cierra: "
                             f"{contribucion_c.sum()} frente a {diferencia}")
    ceros = tp == 0                                        # (K, n): pliegues con 0 aciertos de la seña
    con_cero = ceros.any(axis=0)
    # Efecto aparte: pliegues de tamaño desigual (198-200) en la media simple frente a la ponderada.
    ns = (tp + fn).sum(axis=1)
    f1_macro_k = f1_kc.mean(axis=1)
    media_ponderada = float((f1_macro_k * ns).sum() / ns.sum())
    # Convención 0/0 = 0: afecta a la precisión de las señas que un pliegue nunca predice; al F1 solo si
    # además la seña no tiene muestras en ese pliegue (soporte 0), lo que aquí se mide.
    soporte = tp + fn
    predichas = tp + fp
    nunca = predichas == 0
    prec_kc = _div0(tp, predichas)
    prec_media_convencion = float(prec_kc.mean(axis=1).mean())
    prec_media_sin_indefinidas = float(np.mean([prec_kc[i][~nunca[i]].mean() if (~nunca[i]).any() else 0.0
                                                for i in range(K)]))
    return {
        "media_por_pliegue": media_pliegues,
        "agregado": agregado,
        "diferencia_agregado_menos_media": diferencia,
        "identidad_cierra": True,
        "pares_pliegue_sena_con_0_aciertos": int(ceros.sum()),
        "pares_pliegue_sena_total": int(K * n),
        "senas_con_algun_pliegue_sin_aciertos": int(con_cero.sum()),
        "senas_sin_aciertos_en_el_agregado": int((tp.sum(axis=0) == 0).sum()),
        "aporte_senas_con_algun_pliegue_sin_aciertos": float(contribucion_c[con_cero].sum()),
        "aporte_senas_sin_pliegues_sin_aciertos": float(contribucion_c[~con_cero].sum()),
        "media_simple_vs_ponderada_por_n": {"media_simple": media_pliegues, "media_ponderada": media_ponderada,
                                            "diferencia": media_ponderada - media_pliegues},
        "soporte_minimo_pliegue_sena": int(soporte.min()),
        "pares_pliegue_sena_sin_muestras": int((soporte == 0).sum()),
        "pares_pliegue_sena_nunca_predichos": int(nunca.sum()),
        "precision_macro_media_pliegues": {"con_0_0_igual_a_0": prec_media_convencion,
                                           "sin_las_senas_nunca_predichas": prec_media_sin_indefinidas,
                                           "diferencia": prec_media_sin_indefinidas - prec_media_convencion},
        "por_sena": [{"sena": int(c), "f1_agregado": float(f1_agregado_c[c]), "f1_medio_pliegues": float(f1_medio_c[c]),
                      "pliegues_sin_aciertos": int(ceros[:, c].sum()), "aporte": float(contribucion_c[c]),
                      "f1_por_pliegue": [float(x) for x in f1_kc[:, c]],
                      "pesos_agregado": [float(x) for x in w[:, c]]} for c in range(n)],
        "explicacion": ("F1 agregado de una seña = Σ_k w_kc·F1_kc con w_kc = D_kc/Σ_k D_kc y D = 2TP+FP+FN; "
                        "la media por pliegue usa w = 1/K. Un pliegue con 0 aciertos tiene F1 = 0 y un D sin el término 2TP, "
                        "por lo que suele pesar menos de 1/K en el agregado (no siempre: depende de sus FP) y pesa "
                        "exactamente 1/K en la media."),
    }


# ---------------------------------------------------------------------------------------------------
# Bootstrap


def _indices_estratificados(reales: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Remuestreo con reposición DENTRO de cada clase: devuelve índices sobre `reales` que conservan el
    soporte de cada clase. Vectorizado: para cada posición se sortea un miembro de su propia clase."""
    orden = np.argsort(reales, kind="stable")
    clases_ordenadas = reales[orden]
    _, inicio, cuenta = np.unique(clases_ordenadas, return_index=True, return_counts=True)
    # Para cada posición ordenada i: su clase empieza en inicio[j] y tiene cuenta[j] miembros.
    j = np.repeat(np.arange(len(inicio)), cuenta)
    u = rng.random(len(reales))
    sorteo = inicio[j] + np.floor(u * cuenta[j]).astype(np.int64)
    return orden[sorteo]


def _f1_macro_rapido(reales: np.ndarray, predichas: np.ndarray, n: int) -> float:
    mc = np.bincount(reales * n + predichas, minlength=n * n).reshape(n, n)
    tp = np.diag(mc).astype(np.float64)
    fp = mc.sum(axis=0) - tp
    fn = mc.sum(axis=1) - tp
    return float(_div0(2 * tp, 2 * tp + fp + fn).mean())


def _percentiles(muestras: np.ndarray, alfa: float = ALFA) -> dict:
    return {"ic_inferior": float(np.percentile(muestras, 100 * alfa / 2)),
            "ic_superior": float(np.percentile(muestras, 100 * (1 - alfa / 2))),
            "media_bootstrap": float(muestras.mean()), "de_bootstrap": float(muestras.std(ddof=1))}


def bootstrap_agregado(registros: list[dict], n_boot: int = N_BOOT, semilla: int = SEMILLA,
                       n: int = N_CLASES) -> dict:
    """IC percentil del F1 macro agregado, remuestreando predicciones estratificado por seña."""
    rng = np.random.default_rng(semilla)
    reales = np.asarray([r["real"] for r in registros], dtype=np.int64)
    predichas = np.asarray([r["predicha"] for r in registros], dtype=np.int64)
    valores = np.empty(n_boot)
    for b in range(n_boot):
        idx = _indices_estratificados(reales, rng)
        valores[b] = _f1_macro_rapido(reales[idx], predichas[idx], n)
    out = _percentiles(valores)
    out.update({"n_boot": int(n_boot), "semilla": int(semilla), "estimacion_puntual": _f1_macro_rapido(reales, predichas, n),
                "metodo": "percentil; remuestreo de predicciones con reposición estratificado por seña"})
    return out


def bootstrap_media_pliegues(registros: list[dict], n_boot: int = N_BOOT, semilla: int = SEMILLA,
                             n: int = N_CLASES) -> dict:
    """IC del F1 macro medio entre pliegues: (i) remuestreo de pliegues y, dentro de cada pliegue elegido,
    de predicciones (estratificado por seña); (ii) solo pliegues; (iii) intervalo t clásico."""
    from scipy import stats
    rng = np.random.default_rng(semilla)
    pliegues = sorted({r["pliegue"] for r in registros})
    K = len(pliegues)
    datos = []
    for k in pliegues:
        rs = [r for r in registros if r["pliegue"] == k]
        datos.append((np.asarray([r["real"] for r in rs], dtype=np.int64),
                      np.asarray([r["predicha"] for r in rs], dtype=np.int64)))
    f1_k = np.asarray([_f1_macro_rapido(a, b, n) for a, b in datos])
    dos_niveles = np.empty(n_boot)
    solo_pliegues = np.empty(n_boot)
    for b in range(n_boot):
        elegidos = rng.integers(0, K, K)
        acum = 0.0
        for i in elegidos:
            reales, predichas = datos[i]
            idx = _indices_estratificados(reales, rng)
            acum += _f1_macro_rapido(reales[idx], predichas[idx], n)
        dos_niveles[b] = acum / K
        solo_pliegues[b] = f1_k[elegidos].mean()
    media = float(f1_k.mean()); de = float(f1_k.std(ddof=1)) if K > 1 else float("nan")
    t = float(stats.t.ppf(1 - ALFA / 2, K - 1)) if K > 1 else float("nan")
    return {
        "estimacion_puntual": media, "de_entre_pliegues": de, "n_pliegues": K,
        "pliegues_y_dentro": {**_percentiles(dos_niveles), "n_boot": int(n_boot), "semilla": int(semilla),
                              "metodo": "percentil; pliegues con reposición y, dentro de cada uno, predicciones estratificadas por seña"},
        "solo_pliegues": {**_percentiles(solo_pliegues), "n_boot": int(n_boot), "semilla": int(semilla),
                          "metodo": "percentil; solo pliegues con reposición (5 pliegues: 126 multiconjuntos posibles)"},
        "intervalo_t": {"ic_inferior": media - t * de / np.sqrt(K), "ic_superior": media + t * de / np.sqrt(K),
                        "t_critico": t, "grados_libertad": K - 1,
                        "metodo": "media ± t_{K−1; 0,975} · DE / √K (normalidad de la media entre pliegues)"},
    }


# ---------------------------------------------------------------------------------------------------
# Orden entre sistemas y McNemar


def mcnemar(aciertos_a, aciertos_b) -> dict:
    """McNemar pareado sobre las MISMAS muestras (misma convención que linea_base_clasica.mcnemar):
    exacto binomial si n10 + n01 < 25, si no chi-cuadrado con corrección de continuidad. Solo usa los
    aciertos top-1 por muestra: ningún F1 entra en el cálculo."""
    from scipy import stats
    a = np.asarray(aciertos_a, dtype=bool); b = np.asarray(aciertos_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("los dos sistemas deben evaluarse sobre las mismas muestras")
    n11 = int(np.sum(a & b)); n10 = int(np.sum(a & ~b)); n01 = int(np.sum(~a & b)); n00 = int(np.sum(~a & ~b))
    nd = n10 + n01
    if nd == 0:
        chi2, p_chi2 = 0.0, 1.0
    else:
        chi2 = (abs(n10 - n01) - 1.0) ** 2 / nd
        p_chi2 = float(stats.chi2.sf(chi2, 1))
    p_exacto = float(stats.binomtest(n10, nd, 0.5).pvalue) if nd else 1.0
    usar_exacto = nd < 25
    p = p_exacto if usar_exacto else p_chi2
    return {"n_muestras": int(a.size), "tabla": {"aciertan_ambos": n11, "solo_a": n10, "solo_b": n01, "fallan_ambos": n00},
            "n_discordantes": nd, "chi2_con_correccion_de_continuidad": float(chi2), "p_chi2": float(p_chi2),
            "p_exacto_binomial": p_exacto, "prueba_usada": "exacta binomial" if usar_exacto else "chi2 con corrección de continuidad",
            "p": float(p), "alfa": ALFA, "significativo": bool(p < ALFA),
            "aciertos_a": int(a.sum()), "aciertos_b": int(b.sum()),
            "entradas": "solo los aciertos top-1 por muestra; ningún F1 interviene"}


def alinear(reg_a: list[dict], reg_b: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Aciertos top-1 de dos sistemas sobre las mismas muestras, alineados por nombre."""
    da = {r["muestra"]: r for r in reg_a}
    db = {r["muestra"]: r for r in reg_b}
    if set(da) != set(db):
        raise ValueError("los dos sistemas no tienen las mismas muestras")
    nombres = sorted(da)
    a = np.asarray([da[m]["real"] == da[m]["predicha"] for m in nombres])
    b = np.asarray([db[m]["real"] == db[m]["predicha"] for m in nombres])
    return a, b


def orden_entre_sistemas(valores: dict[str, float], esperados=ORDENES_ESPERADOS) -> dict:
    """valores: clave del sistema → cifra bajo una lectura. Comprueba cada par esperado (mayor, menor)."""
    ranking = sorted(valores, key=lambda k: -valores[k])
    pares = []
    for mayor, menor in esperados:
        if mayor in valores and menor in valores:
            pares.append({"mayor": mayor, "menor": menor, "valor_mayor": valores[mayor], "valor_menor": valores[menor],
                          "se_cumple": bool(valores[mayor] > valores[menor])})
    return {"ranking": ranking, "pares": pares, "todos_se_cumplen": all(p["se_cumple"] for p in pares)}


# ---------------------------------------------------------------------------------------------------
# Controles contra lo guardado


def controles_red(loso_dir: Path, lectura: dict) -> dict:
    """Los F1 macro por pliegue y su media deben coincidir con resultados_loso.json (TOL)."""
    res = _leer_json(Path(loso_dir) / "resultados_loso.json")
    guardados = {int(p["pliegue"]): p for p in res["pliegues"]}
    filas = []
    for m in lectura["pliegues"]:
        g = guardados[m["pliegue"]]
        filas.append({"pliegue": m["pliegue"], "f1_calculado": m["f1_macro"], "f1_guardado": g["f1_macro"],
                      "coincide": abs(m["f1_macro"] - g["f1_macro"]) < TOL and abs(m["top1"] - g["top1"]) < TOL})
    media_g = res["agregado"]["f1_macro"]["media"]
    ok_media = abs(lectura["resumen"]["f1_macro"]["media"] - media_g) < TOL
    todo = all(f["coincide"] for f in filas) and ok_media
    if not todo:
        raise RuntimeError(f"las predicciones de {loso_dir} no reproducen resultados_loso.json: {filas}, media {media_g}")
    return {"por_pliegue": filas, "f1_media_guardada": media_g, "todo_coincide": todo}


def controles_linea_base(familia: str, lectura_p: dict, agregada: dict,
                         resultados: Path = BASE_DIR / "models" / "linea_base_4d_2026-09-15" / "resultados.json") -> dict:
    """Agregado y media por pliegue frente a resultados.json (loso.con_espejo)."""
    if not Path(resultados).exists():
        return {"disponible": False, "motivo": f"no existe {resultados}"}
    res = _leer_json(resultados)["loso"]["con_espejo"]
    if familia not in res["agregado_997"] or familia not in res["media_pliegues"]:
        return {"disponible": False, "motivo": f"la familia {familia!r} no está en {resultados}"}
    ag = res["agregado_997"][familia]; mp = res["media_pliegues"][familia]["f1_macro"]
    ok = (abs(ag["f1_macro"] - agregada["f1_macro"]) < TOL and abs(mp["media"] - lectura_p["resumen"]["f1_macro"]["media"]) < TOL
          and abs(ag["top1"] - agregada["top1"]) < TOL)
    if not ok:
        raise RuntimeError(f"la línea base {familia} no reproduce resultados.json: agregado {ag['f1_macro']} frente a "
                           f"{agregada['f1_macro']}; media {mp['media']} frente a {lectura_p['resumen']['f1_macro']['media']}")
    return {"disponible": True, "f1_agregado_guardado": ag["f1_macro"], "f1_media_guardada": mp["media"], "todo_coincide": True}


# ---------------------------------------------------------------------------------------------------
# Flujo


def analizar_sistema(cfg: dict, n_boot: int, semilla: int, base: Path = BASE_DIR) -> dict:
    out = {"clave": cfg["clave"], "nombre": cfg["nombre"], "fuentes": {}}
    registros = None
    if "loso_dir" in cfg:
        d = base / cfg["loso_dir"]
        registros = cargar_loso_pliegues(d)
        out["fuentes"]["loso"] = {"tipo": "pliegue_k/predicciones.json", "dir": cfg["loso_dir"],
                                  "sha256": {f"pliegue_{k}": sha256(d / f"pliegue_{k}" / "predicciones.json") for k in range(N_PLIEGUES)}}
    elif "loso_plano" in cfg:
        ruta = base / cfg["loso_plano"]
        registros = cargar_loso_plano(ruta)
        out["fuentes"]["loso"] = {"tipo": "archivo plano (pliegue = señante del nombre)", "archivo": cfg["loso_plano"],
                                  "sha256": sha256(ruta)}
    if registros is not None:
        lp = lectura_por_pliegue(registros)
        ag = lectura_agregada(registros)
        out["loso"] = {
            "n": len(registros),
            "por_pliegue": lp,
            "agregado": ag,
            "descomposicion": descomposicion(registros),
            "bootstrap": {"agregado_f1_macro": bootstrap_agregado(registros, n_boot, semilla),
                          "media_pliegues_f1_macro": bootstrap_media_pliegues(registros, n_boot, semilla)},
        }
        if "loso_dir" in cfg:
            out["loso"]["controles"] = controles_red(base / cfg["loso_dir"], lp)
        elif cfg["clave"].startswith("linea_base_"):
            familia = cfg["clave"].replace("linea_base_", "")
            out["loso"]["controles"] = controles_linea_base(
                familia, lp, ag, base / "models" / "linea_base_4d_2026-09-15" / "resultados.json")
        else:
            out["loso"]["controles"] = {"disponible": False, "motivo": "sin archivo de resultados guardado con el que cotejar"}
        out["_registros"] = registros
    if "aleatoria" in cfg and (base / cfg["aleatoria"]).exists():
        ruta = base / cfg["aleatoria"]
        ra = cargar_aleatoria(ruta)
        out["fuentes"]["aleatoria"] = {"archivo": cfg["aleatoria"], "sha256": sha256(ruta)}
        out["aleatoria"] = {"n": len(ra), "agregado": lectura_agregada(ra),
                            "lectura_unica": "un solo conjunto de prueba: solo existe el agregado; no hay media ni DE entre pliegues",
                            "bootstrap_f1_macro": bootstrap_agregado(ra, n_boot, semilla)}
    return out


def ejecutar(sistemas: list[dict], n_boot: int = N_BOOT, semilla: int = SEMILLA, base: Path = BASE_DIR) -> dict:
    inicio = dt.datetime.now()
    resultados = {
        "descripcion": "Variantes del F1 sobre predicciones guardadas (pieza 3, 2026-09-22). Solo reporte.",
        "fecha": inicio.isoformat(timespec="seconds"),
        "script": "src/f1_variantes.py",
        "parametros": {"n_boot": n_boot, "semilla": semilla, "alfa": ALFA, "n_clases": N_CLASES, "ddof": 1},
        "entorno": {"python": platform.python_version(), "numpy": np.__version__, "plataforma": platform.platform()},
        "sistemas": {},
    }
    registros = {}
    for cfg in sistemas:
        r = analizar_sistema(cfg, n_boot, semilla, base)
        if "_registros" in r:
            registros[cfg["clave"]] = r.pop("_registros")
        resultados["sistemas"][cfg["clave"]] = r
    # Orden bajo cada lectura
    lecturas = {
        "f1_macro_media_pliegues": lambda s: s["loso"]["por_pliegue"]["resumen"]["f1_macro"]["media"],
        "f1_macro_agregado": lambda s: s["loso"]["agregado"]["f1_macro"],
        "f1_ponderado_agregado": lambda s: s["loso"]["agregado"]["f1_ponderado"],
        "f1_micro_agregado_top1": lambda s: s["loso"]["agregado"]["f1_micro"],
        "top1_media_pliegues": lambda s: s["loso"]["por_pliegue"]["resumen"]["top1"]["media"],
        "precision_macro_media_pliegues": lambda s: s["loso"]["por_pliegue"]["resumen"]["precision_macro"]["media"],
        "precision_macro_agregada": lambda s: s["loso"]["agregado"]["precision_macro"],
        "recall_macro_media_pliegues": lambda s: s["loso"]["por_pliegue"]["resumen"]["recall_macro"]["media"],
        "recall_macro_agregado": lambda s: s["loso"]["agregado"]["recall_macro"],
    }
    con_loso = {k: s for k, s in resultados["sistemas"].items() if "loso" in s}
    resultados["orden"] = {nombre: orden_entre_sistemas({k: f(s) for k, s in con_loso.items()})
                           for nombre, f in lecturas.items()}
    resultados["orden"]["_mismo_ranking_en_todas_las_lecturas_de_f1"] = len({
        tuple(resultados["orden"][n]["ranking"]) for n in ("f1_macro_media_pliegues", "f1_macro_agregado",
                                                            "f1_ponderado_agregado", "f1_micro_agregado_top1")}) == 1
    # McNemar
    resultados["mcnemar"] = {}
    for a, b in COMPARACIONES_MCNEMAR:
        if a in registros and b in registros:
            aa, bb = alinear(registros[a], registros[b])
            resultados["mcnemar"][f"{a}_vs_{b}"] = mcnemar(aa, bb)
    resultados["mcnemar"]["_nota"] = ("McNemar compara, muestra a muestra, si cada sistema acertó la top-1. No lee ningún F1: "
                                      "cambiar la lectura del F1 (media o agregado) no altera ninguna tabla ni ningún p.")
    resultados["segundos"] = round((dt.datetime.now() - inicio).total_seconds(), 1)
    return resultados


# ---------------------------------------------------------------------------------------------------
# Informe


def _n(x, d=4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def escribir_md(res: dict, ruta: Path) -> None:
    L = []
    w = L.append
    w("# Variantes del F1 sobre las predicciones guardadas (pieza 3, 2026-09-22)\n")
    w(f"Fecha: {res['fecha']}. Script `{res['script']}`. Bootstrap: {res['parametros']['n_boot']} remuestreos, semilla "
      f"{res['parametros']['semilla']}, IC percentil al {100 * (1 - res['parametros']['alfa']):.0f} %. Macro sobre 50 señas, "
      "0/0 = 0, DE entre pliegues con ddof = 1. Todas las cifras están en `f1_variantes.json` con la clave que se indica.\n")
    w("## 1. LOSO: las lecturas del F1, por sistema\n")
    w("| Sistema | F1 macro, media ± DE entre pliegues [IC boot pliegues+dentro] | F1 macro agregado (997) [IC boot] | "
      "F1 ponderado agregado | F1 micro agregado (= top-1) | top-1 media ± DE | P macro media / agregada | R macro media / agregada |")
    w("|---|---|---|---|---|---|---|---|")
    for k, s in res["sistemas"].items():
        if "loso" not in s:
            continue
        r = s["loso"]["por_pliegue"]["resumen"]; a = s["loso"]["agregado"]; b = s["loso"]["bootstrap"]
        bm = b["media_pliegues_f1_macro"]["pliegues_y_dentro"]; ba = b["agregado_f1_macro"]
        w(f"| {s['nombre']} | {_n(r['f1_macro']['media'])} ± {_n(r['f1_macro']['de'])} [{_n(bm['ic_inferior'], 3)}; {_n(bm['ic_superior'], 3)}] | "
          f"{_n(a['f1_macro'])} [{_n(ba['ic_inferior'], 3)}; {_n(ba['ic_superior'], 3)}] | {_n(a['f1_ponderado'])} | {_n(a['f1_micro'])} | "
          f"{_n(r['top1']['media'])} ± {_n(r['top1']['de'])} | {_n(r['precision_macro']['media'])} / {_n(a['precision_macro'])} | "
          f"{_n(r['recall_macro']['media'])} / {_n(a['recall_macro'])} |")
    w("\nClaves: `sistemas.<clave>.loso.por_pliegue.resumen.<métrica>.{media,de}`, `sistemas.<clave>.loso.agregado.<métrica>`, "
      "`sistemas.<clave>.loso.bootstrap.*`.\n")
    w("## 2. Intervalos para el F1 macro del LOSO (tres formas de leer la incertidumbre)\n")
    w("| Sistema | media ± DE | IC t (5 pliegues) | IC boot solo pliegues | IC boot pliegues + dentro | agregado | IC boot agregado |")
    w("|---|---|---|---|---|---|---|")
    for k, s in res["sistemas"].items():
        if "loso" not in s:
            continue
        b = s["loso"]["bootstrap"]["media_pliegues_f1_macro"]; ba = s["loso"]["bootstrap"]["agregado_f1_macro"]
        w(f"| {s['nombre']} | {_n(b['estimacion_puntual'])} ± {_n(b['de_entre_pliegues'])} | [{_n(b['intervalo_t']['ic_inferior'], 3)}; {_n(b['intervalo_t']['ic_superior'], 3)}] | "
          f"[{_n(b['solo_pliegues']['ic_inferior'], 3)}; {_n(b['solo_pliegues']['ic_superior'], 3)}] | [{_n(b['pliegues_y_dentro']['ic_inferior'], 3)}; {_n(b['pliegues_y_dentro']['ic_superior'], 3)}] | "
          f"{_n(ba['estimacion_puntual'])} | [{_n(ba['ic_inferior'], 3)}; {_n(ba['ic_superior'], 3)}] |")
    anchos = []
    for k, s in res["sistemas"].items():
        if "loso" not in s:
            continue
        b = s["loso"]["bootstrap"]["media_pliegues_f1_macro"]["pliegues_y_dentro"]; ba = s["loso"]["bootstrap"]["agregado_f1_macro"]
        anchos.append((b["ic_superior"] - b["ic_inferior"], ba["ic_superior"] - ba["ic_inferior"]))
    n_mas_ancho = sum(1 for m, a in anchos if m > a)
    w(f"\nAnchura medida: el IC de la media entre pliegues (pliegues + dentro) es más ancho que el del agregado en {n_mas_ancho} de "
      f"{len(anchos)} sistemas. El remuestreo del agregado trata las predicciones como independientes; el de la media remuestrea "
      "también los señantes, que son la unidad de réplica del LOSO.\n")
    w("## 3. Por qué difieren la media por pliegue y el agregado\n")
    w("| Sistema | media pliegues | agregado | diferencia | pares (pliegue, seña) con 0 aciertos / 250 | señas con algún pliegue a 0 | "
      "aporte de esas señas | aporte del resto | media ponderada por n − media simple |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k, s in res["sistemas"].items():
        if "loso" not in s:
            continue
        d = s["loso"]["descomposicion"]
        w(f"| {s['nombre']} | {_n(d['media_por_pliegue'])} | {_n(d['agregado'])} | {_n(d['diferencia_agregado_menos_media'])} | "
          f"{d['pares_pliegue_sena_con_0_aciertos']} | {d['senas_con_algun_pliegue_sin_aciertos']} | "
          f"{_n(d['aporte_senas_con_algun_pliegue_sin_aciertos'])} | {_n(d['aporte_senas_sin_pliegues_sin_aciertos'])} | "
          f"{_n(d['media_simple_vs_ponderada_por_n']['diferencia'], 5)} |")
    w("\nIdentidad (comprobada numéricamente en cada sistema, clave `descomposicion.identidad_cierra`): el F1 agregado de una seña es "
      "Σ_k w_kc·F1_kc con w_kc = D_kc / Σ_k D_kc y D = 2·TP + FP + FN; la media por pliegue usa w = 1/5. Un pliegue en el que la seña "
      "no se acierta nunca tiene F1 = 0 y un D sin el término 2·TP, por lo que suele pesar menos de 1/5 en el agregado (no siempre: "
      "depende de sus falsos positivos) y pesa exactamente 1/5 en la media. Como D = 2·TP + FP + FN = (muestras reales de la seña) + (veces que se predijo la seña), el agregado pondera cada "
      "pliegue por cuántas veces aparece y se predice la seña allí.\n")
    filas_d = [s["loso"]["descomposicion"] for s in res["sistemas"].values() if "loso" in s]
    n_pos = sum(1 for d in filas_d if d["diferencia_agregado_menos_media"] > 0)
    n_dom = sum(1 for d in filas_d if abs(d["aporte_senas_con_algun_pliegue_sin_aciertos"]) > abs(d["aporte_senas_sin_pliegues_sin_aciertos"]))
    max_n = max((abs(d["media_simple_vs_ponderada_por_n"]["diferencia"]) for d in filas_d), default=0.0)
    w("| Sistema | soporte mínimo (pliegue, seña) | pares (pliegue, seña) sin muestras | pares nunca predichos / 250 | "
      "P macro media con 0/0 = 0 | P macro media sin las nunca predichas |")
    w("|---|---|---|---|---|---|")
    for s_ in res["sistemas"].values():
        if "loso" not in s_:
            continue
        d = s_["loso"]["descomposicion"]; pm = d["precision_macro_media_pliegues"]
        w(f"| {s_['nombre']} | {d['soporte_minimo_pliegue_sena']} | {d['pares_pliegue_sena_sin_muestras']} | "
          f"{d['pares_pliegue_sena_nunca_predichos']} | {_n(pm['con_0_0_igual_a_0'])} | {_n(pm['sin_las_senas_nunca_predichas'])} |")
    min_sop = min((s_["loso"]["descomposicion"]["soporte_minimo_pliegue_sena"] for s_ in res["sistemas"].values() if "loso" in s_), default=0)
    if min_sop >= 1:
        w(f"\nSoporte mínimo medido en un (pliegue, seña): {min_sop}. Con soporte ≥ 1, el F1 de una seña sin aciertos es 0 bajo cualquier convención (FN > 0), así que "
          "la convención 0/0 = 0 no mueve el F1; sí mueve la precisión macro por pliegue (tabla anterior).\n")
    else:
        w("\nHay pares (pliegue, seña) sin muestras: el F1 de esos pares depende de la convención 0/0.\n")
    w(f"Medido: el agregado supera a la media por pliegue en {n_pos} de {len(filas_d)} sistemas; el aporte de las señas con algún "
      f"pliegue a 0 aciertos es mayor (en valor absoluto) que el del resto en {n_dom} de {len(filas_d)}; el efecto del tamaño desigual "
      f"de los pliegues sobre la media es como mucho {_n(max_n, 5)} (última columna de la primera tabla de esta sección).\n")
    w("## 4. Orden entre sistemas bajo cada lectura\n")
    w("| Lectura | Ranking | Pares esperados que se cumplen |")
    w("|---|---|---|")
    for nombre, o in res["orden"].items():
        if nombre.startswith("_"):
            continue
        cumplidos = sum(p["se_cumple"] for p in o["pares"])
        w(f"| {nombre} | {' > '.join(o['ranking'])} | {cumplidos} de {len(o['pares'])} |")
    w(f"\nMismo ranking bajo las cuatro lecturas del F1: **{res['orden']['_mismo_ranking_en_todas_las_lecturas_de_f1']}** "
      "(clave `orden._mismo_ranking_en_todas_las_lecturas_de_f1`).\n")
    w("## 5. McNemar sobre aciertos top-1 (no usa F1)\n")
    w("| Comparación | solo acierta A | solo acierta B | discordantes | prueba | p | significativo (α = 0,05) |")
    w("|---|---|---|---|---|---|---|")
    for nombre, m in res["mcnemar"].items():
        if nombre.startswith("_"):
            continue
        w(f"| {nombre} | {m['tabla']['solo_a']} | {m['tabla']['solo_b']} | {m['n_discordantes']} | {m['prueba_usada']} | {m['p']:.2e} | {m['significativo']} |")
    w(f"\n{res['mcnemar']['_nota']}\n")
    w("## 6. Esquema aleatorio (un solo conjunto de prueba de 100): solo agregado\n")
    w("| Sistema | n | top-1 | F1 macro agregado [IC boot] | F1 ponderado | P macro | R macro |")
    w("|---|---|---|---|---|---|---|")
    for k, s in res["sistemas"].items():
        if "aleatoria" not in s:
            continue
        a = s["aleatoria"]["agregado"]; b = s["aleatoria"]["bootstrap_f1_macro"]
        w(f"| {s['nombre']} | {a['n']} | {_n(a['top1'], 3)} | {_n(a['f1_macro'])} [{_n(b['ic_inferior'], 3)}; {_n(b['ic_superior'], 3)}] | "
          f"{_n(a['f1_ponderado'])} | {_n(a['precision_macro'])} | {_n(a['recall_macro'])} |")
    w("\nRegla: con un solo pliegue no hay media ni DE entre pliegues; se reporta el agregado y se etiqueta como tal.\n")
    w("## 7. Pliegue a pliegue de la red «manos» (control frente a `resultados_loso.json`)\n")
    s = res["sistemas"].get("red_manos")
    if s and "loso" in s:
        w("| Pliegue | señante | n | top-1 | P macro | R macro | F1 macro | F1 ponderado | señas con F1 = 0 | coincide con lo guardado |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        ctl = {f["pliegue"]: f["coincide"] for f in s["loso"]["controles"].get("por_pliegue", [])}
        for m in s["loso"]["por_pliegue"]["pliegues"]:
            w(f"| {m['pliegue']} | {m['senante']} | {m['n']} | {_n(m['top1'], 3)} | {_n(m['precision_macro'])} | {_n(m['recall_macro'])} | "
              f"{_n(m['f1_macro'])} | {_n(m['f1_ponderado'])} | {m['clases_con_f1_0']} | {ctl.get(m['pliegue'], 'sin control')} |")
    w(f"\nTiempo de cálculo: {res['segundos']} s.\n")
    ruta.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    args = ap.parse_args(argv)
    sistemas = [c for c in SISTEMAS_POR_DEFECTO
                if any((BASE_DIR / c[k]).exists() for k in ("loso_dir", "loso_plano", "aleatoria") if k in c)]
    faltan = [c["clave"] for c in SISTEMAS_POR_DEFECTO if c not in sistemas]
    if faltan:
        print(f"[aviso] sistemas sin predicciones guardadas, se omiten: {faltan}")
    res = ejecutar(sistemas, args.n_boot, args.semilla)
    res["comando"] = " ".join([sys.executable, *sys.argv])
    res["sistemas_omitidos"] = faltan
    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "f1_variantes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    escribir_md(res, args.salida / "f1_variantes.md")
    for k, s in res["sistemas"].items():
        if "loso" in s:
            r = s["loso"]["por_pliegue"]["resumen"]["f1_macro"]; a = s["loso"]["agregado"]
            print(f"{k:28s} F1 macro media {r['media']:.4f} ± {r['de']:.4f} | agregado {a['f1_macro']:.4f} | "
                  f"ponderado {a['f1_ponderado']:.4f} | micro {a['f1_micro']:.4f}")
    print(f"mismo ranking en todas las lecturas de F1: {res['orden']['_mismo_ranking_en_todas_las_lecturas_de_f1']}")
    print(f"escrito: {args.salida / 'f1_variantes.json'} y {args.salida / 'f1_variantes.md'} ({res['segundos']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
