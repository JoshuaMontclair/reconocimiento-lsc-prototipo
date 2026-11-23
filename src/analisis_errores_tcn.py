"""
Análisis de errores del MODELO FINAL (D-14: TCN + imputación + 252 rasgos), actividad 4.d. SOLO REPORTE:
no decide, no reentrena y no ejecuta ningún modelo; lee predicciones ya guardadas.

Fuentes (todas ya en disco; ninguna se regenera):
  · TCN final, LOSO: models/confirmacion_tcn_2026-10-03/T1_imputacion_252/semilla_S/pliegue_k/predicciones.json
    (3 semillas 2026-2028 × 5 pliegues; el pliegue k deja fuera al señante k).
  · Red heredada (canónico R0: Conv1D + BiLSTM + atención, 714.291 parámetros), LOSO: el brazo A de
    models/experimento_mano_ausente_2026-10-02/A_mascara (el mismo R0 que reutilizan
    experimento_arquitecturas.py y experimento_confirmacion_tcn.py).
  · Partición aleatoria 797/100/100 (señantes vistos): models/tcn_imputacion/evaluacion_modelo_seleccionado
    (semilla 2030) y models/canonico_holistic_mascara_manos/evaluacion_modelo_seleccionado (semilla 2026).
  · Propiedades de cada video: data/processed_holistic_mascara_rostro (lengths.npy = fotogramas antes de
    remuestrear; X_padded.npy con la mano ausente en ceros, es decir, ANTES de imputar) y, como control,
    data/extracted/LSC50_holistic/METRICAS_FOTOGRAMA. Solo lectura.
  · Informe previo de la red heredada: models/analisis_errores_manos_2026-09-15/analisis.json.

Antes de reportar se comprueba que las predicciones reproducen lo guardado (top-1, top-3 y F1 de cada
corrida frente a resultados_confirmacion_tcn.json; McNemar del voto 163 / 56; top-1 y errores de la
partición aleatoria frente a metricas.json). Si algo no coincide, el script se detiene.

Unidades: «3 semillas» = las 2.991 predicciones (997 videos × 3 modelos) juntas; «voto» = una predicción por
video, mayoría de las 3 semillas (empate a tres: la de mayor probabilidad), la misma regla de
experimento_confirmacion_tcn.voto_semillas. Las funciones puras se reutilizan de src/analisis_errores.py.

Uso:
    python src/analisis_errores_tcn.py               # escribe docs/ANALISIS_ERRORES_4d_TCN.{md,json} y figuras
    python src/analisis_errores_tcn.py --sin-figuras --permutaciones 2000
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import analisis_errores as ae  # noqa: E402
from src.analisis_errores import _n, _p, _pct  # noqa: E402
from src.conversor_glosas import LEXICO, ConversorGlosas  # noqa: E402
from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402

N_CLASES = ae.N_CLASES
SEMILLAS = (2026, 2027, 2028)
SENANTES = ("0000", "0001", "0002", "0003", "0004")
MODELOS = ("heredada", "tcn")
ETIQUETA = {"heredada": "heredada", "tcn": "TCN final"}

LOSO_DIRS = {
    "tcn": BASE_DIR / "models" / "confirmacion_tcn_2026-10-03" / "T1_imputacion_252",
    "heredada": BASE_DIR / "models" / "experimento_mano_ausente_2026-10-02" / "A_mascara",
}
BRAZO_GUARDADO = {"tcn": "T1", "heredada": "R0"}
# TCN con «mascara» (T0): misma arquitectura que la final, sin imputación; aísla el efecto de la imputación
T0_DIR = BASE_DIR / "models" / "arquitecturas_literatura_2026-10-02" / "T_tcn"
RESULTADOS_CONFIRMACION = BASE_DIR / "models" / "confirmacion_tcn_2026-10-03" / "resultados_confirmacion_tcn.json"
ALEATORIA_DIRS = {
    "tcn": BASE_DIR / "models" / "tcn_imputacion" / "evaluacion_modelo_seleccionado",
    "heredada": BASE_DIR / "models" / "canonico_holistic_mascara_manos" / "evaluacion_modelo_seleccionado",
}
ANALISIS_PREVIO = BASE_DIR / "models" / "analisis_errores_manos_2026-09-15" / "analisis.json"
LOSO_PREVIO = BASE_DIR / "models" / "loso_2026-09-15_holistic_mascara_manos"
PROCESSED = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
METRICAS_FOTOGRAMA = BASE_DIR / "data" / "extracted" / "LSC50_holistic" / "METRICAS_FOTOGRAMA"
SALIDA_MD = BASE_DIR / "docs" / "ANALISIS_ERRORES_4d_TCN.md"
SALIDA_JSON = BASE_DIR / "docs" / "ANALISIS_ERRORES_4d_TCN.json"
DIR_FIGURAS = BASE_DIR / "docs" / "figuras_2026-11-08"
PREFIJO_FIG = "4d_"

# Proxy de «manos usadas»: cociente entre el recorrido de la muñeca del bloque que menos se mueve y el del
# que más se mueve. Fijado en 0,5 mirando SOLO la distribución del cociente (bimodal, valle entre 0,4 y 0,6;
# las medianas por seña van de 0,17 a 0,34 o de 0,64 a 0,94), sin mirar aciertos ni errores.
UMBRAL_MANOS = 0.5
TRAMOS_CONFIANZA = (0.0, 0.25, 0.5, 0.75, 1.0)
TRAMOS_MANO_AUSENTE = (0.0, 0.05, 0.15)   # 0 % | (0, 5 %] | (5, 15 %] | > 15 %
CONTROLES_REGISTRADOS = {   # cifras del encargo / N-63 / N-65, solo para verificar
    "tcn": {"top1": 0.677, "top3": 0.847, "f1": 0.647, "senante_0000": 0.488, "aleatoria_top1": 0.94,
            "aleatoria_errores": 6},
    "heredada": {"top1": 0.589, "top3": 0.764, "f1": 0.544, "senante_0000": 0.452, "aleatoria_top1": 0.88,
                 "aleatoria_errores": 12},
    "mcnemar_voto": {"solo_tcn": 163, "solo_heredada": 56},
}
TOL = 1e-9


# ---------------------------------------------------------------------------------------------------
# Utilidades puras (probadas en tests/test_analisis_errores_tcn.py)
# ---------------------------------------------------------------------------------------------------

def voto_semillas(preds_por_semilla: list[dict]) -> dict:
    """Voto de mayoría entre semillas; preds: [{muestra: (real, predicha, prob)}]. Si las predicciones son
    todas distintas, gana la de mayor probabilidad. Copia literal de la regla de
    experimento_confirmacion_tcn.voto_semillas (no se importa: ese módulo carga TensorFlow)."""
    claves = sorted(preds_por_semilla[0])
    out = {}
    for c in claves:
        votos = [p[c] for p in preds_por_semilla]
        clases = [v[1] for v in votos]
        cuenta = {cl: clases.count(cl) for cl in set(clases)}
        mx = max(cuenta.values())
        if mx > 1:
            elegida = [cl for cl in clases if cuenta[cl] == mx][0]
        else:
            elegida = max(votos, key=lambda v: v[2])[1]
        out[c] = (votos[0][0], int(elegida))
    return out


def tabla_transicion(ok_a, ok_b) -> dict:
    """Tabla 2×2 por video entre el modelo a (TCN) y el b (heredada) y McNemar exacto."""
    a = np.asarray(ok_a, dtype=bool)
    b = np.asarray(ok_b, dtype=bool)
    solo_a, solo_b = int((a & ~b).sum()), int((~a & b).sum())
    return {"n": int(len(a)), "ambos_aciertan": int((a & b).sum()), "solo_tcn": solo_a, "solo_heredada": solo_b,
            "ambos_fallan": int((~a & ~b).sum()), "neto_tcn": solo_a - solo_b,
            "p_mcnemar_exacto": ae.mcnemar_exacto(solo_a, solo_b)}


def parejas_no_ordenadas(reales, predichas) -> list[dict]:
    """Errores agrupados por pareja {a, b} (a < b) con el conteo de cada sentido; orden: total descendente,
    luego a, b."""
    cont = Counter((int(r), int(p)) for r, p in zip(reales, predichas) if int(r) != int(p))
    parejas = {}
    for (r, p), c in cont.items():
        a, b = min(r, p), max(r, p)
        d = parejas.setdefault((a, b), {"a": a, "b": b, "a_a_b": 0, "b_a_a": 0})
        d["a_a_b" if r == a else "b_a_a"] += c
    salida = [dict(d, total=d["a_a_b"] + d["b_a_a"]) for d in parejas.values()]
    salida.sort(key=lambda d: (-d["total"], d["a"], d["b"]))
    return salida


def consistencia_semillas(preds_por_semilla: list[dict]) -> dict:
    """Por video: cuántas semillas fallan (0..k) y, si todas fallan, cuántas clases erróneas distintas dan.
    preds: [{muestra: (real, predicha, ...)}]."""
    k = len(preds_por_semilla)
    por_muestra = {}
    for c in sorted(preds_por_semilla[0]):
        real = preds_por_semilla[0][c][0]
        predichas = [p[c][1] for p in preds_por_semilla]
        fallos = sum(1 for x in predichas if x != real)
        por_muestra[c] = {"fallos": fallos, "predichas": predichas,
                          "clases_erroneas_distintas": len({x for x in predichas if x != real})}
    dist = Counter(v["fallos"] for v in por_muestra.values())
    todas = [v for v in por_muestra.values() if v["fallos"] == k]
    return {"semillas": k, "distribucion_fallos": {str(i): int(dist.get(i, 0)) for i in range(k + 1)},
            "fallan_todas_misma_clase": sum(1 for v in todas if v["clases_erroneas_distintas"] == 1),
            "por_muestra": por_muestra}


def manos_en_movimiento(recorrido_a: float, recorrido_b: float, umbral: float = UMBRAL_MANOS):
    """Proxy medido: «dos» si el bloque de mano que menos se mueve recorre al menos `umbral` veces lo que
    recorre el que más se mueve; si no, «una». (cociente, etiqueta); (None, None) si ninguna se mueve."""
    mx, mn = max(recorrido_a, recorrido_b), min(recorrido_a, recorrido_b)
    if mx <= 0:
        return None, None
    r = mn / mx
    return float(r), ("dos" if r >= umbral else "una")


def tramo_de(valor: float, bordes) -> int:
    """Índice del tramo: bordes (0, b1, b2, ...) → tramo 0 = valor ≤ 0 (exactamente 0 si bordes[0] = 0),
    tramo i = (b_{i-1}, b_i], último = > b_último. Para la confianza se usa `tramo_confianza`."""
    if valor <= bordes[0]:
        return 0
    for i in range(1, len(bordes)):
        if valor <= bordes[i]:
            return i
    return len(bordes)


def tramo_confianza(conf: float, bordes=TRAMOS_CONFIANZA) -> int:
    """[b0, b1), [b1, b2), ..., [b_{n-1}, b_n] (el último incluye 1)."""
    for i in range(1, len(bordes)):
        if conf < bordes[i]:
            return i - 1
    return len(bordes) - 2


def exactitud_por_grupo(grupos, ok) -> dict:
    ok = np.asarray(ok, dtype=bool)
    salida = {}
    for g in sorted(set(grupos), key=str):
        sel = np.array([x == g for x in grupos])
        salida[str(g)] = {"n": int(sel.sum()), "exactitud": float(ok[sel].mean()) if sel.any() else None}
    return salida


def categoria_conversor(sena: int) -> str:
    return LEXICO[SIGN_VOCABULARY[sena]]["cat"]


def clasificar_error_conversor(real: int, predicha: int, conv: ConversorGlosas) -> dict:
    """Cómo ve el conversor (modo inventario, glosa sola) un error real → predicha. Solo descriptivo."""
    gr, gp = SIGN_VOCABULARY[real], SIGN_VOCABULARY[predicha]
    lr, lp = LEXICO[gr], LEXICO[gp]
    er, ep = conv.translate([gr]), conv.translate([gp])
    misma = lr["cat"] == lp["cat"]
    solo_genero = (misma and lr.get("genero") is not None and lp.get("genero") is not None
                   and lr.get("genero") != lp.get("genero") and lr.get("numero") == lp.get("numero")
                   and lr.get("pn") == lp.get("pn"))
    return {"real": real, "predicha": predicha, "categoria_real": lr["cat"], "categoria_predicha": lp["cat"],
            "misma_categoria": misma, "misma_categoria_heredada": ae.categoria_de(real) == ae.categoria_de(predicha),
            "cambia_modalidad": er.endswith("?") != ep.endswith("?"), "misma_categoria_distinto_genero": solo_genero,
            "enunciado_real": er, "enunciado_predicho": ep}


def esperado_misma_categoria(reales, cat_de) -> float:
    """Proporción de errores dentro de la misma categoría si cada error fuera a una de las otras 49 señas al azar."""
    cats = [cat_de(i) for i in range(N_CLASES)]
    v = [(sum(1 for j in range(N_CLASES) if j != r and cats[j] == cats[r]) / (N_CLASES - 1)) for r in reales]
    return float(np.mean(v)) if v else None


def s(i: int) -> str:
    return f"{SIGN_VOCABULARY[i]} ({i})"


# ---------------------------------------------------------------------------------------------------
# Carga y controles
# ---------------------------------------------------------------------------------------------------

def cargar_loso_semillas(d: Path, semillas=SEMILLAS) -> dict[int, dict[str, dict]]:
    """{semilla: {muestra: registro}}; comprueba seña del nombre = real y señante del nombre = pliegue."""
    salida = {}
    for sem in semillas:
        regs = {}
        for k in range(5):
            ruta = d / f"semilla_{sem}" / f"pliegue_{k}" / "predicciones.json"
            for r in json.loads(ruta.read_text(encoding="utf-8")):
                sena, senante, rep = ae.parsear_muestra(r["muestra"])
                if sena != int(r["real"]) or senante != f"{k:04d}":
                    raise ValueError(f"{ruta}: {r['muestra']} no corresponde a real={r['real']} / pliegue {k}")
                regs[r["muestra"]] = {**r, "pliegue": k, "senante": senante, "repeticion": rep, "semilla": sem}
        if len(regs) != 997:
            raise ValueError(f"{d} semilla {sem}: {len(regs)} predicciones, se esperaban 997")
        salida[sem] = regs
    return salida


def cargar_aleatoria(d: Path) -> tuple[list[dict], dict]:
    preds = json.loads((d / "predicciones.json").read_text(encoding="utf-8"))
    met = json.loads((d / "metricas.json").read_text(encoding="utf-8"))
    for r in preds:
        sena, senante, rep = ae.parsear_muestra(r["muestra"])
        r.update(senante=senante, repeticion=rep)
    return preds, met


def _por_pliegue(regs: dict[str, dict], k: int) -> list[dict]:
    return [r for r in regs.values() if r["pliegue"] == k]


def metricas_corrida(rs: list[dict]) -> dict:
    reales = [r["real"] for r in rs]
    return {"top1": float(np.mean([r["real"] == r["predicha"] for r in rs])),
            "top3": ae.top_k(reales, [r["top3"] for r in rs], 3),
            "f1": ae.f1_macro(reales, [r["predicha"] for r in rs])}


def controles(loso: dict, aleatoria: dict, guardado: dict, voto: dict) -> dict:
    """Reproduce lo guardado; RuntimeError si algo no coincide."""
    ctl = {"corridas": [], "resumen": {}, "todo_coincide": True}
    corridas_g = {(c["brazo"], c["semilla"], c["pliegue"]): c for c in guardado["corridas"]}
    for m in MODELOS:
        filas = []
        for sem in SEMILLAS:
            for k in range(5):
                calc = metricas_corrida(_por_pliegue(loso[m][sem], k))
                g = corridas_g[(BRAZO_GUARDADO[m], sem, k)]
                coincide = (abs(calc["top1"] - g["test_top1"]) < 1e-6 and abs(calc["top3"] - g["test_top3"]) < 1e-6
                            and abs(calc["f1"] - g["test_f1_macro"]) < 1e-6)
                filas.append({"modelo": m, "semilla": sem, "pliegue": k, **calc, "top1_guardado": g["test_top1"],
                              "top3_guardado": g["test_top3"], "f1_guardado": g["test_f1_macro"], "coincide": coincide})
                ctl["todo_coincide"] &= coincide
        reg = CONTROLES_REGISTRADOS[m]
        res = {"top1": float(np.mean([f["top1"] for f in filas])), "top3": float(np.mean([f["top3"] for f in filas])),
               "f1": float(np.mean([f["f1"] for f in filas])),
               "senante_0000": float(np.mean([f["top1"] for f in filas if f["pliegue"] == 0]))}
        res["coincide_con_registro_3_decimales"] = all(round(res[c], 3) == reg[c] for c in
                                                       ("top1", "top3", "f1", "senante_0000"))
        preds, met = aleatoria[m]
        top1_a = float(np.mean([r["real"] == r["predicha"] for r in preds]))
        res["aleatoria"] = {"top1": top1_a, "errores": sum(r["real"] != r["predicha"] for r in preds),
                            "top1_guardado": met["top1"], "errores_guardados": met["n_errores"],
                            "sistema": met.get("sistema")}
        res["aleatoria"]["coincide"] = (abs(top1_a - met["top1"]) < 1e-9 and res["aleatoria"]["errores"] ==
                                        met["n_errores"] == reg["aleatoria_errores"])
        ctl["todo_coincide"] &= res["coincide_con_registro_3_decimales"] and res["aleatoria"]["coincide"]
        ctl["resumen"][m] = res
        ctl["corridas"].extend(filas)
    t = tabla_transicion(*(np.array([voto[m][c][0] == voto[m][c][1] for c in sorted(voto[m])]) for m in ("tcn", "heredada")))
    g = guardado["mcnemar"]["T1_vs_R0"]["voto_3_semillas"]
    ctl["mcnemar_voto"] = {"calculado": t, "guardado": {"solo_T1": g["solo_brazo_acierta"], "solo_R0": g["solo_R0_acierta"],
                                                        "p": g["p_exacto_bilateral"]}}
    ctl["mcnemar_voto"]["coincide"] = (t["solo_tcn"] == g["solo_brazo_acierta"] == CONTROLES_REGISTRADOS["mcnemar_voto"]["solo_tcn"]
                                       and t["solo_heredada"] == g["solo_R0_acierta"] == CONTROLES_REGISTRADOS["mcnemar_voto"]["solo_heredada"]
                                       and abs(t["p_mcnemar_exacto"] - g["p_exacto_bilateral"]) < 1e-15 + 1e-6 * g["p_exacto_bilateral"])
    ctl["todo_coincide"] &= ctl["mcnemar_voto"]["coincide"]
    return ctl


def control_semilla_2026_vs_informe_previo(loso_her: dict[str, dict], loso_previo: Path) -> dict:
    """¿Son las predicciones de la heredada (semilla 2026) las mismas del LOSO del informe previo?"""
    if not loso_previo.exists():
        return {"disponible": False}
    previas = {}
    for k in range(5):
        for r in json.loads((loso_previo / f"pliegue_{k}" / "predicciones.json").read_text(encoding="utf-8")):
            previas[r["muestra"]] = r["predicha"]
    iguales = sum(1 for c, r in loso_her.items() if previas.get(c) == r["predicha"])
    return {"disponible": True, "n": len(loso_her), "predicciones_iguales": iguales,
            "identicas": iguales == len(loso_her) == len(previas)}


# ---------------------------------------------------------------------------------------------------
# Propiedades medibles de cada video
# ---------------------------------------------------------------------------------------------------

def propiedades_videos(processed: Path, dir_metricas: Path, muestras: list[str]) -> tuple[dict, dict]:
    """Longitud (fotogramas antes de remuestrear), proporción de fotogramas sin alguna mano y sin la mano
    que más se mueve (bloques en ceros de X_padded, antes de imputar) y proxy de manos en movimiento."""
    longitudes = ae._longitudes(processed)
    mov = ae._movimiento_munecas(processed, muestras, longitudes)
    det = ae._deteccion_por_muestra(dir_metricas, muestras)
    props, ctl = {}, {"metricas_fotograma_disponible": det is not None, "coinciden_bloques_vs_metricas": None}
    discrepancias = 0
    for c in muestras:
        n = longitudes[c]
        mv = mov[c]
        cociente, manos = manos_en_movimiento(mv["recorrido_izq"], mv["recorrido_der"])
        activa = "izq" if mv["recorrido_izq"] >= mv["recorrido_der"] else "der"
        # fotogramas sin alguna mano: hace falta el detalle por fotograma; se toma de METRICAS si existe
        if det is not None:
            sin_alguna = det[c]["sin_alguna"]
            if det[c]["sin_izq"] != mv["bloques_en_ceros_izq"] or det[c]["sin_der"] != mv["bloques_en_ceros_der"] \
                    or det[c]["fotogramas"] != n:
                discrepancias += 1
        else:
            sin_alguna = None
        props[c] = {"longitud": n, "prop_sin_alguna_mano": sin_alguna / n if sin_alguna is not None else None,
                    "prop_sin_mano_activa": mv[f"bloques_en_ceros_{activa}"] / n,
                    "bloque_activo": activa, "cociente_recorrido": cociente, "manos": manos}
    if det is not None:
        ctl["coinciden_bloques_vs_metricas"] = discrepancias == 0
        ctl["discrepancias"] = discrepancias
    return props, ctl


def manos_por_sena(props: dict) -> dict[int, str]:
    """Etiqueta por seña = mayoría de sus videos (empate → «dos»)."""
    por = defaultdict(list)
    for c, p in props.items():
        if p["manos"] is not None:
            por[int(c[:4])].append(p["manos"])
    return {k: ("dos" if v.count("dos") >= v.count("una") else "una") for k, v in sorted(por.items())}


# ---------------------------------------------------------------------------------------------------
# Secciones
# ---------------------------------------------------------------------------------------------------

def _juntar(loso_m: dict) -> list[dict]:
    return [r for sem in SEMILLAS for r in loso_m[sem].values()]


def _recall_precision(reales, predichas) -> dict:
    return ae.metricas_por_clase(reales, predichas)


def seccion_global(loso, voto) -> dict:
    out = {}
    for m in MODELOS:
        todas = _juntar(loso[m])
        vm = voto[m]
        out[m] = {"n_3_semillas": len(todas),
                  "exactitud_3_semillas": float(np.mean([r["real"] == r["predicha"] for r in todas])),
                  "top3_3_semillas": ae.top_k([r["real"] for r in todas], [r["top3"] for r in todas], 3),
                  "errores_3_semillas": sum(r["real"] != r["predicha"] for r in todas),
                  "exactitud_voto": float(np.mean([v[0] == v[1] for v in vm.values()])),
                  "errores_voto": sum(v[0] != v[1] for v in vm.values()),
                  "f1_voto": ae.f1_macro([v[0] for v in vm.values()], [v[1] for v in vm.values()])}
    return out


def seccion_pares(loso, voto, n_top: int = 15) -> dict:
    out = {}
    for m in MODELOS:
        todas = _juntar(loso[m])
        pares3 = ae.pares_confusion([r["real"] for r in todas], [r["predicha"] for r in todas])
        parejas3 = parejas_no_ordenadas([r["real"] for r in todas], [r["predicha"] for r in todas])
        vm = voto[m]
        claves = sorted(vm)
        pares_v = ae.pares_confusion([vm[c][0] for c in claves], [vm[c][1] for c in claves])
        parejas_v = parejas_no_ordenadas([vm[c][0] for c in claves], [vm[c][1] for c in claves])
        # señantes en que ocurre cada par (voto)
        donde = defaultdict(set)
        for c in claves:
            if vm[c][0] != vm[c][1]:
                donde[(vm[c][0], vm[c][1])].add(c[5:9])
        for p in pares_v:
            p["senantes"] = sorted(donde[(p["real"], p["predicha"])])
        # semillas en que aparece cada par
        por_sem = {sem: Counter((r["real"], r["predicha"]) for r in loso[m][sem].values() if r["real"] != r["predicha"])
                   for sem in SEMILLAS}
        for p in pares3:
            p["por_semilla"] = [por_sem[sem].get((p["real"], p["predicha"]), 0) for sem in SEMILLAS]
        n_err = sum(p["conteo"] for p in pares3)
        top10 = sum(p["total"] for p in parejas3[:10])
        out[m] = {"pares_3_semillas": pares3[:n_top], "parejas_3_semillas": parejas3[:n_top],
                  "pares_voto": pares_v[:n_top], "parejas_voto": parejas_v[:n_top],
                  "n_pares_distintos_3_semillas": len(pares3), "n_pares_distintos_voto": len(pares_v),
                  "errores_3_semillas": n_err, "proporcion_errores_en_10_parejas_principales": top10 / n_err,
                  "_todas_parejas_3_semillas": parejas3}
    # coincidencia de las 10 parejas principales
    top = {m: [(p["a"], p["b"]) for p in out[m]["parejas_3_semillas"][:10]] for m in MODELOS}
    comunes = [x for x in top["tcn"] if x in top["heredada"]]
    out["parejas_top10_comunes"] = [list(x) for x in comunes]
    return out


def seccion_por_clase(loso, voto) -> dict:
    out = {}
    for m in MODELOS:
        todas = _juntar(loso[m])
        mc = _recall_precision([r["real"] for r in todas], [r["predicha"] for r in todas])
        vm = voto[m]
        claves = sorted(vm)
        mv = _recall_precision([vm[c][0] for c in claves], [vm[c][1] for c in claves])
        por_sem = {sem: _recall_precision([r["real"] for r in loso[m][sem].values()],
                                          [r["predicha"] for r in loso[m][sem].values()])["recall"] for sem in SEMILLAS}
        filas = []
        for i in range(N_CLASES):
            filas.append({"id": i, "nombre": SIGN_VOCABULARY[i], "categoria": ae.categoria_de(i),
                          "categoria_conversor": categoria_conversor(i),
                          "recall": mc["recall"][i], "precision": mc["precision"][i], "f1": mc["f1"][i],
                          "soporte": mc["soporte"][i], "veces_predicha": mc["n_predichas"][i],
                          "recall_por_semilla": [por_sem[sem][i] for sem in SEMILLAS],
                          "recall_voto": mv["recall"][i], "precision_voto": mv["precision"][i]})
        out[m] = filas
    dif = []
    for i in range(N_CLASES):
        h, t = out["heredada"][i], out["tcn"][i]
        dif.append({"id": i, "nombre": SIGN_VOCABULARY[i], "recall_heredada": h["recall"], "recall_tcn": t["recall"],
                    "diferencia": t["recall"] - h["recall"]})
    out["diferencias"] = sorted(dif, key=lambda d: (-d["diferencia"], d["id"]))
    out["ganan"] = sum(1 for d in dif if d["diferencia"] > TOL)
    out["pierden"] = sum(1 for d in dif if d["diferencia"] < -TOL)
    out["recall_0_tcn_voto"] = [i for i in range(N_CLASES) if out["tcn"][i]["recall_voto"] == 0]
    out["recall_0_heredada_voto"] = [i for i in range(N_CLASES) if out["heredada"][i]["recall_voto"] == 0]
    return out


def seccion_por_senante(loso, voto) -> dict:
    out = {}
    for m in MODELOS:
        out[m] = {}
        for k, sen in enumerate(SENANTES):
            filas = [metricas_corrida(_por_pliegue(loso[m][sem], k)) for sem in SEMILLAS]
            rs = [r for sem in SEMILLAS for r in _por_pliegue(loso[m][sem], k)]
            vm = {c: v for c, v in voto[m].items() if c[5:9] == sen}
            claves = sorted(vm)
            mv = _recall_precision([vm[c][0] for c in claves], [vm[c][1] for c in claves])
            pares = ae.pares_confusion([r["real"] for r in rs], [r["predicha"] for r in rs])
            conf = np.array([r["prob_predicha"] for r in rs])
            ok = np.array([r["real"] == r["predicha"] for r in rs])
            out[m][sen] = {
                "top1": float(np.mean([f["top1"] for f in filas])), "top3": float(np.mean([f["top3"] for f in filas])),
                "f1": float(np.mean([f["f1"] for f in filas])),
                "top1_por_semilla": [f["top1"] for f in filas],
                "exactitud_voto": float(np.mean([vm[c][0] == vm[c][1] for c in claves])),
                "senas_recall_0_voto": [i for i in range(N_CLASES) if mv["recall"][i] == 0],
                "pares_principales_3_semillas": [(p["real"], p["predicha"], p["conteo"]) for p in pares[:6]],
                "confianza_media_aciertos": float(conf[ok].mean()), "confianza_media_errores": float(conf[~ok].mean()),
            }
    return out


def seccion_transicion(voto, n_top: int = 10) -> dict:
    claves = sorted(voto["tcn"])
    ok_t = np.array([voto["tcn"][c][0] == voto["tcn"][c][1] for c in claves])
    ok_h = np.array([voto["heredada"][c][0] == voto["heredada"][c][1] for c in claves])
    out = {"global": tabla_transicion(ok_t, ok_h), "por_senante": {}, "por_sena": [], "por_categoria": {}}
    sen = np.array([c[5:9] for c in claves])
    sena = np.array([int(c[:4]) for c in claves])
    for x in SENANTES:
        sel = sen == x
        out["por_senante"][x] = tabla_transicion(ok_t[sel], ok_h[sel])
    for i in range(N_CLASES):
        sel = sena == i
        out["por_sena"].append({"id": i, "nombre": SIGN_VOCABULARY[i], **tabla_transicion(ok_t[sel], ok_h[sel])})
    for cat in ae.ORDEN_CATEGORIAS:
        sel = np.array([ae.categoria_de(i) == cat for i in sena])
        out["por_categoria"][cat] = tabla_transicion(ok_t[sel], ok_h[sel])
    orden = sorted(out["por_sena"], key=lambda d: (-d["neto_tcn"], d["id"]))
    out["senas_que_mas_ganan"] = orden[:n_top]
    out["senas_que_mas_pierden"] = sorted([d for d in orden if d["neto_tcn"] < 0], key=lambda d: (d["neto_tcn"], d["id"]))[:n_top]
    out["senas_netas"] = {"ganan": sum(1 for d in orden if d["neto_tcn"] > 0),
                          "pierden": sum(1 for d in orden if d["neto_tcn"] < 0),
                          "igual": sum(1 for d in orden if d["neto_tcn"] == 0)}
    # cuando ambos fallan: ¿la misma clase errónea?
    ambos = [c for c, a, b in zip(claves, ok_t, ok_h) if not a and not b]
    misma = sum(1 for c in ambos if voto["tcn"][c][1] == voto["heredada"][c][1])
    out["ambos_fallan_misma_prediccion"] = misma
    # errores que la TCN introduce: ¿a qué pares van?
    intro = Counter((voto["tcn"][c][0], voto["tcn"][c][1]) for c, a, b in zip(claves, ok_t, ok_h) if not a and b)
    corr = Counter((voto["heredada"][c][0], voto["heredada"][c][1]) for c, a, b in zip(claves, ok_t, ok_h) if a and not b)
    out["errores_introducidos_pares"] = [(r, p, n) for (r, p), n in intro.most_common(8)]
    out["errores_corregidos_pares_heredada"] = [(r, p, n) for (r, p), n in corr.most_common(8)]
    return out


def seccion_propiedades(loso, voto, props, n_perm: int, semilla: int) -> dict:
    claves = sorted(voto["tcn"])
    out = {"definiciones": {
        "longitud": "fotogramas del video antes de remuestrear a 30 (lengths.npy)",
        "prop_sin_alguna_mano": "fotogramas en que falta alguna de las dos manos / fotogramas (METRICAS_FOTOGRAMA; "
                                "es lo que la regla «imputacion» rellena; la heredada lo deja en ceros)",
        "prop_sin_mano_activa": "fotogramas sin el bloque de mano que más recorre / fotogramas (X_padded en ceros)",
        "manos": f"proxy: «dos» si el recorrido de la muñeca que menos se mueve es ≥ {_n(UMBRAL_MANOS, 1)} veces el de la "
                 "que más se mueve; no es una anotación lingüística"},
        "modelos": {}}
    msena = manos_por_sena(props)
    out["manos_por_sena"] = {"dos": [i for i, v in msena.items() if v == "dos"],
                             "una": [i for i, v in msena.items() if v == "una"]}
    out["videos_por_manos"] = dict(Counter(props[c]["manos"] for c in claves))
    out["videos_cuya_etiqueta_difiere_de_su_sena"] = sum(1 for c in claves if props[c]["manos"] != msena[int(c[:4])])
    out["descriptivo_por_senante"] = {}
    for x in SENANTES:
        cs = [c for c in claves if c[5:9] == x]
        out["descriptivo_por_senante"][x] = {
            "longitud_media": float(np.mean([props[c]["longitud"] for c in cs])),
            "prop_sin_alguna_mano_media": float(np.mean([props[c]["prop_sin_alguna_mano"] for c in cs])),
            "prop_sin_mano_activa_media": float(np.mean([props[c]["prop_sin_mano_activa"] for c in cs])),
            "videos_con_alguna_mano_ausente": sum(1 for c in cs if props[c]["prop_sin_alguna_mano"] > 0),
            "n": len(cs)}
    etiquetas_tramo = ["0 %", "(0; 5 %]", "(5; 15 %]", "> 15 %"]
    for m in MODELOS:
        todas = _juntar(loso[m])
        ok3 = np.array([r["real"] == r["predicha"] for r in todas])
        conf3 = np.array([r["prob_predicha"] for r in todas])
        c3 = [r["muestra"] for r in todas]
        res = {}
        # tramos de mano ausente
        for medida in ("prop_sin_alguna_mano", "prop_sin_mano_activa"):
            tr = [tramo_de(props[c][medida], TRAMOS_MANO_AUSENTE) for c in c3]
            g = exactitud_por_grupo(tr, ok3)
            res[f"tramos_{medida}"] = [{"tramo": etiquetas_tramo[int(k)], "n_videos": v["n"] // 3, **v} for k, v in g.items()]
            vals = [props[c][medida] for c in c3]
            res[f"quintiles_{medida}_nota"] = "muchos ceros: se usan tramos fijos, no quintiles"
            del vals
        # quintiles de longitud (cortes sobre los 997 videos)
        lv = np.array([props[c]["longitud"] for c in claves], dtype=float)
        bordes = np.quantile(lv, np.linspace(0, 1, 6))
        idx = {c: int(np.clip(np.searchsorted(bordes[1:-1], props[c]["longitud"], side="left"), 0, 4)) for c in claves}
        g = exactitud_por_grupo([idx[c] for c in c3], ok3)
        res["quintiles_longitud"] = [{"quintil": int(k) + 1, "desde": float(bordes[int(k)]), "hasta": float(bordes[int(k) + 1]),
                                      "n_videos": v["n"] // 3, **v} for k, v in g.items()]
        # manos
        g = exactitud_por_grupo([props[c]["manos"] for c in c3], ok3)
        res["manos_video"] = g
        g = exactitud_por_grupo([msena[int(c[:4])] for c in c3], ok3)
        res["manos_sena"] = g
        res["manos_video_por_senante"] = {}
        for x in SENANTES:
            sel = [i for i, c in enumerate(c3) if c[5:9] == x]
            res["manos_video_por_senante"][x] = exactitud_por_grupo([props[c3[i]]["manos"] for i in sel], ok3[sel])
        # confianza: fiabilidad (10 intervalos) y tramos de 0,25
        fia = ae.fiabilidad(conf3, ok3, 10)
        res["fiabilidad_3_semillas"] = fia
        res["ece_por_semilla"] = [ae.fiabilidad([r["prob_predicha"] for r in loso[m][sem].values()],
                                                [r["real"] == r["predicha"] for r in loso[m][sem].values()], 10)["ece"]
                                  for sem in SEMILLAS]
        tr = [tramo_confianza(x) for x in conf3]
        g = exactitud_por_grupo(tr, ok3)
        res["tramos_confianza"] = [{"desde": TRAMOS_CONFIANZA[int(k)], "hasta": TRAMOS_CONFIANZA[int(k) + 1], **v,
                                    "proporcion_de_predicciones": v["n"] / len(conf3)} for k, v in g.items()]
        res["confianza_media_aciertos"] = float(conf3[ok3].mean())
        res["confianza_media_errores"] = float(conf3[~ok3].mean())
        res["errores_con_confianza_ge_0_5"] = int(((~ok3) & (conf3 >= 0.5)).sum())
        res["errores_con_confianza_ge_0_75"] = int(((~ok3) & (conf3 >= 0.75)).sum())
        err = [r for r in todas if r["real"] != r["predicha"]]
        res["errores_3_semillas"] = len(err)
        res["errores_con_real_en_top3"] = sum(1 for r in err if r["real"] in r["top3"])
        res["errores_con_real_en_segundo_lugar"] = sum(1 for r in err if r["top3"][1] == r["real"])
        # pruebas dentro de estrato sobre el VOTO (una fila por video)
        okv = np.array([voto[m][c][0] == voto[m][c][1] for c in claves])
        regs = [{"senante": c[5:9], "real": int(c[:4])} for c in claves]
        pruebas, pvals = {}, {}
        for j, medida in enumerate(("longitud", "prop_sin_alguna_mano", "prop_sin_mano_activa")):
            v = np.array([props[c][medida] for c in claves], dtype=float)
            pruebas[medida] = {}
            for e, (etq, f) in ae.ESTRATIFICACIONES.items():
                est = np.array([f(r) for r in regs])
                pr = ae.permutacion_estratificada(v, okv, est, n_perm, semilla + 10 * j + list(ae.ESTRATIFICACIONES).index(e),
                                                  por_estrato=False)
                pruebas[medida][e] = pr
                pvals[f"{medida}|{e}"] = pr["p_bilateral"]
        holm = ae.holm(pvals)
        for clave, ph in holm.items():
            med, e = clave.split("|")
            pruebas[med][e]["p_holm"] = ph
        res["pruebas_voto"] = pruebas
        # entre señas: recall (3 semillas) frente a la etiqueta de manos de la seña
        rec = _recall_precision([r["real"] for r in todas], [r["predicha"] for r in todas])["recall"]
        x = [1.0 if msena[i] == "dos" else 0.0 for i in range(N_CLASES)]
        res["entre_senas_manos"] = {
            "recall_medio_dos": float(np.mean([rec[i] for i in range(N_CLASES) if msena[i] == "dos"])),
            "recall_medio_una": float(np.mean([rec[i] for i in range(N_CLASES) if msena[i] == "una"])),
            **ae.correlacion_permutacion(x, rec, n_perm, semilla + 77)}
        out["modelos"][m] = res
    # ganancia de la TCN por tramo de mano ausente (voto)
    okt = np.array([voto["tcn"][c][0] == voto["tcn"][c][1] for c in claves])
    okh = np.array([voto["heredada"][c][0] == voto["heredada"][c][1] for c in claves])
    out["transicion_por_tramo_sin_alguna_mano"] = {}
    tr = np.array([tramo_de(props[c]["prop_sin_alguna_mano"], TRAMOS_MANO_AUSENTE) for c in claves])
    for k in range(4):
        sel = tr == k
        out["transicion_por_tramo_sin_alguna_mano"][etiquetas_tramo[k]] = tabla_transicion(okt[sel], okh[sel])
    out["transicion_por_manos"] = {g: tabla_transicion(okt[np.array([props[c]["manos"] == g for c in claves])],
                                                       okh[np.array([props[c]["manos"] == g for c in claves])])
                                   for g in ("una", "dos")}
    return out


def seccion_imputacion(loso_t1: dict, voto_t1: dict, guardado: dict, props: dict) -> dict:
    """T1 (final) frente a T0 (misma TCN con «mascara»), por tramo de mano ausente. Solo cambia la regla
    de mano ausente; ambas con 252 rasgos y las mismas semillas y pliegues."""
    if not T0_DIR.exists():
        return {"disponible": False}
    t0 = cargar_loso_semillas(T0_DIR)
    v0 = voto_semillas([{c: (r["real"], r["predicha"], r["prob_predicha"]) for c, r in t0[sem].items()} for sem in SEMILLAS])
    claves = sorted(voto_t1)
    ok1 = np.array([voto_t1[c][0] == voto_t1[c][1] for c in claves])
    ok0 = np.array([v0[c][0] == v0[c][1] for c in claves])
    g = guardado["mcnemar"]["T1_vs_T0"]["voto_3_semillas"]
    glob = tabla_transicion(ok1, ok0)
    if (glob["solo_tcn"], glob["solo_heredada"]) != (g["solo_brazo_acierta"], g["solo_R0_acierta"]):
        raise RuntimeError("T1 frente a T0 (voto) no reproduce resultados_confirmacion_tcn.json")
    etq = ["0 %", "(0; 5 %]", "(5; 15 %]", "> 15 %"]
    out = {"disponible": True, "fuente": ae._rel(T0_DIR),
           "nota": "en estas tablas «solo_tcn» = solo acierta T1 (imputación) y «solo_heredada» = solo acierta T0 (mascara)",
           "global_voto": glob, "por_tramo": {}}
    todas1, todas0 = _juntar(loso_t1), _juntar(t0)
    for medida in ("prop_sin_alguna_mano", "prop_sin_mano_activa"):
        tr = np.array([tramo_de(props[c][medida], TRAMOS_MANO_AUSENTE) for c in claves])
        filas = {}
        for k in range(4):
            sel = tr == k
            cs = set(np.array(claves)[sel])
            filas[etq[k]] = {**tabla_transicion(ok1[sel], ok0[sel]),
                             "exactitud_t1_3_semillas": float(np.mean([r["real"] == r["predicha"] for r in todas1 if r["muestra"] in cs])),
                             "exactitud_t0_3_semillas": float(np.mean([r["real"] == r["predicha"] for r in todas0 if r["muestra"] in cs]))}
        out["por_tramo"][medida] = filas
    return out


def seccion_consistencia(loso, n_top: int = 12) -> dict:
    out = {}
    for m in MODELOS:
        preds = [{c: (r["real"], r["predicha"]) for c, r in loso[m][sem].items()} for sem in SEMILLAS]
        cs = consistencia_semillas(preds)
        pm = cs.pop("por_muestra")
        por_sen = {x: Counter(v["fallos"] for c, v in pm.items() if c[5:9] == x) for x in SENANTES}
        por_sena = Counter(int(c[:4]) for c, v in pm.items() if v["fallos"] == 3)
        tres_misma = Counter((int(c[:4]), v["predichas"][0]) for c, v in pm.items()
                             if v["fallos"] == 3 and v["clases_erroneas_distintas"] == 1)
        out[m] = {**cs, "por_senante": {x: {str(i): int(por_sen[x].get(i, 0)) for i in range(4)} for x in SENANTES},
                  "senas_con_mas_fallos_en_las_3": [(k, n) for k, n in sorted(por_sena.items(), key=lambda t: (-t[1], t[0]))[:n_top]],
                  "pares_estables_3_de_3": [(r, p, n) for (r, p), n in tres_misma.most_common(10)],
                  "_por_muestra_fallos": {c: v["fallos"] for c, v in pm.items()}}
        n_err_videos = sum(cs["distribucion_fallos"][str(i)] for i in (1, 2, 3))
        out[m]["videos_con_algun_fallo"] = n_err_videos
        out[m]["proporcion_fallos_estables"] = cs["distribucion_fallos"]["3"] / n_err_videos if n_err_videos else None
    return out


def seccion_conversor(loso, voto, aleatoria) -> dict:
    conv = ConversorGlosas(estricto=True)
    out = {"modo": "inventario (ConversorGlosas(estricto=True)); una seña aislada → su enunciado fijo de glosa sola",
           "modelos": {}}
    for m in MODELOS:
        errores = [(v[0], v[1]) for c, v in sorted(voto[m].items()) if v[0] != v[1]]
        cl = [clasificar_error_conversor(r, p, conv) for r, p in errores]
        todas3 = [(r["real"], r["predicha"]) for r in _juntar(loso[m]) if r["real"] != r["predicha"]]
        cl3 = [clasificar_error_conversor(r, p, conv) for r, p in todas3]
        trans = Counter((x["categoria_real"], x["categoria_predicha"]) for x in cl if not x["misma_categoria"])
        mismas = Counter(x["categoria_real"] for x in cl if x["misma_categoria"])
        ejemplos = Counter((x["enunciado_real"], x["enunciado_predicho"]) for x in cl)
        out["modelos"][m] = {
            "errores_voto": len(cl),
            "misma_categoria": sum(x["misma_categoria"] for x in cl),
            "cambia_categoria": sum(not x["misma_categoria"] for x in cl),
            "misma_categoria_distinto_genero": sum(x["misma_categoria_distinto_genero"] for x in cl),
            "cambia_modalidad": sum(x["cambia_modalidad"] for x in cl),
            "misma_categoria_heredada_grammar_mapper": sum(x["misma_categoria_heredada"] for x in cl),
            "esperado_misma_categoria_al_azar": esperado_misma_categoria([r for r, _ in errores], categoria_conversor),
            "esperado_misma_categoria_heredada_al_azar": esperado_misma_categoria([r for r, _ in errores], ae.categoria_de),
            "errores_3_semillas": len(cl3),
            "misma_categoria_3_semillas": sum(x["misma_categoria"] for x in cl3),
            "cambia_modalidad_3_semillas": sum(x["cambia_modalidad"] for x in cl3),
            "transiciones_de_categoria_mas_frecuentes": [(a, b, n) for (a, b), n in trans.most_common(10)],
            "dentro_de_categoria_por_categoria": dict(mismas.most_common()),
            "enunciados_mas_frecuentes": [(a, b, n) for (a, b), n in ejemplos.most_common(10)],
        }
    preds_t, _ = aleatoria["tcn"]
    out["aleatoria_tcn"] = [clasificar_error_conversor(r["real"], r["predicha"], conv) for r in preds_t
                            if r["real"] != r["predicha"]]
    return out


def seccion_aleatoria(aleatoria, props) -> dict:
    pt, ph = aleatoria["tcn"][0], aleatoria["heredada"][0]
    dh = {r["muestra"]: r for r in ph}
    claves = sorted(r["muestra"] for r in pt)
    dt_ = {r["muestra"]: r for r in pt}
    okt = np.array([dt_[c]["real"] == dt_[c]["predicha"] for c in claves])
    okh = np.array([dh[c]["real"] == dh[c]["predicha"] for c in claves])
    filas = {}
    for m, preds in (("tcn", pt), ("heredada", ph)):
        filas[m] = []
        for r in sorted(preds, key=lambda r: r["muestra"]):
            if r["real"] == r["predicha"]:
                continue
            p = props[r["muestra"]]
            otra = dh if m == "tcn" else dt_
            filas[m].append({"muestra": r["muestra"], "senante": r["senante"], "real": r["real"], "predicha": r["predicha"],
                             "confianza": r["prob_predicha"], "real_en_top3": r["real"] in r["top3"],
                             "posicion_real_en_top3": (r["top3"].index(r["real"]) + 1) if r["real"] in r["top3"] else None,
                             "longitud": p["longitud"], "prop_sin_alguna_mano": p["prop_sin_alguna_mano"],
                             "manos": p["manos"], "el_otro_modelo_predice": otra[r["muestra"]]["predicha"]})
    conf_t = np.array([dt_[c]["prob_predicha"] for c in claves])
    return {"transicion": tabla_transicion(okt, okh), "errores": filas,
            "errores_por_senante_tcn": dict(Counter(f["senante"] for f in filas["tcn"])),
            "confianza_media_aciertos_tcn": float(conf_t[okt].mean()),
            "confianza_media_errores_tcn": float(conf_t[~okt].mean()),
            "senantes_en_prueba": dict(Counter(c[5:9] for c in claves))}


def seccion_contraste_previo(previo_path: Path, pares, clase, transicion) -> dict:
    """¿Siguen siendo difíciles las parejas y señas que señaló el informe de la heredada?"""
    if not previo_path.exists():
        return {"disponible": False}
    prev = json.loads(previo_path.read_text(encoding="utf-8"))
    todas = {m: {(p["a"], p["b"]): p for p in pares[m]["_todas_parejas_3_semillas"]} for m in MODELOS}
    filas = []
    for p in prev["confusiones"]["parejas_no_ordenadas"][:10]:
        k = (p["a"], p["b"])
        filas.append({"a": p["a"], "b": p["b"], "total_informe_previo": p["total"],
                      "heredada_3_semillas": todas["heredada"].get(k, {}).get("total", 0),
                      "tcn_3_semillas": todas["tcn"].get(k, {}).get("total", 0)})
    peores_prev = sorted(prev["por_sena"]["senas"], key=lambda f: (f["recall"], f["f1"], f["id"]))[:10]
    rec_t = {f["id"]: f["recall"] for f in clase["tcn"]}
    rec_h = {f["id"]: f["recall"] for f in clase["heredada"]}
    rank_t = {f["id"]: i + 1 for i, f in enumerate(sorted(clase["tcn"], key=lambda f: (f["recall"], f["f1"], f["id"])))}
    return {"disponible": True, "fuente": ae._rel(previo_path), "parejas": filas,
            "peores_senas": [{"id": f["id"], "recall_informe_previo": f["recall"], "recall_heredada_3_semillas": rec_h[f["id"]],
                              "recall_tcn_3_semillas": rec_t[f["id"]], "puesto_tcn_de_peor_a_mejor": rank_t[f["id"]]}
                             for f in peores_prev],
            "proporcion_dentro_categoria_informe_previo": prev["confusiones"]["por_categoria"]["proporcion_dentro"],
            "ece_informe_previo": prev["calibracion"]["global"]["ece"]}


# ---------------------------------------------------------------------------------------------------
# Análisis completo
# ---------------------------------------------------------------------------------------------------

def analizar(n_perm: int = 10000, semilla: int = 2026) -> tuple[dict, dict]:
    loso = {m: cargar_loso_semillas(LOSO_DIRS[m]) for m in MODELOS}
    claves = sorted(loso["tcn"][SEMILLAS[0]])
    for m in MODELOS:
        for sem in SEMILLAS:
            if sorted(loso[m][sem]) != claves:
                raise RuntimeError(f"{m} semilla {sem}: las 997 muestras no son las mismas")
    voto = {m: voto_semillas([{c: (r["real"], r["predicha"], r["prob_predicha"]) for c, r in loso[m][sem].items()}
                              for sem in SEMILLAS]) for m in MODELOS}
    aleatoria = {m: cargar_aleatoria(ALEATORIA_DIRS[m]) for m in MODELOS}
    if sorted(r["muestra"] for r in aleatoria["tcn"][0]) != sorted(r["muestra"] for r in aleatoria["heredada"][0]):
        raise RuntimeError("las dos particiones aleatorias no tienen las mismas 100 muestras de prueba")
    guardado = json.loads(RESULTADOS_CONFIRMACION.read_text(encoding="utf-8"))
    ctl = controles(loso, aleatoria, guardado, voto)
    ctl["heredada_semilla_2026_vs_loso_informe_previo"] = control_semilla_2026_vs_informe_previo(loso["heredada"][2026], LOSO_PREVIO)
    if not ctl["todo_coincide"]:
        raise RuntimeError("los controles no coinciden con lo guardado; ver controles: "
                           + json.dumps({k: v for k, v in ctl.items() if k != "corridas"}, default=str)[:2000])
    muestras = sorted(set(claves) | {r["muestra"] for r in aleatoria["tcn"][0]})
    props, ctl_props = propiedades_videos(PROCESSED, METRICAS_FOTOGRAMA, muestras)
    ctl["propiedades"] = ctl_props
    if ctl_props["metricas_fotograma_disponible"] and not ctl_props["coinciden_bloques_vs_metricas"]:
        raise RuntimeError("METRICAS_FOTOGRAMA y los bloques en ceros de X_padded no coinciden")
    pares = seccion_pares(loso, voto)
    clase = seccion_por_clase(loso, voto)
    transicion = seccion_transicion(voto)
    a = {
        "descripcion": "Análisis de errores del modelo final D-14 (TCN + imputación + 252), actividad 4.d. Solo reporte.",
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "src/analisis_errores_tcn.py",
        "fuentes": {"loso_tcn": ae._rel(LOSO_DIRS["tcn"]), "loso_heredada": ae._rel(LOSO_DIRS["heredada"]),
                    "resultados_guardados": ae._rel(RESULTADOS_CONFIRMACION),
                    "aleatoria_tcn": ae._rel(ALEATORIA_DIRS["tcn"]), "aleatoria_heredada": ae._rel(ALEATORIA_DIRS["heredada"]),
                    "processed": ae._rel(PROCESSED), "metricas_fotograma": ae._rel(METRICAS_FOTOGRAMA),
                    "informe_previo": ae._rel(ANALISIS_PREVIO)},
        "parametros": {"semillas": list(SEMILLAS), "permutaciones": n_perm, "semilla_permutaciones": semilla,
                       "umbral_manos": UMBRAL_MANOS, "tramos_confianza": list(TRAMOS_CONFIANZA),
                       "tramos_mano_ausente": list(TRAMOS_MANO_AUSENTE)},
        "controles": ctl,
        "global": seccion_global(loso, voto),
        "pares": pares,
        "por_clase": clase,
        "por_senante": seccion_por_senante(loso, voto),
        "transicion": transicion,
        "propiedades": seccion_propiedades(loso, voto, props, n_perm, semilla),
        "consistencia": seccion_consistencia(loso),
        "imputacion_t1_vs_t0": seccion_imputacion(loso["tcn"], voto["tcn"], guardado, props),
        "conversor": seccion_conversor(loso, voto, aleatoria),
        "aleatoria": seccion_aleatoria(aleatoria, props),
        "contraste_informe_previo": seccion_contraste_previo(ANALISIS_PREVIO, pares, clase, transicion),
    }
    return a, {"loso": loso, "voto": voto, "props": props, "aleatoria": aleatoria}


# ---------------------------------------------------------------------------------------------------
# Figuras (sin personas ni fotogramas: solo cifras)
# ---------------------------------------------------------------------------------------------------

COLOR = {"heredada": "#eb6834", "tcn": "#2a78d6"}   # validado (dataviz validate_palette.js, modo claro)


def figuras(a: dict, datos: dict, carpeta: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ae._estilo(plt)
    T, T2, REJ = ae.TINTA, ae.TINTA_2, ae.REJILLA_COLOR
    carpeta.mkdir(parents=True, exist_ok=True)
    hechas = []

    def guardar(fig, nombre):
        fig.savefig(carpeta / nombre, dpi=150)
        plt.close(fig)
        hechas.append(ae._rel(carpeta / nombre))

    # 1. Matriz de confusión LOSO de la TCN (3 semillas), normalizada por fila, ordenada por categoría
    todas = _juntar(datos["loso"]["tcn"])
    orden = [i for c in ae.ORDEN_CATEGORIAS for i in range(N_CLASES) if ae.categoria_de(i) == c]
    m = ae.matriz_confusion([r["real"] for r in todas], [r["predicha"] for r in todas]).astype(float)
    mn = (m / m.sum(axis=1, keepdims=True))[np.ix_(orden, orden)]
    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(mn, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    etq = [s(i) for i in orden]
    ax.set_xticks(range(N_CLASES), etq, rotation=90, fontsize=7)
    ax.set_yticks(range(N_CLASES), etq, fontsize=7)
    pos = 0
    for c in ae.ORDEN_CATEGORIAS:
        n = sum(1 for i in range(N_CLASES) if ae.categoria_de(i) == c)
        ax.axhline(pos - 0.5, color=T2, lw=0.6)
        ax.axvline(pos - 0.5, color=T2, lw=0.6)
        ax.text(N_CLASES + 0.8, pos + n / 2 - 0.5, c, va="center", fontsize=8, color=T)
        pos += n
    ax.set_xlabel("seña predicha")
    ax.set_ylabel("seña real")
    ax.set_title("TCN final, LOSO: matriz de confusión de las 2.991 predicciones (997 videos × 3 semillas),\n"
                 "normalizada por fila y ordenada por categoría de grammar_mapper", fontsize=10, color=T)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.09, label="proporción de la fila")
    fig.tight_layout()
    guardar(fig, f"{PREFIJO_FIG}matriz_confusion_loso_tcn.png")

    # 2. Parejas más confundidas: heredada frente a TCN (media por semilla de los dos sentidos)
    todas_p = {mm: {(p["a"], p["b"]): p["total"] for p in a["pares"][mm]["_todas_parejas_3_semillas"]} for mm in MODELOS}
    union = []
    for mm in MODELOS:
        for p in a["pares"][mm]["_todas_parejas_3_semillas"][:10]:
            if (p["a"], p["b"]) not in union:
                union.append((p["a"], p["b"]))
    union.sort(key=lambda k: (todas_p["tcn"].get(k, 0) + todas_p["heredada"].get(k, 0)))
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(union) + 1.6))
    y = np.arange(len(union))
    h = 0.38
    for j, mm in enumerate(("heredada", "tcn")):
        v = [todas_p[mm].get(k, 0) / 3 for k in union]
        ax.barh(y + (h / 2 if mm == "heredada" else -h / 2), v, height=h - 0.04, color=COLOR[mm], label=ETIQUETA[mm])
    ax.set_yticks(y, [f"{SIGN_VOCABULARY[x]} / {SIGN_VOCABULARY[z]}" for x, z in union], fontsize=8)
    ax.set_xlabel("errores por semilla en la pareja (dos sentidos; media de 3 semillas, LOSO)")
    ax.grid(axis="x", color=REJ, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Parejas más confundidas: unión de las 10 primeras de cada red", fontsize=10, color=T)
    fig.tight_layout()
    guardar(fig, f"{PREFIJO_FIG}parejas_confundidas_heredada_vs_tcn.png")

    # 3. Recall por clase, heredada frente a TCN (3 semillas)
    rec = {mm: {f["id"]: f["recall"] for f in a["por_clase"][mm]} for mm in MODELOS}
    ordc = sorted(range(N_CLASES), key=lambda i: (rec["tcn"][i], rec["heredada"][i], i))
    fig, ax = plt.subplots(figsize=(8, 11))
    y = np.arange(N_CLASES)
    for yi, i in zip(y, ordc):
        ax.plot([rec["heredada"][i], rec["tcn"][i]], [yi, yi], color=REJ, lw=2, zorder=1)
    for mm, mk in (("heredada", "o"), ("tcn", "s")):
        ax.scatter([rec[mm][i] for i in ordc], y, s=26, marker=mk, color=COLOR[mm], zorder=3, label=ETIQUETA[mm],
                   edgecolors="#fcfcfb", linewidths=0.8)
    ax.set_yticks(y, [s(i) for i in ordc], fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("recall (LOSO, 3 semillas: 60 predicciones por seña, 57 en 3 señas)")
    ax.grid(axis="x", color=REJ, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Recall por seña: heredada frente a TCN final (ordenado por la TCN)", fontsize=10, color=T)
    fig.tight_layout()
    guardar(fig, f"{PREFIJO_FIG}recall_por_clase_heredada_vs_tcn.png")

    # 4. Acierto frente a propiedades del video
    P = a["propiedades"]["modelos"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    def barras(ax, grupos, etiquetas, titulo, xlabel):
        x = np.arange(len(etiquetas))
        for j, mm in enumerate(("heredada", "tcn")):
            v = [g["exactitud"] for g in grupos[mm]]
            b = ax.bar(x + (j - 0.5) * 0.38, v, width=0.36, color=COLOR[mm], label=ETIQUETA[mm])
            for bi, vi in zip(b, v):
                ax.text(bi.get_x() + bi.get_width() / 2, vi + 0.012, _n(vi, 2), ha="center", fontsize=7, color=T2)
        ax.set_xticks(x, etiquetas, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("exactitud top-1 (LOSO, 3 semillas)")
        ax.set_xlabel(xlabel)
        ax.set_title(titulo, fontsize=9.5, color=T)
        ax.grid(axis="y", color=REJ, lw=0.6)
        ax.set_axisbelow(True)

    g = {mm: P[mm]["tramos_prop_sin_alguna_mano"] for mm in MODELOS}
    barras(axs[0, 0], g, [f"{x['tramo']}\nn = {x['n_videos']}" for x in g["tcn"]],
           "(a) Fotogramas sin alguna mano (antes de imputar)", "proporción de fotogramas del video")
    g = {mm: P[mm]["quintiles_longitud"] for mm in MODELOS}
    barras(axs[0, 1], g, [f"{int(x['desde'])}–{x['hasta']:.0f}\nn = {x['n_videos']}" for x in g["tcn"]],
           "(b) Duración del video (quintiles)", "fotogramas antes de remuestrear a 30")
    ax = axs[1, 0]
    ax.plot([0, 1], [0, 1], color=T2, lw=1, ls="--", label="calibración perfecta")
    for mm, mk in (("heredada", "o"), ("tcn", "s")):
        it = [b for b in P[mm]["fiabilidad_3_semillas"]["intervalos"] if b["n"] >= 20]
        ax.plot([b["confianza_media"] for b in it], [b["exactitud"] for b in it], marker=mk, ms=6, lw=2, color=COLOR[mm],
                label=f"{ETIQUETA[mm]} (ECE {_n(P[mm]['fiabilidad_3_semillas']['ece'], 3)})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confianza media del intervalo (probabilidad máxima; 10 intervalos, n ≥ 20)")
    ax.set_ylabel("exactitud del intervalo")
    ax.set_title("(c) Acierto por tramo de confianza", fontsize=9.5, color=T)
    ax.grid(color=REJ, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    g = {mm: [P[mm]["manos_video"]["una"], P[mm]["manos_video"]["dos"]] for mm in MODELOS}
    barras(axs[1, 1], g, [f"una mano se mueve\nn = {g['tcn'][0]['n'] // 3}", f"se mueven las dos\nn = {g['tcn'][1]['n'] // 3}"],
           "(d) Manos en movimiento (proxy medido por video)", "")
    axs[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Acierto LOSO según propiedades medibles del video (asociaciones, no causas)", fontsize=10.5, color=T)
    fig.tight_layout()
    guardar(fig, f"{PREFIJO_FIG}acierto_vs_propiedades.png")
    return hechas


# ---------------------------------------------------------------------------------------------------
# Informe en Markdown (cada cifra sale de `a`, que se guarda tal cual en el JSON)
# ---------------------------------------------------------------------------------------------------

def _fr(a_: int, b_: int) -> str:
    return f"{a_}/{b_} ({_pct(a_ / b_)})" if b_ else f"{a_}/0"


def _pv(p) -> str:
    if p is None:
        return "—"
    if p < 1e-4:
        m, e = f"{p:.1e}".split("e")
        return f"{m.replace('.', ',')}·10^{int(e)}"
    return _n(p, 4)


def _tabla_trans(t: dict) -> str:
    return (f"{t['ambos_aciertan']} | {t['solo_tcn']} | {t['solo_heredada']} | {t['ambos_fallan']} | "
            f"{t['neto_tcn']:+d} | {_pv(t['p_mcnemar_exacto'])}")


def informe_md(a: dict) -> str:
    L: list[str] = []
    w = L.append
    G, C, PS, TR, PR, CO, CV, AL, IM, CP = (a["global"], a["por_clase"], a["por_senante"], a["transicion"],
                                            a["propiedades"], a["consistencia"], a["conversor"], a["aleatoria"],
                                            a["imputacion_t1_vs_t0"], a["contraste_informe_previo"])
    ctl = a["controles"]
    rt, rh = ctl["resumen"]["tcn"], ctl["resumen"]["heredada"]
    Pt, Ph = PR["modelos"]["tcn"], PR["modelos"]["heredada"]
    w("# Análisis de errores del modelo final: TCN + imputación + 252 rasgos (actividad 4.d)\n")
    w(f"Generado por `src/analisis_errores_tcn.py` el {a['fecha']}. SOLO REPORTE: no decide, no reentrena y no ejecuta "
      "ningún modelo; lee predicciones ya guardadas. Todas las cifras están en `docs/ANALISIS_ERRORES_4d_TCN.json`; "
      "cada sección indica la clave de origen. Decimales con coma. Figuras en `docs/figuras_2026-11-08/` (prefijo `4d_`).\n")
    w("Modelo final (D-14): TCN de 4 bloques residuales Conv1D (dilatación 1-2-4-8, 64 filtros, 156.402 parámetros), "
      "regla de mano ausente «imputacion», 252 rasgos (126 posiciones + 126 velocidades), 30 pasos, umbral 0,00. "
      "«Heredada» = la red canónica anterior (Conv1D + BiLSTM + atención, 714.291 parámetros, regla «mascara»), "
      "la misma que analizó el informe previo `models/analisis_errores_manos_2026-09-15/analisis.md`.\n")

    # ---------------- 0
    w("## 0. Método, fuentes y controles\n")
    w("**Fuentes** (`fuentes`):\n")
    w(f"- TCN final, LOSO: `{a['fuentes']['loso_tcn']}/semilla_S/pliegue_k/predicciones.json` (3 semillas 2026-2028 × "
      "5 pliegues; el pliegue k deja fuera al señante k; 997 videos por semilla).")
    w(f"- Heredada, LOSO: `{a['fuentes']['loso_heredada']}/…` (el brazo R0 que reutilizan los experimentos de "
      "arquitecturas y de confirmación; mismas semillas y pliegues).")
    w(f"- Partición aleatoria 797/100/100 (señantes vistos): `{a['fuentes']['aleatoria_tcn']}` (semilla 2030) y "
      f"`{a['fuentes']['aleatoria_heredada']}` (semilla 2026); las mismas 100 muestras de prueba.")
    w(f"- Propiedades de cada video: `{a['fuentes']['processed']}` (lengths.npy; X_padded.npy con la mano ausente en "
      f"ceros, es decir, antes de imputar) y `{a['fuentes']['metricas_fotograma']}` (solo lectura).")
    w(f"- TCN con «mascara» (T0, para aislar la imputación, §6.2): `{IM.get('fuente', '—')}`.")
    w("- No se usaron los `probabilidades.npz` (ignorados por git): bastan `prob_predicha` y `top3` de los "
      "predicciones.json. No se regeneró nada.\n")
    w("**Unidades.** «3 semillas» = las 2.991 predicciones (997 videos × 3 modelos entrenados con semillas distintas) "
      "juntas: es la exactitud media de los tres modelos y da conteos más estables que una sola semilla. «Voto» = una "
      "predicción por video, la clase que predicen al menos 2 de las 3 semillas (si las tres discrepan, la de mayor "
      "probabilidad; la regla de `experimento_confirmacion_tcn.voto_semillas`). El voto es un recurso de análisis "
      "(un conjunto de 3 modelos), no el modelo desplegado: su exactitud es mayor que la de cada semilla.\n")
    w("**Controles** (`controles`; el script se detiene si alguno falla):\n")
    w("| modelo | top-1 (media de 15 corridas) | top-3 | F1 macro | señante 0000 | coincide con el registro (3 decimales) | aleatoria: top-1 / errores |")
    w("|---|---|---|---|---|---|---|")
    for mm, r in (("heredada", rh), ("tcn", rt)):
        w(f"| {ETIQUETA[mm]} | {_n(r['top1'])} | {_n(r['top3'])} | {_n(r['f1'])} | {_n(r['senante_0000'])} | "
          f"{'sí' if r['coincide_con_registro_3_decimales'] else 'NO'} | {_n(r['aleatoria']['top1'], 2)} / "
          f"{r['aleatoria']['errores']} ({'coincide' if r['aleatoria']['coincide'] else 'NO coincide'} con metricas.json) |")
    mc = ctl["mcnemar_voto"]
    w("")
    w(f"- Las 30 corridas (2 modelos × 3 semillas × 5 pliegues) reproducen top-1, top-3 y F1 de "
      "`resultados_confirmacion_tcn.json › corridas` (tolerancia 10⁻⁶).")
    w(f"- McNemar con el voto, TCN frente a heredada: solo acierta la TCN {mc['calculado']['solo_tcn']}, solo la heredada "
      f"{mc['calculado']['solo_heredada']}, p = {_pv(mc['calculado']['p_mcnemar_exacto'])} (guardado: "
      f"{mc['guardado']['solo_T1']} / {mc['guardado']['solo_R0']}; coincide).")
    c26 = ctl["heredada_semilla_2026_vs_loso_informe_previo"]
    if c26.get("disponible"):
        w(f"- La heredada con semilla 2026 da exactamente las mismas {c26['predicciones_iguales']} de {c26['n']} predicciones "
          "que el LOSO del informe previo (`models/loso_2026-09-15_holistic_mascara_manos`): el informe previo es una de "
          "las tres semillas de la heredada que se usa aquí.")
    w(f"- Fotogramas sin mano: METRICAS_FOTOGRAMA y los bloques en ceros de X_padded coinciden video a video "
      f"(discrepancias: {ctl['propiedades'].get('discrepancias')}).")
    w("")

    # ---------------- 1
    w("## 1. Cifras globales\n")
    w("Fuente: `controles.resumen` (media de pliegues) y `global` (3 semillas juntas y voto).\n")
    w("| modelo | top-1 media de pliegues | top-3 | F1 macro | señante 0000 | errores (3 semillas, de 2.991) | top-1 voto | errores voto (de 997) | aleatoria |")
    w("|---|---|---|---|---|---|---|---|---|")
    for mm, r in (("heredada", rh), ("tcn", rt)):
        w(f"| {ETIQUETA[mm]} | {_n(r['top1'])} | {_n(r['top3'])} | {_n(r['f1'])} | {_n(r['senante_0000'])} | "
          f"{G[mm]['errores_3_semillas']} | {_n(G[mm]['exactitud_voto'])} | {G[mm]['errores_voto']} | "
          f"{_n(r['aleatoria']['top1'], 2)} ({r['aleatoria']['errores']} errores) |")
    w("")

    # ---------------- 2
    P = a["pares"]
    w("## 2. Pares más confundidos\n")
    w("Fuente: `pares`. Pareja = los dos sentidos de una confusión {a, b} juntos; par = un sentido real → predicha.\n")
    w(f"**TCN final, parejas con más errores** (3 semillas; `pares.tcn.parejas_3_semillas`). {P['tcn']['errores_3_semillas']} "
      f"errores en {P['tcn']['n_pares_distintos_3_semillas']} pares dirigidos distintos; las 10 primeras parejas reúnen "
      f"{_pct(P['tcn']['proporcion_errores_en_10_parejas_principales'])} de los errores (heredada: "
      f"{_pct(P['heredada']['proporcion_errores_en_10_parejas_principales'])}).\n")
    w("| pareja {a, b} | a → b | b → a | total 3 semillas | por semilla (media) |")
    w("|---|---|---|---|---|")
    for p in P["tcn"]["parejas_3_semillas"][:12]:
        w(f"| {s(p['a'])} / {s(p['b'])} | {p['a_a_b']} | {p['b_a_a']} | {p['total']} | {_n(p['total'] / 3, 1)} |")
    w("")
    w("**TCN final, pares dirigidos con el voto** (una predicción por video; `pares.tcn.pares_voto`):\n")
    w("| real → predicha | conteo (de 4 videos por señante) | inverso | señantes en que ocurre |")
    w("|---|---|---|---|")
    for p in P["tcn"]["pares_voto"][:12]:
        w(f"| {s(p['real'])} → {s(p['predicha'])} | {p['conteo']} | {p['conteo_inverso']} | {', '.join(p['senantes'])} |")
    w("")
    w("**Heredada, parejas con más errores** (3 semillas; `pares.heredada.parejas_3_semillas`), para comparar:\n")
    w("| pareja {a, b} | a → b | b → a | total 3 semillas | TCN, misma pareja |")
    w("|---|---|---|---|---|")
    tot_t = {(p["a"], p["b"]): p["total"] for p in P["tcn"]["_todas_parejas_3_semillas"]}
    for p in P["heredada"]["parejas_3_semillas"][:12]:
        w(f"| {s(p['a'])} / {s(p['b'])} | {p['a_a_b']} | {p['b_a_a']} | {p['total']} | {tot_t.get((p['a'], p['b']), 0)} |")
    w("")
    com = P["parejas_top10_comunes"]
    w(f"De las 10 parejas principales de cada red, {len(com)} son comunes: "
      + "; ".join(f"{SIGN_VOCABULARY[x]} / {SIGN_VOCABULARY[z]}" for x, z in com) + ". "
      "Figura: `4d_parejas_confundidas_heredada_vs_tcn.png`; matriz completa: `4d_matriz_confusion_loso_tcn.png`.\n")

    # ---------------- 3
    w("## 3. Clases con peor recall y peor precisión\n")
    w("Fuente: `por_clase.tcn` (3 semillas: 60 predicciones por seña, 57 en las señas 2, 8 y 16). Precisión = aciertos / "
      "veces que se predijo la seña.\n")
    w("**Peores 10 por recall (TCN):**\n")
    w("| seña | categoría (conversor) | recall | recall por semilla | precisión | veces predicha | recall heredada |")
    w("|---|---|---|---|---|---|---|")
    rh_ = {f["id"]: f for f in C["heredada"]}
    for f in sorted(C["tcn"], key=lambda f: (f["recall"], f["f1"], f["id"]))[:10]:
        w(f"| {s(f['id'])} | {f['categoria_conversor']} | {_n(f['recall'], 2)} | "
          f"{' / '.join(_n(x, 2) for x in f['recall_por_semilla'])} | {_n(f['precision'], 2)} | {f['veces_predicha']} | "
          f"{_n(rh_[f['id']]['recall'], 2)} |")
    w("")
    w("**Peores 8 por precisión (TCN)** (señas que «atraen» predicciones de otras):\n")
    w("| seña | precisión | veces predicha (de 2.991) | recall |")
    w("|---|---|---|---|")
    for f in sorted(C["tcn"], key=lambda f: (f["precision"], f["id"]))[:8]:
        w(f"| {s(f['id'])} | {_n(f['precision'], 2)} | {f['veces_predicha']} | {_n(f['recall'], 2)} |")
    w("")
    w(f"Recall 0 con el voto: TCN {', '.join(s(i) for i in C['recall_0_tcn_voto']) or 'ninguna'}; heredada "
      f"{', '.join(s(i) for i in C['recall_0_heredada_voto']) or 'ninguna'}. Frente a la heredada (3 semillas), la TCN "
      f"sube el recall en {C['ganan']} señas y lo baja en {C['pierden']}. Mayores subidas: "
      + ", ".join(f"{d['nombre']} {d['diferencia']:+.2f}".replace(".", ",") for d in C["diferencias"][:6])
      + ". Bajadas: " + ", ".join(f"{d['nombre']} {d['diferencia']:+.2f}".replace(".", ",")
                                   for d in sorted(C["diferencias"], key=lambda d: d["diferencia"])[:6] if d["diferencia"] < 0)
      + ". Figura: `4d_recall_por_clase_heredada_vs_tcn.png`.\n")

    # ---------------- 4
    w("## 4. Por señante\n")
    w("Fuente: `por_senante` (media de las 3 semillas del pliegue que deja fuera a ese señante; «voto»: una predicción por video).\n")
    w("| señante | top-1 heredada | top-1 TCN | top-1 TCN por semilla | top-3 TCN | F1 TCN | top-1 voto TCN | señas con recall 0 (voto TCN) |")
    w("|---|---|---|---|---|---|---|---|")
    for x in SENANTES:
        t, h = PS["tcn"][x], PS["heredada"][x]
        w(f"| {x} | {_n(h['top1'])} | {_n(t['top1'])} | {' / '.join(_n(v) for v in t['top1_por_semilla'])} | "
          f"{_n(t['top3'])} | {_n(t['f1'])} | {_n(t['exactitud_voto'])} | {len(t['senas_recall_0_voto'])} |")
    w("")
    t0, h0 = PS["tcn"]["0000"], PS["heredada"]["0000"]
    w(f"**Señante 0000.** Sigue siendo el peor con las dos redes (TCN {_n(t0['top1'])}, heredada {_n(h0['top1'])}). Con el "
      f"voto, la TCN deja {len(t0['senas_recall_0_voto'])} señas con recall 0 ({', '.join(s(i) for i in t0['senas_recall_0_voto'])}) "
      f"frente a {len(h0['senas_recall_0_voto'])} de la heredada. Pares más frecuentes de 0000 con la TCN (3 semillas): "
      + "; ".join(f"{s(r)} → {s(p_)} ×{n}" for r, p_, n in t0["pares_principales_3_semillas"][:5])
      + f". Confianza media de la TCN en 0000: aciertos {_n(t0['confianza_media_aciertos'])}, errores "
      f"{_n(t0['confianza_media_errores'])}. Es también el señante en que más discrepan las semillas (§7): su top-1 por "
      f"semilla va de {_n(min(t0['top1_por_semilla']))} a {_n(max(t0['top1_por_semilla']))} y el voto sube a "
      f"{_n(t0['exactitud_voto'])}.\n")
    w("Pares principales de los otros señantes (TCN, 3 semillas): " + " · ".join(
        f"**{x}**: " + "; ".join(f"{SIGN_VOCABULARY[r]} → {SIGN_VOCABULARY[p_]} ×{n}" for r, p_, n in PS['tcn'][x]['pares_principales_3_semillas'][:3])
        for x in SENANTES[1:]) + ".\n")

    # ---------------- 5
    w("## 5. Qué errores de la heredada corrige la TCN y cuáles introduce\n")
    w("Fuente: `transicion` (voto de 3 semillas de cada red; un video = una fila). «Neto» = solo TCN − solo heredada; "
      "p = McNemar exacto bilateral.\n")
    w("| grupo | ambos aciertan | solo TCN | solo heredada | ambos fallan | neto | p |")
    w("|---|---|---|---|---|---|---|")
    w(f"| todos (997) | {_tabla_trans(TR['global'])} |")
    for x in SENANTES:
        w(f"| señante {x} | {_tabla_trans(TR['por_senante'][x])} |")
    for cat in ae.ORDEN_CATEGORIAS:
        w(f"| categoría {cat} | {_tabla_trans(TR['por_categoria'][cat])} |")
    w("")
    g_ = TR["global"]
    w(f"- De los {g_['solo_tcn'] + g_['ambos_fallan']} videos que la heredada falla, la TCN corrige {g_['solo_tcn']} "
      f"({_pct(g_['solo_tcn'] / (g_['solo_tcn'] + g_['ambos_fallan']))} de los errores de la heredada) e introduce "
      f"{g_['solo_heredada']} errores nuevos ({_pct(g_['solo_heredada'] / (g_['ambos_aciertan'] + g_['solo_heredada']))} de los "
      f"aciertos de la heredada). En {g_['ambos_fallan']} videos fallan las dos; en {TR['ambos_fallan_misma_prediccion']} de ellos "
      f"({_pct(TR['ambos_fallan_misma_prediccion'] / g_['ambos_fallan'])}) predicen la misma seña equivocada.")
    w(f"- El neto es positivo en {sum(1 for v in TR['por_senante'].values() if v['neto_tcn'] > 0)} de 5 señantes y en "
      f"{sum(1 for v in TR['por_categoria'].values() if v['neto_tcn'] > 0)} de 8 categorías. Por seña: ganan {TR['senas_netas']['ganan']}, pierden "
      f"{TR['senas_netas']['pierden']}, igual {TR['senas_netas']['igual']} (4 videos por señante y seña: diferencias de 1-2 "
      "videos son ruido).")
    w("- Señas que más ganan (solo TCN / solo heredada): " + "; ".join(
        f"{d['nombre']} {d['solo_tcn']}/{d['solo_heredada']}" for d in TR["senas_que_mas_ganan"][:8]) + ".")
    w("- Señas con neto negativo: " + "; ".join(
        f"{d['nombre']} {d['solo_tcn']}/{d['solo_heredada']}" for d in TR["senas_que_mas_pierden"]) + ".")
    w("- Errores de la heredada que la TCN corrige, por par: " + "; ".join(
        f"{SIGN_VOCABULARY[r]} → {SIGN_VOCABULARY[p_]} ×{n}" for r, p_, n in TR["errores_corregidos_pares_heredada"]) + ".")
    w("- Errores que introduce la TCN, por par: " + "; ".join(
        f"{SIGN_VOCABULARY[r]} → {SIGN_VOCABULARY[p_]} ×{n}" for r, p_, n in TR["errores_introducidos_pares"]) + ".\n")

    # ---------------- 6
    w("## 6. Error y propiedades medibles del video\n")
    w("Fuente: `propiedades`. Definiciones (`propiedades.definiciones`):\n")
    for k, v in PR["definiciones"].items():
        w(f"- **{k}**: {v}.")
    w("")
    w("Exactitudes con las 3 semillas juntas (cada video cuenta 3 veces); pruebas sobre el voto TCN con el estadístico del "
      "informe previo (`analisis_errores.permutacion_estratificada`: diferencia errores − aciertos dentro de estrato, "
      f"permutación dentro del estrato, {a['parametros']['permutaciones']} permutaciones; Holm sobre las 9 pruebas de cada red).\n")
    w("**Descriptivo por señante** (`propiedades.descriptivo_por_senante`):\n")
    w("| señante | longitud media | fotogramas sin alguna mano | sin la mano que más se mueve | videos con alguna mano ausente |")
    w("|---|---|---|---|---|")
    for x in SENANTES:
        d = PR["descriptivo_por_senante"][x]
        w(f"| {x} | {_n(d['longitud_media'], 1)} | {_pct(d['prop_sin_alguna_mano_media'])} | {_pct(d['prop_sin_mano_activa_media'])} | "
          f"{d['videos_con_alguna_mano_ausente']} de {d['n']} |")
    w("")
    w("### 6.1 Mano ausente (antes de imputar)\n")
    w("| tramo de fotogramas sin alguna mano | videos | exactitud heredada | exactitud TCN | solo TCN / solo heredada (voto) |")
    w("|---|---|---|---|---|")
    for xh, xt in zip(Ph["tramos_prop_sin_alguna_mano"], Pt["tramos_prop_sin_alguna_mano"]):
        tt = PR["transicion_por_tramo_sin_alguna_mano"][xt["tramo"]]
        w(f"| {xt['tramo']} | {xt['n_videos']} | {_n(xh['exactitud'])} | {_n(xt['exactitud'])} | {tt['solo_tcn']} / {tt['solo_heredada']} |")
    w("")
    w("| tramo de fotogramas sin la mano que más se mueve | videos | exactitud heredada | exactitud TCN |")
    w("|---|---|---|---|")
    for xh, xt in zip(Ph["tramos_prop_sin_mano_activa"], Pt["tramos_prop_sin_mano_activa"]):
        w(f"| {xt['tramo']} | {xt['n_videos']} | {_n(xh['exactitud'])} | {_n(xt['exactitud'])} |")
    w("")
    w("**Pruebas dentro de estrato** (`propiedades.modelos.<red>.pruebas_voto`; diferencia = media de errores − media de "
      "aciertos, en puntos porcentuales o fotogramas):\n")
    w("| medida | estratificación | heredada: diferencia (p; p Holm) | TCN: diferencia (p; p Holm) |")
    w("|---|---|---|---|")
    for med in ("prop_sin_alguna_mano", "prop_sin_mano_activa", "longitud"):
        for e, (etq, _) in ae.ESTRATIFICACIONES.items():
            def celda(mm):
                x = PR["modelos"][mm]["pruebas_voto"][med][e]
                d = x["diferencia_dentro_de_estratos"]
                dd = (f"{d:+.2f} fotogramas" if med == "longitud" else f"{100 * d:+.2f} pp").replace(".", ",")
                return f"{dd} ({_p(x['p_bilateral'])}; {_p(x['p_holm'])})"
            w(f"| {med} | {etq} | {celda('heredada')} | {celda('tcn')} |")
    w("")
    def _pt(mm, med, e):
        return PR["modelos"][mm]["pruebas_voto"][med][e]["p_holm"]
    w(f"**Lectura.** Dentro de cada señante, los videos fallados tienen más fotogramas sin la mano que más se mueve, con las "
      f"dos redes (p Holm {_p(_pt('heredada', 'prop_sin_mano_activa', 'senante'))} y {_p(_pt('tcn', 'prop_sin_mano_activa', 'senante'))}), "
      f"y son unos 2 fotogramas más cortos (p Holm {_p(_pt('heredada', 'longitud', 'senante'))} y {_p(_pt('tcn', 'longitud', 'senante'))}). "
      "Al comparar videos de la misma seña, la duración no se asocia con el error en ninguna red; la mano ausente sí con la "
      f"heredada (sin alguna mano, dentro de seña: p Holm {_p(_pt('heredada', 'prop_sin_alguna_mano', 'sena'))}) pero no con la TCN "
      f"(p Holm {_p(_pt('tcn', 'prop_sin_alguna_mano', 'sena'))}). Lo que sí distingue a los videos fallados es sobre todo qué seña y "
      "qué señante son; la diferencia entre las dos redes es compatible con que la imputación quite parte del efecto de la "
      "mano ausente, pero esta comparación no lo aísla (§6.2).\n")
    w("### 6.2 Aislar la imputación: TCN con «mascara» (T0) frente a la TCN final (T1)\n")
    if IM.get("disponible"):
        gi = IM["global_voto"]
        w(f"Fuente: `imputacion_t1_vs_t0`. Misma arquitectura, rasgos, semillas y pliegues; solo cambia la regla de mano "
          f"ausente. Global (voto): solo T1 {gi['solo_tcn']}, solo T0 {gi['solo_heredada']}, p = {_pv(gi['p_mcnemar_exacto'])} "
          "(reproduce `resultados_confirmacion_tcn.json › mcnemar.T1_vs_T0`).\n")
        w("| tramo | medida | videos | exactitud T0 (mascara) | exactitud T1 (imputación) | solo T1 / solo T0 (voto) | p |")
        w("|---|---|---|---|---|---|---|")
        for med, etqm in (("prop_sin_alguna_mano", "sin alguna mano"), ("prop_sin_mano_activa", "sin la mano que más se mueve")):
            for tramo, f in IM["por_tramo"][med].items():
                w(f"| {tramo} | {etqm} | {f['n']} | {_n(f['exactitud_t0_3_semillas'])} | {_n(f['exactitud_t1_3_semillas'])} | "
                  f"{f['solo_tcn']} / {f['solo_heredada']} | {_pv(f['p_mcnemar_exacto'])} |")
        fa = IM["por_tramo"]["prop_sin_alguna_mano"]
        alto = fa["(5; 15 %]"]["neto_tcn"] + fa["> 15 %"]["neto_tcn"]
        bajo = fa["0 %"]["neto_tcn"] + fa["(0; 5 %]"]["neto_tcn"]
        nalto = fa["(5; 15 %]"]["n"] + fa["> 15 %"]["n"]
        w("")
        w(f"Del neto de +{gi['neto_tcn']} videos a favor de la imputación, +{alto} están en los {nalto} videos con más de 5 % de "
          f"fotogramas sin alguna mano y {bajo:+d} en los {997 - nalto} restantes. Es el patrón esperable si la imputación "
          "ayuda donde hay manos que rellenar, pero ninguna celda por separado llega a p < 0,05 salvo «sin la mano que más "
          "se mueve, (5; 15 %]», no se probó la interacción y las celdas se miraron después de ver los datos: es descriptivo.\n")
    w("### 6.3 Duración del video\n")
    w("| quintil | fotogramas | videos | exactitud heredada | exactitud TCN |")
    w("|---|---|---|---|---|")
    for xh, xt in zip(Ph["quintiles_longitud"], Pt["quintiles_longitud"]):
        w(f"| {xt['quintil']} | {_n(xt['desde'], 0)}–{_n(xt['hasta'], 1)} | {xt['n_videos']} | {_n(xh['exactitud'])} | {_n(xt['exactitud'])} |")
    w("")
    w("### 6.4 Manos en movimiento (una o dos)\n")
    w(f"Proxy por video con umbral {_n(UMBRAL_MANOS, 1)} sobre el cociente de recorridos de las muñecas (fijado mirando solo la "
      f"distribución del cociente, sin aciertos). {PR['videos_por_manos'].get('una', 0)} videos con una mano en movimiento y "
      f"{PR['videos_por_manos'].get('dos', 0)} con dos; por mayoría de sus videos, {len(PR['manos_por_sena']['una'])} señas son "
      f"«de una mano» y {len(PR['manos_por_sena']['dos'])} «de dos» ({PR['videos_cuya_etiqueta_difiere_de_su_sena']} videos "
      "difieren de la etiqueta de su seña). Señas «de dos manos»: "
      + ", ".join(SIGN_VOCABULARY[i] for i in PR["manos_por_sena"]["dos"]) + ". Es una medida del movimiento de las muñecas, "
      "no una anotación lingüística: una seña de una mano con la otra en apoyo que se desplaza puede salir «de dos».\n")
    w("| grupo | videos | exactitud heredada | exactitud TCN | solo TCN / solo heredada (voto) |")
    w("|---|---|---|---|---|")
    for gname, etqg in (("una", "una mano se mueve"), ("dos", "se mueven las dos")):
        tt = PR["transicion_por_manos"][gname]
        w(f"| {etqg} (por video) | {Pt['manos_video'][gname]['n'] // 3} | {_n(Ph['manos_video'][gname]['exactitud'])} | "
          f"{_n(Pt['manos_video'][gname]['exactitud'])} | {tt['solo_tcn']} / {tt['solo_heredada']} |")
    for gname, etqg in (("una", "seña «de una mano»"), ("dos", "seña «de dos manos»")):
        w(f"| {etqg} (por seña) | {Pt['manos_sena'][gname]['n'] // 3} | {_n(Ph['manos_sena'][gname]['exactitud'])} | "
          f"{_n(Pt['manos_sena'][gname]['exactitud'])} | — |")
    w("")
    w("Por señante (TCN, por video, una / dos): " + "; ".join(
        f"{x} {_n(Pt['manos_video_por_senante'][x]['una']['exactitud'], 2)} / {_n(Pt['manos_video_por_senante'][x]['dos']['exactitud'], 2)}"
        for x in SENANTES) + f". Entre las 50 señas, correlación entre «de dos manos» y el recall de la TCN: "
      f"{_n(Pt['entre_senas_manos']['pearson'], 3)} (p por permutación {_p(Pt['entre_senas_manos']['p_bilateral'])}); "
      f"heredada {_n(Ph['entre_senas_manos']['pearson'], 3)} (p {_p(Ph['entre_senas_manos']['p_bilateral'])}). Como la etiqueta "
      "es casi constante dentro de cada seña, esto compara señas, no videos de la misma seña.\n")
    w("### 6.5 Confianza de la predicción y top-3\n")
    w("| tramo de confianza | heredada: predicciones (%) | heredada: exactitud | TCN: predicciones (%) | TCN: exactitud |")
    w("|---|---|---|---|---|")
    for xh, xt in zip(Ph["tramos_confianza"], Pt["tramos_confianza"]):
        cierra = "]" if xt["hasta"] == 1.0 else ")"
        w(f"| [{_n(xt['desde'], 2)}; {_n(xt['hasta'], 2)}{cierra} | {xh['n']} ({_pct(xh['proporcion_de_predicciones'])}) | "
          f"{_n(xh['exactitud'])} | {xt['n']} ({_pct(xt['proporcion_de_predicciones'])}) | {_n(xt['exactitud'])} |")
    w("")
    fh, ft = Ph["fiabilidad_3_semillas"], Pt["fiabilidad_3_semillas"]
    w(f"- Confianza media frente a exactitud (3 semillas): TCN {_n(ft['confianza_media'])} frente a {_n(ft['exactitud'])} "
      f"(ECE {_n(ft['ece'])}; por semilla {' / '.join(_n(x) for x in Pt['ece_por_semilla'])}); heredada "
      f"{_n(fh['confianza_media'])} frente a {_n(fh['exactitud'])} (ECE {_n(fh['ece'])}). La TCN es, en promedio, más "
      "confiada que acertada; la heredada, al revés.")
    w(f"- Errores con confianza ≥ 0,75: TCN {Pt['errores_con_confianza_ge_0_75']} de {Pt['errores_3_semillas']} "
      f"({_pct(Pt['errores_con_confianza_ge_0_75'] / Pt['errores_3_semillas'])}); heredada {Ph['errores_con_confianza_ge_0_75']} de "
      f"{Ph['errores_3_semillas']} ({_pct(Ph['errores_con_confianza_ge_0_75'] / Ph['errores_3_semillas'])}). Con el umbral 0,00 "
      "aprobado ninguno se rechaza; esto describe el umbral, no es un criterio para cambiarlo.")
    w(f"- La seña correcta está en el top-3 en {Pt['errores_con_real_en_top3']} de los {Pt['errores_3_semillas']} errores de la TCN "
      f"({_pct(Pt['errores_con_real_en_top3'] / Pt['errores_3_semillas'])}; en segundo lugar en {Pt['errores_con_real_en_segundo_lugar']}); "
      f"heredada {Ph['errores_con_real_en_top3']} de {Ph['errores_3_semillas']} ({_pct(Ph['errores_con_real_en_top3'] / Ph['errores_3_semillas'])}).")
    w("- Figura: `4d_acierto_vs_propiedades.png` (a: mano ausente; b: duración; c: confianza; d: manos).\n")

    # ---------------- 7
    w("## 7. Consistencia entre semillas\n")
    w("Fuente: `consistencia`. Para cada video, cuántas de las 3 semillas lo fallan.\n")
    w("| red | 0 de 3 | 1 de 3 | 2 de 3 | 3 de 3 | de los que fallan las 3, misma seña errónea | fallos estables (3 de 3) / videos con algún fallo |")
    w("|---|---|---|---|---|---|---|")
    for mm in MODELOS:
        c = CO[mm]
        d = c["distribucion_fallos"]
        w(f"| {ETIQUETA[mm]} | {d['0']} | {d['1']} | {d['2']} | {d['3']} | {c['fallan_todas_misma_clase']} | "
          f"{_pct(c['proporcion_fallos_estables'])} |")
    w("")
    w("| señante | TCN: 0 / 1 / 2 / 3 semillas fallan | heredada: 0 / 1 / 2 / 3 |")
    w("|---|---|---|")
    for x in SENANTES:
        w(f"| {x} | {' / '.join(str(CO['tcn']['por_senante'][x][str(i)]) for i in range(4))} | "
          f"{' / '.join(str(CO['heredada']['por_senante'][x][str(i)]) for i in range(4))} |")
    w("")
    w("Señas con más videos fallados por las 3 semillas (TCN): " + "; ".join(
        f"{SIGN_VOCABULARY[k]} {n}" for k, n in CO["tcn"]["senas_con_mas_fallos_en_las_3"]) + ". Pares que las 3 semillas "
      "repiten exactamente (TCN): " + "; ".join(f"{SIGN_VOCABULARY[r]} → {SIGN_VOCABULARY[p_]} ×{n}" for r, p_, n in
                                                CO["tcn"]["pares_estables_3_de_3"][:8]) + ".\n")

    # ---------------- 8
    w("## 8. Los errores vistos por el conversor de glosas (modo «inventario», D-15)\n")
    w(f"Fuente: `conversor`. {CV['modo']}. En el modo inventario cada seña aislada da su enunciado fijo («Niño.», "
      "«¿Cómo?»), así que un error de reconocimiento produce un enunciado bien formado con otro contenido; el conversor "
      "no tiene cómo notarlo. Categorías: las 15 del conversor (`conversor_glosas.LEXICO`). Solo descriptivo.\n")
    w("| red (voto) | errores | misma categoría | cambia de categoría | esperado misma categoría al azar | cambia de modalidad (pregunta ↔ enunciado) | misma categoría y distinto género | misma categoría en grammar_mapper (8 categorías) |")
    w("|---|---|---|---|---|---|---|---|")
    for mm in MODELOS:
        c = CV["modelos"][mm]
        n_ = c["errores_voto"]
        w(f"| {ETIQUETA[mm]} | {n_} | {_fr(c['misma_categoria'], n_)} | {_fr(c['cambia_categoria'], n_)} | "
          f"{_pct(c['esperado_misma_categoria_al_azar'])} | {_fr(c['cambia_modalidad'], n_)} | "
          f"{c['misma_categoria_distinto_genero']} | {_fr(c['misma_categoria_heredada_grammar_mapper'], n_)} (azar "
          f"{_pct(c['esperado_misma_categoria_heredada_al_azar'])}) |")
    ct = CV["modelos"]["tcn"]
    w("")
    w(f"Con las 3 semillas (TCN): {ct['misma_categoria_3_semillas']} de {ct['errores_3_semillas']} errores "
      f"({_pct(ct['misma_categoria_3_semillas'] / ct['errores_3_semillas'])}) quedan en la misma categoría y "
      f"{ct['cambia_modalidad_3_semillas']} ({_pct(ct['cambia_modalidad_3_semillas'] / ct['errores_3_semillas'])}) cambian de modalidad.")
    w("- Cambios de categoría más frecuentes (TCN, voto): " + "; ".join(f"{x} → {y} ×{n}" for x, y, n in
                                                                       ct["transiciones_de_categoria_mas_frecuentes"][:8]) + ".")
    w("- Errores dentro de la misma categoría (TCN, voto), por categoría: " + "; ".join(
        f"{k} {v}" for k, v in ct["dentro_de_categoria_por_categoria"].items()) + ".")
    w("- Enunciados más frecuentes (dicho → mostrado; TCN, voto): " + "; ".join(
        f"«{x}» → «{y}» ×{n}" for x, y, n in ct["enunciados_mas_frecuentes"][:8]) + ".\n")

    # ---------------- 9
    w("## 9. Partición aleatoria (señantes vistos)\n")
    ta = AL["transicion"]
    w(f"Fuente: `aleatoria`. 100 videos de prueba (señantes {', '.join(f'{k}: {v}' for k, v in sorted(AL['senantes_en_prueba'].items()))}). "
      f"TCN 94 aciertos, heredada 88. Tabla por video: ambos aciertan {ta['ambos_aciertan']}, solo TCN {ta['solo_tcn']}, solo "
      f"heredada {ta['solo_heredada']}, ambos fallan {ta['ambos_fallan']} (McNemar p = {_pv(ta['p_mcnemar_exacto'])}: con 100 "
      "videos la diferencia no es distinguible del azar). Confianza media de la TCN: aciertos "
      f"{_n(AL['confianza_media_aciertos_tcn'])}, errores {_n(AL['confianza_media_errores_tcn'])}.\n")
    w("**Los 6 errores de la TCN** (`aleatoria.errores.tcn`; enunciado del conversor en modo inventario):\n")
    w("| video | señante | real → predicha | confianza | posición de la real en el top-3 | fotogramas | sin alguna mano | manos | heredada predice | enunciado dicho → mostrado |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    conv_al = {(e["real"], e["predicha"]): e for e in CV["aleatoria_tcn"]}
    for f in AL["errores"]["tcn"]:
        e = conv_al[(f["real"], f["predicha"])]
        w(f"| {f['muestra']} | {f['senante']} | {s(f['real'])} → {s(f['predicha'])} | {_n(f['confianza'], 2)} | "
          f"{f['posicion_real_en_top3'] or 'fuera'} | {f['longitud']} | {_pct(f['prop_sin_alguna_mano'])} | {f['manos']} | "
          f"{s(f['el_otro_modelo_predice'])} | «{e['enunciado_real']}» → «{e['enunciado_predicho']}» |")
    w("")
    w("Errores de la heredada en la misma partición: " + "; ".join(
        f"{f['muestra']} {SIGN_VOCABULARY[f['real']]} → {SIGN_VOCABULARY[f['predicha']]}" for f in AL["errores"]["heredada"]) + ".\n")

    # ---------------- 10
    w("## 10. Contraste con el informe de la heredada\n")
    if CP.get("disponible"):
        w(f"Fuente: `contraste_informe_previo` (informe previo: `{CP['fuente']}`, una sola semilla, la 2026). Parejas que el informe "
          "previo puso arriba, contadas ahora con 3 semillas (dos sentidos):\n")
        w("| pareja | informe previo (1 semilla) | heredada (3 semillas) | TCN (3 semillas) |")
        w("|---|---|---|---|")
        for f in CP["parejas"]:
            w(f"| {s(f['a'])} / {s(f['b'])} | {f['total_informe_previo']} | {f['heredada_3_semillas']} | {f['tcn_3_semillas']} |")
        w("")
        w("Peores 10 señas del informe previo y su recall ahora:\n")
        w("| seña | recall informe previo | heredada (3 semillas) | TCN (3 semillas) | puesto en la TCN (1 = peor) |")
        w("|---|---|---|---|---|")
        for f in CP["peores_senas"]:
            w(f"| {s(f['id'])} | {_n(f['recall_informe_previo'], 2)} | {_n(f['recall_heredada_3_semillas'], 2)} | "
              f"{_n(f['recall_tcn_3_semillas'], 2)} | {f['puesto_tcn_de_peor_a_mejor']} |")
        w("")
    w(_lectura_contraste(a))
    w("")

    # ---------------- 11
    w("## 11. Qué se puede y qué no se puede concluir\n")
    for linea in _limites(a):
        w(f"- {linea}")
    w("")
    w("## Para la tesis\n")
    for i, linea in enumerate(_hallazgos(a), 1):
        w(f"{i}. {linea}")
    w("")
    w("## Figuras\n")
    for f in a.get("figuras", []):
        w(f"- `{f}`")
    w("")
    w("## Reproducir\n")
    w(f"`{a.get('comando', 'python src/analisis_errores_tcn.py')}` (intérprete de venv312; lectura de `data/` en modo solo "
      "lectura; segundos de cómputo). Pruebas: `python -m unittest tests.test_analisis_errores_tcn -v`.\n")
    return "\n".join(L)


def _lectura_contraste(a: dict) -> str:
    CP, P = a["contraste_informe_previo"], a["pares"]
    if not CP.get("disponible"):
        return ""
    sigue = [f for f in CP["parejas"] if f["tcn_3_semillas"] >= 15]
    baja = [f for f in CP["parejas"] if f["tcn_3_semillas"] <= 0.5 * f["heredada_3_semillas"]]
    cv = a["conversor"]["modelos"]["tcn"]
    cvh = a["conversor"]["modelos"]["heredada"]
    previas = {(f["a"], f["b"]) for f in CP["parejas"]}
    nuevas = [p for p in P["tcn"]["parejas_3_semillas"][:12]
              if (p["a"], p["b"]) not in previas and [p["a"], p["b"]] not in P["parejas_top10_comunes"]][:4]
    return ("**Lectura.** Siguen difíciles con la TCN (≥ 15 errores en 3 semillas) las parejas "
            + "; ".join(f"{SIGN_VOCABULARY[f['a']]} / {SIGN_VOCABULARY[f['b']]} ({f['tcn_3_semillas']})" for f in sigue)
            + ": todas dentro de la misma categoría de grammar_mapper (sustantivos de persona, interrogativos, adverbios). "
            "Bajan a la mitad o menos: "
            + "; ".join(f"{SIGN_VOCABULARY[f['a']]} / {SIGN_VOCABULARY[f['b']]} ({f['heredada_3_semillas']} → {f['tcn_3_semillas']})" for f in baja)
            + ". Aparecen arriba con la TCN parejas que el informe previo no destacaba (p. ej. "
            + "; ".join(f"{SIGN_VOCABULARY[p['a']]} / {SIGN_VOCABULARY[p['b']]} ({p['total']})" for p in nuevas)
            + "). Errores dentro de la misma categoría de grammar_mapper: "
            f"{_pct(CP['proporcion_dentro_categoria_informe_previo'])} en el informe previo (1 semilla); con el voto, "
            f"{_pct(cvh['misma_categoria_heredada_grammar_mapper'] / cvh['errores_voto'])} la heredada y "
            f"{_pct(cv['misma_categoria_heredada_grammar_mapper'] / cv['errores_voto'])} la TCN: casi igual, siempre muy por encima "
            "del azar (≈ 13,6 %). El señante 0000 sigue siendo el peor. La asociación con la mano ausente que encontró el informe "
            "previo se mantiene con la heredada (voto) también dentro de la misma seña, y con la TCN solo dentro de señante (§6.1).")


def _limites(a: dict) -> list[str]:
    return [
        "Hay 5 señantes. Todo lo que se diga «por señante» son 5 casos: no se puede estimar qué propiedad del señante "
        "explica su exactitud, ni generalizar a señantes nuevos con un intervalo útil.",
        "Cada seña tiene 4 videos por señante (3 en tres casos): diferencias de recall por seña de 1-2 videos, o de "
        "0,05-0,10 en el agregado, no deben leerse como efectos.",
        "Las relaciones con la duración, la mano ausente y las manos en movimiento son asociaciones observadas, no causas. "
        "Las tres variables cambian con la seña y con el señante; por eso se comparan dentro de estrato, y aun así un "
        "tercer factor (la forma de la seña, la cámara, el ritmo del señante) puede explicar ambas cosas.",
        "La comparación TCN frente a heredada cambia a la vez la arquitectura y la regla de mano ausente. Solo T1 frente a "
        "T0 (§6.2) aísla la imputación, y sus celdas por tramo son pequeñas y se miraron después de ver los datos.",
        "El voto de 3 semillas es un recurso de análisis; el prototipo usa un solo modelo (semilla 2030, partición "
        "aleatoria). Las cifras del modelo desplegado son las de la partición aleatoria (§9) y, para señantes nuevos, la "
        "media de pliegues.",
        "El proxy «una / dos manos» mide el recorrido de las muñecas, no la estructura lingüística de la seña; no fue "
        "revisado por una persona señante.",
        "La lectura del conversor es descriptiva: dice qué enunciado vería el usuario, no cuánto molesta cada error.",
        "Nada de esto se usó para elegir el modelo final (D-14) ni el umbral; no se toma ninguna decisión a partir de "
        "este análisis.",
    ]


def _hallazgos(a: dict) -> list[str]:
    rt = a["controles"]["resumen"]["tcn"]
    TR, CO, PR, CV = a["transicion"]["global"], a["consistencia"], a["propiedades"], a["conversor"]["modelos"]["tcn"]
    Pt = PR["modelos"]["tcn"]
    P = a["pares"]["tcn"]["parejas_3_semillas"]
    T = a["transicion"]
    pos_sen = sum(1 for v in T["por_senante"].values() if v["neto_tcn"] > 0)
    pos_cat = sum(1 for v in T["por_categoria"].values() if v["neto_tcn"] > 0)
    top3_comunes = sum(1 for p in P[:3] if [p["a"], p["b"]] in a["pares"]["parejas_top10_comunes"])
    una_menor = sum(1 for x in SENANTES if Pt["manos_video_por_senante"][x]["una"]["exactitud"]
                    < Pt["manos_video_por_senante"][x]["dos"]["exactitud"])
    return [
        f"La TCN corrige {TR['solo_tcn']} videos que la heredada fallaba e introduce {TR['solo_heredada']} errores nuevos "
        f"(voto de 3 semillas, McNemar p = {_pv(TR['p_mcnemar_exacto'])}); el saldo es positivo en {pos_sen} de 5 señantes y "
        f"en {pos_cat} de 8 categorías gramaticales.",
        f"Los errores que quedan se concentran en parejas de la misma clase (interrogativos; niño / niña): {SIGN_VOCABULARY[P[0]['a']]} / "
        f"{SIGN_VOCABULARY[P[0]['b']]} ({P[0]['total']} errores en 3 semillas), {SIGN_VOCABULARY[P[1]['a']]} / "
        f"{SIGN_VOCABULARY[P[1]['b']]} ({P[1]['total']}) y {SIGN_VOCABULARY[P[2]['a']]} / {SIGN_VOCABULARY[P[2]['b']]} "
        f"({P[2]['total']}); {top3_comunes} de esas 3 parejas están también entre las 10 más confundidas por la heredada.",
        f"El señante 0000 sigue siendo el peor ({_n(rt['senante_0000'])} frente a 0,452 de la heredada) y es donde más "
        f"discrepan las semillas ({CO['tcn']['por_senante']['0000']['1']} + {CO['tcn']['por_senante']['0000']['2']} de sus 200 "
        "videos los fallan 1 o 2 de las 3 semillas).",
        f"Las señas en que se mueve una sola mano se aciertan menos que las de dos manos ({_n(Pt['manos_video']['una']['exactitud'], 2)} "
        f"frente a {_n(Pt['manos_video']['dos']['exactitud'], 2)} con la TCN; menor en {una_menor} de 5 señantes), y es allí donde la TCN más "
        f"mejora a la heredada (corrige {PR['transicion_por_manos']['una']['solo_tcn']} videos e introduce "
        f"{PR['transicion_por_manos']['una']['solo_heredada']} errores; con dos manos, {PR['transicion_por_manos']['dos']['solo_tcn']} "
        f"y {PR['transicion_por_manos']['dos']['solo_heredada']}).",
        f"La mano ausente se asocia con el error dentro de cada señante (+{_n(100 * Pt['pruebas_voto']['prop_sin_mano_activa']['senante']['diferencia_dentro_de_estratos'], 1)} "
        f"pp de fotogramas sin la mano que más se mueve en los errores, p Holm {_p(Pt['pruebas_voto']['prop_sin_mano_activa']['senante']['p_holm'])}), "
        "pero no al comparar videos de la misma seña: con la TCN no hay evidencia de que un video con más fotogramas "
        "perdidos falle más que otro de la misma seña.",
        f"La TCN es más confiada que acertada (confianza media {_n(Pt['fiabilidad_3_semillas']['confianza_media'])} frente a "
        f"exactitud {_n(Pt['fiabilidad_3_semillas']['exactitud'])}; la heredada era menos confiada que acertada): "
        f"{_pct(Pt['errores_con_confianza_ge_0_75'] / Pt['errores_3_semillas'])} de sus errores llegan con confianza ≥ 0,75 "
        f"(heredada {_pct(a['propiedades']['modelos']['heredada']['errores_con_confianza_ge_0_75'] / a['propiedades']['modelos']['heredada']['errores_3_semillas'])}).",
        f"En {_pct(Pt['errores_con_real_en_top3'] / Pt['errores_3_semillas'])} de los errores la seña correcta está entre las "
        f"3 primeras (heredada {_pct(a['propiedades']['modelos']['heredada']['errores_con_real_en_top3'] / a['propiedades']['modelos']['heredada']['errores_3_semillas'])}).",
        f"Vistos por el conversor (modo inventario), {_pct(CV['cambia_categoria'] / CV['errores_voto'])} de los errores cambian "
        f"de categoría (p. ej. adjetivo por sustantivo de persona; los que la mantienen, {_pct(CV['misma_categoria'] / CV['errores_voto'])}, "
        f"triplican lo esperable al azar, {_pct(CV['esperado_misma_categoria_al_azar'])}) y {_pct(CV['cambia_modalidad'] / CV['errores_voto'])} convierten "
        "una pregunta en enunciado o al revés; el enunciado sale bien formado, de modo que el usuario no ve señal de error.",
    ]


# ---------------------------------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------------------------------

def _publico(x):
    """Quita las claves internas (prefijo «_») antes de escribir el JSON."""
    if isinstance(x, dict):
        return {k: _publico(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, list):
        return [_publico(v) for v in x]
    if isinstance(x, tuple):
        return [_publico(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--permutaciones", type=int, default=10000)
    ap.add_argument("--semilla", type=int, default=2026)
    ap.add_argument("--sin-figuras", action="store_true")
    ap.add_argument("--salida-md", type=Path, default=SALIDA_MD)
    ap.add_argument("--salida-json", type=Path, default=SALIDA_JSON)
    ap.add_argument("--figuras", type=Path, default=DIR_FIGURAS)
    args = ap.parse_args(argv)
    datos_dir = (BASE_DIR / "data").resolve()
    for p in (args.salida_md, args.salida_json, args.figuras):
        r = p.resolve()
        if r == datos_dir or datos_dir in r.parents:
            ap.error("ninguna salida puede estar dentro de data/")
    a, datos = analizar(args.permutaciones, args.semilla)
    a["comando"] = " ".join(["python", "src/analisis_errores_tcn.py", *(sys.argv[1:] if argv is None else argv)])
    a["figuras"] = [] if args.sin_figuras else figuras(a, datos, args.figuras)
    md = informe_md(a)
    args.salida_json.write_text(json.dumps(_publico(a), ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    args.salida_md.write_text(md, encoding="utf-8")
    print(f"controles: {'todo coincide' if a['controles']['todo_coincide'] else 'DIFERENCIAS'}; "
          f"escrito {ae._rel(args.salida_md)}, {ae._rel(args.salida_json)}, {len(a['figuras'])} figuras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
