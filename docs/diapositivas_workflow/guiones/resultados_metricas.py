"""
Pieza «resultados» de las diapositivas del flujo (docs/diapositivas_workflow).

Recalcula, SOLO desde predicciones ya guardadas (no entrena ni carga modelos):
  * LOSO «manos» (models/loso_2026-09-15_holistic_mascara_manos/pliegue_k/predicciones.json):
    top-1, top-3, precisión/recall/F1 macro por pliegue (= por persona excluida), media ± DE
    (ddof=1) entre pliegues y agregado sobre las 997 predicciones; tabla por seña (TP, FN, FP,
    recall, precisión, F1); parejas más confundidas; efecto del umbral 0,20 / 0,55.
  * Esquema aleatorio (personas conocidas), modelo canónico: seleccionado y ensemble de 5 semillas
    (models/canonico_holistic_mascara_manos/evaluacion_*/predicciones.json); validación por semilla
    (resultados.json).
  * Línea base clásica (models/linea_base_4d_2026-09-15/predicciones_*.json).
Compara cada cifra recalculada con la guardada y falla si no coincide.

Uso (desde la raíz del repositorio):
  python docs/diapositivas_workflow/guiones/resultados_metricas.py <salida.json>
Escribe solo <salida.json>. Las figuras las hace figuras_resultados.py a partir de ese JSON.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, ".")
from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402

N = 50
LOSO = Path("models/loso_2026-09-15_holistic_mascara_manos")
CANON = Path("models/canonico_holistic_mascara_manos")
LB = Path("models/linea_base_4d_2026-09-15")


def cargar(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def metricas(preds):
    y = np.array([r["real"] for r in preds])
    yp = np.array([r["predicha"] for r in preds])
    t3 = np.array([r["real"] in r["top3"] for r in preds])
    p, r, f, _ = precision_recall_fscore_support(y, yp, labels=list(range(N)), average="macro",
                                                 zero_division=0)
    return {"n": int(len(y)), "top1": float((y == yp).mean()), "top3": float(t3.mean()),
            "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
            "aciertos": int((y == yp).sum()), "errores": int((y != yp).sum())}


def resumen(vals):
    a = np.array(vals, dtype=float)
    return {"media": float(a.mean()), "de": float(a.std(ddof=1)), "min": float(a.min()),
            "max": float(a.max()), "valores": [float(v) for v in a]}


def cerca(a, b, tol=1e-9):
    return abs(a - b) <= tol


def main(salida):
    out = {"script": "docs/diapositivas_workflow/guiones/resultados_metricas.py", "controles": {}}
    res_loso = cargar(LOSO / "resultados_loso.json")

    # ---------------- LOSO por pliegue ----------------
    todos = []
    pliegues = []
    for pl in res_loso["pliegues"]:
        k = pl["pliegue"]
        preds = cargar(LOSO / f"pliegue_{k}" / "predicciones.json")
        # nombre de muestra = seña_señante_repetición; el señante debe ser el excluido
        sen = {r["muestra"].split("_")[1] for r in preds}
        assert sen == {pl["senante_test"]}, (k, sen)
        assert all(int(r["muestra"].split("_")[0]) == r["real"] for r in preds)
        m = metricas(preds)
        for clave in ("top1", "top3", "precision_macro", "recall_macro", "f1_macro"):
            assert cerca(m[clave], pl[clave]), (k, clave, m[clave], pl[clave])
        m.update({"pliegue": k, "senante": pl["senante_test"], "n_train_original": pl["n_train_original"],
                  "n_val": pl["n_val"], "mejor_val_accuracy": pl["mejor_val_accuracy"]})
        pliegues.append(m)
        for r in preds:
            r = dict(r, senante=pl["senante_test"])
            todos.append(r)
    out["controles"]["loso_por_pliegue_coincide_con_resultados_loso.json"] = True
    out["loso_por_pliegue"] = pliegues
    out["loso_media_de"] = {c: resumen([p[c] for p in pliegues])
                            for c in ("top1", "top3", "precision_macro", "recall_macro", "f1_macro")}
    for c in ("top1", "top3", "precision_macro", "recall_macro", "f1_macro"):
        g = res_loso["agregado"][c]
        assert cerca(out["loso_media_de"][c]["media"], g["media"]) and cerca(out["loso_media_de"][c]["de"], g["desviacion_estandar"])
    out["controles"]["loso_media_de_coincide_con_resultados_loso.json:agregado"] = True
    out["loso_validacion_por_pliegue"] = resumen([p["mejor_val_accuracy"] for p in pliegues])

    ag = metricas(todos)
    out["loso_agregado_997"] = ag
    an = cargar("models/analisis_errores_manos_2026-09-15/analisis.json")
    assert cerca(ag["top1"], an["global"]["top1"]) and cerca(ag["top3"], an["global"]["top3"])
    out["controles"]["loso_agregado_coincide_con_analisis.json:global"] = True
    assert len(todos) == 997

    # ---------------- Por seña (agregado 997) ----------------
    y = np.array([r["real"] for r in todos])
    yp = np.array([r["predicha"] for r in todos])
    cm = np.zeros((N, N), dtype=int)
    for a, b in zip(y, yp):
        cm[a, b] += 1
    por_sena = []
    for c in range(N):
        tp = int(cm[c, c]); fn = int(cm[c].sum() - tp); fp = int(cm[:, c].sum() - tp)
        rec = tp / (tp + fn) if tp + fn else 0.0
        pre = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
        tn = int(len(y) - tp - fn - fp)
        por_sena.append({"id": c, "glosa": SIGN_VOCABULARY[c], "soporte": tp + fn, "TP": tp, "FN": fn,
                         "FP": fp, "TN": tn, "predicha_veces": tp + fp, "recall": rec, "precision": pre,
                         "f1": f1})
    # control contra analisis.json
    an_s = {s["id"]: s for s in an["por_sena"]["senas"]}
    for s in por_sena:
        a = an_s[s["id"]]
        assert s["TP"] == a["aciertos"] and s["soporte"] == a["soporte"] and s["predicha_veces"] == a["n_predichas"]
        assert cerca(s["recall"], a["recall"]) and cerca(s["precision"], a["precision"])
    out["controles"]["por_sena_coincide_con_analisis.json:por_sena"] = True
    out["por_sena"] = por_sena
    orden = sorted(por_sena, key=lambda s: (-s["recall"], -s["f1"], s["id"]))
    out["mejores_5_por_recall"] = orden[:5]
    out["peores_5_por_recall"] = sorted(por_sena, key=lambda s: (s["recall"], s["f1"], s["id"]))[:5]
    out["senas_recall_1"] = [s["glosa"] for s in por_sena if s["recall"] == 1.0]
    out["senas_recall_ge_0_8"] = [s["glosa"] for s in por_sena if s["recall"] >= 0.8]
    out["senas_recall_le_0_25"] = [s["glosa"] for s in por_sena if s["recall"] <= 0.25]
    out["matriz_confusion_997"] = cm.tolist()

    # parejas confundidas (dirigidas y no dirigidas)
    pares = []
    for a in range(N):
        for b in range(N):
            if a != b and cm[a, b] > 0:
                pares.append({"real": SIGN_VOCABULARY[a], "predicha": SIGN_VOCABULARY[b], "conteo": int(cm[a, b]),
                              "inverso": int(cm[b, a])})
    pares.sort(key=lambda p: -p["conteo"])
    out["pares_dirigidos_top12"] = pares[:12]
    nd = []
    for a in range(N):
        for b in range(a + 1, N):
            t = int(cm[a, b] + cm[b, a])
            if t > 0:
                nd.append({"a": SIGN_VOCABULARY[a], "b": SIGN_VOCABULARY[b], "a_como_b": int(cm[a, b]),
                           "b_como_a": int(cm[b, a]), "total": t})
    nd.sort(key=lambda p: -p["total"])
    out["pares_no_dirigidos_top10"] = nd[:10]
    out["n_errores_997"] = int((y != yp).sum())

    # ejemplo «hola»
    ih = SIGN_VOCABULARY and [k for k, v in SIGN_VOCABULARY.items() if v == "hola"][0]
    fila = cm[ih]; col = cm[:, ih]
    out["ejemplo_hola"] = {
        **por_sena[ih],
        "fn_como": {SIGN_VOCABULARY[j]: int(fila[j]) for j in np.argsort(-fila) if j != ih and fila[j] > 0},
        "fp_desde": {SIGN_VOCABULARY[j]: int(col[j]) for j in np.argsort(-col) if j != ih and col[j] > 0},
        "tp_por_senante": {s: int(sum(1 for r in todos if r["senante"] == s and r["real"] == ih and r["predicha"] == ih))
                           for s in ["0000", "0001", "0002", "0003", "0004"]},
    }

    # ---------------- Umbral ----------------
    conf = np.array([r["prob_predicha"] for r in todos])
    ok = y == yp
    sen_arr = np.array([r["senante"] for r in todos])
    def efecto(t, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        acc = m & (conf >= t)
        n_acc = int(acc.sum())
        return {"umbral": t, "n": int(m.sum()), "aceptadas": n_acc,
                "rechazo": float(1 - n_acc / m.sum()),
                "aceptadas_correctas": int((acc & ok).sum()), "aceptadas_incorrectas": int((acc & ~ok).sum()),
                "exactitud_entre_aceptadas": float((acc & ok).sum() / n_acc) if n_acc else None,
                "exactitud_sin_umbral": float(ok[m].mean())}
    out["umbral"] = {"0.20": efecto(0.20), "0.55": efecto(0.55), "0.50": efecto(0.50),
                     "por_senante": {s: {"0.20": efecto(0.20, sen_arr == s), "0.55": efecto(0.55, sen_arr == s)}
                                     for s in ["0000", "0001", "0002", "0003", "0004"]}}
    assert out["umbral"]["0.20"]["aceptadas_correctas"] == 540 and out["umbral"]["0.20"]["aceptadas_incorrectas"] == 297
    ua = cargar("models/umbral_anidado_2026-09-22/resultados_anidado.json")
    ep = ua["efecto_prueba_loso"]
    out["umbral"]["guardado_umbral_anidado"] = {k: {kk: vv for kk, vv in v.items() if kk != "por_pliegue"}
                                                if isinstance(v, dict) else v for k, v in ep.items()}
    out["controles"]["umbral_0.20_reproduce_N-33_540_297"] = True

    # ---------------- Esquema aleatorio: canónico ----------------
    rc = cargar(CANON / "resultados.json")
    out["canonico_validacion_por_semilla"] = [
        {"semilla": s["semilla"], **s["metricas_validacion_checkpoint"], "epocas": s["epocas_ejecutadas"],
         "mejor_epoca": s["mejor_epoca"]} for s in rc["semillas"]]
    out["canonico_validacion_resumen"] = {c: resumen([s["metricas_validacion_checkpoint"][c] for s in rc["semillas"]])
                                          for c in ("top1", "top3", "f1_macro")}
    for sis in ("evaluacion_modelo_seleccionado", "evaluacion_ensemble"):
        pr = cargar(CANON / sis / "predicciones.json")
        m = metricas(pr)
        g = cargar(CANON / sis / "metricas.json")
        for c in ("top1", "top3", "precision_macro", "recall_macro", "f1_macro"):
            assert cerca(m[c], g[c]), (sis, c)
        m["senantes_en_prueba"] = dict(Counter(r["muestra"].split("_")[1] for r in pr))
        out[f"canonico_{sis}"] = m
    out["controles"]["canonico_coincide_con_metricas.json"] = True
    part = cargar("data/partitions/aleatoria_80_10_10.json")
    out["aleatoria_conteos"] = part["conteos"]

    # ---------------- Línea base ----------------
    lb = {}
    for fam in ("random_forest", "svm_rbf", "mlp"):
        lb[fam] = {"loso": metricas(cargar(LB / f"predicciones_{fam}.json")),
                   "aleatoria": metricas(cargar(LB / f"predicciones_{fam}_aleatoria.json"))}
    out["linea_base"] = lb
    ctrl = cargar(LB / "control_aumentacion_completa" / "resultados.json") if (LB / "control_aumentacion_completa" / "resultados.json").exists() else None
    out["linea_base_control_aumentacion_disponible"] = ctrl is not None

    Path(salida).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OK", salida)


if __name__ == "__main__":
    main(sys.argv[1])
