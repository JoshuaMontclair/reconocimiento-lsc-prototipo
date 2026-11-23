"""
Experimento: número de pasos temporales del remuestreo (seq_length) en el LOSO de 5 pliegues.

Reutiliza SIN modificar las piezas canónicas:
  - train_loso.cargar_datos, verificar_particion, importar_pipeline_canonico, remuestrear_todas y
    entrenar_y_evaluar (la misma función que produjo la referencia de N-30, 0,580 con 30 pasos);
  - train_final_100.resample_sequence, add_velocity y build_model (a través de train_loso).
Lo único que cambia es la longitud L con que se remuestrea cada seña. train_loso.py y
train_final_100.py no se tocan: entrenar_y_evaluar construye el modelo con X_seq.shape[1], así que
basta con pasarle las secuencias remuestreadas a L. La procedencia que escribe junto al modelo
(que lleva SEQ_LENGTH = 30 fijo) se reescribe después con la L real.

Criterio de elección (prerregistrado en docs/EXPERIMENTO_PASOS_TEMPORALES.md §2): SOLO la exactitud
de validación del LOSO; la prueba se reporta como descripción.

Subcomandos:
  entrenar   corre las combinaciones (semilla, L, pliegue) que falten; reanudable: salta las que
             ya tienen metricas_pasos.json.
  inferencia mide el tiempo de inferencia por muestra de cada modelo entrenado (en silencio, sin
             entrenamientos en paralelo).
  resumen    agrega todo, aplica el criterio y escribe resultados_pasos.json.

Uso:
  python src/experimento_pasos.py entrenar --pasos 30 --semillas 2026 --folds 0,    # medir el pliegue 0
  python src/experimento_pasos.py entrenar --pasos 15,20,30,45,60 --semillas 2026,2027
  python src/experimento_pasos.py inferencia
  python src/experimento_pasos.py resumen
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import train_loso as tl  # noqa: E402

PROCESSED_CANONICO = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
REPRESENTACION_CANONICA = "manos"
DEFAULT_OUT = BASE_DIR / "models" / "experimento_pasos_2026-09-30"
REFERENCIA_N30 = BASE_DIR / "models" / "loso_2026-09-15_holistic_mascara_manos" / "resultados_loso.json"
PASOS_REFERENCIA = 30


def _lista_int(texto):
    return [int(t) for t in str(texto).split(",") if t.strip() != ""]


def dir_corrida(out, L, semilla, k):
    return Path(out) / f"pasos_{L:02d}" / f"semilla_{semilla}" / f"pliegue_{k}"


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def cargar_todo(processed, representacion):
    X_padded, y, lengths, nombres, senantes = tl.cargar_datos(processed, representacion)
    with open(tl.ARCHIVO_LOSO, encoding="utf-8") as f:
        loso = json.load(f)
    n_clases = int(y.max()) + 1
    for p in loso["pliegues"]:
        tl.verificar_particion(p["conjuntos"], nombres, senante_test=p["senante_test"],
                               n_clases=n_clases)
    pos = {n: i for i, n in enumerate(nombres)}
    return X_padded, y, lengths, nombres, loso, pos


def metricas_validacion(model, X_val_f, y_val):
    from sklearn.metrics import f1_score
    probs = model.predict(X_val_f, verbose=0)
    pred = np.argmax(probs, axis=1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    return {
        "val_top1": float(np.mean(pred == y_val)),
        "val_top3": float(np.mean(np.any(top3 == y_val[:, None], axis=1))),
        "val_f1_macro": float(f1_score(y_val, pred, labels=list(range(probs.shape[1])),
                                       average="macro", zero_division=0)),
    }


def cmd_entrenar(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    processed = Path(args.processed).resolve()
    X_padded, y, lengths, nombres, loso, pos = cargar_todo(processed, args.representacion)
    canon = tl.importar_pipeline_canonico(args.representacion)
    tf, _, add_velocity, resample_sequence, _ = canon
    nombres_senas = tl.cargar_nombres_senas()
    folds = tl.parsear_folds(args.folds, len(loso["pliegues"]))
    pasos = _lista_int(args.pasos)
    semillas = _lista_int(args.semillas)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] experimento_pasos entrenar | "
          f"pasos {pasos} | semillas {semillas} | pliegues {folds} | processed {tl._rel(processed)} | "
          f"representacion {args.representacion} | salida {tl._rel(out)}", flush=True)

    # Orden: semilla -> L -> pliegue. Así, si hay que cortar, cada semilla queda completa por L.
    for semilla in semillas:
        for L in pasos:
            X_seq = tl.remuestrear_todas(X_padded, lengths, resample_sequence, L)
            for k in folds:
                d = dir_corrida(out, L, semilla, k)
                if (d / "metricas_pasos.json").exists():
                    print(f"  ya existe {tl._rel(d)}; se salta", flush=True)
                    continue
                p = loso["pliegues"][k]
                print(f"\n{'=' * 72}\n  L={L} | semilla {semilla} | pliegue {k} (senante "
                      f"{p['senante_test']}) | {datetime.now().isoformat(timespec='seconds')}\n"
                      f"{'=' * 72}", flush=True)
                ns = argparse.Namespace(
                    seed=semilla, aug=args.aug, epochs=args.epochs, batch_size=args.batch_size,
                    patience=args.patience, dropout=args.dropout, processed=str(processed),
                    representacion=args.representacion)
                m = tl.entrenar_y_evaluar(f"L{L}_s{semilla}_pliegue_{k}", p["conjuntos"], pos,
                                          X_seq, y, ns, d, canon, nombres_senas)
                # Procedencia con la L real (entrenar_y_evaluar escribe SEQ_LENGTH = 30).
                from src.representacion import escribir_procedencia
                escribir_procedencia(d / "modelo.keras", processed, args.representacion, L,
                                     "src/experimento_pasos.py (via train_loso.entrenar_y_evaluar)",
                                     BASE_DIR)
                # Métricas de validación del checkpoint (las que deciden).
                model = tf.keras.models.load_model(str(d / "modelo.keras"))
                idx_val = np.array([pos[n] for n in p["conjuntos"]["val"]], dtype=np.int64)
                mv = metricas_validacion(model, add_velocity(X_seq[idx_val]), y[idx_val])
                del model
                tf.keras.backend.clear_session()
                m.update(mv)
                m.update({"pasos": L, "semilla": semilla, "pliegue": k,
                          "senante_test": p["senante_test"],
                          "fecha_fin": datetime.now().isoformat(timespec="seconds")})
                with open(d / "metricas_pasos.json", "w", encoding="utf-8") as f:
                    json.dump(m, f, indent=1)
                print(f"  -> L={L} s={semilla} k={k}: val {m['mejor_val_accuracy']:.3f} "
                      f"(checkpoint {mv['val_top1']:.3f}) | test top-1 {m['top1']:.3f} | "
                      f"epocas {m['epocas_ejecutadas']} | {m['tiempo_total_s']:.0f} s", flush=True)
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] fin", flush=True)


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

def cmd_inferencia(args):
    """Tiempo por muestra como lo hace el prototipo (translator.py: model.predict de un lote de 1),
    más el costo por muestra en lote (todo el conjunto de prueba de una vez). Mediana de
    --repeticiones llamadas tras --calentamiento llamadas de calentamiento."""
    out = Path(args.out).resolve()
    canon = tl.importar_pipeline_canonico(REPRESENTACION_CANONICA)
    tf = canon[0]
    rng = np.random.default_rng(0)
    filas = []
    for dL in sorted(out.glob("pasos_*")):
        L = int(dL.name.split("_")[1])
        for ruta in sorted(dL.glob("semilla_*/pliegue_*/modelo.keras")):
            model = tf.keras.models.load_model(str(ruta))
            n_feat = int(model.input_shape[-1])
            x1 = rng.standard_normal((1, L, n_feat)).astype(np.float32)
            xb = rng.standard_normal((200, L, n_feat)).astype(np.float32)
            for _ in range(args.calentamiento):
                model.predict(x1, verbose=0)
            t = []
            for _ in range(args.repeticiones):
                t0 = time.perf_counter()
                model.predict(x1, verbose=0)
                t.append((time.perf_counter() - t0) * 1000)
            model.predict(xb, verbose=0)
            tb = []
            for _ in range(5):
                t0 = time.perf_counter()
                model.predict(xb, verbose=0)
                tb.append((time.perf_counter() - t0) * 1000 / len(xb))
            fila = {"pasos": L, "modelo": tl._rel(ruta), "n_parametros": int(model.count_params()),
                    "ms_lote1_mediana": float(np.median(t)), "ms_lote1_p90": float(np.percentile(t, 90)),
                    "ms_por_muestra_lote200_mediana": float(np.median(tb))}
            filas.append(fila)
            print(f"  L={L:2d} {tl._rel(ruta)}: lote 1 mediana {fila['ms_lote1_mediana']:.2f} ms "
                  f"(p90 {fila['ms_lote1_p90']:.2f}) | lote 200 {fila['ms_por_muestra_lote200_mediana']:.3f} "
                  f"ms/muestra", flush=True)
            del model
            tf.keras.backend.clear_session()
    with open(out / "inferencia.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": datetime.now().isoformat(timespec="seconds"),
                   "metodo": f"model.predict(verbose=0) de un lote de 1 como translator.py; "
                             f"{args.calentamiento} llamadas de calentamiento y {args.repeticiones} "
                             f"medidas por modelo; ademas lote de 200 (5 medidas). CPU, sin otros "
                             f"entrenamientos corriendo. Solo la red: sin MediaPipe ni remuestreo.",
                   "entorno": tl.info_entorno(), "modelos": filas}, f, indent=1)


# ---------------------------------------------------------------------------
# Resumen y criterio
# ---------------------------------------------------------------------------

def _media_de(v):
    v = [float(x) for x in v]
    return {"media": float(np.mean(v)), "de": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "valores": v}


def pruebas_pareadas(dif):
    """Prueba de signos (binomial exacta, empates fuera) y Wilcoxon exacto sobre las diferencias."""
    from scipy.stats import binomtest, wilcoxon
    # Las exactitudes vienen de float32 (múltiplos de 0,01 promediados entre semillas): dos medias
    # iguales pueden diferir en ~1e-8. Se redondea a 6 decimales para que un empate cuente como
    # empate (criterio §2.4 b) y no como mejora.
    dif = np.round(np.asarray(dif, dtype=float), 6)
    pos_, neg_ = int(np.sum(dif > 0)), int(np.sum(dif < 0))
    res = {"diferencias": dif.tolist(), "mejora": pos_, "empeora": neg_,
           "empate": int(len(dif) - pos_ - neg_)}
    if pos_ + neg_ > 0:
        res["signos_p_bilateral"] = float(binomtest(pos_, pos_ + neg_, 0.5).pvalue)
        res["signos_p_unilateral_mejora"] = float(binomtest(pos_, pos_ + neg_, 0.5,
                                                            alternative="greater").pvalue)
    else:
        res["signos_p_bilateral"] = res["signos_p_unilateral_mejora"] = 1.0
    if np.any(dif != 0):
        try:
            w = wilcoxon(dif, zero_method="zsplit", method="exact")
        except Exception:
            w = wilcoxon(dif, zero_method="zsplit")
        res["wilcoxon_estadistico"] = float(w.statistic)
        res["wilcoxon_p_bilateral"] = float(w.pvalue)
    else:
        res["wilcoxon_estadistico"] = None
        res["wilcoxon_p_bilateral"] = 1.0
    return res


def cmd_resumen(args):
    out = Path(args.out).resolve()
    filas = [json.loads(p.read_text()) for p in sorted(out.glob("pasos_*/semilla_*/pliegue_*/metricas_pasos.json"))]
    if not filas:
        raise SystemExit("no hay corridas")
    pasos = sorted({f["pasos"] for f in filas})
    semillas = sorted({f["semilla"] for f in filas})
    folds = sorted({f["pliegue"] for f in filas})
    tab = {(f["pasos"], f["semilla"], f["pliegue"]): f for f in filas}
    completos = {L: [s for s in semillas if all((L, s, k) in tab for k in folds)] for L in pasos}
    # Solo entran al criterio las semillas completas en TODOS los L comparados.
    sem_comunes = [s for s in semillas if all(s in completos[L] for L in pasos)]
    print(f"pasos {pasos} | semillas {semillas} | completas en todos los L: {sem_comunes} | pliegues {folds}")

    inferencia = {}
    if (out / "inferencia.json").exists():
        inf = json.loads((out / "inferencia.json").read_text())
        for L in pasos:
            v = [m for m in inf["modelos"] if m["pasos"] == L]
            if v:
                inferencia[L] = {
                    "ms_lote1_mediana": float(np.median([m["ms_lote1_mediana"] for m in v])),
                    "ms_lote1_p90_mediana": float(np.median([m["ms_lote1_p90"] for m in v])),
                    "ms_por_muestra_lote200": float(np.median([m["ms_por_muestra_lote200_mediana"] for m in v])),
                    "n_parametros": int(v[0]["n_parametros"]), "n_modelos": len(v)}

    resumen = {}
    for L in pasos:
        ss = sem_comunes if sem_comunes else completos[L]
        r = {"semillas": ss}
        for met in ("mejor_val_accuracy", "val_top3", "val_f1_macro", "top1", "top3", "f1_macro",
                    "epocas_ejecutadas", "mejor_epoca", "tiempo_total_s"):
            por_pliegue = [float(np.mean([tab[(L, s, k)][met] for s in ss])) for k in folds]
            por_semilla = {s: float(np.mean([tab[(L, s, k)][met] for k in folds])) for s in ss}
            r[met] = {"por_pliegue_media_semillas": por_pliegue,
                      "entre_pliegues": _media_de(por_pliegue),
                      "por_semilla_media_pliegues": por_semilla,
                      "por_semilla_por_pliegue": {s: [tab[(L, s, k)][met] for k in folds] for s in ss}}
            if len(ss) > 1:
                r[met]["de_entre_semillas"] = float(np.std(list(por_semilla.values()), ddof=1))
        if L in inferencia:
            r["inferencia"] = inferencia[L]
        resumen[L] = r

    # σ_semilla agrupada (criterio §2.3)
    if len(sem_comunes) > 1:
        varz = [resumen[L]["mejor_val_accuracy"]["de_entre_semillas"] ** 2 for L in pasos]
        sigma = float(np.sqrt(np.mean(varz)))
    else:
        sigma = None
    margen = max(0.01, sigma or 0.0)

    criterio = {"sigma_semilla_agrupada": sigma, "margen": margen, "por_valor": {}}
    ref = resumen.get(PASOS_REFERENCIA)
    candidatos = []
    if ref is not None:
        v30 = np.array(ref["mejor_val_accuracy"]["por_pliegue_media_semillas"])
        for L in pasos:
            if L == PASOS_REFERENCIA:
                continue
            vL = np.array(resumen[L]["mejor_val_accuracy"]["por_pliegue_media_semillas"])
            dif = np.round(vL - v30, 6)   # ver pruebas_pareadas: empates de float32
            pp = pruebas_pareadas(dif)
            # Pareada también en prueba (solo descriptiva)
            t30 = np.array(ref["top1"]["por_pliegue_media_semillas"])
            tL = np.array(resumen[L]["top1"]["por_pliegue_media_semillas"])
            delta = float(np.mean(dif))
            cumple_a = delta > margen + 1e-12
            cumple_b = pp["mejora"] >= 4
            criterio["por_valor"][L] = {
                "delta_val_media": delta, "cumple_margen": bool(cumple_a),
                "pliegues_mejor_que_30": pp["mejora"], "cumple_4_de_5": bool(cumple_b),
                "cumple_criterio": bool(cumple_a and cumple_b), "pareada_validacion": pp,
                "pareada_prueba_descriptiva": pruebas_pareadas(tL - t30)}
            if cumple_a and cumple_b:
                candidatos.append(L)
    if candidatos:
        mejor = max(resumen[L]["mejor_val_accuracy"]["entre_pliegues"]["media"] for L in candidatos)
        empatados = [L for L in candidatos
                     if mejor - resumen[L]["mejor_val_accuracy"]["entre_pliegues"]["media"] < 0.01]
        elegido = min(empatados)
    else:
        elegido = PASOS_REFERENCIA
    criterio["candidatos_que_cumplen"] = candidatos
    criterio["valor_recomendado"] = elegido

    # Reproducción de la referencia N-30 (30 pasos, semilla 2026)
    repro = None
    if REFERENCIA_N30.exists() and all((PASOS_REFERENCIA, 2026, k) in tab for k in folds):
        refj = json.loads(REFERENCIA_N30.read_text())
        repro = []
        for p in refj["pliegues"]:
            k = p["pliegue"]
            if k in folds:
                n = tab[(PASOS_REFERENCIA, 2026, k)]
                repro.append({"pliegue": k, "ref_top1": p["top1"], "nuevo_top1": n["top1"],
                              "ref_val": p["mejor_val_accuracy"], "nuevo_val": n["mejor_val_accuracy"],
                              "ref_epocas": p["epocas_ejecutadas"], "nuevo_epocas": n["epocas_ejecutadas"],
                              "ref_f1": p["f1_macro"], "nuevo_f1": n["f1_macro"]})

    salida = {"fecha": datetime.now().isoformat(timespec="seconds"),
              "criterio_documento": "docs/EXPERIMENTO_PASOS_TEMPORALES.md §2",
              "processed_dir": "data/processed_holistic_mascara_rostro", "representacion": "manos",
              "particion": "data/partitions/loso_5_pliegues.json",
              "pasos": pasos, "semillas": semillas, "semillas_en_criterio": sem_comunes,
              "pliegues": folds, "resumen_por_valor": {str(k): v for k, v in resumen.items()},
              "criterio": {**criterio, "por_valor": {str(k): v for k, v in criterio["por_valor"].items()}},
              "reproduccion_referencia_N30": repro,
              "corridas": filas}
    with open(out / "resultados_pasos.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=1, ensure_ascii=False)

    # Impresión
    print(f"\n{'L':>3} {'val media':>10} {'DE':>6} {'val/pliegue':>32} {'test t1':>8} {'DE':>6} "
          f"{'t3':>6} {'F1':>6} {'epoc':>5} {'ms(1)':>6}")
    for L in pasos:
        r = resumen[L]
        print(f"{L:3d} {r['mejor_val_accuracy']['entre_pliegues']['media']:10.4f} "
              f"{r['mejor_val_accuracy']['entre_pliegues']['de']:6.4f} "
              f"{' '.join(f'{v:.3f}' for v in r['mejor_val_accuracy']['por_pliegue_media_semillas']):>32} "
              f"{r['top1']['entre_pliegues']['media']:8.4f} {r['top1']['entre_pliegues']['de']:6.4f} "
              f"{r['top3']['entre_pliegues']['media']:6.4f} {r['f1_macro']['entre_pliegues']['media']:6.4f} "
              f"{r['epocas_ejecutadas']['entre_pliegues']['media']:5.1f} "
              f"{r.get('inferencia', {}).get('ms_lote1_mediana', float('nan')):6.2f}")
    print(f"\nsigma_semilla {sigma} | margen {margen:.4f}")
    for L, c in criterio["por_valor"].items():
        print(f"  L={L}: delta val {c['delta_val_media']:+.4f} | pliegues mejores {c['pliegues_mejor_que_30']}/5 | "
              f"signos p {c['pareada_validacion']['signos_p_bilateral']:.3f} | wilcoxon p "
              f"{c['pareada_validacion']['wilcoxon_p_bilateral']:.3f} | cumple {c['cumple_criterio']}")
    print(f"valor recomendado: {elegido}")
    if repro:
        print("reproduccion N-30:", json.dumps(repro))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("entrenar")
    e.add_argument("--pasos", default="15,20,30,45,60")
    e.add_argument("--semillas", default="2026")
    e.add_argument("--folds", default="5")
    e.add_argument("--processed", default=str(PROCESSED_CANONICO))
    e.add_argument("--representacion", default=REPRESENTACION_CANONICA)
    e.add_argument("--epochs", type=int, default=200)
    e.add_argument("--aug", type=int, default=10)
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--patience", type=int, default=25)
    e.add_argument("--dropout", type=float, default=0.3)
    i = sub.add_parser("inferencia")
    i.add_argument("--calentamiento", type=int, default=10)
    i.add_argument("--repeticiones", type=int, default=50)
    r = sub.add_parser("resumen")
    for p in (e, i, r):
        p.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    {"entrenar": cmd_entrenar, "inferencia": cmd_inferencia, "resumen": cmd_resumen}[args.cmd](args)


if __name__ == "__main__":
    main()
