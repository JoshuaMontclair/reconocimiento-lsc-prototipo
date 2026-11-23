"""
Experimento: ¿cómo tratar los fotogramas en los que MediaPipe no detectó una mano al construir los
30 pasos? (pedido del director, reunión del 2-oct-2026, N-60). LOSO de 5 pliegues.

Brazos (lo ÚNICO que cambia es cómo se pasa de los T fotogramas a los 30 pasos de posiciones; ver
src/mano_ausente_variantes.py para la definición exacta):
  A  mascara     canónica (D-07): resample_sequence sobre la secuencia con el bloque ausente en 0.
  B  pivote      cada momento usa un fotograma real con la mano detectada (por mano).
  C  imputacion  interpolación lineal de los fotogramas sin mano y luego resample_sequence.

Reutiliza SIN modificar train_loso.cargar_datos, verificar_particion, importar_pipeline_canonico y
entrenar_y_evaluar (aumentación ×10 con espejo sobre las 126 posiciones, add_velocity DESPUÉS, la
misma red y los mismos hiperparámetros). La regla del brazo se aplica al construir X_seq, antes de la
aumentación: entrenamiento, validación y prueba de un brazo usan la misma regla.

Criterio prerregistrado en docs/EXPERIMENTO_MANO_AUSENTE.md §2 (validación; la prueba solo describe).

Subcomandos:
  verificar   sin entrenar: mascara = preprocesamiento actual, conmutación con el espejo, y cuánto
              cambia cada regla (pasos en cero, transiciones, repeticiones y distancia del pivote).
  entrenar    combinaciones (semilla, pliegue, brazo) que falten; reanudable.
  inferencia  costo de construir los 30 pasos con cada regla (ms por ventana).
  resumen     agrega, aplica el criterio y escribe el JSON de datos crudos.
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
from src import mano_ausente_variantes as mav  # noqa: E402

PROCESSED_CANONICO = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
REPRESENTACION = "manos"
DEFAULT_OUT = BASE_DIR / "models" / "experimento_mano_ausente_2026-10-02"
JSON_DOCS = BASE_DIR / "docs" / "EXPERIMENTO_MANO_AUSENTE.json"
REFERENCIA_N30 = BASE_DIR / "models" / "loso_2026-09-15_holistic_mascara_manos" / "resultados_loso.json"

BRAZOS = {
    "A": {"carpeta": "A_mascara", "regla": "mascara", "descripcion": "mascara (canonica, D-07)"},
    "B": {"carpeta": "B_pivote", "regla": "pivote", "descripcion": "pivote: fotograma real detectado mas cercano"},
    "C": {"carpeta": "C_imputacion", "regla": "imputacion", "descripcion": "imputacion lineal entre validos y remuestreo"},
}
PISO_MARGEN = 0.01


def _lista_int(texto):
    return [int(t) for t in str(texto).split(",") if t.strip() != ""]


def _lista_brazos(texto):
    brazos = [t.strip().upper() for t in str(texto).split(",") if t.strip()]
    for b in brazos:
        if b not in BRAZOS:
            raise SystemExit(f"--brazos: {b} no existe (A, B o C)")
    return brazos


def dir_corrida(out, brazo, semilla, k):
    return Path(out) / BRAZOS[brazo]["carpeta"] / f"semilla_{semilla}" / f"pliegue_{k}"


def cargar_todo(processed):
    X_padded, y, lengths, nombres, senantes = tl.cargar_datos(processed, REPRESENTACION)
    with open(tl.ARCHIVO_LOSO, encoding="utf-8") as f:
        loso = json.load(f)
    n_clases = int(y.max()) + 1
    for p in loso["pliegues"]:
        tl.verificar_particion(p["conjuntos"], nombres, senante_test=p["senante_test"], n_clases=n_clases)
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
        "val_f1_macro": float(f1_score(y_val, pred, labels=list(range(int(probs.shape[1]))),
                                       average="macro", zero_division=0)),
    }


def reescribir_procedencia(ruta_modelo, brazo):
    """escribir_procedencia copia mano_ausente del metadata del processed («mascara»); aquí se pone la
    regla real del brazo."""
    from src.representacion import ruta_procedencia
    ruta = ruta_procedencia(ruta_modelo)
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos["mano_ausente"] = BRAZOS[brazo]["regla"]
    datos["mano_ausente_base_del_processed"] = "mascara"
    datos["brazo_experimento_mano_ausente"] = brazo
    datos["nota"] = ("modelo del experimento de mano ausente (docs/EXPERIMENTO_MANO_AUSENTE.md); la regla "
                     "se aplica con src/mano_ausente_variantes.py al construir los 30 pasos. NO es el modelo "
                     "canonico: translator.py no implementa pivote ni imputacion y no debe cargarlo")
    datos["script"] = "src/experimento_mano_ausente.py (via train_loso.entrenar_y_evaluar)"
    ruta.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Verificación sin entrenar
# ---------------------------------------------------------------------------

def describir_regla(X_seq):
    """Pasos con bloque de mano en ceros exactos, transiciones cero<->no cero y norma de la velocidad."""
    res = {}
    for nombre, sl in mav.BLOQUES_MANOS.items():
        bloque = X_seq[:, :, sl]
        cero = np.all(bloque == 0, axis=2)
        trans = cero[:, 1:] != cero[:, :-1]
        vel = np.linalg.norm(bloque[:, 1:] - bloque[:, :-1], axis=2)
        res[nombre] = {
            "pasos_en_cero": int(cero.sum()),
            "secuencias_con_algun_paso_en_cero": int(np.any(cero, axis=1).sum()),
            "transiciones": int(trans.sum()),
            "norma_velocidad_mediana": float(np.median(vel)),
            "norma_velocidad_p99": float(np.percentile(vel, 99)),
            "norma_velocidad_max": float(vel.max()),
        }
    return res


def estadisticas_pivote(X_padded, lengths, n_pasos=30):
    """Cuántos momentos del pivote no usan su fotograma nominal, a qué distancia y cuántos repiten."""
    reemplazos, repetidos, dist = 0, 0, []
    for i in range(X_padded.shape[0]):
        T = int(lengths[i])
        seq = X_padded[i, :T]
        p = mav.posiciones_momentos(T, n_pasos)
        c = np.floor(p + 0.5).astype(int)
        for b in mav.BLOQUES_MANOS.values():
            det = mav.mano_detectada(seq, b)
            idx = mav.indices_pivote(det, n_pasos)
            if idx is None:
                continue
            m = idx != c
            reemplazos += int(m.sum())
            dist.extend(np.abs(idx[m] - p[m]).tolist())
            repetidos += int(n_pasos - len(set(idx.tolist())))
    dist = np.array(dist) if dist else np.zeros(1)
    return {"momentos_reemplazados": reemplazos, "momentos_que_repiten_un_fotograma": repetidos,
            "distancia_al_momento_mediana_fotogramas": float(np.median(dist)),
            "distancia_p95": float(np.percentile(dist, 95)), "distancia_max": float(dist.max())}


def pasos_contaminados(X_padded, lengths, n_pasos=30):
    """Con mascara: pasos remuestreados en los que interviene (con peso > 0) un fotograma sin la mano.
    Incluye los pasos en cero exacto y los «encogidos» hacia el origen."""
    res = {}
    for nombre, b in mav.BLOQUES_MANOS.items():
        total, encogidos, videos = 0, 0, 0
        for i in range(X_padded.shape[0]):
            T = int(lengths[i])
            det = mav.mano_detectada(X_padded[i, :T], b)
            if det.all():
                continue
            p = mav.posiciones_momentos(T, n_pasos)
            lo = np.floor(p).astype(int)
            hi = np.minimum(lo + 1, T - 1)
            w = p - lo
            malo = (~det[lo] & (w < 1 - 1e-9)) | (~det[hi] & (w > 1e-9))
            cero = ~det[lo] & ~det[hi] | (~det[lo] & (w <= 1e-9)) | (~det[hi] & (w >= 1 - 1e-9))
            total += int(malo.sum())
            encogidos += int((malo & ~cero).sum())
            videos += int(malo.any())
        res[nombre] = {"pasos_contaminados": total, "de_ellos_encogidos_no_cero": encogidos,
                       "videos_con_algun_paso_contaminado": videos}
    return res


def cmd_verificar(args):
    processed = Path(args.processed).resolve()
    X_padded, y, lengths, nombres, loso, pos = cargar_todo(processed)
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    tf, _, add_velocity, resample_sequence, transformaciones = canon
    ref = tl.remuestrear_todas(X_padded, lengths, resample_sequence, tl.SEQ_LENGTH)
    seqs = {b: mav.remuestrear_todas_con_regla(X_padded, lengths, BRAZOS[b]["regla"], tl.SEQ_LENGTH) for b in BRAZOS}
    assert np.array_equal(seqs["A"], ref), "A debe ser exactamente train_loso.remuestrear_todas"
    print(f"[ok] A (mascara) = train_loso.remuestrear_todas bit a bit; X_seq {ref.shape}")
    # Conmutación con el espejo de la aumentación (transformaciones[0] = espejar(X, 'manos')).
    esp = transformaciones[0][1]
    max_dif = {}
    for b in BRAZOS:
        d = 0.0
        for i in range(0, X_padded.shape[0], 7):
            T = int(lengths[i])
            s = X_padded[i:i + 1, :T]
            a1 = esp(mav.aplicar_regla(s[0], BRAZOS[b]["regla"], tl.SEQ_LENGTH)[None])[0]
            a2 = mav.aplicar_regla(esp(s)[0], BRAZOS[b]["regla"], tl.SEQ_LENGTH)
            d = max(d, float(np.max(np.abs(a1 - a2))))
        max_dif[b] = d
        assert d < 1e-5, f"{b}: la regla no conmuta con el espejo ({d})"
    print(f"[ok] las tres reglas conmutan con el espejo (dif. max {max_dif}); las velocidades se calculan "
          f"despues de aumentar, como hoy")
    desc = {b: describir_regla(seqs[b]) for b in BRAZOS}
    for b in BRAZOS:
        print(f"[info] {b} {BRAZOS[b]['regla']}: {json.dumps(desc[b])}")
    est = estadisticas_pivote(X_padded, lengths, tl.SEQ_LENGTH)
    print(f"[info] pivote: {est}")
    cont = pasos_contaminados(X_padded, lengths, tl.SEQ_LENGTH)
    print(f"[info] mascara, pasos contaminados por un fotograma sin mano: {cont}")
    dif = {b: float(np.mean(np.abs(seqs[b] - ref))) for b in ("B", "C")}
    print(f"[info] diferencia media absoluta con A: {dif}")
    nunca = {n: int(sum(1 for i in range(len(nombres))
                        if not mav.mano_detectada(X_padded[i, :lengths[i]], sl).any()))
             for n, sl in mav.BLOQUES_MANOS.items()}
    print(f"[info] videos con una mano nunca detectada: {nunca}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "verificacion.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": datetime.now().isoformat(timespec="seconds"), "processed": tl._rel(processed),
                   "comprobaciones": ["A = train_loso.remuestrear_todas bit a bit",
                                      "las tres reglas conmutan con el espejo (1 de cada 7 muestras)"],
                   "espejo_dif_max": max_dif, "por_regla": desc, "pivote": est,
                   "mascara_pasos_contaminados": cont,
                   "diferencia_media_absoluta_con_A": dif, "videos_con_mano_nunca_detectada": nunca},
                  f, indent=1, ensure_ascii=False)
    print(f"  escrito {tl._rel(out / 'verificacion.json')}")


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def cmd_entrenar(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    processed = Path(args.processed).resolve()
    X_padded, y, lengths, nombres, loso, pos = cargar_todo(processed)
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    tf, build_model, add_velocity, resample_sequence, transformaciones = canon
    nombres_senas = tl.cargar_nombres_senas()
    folds = tl.parsear_folds(args.folds, len(loso["pliegues"]))
    semillas = _lista_int(args.semillas)
    brazos = _lista_brazos(args.brazos)
    X_por_brazo = {b: mav.remuestrear_todas_con_regla(X_padded, lengths, BRAZOS[b]["regla"], tl.SEQ_LENGTH)
                   for b in brazos}
    if "A" in X_por_brazo:
        assert np.array_equal(X_por_brazo["A"], tl.remuestrear_todas(X_padded, lengths, resample_sequence,
                                                                     tl.SEQ_LENGTH))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] experimento_mano_ausente entrenar | brazos "
          f"{brazos} | semillas {semillas} | pliegues {folds} | epochs {args.epochs} | aug x{args.aug} | "
          f"paciencia {args.patience} | processed {tl._rel(processed)} | salida {tl._rel(out)}", flush=True)
    for semilla in semillas:
        for k in folds:
            p = loso["pliegues"][k]
            idx_val = np.array([pos[n] for n in p["conjuntos"]["val"]], dtype=np.int64)
            for brazo in brazos:
                d = dir_corrida(out, brazo, semilla, k)
                if (d / "metricas_mano_ausente.json").exists():
                    print(f"  ya existe {tl._rel(d)}; se salta", flush=True)
                    continue
                X_seq = X_por_brazo[brazo]
                print(f"\n{'=' * 72}\n  brazo {brazo} ({BRAZOS[brazo]['descripcion']}) | semilla {semilla} | "
                      f"pliegue {k} (senante {p['senante_test']}) | "
                      f"{datetime.now().isoformat(timespec='seconds')}\n{'=' * 72}", flush=True)
                ns = argparse.Namespace(seed=semilla, aug=args.aug, epochs=args.epochs,
                                        batch_size=args.batch_size, patience=args.patience,
                                        dropout=args.dropout, processed=str(processed),
                                        representacion=REPRESENTACION)
                m = tl.entrenar_y_evaluar(f"{brazo}_s{semilla}_pliegue_{k}", p["conjuntos"], pos, X_seq, y,
                                          ns, d, canon, nombres_senas)
                reescribir_procedencia(d / "modelo.keras", brazo)
                model = tf.keras.models.load_model(str(d / "modelo.keras"))
                mv = metricas_validacion(model, add_velocity(X_seq[idx_val]), y[idx_val])
                n_param = int(model.count_params())
                del model
                tf.keras.backend.clear_session()
                m.update(mv)
                m.update({"brazo": brazo, "regla": BRAZOS[brazo]["regla"], "semilla": semilla, "pliegue": k,
                          "senante_test": p["senante_test"], "n_parametros": n_param,
                          "config": {"epochs_max": args.epochs, "aug": args.aug, "batch_size": args.batch_size,
                                     "patience": args.patience, "dropout": args.dropout,
                                     "seq_length": tl.SEQ_LENGTH, "processed": tl._rel(processed),
                                     "representacion": REPRESENTACION},
                          "fecha_fin": datetime.now().isoformat(timespec="seconds")})
                with open(d / "metricas_mano_ausente.json", "w", encoding="utf-8") as f:
                    json.dump(m, f, indent=1)
                print(f"  -> {brazo} s={semilla} k={k}: val {m['mejor_val_accuracy']:.3f} (checkpoint "
                      f"{mv['val_top1']:.3f}) | test top-1 {m['top1']:.3f} | epocas {m['epocas_ejecutadas']} | "
                      f"{m['tiempo_total_s']:.0f} s", flush=True)
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] fin", flush=True)


# ---------------------------------------------------------------------------
# Costo de la regla (la red es idéntica en los tres brazos: 30 × 252, mismos parámetros)
# ---------------------------------------------------------------------------

def cmd_inferencia(args):
    processed = Path(args.processed).resolve()
    X_padded, y, lengths, nombres, loso, pos = cargar_todo(processed)
    rng = np.random.default_rng(0)
    elegidas = rng.choice(X_padded.shape[0], 200, replace=False)
    costo = {}
    for b in BRAZOS:
        t = []
        for i in elegidas:
            s = X_padded[i, :int(lengths[i])]
            t0 = time.perf_counter()
            mav.aplicar_regla(s, BRAZOS[b]["regla"], tl.SEQ_LENGTH)
            t.append((time.perf_counter() - t0) * 1000)
        costo[b] = {"ms_mediana": float(np.median(t)), "ms_p90": float(np.percentile(t, 90))}
    print(f"costo de construir los 30 pasos (ms por ventana, 200 videos): {costo}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "inferencia.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": datetime.now().isoformat(timespec="seconds"),
                   "metodo": "tiempo de mav.aplicar_regla (T fotogramas x 126 -> 30 x 126) en 200 videos al "
                             "azar (semilla 0); la red es la misma en los tres brazos (30 x 252, 714 291 "
                             "parametros), asi que su latencia no cambia",
                   "entorno": tl.info_entorno(), "ms_construir_30_pasos": costo}, f, indent=1)


# ---------------------------------------------------------------------------
# Resumen y criterio
# ---------------------------------------------------------------------------

def _media_de(v):
    v = [float(x) for x in v]
    return {"media": float(np.mean(v)), "de": float(np.std(v, ddof=1)) if len(v) > 1 else None, "valores": v}


def pruebas_pareadas(dif):
    from scipy.stats import binomtest, wilcoxon
    dif = np.asarray(dif, dtype=float)
    pos_, neg_ = int(np.sum(dif > 1e-12)), int(np.sum(dif < -1e-12))
    res = {"diferencias": dif.tolist(), "a_favor_del_primero": pos_, "a_favor_del_segundo": neg_,
           "empates": int(len(dif) - pos_ - neg_)}
    res["signos_p_bilateral"] = float(binomtest(pos_, pos_ + neg_, 0.5).pvalue) if pos_ + neg_ > 0 else 1.0
    if np.any(np.abs(dif) > 1e-12):
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


METRICAS = ["mejor_val_accuracy", "val_top1", "val_top3", "val_f1_macro", "top1", "top3", "f1_macro",
            "precision_macro", "recall_macro", "epocas_ejecutadas", "mejor_epoca", "tiempo_total_s", "n_parametros"]


def cmd_resumen(args):
    out = Path(args.out).resolve()
    filas = []
    for b in BRAZOS:
        for p in sorted((out / BRAZOS[b]["carpeta"]).glob("semilla_*/pliegue_*/metricas_mano_ausente.json")):
            filas.append(json.loads(p.read_text(encoding="utf-8")))
    if not filas:
        raise SystemExit("no hay corridas")
    brazos = [b for b in BRAZOS if any(f["brazo"] == b for f in filas)]
    folds = sorted({f["pliegue"] for f in filas})
    semillas_por_brazo = {b: sorted({f["semilla"] for f in filas if f["brazo"] == b}) for b in brazos}

    def celda(b, s, k):
        r = [f for f in filas if f["brazo"] == b and f["semilla"] == s and f["pliegue"] == k]
        return r[0] if r else None

    por_brazo = {}
    for b in brazos:
        pf = {}
        for k in folds:
            rs = [f for f in filas if f["brazo"] == b and f["pliegue"] == k]
            pf[k] = {m: _media_de([r[m] for r in rs]) for m in METRICAS}
            pf[k]["semillas"] = [r["semilla"] for r in rs]
        ag = {m: _media_de([pf[k][m]["media"] for k in folds]) for m in METRICAS}
        por_semilla = {}
        for s in semillas_por_brazo[b]:
            rs = [celda(b, s, k) for k in folds]
            if all(rs):
                por_semilla[s] = {m: _media_de([r[m] for r in rs]) for m in
                                  ["mejor_val_accuracy", "top1", "top3", "f1_macro", "epocas_ejecutadas"]}
        por_brazo[b] = {"por_pliegue": {str(k): v for k, v in pf.items()}, "agregado": ag,
                        "por_semilla": {str(s): v for s, v in por_semilla.items()}}

    # σ_semilla (prerregistrada, §2): por brazo, DE (ddof=1) entre semillas de la val media de 5
    # pliegues; agrupada como raíz de la media de las varianzas de los brazos. Margen = max(σ, 0,01).
    var_brazo = {}
    for b in brazos:
        v = [por_brazo[b]["por_semilla"][str(s)]["mejor_val_accuracy"]["media"]
             for s in semillas_por_brazo[b] if str(s) in por_brazo[b]["por_semilla"]]
        if len(v) > 1:
            var_brazo[b] = float(np.var(v, ddof=1))
    sigma = float(np.sqrt(np.mean(list(var_brazo.values())))) if var_brazo else None
    margen = max(sigma, PISO_MARGEN) if sigma is not None else None
    # Referencia: DE de una sola corrida (celda brazo × pliegue), agrupada.
    var_celda = []
    for b in brazos:
        for k in folds:
            v = [f["mejor_val_accuracy"] for f in filas if f["brazo"] == b and f["pliegue"] == k]
            if len(v) > 1:
                var_celda.append(float(np.var(v, ddof=1)))
    sigma_celda = float(np.sqrt(np.mean(var_celda))) if var_celda else None

    comparaciones = {}
    for b1, b2 in [("B", "A"), ("C", "A"), ("C", "B")]:
        if b1 not in brazos or b2 not in brazos:
            continue
        sc = sorted(set(semillas_por_brazo[b1]) & set(semillas_por_brazo[b2]))
        comp = {"semillas_comunes": sc}
        for m in ["mejor_val_accuracy", "top1", "top3", "f1_macro"]:
            dif_k = []
            for k in folds:
                v1 = [celda(b1, s, k)[m] for s in sc if celda(b1, s, k) and celda(b2, s, k)]
                v2 = [celda(b2, s, k)[m] for s in sc if celda(b1, s, k) and celda(b2, s, k)]
                dif_k.append(float(np.mean(v1) - np.mean(v2)))
            dif_ks = [celda(b1, s, k)[m] - celda(b2, s, k)[m] for s in sc for k in folds
                      if celda(b1, s, k) and celda(b2, s, k)]
            comp[m] = {"diferencia_por_pliegue": dif_k, "diferencia_media": float(np.mean(dif_k)),
                       "de_diferencias_entre_pliegues": float(np.std(dif_k, ddof=1)) if len(dif_k) > 1 else None,
                       "pruebas_5_pliegues": pruebas_pareadas(dif_k),
                       "pares_pliegue_semilla": {
                           "n": len(dif_ks), "diferencias": [float(x) for x in dif_ks],
                           "a_favor": int(np.sum(np.array(dif_ks) > 1e-12)),
                           "en_contra": int(np.sum(np.array(dif_ks) < -1e-12)),
                           "nota": "no independientes; solo descriptivo"}}
        comparaciones[f"{b1}_menos_{b2}"] = comp

    # Criterio prerregistrado
    criterio = None
    if margen is not None and "A" in brazos:
        cumple, detalle = {}, {}
        for b in [x for x in ("B", "C") if x in brazos]:
            d = np.array(comparaciones[f"{b}_menos_A"]["mejor_val_accuracy"]["diferencia_por_pliegue"])
            mejor, peor = int(np.sum(d > 1e-12)), int(np.sum(d < -1e-12))
            media = float(np.mean(d))
            ok = bool(media > 0 and mejor >= 4 and media > margen)
            empeora = bool(peor >= 4 and -media > margen)
            cumple[b] = ok
            detalle[b] = {"delta_por_pliegue": d.tolist(), "delta_media": media, "pliegues_mejor": mejor,
                          "pliegues_peor": peor, "pliegues_igual": int(len(d) - mejor - peor),
                          "a_media_mayor_que_0": bool(media > 0), "b_al_menos_4_de_5": bool(mejor >= 4),
                          "c_media_mayor_que_margen": bool(media > margen), "cumple": ok,
                          "empeora_segun_regla_simetrica": empeora}
        quienes = [b for b, ok in cumple.items() if ok]
        val = {b: por_brazo[b]["agregado"]["mejor_val_accuracy"]["media"] for b in brazos}
        if not quienes:
            eleccion, motivo = "A", "ninguna variante cumple: no hay evidencia; se mantiene mascara"
        elif len(quienes) == 1:
            eleccion, motivo = quienes[0], f"solo {quienes[0]} cumple"
        else:
            if abs(val["B"] - val["C"]) < margen:
                eleccion, motivo = "C", "B y C cumplen y quedan a menos del margen entre si: se prefiere C"
            else:
                eleccion = max(("B", "C"), key=lambda b: val[b])
                motivo = "B y C cumplen; se toma la de mayor validacion (diferencia >= margen)"
        criterio = {
            "regla": "Una variante V reemplaza a mascara si (a) media de Δk = val(V,k) − val(A,k) > 0, (b) Δk > 0 "
                     "en >= 4 de 5 pliegues y (c) media de Δk > margen = max(σ_semilla, 0,01). Si ninguna cumple, "
                     "se mantiene mascara. Si B y C cumplen y |val(B) − val(C)| < margen, se prefiere C; si no, "
                     "la de mayor validacion.",
            "sigma_semilla": sigma, "varianza_entre_semillas_por_brazo": var_brazo, "margen": margen,
            "sigma_una_corrida_referencia": sigma_celda, "detalle": detalle,
            "val_media": val, "eleccion": eleccion, "regla_elegida": BRAZOS[eleccion]["regla"], "motivo": motivo,
        }

    reproduccion = None
    if REFERENCIA_N30.exists() and "A" in brazos:
        ref = json.loads(REFERENCIA_N30.read_text(encoding="utf-8"))
        filas_rep = []
        for pr in ref["pliegues"]:
            c = celda("A", 2026, pr["pliegue"])
            if c:
                filas_rep.append({"pliegue": pr["pliegue"], "ref_top1": pr["top1"], "A_top1": c["top1"],
                                  "ref_val": pr["mejor_val_accuracy"], "A_val": c["mejor_val_accuracy"],
                                  "ref_epocas": pr["epocas_ejecutadas"], "A_epocas": c["epocas_ejecutadas"]})
        reproduccion = {"referencia": tl._rel(REFERENCIA_N30), "pliegues": filas_rep,
                        "identica": bool(filas_rep) and all(
                            abs(r["ref_top1"] - r["A_top1"]) < 1e-9 and abs(r["ref_val"] - r["A_val"]) < 1e-9
                            and r["ref_epocas"] == r["A_epocas"] for r in filas_rep)}

    extra = {}
    for nombre in ("verificacion.json", "inferencia.json"):
        if (out / nombre).exists():
            extra[nombre] = json.loads((out / nombre).read_text(encoding="utf-8"))

    resultado = {"experimento": "mano ausente: mascara / pivote / imputacion (docs/EXPERIMENTO_MANO_AUSENTE.md)",
                 "fecha_resumen": datetime.now().isoformat(timespec="seconds"),
                 "brazos": {b: BRAZOS[b] for b in brazos}, "semillas_por_brazo": semillas_por_brazo,
                 "pliegues": folds, "criterio": criterio, "comparaciones": comparaciones,
                 "por_brazo": por_brazo, "reproduccion_referencia_N30": reproduccion,
                 "verificacion": extra.get("verificacion.json"), "inferencia": extra.get("inferencia.json"),
                 "corridas": filas}
    with open(out / "resultados_mano_ausente.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=1, ensure_ascii=False)
    if args.docs:
        with open(JSON_DOCS, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=1, ensure_ascii=False)

    print(f"brazos {brazos} | semillas {semillas_por_brazo} | pliegues {folds}")
    for b in brazos:
        ag = por_brazo[b]["agregado"]
        print(f"\n{b} ({BRAZOS[b]['descripcion']}): val {ag['mejor_val_accuracy']['media']:.4f} ± "
              f"{ag['mejor_val_accuracy']['de'] or 0:.4f} | top-1 {ag['top1']['media']:.4f} ± {ag['top1']['de'] or 0:.4f}"
              f" | top-3 {ag['top3']['media']:.4f} | F1 {ag['f1_macro']['media']:.4f} | epocas "
              f"{ag['epocas_ejecutadas']['media']:.1f}")
        for k in folds:
            c = por_brazo[b]["por_pliegue"][str(k)]
            print(f"   pliegue {k}: val {c['mejor_val_accuracy']['valores']} top-1 "
                  f"{[round(x, 3) for x in c['top1']['valores']]} top-3 {[round(x, 3) for x in c['top3']['valores']]} "
                  f"F1 {[round(x, 3) for x in c['f1_macro']['valores']]} epocas {c['epocas_ejecutadas']['valores']}")
        for s, v in por_brazo[b]["por_semilla"].items():
            print(f"   semilla {s}: val {v['mejor_val_accuracy']['media']:.4f} top-1 {v['top1']['media']:.4f}")
    for nombre, comp in comparaciones.items():
        print(f"\n{nombre}:")
        for m in ["mejor_val_accuracy", "top1", "top3", "f1_macro"]:
            c = comp[m]
            pr = c["pruebas_5_pliegues"]
            print(f"   {m:20s} {[round(x, 4) for x in c['diferencia_por_pliegue']]} media {c['diferencia_media']:+.4f}"
                  f" | a favor {pr['a_favor_del_primero']}/{len(c['diferencia_por_pliegue'])} | signos p="
                  f"{pr['signos_p_bilateral']:.4f} | wilcoxon p={pr['wilcoxon_p_bilateral']:.4f} | pares "
                  f"{c['pares_pliegue_semilla']['a_favor']}/{c['pares_pliegue_semilla']['en_contra']}/"
                  f"{c['pares_pliegue_semilla']['n']}")
    if criterio:
        print(f"\nσ_semilla {sigma} | margen {margen} | σ una corrida {sigma_celda}")
        for b, d in criterio["detalle"].items():
            print(f"   {b}: {d}")
        print(f"CRITERIO: eleccion {criterio['eleccion']} ({criterio['regla_elegida']}): {criterio['motivo']}")
    if reproduccion:
        print(f"\nreproduccion N-30 (A, semilla 2026): identica = {reproduccion['identica']}")
        for r in reproduccion["pliegues"]:
            print(f"   {r}")


def main():
    parser = argparse.ArgumentParser(description="Experimento de mano ausente (LOSO de 5 pliegues).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def comunes(sp):
        sp.add_argument("--out", type=str, default=str(DEFAULT_OUT))
        sp.add_argument("--processed", type=str, default=str(PROCESSED_CANONICO))

    sv = sub.add_parser("verificar")
    comunes(sv)
    se = sub.add_parser("entrenar")
    comunes(se)
    se.add_argument("--brazos", type=str, default="A,B,C")
    se.add_argument("--semillas", type=str, default="2026,2027,2028")
    se.add_argument("--folds", type=str, default="5")
    se.add_argument("--epochs", type=int, default=200)
    se.add_argument("--aug", type=int, default=10)
    se.add_argument("--batch-size", type=int, default=64)
    se.add_argument("--patience", type=int, default=25)
    se.add_argument("--dropout", type=float, default=0.3)
    si = sub.add_parser("inferencia")
    comunes(si)
    sr = sub.add_parser("resumen")
    comunes(sr)
    sr.add_argument("--docs", action="store_true", help=f"copia el JSON a {tl._rel(JSON_DOCS)}")
    args = parser.parse_args()
    if not 1 <= getattr(args, "aug", 10) <= 10:
        raise SystemExit("--aug debe estar entre 1 y 10")
    {"verificar": cmd_verificar, "entrenar": cmd_entrenar, "inferencia": cmd_inferencia,
     "resumen": cmd_resumen}[args.cmd](args)


if __name__ == "__main__":
    main()
