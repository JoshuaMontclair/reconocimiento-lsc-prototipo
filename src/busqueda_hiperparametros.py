"""
Búsqueda de hiperparámetros de la red canónica sobre el LOSO de 5 pliegues (actividad 2.d).

Pregunta: ¿hay una combinación de hiperparámetros claramente mejor que la canónica para generalizar
a personas nuevas? La SELECCIÓN usa solo la exactitud top-1 de VALIDACIÓN interna de los pliegues
LOSO (2 muestras por seña de los cuatro señantes de entrenamiento, D-06). La prueba LOSO (señante no
visto) se calcula y se reporta como descripción; ninguna decisión de este script la lee.
Criterio prerregistrado: docs/BUSQUEDA_HIPERPARAMETROS.md §2.

Reutiliza SIN modificar las piezas canónicas:
  - train_loso: cargar_datos, verificar_particion, ARCHIVO_LOSO, importar_pipeline_canonico
    (add_velocity, resample_sequence y las transformaciones de refined_augmentation con el espejado
    de la representación), aumentar, remuestrear_todas, parsear_folds, info_entorno;
  - train_final_100: build_model (solo para verificar que construir_modelo lo reproduce),
    metricas_basicas.
El entrenamiento sigue el mismo orden de pasos que train_loso.entrenar_y_evaluar (semilla ->
aumentación -> velocidades -> modelo -> compile -> fit con EarlyStopping y ModelCheckpoint sobre
val_accuracy), de modo que con la configuración canónica y la semilla 2026 debe reproducir la
referencia de N-30 (0,460 / 0,533 / 0,606 / 0,615 / 0,685).

Subcomandos:
  verificar    construir_modelo(canónica) == build_model: parámetros (714.291), estructura capa a
               capa y pesos iniciales con la misma semilla. Imprime el plan de la fase 1.
  fase1        canónica + 20 configuraciones aleatorias, 1 semilla (2026), 5 pliegues; reanudable;
               corte temprano tras los pliegues 0 y 1 (> 5 puntos de validación bajo la canónica).
  seleccionar  las 3 mejores de la fase 1 por validación media -> finalistas.json (no lee la prueba).
  fase2        canónica + finalistas, semillas nuevas 2031, 2032 y 2033, 5 pliegues; reanudable.
  inferencia   parámetros y latencia de inferencia (lote de 1, como translator.py) de cada finalista.
  resumen      agrega, aplica el criterio y escribe docs/BUSQUEDA_HIPERPARAMETROS.json.

Uso:
  python src/busqueda_hiperparametros.py verificar
  python src/busqueda_hiperparametros.py fase1 --solo c00 --folds 0, --epochs 2 --aug 2 --out <tmp>  # humo
  python src/busqueda_hiperparametros.py fase1
  python src/busqueda_hiperparametros.py seleccionar
  python src/busqueda_hiperparametros.py fase2
  python src/busqueda_hiperparametros.py inferencia
  python src/busqueda_hiperparametros.py resumen
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
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
SEQ_LENGTH = 30
DEFAULT_OUT = BASE_DIR / "models" / "busqueda_hiperparametros"
JSON_DOCS = BASE_DIR / "docs" / "BUSQUEDA_HIPERPARAMETROS.json"
REFERENCIA_N30 = BASE_DIR / "models" / "loso_2026-09-15_holistic_mascara_manos" / "resultados_loso.json"
PARAMETROS_CANONICOS = 714_291

# ---------------------------------------------------------------------------
# Plan prerregistrado (docs/BUSQUEDA_HIPERPARAMETROS.md §2-§3). No se cambia por ningún resultado.
# ---------------------------------------------------------------------------

CANONICA = {
    "conv_filtros": 128, "conv_ventana": 3, "conv_capas": 2,
    "lstm": [128, 64], "densas": [256, 128], "dropout": 0.3,
    "lr": 1e-3, "batch": 64, "label_smoothing": 0.1,
}
ESPACIO = {
    "conv_filtros": [64, 128],
    "conv_ventana": [3, 5],
    "conv_capas": [1, 2],
    "lstm": [[128, 64], [64, 32], [128]],
    "densas": [[256, 128], [128, 64]],
    "dropout": [0.2, 0.3, 0.5],
    "lr": [5e-4, 1e-3, 2e-3],
    "batch": [32, 64],
    "label_smoothing": [0.0, 0.1],
}
SEMILLA_MUESTREO = 20260930      # generador del muestreo aleatorio de configuraciones
N_ALEATORIAS = 20                # configuraciones aleatorias de la fase 1 (+ la canónica = 21)
SEMILLA_FASE1 = 2026             # la de la referencia N-30: la canónica debe reproducirla
SEMILLAS_FASE2 = [2031, 2032, 2033]   # nuevas: ninguna se usó en la fase 1
CORTE_PUNTOS = 0.05              # corte temprano tras pliegues 0 y 1
PLIEGUES_CORTE = [0, 1]
N_FINALISTAS = 3
MARGEN_MINIMO = 0.01             # piso del margen: 1 muestra de validación por pliegue


def _clave(cfg):
    return json.dumps({k: cfg[k] for k in ESPACIO}, sort_keys=True)


def describir(cfg):
    return (f"conv {cfg['conv_capas']}x{cfg['conv_filtros']}k{cfg['conv_ventana']} | "
            f"lstm {'-'.join(map(str, cfg['lstm']))} | dens {'-'.join(map(str, cfg['densas']))} | "
            f"do {cfg['dropout']} | lr {cfg['lr']:g} | b {cfg['batch']} | ls {cfg['label_smoothing']}")


def generar_configuraciones():
    """c00 = canónica; c01..c20 = muestreo aleatorio uniforme e independiente por dimensión con
    numpy.default_rng(SEMILLA_MUESTREO), sin repetidos ni la canónica. Determinista."""
    rng = np.random.default_rng(SEMILLA_MUESTREO)
    vistas = {_clave(CANONICA)}
    configs = [{"id": "c00", **CANONICA, "canonica": True}]
    while len(configs) < N_ALEATORIAS + 1:
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in ESPACIO.items()}
        if _clave(cfg) in vistas:
            continue
        vistas.add(_clave(cfg))
        configs.append({"id": f"c{len(configs):02d}", **cfg, "canonica": False})
    return configs


def config_por_id(cid):
    for c in generar_configuraciones():
        if c["id"] == cid:
            return c
    raise SystemExit(f"configuracion {cid} inexistente")


# ---------------------------------------------------------------------------
# Arquitectura parametrizable
# ---------------------------------------------------------------------------

def construir_modelo(cfg, n_frames, n_features, n_classes):
    """CNN + BiLSTM + atención con los hiperparámetros de cfg. Con CANONICA produce exactamente
    build_model(n_frames, n_features, n_classes, dropout=0.3) de train_final_100.py (mismo orden
    de capas, mismas configuraciones, mismos pesos iniciales con la misma semilla)."""
    import tensorflow as tf
    d = cfg["dropout"]
    inp = tf.keras.Input(shape=(n_frames, n_features))
    x = inp
    for _ in range(cfg["conv_capas"]):
        x = tf.keras.layers.Conv1D(cfg["conv_filtros"], cfg["conv_ventana"], padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(d * 0.5)(x)

    for u in cfg["lstm"]:
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(u, return_sequences=True))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(d)(x)

    att = tf.keras.layers.Dense(1)(x)
    att = tf.keras.layers.Softmax(axis=1)(att)
    ctx = tf.keras.layers.Multiply()([x, att])
    ctx = tf.keras.layers.GlobalAveragePooling1D()(ctx)
    avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    mx = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([ctx, avg, mx])

    d1, d2 = cfg["densas"]
    x = tf.keras.layers.Dense(d1, activation="relu")(x)
    x = tf.keras.layers.Dropout(d)(x)
    x = tf.keras.layers.Dense(d2, activation="relu")(x)
    x = tf.keras.layers.Dropout(d * 0.5)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, x)


def _firma_capas(model):
    """Estructura comparable: clase y configuración de cada capa, sin nombres."""
    def limpiar(o):
        if isinstance(o, dict):
            return {k: limpiar(v) for k, v in o.items() if k not in ("name",)}
        if isinstance(o, list):
            return [limpiar(v) for v in o]
        return o
    return [(type(l).__name__, limpiar(l.get_config())) for l in model.layers]


def cmd_verificar(args):
    canon = tl.importar_pipeline_canonico(REPRESENTACION_CANONICA)
    tf, build_model = canon[0], canon[1]
    n_feat = 252
    tf.keras.utils.set_random_seed(0)
    a = build_model(SEQ_LENGTH, n_feat, 50, dropout=0.3)
    tf.keras.utils.set_random_seed(0)
    b = construir_modelo(CANONICA, SEQ_LENGTH, n_feat, 50)
    pa, pb = a.count_params(), b.count_params()
    print(f"parametros build_model {pa:,} | construir_modelo(canonica) {pb:,} | esperado {PARAMETROS_CANONICOS:,}")
    assert pa == pb == PARAMETROS_CANONICOS, "numero de parametros distinto"
    fa, fb = _firma_capas(a), _firma_capas(b)
    assert len(fa) == len(fb), f"numero de capas distinto: {len(fa)} vs {len(fb)}"
    for i, (x, y) in enumerate(zip(fa, fb)):
        assert x == y, f"capa {i} distinta:\n{x}\n{y}"
    print(f"estructura: {len(fa)} capas identicas en tipo y configuracion")
    wa, wb = a.get_weights(), b.get_weights()
    assert len(wa) == len(wb) and all(np.array_equal(u, v) for u, v in zip(wa, wb)), "pesos iniciales distintos"
    print(f"pesos iniciales con la misma semilla: {len(wa)} tensores identicos bit a bit")
    print("\nPlan de la fase 1 (generado con la semilla de muestreo "
          f"{SEMILLA_MUESTREO}):")
    for c in generar_configuraciones():
        m = construir_modelo(c, SEQ_LENGTH, n_feat, 50)
        print(f"  {c['id']} {describir(c)} | {m.count_params():>9,} parametros")
        del m
        tf.keras.backend.clear_session()
    if args.guardar:
        out = Path(args.out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "plan_fase1.json", "w", encoding="utf-8") as f:
            json.dump({"fecha": datetime.now().isoformat(timespec="seconds"),
                       "semilla_muestreo": SEMILLA_MUESTREO, "espacio": ESPACIO,
                       "configuraciones": generar_configuraciones(),
                       "semilla_fase1": SEMILLA_FASE1, "semillas_fase2": SEMILLAS_FASE2,
                       "corte": {"pliegues": PLIEGUES_CORTE, "puntos": CORTE_PUNTOS}}, f, indent=1)
        print(f"plan escrito en {tl._rel(out / 'plan_fase1.json')}")


# ---------------------------------------------------------------------------
# Entrenamiento de una corrida (configuración, semilla, pliegue)
# ---------------------------------------------------------------------------

def cargar_todo(processed, representacion):
    X_padded, y, lengths, nombres, _ = tl.cargar_datos(processed, representacion)
    with open(tl.ARCHIVO_LOSO, encoding="utf-8") as f:
        loso = json.load(f)
    n_clases = int(y.max()) + 1
    for p in loso["pliegues"]:
        tl.verificar_particion(p["conjuntos"], nombres, senante_test=p["senante_test"],
                               n_clases=n_clases)
    pos = {n: i for i, n in enumerate(nombres)}
    return X_padded, y, lengths, nombres, loso, pos


def dir_corrida(out, fase, cid, semilla, k):
    return Path(out) / fase / cid / f"semilla_{semilla}" / f"pliegue_{k}"


def _metricas(probs, y_real, loss_fn=None):
    from src.train_final_100 import metricas_basicas
    m = metricas_basicas(probs, y_real)
    return m


def entrenar_corrida(cfg, semilla, k, p, pos, X_seq, y, canon, d, args):
    """Mismo orden que train_loso.entrenar_y_evaluar. Selección del checkpoint SOLO con validación;
    la prueba se evalúa una vez al final y solo se guarda."""
    tf, _, add_velocity, _, transformaciones = canon
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    idx = {kk: np.array([pos[n] for n in p["conjuntos"][kk]], dtype=np.int64) for kk in p["conjuntos"]}
    n_clases = int(y.max()) + 1

    tf.keras.utils.set_random_seed(semilla)
    X_tr, y_tr = tl.aumentar(X_seq[idx["train"]], y[idx["train"]], args.aug, transformaciones)
    X_tr_f = add_velocity(X_tr)
    X_val_f = add_velocity(X_seq[idx["val"]])
    y_tr_oh = tf.keras.utils.to_categorical(y_tr, n_clases)
    y_val_oh = tf.keras.utils.to_categorical(y[idx["val"]], n_clases)
    n_features = X_tr_f.shape[2]

    model = construir_modelo(cfg, X_seq.shape[1], n_features, n_clases)
    n_param = int(model.count_params())
    steps = len(y_tr) // cfg["batch"] + 1
    lr = tf.keras.optimizers.schedules.CosineDecay(cfg["lr"], args.epochs * steps, alpha=1e-6)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=cfg["label_smoothing"]),
                  metrics=["accuracy"])
    tmp = Path(tempfile.mkdtemp(prefix="busqueda_hp_"))
    ruta_ckpt = tmp / "checkpoint.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience,
                                         restore_best_weights=True, verbose=1, mode="max"),
        tf.keras.callbacks.ModelCheckpoint(str(ruta_ckpt), monitor="val_accuracy",
                                           save_best_only=True, verbose=0, mode="max"),
    ]
    t_fit = time.perf_counter()
    hist = model.fit(X_tr_f, y_tr_oh, validation_data=(X_val_f, y_val_oh), epochs=args.epochs,
                     batch_size=cfg["batch"], callbacks=callbacks, verbose=2)
    tiempo_fit = time.perf_counter() - t_fit
    historial = {kk: [float(v) for v in vals] for kk, vals in hist.history.items()}
    epocas = len(historial["loss"])
    mejor_epoca = int(np.argmax(historial["val_accuracy"])) + 1
    del model
    tf.keras.backend.clear_session()

    model = tf.keras.models.load_model(str(ruta_ckpt))
    # Validación del checkpoint (la que decide)
    val_loss_ckpt, _ = model.evaluate(X_val_f, y_val_oh, verbose=0)
    mv = _metricas(model.predict(X_val_f, verbose=0), y[idx["val"]])
    # Prueba: una sola vez, solo se guarda
    X_te_f = add_velocity(X_seq[idx["test"]])
    y_te = y[idx["test"]]
    probs_te = model.predict(X_te_f, verbose=0)
    mt = _metricas(probs_te, y_te)
    pred = np.argmax(probs_te, axis=1)

    # Modelo sin estado del optimizador (≈ 1/3 del tamaño) en la carpeta del experimento
    pesos = model.get_weights()
    del model
    tf.keras.backend.clear_session()
    limpio = construir_modelo(cfg, X_seq.shape[1], n_features, n_clases)
    limpio.set_weights(pesos)
    ruta_modelo = d / "modelo.keras"
    limpio.save(str(ruta_modelo))
    del limpio
    tf.keras.backend.clear_session()
    shutil.rmtree(tmp, ignore_errors=True)
    from src.representacion import escribir_procedencia
    escribir_procedencia(ruta_modelo, args.processed, REPRESENTACION_CANONICA, SEQ_LENGTH,
                         "src/busqueda_hiperparametros.py", BASE_DIR)

    with open(d / "historial.json", "w", encoding="utf-8") as f:
        json.dump(historial, f)
    with open(d / "predicciones_prueba.json", "w", encoding="utf-8") as f:
        json.dump([{"muestra": p["conjuntos"]["test"][i], "real": int(y_te[i]), "predicha": int(pred[i]),
                    "prob_predicha": float(probs_te[i, pred[i]])} for i in range(len(y_te))], f)
    r = {
        "config": cfg["id"], "hiperparametros": {kk: cfg[kk] for kk in ESPACIO},
        "semilla": int(semilla), "pliegue": int(k), "senante_test": p["senante_test"],
        "n_parametros": n_param,
        "val_top1": float(max(historial["val_accuracy"])),
        "val_top1_checkpoint_reevaluado": mv["top1"],
        "val_top3": mv["top3"], "val_f1_macro": mv["f1_macro"], "val_loss": float(val_loss_ckpt),
        "test_top1": mt["top1"], "test_top3": mt["top3"], "test_f1_macro": mt["f1_macro"],
        "test_precision_macro": mt["precision_macro"], "test_recall_macro": mt["recall_macro"],
        "n_train_aumentado": int(len(y_tr)), "n_val": int(len(idx["val"])), "n_test": int(len(idx["test"])),
        "epocas_ejecutadas": int(epocas), "mejor_epoca": mejor_epoca,
        "tiempo_entrenamiento_s": round(tiempo_fit, 1),
        "tiempo_total_s": round(time.perf_counter() - t0, 1),
        "s_por_epoca": round(tiempo_fit / max(epocas, 1), 2),
        "epochs_max": args.epochs, "aug": args.aug, "patience": args.patience,
        "modelo": tl._rel(ruta_modelo),
        "fecha_fin": datetime.now().isoformat(timespec="seconds"),
    }
    with open(d / "resultado.json", "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    print(f"  -> {cfg['id']} s={semilla} k={k}: val {r['val_top1']:.3f} (reeval {mv['top1']:.3f}, "
          f"loss {r['val_loss']:.3f}) | test top-1 {r['test_top1']:.3f} | epocas {epocas} (mejor "
          f"{mejor_epoca}) | {r['tiempo_total_s']:.0f} s ({r['s_por_epoca']:.1f} s/epoca)", flush=True)
    del X_tr, X_tr_f, y_tr, y_tr_oh
    return r


def _leer(d):
    f = d / "resultado.json"
    return json.loads(f.read_text()) if f.exists() else None


def preparar(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    processed = Path(args.processed).resolve()
    args.processed = str(processed)
    X_padded, y, lengths, nombres, loso, pos = cargar_todo(processed, REPRESENTACION_CANONICA)
    canon = tl.importar_pipeline_canonico(REPRESENTACION_CANONICA)
    X_seq = tl.remuestrear_todas(X_padded, lengths, canon[3], SEQ_LENGTH)
    return out, X_seq, y, loso, pos, canon


def correr_si_falta(out, fase, cfg, semilla, k, loso, pos, X_seq, y, canon, args):
    d = dir_corrida(out, fase, cfg["id"], semilla, k)
    r = _leer(d)
    if r is not None:
        print(f"  ya existe {tl._rel(d)}; se salta", flush=True)
        return r
    p = loso["pliegues"][k]
    print(f"\n{'=' * 72}\n  {fase} | {cfg['id']} ({describir(cfg)}) | semilla {semilla} | pliegue {k} "
          f"(senante {p['senante_test']}) | {datetime.now().isoformat(timespec='seconds')}\n{'=' * 72}",
          flush=True)
    return entrenar_corrida(cfg, semilla, k, p, pos, X_seq, y, canon, d, args)


def cmd_fase1(args):
    out, X_seq, y, loso, pos, canon = preparar(args)
    folds = tl.parsear_folds(args.folds, len(loso["pliegues"]))
    configs = generar_configuraciones()
    if args.solo:
        configs = [c for c in configs if c["id"] in args.solo.split(",")]
    print(f"[{datetime.now().isoformat(timespec='seconds')}] fase 1 | {len(configs)} configuraciones | "
          f"semilla {SEMILLA_FASE1} | pliegues {folds} | epochs {args.epochs} | aug {args.aug} | "
          f"salida {tl._rel(out)}", flush=True)
    for cfg in configs:
        hechos = {}
        marca = out / "fase1" / cfg["id"] / "cortada.json"
        if marca.exists():
            print(f"  {cfg['id']} cortada previamente; se salta", flush=True)
            continue
        for k in folds:
            hechos[k] = correr_si_falta(out, "fase1", cfg, SEMILLA_FASE1, k, loso, pos, X_seq, y, canon, args)
            # Corte temprano: tras los pliegues de corte, contra la canónica de esta misma corrida.
            if (not cfg["canonica"] and not args.sin_corte and k == PLIEGUES_CORTE[-1]
                    and all(kk in hechos for kk in PLIEGUES_CORTE)):
                refs = [_leer(dir_corrida(out, "fase1", "c00", SEMILLA_FASE1, kk)) for kk in PLIEGUES_CORTE]
                if all(refs):
                    v_c = float(np.mean([hechos[kk]["val_top1"] for kk in PLIEGUES_CORTE]))
                    v_0 = float(np.mean([r["val_top1"] for r in refs]))
                    if v_c < v_0 - CORTE_PUNTOS - 1e-12:
                        info = {"config": cfg["id"], "val_media_pliegues_corte": v_c,
                                "val_media_canonica": v_0, "pliegues": PLIEGUES_CORTE,
                                "regla": f"val < canonica - {CORTE_PUNTOS}",
                                "fecha": datetime.now().isoformat(timespec="seconds")}
                        marca.write_text(json.dumps(info, indent=1))
                        print(f"  CORTE TEMPRANO {cfg['id']}: val {v_c:.3f} vs canonica {v_0:.3f}", flush=True)
                        break
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] fase 1 terminada", flush=True)


# ---------------------------------------------------------------------------
# Selección de finalistas (SOLO validación)
# ---------------------------------------------------------------------------

def tabla_fase(out, fase, semillas):
    """{cid: {semilla: {k: resultado}}} con lo que haya en disco."""
    tab = {}
    for f in sorted((Path(out) / fase).glob("c*/semilla_*/pliegue_*/resultado.json")):
        r = json.loads(f.read_text())
        if r["semilla"] in semillas:
            tab.setdefault(r["config"], {}).setdefault(r["semilla"], {})[r["pliegue"]] = r
    return tab


def cmd_seleccionar(args):
    out = Path(args.out).resolve()
    tab = tabla_fase(out, "fase1", [SEMILLA_FASE1])
    filas = []
    for c in generar_configuraciones():
        if c["canonica"]:
            continue
        rs = tab.get(c["id"], {}).get(SEMILLA_FASE1, {})
        if (out / "fase1" / c["id"] / "cortada.json").exists() or len(rs) < 5:
            continue
        # Solo campos de VALIDACIÓN y tamaño: la prueba no interviene.
        filas.append({"config": c["id"],
                      "val_top1_media": float(np.mean([rs[k]["val_top1"] for k in range(5)])),
                      "val_loss_media": float(np.mean([rs[k]["val_loss"] for k in range(5)])),
                      "n_parametros": rs[0]["n_parametros"]})
    filas.sort(key=lambda f: (-round(f["val_top1_media"], 6), f["val_loss_media"], f["n_parametros"]))
    fin = filas[:N_FINALISTAS]
    sal = {"fecha": datetime.now().isoformat(timespec="seconds"),
           "criterio": "mayor val top-1 media de 5 pliegues (semilla 2026); empate: menor val_loss "
                       "media; luego menos parametros. No se lee la prueba.",
           "ranking_validacion": filas, "finalistas": [f["config"] for f in fin]}
    (out / "finalistas.json").write_text(json.dumps(sal, indent=1))
    for i, f in enumerate(filas):
        print(f"{i + 1:2d}. {f['config']} val {f['val_top1_media']:.4f} loss {f['val_loss_media']:.4f} "
              f"params {f['n_parametros']:,}" + ("  <- finalista" if i < N_FINALISTAS else ""))


def cmd_fase2(args):
    out, X_seq, y, loso, pos, canon = preparar(args)
    sel = json.loads((out / "finalistas.json").read_text())
    configs = [config_por_id("c00")] + [config_por_id(c) for c in sel["finalistas"]]
    semillas = [int(s) for s in args.semillas.split(",")] if args.semillas else SEMILLAS_FASE2
    folds = tl.parsear_folds(args.folds, len(loso["pliegues"]))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] fase 2 | {[c['id'] for c in configs]} | "
          f"semillas {semillas} | pliegues {folds}", flush=True)
    # Orden semilla -> configuración -> pliegue: si hubiera que cortar, cada semilla queda completa.
    for s in semillas:
        for cfg in configs:
            for k in folds:
                correr_si_falta(out, "fase2", cfg, s, k, loso, pos, X_seq, y, canon, args)
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] fase 2 terminada", flush=True)


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

def otros_entrenamientos():
    """Procesos de Python que entrenan (no las shells de espera de otros experimentos, que contienen los
    mismos nombres en su línea de comandos)."""
    import re
    import subprocess
    r = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True)
    patron = re.compile(r"src/(experimento_pasos|experimento_velocidades|train_loso)\.py|"
                        r"src/busqueda_hiperparametros\.py (fase1|fase2)")
    activos = []
    for linea in r.stdout.splitlines():
        partes = linea.strip().split(None, 2)
        if len(partes) < 2 or int(partes[0]) == os.getpid():
            continue
        if "ython" in partes[1].split("/")[-1] and patron.search(linea):
            activos.append(linea.strip()[:200])
    return activos


def cmd_inferencia(args):
    out = Path(args.out).resolve()
    canon = tl.importar_pipeline_canonico(REPRESENTACION_CANONICA)
    tf = canon[0]
    sel = json.loads((out / "finalistas.json").read_text())
    ids = ["c00"] + sel["finalistas"]
    otros = otros_entrenamientos()
    if otros:
        print("AVISO: procesos de entrenamiento activos durante la medicion:", otros)
    rng = np.random.default_rng(0)
    filas = []
    for cid in ids:
        rutas = sorted((out / "fase2" / cid).glob("semilla_*/pliegue_*/modelo.keras"))[:args.modelos]
        for ruta in rutas:
            model = tf.keras.models.load_model(str(ruta))
            L, F = int(model.input_shape[1]), int(model.input_shape[2])
            x1 = rng.standard_normal((1, L, F)).astype(np.float32)
            for _ in range(args.calentamiento):
                model.predict(x1, verbose=0)
                model(x1, training=False)
            t_pred, t_call = [], []
            for _ in range(args.repeticiones):
                t0 = time.perf_counter()
                model.predict(x1, verbose=0)
                t_pred.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                model(x1, training=False)
                t_call.append((time.perf_counter() - t0) * 1000)
            fila = {"config": cid, "modelo": tl._rel(ruta), "n_parametros": int(model.count_params()),
                    "tamano_archivo_mb": round(ruta.stat().st_size / 2 ** 20, 2),
                    "ms_predict_lote1_mediana": float(np.median(t_pred)),
                    "ms_predict_lote1_p90": float(np.percentile(t_pred, 90)),
                    "ms_llamada_directa_mediana": float(np.median(t_call)),
                    "ms_llamada_directa_p90": float(np.percentile(t_call, 90))}
            filas.append(fila)
            print(f"  {cid} {tl._rel(ruta)}: {fila['n_parametros']:,} param | predict(1) mediana "
                  f"{fila['ms_predict_lote1_mediana']:.2f} ms (p90 {fila['ms_predict_lote1_p90']:.2f}) | "
                  f"model(x) {fila['ms_llamada_directa_mediana']:.2f} ms", flush=True)
            del model
            tf.keras.backend.clear_session()
    with open(out / "inferencia.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": datetime.now().isoformat(timespec="seconds"),
                   "metodo": f"model.predict(verbose=0) con lote de 1, como translator.py, y llamada "
                             f"directa model(x, training=False); {args.calentamiento} llamadas de "
                             f"calentamiento y {args.repeticiones} medidas por modelo; mediana y p90. "
                             f"Solo la red (sin MediaPipe ni remuestreo). CPU.",
                   "procesos_de_entrenamiento_activos": otros,
                   "entorno": tl.info_entorno(), "modelos": filas}, f, indent=1)


# ---------------------------------------------------------------------------
# Resumen y criterio
# ---------------------------------------------------------------------------

def _md(v):
    v = [float(x) for x in v]
    return {"media": float(np.mean(v)), "de": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "valores": v}


def pruebas_pareadas(dif):
    """Prueba de signos (binomial exacta, empates fuera) y Wilcoxon exacto sobre las diferencias."""
    from scipy.stats import binomtest, wilcoxon
    dif = np.asarray(dif, dtype=float)
    pos_, neg_ = int(np.sum(dif > 1e-12)), int(np.sum(dif < -1e-12))
    res = {"diferencias": dif.tolist(), "mejora": pos_, "empeora": neg_, "empate": int(len(dif) - pos_ - neg_)}
    res["signos_p_bilateral"] = float(binomtest(pos_, pos_ + neg_, 0.5).pvalue) if pos_ + neg_ else 1.0
    if np.any(np.abs(dif) > 1e-12):
        try:
            w = wilcoxon(dif, zero_method="zsplit", method="exact")
        except Exception:
            w = wilcoxon(dif, zero_method="zsplit")
        res["wilcoxon_estadistico"] = float(w.statistic)
        res["wilcoxon_p_bilateral"] = float(w.pvalue)
    else:
        res["wilcoxon_estadistico"], res["wilcoxon_p_bilateral"] = None, 1.0
    return res


METRICAS = ["val_top1", "val_f1_macro", "val_loss", "test_top1", "test_top3", "test_f1_macro",
            "epocas_ejecutadas", "mejor_epoca", "tiempo_total_s", "s_por_epoca"]


def cmd_resumen(args):
    from scipy.stats import spearmanr
    out = Path(args.out).resolve()
    configs = {c["id"]: c for c in generar_configuraciones()}

    # ---- Fase 1 ----
    t1 = tabla_fase(out, "fase1", [SEMILLA_FASE1])
    fase1 = []
    for cid, c in configs.items():
        rs = t1.get(cid, {}).get(SEMILLA_FASE1, {})
        cortada = (out / "fase1" / cid / "cortada.json").exists()
        fila = {"config": cid, "descripcion": describir(c), "hiperparametros": {k: c[k] for k in ESPACIO},
                "canonica": c["canonica"], "cortada": cortada, "pliegues_hechos": sorted(rs)}
        if rs:
            fila["n_parametros"] = next(iter(rs.values()))["n_parametros"]
            for m in METRICAS:
                fila[m + "_por_pliegue"] = [rs[k][m] if k in rs else None for k in range(5)]
            ks = sorted(rs)
            for m in METRICAS:
                fila[m + "_media"] = float(np.mean([rs[k][m] for k in ks]))
            fila["tiempo_total_config_s"] = float(sum(rs[k]["tiempo_total_s"] for k in ks))
        fase1.append(fila)
    completas = [f for f in fase1 if len(f["pliegues_hechos"]) == 5]
    corr = None
    if len(completas) >= 4:
        rho = spearmanr([f["val_top1_media"] for f in completas], [f["test_top1_media"] for f in completas])
        corr = {"spearman_rho": float(rho.statistic), "p": float(rho.pvalue), "n_configs": len(completas),
                "nota": "descriptivo: ¿ordena la validacion como la prueba LOSO?"}
    fin_path = out / "finalistas.json"
    finalistas = json.loads(fin_path.read_text())["finalistas"] if fin_path.exists() else []

    # Reproducción de la referencia N-30
    repro = None
    if REFERENCIA_N30.exists() and len(t1.get("c00", {}).get(SEMILLA_FASE1, {})) == 5:
        ref = json.loads(REFERENCIA_N30.read_text())
        rs = t1["c00"][SEMILLA_FASE1]
        repro = [{"pliegue": p["pliegue"], "ref_test_top1": p["top1"], "nuevo_test_top1": rs[p["pliegue"]]["test_top1"],
                  "ref_val": p["mejor_val_accuracy"], "nuevo_val": rs[p["pliegue"]]["val_top1"],
                  "ref_epocas": p["epocas_ejecutadas"], "nuevo_epocas": rs[p["pliegue"]]["epocas_ejecutadas"]}
                 for p in ref["pliegues"]]

    # ---- Fase 2 ----
    t2 = tabla_fase(out, "fase2", SEMILLAS_FASE2)
    ids2 = ["c00"] + finalistas
    sem_comunes = [s for s in SEMILLAS_FASE2 if all(len(t2.get(c, {}).get(s, {})) == 5 for c in ids2)]
    fase2, criterio = {}, None
    if finalistas and len(sem_comunes) >= 2:
        for cid in ids2:
            r = {"config": cid, "descripcion": describir(configs[cid]), "semillas": sem_comunes}
            for m in METRICAS:
                pps = {s: [t2[cid][s][k][m] for k in range(5)] for s in sem_comunes}
                por_pliegue = [float(np.mean([pps[s][k] for s in sem_comunes])) for k in range(5)]
                por_sem = {s: float(np.mean(pps[s])) for s in sem_comunes}
                r[m] = {"por_semilla_por_pliegue": {str(s): v for s, v in pps.items()},
                        "por_pliegue_media_semillas": por_pliegue,
                        "por_semilla_media_pliegues": {str(s): v for s, v in por_sem.items()},
                        "entre_pliegues": _md(por_pliegue),
                        "de_entre_semillas": float(np.std(list(por_sem.values()), ddof=1)),
                        "media_global": float(np.mean(list(por_sem.values())))}
                # DE de celda (una corrida) entre semillas, por pliegue
                r[m]["de_celda_por_pliegue"] = [float(np.std([pps[s][k] for s in sem_comunes], ddof=1)) for k in range(5)]
            r["n_parametros"] = t2[cid][sem_comunes[0]][0]["n_parametros"]
            fase2[cid] = r
        var_sem = [fase2[c]["val_top1"]["de_entre_semillas"] ** 2 for c in ids2]
        sigma = float(np.sqrt(np.mean(var_sem)))
        var_celda = [v ** 2 for c in ids2 for v in fase2[c]["val_top1"]["de_celda_por_pliegue"]]
        sigma_celda = float(np.sqrt(np.mean(var_celda)))
        margen = max(sigma, MARGEN_MINIMO)
        v0 = np.array(fase2["c00"]["val_top1"]["por_pliegue_media_semillas"])
        t0 = np.array(fase2["c00"]["test_top1"]["por_pliegue_media_semillas"])
        f0 = np.array(fase2["c00"]["test_f1_macro"]["por_pliegue_media_semillas"])
        por_cfg, candidatos = {}, []
        for cid in finalistas:
            vc = np.array(fase2[cid]["val_top1"]["por_pliegue_media_semillas"])
            dif = vc - v0
            pp = pruebas_pareadas(dif)
            delta = float(np.mean(dif))
            a, b, cc = delta > 0, pp["mejora"] >= 4, delta > margen + 1e-12
            # Caída búsqueda -> confirmación
            f1row = next(f for f in fase1 if f["config"] == cid)
            f1can = next(f for f in fase1 if f["config"] == "c00")
            por_cfg[cid] = {
                "delta_val_media": delta, "supera_en_media": bool(a), "pliegues_mejor": pp["mejora"],
                "cumple_4_de_5": bool(b), "supera_margen": bool(cc), "cumple_criterio": bool(a and b and cc),
                "pareada_validacion": pp,
                "pareada_prueba_top1_descriptiva": pruebas_pareadas(
                    np.array(fase2[cid]["test_top1"]["por_pliegue_media_semillas"]) - t0),
                "pareada_prueba_f1_descriptiva": pruebas_pareadas(
                    np.array(fase2[cid]["test_f1_macro"]["por_pliegue_media_semillas"]) - f0),
                "caida": {
                    "val_fase1": f1row["val_top1_media"], "val_fase2": fase2[cid]["val_top1"]["media_global"],
                    "caida_val": f1row["val_top1_media"] - fase2[cid]["val_top1"]["media_global"],
                    "ventaja_val_fase1": f1row["val_top1_media"] - f1can["val_top1_media"],
                    "ventaja_val_fase2": delta,
                    "test_fase1": f1row["test_top1_media"], "test_fase2": fase2[cid]["test_top1"]["media_global"],
                    "ventaja_test_fase1": f1row["test_top1_media"] - f1can["test_top1_media"],
                    "ventaja_test_fase2": fase2[cid]["test_top1"]["media_global"] - fase2["c00"]["test_top1"]["media_global"],
                },
            }
            if a and b and cc:
                candidatos.append(cid)
        f1can = next(f for f in fase1 if f["config"] == "c00")
        caida_can = {"val_fase1": f1can["val_top1_media"], "val_fase2": fase2["c00"]["val_top1"]["media_global"],
                     "caida_val": f1can["val_top1_media"] - fase2["c00"]["val_top1"]["media_global"],
                     "test_fase1": f1can["test_top1_media"], "test_fase2": fase2["c00"]["test_top1"]["media_global"]}
        if candidatos:
            mejor = max(fase2[c]["val_top1"]["media_global"] for c in candidatos)
            empatados = [c for c in candidatos if mejor - fase2[c]["val_top1"]["media_global"] < margen]
            elegido = min(empatados, key=lambda c: fase2[c]["n_parametros"])
        else:
            elegido = "c00"
        criterio = {"sigma_semilla_agrupada": sigma, "sigma_celda_agrupada_referencia": sigma_celda,
                    "margen": margen, "por_config": por_cfg, "candidatos_que_cumplen": candidatos,
                    "recomendada": elegido, "caida_canonica": caida_can}

    salida = {"fecha": datetime.now().isoformat(timespec="seconds"),
              "criterio_documento": "docs/BUSQUEDA_HIPERPARAMETROS.md §2",
              "processed_dir": tl._rel(PROCESSED_CANONICO), "representacion": REPRESENTACION_CANONICA,
              "seq_length": SEQ_LENGTH, "particion": tl._rel(tl.ARCHIVO_LOSO),
              "espacio": ESPACIO, "canonica": CANONICA, "semilla_muestreo": SEMILLA_MUESTREO,
              "semilla_fase1": SEMILLA_FASE1, "semillas_fase2": SEMILLAS_FASE2,
              "semillas_fase2_completas": sem_comunes,
              "corte": {"pliegues": PLIEGUES_CORTE, "puntos": CORTE_PUNTOS},
              "fase1": fase1, "correlacion_val_test_fase1": corr, "reproduccion_N30": repro,
              "finalistas": finalistas, "fase2": fase2, "criterio": criterio,
              "inferencia": json.loads((out / "inferencia.json").read_text()) if (out / "inferencia.json").exists() else None,
              "entorno": tl.info_entorno()}
    destino = Path(args.json) if args.json else JSON_DOCS
    destino.write_text(json.dumps(salida, indent=1, ensure_ascii=False))
    print(f"escrito {tl._rel(destino)}")

    # ---- Impresión (tablas en markdown para el reporte) ----
    print("\nFASE 1 (semilla 2026)")
    print("| id | configuración | parámetros | val top-1 media | val por pliegue | prueba top-1 media | prueba F1 | épocas | min |")
    print("|---|---|---:|---:|---|---:|---:|---:|---:|")
    for f in sorted(fase1, key=lambda f: -(f.get("val_top1_media") or 0)):
        if not f["pliegues_hechos"]:
            continue
        vp = " / ".join("—" if v is None else f"{v:.2f}" for v in f["val_top1_por_pliegue"])
        etiqueta = f["config"] + (" (canónica)" if f["canonica"] else "") + (" ✂" if f["cortada"] else "")
        print(f"| {etiqueta} | {f['descripcion']} | {f['n_parametros']:,} | {f['val_top1_media']:.3f} | {vp} | "
              f"{f['test_top1_media']:.3f} | {f['test_f1_macro_media']:.3f} | {f['epocas_ejecutadas_media']:.0f} | "
              f"{f['tiempo_total_config_s'] / 60:.0f} |")
    if corr:
        print(f"\nSpearman val vs prueba (fase 1, {corr['n_configs']} completas): rho {corr['spearman_rho']:.3f} p {corr['p']:.3f}")
    if repro:
        print("reproduccion N-30:", [(r["pliegue"], r["ref_test_top1"], r["nuevo_test_top1"]) for r in repro])
    if criterio:
        print(f"\nFASE 2 (semillas {sem_comunes}) | sigma semilla {criterio['sigma_semilla_agrupada']:.4f} | "
              f"sigma celda {criterio['sigma_celda_agrupada_referencia']:.4f} | margen {criterio['margen']:.4f}")
        for cid in ids2:
            r = fase2[cid]
            print(f"  {cid}: val {r['val_top1']['media_global']:.4f} (DE sem {r['val_top1']['de_entre_semillas']:.4f}) "
                  f"por pliegue {[round(v, 3) for v in r['val_top1']['por_pliegue_media_semillas']]} | test "
                  f"{r['test_top1']['media_global']:.4f} ± {r['test_top1']['entre_pliegues']['de']:.4f} | F1 "
                  f"{r['test_f1_macro']['media_global']:.4f}")
        for cid, c in criterio["por_config"].items():
            print(f"  {cid}: delta val {c['delta_val_media']:+.4f} | pliegues mejores {c['pliegues_mejor']}/5 | "
                  f"signos p {c['pareada_validacion']['signos_p_bilateral']:.3f} | wilcoxon p "
                  f"{c['pareada_validacion']['wilcoxon_p_bilateral']:.3f} | cumple {c['cumple_criterio']} | caida val "
                  f"{c['caida']['caida_val']:+.4f} | ventaja f1->f2 {c['caida']['ventaja_val_fase1']:+.4f} -> "
                  f"{c['caida']['ventaja_val_fase2']:+.4f}")
        print(f"recomendada: {criterio['recomendada']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verificar")
    v.add_argument("--guardar", action="store_true")
    f1 = sub.add_parser("fase1")
    f1.add_argument("--solo", default="", help="ids separados por coma (p. ej. c00,c03)")
    f1.add_argument("--sin-corte", action="store_true")
    f2 = sub.add_parser("fase2")
    f2.add_argument("--semillas", default="")
    for p in (f1, f2):
        p.add_argument("--folds", default="5")
        p.add_argument("--processed", default=str(PROCESSED_CANONICO))
        p.add_argument("--epochs", type=int, default=200)
        p.add_argument("--aug", type=int, default=10)
        p.add_argument("--patience", type=int, default=25)
    s = sub.add_parser("seleccionar")
    i = sub.add_parser("inferencia")
    i.add_argument("--calentamiento", type=int, default=10)
    i.add_argument("--repeticiones", type=int, default=100)
    i.add_argument("--modelos", type=int, default=3, help="modelos por configuracion")
    r = sub.add_parser("resumen")
    r.add_argument("--json", default="")
    for p in (v, f1, f2, s, i, r):
        p.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    {"verificar": cmd_verificar, "fase1": cmd_fase1, "seleccionar": cmd_seleccionar, "fase2": cmd_fase2,
     "inferencia": cmd_inferencia, "resumen": cmd_resumen}[args.cmd](args)


if __name__ == "__main__":
    main()
