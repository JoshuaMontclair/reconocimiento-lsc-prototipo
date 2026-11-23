"""
Experimento: ¿alguna arquitectura recomendada por la literatura acierta más que el modelo canónico con
personas nuevas? (docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md; revisión en
docs/REVISION_LITERATURA_ARQUITECTURAS.md). LOSO de 5 pliegues, particiones D-06.

Todos los brazos usan exactamente la misma entrada canónica: HolisticLandmarker -> normalización por
hombros -> «mascara» (D-07) -> «manos» (126) -> 30 pasos -> aumentación ×10 de train_loso (espejo sobre
las 126 posiciones) -> add_velocity -> 30 × 252. Lo único que cambia es el modelo.

Brazos:
  R0  canónico (control)   build_model de train_final_100 (714.291 parámetros). NO se reentrena: se
                           reutiliza el brazo A de models/experimento_mano_ausente_2026-10-02 (semillas
                           2026-2028), tras verificar que su configuración es la canónica (subcomando
                           verificar).
  R1  c07                  construir_modelo(c07) de busqueda_hiperparametros (389.811 parámetros),
                           lr 5e-4, semillas 2026-2028.
  M   MiniRocket + ridge   src/experimento_arquitecturas_minirocket.py, en un entorno aparte (sktime);
                           este script solo exporta los tensores de cada pliegue (subcomando exportar).
  T   TCN pequeña + GAP    src/arquitecturas_literatura.construir_tcn.
  G   ST-GCN pequeño       src/arquitecturas_literatura.construir_stgcn (opcional).

Entrenamiento de R1, T y G: el mismo orden de pasos que train_loso.entrenar_y_evaluar (semilla ->
aumentación -> velocidades -> modelo -> Adam con CosineDecay -> fit con EarlyStopping y ModelCheckpoint
sobre val_accuracy -> se recarga el checkpoint -> prueba una sola vez).

Subcomandos:
  verificar   R0 reutilizable (configuración canónica, predicciones idénticas en dos experimentos),
              parámetros de cada red, grafo de G y coherencia de la remodelación con el espejo.
  exportar    .npz por (semilla, pliegue) con el entrenamiento aumentado ×10 (idéntico al de las redes)
              para MiniRocket.
  entrenar    combinaciones (brazo, semilla, pliegue) que falten; reanudable.
  inferencia  parámetros y latencia con lote de 1 (como translator.py) de R0, R1, T y G.
  resumen     agrega, aplica el criterio prerregistrado y escribe el JSON de datos crudos.
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
from src import busqueda_hiperparametros as bh  # noqa: E402
from src import arquitecturas_literatura as al  # noqa: E402

PROCESSED_CANONICO = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
REPRESENTACION = "manos"
SEQ_LENGTH = 30
DEFAULT_OUT = BASE_DIR / "models" / "arquitecturas_literatura_2026-10-02"
JSON_DOCS = BASE_DIR / "docs" / "EXPERIMENTO_ARQUITECTURAS_LITERATURA.json"
R0_DIR = BASE_DIR / "models" / "experimento_mano_ausente_2026-10-02" / "A_mascara"
R0_DIR_ESPEJO = BASE_DIR / "models" / "experimento_velocidades_2026-09-30" / "A_con_velocidades"
C07_FASE1 = BASE_DIR / "models" / "busqueda_hiperparametros" / "fase1" / "c07"
SEMILLAS = [2026, 2027, 2028]
PISO_MARGEN = 0.01

BRAZOS = {
    "R0": {"carpeta": "R0_canonico", "descripcion": "canonico CNN+BiLSTM+atencion (build_model), reutilizado"},
    "R1": {"carpeta": "R1_c07", "descripcion": "c07 de la busqueda de hiperparametros"},
    "M": {"carpeta": "M_minirocket", "descripcion": "MiniRocket multivariado + ridge (sktime)"},
    "T": {"carpeta": "T_tcn", "descripcion": "TCN pequena con GAP (4 bloques residuales dilatados)"},
    "G": {"carpeta": "G_stgcn", "descripcion": "ST-GCN pequeno sobre el grafo de las dos manos"},
}
REDES_ENTRENABLES = ("R1", "T", "G")


def _lista_int(texto):
    return [int(t) for t in str(texto).split(",") if t.strip() != ""]


def dir_corrida(out, brazo, semilla, k):
    return Path(out) / BRAZOS[brazo]["carpeta"] / f"semilla_{semilla}" / f"pliegue_{k}"


def cargar_todo(processed=PROCESSED_CANONICO):
    X_padded, y, lengths, nombres, senantes = tl.cargar_datos(processed, REPRESENTACION)
    with open(tl.ARCHIVO_LOSO, encoding="utf-8") as f:
        loso = json.load(f)
    n_clases = int(y.max()) + 1
    for p in loso["pliegues"]:
        tl.verificar_particion(p["conjuntos"], nombres, senante_test=p["senante_test"], n_clases=n_clases)
    pos = {n: i for i, n in enumerate(nombres)}
    return X_padded, y, lengths, nombres, senantes, loso, pos


def preparar(processed=PROCESSED_CANONICO):
    X_padded, y, lengths, nombres, senantes, loso, pos = cargar_todo(processed)
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    X_seq = tl.remuestrear_todas(X_padded, lengths, canon[3], SEQ_LENGTH)
    return X_seq, y, nombres, senantes, loso, pos, canon


# ---------------------------------------------------------------------------
# Constructores e hiperparámetros por brazo
# ---------------------------------------------------------------------------

def constructor(brazo):
    """(función n_frames, n_features, n_classes -> modelo, hiperparámetros de entrenamiento)."""
    if brazo == "R1":
        cfg = bh.config_por_id("c07")
        return (lambda nf, nfe, nc: bh.construir_modelo(cfg, nf, nfe, nc),
                {"lr": cfg["lr"], "batch": cfg["batch"], "label_smoothing": cfg["label_smoothing"],
                 "arquitectura": {k: cfg[k] for k in bh.ESPACIO}})
    if brazo == "T":
        return (lambda nf, nfe, nc: al.construir_tcn(nf, nfe, nc),
                dict(al.ENTRENAMIENTO_COMUN, arquitectura=al.CONFIG_T))
    if brazo == "G":
        return (lambda nf, nfe, nc: al.construir_stgcn(nf, nfe, nc),
                dict(al.ENTRENAMIENTO_COMUN, arquitectura=al.CONFIG_G))
    raise SystemExit(f"brazo {brazo} no se entrena con este script")


# ---------------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------------

def leer_r0(semilla, k):
    d = R0_DIR / f"semilla_{semilla}" / f"pliegue_{k}"
    m = json.loads((d / "metricas_mano_ausente.json").read_text(encoding="utf-8"))
    pred = json.loads((d / "predicciones.json").read_text(encoding="utf-8"))
    return m, pred, d


def cmd_verificar(args):
    informe = {"fecha": datetime.now().isoformat(timespec="seconds"), "comprobaciones": []}

    def ok(texto):
        print(f"[ok] {texto}", flush=True)
        informe["comprobaciones"].append(texto)

    # 1. R0: configuración canónica y predicciones idénticas en dos experimentos independientes.
    esperado = {"epochs_max": 200, "aug": 10, "batch_size": 64, "patience": 25, "dropout": 0.3,
                "seq_length": 30, "processed": "data/processed_holistic_mascara_rostro", "representacion": "manos"}
    filas = []
    for s in SEMILLAS:
        for k in range(5):
            m, pred, d = leer_r0(s, k)
            assert m["regla"] == "mascara" and m["brazo"] == "A", (s, k)
            assert m["config"] == esperado, (s, k, m["config"])
            assert m["n_parametros"] == bh.PARAMETROS_CANONICOS
            assert m["n_train_aumentado"] == 10 * m["n_train_original"]
            proc = json.loads((d / "modelo.procedencia.json").read_text(encoding="utf-8"))
            assert proc["mano_ausente"] == "mascara" and proc["representacion"] == "manos"
            assert proc["n_rasgos_entrada"] == 252 and proc["seq_length"] == 30
            d2 = R0_DIR_ESPEJO / f"semilla_{s}" / f"pliegue_{k}"
            pred2 = json.loads((d2 / "predicciones.json").read_text(encoding="utf-8"))
            m2 = json.loads((d2 / "metricas.json").read_text(encoding="utf-8"))
            iguales = ([(p["muestra"], p["predicha"]) for p in pred] == [(p["muestra"], p["predicha"]) for p in pred2]
                       and abs(m2["mejor_val_accuracy"] - m["mejor_val_accuracy"]) < 1e-12
                       and m2["epocas_ejecutadas"] == m["epocas_ejecutadas"])
            assert iguales, (s, k)
            filas.append({"semilla": s, "pliegue": k, "val": m["mejor_val_accuracy"], "top1": m["top1"],
                          "epocas": m["epocas_ejecutadas"]})
    ok("R0: las 15 corridas del brazo A de experimento_mano_ausente tienen la configuracion canonica "
       "(processed_holistic_mascara_rostro, manos, mascara, aug x10, 200 epocas, paciencia 25, lote 64, "
       "dropout 0,3, 30 pasos, 714.291 parametros)")
    ok("R0: sus predicciones de prueba, val y epocas coinciden exactamente con las del brazo A de "
       "experimento_velocidades_2026-09-30 (dos experimentos independientes, mismas semillas)")
    ref = json.loads(bh.REFERENCIA_N30.read_text(encoding="utf-8"))
    for p in ref["pliegues"]:
        f = [x for x in filas if x["semilla"] == 2026 and x["pliegue"] == p["pliegue"]][0]
        assert abs(f["top1"] - p["top1"]) < 1e-12 and abs(f["val"] - p["mejor_val_accuracy"]) < 1e-12
    ok("R0 semilla 2026 reproduce la referencia N-30 (0,460 / 0,533 / 0,606 / 0,615 / 0,685)")
    informe["r0"] = filas

    # 2. La entrada de R0 es la misma X_seq que usan R1, T, G y M (train_loso.remuestrear_todas).
    X_padded, y, lengths, nombres, senantes, loso, pos = cargar_todo()
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    tf, build_model, add_velocity, resample_sequence, transformaciones = canon
    from src import mano_ausente_variantes as mav
    a = mav.remuestrear_todas_con_regla(X_padded, lengths, "mascara", SEQ_LENGTH)
    X_seq = tl.remuestrear_todas(X_padded, lengths, resample_sequence, SEQ_LENGTH)
    assert np.array_equal(a, X_seq)
    ok(f"la X_seq de este experimento (train_loso.remuestrear_todas) es bit a bit la del brazo A: {X_seq.shape}")

    # 3. Parámetros de cada red.
    n_par = {}
    tf.keras.utils.set_random_seed(0)
    m0 = build_model(SEQ_LENGTH, 252, 50, dropout=0.3)
    n_par["R0"] = int(m0.count_params())
    for b in REDES_ENTRENABLES:
        fn, hp = constructor(b)
        mm = fn(SEQ_LENGTH, 252, 50)
        n_par[b] = int(mm.count_params())
        tf.keras.backend.clear_session()
    assert n_par["R0"] == bh.PARAMETROS_CANONICOS and n_par["R1"] == 389_811
    ok(f"parametros: {n_par}")
    informe["n_parametros"] = n_par

    # 4. Grafo de G.
    A = al.adyacencia_dos_manos()
    assert np.array_equal(A, A.T) and np.all(np.diag(A) == 1)
    assert A[:21, 21:].sum() == 0 and A[21:, :21].sum() == 0
    assert int((A[:21, :21].sum() - 21) / 2) == 21 and int((A[21:, 21:].sum() - 21) / 2) == 21
    An = al.adyacencia_normalizada()
    assert np.allclose(An, An.T)
    ok("grafo de G: 42 nodos, 21 aristas de MediaPipe por mano + autoconexiones, sin aristas entre manos; "
       "adyacencia simetrica normalizada D^-1/2 (A+I) D^-1/2")

    # 5. Remodelación coherente con el espejo y con el orden de columnas de «manos».
    from src.representacion import columnas, espejar
    cols = columnas("manos")
    assert np.array_equal(cols, np.arange(99, 225)), "manos = mano_izq (99:162) + mano_der (162:225)"
    Xp = X_seq[:64]
    F = add_velocity(Xp)
    Fm = add_velocity(espejar(Xp, "manos"))
    Nn, Nm = al.remodelar_a_nodos_np(F), al.remodelar_a_nodos_np(Fm)
    # nodo v = 21*mano + punto; canales [x, y, z, vx, vy, vz]
    assert np.array_equal(Nn[..., 0:3], Xp.reshape(64, SEQ_LENGTH, 42, 3))
    swap = np.r_[np.arange(21, 42), np.arange(0, 21)]
    signo = np.array([-1, 1, 1, -1, 1, 1], dtype=np.float32)
    assert np.allclose(Nm, Nn[:, :, swap, :] * signo, atol=1e-6)
    ok("remodelacion de G: (30,252) -> (30,42,6) con canales [x,y,z,vx,vy,vz] y nodo 21*mano+punto; el "
       "espejo de la aumentacion (aplicado antes de add_velocity) equivale a intercambiar los nodos de las "
       "dos manos y negar x y vx: la remodelacion sigue siendo coherente despues de aumentar")
    # La capa Reshape/Permute del modelo hace lo mismo que la versión numpy.
    mg = al.construir_stgcn()
    sub = tf.keras.Model(mg.input, mg.layers[3].output)   # Input, Reshape, Permute, Reshape
    assert np.allclose(sub(F[:4]).numpy().reshape(4, SEQ_LENGTH, 42, 6), Nn[:4])
    ok("las capas Reshape/Permute de G producen exactamente la remodelacion numpy")
    tf.keras.backend.clear_session()

    # 6. c07 de la fase 1 (semilla 2026) disponible para cotejar la reproducción de R1.
    c07 = bh.config_por_id("c07")
    informe["c07"] = {k: c07[k] for k in bh.ESPACIO}
    ok(f"c07 = {bh.describir(c07)}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "verificacion.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {tl._rel(out / 'verificacion.json')}")


# ---------------------------------------------------------------------------
# Exportación para MiniRocket
# ---------------------------------------------------------------------------

def cmd_exportar(args):
    """Por (semilla, pliegue): el entrenamiento aumentado ×10 con velocidades, IDÉNTICO al que ven las
    redes con esa semilla (misma llamada a set_random_seed y a train_loso.aumentar), su señante (para
    la validación cruzada interna por señante de la ridge), y validación y prueba."""
    X_seq, y, nombres, senantes, loso, pos, canon = preparar()
    tf, _, add_velocity, _, transformaciones = canon
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    lista_sen = sorted(set(senantes))
    sen_idx = np.array([lista_sen.index(s) for s in senantes], dtype=np.int64)
    for s in _lista_int(args.semillas):
        for k in tl.parsear_folds(args.folds, 5):
            ruta = dest / f"semilla_{s}_pliegue_{k}.npz"
            if ruta.exists():
                print(f"  ya existe {ruta.name}")
                continue
            p = loso["pliegues"][k]
            idx = {kk: np.array([pos[n] for n in p["conjuntos"][kk]], dtype=np.int64) for kk in p["conjuntos"]}
            tf.keras.utils.set_random_seed(s)
            # La etiqueta se codifica con el señante (y*10 + señante) para conservarlo tras la permutación
            # de aumentar; la secuencia de números aleatorios es la misma que en entrenar_y_evaluar.
            yc = y[idx["train"]] * 10 + sen_idx[idx["train"]]
            X_tr, yc_tr = tl.aumentar(X_seq[idx["train"]], yc, args.aug, transformaciones)
            # Comprobación: con la etiqueta sin codificar se obtiene el mismo orden.
            tf.keras.utils.set_random_seed(s)
            X_tr2, y_tr2 = tl.aumentar(X_seq[idx["train"]], y[idx["train"]], args.aug, transformaciones)
            assert np.array_equal(X_tr, X_tr2) and np.array_equal(yc_tr // 10, y_tr2)
            np.savez(ruta, X_tr=add_velocity(X_tr).astype(np.float32), y_tr=(yc_tr // 10).astype(np.int64),
                     g_tr=(yc_tr % 10).astype(np.int64),
                     X_val=add_velocity(X_seq[idx["val"]]).astype(np.float32), y_val=y[idx["val"]],
                     X_te=add_velocity(X_seq[idx["test"]]).astype(np.float32), y_te=y[idx["test"]],
                     nombres_val=np.array(p["conjuntos"]["val"]), nombres_te=np.array(p["conjuntos"]["test"]),
                     semilla=s, pliegue=k, senante_test=p["senante_test"], aug=args.aug,
                     senantes=np.array(lista_sen))
            print(f"  {ruta.name}: train {X_tr.shape[0]} (x{args.aug}) | val {len(idx['val'])} | "
                  f"test {len(idx['test'])}", flush=True)


# ---------------------------------------------------------------------------
# Entrenamiento de R1, T y G
# ---------------------------------------------------------------------------

def _metricas(probs, y_real):
    from src.train_final_100 import metricas_basicas
    return metricas_basicas(probs, y_real)


def entrenar_corrida(brazo, semilla, k, p, pos, X_seq, y, canon, d, args):
    tf, _, add_velocity, _, transformaciones = canon
    fn, hp = constructor(brazo)
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

    model = fn(X_seq.shape[1], n_features, n_clases)
    n_param = int(model.count_params())
    steps = len(y_tr) // hp["batch"] + 1
    lr = tf.keras.optimizers.schedules.CosineDecay(hp["lr"], args.epochs * steps, alpha=1e-6)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=hp["label_smoothing"]),
                  metrics=["accuracy"])
    tmp = Path(tempfile.mkdtemp(prefix="arquitecturas_"))
    ruta_ckpt = tmp / "checkpoint.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience,
                                         restore_best_weights=True, verbose=1, mode="max"),
        tf.keras.callbacks.ModelCheckpoint(str(ruta_ckpt), monitor="val_accuracy",
                                           save_best_only=True, verbose=0, mode="max"),
    ]
    t_fit = time.perf_counter()
    hist = model.fit(X_tr_f, y_tr_oh, validation_data=(X_val_f, y_val_oh), epochs=args.epochs,
                     batch_size=hp["batch"], callbacks=callbacks, verbose=2)
    tiempo_fit = time.perf_counter() - t_fit
    historial = {kk: [float(v) for v in vals] for kk, vals in hist.history.items()}
    epocas = len(historial["loss"])
    mejor_epoca = int(np.argmax(historial["val_accuracy"])) + 1
    del model
    tf.keras.backend.clear_session()

    model = tf.keras.models.load_model(str(ruta_ckpt))
    val_loss_ckpt, _ = model.evaluate(X_val_f, y_val_oh, verbose=0)
    probs_val = model.predict(X_val_f, verbose=0)
    mv = _metricas(probs_val, y[idx["val"]])
    X_te_f = add_velocity(X_seq[idx["test"]])
    y_te = y[idx["test"]]
    probs_te = model.predict(X_te_f, verbose=0)     # prueba: una sola vez, solo se guarda
    mt = _metricas(probs_te, y_te)
    pred = np.argmax(probs_te, axis=1)
    top3 = np.argsort(-probs_te, axis=1)[:, :3]

    pesos = model.get_weights()
    del model
    tf.keras.backend.clear_session()
    limpio = fn(X_seq.shape[1], n_features, n_clases)
    limpio.set_weights(pesos)
    ruta_modelo = d / "modelo.keras"
    limpio.save(str(ruta_modelo))
    del limpio
    tf.keras.backend.clear_session()
    shutil.rmtree(tmp, ignore_errors=True)
    from src.representacion import escribir_procedencia, ruta_procedencia
    escribir_procedencia(ruta_modelo, args.processed, REPRESENTACION, SEQ_LENGTH,
                         "src/experimento_arquitecturas.py", BASE_DIR)
    rp = ruta_procedencia(ruta_modelo)
    datos = json.loads(rp.read_text(encoding="utf-8"))
    datos["brazo_experimento_arquitecturas"] = brazo
    datos["nota"] = ("modelo del experimento de arquitecturas de la literatura "
                     "(docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md). NO es el modelo canonico."
                     + (" Para cargarlo hay que importar antes src.arquitecturas_literatura y llamar "
                        "capa_conv_grafo() (capa ConvGrafo)." if brazo == "G" else ""))
    rp.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")

    with open(d / "historial.json", "w", encoding="utf-8") as f:
        json.dump(historial, f)
    with open(d / "predicciones.json", "w", encoding="utf-8") as f:
        json.dump([{"muestra": p["conjuntos"]["test"][i], "real": int(y_te[i]), "predicha": int(pred[i]),
                    "top3": [int(t) for t in top3[i]], "prob_predicha": float(probs_te[i, pred[i]])}
                   for i in range(len(y_te))], f)
    np.savez_compressed(d / "probabilidades.npz", val=probs_val.astype(np.float32),
                        test=probs_te.astype(np.float32), nombres_val=np.array(p["conjuntos"]["val"]),
                        nombres_test=np.array(p["conjuntos"]["test"]))
    r = {
        "brazo": brazo, "semilla": int(semilla), "pliegue": int(k), "senante_test": p["senante_test"],
        "hiperparametros": hp, "n_parametros": n_param,
        "val_top1": float(max(historial["val_accuracy"])),
        "val_top1_checkpoint_reevaluado": mv["top1"],
        "val_top3": mv["top3"], "val_f1_macro": mv["f1_macro"], "val_loss": float(val_loss_ckpt),
        "test_top1": mt["top1"], "test_top3": mt["top3"], "test_f1_macro": mt["f1_macro"],
        "test_precision_macro": mt["precision_macro"], "test_recall_macro": mt["recall_macro"],
        "n_train_aumentado": int(len(y_tr)), "n_val": int(len(idx["val"])), "n_test": int(len(idx["test"])),
        "epocas_ejecutadas": int(epocas), "mejor_epoca": mejor_epoca,
        "tiempo_entrenamiento_s": round(tiempo_fit, 1), "tiempo_total_s": round(time.perf_counter() - t0, 1),
        "s_por_epoca": round(tiempo_fit / max(epocas, 1), 2),
        "epochs_max": args.epochs, "aug": args.aug, "patience": args.patience,
        "processed": tl._rel(Path(args.processed)), "representacion": REPRESENTACION, "seq_length": SEQ_LENGTH,
        "modelo": tl._rel(ruta_modelo), "fecha_fin": datetime.now().isoformat(timespec="seconds"),
    }
    with open(d / "resultado.json", "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1, ensure_ascii=False)
    print(f"  -> {brazo} s={semilla} k={k}: val {r['val_top1']:.3f} (reeval {mv['top1']:.3f}) | test top-1 "
          f"{r['test_top1']:.3f} | epocas {epocas} (mejor {mejor_epoca}) | {r['tiempo_total_s']:.0f} s "
          f"({r['s_por_epoca']:.2f} s/epoca) | {n_param:,} param", flush=True)
    return r


def cmd_entrenar(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.processed = str(Path(args.processed).resolve())
    X_seq, y, nombres, senantes, loso, pos, canon = preparar(Path(args.processed))
    brazos = [b.strip().upper() for b in args.brazos.split(",") if b.strip()]
    for b in brazos:
        if b not in REDES_ENTRENABLES:
            raise SystemExit(f"--brazos: {b} no se entrena aqui ({REDES_ENTRENABLES})")
    semillas = _lista_int(args.semillas)
    folds = tl.parsear_folds(args.folds, len(loso["pliegues"]))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] experimento_arquitecturas entrenar | brazos {brazos}"
          f" | semillas {semillas} | pliegues {folds} | epochs {args.epochs} | aug x{args.aug} | paciencia "
          f"{args.patience} | salida {tl._rel(out)}", flush=True)
    for b in brazos:
        for s in semillas:
            for k in folds:
                d = dir_corrida(out, b, s, k)
                if (d / "resultado.json").exists():
                    print(f"  ya existe {tl._rel(d)}; se salta", flush=True)
                    continue
                p = loso["pliegues"][k]
                print(f"\n{'=' * 72}\n  brazo {b} ({BRAZOS[b]['descripcion']}) | semilla {s} | pliegue {k} "
                      f"(senante {p['senante_test']}) | {datetime.now().isoformat(timespec='seconds')}\n{'=' * 72}",
                      flush=True)
                entrenar_corrida(b, s, k, p, pos, X_seq, y, canon, d, args)
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] fin", flush=True)


# ---------------------------------------------------------------------------
# Inferencia (lote de 1, como translator.py)
# ---------------------------------------------------------------------------

def otros_entrenamientos():
    import re
    import subprocess
    r = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True)
    patron = re.compile(r"src/(experimento_[a-z_]+|train_loso|busqueda_hiperparametros)\.py "
                        r"(entrenar|fase1|fase2|ajustar)|train_loso\.py")
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
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    tf = canon[0]
    al.capa_conv_grafo()
    otros = otros_entrenamientos()
    if otros:
        raise SystemExit(f"hay entrenamientos activos; la latencia se mide sin ellos: {otros}")
    rng = np.random.default_rng(0)
    filas = []
    for b in ("R0", "R1", "T", "G"):
        base = R0_DIR if b == "R0" else out / BRAZOS[b]["carpeta"]
        rutas = sorted(base.glob("semilla_2026/pliegue_*/modelo.keras"))[:args.modelos]
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
            fila = {"brazo": b, "modelo": tl._rel(ruta), "n_parametros": int(model.count_params()),
                    "tamano_archivo_mb": round(ruta.stat().st_size / 2 ** 20, 2),
                    "ms_predict_lote1_mediana": float(np.median(t_pred)),
                    "ms_predict_lote1_p90": float(np.percentile(t_pred, 90)),
                    "ms_llamada_directa_mediana": float(np.median(t_call)),
                    "ms_llamada_directa_p90": float(np.percentile(t_call, 90))}
            filas.append(fila)
            print(f"  {b} {tl._rel(ruta)}: {fila['n_parametros']:,} param | predict(1) mediana "
                  f"{fila['ms_predict_lote1_mediana']:.2f} ms (p90 {fila['ms_predict_lote1_p90']:.2f}) | "
                  f"model(x) {fila['ms_llamada_directa_mediana']:.2f} ms", flush=True)
            del model
            tf.keras.backend.clear_session()
    with open(out / "inferencia_redes.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": datetime.now().isoformat(timespec="seconds"),
                   "metodo": f"model.predict(verbose=0) con lote de 1, como translator.py, y llamada directa "
                             f"model(x, training=False); {args.calentamiento} llamadas de calentamiento y "
                             f"{args.repeticiones} medidas por modelo; mediana y p90. Solo la red (sin MediaPipe "
                             f"ni remuestreo). CPU. Modelos de la semilla 2026.",
                   "procesos_de_entrenamiento_activos": otros, "entorno": tl.info_entorno(), "modelos": filas},
                  f, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Resumen y criterio
# ---------------------------------------------------------------------------

def _md(v):
    v = [float(x) for x in v]
    return {"media": float(np.mean(v)), "de": float(np.std(v, ddof=1)) if len(v) > 1 else None, "valores": v}


def leer_corridas(out):
    """Registros homogéneos de todos los brazos: val_top1 (la que decide), prueba y predicciones."""
    filas = []
    for s in SEMILLAS:
        for k in range(5):
            m, pred, d = leer_r0(s, k)
            filas.append({"brazo": "R0", "semilla": s, "pliegue": k, "senante_test": m["senante_test"],
                          "val_top1": m["mejor_val_accuracy"], "test_top1": m["top1"], "test_top3": m["top3"],
                          "test_f1_macro": m["f1_macro"], "n_parametros": m["n_parametros"],
                          "epocas_ejecutadas": m["epocas_ejecutadas"], "tiempo_total_s": m["tiempo_total_s"],
                          "tiempo_entrenamiento_s": m["tiempo_entrenamiento_s"],
                          "predicciones": {x["muestra"]: (x["real"], x["predicha"]) for x in pred},
                          "fuente": tl._rel(d)})
    for b in ("R1", "M", "T", "G"):
        for f in sorted((Path(out) / BRAZOS[b]["carpeta"]).glob("semilla_*/pliegue_*/resultado.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            pred = json.loads((f.parent / "predicciones.json").read_text(encoding="utf-8"))
            filas.append({"brazo": b, "semilla": r["semilla"], "pliegue": r["pliegue"],
                          "senante_test": r["senante_test"], "val_top1": r["val_top1"],
                          "test_top1": r["test_top1"], "test_top3": r["test_top3"],
                          "test_f1_macro": r["test_f1_macro"], "n_parametros": r["n_parametros"],
                          "epocas_ejecutadas": r.get("epocas_ejecutadas"), "tiempo_total_s": r["tiempo_total_s"],
                          "tiempo_entrenamiento_s": r.get("tiempo_entrenamiento_s"),
                          "predicciones": {x["muestra"]: (x["real"], x["predicha"]) for x in pred},
                          "fuente": tl._rel(f.parent), "extra": {kk: r[kk] for kk in r if kk in (
                              "alpha", "alphas", "cv_interna", "val_top1_checkpoint_reevaluado", "s_por_epoca",
                              "mejor_epoca", "n_rasgos")}})
    return filas


def pruebas_pareadas(dif):
    """Las exactitudes de Keras son float32: dos medias iguales pueden diferir en ~1e-8. Se redondea a
    6 decimales (las exactitudes medias son múltiplos de 1/300 o 1/600) para que esos casos cuenten como
    empate y no como pliegue peor."""
    return bh.pruebas_pareadas(np.round(np.asarray(dif, dtype=float), 6))


def mcnemar(pa, pb):
    """McNemar exacto (binomial bilateral sobre los discordantes). pa, pb: {muestra: (real, predicha)}."""
    from scipy.stats import binomtest, chi2
    claves = sorted(pa)
    assert set(claves) == set(pb)
    ca = np.array([pa[c][0] == pa[c][1] for c in claves])
    cb = np.array([pb[c][0] == pb[c][1] for c in claves])
    b = int(np.sum(ca & ~cb))     # acierta el brazo, falla R0
    c = int(np.sum(~ca & cb))     # falla el brazo, acierta R0
    p_exacto = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    chi = ((abs(b - c) - 1) ** 2 / (b + c)) if b + c else 0.0
    return {"n": len(claves), "aciertos_brazo": int(ca.sum()), "aciertos_R0": int(cb.sum()),
            "solo_brazo_acierta": b, "solo_R0_acierta": c, "ambos_aciertan": int(np.sum(ca & cb)),
            "ninguno": int(np.sum(~ca & ~cb)), "p_exacto_bilateral": p_exacto,
            "chi2_con_correccion": float(chi), "p_chi2": float(chi2.sf(chi, 1)) if b + c else 1.0}


def cmd_resumen(args):
    out = Path(args.out).resolve()
    filas = leer_corridas(out)
    brazos = [b for b in BRAZOS if any(f["brazo"] == b for f in filas)]
    folds = list(range(5))

    def celda(b, s, k):
        r = [f for f in filas if f["brazo"] == b and f["semilla"] == s and f["pliegue"] == k]
        return r[0] if r else None

    sem = {b: sorted({f["semilla"] for f in filas if f["brazo"] == b
                      and all(celda(b, f["semilla"], k) for k in folds)}) for b in brazos}
    incompletos = [b for b in brazos if not sem[b]]
    if incompletos:
        print(f"AVISO: brazos sin ninguna semilla completa (se omiten): {incompletos}")
    brazos = [b for b in brazos if sem[b]]
    METR = ["val_top1", "test_top1", "test_top3", "test_f1_macro", "epocas_ejecutadas", "tiempo_total_s",
            "tiempo_entrenamiento_s"]
    por_brazo = {}
    for b in brazos:
        r = {"semillas": sem[b], "n_parametros": celda(b, sem[b][0], 0)["n_parametros"] if sem[b] else None}
        for m in METR:
            pps = {s: [celda(b, s, k)[m] for k in folds] for s in sem[b]}
            if any(v is None for vs in pps.values() for v in vs):
                continue
            por_pliegue = [float(np.mean([pps[s][k] for s in sem[b]])) for k in folds]
            por_sem = {s: float(np.mean(pps[s])) for s in sem[b]}
            r[m] = {"por_semilla_por_pliegue": {str(s): v for s, v in pps.items()},
                    "por_pliegue_media_semillas": por_pliegue,
                    "por_semilla_media_pliegues": {str(s): v for s, v in por_sem.items()},
                    "entre_pliegues": _md(por_pliegue),
                    "de_entre_semillas": float(np.std(list(por_sem.values()), ddof=1)) if len(sem[b]) > 1 else None,
                    "media_global": float(np.mean(list(por_sem.values())))}
        por_brazo[b] = r

    # Criterio prerregistrado (docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md §2)
    var = {b: por_brazo[b]["val_top1"]["de_entre_semillas"] ** 2 for b in brazos
           if por_brazo[b].get("val_top1") and por_brazo[b]["val_top1"]["de_entre_semillas"] is not None}
    sigma = float(np.sqrt(np.mean(list(var.values()))))
    margen = max(sigma, PISO_MARGEN)
    v0 = np.array(por_brazo["R0"]["val_top1"]["por_pliegue_media_semillas"])
    detalle, cumplen = {}, []
    for b in [x for x in brazos if x != "R0"]:
        vb = np.array(por_brazo[b]["val_top1"]["por_pliegue_media_semillas"])
        dif = vb - v0
        pp = pruebas_pareadas(dif)
        delta = float(np.mean(dif))
        c1, c2 = bool(delta > margen + 1e-12), bool(pp["mejora"] >= 4)
        detalle[b] = {"delta_val_por_pliegue": dif.tolist(), "delta_val_media": delta, "pliegues_mejor": pp["mejora"],
                      "pliegues_peor": pp["empeora"], "pliegues_igual": pp["empate"],
                      "condicion_1_delta_mayor_que_margen": c1, "condicion_2_mejor_en_4_de_5": c2,
                      "supera_al_canonico": bool(c1 and c2),
                      "peor_segun_regla_simetrica": bool(-delta > margen + 1e-12 and pp["empeora"] >= 4),
                      "pruebas_validacion": pp}
        for m in ("test_top1", "test_top3", "test_f1_macro"):
            d_m = (np.array(por_brazo[b][m]["por_pliegue_media_semillas"])
                   - np.array(por_brazo["R0"][m]["por_pliegue_media_semillas"]))
            detalle[b]["pruebas_" + m + "_descriptivas"] = pruebas_pareadas(d_m)
        if c1 and c2:
            cumplen.append(b)
    if cumplen:
        mejor = max(por_brazo[b]["val_top1"]["media_global"] for b in cumplen)
        empatados = [b for b in cumplen if mejor - por_brazo[b]["val_top1"]["media_global"] < margen]
        ganador = min(empatados, key=lambda b: por_brazo[b]["n_parametros"])
    else:
        empatados, ganador = [], "R0"
    criterio = {"sigma_semilla_agrupada": sigma, "varianza_entre_semillas_por_brazo": var, "margen": margen,
                "detalle": detalle, "brazos_que_superan": cumplen, "empatados_dentro_del_margen": empatados,
                "ganador": ganador}

    # McNemar sobre las 997 predicciones agregadas, frente a R0 (principal: semilla 2026).
    mc = {}
    for b in [x for x in brazos if x != "R0"]:
        mc[b] = {}
        for s in sem[b]:
            pa, p0 = {}, {}
            for k in folds:
                pa.update(celda(b, s, k)["predicciones"])
                p0.update(celda("R0", s, k)["predicciones"])
            mc[b][str(s)] = mcnemar(pa, p0)
    # Persona 0000 (pliegue 0)
    p0000 = {b: {m: {"por_semilla": por_brazo[b][m]["por_semilla_por_pliegue"],
                     "media": por_brazo[b][m]["por_pliegue_media_semillas"][0]}
                 for m in ("val_top1", "test_top1", "test_top3", "test_f1_macro")} for b in brazos}
    for b in brazos:
        p0000[b] = {m: {"media": v["media"], "por_semilla": {s: vals[0] for s, vals in v["por_semilla"].items()}}
                    for m, v in p0000[b].items()}

    # Reproducción: R1 semilla 2026 frente a la fase 1 de la búsqueda (mismo pipeline).
    repro = []
    for k in folds:
        c = celda("R1", 2026, k)
        f1 = C07_FASE1 / "semilla_2026" / f"pliegue_{k}" / "resultado.json"
        if c and f1.exists():
            r = json.loads(f1.read_text())
            repro.append({"pliegue": k, "fase1_val": r["val_top1"], "nuevo_val": c["val_top1"],
                          "fase1_test": r["test_top1"], "nuevo_test": c["test_top1"],
                          "fase1_epocas": r["epocas_ejecutadas"], "nuevo_epocas": c["epocas_ejecutadas"]})
    reproduccion = {"filas": repro, "identica": bool(repro) and all(
        abs(r["fase1_val"] - r["nuevo_val"]) < 1e-9 and abs(r["fase1_test"] - r["nuevo_test"]) < 1e-9
        and r["fase1_epocas"] == r["nuevo_epocas"] for r in repro)}

    extra = {}
    for nombre in ("verificacion.json", "inferencia_redes.json", "inferencia_minirocket.json", "humo.json"):
        if (out / nombre).exists():
            extra[nombre] = json.loads((out / nombre).read_text(encoding="utf-8"))

    sin_pred = [{k: v for k, v in f.items() if k != "predicciones"} for f in filas]
    resultado = {"experimento": "arquitecturas de la literatura (docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md)",
                 "fecha_resumen": datetime.now().isoformat(timespec="seconds"),
                 "brazos": {b: BRAZOS[b] for b in brazos}, "semillas_por_brazo": sem,
                 "config_T": al.CONFIG_T, "config_G": al.CONFIG_G, "entrenamiento_T_G": al.ENTRENAMIENTO_COMUN,
                 "c07": {k: bh.config_por_id("c07")[k] for k in bh.ESPACIO},
                 "criterio": criterio, "mcnemar_frente_a_R0": mc, "persona_0000": p0000,
                 "por_brazo": por_brazo, "reproduccion_R1_vs_fase1": reproduccion,
                 "verificacion": extra.get("verificacion.json"), "humo": extra.get("humo.json"),
                 "inferencia_redes": extra.get("inferencia_redes.json"),
                 "inferencia_minirocket": extra.get("inferencia_minirocket.json"), "corridas": sin_pred}
    (out / "resultados_arquitecturas.json").write_text(json.dumps(resultado, indent=1, ensure_ascii=False),
                                                       encoding="utf-8")
    if args.docs:
        JSON_DOCS.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"escrito {tl._rel(JSON_DOCS)}")

    # Impresión
    print(f"brazos {brazos} | semillas {sem}")
    print(f"sigma semilla agrupada {sigma:.4f} | margen {margen:.4f}")
    for b in brazos:
        r = por_brazo[b]
        print(f"\n{b} ({BRAZOS[b]['descripcion']}) | {r['n_parametros']:,} param")
        for m in ("val_top1", "test_top1", "test_top3", "test_f1_macro"):
            print(f"   {m:14s} {r[m]['media_global']:.4f} (DE pliegues {r[m]['entre_pliegues']['de']:.4f}, DE semillas "
                  f"{(r[m]['de_entre_semillas'] or 0):.4f}) por pliegue {[round(x, 3) for x in r[m]['por_pliegue_media_semillas']]}")
        print(f"   val por semilla {r['val_top1']['por_semilla_media_pliegues']}")
        if r.get("tiempo_total_s"):
            print(f"   tiempo por pliegue {r['tiempo_total_s']['media_global']:.0f} s | epocas "
                  f"{r['epocas_ejecutadas']['media_global'] if r.get('epocas_ejecutadas') else '-'}")
    for b, d in detalle.items():
        print(f"\n{b} - R0: dval {[round(x, 4) for x in d['delta_val_por_pliegue']]} media {d['delta_val_media']:+.4f} | "
              f"mejor {d['pliegues_mejor']}/5 | signos p {d['pruebas_validacion']['signos_p_bilateral']:.4f} | wilcoxon p "
              f"{d['pruebas_validacion']['wilcoxon_p_bilateral']:.4f} | supera {d['supera_al_canonico']}")
        for m in ("test_top1", "test_f1_macro"):
            pr = d["pruebas_" + m + "_descriptivas"]
            print(f"   {m}: {[round(x, 3) for x in pr['diferencias']]} | mejor {pr['mejora']}/5 | signos p "
                  f"{pr['signos_p_bilateral']:.4f} | wilcoxon p {pr['wilcoxon_p_bilateral']:.4f}")
        for s, m in mc[b].items():
            print(f"   McNemar s{s}: brazo {m['aciertos_brazo']} vs R0 {m['aciertos_R0']} | solo brazo "
                  f"{m['solo_brazo_acierta']} solo R0 {m['solo_R0_acierta']} | p exacto {m['p_exacto_bilateral']:.4g}")
    print(f"\nGANADOR: {ganador} (cumplen: {cumplen})")
    print("persona 0000:", {b: (round(p0000[b]['val_top1']['media'], 3), round(p0000[b]['test_top1']['media'], 3))
                            for b in brazos})
    print("reproduccion R1 vs fase1:", reproduccion)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verificar")
    ex = sub.add_parser("exportar")
    ex.add_argument("--dest", required=True)
    ex.add_argument("--semillas", default="2026,2027,2028")
    ex.add_argument("--folds", default="5")
    ex.add_argument("--aug", type=int, default=10)
    e = sub.add_parser("entrenar")
    e.add_argument("--brazos", default="R1,T,G")
    e.add_argument("--semillas", default="2026,2027,2028")
    e.add_argument("--folds", default="5")
    e.add_argument("--epochs", type=int, default=200)
    e.add_argument("--aug", type=int, default=10)
    e.add_argument("--patience", type=int, default=25)
    e.add_argument("--processed", default=str(PROCESSED_CANONICO))
    i = sub.add_parser("inferencia")
    i.add_argument("--calentamiento", type=int, default=10)
    i.add_argument("--repeticiones", type=int, default=100)
    i.add_argument("--modelos", type=int, default=3)
    r = sub.add_parser("resumen")
    r.add_argument("--docs", action="store_true")
    for p in (v, e, i, r):
        p.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    {"verificar": cmd_verificar, "exportar": cmd_exportar, "entrenar": cmd_entrenar,
     "inferencia": cmd_inferencia, "resumen": cmd_resumen}[args.cmd](args)


if __name__ == "__main__":
    main()
