"""
Experimento de ablación: ¿aportan las velocidades al modelo canónico? (LOSO de 5 pliegues)

Brazos (lo ÚNICO que cambia es qué se entrega a la red después del remuestreo y la aumentación):
  A  con velocidades   30 x 252  posiciones + add_velocity (configuración canónica, N-30)
  B  sin velocidades   30 x 126  solo posiciones
  C  solo velocidades  30 x 126  solo la mitad de velocidades de add_velocity (informativo)

Reutiliza SIN modificar las piezas canónicas:
  - train_loso.cargar_datos, verificar_particion, importar_pipeline_canonico, remuestrear_todas,
    entrenar_y_evaluar (la función que produjo la referencia de N-30: 0,580 con semilla 2026);
  - train_final_100.resample_sequence, add_velocity y build_model (a través de train_loso).
entrenar_y_evaluar recibe la tupla `canon` = (tf, build_model, add_velocity, resample_sequence,
transformaciones) y llama a canon[2] para construir la entrada de entrenamiento, validación y
prueba DESPUÉS de aumentar; build_model toma el ancho de esa entrada. Por eso basta con sustituir
canon[2] por la función del brazo: la aumentación (espejo incluido) se aplica siempre sobre las
126 posiciones, antes de calcular velocidades, y no depende del brazo.

La procedencia que entrenar_y_evaluar escribe junto al modelo lleva "velocidades": true fijo; se
reescribe después con la entrada real del brazo.

Criterio (prerregistrado en docs/EXPERIMENTO_VELOCIDADES.md §2): exactitud top-1 de VALIDACIÓN
(mejor_val_accuracy, la misma que decidió D-07 y D-08). La prueba LOSO se reporta como descripción.

Subcomandos:
  verificar   comprobaciones sin entrenar: equivalencia de los brazos con add_velocity, aumentación
              independiente del brazo, y conteo de «saltos falsos» de la regla mascara.
  entrenar    corre las combinaciones (semilla, pliegue, brazo) que falten; reanudable.
  inferencia  parámetros y tiempo de inferencia por muestra de cada modelo (sin entrenamientos en
              paralelo).
  resumen     agrega, aplica el criterio y escribe el JSON de datos crudos.

Uso:
  python src/experimento_velocidades.py verificar
  python src/experimento_velocidades.py entrenar --brazos A,B --semillas 2026 --folds 0, \\
         --epochs 2 --aug 2 --out <dir_temporal>                          # prueba de humo
  python src/experimento_velocidades.py entrenar --brazos A,B,C --semillas 2026,2027,2028
  python src/experimento_velocidades.py inferencia
  python src/experimento_velocidades.py resumen
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
DEFAULT_OUT = BASE_DIR / "models" / "experimento_velocidades_2026-09-30"
JSON_DOCS = BASE_DIR / "docs" / "EXPERIMENTO_VELOCIDADES.json"
REFERENCIA_N30 = BASE_DIR / "models" / "loso_2026-09-15_holistic_mascara_manos" / "resultados_loso.json"

BRAZOS = {
    "A": {"carpeta": "A_con_velocidades", "descripcion": "posiciones + velocidades (canonico)"},
    "B": {"carpeta": "B_sin_velocidades", "descripcion": "solo posiciones"},
    "C": {"carpeta": "C_solo_velocidades", "descripcion": "solo velocidades"},
}


def funcion_entrada(brazo, add_velocity):
    """Entrada de la red para cada brazo, a partir de las posiciones ya remuestreadas y aumentadas.
    A y C se derivan de add_velocity canónica (C es exactamente su mitad de velocidades)."""
    if brazo == "A":
        return add_velocity
    if brazo == "B":
        return lambda X: X
    if brazo == "C":
        return lambda X: add_velocity(X)[:, :, X.shape[2]:]
    raise ValueError(f"brazo desconocido: {brazo}")


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
        "val_f1_macro": float(f1_score(y_val, pred, labels=list(range(int(probs.shape[1]))),
                                       average="macro", zero_division=0)),
    }


def reescribir_procedencia(ruta_modelo, brazo, n_pos):
    """La procedencia de representacion.escribir_procedencia declara velocidades=true siempre."""
    from src.representacion import ruta_procedencia
    ruta = ruta_procedencia(ruta_modelo)
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos["velocidades"] = brazo in ("A", "C")
    datos["posiciones"] = brazo in ("A", "B")
    datos["n_rasgos_entrada"] = 2 * n_pos if brazo == "A" else n_pos
    datos["brazo_experimento_velocidades"] = brazo
    datos["nota"] = ("modelo del experimento de ablacion de velocidades (docs/EXPERIMENTO_VELOCIDADES.md); "
                     "NO es el modelo canonico ni debe cargarse en el prototipo")
    datos["script"] = "src/experimento_velocidades.py (via train_loso.entrenar_y_evaluar)"
    ruta.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Verificación sin entrenar
# ---------------------------------------------------------------------------

def saltos_falsos(X_seq):
    """Cuenta, en las secuencias remuestreadas, los pasos en que un bloque de mano pasa de presente
    a ausente (o al revés) según la regla mascara: el bloque entero queda en ceros exactos. Tras el
    remuestreo lineal, el paso de interpolación entre un fotograma real y uno enmascarado puede dar
    un bloque «a medias» (no cero, pero encogido hacia el origen); se cuenta como transición todo
    cambio entre «bloque en ceros exactos» y «bloque no nulo»."""
    n_pos = X_seq.shape[2]
    mitad = n_pos // 2
    res = {}
    for nombre, sl in (("mano_izq", slice(0, mitad)), ("mano_der", slice(mitad, n_pos))):
        bloque = X_seq[:, :, sl]
        cero = np.all(bloque == 0, axis=2)                       # (N, T)
        trans = cero[:, 1:] != cero[:, :-1]                      # (N, T-1)
        vel = np.linalg.norm(bloque[:, 1:] - bloque[:, :-1], axis=2)
        ambos_presentes = (~cero[:, 1:]) & (~cero[:, :-1])
        res[nombre] = {
            "secuencias_con_alguna_transicion": int(np.sum(np.any(trans, axis=1))),
            "secuencias_con_bloque_ausente_en_algun_paso": int(np.sum(np.any(cero, axis=1))),
            "transiciones_totales": int(np.sum(trans)),
            "pasos_totales_con_velocidad": int(trans.size),
            "norma_velocidad_mediana_en_transicion": float(np.median(vel[trans])) if np.any(trans) else None,
            "norma_velocidad_mediana_ambos_presentes": float(np.median(vel[ambos_presentes])),
            "norma_velocidad_p95_ambos_presentes": float(np.percentile(vel[ambos_presentes], 95)),
        }
    return res


def cmd_verificar(args):
    processed = Path(args.processed).resolve()
    X_padded, y, lengths, nombres, loso, pos = cargar_todo(processed, args.representacion)
    canon = tl.importar_pipeline_canonico(args.representacion)
    tf, _, add_velocity, resample_sequence, transformaciones = canon
    X_seq = tl.remuestrear_todas(X_padded, lengths, resample_sequence, tl.SEQ_LENGTH)
    n_pos = X_seq.shape[2]
    print(f"processed {tl._rel(processed)} | representacion {args.representacion} | X_seq {X_seq.shape}")

    # 1. Entrada de cada brazo
    Xa = funcion_entrada("A", add_velocity)(X_seq)
    Xb = funcion_entrada("B", add_velocity)(X_seq)
    Xc = funcion_entrada("C", add_velocity)(X_seq)
    assert Xa.shape[2] == 2 * n_pos and Xb.shape[2] == n_pos and Xc.shape[2] == n_pos
    assert np.array_equal(Xa[:, :, :n_pos], Xb), "A[:, :, :126] debe ser exactamente B"
    assert np.array_equal(Xa[:, :, n_pos:], Xc), "A[:, :, 126:] debe ser exactamente C"
    assert np.all(Xc[:, 0, :] == 0), "la velocidad del paso 1 debe ser 0"
    print(f"[ok] A = [B | C]; anchos A {Xa.shape[2]}, B {Xb.shape[2]}, C {Xc.shape[2]}; C[:,0]=0")

    # 2. La aumentación no depende del brazo: con la misma semilla produce las mismas posiciones,
    #    y el espejo opera sobre 126 columnas (espejar valida el ancho).
    idx_tr = np.array([pos[n] for n in loso["pliegues"][0]["conjuntos"]["train"]], dtype=np.int64)
    sub = X_seq[idx_tr[:60]]
    salidas = []
    for _ in range(2):
        tf.keras.utils.set_random_seed(2026)
        Xt, yt = tl.aumentar(sub, y[idx_tr[:60]], 10, transformaciones)
        salidas.append((Xt, yt))
    assert np.array_equal(salidas[0][0], salidas[1][0]) and np.array_equal(salidas[0][1], salidas[1][1])
    assert salidas[0][0].shape[2] == n_pos
    esp = transformaciones[0][1](sub)
    assert np.array_equal(transformaciones[0][1](esp), sub), "espejar dos veces debe ser la identidad"
    print(f"[ok] aumentacion x10 determinista con la semilla, sobre {n_pos} columnas "
          f"({len(salidas[0][1])} secuencias de 60); el espejo es una involucion")

    # 3. Saltos falsos
    sf = saltos_falsos(X_seq)
    print("[info] saltos falsos de la regla mascara (secuencias remuestreadas a 30 pasos, 1000 muestras):")
    for k, v in sf.items():
        print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")
    if args.guardar:
        out = Path(args.out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "verificacion.json", "w", encoding="utf-8") as f:
            json.dump({"fecha": datetime.now().isoformat(timespec="seconds"),
                       "processed": tl._rel(processed), "representacion": args.representacion,
                       "comprobaciones": ["A = [B | C] exacto", "C[:,0] = 0",
                                          "aumentacion determinista e independiente del brazo",
                                          "espejo involutivo sobre 126 columnas"],
                       "saltos_falsos": sf}, f, indent=1, ensure_ascii=False)
        print(f"  escrito {tl._rel(out / 'verificacion.json')}")


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def cmd_entrenar(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    processed = Path(args.processed).resolve()
    X_padded, y, lengths, nombres, loso, pos = cargar_todo(processed, args.representacion)
    canon = tl.importar_pipeline_canonico(args.representacion)
    tf, build_model, add_velocity, resample_sequence, transformaciones = canon
    nombres_senas = tl.cargar_nombres_senas()
    folds = tl.parsear_folds(args.folds, len(loso["pliegues"]))
    semillas = _lista_int(args.semillas)
    brazos = _lista_brazos(args.brazos)
    X_seq = tl.remuestrear_todas(X_padded, lengths, resample_sequence, tl.SEQ_LENGTH)
    n_pos = X_seq.shape[2]
    print(f"[{datetime.now().isoformat(timespec='seconds')}] experimento_velocidades entrenar | "
          f"brazos {brazos} | semillas {semillas} | pliegues {folds} | epochs {args.epochs} | "
          f"aug x{args.aug} | paciencia {args.patience} | processed {tl._rel(processed)} | "
          f"representacion {args.representacion} | X_seq {X_seq.shape} | salida {tl._rel(out)}",
          flush=True)

    # Orden: semilla -> pliegue -> brazo. Así cada par (A, B) de un pliegue y semilla queda
    # completo antes de pasar al siguiente, y C puede lanzarse aparte sin romper los pares.
    for semilla in semillas:
        for k in folds:
            p = loso["pliegues"][k]
            idx_val = np.array([pos[n] for n in p["conjuntos"]["val"]], dtype=np.int64)
            for brazo in brazos:
                d = dir_corrida(out, brazo, semilla, k)
                if (d / "metricas_velocidades.json").exists():
                    print(f"  ya existe {tl._rel(d)}; se salta", flush=True)
                    continue
                entrada = funcion_entrada(brazo, add_velocity)
                canon_brazo = (tf, build_model, entrada, resample_sequence, transformaciones)
                print(f"\n{'=' * 72}\n  brazo {brazo} ({BRAZOS[brazo]['descripcion']}) | semilla "
                      f"{semilla} | pliegue {k} (senante {p['senante_test']}) | "
                      f"{datetime.now().isoformat(timespec='seconds')}\n{'=' * 72}", flush=True)
                ns = argparse.Namespace(
                    seed=semilla, aug=args.aug, epochs=args.epochs, batch_size=args.batch_size,
                    patience=args.patience, dropout=args.dropout, processed=str(processed),
                    representacion=args.representacion)
                m = tl.entrenar_y_evaluar(f"{brazo}_s{semilla}_pliegue_{k}", p["conjuntos"], pos,
                                          X_seq, y, ns, d, canon_brazo, nombres_senas)
                reescribir_procedencia(d / "modelo.keras", brazo, n_pos)
                model = tf.keras.models.load_model(str(d / "modelo.keras"))
                mv = metricas_validacion(model, entrada(X_seq[idx_val]), y[idx_val])
                n_param = int(model.count_params())
                n_in = int(model.input_shape[-1])
                del model
                tf.keras.backend.clear_session()
                m.update(mv)
                m.update({"brazo": brazo, "semilla": semilla, "pliegue": k,
                          "senante_test": p["senante_test"], "n_parametros": n_param,
                          "n_rasgos_entrada": n_in,
                          "config": {"epochs_max": args.epochs, "aug": args.aug,
                                     "batch_size": args.batch_size, "patience": args.patience,
                                     "dropout": args.dropout, "seq_length": tl.SEQ_LENGTH,
                                     "processed": tl._rel(processed),
                                     "representacion": args.representacion},
                          "fecha_fin": datetime.now().isoformat(timespec="seconds")})
                with open(d / "metricas_velocidades.json", "w", encoding="utf-8") as f:
                    json.dump(m, f, indent=1)
                print(f"  -> {brazo} s={semilla} k={k}: val {m['mejor_val_accuracy']:.3f} "
                      f"(checkpoint {mv['val_top1']:.3f}) | test top-1 {m['top1']:.3f} | "
                      f"entrada {n_in} | params {n_param} | epocas {m['epocas_ejecutadas']} | "
                      f"{m['tiempo_total_s']:.0f} s", flush=True)
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] fin", flush=True)


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

def cmd_inferencia(args):
    """Tiempo por muestra como lo hace el prototipo (translator.py: model.predict de un lote de 1),
    y por muestra en lote de 200. Mediana de --repeticiones llamadas tras --calentamiento. Incluye
    aparte el costo de construir la entrada del brazo (add_velocity o nada) para una muestra."""
    out = Path(args.out).resolve()
    canon = tl.importar_pipeline_canonico(REPRESENTACION_CANONICA)
    tf, _, add_velocity, _, _ = canon
    rng = np.random.default_rng(0)
    filas = []
    for brazo in BRAZOS:
        for ruta in sorted((out / BRAZOS[brazo]["carpeta"]).glob("semilla_*/pliegue_*/modelo.keras")):
            model = tf.keras.models.load_model(str(ruta))
            n_feat = int(model.input_shape[-1])
            x1 = rng.standard_normal((1, tl.SEQ_LENGTH, n_feat)).astype(np.float32)
            xb = rng.standard_normal((200, tl.SEQ_LENGTH, n_feat)).astype(np.float32)
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
            fila = {"brazo": brazo, "modelo": tl._rel(ruta), "n_rasgos_entrada": n_feat,
                    "n_parametros": int(model.count_params()),
                    "ms_lote1_mediana": float(np.median(t)), "ms_lote1_p90": float(np.percentile(t, 90)),
                    "ms_por_muestra_lote200_mediana": float(np.median(tb))}
            filas.append(fila)
            print(f"  {brazo} {tl._rel(ruta)}: entrada {n_feat} | params {fila['n_parametros']} | "
                  f"lote 1 mediana {fila['ms_lote1_mediana']:.2f} ms (p90 {fila['ms_lote1_p90']:.2f}) | "
                  f"lote 200 {fila['ms_por_muestra_lote200_mediana']:.3f} ms/muestra", flush=True)
            del model
            tf.keras.backend.clear_session()
    # Costo de construir la entrada de una muestra (posiciones ya remuestreadas, 30 x 126).
    xp = rng.standard_normal((1, tl.SEQ_LENGTH, 126)).astype(np.float32)
    costo = {}
    for brazo in BRAZOS:
        fn = funcion_entrada(brazo, add_velocity)
        t = []
        for _ in range(2000):
            t0 = time.perf_counter()
            fn(xp)
            t.append((time.perf_counter() - t0) * 1000)
        costo[brazo] = float(np.median(t))
    print(f"  construir la entrada (ms, mediana de 2000): {costo}")
    with open(out / "inferencia.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": datetime.now().isoformat(timespec="seconds"),
                   "metodo": f"model.predict(verbose=0) de un lote de 1 como translator.py; "
                             f"{args.calentamiento} llamadas de calentamiento y {args.repeticiones} "
                             f"medidas por modelo; ademas lote de 200 (5 medidas). CPU, sin otros "
                             f"entrenamientos corriendo. Solo la red: sin MediaPipe ni remuestreo.",
                   "entorno": tl.info_entorno(), "modelos": filas,
                   "ms_construir_entrada_mediana": costo}, f, indent=1)


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
    dif = np.asarray(dif, dtype=float)
    pos_, neg_ = int(np.sum(dif > 1e-12)), int(np.sum(dif < -1e-12))
    res = {"diferencias": dif.tolist(), "a_favor_de_A": pos_, "a_favor_de_B": neg_,
           "empates": int(len(dif) - pos_ - neg_)}
    if pos_ + neg_ > 0:
        res["signos_p_bilateral"] = float(binomtest(pos_, pos_ + neg_, 0.5).pvalue)
    else:
        res["signos_p_bilateral"] = 1.0
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


METRICAS = ["mejor_val_accuracy", "val_top1", "val_top3", "val_f1_macro", "top1", "top3",
            "f1_macro", "precision_macro", "recall_macro", "epocas_ejecutadas", "mejor_epoca",
            "tiempo_total_s", "n_parametros", "n_rasgos_entrada"]


def cmd_resumen(args):
    out = Path(args.out).resolve()
    filas = []
    for brazo in BRAZOS:
        for p in sorted((out / BRAZOS[brazo]["carpeta"]).glob("semilla_*/pliegue_*/metricas_velocidades.json")):
            filas.append(json.loads(p.read_text(encoding="utf-8")))
    if not filas:
        raise SystemExit("no hay corridas")
    brazos = [b for b in BRAZOS if any(f["brazo"] == b for f in filas)]
    folds = sorted({f["pliegue"] for f in filas})
    semillas_por_brazo = {b: sorted({f["semilla"] for f in filas if f["brazo"] == b}) for b in brazos}

    def celda(b, s, k):
        r = [f for f in filas if f["brazo"] == b and f["semilla"] == s and f["pliegue"] == k]
        return r[0] if r else None

    # Por brazo y pliegue: media sobre semillas
    por_brazo = {}
    for b in brazos:
        pf = {}
        for k in folds:
            rs = [f for f in filas if f["brazo"] == b and f["pliegue"] == k]
            pf[k] = {m: _media_de([r[m] for r in rs]) for m in METRICAS}
            pf[k]["semillas"] = [r["semilla"] for r in rs]
        # Agregado LOSO: media y DE entre pliegues de las medias por pliegue
        ag = {m: _media_de([pf[k][m]["media"] for k in folds]) for m in METRICAS}
        # Por semilla: LOSO de 5 pliegues de cada semilla
        por_semilla = {}
        for s in semillas_por_brazo[b]:
            rs = [celda(b, s, k) for k in folds]
            if all(rs):
                por_semilla[s] = {m: _media_de([r[m] for r in rs]) for m in
                                  ["mejor_val_accuracy", "top1", "top3", "f1_macro", "epocas_ejecutadas"]}
        por_brazo[b] = {"por_pliegue": {str(k): v for k, v in pf.items()}, "agregado": ag,
                        "por_semilla": {str(s): v for s, v in por_semilla.items()}}

    # Variación entre semillas (prerregistrada): DE muestral entre semillas del val top-1 dentro de
    # cada celda (brazo, pliegue), combinada como raíz de la media de las varianzas, sobre A y B.
    varianzas = []
    for b in [x for x in ("A", "B") if x in brazos]:
        for k in folds:
            v = [f["mejor_val_accuracy"] for f in filas if f["brazo"] == b and f["pliegue"] == k]
            if len(v) > 1:
                varianzas.append(float(np.var(v, ddof=1)))
    ruido = float(np.sqrt(np.mean(varianzas))) if varianzas else None

    comparaciones = {}
    pares = [("A", "B"), ("A", "C"), ("B", "C")]
    for b1, b2 in pares:
        if b1 not in brazos or b2 not in brazos:
            continue
        semillas_comunes = sorted(set(semillas_por_brazo[b1]) & set(semillas_por_brazo[b2]))
        comp = {"semillas_comunes": semillas_comunes}
        for m in ["mejor_val_accuracy", "top1", "top3", "f1_macro"]:
            dif_k = []
            for k in folds:
                v1 = [celda(b1, s, k)[m] for s in semillas_comunes if celda(b1, s, k) and celda(b2, s, k)]
                v2 = [celda(b2, s, k)[m] for s in semillas_comunes if celda(b1, s, k) and celda(b2, s, k)]
                dif_k.append(float(np.mean(v1) - np.mean(v2)))
            dif_ks = [celda(b1, s, k)[m] - celda(b2, s, k)[m] for s in semillas_comunes for k in folds
                      if celda(b1, s, k) and celda(b2, s, k)]
            comp[m] = {"diferencia_por_pliegue": dif_k,
                       "diferencia_media": float(np.mean(dif_k)),
                       "de_diferencias_entre_pliegues": float(np.std(dif_k, ddof=1)) if len(dif_k) > 1 else None,
                       "pruebas_5_pliegues": pruebas_pareadas(dif_k),
                       "pares_pliegue_semilla": {
                           "n": len(dif_ks), "diferencias": [float(x) for x in dif_ks],
                           "a_favor": int(np.sum(np.array(dif_ks) > 1e-12)),
                           "en_contra": int(np.sum(np.array(dif_ks) < -1e-12)),
                           "nota": "no independientes (las semillas de un mismo pliegue comparten "
                                   "senante de prueba y datos); solo descriptivo"}}
        comparaciones[f"{b1}_menos_{b2}"] = comp

    # Criterio prerregistrado (A frente a B, validación)
    criterio = None
    if "A_menos_B" in comparaciones:
        dv = comparaciones["A_menos_B"]["mejor_val_accuracy"]
        d = np.array(dv["diferencia_por_pliegue"])
        n_a, n_b = int(np.sum(d > 1e-12)), int(np.sum(d < -1e-12))
        media = float(np.mean(d))
        if ruido is None:
            veredicto = "sin medida de variacion entre semillas (hace falta mas de una semilla)"
        elif n_a >= 4 and media > ruido:
            veredicto = "Las velocidades ayudan"
        elif n_b >= 4 and -media > ruido:
            veredicto = "Las velocidades estorban"
        else:
            veredicto = "No hay evidencia"
        criterio = {
            "regla": "Ayudan si A > B en val top-1 (media sobre semillas) en >= 4 de 5 pliegues Y la "
                     "diferencia media A-B supera la variacion entre semillas (DE combinada de val "
                     "top-1 entre semillas dentro de cada celda brazo-pliegue de A y B). Estorban si "
                     "B > A en >= 4 de 5 y la diferencia media B-A supera esa variacion. Si no, no hay "
                     "evidencia. Empates (diferencia 0) no cuentan para ningun lado.",
            "pliegues_a_favor_de_A": n_a, "pliegues_a_favor_de_B": n_b,
            "empates": int(len(d) - n_a - n_b),
            "diferencia_media_val": media, "variacion_entre_semillas": ruido,
            "veredicto": veredicto,
        }

    # Reproducción de la referencia N-30 con el brazo A, semilla 2026
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
                        "identica": all(abs(r["ref_top1"] - r["A_top1"]) < 1e-9 and
                                        abs(r["ref_val"] - r["A_val"]) < 1e-9 and
                                        r["ref_epocas"] == r["A_epocas"] for r in filas_rep)}

    inferencia = None
    if (out / "inferencia.json").exists():
        inf = json.loads((out / "inferencia.json").read_text(encoding="utf-8"))
        inferencia = {"metodo": inf["metodo"], "ms_construir_entrada_mediana": inf.get("ms_construir_entrada_mediana")}
        for b in brazos:
            rs = [r for r in inf["modelos"] if r["brazo"] == b]
            if rs:
                inferencia[b] = {"n_modelos": len(rs), "n_parametros": rs[0]["n_parametros"],
                                 "ms_lote1_mediana_de_medianas": float(np.median([r["ms_lote1_mediana"] for r in rs])),
                                 "ms_lote1_min_max": [float(min(r["ms_lote1_mediana"] for r in rs)),
                                                      float(max(r["ms_lote1_mediana"] for r in rs))],
                                 "ms_por_muestra_lote200_mediana": float(np.median([r["ms_por_muestra_lote200_mediana"] for r in rs]))}

    verif = None
    if (out / "verificacion.json").exists():
        verif = json.loads((out / "verificacion.json").read_text(encoding="utf-8"))

    resultado = {
        "experimento": "ablacion de velocidades (docs/EXPERIMENTO_VELOCIDADES.md)",
        "fecha_resumen": datetime.now().isoformat(timespec="seconds"),
        "brazos": {b: BRAZOS[b] for b in brazos},
        "semillas_por_brazo": semillas_por_brazo, "pliegues": folds,
        "criterio": criterio, "variacion_entre_semillas_val": ruido,
        "comparaciones": comparaciones, "por_brazo": por_brazo,
        "reproduccion_referencia_N30": reproduccion, "inferencia": inferencia,
        "verificacion": verif,
        "corridas": filas,
    }
    with open(out / "resultados_velocidades.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=1, ensure_ascii=False)
    if args.docs:
        with open(JSON_DOCS, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=1, ensure_ascii=False)

    # Impresión
    print(f"brazos {brazos} | semillas {semillas_por_brazo} | pliegues {folds}")
    for b in brazos:
        ag = por_brazo[b]["agregado"]
        print(f"\n{b} ({BRAZOS[b]['descripcion']}): val {ag['mejor_val_accuracy']['media']:.4f} ± "
              f"{ag['mejor_val_accuracy']['de'] or 0:.4f} | top-1 {ag['top1']['media']:.4f} ± "
              f"{ag['top1']['de'] or 0:.4f} | top-3 {ag['top3']['media']:.4f} | F1 {ag['f1_macro']['media']:.4f} | "
              f"epocas {ag['epocas_ejecutadas']['media']:.1f} | params {int(ag['n_parametros']['media'])}")
        for k in folds:
            c = por_brazo[b]["por_pliegue"][str(k)]
            print(f"   pliegue {k}: val {c['mejor_val_accuracy']['valores']} top-1 "
                  f"{[round(x, 3) for x in c['top1']['valores']]} epocas {c['epocas_ejecutadas']['valores']}")
        for s, v in por_brazo[b]["por_semilla"].items():
            print(f"   semilla {s}: val {v['mejor_val_accuracy']['media']:.4f} top-1 {v['top1']['media']:.4f} "
                  f"± {v['top1']['de'] or 0:.4f} top-3 {v['top3']['media']:.4f} F1 {v['f1_macro']['media']:.4f}")
    print(f"\nvariacion entre semillas (val, DE combinada A y B): {ruido}")
    for nombre, comp in comparaciones.items():
        print(f"\n{nombre} (semillas {comp['semillas_comunes']}):")
        for m in ["mejor_val_accuracy", "top1", "top3", "f1_macro"]:
            c = comp[m]
            pr = c["pruebas_5_pliegues"]
            print(f"   {m:20s} dif por pliegue {[round(x, 4) for x in c['diferencia_por_pliegue']]} | media "
                  f"{c['diferencia_media']:+.4f} | a favor {pr['a_favor_de_A']}/{len(c['diferencia_por_pliegue'])} | "
                  f"signos p={pr['signos_p_bilateral']:.4f} | wilcoxon p={pr['wilcoxon_p_bilateral']:.4f} | "
                  f"pares pliegue-semilla {c['pares_pliegue_semilla']['a_favor']}/{c['pares_pliegue_semilla']['n']}")
    if criterio:
        print(f"\nCRITERIO: {criterio['veredicto']} (A>B en {criterio['pliegues_a_favor_de_A']}/5, B>A en "
              f"{criterio['pliegues_a_favor_de_B']}/5, dif media val {criterio['diferencia_media_val']:+.4f}, "
              f"variacion entre semillas {criterio['variacion_entre_semillas']})")
    if reproduccion:
        print(f"\nreproduccion N-30 (A, semilla 2026): identica = {reproduccion['identica']}")
    if inferencia:
        print(f"\ninferencia: {json.dumps(inferencia, ensure_ascii=False)}")
    print(f"\nescrito {tl._rel(out / 'resultados_velocidades.json')}" +
          (f" y {tl._rel(JSON_DOCS)}" if args.docs else ""))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablacion de velocidades en el LOSO de 5 pliegues.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def comunes(sp):
        sp.add_argument("--out", type=str, default=str(DEFAULT_OUT))
        sp.add_argument("--processed", type=str, default=str(PROCESSED_CANONICO))
        sp.add_argument("--representacion", type=str, default=REPRESENTACION_CANONICA)

    sv = sub.add_parser("verificar")
    comunes(sv)
    sv.add_argument("--guardar", action="store_true", help="escribe verificacion.json en --out")

    se = sub.add_parser("entrenar")
    comunes(se)
    se.add_argument("--brazos", type=str, default="A,B")
    se.add_argument("--semillas", type=str, default="2026,2027")
    se.add_argument("--folds", type=str, default="5")
    se.add_argument("--epochs", type=int, default=200)
    se.add_argument("--aug", type=int, default=10)
    se.add_argument("--batch-size", type=int, default=64)
    se.add_argument("--patience", type=int, default=25)
    se.add_argument("--dropout", type=float, default=0.3)

    si = sub.add_parser("inferencia")
    comunes(si)
    si.add_argument("--calentamiento", type=int, default=20)
    si.add_argument("--repeticiones", type=int, default=100)

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
