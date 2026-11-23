"""
Experimento: ¿se pueden reducir las 126 velocidades a 2 (una por mano) o a 6 (una por mano y eje) sin
perder exactitud? (pedido del director, reunión del 2-oct-2026, N-60 encargo 2). LOSO de 5 pliegues.

Brazos (lo ÚNICO que cambia es la entrada de la red, construida DESPUÉS del remuestreo y de la
aumentación a partir de las 126 posiciones; definición en src/velocidades_reducidas.py):
  A  actual          30 x 252  126 posiciones + 126 velocidades (canónica; = add_velocity)
  B  director        30 x 128  126 posiciones + 2 (por mano, media de sus 63 columnas de velocidad)
  C  por_coordenada  30 x 132  126 posiciones + 6 (por mano, media de las 21 vx, 21 vy y 21 vz)

Reutiliza SIN modificar train_loso.cargar_datos, verificar_particion, importar_pipeline_canonico,
remuestrear_todas y entrenar_y_evaluar. entrenar_y_evaluar recibe canon = (tf, build_model, add_velocity,
resample_sequence, transformaciones) y llama a canon[2] DESPUÉS de aumentar (el espejo actúa sobre las
126 posiciones); basta con sustituir canon[2] por la función del brazo. build_model toma el ancho de esa
entrada. Regla de mano ausente: «mascara» (D-07), la vigente.

La procedencia que entrenar_y_evaluar escribe junto al modelo dice velocidades=true y 252 rasgos; se
reescribe después con la entrada real del brazo.

Criterio prerregistrado en docs/EXPERIMENTO_VELOCIDADES_REDUCIDAS.md §2 (no inferioridad en validación;
la prueba LOSO solo describe).

Subcomandos:
  verificar   sin entrenar: A = add_velocity, anchos, espejo (C: manos intercambiadas y vx negada) y
              cuánto de la «energía» de las velocidades conserva cada reducción en el corpus.
  entrenar    combinaciones (semilla, pliegue, brazo) que falten; reanudable.
  inferencia  parámetros y latencia (lote 1, como translator.py) de cada modelo; sin entrenamientos.
  figura      mapas de color de las velocidades de «hola» 0030_0000_0003 (A 30x126, B 30x2, C 30x6).
  resumen     agrega, aplica el criterio y escribe el JSON de datos crudos.

Uso:
  python src/experimento_velocidades_reducidas.py verificar --guardar
  python src/experimento_velocidades_reducidas.py entrenar --brazos A,B,C --semillas 2026 --folds 0, \\
         --epochs 2 --aug 2 --out <carpeta_temporal>                          # prueba de humo
  python src/experimento_velocidades_reducidas.py entrenar --brazos A,B,C --semillas 2026
  python src/experimento_velocidades_reducidas.py inferencia
  python src/experimento_velocidades_reducidas.py figura
  python src/experimento_velocidades_reducidas.py resumen --docs
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import train_loso as tl  # noqa: E402
from src import velocidades_reducidas as vr  # noqa: E402

PROCESSED_CANONICO = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
REPRESENTACION = "manos"
DEFAULT_OUT = BASE_DIR / "models" / "experimento_velocidades_reducidas_2026-10-02"
JSON_DOCS = BASE_DIR / "docs" / "EXPERIMENTO_VELOCIDADES_REDUCIDAS.json"
REFERENCIA_N30 = BASE_DIR / "models" / "loso_2026-09-15_holistic_mascara_manos" / "resultados_loso.json"
FIGURAS = BASE_DIR / "docs" / "figuras_2026-10-02"
PRIVADAS = BASE_DIR.parent / "Presentaciones_privadas" / "explicaciones"
VIDEO_FIGURA = "0030_0000_0003"

BRAZOS = {
    "A": {"carpeta": "A_actual_252", "forma": "actual", "n_rasgos": 252,
          "descripcion": "126 posiciones + 126 velocidades (canonica)"},
    "B": {"carpeta": "B_director_128", "forma": "director", "n_rasgos": 128,
          "descripcion": "126 posiciones + 2 velocidades (media de las 63 columnas por mano)"},
    "C": {"carpeta": "C_por_coordenada_132", "forma": "por_coordenada", "n_rasgos": 132,
          "descripcion": "126 posiciones + 6 velocidades (media de vx, vy, vz por mano)"},
}
PISO_MARGEN = 0.01
TOL = 1e-9   # tolerancia de redondeo de coma flotante al comparar con el margen


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
    from src.representacion import ruta_procedencia
    ruta = ruta_procedencia(ruta_modelo)
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    forma = BRAZOS[brazo]["forma"]
    datos["velocidades"] = forma
    datos["n_rasgos_entrada"] = BRAZOS[brazo]["n_rasgos"]
    datos["columnas_velocidad"] = vr.nombres_columnas_velocidad(forma) if forma != "actual" else "126 (add_velocity)"
    datos["brazo_experimento_velocidades_reducidas"] = brazo
    datos["nota"] = ("modelo del experimento de velocidades reducidas (docs/EXPERIMENTO_VELOCIDADES_REDUCIDAS.md); "
                     "NO es el modelo canonico ni debe cargarse en el prototipo")
    datos["script"] = "src/experimento_velocidades_reducidas.py (via train_loso.entrenar_y_evaluar)"
    ruta.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Verificación sin entrenar
# ---------------------------------------------------------------------------

def energia_conservada(V, cero_izq, cero_der):
    """Qué fracción de la «energía» (suma de cuadrados) de las 126 velocidades queda explicada si cada
    columna se reemplaza por la media de su mano (B) o por la media de su mano y eje (C). Se calcula en
    todos los pasos 2-30 y solo en los pasos con la mano presente en t y t−1 (sin saltos de «mascara»).
    Además, la correlación mediana entre pares de columnas de una mano."""
    res = {}
    for nombre, sl, cero in (("mano_izq", slice(0, 63), cero_izq), ("mano_der", slice(63, 126), cero_der)):
        Vm = V[:, 1:, sl].astype(np.float64)              # (N, 29, 63)
        presente = (~cero[:, 1:]) & (~cero[:, :-1])       # (N, 29)
        movil = np.linalg.norm(Vm, axis=2) > 0
        sub = {}
        for etiqueta, mascara in (("todos_los_pasos", np.ones_like(presente)),
                                  ("mano_presente_en_t_y_t-1", presente)):
            F = Vm[mascara & movil]                       # (n, 63)
            ss = float(np.sum(F ** 2))
            mB = F.mean(axis=1, keepdims=True)
            mC = np.stack([F[:, c::3].mean(axis=1) for c in range(3)], axis=1)   # (n, 3)
            # reconstrucción C: la columna 3p+c recibe la media del eje c
            recC = np.tile(mC, (1, 21)) if len(F) else F
            corr = np.corrcoef(F.T) if len(F) > 2 else np.full((63, 63), np.nan)
            iu = np.triu_indices(63, 1)
            ejes = {}
            for c, e in enumerate("xyz"):
                cc = np.corrcoef(F[:, c::3].T)[np.triu_indices(21, 1)] if len(F) > 2 else np.array([np.nan])
                ejes[e] = float(np.nanmedian(cc))
            sub[etiqueta] = {
                "n_pasos": int(len(F)),
                "fraccion_energia_explicada_B": float(1 - np.sum((F - mB) ** 2) / ss) if ss > 0 else None,
                "fraccion_energia_explicada_C": float(1 - np.sum((F - recC) ** 2) / ss) if ss > 0 else None,
                "correlacion_mediana_entre_las_63_columnas": float(np.nanmedian(corr[iu])),
                "correlacion_mediana_dentro_de_cada_eje_21_columnas": ejes,
                "correlacion_entre_ejes_de_la_media_C": {
                    "x_y": float(np.corrcoef(mC[:, 0], mC[:, 1])[0, 1]),
                    "x_z": float(np.corrcoef(mC[:, 0], mC[:, 2])[0, 1]),
                    "y_z": float(np.corrcoef(mC[:, 1], mC[:, 2])[0, 1])} if len(F) > 2 else None,
            }
        res[nombre] = sub
    return res


def cmd_verificar(args):
    from src.representacion import espejar
    processed = Path(args.processed).resolve()
    X_padded, y, lengths, nombres, loso, pos = cargar_todo(processed)
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    tf, _, add_velocity, resample_sequence, transformaciones = canon
    X_seq = tl.remuestrear_todas(X_padded, lengths, resample_sequence, tl.SEQ_LENGTH)
    print(f"processed {tl._rel(processed)} | X_seq {X_seq.shape}")

    # 1. A es exactamente add_velocity; anchos; posiciones intactas; velocidad del paso 1 = 0
    E = {b: vr.funcion_entrada(BRAZOS[b]["forma"])(X_seq) for b in BRAZOS}
    assert np.array_equal(E["A"], add_velocity(X_seq)), "A debe ser exactamente add_velocity"
    for b in BRAZOS:
        assert E[b].shape[2] == BRAZOS[b]["n_rasgos"], b
        assert np.array_equal(E[b][:, :, :126], X_seq), b
        assert np.all(E[b][:, 0, 126:] == 0), b
    V = E["A"][:, :, 126:]
    assert np.allclose(E["B"][:, :, 126], V[:, :, :63].mean(-1), atol=1e-6)
    assert np.allclose(E["C"][:, :, 126 + 3], V[:, :, 63::3].mean(-1), atol=1e-6)
    print("[ok] A = add_velocity; anchos 252/128/132; posiciones intactas; velocidad del paso 1 = 0")

    # 2. Espejo: espejar posiciones y luego reducir. En C: manos intercambiadas y vx negada.
    Xm = espejar(X_seq, REPRESENTACION)
    Cm = vr.entrada_red(Xm, "por_coordenada")[:, :, 126:]
    err_c = float(np.max(np.abs(Cm - vr.espejo_esperado_reducido(E["C"][:, :, 126:], "por_coordenada"))))
    Bm = vr.entrada_red(Xm, "director")[:, :, 126:]
    Vd = V.astype(np.float64)
    esp_b_izq = (-Vd[:, :, 63::3].sum(-1) + Vd[:, :, 64::3].sum(-1) + Vd[:, :, 65::3].sum(-1)) / 63
    err_b = float(np.max(np.abs(Bm[..., 0] - esp_b_izq)))
    ingenuo = float(np.max(np.abs(Bm - E["B"][:, :, 126:][..., ::-1])))
    assert err_c < 1e-5 and err_b < 1e-5
    print(f"[ok] espejo: C max err {err_c:.2e} (manos intercambiadas, vx negada); B max err {err_b:.2e} "
          f"(media de la otra mano con x negadas); intercambiar solo las 2 columnas de B daria un error de "
          f"hasta {ingenuo:.3f}: por eso se reduce DESPUES de aumentar")

    # 3. La aumentación no depende del brazo (se aplica antes de canon[2]) y es determinista
    idx_tr = np.array([pos[n] for n in loso["pliegues"][0]["conjuntos"]["train"]], dtype=np.int64)
    sub = X_seq[idx_tr[:60]]
    salidas = []
    for _ in range(2):
        tf.keras.utils.set_random_seed(2026)
        salidas.append(tl.aumentar(sub, y[idx_tr[:60]], 10, transformaciones))
    assert np.array_equal(salidas[0][0], salidas[1][0]) and salidas[0][0].shape[2] == 126
    print(f"[ok] aumentacion x10 determinista sobre 126 columnas ({len(salidas[0][1])} secuencias de 60)")

    # 4. Cuánto se parecen las 63 columnas de una mano (la observación del director)
    cero_izq = np.all(X_seq[:, :, :63] == 0, axis=2)
    cero_der = np.all(X_seq[:, :, 63:] == 0, axis=2)
    en = energia_conservada(V, cero_izq, cero_der)
    print("[info] energia de las velocidades explicada por cada reduccion (corpus remuestreado, 1000 muestras):")
    for m, d in en.items():
        for etq, v in d.items():
            print(f"  {m} {etq}: {json.dumps(v, ensure_ascii=False)}")
    if args.guardar:
        out = Path(args.out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "verificacion.json", "w", encoding="utf-8") as f:
            json.dump({"fecha": datetime.now().isoformat(timespec="seconds"), "processed": tl._rel(processed),
                       "comprobaciones": ["A = add_velocity exacto", "anchos 252/128/132", "posiciones intactas",
                                          "velocidad del paso 1 = 0",
                                          "espejo y luego reducir: C = manos intercambiadas y vx negada",
                                          "espejo y luego reducir: B = media de la otra mano con x negadas",
                                          "aumentacion determinista e independiente del brazo"],
                       "espejo_max_error": {"C": err_c, "B": err_b,
                                            "B_si_solo_se_intercambiaran_columnas": ingenuo},
                       "energia_explicada": en}, f, indent=1, ensure_ascii=False)
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
    X_seq = tl.remuestrear_todas(X_padded, lengths, resample_sequence, tl.SEQ_LENGTH)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] experimento_velocidades_reducidas entrenar | "
          f"brazos {brazos} | semillas {semillas} | pliegues {folds} | epochs {args.epochs} | aug x{args.aug} | "
          f"paciencia {args.patience} | X_seq {X_seq.shape} | salida {tl._rel(out)}", flush=True)

    for semilla in semillas:
        for k in folds:
            p = loso["pliegues"][k]
            idx_val = np.array([pos[n] for n in p["conjuntos"]["val"]], dtype=np.int64)
            for brazo in brazos:
                d = dir_corrida(out, brazo, semilla, k)
                if (d / "metricas_velocidades_reducidas.json").exists():
                    print(f"  ya existe {tl._rel(d)}; se salta", flush=True)
                    continue
                entrada = vr.funcion_entrada(BRAZOS[brazo]["forma"])
                if brazo == "A":
                    entrada = add_velocity   # la función canónica misma (verificar: idéntica)
                canon_brazo = (tf, build_model, entrada, resample_sequence, transformaciones)
                print(f"\n{'=' * 72}\n  brazo {brazo} ({BRAZOS[brazo]['descripcion']}) | semilla {semilla} | "
                      f"pliegue {k} (senante {p['senante_test']}) | {datetime.now().isoformat(timespec='seconds')}"
                      f"\n{'=' * 72}", flush=True)
                ns = argparse.Namespace(seed=semilla, aug=args.aug, epochs=args.epochs, batch_size=args.batch_size,
                                        patience=args.patience, dropout=args.dropout, processed=str(processed),
                                        representacion=REPRESENTACION)
                m = tl.entrenar_y_evaluar(f"{brazo}_s{semilla}_pliegue_{k}", p["conjuntos"], pos, X_seq, y, ns, d,
                                          canon_brazo, nombres_senas)
                reescribir_procedencia(d / "modelo.keras", brazo)
                model = tf.keras.models.load_model(str(d / "modelo.keras"))
                mv = metricas_validacion(model, entrada(X_seq[idx_val]), y[idx_val])
                n_param = int(model.count_params())
                n_in = int(model.input_shape[-1])
                del model
                tf.keras.backend.clear_session()
                assert n_in == BRAZOS[brazo]["n_rasgos"], (n_in, brazo)
                m.update(mv)
                m.update({"brazo": brazo, "forma": BRAZOS[brazo]["forma"], "semilla": semilla, "pliegue": k,
                          "senante_test": p["senante_test"], "n_parametros": n_param, "n_rasgos_entrada": n_in,
                          "config": {"epochs_max": args.epochs, "aug": args.aug, "batch_size": args.batch_size,
                                     "patience": args.patience, "dropout": args.dropout,
                                     "seq_length": tl.SEQ_LENGTH, "processed": tl._rel(processed),
                                     "representacion": REPRESENTACION},
                          "fecha_fin": datetime.now().isoformat(timespec="seconds")})
                with open(d / "metricas_velocidades_reducidas.json", "w", encoding="utf-8") as f:
                    json.dump(m, f, indent=1)
                print(f"  -> {brazo} s={semilla} k={k}: val {m['mejor_val_accuracy']:.3f} (checkpoint "
                      f"{mv['val_top1']:.3f}) | test top-1 {m['top1']:.3f} | entrada {n_in} | params {n_param} | "
                      f"epocas {m['epocas_ejecutadas']} | {m['tiempo_total_s']:.0f} s", flush=True)
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] fin", flush=True)


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

def cmd_inferencia(args):
    """Latencia como en el prototipo (translator.py: model.predict de un lote de 1): mediana de
    --repeticiones llamadas tras --calentamiento, por modelo. Además lote de 200 y el costo de construir la
    entrada de una ventana (30 x 126 -> 30 x F)."""
    out = Path(args.out).resolve()
    tf, _, add_velocity, _, _ = tl.importar_pipeline_canonico(REPRESENTACION)
    rng = np.random.default_rng(0)
    filas = []
    # Orden intercalado (A, B, C del mismo pliegue/semilla seguidos) para que una deriva del equipo no
    # favorezca a un brazo.
    rutas = {b: sorted((out / BRAZOS[b]["carpeta"]).glob("semilla_*/pliegue_*/modelo.keras")) for b in BRAZOS}
    n = max(len(v) for v in rutas.values())
    for i in range(n):
        for brazo in BRAZOS:
            if i >= len(rutas[brazo]):
                continue
            ruta = rutas[brazo][i]
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
            # Primera capa: parámetros de la Conv1D de entrada
            conv1 = [l for l in model.layers if l.__class__.__name__ == "Conv1D"][0]
            fila = {"brazo": brazo, "modelo": tl._rel(ruta), "n_rasgos_entrada": n_feat,
                    "n_parametros": int(model.count_params()), "parametros_primera_conv": int(conv1.count_params()),
                    "ms_lote1_mediana": float(np.median(t)), "ms_lote1_p90": float(np.percentile(t, 90)),
                    "ms_por_muestra_lote200_mediana": float(np.median(tb))}
            filas.append(fila)
            print(f"  {brazo} {tl._rel(ruta)}: entrada {n_feat} | params {fila['n_parametros']} | lote 1 "
                  f"{fila['ms_lote1_mediana']:.2f} ms (p90 {fila['ms_lote1_p90']:.2f}) | lote 200 "
                  f"{fila['ms_por_muestra_lote200_mediana']:.3f} ms/muestra", flush=True)
            del model
            tf.keras.backend.clear_session()
    xp = rng.standard_normal((1, tl.SEQ_LENGTH, 126)).astype(np.float32)
    costo = {}
    for brazo in BRAZOS:
        fn = add_velocity if brazo == "A" else vr.funcion_entrada(BRAZOS[brazo]["forma"])
        t = []
        for _ in range(3000):
            t0 = time.perf_counter()
            fn(xp)
            t.append((time.perf_counter() - t0) * 1000)
        costo[brazo] = float(np.median(t))
    print(f"  construir la entrada (ms, mediana de 3000): {costo}")
    import subprocess
    try:
        carga = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    except Exception:
        carga = None
    with open(out / "inferencia.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": datetime.now().isoformat(timespec="seconds"),
                   "metodo": f"model.predict(verbose=0) de un lote de 1 como translator.py; {args.calentamiento} "
                             f"llamadas de calentamiento y {args.repeticiones} medidas por modelo; ademas lote de "
                             f"200 (5 medidas). CPU, sin otros entrenamientos corriendo. Solo la red: sin MediaPipe "
                             f"ni remuestreo. Modelos intercalados A, B, C.",
                   "carga_equipo_al_terminar": carga, "entorno": tl.info_entorno(), "modelos": filas,
                   "ms_construir_entrada_mediana": costo}, f, indent=1)


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def cmd_figura(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from src.grammar_mapper import SIGN_VOCABULARY
    from src.train_final_100 import resample_sequence

    TINTA, TINTA2, GRIS = "#0b0b0b", "#52514e", "#8a8985"
    COL_EJE = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2,
                         "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.titlecolor": TINTA,
                         "axes.spines.top": False, "axes.spines.right": False})

    X_padded, y, L, nombres, _ = tl.cargar_datos(Path(args.processed).resolve(), REPRESENTACION)
    i = nombres.index(VIDEO_FIGURA)
    X = resample_sequence(X_padded[i, :int(L[i])], tl.SEQ_LENGTH)[None]       # (1, 30, 126)
    A = vr.entrada_red(X, "actual")[0, :, 126:]
    B = vr.entrada_red(X, "director")[0, :, 126:]
    C = vr.entrada_red(X, "por_coordenada")[0, :, 126:]
    cero_izq = np.all(X[0, :, :63] == 0, axis=1)
    cero_der = np.all(X[0, :, 63:] == 0, axis=1)
    # Pasos contaminados por «mascara»: interviene (con peso > 0) un cuadro sin la mano (en cero o
    # «encogidos» hacia el origen); la velocidad del paso t es artificial si t o t−1 está contaminado.
    T = int(L[i])
    seq = X_padded[i, :T]
    pj = np.linspace(0, T - 1, tl.SEQ_LENGTH)
    lo = np.floor(pj).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = pj - lo
    contam, vel_art = {}, {}
    for m, sl in (("izq", slice(0, 63)), ("der", slice(63, 126))):
        det = ~np.all(seq[:, sl] == 0, axis=1)
        c = (~det[lo] & (w < 1 - 1e-9)) | (~det[hi] & (w > 1e-9))
        contam[m] = c
        v = np.zeros(tl.SEQ_LENGTH, dtype=bool)
        v[1:] = c[1:] | c[:-1]
        vel_art[m] = v
    lim = float(np.percentile(np.abs(A), 99))
    clase = SIGN_VOCABULARY.get(int(VIDEO_FIGURA.split("_")[0]), "")
    clase_arch = clase.replace("¿", "").replace("?", "").replace("ñ", "n")

    fig = plt.figure(figsize=(16, 10.5))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[10, 1.3, 2.6, 0.25], height_ratios=[1.25, 1],
                  wspace=0.12, hspace=0.42)
    extent_y = (30.5, 0.5)
    axA = fig.add_subplot(gs[0, 0])
    im = axA.imshow(A, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
                    extent=(0.5, 126.5) + extent_y)
    axA.axvline(63.5, color=TINTA, lw=0.8)
    axA.set_title("A «actual»: 126 velocidades (30 × 126)", loc="left")
    axA.set_xticks([1, 32, 63, 95, 126])
    axA.set_xlabel("columna de velocidad (izq 1-63 | der 64-126; dentro de cada mano x, y, z intercalados)")
    axA.set_ylabel("paso (1-30)")
    axA.set_yticks([1, 5, 10, 12, 15, 20, 25, 30])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    axB.imshow(B, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
               extent=(0.5, 2.5) + extent_y)
    axB.set_title("B «director»\n(30 × 2)", loc="left")
    axB.set_xticks([1, 2], ["izq", "der"])
    axB.tick_params(labelleft=False)
    axC = fig.add_subplot(gs[0, 2], sharey=axA)
    axC.imshow(C, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
               extent=(0.5, 6.5) + extent_y)
    axC.axvline(3.5, color=TINTA, lw=0.8)
    axC.set_title("C «por coordenada»\n(30 × 6)", loc="left")
    axC.set_xticks(range(1, 7), ["x", "y", "z", "x", "y", "z"])
    axC.set_xlabel("izq          der")
    axC.tick_params(labelleft=False)
    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("velocidad (anchos de hombro por paso); blanco = 0; misma escala en A, B y C")

    # Fila 2: las 63 curvas de cada mano y sus medias
    for col, (nombre_m, sl, cero) in enumerate((("mano izquierda", slice(0, 63), vel_art["izq"]),
                                                ("mano derecha", slice(63, 126), vel_art["der"]))):
        ax = fig.add_subplot(gs[1, 0] if col == 0 else gs[1, 1:3])
        pasos = np.arange(1, 31)
        bloque = A[:, sl]
        for c, e in enumerate("xyz"):
            for p in range(21):
                ax.plot(pasos, bloque[:, 3 * p + c], color=COL_EJE[e], lw=0.6, alpha=0.25)
        m = 0 if col == 0 else 1
        for c, e in enumerate("xyz"):
            ax.plot(pasos, C[:, 3 * m + c], color=COL_EJE[e], lw=2.2, label=f"C: media de las 21 v{e}")
        ax.plot(pasos, B[:, m], color=TINTA, lw=2.2, ls="--", label="B: media de las 63")
        for j in np.flatnonzero(cero):
            ax.axvspan(j + 0.5, j + 1.5, color=GRIS, alpha=0.15, lw=0)
        ax.axhline(0, color=GRIS, lw=0.6)
        ax.set_xlim(0.5, 30.5)
        ax.set_title(f"{nombre_m}: 63 velocidades (finas, color por eje) y sus medias (gruesas)", loc="left",
                     fontsize=10)
        ax.set_xlabel("paso")
        if col == 0:
            ax.set_ylabel("velocidad")
            ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    def _lst(v):
        return ", ".join(str(j + 1) for j in np.flatnonzero(v)) or "ninguno"
    fig.suptitle(f"«{clase}» {VIDEO_FIGURA}: velocidades de la entrada de la red con cada forma (misma escala de "
                 f"color en A, B y C).\nFranjas grises abajo: velocidades artificiales de la regla «mascara» (el paso "
                 f"o el anterior usa un cuadro sin la mano: en 0 o «encogido» hacia el origen).\n"
                 f"Mano izquierda: pasos {_lst(vel_art['izq'])} (en 0: {_lst(cero_izq)}). "
                 f"Mano derecha: pasos {_lst(vel_art['der'])}.",
                 x=0.01, y=1.02, ha="left", fontsize=11)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS / f"velocidades_reducidas_mapas_{clase_arch}_{VIDEO_FIGURA}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    info = {"video": VIDEO_FIGURA, "clase": clase, "escala_comun_lim_p99_abs_A": lim,
            "pasos_mano_izq_en_cero": (np.flatnonzero(cero_izq) + 1).tolist(),
            "pasos_mano_der_en_cero": (np.flatnonzero(cero_der) + 1).tolist(),
            "pasos_contaminados_posicion": {m: (np.flatnonzero(v) + 1).tolist() for m, v in contam.items()},
            "pasos_velocidad_artificial": {m: (np.flatnonzero(v) + 1).tolist() for m, v in vel_art.items()},
            "max_abs": {"A": float(np.abs(A).max()), "B": float(np.abs(B).max()), "C": float(np.abs(C).max())},
            "corr_B_izq_con_media_C_izq": float(np.corrcoef(B[1:, 0], C[1:, 0:3].mean(1))[0, 1]),
            "fraccion_energia_explicada_hola": {}}
    for m, sl in (("izq", slice(0, 63)), ("der", slice(63, 126))):
        F = A[1:, sl].astype(np.float64)
        ss = float(np.sum(F ** 2))
        k = 0 if m == "izq" else 1
        recB = np.repeat(B[1:, k:k + 1], 63, axis=1)
        recC = np.tile(C[1:, 3 * k:3 * k + 3], (1, 21))
        info["fraccion_energia_explicada_hola"][m] = {"B": 1 - float(np.sum((F - recB) ** 2)) / ss,
                                                      "C": 1 - float(np.sum((F - recC) ** 2)) / ss}
    (FIGURAS / "velocidades_reducidas_figura.json").write_text(json.dumps(info, indent=1, ensure_ascii=False),
                                                               encoding="utf-8")
    if PRIVADAS.exists():
        shutil.copy2(ruta, PRIVADAS / ruta.name)
        print(f"copiada a {PRIVADAS / ruta.name}")
    print(tl._rel(ruta))
    print(json.dumps(info, ensure_ascii=False, indent=1))


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
            "precision_macro", "recall_macro", "epocas_ejecutadas", "mejor_epoca", "tiempo_total_s",
            "n_parametros", "n_rasgos_entrada"]


def aplicar_criterio(val_k, margen):
    """Regla prerregistrada (§2). val_k: {brazo: [val media sobre semillas por pliegue]}."""
    detalle = {}
    for b in [x for x in ("B", "C") if x in val_k]:
        d = np.array(val_k[b]) - np.array(val_k["A"])
        media = float(np.mean(d))
        pierde_mas_margen = int(np.sum(d < -margen - TOL))
        a_ok = bool(media >= -margen - TOL)
        b_ok = bool(pierde_mas_margen <= 1)
        detalle[b] = {"delta_por_pliegue": d.tolist(), "delta_media": media,
                      "pliegues_mejor": int(np.sum(d > 1e-12)), "pliegues_peor": int(np.sum(d < -1e-12)),
                      "pliegues_igual": int(np.sum(np.abs(d) <= 1e-12)),
                      "pliegues_que_pierden_por_mas_del_margen": pierde_mas_margen,
                      "a_delta_media_mayor_o_igual_que_menos_margen": a_ok,
                      "b_pierde_por_mas_del_margen_en_a_lo_sumo_1_pliegue": b_ok,
                      "no_es_peor_que_A": bool(a_ok and b_ok),
                      "informativo_regla_general_de_mejora": bool(media > margen + TOL and np.sum(d > 1e-12) >= 4)}
    cumplen = [b for b, v in detalle.items() if v["no_es_peor_que_A"]]
    val = {b: float(np.mean(v)) for b, v in val_k.items()}
    if not cumplen:
        eleccion, motivo = "A", "ni B ni C cumplen la no inferioridad: se mantienen las 126 velocidades (A)"
    elif cumplen == ["B"]:
        eleccion, motivo = "B", "solo B cumple: se adopta B (128 rasgos)"
    elif cumplen == ["C"]:
        eleccion, motivo = "C", "solo C cumple: se adopta C (132 rasgos)"
    else:
        if val["C"] - val["B"] > margen + TOL:
            eleccion, motivo = "C", "B y C cumplen y C supera a B por mas del margen: se adopta C"
        else:
            eleccion, motivo = "B", "B y C cumplen y C no supera a B por mas del margen: gana la mas pequena (B)"
    return {"detalle": detalle, "cumplen": cumplen, "val_media": val, "eleccion": eleccion, "motivo": motivo}


def cmd_resumen(args):
    out = Path(args.out).resolve()
    filas = []
    for b in BRAZOS:
        for p in sorted((out / BRAZOS[b]["carpeta"]).glob("semilla_*/pliegue_*/metricas_velocidades_reducidas.json")):
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

    # σ_semilla (§2): por brazo, DE (ddof=1) entre semillas de la val media de 5 pliegues; agrupada
    # sobre los tres brazos como raíz de la media de las varianzas. Margen = max(σ, 0,01).
    var_brazo = {}
    for b in brazos:
        v = [por_brazo[b]["por_semilla"][str(s)]["mejor_val_accuracy"]["media"]
             for s in semillas_por_brazo[b] if str(s) in por_brazo[b]["por_semilla"]]
        if len(v) > 1:
            var_brazo[b] = float(np.var(v, ddof=1))
    sigma = float(np.sqrt(np.mean(list(var_brazo.values())))) if var_brazo else None
    margen = max(sigma, PISO_MARGEN) if sigma is not None else None
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

    criterio = None
    if margen is not None and "A" in brazos:
        val_k = {b: [por_brazo[b]["por_pliegue"][str(k)]["mejor_val_accuracy"]["media"] for k in folds]
                 for b in brazos}
        criterio = aplicar_criterio(val_k, margen)
        criterio.update({
            "regla": "Una variante V en {B, C} NO ES PEOR que A (y se adopta) si (a) media de Δk = val(V,k) − "
                     "val(A,k) >= −margen y (b) Δk < −margen en a lo sumo 1 de los 5 pliegues; margen = "
                     "max(σ_semilla, 0,01). Si cumplen B y C, gana B salvo que val(C) − val(B) > margen. Si "
                     "ninguna cumple, se mantiene A. Comparaciones con tolerancia de coma flotante 1e-9.",
            "sigma_semilla": sigma, "varianza_entre_semillas_por_brazo": var_brazo, "margen": margen,
            "sigma_una_corrida_referencia": sigma_celda, "semillas": semillas_por_brazo})

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

    # Coincidencia con el brazo A del experimento de velocidades del 30-sep (mismo código canónico)
    previo = BASE_DIR / "models" / "experimento_velocidades_2026-09-30" / "A_con_velocidades"
    coincide_previo = None
    if previo.exists() and "A" in brazos:
        comp_prev = []
        for f in filas:
            if f["brazo"] != "A":
                continue
            mp = previo / f"semilla_{f['semilla']}" / f"pliegue_{f['pliegue']}" / "metricas_velocidades.json"
            if mp.exists():
                q = json.loads(mp.read_text(encoding="utf-8"))
                comp_prev.append(bool(abs(q["top1"] - f["top1"]) < 1e-9 and
                                      abs(q["mejor_val_accuracy"] - f["mejor_val_accuracy"]) < 1e-9 and
                                      q["epocas_ejecutadas"] == f["epocas_ejecutadas"]))
        coincide_previo = {"n_comparadas": len(comp_prev), "n_identicas": int(sum(comp_prev))}

    extra = {}
    for nombre in ("verificacion.json", "inferencia.json"):
        if (out / nombre).exists():
            extra[nombre] = json.loads((out / nombre).read_text(encoding="utf-8"))
    inferencia_resumen = None
    if "inferencia.json" in extra:
        inf = extra["inferencia.json"]
        inferencia_resumen = {}
        for b in brazos:
            rs = [r for r in inf["modelos"] if r["brazo"] == b]
            if rs:
                inferencia_resumen[b] = {
                    "n_modelos": len(rs), "n_parametros": rs[0]["n_parametros"],
                    "parametros_primera_conv": rs[0].get("parametros_primera_conv"),
                    "ms_lote1_mediana_de_medianas": float(np.median([r["ms_lote1_mediana"] for r in rs])),
                    "ms_lote1_min_max": [float(min(r["ms_lote1_mediana"] for r in rs)),
                                         float(max(r["ms_lote1_mediana"] for r in rs))],
                    "ms_por_muestra_lote200_mediana": float(np.median([r["ms_por_muestra_lote200_mediana"]
                                                                       for r in rs])),
                    "ms_construir_entrada": inf.get("ms_construir_entrada_mediana", {}).get(b)}

    resultado = {"experimento": "velocidades reducidas: 252 / 128 / 132 (docs/EXPERIMENTO_VELOCIDADES_REDUCIDAS.md)",
                 "fecha_resumen": datetime.now().isoformat(timespec="seconds"),
                 "brazos": {b: BRAZOS[b] for b in brazos}, "semillas_por_brazo": semillas_por_brazo,
                 "pliegues": folds, "criterio": criterio, "comparaciones": comparaciones,
                 "por_brazo": por_brazo, "reproduccion_referencia_N30": reproduccion,
                 "coincidencia_con_A_del_experimento_velocidades_2026-09-30": coincide_previo,
                 "inferencia_resumen": inferencia_resumen,
                 "verificacion": extra.get("verificacion.json"), "inferencia": extra.get("inferencia.json"),
                 "corridas": filas}
    with open(out / "resultados_velocidades_reducidas.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=1, ensure_ascii=False)
    if args.docs:
        with open(JSON_DOCS, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=1, ensure_ascii=False)

    print(f"brazos {brazos} | semillas {semillas_por_brazo} | pliegues {folds}")
    for b in brazos:
        ag = por_brazo[b]["agregado"]
        print(f"\n{b} ({BRAZOS[b]['descripcion']}): val {ag['mejor_val_accuracy']['media']:.4f} ± "
              f"{ag['mejor_val_accuracy']['de'] or 0:.4f} | val top-3 {ag['val_top3']['media']:.4f} | val F1 "
              f"{ag['val_f1_macro']['media']:.4f} | top-1 {ag['top1']['media']:.4f} ± {ag['top1']['de'] or 0:.4f}"
              f" | top-3 {ag['top3']['media']:.4f} | F1 {ag['f1_macro']['media']:.4f} | epocas "
              f"{ag['epocas_ejecutadas']['media']:.1f} | mejor epoca {ag['mejor_epoca']['media']:.1f} | params "
              f"{int(ag['n_parametros']['media'])} | s/corrida {ag['tiempo_total_s']['media']:.0f}")
        for k in folds:
            c = por_brazo[b]["por_pliegue"][str(k)]
            print(f"   pliegue {k}: val {c['mejor_val_accuracy']['valores']} top-1 "
                  f"{[round(x, 3) for x in c['top1']['valores']]} top-3 {[round(x, 3) for x in c['top3']['valores']]} "
                  f"F1 {[round(x, 3) for x in c['f1_macro']['valores']]} epocas {c['epocas_ejecutadas']['valores']}")
        for s, v in por_brazo[b]["por_semilla"].items():
            print(f"   semilla {s}: val {v['mejor_val_accuracy']['media']:.4f} top-1 {v['top1']['media']:.4f} "
                  f"F1 {v['f1_macro']['media']:.4f}")
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
        print(f"\nσ_semilla {sigma} (por brazo {var_brazo}) | margen {margen} | σ una corrida {sigma_celda}")
        for b, d in criterio["detalle"].items():
            print(f"   {b}: {d}")
        print(f"CRITERIO: eleccion {criterio['eleccion']}: {criterio['motivo']}")
    if reproduccion:
        print(f"\nreproduccion N-30 (A, semilla 2026): identica = {reproduccion['identica']}")
        for r in reproduccion["pliegues"]:
            print(f"   {r}")
    print(f"coincidencia con A del 30-sep: {coincide_previo}")
    if inferencia_resumen:
        print(f"inferencia: {json.dumps(inferencia_resumen, ensure_ascii=False)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Velocidades reducidas (LOSO de 5 pliegues).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def comunes(sp):
        sp.add_argument("--out", type=str, default=str(DEFAULT_OUT))
        sp.add_argument("--processed", type=str, default=str(PROCESSED_CANONICO))

    sv = sub.add_parser("verificar")
    comunes(sv)
    sv.add_argument("--guardar", action="store_true")
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
    si.add_argument("--calentamiento", type=int, default=20)
    si.add_argument("--repeticiones", type=int, default=100)
    sf = sub.add_parser("figura")
    comunes(sf)
    sr = sub.add_parser("resumen")
    comunes(sr)
    sr.add_argument("--docs", action="store_true")
    args = parser.parse_args()
    if not 1 <= getattr(args, "aug", 10) <= 10:
        raise SystemExit("--aug debe estar entre 1 y 10")
    {"verificar": cmd_verificar, "entrenar": cmd_entrenar, "inferencia": cmd_inferencia, "figura": cmd_figura,
     "resumen": cmd_resumen}[args.cmd](args)


if __name__ == "__main__":
    main()
