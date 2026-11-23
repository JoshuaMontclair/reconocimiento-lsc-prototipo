"""
Experimento: ¿alguna de las tres candidatas de la revisión que quedaron sin probar (Transformer pequeño,
ResNet 1D y BiGRU) supera a la TCN pequeña (brazo T de docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md)?
Informe: docs/EXPERIMENTO_ARQUITECTURAS_2.md. LOSO de 5 pliegues, particiones D-06.

Todos los brazos usan la entrada canónica (30 × 252, regla «mascara», aumentación ×10 de train_loso,
add_velocity) y la receta de entrenamiento de T (Adam con CosineDecay, label smoothing 0,1, lote 64,
paciencia 25 sobre val_accuracy, 200 épocas). Lo único que cambia es el modelo.

Brazos:
  T  TCN pequeña + GAP (control)  NO se reentrena: se reutilizan las 15 corridas de
                                  models/arquitecturas_literatura_2026-10-02/T_tcn, tras verificar su
                                  configuración y reproducir con ESTE script la semilla 2026, pliegue 0.
  X  Transformer pequeño          arquitecturas_2.construir_transformer (CONFIG_X), lr 1e-3.
  I  ResNet 1D (Wang et al.)      arquitecturas_2.construir_resnet1d (CONFIG_I), lr 1e-3.
  B  BiGRU (c07 con GRU)          arquitecturas_2.construir_bigru (CONFIG_B), lr 5e-4 (el de c07).

Nada del entrenamiento se reimplementa: se usa experimento_arquitecturas.entrenar_corrida (la misma
función que entrenó T), ampliando en memoria su función `constructor` con los brazos X, I y B. El archivo
src/experimento_arquitecturas.py no se modifica.

Subcomandos:
  verificar   configuración de las 15 corridas de T, parámetros de cada red y, con --dest, reproducción de T
              (semilla 2026, pliegue 0) con este script en una carpeta aparte.
  entrenar    combinaciones (brazo, semilla, pliegue) que falten; reanudable.
  inferencia  parámetros y latencia con lote de 1 (como translator.py), sin entrenamientos activos.
  resumen     agrega, aplica el criterio prerregistrado y escribe el JSON de datos crudos.

Uso:
  python src/experimento_arquitecturas_2.py verificar --dest <carpeta_temporal>/repro_t
  python src/experimento_arquitecturas_2.py entrenar --brazos X,I,B --semillas 2026 --folds 0, \\
         --epochs 3 --out <carpeta_temporal>/humo                         # prueba de humo
  python src/experimento_arquitecturas_2.py entrenar --brazos X,I,B --semillas 2026   # una por proceso
  python src/experimento_arquitecturas_2.py inferencia
  python src/experimento_arquitecturas_2.py resumen --docs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import train_loso as tl  # noqa: E402
from src import arquitecturas_literatura as al  # noqa: E402
from src import arquitecturas_2 as a2  # noqa: E402
from src import experimento_arquitecturas as ea  # noqa: E402

PROCESSED_CANONICO = ea.PROCESSED_CANONICO
REPRESENTACION = ea.REPRESENTACION
SEQ_LENGTH = ea.SEQ_LENGTH
DEFAULT_OUT = BASE_DIR / "models" / "arquitecturas_2_2026-10-03"
JSON_DOCS = BASE_DIR / "docs" / "EXPERIMENTO_ARQUITECTURAS_2.json"
T_DIR = BASE_DIR / "models" / "arquitecturas_literatura_2026-10-02" / "T_tcn"
SEMILLAS = [2026, 2027, 2028]
PISO_MARGEN = 0.01
TOL = 1e-9
N_PARAM_T = 156_402
N_PARAM = {"X": 187_890, "I": 315_954, "B": 341_171}
CANONICO_R0 = {"val_top1": 0.920, "test_top1": 0.589}

BRAZOS = {
    "T": {"carpeta": "T_tcn", "descripcion": "TCN pequena + GAP (control; reutiliza arquitecturas_literatura)"},
    "X": {"carpeta": "X_transformer", "descripcion": "Transformer pequeno (encoder pre-norm, 2 bloques, d 64)"},
    "I": {"carpeta": "I_resnet1d", "descripcion": "ResNet 1D de Wang et al. (3 bloques 8-5-3, 64 filtros)"},
    "B": {"carpeta": "B_bigru", "descripcion": "BiGRU: c07 con GRU en lugar de LSTM"},
}
ENTRENABLES = ("X", "I", "B")
COMPARADOS = ("T", "X", "I", "B")


# ---------------------------------------------------------------------------
# Constructor ampliado (en memoria) de experimento_arquitecturas
# ---------------------------------------------------------------------------

_CONSTRUCTOR_ORIGINAL = ea.constructor


def constructor(brazo):
    """X, I y B con su configuración y su receta; cualquier otro brazo (T) va al constructor original."""
    if brazo in ENTRENABLES:
        fn = a2.CONSTRUCTORES[brazo]
        return (lambda nf, nfe, nc: fn(nf, nfe, nc),
                dict(a2.ENTRENAMIENTO[brazo], arquitectura=a2.CONFIGS[brazo]))
    return _CONSTRUCTOR_ORIGINAL(brazo)


ea.constructor = constructor        # entrenar_corrida busca `constructor` en el módulo al llamarse
a2.capa_posicion_aprendida()        # registra la capa de X antes de recargar checkpoints


def _lista_int(texto):
    return [int(t) for t in str(texto).split(",") if t.strip() != ""]


def _lista_brazos(texto, permitidos):
    brazos = [t.strip().upper() for t in str(texto).split(",") if t.strip()]
    for b in brazos:
        if b not in permitidos:
            raise SystemExit(f"--brazos: {b} no permitido aqui ({permitidos})")
    return brazos


def dir_corrida(out, brazo, semilla, k):
    if brazo == "T":
        return T_DIR / f"semilla_{semilla}" / f"pliegue_{k}"
    return Path(out) / BRAZOS[brazo]["carpeta"] / f"semilla_{semilla}" / f"pliegue_{k}"


# ---------------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------------

def comprobar_config_t():
    """Las 15 corridas de T_tcn tienen CONFIG_T exacta, la receta de T y la entrada canónica."""
    esperado_hp = json.loads(json.dumps(dict(al.ENTRENAMIENTO_COMUN, arquitectura=al.CONFIG_T)))
    filas = []
    for s in SEMILLAS:
        for k in range(5):
            d = dir_corrida(None, "T", s, k)
            r = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
            proc = json.loads((d / "modelo.procedencia.json").read_text(encoding="utf-8"))
            assert r["brazo"] == "T" and r["semilla"] == s and r["pliegue"] == k, (s, k)
            assert r["hiperparametros"] == esperado_hp, (s, k, r["hiperparametros"])
            assert r["n_parametros"] == N_PARAM_T, (s, k)
            assert (r["epochs_max"], r["aug"], r["patience"], r["seq_length"]) == (200, 10, 25, 30), (s, k)
            assert r["processed"] == "data/processed_holistic_mascara_rostro" and r["representacion"] == "manos"
            assert r["n_train_aumentado"] % 10 == 0 and r["n_val"] == 100
            assert proc["mano_ausente"] == "mascara" and proc["n_rasgos_entrada"] == 252
            assert proc["seq_length"] == 30 and proc["representacion"] == "manos"
            filas.append({"semilla": s, "pliegue": k, "val_top1": r["val_top1"], "test_top1": r["test_top1"],
                          "epocas": r["epocas_ejecutadas"]})
    return filas


def comparar_corridas(d_ref: Path, d_new: Path) -> dict:
    """Compara dos corridas de entrenar_corrida (predicciones, top-3, épocas, validación y probabilidades)."""
    pr = json.loads((d_ref / "predicciones.json").read_text(encoding="utf-8"))
    pn = json.loads((d_new / "predicciones.json").read_text(encoding="utf-8"))
    rr = json.loads((d_ref / "resultado.json").read_text(encoding="utf-8"))
    rn = json.loads((d_new / "resultado.json").read_text(encoding="utf-8"))
    hr = json.loads((d_ref / "historial.json").read_text(encoding="utf-8"))
    hn = json.loads((d_new / "historial.json").read_text(encoding="utf-8"))
    zr, zn = np.load(d_ref / "probabilidades.npz"), np.load(d_new / "probabilidades.npz")
    res = {
        "predicciones_json_identico_byte_a_byte": (d_ref / "predicciones.json").read_bytes()
        == (d_new / "predicciones.json").read_bytes(),
        "predicha_y_top3_identicas": [(p["muestra"], p["predicha"], p["top3"]) for p in pr]
        == [(p["muestra"], p["predicha"], p["top3"]) for p in pn],
        "max_dif_probabilidades_test": float(np.max(np.abs(zr["test"] - zn["test"]))),
        "max_dif_probabilidades_val": float(np.max(np.abs(zr["val"] - zn["val"]))),
        "historial_identico": hr == hn,
        "epocas": [rr["epocas_ejecutadas"], rn["epocas_ejecutadas"]],
        "val_top1": [rr["val_top1"], rn["val_top1"]],
        "test_top1": [rr["test_top1"], rn["test_top1"]],
        "test_f1_macro": [rr["test_f1_macro"], rn["test_f1_macro"]],
    }
    res["coincide"] = bool(res["predicha_y_top3_identicas"] and res["epocas"][0] == res["epocas"][1]
                           and abs(res["val_top1"][0] - res["val_top1"][1]) < 1e-12
                           and res["max_dif_probabilidades_test"] < 1e-5)
    return res


def cmd_verificar(args):
    informe = {"fecha": datetime.now().isoformat(timespec="seconds"), "comprobaciones": []}

    def ok(texto):
        print(f"[ok] {texto}", flush=True)
        informe["comprobaciones"].append(texto)

    informe["t"] = comprobar_config_t()
    ok(f"T: las 15 corridas de T_tcn tienen CONFIG_T exacta ({al.CONFIG_T}), lr 1e-3, lote 64, label "
       f"smoothing 0,1, 200 epocas, paciencia 25, aug x10, 30 pasos, processed_holistic_mascara_rostro, manos, "
       f"mascara, 252 rasgos y {N_PARAM_T:,} parametros")

    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    tf = canon[0]
    n_par = {}
    for b in COMPARADOS:
        fn, hp = constructor(b)
        tf.keras.utils.set_random_seed(0)
        n_par[b] = int(fn(SEQ_LENGTH, 252, 50).count_params())
        tf.keras.backend.clear_session()
    assert n_par["T"] == N_PARAM_T and all(n_par[b] == N_PARAM[b] for b in ENTRENABLES), n_par
    assert all(150_000 <= v <= 400_000 for v in n_par.values()), n_par
    ok(f"parametros (todos entre 150 y 400 mil): {n_par}")
    informe["n_parametros"] = n_par
    informe["configs"] = {"T": al.CONFIG_T, **a2.CONFIGS}
    informe["entrenamiento"] = {"T": al.ENTRENAMIENTO_COMUN, **a2.ENTRENAMIENTO}

    if args.dest:
        X_seq, y, nombres, senantes, loso, pos, canon = ea.preparar()
        dest = Path(args.dest).resolve()
        d = dest / "T_repro" / "semilla_2026" / "pliegue_0"
        if not (d / "resultado.json").exists():
            ns = argparse.Namespace(aug=10, epochs=200, patience=25, processed=str(PROCESSED_CANONICO))
            t0 = time.perf_counter()
            ea.entrenar_corrida("T", 2026, 0, loso["pliegues"][0], pos, X_seq, y, canon, d, ns)
            print(f"  reproduccion en {time.perf_counter() - t0:.0f} s", flush=True)
        comp = comparar_corridas(dir_corrida(None, "T", 2026, 0), d)
        informe["reproduccion_T_s2026_k0"] = comp
        print(json.dumps(comp, indent=1), flush=True)
        if not comp["coincide"]:
            print("[FALLA] la reproduccion de T no coincide", flush=True)
        else:
            ok("T semilla 2026 pliegue 0 reproducido con este script (entrenar_corrida con el constructor "
               f"ampliado): predicciones y top-3 identicos, mismas epocas y validacion; max |dif| de "
               f"probabilidades de prueba {comp['max_dif_probabilidades_test']:.2e}; predicciones.json "
               f"identico byte a byte: {comp['predicciones_json_identico_byte_a_byte']}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "verificacion.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {tl._rel(out / 'verificacion.json')}")


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def anotar_corrida(d: Path, brazo: str):
    """Corrige la procedencia y el resultado que escribe entrenar_corrida (pensados para
    experimento_arquitecturas) con el brazo y el script reales."""
    from src.representacion import ruta_procedencia
    rp = ruta_procedencia(d / "modelo.keras")
    proc = json.loads(rp.read_text(encoding="utf-8"))
    proc["script"] = "src/experimento_arquitecturas_2.py (via experimento_arquitecturas.entrenar_corrida)"
    proc.pop("brazo_experimento_arquitecturas", None)
    proc["brazo_arquitecturas_2"] = brazo
    proc["nota"] = ("modelo del experimento de arquitecturas 2 (docs/EXPERIMENTO_ARQUITECTURAS_2.md). NO es el "
                    "modelo canonico ni debe cargarse en el prototipo."
                    + (" Para cargarlo hay que importar antes src.arquitecturas_2 y llamar "
                       "capa_posicion_aprendida() (capa PosicionAprendida)." if brazo == "X" else ""))
    rp.write_text(json.dumps(proc, indent=1, ensure_ascii=False), encoding="utf-8")
    rr = d / "resultado.json"
    r = json.loads(rr.read_text(encoding="utf-8"))
    r["constructor"] = {"X": "arquitecturas_2.construir_transformer", "I": "arquitecturas_2.construir_resnet1d",
                        "B": "arquitecturas_2.construir_bigru"}[brazo]
    r["script"] = "src/experimento_arquitecturas_2.py"
    rr.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


def cmd_entrenar(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.processed = str(Path(args.processed).resolve())
    X_seq, y, nombres, senantes, loso, pos, canon = ea.preparar(Path(args.processed))
    brazos = _lista_brazos(args.brazos, ENTRENABLES)
    semillas = _lista_int(args.semillas)
    folds = tl.parsear_folds(args.folds, len(loso["pliegues"]))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] experimento_arquitecturas_2 entrenar | brazos "
          f"{brazos} | semillas {semillas} | pliegues {folds} | epochs {args.epochs} | aug x{args.aug} | "
          f"paciencia {args.patience} | salida {tl._rel(out)}", flush=True)
    for b in brazos:
        for s in semillas:
            for k in folds:
                d = Path(out) / BRAZOS[b]["carpeta"] / f"semilla_{s}" / f"pliegue_{k}"
                if (d / "resultado.json").exists():
                    print(f"  ya existe {tl._rel(d)}; se salta", flush=True)
                    continue
                p = loso["pliegues"][k]
                print(f"\n{'=' * 72}\n  brazo {b} ({BRAZOS[b]['descripcion']}) | semilla {s} | pliegue {k} "
                      f"(senante {p['senante_test']}) | {datetime.now().isoformat(timespec='seconds')}\n{'=' * 72}",
                      flush=True)
                ea.entrenar_corrida(b, s, k, p, pos, X_seq, y, canon, d, args)
                anotar_corrida(d, b)
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] fin", flush=True)


# ---------------------------------------------------------------------------
# Inferencia (lote de 1, como translator.py)
# ---------------------------------------------------------------------------

def otros_entrenamientos():
    """Procesos de Python de entrenamiento activos (cualquier experimento_*, train_* o la búsqueda)."""
    r = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True)
    patron = re.compile(r"(experimento_[a-z0-9_]+|train_[a-z0-9_]+|busqueda_hiperparametros)\.py")
    activos = []
    for linea in r.stdout.splitlines():
        partes = linea.strip().split(None, 2)
        if len(partes) < 2 or int(partes[0]) == os.getpid():
            continue
        if "ython" in partes[1].split("/")[-1] and patron.search(linea) and " inferencia" not in linea:
            activos.append(linea.strip()[:200])
    return activos


def cmd_inferencia(args):
    out = Path(args.out).resolve()
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    tf = canon[0]
    otros = otros_entrenamientos()
    if otros:
        raise SystemExit(f"hay entrenamientos activos; la latencia se mide sin ellos: {otros}")
    rng = np.random.default_rng(0)
    filas = []
    for b in ("R0",) + COMPARADOS:
        base = ea.R0_DIR if b == "R0" else (T_DIR if b == "T" else out / BRAZOS[b]["carpeta"])
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
    otros_fin = otros_entrenamientos()
    with open(out / "inferencia_redes.json", "w", encoding="utf-8") as f:
        json.dump({"fecha": datetime.now().isoformat(timespec="seconds"),
                   "metodo": f"model.predict(verbose=0) con lote de 1, como translator.py, y llamada directa "
                             f"model(x, training=False); {args.calentamiento} llamadas de calentamiento y "
                             f"{args.repeticiones} medidas por modelo; mediana y p90. Solo la red (sin MediaPipe "
                             f"ni remuestreo). CPU. Modelos de la semilla 2026.",
                   "procesos_de_entrenamiento_activos_al_inicio": otros,
                   "procesos_de_entrenamiento_activos_al_final": otros_fin,
                   "entorno": tl.info_entorno(), "modelos": filas}, f, indent=1, ensure_ascii=False)
    print(f"escrito {tl._rel(out / 'inferencia_redes.json')}")


# ---------------------------------------------------------------------------
# Resumen y criterio
# ---------------------------------------------------------------------------

def _md(v):
    v = [float(x) for x in v]
    return {"media": float(np.mean(v)), "de": float(np.std(v, ddof=1)) if len(v) > 1 else None, "valores": v}


def leer_corridas(out):
    filas = []
    for s in SEMILLAS:
        for k in range(5):
            m, pred, d = ea.leer_r0(s, k)
            filas.append({"brazo": "R0", "semilla": s, "pliegue": k, "senante_test": m["senante_test"],
                          "val_top1": m["mejor_val_accuracy"], "test_top1": m["top1"], "test_top3": m["top3"],
                          "test_f1_macro": m["f1_macro"], "n_parametros": m["n_parametros"],
                          "epocas_ejecutadas": m["epocas_ejecutadas"], "tiempo_total_s": m["tiempo_total_s"],
                          "tiempo_entrenamiento_s": m["tiempo_entrenamiento_s"], "s_por_epoca": None,
                          "predicciones": {x["muestra"]: (x["real"], x["predicha"]) for x in pred},
                          "confianza": {x["muestra"]: x["prob_predicha"] for x in pred},
                          "fuente": tl._rel(d)})
    for b in COMPARADOS:
        base = T_DIR if b == "T" else Path(out) / BRAZOS[b]["carpeta"]
        for f in sorted(base.glob("semilla_*/pliegue_*/resultado.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            pred = json.loads((f.parent / "predicciones.json").read_text(encoding="utf-8"))
            filas.append({"brazo": b, "semilla": r["semilla"], "pliegue": r["pliegue"],
                          "senante_test": r["senante_test"], "val_top1": r["val_top1"],
                          "test_top1": r["test_top1"], "test_top3": r["test_top3"],
                          "test_f1_macro": r["test_f1_macro"], "n_parametros": r["n_parametros"],
                          "epocas_ejecutadas": r["epocas_ejecutadas"], "mejor_epoca": r.get("mejor_epoca"),
                          "tiempo_total_s": r["tiempo_total_s"],
                          "tiempo_entrenamiento_s": r.get("tiempo_entrenamiento_s"),
                          "s_por_epoca": r.get("s_por_epoca"),
                          "val_top1_checkpoint_reevaluado": r.get("val_top1_checkpoint_reevaluado"),
                          "predicciones": {x["muestra"]: (x["real"], x["predicha"]) for x in pred},
                          "confianza": {x["muestra"]: x["prob_predicha"] for x in pred},
                          "fuente": tl._rel(f.parent)})
    return filas


def mcnemar(pa, pb, nombre_a="brazo", nombre_b="T"):
    """McNemar exacto (binomial bilateral sobre los discordantes). pa, pb: {muestra: (real, predicha)}."""
    from scipy.stats import binomtest
    claves = sorted(pa)
    assert set(claves) == set(pb)
    ca = np.array([pa[c][0] == pa[c][1] for c in claves])
    cb = np.array([pb[c][0] == pb[c][1] for c in claves])
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    return {"n": len(claves), f"aciertos_{nombre_a}": int(ca.sum()), f"aciertos_{nombre_b}": int(cb.sum()),
            f"solo_{nombre_a}_acierta": b, f"solo_{nombre_b}_acierta": c,
            "ambos_aciertan": int(np.sum(ca & cb)), "ninguno": int(np.sum(~ca & ~cb)),
            "p_exacto_bilateral": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}


def voto_3_semillas(preds, confianzas):
    """Predicción por voto de mayoría de las 3 semillas. preds: lista de 3 arreglos de clases (n,);
    confianzas: lista de 3 arreglos (n,) con la probabilidad de la clase predicha. Si las tres semillas
    difieren, decide la semilla más confiada (la de mayor probabilidad predicha). Se usa la misma regla en
    todos los brazos porque R0 solo guarda la probabilidad de la clase predicha."""
    P = np.stack(preds, axis=0)
    C = np.stack(confianzas, axis=0)
    n = P.shape[1]
    salida = np.empty(n, dtype=np.int64)
    sin_mayoria = 0
    for i in range(n):
        vals, cnt = np.unique(P[:, i], return_counts=True)
        if cnt.max() >= 2:
            salida[i] = vals[np.argmax(cnt)]
        else:
            sin_mayoria += 1
            salida[i] = P[int(np.argmax(C[:, i])), i]
    return salida, sin_mayoria


def predicciones_voto(filas, brazo, celda):
    """{muestra: (real, predicha por voto)} sobre los 5 pliegues, y número de muestras sin mayoría."""
    out, sin_m = {}, 0
    for k in range(5):
        fs = [celda(brazo, s, k) for s in SEMILLAS]
        claves = sorted(fs[0]["predicciones"])
        assert all(sorted(f["predicciones"]) == claves for f in fs)
        preds = [np.array([f["predicciones"][c][1] for c in claves]) for f in fs]
        confs = [np.array([f["confianza"][c] for c in claves]) for f in fs]
        v, sm = voto_3_semillas(preds, confs)
        sin_m += sm
        out.update({c: (fs[0]["predicciones"][c][0], int(v[i])) for i, c in enumerate(claves)})
    return out, sin_m


def cmd_resumen(args):
    out = Path(args.out).resolve()
    filas = leer_corridas(out)
    folds = list(range(5))

    def celda(b, s, k):
        r = [f for f in filas if f["brazo"] == b and f["semilla"] == s and f["pliegue"] == k]
        return r[0] if r else None

    brazos = [b for b in ("R0",) + COMPARADOS if all(celda(b, s, k) for s in SEMILLAS for k in folds)]
    faltan = [b for b in ("R0",) + COMPARADOS if b not in brazos]
    if faltan:
        print(f"AVISO: brazos incompletos (se omiten): {faltan}")
    METR = ["val_top1", "test_top1", "test_top3", "test_f1_macro", "epocas_ejecutadas", "tiempo_total_s",
            "tiempo_entrenamiento_s", "s_por_epoca"]
    por_brazo = {}
    for b in brazos:
        r = {"n_parametros": celda(b, SEMILLAS[0], 0)["n_parametros"]}
        for m in METR:
            pps = {s: [celda(b, s, k)[m] for k in folds] for s in SEMILLAS}
            if any(v is None for vs in pps.values() for v in vs):
                continue
            por_pliegue = [float(np.mean([pps[s][k] for s in SEMILLAS])) for k in folds]
            por_sem = {s: float(np.mean(pps[s])) for s in SEMILLAS}
            r[m] = {"por_semilla_por_pliegue": {str(s): v for s, v in pps.items()},
                    "por_pliegue_media_semillas": por_pliegue,
                    "por_semilla_media_pliegues": {str(s): v for s, v in por_sem.items()},
                    "entre_pliegues": _md(por_pliegue),
                    "de_entre_semillas": float(np.std(list(por_sem.values()), ddof=1)),
                    "media_global": float(np.mean(list(por_sem.values())))}
        por_brazo[b] = r

    # Criterio prerregistrado (docs/EXPERIMENTO_ARQUITECTURAS_2.md §2): frente a T.
    agrupados = [b for b in COMPARADOS if b in brazos]
    var = {b: por_brazo[b]["val_top1"]["de_entre_semillas"] ** 2 for b in agrupados}
    sigma = float(np.sqrt(np.mean(list(var.values()))))
    margen = max(sigma, PISO_MARGEN)
    vT = np.array(por_brazo["T"]["val_top1"]["por_pliegue_media_semillas"])
    detalle, cumplen = {}, []
    for b in [x for x in ENTRENABLES if x in brazos]:
        vb = np.array(por_brazo[b]["val_top1"]["por_pliegue_media_semillas"])
        dif = vb - vT
        pp = ea.pruebas_pareadas(dif)
        delta = float(np.mean(np.round(dif, 6)))
        c1, c2 = bool(delta > margen + TOL), bool(pp["mejora"] >= 4)
        detalle[b] = {"delta_val_por_pliegue": dif.tolist(), "delta_val_media": delta,
                      "pliegues_mejor": pp["mejora"], "pliegues_peor": pp["empeora"], "pliegues_igual": pp["empate"],
                      "condicion_1_delta_mayor_que_margen": c1, "condicion_2_mejor_en_4_de_5": c2,
                      "supera_a_T": bool(c1 and c2),
                      "peor_que_T_segun_regla_simetrica": bool(-delta > margen + TOL and pp["empeora"] >= 4),
                      "pruebas_validacion": pp}
        for m in ("test_top1", "test_top3", "test_f1_macro"):
            d_m = (np.array(por_brazo[b][m]["por_pliegue_media_semillas"])
                   - np.array(por_brazo["T"][m]["por_pliegue_media_semillas"]))
            detalle[b]["pruebas_" + m + "_descriptivas"] = ea.pruebas_pareadas(d_m)
        if c1 and c2:
            cumplen.append(b)
    if cumplen:
        mejor = max(por_brazo[b]["val_top1"]["media_global"] for b in cumplen)
        empatados = [b for b in cumplen if mejor - por_brazo[b]["val_top1"]["media_global"] < margen]
        ganador = min(empatados, key=lambda b: por_brazo[b]["n_parametros"])
    else:
        empatados, ganador = [], "T"
    criterio = {"brazos_del_sigma": agrupados, "sigma_semilla_agrupada": sigma,
                "varianza_entre_semillas_por_brazo": var, "margen": margen, "detalle": detalle,
                "brazos_que_superan_a_T": cumplen, "empatados_dentro_del_margen": empatados, "ganador": ganador}

    # Comparación descriptiva con el canónico R0 (misma regla, aplicada frente a R0).
    frente_r0 = {}
    v0 = np.array(por_brazo["R0"]["val_top1"]["por_pliegue_media_semillas"])
    for b in [x for x in COMPARADOS if x in brazos]:
        dif = np.array(por_brazo[b]["val_top1"]["por_pliegue_media_semillas"]) - v0
        pp = ea.pruebas_pareadas(dif)
        dt = (np.array(por_brazo[b]["test_top1"]["por_pliegue_media_semillas"])
              - np.array(por_brazo["R0"]["test_top1"]["por_pliegue_media_semillas"]))
        frente_r0[b] = {"delta_val_media": float(np.mean(np.round(dif, 6))), "delta_val_por_pliegue": dif.tolist(),
                        "pliegues_mejor_val": pp["mejora"], "pruebas_validacion": pp,
                        "delta_test_top1_media": float(np.mean(dt)), "pruebas_test_top1": ea.pruebas_pareadas(dt),
                        "cumpliria_criterio_frente_a_R0": bool(np.mean(np.round(dif, 6)) > margen + TOL
                                                               and pp["mejora"] >= 4)}

    # McNemar frente a T: semilla 2026 (principal), 2027 y 2028, y voto de las 3 semillas.
    mc = {}
    pT_voto, smT = predicciones_voto(filas, "T", celda)
    votos_sin_mayoria = {"T": smT}
    for b in [x for x in ENTRENABLES + ("R0",) if x in brazos]:
        mc[b] = {}
        for s in SEMILLAS:
            pa, pt = {}, {}
            for k in folds:
                pa.update(celda(b, s, k)["predicciones"])
                pt.update(celda("T", s, k)["predicciones"])
            mc[b][str(s)] = mcnemar(pa, pt)
        pv, sm = predicciones_voto(filas, b, celda)
        votos_sin_mayoria[b] = sm
        mc[b]["voto_3_semillas"] = mcnemar(pv, pT_voto)
    exactitud_voto = {}
    for b in brazos:
        pv, _ = predicciones_voto(filas, b, celda)
        exactitud_voto[b] = float(np.mean([r == p for r, p in pv.values()]))

    p0000 = {b: {m: {"media": por_brazo[b][m]["por_pliegue_media_semillas"][0],
                     "por_semilla": {s: v[0] for s, v in por_brazo[b][m]["por_semilla_por_pliegue"].items()}}
                 for m in ("val_top1", "test_top1", "test_top3", "test_f1_macro")} for b in brazos}

    extra = {}
    for nombre in ("verificacion.json", "inferencia_redes.json", "humo.json"):
        if (out / nombre).exists():
            extra[nombre] = json.loads((out / nombre).read_text(encoding="utf-8"))

    sin_pred = [{k: v for k, v in f.items() if k not in ("predicciones", "confianza")} for f in filas]
    resultado = {"experimento": "arquitecturas 2 (docs/EXPERIMENTO_ARQUITECTURAS_2.md)",
                 "fecha_resumen": datetime.now().isoformat(timespec="seconds"),
                 "brazos": {b: BRAZOS.get(b, {"descripcion": "canonico CNN+BiLSTM+atencion (referencia)"})
                            for b in brazos},
                 "configs": {"T": al.CONFIG_T, **a2.CONFIGS},
                 "entrenamiento": {"T": al.ENTRENAMIENTO_COMUN, **a2.ENTRENAMIENTO},
                 "criterio_frente_a_T": criterio, "comparacion_descriptiva_frente_a_R0": frente_r0,
                 "mcnemar_frente_a_T": mc, "voto_muestras_sin_mayoria": votos_sin_mayoria,
                 "exactitud_prueba_voto_3_semillas": exactitud_voto, "persona_0000": p0000,
                 "por_brazo": por_brazo, "verificacion": extra.get("verificacion.json"),
                 "humo": extra.get("humo.json"), "inferencia_redes": extra.get("inferencia_redes.json"),
                 "corridas": sin_pred}
    (out / "resultados_arquitecturas_2.json").write_text(json.dumps(resultado, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    if args.docs:
        JSON_DOCS.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"escrito {tl._rel(JSON_DOCS)}")

    print(f"brazos {brazos}")
    print(f"sigma semilla agrupada ({agrupados}) {sigma:.4f} | margen {margen:.4f}")
    for b in brazos:
        r = por_brazo[b]
        print(f"\n{b} | {r['n_parametros']:,} param")
        for m in ("val_top1", "test_top1", "test_top3", "test_f1_macro"):
            print(f"   {m:14s} {r[m]['media_global']:.4f} (DE pliegues {r[m]['entre_pliegues']['de']:.4f}, DE "
                  f"semillas {r[m]['de_entre_semillas']:.4f}) por pliegue "
                  f"{[round(x, 3) for x in r[m]['por_pliegue_media_semillas']]}")
        print(f"   val por semilla {r['val_top1']['por_semilla_media_pliegues']}")
        print(f"   test por semilla {r['test_top1']['por_semilla_media_pliegues']}")
        print(f"   tiempo por pliegue {r['tiempo_total_s']['media_global']:.0f} s | epocas "
              f"{r['epocas_ejecutadas']['media_global']:.1f}"
              + (f" | s/epoca {r['s_por_epoca']['media_global']:.2f}" if r.get("s_por_epoca") else ""))
    for b, d in detalle.items():
        print(f"\n{b} - T: dval {[round(x, 4) for x in d['delta_val_por_pliegue']]} media {d['delta_val_media']:+.4f}"
              f" | mejor {d['pliegues_mejor']}/5 | signos p {d['pruebas_validacion']['signos_p_bilateral']:.4f} | "
              f"wilcoxon p {d['pruebas_validacion']['wilcoxon_p_bilateral']:.4f} | supera {d['supera_a_T']}")
        for m in ("test_top1", "test_f1_macro"):
            pr = d["pruebas_" + m + "_descriptivas"]
            print(f"   {m}: {[round(x, 3) for x in pr['diferencias']]} | mejor {pr['mejora']}/5 | signos p "
                  f"{pr['signos_p_bilateral']:.4f} | wilcoxon p {pr['wilcoxon_p_bilateral']:.4f}")
    for b, d in mc.items():
        for s, m in d.items():
            print(f"   McNemar {b} vs T [{s}]: {b} {m['aciertos_brazo']} vs T {m['aciertos_T']} | solo {b} "
                  f"{m['solo_brazo_acierta']} solo T {m['solo_T_acierta']} | p {m['p_exacto_bilateral']:.4g}")
    print("\nfrente a R0:", {b: (round(v["delta_val_media"], 4), v["pliegues_mejor_val"],
                               round(v["delta_test_top1_media"], 4)) for b, v in frente_r0.items()})
    print(f"\nGANADOR: {ganador} (superan a T: {cumplen})")
    print("persona 0000:", {b: (round(p0000[b]['val_top1']['media'], 3), round(p0000[b]['test_top1']['media'], 3))
                            for b in brazos})
    print("exactitud voto:", exactitud_voto, "| sin mayoria:", votos_sin_mayoria)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verificar")
    v.add_argument("--dest", default=None, help="carpeta (carpeta_temporal) para reproducir T s2026 k0")
    e = sub.add_parser("entrenar")
    e.add_argument("--brazos", default="X,I,B")
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
    {"verificar": cmd_verificar, "entrenar": cmd_entrenar, "inferencia": cmd_inferencia,
     "resumen": cmd_resumen}[args.cmd](args)


if __name__ == "__main__":
    main()
