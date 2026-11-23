"""
Diseño factorial con la TCN: {mascara, pivote, imputación} × {252, 132} (docs/EXPERIMENTO_FACTORIAL_TCN.md).

La confirmación conjunta (docs/EXPERIMENTO_CONFIRMACION_TCN.md) midió tres de las seis celdas y encontró la
exactitud de validación SATURADA (~0,95 con 100 muestras por pliegue). Aquí se completan las tres celdas que
faltan, siempre con la TCN (CONFIG_T exacta) y su receta, y se decide con un criterio de validación menos
saturado (top-1 -> val_loss -> simplicidad).

  T0  mascara    + 252   REUTILIZADO: models/arquitecturas_literatura_2026-10-02/T_tcn
  T1  imputacion + 252   REUTILIZADO: models/confirmacion_tcn_2026-10-03/T1_imputacion_252
  T2  imputacion + 132   REUTILIZADO: models/confirmacion_tcn_2026-10-03/T2_imputacion_132
  T3  mascara    + 132   NUEVO
  T4  pivote     + 252   NUEVO
  T5  pivote     + 132   NUEVO

«252» = add_velocity (126 posiciones + 126 velocidades); «132» = velocidades_reducidas «por_coordenada»
(126 posiciones + 6 velocidades), calculadas DESPUÉS de la aumentación y del remuestreo. «pivote» e
«imputacion» = mano_ausente_variantes.remuestrear_todas_con_regla sobre el processed «mascara», antes de
remuestrear; «mascara» = train_loso.remuestrear_todas (canónico).

Nada se reimplementa: el entrenamiento es experimento_arquitecturas.entrenar_corrida (constructor «T»); la
comparación de corridas, la σ agrupada, las Δ por pliegue, la regla de mejora y el voto de semillas son los de
experimento_confirmacion_tcn; McNemar y las pruebas pareadas, los de experimento_arquitecturas.

Subcomandos:
  fijar_val_loss  extrae y fija la val_loss de T0, T1 y T2 (antes de ver los brazos nuevos).
  verificar       configuración de T0-T2, entradas de los 6 brazos, parámetros y reproducción byte a byte de
                  T1 (semilla 2026, pliegue 0) en una carpeta aparte.
  entrenar        combinaciones (brazo, semilla, pliegue) de T3-T5 que falten; reanudable.
  inferencia      latencia con lote de 1 (red) y costo de armar la entrada, sin entrenamientos activos.
  resumen         agrega, aplica el criterio prerregistrado y escribe el JSON de datos crudos.

Uso:
  python src/experimento_factorial_tcn.py fijar_val_loss
  python src/experimento_factorial_tcn.py verificar --dest <carpeta_temporal>/repro_t1
  python src/experimento_factorial_tcn.py entrenar --semillas 2026 --folds 0, --epochs 3 --out <scratch>/humo
  python src/experimento_factorial_tcn.py entrenar --semillas 2026        # una semilla por proceso
  python src/experimento_factorial_tcn.py inferencia
  python src/experimento_factorial_tcn.py resumen --docs
"""

from __future__ import annotations

import argparse
import itertools
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
from src import arquitecturas_literatura as al  # noqa: E402
from src import experimento_arquitecturas as ea  # noqa: E402
from src import experimento_confirmacion_tcn as ect  # noqa: E402
from src import mano_ausente_variantes as mav  # noqa: E402
from src import velocidades_reducidas as vr  # noqa: E402

PROCESSED_CANONICO = ea.PROCESSED_CANONICO
REPRESENTACION = ea.REPRESENTACION
SEQ_LENGTH = ea.SEQ_LENGTH
DEFAULT_OUT = BASE_DIR / "models" / "factorial_tcn_2026-10-03"
JSON_DOCS = BASE_DIR / "docs" / "EXPERIMENTO_FACTORIAL_TCN.json"
CONFIRMACION_DIR = ect.DEFAULT_OUT
SEMILLAS = ect.SEMILLAS
PISO_MARGEN = ect.PISO_MARGEN
TOL = ect.TOL
LABEL_SMOOTHING = al.ENTRENAMIENTO_COMUN["label_smoothing"]
ARCHIVO_VAL_LOSS_FIJADA = "val_loss_T0_T1_T2_fijada.json"

# Orden de simplicidad (paso 3 del criterio): preprocesamiento primero.
ORDEN_PREPROC = {"mascara": 0, "imputacion": 1, "pivote": 2}
UMBRAL_COSTO_MS = 1.0
UMBRAL_COSTO_MB = 0.5

BRAZOS = {
    "T0": {"carpeta": "T0_mascara_252", "regla": "mascara", "forma": "actual", "n_rasgos": 252,
           "origen": "reutilizado", "descripcion": "TCN + mascara + 252 (T_tcn de arquitecturas_literatura)"},
    "T1": {"carpeta": "T1_imputacion_252", "regla": "imputacion", "forma": "actual", "n_rasgos": 252,
           "origen": "reutilizado", "descripcion": "TCN + imputacion + 252 (confirmacion_tcn)"},
    "T2": {"carpeta": "T2_imputacion_132", "regla": "imputacion", "forma": "por_coordenada", "n_rasgos": 132,
           "origen": "reutilizado", "descripcion": "TCN + imputacion + 132 (confirmacion_tcn)"},
    "T3": {"carpeta": "T3_mascara_132", "regla": "mascara", "forma": "por_coordenada", "n_rasgos": 132,
           "origen": "nuevo", "descripcion": "TCN + mascara + 132 (6 velocidades por coordenada)"},
    "T4": {"carpeta": "T4_pivote_252", "regla": "pivote", "forma": "actual", "n_rasgos": 252,
           "origen": "nuevo", "descripcion": "TCN + pivote + 252"},
    "T5": {"carpeta": "T5_pivote_132", "regla": "pivote", "forma": "por_coordenada", "n_rasgos": 132,
           "origen": "nuevo", "descripcion": "TCN + pivote + 132 (6 velocidades por coordenada)"},
}
TODOS = tuple(BRAZOS)
NUEVOS = ("T3", "T4", "T5")
REUTILIZADOS = ("T0", "T1", "T2")
PAR_VELOCIDADES = {"mascara": ("T0", "T3"), "imputacion": ("T1", "T2"), "pivote": ("T4", "T5")}
N_PARAM = {252: 156_402, 132: 125_682}


def _lista_int(texto):
    return [int(t) for t in str(texto).split(",") if t.strip() != ""]


def _lista_brazos(texto, permitidos):
    brazos = [t.strip().upper() for t in str(texto).split(",") if t.strip()]
    for b in brazos:
        if b not in permitidos:
            raise SystemExit(f"--brazos: {b} no permitido aqui ({permitidos})")
    return brazos


def dir_corrida(out, brazo, semilla, k):
    if brazo == "T0":
        return ect.T0_DIR / f"semilla_{semilla}" / f"pliegue_{k}"
    if brazo in ("T1", "T2"):
        return CONFIRMACION_DIR / BRAZOS[brazo]["carpeta"] / f"semilla_{semilla}" / f"pliegue_{k}"
    return Path(out) / BRAZOS[brazo]["carpeta"] / f"semilla_{semilla}" / f"pliegue_{k}"


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def preparar(processed=PROCESSED_CANONICO):
    """X_seq por regla: «mascara» = train_loso.remuestrear_todas (como experimento_arquitecturas);
    «imputacion» y «pivote» = mano_ausente_variantes.remuestrear_todas_con_regla."""
    X_padded, y, lengths, nombres, senantes, loso, pos = ea.cargar_todo(processed)
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    X_seq = {"mascara": tl.remuestrear_todas(X_padded, lengths, canon[3], SEQ_LENGTH)}
    for regla in ("imputacion", "pivote"):
        X_seq[regla] = mav.remuestrear_todas_con_regla(X_padded, lengths, regla, SEQ_LENGTH)
    return X_seq, y, nombres, senantes, loso, pos, canon, (X_padded, lengths)


def funcion_entrada_brazo(brazo, add_velocity):
    forma = BRAZOS[brazo]["forma"]
    return add_velocity if forma == "actual" else vr.funcion_entrada(forma)


def canon_de_brazo(canon, brazo):
    """Tupla canon con canon[2] = función de entrada del brazo (se aplica DESPUÉS de aumentar)."""
    tf, build_model, add_velocity, resample_sequence, transformaciones = canon
    return (tf, build_model, funcion_entrada_brazo(brazo, add_velocity), resample_sequence, transformaciones)


# ---------------------------------------------------------------------------
# val_loss fijada de los brazos reutilizados
# ---------------------------------------------------------------------------

def entropia_cruzada(probs, y_real, n_clases=50, suavizado=0.0):
    """Entropía cruzada media (la de Keras, con label smoothing opcional). Para verificar val_loss y para la
    NLL sin suavizado (descriptiva)."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1 - 1e-7)
    y_oh = np.eye(n_clases)[np.asarray(y_real, dtype=np.int64)]
    y_oh = y_oh * (1 - suavizado) + suavizado / n_clases
    return float(np.mean(-np.sum(y_oh * np.log(p), axis=1)))


def _y_por_nombre():
    X_padded, y, lengths, nombres, _ = tl.cargar_datos(PROCESSED_CANONICO, REPRESENTACION)
    return {n: int(y[i]) for i, n in enumerate(nombres)}


def perdidas_corrida(d: Path, y_por_nombre: dict) -> dict:
    """val_loss guardada (Keras, checkpoint recargado, con label smoothing 0,1) y su recálculo desde
    probabilidades.npz; más la NLL sin suavizado (descriptiva)."""
    r = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
    z = np.load(d / "probabilidades.npz")
    yv = np.array([y_por_nombre[str(n)] for n in z["nombres_val"]])
    return {"val_loss": float(r["val_loss"]),
            "val_loss_recalculada_npz": entropia_cruzada(z["val"], yv, suavizado=LABEL_SMOOTHING),
            "val_nll_sin_suavizado": entropia_cruzada(z["val"], yv, suavizado=0.0)}


def cmd_fijar_val_loss(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    destino = out / ARCHIVO_VAL_LOSS_FIJADA
    if destino.exists() and not args.forzar:
        raise SystemExit(f"{tl._rel(destino)} ya existe (fijado); no se sobrescribe")
    ypn = _y_por_nombre()
    filas = []
    for b in REUTILIZADOS:
        for s in SEMILLAS:
            for k in range(5):
                d = dir_corrida(out, b, s, k)
                pr = perdidas_corrida(d, ypn)
                assert abs(pr["val_loss"] - pr["val_loss_recalculada_npz"]) < 1e-4, (b, s, k, pr)
                filas.append({"brazo": b, "semilla": s, "pliegue": k, **pr, "fuente": tl._rel(d)})
    resumen = {}
    for b in REUTILIZADOS:
        M = np.array([[f["val_loss"] for f in filas if f["brazo"] == b and f["semilla"] == s]
                      for s in SEMILLAS])
        resumen[b] = {"media_global": float(M.mean()), "por_pliegue_media_semillas": M.mean(axis=0).tolist(),
                      "por_semilla_media_pliegues": {str(s): float(M[i].mean()) for i, s in enumerate(SEMILLAS)},
                      "de_entre_semillas": float(np.std(M.mean(axis=1), ddof=1))}
    res = {"fecha": datetime.now().isoformat(timespec="seconds"),
           "nota": "val_loss de T0, T1 y T2 fijada ANTES de ver cualquier resultado de T3, T4 y T5. Es la que "
                   "guarda experimento_arquitecturas.entrenar_corrida: model.evaluate del checkpoint recargado "
                   "(elegido por val_accuracy) sobre la validacion del pliegue; CategoricalCrossentropy con label "
                   "smoothing 0,1. Se comprobo que coincide (|dif| < 1e-4) con el recalculo desde "
                   "probabilidades.npz.",
           "resumen": resumen, "corridas": filas}
    destino.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for b, r in resumen.items():
        print(f"{b}: val_loss {r['media_global']:.4f} | por pliegue "
              f"{[round(x, 4) for x in r['por_pliegue_media_semillas']]} | DE sem {r['de_entre_semillas']:.4f}")
    print(f"escrito {tl._rel(destino)}")


# ---------------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------------

def comprobar_config_reutilizados():
    """T0 con experimento_confirmacion_tcn.comprobar_config_t0; T1 y T2: misma receta, regla y entrada."""
    filas = {"T0": ect.comprobar_config_t0()}
    esperado_hp = json.loads(json.dumps(ect.hiperparametros_t()))
    for b in ("T1", "T2"):
        filas[b] = []
        for s in SEMILLAS:
            for k in range(5):
                d = dir_corrida(None, b, s, k)
                r = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
                assert r["brazo"] == b and r["semilla"] == s and r["pliegue"] == k, (b, s, k)
                assert r["hiperparametros"] == esperado_hp, (b, s, k)
                assert r["n_parametros"] == N_PARAM[BRAZOS[b]["n_rasgos"]], (b, s, k)
                assert (r["epochs_max"], r["aug"], r["patience"], r["seq_length"]) == (200, 10, 25, 30), (b, s, k)
                assert r["processed"] == "data/processed_holistic_mascara_rostro" and r["representacion"] == "manos"
                assert r["regla_mano_ausente"] == BRAZOS[b]["regla"] and r["forma_velocidades"] == BRAZOS[b]["forma"]
                assert r["n_rasgos_entrada"] == BRAZOS[b]["n_rasgos"] and r["n_val"] == 100
                filas[b].append({"semilla": s, "pliegue": k, "val_top1": r["val_top1"], "val_loss": r["val_loss"]})
    return filas


def cmd_verificar(args):
    informe = {"fecha": datetime.now().isoformat(timespec="seconds"), "comprobaciones": []}

    def ok(texto):
        print(f"[ok] {texto}", flush=True)
        informe["comprobaciones"].append(texto)

    informe["reutilizados"] = comprobar_config_reutilizados()
    ok("T0, T1 y T2: las 45 corridas tienen CONFIG_T, lr 1e-3, lote 64, label smoothing 0,1, 200 epocas, "
       "paciencia 25, aug x10, 30 pasos, processed_holistic_mascara_rostro, manos, y la regla y entrada de su brazo")

    X_seq, y, nombres, senantes, loso, pos, canon, (X_padded, lengths) = preparar()
    tf, _, add_velocity, _, _ = canon
    Xc = ect.preparar()[0]
    assert np.array_equal(X_seq["mascara"], Xc["mascara"])
    assert np.array_equal(X_seq["imputacion"], Xc["imputacion"])
    ok("X_seq «mascara» e «imputacion» de este script = las de experimento_confirmacion_tcn.preparar, bit a bit")
    i0 = nombres.index("0030_0000_0003")
    assert np.array_equal(X_seq["pivote"][i0], mav.regla_pivote(X_padded[i0, :int(lengths[i0])], SEQ_LENGTH))
    n_piv = int(np.sum(np.any(X_seq["pivote"] != X_seq["mascara"], axis=(1, 2))))
    n_imp = int(np.sum(np.any(X_seq["imputacion"] != X_seq["mascara"], axis=(1, 2))))
    informe["videos_que_cambian"] = {"pivote_vs_mascara": n_piv, "imputacion_vs_mascara": n_imp,
                                     "n_videos": int(len(nombres))}
    ok(f"X_seq «pivote» = mano_ausente_variantes.remuestrear_todas_con_regla; cambia {n_piv} de {len(nombres)} "
       f"videos frente a «mascara» (imputacion: {n_imp})")
    for b in BRAZOS:
        f = canon_de_brazo(canon, b)[2]
        E = f(X_seq[BRAZOS[b]["regla"]][:50])
        assert E.shape[2] == BRAZOS[b]["n_rasgos"], b
        assert np.array_equal(E[:, :, :126], X_seq[BRAZOS[b]["regla"]][:50])
        assert (f is add_velocity) == (BRAZOS[b]["forma"] == "actual"), b
    ok("entradas: T0, T1 y T4 usan add_velocity (252); T2, T3 y T5 usan «por_coordenada» (132); posiciones "
       "intactas")

    n_par = {}
    for b in BRAZOS:
        tf.keras.utils.set_random_seed(0)
        n_par[b] = int(al.construir_tcn(SEQ_LENGTH, BRAZOS[b]["n_rasgos"], 50).count_params())
        tf.keras.backend.clear_session()
        assert n_par[b] == N_PARAM[BRAZOS[b]["n_rasgos"]], (b, n_par[b])
    ok(f"parametros: {n_par}")
    informe["n_parametros"] = n_par

    if args.dest:
        dest = Path(args.dest).resolve()
        d = dest / "T1_repro" / "semilla_2026" / "pliegue_0"
        if not (d / "resultado.json").exists():
            ns = argparse.Namespace(aug=10, epochs=200, patience=25, processed=str(PROCESSED_CANONICO))
            t0 = time.perf_counter()
            ea.entrenar_corrida("T", 2026, 0, loso["pliegues"][0], pos, X_seq["imputacion"], y,
                                canon_de_brazo(canon, "T1"), d, ns)
            print(f"  reproduccion en {time.perf_counter() - t0:.0f} s", flush=True)
        comp = ect.comparar_corridas(dir_corrida(None, "T1", 2026, 0), d)
        rr = json.loads((dir_corrida(None, "T1", 2026, 0) / "resultado.json").read_text(encoding="utf-8"))
        rn = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
        comp["val_loss"] = [rr["val_loss"], rn["val_loss"]]
        comp["probabilidades_npz_val_y_test_identicas"] = bool(
            comp["max_dif_probabilidades_val"] == 0.0 and comp["max_dif_probabilidades_test"] == 0.0)
        informe["reproduccion_T1_s2026_k0"] = comp
        print(json.dumps(comp, indent=1), flush=True)
        if comp["coincide"] and comp["predicciones_json_identico_byte_a_byte"] and comp["val_loss"][0] == comp["val_loss"][1]:
            ok("T1 semilla 2026 pliegue 0 reproducido con este script: predicciones.json identico byte a byte, "
               f"historial identico: {comp['historial_identico']}, misma val_loss, max |dif| de probabilidades "
               f"val {comp['max_dif_probabilidades_val']:.1e} y prueba {comp['max_dif_probabilidades_test']:.1e}")
        else:
            print("[FALLA] la reproduccion de T1 no coincide byte a byte: no se pueden reutilizar T0-T2", flush=True)
            informe["FALLA_reproduccion"] = True
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "verificacion.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {tl._rel(out / 'verificacion.json')}")


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def anotar_corrida(d: Path, brazo: str):
    """Corrige la procedencia y el resultado que escribe entrenar_corrida con la regla y la entrada reales."""
    from src.representacion import ruta_procedencia
    info = BRAZOS[brazo]
    rp = ruta_procedencia(d / "modelo.keras")
    proc = json.loads(rp.read_text(encoding="utf-8"))
    proc["mano_ausente"] = (info["regla"] if info["regla"] == "mascara" else
                            f"{info['regla']} (mano_ausente_variantes.regla_{info['regla']} sobre el processed "
                            f"«mascara», antes de remuestrear)")
    proc["velocidades"] = info["forma"]
    proc["n_rasgos_entrada"] = info["n_rasgos"]
    proc["columnas_velocidad"] = ("126 (add_velocity)" if info["forma"] == "actual"
                                  else vr.nombres_columnas_velocidad(info["forma"]))
    proc["script"] = "src/experimento_factorial_tcn.py (via experimento_arquitecturas.entrenar_corrida)"
    proc.pop("brazo_experimento_arquitecturas", None)
    proc["brazo_factorial_tcn"] = brazo
    proc["nota"] = ("modelo del diseno factorial con la TCN (docs/EXPERIMENTO_FACTORIAL_TCN.md). NO es el modelo "
                    "canonico ni debe cargarse en el prototipo.")
    rp.write_text(json.dumps(proc, indent=1, ensure_ascii=False), encoding="utf-8")
    rr = d / "resultado.json"
    r = json.loads(rr.read_text(encoding="utf-8"))
    r["constructor"] = "T (arquitecturas_literatura.construir_tcn, CONFIG_T)"
    r["brazo"] = brazo
    r.update({"regla_mano_ausente": info["regla"], "forma_velocidades": info["forma"],
              "n_rasgos_entrada": info["n_rasgos"]})
    rr.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


def cmd_entrenar(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.processed = str(Path(args.processed).resolve())
    X_seq, y, nombres, senantes, loso, pos, canon, _ = preparar(Path(args.processed))
    brazos = _lista_brazos(args.brazos, NUEVOS)
    semillas = _lista_int(args.semillas)
    folds = tl.parsear_folds(args.folds, len(loso["pliegues"]))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] experimento_factorial_tcn entrenar | brazos "
          f"{brazos} | semillas {semillas} | pliegues {folds} | epochs {args.epochs} | aug x{args.aug} | "
          f"paciencia {args.patience} | salida {tl._rel(out)}", flush=True)
    for s in semillas:
        for k in folds:
            p = loso["pliegues"][k]
            for b in brazos:
                d = Path(out) / BRAZOS[b]["carpeta"] / f"semilla_{s}" / f"pliegue_{k}"
                if (d / "resultado.json").exists():
                    print(f"  ya existe {tl._rel(d)}; se salta", flush=True)
                    continue
                print(f"\n{'=' * 72}\n  brazo {b} ({BRAZOS[b]['descripcion']}) | semilla {s} | pliegue {k} "
                      f"(senante {p['senante_test']}) | {datetime.now().isoformat(timespec='seconds')}\n{'=' * 72}",
                      flush=True)
                r = ea.entrenar_corrida("T", s, k, p, pos, X_seq[BRAZOS[b]["regla"]], y, canon_de_brazo(canon, b),
                                        d, args)
                assert r["n_parametros"] == N_PARAM[BRAZOS[b]["n_rasgos"]]
                anotar_corrida(d, b)
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] fin", flush=True)


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

def cmd_inferencia(args):
    out = Path(args.out).resolve()
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    tf, _, add_velocity, resample_sequence, _ = canon
    otros = ea.otros_entrenamientos()
    if otros:
        raise SystemExit(f"hay entrenamientos activos; la latencia se mide sin ellos: {otros}")
    rng = np.random.default_rng(0)
    rutas = {"R0": sorted(ea.R0_DIR.glob("semilla_2026/pliegue_*/modelo.keras"))[:args.modelos]}
    for b in BRAZOS:
        rutas[b] = [dir_corrida(out, b, 2026, k) / "modelo.keras" for k in range(args.modelos)]
    filas = []
    # Orden intercalado (R0, T0..T5 del mismo pliegue seguidos) para que una deriva no favorezca a nadie.
    for i in range(args.modelos):
        for b, lista in rutas.items():
            ruta = lista[i]
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
            fila = {"brazo": b, "modelo": tl._rel(ruta), "n_rasgos_entrada": F,
                    "n_parametros": int(model.count_params()),
                    "tamano_archivo_mb": round(ruta.stat().st_size / 2 ** 20, 3),
                    "ms_predict_lote1_mediana": float(np.median(t_pred)),
                    "ms_predict_lote1_p90": float(np.percentile(t_pred, 90)),
                    "ms_llamada_directa_mediana": float(np.median(t_call)),
                    "ms_llamada_directa_p90": float(np.percentile(t_call, 90))}
            filas.append(fila)
            print(f"  {b} {tl._rel(ruta)}: {F} rasgos | {fila['n_parametros']:,} param | "
                  f"{fila['tamano_archivo_mb']:.3f} MB | predict(1) {fila['ms_predict_lote1_mediana']:.2f} ms | "
                  f"model(x) {fila['ms_llamada_directa_mediana']:.2f} ms", flush=True)
            del model
            tf.keras.backend.clear_session()

    # Costo de armar la entrada de una ventana (processed -> 30 x F), por regla y forma; varios videos.
    X_padded, y, lengths, nombres, _ = tl.cargar_datos(PROCESSED_CANONICO, REPRESENTACION)
    videos = ["0030_0000_0003"] + [n for n in nombres[::97]][:args.videos - 1]
    prep = {}
    for b, info in BRAZOS.items():
        f_ent = funcion_entrada_brazo(b, add_velocity)
        meds = []
        for v in videos:
            i = nombres.index(v)
            seq = X_padded[i, :int(lengths[i])]

            def construir():
                x = (resample_sequence(seq, SEQ_LENGTH) if info["regla"] == "mascara"
                     else mav.aplicar_regla(seq, info["regla"], SEQ_LENGTH))
                return f_ent(np.asarray(x, dtype=np.float32)[None])
            for _ in range(10):
                construir()
            t = []
            for _ in range(args.repeticiones):
                t0 = time.perf_counter()
                construir()
                t.append((time.perf_counter() - t0) * 1000)
            meds.append(float(np.median(t)))
        prep[b] = {"ms_mediana_de_medianas": float(np.median(meds)), "ms_max_de_medianas": float(np.max(meds)),
                   "ms_por_video": dict(zip(videos, meds)), "videos": videos}
        print(f"  entrada {b}: {prep[b]['ms_mediana_de_medianas']:.3f} ms (mediana de {len(videos)} videos; "
              f"max {prep[b]['ms_max_de_medianas']:.3f})", flush=True)

    costo = {}
    for b in BRAZOS:
        fb = [f for f in filas if f["brazo"] == b]
        costo[b] = {"ms_red_predict_lote1": float(np.mean([f["ms_predict_lote1_mediana"] for f in fb])),
                    "ms_red_llamada_directa": float(np.mean([f["ms_llamada_directa_mediana"] for f in fb])),
                    "ms_entrada": prep[b]["ms_mediana_de_medianas"],
                    "mb_archivo": float(np.mean([f["tamano_archivo_mb"] for f in fb]))}
        costo[b]["ms_total"] = costo[b]["ms_red_predict_lote1"] + costo[b]["ms_entrada"]
    res = {"fecha": datetime.now().isoformat(timespec="seconds"),
           "metodo": f"model.predict(verbose=0) con lote de 1, como translator.py, y llamada directa model(x, "
                     f"training=False); {args.calentamiento} llamadas de calentamiento y {args.repeticiones} medidas "
                     f"por modelo; mediana y p90. Solo la red (sin MediaPipe). CPU. Modelos de la semilla 2026 "
                     f"(pliegues 0-{args.modelos - 1}), en orden intercalado R0, T0..T5. Entrada: processed -> "
                     f"regla -> 30 pasos -> velocidades, mediana de {args.repeticiones} repeticiones por video, "
                     f"{len(videos)} videos.",
           "procesos_de_entrenamiento_activos": otros, "entorno": tl.info_entorno(), "modelos": filas,
           "construccion_de_la_entrada": prep, "costo_por_brazo": costo}
    (out / "inferencia.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(costo, indent=1))
    print(f"escrito {tl._rel(out / 'inferencia.json')}")


# ---------------------------------------------------------------------------
# Criterio (funciones puras, probadas en tests/test_factorial_tcn.py)
# ---------------------------------------------------------------------------

def supera_en_validacion(val_a, val_b, margen) -> dict:
    """Regla 1: A supera a B si la media de Δk = val(A,k) − val(B,k) es > margen y Δk > 0 en ≥ 4 de 5."""
    return ect.regla_mejora(ect.deltas_por_pliegue(val_a, val_b), margen)


def supera_en_perdida(loss_a, loss_b, sigma_loss) -> dict:
    """Regla 2: A supera a B en val_loss si dk = loss(B,k) − loss(A,k) (positivo = A menor) es > 0 en ≥ 4 de 5
    y la media de dk es > sigma_loss (DE entre semillas agrupada de val_loss; sin piso)."""
    dif = np.round(np.asarray(loss_b, float).mean(axis=0) - np.asarray(loss_a, float).mean(axis=0), 6)
    media = float(dif.mean())
    mejor = int(np.sum(dif > 0))
    return {"reduccion_por_pliegue": dif.tolist(), "reduccion_media": media, "pliegues_menor": mejor,
            "pliegues_mayor": int(np.sum(dif < 0)), "a_reduccion_mayor_que_sigma": bool(media > sigma_loss + TOL),
            "b_menor_en_4_de_5": bool(mejor >= 4), "cumple": bool(media > sigma_loss + TOL and mejor >= 4)}


def no_superados(brazos, matriz) -> list:
    """Brazos que ningún otro de la lista supera. matriz[(a, b)]["cumple"] = a supera a b."""
    return [b for b in brazos if not any(matriz[(a, b)]["cumple"] for a in brazos if a != b)]


def orden_simplicidad(brazos, costo_velocidades: dict | None) -> list:
    """Paso 3. Primero el preprocesamiento (mascara < imputacion < pivote); dentro de cada uno, 252 antes que
    132 si la diferencia de costo entre los dos es despreciable (< 1 ms de latencia total Y < 0,5 MB);
    si no es despreciable, 132 antes que 252. costo_velocidades: {regla: {"dif_ms": .., "dif_mb": ..}}; si
    falta la medición de una regla, se toma como despreciable (la medición previa de T0 frente a T2 fue de
    0,3 ms y 0,12 MB)."""
    def despreciable(regla):
        c = (costo_velocidades or {}).get(regla)
        if c is None:
            return True
        return abs(c["dif_ms"]) < UMBRAL_COSTO_MS and abs(c["dif_mb"]) < UMBRAL_COSTO_MB

    def clave(b):
        regla, n = BRAZOS[b]["regla"], BRAZOS[b]["n_rasgos"]
        rango_vel = (0 if n == 252 else 1) if despreciable(regla) else (0 if n == 132 else 1)
        return (ORDEN_PREPROC[regla], rango_vel)
    return sorted(brazos, key=clave)


def aplicar_criterio(val: dict, loss: dict, costo_velocidades: dict | None = None) -> dict:
    """val, loss: {brazo: array (n_semillas, 5)}. Criterio de docs/EXPERIMENTO_FACTORIAL_TCN.md §2."""
    brazos = list(val)
    sigma, var = ect.sigma_agrupada(val)
    margen = max(sigma, PISO_MARGEN)
    sigma_loss, var_loss = ect.sigma_agrupada(loss)
    medias_val = {b: float(np.mean(val[b])) for b in brazos}
    medias_loss = {b: float(np.mean(loss[b])) for b in brazos}

    m1 = {(a, b): supera_en_validacion(val[a], val[b], margen) for a in brazos for b in brazos if a != b}
    n1 = no_superados(brazos, m1)
    nota1 = None
    if not n1:   # ciclo (no transitivo): se toman los que están a ≤ margen del mejor promedio
        mx = max(medias_val.values())
        n1 = [b for b in brazos if medias_val[b] >= mx - margen - TOL]
        nota1 = "ciclo en la regla 1: candidatos = brazos a <= margen de la mejor media"
    res = {"brazos": brazos, "sigma_val_agrupada": sigma, "varianza_val_entre_semillas": var, "margen": margen,
           "sigma_val_loss_agrupada": sigma_loss, "varianza_val_loss_entre_semillas": var_loss,
           "media_val": medias_val, "media_val_loss": medias_loss,
           "paso1": {"pares_que_cumplen": [f"{a}>{b}" for (a, b), r in m1.items() if r["cumple"]],
                     "no_superados": n1, "nota": nota1,
                     "pares": {f"{a}_vs_{b}": r for (a, b), r in m1.items()}}}
    if len(n1) == 1:
        res.update({"decide_en_paso": 1, "configuracion_final": n1[0]})
        return res

    m2 = {(a, b): supera_en_perdida(loss[a], loss[b], sigma_loss) for a in n1 for b in n1 if a != b}
    n2 = no_superados(n1, m2)
    nota2 = None
    if not n2:
        n2 = list(n1)
        nota2 = "ciclo en la regla 2: se pasa al paso 3 con todos los candidatos del paso 1"
    res["paso2"] = {"candidatos": n1, "pares_que_cumplen": [f"{a}>{b}" for (a, b), r in m2.items() if r["cumple"]],
                    "no_superados": n2, "nota": nota2, "pares": {f"{a}_vs_{b}": r for (a, b), r in m2.items()}}
    if len(n2) == 1:
        res.update({"decide_en_paso": 2, "configuracion_final": n2[0]})
        return res

    orden = orden_simplicidad(n2, costo_velocidades)
    res["paso3"] = {"candidatos": n2, "costo_velocidades": costo_velocidades, "orden_simplicidad": orden}
    res.update({"decide_en_paso": 3, "configuracion_final": orden[0]})
    return res


def costo_velocidades_desde_inferencia(inf: dict | None) -> dict | None:
    if not inf or "costo_por_brazo" not in inf:
        return None
    c = inf["costo_por_brazo"]
    return {regla: {"dif_ms": c[b132]["ms_total"] - c[b252]["ms_total"],
                    "dif_mb": c[b252]["mb_archivo"] - c[b132]["mb_archivo"]}
            for regla, (b252, b132) in PAR_VELOCIDADES.items()}


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def leer_corridas(out, y_por_nombre):
    filas = []
    for s in SEMILLAS:
        for k in range(5):
            m, pred, d = ea.leer_r0(s, k)
            filas.append({"brazo": "R0", "semilla": s, "pliegue": k, "senante_test": m["senante_test"],
                          "val_top1": m["mejor_val_accuracy"], "val_loss": None, "test_top1": m["top1"],
                          "test_top3": m["top3"], "test_f1_macro": m["f1_macro"], "n_parametros": m["n_parametros"],
                          "epocas_ejecutadas": m["epocas_ejecutadas"],
                          "predicciones": {x["muestra"]: (x["real"], x["predicha"], x["prob_predicha"])
                                           for x in pred}, "fuente": tl._rel(d)})
    for b in BRAZOS:
        for s in SEMILLAS:
            for k in range(5):
                d = dir_corrida(out, b, s, k)
                if not (d / "resultado.json").exists():
                    continue
                r = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
                pred = json.loads((d / "predicciones.json").read_text(encoding="utf-8"))
                pr = perdidas_corrida(d, y_por_nombre)
                filas.append({"brazo": b, "semilla": s, "pliegue": k, "senante_test": r["senante_test"],
                              "val_top1": r["val_top1"], "val_top1_checkpoint_reevaluado":
                              r["val_top1_checkpoint_reevaluado"], "val_loss": r["val_loss"],
                              "val_loss_recalculada_npz": pr["val_loss_recalculada_npz"],
                              "val_nll_sin_suavizado": pr["val_nll_sin_suavizado"],
                              "val_top3": r["val_top3"], "val_f1_macro": r["val_f1_macro"],
                              "test_top1": r["test_top1"], "test_top3": r["test_top3"],
                              "test_f1_macro": r["test_f1_macro"], "n_parametros": r["n_parametros"],
                              "epocas_ejecutadas": r["epocas_ejecutadas"], "mejor_epoca": r["mejor_epoca"],
                              "tiempo_total_s": r["tiempo_total_s"], "s_por_epoca": r["s_por_epoca"],
                              "predicciones": {x["muestra"]: (x["real"], x["predicha"], x["prob_predicha"])
                                               for x in pred}, "fuente": tl._rel(d)})
    return filas


def cmd_resumen(args):
    out = Path(args.out).resolve()
    ypn = _y_por_nombre()
    filas = leer_corridas(out, ypn)
    folds = list(range(5))
    todos = ["R0"] + list(TODOS)

    def celda(b, s, k):
        r = [f for f in filas if f["brazo"] == b and f["semilla"] == s and f["pliegue"] == k]
        return r[0] if r else None

    faltan = [(b, s, k) for b in todos for s in SEMILLAS for k in folds if celda(b, s, k) is None]
    if faltan:
        raise SystemExit(f"faltan corridas: {faltan}")

    # La val_loss fijada de T0-T2 no debe haber cambiado.
    fij = json.loads((out / ARCHIVO_VAL_LOSS_FIJADA).read_text(encoding="utf-8"))
    for f in fij["corridas"]:
        assert celda(f["brazo"], f["semilla"], f["pliegue"])["val_loss"] == f["val_loss"], f

    METR = ["val_top1", "val_loss", "val_nll_sin_suavizado", "val_top3", "val_f1_macro", "test_top1", "test_top3",
            "test_f1_macro", "epocas_ejecutadas", "mejor_epoca", "tiempo_total_s", "s_por_epoca"]
    por_brazo = {}
    for b in todos:
        r = {"n_parametros": celda(b, 2026, 0)["n_parametros"]}
        for m in METR:
            if celda(b, 2026, 0).get(m) is None:
                continue
            M = np.array([[celda(b, s, k)[m] for k in folds] for s in SEMILLAS], dtype=float)
            r[m] = {"por_semilla_por_pliegue": {str(s): M[i].tolist() for i, s in enumerate(SEMILLAS)},
                    "por_pliegue_media_semillas": M.mean(axis=0).tolist(),
                    "por_semilla_media_pliegues": {str(s): float(M[i].mean()) for i, s in enumerate(SEMILLAS)},
                    "de_entre_pliegues": float(np.std(M.mean(axis=0), ddof=1)),
                    "de_entre_semillas": float(np.std(M.mean(axis=1), ddof=1)),
                    "media_global": float(M.mean())}
        por_brazo[b] = r

    inf = json.loads((out / "inferencia.json").read_text(encoding="utf-8")) if (out / "inferencia.json").exists() \
        else None
    costo_vel = costo_velocidades_desde_inferencia(inf)
    val = {b: np.array([[celda(b, s, k)["val_top1"] for k in folds] for s in SEMILLAS]) for b in TODOS}
    loss = {b: np.array([[celda(b, s, k)["val_loss"] for k in folds] for s in SEMILLAS]) for b in TODOS}
    criterio = aplicar_criterio(val, loss, costo_vel)

    # Pruebas pareadas por pliegue (descriptivas): todos los pares de T0-T5 y cada uno frente a R0.
    pares = list(itertools.combinations(TODOS, 2)) + [(b, "R0") for b in TODOS]
    pruebas = {}
    for a, b in pares:
        clave = f"{b}_vs_{a}" if b != "R0" else f"{a}_vs_R0"
        x, ref = (b, a) if b != "R0" else (a, "R0")
        pruebas[clave] = {}
        for m in ("val_top1", "val_loss", "test_top1", "test_top3", "test_f1_macro"):
            if m not in por_brazo[ref]:
                continue
            dif = (np.array(por_brazo[x][m]["por_pliegue_media_semillas"])
                   - np.array(por_brazo[ref][m]["por_pliegue_media_semillas"]))
            pruebas[clave][m] = ea.pruebas_pareadas(dif)

    def agregadas(b, s):
        p = {}
        for k in folds:
            p.update(celda(b, s, k)["predicciones"])
        return p

    agg = {b: {s: agregadas(b, s) for s in SEMILLAS} for b in todos}
    voto = {b: ect.voto_semillas([agg[b][s] for s in SEMILLAS]) for b in todos}
    for b in todos:
        assert len(voto[b]) == 997, (b, len(voto[b]))
    mcnemar = {}
    for a, b in pares:
        x, ref = (b, a) if b != "R0" else (a, "R0")
        mcnemar[f"{x}_vs_{ref}"] = {"2026": ect._mc(agg[x][2026], agg[ref][2026]),
                                    "voto_3_semillas": ea.mcnemar(voto[x], voto[ref])}
    exact_voto = {b: float(np.mean([v[0] == v[1] for v in voto[b].values()])) for b in todos}

    p0000 = {b: {m: {"media": por_brazo[b][m]["por_pliegue_media_semillas"][0],
                     "por_semilla": {s: v[0] for s, v in por_brazo[b][m]["por_semilla_por_pliegue"].items()}}
                 for m in ("val_top1", "val_loss", "test_top1", "test_top3", "test_f1_macro") if m in por_brazo[b]}
             for b in todos}

    tabla_3x2 = {regla: {str(BRAZOS[b]["n_rasgos"]): {
        "brazo": b, "val_top1": por_brazo[b]["val_top1"]["media_global"],
        "val_loss": por_brazo[b]["val_loss"]["media_global"],
        "test_top1": por_brazo[b]["test_top1"]["media_global"],
        "test_f1_macro": por_brazo[b]["test_f1_macro"]["media_global"]} for b in par}
        for regla, par in PAR_VELOCIDADES.items()}

    extra = {}
    for nombre in ("verificacion.json", "humo.json", ARCHIVO_VAL_LOSS_FIJADA):
        if (out / nombre).exists():
            extra[nombre] = json.loads((out / nombre).read_text(encoding="utf-8"))
    sin_pred = [{k: v for k, v in f.items() if k != "predicciones"} for f in filas]
    resultado = {"experimento": "factorial {mascara, pivote, imputacion} x {252, 132} con la TCN "
                                "(docs/EXPERIMENTO_FACTORIAL_TCN.md)",
                 "fecha_resumen": datetime.now().isoformat(timespec="seconds"),
                 "brazos": BRAZOS, "R0": "canonico CNN+BiLSTM+atencion; " + tl._rel(ea.R0_DIR),
                 "config_T": al.CONFIG_T, "entrenamiento_T": al.ENTRENAMIENTO_COMUN, "semillas": SEMILLAS,
                 "criterio": criterio, "tabla_3x2": tabla_3x2, "pruebas_pareadas_por_pliegue": pruebas,
                 "mcnemar": mcnemar, "exactitud_voto_3_semillas": exact_voto, "persona_0000": p0000,
                 "por_brazo": por_brazo, "inferencia": inf, "verificacion": extra.get("verificacion.json"),
                 "humo": extra.get("humo.json"), "val_loss_fijada_T0_T1_T2": extra.get(ARCHIVO_VAL_LOSS_FIJADA),
                 "corridas": sin_pred}
    texto = json.dumps(resultado, indent=1, ensure_ascii=False, default=lambda o: o.tolist()
                       if isinstance(o, np.ndarray) else str(o))
    (out / "resultados_factorial_tcn.json").write_text(texto, encoding="utf-8")
    if args.docs:
        JSON_DOCS.write_text(texto, encoding="utf-8")
        print(f"escrito {tl._rel(JSON_DOCS)}")

    # Impresión
    print(f"sigma val {criterio['sigma_val_agrupada']:.4f} | margen {criterio['margen']:.4f} | sigma val_loss "
          f"{criterio['sigma_val_loss_agrupada']:.4f}")
    for b in todos:
        r = por_brazo[b]
        print(f"\n{b} | {r['n_parametros']:,} param | voto3 {exact_voto[b]:.4f}")
        for m in METR:
            if m in r:
                print(f"   {m:22s} {r[m]['media_global']:.4f} (DE pl {r[m]['de_entre_pliegues']:.4f}, DE sem "
                      f"{r[m]['de_entre_semillas']:.4f}) {[round(x, 3) for x in r[m]['por_pliegue_media_semillas']]}"
                      f" sem {[round(x, 4) for x in r[m]['por_semilla_media_pliegues'].values()]}")
    print("\nCRITERIO:", json.dumps({k: v for k, v in criterio.items() if k not in ("paso1", "paso2")}, indent=1,
                                    ensure_ascii=False))
    print("paso1 pares que cumplen:", criterio["paso1"]["pares_que_cumplen"], "| no superados:",
          criterio["paso1"]["no_superados"])
    if "paso2" in criterio:
        print("paso2 pares que cumplen:", criterio["paso2"]["pares_que_cumplen"], "| no superados:",
              criterio["paso2"]["no_superados"])
        for c, r in criterio["paso2"]["pares"].items():
            print(f"   {c}: {[round(x, 4) for x in r['reduccion_por_pliegue']]} media {r['reduccion_media']:.4f} "
                  f"menor en {r['pliegues_menor']}")
    for c, d in pruebas.items():
        for m, pr in d.items():
            print(f"  {c} {m}: {[round(x, 4) for x in pr['diferencias']]} +{pr['mejora']} -{pr['empeora']} "
                  f"={pr['empate']} | signos {pr['signos_p_bilateral']:.4f} | wilcoxon {pr['wilcoxon_p_bilateral']:.4f}")
    for c, d in mcnemar.items():
        for s, m in d.items():
            print(f"  McNemar {c} {s}: {m['aciertos_brazo']} vs {m['aciertos_R0']} | solo brazo "
                  f"{m['solo_brazo_acierta']} solo ref {m['solo_R0_acierta']} | p {m['p_exacto_bilateral']:.4g}")
    print("persona 0000:", {b: {m: round(v["media"], 3) for m, v in p0000[b].items()} for b in todos})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fijar_val_loss")
    f.add_argument("--forzar", action="store_true")
    v = sub.add_parser("verificar")
    v.add_argument("--dest", default=None, help="carpeta (carpeta_temporal) para reproducir T1 s2026 k0")
    e = sub.add_parser("entrenar")
    e.add_argument("--brazos", default="T3,T4,T5")
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
    i.add_argument("--videos", type=int, default=5)
    r = sub.add_parser("resumen")
    r.add_argument("--docs", action="store_true")
    for p in (f, v, e, i, r):
        p.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    {"fijar_val_loss": cmd_fijar_val_loss, "verificar": cmd_verificar, "entrenar": cmd_entrenar,
     "inferencia": cmd_inferencia, "resumen": cmd_resumen}[args.cmd](args)


if __name__ == "__main__":
    main()
