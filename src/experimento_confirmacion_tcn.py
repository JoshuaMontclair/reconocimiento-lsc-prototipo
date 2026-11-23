"""
Experimento de confirmación conjunta con la TCN (docs/EXPERIMENTO_CONFIRMACION_TCN.md).

Tres experimentos anteriores encontraron, cada uno por separado:
  · la TCN pequeña con GAP (brazo T de docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md) supera al canónico;
  · la imputación de la mano ausente (brazo C de docs/EXPERIMENTO_MANO_AUSENTE.md) no empeora;
  · las 6 velocidades por coordenada (brazo C de docs/EXPERIMENTO_VELOCIDADES_REDUCIDAS.md) no son peores.
Aquí se miden JUNTAS, siempre con la TCN (CONFIG_T exacta) y su receta de entrenamiento:

  T0  TCN + «mascara» + 252   control. NO se reentrena: se reutiliza
                              models/arquitecturas_literatura_2026-10-02/T_tcn tras verificar su
                              configuración y reproducir bit a bit la semilla 2026, pliegue 0.
  T1  TCN + imputación + 252  mano_ausente_variantes.regla_imputacion sobre los cuadros originales
                              (antes de remuestrear) -> aumentación ×10 -> add_velocity.
  T2  TCN + imputación + 132  igual que T1, pero la entrada es velocidades_reducidas.entrada_red(·,
                              «por_coordenada»): 126 posiciones + 6 velocidades, calculadas DESPUÉS de la
                              aumentación y del remuestreo.

Nada se reimplementa: el entrenamiento es experimento_arquitecturas.entrenar_corrida (brazo «T»), al que se
le pasa la X_seq de la regla y una tupla canon cuyo canon[2] es la función de entrada del brazo; la
imputación es mano_ausente_variantes.remuestrear_todas_con_regla; las velocidades reducidas,
velocidades_reducidas.funcion_entrada; McNemar y las pruebas pareadas, las de experimento_arquitecturas.

Subcomandos:
  verificar   configuración de T0 en las 15 corridas, reproducción de T0 (semilla 2026, pliegue 0) en una
              carpeta aparte, entradas de T1 y T2 y parámetros de cada red.
  entrenar    combinaciones (brazo, semilla, pliegue) que falten; reanudable.
  inferencia  parámetros y latencia con lote de 1 (como translator.py), sin entrenamientos activos, y costo
              de construir la entrada de una ventana con cada regla.
  resumen     agrega, aplica el criterio prerregistrado y escribe el JSON de datos crudos.

Uso:
  python src/experimento_confirmacion_tcn.py verificar --dest <carpeta_temporal>/repro_t0
  python src/experimento_confirmacion_tcn.py entrenar --brazos T1,T2 --semillas 2026 --folds 0, \\
         --epochs 3 --out <carpeta_temporal>/humo                                # prueba de humo
  python src/experimento_confirmacion_tcn.py entrenar --brazos T1,T2 --semillas 2026   # una por proceso
  python src/experimento_confirmacion_tcn.py inferencia
  python src/experimento_confirmacion_tcn.py resumen --docs
"""

from __future__ import annotations

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
from src import arquitecturas_literatura as al  # noqa: E402
from src import experimento_arquitecturas as ea  # noqa: E402
from src import mano_ausente_variantes as mav  # noqa: E402
from src import velocidades_reducidas as vr  # noqa: E402

PROCESSED_CANONICO = ea.PROCESSED_CANONICO
REPRESENTACION = ea.REPRESENTACION
SEQ_LENGTH = ea.SEQ_LENGTH
DEFAULT_OUT = BASE_DIR / "models" / "confirmacion_tcn_2026-10-03"
JSON_DOCS = BASE_DIR / "docs" / "EXPERIMENTO_CONFIRMACION_TCN.json"
T0_DIR = BASE_DIR / "models" / "arquitecturas_literatura_2026-10-02" / "T_tcn"
SEMILLAS = [2026, 2027, 2028]
PISO_MARGEN = 0.01
TOL = 1e-9          # tolerancia de coma flotante al comparar con el margen
N_PARAM_T0 = 156_402

BRAZOS = {
    "T0": {"carpeta": "T0_mascara_252", "regla": "mascara", "forma": "actual", "n_rasgos": 252,
           "descripcion": "TCN + mascara + 252 (control; reutiliza T_tcn de arquitecturas_literatura)"},
    "T1": {"carpeta": "T1_imputacion_252", "regla": "imputacion", "forma": "actual", "n_rasgos": 252,
           "descripcion": "TCN + imputacion + 252"},
    "T2": {"carpeta": "T2_imputacion_132", "regla": "imputacion", "forma": "por_coordenada", "n_rasgos": 132,
           "descripcion": "TCN + imputacion + 132 (6 velocidades por coordenada)"},
}
ENTRENABLES = ("T1", "T2")


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
        return T0_DIR / f"semilla_{semilla}" / f"pliegue_{k}"
    return Path(out) / BRAZOS[brazo]["carpeta"] / f"semilla_{semilla}" / f"pliegue_{k}"


def hiperparametros_t():
    """Los hiperparámetros que entrenar_corrida guarda para el brazo T (CONFIG_T y receta común)."""
    return dict(al.ENTRENAMIENTO_COMUN, arquitectura=al.CONFIG_T)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def preparar(processed=PROCESSED_CANONICO):
    """X_seq por regla («mascara» = train_loso.remuestrear_todas, como experimento_arquitecturas;
    «imputacion» = mano_ausente_variantes.remuestrear_todas_con_regla)."""
    X_padded, y, lengths, nombres, senantes, loso, pos = ea.cargar_todo(processed)
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    X_seq = {"mascara": tl.remuestrear_todas(X_padded, lengths, canon[3], SEQ_LENGTH),
             "imputacion": mav.remuestrear_todas_con_regla(X_padded, lengths, "imputacion", SEQ_LENGTH)}
    return X_seq, y, nombres, senantes, loso, pos, canon, (X_padded, lengths)


def canon_de_brazo(canon, brazo):
    """Tupla canon con canon[2] = función de entrada del brazo (se aplica DESPUÉS de aumentar)."""
    tf, build_model, add_velocity, resample_sequence, transformaciones = canon
    forma = BRAZOS[brazo]["forma"]
    entrada = add_velocity if forma == "actual" else vr.funcion_entrada(forma)
    return (tf, build_model, entrada, resample_sequence, transformaciones)


# ---------------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------------

def comprobar_config_t0():
    """Las 15 corridas de T_tcn tienen exactamente CONFIG_T, la receta de T y la entrada canónica."""
    esperado_hp = json.loads(json.dumps(hiperparametros_t()))
    filas = []
    for s in SEMILLAS:
        for k in range(5):
            d = dir_corrida(None, "T0", s, k)
            r = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
            proc = json.loads((d / "modelo.procedencia.json").read_text(encoding="utf-8"))
            assert r["brazo"] == "T" and r["semilla"] == s and r["pliegue"] == k, (s, k)
            assert r["hiperparametros"] == esperado_hp, (s, k, r["hiperparametros"])
            assert r["n_parametros"] == N_PARAM_T0, (s, k)
            assert (r["epochs_max"], r["aug"], r["patience"], r["seq_length"]) == (200, 10, 25, 30), (s, k)
            assert r["processed"] == "data/processed_holistic_mascara_rostro" and r["representacion"] == "manos"
            assert r["n_train_aumentado"] % 10 == 0 and r["n_val"] == 100
            assert proc["mano_ausente"] == "mascara" and proc["n_rasgos_entrada"] == 252
            assert proc["seq_length"] == 30 and proc["representacion"] == "manos"
            filas.append({"semilla": s, "pliegue": k, "val_top1": r["val_top1"], "test_top1": r["test_top1"],
                          "epocas": r["epocas_ejecutadas"]})
    return filas


def comparar_corridas(d_ref: Path, d_new: Path) -> dict:
    """Compara bit a bit (o con la tolerancia observada) dos corridas de entrenar_corrida."""
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
        "max_dif_prob_predicha": float(max(abs(a["prob_predicha"] - b["prob_predicha"]) for a, b in zip(pr, pn))),
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

    # 1. Configuración de T0.
    informe["t0"] = comprobar_config_t0()
    ok("T0: las 15 corridas de T_tcn tienen CONFIG_T exacta, lr 1e-3, lote 64, label smoothing 0,1, 200 "
       "epocas, paciencia 25, aug x10, 30 pasos, processed_holistic_mascara_rostro, manos, mascara, 252 rasgos "
       f"y {N_PARAM_T0:,} parametros")

    # 2. Entradas de los brazos.
    X_seq, y, nombres, senantes, loso, pos, canon, (X_padded, lengths) = preparar()
    tf, _, add_velocity, _, transformaciones = canon
    Xa, _, _, _, _, _, _ = ea.preparar()
    assert np.array_equal(X_seq["mascara"], Xa)
    ok(f"X_seq «mascara» de este script = la de experimento_arquitecturas.preparar, bit a bit {Xa.shape}")
    Xi = X_seq["imputacion"]
    i0 = nombres.index("0030_0000_0003") if "0030_0000_0003" in nombres else 0
    assert np.array_equal(Xi[i0], mav.regla_imputacion(X_padded[i0, :int(lengths[i0])], SEQ_LENGTH))
    n_dif = int(np.sum(np.any(Xi != X_seq["mascara"], axis=(1, 2))))
    ok(f"X_seq «imputacion» = mano_ausente_variantes.remuestrear_todas_con_regla; difiere de «mascara» en "
       f"{n_dif} de {len(Xi)} videos")
    informe["videos_que_cambian_con_imputacion"] = n_dif
    for b in BRAZOS:
        f = canon_de_brazo(canon, b)[2]
        E = f(X_seq[BRAZOS[b]["regla"]][:50])
        assert E.shape[2] == BRAZOS[b]["n_rasgos"], b
        assert np.array_equal(E[:, :, :126], X_seq[BRAZOS[b]["regla"]][:50])
    assert canon_de_brazo(canon, "T0")[2] is add_velocity and canon_de_brazo(canon, "T1")[2] is add_velocity
    ok("entradas: T0 y T1 usan add_velocity (252); T2 usa velocidades_reducidas «por_coordenada» (132); "
       "posiciones intactas")

    # 3. Parámetros.
    n_par = {}
    for b in BRAZOS:
        tf.keras.utils.set_random_seed(0)
        n_par[b] = int(al.construir_tcn(SEQ_LENGTH, BRAZOS[b]["n_rasgos"], 50).count_params())
        tf.keras.backend.clear_session()
    assert n_par["T0"] == n_par["T1"] == N_PARAM_T0
    ok(f"parametros: {n_par}")
    informe["n_parametros"] = n_par

    # 4. Reproducción de T0: semilla 2026, pliegue 0, con el mismo código que entrenará T1 y T2.
    if args.dest:
        dest = Path(args.dest).resolve()
        d = dest / "T0_repro" / "semilla_2026" / "pliegue_0"
        if not (d / "resultado.json").exists():
            ns = argparse.Namespace(aug=10, epochs=200, patience=25, processed=str(PROCESSED_CANONICO))
            t0 = time.perf_counter()
            ea.entrenar_corrida("T", 2026, 0, loso["pliegues"][0], pos, X_seq["mascara"], y,
                                canon_de_brazo(canon, "T0"), d, ns)
            print(f"  reproduccion en {time.perf_counter() - t0:.0f} s", flush=True)
        comp = comparar_corridas(dir_corrida(None, "T0", 2026, 0), d)
        informe["reproduccion_T0_s2026_k0"] = comp
        print(json.dumps(comp, indent=1), flush=True)
        if not comp["coincide"]:
            print("[FALLA] la reproduccion de T0 no coincide: hay que reentrenar T0", flush=True)
        else:
            ok("T0 semilla 2026 pliegue 0 reproducido con este script: predicciones y top-3 identicos, mismas "
               f"epocas y validacion; max |dif| de probabilidades de prueba {comp['max_dif_probabilidades_test']:.2e}"
               f"; predicciones.json identico byte a byte: {comp['predicciones_json_identico_byte_a_byte']}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "verificacion.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {tl._rel(out / 'verificacion.json')}")


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def anotar_corrida(d: Path, brazo: str):
    """Corrige la procedencia y el resultado que escribe entrenar_corrida (pensados para el brazo T de
    arquitecturas_literatura) con la regla y la entrada reales del brazo."""
    from src.representacion import ruta_procedencia
    info = BRAZOS[brazo]
    rp = ruta_procedencia(d / "modelo.keras")
    proc = json.loads(rp.read_text(encoding="utf-8"))
    proc["mano_ausente"] = (info["regla"] if info["regla"] == "mascara" else
                            "imputacion (mano_ausente_variantes.regla_imputacion sobre el processed «mascara», "
                            "antes de remuestrear)")
    proc["velocidades"] = info["forma"]
    proc["n_rasgos_entrada"] = info["n_rasgos"]
    proc["columnas_velocidad"] = ("126 (add_velocity)" if info["forma"] == "actual"
                                  else vr.nombres_columnas_velocidad(info["forma"]))
    proc["script"] = "src/experimento_confirmacion_tcn.py (via experimento_arquitecturas.entrenar_corrida)"
    proc.pop("brazo_experimento_arquitecturas", None)
    proc["brazo_confirmacion_tcn"] = brazo
    proc["nota"] = ("modelo del experimento de confirmacion conjunta con la TCN "
                    "(docs/EXPERIMENTO_CONFIRMACION_TCN.md). NO es el modelo canonico ni debe cargarse en el "
                    "prototipo.")
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
    brazos = _lista_brazos(args.brazos, ENTRENABLES + (("T0",) if args.reentrenar_t0 else ()))
    semillas = _lista_int(args.semillas)
    folds = tl.parsear_folds(args.folds, len(loso["pliegues"]))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] experimento_confirmacion_tcn entrenar | brazos "
          f"{brazos} | semillas {semillas} | pliegues {folds} | epochs {args.epochs} | aug x{args.aug} | "
          f"paciencia {args.patience} | salida {tl._rel(out)}", flush=True)
    for s in semillas:
        for k in folds:
            p = loso["pliegues"][k]
            for b in brazos:
                d = (Path(out) / BRAZOS[b]["carpeta"] / f"semilla_{s}" / f"pliegue_{k}")
                if (d / "resultado.json").exists():
                    print(f"  ya existe {tl._rel(d)}; se salta", flush=True)
                    continue
                print(f"\n{'=' * 72}\n  brazo {b} ({BRAZOS[b]['descripcion']}) | semilla {s} | pliegue {k} "
                      f"(senante {p['senante_test']}) | {datetime.now().isoformat(timespec='seconds')}\n{'=' * 72}",
                      flush=True)
                r = ea.entrenar_corrida("T", s, k, p, pos, X_seq[BRAZOS[b]["regla"]], y, canon_de_brazo(canon, b),
                                        d, args)
                assert r["n_parametros"] > 0
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
        rutas[b] = sorted((T0_DIR if b == "T0" else out / BRAZOS[b]["carpeta"])
                          .glob("semilla_2026/pliegue_*/modelo.keras"))[:args.modelos]
    filas = []
    # Orden intercalado (R0, T0, T1, T2 del mismo pliegue seguidos) para que una deriva no favorezca a nadie.
    for i in range(max(len(v) for v in rutas.values())):
        for b, lista in rutas.items():
            if i >= len(lista):
                continue
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
            print(f"  {b} {tl._rel(ruta)}: {F} rasgos | {fila['n_parametros']:,} param | predict(1) "
                  f"{fila['ms_predict_lote1_mediana']:.2f} ms | model(x) {fila['ms_llamada_directa_mediana']:.2f} ms",
                  flush=True)
            del model
            tf.keras.backend.clear_session()

    # Costo de construir la entrada de una ventana (processed -> 30 x F), por regla y forma.
    X_padded, y, lengths, nombres, _ = tl.cargar_datos(PROCESSED_CANONICO, REPRESENTACION)
    i0 = nombres.index("0030_0000_0003")
    seq = X_padded[i0, :int(lengths[i0])]
    prep = {}
    for b, info in BRAZOS.items():
        f_ent = canon_de_brazo(canon, b)[2]

        def construir():
            x = mav.aplicar_regla(seq, info["regla"], SEQ_LENGTH) if info["regla"] == "imputacion" \
                else resample_sequence(seq, SEQ_LENGTH)
            return f_ent(x[None])
        for _ in range(20):
            construir()
        t = []
        for _ in range(args.repeticiones * 5):
            t0 = time.perf_counter()
            construir()
            t.append((time.perf_counter() - t0) * 1000)
        prep[b] = {"ms_mediana": float(np.median(t)), "ms_p90": float(np.percentile(t, 90)),
                   "video": "0030_0000_0003", "n_cuadros": int(lengths[i0])}
        print(f"  entrada {b}: {prep[b]['ms_mediana']:.3f} ms (mediana)", flush=True)
    res = {"fecha": datetime.now().isoformat(timespec="seconds"),
           "metodo": f"model.predict(verbose=0) con lote de 1, como translator.py, y llamada directa model(x, "
                     f"training=False); {args.calentamiento} llamadas de calentamiento y {args.repeticiones} medidas "
                     f"por modelo; mediana y p90. Solo la red (sin MediaPipe). CPU. Modelos de la semilla 2026 "
                     f"(pliegues 0-{args.modelos - 1}), en orden intercalado R0, T0, T1, T2.",
           "procesos_de_entrenamiento_activos": otros, "entorno": tl.info_entorno(), "modelos": filas,
           "construccion_de_la_entrada": prep}
    (out / "inferencia.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {tl._rel(out / 'inferencia.json')}")


# ---------------------------------------------------------------------------
# Criterio (funciones puras, probadas en tests/test_confirmacion_tcn.py)
# ---------------------------------------------------------------------------

def sigma_agrupada(val_por_brazo: dict) -> tuple[float, dict]:
    """val_por_brazo: {brazo: array (n_semillas, 5)}. Para cada brazo, DE (ddof = 1) entre semillas de la
    validación media de 5 pliegues; se agrupa como raíz de la media de las varianzas."""
    var = {b: float(np.var(np.asarray(v, dtype=float).mean(axis=1), ddof=1)) for b, v in val_por_brazo.items()}
    return float(np.sqrt(np.mean(list(var.values())))), var


def deltas_por_pliegue(val_x, val_ref) -> np.ndarray:
    """Δk = media de semillas de X − media de semillas de ref, redondeado a 6 decimales (float32 de Keras)."""
    return np.round(np.asarray(val_x, float).mean(axis=0) - np.asarray(val_ref, float).mean(axis=0), 6)


def regla_mejora(dif, margen) -> dict:
    dif = np.asarray(dif, float)
    media = float(dif.mean())
    mejor = int(np.sum(dif > 0))
    return {"delta_media": media, "pliegues_mejor": mejor, "pliegues_peor": int(np.sum(dif < 0)),
            "a_delta_mayor_que_margen": bool(media > margen + TOL), "b_mejor_en_4_de_5": bool(mejor >= 4),
            "cumple": bool(media > margen + TOL and mejor >= 4)}


def regla_no_inferioridad(dif, margen) -> dict:
    dif = np.asarray(dif, float)
    media = float(dif.mean())
    pierde = int(np.sum(dif < -margen - TOL))
    return {"delta_media": media, "pliegues_que_pierden_por_mas_del_margen": pierde,
            "a_delta_no_menor_que_menos_margen": bool(media >= -margen - TOL),
            "b_pierde_por_mas_del_margen_en_a_lo_sumo_1": bool(pierde <= 1),
            "cumple": bool(media >= -margen - TOL and pierde <= 1)}


def aplicar_criterio(val: dict) -> dict:
    """val: {"T0","T1","T2": array (n_semillas, 5)} de top-1 de validación. Criterio de
    docs/EXPERIMENTO_CONFIRMACION_TCN.md §2."""
    sigma, var = sigma_agrupada(val)
    margen = max(sigma, PISO_MARGEN)
    d1 = deltas_por_pliegue(val["T1"], val["T0"])
    m1 = regla_mejora(d1, margen)
    n1 = regla_no_inferioridad(d1, margen)
    if m1["cumple"]:
        paso1, referencia = "T1 reemplaza a T0 (la imputacion mejora con la TCN)", "T1"
    elif n1["cumple"]:
        paso1, referencia = "se mantiene T0 (la imputacion es neutra con la TCN)", "T0"
    else:
        paso1, referencia = "se mantiene T0 (la imputacion empeora con la TCN)", "T0"
    d2 = deltas_por_pliegue(val["T2"], val[referencia])
    n2 = regla_no_inferioridad(d2, margen)
    m2 = regla_mejora(d2, margen)
    final = "T2" if n2["cumple"] else referencia
    return {"sigma_semilla_agrupada": sigma, "varianza_entre_semillas_por_brazo": var, "margen": margen,
            "paso1_T1_vs_T0": {"delta_por_pliegue": d1.tolist(), "regla_mejora": m1, "regla_no_peor": n1,
                               "resultado": paso1},
            "referencia_para_T2": referencia,
            "paso2_T2_vs_referencia": {"delta_por_pliegue": d2.tolist(), "regla_no_inferioridad": n2,
                                       "regla_mejora_informativa": m2,
                                       "resultado": ("T2 se adopta (no es peor que " + referencia + ")") if n2["cumple"]
                                       else ("T2 no se adopta (es peor que " + referencia + ")")},
            "configuracion_final": final}


def voto_semillas(preds_por_semilla: list) -> dict:
    """Voto de mayoría de la clase predicha entre semillas. preds: lista de {muestra: (real, predicha, prob)}.
    Si las tres semillas discrepan, gana la predicción de la semilla con mayor probabilidad predicha."""
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


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def leer_corridas(out):
    filas = []
    for s in SEMILLAS:
        for k in range(5):
            m, pred, d = ea.leer_r0(s, k)
            filas.append({"brazo": "R0", "semilla": s, "pliegue": k, "senante_test": m["senante_test"],
                          "val_top1": m["mejor_val_accuracy"], "test_top1": m["top1"], "test_top3": m["top3"],
                          "test_f1_macro": m["f1_macro"], "n_parametros": m["n_parametros"],
                          "epocas_ejecutadas": m["epocas_ejecutadas"], "tiempo_total_s": m["tiempo_total_s"],
                          "s_por_epoca": None,
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
                filas.append({"brazo": b, "semilla": s, "pliegue": k, "senante_test": r["senante_test"],
                              "val_top1": r["val_top1"], "val_top1_checkpoint_reevaluado":
                              r["val_top1_checkpoint_reevaluado"], "test_top1": r["test_top1"],
                              "test_top3": r["test_top3"], "test_f1_macro": r["test_f1_macro"],
                              "n_parametros": r["n_parametros"], "epocas_ejecutadas": r["epocas_ejecutadas"],
                              "mejor_epoca": r["mejor_epoca"], "tiempo_total_s": r["tiempo_total_s"],
                              "s_por_epoca": r["s_por_epoca"],
                              "predicciones": {x["muestra"]: (x["real"], x["predicha"], x["prob_predicha"])
                                               for x in pred}, "fuente": tl._rel(d)})
    return filas


def _mc(pa, pb):
    """McNemar de experimento_arquitecturas con predicciones (real, predicha[, prob])."""
    return ea.mcnemar({c: v[:2] for c, v in pa.items()}, {c: v[:2] for c, v in pb.items()})


def cmd_resumen(args):
    out = Path(args.out).resolve()
    filas = leer_corridas(out)
    folds = list(range(5))
    todos = ["R0", "T0", "T1", "T2"]

    def celda(b, s, k):
        r = [f for f in filas if f["brazo"] == b and f["semilla"] == s and f["pliegue"] == k]
        return r[0] if r else None

    faltan = [(b, s, k) for b in todos for s in SEMILLAS for k in folds if celda(b, s, k) is None]
    if faltan:
        raise SystemExit(f"faltan corridas: {faltan}")
    METR = ["val_top1", "test_top1", "test_top3", "test_f1_macro", "epocas_ejecutadas", "tiempo_total_s"]
    por_brazo = {}
    for b in todos:
        r = {"n_parametros": celda(b, 2026, 0)["n_parametros"]}
        for m in METR:
            M = np.array([[celda(b, s, k)[m] for k in folds] for s in SEMILLAS], dtype=float)
            r[m] = {"por_semilla_por_pliegue": {str(s): M[i].tolist() for i, s in enumerate(SEMILLAS)},
                    "por_pliegue_media_semillas": M.mean(axis=0).tolist(),
                    "por_semilla_media_pliegues": {str(s): float(M[i].mean()) for i, s in enumerate(SEMILLAS)},
                    "de_entre_pliegues": float(np.std(M.mean(axis=0), ddof=1)),
                    "de_entre_semillas": float(np.std(M.mean(axis=1), ddof=1)),
                    "media_global": float(M.mean())}
        if b != "R0":
            sp = np.array([[celda(b, s, k)["s_por_epoca"] for k in folds] for s in SEMILLAS], dtype=float)
            r["s_por_epoca_media"] = float(sp.mean())
        por_brazo[b] = r

    val = {b: np.array([[celda(b, s, k)["val_top1"] for k in folds] for s in SEMILLAS]) for b in ("T0", "T1", "T2")}
    criterio = aplicar_criterio(val)

    # Pruebas pareadas por pliegue (descriptivas) frente a T0 y frente a R0.
    pruebas = {}
    for ref in ("T0", "R0"):
        for b in [x for x in ("T0", "T1", "T2") if x != ref]:
            clave = f"{b}_vs_{ref}"
            pruebas[clave] = {}
            for m in ("val_top1", "test_top1", "test_top3", "test_f1_macro"):
                dif = (np.array(por_brazo[b][m]["por_pliegue_media_semillas"])
                       - np.array(por_brazo[ref][m]["por_pliegue_media_semillas"]))
                pruebas[clave][m] = ea.pruebas_pareadas(dif)
    dif21 = (np.array(por_brazo["T2"]["val_top1"]["por_pliegue_media_semillas"])
             - np.array(por_brazo["T1"]["val_top1"]["por_pliegue_media_semillas"]))
    pruebas["T2_vs_T1"] = {"val_top1": ea.pruebas_pareadas(dif21)}
    for m in ("test_top1", "test_f1_macro"):
        pruebas["T2_vs_T1"][m] = ea.pruebas_pareadas(
            np.array(por_brazo["T2"][m]["por_pliegue_media_semillas"])
            - np.array(por_brazo["T1"][m]["por_pliegue_media_semillas"]))

    # McNemar sobre las 997 predicciones agregadas: por semilla y con el voto de las 3 semillas.
    def agregadas(b, s):
        p = {}
        for k in folds:
            p.update(celda(b, s, k)["predicciones"])
        return p

    agg = {b: {s: agregadas(b, s) for s in SEMILLAS} for b in todos}
    voto = {b: voto_semillas([agg[b][s] for s in SEMILLAS]) for b in todos}
    for b in todos:
        assert len(voto[b]) == 997, (b, len(voto[b]))
    mcnemar = {}
    for ref in ("T0", "R0"):
        for b in [x for x in ("T0", "T1", "T2") if x != ref]:
            clave = f"{b}_vs_{ref}"
            mcnemar[clave] = {str(s): _mc(agg[b][s], agg[ref][s]) for s in SEMILLAS}
            mcnemar[clave]["voto_3_semillas"] = ea.mcnemar(voto[b], voto[ref])
    mcnemar["T2_vs_T1"] = {"2026": _mc(agg["T2"][2026], agg["T1"][2026]),
                           "voto_3_semillas": ea.mcnemar(voto["T2"], voto["T1"])}
    exact_voto = {b: float(np.mean([v[0] == v[1] for v in voto[b].values()])) for b in todos}

    # Persona 0000 (pliegue 0).
    p0000 = {b: {m: {"media": por_brazo[b][m]["por_pliegue_media_semillas"][0],
                     "por_semilla": {s: v[0] for s, v in por_brazo[b][m]["por_semilla_por_pliegue"].items()}}
                 for m in ("val_top1", "test_top1", "test_top3", "test_f1_macro")} for b in todos}

    extra = {}
    for nombre in ("verificacion.json", "inferencia.json", "humo.json"):
        if (out / nombre).exists():
            extra[nombre] = json.loads((out / nombre).read_text(encoding="utf-8"))
    sin_pred = [{k: v for k, v in f.items() if k != "predicciones"} for f in filas]
    resultado = {"experimento": "confirmacion conjunta con la TCN (docs/EXPERIMENTO_CONFIRMACION_TCN.md)",
                 "fecha_resumen": datetime.now().isoformat(timespec="seconds"),
                 "brazos": BRAZOS, "R0": "canonico CNN+BiLSTM+atencion; " + tl._rel(ea.R0_DIR),
                 "config_T": al.CONFIG_T, "entrenamiento_T": al.ENTRENAMIENTO_COMUN, "semillas": SEMILLAS,
                 "criterio": criterio, "pruebas_pareadas_por_pliegue": pruebas, "mcnemar": mcnemar,
                 "exactitud_voto_3_semillas": exact_voto, "persona_0000": p0000, "por_brazo": por_brazo,
                 "verificacion": extra.get("verificacion.json"), "humo": extra.get("humo.json"),
                 "inferencia": extra.get("inferencia.json"), "corridas": sin_pred}
    (out / "resultados_confirmacion_tcn.json").write_text(json.dumps(resultado, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    if args.docs:
        JSON_DOCS.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"escrito {tl._rel(JSON_DOCS)}")

    # Impresión
    print(f"sigma {criterio['sigma_semilla_agrupada']:.4f} | margen {criterio['margen']:.4f} | var "
          f"{criterio['varianza_entre_semillas_por_brazo']}")
    for b in todos:
        r = por_brazo[b]
        print(f"\n{b} | {r['n_parametros']:,} param | voto3 {exact_voto[b]:.4f}")
        for m in ("val_top1", "test_top1", "test_top3", "test_f1_macro", "epocas_ejecutadas", "tiempo_total_s"):
            print(f"   {m:18s} {r[m]['media_global']:.4f} (DE pl {r[m]['de_entre_pliegues']:.4f}, DE sem "
                  f"{r[m]['de_entre_semillas']:.4f}) {[round(x, 3) for x in r[m]['por_pliegue_media_semillas']]} "
                  f"sem {[round(x, 4) for x in r[m]['por_semilla_media_pliegues'].values()]}")
    print("\nCRITERIO:", json.dumps({k: v for k, v in criterio.items()}, indent=1, ensure_ascii=False))
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
    v = sub.add_parser("verificar")
    v.add_argument("--dest", default=None, help="carpeta (carpeta_temporal) para reproducir T0 s2026 k0")
    e = sub.add_parser("entrenar")
    e.add_argument("--brazos", default="T1,T2")
    e.add_argument("--semillas", default="2026,2027,2028")
    e.add_argument("--folds", default="5")
    e.add_argument("--epochs", type=int, default=200)
    e.add_argument("--aug", type=int, default=10)
    e.add_argument("--patience", type=int, default=25)
    e.add_argument("--processed", default=str(PROCESSED_CANONICO))
    e.add_argument("--reentrenar-t0", action="store_true",
                   help="solo si la verificacion de T0 falla: permite entrenar T0 en --out")
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
