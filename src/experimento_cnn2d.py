"""
Experimento de cierre: CNN 2D pequeña («esqueleto como imagen») frente a la TCN final (D-14).
Informe: docs/EXPERIMENTO_CNN2D.md. LOSO de 5 pliegues, particiones D-06, semillas 2026-2028.

El director pidió «un modelo más para que la investigación quede más completa» (D-14) y el estudiante eligió
una CNN 2D. Es una comparación de CIERRE: una sola configuración fijada antes de entrenar
(arquitectura_cnn2d.CONFIG_C2D), sin búsqueda de hiperparámetros. El resultado se reporta salga como salga y no
cambia por sí solo el modelo final.

Brazos (misma entrada que T1: imputación de la mano ausente + 252 rasgos, 30 pasos, aumentación ×10; misma
receta: Adam lr 1e-3 con CosineDecay, label smoothing 0,1, lote 64, paciencia 25 sobre val_accuracy, 200 épocas):

  T1   TCN final (control)   NO se reentrena: se reutilizan las 15 corridas de
                             models/confirmacion_tcn_2026-10-03/T1_imputacion_252, tras verificar su
                             configuración y reproducir con ESTE script la semilla 2026, pliegue 0.
  C2D  CNN 2D pequeña        arquitectura_cnn2d.construir_cnn2d: la ventana 30 × 252 se remodela DENTRO del
                             modelo como imagen 30 (tiempo) × 21 (puntos) × 12 canales (x, y, z, vx, vy, vz de
                             la mano izquierda + los de la derecha). Por estar dentro del modelo, la
                             remodelación va después de la aumentación (espejo incluido) y de add_velocity.

Nada del entrenamiento se reimplementa: se usa experimento_arquitecturas.entrenar_corrida (la que entrenó T1),
ampliando en memoria su función `constructor` con el brazo C2D (como experimento_arquitecturas_2), con la X_seq
«imputacion» y la función de entrada de T1 de experimento_confirmacion_tcn. Ningún script existente se modifica.

Subcomandos:
  verificar   configuración de las 15 corridas de T1, orden de columnas y remodelación (también tras el espejo),
              parámetros y, con --dest, reproducción de T1 (semilla 2026, pliegue 0) con este script.
  entrenar    combinaciones (semilla, pliegue) de C2D que falten; reanudable.
  inferencia  parámetros y latencia con lote de 1 (como translator.py), sin entrenamientos activos.
  resumen     agrega, aplica el criterio prerregistrado, arma la tabla de las 13 filas de familias y escribe el
              JSON de datos crudos.

Uso:
  python src/experimento_cnn2d.py verificar --dest <carpeta_temporal>/repro_t1
  python src/experimento_cnn2d.py entrenar --semillas 2026 --folds 0, --epochs 3 --out <carpeta_temporal>/humo
  python src/experimento_cnn2d.py entrenar --semillas 2026          # una semilla por proceso
  python src/experimento_cnn2d.py inferencia
  python src/experimento_cnn2d.py resumen --docs
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
from src import arquitectura_cnn2d as ac  # noqa: E402
from src import experimento_arquitecturas as ea  # noqa: E402
from src import experimento_confirmacion_tcn as ect  # noqa: E402

PROCESSED_CANONICO = ea.PROCESSED_CANONICO
REPRESENTACION = ea.REPRESENTACION
SEQ_LENGTH = ea.SEQ_LENGTH
DEFAULT_OUT = BASE_DIR / "models" / "cnn2d_2026-11-08"
JSON_DOCS = BASE_DIR / "docs" / "EXPERIMENTO_CNN2D.json"
T1_DIR = BASE_DIR / "models" / "confirmacion_tcn_2026-10-03" / "T1_imputacion_252"
SEMILLAS = [2026, 2027, 2028]
PISO_MARGEN = 0.01
TOL = 1e-9
N_PARAM_T1 = 156_402

BRAZOS = {
    "T1": {"carpeta": "T1_imputacion_252", "descripcion": "TCN final (D-14): TCN + GAP + imputacion + 252 "
                                                          "(control; reutiliza confirmacion_tcn)"},
    "C2D": {"carpeta": "C2D_cnn2d", "descripcion": "CNN 2D pequena: imagen 30 x 21 x 12, 3 bloques 32-64-64, "
                                                   "GAP 2D (imputacion + 252)"},
}
COMPARADOS = ("T1", "C2D")

# Fuentes de la tabla de familias (cifras de los informes previos, leídas de sus JSON de datos crudos).
FUENTE_LITERATURA = BASE_DIR / "docs" / "EXPERIMENTO_ARQUITECTURAS_LITERATURA.json"
FUENTE_ARQ2 = BASE_DIR / "docs" / "EXPERIMENTO_ARQUITECTURAS_2.json"
FUENTE_LINEA_BASE = BASE_DIR / "models" / "linea_base_4d_2026-09-15" / "resultados.json"


# ---------------------------------------------------------------------------
# Constructor ampliado (en memoria) de experimento_arquitecturas
# ---------------------------------------------------------------------------

_CONSTRUCTOR_ORIGINAL = ea.constructor


def constructor(brazo):
    """C2D con su configuración y la receta de la TCN; cualquier otro brazo (T) va al constructor original."""
    if brazo == "C2D":
        return (lambda nf, nfe, nc: ac.construir_cnn2d(nf, nfe, nc),
                dict(ac.ENTRENAMIENTO_C2D, arquitectura=ac.CONFIG_C2D))
    return _CONSTRUCTOR_ORIGINAL(brazo)


ea.constructor = constructor        # entrenar_corrida busca `constructor` en el módulo al llamarse


def _lista_int(texto):
    return [int(t) for t in str(texto).split(",") if t.strip() != ""]


def dir_corrida(out, brazo, semilla, k):
    if brazo == "T1":
        return T1_DIR / f"semilla_{semilla}" / f"pliegue_{k}"
    return Path(out) / BRAZOS[brazo]["carpeta"] / f"semilla_{semilla}" / f"pliegue_{k}"


# ---------------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------------

def comprobar_config_t1():
    """Las 15 corridas de T1 tienen CONFIG_T exacta, la receta de la TCN, imputación y 252 rasgos."""
    esperado_hp = json.loads(json.dumps(dict(al.ENTRENAMIENTO_COMUN, arquitectura=al.CONFIG_T)))
    filas = []
    for s in SEMILLAS:
        for k in range(5):
            d = dir_corrida(None, "T1", s, k)
            r = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
            proc = json.loads((d / "modelo.procedencia.json").read_text(encoding="utf-8"))
            assert r["brazo"] == "T1" and r["semilla"] == s and r["pliegue"] == k, (s, k)
            assert r["hiperparametros"] == esperado_hp, (s, k, r["hiperparametros"])
            assert r["n_parametros"] == N_PARAM_T1, (s, k)
            assert (r["epochs_max"], r["aug"], r["patience"], r["seq_length"]) == (200, 10, 25, 30), (s, k)
            assert r["processed"] == "data/processed_holistic_mascara_rostro" and r["representacion"] == "manos"
            assert r["regla_mano_ausente"] == "imputacion" and r["forma_velocidades"] == "actual"
            assert r["n_rasgos_entrada"] == 252 and r["n_train_aumentado"] % 10 == 0 and r["n_val"] == 100
            assert proc["mano_ausente"].startswith("imputacion") and proc["n_rasgos_entrada"] == 252
            assert proc["seq_length"] == 30 and proc["representacion"] == "manos"
            filas.append({"semilla": s, "pliegue": k, "val_top1": r["val_top1"], "val_loss": r["val_loss"],
                          "test_top1": r["test_top1"], "epocas": r["epocas_ejecutadas"]})
    return filas


def comprobar_remodelacion(X126: np.ndarray, add_velocity, espejo) -> dict:
    """Comprueba sobre datos reales que la imagen (30, 21, 12) toma las columnas correctas, que la remodelación
    del modelo es la de numpy, y que tras el espejo (sobre las 126 posiciones) + add_velocity la imagen es la
    original con las manos intercambiadas y x, vx con el signo cambiado."""
    from src.representacion import columnas, diseno_processed
    # Orden real de columnas de «manos» en el processed: mano_izq (99:162) y luego mano_der (162:225).
    bloques = {n: (a, b) for n, a, b in diseno_processed("ninguno")}
    cols = columnas("manos")
    assert bloques["mano_izq"] == (99, 162) and bloques["mano_der"] == (162, 225), bloques
    assert np.array_equal(cols, np.arange(99, 225)), "«manos» debe ser mano_izq seguida de mano_der"
    X252 = add_velocity(X126)
    assert X252.shape[1:] == (SEQ_LENGTH, 252)
    im = ac.remodelar_a_imagen_np(X252)
    # (1) Cada canal es la columna esperada.
    for mano in range(2):
        for v in range(2):
            for p in range(ac.N_MANO):
                for c in range(ac.COORDS):
                    assert np.array_equal(im[:, :, p, ac.canal(mano, v, c)],
                                          X252[:, :, ac.indice_columna(mano, v, p, c)])
    # (2) La remodelación dentro del modelo es la de numpy.
    import tensorflow as tf
    m = ac.construir_cnn2d()
    im_modelo = ac.submodelo_imagen(m)(X252.astype(np.float32)).numpy()
    assert np.array_equal(im_modelo, im.astype(np.float32))
    del m
    tf.keras.backend.clear_session()
    # (3) Espejo de la aumentación sobre las 126 posiciones, después add_velocity y la imagen.
    im_esp = ac.remodelar_a_imagen_np(add_velocity(espejo(X126)))
    esperado = np.empty_like(im)
    for mano in range(2):
        for v in range(2):
            for c in range(ac.COORDS):
                signo = -1.0 if c == 0 else 1.0
                esperado[..., ac.canal(mano, v, c)] = signo * im[..., ac.canal(1 - mano, v, c)]
    max_dif = float(np.max(np.abs(im_esp - esperado)))
    assert max_dif < 1e-12, max_dif
    return {"n_muestras": int(len(X126)), "columnas_manos": "processed 99:225 (mano_izq 99:162, mano_der 162:225)",
            "canal_por_columna_ok": True, "remodelacion_modelo_igual_a_numpy": True,
            "espejo_intercambia_manos_y_niega_x_vx": True, "max_dif_espejo": max_dif}


def cmd_verificar(args):
    informe = {"fecha": datetime.now().isoformat(timespec="seconds"), "comprobaciones": []}

    def ok(texto):
        print(f"[ok] {texto}", flush=True)
        informe["comprobaciones"].append(texto)

    informe["t1"] = comprobar_config_t1()
    ok(f"T1: las 15 corridas tienen CONFIG_T exacta, lr 1e-3, lote 64, label smoothing 0,1, 200 epocas, paciencia "
       f"25, aug x10, 30 pasos, processed_holistic_mascara_rostro, manos, imputacion, 252 rasgos y "
       f"{N_PARAM_T1:,} parametros")

    X_seq, y, nombres, senantes, loso, pos, canon, _ = ect.preparar()
    tf, _, add_velocity, _, transformaciones = canon
    assert transformaciones[0][0] == "espejo"
    assert ect.canon_de_brazo(canon, "T1")[2] is add_velocity
    Xi = X_seq["imputacion"]
    rem = comprobar_remodelacion(Xi[:64], add_velocity, transformaciones[0][1])
    informe["remodelacion"] = rem
    ok("remodelacion: «manos» = mano_izq (processed 99:162) y mano_der (162:225); canal 6*mano + 3*(pos/vel) + "
       "coord toma la columna 126*(pos/vel) + 63*mano + 3*punto + coord; la capa Reshape/Permute del modelo da la "
       "misma imagen que numpy; tras el espejo de la aumentacion + add_velocity la imagen es la original con las "
       f"manos intercambiadas y x, vx negadas (max |dif| {rem['max_dif_espejo']:.1e}); 64 videos reales")

    n_par = {}
    for b, fn in (("T1", lambda: al.construir_tcn(SEQ_LENGTH, 252, 50)),
                  ("C2D", lambda: ac.construir_cnn2d(SEQ_LENGTH, 252, 50))):
        tf.keras.utils.set_random_seed(0)
        n_par[b] = int(fn().count_params())
        tf.keras.backend.clear_session()
    assert n_par == {"T1": N_PARAM_T1, "C2D": ac.N_PARAM_C2D}, n_par
    assert 100_000 <= n_par["C2D"] <= 350_000
    ok(f"parametros: {n_par} (C2D dentro de 100-350 mil)")
    informe["n_parametros"] = n_par
    informe["config_C2D"] = ac.CONFIG_C2D
    informe["entrenamiento_C2D"] = ac.ENTRENAMIENTO_C2D

    if args.dest:
        dest = Path(args.dest).resolve()
        d = dest / "T1_repro" / "semilla_2026" / "pliegue_0"
        if not (d / "resultado.json").exists():
            ns = argparse.Namespace(aug=10, epochs=200, patience=25, processed=str(PROCESSED_CANONICO))
            t0 = time.perf_counter()
            ea.entrenar_corrida("T", 2026, 0, loso["pliegues"][0], pos, Xi, y, ect.canon_de_brazo(canon, "T1"), d, ns)
            print(f"  reproduccion en {time.perf_counter() - t0:.0f} s", flush=True)
        comp = ect.comparar_corridas(dir_corrida(None, "T1", 2026, 0), d)
        rr = json.loads((dir_corrida(None, "T1", 2026, 0) / "resultado.json").read_text(encoding="utf-8"))
        rn = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
        comp["val_loss"] = [rr["val_loss"], rn["val_loss"]]
        informe["reproduccion_T1_s2026_k0"] = comp
        print(json.dumps(comp, indent=1), flush=True)
        if not (comp["coincide"] and comp["predicciones_json_identico_byte_a_byte"]):
            print("[FALLA] la reproduccion de T1 no es identica byte a byte: no se puede reutilizar T1", flush=True)
        else:
            ok("T1 semilla 2026 pliegue 0 reproducido con este script (entrenar_corrida con el constructor "
               f"ampliado): predicciones.json identico byte a byte; historial identico: {comp['historial_identico']}; "
               f"max |dif| de probabilidades de prueba {comp['max_dif_probabilidades_test']:.2e}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "verificacion.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {tl._rel(out / 'verificacion.json')}")


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def anotar_corrida(d: Path):
    """Corrige la procedencia y el resultado que escribe entrenar_corrida con la regla, la entrada y el script
    reales del brazo C2D."""
    from src.representacion import ruta_procedencia
    rp = ruta_procedencia(d / "modelo.keras")
    proc = json.loads(rp.read_text(encoding="utf-8"))
    proc["mano_ausente"] = ("imputacion (mano_ausente_variantes.regla_imputacion sobre el processed «mascara», "
                            "antes de remuestrear)")
    proc["velocidades"] = "actual"
    proc["n_rasgos_entrada"] = 252
    proc["columnas_velocidad"] = "126 (add_velocity)"
    proc["imagen_interna"] = ac.CONFIG_C2D["imagen"]
    proc["script"] = "src/experimento_cnn2d.py (via experimento_arquitecturas.entrenar_corrida)"
    proc.pop("brazo_experimento_arquitecturas", None)
    proc["brazo_cnn2d"] = "C2D"
    proc["nota"] = ("modelo del experimento de cierre CNN 2D (docs/EXPERIMENTO_CNN2D.md). NO es el modelo final "
                    "(D-14) ni debe cargarse en el prototipo.")
    rp.write_text(json.dumps(proc, indent=1, ensure_ascii=False), encoding="utf-8")
    rr = d / "resultado.json"
    r = json.loads(rr.read_text(encoding="utf-8"))
    r["constructor"] = "arquitectura_cnn2d.construir_cnn2d (CONFIG_C2D)"
    r["script"] = "src/experimento_cnn2d.py"
    r.update({"regla_mano_ausente": "imputacion", "forma_velocidades": "actual", "n_rasgos_entrada": 252})
    rr.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


def cmd_entrenar(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.processed = str(Path(args.processed).resolve())
    X_seq, y, nombres, senantes, loso, pos, canon, _ = ect.preparar(Path(args.processed))
    canon_t1 = ect.canon_de_brazo(canon, "T1")
    semillas = _lista_int(args.semillas)
    folds = tl.parsear_folds(args.folds, len(loso["pliegues"]))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] experimento_cnn2d entrenar | brazo C2D | semillas "
          f"{semillas} | pliegues {folds} | epochs {args.epochs} | aug x{args.aug} | paciencia {args.patience} | "
          f"salida {tl._rel(out)}", flush=True)
    for s in semillas:
        for k in folds:
            d = Path(out) / BRAZOS["C2D"]["carpeta"] / f"semilla_{s}" / f"pliegue_{k}"
            if (d / "resultado.json").exists():
                print(f"  ya existe {tl._rel(d)}; se salta", flush=True)
                continue
            p = loso["pliegues"][k]
            print(f"\n{'=' * 72}\n  brazo C2D ({BRAZOS['C2D']['descripcion']}) | semilla {s} | pliegue {k} "
                  f"(senante {p['senante_test']}) | {datetime.now().isoformat(timespec='seconds')}\n{'=' * 72}",
                  flush=True)
            r = ea.entrenar_corrida("C2D", s, k, p, pos, X_seq["imputacion"], y, canon_t1, d, args)
            assert r["n_parametros"] == ac.N_PARAM_C2D
            anotar_corrida(d)
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
    rutas = {b: sorted((T1_DIR if b == "T1" else out / BRAZOS[b]["carpeta"])
                       .glob("semilla_2026/pliegue_*/modelo.keras"))[:args.modelos] for b in COMPARADOS}
    filas = []
    for i in range(max(len(v) for v in rutas.values())):       # intercalado T1, C2D del mismo pliegue
        for b in COMPARADOS:
            if i >= len(rutas[b]):
                continue
            ruta = rutas[b][i]
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
            print(f"  {b} {tl._rel(ruta)}: {fila['n_parametros']:,} param | predict(1) "
                  f"{fila['ms_predict_lote1_mediana']:.2f} ms (p90 {fila['ms_predict_lote1_p90']:.2f}) | model(x) "
                  f"{fila['ms_llamada_directa_mediana']:.2f} ms", flush=True)
            del model
            tf.keras.backend.clear_session()
    res = {"fecha": datetime.now().isoformat(timespec="seconds"),
           "metodo": f"model.predict(verbose=0) con lote de 1, como translator.py, y llamada directa model(x, "
                     f"training=False); {args.calentamiento} llamadas de calentamiento y {args.repeticiones} medidas "
                     f"por modelo; mediana y p90. Solo la red (sin MediaPipe ni armado de la entrada, que es el "
                     f"mismo para los dos brazos). CPU. Modelos de la semilla 2026 (pliegues 0-{args.modelos - 1}), "
                     f"en orden intercalado T1, C2D.",
           "procesos_de_entrenamiento_activos": otros, "entorno": tl.info_entorno(), "modelos": filas}
    (out / "inferencia.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {tl._rel(out / 'inferencia.json')}")


# ---------------------------------------------------------------------------
# Criterio (funciones puras, probadas en tests/test_cnn2d.py)
# ---------------------------------------------------------------------------

def supera(dif, margen) -> dict:
    """«Gana» si la media de las diferencias por pliegue (a favor del que gana) es > margen y es > 0 en al
    menos 4 de los 5 pliegues (0 no cuenta)."""
    dif = np.asarray(dif, float)
    media = float(dif.mean())
    a_favor = int(np.sum(dif > 0))
    return {"delta_media": media, "pliegues_a_favor": a_favor, "pliegues_en_contra": int(np.sum(dif < 0)),
            "a_media_mayor_que_margen": bool(media > margen + TOL), "b_a_favor_en_4_de_5": bool(a_favor >= 4),
            "cumple": bool(media > margen + TOL and a_favor >= 4)}


def aplicar_criterio(val: dict, loss: dict) -> dict:
    """val, loss: {"T1", "C2D": array (n_semillas, 5)} de top-1 y val_loss de validación (checkpoint).
    Criterio de docs/EXPERIMENTO_CNN2D.md §2."""
    sigma, var = ect.sigma_agrupada({b: val[b] for b in COMPARADOS})
    margen = max(sigma, PISO_MARGEN)
    d_top1 = ect.deltas_por_pliegue(val["C2D"], val["T1"])            # > 0: C2D acierta más
    c2d_top1 = supera(d_top1, margen)
    t1_top1 = supera(-d_top1, margen)
    sigma_loss, var_loss = ect.sigma_agrupada({b: loss[b] for b in COMPARADOS})
    d_loss = ect.deltas_por_pliegue(loss["T1"], loss["C2D"])          # > 0: C2D tiene menor pérdida
    c2d_loss = supera(d_loss, sigma_loss)
    t1_loss = supera(-d_loss, sigma_loss)
    if c2d_top1["cumple"]:
        resultado, por = "C2D supera a T1", "top-1 de validacion"
    elif t1_top1["cumple"]:
        resultado, por = "C2D pierde frente a T1", "top-1 de validacion"
    elif c2d_loss["cumple"]:
        resultado, por = "C2D supera a T1", "empate en top-1; desempata la val_loss"
    elif t1_loss["cumple"]:
        resultado, por = "C2D pierde frente a T1", "empate en top-1; desempata la val_loss"
    else:
        resultado, por = "empate", "empate en top-1 y en val_loss"
    return {"sigma_top1_entre_semillas_agrupada": sigma, "varianza_top1_por_brazo": var, "margen_top1": margen,
            "sigma_val_loss_entre_semillas_agrupada": sigma_loss, "varianza_val_loss_por_brazo": var_loss,
            "margen_val_loss": sigma_loss,
            "regla_1_top1": {"delta_por_pliegue_C2D_menos_T1": d_top1.tolist(), "C2D_gana": c2d_top1,
                             "T1_gana": t1_top1,
                             "empate": bool(not c2d_top1["cumple"] and not t1_top1["cumple"])},
            "regla_2_val_loss": {"aplica": bool(not c2d_top1["cumple"] and not t1_top1["cumple"]),
                                 "delta_por_pliegue_T1_menos_C2D": d_loss.tolist(), "C2D_gana": c2d_loss,
                                 "T1_gana": t1_loss},
            "resultado": resultado, "decidido_por": por,
            "modelo_final": ("TCN (D-14) se mantiene salvo decision del director" if resultado == "C2D supera a T1"
                             else "TCN (D-14) se mantiene"),
            "c2d_supera": resultado == "C2D supera a T1"}


def mcnemar_t1(pa, pb):
    """McNemar exacto de experimento_arquitecturas, con los nombres del control T1."""
    r = ea.mcnemar({c: v[:2] for c, v in pa.items()}, {c: v[:2] for c, v in pb.items()})
    return {"n": r["n"], "aciertos_C2D": r["aciertos_brazo"], "aciertos_T1": r["aciertos_R0"],
            "solo_C2D_acierta": r["solo_brazo_acierta"], "solo_T1_acierta": r["solo_R0_acierta"],
            "ambos_aciertan": r["ambos_aciertan"], "ninguno": r["ninguno"],
            "p_exacto_bilateral": r["p_exacto_bilateral"]}


# ---------------------------------------------------------------------------
# Tabla de familias (cifras de los informes previos, con su fuente)
# ---------------------------------------------------------------------------

def _media_matriz(entrada):
    M = np.array(list(entrada["por_semilla_por_pliegue"].values()), dtype=float)
    return float(M.mean()), int(M.shape[0])


def tabla_familias(por_brazo_actual: dict) -> list:
    """13 filas: 11 familias de los informes previos (la TCN medida con «mascara») + la TCN final (T1) y la CNN 2D,
    medidas con imputación. Validación y prueba top-1 = media de semillas × 5 pliegues LOSO."""
    lit = json.loads(FUENTE_LITERATURA.read_text(encoding="utf-8"))["por_brazo"]
    a2 = json.loads(FUENTE_ARQ2.read_text(encoding="utf-8"))["por_brazo"]
    lb = json.loads(FUENTE_LINEA_BASE.read_text(encoding="utf-8"))["loso"]["con_espejo"]
    filas = []
    ent_lb = "«mascara» + 252 -> 8 estadisticos por columna (2.016 rasgos); aumentacion espejo x2"
    for fam, nombre in (("random_forest", "Random Forest"), ("svm_rbf", "SVM RBF"), ("mlp", "MLP (1 capa de 256)")):
        val = float(np.mean([p["familias"][fam]["seleccion"]["validacion_elegida"]["top1"] for p in lb["pliegues"]]))
        filas.append({"familia": nombre, "brazo": fam, "entrada": ent_lb, "semillas": 1, "n_parametros": None,
                      "val_top1": val, "test_top1": float(lb["media_pliegues"][fam]["top1"]["media"]),
                      "test_f1_macro": float(lb["media_pliegues"][fam]["f1_macro"]["media"]),
                      "fuente": "models/linea_base_4d_2026-09-15/resultados.json (D-10, N-39)",
                      "nota": "validacion = la de la configuracion elegida por maximo de validacion en cada pliegue "
                              "(optimista); 1 semilla"})
    ent = "«mascara» + 252 (30 pasos, aumentacion x10)"
    for b, nombre, fuente, datos in (
            ("R0", "Conv1D + BiLSTM + atencion (canonico)", "docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.json", lit),
            ("R1", "c07 (Conv1D + BiLSTM, busqueda de hiperparametros)",
             "docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.json", lit),
            ("M", "MiniRocket + ridge", "docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.json", lit),
            ("T", "TCN pequena + GAP", "docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.json", lit),
            ("G", "ST-GCN pequeno", "docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.json", lit),
            ("X", "Transformer pequeno", "docs/EXPERIMENTO_ARQUITECTURAS_2.json", a2),
            ("I", "ResNet 1D", "docs/EXPERIMENTO_ARQUITECTURAS_2.json", a2),
            ("B", "BiGRU (c07 con GRU)", "docs/EXPERIMENTO_ARQUITECTURAS_2.json", a2)):
        v, ns = _media_matriz(datos[b]["val_top1"])
        t, _ = _media_matriz(datos[b]["test_top1"])
        f1, _ = _media_matriz(datos[b]["test_f1_macro"])
        filas.append({"familia": nombre, "brazo": b, "entrada": ent, "semillas": ns,
                      "n_parametros": datos[b].get("n_parametros"), "val_top1": v, "test_top1": t,
                      "test_f1_macro": f1, "fuente": f"{fuente} (por_brazo.{b})", "nota": ""})
    for b, nombre, fuente in (("T1", "TCN final (D-14): TCN pequena + GAP",
                               "models/confirmacion_tcn_2026-10-03/T1_imputacion_252 (este resumen)"),
                              ("C2D", "CNN 2D pequena (esqueleto como imagen)",
                               "models/cnn2d_2026-11-08/C2D_cnn2d (este experimento)")):
        r = por_brazo_actual[b]
        filas.append({"familia": nombre, "brazo": b, "entrada": "imputacion + 252 (30 pasos, aumentacion x10)",
                      "semillas": 3, "n_parametros": r["n_parametros"], "val_top1": r["val_top1"]["media_global"],
                      "test_top1": r["test_top1"]["media_global"], "test_f1_macro": r["test_f1_macro"]["media_global"],
                      "fuente": fuente, "nota": ""})
    return filas


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def leer_corridas(out):
    filas = []
    for b in COMPARADOS:
        for s in SEMILLAS:
            for k in range(5):
                d = dir_corrida(out, b, s, k)
                if not (d / "resultado.json").exists():
                    continue
                r = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
                pred = json.loads((d / "predicciones.json").read_text(encoding="utf-8"))
                filas.append({"brazo": b, "semilla": s, "pliegue": k, "senante_test": r["senante_test"],
                              "val_top1": r["val_top1"], "val_top1_checkpoint_reevaluado":
                              r["val_top1_checkpoint_reevaluado"], "val_loss": r["val_loss"],
                              "test_top1": r["test_top1"], "test_top3": r["test_top3"],
                              "test_f1_macro": r["test_f1_macro"], "n_parametros": r["n_parametros"],
                              "epocas_ejecutadas": r["epocas_ejecutadas"], "mejor_epoca": r["mejor_epoca"],
                              "tiempo_entrenamiento_s": r["tiempo_entrenamiento_s"],
                              "tiempo_total_s": r["tiempo_total_s"], "s_por_epoca": r["s_por_epoca"],
                              "predicciones": {x["muestra"]: (x["real"], x["predicha"], x["prob_predicha"])
                                               for x in pred}, "fuente": tl._rel(d)})
    return filas


def cmd_resumen(args):
    out = Path(args.out).resolve()
    filas = leer_corridas(out)
    folds = list(range(5))

    def celda(b, s, k):
        r = [f for f in filas if f["brazo"] == b and f["semilla"] == s and f["pliegue"] == k]
        return r[0] if r else None

    faltan = [(b, s, k) for b in COMPARADOS for s in SEMILLAS for k in folds if celda(b, s, k) is None]
    if faltan:
        raise SystemExit(f"faltan corridas: {faltan}")
    coherencia = [(f["brazo"], f["semilla"], f["pliegue"]) for f in filas
                  if abs(f["val_top1"] - f["val_top1_checkpoint_reevaluado"]) > 1e-6]
    METR = ["val_top1", "val_loss", "test_top1", "test_top3", "test_f1_macro", "epocas_ejecutadas", "mejor_epoca",
            "tiempo_entrenamiento_s", "tiempo_total_s", "s_por_epoca"]
    por_brazo = {}
    for b in COMPARADOS:
        r = {"n_parametros": celda(b, 2026, 0)["n_parametros"]}
        for m in METR:
            M = np.array([[celda(b, s, k)[m] for k in folds] for s in SEMILLAS], dtype=float)
            r[m] = {"por_semilla_por_pliegue": {str(s): M[i].tolist() for i, s in enumerate(SEMILLAS)},
                    "por_pliegue_media_semillas": M.mean(axis=0).tolist(),
                    "por_semilla_media_pliegues": {str(s): float(M[i].mean()) for i, s in enumerate(SEMILLAS)},
                    "de_entre_pliegues": float(np.std(M.mean(axis=0), ddof=1)),
                    "de_entre_semillas": float(np.std(M.mean(axis=1), ddof=1)),
                    "media_global": float(M.mean())}
        por_brazo[b] = r

    val = {b: np.array([[celda(b, s, k)["val_top1"] for k in folds] for s in SEMILLAS]) for b in COMPARADOS}
    loss = {b: np.array([[celda(b, s, k)["val_loss"] for k in folds] for s in SEMILLAS]) for b in COMPARADOS}
    criterio = aplicar_criterio(val, loss)

    pruebas = {}
    for m in ("val_top1", "val_loss", "test_top1", "test_top3", "test_f1_macro"):
        dif = (np.array(por_brazo["C2D"][m]["por_pliegue_media_semillas"])
               - np.array(por_brazo["T1"][m]["por_pliegue_media_semillas"]))
        pruebas[m] = ea.pruebas_pareadas(dif)
    pruebas["nota"] = ("diferencias C2D - T1 por pliegue (media de 3 semillas); en val_loss un valor negativo "
                       "favorece a C2D. Con 5 pliegues el p bilateral minimo es 2/2^5 = 0,0625.")

    def agregadas(b, s):
        p = {}
        for k in folds:
            p.update(celda(b, s, k)["predicciones"])
        return p

    agg = {b: {s: agregadas(b, s) for s in SEMILLAS} for b in COMPARADOS}
    voto = {b: ect.voto_semillas([agg[b][s] for s in SEMILLAS]) for b in COMPARADOS}
    for b in COMPARADOS:
        assert len(voto[b]) == 997, (b, len(voto[b]))
    mcnemar = {str(s): mcnemar_t1(agg["C2D"][s], agg["T1"][s]) for s in SEMILLAS}
    mcnemar["voto_3_semillas"] = mcnemar_t1({c: v + (1.0,) for c, v in voto["C2D"].items()},
                                            {c: v + (1.0,) for c, v in voto["T1"].items()})
    exact_voto = {b: float(np.mean([v[0] == v[1] for v in voto[b].values()])) for b in COMPARADOS}

    p0000 = {b: {m: {"media": por_brazo[b][m]["por_pliegue_media_semillas"][0],
                     "por_semilla": {s: v[0] for s, v in por_brazo[b][m]["por_semilla_por_pliegue"].items()}}
                 for m in ("val_top1", "val_loss", "test_top1", "test_top3", "test_f1_macro")} for b in COMPARADOS}

    familias = tabla_familias(por_brazo)
    extra = {}
    for nombre in ("verificacion.json", "inferencia.json", "humo.json"):
        if (out / nombre).exists():
            extra[nombre] = json.loads((out / nombre).read_text(encoding="utf-8"))
    sin_pred = [{k: v for k, v in f.items() if k != "predicciones"} for f in filas]
    resultado = {"experimento": "cierre: CNN 2D pequena frente a la TCN final (docs/EXPERIMENTO_CNN2D.md)",
                 "fecha_resumen": datetime.now().isoformat(timespec="seconds"), "brazos": BRAZOS,
                 "config_C2D": ac.CONFIG_C2D, "entrenamiento_C2D": ac.ENTRENAMIENTO_C2D, "config_T": al.CONFIG_T,
                 "semillas": SEMILLAS, "criterio": criterio, "pruebas_pareadas_por_pliegue": pruebas,
                 "mcnemar_C2D_vs_T1": mcnemar, "exactitud_voto_3_semillas": exact_voto, "persona_0000": p0000,
                 "por_brazo": por_brazo, "tabla_familias": familias,
                 "incoherencias_val_top1_vs_reevaluado": coherencia,
                 "verificacion": extra.get("verificacion.json"), "humo": extra.get("humo.json"),
                 "inferencia": extra.get("inferencia.json"), "corridas": sin_pred}
    (out / "resultados_cnn2d.json").write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.docs:
        JSON_DOCS.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"escrito {tl._rel(JSON_DOCS)}")

    print(f"incoherencias val vs reevaluado: {coherencia}")
    for b in COMPARADOS:
        r = por_brazo[b]
        print(f"\n{b} | {r['n_parametros']:,} param | voto3 {exact_voto[b]:.4f}")
        for m in METR:
            print(f"   {m:24s} {r[m]['media_global']:.4f} (DE pl {r[m]['de_entre_pliegues']:.4f}, DE sem "
                  f"{r[m]['de_entre_semillas']:.4f}) {[round(x, 3) for x in r[m]['por_pliegue_media_semillas']]} "
                  f"sem {[round(x, 4) for x in r[m]['por_semilla_media_pliegues'].values()]}")
    print("\nCRITERIO:", json.dumps(criterio, indent=1, ensure_ascii=False))
    for m, pr in pruebas.items():
        if m == "nota":
            continue
        print(f"  {m}: {[round(x, 4) for x in pr['diferencias']]} +{pr['mejora']} -{pr['empeora']} ={pr['empate']} "
              f"| signos {pr['signos_p_bilateral']:.4f} | wilcoxon {pr['wilcoxon_p_bilateral']:.4f}")
    for s, m in mcnemar.items():
        print(f"  McNemar {s}: C2D {m['aciertos_C2D']} vs T1 {m['aciertos_T1']} | solo C2D {m['solo_C2D_acierta']} "
              f"solo T1 {m['solo_T1_acierta']} | p {m['p_exacto_bilateral']:.4g}")
    print("persona 0000:", {b: {m: round(v["media"], 3) for m, v in p0000[b].items()} for b in COMPARADOS})
    print("\nFAMILIAS:")
    for f in familias:
        print(f"  {f['familia'][:45]:45s} val {f['val_top1']:.3f} prueba {f['test_top1']:.3f} F1 "
              f"{f['test_f1_macro']:.3f} | {f['n_parametros']} | {f['entrada'][:30]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verificar")
    v.add_argument("--dest", default=None, help="carpeta (carpeta_temporal) para reproducir T1 s2026 k0")
    e = sub.add_parser("entrenar")
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
