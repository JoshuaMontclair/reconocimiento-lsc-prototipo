"""
Umbral de confianza (D-09) del modelo EXPERIMENTAL TCN + imputación + 252 (models/tcn_imputacion/modelo.keras,
src/train_final_tcn.py; N-63).

Por qué un script aparte
------------------------
src/umbral_confianza.py fija el criterio de D-09 pero no se puede correr tal cual con este modelo:
  · recalcula las predicciones de validación con train_loso.remuestrear_todas, es decir, con la regla
    «mascara»; este modelo y sus modelos LOSO esperan la entrada «imputacion» (mano_ausente_variantes);
  · espera <loso-dir>/pliegue_k/{modelo.keras, metricas.json}; los modelos LOSO de esta configuración
    (brazo T1 de docs/EXPERIMENTO_CONFIRMACION_TCN.md) están en
    models/confirmacion_tcn_2026-10-03/T1_imputacion_252/semilla_<s>/pliegue_k/ y guardan la validación en
    resultado.json (val_top1) y las probabilidades en probabilidades.npz;
  · escribe <modelo>.umbral.json junto al modelo y el informe en models/umbral_D09_<fecha>.*.
Por eso este script REPLICA exactamente su criterio importando sus funciones (REJILLA, utilidad,
elegir_umbral, efecto, desde_predicciones): solo cambian la entrada con que se recalculan las predicciones
y las rutas.

Criterio (D-09, sin cambios)
----------------------------
    U(t) = aciertos aceptados − c · errores aceptados, c = 1; aceptada si confianza >= t (la comparación de
    LSCTranslator); confianza = máximo del softmax; rejilla t = 0,00 … 0,99; empate -> el menor t.
Datos de selección: la VALIDACIÓN de los 5 modelos LOSO de esta misma configuración (TCN + imputación + 252,
semilla 2026, como el D-09 del canónico usó una corrida LOSO), 500 muestras. Las predicciones se recalculan
cargando cada modelo sobre la validación de su pliegue con la entrada «imputacion» y, antes de usarse, deben
reproducir el val_top1 guardado y las probabilidades de validación guardadas en probabilidades.npz.
Nunca se usa un conjunto de prueba ni ensayos en vivo para elegir. Se reporta además (solo reporte) el
efecto en la prueba LOSO (señantes nuevos), en la validación y en la prueba del esquema aleatorio del modelo
final, y la sensibilidad a c = 0,5 y c = 2.

Salidas: models/tcn_imputacion/modelo.umbral.json (lo lee el prototipo; incluye el sha256 del modelo) y
models/tcn_imputacion/umbral_D09_<fecha>.{json,png}.

Uso:
    python src/umbral_tcn_imputacion.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.umbral_confianza import (  # noqa: E402  (criterio de D-09: se reutiliza, no se reimplementa)
    REJILLA, _cargar_modelo, _rel, desde_predicciones, efecto, elegir_umbral,
)

LOSO_DIR = "models/confirmacion_tcn_2026-10-03/T1_imputacion_252/semilla_2026"
MODELO = "models/tcn_imputacion/modelo.keras"
REGLA = "imputacion"


def preparar_entrada(processed_rel: str, representacion: str, seq_length: int, cache: dict):
    """X_seq con la regla «imputacion» (la de train_final_tcn.construir_X_seq y del brazo T1), y y pos."""
    from src import mano_ausente_variantes as mav
    from src.train_loso import cargar_datos
    clave = (processed_rel, representacion, seq_length)
    if clave not in cache:
        X_padded, y, lengths, nombres, _ = cargar_datos(BASE_DIR / processed_rel, representacion)
        cache[clave] = (mav.remuestrear_todas_con_regla(X_padded, lengths, REGLA, seq_length), y,
                        {n: i for i, n in enumerate(nombres)})
    return cache[clave]


def probabilidades(tf, ruta_modelo: Path, nombres: list[str], processed_rel: str, representacion: str,
                   seq_length: int, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Softmax del modelo sobre las muestras nombradas, con la entrada «imputacion» + add_velocity."""
    from src.train_final_100 import add_velocity
    X_seq, y, pos = preparar_entrada(processed_rel, representacion, seq_length, cache)
    idx = np.array([pos[n] for n in nombres])
    modelo = _cargar_modelo(tf, ruta_modelo)
    prob = modelo.predict(add_velocity(X_seq[idx]), verbose=0, batch_size=256)
    tf.keras.backend.clear_session()
    return prob, y[idx]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Umbral D-09 del modelo TCN + imputación + 252 (criterio de "
                                             "src/umbral_confianza.py)")
    ap.add_argument("--loso-dir", default=LOSO_DIR)
    ap.add_argument("--modelo", default=MODELO)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--salida", default=None, help="prefijo del informe (defecto models/tcn_imputacion/umbral_D09_<fecha>)")
    args = ap.parse_args(argv)

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    loso_dir = BASE_DIR / args.loso_dir
    modelo = BASE_DIR / args.modelo
    fecha = dt.date.today().isoformat()
    prefijo = BASE_DIR / (args.salida or f"{Path(args.modelo).parent}/umbral_D09_{fecha}")
    cache: dict = {}

    # (a) Validación de selección: 5 pliegues LOSO de la misma configuración, con control de reproducción.
    loso = json.loads((BASE_DIR / "data" / "partitions" / "loso_5_pliegues.json").read_text(encoding="utf-8"))
    conf_val, ok_val, control = [], [], []
    for p in loso["pliegues"]:
        k = p["pliegue"]
        carpeta = loso_dir / f"pliegue_{k}"
        res = json.loads((carpeta / "resultado.json").read_text(encoding="utf-8"))
        proc = json.loads((carpeta / "modelo.procedencia.json").read_text(encoding="utf-8"))
        assert res["regla_mano_ausente"] == REGLA and res["n_rasgos_entrada"] == 252, carpeta
        assert proc["mano_ausente"].startswith(REGLA) and proc["representacion"] == "manos", carpeta
        nombres_val = list(p["conjuntos"]["val"])
        prob, y_val = probabilidades(tf, carpeta / "modelo.keras", nombres_val, proc["processed_dir"],
                                     proc["representacion"], int(proc["seq_length"]), cache)
        guardadas = np.load(carpeta / "probabilidades.npz")
        assert list(guardadas["nombres_val"]) == nombres_val, carpeta
        dif = float(np.max(np.abs(prob - guardadas["val"])))
        ok = prob.argmax(axis=1) == y_val
        reproducida = float(ok.mean())
        control.append({"pliegue": k, "n_val": int(len(ok)), "val_guardada": res["val_top1"],
                        "val_reproducida": reproducida, "max_dif_prob_vs_npz": dif,
                        "coincide": abs(reproducida - res["val_top1"]) < 1e-6 and dif < 1e-4})
        print(f"  pliegue {k}: val reproducida {reproducida:.4f} | guardada {res['val_top1']:.4f} | "
              f"max |dif| prob vs npz {dif:.2e}")
        conf_val.append(prob.max(axis=1)); ok_val.append(ok)
    if not all(x["coincide"] for x in control):
        raise SystemExit("las predicciones de validación NO reproducen lo guardado: no se elige umbral")
    conf_val, ok_val = np.concatenate(conf_val), np.concatenate(ok_val)

    t_elegido, curva = elegir_umbral(conf_val, ok_val, args.c)
    sensibilidad = {str(c): elegir_umbral(conf_val, ok_val, c)[0] for c in (0.5, 1.0, 2.0)}
    print(f"\n  umbral elegido (c = {args.c}): {t_elegido:.2f} | sensibilidad {sensibilidad}")

    # (b) Prueba LOSO y (c) modelo final del esquema aleatorio: solo reporte.
    pred_te = [desde_predicciones(loso_dir / f"pliegue_{k}" / "predicciones.json") for k in range(5)]
    conf_te, ok_te = np.concatenate([a for a, _ in pred_te]), np.concatenate([b for _, b in pred_te])
    aleatoria = json.loads((BASE_DIR / "data" / "partitions" / "aleatoria_80_10_10.json").read_text(encoding="utf-8"))
    proc_f = json.loads(modelo.with_name(modelo.stem + ".procedencia.json").read_text(encoding="utf-8"))
    assert proc_f["mano_ausente"] == REGLA
    prob_fv, y_fv = probabilidades(tf, modelo, list(aleatoria["conjuntos"]["val"]), proc_f["processed_dir"],
                                   proc_f["representacion"], int(proc_f["seq_length"]), cache)
    conf_fv, ok_fv = prob_fv.max(axis=1), prob_fv.argmax(axis=1) == y_fv
    res_f = json.loads((modelo.parent / "resultados.json").read_text(encoding="utf-8"))
    val_f_guardada = res_f["seleccion"]["val_accuracy_checkpoint"]
    conf_ft, ok_ft = desde_predicciones(modelo.parent / "evaluacion_modelo_seleccionado" / "predicciones.json")

    conjuntos = {
        "validacion_loso_seleccion": (conf_val, ok_val),
        "prueba_loso_senantes_nuevos": (conf_te, ok_te),
        "validacion_modelo_final": (conf_fv, ok_fv),
        "prueba_modelo_final": (conf_ft, ok_ft),
    }
    informe = {
        "decision": "D-09",
        "criterio": "U(t) = aciertos aceptados - c * errores aceptados; aceptada si confianza >= t; rejilla 0,00-0,99; empate -> menor t",
        "criterio_de": "src/umbral_confianza.py (REJILLA, utilidad, elegir_umbral, efecto importados sin cambios)",
        "adaptacion": "predicciones recalculadas con la entrada «imputacion» (mano_ausente_variantes) en vez de "
                      "train_loso.remuestrear_todas; validación guardada en resultado.json (val_top1) y "
                      "probabilidades.npz de los modelos LOSO T1 de docs/EXPERIMENTO_CONFIRMACION_TCN.md",
        "c": args.c,
        "umbral_elegido": t_elegido,
        "sensibilidad_c": sensibilidad,
        "control_reproduccion_validacion": control,
        "control_validacion_modelo_final": {"guardada": val_f_guardada, "reproducida": float(ok_fv.mean()),
                                            "coincide": abs(float(ok_fv.mean()) - val_f_guardada) < 1e-6},
        "efecto": {nombre: {"umbral_elegido": efecto(c_, o_, t_elegido), "umbral_0.5": efecto(c_, o_, 0.5),
                            "umbral_0.2_del_canonico": efecto(c_, o_, 0.2)}
                   for nombre, (c_, o_) in conjuntos.items()},
        "curvas": {"t": REJILLA.tolist(), "utilidad_validacion": curva,
                   **{f"rechazo_{n}": [float(1 - (c_ >= t).mean()) for t in REJILLA] for n, (c_, _) in conjuntos.items()},
                   **{f"exactitud_aceptadas_{n}": [float(o_[c_ >= t].mean()) if (c_ >= t).any() else None for t in REJILLA]
                      for n, (c_, o_) in conjuntos.items()}},
        "modelos": {"loso": _rel(loso_dir), "modelo_final": _rel(modelo)},
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "src/umbral_tcn_imputacion.py",
    }
    if not informe["control_validacion_modelo_final"]["coincide"]:
        raise SystemExit("la validación recalculada del modelo final no reproduce resultados.json")
    Path(f"{prefijo}.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")

    destino = modelo.with_name(modelo.stem + ".umbral.json")
    destino.write_text(json.dumps({
        "umbral": t_elegido, "decision": "D-09", "c": args.c,
        "seleccionado_con": f"validacion de {_rel(loso_dir)} (5 pliegues, {len(conf_val)} muestras; "
                            f"TCN + imputacion + 252, misma configuracion que el modelo)",
        "modelo": modelo.name,
        "sha256_modelo": hashlib.sha256(modelo.read_bytes()).hexdigest(),
        "script": "src/umbral_tcn_imputacion.py (criterio de src/umbral_confianza.py)",
        "informe": _rel(f"{prefijo}.json"), "fecha": informe["fecha"],
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(REJILLA, curva); ax[0].axvline(t_elegido, ls="--", c="k"); ax[0].set_title(f"U(t), validación (c = {args.c})")
    for n in conjuntos:
        ax[1].plot(REJILLA, informe["curvas"][f"rechazo_{n}"], label=n)
        ax[2].plot(REJILLA, [np.nan if v is None else v for v in informe["curvas"][f"exactitud_aceptadas_{n}"]], label=n)
    for a, tit in ((ax[1], "tasa de rechazo"), (ax[2], "exactitud entre aceptadas")):
        a.axvline(t_elegido, ls="--", c="k"); a.axvline(0.5, ls=":", c="gray"); a.set_title(tit); a.set_xlabel("umbral t")
    ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{prefijo}.png", dpi=130); plt.close(fig)

    print(f"\n  {'conjunto':30s} {'t':>5s} {'rechazo':>8s} {'exact. aceptadas':>17s} {'aciertos perdidos':>18s}")
    for nombre in conjuntos:
        for clave, t in (("umbral_elegido", t_elegido), ("umbral_0.5", 0.5)):
            e = informe["efecto"][nombre][clave]
            print(f"  {nombre:30s} {t:5.2f} {e['tasa_rechazo']:8.3f} {e['exactitud_entre_aceptadas'] or 0:17.3f} "
                  f"{e['aciertos_perdidos_por_rechazo'] or 0:18.3f}")
    print(f"\n  escrito: {_rel(destino)}, {_rel(f'{prefijo}.json')}, {_rel(f'{prefijo}.png')}")


if __name__ == "__main__":
    main()
