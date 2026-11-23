"""
Umbral de confianza del prototipo (D-09; §2.2.6 del anteproyecto).

El valor de corte bajo el cual el prototipo responde «seña no reconocida» se elige sobre VALIDACIÓN:
las predicciones de los modelos LOSO de la representación adoptada sobre sus conjuntos «val», que se
recalculan aquí con el mismo preprocesamiento del entrenador (cargar_datos, remuestrear_todas,
add_velocity de src/train_loso.py) y que deben reproducir el mejor_val_accuracy guardado en cada
pliegue antes de usarse. Nunca se usa el conjunto de prueba ni los ensayos en vivo para elegir.

Criterio (fijado en D-09 antes de calcular):
    U(t) = aciertos aceptados − c · errores aceptados,  c = 1,
    aceptada si confianza >= t (la misma comparación que LSCTranslator), confianza = máximo del softmax,
    rejilla t = 0,00 … 0,99; con empate, el menor t.
Se reporta, para el t elegido y para 0,5: tasa de rechazo, exactitud entre aceptadas y proporción de
aciertos perdidos por rechazo, en la validación de selección, en la prueba LOSO (señantes nuevos, solo
reporte) y en la validación y la prueba del modelo canónico; y la sensibilidad a c = 0,5 y c = 2.

Salidas: <modelo canónico>.umbral.json (lo lee el prototipo), models/umbral_D09_<fecha>.json con todas
las curvas, y models/umbral_D09_<fecha>.png.

Uso:
    python src/umbral_confianza.py
    python src/umbral_confianza.py --loso-dir models/loso_2026-09-15_holistic_mascara_manos \\
        --canonico models/canonico_holistic_mascara_manos/modelo_canonico.keras --c 1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

REJILLA = np.round(np.arange(0.0, 1.0, 0.01), 2)


def _rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(BASE_DIR)) if p.is_relative_to(BASE_DIR) else str(p)


def utilidad(conf: np.ndarray, ok: np.ndarray, t: float, c: float) -> float:
    acept = conf >= t
    return float(np.sum(ok & acept) - c * np.sum(~ok & acept))


def elegir_umbral(conf: np.ndarray, ok: np.ndarray, c: float) -> tuple[float, list[float]]:
    """Menor t de la rejilla con U(t) máxima."""
    curva = [utilidad(conf, ok, t, c) for t in REJILLA]
    return float(REJILLA[int(np.argmax(curva))]), curva       # argmax devuelve el primer máximo


def efecto(conf: np.ndarray, ok: np.ndarray, t: float) -> dict:
    acept = conf >= t
    n_ok = int(ok.sum())
    return {
        "n": int(len(conf)),
        "exactitud_sin_umbral": float(ok.mean()) if len(ok) else None,
        "tasa_rechazo": float(1 - acept.mean()) if len(conf) else None,
        "exactitud_entre_aceptadas": float(ok[acept].mean()) if acept.any() else None,
        "aciertos_perdidos_por_rechazo": float((ok & ~acept).sum() / n_ok) if n_ok else None,
        "errores_evitados_por_rechazo": float((~ok & ~acept).sum() / max(1, int((~ok).sum()))),
        "aceptadas_correctas": int((ok & acept).sum()),
        "aceptadas_incorrectas": int((~ok & acept).sum()),
    }


def _cargar_modelo(tf, ruta):
    try:
        return tf.keras.models.load_model(str(ruta))
    except ValueError:
        return tf.keras.models.load_model(str(ruta), safe_mode=False)


def predicciones_validacion(tf, ruta_modelo: Path, nombres_val: list[str], cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Confianza y acierto del modelo sobre las muestras nombradas, con su procedencia."""
    from src.representacion import leer_procedencia
    from src.train_final_100 import add_velocity, resample_sequence
    from src.train_loso import cargar_datos, remuestrear_todas
    proc = leer_procedencia(ruta_modelo)
    if proc is None:
        raise SystemExit(f"{_rel(ruta_modelo)} no tiene procedencia: no se sabe con qué datos predecir")
    clave = (proc["processed_dir"], proc["representacion"], int(proc["seq_length"]))
    if clave not in cache:
        X_padded, y, lengths, nombres, _ = cargar_datos(BASE_DIR / proc["processed_dir"], proc["representacion"])
        cache[clave] = (remuestrear_todas(X_padded, lengths, resample_sequence, int(proc["seq_length"])), y,
                        {n: i for i, n in enumerate(nombres)})
    X_seq, y, pos = cache[clave]
    idx = np.array([pos[n] for n in nombres_val])
    modelo = _cargar_modelo(tf, ruta_modelo)
    prob = modelo.predict(add_velocity(X_seq[idx]), verbose=0, batch_size=256)
    tf.keras.backend.clear_session()
    return prob.max(axis=1), prob.argmax(axis=1) == y[idx]


def desde_predicciones(ruta: Path) -> tuple[np.ndarray, np.ndarray]:
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return (np.array([r["prob_predicha"] for r in datos]), np.array([r["predicha"] == r["real"] for r in datos]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Umbral de confianza del prototipo por validación (D-09)")
    ap.add_argument("--loso-dir", default="models/loso_2026-09-15_holistic_mascara_manos")
    ap.add_argument("--canonico", default="models/canonico_holistic_mascara_manos/modelo_canonico.keras")
    ap.add_argument("--c", type=float, default=1.0, help="costo de un error aceptado frente a un acierto (D-09: 1)")
    ap.add_argument("--salida", default=None, help="prefijo de salida (defecto models/umbral_D09_<fecha>)")
    args = ap.parse_args(argv)

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    loso_dir = BASE_DIR / args.loso_dir
    canonico = BASE_DIR / args.canonico
    fecha = dt.date.today().isoformat()
    prefijo = BASE_DIR / (args.salida or f"models/umbral_D09_{fecha}")
    cache: dict = {}

    # (a) Validación de selección: 5 pliegues LOSO, con control de reproducción.
    loso = json.loads((BASE_DIR / "data" / "partitions" / "loso_5_pliegues.json").read_text(encoding="utf-8"))
    conf_val, ok_val, control = [], [], []
    for p in loso["pliegues"]:
        k = p["pliegue"]
        carpeta = loso_dir / f"pliegue_{k}"
        conf, ok = predicciones_validacion(tf, carpeta / "modelo.keras", p["conjuntos"]["val"], cache)
        guardada = json.loads((carpeta / "metricas.json").read_text(encoding="utf-8"))["mejor_val_accuracy"]
        reproducida = float(ok.mean())
        control.append({"pliegue": k, "n_val": int(len(ok)), "val_guardada": guardada, "val_reproducida": reproducida,
                        "coincide": abs(reproducida - guardada) < 1e-6})
        print(f"  pliegue {k}: val reproducida {reproducida:.4f} | guardada {guardada:.4f}")
        conf_val.append(conf); ok_val.append(ok)
    if not all(x["coincide"] for x in control):
        raise SystemExit("las predicciones de validación NO reproducen el mejor_val_accuracy guardado: no se elige umbral")
    conf_val, ok_val = np.concatenate(conf_val), np.concatenate(ok_val)

    t_elegido, curva = elegir_umbral(conf_val, ok_val, args.c)
    sensibilidad = {str(c): elegir_umbral(conf_val, ok_val, c)[0] for c in (0.5, 1.0, 2.0)}
    print(f"\n  umbral elegido (c = {args.c}): {t_elegido:.2f} | sensibilidad {sensibilidad}")

    # (b) Prueba LOSO (señantes nuevos) y (c) canónico: solo reporte.
    conf_te = np.concatenate([desde_predicciones(loso_dir / f"pliegue_{k}" / "predicciones.json")[0] for k in range(5)])
    ok_te = np.concatenate([desde_predicciones(loso_dir / f"pliegue_{k}" / "predicciones.json")[1] for k in range(5)])
    aleatoria = json.loads((BASE_DIR / "data" / "partitions" / "aleatoria_80_10_10.json").read_text(encoding="utf-8"))
    conf_cv, ok_cv = predicciones_validacion(tf, canonico, aleatoria["conjuntos"]["val"], cache)
    res_can = json.loads((canonico.parent / "resultados.json").read_text(encoding="utf-8"))
    val_can_guardada = res_can["seleccion"]["val_accuracy_checkpoint"]
    conf_ct, ok_ct = desde_predicciones(canonico.parent / "evaluacion_modelo_seleccionado" / "predicciones.json")

    conjuntos = {
        "validacion_loso_seleccion": (conf_val, ok_val),
        "prueba_loso_senantes_nuevos": (conf_te, ok_te),
        "validacion_canonico": (conf_cv, ok_cv),
        "prueba_canonico": (conf_ct, ok_ct),
    }
    informe = {
        "decision": "D-09",
        "criterio": "U(t) = aciertos aceptados - c * errores aceptados; aceptada si confianza >= t; rejilla 0,00-0,99; empate -> menor t",
        "c": args.c,
        "umbral_elegido": t_elegido,
        "sensibilidad_c": sensibilidad,
        "control_reproduccion_validacion": control,
        "control_validacion_canonico": {"guardada": val_can_guardada, "reproducida": float(ok_cv.mean()),
                                        "coincide": abs(float(ok_cv.mean()) - val_can_guardada) < 1e-6},
        "efecto": {nombre: {"umbral_elegido": efecto(c_, o_, t_elegido), "umbral_0.5": efecto(c_, o_, 0.5)}
                   for nombre, (c_, o_) in conjuntos.items()},
        "curvas": {"t": REJILLA.tolist(), "utilidad_validacion": curva,
                   **{f"rechazo_{n}": [float(1 - (c_ >= t).mean()) for t in REJILLA] for n, (c_, _) in conjuntos.items()},
                   **{f"exactitud_aceptadas_{n}": [float(o_[c_ >= t].mean()) if (c_ >= t).any() else None for t in REJILLA]
                      for n, (c_, o_) in conjuntos.items()}},
        "modelos": {"loso": _rel(loso_dir), "canonico": _rel(canonico)},
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "script": "src/umbral_confianza.py",
    }
    Path(f"{prefijo}.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")

    destino = canonico.with_name(canonico.stem + ".umbral.json")
    import hashlib
    destino.write_text(json.dumps({
        "umbral": t_elegido, "decision": "D-09", "c": args.c,
        "seleccionado_con": f"validacion de {_rel(loso_dir)} (5 pliegues, {len(conf_val)} muestras)",
        "modelo": canonico.name,
        "sha256_modelo": hashlib.sha256(canonico.read_bytes()).hexdigest(),
        "script": "src/umbral_confianza.py",
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
        for etiqueta, t in (("elegido", t_elegido), ("0,5", 0.5)):
            e = informe["efecto"][nombre]["umbral_elegido" if etiqueta == "elegido" else "umbral_0.5"]
            print(f"  {nombre:30s} {t:5.2f} {e['tasa_rechazo']:8.3f} {e['exactitud_entre_aceptadas'] or 0:17.3f} "
                  f"{e['aciertos_perdidos_por_rechazo'] or 0:18.3f}")
    print(f"\n  escrito: {_rel(destino)}, {_rel(f'{prefijo}.json')}, {_rel(f'{prefijo}.png')}")


if __name__ == "__main__":
    main()
