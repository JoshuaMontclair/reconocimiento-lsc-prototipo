"""
Tabla exacta de capas del modelo canónico (diapositivas del flujo, pieza «modelo»).

Carga models/canonico_holistic_mascara_manos/modelo_canonico.keras (N-31), recorre sus capas y
escribe nombre, tipo, forma de salida y parámetros (entrenables y no entrenables) de cada una.
Después construye build_model(30, 252, 50) de src/train_final_100.py y comprueba que la secuencia de
tipos, las formas de salida y los recuentos de parámetros coinciden.

Solo lee. Escribe únicamente en docs/diapositivas_workflow/figuras/tabla_capas_modelo.json.

Uso (desde la raíz del repositorio):
    python docs/diapositivas_workflow/guiones/tabla_capas_modelo.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.train_final_100 import build_model  # noqa: E402

MODELO = RAIZ / "models" / "canonico_holistic_mascara_manos" / "modelo_canonico.keras"
SALIDA = RAIZ / "docs" / "diapositivas_workflow" / "figuras" / "tabla_capas_modelo.json"


def contar(pesos):
    return int(sum(int(np.prod(w.shape)) for w in pesos))


def tabla(model):
    filas = []
    for capa in model.layers:
        forma = capa.output.shape if hasattr(capa, "output") else None
        filas.append({
            "nombre": capa.name,
            "tipo": capa.__class__.__name__,
            "forma_salida": [None if d is None else int(d) for d in forma] if forma is not None else None,
            "param_entrenables": contar(capa.trainable_weights),
            "param_no_entrenables": contar(capa.non_trainable_weights),
            "entradas": [t.name if hasattr(t, "name") else str(t) for t in
                         (capa.input if isinstance(capa.input, list) else [capa.input])]
            if capa.__class__.__name__ != "InputLayer" else [],
        })
    return filas


def main():
    try:
        m = tf.keras.models.load_model(str(MODELO))
    except ValueError:
        m = tf.keras.models.load_model(str(MODELO), safe_mode=False)
    t_can = tabla(m)
    tot = {
        "entrenables": contar(m.trainable_weights),
        "no_entrenables": contar(m.non_trainable_weights),
    }
    tot["total"] = tot["entrenables"] + tot["no_entrenables"]

    ref = build_model(30, 252, 50)
    t_ref = tabla(ref)
    coincide = (
        len(t_can) == len(t_ref)
        and all(a["tipo"] == b["tipo"] and a["forma_salida"] == b["forma_salida"]
                and a["param_entrenables"] == b["param_entrenables"]
                and a["param_no_entrenables"] == b["param_no_entrenables"]
                for a, b in zip(t_can, t_ref))
        and contar(ref.trainable_weights) == tot["entrenables"]
        and contar(ref.non_trainable_weights) == tot["no_entrenables"]
    )
    # Configuración declarada de las capas clave (para citar en las diapositivas).
    config = {}
    for capa in m.layers:
        c = capa.get_config()
        tipo = capa.__class__.__name__
        if tipo == "Conv1D":
            config[capa.name] = {k: c[k] for k in ("filters", "kernel_size", "padding", "activation")}
        elif tipo == "Bidirectional":
            inner = c["layer"]["config"]
            config[capa.name] = {"units": inner["units"], "return_sequences": inner["return_sequences"],
                                 "activation": inner["activation"],
                                 "recurrent_activation": inner["recurrent_activation"],
                                 "merge_mode": c.get("merge_mode")}
        elif tipo == "Dense":
            config[capa.name] = {"units": c["units"], "activation": c["activation"]}
        elif tipo == "Dropout":
            config[capa.name] = {"rate": c["rate"]}
        elif tipo == "Softmax":
            config[capa.name] = {"axis": c["axis"]}
    salida = {
        "modelo": str(MODELO.relative_to(RAIZ)),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else None,
        "entrada": [None if d is None else int(d) for d in m.input.shape],
        "salida": [None if d is None else int(d) for d in m.output.shape],
        "capas": t_can,
        "totales": tot,
        "config_capas_clave": config,
        "coincide_con_build_model_30_252_50": bool(coincide),
    }
    SALIDA.write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
    for f in t_can:
        print(f"{f['nombre']:28s} {f['tipo']:24s} {str(f['forma_salida']):18s} "
              f"{f['param_entrenables']:>9,d} {f['param_no_entrenables']:>6,d}")
    print("totales", tot, "| coincide con build_model(30,252,50):", coincide)
    print("->", SALIDA)


if __name__ == "__main__":
    main()
