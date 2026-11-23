"""
Prueba de escritorio de la RED con un video real del conjunto de prueba (pieza «modelo»).

Toma una muestra del conjunto de PRUEBA del esquema aleatorio (nunca vista en entrenamiento ni en
validación), construye su entrada exactamente como el entrenador (src/train_final_100.py:
cargar_datos(processed, "manos") -> resample_sequence a 30 pasos -> add_velocity) y la pasa por
models/canonico_holistic_mascara_manos/modelo_canonico.keras. Extrae:
  · la forma del tensor en cada bloque (sub-modelos de Keras sobre las capas reales);
  · los 30 pesos de la atención (salida de la capa «softmax», eje del tiempo);
  · las 5 señas más probables de la salida softmax.
Comprueba que la probabilidad obtenida coincide con la guardada en
evaluacion_modelo_seleccionado/predicciones.json (si no, falla).

Solo lee. Escribe figuras/pasada_modelo_<muestra>.pdf/.png y .json.
Uso: python docs/diapositivas_workflow/guiones/pasada_modelo_un_video.py [SSSS_VVVV_RRRR]
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
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.train_final_100 import cargar_datos, resample_sequence, add_velocity  # noqa: E402
from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402

DIR = RAIZ / "models" / "canonico_holistic_mascara_manos"
PROCESSED = RAIZ / "data" / "processed_holistic_mascara_rostro"
FIG = RAIZ / "docs" / "diapositivas_workflow" / "figuras"
AZUL, GRIS = "#2a78d6", "#b9b8b2"
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def c(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def main():
    muestra = sys.argv[1] if len(sys.argv) > 1 else "0000_0003_0002"
    proc = json.loads((DIR / "modelo_canonico.procedencia.json").read_text(encoding="utf-8"))
    assert proc["representacion"] == "manos" and proc["seq_length"] == 30
    preds = json.loads((DIR / "evaluacion_modelo_seleccionado" / "predicciones.json").read_text(encoding="utf-8"))
    guardada = {p["muestra"]: p for p in preds}
    assert muestra in guardada, "la muestra debe ser del conjunto de prueba"

    X_padded, y, lengths, nombres = cargar_datos(PROCESSED, "manos")
    i = nombres.index(muestra)
    T = int(lengths[i])
    x30 = resample_sequence(X_padded[i, :T, :], 30)            # (30, 126)
    x = add_velocity(x30[None])                                   # (1, 30, 252)

    m = tf.keras.models.load_model(str(DIR / "modelo_canonico.keras"))
    capas = ["conv1d_1", "bidirectional", "bidirectional_1", "softmax", "concatenate", "dense_3"]
    sub = tf.keras.Model(m.input, [m.get_layer(n).output for n in capas])
    salidas = dict(zip(capas, sub.predict(x, verbose=0)))
    probs = salidas["dense_3"][0]
    aten = salidas["softmax"][0, :, 0]
    top = np.argsort(-probs)[:5]
    assert abs(float(probs[top[0]]) - guardada[muestra]["prob_predicha"]) < 1e-4
    assert int(top[0]) == guardada[muestra]["predicha"]

    real = int(y[i])
    nombre = lambda k: SIGN_VOCABULARY.get(int(k), str(k))  # noqa: E731
    info = {
        "muestra": muestra, "sena_real": [real, nombre(real)], "conjunto": "prueba (aleatoria 80/10/10)",
        "fotogramas_originales": T, "entrada": list(x.shape),
        "formas": {n: list(v.shape) for n, v in salidas.items()},
        "manos_enmascaradas_en_los_30_pasos": {
            "izquierda_en_ceros": int(np.sum(np.all(x30[:, :63] == 0, axis=1))),
            "derecha_en_ceros": int(np.sum(np.all(x30[:, 63:] == 0, axis=1)))},
        "atencion": [round(float(a), 4) for a in aten], "atencion_suma": round(float(aten.sum()), 4),
        "atencion_paso_max_0a29": int(np.argmax(aten)), "atencion_uniforme": round(1 / 30, 4),
        "top5": [[int(k), nombre(k), round(float(probs[k]), 4)] for k in top],
        "prob_guardada_en_predicciones_json": guardada[muestra]["prob_predicha"],
    }
    (FIG / f"pasada_modelo_{muestra}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False),
                                                       encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.edgecolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    pasos = np.arange(0, 30)
    a1.bar(pasos, aten, width=0.8, color=AZUL, edgecolor="white", linewidth=1)
    a1.axhline(1 / 30, color=TINTA2, ls="--", lw=1, label="peso si todos los pasos valieran igual (1/30)")
    a1.legend(loc="upper right", frameon=False, fontsize=9)
    a1.set_xlim(-0.7, 29.7)
    a1.set_xlabel(f"paso de tiempo (0 a 29; 30 pasos remuestreados de {T} cuadros)", color=TINTA2)
    a1.set_title("Atención: cuánto pesa cada paso en el resumen (suman 1)", loc="left",
                 fontsize=12, color=TINTA)
    a2.set_title("Salida softmax: 5 señas más probables", loc="left", fontsize=12, color=TINTA)
    ys = np.arange(5)[::-1]
    a2.barh(ys, probs[top], height=0.6, color=[AZUL] + [GRIS] * 4, edgecolor="white", linewidth=2)
    a2.set_yticks(ys, [nombre(k) for k in top])
    a2.set_xlim(0, 1)
    for yy, k in zip(ys, top):
        a2.text(probs[k] + 0.02, yy, c(float(probs[k]), 3), va="center", fontsize=10, color=TINTA)
    for ax in (a1, a2):
        ax.grid(True, axis="y" if ax is a1 else "x", color=REJILLA)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            (lambda v, _: c(v, 1)) if ax is a2 else (lambda v, _: f"{int(v)}")))
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: c(v, 2)))
    fig.suptitle(f"Video de prueba {muestra} (seña real: «{nombre(real)}») — entrada 1 × 30 × 252",
                 x=0.01, ha="left", fontsize=13, color=TINTA)
    fig.tight_layout()
    fig.savefig(FIG / f"pasada_modelo_{muestra}.pdf")
    fig.savefig(FIG / f"pasada_modelo_{muestra}.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in info.items() if k != "atencion"}, indent=1, ensure_ascii=False))
    print("atencion", info["atencion"])


if __name__ == "__main__":
    main()
