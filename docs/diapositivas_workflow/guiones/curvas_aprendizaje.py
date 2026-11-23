"""
Curvas de aprendizaje reales del modelo canónico (diapositivas del flujo, pieza «modelo»).

Fuente: models/canonico_holistic_mascara_manos/semilla_2026/historial.json (la semilla seleccionada
por validación, resultados.json -> seleccion.semilla = 2026; N-31). Dibuja pérdida y exactitud de
entrenamiento y validación por época y marca la mejor época (máxima val_accuracy; entrenamiento.json
-> mejor_epoca = 17). Además dibuja el piso teórico de la pérdida con label smoothing 0,1 y 50
clases (la entropía cruzada no puede bajar de la entropía de la etiqueta suavizada).

También calcula la tasa de aprendizaje de la programación CosineDecay(0.001, 200*125, alpha=1e-6)
en la mejor época y en la última (src/train_final_100.py, entrenar_semilla) para citarla.

Solo lee. Escribe docs/diapositivas_workflow/figuras/curvas_aprendizaje_semilla2026.pdf (y .png)
y curvas_aprendizaje_semilla2026.json con las cifras citadas.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parents[3]
DIR_MODELO = RAIZ / "models" / "canonico_holistic_mascara_manos"
FIG = RAIZ / "docs" / "diapositivas_workflow" / "figuras"

AZUL, NARANJA = "#2a78d6", "#eb6834"          # ranuras categóricas 1 y 2 (paleta validada)
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    res = json.loads((DIR_MODELO / "resultados.json").read_text(encoding="utf-8"))
    semilla = res["seleccion"]["semilla"]
    d = DIR_MODELO / f"semilla_{semilla}"
    h = json.loads((d / "historial.json").read_text(encoding="utf-8"))
    ficha = json.loads((d / "entrenamiento.json").read_text(encoding="utf-8"))
    conf = res["configuracion"]
    E = len(h["loss"])
    mejor = int(np.argmax(h["val_accuracy"])) + 1
    assert mejor == ficha["mejor_epoca"], (mejor, ficha["mejor_epoca"])
    ep = np.arange(1, E + 1)

    # Piso de la pérdida con label smoothing eps y K clases: objetivo (1-eps+eps/K) en la clase
    # real y eps/K en las otras; la entropía cruzada mínima es la entropía de ese objetivo.
    eps, K = 0.1, 50
    p_real, p_otra = 1 - eps + eps / K, eps / K
    piso = -(p_real * math.log(p_real) + (K - 1) * p_otra * math.log(p_otra))

    # Tasa de aprendizaje de CosineDecay por época (pasos por época = n_train_aumentado//64 + 1).
    pasos = ficha["n_train_aumentado"] // conf["batch_size"] + 1
    decay = conf["epochs_max"] * pasos

    def lr(paso, ini=0.001, alpha=1e-6):
        c = 0.5 * (1 + math.cos(math.pi * min(paso, decay) / decay))
        return ini * ((1 - alpha) * c + alpha)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA2,
                         "xtick.color": TINTA2, "ytick.color": TINTA2})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (m, titulo) in zip(axes, [("loss", "Pérdida (entropía cruzada)"),
                                      ("accuracy", "Exactitud")]):
        ax.plot(ep, h[m], color=AZUL, lw=2, label="entrenamiento (7.970 secuencias aumentadas)")
        ax.plot(ep, h["val_" + m], color=NARANJA, lw=2, label="validación (100 videos)")
        ax.axvline(mejor, color=TINTA2, ls="--", lw=1)
        ax.set_title(titulo, color=TINTA, fontsize=12, loc="left")
        ax.set_xlabel("época")
        ax.set_xlim(1, E)
        ax.grid(True, color=REJILLA, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax_l, ax_a = axes
    ax_l.axhline(piso, color=TINTA2, ls=":", lw=1)
    ax_l.text(E, piso - 0.08, f"mínimo posible con suavizado de etiquetas 0,1: {piso:.2f}".replace(".", ","),
              ha="right", va="top", fontsize=9, color=TINTA2)
    ax_l.set_ylim(0.5, max(h["val_loss"][0], h["loss"][0]) + 0.1)
    ax_a.set_ylim(0, 1.03)
    ax_a.plot([mejor], [h["val_accuracy"][mejor - 1]], "o", ms=8, color=NARANJA,
              markeredgecolor="white", markeredgewidth=2, zorder=5)
    ax_a.annotate(f"mejor época {mejor}: validación {h['val_accuracy'][mejor-1]:.2f}".replace(".", ","),
                  xy=(mejor, h["val_accuracy"][mejor - 1]), xytext=(mejor + 3, 0.55),
                  fontsize=10, color=TINTA,
                  arrowprops=dict(arrowstyle="-", color=TINTA2, lw=0.8))
    ax_a.text(E, 0.08, f"parada temprana en la época {E}\n(paciencia {conf['patience']} épocas sin mejorar)",
              ha="right", va="bottom", fontsize=9, color=TINTA2)
    ax_a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.22), frameon=False, fontsize=9)
    fig.suptitle(f"Modelo canónico «manos», semilla {semilla} (seleccionada por validación)",
                 x=0.01, ha="left", fontsize=13, color=TINTA)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "curvas_aprendizaje_semilla2026.pdf")
    fig.savefig(FIG / "curvas_aprendizaje_semilla2026.png", dpi=160)
    plt.close(fig)

    cifras = {
        "fuente": str((d / "historial.json").relative_to(RAIZ)),
        "semilla": semilla, "epocas_ejecutadas": E, "mejor_epoca": mejor,
        "en_mejor_epoca": {k: round(h[k][mejor - 1], 4) for k in h},
        "epoca_1": {k: round(h[k][0], 4) for k in h},
        "ultima_epoca": {k: round(h[k][-1], 4) for k in h},
        "min_val_loss": {"epoca": int(np.argmin(h["val_loss"])) + 1, "valor": round(min(h["val_loss"]), 4)},
        "piso_perdida_label_smoothing_0.1_50_clases": round(piso, 4),
        "pasos_por_epoca": pasos, "decay_steps": decay,
        "lr_inicio": lr(0), "lr_fin_mejor_epoca": lr(mejor * pasos), "lr_fin_ultima_epoca": lr(E * pasos),
    }
    (FIG / "curvas_aprendizaje_semilla2026.json").write_text(
        json.dumps(cifras, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(cifras, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
