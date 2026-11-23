"""
Por qué una red temporal y por qué solo manos (diapositivas del flujo, pieza «modelo»).

Panel izquierdo: top-1 LOSO agregado sobre las 997 predicciones (5 personas nuevas) de la red y de
las tres líneas base clásicas con aumentación de espejo (D-10, N-39):
    models/linea_base_4d_2026-09-15/resultados.json -> red.metricas_agregadas.top1 y
    loso.con_espejo.agregado_997.<familia>.top1; McNemar en comparacion_mcnemar.por_familia.<f>.p
Panel derecho: top-1 LOSO, media ± DE entre los 5 pliegues, de los tres candidatos de representación
(D-08, N-30): models/comparacion_D08_2026-09-15.json -> candidatos[].t1, t1_de, al1 (aleatoria).

Solo lee. Escribe figuras/comparacion_red_y_representacion.pdf y .png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

RAIZ = Path(__file__).resolve().parents[3]
FIG = RAIZ / "docs" / "diapositivas_workflow" / "figuras"
AZUL, GRIS = "#2a78d6", "#b9b8b2"
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def c(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def main():
    lb = json.loads((RAIZ / "models/linea_base_4d_2026-09-15/resultados.json").read_text(encoding="utf-8"))
    d8 = json.loads((RAIZ / "models/comparacion_D08_2026-09-15.json").read_text(encoding="utf-8"))
    agg = lb["loso"]["con_espejo"]["agregado_997"]
    mc = lb["comparacion_mcnemar"]["por_familia"]
    izq = [("Red temporal\n(Conv1D + BiLSTM + atención)", lb["red"]["metricas_agregadas"]["top1"], None),
           ("Random Forest", agg["random_forest"]["top1"], mc["random_forest"]["p"]),
           ("SVM RBF", agg["svm_rbf"]["top1"], mc["svm_rbf"]["p"]),
           ("MLP", agg["mlp"]["top1"], mc["mlp"]["p"])]
    der = [(f"{k['nombre'].split(' ', 1)[1].replace('+', ' + ')}\n"
             f"{k['nf']} rasgos/paso · LOSO {c(k['t1'])} ± {c(k['t1_de'])}",
            k["t1"], k["t1_de"], k["al1"]) for k in d8["candidatos"]]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.edgecolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.0))
    for ax in (a1, a2):
        ax.set_xlim(0, 1)
        ax.grid(True, axis="x", color=REJILLA)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: c(v, 1)))

    ys = list(range(len(izq)))[::-1]
    a1.barh(ys, [v for _, v, _ in izq], height=0.55,
            color=[AZUL] + [GRIS] * 3, edgecolor="white", linewidth=2)
    a1.set_yticks(ys, [n for n, _, _ in izq])
    for y, (_, v, p) in zip(ys, izq):
        txt = c(v) + ("" if p is None else f"   McNemar p = {p:.1e}".replace(".", ",").replace("e-", "·10⁻")
                      .replace("⁻18", "⁻¹⁸").replace("⁻57", "⁻⁵⁷").replace("⁻60", "⁻⁶⁰"))
        a1.text(v + 0.015, y, txt, va="center", fontsize=10, color=TINTA)
    a1.set_title("Misma entrada (30 × 252): red temporal frente a\nestadísticos que resumen la secuencia",
                 loc="left", fontsize=12, color=TINTA)
    a1.set_xlabel("top-1 LOSO agregado (997 videos de personas nuevas)", color=TINTA2)

    ys2 = list(range(len(der)))[::-1]
    a2.barh(ys2, [t for _, t, _, _ in der], height=0.55, color=[AZUL, GRIS, GRIS],
            edgecolor="white", linewidth=2)
    a2.errorbar([t for _, t, _, _ in der], ys2, xerr=[de for _, _, de, _ in der], fmt="none",
                ecolor=TINTA2, elinewidth=1.2, capsize=4)
    a2.plot([al for *_, al in der], ys2, "D", ms=8, mfc="white", mec=TINTA, mew=1.5,
            label="aleatoria 80/10/10 (personas conocidas)")
    a2.set_yticks(ys2, [n for n, *_ in der])
    a2.set_title("¿Qué puntos entran a la red?\nTres candidatos (D-08)", loc="left", fontsize=12, color=TINTA)
    a2.set_xlabel("top-1 LOSO, media ± DE entre 5 personas (barra)", color=TINTA2)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "comparacion_red_y_representacion.pdf")
    fig.savefig(FIG / "comparacion_red_y_representacion.png", dpi=160)
    plt.close(fig)
    print([(n.split("\n")[0], round(v, 4), p) for n, v, p in izq])
    print([(n.split("\n")[0], round(t, 4), round(de, 4), al) for n, t, de, al in der])


if __name__ == "__main__":
    main()
