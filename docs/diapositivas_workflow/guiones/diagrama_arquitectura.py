"""
Diagrama de bloques de la arquitectura del modelo canónico (diapositivas del flujo, pieza «modelo»).

Todas las formas y parámetros se leen de figuras/tabla_capas_modelo.json, que escribe
tabla_capas_modelo.py cargando models/canonico_holistic_mascara_manos/modelo_canonico.keras.
Si una forma no coincide con la esperada, el guion falla (no dibuja nada inventado).

Uso (desde la raíz, después de tabla_capas_modelo.py):
    python docs/diapositivas_workflow/guiones/diagrama_arquitectura.py
Escribe figuras/arquitectura_modelo.pdf y .png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[3]
FIG = RAIZ / "docs" / "diapositivas_workflow" / "figuras"
T = json.loads((FIG / "tabla_capas_modelo.json").read_text(encoding="utf-8"))
C = {c["nombre"]: c for c in T["capas"]}
CFG = T["config_capas_clave"]

TINTA, TINTA2 = "#0b0b0b", "#52514e"
# Tintes claros de la paleta (relleno) y su tono (borde), uno por familia de bloque.
FAM = {
    "entrada": ("#f0efec", "#52514e"),
    "conv":    ("#cde2fb", "#2a78d6"),   # azul
    "lstm":    ("#fbd9c9", "#eb6834"),   # naranja
    "resumen": ("#c9eedf", "#1baf7a"),   # aqua
    "densa":   ("#dcd8f5", "#4a3aa7"),   # violeta
}


def forma(nombre):
    f = C[nombre]["forma_salida"][1:]
    return " × ".join(str(x) for x in f)


def p(*nombres, no_entr=False):
    k = "param_no_entrenables" if no_entr else "param_entrenables"
    return sum(C[n][k] for n in nombres)


def dec(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def miles(n):
    return f"{n:,}".replace(",", ".")


def comprobar():
    esperado = {
        "input_layer": [30, 252], "conv1d": [30, 128], "conv1d_1": [30, 128],
        "bidirectional": [30, 256], "bidirectional_1": [30, 128], "dense": [30, 1],
        "multiply": [30, 128], "global_average_pooling1d": [128],
        "global_average_pooling1d_1": [128], "global_max_pooling1d": [128],
        "concatenate": [384], "dense_1": [256], "dense_2": [128], "dense_3": [50],
    }
    for n, f in esperado.items():
        assert C[n]["forma_salida"][1:] == f, (n, C[n]["forma_salida"], f)
    assert T["coincide_con_build_model_30_252_50"]


def caja(ax, x, y, w, h, fam, titulo, cuerpo, forma_txt):
    relleno, borde = FAM[fam]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=relleno, ec=borde, lw=1.6))
    ax.text(x + w / 2, y + h - 0.2, titulo, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=TINTA)
    ax.text(x + w / 2, y + h / 2 - 0.08, cuerpo, ha="center", va="center", fontsize=8.6,
            color=TINTA2, linespacing=1.35)
    ax.text(x + w / 2, y + 0.16, forma_txt, ha="center", va="bottom", fontsize=10,
            color=TINTA, family="DejaVu Sans Mono")


def flecha(ax, a, b, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3,
                                 color=TINTA2, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def main():
    comprobar()
    fig, ax = plt.subplots(figsize=(14, 6.4))
    ax.set_xlim(0, 28)
    ax.set_ylim(0.9, 12.6)
    ax.axis("off")
    W, H = 4.6, 3.3
    yA = 8.6        # fila superior
    yB = 1.6        # fila inferior

    # ---- Fila superior: entrada -> convoluciones -> BiLSTM -> BiLSTM
    xs = [0.3, 5.8, 11.9, 18.0, 23.2]
    caja(ax, xs[0], yA, W, H, "entrada", "Entrada: un video",
         "30 pasos de tiempo\n126 posiciones (42 puntos × 3)\n+ 126 velocidades", "30 × 252")
    caja(ax, xs[1], yA, W + 0.9, H, "conv",
         "2 × [Conv1D + BN + ReLU]",
         f"128 filtros, ventana de 3 pasos\n+ Dropout {dec(CFG['dropout']['rate'], 2)}\n"
         f"{miles(p('conv1d', 'conv1d_1', 'batch_normalization', 'batch_normalization_1'))} parámetros",
         forma("conv1d_1"))
    u1 = CFG["bidirectional"]["units"]
    caja(ax, xs[2], yA, W + 0.9, H, "lstm", f"BiLSTM {u1} (+ BN + Dropout)",
         f"{u1} hacia adelante + {u1} hacia atrás\nDropout {dec(CFG['dropout_1']['rate'], 1)}\n"
         f"{miles(p('bidirectional', 'batch_normalization_2'))} parámetros",
         forma("bidirectional"))
    u2 = CFG["bidirectional_1"]["units"]
    caja(ax, xs[3], yA, W + 0.9, H, "lstm", f"BiLSTM {u2} (+ BN + Dropout)",
         f"{u2} hacia adelante + {u2} hacia atrás\nDropout {dec(CFG['dropout_2']['rate'], 1)}\n"
         f"{miles(p('bidirectional_1', 'batch_normalization_3'))} parámetros",
         forma("bidirectional_1"))
    for i in range(3):
        x0 = xs[i] + (W if i == 0 else W + 0.9)
        flecha(ax, (x0 + 0.05, yA + H / 2), (xs[i + 1] - 0.05, yA + H / 2))

    # ---- Resumen de la secuencia: tres ramas en paralelo
    xr, wr, hr = 1.0, 5.4, 1.75
    ramas = [
        (yB + 3.9, "Atención", f"peso por paso (Dense 1 + softmax sobre 30)\n"
                               f"promedio ponderado · {miles(p('dense'))} parámetros",
         forma("global_average_pooling1d")),
        (yB + 1.95, "Promedio en el tiempo", "media de los 30 pasos",
         forma("global_average_pooling1d_1")),
        (yB + 0.0, "Máximo en el tiempo", "máximo de cada canal",
         forma("global_max_pooling1d")),
    ]
    for (y, tit, cu, fo) in ramas:
        relleno, borde = FAM["resumen"]
        ax.add_patch(FancyBboxPatch((xr, y), wr, hr, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc=relleno, ec=borde, lw=1.4))
        ax.text(xr + 0.2, y + hr - 0.18, tit, ha="left", va="top", fontsize=10,
                fontweight="bold", color=TINTA)
        ax.text(xr + 0.2, y + 0.18, cu, ha="left", va="bottom", fontsize=8, color=TINTA2)
        ax.text(xr + wr - 0.2, y + hr - 0.18, fo, ha="right", va="top", fontsize=10,
                color=TINTA, family="DejaVu Sans Mono")
    # Flecha de la segunda BiLSTM hacia las ramas (baja por la derecha y vuelve a la izquierda)
    xb = xs[3] + (W + 0.9) / 2
    ax.plot([xb, xb], [yA - 0.05, yA - 0.75], color=TINTA2, lw=1.3)
    ax.plot([xb, xr + wr / 2], [yA - 0.75, yA - 0.75], color=TINTA2, lw=1.3)
    # Bus vertical a la izquierda: la MISMA salida 30 × 128 alimenta las tres ramas.
    xbus = 0.45
    ax.plot([xr + wr / 2, xbus], [yA - 0.75, yA - 0.75], color=TINTA2, lw=1.3)
    ax.plot([xbus, xbus], [yA - 0.75, yB + hr / 2], color=TINTA2, lw=1.3)
    for (y, *_r) in ramas:
        flecha(ax, (xbus, y + hr / 2), (xr - 0.05, y + hr / 2))
    ax.text((xb + xr + wr / 2) / 2, yA - 0.62, "30 × 128: una fila por paso de tiempo",
            ha="center", va="bottom", fontsize=9, color=TINTA2)

    # ---- Fila inferior: concatenar -> densas -> salida
    xc = [7.0, 11.6, 16.2, 20.8]
    wc, hc = 3.9, 3.3
    ycb = yB + 1.0
    caja(ax, xc[0], ycb, wc, hc, "resumen", "Concatenar", "atención + promedio\n+ máximo\n(128 + 128 + 128)",
         forma("concatenate"))
    caja(ax, xc[1], ycb, wc, hc, "densa", "Dense 256 ReLU",
         f"+ Dropout {dec(CFG['dropout_3']['rate'], 1)}\n{miles(p('dense_1'))} parámetros", forma("dense_1"))
    caja(ax, xc[2], ycb, wc, hc, "densa", "Dense 128 ReLU",
         f"+ Dropout {dec(CFG['dropout_4']['rate'], 2)}\n{miles(p('dense_2'))} parámetros", forma("dense_2"))
    caja(ax, xc[3], ycb, wc + 1.2, hc, "densa", "Dense 50 softmax",
         f"una probabilidad por seña\n(suman 1)\n{miles(p('dense_3'))} parámetros", forma("dense_3"))
    for (y, *_r) in ramas:
        flecha(ax, (xr + wr + 0.05, y + hr / 2), (xc[0] - 0.05, ycb + hc / 2), rad=0.0)
    for i in range(3):
        flecha(ax, (xc[i] + wc + 0.05, ycb + hc / 2), (xc[i + 1] - 0.05, ycb + hc / 2))
    ax.text(xc[3] + (wc + 1.2) / 2, ycb - 0.25, "seña predicha = la de mayor probabilidad",
            ha="center", va="top", fontsize=9.5, color=TINTA)

    tot = T["totales"]
    ax.text(27.9, 0.95,
            f"{miles(tot['entrenables'])} parámetros entrenables · {miles(tot['no_entrenables'])} no entrenables "
            f"(medias y varianzas móviles de BN) · TensorFlow {T['tensorflow']} / Keras {T['keras']}",
            ha="right", va="bottom", fontsize=9, color=TINTA2)
    ax.text(0.3, 12.5, "Red temporal del prototipo: Conv1D + LSTM bidireccional + atención",
            ha="left", va="top", fontsize=14, color=TINTA)
    fig.tight_layout()
    fig.savefig(FIG / "arquitectura_modelo.pdf")
    fig.savefig(FIG / "arquitectura_modelo.png", dpi=160)
    plt.close(fig)
    print("->", FIG / "arquitectura_modelo.pdf")


if __name__ == "__main__":
    main()
