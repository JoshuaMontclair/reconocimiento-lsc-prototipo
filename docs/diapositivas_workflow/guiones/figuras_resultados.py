"""
Figuras de la pieza «resultados» (PDF para Beamer 16:9). Lee el JSON que escribe
resultados_metricas.py (todas las cifras salen de predicciones guardadas; ver ese guion).

Uso (desde la raíz del repositorio):
  python docs/diapositivas_workflow/guiones/figuras_resultados.py <metricas.json>
Escribe en docs/diapositivas_workflow/figuras/:
  res_esquemas_particion.pdf     quién está en entrenamiento / validación / prueba (LOSO y aleatorio)
  res_top1_por_persona.pdf       top-1 y top-3 por persona nueva (LOSO), media y azar
  res_red_vs_linea_base.pdf      top-1 de la red y de la línea base, personas nuevas y conocidas
  res_recall_por_sena.pdf        recall por seña (LOSO agregado, 997), ordenado
  res_matriz_confusion_50.pdf    matriz de confusión normalizada por fila, 50 × 50
  res_matriz_confusion_15.pdf    submatriz de las 15 señas con más errores (conteos)
  res_umbral.pdf                 rechazo y exactitud entre aceptadas según el umbral (LOSO, 997)
No contiene ninguna imagen de personas.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

OUT = Path("docs/diapositivas_workflow/figuras")
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#8a8984"
AZUL_CLARO = "#86b6ef"
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"
AZAR = 1 / 50

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def coma(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def rejilla_y(ax):
    ax.yaxis.grid(True, color=REJILLA, lw=0.8)
    ax.set_axisbelow(True)


def fig_esquemas(d):
    fig, axs = plt.subplots(2, 1, figsize=(8.4, 6.6), gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.55})
    sen = ["0000", "0001", "0002", "0003", "0004"]
    ax = axs[0]
    for k in range(5):
        for j in range(5):
            prueba = j == k
            ax.add_patch(Rectangle((j, 4 - k), 0.94, 0.86, color=NARANJA if prueba else AZUL, lw=0))
            ax.text(j + 0.47, 4 - k + 0.43, "prueba" if prueba else "entrena.\n+ val.", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold" if prueba else None)
        p = d["loso_por_pliegue"][k]
        ax.text(5.1, 4 - k + 0.43, f"top-1 {coma(p['top1'])}", va="center", fontsize=9, color=TINTA)
    ax.set_xlim(-0.05, 6.1); ax.set_ylim(-0.1, 5)
    ax.set_xticks([j + 0.47 for j in range(5)], [f"persona {s}" for s in sen], fontsize=8.5)
    ax.set_yticks([4 - k + 0.43 for k in range(5)], [f"pliegue {k}" for k in range(5)], fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Personas NUEVAS: LOSO, 5 pliegues (uno por persona excluida)", fontsize=10, loc="left")
    ax.text(0, -0.75, "validación = 2 videos por seña de las 4 personas de entrenamiento (100); "
            "prueba = los ≈200 videos de la persona excluida", fontsize=7.5, color=TINTA2)

    ax = axs[1]
    cont = d["canonico_evaluacion_modelo_seleccionado"]["senantes_en_prueba"]
    # videos por persona = n de prueba de su pliegue LOSO (0001: 199 y 0002: 198 por los duplicados D-06)
    tot = {p["senante"]: p["n"] for p in d["loso_por_pliegue"]}
    for i, s in enumerate(sen):
        te = cont.get(s, 0)
        y = 4 - i
        ax.add_patch(Rectangle((0, y), (tot[s] - te) / 200, 0.86, color=AZUL, lw=0))
        ax.add_patch(Rectangle(((tot[s] - te) / 200, y), te / 200 - 0.006, 0.86, color=NARANJA, lw=0))
        ax.text(1.02, y + 0.43, f"{te} en prueba", va="center", fontsize=8, color=TINTA2)
    ax.set_xlim(0, 1.35); ax.set_ylim(-0.1, 5)
    ax.set_yticks([4 - i + 0.43 for i in range(5)], [f"persona {s}" for s in sen], fontsize=9)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Personas CONOCIDAS: esquema aleatorio (1 sola partición)", fontsize=10, loc="left")
    c = d["aleatoria_conteos"]
    ax.text(0, -0.45, f"entrenamiento {c['train']} + validación {c['val']} (azul) · prueba {c['test']} "
            f"= 2 por seña (naranja)", fontsize=7.5, color=TINTA2)
    fig.legend(handles=[Patch(color=AZUL, label="entrenamiento / validación"), Patch(color=NARANJA, label="prueba")],
               loc="upper right", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(1.0, 1.0))
    fig.savefig(OUT / "res_esquemas_particion.pdf")
    plt.close(fig)


def fig_top1_persona(d):
    pl = d["loso_por_pliegue"]
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, [p["top1"] for p in pl], w, color=AZUL, label="top-1 (acierta la 1.ª opción)")
    b3 = ax.bar(x + w / 2 + 0.01, [p["top3"] for p in pl], w, color=AZUL_CLARO, label="top-3 (está entre las 3 primeras)")
    for b, p in zip(b1, pl):
        ax.text(b.get_x() + b.get_width() / 2, p["top1"] + 0.015, coma(p["top1"]), ha="center", fontsize=8.5)
    for b, p in zip(b3, pl):
        ax.text(b.get_x() + b.get_width() / 2, p["top3"] + 0.015, coma(p["top3"]), ha="center", fontsize=8.5, color=TINTA2)
    m = d["loso_media_de"]["top1"]
    ax.axhline(m["media"], color=TINTA, lw=1.2, ls="--",
               label=f"media top-1 entre personas: {coma(m['media'])} ± {coma(m['de'], 3)} (DE)")
    ax.axhline(AZAR, color=NARANJA, lw=1.4, ls=":", label="azar: 1/50 = 0,02")
    ax.set_xticks(x, [f"persona {p['senante']}\n(n = {p['n']})" for p in pl], fontsize=8.5)
    ax.set_ylim(0, 1.0); ax.set_ylabel("proporción de videos de prueba")
    ax.set_yticks(np.arange(0, 1.01, 0.2), [coma(v, 1) for v in np.arange(0, 1.01, 0.2)])
    rejilla_y(ax)
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.set_title("Personas nuevas (LOSO): cada barra es una persona que el modelo nunca vio", fontsize=10, loc="left")
    fig.savefig(OUT / "res_top1_por_persona.pdf")
    plt.close(fig)


def fig_red_vs_lb(d):
    sistemas = [("Red temporal\n(Conv1D+BiLSTM+atención)", d["loso_agregado_997"]["top1"],
                 d["canonico_evaluacion_modelo_seleccionado"]["top1"]),
                ("Random Forest", d["linea_base"]["random_forest"]["loso"]["top1"], d["linea_base"]["random_forest"]["aleatoria"]["top1"]),
                ("SVM (RBF)", d["linea_base"]["svm_rbf"]["loso"]["top1"], d["linea_base"]["svm_rbf"]["aleatoria"]["top1"]),
                ("MLP", d["linea_base"]["mlp"]["loso"]["top1"], d["linea_base"]["mlp"]["aleatoria"]["top1"])]
    x = np.arange(len(sistemas))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    bn = ax.bar(x - w / 2 - 0.01, [s[1] for s in sistemas], w, color=AZUL, label="personas nuevas (LOSO, 997 videos)")
    bc = ax.bar(x + w / 2 + 0.01, [s[2] for s in sistemas], w, color=GRIS, label="personas conocidas (aleatorio, 100 videos)")
    for bars, idx in ((bn, 1), (bc, 2)):
        for b, s in zip(bars, sistemas):
            ax.text(b.get_x() + b.get_width() / 2, s[idx] + 0.015, coma(s[idx]), ha="center", fontsize=8.5)
    ax.axhline(AZAR, color=NARANJA, lw=1.4, ls=":", label="azar: 1/50 = 0,02")
    ax.set_xticks(x, [s[0] for s in sistemas], fontsize=8.5)
    ax.set_ylim(0, 1.0); ax.set_ylabel("top-1")
    ax.set_yticks(np.arange(0, 1.01, 0.2), [coma(v, 1) for v in np.arange(0, 1.01, 0.2)])
    rejilla_y(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Misma entrada (30 × 252), mismas particiones: la red frente a la línea base clásica",
                 fontsize=10, loc="left")
    fig.savefig(OUT / "res_red_vs_linea_base.pdf")
    plt.close(fig)


def fig_recall(d):
    ps = sorted(d["por_sena"], key=lambda s: (s["recall"], s["f1"], s["id"]))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(ps))
    ax.bar(x, [s["recall"] for s in ps], 0.78, color=[NARANJA if s["recall"] <= 0.25 else AZUL for s in ps])
    m = d["loso_agregado_997"]["recall_macro"]
    ax.axhline(m, color=TINTA, lw=1, ls="--")
    ax.text(0.2, m + 0.02, f"recall macro {coma(m)}", fontsize=8.5)
    ax.set_xticks(x, [s["glosa"] for s in ps], rotation=70, ha="right", fontsize=7.5)
    ax.set_xlim(-0.7, len(ps) - 0.3)
    ax.set_ylim(0, 1.05); ax.set_ylabel("recall = TP / (TP + FN)")
    ax.set_yticks(np.arange(0, 1.01, 0.25), [coma(v) for v in np.arange(0, 1.01, 0.25)])
    rejilla_y(ax)
    ax.legend(handles=[Patch(color=NARANJA, label="recall ≤ 0,25"), Patch(color=AZUL, label="recall > 0,25")],
              frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax.set_title("Recall por seña con personas nuevas (LOSO agregado, 20 videos por seña; 19 en 3 señas)",
                 fontsize=10, loc="left")
    fig.savefig(OUT / "res_recall_por_sena.pdf")
    plt.close(fig)


CMAP = LinearSegmentedColormap.from_list("azul", ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])


def fig_matriz(d):
    cm = np.array(d["matriz_confusion_997"])
    nombres = [s["glosa"] for s in d["por_sena"]]
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    im = ax.imshow(norm, cmap=CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(50), nombres, rotation=90, fontsize=5.8)
    ax.set_yticks(range(50), nombres, fontsize=5.8)
    ax.set_xlabel("seña predicha"); ax.set_ylabel("seña real")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("proporción de los videos de la fila", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Matriz de confusión, personas nuevas (LOSO, 997 videos; fila normalizada)", fontsize=10, loc="left")
    fig.savefig(OUT / "res_matriz_confusion_50.pdf")
    plt.close(fig)

    # 15 señas con más errores (FN + FP)
    ps = d["por_sena"]
    sel = sorted(range(50), key=lambda i: -(ps[i]["FN"] + ps[i]["FP"]))[:15]
    # ordenar por grupo semántico aproximado: agrupar parejas conocidas juntas usando el orden de ids
    sel = sorted(sel)
    sub = cm[np.ix_(sel, sel)]
    filas = cm[sel].sum(axis=1)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(sub / filas[:, None], cmap=CMAP, vmin=0, vmax=0.6)
    for i in range(len(sel)):
        for j in range(len(sel)):
            v = sub[i, j]
            if v:
                frac = v / filas[i]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if frac > 0.3 else TINTA, fontweight="bold" if i == j else None)
    ax.set_xticks(range(len(sel)), [nombres[i] for i in sel], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(sel)), [f"{nombres[i]} ({filas[k]})" for k, i in enumerate(sel)], fontsize=8)
    ax.set_xlabel("seña predicha"); ax.set_ylabel("seña real (videos de la fila)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    cb.set_label("proporción de la fila (escala hasta 0,6)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Las 15 señas con más errores (FN + FP): conteos de videos", fontsize=10, loc="left")
    fig.savefig(OUT / "res_matriz_confusion_15.pdf")
    plt.close(fig)
    return [nombres[i] for i in sel]


def fig_umbral(d, metricas_path):
    # curva sobre las 997 predicciones: se reconstruye desde los predicciones.json del LOSO
    loso = Path("models/loso_2026-09-15_holistic_mascara_manos")
    conf, ok = [], []
    for k in range(5):
        for r in json.loads((loso / f"pliegue_{k}" / "predicciones.json").read_text(encoding="utf-8")):
            conf.append(r["prob_predicha"]); ok.append(r["real"] == r["predicha"])
    conf = np.array(conf); ok = np.array(ok)
    ts = np.round(np.arange(0, 1.0, 0.01), 2)
    rech = np.array([(conf < t).mean() for t in ts])
    exac = np.array([ok[conf >= t].mean() if (conf >= t).any() else np.nan for t in ts])
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    ax.plot(ts, exac, color=AZUL, lw=2, label="exactitud entre las aceptadas")
    ax.plot(ts, rech, color=GRIS, lw=2, label="tasa de rechazo (pide repetir)")
    for t, lab in ((0.20, "0,20 vigente (D-09)"), (0.55, "0,55 candidato (N-47)")):
        e = d["umbral"][f"{t:.2f}"]
        ax.axvline(t, color=TINTA2, lw=0.8, ls="--")
        ax.plot([t], [e["exactitud_entre_aceptadas"]], "o", color=AZUL, ms=8, mec="white", mew=2)
        ax.plot([t], [e["rechazo"]], "o", color=GRIS, ms=8, mec="white", mew=2)
        ax.text(t + 0.01, 0.97, lab, fontsize=8.5, va="top")
        ax.text(t + 0.012, e["exactitud_entre_aceptadas"] + 0.03, coma(e["exactitud_entre_aceptadas"]), fontsize=8.5, color=AZUL)
        ax.text(t + 0.012, e["rechazo"] - 0.07, coma(e["rechazo"]), fontsize=8.5, color=TINTA2)
    ax.set_xlim(0, 0.99); ax.set_ylim(0, 1.02)
    ax.set_xlabel("umbral t: se dice en voz alta solo si la confianza del modelo ≥ t")
    ax.set_xticks(np.arange(0, 1.0, 0.1), [coma(v, 1) for v in np.arange(0, 1.0, 0.1)])
    ax.set_yticks(np.arange(0, 1.01, 0.2), [coma(v, 1) for v in np.arange(0, 1.01, 0.2)])
    rejilla_y(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    ax.set_title("Efecto del umbral con personas nuevas (LOSO, 997 videos)", fontsize=10, loc="left")
    fig.savefig(OUT / "res_umbral.pdf")
    plt.close(fig)


def main(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    fig_esquemas(d)
    fig_top1_persona(d)
    fig_red_vs_lb(d)
    fig_recall(d)
    sel = fig_matriz(d)
    fig_umbral(d, p)
    print("15 señas:", sel)


if __name__ == "__main__":
    main(sys.argv[1])
