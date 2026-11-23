"""Figuras de la pieza «datos» (corpus, EDA y particiones) para las diapositivas del flujo.

Solo LEE archivos del repositorio y escribe PDF vectoriales en docs/diapositivas_workflow/figuras.
No abre videos ni decodifica fotogramas (ética, 07_etica.tex): todo sale de conteos y cabeceras.

Fuentes:
  - data/processed/caracterizacion/bitacora_integridad.json -> muestras[*].video.COLOR_BODY.fotogramas
    (cabecera AVI; 24 fot./s en los 1.000 videos) y nombres SSSS_VVVV_RRRR.
  - data/processed/caracterizacion/caracterizacion.json -> vocabulario.senas (glosas).
  - data/partitions/aleatoria_80_10_10.json y loso_5_pliegues.json -> excluidas y conteos (D-06).
  - src/realtime.py, --ventana-seg por defecto 2,5 s (N-11).

Uso (desde la raíz del repositorio):
  python docs/diapositivas_workflow/guiones/figuras_datos.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RAIZ = Path(__file__).resolve().parents[3]
SALIDA = RAIZ / "docs" / "diapositivas_workflow" / "figuras"
SALIDA.mkdir(parents=True, exist_ok=True)

# Paleta (skill dataviz, instancia de referencia, modo claro; validada: 3 primeras ranuras).
AZUL, NARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, TENUE = "#0b0b0b", "#52514e", "#898781"
GRIS_BARRA = "#b9c9dd"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.edgecolor": TENUE,
    "axes.labelcolor": TINTA2,
    "xtick.color": TINTA2,
    "ytick.color": TINTA2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

FPS_COLOR_BODY = 24.0          # caracterizacion.md §6: 24,0 fot./s en los 1.000 videos
VENTANA_PROTOTIPO_S = 2.5      # src/realtime.py --ventana-seg (N-11)


def coma(x: float, dec: int = 1) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


def cargar():
    bit = json.load(open(RAIZ / "data/processed/caracterizacion/bitacora_integridad.json"))
    car = json.load(open(RAIZ / "data/processed/caracterizacion/caracterizacion.json"))
    ale = json.load(open(RAIZ / "data/partitions/aleatoria_80_10_10.json"))
    loso = json.load(open(RAIZ / "data/partitions/loso_5_pliegues.json"))
    excluidas = set(ale["excluidas"])
    assert excluidas == set(loso["excluidas"])
    fps = {m["video"]["COLOR_BODY"]["fps"] for m in bit["muestras"]}
    assert fps == {FPS_COLOR_BODY}, fps
    fot = {m["muestra"]: m["video"]["COLOR_BODY"]["fotogramas"] for m in bit["muestras"]}
    senas = car["vocabulario"]["senas"]
    glosa = {}
    for s in (senas if isinstance(senas, list) else senas.values()):
        glosa[int(s["id"])] = s.get("glosa_codigo") or s.get("glosa")
    return fot, excluidas, glosa, ale, loso


def fig_muestras_por_clase(fot, excluidas, glosa):
    c = Counter(int(n[:4]) for n in fot if n not in excluidas)
    ids = list(range(50))
    vals = [c[i] for i in ids]
    assert sum(vals) == 997 and sorted(set(vals)) == [19, 20]
    menos = [i for i in ids if c[i] == 19]
    colores = [NARANJA if i in menos else GRIS_BARRA for i in ids]

    fig, ax = plt.subplots(figsize=(6.3, 2.75))
    ax.bar(ids, vals, width=0.78, color=colores, edgecolor="white", linewidth=0.6, zorder=2)
    ax.axhline(20, color=TENUE, lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(-0.8, 49.8)
    ax.set_ylim(0, 23.5)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel("Videos (muestras) por seña")
    ax.set_xticks(ids)
    ax.set_xticklabels([glosa[i] for i in ids], rotation=90, fontsize=5.6)
    for lab, i in zip(ax.get_xticklabels(), ids):
        if i in menos:
            lab.set_color(NARANJA)
            lab.set_fontweight("bold")
    for i in menos:
        ax.annotate("19", (i, 19), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=TINTA, fontweight="bold")
    ax.text(49.8, 21.2, "47 señas con 20 videos  ·  3 señas con 19  ·  total 997",
            ha="right", va="bottom", fontsize=7.5, color=TINTA)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.grid(axis="y", color="#e6e5e1", lw=0.5, zorder=0)
    fig.savefig(SALIDA / "fig_muestras_por_clase.pdf")
    plt.close(fig)
    return {glosa[i]: c[i] for i in menos}


def fig_duracion_clips(fot, excluidas):
    s = np.array([fot[n] / FPS_COLOR_BODY for n in sorted(fot) if n not in excluidas])
    assert len(s) == 997
    med = float(np.median(s))
    frac25 = float((s <= VENTANA_PROTOTIPO_S).mean())
    frac30 = float((s <= 3.0).mean())
    paso = 3 / FPS_COLOR_BODY  # 3 fotogramas = 0,125 s
    bordes = np.arange(np.floor(s.min() / paso) * paso, s.max() + paso, paso)

    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.hist(s, bins=bordes, color=AZUL, edgecolor="white", linewidth=0.6, zorder=2)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.30)
    ax.axvline(VENTANA_PROTOTIPO_S, color=NARANJA, lw=1.6, zorder=3)
    ax.axvline(med, color=TINTA, lw=1.0, ls=(0, (4, 2)), zorder=3)
    ax.text(VENTANA_PROTOTIPO_S - 0.04, ymax * 1.27,
            f"ventana del\nprototipo: 2,5 s\n{coma(100 * frac25)} % de los\nvideos dura ≤ 2,5 s",
            ha="right", va="top", fontsize=7, color=NARANJA)
    ax.text(med + 0.04, ymax * 1.27, f"mediana: {coma(med, 2)} s ({int(round(med * 24))} cuadros)",
            ha="left", va="top", fontsize=7, color=TINTA)
    ax.set_xlabel("Duración del video (s) = cuadros del video COLOR_BODY / 24 fps")
    ax.set_ylabel("Número de videos")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: coma(x, 1)))
    ax.set_xlim(1.5, 5.8)
    ax.text(5.75, ymax * 0.55,
            f"n = 997 videos\nmín. {coma(s.min(), 3)} s · máx. {coma(s.max(), 3)} s\n"
            f"({int(round(s.min() * 24))} a {int(round(s.max() * 24))} cuadros)\n"
            f"{coma(100 * frac30)} % dura ≤ 3,0 s",
            ha="right", va="center", fontsize=7, color=TINTA2)
    ax.grid(axis="y", color="#e6e5e1", lw=0.5, zorder=0)
    fig.savefig(SALIDA / "fig_duracion_clips.pdf")
    plt.close(fig)
    return dict(n=len(s), mediana=med, media=float(s.mean()), minimo=float(s.min()), maximo=float(s.max()),
                p90=float(np.percentile(s, 90)), frac_le_25=frac25, frac_le_30=frac30)


def fig_duracion_por_senante(fot, excluidas):
    grupos = []
    for v in range(5):
        grupos.append(np.array([fot[n] / FPS_COLOR_BODY for n in sorted(fot)
                                if n not in excluidas and int(n[5:9]) == v]))
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    rng = np.random.default_rng(2026)
    for v, g in enumerate(grupos):
        x = v + rng.uniform(-0.18, 0.18, size=len(g))
        ax.scatter(x, g, s=5, color=AZUL, alpha=0.45, linewidths=0, zorder=2)
        m = float(np.median(g))
        ax.plot([v - 0.3, v + 0.3], [m, m], color=TINTA, lw=1.6, zorder=3)
        ax.text(v + 0.33, m, f"{coma(m, 2)} s", va="center", ha="left", fontsize=7, color=TINTA)
    ax.axhline(VENTANA_PROTOTIPO_S, color=NARANJA, lw=1.2, zorder=1)
    ax.text(4.75, VENTANA_PROTOTIPO_S - 0.06, "2,5 s", color=NARANJA, fontsize=7, ha="right", va="top")
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"persona {v:04d}\n(n = {len(g)})" for v, g in enumerate(grupos)])
    ax.set_ylabel("Duración del video (s)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: coma(x, 1)))
    ax.set_xlim(-0.6, 4.8)
    ax.grid(axis="y", color="#e6e5e1", lw=0.5, zorder=0)
    ax.text(-0.55, ax.get_ylim()[1], "raya = mediana de la persona; cada punto es un video",
            fontsize=6.5, color=TINTA2, va="top")
    fig.savefig(SALIDA / "fig_duracion_por_senante.pdf")
    plt.close(fig)
    return {f"{v:04d}": dict(n=len(g), mediana=float(np.median(g))) for v, g in enumerate(grupos)}


def fig_particiones(ale, loso):
    filas = [("Aleatoria estratificada\n(semilla 2026)", ale["conteos"], "las mismas 5 personas en los tres conjuntos")]
    for p in loso["pliegues"]:
        filas.append((f"LOSO pliegue {p['pliegue']}", p["conteos"], f"prueba = persona {p['senante_test']}"))
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    y = np.arange(len(filas))[::-1].astype(float)
    y[0] += 0.35  # separa la aleatoria de los pliegues LOSO
    for yi, (nom, c, nota) in zip(y, filas):
        izq = 0
        for clave, col, tcol in (("train", AZUL, "white"), ("val", AGUA, TINTA), ("test", NARANJA, TINTA)):
            w = c[clave]
            ax.barh(yi, w, left=izq, height=0.62, color=col, edgecolor="white", linewidth=1.2)
            ax.text(izq + w / 2, yi, f"{w}", ha="center", va="center", fontsize=7, color=tcol, fontweight="bold")
            izq += w
        ax.text(1005, yi, nota, va="center", ha="left", fontsize=6.8, color=TINTA2)
    ax.set_yticks(y)
    ax.set_yticklabels([f[0] for f in filas], fontsize=7.2)
    ax.set_xlim(0, 1000)
    ax.set_xlabel("Videos (muestras); total 997 en cada fila")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AZUL, label="entrenamiento"),
                       Patch(color=AGUA, label="validación (2 por seña = 100)"),
                       Patch(color=NARANJA, label="prueba")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7)
    fig.savefig(SALIDA / "fig_particiones.pdf")
    plt.close(fig)
    return {f[0].split("\n")[0]: f[1] for f in filas}


def fig_cuadro_vs_video(fot, excluidas, glosa):
    """Dos formas de contar por clase: cuadros (NO es la unidad de muestra) frente a videos (sí lo es).

    Cada video, dure lo que dure, se remuestrea a 30 pasos (src/train_final_100.py, resample_sequence,
    --seq-length 30) y entra al modelo como UNA muestra de 30 x 252 (procedencia del modelo canónico).
    """
    nombres = [n for n in sorted(fot) if n not in excluidas]
    cuadros = np.zeros(50, dtype=int)
    videos = np.zeros(50, dtype=int)
    for n in nombres:
        cuadros[int(n[:4])] += fot[n]
        videos[int(n[:4])] += 1
    ids = np.arange(50)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 3.3), sharex=True,
                                 gridspec_kw=dict(hspace=0.45))
    a1.bar(ids, cuadros, width=0.78, color=GRIS_BARRA, edgecolor="white", linewidth=0.6, zorder=2)
    a1.set_ylim(0, cuadros.max() * 1.25)
    a1.set_title("Si se contaran CUADROS por seña (no es la unidad de muestra): "
                 f"de {cuadros.min():,} a {cuadros.max():,}".replace(",", "."),
                 fontsize=7.5, loc="left", color=TINTA2)
    a1.set_ylabel("cuadros", fontsize=7)
    a2.bar(ids, videos, width=0.78, color=AZUL, edgecolor="white", linewidth=0.6, zorder=2)
    a2.set_ylim(0, 23)
    a2.set_title("Lo que se cuenta: VIDEOS por seña = muestras (20; 19 en 3 señas); "
                 "cada video → 1 muestra de 30 × 252", fontsize=7.5, loc="left", color=TINTA)
    a2.set_ylabel("videos", fontsize=7)
    a2.set_yticks([0, 10, 20])
    a2.set_xticks(ids)
    a2.set_xticklabels([glosa[i] for i in ids], rotation=90, fontsize=5.6)
    a2.tick_params(axis="x", length=0, pad=2)
    a1.tick_params(axis="x", length=0)
    for a in (a1, a2):
        a.grid(axis="y", color="#e6e5e1", lw=0.5, zorder=0)
        a.set_xlim(-0.8, 49.8)
    fig.savefig(SALIDA / "fig_cuadro_vs_video.pdf")
    plt.close(fig)
    return dict(cuadros_min=int(cuadros.min()), clase_min=glosa[int(cuadros.argmin())],
                cuadros_max=int(cuadros.max()), clase_max=glosa[int(cuadros.argmax())],
                cuadros_total=int(cuadros.sum()))


if __name__ == "__main__":
    fot, excluidas, glosa, ale, loso = cargar()
    r = {
        "cuadro_vs_video": fig_cuadro_vs_video(fot, excluidas, glosa),
        "clases_con_19": fig_muestras_por_clase(fot, excluidas, glosa),
        "duracion": fig_duracion_clips(fot, excluidas),
        "duracion_por_senante": fig_duracion_por_senante(fot, excluidas),
        "particiones": fig_particiones(ale, loso),
    }
    print(json.dumps(r, ensure_ascii=False, indent=1))
