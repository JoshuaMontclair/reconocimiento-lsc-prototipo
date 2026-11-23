"""
Figuras del experimento de mano ausente (docs/EXPERIMENTO_MANO_AUSENTE.md). Sin imágenes de personas: solo
trayectorias, manos 3D (21 puntos) y mapas de color de los números. Se escriben en docs/figuras_2026-10-02/
y se copian a Tesis/Presentaciones_privadas/explicaciones/.

  a) mano_ausente_a_trayectoria_<video>.png  altura y x de la muñeca de la mano afectada: cuadros
     originales y los 30 pasos de cada regla (mascara, pivote, imputación);
  b) mano_ausente_b_mano3d_<video>.png        un cuadro faltante: cuadro válido anterior, cuadro imputado
     (C) y cuadro válido siguiente, superpuestos (de frente y en perspectiva);
  c) mano_ausente_c_mapas_hola.png            mapas 30 × 126 de posiciones de las tres reglas para «hola».

Numeración: cuadros y pasos se muestran desde 1 (como en la reunión); en el código empiezan en 0.

Uso: python src/figuras_mano_ausente.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import train_loso as tl  # noqa: E402
from src import mano_ausente_variantes as mav  # noqa: E402
from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402

PROCESSED = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
SALIDA = BASE_DIR / "docs" / "figuras_2026-10-02"
PRIVADAS = BASE_DIR.parent / "Presentaciones_privadas" / "explicaciones"

# (video, mano afectada, cuadro faltante para la figura b, en base 0)
VIDEOS = [
    ("0030_0000_0003", "mano_izq", 21),   # «hola»: cuadros 22, 23, 26, 27 sin mano izquierda (huecos cortos)
    ("0007_0000_0002", "mano_izq", 18),   # «comer»: hueco de 4 cuadros (18-21) con mucho movimiento
    ("0015_0001_0000", "mano_der", 28),   # «niña»: hueco largo de 28 cuadros (15-42) en la mano derecha
]

# Paleta (dataviz, referencia validada; primeras 3 ranuras validan todos los pares)
TINTA, TINTA2, GRIS = "#0b0b0b", "#52514e", "#8a8985"
COLOR = {"mascara": "#2a78d6", "pivote": "#eb6834", "imputacion": "#1baf7a"}
MARCA = {"mascara": "o", "pivote": "s", "imputacion": "^"}
NOMBRE = {"mascara": "A mascara (hoy)", "pivote": "B pivote", "imputacion": "C imputación"}
CONEXIONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
NOMBRE_MANO = {"mano_izq": "mano izquierda", "mano_der": "mano derecha"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.titlecolor": TINTA,
                     "axes.spines.top": False, "axes.spines.right": False})


def huecos(det):
    miss = np.flatnonzero(~det)
    return np.split(miss, np.where(np.diff(miss) > 1)[0] + 1) if len(miss) else []


def fig_trayectoria(nombre, seq, mano, ruta):
    b = mav.BLOQUES_MANOS[mano]
    T = seq.shape[0]
    det = mav.mano_detectada(seq, b)
    p = mav.posiciones_momentos(T, 30)
    salidas = {r: mav.aplicar_regla(seq, r, 30) for r in mav.REGLAS}
    _, ind = mav.regla_pivote(seq, 30, devolver_indices=True)
    imp = mav.imputar_fotogramas(seq)
    t = np.arange(T) + 1
    clase = SIGN_VOCABULARY.get(int(nombre.split("_")[0]), "")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    for ax, (col, etiqueta, signo) in zip(axes, [(b.start + 1, "altura de la muñeca\n(−y, anchos de hombro)", -1),
                                                 (b.start, "x de la muñeca\n(anchos de hombro)", 1)]):
        for h in huecos(det):
            ax.axvspan(h[0] + 0.5, h[-1] + 1.5, color="#e6e5e0", zorder=0)
        v = signo * seq[:, col]
        ax.plot(t[det], v[det], color=TINTA2, lw=1.2, zorder=2, label="cuadros originales (mano detectada)")
        ax.scatter(t[det], v[det], s=10, color=TINTA2, zorder=2)
        ax.plot(t, signo * imp[:, col], color=COLOR["imputacion"], lw=1, ls=":", zorder=1,
                label="C: cuadros imputados antes de remuestrear")
        for r in mav.REGLAS:
            ax.plot(p + 1, signo * salidas[r][:, col], color=COLOR[r], lw=1.2, alpha=0.9, zorder=3)
            ax.scatter(p + 1, signo * salidas[r][:, col], s=34, marker=MARCA[r], color=COLOR[r],
                       edgecolor="white", linewidth=1, zorder=4, label=f"{NOMBRE[r]}: 30 pasos")
        ax.axhline(0, color=GRIS, lw=0.6, ls="--", zorder=0)
        ax.set_ylabel(etiqueta)
        ax.grid(True, axis="y", color="#e6e5e0", lw=0.6)
    # pasos en cero de mascara
    cero = np.flatnonzero(np.all(salidas["mascara"][:, b] == 0, axis=1))
    texto = (f"pasos de «mascara» con la muñeca en 0: {', '.join(str(j + 1) for j in cero)}" if len(cero)
             else "«mascara» no deja pasos en 0 exacto, pero sí pasos encogidos hacia 0")
    rep = 30 - len(set(ind[mano].tolist()))
    axes[0].set_title(f"«{clase}» {nombre}, {NOMBRE_MANO[mano]}: {T} cuadros, {int((~det).sum())} sin la mano "
                      f"(franjas grises)\n{texto}; el pivote repite {rep} cuadro(s)", fontsize=11, loc="left")
    axes[1].set_xlabel("cuadro del video (los 30 pasos se dibujan en la posición del cuadro que representan)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return {"cuadros": int(T), "sin_mano": [int(x) + 1 for x in np.flatnonzero(~det)],
            "pasos_mascara_en_cero": [int(j) + 1 for j in cero], "pivote_cuadros": [int(i) + 1 for i in ind[mano]]}


def fig_mano3d(nombre, seq, mano, cuadro, ruta):
    b = mav.BLOQUES_MANOS[mano]
    det = mav.mano_detectada(seq, b)
    assert not det[cuadro]
    ant = max(f for f in range(cuadro) if det[f])
    sig = min(f for f in range(cuadro + 1, len(det)) if det[f])
    imp = mav.imputar_fotogramas(seq)
    P = {"ant": seq[ant, b].reshape(21, 3), "imp": imp[cuadro, b].reshape(21, 3), "sig": seq[sig, b].reshape(21, 3)}
    w = (cuadro - ant) / (sig - ant)
    estilos = {"ant": (f"cuadro {ant + 1} (válido anterior)", GRIS, "o", "-"),
               "imp": (f"cuadro {cuadro + 1} imputado (C) = {1 - w:.2f}·anterior + {w:.2f}·siguiente",
                       COLOR["imputacion"], "^", "--"),
               "sig": (f"cuadro {sig + 1} (válido siguiente)", TINTA, "s", "-")}
    clase = SIGN_VOCABULARY.get(int(nombre.split("_")[0]), "")
    fig = plt.figure(figsize=(16, 6.2))
    ax1 = fig.add_subplot(1, 3, 1)
    ax3 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    for k, (lab, c, m, ls) in estilos.items():
        X = P[k]
        Xc = X - X[0]                                   # forma: centrada en la muñeca
        for i, j in CONEXIONES:
            ax1.plot([X[i, 0], X[j, 0]], [-X[i, 1], -X[j, 1]], color=c, lw=1.6, ls=ls)
            ax3.plot([Xc[i, 0], Xc[j, 0]], [-Xc[i, 1], -Xc[j, 1]], color=c, lw=1.6, ls=ls)
            ax2.plot([X[i, 0], X[j, 0]], [X[i, 2], X[j, 2]], [-X[i, 1], -X[j, 1]], color=c, lw=1.6, ls=ls)
        ax1.scatter(X[:, 0], -X[:, 1], s=22, marker=m, color=c, edgecolor="white", linewidth=0.8, zorder=3, label=lab)
        ax3.scatter(Xc[:, 0], -Xc[:, 1], s=22, marker=m, color=c, edgecolor="white", linewidth=0.8, zorder=3)
        ax2.scatter(X[:, 0], X[:, 2], -X[:, 1], s=18, marker=m, color=c, depthshade=False)
        ax1.annotate("0", (X[0, 0], -X[0, 1]), textcoords="offset points", xytext=(5, -3), color=c, fontsize=9)
        for punta in (4, 8, 12, 16, 20):
            ax3.annotate(str(punta), (Xc[punta, 0], -Xc[punta, 1]), textcoords="offset points", xytext=(3, 3),
                         color=c, fontsize=7.5)
    todos = np.concatenate([P[k] for k in P])
    centro = (todos.max(axis=0) + todos.min(axis=0)) / 2
    radio = float(np.max(np.ptp(todos, axis=0))) / 2 * 1.05
    ax2.set_xlim(centro[0] - radio, centro[0] + radio)
    ax2.set_ylim(centro[2] - radio, centro[2] + radio)
    ax2.set_zlim(-centro[1] - radio, -centro[1] + radio)
    ax2.set_box_aspect((1, 1, 1))
    for ax in (ax1, ax3):
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.set_xlabel("x (anchos de hombro)")
    ax1.set_ylabel("altura −y (anchos de hombro)")
    ax1.set_title("1. Posición, de frente", loc="left")
    ax3.set_title("2. Forma (cada mano con la muñeca en el origen)", loc="left")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.set_zlabel("altura −y")
    ax2.set_title("3. Posición en perspectiva (x, z, altura)", loc="left")
    ax2.view_init(elev=15, azim=-65)
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    desp = np.linalg.norm(P["sig"][0] - P["ant"][0])
    def tam(X):
        return float(np.linalg.norm(X[12] - X[0]))   # muñeca a punta del dedo medio
    fig.suptitle(f"«{clase}» {nombre}, {NOMBRE_MANO[mano]}: el cuadro {cuadro + 1} no tiene la mano. La muñeca se "
                 f"mueve {desp:.2f} anchos de hombro entre los cuadros {ant + 1} y {sig + 1}.\nEl imputado queda en el "
                 f"punto de la recta que corresponde a su tiempo.\nLongitud muñeca-punta del dedo medio: anterior "
                 f"{tam(P['ant']):.2f}, imputado {tam(P['imp']):.2f}, siguiente {tam(P['sig']):.2f} (si la mano gira "
                 f"entre los dos cuadros, el promedio la encoge)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.89))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return {"cuadro_faltante": cuadro + 1, "anterior": ant + 1, "siguiente": sig + 1, "peso_siguiente": float(w),
            "desplazamiento_muneca": float(desp),
            "long_muneca_dedo_medio": {k: float(np.linalg.norm(P[k][12] - P[k][0])) for k in P}}


def fig_mapas(nombre, seq, ruta):
    salidas = {r: mav.aplicar_regla(seq, r, 30) for r in mav.REGLAS}
    lim = float(np.percentile(np.abs(np.concatenate([s.ravel() for s in salidas.values()])), 99))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for ax, r in zip(axes, mav.REGLAS):
        im = ax.imshow(salidas[r], aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
                       extent=(0.5, 126.5, 30.5, 0.5))
        ax.axvline(63.5, color=TINTA, lw=0.8)
        ax.set_title(NOMBRE[r], loc="left")
        ax.set_xticks([1, 32, 63, 95, 126])
        ax.text(32, 31.6, "mano izquierda", ha="center", va="top", fontsize=9, color=TINTA2)
        ax.text(95, 31.6, "mano derecha", ha="center", va="top", fontsize=9, color=TINTA2)
        cero = np.flatnonzero(np.all(salidas[r][:, :63] == 0, axis=1))
        for j in cero:
            ax.text(32, j + 1, f"paso {j + 1} = 0", ha="center", va="center", fontsize=8.5, color=TINTA)
    axes[1].set_xlabel("rasgo (columna)", labelpad=16)
    axes[0].set_ylabel("paso (1-30)")
    axes[0].set_yticks([1, 5, 10, 12, 15, 20, 25, 30])
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("valor normalizado (anchos de hombro); blanco = 0")
    clase = SIGN_VOCABULARY.get(int(nombre.split("_")[0]), "")
    fig.suptitle(f"«{clase}» {nombre}: las 126 posiciones en los 30 pasos con cada regla. En «mascara» los pasos 10 "
                 f"y 12 de la mano izquierda quedan en blanco (0)\ny los pasos 19, 24, 25 y 28 de la mano derecha quedan «encogidos» (más pálidos) por cuadros sueltos sin la mano; en B y C desaparecen", x=0.01, y=1.03, ha="left",
                 fontsize=11)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    X, y, L, nombres, _ = tl.cargar_datos(PROCESSED, "manos")
    hechos = []
    info = {}
    for nombre, mano, cuadro in VIDEOS:
        i = nombres.index(nombre)
        seq = X[i, :int(L[i])]
        clase = SIGN_VOCABULARY.get(int(nombre.split("_")[0]), "").replace("¿", "").replace("?", "").replace("ñ", "n")
        ra = SALIDA / f"mano_ausente_a_trayectoria_{clase}_{nombre}.png"
        rb = SALIDA / f"mano_ausente_b_mano3d_{clase}_{nombre}.png"
        info[nombre] = {"a": fig_trayectoria(nombre, seq, mano, ra), "b": fig_mano3d(nombre, seq, mano, cuadro, rb)}
        hechos += [ra, rb]
        if nombre == "0030_0000_0003":
            rc = SALIDA / f"mano_ausente_c_mapas_{clase}_{nombre}.png"
            fig_mapas(nombre, seq, rc)
            hechos.append(rc)
    import json
    (SALIDA / "mano_ausente_figuras.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    if PRIVADAS.exists():
        for r in hechos:
            shutil.copy2(r, PRIVADAS / r.name)
    for r in hechos:
        print(tl._rel(r))
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
