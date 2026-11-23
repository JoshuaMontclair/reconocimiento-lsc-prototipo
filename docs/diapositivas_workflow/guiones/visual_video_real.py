"""El proceso completo, de forma gráfica, sobre UN video real del corpus LSC50 (con la persona visible).

Toma una muestra del corpus y la pasa por el MISMO camino del sistema (prueba_escritorio.camino_corpus:
MediaPipe HolisticLandmarker, normalización por hombros, regla «mascara», columnas «manos», 30 pasos y
velocidades) y por el modelo canónico. Genera imágenes y una animación para explicar el proceso.

ÉTICA (07_etica.tex del anteproyecto): las salidas contienen la imagen de una persona del corpus. El
proyecto se comprometió a no publicarlas ni redistribuirlas y a mantenerlas con acceso restringido al
estudiante y al director. Por eso este guion EXIGE una carpeta de salida FUERA del repositorio y se niega
a escribir dentro de él. Uso: mostrar al director y en la sustentación; no entregar el archivo ni
incluirlo en el documento de tesis (ahí van los esqueletos de explicacion_simple / workflow_lsc).

    python docs/diapositivas_workflow/guiones/visual_video_real.py 0030_0000_0003 \
        --salida "…/Tesis/Presentaciones_privadas/2026-09-30_video_real"
"""
import argparse
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prueba_escritorio as pe  # noqa: E402

RAIZ = pe.RAIZ
# colores (BGR para cv2; hex para matplotlib)
HEX = {"muñeca": "#374151", "pulgar": "#2563EB", "índice": "#D9642B", "medio": "#0F9D8A",
       "anular": "#E0A100", "meñique": "#C2408F"}
BGR = {k: tuple(int(v[i:i + 2], 16) for i in (5, 3, 1)) for k, v in HEX.items()}
DEDO_DE = {p: dedo for dedo, idx in pe.DEDOS.items() for p in idx}
AZUL, NARANJA, GRIS = "#1F4E79", "#D9642B", "#9CA3AF"


# ------------------------------------------------------------------------------------------ dibujo
def tramos_sin_mano(aus):
    """Intervalos [ini, fin] (índices de cuadro) en que la mano no se detectó."""
    tramos, ini = [], None
    for k, a in enumerate(list(aus) + [False]):
        if a and ini is None:
            ini = k
        elif not a and ini is not None:
            tramos.append((ini, k - 1)); ini = None
    return tramos


def sombrear_sin_mano(ax, aus):
    for n, (i, f) in enumerate(tramos_sin_mano(aus)):
        ax.axvspan(i + 0.5, f + 1.5, color="#E5E7EB", zorder=0, lw=0,
                   label="mano no detectada → 0 (sin dato)" if n == 0 else None)


def puntos_cuadro(d, k):
    """Coordenadas en píxeles del cuadro k: postura (33,2), mano izq (21,2) o None, mano der (21,2) o None."""
    W, H = d["video"]["ancho"], d["video"]["alto"]
    c = d["cruda"][k]
    post = c[:99].reshape(33, 3)[:, :2] * [W, H]
    izq = None if d["sin_izq"][k] else c[99:162].reshape(21, 3)[:, :2] * [W, H]
    der = None if d["sin_der"][k] else c[162:225].reshape(21, 3)[:, :2] * [W, H]
    return post, izq, der


def dibujar(img_rgb, d, k, grosor=3, radio=6, postura=True):
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    post, izq, der = puntos_cuadro(d, k)
    if postura:
        for i, j in pe.CONEX_POSTURA:
            if max(i, j) < 25:
                cv2.line(img, tuple(post[i].astype(int)), tuple(post[j].astype(int)), (200, 200, 200), grosor, cv2.LINE_AA)
        for i in (11, 12):
            cv2.circle(img, tuple(post[i].astype(int)), radio + 3, (143, 64, 124), -1, cv2.LINE_AA)
    for mano in (izq, der):
        if mano is None:
            continue
        for i, j in pe.CONEX_MANO:
            cv2.line(img, tuple(mano[i].astype(int)), tuple(mano[j].astype(int)), BGR[DEDO_DE[j]], grosor, cv2.LINE_AA)
        for p in range(21):
            cv2.circle(img, tuple(mano[p].astype(int)), radio, BGR[DEDO_DE[p]], -1, cv2.LINE_AA)
            cv2.circle(img, tuple(mano[p].astype(int)), radio, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def caja_persona(d, margen=0.12):
    """Recorte fijo para todo el video: la unión de postura y manos de todos los cuadros, con margen."""
    W, H = d["video"]["ancho"], d["video"]["alto"]
    xs, ys = [], []
    for k in range(d["T"]):
        post, izq, der = puntos_cuadro(d, k)
        pts = [post[:25]] + [m for m in (izq, der) if m is not None]
        a = np.vstack(pts)
        xs += [a[:, 0].min(), a[:, 0].max()]; ys += [a[:, 1].min(), a[:, 1].max()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    mx, my = (x1 - x0) * margen, (y1 - y0) * margen
    x0, x1 = int(max(0, x0 - mx)), int(min(W, x1 + mx))
    y0, y1 = int(max(0, y0 - my * 2)), int(min(H, y1 + my))
    return x0, y0, x1, y1


# ------------------------------------------------------------------------------------------ figuras
def fig_cuadros(d, caja, ruta, plt, con_puntos):
    idx = np.linspace(0, d["T"] - 1, 6).round().astype(int)
    x0, y0, x1, y1 = caja
    fig, axes = plt.subplots(1, 6, figsize=(15, 4.2))
    for a, k in zip(axes, idx):
        im = dibujar(d["cuadros"][k], d, k) if con_puntos else d["cuadros"][k]
        a.imshow(im[y0:y1, x0:x1]); a.axis("off")
        a.set_title(f"cuadro {k + 1}\n{k / d['video']['fps']:.2f} s".replace(".", ","), fontsize=13)
    fig.tight_layout(); fig.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_mano(d, ruta, plt):
    # cuadro con la mano más visible y alta (la seña en curso)
    dom = pe.mano_dominante(d)[0]
    aus = d["sin_izq"] if dom == "izquierda" else d["sin_der"]
    ywr = d["cruda"][:, (99 if dom == "izquierda" else 162) + 1].astype(float); ywr[aus] = np.inf
    k = int(np.argmin(ywr))
    post, izq, der = puntos_cuadro(d, k)
    mano = izq if dom == "izquierda" else der
    img = dibujar(d["cuadros"][k], d, k, grosor=4, radio=9, postura=False)
    c0 = mano.min(axis=0); c1 = mano.max(axis=0); lado = (c1 - c0).max() * 0.75
    cx, cy = (c0 + c1) / 2
    xa, xb = int(cx - lado), int(cx + lado); ya, yb = int(cy - lado), int(cy + lado)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    a1.imshow(img[max(0, ya):yb, max(0, xa):xb]); a1.axis("off")
    for p in range(21):
        a1.text(mano[p, 0] - max(0, xa) + 12, mano[p, 1] - max(0, ya) - 8, str(p), fontsize=11,
                fontweight="bold", color=HEX[DEDO_DE[p]],
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    a1.set_title(f"Mano {dom}, cuadro {k + 1}: 21 puntos", fontsize=15)
    base = 99 if dom == "izquierda" else 162
    vals = d["cruda"][k, base:base + 63].reshape(21, 3)
    a2.axis("off")
    filas = [[str(p), DEDO_DE[p]] + [f"{v:.3f}".replace(".", ",") for v in vals[p]] for p in range(21)]
    t = a2.table(cellText=filas, colLabels=["punto", "dedo", "x (horizontal)", "y (vertical)", "z (profundidad)"],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1, 1.18)
    for (r, c), cel in t.get_celld().items():
        cel.set_edgecolor("#D1D5DB")
        if r == 0:
            cel.set_facecolor("#DCE8F5"); cel.set_text_props(fontweight="bold")
        elif c <= 1:
            cel.set_text_props(color=HEX[DEDO_DE[r - 1]], fontweight="bold")
    a2.set_title("Lo que MediaPipe entrega: 21 × 3 = 63 números", fontsize=15)
    fig.tight_layout(); fig.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close(fig)
    return k, dom


def fig_tabla_video(d, ruta, plt):
    fig, a = plt.subplots(figsize=(13, 3.4)); a.axis("off")
    cols = ["cuadro", "muñeca izq x", "muñeca izq y", "muñeca izq z", "pulgar izq x", "…", "meñique der z"]
    filas = []
    for k in list(range(5)) + [None] + [d["T"] - 1]:
        if k is None:
            filas.append(["⋮"] * len(cols)); continue
        m = d["cruda"][k, 99:225]
        f = lambda v: f"{v:.3f}".replace(".", ",")
        filas.append([str(k + 1), f(m[0]), f(m[1]), f(m[2]), f(m[3]), "…", f(m[125])])
    t = a.table(cellText=filas, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 1.6)
    for (r, c), cel in t.get_celld().items():
        cel.set_edgecolor("#D1D5DB")
        if r == 0:
            cel.set_facecolor("#DCE8F5"); cel.set_text_props(fontweight="bold")
    a.set_title(f"El video completo como tabla: {d['T']} filas (una por cuadro) × 126 números de las manos",
                fontsize=15)
    fig.tight_layout(); fig.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_normalizacion(d, caja, ruta, plt):
    k = d["T"] // 2
    post, izq, der = puntos_cuadro(d, k)
    x0, y0, x1, y1 = caja
    img = dibujar(d["cuadros"][k], d, k)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"wspace": 0.35})
    a1.imshow(img[y0:y1, x0:x1]); a1.axis("off")
    a1.set_title("Antes: posición en píxeles\n(depende de dónde está la persona)", fontsize=13)
    n = d["norm"][k]
    P = n[:99].reshape(33, 3)[:, :2]
    for i, j in pe.CONEX_POSTURA:
        if max(i, j) < 25:
            a2.plot(*P[[i, j]].T, color=GRIS, lw=2)
    for base, aus in ((99, d["sin_izq"][k]), (162, d["sin_der"][k])):
        if aus:
            continue
        M = n[base:base + 63].reshape(21, 3)[:, :2]
        for i, j in pe.CONEX_MANO:
            a2.plot(*M[[i, j]].T, color=HEX[DEDO_DE[j]], lw=2)
    a2.scatter([0], [0], s=160, marker="+", color="#8F407C", linewidths=3, zorder=5)
    a2.annotate("(0, 0) = centro de los hombros", (0, 0), xytext=(25, -25), textcoords="offset points",
                fontsize=12, color="#8F407C")
    a2.axhline(0, color="#E5E7EB", lw=1, zorder=0); a2.axvline(0, color="#E5E7EB", lw=1, zorder=0)
    a2.set_aspect("equal"); a2.invert_yaxis()
    a2.set_xlabel("anchos de hombros"); a2.set_ylabel("anchos de hombros")
    a2.set_title("Después: medida desde los hombros\n(igual para cualquier persona y distancia)", fontsize=13)
    fig.tight_layout(); fig.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_remuestreo(d, caja, ruta, plt):
    T, pos = d["T"], d["posiciones"]
    x0, y0, x1, y1 = caja
    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.35)
    a1 = fig.add_subplot(gs[0])
    # tira de miniaturas: 10 de los 30 pasos
    sel = np.linspace(0, 29, 10).round().astype(int)
    a1.axis("off")
    for n, s in enumerate(sel):
        k = int(round(pos[s]))
        im = d["cuadros"][k][y0:y1, x0:x1]
        a1.imshow(im, extent=(n * 1.3, n * 1.3 + 1, 0, im.shape[0] / im.shape[1]))
        a1.text(n * 1.3 + 0.5, -0.1, f"paso {s + 1}\ncuadro {pos[s] + 1:.1f}".replace(".", ","),
                ha="center", va="top", fontsize=10)
    a1.set_xlim(-0.1, 13); a1.set_ylim(-0.6, 1.6)
    a1.set_title(f"De {T} cuadros se toman 30 momentos repartidos de principio a fin (se muestran 10)", fontsize=14)
    a2 = fig.add_subplot(gs[1])
    dom = pe.mano_dominante(d)[0]
    base = 0 if dom == "izquierda" else 63
    aus = d["sin_izq"] if dom == "izquierda" else d["sin_der"]
    y = d["manos"][:, base + 1].astype(float).copy(); y[aus] = np.nan
    t = np.arange(T)
    sombrear_sin_mano(a2, aus)
    a2.plot(t + 1, y, color=AZUL, lw=1.5, marker="o", ms=3.5, alpha=0.6, label=f"{T} cuadros originales")
    a2.scatter(pos + 1, d["remuestreo"][:, base + 1], s=70, marker="D", facecolor="white", edgecolor=NARANJA,
               linewidths=2, zorder=3, label="30 pasos (interpolación lineal)")
    a2.invert_yaxis(); a2.set_xlabel("cuadro"); a2.set_ylabel(f"altura de la muñeca {dom}\n(↑ sube)")
    a2.legend(loc="upper right", fontsize=11)
    fig.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_matriz(d, ruta, plt):
    X = d["X"][0]
    fig, a = plt.subplots(figsize=(14, 5.2))
    v = np.nanpercentile(np.abs(X), 98)
    im = a.imshow(X, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    for xline in (62.5, 125.5, 188.5):
        a.axvline(xline, color="black", lw=1.5 if xline == 125.5 else 0.8)
    for xc, txt in ((31, "posición\nmano izquierda"), (94, "posición\nmano derecha"),
                    (157, "velocidad\nmano izquierda"), (220, "velocidad\nmano derecha")):
        a.text(xc, -1.3, txt, ha="center", va="bottom", fontsize=12)
    a.set_xlabel("252 números por paso"); a.set_ylabel("30 pasos (tiempo ↓)")
    a.set_yticks([0, 9, 19, 29]); a.set_yticklabels(["1", "10", "20", "30"])
    fig.colorbar(im, ax=a, fraction=0.02, pad=0.01)
    fig.tight_layout(); fig.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_prediccion(d, prob, umbral, ruta, plt):
    from src.grammar_mapper import SIGN_VOCABULARY
    orden = np.argsort(prob)[::-1][:10]
    fig, a = plt.subplots(figsize=(11, 5))
    cols = [NARANJA if j == orden[0] else GRIS for j in orden]
    a.barh(range(10)[::-1], prob[orden] * 100, color=cols)
    a.set_yticks(range(10)[::-1]); a.set_yticklabels([SIGN_VOCABULARY[int(j)] for j in orden], fontsize=13)
    for r, j in enumerate(orden):
        a.text(prob[j] * 100 + 1, 9 - r, f"{prob[j] * 100:.1f} %".replace(".", ","), va="center", fontsize=12)
    a.axvline(umbral * 100, color="#2E7D4F", ls="--", lw=2)
    a.text(umbral * 100 + 1, 9.6, f"umbral {umbral * 100:.0f} %", color="#2E7D4F", fontsize=12)
    a.set_xlim(0, 105); a.set_xlabel("probabilidad (%)")
    real = d["glosa"]
    a.set_title(f"Las 10 señas más probables (de 50). Seña real: «{real}»", fontsize=14)
    fig.tight_layout(); fig.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------------------------------ animación
def guardar_mp4(frames, ruta, fps):
    """MP4 en H.264 (avc1), que QuickTime reproduce bien; el códec «mp4v» salía con manchas.
    H.264 exige ancho y alto pares: se recorta un píxel si hace falta."""
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    vw = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"avc1"), fps, (w2, h2))
    if not vw.isOpened():
        raise SystemExit("OpenCV no puede escribir H.264 (avc1) en este equipo")
    for f in frames:
        vw.write(cv2.cvtColor(np.ascontiguousarray(f[:h2, :w2]), cv2.COLOR_RGB2BGR))
    vw.release()


def animacion(d, caja, prob, umbral, ruta_mp4, ruta_gif, plt):
    from src.grammar_mapper import SIGN_VOCABULARY
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    x0, y0, x1, y1 = caja
    T, fps = d["T"], d["video"]["fps"]
    dom = pe.mano_dominante(d)[0]
    base = 0 if dom == "izquierda" else 63
    aus = d["sin_izq"] if dom == "izquierda" else d["sin_der"]
    yv = d["manos"][:, base + 1].astype(float).copy(); yv[aus] = np.nan
    ymin, ymax = np.nanmin(yv), np.nanmax(yv)
    frames = []

    def lienzo(k, fase):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 1], wspace=0.15, hspace=0.35)
        a1 = fig.add_subplot(gs[:, 0])
        kk = min(k, T - 1)
        a1.imshow(dibujar(d["cuadros"][kk], d, kk)[y0:y1, x0:x1]); a1.axis("off")
        a1.set_title(f"1-2. Video y puntos de MediaPipe · cuadro {kk + 1} de {T}", fontsize=13)
        a2 = fig.add_subplot(gs[0, 1])
        sombrear_sin_mano(a2, aus[:kk + 1])
        a2.plot(np.arange(kk + 1) + 1, yv[:kk + 1], color=AZUL, lw=2)
        a2.set_xlim(0, T + 1); a2.set_ylim(ymax + 0.1, ymin - 0.1)
        a2.set_title(f"3. Altura de la muñeca {dom} (normalizada)", fontsize=12)
        a2.set_xlabel("cuadro")
        if fase >= 1:
            a2.scatter(d["posiciones"] + 1, d["remuestreo"][:, base + 1], s=45, marker="D", facecolor="white",
                       edgecolor=NARANJA, linewidths=1.8, zorder=3)
            a2.text(0.98, 0.95, "30 pasos", transform=a2.transAxes, color=NARANJA, fontsize=12, fontweight="bold",
                    ha="right", va="top")
            a2.text(0.98, 0.80, "gris: mano no vista → 0", transform=a2.transAxes, color="#6B7280", fontsize=10,
                    ha="right", va="top")
        a3 = fig.add_subplot(gs[1, 1])
        if fase >= 2:
            orden = np.argsort(prob)[::-1][:5]
            a3.barh(range(5)[::-1], prob[orden] * 100,
                    color=[NARANJA if j == orden[0] else GRIS for j in orden])
            a3.set_yticks(range(5)[::-1]); a3.set_yticklabels([SIGN_VOCABULARY[int(j)] for j in orden], fontsize=12)
            a3.axvline(umbral * 100, color="#2E7D4F", ls="--", lw=1.5)
            a3.set_xlim(0, 100)
            g = SIGN_VOCABULARY[int(orden[0])]
            pct = f"{prob[orden[0]] * 100:.1f}".replace(".", ",")
            a3.set_title(f"4-5. La red dice: «{g}» ({pct} %)",
                         fontsize=13, color=NARANJA, fontweight="bold")
        else:
            a3.axis("off")
            a3.text(0.5, 0.5, "la red espera a que termine la seña…" if fase == 0 else "30 × 252 números → red neuronal",
                    ha="center", va="center", fontsize=13, color="#6B7280", transform=a3.transAxes)
        canvas = FigureCanvasAgg(fig); canvas.draw()
        arr = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return arr

    for k in range(T):
        frames.append(lienzo(k, 0))
    frames += [lienzo(T - 1, 1)] * int(fps * 1.5)
    frames += [lienzo(T - 1, 2)] * int(fps * 3)
    guardar_mp4(frames, ruta_mp4, fps)
    h, w = frames[0].shape[:2]
    from PIL import Image
    ims = [Image.fromarray(f).resize((w // 2, h // 2)) for f in frames[::2]]
    ims[0].save(ruta_gif, save_all=True, append_images=ims[1:], duration=int(2000 / fps), loop=0, optimize=True)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("muestra", nargs="?", default="0030_0000_0003")
    ap.add_argument("--salida", type=Path, required=True, help="carpeta FUERA del repositorio")
    a = ap.parse_args()
    salida = a.salida.resolve()
    if salida == RAIZ.resolve() or RAIZ.resolve() in salida.parents:
        raise SystemExit("La salida contiene imágenes de personas del corpus: debe quedar FUERA del repositorio "
                         "(07_etica.tex: no se redistribuye ni se publica).")
    salida.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pe._estilo(plt)
    import tensorflow as tf
    from src import representacion as rp
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        d = pe.camino_corpus(a.muestra, Path(tmp) / a.muestra)
    umbral = float(rp.leer_umbral(pe.MODELO)["umbral"])
    modelo = tf.keras.models.load_model(pe.MODELO)
    prob = np.asarray(modelo.predict(d["X"], verbose=0))[0]
    caja = caja_persona(d)
    m = a.muestra
    fig_cuadros(d, caja, salida / f"01_cuadros_{m}.png", plt, con_puntos=False)
    fig_cuadros(d, caja, salida / f"02_cuadros_con_puntos_{m}.png", plt, con_puntos=True)
    k_mano, dom = fig_mano(d, salida / f"03_mano_21_puntos_{m}.png", plt)
    fig_tabla_video(d, salida / f"04_video_como_tabla_{m}.png", plt)
    fig_normalizacion(d, caja, salida / f"05_normalizacion_{m}.png", plt)
    fig_remuestreo(d, caja, salida / f"06_30_pasos_{m}.png", plt)
    fig_matriz(d, salida / f"07_entrada_30x252_{m}.png", plt)
    fig_prediccion(d, prob, umbral, salida / f"08_prediccion_{m}.png", plt)
    animacion(d, caja, prob, umbral, salida / f"proceso_{m}.mp4", salida / f"proceso_{m}.gif", plt)
    j = int(np.argmax(prob))
    from src.grammar_mapper import SIGN_VOCABULARY
    print(f"muestra {m} «{d['glosa']}» · {d['T']} cuadros a {d['video']['fps']:g} fps "
          f"({d['T'] / d['video']['fps']:.3f} s) · partición aleatoria: {d['particion']['aleatoria']}")
    print(f"predicción: «{SIGN_VOCABULARY[j]}» {prob[j]:.4f} · umbral {umbral} · "
          f"{'ACIERTO' if j == d['sena_id'] else 'ERROR'} · cuadro de la mano: {k_mano + 1} ({dom})")
    print(f"salida: {salida}")


if __name__ == "__main__":
    main()
