"""Regenera solo la figura del esqueleto (escritorio_a_esqueleto_<muestra>.pdf) sin repetir la prueba completa.

Usa las mismas funciones que prueba_escritorio.py (camino_corpus: extracción con MediaPipe y normalización;
fig_esqueleto: dibujo), así que la figura es idéntica a la que produce la prueba de escritorio. No carga el
modelo, no abre la cámara ni reproduce audio; solo escribe la figura en --figuras. Ninguna imagen de la
persona: solo el esqueleto de puntos sobre fondo blanco (07_etica.tex).

    python docs/diapositivas_workflow/guiones/figura_esqueleto.py 0030_0000_0000 --figuras docs/diapositivas_workflow/figuras
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prueba_escritorio as pe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("muestra", nargs="?", default="0030_0000_0000")
    ap.add_argument("--figuras", type=Path, default=Path(__file__).resolve().parents[1] / "figuras")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pe._estilo(plt)
    with tempfile.TemporaryDirectory() as tmp:
        d = pe.camino_corpus(a.muestra, Path(tmp) / a.muestra)
    ruta = a.figuras / f"escritorio_a_esqueleto_{a.muestra}.pdf"
    info = pe.fig_esqueleto(d, ruta, plt)
    print(f"figura: {ruta}  {info}")


if __name__ == "__main__":
    main()
