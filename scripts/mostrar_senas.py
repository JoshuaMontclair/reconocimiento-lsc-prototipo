#!/usr/bin/env python3
"""Visor de las 50 señas del corpus LSC50, una por una, con su glosa en pantalla.

Para la revisión de glosas con la persona señante (actividad 1.d). Solo LEE los videos de
data/raw/LSC50/VIDEOS/COLOR_BODY; no escribe nada a disco.

Uso:
    python scripts/mostrar_senas.py                    # señante 0000, las 50 señas
    python scripts/mostrar_senas.py --senante 3        # otra persona del corpus (0 a 4)
    python scripts/mostrar_senas.py --desde 14         # empezar en la seña 14 (niño)
    python scripts/mostrar_senas.py --solo 14 15 26 44 # solo esas señas

Teclas:  ESPACIO o →  siguiente   ·   ←  anterior   ·   R  repetir
         O  ver la misma seña hecha por la otra persona (rota entre los 5 señantes)
         Q o ESC  salir
El video se repite solo hasta que se pasa a otra seña.
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402

VIDEOS = ROOT / "data" / "raw" / "LSC50" / "VIDEOS" / "COLOR_BODY"
VENTANA = "Senas LSC50 - revision de glosas"


def ruta(sena: int, senante: int) -> Path:
    return VIDEOS / f"{sena:04d}_{senante:04d}_0000.avi"


def rotulo(frame, sena: int, senante: int, pos: int, total: int):
    h, w = frame.shape[:2]
    escala = w / 1920
    texto = f"{sena:02d}  {SIGN_VOCABULARY[sena]}"
    sub = f"senante {senante:04d}   ({pos + 1}/{total})   ESPACIO sig. | <- ant. | R repetir | O otra persona | Q salir"
    cv2.rectangle(frame, (0, 0), (w, int(150 * escala)), (0, 0, 0), -1)
    cv2.putText(frame, texto, (int(30 * escala), int(85 * escala)), cv2.FONT_HERSHEY_SIMPLEX,
                2.4 * escala, (255, 255, 255), max(2, int(5 * escala)), cv2.LINE_AA)
    cv2.putText(frame, sub, (int(30 * escala), int(135 * escala)), cv2.FONT_HERSHEY_SIMPLEX,
                0.9 * escala, (200, 200, 200), max(1, int(2 * escala)), cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser(description="Visor de las señas de LSC50 con su glosa")
    ap.add_argument("--senante", type=int, default=0, help="persona del corpus, 0 a 4 (defecto 0)")
    ap.add_argument("--desde", type=int, default=0, help="número de seña por el que empezar")
    ap.add_argument("--solo", type=int, nargs="+", help="mostrar solo estas señas")
    a = ap.parse_args()

    senas = a.solo if a.solo else list(range(50))
    i = senas.index(a.desde) if (not a.solo and a.desde in senas) else 0
    senante = a.senante
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA, 1280, 720)

    while 0 <= i < len(senas):
        sena = senas[i]
        p = ruta(sena, senante)
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            print(f"[AVISO] no se pudo abrir {p.name}")
            i += 1
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        espera = max(1, int(1000 / fps))
        accion = None
        while accion is None:
            ok, frame = cap.read()
            if not ok:  # fin del video: repetir
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cv2.imshow(VENTANA, rotulo(frame, sena, senante, i, len(senas)))
            k = cv2.waitKeyEx(espera)
            if k in (ord("q"), ord("Q"), 27):
                accion = "salir"
            elif k in (32, 63235, 2555904, 65363):          # espacio o flecha derecha
                accion = "sig"
            elif k in (63234, 2424832, 65361):               # flecha izquierda
                accion = "ant"
            elif k in (ord("r"), ord("R")):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif k in (ord("o"), ord("O")):
                accion = "otra"
        cap.release()
        if accion == "salir":
            break
        if accion == "sig":
            i += 1
        elif accion == "ant":
            i = max(0, i - 1)
        elif accion == "otra":
            senante = (senante + 1) % 5
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
