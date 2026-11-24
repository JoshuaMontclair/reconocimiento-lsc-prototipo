"""
Qué lleva la coordenada z de las manos después de la normalización por hombros.

La normalización (src/preprocess_lsc50.py, normalize_sample_vectorized; la misma en src/translator.py)
resta a TODOS los puntos el punto medio de los hombros de la postura, también en z, y divide por el ancho
de hombros medido en x-y. En las manos, MediaPipe da z relativa a la muñeca (la z de la muñeca es ~0 en
la salida cruda), y la z de la postura es la que la documentación de Holistic pide descartar. Por eso,
tras normalizar, la z de cada punto de la mano es

    z' = (z_rel_muñeca - z_hombros) / d,

y la z normalizada de la muñeca es exactamente -z_hombros / d: un desplazamiento común, por cuadro, que
viene de la z de la postura.

Este análisis mide, sobre los cuadros con la mano detectada de los 997 videos válidos, qué parte de la
varianza de las 21 columnas z de cada mano es ese desplazamiento común (la z de la muñeca) y qué parte es
la forma de la mano (z' - z'_muñeca). No cambia nada del preprocesamiento ni del modelo.

SOLO ANÁLISIS. Determinista. Lee data/processed_holistic_mascara_rostro (el processed del modelo final) y,
para comprobar que la z cruda de la muñeca es ~0, data/extracted/LSC50_holistic.

Uso:
    python src/coordenada_z.py
Salida: docs/COORDENADA_Z.json y docs/COORDENADA_Z.md
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

PROCESSED = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
EXTRAIDOS = BASE_DIR / "data" / "extracted" / "LSC50_holistic"
JSON_SALIDA = BASE_DIR / "docs" / "COORDENADA_Z.json"
MD_SALIDA = BASE_DIR / "docs" / "COORDENADA_Z.md"
MANOS = {"izquierda": 99, "derecha": 162}                            # columna inicial de cada bloque de 63


def z_cruda_muneca(n_archivos: int = 50) -> dict:
    """|z| de la muñeca (punto 0) en los CSV crudos de las manos: debe ser ~0 (z relativa a la muñeca)."""
    vals = []
    for carpeta in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
        for f in sorted((EXTRAIDOS / carpeta).glob("*.csv"))[:n_archivos]:
            for linea in f.read_text(encoding="utf-8").splitlines()[1:]:
                partes = linea.split(",")
                if len(partes) > 3:
                    vals.append(abs(float(partes[3])))
    v = np.array(vals)
    v = v[v > 0] if (v > 0).any() else v
    return {"archivos_por_mano": n_archivos, "n": int(v.size), "mediana_abs": float(np.median(v)),
            "max_abs": float(v.max())}


def main():
    meta = json.loads((PROCESSED / "metadata.json").read_text(encoding="utf-8"))
    nombres = list(meta["sample_order"])
    L = np.load(PROCESSED / "lengths.npy").astype(np.int64)
    X = np.load(PROCESSED / "X_padded.npy", mmap_mode="r")
    from src import train_loso as tl  # noqa: E402  (lista de duplicados de D-06)
    duplicados = set(tl.DUPLICADOS_EXCLUIDOS)

    out = {"fecha": dt.datetime.now().isoformat(timespec="seconds"), "script": "src/coordenada_z.py",
           "fuente": str(PROCESSED.relative_to(BASE_DIR)), "z_cruda_muneca": z_cruda_muneca(), "manos": {}}
    for mano, c0 in MANOS.items():
        bloques = []
        for i, n in enumerate(nombres):
            if n in duplicados:
                continue
            b = np.asarray(X[i, :L[i], c0:c0 + 63], dtype=np.float64).reshape(-1, 21, 3)
            detectada = np.any(b != 0, axis=(1, 2))          # «mascara»: bloque en ceros = mano ausente
            bloques.append(b[detectada])
        B = np.concatenate(bloques)                          # (cuadros, 21, 3)
        z = B[:, :, 2]
        muneca = z[:, :1]
        forma = z[:, 1:] - muneca
        var_total = float(z[:, 1:].var(axis=0).mean())       # varianza media de las 20 columnas z no-muñeca
        var_forma = float(forma.var(axis=0).mean())
        var_xy = float(B[:, 1:, :2].var(axis=0).mean())
        out["manos"][mano] = {
            "cuadros": int(z.shape[0]),
            "z_muneca_normalizada": {"mediana": float(np.median(muneca)), "p10": float(np.percentile(muneca, 10)),
                                     "p90": float(np.percentile(muneca, 90)), "de": float(muneca.std())},
            "z_forma_abs_mediana": float(np.median(np.abs(forma))),
            "varianza_media_columnas_z": var_total,
            "varianza_media_forma_z": var_forma,
            "frac_varianza_z_del_desplazamiento_comun": 1.0 - var_forma / var_total,
            "varianza_media_columnas_xy": var_xy,
        }
    JSON_SALIDA.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    def n(x, d=3):
        return f"{x:.{d}f}".replace(".", ",")

    zc = out["z_cruda_muneca"]
    Lm = ["# La coordenada z de las manos tras la normalización por hombros", "",
          f"Generado por `src/coordenada_z.py` el {out['fecha']}. Solo análisis; fuente `{out['fuente']}` "
          "(997 videos válidos, cuadros con la mano detectada).", "",
          "La normalización resta a todos los puntos el centro de los hombros también en z, y la z de la mano es "
          "relativa a la muñeca, así que la z normalizada de la muñeca es −z_hombros / d (la z de la postura, "
          f"que la documentación pide descartar). Comprobación en los CSV crudos: |z| de la muñeca, mediana "
          f"{zc['mediana_abs']:.1e}, máx. {zc['max_abs']:.1e} ({zc['n']} cuadros).", "",
          "| mano | cuadros | z' muñeca: mediana (p10–p90) | |z' − z'muñeca| mediana | varianza z' (media de 20 columnas) | de ella, desplazamiento común | varianza x', y' |",
          "|---|---|---|---|---|---|---|"]
    for mano, f in out["manos"].items():
        zm = f["z_muneca_normalizada"]
        Lm.append(f"| {mano} | {f['cuadros']} | {n(zm['mediana'], 2)} ({n(zm['p10'], 2)} a {n(zm['p90'], 2)}) | "
                  f"{n(f['z_forma_abs_mediana'])} | {n(f['varianza_media_columnas_z'])} | "
                  f"{n(100 * f['frac_varianza_z_del_desplazamiento_comun'], 1)} % | {n(f['varianza_media_columnas_xy'])} |")
    Lm += ["", "Lectura: casi toda la variación de las columnas z de la mano es el desplazamiento común que aporta "
           "la z de los hombros (la misma cantidad en los 21 puntos de una mano y un cuadro); la forma de la mano "
           "en profundidad es una parte pequeña. No hay en el repositorio una ablación que entrene sin z; su "
           "efecto en la exactitud no está medido.", ""]
    MD_SALIDA.write_text("\n".join(Lm), encoding="utf-8")
    print(json.dumps(out["manos"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
