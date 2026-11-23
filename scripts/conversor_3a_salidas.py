"""
Tabla comparativa conversor heredado → conversor por reglas (actividad 3.a).

Escribe docs/CONVERSOR_3a_SALIDAS.csv (UTF-8) con TODAS estas salidas, una fila por entrada:
  - tipo «sola»: las 50 glosas solas (modo aislado);
  - tipo «plantilla»: las 22 plantillas de §4.2 (versión 3a-2), con las glosas en el orden de la tabla;
  - tipo «par»: los 2.450 pares ORDENADOS de glosas distintas, en orden de id;
  - tipo «trío»: la muestra determinista de tríos ordenados de src.conversor_glosas.muestra_trios() (n=1000, semilla 2026)
    (la misma que recorren las pruebas de tests/test_conversor_glosas.py).
Columnas: tipo, entrada (glosas unidas por « + », en el orden de entrada), salida_heredada
(GrammarMapper().translate), salida_nueva (ConversorGlosas().translate), via, reglas, plantilla,
respaldo, motivo, suplido, supuestos.

Solo lee los dos módulos y escribe el CSV; no toca el corpus ni el modelo. Ninguna salida «nueva» está
revisada por la persona señante (Actividad B pendiente).

Uso (desde la raíz del repositorio):
    python scripts/conversor_3a_salidas.py                       # → docs/CONVERSOR_3a_SALIDAS.csv
    python scripts/conversor_3a_salidas.py --salida otra_ruta.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.conversor_glosas import PLANTILLAS, ConversorGlosas, muestra_trios  # noqa: E402
from src.grammar_mapper import SIGN_VOCABULARY, GrammarMapper  # noqa: E402

COLUMNAS = ["tipo", "entrada", "salida_heredada", "salida_nueva", "via", "reglas", "plantilla", "respaldo",
            "motivo", "suplido", "supuestos"]


def entradas():
    """(tipo, lista de glosas) en el orden en que van al CSV."""
    for i in sorted(SIGN_VOCABULARY):
        yield "sola", [SIGN_VOCABULARY[i]]
    for tid in sorted(PLANTILLAS):
        yield "plantilla", list(PLANTILLAS[tid]["glosas"])
    for a, b in itertools.permutations(sorted(SIGN_VOCABULARY), 2):
        yield "par", [SIGN_VOCABULARY[a], SIGN_VOCABULARY[b]]
    for trio in muestra_trios():
        yield "trío", trio


def filas() -> list[dict]:
    heredado = GrammarMapper()
    nuevo = ConversorGlosas()
    salida = []
    for tipo, glosas in entradas():
        e = nuevo.explicar(glosas)
        salida.append({
            "tipo": tipo,
            "entrada": " + ".join(glosas),
            "salida_heredada": heredado.translate(list(glosas)),
            "salida_nueva": e["frase"],
            "via": e["via"],
            "reglas": " ".join(e["reglas"]),
            "plantilla": e["plantilla"] or "",
            "respaldo": "sí" if e["respaldo"] else "no",
            "motivo": e["motivo"] or "",
            "suplido": " ".join(e["suplido"]),
            "supuestos": " ".join(e["supuestos"]),
        })
    return salida


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="CSV comparativo heredado → reglas (3.a)")
    parser.add_argument("--salida", default=str(ROOT_DIR / "docs" / "CONVERSOR_3a_SALIDAS.csv"))
    args = parser.parse_args(argv)
    datos = filas()
    with open(args.salida, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(datos)
    print(f"{args.salida}: {len(datos)} filas")
    for tipo, n in Counter(d["tipo"] for d in datos).items():
        vias = Counter(d["via"] for d in datos if d["tipo"] == tipo)
        print(f"  {tipo:9s} {n:5d}  " + ", ".join(f"{v} {c}" for v, c in sorted(vias.items())))


if __name__ == "__main__":
    main()
