"""
Tabla por seña del apéndice de las diapositivas (LOSO agregado, 997 predicciones, modelo «manos»).

Lee docs/diapositivas_workflow/figuras/res_metricas.json (clave «por_sena», que escribe
resultados_metricas.py a partir de models/loso_2026-09-15_holistic_mascara_manos/pliegue_k/predicciones.json)
y escribe dos fragmentos LaTeX (señas 0-24 y 25-49) para \\input dentro de un frame:

    figuras/tabla_por_sena_1.tex
    figuras/tabla_por_sena_2.tex

No lee ni escribe nada más. Uso (desde la raíz del repositorio):
    python docs/diapositivas_workflow/guiones/tabla_por_sena.py
"""
import json
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figuras"


def coma(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def escapar(s):
    return s.replace("¿", "¿").replace("_", r"\_")


def fragmento(filas):
    out = [
        r"\begin{tabular}{@{}r l r r r r r r@{}}",
        r"\toprule",
        r"ID & Seña & Videos & TP & FN & FP & Recall & Precisión \\",
        r"\midrule",
    ]
    for f in filas:
        marca = r"\rowcolor{naranja!15}" if f["recall"] <= 0.25 else (
            r"\rowcolor{verde!15}" if f["recall"] >= 0.80 else "")
        out.append(
            f"{marca}{f['id']} & {escapar(f['glosa'])} & {f['soporte']} & {f['TP']} & {f['FN']} & "
            f"{f['FP']} & {coma(f['recall'])} & {coma(f['precision'])} \\\\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def main():
    d = json.loads((FIG / "res_metricas.json").read_text(encoding="utf-8"))
    ps = sorted(d["por_sena"], key=lambda f: f["id"])
    assert len(ps) == 50
    assert sum(f["TP"] for f in ps) == d["loso_agregado_997"].get("aciertos", sum(f["TP"] for f in ps))
    assert sum(f["soporte"] for f in ps) == 997
    cab = ("% Generado por docs/diapositivas_workflow/guiones/tabla_por_sena.py desde "
           "figuras/res_metricas.json:por_sena (LOSO agregado, 997).\n")
    (FIG / "tabla_por_sena_1.tex").write_text(cab + fragmento(ps[:25]), encoding="utf-8")
    (FIG / "tabla_por_sena_2.tex").write_text(cab + fragmento(ps[25:]), encoding="utf-8")
    print("TP total:", sum(f["TP"] for f in ps), "soporte:", sum(f["soporte"] for f in ps))


if __name__ == "__main__":
    main()
