"""
Espera tras terminar la seña con la ventana fija de 2,5 s del prototipo.

L_resp = t5 - t1 se mide desde el CIERRE de la ventana (t1), no desde el fin de la seña. Con la ventana
fija, quien termina la seña antes de que la ventana cierre espera primero a que cierre y después L_resp.
Esta espera es

    espera tras la seña = (cierre de la ventana - fin de la seña) + L_resp.

SOLO ANÁLISIS: no cambia el prototipo, ni modelos, ni umbrales, ni particiones. No escribe en data/.

Datos y supuestos:
  * Fin (e inicio) de la seña de cada clip: el tramo activo de src/simular_ventana_25.py (tramo_sena:
    reposo = muñecas quietas y bajas), sobre los 997 clips válidos del video COLOR_BODY a 24 fps
    (data/processed_holistic_mascara_rostro). Es la misma medida de docs/SIMULACION_VENTANA_25.json.
  * Supuesto del protocolo (principal): la ventana abre al inicio del clip, es decir, la persona pulsa
    ESPACIO y arranca desde el reposo, como en el corpus (reposo inicial mediano de 0,33 s). La ventana
    cierra a los 2,5 s (60 fotogramas a 24 fps).
  * Solo cuentan los clips cuya seña termina dentro de la ventana; en los demás la ventana corta la seña
    y la espera tras la seña no está definida (se informa cuántos son).
  * L_resp: la mediana de la sesión en vivo del 23-nov (registros/sesion_final_2026-11-23, ensayos con
    audio, L_resp_ns). Se suma la mediana a cada clip; además se da la distribución que resulta de sumar
    a cada clip cada uno de los L_resp medidos (convolución empírica), que apenas cambia por la poca
    dispersión de L_resp.
  * Sensibilidad: la ventana abre D s antes del inicio de la seña (D = 0,25 / 0,5 / 1,0), en vez de al
    inicio del clip; entonces la espera es 2,5 - D - duración de la seña.

Además (avisos de seña cortada): tasa de aviso_fin / aviso_inicio de la cadena integrada sobre los 997
videos (models/evaluacion_punta_a_punta_2026-11-23/crudo.json, fuente video_1280x720, modelos LOSO) con
el clip completo y con los primeros 2,5 s, frente a la de la sesión en vivo del 23-nov. El campo
«aviso_bordes» de docs/EVALUACION_PUNTA_A_PUNTA.json cuenta inicio O fin; aquí se separan.

Determinista: sin aleatoriedad.

Uso:
    python src/espera_tras_sena.py
Salida: docs/ESPERA_TRAS_SENA.json y docs/ESPERA_TRAS_SENA.md
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import simular_ventana_25 as sv  # noqa: E402

JSON_SALIDA = BASE_DIR / "docs" / "ESPERA_TRAS_SENA.json"
MD_SALIDA = BASE_DIR / "docs" / "ESPERA_TRAS_SENA.md"
RETRASOS_S = (0.25, 0.5, 1.0)
CRUDO_PUNTA = BASE_DIR / "models" / "evaluacion_punta_a_punta_2026-11-23" / "crudo.json"


def resumen_ms(a) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {"n": int(a.size), "p10_ms": round(float(np.percentile(a, 10)), 1),
            "mediana_ms": round(float(np.median(a)), 1), "media_ms": round(float(a.mean()), 1),
            "p90_ms": round(float(np.percentile(a, 90)), 1), "min_ms": round(float(a.min()), 1),
            "max_ms": round(float(a.max()), 1)}


def l_resp_vivo_ms() -> np.ndarray:
    vals = []
    for linea in sv.REGISTRO_VIVO.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        r = json.loads(linea)
        if r.get("tipo") == "ensayo" and r.get("L_resp_ns") is not None:
            vals.append(r["L_resp_ns"] / 1e6)
    return np.array(sorted(vals))


def avisos_corpus() -> dict:
    """Avisos de borde (N-50) de la cadena integrada sobre los 997 videos (src/evaluacion_punta_a_punta.py):
    por video, con la fuente video_1280x720 (la resolución del prototipo) y los modelos LOSO. Los avisos
    dependen solo de los puntos de la ventana, no del modelo; se comprueba que coinciden entre semillas.
    El campo «aviso_bordes» del informe cuenta inicio O fin; aquí se separan."""
    if not CRUDO_PUNTA.exists():
        return {"disponible": False}
    filas = json.loads(CRUDO_PUNTA.read_text(encoding="utf-8"))["filas"]
    out = {"disponible": True, "fuente": str(CRUDO_PUNTA.relative_to(BASE_DIR)), "fuente_puntos": "video_1280x720"}
    for ventana in ("clip", "v2_5s"):
        por_video = {}
        for f in filas:
            if (f["fuente"] == "video_1280x720" and f["ventana"] == ventana and f["modelo"].startswith("loso")
                    and f["aviso_inicio"] is not None):
                por_video.setdefault(f["video"], set()).add((bool(f["aviso_inicio"]), bool(f["aviso_fin"])))
        assert all(len(v) == 1 for v in por_video.values()), "avisos distintos entre semillas"
        pares = [next(iter(v)) for v in por_video.values()]
        n = len(pares)
        ini = sum(a for a, _ in pares)
        fin = sum(b for _, b in pares)
        alguno = sum(a or b for a, b in pares)
        out[ventana] = {"n_videos": n, "aviso_inicio": ini, "aviso_fin": fin, "aviso_inicio_o_fin": alguno,
                        "frac_aviso_inicio": round(ini / n, 4), "frac_aviso_fin": round(fin / n, 4),
                        "frac_aviso_inicio_o_fin": round(alguno / n, 4)}
    vivo = sv.avisos_en_vivo()
    if vivo.get("disponible"):
        n = vivo["ventanas_aislado_con_bordes"]
        vivo["frac_aviso_fin"] = round(vivo["aviso_fin"] / n, 4)
        vivo["frac_aviso_inicio"] = round(vivo["aviso_inicio"] / n, 4)
    out["vivo_23_nov"] = vivo
    return out


def main():
    meta = json.loads((sv.PROCESSED / "metadata.json").read_text(encoding="utf-8"))
    nombres = list(meta["sample_order"])
    lengths = np.load(sv.PROCESSED / "lengths.npy").astype(np.int64)
    X = np.load(sv.PROCESSED / "X_padded.npy", mmap_mode="r")
    validas = sv.mascara_validas(nombres)

    tramos = {}
    for i in np.flatnonzero(validas):
        tramos[nombres[i]] = sv.tramo_sena(X[i, :lengths[i], :sv.COL_POSTURA])
    ok = [n for n in nombres if tramos.get(n) is not None]
    fin = np.array([tramos[n][1] + 1 for n in ok])            # fotogramas desde el inicio del clip al fin de la seña
    ini = np.array([tramos[n][0] for n in ok])
    dur = fin - ini                                            # fotogramas de seña
    senante = np.array([n.split("_")[1] for n in ok])

    lr = l_resp_vivo_ms()
    lr_med = float(np.median(lr))

    dentro = fin <= sv.N_VENTANA
    hueco_ms = (sv.N_VENTANA - fin[dentro]) / sv.FPS_CORPUS * 1000.0
    espera_ms = hueco_ms + lr_med
    conv = (hueco_ms[:, None] + lr[None, :]).ravel()

    datos = {
        "fecha": dt.datetime.now().isoformat(timespec="seconds"), "script": "src/espera_tras_sena.py",
        "definicion": "espera tras la seña = (cierre de la ventana - fin de la seña) + L_resp; L_resp = t5 - t1",
        "supuesto": "la ventana abre al inicio del clip (la persona arranca desde el reposo tras pulsar ESPACIO) y "
                    "cierra a los 2,5 s (60 fotogramas a 24 fps); fin de la seña = último fotograma activo de "
                    "src/simular_ventana_25.py (tramo_sena)",
        "fuente_tramos": "data/processed_holistic_mascara_rostro (997 clips válidos, video COLOR_BODY, 24 fps)",
        "fuente_l_resp": str(sv.REGISTRO_VIVO.relative_to(BASE_DIR)),
        "l_resp_vivo": resumen_ms(lr),
        "n_clips": len(ok),
        "n_sena_termina_dentro": int(dentro.sum()),
        "frac_sena_termina_dentro": round(float(dentro.mean()), 4),
        "n_sena_cortada": int((~dentro).sum()),
        "cierre_menos_fin_sena": resumen_ms(hueco_ms),
        "espera_tras_sena_con_mediana_l_resp": resumen_ms(espera_ms),
        "espera_tras_sena_convolucion_l_resp": resumen_ms(conv),
        "frac_espera_le_200ms": round(float((espera_ms <= 200).mean()), 4),
        "frac_espera_le_500ms": round(float((espera_ms <= 500).mean()), 4),
        "frac_espera_ge_1000ms": round(float((espera_ms >= 1000).mean()), 4),
        "por_senante": {},
        "sensibilidad_retraso": [],
        "avisos_bordes": avisos_corpus(),
    }
    for sn in sorted(set(senante)):
        m = (senante == sn)
        md = dentro & m
        datos["por_senante"][sn] = {"n_dentro": int(md.sum()), "n_total": int(m.sum()),
                                    **resumen_ms((sv.N_VENTANA - fin[md]) / sv.FPS_CORPUS * 1000.0 + lr_med)}
    for d in RETRASOS_S:
        dfot = int(round(d * sv.FPS_CORPUS))
        cabe = dur + dfot <= sv.N_VENTANA
        h = (sv.N_VENTANA - dfot - dur[cabe]) / sv.FPS_CORPUS * 1000.0 + lr_med
        datos["sensibilidad_retraso"].append({"retraso_s": d, "n_cabe": int(cabe.sum()),
                                              "frac_cabe": round(float(cabe.mean()), 4), **resumen_ms(h)})

    JSON_SALIDA.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")

    def s(x):
        return f"{x / 1000:.2f}".replace(".", ",")

    def ms(x):
        return f"{x:.1f}".replace(".", ",")

    def pc(x):
        return f"{100 * x:.1f}".replace(".", ",")

    e = datos["espera_tras_sena_con_mediana_l_resp"]
    c = datos["espera_tras_sena_convolucion_l_resp"]
    h = datos["cierre_menos_fin_sena"]
    L = ["# Espera tras terminar la seña con la ventana fija de 2,5 s", "",
         f"Generado por `src/espera_tras_sena.py` el {datos['fecha']}. Solo análisis.", "",
         "Definición: espera tras la seña = (cierre de la ventana − fin de la seña) + L_resp, con L_resp = t5 − t1 "
         "medido desde el cierre de la ventana.", "",
         f"Supuesto: {datos['supuesto']}.", "",
         f"- L_resp en vivo (23-nov, {datos['l_resp_vivo']['n']} ensayos con audio): mediana "
         f"{ms(datos['l_resp_vivo']['mediana_ms'])} ms, p10 {ms(datos['l_resp_vivo']['p10_ms'])} ms, p90 "
         f"{ms(datos['l_resp_vivo']['p90_ms'])} ms.",
         f"- Clips: {datos['n_clips']}; la seña termina dentro de la ventana en {datos['n_sena_termina_dentro']} "
         f"({pc(datos['frac_sena_termina_dentro'])} %); en {datos['n_sena_cortada']} la ventana la corta.",
         f"- Cierre − fin de la seña: mediana {s(h['mediana_ms'])} s, p10 {s(h['p10_ms'])} s, p90 {s(h['p90_ms'])} s.",
         f"- **Espera tras la seña** (con la mediana de L_resp): mediana {s(e['mediana_ms'])} s, p10 "
         f"{s(e['p10_ms'])} s, p90 {s(e['p90_ms'])} s, máx. {s(e['max_ms'])} s.",
         f"- Con todos los L_resp medidos (convolución): mediana {s(c['mediana_ms'])} s, p10 {s(c['p10_ms'])} s, "
         f"p90 {s(c['p90_ms'])} s.",
         f"- Espera ≤ 0,2 s: {pc(datos['frac_espera_le_200ms'])} %; ≤ 0,5 s: {pc(datos['frac_espera_le_500ms'])} %; "
         f"≥ 1 s: {pc(datos['frac_espera_ge_1000ms'])} %.", "",
         "Por señante:", "", "| señante | clips dentro / total | mediana (s) | p10 (s) | p90 (s) |", "|---|---|---|---|---|"]
    for sn, f in datos["por_senante"].items():
        L.append(f"| {sn} | {f['n_dentro']} / {f['n_total']} | {s(f['mediana_ms'])} | {s(f['p10_ms'])} | {s(f['p90_ms'])} |")
    L += ["", "Sensibilidad: la ventana abre D s antes del inicio de la seña:", "",
          "| D (s) | la seña cabe | mediana (s) | p10 (s) | p90 (s) |", "|---|---|---|---|---|"]
    for f in datos["sensibilidad_retraso"]:
        L.append(f"| {f['retraso_s']:.2f}".replace(".", ",") + f" | {pc(f['frac_cabe'])} % | {s(f['mediana_ms'])} | "
                 f"{s(f['p10_ms'])} | {s(f['p90_ms'])} |")
    av = datos["avisos_bordes"]
    if av.get("disponible"):
        L += ["", "## Avisos de seña cortada: corpus frente a en vivo", "",
              f"Corpus: cadena integrada sobre {av['v2_5s']['n_videos']} videos (`{av['fuente']}`, puntos "
              "`video_1280x720`, modelos LOSO; los avisos no dependen del modelo). En vivo: "
              f"`{av['vivo_23_nov'].get('fuente', '—')}`, ventanas en modo aislado.", "",
              "| fuente | ventana | aviso de fin | aviso de inicio | inicio o fin |", "|---|---|---|---|---|"]
        for ventana, etiqueta in (("clip", "clip completo"), ("v2_5s", "primeros 2,5 s")):
            f = av[ventana]
            L.append(f"| corpus | {etiqueta} | {f['aviso_fin']} ({pc(f['frac_aviso_fin'])} %) | "
                     f"{f['aviso_inicio']} ({pc(f['frac_aviso_inicio'])} %) | "
                     f"{f['aviso_inicio_o_fin']} ({pc(f['frac_aviso_inicio_o_fin'])} %) |")
        v = av["vivo_23_nov"]
        if v.get("disponible"):
            L.append(f"| en vivo 23-nov | 2,5 s | {v['aviso_fin']} de {v['ventanas_aislado_con_bordes']} "
                     f"({pc(v['frac_aviso_fin'])} %) | {v['aviso_inicio']} ({pc(v['frac_aviso_inicio'])} %) | — |")
    L += ["", "Límites: el inicio de la ventana en vivo depende de cuándo pulsa la persona y cuándo arranca; aquí se "
          "toma el del clip del corpus. El fin de la seña sale de un criterio heurístico de reposo (ver "
          "docs/SIMULACION_VENTANA_25.md, tabla de sensibilidad). La espera está dominada por la ventana fija, que "
          "es una decisión de diseño, no por el cómputo.", ""]
    MD_SALIDA.write_text("\n".join(L), encoding="utf-8")
    print(json.dumps({k: datos[k] for k in ("n_sena_termina_dentro", "espera_tras_sena_con_mediana_l_resp",
                                            "espera_tras_sena_convolucion_l_resp", "l_resp_vivo",
                                            "sensibilidad_retraso", "avisos_bordes")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
