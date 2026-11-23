"""
Experimento EXPLICATIVO: ¿por qué reducir las velocidades a 6 (media por mano y eje) empeora con imputación y no
con «mascara»? (docs/EXPERIMENTO_HIPOTESIS_VELOCIDAD_DEDOS.md). No es un experimento de adopción.

Hipótesis 1. Promediar las 21 velocidades de una mano conserva el movimiento global y borra el movimiento
RELATIVO de los dedos (abrir/cerrar la mano, doblar dedos): poca energía, pero discriminativo.
Hipótesis 2. Con «mascara», las velocidades por punto están dominadas por saltos falsos (la mano cae a 0 y
vuelve), que tapan el componente de los dedos; por eso reducirlas no cambiaba nada con «mascara».

Brazos (todos con la TCN, CONFIG_T exacta, y la receta de la confirmación; semillas 2026-2028 × 5 pliegues):
  T0  mascara    + 252 (126 pos + 126 vel)            REUTILIZADO (arquitecturas_literatura T_tcn)
  T3  mascara    + 132 (126 pos + 6 vel)              REUTILIZADO (factorial_tcn)
  T1  imputacion + 252                                REUTILIZADO (confirmacion_tcn)
  T2  imputacion + 132                                REUTILIZADO (confirmacion_tcn)
  T4  pivote     + 252 / T5 pivote + 132              REUTILIZADOS (factorial_tcn), solo descriptivos
  H1  imputacion + 126 pos + g (6) + r (126) = 258    NUEVO (control: reparametrización lineal de T1)
  H2  imputacion + 126 pos + r (126)         = 252    NUEVO (sin velocidad global)
  H3  mascara    + 126 pos + r (126)         = 252    NUEVO

g y r: src/velocidad_dedos.py, calculadas DESPUÉS de la aumentación y del remuestreo (sustituyen a add_velocity
en la tupla canon). Nada del entrenamiento se reimplementa: experimento_arquitecturas.entrenar_corrida
(constructor «T»).

Subcomandos:
  verificar   configuración de los reutilizados, entradas (incluido el espejo con datos reales), parámetros y
              reproducción byte a byte de T1 (semilla 2026, pliegue 0) con la ruta de entrada de este script.
  analisis    sin entrenar: fracción de energía g/r, saltos falsos, sonda lineal LOSO y figura.
  entrenar    combinaciones (brazo, semilla, pliegue) de H1-H3 que falten; reanudable.
  humo        comprueba los modelos de una prueba de humo (recarga, forma de entrada, parámetros).
  resumen     agrega, aplica el criterio prerregistrado y escribe el JSON de datos crudos.

Uso:
  python src/experimento_hipotesis_velocidad_dedos.py verificar --dest <carpeta_temporal>/repro_t1
  python src/experimento_hipotesis_velocidad_dedos.py entrenar --semillas 2026 --folds 0, --epochs 3 --out <scratch>
  python src/experimento_hipotesis_velocidad_dedos.py humo --out <scratch>
  python src/experimento_hipotesis_velocidad_dedos.py analisis
  python src/experimento_hipotesis_velocidad_dedos.py entrenar --semillas 2026      # una semilla por proceso
  python src/experimento_hipotesis_velocidad_dedos.py resumen --docs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import train_loso as tl  # noqa: E402
from src import arquitecturas_literatura as al  # noqa: E402
from src import experimento_arquitecturas as ea  # noqa: E402
from src import experimento_confirmacion_tcn as ect  # noqa: E402
from src import experimento_factorial_tcn as ef  # noqa: E402
from src import mano_ausente_variantes as mav  # noqa: E402
from src import velocidades_reducidas as vr  # noqa: E402
from src import velocidad_dedos as vd  # noqa: E402

PROCESSED_CANONICO = ea.PROCESSED_CANONICO
REPRESENTACION = ea.REPRESENTACION
SEQ_LENGTH = ea.SEQ_LENGTH
DEFAULT_OUT = BASE_DIR / "models" / "hipotesis_velocidad_dedos_2026-10-03"
JSON_DOCS = BASE_DIR / "docs" / "EXPERIMENTO_HIPOTESIS_VELOCIDAD_DEDOS.json"
FIGURAS = BASE_DIR / "docs" / "figuras_2026-10-03"
SEMILLAS = ect.SEMILLAS
PISO_MARGEN = ect.PISO_MARGEN
TOL = ect.TOL
AZAR = 1 / 50
VIDEO_HOLA = "0030_0000_0003"

BRAZOS = {
    "T0": {"regla": "mascara", "forma": "actual", "n_rasgos": 252, "origen": "reutilizado",
           "descripcion": "mascara + 252 (126 pos + 126 vel)"},
    "T3": {"regla": "mascara", "forma": "por_coordenada", "n_rasgos": 132, "origen": "reutilizado",
           "descripcion": "mascara + 132 (126 pos + 6 vel: g)"},
    "T1": {"regla": "imputacion", "forma": "actual", "n_rasgos": 252, "origen": "reutilizado",
           "descripcion": "imputacion + 252"},
    "T2": {"regla": "imputacion", "forma": "por_coordenada", "n_rasgos": 132, "origen": "reutilizado",
           "descripcion": "imputacion + 132 (g)"},
    "T4": {"regla": "pivote", "forma": "actual", "n_rasgos": 252, "origen": "reutilizado",
           "descripcion": "pivote + 252 (descriptivo)"},
    "T5": {"regla": "pivote", "forma": "por_coordenada", "n_rasgos": 132, "origen": "reutilizado",
           "descripcion": "pivote + 132 (descriptivo)"},
    "H1": {"regla": "imputacion", "forma": "g_r", "n_rasgos": 258, "origen": "nuevo", "carpeta": "H1_imputacion_g_r_258",
           "descripcion": "imputacion + 126 pos + g (6) + r (126) = 258 (control)"},
    "H2": {"regla": "imputacion", "forma": "r", "n_rasgos": 252, "origen": "nuevo", "carpeta": "H2_imputacion_r_252",
           "descripcion": "imputacion + 126 pos + r (126) = 252 (sin velocidad global)"},
    "H3": {"regla": "mascara", "forma": "r", "n_rasgos": 252, "origen": "nuevo", "carpeta": "H3_mascara_r_252",
           "descripcion": "mascara + 126 pos + r (126) = 252"},
}
NUEVOS = ("H1", "H2", "H3")
REUTILIZADOS = ("T0", "T3", "T1", "T2", "T4", "T5")
BRAZOS_SIGMA = ("T0", "T3", "T1", "T2", "H1", "H2", "H3")   # σ agrupada (prerregistrado)
N_PARAM = {252: 156_402, 132: 125_682, 258: 157_938}
C_SONDA = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def _lista_int(texto):
    return [int(t) for t in str(texto).split(",") if t.strip() != ""]


def dir_corrida(out, brazo, semilla, k):
    if brazo in REUTILIZADOS:
        return ef.dir_corrida(ef.DEFAULT_OUT, brazo, semilla, k)
    return Path(out) / BRAZOS[brazo]["carpeta"] / f"semilla_{semilla}" / f"pliegue_{k}"


# ---------------------------------------------------------------------------
# Datos y entradas
# ---------------------------------------------------------------------------

def preparar(processed=PROCESSED_CANONICO):
    """X_seq «mascara» (train_loso.remuestrear_todas) e «imputacion» (remuestrear_todas_con_regla)."""
    X_padded, y, lengths, nombres, senantes, loso, pos = ea.cargar_todo(processed)
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    X_seq = {"mascara": tl.remuestrear_todas(X_padded, lengths, canon[3], SEQ_LENGTH),
             "imputacion": mav.remuestrear_todas_con_regla(X_padded, lengths, "imputacion", SEQ_LENGTH)}
    return X_seq, y, nombres, senantes, loso, pos, canon, (X_padded, lengths)


def funcion_entrada_brazo(brazo, add_velocity):
    forma = BRAZOS[brazo]["forma"]
    if forma == "actual":
        return add_velocity
    if forma == "por_coordenada":
        return vr.funcion_entrada(forma)
    return vd.funcion_entrada(forma)


def canon_de_brazo(canon, brazo):
    """Tupla canon con canon[2] = función de entrada del brazo (se aplica DESPUÉS de aumentar)."""
    tf, build_model, add_velocity, resample_sequence, transformaciones = canon
    return (tf, build_model, funcion_entrada_brazo(brazo, add_velocity), resample_sequence, transformaciones)


# ---------------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------------

def comprobar_config_reutilizados():
    """Todas las corridas reutilizadas: CONFIG_T, receta, regla y entrada del brazo."""
    esperado_hp = json.loads(json.dumps(ect.hiperparametros_t()))
    filas = {"T0": ect.comprobar_config_t0()}
    for b in ("T1", "T2", "T3", "T4", "T5"):
        filas[b] = []
        for s in SEMILLAS:
            for k in range(5):
                d = dir_corrida(None, b, s, k)
                r = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
                assert r["brazo"] == b and r["semilla"] == s and r["pliegue"] == k, (b, s, k)
                assert r["hiperparametros"] == esperado_hp, (b, s, k)
                assert r["n_parametros"] == N_PARAM[BRAZOS[b]["n_rasgos"]], (b, s, k)
                assert (r["epochs_max"], r["aug"], r["patience"], r["seq_length"]) == (200, 10, 25, 30), (b, s, k)
                assert r["processed"] == "data/processed_holistic_mascara_rostro" and r["representacion"] == "manos"
                assert r["regla_mano_ausente"] == BRAZOS[b]["regla"] and r["forma_velocidades"] == BRAZOS[b]["forma"]
                assert r["n_rasgos_entrada"] == BRAZOS[b]["n_rasgos"] and r["n_val"] == 100
                filas[b].append({"semilla": s, "pliegue": k, "val_top1": r["val_top1"], "val_loss": r["val_loss"]})
    return filas


def comprobar_espejo(X: np.ndarray, forma: str) -> dict:
    """Espejar las posiciones y DESPUÉS calcular la entrada debe dar lo esperado a partir de la entrada sin
    espejar: posiciones espejadas; g con las manos intercambiadas y gx con el signo cambiado; r espejada como
    unas posiciones (manos intercambiadas, mismo orden de puntos, rx con el signo cambiado)."""
    from src.representacion import espejar
    E = vd.entrada_red(X, forma)
    Em = vd.entrada_red(espejar(X, REPRESENTACION), forma)
    esperado = [espejar(E[..., :126], REPRESENTACION)]
    if forma == "g_r":
        esperado.append(vr.espejo_esperado_reducido(E[..., 126:132], "por_coordenada"))
        esperado.append(espejar(E[..., 132:], REPRESENTACION))
    elif forma in ("r", "actual"):
        esperado.append(espejar(E[..., 126:], REPRESENTACION))
    esp = np.concatenate(esperado, axis=-1)
    return {"forma": forma, "max_abs_dif": float(np.max(np.abs(Em - esp))), "identico": bool(np.array_equal(Em, esp))}


def cmd_verificar(args):
    informe = {"fecha": datetime.now().isoformat(timespec="seconds"), "comprobaciones": []}

    def ok(texto):
        print(f"[ok] {texto}", flush=True)
        informe["comprobaciones"].append(texto)

    informe["reutilizados"] = comprobar_config_reutilizados()
    ok("T0-T5: las 90 corridas reutilizadas tienen CONFIG_T, la receta comun, 200 epocas, paciencia 25, aug x10, "
       "30 pasos, el processed canonico, «manos» y la regla y la entrada de su brazo")

    X_seq, y, nombres, senantes, loso, pos, canon, (X_padded, lengths) = preparar()
    tf, _, add_velocity, _, _ = canon
    Xc = ect.preparar()[0]
    assert np.array_equal(X_seq["mascara"], Xc["mascara"]) and np.array_equal(X_seq["imputacion"], Xc["imputacion"])
    ok("X_seq «mascara» e «imputacion» = las de experimento_confirmacion_tcn.preparar, bit a bit")

    # Entradas
    det = {}
    for regla in ("mascara", "imputacion"):
        X = X_seq[regla]
        V = add_velocity(X)[..., 126:]
        A = vd.entrada_red(X, "actual")
        assert np.array_equal(A, add_velocity(X)), regla
        GR = vd.entrada_red(X, "g_r")
        R = vd.entrada_red(X, "r")
        assert GR.shape == (len(X), 30, 258) and R.shape == (len(X), 30, 252)
        assert np.array_equal(GR[..., :126], X) and np.array_equal(R[..., :126], X)
        assert np.array_equal(GR[..., 132:], R[..., 126:])
        G6 = vr.entrada_red(X, "por_coordenada")[..., 126:]
        dif_g = float(np.max(np.abs(GR[..., 126:132] - G6)))
        rec = vd.reconstruir_v(GR[..., 126:132], GR[..., 132:])
        dif_rec = float(np.max(np.abs(rec - V)))
        suma_r = float(np.max(np.abs(vd.por_mano(R[..., 126:]).sum(axis=-2))))
        assert dif_g < 1e-5 and dif_rec < 1e-5 and suma_r < 1e-4, (regla, dif_g, dif_rec, suma_r)
        espejo = {f: comprobar_espejo(X, f) for f in ("actual", "g_r", "r")}
        for f, e in espejo.items():
            assert e["max_abs_dif"] < 1e-5, (regla, f, e)
        det[regla] = {"max_dif_g_vs_por_coordenada": dif_g, "max_dif_v_reconstruida_desde_g_r": dif_rec,
                      "max_abs_suma_de_r_por_mano": suma_r, "espejo": espejo}
    informe["entradas"] = det
    ok(f"entradas: «actual» = add_velocity bit a bit; g (6) = velocidades_reducidas «por_coordenada» "
       f"(max |dif| {max(d['max_dif_g_vs_por_coordenada'] for d in det.values()):.1e}); v = g + r "
       f"(max |dif| {max(d['max_dif_v_reconstruida_desde_g_r'] for d in det.values()):.1e}); Σ r = 0 por mano; "
       f"posiciones intactas")
    ok("espejo con datos reales (997+3 videos, mascara e imputacion): espejar y DESPUES calcular la entrada = "
       "entrada espejada esperada (manos intercambiadas, gx y rx con el signo cambiado); max |dif| "
       f"{max(e['max_abs_dif'] for d in det.values() for e in d['espejo'].values()):.1e}")

    n_par = {}
    for n in (252, 258):
        tf.keras.utils.set_random_seed(0)
        n_par[n] = int(al.construir_tcn(SEQ_LENGTH, n, 50).count_params())
        tf.keras.backend.clear_session()
        assert n_par[n] == N_PARAM[n], (n, n_par[n])
    informe["n_parametros"] = {b: N_PARAM[BRAZOS[b]["n_rasgos"]] for b in BRAZOS}
    ok(f"parametros: 252 rasgos -> {n_par[252]:,}; 258 -> {n_par[258]:,}")

    if args.dest:
        dest = Path(args.dest).resolve()
        d = dest / "T1_repro" / "semilla_2026" / "pliegue_0"
        if not (d / "resultado.json").exists():
            ns = argparse.Namespace(aug=10, epochs=200, patience=25, processed=str(PROCESSED_CANONICO))
            t0 = time.perf_counter()
            # Misma ruta que los brazos nuevos: canon_de_brazo de ESTE script.
            ea.entrenar_corrida("T", 2026, 0, loso["pliegues"][0], pos, X_seq["imputacion"], y,
                                canon_de_brazo(canon, "T1"), d, ns)
            print(f"  reproduccion en {time.perf_counter() - t0:.0f} s", flush=True)
        ref = dir_corrida(None, "T1", 2026, 0)
        comp = ect.comparar_corridas(ref, d)
        rr = json.loads((ref / "resultado.json").read_text(encoding="utf-8"))
        rn = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
        comp["val_loss"] = [rr["val_loss"], rn["val_loss"]]
        comp["probabilidades_npz_identicas"] = bool(comp["max_dif_probabilidades_val"] == 0.0
                                                    and comp["max_dif_probabilidades_test"] == 0.0)
        informe["reproduccion_T1_s2026_k0"] = comp
        print(json.dumps(comp, indent=1), flush=True)
        if comp["coincide"] and comp["predicciones_json_identico_byte_a_byte"] and comp["val_loss"][0] == comp["val_loss"][1]:
            ok("T1 semilla 2026 pliegue 0 reproducido con la ruta de entrada de este script: predicciones.json "
               f"identico byte a byte, historial identico: {comp['historial_identico']}, misma val_loss, max |dif| "
               f"de probabilidades val {comp['max_dif_probabilidades_val']:.1e} y prueba "
               f"{comp['max_dif_probabilidades_test']:.1e}")
        else:
            print("[FALLA] la reproduccion de T1 no coincide byte a byte", flush=True)
            informe["FALLA_reproduccion"] = True
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "verificacion.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {tl._rel(out / 'verificacion.json')}")


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def anotar_corrida(d: Path, brazo: str):
    from src.representacion import ruta_procedencia
    info = BRAZOS[brazo]
    rp = ruta_procedencia(d / "modelo.keras")
    proc = json.loads(rp.read_text(encoding="utf-8"))
    proc["mano_ausente"] = (info["regla"] if info["regla"] == "mascara" else
                            "imputacion (mano_ausente_variantes.regla_imputacion sobre el processed «mascara», "
                            "antes de remuestrear)")
    proc["velocidades"] = info["forma"]
    proc["n_rasgos_entrada"] = info["n_rasgos"]
    proc["columnas_entrada"] = vd.nombres_columnas(info["forma"])
    proc["funcion_entrada"] = f"src/velocidad_dedos.py entrada_red(X, {info['forma']!r}) tras remuestrear y aumentar"
    proc["script"] = "src/experimento_hipotesis_velocidad_dedos.py (via experimento_arquitecturas.entrenar_corrida)"
    proc.pop("brazo_experimento_arquitecturas", None)
    proc["brazo_hipotesis_velocidad_dedos"] = brazo
    proc["nota"] = ("modelo del experimento explicativo docs/EXPERIMENTO_HIPOTESIS_VELOCIDAD_DEDOS.md. NO es el "
                    "modelo canonico ni debe cargarse en el prototipo.")
    rp.write_text(json.dumps(proc, indent=1, ensure_ascii=False), encoding="utf-8")
    rr = d / "resultado.json"
    r = json.loads(rr.read_text(encoding="utf-8"))
    r["constructor"] = "T (arquitecturas_literatura.construir_tcn, CONFIG_T)"
    r["brazo"] = brazo
    r.update({"regla_mano_ausente": info["regla"], "forma_velocidades": info["forma"],
              "n_rasgos_entrada": info["n_rasgos"]})
    rr.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


def cmd_entrenar(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.processed = str(Path(args.processed).resolve())
    X_seq, y, nombres, senantes, loso, pos, canon, _ = preparar(Path(args.processed))
    brazos = [b.strip().upper() for b in args.brazos.split(",") if b.strip()]
    for b in brazos:
        if b not in NUEVOS:
            raise SystemExit(f"--brazos: {b} no se entrena aqui ({NUEVOS})")
    semillas = _lista_int(args.semillas)
    folds = tl.parsear_folds(args.folds, len(loso["pliegues"]))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] experimento_hipotesis_velocidad_dedos entrenar | brazos "
          f"{brazos} | semillas {semillas} | pliegues {folds} | epochs {args.epochs} | aug x{args.aug} | "
          f"paciencia {args.patience} | salida {tl._rel(out)}", flush=True)
    for s in semillas:
        for k in folds:
            p = loso["pliegues"][k]
            for b in brazos:
                d = dir_corrida(out, b, s, k)
                if (d / "resultado.json").exists():
                    print(f"  ya existe {tl._rel(d)}; se salta", flush=True)
                    continue
                print(f"\n{'=' * 72}\n  brazo {b} ({BRAZOS[b]['descripcion']}) | semilla {s} | pliegue {k} "
                      f"(senante {p['senante_test']}) | {datetime.now().isoformat(timespec='seconds')}\n{'=' * 72}",
                      flush=True)
                r = ea.entrenar_corrida("T", s, k, p, pos, X_seq[BRAZOS[b]["regla"]], y, canon_de_brazo(canon, b),
                                        d, args)
                assert r["n_parametros"] == N_PARAM[BRAZOS[b]["n_rasgos"]], (b, r["n_parametros"])
                anotar_corrida(d, b)
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] fin", flush=True)


def cmd_humo(args):
    """Recarga los modelos de una prueba de humo y comprueba forma de entrada, parámetros y procedencia."""
    out = Path(args.out).resolve()
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    tf = canon[0]
    X_seq, *_ = preparar()
    filas = []
    for b in NUEVOS:
        d = dir_corrida(out, b, 2026, 0)
        m = tf.keras.models.load_model(str(d / "modelo.keras"))
        proc = json.loads((d / "modelo.procedencia.json").read_text(encoding="utf-8"))
        r = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
        E = funcion_entrada_brazo(b, canon[2])(X_seq[BRAZOS[b]["regla"]][:8])
        p = m.predict(E, verbose=0)
        assert tuple(m.input_shape[1:]) == (30, BRAZOS[b]["n_rasgos"]) and E.shape[2] == BRAZOS[b]["n_rasgos"]
        assert int(m.count_params()) == N_PARAM[BRAZOS[b]["n_rasgos"]]
        assert proc["velocidades"] == BRAZOS[b]["forma"] and proc["brazo_hipotesis_velocidad_dedos"] == b
        assert r["regla_mano_ausente"] == BRAZOS[b]["regla"] and np.allclose(p.sum(1), 1, atol=1e-4)
        filas.append({"brazo": b, "input_shape": list(m.input_shape[1:]), "n_parametros": int(m.count_params()),
                      "epocas": r["epocas_ejecutadas"], "val_top1": r["val_top1"], "val_loss": r["val_loss"]})
        print(f"[ok] {b}: entrada {m.input_shape[1:]}, {m.count_params():,} param, procedencia {proc['velocidades']}",
              flush=True)
        del m
        tf.keras.backend.clear_session()
    res = {"fecha": datetime.now().isoformat(timespec="seconds"), "carpeta": str(out), "modelos": filas,
           "nota": "prueba de humo (3 epocas, semilla 2026, pliegue 0); sus cifras no se usan"}
    (out / "humo.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.copiar_a:
        Path(args.copiar_a).mkdir(parents=True, exist_ok=True)
        (Path(args.copiar_a) / "humo.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------
# Análisis sin entrenamiento
# ---------------------------------------------------------------------------

def energia(X_seq: np.ndarray, trans: np.ndarray, artif: np.ndarray) -> dict:
    """Energía (suma de cuadrados) de las velocidades, global (21·||g||²) y relativa (Σ||r_i||²), por mano y en
    total, en todos los pasos y en los pasos de transición."""
    V = vd.velocidades(X_seq)
    e_tot, e_g, e_r = vd.energias_por_paso(V)               # (N, 30, 2)
    res = {}
    for nombre, sel in (("izq", [0]), ("der", [1]), ("ambas", [0, 1])):
        et, eg, er = e_tot[..., sel], e_g[..., sel], e_r[..., sel]
        tr, ar = trans[..., sel], artif[..., sel]
        res[nombre] = {
            "energia_total": float(et.sum()), "energia_global": float(eg.sum()), "energia_relativa": float(er.sum()),
            "fraccion_global": float(eg.sum() / et.sum()), "fraccion_relativa": float(er.sum() / et.sum()),
            "energia_relativa_en_transiciones": float(er[tr].sum()), "energia_global_en_transiciones": float(eg[tr].sum()),
            "fraccion_de_r_en_pasos_de_transicion": float(er[tr].sum() / er.sum()),
            "fraccion_de_g_en_pasos_de_transicion": float(eg[tr].sum() / eg.sum()),
            "fraccion_de_r_en_pasos_artificiales_def_amplia": float(er[ar].sum() / er.sum()),
            "fraccion_relativa_fuera_de_transiciones": float(er[~tr].sum() / et[~tr].sum()),
            "fraccion_relativa_en_transiciones": float(er[tr].sum() / et[tr].sum()) if et[tr].sum() > 0 else None,
            "fraccion_de_pasos_de_transicion": float(tr[:, 1:].mean()),
            "fraccion_de_pasos_artificiales": float(ar[:, 1:].mean()),
            "n_pasos_mano_con_velocidad": int(np.prod(et[:, 1:].shape)),
        }
    return res


def sonda_lineal(F: np.ndarray, y: np.ndarray, loso: dict, pos: dict) -> dict:
    """Regresión logística multinomial (rasgos estandarizados con la media y la DE del entrenamiento). Por
    pliegue LOSO: se ajusta con «train», C se elige por exactitud en «val» (empate: menor log-loss en val; luego
    el C menor) y se evalúa una vez en «test» (el señante excluido)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import StandardScaler
    pliegues = []
    for k, p in enumerate(loso["pliegues"]):
        idx = {c: np.array([pos[n] for n in p["conjuntos"][c]]) for c in ("train", "val", "test")}
        sc = StandardScaler().fit(F[idx["train"]])
        Z = {c: sc.transform(F[i]) for c, i in idx.items()}
        cands = []
        for C in C_SONDA:
            m = LogisticRegression(C=C, max_iter=5000)
            m.fit(Z["train"], y[idx["train"]])
            pv = m.predict_proba(Z["val"])
            acc = float(np.mean(m.classes_[pv.argmax(1)] == y[idx["val"]]))
            ll = float(log_loss(y[idx["val"]], pv, labels=m.classes_))
            cands.append((-acc, ll, C, m, acc))
        cands.sort(key=lambda t: (t[0], t[1], t[2]))
        _, ll, C, m, acc = cands[0]
        pt = m.predict(Z["test"])
        pliegues.append({"pliegue": k, "senante_test": p["senante_test"], "C": C, "val_top1": acc,
                         "val_log_loss": ll, "test_top1": float(np.mean(pt == y[idx["test"]])),
                         "val_top1_por_C": {str(c[2]): c[4] for c in sorted(cands, key=lambda t: t[2])}})
    t = [f["test_top1"] for f in pliegues]
    return {"n_rasgos": int(F.shape[1]), "test_top1_media": float(np.mean(t)),
            "test_top1_de_entre_pliegues": float(np.std(t, ddof=1)),
            "val_top1_media": float(np.mean([f["val_top1"] for f in pliegues])), "pliegues": pliegues}


def elegir_video_configuracion(X_seq_imp, nombres, y, nombres_senas):
    """Seña con cambio de configuración, elegida con los datos: la clase con la mayor fracción mediana de energía
    relativa (r) con imputación entre sus videos; de ella, el video del señante 0000 más cercano a esa mediana
    (si no hay, el de cualquier señante)."""
    V = vd.velocidades(X_seq_imp)
    e_tot, e_g, e_r = vd.energias_por_paso(V)
    frac = e_r.sum(axis=(1, 2)) / np.maximum(e_tot.sum(axis=(1, 2)), 1e-12)
    clases = sorted(set(int(c) for c in y))
    med = {c: float(np.median(frac[y == c])) for c in clases}
    orden = sorted(clases, key=lambda c: -med[c])
    c = orden[0]
    idx = [i for i in range(len(nombres)) if y[i] == c and nombres[i].split("_")[1] == "0000"] or \
          [i for i in range(len(nombres)) if y[i] == c]
    i = min(idx, key=lambda i: abs(frac[i] - med[c]))
    ranking = [{"clase": int(cc), "sena": nombres_senas[cc], "fraccion_relativa_mediana": med[cc]} for cc in orden]
    return nombres[i], ranking, float(frac[i])


def figura(videos, X_seq, trans, nombres, y, nombres_senas, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    filas = []
    for v in videos:
        i = nombres.index(v)
        for regla in ("mascara", "imputacion"):
            V = vd.velocidades(X_seq[regla][i:i + 1])[0]
            g, r = vd.global_y_relativa(V)
            filas.append((v, regla, g, r, trans[i]))
    fig, axes = plt.subplots(len(filas), 2, figsize=(15, 3.1 * len(filas)),
                             gridspec_kw={"width_ratios": [1, 3.2]}, constrained_layout=True)
    info = {"videos": {}, "nota": "mapas de g (30 x 6) y r (30 x 126); escala simetrica por video y por columna, "
                                  "la misma para mascara e imputacion (percentil 99 de |valor| en ambas)"}
    for vid in videos:
        sub = [(j, f) for j, f in enumerate(filas) if f[0] == vid]
        lim_g = float(np.percentile(np.abs(np.concatenate([f[2][1:].ravel() for _, f in sub])), 99)) or 1e-6
        lim_r = float(np.percentile(np.abs(np.concatenate([f[3][1:].ravel() for _, f in sub])), 99)) or 1e-6
        i = nombres.index(vid)
        info["videos"][vid] = {"sena": nombres_senas[int(y[i])], "lim_g": lim_g, "lim_r": lim_r,
                               "pasos_transicion_izq": (np.flatnonzero(trans[i][:, 0]) + 1).tolist(),
                               "pasos_transicion_der": (np.flatnonzero(trans[i][:, 1]) + 1).tolist(), "energia": {}}
        for _, f in sub:
            e_tot, e_g, e_r = vd.energias_por_paso(vd.velocidades(X_seq[f[1]][i:i + 1]))
            tr = f[4][None]
            info["videos"][vid]["energia"][f[1]] = {
                "fraccion_relativa": float(e_r.sum() / e_tot.sum()),
                "fraccion_de_r_en_pasos_de_transicion": float(e_r[tr].sum() / e_r.sum()) if e_r.sum() else None,
                "max_abs_g": float(np.abs(f[2]).max()), "max_abs_r": float(np.abs(f[3]).max())}
        for j, f in sub:
            for col, (M, lim, nom, ancho) in enumerate(((f[2], lim_g, "g", 6), (f[3], lim_r, "r", 126))):
                ax = axes[j, col]
                im = ax.imshow(M.T, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                               norm=TwoSlopeNorm(0, -lim, lim), extent=(0.5, 30.5, ancho - 0.5, -0.5))
                ax.axhline(ancho / 2 - 0.5, color="k", lw=0.8)
                for m in range(2):
                    for t in np.flatnonzero(f[4][:, m]):
                        y0 = -0.5 + m * ancho / 2
                        ax.add_patch(plt.Rectangle((t + 0.5, y0), 1, ancho / 2, fill=False, ec="#222222", lw=0.6,
                                                   ls="--"))
                ax.set_yticks([ancho / 4 - 0.5, 3 * ancho / 4 - 0.5])
                ax.set_yticklabels(["mano izq.", "mano der."])
                ax.set_xlabel("paso")
                sena = nombres_senas[int(y[i])]
                ax.set_title((f"«{sena}» | {f[1]} | g: x, y, z por mano" if nom == "g" else
                              f"«{sena}» {vid} | {f[1]} | r = v − g: 21 puntos × (x, y, z) por mano"),
                             loc="left", fontsize=9)
                fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    fig.suptitle("Velocidad global g (izquierda) y velocidad de los puntos respecto a la mano r = v − g (derecha). "
                 "Recuadros discontinuos: pasos tocados por una transición presente↔ausente de la mano.\n"
                 "Misma escala de color para «mascara» e «imputación» dentro de cada video y columna.",
                 x=0.01, ha="left", fontsize=11)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return info


def cmd_analisis(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    X_padded, y, lengths, nombres, senantes, loso, pos = ea.cargar_todo(PROCESSED_CANONICO)
    from src.train_final_100 import resample_sequence
    X_seq = {"mascara": tl.remuestrear_todas(X_padded, lengths, resample_sequence, SEQ_LENGTH),
             "imputacion": mav.remuestrear_todas_con_regla(X_padded, lengths, "imputacion", SEQ_LENGTH)}
    usados = sorted({n for p in loso["pliegues"] for n in p["conjuntos"]["test"]}, key=lambda n: pos[n])
    iu = np.array([pos[n] for n in usados])
    assert len(iu) == 997
    trans, artif = vd.mascaras_transicion(X_padded, lengths, SEQ_LENGTH)
    # Comprobación: con «mascara», fuera de los pasos «artificiales» la velocidad es la real (la de imputación).
    Vm, Vi = vd.velocidades(X_seq["mascara"]), vd.velocidades(X_seq["imputacion"])
    fuera = ~artif
    dif_fuera = max(float(np.max(np.abs((Vm - Vi)[..., m * 63:(m + 1) * 63][fuera[..., m]]))) for m in range(2))

    res = {"fecha": datetime.now().isoformat(timespec="seconds"), "n_videos": int(len(iu)),
           "definiciones": {
               "energia": "suma de cuadrados; por mano y paso, Σ_i||v_i||² = 21·||g||² + Σ_i||r_i||²; pasos 2..30",
               "paso_de_transicion": "velocidad del paso t que mezcla fotogramas originales con la mano y sin ella "
                                     "(entre los fotogramas con peso > 0 en los pasos t−1 y t hay uno detectado y "
                                     "uno no detectado); definido sobre los fotogramas originales, el mismo para "
                                     "mascara e imputacion",
               "paso_artificial_def_amplia": "en t o t−1 interviene algun fotograma sin la mano "
                                             "(docs/EXPERIMENTO_VELOCIDADES_REDUCIDAS.md)"},
           "comprobacion_mascara_igual_a_imputacion_fuera_de_pasos_artificiales_max_abs_dif": dif_fuera,
           "energia": {}, "sonda": {}}
    for regla in ("mascara", "imputacion"):
        res["energia"][regla] = energia(X_seq[regla][iu], trans[iu], artif[iu])
        e = res["energia"][regla]["ambas"]
        print(f"[energia] {regla}: global {e['fraccion_global']:.4f} | relativa {e['fraccion_relativa']:.4f} | "
              f"r en transiciones {e['fraccion_de_r_en_pasos_de_transicion']:.4f} (pasos de transicion "
              f"{e['fraccion_de_pasos_de_transicion']:.4f}) | r en artificiales "
              f"{e['fraccion_de_r_en_pasos_artificiales_def_amplia']:.4f}", flush=True)

    res["r_rigida"] = {}
    for regla in ("mascara", "imputacion"):
        e_r, e_gs, e_def = vd.descomponer_r_rigida(X_seq[regla][iu])
        tr = trans[iu]
        res["r_rigida"][regla] = {}
        for nombre, sel in (("fuera_de_transiciones", ~tr), ("en_transiciones", tr)):
            sel = sel.copy()
            sel[:, 0] = False
            res["r_rigida"][regla][nombre] = {
                "fraccion_de_r_giro_y_escala": float(e_gs[sel].sum() / e_r[sel].sum()),
                "fraccion_de_r_deformacion": float(e_def[sel].sum() / e_r[sel].sum()),
                "energia_r": float(e_r[sel].sum())}
        q = res["r_rigida"][regla]
        print(f"[r rigida] {regla}: fuera de transiciones giro+escala {q['fuera_de_transiciones']['fraccion_de_r_giro_y_escala']:.3f}"
              f" / deformacion {q['fuera_de_transiciones']['fraccion_de_r_deformacion']:.3f} | en transiciones giro+escala "
              f"{q['en_transiciones']['fraccion_de_r_giro_y_escala']:.3f}", flush=True)
    res["r_rigida"]["definicion"] = ("r_i ≈ ω × d_i + s·d_i por minimos cuadrados (d_i = punto medio de t−1 y t menos "
                                     "el centroide); deformacion = residuo; velocidad_dedos.descomponer_r_rigida")

    t0 = time.perf_counter()
    for regla in ("mascara", "imputacion"):
        R = vd.rasgos_sonda(X_seq[regla])
        res["sonda"][regla] = {}
        for nombre in ("g", "r", "r_12", "g_r", "v"):
            s = sonda_lineal(R[nombre], y, loso, pos)
            res["sonda"][regla][nombre] = s
            print(f"[sonda] {regla} {nombre} ({s['n_rasgos']} rasgos): prueba {s['test_top1_media']:.4f} "
                  f"{[round(f['test_top1'], 3) for f in s['pliegues']]} | val {s['val_top1_media']:.4f} | C "
                  f"{[f['C'] for f in s['pliegues']]} | {time.perf_counter() - t0:.0f} s", flush=True)

    nombres_senas = tl.cargar_nombres_senas()
    vid, ranking, frac_vid = elegir_video_configuracion(X_seq["imputacion"][iu], [nombres[i] for i in iu], y[iu],
                                                        nombres_senas)
    res["video_configuracion"] = {"video": vid, "sena": nombres_senas[int(y[pos[vid]])],
                                  "fraccion_relativa_del_video_imputacion": frac_vid,
                                  "criterio": "clase con la mayor fraccion mediana de energia relativa (r) con "
                                              "imputacion; video del senante 0000 mas cercano a esa mediana",
                                  "ranking_clases": ranking}
    print(f"[figura] seña con cambio de configuracion: {vid} «{res['video_configuracion']['sena']}»; top-5 "
          f"{[(r['sena'], round(r['fraccion_relativa_mediana'], 3)) for r in ranking[:5]]}", flush=True)
    ruta = FIGURAS / f"velocidad_dedos_g_r_hola_{VIDEO_HOLA}_y_{vid}.png"
    res["figura"] = {"ruta": tl._rel(ruta), **figura([VIDEO_HOLA, vid], X_seq, trans, nombres, y, nombres_senas, ruta)}
    for v in (VIDEO_HOLA, vid):
        i = pos[v]
        e_r, e_gs, e_def = vd.descomponer_r_rigida(X_seq["imputacion"][i:i + 1])
        sel = ~trans[i:i + 1].copy()
        sel[:, 0] = False
        res["figura"]["videos"][v]["imputacion_fuera_de_transiciones_fraccion_de_r_deformacion"] = \
            float(e_def[sel].sum() / e_r[sel].sum())
        print(f"[figura] {v}: deformacion / r (imputacion, fuera de transiciones) "
              f"{res['figura']['videos'][v]['imputacion_fuera_de_transiciones_fraccion_de_r_deformacion']:.3f}")
    (FIGURAS / "velocidad_dedos_figura.json").write_text(json.dumps(res["figura"], indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    (out / "analisis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {tl._rel(ruta)} y {tl._rel(out / 'analisis.json')}")


# ---------------------------------------------------------------------------
# Criterio prerregistrado (funciones puras, probadas en tests/test_hipotesis_velocidad_dedos.py)
# ---------------------------------------------------------------------------

def margenes(val: dict, loss: dict, brazos=BRAZOS_SIGMA) -> dict:
    """σ agrupada entre semillas (raíz de la media de las varianzas de la media de 5 pliegues) sobre los brazos
    prerregistrados. Margen top-1 = máx(σ, 0,01); margen de val_loss = σ de val_loss (sin piso)."""
    s, var = ect.sigma_agrupada({b: val[b] for b in brazos})
    sl, var_l = ect.sigma_agrupada({b: loss[b] for b in brazos})
    return {"sigma_top1": s, "margen_top1": max(s, PISO_MARGEN), "sigma_val_loss": sl, "margen_val_loss": sl,
            "varianzas_top1": var, "varianzas_val_loss": var_l, "brazos": list(brazos)}


def comparar(val_a, val_b, loss_a, loss_b) -> dict:
    """A frente a B. Δ top-1 = A − B (positivo = A mejor); Δ val_loss = A − B (negativo = A mejor). Por pliegue
    con la media de las semillas, redondeado a 6 decimales."""
    d1 = ect.deltas_por_pliegue(val_a, val_b)
    dl = np.round(np.asarray(loss_a, float).mean(axis=0) - np.asarray(loss_b, float).mean(axis=0), 6)
    return {"delta_top1_por_pliegue": d1.tolist(), "delta_top1": float(d1.mean()),
            "delta_val_loss_por_pliegue": dl.tolist(), "delta_val_loss": float(dl.mean()),
            "pliegues_val_loss_menor": int(np.sum(dl < 0)), "pliegues_val_loss_mayor": int(np.sum(dl > 0))}


def no_peor(c: dict, m: dict) -> bool:
    """A no es peor que B: Δ top-1 ≥ −margen y Δ val_loss ≤ +margen."""
    return bool(c["delta_top1"] >= -m["margen_top1"] - TOL and c["delta_val_loss"] <= m["margen_val_loss"] + TOL)


def peor_por_mas_del_margen(c: dict, m: dict) -> bool:
    """A es peor que B por más del margen en alguna de las dos métricas."""
    return not no_peor(c, m)


def supera(c: dict, m: dict) -> bool:
    """A supera a B por más del margen en alguna de las dos métricas (Δ top-1 > margen o reducción de val_loss
    > margen de val_loss)."""
    return bool(c["delta_top1"] > m["margen_top1"] + TOL or -c["delta_val_loss"] > m["margen_val_loss"] + TOL)


def criterio_sonda(sonda_imp: dict) -> dict:
    """(c): con imputación, la sonda sobre r (252 rasgos) tiene top-1 LOSO medio ≥ 0,10 (5 × azar) y ≥ 0,5 × el
    de la sonda sobre g (12 rasgos)."""
    r, g = sonda_imp["r"]["test_top1_media"], sonda_imp["g"]["test_top1_media"]
    return {"r_test_top1": r, "g_test_top1": g, "r12_test_top1": sonda_imp["r_12"]["test_top1_media"],
            "azar": AZAR, "r_por_encima_de_5x_azar": bool(r >= 5 * AZAR), "r_al_menos_la_mitad_de_g": bool(r >= 0.5 * g),
            "cumple": bool(r >= 5 * AZAR and r >= 0.5 * g)}


def veredicto_h1(val, loss, m, sonda_imp) -> dict:
    h2_t1 = comparar(val["H2"], val["T1"], loss["H2"], loss["T1"])
    h2_t2 = comparar(val["H2"], val["T2"], loss["H2"], loss["T2"])
    a = no_peor(h2_t1, m)
    b = bool(h2_t2["pliegues_val_loss_menor"] >= 4)
    c = criterio_sonda(sonda_imp)
    peor_t2 = bool(h2_t2["delta_top1"] < -m["margen_top1"] - TOL or h2_t2["delta_val_loss"] > m["margen_val_loss"] + TOL)
    peor_t1 = peor_por_mas_del_margen(h2_t1, m)
    if a and b and c["cumple"]:
        v = "CONFIRMADA"
    elif peor_t2 or peor_t1:
        v = "REFUTADA"
    else:
        v = "NO CONCLUYENTE"
    return {"H2_vs_T1": h2_t1, "H2_vs_T2": h2_t2, "a_H2_no_peor_que_T1": a, "b_H2_menor_val_loss_que_T2_en_4_de_5": b,
            "c_sonda": c, "H2_peor_que_T2": peor_t2, "H2_peor_que_T1_por_mas_del_margen": peor_t1, "veredicto": v}


def control_h1(val, loss, m) -> dict:
    c = comparar(val["H1"], val["T1"], loss["H1"], loss["T1"])
    c["equivalente_a_T1"] = bool(abs(c["delta_top1"]) <= m["margen_top1"] + TOL
                                 and abs(c["delta_val_loss"]) <= m["margen_val_loss"] + TOL)
    return c


def veredicto_h2(energia_res, val, loss, m) -> dict:
    fm = energia_res["mascara"]["ambas"]["fraccion_de_r_en_pasos_de_transicion"]
    fi = energia_res["imputacion"]["ambas"]["fraccion_de_r_en_pasos_de_transicion"]
    e1 = bool(fm >= 0.5)
    e2 = bool(fi <= 0.5 * fm)
    h3_t3 = comparar(val["H3"], val["T3"], loss["H3"], loss["T3"])
    sup = supera(h3_t3, m)
    if e1 and e2 and not sup:
        v = "CONFIRMADA"
    elif (not e1) or sup:
        v = "REFUTADA"
    else:
        v = "NO CONCLUYENTE"
    return {"fraccion_r_transicion_mascara": fm, "fraccion_r_transicion_imputacion": fi,
            "e1_mascara_al_menos_50pc": e1, "e2_imputacion_a_lo_sumo_la_mitad": e2, "H3_vs_T3": h3_t3,
            "H3_supera_a_T3_por_mas_del_margen": sup, "veredicto": v}


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def leer_corridas(out, ypn):
    filas = []
    for b in BRAZOS:
        for s in SEMILLAS:
            for k in range(5):
                d = dir_corrida(out, b, s, k)
                if not (d / "resultado.json").exists():
                    continue
                r = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
                pred = json.loads((d / "predicciones.json").read_text(encoding="utf-8"))
                pr = ef.perdidas_corrida(d, ypn)
                filas.append({"brazo": b, "semilla": s, "pliegue": k, "senante_test": r["senante_test"],
                              "val_top1": r["val_top1"], "val_loss": r["val_loss"],
                              "val_loss_recalculada_npz": pr["val_loss_recalculada_npz"],
                              "val_nll_sin_suavizado": pr["val_nll_sin_suavizado"], "val_f1_macro": r["val_f1_macro"],
                              "test_top1": r["test_top1"], "test_top3": r["test_top3"],
                              "test_f1_macro": r["test_f1_macro"], "n_parametros": r["n_parametros"],
                              "epocas_ejecutadas": r["epocas_ejecutadas"], "mejor_epoca": r["mejor_epoca"],
                              "tiempo_total_s": r["tiempo_total_s"],
                              "predicciones": {x["muestra"]: (x["real"], x["predicha"], x["prob_predicha"])
                                               for x in pred}, "fuente": tl._rel(d)})
    return filas


def cmd_resumen(args):
    out = Path(args.out).resolve()
    ypn = ef._y_por_nombre()
    filas = leer_corridas(out, ypn)
    folds = list(range(5))

    def celda(b, s, k):
        r = [f for f in filas if f["brazo"] == b and f["semilla"] == s and f["pliegue"] == k]
        return r[0] if r else None

    faltan = [(b, s, k) for b in BRAZOS for s in SEMILLAS for k in folds if celda(b, s, k) is None]
    if faltan:
        raise SystemExit(f"faltan corridas: {faltan}")
    for f in filas:
        assert abs(f["val_loss"] - f["val_loss_recalculada_npz"]) < 1e-4, f["fuente"]

    METR = ["val_top1", "val_loss", "val_nll_sin_suavizado", "val_f1_macro", "test_top1", "test_top3",
            "test_f1_macro", "epocas_ejecutadas", "mejor_epoca", "tiempo_total_s"]
    por_brazo = {}
    for b in BRAZOS:
        r = {"n_parametros": celda(b, 2026, 0)["n_parametros"], **BRAZOS[b]}
        for m in METR:
            M = np.array([[celda(b, s, k)[m] for k in folds] for s in SEMILLAS], dtype=float)
            r[m] = {"por_semilla_por_pliegue": {str(s): M[i].tolist() for i, s in enumerate(SEMILLAS)},
                    "por_pliegue_media_semillas": M.mean(axis=0).tolist(),
                    "de_entre_pliegues": float(np.std(M.mean(axis=0), ddof=1)),
                    "de_entre_semillas": float(np.std(M.mean(axis=1), ddof=1)), "media_global": float(M.mean())}
        por_brazo[b] = r

    val = {b: np.array([[celda(b, s, k)["val_top1"] for k in folds] for s in SEMILLAS]) for b in BRAZOS}
    loss = {b: np.array([[celda(b, s, k)["val_loss"] for k in folds] for s in SEMILLAS]) for b in BRAZOS}
    an = json.loads((out / "analisis.json").read_text(encoding="utf-8"))
    m = margenes(val, loss)
    h1 = veredicto_h1(val, loss, m, an["sonda"]["imputacion"])
    ctrl = control_h1(val, loss, m)
    h2 = veredicto_h2(an["energia"], val, loss, m)
    otras = {f"{a}_vs_{b}": comparar(val[a], val[b], loss[a], loss[b])
             for a, b in (("H1", "T1"), ("H2", "T1"), ("H2", "T2"), ("H3", "T3"), ("H3", "T0"), ("H2", "H1"),
                          ("H2", "H3"), ("T1", "T2"), ("T0", "T3"))}

    # Prueba LOSO (descriptiva): pruebas pareadas por pliegue y McNemar.
    pares = (("H1", "T1"), ("H2", "T1"), ("H2", "T2"), ("H3", "T3"), ("H3", "T0"), ("H2", "H1"))
    pruebas = {}
    for a, b in pares:
        pruebas[f"{a}_vs_{b}"] = {}
        for mt in ("test_top1", "test_top3", "test_f1_macro"):
            dif = np.array(por_brazo[a][mt]["por_pliegue_media_semillas"]) - np.array(
                por_brazo[b][mt]["por_pliegue_media_semillas"])
            pruebas[f"{a}_vs_{b}"][mt] = ea.pruebas_pareadas(dif)

    def agregadas(b, s):
        p = {}
        for k in folds:
            p.update(celda(b, s, k)["predicciones"])
        return p

    agg = {b: {s: agregadas(b, s) for s in SEMILLAS} for b in BRAZOS}
    voto = {b: ect.voto_semillas([agg[b][s] for s in SEMILLAS]) for b in BRAZOS}
    for b in BRAZOS:
        assert len(voto[b]) == 997, (b, len(voto[b]))
    mcnemar = {f"{a}_vs_{b}": {"2026": ect._mc(agg[a][2026], agg[b][2026]), "voto_3_semillas": ea.mcnemar(voto[a], voto[b])}
               for a, b in pares}
    exact_voto = {b: float(np.mean([v[0] == v[1] for v in voto[b].values()])) for b in BRAZOS}
    p0000 = {b: {mt: {"media": por_brazo[b][mt]["por_pliegue_media_semillas"][0],
                      "por_semilla": {s: v[0] for s, v in por_brazo[b][mt]["por_semilla_por_pliegue"].items()}}
                 for mt in ("val_top1", "val_loss", "test_top1", "test_top3", "test_f1_macro")} for b in BRAZOS}

    extra = {n: json.loads((out / n).read_text(encoding="utf-8")) for n in ("verificacion.json", "humo.json")
             if (out / n).exists()}
    sin_pred = [{k: v for k, v in f.items() if k != "predicciones"} for f in filas]
    resultado = {"experimento": "hipotesis de la velocidad de los dedos (docs/EXPERIMENTO_HIPOTESIS_VELOCIDAD_DEDOS.md)",
                 "fecha_resumen": datetime.now().isoformat(timespec="seconds"), "brazos": BRAZOS,
                 "config_T": al.CONFIG_T, "entrenamiento_T": al.ENTRENAMIENTO_COMUN, "semillas": SEMILLAS,
                 "margenes": m, "hipotesis_1": h1, "control_H1": ctrl, "hipotesis_2": h2, "comparaciones": otras,
                 "pruebas_pareadas_por_pliegue": pruebas, "mcnemar": mcnemar, "exactitud_voto_3_semillas": exact_voto,
                 "persona_0000": p0000, "por_brazo": por_brazo, "analisis_sin_entrenamiento": an,
                 "verificacion": extra.get("verificacion.json"), "humo": extra.get("humo.json"), "corridas": sin_pred}
    texto = json.dumps(resultado, indent=1, ensure_ascii=False,
                       default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    (out / "resultados_hipotesis_velocidad_dedos.json").write_text(texto, encoding="utf-8")
    if args.docs:
        JSON_DOCS.write_text(texto, encoding="utf-8")
        print(f"escrito {tl._rel(JSON_DOCS)}")

    print(f"margenes: top-1 {m['margen_top1']:.4f} (sigma {m['sigma_top1']:.4f}) | val_loss {m['margen_val_loss']:.4f}")
    for b in BRAZOS:
        r = por_brazo[b]
        print(f"{b:3s} {BRAZOS[b]['descripcion']:55s} val {r['val_top1']['media_global']:.4f} "
              f"(DE sem {r['val_top1']['de_entre_semillas']:.4f}) | loss {r['val_loss']['media_global']:.4f} "
              f"(DE sem {r['val_loss']['de_entre_semillas']:.4f}) | test {r['test_top1']['media_global']:.4f} | "
              f"top3 {r['test_top3']['media_global']:.4f} | F1 {r['test_f1_macro']['media_global']:.4f} | 0000 "
              f"{r['test_top1']['por_pliegue_media_semillas'][0]:.4f} | voto {exact_voto[b]:.4f} | "
              f"loss pl {[round(x, 3) for x in r['val_loss']['por_pliegue_media_semillas']]}")
    print("\nH1:", json.dumps(h1, indent=1, ensure_ascii=False))
    print("\ncontrol H1:", json.dumps(ctrl, indent=1))
    print("\nH2:", json.dumps(h2, indent=1, ensure_ascii=False))
    for c, d in pruebas.items():
        for mt, pr in d.items():
            print(f"  {c} {mt}: {[round(x, 4) for x in pr['diferencias']]} +{pr['mejora']} -{pr['empeora']} "
                  f"={pr['empate']} | signos {pr['signos_p_bilateral']:.4f} | wilcoxon {pr['wilcoxon_p_bilateral']:.4f}")
    for c, d in mcnemar.items():
        for s, mm in d.items():
            print(f"  McNemar {c} {s}: {mm['aciertos_brazo']} vs {mm['aciertos_R0']} | solo primero "
                  f"{mm['solo_brazo_acierta']} solo segundo {mm['solo_R0_acierta']} | p {mm['p_exacto_bilateral']:.4g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verificar")
    v.add_argument("--dest", default=None, help="carpeta (carpeta_temporal) para reproducir T1 s2026 k0")
    a = sub.add_parser("analisis")
    e = sub.add_parser("entrenar")
    e.add_argument("--brazos", default="H1,H2,H3")
    e.add_argument("--semillas", default="2026,2027,2028")
    e.add_argument("--folds", default="5")
    e.add_argument("--epochs", type=int, default=200)
    e.add_argument("--aug", type=int, default=10)
    e.add_argument("--patience", type=int, default=25)
    e.add_argument("--processed", default=str(PROCESSED_CANONICO))
    h = sub.add_parser("humo")
    h.add_argument("--copiar_a", default=None)
    r = sub.add_parser("resumen")
    r.add_argument("--docs", action="store_true")
    for p in (v, a, e, h, r):
        p.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    {"verificar": cmd_verificar, "analisis": cmd_analisis, "entrenar": cmd_entrenar, "humo": cmd_humo,
     "resumen": cmd_resumen}[args.cmd](args)


if __name__ == "__main__":
    main()
