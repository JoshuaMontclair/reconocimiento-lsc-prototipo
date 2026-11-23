"""
30 frente a 60 pasos temporales con el modelo final (docs/EXPERIMENTO_PASOS_TCN.md).

El experimento de pasos de septiembre (docs/EXPERIMENTO_PASOS_TEMPORALES.md, N-58) se hizo con la red de partida
(CNN + BiLSTM + atención). Aquí se repite la comparación que importa (30 frente a 60) con el modelo final de
D-14: TCN pequeña con GAP (CONFIG_T exacta) + imputación de la mano ausente + 252 rasgos. Todo lo demás es T1
de la confirmación (receta, aumentación, particiones LOSO, semillas 2026-2028).

  P30  30 pasos   REUTILIZADO: models/confirmacion_tcn_2026-10-03/T1_imputacion_252, si la reproducción de la
                  semilla 2026, pliegue 0, con este script coincide (subcomando verificar).
  P60  60 pasos   NUEVO: mano_ausente_variantes.remuestrear_todas_con_regla(·, "imputacion", 60) -> aumentación
                  ×10 -> add_velocity -> 60 × 252 -> construir_tcn(60, 252, 50).

Nada se reimplementa: el entrenamiento es experimento_arquitecturas.entrenar_corrida (constructor «T»), que toma
la longitud de la X_seq que recibe; la entrada es la de experimento_confirmacion_tcn (canon_de_brazo «T1»); la
comparación de corridas, la σ agrupada, las Δ y el voto de semillas son los de experimento_confirmacion_tcn; las
reglas de validación y de pérdida, las de experimento_factorial_tcn; McNemar y las pruebas pareadas, las de
experimento_arquitecturas.

entrenar_corrida escribe seq_length = 30 (constante de su módulo) en la procedencia y en resultado.json; aquí se
corrige después con la longitud real (anotar_corrida), igual que la confirmación corrige la regla y la entrada.

Subcomandos:
  verificar   configuración de las 15 corridas de T1, entradas de 30 y 60 pasos, parámetros, campo receptivo y
              reproducción de T1 (semilla 2026, pliegue 0) en una carpeta aparte.
  entrenar    corridas de P60 (o de P30, solo si la reproducción falla) que falten; reanudable.
  inferencia  parámetros, disco y latencia con lote de 1 (como translator.py), sin entrenamientos activos, y costo
              de armar la entrada de una ventana con 30 y 60 pasos.
  resumen     agrega, aplica el criterio prerregistrado y escribe el JSON de datos crudos.

Uso:
  python src/experimento_pasos_tcn.py verificar
  python src/experimento_pasos_tcn.py entrenar --semillas 2026 --folds 0, --epochs 3 --out <scratch>/humo
  python src/experimento_pasos_tcn.py entrenar --semillas 2026        # una semilla por proceso
  python src/experimento_pasos_tcn.py inferencia
  python src/experimento_pasos_tcn.py resumen --docs
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

PROCESSED_CANONICO = ea.PROCESSED_CANONICO
REPRESENTACION = ea.REPRESENTACION
REGLA = "imputacion"
DEFAULT_OUT = BASE_DIR / "models" / "pasos_tcn_2026-11-23"
JSON_DOCS = BASE_DIR / "docs" / "EXPERIMENTO_PASOS_TCN.json"
T1_DIR = ect.DEFAULT_OUT / ect.BRAZOS["T1"]["carpeta"]
SEMILLAS = ect.SEMILLAS
PISO_MARGEN = ect.PISO_MARGEN
TOL = ect.TOL
N_PARAM = 156_402
N_RASGOS = 252
VIDEO_REFERENCIA = "0030_0000_0003"

BRAZOS = {
    "P30": {"pasos": 30, "carpeta": "P30_imputacion_252",
            "descripcion": "TCN + imputacion + 252, 30 pasos (T1 de la confirmacion; reutilizado si se reproduce)"},
    "P60": {"pasos": 60, "carpeta": "P60_imputacion_252",
            "descripcion": "TCN + imputacion + 252, 60 pasos (nuevo)"},
}
ORDEN_SIMPLICIDAD = ("P30", "P60")      # regla 3: 30 es la vigente y la más barata


def _lista_int(texto):
    return [int(t) for t in str(texto).split(",") if t.strip() != ""]


def dir_corrida(out, brazo, semilla, k, reutilizar_p30=True):
    if brazo == "P30" and reutilizar_p30:
        return T1_DIR / f"semilla_{semilla}" / f"pliegue_{k}"
    return Path(out) / BRAZOS[brazo]["carpeta"] / f"semilla_{semilla}" / f"pliegue_{k}"


def campo_receptivo(cfg=al.CONFIG_T) -> int:
    """Campo receptivo (en pasos) de la pila de bloques residuales de construir_tcn:
    1 + convs_por_bloque · (ventana − 1) · Σ dilataciones."""
    return 1 + cfg["convs_por_bloque"] * (cfg["ventana"] - 1) * int(sum(cfg["dilataciones"]))


def posiciones_con_ventana_completa(n_pasos: int, cfg=al.CONFIG_T) -> int:
    """Cuántas de las n_pasos posiciones de salida ven la ventana entera (padding «same», no causal)."""
    r = (campo_receptivo(cfg) - 1) // 2
    return int(sum(1 for t in range(n_pasos) if t - r <= 0 and t + r >= n_pasos - 1))


def leer_reutilizar_p30(out) -> bool:
    """La decisión de reutilizar P30 la toma verificar (condición previa del criterio, §1.3.6)."""
    v = Path(out) / "verificacion.json"
    if not v.exists():
        return True
    return bool(json.loads(v.read_text(encoding="utf-8")).get("reutilizar_P30", True))


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def preparar(processed=PROCESSED_CANONICO, pasos=(30, 60)):
    """X_seq por longitud con la regla «imputacion» (mano_ausente_variantes.remuestrear_todas_con_regla)."""
    X_padded, y, lengths, nombres, senantes, loso, pos = ea.cargar_todo(processed)
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    X_seq = {L: mav.remuestrear_todas_con_regla(X_padded, lengths, REGLA, L) for L in pasos}
    return X_seq, y, nombres, senantes, loso, pos, canon, (X_padded, lengths)


def canon_t1(canon):
    """Tupla canon de T1 (canon[2] = add_velocity, aplicada después de aumentar)."""
    return ect.canon_de_brazo(canon, "T1")


# ---------------------------------------------------------------------------
# Verificación
# ---------------------------------------------------------------------------

def comprobar_config_t1():
    """Las 15 corridas de T1 tienen CONFIG_T, la receta común, 30 pasos, imputación y 252 rasgos."""
    esperado_hp = json.loads(json.dumps(ect.hiperparametros_t()))
    filas = []
    for s in SEMILLAS:
        for k in range(5):
            d = dir_corrida(None, "P30", s, k)
            r = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
            proc = json.loads((d / "modelo.procedencia.json").read_text(encoding="utf-8"))
            assert r["brazo"] == "T1" and r["semilla"] == s and r["pliegue"] == k, (s, k)
            assert r["hiperparametros"] == esperado_hp, (s, k)
            assert r["n_parametros"] == N_PARAM, (s, k)
            assert (r["epochs_max"], r["aug"], r["patience"], r["seq_length"]) == (200, 10, 25, 30), (s, k)
            assert r["processed"] == "data/processed_holistic_mascara_rostro" and r["representacion"] == "manos"
            assert r["regla_mano_ausente"] == REGLA and r["forma_velocidades"] == "actual"
            assert r["n_rasgos_entrada"] == N_RASGOS and r["n_val"] == 100
            assert proc["seq_length"] == 30 and proc["n_rasgos_entrada"] == N_RASGOS
            filas.append({"semilla": s, "pliegue": k, "val_top1": r["val_top1"], "val_loss": r["val_loss"],
                          "epocas": r["epocas_ejecutadas"], "tiempo_total_s": r["tiempo_total_s"],
                          "s_por_epoca": r["s_por_epoca"]})
    return filas


def coincide_reproduccion(comp: dict) -> dict:
    """Condición previa (§1.3.6): byte a byte o, si no, la tolerancia de la confirmación."""
    byte = bool(comp["predicciones_json_identico_byte_a_byte"])
    tol = bool(comp["coincide"])
    return {"byte_a_byte": byte, "dentro_de_tolerancia_confirmacion": tol, "reutilizar_P30": bool(byte or tol)}


def cmd_verificar(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    informe = {"fecha": datetime.now().isoformat(timespec="seconds"), "comprobaciones": []}

    def ok(texto):
        print(f"[ok] {texto}", flush=True)
        informe["comprobaciones"].append(texto)

    informe["t1"] = comprobar_config_t1()
    ok("P30 = T1: las 15 corridas tienen CONFIG_T, lr 1e-3, lote 64, label smoothing 0,1, 200 epocas, paciencia 25, "
       "aug x10, 30 pasos, processed_holistic_mascara_rostro, manos, imputacion, 252 rasgos y 156.402 parametros")

    X_seq, y, nombres, senantes, loso, pos, canon, (X_padded, lengths) = preparar()
    tf, _, add_velocity, _, _ = canon
    Xc = ect.preparar()[0]["imputacion"]
    assert np.array_equal(X_seq[30], Xc)
    ok("X_seq de 30 pasos = la de experimento_confirmacion_tcn.preparar (la que entreno T1), bit a bit")
    i0 = nombres.index(VIDEO_REFERENCIA)
    assert np.array_equal(X_seq[60][i0], mav.regla_imputacion(X_padded[i0, :int(lengths[i0])], 60))
    assert X_seq[60].shape == (len(nombres), 60, 126)
    largos = np.asarray(lengths, dtype=int)
    n_estirados = {L: int(np.sum(largos < L)) for L in (30, 60)}
    informe["videos_estirados"] = {"menos_de_30_cuadros": n_estirados[30], "menos_de_60_cuadros": n_estirados[60],
                                   "n_videos": int(len(nombres)), "min_cuadros": int(largos.min()),
                                   "media_cuadros": float(largos.mean()), "max_cuadros": int(largos.max())}
    ok(f"X_seq de 60 pasos = remuestrear_todas_con_regla(·, imputacion, 60) {X_seq[60].shape}; videos con menos de "
       f"60 cuadros (se estiran): {n_estirados[60]} de {len(nombres)}; con menos de 30: {n_estirados[30]}")
    f = canon_t1(canon)[2]
    assert f is add_velocity
    for L in (30, 60):
        E = f(X_seq[L][:20])
        assert E.shape == (20, L, N_RASGOS) and np.array_equal(E[:, :, :126], X_seq[L][:20])
    ok("entrada: add_velocity despues de aumentar -> 30 x 252 y 60 x 252; posiciones intactas")

    n_par = {}
    for b, info in BRAZOS.items():
        tf.keras.utils.set_random_seed(0)
        m = al.construir_tcn(info["pasos"], N_RASGOS, 50)
        n_par[b] = int(m.count_params())
        assert tuple(m.input_shape[1:]) == (info["pasos"], N_RASGOS)
        tf.keras.backend.clear_session()
        assert n_par[b] == N_PARAM, (b, n_par[b])
    informe["n_parametros"] = n_par
    cr = campo_receptivo()
    informe["campo_receptivo"] = {"pasos": cr, "posiciones_con_ventana_completa": {
        str(L): posiciones_con_ventana_completa(L) for L in (30, 60)}}
    ok(f"parametros {n_par}; campo receptivo {cr} pasos: ventana completa en "
       f"{posiciones_con_ventana_completa(30)} de 30 posiciones y {posiciones_con_ventana_completa(60)} de 60")

    if not args.sin_reproduccion:
        d = out / "reproduccion_T1" / "semilla_2026" / "pliegue_0"
        if not (d / "resultado.json").exists():
            ns = argparse.Namespace(aug=10, epochs=200, patience=25, processed=str(PROCESSED_CANONICO))
            t0 = time.perf_counter()
            ea.entrenar_corrida("T", 2026, 0, loso["pliegues"][0], pos, X_seq[30], y, canon_t1(canon), d, ns)
            print(f"  reproduccion en {time.perf_counter() - t0:.0f} s", flush=True)
        ref = dir_corrida(None, "P30", 2026, 0)
        comp = ect.comparar_corridas(ref, d)
        rr = json.loads((ref / "resultado.json").read_text(encoding="utf-8"))
        rn = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
        comp["val_loss"] = [rr["val_loss"], rn["val_loss"]]
        comp["tiempo_total_s_reproduccion"] = rn["tiempo_total_s"]
        dec = coincide_reproduccion(comp)
        informe["reproduccion_T1_s2026_k0"] = comp
        informe.update(dec)
        print(json.dumps(comp, indent=1), flush=True)
        if dec["reutilizar_P30"]:
            ok(f"T1 semilla 2026 pliegue 0 reproducido con este script: byte a byte {dec['byte_a_byte']}; "
               f"historial identico {comp['historial_identico']}; epocas {comp['epocas']}; val {comp['val_top1']}; "
               f"max |dif| prob. val {comp['max_dif_probabilidades_val']:.1e}, prueba "
               f"{comp['max_dif_probabilidades_test']:.1e} -> P30 se reutiliza")
        else:
            print("[FALLA] la reproduccion de T1 no coincide: hay que entrenar tambien P30 con este script", flush=True)
    (out / "verificacion.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {tl._rel(out / 'verificacion.json')}")


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def anotar_corrida(d: Path, brazo: str):
    """Corrige la procedencia y el resultado que escribe entrenar_corrida (seq_length fijo en 30 en ese módulo)
    con la longitud, la regla y la entrada reales."""
    from src.representacion import ruta_procedencia
    L = BRAZOS[brazo]["pasos"]
    rp = ruta_procedencia(d / "modelo.keras")
    proc = json.loads(rp.read_text(encoding="utf-8"))
    proc["seq_length"] = L
    proc["mano_ausente"] = ("imputacion (mano_ausente_variantes.regla_imputacion sobre el processed «mascara», "
                            "antes de remuestrear)")
    proc["velocidades"] = "actual"
    proc["n_rasgos_entrada"] = N_RASGOS
    proc["columnas_velocidad"] = "126 (add_velocity)"
    proc["script"] = "src/experimento_pasos_tcn.py (via experimento_arquitecturas.entrenar_corrida)"
    proc.pop("brazo_experimento_arquitecturas", None)
    proc["brazo_pasos_tcn"] = brazo
    proc["nota"] = ("modelo del experimento de pasos con la TCN (docs/EXPERIMENTO_PASOS_TCN.md). NO es el modelo "
                    "final ni debe cargarse en el prototipo.")
    rp.write_text(json.dumps(proc, indent=1, ensure_ascii=False), encoding="utf-8")
    rr = d / "resultado.json"
    r = json.loads(rr.read_text(encoding="utf-8"))
    r["constructor"] = "T (arquitecturas_literatura.construir_tcn, CONFIG_T)"
    r["brazo"] = brazo
    r["seq_length"] = L
    r.update({"regla_mano_ausente": REGLA, "forma_velocidades": "actual", "n_rasgos_entrada": N_RASGOS})
    rr.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


def cmd_entrenar(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.processed = str(Path(args.processed).resolve())
    brazos = [b.strip().upper() for b in args.brazos.split(",") if b.strip()]
    for b in brazos:
        if b not in BRAZOS:
            raise SystemExit(f"--brazos: {b} desconocido ({tuple(BRAZOS)})")
        if b == "P30" and not args.reentrenar_p30:
            raise SystemExit("P30 se reutiliza de T1; solo se entrena con --reentrenar-p30 (si la reproduccion falla)")
    pasos = tuple(sorted({BRAZOS[b]["pasos"] for b in brazos}))
    X_seq, y, nombres, senantes, loso, pos, canon, _ = preparar(Path(args.processed), pasos)
    semillas = _lista_int(args.semillas)
    folds = tl.parsear_folds(args.folds, len(loso["pliegues"]))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] experimento_pasos_tcn entrenar | brazos {brazos} | "
          f"semillas {semillas} | pliegues {folds} | epochs {args.epochs} | aug x{args.aug} | paciencia "
          f"{args.patience} | salida {tl._rel(out)}", flush=True)
    for s in semillas:
        for k in folds:
            p = loso["pliegues"][k]
            for b in brazos:
                d = dir_corrida(out, b, s, k, reutilizar_p30=False)
                if (d / "resultado.json").exists():
                    print(f"  ya existe {tl._rel(d)}; se salta", flush=True)
                    continue
                print(f"\n{'=' * 72}\n  brazo {b} ({BRAZOS[b]['descripcion']}) | semilla {s} | pliegue {k} "
                      f"(senante {p['senante_test']}) | {datetime.now().isoformat(timespec='seconds')}\n{'=' * 72}",
                      flush=True)
                X = X_seq[BRAZOS[b]["pasos"]]
                r = ea.entrenar_corrida("T", s, k, p, pos, X, y, canon_t1(canon), d, args)
                assert r["n_parametros"] == N_PARAM
                anotar_corrida(d, b)
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] fin", flush=True)


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

def cmd_inferencia(args):
    out = Path(args.out).resolve()
    reutilizar = leer_reutilizar_p30(out)
    canon = tl.importar_pipeline_canonico(REPRESENTACION)
    tf, _, add_velocity, _, _ = canon
    otros = ea.otros_entrenamientos()
    if otros:
        raise SystemExit(f"hay entrenamientos activos; la latencia se mide sin ellos: {otros}")
    rng = np.random.default_rng(0)
    rutas = {b: [dir_corrida(out, b, 2026, k, reutilizar) / "modelo.keras" for k in range(args.modelos)]
             for b in BRAZOS}
    filas = []
    for i in range(args.modelos):          # orden intercalado P30, P60 del mismo pliegue
        for b, lista in rutas.items():
            ruta = lista[i]
            model = tf.keras.models.load_model(str(ruta))
            L, F = int(model.input_shape[1]), int(model.input_shape[2])
            assert L == BRAZOS[b]["pasos"] and F == N_RASGOS, (b, L, F)
            x1 = rng.standard_normal((1, L, F)).astype(np.float32)
            for _ in range(args.calentamiento):
                model.predict(x1, verbose=0)
                model(x1, training=False)
            t_pred, t_call = [], []
            for _ in range(args.repeticiones):
                t0 = time.perf_counter()
                model.predict(x1, verbose=0)
                t_pred.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                model(x1, training=False)
                t_call.append((time.perf_counter() - t0) * 1000)
            fila = {"brazo": b, "pasos": L, "modelo": tl._rel(ruta), "n_parametros": int(model.count_params()),
                    "tamano_archivo_mb": round(ruta.stat().st_size / 2 ** 20, 3),
                    "ms_predict_lote1_mediana": float(np.median(t_pred)),
                    "ms_predict_lote1_p90": float(np.percentile(t_pred, 90)),
                    "ms_llamada_directa_mediana": float(np.median(t_call)),
                    "ms_llamada_directa_p90": float(np.percentile(t_call, 90))}
            filas.append(fila)
            print(f"  {b} {tl._rel(ruta)}: {L} x {F} | {fila['n_parametros']:,} param | "
                  f"{fila['tamano_archivo_mb']:.3f} MB | predict(1) {fila['ms_predict_lote1_mediana']:.2f} ms | "
                  f"model(x) {fila['ms_llamada_directa_mediana']:.2f} ms", flush=True)
            del model
            tf.keras.backend.clear_session()

    # Costo de armar la entrada de una ventana: processed -> imputación -> L pasos -> velocidades.
    X_padded, y, lengths, nombres, _ = tl.cargar_datos(PROCESSED_CANONICO, REPRESENTACION)
    videos = [VIDEO_REFERENCIA] + [n for n in nombres[::97]][:args.videos - 1]
    prep = {}
    for b, info in BRAZOS.items():
        meds = []
        for v in videos:
            i = nombres.index(v)
            seq = X_padded[i, :int(lengths[i])]

            def construir():
                return add_velocity(np.asarray(mav.aplicar_regla(seq, REGLA, info["pasos"]), np.float32)[None])
            for _ in range(10):
                construir()
            t = []
            for _ in range(args.repeticiones):
                t0 = time.perf_counter()
                construir()
                t.append((time.perf_counter() - t0) * 1000)
            meds.append(float(np.median(t)))
        prep[b] = {"ms_mediana_de_medianas": float(np.median(meds)), "ms_max_de_medianas": float(np.max(meds)),
                   "ms_por_video": dict(zip(videos, meds)), "videos": videos}
        print(f"  entrada {b}: {prep[b]['ms_mediana_de_medianas']:.3f} ms (mediana de {len(videos)} videos)",
              flush=True)
    costo = {}
    for b in BRAZOS:
        fb = [f for f in filas if f["brazo"] == b]
        costo[b] = {"ms_red_predict_lote1": float(np.mean([f["ms_predict_lote1_mediana"] for f in fb])),
                    "ms_red_llamada_directa": float(np.mean([f["ms_llamada_directa_mediana"] for f in fb])),
                    "ms_entrada": prep[b]["ms_mediana_de_medianas"],
                    "mb_archivo": float(np.mean([f["tamano_archivo_mb"] for f in fb])),
                    "n_parametros": fb[0]["n_parametros"]}
        costo[b]["ms_total"] = costo[b]["ms_red_predict_lote1"] + costo[b]["ms_entrada"]
    res = {"fecha": datetime.now().isoformat(timespec="seconds"),
           "metodo": f"model.predict(verbose=0) con lote de 1, como translator.py, y llamada directa model(x, "
                     f"training=False); {args.calentamiento} llamadas de calentamiento y {args.repeticiones} medidas "
                     f"por modelo; mediana y p90. Solo la red (sin MediaPipe). CPU. Modelos de la semilla 2026 "
                     f"(pliegues 0-{args.modelos - 1}), en orden intercalado P30, P60. Entrada: processed -> "
                     f"imputacion -> L pasos -> add_velocity, mediana de {args.repeticiones} repeticiones por video, "
                     f"{len(videos)} videos.",
           "procesos_de_entrenamiento_activos": otros, "entorno": tl.info_entorno(), "modelos": filas,
           "construccion_de_la_entrada": prep, "costo_por_brazo": costo}
    (out / "inferencia.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(costo, indent=1))
    print(f"escrito {tl._rel(out / 'inferencia.json')}")


# ---------------------------------------------------------------------------
# Criterio (funciones puras, probadas en tests/test_experimento_pasos_tcn.py)
# ---------------------------------------------------------------------------

def aplicar_criterio(val: dict, loss: dict) -> dict:
    """val, loss: {"P30", "P60": array (n_semillas, 5)}. Criterio de docs/EXPERIMENTO_PASOS_TCN.md §1.3:
    regla 1 (top-1 de validación con margen, en los dos sentidos) -> regla 2 (val_loss) -> regla 3 (30)."""
    sigma, var = ect.sigma_agrupada(val)
    margen = max(sigma, PISO_MARGEN)
    sigma_loss, var_loss = ect.sigma_agrupada(loss)
    res = {"sigma_val_agrupada": sigma, "varianza_val_entre_semillas": var, "margen": margen,
           "sigma_val_loss_agrupada": sigma_loss, "varianza_val_loss_entre_semillas": var_loss,
           "media_val": {b: float(np.mean(val[b])) for b in BRAZOS},
           "media_val_loss": {b: float(np.mean(loss[b])) for b in BRAZOS}}
    r1 = {"P60_supera_a_P30": ef.supera_en_validacion(val["P60"], val["P30"], margen),
          "P30_supera_a_P60": ef.supera_en_validacion(val["P30"], val["P60"], margen)}
    res["regla1"] = r1
    if r1["P60_supera_a_P30"]["cumple"]:
        res.update({"decide_en_regla": 1, "configuracion_final": "P60"})
        return res
    if r1["P30_supera_a_P60"]["cumple"]:
        res.update({"decide_en_regla": 1, "configuracion_final": "P30"})
        return res
    r2 = {"P60_supera_a_P30": ef.supera_en_perdida(loss["P60"], loss["P30"], sigma_loss),
          "P30_supera_a_P60": ef.supera_en_perdida(loss["P30"], loss["P60"], sigma_loss)}
    res["regla2"] = r2
    if r2["P60_supera_a_P30"]["cumple"]:
        res.update({"decide_en_regla": 2, "configuracion_final": "P60"})
        return res
    if r2["P30_supera_a_P60"]["cumple"]:
        res.update({"decide_en_regla": 2, "configuracion_final": "P30"})
        return res
    res["regla3"] = {"orden_simplicidad": list(ORDEN_SIMPLICIDAD),
                     "motivo": "empate en validacion: 30 es la longitud vigente y la mas barata"}
    res.update({"decide_en_regla": 3, "configuracion_final": ORDEN_SIMPLICIDAD[0]})
    return res


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def leer_corridas(out, y_por_nombre, reutilizar_p30=True):
    filas = []
    for b in BRAZOS:
        for s in SEMILLAS:
            for k in range(5):
                d = dir_corrida(out, b, s, k, reutilizar_p30)
                if not (d / "resultado.json").exists():
                    continue
                r = json.loads((d / "resultado.json").read_text(encoding="utf-8"))
                pred = json.loads((d / "predicciones.json").read_text(encoding="utf-8"))
                pr = ef.perdidas_corrida(d, y_por_nombre)
                assert r["seq_length"] == BRAZOS[b]["pasos"], (b, s, k, r["seq_length"])
                filas.append({"brazo": b, "pasos": BRAZOS[b]["pasos"], "semilla": s, "pliegue": k,
                              "senante_test": r["senante_test"], "val_top1": r["val_top1"],
                              "val_top1_checkpoint_reevaluado": r["val_top1_checkpoint_reevaluado"],
                              "val_loss": r["val_loss"], "val_loss_recalculada_npz": pr["val_loss_recalculada_npz"],
                              "val_nll_sin_suavizado": pr["val_nll_sin_suavizado"],
                              "test_top1": r["test_top1"], "test_top3": r["test_top3"],
                              "test_f1_macro": r["test_f1_macro"], "n_parametros": r["n_parametros"],
                              "epocas_ejecutadas": r["epocas_ejecutadas"], "mejor_epoca": r["mejor_epoca"],
                              "tiempo_total_s": r["tiempo_total_s"], "s_por_epoca": r["s_por_epoca"],
                              "fecha_fin": r.get("fecha_fin"),
                              "predicciones": {x["muestra"]: (x["real"], x["predicha"], x["prob_predicha"])
                                               for x in pred}, "fuente": tl._rel(d)})
    return filas


def cmd_resumen(args):
    out = Path(args.out).resolve()
    reutilizar = leer_reutilizar_p30(out)
    ypn = ef._y_por_nombre()
    filas = leer_corridas(out, ypn, reutilizar)
    folds = list(range(5))

    def celda(b, s, k):
        r = [f for f in filas if f["brazo"] == b and f["semilla"] == s and f["pliegue"] == k]
        return r[0] if r else None

    completas = [s for s in SEMILLAS if all(celda(b, s, k) for b in BRAZOS for k in folds)]
    if len(completas) < 2:
        raise SystemExit(f"faltan corridas: semillas completas {completas}")
    semillas = completas
    for f in filas:
        assert abs(f["val_top1"] - f["val_top1_checkpoint_reevaluado"]) < 1e-6, f["fuente"]
        assert abs(f["val_loss"] - f["val_loss_recalculada_npz"]) < 1e-4, f["fuente"]
    METR = ["val_top1", "val_loss", "val_nll_sin_suavizado", "test_top1", "test_top3", "test_f1_macro",
            "epocas_ejecutadas", "mejor_epoca", "tiempo_total_s", "s_por_epoca"]
    por_brazo = {}
    for b in BRAZOS:
        r = {"n_parametros": celda(b, semillas[0], 0)["n_parametros"], "pasos": BRAZOS[b]["pasos"]}
        for m in METR:
            M = np.array([[celda(b, s, k)[m] for k in folds] for s in semillas], dtype=float)
            r[m] = {"por_semilla_por_pliegue": {str(s): M[i].tolist() for i, s in enumerate(semillas)},
                    "por_pliegue_media_semillas": M.mean(axis=0).tolist(),
                    "por_semilla_media_pliegues": {str(s): float(M[i].mean()) for i, s in enumerate(semillas)},
                    "de_entre_pliegues": float(np.std(M.mean(axis=0), ddof=1)),
                    "de_entre_semillas": float(np.std(M.mean(axis=1), ddof=1)),
                    "media_global": float(M.mean())}
        por_brazo[b] = r

    val = {b: np.array([[celda(b, s, k)["val_top1"] for k in folds] for s in semillas]) for b in BRAZOS}
    loss = {b: np.array([[celda(b, s, k)["val_loss"] for k in folds] for s in semillas]) for b in BRAZOS}
    criterio = aplicar_criterio(val, loss)

    pruebas = {}
    for m in ("val_top1", "val_loss", "test_top1", "test_top3", "test_f1_macro"):
        dif = (np.array(por_brazo["P60"][m]["por_pliegue_media_semillas"])
               - np.array(por_brazo["P30"][m]["por_pliegue_media_semillas"]))
        if m == "val_loss":
            dif = -dif          # «a favor» = menor pérdida con 60
        pruebas[m] = ea.pruebas_pareadas(dif)

    def agregadas(b, s):
        p = {}
        for k in folds:
            p.update(celda(b, s, k)["predicciones"])
        return p

    agg = {b: {s: agregadas(b, s) for s in semillas} for b in BRAZOS}
    voto = {b: ect.voto_semillas([agg[b][s] for s in semillas]) for b in BRAZOS}
    for b in BRAZOS:
        assert len(voto[b]) == 997, (b, len(voto[b]))
    mcnemar = {str(s): ect._mc(agg["P60"][s], agg["P30"][s]) for s in semillas}
    mcnemar["voto_semillas"] = ea.mcnemar(voto["P60"], voto["P30"])
    exact_voto = {b: float(np.mean([v[0] == v[1] for v in voto[b].values()])) for b in BRAZOS}

    p0000 = {b: {m: {"media": por_brazo[b][m]["por_pliegue_media_semillas"][0],
                     "por_semilla": {s: v[0] for s, v in por_brazo[b][m]["por_semilla_por_pliegue"].items()}}
                 for m in ("val_top1", "val_loss", "test_top1", "test_top3", "test_f1_macro")} for b in BRAZOS}

    extra = {}
    for nombre in ("verificacion.json", "inferencia.json", "humo.json"):
        if (out / nombre).exists():
            extra[nombre] = json.loads((out / nombre).read_text(encoding="utf-8"))
    crit_txt = out / "criterio_prerregistrado.sha256"
    sin_pred = [{k: v for k, v in f.items() if k != "predicciones"} for f in filas]
    resultado = {"experimento": "30 frente a 60 pasos con el modelo final TCN + imputacion + 252 "
                                "(docs/EXPERIMENTO_PASOS_TCN.md)",
                 "fecha_resumen": datetime.now().isoformat(timespec="seconds"),
                 "brazos": BRAZOS, "P30_reutilizado_de_T1": reutilizar, "T1_dir": tl._rel(T1_DIR),
                 "config_T": al.CONFIG_T, "entrenamiento_T": al.ENTRENAMIENTO_COMUN,
                 "campo_receptivo_pasos": campo_receptivo(),
                 "posiciones_con_ventana_completa": {str(L): posiciones_con_ventana_completa(L) for L in (30, 60)},
                 "semillas": semillas,
                 "criterio_prerregistrado_sha256": crit_txt.read_text(encoding="utf-8").split()[0]
                 if crit_txt.exists() else None,
                 "criterio": criterio, "pruebas_pareadas_por_pliegue_P60_menos_P30": pruebas, "mcnemar_P60_vs_P30": mcnemar,
                 "exactitud_voto_semillas": exact_voto, "persona_0000": p0000, "por_brazo": por_brazo,
                 "verificacion": extra.get("verificacion.json"), "humo": extra.get("humo.json"),
                 "inferencia": extra.get("inferencia.json"), "corridas": sin_pred}
    (out / "resultados_pasos_tcn.json").write_text(json.dumps(resultado, indent=1, ensure_ascii=False),
                                                   encoding="utf-8")
    if args.docs:
        JSON_DOCS.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"escrito {tl._rel(JSON_DOCS)}")

    print(f"semillas {semillas} | P30 reutilizado {reutilizar}")
    for b in BRAZOS:
        r = por_brazo[b]
        print(f"\n{b} | {r['n_parametros']:,} param | voto {exact_voto[b]:.4f}")
        for m in METR:
            print(f"   {m:22s} {r[m]['media_global']:.4f} (DE pl {r[m]['de_entre_pliegues']:.4f}, DE sem "
                  f"{r[m]['de_entre_semillas']:.4f}) {[round(x, 3) for x in r[m]['por_pliegue_media_semillas']]} "
                  f"sem {[round(x, 4) for x in r[m]['por_semilla_media_pliegues'].values()]}")
    print("\nCRITERIO:", json.dumps(criterio, indent=1, ensure_ascii=False))
    for m, pr in pruebas.items():
        print(f"  P60-P30 {m}: {[round(x, 4) for x in pr['diferencias']]} +{pr['mejora']} -{pr['empeora']} "
              f"={pr['empate']} | signos {pr['signos_p_bilateral']:.4f} | wilcoxon {pr['wilcoxon_p_bilateral']:.4f}")
    for s, m in mcnemar.items():
        print(f"  McNemar P60 vs P30 {s}: {m['aciertos_brazo']} vs {m['aciertos_R0']} | solo P60 "
              f"{m['solo_brazo_acierta']} solo P30 {m['solo_R0_acierta']} | p {m['p_exacto_bilateral']:.4g}")
    print("persona 0000:", {b: {m: round(v["media"], 3) for m, v in p0000[b].items()} for b in BRAZOS})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verificar")
    v.add_argument("--sin-reproduccion", action="store_true", help="omite la reproduccion de T1 s2026 k0")
    e = sub.add_parser("entrenar")
    e.add_argument("--brazos", default="P60")
    e.add_argument("--semillas", default="2026,2027,2028")
    e.add_argument("--folds", default="5")
    e.add_argument("--epochs", type=int, default=200)
    e.add_argument("--aug", type=int, default=10)
    e.add_argument("--patience", type=int, default=25)
    e.add_argument("--processed", default=str(PROCESSED_CANONICO))
    e.add_argument("--reentrenar-p30", action="store_true",
                   help="solo si la reproduccion de T1 falla: permite entrenar P30 en --out")
    i = sub.add_parser("inferencia")
    i.add_argument("--calentamiento", type=int, default=10)
    i.add_argument("--repeticiones", type=int, default=100)
    i.add_argument("--modelos", type=int, default=3)
    i.add_argument("--videos", type=int, default=5)
    r = sub.add_parser("resumen")
    r.add_argument("--docs", action="store_true")
    for p in (v, e, i, r):
        p.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    {"verificar": cmd_verificar, "entrenar": cmd_entrenar, "inferencia": cmd_inferencia,
     "resumen": cmd_resumen}[args.cmd](args)


if __name__ == "__main__":
    main()
