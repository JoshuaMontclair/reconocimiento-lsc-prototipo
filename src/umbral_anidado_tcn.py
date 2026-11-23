"""
Umbral de confianza con validación ANIDADA por señante, repetido con el MODELO FINAL de D-14
(TCN + imputación + 252 rasgos). Pendiente derivado de D-14 («rehacer el umbral anidado (N-47)»).
Criterio fijado antes de escribir este guion en models/umbral_anidado_tcn_2026-11-22/informe.md §1.

Qué cambia respecto de src/umbral_anidado.py (N-47): SOLO el modelo, la entrada y la receta, que son
las del brazo T1 de src/experimento_confirmacion_tcn.py:
  · entrada: experimento_confirmacion_tcn.preparar() -> X_seq["imputacion"]
    (= mano_ausente_variantes.remuestrear_todas_con_regla(·, "imputacion", 30)) y la función de entrada
    de experimento_confirmacion_tcn.canon_de_brazo(canon, "T1") (add_velocity: 126 + 126 = 252);
  · modelo y receta: experimento_arquitecturas.entrenar_corrida("T", …) (construir_tcn con CONFIG_T,
    ENTRENAMIENTO_COMUN, aumentación ×10, 200 épocas, paciencia 25, val_accuracy), semilla 2026.
Lo demás se IMPORTA de src/umbral_anidado.py sin cambios: particiones por pares (deben coincidir con las
de N-47), casillas, guardado del softmax por señante, selector de D-09 (que a su vez viene de
src/umbral_confianza.py), agregación por pliegue y global, efecto sobre la prueba LOSO, control de
transferencia y formato del candidato.

La prueba LOSO con que se mide el efecto son las 997 predicciones guardadas del brazo T1
(models/confirmacion_tcn_2026-10-03/T1_imputacion_252/semilla_2026), los mismos 5 modelos sobre cuya
validación N-65 eligió el umbral vigente 0,00. Las semillas 2027 y 2028 se reportan aparte.

Qué NO hace: no toca models/tcn_imputacion/ (ni modelo.keras ni modelo.umbral.json), ni los modelos T1,
ni el experimento de N-47, ni data/partitions/. Escribe todo en --out.

Uso:
    python src/umbral_anidado_tcn.py --construir      # particiones (sin TensorFlow) y comprobación con N-47
    python src/umbral_anidado_tcn.py --entrenar       # 10 entrenamientos (lanzar con nohup)
    python src/umbral_anidado_tcn.py --evaluar        # umbrales, efecto en la prueba LOSO y candidato
    python src/umbral_anidado_tcn.py                  # los tres pasos seguidos
    python src/umbral_anidado_tcn.py --entrenar --pares par_0000_0001 --epochs 2 --aug 1 --out /tmp/humo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import train_loso as tl  # noqa: E402  (ninguno de estos importa TensorFlow al importarse)
from src import umbral_anidado as ua  # noqa: E402
from src import umbral_confianza as uc  # noqa: E402
from src import experimento_confirmacion_tcn as ect  # noqa: E402
from src import experimento_arquitecturas as ea  # noqa: E402
from src import mano_ausente_variantes as mav  # noqa: E402

# El selector de D-09 y la maquinaria de N-47, tal cual (tests/test_umbral_anidado_tcn.py comprueba que
# son los mismos objetos).
elegir_umbral = ua.elegir_umbral
utilidad = ua.utilidad
efecto = ua.efecto
REJILLA = ua.REJILLA
construir_particion = ua.construir_particion
elegir_umbrales = ua.elegir_umbrales
efecto_prueba_loso = ua.efecto_prueba_loso
resumen = ua.resumen
mediana_rejilla = ua.mediana_rejilla
C_PRINCIPAL = ua.C_PRINCIPAL
C_SENSIBILIDAD = ua.C_SENSIBILIDAD
SEMILLA_PARTICION = ua.SEMILLA_PARTICION

BRAZO = "T1"                                   # brazo de docs/EXPERIMENTO_CONFIRMACION_TCN.md = modelo final (D-14)
REGLA = ect.BRAZOS[BRAZO]["regla"]             # "imputacion"
SEQ_LENGTH = ect.SEQ_LENGTH                    # 30
N_RASGOS = ect.BRAZOS[BRAZO]["n_rasgos"]       # 252
CONSTRUCTOR = "T"                              # experimento_arquitecturas.constructor("T") = construir_tcn(CONFIG_T)

OUT_DEFECTO = "models/umbral_anidado_tcn_2026-11-22"
PROCESSED_DEFECTO = ect.PROCESSED_CANONICO
PARTICION_N47 = "models/umbral_anidado_2026-09-22/particiones_anidadas.json"
RESULTADOS_N47 = "models/umbral_anidado_2026-09-22/resultados_anidado.json"
T1_DIR = "models/confirmacion_tcn_2026-10-03/T1_imputacion_252"
SEMILLA_LOSO_PRINCIPAL = 2026
SEMILLAS_LOSO_EXTRA = (2027, 2028)
MODELO_FINAL = "models/tcn_imputacion/modelo.keras"
INFORME_N65 = "models/tcn_imputacion/umbral_D09_2026-10-03.json"
UMBRAL_VIGENTE_JSON = "models/tcn_imputacion/modelo.umbral.json"
T_VIGENTE = 0.0
T_REFERENCIA = 0.5
EMPATE_U = 5.0
N_BOOTSTRAP = 10_000
SEMILLA_BOOTSTRAP = 2026

_rel = ua._rel
_ahora = ua._ahora


def _ruta(p) -> Path:
    p = Path(p)
    return (p if p.is_absolute() else BASE_DIR / p).resolve()


# ---------------------------------------------------------------------------
# 1. Particiones: las de N-47, regeneradas con su función y comparadas con su archivo
# ---------------------------------------------------------------------------

def comparar_con_n47(particion: dict, ruta_n47=PARTICION_N47) -> dict:
    """Las particiones regeneradas deben ser las de N-47 (mismos conjuntos por par y mismas casillas)."""
    ruta = _ruta(ruta_n47)
    if not ruta.exists():
        return {"archivo": _rel(ruta), "existe": False, "coincide": None}
    previa = json.loads(ruta.read_text(encoding="utf-8"))
    conj = ua._comparable(previa) == ua._comparable(particion)
    cas = previa["casillas"] == particion["casillas"]
    return {"archivo": _rel(ruta), "existe": True, "conjuntos_identicos": conj, "casillas_identicas": cas,
            "coincide": bool(conj and cas)}


def preparar_particion(processed_dir: Path, out_dir: Path, exigir_n47=True) -> dict:
    loso = json.loads(tl.ARCHIVO_LOSO.read_text(encoding="utf-8"))
    nombres, y, senantes = ua.cargar_nombres(processed_dir)
    particion = construir_particion(nombres, y, senantes, loso, SEMILLA_PARTICION)
    comp = comparar_con_n47(particion)
    print(f"\n[particiones] {len(particion['pares'])} pares, {len(particion['casillas'])} casillas, semilla "
          f"{particion['semilla']}; identicas a N-47 ({comp['archivo']}): {comp['coincide']}")
    if exigir_n47 and comp["coincide"] is not True:
        raise SystemExit("las particiones regeneradas NO coinciden con las de N-47: no se sigue")
    particion["identica_a_N47"] = comp
    particion["script"] = "src/umbral_anidado_tcn.py (construir_particion de src/umbral_anidado.py)"
    out_dir.mkdir(parents=True, exist_ok=True)
    ua.persistir_particion(particion, out_dir / "particiones_anidadas.json")
    return json.loads((out_dir / "particiones_anidadas.json").read_text(encoding="utf-8")), loso


# ---------------------------------------------------------------------------
# 2. Entrada y entrenamiento del brazo T1
# ---------------------------------------------------------------------------

def entrada_posiciones(X_padded, lengths) -> np.ndarray:
    """30 × 126 posiciones con la regla de T1 (la misma llamada que experimento_confirmacion_tcn.preparar)."""
    return mav.remuestrear_todas_con_regla(X_padded, lengths, REGLA, SEQ_LENGTH)


def canon_t1(canon):
    """Tupla canon del brazo T1: canon[2] = función de entrada (add_velocity, 252 rasgos)."""
    return ect.canon_de_brazo(canon, BRAZO)


def preparar_t1(processed_dir: Path):
    """X_seq de T1, y, nombres, loso, pos y canon del brazo, exactamente como experimento_confirmacion_tcn."""
    X_seq_por_regla, y, nombres, senantes, loso, pos, canon, (X_padded, lengths) = ect.preparar(processed_dir)
    X_seq = X_seq_por_regla[REGLA]
    muestra = np.arange(0, len(X_seq), 37)
    if not np.array_equal(X_seq[muestra], entrada_posiciones(X_padded[muestra], lengths[muestra])):
        raise SystemExit("la entrada de experimento_confirmacion_tcn.preparar no es la regla «imputacion»")
    del X_seq_por_regla, X_padded
    return X_seq, y, nombres, loso, pos, canon_t1(canon)


def pliegue_de_par(par: dict, indice: int) -> dict:
    """El «pliegue» que espera entrenar_corrida: train, val y, como «test», la unión de los dos
    señantes excluidos (conjunto de calibración, NO una prueba)."""
    c = par["conjuntos"]
    return {"pliegue": int(indice), "senante_test": f"{par['a']}+{par['b']}",
            "conjuntos": {"train": list(c["train"]), "val": list(c["val"]),
                          "test": list(c["umbral"][par["a"]]) + list(c["umbral"][par["b"]])}}


def control_entrada_t1(tf, X_seq, pos, entrada, loso, t1_dir: Path, semilla=SEMILLA_LOSO_PRINCIPAL, k=0) -> dict:
    """Control de punta a punta: el modelo T1 guardado (semilla, pliegue k) predice con ESTA entrada las
    mismas probabilidades de validación y de prueba que guardó en probabilidades.npz."""
    d = t1_dir / f"semilla_{semilla}" / f"pliegue_{k}"
    guardadas = np.load(d / "probabilidades.npz")
    modelo = uc._cargar_modelo(tf, d / "modelo.keras")
    salida = {"modelo": _rel(d / "modelo.keras")}
    for conj in ("val", "test"):
        nombres = list(guardadas[f"nombres_{conj}"])
        assert nombres == list(loso["pliegues"][k]["conjuntos"][conj]), conj
        idx = np.array([pos[n] for n in nombres], dtype=np.int64)
        prob = modelo.predict(entrada(X_seq[idx]), verbose=0, batch_size=256)
        salida[conj] = {"n": len(nombres), "max_dif_prob": float(np.max(np.abs(prob - guardadas[conj]))),
                        "misma_clase": bool(np.array_equal(prob.argmax(1), guardadas[conj].argmax(1)))}
    del modelo
    tf.keras.backend.clear_session()
    salida["coincide"] = all(salida[c]["misma_clase"] and salida[c]["max_dif_prob"] < 1e-4 for c in ("val", "test"))
    return salida


def anotar_par(carpeta: Path, par: dict):
    """Procedencia y resultado de entrenar_corrida, con la regla de T1 (anotar_corrida de la confirmación)
    y con el uso real: modelo interno del par."""
    from src.representacion import ruta_procedencia
    ect.anotar_corrida(carpeta, BRAZO)
    rp = ruta_procedencia(carpeta / "modelo.keras")
    d = json.loads(rp.read_text(encoding="utf-8"))
    d["script"] = "src/umbral_anidado_tcn.py (via experimento_arquitecturas.entrenar_corrida, brazo T1)"
    d["uso"] = "modelo interno del par: solo para elegir el umbral de confianza; no es un modelo de entrega"
    d["nota"] = ("modelo interno de la validacion anidada del umbral con el modelo final (D-14). NO es el "
                 "modelo final ni debe cargarse en el prototipo.")
    d["par_umbral_anidado"] = par["par"]
    rp.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    rr = carpeta / "resultado.json"
    r = json.loads(rr.read_text(encoding="utf-8"))
    r.update({"par": par["par"], "a": par["a"], "b": par["b"],
              "senantes_entrenamiento": par["senantes_entrenamiento"],
              "nota_test": "«test» es la union de los dos senantes excluidos (calibracion del umbral), NO una prueba"})
    rr.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


def _archivos_par(carpeta: Path, par: dict) -> list[Path]:
    return [carpeta / "modelo.keras", carpeta / "resultado.json", carpeta / "predicciones.json",
            carpeta / f"predicciones_{par['a']}.json", carpeta / f"predicciones_{par['b']}.json"]


def entrenar_pares(particion: dict, args, out_dir: Path, pares_pedidos=None) -> dict:
    processed_dir = _ruta(args.processed)
    print(f"\n[datos] {_rel(processed_dir)} (manos), entrada del brazo {BRAZO}: regla «{REGLA}», {N_RASGOS} rasgos")
    t0 = time.perf_counter()
    X_seq, y, nombres, loso, pos, canon = preparar_t1(processed_dir)
    tf, _, entrada, _, _ = canon
    print(f"     X_seq {X_seq.shape}; TensorFlow {tf.__version__} ({time.perf_counter() - t0:.1f} s); "
          f"GPUs visibles: {tf.config.list_physical_devices('GPU')}")
    t1_dir = _ruta(args.t1_dir)
    ctrl = control_entrada_t1(tf, X_seq, pos, entrada, loso, t1_dir)
    print(f"[control de entrada] {ctrl['modelo']}: val max|dif| {ctrl['val']['max_dif_prob']:.2e}, prueba max|dif| "
          f"{ctrl['test']['max_dif_prob']:.2e}, misma clase {ctrl['val']['misma_clase'] and ctrl['test']['misma_clase']}")
    if not ctrl["coincide"]:
        raise SystemExit("la entrada de este guion NO reproduce las probabilidades guardadas de T1: no se entrena")

    out_dir.mkdir(parents=True, exist_ok=True)
    ruta_res = out_dir / "resultados_entrenamiento.json"
    resultados = json.loads(ruta_res.read_text(encoding="utf-8")) if ruta_res.exists() else {
        "esquema": particion["esquema"], "processed_dir": _rel(processed_dir), "representacion": ect.REPRESENTACION,
        "particion_anidada": _rel(out_dir / "particiones_anidadas.json"),
        "configuracion": {
            "brazo": BRAZO, "descripcion": ect.BRAZOS[BRAZO]["descripcion"],
            "arquitectura": "TCN + GAP (arquitecturas_literatura.construir_tcn, CONFIG_T) via "
                            "experimento_arquitecturas.entrenar_corrida",
            "hiperparametros": ect.hiperparametros_t(), "regla_mano_ausente": REGLA, "n_rasgos": N_RASGOS,
            "seq_length": SEQ_LENGTH, "epochs_max": args.epochs, "aug": args.aug, "patience": args.patience,
            "seed_entrenamiento": args.seed, "monitor": "val_accuracy", "semilla_particion": particion["semilla"],
            "nota_test": "en resultado.json de cada par, «test» es la union de los dos senantes excluidos "
                         "(conjunto de calibracion del umbral), NO una prueba",
        },
        "control_entrada_t1": ctrl, "pares": [], "entorno": tl.info_entorno(),
    }
    resultados["control_entrada_t1"] = ctrl
    hechos = {p["par"] for p in resultados["pares"]}

    def guardar():
        resultados["fecha_ultima_escritura"] = _ahora()
        ruta_res.write_text(json.dumps(resultados, indent=1, ensure_ascii=False), encoding="utf-8")

    ns = argparse.Namespace(aug=args.aug, epochs=args.epochs, patience=args.patience, processed=str(processed_dir))
    indice = {p["par"]: i for i, p in enumerate(particion["pares"])}
    plan = [p for p in particion["pares"] if not pares_pedidos or p["par"] in pares_pedidos]
    print(f"\n[plan] {len(plan)} pares: {', '.join(p['par'] for p in plan)}")
    print(f"       epochs max {args.epochs} | aug x{args.aug} | paciencia {args.patience} | semilla {args.seed} "
          f"| salida {_rel(out_dir)}")
    t_todo = time.perf_counter()
    for par in plan:
        carpeta = out_dir / par["par"]
        if par["par"] in hechos and all(f.exists() for f in _archivos_par(carpeta, par)) and not args.rehacer:
            print(f"\n  {par['par']}: ya entrenado (se conserva; --rehacer para repetir)")
            continue
        n_a, n_b = par["conteos"]["umbral_" + par["a"]], par["conteos"]["umbral_" + par["b"]]
        print(f"\n{'=' * 72}\n  {par['par'].upper()} - entrena con {par['senantes_entrenamiento']}; "
              f"umbral sobre {par['a']} ({n_a}) y {par['b']} ({n_b}) | {_ahora()}\n{'=' * 72}", flush=True)
        intentos, r, error = 0, None, None
        while intentos < 2 and r is None:
            intentos += 1
            try:
                r = ea.entrenar_corrida(CONSTRUCTOR, args.seed, indice[par["par"]], pliegue_de_par(par, indice[par["par"]]),
                                        pos, X_seq, y, canon, carpeta, ns)
            except Exception as exc:  # noqa: BLE001 - se registra y se reintenta una vez
                error = f"{type(exc).__name__}: {exc}"
                print(f"  !! {par['par']} fallo en el intento {intentos}: {error}", flush=True)
                tf.keras.backend.clear_session()
        if r is None:
            resultados.setdefault("fallos", []).append({"par": par["par"], "error": error, "intentos": intentos})
            guardar()
            raise SystemExit(f"{par['par']} fallo dos veces; se detiene la corrida")
        anotar_par(carpeta, par)
        control = ua.guardar_softmax(tf, carpeta / "modelo.keras", X_seq, y, pos, entrada, par, carpeta)
        fila = {k: r.get(k) for k in ("n_parametros", "val_top1", "val_top1_checkpoint_reevaluado", "val_loss",
                                      "test_top1", "n_train_aumentado", "n_val", "n_test", "epocas_ejecutadas",
                                      "mejor_epoca", "tiempo_entrenamiento_s", "tiempo_total_s", "s_por_epoca")}
        fila.update({"par": par["par"], "a": par["a"], "b": par["b"], "senantes_entrenamiento": par["senantes_entrenamiento"],
                     "n_train_original": par["conteos"]["train"], "intentos": intentos, "error_primer_intento": error,
                     "control_softmax_vs_predicciones": control, "fecha": _ahora()})
        resultados["pares"] = [p for p in resultados["pares"] if p["par"] != par["par"]] + [fila]
        guardar()
        print(f"  {par['par']}: val {r['val_top1']:.3f} | exactitud sobre {par['a']} {control[par['a']]['exactitud']:.3f}, "
              f"sobre {par['b']} {control[par['b']]['exactitud']:.3f} | epocas {r['epocas_ejecutadas']} (mejor "
              f"{r['mejor_epoca']}) | {r['tiempo_total_s']:.0f} s | control softmax "
              f"{max(v['max_dif_confianza'] for v in control.values()):.2e}", flush=True)
    resultados["tiempo_total_corrida_s"] = round(time.perf_counter() - t_todo, 1)
    guardar()
    print(f"\n[fin entrenamiento] {len(resultados['pares'])} pares en {resultados['tiempo_total_corrida_s']:.0f} s "
          f"| {_ahora()} | {_rel(ruta_res)}", flush=True)
    return resultados


# ---------------------------------------------------------------------------
# 3. Evaluación (sin TensorFlow)
# ---------------------------------------------------------------------------

def cargar_prueba_t1(t1_dir: Path, loso: dict, semilla: int) -> dict[int, tuple]:
    """(conf, ok) de las predicciones de prueba guardadas del brazo T1 (umbral_anidado.cargar_prueba_loso),
    comprobando que son exactamente los videos de prueba de cada pliegue."""
    d = t1_dir / f"semilla_{semilla}"
    for p in loso["pliegues"]:
        filas = json.loads((d / f"pliegue_{p['pliegue']}" / "predicciones.json").read_text(encoding="utf-8"))
        if [f["muestra"] for f in filas] != list(p["conjuntos"]["test"]):
            raise SystemExit(f"{_rel(d)}/pliegue_{p['pliegue']}: las predicciones no son la prueba del pliegue")
    return ua.cargar_prueba_loso(d, loso)


def control_n65(prueba: dict[int, tuple], informe_n65=INFORME_N65) -> dict:
    """Con 0,00 y 0,50 sobre las 997 se deben reproducir las cifras guardadas en N-65."""
    d = json.loads(_ruta(informe_n65).read_text(encoding="utf-8"))
    guardado = d["efecto"]["prueba_loso_senantes_nuevos"]
    if float(d["umbral_elegido"]) != T_VIGENTE:
        raise SystemExit(f"{informe_n65}: el umbral elegido no es {T_VIGENTE}")
    salida = {}
    for t, clave in ((T_VIGENTE, "umbral_elegido"), (T_REFERENCIA, "umbral_0.5")):
        e = efecto_prueba_loso(prueba, {k: t for k in prueba})["agregado"]
        esp = {"aceptadas_correctas": guardado[clave]["aceptadas_correctas"],
               "aceptadas_incorrectas": guardado[clave]["aceptadas_incorrectas"],
               "utilidad": float(guardado[clave]["aceptadas_correctas"] - guardado[clave]["aceptadas_incorrectas"])}
        obt = {k: e[k] for k in esp}
        salida[f"{t:.2f}"] = {"obtenido": obt, "esperado": esp, "coincide": obt == esp,
                              "tasa_rechazo": [e["tasa_rechazo"], guardado[clave]["tasa_rechazo"]]}
    return salida


def bootstrap_diferencia_u(ok, acept_1, acept_0, n=N_BOOTSTRAP, semilla=SEMILLA_BOOTSTRAP, c=C_PRINCIPAL) -> dict:
    """Bootstrap pareado (mismas muestras remuestreadas para los dos procedimientos) de U(P1) − U(P0)."""
    ok, a1, a0 = (np.asarray(v, dtype=bool) for v in (ok, acept_1, acept_0))
    contrib = (ok & a1).astype(float) - c * (~ok & a1) - ((ok & a0).astype(float) - c * (~ok & a0))
    rng = np.random.default_rng(semilla)
    difs = np.empty(n)
    for i0 in range(0, n, 1000):
        m = min(1000, n - i0)
        idx = rng.integers(0, len(ok), size=(m, len(ok)))
        difs[i0:i0 + m] = contrib[idx].sum(axis=1)
    return {"diferencia_observada": float(contrib.sum()), "ic95": [float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))],
            "fraccion_menor_o_igual_a_0": float(np.mean(difs <= 0)), "n_remuestreos": int(n), "semilla": int(semilla)}


def aplicar_regla(control: dict, p1: dict, p0: dict, empate=EMPATE_U) -> dict:
    """Regla de §1.3 (la de N-47): (i) control, (ii) U(P1) >= U(P0) (manda), (iii) errores aceptados de P1 < P0."""
    i = all(v["coincide"] for v in control.values())
    ii = p1["utilidad"] >= p0["utilidad"]
    iii = p1["aceptadas_incorrectas"] < p0["aceptadas_incorrectas"]
    du = p1["utilidad"] - p0["utilidad"]
    if not i:
        rec = "detenerse: el control no se reproduce"
    elif abs(du) <= empate:
        rec = "los datos no distinguen (|ΔU| <= 5): recomendar por el argumento de metodo y por el costo en rechazo"
    elif ii and iii:
        rec = "adoptar el umbral anidado (t_global)"
    elif not ii:
        rec = "quedarse con el umbral vigente 0,00"
    else:
        rec = "U(P1) >= U(P0) pero sin menos errores aceptados: la regla no se cumple completa; quedarse con 0,00"
    return {"i_control": i, "ii_U_P1_mayor_o_igual": ii, "iii_menos_errores_aceptados": iii, "delta_U": float(du),
            "empate_dentro_de_5": bool(abs(du) <= empate), "recomendacion": rec}


def efecto_modelo_final(modelo: Path, t: float, informe_n65=INFORME_N65) -> dict:
    """Efecto de t sobre la prueba del esquema aleatorio del modelo final (predicciones guardadas) y sobre
    su validación y la validación LOSO de selección (curvas de N-65; no se vuelve a inferir)."""
    salida = {}
    pred = modelo.parent / "evaluacion_modelo_seleccionado" / "predicciones.json"
    if pred.exists():
        conf, ok = uc.desde_predicciones(pred)
        salida["prueba_modelo_final_aleatoria"] = ua.efecto_con_umbral(conf, ok, t)
    d = json.loads(_ruta(informe_n65).read_text(encoding="utf-8"))
    i = int(round(t * 100))
    for nombre in ("validacion_modelo_final", "validacion_loso_seleccion"):
        salida[f"{nombre}_desde_curvas_N65"] = {
            "umbral": float(t), "n": d["efecto"][nombre]["umbral_elegido"]["n"],
            "tasa_rechazo": d["curvas"][f"rechazo_{nombre}"][i],
            "exactitud_entre_aceptadas": d["curvas"][f"exactitud_aceptadas_{nombre}"][i], "fuente": _rel(_ruta(informe_n65))}
    return salida


def graficar(out_dir: Path, seleccion: dict, prueba: dict[int, tuple], t_global: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conf = np.concatenate([prueba[k][0] for k in sorted(prueba)])
    ok = np.concatenate([prueba[k][1] for k in sorted(prueba)])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for p in seleccion["pliegues"]:
        ax[0].plot(REJILLA, p["curva_utilidad"], lw=1, label=f"pliegue {p['pliegue']} (t={p['t']['1.0']:.2f})")
    ax[0].plot(REJILLA, np.array(seleccion["global"]["curva_utilidad"]) / 5, lw=2, c="k", label=f"global/5 (t={t_global:.2f})")
    ax[0].axvline(t_global, ls="--", c="k")
    ax[0].set_title("U(t) en la validación anidada (c = 1), TCN + imputación"); ax[0].set_xlabel("umbral t")
    ax[0].legend(fontsize=7)
    rech = [float(1 - (conf >= t).mean()) for t in REJILLA]
    exac = [float(ok[conf >= t].mean()) if (conf >= t).any() else np.nan for t in REJILLA]
    for a, serie, tit in ((ax[1], rech, "tasa de rechazo, prueba LOSO T1 (997)"),
                          (ax[2], exac, "exactitud entre aceptadas, prueba LOSO T1")):
        a.plot(REJILLA, serie, c="C3")
        a.axvline(T_VIGENTE, ls=":", c="gray", label="0,00 vigente (N-65)")
        a.axvline(T_REFERENCIA, ls="-.", c="gray", label="0,50 referencia")
        a.axvline(t_global, ls="--", c="k", label=f"anidado {t_global:.2f}")
        a.set_title(tit); a.set_xlabel("umbral t"); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "curvas.png", dpi=130)
    plt.close(fig)


def evaluar(out_dir: Path, particion: dict, loso: dict, t1_dir: Path, modelo_final: Path) -> dict:
    faltan = [_rel(f) for p in particion["pares"] for f in _archivos_par(out_dir / p["par"], p) if not f.exists()]
    if faltan:
        raise SystemExit(f"faltan {len(faltan)} archivos de pares entrenados, p. ej. {faltan[:3]}")
    vigente = json.loads(_ruta(UMBRAL_VIGENTE_JSON).read_text(encoding="utf-8"))
    # Desde N-75 el prototipo usa 0,68; el umbral de D-09 (0,00) con que se compara queda guardado aparte.
    vigente = vigente.get("umbral_d09_validacion_vistos", vigente)
    if float(vigente["umbral"]) != T_VIGENTE:
        raise SystemExit(f"{UMBRAL_VIGENTE_JSON} ya no es {T_VIGENTE}: revisar antes de comparar")
    seleccion = elegir_umbrales(out_dir, particion)
    t_global = seleccion["umbral_candidato"]
    t_k = {p["pliegue"]: p["t"]["1.0"] for p in seleccion["pliegues"]}
    t_k_med = {p["pliegue"]: p["t_mediana_casillas"]["1.0"] for p in seleccion["pliegues"]}
    t_med_med = seleccion["global"]["t_mediana_de_medianas"]["1.0"]
    t_med_tk = seleccion["global"]["t_mediana_de_pliegues"]["1.0"]
    prueba = cargar_prueba_t1(t1_dir, loso, SEMILLA_LOSO_PRINCIPAL)
    control = control_n65(prueba)
    if not all(v["coincide"] for v in control.values()):
        raise SystemExit(f"las 997 predicciones de prueba de T1 NO reproducen N-65: {control}")

    def procedimientos(pr):
        return {
            "P1_anidado_por_pliegue_t_k (estimado limpio)": efecto_prueba_loso(pr, t_k),
            f"anidado_global_{t_global:.2f} (candidato; ligeramente optimista)": efecto_prueba_loso(pr, {k: t_global for k in pr}),
            "mediana_casillas_por_pliegue (robustez)": efecto_prueba_loso(pr, t_k_med),
            f"mediana_de_medianas_{t_med_med:.2f} (robustez)": efecto_prueba_loso(pr, {k: t_med_med for k in pr}),
            f"mediana_de_t_k_{t_med_tk:.2f} (robustez)": efecto_prueba_loso(pr, {k: t_med_tk for k in pr}),
            "P0_vigente_0.00 (N-65)": efecto_prueba_loso(pr, {k: T_VIGENTE for k in pr}),
            "referencia_0.50": efecto_prueba_loso(pr, {k: T_REFERENCIA for k in pr}),
        }

    efectos = procedimientos(prueba)
    p1 = efectos["P1_anidado_por_pliegue_t_k (estimado limpio)"]["agregado"]
    p0 = efectos["P0_vigente_0.00 (N-65)"]["agregado"]
    regla = aplicar_regla(control, p1, p0)

    conf_all = np.concatenate([prueba[k][0] for k in sorted(prueba)])
    ok_all = np.concatenate([prueba[k][1] for k in sorted(prueba)])
    acept_p1 = np.concatenate([prueba[k][0] >= t_k[k] for k in sorted(prueba)])
    boot = {"P1_t_k_vs_P0": bootstrap_diferencia_u(ok_all, acept_p1, conf_all >= T_VIGENTE),
            "t_global_vs_P0": bootstrap_diferencia_u(ok_all, conf_all >= t_global, conf_all >= T_VIGENTE)}

    otras_semillas = {}
    for s in SEMILLAS_LOSO_EXTRA:
        try:
            pr = cargar_prueba_t1(t1_dir, loso, s)
        except (FileNotFoundError, SystemExit) as exc:
            otras_semillas[str(s)] = {"error": str(exc)}
            continue
        otras_semillas[str(s)] = {k: v["agregado"] for k, v in procedimientos(pr).items()}

    sensibilidad = {}
    for c in C_SENSIBILIDAD:
        tk_c = {p["pliegue"]: p["t"][str(c)] for p in seleccion["pliegues"]}
        tg_c = seleccion["global"]["t"][str(c)]
        sensibilidad[str(c)] = {
            "t_por_pliegue": {str(k): v for k, v in tk_c.items()}, "t_global": tg_c,
            "t_mediana_de_pliegues": seleccion["global"]["t_mediana_de_pliegues"][str(c)],
            "t_mediana_de_medianas": seleccion["global"]["t_mediana_de_medianas"][str(c)],
            "efecto_prueba_loso_t_k": efecto_prueba_loso(prueba, tk_c, C_PRINCIPAL)["agregado"],
            "efecto_prueba_loso_t_global": efecto_prueba_loso(prueba, {k: tg_c for k in prueba}, C_PRINCIPAL)["agregado"],
        }
    transferencia = ua.control_transferencia(out_dir, particion, prueba, loso, t_global)
    final_ef = {"t_global": efecto_modelo_final(modelo_final, t_global), "vigente_0.00": efecto_modelo_final(modelo_final, T_VIGENTE)}
    n47 = None
    if _ruta(RESULTADOS_N47).exists():
        d47 = json.loads(_ruta(RESULTADOS_N47).read_text(encoding="utf-8"))
        n47 = {"t_k": [p["t"]["1.0"] for p in d47["seleccion"]["pliegues"]], "t_global": d47["seleccion"]["global"]["t"],
               "exactitud_modelos_internos": d47["seleccion"]["global"]["exactitud"],
               "efecto_prueba_loso": {k: v["agregado"] for k, v in d47["efecto_prueba_loso"].items()}, "fuente": RESULTADOS_N47}
    ruta_ent = out_dir / "resultados_entrenamiento.json"
    entrenamiento = json.loads(ruta_ent.read_text(encoding="utf-8")) if ruta_ent.exists() else None
    informe = {
        "experimento": "umbral de confianza con validacion anidada por senante, modelo final D-14 (TCN + imputacion + 252)",
        "criterio_fijado_en": _rel(out_dir / "informe.md") + " §1",
        "seleccion": seleccion,
        "control_N65": control,
        "efecto_prueba_loso_T1_semilla_2026": efectos,
        "regla_1_3": regla,
        "bootstrap_diferencia_U": boot,
        "efecto_prueba_loso_T1_otras_semillas": otras_semillas,
        "sensibilidad_c": sensibilidad,
        "control_transferencia": transferencia,
        "efecto_modelo_final": final_ef,
        "comparacion_N47": n47,
        "umbral_vigente": vigente,
        "entrenamiento": None if entrenamiento is None else {
            "tiempo_total_corrida_s": entrenamiento.get("tiempo_total_corrida_s"),
            "pares": entrenamiento["pares"], "fallos": entrenamiento.get("fallos", []),
            "configuracion": entrenamiento.get("configuracion"), "control_entrada_t1": entrenamiento.get("control_entrada_t1"),
            "entorno": entrenamiento.get("entorno")},
        "modelos": {"loso_T1": _rel(t1_dir), "modelo_final": _rel(modelo_final), "internos": _rel(out_dir)},
        "fecha": _ahora(), "script": "src/umbral_anidado_tcn.py",
    }
    (out_dir / "resultados_anidado_tcn.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")

    destino = out_dir / "candidato_umbral.json"
    ua.escribir_candidato(destino, t_global, modelo_final, out_dir / "informe.md", seleccion,
                          {k: v["agregado"] for k, v in efectos.items()}, fecha=informe["fecha"])
    cand = json.loads(destino.read_text(encoding="utf-8"))
    cand.update({
        "decision": "pendiente: candidato de la validacion anidada con el modelo final (D-14); sustituiria al "
                    "0,00 de N-65 solo si el director lo adopta",
        "script": "src/umbral_anidado_tcn.py (funciones de src/umbral_anidado.py y src/umbral_confianza.py)",
        "sustituye_a": "0,00 de models/tcn_imputacion/modelo.umbral.json (N-65, aprobado en N-66 a) si se adopta",
        "regla_1_3": regla,
    })
    cand["metodo"]["modelos_internos"] = (f"10 TCN + imputacion + 252 (brazo T1), una por par de senantes excluidos, "
                                          f"semilla de entrenamiento {entrenamiento['configuracion']['seed_entrenamiento'] if entrenamiento else 'desconocida'}")
    destino.write_text(json.dumps(cand, indent=1, ensure_ascii=False), encoding="utf-8")
    try:
        graficar(out_dir, seleccion, prueba, t_global)
    except Exception as exc:  # noqa: BLE001 - la figura es accesoria
        print(f"  (sin figura: {type(exc).__name__}: {exc})")

    print(f"\n  {'casilla (pliegue, interno)':28s} {'par':16s} {'n':>4s} {'exact.':>7s} {'conf.med':>9s} {'t c=1':>6s} {'c=0.5':>6s} {'c=2':>6s}")
    for cs in seleccion["casillas"]:
        print(f"  ({cs['pliegue']}, {cs['senante_interno']}){'':18s} {cs['par']:16s} {cs['n']:4d} {cs['exactitud']:7.3f} "
              f"{cs['confianza_mediana']:9.3f} {cs['t']['1.0']:6.2f} {cs['t']['0.5']:6.2f} {cs['t']['2.0']:6.2f}")
    print(f"\n  {'pliegue':8s} {'n':>5s} {'exact.':>7s} {'t_k (c=1)':>10s} {'mediana 4':>10s} {'c=0.5':>6s} {'c=2':>6s}")
    for p in seleccion["pliegues"]:
        print(f"  {p['pliegue']:<8d} {p['n']:5d} {p['exactitud']:7.3f} {p['t']['1.0']:10.2f} {p['t_mediana_casillas']['1.0']:10.2f} "
              f"{p['t']['0.5']:6.2f} {p['t']['2.0']:6.2f}")
    g = seleccion["global"]
    print(f"  global   {g['n']:5d} {g['exactitud']:7.3f} {t_global:10.2f} (mediana de t_k {t_med_tk:.2f}; mediana de "
          f"medianas {t_med_med:.2f}) c=0.5 {g['t']['0.5']:.2f} c=2 {g['t']['2.0']:.2f}")
    print("\n  efecto sobre la prueba LOSO del modelo final (T1, semilla 2026, 997):")
    print(f"  {'procedimiento':55s} {'top1':>6s} {'rechazo':>8s} {'exact.acep':>10s} {'perdidos':>9s} {'ok/mal':>9s} {'U':>6s}")
    for nombre, e in efectos.items():
        a = e["agregado"]
        print(f"  {nombre:55s} {a['exactitud_sin_umbral']:6.3f} {a['tasa_rechazo']:8.3f} {a['exactitud_entre_aceptadas']:10.3f} "
              f"{a['aciertos_perdidos_por_rechazo']:9.3f} {a['aceptadas_correctas']:4d}/{a['aceptadas_incorrectas']:<4d} {a['utilidad']:6.0f}")
    print(f"\n  control N-65: {'OK' if regla['i_control'] else 'FALLA'} | regla §1.3: {regla} | bootstrap P1-P0 "
          f"{boot['P1_t_k_vs_P0']['ic95']} | candidato {t_global:.2f} -> {_rel(destino)}")
    return informe


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Umbral con validacion anidada por senante, modelo final (D-14)")
    ap.add_argument("--construir", action="store_true")
    ap.add_argument("--entrenar", action="store_true")
    ap.add_argument("--evaluar", action="store_true")
    ap.add_argument("--pares", type=str, default=None, help="'par_0000_0001,par_0000_0002' (defecto: todos)")
    ap.add_argument("--rehacer", action="store_true")
    ap.add_argument("--out", type=str, default=OUT_DEFECTO)
    ap.add_argument("--processed", type=str, default=str(PROCESSED_DEFECTO))
    ap.add_argument("--t1-dir", type=str, default=T1_DIR)
    ap.add_argument("--modelo-final", type=str, default=MODELO_FINAL)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--aug", type=int, default=10)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=2026, help="semilla de entrenamiento (no de particion)")
    args = ap.parse_args(argv)
    if not 1 <= args.aug <= 10:
        raise SystemExit("--aug debe estar entre 1 y 10")
    if not (args.construir or args.entrenar or args.evaluar):
        args.construir = args.entrenar = args.evaluar = True
    out_dir = _ruta(args.out)
    if out_dir.is_relative_to((BASE_DIR / "data").resolve()):
        raise SystemExit("--out no puede estar dentro de data/")
    for prohibida in (_ruta(T1_DIR), _ruta(MODELO_FINAL).parent, _ruta(PARTICION_N47).parent):
        if out_dir == prohibida or out_dir.is_relative_to(prohibida):
            raise SystemExit(f"--out no puede estar dentro de {_rel(prohibida)}")

    print("=" * 72)
    print("  LSC50 - Umbral con validacion ANIDADA por senante, modelo final D-14 (TCN + imputacion + 252)")
    print(f"  inicio {_ahora()} | salida {_rel(out_dir)} | processed {_rel(_ruta(args.processed))}")
    print("=" * 72, flush=True)
    particion, loso = preparar_particion(_ruta(args.processed), out_dir)
    if args.construir and not (args.entrenar or args.evaluar):
        return
    if args.entrenar:
        entrenar_pares(particion, args, out_dir, set(args.pares.split(",")) if args.pares else None)
    if args.evaluar:
        evaluar(out_dir, particion, loso, _ruta(args.t1_dir), _ruta(args.modelo_final))


if __name__ == "__main__":
    main()
