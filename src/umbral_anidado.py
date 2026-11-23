"""
Umbral de confianza con validación ANIDADA por señante (experimento propuesto en N-46,
docs/PROPUESTA_UMBRAL_ANIDADO.md; criterio fijado en models/umbral_anidado_2026-09-22/informe.md §1).

Idea: el umbral de D-09 (§2.2.6 del anteproyecto: «sobre el conjunto de validación») se eligió con
personas que el modelo ya había visto y quedó permisivo para personas nuevas (N-33). Aquí la
validación se construye con una persona NO vista: para cada PAR de señantes {a, b} se entrena un
modelo con los otros tres (selección de época con 2 muestras por seña de esos tres, D-06) y se
predice sobre todas las muestras de a y de b. La casilla (k, j) del pliegue LOSO k usa las
predicciones del modelo del par {k, j} sobre el señante j. Con 5 señantes: 10 modelos, 20 casillas.

Selector: el MISMO de D-09, importado de src/umbral_confianza.py (no se reimplementa).
Agregación principal (la de la propuesta §2.1): t_k sobre la unión de las 4 casillas del pliegue k;
t_global (candidato) sobre la unión de las 20 casillas. Como comprobación de robustez se reportan
también las medianas (de casillas por pliegue y de pliegues).

Qué NO hace: no toca data/partitions/, ni los modelos LOSO de N-30, ni el canónico, ni su
.umbral.json vigente. Escribe todo en --out (por defecto models/umbral_anidado_2026-09-22/).

Uso:
    python src/umbral_anidado.py --construir      # particiones anidadas (sin TensorFlow)
    python src/umbral_anidado.py --entrenar       # 10 entrenamientos (lanzar con nohup)
    python src/umbral_anidado.py --evaluar        # umbrales, efecto en la prueba LOSO y candidato (sin TensorFlow)
    python src/umbral_anidado.py                  # los tres pasos seguidos
    python src/umbral_anidado.py --entrenar --pares par_0000_0001 --epochs 1 --aug 1 --out /tmp/humo
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import train_loso  # noqa: E402  (no importa TensorFlow al importarse)
from src import umbral_confianza as uc  # noqa: E402  (idem)

# El selector de D-09, tal cual. tests/test_umbral_anidado.py comprueba que son los mismos objetos.
elegir_umbral = uc.elegir_umbral
utilidad = uc.utilidad
efecto = uc.efecto
REJILLA = uc.REJILLA

SEMILLA_PARTICION = train_loso.SEMILLA_PARTICION          # 2026, a priori (D-02/D-06)
VAL_POR_CLASE = train_loso.VAL_POR_CLASE                  # 2 por seña (D-06)
DUPLICADOS_EXCLUIDOS = train_loso.DUPLICADOS_EXCLUIDOS    # D-06
C_PRINCIPAL = 1.0
C_SENSIBILIDAD = (0.5, 1.0, 2.0)

OUT_DEFECTO = "models/umbral_anidado_2026-09-22"
LOSO_DIR_DEFECTO = "models/loso_2026-09-15_holistic_mascara_manos"
CANONICO_DEFECTO = "models/canonico_holistic_mascara_manos/modelo_canonico.keras"
PROCESSED_DEFECTO = "data/processed_holistic_mascara_rostro"
REPRESENTACION_DEFECTO = "manos"
ARCHIVO_LOSO = train_loso.ARCHIVO_LOSO
INFORME_D09 = "models/umbral_D09_2026-09-15.json"


def _rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(BASE_DIR)) if p.is_relative_to(BASE_DIR) else str(p)


def _ahora() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 1. Particiones anidadas por PAR de señantes (numpy puro; sin TensorFlow)
# ---------------------------------------------------------------------------

def semilla_par(semilla: int, a: int, b: int) -> int:
    """Semilla de la validación de época del par (a, b), a < b índices de señante: distinta de las
    del LOSO (semilla*10 + k) y fijada antes de correr."""
    return int(semilla * 100 + 10 * a + b)


def nombre_par(sa: str, sb: str) -> str:
    return f"par_{sa}_{sb}"


def construir_pares(nombres, y, senantes, semilla=SEMILLA_PARTICION, excluidas=DUPLICADOS_EXCLUIDOS):
    """Para cada par {a, b} (a < b) de señantes: entrenamiento y validación de época (2 por seña,
    D-06) con los otros tres; conjuntos «umbral» = TODAS las muestras no excluidas de a y de b."""
    lista = sorted(set(senantes))
    pares = []
    for a in range(len(lista)):
        for b in range(a + 1, len(lista)):
            sa, sb = lista[a], lista[b]
            rng = np.random.default_rng(semilla_par(semilla, a, b))
            idx_a = sorted(i for i, v in enumerate(senantes) if v == sa and nombres[i] not in excluidas)
            idx_b = sorted(i for i, v in enumerate(senantes) if v == sb and nombres[i] not in excluidas)
            idx_resto = [i for i, v in enumerate(senantes) if v not in (sa, sb) and nombres[i] not in excluidas]
            grupos = train_loso._agrupar_por_clase(idx_resto, y)
            val, train = train_loso.seleccionar_por_clase(grupos, VAL_POR_CLASE, rng)
            indices = {"train": train_loso._aplanar(train), "val": train_loso._aplanar(val),
                       "umbral": {sa: idx_a, sb: idx_b}}
            conjuntos = {"train": [nombres[i] for i in indices["train"]],
                         "val": [nombres[i] for i in indices["val"]],
                         "umbral": {s: [nombres[i] for i in idx] for s, idx in indices["umbral"].items()}}
            pares.append({
                "par": nombre_par(sa, sb), "a": sa, "b": sb,
                "senantes_entrenamiento": [s for s in lista if s not in (sa, sb)],
                "semilla_validacion": semilla_par(semilla, a, b),
                "conteos": {"train": len(conjuntos["train"]), "val": len(conjuntos["val"]),
                            f"umbral_{sa}": len(idx_a), f"umbral_{sb}": len(idx_b)},
                "conjuntos": conjuntos, "indices": indices,
            })
    return pares


def casillas_desde_loso(loso: dict, pares: list[dict]) -> list[dict]:
    """Casilla (k, j): pliegue externo k (señante de prueba s_k) y señante interno s_j != s_k; sus
    predicciones son las del modelo del par {s_k, s_j} sobre s_j."""
    por_par = {p["par"]: p for p in pares}
    senantes = list(loso["senantes"])
    casillas = []
    for p in loso["pliegues"]:
        sk = p["senante_test"]
        for sj in senantes:
            if sj == sk:
                continue
            sa, sb = sorted((sk, sj))
            par = nombre_par(sa, sb)
            if par not in por_par:
                raise SystemExit(f"falta el par {par} para la casilla ({p['pliegue']}, {sj})")
            casillas.append({"pliegue": int(p["pliegue"]), "senante_test": sk, "senante_interno": sj,
                             "par": par, "n": len(por_par[par]["conjuntos"]["umbral"][sj])})
    return casillas


def verificar_par(par: dict, nombres, n_clases=50, excluidas=DUPLICADOS_EXCLUIDOS):
    """Las comprobaciones de train_loso.verificar_particion (disjuntos, cobertura exacta, n_clases
    clases por conjunto) sobre train/val/(a ∪ b) más las propias del par: ni a ni b en train o val;
    cada conjunto «umbral» es de un solo señante; validación de exactamente 2 por seña."""
    c = par["conjuntos"]
    sa, sb = par["a"], par["b"]
    train_loso.verificar_particion({"train": c["train"], "val": c["val"], "test": c["umbral"][sa] + c["umbral"][sb]},
                                   nombres, senante_test=None, n_clases=n_clases, excluidas=excluidas)
    for n in c["train"] + c["val"]:
        if n.split("_")[1] in (sa, sb):
            raise SystemExit(f"{par['par']}: fuga, {n} (señante excluido) está en train/val")
    for s in (sa, sb):
        if any(n.split("_")[1] != s for n in c["umbral"][s]):
            raise SystemExit(f"{par['par']}: el conjunto umbral_{s} contiene otro señante")
    por_clase = {}
    for n in c["val"]:
        por_clase[int(n.split("_")[0])] = por_clase.get(int(n.split("_")[0]), 0) + 1
    if len(por_clase) != n_clases or any(v != VAL_POR_CLASE for v in por_clase.values()):
        raise SystemExit(f"{par['par']}: la validación no es de exactamente {VAL_POR_CLASE} por seña")


def verificar_casillas(casillas: list[dict], pares: list[dict]):
    """En la casilla (k, j) ni el señante de prueba ni el de umbral están en train/val del par."""
    por_par = {p["par"]: p for p in pares}
    for cas in casillas:
        c = por_par[cas["par"]]["conjuntos"]
        prohibidos = {cas["senante_test"], cas["senante_interno"]}
        if any(n.split("_")[1] in prohibidos for n in c["train"] + c["val"]):
            raise SystemExit(f"casilla ({cas['pliegue']}, {cas['senante_interno']}): fuga en {cas['par']}")
        if any(n.split("_")[1] != cas["senante_interno"] for n in c["umbral"][cas["senante_interno"]]):
            raise SystemExit(f"casilla ({cas['pliegue']}, {cas['senante_interno']}): umbral con otro señante")


def construir_particion(nombres, y, senantes, loso: dict, semilla=SEMILLA_PARTICION,
                        excluidas=DUPLICADOS_EXCLUIDOS, verificar=True) -> dict:
    """La partición anidada completa, verificada, lista para persistir FUERA de data/partitions."""
    n_clases = int(max(int(n.split("_")[0]) for n in nombres)) + 1
    pares = construir_pares(nombres, y, senantes, semilla, excluidas)
    casillas = casillas_desde_loso(loso, pares)
    if verificar:
        for p in pares:
            verificar_par(p, nombres, n_clases=n_clases, excluidas=excluidas)
        verificar_casillas(casillas, pares)
    return {
        "esquema": "loso_anidado_por_pares",
        "descripcion": "Validacion anidada por senante para elegir el umbral de confianza (N-46). Para "
                       "cada par {a, b} de senantes se entrena un modelo con los otros tres: de esos tres "
                       "se apartan 2 muestras por sena (100) como validacion de epoca (D-06) y el resto "
                       "entrena. Los conjuntos umbral_a y umbral_b son todas las muestras no excluidas "
                       "de a y de b (personas no vistas por ese modelo). La casilla (k, j) del pliegue "
                       "LOSO k toma las predicciones del par {s_k, s_j} sobre s_j. Los 5 modelos LOSO "
                       "externos y la particion loso_5_pliegues.json no cambian.",
        "excluidas": dict(excluidas),
        "muestras_disponibles": sum(1 for n in nombres if n not in excluidas),
        "fuente": "metadata.json -> sample_order (nombres SSSS_VVVV_RRRR); el senante es el segundo campo",
        "senantes": sorted(set(senantes)),
        "semilla": int(semilla),
        "generador": "numpy.random.default_rng(semilla*100 + 10*a + b) por par, a < b indices de senante (PCG64)",
        "criterio": "D-06 para la validacion de epoca (2 por sena de los 3 senantes de entrenamiento, al azar "
                    "dentro de cada clase); el senante interno NO se usa para parar el entrenamiento "
                    "(propuesta §2.2); los duplicados quedan fuera de todos los conjuntos.",
        "uso": "Solo para elegir el umbral de confianza (D-09, U(t) con c = 1). Nunca para seleccionar "
               "arquitectura, epocas, representacion ni modelo canonico.",
        "particion_loso_externa": _rel(ARCHIVO_LOSO),
        "script": "src/umbral_anidado.py",
        "indices_relativos_a": "posicion en metadata.json sample_order",
        "casillas": casillas,
        "pares": pares,
    }


def _comparable(particion: dict) -> dict:
    return {"senantes": particion["senantes"], "semilla": particion["semilla"],
            "excluidas": sorted(particion.get("excluidas", {})),
            "pares": [p["conjuntos"] for p in particion["pares"]]}


def persistir_particion(particion: dict, ruta: Path) -> dict:
    """Escribe la partición anidada en `ruta` (nunca en data/partitions/). Si ya existe y es la
    misma, no la reescribe; si existe y difiere, se detiene."""
    ruta = Path(ruta)
    if ruta.resolve().is_relative_to((BASE_DIR / "data" / "partitions").resolve()):
        raise ValueError("la particion anidada no se escribe en data/partitions (entregable de 1.e)")
    if ruta.exists():
        previa = json.loads(ruta.read_text(encoding="utf-8"))
        if _comparable(previa) == _comparable(particion):
            print(f"  {_rel(ruta)}: ya existe y es identica (no se reescribe)")
            return previa
        raise SystemExit(f"{_rel(ruta)} existe y NO coincide con la particion regenerada; no se sobrescribe")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    particion = dict(particion)
    particion["fecha_creacion"] = _ahora()
    ruta.write_text(json.dumps(particion, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  {_rel(ruta)}: escrita")
    return particion


def cargar_nombres(processed_dir: Path):
    """nombres, y y señantes del processed sin cargar X_padded (para --construir y --evaluar)."""
    processed_dir = Path(processed_dir)
    meta = json.loads((processed_dir / "metadata.json").read_text(encoding="utf-8"))
    nombres = list(meta["sample_order"])
    y = np.load(processed_dir / "y_labels.npy").astype(np.int64)
    if not np.array_equal(np.array([int(n.split("_")[0]) for n in nombres]), y):
        raise SystemExit("y_labels.npy no coincide con el campo de seña de sample_order")
    return nombres, y, [n.split("_")[1] for n in nombres]


# ---------------------------------------------------------------------------
# 2. Entrenamiento de los 10 pares (reutiliza train_loso.entrenar_y_evaluar: mismo modelo que N-30)
# ---------------------------------------------------------------------------

def _archivos_par(carpeta: Path, par: dict) -> list[Path]:
    return [carpeta / "modelo.keras", carpeta / "metricas.json", carpeta / "predicciones.json",
            carpeta / f"predicciones_{par['a']}.json", carpeta / f"predicciones_{par['b']}.json"]


def guardar_softmax(tf, ruta_modelo: Path, X_seq, y, pos, add_velocity, par: dict, carpeta: Path) -> dict:
    """Softmax completo del modelo del par sobre cada uno de sus dos señantes no vistos, y control
    contra el predicciones.json que escribió entrenar_y_evaluar."""
    modelo = uc._cargar_modelo(tf, ruta_modelo)
    guardadas = {r["muestra"]: r for r in json.loads((carpeta / "predicciones.json").read_text(encoding="utf-8"))}
    control = {}
    for s in (par["a"], par["b"]):
        muestras = par["conjuntos"]["umbral"][s]
        idx = np.array([pos[n] for n in muestras], dtype=np.int64)
        prob = modelo.predict(add_velocity(X_seq[idx]), verbose=0, batch_size=256)
        pred = prob.argmax(axis=1)
        filas = [{"muestra": muestras[i], "senante": s, "real": int(y[idx[i]]), "predicha": int(pred[i]),
                  "confianza": float(prob[i, pred[i]]),
                  "softmax": [round(float(v), 6) for v in prob[i]]} for i in range(len(idx))]
        (carpeta / f"predicciones_{s}.json").write_text(json.dumps(filas, indent=1), encoding="utf-8")
        dif = max(abs(f["confianza"] - guardadas[f["muestra"]]["prob_predicha"]) for f in filas)
        cls = all(f["predicha"] == guardadas[f["muestra"]]["predicha"] for f in filas)
        control[s] = {"n": len(filas), "max_dif_confianza": float(dif), "misma_clase": bool(cls),
                      "exactitud": float(np.mean(pred == y[idx]))}
    del modelo
    tf.keras.backend.clear_session()
    return control


def entrenar_pares(particion: dict, args, out_dir: Path, pares_pedidos=None) -> dict:
    """Entrena los pares (todos o --pares), uno tras otro, con train_loso.entrenar_y_evaluar."""
    processed_dir = Path(args.processed).expanduser().resolve()
    X_padded, y, lengths, nombres, senantes = train_loso.cargar_datos(processed_dir, args.representacion)
    print(f"\n[datos] {_rel(processed_dir)}, representacion {args.representacion}: {len(nombres)} muestras, "
          f"{len(set(senantes))} senantes; X_padded {X_padded.shape}")
    print("[tf] cargando TensorFlow y la arquitectura canonica (train_final_100.build_model)...")
    t0 = time.perf_counter()
    canon = train_loso.importar_pipeline_canonico(args.representacion)
    tf, _, add_velocity, resample_sequence, _ = canon
    print(f"     TensorFlow {tf.__version__} en {time.perf_counter() - t0:.1f} s; GPUs visibles: "
          f"{tf.config.list_physical_devices('GPU')} (solo CPU)")
    X_seq = train_loso.remuestrear_todas(X_padded, lengths, resample_sequence, train_loso.SEQ_LENGTH)
    del X_padded
    pos = {n: i for i, n in enumerate(nombres)}
    nombres_senas = train_loso.cargar_nombres_senas()

    out_dir.mkdir(parents=True, exist_ok=True)
    ruta_res = out_dir / "resultados_entrenamiento.json"
    resultados = json.loads(ruta_res.read_text(encoding="utf-8")) if ruta_res.exists() else {
        "esquema": particion["esquema"], "processed_dir": _rel(processed_dir), "representacion": args.representacion,
        "particion_anidada": _rel(out_dir / "particiones_anidadas.json"),
        "configuracion": {
            "arquitectura": "CNN+BiLSTM+atencion (train_final_100.build_model via train_loso.entrenar_y_evaluar)",
            "seq_length": train_loso.SEQ_LENGTH, "n_features": int(2 * X_seq.shape[2]), "velocidades": True,
            "epochs_max": args.epochs, "aug": args.aug, "batch_size": args.batch_size, "patience": args.patience,
            "dropout": args.dropout, "seed_entrenamiento": args.seed, "monitor": "val_accuracy",
            "semilla_particion": particion["semilla"],
            "nota_test": "en metricas.json de cada par, «test» es la union de los dos senantes excluidos "
                         "(conjunto de calibracion del umbral), NO una prueba",
        },
        "pares": [], "entorno": train_loso.info_entorno(),
    }
    hechos = {p["par"] for p in resultados["pares"]}

    def guardar():
        resultados["fecha_ultima_escritura"] = _ahora()
        ruta_res.write_text(json.dumps(resultados, indent=1, ensure_ascii=False), encoding="utf-8")

    plan = [p for p in particion["pares"] if not pares_pedidos or p["par"] in pares_pedidos]
    print(f"\n[plan] {len(plan)} pares: {', '.join(p['par'] for p in plan)}")
    print(f"       epochs max {args.epochs} | aug x{args.aug} | batch {args.batch_size} | paciencia {args.patience} "
          f"| dropout {args.dropout} | semilla entrenamiento {args.seed} | salida {_rel(out_dir)}")
    t_todo = time.perf_counter()
    for par in plan:
        carpeta = out_dir / par["par"]
        if par["par"] in hechos and all(f.exists() for f in _archivos_par(carpeta, par)) and not args.rehacer:
            print(f"\n  {par['par']}: ya entrenado (se conserva; --rehacer para repetir)")
            continue
        conjuntos = {"train": par["conjuntos"]["train"], "val": par["conjuntos"]["val"],
                     "test": par["conjuntos"]["umbral"][par["a"]] + par["conjuntos"]["umbral"][par["b"]]}
        n_a, n_b = par["conteos"]["umbral_" + par["a"]], par["conteos"]["umbral_" + par["b"]]
        print(f"\n{'=' * 72}\n  {par['par'].upper()} - entrena con {par['senantes_entrenamiento']}; "
              f"umbral sobre {par['a']} ({n_a}) y {par['b']} ({n_b}) | {_ahora()}\n{'=' * 72}")
        intentos, m, error = 0, None, None
        while intentos < 2 and m is None:
            intentos += 1
            try:
                m = train_loso.entrenar_y_evaluar(par["par"], conjuntos, pos, X_seq, y, args, carpeta, canon, nombres_senas)
            except Exception as exc:  # noqa: BLE001 - se registra y se reintenta una vez
                error = f"{type(exc).__name__}: {exc}"
                print(f"  !! {par['par']} fallo en el intento {intentos}: {error}")
                tf.keras.backend.clear_session()
        if m is None:
            resultados.setdefault("fallos", []).append({"par": par["par"], "error": error, "intentos": intentos})
            guardar()
            raise SystemExit(f"{par['par']} fallo dos veces; se detiene la corrida")
        control = guardar_softmax(tf, carpeta / "modelo.keras", X_seq, y, pos, add_velocity, par, carpeta)
        proc = carpeta / "modelo.procedencia.json"
        if proc.exists():
            d = json.loads(proc.read_text(encoding="utf-8"))
            d["script"] = "src/umbral_anidado.py (entrenamiento via src/train_loso.entrenar_y_evaluar)"
            d["uso"] = "modelo interno del par: solo para elegir el umbral de confianza; no es un modelo de entrega"
            proc.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
        m.update({"par": par["par"], "a": par["a"], "b": par["b"], "senantes_entrenamiento": par["senantes_entrenamiento"],
                  "intentos": intentos, "error_primer_intento": error,
                  "control_softmax_vs_predicciones": control, "fecha": _ahora()})
        resultados["pares"] = [p for p in resultados["pares"] if p["par"] != par["par"]] + [m]
        guardar()
        print(f"  {par['par']}: val {m['mejor_val_accuracy']:.3f} | exactitud sobre {par['a']} "
              f"{control[par['a']]['exactitud']:.3f}, sobre {par['b']} {control[par['b']]['exactitud']:.3f} | "
              f"epocas {m['epocas_ejecutadas']} | {m['tiempo_total_s']:.0f} s | control softmax "
              f"{max(v['max_dif_confianza'] for v in control.values()):.2e}")
    resultados["tiempo_total_corrida_s"] = round(time.perf_counter() - t_todo, 1)
    guardar()
    print(f"\n[fin entrenamiento] {len(resultados['pares'])} pares en {resultados['tiempo_total_corrida_s']:.0f} s "
          f"| {_ahora()} | {_rel(ruta_res)}")
    return resultados


# ---------------------------------------------------------------------------
# 3. Umbrales (selector de D-09), efecto sobre la prueba LOSO original y candidato
# ---------------------------------------------------------------------------

def cargar_predicciones_senante(carpeta: Path, s: str):
    """(confianza, acierto, muestras) del predicciones_<s>.json de un par."""
    filas = json.loads((carpeta / f"predicciones_{s}.json").read_text(encoding="utf-8"))
    return (np.array([f["confianza"] for f in filas], dtype=np.float64),
            np.array([f["predicha"] == f["real"] for f in filas], dtype=bool),
            [f["muestra"] for f in filas])


def mediana_rejilla(valores) -> float:
    """Mediana de umbrales, redondeada a la rejilla de 0,01."""
    return float(round(float(np.median(np.asarray(valores, dtype=np.float64))), 2))


def resumen(ok: np.ndarray, acept: np.ndarray, c: float = C_PRINCIPAL) -> dict:
    """Las mismas cantidades que umbral_confianza.efecto, pero a partir de la máscara de aceptación
    (para agregar pliegues con umbrales distintos), más U(c)."""
    ok = np.asarray(ok, dtype=bool)
    acept = np.asarray(acept, dtype=bool)
    n_ok = int(ok.sum())
    return {
        "n": int(len(ok)),
        "exactitud_sin_umbral": float(ok.mean()) if len(ok) else None,
        "tasa_rechazo": float(1 - acept.mean()) if len(ok) else None,
        "exactitud_entre_aceptadas": float(ok[acept].mean()) if acept.any() else None,
        "aciertos_perdidos_por_rechazo": float((ok & ~acept).sum() / n_ok) if n_ok else None,
        "errores_evitados_por_rechazo": float((~ok & ~acept).sum() / max(1, int((~ok).sum()))),
        "aceptadas_correctas": int((ok & acept).sum()),
        "aceptadas_incorrectas": int((~ok & acept).sum()),
        "utilidad": float((ok & acept).sum() - c * (~ok & acept).sum()),
    }


def efecto_con_umbral(conf: np.ndarray, ok: np.ndarray, t: float, c: float = C_PRINCIPAL) -> dict:
    """efecto() de D-09 más U(c) y el umbral aplicado."""
    e = efecto(conf, ok, t)
    e["utilidad"] = utilidad(conf, ok, t, c)
    e["umbral"] = float(t)
    return e


def elegir_umbrales(out_dir: Path, particion: dict, c_lista=C_SENSIBILIDAD) -> dict:
    """Umbrales por casilla, por pliegue (unión de sus 4 casillas: regla principal) y global (unión de
    las 20 casillas: candidato), con el selector de D-09; medianas como comprobación de robustez."""
    por_pliegue: dict[int, dict] = {}
    casillas = []
    datos_casilla = {}
    for cas in particion["casillas"]:
        conf, ok, muestras = cargar_predicciones_senante(out_dir / cas["par"], cas["senante_interno"])
        esperadas = next(p for p in particion["pares"] if p["par"] == cas["par"])["conjuntos"]["umbral"][cas["senante_interno"]]
        if muestras != esperadas:
            raise SystemExit(f"{cas['par']}/predicciones_{cas['senante_interno']}.json no corresponde a la particion")
        k = cas["pliegue"]
        t_c = {str(c): elegir_umbral(conf, ok, c)[0] for c in c_lista}
        fila = {**cas, "n": int(len(conf)), "exactitud": float(ok.mean()),
                "confianza_mediana": float(np.median(conf)), "t": t_c}
        casillas.append(fila)
        datos_casilla[(k, cas["senante_interno"])] = (conf, ok)
        por_pliegue.setdefault(k, {"senante_test": cas["senante_test"], "conf": [], "ok": [], "t_casillas": []})
        por_pliegue[k]["conf"].append(conf)
        por_pliegue[k]["ok"].append(ok)
        por_pliegue[k]["t_casillas"].append(t_c)
    pliegues, todos_conf, todos_ok = [], [], []
    for k in sorted(por_pliegue):
        d = por_pliegue[k]
        conf, ok = np.concatenate(d["conf"]), np.concatenate(d["ok"])
        todos_conf.append(conf)
        todos_ok.append(ok)
        t_k, curva = elegir_umbral(conf, ok, C_PRINCIPAL)
        pliegues.append({
            "pliegue": k, "senante_test": d["senante_test"], "n": int(len(conf)), "exactitud": float(ok.mean()),
            "confianza_mediana": float(np.median(conf)),
            "t": {str(c): elegir_umbral(conf, ok, c)[0] for c in c_lista},
            "t_mediana_casillas": {str(c): mediana_rejilla([tc[str(c)] for tc in d["t_casillas"]]) for c in c_lista},
            "utilidad_max": float(max(curva)), "curva_utilidad": curva,
            "efecto_en_seleccion": efecto_con_umbral(conf, ok, t_k),
        })
    conf, ok = np.concatenate(todos_conf), np.concatenate(todos_ok)
    t_global, curva_global = elegir_umbral(conf, ok, C_PRINCIPAL)
    return {
        "criterio": uc_criterio(),
        "c_principal": C_PRINCIPAL,
        "casillas": casillas,
        "pliegues": pliegues,
        "global": {
            "n": int(len(conf)), "exactitud": float(ok.mean()), "confianza_mediana": float(np.median(conf)),
            "t": {str(c): elegir_umbral(conf, ok, c)[0] for c in c_lista},
            "t_mediana_de_pliegues": {str(c): mediana_rejilla([p["t"][str(c)] for p in pliegues]) for c in c_lista},
            "t_mediana_de_medianas": {str(c): mediana_rejilla([p["t_mediana_casillas"][str(c)] for p in pliegues]) for c in c_lista},
            "utilidad_max": float(max(curva_global)), "curva_utilidad": curva_global,
            "efecto_en_seleccion": efecto_con_umbral(conf, ok, t_global),
        },
        "umbral_candidato": float(t_global),
        "dispersion_t_pliegues": {"min": float(min(p["t"]["1.0"] for p in pliegues)),
                                  "max": float(max(p["t"]["1.0"] for p in pliegues)),
                                  "valores": [p["t"]["1.0"] for p in pliegues]},
    }


def uc_criterio() -> str:
    return ("D-09, importado de src/umbral_confianza.py: U(t) = aciertos aceptados - c * errores aceptados; "
            "aceptada si confianza >= t; rejilla 0,00-0,99; empate -> menor t")


def cargar_prueba_loso(loso_dir: Path, loso: dict) -> dict[int, tuple]:
    """(conf, ok) de las 997 predicciones de prueba ya guardadas de cada pliegue externo."""
    return {int(p["pliegue"]): uc.desde_predicciones(loso_dir / f"pliegue_{p['pliegue']}" / "predicciones.json")
            for p in loso["pliegues"]}


def efecto_prueba_loso(prueba: dict[int, tuple], umbral_por_pliegue: dict[int, float], c: float = C_PRINCIPAL) -> dict:
    """Efecto de aplicar a cada pliegue su umbral (o el mismo a todos) y agregado sobre las 997."""
    por_pliegue, oks, aceptadas = {}, [], []
    for k in sorted(prueba):
        conf, ok = prueba[k]
        t = float(umbral_por_pliegue[k])
        por_pliegue[str(k)] = efecto_con_umbral(conf, ok, t, c)
        oks.append(ok)
        aceptadas.append(conf >= t)
    agregado = resumen(np.concatenate(oks), np.concatenate(aceptadas), c)
    agregado["umbrales"] = {str(k): float(umbral_por_pliegue[k]) for k in sorted(prueba)}
    return {"por_pliegue": por_pliegue, "agregado": agregado}


CONTROL_N33 = {  # models/umbral_D09_2026-09-15.json -> efecto.prueba_loso_senantes_nuevos (N-33)
    "0.2": {"aceptadas_correctas": 540, "aceptadas_incorrectas": 297, "utilidad": 243.0},
    "0.5": {"aceptadas_correctas": 410, "aceptadas_incorrectas": 129, "utilidad": 281.0},
}


def control_n33(prueba: dict[int, tuple]) -> dict:
    """Con 0,20 y 0,50 sobre las 997 se deben reproducir las cifras guardadas de N-33."""
    salida = {}
    for t_txt, esperado in CONTROL_N33.items():
        e = efecto_prueba_loso(prueba, {k: float(t_txt) for k in prueba})["agregado"]
        salida[t_txt] = {"obtenido": {k: e[k] for k in esperado}, "esperado": esperado,
                         "coincide": all(e[k] == v for k, v in esperado.items())}
    return salida


def control_transferencia(out_dir: Path, particion: dict, prueba: dict[int, tuple], loso: dict, t_ref: float) -> dict:
    """Propuesta §4: escala de confianza de los modelos internos (3 señantes) sobre su persona no vista
    frente a la del modelo externo (4 señantes) sobre la misma persona como prueba."""
    k_de = {p["senante_test"]: int(p["pliegue"]) for p in loso["pliegues"]}
    filas = {}
    for s in particion["senantes"]:
        confs, oks = [], []
        for p in particion["pares"]:
            if s in (p["a"], p["b"]):
                conf, ok, _ = cargar_predicciones_senante(out_dir / p["par"], s)
                confs.append(conf)
                oks.append(ok)
        ci, oi = np.concatenate(confs), np.concatenate(oks)
        ce, oe = prueba[k_de[s]]
        filas[s] = {
            "interno_3_senantes": {"n": int(len(ci)), "modelos": len(confs), "exactitud": float(oi.mean()),
                                   "confianza_mediana": float(np.median(ci)),
                                   "fraccion_conf_menor_0.20": float((ci < 0.2).mean()),
                                   f"fraccion_conf_menor_{t_ref:.2f}": float((ci < t_ref).mean())},
            "externo_4_senantes": {"n": int(len(ce)), "modelos": 1, "exactitud": float(oe.mean()),
                                   "confianza_mediana": float(np.median(ce)),
                                   "fraccion_conf_menor_0.20": float((ce < 0.2).mean()),
                                   f"fraccion_conf_menor_{t_ref:.2f}": float((ce < t_ref).mean())},
        }
    ci = np.concatenate([cargar_predicciones_senante(out_dir / p["par"], s)[0]
                         for p in particion["pares"] for s in (p["a"], p["b"])])
    ce = np.concatenate([prueba[k][0] for k in sorted(prueba)])
    filas["global"] = {"interno_confianza_mediana": float(np.median(ci)), "externo_confianza_mediana": float(np.median(ce)),
                       "diferencia_medianas": float(np.median(ci) - np.median(ce))}
    return filas


def efecto_canonico(canonico: Path, t: float, c: float = C_PRINCIPAL) -> dict:
    """Efecto del umbral sobre la prueba del canónico (predicciones guardadas) y sobre su validación
    (curvas ya calculadas en el informe D-09 para toda la rejilla; no se vuelve a inferir)."""
    salida = {}
    pred = canonico.parent / "evaluacion_modelo_seleccionado" / "predicciones.json"
    if pred.exists():
        conf, ok = uc.desde_predicciones(pred)
        salida["prueba_canonico"] = efecto_con_umbral(conf, ok, t, c)
    d09 = BASE_DIR / INFORME_D09
    if d09.exists():
        d = json.loads(d09.read_text(encoding="utf-8"))
        i = int(round(t * 100))
        salida["validacion_canonico_desde_curvas_D09"] = {
            "umbral": float(t), "n": d["efecto"]["validacion_canonico"]["umbral_elegido"]["n"],
            "tasa_rechazo": d["curvas"]["rechazo_validacion_canonico"][i],
            "exactitud_entre_aceptadas": d["curvas"]["exactitud_aceptadas_validacion_canonico"][i],
            "fuente": _rel(d09)}
    return salida


def escribir_candidato(destino: Path, umbral: float, canonico: Path, informe: Path, seleccion: dict,
                       efecto_loso: dict, c: float = C_PRINCIPAL, fecha: str | None = None) -> dict:
    """candidato_umbral.json con el formato del .umbral.json vigente (umbral, decision, c,
    seleccionado_con, modelo, sha256_modelo, script, informe, fecha) más el método. NUNCA escribe
    sobre un <modelo>.umbral.json."""
    destino = Path(destino)
    if destino.name.endswith(".umbral.json"):
        raise ValueError(f"{destino.name}: el candidato no se escribe sobre un .umbral.json vigente")
    g = seleccion["global"]
    datos = {
        "umbral": float(umbral),
        "decision": "pendiente: candidato de la validacion anidada (N-46); sustituiria a D-09 solo si el director lo adopta",
        "c": float(c),
        "seleccionado_con": (f"validacion anidada por senante: {len(seleccion['casillas'])} casillas de "
                             f"{len({cs['par'] for cs in seleccion['casillas']})} modelos internos (3 senantes cada uno), "
                             f"{g['n']} predicciones sobre personas no vistas; t elegido sobre la union de las casillas"),
        "modelo": canonico.name,
        "sha256_modelo": hashlib.sha256(canonico.read_bytes()).hexdigest() if canonico.exists() else None,
        "script": "src/umbral_anidado.py",
        "informe": _rel(informe),
        "fecha": fecha or _ahora(),
        "estado": "candidato_no_adoptado",
        "sustituye_a": "D-09 (0,20) si se adopta",
        "metodo": {
            "criterio": seleccion["criterio"],
            "agregacion_principal": "t_k sobre la union de las 4 casillas del pliegue k; t_global sobre la union de las 20 casillas (propuesta §2.1)",
            "agregacion_robustez": "mediana de los 4 t de casilla por pliegue y mediana de los 5 por pliegue (regla del encargo; no es el candidato)",
            "sensibilidad_c": list(map(str, C_SENSIBILIDAD)),
        },
        "umbrales_por_casilla": [{"pliegue": cs["pliegue"], "senante_test": cs["senante_test"], "senante_interno": cs["senante_interno"],
                                  "par": cs["par"], "n": cs["n"], "t": cs["t"]} for cs in seleccion["casillas"]],
        "umbrales_por_pliegue": [{"pliegue": p["pliegue"], "senante_test": p["senante_test"], "n": p["n"], "t": p["t"],
                                  "t_mediana_casillas": p["t_mediana_casillas"]} for p in seleccion["pliegues"]],
        "umbral_global": g["t"],
        "umbral_global_mediana_de_pliegues": g["t_mediana_de_pliegues"],
        "umbral_global_mediana_de_medianas": g["t_mediana_de_medianas"],
        "dispersion_t_pliegues": seleccion["dispersion_t_pliegues"],
        "efecto_prueba_loso": efecto_loso,
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")
    return datos


def graficar(out_dir: Path, seleccion: dict, prueba: dict[int, tuple], t_global: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conf = np.concatenate([prueba[k][0] for k in sorted(prueba)])
    ok = np.concatenate([prueba[k][1] for k in sorted(prueba)])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for p in seleccion["pliegues"]:
        ax[0].plot(REJILLA, p["curva_utilidad"], lw=1, label=f"pliegue {p['pliegue']} (t={p['t']['1.0']:.2f})")
        ax[0].axvline(p["t"]["1.0"], lw=0.6, ls=":", c="gray")
    ax[0].plot(REJILLA, np.array(seleccion["global"]["curva_utilidad"]) / 5, lw=2, c="k", label=f"global/5 (t={t_global:.2f})")
    ax[0].axvline(t_global, ls="--", c="k")
    ax[0].set_title("U(t) en la validación anidada (c = 1)"); ax[0].set_xlabel("umbral t"); ax[0].legend(fontsize=7)
    rech = [float(1 - (conf >= t).mean()) for t in REJILLA]
    exac = [float(ok[conf >= t].mean()) if (conf >= t).any() else np.nan for t in REJILLA]
    for a, serie, tit in ((ax[1], rech, "tasa de rechazo, prueba LOSO (997)"), (ax[2], exac, "exactitud entre aceptadas, prueba LOSO")):
        a.plot(REJILLA, serie, c="C3")
        a.axvline(0.2, ls=":", c="gray", label="0,20 vigente (D-09)")
        a.axvline(0.5, ls="-.", c="gray", label="0,50 referencia")
        a.axvline(t_global, ls="--", c="k", label=f"anidado {t_global:.2f}")
        a.set_title(tit); a.set_xlabel("umbral t"); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "curvas.png", dpi=130)
    plt.close(fig)


def evaluar(out_dir: Path, particion: dict, loso: dict, loso_dir: Path, canonico: Path) -> dict:
    faltan = [str(_rel(f)) for p in particion["pares"] for f in _archivos_par(out_dir / p["par"], p) if not f.exists()]
    if faltan:
        raise SystemExit(f"faltan {len(faltan)} archivos de pares entrenados, p. ej. {faltan[:3]}")
    seleccion = elegir_umbrales(out_dir, particion)
    t_global = seleccion["umbral_candidato"]
    t_k = {p["pliegue"]: p["t"]["1.0"] for p in seleccion["pliegues"]}
    t_k_med = {p["pliegue"]: p["t_mediana_casillas"]["1.0"] for p in seleccion["pliegues"]}
    prueba = cargar_prueba_loso(loso_dir, loso)
    control = control_n33(prueba)
    if not all(v["coincide"] for v in control.values()):
        raise SystemExit(f"las 997 predicciones de prueba NO reproducen N-33: {control}")
    efectos = {
        "anidado_por_pliegue_t_k (estimado limpio)": efecto_prueba_loso(prueba, t_k),
        f"anidado_global_{t_global:.2f} (ligeramente optimista)": efecto_prueba_loso(prueba, {k: t_global for k in prueba}),
        "mediana_casillas_por_pliegue (robustez)": efecto_prueba_loso(prueba, t_k_med),
        "vigente_0.20 (D-09)": efecto_prueba_loso(prueba, {k: 0.2 for k in prueba}),
        "referencia_0.50": efecto_prueba_loso(prueba, {k: 0.5 for k in prueba}),
    }
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
    transferencia = control_transferencia(out_dir, particion, prueba, loso, t_global)
    canon_ef = efecto_canonico(canonico, t_global)
    ruta_res_ent = out_dir / "resultados_entrenamiento.json"
    entrenamiento = json.loads(ruta_res_ent.read_text(encoding="utf-8")) if ruta_res_ent.exists() else None
    informe = {
        "experimento": "umbral de confianza con validacion anidada por senante (N-46)",
        "criterio_fijado_en": _rel(out_dir / "informe.md"),
        "seleccion": seleccion,
        "control_N33": control,
        "efecto_prueba_loso": efectos,
        "sensibilidad_c": sensibilidad,
        "control_transferencia": transferencia,
        "efecto_canonico_con_t_global": canon_ef,
        "comparacion_N46_simulacion": {"t_mediana": 0.47, "tasa_rechazo": 0.445, "exactitud_entre_aceptadas": 0.747,
                                       "aceptadas_correctas": 413, "aceptadas_incorrectas": 140, "utilidad": 273,
                                       "nota": "simulacion de docs/PROPUESTA_UMBRAL_ANIDADO.md §6 con predicciones de prueba de modelos de 4 senantes; no es el metodo"},
        "entrenamiento": None if entrenamiento is None else {
            "tiempo_total_corrida_s": entrenamiento.get("tiempo_total_corrida_s"),
            "pares": [{k: p.get(k) for k in ("par", "a", "b", "n_train_original", "n_train_aumentado", "n_val", "n_test",
                                              "top1", "mejor_val_accuracy", "epocas_ejecutadas", "mejor_epoca",
                                              "tiempo_entrenamiento_s", "tiempo_total_s", "intentos", "error_primer_intento",
                                              "control_softmax_vs_predicciones")} for p in entrenamiento["pares"]],
            "fallos": entrenamiento.get("fallos", []),
            "configuracion": entrenamiento.get("configuracion"), "entorno": entrenamiento.get("entorno"),
        },
        "modelos": {"loso_externos": _rel(loso_dir), "canonico": _rel(canonico), "internos": _rel(out_dir)},
        "fecha": _ahora(), "script": "src/umbral_anidado.py",
    }
    (out_dir / "resultados_anidado.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")
    candidato = escribir_candidato(out_dir / "candidato_umbral.json", t_global, canonico, out_dir / "informe.md",
                                   seleccion, {k: v["agregado"] for k, v in efectos.items()}, fecha=informe["fecha"])
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
    print(f"  global   {g['n']:5d} {g['exactitud']:7.3f} {t_global:10.2f} (mediana de t_k {g['t_mediana_de_pliegues']['1.0']:.2f}; "
          f"mediana de medianas {g['t_mediana_de_medianas']['1.0']:.2f}) c=0.5 {g['t']['0.5']:.2f} c=2 {g['t']['2.0']:.2f}")
    print(f"\n  efecto sobre la prueba LOSO original (997):")
    print(f"  {'procedimiento':48s} {'rechazo':>8s} {'exact.acep':>10s} {'perdidos':>9s} {'ok/mal':>9s} {'U':>6s}")
    for nombre, e in efectos.items():
        a = e["agregado"]
        print(f"  {nombre:48s} {a['tasa_rechazo']:8.3f} {a['exactitud_entre_aceptadas']:10.3f} {a['aciertos_perdidos_por_rechazo']:9.3f} "
              f"{a['aceptadas_correctas']:4d}/{a['aceptadas_incorrectas']:<4d} {a['utilidad']:6.0f}")
    print(f"\n  control N-33: {'OK' if all(v['coincide'] for v in control.values()) else 'FALLA'} | "
          f"candidato {t_global:.2f} -> {_rel(out_dir / 'candidato_umbral.json')} | {_rel(out_dir / 'resultados_anidado.json')}")
    return informe


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Umbral de confianza con validacion anidada por senante (N-46)")
    ap.add_argument("--construir", action="store_true", help="solo construye y persiste las particiones anidadas")
    ap.add_argument("--entrenar", action="store_true", help="entrena los pares (todos o --pares)")
    ap.add_argument("--evaluar", action="store_true", help="elige umbrales, evalua el efecto y escribe el candidato")
    ap.add_argument("--pares", type=str, default=None, help="lista 'par_0000_0001,par_0000_0002' (defecto: todos)")
    ap.add_argument("--rehacer", action="store_true", help="reentrena pares ya hechos")
    ap.add_argument("--out", type=str, default=OUT_DEFECTO)
    ap.add_argument("--processed", type=str, default=str(BASE_DIR / PROCESSED_DEFECTO))
    ap.add_argument("--representacion", choices=["manos", "manos_postura", "manos_postura_rostro"], default=REPRESENTACION_DEFECTO)
    ap.add_argument("--loso-dir", type=str, default=LOSO_DIR_DEFECTO)
    ap.add_argument("--canonico", type=str, default=CANONICO_DEFECTO)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--aug", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=2026, help="semilla de entrenamiento (no de particion)")
    args = ap.parse_args(argv)
    if not 1 <= args.aug <= 10:
        raise SystemExit("--aug debe estar entre 1 y 10")
    if not (args.construir or args.entrenar or args.evaluar):
        args.construir = args.entrenar = args.evaluar = True

    out_dir = (BASE_DIR / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir = out_dir.resolve()
    if out_dir.is_relative_to((BASE_DIR / "data").resolve()):
        raise SystemExit("--out no puede estar dentro de data/")
    loso_dir = (BASE_DIR / args.loso_dir) if not Path(args.loso_dir).is_absolute() else Path(args.loso_dir)
    canonico = (BASE_DIR / args.canonico) if not Path(args.canonico).is_absolute() else Path(args.canonico)
    processed_dir = Path(args.processed).expanduser().resolve()

    print("=" * 72)
    print("  LSC50 - Umbral de confianza con validacion ANIDADA por senante (N-46)")
    print(f"  inicio {_ahora()} | salida {_rel(out_dir)} | processed {_rel(processed_dir)} | representacion {args.representacion}")
    print("=" * 72)
    loso = json.loads(ARCHIVO_LOSO.read_text(encoding="utf-8"))
    nombres, y, senantes = cargar_nombres(processed_dir)
    ruta_particion = out_dir / "particiones_anidadas.json"
    particion = construir_particion(nombres, y, senantes, loso, SEMILLA_PARTICION)
    print(f"\n[particiones] {len(particion['pares'])} pares, {len(particion['casillas'])} casillas, semilla {particion['semilla']}, "
          f"{particion['muestras_disponibles']} muestras disponibles (D-06)")
    for p in particion["pares"]:
        n_a, n_b = p["conteos"]["umbral_" + p["a"]], p["conteos"]["umbral_" + p["b"]]
        print(f"  {p['par']}: entrena {p['senantes_entrenamiento']} train {p['conteos']['train']} val {p['conteos']['val']} | "
              f"umbral {p['a']}:{n_a} {p['b']}:{n_b} | semilla val {p['semilla_validacion']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    particion = persistir_particion(particion, ruta_particion)
    particion = json.loads(ruta_particion.read_text(encoding="utf-8"))   # lo que queda en disco es lo que se usa
    if args.construir and not (args.entrenar or args.evaluar):
        return

    if args.entrenar:
        pares = set(args.pares.split(",")) if args.pares else None
        entrenar_pares(particion, args, out_dir, pares)
    if args.evaluar:
        evaluar(out_dir, particion, loso, loso_dir, canonico)


if __name__ == "__main__":
    main()
