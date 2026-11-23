"""
Partición independiente del señante (LOSO) e índices de partición para LSC50.

Cubre dos compromisos del anteproyecto:
  - Actividad 1.e: los dos esquemas de partición como índices explícitos y reproducibles,
    persistidos en data/partitions/ con nombres de muestra por conjunto, semilla y criterio.
  - §2.2.4: evaluación LOSO (leave-one-signer-out) de 5 pliegues, un señante fuera por
    pliegue, con media y dispersión entre pliegues de cada métrica.
  - D-06 (14-sep-2026): se excluyen los tres archivos duplicados del corpus (N-15, N-16) y la
    validación se reparte por CONTEO FIJO, 2 muestras por seña (100), en ambos esquemas.

Reglas que aplica este script (y que no aplicaban los scripts de marzo):
  - El ID de señante sale de data/processed/metadata.json -> sample_order (segundo campo del
    nombre SSSS_VVVV_RRRR). NO se usa data/dataset/, que fue sobrescrito con una partición
    elegida mirando el conjunto de prueba.
  - Selección del modelo (checkpoint y early stopping) SOLO con validación. El conjunto de
    prueba se toca una única vez, al final de cada corrida.
  - Sin umbrales de exactitud: se reportan las métricas que salgan.
  - Íntegramente en CPU.

Métricas por corrida: top-1, top-3, precisión, exhaustividad (recall) y F1 macro, matriz de
confusión (.json y .png) y classification_report (.json).

Uso:
    python src/train_loso.py --dry-run                 # solo construye y persiste las particiones
    python src/train_loso.py                           # LOSO completo, 5 pliegues
    python src/train_loso.py --tambien-aleatoria       # LOSO + esquema aleatorio 80/10/10
    python src/train_loso.py --folds 1 --epochs 1 --aug 1 --out /tmp/humo   # prueba de humo
    python src/train_loso.py --folds 0,3               # solo los pliegues 0 y 3
    python src/train_loso.py --dry-run --processed data/processed_holistic   # D-05: otro processed

--processed DIR (D-05, 14-sep-2026): directorio con X_padded.npy, y_labels.npy, lengths.npy y
metadata.json (por defecto data/processed). Permite entrenar sobre la re-extracción propia sin
tocar los archivos de marzo. Las particiones siguen en data/partitions/ por NOMBRE de muestra y
se cotejan con el sample_order del processed elegido; la ruta usada queda en la impresión
inicial y en resultados_loso.json (`processed_dir`).
"""

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PARTITIONS_DIR = BASE_DIR / "data" / "partitions"
DEFAULT_OUT_DIR = BASE_DIR / "models" / "loso"

sys.path.insert(0, str(BASE_DIR))

# Semilla de las particiones, fijada A PRIORI (año de ejecución del proyecto) antes de
# entrenar nada. No se usan 42 ni 0: ambas se eligieron en marzo de 2026 mirando el conjunto
# de prueba (la 42 fue la partición original; la 0 se adoptó en train_final_100.py porque
# sacaba del test una muestra mal clasificada, 0011_0004_0000). Cualquier cifra obtenida con
# ellas está contaminada. Esta semilla no se cambiará por ningún resultado.
SEMILLA_PARTICION = 2026
POR_QUE_NO_42_NI_0 = (
    "Las semillas 42 y 0 se descartan porque ambas se eligieron en marzo de 2026 mirando el "
    "conjunto de prueba: la 42 fue la particion original y la 0 se adopto en train_final_100.py "
    "porque sacaba del test una muestra mal clasificada (0011_0004_0000). Cualquier cifra "
    "obtenida con ellas esta contaminada por seleccion sobre el test. La semilla 2026 se fija "
    "a priori (ano de ejecucion) antes de entrenar nada y no se cambiara por ningun resultado."
)

ARCHIVO_ALEATORIA = PARTITIONS_DIR / "aleatoria_80_10_10.json"
ARCHIVO_LOSO = PARTITIONS_DIR / "loso_5_pliegues.json"


def _rel(ruta):
    """Ruta relativa al repositorio para imprimir; absoluta si está fuera de él."""
    ruta = Path(ruta)
    return str(ruta.relative_to(BASE_DIR)) if ruta.is_relative_to(BASE_DIR) else str(ruta)


# D-06 (14-sep-2026). LSC50 publica tres pares de muestras byte a byte idénticas en COLOR_BODY y en
# los CSV body/lh/rh (verificado contra el depósito de figshare, N-16). De cada par se conserva la
# repetición de menor índice y la otra se excluye de TODAS las particiones. Quedan 997 muestras.
DUPLICADOS_EXCLUIDOS = {
    "0002_0001_0001": "identica byte a byte a 0002_0001_0000 (COLOR_BODY, body, lh, rh); N-15/N-16",
    "0008_0002_0001": "identica byte a byte a 0008_0002_0000 (COLOR_BODY, body, lh, rh); N-15/N-16",
    "0016_0002_0001": "identica byte a byte a 0016_0002_0000 (COLOR_BODY, body, lh, rh); N-15/N-16",
}
VAL_POR_CLASE = 2        # validación: 2 muestras por seña (100) en ambos esquemas (D-06)
TEST_POR_CLASE = 2       # prueba del esquema aleatorio: 2 muestras por seña (100) (D-06)
SEQ_LENGTH = 30          # remuestreo a 30 pasos (train_final_100.py, --seq-length por defecto)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def cargar_datos(processed_dir=PROCESSED_DIR, representacion="manos_postura"):
    """Carga X_padded, y_labels, lengths y sample_order de processed_dir (--processed; por defecto
    data/processed); deriva el señante de cada muestra."""
    processed_dir = Path(processed_dir)
    for nombre in ("X_padded.npy", "y_labels.npy", "lengths.npy", "metadata.json"):
        if not (processed_dir / nombre).exists():
            raise SystemExit(f"falta {_rel(processed_dir / nombre)}: ejecute antes "
                             f"src/preprocess_lsc50.py (--salida {_rel(processed_dir)})")
    X_padded = np.load(processed_dir / "X_padded.npy")
    y = np.load(processed_dir / "y_labels.npy").astype(np.int64)
    lengths = np.load(processed_dir / "lengths.npy").astype(np.int64)
    with open(processed_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    nombres = list(meta["sample_order"])

    if not (len(nombres) == len(y) == len(lengths) == X_padded.shape[0]):
        raise SystemExit("metadata.json sample_order, y_labels.npy, lengths.npy y X_padded.npy "
                         "no tienen el mismo número de muestras")
    # Comprobación de alineación: el primer campo del nombre es la clase.
    clases_nombre = np.array([int(n.split("_")[0]) for n in nombres])
    if not np.array_equal(clases_nombre, y):
        raise SystemExit("y_labels.npy no coincide con el campo de seña de sample_order: "
                         "los archivos procesados no están alineados")
    senantes = [n.split("_")[1] for n in nombres]
    # D-08: columnas de la representación pedida (src/representacion.py).
    from src.representacion import columnas as columnas_representacion
    rostro = meta.get("rostro", {}).get("subconjunto", "ninguno")
    try:
        cols = columnas_representacion(representacion, rostro)
    except ValueError as exc:
        raise SystemExit(str(exc))
    esperado = 225 if rostro == "ninguno" else int(meta["rostro"]["columnas"][1])
    if X_padded.shape[2] != esperado:
        raise SystemExit(f"X_padded tiene {X_padded.shape[2]} rasgos por paso y metadata.json declara "
                         f"{esperado} (rostro: {rostro}); el processed no es coherente")
    return X_padded[:, :, cols], y, lengths, nombres, senantes


# ---------------------------------------------------------------------------
# Construcción de particiones (numpy puro, sin sklearn: el criterio queda explícito)
# ---------------------------------------------------------------------------

def _agrupar_por_clase(indices, y):
    grupos = {}
    for i in indices:
        grupos.setdefault(int(y[i]), []).append(int(i))
    return grupos


def seleccionar_por_clase(grupos, n_por_clase, rng):
    """Elige exactamente n_por_clase índices de cada clase, al azar dentro de la clase (D-06:
    conteo fijo por seña, no porcentaje). Devuelve (seleccionados, restantes) como diccionarios
    clase -> lista de índices. Exige que cada clase conserve al menos una muestra restante."""
    seleccion, resto = {}, {}
    for c in sorted(grupos):
        idx = np.array(grupos[c], dtype=np.int64)
        if len(idx) <= n_por_clase:
            raise SystemExit(f"clase {c}: {len(idx)} muestras disponibles; no alcanzan para "
                             f"apartar {n_por_clase} y dejar al menos 1")
        idx = idx[rng.permutation(len(idx))]
        seleccion[c] = sorted(int(i) for i in idx[:n_por_clase])
        resto[c] = sorted(int(i) for i in idx[n_por_clase:])
    return seleccion, resto


def _aplanar(grupos):
    return sorted(i for idx in grupos.values() for i in idx)


def construir_aleatoria(nombres, y, semilla, excluidas=DUPLICADOS_EXCLUIDOS):
    """Esquema 1: partición aleatoria estratificada por clase, ~80/10/10 por conteo fijo:
    2 de prueba y 2 de validación por seña; el resto entrenamiento. Excluye los duplicados."""
    rng = np.random.default_rng(semilla)
    N = len(nombres)
    incluidos = [i for i in range(N) if nombres[i] not in excluidas]
    grupos = _agrupar_por_clase(incluidos, y)
    test, resto = seleccionar_por_clase(grupos, TEST_POR_CLASE, rng)
    val, train = seleccionar_por_clase(resto, VAL_POR_CLASE, rng)
    indices = {"train": _aplanar(train), "val": _aplanar(val), "test": _aplanar(test)}
    return {
        "esquema": "aleatoria_80_10_10",
        "descripcion": "Particion aleatoria estratificada por clase, por conteo fijo: 2 muestras "
                       "de prueba y 2 de validacion por sena (100 y 100), el resto entrenamiento "
                       "(797 con 997 muestras). Es el esquema que el anteproyecto declara "
                       "optimista (mismos senantes en train y test); se reporta como contraste "
                       "del LOSO, no como cifra principal.",
        "excluidas": dict(excluidas),
        "muestras_disponibles": len(incluidos),
        "fuente": "data/processed/metadata.json -> sample_order (nombres SSSS_VVVV_RRRR: "
                  "sena, voluntario, repeticion)",
        "semilla": int(semilla),
        "generador": "numpy.random.default_rng(semilla) (PCG64)",
        "criterio": "D-06: se excluyen los duplicados; por clase se eligen al azar exactamente "
                    "2 muestras de prueba y luego 2 de validacion entre las restantes; el resto es "
                    "entrenamiento. Con 20 muestras por clase: 16 train / 2 val / 2 test; las tres "
                    "clases con 19 muestras quedan 15 / 2 / 2.",
        "por_que_no_42_ni_0": POR_QUE_NO_42_NI_0,
        "uso_del_test": "Solo para la evaluacion final. Seleccion de modelo, early stopping y "
                        "checkpoint unicamente con validacion.",
        "script": "src/train_loso.py",
        "indices_relativos_a": "posicion en metadata.json sample_order",
        "conteos": {k: len(v) for k, v in indices.items()},
        "conjuntos": {k: [nombres[i] for i in v] for k, v in indices.items()},
        "indices": indices,
    }


def construir_loso(nombres, y, senantes, semilla, excluidas=DUPLICADOS_EXCLUIDOS):
    """Esquema 2: LOSO de 5 pliegues. En el pliegue k, el señante k es la prueba completa;
    de los otros cuatro se separan 2 muestras por seña (100) como validación (D-06)."""
    lista_senantes = sorted(set(senantes))
    pliegues = []
    for k, s in enumerate(lista_senantes):
        # Una semilla derivada por pliegue para que la validación de cada pliegue sea
        # independiente y reproducible: semilla*10 + k.
        rng = np.random.default_rng(semilla * 10 + k)
        idx_test = [i for i, v in enumerate(senantes) if v == s and nombres[i] not in excluidas]
        idx_resto = [i for i, v in enumerate(senantes) if v != s and nombres[i] not in excluidas]
        grupos = _agrupar_por_clase(idx_resto, y)
        val, train = seleccionar_por_clase(grupos, VAL_POR_CLASE, rng)
        indices = {"train": _aplanar(train), "val": _aplanar(val), "test": sorted(idx_test)}
        pliegues.append({
            "pliegue": k,
            "senante_test": s,
            "semilla_validacion": int(semilla * 10 + k),
            "conteos": {kk: len(v) for kk, v in indices.items()},
            "conjuntos": {kk: [nombres[i] for i in v] for kk, v in indices.items()},
            "indices": indices,
        })
    return {
        "esquema": "loso_5_pliegues",
        "descripcion": "Leave-one-signer-out: cinco pliegues, uno por senante. En el pliegue k "
                       "todas las muestras del senante k son el conjunto de prueba; de los otros "
                       "cuatro senantes se separan 2 muestras por sena (100) como validacion "
                       "(early stopping y checkpoint) y el resto es entrenamiento. El senante de "
                       "prueba no se toca hasta la evaluacion final del pliegue. Los duplicados "
                       "del corpus quedan fuera de todos los conjuntos (D-06).",
        "excluidas": dict(excluidas),
        "muestras_disponibles": sum(1 for n in nombres if n not in excluidas),
        "fuente": "data/processed/metadata.json -> sample_order (nombres SSSS_VVVV_RRRR); "
                  "el senante es el segundo campo del nombre",
        "senantes": lista_senantes,
        "semilla": int(semilla),
        "generador": "numpy.random.default_rng(semilla*10 + k) por pliegue (PCG64)",
        "criterio": "D-06: prueba = todas las muestras no excluidas del senante k (sin azar). "
                    "Validacion = exactamente 2 muestras por sena de los otros cuatro senantes, "
                    "elegidas al azar dentro de cada clase (100). Entrenamiento = el resto "
                    "(697-700 segun cuantos duplicados excluidos caen fuera del senante de prueba). "
                    "Antes (13-sep) era round(10 %) = 80, desigual entre clases; ver D-06.",
        "por_que_no_42_ni_0": POR_QUE_NO_42_NI_0,
        "uso_del_test": "Solo para la evaluacion final de cada pliegue. Seleccion de modelo, "
                        "early stopping y checkpoint unicamente con validacion.",
        "script": "src/train_loso.py",
        "indices_relativos_a": "posicion en metadata.json sample_order",
        "pliegues": pliegues,
    }


# ---------------------------------------------------------------------------
# Verificación y persistencia
# ---------------------------------------------------------------------------

def verificar_particion(conjuntos, nombres, senante_test=None, n_clases=50,
                        excluidas=DUPLICADOS_EXCLUIDOS):
    """Comprueba que los conjuntos son disjuntos, cubren exactamente las N muestras menos las
    excluidas, contienen las n_clases clases y (en LOSO) que el señante de prueba no aparece
    en train ni val."""
    tr, va, te = (set(conjuntos[k]) for k in ("train", "val", "test"))
    if tr & va or tr & te or va & te:
        raise SystemExit("particion invalida: conjuntos no disjuntos")
    if (tr | va | te) != set(nombres) - set(excluidas):
        raise SystemExit("particion invalida: los conjuntos no cubren exactamente las muestras "
                         "no excluidas")
    if (tr | va | te) & set(excluidas):
        raise SystemExit("particion invalida: contiene muestras excluidas (duplicados, D-06)")
    for k in ("train", "val", "test"):
        clases = {int(n.split("_")[0]) for n in conjuntos[k]}
        if len(clases) != n_clases:
            raise SystemExit(f"particion invalida: el conjunto {k} tiene {len(clases)} clases")
    if senante_test is not None:
        if any(n.split("_")[1] != senante_test for n in te):
            raise SystemExit("particion invalida: el test contiene otro senante")
        if any(n.split("_")[1] == senante_test for n in tr | va):
            raise SystemExit("particion invalida: el senante de prueba aparece en train/val")


def _comparable(d):
    """Parte del JSON que define la partición (sin fecha ni entorno)."""
    if "pliegues" in d:
        return {"senantes": d["senantes"], "semilla": d["semilla"],
                "excluidas": sorted(d.get("excluidas", {})),
                "pliegues": [p["conjuntos"] for p in d["pliegues"]]}
    return {"semilla": d["semilla"], "excluidas": sorted(d.get("excluidas", {})),
            "conjuntos": d["conjuntos"]}


def persistir(particion, ruta):
    """Escribe la partición si no existe. Si existe, exige que sea idéntica: el archivo
    persistido es el entregable de 1.e y no se sobrescribe en silencio."""
    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)
    if ruta.exists():
        with open(ruta, encoding="utf-8") as f:
            previa = json.load(f)
        if _comparable(previa) == _comparable(particion):
            print(f"  {_rel(ruta)}: ya existe y es identica (no se reescribe)")
            return previa
        raise SystemExit(
            f"{ruta} existe y NO coincide con la particion regenerada. No se sobrescribe: "
            "si de verdad hay que cambiar la particion, borre el archivo a mano y registre "
            "el motivo en DECISIONES_TESIS.txt.")
    particion = dict(particion)
    particion["fecha_creacion"] = datetime.now().isoformat(timespec="seconds")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(particion, f, indent=1, ensure_ascii=False)
    print(f"  {_rel(ruta)}: escrita")
    return particion


def describir_conjunto(nombres_conj):
    """Resumen de un conjunto: total, clases, muestras por clase y por señante."""
    por_clase = Counter(int(n.split("_")[0]) for n in nombres_conj)
    por_senante = Counter(n.split("_")[1] for n in nombres_conj)
    hist = Counter(por_clase.values())
    partes = [f"{len(nombres_conj):4d} muestras", f"{len(por_clase)} clases"]
    partes.append("por clase: " + ", ".join(
        f"{v} muestras en {k} clases" for v, k in sorted(hist.items())))
    partes.append("por senante: " + ", ".join(f"{s}:{c}" for s, c in sorted(por_senante.items())))
    return " | ".join(partes)


def tabla_por_clase(conjuntos_lista, encabezados, nombres_senas=None):
    """Imprime una tabla clase x conjunto con los conteos (train/val/test)."""
    cab = "clase" + (" nombre".ljust(18) if nombres_senas else "")
    for e in encabezados:
        cab += f" | {e:>12}"
    print("  " + cab)
    print("  " + "-" * len(cab))
    for c in range(50):
        fila = f"{c:5d}"
        if nombres_senas:
            fila += " " + nombres_senas.get(c, "")[:16].ljust(17)
        for conj in conjuntos_lista:
            cel = "/".join(
                str(sum(1 for n in conj[k] if int(n.split("_")[0]) == c))
                for k in ("train", "val", "test"))
            fila += f" | {cel:>12}"
        print("  " + fila)


def cargar_nombres_senas():
    """Nombres de las 50 señas (grammar_mapper.SIGN_VOCABULARY, verificado contra la Tabla 2
    del artículo de LSC50). Si no se puede importar, se usan solo los ids."""
    try:
        from src.grammar_mapper import SIGN_VOCABULARY
        return {int(k): v for k, v in SIGN_VOCABULARY.items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Pipeline canónico (importado de train_final_100.py; verificado que importa sin efectos
# secundarios: no escribe archivos ni entrena nada al importarse)
# ---------------------------------------------------------------------------

def importar_pipeline_canonico(representacion="manos_postura"):
    """Importa TensorFlow y la arquitectura canónica solo cuando hace falta entrenar. El espejado
    de la aumentación es el de la representación (D-08); para manos_postura, augment_mirror."""
    import tensorflow as tf
    # CPU únicamente (§1.4, §2.5.2). En el equipo del proyecto no hay GPU visible; esta
    # llamada lo garantiza aunque hubiera una.
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    from src.train_final_100 import build_model, add_velocity, resample_sequence
    from src.train_lstm import augment_mirror, augment_noise, augment_temporal, augment_scale
    from src.representacion import espejar
    if representacion != "manos_postura":
        augment_mirror = lambda X: espejar(X, representacion)  # noqa: E731

    # Lista de transformaciones de refined_augmentation (train_final_100.py, def en linea 113),
    # desglosada para poder controlar el factor con --aug. Mismo orden y parametros: con
    # --aug 10 el resultado es exactamente el de refined_augmentation (original + 9).
    transformaciones = [
        ("espejo", lambda X: augment_mirror(X)),
        ("ruido_0.005", lambda X: augment_noise(X, 0.005)),
        ("ruido_0.012", lambda X: augment_noise(X, 0.012)),
        ("ruido_0.020", lambda X: augment_noise(X, 0.020)),
        ("temporal_0.85-0.95", lambda X: augment_temporal(X, (0.85, 0.95))),
        ("temporal_1.05-1.15", lambda X: augment_temporal(X, (1.05, 1.15))),
        ("escala_0.85-1.15", lambda X: augment_scale(X, (0.85, 1.15))),
        ("espejo+ruido_0.008", lambda X: augment_noise(augment_mirror(X), 0.008)),
        ("espejo+temporal_0.9-1.1", lambda X: augment_temporal(augment_mirror(X), (0.9, 1.1))),
    ]
    return tf, build_model, add_velocity, resample_sequence, transformaciones


def aumentar(X, y, factor, transformaciones):
    """Aumentación con factor entero: 1 = sin aumentar; k = original + (k-1) transformaciones
    en el orden de refined_augmentation; máximo 10. Usa np.random como la función canónica."""
    if factor <= 1:
        return X, y
    partes_X, partes_y = [X], [y]
    for _, fn in transformaciones[:factor - 1]:
        partes_X.append(fn(X))
        partes_y.append(y.copy())
    Xo = np.concatenate(partes_X)
    yo = np.concatenate(partes_y)
    perm = np.random.permutation(len(yo))
    return Xo[perm], yo[perm]


def remuestrear_todas(X_padded, lengths, resample_sequence, seq_length):
    """Remuestreo de cada secuencia a seq_length pasos (mismo criterio que train_final_100)."""
    N = X_padded.shape[0]
    X = np.zeros((N, seq_length, X_padded.shape[2]), dtype=np.float32)
    for i in range(N):
        T = int(lengths[i])
        X[i] = resample_sequence(X_padded[i, :T, :], seq_length)
    return X


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación de una corrida
# ---------------------------------------------------------------------------

def entrenar_y_evaluar(etiqueta, conjuntos, pos, X_seq, y, args, out_dir, canon, nombres_senas):
    """Entrena con train, selecciona con val, evalúa UNA vez en test. Guarda todo en out_dir."""
    tf, build_model, add_velocity, _, transformaciones = canon
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 precision_recall_fscore_support)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    t_inicio = time.perf_counter()

    idx = {k: np.array([pos[n] for n in conjuntos[k]], dtype=np.int64) for k in conjuntos}
    n_clases = int(y.max()) + 1

    # Semilla de entrenamiento (inicialización, aumentación, barajado). No es la de partición.
    tf.keras.utils.set_random_seed(args.seed)

    X_tr, y_tr = aumentar(X_seq[idx["train"]], y[idx["train"]], args.aug, transformaciones)
    X_tr_f = add_velocity(X_tr)
    X_val_f = add_velocity(X_seq[idx["val"]])
    y_tr_oh = tf.keras.utils.to_categorical(y_tr, n_clases)
    y_val_oh = tf.keras.utils.to_categorical(y[idx["val"]], n_clases)
    n_features = X_tr_f.shape[2]

    print(f"  entrenamiento: {len(y_tr)} secuencias (factor {args.aug}) x {X_tr_f.shape[1]} "
          f"pasos x {n_features} rasgos | validacion: {len(idx['val'])} | prueba: {len(idx['test'])}")

    # Hiperparámetros de train_final_100.py (CosineDecay en linea 389)
    model = build_model(X_seq.shape[1], n_features, n_clases, dropout=args.dropout)
    steps = len(y_tr) // args.batch_size + 1
    lr = tf.keras.optimizers.schedules.CosineDecay(0.001, args.epochs * steps, alpha=1e-6)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    ruta_modelo = out_dir / "modelo.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.patience, restore_best_weights=True,
            verbose=1, mode="max"),
        tf.keras.callbacks.ModelCheckpoint(
            str(ruta_modelo), monitor="val_accuracy", save_best_only=True,
            verbose=0, mode="max"),
    ]

    t_fit = time.perf_counter()
    hist = model.fit(
        X_tr_f, y_tr_oh,
        validation_data=(X_val_f, y_val_oh),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, verbose=2,
    )
    tiempo_fit = time.perf_counter() - t_fit
    historial = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    epocas = len(historial["loss"])
    mejor_epoca = int(np.argmax(historial["val_accuracy"])) + 1
    with open(out_dir / "historial.json", "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=1)

    # Curvas de entrenamiento (reales, no placeholder)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (m, titulo) in zip(axes, [("accuracy", "Exactitud"), ("loss", "Perdida")]):
        ax.plot(range(1, epocas + 1), historial[m], label="entrenamiento")
        ax.plot(range(1, epocas + 1), historial["val_" + m], label="validacion")
        ax.axvline(mejor_epoca, color="gray", linestyle="--", linewidth=0.8,
                   label=f"mejor epoca ({mejor_epoca})")
        ax.set_title(titulo)
        ax.set_xlabel("epoca")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Curvas de entrenamiento - {etiqueta}")
    fig.tight_layout()
    fig.savefig(out_dir / "curvas_entrenamiento.png", dpi=130)
    plt.close(fig)

    # Modelo seleccionado con validación: el checkpoint de mejor val_accuracy.
    from src.representacion import escribir_procedencia
    escribir_procedencia(ruta_modelo, args.processed, args.representacion, SEQ_LENGTH,
                         "src/train_loso.py", BASE_DIR)
    del model
    tf.keras.backend.clear_session()
    try:
        model = tf.keras.models.load_model(str(ruta_modelo))
    except ValueError:
        model = tf.keras.models.load_model(str(ruta_modelo), safe_mode=False)

    # ----- Evaluación ÚNICA sobre el conjunto de prueba -----
    X_te_f = add_velocity(X_seq[idx["test"]])
    y_te = y[idx["test"]]
    probs = model.predict(X_te_f, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    top1_acc = float(np.mean(y_pred == y_te))
    top3_acc = float(np.mean(np.any(top3 == y_te[:, None], axis=1)))
    etiquetas = list(range(n_clases))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=etiquetas, average="macro", zero_division=0)
    nombres_obj = [f"{c:02d}_{nombres_senas.get(c, '')}" if nombres_senas else str(c)
                   for c in etiquetas]
    reporte = classification_report(
        y_te, y_pred, labels=etiquetas, target_names=nombres_obj,
        output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=etiquetas)

    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=1, ensure_ascii=False)
    with open(out_dir / "matriz_confusion.json", "w", encoding="utf-8") as f:
        json.dump({"filas": "clase real", "columnas": "clase predicha",
                   "etiquetas": etiquetas, "nombres": nombres_obj,
                   "matriz": cm.tolist()}, f, indent=1, ensure_ascii=False)
    with open(out_dir / "predicciones.json", "w", encoding="utf-8") as f:
        json.dump([{"muestra": conjuntos["test"][i], "real": int(y_te[i]),
                    "predicha": int(y_pred[i]), "top3": [int(t) for t in top3[i]],
                    "prob_predicha": float(probs[i, y_pred[i]])}
                   for i in range(len(y_te))], f, indent=1)

    # Matriz de confusión en PNG
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title(f"Matriz de confusion - {etiqueta} (top-1 {top1_acc:.1%}, n={len(y_te)})",
                 fontsize=13)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_xticks(etiquetas)
    ax.set_yticks(etiquetas)
    ax.set_xticklabels(nombres_obj, fontsize=5, rotation=90)
    ax.set_yticklabels(nombres_obj, fontsize=5)
    umbral = cm.max() / 2 if cm.max() > 0 else 0
    for i in etiquetas:
        for j in etiquetas:
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=4,
                        color="white" if cm[i, j] > umbral else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "matriz_confusion.png", dpi=150)
    plt.close(fig)

    tiempo_total = time.perf_counter() - t_inicio
    metricas = {
        "corrida": etiqueta,
        "n_train_original": int(len(idx["train"])),
        "n_train_aumentado": int(len(y_tr)),
        "n_val": int(len(idx["val"])),
        "n_test": int(len(idx["test"])),
        "top1": top1_acc,
        "top3": top3_acc,
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "n_errores": int(np.sum(y_pred != y_te)),
        "mejor_val_accuracy": float(max(historial["val_accuracy"])),
        "epocas_ejecutadas": int(epocas),
        "mejor_epoca": mejor_epoca,
        "tiempo_entrenamiento_s": round(tiempo_fit, 1),
        "tiempo_total_s": round(tiempo_total, 1),
        "modelo": _rel(ruta_modelo),
    }
    with open(out_dir / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=1)

    print(f"  [{etiqueta}] top-1 {top1_acc:.4f} | top-3 {top3_acc:.4f} | P {prec:.4f} | "
          f"R {rec:.4f} | F1 {f1:.4f} | errores {metricas['n_errores']}/{len(y_te)} | "
          f"epocas {epocas} (mejor {mejor_epoca}) | {tiempo_total:.0f} s")

    del model, X_tr, X_tr_f, y_tr, y_tr_oh
    tf.keras.backend.clear_session()
    return metricas


METRICAS_AGREGADAS = ["top1", "top3", "precision_macro", "recall_macro", "f1_macro",
                      "mejor_val_accuracy", "epocas_ejecutadas", "tiempo_total_s"]


def agregar(lista_metricas):
    """Media y desviación estándar muestral (ddof=1) entre pliegues de cada métrica."""
    agregado = {}
    for m in METRICAS_AGREGADAS:
        valores = [float(d[m]) for d in lista_metricas]
        agregado[m] = {
            "media": float(np.mean(valores)),
            "desviacion_estandar": float(np.std(valores, ddof=1)) if len(valores) > 1 else None,
            "min": float(np.min(valores)),
            "max": float(np.max(valores)),
            "valores": valores,
        }
    agregado["_nota"] = ("desviacion_estandar es la muestral (ddof=1) sobre los pliegues "
                         "ejecutados; null si hay un solo pliegue")
    return agregado


def info_entorno():
    ent = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "plataforma": platform.platform(),
        "maquina": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import tensorflow as tf
        import keras
        import sklearn
        ent["tensorflow"] = tf.__version__
        ent["keras"] = keras.__version__
        ent["scikit_learn"] = sklearn.__version__
        ent["gpus_visibles"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        pass
    return ent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parsear_folds(texto, n_pliegues):
    """'5' -> los primeros 5 pliegues; '0,3' -> los pliegues 0 y 3."""
    texto = texto.strip()
    if "," in texto:
        folds = sorted({int(t) for t in texto.split(",") if t.strip() != ""})
    else:
        folds = list(range(int(texto)))
    for k in folds:
        if k < 0 or k >= n_pliegues:
            raise SystemExit(f"--folds: el pliegue {k} no existe (hay {n_pliegues})")
    if not folds:
        raise SystemExit("--folds: no se indicó ningún pliegue")
    return folds


def main():
    parser = argparse.ArgumentParser(
        description="Particiones LSC50 (aleatoria 80/10/10 y LOSO 5 pliegues) y "
                    "entrenamiento/evaluacion LOSO con la arquitectura canonica, en CPU.")
    parser.add_argument("--epochs", type=int, default=200, help="epocas maximas (defecto 200)")
    parser.add_argument("--aug", type=int, default=10,
                        help="factor de aumento: 1 = sin aumentar, 10 = refined_augmentation "
                             "completa (defecto 10)")
    parser.add_argument("--folds", type=str, default="5",
                        help="numero N (los primeros N pliegues) o lista 'a,b,c' (defecto 5)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="semilla de entrenamiento (no de particion; defecto 2026)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR),
                        help="directorio de salida (defecto models/loso)")
    parser.add_argument("--processed", type=str, default=str(PROCESSED_DIR),
                        help="directorio con X_padded.npy, y_labels.npy, lengths.npy y metadata.json "
                             "(defecto data/processed; D-05: la salida de preprocess_lsc50.py --salida)")
    parser.add_argument("--representacion", choices=["manos", "manos_postura", "manos_postura_rostro"],
                        default="manos_postura",
                        help="D-08: bloques de entrada: manos (126), manos_postura (225, defecto) o "
                             "manos_postura_rostro (405; requiere un processed con --rostro cejas_labios)")
    parser.add_argument("--dry-run", action="store_true",
                        help="solo construye y persiste las particiones e imprime los conteos")
    parser.add_argument("--tambien-aleatoria", action="store_true",
                        help="ademas del LOSO, una corrida con el esquema aleatorio 80/10/10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=25,
                        help="paciencia del early stopping sobre val_accuracy (defecto 25)")
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    if not 1 <= args.aug <= 10:
        raise SystemExit("--aug debe estar entre 1 y 10")

    print("=" * 72)
    print("  LSC50 - Particiones e independencia del senante (LOSO)")
    print("=" * 72)

    # 1. Datos
    processed_dir = Path(args.processed).expanduser().resolve()
    X_padded, y, lengths, nombres, senantes = cargar_datos(processed_dir, args.representacion)
    n_clases = int(y.max()) + 1
    print(f"\n[1] Datos ({_rel(processed_dir)}, representacion {args.representacion}): "
          f"{len(nombres)} muestras, {n_clases} clases, "
          f"{len(set(senantes))} senantes ({', '.join(sorted(set(senantes)))}); "
          f"X_padded {X_padded.shape}, longitudes {int(lengths.min())}-{int(lengths.max())}")
    nombres_senas = cargar_nombres_senas()

    # 2. Particiones (construir, verificar, persistir)
    print(f"\n[2] Particiones (semilla a priori {SEMILLA_PARTICION}; no 42 ni 0: elegidas mirando el test)")
    print(f"    Excluidas (D-06, duplicados byte a byte): {', '.join(sorted(DUPLICADOS_EXCLUIDOS))} "
          f"-> {len(nombres) - len(DUPLICADOS_EXCLUIDOS)} muestras disponibles; "
          f"validacion {VAL_POR_CLASE} por sena en ambos esquemas")
    aleatoria = construir_aleatoria(nombres, y, SEMILLA_PARTICION)
    verificar_particion(aleatoria["conjuntos"], nombres, n_clases=n_clases)
    aleatoria = persistir(aleatoria, ARCHIVO_ALEATORIA)

    loso = construir_loso(nombres, y, senantes, SEMILLA_PARTICION)
    for p in loso["pliegues"]:
        verificar_particion(p["conjuntos"], nombres, senante_test=p["senante_test"],
                            n_clases=n_clases)
    loso = persistir(loso, ARCHIVO_LOSO)

    # Lo que se usa de aquí en adelante es lo que quedó en disco.
    with open(ARCHIVO_ALEATORIA, encoding="utf-8") as f:
        aleatoria = json.load(f)
    with open(ARCHIVO_LOSO, encoding="utf-8") as f:
        loso = json.load(f)
    pos = {n: i for i, n in enumerate(nombres)}
    for conj in [aleatoria["conjuntos"]] + [p["conjuntos"] for p in loso["pliegues"]]:
        for k in conj:
            faltantes = [n for n in conj[k] if n not in pos]
            if faltantes:
                raise SystemExit(f"la particion persistida nombra muestras que no existen: "
                                 f"{faltantes[:3]}")

    print("\n  Esquema aleatorio 80/10/10 (por conteo fijo, D-06):")
    for k in ("train", "val", "test"):
        print(f"    {k:5s}: {describir_conjunto(aleatoria['conjuntos'][k])}")
    print(f"\n  Esquema LOSO ({len(loso['pliegues'])} pliegues):")
    for p in loso["pliegues"]:
        print(f"    pliegue {p['pliegue']} (senante de prueba {p['senante_test']}):")
        for k in ("train", "val", "test"):
            print(f"      {k:5s}: {describir_conjunto(p['conjuntos'][k])}")

    if args.dry_run:
        print("\n  Conteos por clase (train/val/test), esquema aleatorio:")
        tabla_por_clase([aleatoria["conjuntos"]], ["aleatoria"], nombres_senas)
        print("\n  Conteos por clase (train/val/test), LOSO por pliegue:")
        tabla_por_clase([p["conjuntos"] for p in loso["pliegues"]],
                        [f"pliegue {p['pliegue']}" for p in loso["pliegues"]], nombres_senas)
        print("\n[dry-run] Particiones construidas y persistidas. No se entrena nada.")
        return

    # 3. Plan de ejecución
    folds = parsear_folds(args.folds, len(loso["pliegues"]))
    out_dir = Path(args.out).expanduser().resolve()
    print(f"\n[3] Plan: LOSO pliegues {folds}" + (" + esquema aleatorio" if args.tambien_aleatoria else ""))
    print(f"    epochs max {args.epochs} | aug x{args.aug} | batch {args.batch_size} | "
          f"paciencia {args.patience} | dropout {args.dropout} | semilla entrenamiento {args.seed}")
    print(f"    arquitectura: build_model de train_final_100.py (CNN+BiLSTM+atencion), entrada "
          f"({SEQ_LENGTH}, {2 * X_padded.shape[2]}) = {X_padded.shape[2]} coordenadas ({args.representacion}) + "
          f"{X_padded.shape[2]} velocidades")
    print(f"    seleccion de modelo: checkpoint y early stopping sobre val_accuracy; "
          f"el test se evalua una sola vez al final de cada corrida")
    print(f"    salida: {out_dir}")
    for k in folds:
        c = loso["pliegues"][k]["conteos"]
        print(f"    pliegue {k}: train {c['train']} (x{args.aug} = {c['train'] * args.aug}) | "
              f"val {c['val']} | test {c['test']} (senante {loso['pliegues'][k]['senante_test']})")
    if args.tambien_aleatoria:
        c = aleatoria["conteos"]
        print(f"    aleatoria: train {c['train']} (x{args.aug} = {c['train'] * args.aug}) | "
              f"val {c['val']} | test {c['test']}")

    print("\n[4] Cargando TensorFlow y la arquitectura canonica...")
    t0 = time.perf_counter()
    canon = importar_pipeline_canonico(args.representacion)
    tf = canon[0]
    print(f"    TensorFlow {tf.__version__} en {time.perf_counter() - t0:.1f} s; "
          f"GPUs visibles: {tf.config.list_physical_devices('GPU')} (se usa solo CPU)")
    X_seq = remuestrear_todas(X_padded, lengths, canon[3], SEQ_LENGTH)
    del X_padded
    print(f"    secuencias remuestreadas a {SEQ_LENGTH} pasos: {X_seq.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ruta_resultados = out_dir / "resultados_loso.json"
    resultados = {
        "esquema": "loso_5_pliegues",
        "processed_dir": _rel(processed_dir),
        "representacion": args.representacion,
        "particion_loso": _rel(ARCHIVO_LOSO),
        "particion_aleatoria": _rel(ARCHIVO_ALEATORIA),
        "configuracion": {
            "arquitectura": "CNN+BiLSTM+atencion (train_final_100.build_model)",
            "seq_length": SEQ_LENGTH, "n_features": int(2 * X_seq.shape[2]), "velocidades": True,
            "epochs_max": args.epochs, "aug": args.aug, "batch_size": args.batch_size,
            "patience": args.patience, "dropout": args.dropout, "seed_entrenamiento": args.seed,
            "monitor": "val_accuracy", "pliegues_solicitados": folds,
            "semilla_particion": SEMILLA_PARTICION,
        },
        "pliegues": [],
        "agregado": None,
        "aleatoria_80_10_10": None,
        "entorno": info_entorno(),
    }

    def guardar_resultados():
        with open(ruta_resultados, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=1, ensure_ascii=False)

    # 5. LOSO
    for k in folds:
        p = loso["pliegues"][k]
        print(f"\n{'=' * 72}\n  PLIEGUE {k} - senante de prueba {p['senante_test']}\n{'=' * 72}")
        m = entrenar_y_evaluar(f"pliegue_{k}", p["conjuntos"], pos, X_seq, y, args,
                               out_dir / f"pliegue_{k}", canon, nombres_senas)
        m["pliegue"] = k
        m["senante_test"] = p["senante_test"]
        resultados["pliegues"].append(m)
        resultados["agregado"] = agregar(resultados["pliegues"])
        guardar_resultados()

    # 6. Esquema aleatorio (contraste)
    if args.tambien_aleatoria:
        print(f"\n{'=' * 72}\n  ESQUEMA ALEATORIO 80/10/10 (contraste)\n{'=' * 72}")
        m = entrenar_y_evaluar("aleatoria_80_10_10", aleatoria["conjuntos"], pos, X_seq, y,
                               args, out_dir / "aleatoria_80_10_10", canon, nombres_senas)
        resultados["aleatoria_80_10_10"] = m
        guardar_resultados()

    # 7. Resumen
    print(f"\n{'=' * 72}\n  RESUMEN\n{'=' * 72}")
    print(f"  {'corrida':22s} {'top-1':>7} {'top-3':>7} {'P':>7} {'R':>7} {'F1':>7} {'epocas':>7} {'seg':>7}")
    for m in resultados["pliegues"]:
        print(f"  {m['corrida'] + ' (' + m['senante_test'] + ')':22s} {m['top1']:7.4f} {m['top3']:7.4f} "
              f"{m['precision_macro']:7.4f} {m['recall_macro']:7.4f} {m['f1_macro']:7.4f} "
              f"{m['epocas_ejecutadas']:7d} {m['tiempo_total_s']:7.0f}")
    ag = resultados["agregado"]
    de = lambda mm: "   n/a" if ag[mm]["desviacion_estandar"] is None else f"{ag[mm]['desviacion_estandar']:6.4f}"
    print(f"  {'media LOSO':22s} {ag['top1']['media']:7.4f} {ag['top3']['media']:7.4f} "
          f"{ag['precision_macro']['media']:7.4f} {ag['recall_macro']['media']:7.4f} "
          f"{ag['f1_macro']['media']:7.4f}")
    print(f"  {'DE LOSO (ddof=1)':22s} {de('top1'):>7} {de('top3'):>7} {de('precision_macro'):>7} "
          f"{de('recall_macro'):>7} {de('f1_macro'):>7}")
    if resultados["aleatoria_80_10_10"]:
        m = resultados["aleatoria_80_10_10"]
        print(f"  {'aleatoria 80/10/10':22s} {m['top1']:7.4f} {m['top3']:7.4f} "
              f"{m['precision_macro']:7.4f} {m['recall_macro']:7.4f} {m['f1_macro']:7.4f} "
              f"{m['epocas_ejecutadas']:7d} {m['tiempo_total_s']:7.0f}")
    print(f"\n  Resultados: {ruta_resultados}")


if __name__ == "__main__":
    main()
