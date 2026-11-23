"""
Entrenador canónico de LSC50 sobre el esquema aleatorio 80/10/10 (CNN + BiLSTM + atención).

Es el único script de entrenamiento canónico del proyecto para el esquema aleatorio. El esquema
independiente del señante (LOSO, §2.2.4) vive en src/train_loso.py, que IMPORTA de aquí la
arquitectura y el preprocesamiento: build_model, add_velocity, resample_sequence y
refined_augmentation. Esas cuatro funciones conservan nombre, firma y comportamiento exactos;
no se modifican en este archivo. El nombre del archivo se conserva por esa dependencia y por la
historia del repositorio (tag exploracion-marzo-2026); ya no describe un objetivo de exactitud.

Qué se quitó el 13-sep-2026 y por qué
--------------------------------------
ESTADO_PROYECTO_2026-09-13.md, hallazgo B: «todas las cifras están contaminadas por el conjunto
de prueba [...] Ninguna cifra actual es reportable. Esto no se arregla: se rehace». La versión de
marzo de este script era parte de esa contaminación, así que se eliminó:
  - La partición con train_test_split y una semilla elegida en marzo porque dejaba fuera del
    conjunto de prueba una muestra que ningún modelo acertaba. Elegir la semilla mirando el test
    es selección sobre el test.
  - La evaluación de cada modelo individual sobre el test para escoger «el mejor» y copiarlo a
    models/best_model.keras (selección con el conjunto de prueba).
  - El TTA (aumentación en tiempo de prueba) y la comparación ensemble vs. ensemble+TTA decidida
    por la exactitud en test.
  - Los comentarios y comprobaciones sobre muestras concretas del conjunto de prueba, los
    mensajes de «objetivo alcanzado» y la gráfica de curvas de aprendizaje que era un marcador de
    posición sin datos.
  - La escritura de la partición en data/dataset/, que sobrescribía el dataset del proyecto con
    una partición elegida mirando el test.
  - Los imports no usados de train_lstm.py.

Qué hace ahora
--------------
  - La partición viene de data/partitions/aleatoria_80_10_10.json (actividad 1.e; D-02 en
    DECISIONES_TESIS.txt): nombres de muestra SSSS_VVVV_RRRR traducidos a índices con
    data/processed/metadata.json -> sample_order. Este script NO construye particiones; si el
    archivo no existe hay que generarlo con `python src/train_loso.py --dry-run`.
  - Selección del modelo (checkpoint y early stopping sobre val_accuracy) SOLO con validación.
    Con varias semillas, la semilla «seleccionada» es la de mejor val_accuracy del checkpoint.
    De cada checkpoint se registran sobre VALIDACIÓN las métricas que §2.2.4 nombra para la
    selección (top-1, top-3, precisión/recall/F1 macro), en entrenamiento.json y resultados.json.
  - El conjunto de prueba se evalúa UNA sola vez, al final, sobre los sistemas declarados de
    antemano: el modelo seleccionado por validación y, si se pidió más de una semilla, el
    promedio de softmax de todas (ensemble), reportado como tal. Nada se elige mirando el test.
  - Salidas en --out (por defecto models/canonico/); nunca escribe models/best_model.keras,
    data/dataset/ ni las salidas de otras piezas (models/loso/, models/mediapipe/). Guarda top-1,
    top-3, precisión/recall/F1 macro, matriz de confusión (.json y .png), classification_report,
    predicciones y curvas de aprendizaje reales (historial.json y .png por semilla). No publica
    el top-1 de prueba por semilla: entre semillas solo se compara la validación.
  - Sin umbrales de exactitud (el anteproyecto no fija ninguno). Íntegramente en CPU.
  - --processed DIR (D-05, 14-sep-2026): directorio con X_padded.npy, y_labels.npy, lengths.npy
    y metadata.json (por defecto data/processed). Permite entrenar sobre la re-extracción propia
    sin tocar los archivos de marzo. La partición sigue en data/partitions/ por NOMBRE de muestra
    y se coteja con el sample_order del processed elegido; la ruta usada queda en la impresión
    inicial y en resultados.json (`processed_dir`).

Uso
---
    python src/train_final_100.py --dry-run                      # verifica datos y partición
    python src/train_final_100.py --dry-run --processed data/processed_holistic   # D-05
    python src/train_final_100.py                                # 1 semilla, 200 épocas, aug x10
    python src/train_final_100.py --semillas 5                   # 5 semillas + ensemble
    python src/train_final_100.py --epochs 1 --aug 1 --semillas 1 --out /tmp/humo   # humo
"""

import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PARTITIONS_DIR = BASE_DIR / "data" / "partitions"
ARCHIVO_PARTICION = PARTITIONS_DIR / "aleatoria_80_10_10.json"
DEFAULT_OUT_DIR = BASE_DIR / "models" / "canonico"

sys.path.insert(0, str(BASE_DIR))
from src.train_lstm import (
    N_TOTAL_FEATURES,
    augment_mirror, augment_noise, augment_scale, augment_temporal,
)
from src.representacion import (  # D-08: candidatos de representación y espejado generalizado
    REPRESENTACIONES, REPRESENTACION_POR_DEFECTO, columnas as columnas_representacion, espejar,
    escribir_procedencia,
)

# Semilla base de ENTRENAMIENTO (inicialización de pesos, aumentación, barajado). No es la
# semilla de partición: esa está fijada en data/partitions/ (D-02) y este script no la toca.
SEMILLA_BASE = 2026


# ---------------------------------------------------------------------------
# Arquitectura y preprocesamiento canónicos (importados por src/train_loso.py).
# INVARIANTE: nombre, firma y comportamiento de estas cuatro funciones no cambian.
# ---------------------------------------------------------------------------

def resample_sequence(seq, target):
    T, F = seq.shape
    if T == target:
        return seq
    idx = np.linspace(0, T - 1, target)
    out = np.zeros((target, F), dtype=np.float32)
    for f in range(F):
        out[:, f] = np.interp(idx, np.arange(T), seq[:, f])
    return out


def add_velocity(X):
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, vel], axis=-1)


def refined_augmentation(X, y):
    """10x augmentation with proven transforms."""
    all_X, all_y = [X], [y]
    def add(Xn):
        all_X.append(Xn)
        all_y.append(y.copy())

    add(augment_mirror(X))
    add(augment_noise(X, 0.005))
    add(augment_noise(X, 0.012))
    add(augment_noise(X, 0.020))
    add(augment_temporal(X, (0.85, 0.95)))
    add(augment_temporal(X, (1.05, 1.15)))
    add(augment_scale(X, (0.85, 1.15)))
    add(augment_noise(augment_mirror(X), 0.008))
    add(augment_temporal(augment_mirror(X), (0.9, 1.1)))

    Xo = np.concatenate(all_X)
    yo = np.concatenate(all_y)
    perm = np.random.permutation(len(yo))
    return Xo[perm], yo[perm]


def build_model(n_frames, n_features, n_classes, dropout=0.3):
    """CNN + BiLSTM + Attention (same as v3)."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    x = tf.keras.layers.Conv1D(128, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    att = tf.keras.layers.Dense(1)(x)
    att = tf.keras.layers.Softmax(axis=1)(att)
    ctx = tf.keras.layers.Multiply()([x, att])
    ctx = tf.keras.layers.GlobalAveragePooling1D()(ctx)
    avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    mx = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([ctx, avg, mx])

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


# ---------------------------------------------------------------------------
# Aumentación con factor reducible (pruebas de humo). Con --aug 10 es refined_augmentation.
# ---------------------------------------------------------------------------

# Mismas transformaciones, orden y parámetros que refined_augmentation, desglosadas para poder
# reducir el factor con --aug. Con factor 10 se llama a refined_augmentation tal cual.
TRANSFORMACIONES_AUMENTO = [
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


def transformaciones_para(representacion):
    """Las transformaciones de refined_augmentation con el espejado de la representación (D-08).
    Para «manos_postura» es TRANSFORMACIONES_AUMENTO tal cual (augment_mirror de marzo)."""
    if representacion == "manos_postura":
        return TRANSFORMACIONES_AUMENTO
    esp = lambda X: espejar(X, representacion)  # noqa: E731
    return [
        ("espejo", lambda X: esp(X)),
        ("ruido_0.005", lambda X: augment_noise(X, 0.005)),
        ("ruido_0.012", lambda X: augment_noise(X, 0.012)),
        ("ruido_0.020", lambda X: augment_noise(X, 0.020)),
        ("temporal_0.85-0.95", lambda X: augment_temporal(X, (0.85, 0.95))),
        ("temporal_1.05-1.15", lambda X: augment_temporal(X, (1.05, 1.15))),
        ("escala_0.85-1.15", lambda X: augment_scale(X, (0.85, 1.15))),
        ("espejo+ruido_0.008", lambda X: augment_noise(esp(X), 0.008)),
        ("espejo+temporal_0.9-1.1", lambda X: augment_temporal(esp(X), (0.9, 1.1))),
    ]


def aumentar(X, y, factor, representacion="manos_postura"):
    """Factor entero 1..10: 1 = sin aumentar; 10 = refined_augmentation completa (original + 9);
    k intermedio = original + las primeras k-1 transformaciones, en el mismo orden. Con otra
    representación (D-08) las mismas transformaciones con su espejado; mismo consumo de np.random."""
    if factor <= 1:
        return X, y
    if factor >= 10 and representacion == "manos_postura":
        return refined_augmentation(X, y)
    partes_X, partes_y = [X], [y]
    for _, fn in transformaciones_para(representacion)[:factor - 1]:
        partes_X.append(fn(X))
        partes_y.append(y.copy())
    Xo = np.concatenate(partes_X)
    yo = np.concatenate(partes_y)
    perm = np.random.permutation(len(yo))
    return Xo[perm], yo[perm]


# ---------------------------------------------------------------------------
# Datos y partición
# ---------------------------------------------------------------------------

def _rel(ruta):
    """Ruta relativa al repositorio para imprimir y guardar; absoluta si está fuera de él."""
    ruta = Path(ruta)
    return str(ruta.relative_to(BASE_DIR)) if ruta.is_relative_to(BASE_DIR) else str(ruta)


def cargar_datos(processed_dir=PROCESSED_DIR, representacion=REPRESENTACION_POR_DEFECTO):
    """Carga X_padded, y_labels, lengths y sample_order de processed_dir (--processed; por defecto
    data/processed/) y comprueba que están alineados entre sí y con el diseño de rasgos que asume
    la aumentación."""
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
    # El primer campo del nombre SSSS_VVVV_RRRR es la clase: debe coincidir con y_labels.
    clases_nombre = np.array([int(n.split("_")[0]) for n in nombres])
    if not np.array_equal(clases_nombre, y):
        raise SystemExit("y_labels.npy no coincide con el campo de seña de sample_order: los "
                         "archivos procesados no están alineados")
    # D-08: el processed tiene 225 columnas (postura + manos) o 405 (+ cejas y labios); la
    # representación toma sus bloques y el espejado de la aumentación se ajusta a ellos
    # (src/representacion.py). Un ancho que no corresponde a ningún diseño conocido se rechaza.
    rostro = meta.get("rostro", {}).get("subconjunto", "ninguno")
    try:
        cols = columnas_representacion(representacion, rostro)
    except ValueError as exc:
        raise SystemExit(str(exc))
    esperado = N_TOTAL_FEATURES if rostro == "ninguno" else int(meta["rostro"]["columnas"][1])
    if X_padded.shape[2] != esperado:
        raise SystemExit(f"X_padded tiene {X_padded.shape[2]} rasgos por paso y metadata.json declara "
                         f"{esperado} (rostro: {rostro}); el processed no es coherente")
    return X_padded[:, :, cols], y, lengths, nombres


def cargar_particion(nombres, y):
    """Lee data/partitions/aleatoria_80_10_10.json y traduce los nombres de muestra de cada
    conjunto a índices de sample_order. Verifica que los conjuntos son disjuntos, cubren las N
    muestras menos las excluidas por la partición (duplicados del corpus, D-06), contienen todas
    las clases y coinciden con los índices guardados en el archivo."""
    if not ARCHIVO_PARTICION.exists():
        raise SystemExit(f"no existe {_rel(ARCHIVO_PARTICION)}: genérela con "
                         f"`python src/train_loso.py --dry-run` (actividad 1.e, D-02)")
    with open(ARCHIVO_PARTICION, encoding="utf-8") as f:
        particion = json.load(f)
    if particion.get("esquema") != "aleatoria_80_10_10":
        raise SystemExit(f"{_rel(ARCHIVO_PARTICION)}: esquema inesperado "
                         f"{particion.get('esquema')!r}")

    pos = {n: i for i, n in enumerate(nombres)}
    idx = {}
    for k in ("train", "val", "test"):
        conj = particion["conjuntos"][k]
        faltantes = [n for n in conj if n not in pos]
        if faltantes:
            raise SystemExit(f"la partición nombra muestras que no están en sample_order: "
                             f"{faltantes[:3]}")
        idx[k] = np.array([pos[n] for n in conj], dtype=np.int64)

    tr, va, te = (set(idx[k].tolist()) for k in ("train", "val", "test"))
    if tr & va or tr & te or va & te:
        raise SystemExit("partición inválida: conjuntos no disjuntos")
    excluidas = {pos[n] for n in particion.get("excluidas", {}) if n in pos}
    if (tr | va | te) != set(range(len(nombres))) - excluidas:
        raise SystemExit("partición inválida: los conjuntos no cubren exactamente las muestras "
                         "no excluidas")
    if excluidas:
        print(f"    muestras excluidas por la partición (D-06): "
              f"{', '.join(sorted(particion['excluidas']))}")
    n_clases = int(y.max()) + 1
    for k in idx:
        if len(set(y[idx[k]].tolist())) != n_clases:
            raise SystemExit(f"partición inválida: el conjunto {k} no contiene las {n_clases} clases")
    # Coherencia con los índices que la partición trae guardados (mismo sample_order).
    if "indices" in particion:
        for k in idx:
            if sorted(int(i) for i in particion["indices"][k]) != sorted(idx[k].tolist()):
                raise SystemExit(f"partición inválida: los índices guardados de {k} no coinciden "
                                 f"con los nombres traducidos por sample_order")
    return particion, idx


def remuestrear_todas(X_padded, lengths, seq_length):
    """Remuestrea cada secuencia (solo sus T fotogramas reales) a seq_length pasos."""
    N = X_padded.shape[0]
    X = np.zeros((N, seq_length, X_padded.shape[2]), dtype=np.float32)
    for i in range(N):
        T = int(lengths[i])
        X[i] = resample_sequence(X_padded[i, :T, :], seq_length)
    return X


def cargar_nombres_senas():
    """Nombres de las 50 señas (grammar_mapper.SIGN_VOCABULARY, verificado contra la Tabla 2 del
    artículo de LSC50) para etiquetar la matriz de confusión. Si falla, solo los ids."""
    try:
        from src.grammar_mapper import SIGN_VOCABULARY
        return {int(k): v for k, v in SIGN_VOCABULARY.items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Métricas básicas (las que §2.2.4 nombra: top-1, top-3, precisión/recall/F1 macro)
# ---------------------------------------------------------------------------

def metricas_basicas(probs, y_real):
    """A partir de las probabilidades softmax (N, n_clases) y las etiquetas reales (N,):
    top-1, top-3, precisión, recall y F1 macro. Sirve igual para validación y para prueba."""
    from sklearn.metrics import precision_recall_fscore_support
    n_clases = probs.shape[1]
    y_pred = np.argmax(probs, axis=1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_real, y_pred, labels=list(range(n_clases)), average="macro", zero_division=0)
    return {
        "top1": float(np.mean(y_pred == y_real)),
        "top3": float(np.mean(np.any(top3 == y_real[:, None], axis=1))),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
    }


# ---------------------------------------------------------------------------
# Entrenamiento de una semilla (selección solo con validación)
# ---------------------------------------------------------------------------

def _cargar_modelo(ruta):
    try:
        return tf.keras.models.load_model(str(ruta))
    except ValueError:
        return tf.keras.models.load_model(str(ruta), safe_mode=False)


def graficar_curvas(historial, mejor_epoca, titulo, ruta_png):
    """Curvas de aprendizaje reales (exactitud y pérdida, entrenamiento y validación)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epocas = len(historial["loss"])
    marca = "o" if epocas <= 5 else None   # con pocas épocas no hay línea que trazar
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (m, nombre) in zip(axes, [("accuracy", "Exactitud"), ("loss", "Pérdida")]):
        ax.plot(range(1, epocas + 1), historial[m], marker=marca, label="entrenamiento")
        ax.plot(range(1, epocas + 1), historial["val_" + m], marker=marca, label="validación")
        ax.axvline(mejor_epoca, color="gray", linestyle="--", linewidth=0.8,
                   label=f"mejor época ({mejor_epoca})")
        ax.set_title(nombre)
        ax.set_xlabel("época")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=130)
    plt.close(fig)


def entrenar_semilla(semilla, X_seq, y, idx, args, out_dir):
    """Entrena un modelo con la semilla dada: aumenta train, selecciona checkpoint y detiene
    con val_accuracy. NO toca el conjunto de prueba. Devuelve la ficha de la corrida."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t_inicio = time.perf_counter()
    n_clases = int(y.max()) + 1

    tf.keras.utils.set_random_seed(semilla)
    X_tr, y_tr = aumentar(X_seq[idx["train"]], y[idx["train"]], args.aug, args.representacion)
    X_tr_f = add_velocity(X_tr)
    X_val_f = add_velocity(X_seq[idx["val"]])
    y_tr_oh = tf.keras.utils.to_categorical(y_tr, n_clases)
    y_val_oh = tf.keras.utils.to_categorical(y[idx["val"]], n_clases)
    n_features = X_tr_f.shape[2]
    print(f"  entrenamiento: {len(y_tr)} secuencias (factor {args.aug}) x {X_tr_f.shape[1]} "
          f"pasos x {n_features} rasgos | validación: {len(idx['val'])}")

    # Hiperparámetros de la exploración de marzo (los que produjeron la arquitectura de §1.3.4).
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
    graficar_curvas(historial, mejor_epoca, f"Curvas de entrenamiento - semilla {semilla}",
                    out_dir / "curvas_entrenamiento.png")

    # El modelo de la corrida es el checkpoint de mejor val_accuracy. Sus métricas de
    # VALIDACIÓN son las únicas que intervienen en la selección entre semillas.
    del model
    tf.keras.backend.clear_session()
    escribir_procedencia(ruta_modelo, args.processed, args.representacion, args.seq_length,
                         "src/train_final_100.py", BASE_DIR)
    model = _cargar_modelo(ruta_modelo)
    val_loss, val_acc = model.evaluate(X_val_f, y_val_oh, verbose=0)
    # Métricas de validación de §2.2.4 (top-1, top-3, P/R/F1 macro) del checkpoint. Solo se
    # registran: el criterio de selección entre semillas es val_accuracy (= top-1).
    metricas_val = metricas_basicas(model.predict(X_val_f, verbose=0), y[idx["val"]])

    ficha = {
        "semilla": int(semilla),
        "modelo": _rel(ruta_modelo),
        "n_train_original": int(len(idx["train"])),
        "n_train_aumentado": int(len(y_tr)),
        "n_val": int(len(idx["val"])),
        "epocas_ejecutadas": int(epocas),
        "mejor_epoca": mejor_epoca,
        "val_accuracy_checkpoint": float(val_acc),
        "val_loss_checkpoint": float(val_loss),
        "metricas_validacion_checkpoint": metricas_val,
        "mejor_val_accuracy_historial": float(max(historial["val_accuracy"])),
        "tiempo_entrenamiento_s": round(tiempo_fit, 1),
        "tiempo_total_s": round(time.perf_counter() - t_inicio, 1),
    }
    with open(out_dir / "entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump(ficha, f, indent=1)
    print(f"  [semilla {semilla}] val_accuracy {val_acc:.4f} | val_loss {val_loss:.4f} | "
          f"val top-3 {metricas_val['top3']:.4f} | val F1 {metricas_val['f1_macro']:.4f} | "
          f"épocas {epocas} (mejor {mejor_epoca}) | {ficha['tiempo_total_s']:.0f} s")

    del model, X_tr, X_tr_f, y_tr, y_tr_oh
    tf.keras.backend.clear_session()
    return ficha, ruta_modelo


# ---------------------------------------------------------------------------
# Evaluación única sobre el conjunto de prueba
# ---------------------------------------------------------------------------

def evaluar_en_prueba(etiqueta, probs, y_te, muestras_prueba, nombres_senas, out_dir):
    """Métricas de un sistema sobre el conjunto de prueba a partir de sus probabilidades:
    top-1, top-3, precisión/recall/F1 macro, matriz de confusión (.json y .png),
    classification_report y predicciones por muestra."""
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n_clases = probs.shape[1]
    etiquetas = list(range(n_clases))
    y_pred = np.argmax(probs, axis=1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    basicas = metricas_basicas(probs, y_te)
    top1_acc, top3_acc = basicas["top1"], basicas["top3"]
    prec, rec, f1 = (basicas["precision_macro"], basicas["recall_macro"], basicas["f1_macro"])
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
        json.dump([{"muestra": muestras_prueba[i], "real": int(y_te[i]),
                    "predicha": int(y_pred[i]), "top3": [int(t) for t in top3[i]],
                    "prob_predicha": float(probs[i, y_pred[i]])}
                   for i in range(len(y_te))], f, indent=1)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title(f"Matriz de confusión - {etiqueta} (top-1 {top1_acc:.1%}, n={len(y_te)})",
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

    metricas = {
        "sistema": etiqueta,
        "n_test": int(len(y_te)),
        "top1": top1_acc,
        "top3": top3_acc,
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "n_errores": int(np.sum(y_pred != y_te)),
        "archivos": {k: _rel(out_dir / k) for k in (
            "metricas.json", "classification_report.json", "matriz_confusion.json",
            "matriz_confusion.png", "predicciones.json")},
    }
    with open(out_dir / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=1, ensure_ascii=False)
    print(f"  [{etiqueta}] top-1 {top1_acc:.4f} | top-3 {top3_acc:.4f} | P {prec:.4f} | "
          f"R {rec:.4f} | F1 {f1:.4f} | errores {metricas['n_errores']}/{len(y_te)}")
    return metricas


def info_entorno():
    ent = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "plataforma": platform.platform(),
        "maquina": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "gpus_visibles": [d.name for d in tf.config.list_physical_devices("GPU")],
    }
    try:
        import keras
        import sklearn
        ent["keras"] = keras.__version__
        ent["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    return ent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parsear_semillas(texto, base):
    """'3' -> [base, base+1, base+2]; '2026,7,99' -> esa lista."""
    texto = texto.strip()
    if "," in texto:
        semillas = [int(t) for t in texto.split(",") if t.strip() != ""]
    else:
        n = int(texto)
        if n < 1:
            raise SystemExit("--semillas debe ser >= 1")
        semillas = [base + i for i in range(n)]
    if not semillas or len(set(semillas)) != len(semillas):
        raise SystemExit("--semillas: lista vacía o con repetidos")
    return semillas


def main():
    parser = argparse.ArgumentParser(
        description="Entrenador canónico LSC50 (esquema aleatorio 80/10/10 de data/partitions/), "
                    "CNN+BiLSTM+atención, selección solo con validación, evaluación única en "
                    "prueba, en CPU.")
    parser.add_argument("--epochs", type=int, default=200, help="épocas máximas (defecto 200)")
    parser.add_argument("--aug", type=int, default=10,
                        help="factor de aumento: 1 = sin aumentar, 10 = refined_augmentation "
                             "completa (defecto 10)")
    parser.add_argument("--semillas", type=str, default="1",
                        help="número N de semillas de entrenamiento (base, base+1, ...) o lista "
                             "'a,b,c'. Con N>1 se reporta además el ensemble (promedio de "
                             "softmax), como tal (defecto 1)")
    parser.add_argument("--semilla-base", type=int, default=SEMILLA_BASE,
                        help=f"primera semilla de entrenamiento cuando --semillas es un número "
                             f"(defecto {SEMILLA_BASE}); no es la semilla de partición")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=25,
                        help="paciencia del early stopping sobre val_accuracy (defecto 25)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seq-length", type=int, default=30,
                        help="pasos del remuestreo temporal (defecto 30)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR),
                        help="directorio de salida (defecto models/canonico/)")
    parser.add_argument("--processed", type=str, default=str(PROCESSED_DIR),
                        help="directorio con X_padded.npy, y_labels.npy, lengths.npy y metadata.json "
                             "(defecto data/processed; D-05: la salida de preprocess_lsc50.py --salida)")
    parser.add_argument("--representacion", choices=list(REPRESENTACIONES), default=REPRESENTACION_POR_DEFECTO,
                        help="D-08: bloques de entrada: manos (126), manos_postura (225, defecto) o "
                             "manos_postura_rostro (405; requiere un processed con --rostro cejas_labios)")
    parser.add_argument("--dry-run", action="store_true",
                        help="solo carga datos y partición, verifica e imprime conteos")
    args = parser.parse_args()

    if not 1 <= args.aug <= 10:
        raise SystemExit("--aug debe estar entre 1 y 10")
    semillas = parsear_semillas(args.semillas, args.semilla_base)
    out_dir = Path(args.out).expanduser().resolve()
    # Salvaguardas: nunca dentro de data/ (data/dataset/ incluido), ni directamente en models/
    # (donde viven models/best_model.keras y los modelos de marzo), ni dentro de las salidas de
    # otras piezas: models/loso/ (corridas LOSO de train_loso.py) y models/mediapipe/ (.task).
    prohibidos = [BASE_DIR / "data", BASE_DIR / "models" / "loso",
                  BASE_DIR / "models" / "mediapipe"]
    if out_dir == (BASE_DIR / "models") or any(out_dir.is_relative_to(p) for p in prohibidos):
        raise SystemExit(f"--out no puede ser {out_dir}: use un subdirectorio propio "
                         f"(por defecto {_rel(DEFAULT_OUT_DIR)})")

    print("=" * 72)
    print("  LSC50 - Entrenador canónico (esquema aleatorio 80/10/10 de data/partitions/)")
    print("=" * 72)

    # 1. Datos y partición
    processed_dir = Path(args.processed).expanduser().resolve()
    X_padded, y, lengths, nombres = cargar_datos(processed_dir, args.representacion)
    n_clases = int(y.max()) + 1
    print(f"\n[1] Datos ({_rel(processed_dir)}, representación {args.representacion}): "
          f"{len(nombres)} muestras, {n_clases} clases; "
          f"X_padded {X_padded.shape}, "
          f"longitudes {int(lengths.min())}-{int(lengths.max())} fotogramas")
    particion, idx = cargar_particion(nombres, y)
    print(f"\n[2] Partición: {_rel(ARCHIVO_PARTICION)} (semilla de partición "
          f"{particion.get('semilla')}, creada {particion.get('fecha_creacion', '?')})")
    for k in ("train", "val", "test"):
        por_clase = np.bincount(y[idx[k]], minlength=n_clases)
        print(f"    {k:5s}: {len(idx[k]):4d} muestras | por clase min {por_clase.min()} "
              f"max {por_clase.max()}")
    print(f"    uso del test declarado en la partición: {particion.get('uso_del_test', '?')}")

    if args.dry_run:
        print("\n[dry-run] Datos y partición verificados. No se entrena nada.")
        return

    # 2. Plan
    print(f"\n[3] Plan: semillas de entrenamiento {semillas}"
          + (" (+ ensemble, promedio de softmax)" if len(semillas) > 1 else ""))
    print(f"    epochs max {args.epochs} | aug x{args.aug} | batch {args.batch_size} | "
          f"paciencia {args.patience} | dropout {args.dropout} | seq_length {args.seq_length}")
    print(f"    arquitectura: build_model (CNN+BiLSTM+atención), entrada ({args.seq_length}, "
          f"{2 * X_padded.shape[2]}) = {X_padded.shape[2]} coordenadas + velocidades")
    print(f"    selección: checkpoint y early stopping sobre val_accuracy; entre semillas, la de "
          f"mayor val_accuracy; el test se evalúa una sola vez al final")
    print(f"    salida: {out_dir}")

    # CPU únicamente (§1.4, §2.5.2): en el equipo del proyecto no hay GPU; esto lo garantiza.
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    print(f"    TensorFlow {tf.__version__}; GPUs visibles: {tf.config.list_physical_devices('GPU')}")

    X_seq = remuestrear_todas(X_padded, lengths, args.seq_length)
    del X_padded
    print(f"    secuencias remuestreadas a {args.seq_length} pasos: {X_seq.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ruta_resultados = out_dir / "resultados.json"
    resultados = {
        "script": "src/train_final_100.py",
        "esquema": "aleatoria_80_10_10",
        "processed_dir": _rel(processed_dir),
        "representacion": args.representacion,
        "particion": _rel(ARCHIVO_PARTICION),
        "semilla_particion": particion.get("semilla"),
        "configuracion": {
            "arquitectura": "CNN+BiLSTM+atencion (build_model)",
            "seq_length": args.seq_length, "n_features": int(2 * X_seq.shape[2]),
            "velocidades": True, "epochs_max": args.epochs, "aug": args.aug,
            "batch_size": args.batch_size, "patience": args.patience, "dropout": args.dropout,
            "semillas_entrenamiento": semillas, "monitor": "val_accuracy",
        },
        "semillas": [],
        "seleccion": None,
        "prueba": {},
        "entorno": info_entorno(),
    }

    def guardar_resultados():
        with open(ruta_resultados, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=1, ensure_ascii=False)

    guardar_resultados()

    # 3. Entrenamiento por semilla (el test no se toca)
    rutas_modelo = {}
    for s in semillas:
        print(f"\n{'=' * 72}\n  SEMILLA {s}\n{'=' * 72}")
        ficha, ruta = entrenar_semilla(s, X_seq, y, idx, args, out_dir / f"semilla_{s}")
        rutas_modelo[s] = ruta
        resultados["semillas"].append(ficha)
        guardar_resultados()

    # 4. Selección entre semillas SOLO con validación (empate: menor val_loss; luego el orden
    #    en que se pidieron las semillas). Se fija ANTES de tocar el conjunto de prueba.
    fichas = resultados["semillas"]
    orden = sorted(range(len(fichas)), key=lambda i: (
        -fichas[i]["val_accuracy_checkpoint"], fichas[i]["val_loss_checkpoint"], i))
    sel = fichas[orden[0]]
    ruta_canonico = out_dir / "modelo_canonico.keras"
    shutil.copy2(rutas_modelo[sel["semilla"]], ruta_canonico)
    escribir_procedencia(ruta_canonico, processed_dir, args.representacion, args.seq_length,
                         "src/train_final_100.py", BASE_DIR)
    resultados["seleccion"] = {
        "criterio": "mayor val_accuracy del checkpoint; empate: menor val_loss; luego orden de "
                    "las semillas. Sin intervención del conjunto de prueba.",
        "semilla": sel["semilla"],
        "val_accuracy_checkpoint": sel["val_accuracy_checkpoint"],
        "metricas_validacion_checkpoint": sel["metricas_validacion_checkpoint"],
        "modelo": _rel(ruta_canonico),
        "copiado_de": sel["modelo"],
    }
    guardar_resultados()
    print(f"\n[4] Seleccionado por validación: semilla {sel['semilla']} "
          f"(val_accuracy {sel['val_accuracy_checkpoint']:.4f}) -> {_rel(ruta_canonico)}")

    # 5. Evaluación ÚNICA sobre el conjunto de prueba: una pasada por modelo, y con ellas se
    #    evalúan los sistemas declarados de antemano (modelo seleccionado y, si hay varias
    #    semillas, el ensemble). Nada se elige a partir de estas cifras.
    print(f"\n[5] Evaluación única en prueba ({len(idx['test'])} muestras)")
    X_te_f = add_velocity(X_seq[idx["test"]])
    y_te = y[idx["test"]]
    muestras_prueba = list(particion["conjuntos"]["test"])
    nombres_senas = cargar_nombres_senas()
    probs = {}
    for s in semillas:
        model = _cargar_modelo(rutas_modelo[s])
        probs[s] = model.predict(X_te_f, verbose=0)
        del model
        tf.keras.backend.clear_session()

    resultados["prueba"]["modelo_seleccionado"] = evaluar_en_prueba(
        f"modelo seleccionado (semilla {sel['semilla']})", probs[sel["semilla"]], y_te,
        muestras_prueba, nombres_senas, out_dir / "evaluacion_modelo_seleccionado")
    if len(semillas) > 1:
        probs_ens = np.mean([probs[s] for s in semillas], axis=0)
        resultados["prueba"]["ensemble"] = evaluar_en_prueba(
            f"ensemble de {len(semillas)} semillas (promedio de softmax)", probs_ens, y_te,
            muestras_prueba, nombres_senas, out_dir / "evaluacion_ensemble")
        # No se reporta el top-1 de prueba de cada semilla por separado: esas cifras no
        # intervienen en ninguna selección y publicarlas invita a escoger «la mejor semilla»
        # mirando el test (hallazgo B). Entre semillas solo se compara la validación.
    guardar_resultados()

    # 6. Resumen
    print(f"\n{'=' * 72}\n  RESUMEN (esquema aleatorio 80/10/10; el anteproyecto lo declara "
          f"optimista: se reporta como contraste del LOSO)\n{'=' * 72}")
    print(f"  {'sistema':44s} {'top-1':>7} {'top-3':>7} {'P':>7} {'R':>7} {'F1':>7}")
    for clave in ("modelo_seleccionado", "ensemble"):
        m = resultados["prueba"].get(clave)
        if m:
            print(f"  {m['sistema'][:44]:44s} {m['top1']:7.4f} {m['top3']:7.4f} "
                  f"{m['precision_macro']:7.4f} {m['recall_macro']:7.4f} {m['f1_macro']:7.4f}")
    print(f"\n  Resultados: {ruta_resultados}")


if __name__ == "__main__":
    main()
