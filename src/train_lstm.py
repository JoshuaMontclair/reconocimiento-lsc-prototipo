"""
Entrenamiento del modelo LSTM para reconocimiento de LSC.

Incluye:
  - Arquitectura LSTM con BatchNorm y Dropout
  - Data augmentation (espejado, ruido, variación temporal, escalado)
  - Callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
  - Evaluación (accuracy, F1-score, matriz de confusión)

Uso:
    python src/train_lstm.py
    python src/train_lstm.py --epochs 200 --batch-size 64 --patience 20
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reducir logs de TF

import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "dataset"
MODELS_DIR = BASE_DIR / "models"

# Estructura de features (body + left_hand + right_hand)
N_BODY = 33
N_HAND = 21
COORDS = 3
N_BODY_FEATURES = N_BODY * COORDS       # 99
N_HAND_FEATURES = N_HAND * COORDS       # 63
N_TOTAL_FEATURES = N_BODY_FEATURES + 2 * N_HAND_FEATURES  # 225

# Pares simétricos de body landmarks (izq, der)
BODY_SYMMETRIC_PAIRS = [
    (1, 4),    # eye inner
    (2, 5),    # eye
    (3, 6),    # eye outer
    (7, 8),    # ear
    (9, 10),   # mouth
    (11, 12),  # shoulder
    (13, 14),  # elbow
    (15, 16),  # wrist
    (17, 18),  # pinky
    (19, 20),  # index
    (21, 22),  # thumb
    (23, 24),  # hip
    (25, 26),  # knee
    (27, 28),  # ankle
    (29, 30),  # heel
    (31, 32),  # foot index
]


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

def augment_mirror(X: np.ndarray) -> np.ndarray:
    """Espejado horizontal: invierte x coords y swapea izquierda/derecha.

    - Negar coordenadas x (ya centradas en 0 por normalización)
    - Intercambiar landmarks simétricos del body
    - Intercambiar left_hand ↔ right_hand
    """
    X_mirror = X.copy()
    N, T, F = X_mirror.shape

    # 1. Negar todas las coordenadas x (cada 3 valores: idx 0, 3, 6, ...)
    for i in range(0, F, COORDS):
        X_mirror[:, :, i] *= -1

    # 2. Swap pares simétricos del body
    for left_lm, right_lm in BODY_SYMMETRIC_PAIRS:
        l_start = left_lm * COORDS
        r_start = right_lm * COORDS
        X_mirror[:, :, l_start:l_start + COORDS], X_mirror[:, :, r_start:r_start + COORDS] = (
            X_mirror[:, :, r_start:r_start + COORDS].copy(),
            X_mirror[:, :, l_start:l_start + COORDS].copy(),
        )

    # 3. Swap left_hand ↔ right_hand
    lh_start = N_BODY_FEATURES
    lh_end = N_BODY_FEATURES + N_HAND_FEATURES
    rh_start = lh_end
    rh_end = rh_start + N_HAND_FEATURES

    X_mirror[:, :, lh_start:lh_end], X_mirror[:, :, rh_start:rh_end] = (
        X_mirror[:, :, rh_start:rh_end].copy(),
        X_mirror[:, :, lh_start:lh_end].copy(),
    )

    return X_mirror


def augment_noise(X: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """Añade ruido gaussiano a los landmarks."""
    noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
    return X + noise


def augment_temporal(X: np.ndarray, speed_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """Variación temporal: speed up / slow down por interpolación.

    Resamplea cada secuencia con un factor de velocidad aleatorio.
    """
    N, T, F = X.shape
    X_aug = np.zeros_like(X)

    for i in range(N):
        speed = np.random.uniform(*speed_range)
        # Número de frames originales a usar
        n_orig = int(T * speed)
        n_orig = max(2, min(n_orig, T * 3))  # limitar

        # Crear secuencia temporal modificada
        orig_indices = np.linspace(0, T - 1, n_orig)
        new_indices = np.linspace(0, T - 1, T)

        # Si speed > 1 (más rápido), usamos menos frames originales
        # Si speed < 1 (más lento), usamos más frames originales
        sample_indices = np.linspace(0, n_orig - 1, T)

        for f in range(F):
            orig_values = np.interp(orig_indices, np.arange(T), X[i, :, f])
            X_aug[i, :, f] = np.interp(sample_indices, np.arange(n_orig), orig_values)

    return X_aug


def augment_scale(X: np.ndarray, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """Pequeñas variaciones de escala en las coordenadas."""
    N = X.shape[0]
    scales = np.random.uniform(*scale_range, size=(N, 1, 1)).astype(np.float32)
    return X * scales


def apply_augmentation(X: np.ndarray, y: np.ndarray,
                       use_mirror: bool = True,
                       use_noise: bool = True,
                       use_temporal: bool = True,
                       use_scale: bool = True,
                       noise_std: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Aplica todas las augmentaciones y concatena con los datos originales."""
    augmented_X = [X]
    augmented_y = [y]

    if use_mirror:
        augmented_X.append(augment_mirror(X))
        augmented_y.append(y.copy())

    if use_noise:
        augmented_X.append(augment_noise(X, noise_std))
        augmented_y.append(y.copy())

    if use_temporal:
        augmented_X.append(augment_temporal(X))
        augmented_y.append(y.copy())

    if use_scale:
        augmented_X.append(augment_scale(X))
        augmented_y.append(y.copy())

    X_out = np.concatenate(augmented_X, axis=0)
    y_out = np.concatenate(augmented_y, axis=0)

    # Shuffle
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ---------------------------------------------------------------------------
# Modelo LSTM
# ---------------------------------------------------------------------------

def build_model(n_frames: int, n_features: int, n_classes: int,
                lstm_units: tuple = (128, 128, 64),
                dropout_rate: float = 0.3,
                learning_rate: float = 0.001) -> tf.keras.Model:
    """Construye el modelo LSTM con BatchNorm y Dropout."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_frames, n_features)),

        tf.keras.layers.LSTM(lstm_units[0], return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.LSTM(lstm_units[1], return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.LSTM(lstm_units[2], return_sequences=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray,
                   output_dir: Path) -> dict:
    """Evalúa el modelo y genera reportes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Predicciones
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Métricas
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"\n  Resultados en test:")
    print(f"    Accuracy:    {test_acc:.4f}")
    print(f"    F1 (macro):  {f1_macro:.4f}")
    print(f"    F1 (weighted): {f1_weighted:.4f}")
    print(f"    Loss:        {test_loss:.4f}")

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)
    print(f"\n  Classification Report:\n{report_text}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    n_classes = len(np.unique(y_test))

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Matriz de Confusion — LSC50 LSTM", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xlabel("Prediccion", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)

    if n_classes <= 50:
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(range(n_classes), fontsize=6, rotation=90)
        ax.set_yticklabels(range(n_classes), fontsize=6)

    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Matriz de confusion guardada: {cm_path}")

    return {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def plot_training_history(history, output_dir: Path) -> None:
    """Grafica las curvas de entrenamiento."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Curvas de Entrenamiento — LSTM LSC50", fontsize=13)
    fig.tight_layout()
    hist_path = output_dir / "training_history.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"  Curvas de entrenamiento guardadas: {hist_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento LSTM para reconocimiento de LSC"
    )
    parser.add_argument("--epochs", type=int, default=150,
                        help="Epochs maximos (default: 150)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Patience para EarlyStopping (default: 20)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate inicial (default: 0.001)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default: 0.3)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Desactivar data augmentation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria (default: 42)")
    args = parser.parse_args()

    # Reproducibilidad
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("=" * 60)
    print("  LSC50 — Entrenamiento LSTM")
    print("=" * 60)

    # --- Cargar datos ---
    print("\n[1] Cargando dataset...")
    X_train = np.load(DATASET_DIR / "X_train.npy")
    y_train = np.load(DATASET_DIR / "y_train.npy")
    X_val = np.load(DATASET_DIR / "X_val.npy")
    y_val = np.load(DATASET_DIR / "y_val.npy")
    X_test = np.load(DATASET_DIR / "X_test.npy")
    y_test = np.load(DATASET_DIR / "y_test.npy")

    n_frames = X_train.shape[1]
    n_features = X_train.shape[2]
    n_classes = len(np.unique(y_train))

    print(f"  Train: {X_train.shape} — {len(y_train)} muestras")
    print(f"  Val:   {X_val.shape} — {len(y_val)} muestras")
    print(f"  Test:  {X_test.shape} — {len(y_test)} muestras")
    print(f"  Clases: {n_classes}, Frames: {n_frames}, Features: {n_features}")

    # --- Data Augmentation ---
    if not args.no_augment:
        print("\n[2] Aplicando data augmentation...")
        X_train_aug, y_train_aug = apply_augmentation(X_train, y_train)
        print(f"  Train original: {X_train.shape[0]} muestras")
        print(f"  Train aumentado: {X_train_aug.shape[0]} muestras "
              f"({X_train_aug.shape[0] / X_train.shape[0]:.1f}x)")
    else:
        print("\n[2] Data augmentation desactivado")
        X_train_aug, y_train_aug = X_train, y_train

    # --- Construir modelo ---
    print("\n[3] Construyendo modelo...")
    model = build_model(n_frames, n_features, n_classes,
                        dropout_rate=args.dropout,
                        learning_rate=args.lr)
    model.summary()

    # --- Callbacks ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # --- Entrenamiento ---
    print(f"\n[4] Entrenando (max {args.epochs} epochs, batch={args.batch_size}, "
          f"patience={args.patience})...")
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    actual_epochs = len(history.history["loss"])
    print(f"\n  Entrenamiento finalizado en {actual_epochs} epochs")

    # --- Evaluación ---
    print("\n[5] Evaluando modelo...")
    results = evaluate_model(model, X_test, y_test, MODELS_DIR)
    plot_training_history(history, MODELS_DIR)

    # --- Guardar resultados ---
    results_summary = {
        "model": "LSTM 3-layer (128-128-64)",
        "n_frames": n_frames,
        "n_features": n_features,
        "n_classes": n_classes,
        "train_samples_original": int(X_train.shape[0]),
        "train_samples_augmented": int(X_train_aug.shape[0]),
        "augmentation": not args.no_augment,
        "epochs_trained": actual_epochs,
        "epochs_max": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "dropout": args.dropout,
        "patience": args.patience,
        "test_accuracy": results["test_accuracy"],
        "test_loss": results["test_loss"],
        "f1_macro": results["f1_macro"],
        "f1_weighted": results["f1_weighted"],
    }

    results_path = MODELS_DIR / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultados guardados: {results_path}")

    # --- Resumen final ---
    acc = results["test_accuracy"]
    target = 0.85
    status = "CUMPLE" if acc >= target else "NO CUMPLE"
    print(f"\n{'=' * 60}")
    print(f"  Accuracy: {acc:.4f} — Umbral minimo: {target:.0%} — {status}")
    print(f"  Modelo guardado: {MODELS_DIR / 'best_model.keras'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
