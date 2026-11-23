"""
Advanced LSC50 training pipeline — targeting 98%+ accuracy.

Key improvements over train_lstm.py:
  1. Velocity features (225 → 450 features)
  2. 15x data augmentation with diverse transformations
  3. CNN + BiLSTM + Attention architecture
  4. Cosine decay learning rate
  5. Label smoothing (0.1)
  6. Longer training (300 epochs, patience 40)

Uso:
    python src/train_v2.py
    python src/train_v2.py --seq-length 45 --epochs 300
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

sys.path.insert(0, str(BASE_DIR))
from src.train_lstm import (
    BODY_SYMMETRIC_PAIRS,
    COORDS,
    N_BODY,
    N_BODY_FEATURES,
    N_HAND,
    N_HAND_FEATURES,
    N_TOTAL_FEATURES,
    augment_mirror,
    augment_noise,
    augment_scale,
    augment_temporal,
)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def add_velocity(X: np.ndarray) -> np.ndarray:
    """Add frame-to-frame velocity as additional features. 225 → 450."""
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, vel], axis=-1)


def add_acceleration(X: np.ndarray) -> np.ndarray:
    """Add acceleration (second derivative). 450 → 675."""
    # X already has [position, velocity]
    n_orig = X.shape[2] // 2  # half is position, half is velocity
    vel = X[:, :, n_orig:]  # velocity portion
    acc = np.zeros_like(vel)
    acc[:, 1:, :] = vel[:, 1:, :] - vel[:, :-1, :]
    return np.concatenate([X, acc], axis=-1)


# ---------------------------------------------------------------------------
# New Augmentation Functions
# ---------------------------------------------------------------------------

def augment_landmark_dropout(X: np.ndarray, drop_rate: float = 0.1) -> np.ndarray:
    """Randomly zero out some landmarks to improve robustness."""
    X_aug = X.copy()
    N, T, F = X_aug.shape
    n_landmarks = F // COORDS
    for i in range(N):
        n_drop = max(1, int(n_landmarks * drop_rate))
        drop_idx = np.random.choice(n_landmarks, n_drop, replace=False)
        for idx in drop_idx:
            X_aug[i, :, idx * COORDS:(idx + 1) * COORDS] = 0.0
    return X_aug


def augment_frame_shift(X: np.ndarray, max_shift: int = 3) -> np.ndarray:
    """Shift frames by a random amount (circular)."""
    X_aug = X.copy()
    for i in range(X_aug.shape[0]):
        shift = np.random.randint(-max_shift, max_shift + 1)
        X_aug[i] = np.roll(X_aug[i], shift, axis=0)
    return X_aug


def augment_rotation_xy(X: np.ndarray, max_angle_deg: float = 10.0) -> np.ndarray:
    """Small rotation in the XY plane."""
    X_aug = X.copy()
    N, T, F = X_aug.shape
    n_landmarks = F // COORDS
    for i in range(N):
        angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
        c, s = np.cos(angle), np.sin(angle)
        for lm in range(n_landmarks):
            xi = lm * COORDS
            yi = xi + 1
            x = X_aug[i, :, xi].copy()
            y = X_aug[i, :, yi].copy()
            X_aug[i, :, xi] = x * c - y * s
            X_aug[i, :, yi] = x * s + y * c
    return X_aug


def augment_mixup_intraclass(X: np.ndarray, y: np.ndarray,
                              alpha: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Mixup between samples of the same class."""
    X_mix = X.copy()
    classes = np.unique(y)
    for cls in classes:
        idx = np.where(y == cls)[0]
        if len(idx) < 2:
            continue
        for i in idx:
            j = np.random.choice(idx)
            lam = np.random.beta(alpha, alpha)
            X_mix[i] = lam * X[i] + (1 - lam) * X[j]
    return X_mix, y.copy()


def heavy_augmentation(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """15x augmentation with diverse transformations on position data."""
    all_X = [X]
    all_y = [y]

    def add(X_new):
        all_X.append(X_new)
        all_y.append(y.copy())

    add(augment_mirror(X))                                       # 2x
    add(augment_noise(X, 0.005))                                 # 3x
    add(augment_noise(X, 0.015))                                 # 4x
    add(augment_noise(X, 0.025))                                 # 5x
    add(augment_temporal(X, (0.85, 0.95)))                       # 6x
    add(augment_temporal(X, (1.05, 1.15)))                       # 7x
    add(augment_scale(X, (0.85, 1.15)))                          # 8x
    add(augment_noise(augment_mirror(X), 0.01))                  # 9x
    add(augment_temporal(augment_mirror(X), (0.9, 1.1)))         # 10x
    add(augment_landmark_dropout(X, 0.1))                        # 11x
    add(augment_frame_shift(X, 3))                               # 12x
    add(augment_rotation_xy(X, 10))                              # 13x
    add(augment_scale(augment_mirror(X), (0.9, 1.1)))            # 14x

    # 15x: intra-class mixup
    X_mix, y_mix = augment_mixup_intraclass(X, y)
    all_X.append(X_mix)
    all_y.append(y_mix)

    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ---------------------------------------------------------------------------
# Model Architecture: CNN + BiLSTM + Attention
# ---------------------------------------------------------------------------

def build_cnn_bilstm(n_frames: int, n_features: int, n_classes: int,
                     dropout: float = 0.3) -> tf.keras.Model:
    """CNN + Bidirectional LSTM with attention pooling."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # 1D CNN for local feature extraction
    x = tf.keras.layers.Conv1D(128, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    # BiLSTM layers
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Global pooling branches (replaces Lambda for serialization compatibility)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

    # Attention-weighted pooling via dot product
    att = tf.keras.layers.Dense(1)(x)           # (batch, T, 1)
    att = tf.keras.layers.Softmax(axis=1)(att)  # (batch, T, 1)
    ctx = tf.keras.layers.Multiply()([x, att])  # (batch, T, 128)
    ctx = tf.keras.layers.GlobalAveragePooling1D()(ctx)  # sum ≈ avg * T
    # Scale to approximate sum (GlobalAvg divides by T)
    # Not critical since Dense layers after will learn the right scale

    # Merge all representations
    x = tf.keras.layers.Concatenate()([ctx, avg_pool, max_pool])

    # Classification head
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_test(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray,
                  output_dir: Path) -> dict:
    """Evaluate model and generate reports."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = float(np.mean(y_pred == y_test))
    f1_mac = float(f1_score(y_test, y_pred, average="macro"))
    f1_wt = float(f1_score(y_test, y_pred, average="weighted"))

    print(f"\n  Results on test set:")
    print(f"    Accuracy:      {acc:.4f}")
    print(f"    F1 (macro):    {f1_mac:.4f}")
    print(f"    F1 (weighted): {f1_wt:.4f}")

    report = classification_report(y_test, y_pred)
    print(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    n_cls = len(np.unique(y_test))
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix — LSC50 v2")
    fig.colorbar(ax.images[0], ax=ax, shrink=0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if n_cls <= 50:
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        ax.set_xticklabels(range(n_cls), fontsize=6, rotation=90)
        ax.set_yticklabels(range(n_cls), fontsize=6)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_v2.png", dpi=150)
    plt.close(fig)

    # Training history plot would be done separately
    return {"test_accuracy": acc, "f1_macro": f1_mac, "f1_weighted": f1_wt}


def plot_history(history, output_dir: Path) -> None:
    """Plot training curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training Curves — LSC50 v2 (CNN+BiLSTM+Attention)")
    fig.tight_layout()
    fig.savefig(output_dir / "training_history_v2.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a sequence to target_length frames via interpolation."""
    T, F = sequence.shape
    if T == target_length:
        return sequence
    new_idx = np.linspace(0, T - 1, target_length)
    result = np.zeros((target_length, F), dtype=np.float32)
    for f in range(F):
        result[:, f] = np.interp(new_idx, np.arange(T), sequence[:, f])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Advanced LSC50 Training v2")
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-velocity", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("=" * 60)
    print("  LSC50 — Advanced Training v2")
    print("  CNN + BiLSTM + Attention | Velocity | 15x Augmentation")
    print("=" * 60)

    # 1. Load processed data
    print("\n[1] Loading processed data...")
    X_padded = np.load(PROCESSED_DIR / "X_padded.npy")
    y_labels = np.load(PROCESSED_DIR / "y_labels.npy")
    lengths = np.load(PROCESSED_DIR / "lengths.npy")
    print(f"  Loaded: {len(y_labels)} samples, {len(np.unique(y_labels))} classes")

    # 2. Resample to target sequence length
    print(f"\n[2] Resampling to {args.seq_length} frames...")
    N = len(y_labels)
    X_all = np.zeros((N, args.seq_length, N_TOTAL_FEATURES), dtype=np.float32)
    for i in range(N):
        T = int(lengths[i])
        X_all[i] = resample_sequence(X_padded[i, :T, :], args.seq_length)
    print(f"  Shape: {X_all.shape}")

    # 3. Stratified split 80/10/10
    print("\n[3] Splitting 80/10/10 (stratified)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_labels, test_size=0.2, stratify=y_labels, random_state=args.seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=args.seed)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Samples/class: train={X_train.shape[0] // 50}, "
          f"val={X_val.shape[0] // 50}, test={X_test.shape[0] // 50}")

    # 4. Heavy augmentation (on position data)
    print("\n[4] Heavy augmentation (15x on position data)...")
    X_train_aug, y_train_aug = heavy_augmentation(X_train, y_train)
    print(f"  Augmented: {X_train_aug.shape[0]} samples "
          f"({X_train_aug.shape[0] / len(X_train):.0f}x)")

    # 5. Feature engineering
    if not args.no_velocity:
        print("\n[5] Adding velocity features...")
        X_train_aug = add_velocity(X_train_aug)
        X_val_feat = add_velocity(X_val)
        X_test_feat = add_velocity(X_test)
        print(f"  Features: {X_train_aug.shape[2]} (225 position + 225 velocity)")
    else:
        X_val_feat = X_val
        X_test_feat = X_test
        print("\n[5] Velocity features disabled.")

    n_frames = X_train_aug.shape[1]
    n_features = X_train_aug.shape[2]
    n_classes = len(np.unique(y_train))

    # 6. Build model
    print(f"\n[6] Building CNN + BiLSTM + Attention model...")
    model = build_cnn_bilstm(n_frames, n_features, n_classes, dropout=args.dropout)
    model.summary()

    # Convert to one-hot for label smoothing
    y_train_oh = tf.keras.utils.to_categorical(y_train_aug, n_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes)

    # Learning rate schedule: cosine decay
    steps_per_epoch = len(y_train_aug) // args.batch_size + 1
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=args.epochs * steps_per_epoch,
        alpha=1e-6,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    # 7. Callbacks
    model_path = str(MODELS_DIR / "best_model_v2.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=args.patience,
            restore_best_weights=True, verbose=1, mode="max"),
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy",
            save_best_only=True, verbose=1, mode="max"),
    ]

    # 8. Train
    print(f"\n[7] Training (max {args.epochs} epochs, batch={args.batch_size}, "
          f"patience={args.patience})...")
    history = model.fit(
        X_train_aug, y_train_oh,
        validation_data=(X_val_feat, y_val_oh),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # 9. Plot training curves
    plot_history(history, MODELS_DIR)

    # 10. Evaluate on test
    print("\n[8] Evaluation on test set...")
    try:
        model = tf.keras.models.load_model(model_path)
    except ValueError:
        model = tf.keras.models.load_model(model_path, safe_mode=False)
    results = evaluate_test(model, X_test_feat, y_test, MODELS_DIR)

    # Save results
    results_data = {
        "model": "CNN+BiLSTM+Attention",
        "seq_length": args.seq_length,
        "n_features": n_features,
        "n_classes": n_classes,
        "augmentation": "15x",
        "velocity_features": not args.no_velocity,
        "epochs_max": args.epochs,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "label_smoothing": 0.1,
        **results,
    }
    with open(MODELS_DIR / "training_results_v2.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Also copy as best_model.keras if it's the best we have
    test_acc = results["test_accuracy"]
    if test_acc >= 0.90:
        import shutil
        shutil.copy2(model_path, str(MODELS_DIR / "best_model.keras"))
        print(f"  Copied to best_model.keras (acc={test_acc:.1%})")

    print(f"\n{'=' * 60}")
    print(f"  Test Accuracy: {test_acc:.1%}")
    print(f"  Target: 98%")
    print(f"  {'PASSED!' if test_acc >= 0.98 else 'Need more work...'}")
    print(f"{'=' * 60}")

    return test_acc


if __name__ == "__main__":
    main()
