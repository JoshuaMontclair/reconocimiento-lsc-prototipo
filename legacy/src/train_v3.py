"""
Ensemble + TTA training for LSC50 — targeting 98%+ accuracy.

Strategy:
  - Train 5 CNN+BiLSTM models with different random seeds
  - Each model gets different augmented data → learns different features
  - Ensemble: average softmax predictions across models
  - TTA: augment each test sample, predict, average
  - Combined ensemble + TTA for maximum accuracy

Uso:
    python src/train_v3.py
    python src/train_v3.py --n-models 3 --epochs 200
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

sys.path.insert(0, str(BASE_DIR))
from src.train_lstm import (
    COORDS,
    N_BODY_FEATURES,
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
    """Add frame-to-frame velocity. 225 → 450 features."""
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, vel], axis=-1)


# ---------------------------------------------------------------------------
# Refined Augmentation (10x — no exotic transforms)
# ---------------------------------------------------------------------------

def refined_augmentation(X: np.ndarray, y: np.ndarray) -> tuple:
    """10x augmentation with only proven transforms for sign language."""
    all_X, all_y = [X], [y]

    def add(X_new):
        all_X.append(X_new)
        all_y.append(y.copy())

    add(augment_mirror(X))                                    # 2x
    add(augment_noise(X, 0.005))                              # 3x
    add(augment_noise(X, 0.012))                              # 4x
    add(augment_noise(X, 0.020))                              # 5x
    add(augment_temporal(X, (0.85, 0.95)))                    # 6x
    add(augment_temporal(X, (1.05, 1.15)))                    # 7x
    add(augment_scale(X, (0.85, 1.15)))                       # 8x
    add(augment_noise(augment_mirror(X), 0.008))              # 9x
    add(augment_temporal(augment_mirror(X), (0.9, 1.1)))      # 10x

    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ---------------------------------------------------------------------------
# Model Architecture (same as v2 but serialization-safe)
# ---------------------------------------------------------------------------

def build_model(n_frames: int, n_features: int, n_classes: int,
                dropout: float = 0.3) -> tf.keras.Model:
    """CNN + BiLSTM + Attention (no Lambda layers)."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # CNN block
    x = tf.keras.layers.Conv1D(128, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    # BiLSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Pooling (attention-weighted + global avg + global max)
    att = tf.keras.layers.Dense(1)(x)
    att = tf.keras.layers.Softmax(axis=1)(att)
    ctx = tf.keras.layers.Multiply()([x, att])
    ctx = tf.keras.layers.GlobalAveragePooling1D()(ctx)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

    x = tf.keras.layers.Concatenate()([ctx, avg_pool, max_pool])

    # Head
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------

def predict_with_tta(model, X_test_pos: np.ndarray) -> np.ndarray:
    """TTA: predict on original + augmented versions, average softmax."""
    preds = []

    # 1. Original
    preds.append(model.predict(add_velocity(X_test_pos), verbose=0))

    # 2. Mirror
    X_m = augment_mirror(X_test_pos)
    preds.append(model.predict(add_velocity(X_m), verbose=0))

    # 3-5. Noise at different levels
    for std in [0.003, 0.005, 0.008]:
        X_n = X_test_pos + np.random.normal(0, std, X_test_pos.shape).astype(np.float32)
        preds.append(model.predict(add_velocity(X_n), verbose=0))

    # 6. Slight temporal variation
    X_t = augment_temporal(X_test_pos, (0.95, 1.05))
    preds.append(model.predict(add_velocity(X_t), verbose=0))

    # 7. Mirror + noise
    X_mn = X_m + np.random.normal(0, 0.005, X_m.shape).astype(np.float32)
    preds.append(model.predict(add_velocity(X_mn), verbose=0))

    return np.mean(preds, axis=0)


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------

def resample_sequence(seq: np.ndarray, target: int) -> np.ndarray:
    T, F = seq.shape
    if T == target:
        return seq
    idx = np.linspace(0, T - 1, target)
    out = np.zeros((target, F), dtype=np.float32)
    for f in range(F):
        out[:, f] = np.interp(idx, np.arange(T), seq[:, f])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ensemble + TTA training")
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    SEEDS = [42, 123, 456, 789, 1024][:args.n_models]

    print("=" * 60)
    print(f"  LSC50 — Ensemble Training ({args.n_models} models + TTA)")
    print("=" * 60)

    # 1. Load and prepare data
    print("\n[1] Loading data...")
    X_padded = np.load(PROCESSED_DIR / "X_padded.npy")
    y_labels = np.load(PROCESSED_DIR / "y_labels.npy")
    lengths = np.load(PROCESSED_DIR / "lengths.npy")

    N = len(y_labels)
    X_all = np.zeros((N, args.seq_length, N_TOTAL_FEATURES), dtype=np.float32)
    for i in range(N):
        T = int(lengths[i])
        X_all[i] = resample_sequence(X_padded[i, :T, :], args.seq_length)

    # Fixed split (seed=42 for consistency with previous runs)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_labels, test_size=0.2, stratify=y_labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    n_classes = len(np.unique(y_train))
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Validation data with velocity (same for all models)
    X_val_feat = add_velocity(X_val)
    X_test_feat = add_velocity(X_test)
    y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes)

    # 2. Train N models
    model_paths = []
    individual_accs = []

    for model_idx, seed in enumerate(SEEDS):
        print(f"\n{'=' * 60}")
        print(f"  MODEL {model_idx + 1}/{args.n_models} (seed={seed})")
        print(f"{'=' * 60}")

        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Augment with this seed's randomness
        X_aug, y_aug = refined_augmentation(X_train, y_train)
        X_aug_feat = add_velocity(X_aug)
        y_aug_oh = tf.keras.utils.to_categorical(y_aug, n_classes)

        n_features = X_aug_feat.shape[2]
        print(f"  Augmented: {X_aug.shape[0]} samples, {n_features} features")

        # Build
        model = build_model(args.seq_length, n_features, n_classes,
                            dropout=args.dropout)
        if model_idx == 0:
            model.summary()

        steps_per_epoch = len(y_aug) // args.batch_size + 1
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=args.epochs * steps_per_epoch,
            alpha=1e-6,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"],
        )

        model_path = str(MODELS_DIR / f"ensemble_{model_idx}.keras")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=args.patience,
                restore_best_weights=True, verbose=1, mode="max"),
            tf.keras.callbacks.ModelCheckpoint(
                model_path, monitor="val_accuracy",
                save_best_only=True, verbose=1, mode="max"),
        ]

        model.fit(
            X_aug_feat, y_aug_oh,
            validation_data=(X_val_feat, y_val_oh),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=2,  # one line per epoch
        )

        # Individual evaluation
        try:
            model = tf.keras.models.load_model(model_path)
        except ValueError:
            model = tf.keras.models.load_model(model_path, safe_mode=False)

        y_pred = np.argmax(model.predict(X_test_feat, verbose=0), axis=1)
        acc = float(np.mean(y_pred == y_test))
        individual_accs.append(acc)
        model_paths.append(model_path)

        print(f"\n  Model {model_idx + 1} test accuracy: {acc:.1%}")

        # Free memory
        del X_aug, X_aug_feat, y_aug, y_aug_oh
        tf.keras.backend.clear_session()

    # 3. Ensemble evaluation
    print(f"\n{'=' * 60}")
    print("  ENSEMBLE EVALUATION")
    print(f"{'=' * 60}")

    # Load all models
    models = []
    for mp in model_paths:
        try:
            m = tf.keras.models.load_model(mp)
        except ValueError:
            m = tf.keras.models.load_model(mp, safe_mode=False)
        models.append(m)

    # 3a. Simple ensemble (average softmax on test)
    all_preds = [m.predict(X_test_feat, verbose=0) for m in models]
    ensemble_pred = np.mean(all_preds, axis=0)
    y_ensemble = np.argmax(ensemble_pred, axis=1)
    acc_ensemble = float(np.mean(y_ensemble == y_test))

    print(f"\n  Individual accuracies: {[f'{a:.1%}' for a in individual_accs]}")
    print(f"  Ensemble accuracy:     {acc_ensemble:.1%}")

    # 3b. Ensemble + TTA
    all_tta_preds = [predict_with_tta(m, X_test) for m in models]
    ensemble_tta_pred = np.mean(all_tta_preds, axis=0)
    y_etta = np.argmax(ensemble_tta_pred, axis=1)
    acc_etta = float(np.mean(y_etta == y_test))

    print(f"  Ensemble + TTA:        {acc_etta:.1%}")

    # Detailed report for best method
    best_acc = max(acc_ensemble, acc_etta)
    best_pred = y_ensemble if acc_ensemble >= acc_etta else y_etta
    best_method = "Ensemble" if acc_ensemble >= acc_etta else "Ensemble+TTA"

    print(f"\n  Best: {best_method} — {best_acc:.1%}")
    print(f"\n{classification_report(y_test, best_pred)}")

    # 4. Save the best single model as best_model.keras for the translator
    best_single_idx = int(np.argmax(individual_accs))
    best_single_path = model_paths[best_single_idx]

    import shutil
    shutil.copy2(best_single_path, str(MODELS_DIR / "best_model.keras"))
    print(f"  Best single model (#{best_single_idx}, {individual_accs[best_single_idx]:.1%}) "
          f"saved as best_model.keras")

    # Save results
    results = {
        "n_models": args.n_models,
        "seeds": SEEDS,
        "individual_accuracies": individual_accs,
        "ensemble_accuracy": acc_ensemble,
        "ensemble_tta_accuracy": acc_etta,
        "best_method": best_method,
        "best_accuracy": best_acc,
        "model_paths": [str(p) for p in model_paths],
    }
    with open(MODELS_DIR / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  FINAL: {best_method} = {best_acc:.1%}")
    print(f"  Target: 98%")
    print(f"  {'PASSED!' if best_acc >= 0.98 else 'Need more work...'}")
    print(f"{'=' * 60}")

    return best_acc


if __name__ == "__main__":
    main()
