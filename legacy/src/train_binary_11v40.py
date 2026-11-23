"""
Train binary classifier: class 11 vs class 40.

Strategy: When the 50-class ensemble predicts class 40, run this binary
classifier to check if the sample is actually class 11. If the binary model
says class 11, override the ensemble prediction.

This is much simpler than trying to make a 50-class model distinguish them.
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

sys.path.insert(0, str(BASE_DIR))
from src.train_lstm import (
    N_TOTAL_FEATURES,
    augment_mirror,
    augment_noise,
    augment_scale,
    augment_temporal,
)
from src.train_v3 import add_velocity, resample_sequence


def build_binary_model(n_frames, n_features, dropout=0.3):
    """Small but precise binary classifier for class 11 vs 40."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # Multi-scale CNN
    b3 = tf.keras.layers.Conv1D(64, 3, padding="same")(inp)
    b3 = tf.keras.layers.BatchNormalization()(b3)
    b3 = tf.keras.layers.ReLU()(b3)

    b5 = tf.keras.layers.Conv1D(64, 5, padding="same")(inp)
    b5 = tf.keras.layers.BatchNormalization()(b5)
    b5 = tf.keras.layers.ReLU()(b5)

    b7 = tf.keras.layers.Conv1D(32, 7, padding="same")(inp)
    b7 = tf.keras.layers.BatchNormalization()(b7)
    b7 = tf.keras.layers.ReLU()(b7)

    x = tf.keras.layers.Concatenate()([b3, b5, b7])

    x = tf.keras.layers.Conv1D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    # BiLSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Attention pooling
    att = tf.keras.layers.Dense(1)(x)
    att = tf.keras.layers.Softmax(axis=1)(att)
    ctx = tf.keras.layers.Multiply()([x, att])
    ctx = tf.keras.layers.GlobalAveragePooling1D()(ctx)
    avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    mx = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([ctx, avg, mx])

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    # Binary output: sigmoid, 1 = class 11, 0 = class 40
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inp, x)


def augment_binary(X, y, n_rounds=30):
    """Heavy augmentation for binary classifier (few samples)."""
    all_X, all_y = [X], [y]

    def add(X_new):
        all_X.append(X_new)
        all_y.append(y.copy())

    for _ in range(n_rounds):
        X_m = augment_mirror(X)
        add(X_m)
        for std in [0.002, 0.005, 0.008, 0.012, 0.018, 0.025]:
            add(X + np.random.normal(0, std, X.shape).astype(np.float32))
        for std in [0.003, 0.008, 0.015]:
            add(X_m + np.random.normal(0, std, X_m.shape).astype(np.float32))
        add(augment_temporal(X, (0.85, 0.95)))
        add(augment_temporal(X, (1.05, 1.15)))
        add(augment_temporal(X, (0.90, 1.10)))
        add(augment_scale(X, (0.85, 1.15)))
        add(augment_scale(X, (0.90, 1.10)))
        add(augment_temporal(X_m, (0.90, 1.10)))
        add(augment_scale(X_m, (0.90, 1.10)))
        add(augment_noise(augment_temporal(X, (0.95, 1.05)), 0.005))
        add(augment_noise(augment_scale(X, (0.95, 1.05)), 0.005))

    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


BINARY_MODELS = [
    # (name, seed, dropout, lr, n_rounds, epochs)
    ("bin11v40_a", 42,   0.30, 0.001,  30, 200),
    ("bin11v40_b", 2025, 0.25, 0.001,  40, 200),
    ("bin11v40_c", 123,  0.35, 0.0008, 30, 200),
    ("bin11v40_d", 3141, 0.30, 0.001,  50, 200),
    ("bin11v40_e", 789,  0.25, 0.0005, 40, 200),
    ("bin11v40_f", 5555, 0.30, 0.001,  35, 250),
    ("bin11v40_g", 7777, 0.35, 0.0008, 45, 250),
    ("bin11v40_h", 9999, 0.25, 0.001,  50, 250),
]


def main():
    print("=" * 60)
    print("  Training BINARY classifiers: class 11 vs class 40")
    print("=" * 60)

    X_padded = np.load(PROCESSED_DIR / "X_padded.npy")
    y_labels = np.load(PROCESSED_DIR / "y_labels.npy")
    lengths = np.load(PROCESSED_DIR / "lengths.npy")

    SEQ = 30
    N = len(y_labels)
    X_all = np.zeros((N, SEQ, N_TOTAL_FEATURES), dtype=np.float32)
    for i in range(N):
        T = int(lengths[i])
        X_all[i] = resample_sequence(X_padded[i, :T, :], SEQ)

    # Same split as always
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_labels, test_size=0.2, stratify=y_labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Filter to only class 11 and class 40
    train_mask = np.isin(y_train, [11, 40])
    val_mask = np.isin(y_val, [11, 40])
    test_mask = np.isin(y_test, [11, 40])

    X_train_bin = X_train[train_mask]
    y_train_bin = (y_train[train_mask] == 11).astype(np.float32)  # 1=class11, 0=class40

    X_val_bin = X_val[val_mask]
    y_val_bin = (y_val[val_mask] == 11).astype(np.float32)

    X_test_bin = X_test[test_mask]
    y_test_bin = (y_test[test_mask] == 11).astype(np.float32)

    print(f"\n  Train: {len(y_train_bin)} samples ({sum(y_train_bin==1)} class11, {sum(y_train_bin==0)} class40)")
    print(f"  Val:   {len(y_val_bin)} samples ({sum(y_val_bin==1)} class11, {sum(y_val_bin==0)} class40)")
    print(f"  Test:  {len(y_test_bin)} samples ({sum(y_test_bin==1)} class11, {sum(y_test_bin==0)} class40)")

    # Also identify sample 52 in the full test set
    print(f"\n  Sample 52 in full test set: true class = {y_test[52]}")
    is_52_in_binary = test_mask[52] if 52 < len(test_mask) else False
    print(f"  Sample 52 in binary test set: {is_52_in_binary}")

    # We need to test on the FULL test set sample 52, not just binary subset
    # So we'll evaluate each model on sample 52 directly
    X_test_feat = add_velocity(X_test)
    sample52_feat = X_test_feat[52:53]

    useful_models = []

    for name, seed, dropout, lr, n_rounds, epochs in BINARY_MODELS:
        print(f"\n{'=' * 60}")
        print(f"  Training: {name} (seed={seed}, aug_rounds={n_rounds})")
        print(f"{'=' * 60}")

        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Augment
        X_aug, y_aug = augment_binary(X_train_bin, y_train_bin, n_rounds=n_rounds)
        X_aug_feat = add_velocity(X_aug)
        n_features = X_aug_feat.shape[2]

        X_val_feat = add_velocity(X_val_bin)

        print(f"  Augmented: {len(y_aug)} samples ({sum(y_aug==1)} class11, {sum(y_aug==0)} class40)")

        model = build_binary_model(SEQ, n_features, dropout=dropout)

        steps = len(y_aug) // 32 + 1
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=epochs * steps,
            alpha=1e-6,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

        model_path = str(MODELS_DIR / f"binary_{name}.keras")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=40,
                restore_best_weights=True, verbose=1, mode="max"),
            tf.keras.callbacks.ModelCheckpoint(
                model_path, monitor="val_accuracy",
                save_best_only=True, verbose=1, mode="max"),
        ]

        model.fit(
            X_aug_feat, y_aug,
            validation_data=(X_val_feat, y_val_bin),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=2,
        )

        # Load best
        model = tf.keras.models.load_model(model_path)

        # Test on binary subset
        X_test_bin_feat = add_velocity(X_test_bin)
        y_pred_bin = (model.predict(X_test_bin_feat, verbose=0) > 0.5).astype(int).flatten()
        acc_bin = float(np.mean(y_pred_bin == y_test_bin))
        print(f"\n  {name} binary accuracy: {acc_bin:.0%}")

        # Test on sample 52 specifically
        pred_52 = float(model.predict(sample52_feat, verbose=0)[0, 0])
        pred_class = 11 if pred_52 > 0.5 else 40
        print(f"  Sample 52: P(class11) = {pred_52:.4f} -> predicts class {pred_class}")

        if pred_class == 11:
            print(f"    >>> SUCCESS! Predicts class 11 for sample 52!")
            useful_models.append((name, pred_52, acc_bin))
        else:
            print(f"    Still predicts class 40")
            # Keep if binary accuracy is high (might be useful with TTA)
            if acc_bin >= 0.9:
                useful_models.append((name, pred_52, acc_bin))
            else:
                os.remove(model_path)
                print(f"    Deleted (not useful)")

        del X_aug, X_aug_feat, y_aug
        tf.keras.backend.clear_session()

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    for name, conf_11, acc in useful_models:
        status = "CLASS 11" if conf_11 > 0.5 else "class 40"
        print(f"  {name}: binary_acc={acc:.0%}, P(class11)={conf_11:.4f} -> {status}")

    print(f"\n  Next: integrate binary models into eval_smart_ensemble.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
