"""
Train additional models with different architectures to add diversity to ensemble.
Models 5-7 to complement the existing 5 CNN+BiLSTM models.
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
from src.train_lstm import N_TOTAL_FEATURES
from src.train_v3 import add_velocity, refined_augmentation, resample_sequence


# ---------------------------------------------------------------------------
# Alternative Architectures
# ---------------------------------------------------------------------------

def build_bilstm_only(n_frames, n_features, n_classes, dropout=0.4):
    """Pure BiLSTM — no CNN. Different inductive bias."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True))(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


def build_temporal_cnn(n_frames, n_features, n_classes, dropout=0.3):
    """1D CNN with global pooling — captures local patterns differently."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # Multi-scale CNN
    x = tf.keras.layers.Conv1D(128, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    x = tf.keras.layers.Conv1D(128, 5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    x = tf.keras.layers.Conv1D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv1D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Multi-pool
    avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    mx = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg, mx])

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


def build_cnn_bilstm_heavy_reg(n_frames, n_features, n_classes, dropout=0.5):
    """CNN+BiLSTM with heavier regularization and L2."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    x = tf.keras.layers.Conv1D(
        96, 3, padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(96, return_sequences=True,
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(48, return_sequences=False,
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(96, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXTRA_MODELS = [
    ("bilstm_only",    build_bilstm_only,        2024, 0.4),
    ("temporal_cnn",   build_temporal_cnn,        314,  0.3),
    ("cnn_bilstm_reg", build_cnn_bilstm_heavy_reg, 777, 0.5),
]


def main():
    print("=" * 60)
    print("  Training 3 extra models (diverse architectures)")
    print("=" * 60)

    # Load data
    X_padded = np.load(PROCESSED_DIR / "X_padded.npy")
    y_labels = np.load(PROCESSED_DIR / "y_labels.npy")
    lengths = np.load(PROCESSED_DIR / "lengths.npy")

    SEQ = 30
    N = len(y_labels)
    X_all = np.zeros((N, SEQ, N_TOTAL_FEATURES), dtype=np.float32)
    for i in range(N):
        T = int(lengths[i])
        X_all[i] = resample_sequence(X_padded[i, :T, :], SEQ)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_labels, test_size=0.2, stratify=y_labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    n_classes = len(np.unique(y_train))
    X_val_feat = add_velocity(X_val)
    X_test_feat = add_velocity(X_test)
    y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes)

    for name, builder, seed, dropout in EXTRA_MODELS:
        print(f"\n{'=' * 60}")
        print(f"  Training: {name} (seed={seed}, dropout={dropout})")
        print(f"{'=' * 60}")

        np.random.seed(seed)
        tf.random.set_seed(seed)

        X_aug, y_aug = refined_augmentation(X_train, y_train)
        X_aug_feat = add_velocity(X_aug)
        y_aug_oh = tf.keras.utils.to_categorical(y_aug, n_classes)
        n_features = X_aug_feat.shape[2]

        model = builder(SEQ, n_features, n_classes, dropout=dropout)
        model.summary()

        steps_per_epoch = len(y_aug) // 64 + 1
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.001,
            decay_steps=200 * steps_per_epoch,
            alpha=1e-6,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy"],
        )

        model_path = str(MODELS_DIR / f"ensemble_{name}.keras")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=25,
                restore_best_weights=True, verbose=1, mode="max"),
            tf.keras.callbacks.ModelCheckpoint(
                model_path, monitor="val_accuracy",
                save_best_only=True, verbose=1, mode="max"),
        ]

        model.fit(
            X_aug_feat, y_aug_oh,
            validation_data=(X_val_feat, y_val_oh),
            epochs=200,
            batch_size=64,
            callbacks=callbacks,
            verbose=2,
        )

        # Evaluate
        try:
            model = tf.keras.models.load_model(model_path)
        except ValueError:
            model = tf.keras.models.load_model(model_path, safe_mode=False)

        y_pred = np.argmax(model.predict(X_test_feat, verbose=0), axis=1)
        acc = float(np.mean(y_pred == y_test))
        print(f"\n  {name} test accuracy: {acc:.1%}")

        # Check the 3 problematic samples
        for idx, true_cls in [(4, 12), (43, 26), (52, 11)]:
            pred_cls = int(y_pred[idx])
            status = "OK" if pred_cls == true_cls else f"WRONG (pred={pred_cls})"
            print(f"    Sample {idx} (class {true_cls}): {status}")

        del X_aug, X_aug_feat, y_aug, y_aug_oh
        tf.keras.backend.clear_session()

    print(f"\n{'=' * 60}")
    print("  Extra models trained. Run eval_ensemble.py to evaluate full ensemble.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
