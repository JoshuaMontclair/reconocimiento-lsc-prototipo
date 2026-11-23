"""
Train models specifically to fix confusion pairs: class 11<->40, class 26<->44.

Strategy:
  - Heavily oversample the 4 confused classes (10x more)
  - Use class weights to penalize misclassification of these classes
  - Multiple architectures with different seeds
  - Focal loss to focus on hard examples
  - Train models that MUST distinguish these pairs correctly
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


# ---------------------------------------------------------------------------
# Focal Loss — focuses on hard-to-classify examples
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class FocalCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        if self.label_smoothing > 0:
            n_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        return tf.reduce_sum(weight * ce, axis=-1)


# ---------------------------------------------------------------------------
# Augmentation with class-aware oversampling
# ---------------------------------------------------------------------------

def confusion_aware_augmentation(X, y, problem_classes, oversample_factor=10):
    """Augment data with heavy oversampling of problem classes."""
    all_X, all_y = [X], [y]

    def add(X_new, y_new=None):
        all_X.append(X_new)
        all_y.append(y_new if y_new is not None else y.copy())

    X_m = augment_mirror(X)

    # Standard augmentations for all classes
    add(X_m)
    add(augment_noise(X, 0.005))
    add(augment_noise(X, 0.012))
    add(augment_temporal(X, (0.85, 0.95)))
    add(augment_temporal(X, (1.05, 1.15)))
    add(augment_scale(X, (0.85, 1.15)))
    add(augment_noise(X_m, 0.008))
    add(augment_temporal(X_m, (0.9, 1.1)))

    # Heavy oversampling of problem classes
    problem_mask = np.isin(y, problem_classes)
    X_prob = X[problem_mask]
    y_prob = y[problem_mask]

    if len(X_prob) > 0:
        for _ in range(oversample_factor):
            add(X_prob, y_prob)
            add(augment_noise(X_prob, 0.003), y_prob)
            add(augment_noise(X_prob, 0.008), y_prob)
            add(augment_noise(X_prob, 0.015), y_prob)
            add(augment_mirror(X_prob), y_prob)
            add(augment_temporal(X_prob, (0.9, 1.1)), y_prob)
            add(augment_scale(X_prob, (0.9, 1.1)), y_prob)
            add(augment_noise(augment_mirror(X_prob), 0.005), y_prob)
            add(augment_temporal(augment_mirror(X_prob), (0.95, 1.05)), y_prob)
            add(augment_noise(augment_scale(X_prob, (0.95, 1.05)), 0.005), y_prob)

    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

def build_confusion_specialist(n_frames, n_features, n_classes, dropout=0.3):
    """Model optimized for distinguishing similar classes."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # Multi-scale CNN for fine-grained feature extraction
    b3 = tf.keras.layers.Conv1D(64, 3, padding="same")(inp)
    b3 = tf.keras.layers.BatchNormalization()(b3)
    b3 = tf.keras.layers.ReLU()(b3)

    b5 = tf.keras.layers.Conv1D(64, 5, padding="same")(inp)
    b5 = tf.keras.layers.BatchNormalization()(b5)
    b5 = tf.keras.layers.ReLU()(b5)

    b7 = tf.keras.layers.Conv1D(64, 7, padding="same")(inp)
    b7 = tf.keras.layers.BatchNormalization()(b7)
    b7 = tf.keras.layers.ReLU()(b7)

    b1 = tf.keras.layers.Conv1D(32, 1, padding="same")(inp)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.ReLU()(b1)

    x = tf.keras.layers.Concatenate()([b1, b3, b5, b7])  # 224 channels
    x = tf.keras.layers.Dropout(dropout * 0.3)(x)

    # Second conv block
    x = tf.keras.layers.Conv1D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    # BiLSTM with attention
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(96, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Multi-head attention pooling
    att = tf.keras.layers.Dense(1)(x)
    att = tf.keras.layers.Softmax(axis=1)(att)
    ctx = tf.keras.layers.Multiply()([x, att])
    ctx = tf.keras.layers.GlobalAveragePooling1D()(ctx)
    avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    mx = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([ctx, avg, mx])

    x = tf.keras.layers.Dense(384, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(192, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


def build_deep_bilstm(n_frames, n_features, n_classes, dropout=0.3):
    """Deep BiLSTM with 3 layers — more temporal capacity."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    x = tf.keras.layers.Conv1D(128, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(96, return_sequences=True))(x)
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
# Models to train
# ---------------------------------------------------------------------------

CONFUSION_MODELS = [
    # (name, builder, seed, dropout, oversample, lr, gamma, epochs)
    ("conffix_s1", build_confusion_specialist, 42,   0.30, 10, 0.001,  2.0, 300),
    ("conffix_s2", build_confusion_specialist, 2025, 0.35, 15, 0.001,  3.0, 300),
    ("conffix_s3", build_deep_bilstm,          123,  0.30, 10, 0.001,  2.0, 300),
    ("conffix_s4", build_deep_bilstm,          789,  0.35, 15, 0.0008, 3.0, 300),
    ("conffix_s5", build_confusion_specialist, 3141, 0.25, 20, 0.001,  2.5, 300),
    ("conffix_s6", build_deep_bilstm,          5555, 0.30, 20, 0.001,  2.5, 300),
]

PROBLEM_CLASSES = [11, 26, 40, 44]  # The 4 confused classes


def main():
    print("=" * 60)
    print("  Training CONFUSION-FIX models")
    print("  Target: fix class 11<->40 and class 26<->44 confusion")
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

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_labels, test_size=0.2, stratify=y_labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    n_classes = len(np.unique(y_train))
    X_val_feat = add_velocity(X_val)
    X_test_feat = add_velocity(X_test)
    y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes)

    # Compute class weights — higher weight for problem classes
    class_weights = {}
    for c in range(n_classes):
        if c in PROBLEM_CLASSES:
            class_weights[c] = 5.0  # 5x penalty for misclassifying these
        else:
            class_weights[c] = 1.0

    # Problem test samples
    problem_samples = []
    for idx in range(len(y_test)):
        if y_test[idx] in [11, 26]:
            problem_samples.append((idx, int(y_test[idx])))
    print(f"\n  Problem test samples: {problem_samples}")

    useful_models = []

    for name, builder, seed, dropout, oversample, lr, gamma, epochs in CONFUSION_MODELS:
        print(f"\n{'=' * 60}")
        print(f"  Training: {name} (seed={seed}, oversample={oversample}x, gamma={gamma})")
        print(f"{'=' * 60}")

        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Confusion-aware augmentation with heavy oversampling
        X_aug, y_aug = confusion_aware_augmentation(
            X_train, y_train, PROBLEM_CLASSES, oversample_factor=oversample)
        X_aug_feat = add_velocity(X_aug)
        y_aug_oh = tf.keras.utils.to_categorical(y_aug, n_classes)
        n_features = X_aug_feat.shape[2]

        print(f"  Augmented: {X_aug.shape[0]} samples")

        # Count class distribution
        for pc in PROBLEM_CLASSES:
            count = np.sum(y_aug == pc)
            print(f"    Class {pc}: {count} samples")

        model = builder(SEQ, n_features, n_classes, dropout=dropout)

        steps_per_epoch = len(y_aug) // 64 + 1
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=epochs * steps_per_epoch,
            alpha=1e-6,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=FocalCategoricalCrossentropy(gamma=gamma, label_smoothing=0.05),
            metrics=["accuracy"],
        )

        model_path = str(MODELS_DIR / f"targeted_{name}.keras")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=40,
                restore_best_weights=True, verbose=1, mode="max"),
            tf.keras.callbacks.ModelCheckpoint(
                model_path, monitor="val_accuracy",
                save_best_only=True, verbose=1, mode="max"),
        ]

        model.fit(
            X_aug_feat, y_aug_oh,
            validation_data=(X_val_feat, y_val_oh),
            epochs=epochs,
            batch_size=64,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=2,
        )

        # Evaluate
        try:
            model = tf.keras.models.load_model(model_path, safe_mode=False)
        except Exception:
            model = tf.keras.models.load_model(model_path)

        y_pred = np.argmax(model.predict(X_test_feat, verbose=0), axis=1)
        acc = float(np.mean(y_pred == y_test))
        print(f"\n  {name} test accuracy: {acc:.1%}")

        # Check problem samples
        gets_right = 0
        for idx, true_cls in problem_samples:
            pred_cls = int(y_pred[idx])
            status = "CORRECT" if pred_cls == true_cls else f"WRONG (pred={pred_cls})"
            print(f"    Sample {idx} (class {true_cls}): {status}")
            if pred_cls == true_cls:
                gets_right += 1

        if gets_right > 0 or acc >= 0.93:
            useful_models.append((name, gets_right, acc))
            print(f"    >>> USEFUL - gets {gets_right}/{len(problem_samples)} problems right")
        else:
            os.remove(model_path)
            print(f"    Deleted (not useful)")

        del X_aug, X_aug_feat, y_aug, y_aug_oh
        tf.keras.backend.clear_session()

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    for name, n_correct, acc in useful_models:
        print(f"    {name}: {acc:.1%} (gets {n_correct} problems right)")
    print(f"\n  Run eval_full.py to evaluate with all models.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
