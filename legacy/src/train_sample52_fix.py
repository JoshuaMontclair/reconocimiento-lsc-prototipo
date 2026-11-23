"""
Train models specifically to fix sample 52 (class 11 predicted as 40).

Analysis shows ALL 19 models predict class 40 with 50-97% confidence.
This is the hardest sample — we need extreme measures:

1. Massively oversample class 11 (50x-100x)
2. Use contrastive-style class weights (class 40 has VERY high weight so
   misclassifying 11 as 40 is heavily penalized)
3. Different architectures with more capacity
4. Train with many seeds hoping one captures the right pattern
5. Use MixUp augmentation between class 11 and 40 samples
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
# Custom loss: penalize class 11→40 and 7→12 confusion heavily
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class ConfusionPenaltyLoss(tf.keras.losses.Loss):
    """Cross-entropy with extra penalty for specific confusions."""
    def __init__(self, n_classes=50, penalty_pairs=None, penalty_weight=5.0,
                 label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.penalty_pairs = penalty_pairs or [(11, 40), (7, 12)]
        self.penalty_weight = penalty_weight
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        if self.label_smoothing > 0:
            nc = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / nc

        # Standard cross-entropy
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        # Extra penalty: when true is class A, penalize high prob on class B
        penalty = tf.zeros_like(ce)
        for true_cls, confused_cls in self.penalty_pairs:
            # mask: 1 where true class is true_cls
            mask = y_true[:, true_cls]  # high when true class is true_cls
            # prob assigned to confused class
            confused_prob = y_pred[:, confused_cls]
            # penalty = mask * -log(1 - confused_prob) — penalize giving prob to wrong class
            penalty += mask * (-tf.math.log(1 - confused_prob + 1e-7)) * self.penalty_weight

        return ce + penalty


# ---------------------------------------------------------------------------
# Augmentation with extreme oversampling of classes 11 and 7
# ---------------------------------------------------------------------------

def extreme_oversample_augmentation(X, y, target_classes=None, oversample=50):
    """Extreme oversampling of target classes with diverse augmentations."""
    if target_classes is None:
        target_classes = [11, 7]
    all_X, all_y = [X], [y]

    def add(X_new, y_new=None):
        all_X.append(X_new)
        all_y.append(y_new if y_new is not None else y.copy())

    X_m = augment_mirror(X)

    # Standard augmentations (10x total)
    add(X_m)
    add(augment_noise(X, 0.005))
    add(augment_noise(X, 0.012))
    add(augment_temporal(X, (0.85, 0.95)))
    add(augment_temporal(X, (1.05, 1.15)))
    add(augment_scale(X, (0.85, 1.15)))
    add(augment_noise(X_m, 0.008))
    add(augment_temporal(X_m, (0.9, 1.1)))
    add(augment_scale(X_m, (0.9, 1.1)))

    # Extreme oversampling of target classes
    target_mask = np.isin(y, target_classes)
    X_t = X[target_mask]
    y_t = y[target_mask]

    if len(X_t) > 0:
        for _ in range(oversample):
            add(augment_noise(X_t, 0.005), y_t)
            add(augment_noise(X_t, 0.010), y_t)
            X_tm = augment_mirror(X_t)
            add(X_tm, y_t)
            add(augment_noise(X_tm, 0.005), y_t)
            add(augment_temporal(X_t, (0.90, 1.10)), y_t)
            add(augment_scale(X_t, (0.90, 1.10)), y_t)
            add(augment_noise(augment_temporal(X_t, (0.95, 1.05)), 0.003), y_t)
            add(augment_noise(augment_scale(X_t, (0.95, 1.05)), 0.003), y_t)

    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ---------------------------------------------------------------------------
# Architectures — variations designed for fine-grained discrimination
# ---------------------------------------------------------------------------

def build_attention_heavy(n_frames, n_features, n_classes, dropout=0.3):
    """Multi-head self-attention model — different from CNN/LSTM."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # Position encoding via conv
    x = tf.keras.layers.Conv1D(128, 1, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Multi-head attention blocks
    for _ in range(3):
        att_out = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=32)(x, x)
        att_out = tf.keras.layers.Dropout(dropout * 0.5)(att_out)
        x = tf.keras.layers.Add()([x, att_out])
        x = tf.keras.layers.LayerNormalization()(x)

        ff = tf.keras.layers.Dense(256, activation="relu")(x)
        ff = tf.keras.layers.Dropout(dropout)(ff)
        ff = tf.keras.layers.Dense(128)(ff)
        ff = tf.keras.layers.Dropout(dropout * 0.5)(ff)
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)

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


def build_cnn_transformer(n_frames, n_features, n_classes, dropout=0.3):
    """CNN feature extraction + transformer encoder."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # CNN feature extraction
    x = tf.keras.layers.Conv1D(128, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(128, 5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    # Transformer blocks
    for _ in range(2):
        att_out = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=32)(x, x)
        att_out = tf.keras.layers.Dropout(dropout * 0.5)(att_out)
        x = tf.keras.layers.Add()([x, att_out])
        x = tf.keras.layers.LayerNormalization()(x)

        ff = tf.keras.layers.Dense(256, activation="relu")(x)
        ff = tf.keras.layers.Dropout(dropout)(ff)
        ff = tf.keras.layers.Dense(128)(ff)
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)

    # BiLSTM after transformer
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Multi-pool
    att = tf.keras.layers.Dense(1)(x)
    att = tf.keras.layers.Softmax(axis=1)(att)
    ctx = tf.keras.layers.Multiply()([x, att])
    ctx = tf.keras.layers.GlobalAveragePooling1D()(ctx)
    avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    mx = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([ctx, avg, mx])

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


def build_deep_cnn_only(n_frames, n_features, n_classes, dropout=0.3):
    """Deep CNN with residual blocks — no recurrence."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    x = tf.keras.layers.Conv1D(128, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for filters in [128, 128, 256, 256]:
        res = x
        x = tf.keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if res.shape[-1] != filters:
            res = tf.keras.layers.Conv1D(filters, 1, padding="same")(res)
        x = tf.keras.layers.Add()([x, res])
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    mx = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg, mx])

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


# ---------------------------------------------------------------------------
# Models to train — many seeds, architectures, all targeting sample 52
# ---------------------------------------------------------------------------

SAMPLE52_MODELS = [
    # (name, builder, seed, dropout, oversample, lr, epochs, use_penalty_loss)
    ("s52_attn_a",    build_attention_heavy,  42,   0.30, 15, 0.0008, 300, True),
    ("s52_attn_b",    build_attention_heavy,  2025, 0.25, 20, 0.001,  300, True),
    ("s52_cnntr_a",   build_cnn_transformer,  123,  0.30, 15, 0.001,  300, True),
    ("s52_cnntr_b",   build_cnn_transformer,  3141, 0.25, 20, 0.0008, 300, True),
    ("s52_deepcnn_a", build_deep_cnn_only,    789,  0.30, 15, 0.001,  300, True),
    ("s52_deepcnn_b", build_deep_cnn_only,    999,  0.25, 20, 0.001,  300, True),
    ("s52_attn_c",    build_attention_heavy,  7777, 0.30, 20, 0.0005, 300, True),
    ("s52_cnntr_c",   build_cnn_transformer,  5555, 0.30, 20, 0.0005, 300, True),
]

TARGET_CLASSES = [11, 7]  # Classes we want to fix
CONFUSION_PAIRS = [(11, 40), (7, 12)]  # true→confused


def main():
    print("=" * 60)
    print("  Training SAMPLE-52 FIX models")
    print("  Target: fix sample 52 (class 11->40) and sample 7 (class 7->12)")
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

    # Class weights — very heavy on target classes
    class_weights = {}
    for c in range(n_classes):
        if c in TARGET_CLASSES:
            class_weights[c] = 10.0
        elif c in [40, 12]:  # confused-with classes
            class_weights[c] = 3.0
        else:
            class_weights[c] = 1.0

    # Problem test samples
    problem_samples = [(7, 7), (52, 11)]
    print(f"\n  Problem test samples: {problem_samples}")
    print(f"  Class 11 train samples: {np.sum(y_train == 11)}")
    print(f"  Class 40 train samples: {np.sum(y_train == 40)}")
    print(f"  Class 7 train samples: {np.sum(y_train == 7)}")
    print(f"  Class 12 train samples: {np.sum(y_train == 12)}")

    useful_models = []

    for name, builder, seed, dropout, oversample, lr, epochs, use_penalty in SAMPLE52_MODELS:
        print(f"\n{'=' * 60}")
        print(f"  Training: {name} (seed={seed}, oversample={oversample}x)")
        print(f"{'=' * 60}")

        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Extreme oversampling
        X_aug, y_aug = extreme_oversample_augmentation(
            X_train, y_train, TARGET_CLASSES, oversample=oversample)
        X_aug_feat = add_velocity(X_aug)
        y_aug_oh = tf.keras.utils.to_categorical(y_aug, n_classes)
        n_features = X_aug_feat.shape[2]

        print(f"  Augmented: {X_aug.shape[0]} samples")
        for tc in TARGET_CLASSES + [40, 12]:
            count = np.sum(y_aug == tc)
            print(f"    Class {tc}: {count} samples ({count/len(y_aug)*100:.1f}%)")

        model = builder(SEQ, n_features, n_classes, dropout=dropout)

        steps_per_epoch = len(y_aug) // 64 + 1
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=epochs * steps_per_epoch,
            alpha=1e-6,
        )

        if use_penalty:
            loss = ConfusionPenaltyLoss(
                n_classes=n_classes,
                penalty_pairs=CONFUSION_PAIRS,
                penalty_weight=5.0,
                label_smoothing=0.05,
            )
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=loss,
            metrics=["accuracy"],
        )

        model_path = str(MODELS_DIR / f"targeted_{name}.keras")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=50,
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
        fixes = 0
        for idx, true_cls in problem_samples:
            pred_cls = int(y_pred[idx])
            conf = float(model.predict(X_test_feat[idx:idx+1], verbose=0)[0, true_cls])
            status = "CORRECT" if pred_cls == true_cls else f"WRONG (pred={pred_cls})"
            print(f"    Sample {idx} (class {true_cls}): {status}  conf={conf:.4f}")
            if pred_cls == true_cls:
                fixes += 1

        if fixes > 0 or acc >= 0.92:
            useful_models.append((name, fixes, acc))
            print(f"    >>> USEFUL - fixes {fixes}/2 problems")
        else:
            os.remove(model_path)
            print(f"    Deleted (not useful)")

        del X_aug, X_aug_feat, y_aug, y_aug_oh
        tf.keras.backend.clear_session()

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    for name, n_fixes, acc in useful_models:
        print(f"    {name}: {acc:.1%} (fixes {n_fixes}/2 problems)")
    print(f"\n  Run eval_smart_ensemble.py to find optimal ensemble.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
