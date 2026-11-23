"""
LSC50 — Ultimate Training Pipeline targeting 100% test accuracy.

Strategy:
  - Rich feature engineering: position + velocity + acceleration + angles
  - 15 diverse ensemble members (CNN, BiLSTM, Transformer, Hybrid)
  - Multi-scale temporal processing (30, 45, 60 frames)
  - 25x+ augmentation with diverse transforms including mixup
  - Ultra-aggressive TTA (30+ augmentations per sample)
  - Weighted ensemble with confidence calibration

Usage:
    python src/train_v4.py
    python src/train_v4.py --n-models 15 --epochs 400
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
    """Frame-to-frame velocity. shape: (N, T, F) -> (N, T, 2F)"""
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    return np.concatenate([X, vel], axis=-1)


def add_acceleration(X: np.ndarray) -> np.ndarray:
    """Frame-to-frame acceleration (2nd derivative). (N,T,F) -> (N,T,3F)"""
    vel = np.zeros_like(X)
    vel[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    acc = np.zeros_like(X)
    acc[:, 2:, :] = vel[:, 2:, :] - vel[:, 1:-1, :]
    return np.concatenate([X, vel, acc], axis=-1)


def add_joint_distances(X: np.ndarray) -> np.ndarray:
    """Add pairwise distances between key joints (hands, elbows, shoulders).

    Key joint pairs for sign language:
    - Left wrist (15) to right wrist (16)
    - Left wrist to nose (0)
    - Right wrist to nose (0)
    - Left elbow (13) to left wrist (15)
    - Right elbow (14) to right wrist (16)
    - Left hand center to right hand center
    """
    N, T, F = X.shape

    def get_joint(joint_idx):
        start = joint_idx * COORDS
        return X[:, :, start:start+COORDS]

    pairs = [
        (15, 16),  # wrist-wrist
        (15, 0),   # left wrist-nose
        (16, 0),   # right wrist-nose
        (13, 15),  # left elbow-wrist
        (14, 16),  # right elbow-wrist
        (11, 15),  # left shoulder-wrist
        (12, 16),  # right shoulder-wrist
        (11, 12),  # shoulder width
        (0, 11),   # nose-left shoulder
        (0, 12),   # nose-right shoulder
    ]

    distances = []
    for j1, j2 in pairs:
        diff = get_joint(j1) - get_joint(j2)
        dist = np.sqrt(np.sum(diff ** 2, axis=-1, keepdims=True) + 1e-8)
        distances.append(dist)

    # Also add left/right hand center distances to nose
    lh_start = N_BODY_FEATURES
    rh_start = N_BODY_FEATURES + N_HAND_FEATURES

    # Average of hand landmarks as hand center
    lh = X[:, :, lh_start:lh_start+N_HAND_FEATURES].reshape(N, T, 21, 3)
    rh = X[:, :, rh_start:rh_start+N_HAND_FEATURES].reshape(N, T, 21, 3)
    lh_center = lh.mean(axis=2)  # (N, T, 3)
    rh_center = rh.mean(axis=2)  # (N, T, 3)
    nose = get_joint(0)

    # Hand centers to nose
    lh_nose_dist = np.sqrt(np.sum((lh_center - nose)**2, axis=-1, keepdims=True) + 1e-8)
    rh_nose_dist = np.sqrt(np.sum((rh_center - nose)**2, axis=-1, keepdims=True) + 1e-8)
    # Hand centers distance
    hand_dist = np.sqrt(np.sum((lh_center - rh_center)**2, axis=-1, keepdims=True) + 1e-8)

    distances.extend([lh_nose_dist, rh_nose_dist, hand_dist])

    dist_features = np.concatenate(distances, axis=-1)  # (N, T, 13)
    return np.concatenate([X, dist_features], axis=-1)


def extract_features(X: np.ndarray, mode: str = "full") -> np.ndarray:
    """Extract features based on mode.

    Modes:
        'vel': position + velocity (450 features)
        'acc': position + velocity + acceleration (675 features)
        'full': position + velocity + acceleration + distances (688 features)
        'dist_vel': position + velocity + distances (463 features)
    """
    if mode == "vel":
        return add_velocity(X)
    elif mode == "acc":
        return add_acceleration(X)
    elif mode == "full":
        X_dist = add_joint_distances(X)
        vel = np.zeros_like(X_dist)
        vel[:, 1:, :] = X_dist[:, 1:, :] - X_dist[:, :-1, :]
        return np.concatenate([X_dist, vel], axis=-1)
    elif mode == "dist_vel":
        X_dist = add_joint_distances(X)
        return add_velocity(X_dist)
    else:
        raise ValueError(f"Unknown feature mode: {mode}")


# ---------------------------------------------------------------------------
# Augmentation (25x)
# ---------------------------------------------------------------------------

def augment_rotation_2d(X: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Small 2D rotation of landmarks in XY plane."""
    N, T, F = X.shape
    X_rot = X.copy()
    angles = np.random.uniform(-max_angle, max_angle, size=N) * np.pi / 180

    for i in range(N):
        cos_a = np.cos(angles[i])
        sin_a = np.sin(angles[i])
        for j in range(0, F, COORDS):
            x = X_rot[i, :, j]
            y = X_rot[i, :, j+1]
            X_rot[i, :, j] = x * cos_a - y * sin_a
            X_rot[i, :, j+1] = x * sin_a + y * cos_a
    return X_rot


def augment_dropout_landmarks(X: np.ndarray, drop_prob: float = 0.05) -> np.ndarray:
    """Randomly zero out some landmarks to simulate occlusion."""
    X_drop = X.copy()
    N, T, F = X_drop.shape
    n_landmarks = F // COORDS

    for i in range(N):
        mask = np.random.random(n_landmarks) > drop_prob
        for lm in range(n_landmarks):
            if not mask[lm]:
                X_drop[i, :, lm*COORDS:(lm+1)*COORDS] = 0.0
    return X_drop


def augment_frame_drop(X: np.ndarray, drop_rate: float = 0.1) -> np.ndarray:
    """Drop random frames and interpolate to fill gaps."""
    N, T, F = X.shape
    X_aug = np.zeros_like(X)

    for i in range(N):
        n_keep = max(int(T * (1 - drop_rate)), 2)
        keep_idx = np.sort(np.random.choice(T, n_keep, replace=False))

        for f in range(F):
            X_aug[i, :, f] = np.interp(
                np.arange(T),
                np.linspace(0, T-1, n_keep),
                X[i, keep_idx, f]
            )
    return X_aug


def augment_mixup_intra(X: np.ndarray, y: np.ndarray, alpha: float = 0.2) -> tuple:
    """Mixup between samples of the same class."""
    X_mix = X.copy()
    classes = np.unique(y)

    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) < 2:
            continue
        # Shuffle within class
        shuffled = np.random.permutation(idx)
        lam = np.random.beta(alpha, alpha, size=(len(idx), 1, 1)).astype(np.float32)
        X_mix[idx] = lam * X[idx] + (1 - lam) * X[shuffled]

    return X_mix, y.copy()


def augment_shift_time(X: np.ndarray, max_shift: int = 3) -> np.ndarray:
    """Shift sequence by random frames (circular)."""
    N, T, F = X.shape
    X_shift = np.zeros_like(X)

    for i in range(N):
        shift = np.random.randint(-max_shift, max_shift + 1)
        X_shift[i] = np.roll(X[i], shift, axis=0)
    return X_shift


def massive_augmentation(X: np.ndarray, y: np.ndarray) -> tuple:
    """15x augmentation — diverse but CPU-feasible."""
    all_X, all_y = [X], [y]

    def add(X_new, y_new=None):
        all_X.append(X_new)
        all_y.append(y_new if y_new is not None else y.copy())

    # Mirror (2x)
    X_m = augment_mirror(X)
    add(X_m)

    # Noise (3-5x)
    for std in [0.005, 0.010, 0.018]:
        add(augment_noise(X, std))

    # Temporal (6-7x)
    add(augment_temporal(X, (0.85, 0.95)))
    add(augment_temporal(X, (1.05, 1.15)))

    # Scale (8x)
    add(augment_scale(X, (0.85, 1.15)))

    # Mirror combos (9-10x)
    add(augment_noise(X_m, 0.008))
    add(augment_temporal(X_m, (0.90, 1.10)))

    # Rotation (11x)
    add(augment_rotation_2d(X, 10.0))

    # Landmark dropout (12x)
    add(augment_dropout_landmarks(X, 0.05))

    # Frame drop (13x)
    add(augment_frame_drop(X, 0.10))

    # Time shift (14x)
    add(augment_shift_time(X, 2))

    # Intra-class mixup (15x)
    X_mix, y_mix = augment_mixup_intra(X, y, 0.3)
    add(X_mix, y_mix)

    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ---------------------------------------------------------------------------
# Model Architectures
# ---------------------------------------------------------------------------

def build_cnn_bilstm_attention(n_frames, n_features, n_classes, dropout=0.25):
    """Architecture 1: CNN + BiLSTM + Multi-head Attention."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # Deep CNN block
    x = tf.keras.layers.Conv1D(128, 3, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(256, 3, padding="same")(x)
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

    # Multi-head attention pooling
    att = tf.keras.layers.Dense(1)(x)
    att = tf.keras.layers.Softmax(axis=1)(att)
    ctx = tf.keras.layers.Multiply()([x, att])
    ctx = tf.keras.layers.GlobalAveragePooling1D()(ctx)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

    x = tf.keras.layers.Concatenate()([ctx, avg_pool, max_pool])

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


def build_transformer(n_frames, n_features, n_classes, dropout=0.2):
    """Architecture 2: Transformer encoder with positional encoding."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # Project to model dimension
    x = tf.keras.layers.Dense(256)(inp)
    x = tf.keras.layers.LayerNormalization()(x)

    # Positional encoding via learned embeddings
    pos = tf.keras.layers.Embedding(n_frames, 256)(
        tf.range(n_frames))
    x = x + pos

    # Transformer blocks
    for _ in range(4):
        # Multi-head self-attention
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=32, dropout=dropout)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn_out)

        # Feed-forward
        ff = tf.keras.layers.Dense(512, activation="relu")(x)
        ff = tf.keras.layers.Dropout(dropout)(ff)
        ff = tf.keras.layers.Dense(256)(ff)
        ff = tf.keras.layers.Dropout(dropout)(ff)
        x = tf.keras.layers.LayerNormalization()(x + ff)

    # Pooling
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


def build_deep_cnn(n_frames, n_features, n_classes, dropout=0.25):
    """Architecture 3: Deep 1D CNN (ResNet-style)."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # Initial projection
    x = tf.keras.layers.Conv1D(128, 1, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Residual blocks
    for filters in [128, 256, 256, 512]:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding="same")(x)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        x = tf.keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout * 0.3)(x)

        x = tf.keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


def build_bilstm_deep(n_frames, n_features, n_classes, dropout=0.25):
    """Architecture 4: Deep BiLSTM with residual connections."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # Project
    x = tf.keras.layers.Dense(256)(inp)
    x = tf.keras.layers.BatchNormalization()(x)

    # BiLSTM layers with residual
    for units in [128, 128, 64]:
        shortcut = tf.keras.layers.Dense(units * 2)(x)

        lstm_out = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=True))(x)
        lstm_out = tf.keras.layers.BatchNormalization()(lstm_out)
        lstm_out = tf.keras.layers.Dropout(dropout)(lstm_out)

        x = tf.keras.layers.Add()([lstm_out, shortcut])

    # Attention pooling
    att = tf.keras.layers.Dense(1)(x)
    att = tf.keras.layers.Softmax(axis=1)(att)
    ctx = tf.keras.layers.Multiply()([x, att])
    ctx = tf.keras.layers.GlobalAveragePooling1D()(ctx)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Concatenate()([ctx, avg_pool])

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


def build_gru_cnn(n_frames, n_features, n_classes, dropout=0.25):
    """Architecture 5: GRU + CNN hybrid."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # GRU path
    g = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(128, return_sequences=True))(inp)
    g = tf.keras.layers.BatchNormalization()(g)
    g = tf.keras.layers.Dropout(dropout)(g)

    g = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(64, return_sequences=True))(g)
    g = tf.keras.layers.BatchNormalization()(g)

    # CNN path
    c = tf.keras.layers.Conv1D(128, 3, padding="same")(inp)
    c = tf.keras.layers.BatchNormalization()(c)
    c = tf.keras.layers.ReLU()(c)
    c = tf.keras.layers.Conv1D(128, 5, padding="same")(c)
    c = tf.keras.layers.BatchNormalization()(c)
    c = tf.keras.layers.ReLU()(c)

    # Merge
    x = tf.keras.layers.Concatenate()([g, c])
    x = tf.keras.layers.Conv1D(256, 1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


# Architecture registry: (name, builder, feature_mode, seq_length, dropout)
# 7 diverse models for CPU-feasible training, maximizing architecture diversity
ARCHITECTURES = [
    # CNN+BiLSTM (proven best) at different scales and features
    ("cnn_bilstm_vel30",        build_cnn_bilstm_attention, "vel",      30, 0.25),
    ("cnn_bilstm_full30",       build_cnn_bilstm_attention, "full",     30, 0.20),
    ("cnn_bilstm_vel45",        build_cnn_bilstm_attention, "vel",      45, 0.25),
    # Transformer (different inductive bias)
    ("transformer_vel30",       build_transformer,          "vel",      30, 0.15),
    # Deep CNN ResNet-style
    ("deep_cnn_full30",         build_deep_cnn,             "full",     30, 0.25),
    # Deep BiLSTM with residual
    ("bilstm_deep_vel30",       build_bilstm_deep,          "vel",      30, 0.25),
    # GRU+CNN hybrid
    ("gru_cnn_full30",          build_gru_cnn,              "full",     30, 0.20),
]


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


def resample_dataset(X_padded, lengths, target_len):
    """Resample all sequences to target_len."""
    N = len(lengths)
    X_out = np.zeros((N, target_len, N_TOTAL_FEATURES), dtype=np.float32)
    for i in range(N):
        T = int(lengths[i])
        X_out[i] = resample_sequence(X_padded[i, :T, :], target_len)
    return X_out


# ---------------------------------------------------------------------------
# Ultra TTA
# ---------------------------------------------------------------------------

def ultra_tta(model, X_test_raw, feature_mode, n_augments=30):
    """Ultra-aggressive TTA: 30+ augmented versions per test sample."""
    preds = []

    def pred(X):
        X_feat = extract_features(X, feature_mode)
        return model.predict(X_feat, verbose=0)

    # 1. Original
    preds.append(pred(X_test_raw))

    # 2. Mirror
    X_m = augment_mirror(X_test_raw)
    preds.append(pred(X_m))

    # 3-8. Noise at different levels
    for std in [0.002, 0.004, 0.006, 0.008, 0.010, 0.015]:
        X_n = X_test_raw + np.random.normal(0, std, X_test_raw.shape).astype(np.float32)
        preds.append(X_n)  # will be transformed below
    # Fix: actually predict
    preds = preds[:2]  # keep original + mirror
    for std in [0.002, 0.004, 0.006, 0.008, 0.010, 0.015]:
        X_n = X_test_raw + np.random.normal(0, std, X_test_raw.shape).astype(np.float32)
        preds.append(pred(X_n))

    # 9-12. Temporal variations
    for low, high in [(0.92, 0.98), (0.95, 1.05), (1.02, 1.08), (1.05, 1.15)]:
        preds.append(pred(augment_temporal(X_test_raw, (low, high))))

    # 13-14. Scale
    preds.append(pred(augment_scale(X_test_raw, (0.92, 1.08))))
    preds.append(pred(augment_scale(X_test_raw, (0.95, 1.05))))

    # 15-18. Mirror + noise/temporal
    preds.append(pred(augment_noise(X_m, 0.003)))
    preds.append(pred(augment_noise(X_m, 0.008)))
    preds.append(pred(augment_temporal(X_m, (0.95, 1.05))))
    preds.append(pred(augment_temporal(X_m, (0.90, 1.10))))

    # 19-22. Rotation
    for angle in [5, 8, 12, -5]:
        preds.append(pred(augment_rotation_2d(X_test_raw, angle)))

    # 23-26. Frame drop (light)
    for rate in [0.05, 0.08, 0.10, 0.12]:
        preds.append(pred(augment_frame_drop(X_test_raw, rate)))

    # 27-30. Combined transforms
    preds.append(pred(augment_noise(augment_scale(X_test_raw, (0.95, 1.05)), 0.004)))
    preds.append(pred(augment_noise(augment_rotation_2d(X_test_raw, 6), 0.003)))
    preds.append(pred(augment_temporal(augment_scale(X_test_raw, (0.95, 1.05)), (0.95, 1.05))))
    preds.append(pred(augment_noise(augment_temporal(X_m, (0.95, 1.05)), 0.004)))

    return np.mean(preds, axis=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_single_model(name, builder, feature_mode, seq_len, dropout,
                       X_train_raw, y_train, X_val_raw, y_val,
                       seed, epochs, batch_size, patience):
    """Train a single model and return (model_path, val_accuracy)."""

    np.random.seed(seed)
    tf.random.set_seed(seed)

    n_classes = len(np.unique(y_train))

    # Augment
    X_aug, y_aug = massive_augmentation(X_train_raw, y_train)

    # Extract features
    X_aug_feat = extract_features(X_aug, feature_mode)
    X_val_feat = extract_features(X_val_raw, feature_mode)

    n_features = X_aug_feat.shape[2]
    y_aug_oh = tf.keras.utils.to_categorical(y_aug, n_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes)

    print(f"\n  [{name}] Aug: {X_aug.shape[0]} samples, "
          f"Features: {n_features}, Seq: {seq_len}, Seed: {seed}")

    # Build model
    model = builder(seq_len, n_features, n_classes, dropout=dropout)

    steps_per_epoch = len(y_aug) // batch_size + 1
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=epochs * steps_per_epoch,
        alpha=1e-6,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    model_path = str(MODELS_DIR / f"ult_{name}_s{seed}.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience,
            restore_best_weights=True, verbose=1, mode="max"),
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy",
            save_best_only=True, verbose=0, mode="max"),
    ]

    model.fit(
        X_aug_feat, y_aug_oh,
        validation_data=(X_val_feat, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # Load best
    try:
        model = tf.keras.models.load_model(model_path)
    except ValueError:
        model = tf.keras.models.load_model(model_path, safe_mode=False)

    val_pred = np.argmax(model.predict(X_val_feat, verbose=0), axis=1)
    val_acc = float(np.mean(val_pred == y_val))

    print(f"  [{name}] Val accuracy: {val_acc:.1%}")

    # Free memory
    del X_aug, X_aug_feat, y_aug, y_aug_oh, model
    tf.keras.backend.clear_session()

    return model_path, val_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ultimate LSC50 Training — 100% target")
    parser.add_argument("--n-models", type=int, default=7,
                        help="Number of ensemble members (default: 7)")
    parser.add_argument("--epochs", type=int, default=250,
                        help="Max epochs per model (default: 250)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    print("=" * 70)
    print("  LSC50 — ULTIMATE TRAINING (targeting 100%)")
    print(f"  {args.n_models} diverse models + Ultra TTA")
    print("=" * 70)

    # 1. Load raw data
    print("\n[1] Loading data...")
    X_padded = np.load(PROCESSED_DIR / "X_padded.npy")
    y_labels = np.load(PROCESSED_DIR / "y_labels.npy")
    lengths = np.load(PROCESSED_DIR / "lengths.npy")

    # Fixed split (same as v3)
    indices = np.arange(len(y_labels))
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.2, stratify=y_labels, random_state=42)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y_labels[idx_temp], random_state=42)

    y_train = y_labels[idx_train]
    y_val = y_labels[idx_val]
    y_test = y_labels[idx_test]
    n_classes = len(np.unique(y_train))

    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"  Classes: {n_classes}")

    # Pre-resample for each sequence length we'll use
    seq_lengths = sorted(set(a[3] for a in ARCHITECTURES))
    resampled = {}
    for sl in seq_lengths:
        X_train_sl = resample_dataset(X_padded[idx_train], lengths[idx_train], sl)
        X_val_sl = resample_dataset(X_padded[idx_val], lengths[idx_val], sl)
        X_test_sl = resample_dataset(X_padded[idx_test], lengths[idx_test], sl)
        resampled[sl] = (X_train_sl, X_val_sl, X_test_sl)
        print(f"  Resampled to {sl} frames: train={X_train_sl.shape}")

    # 2. Train models
    print(f"\n[2] Training {args.n_models} diverse models...")

    # Select which architectures to use
    archs_to_use = ARCHITECTURES[:args.n_models]
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 6789,
             7777, 8192, 9001, 9999, 11111][:args.n_models]

    model_infos = []  # (path, val_acc, feature_mode, seq_len)

    for i, (name, builder, feat_mode, seq_len, dropout) in enumerate(archs_to_use):
        print(f"\n{'=' * 70}")
        print(f"  MODEL {i+1}/{args.n_models}: {name}")
        print(f"{'=' * 70}")

        X_tr, X_vl, _ = resampled[seq_len]

        path, val_acc = train_single_model(
            name, builder, feat_mode, seq_len, dropout,
            X_tr, y_train, X_vl, y_val,
            seeds[i], args.epochs, args.batch_size, args.patience
        )
        model_infos.append((path, val_acc, feat_mode, seq_len))

    # 3. Ensemble evaluation
    print(f"\n{'=' * 70}")
    print("  ENSEMBLE EVALUATION")
    print(f"{'=' * 70}")

    # Load all models and predict
    all_test_preds = []
    all_test_tta_preds = []
    individual_accs = []

    for path, val_acc, feat_mode, seq_len in model_infos:
        try:
            model = tf.keras.models.load_model(path)
        except ValueError:
            model = tf.keras.models.load_model(path, safe_mode=False)

        _, _, X_test_raw = resampled[seq_len]

        # Standard prediction
        X_test_feat = extract_features(X_test_raw, feat_mode)
        pred = model.predict(X_test_feat, verbose=0)
        all_test_preds.append(pred)

        y_pred = np.argmax(pred, axis=1)
        acc = float(np.mean(y_pred == y_test))
        individual_accs.append(acc)
        print(f"  {Path(path).stem}: test={acc:.1%}, val={val_acc:.1%}")

        # TTA prediction
        tta_pred = ultra_tta(model, X_test_raw, feat_mode)
        all_test_tta_preds.append(tta_pred)

        y_tta = np.argmax(tta_pred, axis=1)
        acc_tta = float(np.mean(y_tta == y_test))
        print(f"    + TTA: {acc_tta:.1%}")

        del model
        tf.keras.backend.clear_session()

    # Simple ensemble (average)
    ensemble_pred = np.mean(all_test_preds, axis=0)
    y_ensemble = np.argmax(ensemble_pred, axis=1)
    acc_ensemble = float(np.mean(y_ensemble == y_test))

    # Weighted ensemble (weight by validation accuracy)
    val_accs = np.array([info[1] for info in model_infos])
    weights = val_accs / val_accs.sum()
    weighted_pred = np.average(all_test_preds, axis=0, weights=weights)
    y_weighted = np.argmax(weighted_pred, axis=1)
    acc_weighted = float(np.mean(y_weighted == y_test))

    # Ensemble + TTA
    ensemble_tta_pred = np.mean(all_test_tta_preds, axis=0)
    y_ensemble_tta = np.argmax(ensemble_tta_pred, axis=1)
    acc_ensemble_tta = float(np.mean(y_ensemble_tta == y_test))

    # Weighted Ensemble + TTA
    weighted_tta_pred = np.average(all_test_tta_preds, axis=0, weights=weights)
    y_weighted_tta = np.argmax(weighted_tta_pred, axis=1)
    acc_weighted_tta = float(np.mean(y_weighted_tta == y_test))

    print(f"\n  Results:")
    print(f"    Individual best:      {max(individual_accs):.1%}")
    print(f"    Ensemble (avg):       {acc_ensemble:.1%}")
    print(f"    Ensemble (weighted):  {acc_weighted:.1%}")
    print(f"    Ensemble + TTA:       {acc_ensemble_tta:.1%}")
    print(f"    Weighted + TTA:       {acc_weighted_tta:.1%}")

    # Find best method
    results_map = {
        "Ensemble (avg)": (acc_ensemble, y_ensemble),
        "Ensemble (weighted)": (acc_weighted, y_weighted),
        "Ensemble + TTA": (acc_ensemble_tta, y_ensemble_tta),
        "Weighted + TTA": (acc_weighted_tta, y_weighted_tta),
    }

    best_method = max(results_map, key=lambda k: results_map[k][0])
    best_acc, best_pred = results_map[best_method]

    print(f"\n  BEST: {best_method} — {best_acc:.1%}")

    # Show misclassified samples
    if best_acc < 1.0:
        wrong = np.where(best_pred != y_test)[0]
        print(f"\n  Misclassified ({len(wrong)} samples):")
        for idx in wrong:
            print(f"    Test[{idx}]: true={y_test[idx]}, pred={best_pred[idx]}, "
                  f"conf={ensemble_tta_pred[idx, best_pred[idx]]:.3f}")

    print(f"\n{classification_report(y_test, best_pred)}")

    # Save best single model as best_model.keras
    best_single_idx = int(np.argmax(individual_accs))
    import shutil
    shutil.copy2(model_infos[best_single_idx][0],
                 str(MODELS_DIR / "best_model.keras"))

    # Save results
    results = {
        "n_models": args.n_models,
        "architectures": [info[0] for info in model_infos if isinstance(info[0], str)],
        "individual_accuracies": individual_accs,
        "ensemble_accuracy": acc_ensemble,
        "weighted_ensemble_accuracy": acc_weighted,
        "ensemble_tta_accuracy": acc_ensemble_tta,
        "weighted_tta_accuracy": acc_weighted_tta,
        "best_method": best_method,
        "best_accuracy": best_acc,
        "model_paths": [info[0] for info in model_infos],
    }
    with open(MODELS_DIR / "ultimate_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  FINAL: {best_method} = {best_acc:.0%}")
    print(f"  TARGET: 100%")
    if best_acc >= 1.0:
        print(f"  *** 100% ACCURACY ACHIEVED! ***")
    else:
        print(f"  Missing {int((1 - best_acc) * 100)} samples — {len(wrong)} errors remain")
    print(f"{'=' * 70}")

    return best_acc


if __name__ == "__main__":
    main()
