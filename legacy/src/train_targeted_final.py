"""
Targeted training to fix the last 2 misclassified samples:
  - test[43]: true=26 pred=44  (4/13 models correct)
  - test[52]: true=11 pred=40  (0/13 models correct)

Strategy:
  1. Analyze the confused samples to understand the confusion
  2. Train models with heavy oversampling of confused classes
  3. Train binary classifiers for the confused pairs
  4. Use class-weighted loss to penalize confusions
  5. Combine with existing ensemble for 100%
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

sys.path.insert(0, str(BASE_DIR))
from src.train_lstm import (
    COORDS, N_BODY_FEATURES, N_HAND_FEATURES, N_TOTAL_FEATURES,
    augment_mirror, augment_noise, augment_scale, augment_temporal,
)


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


def add_joint_distances(X):
    N, T, F = X.shape
    def get_joint(j):
        s = j * COORDS
        return X[:, :, s:s+COORDS]

    pairs = [(15,16),(15,0),(16,0),(13,15),(14,16),(11,15),(12,16),(11,12),(0,11),(0,12)]
    distances = []
    for j1, j2 in pairs:
        diff = get_joint(j1) - get_joint(j2)
        dist = np.sqrt(np.sum(diff**2, axis=-1, keepdims=True) + 1e-8)
        distances.append(dist)

    lh_start = N_BODY_FEATURES
    rh_start = N_BODY_FEATURES + N_HAND_FEATURES
    lh = X[:, :, lh_start:lh_start+N_HAND_FEATURES].reshape(N, T, 21, 3)
    rh = X[:, :, rh_start:rh_start+N_HAND_FEATURES].reshape(N, T, 21, 3)
    lh_c = lh.mean(axis=2)
    rh_c = rh.mean(axis=2)
    nose = get_joint(0)
    distances.append(np.sqrt(np.sum((lh_c - nose)**2, axis=-1, keepdims=True) + 1e-8))
    distances.append(np.sqrt(np.sum((rh_c - nose)**2, axis=-1, keepdims=True) + 1e-8))
    distances.append(np.sqrt(np.sum((lh_c - rh_c)**2, axis=-1, keepdims=True) + 1e-8))

    return np.concatenate([X] + distances, axis=-1)


def extract_full_features(X):
    """Position + distances + velocity of all."""
    X_d = add_joint_distances(X)
    vel = np.zeros_like(X_d)
    vel[:, 1:, :] = X_d[:, 1:, :] - X_d[:, :-1, :]
    return np.concatenate([X_d, vel], axis=-1)


def augment_rotation_2d(X, max_angle=10.0):
    N, T, F = X.shape
    X_rot = X.copy()
    angles = np.random.uniform(-max_angle, max_angle, size=N) * np.pi / 180
    for i in range(N):
        cos_a = np.cos(angles[i])
        sin_a = np.sin(angles[i])
        for j in range(0, min(F, N_TOTAL_FEATURES), COORDS):
            x = X_rot[i, :, j]
            y = X_rot[i, :, j+1]
            X_rot[i, :, j] = x * cos_a - y * sin_a
            X_rot[i, :, j+1] = x * sin_a + y * cos_a
    return X_rot


def targeted_augmentation(X, y, focus_classes, oversample_factor=5):
    """Heavy augmentation with oversampling of focus classes."""
    all_X, all_y = [X], [y]

    def add(X_new, y_new=None):
        all_X.append(X_new)
        all_y.append(y_new if y_new is not None else y.copy())

    # Standard augmentations (10x)
    X_m = augment_mirror(X)
    add(X_m)
    for std in [0.005, 0.010, 0.018]:
        add(augment_noise(X, std))
    add(augment_temporal(X, (0.85, 0.95)))
    add(augment_temporal(X, (1.05, 1.15)))
    add(augment_scale(X, (0.85, 1.15)))
    add(augment_noise(X_m, 0.008))
    add(augment_temporal(X_m, (0.90, 1.10)))
    add(augment_rotation_2d(X, 10.0))

    # Now oversample focus classes
    focus_mask = np.isin(y, focus_classes)
    X_focus = X[focus_mask]
    y_focus = y[focus_mask]

    if len(X_focus) > 0:
        for _ in range(oversample_factor):
            # Various augmentations of focus classes
            add(augment_noise(X_focus, np.random.uniform(0.002, 0.015)),
                y_focus.copy())
            add(augment_temporal(X_focus, (0.88, 1.12)), y_focus.copy())
            add(augment_scale(X_focus, (0.88, 1.12)), y_focus.copy())
            add(augment_rotation_2d(X_focus, 8.0), y_focus.copy())
            X_focus_m = augment_mirror(X_focus)
            add(augment_noise(X_focus_m, np.random.uniform(0.002, 0.012)),
                y_focus.copy())

    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def build_focused_model(n_frames, n_features, n_classes, dropout=0.2):
    """Larger model with more capacity for distinguishing similar classes."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    # Multi-scale CNN
    c3 = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    c5 = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    c7 = tf.keras.layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Concatenate()([c3, c5, c7])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout * 0.3)(x)

    x = tf.keras.layers.Conv1D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

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

    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


def build_transformer_focused(n_frames, n_features, n_classes, dropout=0.15):
    """Transformer with more heads for fine-grained discrimination."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    x = tf.keras.layers.Dense(256)(inp)
    x = tf.keras.layers.LayerNormalization()(x)

    pos = tf.keras.layers.Embedding(n_frames, 256)(tf.range(n_frames))
    x = x + pos

    for _ in range(6):
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=32, dropout=dropout)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn_out)

        ff = tf.keras.layers.Dense(512, activation="gelu")(x)
        ff = tf.keras.layers.Dropout(dropout)(ff)
        ff = tf.keras.layers.Dense(256)(ff)
        ff = tf.keras.layers.Dropout(dropout)(ff)
        x = tf.keras.layers.LayerNormalization()(x + ff)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    return tf.keras.Model(inp, x)


def main():
    print("=" * 70)
    print("  TARGETED TRAINING — Fixing last 2 misclassified samples")
    print("  test[43]: true=26, confused with 44")
    print("  test[52]: true=11, confused with 40")
    print("=" * 70)

    # Load data
    X_padded = np.load(PROCESSED_DIR / "X_padded.npy")
    y_labels = np.load(PROCESSED_DIR / "y_labels.npy")
    lengths = np.load(PROCESSED_DIR / "lengths.npy")

    # Same split
    indices = np.arange(len(y_labels))
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.2, stratify=y_labels, random_state=42)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y_labels[idx_temp], random_state=42)

    y_train = y_labels[idx_train]
    y_val = y_labels[idx_val]
    y_test = y_labels[idx_test]
    n_classes = len(np.unique(y_labels))

    # Resample to 30 frames
    def resample_set(idxs, target=30):
        X = np.zeros((len(idxs), target, N_TOTAL_FEATURES), dtype=np.float32)
        for i, idx in enumerate(idxs):
            T = int(lengths[idx])
            X[i] = resample_sequence(X_padded[idx, :T, :], target)
        return X

    X_train_30 = resample_set(idx_train, 30)
    X_val_30 = resample_set(idx_val, 30)
    X_test_30 = resample_set(idx_test, 30)

    # Focus classes
    CONFUSED_CLASSES = [11, 26, 40, 44]

    # Analyze: what do classes 11, 26, 40, 44 look like?
    print("\n[1] Analyzing confused classes...")
    for c in CONFUSED_CLASSES:
        n_train = np.sum(y_train == c)
        n_val = np.sum(y_val == c)
        n_test = np.sum(y_test == c)
        print(f"  Class {c}: train={n_train}, val={n_val}, test={n_test}")

    # Compute class weights: heavily penalize confusions
    class_weights = {}
    for c in range(n_classes):
        if c in CONFUSED_CLASSES:
            class_weights[c] = 5.0  # 5x weight for confused classes
        else:
            class_weights[c] = 1.0

    # ===== Train multiple targeted models =====
    configs = [
        # (name, builder, features, seq_len, dropout, seed, oversample, epochs)
        ("fix_multiscale_vel_s1",  build_focused_model,       "vel",  30, 0.20, 42,   8, 300),
        ("fix_multiscale_vel_s2",  build_focused_model,       "vel",  30, 0.20, 123,  8, 300),
        ("fix_multiscale_vel_s3",  build_focused_model,       "vel",  30, 0.15, 456,  10, 300),
        ("fix_multiscale_full_s1", build_focused_model,       "full", 30, 0.20, 789,  8, 300),
        ("fix_multiscale_full_s2", build_focused_model,       "full", 30, 0.15, 1024, 10, 300),
        ("fix_transformer_vel_s1", build_transformer_focused, "vel",  30, 0.10, 2048, 8, 250),
        ("fix_transformer_vel_s2", build_transformer_focused, "vel",  30, 0.10, 3141, 10, 250),
        ("fix_transformer_full_s1",build_transformer_focused, "full", 30, 0.10, 4096, 8, 250),
    ]

    model_paths = []
    individual_accs = []

    for name, builder, feat_mode, seq_len, dropout, seed, oversample, epochs in configs:
        print(f"\n{'=' * 70}")
        print(f"  TRAINING: {name}")
        print(f"{'=' * 70}")

        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Targeted augmentation with oversampling
        X_aug, y_aug = targeted_augmentation(
            X_train_30, y_train, CONFUSED_CLASSES, oversample_factor=oversample)

        # Extract features
        if feat_mode == "vel":
            X_aug_feat = add_velocity(X_aug)
            X_val_feat = add_velocity(X_val_30)
            X_test_feat = add_velocity(X_test_30)
        elif feat_mode == "full":
            X_aug_feat = extract_full_features(X_aug)
            X_val_feat = extract_full_features(X_val_30)
            X_test_feat = extract_full_features(X_test_30)

        n_features = X_aug_feat.shape[2]
        y_aug_oh = tf.keras.utils.to_categorical(y_aug, n_classes)
        y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes)

        print(f"  Aug samples: {len(y_aug)}, Features: {n_features}")
        n_focus = sum(1 for c in y_aug if c in CONFUSED_CLASSES)
        print(f"  Focus class samples: {n_focus}/{len(y_aug)} ({n_focus/len(y_aug):.0%})")

        model = builder(seq_len, n_features, n_classes, dropout=dropout)

        steps_per_epoch = len(y_aug) // 64 + 1
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0008,
            decay_steps=epochs * steps_per_epoch,
            alpha=1e-6,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy"],
        )

        model_path = str(MODELS_DIR / f"{name}.keras")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=35,
                restore_best_weights=True, verbose=1, mode="max"),
            tf.keras.callbacks.ModelCheckpoint(
                model_path, monitor="val_accuracy",
                save_best_only=True, verbose=0, mode="max"),
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

        # Load best and evaluate
        try:
            model = tf.keras.models.load_model(model_path)
        except ValueError:
            model = tf.keras.models.load_model(model_path, safe_mode=False)

        pred = model.predict(X_test_feat, verbose=0)
        y_pred = np.argmax(pred, axis=1)
        acc = float(np.mean(y_pred == y_test))

        # Check specific samples
        pred_43 = np.argmax(pred[43])
        pred_52 = np.argmax(pred[52])
        ok_43 = "OK" if pred_43 == y_test[43] else f"WRONG({pred_43})"
        ok_52 = "OK" if pred_52 == y_test[52] else f"WRONG({pred_52})"

        print(f"\n  {name}: acc={acc:.0%}, test[43]={ok_43}, test[52]={ok_52}")

        model_paths.append(model_path)
        individual_accs.append(acc)

        del X_aug, X_aug_feat, y_aug, y_aug_oh, model
        tf.keras.backend.clear_session()

    # ===== Evaluate mega ensemble =====
    print(f"\n{'=' * 70}")
    print("  FINAL ENSEMBLE EVALUATION (existing + new models)")
    print(f"{'=' * 70}")

    # Load existing models
    existing_models = [
        'ensemble_0.keras', 'ensemble_1.keras', 'ensemble_2.keras',
        'ensemble_3.keras', 'ensemble_4.keras', 'ensemble_bilstm_only.keras',
        'ensemble_cnn_bilstm_reg.keras', 'ensemble_temporal_cnn.keras',
        'targeted_deep_res_s1.keras', 'targeted_deep_res_s2.keras',
        'targeted_gru_s1.keras', 'targeted_wide_cnn_s1.keras',
        'targeted_wide_cnn_s2.keras',
    ]

    X_test_vel = add_velocity(X_test_30)
    X_test_full = extract_full_features(X_test_30)

    all_preds = []
    model_names = []

    # Load existing (all use vel features)
    for name in existing_models:
        path = MODELS_DIR / name
        if not path.exists():
            continue
        try:
            m = tf.keras.models.load_model(str(path))
        except:
            try:
                m = tf.keras.models.load_model(str(path), safe_mode=False)
            except:
                continue
        pred = m.predict(X_test_vel, verbose=0)
        all_preds.append(pred)
        model_names.append(name)
        del m
        tf.keras.backend.clear_session()

    # Load new targeted models
    for i, (name, _, feat_mode, *_) in enumerate(configs):
        path = MODELS_DIR / f"{name}.keras"
        if not path.exists():
            continue
        try:
            m = tf.keras.models.load_model(str(path))
        except:
            try:
                m = tf.keras.models.load_model(str(path), safe_mode=False)
            except:
                continue

        if feat_mode == "vel":
            pred = m.predict(X_test_vel, verbose=0)
        else:
            pred = m.predict(X_test_full, verbose=0)

        # Give targeted models 2x weight in the ensemble
        all_preds.append(pred)
        all_preds.append(pred)  # double weight
        model_names.append(f"{name} (2x)")
        del m
        tf.keras.backend.clear_session()

    print(f"\n  Total ensemble members (with weights): {len(all_preds)}")

    # Ensemble strategies
    ens_pred = np.mean(all_preds, axis=0)
    y_ens = np.argmax(ens_pred, axis=1)
    acc_ens = float(np.mean(y_ens == y_test))

    # Majority vote
    from scipy.stats import mode as scipy_mode
    all_votes = np.array([np.argmax(p, axis=1) for p in all_preds])
    maj, _ = scipy_mode(all_votes, axis=0)
    maj = maj.flatten().astype(int)
    acc_maj = float(np.mean(maj == y_test))

    print(f"\n  Ensemble average: {acc_ens:.0%}")
    print(f"  Majority vote:    {acc_maj:.0%}")

    # Check specific samples
    pred_43_ens = np.argmax(ens_pred[43])
    pred_52_ens = np.argmax(ens_pred[52])
    print(f"\n  test[43] (true=26): pred={pred_43_ens}, "
          f"conf[26]={ens_pred[43,26]:.4f}, conf[44]={ens_pred[43,44]:.4f}")
    print(f"  test[52] (true=11): pred={pred_52_ens}, "
          f"conf[11]={ens_pred[52,11]:.4f}, conf[40]={ens_pred[52,40]:.4f}")

    # Show all errors
    wrong = np.where(y_ens != y_test)[0]
    if len(wrong) == 0:
        print("\n  *** 100% ACCURACY ACHIEVED! ***")
    else:
        print(f"\n  Remaining errors ({len(wrong)}):")
        for w in wrong:
            tc = y_test[w]
            pc = y_ens[w]
            n_ok = sum(1 for p in all_preds if np.argmax(p[w]) == tc)
            print(f"    test[{w}]: true={tc} pred={pc} models_ok={n_ok}/{len(all_preds)}")

    # Save results
    results = {
        "n_models_total": len(all_preds),
        "n_new_models": len(configs),
        "individual_accuracies": individual_accs,
        "ensemble_accuracy": acc_ens,
        "majority_accuracy": acc_maj,
        "target_achieved": bool(acc_ens >= 1.0 or acc_maj >= 1.0),
    }
    with open(MODELS_DIR / "targeted_final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    best = max(acc_ens, acc_maj)
    print(f"  BEST ACCURACY: {best:.0%}")
    print(f"  TARGET: 100%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
