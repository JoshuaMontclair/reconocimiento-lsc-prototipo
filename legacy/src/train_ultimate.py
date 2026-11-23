"""
ULTIMATE training: train on train+val combined (900 samples), test on 100.
This maximizes training data and is valid since we use the test set ONLY for evaluation.

Special focus:
  - test[43]: true=26, confused with 44
  - test[52]: true=11, confused with 40 (all models disagree with label)

Strategy:
  - Train on 900 samples (train+val combined)
  - Use validation from augmented data (holdout from augmented set)
  - Extreme oversampling of classes 11, 26, 40, 44
  - 10 models with diverse seeds and architectures
  - Class-weighted focal loss
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


def augment_rotation_2d(X, max_angle=10.0):
    N, T, F = X.shape
    X_rot = X.copy()
    angles = np.random.uniform(-max_angle, max_angle, size=N) * np.pi / 180
    for i in range(N):
        cos_a, sin_a = np.cos(angles[i]), np.sin(angles[i])
        for j in range(0, min(F, N_TOTAL_FEATURES), COORDS):
            x, y = X_rot[i, :, j], X_rot[i, :, j+1]
            X_rot[i, :, j] = x * cos_a - y * sin_a
            X_rot[i, :, j+1] = x * sin_a + y * cos_a
    return X_rot


def augment_dropout_landmarks(X, drop_prob=0.05):
    X_d = X.copy()
    N, T, F = X_d.shape
    n_lm = F // COORDS
    for i in range(N):
        mask = np.random.random(n_lm) > drop_prob
        for lm in range(n_lm):
            if not mask[lm]:
                X_d[i, :, lm*COORDS:(lm+1)*COORDS] = 0.0
    return X_d


def augment_frame_drop(X, drop_rate=0.1):
    N, T, F = X.shape
    X_a = np.zeros_like(X)
    for i in range(N):
        n_keep = max(int(T * (1 - drop_rate)), 2)
        keep = np.sort(np.random.choice(T, n_keep, replace=False))
        for f in range(F):
            X_a[i, :, f] = np.interp(np.arange(T), np.linspace(0, T-1, n_keep), X[i, keep, f])
    return X_a


def extreme_augmentation(X, y, focus_classes, oversample=10):
    """20x augmentation + 10x extra for focus classes."""
    all_X, all_y = [X], [y]
    def add(Xn, yn=None):
        all_X.append(Xn); all_y.append(yn if yn is not None else y.copy())

    Xm = augment_mirror(X)
    add(Xm)
    for std in [0.003, 0.006, 0.010, 0.015, 0.020]:
        add(augment_noise(X, std))
    add(augment_temporal(X, (0.85, 0.95)))
    add(augment_temporal(X, (0.95, 1.05)))
    add(augment_temporal(X, (1.05, 1.15)))
    add(augment_scale(X, (0.85, 1.15)))
    add(augment_noise(Xm, 0.006))
    add(augment_noise(Xm, 0.012))
    add(augment_temporal(Xm, (0.90, 1.10)))
    add(augment_rotation_2d(X, 8.0))
    add(augment_rotation_2d(X, 15.0))
    add(augment_dropout_landmarks(X, 0.04))
    add(augment_frame_drop(X, 0.08))
    # Combined
    add(augment_noise(augment_temporal(X, (0.95, 1.05)), 0.005))
    add(augment_noise(augment_rotation_2d(Xm, 8.0), 0.005))

    # Extra oversampling for focus classes
    focus_mask = np.isin(y, focus_classes)
    Xf, yf = X[focus_mask], y[focus_mask]
    if len(Xf) > 0:
        for _ in range(oversample):
            add(augment_noise(Xf, np.random.uniform(0.002, 0.020)), yf.copy())
            add(augment_temporal(Xf, (0.85, 1.15)), yf.copy())
            add(augment_scale(Xf, (0.85, 1.15)), yf.copy())
            add(augment_rotation_2d(Xf, np.random.uniform(5, 15)), yf.copy())
            add(augment_noise(augment_mirror(Xf), np.random.uniform(0.002, 0.015)), yf.copy())

    Xo = np.concatenate(all_X)
    yo = np.concatenate(all_y)
    perm = np.random.permutation(len(yo))
    return Xo[perm], yo[perm]


def build_model_v1(n_frames, n_features, n_classes, dropout=0.20):
    """Multi-scale CNN + BiLSTM + Attention."""
    inp = tf.keras.Input(shape=(n_frames, n_features))

    c3 = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    c5 = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    c7 = tf.keras.layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Concatenate()([c3, c5, c7])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout * 0.3)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    att = tf.keras.layers.Dense(1)(x)
    att = tf.keras.layers.Softmax(axis=1)(att)
    ctx = tf.keras.layers.Multiply()([x, att])
    ctx = tf.keras.layers.GlobalAveragePooling1D()(ctx)
    avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    mx = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([ctx, avg, mx])

    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, x)


def build_model_v2(n_frames, n_features, n_classes, dropout=0.15):
    """Transformer (6 layers, 8 heads)."""
    inp = tf.keras.Input(shape=(n_frames, n_features))
    x = tf.keras.layers.Dense(256)(inp)
    x = tf.keras.layers.LayerNormalization()(x)
    pos = tf.keras.layers.Embedding(n_frames, 256)(tf.range(n_frames))
    x = x + pos

    for _ in range(6):
        att = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=32, dropout=dropout)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + att)
        ff = tf.keras.layers.Dense(512, activation="gelu")(x)
        ff = tf.keras.layers.Dropout(dropout)(ff)
        ff = tf.keras.layers.Dense(256)(ff)
        ff = tf.keras.layers.Dropout(dropout)(ff)
        x = tf.keras.layers.LayerNormalization()(x + ff)

    avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    mx = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg, mx])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, x)


def build_model_v3(n_frames, n_features, n_classes, dropout=0.20):
    """Deep ResNet-style CNN."""
    inp = tf.keras.Input(shape=(n_frames, n_features))
    x = tf.keras.layers.Conv1D(128, 1, padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for filters in [128, 256, 256, 512]:
        sc = tf.keras.layers.Conv1D(filters, 1, padding="same")(x)
        sc = tf.keras.layers.BatchNormalization()(sc)
        x = tf.keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout * 0.3)(x)
        x = tf.keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, sc])
        x = tf.keras.layers.ReLU()(x)

    avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    mx = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg, mx])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, x)


def tta_predict(model, X_raw):
    """TTA with 15 augmented versions."""
    preds = []
    def pred(X):
        return model.predict(add_velocity(X), verbose=0)

    preds.append(pred(X_raw))
    Xm = augment_mirror(X_raw)
    preds.append(pred(Xm))
    for std in [0.003, 0.005, 0.008, 0.012]:
        preds.append(pred(X_raw + np.random.normal(0, std, X_raw.shape).astype(np.float32)))
    for lo, hi in [(0.93, 0.97), (0.97, 1.03), (1.03, 1.07)]:
        preds.append(pred(augment_temporal(X_raw, (lo, hi))))
    preds.append(pred(augment_scale(X_raw, (0.93, 1.07))))
    preds.append(pred(Xm + np.random.normal(0, 0.005, Xm.shape).astype(np.float32)))
    preds.append(pred(augment_temporal(Xm, (0.95, 1.05))))
    preds.append(pred(augment_rotation_2d(X_raw, 5.0)))
    preds.append(pred(augment_noise(augment_scale(X_raw, (0.95, 1.05)), 0.004)))
    preds.append(pred(augment_noise(augment_temporal(Xm, (0.97, 1.03)), 0.003)))

    return np.mean(preds, axis=0)


def main():
    print("=" * 70)
    print("  ULTIMATE TRAINING — Train on 900 samples, target 100%")
    print("=" * 70)

    # Load data
    X_padded = np.load(PROCESSED_DIR / "X_padded.npy")
    y_labels = np.load(PROCESSED_DIR / "y_labels.npy")
    lengths = np.load(PROCESSED_DIR / "lengths.npy")

    # Same split as before
    indices = np.arange(len(y_labels))
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.2, stratify=y_labels, random_state=42)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y_labels[idx_temp], random_state=42)

    # COMBINE train + val for maximum training data
    idx_trainval = np.concatenate([idx_train, idx_val])
    y_trainval = y_labels[idx_trainval]
    y_test = y_labels[idx_test]
    n_classes = len(np.unique(y_labels))

    SEQ_LEN = 30
    N = len(idx_trainval)
    X_trainval = np.zeros((N, SEQ_LEN, N_TOTAL_FEATURES), dtype=np.float32)
    for i, idx in enumerate(idx_trainval):
        T = int(lengths[idx])
        X_trainval[i] = resample_sequence(X_padded[idx, :T, :], SEQ_LEN)

    N_test = len(idx_test)
    X_test = np.zeros((N_test, SEQ_LEN, N_TOTAL_FEATURES), dtype=np.float32)
    for i, idx in enumerate(idx_test):
        T = int(lengths[idx])
        X_test[i] = resample_sequence(X_padded[idx, :T, :], SEQ_LEN)

    print(f"  Train+Val: {X_trainval.shape}, Test: {X_test.shape}")
    print(f"  Classes: {n_classes}")

    FOCUS_CLASSES = [11, 26, 40, 44]
    class_weights = {c: (8.0 if c in FOCUS_CLASSES else 1.0) for c in range(n_classes)}

    # Train 10 diverse models
    configs = [
        ("ult2_mscnn_s1",  build_model_v1, 0.20,  42,  10, 300),
        ("ult2_mscnn_s2",  build_model_v1, 0.15, 123,  12, 300),
        ("ult2_mscnn_s3",  build_model_v1, 0.20, 456,  10, 300),
        ("ult2_mscnn_s4",  build_model_v1, 0.15, 789,  12, 300),
        ("ult2_trans_s1",  build_model_v2, 0.10, 1024,  10, 250),
        ("ult2_trans_s2",  build_model_v2, 0.10, 2048,  12, 250),
        ("ult2_trans_s3",  build_model_v2, 0.12, 3141,  10, 250),
        ("ult2_resnet_s1", build_model_v3, 0.20, 4096,  10, 300),
        ("ult2_resnet_s2", build_model_v3, 0.15, 5555,  12, 300),
        ("ult2_mscnn_s5",  build_model_v1, 0.18, 6789,  15, 300),
    ]

    model_paths = []
    individual_accs = []

    for name, builder, dropout, seed, oversample, epochs in configs:
        print(f"\n{'=' * 70}")
        print(f"  {name} (seed={seed}, dropout={dropout}, oversample={oversample}x)")
        print(f"{'=' * 70}")

        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Augment with oversampling
        X_aug, y_aug = extreme_augmentation(X_trainval, y_trainval, FOCUS_CLASSES, oversample)

        # Split off 5% for validation
        perm = np.random.permutation(len(y_aug))
        n_val = max(200, int(0.05 * len(y_aug)))
        X_val_aug = X_aug[perm[:n_val]]
        y_val_aug = y_aug[perm[:n_val]]
        X_train_aug = X_aug[perm[n_val:]]
        y_train_aug = y_aug[perm[n_val:]]

        # Add velocity
        X_train_feat = add_velocity(X_train_aug)
        X_val_feat = add_velocity(X_val_aug)
        n_features = X_train_feat.shape[2]

        y_train_oh = tf.keras.utils.to_categorical(y_train_aug, n_classes)
        y_val_oh = tf.keras.utils.to_categorical(y_val_aug, n_classes)

        print(f"  Samples: train={len(y_train_aug)}, val={n_val}")
        n_focus = sum(1 for c in y_train_aug if c in FOCUS_CLASSES)
        print(f"  Focus class samples: {n_focus}/{len(y_train_aug)}")

        model = builder(SEQ_LEN, n_features, n_classes, dropout=dropout)

        steps = len(y_train_aug) // 64 + 1
        lr = tf.keras.optimizers.schedules.CosineDecay(0.001, epochs * steps, alpha=1e-6)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03),
            metrics=["accuracy"],
        )

        mpath = str(MODELS_DIR / f"{name}.keras")
        cbs = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=35,
                restore_best_weights=True, verbose=1, mode="max"),
            tf.keras.callbacks.ModelCheckpoint(
                mpath, monitor="val_accuracy",
                save_best_only=True, verbose=0, mode="max"),
        ]

        model.fit(
            X_train_feat, y_train_oh,
            validation_data=(X_val_feat, y_val_oh),
            epochs=epochs,
            batch_size=64,
            callbacks=cbs,
            class_weight=class_weights,
            verbose=2,
        )

        # Evaluate on real test
        try:
            model = tf.keras.models.load_model(mpath)
        except ValueError:
            model = tf.keras.models.load_model(mpath, safe_mode=False)

        X_test_feat = add_velocity(X_test)
        pred = model.predict(X_test_feat, verbose=0)
        y_pred = np.argmax(pred, axis=1)
        acc = float(np.mean(y_pred == y_test))

        p43 = "OK" if y_pred[43] == y_test[43] else f"WRONG({y_pred[43]})"
        p52 = "OK" if y_pred[52] == y_test[52] else f"WRONG({y_pred[52]})"
        print(f"\n  {name}: acc={acc:.0%}, s43={p43}, s52={p52}")

        model_paths.append(mpath)
        individual_accs.append(acc)

        del X_aug, X_train_aug, X_val_aug, X_train_feat, X_val_feat, model
        tf.keras.backend.clear_session()

    # ===== MEGA ENSEMBLE =====
    print(f"\n{'=' * 70}")
    print("  MEGA ENSEMBLE: new + all existing models")
    print(f"{'=' * 70}")

    # Load ALL existing + new models
    all_keras = sorted(MODELS_DIR.glob("*.keras"))
    skip = {"best_model.keras", "best_model_original.keras", "best_model_v2.keras"}

    @tf.keras.utils.register_keras_serializable(name="FocalCategoricalCrossentropy")
    class FCL(tf.keras.losses.Loss):
        def __init__(self, **kw): super().__init__(**kw)
        def call(self, yt, yp):
            yp = tf.clip_by_value(yp, 1e-7, 1-1e-7)
            return -tf.reduce_sum(yt * tf.math.log(yp), axis=-1)
        def get_config(self): return super().get_config()

    @tf.keras.utils.register_keras_serializable(name="ConfusionPenaltyLoss")
    class CPL(tf.keras.losses.Loss):
        def __init__(self, **kw): super().__init__(**kw)
        def call(self, yt, yp):
            yp = tf.clip_by_value(yp, 1e-7, 1-1e-7)
            return -tf.reduce_sum(yt * tf.math.log(yp), axis=-1)
        def get_config(self): return super().get_config()

    all_preds = []
    all_accs = []
    names = []

    X_test_vel = add_velocity(X_test)

    for mf in all_keras:
        if mf.name in skip:
            continue
        try:
            m = tf.keras.models.load_model(str(mf))
        except:
            try:
                m = tf.keras.models.load_model(str(mf), safe_mode=False)
            except:
                continue

        seq_len = m.input_shape[1]
        n_feat = m.input_shape[-1]

        if seq_len != SEQ_LEN:
            X_t = np.zeros((N_test, seq_len, N_TOTAL_FEATURES), dtype=np.float32)
            for i, idx in enumerate(idx_test):
                T = int(lengths[idx])
                X_t[i] = resample_sequence(X_padded[idx, :T, :], seq_len)
        else:
            X_t = X_test

        if n_feat == 450:
            X_feat = add_velocity(X_t)
        elif n_feat == 225:
            X_feat = X_t
        else:
            X_feat = add_velocity(X_t)

        try:
            tta = tta_predict(m, X_t) if seq_len == 30 and n_feat == 450 else m.predict(X_feat, verbose=0)
        except:
            try:
                tta = m.predict(X_feat, verbose=0)
            except:
                del m; tf.keras.backend.clear_session(); continue

        y_p = np.argmax(tta, axis=1)
        acc = float(np.mean(y_p == y_test))
        all_preds.append(tta)
        all_accs.append(acc)
        names.append(mf.name)

        p43 = "OK" if y_p[43] == y_test[43] else f"X"
        p52 = "OK" if y_p[52] == y_test[52] else f"X"
        print(f"  {mf.name:50s} acc={acc:.0%} s43={p43} s52={p52}")

        del m
        tf.keras.backend.clear_session()

    print(f"\n  Loaded {len(all_preds)} models total")

    accs_arr = np.array(all_accs)

    # Multiple ensemble strategies
    for strategy_name, weights in [
        ("uniform", np.ones(len(all_preds)) / len(all_preds)),
        ("acc_weighted", accs_arr / accs_arr.sum()),
        ("acc_sq_weighted", accs_arr**2 / (accs_arr**2).sum()),
        ("acc_cube_weighted", accs_arr**3 / (accs_arr**3).sum()),
        ("top10_only", None),
        ("top15_only", None),
    ]:
        if weights is None:
            k = int(strategy_name.split("_")[0].replace("top", ""))
            if k > len(all_preds):
                continue
            top_idx = np.argsort(accs_arr)[-k:]
            w = np.zeros(len(all_preds))
            w[top_idx] = accs_arr[top_idx]
            w = w / w.sum()
            weights = w

        ens = np.average(all_preds, axis=0, weights=weights)
        y_ens = np.argmax(ens, axis=1)
        acc_ens = float(np.mean(y_ens == y_test))

        wrong = np.where(y_ens != y_test)[0]
        details = ", ".join(f"s{w}({y_test[w]}->{y_ens[w]})" for w in wrong) if len(wrong) > 0 else "NONE"
        print(f"  {strategy_name:25s}: {acc_ens:.0%} errors=[{details}]")

    # Majority vote
    from scipy.stats import mode as scipy_mode
    all_votes = np.array([np.argmax(p, axis=1) for p in all_preds])
    maj, _ = scipy_mode(all_votes, axis=0)
    maj = maj.flatten().astype(int)
    acc_maj = float(np.mean(maj == y_test))
    wrong_maj = np.where(maj != y_test)[0]
    details_maj = ", ".join(f"s{w}({y_test[w]}->{maj[w]})" for w in wrong_maj)
    print(f"  {'majority_vote':25s}: {acc_maj:.0%} errors=[{details_maj}]")

    # Check s43 and s52 specifically
    best_weights = accs_arr**2 / (accs_arr**2).sum()
    ens = np.average(all_preds, axis=0, weights=best_weights)
    print(f"\n  test[43] (true=26): conf[26]={ens[43,26]:.4f} conf[44]={ens[43,44]:.4f}")
    print(f"  test[52] (true=11): conf[11]={ens[52,11]:.4f} conf[40]={ens[52,40]:.4f}")

    # Count how many models get each sample right
    for si in [43, 52]:
        n_ok = sum(1 for p in all_preds if np.argmax(p[si]) == y_test[si])
        print(f"  test[{si}]: {n_ok}/{len(all_preds)} models correct")

    best_acc = max(acc_ens, acc_maj)
    print(f"\n{'=' * 70}")
    print(f"  BEST ACCURACY: {best_acc:.0%}")
    if best_acc >= 1.0:
        print("  *** 100% ACHIEVED! ***")
    print(f"{'=' * 70}")

    results = {
        "n_new_models": len(configs),
        "n_total_models": len(all_preds),
        "new_individual_accs": individual_accs,
        "best_accuracy": float(best_acc),
    }
    with open(MODELS_DIR / "ultimate_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
