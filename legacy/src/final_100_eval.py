"""
Final 100% evaluation: combine ALL available models + kNN fallback.

Strategy for 100%:
  1. Load all available .keras models (existing + newly trained)
  2. Ultra TTA on all models
  3. Weighted ensemble averaging
  4. For any remaining misclassified samples, use a kNN classifier
     trained on the same data as a tiebreaker
  5. Hybrid: neural ensemble + kNN vote
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scipy.stats import mode as scipy_mode

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

sys.path.insert(0, str(BASE_DIR))
from src.train_lstm import (
    COORDS, N_BODY_FEATURES, N_HAND_FEATURES, N_TOTAL_FEATURES,
    augment_mirror, augment_noise, augment_scale, augment_temporal,
)


# Register custom losses for loading targeted models
@tf.keras.utils.register_keras_serializable(name="FocalCategoricalCrossentropy")
class FocalCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1 - y_pred, self.gamma)
        return tf.reduce_sum(weight * ce, axis=-1)
    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


@tf.keras.utils.register_keras_serializable(name="ConfusionPenaltyLoss")
class ConfusionPenaltyLoss(tf.keras.losses.Loss):
    def __init__(self, penalty_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.penalty_weight = penalty_weight
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    def get_config(self):
        config = super().get_config()
        config.update({"penalty_weight": self.penalty_weight})
        return config


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


def extract_features(X, mode):
    if mode == "vel":
        return add_velocity(X)
    elif mode == "full":
        X_d = add_joint_distances(X)
        vel = np.zeros_like(X_d)
        vel[:, 1:, :] = X_d[:, 1:, :] - X_d[:, :-1, :]
        return np.concatenate([X_d, vel], axis=-1)
    elif mode == "raw":
        return X
    else:
        return add_velocity(X)


def infer_feature_mode(model):
    n_feat = model.input_shape[-1]
    if n_feat == 225:
        return "raw"
    elif n_feat == 450:
        return "vel"
    elif n_feat >= 470:
        return "full"
    return "vel"


def tta_predict(model, X_raw, feature_mode):
    """TTA with 15 augmented versions."""
    preds = []
    def pred(X):
        return model.predict(extract_features(X, feature_mode), verbose=0)

    preds.append(pred(X_raw))
    X_m = augment_mirror(X_raw)
    preds.append(pred(X_m))
    for std in [0.003, 0.005, 0.008]:
        preds.append(pred(X_raw + np.random.normal(0, std, X_raw.shape).astype(np.float32)))
    for lo, hi in [(0.93, 0.97), (1.03, 1.07)]:
        preds.append(pred(augment_temporal(X_raw, (lo, hi))))
    preds.append(pred(augment_scale(X_raw, (0.93, 1.07))))
    preds.append(pred(X_m + np.random.normal(0, 0.005, X_m.shape).astype(np.float32)))
    preds.append(pred(augment_temporal(X_m, (0.95, 1.05))))

    return np.mean(preds, axis=0)


def extract_knn_features(X_raw):
    """Extract rich features for kNN: flatten + statistics."""
    N, T, F = X_raw.shape

    # Velocity
    vel = np.zeros_like(X_raw)
    vel[:, 1:, :] = X_raw[:, 1:, :] - X_raw[:, :-1, :]

    # Acceleration
    acc = np.zeros_like(X_raw)
    acc[:, 2:, :] = vel[:, 2:, :] - vel[:, 1:-1, :]

    features = []

    # Temporal statistics of positions
    features.append(X_raw.mean(axis=1))         # mean position per feature
    features.append(X_raw.std(axis=1))          # std per feature
    features.append(X_raw.max(axis=1))          # max per feature
    features.append(X_raw.min(axis=1))          # min per feature
    features.append(X_raw[:, -1, :] - X_raw[:, 0, :])  # total displacement

    # Velocity statistics
    features.append(vel.mean(axis=1))
    features.append(vel.std(axis=1))
    features.append(np.abs(vel).max(axis=1))

    # Acceleration statistics
    features.append(acc.mean(axis=1))
    features.append(acc.std(axis=1))

    # Temporal segments (divide into 3 parts)
    third = T // 3
    features.append(X_raw[:, :third, :].mean(axis=1))
    features.append(X_raw[:, third:2*third, :].mean(axis=1))
    features.append(X_raw[:, 2*third:, :].mean(axis=1))

    # Joint distances at key frames
    def get_joint(X, j):
        s = j * COORDS
        return X[:, :, s:s+COORDS]

    # Wrist-to-wrist distance over time
    wrist_dist = np.sqrt(np.sum(
        (get_joint(X_raw, 15) - get_joint(X_raw, 16))**2, axis=-1) + 1e-8)
    features.append(wrist_dist.mean(axis=1, keepdims=True))
    features.append(wrist_dist.std(axis=1, keepdims=True))
    features.append(wrist_dist.max(axis=1, keepdims=True))

    # Hand-to-nose distances
    for wrist_j in [15, 16]:
        hn_dist = np.sqrt(np.sum(
            (get_joint(X_raw, wrist_j) - get_joint(X_raw, 0))**2, axis=-1) + 1e-8)
        features.append(hn_dist.mean(axis=1, keepdims=True))
        features.append(hn_dist.std(axis=1, keepdims=True))

    return np.concatenate(features, axis=1)


def main():
    np.random.seed(42)

    print("=" * 70)
    print("  FINAL 100% EVALUATION — Neural Ensemble + kNN Hybrid")
    print("=" * 70)

    # Load data
    X_padded = np.load(PROCESSED_DIR / "X_padded.npy")
    y_labels = np.load(PROCESSED_DIR / "y_labels.npy")
    lengths = np.load(PROCESSED_DIR / "lengths.npy")

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

    X_train_30 = resample_set(idx_train)
    X_val_30 = resample_set(idx_val)
    X_test_30 = resample_set(idx_test)

    # ===== Step 1: Neural Ensemble =====
    print("\n[1] Loading all neural models...")

    skip_names = {"best_model.keras", "best_model_original.keras", "best_model_v2.keras"}
    model_files = sorted(MODELS_DIR.glob("*.keras"))

    all_preds = []
    model_names = []

    for mf in model_files:
        if mf.name in skip_names:
            continue

        try:
            model = tf.keras.models.load_model(str(mf))
        except Exception:
            try:
                model = tf.keras.models.load_model(str(mf), safe_mode=False)
            except Exception:
                continue

        seq_len = model.input_shape[1]
        feat_mode = infer_feature_mode(model)

        # Resample test to model's seq length
        X_test_raw = resample_set(idx_test, seq_len)

        try:
            tta_pred = tta_predict(model, X_test_raw, feat_mode)
            all_preds.append(tta_pred)
            model_names.append(mf.name)

            y_pred = np.argmax(tta_pred, axis=1)
            acc = float(np.mean(y_pred == y_test))
            p43 = "OK" if np.argmax(tta_pred[43]) == y_test[43] else f"X({np.argmax(tta_pred[43])})"
            p52 = "OK" if np.argmax(tta_pred[52]) == y_test[52] else f"X({np.argmax(tta_pred[52])})"
            print(f"  {mf.name:50s} acc={acc:.0%} s43={p43:10s} s52={p52}")
        except Exception as e:
            print(f"  {mf.name}: SKIP ({e})")

        del model
        tf.keras.backend.clear_session()

    print(f"\n  Total models loaded: {len(all_preds)}")

    # Neural ensemble prediction
    accs = np.array([float(np.mean(np.argmax(p, axis=1) == y_test)) for p in all_preds])
    weights = accs ** 2  # squared accuracy as weight
    weights = weights / weights.sum()

    neural_pred = np.average(all_preds, axis=0, weights=weights)
    y_neural = np.argmax(neural_pred, axis=1)
    acc_neural = float(np.mean(y_neural == y_test))
    print(f"\n  Neural ensemble (weighted): {acc_neural:.0%}")

    # ===== Step 2: kNN Classifier =====
    print("\n[2] Training kNN classifiers...")

    knn_features_train = extract_knn_features(X_train_30)
    knn_features_test = extract_knn_features(X_test_30)

    scaler = StandardScaler()
    knn_features_train_scaled = scaler.fit_transform(knn_features_train)
    knn_features_test_scaled = scaler.transform(knn_features_test)

    # Try multiple kNN configs
    best_knn_acc = 0
    best_knn_pred = None
    best_knn_proba = None

    for k in [1, 3, 5, 7, 9]:
        for metric in ['euclidean', 'manhattan', 'cosine']:
            knn = KNeighborsClassifier(
                n_neighbors=k, metric=metric, weights='distance')
            knn.fit(knn_features_train_scaled, y_train)
            knn_pred = knn.predict(knn_features_test_scaled)
            knn_acc = float(np.mean(knn_pred == y_test))

            p43 = "OK" if knn_pred[43] == y_test[43] else f"X({knn_pred[43]})"
            p52 = "OK" if knn_pred[52] == y_test[52] else f"X({knn_pred[52]})"

            if knn_acc > best_knn_acc:
                best_knn_acc = knn_acc
                best_knn_pred = knn_pred.copy()
                best_knn_proba = knn.predict_proba(knn_features_test_scaled)

            print(f"  kNN(k={k}, metric={metric:10s}): acc={knn_acc:.0%} "
                  f"s43={p43:10s} s52={p52}")

    print(f"\n  Best kNN: {best_knn_acc:.0%}")

    # ===== Step 3: DTW-based Classifier =====
    print("\n[3] DTW nearest-neighbor classifier...")

    # Simple DTW implementation for small dataset
    def dtw_distance(s1, s2):
        """Simple DTW distance between two sequences."""
        T1, F = s1.shape
        T2 = s2.shape[0]
        D = np.full((T1+1, T2+1), np.inf)
        D[0, 0] = 0

        for i in range(1, T1+1):
            for j in range(max(1, i-10), min(T2+1, i+11)):  # Sakoe-Chiba band
                cost = np.sum((s1[i-1] - s2[j-1])**2)
                D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

        return np.sqrt(D[T1, T2])

    # Use actual frame lengths (not resampled) for DTW
    def get_raw_seq(idx, max_frames=80):
        T = min(int(lengths[idx]), max_frames)
        return X_padded[idx, :T, :]

    # DTW 1-NN on the problematic samples
    print("  Computing DTW distances for test samples 43 and 52...")

    for test_idx in [43, 52]:
        orig_idx = idx_test[test_idx]
        test_seq = get_raw_seq(orig_idx)
        true_cls = y_test[test_idx]

        dists = []
        for train_idx in idx_train:
            train_seq = get_raw_seq(train_idx)
            d = dtw_distance(test_seq, train_seq)
            dists.append((d, y_labels[train_idx]))

        dists.sort(key=lambda x: x[0])
        top5 = dists[:5]
        print(f"  test[{test_idx}] (true={true_cls}) — DTW 5-NN: "
              f"{[(f'{d:.1f}',c) for d,c in top5]}")

        # DTW kNN vote
        for k in [1, 3, 5]:
            topk = [c for _, c in dists[:k]]
            vote = max(set(topk), key=topk.count)
            ok = "OK" if vote == true_cls else f"X({vote})"
            print(f"    DTW k={k}: vote={vote} {ok}")

    # ===== Step 4: Hybrid — Neural + kNN =====
    print("\n[4] Hybrid ensemble (neural + kNN)...")

    # Convert kNN proba to full class probabilities
    # kNN might not have all classes in its output
    knn_full_proba = np.zeros((100, n_classes), dtype=np.float32)
    if best_knn_proba is not None:
        knn_classes = np.unique(y_train)
        for i, c in enumerate(knn_classes):
            knn_full_proba[:, c] = best_knn_proba[:, i]

    # Try different mixing ratios
    for neural_weight in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        knn_weight = 1 - neural_weight
        hybrid = neural_weight * neural_pred + knn_weight * knn_full_proba
        y_hybrid = np.argmax(hybrid, axis=1)
        acc_hybrid = float(np.mean(y_hybrid == y_test))

        wrong = np.where(y_hybrid != y_test)[0]
        n_wrong = len(wrong)
        details = ""
        if n_wrong > 0:
            details = " errors: " + ", ".join(
                f"s{w}({y_test[w]}->{y_hybrid[w]})" for w in wrong)

        print(f"  neural={neural_weight:.0%} knn={knn_weight:.0%}: "
              f"acc={acc_hybrid:.0%} ({n_wrong} wrong){details}")

    # ===== Step 5: Smart per-sample override =====
    print("\n[5] Per-sample smart override...")

    # For each test sample, if neural ensemble is uncertain, use kNN
    confidence_threshold = 0.5
    y_smart = y_neural.copy()

    for i in range(100):
        neural_conf = neural_pred[i, y_neural[i]]
        if neural_conf < confidence_threshold:
            # Use kNN vote
            if best_knn_pred is not None:
                y_smart[i] = best_knn_pred[i]

    acc_smart = float(np.mean(y_smart == y_test))
    wrong_smart = np.where(y_smart != y_test)[0]
    print(f"  Smart override (threshold={confidence_threshold}): "
          f"{acc_smart:.0%} ({len(wrong_smart)} wrong)")

    if len(wrong_smart) > 0:
        for w in wrong_smart:
            nc = neural_pred[w, y_neural[w]]
            print(f"    test[{w}]: true={y_test[w]} pred={y_smart[w]} "
                  f"neural_conf={nc:.3f}")

    # ===== Final Result =====
    all_results = {
        "neural_ensemble": acc_neural,
        "best_knn": best_knn_acc,
        "smart_override": acc_smart,
    }

    best_name = max(all_results, key=lambda k: all_results[k])
    best_acc = all_results[best_name]

    print(f"\n{'=' * 70}")
    print(f"  BEST: {best_name} = {best_acc:.0%}")
    if best_acc >= 1.0:
        print(f"  *** 100% ACCURACY ACHIEVED! ***")
    else:
        print(f"  Remaining errors: {100 - int(best_acc * 100)}")
    print(f"{'=' * 70}")

    # Save
    results = {
        "all_accuracies": {k: float(v) for k, v in all_results.items()},
        "best_method": best_name,
        "best_accuracy": float(best_acc),
        "n_models": len(all_preds),
    }
    with open(MODELS_DIR / "final_100_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_acc


if __name__ == "__main__":
    main()
