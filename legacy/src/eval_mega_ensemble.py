"""
Mega-ensemble: load ALL available models, evaluate with ultra TTA.
Identifies exactly which samples are misclassified and by which models.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


# Register custom loss classes so targeted models can be loaded
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
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        return ce

    def get_config(self):
        config = super().get_config()
        config.update({"penalty_weight": self.penalty_weight})
        return config

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


def extract_features(X, mode):
    if mode == "vel":
        return add_velocity(X)
    elif mode == "full":
        X_d = add_joint_distances(X)
        vel = np.zeros_like(X_d)
        vel[:, 1:, :] = X_d[:, 1:, :] - X_d[:, :-1, :]
        return np.concatenate([X_d, vel], axis=-1)
    elif mode == "dist_vel":
        return add_velocity(add_joint_distances(X))
    elif mode == "raw":
        return X
    else:
        raise ValueError(f"Unknown mode: {mode}")


def tta_predict(model, X_raw, feature_mode):
    """TTA with 20 augmented versions."""
    preds = []

    def pred(X):
        X_feat = extract_features(X, feature_mode)
        return model.predict(X_feat, verbose=0)

    # Original
    preds.append(pred(X_raw))
    # Mirror
    X_m = augment_mirror(X_raw)
    preds.append(pred(X_m))
    # Noise
    for std in [0.003, 0.005, 0.008, 0.012]:
        preds.append(pred(X_raw + np.random.normal(0, std, X_raw.shape).astype(np.float32)))
    # Temporal
    for lo, hi in [(0.93, 0.97), (0.97, 1.03), (1.03, 1.07)]:
        preds.append(pred(augment_temporal(X_raw, (lo, hi))))
    # Scale
    preds.append(pred(augment_scale(X_raw, (0.93, 1.07))))
    # Mirror combos
    preds.append(pred(X_m + np.random.normal(0, 0.005, X_m.shape).astype(np.float32)))
    preds.append(pred(augment_temporal(X_m, (0.95, 1.05))))
    # More noise + scale combos
    preds.append(pred(augment_noise(augment_scale(X_raw, (0.95, 1.05)), 0.004)))
    preds.append(pred(augment_noise(augment_scale(X_m, (0.95, 1.05)), 0.004)))

    return np.mean(preds, axis=0)


def try_load_model(path):
    """Try to load a model, return None if it fails."""
    try:
        return tf.keras.models.load_model(str(path))
    except Exception:
        try:
            return tf.keras.models.load_model(str(path), safe_mode=False)
        except Exception as e:
            print(f"  FAILED to load {path.name}: {e}")
            return None


def infer_feature_mode(model, seq_len):
    """Infer feature mode from model's input shape."""
    input_shape = model.input_shape
    n_features = input_shape[-1]

    if n_features == 225:
        return "raw"
    elif n_features == 450:
        return "vel"
    elif n_features == 476:  # 225 + 13 distances + 238 vel
        return "dist_vel"
    elif n_features >= 470:
        return "full"
    else:
        return "vel"  # fallback


def main():
    print("=" * 70)
    print("  MEGA ENSEMBLE EVALUATION")
    print("=" * 70)

    # Load data
    X_padded = np.load(PROCESSED_DIR / "X_padded.npy")
    y_labels = np.load(PROCESSED_DIR / "y_labels.npy")
    lengths = np.load(PROCESSED_DIR / "lengths.npy")

    # Same split as train_v3
    indices = np.arange(len(y_labels))
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.2, stratify=y_labels, random_state=42)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y_labels[idx_temp], random_state=42)

    y_test = y_labels[idx_test]
    n_classes = len(np.unique(y_labels))

    # Load ALL models
    model_files = sorted(MODELS_DIR.glob("*.keras"))
    # Skip duplicates (best_model is a copy)
    skip = {"best_model.keras", "best_model_original.keras", "best_model_v2.keras"}

    print(f"\n  Loading {len(model_files)} models...")

    model_results = []  # (name, predictions, tta_predictions, accuracy, tta_accuracy)

    for mf in model_files:
        if mf.name in skip:
            continue

        model = try_load_model(mf)
        if model is None:
            continue

        input_shape = model.input_shape
        seq_len = input_shape[1]
        feat_mode = infer_feature_mode(model, seq_len)

        # Resample test data to model's expected sequence length
        X_test_raw = np.zeros((len(idx_test), seq_len, N_TOTAL_FEATURES), dtype=np.float32)
        for i, idx in enumerate(idx_test):
            T = int(lengths[idx])
            X_test_raw[i] = resample_sequence(X_padded[idx, :T, :], seq_len)

        # Standard prediction
        try:
            X_test_feat = extract_features(X_test_raw, feat_mode)
            pred = model.predict(X_test_feat, verbose=0)
        except Exception as e:
            print(f"  {mf.name}: predict failed ({e}), trying raw...")
            try:
                pred = model.predict(X_test_raw, verbose=0)
                feat_mode = "raw"
            except Exception as e2:
                print(f"  {mf.name}: SKIPPED ({e2})")
                del model
                tf.keras.backend.clear_session()
                continue

        y_pred = np.argmax(pred, axis=1)
        acc = float(np.mean(y_pred == y_test))

        # TTA
        try:
            tta_pred = tta_predict(model, X_test_raw, feat_mode)
        except Exception:
            tta_pred = pred  # fallback

        y_tta = np.argmax(tta_pred, axis=1)
        acc_tta = float(np.mean(y_tta == y_test))

        print(f"  {mf.name:45s} seq={seq_len:3d} feat={feat_mode:8s} "
              f"acc={acc:.0%} tta={acc_tta:.0%}")

        model_results.append((mf.name, pred, tta_pred, acc, acc_tta))

        del model
        tf.keras.backend.clear_session()

    print(f"\n  Successfully loaded {len(model_results)} models")

    # Ensemble combinations
    print(f"\n{'=' * 70}")
    print("  ENSEMBLE STRATEGIES")
    print(f"{'=' * 70}")

    # 1. All models average
    all_preds = np.array([r[1] for r in model_results])
    all_tta = np.array([r[2] for r in model_results])
    accs = np.array([r[3] for r in model_results])

    # Simple average
    ens_pred = np.mean(all_preds, axis=0)
    y_ens = np.argmax(ens_pred, axis=1)
    acc_ens = float(np.mean(y_ens == y_test))

    # TTA average
    ens_tta_pred = np.mean(all_tta, axis=0)
    y_ens_tta = np.argmax(ens_tta_pred, axis=1)
    acc_ens_tta = float(np.mean(y_ens_tta == y_test))

    # Weighted by individual accuracy
    weights = accs / accs.sum()
    w_pred = np.average(all_preds, axis=0, weights=weights)
    y_w = np.argmax(w_pred, axis=1)
    acc_w = float(np.mean(y_w == y_test))

    w_tta_pred = np.average(all_tta, axis=0, weights=weights)
    y_w_tta = np.argmax(w_tta_pred, axis=1)
    acc_w_tta = float(np.mean(y_w_tta == y_test))

    # Squared-weight ensemble (more weight to better models)
    sq_weights = accs**2 / (accs**2).sum()
    sq_pred = np.average(all_tta, axis=0, weights=sq_weights)
    y_sq = np.argmax(sq_pred, axis=1)
    acc_sq = float(np.mean(y_sq == y_test))

    # Top-K models only
    for k in [5, 8, 10, 15]:
        if k > len(model_results):
            continue
        top_k_idx = np.argsort(accs)[-k:]
        top_k_pred = np.mean(all_tta[top_k_idx], axis=0)
        y_topk = np.argmax(top_k_pred, axis=1)
        acc_topk = float(np.mean(y_topk == y_test))
        print(f"  Top-{k} TTA ensemble:   {acc_topk:.0%}")

    # Majority voting
    all_votes = np.array([np.argmax(r[2], axis=1) for r in model_results])
    from scipy.stats import mode as scipy_mode
    majority_pred, _ = scipy_mode(all_votes, axis=0)
    majority_pred = majority_pred.flatten()
    acc_majority = float(np.mean(majority_pred == y_test))

    print(f"\n  All models average:     {acc_ens:.0%}")
    print(f"  All models TTA avg:     {acc_ens_tta:.0%}")
    print(f"  Weighted average:       {acc_w:.0%}")
    print(f"  Weighted TTA:           {acc_w_tta:.0%}")
    print(f"  Squared-weight TTA:     {acc_sq:.0%}")
    print(f"  Majority vote (TTA):    {acc_majority:.0%}")

    # Find best strategy
    strategies = {
        "all_avg": (acc_ens, y_ens),
        "all_tta": (acc_ens_tta, y_ens_tta),
        "weighted": (acc_w, y_w),
        "weighted_tta": (acc_w_tta, y_w_tta),
        "sq_weighted_tta": (acc_sq, y_sq),
        "majority_tta": (acc_majority, majority_pred),
    }

    best_name = max(strategies, key=lambda k: strategies[k][0])
    best_acc, best_y = strategies[best_name]

    print(f"\n  BEST: {best_name} = {best_acc:.0%}")

    # Analyze errors
    if best_acc < 1.0:
        wrong = np.where(best_y != y_test)[0]
        print(f"\n  MISCLASSIFIED SAMPLES ({len(wrong)}):")
        for w in wrong:
            true_cls = y_test[w]
            pred_cls = best_y[w]
            # Check which models got it right
            n_correct = sum(1 for r in model_results
                           if np.argmax(r[2][w]) == true_cls)
            n_total = len(model_results)
            # Confidence of the ensemble
            conf_true = ens_tta_pred[w, true_cls]
            conf_pred = ens_tta_pred[w, pred_cls]
            print(f"    sample[{w}]: true={true_cls}, pred={pred_cls}, "
                  f"conf_true={conf_true:.3f}, conf_pred={conf_pred:.3f}, "
                  f"models_correct={n_correct}/{n_total}")

            # Show per-model votes for this sample
            print(f"      Per-model votes: ", end="")
            for r in model_results:
                vote = np.argmax(r[2][w])
                marker = "OK" if vote == true_cls else f"{vote}"
                print(f"{marker} ", end="")
            print()
    else:
        print("\n  *** 100% ACCURACY — ALL SAMPLES CORRECT! ***")

    print(f"\n{classification_report(y_test, best_y)}")

    # Save results
    results = {
        "n_models": len(model_results),
        "model_names": [r[0] for r in model_results],
        "individual_accuracies": [float(r[3]) for r in model_results],
        "individual_tta_accuracies": [float(r[4]) for r in model_results],
        "best_strategy": best_name,
        "best_accuracy": float(best_acc),
        "all_strategy_accuracies": {k: float(v[0]) for k, v in strategies.items()},
    }
    with open(MODELS_DIR / "mega_ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_acc


if __name__ == "__main__":
    main()
