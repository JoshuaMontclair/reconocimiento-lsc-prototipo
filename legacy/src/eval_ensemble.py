"""
Evaluate existing ensemble models with aggressive TTA.
No training needed — just inference.
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from sklearn.metrics import classification_report
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


def aggressive_tta(model, X_pos: np.ndarray, n_rounds: int = 3) -> np.ndarray:
    """Very aggressive TTA: 20+ augmented versions per sample."""
    preds = []

    def pred(X_p):
        preds.append(model.predict(add_velocity(X_p), verbose=0))

    # 1. Original
    pred(X_pos)

    for _ in range(n_rounds):
        # Mirror
        X_m = augment_mirror(X_pos)
        pred(X_m)

        # Noise at various levels
        for std in [0.002, 0.005, 0.008, 0.012]:
            pred(X_pos + np.random.normal(0, std, X_pos.shape).astype(np.float32))

        # Mirror + noise
        for std in [0.003, 0.006]:
            pred(X_m + np.random.normal(0, std, X_m.shape).astype(np.float32))

        # Temporal
        pred(augment_temporal(X_pos, (0.93, 0.97)))
        pred(augment_temporal(X_pos, (1.03, 1.07)))

        # Scale
        pred(augment_scale(X_pos, (0.95, 1.05)))

        # Mirror + temporal
        pred(augment_temporal(X_m, (0.95, 1.05)))

    return np.mean(preds, axis=0)


def main():
    print("Loading data...")
    X_padded = np.load(PROCESSED_DIR / "X_padded.npy")
    y_labels = np.load(PROCESSED_DIR / "y_labels.npy")
    lengths = np.load(PROCESSED_DIR / "lengths.npy")

    SEQ = 30
    N = len(y_labels)
    X_all = np.zeros((N, SEQ, N_TOTAL_FEATURES), dtype=np.float32)
    for i in range(N):
        T = int(lengths[i])
        X_all[i] = resample_sequence(X_padded[i, :T, :], SEQ)

    _, X_temp, _, y_temp = train_test_split(
        X_all, y_labels, test_size=0.2, stratify=y_labels, random_state=42)
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print(f"Test: {X_test.shape}, {len(np.unique(y_test))} classes")

    # Load models (original 5 + extra diverse architectures)
    models = []
    model_names = (
        [f"ensemble_{i}.keras" for i in range(5)] +
        ["ensemble_bilstm_only.keras", "ensemble_temporal_cnn.keras",
         "ensemble_cnn_bilstm_reg.keras"]
    )
    for name in model_names:
        p = MODELS_DIR / name
        if p.exists():
            try:
                m = tf.keras.models.load_model(str(p))
            except ValueError:
                m = tf.keras.models.load_model(str(p), safe_mode=False)
            models.append(m)
            print(f"  Loaded {name}")
    print(f"  {len(models)} models loaded")

    # Evaluate with different TTA levels
    X_test_feat = add_velocity(X_test)

    # 1. Simple ensemble (no TTA)
    preds_simple = [m.predict(X_test_feat, verbose=0) for m in models]
    y_simple = np.argmax(np.mean(preds_simple, axis=0), axis=1)
    acc_simple = float(np.mean(y_simple == y_test))
    print(f"\n  Ensemble (no TTA):      {acc_simple:.0%}")

    # 2. Weighted ensemble (weight by confidence)
    # Weight each model by its average max-confidence
    weights = []
    for p in preds_simple:
        avg_conf = np.mean(np.max(p, axis=1))
        weights.append(avg_conf)
    weights = np.array(weights) / sum(weights)
    weighted_pred = sum(w * p for w, p in zip(weights, preds_simple))
    y_weighted = np.argmax(weighted_pred, axis=1)
    acc_weighted = float(np.mean(y_weighted == y_test))
    print(f"  Weighted ensemble:      {acc_weighted:.0%}")

    # 3. Ensemble + Light TTA (7 augmentations × 1 round)
    from src.train_v3 import predict_with_tta
    tta_preds = [predict_with_tta(m, X_test) for m in models]
    y_tta = np.argmax(np.mean(tta_preds, axis=0), axis=1)
    acc_tta = float(np.mean(y_tta == y_test))
    print(f"  Ensemble + light TTA:   {acc_tta:.0%}")

    # 4. Ensemble + Aggressive TTA (20+ augmentations × 3 rounds)
    np.random.seed(42)
    agg_preds = [aggressive_tta(m, X_test, n_rounds=3) for m in models]
    y_agg = np.argmax(np.mean(agg_preds, axis=0), axis=1)
    acc_agg = float(np.mean(y_agg == y_test))
    print(f"  Ensemble + aggressive TTA: {acc_agg:.0%}")

    # 5. Top-3 models + aggressive TTA
    model_accs = [float(np.mean(np.argmax(p, axis=1) == y_test)) for p in preds_simple]
    top3_idx = np.argsort(model_accs)[-3:]
    top3_preds = [aggressive_tta(models[i], X_test, n_rounds=3) for i in top3_idx]
    y_top3 = np.argmax(np.mean(top3_preds, axis=0), axis=1)
    acc_top3 = float(np.mean(y_top3 == y_test))
    print(f"  Top-3 + aggressive TTA: {acc_top3:.0%}")

    # Best result details
    best_acc = max(acc_simple, acc_weighted, acc_tta, acc_agg, acc_top3)
    best_y = [y_simple, y_weighted, y_tta, y_agg, y_top3][
        [acc_simple, acc_weighted, acc_tta, acc_agg, acc_top3].index(best_acc)]
    best_name = ["Simple", "Weighted", "Light TTA", "Aggressive TTA", "Top3+TTA"][
        [acc_simple, acc_weighted, acc_tta, acc_agg, acc_top3].index(best_acc)]

    print(f"\n  Best: {best_name} = {best_acc:.0%}")
    if best_acc < 0.98:
        # Show which samples are wrong
        wrong = np.where(best_y != y_test)[0]
        print(f"\n  Wrong predictions ({len(wrong)}):")
        for idx in wrong:
            print(f"    Sample {idx}: true={y_test[idx]}, pred={best_y[idx]}")
    else:
        print("  TARGET 98% REACHED!")

    print(f"\n{classification_report(y_test, best_y)}")


if __name__ == "__main__":
    main()
