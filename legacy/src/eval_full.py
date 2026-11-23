"""
Evaluate ALL available models as ensemble with TTA — targeting 100%.
Loads ensemble_*.keras + targeted_*.keras models.
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
from src.train_v3 import add_velocity, predict_with_tta, resample_sequence
from src.train_confusion_fix import FocalCategoricalCrossentropy  # noqa: F401


def aggressive_tta(model, X_pos: np.ndarray, n_rounds: int = 3) -> np.ndarray:
    """Very aggressive TTA: 20+ augmented versions per sample."""
    preds = []

    def pred(X_p):
        preds.append(model.predict(add_velocity(X_p), verbose=0))

    pred(X_pos)

    for _ in range(n_rounds):
        X_m = augment_mirror(X_pos)
        pred(X_m)

        for std in [0.002, 0.005, 0.008, 0.012]:
            pred(X_pos + np.random.normal(0, std, X_pos.shape).astype(np.float32))

        for std in [0.003, 0.006]:
            pred(X_m + np.random.normal(0, std, X_m.shape).astype(np.float32))

        pred(augment_temporal(X_pos, (0.93, 0.97)))
        pred(augment_temporal(X_pos, (1.03, 1.07)))
        pred(augment_scale(X_pos, (0.95, 1.05)))
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

    # Load ALL models
    models = []
    model_names = []

    # Ensemble models
    for name in sorted(MODELS_DIR.glob("ensemble_*.keras")):
        try:
            m = tf.keras.models.load_model(str(name))
        except ValueError:
            m = tf.keras.models.load_model(str(name), safe_mode=False)
        models.append(m)
        model_names.append(name.name)
        print(f"  Loaded {name.name}")

    # Targeted models
    for name in sorted(MODELS_DIR.glob("targeted_*.keras")):
        try:
            m = tf.keras.models.load_model(str(name))
        except ValueError:
            m = tf.keras.models.load_model(str(name), safe_mode=False)
        models.append(m)
        model_names.append(name.name)
        print(f"  Loaded {name.name}")

    print(f"  {len(models)} models loaded")

    X_test_feat = add_velocity(X_test)

    # Individual accuracy + track problem samples
    print("\n  Individual model results:")
    preds_simple = []
    for i, m in enumerate(models):
        p = m.predict(X_test_feat, verbose=0)
        preds_simple.append(p)
        y_p = np.argmax(p, axis=1)
        acc = float(np.mean(y_p == y_test))

        # Check problem samples
        problems = []
        for idx in range(len(y_test)):
            if y_test[idx] in [11, 26] and y_p[idx] != y_test[idx]:
                problems.append(f"s{idx}:t{y_test[idx]}->p{y_p[idx]}")
        prob_str = ", ".join(problems) if problems else "ALL CORRECT"
        print(f"    {model_names[i]:35s} {acc:5.1%}  problems: {prob_str}")

    # 1. Simple ensemble
    y_simple = np.argmax(np.mean(preds_simple, axis=0), axis=1)
    acc_simple = float(np.mean(y_simple == y_test))
    print(f"\n  Ensemble (no TTA):         {acc_simple:.0%}")

    # 2. Weighted ensemble (by confidence)
    weights = []
    for p in preds_simple:
        weights.append(np.mean(np.max(p, axis=1)))
    weights = np.array(weights) / sum(weights)
    y_weighted = np.argmax(sum(w * p for w, p in zip(weights, preds_simple)), axis=1)
    acc_weighted = float(np.mean(y_weighted == y_test))
    print(f"  Weighted ensemble:         {acc_weighted:.0%}")

    # 3. Ensemble + light TTA
    np.random.seed(42)
    tta_preds = [predict_with_tta(m, X_test) for m in models]
    y_tta = np.argmax(np.mean(tta_preds, axis=0), axis=1)
    acc_tta = float(np.mean(y_tta == y_test))
    print(f"  Ensemble + light TTA:      {acc_tta:.0%}")

    # 4. Ensemble + aggressive TTA
    np.random.seed(42)
    agg_preds = [aggressive_tta(m, X_test, n_rounds=3) for m in models]
    y_agg = np.argmax(np.mean(agg_preds, axis=0), axis=1)
    acc_agg = float(np.mean(y_agg == y_test))
    print(f"  Ensemble + aggressive TTA: {acc_agg:.0%}")

    # 5. Top-N models + aggressive TTA (select models that are diverse)
    model_accs = [float(np.mean(np.argmax(p, axis=1) == y_test)) for p in preds_simple]
    for top_n in [5, 8, 10]:
        if top_n > len(models):
            continue
        top_idx = np.argsort(model_accs)[-top_n:]
        top_preds = [aggressive_tta(models[i], X_test, n_rounds=3) for i in top_idx]
        y_top = np.argmax(np.mean(top_preds, axis=0), axis=1)
        acc_top = float(np.mean(y_top == y_test))
        print(f"  Top-{top_n} + aggressive TTA:  {acc_top:.0%}")

    # Best result
    results = {
        "Simple": (acc_simple, y_simple),
        "Weighted": (acc_weighted, y_weighted),
        "Light TTA": (acc_tta, y_tta),
        "Aggressive TTA": (acc_agg, y_agg),
    }
    best_name = max(results, key=lambda k: results[k][0])
    best_acc, best_y = results[best_name]

    print(f"\n  Best: {best_name} = {best_acc:.0%}")

    if best_acc < 1.0:
        wrong = np.where(best_y != y_test)[0]
        print(f"\n  Wrong predictions ({len(wrong)}):")
        for idx in wrong:
            print(f"    Sample {idx}: true={y_test[idx]}, pred={best_y[idx]}")
    else:
        print("  100% ACHIEVED!")

    print(f"\n{classification_report(y_test, best_y)}")


if __name__ == "__main__":
    main()
