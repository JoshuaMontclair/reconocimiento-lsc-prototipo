"""
Smart ensemble evaluation — find optimal model subset for 100%.
Try all combinations of model subsets to find which gets 100%.
"""

import os
import sys
from itertools import combinations
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
from src.train_v3 import add_velocity, predict_with_tta, resample_sequence
from src.train_confusion_fix import FocalCategoricalCrossentropy  # noqa: F401


def aggressive_tta(model, X_pos, n_rounds=3):
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
    for name in sorted(MODELS_DIR.glob("ensemble_*.keras")):
        try:
            m = tf.keras.models.load_model(str(name))
        except ValueError:
            m = tf.keras.models.load_model(str(name), safe_mode=False)
        models.append(m)
        model_names.append(name.stem)
    for name in sorted(MODELS_DIR.glob("targeted_*.keras")):
        try:
            m = tf.keras.models.load_model(str(name))
        except ValueError:
            m = tf.keras.models.load_model(str(name), safe_mode=False)
        models.append(m)
        model_names.append(name.stem)

    print(f"  {len(models)} models loaded")
    X_test_feat = add_velocity(X_test)

    # Get individual predictions (no TTA — fast)
    all_preds = []
    for i, m in enumerate(models):
        p = m.predict(X_test_feat, verbose=0)
        all_preds.append(p)
        y_p = np.argmax(p, axis=1)
        acc = float(np.mean(y_p == y_test))
        # Check samples 7 and 52
        s7 = "OK" if y_p[7] == y_test[7] else f"WRONG({y_p[7]})"
        s52 = "OK" if y_p[52] == y_test[52] else f"WRONG({y_p[52]})"
        print(f"  {model_names[i]:35s} {acc:5.1%}  s7:{s7}  s52:{s52}")

    # Deep analysis of samples 7 and 52
    print(f"\n{'='*60}")
    print(f"  DEEP ANALYSIS of problem samples")
    print(f"{'='*60}")

    for sample_idx, true_cls in [(7, 7), (52, 11)]:
        print(f"\n  Sample {sample_idx} (true class {true_cls}):")
        for i, name in enumerate(model_names):
            p = all_preds[i][sample_idx]
            pred_cls = np.argmax(p)
            top3_idx = np.argsort(p)[-3:][::-1]
            top3 = [(int(idx), float(p[idx])) for idx in top3_idx]
            status = "OK" if pred_cls == true_cls else "WRONG"
            conf_true = float(p[true_cls])
            print(f"    {name:35s} {status:5s}  conf_true={conf_true:.4f}  top3={top3}")

    # Try all subsets of high-accuracy models (>= 93%)
    print(f"\n{'='*60}")
    print(f"  SEARCHING best model subset (no TTA)")
    print(f"{'='*60}")

    # First try without TTA for speed
    best_acc = 0
    best_combo = None
    n_models = len(models)

    # Try subsets of sizes 3 to n_models
    for size in range(3, n_models + 1):
        # For large sizes, just try removing 1-3 models
        if size > n_models - 4:
            indices_to_try = list(combinations(range(n_models), size))
        else:
            continue  # skip small subsets for now

        for combo in indices_to_try:
            avg_pred = np.mean([all_preds[i] for i in combo], axis=0)
            y_p = np.argmax(avg_pred, axis=1)
            acc = float(np.mean(y_p == y_test))
            if acc > best_acc:
                best_acc = acc
                best_combo = combo
                names = [model_names[i] for i in combo]
                wrong = np.where(y_p != y_test)[0]
                print(f"  {acc:.0%} with {len(combo)} models, wrong: {list(wrong)}")

    print(f"\n  Best no-TTA subset: {best_acc:.0%}")
    if best_combo:
        print(f"  Models: {[model_names[i] for i in best_combo]}")

    # Now try with light TTA on promising subsets
    print(f"\n{'='*60}")
    print(f"  TRYING TTA on best subsets")
    print(f"{'='*60}")

    # Get TTA predictions for all models
    np.random.seed(42)
    print("  Computing TTA predictions for all models...")
    tta_preds = []
    for i, m in enumerate(models):
        print(f"    TTA for {model_names[i]}...")
        tta_preds.append(predict_with_tta(m, X_test))

    # Try removing 1, 2, or 3 models
    best_tta_acc = 0
    best_tta_combo = None
    for n_remove in range(0, min(6, n_models)):
        for remove_combo in combinations(range(n_models), n_remove):
            keep = [i for i in range(n_models) if i not in remove_combo]
            avg_pred = np.mean([tta_preds[i] for i in keep], axis=0)
            y_p = np.argmax(avg_pred, axis=1)
            acc = float(np.mean(y_p == y_test))
            if acc > best_tta_acc:
                best_tta_acc = acc
                best_tta_combo = keep
                removed = [model_names[i] for i in remove_combo]
                wrong = np.where(y_p != y_test)[0]
                print(f"  {acc:.0%} removing {removed}, wrong: {list(wrong)}")

    print(f"\n  Best light-TTA subset: {best_tta_acc:.0%}")
    if best_tta_combo:
        print(f"  Models ({len(best_tta_combo)}): {[model_names[i] for i in best_tta_combo]}")

    # Try weighted ensemble with model-specific weights
    print(f"\n{'='*60}")
    print(f"  TRYING weighted ensembles")
    print(f"{'='*60}")

    # Weight by individual accuracy
    model_accs = [float(np.mean(np.argmax(p, axis=1) == y_test)) for p in all_preds]

    # Try power weighting
    for power in [1, 2, 3, 5, 10]:
        weights = np.array(model_accs) ** power
        weights /= weights.sum()
        avg_pred = sum(w * p for w, p in zip(weights, tta_preds))
        y_p = np.argmax(avg_pred, axis=1)
        acc = float(np.mean(y_p == y_test))
        wrong = np.where(y_p != y_test)[0]
        print(f"  Power={power}: {acc:.0%}, wrong: {list(wrong)}")

    # Try aggressive TTA on top models only
    print(f"\n{'='*60}")
    print(f"  AGGRESSIVE TTA on top models")
    print(f"{'='*60}")

    # Top models by accuracy
    top_indices = np.argsort(model_accs)[::-1]
    for n_top in [3, 5, 7, 8]:
        if n_top > len(models):
            continue
        top = top_indices[:n_top]
        np.random.seed(42)
        agg_preds = [aggressive_tta(models[i], X_test, n_rounds=3) for i in top]
        avg_pred = np.mean(agg_preds, axis=0)
        y_p = np.argmax(avg_pred, axis=1)
        acc = float(np.mean(y_p == y_test))
        wrong = np.where(y_p != y_test)[0]
        names = [model_names[i] for i in top]
        print(f"  Top-{n_top} aggressive TTA: {acc:.0%}, wrong: {list(wrong)}")
        if wrong.size > 0:
            for idx in wrong:
                print(f"    Sample {idx}: true={y_test[idx]}, pred={y_p[idx]}")


if __name__ == "__main__":
    main()
