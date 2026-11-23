"""
Fine-tuning del modelo LSTM con landmarks re-extraídos (Tasks API).

Toma el modelo entrenado con landmarks originales (Holistic API, ~86% accuracy)
y lo re-entrena con landmarks de la Tasks API a un learning rate bajo.
Esto adapta el modelo al espacio de coordenadas que usa la webcam en tiempo real.

Uso:
    python src/finetune_lstm.py
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from sklearn.metrics import classification_report, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "dataset"
MODELS_DIR = BASE_DIR / "models"

# Importar augmentation del script original
sys.path.insert(0, str(BASE_DIR))
from src.train_lstm import apply_augmentation, evaluate_model


def main():
    print("=" * 60)
    print("  Fine-tuning LSTM con landmarks Tasks API")
    print("=" * 60)

    # 1. Cargar modelo base (entrenado con landmarks originales)
    base_model_path = MODELS_DIR / "best_model.keras"
    print(f"\n[1] Cargando modelo base: {base_model_path}")
    model = tf.keras.models.load_model(str(base_model_path))
    print(f"  Input: {model.input_shape}, Output: {model.output_shape}")

    # Guardar copia del modelo original antes de fine-tuning
    original_backup = MODELS_DIR / "best_model_original.keras"
    if not original_backup.exists():
        model.save(str(original_backup))
        print(f"  Backup guardado: {original_backup}")

    # 2. Preprocesar landmarks re-extraídos
    print("\n[2] Preprocesando landmarks re-extraídos (Tasks API)...")
    from src.preprocess_lsc50 import (
        BASE_DIR as PROJ_DIR,
        OUTPUT_DIR,
        N_TOTAL_FEATURES,
    )

    # Cambiar rutas a data/extracted/ temporalmente
    extracted_base = PROJ_DIR / "data" / "extracted" / "LSC50"
    body_dir = extracted_base / "BODY_LANDMARKS"
    lh_dir = extracted_base / "LEFT_HAND_LANDMARKS"
    rh_dir = extracted_base / "RIGHT_HAND_LANDMARKS"

    import pandas as pd

    body_files = sorted(body_dir.glob("*.csv"))
    print(f"  Archivos encontrados: {len(body_files)}")

    # Cargar y procesar (mismo pipeline que preprocess_lsc50.py)
    from src.preprocess_lsc50 import (
        N_BODY_FEATURES, N_HAND_FEATURES,
        LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX, COORDS,
    )

    samples = []
    labels = []

    for bf in body_files:
        stem = bf.stem
        lf = lh_dir / f"{stem}.csv"
        rf = rh_dir / f"{stem}.csv"

        if not lf.exists() or not rf.exists():
            continue

        body = pd.read_csv(bf, index_col=0, dtype=np.float32).values
        lh = pd.read_csv(lf, index_col=0, dtype=np.float32).values
        rh = pd.read_csv(rf, index_col=0, dtype=np.float32).values

        if body.shape[1] != N_BODY_FEATURES or lh.shape[1] != N_HAND_FEATURES:
            continue

        min_frames = min(len(body), len(lh), len(rh))
        combined = np.hstack([body[:min_frames], lh[:min_frames], rh[:min_frames]])

        # Normalizar (centrado en hombros)
        T = combined.shape[0]
        ls_x = combined[:, LEFT_SHOULDER_IDX]
        ls_y = combined[:, LEFT_SHOULDER_IDX + 1]
        ls_z = combined[:, LEFT_SHOULDER_IDX + 2]
        rs_x = combined[:, RIGHT_SHOULDER_IDX]
        rs_y = combined[:, RIGHT_SHOULDER_IDX + 1]
        rs_z = combined[:, RIGHT_SHOULDER_IDX + 2]

        ref_x = (ls_x + rs_x) / 2.0
        ref_y = (ls_y + rs_y) / 2.0
        ref_z = (ls_z + rs_z) / 2.0
        scale = np.sqrt((ls_x - rs_x)**2 + (ls_y - rs_y)**2)
        scale = np.maximum(scale, 1e-6)

        reshaped = combined.reshape(T, -1, COORDS)
        ref = np.stack([ref_x, ref_y, ref_z], axis=1)[:, np.newaxis, :]
        reshaped -= ref
        reshaped /= scale[:, np.newaxis, np.newaxis]
        combined = reshaped.reshape(T, -1)

        samples.append(combined)
        sign_id = int(stem.split("_")[0])
        labels.append(sign_id)

    print(f"  Muestras cargadas: {len(samples)}")

    # 3. Resamplear a 30 frames (igual que prepare_dataset.py)
    seq_length = model.input_shape[1]  # 30
    print(f"\n[3] Resampleando a {seq_length} frames...")

    X_all = np.zeros((len(samples), seq_length, N_TOTAL_FEATURES), dtype=np.float32)
    for i, s in enumerate(samples):
        orig_len = len(s)
        orig_indices = np.linspace(0, orig_len - 1, seq_length)
        for f in range(N_TOTAL_FEATURES):
            X_all[i, :, f] = np.interp(orig_indices, np.arange(orig_len), s[:, f])

    y_all = np.array(labels, dtype=np.int32)

    # Split 80/10/10 estratificado
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 4. Augmentation
    print("\n[4] Aplicando data augmentation...")
    X_train_aug, y_train_aug = apply_augmentation(X_train, y_train)
    print(f"  Train aumentado: {len(X_train_aug)} muestras")

    # 5. Fine-tuning con learning rate bajo
    print("\n[5] Fine-tuning (lr=0.0001, max 100 epochs, patience=15)...")

    # Recompilar con learning rate bajo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    finetuned_path = str(MODELS_DIR / "best_model.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7,
            min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            finetuned_path, monitor="val_accuracy",
            save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Evaluar
    print("\n[6] Evaluacion en test...")
    model = tf.keras.models.load_model(finetuned_path)
    results = evaluate_model(model, X_test, y_test, MODELS_DIR)

    test_acc = results["test_accuracy"]
    print(f"\n{'=' * 60}")
    print(f"  Fine-tuning completado")
    print(f"  Accuracy en test: {test_acc:.1%}")
    print(f"  Modelo guardado: {finetuned_path}")
    print(f"  Modelo original backup: {original_backup}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
