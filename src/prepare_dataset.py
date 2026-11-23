"""
Preparación del dataset LSC50 para entrenamiento LSTM.

Carga los datos preprocesados, aplica ventana temporal (padding/truncamiento
o resampling), y genera splits train/val/test estratificados.

Uso:
    python src/prepare_dataset.py
    python src/prepare_dataset.py --seq-length 30 --strategy truncate
    python src/prepare_dataset.py --seq-length 60 --strategy resample
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATASET_DIR = BASE_DIR / "data" / "dataset"


# ---------------------------------------------------------------------------
# Ventana temporal
# ---------------------------------------------------------------------------

def pad_or_truncate(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Ajusta la secuencia a target_length frames.

    - Si es más larga: trunca (toma los primeros target_length frames)
    - Si es más corta: padding con ceros al final
    """
    T, F = sequence.shape
    if T >= target_length:
        return sequence[:target_length]
    else:
        padded = np.zeros((target_length, F), dtype=sequence.dtype)
        padded[:T] = sequence
        return padded


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Resamplea la secuencia a target_length frames usando interpolación lineal.

    Mantiene toda la información temporal sin truncar ni rellenar con ceros.
    """
    T, F = sequence.shape
    if T == target_length:
        return sequence

    # Índices originales y nuevos
    original_indices = np.linspace(0, T - 1, T)
    new_indices = np.linspace(0, T - 1, target_length)

    # Interpolar cada feature
    resampled = np.zeros((target_length, F), dtype=sequence.dtype)
    for f in range(F):
        resampled[:, f] = np.interp(new_indices, original_indices, sequence[:, f])

    return resampled


def sliding_window(sequence: np.ndarray, window_size: int,
                   stride: int) -> np.ndarray:
    """Genera ventanas deslizantes de tamaño fijo.

    Retorna array de shape (n_windows, window_size, n_features).
    Útil para reconocimiento continuo.
    """
    T, F = sequence.shape
    if T < window_size:
        # Padding y retornar una sola ventana
        padded = np.zeros((window_size, F), dtype=sequence.dtype)
        padded[:T] = sequence
        return padded[np.newaxis, :]

    n_windows = (T - window_size) // stride + 1
    windows = np.zeros((n_windows, window_size, F), dtype=sequence.dtype)
    for i in range(n_windows):
        start = i * stride
        windows[i] = sequence[start:start + window_size]
    return windows


# ---------------------------------------------------------------------------
# Carga de datos procesados
# ---------------------------------------------------------------------------

def load_processed_data(processed_dir: Path) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Carga los datos procesados de la Fase 1.

    Retorna:
      - sequences: lista de arrays (T_i, 225) sin padding
      - labels: array (N,) con sign_ids
      - lengths: array (N,) con frames reales
    """
    X_padded = np.load(processed_dir / "X_padded.npy")
    labels = np.load(processed_dir / "y_labels.npy")
    lengths = np.load(processed_dir / "lengths.npy")

    # Extraer secuencias sin padding
    sequences = []
    for i in range(len(labels)):
        T = int(lengths[i])
        sequences.append(X_padded[i, :T, :])

    return sequences, labels, lengths


# ---------------------------------------------------------------------------
# Pipeline de preparación
# ---------------------------------------------------------------------------

def prepare_fixed_length(sequences: list[np.ndarray], labels: np.ndarray,
                         target_length: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta todas las secuencias a longitud fija.

    Args:
        sequences: lista de arrays (T_i, F)
        labels: array (N,)
        target_length: longitud objetivo en frames
        strategy: 'truncate' (pad/truncate) o 'resample' (interpolación)

    Retorna:
        X: array (N, target_length, F)
        y: array (N,)
    """
    N = len(sequences)
    F = sequences[0].shape[1]
    X = np.zeros((N, target_length, F), dtype=np.float32)

    for i, seq in enumerate(sequences):
        if strategy == "resample":
            X[i] = resample_sequence(seq, target_length)
        else:  # truncate
            X[i] = pad_or_truncate(seq, target_length)

    return X, labels


def split_dataset(X: np.ndarray, y: np.ndarray,
                  train_ratio: float = 0.8, val_ratio: float = 0.1,
                  test_ratio: float = 0.1, random_state: int = 42
                  ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Divide el dataset en train/val/test con estratificación por clase.

    Retorna dict con keys 'train', 'val', 'test', cada uno (X, y).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Los ratios deben sumar 1.0"

    # Primer split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_ratio, random_state=random_state, stratify=y
    )

    # Segundo split: val vs test
    test_relative = test_ratio / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_relative, random_state=random_state, stratify=y_temp
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------

def save_dataset(splits: dict, dataset_dir: Path, config: dict) -> None:
    """Guarda los splits y metadatos."""
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for split_name, (X, y) in splits.items():
        np.save(dataset_dir / f"X_{split_name}.npy", X)
        np.save(dataset_dir / f"y_{split_name}.npy", y)

    # Metadata del dataset
    metadata = {
        **config,
        "splits": {},
    }
    for split_name, (X, y) in splits.items():
        unique, counts = np.unique(y, return_counts=True)
        metadata["splits"][split_name] = {
            "n_samples": int(X.shape[0]),
            "shape": list(X.shape),
            "classes": int(len(unique)),
            "samples_per_class": {int(c): int(n) for c, n in zip(unique, counts)},
        }

    meta_path = dataset_dir / "dataset_info.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Metadata: {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preparacion del dataset LSC50 para entrenamiento LSTM"
    )
    parser.add_argument("--seq-length", type=int, default=30,
                        help="Longitud fija de secuencia en frames (default: 30)")
    parser.add_argument("--strategy", choices=["truncate", "resample"], default="resample",
                        help="Estrategia para ajustar longitud (default: resample)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Proporcion de entrenamiento (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Proporcion de validacion (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Proporcion de test (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria (default: 42)")
    args = parser.parse_args()

    print("=" * 60)
    print("  LSC50 — Preparacion del Dataset")
    print("=" * 60)

    # Verificar datos procesados
    if not (PROCESSED_DIR / "X_padded.npy").exists():
        print("[ERROR] No se encontraron datos procesados.")
        print("  Ejecuta primero: python src/preprocess_lsc50.py")
        sys.exit(1)

    # Cargar datos
    print("\n[1] Cargando datos procesados...")
    sequences, labels, lengths = load_processed_data(PROCESSED_DIR)
    print(f"  Muestras: {len(sequences)}")
    print(f"  Clases: {len(np.unique(labels))}")
    print(f"  Frames — min: {lengths.min()}, max: {lengths.max()}, "
          f"media: {lengths.mean():.1f}")
    print(f"  Features por frame: {sequences[0].shape[1]}")

    # Ventana temporal
    print(f"\n[2] Ajustando secuencias a {args.seq_length} frames (estrategia: {args.strategy})...")
    X, y = prepare_fixed_length(sequences, labels, args.seq_length, args.strategy)
    print(f"  Shape resultante: {X.shape}")

    # Estadísticas de truncamiento/padding
    shorter = sum(1 for l in lengths if l < args.seq_length)
    longer = sum(1 for l in lengths if l > args.seq_length)
    exact = sum(1 for l in lengths if l == args.seq_length)
    print(f"  Secuencias mas cortas (padding/interpoladas): {shorter}")
    print(f"  Secuencias mas largas (truncadas/interpoladas): {longer}")
    print(f"  Secuencias exactas: {exact}")

    # Split
    print(f"\n[3] Dividiendo dataset ({args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%})...")
    splits = split_dataset(X, y, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    for name, (X_s, y_s) in splits.items():
        unique, counts = np.unique(y_s, return_counts=True)
        print(f"  {name:5s}: {X_s.shape[0]:4d} muestras, shape {X_s.shape}, "
              f"clases: {len(unique)}, muestras/clase: {counts.min()}-{counts.max()}")

    # Guardar
    print(f"\n[4] Guardando dataset...")
    config = {
        "source": "LSC50",
        "seq_length": args.seq_length,
        "strategy": args.strategy,
        "n_features": int(X.shape[2]),
        "n_classes": int(len(np.unique(y))),
        "total_samples": int(len(y)),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "random_state": args.seed,
    }

    save_dataset(splits, DATASET_DIR, config)

    for name in splits:
        print(f"  X_{name}.npy, y_{name}.npy")

    print(f"\n  Archivos guardados en: {DATASET_DIR}")
    print("=" * 60)
    print("  Dataset listo para entrenamiento!")
    print("=" * 60)


if __name__ == "__main__":
    main()
