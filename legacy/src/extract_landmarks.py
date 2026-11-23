"""
Extracción de landmarks con MediaPipe Holistic.

Soporta:
  - Videos (.avi, .mp4): extrae landmarks frame a frame
  - Imágenes (.jpg, .png): extrae landmarks de cada imagen
  - Procesamiento por lotes de directorios completos

Uso:
    # Extraer de un solo video
    python src/extract_landmarks.py --input video.avi --output output.csv

    # Procesar directorio de videos LSC50
    python src/extract_landmarks.py --input data/raw/LSC50/VIDEOS/COLOR_BODY/ --output data/extracted/LSC50/

    # Procesar directorio de imágenes LSC70
    python src/extract_landmarks.py --input data/raw/LSC70/LSC70/LSC70AN/ --output data/extracted/LSC70/ --mode images
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

mp_holistic = mp.solutions.holistic

# Landmarks por tipo
N_POSE = 33
N_HAND = 21
N_FACE = 468

# Extensiones soportadas
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Extracción de landmarks
# ---------------------------------------------------------------------------

def extract_frame_landmarks(results, include_face: bool = False) -> list[float]:
    """Extrae landmarks de un frame procesado por MediaPipe Holistic.

    Retorna lista de floats:
      - Pose: 33 landmarks × 3 (x, y, z) = 99
      - Left hand: 21 × 3 = 63
      - Right hand: 21 × 3 = 63
      - Face (opcional): 468 × 3 = 1404
    """
    keypoints = []

    # Pose (33 landmarks × 3 coords)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * N_POSE * 3)

    # Mano izquierda (21 × 3)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * N_HAND * 3)

    # Mano derecha (21 × 3)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0.0] * N_HAND * 3)

    # Face (468 × 3) — opcional
    if include_face:
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([0.0] * N_FACE * 3)

    return keypoints


def extract_from_video(video_path: str, include_face: bool = False,
                       min_detection_confidence: float = 0.5,
                       min_tracking_confidence: float = 0.5) -> np.ndarray:
    """Extrae landmarks de todos los frames de un video.

    Retorna: np.ndarray de shape (n_frames, n_features)
      - Sin face: (n_frames, 225)
      - Con face: (n_frames, 1629)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] No se pudo abrir: {video_path}")
        return np.array([])

    sequence = []
    with mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_frame_landmarks(results, include_face)
            sequence.append(keypoints)

    cap.release()
    return np.array(sequence, dtype=np.float32)


def extract_from_image(image_path: str, include_face: bool = False,
                       min_detection_confidence: float = 0.5) -> np.ndarray:
    """Extrae landmarks de una sola imagen.

    Retorna: np.ndarray de shape (1, n_features)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [ERROR] No se pudo leer: {image_path}")
        return np.array([])

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=min_detection_confidence,
    ) as holistic:
        results = holistic.process(image_rgb)
        keypoints = extract_frame_landmarks(results, include_face)

    return np.array([keypoints], dtype=np.float32)


# ---------------------------------------------------------------------------
# Generación de headers CSV
# ---------------------------------------------------------------------------

def generate_csv_header(include_face: bool = False) -> list[str]:
    """Genera los nombres de columnas para el CSV de salida."""
    headers = []
    # Body
    for i in range(N_POSE):
        for c in ("x", "y", "z"):
            headers.append(f"body_landmark_{i}_{c}")
    # Left hand
    for i in range(N_HAND):
        for c in ("x", "y", "z"):
            headers.append(f"lh_landmark_{i}_{c}")
    # Right hand
    for i in range(N_HAND):
        for c in ("x", "y", "z"):
            headers.append(f"rh_landmark_{i}_{c}")
    # Face
    if include_face:
        for i in range(N_FACE):
            for c in ("x", "y", "z"):
                headers.append(f"face_landmark_{i}_{c}")
    return headers


def save_to_csv(data: np.ndarray, output_path: Path, include_face: bool = False) -> None:
    """Guarda landmarks extraídos en formato CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = generate_csv_header(include_face)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in data:
            writer.writerow(row.tolist())


# ---------------------------------------------------------------------------
# Procesamiento por lotes
# ---------------------------------------------------------------------------

def process_video_directory(input_dir: Path, output_dir: Path,
                            include_face: bool = False) -> None:
    """Procesa todos los videos en un directorio."""
    video_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in VIDEO_EXTS
    )
    total = len(video_files)
    print(f"  Encontrados {total} videos en {input_dir}")

    for i, vpath in enumerate(video_files):
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Progreso: {i + 1}/{total}")

        data = extract_from_video(vpath, include_face)
        if data.size == 0:
            continue

        out_path = output_dir / f"{vpath.stem}.csv"
        save_to_csv(data, out_path, include_face)


def process_image_directory(input_dir: Path, output_dir: Path,
                            include_face: bool = False) -> None:
    """Procesa imágenes organizadas en subdirectorios (estructura LSC70).

    Estructura esperada: input_dir/PersonXX/SignLabel/PersonXX_SignLabel_N.jpg
    Salida: un CSV por persona/seña con todos los frames de esa combinación.
    """
    # Recopilar todas las imágenes recursivamente
    image_files = sorted(
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in IMAGE_EXTS
    )
    total = len(image_files)
    print(f"  Encontradas {total} imagenes en {input_dir}")

    # Agrupar por directorio padre (persona/seña)
    groups: dict[str, list[Path]] = {}
    for img in image_files:
        # Usar ruta relativa como clave de grupo
        rel = img.relative_to(input_dir)
        group_key = str(rel.parent)
        groups.setdefault(group_key, []).append(img)

    print(f"  Grupos (persona/sena): {len(groups)}")
    processed = 0

    for group_key, imgs in sorted(groups.items()):
        all_landmarks = []
        for img_path in sorted(imgs):
            data = extract_from_image(img_path, include_face)
            if data.size > 0:
                all_landmarks.append(data[0])

        if all_landmarks:
            arr = np.array(all_landmarks, dtype=np.float32)
            # Nombre de salida basado en la estructura de directorios
            safe_name = group_key.replace("/", "_").replace("\\", "_")
            out_path = output_dir / f"{safe_name}.csv"
            save_to_csv(arr, out_path, include_face)

        processed += 1
        if processed % 100 == 0 or processed == len(groups):
            print(f"  Progreso: {processed}/{len(groups)} grupos")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extraccion de landmarks con MediaPipe Holistic"
    )
    parser.add_argument("--input", required=True, help="Archivo o directorio de entrada")
    parser.add_argument("--output", required=True, help="Archivo o directorio de salida")
    parser.add_argument("--mode", choices=["videos", "images", "auto"], default="auto",
                        help="Modo de procesamiento (default: auto)")
    parser.add_argument("--include-face", action="store_true",
                        help="Incluir 468 face landmarks (default: solo body + manos)")
    parser.add_argument("--min-detection", type=float, default=0.5,
                        help="Confianza minima de deteccion (default: 0.5)")
    parser.add_argument("--min-tracking", type=float, default=0.5,
                        help="Confianza minima de tracking (default: 0.5)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 60)
    print("  MediaPipe Holistic — Extraccion de Landmarks")
    print("=" * 60)

    n_features = 225 if not args.include_face else 1629
    print(f"  Features por frame: {n_features}")
    print(f"  Face landmarks: {'SI' if args.include_face else 'NO'}")

    if input_path.is_file():
        # Archivo individual
        suffix = input_path.suffix.lower()
        if suffix in VIDEO_EXTS:
            print(f"\n  Procesando video: {input_path}")
            data = extract_from_video(input_path, args.include_face,
                                      args.min_detection, args.min_tracking)
        elif suffix in IMAGE_EXTS:
            print(f"\n  Procesando imagen: {input_path}")
            data = extract_from_image(input_path, args.include_face, args.min_detection)
        else:
            print(f"[ERROR] Extension no soportada: {suffix}")
            sys.exit(1)

        if data.size > 0:
            if output_path.suffix.lower() == ".csv":
                save_to_csv(data, output_path, args.include_face)
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, data)
            print(f"  Guardado: {output_path} — shape {data.shape}")

    elif input_path.is_dir():
        # Directorio
        output_path.mkdir(parents=True, exist_ok=True)

        # Autodetección de modo
        mode = args.mode
        if mode == "auto":
            has_videos = any(f.suffix.lower() in VIDEO_EXTS for f in input_path.iterdir())
            mode = "videos" if has_videos else "images"
            print(f"  Modo autodetectado: {mode}")

        if mode == "videos":
            process_video_directory(input_path, output_path, args.include_face)
        else:
            process_image_directory(input_path, output_path, args.include_face)
    else:
        print(f"[ERROR] Ruta no encontrada: {input_path}")
        sys.exit(1)

    print("\n  Extraccion completada!")


if __name__ == "__main__":
    main()
