#!/usr/bin/env python3
# Traza fotograma a fotograma de HolisticLandmarker, con flush, para localizar abortos (D-05, 14-sep-2026).
# Uso: python scripts/humo_holistic_fotograma.py <modelo.task> <carpeta COLOR_BODY> <muestra> [CPU|GPU]
"""Fotograma a fotograma, con flush, para ver dónde y por qué aborta HolisticLandmarker."""
import sys, time, glob, os
import cv2, mediapipe as mp
MODELO, VIDEOS, muestra = sys.argv[1], sys.argv[2], sys.argv[3]
delegado = sys.argv[4] if len(sys.argv) > 4 else "CPU"
V = mp.tasks.vision; B = mp.tasks.BaseOptions
opts = V.HolisticLandmarkerOptions(base_options=B(model_asset_path=MODELO, delegate=getattr(B.Delegate, delegado)), running_mode=V.RunningMode.VIDEO)
lm = V.HolisticLandmarker.create_from_options(opts)
ruta = glob.glob(os.path.join(VIDEOS, muestra + ".*"))[0]; cap = cv2.VideoCapture(ruta); i = 0; c = [0,0,0,0]; ms=[]
print(f"mediapipe {mp.__version__} | delegado {delegado} | {muestra}", flush=True)
while True:
    ok, bgr = cap.read()
    if not ok: break
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    print(f"  f{i:03d} ->", end=" ", flush=True)
    t=time.perf_counter(); r = lm.detect_for_video(img, int(i*1000/24)); ms.append((time.perf_counter()-t)*1000)
    flags = (bool(r.pose_landmarks), bool(r.face_landmarks), bool(r.left_hand_landmarks), bool(r.right_hand_landmarks))
    for k,v in enumerate(flags): c[k]+=v
    print(f"pose {int(flags[0])} rostro {int(flags[1])} izq {int(flags[2])} der {int(flags[3])} | {ms[-1]:.0f} ms | puntos rostro {len(r.face_landmarks) if r.face_landmarks else 0}", flush=True)
    i += 1
cap.release(); lm.close()
print(f"FIN {muestra}: {i} fotogramas | pose {c[0]} rostro {c[1]} izq {c[2]} der {c[3]} | media {sum(ms)/len(ms):.1f} ms", flush=True)
