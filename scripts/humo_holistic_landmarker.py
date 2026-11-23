#!/usr/bin/env python3
# Prueba de humo de mp.tasks.vision.HolisticLandmarker (D-05, 14-sep-2026). Solo lectura sobre data/raw.
# Uso: python scripts/humo_holistic_landmarker.py models/mediapipe/holistic_landmarker.task data/raw/LSC50/VIDEOS/COLOR_BODY
# Requiere mediapipe >= 0.10.33 (0.10.21 aborta si falta una mano; 0.10.32 no trae HolisticLandmarker; 1.0.1 falla en macOS).
"""Prueba de humo de mp.tasks.vision.HolisticLandmarker sobre 3 videos de LSC50 (solo lectura)."""
import sys, time, glob, os, statistics as st
import cv2, numpy as np, mediapipe as mp
MODELO = sys.argv[1]; VIDEOS = sys.argv[2]
muestras = ["0000_0000_0000", "0025_0002_0001", "0049_0004_0003"]
V = mp.tasks.vision; B = mp.tasks.BaseOptions
opts = V.HolisticLandmarkerOptions(base_options=B(model_asset_path=MODELO, delegate=B.Delegate.CPU),
                                   running_mode=V.RunningMode.VIDEO)
print("mediapipe", mp.__version__, "| opciones disponibles:", [a for a in dir(opts) if not a.startswith('_') and 'min_' in a])
tot = {"n":0,"pose":0,"face":0,"lh":0,"rh":0}; tiempos=[]; PRIMERA=[]; ncoords=None; dist_lh=[]; dist_rh=[]
for m in muestras:
    ruta = glob.glob(os.path.join(VIDEOS, m + ".*"))[0]
    lm = V.HolisticLandmarker.create_from_options(opts)   # uno por video: el reloj del modo VIDEO debe ser creciente
    cap = cv2.VideoCapture(ruta); i=0; c={"n":0,"pose":0,"face":0,"lh":0,"rh":0}
    while True:
        ok, bgr = cap.read()
        if not ok: break
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        t=time.perf_counter(); r = lm.detect_for_video(img, int(i*1000/24)); tiempos.append((time.perf_counter()-t)*1000)
        c["n"]+=1
        if r.pose_landmarks: c["pose"]+=1
        if r.face_landmarks: c["face"]+=1
        if r.left_hand_landmarks: c["lh"]+=1
        if r.right_hand_landmarks: c["rh"]+=1
        if ncoords is None and r.pose_landmarks and r.face_landmarks and r.left_hand_landmarks:
            ncoords = (len(r.pose_landmarks), len(r.face_landmarks), len(r.left_hand_landmarks), len(r.right_hand_landmarks or []))
            print("campos del resultado:", [a for a in dir(r) if not a.startswith('_')])
        # lateralidad: muñeca de la mano (punto 0) frente a muñecas de la postura (15 izq, 16 der)
        if r.pose_landmarks:
            p15=(r.pose_landmarks[15].x, r.pose_landmarks[15].y); p16=(r.pose_landmarks[16].x, r.pose_landmarks[16].y)
            d=lambda a,b: ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
            if r.left_hand_landmarks:  w=(r.left_hand_landmarks[0].x, r.left_hand_landmarks[0].y);  dist_lh.append((d(w,p15), d(w,p16)))
            if r.right_hand_landmarks: w=(r.right_hand_landmarks[0].x, r.right_hand_landmarks[0].y); dist_rh.append((d(w,p15), d(w,p16)))
        if m == muestras[0]: PRIMERA.append(((r.left_hand_landmarks[0].x, r.left_hand_landmarks[0].y) if r.left_hand_landmarks else None, (r.right_hand_landmarks[0].x, r.right_hand_landmarks[0].y) if r.right_hand_landmarks else None))
        i+=1
    cap.release(); lm.close()
    print(f"{m}: fotogramas {c['n']} | pose {c['pose']} | rostro {c['face']} | mano izq {c['lh']} | mano der {c['rh']}")
    for k in tot: tot[k]+=c[k]
print("TOTAL:", tot)
import csv
pub = os.path.join(os.path.dirname(os.path.dirname(VIDEOS)), "LANDMARKS", "HANDS_LANDMARKS", "LEFT_HAND_LANDMARKS", "0000_0000_0000.csv")
if os.path.exists(pub) and PRIMERA:
    filas = list(csv.DictReader(open(pub, newline="")))
    n = min(len(filas), len(PRIMERA))
    dl = [abs(float(filas[k]["landmark_0_x"]) - PRIMERA[k][0][0]) + abs(float(filas[k]["landmark_0_y"]) - PRIMERA[k][0][1]) for k in range(n) if PRIMERA[k][0]]
    dr = [abs(float(filas[k]["landmark_0_x"]) - PRIMERA[k][1][0]) + abs(float(filas[k]["landmark_0_y"]) - PRIMERA[k][1][1]) for k in range(n) if PRIMERA[k][1]]
    print(f"0000_0000_0000 · muñeca del LEFT_HAND publicado frente a Holistic: |dxy| medio con la mano IZQ de Holistic {st.mean(dl):.4f} (n={len(dl)}) · con la DER {st.mean(dr):.4f} (n={len(dr)}); filas publicadas {len(filas)}")
print("puntos por fotograma (pose, rostro, mano izq, mano der):", ncoords)
print(f"ms por fotograma: media {st.mean(tiempos):.1f} | mediana {st.median(tiempos):.1f} | p90 {sorted(tiempos)[int(0.9*len(tiempos))-1]:.1f} | fot./s {1000/st.mean(tiempos):.1f}")
if dist_lh: print(f"LEFT_HAND: dist. mediana a muñeca pose 15 (izq) {st.median(a for a,b in dist_lh):.3f} | a 16 (der) {st.median(b for a,b in dist_lh):.3f} | n={len(dist_lh)} | más cerca de 15: {sum(a<b for a,b in dist_lh)}")
if dist_rh: print(f"RIGHT_HAND: dist. mediana a muñeca pose 15 (izq) {st.median(a for a,b in dist_rh):.3f} | a 16 (der) {st.median(b for a,b in dist_rh):.3f} | n={len(dist_rh)} | más cerca de 16: {sum(b<a for a,b in dist_rh)}")
