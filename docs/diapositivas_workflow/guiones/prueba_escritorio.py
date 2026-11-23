"""
Prueba de escritorio: UN video del corpus LSC50, de principio a fin, por el camino real del sistema.

Pedida por el director en la reunión del 28-sep-2026 («este es el video de la seña 1 y dura tanto; y
mira, este es el resultado»). Sigue una muestra (por defecto 0030_0000_0000, «hola», señante 0000)
e imprime en cada etapa QUÉ ENTRA y QUÉ SALE, con la forma de los datos. No reimplementa nada: importa
las funciones del repositorio.

  Etapa 1  Video (cv2, solo lectura)                        → T cuadros de 1920x1080 a 24 fps
  Etapa 2  HolisticLandmarker (src/reextract_lsc50.py:       → por cuadro 543 puntos (x, y, z)
           crear_holistic, extraer_video, OPCIONES_HOLISTIC)    = 33 postura + 468 rostro + 21 + 21 manos
  Etapa 3  Fila por cuadro (src/preprocess_lsc50.py:         → (T, 405) = postura 99 + manos 126 + cejas/labios 180
           escribir_csv + load_single_sample, igual que el corpus)
  Etapa 4  Regla sin postura, regla «mascara» (D-07) y       → (T, 405) normalizado por hombros
           normalización por hombros (preprocess_lsc50)
  Etapa 5  Representación «manos» (src/representacion.py, D-08) → (T, 126) = 2 manos x 21 puntos x 3
  Etapa 6  resample_sequence (src/train_final_100.py)        → (30, 126)  interpolación lineal
  Etapa 7  add_velocity (src/train_final_100.py)             → (1, 30, 252) = 126 posición + 126 velocidad
  Etapa 8  Modelo canónico (models/canonico_holistic_mascara_manos/modelo_canonico.keras, N-31)
                                                             → 50 probabilidades; top-3; umbral 0,20 (D-09)
  Etapa 9  GrammarMapper (src/grammar_mapper.py)             → texto que diría la voz (aquí no suena nada)

Verificaciones:
  (a) la entrada de la Etapa 7 frente a la del ENTRENADOR para esa muestra (cargar_datos +
      remuestrear_todas + add_velocity sobre data/processed_holistic_mascara_rostro): diferencia máxima;
  (b) la fila normalizada de la Etapa 4 frente a data/processed_holistic_mascara_rostro/samples/<m>.npy;
  (c) el PROTOTIPO: LSCTranslator (src/translator.py) sin cámara, tts_mode="none", registro=False,
      alimentado con los cuadros del video en una sola ventana (como en N-25/N-31): entrada al modelo y
      predicción. Dos variantes: reloj real (como en vivo) y reloj del video (marcas round(i*1000/fps),
      las mismas de la re-extracción del corpus).

Solo LEE data/ y models/. Escribe: (i) los CSV intermedios de la Etapa 3 en --tmp (por defecto una
carpeta temporal del sistema que se borra al terminar); (ii) con --figuras DIR, las figuras PDF; (iii)
con --json RUTA, un resumen con todas las cifras. Nada se escribe en registros/ (registro=False). No
abre la cámara ni reproduce audio. Las figuras NO contienen imágenes de la persona: solo el esqueleto de
puntos sobre fondo liso (07_etica.tex del anteproyecto).

Uso (desde la raíz del repositorio):
    python docs/diapositivas_workflow/guiones/prueba_escritorio.py                       # «hola»
    python docs/diapositivas_workflow/guiones/prueba_escritorio.py 0033_0000_0001        # otra muestra
    python docs/diapositivas_workflow/guiones/prueba_escritorio.py 0030_0000_0000 0033_0000_0001 \\
        --figuras docs/diapositivas_workflow/figuras --json /tmp/escritorio.json
    (con varias muestras, las figuras a-d son de la PRIMERA; la figura de duraciones las reúne todas)
    --sin-prototipo   omite la verificación (c) (más rápido para mostrar en vivo)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src import preprocess_lsc50 as pp  # noqa: E402
from src import reextract_lsc50 as rx  # noqa: E402
from src import representacion as rp  # noqa: E402
from src.grammar_mapper import GrammarMapper, SIGN_VOCABULARY  # noqa: E402
from src.train_final_100 import (  # noqa: E402  (importa TensorFlow)
    add_velocity, cargar_datos, remuestrear_todas, resample_sequence,
)
import src.translator as trm  # noqa: E402

VIDEOS = RAIZ / "data" / "raw" / "LSC50" / "VIDEOS" / "COLOR_BODY"
PROCESSED = RAIZ / "data" / "processed_holistic_mascara_rostro"
EXTRAIDO = RAIZ / "data" / "extracted" / "LSC50_holistic"
PARTICION_ALEATORIA = RAIZ / "data" / "partitions" / "aleatoria_80_10_10.json"
PARTICION_LOSO = RAIZ / "data" / "partitions" / "loso_5_pliegues.json"
MODELO = trm.MODELO_POR_DEFECTO          # models/canonico_holistic_mascara_manos/modelo_canonico.keras
REPRESENTACION = "manos"
ROSTRO = "cejas_labios"                  # el processed del modelo canónico se generó con este rostro
REGLA = "mascara"                        # D-07
PASOS = 30

DEDOS = {"muñeca": [0], "pulgar": [1, 2, 3, 4], "índice": [5, 6, 7, 8], "medio": [9, 10, 11, 12],
         "anular": [13, 14, 15, 16], "meñique": [17, 18, 19, 20]}


def linea(t=""):
    print(("── " + t + " ").ljust(96, "─") if t else "─" * 96)


def fmt(x, d=4):
    """Número con coma decimal (convención del documento)."""
    return f"{x:.{d}f}".replace(".", ",")


# ------------------------------------------------------------------------------------------------
# Particiones: ¿la muestra se usó para entrenar el modelo canónico?
# ------------------------------------------------------------------------------------------------

def pertenencia(muestra):
    a = json.loads(PARTICION_ALEATORIA.read_text(encoding="utf-8"))
    conj = [k for k in ("train", "val", "test") if muestra in a["conjuntos"][k]]
    excl = muestra in a.get("excluidas", {})
    loso = json.loads(PARTICION_LOSO.read_text(encoding="utf-8"))
    pl = next((p["pliegue"] for p in loso["pliegues"] if muestra in p["conjuntos"]["test"]), None)
    return {"aleatoria": conj[0] if conj else ("excluida" if excl else None),
            "excluida_duplicado": excl, "loso_pliegue_de_prueba": pl}


# ------------------------------------------------------------------------------------------------
# Camino del corpus (re-extracción + preprocesamiento + entrenador)
# ------------------------------------------------------------------------------------------------

def camino_corpus(muestra, tmp_dir):
    d = {"muestra": muestra, "sena_id": int(muestra.split("_")[0]),
         "senante": muestra.split("_")[1], "repeticion": muestra.split("_")[2]}
    d["glosa"] = SIGN_VOCABULARY[d["sena_id"]]
    d["particion"] = pertenencia(muestra)
    video = VIDEOS / f"{muestra}.avi"

    # Etapa 1: video
    linea(f"ETAPA 1 · Video  ({video.relative_to(RAIZ)})")
    cap = cv2.VideoCapture(str(video))
    cab = rx.leer_cabecera_video(cap)
    cuadros = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cuadros.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    T = len(cuadros)
    d["video"] = {**cab, "cuadros_decodificados": T, "duracion_s": T / cab["fps"]}
    print(f"  ENTRA: archivo .avi ({cab['codec']})")
    print(f"  SALE : {T} cuadros de {cab['ancho']}x{cab['alto']} px a {cab['fps']:g} cuadros/s "
          f"(cabecera: {cab['fotogramas_cabecera']}) → duración {fmt(T / cab['fps'], 3)} s")
    print(f"         seña {d['sena_id']} «{d['glosa']}», señante {d['senante']}, repetición {d['repeticion']}")
    print(f"         partición aleatoria (modelo canónico): {d['particion']['aleatoria']}; "
          f"LOSO: prueba del pliegue {d['particion']['loso_pliegue_de_prueba']}")

    # Etapa 2: HolisticLandmarker, exactamente como la re-extracción del corpus
    linea("ETAPA 2 · Extracción de puntos: MediaPipe HolisticLandmarker (D-05)")
    ruta_task, sha = rx.ensure_modelo()
    holistic = rx.crear_holistic(ruta_task)
    t0 = time.perf_counter()
    try:
        res = rx.extraer_video(video, holistic, con_rostro=True)
    finally:
        holistic.close()
    seg = time.perf_counter() - t0
    body, lh, rh, face = (np.asarray(res[k], dtype=np.float32) for k in ("body", "lh", "rh", "face"))
    reg = res["registros"]
    det = {k: int(sum(r[k] for r in reg)) for k in
           ("pose_detectada", "mano_izq_detectada", "mano_der_detectada", "rostro_detectado")}
    ms = [r["holistic_ms"] for r in reg]
    d["extraccion"] = {"task_sha256": sha, "opciones": rx.OPCIONES_HOLISTIC,
                       "opciones_iguales_al_prototipo": rx.OPCIONES_HOLISTIC == trm.OPCIONES_HOLISTIC,
                       "formas": {"postura": list(body.shape), "mano_izq": list(lh.shape),
                                  "mano_der": list(rh.shape), "rostro": list(face.shape)},
                       "puntos_por_cuadro": (body.shape[1] + lh.shape[1] + rh.shape[1] + face.shape[1]) // 3,
                       "detecciones": det, "holistic_ms_media": float(np.mean(ms)),
                       "holistic_ms_mediana": float(np.median(ms)), "segundos": seg}
    print(f"  ENTRA: {T} cuadros RGB; modelo holistic_landmarker.task (SHA256 {sha[:12]}…), modo VIDEO, CPU,")
    print(f"         marcas de tiempo round(i·1000/fps); opciones = las del prototipo: "
          f"{d['extraccion']['opciones_iguales_al_prototipo']}")
    print(f"  SALE : por cuadro {d['extraccion']['puntos_por_cuadro']} puntos (x, y, z) = "
          f"postura 33 + rostro 468 + mano izq. 21 + mano der. 21")
    print(f"         matrices: postura {body.shape}, mano izq. {lh.shape}, mano der. {rh.shape}, rostro {face.shape}")
    print(f"         detección: postura {det['pose_detectada']}/{T}, mano izq. {det['mano_izq_detectada']}/{T}, "
          f"mano der. {det['mano_der_detectada']}/{T}, rostro {det['rostro_detectado']}/{T}")
    print(f"         costo del extractor en esta corrida: media {fmt(np.mean(ms), 1)} ms por cuadro "
          f"(mediana {fmt(np.median(ms), 1)})")
    k = T // 2
    print(f"         ejemplo, cuadro {k}, muñeca de la mano der. (punto 0): x={fmt(rh[k, 0])} y={fmt(rh[k, 1])} "
          f"z={fmt(rh[k, 2])} (x, y en fracción del ancho/alto de la imagen)")
    # Cotejo con la extracción guardada del corpus (data/extracted/LSC50_holistic)
    try:
        import pandas as pd
        dif_ext = max(float(np.abs(pd.read_csv(EXTRAIDO / c / f"{muestra}.csv", index_col=0).to_numpy(np.float32)
                                   - a).max())
                      for c, a in (("BODY_LANDMARKS", body), ("LEFT_HAND_LANDMARKS", lh), ("RIGHT_HAND_LANDMARKS", rh)))
    except Exception as exc:  # noqa: BLE001
        dif_ext = None
        print(f"  [aviso] no se pudo cotejar con {EXTRAIDO.relative_to(RAIZ)}: {exc}")
    d["extraccion"]["dif_max_vs_extraccion_guardada"] = dif_ext
    if dif_ext is not None:
        print(f"  COTEJO: frente a {EXTRAIDO.relative_to(RAIZ)}/…/{muestra}.csv (corrida del 14-sep): "
              f"diferencia máxima {dif_ext:.3g}")

    # Etapa 3: fila por cuadro, por el MISMO lector del corpus (CSV → load_single_sample)
    linea("ETAPA 3 · Fila por cuadro (CSV → preprocess_lsc50.load_single_sample)")
    dirs = {"body": tmp_dir / "BODY_LANDMARKS", "lh": tmp_dir / "LEFT_HAND_LANDMARKS",
            "rh": tmp_dir / "RIGHT_HAND_LANDMARKS", "face": tmp_dir / "FACE_LANDMARKS"}
    for (clave, carpeta), filas, n in zip(dirs.items(), (res["body"], res["lh"], res["rh"], res["face"]),
                                          (rx.N_POSE, rx.N_HAND, rx.N_HAND, rx.N_FACE)):
        carpeta.mkdir(parents=True, exist_ok=True)
        rx.escribir_csv(carpeta / f"{muestra}.csv", rx.make_headers("landmark", n), filas)
    data = pp.load_single_sample(muestra, dirs, ROSTRO)
    cruda = data.copy()
    d["fila"] = {"forma": list(data.shape), "bloques": rp.diseno_processed(ROSTRO)}
    print(f"  ENTRA: los 4 CSV de la Etapa 2 (1 fila por cuadro)")
    print(f"  SALE : matriz {data.shape} = por cuadro postura 99 (0:99) + mano izq. 63 (99:162) + "
          f"mano der. 63 (162:225) + cejas y labios 180 (225:405)")
    print(f"         (del rostro solo se conservan 60 puntos; el modelo «manos» no los usa)")

    # Etapa 4: reglas y normalización (orden de preprocess_lsc50.main)
    linea("ETAPA 4 · Regla sin postura, regla «mascara» (D-07) y normalización por hombros")
    muestras = {muestra: {"data": data, "sign_id": d["sena_id"], "volunteer_id": int(d["senante"]),
                          "repetition_id": int(d["repeticion"]), "n_frames": data.shape[0]}}
    with contextlib.redirect_stdout(io.StringIO()):
        sin_postura = pp.aplicar_regla_sin_postura(muestras)
        ausente = pp.aplicar_regla_mano_ausente(muestras, REGLA)
        pp.normalize_all_samples(muestras)
        enmascarados = pp.aplicar_mascara_mano_ausente(muestras)
    norm = muestras[muestra]["data"]
    ls, rs = cruda[:, pp.LEFT_SHOULDER_IDX:pp.LEFT_SHOULDER_IDX + 2], cruda[:, pp.RIGHT_SHOULDER_IDX:pp.RIGHT_SHOULDER_IDX + 2]
    escala = np.sqrt(((ls - rs) ** 2).sum(axis=1))
    sin_izq = (cruda[:, 99:162] == 0).all(axis=1)
    sin_der = (cruda[:, 162:225] == 0).all(axis=1)
    d["normalizacion"] = {"fotogramas_sin_postura": sin_postura["filas_postura_en_ceros"],
                          "fotogramas_sin_mano_izq": int(sin_izq.sum()), "fotogramas_sin_mano_der": int(sin_der.sum()),
                          "bloques_enmascarados": int(enmascarados),
                          "escala_hombros_media": float(escala.mean()),
                          "escala_hombros_min": float(escala.min()), "escala_hombros_max": float(escala.max())}
    ref = ((ls + rs) / 2)[k]
    print(f"  ENTRA: {cruda.shape}; cuadros sin postura {sin_postura['filas_postura_en_ceros']}, "
          f"sin mano izq. {int(sin_izq.sum())}, sin mano der. {int(sin_der.sum())}")
    print(f"  SALE : {norm.shape}; cada punto → (punto − punto medio de los hombros) / distancia entre hombros;")
    print(f"         bloques de mano ausente puestos en 0 DESPUÉS de normalizar: {enmascarados}")
    print(f"         distancia entre hombros (fracción del ancho de imagen): media {fmt(escala.mean())}, "
          f"mín. {fmt(escala.min())}, máx. {fmt(escala.max())}")
    xr, yr = cruda[k, 162], cruda[k, 163]
    print(f"         ejemplo, cuadro {k}, muñeca der.: x = ({fmt(xr)} − {fmt(ref[0])}) / {fmt(escala[k])} = "
          f"{fmt(norm[k, 162])};  y = ({fmt(yr)} − {fmt(ref[1])}) / {fmt(escala[k])} = {fmt(norm[k, 163])}")
    d["normalizacion"]["ejemplo"] = {"cuadro": k, "x_crudo": float(xr), "y_crudo": float(yr),
                                     "ref_x": float(ref[0]), "ref_y": float(ref[1]), "escala": float(escala[k]),
                                     "x_norm": float(norm[k, 162]), "y_norm": float(norm[k, 163])}
    disco = np.load(PROCESSED / "samples" / f"{muestra}.npy")
    dif_disco = float(np.abs(disco - norm).max()) if disco.shape == norm.shape else None
    d["normalizacion"]["dif_max_vs_processed_samples"] = dif_disco
    print(f"  COTEJO: frente a {PROCESSED.relative_to(RAIZ)}/samples/{muestra}.npy {disco.shape}: "
          f"diferencia máxima {dif_disco if dif_disco is None else f'{dif_disco:.3g}'}")

    # Etapa 5: representación «manos»
    linea("ETAPA 5 · Representación «manos» (D-08): solo las dos manos")
    cols = rp.columnas(REPRESENTACION, ROSTRO)
    manos = norm[:, cols]
    print(f"  ENTRA: {norm.shape}")
    print(f"  SALE : {manos.shape} = mano izq. 21 puntos × (x, y, z) = 63  +  mano der. 21 × 3 = 63  "
          f"(columnas {cols[0]}:{cols[-1] + 1} del processed)")
    print(f"         orden de columnas: izq. p0x p0y p0z p1x … p20z | der. p0x … p20z")

    # Etapa 6: remuestreo a 30 pasos (interpolación lineal)
    linea(f"ETAPA 6 · Remuestreo temporal a {PASOS} pasos (resample_sequence: interpolación lineal)")
    rem = resample_sequence(manos, PASOS)
    pos = np.linspace(0, T - 1, PASOS)
    print(f"  ENTRA: {manos.shape} ({T} cuadros)")
    print(f"  SALE : {rem.shape}; el paso j toma la posición j·(T−1)/29 = j·{T - 1}/29 = j·{fmt((T - 1) / 29)} "
          f"sobre el eje de cuadros")
    print(f"         posiciones: " + ", ".join(fmt(p, 2) for p in pos[:6]) + ", …, " + fmt(pos[-1], 2))

    # Etapa 7: velocidades
    linea("ETAPA 7 · Velocidades (add_velocity): diferencia entre pasos consecutivos")
    X = add_velocity(rem[np.newaxis, ...].astype(np.float32))
    print(f"  ENTRA: {rem[np.newaxis].shape}")
    print(f"  SALE : {X.shape} = 126 de posición + 126 de velocidad (v_j = p_j − p_(j−1); v_0 = 0)")

    # Verificación (a): camino del entrenador
    with contextlib.redirect_stdout(io.StringIO()):
        Xp, y, lengths, nombres = cargar_datos(PROCESSED, REPRESENTACION)
    i = nombres.index(muestra)
    Xe = add_velocity(remuestrear_todas(Xp[i:i + 1], lengths[i:i + 1], PASOS))
    dif_ent = float(np.abs(Xe - X).max())
    print(f"  COTEJO con el ENTRENADOR (cargar_datos + remuestrear_todas + add_velocity sobre "
          f"{PROCESSED.relative_to(RAIZ)}): forma {Xe.shape}, T guardado {int(lengths[i])}, "
          f"diferencia máxima {dif_ent:.3g}")
    del Xp

    d.update({"T": T, "manos": manos, "remuestreo": rem, "posiciones": pos, "X": X, "cruda": cruda,
              "norm": norm, "sin_izq": sin_izq, "sin_der": sin_der, "cuadros": cuadros,
              "formas": {"video": [T, cab["alto"], cab["ancho"], 3], "puntos_por_cuadro": [T, 543, 3],
                         "fila": list(cruda.shape), "manos": list(manos.shape), "remuestreo": list(rem.shape),
                         "entrada_modelo": list(X.shape)},
              "entrenador": {"T_guardado": int(lengths[i]), "dif_max": dif_ent, "forma": list(Xe.shape)}})
    return d


def predecir(d, modelo, umbral):
    linea("ETAPA 8 · Modelo canónico: Conv1D + BiLSTM + atención (TensorFlow/Keras)")
    prob = np.asarray(modelo.predict(d["X"], verbose=0))[0]
    orden = np.argsort(prob)[::-1]
    top3 = [{"id": int(j), "glosa": SIGN_VOCABULARY[int(j)], "p": float(prob[j])} for j in orden[:3]]
    clase = top3[0]
    decision = "aceptada" if clase["p"] >= umbral else "rechazada_confianza"
    enunciado = GrammarMapper().translate([clase["glosa"]]) if decision == "aceptada" else None
    print(f"  ENTRA: {d['X'].shape}")
    print(f"  SALE : {prob.shape[0]} probabilidades (softmax; suman {fmt(float(prob.sum()), 4)})")
    for r, t in enumerate(top3, 1):
        print(f"         top-{r}: {t['id']:2d} «{t['glosa']}»  p = {fmt(t['p'])}")
    print(f"         decisión con umbral {fmt(umbral, 2)} (D-09): {decision}"
          + ("  ✓ coincide con la seña real" if clase["id"] == d["sena_id"] else
             f"  ✗ la seña real es {d['sena_id']} «{d['glosa']}»"))
    linea("ETAPA 9 · GrammarMapper → texto para la voz (aquí no se reproduce audio)")
    print(f"  ENTRA: glosa «{clase['glosa']}»   SALE: «{enunciado}»" if enunciado else
          "  (rechazada: el prototipo pide repetir la seña y no emite audio)")
    d.update({"prob": prob, "top3": top3, "decision": decision, "enunciado": enunciado, "umbral": umbral,
              "acierto": clase["id"] == d["sena_id"],
              "rango_real": int(np.where(orden == d["sena_id"])[0][0]) + 1})


# ------------------------------------------------------------------------------------------------
# Verificación (c): el prototipo (LSCTranslator) alimentado con los cuadros del video
# ------------------------------------------------------------------------------------------------

class RelojDelVideo:
    """Reloj en ns para LandmarkExtractor: la 1.ª llamada (en __init__) vale 0 y la llamada i+1 vale
    round(i·1000/fps) ms, de modo que las marcas del modo VIDEO son las de la re-extracción del corpus."""

    def __init__(self, fps):
        self.fps, self.n = fps, 0

    def __call__(self):
        v = 0 if self.n == 0 else rx.ts_fotograma(self.n - 1, self.fps) * 1_000_000
        self.n += 1
        return v


def prototipo(d, reloj_video=False):
    extractor = None
    if reloj_video:
        with contextlib.redirect_stdout(io.StringIO()):
            extractor = trm.LandmarkExtractor(trm._ensure_models_downloaded(), reloj=RelojDelVideo(d["video"]["fps"]))
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        tr = trm.LSCTranslator(model_path=str(MODELO), tts_mode="none", registro=False,
                               ventana_frames=d["T"], ventana_max_seg=3600.0, extractor=extractor)
    entradas = []
    pred = tr.modelo.predict
    tr.modelo.predict = lambda X, verbose=0: (entradas.append(np.array(X)), pred(X, verbose=verbose))[1]
    tr.abrir_ventana()
    reg = None
    for f in d["cuadros"]:
        reg = tr.procesar_fotograma(f).registro or reg
    tr.close()
    Xp = entradas[0] if entradas else None
    # ¿En cuántos pasos cambia la presencia de cada mano (bloque en ceros) frente al camino del corpus?
    cero = lambda M: np.stack([(M[0, :, :63] == 0).all(axis=1), (M[0, :, 63:126] == 0).all(axis=1)], axis=1)
    distinta = int((cero(Xp) != cero(d["X"])).any(axis=1).sum()) if Xp is not None else None
    return {"estado": reg["estado"], "clase": reg["clase"], "id_clase": reg["id_clase"],
            "confianza": reg["confianza"], "top3": reg["top3"], "umbral": tr.umbral,
            "umbral_origen": str(Path(tr.umbral_origen).relative_to(RAIZ)) if Path(tr.umbral_origen).is_absolute() else tr.umbral_origen,
            "representacion": tr.representacion, "mano_ausente": tr.mano_ausente,
            "forma_entrada": list(Xp.shape) if Xp is not None else None,
            "dif_max_entrada_vs_corpus": float(np.abs(Xp - d["X"]).max()) if Xp is not None else None,
            "dif_max_prob_vs_corpus": float(np.abs(np.array([t["p"] for t in reg["top3"]])
                                                   - d["prob"][[t["id"] for t in reg["top3"]]]).max()),
            "misma_clase": reg["id_clase"] == d["top3"][0]["id"],
            "pasos_con_presencia_de_mano_distinta": distinta,
            "cuadros_sin_manos_ventana": reg.get("frames_sin_manos"),
            "extractor_ms_media": (reg["costo_extractor_ns"]["media"] / 1e6) if reg.get("costo_extractor_ns") else None,
            "inferencia_ms": (reg["etapas_ns"]["inferencia"] / 1e6) if reg["etapas_ns"].get("inferencia") else None}


# ------------------------------------------------------------------------------------------------
# Ejemplo numérico de interpolación con valores reales
# ------------------------------------------------------------------------------------------------

def mano_dominante(d):
    """La mano que más se mueve (suma de |Δ| de la muñeca normalizada en los cuadros con la mano)."""
    mov = {}
    for nombre, c0, ausente in (("izquierda", 99, d["sin_izq"]), ("derecha", 162, d["sin_der"])):
        w = d["norm"][:, c0:c0 + 2]
        ok = ~ausente
        mov[nombre] = float(np.abs(np.diff(w[ok], axis=0)).sum()) if ok.sum() > 1 else 0.0
    return max(mov, key=mov.get), mov


def ejemplo_interpolacion(d):
    mano, mov = mano_dominante(d)
    base = 0 if mano == "izquierda" else 63              # columna dentro de los 126
    ausente = d["sin_izq"] if mano == "izquierda" else d["sin_der"]
    x = d["manos"][:, base]                              # x de la muñeca (punto 0)
    mejor = None
    for j, p in enumerate(d["posiciones"]):
        a = int(np.floor(p)); f = p - a
        if a + 1 >= d["T"] or ausente[a] or ausente[a + 1] or not 0.2 <= f <= 0.8:
            continue
        # el paso anterior también debe venir de cuadros con la mano, para que la velocidad del ejemplo
        # no mezcle el 0 de la máscara (D-07)
        pa = d["posiciones"][j - 1] if j > 0 else None
        if pa is None or ausente[int(np.floor(pa))] or ausente[min(int(np.floor(pa)) + 1, d["T"] - 1)]:
            continue
        s = abs(x[a + 1] - x[a]) * min(f, 1 - f)
        if mejor is None or s > mejor[0]:
            mejor = (s, j, a, f)
    _, j, a, f = mejor
    val = float(x[a] + f * (x[a + 1] - x[a]))
    rem = float(d["remuestreo"][j, base])
    v = float(d["X"][0, j, 126 + base])
    ej = {"mano": mano, "movimiento": mov, "punto": "muñeca (0)", "coordenada": "x normalizada",
          "columna_en_126": base, "paso": j, "T": d["T"], "posicion": float(d["posiciones"][j]),
          "cuadro_a": a, "cuadro_b": a + 1, "fraccion": float(f), "x_a": float(x[a]), "x_b": float(x[a + 1]),
          "x_interpolado": val, "x_resample_sequence": rem, "dif": abs(val - rem),
          "paso_anterior_x": float(d["remuestreo"][j - 1, base]), "velocidad": v}
    linea("EJEMPLO NUMÉRICO · interpolación lineal con valores reales de esta muestra")
    print(f"  mano que más se mueve: {mano} (Σ|Δ| muñeca: izq. {fmt(mov['izquierda'], 2)}, der. {fmt(mov['derecha'], 2)})")
    print(f"  paso j = {j}: posición = {j} × ({d['T']} − 1) / 29 = {fmt(ej['posicion'])} → entre el cuadro {a} "
          f"y el {a + 1}, fracción {fmt(f)}")
    print(f"  x(muñeca) = x_{a} + {fmt(f)} × (x_{a + 1} − x_{a}) = {fmt(ej['x_a'])} + {fmt(f)} × "
          f"({fmt(ej['x_b'])} − {fmt(ej['x_a'])}) = {fmt(val)}")
    print(f"  resample_sequence da {fmt(rem)} (diferencia {abs(val - rem):.2g})")
    print(f"  velocidad en ese paso: v_{j} = p_{j} − p_{j - 1} = {fmt(rem)} − {fmt(ej['paso_anterior_x'])} = {fmt(v)}")
    d["ejemplo"] = ej


# ------------------------------------------------------------------------------------------------
# Figuras (solo puntos: ninguna imagen de la persona)
# ------------------------------------------------------------------------------------------------

AZUL, NARANJA, AGUA, AMARILLO, MAGENTA, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
GRIS, GRIS_CLARO, TINTA, TINTA2 = "#8a8985", "#d9d8d4", "#0b0b0b", "#52514e"
COLOR_DEDO = {"muñeca": TINTA2, "pulgar": AZUL, "índice": NARANJA, "medio": AGUA, "anular": AMARILLO, "meñique": MAGENTA}
CONEX_MANO = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
CONEX_POSTURA = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
                 (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
                 (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20)]


def _estilo(plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRIS,
                         "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
                         "pdf.fonttype": 42, "legend.frameon": False})


def fig_esqueleto(d, ruta, plt):
    T = d["T"]
    # Cuadro ilustrativo: aquel en que la muñeca de la mano que más se mueve está más alta (la seña en
    # curso, no el reposo), entre los cuadros con esa mano detectada.
    dom = mano_dominante(d)[0]
    aus = d["sin_izq"] if dom == "izquierda" else d["sin_der"]
    ywr = d["cruda"][:, (99 if dom == "izquierda" else 162) + 1].astype(float)
    ywr[aus] = np.inf
    k = int(np.argmin(ywr))
    W, H = d["video"]["ancho"], d["video"]["alto"]
    c = d["cruda"][k]
    post = c[:99].reshape(33, 3)[:, :2] * [W, H]
    manos = {"izquierda": c[99:162].reshape(21, 3)[:, :2] * [W, H],
             "derecha": c[162:225].reshape(21, 3)[:, :2] * [W, H]}
    presente = {"izquierda": not d["sin_izq"][k], "derecha": not d["sin_der"][k]}
    mano_zoom = mano_dominante(d)[0]
    if not presente[mano_zoom]:
        mano_zoom = "izquierda" if presente["izquierda"] else "derecha"

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 5.6), gridspec_kw={"width_ratios": [1, 1]})
    for a in (a1, a2):
        a.set_facecolor("white"); a.set_aspect("equal"); a.invert_yaxis(); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    for i, j in CONEX_POSTURA:
        a1.plot(*post[[i, j]].T, color=GRIS, lw=2, zorder=1)
    a1.scatter(*post[:25].T, s=14, color=GRIS, zorder=2)
    a1.scatter(*post[[11, 12]].T, s=60, color=VIOLETA, zorder=3)
    medio = post[[11, 12]].mean(axis=0)
    a1.scatter(*medio, s=60, marker="+", color=VIOLETA, zorder=3, linewidths=2)
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], marker="o", ls="", color=VIOLETA, markersize=8,
                              label="hombros 11 y 12: referencia y escala")],
              loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=13,
              handletextpad=0.3)
    for nombre, pts in manos.items():
        if not presente[nombre]:
            continue
        for i, j in CONEX_MANO:
            a1.plot(*pts[[i, j]].T, color=TINTA2, lw=1.2, zorder=4)
        for dedo, idx in DEDOS.items():
            a1.scatter(*pts[idx].T, s=10, color=COLOR_DEDO[dedo], zorder=5)
        fuera = pts[:, 0].max() if pts[0, 0] > medio[0] else pts[:, 0].min()   # lado exterior de la mano
        a1.annotate(f"mano\n{nombre}", (fuera, pts[0, 1]),
                    xytext=(10 if pts[0, 0] > medio[0] else -10, 0), textcoords="offset points",
                    ha="left" if pts[0, 0] > medio[0] else "right", va="center", fontsize=14, color=TINTA,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.95), zorder=6)
    todos = np.vstack([post[:25]] + [p for n, p in manos.items() if presente[n]])
    x0, y0 = todos.min(axis=0); x1, y1 = todos.max(axis=0)
    a1.set_xlim(x0 - 470, x1 + 470); a1.set_ylim(y1 + 90, y0 - 110)
    a1.set_title(f"Cuadro {k} de {T}\npostura 33 + manos 21 + 21", fontsize=18, color=TINTA)

    pts = manos[mano_zoom]
    for i, j in CONEX_MANO:
        dedo = next(n for n, idx in DEDOS.items() if j in idx)
        a2.plot(*pts[[i, j]].T, color=COLOR_DEDO[dedo], lw=2.2, zorder=1)
    esc = np.ptp(pts, axis=0).max()
    for dedo, idx in DEDOS.items():
        a2.scatter(*pts[idx].T, s=520, color="white", edgecolors=COLOR_DEDO[dedo], linewidths=2, zorder=2,
                   label=f"{dedo} ({idx[0]}" + (f"–{idx[-1]})" if len(idx) > 1 else ")"))
        for p in idx:
            a2.text(*pts[p], str(p), ha="center", va="center", fontsize=15, fontweight="bold", color=TINTA, zorder=3)
    a2.set_xlim(pts[:, 0].min() - 0.18 * esc, pts[:, 0].max() + 0.18 * esc)
    a2.set_ylim(pts[:, 1].max() + 0.18 * esc, pts[:, 1].min() - 0.18 * esc)
    a2.set_title(f"Mano {mano_zoom}\n21 puntos, cada uno (x, y, z)", fontsize=18, color=TINTA)
    a2.legend(loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=16, handletextpad=0.2, markerscale=0.6,
               labelspacing=0.9)
    # El pie va en la diapositiva (una sola vez).
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
    return {"cuadro": k, "mano_ampliada": mano_zoom}


def fig_trayectoria(d, ruta, plt):
    ej = d["ejemplo"]
    mano = ej["mano"]
    base = 0 if mano == "izquierda" else 63
    ausente = d["sin_izq"] if mano == "izquierda" else d["sin_der"]
    T, pos = d["T"], d["posiciones"]
    t = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True)
    for fila, (coord, off) in enumerate((("x", 0), ("y", 1))):
        a = axes[fila]
        for p, color, nombre in ((0, AZUL, "muñeca (0)"), (8, NARANJA, "punta del índice (8)")):
            col = base + 3 * p + off
            orig = d["manos"][:, col].astype(float).copy()
            orig[ausente] = np.nan
            a.plot(t, orig, color=color, lw=1.4, alpha=0.9, zorder=1)
            a.scatter(t[~ausente], orig[~ausente], s=9, color=color, alpha=0.55, zorder=2,
                      label=f"{nombre}: {T} cuadros originales")
            r = d["remuestreo"][:, col]
            a.scatter(pos, r, s=46, marker="D", facecolor="white", edgecolor=color, linewidths=1.6, zorder=3,
                      label=f"{nombre}: 30 pasos interpolados")
        for i in np.where(ausente)[0]:
            a.axvspan(i - 0.5, i + 0.5, color=GRIS_CLARO, zorder=0, lw=0)
        a.set_ylabel(f"{coord} normalizada\n(anchos de hombros)")
        if coord == "y":
            a.invert_yaxis()
            a.set_ylabel("y normalizada (anchos de hombros)\n↑ la mano sube")
    j = ej["paso"]
    axes[0].annotate(f"paso {j}: posición {fmt(ej['posicion'], 3)}\n= cuadro {ej['cuadro_a']} + {fmt(ej['fraccion'], 3)}",
                     (ej["posicion"], ej["x_resample_sequence"]), xytext=(-190, 30), textcoords="offset points",
                     fontsize=8.5, color=TINTA, arrowprops=dict(arrowstyle="-", color=TINTA2, lw=0.8))
    axes[1].set_xlabel("número de cuadro del video (24 cuadros/s)")
    sec = axes[0].secondary_xaxis("top", functions=(lambda c: c * 29 / (T - 1), lambda s: s * (T - 1) / 29))
    sec.set_xticks([0, 5, 10, 15, 20, 25, 29]); sec.set_xlabel("paso remuestreado (0 a 29)", color=TINTA2)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=8.5)
    txt = f"{d['muestra']} («{d['glosa']}»), mano {mano} de la persona. "
    txt += (f"Franjas grises: cuadros sin esa mano ({int(ausente.sum())}); ahí el valor es 0 («máscara», D-07) y "
            f"entra en la interpolación." if ausente.any() else "La mano se detectó en todos los cuadros.")
    fig.text(0.01, 0.005, txt, fontsize=7.5, color=TINTA2)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(ruta); plt.close(fig)


def fig_matriz(d, ruta, plt):
    from matplotlib.colors import TwoSlopeNorm
    X = d["X"][0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    partes = (("Posición (126 = 2 manos × 21 puntos × x, y, z)", X[:, :126]),
              ("Velocidad (126 = diferencia con el paso anterior)", X[:, 126:]))
    for n_parte, (a, (titulo, M)) in enumerate(zip(axes, partes)):
        v = float(np.abs(M).max()) or 1.0
        im = a.imshow(M, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), interpolation="nearest")
        a.axvline(62.5, color=TINTA, lw=1.5)
        a.set_title(titulo, fontsize=10.5, color=TINTA, pad=22)
        a.set_xticks([0, 62, 125]); a.set_xticklabels(["0", "62 | 63", "125"])
        a.text(31, -0.9, "mano izquierda (63)", ha="center", va="bottom", fontsize=9, color=TINTA2)
        a.text(94, -0.9, "mano derecha (63)", ha="center", va="bottom", fontsize=9, color=TINTA2)
        a.set_xlabel("columna (p0x, p0y, p0z, p1x, …, p20z)")
        cb = fig.colorbar(im, ax=a, orientation="horizontal", fraction=0.05, pad=0.16)
        cb.set_label("valor (anchos de hombros" + (")" if n_parte == 0 else " por paso)"), fontsize=8.5)
        cb.outline.set_visible(False)
    axes[0].set_ylabel("paso (0 a 29)")
    axes[0].set_yticks([0, 5, 10, 15, 20, 25, 29])
    fig.suptitle(f"Entrada al modelo para {d['muestra']} («{d['glosa']}»): matriz 30 × 252", fontsize=11.5,
                 color=TINTA, y=0.995)
    fig.text(0.01, 0.005, "Cada mitad tiene su propia escala de color. Rojo = positivo, azul = negativo, blanco = 0 "
             "(también el valor de una mano ausente).\nLa fila 0 de velocidad es 0 por definición.",
             fontsize=7.5, color=TINTA2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(ruta); plt.close(fig)


def fig_probabilidades(d, ruta, plt):
    prob, top = d["prob"], d["top3"]
    ids = np.arange(prob.size)
    colores = [GRIS_CLARO] * prob.size
    for t, c in zip(top, (AZUL, NARANJA, AGUA)):
        colores[t["id"]] = c
    fig, a = plt.subplots(figsize=(11.5, 4.6))
    a.bar(ids, prob, color=colores, width=0.78, zorder=2)
    a.axhline(d["umbral"], color=TINTA, lw=1.2, ls=(0, (5, 3)), zorder=3)
    a.text(prob.size - 0.5, d["umbral"] + 0.02, f"umbral {fmt(d['umbral'], 2)} (D-09): por debajo, «repita la seña»",
           ha="right", fontsize=12, color=TINTA)
    for r, t in enumerate(top, 1):
        a.annotate(f"top-{r}: {t['glosa']}\np = {fmt(t['p'], 3)}", (t["id"], t["p"]), xytext=(0, 6 if r == 1 else 10),
                   textcoords="offset points", ha="center", fontsize=12, color=TINTA)
    a.set_xticks([])
    a.set_xlabel("las 50 señas (una barra por seña; se rotulan las 3 más probables)", fontsize=12)
    a.set_xlim(-0.8, prob.size - 0.2); a.set_ylim(0, 1.12)
    a.set_ylabel("probabilidad (softmax)", fontsize=12)
    a.grid(axis="y", color=GRIS_CLARO, lw=0.6, zorder=0)
    a.set_title(f"Salida del modelo para {d['muestra']} (seña real: {d['sena_id']} «{d['glosa']}»): 50 probabilidades "
                f"que suman 1", fontsize=11, color=TINTA)
    fig.tight_layout()
    fig.savefig(ruta); plt.close(fig)


def fig_duraciones(corridas, ruta, plt):
    fig, a = plt.subplots(figsize=(11, 0.9 + 0.85 * len(corridas)))
    tmax = max(c["T"] / c["video"]["fps"] for c in corridas)
    for fila, c in enumerate(corridas):
        fps, T = c["video"]["fps"], c["T"]
        yv = len(corridas) - 1 - fila
        a.scatter(np.arange(T) / fps, np.full(T, yv), marker="|", s=90, color=GRIS, lw=1, zorder=1)
        a.scatter(c["posiciones"] / fps, np.full(PASOS, yv), marker="o", s=22, color=AZUL, zorder=2)
        a.text(-0.05, yv + 0.28, f"{c['muestra']} «{c['glosa']}»: {T} cuadros = {fmt(T / fps, 3)} s  →  30 × 252",
               fontsize=9, color=TINTA, ha="left")
    a.set_yticks([]); a.set_ylim(-0.6, len(corridas) - 0.25)
    a.set_xlim(-0.1, tmax + 0.1)
    a.spines["left"].set_visible(False)
    a.set_xlabel("tiempo en el video (s)")
    a.scatter([], [], marker="|", s=90, color=GRIS, label="cuadro del video")
    a.scatter([], [], marker="o", s=22, color=AZUL, label="uno de los 30 pasos (interpolado)")
    a.legend(loc="upper right", bbox_to_anchor=(1, 1.02 + 0.35 / len(corridas)), ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ruta); plt.close(fig)


def figuras(corridas, carpeta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _estilo(plt)
    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    d = corridas[0]
    m = d["muestra"]
    rutas = {"esqueleto": carpeta / f"escritorio_a_esqueleto_{m}.pdf",
             "trayectoria": carpeta / f"escritorio_b_interpolacion_{m}.pdf",
             "matriz": carpeta / f"escritorio_c_matriz_entrada_{m}.pdf",
             "probabilidades": carpeta / f"escritorio_d_probabilidades_{m}.pdf"}
    info = fig_esqueleto(d, rutas["esqueleto"], plt)
    fig_trayectoria(d, rutas["trayectoria"], plt)
    fig_matriz(d, rutas["matriz"], plt)
    fig_probabilidades(d, rutas["probabilidades"], plt)
    if len(corridas) > 1:
        rutas["duraciones"] = carpeta / "escritorio_e_duraciones.pdf"
        fig_duraciones(corridas, rutas["duraciones"], plt)
    linea("FIGURAS")
    for k, r in rutas.items():
        print(f"  {k:15s} {r}")
    return {k: str(v) for k, v in rutas.items()}, info


# ------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Prueba de escritorio de una o varias muestras de LSC50.")
    ap.add_argument("muestras", nargs="*", default=["0030_0000_0000"], help="SSSS_VVVV_RRRR (defecto 0030_0000_0000)")
    ap.add_argument("--figuras", type=Path, help="carpeta donde escribir las figuras PDF")
    ap.add_argument("--json", type=Path, help="ruta del resumen JSON")
    ap.add_argument("--tmp", type=Path, help="carpeta para los CSV intermedios (defecto: temporal que se borra)")
    ap.add_argument("--sin-prototipo", action="store_true", help="omitir la verificación con LSCTranslator")
    args = ap.parse_args(argv)

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:  # noqa: BLE001
        pass
    procedencia = rp.leer_procedencia(MODELO)
    info_umbral = rp.leer_umbral(MODELO)
    umbral = float(info_umbral["umbral"])
    print("=" * 96)
    print(f"  PRUEBA DE ESCRITORIO · modelo {MODELO.relative_to(RAIZ)}")
    print(f"  procedencia: representación «{procedencia['representacion']}», {procedencia['n_rasgos_entrada']} rasgos, "
          f"{procedencia['seq_length']} pasos, velocidades {procedencia['velocidades']}, mano ausente "
          f"«{procedencia['mano_ausente']}», processed {procedencia['processed_dir']}")
    print(f"  umbral {umbral} ({info_umbral['decision']}, {Path(info_umbral['ruta']).relative_to(RAIZ)})")
    print("=" * 96)
    assert procedencia["representacion"] == REPRESENTACION and procedencia["mano_ausente"] == REGLA
    assert procedencia["processed_dir"] == str(PROCESSED.relative_to(RAIZ))
    modelo = tf.keras.models.load_model(MODELO)

    corridas = []
    with tempfile.TemporaryDirectory() as tmp_sistema:
        for muestra in args.muestras:
            print("\n" + "#" * 96 + f"\n#  MUESTRA {muestra}\n" + "#" * 96)
            tmp = (args.tmp / muestra) if args.tmp else Path(tmp_sistema) / muestra
            d = camino_corpus(muestra, tmp)
            predecir(d, modelo, umbral)
            ejemplo_interpolacion(d)
            if not args.sin_prototipo:
                linea("VERIFICACIÓN · prototipo LSCTranslator alimentado con los cuadros del video")
                d["prototipo"] = {}
                for nombre, rv in (("reloj_real", False), ("reloj_del_video", True)):
                    p = prototipo(d, reloj_video=rv)
                    d["prototipo"][nombre] = p
                    print(f"  {nombre:15s}: estado {p['estado']}, «{p['clase']}» p = {fmt(p['confianza'])}; entrada "
                          f"{tuple(p['forma_entrada'])}; |Δ entrada| máx. {p['dif_max_entrada_vs_corpus']:.3g}; "
                          f"|Δ prob. top-3| máx. {p['dif_max_prob_vs_corpus']:.3g}; misma clase: {p['misma_clase']}; "
                          f"pasos con otra presencia de mano: {p['pasos_con_presencia_de_mano_distinta']}; "
                          f"umbral {p['umbral']} ({p['umbral_origen']})")
            corridas.append(d)

    linea("RESUMEN")
    for d in corridas:
        print(f"  {d['muestra']} «{d['glosa']}» [{d['particion']['aleatoria']}]: T = {d['T']} cuadros "
              f"({fmt(d['video']['duracion_s'], 2)} s) → manos {tuple(d['manos'].shape)} → {tuple(d['remuestreo'].shape)} "
              f"→ {tuple(d['X'].shape)} → «{d['top3'][0]['glosa']}» {fmt(d['top3'][0]['p'], 3)} ({d['decision']})")

    rutas = info_fig = None
    if args.figuras:
        rutas, info_fig = figuras(corridas, args.figuras)

    if args.json:
        claves = ("muestra", "sena_id", "glosa", "senante", "repeticion", "particion", "video", "extraccion", "fila",
                  "normalizacion", "formas", "entrenador", "T", "top3", "decision", "enunciado", "umbral", "acierto",
                  "rango_real", "ejemplo", "prototipo")
        salida = {"script": "docs/diapositivas_workflow/guiones/prueba_escritorio.py",
                  "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"), "modelo": str(MODELO.relative_to(RAIZ)),
                  "procedencia": procedencia, "umbral": {k: v for k, v in info_umbral.items() if k != "ruta"},
                  "figuras": rutas, "figura_esqueleto": info_fig,
                  "corridas": [{**{k: d[k] for k in claves if k in d},
                                "posiciones_remuestreo": d["posiciones"].tolist(),
                                "probabilidades": d["prob"].tolist()} for d in corridas]}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(salida, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        print(f"  resumen JSON: {args.json}")


if __name__ == "__main__":
    main()
