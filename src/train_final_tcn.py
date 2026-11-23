"""
Modelo final TCN + imputación + 252 rasgos sobre el esquema aleatorio 80/10/10 (opción EXPERIMENTAL del
prototipo, pendiente de la decisión del director; N-63, docs/EXPERIMENTO_CONFIRMACION_TCN.md).

Qué entrena
-----------
La configuración que recomiendan docs/EXPERIMENTO_FACTORIAL_TCN.md y docs/EXPERIMENTO_CONFIRMACION_TCN.md
(brazo T1), con EXACTAMENTE el procedimiento con que src/train_final_100.py entrenó el modelo canónico
vigente (models/canonico_holistic_mascara_manos):

  · datos: data/processed_holistic_mascara_rostro, representación «manos» (126 posiciones normalizadas por
    hombros, con la regla «mascara» de D-07 ya aplicada), cargados con train_final_100.cargar_datos;
  · partición: data/partitions/aleatoria_80_10_10.json sin los 3 duplicados de D-06
    (train_final_100.cargar_particion: 797 / 100 / 100);
  · entrada: mano_ausente_variantes.remuestrear_todas_con_regla(·, «imputacion», 30) —interpolación por
    mano entre cuadros válidos sobre los cuadros originales ya normalizados, ANTES de remuestrear a 30
    pasos— -> aumentación ×10 -> add_velocity = 30 × 252. Es la misma función que usó el experimento de
    confirmación (experimento_confirmacion_tcn.preparar); verificar_entrada() lo comprueba bit a bit
    antes de entrenar;
  · red: arquitecturas_literatura.construir_tcn con CONFIG_T (156.402 parámetros);
  · receta: la de la TCN (arquitecturas_literatura.ENTRENAMIENTO_COMUN: Adam, lr 1e-3 con CosineDecay,
    lote 64, label smoothing 0,1), 200 épocas máximas, paciencia 25, early stopping y checkpoint sobre
    val_accuracy. Coincide con la del canónico salvo la red: por eso cada semilla se entrena con
    train_final_100.entrenar_semilla tal cual, con build_model sustituido por construir_tcn
    (arquitectura_tcn(); el `dropout` de build_model no aplica: CONFIG_T fija los suyos);
  · 5 semillas de entrenamiento, 2026-2030;
  · selección entre semillas SOLO con validación y el mismo criterio del canónico: mayor val_accuracy del
    checkpoint; empate, menor val_loss; luego el orden de las semillas;
  · el conjunto de prueba se evalúa una sola vez al final (modelo elegido y ensemble de las 5 semillas,
    como el canónico) con train_final_100.evaluar_en_prueba. Solo se reporta; no decide nada.

Salidas en models/tcn_imputacion/ (misma estructura que el canónico):
  semilla_<s>/            modelo.keras, modelo.procedencia.json, entrenamiento.json, historial.json, curvas
  modelo.keras            copia del elegido por validación
  modelo.procedencia.json campos de la del canónico + mano_ausente «imputacion» + arquitectura (CONFIG_T)
  resultados.json         configuración, semillas, selección, prueba, entorno y verificación de la entrada
  evaluacion_modelo_seleccionado/, evaluacion_ensemble/

El umbral D-09 de este modelo lo calcula src/umbral_tcn_imputacion.py (modelo.umbral.json).

Uso
---
    python src/train_final_tcn.py --dry-run                  # datos, partición y verificación de la entrada
    python src/train_final_tcn.py                            # 5 semillas (2026-2030), 200 épocas, aug ×10
    python src/train_final_tcn.py --epochs 1 --aug 1 --semillas 1 --out <carpeta_temporal>/humo   # humo
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import train_final_100 as tfc  # noqa: E402  (entrenador canónico: se reutiliza tal cual)
from src import arquitecturas_literatura as al  # noqa: E402
from src import mano_ausente_variantes as mav  # noqa: E402
from src.representacion import escribir_procedencia, ruta_procedencia  # noqa: E402

tf = tfc.tf

SCRIPT = "src/train_final_tcn.py"
PROCESSED = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
REPRESENTACION = "manos"
SEQ_LENGTH = 30
REGLA = "imputacion"
DEFAULT_OUT = BASE_DIR / "models" / "tcn_imputacion"
SEMILLA_BASE = 2026
N_PARAMETROS_TCN = 156_402          # CONFIG_T con 252 rasgos y 50 clases (experimento_confirmacion_tcn)
VIDEO_CONTROL = "0030_0000_0003"    # «hola», el mismo video de control del experimento de confirmación

DESCRIPCION_ARQUITECTURA = (
    "TCN pequeña con GAP (arquitecturas_literatura.construir_tcn, CONFIG_T): 4 bloques residuales con "
    "convoluciones dilatadas 1-2-4-8 (2 Conv1D de 64 filtros, ventana 3, BN, ReLU y dropout 0,2 por bloque; "
    "atajo 1×1 si cambian los canales), padding «same», GlobalAveragePooling1D, dropout 0,3 y softmax de 50")


def _rel(ruta) -> str:
    return tfc._rel(ruta)


# ---------------------------------------------------------------------------
# Arquitectura: build_model del canónico -> construir_tcn (solo mientras se entrena)
# ---------------------------------------------------------------------------

def construir_modelo_tcn(n_frames, n_features, n_classes, dropout=None):
    """Misma firma que train_final_100.build_model. `dropout` no se usa: CONFIG_T fija los suyos."""
    return al.construir_tcn(n_frames, n_features, n_classes)


@contextlib.contextmanager
def arquitectura_tcn():
    """Sustituye build_model por construir_tcn dentro de train_final_100 mientras dura el bloque, para que
    train_final_100.entrenar_semilla (semilla -> aumentación -> velocidades -> modelo -> Adam con CosineDecay
    -> fit con EarlyStopping y ModelCheckpoint sobre val_accuracy -> recarga del checkpoint) entrene la TCN
    sin reimplementar nada. Al salir se restituye build_model."""
    original = tfc.build_model
    tfc.build_model = construir_modelo_tcn
    try:
        yield
    finally:
        tfc.build_model = original


def comprobar_receta():
    """La receta que train_final_100.entrenar_semilla tiene fija (lr 1e-3, label smoothing 0,1) y el lote por
    defecto son los de la TCN (arquitecturas_literatura.ENTRENAMIENTO_COMUN). Se comprueba, no se supone."""
    import inspect
    fuente = inspect.getsource(tfc.entrenar_semilla)
    receta = al.ENTRENAMIENTO_COMUN
    assert receta == {"lr": 1e-3, "batch": 64, "label_smoothing": 0.1}, receta
    assert "CosineDecay(0.001, args.epochs * steps, alpha=1e-6)" in fuente
    assert "CategoricalCrossentropy(label_smoothing=0.1)" in fuente
    assert "build_model(X_seq.shape[1], n_features, n_clases, dropout=args.dropout)" in fuente
    return dict(receta, optimizador="Adam", programa_lr="CosineDecay(lr, epochs * pasos, alpha=1e-6)",
                monitor="val_accuracy", lote_entrenar_semilla="args.batch_size")


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------

def construir_X_seq(X_padded, lengths):
    """(N, 30, 126): regla «imputacion» de mano_ausente_variantes sobre los cuadros originales."""
    return mav.remuestrear_todas_con_regla(X_padded, lengths, REGLA, SEQ_LENGTH)


def verificar_entrada(X_padded, y, lengths, nombres, X_seq, idx) -> dict:
    """La entrada de este script es la del experimento de confirmación (brazo T1), bit a bit:
      1. X_seq = la «imputacion» de experimento_confirmacion_tcn.preparar (mismo processed y representación);
      2. el video de control = mav.regla_imputacion sobre sus cuadros reales;
      3. la aumentación ×10 de train_final_100.aumentar («manos») = train_loso.aumentar con las
         transformaciones de train_loso.importar_pipeline_canonico («manos»), que usó el experimento, con
         la misma semilla;
      4. add_velocity es el mismo objeto que usa el brazo T1;
      5. la TCN tiene los 156.402 parámetros del experimento."""
    from src import experimento_confirmacion_tcn as ec
    from src import train_loso as tl
    informe = {}
    Xc, yc, nombres_c, _, _, _, canon, _ = ec.preparar(PROCESSED)
    assert nombres_c == nombres and np.array_equal(yc, y)
    assert X_seq.dtype == Xc["imputacion"].dtype == np.float32
    assert np.array_equal(X_seq, Xc["imputacion"])
    informe["X_seq_igual_a_experimento_confirmacion_tcn"] = True
    informe["videos_que_cambian_respecto_de_mascara"] = int(np.sum(np.any(X_seq != Xc["mascara"], axis=(1, 2))))

    i0 = nombres.index(VIDEO_CONTROL)
    assert np.array_equal(X_seq[i0], mav.regla_imputacion(X_padded[i0, :int(lengths[i0])], SEQ_LENGTH))
    informe["video_control"] = VIDEO_CONTROL

    Xtr, ytr = X_seq[idx["train"]], y[idx["train"]]
    tf.keras.utils.set_random_seed(SEMILLA_BASE)
    a_X, a_y = tfc.aumentar(Xtr, ytr, 10, REPRESENTACION)
    tf.keras.utils.set_random_seed(SEMILLA_BASE)
    b_X, b_y = tl.aumentar(Xtr, ytr, 10, canon[4])
    assert np.array_equal(a_X, b_X) and np.array_equal(a_y, b_y)
    informe["aumentacion_x10_igual_a_la_del_experimento"] = True
    from src.experimento_confirmacion_tcn import canon_de_brazo
    assert canon_de_brazo(canon, "T1")[2] is tfc.add_velocity
    informe["add_velocity_igual_a_la_del_brazo_T1"] = True

    tf.keras.utils.set_random_seed(0)
    n_par = int(construir_modelo_tcn(SEQ_LENGTH, 2 * X_seq.shape[2], int(y.max()) + 1).count_params())
    tf.keras.backend.clear_session()
    assert n_par == N_PARAMETROS_TCN, n_par
    informe["n_parametros"] = n_par
    return informe


# ---------------------------------------------------------------------------
# Procedencia
# ---------------------------------------------------------------------------

def escribir_procedencia_tcn(ruta_modelo, processed_dir) -> Path:
    """Los campos de la procedencia del canónico (representacion.escribir_procedencia) con mano_ausente
    «imputacion» (la que lee el prototipo) y la arquitectura."""
    destino = escribir_procedencia(ruta_modelo, processed_dir, REPRESENTACION, SEQ_LENGTH, SCRIPT, BASE_DIR)
    datos = json.loads(destino.read_text(encoding="utf-8"))
    datos["mano_ausente"] = REGLA
    datos["mano_ausente_detalle"] = (
        "mano_ausente_variantes.imputar_fotogramas sobre los cuadros originales del processed «mascara» "
        "(normalizados por hombros, mano no detectada en 0): por mano, interpolación lineal entre el cuadro "
        "válido anterior y el siguiente; bordes = copia del válido más cercano; mano nunca detectada = 0. "
        "Después resample_sequence a 30 pasos y add_velocity")
    datos["arquitectura"] = {"descripcion": DESCRIPCION_ARQUITECTURA,
                             "constructor": "src/arquitecturas_literatura.py:construir_tcn",
                             "config_T": al.CONFIG_T, "n_parametros": N_PARAMETROS_TCN}
    datos["entrenamiento"] = dict(al.ENTRENAMIENTO_COMUN)
    datos["nota"] = ("opcion EXPERIMENTAL del prototipo (N-63), pendiente de la decision del director; el modelo "
                     "por defecto sigue siendo models/canonico_holistic_mascara_manos/modelo_canonico.keras")
    destino.write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")
    return destino


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="TCN + imputación + 252 sobre el esquema aleatorio 80/10/10, con el "
                                             "procedimiento del canónico (train_final_100.py)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--aug", type=int, default=10)
    ap.add_argument("--semillas", type=str, default="5",
                    help="número N de semillas (2026, 2027, ...) o lista 'a,b,c' (defecto 5: 2026-2030)")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--processed", type=str, default=str(PROCESSED))
    ap.add_argument("--dry-run", action="store_true", help="datos, partición y verificación de la entrada")
    ap.add_argument("--sin-verificar", action="store_true",
                    help="omite verificar_entrada (solo pruebas de humo fuera del processed canónico)")
    args = ap.parse_args(argv)

    if not 1 <= args.aug <= 10:
        raise SystemExit("--aug debe estar entre 1 y 10")
    semillas = tfc.parsear_semillas(args.semillas, SEMILLA_BASE)
    out_dir = Path(args.out).expanduser().resolve()
    protegidos = [BASE_DIR / "data", BASE_DIR / "models" / "canonico_holistic_mascara_manos",
                  BASE_DIR / "models" / "loso", BASE_DIR / "models" / "mediapipe",
                  BASE_DIR / "models" / "confirmacion_tcn_2026-10-03"]
    if out_dir == BASE_DIR / "models" or any(out_dir.is_relative_to(p) for p in protegidos):
        raise SystemExit(f"--out no puede ser {out_dir}")
    processed_dir = Path(args.processed).expanduser().resolve()
    # Argumentos que train_final_100.entrenar_semilla lee de `args`.
    ns = argparse.Namespace(aug=args.aug, epochs=args.epochs, patience=args.patience,
                            batch_size=al.ENTRENAMIENTO_COMUN["batch"], dropout=None,
                            representacion=REPRESENTACION, seq_length=SEQ_LENGTH, processed=processed_dir)

    print("=" * 72)
    print("  LSC50 - TCN + imputación + 252 (esquema aleatorio 80/10/10; procedimiento del canónico)")
    print("=" * 72)
    receta = comprobar_receta()
    X_padded, y, lengths, nombres = tfc.cargar_datos(processed_dir, REPRESENTACION)
    n_clases = int(y.max()) + 1
    print(f"\n[1] Datos ({_rel(processed_dir)}, representación {REPRESENTACION}): {len(nombres)} muestras, "
          f"{n_clases} clases; X_padded {X_padded.shape}")
    particion, idx = tfc.cargar_particion(nombres, y)
    for k in ("train", "val", "test"):
        print(f"    {k:5s}: {len(idx[k]):4d} muestras")
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    X_seq = construir_X_seq(X_padded, lengths)
    print(f"\n[2] Entrada: regla «{REGLA}» -> {X_seq.shape} (+ velocidades = {2 * X_seq.shape[2]} rasgos)")
    verificacion = None
    if not args.sin_verificar:
        verificacion = verificar_entrada(X_padded, y, lengths, nombres, X_seq, idx)
        print(f"    verificación: {json.dumps(verificacion, ensure_ascii=False)}")
    if args.dry_run:
        print("\n[dry-run] Datos, partición y entrada verificados. No se entrena nada.")
        return

    from src.experimento_arquitecturas import otros_entrenamientos
    otros = otros_entrenamientos()
    if otros:
        raise SystemExit(f"hay entrenamientos activos; no se entrena en paralelo: {otros}")

    print(f"\n[3] Plan: semillas {semillas} | epochs max {args.epochs} | aug x{args.aug} | lote "
          f"{ns.batch_size} | paciencia {args.patience} | salida {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ruta_resultados = out_dir / "resultados.json"
    resultados = {
        "script": SCRIPT,
        "esquema": "aleatoria_80_10_10",
        "processed_dir": _rel(processed_dir),
        "representacion": REPRESENTACION,
        "particion": _rel(tfc.ARCHIVO_PARTICION),
        "semilla_particion": particion.get("semilla"),
        "configuracion": {
            "arquitectura": "TCN (arquitecturas_literatura.construir_tcn, CONFIG_T)",
            "arquitectura_descripcion": DESCRIPCION_ARQUITECTURA,
            "config_T": al.CONFIG_T,
            "mano_ausente": REGLA,
            "seq_length": SEQ_LENGTH, "n_features": int(2 * X_seq.shape[2]),
            "velocidades": True, "epochs_max": args.epochs, "aug": args.aug,
            "batch_size": ns.batch_size, "patience": args.patience, "dropout": "CONFIG_T (0,2 por bloque; 0,3 final)",
            "receta": receta,
            "semillas_entrenamiento": semillas, "monitor": "val_accuracy",
            "entrenamiento_por_semilla": "train_final_100.entrenar_semilla con build_model -> construir_tcn",
        },
        "verificacion_entrada": verificacion,
        "semillas": [],
        "seleccion": None,
        "prueba": {},
        "entorno": tfc.info_entorno(),
    }

    def guardar():
        ruta_resultados.write_text(json.dumps(resultados, indent=1, ensure_ascii=False), encoding="utf-8")

    guardar()

    # Entrenamiento por semilla (el test no se toca).
    rutas = {}
    for s in semillas:
        print(f"\n{'=' * 72}\n  SEMILLA {s}\n{'=' * 72}", flush=True)
        with arquitectura_tcn():
            ficha, ruta = tfc.entrenar_semilla(s, X_seq, y, idx, ns, out_dir / f"semilla_{s}")
        escribir_procedencia_tcn(ruta, processed_dir)
        rutas[s] = ruta
        resultados["semillas"].append(ficha)
        guardar()

    # Selección SOLO con validación (mismo criterio y desempate que el canónico).
    fichas = resultados["semillas"]
    orden = sorted(range(len(fichas)), key=lambda i: (
        -fichas[i]["val_accuracy_checkpoint"], fichas[i]["val_loss_checkpoint"], i))
    sel = fichas[orden[0]]
    ruta_final = out_dir / "modelo.keras"
    shutil.copy2(rutas[sel["semilla"]], ruta_final)
    escribir_procedencia_tcn(ruta_final, processed_dir)
    resultados["seleccion"] = {
        "criterio": "mayor val_accuracy del checkpoint; empate: menor val_loss; luego orden de "
                    "las semillas. Sin intervención del conjunto de prueba.",
        "semilla": sel["semilla"],
        "val_accuracy_checkpoint": sel["val_accuracy_checkpoint"],
        "val_loss_checkpoint": sel["val_loss_checkpoint"],
        "metricas_validacion_checkpoint": sel["metricas_validacion_checkpoint"],
        "orden_por_validacion": [fichas[i]["semilla"] for i in orden],
        "modelo": _rel(ruta_final),
        "copiado_de": sel["modelo"],
    }
    guardar()
    print(f"\n[4] Seleccionado por validación: semilla {sel['semilla']} "
          f"(val_accuracy {sel['val_accuracy_checkpoint']:.4f}) -> {_rel(ruta_final)}")

    # Evaluación ÚNICA en prueba (solo reporte).
    print(f"\n[5] Evaluación única en prueba ({len(idx['test'])} muestras)")
    X_te_f = tfc.add_velocity(X_seq[idx["test"]])
    y_te = y[idx["test"]]
    muestras = list(particion["conjuntos"]["test"])
    nombres_senas = tfc.cargar_nombres_senas()
    probs = {}
    for s in semillas:
        m = tfc._cargar_modelo(rutas[s])
        probs[s] = m.predict(X_te_f, verbose=0)
        del m
        tf.keras.backend.clear_session()
    resultados["prueba"]["modelo_seleccionado"] = tfc.evaluar_en_prueba(
        f"modelo seleccionado (semilla {sel['semilla']})", probs[sel["semilla"]], y_te, muestras, nombres_senas,
        out_dir / "evaluacion_modelo_seleccionado")
    if len(semillas) > 1:
        resultados["prueba"]["ensemble"] = tfc.evaluar_en_prueba(
            f"ensemble de {len(semillas)} semillas (promedio de softmax)",
            np.mean([probs[s] for s in semillas], axis=0), y_te, muestras, nombres_senas,
            out_dir / "evaluacion_ensemble")
    guardar()
    print(f"\n  Resultados: {_rel(ruta_resultados)}")


if __name__ == "__main__":
    main()
