"""
Línea base clásica sobre características agregadas (actividades 2.c y 2.d; decisión D-10).

POR QUÉ: §2.2.4 del anteproyecto obliga a implementar al menos una línea base más simple
(clasificador clásico o MLP sobre características agregadas de la secuencia) y a compararla con la
red temporal; sin ella «el uso de un modelo temporal quedaría sin justificación empírica».

QUÉ HACE ESTE ARCHIVO, exactamente lo que fija D-10 y nada más:
  1. Datos: los MISMOS de la red. Particiones D-06 (data/partitions/), processed
     data/processed_holistic_mascara_rostro (regla de mano ausente «mascara», D-07),
     representación «manos» (D-08) y semilla 2026.
  2. Entrada: características agregadas sobre EXACTAMENTE la misma secuencia que recibe la red.
     La secuencia se construye con el código del entrenador (src.train_loso.cargar_datos +
     src.train_loso.remuestrear_todas con src.train_final_100.resample_sequence, y después
     src.train_final_100.add_velocity), de modo que el tensor por muestra es (30, 252):
     126 coordenadas de las dos manos + 126 velocidades. Aquí NO se reimplementa nada de eso;
     verificar_tensor() lo comprueba muestra a muestra contra el mismo camino del entrenador.
     Por cada una de las 252 columnas, ocho estadísticos fijados de antemano (ESTADISTICOS):
     media, desviación estándar, mínimo, máximo, mediana, primer paso, último paso y media de
     |diferencia entre pasos consecutivos| -> 2.016 rasgos. Sin PCA ni selección de rasgos.
     Escalado estándar (StandardScaler) ajustado SOLO con el entrenamiento de cada pliegue.
  3. Familias y rejilla (REJILLA): SVM RBF (C in {1, 10, 100}; gamma in {scale, 0,01}),
     Random Forest (400 árboles; max_features = sqrt; min_samples_leaf in {1, 2}) y MLP (una capa
     oculta de 256; alpha in {1e-4, 1e-2}; parada temprana). Con la aumentación de espejo del
     entrenamiento de la red (src.representacion.espejar) y, como control, sin ella.
  4. Selección: la configuración de cada familia se elige por top-1 en la VALIDACIÓN de ese
     pliegue. La prueba se obtiene a través de un invocable (obtener_prueba) al que ejecutar_corrida
     llama SOLO después de haber elegido; así la regla es comprobable (tests/test_linea_base.py).
  5. Métricas: top-1, top-3, precisión, recall y F1 macro; por pliegue, en agregado (las 997
     predicciones juntas) y en el esquema aleatorio.
  6. Comparación con la red: McNemar sobre las 997 predicciones de prueba agregadas del LOSO,
     alfa = 0,05, frente a models/loso_2026-09-15_holistic_mascara_manos (N-30).

LÍMITE DECLARADO (D-10): la línea base recibe la secuencia resumida en estadísticos, así que pierde
el orden temporal salvo lo que capturan el primer paso, el último y la media de las diferencias.
Esa es justamente la comparación que pide el anteproyecto.

PRECISIONES DE IMPLEMENTACIÓN (D-10 no las fija; se declaran aquí y en comparacion.md):
  · «la misma aumentación de espejo» se lee como el espejo y solo el espejo: entrenamiento =
    original + espejo (factor 2). La red usa además ruido, escalado y remuestreo temporal (factor
    10); esas no son de espejo y no se aplican aquí.
  · El espejo se aplica a la secuencia de 126 coordenadas ANTES de añadir velocidades, igual que en
    el entrenador de la red (aumentar() actúa sobre X_seq y add_velocity() va después).
  · La desviación estándar es la de numpy (ddof = 0).
  · top-1 y top-3 salen del mismo vector de probabilidades (predict_proba); la clase predicha es su
    argmax, como en la red. Por eso el SVM se ajusta con probability=True.
  · Empate en la validación: gana la primera configuración en el orden declarado de REJILLA.

Uso:
    python src/linea_base_clasica.py
    python src/linea_base_clasica.py --solo-loso
    python src/linea_base_clasica.py --processed data/processed_holistic_mascara_rostro \
                                     --salida models/linea_base_4d_2026-09-15
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Sesión con cámara en curso el 15-sep-2026: el uso de CPU se limita a dos hilos salvo que el
# entorno ya diga otra cosa. Debe ir antes de importar numpy/sklearn/TensorFlow de verdad.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

PROCESSED_POR_DEFECTO = BASE_DIR / "data" / "processed_holistic_mascara_rostro"
SALIDA_POR_DEFECTO = BASE_DIR / "models" / "linea_base_4d_2026-09-15"
RED_POR_DEFECTO = BASE_DIR / "models" / "loso_2026-09-15_holistic_mascara_manos"
PARTICION_LOSO = BASE_DIR / "data" / "partitions" / "loso_5_pliegues.json"
PARTICION_ALEATORIA = BASE_DIR / "data" / "partitions" / "aleatoria_80_10_10.json"

SEMILLA = 2026            # D-10, punto 1
SEQ_LENGTH = 30           # D-10, punto 2 (el mismo de la red)
REPRESENTACION = "manos"  # D-08 / N-30
N_CLASES = 50
ALFA = 0.05               # D-10, punto 6

# D-10, punto 2: ocho estadísticos por columna, en este orden fijo. El vector de rasgos es la
# concatenación por bloques: [media(252) | de(252) | min | max | mediana | primero | ultimo | mad].
ESTADISTICOS = ("media", "desviacion_estandar", "minimo", "maximo", "mediana",
                "primer_paso", "ultimo_paso", "media_abs_diferencia")

# D-10, punto 3. El orden importa: en caso de empate en validación gana la primera.
REJILLA: dict[str, list[tuple[str, dict]]] = {
    "svm_rbf": [
        ("C=1,gamma=scale",    {"C": 1.0,   "gamma": "scale"}),
        ("C=1,gamma=0.01",     {"C": 1.0,   "gamma": 0.01}),
        ("C=10,gamma=scale",   {"C": 10.0,  "gamma": "scale"}),
        ("C=10,gamma=0.01",    {"C": 10.0,  "gamma": 0.01}),
        ("C=100,gamma=scale",  {"C": 100.0, "gamma": "scale"}),
        ("C=100,gamma=0.01",   {"C": 100.0, "gamma": 0.01}),
    ],
    "random_forest": [
        ("min_samples_leaf=1", {"min_samples_leaf": 1}),
        ("min_samples_leaf=2", {"min_samples_leaf": 2}),
    ],
    "mlp": [
        ("alpha=1e-4", {"alpha": 1e-4}),
        ("alpha=1e-2", {"alpha": 1e-2}),
    ],
}
FAMILIAS = tuple(REJILLA)
NOMBRE_FAMILIA = {"svm_rbf": "SVM RBF", "random_forest": "Random Forest", "mlp": "MLP"}

# --- Aumentación (opción --aumentacion) ---------------------------------------------------------
# D-10 fijó «la misma aumentación de espejo del entrenamiento de la red»: original + espejo,
# factor 2. Pero la red NO entrena con factor 2 sino con factor 10 (src/train_loso.py: original +
# nueve transformaciones — espejo, tres niveles de ruido, dos temporales, escala, espejo+ruido y
# espejo+temporal). «completa» le da a la línea base EXACTAMENTE esas nueve transformaciones, con
# las funciones del entrenador (src/train_lstm.py a través de
# src.train_loso.importar_pipeline_canonico), para acotar cuánto de la diferencia con la red puede
# venir de que una reciba diez versiones de cada muestra y la otra dos. No sustituye a D-10: es un
# control añadido, y el modo por defecto sigue siendo «espejo».
AUMENTACIONES = ("espejo", "completa", "ninguna")
FACTOR_AUMENTACION = {"ninguna": 1, "espejo": 2, "completa": 10}
CLAVE_PRINCIPAL = {"espejo": "con_espejo", "completa": "con_aumentacion_completa",
                   "ninguna": "sin_aumentacion"}
DESCRIPCION_AUMENTACION = {
    "espejo": "espejo (src.representacion.espejar), factor 2; control sin espejo",
    "completa": ("completa: las mismas nueve transformaciones de la red (src/train_loso.py, "
                 "importar_pipeline_canonico), factor 10"),
    "ninguna": "sin aumentación (factor 1)",
}
ETIQUETA_AUMENTACION = {"espejo": "con aumentación de espejo",
                        "completa": "con la aumentación completa de la red (factor 10)",
                        "ninguna": "sin aumentación"}
TITULO_CONTROL_EQUIDAD = "## Control de equidad: misma aumentación que la red"


def _rel(ruta) -> str:
    ruta = Path(ruta)
    try:
        return str(ruta.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(ruta)


# ---------------------------------------------------------------------------
# Entrada: la MISMA secuencia que recibe la red (código del entrenador, no reimplementado)
# ---------------------------------------------------------------------------

def _pipeline_canonico():
    """resample_sequence y add_velocity del entrenador de la red (src/train_final_100.py, que los
    declara invariantes). Importa TensorFlow; por eso se hace tarde y una sola vez."""
    from src.train_final_100 import add_velocity, resample_sequence
    return resample_sequence, add_velocity


def cargar_secuencias(processed_dir=PROCESSED_POR_DEFECTO, representacion=REPRESENTACION,
                      seq_length=SEQ_LENGTH):
    """Secuencias remuestreadas tal como las construye src/train_loso.py antes de entrenar:
    (N, seq_length, 126) para la representación «manos». Devuelve (X_seq, y, nombres, senantes)."""
    from src.train_loso import cargar_datos, remuestrear_todas
    resample_sequence, _ = _pipeline_canonico()
    X_padded, y, lengths, nombres, senantes = cargar_datos(Path(processed_dir), representacion)
    X_seq = remuestrear_todas(X_padded, lengths, resample_sequence, seq_length)
    return X_seq, y, nombres, senantes


def tensor_de_red(X_seq: np.ndarray, seq_length=SEQ_LENGTH) -> np.ndarray:
    """Tensor que recibe la red: coordenadas + velocidades, (N, 30, 252) con «manos»."""
    _, add_velocity = _pipeline_canonico()
    X = add_velocity(np.asarray(X_seq, dtype=np.float32))
    if X.ndim != 3 or X.shape[1] != seq_length:
        raise AssertionError(f"el tensor por muestra debería tener {seq_length} pasos y tiene "
                             f"{X.shape}")
    return X


def verificar_tensor(X_seq, nombres, processed_dir=PROCESSED_POR_DEFECTO, muestras=None,
                     representacion=REPRESENTACION, seq_length=SEQ_LENGTH) -> dict:
    """Comprueba (a) que el tensor por muestra es (30, 252) y (b) que para tres muestras concretas
    coincide, valor a valor, con el que produce el entrenador de la red por su propio camino."""
    from src.train_loso import cargar_datos, remuestrear_todas
    resample_sequence, add_velocity = _pipeline_canonico()

    X = tensor_de_red(X_seq, seq_length)
    n_rasgos = 2 * 3 * 21 * 2   # (x,y,z) x 21 puntos x 2 manos, duplicado por las velocidades
    if X.shape[1:] != (seq_length, n_rasgos):
        raise AssertionError(f"el tensor por muestra es {X.shape[1:]} y debería ser "
                             f"({seq_length}, {n_rasgos})")

    # Camino del entrenador de la red, rehecho aquí de cero sobre el mismo processed.
    Xp, y_e, lengths, nombres_e, _ = cargar_datos(Path(processed_dir), representacion)
    if list(nombres_e) != list(nombres):
        raise AssertionError("el sample_order del processed no coincide con el cargado")
    muestras = list(muestras) if muestras else [nombres[0], nombres[len(nombres) // 2], nombres[-1]]
    pos = {n: i for i, n in enumerate(nombres)}
    comprobadas = []
    for nombre in muestras:
        i = pos[nombre]
        esperado = add_velocity(
            remuestrear_todas(Xp[i:i + 1], lengths[i:i + 1], resample_sequence, seq_length))[0]
        if esperado.shape != (seq_length, n_rasgos):
            raise AssertionError(f"{nombre}: el entrenador produce {esperado.shape}")
        if not np.array_equal(X[i], esperado):
            raise AssertionError(f"{nombre}: el tensor de la línea base no coincide con el del "
                                 f"entrenador de la red")
        comprobadas.append(nombre)
    return {"forma_por_muestra": [int(seq_length), int(n_rasgos)],
            "muestras_comprobadas": comprobadas,
            "identico_al_entrenador": True}


# ---------------------------------------------------------------------------
# Características agregadas (D-10, punto 2)
# ---------------------------------------------------------------------------

def estadisticos(X: np.ndarray) -> np.ndarray:
    """Los ocho estadísticos de D-10 sobre el eje del tiempo de X (N, T, F) -> (N, 8*F).

    Bloques, en el orden de ESTADISTICOS: media, desviación estándar (ddof = 0), mínimo, máximo,
    mediana, primer paso, último paso y media de |x[t+1] - x[t]|.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"se esperaba (N, T, F) y llegó {X.shape}")
    if X.shape[1] < 2:
        raise ValueError("hacen falta al menos dos pasos para la media de |diferencia|")
    bloques = [
        X.mean(axis=1),
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
        np.median(X, axis=1),
        X[:, 0, :],
        X[:, -1, :],
        np.abs(np.diff(X, axis=1)).mean(axis=1),
    ]
    return np.concatenate(bloques, axis=1).astype(np.float64)


def nombres_de_rasgos(n_columnas: int) -> list[str]:
    """Nombres de las 8*n_columnas características, en el mismo orden que estadisticos()."""
    return [f"{e}_col{c:03d}" for e in ESTADISTICOS for c in range(n_columnas)]


def aumentar_espejo(X_seq: np.ndarray, y: np.ndarray, representacion=REPRESENTACION):
    """Aumentación de espejo del entrenamiento de la red: original + espejo. La etiqueta no cambia
    (una seña espejada sigue siendo la misma seña; es la aumentación que ya usa train_loso)."""
    from src.representacion import espejar
    X_esp = espejar(np.asarray(X_seq, dtype=np.float32), representacion)
    return np.concatenate([X_seq, X_esp]), np.concatenate([y, y])


def transformaciones_de_la_red(representacion=REPRESENTACION):
    """Las NUEVE transformaciones con las que se entrena la red, tal cual las arma el entrenador.

    No se reimplementa ninguna: es la lista que devuelve src.train_loso.importar_pipeline_canonico,
    que a su vez toma augment_noise / augment_temporal / augment_scale de src/train_lstm.py y el
    espejado de la representación (src.representacion.espejar para «manos»). Orden y parámetros son
    los de refined_augmentation: espejo, ruido 0,005 / 0,012 / 0,020, temporal 0,85-0,95 y
    1,05-1,15, escala 0,85-1,15, espejo+ruido 0,008 y espejo+temporal 0,9-1,1.
    """
    from src.train_loso import importar_pipeline_canonico
    *_, transformaciones = importar_pipeline_canonico(representacion)
    esperadas = FACTOR_AUMENTACION["completa"] - 1
    if len(transformaciones) != esperadas:
        raise AssertionError(f"el entrenador declara {len(transformaciones)} transformaciones y la "
                             f"aumentación completa necesita {esperadas}")
    return transformaciones


def aumentar_completa(X_seq: np.ndarray, y: np.ndarray, representacion=REPRESENTACION,
                      semilla=SEMILLA):
    """Aumentación completa de la red: original + las nueve transformaciones (factor 10).

    Las transformaciones son las del entrenador (transformaciones_de_la_red); aquí solo se
    concatenan, en el mismo orden y sobre el mismo X que usa src.train_loso.aumentar. Diferencia
    deliberada y única con aumentar(): NO se baraja al final. Ese barajado existe para el descenso
    por lotes de Keras y no cambia nada en sklearn (SVM y Random Forest no dependen del orden, y el
    MLP hace su propio reparto estratificado de validación); dejar el original al principio permite
    comprobar, muestra a muestra, que la línea base ve el original intacto más nueve copias
    transformadas (tests/test_linea_base.py). Las transformaciones con azar (ruido, temporal,
    escala) usan np.random como en el entrenador; con `semilla` se siembra antes para que la
    corrida sea reproducible.
    """
    transformaciones = transformaciones_de_la_red(representacion)   # antes de sembrar: no sortea
    X_seq = np.asarray(X_seq)
    y = np.asarray(y)
    if semilla is not None:
        np.random.seed(semilla)
    partes_X, partes_y = [X_seq], [y]
    for _, fn in transformaciones[:FACTOR_AUMENTACION["completa"] - 1]:
        partes_X.append(fn(X_seq))
        partes_y.append(y.copy())
    return np.concatenate(partes_X), np.concatenate(partes_y)


def aumentar_muestras(X_seq: np.ndarray, y: np.ndarray, aumentacion: str = "espejo",
                      representacion=REPRESENTACION, semilla=SEMILLA):
    """Conjunto de entrenamiento aumentado según el modo: «espejo» (D-10, factor 2), «completa»
    (la de la red, factor 10) o «ninguna» (factor 1). La etiqueta nunca cambia."""
    if aumentacion not in AUMENTACIONES:
        raise ValueError(f"aumentación desconocida: {aumentacion!r}; opciones {list(AUMENTACIONES)}")
    if aumentacion == "ninguna":
        return np.asarray(X_seq), np.asarray(y)
    if aumentacion == "espejo":
        return aumentar_espejo(X_seq, y, representacion)
    return aumentar_completa(X_seq, y, representacion, semilla)


def verificar_aumentacion_completa(X_seq, y, representacion=REPRESENTACION, semilla=SEMILLA,
                                   muestra=0) -> dict:
    """Aserción SOBRE UNA MUESTRA: lo que aplica la línea base en modo «completa» es, valor a valor,
    lo que aplica el entrenador de la red.

    Se comparan las diez versiones de una sola muestra contra
    src.train_loso.aumentar(X, y, 10, transformaciones) con la misma semilla. Como aumentar()
    termina barajando con np.random.permutation y aquí no se baraja, se reproduce ese mismo sorteo
    (el estado del generador es idéntico en los dos caminos) y se compara la permutación aplicada.
    """
    from src.train_loso import aumentar
    X1 = np.asarray(X_seq)[muestra:muestra + 1]
    y1 = np.asarray(y)[muestra:muestra + 1]
    transformaciones = transformaciones_de_la_red(representacion)   # importa TF una vez, sin azar

    Xa, ya = aumentar_completa(X1, y1, representacion, semilla)      # siembra con `semilla`
    perm = np.random.permutation(len(ya))                           # el sorteo que hace aumentar()
    np.random.seed(semilla)
    Xb, yb = aumentar(X1, y1, FACTOR_AUMENTACION["completa"], transformaciones)

    if Xa.shape[0] != FACTOR_AUMENTACION["completa"]:
        raise AssertionError(f"la aumentación completa de una muestra debería dar 10 versiones y "
                             f"da {Xa.shape[0]}")
    if not np.array_equal(Xa[:1], X1):
        raise AssertionError("la primera versión debería ser el original intacto")
    if not (np.array_equal(Xa[perm], Xb) and np.array_equal(ya[perm], yb)):
        raise AssertionError("las transformaciones de la línea base no coinciden con las del "
                             "entrenador de la red (src.train_loso.aumentar)")
    return {"muestra_comprobada": int(muestra),
            "transformaciones": [n for n, _ in transformaciones],
            "factor": FACTOR_AUMENTACION["completa"],
            "identico_al_entrenador": True,
            "fuente": "src.train_loso.importar_pipeline_canonico (src/train_lstm.py + "
                      "src.representacion.espejar)"}


def rasgos_agregados(X_seq: np.ndarray, seq_length=SEQ_LENGTH) -> np.ndarray:
    """De secuencias (N, 30, 126) a la matriz de 2.016 características de D-10."""
    return estadisticos(tensor_de_red(X_seq, seq_length))


# ---------------------------------------------------------------------------
# Familias y rejilla (D-10, punto 3)
# ---------------------------------------------------------------------------

def crear_modelo(familia: str, parametros: dict, semilla=SEMILLA, n_jobs=2):
    """Estimador de sklearn de una configuración de la rejilla de D-10."""
    if familia == "svm_rbf":
        from sklearn.svm import SVC
        return SVC(kernel="rbf", C=parametros["C"], gamma=parametros["gamma"],
                   probability=True, random_state=semilla)
    if familia == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=400, max_features="sqrt",
                                      min_samples_leaf=parametros["min_samples_leaf"],
                                      random_state=semilla, n_jobs=n_jobs)
    if familia == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(256,), alpha=parametros["alpha"],
                             early_stopping=True, max_iter=500, random_state=semilla)
    raise ValueError(f"familia desconocida: {familia!r}; opciones {list(REJILLA)}")


def probabilidades(modelo, F: np.ndarray, n_clases=N_CLASES) -> np.ndarray:
    """Matriz (N, n_clases) con las columnas en el orden de las etiquetas 0..n_clases-1."""
    p = modelo.predict_proba(F)
    salida = np.zeros((F.shape[0], n_clases), dtype=np.float64)
    salida[:, np.asarray(modelo.classes_, dtype=int)] = p
    return salida


def metricas(y_true: np.ndarray, probs: np.ndarray, n_clases=N_CLASES) -> dict:
    """top-1, top-3, precisión, recall y F1 macro (D-10, punto 5), con el mismo criterio que
    src/train_loso.py: macro sobre las 50 etiquetas, zero_division = 0."""
    from sklearn.metrics import precision_recall_fscore_support
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(axis=1)
    # Orden estable: con empates (frecuentes en Random Forest) el argsort coincide con el argmax,
    # asi que top3[0] es siempre la clase predicha, como en los predicciones.json de la red.
    orden = np.argsort(-probs, axis=1, kind="stable")[:, :3]
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_clases)), average="macro", zero_division=0)
    return {
        "n": int(len(y_true)),
        "top1": float(np.mean(y_pred == y_true)),
        "top3": float(np.mean(np.any(orden == y_true[:, None], axis=1))),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "n_errores": int(np.sum(y_pred != y_true)),
    }


def predicciones_json(nombres, y_true, probs) -> list[dict]:
    """Mismo formato que models/loso_*/pliegue_k/predicciones.json."""
    y_pred = probs.argmax(axis=1)
    top3 = np.argsort(-probs, axis=1, kind="stable")[:, :3]   # empates: top3[0] == predicha
    if not np.array_equal(top3[:, 0], y_pred):
        raise AssertionError("top3[0] deberia ser siempre la clase predicha")
    return [{"muestra": str(nombres[i]), "real": int(y_true[i]), "predicha": int(y_pred[i]),
             "top3": [int(t) for t in top3[i]], "prob_predicha": float(probs[i, y_pred[i]])}
            for i in range(len(y_true))]


# ---------------------------------------------------------------------------
# Selección con VALIDACIÓN (D-10, punto 4). Estas funciones no reciben la prueba.
# ---------------------------------------------------------------------------

def elegir_configuracion(familia, F_tr, y_tr, F_val, y_val, *, semilla=SEMILLA, n_jobs=2,
                         rejilla=None, n_clases=N_CLASES, verboso=False):
    """Ajusta cada configuración de la familia con el ENTRENAMIENTO y la puntúa con la VALIDACIÓN;
    devuelve (nombre_elegido, modelo_elegido, detalle). La prueba no entra en esta función: su
    firma no la admite."""
    opciones = (rejilla or REJILLA)[familia]
    detalle, mejor = [], None
    for nombre, parametros in opciones:
        t0 = time.perf_counter()
        modelo = crear_modelo(familia, parametros, semilla=semilla, n_jobs=n_jobs)
        modelo.fit(F_tr, y_tr)
        m = metricas(y_val, probabilidades(modelo, F_val, n_clases), n_clases)
        m = {"configuracion": nombre, "parametros": {k: str(v) for k, v in parametros.items()},
             "validacion": m, "segundos": round(time.perf_counter() - t0, 1)}
        detalle.append(m)
        if verboso:
            print(f"      {familia:14s} {nombre:20s} val top-1 {m['validacion']['top1']:.4f} "
                  f"({m['segundos']:.0f} s)")
        # Empate: gana la primera en el orden declarado (> estricto).
        if mejor is None or m["validacion"]["top1"] > mejor[2]["validacion"]["top1"]:
            mejor = (nombre, modelo, m)
    return mejor[0], mejor[1], {"rejilla": detalle, "elegida": mejor[0],
                                "criterio": "top-1 en la validación del pliegue (D-10, punto 4)",
                                "validacion_elegida": mejor[2]["validacion"]}


def ejecutar_corrida(etiqueta, X_tr, y_tr, X_val, y_val, obtener_prueba, *,
                     familias=FAMILIAS, espejo=True, aumentacion=None,
                     representacion=REPRESENTACION,
                     semilla=SEMILLA, n_jobs=2, rejilla=None, n_clases=N_CLASES,
                     seq_length=SEQ_LENGTH, traza=None, verboso=False) -> dict:
    """Una corrida completa (un pliegue LOSO o el esquema aleatorio) para todas las familias.

    obtener_prueba es un invocable sin argumentos que devuelve (X_test, y_test, nombres_test). Se
    llama UNA sola vez y SOLO después de haber elegido la configuración de todas las familias con
    la validación: así la regla de D-10 («la prueba no se mira para elegir nada») es comprobable
    con un doble que falle si se la toca antes de tiempo (tests/test_linea_base.py).
    `traza`, si se pasa, recibe un registro de lo que se va haciendo, en orden.

    `aumentacion` es «espejo» (D-10, por defecto), «completa» (la de la red, factor 10) o
    «ninguna». `espejo` se mantiene por compatibilidad: si no se pasa `aumentacion`, True significa
    «espejo» y False «ninguna», exactamente como antes.
    """
    from sklearn.preprocessing import StandardScaler
    traza = traza if traza is not None else []
    t0 = time.perf_counter()

    modo = aumentacion if aumentacion is not None else ("espejo" if espejo else "ninguna")
    if modo not in AUMENTACIONES:
        raise ValueError(f"aumentación desconocida: {modo!r}; opciones {list(AUMENTACIONES)}")
    X_tr_u, y_tr_u = aumentar_muestras(X_tr, y_tr, modo, representacion, semilla)
    F_tr = rasgos_agregados(X_tr_u, seq_length)
    F_val = rasgos_agregados(X_val, seq_length)
    escalador = StandardScaler().fit(F_tr)          # D-10: ajustado solo con el entrenamiento
    F_tr_s, F_val_s = escalador.transform(F_tr), escalador.transform(F_val)
    traza.append({"paso": "rasgos", "corrida": etiqueta, "espejo": bool(modo != "ninguna"),
                  "aumentacion": modo, "factor_aumentacion": FACTOR_AUMENTACION[modo],
                  "n_train": int(F_tr.shape[0]), "n_rasgos": int(F_tr.shape[1]),
                  "n_val": int(F_val.shape[0])})
    if verboso:
        print(f"    {etiqueta} | aumentacion {modo} (factor {FACTOR_AUMENTACION[modo]}) | "
              f"entrenamiento {F_tr.shape[0]} x {F_tr.shape[1]} rasgos | "
              f"validacion {F_val.shape[0]}")

    elegidos = {}
    for familia in familias:
        nombre, modelo, detalle = elegir_configuracion(
            familia, F_tr_s, y_tr_u, F_val_s, y_val, semilla=semilla, n_jobs=n_jobs,
            rejilla=rejilla, n_clases=n_clases, verboso=verboso)
        elegidos[familia] = (modelo, detalle)
        traza.append({"paso": "seleccion", "corrida": etiqueta, "familia": familia,
                      "elegida": nombre, "val_top1": detalle["validacion_elegida"]["top1"]})

    # ---- A partir de aquí, y solo aquí, se toca la prueba ----
    X_te, y_te, nombres_te = obtener_prueba()
    traza.append({"paso": "prueba", "corrida": etiqueta, "n_test": int(len(y_te))})
    F_te_s = escalador.transform(rasgos_agregados(X_te, seq_length))
    y_te = np.asarray(y_te)

    salida = {"corrida": etiqueta, "espejo": bool(modo != "ninguna"),
              "aumentacion": modo, "factor_aumentacion": FACTOR_AUMENTACION[modo],
              "n_train_original": int(len(y_tr)), "n_train_usado": int(len(y_tr_u)),
              "n_val": int(len(y_val)), "n_test": int(len(y_te)),
              "n_rasgos": int(F_tr.shape[1]), "familias": {}}
    for familia in familias:
        modelo, detalle = elegidos[familia]
        probs = probabilidades(modelo, F_te_s, n_clases)
        salida["familias"][familia] = {
            "configuracion_elegida": detalle["elegida"],
            "seleccion": detalle,
            "prueba": metricas(y_te, probs, n_clases),
            "predicciones": predicciones_json(nombres_te, y_te, probs),
        }
        if verboso:
            p = salida["familias"][familia]["prueba"]
            print(f"      -> {familia:14s} [{detalle['elegida']}] prueba top-1 {p['top1']:.4f} | "
                  f"top-3 {p['top3']:.4f} | F1 {p['f1_macro']:.4f}")
    salida["segundos"] = round(time.perf_counter() - t0, 1)
    return salida


# ---------------------------------------------------------------------------
# Agregación y McNemar (D-10, puntos 5 y 6)
# ---------------------------------------------------------------------------

CLAVES_METRICA = ("top1", "top3", "precision_macro", "recall_macro", "f1_macro")


def agregar_pliegues(lista: list[dict]) -> dict:
    """Media y desviación estándar muestral (ddof = 1) entre pliegues, como en src/train_loso.py."""
    salida = {}
    for clave in CLAVES_METRICA:
        v = np.array([m[clave] for m in lista], dtype=float)
        salida[clave] = {"media": float(v.mean()),
                         "de": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                         "por_pliegue": [float(x) for x in v]}
    return salida


def mcnemar(aciertos_a, aciertos_b, exacto=None) -> dict:
    """McNemar pareado entre dos sistemas sobre las MISMAS muestras.

    aciertos_a / aciertos_b: booleanos «acertó» por muestra. Devuelve la tabla de discordancias
    (n10 = acierta A y falla B; n01 = falla A y acierta B), la chi-cuadrado con corrección de
    continuidad y el p exacto (binomial de dos colas con p = 0,5). Con n10 + n01 < 25 el exacto es
    el que manda; por eso `exacto` por defecto se decide con ese umbral y se declara cuál se usó.
    """
    from scipy import stats
    a = np.asarray(aciertos_a, dtype=bool)
    b = np.asarray(aciertos_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("los dos sistemas deben evaluarse sobre las mismas muestras")
    n11 = int(np.sum(a & b)); n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b)); n00 = int(np.sum(~a & ~b))
    n = n10 + n01
    if n == 0:
        p_chi2, chi2 = 1.0, 0.0
    else:
        chi2 = (abs(n10 - n01) - 1.0) ** 2 / n
        p_chi2 = float(stats.chi2.sf(chi2, 1))
    p_exacto = float(stats.binomtest(n10, n, 0.5).pvalue) if n else 1.0
    usar_exacto = (n < 25) if exacto is None else bool(exacto)
    return {
        "n_muestras": int(a.size),
        "tabla": {"aciertan_ambos": n11, "solo_a": n10, "solo_b": n01, "fallan_ambos": n00},
        "n_discordantes": n,
        "chi2_con_correccion_de_continuidad": float(chi2),
        "p_chi2": float(p_chi2),
        "p_exacto_binomial": p_exacto,
        "prueba_usada": "exacta binomial" if usar_exacto else "chi2 con corrección de continuidad",
        "p": p_exacto if usar_exacto else float(p_chi2),
        "alfa": ALFA,
        "significativo": bool((p_exacto if usar_exacto else p_chi2) < ALFA),
        "aciertos_a": int(np.sum(a)),
        "aciertos_b": int(np.sum(b)),
    }


def cargar_predicciones_red(dir_red=RED_POR_DEFECTO) -> tuple[list[dict], dict]:
    """Las 997 predicciones de prueba agregadas del LOSO de la red (N-30), pliegue a pliegue."""
    dir_red = Path(dir_red)
    filas, por_pliegue = [], []
    for k in range(5):
        ruta = dir_red / f"pliegue_{k}" / "predicciones.json"
        if not ruta.exists():
            raise SystemExit(f"falta {_rel(ruta)}: la comparación de D-10 necesita el LOSO de la red")
        p = json.loads(ruta.read_text(encoding="utf-8"))
        for fila in p:
            fila["pliegue"] = k
        filas.extend(p)
        por_pliegue.append(float(np.mean([f["real"] == f["predicha"] for f in p])))
    resumen = {
        "dir": _rel(dir_red),
        "n": len(filas),
        "top1_por_pliegue": por_pliegue,
        "top1_media_pliegues": float(np.mean(por_pliegue)),
        "top1_de_pliegues": float(np.std(por_pliegue, ddof=1)),
        "top1_agregado": float(np.mean([f["real"] == f["predicha"] for f in filas])),
        "top3_agregado": float(np.mean([f["real"] in f["top3"] for f in filas])),
    }
    return filas, resumen


def comprobar_referencia_red(resumen: dict, top1=0.580, de=0.086, tol=0.001) -> dict:
    """D-10 / N-30: antes de comparar nada hay que reproducir 0,580 ± 0,086 desde predicciones.json."""
    ok = (abs(resumen["top1_media_pliegues"] - top1) <= tol
          and abs(resumen["top1_de_pliegues"] - de) <= tol)
    if not ok:
        raise SystemExit(
            f"no se reproduce la cifra de N-30 desde {resumen['dir']}: "
            f"{resumen['top1_media_pliegues']:.4f} +- {resumen['top1_de_pliegues']:.4f} "
            f"frente a {top1} +- {de}. No se compara nada hasta aclararlo.")
    return {"esperado": f"{top1:.3f} ± {de:.3f}".replace(".", ","),
            "medido": (f"{resumen['top1_media_pliegues']:.4f} ± "
                       f"{resumen['top1_de_pliegues']:.4f}").replace(".", ","),
            "reproduce": True}


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------

def clave_control(resultados: dict) -> str | None:
    """Clave del modo de control de una corrida (el segundo modo, si lo hubo). Antes de la opción
    --aumentacion siempre era «sin_espejo»; hoy es «sin_aumentacion» cuando la principal es espejo."""
    principal = clave_principal(resultados)
    for c in resultados.get("modos") or list(resultados.get("loso", {})):
        if c != principal and c in resultados.get("loso", {}):
            return c
    return "sin_espejo" if "sin_espejo" in resultados.get("loso", {}) else None


def clave_principal(resultados: dict) -> str:
    """Clave del modo de aumentación principal de una corrida dentro de resultados["loso"].
    Los resultados anteriores a la opción --aumentacion no la traen: eran siempre «con_espejo»."""
    return resultados.get("clave_principal", "con_espejo")


def figura_comparacion(resultados: dict, destino: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    red = resultados["red"]["resumen"]
    clave = clave_principal(resultados)
    loso = resultados["loso"][clave]["agregado_997"]
    sistemas = ["Red (CNN+BiLSTM)"] + [NOMBRE_FAMILIA[f] for f in FAMILIAS]
    series = {
        "top-1": [red["top1_agregado"]] + [loso[f]["top1"] for f in FAMILIAS],
        "top-3": [red["top3_agregado"]] + [loso[f]["top3"] for f in FAMILIAS],
        "F1 macro": [resultados["red"]["f1_macro_agregado"]] + [loso[f]["f1_macro"] for f in FAMILIAS],
    }
    colores = {"top-1": "#2f6f9f", "top-3": "#8fb8d8", "F1 macro": "#c2543a"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    x = np.arange(len(sistemas)); ancho = 0.26
    for i, (nombre, valores) in enumerate(series.items()):
        barras = ax.bar(x + (i - 1) * ancho, valores, ancho, label=nombre, color=colores[nombre])
        ax.bar_label(barras, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(sistemas, fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("proporción")
    ax.set_title(f"LOSO agregado (997 muestras de prueba, "
                 f"{ETIQUETA_AUMENTACION[resultados.get('aumentacion_modo', 'espejo')]})",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1]
    if resultados.get("aleatoria"):
        ale = resultados["aleatoria"][clave]
        valores = [resultados["red"]["aleatoria_top1"]] + [ale["familias"][f]["prueba"]["top1"]
                                                           for f in FAMILIAS]
        etiquetas = ["Red"] + [NOMBRE_FAMILIA[f] for f in FAMILIAS]
        barras = ax.bar(np.arange(len(valores)), valores, 0.6, color="#2f6f9f")
        ax.bar_label(barras, fmt="%.3f", fontsize=8, padding=2)
        ax.set_xticks(np.arange(len(valores)))
        ax.set_xticklabels(etiquetas, fontsize=9, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title("Esquema aleatorio 797/100/100\n(top-1, señantes vistos)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "esquema aleatorio no ejecutado\n(--solo-loso)", ha="center", va="center")
    fig.suptitle("Actividades 2.c y 2.d (D-10): red temporal frente a la línea base clásica sobre "
                 "características agregadas", fontsize=12)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


def _n(x, decimales=3) -> str:
    """Numero con coma decimal, como el resto de la tesis (DECISIONES_TESIS.txt)."""
    if x is None:
        return "—"
    return f"{x:.{decimales}f}".replace(".", ",")


def _p(x) -> str:
    """Valor p con coma decimal (notacion cientifica incluida)."""
    return f"{x:.3g}".replace(".", ",")


def _fila(nombre, m):
    return (f"| {nombre} | {_n(m['top1'])} | {_n(m['top3'])} | {_n(m['precision_macro'])} | "
            f"{_n(m['recall_macro'])} | {_n(m['f1_macro'])} |")


def escribir_comparacion(resultados: dict, destino: Path) -> Path:
    red = resultados["red"]["resumen"]
    clave = clave_principal(resultados)
    etiqueta_aum = ETIQUETA_AUMENTACION[resultados.get("aumentacion_modo", "espejo")]
    agr = resultados["loso"][clave]["agregado_997"]
    ctrl = clave_control(resultados)
    sin = resultados["loso"].get(ctrl, {}).get("agregado_997", {}) if ctrl else {}
    mej = resultados["comparacion_mcnemar"]["mejor_familia"]
    L = []
    A = L.append
    A("# Línea base clásica frente a la red temporal (actividades 2.c y 2.d, D-10)")
    A("")
    A(f"Generado por `src/linea_base_clasica.py` el {resultados['fecha']}. "
      "Todo lo que sigue sale de esta corrida; el criterio estaba fijado en D-10 antes de correr.")
    A("")
    A("## 0. Lo que se comprobó antes de comparar")
    A("")
    A(f"- Tensor de entrada: {resultados['verificacion_tensor']['forma_por_muestra']} por muestra "
      f"(30 pasos x 252 rasgos: 126 coordenadas de las dos manos + 126 velocidades), idéntico valor "
      f"a valor al que produce el entrenador de la red para "
      f"{', '.join(resultados['verificacion_tensor']['muestras_comprobadas'])}.")
    A(f"- Cifra de la red (N-30) reproducida desde sus `predicciones.json`: "
      f"{resultados['red']['comprobacion']['medido']} top-1 LOSO "
      f"(esperado {resultados['red']['comprobacion']['esperado']}). "
      f"Agregado: {_n(red['top1_agregado'], 4)} "
      f"({int(round(red['top1_agregado']*red['n']))}/{red['n']}), top-3 {_n(red['top3_agregado'], 4)}.")
    A(f"- Características: {resultados['n_rasgos']} (8 estadísticos x 252 columnas), sin PCA ni "
      "selección; escalado estándar ajustado solo con el entrenamiento de cada pliegue.")
    A("")
    A(f"## 1. LOSO agregado (las 997 predicciones de prueba juntas), {etiqueta_aum}")
    A("")
    A("| sistema | top-1 | top-3 | precisión | recall | F1 macro |")
    A("|---|---|---|---|---|---|")
    A(_fila("Red temporal (CNN+BiLSTM+atención, N-30)", resultados["red"]["metricas_agregadas"]))
    for f in FAMILIAS:
        A(_fila(NOMBRE_FAMILIA[f], agr[f]))
    A("")
    A("## 2. LOSO por pliegue (top-1) y media entre pliegues")
    A("")
    A("| sistema | p0 (0000) | p1 (0001) | p2 (0002) | p3 (0003) | p4 (0004) | media ± DE |")
    A("|---|---|---|---|---|---|---|")
    v = red["top1_por_pliegue"]
    A("| Red temporal | " + " | ".join(_n(x) for x in v)
      + f" | {_n(red['top1_media_pliegues'])} ± {_n(red['top1_de_pliegues'])} |")
    for f in FAMILIAS:
        m = resultados["loso"][clave]["media_pliegues"][f]["top1"]
        A(f"| {NOMBRE_FAMILIA[f]} | " + " | ".join(_n(x) for x in m["por_pliegue"])
          + f" | {_n(m['media'])} ± {_n(m['de'])} |")
    A("")
    A("## 3. Control sin aumentación de espejo (LOSO agregado)")
    A("")
    if sin:
        A("| familia | top-1 con espejo | top-1 sin espejo | diferencia |")
        A("|---|---|---|---|")
        for f in FAMILIAS:
            d = agr[f]["top1"] - sin[f]["top1"]
            A(f"| {NOMBRE_FAMILIA[f]} | {_n(agr[f]['top1'])} | {_n(sin[f]['top1'])} | "
              f"{('+' if d >= 0 else '')}{_n(d)} |")
    else:
        A("No ejecutado en esta corrida.")
    A("")
    A("## 4. Configuración elegida por pliegue (top-1 de validación; la prueba no se miró)")
    A("")
    A("| familia | p0 | p1 | p2 | p3 | p4 | aleatoria |")
    A("|---|---|---|---|---|---|---|")
    for f in FAMILIAS:
        elegidas = [p["familias"][f]["configuracion_elegida"]
                    for p in resultados["loso"][clave]["pliegues"]]
        ale = (resultados["aleatoria"][clave]["familias"][f]["configuracion_elegida"]
               if resultados.get("aleatoria") else "—")
        A(f"| {NOMBRE_FAMILIA[f]} | " + " | ".join(elegidas) + f" | {ale} |")
    A("")
    if resultados.get("aleatoria"):
        A(f"## 5. Esquema aleatorio 797/100/100 (señantes vistos), {etiqueta_aum}")
        A("")
        A("| sistema | top-1 | top-3 | precisión | recall | F1 macro |")
        A("|---|---|---|---|---|---|")
        A(f"| Red temporal (N-30) | {_n(resultados['red']['aleatoria_top1'])} | "
          f"{_n(resultados['red']['aleatoria_top3'])} | "
          f"{_n(resultados['red']['aleatoria_precision'])} | "
          f"{_n(resultados['red']['aleatoria_recall'])} | "
          f"{_n(resultados['red']['aleatoria_f1'])} |")
        for f in FAMILIAS:
            A(_fila(NOMBRE_FAMILIA[f],
                    resultados["aleatoria"][clave]["familias"][f]["prueba"]))
        A("")
    A("## 6. McNemar frente a la red (997 predicciones LOSO agregadas, α = 0,05)")
    A("")
    A("| línea base | aciertos red | aciertos línea base | solo red | solo línea base | prueba | p |")
    A("|---|---|---|---|---|---|---|")
    for f in FAMILIAS:
        m = resultados["comparacion_mcnemar"]["por_familia"][f]
        A(f"| {NOMBRE_FAMILIA[f]} | {m['aciertos_a']} | {m['aciertos_b']} | "
          f"{m['tabla']['solo_a']} | {m['tabla']['solo_b']} | {m['prueba_usada']} | "
          f"{_p(m['p'])} |")
    A("")
    A("## 7. Conclusión")
    A("")
    for linea in resultados["conclusion"]:
        A(f"- {linea}")
    A("")
    A("## 8. Límites de esta comparación")
    A("")
    A("- Límite declarado en D-10: la línea base recibe la secuencia resumida en estadísticos y "
      "pierde el orden temporal salvo lo que capturan el primer paso, el último y la media de las "
      "diferencias entre pasos consecutivos.")
    if resultados.get("aumentacion_modo", "espejo") == "completa":
        A("- Esta corrida SÍ entrena la línea base con las nueve transformaciones de la red "
          "(factor 10), que es lo que la hace comparable en aumentación; D-10 solo pedía el "
          "espejo, así que la corrida de referencia de D-10 sigue siendo la de espejo.")
    else:
        A("- La línea base no se entrena sobre las nueve transformaciones de la red (factor 10), solo "
          "sobre el espejo, que es lo que dice D-10; el control sin espejo acota cuánto aporta.")
    A("- 997 muestras y 5 señantes: la dispersión entre pliegues (señantes) es grande en los dos "
      "sistemas y domina cualquier diferencia de décimas.")
    A("- McNemar compara aciertos pareados sobre las mismas muestras; no dice nada sobre latencia, "
      "coste de entrenamiento ni comportamiento en el prototipo en vivo.")
    A("")
    destino.write_text("\n".join(L) + "\n", encoding="utf-8")
    return destino


# ---------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------

def _conjunto(nombres_conj, pos, X_seq, y):
    idx = np.array([pos[n] for n in nombres_conj], dtype=np.int64)
    return X_seq[idx], y[idx], list(nombres_conj)


def info_entorno() -> dict:
    import sklearn
    import scipy
    return {"python": platform.python_version(), "plataforma": platform.platform(),
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "hilos": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                     "MKL_NUM_THREADS")}}


def metricas_agregadas_red(filas_red) -> dict:
    probs = np.zeros((len(filas_red), N_CLASES))
    y = np.array([f["real"] for f in filas_red])
    for i, f in enumerate(filas_red):           # reconstruye un orden top-3 coherente
        for r, c in enumerate(f["top3"]):
            probs[i, c] = 3 - r
    return metricas(y, probs)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Linea base clasica sobre caracteristicas agregadas (actividades 2.c y 2.d, D-10).")
    parser.add_argument("--processed", type=str, default=str(PROCESSED_POR_DEFECTO),
                        help="directorio processed (defecto data/processed_holistic_mascara_rostro)")
    parser.add_argument("--salida", type=str, default=str(SALIDA_POR_DEFECTO),
                        help="directorio de salida (defecto models/linea_base_4d_2026-09-15)")
    parser.add_argument("--solo-loso", action="store_true",
                        help="omite el esquema aleatorio 797/100/100")
    parser.add_argument("--red", type=str, default=str(RED_POR_DEFECTO),
                        help="LOSO de la red para McNemar (defecto "
                             "models/loso_2026-09-15_holistic_mascara_manos)")
    parser.add_argument("--n-jobs", type=int, default=2,
                        help="hilos de sklearn (defecto 2; el equipo esta grabando sesiones)")
    parser.add_argument("--sin-control", action="store_true",
                        help="omite el control sin aumentacion de espejo (solo para pruebas)")
    parser.add_argument("--aumentacion", choices=AUMENTACIONES, default="espejo",
                        help="aumentacion del entrenamiento de la linea base: espejo (D-10, por "
                             "defecto), completa (las nueve transformaciones de la red, factor 10) "
                             "o ninguna")
    args = parser.parse_args(argv)

    salida = Path(args.salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)
    t_inicio = time.perf_counter()
    print("=" * 78)
    print("  LSC50 - Linea base clasica sobre caracteristicas agregadas (D-10, actividades 2.c y 2.d)")
    print("=" * 78)

    # [1] La cifra de la red, ANTES de nada (D-10, punto 6).
    filas_red, resumen_red = cargar_predicciones_red(args.red)
    comprobacion = comprobar_referencia_red(resumen_red)
    print(f"\n[1] Red de referencia ({resumen_red['dir']}): LOSO top-1 "
          f"{comprobacion['medido']} (N-30 declara {comprobacion['esperado']}) | agregado "
          f"{resumen_red['top1_agregado']:.4f} sobre {resumen_red['n']} muestras")

    # [2] Datos: la misma secuencia que recibe la red.
    processed = Path(args.processed).expanduser().resolve()
    X_seq, y, nombres, senantes = cargar_secuencias(processed)
    verificacion = verificar_tensor(X_seq, nombres, processed)
    print(f"\n[2] Datos ({_rel(processed)}, representacion {REPRESENTACION}): {len(nombres)} "
          f"muestras, secuencias {X_seq.shape} -> tensor por muestra "
          f"{tuple(verificacion['forma_por_muestra'])}")
    print(f"    Verificado contra el entrenador de la red en: "
          f"{', '.join(verificacion['muestras_comprobadas'])}")
    pos = {n: i for i, n in enumerate(nombres)}

    loso = json.loads(PARTICION_LOSO.read_text(encoding="utf-8"))
    aleatoria = json.loads(PARTICION_ALEATORIA.read_text(encoding="utf-8"))
    print(f"    Particiones D-06: {_rel(PARTICION_LOSO)} ({len(loso['pliegues'])} pliegues) y "
          f"{_rel(PARTICION_ALEATORIA)}")

    CLAVE = CLAVE_PRINCIPAL[args.aumentacion]
    modos = [(CLAVE, args.aumentacion)]
    if not args.sin_control:
        control = "ninguna" if args.aumentacion == "espejo" else "espejo"
        modos.append((CLAVE_PRINCIPAL[control], control))
    print(f"\n    Aumentacion del entrenamiento: {DESCRIPCION_AUMENTACION[args.aumentacion]}")
    resultados = {
        "decision": "D-10 (actividades 2.c y 2.d): linea base clasica sobre caracteristicas agregadas",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "processed_dir": _rel(processed),
        "representacion": REPRESENTACION,
        "seq_length": SEQ_LENGTH,
        "n_rasgos": 8 * 252,
        "estadisticos": list(ESTADISTICOS),
        "rejilla": {f: [n for n, _ in REJILLA[f]] for f in FAMILIAS},
        "semilla": SEMILLA,
        "aumentacion": DESCRIPCION_AUMENTACION[args.aumentacion],
        "aumentacion_modo": args.aumentacion,
        "clave_principal": CLAVE,
        "modos": [c for c, _ in modos],
        "particiones": {"loso": _rel(PARTICION_LOSO), "aleatoria": _rel(PARTICION_ALEATORIA)},
        "verificacion_tensor": verificacion,
        "entorno": info_entorno(),
        "loso": {}, "aleatoria": None,
    }

    # [3] LOSO
    print(f"\n[3] LOSO, {len(loso['pliegues'])} pliegues x {len(modos)} modo(s) x "
          f"{sum(len(REJILLA[f]) for f in FAMILIAS)} configuraciones")
    for clave, modo_aum in modos:
        pliegues = []
        for p in loso["pliegues"]:
            c = p["conjuntos"]
            X_tr, y_tr, _ = _conjunto(c["train"], pos, X_seq, y)
            X_val, y_val, _ = _conjunto(c["val"], pos, X_seq, y)
            obtener = (lambda c=c: _conjunto(c["test"], pos, X_seq, y))
            r = ejecutar_corrida(f"pliegue_{p['pliegue']}", X_tr, y_tr, X_val, y_val, obtener,
                                 aumentacion=modo_aum, n_jobs=args.n_jobs, verboso=True)
            r["pliegue"] = p["pliegue"]; r["senante_test"] = p["senante_test"]
            pliegues.append(r)
        agregado, medias, predicciones = {}, {}, {}
        for f in FAMILIAS:
            todas = [fila for p in pliegues for fila in p["familias"][f]["predicciones"]]
            predicciones[f] = todas
            probs = np.zeros((len(todas), N_CLASES))
            for i, fila in enumerate(todas):
                for r_, cls in enumerate(fila["top3"]):
                    probs[i, cls] = 3 - r_
                probs[i, fila["predicha"]] = 4        # el argmax es siempre la clase predicha
            agregado[f] = metricas(np.array([x["real"] for x in todas]), probs)
            medias[f] = agregar_pliegues([p["familias"][f]["prueba"] for p in pliegues])
        resultados["loso"][clave] = {"pliegues": pliegues, "agregado_997": agregado,
                                     "media_pliegues": medias}
        resultados["loso"][clave]["_predicciones"] = predicciones

    # [4] Esquema aleatorio
    if not args.solo_loso:
        print("\n[4] Esquema aleatorio 797/100/100")
        resultados["aleatoria"] = {}
        for clave, modo_aum in modos:
            c = aleatoria["conjuntos"]
            X_tr, y_tr, _ = _conjunto(c["train"], pos, X_seq, y)
            X_val, y_val, _ = _conjunto(c["val"], pos, X_seq, y)
            r = ejecutar_corrida("aleatoria", X_tr, y_tr, X_val, y_val,
                                 lambda c=c: _conjunto(c["test"], pos, X_seq, y),
                                 aumentacion=modo_aum, n_jobs=args.n_jobs, verboso=True)
            resultados["aleatoria"][clave] = r
    else:
        print("\n[4] Esquema aleatorio omitido (--solo-loso)")

    # [5] McNemar contra la red (D-10, punto 6)
    print("\n[5] McNemar frente a la red (997 predicciones LOSO agregadas, alfa = 0,05)")
    orden_red = {f["muestra"]: f for f in filas_red}
    por_familia = {}
    for f in FAMILIAS:
        base = resultados["loso"][CLAVE]["_predicciones"][f]
        if sorted(x["muestra"] for x in base) != sorted(orden_red):
            raise SystemExit("las muestras de prueba de la linea base y de la red no coinciden")
        a = np.array([orden_red[x["muestra"]]["real"] == orden_red[x["muestra"]]["predicha"]
                      for x in base])
        b = np.array([x["real"] == x["predicha"] for x in base])
        por_familia[f] = mcnemar(a, b)
        m = por_familia[f]
        print(f"    red {m['aciertos_a']} aciertos vs {NOMBRE_FAMILIA[f]} {m['aciertos_b']} | "
              f"solo red {m['tabla']['solo_a']} | solo base {m['tabla']['solo_b']} | "
              f"{m['prueba_usada']} p = {m['p']:.3g} | "
              f"{'significativo' if m['significativo'] else 'no significativo'}")
    mejor = max(FAMILIAS, key=lambda f: resultados["loso"][CLAVE]["agregado_997"][f]["top1"])
    resultados["comparacion_mcnemar"] = {"por_familia": por_familia, "mejor_familia": mejor,
                                         "criterio_mejor": f"mayor top-1 LOSO agregado ({ETIQUETA_AUMENTACION[args.aumentacion]})"}

    # Conclusión (D-10, punto 6): se escribe la que toque, gane quien gane.
    red_top1 = resumen_red["top1_agregado"]
    base_top1 = resultados["loso"][CLAVE]["agregado_997"][mejor]["top1"]
    m = por_familia[mejor]
    if red_top1 > base_top1 and m["significativo"]:
        veredicto = (f"La arquitectura temporal QUEDA JUSTIFICADA empíricamente según el criterio de "
                     f"D-10: top-1 agregado de la red {_n(red_top1)} > {_n(base_top1)} de la mejor "
                     f"línea base ({NOMBRE_FAMILIA[mejor]}) y McNemar p = {_p(m['p'])} < {_n(ALFA, 2)}.")
    elif red_top1 > base_top1:
        veredicto = (f"La red gana en top-1 agregado ({_n(red_top1)} frente a {_n(base_top1)}, "
                     f"{NOMBRE_FAMILIA[mejor]}) pero McNemar da p = {_p(m['p'])} >= {_n(ALFA, 2)}: la "
                     f"diferencia NO es significativa y el criterio de D-10 NO se cumple. La "
                     f"arquitectura temporal no queda justificada por esta comparación.")
    elif base_top1 > red_top1:
        veredicto = (f"La LÍNEA BASE CLÁSICA SUPERA a la red: {NOMBRE_FAMILIA[mejor]} obtiene "
                     f"{_n(base_top1)} de top-1 agregado frente a {_n(red_top1)} de la red "
                     f"(McNemar p = {_p(m['p'])}, {'significativo' if m['significativo'] else 'no significativo'}). "
                     f"D-10 lo previó: se declara así y se discute; no se reentrena la red ni se "
                     f"cambia el prototipo para ganar la comparación.")
    else:
        veredicto = (f"Empate en top-1 agregado ({_n(red_top1)}); McNemar p = {_p(m['p'])}. El "
                     f"criterio de D-10 no se cumple: la arquitectura temporal no queda justificada "
                     f"por esta comparación.")
    por_pliegue_base = resultados["loso"][CLAVE]["media_pliegues"][mejor]["top1"]
    gana_red = sum(1 for r_, b_ in zip(resumen_red["top1_por_pliegue"],
                                       por_pliegue_base["por_pliegue"]) if r_ > b_)
    peor_red = int(np.argmin(resumen_red["top1_por_pliegue"]))
    peores = [f"red 0{peor_red} ({_n(min(resumen_red['top1_por_pliegue']))})"]
    for f in sorted(FAMILIAS, key=lambda f: -resultados["loso"][CLAVE]["agregado_997"][f]["top1"]):
        por_p = resultados["loso"][CLAVE]["media_pliegues"][f]["top1"]["por_pliegue"]
        i = int(np.argmin(por_p))
        peores.append(f"{NOMBRE_FAMILIA[f]} 0{i} ({_n(min(por_p))})")
    peor_pliegue = "; ".join(peores)
    orden_familias = ", ".join(
        f"{NOMBRE_FAMILIA[f]} {_n(resultados['loso'][CLAVE]['agregado_997'][f]['top1'])}"
        for f in sorted(FAMILIAS,
                        key=lambda f: -resultados["loso"][CLAVE]["agregado_997"][f]["top1"]))
    ctrl_clave = clave_control(resultados)
    if args.aumentacion == "espejo" and ctrl_clave:
        agr_con = resultados["loso"][CLAVE]["agregado_997"]
        agr_sin = resultados["loso"][ctrl_clave]["agregado_997"]
        delta_espejo = _n(max(agr_con[f]["top1"] - agr_sin[f]["top1"] for f in FAMILIAS))
    else:
        delta_espejo = "—"
    conclusion = [veredicto,
                  f"Mejor línea base por top-1 LOSO agregado: {NOMBRE_FAMILIA[mejor]} "
                  f"({_n(base_top1)}); red {_n(red_top1)}.",
                  f"Concordancia: {m['tabla']['aciertan_ambos']} muestras las aciertan los dos, "
                  f"{m['tabla']['fallan_ambos']} las fallan los dos, {m['tabla']['solo_a']} solo la "
                  f"red y {m['tabla']['solo_b']} solo la línea base.",
                  f"La red supera a {NOMBRE_FAMILIA[mejor]} en {gana_red} de los 5 pliegues LOSO, "
                  f"así que la ventaja no viene de un solo señante.",
                  f"Orden de las tres familias por top-1 LOSO agregado: {orden_familias}. "
                  f"Peor pliegue de cada sistema: {peor_pliegue}.",
                  (f"La aumentación de espejo aporta como mucho {delta_espejo} de top-1 agregado: "
                   f"la distancia con la red no se explica por la aumentación."
                   if delta_espejo != "—" else
                   "Esta corrida no incluye el control de aumentación, así que no mide cuánto aporta "
                   "la aumentación; esa cifra está en la corrida principal."),
                  "Lectura prudente: esto NO dice que un clasificador clásico no sirva para LSC; "
                  "dice que, con esta entrada (ocho estadísticos por columna, que borran el orden "
                  "temporal) y este corpus, resumir la secuencia cuesta caro."
                  + (" Sigue en pie la brecha entre señantes vistos y no vistos: en el esquema "
                     "aleatorio la línea base también sube mucho, así que el problema de fondo "
                     "(generalizar a un señante nuevo con 5 señantes) lo tienen los dos sistemas."
                     if resultados.get("aleatoria") else
                     " Esta corrida no ejecutó el esquema aleatorio, así que no dice nada sobre la "
                     "brecha entre señantes vistos y nuevos.")]
    resultados["conclusion"] = conclusion
    print("\n" + "\n".join("    " + c for c in conclusion))

    # [6] Salidas
    resultados["red"] = {
        "resumen": resumen_red, "comprobacion": comprobacion,
        "metricas_agregadas": metricas_agregadas_red(filas_red),
    }
    resultados["red"]["f1_macro_agregado"] = resultados["red"]["metricas_agregadas"]["f1_macro"]
    met_red = json.loads((Path(args.red) / "resultados_loso.json").read_text(encoding="utf-8"))
    ale_red = met_red.get("aleatoria_80_10_10") or {}
    resultados["red"].update({
        "aleatoria_top1": ale_red.get("top1"), "aleatoria_top3": ale_red.get("top3"),
        "aleatoria_precision": ale_red.get("precision_macro"),
        "aleatoria_recall": ale_red.get("recall_macro"),
        "aleatoria_f1": ale_red.get("f1_macro"),
    })

    print("\n[6] Salidas")
    for clave, _ in modos:
        sufijo = "" if clave == CLAVE else f"_{clave}"
        for f in FAMILIAS:
            ruta = salida / f"predicciones_{f}{sufijo}.json"
            ruta.write_text(json.dumps(resultados["loso"][clave]["_predicciones"][f], indent=1,
                                       ensure_ascii=False), encoding="utf-8")
            print(f"    {_rel(ruta)} ({len(resultados['loso'][clave]['_predicciones'][f])} filas)")
        if resultados.get("aleatoria"):
            for f in FAMILIAS:
                ruta = salida / f"predicciones_{f}_aleatoria{sufijo}.json"
                ruta.write_text(json.dumps(
                    resultados["aleatoria"][clave]["familias"][f]["predicciones"], indent=1,
                    ensure_ascii=False), encoding="utf-8")
    # Las predicciones ya están en sus archivos; fuera del JSON grande.
    for clave, _ in modos:
        resultados["loso"][clave].pop("_predicciones", None)
        for p in resultados["loso"][clave]["pliegues"]:
            for f in FAMILIAS:
                p["familias"][f].pop("predicciones", None)
        if resultados.get("aleatoria"):
            for f in FAMILIAS:
                resultados["aleatoria"][clave]["familias"][f].pop("predicciones", None)
    resultados["segundos_totales"] = round(time.perf_counter() - t_inicio, 1)
    (salida / "resultados.json").write_text(
        json.dumps(resultados, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"    {_rel(salida / 'resultados.json')}")
    print(f"    {_rel(escribir_comparacion(resultados, salida / 'comparacion.md'))}")
    print(f"    {_rel(figura_comparacion(resultados, salida / 'comparacion_red_vs_linea_base.png'))}")
    print(f"\nTiempo total: {resultados['segundos_totales']:.0f} s")
    return resultados


if __name__ == "__main__":
    main()
