"""
Arquitecturas de la revisión bibliográfica (docs/REVISION_LITERATURA_ARQUITECTURAS.md, §4-§5) para el
experimento docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md. Todas reciben la entrada canónica de 30 × 252
(«manos»: 126 posiciones + 126 velocidades) y devuelven softmax de 50.

  · construir_tcn      brazo T: CNN 1D / TCN pequeña con bloques residuales dilatados (1-2-4-8) y GAP
                       (Bai et al. 2018 [12]; Ismail Fawaz et al. 2019 [6]; DD-Net [5]).
  · construir_stgcn    brazo G: ST-GCN pequeño sobre el grafo de las dos manos (42 nodos, aristas de la
                       mano de MediaPipe + autoconexiones, las dos manos como componentes separadas;
                       Yan et al. 2018; OpenHands [3]).

La configuración de cada brazo se fija ANTES de ver resultados (CONFIG_T, CONFIG_G) y no se cambia.

Orden de columnas de la representación «manos» (src/representacion.py): mano_izq (21 puntos × x, y, z,
intercalados) y luego mano_der; add_velocity agrega después las 126 velocidades en el mismo orden. Por
eso remodelar_a_nodos lleva (30, 252) a (30, 2, 42, 3) -> (30, 42, 2, 3) -> (30, 42, 6), con los canales
[x, y, z, vx, vy, vz] y el nodo v = 21·mano + punto (mano 0 = izquierda de la persona).

Este módulo no entrena nada; lo importa src/experimento_arquitecturas.py y hay que importarlo antes de
cargar un modelo G guardado (registra la capa ConvGrafo).
"""

from __future__ import annotations

import numpy as np

N_PASOS = 30
N_MANO = 21
N_NODOS = 2 * N_MANO
COORDS = 3

# Aristas de la mano de MediaPipe (mediapipe.tasks.python.vision.HandLandmarksConnections.HAND_CONNECTIONS,
# leídas del mediapipe instalado en el entorno del proyecto el 2-oct-2026; 21 aristas).
ARISTAS_MANO = [(0, 1), (0, 17), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (5, 9), (6, 7), (7, 8), (9, 10),
                (9, 13), (10, 11), (11, 12), (13, 14), (13, 17), (14, 15), (15, 16), (17, 18), (18, 19),
                (19, 20)]

# Configuraciones fijadas antes de ver resultados (una por familia; no hay variantes).
CONFIG_T = {"filtros": 64, "ventana": 3, "dilataciones": [1, 2, 4, 8], "convs_por_bloque": 2,
            "dropout_bloque": 0.2, "dropout_final": 0.3, "padding": "same"}
CONFIG_G = {"canales": [32, 64, 64], "pasos_temporales": [1, 2, 1], "ventana_temporal": 9,
            "dropout_bloque": 0.2, "dropout_final": 0.3}
# Entrenamiento común de T y G: el del canónico (train_loso.entrenar_y_evaluar).
ENTRENAMIENTO_COMUN = {"lr": 1e-3, "batch": 64, "label_smoothing": 0.1}


# ---------------------------------------------------------------------------
# Grafo de las dos manos
# ---------------------------------------------------------------------------

def adyacencia_dos_manos() -> np.ndarray:
    """A (42 × 42) con autoconexiones: aristas de MediaPipe en cada mano, sin aristas entre manos."""
    A = np.eye(N_NODOS, dtype=np.float32)
    for mano in range(2):
        o = mano * N_MANO
        for i, j in ARISTAS_MANO:
            A[o + i, o + j] = 1.0
            A[o + j, o + i] = 1.0
    return A


def adyacencia_normalizada() -> np.ndarray:
    """Â = D^-1/2 (A + I) D^-1/2 (normalización simétrica de Kipf y Welling, usada en ST-GCN)."""
    A = adyacencia_dos_manos()
    d = A.sum(axis=1)
    Dm = np.diag(1.0 / np.sqrt(d)).astype(np.float32)
    return (Dm @ A @ Dm).astype(np.float32)


def remodelar_a_nodos_np(X: np.ndarray) -> np.ndarray:
    """Versión numpy de la remodelación del modelo G: (N, 30, 252) -> (N, 30, 42, 6)."""
    N, T, F = X.shape
    assert F == 2 * N_NODOS * COORDS, F
    return X.reshape(N, T, 2, N_NODOS, COORDS).transpose(0, 1, 3, 2, 4).reshape(N, T, N_NODOS, 2 * COORDS)


# ---------------------------------------------------------------------------
# Modelos (TensorFlow se importa dentro para no cargarlo al importar el módulo)
# ---------------------------------------------------------------------------

def _tf():
    import tensorflow as tf
    return tf


_REGISTRADA = {}


def capa_conv_grafo():
    """Clase ConvGrafo registrada para serialización (.keras)."""
    if "ConvGrafo" in _REGISTRADA:
        return _REGISTRADA["ConvGrafo"]
    tf = _tf()

    @tf.keras.utils.register_keras_serializable(package="arquitecturas_literatura")
    class ConvGrafo(tf.keras.layers.Layer):
        """Convolución de grafo espacial: Y[b,t,v,:] = Σ_w Â[v,w] · (X[b,t,w,:] W) + b, con Â fija."""

        def __init__(self, unidades, **kwargs):
            super().__init__(**kwargs)
            self.unidades = int(unidades)
            self.A = tf.constant(adyacencia_normalizada())

        def build(self, input_shape):
            self.W = self.add_weight(name="W", shape=(int(input_shape[-1]), self.unidades),
                                     initializer="glorot_uniform", trainable=True)
            self.b = self.add_weight(name="b", shape=(self.unidades,), initializer="zeros", trainable=True)

        def call(self, x):
            x = tf.einsum("btwc,cd->btwd", x, self.W)
            x = tf.einsum("vw,btwd->btvd", self.A, x)
            return x + self.b

        def get_config(self):
            c = super().get_config()
            c.update({"unidades": self.unidades})
            return c

    _REGISTRADA["ConvGrafo"] = ConvGrafo
    return ConvGrafo


def construir_tcn(n_frames=N_PASOS, n_features=252, n_classes=50, cfg=CONFIG_T):
    """Brazo T. Bloque residual de Bai et al. (2 convoluciones dilatadas por bloque, BN, ReLU, dropout,
    atajo 1×1 si cambian los canales), padding «same» (no causal: se clasifica la ventana completa),
    GlobalAveragePooling1D y softmax. Campo receptivo con k = 3 y d = 1-2-4-8, 2 convs por bloque: 61 pasos
    (cubre los 30)."""
    tf = _tf()
    L = tf.keras.layers
    inp = tf.keras.Input(shape=(n_frames, n_features))
    x = inp
    for d in cfg["dilataciones"]:
        atajo = x
        for j in range(cfg["convs_por_bloque"]):
            x = L.Conv1D(cfg["filtros"], cfg["ventana"], dilation_rate=d, padding=cfg["padding"])(x)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
            x = L.Dropout(cfg["dropout_bloque"])(x)
        if atajo.shape[-1] != cfg["filtros"]:
            atajo = L.Conv1D(cfg["filtros"], 1, padding="same")(atajo)
        x = L.Add()([x, atajo])
        x = L.ReLU()(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(cfg["dropout_final"])(x)
    x = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, x, name="tcn_literatura")


def construir_stgcn(n_frames=N_PASOS, n_features=252, n_classes=50, cfg=CONFIG_G):
    """Brazo G. (30, 252) -> (30, 42, 6) -> BN de datos por nodo y canal -> 3 bloques ST-GCN
    [ConvGrafo (Â fija) -> BN -> ReLU -> Conv2D temporal (9 × 1, paso s) -> BN -> Dropout -> + atajo ->
    ReLU] -> GAP sobre tiempo y nodos -> Dropout -> softmax."""
    tf = _tf()
    L = tf.keras.layers
    ConvGrafo = capa_conv_grafo()
    assert n_features == 2 * N_NODOS * COORDS
    inp = tf.keras.Input(shape=(n_frames, n_features))
    x = L.Reshape((n_frames, 2, N_NODOS, COORDS))(inp)
    x = L.Permute((1, 3, 2, 4))(x)
    x = L.Reshape((n_frames, N_NODOS * 2 * COORDS))(x)
    x = L.BatchNormalization()(x)                      # data_bn de ST-GCN: por nodo y canal
    x = L.Reshape((n_frames, N_NODOS, 2 * COORDS))(x)
    for c, s in zip(cfg["canales"], cfg["pasos_temporales"]):
        atajo = x
        y = ConvGrafo(c)(x)
        y = L.BatchNormalization()(y)
        y = L.ReLU()(y)
        y = L.Conv2D(c, (cfg["ventana_temporal"], 1), strides=(s, 1), padding="same")(y)
        y = L.BatchNormalization()(y)
        y = L.Dropout(cfg["dropout_bloque"])(y)
        if atajo.shape[-1] != c or s != 1:
            atajo = L.Conv2D(c, (1, 1), strides=(s, 1), padding="same")(atajo)
            atajo = L.BatchNormalization()(atajo)
        x = L.ReLU()(L.Add()([y, atajo]))
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(cfg["dropout_final"])(x)
    x = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, x, name="stgcn_literatura")
