"""
CNN 2D pequeña para el experimento de cierre docs/EXPERIMENTO_CNN2D.md (brazo C2D). Es la «red convolucional
compleja» que pidió el director (D-14, pedido adicional): un modelo más, con una sola configuración fijada
antes de entrenar y sin búsqueda de hiperparámetros.

Idea («esqueleto como imagen»): la ventana de 30 × 252 se ve como una imagen cuyo eje vertical es el tiempo,
cuyo eje horizontal son los 21 puntos de la mano (orden de MediaPipe) y cuyos canales son las coordenadas
(Du, Fu y Wang 2015, ACPR, doi:10.1109/ACPR.2015.7486569; Li, Zhong, Xie y Pu 2018, IJCAI,
doi:10.24963/ijcai.2018/109, que usan un tensor cuadros × articulaciones × 3 con las coordenadas como canal).

Remodelación (dentro del modelo, así que se aplica DESPUÉS de la aumentación y de add_velocity, y la entrada
del modelo sigue siendo 30 × 252 como en translator.py):

    columnas de «manos» con velocidades (src/representacion.py + add_velocity):
        [  0:126]  posiciones: mano_izq (21 puntos × x, y, z intercalados) y luego mano_der
        [126:252]  velocidades, en el mismo orden
    (30, 252) -> (30, 2 [pos/vel], 2 [mano], 21 [punto], 3 [coord])
              -> (30, 21 [punto], 2 [mano], 2 [pos/vel], 3 [coord])
              -> (30, 21, 12)
    canal c = 6·mano + 3·(pos/vel) + coord: 0-5 = (x, y, z, vx, vy, vz) de la mano izquierda DE LA PERSONA,
    6-11 = los mismos de la derecha.

Así las dos manos no quedan una al lado de la otra en el eje espacial: son canales del mismo punto. Los vecinos
del eje de puntos son puntos consecutivos de MediaPipe en la misma mano (0 muñeca, 1-4 pulgar, 5-8 índice,
9-12 medio, 13-16 anular, 17-20 meñique); solo los pasos 4-5, 8-9, 12-13 y 16-17 saltan de la punta de un dedo
a la base del siguiente.

Este módulo no entrena nada; lo importa src/experimento_cnn2d.py. Solo usa capas estándar de Keras, así que los
modelos guardados se cargan sin registrar nada.
"""

from __future__ import annotations

import numpy as np

N_PASOS = 30
N_MANO = 21
COORDS = 3
N_CANALES = 2 * 2 * COORDS   # 2 manos × (posición, velocidad) × 3 coordenadas

# Configuración fijada ANTES de entrenar (una sola; sin variantes).
CONFIG_C2D = {
    "imagen": "30 (tiempo) x 21 (puntos MediaPipe) x 12 canales (x,y,z,vx,vy,vz izq + der)",
    "filtros": [32, 64, 64],
    "convs_por_bloque": 2,
    "kernel": [3, 3],
    "padding": "same",
    "pool_por_bloque": [[2, 1], [2, 1], None],     # 2×1 solo en el tiempo, solo en los dos primeros bloques
    "dropout_espacial": 0.15,                       # SpatialDropout2D al final de cada bloque
    "dropout_final": 0.3,
    "pooling_global": "GlobalAveragePooling2D",
}
# Receta de entrenamiento: la de la TCN (Adam lr 1e-3 con CosineDecay, label smoothing 0,1, lote 64).
ENTRENAMIENTO_C2D = {"lr": 1e-3, "batch": 64, "label_smoothing": 0.1}
N_PARAM_C2D = 146_546


def indice_columna(mano: int, es_velocidad: int, punto: int, coord: int) -> int:
    """Columna de la entrada de 252 que corresponde a (mano, pos/vel, punto, coordenada)."""
    return 126 * es_velocidad + 63 * mano + 3 * punto + coord


def canal(mano: int, es_velocidad: int, coord: int) -> int:
    """Canal de la imagen de 12 que corresponde a (mano, pos/vel, coordenada)."""
    return 6 * mano + 3 * es_velocidad + coord


def remodelar_a_imagen_np(X: np.ndarray) -> np.ndarray:
    """Versión numpy de la remodelación del modelo: (N, 30, 252) -> (N, 30, 21, 12)."""
    N, T, F = X.shape
    assert F == 2 * 2 * N_MANO * COORDS, F
    return (X.reshape(N, T, 2, 2, N_MANO, COORDS)        # (pos/vel, mano, punto, coord)
            .transpose(0, 1, 4, 3, 2, 5)                 # (punto, mano, pos/vel, coord)
            .reshape(N, T, N_MANO, N_CANALES))


def _tf():
    import tensorflow as tf
    return tf


def construir_cnn2d(n_frames=N_PASOS, n_features=252, n_classes=50, cfg=CONFIG_C2D):
    """Brazo C2D. (30, 252) -> imagen (30, 21, 12) -> 3 bloques [Conv2D 3×3 -> BN -> ReLU] × 2 ->
    SpatialDropout2D -> MaxPooling 2×1 (solo bloques 1 y 2) -> GAP 2D -> Dropout 0,3 -> softmax.
    Eje de puntos sin reducir (21 en todo el camino); tiempo 30 -> 15 -> 7. Campo receptivo: 13 puntos en el eje
    de puntos y unos 30 pasos en el tiempo."""
    tf = _tf()
    L = tf.keras.layers
    assert n_features == 2 * 2 * N_MANO * COORDS, n_features
    inp = tf.keras.Input(shape=(n_frames, n_features))
    x = L.Reshape((n_frames, 2, 2, N_MANO, COORDS))(inp)       # (pos/vel, mano, punto, coord)
    x = L.Permute((1, 4, 3, 2, 5))(x)                           # (punto, mano, pos/vel, coord)
    x = L.Reshape((n_frames, N_MANO, N_CANALES))(x)
    for filtros, pool in zip(cfg["filtros"], cfg["pool_por_bloque"]):
        for _ in range(cfg["convs_por_bloque"]):
            x = L.Conv2D(filtros, tuple(cfg["kernel"]), padding=cfg["padding"])(x)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
        x = L.SpatialDropout2D(cfg["dropout_espacial"])(x)
        if pool is not None:
            x = L.MaxPooling2D(pool_size=tuple(pool))(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(cfg["dropout_final"])(x)
    x = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, x, name="cnn2d_esqueleto_imagen")


def submodelo_imagen(model):
    """Modelo que devuelve la imagen (30, 21, 12) que ve la primera Conv2D (para las pruebas)."""
    tf = _tf()
    capa = [l for l in model.layers if isinstance(l, tf.keras.layers.Reshape)][-1]
    return tf.keras.Model(model.input, capa.output)
