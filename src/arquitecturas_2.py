"""
Segunda tanda de arquitecturas de la revisión bibliográfica (docs/REVISION_LITERATURA_ARQUITECTURAS.md) para
el experimento docs/EXPERIMENTO_ARQUITECTURAS_2.md: las tres candidatas que quedaron sin probar en
docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md. Todas reciben la entrada canónica de 30 × 252 («manos»: 126
posiciones + 126 velocidades) y devuelven softmax de 50.

  · construir_transformer  brazo X: encoder Transformer pequeño, pre-norm, 2 bloques, d = 64, 4 cabezas,
                           FFN 128, posición aprendida, GAP (Vaswani et al. 2017; versión reducida de
                           build_transformer de legacy/src/train_v4.py).
  · construir_resnet1d     brazo I: ResNet 1D de Wang et al. (2017), la línea base de clasificación de
                           series de tiempo que Ismail Fawaz et al. (2019) encontraron entre las mejores:
                           3 bloques residuales con ventanas 8-5-3 y 64 filtros, BN, ReLU, GAP.
  · construir_bigru        brazo B: la arquitectura canónica (Conv1D + recurrente bidireccional + atención
                           + densas) con GRU en lugar de LSTM, en los tamaños de c07
                           (docs/BUSQUEDA_HIPERPARAMETROS.md): conv 1×64 k = 5, BiGRU 128, densas 128-64.

Las configuraciones (CONFIG_X, CONFIG_I, CONFIG_B) se fijan ANTES de ver resultados y no se cambian. Una
por familia, sin variantes.

Este módulo no entrena nada; lo importa src/experimento_arquitecturas_2.py. Hay que importarlo antes de
cargar un modelo X guardado: registra la capa PosicionAprendida.
"""

from __future__ import annotations

N_PASOS = 30

# ---------------------------------------------------------------------------
# Configuraciones fijadas antes de ver resultados (una por familia)
# ---------------------------------------------------------------------------

CONFIG_X = {"d_modelo": 64, "bloques": 2, "cabezas": 4, "key_dim": 64, "ffn": 128, "activacion_ffn": "gelu",
            "posicion": "aprendida", "norma": "pre-norm", "dropout_bloque": 0.2, "dropout_final": 0.3,
            "pooling": "GAP", "epsilon_ln": 1e-6}
CONFIG_I = {"filtros": 64, "ventanas": [8, 5, 3], "bloques": 3, "padding": "same", "dropout_final": 0.3,
            "pooling": "GAP"}
CONFIG_B = {"conv_capas": 1, "conv_filtros": 64, "conv_ventana": 5, "gru": [128], "densas": [128, 64],
            "dropout": 0.3, "recurrente": "GRU (reset_after=True, por defecto de Keras)"}

# Receta de entrenamiento: la de T (Adam lr 1e-3 con CosineDecay, label smoothing 0,1, lote 64). B usa el
# lr de c07 (5e-4), la mejor versión de su familia en la búsqueda de hiperparámetros.
ENTRENAMIENTO = {
    "X": {"lr": 1e-3, "batch": 64, "label_smoothing": 0.1},
    "I": {"lr": 1e-3, "batch": 64, "label_smoothing": 0.1},
    "B": {"lr": 5e-4, "batch": 64, "label_smoothing": 0.1},
}


def _tf():
    import tensorflow as tf
    return tf


# ---------------------------------------------------------------------------
# Capa propia: codificación posicional aprendida
# ---------------------------------------------------------------------------

_REGISTRADA = {}


def capa_posicion_aprendida():
    """Clase PosicionAprendida registrada para serialización (.keras): suma a la entrada (b, T, d) una
    matriz entrenable (T, d), una fila por posición."""
    if "PosicionAprendida" in _REGISTRADA:
        return _REGISTRADA["PosicionAprendida"]
    tf = _tf()

    @tf.keras.utils.register_keras_serializable(package="arquitecturas_2")
    class PosicionAprendida(tf.keras.layers.Layer):
        def build(self, input_shape):
            self.pos = self.add_weight(name="pos", shape=(int(input_shape[1]), int(input_shape[2])),
                                       initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                                       trainable=True)

        def call(self, x):
            return x + self.pos

    _REGISTRADA["PosicionAprendida"] = PosicionAprendida
    return PosicionAprendida


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def construir_transformer(n_frames=N_PASOS, n_features=252, n_classes=50, cfg=CONFIG_X):
    """Brazo X. Dense(252 -> 64) -> + posición aprendida (30 × 64) -> Dropout -> 2 bloques pre-norm
    [LN -> MHA (4 cabezas, key_dim 64) -> Dropout -> + atajo; LN -> Dense 128 (GELU) -> Dropout ->
    Dense 64 -> Dropout -> + atajo] -> LN final -> GAP -> Dropout 0,3 -> softmax.

    key_dim = 64 por cabeza (el tamaño de cabeza del ejemplo de Keras para series de tiempo; con key_dim =
    d/cabezas = 16 el modelo tendría unos 88 mil parámetros, por debajo del rango 150-400 mil pedido)."""
    tf = _tf()
    L = tf.keras.layers
    Pos = capa_posicion_aprendida()
    d, dr, eps = cfg["d_modelo"], cfg["dropout_bloque"], cfg["epsilon_ln"]
    inp = tf.keras.Input(shape=(n_frames, n_features))
    x = L.Dense(d)(inp)
    x = Pos()(x)
    x = L.Dropout(dr)(x)
    for _ in range(cfg["bloques"]):
        h = L.LayerNormalization(epsilon=eps)(x)
        h = L.MultiHeadAttention(num_heads=cfg["cabezas"], key_dim=cfg["key_dim"], dropout=dr)(h, h)
        h = L.Dropout(dr)(h)
        x = L.Add()([x, h])
        h = L.LayerNormalization(epsilon=eps)(x)
        h = L.Dense(cfg["ffn"], activation=cfg["activacion_ffn"])(h)
        h = L.Dropout(dr)(h)
        h = L.Dense(d)(h)
        h = L.Dropout(dr)(h)
        x = L.Add()([x, h])
    x = L.LayerNormalization(epsilon=eps)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(cfg["dropout_final"])(x)
    x = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, x, name="transformer_pequeno")


def construir_resnet1d(n_frames=N_PASOS, n_features=252, n_classes=50, cfg=CONFIG_I):
    """Brazo I. ResNet de Wang et al. (2017) con 64 filtros en los 3 bloques. Cada bloque:
    [Conv1D k=8 -> BN -> ReLU -> Conv1D k=5 -> BN -> ReLU -> Conv1D k=3 -> BN] + atajo -> ReLU.
    Atajo: Conv1D 1×1 + BN si cambian los canales (primer bloque, 252 -> 64); si no, BN del atajo (como la
    implementación de Ismail Fawaz et al. 2019). Después GAP -> Dropout 0,3 (como T) -> softmax.
    Campo receptivo: 3 · (7 + 4 + 2) + 1 = 40 pasos (cubre los 30)."""
    tf = _tf()
    L = tf.keras.layers
    f = cfg["filtros"]
    inp = tf.keras.Input(shape=(n_frames, n_features))
    x = inp
    for _ in range(cfg["bloques"]):
        atajo = x
        h = x
        for j, k in enumerate(cfg["ventanas"]):
            h = L.Conv1D(f, k, padding=cfg["padding"])(h)
            h = L.BatchNormalization()(h)
            if j < len(cfg["ventanas"]) - 1:
                h = L.ReLU()(h)
        if atajo.shape[-1] != f:
            atajo = L.Conv1D(f, 1, padding="same")(atajo)
        atajo = L.BatchNormalization()(atajo)
        x = L.ReLU()(L.Add()([h, atajo]))
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(cfg["dropout_final"])(x)
    x = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, x, name="resnet1d_pequena")


def construir_recurrente(cfg, n_frames, n_features, n_classes, celda="GRU"):
    """Misma estructura que busqueda_hiperparametros.construir_modelo (la del canónico) con la celda
    recurrente como parámetro. Con celda = «LSTM» y cfg = c07 reproduce construir_modelo(c07) capa por capa
    (lo comprueba la prueba unitaria); con «GRU» es el brazo B."""
    tf = _tf()
    L = tf.keras.layers
    Celda = {"GRU": L.GRU, "LSTM": L.LSTM}[celda]
    unidades = cfg["gru"] if "gru" in cfg else cfg["lstm"]
    d = cfg["dropout"]
    inp = tf.keras.Input(shape=(n_frames, n_features))
    x = inp
    for _ in range(cfg["conv_capas"]):
        x = L.Conv1D(cfg["conv_filtros"], cfg["conv_ventana"], padding="same")(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
    x = L.Dropout(d * 0.5)(x)
    for u in unidades:
        x = L.Bidirectional(Celda(u, return_sequences=True))(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(d)(x)
    att = L.Dense(1)(x)
    att = L.Softmax(axis=1)(att)
    ctx = L.Multiply()([x, att])
    ctx = L.GlobalAveragePooling1D()(ctx)
    avg = L.GlobalAveragePooling1D()(x)
    mx = L.GlobalMaxPooling1D()(x)
    x = L.Concatenate()([ctx, avg, mx])
    d1, d2 = cfg["densas"]
    x = L.Dense(d1, activation="relu")(x)
    x = L.Dropout(d)(x)
    x = L.Dense(d2, activation="relu")(x)
    x = L.Dropout(d * 0.5)(x)
    x = L.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, x)


def construir_bigru(n_frames=N_PASOS, n_features=252, n_classes=50, cfg=CONFIG_B):
    """Brazo B: c07 con BiGRU en lugar de BiLSTM."""
    return construir_recurrente(cfg, n_frames, n_features, n_classes, celda="GRU")


CONSTRUCTORES = {"X": construir_transformer, "I": construir_resnet1d, "B": construir_bigru}
CONFIGS = {"X": CONFIG_X, "I": CONFIG_I, "B": CONFIG_B}
