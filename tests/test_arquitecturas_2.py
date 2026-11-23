"""Pruebas de src/arquitecturas_2.py y de las utilidades de src/experimento_arquitecturas_2.py.
Construyen modelos (TensorFlow) pero no entrenan nada."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

tf = pytest.importorskip("tensorflow")

from src import arquitecturas_2 as a2  # noqa: E402
from src import arquitecturas_literatura as al  # noqa: E402
from src import busqueda_hiperparametros as bh  # noqa: E402
from src import experimento_arquitecturas as ea  # noqa: E402
from src import experimento_arquitecturas_2 as ea2  # noqa: E402


def _firma(model):
    def limpiar(o):
        if isinstance(o, dict):
            return {k: limpiar(v) for k, v in o.items() if k != "name"}
        if isinstance(o, list):
            return [limpiar(v) for v in o]
        return o
    return [(type(l).__name__, limpiar(l.get_config())) for l in model.layers]


def test_parametros_en_rango_y_fijos():
    esperado = {"X": 187_890, "I": 315_954, "B": 341_171}
    for b, fn in a2.CONSTRUCTORES.items():
        m = fn(30, 252, 50)
        assert m.count_params() == esperado[b] == ea2.N_PARAM[b], b
        assert 150_000 <= m.count_params() <= 400_000
        assert m.input_shape == (None, 30, 252) and m.output_shape == (None, 50)
        tf.keras.backend.clear_session()


def test_salida_softmax():
    x = np.random.default_rng(0).standard_normal((3, 30, 252)).astype(np.float32)
    for fn in a2.CONSTRUCTORES.values():
        p = fn()(x, training=False).numpy()
        assert p.shape == (3, 50) and np.allclose(p.sum(axis=1), 1, atol=1e-5) and np.all(p >= 0)
        tf.keras.backend.clear_session()


def test_recurrente_con_lstm_reproduce_c07():
    """construir_recurrente con LSTM y c07 es exactamente construir_modelo(c07): B solo cambia la celda."""
    c07 = bh.config_por_id("c07")
    tf.keras.utils.set_random_seed(0)
    a = bh.construir_modelo(c07, 30, 252, 50)
    tf.keras.utils.set_random_seed(0)
    b = a2.construir_recurrente(c07, 30, 252, 50, celda="LSTM")
    assert a.count_params() == b.count_params() == 389_811
    assert _firma(a) == _firma(b)
    for wa, wb in zip(a.get_weights(), b.get_weights()):
        assert np.array_equal(wa, wb)
    # B tiene los mismos tamaños que c07
    assert {k: a2.CONFIG_B[k] for k in ("conv_capas", "conv_filtros", "conv_ventana", "densas", "dropout")} == \
        {k: c07[k] for k in ("conv_capas", "conv_filtros", "conv_ventana", "densas", "dropout")}
    assert a2.CONFIG_B["gru"] == c07["lstm"] and a2.ENTRENAMIENTO["B"]["lr"] == c07["lr"]
    g = a2.construir_bigru()
    assert sum(isinstance(l, tf.keras.layers.Bidirectional) and isinstance(l.forward_layer, tf.keras.layers.GRU)
               for l in g.layers) == 1
    tf.keras.backend.clear_session()


def test_resnet_estructura():
    m = a2.construir_resnet1d()
    convs = [l for l in m.layers if isinstance(l, tf.keras.layers.Conv1D)]
    ventanas = [l.kernel_size[0] for l in convs if l.kernel_size[0] != 1]
    assert ventanas == [8, 5, 3] * 3
    assert all(l.filters == 64 for l in convs)
    assert sum(isinstance(l, tf.keras.layers.Add) for l in m.layers) == 3
    assert sum(l.kernel_size[0] == 1 for l in convs) == 1          # atajo 1x1 solo en el primer bloque
    assert any(isinstance(l, tf.keras.layers.GlobalAveragePooling1D) for l in m.layers)
    tf.keras.backend.clear_session()


def test_transformer_estructura_y_posicion():
    m = a2.construir_transformer()
    mha = [l for l in m.layers if isinstance(l, tf.keras.layers.MultiHeadAttention)]
    assert len(mha) == 2 and all(l.num_heads == 4 for l in mha)
    assert sum(isinstance(l, tf.keras.layers.LayerNormalization) for l in m.layers) == 5   # pre-norm + final
    # Con la posición aprendida el modelo distingue el orden temporal.
    x = np.random.default_rng(1).standard_normal((2, 30, 252)).astype(np.float32)
    capa_pos = [l for l in m.layers if type(l).__name__ == "PosicionAprendida"][0]
    w = capa_pos.get_weights()[0]
    capa_pos.set_weights([np.random.default_rng(2).standard_normal(w.shape).astype(np.float32)])
    p1 = m(x, training=False).numpy()
    p2 = m(x[:, ::-1, :].copy(), training=False).numpy()
    assert not np.allclose(p1, p2, atol=1e-6)
    tf.keras.backend.clear_session()


def test_transformer_guarda_y_recarga(tmp_path):
    tf.keras.utils.set_random_seed(3)
    m = a2.construir_transformer()
    x = np.random.default_rng(3).standard_normal((2, 30, 252)).astype(np.float32)
    p = m(x, training=False).numpy()
    ruta = tmp_path / "x.keras"
    m.save(str(ruta))
    tf.keras.backend.clear_session()
    m2 = tf.keras.models.load_model(str(ruta))
    assert np.allclose(m2(x, training=False).numpy(), p, atol=1e-6)
    tf.keras.backend.clear_session()


def test_constructor_ampliado():
    fn, hp = ea2.constructor("T")
    assert hp == dict(al.ENTRENAMIENTO_COMUN, arquitectura=al.CONFIG_T)
    assert fn(30, 252, 50).count_params() == 156_402
    assert ea.constructor is ea2.constructor            # entrenar_corrida usa el ampliado
    for b in ("X", "I", "B"):
        fn, hp = ea2.constructor(b)
        assert hp["batch"] == 64 and hp["label_smoothing"] == 0.1 and hp["arquitectura"] == a2.CONFIGS[b]
    assert ea2.constructor("B")[1]["lr"] == 5e-4 and ea2.constructor("X")[1]["lr"] == 1e-3
    tf.keras.backend.clear_session()


def test_voto_3_semillas():
    preds = [np.array([1, 2, 3, 4]), np.array([1, 5, 6, 4]), np.array([7, 5, 9, 0])]
    conf = [np.array([.9, .9, .2, .5]), np.array([.9, .9, .8, .5]), np.array([.9, .9, .3, .5])]
    v, sm = ea2.voto_3_semillas(preds, conf)
    assert v.tolist() == [1, 5, 6, 4] and sm == 1            # muestra 2: sin mayoría -> semilla más confiada


def test_mcnemar():
    pa = {"a": (0, 0), "b": (1, 1), "c": (2, 0), "d": (3, 3)}
    pb = {"a": (0, 0), "b": (1, 0), "c": (2, 2), "d": (3, 0)}
    r = ea2.mcnemar(pa, pb)
    assert r["aciertos_brazo"] == 3 and r["aciertos_T"] == 2
    assert r["solo_brazo_acierta"] == 2 and r["solo_T_acierta"] == 1 and r["ambos_aciertan"] == 1
    assert r["ninguno"] == 0 and r["p_exacto_bilateral"] == pytest.approx(1.0)
