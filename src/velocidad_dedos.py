"""
Descomposición de las velocidades de cada mano en velocidad GLOBAL y velocidad RELATIVA de los puntos
(docs/EXPERIMENTO_HIPOTESIS_VELOCIDAD_DEDOS.md).

Por mano y en cada paso t (representación «manos», src/representacion.py: mano izquierda = columnas 0-62,
mano derecha = 63-125; dentro de cada mano la columna 3·p + c es la coordenada c del punto p):
  v_i = velocidad del punto i (i = 0..20) en (x, y, z); la de add_velocity: v[t] = x[t] − x[t−1], v[0] = 0
  g   = media de las 21 v_i                         (velocidad global de la mano, 3 valores)
  r_i = v_i − g                                     (velocidad del punto respecto a la mano, 21 × 3 = 63)

Como Σ_i r_i = 0, Σ_i ||v_i||² = 21·||g||² + Σ_i ||r_i||² (Pitágoras): la energía de las 63 velocidades se
reparte exactamente en una parte global y una relativa.

Formas de la entrada de la red (siempre a partir de las posiciones YA remuestreadas y aumentadas, como
add_velocity, para que el espejo de la aumentación siga siendo correcto):
  «actual»   126 posiciones + 126 velocidades (add_velocity)                       = 252  (T1, control)
  «g_r»      126 posiciones + g (6: izq_vx, izq_vy, izq_vz, der_vx, der_vy, der_vz) + r (126) = 258  (H1)
  «r»        126 posiciones + r (126)                                                = 252  (H2, H3)
Las 6 columnas de g son exactamente las de velocidades_reducidas «por_coordenada»; las 126 de r siguen el
orden de las velocidades de add_velocity.

Solo NumPy: no importa TensorFlow.
"""

from __future__ import annotations

import numpy as np

COORDS = 3
N_PUNTOS_MANO = 21
ANCHO_MANO = N_PUNTOS_MANO * COORDS          # 63
N_POS = 2 * ANCHO_MANO                       # 126
MANOS = ("izq", "der")
EJES = ("x", "y", "z")

FORMAS = {
    "actual": {"n_rasgos": 252, "descripcion": "126 posiciones + 126 velocidades (add_velocity)"},
    "g_r": {"n_rasgos": 258, "descripcion": "126 posiciones + g (6) + r (126)"},
    "r": {"n_rasgos": 252, "descripcion": "126 posiciones + r (126), sin velocidad global"},
}


def _comprobar_ancho(V: np.ndarray):
    if V.shape[-1] != N_POS:
        raise ValueError(f"se esperaban {N_POS} columnas (representacion «manos»); hay {V.shape[-1]}")


def velocidades(X: np.ndarray) -> np.ndarray:
    """vel[..., 0, :] = 0; vel[..., t, :] = X[..., t, :] − X[..., t−1, :] (la de add_velocity)."""
    X = np.asarray(X)
    vel = np.zeros_like(X)
    vel[..., 1:, :] = X[..., 1:, :] - X[..., :-1, :]
    return vel


def por_mano(V: np.ndarray) -> np.ndarray:
    """(…, 126) -> (…, 2, 21, 3)."""
    _comprobar_ancho(V)
    return V.reshape(V.shape[:-1] + (2, N_PUNTOS_MANO, COORDS))


def global_y_relativa(V: np.ndarray):
    """V (…, 126) -> g (…, 6) [izq_vx, izq_vy, izq_vz, der_vx, der_vy, der_vz] y r (…, 126) en el orden de V."""
    P = por_mano(V)
    g = P.mean(axis=-2, keepdims=True)                    # (…, 2, 1, 3)
    r = (P - g).reshape(V.shape)
    return g.reshape(V.shape[:-1] + (2 * COORDS,)).astype(V.dtype, copy=False), r.astype(V.dtype, copy=False)


def entrada_red(X: np.ndarray, forma: str) -> np.ndarray:
    """Entrada de la red a partir de las posiciones remuestreadas (y aumentadas) X (N, T, 126)."""
    _comprobar_ancho(X)
    V = velocidades(X)
    if forma == "actual":
        return np.concatenate([X, V], axis=-1)
    g, r = global_y_relativa(V)
    if forma == "g_r":
        return np.concatenate([X, g, r], axis=-1)
    if forma == "r":
        return np.concatenate([X, r], axis=-1)
    raise ValueError(f"forma desconocida: {forma}")


def funcion_entrada(forma: str):
    """X -> entrada de la red, para sustituir add_velocity en la tupla canon de train_loso."""
    if forma not in FORMAS:
        raise ValueError(f"forma desconocida: {forma}")
    return lambda X: entrada_red(X, forma)


def nombres_columnas(forma: str) -> list[str]:
    pos = [f"{m}_p{p}_{e}" for m in MANOS for p in range(N_PUNTOS_MANO) for e in EJES]
    v = [f"v_{m}_p{p}_{e}" for m in MANOS for p in range(N_PUNTOS_MANO) for e in EJES]
    g = [f"g_{m}_{e}" for m in MANOS for e in EJES]
    r = [f"r_{m}_p{p}_{e}" for m in MANOS for p in range(N_PUNTOS_MANO) for e in EJES]
    return {"actual": pos + v, "g_r": pos + g + r, "r": pos + r}[forma]


def reconstruir_v(g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Inversa de global_y_relativa: v_i = r_i + g (comprueba que «g_r» es una reparametrización de «actual»)."""
    G = g.reshape(g.shape[:-1] + (2, 1, COORDS))
    return (por_mano(r) + G).reshape(r.shape)


# ---------------------------------------------------------------------------
# Energía
# ---------------------------------------------------------------------------

def energias_por_paso(V: np.ndarray):
    """V (…, T, 126) -> (E_total, E_global, E_relativa), cada una (…, T, 2) por paso y mano:
    E_total = Σ_i ||v_i||², E_global = 21·||g||², E_relativa = Σ_i ||r_i||² (E_total = E_global + E_relativa)."""
    P = por_mano(np.asarray(V, dtype=np.float64))
    g = P.mean(axis=-2)                                   # (…, 2, 3)
    e_tot = np.sum(P ** 2, axis=(-1, -2))
    e_g = N_PUNTOS_MANO * np.sum(g ** 2, axis=-1)
    e_r = np.sum((P - g[..., None, :]) ** 2, axis=(-1, -2))
    return e_tot, e_g, e_r


# ---------------------------------------------------------------------------
# Pasos tocados por una transición presente <-> ausente
# ---------------------------------------------------------------------------

def fotogramas_que_intervienen(T: int, n_pasos: int = 30):
    """Para cada paso remuestreado j (posición p_j = linspace(0, T−1, n_pasos), la de resample_sequence),
    los fotogramas originales que intervienen con peso > 0: (lo, hi, usa_lo, usa_hi)."""
    p = np.linspace(0, T - 1, n_pasos)
    lo = np.floor(p).astype(np.int64)
    hi = np.minimum(lo + 1, T - 1)
    w = p - lo
    return lo, hi, w < 1 - 1e-9, w > 1e-9


def pasos_de_transicion(detectada: np.ndarray, n_pasos: int = 30):
    """detectada (T,) booleano de una mano en los fotogramas originales. Devuelve dos (n_pasos,) booleanos:
      transicion[t]: la velocidad del paso t (t ≥ 1) mezcla fotogramas con la mano y sin ella: entre los
                     fotogramas que intervienen en los pasos t−1 y t hay al menos uno detectado y uno no
                     detectado. Con «mascara» es el salto a 0 (o el «encogimiento» hacia el origen) y vuelta.
      artificial[t]: la definición más amplia de docs/EXPERIMENTO_VELOCIDADES_REDUCIDAS.md: en t o t−1
                     interviene algún fotograma sin la mano (incluye pares de pasos sin la mano, donde con
                     «mascara» la velocidad es 0).
    Si la mano nunca se detecta, ambos son todo False."""
    detectada = np.asarray(detectada, dtype=bool)
    T = len(detectada)
    trans = np.zeros(n_pasos, dtype=bool)
    artif = np.zeros(n_pasos, dtype=bool)
    if T == 0 or not detectada.any() or detectada.all():
        return trans, artif
    lo, hi, ulo, uhi = fotogramas_que_intervienen(T, n_pasos)
    hay_det = (ulo & detectada[lo]) | (uhi & detectada[hi])          # el paso usa algún fotograma con la mano
    hay_aus = (ulo & ~detectada[lo]) | (uhi & ~detectada[hi])        # el paso usa algún fotograma sin la mano
    trans[1:] = (hay_det[1:] | hay_det[:-1]) & (hay_aus[1:] | hay_aus[:-1])
    artif[1:] = hay_aus[1:] | hay_aus[:-1]
    return trans, artif


def mascaras_transicion(X_padded: np.ndarray, lengths, n_pasos: int = 30):
    """(N, n_pasos, 2) booleanos «transicion» y «artificial» por video, paso y mano (0 = izq, 1 = der)."""
    N = X_padded.shape[0]
    trans = np.zeros((N, n_pasos, 2), dtype=bool)
    artif = np.zeros((N, n_pasos, 2), dtype=bool)
    for i in range(N):
        T = int(lengths[i])
        for m in range(2):
            det = ~np.all(X_padded[i, :T, m * ANCHO_MANO:(m + 1) * ANCHO_MANO] == 0, axis=1)
            trans[i, :, m], artif[i, :, m] = pasos_de_transicion(det, n_pasos)
    return trans, artif


def descomponer_r_rigida(X: np.ndarray):
    """Qué parte de r es movimiento RÍGIDO de la mano (giro pequeño + cambio de escala aparente) y qué parte es
    deformación (dedos que se doblan, mano que se abre o se cierra). Por mano y paso t ≥ 1, con d_i = posición del
    punto i en el punto medio de t−1 y t menos el centroide de la mano, se ajusta por mínimos cuadrados
        r_i ≈ ω × d_i + s · d_i        (ω: 3 valores, giro; s: 1 valor, escala)
    sobre los 63 valores. Devuelve (E_r, E_giro_escala, E_deformacion), cada una (N, T, 2), con
    E_r = Σ_i||r_i||², E_deformacion = residuo del ajuste y E_giro_escala = E_r − E_deformacion. Descriptivo: z de
    MediaPipe no tiene la misma escala que x, y, así que el «giro» es aproximado."""
    X = np.asarray(X, dtype=np.float64)
    V = velocidades(X)
    _, r = global_y_relativa(V)
    R = por_mano(r)[:, 1:]                                         # (N, T−1, 2, 21, 3)
    P = por_mano(X)
    M = 0.5 * (P[:, 1:] + P[:, :-1])
    D = M - M.mean(axis=-2, keepdims=True)                         # (N, T−1, 2, 21, 3)
    dx, dy, dz = D[..., 0], D[..., 1], D[..., 2]
    z = np.zeros_like(dx)
    # Columnas de la matriz de diseño (63 × 4) para (ω_x, ω_y, ω_z, s): ω × d = (ω_y d_z − ω_z d_y,
    # ω_z d_x − ω_x d_z, ω_x d_y − ω_y d_x).
    cols = [np.stack([z, -dz, dy], axis=-1), np.stack([dz, z, -dx], axis=-1), np.stack([-dy, dx, z], axis=-1), D]
    A = np.stack([c.reshape(c.shape[:-2] + (63,)) for c in cols], axis=-1)      # (N, T−1, 2, 63, 4)
    b = R.reshape(R.shape[:-2] + (63,))
    AtA = np.einsum("...ki,...kj->...ij", A, A) + 1e-12 * np.eye(4)
    Atb = np.einsum("...ki,...k->...i", A, b)
    coef = np.linalg.solve(AtA, Atb[..., None])[..., 0]
    res = b - np.einsum("...kj,...j->...k", A, coef)
    e_r = np.sum(b ** 2, axis=-1)
    e_def = np.sum(res ** 2, axis=-1)
    pad = np.zeros((X.shape[0], 1, 2))
    e_r, e_def = np.concatenate([pad, e_r], axis=1), np.concatenate([pad, e_def], axis=1)
    return e_r, e_r - e_def, e_def


# ---------------------------------------------------------------------------
# Rasgos resumidos para la sonda lineal
# ---------------------------------------------------------------------------

def resumen_temporal(F: np.ndarray) -> np.ndarray:
    """F (N, T, k) -> (N, 2k): media y desviación a lo largo de los pasos 2..T (el paso 1 tiene velocidad 0)."""
    F = np.asarray(F, dtype=np.float64)[:, 1:, :]
    return np.concatenate([F.mean(axis=1), F.std(axis=1)], axis=1)


def rasgos_sonda(X: np.ndarray) -> dict:
    """Rasgos por video para la sonda lineal, a partir de las posiciones remuestreadas X (N, T, 126):
      g      media y DE temporal de las 6 columnas de g                      -> 12
      r      media y DE temporal de las 126 columnas de r                    -> 252
      r_12   por mano y eje, la raíz cuadrática media de r sobre los 21 puntos; media y DE temporal -> 12
             (mismo número de rasgos que g: separa «información» de «dimensión»)
      g_r    g y r juntos                                                    -> 264
      v      media y DE temporal de las 126 velocidades completas            -> 252
    """
    V = velocidades(np.asarray(X, dtype=np.float32))
    g, r = global_y_relativa(V)
    R = por_mano(r.astype(np.float64))                                    # (N, T, 2, 21, 3)
    rms = np.sqrt(np.mean(R ** 2, axis=-2)).reshape(R.shape[0], R.shape[1], 6)
    out = {"g": resumen_temporal(g), "r": resumen_temporal(r), "r_12": resumen_temporal(rms)}
    out["g_r"] = np.concatenate([out["g"], out["r"]], axis=1)
    out["v"] = resumen_temporal(V)
    return out
