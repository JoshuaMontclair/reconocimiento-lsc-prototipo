"""
Variantes del tratamiento de la mano no detectada al construir los 30 pasos (pedido del director,
reunión del 2-oct-2026, N-60). Módulo NUEVO: no modifica preprocess_lsc50.py, train_final_100.py ni
translator.py; experimento en src/experimento_mano_ausente.py, informe en docs/EXPERIMENTO_MANO_AUSENTE.md.

Punto de partida: la secuencia de un video tal como queda en el processed canónico
(data/processed_holistic_mascara_rostro, representación «manos»): T fotogramas × 126 rasgos
(mano izquierda 0:63, mano derecha 63:126), normalizados por hombros y con la regla «mascara» de D-07
ya aplicada, es decir, con el bloque de 63 valores de la mano no detectada puesto en 0 DESPUÉS de
normalizar. Una mano detectada nunca normaliza a 63 ceros exactos, así que «bloque todo en ceros» =
«mano no detectada en ese fotograma» (mismo criterio que preprocess_lsc50._manos_ausentes, aplicado
después de normalizar).

Las tres reglas devuelven una secuencia de n_pasos (30) × 126 posiciones. Velocidades, aumentación
y espejo se aplican DESPUÉS, igual que hoy (train_loso.entrenar_y_evaluar).

  mascara     (A, canónica) resample_sequence de train_final_100 sobre la secuencia enmascarada, sin
              cambios: el remuestreo lineal interpola entre un fotograma real y uno en ceros y produce
              pasos en cero o «encogidos» hacia el origen.
  pivote      (B) para cada uno de los n_pasos momentos se toma un fotograma REAL en el que la mano sí
              se detectó, por mano (las dos manos pueden salir de fotogramas distintos):
                1. posición continua del momento j: p_j = j·(T−1)/(n_pasos−1) (la misma de
                   resample_sequence) y fotograma nominal c_j = floor(p_j + 0,5);
                2. si la mano se detectó en c_j, se usa c_j (con T ≥ n_pasos los c_j son distintos);
                3. si no, se busca el fotograma detectado y todavía no usado más cercano a p_j
                   (orden ±1, ±2, …; a igual distancia, primero hacia adelante), dentro del intervalo
                   abierto entre el fotograma elegido para el momento anterior y el fotograma nominal
                   del siguiente momento cuya mano sí se detectó. Esa ventana conserva el orden
                   temporal (nunca se retrocede en el tiempo) y deja reservados los fotogramas
                   nominales detectados de los demás momentos. «Cercano» tiene un tope: el candidato
                   libre no puede estar más de un espaciado entre momentos ((T−1)/(n_pasos−1)
                   fotogramas, ≈ 2,3 con T = 67) más lejos de p_j que el fotograma que se repetiría en
                   el paso 4. Sin ese tope, en un hueco largo el primer momento saltaría al otro lado
                   del hueco (hasta 27 fotogramas en «niña» 0015_0001_0000) y todos los siguientes lo
                   repetirían;
                4. si no queda ninguno libre y cercano, se repite el más cercano a p_j de los dos
                   extremos de la ventana (el fotograma del momento anterior o el del siguiente
                   detectado; a igual distancia, el de adelante). En un hueco largo esto equivale a
                   «vecino más cercano»: la primera mitad repite el último cuadro antes del hueco y la
                   segunda el primero después;
                5. mano nunca detectada en el video: bloque en 0 (como hoy).
              Consecuencia: B no interpola nada; incluso los momentos con mano detectada pasan de
              interpolación lineal entre dos fotogramas a «fotograma más cercano».
  imputacion  (C) sobre los T fotogramas originales, por mano: cada fotograma sin la mano se rellena
              interpolando linealmente cada coordenada entre el último fotograma válido anterior y el
              primero válido siguiente (hueco de un solo fotograma = promedio de los dos vecinos);
              huecos al inicio o al final copian el valor válido más cercano; mano nunca detectada = 0.
              Luego resample_sequence a n_pasos como hoy. La interpolación se hace en el espacio
              normalizado por hombros (posición de la mano respecto del cuerpo), que es el que existe
              en el processed y el que tendría el prototipo justo después de normalize_sequence.
"""

from __future__ import annotations

import numpy as np

N_MANO = 63                       # 21 puntos × 3 coordenadas
BLOQUES_MANOS = {"mano_izq": slice(0, N_MANO), "mano_der": slice(N_MANO, 2 * N_MANO)}
REGLAS = ("mascara", "pivote", "imputacion")


def resample_sequence(seq, target):
    """Importa la función canónica de train_final_100 sin efectos secundarios (perezoso)."""
    from src.train_final_100 import resample_sequence as rs
    return rs(seq, target)


def mano_detectada(seq: np.ndarray, bloque: slice) -> np.ndarray:
    """(T,) booleano: True si el bloque de la mano NO es todo ceros en ese fotograma."""
    return ~np.all(seq[:, bloque] == 0, axis=1)


# ---------------------------------------------------------------------------
# B: pivote
# ---------------------------------------------------------------------------

def posiciones_momentos(T: int, n_pasos: int) -> np.ndarray:
    """Posición continua de cada momento en el eje de fotogramas (la de resample_sequence)."""
    return np.linspace(0, T - 1, n_pasos)


def indices_pivote(detectada: np.ndarray, n_pasos: int) -> np.ndarray | None:
    """Fotograma real elegido para cada uno de los n_pasos momentos (regla B). None si la mano no se
    detectó en ningún fotograma. Ver el docstring del módulo para la definición exacta."""
    detectada = np.asarray(detectada, dtype=bool)
    T = len(detectada)
    if not detectada.any():
        return None
    p = posiciones_momentos(T, n_pasos)
    c = np.floor(p + 0.5).astype(np.int64)
    c = np.clip(c, 0, T - 1)
    espaciado = (T - 1) / (n_pasos - 1) if n_pasos > 1 else float(T)
    elegido = np.full(n_pasos, -1, dtype=np.int64)
    usados = set()
    for j in range(n_pasos):
        if detectada[c[j]] and c[j] not in usados:
            elegido[j] = c[j]
            usados.add(int(c[j]))
    for j in range(n_pasos):
        if elegido[j] >= 0:
            continue
        lo = int(elegido[j - 1]) if j > 0 else -1
        siguientes = [int(elegido[k]) for k in range(j + 1, n_pasos)
                      if elegido[k] >= 0 and detectada[c[k]] and elegido[k] == c[k]]
        hi = siguientes[0] if siguientes else T
        extremos = [e for e in (lo, hi) if 0 <= e < T and detectada[e]]
        if not extremos:          # solo si no hay ni anterior ni siguiente: el detectado más cercano
            extremos = [int(x) for x in np.flatnonzero(detectada)]
        repetir = min(extremos, key=lambda e: (abs(e - p[j]), 0 if e >= p[j] else 1))
        # «Cercano»: a lo sumo un espaciado entre momentos más lejos que el cuadro que se repetiría.
        tope = abs(repetir - p[j]) + espaciado
        mejor = None
        for f in range(lo + 1, hi):
            if detectada[f] and f not in usados and abs(f - p[j]) <= tope + 1e-9:
                clave = (abs(f - p[j]), 0 if f >= p[j] else 1)   # a igual distancia, adelante primero
                if mejor is None or clave < mejor[0]:
                    mejor = (clave, f)
        f = mejor[1] if mejor is not None else repetir
        elegido[j] = f
        usados.add(int(f))
    return elegido


def regla_pivote(seq: np.ndarray, n_pasos: int = 30, devolver_indices: bool = False):
    """Regla B sobre una secuencia (T, 126) -> (n_pasos, 126)."""
    seq = np.asarray(seq, dtype=np.float32)
    out = np.zeros((n_pasos, seq.shape[1]), dtype=np.float32)
    indices = {}
    for nombre, b in BLOQUES_MANOS.items():
        idx = indices_pivote(mano_detectada(seq, b), n_pasos)
        indices[nombre] = idx
        if idx is not None:
            out[:, b] = seq[idx, b]
    return (out, indices) if devolver_indices else out


# ---------------------------------------------------------------------------
# C: imputación
# ---------------------------------------------------------------------------

def imputar_fotogramas(seq: np.ndarray) -> np.ndarray:
    """Rellena, por mano, los fotogramas sin la mano interpolando linealmente entre el válido anterior
    y el siguiente; bordes = copia del válido más cercano; mano nunca detectada = 0. (T, 126)."""
    seq = np.asarray(seq, dtype=np.float32)
    out = seq.copy()
    t = np.arange(seq.shape[0], dtype=np.float64)
    for b in BLOQUES_MANOS.values():
        det = mano_detectada(seq, b)
        if det.all() or not det.any():
            continue
        tv = t[det]
        faltan = ~det
        for col in range(b.start, b.stop):
            # np.interp: lineal entre los válidos vecinos; constante fuera del rango (bordes).
            out[faltan, col] = np.interp(t[faltan], tv, seq[det, col].astype(np.float64))
    return out


def regla_imputacion(seq: np.ndarray, n_pasos: int = 30) -> np.ndarray:
    """Regla C: imputar sobre los fotogramas originales y remuestrear linealmente como hoy."""
    return np.asarray(resample_sequence(imputar_fotogramas(seq), n_pasos), dtype=np.float32)


def regla_mascara(seq: np.ndarray, n_pasos: int = 30) -> np.ndarray:
    """Regla A (canónica): exactamente resample_sequence sobre la secuencia enmascarada."""
    return resample_sequence(seq, n_pasos)


FUNCIONES = {"mascara": regla_mascara, "pivote": regla_pivote, "imputacion": regla_imputacion}


def aplicar_regla(seq: np.ndarray, regla: str, n_pasos: int = 30) -> np.ndarray:
    if regla not in FUNCIONES:
        raise ValueError(f"regla desconocida: {regla!r}; opciones {REGLAS}")
    return FUNCIONES[regla](seq, n_pasos)


def remuestrear_todas_con_regla(X_padded: np.ndarray, lengths, regla: str, n_pasos: int = 30) -> np.ndarray:
    """Equivalente a train_loso.remuestrear_todas, con la regla elegida. (N, n_pasos, F) float32."""
    N = X_padded.shape[0]
    X = np.zeros((N, n_pasos, X_padded.shape[2]), dtype=np.float32)
    for i in range(N):
        T = int(lengths[i])
        X[i] = aplicar_regla(X_padded[i, :T, :], regla, n_pasos)
    return X
