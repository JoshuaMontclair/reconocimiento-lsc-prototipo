"""
Pruebas del conversor de glosas por reglas (src/conversor_glosas.py, actividad 3.a).

PROPÓSITO: comprobar que la implementación reproduce EXACTAMENTE docs/CONVERSOR_3a_ESPECIFICACION.md
(«la especificación»): las 50 salidas solas de §2.3, las 22 plantillas de §4.2, las 92 secuencias de
§4.3, los 35 motivos de §5.2, las cifras de §9.1 y las garantías de §5.1. Las cifras de §9.1 se midieron
primero sobre el borrador 2 de la especificación (versión 3a-1); desde la versión 3a-2 (segunda revisión
del 8-nov: 5 plantillas y 2 ramas nuevas) son las medidas con la implementación, y el recuento de motivos
de los 251.125 conjuntos (MOTIVOS_CONJUNTOS) es un dato de regresión medido con ella, no con el borrador. Las tablas esperadas de este archivo se copiaron de la especificación, no del
módulo, para que una deriva de cualquiera de los dos se vea como una prueba que falla.

QUÉ NO PRUEBAN: que las frases sean las que diría una persona señante. Ninguna regla está revisada
todavía por el profesor de LSC vinculado (Actividad B, §9.2 de la especificación).

Sin TensorFlow, sin mediapipe, sin corpus y sin audio. Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_conversor_glosas -v
"""

from __future__ import annotations

import ast
import collections
import functools
import hashlib
import itertools
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import conversor_glosas as cg  # noqa: E402
from src.conversor_glosas import ConversorGlosas  # noqa: E402
from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402

PY = sys.executable
MD5_GRAMMAR_MAPPER = "ce3ac6c8f7b841bbd02118eca0f2d320"


# ---------------------------------------------------------------------------
# Tablas esperadas (copiadas de la especificación)
# ---------------------------------------------------------------------------

# §2.3, columna «sola»
SOLAS: dict[int, str] = {
    0: "Gracias.", 1: "Buenos días.", 2: "Buenas tardes.", 3: "Buenas noches.", 4: "Seña.",
    5: "Nombre.", 6: "Trabajar.", 7: "Comer.", 8: "Vivir.", 9: "Poco.", 10: "Familia.",
    11: "Personas.", 12: "Mujer.", 13: "Hombre.", 14: "Niño.", 15: "Niña.", 16: "Abuelo.",
    17: "Tío.", 18: "Hermano.", 19: "Hambre.", 20: "Feliz.", 21: "Contento.", 22: "Triste.",
    23: "Aburrido.", 24: "Bien.", 25: "Mal.", 26: "¿Cómo estás?", 27: "Más o menos.",
    28: "Sentir.", 29: "Juicioso.", 30: "Hola.", 31: "Adiós.", 32: "Por favor.", 33: "Con gusto.",
    34: "Bienvenido.", 35: "Perdón.", 36: "Permiso.", 37: "Nunca.", 38: "Yo.", 39: "Tú.",
    40: "Ustedes.", 41: "¿Qué?", 42: "¿Cuándo?", 43: "¿Dónde?", 44: "¿Cómo?", 45: "¿Por qué?",
    46: "¿Quién?", 47: "Diferente.", 48: "Todos.", 49: "Mucho.",
}

# §4.2: id, glosas, enunciado fijo, admite saludo
PLANTILLAS_ESPERADAS = [
    ("T-01", ["por favor", "seña"], "La seña, por favor.", False),
    ("T-02", ["por favor", "nombre"], "El nombre, por favor.", False),
    ("T-03", ["tú", "nombre", "¿qué?"], "¿Cuál es tu nombre?", True),
    ("T-04", ["tú", "seña", "¿qué?"], "¿Cuál es tu seña?", True),
    ("T-05", ["gracias", "mucho"], "Muchas gracias.", False),
    ("T-06", ["con gusto", "mucho"], "Con mucho gusto.", False),
    ("T-07", ["gracias", "todos"], "Gracias a todos.", False),
    ("T-08", ["perdón", "permiso"], "Perdón, con permiso.", False),
    ("T-09", ["por favor", "permiso"], "Permiso, por favor.", False),
    ("T-10", ["perdón", "¿qué?"], "Perdón, ¿qué?", False),
    ("T-11", ["perdón", "¿cómo?"], "Perdón, ¿cómo?", False),
    ("T-12", ["bien", "gracias"], "Bien, gracias.", False),
    ("T-13", ["más o menos", "gracias"], "Más o menos, gracias.", False),
    ("T-14", ["yo", "bien", "gracias"], "Yo estoy bien, gracias.", False),
    ("T-15", ["gracias", "adiós"], "Gracias, adiós.", False),
    ("T-16", ["gracias", "buenas noches"], "Gracias, buenas noches.", False),
    ("T-17", ["mucho", "gracias", "todos"], "Muchas gracias a todos.", False),
    ("T-18", ["gracias", "buenos días"], "Gracias, buenos días.", False),
    ("T-19", ["gracias", "buenas tardes"], "Gracias, buenas tardes.", False),
    ("T-20", ["gracias", "ustedes"], "Gracias a ustedes.", False),
    ("T-21", ["gracias", "tú"], "Gracias a ti.", False),
    ("T-22", ["mal", "gracias"], "Mal, gracias.", False),
]

# §4.3: S | glosas | enunciado fijo | vía | reglas (o plantilla)
_INVENTARIO_TXT = """
S-01|buenos días + todos|Buenos días a todos.|reglas|R-09, R-10
S-02|buenas tardes + todos|Buenas tardes a todos.|reglas|R-09, R-10
S-03|tío + juicioso|El tío es juicioso.|reglas|R-15, R-18, R-20, R-21
S-04|familia + feliz|La familia está feliz.|reglas|R-15, R-17, R-18, R-20, R-21
S-05|por favor + seña|La seña, por favor.|plantilla|T-01
S-06|hola + bienvenido|Hola, bienvenido.|reglas|R-09, R-12
S-07|hermano + trabajar|El hermano trabaja.|reglas|R-15, R-18, R-19
S-08|¿dónde? + hermano + vivir|¿Dónde vive el hermano?|reglas|R-15, R-18, R-30
S-09|hola + ¿cómo estás?|Hola, ¿cómo estás?|reglas|R-09, R-12
S-10|hola + todos|Hola a todos.|reglas|R-09, R-10
S-11|buenas noches + todos|Buenas noches a todos.|reglas|R-09, R-10
S-12|adiós + todos|Adiós a todos.|reglas|R-09, R-10
S-13|hola + buenos días|Hola, buenos días.|reglas|R-09
S-14|buenas tardes + ¿cómo estás?|Buenas tardes, ¿cómo estás?|reglas|R-09, R-12
S-15|hola + abuelo|Hola, abuelo.|reglas|R-09, R-11
S-16|buenos días + familia|Buenos días, familia.|reglas|R-09, R-11
S-17|hola + ¿cómo estás? + familia|Hola, ¿cómo está la familia?|reglas|R-09, R-15, R-18, R-32
S-18|¿cómo estás? + ustedes|¿Cómo están ustedes?|reglas|R-18, R-32
S-19|¿cómo estás? + todos|¿Cómo están todos?|reglas|R-18, R-32
S-20|buenos días + bienvenido|Buenos días, bienvenido.|reglas|R-09, R-12
S-21|por favor + nombre|El nombre, por favor.|plantilla|T-02
S-22|tú + nombre + ¿qué?|¿Cuál es tu nombre?|plantilla|T-03
S-23|tú + seña + ¿qué?|¿Cuál es tu seña?|plantilla|T-04
S-24|gracias + mucho|Muchas gracias.|plantilla|T-05
S-25|con gusto + mucho|Con mucho gusto.|plantilla|T-06
S-26|gracias + todos|Gracias a todos.|plantilla|T-07
S-27|perdón + permiso|Perdón, con permiso.|plantilla|T-08
S-28|por favor + permiso|Permiso, por favor.|plantilla|T-09
S-29|perdón + ¿qué?|Perdón, ¿qué?|plantilla|T-10
S-30|perdón + ¿cómo?|Perdón, ¿cómo?|plantilla|T-11
S-31|bien + gracias|Bien, gracias.|plantilla|T-12
S-32|más o menos + gracias|Más o menos, gracias.|plantilla|T-13
S-33|yo + bien + gracias|Yo estoy bien, gracias.|plantilla|T-14
S-34|gracias + adiós|Gracias, adiós.|plantilla|T-15
S-35|gracias + buenas noches|Gracias, buenas noches.|plantilla|T-16
S-36|mucho + gracias + todos|Muchas gracias a todos.|plantilla|T-17
S-37|yo + trabajar|Yo trabajo.|reglas|R-16, R-18, R-19
S-38|tú + comer|Tú comes.|reglas|R-16, R-18, R-19
S-39|ustedes + trabajar|Ustedes trabajan.|reglas|R-16, R-17, R-18, R-19
S-40|personas + comer|Las personas comen.|reglas|R-15, R-18, R-19
S-41|todos + trabajar + mucho|Todos trabajan mucho.|reglas|R-18, R-19, R-25
S-42|yo + nunca + comer|Yo nunca como.|reglas|R-16, R-18, R-26
S-43|familia + vivir + bien|La familia vive bien.|reglas|R-15, R-18, R-22
S-44|yo + feliz|Yo estoy feliz.|reglas|R-16, R-18, R-20, R-21
S-45|niña + contento|La niña está contenta.|reglas|R-15, R-18, R-20, R-21
S-46|ustedes + aburrido|Ustedes están aburridos.|reglas|R-17, R-18, R-20, R-21
S-47|hermano + diferente|El hermano es diferente.|reglas|R-15, R-18, R-20, R-21
S-48|yo + bien|Yo estoy bien.|reglas|R-16, R-18, R-22
S-49|abuelo + mal|El abuelo está mal.|reglas|R-15, R-18, R-22
S-50|yo + mucho + feliz|Yo estoy muy feliz.|reglas|R-18, R-20, R-21, R-25
S-51|yo + sentir + triste|Yo me siento triste.|reglas|R-18, R-21, R-23
S-52|tú + sentir + bien|Tú te sientes bien.|reglas|R-18, R-23
S-53|yo + hambre|Yo tengo hambre.|reglas|R-18, R-24
S-54|niño + hambre + mucho|El niño tiene mucha hambre.|reglas|R-15, R-18, R-24, R-25
S-55|todos + bienvenido|Todos son bienvenidos.|reglas|R-17, R-18, R-20, R-21
S-56|tú + bienvenido|Tú eres bienvenido.|reglas|R-17, R-18, R-20, R-21
S-57|¿dónde? + tú + vivir|¿Dónde vives tú?|reglas|R-16, R-18, R-30
S-58|¿cuándo? + hermano + trabajar|¿Cuándo trabaja el hermano?|reglas|R-15, R-18, R-30
S-59|¿qué? + niño + comer|¿Qué come el niño?|reglas|R-15, R-18, R-30
S-60|¿quién? + comer|¿Quién come?|reglas|R-18, R-31
S-61|¿quién? + hambre|¿Quién tiene hambre?|reglas|R-24, R-31
S-62|¿quién? + tú|¿Quién eres tú?|reglas|R-18, R-31
S-63|¿quién? + ustedes|¿Quiénes son ustedes?|reglas|R-18, R-31
S-64|¿cómo? + abuelo|¿Cómo está el abuelo?|reglas|R-15, R-18, R-30
S-65|¿cómo? + tú + sentir|¿Cómo te sientes tú?|reglas|R-18, R-23, R-30
S-66|¿por qué? + niño + triste|¿Por qué está triste el niño?|reglas|R-15, R-18, R-20, R-21, R-30
S-67|¿por qué? + tú + trabajar + mucho|¿Por qué trabajas mucho tú?|reglas|R-18, R-25, R-30
S-68|hola + ¿dónde? + hermano + vivir|Hola, ¿dónde vive el hermano?|reglas|R-09, R-15, R-18, R-30
S-69|buenos días + familia + feliz|Buenos días, la familia está feliz.|reglas|R-09, R-15, R-17, R-18, R-20, R-21
S-70|hola + tú + nombre + ¿qué?|Hola, ¿cuál es tu nombre?|plantilla|T-03
S-71|hola + buenos días + ¿cómo? + ustedes|Hola, buenos días, ¿cómo están ustedes?|reglas|R-09, R-18, R-30
S-72|hola + buenos días + ¿cómo estás?|Hola, buenos días, ¿cómo estás?|reglas|R-09, R-12
S-73|tío + aburrido + nunca|El tío nunca está aburrido.|reglas|R-15, R-18, R-20, R-21, R-26
S-74|yo + nunca + hambre|Yo nunca tengo hambre.|reglas|R-16, R-18, R-24, R-26
S-75|yo + nunca + sentir + diferente|Yo nunca me siento diferente.|reglas|R-16, R-18, R-21, R-23, R-26
S-76|hermano + comer + poco|El hermano come poco.|reglas|R-15, R-18, R-25
S-77|yo + hambre + poco|Yo tengo poca hambre.|reglas|R-16, R-18, R-24, R-25
S-78|¿quién? + feliz|¿Quién está feliz?|reglas|R-18, R-20, R-31
S-79|¿quién? + mal|¿Quién está mal?|reglas|R-18, R-22, R-31
S-80|¿por qué? + tú + sentir + triste|¿Por qué te sientes triste tú?|reglas|R-18, R-21, R-23, R-30
S-81|¿por qué? + niño + hambre|¿Por qué tiene hambre el niño?|reglas|R-15, R-18, R-24, R-30
S-82|¿cómo estás? + tú|¿Cómo estás tú?|reglas|R-18, R-32
S-83|tío + trabajar + bien + mucho|El tío trabaja muy bien.|reglas|R-15, R-18, R-22, R-25
S-84|yo + sentir + bienvenido|Yo me siento bienvenido.|reglas|R-16, R-18, R-21, R-23
S-85|adiós + abuelo|Adiós, abuelo.|reglas|R-09, R-11
S-86|gracias + buenos días|Gracias, buenos días.|plantilla|T-18
S-87|gracias + buenas tardes|Gracias, buenas tardes.|plantilla|T-19
S-88|gracias + ustedes|Gracias a ustedes.|plantilla|T-20
S-89|gracias + tú|Gracias a ti.|plantilla|T-21
S-90|mal + gracias|Mal, gracias.|plantilla|T-22
S-91|gracias + abuelo|Gracias, abuelo.|reglas|R-11, R-13
S-92|familia + bienvenido|Bienvenida, familia.|reglas|R-11, R-12, R-20, R-21
"""
INVENTARIO_ESPERADO = [
    (s, g.split(" + "), f, v, [x.strip() for x in r.split(",")])
    for s, g, f, v, r in (linea.split("|") for linea in _INVENTARIO_TXT.strip().splitlines())
]

# §5.2: motivo, entrada de ejemplo, salida de respaldo (los 34 alcanzables con estricto=False)
MOTIVOS_EJEMPLO = [
    ("fuera_de_vocabulario", ["yo", "perro", "comer"], "yo, comer"),
    ("demasiadas_glosas", ["hola", "yo", "trabajar", "mucho", "bien"], "hola, yo, trabajar, mucho, bien"),
    ("saludos_no_combinables", ["hola", "adiós"], "hola, adiós"),
    ("despedida_con_contenido", ["adiós", "bienvenido"], "adiós, bienvenido"),
    ("saludo_con_glosa_no_admitida", ["hola", "tú"], "hola, tú"),
    ("sustantivo_abstracto_fuera_de_plantilla", ["yo", "nombre"], "yo, nombre"),
    ("cortesia_fuera_de_plantilla", ["gracias", "yo", "comer"], "gracias, yo, comer"),
    ("dos_sujetos", ["hermano", "tío", "comer"], "hermano, tío, comer"),
    ("dos_interrogativos", ["¿qué?", "¿dónde?", "tú", "vivir"], "¿qué?, ¿dónde?, tú, vivir"),
    ("como_estas_no_admitido", ["¿cómo estás?", "yo"], "¿cómo estás?, yo"),
    ("quien_con_sujeto_y_predicado", ["¿quién?", "hermano", "comer"], "¿quién?, hermano, comer"),
    ("quien_sujeto_no_admitido", ["¿quién?", "familia"], "¿quién?, familia"),
    ("pregunta_sin_sujeto", ["¿dónde?", "comer"], "¿dónde?, comer"),
    ("pregunta_no_admitida", ["¿cuándo?", "hermano", "vivir"], "¿cuándo?, hermano, vivir"),
    ("sin_sujeto", ["comer", "mucho"], "comer, mucho"),
    ("dos_VERBO", ["yo", "comer", "trabajar"], "yo, comer, trabajar"),
    ("dos_ADJ", ["yo", "feliz", "triste"], "yo, feliz, triste"),
    ("dos_ADV_EST", ["yo", "bien", "mal"], "yo, bien, mal"),
    ("dos_ADV_CANT", ["yo", "mucho", "poco", "comer"], "yo, mucho, poco, comer"),
    ("nunca_con_cantidad", ["yo", "nunca", "mucho", "comer"], "yo, nunca, mucho, comer"),
    ("todos_con_nunca", ["todos", "nunca", "comer"], "todos, nunca, comer"),
    ("nunca_con_modo", ["yo", "nunca", "trabajar", "bien"], "yo, nunca, trabajar, bien"),
    ("dos_predicados", ["yo", "feliz", "bien"], "yo, feliz, bien"),
    ("dos_verbos", ["yo", "sentir", "trabajar"], "yo, sentir, trabajar"),
    ("verbo_con_predicado", ["hermano", "trabajar", "feliz"], "hermano, trabajar, feliz"),
    ("nunca_no_admitido", ["tío", "nunca", "juicioso"], "tío, nunca, juicioso"),
    ("cantidad_y_modo", ["tío", "trabajar", "bien", "poco"], "tío, trabajar, bien, poco"),
    ("cantidad_no_admitida", ["yo", "poco", "feliz"], "yo, poco, feliz"),
    ("vivir_sin_complemento", ["¿por qué?", "hermano", "vivir"], "¿por qué?, hermano, vivir"),
    ("sentir_sin_predicativo", ["yo", "sentir"], "yo, sentir"),
    ("sentir_hambre", ["yo", "sentir", "hambre"], "yo, sentir, hambre"),
    ("adjetivo_no_admite_sentir", ["yo", "sentir", "juicioso"], "yo, sentir, juicioso"),
    ("sin_predicado", ["yo", "nunca"], "yo, nunca"),
    ("yo_bienvenido", ["yo", "bienvenido"], "yo, bienvenido"),
]

# §9.1: motivos de respaldo en los 2.450 pares ordenados (17 motivos, suman 1.880; versión 3a-2)
MOTIVOS_PARES = {
    "cortesia_fuera_de_plantilla": 364, "sin_sujeto": 306, "saludo_con_glosa_no_admitida": 282,
    "pregunta_sin_sujeto": 180, "sustantivo_abstracto_fuera_de_plantilla": 170, "dos_sujetos": 156,
    "sin_predicado": 108, "pregunta_no_admitida": 80, "despedida_con_contenido": 74,
    "dos_interrogativos": 42, "como_estas_no_admitido": 38, "vivir_sin_complemento": 28,
    "sentir_sin_predicativo": 28, "saludos_no_combinables": 14, "quien_sujeto_no_admitido": 6,
    "yo_bienvenido": 2, "todos_con_nunca": 2,
}

# §9.1: motivos de respaldo en los 251.125 conjuntos de 2 a 4 glosas (32 motivos, suman 244.130). Dato de
# REGRESIÓN medido con la implementación 3a-2 (no con el borrador 2): fija el orden normativo de §3.1 en
# las entradas de 3 y 4 glosas, donde dos comprobaciones pueden chocar y los pares no lo ven.
MOTIVOS_CONJUNTOS = {
    "adjetivo_no_admite_sentir": 124, "cantidad_no_admitida": 1213, "cantidad_y_modo": 168,
    "como_estas_no_admitido": 4892, "cortesia_fuera_de_plantilla": 70700, "despedida_con_contenido": 15217,
    "dos_ADJ": 5467, "dos_ADV_CANT": 307, "dos_ADV_EST": 893, "dos_VERBO": 893, "dos_interrogativos": 12845,
    "dos_predicados": 5874, "dos_sujetos": 43069, "dos_verbos": 834, "nunca_con_cantidad": 586,
    "nunca_con_modo": 429, "nunca_no_admitido": 392, "pregunta_no_admitida": 12356, "pregunta_sin_sujeto": 8625,
    "quien_con_sujeto_y_predicado": 3159, "quien_sujeto_no_admitido": 24, "saludo_con_glosa_no_admitida": 249,
    "saludos_no_combinables": 7717, "sentir_hambre": 124, "sentir_sin_predicativo": 420, "sin_predicado": 622,
    "sin_sujeto": 8364, "sustantivo_abstracto_fuera_de_plantilla": 35514, "todos_con_nunca": 181,
    "verbo_con_predicado": 2664, "vivir_sin_complemento": 177, "yo_bienvenido": 31,
}

CLAVES_EXPLICAR = ["frase", "via", "plantilla", "reglas", "suplido", "supuestos", "respaldo", "motivo",
                   "glosas", "fuera_de_vocabulario", "n_fuera_de_vocabulario", "alias", "repeticiones",
                   "omitidas", "avisos", "parametros", "version"]

# §2.4 (copia independiente para el barrido de concordancia)
SUJETOS = {   # glosa → (texto, persona-número, género, número)
    "yo": ("yo", "1s", "m", "sg"), "tú": ("tú", "2s", "m", "sg"), "ustedes": ("ustedes", "3p", "m", "pl"),
    "todos": ("todos", "3p", "m", "pl"), "familia": ("la familia", "3s", "f", "sg"),
    "personas": ("las personas", "3p", "f", "pl"), "mujer": ("la mujer", "3s", "f", "sg"),
    "hombre": ("el hombre", "3s", "m", "sg"), "niño": ("el niño", "3s", "m", "sg"),
    "niña": ("la niña", "3s", "f", "sg"), "abuelo": ("el abuelo", "3s", "m", "sg"),
    "tío": ("el tío", "3s", "m", "sg"), "hermano": ("el hermano", "3s", "m", "sg"),
}
VERBOS = {
    "trabajar": {"1s": "trabajo", "2s": "trabajas", "3s": "trabaja", "3p": "trabajan"},
    "comer": {"1s": "como", "2s": "comes", "3s": "come", "3p": "comen"},
    "vivir": {"1s": "vivo", "2s": "vives", "3s": "vive", "3p": "viven"},
    "sentir": {"1s": "me siento", "2s": "te sientes", "3s": "se siente", "3p": "se sienten"},
    "ser": {"1s": "soy", "2s": "eres", "3s": "es", "3p": "son"},
    "estar": {"1s": "estoy", "2s": "estás", "3s": "está", "3p": "están"},
}
ADJETIVOS = {   # adjetivo → (cópula, ms, fs, mp, fp, con_sentir)
    "feliz": ("estar", "feliz", "feliz", "felices", "felices", True),
    "contento": ("estar", "contento", "contenta", "contentos", "contentas", True),
    "triste": ("estar", "triste", "triste", "tristes", "tristes", True),
    "aburrido": ("estar", "aburrido", "aburrida", "aburridos", "aburridas", True),
    "juicioso": ("ser", "juicioso", "juiciosa", "juiciosos", "juiciosas", False),
    "diferente": ("ser", "diferente", "diferente", "diferentes", "diferentes", True),
    "bienvenido": ("ser", "bienvenido", "bienvenida", "bienvenidos", "bienvenidas", True),
}


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def _forma_adj(adj: str, genero: str, numero: str) -> str:
    _, ms, fs, mp, fp, _ = ADJETIVOS[adj]
    return {"ms": ms, "fs": fs, "mp": mp, "fp": fp}[genero + ("s" if numero == "sg" else "p")]


# ---------------------------------------------------------------------------
# Barridos compartidos (se calculan una vez)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def barrido_pares():
    conv = ConversorGlosas()
    return {(a, b): conv.explicar([a, b]) for a, b in itertools.permutations(SIGN_VOCABULARY.values(), 2)}


@functools.lru_cache(maxsize=None)
def barrido_conjuntos(estricto: bool = False):
    """Los 251.125 conjuntos de 2 a 4 glosas, en orden de id → explicar."""
    conv = ConversorGlosas(estricto=estricto)
    return {tuple(g): conv.explicar(g) for g in cg.conjuntos()}


def problemas_de_forma(e: dict) -> list[str]:
    """Propiedades de §9.1 (prueba 3) sobre una salida; lista vacía si cumple todas."""
    f = e["frase"]
    p = []
    if not f:
        p.append("vacía")
    if "  " in f or " ," in f or ",," in f or f.endswith(",") or f.endswith(", ") or f != f.strip():
        p.append("espacios o comas colgantes")
    if f.count("¿") != f.count("?"):
        p.append("¿? desbalanceados")
    if e["respaldo"]:
        if f != ", ".join(e["glosas"][:8]) or e["omitidas"] != e["glosas"][8:]:
            p.append("respaldo mal formado")
        if f.endswith("."):
            p.append("respaldo con punto")
        if e["motivo"] is None:
            p.append("respaldo sin motivo")
    else:
        if not (f[0].isupper() or (f[0] == "¿" and f[1:2].isupper())):
            p.append("sin mayúscula inicial")
        if not (f.endswith(".") or f.endswith("?")):
            p.append("sin cierre")
        if ".." in f or ".?" in f:
            p.append("doble cierre")
        if e["motivo"] is not None:
            p.append("frase con motivo")
    return p


# ---------------------------------------------------------------------------
# 1. Las 50 glosas solas (§2.3, R-06)
# ---------------------------------------------------------------------------

class TestGlosasSolas(unittest.TestCase):

    def setUp(self):
        self.conv = ConversorGlosas()

    def test_salida_exacta_de_las_50(self):
        for i, esperada in SOLAS.items():
            g = SIGN_VOCABULARY[i]
            with self.subTest(glosa=g):
                e = self.conv.explicar([g])
                self.assertEqual(e["frase"], esperada)
                self.assertEqual(e["via"], "lexico")
                self.assertEqual(e["suplido"], ["puntuacion", "mayuscula"])
                self.assertEqual(e["supuestos"], [])

    def test_ids_repeticiones_y_distintas(self):
        for i, g in SIGN_VOCABULARY.items():
            self.assertEqual(self.conv.translate([g]), self.conv.translate([i]))
            self.assertEqual(self.conv.translate([g, g, g]), self.conv.translate([g]))
        salidas = [self.conv.translate([g]) for g in SIGN_VOCABULARY.values()]
        self.assertEqual(len(set(salidas)), 50)                     # requisito 5(b) del Apéndice A
        self.assertEqual(sum(s.endswith(".") for s in salidas), 43)
        self.assertEqual(sum(s.endswith("?") for s in salidas), 7)

    def test_tabla_lexico_del_modulo(self):
        self.assertEqual({SIGN_VOCABULARY[i]: s for i, s in SOLAS.items()},
                         {g: d["sola"] for g, d in cg.LEXICO.items()})


# ---------------------------------------------------------------------------
# 2. Plantillas (§4.2, R-07)
# ---------------------------------------------------------------------------

class TestPlantillas(unittest.TestCase):

    def setUp(self):
        self.conv = ConversorGlosas()

    def test_las_22_en_cualquier_orden(self):
        self.assertEqual(len(cg.PLANTILLAS), 22)
        for tid, glosas, frase, admite in PLANTILLAS_ESPERADAS:
            self.assertEqual(cg.PLANTILLAS[tid]["frase"], frase)
            self.assertEqual(cg.PLANTILLAS[tid]["admite_saludo"], admite)
            for perm in itertools.permutations(glosas):
                with self.subTest(plantilla=tid, orden=perm):
                    e = self.conv.explicar(list(perm))
                    self.assertEqual(e["frase"], frase)
                    self.assertEqual((e["via"], e["plantilla"]), ("plantilla", tid))
                    self.assertIn("R-07", e["reglas"])
                    self.assertEqual(e["suplido"], [])
                    self.assertEqual(e["supuestos"], ["tratamiento_tu"] if "tú" in glosas else [])

    def test_t03_y_t04_con_un_saludo(self):
        for saludo in ("hola", "buenos días", "buenas tardes", "buenas noches"):
            for tid, frase in (("T-03", "¿cuál es tu nombre?"), ("T-04", "¿cuál es tu seña?")):
                glosas = [saludo] + list(cg.PLANTILLAS[tid]["glosas"])
                e = self.conv.explicar(glosas)
                self.assertEqual(e["frase"], f"{_cap(saludo)}, {frase}")
                self.assertEqual((e["via"], e["plantilla"]), ("plantilla", tid))
                self.assertEqual(e["suplido"], ["puntuacion"])
                self.assertEqual(e["supuestos"], ["tratamiento_tu"])
                self.assertTrue({"R-07", "R-09"} <= set(e["reglas"]))
        # Con ADIÓS no: B5 va antes que B6
        self.assertEqual(self.conv.explicar(["adiós", "tú", "nombre", "¿qué?"])["motivo"],
                         "despedida_con_contenido")
        # Las que no admiten saludo pasan por el núcleo y la cortesía no se compone
        self.assertEqual(self.conv.explicar(["hola", "gracias", "mucho"])["motivo"], "cortesia_fuera_de_plantilla")


# ---------------------------------------------------------------------------
# 3. Inventario de 85 secuencias (§4.3)
# ---------------------------------------------------------------------------

class TestInventario(unittest.TestCase):

    def setUp(self):
        self.conv = ConversorGlosas()

    def test_92_secuencias_en_su_orden_y_en_el_inverso(self):
        self.assertEqual(len(INVENTARIO_ESPERADO), 92)
        for s, glosas, frase, via, reglas in INVENTARIO_ESPERADO:
            for orden in (glosas, glosas[::-1]):
                with self.subTest(secuencia=s, orden=orden):
                    e = self.conv.explicar(orden)
                    self.assertEqual(e["frase"], frase)
                    self.assertEqual(e["via"], via)
                    if via == "plantilla":
                        self.assertEqual(e["plantilla"], reglas[0])
                    else:
                        self.assertIsNone(e["plantilla"])
                        self.assertTrue(set(reglas) <= set(e["reglas"]), set(reglas) - set(e["reglas"]))

    def test_cuentas_del_inventario(self):
        vias = collections.Counter(v for _, _, _, v, _ in INVENTARIO_ESPERADO)
        self.assertEqual(vias, {"plantilla": 23, "reglas": 69})
        self.assertEqual(len({frozenset(g) for _, g, _, _, _ in INVENTARIO_ESPERADO}), 92)
        # Toda plantilla está en el inventario (en modo estricto todas dan frase)
        claves = {frozenset(g) for _, g, _, _, _ in INVENTARIO_ESPERADO}
        self.assertTrue(all(frozenset(g) in claves for _, g, _, _ in PLANTILLAS_ESPERADAS))

    def test_constante_inventario_del_modulo(self):
        self.assertEqual([(s["id"], list(s["glosas"]), s["frase"], s["via"]) for s in cg.INVENTARIO],
                         [(s, g, f, v) for s, g, f, v, _ in INVENTARIO_ESPERADO])

    def test_nueve_frases_de_ejemplo(self):
        """§10: las 9 frases usadas con el autor, en cualquier orden de entrada."""
        nuevas = {
            ("buenos días", "todos"): "Buenos días a todos.",
            ("buenas tardes", "todos"): "Buenas tardes a todos.",
            ("tío", "juicioso"): "El tío es juicioso.",
            ("familia", "feliz"): "La familia está feliz.",
            ("por favor", "seña"): "La seña, por favor.",
            ("hola", "bienvenido"): "Hola, bienvenido.",
            ("hermano", "trabajar"): "El hermano trabaja.",
            ("¿dónde?", "hermano", "vivir"): "¿Dónde vive el hermano?",
            ("hola", "¿cómo estás?"): "Hola, ¿cómo estás?",
        }
        for glosas, frase in nuevas.items():
            for perm in itertools.permutations(glosas):
                self.assertEqual(self.conv.translate(list(perm)), frase)


# ---------------------------------------------------------------------------
# 4. Un caso por regla R-01 a R-35
# ---------------------------------------------------------------------------

class TestUnCasoPorRegla(unittest.TestCase):
    # regla → (entrada, frase esperada, motivo esperado o None)
    CASOS = {
        "R-01": (["  HOLA "], "Hola.", None),
        "R-02": (["yo", "perro", "comer"], "yo, comer", "fuera_de_vocabulario"),
        "R-03": (["gracias", "gracias"], "Gracias.", None),
        "R-04": (["comer", "yo"], "Yo como.", None),
        "R-05": (["xyz"], "", None),
        "R-06": (["comer"], "Comer.", None),
        "R-07": (["por favor", "seña"], "La seña, por favor.", None),
        "R-08": (["hola", "yo", "trabajar", "mucho", "bien"], "hola, yo, trabajar, mucho, bien", "demasiadas_glosas"),
        "R-09": (["buenos días", "familia", "feliz"], "Buenos días, la familia está feliz.", None),
        "R-10": (["adiós", "todos"], "Adiós a todos.", None),
        "R-11": (["hola", "abuelo"], "Hola, abuelo.", None),
        "R-12": (["hola", "bienvenido"], "Hola, bienvenido.", None),
        "R-13": (["perdón", "yo", "vivir"], "perdón, yo, vivir", "cortesia_fuera_de_plantilla"),
        "R-14": (["hermano", "tío", "comer"], "hermano, tío, comer", "dos_sujetos"),
        "R-15": (["personas", "comer"], "Las personas comen.", None),
        "R-16": (["yo", "trabajar"], "Yo trabajo.", None),
        "R-17": (["familia", "contento"], "La familia está contenta.", None),
        "R-18": (["ustedes", "comer"], "Ustedes comen.", None),
        "R-19": (["hermano", "trabajar"], "El hermano trabaja.", None),
        "R-20": (["tío", "juicioso"], "El tío es juicioso.", None),
        "R-21": (["niña", "contento"], "La niña está contenta.", None),
        "R-22": (["yo", "bien"], "Yo estoy bien.", None),
        "R-23": (["yo", "sentir", "triste"], "Yo me siento triste.", None),
        "R-24": (["yo", "hambre"], "Yo tengo hambre.", None),
        "R-25": (["tío", "trabajar", "bien", "mucho"], "El tío trabaja muy bien.", None),
        "R-26": (["yo", "nunca", "comer"], "Yo nunca como.", None),
        "R-27": (["yo", "vivir"], "yo, vivir", "vivir_sin_complemento"),
        "R-28": (["yo", "nombre"], "yo, nombre", "sustantivo_abstracto_fuera_de_plantilla"),
        "R-29": (["¿qué?", "¿dónde?", "tú", "vivir"], "¿qué?, ¿dónde?, tú, vivir", "dos_interrogativos"),
        "R-30": (["¿por qué?", "ustedes", "trabajar"], "¿Por qué trabajan ustedes?", None),
        "R-31": (["¿quién?", "ustedes"], "¿Quiénes son ustedes?", None),
        "R-32": (["¿cómo estás?", "familia"], "¿Cómo está la familia?", None),
        "R-33": (["hola", "¿cómo estás?", "familia"], "Hola, ¿cómo está la familia?", None),
        "R-34": (["¿quién?", "comer"], "¿Quién come?", None),
        "R-35": (["hermano", "tío", "comer"], "hermano, tío, comer", "dos_sujetos"),
    }

    def test_cada_regla(self):
        conv = ConversorGlosas()
        self.assertEqual(sorted(self.CASOS), [f"R-{i:02d}" for i in range(1, 36)])
        for regla, (entrada, frase, motivo) in self.CASOS.items():
            with self.subTest(regla=regla):
                e = conv.explicar(entrada)
                self.assertEqual(e["frase"], frase)
                self.assertEqual(e["motivo"], motivo)
                self.assertIn(regla, e["reglas"])

    def test_suplido_y_supuestos_por_regla(self):
        conv = ConversorGlosas()
        casos = [   # entrada, códigos que deben estar en suplido, en supuestos
            (["personas", "comer"], {"articulo", "persona_numero", "tiempo_verbal"},
             {"articulo_definido", "tiempo_presente"}),
            (["tío", "juicioso"], {"copula", "concordancia_adjetivo"}, set()),
            (["yo", "hambre"], {"verbo_soporte"}, {"tiempo_presente"}),
            (["yo", "sentir", "triste"], {"pronombre_reflexivo"}, set()),
            (["hola", "todos"], {"preposicion", "puntuacion", "mayuscula", "orden_fijo"}, set()),
            (["yo", "mucho", "feliz"], {"intensificador"}, set()),
            (["tú", "comer"], set(), {"tratamiento_tu"}),
            (["yo", "contento"], set(), {"genero_por_defecto"}),
        ]
        for entrada, sup, supuestos in casos:
            e = conv.explicar(entrada)
            self.assertTrue(sup <= set(e["suplido"]), (entrada, e["suplido"]))
            self.assertTrue(supuestos <= set(e["supuestos"]), (entrada, e["supuestos"]))
            self.assertFalse(set(e["suplido"]) & set(e["supuestos"]))
            self.assertEqual(len(e["suplido"]), len(set(e["suplido"])))
        self.assertNotIn("genero_por_defecto", conv.explicar(["yo", "feliz"])["supuestos"])   # invariable


# ---------------------------------------------------------------------------
# 5. Motivos de respaldo (§5.2)
# ---------------------------------------------------------------------------

class TestMotivos(unittest.TestCase):

    def test_constante_motivos(self):
        self.assertEqual(list(cg.MOTIVOS), [m for m, _, _ in MOTIVOS_EJEMPLO] + ["fuera_de_inventario"])
        self.assertEqual(len(cg.MOTIVOS), 35)
        self.assertNotIn("no_previsto", cg.MOTIVOS)

    def test_ejemplos_de_los_34_motivos_alcanzables(self):
        conv = ConversorGlosas()
        for motivo, entrada, salida in MOTIVOS_EJEMPLO:
            with self.subTest(motivo=motivo):
                e = conv.explicar(entrada)
                self.assertEqual((e["frase"], e["motivo"], e["respaldo"], e["via"]),
                                 (salida, motivo, True, "respaldo"))
                self.assertEqual((e["suplido"], e["supuestos"]), ([], []))
                self.assertIn("R-35", e["reglas"])
                self.assertNotIn("R-04", e["reglas"])

    def test_fuera_de_inventario_solo_en_estricto(self):
        self.assertEqual(ConversorGlosas().translate(["¿quién?", "bienvenido"]), "¿Quién es bienvenido?")
        e = ConversorGlosas(estricto=True).explicar(["¿quién?", "bienvenido"])
        self.assertEqual((e["frase"], e["motivo"]), ("¿quién?, bienvenido", "fuera_de_inventario"))


# ---------------------------------------------------------------------------
# 6. Los 2.450 pares ordenados, exhaustivo (§9.1, prueba 3)
# ---------------------------------------------------------------------------

class TestPares(unittest.TestCase):

    def test_propiedades_de_forma(self):
        pares = barrido_pares()
        self.assertEqual(len(pares), 2450)
        for par, e in pares.items():
            self.assertEqual(problemas_de_forma(e), [], par)
            self.assertEqual(e["omitidas"], [])
            self.assertTrue(e["frase"])

    def test_recuentos_por_via_y_motivo(self):
        pares = barrido_pares()
        self.assertEqual(collections.Counter(e["via"] for e in pares.values()),
                         {"plantilla": 36, "reglas": 534, "respaldo": 1880})
        self.assertEqual(dict(collections.Counter(e["motivo"] for e in pares.values() if e["respaldo"])),
                         MOTIVOS_PARES)

    def test_invariancia_al_orden(self):
        pares = barrido_pares()
        cambia_frase = sum(1 for (a, b), e in pares.items()
                           if not e["respaldo"] and e["frase"] != pares[(b, a)]["frase"])
        cambia_respaldo = sum(1 for (a, b), e in pares.items() if e["respaldo"] != pares[(b, a)]["respaldo"])
        self.assertEqual((cambia_frase, cambia_respaldo), (0, 0))

    def test_respaldo_conserva_el_orden_de_entrada(self):
        conv = ConversorGlosas()
        self.assertEqual(conv.translate(["tío", "hermano", "comer"]), "tío, hermano, comer")
        self.assertEqual(conv.translate(["comer", "hermano", "tío"]), "comer, hermano, tío")

    def test_determinismo_y_sin_estado(self):
        """§9.1, prueba 7: dos pasadas, la segunda en orden inverso, sobre un conversor nuevo."""
        pares = barrido_pares()
        conv = ConversorGlosas()
        for par in reversed(list(pares)):
            e = conv.explicar(list(par))
            self.assertEqual((e["frase"], e["motivo"]), (pares[par]["frase"], pares[par]["motivo"]))


# ---------------------------------------------------------------------------
# 7. Conjuntos de 2 a 4 glosas, exhaustivo (§9.1, prueba 4), y muestra de tríos ordenados
# ---------------------------------------------------------------------------

class TestConjuntos(unittest.TestCase):

    def test_recuentos(self):
        res = barrido_conjuntos()
        self.assertEqual(len(res), 1225 + 19600 + 230300)   # 251.125
        por_tam = collections.defaultdict(collections.Counter)
        for g, e in res.items():
            por_tam[len(g)][e["via"]] += 1
        self.assertEqual([por_tam[n]["plantilla"] for n in (2, 3, 4)], [18, 4, 8])
        self.assertEqual([por_tam[n]["reglas"] for n in (2, 3, 4)], [267, 1753, 4945])
        con_frase = sum(1 for e in res.values() if not e["respaldo"])
        self.assertEqual(con_frase, 6995)
        self.assertEqual(sum(1 for e in res.values() if e["respaldo"]), 244130)

    def test_motivos_y_no_previsto(self):
        motivos = collections.Counter(e["motivo"] for e in barrido_conjuntos().values() if e["respaldo"])
        self.assertEqual(len(motivos), 32)
        self.assertNotIn("no_previsto", motivos)
        self.assertTrue(set(motivos) <= set(cg.MOTIVOS))
        # Los 34 alcanzables con estricto=False: 32 aquí + fuera_de_vocabulario + demasiadas_glosas
        self.assertEqual(set(motivos) | {"fuera_de_vocabulario", "demasiadas_glosas"}, set(cg.MOTIVOS[:34]))
        # Cuántas veces sale cada motivo (dato de regresión de 3a-2, ver MOTIVOS_CONJUNTOS)
        self.assertEqual(dict(motivos), MOTIVOS_CONJUNTOS)

    def test_propiedades_de_forma(self):
        for g, e in barrido_conjuntos().items():
            p = problemas_de_forma(e)
            if p:
                self.fail(f"{g}: {p} ({e['frase']!r})")

    def test_frases_distintas_y_coincidencias(self):
        por_frase = collections.defaultdict(list)
        for g, e in barrido_conjuntos().items():
            if not e["respaldo"]:
                por_frase[e["frase"]].append(set(g))
        self.assertEqual(len(por_frase), 6899)
        coincidencias = [c for c in por_frase.values() if len(c) > 1]
        self.assertEqual(sum(len(c) - 1 for c in coincidencias), 96)
        for c in coincidencias:   # todas son ¿CÓMO? + X frente a ¿CÓMO ESTÁS? + X (R-32)
            self.assertEqual(len(c), 2)
            a, b = c
            self.assertEqual(a ^ b, {"¿cómo?", "¿cómo estás?"})

    def test_permutaciones_de_los_conjuntos_con_frase(self):
        conv = ConversorGlosas()
        n = distintas = 0
        for g, e in barrido_conjuntos().items():
            if len(g) < 3 or e["respaldo"]:
                continue
            for perm in itertools.permutations(g):
                n += 1
                if conv.translate(list(perm)) != e["frase"]:
                    distintas += 1
        self.assertEqual((n, distintas), (129414, 0))

    def test_motivo_independiente_de_pythonhashseed(self):
        """§9.1, prueba 7: el barrido se repite en subprocesos con PYTHONHASHSEED 1 y 7."""
        def huella(resultados):
            texto = "\n".join(f"{'+'.join(g)}|{e['frase']}|{e['motivo']}" for g, e in resultados)
            return hashlib.sha256(texto.encode("utf-8")).hexdigest()
        local = huella(barrido_conjuntos().items())
        codigo = ("import hashlib, sys; sys.path.insert(0, '.');"
                  "from src.conversor_glosas import ConversorGlosas, conjuntos;"
                  "c = ConversorGlosas();"
                  "r = ((g, c.explicar(g)) for g in conjuntos());"
                  "t = '\\n'.join(f\"{'+'.join(g)}|{e['frase']}|{e['motivo']}\" for g, e in r);"
                  "print(hashlib.sha256(t.encode('utf-8')).hexdigest())")
        procesos = []
        for semilla in ("1", "7"):
            env = dict(os.environ, PYTHONHASHSEED=semilla)
            procesos.append(subprocess.Popen([PY, "-c", codigo], cwd=ROOT_DIR, env=env,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for p in procesos:
            out, err = p.communicate(timeout=300)
            self.assertEqual(p.returncode, 0, err)
            self.assertEqual(out.strip(), local)

    def test_muestra_determinista_de_trios_ordenados(self):
        """Tarea (d): la misma muestra que scripts/conversor_3a_salidas.py escribe en el CSV."""
        muestra = cg.muestra_trios()
        self.assertEqual(muestra, cg.muestra_trios())
        self.assertEqual(len(muestra), 1000)
        self.assertEqual(len({tuple(t) for t in muestra}), 1000)
        conv = ConversorGlosas()
        res = barrido_conjuntos()
        for trio in muestra:
            self.assertEqual(len(set(trio)), 3)
            e = conv.explicar(trio)
            self.assertEqual(problemas_de_forma(e), [], trio)
            clave = tuple(sorted(trio, key=list(SIGN_VOCABULARY.values()).index))
            if e["respaldo"]:
                self.assertEqual(e["motivo"], res[clave]["motivo"])
                self.assertEqual(e["frase"], ", ".join(trio))          # orden de entrada
            else:
                self.assertEqual(e["frase"], res[clave]["frase"])        # invariante al orden
            self.assertEqual(conv.explicar(trio), e)                     # determinismo


# ---------------------------------------------------------------------------
# 8. Concordancia por barrido (§9.1, prueba 5)
# ---------------------------------------------------------------------------

class TestConcordancia(unittest.TestCase):

    def setUp(self):
        self.conv = ConversorGlosas()

    def test_sujeto_por_verbo(self):
        for suj, (texto, pn, _, _) in SUJETOS.items():
            for verbo in ("trabajar", "comer"):
                self.assertEqual(self.conv.translate([suj, verbo]), f"{_cap(texto)} {VERBOS[verbo][pn]}.")
            self.assertEqual(self.conv.translate([suj, "vivir", "bien"]),
                             f"{_cap(texto)} {VERBOS['vivir'][pn]} bien.")

    def test_sujeto_por_adjetivo(self):
        for suj, (texto, pn, genero, numero) in SUJETOS.items():
            for adj, (copula, *_resto) in ADJETIVOS.items():
                e = self.conv.explicar([suj, adj])
                if (suj, adj) == ("yo", "bienvenido"):
                    self.assertEqual(e["motivo"], "yo_bienvenido")
                    continue
                if adj == "bienvenido" and suj in TestSegundaRevision.VOCATIVOS:   # R-20, fórmula (3a-2)
                    self.assertEqual(e["frase"], f"{_cap(_forma_adj(adj, genero, numero))}, {suj}.")
                    continue
                self.assertEqual(e["frase"],
                                 f"{_cap(texto)} {VERBOS[copula][pn]} {_forma_adj(adj, genero, numero)}.")

    def test_sujeto_sentir_adjetivo(self):
        for suj, (texto, pn, genero, numero) in SUJETOS.items():
            for adj, datos in ADJETIVOS.items():
                if not datos[5]:
                    self.assertEqual(self.conv.explicar([suj, "sentir", adj])["motivo"], "adjetivo_no_admite_sentir")
                    continue
                self.assertEqual(self.conv.translate([suj, "sentir", adj]),
                                 f"{_cap(texto)} {VERBOS['sentir'][pn]} {_forma_adj(adj, genero, numero)}.")

    def test_ninguna_salida_por_reglas_agramatical_conocida(self):
        singulares = {"trabaja", "come", "vive", "siente", "es", "está", "tiene"}
        plurales = {"ustedes", "todos", "personas"}
        for g, e in barrido_conjuntos().items():
            if e["via"] != "reglas":
                continue
            f = e["frase"]
            self.assertNotIn("la hambre", f.lower())
            self.assertNotIn("Personas come", f)
            if plurales & set(g):
                palabras = set(f.strip(".?¿").replace(",", " ").replace("¿", " ").replace("?", " ").split())
                self.assertFalse(palabras & singulares, f)


# ---------------------------------------------------------------------------
# 9. Bordes (§9.1, prueba 6)
# ---------------------------------------------------------------------------

class TestBordes(unittest.TestCase):

    def setUp(self):
        self.conv = ConversorGlosas()

    def _sin_excepcion(self, x):
        e = self.conv.explicar(x)
        self.assertEqual(e["frase"], self.conv.translate(x))
        self.assertEqual(list(e), CLAVES_EXPLICAR)
        return e

    def test_entradas_vacias_o_fuera_de_vocabulario(self):
        for x in ([], ["xyz"], [99], [-1], [None], [True], [1.5], [np.bool_(True)]):
            with self.subTest(entrada=repr(x)):
                e = self._sin_excepcion(x)
                self.assertEqual((e["frase"], e["via"]), ("", "vacio"))
        self.assertEqual(self.conv.translate([np.int64(7)]), "Comer.")
        self.assertEqual(self.conv.translate([7]), "Comer.")
        self.assertEqual(self.conv.translate((38, 7)), "Yo como.")            # tupla de ids
        self.assertEqual(self.conv.translate(iter(["yo", "comer"])), "Yo como.")   # generador

    def test_argumentos_que_no_son_lista(self):
        e = self._sin_excepcion("hola")
        self.assertEqual((e["frase"], e["avisos"]), ("Hola.", ["entrada_str_suelta"]))
        for x in (None, b"hola", bytearray(b"hola"), {}, {"hola": 1}, 5, object()):
            with self.subTest(argumento=repr(x)):
                e = self._sin_excepcion(x)
                self.assertEqual((e["frase"], e["via"], e["avisos"]), ("", "vacio", ["entrada_no_lista"]))

        def generador_roto():
            yield "yo"
            raise RuntimeError("se rompió")
        self.assertEqual(self._sin_excepcion(generador_roto())["avisos"], ["entrada_no_lista"])

    def test_repr_roto_y_muchos_rechazados(self):
        class ReprRoto:
            def __repr__(self):
                raise ValueError("roto")
        e = self._sin_excepcion([ReprRoto(), "comer"])
        self.assertEqual(e["fuera_de_vocabulario"], ["<repr falló: ReprRoto>"])
        self.assertEqual((e["frase"], e["motivo"]), ("comer", "fuera_de_vocabulario"))
        e = self._sin_excepcion(["x" * 100 + str(i) for i in range(1000)])
        self.assertEqual(e["n_fuera_de_vocabulario"], 1000)
        self.assertEqual(len(e["fuera_de_vocabulario"]), 10)
        self.assertTrue(all(len(r) <= 40 for r in e["fuera_de_vocabulario"]))
        self.assertTrue(e["fuera_de_vocabulario"][0].endswith("…"))
        self.assertEqual(e["frase"], "")

    def test_fuera_de_vocabulario_con_glosa_valida(self):
        e = self._sin_excepcion(["xyz", "comer"])
        self.assertEqual((e["frase"], e["respaldo"], e["motivo"]), ("comer", True, "fuera_de_vocabulario"))
        self.assertEqual(e["fuera_de_vocabulario"], ["'xyz'"])

    def test_mayusculas_espacios_y_alias(self):
        self.assertEqual(self.conv.translate(["  HOLA  "]), "Hola.")
        self.assertEqual(self.conv.translate(["Buenos   DÍAS", "Todos"]), "Buenos días a todos.")
        self.assertEqual(self.conv.translate(["tu", "feliz"]), "Tú estás feliz.")
        alias = {"buenos dias": "buenos días", "tio": "tío", "como estas": "¿cómo estás?",
                 "mas o menos": "más o menos", "adios": "adiós", "perdon": "perdón", "tu": "tú",
                 "que": "¿qué?", "cuando": "¿cuándo?", "donde": "¿dónde?", "como": "¿cómo?",
                 "por que": "¿por qué?", "quien": "¿quién?"}
        self.assertEqual(cg.ALIAS, alias)
        for a, g in alias.items():
            e = self.conv.explicar([a.upper()])
            self.assertEqual((e["glosas"], e["alias"], e["frase"]), ([g], [a], self.conv.translate([g])))
        # NFC: «ñ» y «í» descompuestas también valen
        import unicodedata
        self.assertEqual(self.conv.translate([unicodedata.normalize("NFD", "niña")]), "Niña.")

    def test_demasiadas_glosas_y_omitidas(self):
        e = self._sin_excepcion(["hola", "yo", "trabajar", "mucho", "bien"])
        self.assertEqual((e["motivo"], e["omitidas"]), ("demasiadas_glosas", []))
        diez = [SIGN_VOCABULARY[i] for i in range(10)]
        e = self._sin_excepcion(diez)
        self.assertEqual(e["frase"], ", ".join(diez[:8]))
        self.assertEqual(e["omitidas"], diez[8:])
        e = self._sin_excepcion(["comer"] * 1000)
        self.assertEqual((e["frase"], e["repeticiones"]), ("Comer.", 999))

    def test_modo_estricto(self):
        estricto = ConversorGlosas(estricto=True)
        for g in SIGN_VOCABULARY.values():
            self.assertEqual(estricto.translate([g]), self.conv.translate([g]))
        for s, glosas, frase, _, _ in INVENTARIO_ESPERADO:
            self.assertEqual(estricto.translate(glosas), frase, s)
        self.assertEqual(estricto.explicar(["hermano", "tío", "comer"])["motivo"], "dos_sujetos")
        self.assertEqual(estricto.explicar(["buenos días", "tú", "nombre", "¿qué?"])["motivo"],
                         "fuera_de_inventario")
        inventario = {frozenset(g) for _, g, _, _, _ in INVENTARIO_ESPERADO}
        normal = barrido_conjuntos()
        for g, e in barrido_conjuntos(estricto=True).items():
            if not e["respaldo"]:
                self.assertIn(frozenset(g), inventario, g)
            elif not normal[g]["respaldo"]:
                self.assertEqual(e["motivo"], "fuera_de_inventario", g)
            else:
                self.assertEqual(e["motivo"], normal[g]["motivo"], g)
        self.assertEqual(estricto.explicar(["yo"])["parametros"]["estricto"], True)

    def test_constructor(self):
        for kw in ({"genero_primera": "x"}, {"genero_segunda": "F"}, {"genero_primera": 1},
                   {"estricto": "sí"}, {"estricto": 1}):
            with self.assertRaises(ValueError, msg=kw):
                ConversorGlosas(**kw)
        f = ConversorGlosas(genero_primera="f")
        e = f.explicar(["yo", "contento"])
        self.assertEqual(e["frase"], "Yo estoy contenta.")
        self.assertNotIn("genero_por_defecto", e["supuestos"])
        self.assertEqual(e["parametros"], {"genero_primera": "f", "genero_segunda": "m", "estricto": False})
        e = ConversorGlosas(genero_primera="m").explicar(["yo", "contento"])   # explícito: no se registra
        self.assertNotIn("genero_por_defecto", e["supuestos"])
        self.assertIn("genero_por_defecto", self.conv.explicar(["yo", "contento"])["supuestos"])
        self.assertEqual(ConversorGlosas(genero_segunda="f").translate(["tú", "aburrido"]), "Tú estás aburrida.")

    def test_tipos_de_la_traza(self):
        tipos = {"frase": str, "via": str, "reglas": list, "suplido": list, "supuestos": list, "respaldo": bool,
                 "glosas": list, "fuera_de_vocabulario": list, "n_fuera_de_vocabulario": int, "alias": list,
                 "repeticiones": int, "omitidas": list, "avisos": list, "parametros": dict, "version": str}
        for x in ([], ["comer"], ["tú", "comer"], ["por favor", "seña"], ["hermano", "tío", "comer"], None):
            e = self.conv.explicar(x)
            self.assertEqual(list(e), CLAVES_EXPLICAR)
            for k, tipo in tipos.items():
                self.assertIsInstance(e[k], tipo, (x, k))
            self.assertTrue(e["plantilla"] is None or isinstance(e["plantilla"], str))
            self.assertTrue(e["motivo"] is None or isinstance(e["motivo"], str))
            self.assertEqual(e["version"], "3a-2")
            self.assertIn(e["via"], ("vacio", "lexico", "plantilla", "reglas", "respaldo"))
            self.assertEqual(e["respaldo"], e["via"] == "respaldo")
            if e["via"] in ("vacio", "respaldo"):
                self.assertEqual((e["suplido"], e["supuestos"]), ([], []))
        self.assertEqual(ConversorGlosas.VERSION, "3a-2")


# ---------------------------------------------------------------------------
# 9b. Precedencia normativa de §3.1: cuando una entrada de 3 o 4 glosas choca con dos comprobaciones,
#     el motivo es el de la primera en el orden de §3.1 (segunda revisión del 8-nov, hallazgo 6)
# ---------------------------------------------------------------------------

class TestPrecedencia(unittest.TestCase):
    # (entrada, motivo esperado, motivo que daría el orden contrario, par de comprobaciones de §3.1)
    CASOS = [
        (["gracias", "seña", "yo"], "sustantivo_abstracto_fuera_de_plantilla", "cortesia_fuera_de_plantilla", "C1 > C2"),
        (["gracias", "hermano", "tío"], "cortesia_fuera_de_plantilla", "dos_sujetos", "C2 > C3"),
        (["¿qué?", "¿dónde?", "yo", "tú"], "dos_sujetos", "dos_interrogativos", "C3 > C4"),
        (["¿cómo estás?", "¿quién?", "comer"], "dos_interrogativos", "como_estas_no_admitido", "C4 > D"),
        (["¿quién?", "yo", "comer"], "quien_con_sujeto_y_predicado", "quien_sujeto_no_admitido", "D2: resto > rasgo quien"),
        (["¿quién?", "todos", "feliz"], "quien_con_sujeto_y_predicado", "quien_sujeto_no_admitido", "D2: resto > rasgo quien"),
        (["¿dónde?", "yo", "comer", "mucho"], "pregunta_no_admitida", "nunca_con_cantidad", "D3 > P"),
        (["yo", "mucho", "poco", "nunca"], "dos_ADV_CANT", "nunca_con_cantidad", "P1 > P2"),
        (["todos", "nunca", "mucho", "comer"], "nunca_con_cantidad", "todos_con_nunca", "P2 > P3"),
        (["todos", "nunca", "trabajar", "bien"], "todos_con_nunca", "nunca_con_modo", "P3 > P4"),
        (["yo", "nunca", "trabajar", "feliz"], "nunca_con_modo", "verbo_con_predicado", "P4 > P7"),
        (["yo", "feliz", "bien", "sentir"], "dos_predicados", "frase por P8, que no mira dos predicados", "P5 > P8"),
        (["yo", "sentir", "comer", "feliz"], "dos_verbos", "verbo_con_predicado", "P6 > P7"),
        (["yo", "nunca", "vivir", "mucho"], "nunca_con_cantidad", "nunca_no_admitido", "P2 > P7"),
        (["yo", "nunca", "bienvenido"], "yo_bienvenido", "nunca_no_admitido", "P9: yo_bienvenido > nunca_no_admitido"),
        (["yo", "poco", "bienvenido"], "yo_bienvenido", "cantidad_no_admitida", "P9: yo_bienvenido > cantidad"),
        (["hola", "adiós", "buenos días"], "saludos_no_combinables", "despedida_con_contenido", "B1 > B5"),
        (["adiós", "tú", "nombre", "¿qué?"], "despedida_con_contenido", "plantilla T-03", "B5 > B6"),
    ]

    def test_cada_par_de_comprobaciones(self):
        conv = ConversorGlosas()
        for entrada, motivo, contrario, par in self.CASOS:
            with self.subTest(par=par, entrada=entrada):
                self.assertNotEqual(motivo, contrario)
                for perm in itertools.permutations(entrada):
                    self.assertEqual(conv.explicar(list(perm))["motivo"], motivo)


# ---------------------------------------------------------------------------
# 9c. Segunda revisión del 8-nov (versión 3a-2): reglas nuevas, traza y entrada
# ---------------------------------------------------------------------------

class TestSegundaRevision(unittest.TestCase):

    def setUp(self):
        self.conv = ConversorGlosas()

    VOCATIVOS = {"familia": ("la familia", "f"), "niño": ("el niño", "m"), "niña": ("la niña", "f"),
                 "abuelo": ("el abuelo", "m"), "tío": ("el tío", "m"), "hermano": ("el hermano", "m")}

    def test_como_estas_con_vocativo_registra_la_lectura(self):
        """R-32: se mantiene la 3.ª persona y se registra «lectura_tercera_persona» (B5, §9.2)."""
        for suj, (texto, _) in self.VOCATIVOS.items():
            e = self.conv.explicar(["¿cómo estás?", suj])
            self.assertEqual(e["frase"], f"¿Cómo está {texto}?")
            self.assertIn("lectura_tercera_persona", e["supuestos"])
            self.assertNotIn("copula", e["suplido"])          # R-32: «estar» viene de la glosa, no se suple
        for suj in ("mujer", "hombre", "personas", "tú", "ustedes", "todos"):
            self.assertNotIn("lectura_tercera_persona", self.conv.explicar(["¿cómo estás?", suj])["supuestos"])
        e = self.conv.explicar(["tío", "¿cómo estás?", "buenas noches"])
        self.assertEqual(e["frase"], "Buenas noches, ¿cómo está el tío?")
        self.assertIn("lectura_tercera_persona", e["supuestos"])
        self.assertIn("copula", self.conv.explicar(["¿cómo?", "familia"])["suplido"])   # R-30 sí la suple

    def test_bienvenido_con_vocativo_es_formula(self):
        """R-20, excepción «fórmula»: vocativo + BIENVENIDO → «Bienvenido, x.», sin artículo ni cópula."""
        for suj, (_, genero) in self.VOCATIVOS.items():
            e = self.conv.explicar([suj, "bienvenido"])
            forma = "bienvenida" if genero == "f" else "bienvenido"
            self.assertEqual(e["frase"], f"{_cap(forma)}, {suj}.")
            self.assertIn("lectura_vocativa", e["supuestos"])
            self.assertNotIn("copula", e["suplido"])
            self.assertTrue({"R-11", "R-12", "R-20", "R-21"} <= set(e["reglas"]))
        self.assertEqual(self.conv.translate(["hola", "familia", "bienvenido"]), "Hola, bienvenida, familia.")
        # Se conserva la cópula fuera del vocativo
        self.assertEqual(self.conv.translate(["mujer", "bienvenido"]), "La mujer es bienvenida.")
        self.assertEqual(self.conv.translate(["tú", "bienvenido"]), "Tú eres bienvenido.")
        self.assertEqual(self.conv.translate(["todos", "bienvenido"]), "Todos son bienvenidos.")
        self.assertEqual(self.conv.translate(["tío", "bienvenido", "mucho"]), "El tío es muy bienvenido.")
        self.assertEqual(self.conv.translate(["¿quién?", "bienvenido"]), "¿Quién es bienvenido?")

    def test_gracias_con_vocativo(self):
        """R-11 extendida (3a-2): GRACIAS + vocativo, sin saludo → «Gracias, x.»."""
        for suj in self.VOCATIVOS:
            e = self.conv.explicar([suj, "gracias"])
            self.assertEqual((e["frase"], e["via"]), (f"Gracias, {suj}.", "reglas"))
            self.assertTrue({"R-11", "R-13"} <= set(e["reglas"]))
        for entrada in (["gracias", "mujer"], ["gracias", "yo"], ["hola", "gracias", "abuelo"],
                        ["gracias", "abuelo", "tío"]):
            self.assertEqual(self.conv.explicar(entrada)["motivo"], "cortesia_fuera_de_plantilla", entrada)

    def test_bienvenido_tras_saludo_concuerda_con_quien_escucha(self):
        e = self.conv.explicar(["hola", "bienvenido"])
        self.assertEqual(e["frase"], "Hola, bienvenido.")
        self.assertIn("genero_por_defecto", e["supuestos"])
        e = ConversorGlosas(genero_segunda="f").explicar(["buenos días", "bienvenido"])
        self.assertEqual(e["frase"], "Buenos días, bienvenida.")
        self.assertNotIn("genero_por_defecto", e["supuestos"])
        self.assertNotIn("genero_por_defecto", ConversorGlosas(genero_segunda="m").explicar(
            ["hola", "bienvenido"])["supuestos"])
        self.assertEqual(self.conv.explicar(["contento"])["supuestos"], [])   # vía lexico: forma de cita (R-17)

    def test_registros_de_r17_con_tu_y_quien(self):
        """Hallazgo 7: genero_por_defecto con TÚ; ¿QUIÉN? en masculino sin registrarlo; copula en «¿Quién es…?»."""
        self.assertEqual(self.conv.explicar(["tú", "contento"])["supuestos"],
                         ["tratamiento_tu", "tiempo_presente", "genero_por_defecto"])
        self.assertNotIn("genero_por_defecto",
                         ConversorGlosas(genero_segunda="m").explicar(["tú", "contento"])["supuestos"])
        e = self.conv.explicar(["¿quién?", "contento"])
        self.assertEqual(e["frase"], "¿Quién está contento?")
        self.assertNotIn("genero_por_defecto", e["supuestos"])
        self.assertIn("copula", self.conv.explicar(["¿quién?", "tú"])["suplido"])
        self.assertIn("copula", self.conv.explicar(["¿quién?", "ustedes"])["suplido"])

    def test_set_y_frozenset_en_orden_de_id(self):
        e = self.conv.explicar(frozenset({"hermano", "tío", "comer"}))
        self.assertEqual((e["frase"], e["avisos"]), ("comer, tío, hermano", ["entrada_sin_orden"]))
        self.assertEqual(self.conv.translate({"comer", "yo"}), "Yo como.")
        e = self.conv.explicar({"xyz", "comer"})
        self.assertEqual((e["frase"], e["motivo"]), ("comer", "fuera_de_vocabulario"))

    def test_iterable_infinito_y_entrada_truncada(self):
        e = self.conv.explicar(itertools.count())          # sin tope no volvería nunca
        self.assertIn("entrada_truncada", e["avisos"])
        self.assertEqual(e["frase"], ", ".join(SIGN_VOCABULARY[i] for i in range(8)))
        e = self.conv.explicar(itertools.repeat("comer"))
        self.assertEqual((e["frase"], e["avisos"], e["repeticiones"]),
                         ("Comer.", ["entrada_truncada"], cg.MAX_ELEMENTOS_ENTRADA - 1))
        self.assertEqual(self.conv.explicar(["comer"] * cg.MAX_ELEMENTOS_ENTRADA)["avisos"], [])

    def test_muestra_b3_estratificada(self):
        b3 = cg.muestra_b3()
        self.assertEqual(b3, cg.muestra_b3())
        self.assertEqual(len(b3), 30)
        self.assertEqual(len(set(b3)), 30)
        estratos = collections.Counter((len(g), any(cg.LEXICO[x]["cat"] == cg.SAL for x in g)) for g in b3)
        self.assertEqual(estratos, {(n, s): 5 for n in (2, 3, 4) for s in (False, True)})
        inventario = {frozenset(s["glosas"]) for s in cg.INVENTARIO}
        for g in b3:
            self.assertNotIn(frozenset(g), inventario)
            self.assertFalse(self.conv.explicar(list(g))["respaldo"])

    def test_orden_de_presentacion(self):
        self.assertEqual(cg.orden_presentacion(["buenas noches", "niña", "juicioso", "hola"]),
                         ["hola", "buenas noches", "niña", "juicioso"])
        self.assertEqual(cg.orden_presentacion(["triste", "hola", "todos", "mucho"]),
                         ["hola", "todos", "triste", "mucho"])
        self.assertEqual(cg.orden_presentacion(["vivir", "hermano", "¿dónde?"]), ["¿dónde?", "hermano", "vivir"])
        self.assertEqual(sorted(cg.ORDEN_PRESENTACION), sorted(cg.CATEGORIAS))

    def test_tabla_de_salidas(self):
        filas = cg.tabla_salidas()
        self.assertEqual(len(filas), 50 + 6995)
        self.assertEqual(list(filas[0]), cg.COLUMNAS_TABLA)
        self.assertNotIn("revisado", cg.COLUMNAS_TABLA)
        self.assertTrue(all(f["estado_revision"] == "pendiente" for f in filas))
        bloques = collections.Counter(f["bloque_actividad_B"] for f in filas)
        self.assertEqual((bloques["B1"], bloques["B3"], bloques[""]), (50, 30, 6995 - 92 - 30))
        self.assertEqual(sum(n for b, n in bloques.items() if b.startswith("S-")), 92)
        self.assertEqual(collections.Counter(f["via"] for f in filas),
                         {"lexico": 50, "plantilla": 30, "reglas": 6965})


# ---------------------------------------------------------------------------
# 10. Rendimiento (§9.1, prueba 8)
# ---------------------------------------------------------------------------

class TestRendimiento(unittest.TestCase):

    def test_peor_par_menos_de_1_ms(self):
        """Por par se toma el mínimo de 3 llamadas, para que un cambio de contexto del sistema no se
        cuente como lentitud del conversor (decisión de la implementación)."""
        conv = ConversorGlosas()
        peor = 0.0
        for a, b in itertools.permutations(SIGN_VOCABULARY.values(), 2):
            tiempos = []
            for _ in range(3):
                t0 = time.perf_counter()
                conv.translate([a, b])
                tiempos.append(time.perf_counter() - t0)
            peor = max(peor, min(tiempos))
        self.assertLess(peor, 1e-3)


# ---------------------------------------------------------------------------
# 11. No regresión del punto de partida (§9.1, prueba 9)
# ---------------------------------------------------------------------------

class TestNoRegresion(unittest.TestCase):

    def test_md5_de_grammar_mapper(self):
        md5 = hashlib.md5((ROOT_DIR / "src" / "grammar_mapper.py").read_bytes()).hexdigest()
        self.assertEqual(md5, MD5_GRAMMAR_MAPPER)

    def test_pruebas_del_heredado_siguen_siendo_89(self):
        n = unittest.TestLoader().loadTestsFromName("tests.test_grammar_mapper").countTestCases()
        self.assertEqual(n, 89)

    def test_no_usa_grammar_categories_ni_grammar_mapper(self):
        arbol = ast.parse((ROOT_DIR / "src" / "conversor_glosas.py").read_text(encoding="utf-8"))
        nombres = set()
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Name):
                nombres.add(nodo.id)
            elif isinstance(nodo, ast.Attribute):
                nombres.add(nodo.attr)
            elif isinstance(nodo, (ast.Import, ast.ImportFrom)):
                nombres.update(a.asname or a.name for a in nodo.names)
        self.assertFalse(nombres & {"GRAMMAR_CATEGORIES", "GrammarMapper", "CONJUGATIONS"})
        self.assertIs(cg.SIGN_VOCABULARY, SIGN_VOCABULARY)   # se importa, no se redefine

    def test_comprobacion_del_lexico_con_python_O(self):
        ok = subprocess.run([PY, "-O", "-c", "import src.conversor_glosas"], cwd=ROOT_DIR,
                            capture_output=True, text=True, timeout=120)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        roto = subprocess.run(
            [PY, "-O", "-c", "import src.grammar_mapper as g; g.SIGN_VOCABULARY[50] = 'perro'; "
                             "import src.conversor_glosas"],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(roto.returncode, 0)
        self.assertIn("RuntimeError", roto.stderr)

    def test_translator_no_importa_el_conversor_nuevo_por_defecto(self):
        """Hallazgo 8: el heredado no depende de que cargue src.conversor_glosas."""
        r = subprocess.run([PY, "-c", "import sys; import src.translator as t; "
                                      "print('src.conversor_glosas' in sys.modules, t.construir_conversor() is None)"],
                           cwd=ROOT_DIR, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "False True")
        # Con el módulo nuevo roto al importar, el prototipo con el heredado sigue cargando
        r = subprocess.run([PY, "-c", "import src.grammar_mapper as g; g.SIGN_VOCABULARY[50] = 'perro'; "
                                      "import src.translator as t; print(t.construir_conversor('heredado'))"],
                           cwd=ROOT_DIR, capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "None")

    def test_main_imprime_el_inventario(self):
        r = subprocess.run([PY, "-m", "src.conversor_glosas"], cwd=ROOT_DIR, capture_output=True,
                           text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("92/92 secuencias dan su enunciado fijo", r.stdout)


# ---------------------------------------------------------------------------
# 12. Red de seguridad (§9.1, prueba 10): cada mutación de una tabla hace fallar alguna verificación
# ---------------------------------------------------------------------------

def verificaciones() -> list[str]:
    """Las comprobaciones de salida exacta de este archivo, como lista de fallos (vacía = todo bien)."""
    conv = ConversorGlosas()
    fallos = []
    for i, esperada in SOLAS.items():
        if conv.translate([i]) != esperada:
            fallos.append(f"sola {i}")
    for s, glosas, frase, _, _ in INVENTARIO_ESPERADO:
        if conv.translate(glosas) != frase:
            fallos.append(s)
    for tid, glosas, frase, _ in PLANTILLAS_ESPERADAS:
        if conv.translate(glosas) != frase:
            fallos.append(tid)
    for suj, (texto, pn, genero, numero) in SUJETOS.items():
        for verbo in ("trabajar", "comer"):
            if conv.translate([suj, verbo]) != f"{_cap(texto)} {VERBOS[verbo][pn]}.":
                fallos.append(f"{suj}+{verbo}")
        for adj, (copula, *_r) in ADJETIVOS.items():
            if (suj, adj) == ("yo", "bienvenido"):
                continue
            if adj == "bienvenido" and suj in TestSegundaRevision.VOCATIVOS:   # R-20, fórmula (3a-2)
                esperada = f"{_cap(_forma_adj(adj, genero, numero))}, {suj}."
            else:
                esperada = f"{_cap(texto)} {VERBOS[copula][pn]} {_forma_adj(adj, genero, numero)}."
            if conv.translate([suj, adj]) != esperada:
                fallos.append(f"{suj}+{adj}")
    return fallos


class TestRedDeSeguridad(unittest.TestCase):

    def test_sin_mutar_no_falla_nada(self):
        self.assertEqual(verificaciones(), [])

    def test_cada_mutacion_se_detecta(self):
        mutaciones = {
            "cópula de juicioso": mock.patch.dict(cg.LEXICO["juicioso"], {"copula": "estar"}),
            "cópula de feliz": mock.patch.dict(cg.LEXICO["feliz"], {"copula": "ser"}),
            "género de niña": mock.patch.dict(cg.LEXICO["niña"], {"genero": "m"}),
            "forma verbal comen": mock.patch.dict(cg.FORMAS_VERBALES["comer"], {"3p": "come"}),
            "forma de contenta": mock.patch.dict(cg.FORMAS_ADJETIVO["contento"], {"fs": "contento"}),
            "artículo femenino": mock.patch.dict(cg.ARTICULOS, {"fs": "el"}),
            "plantilla T-05": mock.patch.dict(cg.PLANTILLAS["T-05"], {"frase": "Mucho gracias."}),
            "orden del sujeto en R-30": mock.patch.object(cg, "ORDEN_PREGUNTA_PARCIAL",
                                                          ("interrogativo", "sujeto", "predicado")),
        }
        for nombre, parche in mutaciones.items():
            with self.subTest(mutacion=nombre):
                with parche:
                    self.assertNotEqual(verificaciones(), [], f"la mutación «{nombre}» pasa inadvertida")
        self.assertEqual(verificaciones(), [])   # los parches se deshicieron


if __name__ == "__main__":
    unittest.main()
