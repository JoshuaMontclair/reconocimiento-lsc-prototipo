"""
Conversor de glosas LSC50 a enunciados en español por reglas explícitas y plantillas (actividad 3.a).

QUÉ ES
  Implementación de docs/CONVERSOR_3a_ESPECIFICACION.md (8-nov-2026, «la especificación»). Cada
  regla del código lleva su número R-xx; las tablas de este módulo (LEXICO, FORMAS_VERBALES,
  FORMAS_ADJETIVO, ARTICULOS, PLANTILLAS, PREGUNTAS_ADMITIDAS, INVENTARIO, MOTIVOS, ALIAS) son las
  de la especificación, escritas como constantes legibles. Si una tabla y la especificación no
  coinciden, manda la especificación y hay que corregir una de las dos de forma explícita.

  - Una sola glosa distinta: su forma de cita, con mayúscula y punto o «?» (D-L1, R-06).
  - Un conjunto que es clave de una plantilla: su enunciado fijo (R-07, T-01 a T-22).
  - Un conjunto que encaja en un patrón admitido (§3.1 de la especificación): un enunciado armado
    por reglas (saludo, sujeto, cópula, concordancia, pregunta parcial...).
  - Todo lo demás: RESPALDO, las glosas en minúscula separadas por comas, marcado como tal y con
    su motivo (R-35, §5.2). Nunca una frase agramatical presentada como frase.
  - Sin glosas del vocabulario: "" (R-05).

QUÉ NO ES
  No es traducción abierta: toda salida que no es respaldo pertenece a un conjunto finito que
  se puede listar (`python -m src.conversor_glosas --exportar RUTA.csv`). No usa modelos de
  lenguaje, redes, estadística ni corpus: son tablas y condiciones escritas a mano. No infiere
  funciones del orden de entrada (R-04): la frase depende solo del CONJUNTO de glosas.

ESTADO DE VALIDACIÓN
  NINGUNA regla ni enunciado de este módulo ha sido revisado por la persona señante vinculada
  (profesor de LSC vinculado, Actividad B del consentimiento). Las reglas marcadas «validar: sí» en la
  especificación son decisiones de diseño del autor a la espera de esa revisión.

RELACIÓN CON EL CONVERSOR HEREDADO
  src/grammar_mapper.py (GrammarMapper) no se toca y sigue siendo el conversor por defecto del
  prototipo. De él solo se importa SIGN_VOCABULARY; no se usan GRAMMAR_CATEGORIES ni GrammarMapper.
  La interfaz translate(signs) -> str es la misma, así que el prototipo puede inyectar este
  conversor como grammar_mapper (src/realtime.py --conversor reglas | inventario).

Uso:
    from src.conversor_glosas import ConversorGlosas

    conv = ConversorGlosas()
    conv.translate(["tío", "juicioso"])              # → "El tío es juicioso."
    conv.translate(["¿dónde?", "hermano", "vivir"])  # → "¿Dónde vive el hermano?"
    conv.translate([38, 7])                          # → "Yo como."  (38 = yo, 7 = comer)
    conv.translate(["hermano", "tío", "comer"])      # → "hermano, tío, comer"  (respaldo)
    conv.explicar(["yo", "contento"])["supuestos"]   # → ["tiempo_presente", "genero_por_defecto"]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import numbers
import random
import sys
import unicodedata
from pathlib import Path

if __package__ in (None, ""):   # ejecución directa: python src/conversor_glosas.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402  (se importa, no se redefine)


VERSION = "3a-2"   # cambia con cualquier cambio de tabla o regla (§6, §9.2). 3a-2: segunda revisión del 8-nov


# ---------------------------------------------------------------------------
# Categorías (§2.1). Una categoría es el conjunto de glosas que cumplen la misma función en las
# plantillas de núcleo; las diferencias dentro de una categoría van como rasgos (§2.3).
# ---------------------------------------------------------------------------

SAL = "SAL"                    # saludo
CORT = "CORT"                  # cortesía
PREG_FIJA = "PREG_FIJA"        # pregunta fija (¿cómo estás?)
INTERR = "INTERR"              # interrogativo
PRON = "PRON"                  # pronombre personal
CUANT_PRON = "CUANT_PRON"      # pronombre cuantificador (todos)
NOM_PERS = "NOM_PERS"          # sustantivo de persona
NOM_ABS = "NOM_ABS"            # sustantivo abstracto (seña, nombre)
NOM_TENER = "NOM_TENER"        # sustantivo de estado con «tener» (hambre)
VERBO = "VERBO"                # verbo
VERBO_SENTIR = "VERBO_SENTIR"  # verbo pronominal (sentir)
ADJ = "ADJ"                    # adjetivo
ADV_EST = "ADV_EST"            # adverbio de estado
ADV_CANT = "ADV_CANT"          # cuantificador (mucho, poco)
ADV_FREC = "ADV_FREC"          # frecuencia (nunca)

CATEGORIAS = (SAL, CORT, PREG_FIJA, INTERR, PRON, CUANT_PRON, NOM_PERS, NOM_ABS, NOM_TENER,
              VERBO, VERBO_SENTIR, ADJ, ADV_EST, ADV_CANT, ADV_FREC)

# SUJ = PRON ∪ CUANT_PRON ∪ NOM_PERS (13 glosas, §2.1)
CATEGORIAS_SUJETO = (PRON, CUANT_PRON, NOM_PERS)


# ---------------------------------------------------------------------------
# Léxico (§2.3): glosa de SIGN_VOCABULARY → categoría, forma de cita, salida sola y rasgos.
#   pn       persona y número para las tablas de formas: "1s", "2s", "3s", "3p"
#   genero   "m" / "f"; en YO y TÚ es None: sale de genero_primera / genero_segunda (R-17)
#   numero   "sg" / "pl"
#   vocativo NOM_PERS que puede ir tras un saludo como vocativo (R-11)
#   quien    SUJ admitido en «¿Quién es…?» (R-31)
#   cant, modo, nunca, exige_complemento   rasgos de VERBO (R-25, R-22, R-26, R-27)
#   copula, con_sentir                      rasgos de ADJ (R-20, R-23)
#   muy      ADV_EST que admite «muy» (R-25)
#   despedida  ADIÓS (R-09)
# La columna «sola» es la salida exacta para la glosa sola (D-L1, R-06).
# ---------------------------------------------------------------------------

LEXICO: dict[str, dict] = {
    # --- SAL (R-09 a R-12) ---
    "hola":          {"cat": SAL, "forma": "hola", "sola": "Hola."},
    "buenos días":   {"cat": SAL, "forma": "buenos días", "sola": "Buenos días."},
    "buenas tardes": {"cat": SAL, "forma": "buenas tardes", "sola": "Buenas tardes."},
    "buenas noches": {"cat": SAL, "forma": "buenas noches", "sola": "Buenas noches."},
    "adiós":         {"cat": SAL, "forma": "adiós", "sola": "Adiós.", "despedida": True},
    # --- CORT: solo por plantilla (R-13). «permiso» era «objeto» en el heredado (§2.1) ---
    "gracias":   {"cat": CORT, "forma": "gracias", "sola": "Gracias."},
    "por favor": {"cat": CORT, "forma": "por favor", "sola": "Por favor."},
    "con gusto": {"cat": CORT, "forma": "con gusto", "sola": "Con gusto."},
    "perdón":    {"cat": CORT, "forma": "perdón", "sola": "Perdón."},
    "permiso":   {"cat": CORT, "forma": "permiso", "sola": "Permiso."},
    # --- PREG_FIJA (R-32) ---
    "¿cómo estás?": {"cat": PREG_FIJA, "forma": "¿cómo estás?", "sola": "¿Cómo estás?"},
    # --- INTERR (R-29 a R-31) ---
    "¿qué?":     {"cat": INTERR, "forma": "qué", "sola": "¿Qué?"},
    "¿cuándo?":  {"cat": INTERR, "forma": "cuándo", "sola": "¿Cuándo?"},
    "¿dónde?":   {"cat": INTERR, "forma": "dónde", "sola": "¿Dónde?"},
    "¿cómo?":    {"cat": INTERR, "forma": "cómo", "sola": "¿Cómo?"},
    "¿por qué?": {"cat": INTERR, "forma": "por qué", "sola": "¿Por qué?"},
    "¿quién?":   {"cat": INTERR, "forma": "quién", "sola": "¿Quién?"},
    # --- PRON (R-16, R-17) ---
    "yo":      {"cat": PRON, "forma": "yo", "sola": "Yo.", "pn": "1s", "genero": None, "numero": "sg",
                "quien": False},
    "tú":      {"cat": PRON, "forma": "tú", "sola": "Tú.", "pn": "2s", "genero": None, "numero": "sg",
                "quien": True},
    "ustedes": {"cat": PRON, "forma": "ustedes", "sola": "Ustedes.", "pn": "3p", "genero": "m",
                "numero": "pl", "quien": True},
    # --- CUANT_PRON (R-10, R-26) ---
    "todos": {"cat": CUANT_PRON, "forma": "todos", "sola": "Todos.", "pn": "3p", "genero": "m",
              "numero": "pl", "quien": False},
    # --- NOM_PERS (R-11, R-15, R-17, R-31) ---
    "familia":  {"cat": NOM_PERS, "forma": "familia", "sola": "Familia.", "pn": "3s", "genero": "f",
                 "numero": "sg", "vocativo": True, "quien": False},   # colectivo en singular (R-17)
    "personas": {"cat": NOM_PERS, "forma": "personas", "sola": "Personas.", "pn": "3p", "genero": "f",
                 "numero": "pl", "vocativo": False, "quien": True},
    "mujer":    {"cat": NOM_PERS, "forma": "mujer", "sola": "Mujer.", "pn": "3s", "genero": "f",
                 "numero": "sg", "vocativo": False, "quien": True},
    "hombre":   {"cat": NOM_PERS, "forma": "hombre", "sola": "Hombre.", "pn": "3s", "genero": "m",
                 "numero": "sg", "vocativo": False, "quien": True},
    "niño":     {"cat": NOM_PERS, "forma": "niño", "sola": "Niño.", "pn": "3s", "genero": "m",
                 "numero": "sg", "vocativo": True, "quien": True},
    "niña":     {"cat": NOM_PERS, "forma": "niña", "sola": "Niña.", "pn": "3s", "genero": "f",
                 "numero": "sg", "vocativo": True, "quien": True},
    "abuelo":   {"cat": NOM_PERS, "forma": "abuelo", "sola": "Abuelo.", "pn": "3s", "genero": "m",
                 "numero": "sg", "vocativo": True, "quien": True},
    "tío":      {"cat": NOM_PERS, "forma": "tío", "sola": "Tío.", "pn": "3s", "genero": "m",
                 "numero": "sg", "vocativo": True, "quien": True},
    "hermano":  {"cat": NOM_PERS, "forma": "hermano", "sola": "Hermano.", "pn": "3s", "genero": "m",
                 "numero": "sg", "vocativo": True, "quien": True},
    # --- NOM_ABS: solo por plantilla (R-28) ---
    "seña":   {"cat": NOM_ABS, "forma": "seña", "sola": "Seña.", "genero": "f", "numero": "sg"},
    "nombre": {"cat": NOM_ABS, "forma": "nombre", "sola": "Nombre.", "genero": "m", "numero": "sg"},
    # --- NOM_TENER (R-24) ---
    "hambre": {"cat": NOM_TENER, "forma": "hambre", "sola": "Hambre.", "genero": "f", "numero": "sg"},
    # --- VERBO (R-18, R-22, R-25 a R-27) ---
    "trabajar": {"cat": VERBO, "forma": "trabajar", "sola": "Trabajar.", "cant": True, "modo": True,
                 "nunca": True, "exige_complemento": False},
    "comer":    {"cat": VERBO, "forma": "comer", "sola": "Comer.", "cant": True, "modo": True,
                 "nunca": True, "exige_complemento": False},
    "vivir":    {"cat": VERBO, "forma": "vivir", "sola": "Vivir.", "cant": False, "modo": True,
                 "nunca": False, "exige_complemento": True},
    # --- VERBO_SENTIR (R-23) ---
    "sentir": {"cat": VERBO_SENTIR, "forma": "sentir(se)", "sola": "Sentir."},
    # --- ADJ (R-20, R-21, R-23) ---
    "feliz":      {"cat": ADJ, "forma": "feliz", "sola": "Feliz.", "copula": "estar", "con_sentir": True},
    "contento":   {"cat": ADJ, "forma": "contento", "sola": "Contento.", "copula": "estar", "con_sentir": True},
    "triste":     {"cat": ADJ, "forma": "triste", "sola": "Triste.", "copula": "estar", "con_sentir": True},
    "aburrido":   {"cat": ADJ, "forma": "aburrido", "sola": "Aburrido.", "copula": "estar", "con_sentir": True},
    "juicioso":   {"cat": ADJ, "forma": "juicioso", "sola": "Juicioso.", "copula": "ser", "con_sentir": False},
    "diferente":  {"cat": ADJ, "forma": "diferente", "sola": "Diferente.", "copula": "ser", "con_sentir": True},
    "bienvenido": {"cat": ADJ, "forma": "bienvenido", "sola": "Bienvenido.", "copula": "ser", "con_sentir": True},
    # --- ADV_EST (R-22, R-25) ---
    "bien":        {"cat": ADV_EST, "forma": "bien", "sola": "Bien.", "muy": True},
    "mal":         {"cat": ADV_EST, "forma": "mal", "sola": "Mal.", "muy": True},
    "más o menos": {"cat": ADV_EST, "forma": "más o menos", "sola": "Más o menos.", "muy": False},
    # --- ADV_CANT (R-25): forma ante HAMBRE, concordada con el femenino (R-24) ---
    "mucho": {"cat": ADV_CANT, "forma": "mucho", "sola": "Mucho.", "ante_hambre": "mucha"},
    "poco":  {"cat": ADV_CANT, "forma": "poco", "sola": "Poco.", "ante_hambre": "poca"},
    # --- ADV_FREC (R-26) ---
    "nunca": {"cat": ADV_FREC, "forma": "nunca", "sola": "Nunca."},
}

# Id de cada glosa: fija el orden de recorrido para que nada dependa del orden de un frozenset.
_ID: dict[str, int] = {g: i for i, g in SIGN_VOCABULARY.items()}


# ---------------------------------------------------------------------------
# Tablas de formas (§2.4): cerradas, sin derivar por terminación. Solo las personas que el
# léxico puede producir: 1.ª sg (yo), 2.ª sg (tú), 3.ª sg y 3.ª pl (ustedes, todos, personas).
# ---------------------------------------------------------------------------

FORMAS_VERBALES: dict[str, dict[str, str]] = {
    "trabajar": {"1s": "trabajo",   "2s": "trabajas",   "3s": "trabaja",   "3p": "trabajan"},
    "comer":    {"1s": "como",      "2s": "comes",      "3s": "come",      "3p": "comen"},
    "vivir":    {"1s": "vivo",      "2s": "vives",      "3s": "vive",      "3p": "viven"},
    "sentir":   {"1s": "me siento", "2s": "te sientes", "3s": "se siente", "3p": "se sienten"},
    "ser":      {"1s": "soy",       "2s": "eres",       "3s": "es",        "3p": "son"},
    "estar":    {"1s": "estoy",     "2s": "estás",      "3s": "está",      "3p": "están"},
    "tener":    {"1s": "tengo",     "2s": "tienes",     "3s": "tiene",     "3p": "tienen"},
}

# Género y número: "ms", "fs", "mp", "fp" (R-21)
FORMAS_ADJETIVO: dict[str, dict[str, str]] = {
    "feliz":      {"ms": "feliz",      "fs": "feliz",      "mp": "felices",     "fp": "felices"},
    "contento":   {"ms": "contento",   "fs": "contenta",   "mp": "contentos",   "fp": "contentas"},
    "triste":     {"ms": "triste",     "fs": "triste",     "mp": "tristes",     "fp": "tristes"},
    "aburrido":   {"ms": "aburrido",   "fs": "aburrida",   "mp": "aburridos",   "fp": "aburridas"},
    "juicioso":   {"ms": "juicioso",   "fs": "juiciosa",   "mp": "juiciosos",   "fp": "juiciosas"},
    "diferente":  {"ms": "diferente",  "fs": "diferente",  "mp": "diferentes",  "fp": "diferentes"},
    "bienvenido": {"ms": "bienvenido", "fs": "bienvenida", "mp": "bienvenidos", "fp": "bienvenidas"},
}

ARTICULOS: dict[str, str] = {"ms": "el", "fs": "la", "mp": "los", "fp": "las"}   # R-15

GENEROS = ("m", "f")


# ---------------------------------------------------------------------------
# Plantillas fijas (§4.2, R-07). La clave es el CONJUNTO de glosas (R-04). «admite_saludo»:
# la clave más un solo saludo distinto de ADIÓS sale como «Saludo, » + enunciado (paso B6).
# Las eligió el autor; ninguna está revisada por la persona señante.
# ---------------------------------------------------------------------------

PLANTILLAS: dict[str, dict] = {
    "T-01": {"glosas": ("por favor", "seña"), "frase": "La seña, por favor.", "admite_saludo": False},
    "T-02": {"glosas": ("por favor", "nombre"), "frase": "El nombre, por favor.", "admite_saludo": False},
    "T-03": {"glosas": ("tú", "nombre", "¿qué?"), "frase": "¿Cuál es tu nombre?", "admite_saludo": True},
    "T-04": {"glosas": ("tú", "seña", "¿qué?"), "frase": "¿Cuál es tu seña?", "admite_saludo": True},
    "T-05": {"glosas": ("gracias", "mucho"), "frase": "Muchas gracias.", "admite_saludo": False},
    "T-06": {"glosas": ("con gusto", "mucho"), "frase": "Con mucho gusto.", "admite_saludo": False},
    "T-07": {"glosas": ("gracias", "todos"), "frase": "Gracias a todos.", "admite_saludo": False},
    "T-08": {"glosas": ("perdón", "permiso"), "frase": "Perdón, con permiso.", "admite_saludo": False},
    "T-09": {"glosas": ("por favor", "permiso"), "frase": "Permiso, por favor.", "admite_saludo": False},
    "T-10": {"glosas": ("perdón", "¿qué?"), "frase": "Perdón, ¿qué?", "admite_saludo": False},
    "T-11": {"glosas": ("perdón", "¿cómo?"), "frase": "Perdón, ¿cómo?", "admite_saludo": False},
    "T-12": {"glosas": ("bien", "gracias"), "frase": "Bien, gracias.", "admite_saludo": False},
    "T-13": {"glosas": ("más o menos", "gracias"), "frase": "Más o menos, gracias.", "admite_saludo": False},
    "T-14": {"glosas": ("yo", "bien", "gracias"), "frase": "Yo estoy bien, gracias.", "admite_saludo": False},
    "T-15": {"glosas": ("gracias", "adiós"), "frase": "Gracias, adiós.", "admite_saludo": False},
    "T-16": {"glosas": ("gracias", "buenas noches"), "frase": "Gracias, buenas noches.", "admite_saludo": False},
    "T-17": {"glosas": ("mucho", "gracias", "todos"), "frase": "Muchas gracias a todos.", "admite_saludo": False},
    # 3a-2 (segunda revisión del 8-nov): simetría con T-16, T-07 y T-12/T-13. Sin revisar por la persona señante.
    "T-18": {"glosas": ("gracias", "buenos días"), "frase": "Gracias, buenos días.", "admite_saludo": False},
    "T-19": {"glosas": ("gracias", "buenas tardes"), "frase": "Gracias, buenas tardes.", "admite_saludo": False},
    "T-20": {"glosas": ("gracias", "ustedes"), "frase": "Gracias a ustedes.", "admite_saludo": False},
    "T-21": {"glosas": ("gracias", "tú"), "frase": "Gracias a ti.", "admite_saludo": False},
    "T-22": {"glosas": ("mal", "gracias"), "frase": "Mal, gracias.", "admite_saludo": False},
}


# ---------------------------------------------------------------------------
# Pregunta parcial (R-30). Para cada interrogativo distinto de ¿QUIÉN?: los restos admitidos
# (exactos) y si admite YO. ¿POR QUÉ? admite cualquier predicado válido de P1 a P9 (None).
# ---------------------------------------------------------------------------

PREGUNTAS_ADMITIDAS: dict[str, dict] = {
    "¿dónde?":   {"restos": (("trabajar",), ("comer",), ("vivir",)), "admite_yo": False},
    "¿cuándo?":  {"restos": (("trabajar",), ("comer",)), "admite_yo": False},
    "¿qué?":     {"restos": (("comer",),), "admite_yo": False},
    "¿cómo?":    {"restos": ((), ("trabajar",), ("comer",), ("vivir",), ("sentir",)), "admite_yo": False},
    "¿por qué?": {"restos": None, "admite_yo": True},
}

# Orden de salida de la pregunta parcial (R-30): «¿» + interrogativo + [nunca] + verbo concordado +
# complementos + SUJETO + «?». El sujeto va al final (validar: sí, B5). Es una tabla para que la
# red de seguridad de las pruebas (§9.1, prueba 10) pueda mutarla.
ORDEN_PREGUNTA_PARCIAL: tuple[str, ...] = ("interrogativo", "predicado", "sujeto")

# Saludos que se combinan con HOLA: «Hola, buenos días» (R-09)
SALUDOS_COMBINABLES_CON_HOLA = ("buenos días", "buenas tardes", "buenas noches")


# ---------------------------------------------------------------------------
# Inventario de secuencias previstas (§4.3): las 92 secuencias con su enunciado fijo. El orden de
# las glosas es solo de presentación (R-04). «reglas» son las que intervienen según §4.3 (sin las
# de entrada R-01 a R-06 ni el cierre R-33/R-34); en las de plantilla, su id.
# ---------------------------------------------------------------------------

INVENTARIO: tuple[dict, ...] = tuple(
    {"id": s, "glosas": tuple(g.split(" + ")), "frase": f, "via": v, "reglas": tuple(r.split(", ")) if r else ()}
    for s, g, f, v, r in (
        ("S-01", "buenos días + todos", "Buenos días a todos.", "reglas", "R-09, R-10"),
        ("S-02", "buenas tardes + todos", "Buenas tardes a todos.", "reglas", "R-09, R-10"),
        ("S-03", "tío + juicioso", "El tío es juicioso.", "reglas", "R-15, R-18, R-20, R-21"),
        ("S-04", "familia + feliz", "La familia está feliz.", "reglas", "R-15, R-17, R-18, R-20, R-21"),
        ("S-05", "por favor + seña", "La seña, por favor.", "plantilla", "T-01"),
        ("S-06", "hola + bienvenido", "Hola, bienvenido.", "reglas", "R-09, R-12"),
        ("S-07", "hermano + trabajar", "El hermano trabaja.", "reglas", "R-15, R-18, R-19"),
        ("S-08", "¿dónde? + hermano + vivir", "¿Dónde vive el hermano?", "reglas", "R-15, R-18, R-30"),
        ("S-09", "hola + ¿cómo estás?", "Hola, ¿cómo estás?", "reglas", "R-09, R-12"),
        ("S-10", "hola + todos", "Hola a todos.", "reglas", "R-09, R-10"),
        ("S-11", "buenas noches + todos", "Buenas noches a todos.", "reglas", "R-09, R-10"),
        ("S-12", "adiós + todos", "Adiós a todos.", "reglas", "R-09, R-10"),
        ("S-13", "hola + buenos días", "Hola, buenos días.", "reglas", "R-09"),
        ("S-14", "buenas tardes + ¿cómo estás?", "Buenas tardes, ¿cómo estás?", "reglas", "R-09, R-12"),
        ("S-15", "hola + abuelo", "Hola, abuelo.", "reglas", "R-09, R-11"),
        ("S-16", "buenos días + familia", "Buenos días, familia.", "reglas", "R-09, R-11"),
        ("S-17", "hola + ¿cómo estás? + familia", "Hola, ¿cómo está la familia?", "reglas",
         "R-09, R-15, R-18, R-32"),
        ("S-18", "¿cómo estás? + ustedes", "¿Cómo están ustedes?", "reglas", "R-18, R-32"),
        ("S-19", "¿cómo estás? + todos", "¿Cómo están todos?", "reglas", "R-18, R-32"),
        ("S-20", "buenos días + bienvenido", "Buenos días, bienvenido.", "reglas", "R-09, R-12"),
        ("S-21", "por favor + nombre", "El nombre, por favor.", "plantilla", "T-02"),
        ("S-22", "tú + nombre + ¿qué?", "¿Cuál es tu nombre?", "plantilla", "T-03"),
        ("S-23", "tú + seña + ¿qué?", "¿Cuál es tu seña?", "plantilla", "T-04"),
        ("S-24", "gracias + mucho", "Muchas gracias.", "plantilla", "T-05"),
        ("S-25", "con gusto + mucho", "Con mucho gusto.", "plantilla", "T-06"),
        ("S-26", "gracias + todos", "Gracias a todos.", "plantilla", "T-07"),
        ("S-27", "perdón + permiso", "Perdón, con permiso.", "plantilla", "T-08"),
        ("S-28", "por favor + permiso", "Permiso, por favor.", "plantilla", "T-09"),
        ("S-29", "perdón + ¿qué?", "Perdón, ¿qué?", "plantilla", "T-10"),
        ("S-30", "perdón + ¿cómo?", "Perdón, ¿cómo?", "plantilla", "T-11"),
        ("S-31", "bien + gracias", "Bien, gracias.", "plantilla", "T-12"),
        ("S-32", "más o menos + gracias", "Más o menos, gracias.", "plantilla", "T-13"),
        ("S-33", "yo + bien + gracias", "Yo estoy bien, gracias.", "plantilla", "T-14"),
        ("S-34", "gracias + adiós", "Gracias, adiós.", "plantilla", "T-15"),
        ("S-35", "gracias + buenas noches", "Gracias, buenas noches.", "plantilla", "T-16"),
        ("S-36", "mucho + gracias + todos", "Muchas gracias a todos.", "plantilla", "T-17"),
        ("S-37", "yo + trabajar", "Yo trabajo.", "reglas", "R-16, R-18, R-19"),
        ("S-38", "tú + comer", "Tú comes.", "reglas", "R-16, R-18, R-19"),
        ("S-39", "ustedes + trabajar", "Ustedes trabajan.", "reglas", "R-16, R-17, R-18, R-19"),
        ("S-40", "personas + comer", "Las personas comen.", "reglas", "R-15, R-18, R-19"),
        ("S-41", "todos + trabajar + mucho", "Todos trabajan mucho.", "reglas", "R-18, R-19, R-25"),
        ("S-42", "yo + nunca + comer", "Yo nunca como.", "reglas", "R-16, R-18, R-26"),
        ("S-43", "familia + vivir + bien", "La familia vive bien.", "reglas", "R-15, R-18, R-22"),
        ("S-44", "yo + feliz", "Yo estoy feliz.", "reglas", "R-16, R-18, R-20, R-21"),
        ("S-45", "niña + contento", "La niña está contenta.", "reglas", "R-15, R-18, R-20, R-21"),
        ("S-46", "ustedes + aburrido", "Ustedes están aburridos.", "reglas", "R-17, R-18, R-20, R-21"),
        ("S-47", "hermano + diferente", "El hermano es diferente.", "reglas", "R-15, R-18, R-20, R-21"),
        ("S-48", "yo + bien", "Yo estoy bien.", "reglas", "R-16, R-18, R-22"),
        ("S-49", "abuelo + mal", "El abuelo está mal.", "reglas", "R-15, R-18, R-22"),
        ("S-50", "yo + mucho + feliz", "Yo estoy muy feliz.", "reglas", "R-18, R-20, R-21, R-25"),
        ("S-51", "yo + sentir + triste", "Yo me siento triste.", "reglas", "R-18, R-21, R-23"),
        ("S-52", "tú + sentir + bien", "Tú te sientes bien.", "reglas", "R-18, R-23"),
        ("S-53", "yo + hambre", "Yo tengo hambre.", "reglas", "R-18, R-24"),
        ("S-54", "niño + hambre + mucho", "El niño tiene mucha hambre.", "reglas", "R-15, R-18, R-24, R-25"),
        ("S-55", "todos + bienvenido", "Todos son bienvenidos.", "reglas", "R-17, R-18, R-20, R-21"),
        ("S-56", "tú + bienvenido", "Tú eres bienvenido.", "reglas", "R-17, R-18, R-20, R-21"),
        ("S-57", "¿dónde? + tú + vivir", "¿Dónde vives tú?", "reglas", "R-16, R-18, R-30"),
        ("S-58", "¿cuándo? + hermano + trabajar", "¿Cuándo trabaja el hermano?", "reglas", "R-15, R-18, R-30"),
        ("S-59", "¿qué? + niño + comer", "¿Qué come el niño?", "reglas", "R-15, R-18, R-30"),
        ("S-60", "¿quién? + comer", "¿Quién come?", "reglas", "R-18, R-31"),
        ("S-61", "¿quién? + hambre", "¿Quién tiene hambre?", "reglas", "R-24, R-31"),
        ("S-62", "¿quién? + tú", "¿Quién eres tú?", "reglas", "R-18, R-31"),
        ("S-63", "¿quién? + ustedes", "¿Quiénes son ustedes?", "reglas", "R-18, R-31"),
        ("S-64", "¿cómo? + abuelo", "¿Cómo está el abuelo?", "reglas", "R-15, R-18, R-30"),
        ("S-65", "¿cómo? + tú + sentir", "¿Cómo te sientes tú?", "reglas", "R-18, R-23, R-30"),
        ("S-66", "¿por qué? + niño + triste", "¿Por qué está triste el niño?", "reglas",
         "R-15, R-18, R-20, R-21, R-30"),
        ("S-67", "¿por qué? + tú + trabajar + mucho", "¿Por qué trabajas mucho tú?", "reglas", "R-18, R-25, R-30"),
        ("S-68", "hola + ¿dónde? + hermano + vivir", "Hola, ¿dónde vive el hermano?", "reglas",
         "R-09, R-15, R-18, R-30"),
        ("S-69", "buenos días + familia + feliz", "Buenos días, la familia está feliz.", "reglas",
         "R-09, R-15, R-17, R-18, R-20, R-21"),
        ("S-70", "hola + tú + nombre + ¿qué?", "Hola, ¿cuál es tu nombre?", "plantilla", "T-03, R-07, R-09"),
        ("S-71", "hola + buenos días + ¿cómo? + ustedes", "Hola, buenos días, ¿cómo están ustedes?", "reglas",
         "R-09, R-18, R-30"),
        ("S-72", "hola + buenos días + ¿cómo estás?", "Hola, buenos días, ¿cómo estás?", "reglas", "R-09, R-12"),
        ("S-73", "tío + aburrido + nunca", "El tío nunca está aburrido.", "reglas", "R-15, R-18, R-20, R-21, R-26"),
        ("S-74", "yo + nunca + hambre", "Yo nunca tengo hambre.", "reglas", "R-16, R-18, R-24, R-26"),
        ("S-75", "yo + nunca + sentir + diferente", "Yo nunca me siento diferente.", "reglas",
         "R-16, R-18, R-21, R-23, R-26"),
        ("S-76", "hermano + comer + poco", "El hermano come poco.", "reglas", "R-15, R-18, R-25"),
        ("S-77", "yo + hambre + poco", "Yo tengo poca hambre.", "reglas", "R-16, R-18, R-24, R-25"),
        ("S-78", "¿quién? + feliz", "¿Quién está feliz?", "reglas", "R-18, R-20, R-31"),
        ("S-79", "¿quién? + mal", "¿Quién está mal?", "reglas", "R-18, R-22, R-31"),
        ("S-80", "¿por qué? + tú + sentir + triste", "¿Por qué te sientes triste tú?", "reglas",
         "R-18, R-21, R-23, R-30"),
        ("S-81", "¿por qué? + niño + hambre", "¿Por qué tiene hambre el niño?", "reglas", "R-15, R-18, R-24, R-30"),
        ("S-82", "¿cómo estás? + tú", "¿Cómo estás tú?", "reglas", "R-18, R-32"),
        ("S-83", "tío + trabajar + bien + mucho", "El tío trabaja muy bien.", "reglas", "R-15, R-18, R-22, R-25"),
        ("S-84", "yo + sentir + bienvenido", "Yo me siento bienvenido.", "reglas", "R-16, R-18, R-21, R-23"),
        ("S-85", "adiós + abuelo", "Adiós, abuelo.", "reglas", "R-09, R-11"),
        # 3a-2 (segunda revisión del 8-nov): las plantillas nuevas y las dos ramas nuevas
        ("S-86", "gracias + buenos días", "Gracias, buenos días.", "plantilla", "T-18"),
        ("S-87", "gracias + buenas tardes", "Gracias, buenas tardes.", "plantilla", "T-19"),
        ("S-88", "gracias + ustedes", "Gracias a ustedes.", "plantilla", "T-20"),
        ("S-89", "gracias + tú", "Gracias a ti.", "plantilla", "T-21"),
        ("S-90", "mal + gracias", "Mal, gracias.", "plantilla", "T-22"),
        ("S-91", "gracias + abuelo", "Gracias, abuelo.", "reglas", "R-11, R-13"),
        ("S-92", "familia + bienvenido", "Bienvenida, familia.", "reglas", "R-11, R-12, R-20, R-21"),
    )
)


# ---------------------------------------------------------------------------
# Motivos de respaldo (§5.2): EXACTAMENTE estos 35 códigos, en el orden de §3.1. El código
# defensivo «no_previsto» (§3.1) no está aquí a propósito: si apareciera, sería un fallo.
# ---------------------------------------------------------------------------

MOTIVOS: tuple[str, ...] = (
    "fuera_de_vocabulario", "demasiadas_glosas", "saludos_no_combinables", "despedida_con_contenido",
    "saludo_con_glosa_no_admitida", "sustantivo_abstracto_fuera_de_plantilla", "cortesia_fuera_de_plantilla",
    "dos_sujetos", "dos_interrogativos", "como_estas_no_admitido", "quien_con_sujeto_y_predicado",
    "quien_sujeto_no_admitido", "pregunta_sin_sujeto", "pregunta_no_admitida", "sin_sujeto", "dos_VERBO",
    "dos_ADJ", "dos_ADV_EST", "dos_ADV_CANT", "nunca_con_cantidad", "todos_con_nunca", "nunca_con_modo",
    "dos_predicados", "dos_verbos", "verbo_con_predicado", "nunca_no_admitido", "cantidad_y_modo",
    "cantidad_no_admitida", "vivir_sin_complemento", "sentir_sin_predicativo", "sentir_hambre",
    "adjetivo_no_admite_sentir", "sin_predicado", "yo_bienvenido", "fuera_de_inventario",
)
MOTIVO_NO_PREVISTO = "no_previsto"

# Regla de cada motivo (columna «regla» de §5.2). fuera_de_inventario es del §7.1 (modo estricto),
# no de una regla R-xx, y no se registra en «reglas».
REGLA_DE_MOTIVO: dict[str, str | None] = {
    "fuera_de_vocabulario": "R-02", "demasiadas_glosas": "R-08", "saludos_no_combinables": "R-09",
    "despedida_con_contenido": "R-09", "saludo_con_glosa_no_admitida": "R-12",
    "sustantivo_abstracto_fuera_de_plantilla": "R-28", "cortesia_fuera_de_plantilla": "R-13",
    "dos_sujetos": "R-14", "dos_interrogativos": "R-29", "como_estas_no_admitido": "R-32",
    "quien_con_sujeto_y_predicado": "R-31", "quien_sujeto_no_admitido": "R-31",
    "pregunta_sin_sujeto": "R-30", "pregunta_no_admitida": "R-30", "sin_sujeto": "R-14",
    "dos_VERBO": "R-14", "dos_ADJ": "R-14", "dos_ADV_EST": "R-14", "dos_ADV_CANT": "R-14",
    "nunca_con_cantidad": "R-26", "todos_con_nunca": "R-26", "nunca_con_modo": "R-26",
    "dos_predicados": "R-14", "dos_verbos": "R-14", "verbo_con_predicado": "R-14",
    "nunca_no_admitido": "R-26", "cantidad_y_modo": "R-25", "cantidad_no_admitida": "R-25",
    "vivir_sin_complemento": "R-27", "sentir_sin_predicativo": "R-23", "sentir_hambre": "R-23",
    "adjetivo_no_admite_sentir": "R-23", "sin_predicado": "R-14", "yo_bienvenido": "R-20",
    "fuera_de_inventario": None, MOTIVO_NO_PREVISTO: None,
}

# Respaldo: a lo sumo 8 glosas escritas; el resto va en «omitidas» (§5.3, R-35)
MAX_GLOSAS_RESPALDO = 8
# R-08: más de 4 glosas distintas y sin plantilla → respaldo
MAX_GLOSAS_FRASE = 4
# R-01: registro de lo rechazado
MAX_RECHAZADOS_REGISTRADOS = 10
# R-01: tope de elementos leídos del argumento; lo que pase de aquí no se lee (aviso «entrada_truncada»).
# Evita que un iterable infinito deje explicar sin volver (§5.3).
MAX_ELEMENTOS_ENTRADA = 10_000
MAX_CARACTERES_REPR = 40


# ---------------------------------------------------------------------------
# Alias (R-01): la misma glosa sin tildes (salvo ñ) y sin ¿?, generada de forma automática.
# Solo se guardan las 13 formas que difieren de la glosa.
# ---------------------------------------------------------------------------

def _sin_tildes_ni_interrogacion(texto: str) -> str:
    """«¿Cómo estás?» → «como estas»; conserva la ñ."""
    salida = []
    for c in unicodedata.normalize("NFC", texto):
        if c in "¿?":
            continue
        if c in "ñÑ":
            salida.append(c)
            continue
        salida.append("".join(d for d in unicodedata.normalize("NFD", c) if unicodedata.category(d) != "Mn"))
    return " ".join("".join(salida).split())


ALIAS: dict[str, str] = {
    _sin_tildes_ni_interrogacion(g): g
    for g in SIGN_VOCABULARY.values()
    if _sin_tildes_ni_interrogacion(g) != g
}


# ---------------------------------------------------------------------------
# Comprobaciones al importar (§7.1): con `if` explícito, no con `assert` (python -O lo quitaría).
# ---------------------------------------------------------------------------

if set(LEXICO) != set(SIGN_VOCABULARY.values()):
    raise RuntimeError(
        "conversor_glosas: el léxico no coincide con SIGN_VOCABULARY "
        f"(sobran {sorted(set(LEXICO) - set(SIGN_VOCABULARY.values()))}, "
        f"faltan {sorted(set(SIGN_VOCABULARY.values()) - set(LEXICO))})")
if len(set(ALIAS)) != len(ALIAS) or set(ALIAS) & set(LEXICO):
    raise RuntimeError("conversor_glosas: los alias generados chocan entre sí o con una glosa")


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

class _Respaldo(Exception):
    """Una condición de respaldo detiene el algoritmo y lleva su código (§3.1)."""

    def __init__(self, motivo: str):
        super().__init__(motivo)
        self.motivo = motivo


class _Traza:
    """Acumula reglas, categorías suplidas y supuestos sin repetir y en orden de aparición."""

    def __init__(self) -> None:
        self.entrada: list[str] = []     # reglas de la etapa A (también en el respaldo)
        self.reglas: list[str] = []
        self.suplido: list[str] = []
        self.supuestos: list[str] = []

    @staticmethod
    def _anadir(lista: list[str], *codigos: str) -> None:
        for c in codigos:
            if c not in lista:
                lista.append(c)

    def regla_entrada(self, *ids: str) -> None:
        self._anadir(self.entrada, *ids)

    def regla(self, *ids: str) -> None:
        self._anadir(self.reglas, *ids)

    def suple(self, *codigos: str) -> None:
        self._anadir(self.suplido, *codigos)

    def supone(self, *codigos: str) -> None:
        self._anadir(self.supuestos, *codigos)


def _ordenar(glosas) -> list[str]:
    """Orden fijo (id de SIGN_VOCABULARY) para que nada dependa del orden de un frozenset."""
    return sorted(glosas, key=_ID.__getitem__)


def _mayuscula_inicial(texto: str) -> str:
    """R-33: mayúscula en la primera letra o en la que sigue a un «¿» inicial."""
    if not texto:
        return texto
    if texto[0] == "¿":
        return texto[:1] + texto[1:2].upper() + texto[2:]
    return texto[0].upper() + texto[1:]


def _minuscula_inicial(texto: str) -> str:
    """Enunciado de plantilla tras un saludo (R-07, R-09): «¿Cuál es…?» → «¿cuál es…?»."""
    if texto[:1] == "¿":
        return texto[:1] + texto[1:2].lower() + texto[2:]
    return texto[:1].lower() + texto[1:]


def _repr_seguro(x) -> str:
    """repr protegido y truncado a 40 caracteres (R-01)."""
    try:
        r = repr(x)
        if not isinstance(r, str):
            raise TypeError
    except Exception:   # noqa: BLE001 — un __repr__ roto no rompe la garantía de no lanzar
        r = f"<repr falló: {type(x).__name__}>"
    if len(r) > MAX_CARACTERES_REPR:
        r = r[:MAX_CARACTERES_REPR - 1] + "…"
    return r


_CLAVES_PLANTILLA: dict[frozenset, str] = {frozenset(p["glosas"]): tid for tid, p in PLANTILLAS.items()}
_CLAVES_INVENTARIO: dict[frozenset, dict] = {frozenset(s["glosas"]): s for s in INVENTARIO}

# Rasgos del sujeto virtual de «¿Quién…?»: 3.ª singular (R-31). El adjetivo tras ¿QUIÉN? va en masculino
# singular («¿Quién está contento?»), el masculino genérico de un referente desconocido, sin registrar
# «genero_por_defecto» (R-17, R-31 de la especificación desde 3a-2).
_QUIEN = {"glosa": "¿quién?", "cat": INTERR, "texto": "", "pn": "3s", "genero": "m", "numero": "sg",
          "genero_por_defecto": False}


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class ConversorGlosas:
    """Convierte la lista de glosas de un turno en UN enunciado en español (o en un respaldo marcado)."""

    VERSION = VERSION

    def __init__(self, genero_primera: str | None = None, genero_segunda: str | None = None,
                 estricto: bool = False) -> None:
        """
        Args:
            genero_primera: género de quien habla (YO), "m" o "f". None → "m" por defecto (R-17); con
                el valor por defecto, un adjetivo que varía en género registra «genero_por_defecto»
                en los supuestos. HUECO DE LA ESPECIFICACIÓN (§7.1 escribe `= "m"`): se usa None como
                valor por defecto para poder distinguir el «m» por defecto del «m» explícito, que
                según R-17 no se registra.
            genero_segunda: ídem para quien escucha (TÚ).
            estricto: True → modo estricto (§7.1, paso F): solo dan frase las glosas solas y las 85
                secuencias del inventario; cualquier otra frase pasa a respaldo «fuera_de_inventario».

        Raises:
            ValueError: género distinto de "m"/"f" o estricto que no es bool. Es la única excepción
                del módulo; translate y explicar nunca lanzan.
        """
        for nombre, valor in (("genero_primera", genero_primera), ("genero_segunda", genero_segunda)):
            if valor is not None and (not isinstance(valor, str) or valor not in GENEROS):
                raise ValueError(f"{nombre} debe ser 'm' o 'f', no {valor!r}")
        if not isinstance(estricto, bool):
            raise ValueError(f"estricto debe ser True o False, no {estricto!r}")
        self._genero_explicito = {"yo": genero_primera is not None, "tú": genero_segunda is not None}
        self.genero_primera = genero_primera or "m"
        self.genero_segunda = genero_segunda or "m"
        self.estricto = estricto

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def translate(self, signs) -> str:
        """Compatible con GrammarMapper.translate: misma firma, devuelve solo la frase."""
        return self.explicar(signs)["frase"]

    def translate_ids(self, sign_ids) -> str:
        """Igual que GrammarMapper.translate_ids: los ids ya se aceptan en translate."""
        return self.translate(sign_ids)

    def explicar(self, signs) -> dict:
        """Frase más trazabilidad (§6). Acepta cualquier argumento sin lanzar (R-01).

        Devuelve un dict con las claves de §6, siempre presentes y en este orden: frase, via,
        plantilla, reglas, suplido, supuestos, respaldo, motivo, glosas, fuera_de_vocabulario,
        n_fuera_de_vocabulario, alias, repeticiones, omitidas, avisos, parametros, version.
        """
        t = _Traza()
        avisos: list[str] = []
        rechazados: list[str] = []
        n_rechazados = 0
        alias: list[str] = []
        glosas: list[str] = []
        repeticiones = 0
        try:
            elementos = self._leer_argumento(signs, avisos)
            validas: list[str] = []
            if elementos is not None:
                t.regla_entrada("R-01")
                for x in elementos:
                    g, por_alias = self._normalizar_elemento(x)
                    if g is None:
                        n_rechazados += 1
                        if len(rechazados) < MAX_RECHAZADOS_REGISTRADOS:
                            rechazados.append(_repr_seguro(x))
                    else:
                        validas.append(g)
                        if por_alias is not None:
                            alias.append(por_alias)
            # R-03: glosas distintas en orden de primera aparición
            vistas: set[str] = set()
            for g in validas:
                if g not in vistas:
                    vistas.add(g)
                    glosas.append(g)
            repeticiones = len(validas) - len(glosas)

            if not glosas:                                            # A2, R-05
                t.regla_entrada("R-05")
                return self._salida("", "vacio", t, glosas=glosas, rechazados=rechazados,
                                    n_rechazados=n_rechazados, alias=alias, repeticiones=repeticiones,
                                    avisos=avisos)
            if n_rechazados:                                          # A3, R-02
                raise _Respaldo("fuera_de_vocabulario")
            if repeticiones:                                          # A4, R-03
                t.regla_entrada("R-03")
            if len(glosas) == 1:                                      # A5, R-06
                t.regla_entrada("R-06")
                t.suple("puntuacion", "mayuscula")
                return self._salida(LEXICO[glosas[0]]["sola"], "lexico", t, glosas=glosas,
                                    rechazados=rechazados, n_rechazados=n_rechazados, alias=alias,
                                    repeticiones=repeticiones, avisos=avisos)
            conjunto = frozenset(glosas)                              # A6, R-04
            via, frase, plantilla = self._convertir_conjunto(conjunto, t)
            # F: modo estricto (§7.1)
            if self.estricto and conjunto not in _CLAVES_INVENTARIO:
                raise _Respaldo("fuera_de_inventario")
            t.regla_entrada("R-04")
            return self._salida(frase, via, t, plantilla=plantilla, glosas=glosas, rechazados=rechazados,
                                n_rechazados=n_rechazados, alias=alias, repeticiones=repeticiones,
                                avisos=avisos)
        except _Respaldo as r:
            return self._respaldo(r.motivo, t, glosas=glosas, rechazados=rechazados,
                                  n_rechazados=n_rechazados, alias=alias, repeticiones=repeticiones,
                                  avisos=avisos)
        except Exception:   # noqa: BLE001 — rama defensiva (§3.1): nunca lanza (§5.1, garantía 1)
            if glosas:
                return self._respaldo(MOTIVO_NO_PREVISTO, t, glosas=glosas, rechazados=rechazados,
                                      n_rechazados=n_rechazados, alias=alias, repeticiones=repeticiones,
                                      avisos=avisos)
            return self._salida("", "vacio", _Traza(), glosas=[], rechazados=rechazados,
                                n_rechazados=n_rechazados, alias=alias, repeticiones=0, avisos=avisos)

    # ------------------------------------------------------------------
    # Etapa A — Entrada (R-01)
    # ------------------------------------------------------------------

    @classmethod
    def _leer_argumento(cls, signs, avisos: list[str]) -> list | None:
        """R-01 sobre el argumento: str suelto → [str] con aviso; None, bytes, bytearray, dict o no
        iterable → None (vía vacio) con aviso; set o frozenset → sus elementos en orden de id de
        SIGN_VOCABULARY, con aviso «entrada_sin_orden» (un conjunto no trae orden, y el de su iteración
        depende de PYTHONHASHSEED); cualquier otro iterable se recorre una vez. Se leen a lo sumo
        MAX_ELEMENTOS_ENTRADA elementos: si hay más, el resto no se lee y se avisa «entrada_truncada».

        Un iterable que lanza al recorrerlo (por ejemplo, un generador que falla a mitad) se trata como
        «no es una lista» (vía vacio con aviso entrada_no_lista), sin usar los elementos leídos hasta el
        fallo. La garantía 2 de §5.1 («nunca "" si hubo glosas») se refiere a un argumento que se pudo
        recorrer entero (§5.1).
        """
        if isinstance(signs, str):
            avisos.append("entrada_str_suelta")
            return [signs]
        if signs is None or isinstance(signs, (bytes, bytearray, dict)):
            avisos.append("entrada_no_lista")
            return None
        try:
            if isinstance(signs, (set, frozenset)):
                avisos.append("entrada_sin_orden")
                elementos = sorted(signs, key=cls._clave_de_orden)
            else:
                elementos = list(itertools.islice(iter(signs), MAX_ELEMENTOS_ENTRADA + 1))
        except Exception:   # noqa: BLE001 — no iterable, o iterable que falla
            avisos.append("entrada_no_lista")
            return None
        if len(elementos) > MAX_ELEMENTOS_ENTRADA:
            avisos.append("entrada_truncada")
            elementos = elementos[:MAX_ELEMENTOS_ENTRADA]
        return elementos

    @classmethod
    def _clave_de_orden(cls, x) -> tuple:
        """Orden de los elementos de un set (R-01): las glosas por id; lo que no es glosa, después, por repr."""
        g, _ = cls._normalizar_elemento(x)
        return (0, _ID[g], "") if g is not None else (1, 0, _repr_seguro(x))

    @staticmethod
    def _normalizar_elemento(x) -> tuple[str | None, str | None]:
        """R-01 sobre cada elemento. Devuelve (glosa o None, entrada aceptada por alias o None)."""
        try:
            if isinstance(x, numbers.Integral) and not isinstance(x, bool):
                return SIGN_VOCABULARY.get(int(x)), None
            if isinstance(x, str):
                s = " ".join(unicodedata.normalize("NFC", x).lower().split())
                if s in LEXICO:
                    return s, None
                if s in ALIAS:
                    return ALIAS[s], s
        except Exception:   # noqa: BLE001 — un elemento raro queda fuera de vocabulario
            pass
        return None, None

    # ------------------------------------------------------------------
    # Conjunto de 2 o más glosas: plantilla, periferia o núcleo
    # ------------------------------------------------------------------

    def _convertir_conjunto(self, conjunto: frozenset, t: _Traza) -> tuple[str, str, str | None]:
        """Devuelve (via, frase, plantilla) o lanza _Respaldo."""
        tid = _CLAVES_PLANTILLA.get(conjunto)                      # A7, R-07
        if tid is not None:
            return self._via_plantilla(tid, conjunto, None, t)
        if len(conjunto) > MAX_GLOSAS_FRASE:                       # A8, R-08
            raise _Respaldo("demasiadas_glosas")
        saludos = [g for g in _ordenar(conjunto) if LEXICO[g]["cat"] == SAL]
        if saludos:
            resultado = self._periferia(conjunto, saludos, t)
            if resultado[0] == "plantilla":
                return resultado
            texto = resultado[1]
        else:
            texto = self._nucleo(conjunto, t)
        # E: R-33 mayúscula inicial, R-34 cierre
        frase = _mayuscula_inicial(texto)
        if not frase.endswith("?"):
            frase += "."
        t.regla("R-33", "R-34")
        t.suple("puntuacion", "mayuscula", "orden_fijo")
        return "reglas", frase, None

    def _via_plantilla(self, tid: str, conjunto: frozenset, saludo: str | None,
                       t: _Traza) -> tuple[str, str, str]:
        """R-07. La plantilla se valida como unidad (§6): suplido solo recoge la coma de R-09 al
        anteponer un saludo; supuestos, «tratamiento_tu» si la plantilla elige «tu»."""
        t.reglas = []
        t.suplido = []
        t.supuestos = []
        frase = PLANTILLAS[tid]["frase"]
        if saludo is not None:
            t.regla("R-07", "R-09")
            t.suple("puntuacion")
            frase = f"{_mayuscula_inicial(saludo)}, {_minuscula_inicial(frase)}"
        else:
            t.regla("R-07")
        if "tú" in conjunto:
            t.supone("tratamiento_tu")
        return "plantilla", frase, tid

    # ------------------------------------------------------------------
    # Etapa B — Periferia: saludos (R-09 a R-12)
    # ------------------------------------------------------------------

    def _periferia(self, conjunto: frozenset, saludos: list[str], t: _Traza) -> tuple:
        """Pasos B1 a B8. Devuelve ("reglas", texto_sin_cierre) o ("plantilla", frase, tid)."""
        t.regla("R-09")
        resto = frozenset(conjunto - set(saludos))
        # B1: un saludo, o HOLA + {BUENOS DÍAS | BUENAS TARDES | BUENAS NOCHES}
        if len(saludos) >= 3:
            raise _Respaldo("saludos_no_combinables")
        if len(saludos) == 2:
            otro = [s for s in saludos if s != "hola"]
            if "hola" not in saludos or otro[0] not in SALUDOS_COMBINABLES_CON_HOLA:
                raise _Respaldo("saludos_no_combinables")
            texto_saludo = f"hola, {LEXICO[otro[0]]['forma']}"
        else:
            texto_saludo = LEXICO[saludos[0]]["forma"]
        es_despedida = any(LEXICO[s].get("despedida") for s in saludos)
        t.suple("puntuacion")   # coma entre saludos o ante el contenido; punto o «?» (R-09, R-34)

        if not resto:                                              # B2
            return "reglas", texto_saludo
        if resto == {"todos"}:                                     # B3, R-10
            t.regla("R-10")
            t.suple("preposicion")
            return "reglas", f"{texto_saludo} a todos"
        if len(resto) == 1:
            unica = next(iter(resto))
            info = LEXICO[unica]
            if info["cat"] == NOM_PERS and info.get("vocativo"):   # B4, R-11
                t.regla("R-11")
                return "reglas", f"{texto_saludo}, {info['forma']}"
        if es_despedida:                                           # B5
            raise _Respaldo("despedida_con_contenido")
        tid = _CLAVES_PLANTILLA.get(resto)                         # B6, R-07 (T-03, T-04)
        if tid is not None and PLANTILLAS[tid]["admite_saludo"] and len(saludos) == 1:
            return self._via_plantilla(tid, conjunto, texto_saludo, t)
        if len(resto) == 1:                                        # B7, R-12
            unica = next(iter(resto))
            if unica == "bienvenido":
                # R-12 (3a-2): el destinatario es quien escucha; el género sale de genero_segunda, en
                # singular. Con el valor por defecto se registra «genero_por_defecto» (R-17).
                t.regla("R-12", "R-21")
                t.suple("concordancia_adjetivo")
                if not self._genero_explicito["tú"]:
                    t.supone("genero_por_defecto")
                return "reglas", f"{texto_saludo}, {FORMAS_ADJETIVO['bienvenido'][self.genero_segunda + 's']}"
            if unica == "¿cómo estás?":
                t.regla("R-12")
                return "reglas", f"{texto_saludo}, {LEXICO[unica]['forma']}"
            raise _Respaldo("saludo_con_glosa_no_admitida")
        # B8: resto de 2 o 3 glosas → núcleo; «Saludo, » + núcleo (el núcleo en minúscula)
        return "reglas", f"{texto_saludo}, {self._nucleo(resto, t, con_saludo=True)}"

    # ------------------------------------------------------------------
    # Etapa C — Núcleo
    # ------------------------------------------------------------------

    def _nucleo(self, conjunto: frozenset, t: _Traza, con_saludo: bool = False) -> str:
        """Pasos C1 a C7. Devuelve el texto del núcleo sin mayúscula ni cierre (la pregunta ya
        termina en «?»), o lanza _Respaldo. «con_saludo»: el núcleo es el resto de B8."""
        por_cat: dict[str, list[str]] = {c: [] for c in CATEGORIAS}
        for g in _ordenar(conjunto):
            por_cat[LEXICO[g]["cat"]].append(g)
        if por_cat[NOM_ABS]:                                       # C1, R-28
            raise _Respaldo("sustantivo_abstracto_fuera_de_plantilla")
        if por_cat[CORT]:                                          # C2, R-13
            # R-11 (3a-2): GRACIAS + un vocativo, sin saludo y sin nada más → «Gracias, x.»
            otros = [g for g in conjunto if g != "gracias"]
            if (not con_saludo and "gracias" in conjunto and len(otros) == 1
                    and LEXICO[otros[0]]["cat"] == NOM_PERS and LEXICO[otros[0]].get("vocativo")):
                t.regla("R-13", "R-11")
                t.suple("puntuacion")
                return f"gracias, {LEXICO[otros[0]]['forma']}"
            raise _Respaldo("cortesia_fuera_de_plantilla")
        sujetos = [g for c in CATEGORIAS_SUJETO for g in por_cat[c]]
        if len(sujetos) > 1:                                       # C3, R-14
            raise _Respaldo("dos_sujetos")
        t.regla("R-14")
        interrogativos = por_cat[INTERR] + por_cat[PREG_FIJA]
        if len(interrogativos) > 1:                                # C4, R-29
            raise _Respaldo("dos_interrogativos")
        sujeto = sujetos[0] if sujetos else None
        resto = frozenset(conjunto - set(sujetos) - set(interrogativos))
        if interrogativos:                                         # C5 → D
            return self._interrogativa(interrogativos[0], sujeto, resto, t)
        if sujeto is None:                                         # C6, R-14
            raise _Respaldo("sin_sujeto")
        info = LEXICO[sujeto]
        if resto == {"bienvenido"} and info["cat"] == NOM_PERS and info.get("vocativo"):   # C7a, R-20
            # Excepción «fórmula» de R-20 (3a-2): BIENVENIDO es fórmula de cortesía (INSOR 2006, p. 486,
            # como en R-12) y el sustantivo va como vocativo (R-11), sin artículo ni cópula: «Bienvenida,
            # familia.». Se registra la lectura elegida frente a «La familia es bienvenida.».
            t.regla("R-20", "R-12", "R-11", "R-21")
            t.suple("concordancia_adjetivo", "puntuacion")
            t.supone("lectura_vocativa")
            forma = FORMAS_ADJETIVO["bienvenido"][info["genero"] + ("s" if info["numero"] == "sg" else "p")]
            return f"{forma}, {info['forma']}"
        suj = self._sujeto(sujeto, t)                              # C7
        predicado = self._predicado(resto, suj, t)
        return f"{suj['texto']} {predicado}"

    def _sujeto(self, glosa: str, t: _Traza) -> dict:
        """R-15 (artículo), R-16 (pronombre explícito), R-17 (persona, número y género)."""
        info = LEXICO[glosa]
        t.regla("R-17")
        genero = info["genero"]
        por_defecto = False
        if glosa == "yo":
            genero, por_defecto = self.genero_primera, not self._genero_explicito["yo"]
        elif glosa == "tú":
            genero, por_defecto = self.genero_segunda, not self._genero_explicito["tú"]
        if info["cat"] == NOM_PERS:
            articulo = ARTICULOS[genero + ("s" if info["numero"] == "sg" else "p")]
            texto = f"{articulo} {info['forma']}"
            t.regla("R-15")
            t.suple("articulo")
            t.supone("articulo_definido")
        else:
            texto = info["forma"]
            if info["cat"] == PRON:
                t.regla("R-16")
        if glosa == "tú":
            t.supone("tratamiento_tu")
        return {"glosa": glosa, "cat": info["cat"], "texto": texto, "pn": info["pn"], "genero": genero,
                "numero": info["numero"], "quien": info.get("quien", False), "genero_por_defecto": por_defecto}

    def _forma_verbal(self, verbo: str, suj: dict, t: _Traza) -> str:
        """R-18 (concordancia) y R-19 (presente de indicativo)."""
        t.regla("R-18", "R-19")
        t.suple("persona_numero", "tiempo_verbal")
        t.supone("tiempo_presente")
        return FORMAS_VERBALES[verbo][suj["pn"]]

    def _adjetivo(self, adj: str, suj: dict, t: _Traza) -> str:
        """R-21: adjetivo concordado con el sujeto. Registra «genero_por_defecto» (R-17) si el sujeto
        es YO o TÚ con el género por defecto y el adjetivo varía en género."""
        formas = FORMAS_ADJETIVO[adj]
        t.regla("R-21")
        t.suple("concordancia_adjetivo")
        if suj.get("genero_por_defecto") and formas["ms"] != formas["fs"]:
            t.supone("genero_por_defecto")
        return formas[suj["genero"] + ("s" if suj["numero"] == "sg" else "p")]

    # ------------------------------------------------------------------
    # Etapa D — Núcleo interrogativo (R-29 a R-32)
    # ------------------------------------------------------------------

    def _interrogativa(self, q: str, sujeto: str | None, resto: frozenset, t: _Traza) -> str:
        t.regla("R-29")
        t.suple("puntuacion")
        if q == "¿cómo estás?":                                    # D1, R-32
            if resto or sujeto is None or sujeto == "yo":
                raise _Respaldo("como_estas_no_admitido")
            t.regla("R-32")
            suj = self._sujeto(sujeto, t)
            if LEXICO[sujeto].get("vocativo"):
                # 3a-2: con un sustantivo que admite vocativo cabe también «Hermano, ¿cómo estás?». Se
                # mantiene la 3.ª persona (R-32) y se registra la lectura elegida (B5, §9.2).
                t.supone("lectura_tercera_persona")
            return f"¿cómo {self._forma_verbal('estar', suj, t)} {suj['texto']}?"
        if q == "¿quién?":                                         # D2, R-31
            t.regla("R-31")
            if sujeto is None:
                return f"¿quién {self._predicado(resto, _QUIEN, t)}?"
            if resto:
                raise _Respaldo("quien_con_sujeto_y_predicado")
            if not LEXICO[sujeto].get("quien"):
                raise _Respaldo("quien_sujeto_no_admitido")
            suj = self._sujeto(sujeto, t)
            t.suple("copula")
            interrogativo = "quiénes" if suj["numero"] == "pl" else "quién"
            return f"¿{interrogativo} {self._forma_verbal('ser', suj, t)} {suj['texto']}?"
        # D3: otro interrogativo, R-30
        t.regla("R-30")
        if sujeto is None:
            raise _Respaldo("pregunta_sin_sujeto")
        admitidas = PREGUNTAS_ADMITIDAS[q]
        if admitidas["restos"] is not None:
            clave = tuple(_ordenar(resto))
            if (sujeto == "yo" and not admitidas["admite_yo"]) or clave not in admitidas["restos"]:
                raise _Respaldo("pregunta_no_admitida")
            suj = self._sujeto(sujeto, t)
            if not clave:                                          # «¿Cómo está…?»
                t.suple("copula")
                predicado = self._forma_verbal("estar", suj, t)
            elif clave == ("sentir",):                             # «¿Cómo se siente…?», R-23
                t.regla("R-23")
                t.suple("pronombre_reflexivo")
                predicado = self._forma_verbal("sentir", suj, t)
            else:
                predicado = self._forma_verbal(clave[0], suj, t)
        else:                                                      # ¿POR QUÉ?: P1 a P9
            suj = self._sujeto(sujeto, t)
            predicado = self._predicado(resto, suj, t)
        partes = {"interrogativo": LEXICO[q]["forma"], "predicado": predicado, "sujeto": suj["texto"]}
        return "¿" + " ".join(partes[p] for p in ORDEN_PREGUNTA_PARCIAL) + "?"

    # ------------------------------------------------------------------
    # Etapa P — Predicado (P1 a P9)
    # ------------------------------------------------------------------

    def _predicado(self, resto: frozenset, suj: dict, t: _Traza) -> str:
        """Predicado del núcleo, en el orden normativo P1…P9 de §3.1. «suj» es el sujeto (o el
        sujeto virtual de ¿QUIÉN?, en 3.ª singular). Devuelve el texto o lanza _Respaldo."""
        por_cat: dict[str, list[str]] = {c: [] for c in CATEGORIAS}
        for g in _ordenar(resto):
            por_cat[LEXICO[g]["cat"]].append(g)
        admitidas = {VERBO, VERBO_SENTIR, ADJ, ADV_EST, ADV_CANT, ADV_FREC, NOM_TENER}
        if any(por_cat[c] for c in CATEGORIAS if c not in admitidas):
            raise _Respaldo(MOTIVO_NO_PREVISTO)                     # rama defensiva (§3.1)
        # P1, R-14: dos glosas de una misma categoría, en este orden fijo
        for c in (VERBO, ADJ, ADV_EST, ADV_CANT):
            if len(por_cat[c]) > 1:
                raise _Respaldo(f"dos_{c}")
        verbo = por_cat[VERBO][0] if por_cat[VERBO] else None
        sentir = bool(por_cat[VERBO_SENTIR])
        adj = por_cat[ADJ][0] if por_cat[ADJ] else None
        adv = por_cat[ADV_EST][0] if por_cat[ADV_EST] else None
        cant = por_cat[ADV_CANT][0] if por_cat[ADV_CANT] else None
        nunca = bool(por_cat[ADV_FREC])
        hambre = bool(por_cat[NOM_TENER])
        if nunca and cant:                                         # P2, R-26
            raise _Respaldo("nunca_con_cantidad")
        if nunca and suj["glosa"] == "todos":                      # P3, R-26
            raise _Respaldo("todos_con_nunca")
        if nunca and verbo and (adj or adv or hambre):             # P4, R-26
            raise _Respaldo("nunca_con_modo")
        if sum(1 for p in (adj, adv, hambre) if p) > 1:            # P5, R-14
            raise _Respaldo("dos_predicados")
        if verbo and sentir:                                       # P6, R-14
            raise _Respaldo("dos_verbos")
        pre = "nunca " if nunca else ""

        if verbo:                                                  # P7
            v = LEXICO[verbo]
            if adj or hambre:
                raise _Respaldo("verbo_con_predicado")
            if nunca and not v["nunca"]:
                raise _Respaldo("nunca_no_admitido")
            if adv:
                if not v["modo"]:
                    raise _Respaldo(MOTIVO_NO_PREVISTO)             # rama defensiva
                if cant == "mucho":
                    if not LEXICO[adv]["muy"]:
                        raise _Respaldo("cantidad_y_modo")
                    t.regla("R-22", "R-25")
                    t.suple("intensificador")
                    return f"{pre}{self._forma_verbal(verbo, suj, t)} muy {LEXICO[adv]['forma']}"
                if cant == "poco":
                    raise _Respaldo("cantidad_y_modo")
                t.regla("R-22")
                return f"{pre}{self._forma_verbal(verbo, suj, t)} {LEXICO[adv]['forma']}"
            if cant:
                if not v["cant"]:
                    raise _Respaldo("cantidad_no_admitida")
                t.regla("R-25")
                return f"{pre}{self._forma_verbal(verbo, suj, t)} {LEXICO[cant]['forma']}"
            if v["exige_complemento"]:
                raise _Respaldo("vivir_sin_complemento")
            if nunca:
                t.regla("R-26")
            return f"{pre}{self._forma_verbal(verbo, suj, t)}"

        if sentir:                                                 # P8, R-23
            # HUECO DE LA ESPECIFICACIÓN: «sin predicativo» se lee como «sin ADJ, ADV_EST ni HAMBRE»;
            # si HAMBRE no contara, «sentir_hambre» (siguiente línea) sería inalcanzable. Con esta lectura
            # se reproducen las cifras de §9.1 (28 pares con sentir_sin_predicativo).
            if not (adj or adv or hambre):
                raise _Respaldo("sentir_sin_predicativo")
            if hambre:
                raise _Respaldo("sentir_hambre")
            if adj and not LEXICO[adj]["con_sentir"]:
                raise _Respaldo("adjetivo_no_admite_sentir")
            if cant == "poco" or (cant == "mucho" and adv and not LEXICO[adv]["muy"]):
                raise _Respaldo("cantidad_no_admitida")
            if nunca and adv and not LEXICO[adv]["muy"]:           # NUNCA + MÁS O MENOS
                raise _Respaldo("nunca_no_admitido")
            t.regla("R-23")
            t.suple("pronombre_reflexivo")
            if nunca:
                t.regla("R-26")
            muy = ""
            if cant == "mucho":
                t.regla("R-25")
                t.suple("intensificador")
                muy = "muy "
            forma = self._forma_verbal("sentir", suj, t)
            predicativo = self._adjetivo(adj, suj, t) if adj else LEXICO[adv]["forma"]
            return f"{pre}{forma} {muy}{predicativo}"

        # P9: sin verbo
        if not (adj or adv or hambre):
            raise _Respaldo("sin_predicado")
        if adj:
            info = LEXICO[adj]
            # R-20, excepción yo_bienvenido. Va antes que nunca_no_admitido, en el orden en que §3.1 (P9)
            # lista las dos: YO + NUNCA + BIENVENIDO sale con yo_bienvenido (no cambia ninguna cifra de §9.1).
            if suj["glosa"] == "yo" and adj == "bienvenido":
                raise _Respaldo("yo_bienvenido")
            if cant == "poco":
                raise _Respaldo("cantidad_no_admitida")
            if nunca and info["copula"] == "ser":
                raise _Respaldo("nunca_no_admitido")
            t.regla("R-20")
            t.suple("copula")
            if nunca:
                t.regla("R-26")
            muy = ""
            if cant == "mucho":
                t.regla("R-25")
                t.suple("intensificador")
                muy = "muy "
            copula = self._forma_verbal(info["copula"], suj, t)
            return f"{pre}{copula} {muy}{self._adjetivo(adj, suj, t)}"
        if adv:
            info = LEXICO[adv]
            if cant and not (cant == "mucho" and info["muy"]):
                raise _Respaldo("cantidad_no_admitida")
            if nunca and not info["muy"]:                          # NUNCA + MÁS O MENOS
                raise _Respaldo("nunca_no_admitido")
            t.regla("R-22")
            t.suple("copula")
            if nunca:
                t.regla("R-26")
            muy = ""
            if cant == "mucho":
                t.regla("R-25")
                t.suple("intensificador")
                muy = "muy "
            return f"{pre}{self._forma_verbal('estar', suj, t)} {muy}{info['forma']}"
        # HAMBRE, R-24: «tener» + [mucha|poca] + hambre
        t.regla("R-24")
        t.suple("verbo_soporte")
        if nunca:
            t.regla("R-26")
        cuanto = ""
        if cant:
            t.regla("R-25")
            cuanto = LEXICO[cant]["ante_hambre"] + " "
        return f"{pre}{self._forma_verbal('tener', suj, t)} {cuanto}hambre"

    # ------------------------------------------------------------------
    # Salida (§6)
    # ------------------------------------------------------------------

    def _parametros(self) -> dict:
        return {"genero_primera": self.genero_primera, "genero_segunda": self.genero_segunda,
                "estricto": self.estricto}

    def _salida(self, frase: str, via: str, t: _Traza, *, plantilla: str | None = None, glosas: list[str],
                rechazados: list[str], n_rechazados: int, alias: list[str], repeticiones: int,
                avisos: list[str]) -> dict:
        if via in ("vacio",):
            suplido, supuestos = [], []
        else:
            suplido, supuestos = list(t.suplido), list(t.supuestos)
        return {
            "frase": frase,
            "via": via,
            "plantilla": plantilla,
            "reglas": list(t.entrada) + [r for r in t.reglas if r not in t.entrada],
            "suplido": suplido,
            "supuestos": supuestos,
            "respaldo": False,
            "motivo": None,
            "glosas": list(glosas),
            "fuera_de_vocabulario": list(rechazados),
            "n_fuera_de_vocabulario": int(n_rechazados),
            "alias": list(alias),
            "repeticiones": int(repeticiones),
            "omitidas": [],
            "avisos": list(avisos),
            "parametros": self._parametros(),
            "version": VERSION,
        }

    def _respaldo(self, motivo: str, t: _Traza, *, glosas: list[str], rechazados: list[str],
                  n_rechazados: int, alias: list[str], repeticiones: int, avisos: list[str]) -> dict:
        """R-35: las glosas válidas distintas, en orden de primera aparición, unidas por «, »; sin
        mayúscula inicial ni punto final. Más de 8 → las 8 primeras y el resto en «omitidas» (§5.3).
        Las reglas son las de entrada (sin R-04, que no vale en el respaldo), la del motivo y R-35."""
        reglas = [r for r in t.entrada if r != "R-04"]
        regla_motivo = REGLA_DE_MOTIVO.get(motivo)
        for r in ((regla_motivo,) if regla_motivo else ()) + ("R-35",):
            if r not in reglas:
                reglas.append(r)
        return {
            "frase": ", ".join(glosas[:MAX_GLOSAS_RESPALDO]),
            "via": "respaldo",
            "plantilla": None,
            "reglas": reglas,
            "suplido": [],
            "supuestos": [],
            "respaldo": True,
            "motivo": motivo,
            "glosas": list(glosas),
            "fuera_de_vocabulario": list(rechazados),
            "n_fuera_de_vocabulario": int(n_rechazados),
            "alias": list(alias),
            "repeticiones": int(repeticiones),
            "omitidas": list(glosas[MAX_GLOSAS_RESPALDO:]),
            "avisos": list(avisos),
            "parametros": self._parametros(),
            "version": VERSION,
        }


# ---------------------------------------------------------------------------
# Comprobación al importar (§7.1): cada clave de INVENTARIO da su enunciado fijo.
# ---------------------------------------------------------------------------

def _comprobar_inventario() -> list[str]:
    """Lista de discrepancias entre INVENTARIO y lo que produce el conversor (vacía si todo cuadra)."""
    conv = ConversorGlosas()
    fallos = []
    for s in INVENTARIO:
        frase = conv.translate(list(s["glosas"]))
        if frase != s["frase"]:
            fallos.append(f"{s['id']}: se esperaba {s['frase']!r} y sale {frase!r}")
    return fallos


_fallos_inventario = _comprobar_inventario()
if _fallos_inventario:
    raise RuntimeError("conversor_glosas: el inventario no reproduce su enunciado fijo: "
                       + "; ".join(_fallos_inventario))
del _fallos_inventario


# ---------------------------------------------------------------------------
# Recorridos y exportación (§6, §9.1)
# ---------------------------------------------------------------------------

def conjuntos(tamanos=(2, 3, 4)):
    """Todos los conjuntos de glosas distintas de los tamaños dados, como listas en orden de id."""
    for n in tamanos:
        for ids in itertools.combinations(sorted(SIGN_VOCABULARY), n):
            yield [SIGN_VOCABULARY[i] for i in ids]


def muestra_trios(n: int = 1000, semilla: int = 2026) -> list[list[str]]:
    """Muestra determinista de tríos ORDENADOS de glosas distintas (de los 117.600 posibles).

    La usan tests/test_conversor_glosas.py y scripts/conversor_3a_salidas.py, para que el CSV y las
    pruebas recorran la misma muestra. La semilla y el tamaño son decisión de la implementación
    (la especificación no los fija)."""
    ids = sorted(SIGN_VOCABULARY)
    todos = list(itertools.permutations(ids, 3))
    elegidos = random.Random(semilla).sample(todos, n)
    return [[SIGN_VOCABULARY[i] for i in trio] for trio in elegidos]


# Orden fijo de presentación de una secuencia en las hojas de la Actividad B y en el CSV de §6 (§9.2).
# NO es una hipótesis sobre el orden de la LSC: el conversor ignora el orden (R-04). Solo fija cómo se
# escribe la secuencia para que todas las hojas sigan el mismo criterio: saludo (HOLA primero),
# cortesía, interrogativo, sujeto, NUNCA, verbo, predicado, cuantificador.
ORDEN_PRESENTACION: tuple[str, ...] = (SAL, CORT, PREG_FIJA, INTERR, PRON, CUANT_PRON, NOM_PERS, ADV_FREC,
                                       VERBO, VERBO_SENTIR, NOM_ABS, NOM_TENER, ADJ, ADV_EST, ADV_CANT)


def orden_presentacion(glosas) -> list[str]:
    """Las glosas en el orden fijo de presentación (ORDEN_PRESENTACION; dentro de una categoría, HOLA
    primero y luego por id)."""
    return sorted(glosas, key=lambda g: (ORDEN_PRESENTACION.index(LEXICO[g]["cat"]), g != "hola", _ID[g]))


# Muestra B3 (§9.2): estratificada por tamaño (2, 3, 4 glosas) y por presencia de saludo, 5 por estrato.
TAMANOS_B3 = (2, 3, 4)
POR_ESTRATO_B3 = 5
SEMILLA_B3 = 2026


def muestra_b3(por_estrato: int = POR_ESTRATO_B3, semilla: int = SEMILLA_B3) -> list[tuple[str, ...]]:
    """Bloque B3 de la Actividad B (§9.2): conjuntos con frase que NO están en el inventario, elegidos al
    azar con semilla fija dentro de 6 estratos (2, 3 o 4 glosas × con o sin saludo), «por_estrato» en
    cada uno: 30 con los valores por defecto. Desde 3a-2 es estratificada: la muestra simple de 3a-1 dio
    23 de 30 ítems de 4 glosas y 24 con saludo. Los estratos se recorren en orden fijo y cada uno usa su
    propio random.Random(semilla), así que la muestra no depende del orden de iteración de Python."""
    conv = ConversorGlosas()
    estratos: dict[tuple[int, bool], list[tuple[str, ...]]] = {(n, sal): [] for n in TAMANOS_B3 for sal in (False, True)}
    for glosas in conjuntos(TAMANOS_B3):
        if frozenset(glosas) in _CLAVES_INVENTARIO:
            continue
        if not conv.explicar(glosas)["respaldo"]:
            con_saludo = any(LEXICO[g]["cat"] == SAL for g in glosas)
            estratos[(len(glosas), con_saludo)].append(tuple(glosas))
    muestra: list[tuple[str, ...]] = []
    for clave in sorted(estratos):
        muestra.extend(random.Random(semilla).sample(estratos[clave], por_estrato))
    return muestra


COLUMNAS_TABLA = ["glosas", "presentacion", "frase", "via", "plantilla", "reglas", "suplido", "supuestos",
                  "bloque_actividad_B", "estado_revision"]


def tabla_salidas() -> list[dict]:
    """Tabla de §6: las salidas no respaldo (50 solas + los conjuntos de 2 a 4 glosas con frase).

    «glosas» va en orden de id (es la clave); «presentacion», en el orden fijo de las hojas (§9.2).
    «bloque_actividad_B» dice en qué bloque se le MOSTRARÁ el ítem a la persona señante (B1, S-xx del
    inventario, que va en B2, o B3; vacío si no se le muestra). «estado_revision» es «pendiente» en todas
    las filas hasta la sesión de la Actividad B: ninguna salida está revisada."""
    conv = ConversorGlosas()
    b3 = set(muestra_b3())
    filas = []

    def fila(glosas, e, bloque):
        return {"glosas": " + ".join(glosas), "presentacion": " + ".join(orden_presentacion(glosas)),
                "frase": e["frase"], "via": e["via"], "plantilla": e["plantilla"] or "",
                "reglas": " ".join(e["reglas"]), "suplido": " ".join(e["suplido"]),
                "supuestos": " ".join(e["supuestos"]), "bloque_actividad_B": bloque, "estado_revision": "pendiente"}

    for i in sorted(SIGN_VOCABULARY):
        g = SIGN_VOCABULARY[i]
        filas.append(fila([g], conv.explicar([g]), "B1"))
    for glosas in conjuntos():
        e = conv.explicar(glosas)
        if e["respaldo"]:
            continue
        clave = frozenset(glosas)
        if clave in _CLAVES_INVENTARIO:
            bloque = _CLAVES_INVENTARIO[clave]["id"]
        elif tuple(glosas) in b3:
            bloque = "B3"
        else:
            bloque = ""
        filas.append(fila(glosas, e, bloque))
    return filas


def exportar_tabla(ruta: str | Path) -> int:
    """Escribe la tabla de §6 en CSV (UTF-8). Devuelve el número de filas."""
    filas = tabla_salidas()
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_TABLA)
        w.writeheader()
        w.writerows(filas)
    return len(filas)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    """Imprime las secuencias previstas (§4.3) con su salida y cuatro ejemplos de respaldo; con
    --exportar escribe la tabla de §6 (docs/CONVERSOR_3a_SALIDAS_NO_RESPALDO.csv en el repositorio).

    Ninguna de estas salidas está revisada por la persona señante: la columna «bloque_actividad_B» de
    la tabla exportada solo marca qué se le MOSTRARÁ en la Actividad B (B1, S-xx, B3), y
    «estado_revision» vale «pendiente» en todas las filas.
    """
    parser = argparse.ArgumentParser(description="Conversor de glosas LSC50 → español por reglas (3.a)")
    parser.add_argument("--exportar", metavar="RUTA.csv", default=None,
                        help="escribe la tabla de salidas no respaldo (§6) en RUTA.csv")
    args = parser.parse_args(argv)

    conv = ConversorGlosas()
    print("=" * 72)
    print(f"  Conversor de glosas (versión {VERSION}): las {len(INVENTARIO)} secuencias previstas (§4.3)")
    print("  Sin revisar por la persona señante (Actividad B pendiente)")
    print("=" * 72)
    coinciden = 0
    for s in INVENTARIO:
        e = conv.explicar(list(s["glosas"]))
        igual = e["frase"] == s["frase"]
        coinciden += igual
        marca = "" if igual else f"   (esperada: {s['frase']})"
        print(f"  {s['id']}  {' + '.join(g.upper() for g in s['glosas']):42s} → {e['frase']}{marca}")
    print("-" * 72)
    print(f"  {coinciden}/{len(INVENTARIO)} secuencias dan su enunciado fijo")
    print("\n  Ejemplos de respaldo (lista de glosas, sin mayúscula ni punto):")
    for glosas in (["hermano", "tío", "comer"], ["comer", "mucho"], ["hola", "tú"], ["yo", "sentir"]):
        e = conv.explicar(glosas)
        print(f"    {' + '.join(g.upper() for g in glosas):30s} → {e['frase']!r}  [{e['motivo']}]")
    if args.exportar:
        n = exportar_tabla(args.exportar)
        print(f"\n  Tabla de §6 escrita en {args.exportar}: {n} filas")


if __name__ == "__main__":
    main()
