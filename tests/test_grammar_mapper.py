"""
Pruebas de salida esperada del mapeo gramatical (src/grammar_mapper.py).

PROPÓSITO: dejar clavado POR PRUEBAS el comportamiento de HOY, antes de la rearquitectura del
módulo (actividad 3.a, a la espera de la persona señante / 0.c). No son pruebas de corrección
lingüística: varias salidas que aquí se dan por esperadas son AGRAMATICALES en español y están
marcadas como tales. Si el rediseño cambia una de ellas a propósito, el cambio tiene que verse
aquí como una prueba que falla y que se actualiza a conciencia, no como una regresión silenciosa.

Cada salida esperada de este archivo fue OBTENIDA EJECUTANDO el módulo el 16-sep-2026 con
python; ninguna fue supuesta. El estado que
describen está narrado en docs/estado_mapeo_gramatical.md.

Sin TensorFlow, sin mediapipe, sin corpus y sin audio: el módulo es una tabla de categorías y
plantillas, así que estas pruebas son puro texto. Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_grammar_mapper -v
"""

from __future__ import annotations

import contextlib
import io
import itertools
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import grammar_mapper as gm  # noqa: E402
from src.grammar_mapper import GrammarMapper  # noqa: E402


# ---------------------------------------------------------------------------
# Tabla maestra: la salida de HOY para cada una de las 50 señas presentada sola.
# Es el caso que ejecuta el prototipo en modo aislado (translator.py llama a
# translate([clase]) con una sola glosa por ventana aceptada).
# ---------------------------------------------------------------------------

SALIDA_DE_UNA_SOLA_GLOSA: dict[int, str] = {
    0: "Gracias",          1: "Buenos días",      2: "Buenas tardes",   3: "Buenas noches",
    4: "La seña",          5: "El nombre",        6: "Trabaja",         7: "Come",
    8: "Vive",             9: "Poco",            10: "La familia",     11: "Personas",
    12: "La mujer",       13: "El hombre",       14: "El niño",        15: "La niña",
    16: "El abuelo",      17: "El tío",          18: "El hermano",     19: "La hambre",
    20: "Feliz",          21: "Contento",        22: "Triste",         23: "Aburrido",
    24: "Bien",           25: "Mal",             26: "¿Cómo estás?",   27: "Más o menos",
    28: "Siente",         29: "Juicioso",        30: "Hola",           31: "Adiós",
    32: "Por favor",      33: "Con gusto",       34: "Bienvenido",     35: "Perdón",
    36: "El permiso",     37: "Nunca",           38: "Yo",             39: "Tú",
    40: "Ustedes",        41: "¿Qué?",           42: "¿Cuándo?",       43: "¿Dónde?",
    44: "¿Cómo?",         45: "¿Por qué?",       46: "¿Quién?",        47: "Diferente",
    48: "Todos",          49: "Mucho",
}


class _Base(unittest.TestCase):

    def setUp(self):
        self.m = GrammarMapper()

    def esperar(self, glosas, salida):
        self.assertEqual(self.m.translate(list(glosas)), salida)


# ===========================================================================
# 1. Léxico y categorías
# ===========================================================================

class TestLexicoYCategorias(_Base):
    """El vocabulario y la tabla de categorías son los datos de los que sale todo lo demás."""

    def test_cincuenta_senas_con_ids_consecutivos(self):
        self.assertEqual(len(gm.SIGN_VOCABULARY), 50)
        self.assertEqual(sorted(gm.SIGN_VOCABULARY), list(range(50)))
        self.assertEqual(len(set(gm.SIGN_VOCABULARY.values())), 50)

    def test_cada_sena_tiene_exactamente_una_categoria(self):
        # 50 señas ↔ 50 entradas en GRAMMAR_CATEGORIES: no sobra ni falta ninguna.
        nombres = set(gm.SIGN_VOCABULARY.values())
        self.assertEqual(set(gm.GRAMMAR_CATEGORIES), nombres)
        for nombre, info in gm.GRAMMAR_CATEGORIES.items():
            with self.subTest(sena=nombre):
                self.assertIsInstance(info["cat"], str)

    def test_reparto_por_categoria(self):
        from collections import Counter
        conteo = Counter(v["cat"] for v in gm.GRAMMAR_CATEGORIES.values())
        self.assertEqual(dict(conteo), {
            "sujeto": 13, "verbo": 4, "objeto": 4, "adjetivo": 7,
            "adverbio": 6, "saludo": 5, "cortesia": 4, "pregunta": 7,
        })
        self.assertEqual(sum(conteo.values()), 50)

    def test_mapa_inverso(self):
        self.assertEqual(gm.SIGN_NAME_TO_ID["hola"], 30)
        self.assertEqual(gm.SIGN_NAME_TO_ID["¿cómo estás?"], 26)
        self.assertEqual(len(gm.SIGN_NAME_TO_ID), 50)

    def test_id_a_sena_y_marcador_de_id_desconocido(self):
        self.assertEqual(self.m.id_to_sign(30), "hola")
        self.assertEqual(self.m.id_to_sign(999), "[seña_999]")
        self.assertEqual(self.m.id_to_sign(-1), "[seña_-1]")
        self.assertEqual(self.m.ids_to_signs([30, 26, 999]),
                         ["hola", "¿cómo estás?", "[seña_999]"])

    def test_las_instancias_comparten_las_tablas_del_modulo(self):
        # Hoy no hay copia defensiva: dos mapeadores comparten el mismo dict.
        otro = GrammarMapper()
        self.assertIs(self.m.categories, gm.GRAMMAR_CATEGORIES)
        self.assertIs(self.m.categories, otro.categories)


# ===========================================================================
# 2. Una sola glosa: todas las categorías, y el modo aislado del prototipo
# ===========================================================================

class TestUnaSolaGlosa(_Base):
    """translate([g]) para una glosa: es lo que el prototipo emite por cada ventana aceptada."""

    def test_las_50_senas_por_nombre(self):
        for sid, esperada in sorted(SALIDA_DE_UNA_SOLA_GLOSA.items()):
            nombre = gm.SIGN_VOCABULARY[sid]
            with self.subTest(id=sid, sena=nombre):
                self.assertEqual(self.m.translate([nombre]), esperada)

    def test_las_50_senas_por_id(self):
        # translate_ids es la vía del prototipo cuando trabaja con la salida cruda del modelo.
        for sid, esperada in sorted(SALIDA_DE_UNA_SOLA_GLOSA.items()):
            with self.subTest(id=sid):
                self.assertEqual(self.m.translate_ids([sid]), esperada)
                self.assertEqual(self.m.translate([sid]), esperada)

    def test_ninguna_sena_sola_se_queda_sin_salida(self):
        for sid in range(50):
            with self.subTest(id=sid):
                self.assertNotEqual(self.m.translate_ids([sid]).strip(), "")

    # --- un ejemplo declarado por cada categoría gramatical ---

    def test_categoria_sujeto(self):
        self.esperar(["yo"], "Yo")                 # pronombre: sin artículo
        self.esperar(["tú"], "Tú")                 # pronombre: sin artículo
        self.esperar(["hermano"], "El hermano")    # sustantivo masculino: artículo definido
        self.esperar(["niña"], "La niña")          # sustantivo femenino
        self.esperar(["ustedes"], "Ustedes")       # sin género declarado: sin artículo
        self.esperar(["personas"], "Personas")     # HALLAZGO: plural sin género → sin «las»

    def test_categoria_verbo(self):
        # Sin sujeto se conjuga en 3.ª persona del singular del presente.
        self.esperar(["trabajar"], "Trabaja")
        self.esperar(["comer"], "Come")
        self.esperar(["vivir"], "Vive")
        self.esperar(["sentir"], "Siente")

    def test_categoria_objeto(self):
        self.esperar(["nombre"], "El nombre")
        self.esperar(["seña"], "La seña")
        self.esperar(["permiso"], "El permiso")
        self.esperar(["hambre"], "La hambre")      # HALLAZGO: en español es «el hambre»

    def test_categoria_adjetivo(self):
        self.esperar(["feliz"], "Feliz")
        self.esperar(["contento"], "Contento")
        self.esperar(["juicioso"], "Juicioso")
        self.esperar(["bienvenido"], "Bienvenido")

    def test_categoria_adverbio(self):
        self.esperar(["bien"], "Bien")             # tipo estado
        self.esperar(["mucho"], "Mucho")           # tipo cantidad
        self.esperar(["nunca"], "Nunca")           # tipo frecuencia
        self.esperar(["más o menos"], "Más o menos")

    def test_categoria_saludo(self):
        self.esperar(["hola"], "Hola")
        self.esperar(["buenos días"], "Buenos días")
        self.esperar(["adiós"], "Adiós")

    def test_categoria_cortesia(self):
        self.esperar(["gracias"], "Gracias")
        self.esperar(["por favor"], "Por favor")
        self.esperar(["con gusto"], "Con gusto")
        self.esperar(["perdón"], "Perdón")

    def test_categoria_pregunta(self):
        self.esperar(["¿cómo estás?"], "¿Cómo estás?")   # tipo frase_completa
        self.esperar(["¿dónde?"], "¿Dónde?")             # tipo interrogativo
        self.esperar(["¿por qué?"], "¿Por qué?")


# ===========================================================================
# 3. El orden de ENTRADA
# ===========================================================================

class TestOrdenDeEntrada(_Base):
    """La salida se arma con un ORDEN FIJO POR CATEGORÍA, así que en el caso general el orden
    en que llegan las glosas no la cambia. Pero la independencia del orden NO es total: hay
    DOS vías por las que el orden de entrada sí manda, las dos clavadas en TestHallazgos.

      (a) Dentro de una misma categoría (H7a): los tokens se recogen con comprensiones de
          lista sobre «tagged», que conservan el orden de llegada.
      (b) Entre CATEGORÍAS DISTINTAS, en turnos de exactamente dos glosas (H7b): la plantilla
          ("saludo", "pregunta") de PHRASE_TEMPLATES casa solo en ese orden y une con coma;
          invertida no casa, la secuencia cae en las reglas generales y el prefijo se une con
          espacio. Por eso es FALSO decir que el orden solo importa dentro de una categoría.

    Medido por barrido exhaustivo de los 1.225 pares de señas distintas: 188 son sensibles al
    orden = 158 de la misma categoría + 30 de categorías distintas (5 saludos × 6
    interrogativos). Los casos de abajo son los del caso general, en los que sí da igual."""

    def _una_sola_salida(self, glosas, esperada):
        salidas = {self.m.translate(list(p)) for p in itertools.permutations(glosas)}
        self.assertEqual(salidas, {esperada},
                         f"{glosas} produjo más de una salida: {sorted(salidas)}")

    def test_dos_glosas_en_los_dos_ordenes(self):
        self._una_sola_salida(["yo", "comer"], "Yo como")
        self._una_sola_salida(["hermano", "trabajar"], "El hermano trabaja")
        self._una_sola_salida(["mujer", "trabajar"], "La mujer trabaja")
        self._una_sola_salida(["tú", "vivir"], "Tú vives")

    def test_tres_glosas_en_los_seis_ordenes(self):
        self._una_sola_salida(["yo", "comer", "mucho"], "Yo como mucho")
        self._una_sola_salida(["familia", "vivir", "bien"], "La familia vive bien")
        self._una_sola_salida(["yo", "nunca", "comer"], "Yo nunca como")
        self._una_sola_salida(["hola", "yo", "trabajar"], "Hola yo trabajo")
        self._una_sola_salida(["perdón", "yo", "vivir"], "Yo vivo perdón")
        self._una_sola_salida(["¿dónde?", "tú", "vivir"], "¿Dónde vives?")

    def test_el_orden_de_salida_es_fijo_aunque_la_entrada_lo_contradiga(self):
        # La cortesía siempre cierra, aunque llegue primero; el saludo siempre abre.
        self.esperar(["perdón", "yo", "vivir"], "Yo vivo perdón")
        self.esperar(["trabajar", "yo", "hola"], "Hola yo trabajo")
        # El adverbio de frecuencia se antepone al verbo aunque llegue después.
        self.esperar(["yo", "comer", "nunca"], "Yo nunca como")
        # El adverbio de cantidad se pospone aunque llegue antes.
        self.esperar(["mucho", "yo", "trabajar"], "Yo trabajo mucho")


# ===========================================================================
# 4. Plantillas (PHRASE_TEMPLATES)
# ===========================================================================

class TestPlantillas(_Base):
    """Siete patrones de categorías con salida directa. Se comparan contra la tupla de
    categorías EN EL ORDEN DE ENTRADA, y de ahí sale la sensibilidad al orden ENTRE categorías
    distintas (H7b). Pero NO son las únicas reglas sensibles al orden: las reglas generales
    también lo son, dentro de una misma categoría (H7a), porque recogen los tokens con
    comprensiones de lista. Medido en la verificación del 16-sep-2026: de los 188 pares
    sensibles al orden, 148 no casan con ninguna plantilla en ninguno de los dos órdenes
    (p. ej. ["gracias","por favor"] → «Gracias, por favor» frente a ["por favor","gracias"] →
    «Por favor, gracias», categorías ("cortesia","cortesia"), que no es un patrón de la tabla),
    y las 6.669 tercias sensibles al orden tampoco, porque ninguna plantilla tiene 3 categorías."""

    def test_plantillas_declaradas(self):
        self.assertEqual([p for p, _ in gm.PHRASE_TEMPLATES], [
            ("saludo",), ("saludo", "saludo"), ("saludo", "pregunta"),
            ("cortesia",), ("afirmacion",), ("negacion",), ("pregunta",),
        ])

    def test_saludo_solo(self):
        self.esperar(["hola"], "Hola")

    def test_saludo_mas_saludo(self):
        self.esperar(["hola", "buenos días"], "Hola, buenos días")
        self.esperar(["hola", "adiós"], "Hola, adiós")
        self.esperar(["hola", "hola"], "Hola, hola")        # la repetición no se colapsa

    def test_saludo_mas_pregunta(self):
        self.esperar(["hola", "¿cómo estás?"], "Hola, ¿cómo estás?")
        self.esperar(["buenos días", "¿cómo estás?"], "Buenos días, ¿cómo estás?")

    def test_cortesia_sola(self):
        self.esperar(["gracias"], "Gracias")

    def test_pregunta_sola(self):
        self.esperar(["¿cómo estás?"], "¿Cómo estás?")
        self.esperar(["¿quién?"], "¿Quién?")

    def test_plantillas_de_afirmacion_y_negacion_son_inalcanzables(self):
        # HALLAZGO: ninguna de las 50 señas recibe la categoría «afirmacion» ni «negacion»,
        # así que esas dos plantillas y las ramas «sí» / «no» de _apply_grammar_rules
        # no se pueden ejercitar con el vocabulario real. Código muerto declarado.
        con_esas_cats = [k for k, v in gm.GRAMMAR_CATEGORIES.items()
                         if v["cat"] in ("afirmacion", "negacion")]
        self.assertEqual(con_esas_cats, [])

    def test_tres_saludos_ya_no_usan_plantilla_pero_dan_lo_mismo(self):
        # ("saludo","saludo","saludo") no está en la tabla: cae en las reglas generales,
        # que también unen los saludos con coma.
        self.esperar(["hola", "buenos días", "buenas tardes"],
                     "Hola, buenos días, buenas tardes")


# ===========================================================================
# 5. Reglas generales: sujeto, verbo, objeto, adjetivo, adverbio, cortesía
# ===========================================================================

class TestReglasGenerales(_Base):

    def test_conjugacion_por_persona_del_primer_sujeto(self):
        self.esperar(["yo", "comer"], "Yo como")
        self.esperar(["tú", "comer"], "Tú comes")
        self.esperar(["hermano", "comer"], "El hermano come")
        self.esperar(["yo", "trabajar"], "Yo trabajo")
        self.esperar(["tú", "vivir"], "Tú vives")
        self.esperar(["yo", "sentir"], "Yo siento")

    def test_articulo_definido_segun_el_genero_declarado(self):
        self.esperar(["hermano", "trabajar"], "El hermano trabaja")
        self.esperar(["mujer", "trabajar"], "La mujer trabaja")
        self.esperar(["familia", "vivir", "bien"], "La familia vive bien")
        self.esperar(["abuelo", "comer"], "El abuelo come")

    def test_los_pronombres_no_llevan_articulo(self):
        self.esperar(["yo", "vivir"], "Yo vivo")
        self.esperar(["tú", "vivir"], "Tú vives")
        # «ustedes» y «todos» tampoco, pero por no tener género declarado, no por ser pronombres.
        self.esperar(["ustedes", "vivir"], "Ustedes vive")

    def test_adverbio_de_frecuencia_antes_del_verbo(self):
        self.esperar(["yo", "nunca", "comer"], "Yo nunca como")
        self.esperar(["hermano", "nunca", "trabajar"], "El hermano nunca trabaja")

    def test_adverbios_de_cantidad_y_estado_despues(self):
        self.esperar(["yo", "trabajar", "mucho"], "Yo trabajo mucho")
        self.esperar(["yo", "comer", "poco"], "Yo como poco")
        self.esperar(["familia", "vivir", "bien"], "La familia vive bien")

    def test_objeto_con_articulo(self):
        self.esperar(["yo", "comer", "hambre"], "Yo como la hambre")
        self.esperar(["tú", "trabajar", "nombre"], "Tú trabajas el nombre")

    def test_saludo_al_principio_y_cortesia_al_final_sin_coma(self):
        self.esperar(["hola", "yo", "trabajar"], "Hola yo trabajo")
        self.esperar(["hola", "gracias"], "Hola gracias")
        self.esperar(["por favor", "comer"], "Come por favor")
        self.esperar(["hola", "yo", "comer", "gracias"], "Hola yo como gracias")

    def test_varios_saludos_o_varias_cortesias_se_unen_con_coma(self):
        self.esperar(["gracias", "por favor", "con gusto"], "Gracias, por favor, con gusto")

    def test_orden_de_emision_entre_objeto_adjetivo_y_adverbio_pospuesto(self):
        # Añadido en la verificación independiente del 16-sep-2026: el orden relativo del
        # tramo final (objetos → adjetivos → adverbios pospuestos → cortesía) no estaba
        # fijado por ninguna prueba, y mutaciones que lo intercambiaban pasaban sin fallar.
        # Salidas obtenidas ejecutando el módulo, como todas las de este archivo.
        self.esperar(["yo", "hambre", "feliz"], "Yo la hambre feliz")    # objeto antes que adjetivo
        self.esperar(["yo", "hambre", "mucho"], "Yo la hambre mucho")    # objeto antes que adverbio
        self.esperar(["yo", "feliz", "mucho"], "Yo feliz mucho")         # adjetivo antes que adverbio
        self.esperar(["tú", "nombre", "bien"], "Tú el nombre bien")
        self.esperar(["yo", "comer", "nombre", "feliz", "mucho"], "Yo como el nombre feliz mucho")
        self.esperar(["yo", "comer", "hambre", "feliz", "mucho", "gracias"],
                     "Yo como la hambre feliz mucho gracias")


# ===========================================================================
# 6. Preguntas
# ===========================================================================

class TestPreguntas(_Base):
    """Con un interrogativo el orden de salida es otro: ¿interrogativo + verbo + sujeto +
    objetos + adjetivos + adverbios?"""

    def test_interrogativo_mas_verbo_con_pronombre_omitido(self):
        self.esperar(["¿dónde?", "tú", "vivir"], "¿Dónde vives?")
        self.esperar(["¿qué?", "tú", "comer"], "¿Qué comes?")
        self.esperar(["¿cuándo?", "yo", "comer"], "¿Cuándo como?")
        self.esperar(["¿cómo?", "tú", "sentir"], "¿Cómo sientes?")

    def test_interrogativo_sin_sujeto_usa_tercera_persona(self):
        self.esperar(["¿quién?", "comer"], "¿Quién come?")
        self.esperar(["¿quién?", "comer", "mucho"], "¿Quién come mucho?")

    def test_sujeto_no_pronominal_va_despues_del_verbo(self):
        self.esperar(["¿dónde?", "hermano", "vivir"], "¿Dónde vive el hermano?")
        self.esperar(["¿dónde?", "familia", "vivir"], "¿Dónde vive la familia?")

    def test_objeto_adjetivo_y_adverbio_en_la_pregunta(self):
        self.esperar(["¿qué?", "nombre"], "¿Qué el nombre?")               # AGRAMATICAL
        self.esperar(["¿dónde?", "tú", "vivir", "bien"], "¿Dónde vives bien?")
        self.esperar(["¿cómo?", "tú", "feliz"], "¿Cómo feliz?")            # AGRAMATICAL

    def test_pregunta_de_frase_completa_ignora_todo_lo_demas(self):
        self.esperar(["¿cómo estás?"], "¿Cómo estás?")
        self.esperar(["tú", "¿cómo estás?"], "¿Cómo estás?")               # el sujeto se pierde
        self.esperar(["¿cómo estás?", "yo", "comer"], "¿Cómo estás?")      # todo se pierde

    def test_saludo_antes_de_la_pregunta_pero_sin_coma_ni_mayuscula(self):
        self.esperar(["hola", "¿dónde?", "tú", "vivir"], "Hola ¿dónde vives?")

    def test_mayuscula_tras_el_signo_de_apertura(self):
        self.esperar(["¿por qué?", "tú", "trabajar"], "¿Por qué trabajas?")

    def test_orden_de_emision_completo_dentro_de_la_pregunta(self):
        # Mismo hueco que en las reglas generales, en la rama de _build_question:
        # objetos → adjetivos → adverbios. Añadido el 16-sep-2026 en la verificación.
        self.esperar(["¿cómo?", "tú", "nombre", "feliz"], "¿Cómo el nombre feliz?")
        self.esperar(["¿qué?", "tú", "comer", "nombre", "feliz", "bien"],
                     "¿Qué comes el nombre feliz bien?")


# ===========================================================================
# 7. Fuera de vocabulario: descarte silencioso
# ===========================================================================

class TestFueraDeVocabulario(_Base):
    """Toda glosa que no esté en GRAMMAR_CATEGORIES recibe cat «desconocido», ninguna regla
    la recoge y desaparece SIN aviso ni marca en la salida. No hay excepción, ni valor de
    retorno distinto, ni campo de diagnóstico."""

    def test_una_glosa_desconocida_sola_devuelve_cadena_vacia(self):
        self.esperar(["perro"], "")
        self.esperar(["casa"], "")
        self.esperar(["perro", "gato"], "")

    def test_la_glosa_desconocida_desaparece_del_enunciado(self):
        self.esperar(["yo", "perro", "comer"], "Yo como")
        self.esperar(["yo", "comer", "perro"], "Yo como")
        self.esperar(["yo", "casa", "trabajar", "rápido"], "Yo trabajo")

    def test_la_salida_es_identica_a_la_de_la_secuencia_sin_la_glosa_desconocida(self):
        # No hay forma, mirando la salida, de saber que se descartó algo.
        self.assertEqual(self.m.translate(["yo", "perro", "comer"]),
                         self.m.translate(["yo", "comer"]))

    def test_id_fuera_de_rango_tambien_se_descarta(self):
        # id_to_sign devuelve «[seña_999]», que tampoco está en la tabla de categorías.
        self.esperar([999], "")
        self.esperar([50], "")
        self.esperar([-1], "")
        self.esperar([30, 999], "Hola")

    def test_no_se_levanta_ninguna_excepcion(self):
        # Contraste explícito para el rediseño 3.a, que sí debería poder rechazar.
        self.assertEqual(self.m.translate(["glosa que no existe"]), "")

    def test_tipos_no_previstos_si_rompen(self):
        # HALLAZGO: la entrada solo se valida implícitamente (isinstance int, si no .lower()).
        for valor in (None, 3.0, ["yo"]):
            with self.subTest(valor=valor):
                with self.assertRaises(AttributeError):
                    self.m.translate([valor])

    def test_los_booleanos_no_rompen_sino_que_se_traducen_como_ids(self):
        # HALLAZGO: isinstance(True, int) es True en Python, así que un booleano entra por la
        # rama de los IDs y se traduce como la seña 1 / la seña 0 en vez de rechazarse.
        self.assertTrue(isinstance(True, int) and isinstance(False, int))
        self.esperar([True], "Buenos días")                           # id 1 = «buenos días»
        self.esperar([False], "Gracias")                              # id 0 = «gracias»
        self.esperar([True, "hola"], "Buenos días, hola")
        self.esperar(["yo", True], "Buenos días yo")


# ===========================================================================
# 8. Lista vacía
# ===========================================================================

class TestListaVacia(_Base):

    def test_lista_vacia(self):
        self.esperar([], "")
        self.assertEqual(self.m.translate_ids([]), "")

    def test_cadenas_vacias_o_de_espacios_equivalen_a_vacio(self):
        self.esperar([""], "")
        self.esperar(["   "], "")
        self.esperar(["yo", ""], "Yo")


# ===========================================================================
# 9. Mayúsculas, espacios y acentos
# ===========================================================================

class TestMayusculasYAcentos(_Base):
    """La normalización de entrada es exactamente s.lower().strip(): baja mayúsculas y quita
    espacios de los extremos. NO quita acentos ni normaliza Unicode."""

    def test_mayusculas_se_normalizan(self):
        self.esperar(["HOLA"], "Hola")
        self.esperar(["Hola"], "Hola")
        self.esperar(["YO", "COMER"], "Yo como")
        self.esperar(["Yo", "Comer"], "Yo como")
        self.esperar(["Buenos Días"], "Buenos días")
        self.esperar(["MÁS O MENOS"], "Más o menos")

    def test_mayusculas_acentuadas_tambien(self):
        self.esperar(["TÚ", "VIVIR"], "Tú vives")
        self.esperar(["¿DÓNDE?", "TÚ", "VIVIR"], "¿Dónde vives?")
        self.esperar(["NIÑA", "CONTENTO"], "La niña contento")

    def test_espacios_en_los_extremos_se_recortan(self):
        self.esperar(["  hola  "], "Hola")
        self.esperar([" yo ", " comer "], "Yo como")

    def test_sin_acento_la_glosa_deja_de_reconocerse(self):
        # HALLAZGO: no hay normalización de acentos. Cada caso cae en el descarte silencioso.
        for sin_acento, con_acento, salida in [
            ("adios", "adiós", "Adiós"),
            ("perdon", "perdón", "Perdón"),
            ("mas o menos", "más o menos", "Más o menos"),
            ("¿que?", "¿qué?", "¿Qué?"),
            ("¿donde?", "¿dónde?", "¿Dónde?"),
            ("nina", "niña", "La niña"),
        ]:
            with self.subTest(glosa=sin_acento):
                self.assertEqual(self.m.translate([con_acento]), salida)
                self.assertEqual(self.m.translate([sin_acento]), "")

    def test_perder_el_acento_del_sujeto_cambia_la_persona_del_verbo(self):
        # «tu» se descarta, el verbo se queda sin sujeto y sale en 3.ª persona.
        self.esperar(["tú", "vivir"], "Tú vives")
        self.esperar(["tu", "vivir"], "Vive")

    def test_la_salida_conserva_la_grafia_del_vocabulario(self):
        # La capitalización solo toca la primera letra (o la que sigue a «¿»).
        self.esperar(["niña", "contento"], "La niña contento")
        self.esperar(["¿cómo estás?"], "¿Cómo estás?")


# ===========================================================================
# 10. Modo oración del prototipo (translator.py)
# ===========================================================================

class TestModoOracionDelPrototipo(_Base):
    """src/translator.py llama a GrammarMapper de dos maneras:
      - modo aislado: translate([ens.clase]) con UNA glosa por ventana aceptada;
      - modo oración: fin_de_turno() acumula las glosas aceptadas del turno y llama
        translate(glosas) una sola vez.
    En ambos casos las glosas provienen de SIGN_VOCABULARY, nunca de texto libre.
    Estas son las salidas que hoy llegarían al TTS."""

    def test_modo_aislado_las_50_glosas(self):
        for sid in range(50):
            with self.subTest(id=sid):
                clase = gm.SIGN_VOCABULARY[sid]
                self.assertEqual(self.m.translate([clase]), SALIDA_DE_UNA_SOLA_GLOSA[sid])

    def test_turnos_tipicos_de_dos_y_tres_glosas(self):
        for glosas, salida in [
            (["hola", "¿cómo estás?"], "Hola, ¿cómo estás?"),
            (["yo", "comer"], "Yo como"),
            (["hermano", "trabajar"], "El hermano trabaja"),
            (["tú", "trabajar", "mucho"], "Tú trabajas mucho"),
            (["familia", "vivir", "bien"], "La familia vive bien"),
            (["¿dónde?", "tú", "vivir"], "¿Dónde vives?"),
            (["hola", "yo", "trabajar"], "Hola yo trabajo"),
            (["yo", "trabajar", "gracias"], "Yo trabajo gracias"),
        ]:
            with self.subTest(glosas=glosas):
                self.assertEqual(self.m.translate(list(glosas)), salida)

    def test_turno_desde_los_ids_del_modelo(self):
        self.esperar([30, 26], "Hola, ¿cómo estás?")
        self.esperar([18, 6], "El hermano trabaja")
        self.esperar([38, 7], "Yo como")
        self.esperar([39, 43, 8], "¿Dónde vives?")
        self.esperar([30, 38, 6, 0], "Hola yo trabajo gracias")

    def test_mezcla_de_ids_y_nombres(self):
        self.esperar([38, "comer"], "Yo como")
        self.esperar(["yo", 7], "Yo como")

    def test_translate_no_muta_la_lista_de_glosas_del_turno(self):
        # translator.py pasa list(self.glosas_turno) y luego lo registra en el log.
        glosas = ["hola", "yo", "comer"]
        copia = list(glosas)
        self.m.translate(glosas)
        self.assertEqual(glosas, copia)

    def test_turno_largo_pierde_casi_todo(self):
        # HALLAZGO: un turno con las 50 señas sale reducido a los saludos y la pregunta
        # (6 glosas); las otras 44 se descartan sin aviso.
        todas = [gm.SIGN_VOCABULARY[i] for i in range(50)]
        self.esperar(todas, "Buenos días, buenas tardes, buenas noches, hola, adiós, ¿cómo estás?")

    def test_el_mapeador_es_sin_estado_entre_turnos(self):
        self.assertEqual(self.m.translate(["yo", "comer"]), "Yo como")
        self.assertEqual(self.m.translate(["tú", "vivir"]), "Tú vives")
        self.assertEqual(self.m.translate(["yo", "comer"]), "Yo como")


# ===========================================================================
# 11. Hallazgos: lo que HOY sale mal, clavado tal cual
# ===========================================================================

class TestHallazgos(_Base):
    """Salidas AGRAMATICALES o con pérdida de información. Se prueban a propósito: son el
    estado actual, no el objetivo. Cada una está explicada en
    docs/estado_mapeo_gramatical.md. NO corregir el módulo para que estas pruebas cambien:
    el rediseño es la actividad 3.a."""

    # --- H1. Sin concordancia de número ---
    def test_h1_sin_concordancia_de_numero(self):
        self.esperar(["ustedes", "comer"], "Ustedes come")            # debería ser «comen»
        self.esperar(["todos", "comer", "mucho"], "Todos come mucho")
        self.esperar(["personas", "trabajar"], "Personas trabaja")
        self.esperar(["ustedes", "vivir", "bien"], "Ustedes vive bien")
        # La causa: CONJUGATIONS solo tiene personas 1/2/3 y no mira «numero».
        self.assertEqual(set(gm.CONJUGATIONS["comer"]), {1, 2, 3})
        self.assertEqual(gm.GRAMMAR_CATEGORIES["ustedes"]["numero"], "plural")

    # --- H2. Sin concordancia de género ni número en el adjetivo ---
    def test_h2_sin_concordancia_en_el_adjetivo(self):
        self.esperar(["niña", "contento"], "La niña contento")        # debería ser «contenta»
        self.esperar(["mujer", "aburrido"], "La mujer aburrido")
        self.esperar(["ustedes", "bienvenido"], "Ustedes bienvenido")
        # La causa: los adjetivos no declaran género ni número.
        self.assertNotIn("genero", gm.GRAMMAR_CATEGORIES["contento"])

    # --- H3. Sin cópula ---
    def test_h3_sin_copula_ser_estar(self):
        self.esperar(["yo", "feliz"], "Yo feliz")                     # debería ser «estoy feliz»
        self.esperar(["tú", "triste"], "Tú triste")
        self.esperar(["hermano", "juicioso"], "El hermano juicioso")
        self.esperar(["niña", "feliz"], "La niña feliz")

    # --- H4. Objeto sin verbo, y artículo incorrecto ---
    def test_h4_objeto_sin_verbo_y_articulo(self):
        self.esperar(["yo", "hambre"], "Yo la hambre")   # «tengo hambre»; y «la» por «el»
        self.esperar(["yo", "permiso"], "Yo el permiso")
        self.esperar(["tú", "nombre"], "Tú el nombre")
        self.assertEqual(gm.GRAMMAR_CATEGORIES["hambre"]["genero"], "f")

    # --- H5. Solo cuenta el primer sujeto; los demás se descartan ---
    def test_h5_solo_el_primer_sujeto(self):
        self.esperar(["hermano", "tío", "comer"], "El hermano come")  # «tío» desaparece
        self.esperar(["yo", "tú", "comer"], "Yo como")
        self.assertEqual(self.m.translate(["hermano", "tío", "comer"]),
                         self.m.translate(["hermano", "comer"]))

    # --- H6. Solo se conjuga el primer verbo ---
    def test_h6_el_segundo_verbo_sale_en_infinitivo(self):
        self.esperar(["yo", "comer", "trabajar"], "Yo como trabajar")
        self.esperar(["yo", "comer", "trabajar", "vivir"], "Yo como trabajar vivir")
        self.esperar(["yo", "comer", "comer"], "Yo como comer")       # ni la repetición se filtra

    # --- H7a. Dentro de una misma categoría el orden de entrada SÍ manda ---
    def test_h7a_el_orden_manda_dentro_de_la_misma_categoria(self):
        # Primera de las dos vías por las que el orden de entrada llega a la salida: las
        # categorías se recogen con comprensiones de lista, que conservan el orden.
        self.esperar(["hola", "buenos días"], "Hola, buenos días")
        self.esperar(["buenos días", "hola"], "Buenos días, hola")
        self.esperar(["gracias", "por favor"], "Gracias, por favor")
        self.esperar(["por favor", "gracias"], "Por favor, gracias")
        self.esperar(["yo", "seña", "nombre"], "Yo la seña el nombre")
        self.esperar(["yo", "nombre", "seña"], "Yo el nombre la seña")

    # --- H7b. Saludo + interrogativo: el orden manda ENTRE categorías distintas ---
    def test_h7b_el_orden_manda_entre_saludo_e_interrogativo(self):
        # Segunda vía, y la que desmiente que «el orden solo importa dentro de una categoría»:
        # la plantilla ("saludo","pregunta") casa SOLO en ese orden y une con coma; invertida
        # no casa, se va a _apply_grammar_rules y el prefijo se une con espacio (sin coma).
        self.esperar(["hola", "¿dónde?"], "Hola, ¿dónde?")
        self.esperar(["¿dónde?", "hola"], "Hola ¿dónde?")            # se pierde la coma
        self.esperar(["buenos días", "¿qué?"], "Buenos días, ¿qué?")
        self.esperar(["¿qué?", "buenos días"], "Buenos días ¿qué?")
        self.esperar(["adiós", "¿quién?"], "Adiós, ¿quién?")
        self.esperar(["¿quién?", "adiós"], "Adiós ¿quién?")
        # Barrido de los 5 saludos × 6 interrogativos: los 30 pares son sensibles al orden.
        saludos = [s for s, i in gm.GRAMMAR_CATEGORIES.items() if i["cat"] == "saludo"]
        interrogativos = [s for s, i in gm.GRAMMAR_CATEGORIES.items()
                          if i["cat"] == "pregunta" and i.get("tipo") == "interrogativo"]
        self.assertEqual((len(saludos), len(interrogativos)), (5, 6))
        for sal in saludos:
            for interr in interrogativos:
                with self.subTest(saludo=sal, interrogativo=interr):
                    self.assertNotEqual(self.m.translate([sal, interr]),
                                        self.m.translate([interr, sal]))
        # Excepción: «¿cómo estás?» es de tipo frase_completa y esa rama TAMBIÉN une con coma,
        # así que sus 5 pares con saludo son los únicos insensibles al orden. Por eso son 30
        # pares cruzados y no 35.
        for sal in saludos:
            with self.subTest(saludo=sal):
                self.assertEqual(self.m.translate([sal, "¿cómo estás?"]),
                                 self.m.translate(["¿cómo estás?", sal]))
        self.esperar(["hola", "¿cómo estás?"], "Hola, ¿cómo estás?")
        self.esperar(["¿cómo estás?", "hola"], "Hola, ¿cómo estás?")

    # --- H7c. La sensibilidad cruzada solo existe en turnos de DOS glosas ---
    def test_h7c_el_cruce_de_categorias_solo_afecta_a_turnos_de_dos(self):
        # _try_templates compara la tupla de categorías del turno ENTERO, así que basta una
        # tercera glosa para que la plantilla deje de casar y el par vuelva a dar igual.
        self.esperar(["hola", "¿dónde?"], "Hola, ¿dónde?")
        self.esperar(["hola", "¿dónde?", "tú"], "Hola ¿dónde?")      # «tú» además se pierde
        self.esperar(["¿dónde?", "hola", "tú"], "Hola ¿dónde?")
        self.esperar(["tú", "¿dónde?", "hola"], "Hola ¿dónde?")
        for tercia in (["hola", "¿dónde?", "tú"], ["hola", "¿cómo estás?", "yo"]):
            with self.subTest(tercia=tercia):
                salidas = {self.m.translate(list(p)) for p in itertools.permutations(tercia)}
                self.assertEqual(len(salidas), 1)
        # Medido por barrido exhaustivo: de las 12.572 tercias de señas distintas cuyas tres
        # categorías son distintas entre sí, 0 son sensibles al orden de entrada.

    # --- H8. En las preguntas se pierde la cortesía y el segundo interrogativo ---
    def test_h8_la_pregunta_descarta_cortesia_y_otros_interrogativos(self):
        # _build_question se invoca antes de añadir la cortesía y solo mira preguntas[0].
        self.esperar(["gracias", "¿cómo estás?"], "¿Cómo estás?")     # «gracias» desaparece
        self.esperar(["¿dónde?", "tú", "vivir", "por favor"], "¿Dónde vives?")
        self.esperar(["¿qué?", "¿dónde?", "tú", "vivir"], "¿Qué vives?")

    # --- H9. Puntuación y mayúsculas dentro del enunciado ---
    def test_h9_puntuacion_incompleta(self):
        # Sin coma entre el saludo y la oración, sin punto final, y el «¿» interno no
        # dispara mayúscula porque _capitalize solo mira el primer carácter.
        self.esperar(["hola", "yo", "trabajar"], "Hola yo trabajo")
        self.esperar(["hola", "¿dónde?", "tú", "vivir"], "Hola ¿dónde vives?")
        for glosas in (["yo", "comer"], ["hola"], ["hermano", "trabajar"]):
            with self.subTest(glosas=glosas):
                self.assertFalse(self.m.translate(list(glosas)).endswith("."))

    # --- H10. Sujeto plural pospuesto en preguntas ---
    def test_h10_sujeto_plural_pospuesto_en_preguntas(self):
        self.esperar(["¿por qué?", "ustedes", "trabajar"], "¿Por qué trabaja ustedes?")

    # --- H11. Ramas de código inalcanzables con el vocabulario real ---
    def test_h11_codigo_inalcanzable(self):
        cats = gm.GRAMMAR_CATEGORIES
        # (a) «gustar»: rama especial de _apply_grammar_rules; no es una de las 50 señas.
        self.assertNotIn("gustar", cats)
        self.assertNotIn("gustar", set(gm.SIGN_VOCABULARY.values()))
        # (b) PREPOSITIONS solo se usa si un objeto tiene tipo «lugar»: ninguno lo tiene.
        self.assertEqual([k for k, v in cats.items()
                          if v["cat"] == "objeto" and v.get("tipo") == "lugar"], [])
        # (c) La contracción «al» exige además un verbo de tipo «movimiento»: no hay.
        self.assertEqual([k for k, v in cats.items()
                          if v["cat"] == "verbo" and v.get("tipo") == "movimiento"], [])
        # (d) El artículo indefinido de ARTICLES nunca se pide: _get_article solo lee «def».
        self.assertEqual(sorted(gm.ARTICLES["m"]), ["def", "indef"])
        self.assertEqual(self.m._get_article({"genero": "m"}), "el")
        # (e) La conjugación por terminación nunca se alcanza: los 4 verbos están en la tabla.
        self.assertEqual([k for k, v in cats.items()
                          if v["cat"] == "verbo" and k not in gm.CONJUGATIONS], [])

    # --- H12. Los rasgos declarados que nadie lee ---
    def test_h12_rasgos_declarados_sin_efecto(self):
        # «numero» no lo lee nadie (no aparece ni una vez en el cuerpo de la clase) y los
        # «tipo» de verbo, objeto y adjetivo tampoco cambian ninguna salida hoy.
        self.assertEqual(gm.GRAMMAR_CATEGORIES["vivir"]["tipo"], "estado")
        self.assertEqual(gm.GRAMMAR_CATEGORIES["comer"]["tipo"], "accion")
        self.esperar(["hermano", "vivir"], "El hermano vive")
        self.esperar(["hermano", "comer"], "El hermano come")
        # Los DOS «tipo» que SÍ tienen efecto observable. Corregido el 16-sep-2026 en la
        # verificación independiente: esta prueba decía que el de los adverbios era el único,
        # y era falso — el de las preguntas manda en la rama de _build_question, que es de lo
        # que depende la excepción de «¿cómo estás?» en H7b.
        # (a) «frecuencia» en adverbios: antepone el adverbio al verbo.
        self.esperar(["yo", "nunca", "comer"], "Yo nunca como")
        self.esperar(["yo", "mucho", "comer"], "Yo como mucho")
        # (b) «frase_completa» en preguntas: toma la rama que descarta todo lo demás.
        self.assertEqual(gm.GRAMMAR_CATEGORIES["¿cómo estás?"]["tipo"], "frase_completa")
        self.assertEqual(gm.GRAMMAR_CATEGORIES["¿dónde?"]["tipo"], "interrogativo")
        self.esperar(["¿cómo estás?", "yo", "comer"], "¿Cómo estás?")
        self.esperar(["¿dónde?", "yo", "comer"], "¿Dónde como?")


# ===========================================================================
# 12. Auxiliares internos (documentan el mecanismo, no la API pública)
# ===========================================================================

class TestAuxiliaresInternos(_Base):

    def test_conjugate_desde_la_tabla(self):
        self.assertEqual(self.m._conjugate("comer", 1), "como")
        self.assertEqual(self.m._conjugate("comer", 2), "comes")
        self.assertEqual(self.m._conjugate("comer", 3), "come")
        self.assertEqual(self.m._conjugate("sentir", 1), "siento")

    def test_conjugate_por_terminacion(self):
        # Rama inalcanzable con las 50 señas (ver H11e), pero es el respaldo del módulo.
        self.assertEqual(self.m._conjugate("bailar", 1), "bailo")
        self.assertEqual(self.m._conjugate("bailar", 2), "bailas")
        self.assertEqual(self.m._conjugate("bailar", 3), "baila")
        self.assertEqual(self.m._conjugate("beber", 2), "bebes")
        self.assertEqual(self.m._conjugate("subir", 1), "subo")
        self.assertEqual(self.m._conjugate("sol", 1), "sol")       # no termina en ar/er/ir

    def test_conjugate_con_persona_fuera_de_1_2_3(self):
        # HALLAZGO: incoherente entre las dos ramas. Persona solo vale 1/2/3 hoy.
        self.assertEqual(self.m._conjugate("comer", 9), "comer")   # tabla → el infinitivo
        self.assertEqual(self.m._conjugate("bailar", 9), "baila")  # terminación → 3.ª persona

    def test_get_article(self):
        self.assertEqual(self.m._get_article({"genero": "m"}), "el")
        self.assertEqual(self.m._get_article({"genero": "f"}), "la")
        self.assertEqual(self.m._get_article({}), "")
        self.assertEqual(self.m._get_article({"genero": "n"}), "")

    def test_capitalize(self):
        self.assertEqual(self.m._capitalize(""), "")
        self.assertEqual(self.m._capitalize("hola"), "Hola")
        self.assertEqual(self.m._capitalize("¿dónde vives?"), "¿Dónde vives?")
        self.assertEqual(self.m._capitalize("¿"), "¿")
        self.assertEqual(self.m._capitalize("hola ¿qué?"), "Hola ¿qué?")   # el «¿» interno, no

    def test_tag_tokens_marca_lo_desconocido(self):
        etiquetas = self.m._tag_tokens(["yo", "perro"])
        self.assertEqual(etiquetas[0]["cat"], "sujeto")
        self.assertEqual(etiquetas[1], {"word": "perro", "cat": "desconocido"})

    def test_format_object(self):
        self.assertEqual(self.m._format_object({"word": "nombre", "genero": "m"}, []), "el nombre")
        self.assertEqual(self.m._format_object({"word": "seña", "genero": "f"}, []), "la seña")
        self.assertEqual(self.m._format_object({"word": "x"}, []), "x")


# ===========================================================================
# 13. El CLI de referencia del módulo
# ===========================================================================

class TestCliDeReferencia(_Base):
    """main() imprime 25 casos con la salida registrada el 13-sep-2026 y cuenta cuántos
    siguen coincidiendo. Sirve de segunda red: si la lógica cambia, el contador baja."""

    def test_main_reproduce_sus_25_casos_registrados(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gm.main()
        salida = buf.getvalue()
        self.assertIn("25/25 salidas coinciden con las registradas el 13-sep-2026", salida)
        self.assertNotIn("(registrada el 13-sep-2026:", salida)   # ninguna discrepancia
        for sid in range(50):                                     # imprime el vocabulario entero
            self.assertIn(gm.SIGN_VOCABULARY[sid], salida)


if __name__ == "__main__":
    unittest.main(verbosity=2)
