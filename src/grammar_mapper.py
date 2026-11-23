"""
Conversión de glosas LSC50 a enunciados en español (estado: exploración de marzo de 2026).

QUÉ HACE HOY ESTE MÓDULO (descripción fiel al código, no a la intención original)
  - Léxico de las 50 señas de LSC50 (SIGN_VOCABULARY, verificado contra la Tabla 2 del artículo
    del corpus) con UNA categoría fija por seña y unos pocos rasgos (GRAMMAR_CATEGORIES: persona
    y número en sujetos, género en algunos sustantivos, tipo en verbos, objetos y adverbios).
  - Siete plantillas para secuencias de una o dos categorías (saludo, cortesía, pregunta; las dos
    de afirmación y negación solas no se alcanzan: ninguna seña recibe esas categorías).
  - Si no aplica plantilla, un ORDEN DE EMISIÓN FIJO POR CATEGORÍA: saludos → primer sujeto (con
    artículo definido según su género) → «no» (categoría negación, hoy sin seña) → adverbios de
    frecuencia → primer verbo conjugado en presente según la persona del sujeto (cuatro verbos
    en tabla, el resto por terminación) → objetos (con artículo) → adjetivos → demás adverbios →
    cortesía. Las preguntas con interrogativo usan otro orden fijo: ¿interrogativo + verbo +
    sujeto + objetos + adjetivos + adverbios?
  - El orden de ENTRADA se ignora: ["yo", "comer"] y ["comer", "yo"] producen «Yo como».
  - No hay concordancia de número («Ustedes come»), ni de género en adjetivos («La niña
    contento»), ni cópula («Yo feliz»); no hay más tiempo que el presente; solo cuenta el primer
    sujeto; solo el primer verbo se conjuga y los siguientes salen en infinitivo («Yo como
    trabajar»); las glosas fuera del vocabulario se descartan en silencio.
  Es, por tanto, una tabla de categorías con un orden de salida fijo, no un procedimiento
  gramatical, y no tiene pruebas con salida esperada.

FUNDAMENTO LINGÜÍSTICO
  La versión anterior de este docstring afirmaba que «la LSC usa orden tópico-comentario / SOV,
  mientras que el español usa SVO». Esa premisa fue retirada del anteproyecto (§2.2.5). La
  caracterización correcta es la de Oviedo (2001), «Apuntes para una gramática de la Lengua de
  Señas Colombiana»:
  - en tareas de elicitación «el orden es siempre sujeto-verbo-objeto» (p. 275);
  - «en discurso, el orden se hace muy flexible» (p. 283);
  - el tópico-comentario es «un ordenamiento sintáctico distinto a la oración» —una unidad
    sintáctica, no una clasificación tipológica de la lengua— (pp. 278-279);
  - los verbos marcan persona, rol temático, aspecto y modo, pero no tiempo, que se expresa
    léxicamente (p. 199).
  En consecuencia, este componente no se justifica por un contraste de orden entre la LSC y el
  español, sino porque la glosa reconocida es una etiqueta léxica sin las categorías que el
  español exige para un enunciado bien formado (concordancia, artículos, cópula, tiempo), y el
  procedimiento debe suplirlas de manera documentada y verificable (anteproyecto, §2.2.5).

PENDIENTE — rearquitectura como procedimiento, mes 3 del cronograma
  Concordancia de número y género, cópula ser/estar, tratamiento declarado de los casos no
  cubiertos (segundo sujeto, segundo verbo, fuera de vocabulario, verbos direccionales), rechazo
  explícito en vez de descarte silencioso, acumulación de varias glosas por turno y un conjunto
  de pruebas con salida esperada verificado con la persona señante o intérprete. Hasta entonces
  la lógica de este módulo NO se modifica: el 13-sep-2026 solo se corrigieron este docstring y
  los ejemplos de main().

Uso (salidas REALES del código, obtenidas el 13-sep-2026):
    from grammar_mapper import GrammarMapper

    mapper = GrammarMapper()
    mapper.translate(["hermano", "trabajar"])     # → "El hermano trabaja"
    mapper.translate(["¿dónde?", "tú", "vivir"])  # → "¿Dónde vives?"
    mapper.translate_ids([18, 6])                 # → "El hermano trabaja"  (18 = hermano, 6 = trabajar)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Vocabulario LSC50 — 50 señas extraídas de Table 1 / Figure 1 del paper:
# Flórez-Sierra et al. (2024) "LSC50: Colombian Sign Language Video and
# Inertial Measurement dataset", Scientific Data, Nature.
#
# sign_id (0-49) → nombre de la seña
# Agrupadas en 7 campos semánticos según el paper.
# ---------------------------------------------------------------------------

SIGN_VOCABULARY: dict[int, str] = {
    # Mapeo verificado desde Timestamps.xlsx del dataset LSC50
    0: "gracias",
    1: "buenos días",
    2: "buenas tardes",
    3: "buenas noches",
    4: "seña",
    5: "nombre",
    6: "trabajar",
    7: "comer",
    8: "vivir",
    9: "poco",
    10: "familia",
    11: "personas",
    12: "mujer",
    13: "hombre",
    14: "niño",
    15: "niña",
    16: "abuelo",
    17: "tío",
    18: "hermano",
    19: "hambre",
    20: "feliz",
    21: "contento",
    22: "triste",
    23: "aburrido",
    24: "bien",
    25: "mal",
    26: "¿cómo estás?",
    27: "más o menos",
    28: "sentir",
    29: "juicioso",
    30: "hola",
    31: "adiós",
    32: "por favor",
    33: "con gusto",
    34: "bienvenido",
    35: "perdón",
    36: "permiso",
    37: "nunca",
    38: "yo",
    39: "tú",
    40: "ustedes",
    41: "¿qué?",
    42: "¿cuándo?",
    43: "¿dónde?",
    44: "¿cómo?",
    45: "¿por qué?",
    46: "¿quién?",
    47: "diferente",
    48: "todos",
    49: "mucho",
}

# Mapa inverso: nombre → sign_id
SIGN_NAME_TO_ID: dict[str, int] = {v: k for k, v in SIGN_VOCABULARY.items()}


# ---------------------------------------------------------------------------
# Categorías gramaticales
# ---------------------------------------------------------------------------

# Cada seña se clasifica en una o más categorías gramaticales.
# Categorías: sujeto, verbo, objeto, adjetivo, adverbio, saludo,
#             pregunta, afirmacion, negacion, tiempo, lugar, cortesia

GRAMMAR_CATEGORIES: dict[str, dict] = {
    # --- Sujetos (pronombres y sustantivos que pueden ser sujeto) ---
    "yo":        {"cat": "sujeto", "persona": 1, "numero": "singular"},
    "tú":        {"cat": "sujeto", "persona": 2, "numero": "singular"},
    "ustedes":   {"cat": "sujeto", "persona": 3, "numero": "plural"},
    "todos":     {"cat": "sujeto", "persona": 3, "numero": "plural"},
    "hermano":   {"cat": "sujeto", "persona": 3, "numero": "singular", "genero": "m"},
    "niño":      {"cat": "sujeto", "persona": 3, "numero": "singular", "genero": "m"},
    "niña":      {"cat": "sujeto", "persona": 3, "numero": "singular", "genero": "f"},
    "tío":       {"cat": "sujeto", "persona": 3, "numero": "singular", "genero": "m"},
    "abuelo":    {"cat": "sujeto", "persona": 3, "numero": "singular", "genero": "m"},
    "mujer":     {"cat": "sujeto", "persona": 3, "numero": "singular", "genero": "f"},
    "hombre":    {"cat": "sujeto", "persona": 3, "numero": "singular", "genero": "m"},
    "personas":  {"cat": "sujeto", "persona": 3, "numero": "plural"},
    "familia":   {"cat": "sujeto", "persona": 3, "numero": "singular", "genero": "f"},
    # --- Verbos ---
    "trabajar":  {"cat": "verbo", "tipo": "accion"},
    "comer":     {"cat": "verbo", "tipo": "accion"},
    "vivir":     {"cat": "verbo", "tipo": "estado"},
    "sentir":    {"cat": "verbo", "tipo": "sentimiento"},
    # --- Objetos / Abstractos ---
    "nombre":    {"cat": "objeto", "tipo": "abstracto", "genero": "m"},
    "seña":      {"cat": "objeto", "tipo": "abstracto", "genero": "f"},
    "hambre":    {"cat": "objeto", "tipo": "estado", "genero": "f"},
    "permiso":   {"cat": "objeto", "tipo": "abstracto", "genero": "m"},
    # --- Adjetivos ---
    "feliz":     {"cat": "adjetivo", "tipo": "sentimiento"},
    "triste":    {"cat": "adjetivo", "tipo": "sentimiento"},
    "contento":  {"cat": "adjetivo", "tipo": "sentimiento"},
    "aburrido":  {"cat": "adjetivo", "tipo": "sentimiento"},
    "juicioso":  {"cat": "adjetivo", "tipo": "descripcion"},
    "diferente": {"cat": "adjetivo", "tipo": "descripcion"},
    "bienvenido": {"cat": "adjetivo", "tipo": "social"},
    # --- Adverbios ---
    "bien":      {"cat": "adverbio", "tipo": "estado"},
    "mal":       {"cat": "adverbio", "tipo": "estado"},
    "poco":      {"cat": "adverbio", "tipo": "cantidad"},
    "mucho":     {"cat": "adverbio", "tipo": "cantidad"},
    "nunca":     {"cat": "adverbio", "tipo": "frecuencia"},
    "más o menos": {"cat": "adverbio", "tipo": "estado"},
    # --- Saludos y cortesía ---
    "hola":           {"cat": "saludo"},
    "buenos días":    {"cat": "saludo"},
    "buenas tardes":  {"cat": "saludo"},
    "buenas noches":  {"cat": "saludo"},
    "adiós":          {"cat": "saludo"},
    "por favor":      {"cat": "cortesia"},
    "con gusto":      {"cat": "cortesia"},
    "perdón":         {"cat": "cortesia"},
    "gracias":        {"cat": "cortesia"},
    # --- Preguntas ---
    "¿cómo estás?":   {"cat": "pregunta", "tipo": "frase_completa"},
    "¿quién?":        {"cat": "pregunta", "tipo": "interrogativo"},
    "¿qué?":          {"cat": "pregunta", "tipo": "interrogativo"},
    "¿cuándo?":       {"cat": "pregunta", "tipo": "interrogativo"},
    "¿dónde?":        {"cat": "pregunta", "tipo": "interrogativo"},
    "¿por qué?":      {"cat": "pregunta", "tipo": "interrogativo"},
    "¿cómo?":         {"cat": "pregunta", "tipo": "interrogativo"},
}


# ---------------------------------------------------------------------------
# Conjugaciones verbales básicas (presente indicativo)
# ---------------------------------------------------------------------------

CONJUGATIONS: dict[str, dict[int, str]] = {
    "trabajar": {1: "trabajo",   2: "trabajas",  3: "trabaja"},
    "comer":    {1: "como",      2: "comes",     3: "come"},
    "vivir":    {1: "vivo",      2: "vives",     3: "vive"},
    "sentir":   {1: "siento",    2: "sientes",   3: "siente"},
}


# ---------------------------------------------------------------------------
# Artículos y preposiciones
# ---------------------------------------------------------------------------

ARTICLES: dict[str, dict[str, str]] = {
    # genero → artículo definido
    "m": {"def": "el", "indef": "un"},
    "f": {"def": "la", "indef": "una"},
}

# Preposiciones asociadas a tipos de complemento
PREPOSITIONS: dict[str, str] = {
    "lugar": "en",
    "movimiento_destino": "a",
}


# ---------------------------------------------------------------------------
# Templates para frases frecuentes
# ---------------------------------------------------------------------------

# Patrones reconocidos: tupla de categorías → función de construcción.
# Se prueban en orden; el primero que coincida se usa.

PHRASE_TEMPLATES: list[tuple[tuple[str, ...], str]] = [
    # Saludos simples
    (("saludo",), "{0}"),
    (("saludo", "saludo"), "{0}, {1}"),
    (("saludo", "pregunta"), "{0}, {1}"),
    # Cortesía sola
    (("cortesia",), "{0}"),
    # Afirmación/negación + contenido
    (("afirmacion",), "{0}"),
    (("negacion",), "{0}"),
    # Pregunta completa
    (("pregunta",), "{0}"),
]


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class GrammarMapper:
    """Convierte secuencias de señas LSC en oraciones en español."""

    def __init__(self) -> None:
        self.vocabulary = SIGN_VOCABULARY
        self.categories = GRAMMAR_CATEGORIES
        self.conjugations = CONJUGATIONS

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def id_to_sign(self, sign_id: int) -> str:
        """Convierte un ID de seña (0-49) a su nombre."""
        return self.vocabulary.get(sign_id, f"[seña_{sign_id}]")

    def ids_to_signs(self, sign_ids: list[int]) -> list[str]:
        """Convierte una lista de IDs a nombres de señas."""
        return [self.id_to_sign(sid) for sid in sign_ids]

    def translate(self, signs: list[str | int]) -> str:
        """Traduce una secuencia de señas a español.

        Args:
            signs: Lista de nombres de señas (str) o IDs (int).

        Returns:
            Oración en español con gramática corregida.
        """
        # Normalizar entrada: convertir IDs a nombres
        tokens = []
        for s in signs:
            if isinstance(s, int):
                tokens.append(self.id_to_sign(s))
            else:
                tokens.append(s.lower().strip())

        if not tokens:
            return ""

        # Etiquetar cada token con su categoría
        tagged = self._tag_tokens(tokens)

        # Intentar template directo
        template_result = self._try_templates(tagged)
        if template_result is not None:
            return self._capitalize(template_result)

        # Aplicar reglas gramaticales
        sentence = self._apply_grammar_rules(tagged)

        return self._capitalize(sentence)

    def translate_ids(self, sign_ids: list[int]) -> str:
        """Traduce directamente desde IDs de señas."""
        return self.translate(sign_ids)

    # ------------------------------------------------------------------
    # Etiquetado gramatical
    # ------------------------------------------------------------------

    def _tag_tokens(self, tokens: list[str]) -> list[dict]:
        """Etiqueta cada token con su información gramatical."""
        tagged = []
        for token in tokens:
            info = self.categories.get(token, {"cat": "desconocido"})
            tagged.append({"word": token, **info})
        return tagged

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def _try_templates(self, tagged: list[dict]) -> str | None:
        """Intenta aplicar un template predefinido."""
        cats = tuple(t["cat"] for t in tagged)

        for pattern, template in PHRASE_TEMPLATES:
            if cats == pattern:
                words = [t["word"] for t in tagged]
                return template.format(*words)

        return None

    # ------------------------------------------------------------------
    # Reglas gramaticales (SOV → SVO)
    # ------------------------------------------------------------------

    def _apply_grammar_rules(self, tagged: list[dict]) -> str:
        """Aplica reglas de reordenamiento y enriquecimiento gramatical.

        Transformaciones principales:
        1. LSC SOV → Español SVO
        2. LSC Tópico-Comentario → Español SVO + complementos
        3. Añadir artículos, preposiciones y conjugaciones
        """
        # Clasificar tokens por rol
        sujetos = [t for t in tagged if t["cat"] == "sujeto"]
        verbos = [t for t in tagged if t["cat"] == "verbo"]
        objetos = [t for t in tagged if t["cat"] == "objeto"]
        adjetivos = [t for t in tagged if t["cat"] == "adjetivo"]
        adverbios = [t for t in tagged if t["cat"] == "adverbio"]
        saludos = [t for t in tagged if t["cat"] == "saludo"]
        cortesia = [t for t in tagged if t["cat"] == "cortesia"]
        preguntas = [t for t in tagged if t["cat"] == "pregunta"]
        negacion = [t for t in tagged if t["cat"] == "negacion"]
        afirmacion = [t for t in tagged if t["cat"] == "afirmacion"]

        parts: list[str] = []

        # 1. Saludos van primero
        if saludos:
            parts.append(", ".join(s["word"] for s in saludos))

        # 2. Afirmación/negación (solo como prefijo aislado, no duplicar)
        has_negation = bool(negacion)
        if afirmacion:
            parts.append("sí")

        # 3. Preguntas interrogativas: construir oración interrogativa
        if preguntas:
            return self._build_question(preguntas, sujetos, verbos, objetos,
                                        adjetivos, adverbios, parts)

        # 4. Construir oración SVO
        persona = 3  # default tercera persona

        # Sujeto
        if sujetos:
            sujeto = sujetos[0]
            persona = sujeto.get("persona", 3)
            # Pronombres no llevan artículo
            if sujeto["word"] in ("yo", "tú"):
                parts.append(sujeto["word"])
            else:
                art = self._get_article(sujeto)
                parts.append(f"{art} {sujeto['word']}" if art else sujeto["word"])

        # Separar adverbios de frecuencia (van antes del verbo) de otros
        adv_pre = [a for a in adverbios if a.get("tipo") == "frecuencia"]
        adv_post = [a for a in adverbios if a.get("tipo") != "frecuencia"]

        # Negación antes del verbo
        if has_negation:
            parts.append("no")

        # Adverbios de frecuencia antes del verbo ("nunca como")
        for adv in adv_pre:
            parts.append(adv["word"])

        # Verbo conjugado
        if verbos:
            verbo = verbos[0]
            conjugated = self._conjugate(verbo["word"], persona)

            # "gustar" tiene estructura especial
            if verbo["word"] == "gustar":
                # Remover sujeto pronombre de parts (gustar lo absorbe)
                parts = [p for p in parts if p not in ("yo", "tú")]
                neg_prefix = "no " if has_negation else ""
                # Reinsert negación en gustar
                parts = [p for p in parts if p != "no"]
                if objetos:
                    obj = objetos[0]
                    art = self._get_article(obj)
                    obj_str = f"{art} {obj['word']}" if art else obj["word"]
                    parts.append(f"{neg_prefix}{conjugated} {obj_str}")
                    objetos = []  # ya procesado
                else:
                    parts.append(f"{neg_prefix}{conjugated}")
            else:
                parts.append(conjugated)

            # Si hay un segundo verbo, va en infinitivo
            for v in verbos[1:]:
                parts.append(v["word"])

        # Objeto (complemento directo/lugar)
        if objetos:
            for obj in objetos:
                obj_str = self._format_object(obj, verbos)
                parts.append(obj_str)

        # Adjetivos (después del sustantivo o como predicado)
        if adjetivos:
            for adj in adjetivos:
                parts.append(adj["word"])

        # Adverbios pospuestos (cantidad, estado)
        for adv in adv_post:
            parts.append(adv["word"])

        # Cortesía al final
        if cortesia:
            parts.append(", ".join(c["word"] for c in cortesia))

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Construcción de preguntas
    # ------------------------------------------------------------------

    def _build_question(self, preguntas: list[dict], sujetos: list[dict],
                        verbos: list[dict], objetos: list[dict],
                        adjetivos: list[dict], adverbios: list[dict],
                        prefix_parts: list[str]) -> str:
        """Construye oraciones interrogativas."""
        pregunta = preguntas[0]

        # Preguntas completas
        if pregunta.get("tipo") == "frase_completa":
            if prefix_parts:
                return " ".join(prefix_parts) + ", " + pregunta["word"]
            return pregunta["word"]

        # Interrogativo + oración
        # Formato: ¿Interrogativo + verbo + sujeto + complemento?
        interrogativo = pregunta["word"].replace("¿", "").replace("?", "").strip()
        parts = [f"¿{interrogativo}"]

        persona = 3
        if sujetos:
            persona = sujetos[0].get("persona", 3)

        if verbos:
            parts.append(self._conjugate(verbos[0]["word"], persona))

        if sujetos:
            suj = sujetos[0]
            if suj["word"] not in ("yo", "tú"):
                art = self._get_article(suj)
                parts.append(f"{art} {suj['word']}" if art else suj["word"])

        if objetos:
            for obj in objetos:
                parts.append(self._format_object(obj, verbos))

        if adjetivos:
            for adj in adjetivos:
                parts.append(adj["word"])

        if adverbios:
            for adv in adverbios:
                parts.append(adv["word"])

        result = " ".join(parts) + "?"
        if prefix_parts:
            return " ".join(prefix_parts) + " " + result
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _conjugate(self, verb: str, persona: int) -> str:
        """Conjuga un verbo en presente indicativo."""
        if verb in self.conjugations:
            return self.conjugations[verb].get(persona, verb)
        # Conjugación regular por defecto (presente indicativo)
        if verb.endswith("ar"):
            suffixes = {1: "o", 2: "as", 3: "a"}
            return verb[:-2] + suffixes.get(persona, "a")
        elif verb.endswith("er"):
            suffixes = {1: "o", 2: "es", 3: "e"}
            return verb[:-2] + suffixes.get(persona, "e")
        elif verb.endswith("ir"):
            suffixes = {1: "o", 2: "es", 3: "e"}
            return verb[:-2] + suffixes.get(persona, "e")
        return verb

    def _get_article(self, token: dict) -> str:
        """Obtiene el artículo definido para un token."""
        genero = token.get("genero")
        if genero and genero in ARTICLES:
            return ARTICLES[genero]["def"]
        return ""

    def _format_object(self, obj: dict, verbos: list[dict]) -> str:
        """Formatea un complemento con su preposición y artículo."""
        art = self._get_article(obj)
        word = obj["word"]

        tipo = obj.get("tipo", "")
        prep = ""

        if tipo == "lugar" and verbos:
            verbo_tipo = verbos[0].get("tipo", "")
            if verbo_tipo == "movimiento":
                prep = "a"
            else:
                prep = "en"

        if prep and art:
            # Contracciones: a + el = al, de + el = del
            if prep == "a" and art == "el":
                return f"al {word}"
            return f"{prep} {art} {word}"
        elif prep:
            return f"{prep} {word}"
        elif art:
            return f"{art} {word}"
        return word

    def _capitalize(self, sentence: str) -> str:
        """Capitaliza la primera letra de la oración."""
        if not sentence:
            return sentence
        # No capitalizar si empieza con ¿ o ¡
        if sentence[0] in ("¿", "¡"):
            if len(sentence) > 1:
                return sentence[0] + sentence[1].upper() + sentence[2:]
            return sentence
        return sentence[0].upper() + sentence[1:]


# ---------------------------------------------------------------------------
# CLI de prueba
# ---------------------------------------------------------------------------

def main() -> None:
    """Ejemplos con vocabulario real de LSC50 y la salida REAL registrada el 13-sep-2026.

    Imprime cada caso y, al final, cuántas salidas siguen coincidiendo con las registradas.
    NO es un conjunto de pruebas con salida esperada (eso es tarea del mes 3): las salidas
    agramaticales se registran tal cual, para dejar constancia de lo que el módulo hace hoy.
    """
    mapper = GrammarMapper()

    print("=" * 60)
    print("  Conversión de glosas LSC50 → español (hoy: tabla por categorías)")
    print("=" * 60)

    # (glosas de entrada, salida REAL registrada el 13-sep-2026, qué ilustra)
    casos: list[tuple[list[str | int], str, str]] = [
        # --- Plantillas: saludo, cortesía, pregunta ---
        (["hola"], "Hola", "plantilla: saludo"),
        (["hola", "buenos días"], "Hola, buenos días", "plantilla: saludo + saludo"),
        (["hola", "¿cómo estás?"], "Hola, ¿cómo estás?", "plantilla: saludo + pregunta"),
        (["gracias"], "Gracias", "plantilla: cortesía"),
        # --- Orden fijo por categoría y conjugación en presente ---
        (["yo", "comer"], "Yo como", "sujeto + verbo, 1.ª persona"),
        (["comer", "yo"], "Yo como", "misma salida: el orden de entrada se ignora"),
        (["hermano", "trabajar"], "El hermano trabaja", "artículo según género + 3.ª persona"),
        (["mujer", "trabajar"], "La mujer trabaja", "artículo femenino"),
        (["familia", "vivir", "bien"], "La familia vive bien", "verbo + adverbio de estado"),
        (["yo", "nunca", "comer"], "Yo nunca como", "adverbio de frecuencia antes del verbo"),
        (["yo", "trabajar", "mucho"], "Yo trabajo mucho", "adverbio de cantidad después"),
        (["hola", "yo", "trabajar"], "Hola yo trabajo", "saludo + oración (sin coma)"),
        (["perdón", "yo", "vivir"], "Yo vivo perdón", "la cortesía siempre va al final"),
        # --- Preguntas con interrogativo ---
        (["¿dónde?", "tú", "vivir"], "¿Dónde vives?", "interrogativo + verbo; el pronombre se omite"),
        (["¿quién?", "comer"], "¿Quién come?", "sin sujeto: 3.ª persona por defecto"),
        # --- Desde IDs del modelo (0-49) ---
        ([30, 26], "Hola, ¿cómo estás?", "IDs: 30 = hola, 26 = ¿cómo estás?"),
        ([18, 6], "El hermano trabaja", "IDs: 18 = hermano, 6 = trabajar"),
        # --- Limitaciones actuales: salidas AGRAMATICALES registradas tal cual ---
        (["ustedes", "comer"], "Ustedes come", "AGRAMATICAL: sin concordancia de número"),
        (["todos", "comer", "mucho"], "Todos come mucho", "AGRAMATICAL: sin concordancia de número"),
        (["niña", "contento"], "La niña contento", "AGRAMATICAL: adjetivo sin concordancia de género"),
        (["yo", "feliz"], "Yo feliz", "AGRAMATICAL: sin cópula"),
        (["yo", "hambre"], "Yo la hambre", "AGRAMATICAL: objeto con artículo y sin verbo"),
        (["yo", "seña", "nombre"], "Yo la seña el nombre", "AGRAMATICAL: dos objetos sin verbo"),
        (["hermano", "tío", "comer"], "El hermano come", "el segundo sujeto se descarta en silencio"),
        (["yo", "comer", "trabajar"], "Yo como trabajar", "AGRAMATICAL: segundo verbo en infinitivo suelto"),
    ]

    coinciden = 0
    for glosas, registrada, nota in casos:
        salida = mapper.translate(glosas)
        igual = salida == registrada
        if igual:
            coinciden += 1
        entrada = " + ".join(str(g) for g in glosas)
        print(f"\n  LSC:  [{entrada}]")
        print(f"  ESP:  {salida}" + ("" if igual else f"   (registrada el 13-sep-2026: {registrada})"))
        print(f"  Nota: {nota}")

    print("\n" + "-" * 60)
    print(f"  {coinciden}/{len(casos)} salidas coinciden con las registradas el 13-sep-2026")

    # Vocabulario completo con la categoría que le asigna GRAMMAR_CATEGORIES
    print("\n" + "=" * 60)
    print("  Vocabulario LSC50 (50 señas)")
    print("=" * 60)
    for sign_id, name in sorted(SIGN_VOCABULARY.items()):
        cat = GRAMMAR_CATEGORIES.get(name, {}).get("cat", "?")
        print(f"  {sign_id:3d} → {name:20s} [{cat}]")


if __name__ == "__main__":
    main()
