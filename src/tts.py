"""
Módulo de síntesis de voz (TTS) en español para el prototipo.

Modo por defecto: «offline» —el motor de voz del sistema operativo, sin internet—, conforme
a §2.2.5 del anteproyecto, que prioriza un motor TTS en español ejecutable localmente en CPU.
pyttsx3 enumera y elige la voz; la locución la ejecuta pyttsx3 o, en macOS, el comando `say`
del sistema (ver «Ejecución en macOS»). Los otros dos modos son OPCIONALES, de contraste:
  - «edge»: Microsoft Edge TTS (voz es-CO-SalomeNeural). Requiere internet y el paquete
    edge-tts, que NO está en requirements.txt.
  - «gtts»: Google TTS (español, tld com.co). Requiere internet; el paquete gTTS sí está en
    requirements.txt.
  Ambos generan un MP3 y lo reproducen con pygame (NO está en requirements.txt) o, si falta,
  con afplay (macOS) o ffplay. Si falta el paquete del modo o no hay ningún reproductor, el
  constructor falla con TTSDependenciaFaltante y un mensaje claro. Importar este módulo nunca
  falla por esas dependencias: solo se importan dentro del modo que las usa.

Selección de voz en modo offline: se busca una voz en ESPAÑOL por, en este orden,
  (1) el atributo `languages` que entrega pyttsx3 (p. ej. ['es_MX'] en macOS, b'\x05es' en
      espeak, un LCID hexadecimal como '40A' en Windows);
  (2) un código es-XX / es_XX dentro del identificador de la voz;
  (3) nombres de voces en español conocidos (Mónica, Paulina, Jorge, Juan, Diego, ...) o la
      palabra «spanish»/«español» en el nombre.
  NUNCA por «"es" in id»: ese filtro, que usaba la versión de marzo de 2026, casaba con
  «synthesis» en «com.apple.speech.synthesis.voice.Albert» y elegía una voz inglesa en macOS.
  Entre varias voces en español se prefiere es-CO, luego las variantes latinoamericanas
  (es-MX, es-419, ...), luego es-ES; dentro de la misma variante, las voces premium/enhanced/
  compact de Apple antes que las «eloquence» (síntesis por formantes, menos natural). Si no hay
  ninguna voz en español se avisa, se listan las disponibles y se usa la voz del sistema.

Ejecución en macOS (verificado el 14-sep-2026: macOS 26.6.2, pyttsx3 2.99, driver nsss). El
driver nsss de pyttsx3 tiene dos defectos que impiden usarlo tal cual en el prototipo:
  (a) solo la PRIMERA llamada a runAndWait() sintetiza la locución completa; en las siguientes
      el bucle termina de inmediato (say() se ejecuta fuera del bucle porque busy ya es False;
      al arrancar, el temporizador onPumpFirst vuelve a poner busy=False y ejecuta endLoop
      antes de que acabe la locución);
  (b) desde un hilo secundario —speak_async, que es lo que usa translator.py— el NSRunLoop
      del hilo no recibe el aviso de fin de locución: la síntesis se trunca (o el bucle se
      cuelga si se mantiene vivo con un temporizador).
  Por eso en macOS la locución se ejecuta con `say -v <nombre de voz> -r <palabras/min>`
  (el mismo sintetizador de Apple que envuelve nsss; local, en CPU, bloquea solo el hilo
  que lo llama; el texto se pasa por stdin) y pyttsx3 queda para enumerar y elegir la voz.
  En las demás plataformas la locución usa pyttsx3 con la orden encolada antes de arrancar
  el bucle (proxy.setBusy(True)), que corrige (a). Atributo `ejecutor_offline` ∈ {"say",
  "pyttsx3"}; `--listar-voces` lo muestra.
  Salvaguarda (14-sep-2026): `say -v NombreInexistente` NO falla (código 0) sino que habla con
  la voz por defecto del sistema sin avisar. Por eso, al instanciar en macOS, el nombre de la
  voz elegida se contrasta con la lista de `say -v '?'` y, si no aparece, se imprime un aviso.

Síntesis antes de la orden de reproducción (§2.2.7; 15-sep-2026). prepare(texto) sintetiza a un
archivo en segundo plano en los tres modos —offline con `say -o archivo.aiff` en macOS o
engine.save_to_file en las demás plataformas; edge y gtts a MP3—; esperar_preparado(tope) espera a
que termine (o falle) y play_prepared_async() emite la orden de reproducción del archivo ya
sintetizado. Así la etapa «síntesis» (t5 − t4) del prototipo incluye la síntesis real y excluye la
reproducción. Las locuciones se reproducen una tras otra, sin solaparse: si llega una nueva mientras
suena la anterior, espera su turno (antes se descartaba en silencio). El candado no garantiza por sí
mismo el orden de llegada; en el prototipo cada orden va precedida de ≈0,45 s de síntesis. Fuera de
macOS, el modo offline solo sintetiza a archivo si hay con qué reproducir un WAV en el mismo proceso
(pygame, afplay o ffplay); si no, el prototipo usa speak_async y lo declara (revisión 15-sep-2026).

Audios temporales y errores en segundo plano (pendientes de N-34, 15-sep-2026).
  · Cada audio sintetizado (AIFF, WAV o MP3) vive en la carpeta temporal del módulo solo mientras hace
    falta: se borra al terminar su reproducción (con éxito o con error), cuando una preparación se
    reemplaza sin haberse reproducido (prepare() nuevo, o una síntesis vieja que termina tarde) y si la
    síntesis falla. play_prepared_async() toma el archivo en el momento de la llamada y el hilo que lo
    reproduce es su único dueño: un prepare() posterior no lo toca. Excepción: con el reproductor
    externo de Windows («start») el archivo se deja para cleanup(), porque puede seguir abierto.
  · Las excepciones de los hilos de locución (speak_async, play_prepared_async) —reproductor que falla,
    archivo que ya no existe, motor que falla— no salen del hilo: se cuentan en errores_reproduccion,
    la última queda en ultimo_error_reproduccion / ultima_excepcion_reproduccion (también
    estado_reproduccion()) y se avisa por consola una sola vez por tipo de excepción. speak() síncrono
    sí propaga la excepción a quien llama.
  · cleanup() cierra el módulo (idempotente): borra la carpeta temporal; las locuciones que esperaban
    turno ya no suenan y se cuentan en locuciones_descartadas_al_cerrar, sin aviso ni traza, igual que
    un error que ocurra por el cierre (p. ej. el archivo borrado mientras la locución esperaba). Tras
    cleanup(), prepare() falla de inmediato (error_preparacion) y speak() no habla.

Cache de audio y sintetizador persistente (16-sep-2026; pendiente de N-34).
  · CACHE por texto. prepare(texto) mira primero una cache de audios ya sintetizados en esta sesión:
    si acierta, copia el archivo y publica la preparación en el acto (medido: mediana 0,14 ms frente a
    458 ms sintetizando); si falla, sintetiza como antes y guarda el resultado. La clave es el TEXTO y
    la cache entera lleva una FIRMA (modo, voz, ritmo y ejecutor): si la voz o el ritmo cambian, la
    firma cambia, la cache se vacía y no se reutiliza ni un audio (medido: `say -o` es determinista
    —mismo sha256 para el mismo texto, voz y ritmo— pero cambia con la voz y con el ritmo).
    precalentar(frases) la llena en segundo plano con las frases más probables (las 50 glosas sueltas
    del prototipo); espera a que no haya ninguna preparación en curso antes de cada frase.
    TOPES Y CICLO DE VIDA (cierre de temporales de N-36): a lo sumo cache_max_entradas (64 por defecto)
    y cache_max_bytes (8 MiB); al pasarse, desaloja la entrada usada hace más tiempo (LRU). Los audios
    viven en una SUBCARPETA de la carpeta temporal del módulo, así que cleanup() los borra con todo lo
    demás y no sobrevive nada al proceso. La cache NUNCA entrega el archivo original: entrega una copia,
    de modo que el hilo que reproduce sigue siendo el dueño del archivo que borra al terminar.
    ultima_preparacion_desde_cache dice si la preparación vigente salió de la cache (el prototipo lo
    lleva al JSONL como tts_sintesis_desde_cache: si no, la etapa «síntesis» ≈ 0 ms aparecería sin
    explicación y N-34 quedaría contradicha en silencio). estado_cache() da entradas, bytes y aciertos.
  · SINTETIZADOR PERSISTENTE en macOS: un NSSpeechSynthesizer vivo (pyobjc, que YA es dependencia de
    pyttsx3 en macOS y está en requirements.txt) evita arrancar el proceso `say` en cada frase. Medido:
    mediana 14,2 ms por frase frente a 458 ms con `say -o`, y el AIFF resultante es byte a byte el mismo
    (sha256 idéntico en 190 de 190 frases con la misma voz y el mismo ritmo). Se usa solo para
    sintetizar a archivo y solo si hay voz elegida; cualquier fallo (pyobjc ausente, initWithVoice_ nulo,
    startSpeakingString_toURL_ falso, o más de NS_TOPE_SINTESIS_SEG sin terminar) cae de vuelta a `say`
    y lo deja anotado en ultimo_error_sintetizador. UN CANDADO ES OBLIGATORIO: dos hilos usando el mismo
    objeto a la vez producen archivos truncados o mezclados (medido: 4 096 y 37 376 bytes, ninguno igual
    al de `say`); con candado, cuatro peticiones simultáneas salen correctas y en fila.

Voz del prototipo desde el 23-nov-2026 (decisión del autor; docs/VOZ.md).
  · VOZ LOCAL = Mónica. En modo offline, sin --voice, se usa la voz del sistema llamada «Mónica» si existe
    (VOZ_LOCAL_PREFERIDA; la eligió el autor en la escucha a ciegas). Si no existe, se aplica la regla de
    siempre (es-CO > América Latina > es-ES, por calidad). voz_preferida=None vuelve a la regla sola.
  · MODO «pregrabada» (el del prototipo cuando existe models/voz_pregrabada/indice.json). Cadena por frase:
      1. «pregrabada»: el enunciado EXACTO está en el índice (src/voz_pregrabada.py) → se publica su WAV tal cual
         (no se copia ni se sintetiza: la preparación queda lista antes de que prepare() retorne);
      2. «xtts»: si no está y hay servicio XTTS-v2 (src/voz_xtts_servicio.py, proceso aparte lanzado con el
         intérprete de python_voz o de la variable TESIS_PYTHON_VOZ) y está listo → sintetiza a WAV. Lo que
         sintetiza XTTS se guarda en la cache de la sesión (la misma cache por texto de arriba): la segunda vez
         la frase sale de la cache;
      3. «local»: si no hay servicio, si aún está cargando, si cayó o si la frase falla → Mónica (mismo camino
         que el modo offline: sintetizador persistente o `say`). Lo local NO se guarda en la cache en este modo,
         para que la frase use XTTS en cuanto el servicio esté listo.
    El servicio se lanza al construir el módulo y carga en segundo plano: el constructor NO espera (el prototipo
    decide cuánto esperar con esperar_servicio_voz). Cualquier fallo del servicio, en cualquier momento, deja el
    resto de la sesión con Mónica y se avisa una vez por consola. cleanup() detiene el servicio.
  · Qué se usó en cada frase: ultima_fuente ∈ {"pregrabada", "xtts", "local", "edge", "gtts"} y
    ultima_sintesis_ns (desde prepare() hasta que el audio quedó listo). El prototipo los lleva al JSONL como
    tts_fuente y tts_tiempo_sintesis_ns. estado_voz() da los contadores por fuente y el estado del servicio.

Uso:
    from tts import TTSModule
    tts = TTSModule()                        # offline, voz local (Mónica si existe)
    tts = TTSModule(mode="pregrabada", python_voz="/ruta/python-con-coqui")   # la del prototipo
    tts.speak("Hola, buenos días")           # bloquea hasta terminar
    tts.speak_async("Yo como mucho")         # no bloquea (hilo aparte); lo usa realtime.py
    print(tts.voice, tts.voice_reason)       # qué voz se eligió y por qué

CLI:
    python src/tts.py --listar-voces         # voces del sistema y cuál se elegiría; SIN audio
    python src/tts.py --text "Gracias"       # habla con el modo por defecto (offline)
    python src/tts.py --mode pregrabada --text "Gracias."   # WAV pregrabado (o respaldo)
    python src/tts.py --mode edge --text "Hola"   # opcional: requiere edge-tts e internet
"""

from __future__ import annotations

import asyncio
import collections
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata

MODOS: tuple[str, ...] = ("offline", "edge", "gtts", "pregrabada")
MODO_POR_DEFECTO = "offline"           # del módulo; el del prototipo lo decide modo_tts_prototipo()
VOZ_EDGE_POR_DEFECTO = "es-CO-SalomeNeural"
VOZ_LOCAL_PREFERIDA = "Mónica"         # elegida por el autor en la escucha a ciegas (docs/VOZ.md)
# Qué produjo el audio de una locución (JSONL: tts_fuente).
FUENTES: tuple[str, ...] = ("pregrabada", "xtts", "local", "edge", "gtts")

# Topes de la cache de audio (16-sep-2026). 64 entradas cubren las 50 glosas del vocabulario con
# margen; 8 MiB es algo más de tres veces lo que ocupan las 62 frases medidas (2 508 992 bytes,
# 2,4 MiB; corregido en la verificación: 8 MiB / 2,4 MiB = 3,3). Al pasarse
# cualquiera de los dos topes se desaloja la entrada usada hace más tiempo.
CACHE_MAX_ENTRADAS = 64
CACHE_MAX_BYTES = 8 * 1024 * 1024
# Tope de la síntesis con el sintetizador persistente antes de volver a `say` (medido: mediana
# 14,2 ms, máximo 0,52 s en 186 frases). Dos segundos es muy holgado y evita un bloqueo largo.
NS_TOPE_SINTESIS_SEG = 2.0


class TTSDependenciaFaltante(RuntimeError):
    """Falta un paquete o un reproductor que el modo TTS pedido necesita."""


# ---------------------------------------------------------------------------
# Selección de voz en español (funciones puras, sin audio; se prueban con voces falsas)
# ---------------------------------------------------------------------------

# Nombres de voces en español que traen macOS (Mónica es-ES, Paulina es-MX, Jorge es-ES,
# Juan es-MX, Diego es-AR, Angélica es-MX, Marisol es-ES, Carlos y Soledad es-CO, Francisca
# es-CL, Isabela es-AR, Jimena es-CO) y Windows (Helena y Pablo es-ES, Sabina y Raúl es-MX).
# Solo se usan si ni `languages` ni el identificador dicen el idioma. «Laura» queda fuera:
# en macOS es una voz eslovaca.
NOMBRES_VOCES_ES: frozenset[str] = frozenset({
    "monica", "paulina", "jorge", "juan", "diego", "angelica", "marisol", "carlos",
    "soledad", "francisca", "isabela", "jimena", "helena", "pablo", "sabina", "raul",
})

# Regiones hispanohablantes de América (preferidas tras es-CO; el proyecto es colombiano).
_REGIONES_LATAM: frozenset[str] = frozenset({
    "co", "mx", "419", "us", "ar", "cl", "pe", "ve", "ec", "bo", "uy", "py", "cr", "pa",
    "gt", "hn", "sv", "ni", "cu", "do", "pr",
})

# Windows (SAPI5): pyttsx3 entrega el idioma como LCID hexadecimal. El idioma primario son
# los 10 bits bajos (0x0A = español); el sub-idioma (bits altos) indica la región.
_LCID_SUBIDIOMA_ES: dict[int, str] = {
    1: "es", 2: "mx", 3: "es", 4: "gt", 5: "cr", 6: "pa", 7: "do", 8: "ve", 9: "co",
    10: "pe", 11: "ar", 12: "ec", 13: "cl", 14: "uy", 15: "py", 16: "bo", 17: "sv",
    18: "hn", 19: "ni", 20: "pr", 21: "us", 22: "419", 23: "cu",
}

# Código de locale xx-YY o xx_YY dentro de un identificador, con frontera antes y después.
# Búsqueda solapada (lookahead) para no perder «es-es» detrás de «ms_es» en ids de SAPI5.
_RE_LOCALE_EN_ID = re.compile(r"(?=(?<![a-z0-9])([a-z]{2})[-_]([a-z0-9]{2,3})(?![a-z0-9]))")
_RE_LOCALE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*")
_RE_LCID = re.compile(r"[0-9a-f]{1,4}")


def _sin_acentos(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def _normalizar_locale(valor) -> str | None:
    """'es_MX' → 'es-mx'; b'\\x05es-419' (espeak) → 'es-419'; None/'' → None."""
    if valor is None:
        return None
    if isinstance(valor, bytes):
        valor = valor.decode("latin-1", "ignore")
    limpio = "".join(ch for ch in str(valor) if ch.isprintable()).strip().lower()
    return limpio.replace("_", "-") or None


def _es_locale_espanol(locale: str) -> bool:
    return locale == "es" or locale.startswith("es-")


def _locales_en_id(voz_id) -> list[str]:
    """Todos los códigos xx-yy que aparecen en un identificador de voz, en orden."""
    texto = (voz_id or "").lower()
    return [f"{m.group(1)}-{m.group(2)}" for m in _RE_LOCALE_EN_ID.finditer(texto)]


def _nombre_parece_espanol(nombre) -> bool:
    texto = _sin_acentos(nombre or "").lower()
    if "spanish" in texto or "espanol" in texto:
        return True
    palabras = set(re.findall(r"[a-z]+", texto))
    return bool(palabras & NOMBRES_VOCES_ES)


def clasificar_voz(voz) -> tuple[bool, str | None, str]:
    """Decide si una voz de pyttsx3 es en español.

    Devuelve (es_espanol, locale_normalizado_o_None, criterio) con criterio en
    {"languages", "id", "nombre", "ninguno"}. El primer criterio que dé un idioma decide.
    """
    # 1) atributo languages
    for lang in (getattr(voz, "languages", None) or []):
        for trozo in str(_normalizar_locale(lang) or "").split(";"):
            trozo = trozo.strip()
            if not trozo:
                continue
            if _RE_LOCALE.fullmatch(trozo):
                return _es_locale_espanol(trozo), trozo, "languages"
            if _RE_LCID.fullmatch(trozo):
                lcid = int(trozo, 16)
                if (lcid & 0x3FF) == 0x0A:
                    region = _LCID_SUBIDIOMA_ES.get(lcid >> 10, "")
                    return True, f"es-{region}" if region else "es", "languages"
                return False, f"lcid-{trozo}", "languages"
    # 2) código es-XX en el identificador
    codigos = _locales_en_id(getattr(voz, "id", None))
    if codigos:
        espanoles = [c for c in codigos if _es_locale_espanol(c)]
        if espanoles:
            return True, espanoles[0], "id"
        return False, codigos[0], "id"
    # 3) nombre conocido
    if _nombre_parece_espanol(getattr(voz, "name", None)):
        return True, None, "nombre"
    return False, None, "ninguno"


def _rango_variante(locale: str | None) -> int:
    """0 es-CO, 1 América Latina, 2 es-ES, 3 otras/desconocida (menor = preferida)."""
    region = locale.split("-", 1)[1] if locale and "-" in locale else ""
    if region == "co":
        return 0
    if region in _REGIONES_LATAM:
        return 1
    if region == "es":
        return 2
    return 3


def _rango_calidad(voz_id) -> int:
    """Heurística sobre el identificador (macOS): premium < enhanced < compact < otras < eloquence."""
    texto = (voz_id or "").lower()
    if "eloquence" in texto:
        return 4
    if "premium" in texto:
        return 0
    if "enhanced" in texto:
        return 1
    if "compact" in texto:
        return 2
    return 3


def elegir_voz_espanol(voces) -> tuple[object | None, str]:
    """Elige la voz en español preferida entre las de pyttsx3.

    Devuelve (voz, motivo). voz es None si no hay ninguna en español; el motivo lo explica.
    Orden de preferencia: variante (es-CO, América Latina, es-ES, otras), calidad según el
    identificador y, en empate, el orden en que el sistema las lista.
    """
    candidatas = []
    for indice, voz in enumerate(voces):
        es_espanol, locale, criterio = clasificar_voz(voz)
        if es_espanol:
            clave = (_rango_variante(locale), _rango_calidad(getattr(voz, "id", "")), indice)
            candidatas.append((clave, voz, locale, criterio))
    if not candidatas:
        return None, f"ninguna voz en español entre las {len(list(voces))} del sistema"
    candidatas.sort(key=lambda c: c[0])
    _, voz, locale, criterio = candidatas[0]
    motivo = f"español detectado por «{criterio}»"
    if locale:
        motivo += f" ({locale})"
    motivo += f"; {len(candidatas)} candidatas en español, preferida por variante y calidad"
    return voz, motivo


def elegir_voz_local(voces, preferida: str | None = VOZ_LOCAL_PREFERIDA) -> tuple[object | None, str]:
    """Voz local del prototipo: la llamada `preferida` (sin tildes ni mayúsculas; «Mónica (Spanish (Spain))»
    cuenta como «Mónica») si el sistema la tiene, con las premium/enhanced antes que las compact; si no está, o
    si preferida es None, la regla general de elegir_voz_espanol."""
    if preferida:
        clave = _sin_acentos(preferida).lower().strip()
        candidatas = []
        for indice, voz in enumerate(voces):
            nombre = _sin_acentos(getattr(voz, "name", "") or "").lower().strip()
            primera = re.split(r"[\s(]+", nombre, maxsplit=1)[0] if nombre else ""
            if primera == clave:
                candidatas.append(((_rango_calidad(getattr(voz, "id", "")), indice), voz))
        if candidatas:
            candidatas.sort(key=lambda c: c[0])
            return candidatas[0][1], f"voz local elegida por el autor («{preferida}», docs/VOZ.md)"
    elegida, motivo = elegir_voz_espanol(voces)
    if preferida and elegida is not None:
        motivo = f"no hay voz «{preferida}» en el sistema; {motivo}"
    return elegida, motivo


def modo_tts_prototipo(indice=None) -> tuple[str, str | None]:
    """Modo de voz por defecto del prototipo y aviso (o None): «pregrabada» si existe el índice de voz
    pregrabada (models/voz_pregrabada/indice.json); si no, «offline» (Mónica) con un aviso."""
    vp = _importar_del_proyecto("voz_pregrabada")
    if vp.existe_indice(indice):
        return "pregrabada", None
    ruta = indice or vp.RUTA_INDICE_POR_DEFECTO
    return "offline", (f"  [AVISO TTS] no existe el índice de voz pregrabada ({ruta}); se usa la voz local "
                       f"(modo offline, {VOZ_LOCAL_PREFERIDA} si está instalada). Ver docs/VOZ.md.")


def _importar_del_proyecto(nombre: str):
    """Módulo hermano de src/ (voz_pregrabada, voz_xtts_servicio) tanto si este módulo se importó como
    src.tts como si se ejecuta src/tts.py directamente."""
    import importlib
    try:
        return importlib.import_module(f"src.{nombre}")
    except ImportError:
        return importlib.import_module(nombre)


def describir_voz(voz) -> str:
    return f"{getattr(voz, 'id', '?')} | {getattr(voz, 'name', '?')} | {getattr(voz, 'languages', [])}"


def _importar_pyttsx3():
    try:
        import pyttsx3
    except ImportError as exc:
        raise TTSDependenciaFaltante(
            "El modo 'offline' necesita pyttsx3 (está en requirements.txt): "
            "pip install pyttsx3   (en macOS arrastra pyobjc)."
        ) from exc
    return pyttsx3


def _importar_nsspeech():
    """Entrega las clases de pyobjc que usa el sintetizador persistente en macOS.

    Está aparte para que las pruebas puedan sustituirla por dobles (sin pyobjc y sin audio) y para que
    un fallo de importación se trate como «no hay sintetizador persistente», no como un error del módulo.
    pyobjc ya es dependencia de pyttsx3 en macOS y está en requirements.txt: no se añade nada.
    """
    from AppKit import NSSpeechSynthesizer
    from Foundation import NSDate, NSRunLoop, NSURL
    return NSSpeechSynthesizer, NSDate, NSRunLoop, NSURL


def ejecutor_offline_por_defecto() -> str:
    """'say' en macOS si el comando existe (ver docstring, «Ejecución en macOS»); si no, 'pyttsx3'."""
    if sys.platform == "darwin" and shutil.which("say"):
        return "say"
    return "pyttsx3"


_RE_LINEA_SAY = re.compile(r"^(?P<nombre>.+?)\s+(?P<locale>[a-zA-Z]{2,3}[_-][A-Za-z0-9]{2,4})\s+#")


def _parsear_salida_say(texto: str) -> dict[str, str]:
    """Nombre → locale a partir de la salida de `say -v '?'` (función pura, para pruebas).

    Cada línea tiene la forma «Paulina             es_MX    # ¡Hola! Me llamo Paulina.»; el
    nombre puede llevar espacios y paréntesis («Eddy (Spanish (Mexico)) es_MX    # ...»).
    """
    voces: dict[str, str] = {}
    for linea in texto.splitlines():
        m = _RE_LINEA_SAY.match(linea)
        if m:
            voces[m.group("nombre").strip()] = m.group("locale")
    return voces


def nombres_voces_say() -> set[str]:
    """Nombres de voz que acepta `say` en macOS; conjunto vacío si no se puede consultar."""
    if sys.platform != "darwin" or not shutil.which("say"):
        return set()
    try:
        salida = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return set(_parsear_salida_say(salida or ""))


def listar_voces_sistema():
    """Voces que ve pyttsx3 y cuál se elegiría. No reproduce audio."""
    motor = _importar_pyttsx3().init()
    voces = list(motor.getProperty("voices"))
    elegida, motivo = elegir_voz_local(voces)
    return voces, elegida, motivo


def imprimir_voces(voice: str | None = None) -> None:
    """Imprime las voces del sistema, marca las españolas y la que se elegiría."""
    voces, elegida, motivo = listar_voces_sistema()
    print(f"Voces del sistema según pyttsx3 ({sys.platform}): {len(voces)}")
    print("  marca [es] = clasificada como español; ==> = la que elegiría TTSModule(mode='offline')")
    n_es = 0
    for indice, voz in enumerate(voces):
        es_espanol, locale, criterio = clasificar_voz(voz)
        n_es += es_espanol
        marca = "==>" if elegida is not None and voz is elegida else "   "
        etiqueta = f"[es:{criterio}]" if es_espanol else "     "
        print(f"{marca} {indice:3d} {etiqueta:14s} {describir_voz(voz)}")
    print(f"Voces en español: {n_es} de {len(voces)}")
    ejecutor = ejecutor_offline_por_defecto()
    print(f"Locución en modo offline: {ejecutor}"
          + (" (comando del sistema; la voz se pasa por nombre con -v)" if ejecutor == "say" else ""))
    if ejecutor == "say" and elegida is not None:
        conocidas = nombres_voces_say()
        nombre = getattr(elegida, "name", None)
        if conocidas:
            estado = "la conoce" if nombre in conocidas else "NO la lista (hablaría con la voz por defecto sin avisar)"
            print(f"Comprobación con `say -v '?'` ({len(conocidas)} voces): «{nombre}» → {estado}")
        else:
            print("Comprobación con `say -v '?'`: no se pudo consultar")
    if voice:
        ids = {getattr(v, "id", None) for v in voces}
        print(f"--voice indicada: {voice} → {'existe' if voice in ids else 'NO existe en el sistema'}")
    elif elegida is not None:
        print(f"Elegida: {describir_voz(elegida)}")
        print(f"Motivo:  {motivo}")
    else:
        print(f"Elegida: ninguna ({motivo}); se usaría la voz por defecto del sistema")


# ---------------------------------------------------------------------------
# Reproductor de archivos MP3 (solo modos edge y gtts)
# ---------------------------------------------------------------------------

def _detectar_reproductor() -> str | None:
    """pygame si está instalado; si no, afplay (macOS) o ffplay en el PATH; None si nada."""
    try:
        import pygame  # noqa: F401
        return "pygame"
    except ImportError:
        pass
    for programa in ("afplay", "ffplay"):
        if shutil.which(programa):
            return programa
    if sys.platform == "win32":
        return "start"
    return None


class TTSModule:
    """Motor de síntesis de voz con cuatro modos: offline (por defecto del módulo), edge, gtts y pregrabada."""

    def __init__(self, mode: str = MODO_POR_DEFECTO, voice: str | None = None,
                 rate: int = 150, verbose: bool = True, cache: bool = True,
                 cache_max_entradas: int = CACHE_MAX_ENTRADAS,
                 cache_max_bytes: int = CACHE_MAX_BYTES,
                 sintetizador_persistente: bool = True,
                 voz_preferida: str | None = VOZ_LOCAL_PREFERIDA,
                 indice_pregrabada=None,
                 python_voz: str | None = None,
                 xtts_hablante: str | None = None,
                 xtts_dispositivo: str = "auto",
                 xtts_motor: str = "xtts",
                 xtts_argumentos=(),
                 xtts_tope_arranque_seg: float | None = None,
                 xtts_tope_sintesis_seg: float | None = None,
                 servicio_xtts=None):
        """
        Args:
            mode: "offline" (por defecto del módulo), "edge", "gtts" o "pregrabada" (el del prototipo).
            voice: voz concreta. offline y pregrabada (voz local de respaldo): identificador de voz del sistema
                   (ver --listar-voces); edge: p. ej. "es-CO-SalomeNeural"; gtts: se ignora. Si es None,
                   offline/pregrabada usan voz_preferida (Mónica) si existe y edge usa Salomé.
            rate: palabras por minuto en modo offline.
            verbose: imprime qué voz se eligió.
            cache: guarda cada audio sintetizado y lo reutiliza si vuelve el mismo texto con la
                   misma voz y el mismo ritmo (False lo desactiva: se sintetiza siempre).
            cache_max_entradas / cache_max_bytes: topes de la cache; al pasarse se desaloja la
                   entrada usada hace más tiempo. Los audios se borran con cleanup().
            sintetizador_persistente: en macOS, sintetizar a archivo con un NSSpeechSynthesizer vivo
                   (pyobjc) en vez de arrancar `say` en cada frase. Si falla, se usa `say`.
            voz_preferida: nombre de la voz local preferida (default «Mónica»); None → regla general.
            indice_pregrabada: (pregrabada) ruta del indice.json o de su carpeta; None → models/voz_pregrabada.
            python_voz: (pregrabada) intérprete con coqui-tts para el servicio XTTS. None → variable de entorno
                   TESIS_PYTHON_VOZ; "" o sin variable → sin servicio (el respaldo es la voz local).
            xtts_hablante / xtts_dispositivo ("auto", "mps", "cpu") / xtts_motor ("xtts"; "prueba" solo para
                   pruebas) / xtts_argumentos / topes: opciones del servicio (src/voz_xtts_servicio.py).
            servicio_xtts: cliente ya construido (pruebas); si se da, no se lanza otro.
        """
        if mode not in MODOS:
            raise ValueError(f"Modo TTS no soportado: {mode!r}; use uno de {MODOS}")
        self.mode = mode
        self.voice: str | None = voice
        self.voice_name: str | None = None
        self.voice_reason: str = ""
        self.engine = None
        self.rate = rate
        self.ejecutor_offline: str | None = None   # "say" (macOS) o "pyttsx3"; None en edge/gtts
        self._reproductor: str | None = None
        self._speaking = False
        self._candado_reproduccion = threading.Lock()   # una locución a la vez (Lock no garantiza FIFO)
        # Protege _prepared_file, _prepared_text, _evento_preparado, _audio_counter, _cerrado y los
        # contadores. Nunca se toma mientras se espera _candado_reproduccion (no hay interbloqueo).
        self._candado_estado = threading.Lock()
        self._evento_preparado = threading.Event()
        self._prepared_text: str | None = None
        self._prepared_file: str | None = None
        self.error_preparacion: str | None = None
        self._temp_dir: str | None = None          # se crea al final: un constructor que falla no deja carpeta
        self._audio_counter = 0
        self._cerrado = False
        # Errores de las locuciones en segundo plano (pendiente de N-34): se cuentan y se exponen.
        self.errores_reproduccion = 0
        self.ultimo_error_reproduccion: str | None = None
        self.ultima_excepcion_reproduccion: BaseException | None = None
        self.locuciones_descartadas_al_cerrar = 0
        self._tipos_error_avisados: set[str] = set()
        # Cache de audio por texto (16-sep-2026). Protegida por _candado_estado. El valor es
        # (ruta dentro de _cache_dir, bytes); el orden del OrderedDict es el de uso (LRU al final).
        self.cache_activa = bool(cache)
        self.cache_max_entradas = max(0, int(cache_max_entradas))
        self.cache_max_bytes = max(0, int(cache_max_bytes))
        self._cache: "collections.OrderedDict[str, tuple[str, int]]" = collections.OrderedDict()
        self._cache_firma: tuple | None = None
        self._cache_dir: str | None = None
        self.cache_bytes = 0
        self.cache_aciertos = 0
        self.cache_fallos = 0
        self.cache_desalojos = 0
        self.cache_errores = 0
        # None: no hubo preparación (o el TTS está cerrado). True/False: la preparación vigente salió
        # o no de la cache. El prototipo lo lleva al JSONL (tts_sintesis_desde_cache).
        self.ultima_preparacion_desde_cache: bool | None = None
        self.ultimo_sintetizador: str | None = None     # "cache" | "nsspeech" | "say" | "pyttsx3" | modo
        # Precalentado: llena la cache en segundo plano sin estorbar a una preparación en curso.
        self._evento_sin_preparacion = threading.Event()
        self._evento_sin_preparacion.set()
        self._preparaciones_en_curso = 0     # pueden solaparse dos: la vieja aún no ha terminado
        self._hilo_precalentado: threading.Thread | None = None
        self._evento_precalentado = threading.Event()
        self._evento_precalentado.set()
        self.precalentadas = 0
        self.precalentado_fallos = 0
        # Sintetizador persistente (macOS): un objeto vivo y un candado, porque dos hilos que lo usen
        # a la vez producen audios truncados o mezclados (medido el 16-sep-2026).
        self.sintetizador_persistente = bool(sintetizador_persistente)
        self._ns_syn = None
        self._ns_voz: str | None = None    # voz con la que se creó el objeto vivo (ver más abajo)
        self._ns_descartado = False
        self._candado_ns = threading.Lock()
        self.ultimo_error_sintetizador: str | None = None
        # Voz pregrabada y cadena de respaldo (23-nov-2026). ultima_fuente / ultima_sintesis_ns describen la
        # preparación vigente (None mientras no hay una lista); el prototipo las lleva al JSONL.
        self.voz_preferida = voz_preferida
        self.indice = None                              # IndiceVozPregrabada (modo pregrabada)
        self.local_disponible = mode == "offline"       # en pregrabada, si la voz local arrancó
        self.xtts = None                                # ServicioXTTS (modo pregrabada, si hay intérprete)
        self.xtts_hablante = xtts_hablante
        self.ultima_fuente: str | None = None
        self.ultima_sintesis_ns: int | None = None
        self.conteo_fuentes: dict[str, int] = {}
        self._avisos_voz: set[str] = set()

        if mode == "offline":
            self._init_offline(voice, rate, verbose)
        elif mode == "pregrabada":
            self._init_pregrabada(voice, rate, verbose, indice_pregrabada, python_voz, xtts_hablante,
                                  xtts_dispositivo, xtts_motor, xtts_argumentos, xtts_tope_arranque_seg,
                                  xtts_tope_sintesis_seg, servicio_xtts)
        elif mode == "edge":
            self._init_edge(voice, verbose)
        else:
            self._init_gtts(verbose)
        self._temp_dir = tempfile.mkdtemp(prefix="lsc_tts_")
        self._cache_dir = os.path.join(self._temp_dir, "cache")
        os.makedirs(self._cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Inicialización por modo
    # ------------------------------------------------------------------

    def _init_offline(self, voice: str | None, rate: int, verbose: bool) -> None:
        pyttsx3 = _importar_pyttsx3()
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        voces = list(self.engine.getProperty("voices"))
        if voice:
            coincidencias = [v for v in voces if getattr(v, "id", None) == voice]
            if not coincidencias:
                raise ValueError(
                    f"La voz {voice!r} no existe en el sistema. "
                    "Vea las disponibles con: python src/tts.py --listar-voces"
                )
            self.engine.setProperty("voice", voice)
            self.voice = voice
            self.voice_name = getattr(coincidencias[0], "name", None)
            self.voice_reason = "indicada por la persona usuaria"
        else:
            elegida, motivo = elegir_voz_local(voces, self.voz_preferida)
            if elegida is not None:
                self.engine.setProperty("voice", elegida.id)
                self.voice = elegida.id
                self.voice_name = getattr(elegida, "name", None)
                self.voice_reason = motivo
            else:
                # Sin voz en español: avisar, listar y dejar la voz por defecto del sistema.
                self.voice = None
                self.voice_reason = motivo
                print(f"  [AVISO TTS] {motivo}. Se usa la voz por defecto del sistema, "
                      "que probablemente no es en español. Voces disponibles:")
                for v in voces:
                    print(f"      {describir_voz(v)}")
        self.ejecutor_offline = ejecutor_offline_por_defecto()
        if self.ejecutor_offline == "say" and self.voice_name:
            # `say -v NombreInexistente` no falla: habla con la voz por defecto sin avisar.
            conocidas = nombres_voces_say()
            if conocidas and self.voice_name not in conocidas:
                print(f"  [AVISO TTS] `say -v '?'` no lista la voz «{self.voice_name}»: la locución "
                      "saldría con la voz por defecto del sistema sin avisar. Compruebe con "
                      "`say -v '?'` o indique otra con --voice (ver --listar-voces).")
        if verbose:
            print(f"  TTS offline (voces por pyttsx3/{getattr(self.engine, 'driver_name', '?')}; "
                  f"locución con {self.ejecutor_offline}): "
                  f"voz {self.voice_name or self.voice or 'del sistema'} — {self.voice_reason}")

    def _init_pregrabada(self, voice, rate, verbose, indice, python_voz, xtts_hablante, xtts_dispositivo,
                         xtts_motor, xtts_argumentos, tope_arranque, tope_sintesis, servicio) -> None:
        """Índice de voz pregrabada (obligatorio), voz local de respaldo (Mónica; si falta pyttsx3 se sigue sin
        ella) y servicio XTTS (si hay intérprete). El servicio se lanza y carga en segundo plano: aquí no se
        espera."""
        vp = _importar_del_proyecto("voz_pregrabada")
        try:
            self.indice = vp.IndiceVozPregrabada.cargar(indice)
        except vp.IndiceInvalido as exc:
            raise TTSDependenciaFaltante(f"El modo 'pregrabada' necesita el índice de voz: {exc}") from exc
        self._reproductor = None if shutil.which("afplay") else _detectar_reproductor()
        if not shutil.which("afplay") and self._reproductor is None:
            raise TTSDependenciaFaltante("El modo 'pregrabada' necesita con qué reproducir WAV: afplay (macOS), "
                                         "pygame o ffplay.")
        try:
            self._init_offline(voice, rate, verbose=False)
            self.local_disponible = True
        except TTSDependenciaFaltante as exc:
            self.local_disponible = False
            self.voice_reason = f"sin voz local ({exc})"
            print(f"  [AVISO TTS] sin voz local de respaldo ({exc}); las frases que no estén pregrabadas "
                  "solo sonarán si el servicio XTTS está listo.")
        sx = _importar_del_proyecto("voz_xtts_servicio")
        self.xtts_hablante = xtts_hablante or sx.HABLANTE_POR_DEFECTO
        if servicio is not None:
            self.xtts = servicio
        else:
            python = sx.python_voz_desde_entorno() if python_voz is None else (python_voz or None)
            if python:
                kw = {}
                if tope_arranque is not None:
                    kw["tope_arranque_seg"] = tope_arranque
                if tope_sintesis is not None:
                    kw["tope_sintesis_seg"] = tope_sintesis
                self.xtts = sx.ServicioXTTS(python, hablante=self.xtts_hablante, dispositivo=xtts_dispositivo,
                                            motor=xtts_motor, argumentos_extra=xtts_argumentos,
                                            al_cambiar=self._al_cambiar_xtts, **kw)
                try:
                    self.xtts.iniciar()
                except sx.ErrorServicioVoz as exc:
                    print(f"  [AVISO TTS] no se pudo lanzar el servicio XTTS ({exc}); el respaldo es la voz local.")
        if verbose:
            print(f"  TTS pregrabada: {self.indice.describir()} ({self.indice.ruta}); "
                  f"respaldo: {self.describir_respaldo()}")

    def describir_respaldo(self) -> str:
        """Cadena de respaldo en una línea, para la consola del prototipo."""
        local = (f"{self.voice_name or self.voice or 'voz del sistema'} ({self.ejecutor_offline})"
                 if self.local_disponible else "sin voz local")
        if self.xtts is None:
            return f"sin servicio XTTS → {local}"
        e = self.xtts.estado() if hasattr(self.xtts, "estado") else {}
        return f"XTTS-v2 «{self.xtts_hablante}» ({e.get('estado', '?')}) → {local}"

    def _al_cambiar_xtts(self, estado: str, detalle: str | None) -> None:
        """Avisos del servicio XTTS por consola (una vez por estado)."""
        if estado == "listo":
            print(f"  [voz] servicio XTTS-v2 listo ({detalle}); las frases no pregrabadas usan «{self.xtts_hablante}»")
        elif estado == "caido" and not self._cerrado:
            self._avisar_una_vez("xtts_caido", f"  [AVISO TTS] el servicio XTTS no está disponible ({detalle}); "
                                               f"las frases no pregrabadas usan la voz local en lo que queda de sesión.")

    def _avisar_una_vez(self, clave: str, mensaje: str) -> None:
        with self._candado_estado:
            if clave in self._avisos_voz:
                return
            self._avisos_voz.add(clave)
        print(mensaje)

    def esperar_servicio_voz(self, tope_seg: float) -> bool:
        """Espera como mucho tope_seg a que el servicio XTTS esté listo. True si lo está; False si no hay
        servicio, si cayó o si sigue cargando (en ese caso se usa la voz local hasta que esté listo)."""
        if self.xtts is None:
            return False
        return bool(self.xtts.esperar_listo(max(0.0, float(tope_seg))))

    def _init_edge(self, voice: str | None, verbose: bool) -> None:
        faltan = []
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            faltan.append("edge-tts")
        self._reproductor = _detectar_reproductor()
        if self._reproductor is None:
            faltan.append("pygame (o afplay/ffplay en el PATH)")
        if faltan:
            raise TTSDependenciaFaltante(
                "El modo 'edge' es opcional y necesita paquetes que no están en "
                f"requirements.txt: {', '.join(faltan)}. Instálelos con "
                "`pip install edge-tts pygame` o use el modo por defecto 'offline' (pyttsx3), "
                "que no necesita internet."
            )
        self.voice = voice or VOZ_EDGE_POR_DEFECTO
        self.voice_name = self.voice
        self.voice_reason = "voz Edge TTS (requiere internet)"
        if verbose:
            print(f"  TTS edge: voz {self.voice}; reproductor {self._reproductor}")

    def _init_gtts(self, verbose: bool) -> None:
        faltan = []
        try:
            from gtts import gTTS  # noqa: F401
        except ImportError:
            faltan.append("gTTS")
        self._reproductor = _detectar_reproductor()
        if self._reproductor is None:
            faltan.append("pygame (o afplay/ffplay en el PATH)")
        if faltan:
            raise TTSDependenciaFaltante(
                "El modo 'gtts' es opcional y necesita: "
                f"{', '.join(faltan)}. Instálelos con `pip install gTTS pygame` o use el modo "
                "por defecto 'offline' (pyttsx3), que no necesita internet."
            )
        self.voice = None
        self.voice_name = "Google TTS es (tld com.co)"
        self.voice_reason = "voz Google TTS (requiere internet)"
        if verbose:
            print(f"  TTS gtts: {self.voice_name}; reproductor {self._reproductor}")

    # ------------------------------------------------------------------
    # API pública (misma que en marzo de 2026; la usa translator.py)
    # ------------------------------------------------------------------

    @property
    def soporta_preparar(self) -> bool:
        """True si el audio sintetizado a archivo puede reproducirse: edge, gtts y offline con `say`
        siempre; offline con pyttsx3 (fuera de macOS) solo si hay pygame, afplay o ffplay. Sin ellos el
        WAV no sonaría (os.startfile abre un reproductor externo), así que el prototipo usa speak_async,
        como antes del 15-sep-2026, y declara la síntesis como no incluida (revisión)."""
        if self.mode != "offline" or self.ejecutor_offline == "say":
            return True
        return _detectar_reproductor() in ("pygame", "afplay", "ffplay")

    def speak(self, text: str) -> None:
        """Sintetiza y reproduce texto. Bloquea hasta terminar; si otra locución está sonando,
        espera a que acabe (no la descarta). Las excepciones del motor se propagan a quien llama
        (speak_async las captura). Tras cleanup() no habla: cuenta la locución como descartada."""
        if not text:
            return
        with self._candado_reproduccion:
            if self._cerrado:
                self._contar_descartada()
                return
            self._speaking = True
            try:
                if self.mode == "edge":
                    self._speak_edge(text)
                elif self.mode == "gtts":
                    self._speak_gtts(text)
                elif self.mode == "pregrabada":
                    self._speak_pregrabada(text)
                else:
                    self._speak_offline(text)
            finally:
                self._speaking = False

    def speak_async(self, text: str) -> None:
        """Sintetiza y reproduce en un hilo aparte (no bloquea). Las locuciones se encolan. Una
        excepción no sale del hilo: queda en errores_reproduccion / ultimo_error_reproduccion."""
        if not text:
            return
        threading.Thread(target=self._en_hilo, args=(self.speak, text), daemon=True).start()

    def prepare(self, text: str) -> None:
        """Sintetiza el audio a archivo en segundo plano, sin reproducirlo. Luego
        esperar_preparado() y play_prepared_async().

        Si el texto ya está en la cache con la misma voz y el mismo ritmo, la preparación queda lista
        ANTES de retornar (copiar el audio cuesta ~0,1 ms frente a ~460 ms de sintetizarlo) y
        ultima_preparacion_desde_cache queda en True."""
        if not text:
            return
        t_ini = time.perf_counter_ns()
        evento = threading.Event()
        with self._candado_estado:
            reemplazado = self._prepared_file         # sintetizado y nunca reproducido: sobra
            self._prepared_text = text
            self._prepared_file = None
            self._evento_preparado = evento
            cerrado = self._cerrado
            self.error_preparacion = "TTS cerrado con cleanup()" if cerrado else None
            self.ultima_preparacion_desde_cache = None if cerrado else False
            self.ultima_fuente = None
            self.ultima_sintesis_ns = None
        self._borrar_audio(reemplazado)
        if cerrado:
            evento.set()
            return
        if self.mode == "pregrabada":
            # 1. Enunciado pregrabado: se publica el WAV del índice tal cual (_borrar_audio nunca lo toca: está
            # fuera de la carpeta temporal). La cache no interviene (ultima_preparacion_desde_cache = None).
            ruta = self.indice.buscar(text) if self.indice is not None else None
            if ruta is not None and os.path.exists(ruta):
                self._publicar(evento, str(ruta), "pregrabada", t_ini, desde_cache=None)
                evento.set()
                return
        copia = self._copia_desde_cache(text)
        if copia is not None:
            if not self._publicar(evento, copia, self._fuente_de_la_cache(), t_ini, desde_cache=True):
                self._borrar_audio(copia)
            evento.set()
            return
        threading.Thread(target=self._prepare_audio, args=(text, evento, t_ini), daemon=True).start()

    def _publicar(self, evento, archivo: str, fuente: str, t_ini: int, desde_cache) -> bool:
        """Publica la preparación si sigue vigente. Devuelve False si ya fue reemplazada o el módulo se cerró
        (quien llama decide qué hacer con el archivo)."""
        with self._candado_estado:
            if evento is not self._evento_preparado or self._cerrado:
                return False
            self._prepared_file = archivo
            self.ultima_preparacion_desde_cache = desde_cache
            self.ultima_fuente = fuente
            self.ultima_sintesis_ns = time.perf_counter_ns() - t_ini
            self.conteo_fuentes[fuente] = self.conteo_fuentes.get(fuente, 0) + 1
            if desde_cache:
                self.ultimo_sintetizador = "cache"
            elif fuente == "pregrabada":
                self.ultimo_sintetizador = "pregrabada"
            return True

    def _fuente_de_la_cache(self) -> str:
        """En pregrabada la cache solo guarda lo de XTTS; en los demás modos, lo del propio modo."""
        if self.mode == "pregrabada":
            return "xtts"
        return "local" if self.mode == "offline" else self.mode

    def esperar_preparado(self, tope_seg: float) -> bool:
        """Espera a que termine la síntesis iniciada con prepare() (con éxito o con error) o a que venza
        el tope. Devuelve True si el audio quedó sintetizado."""
        self._evento_preparado.wait(timeout=tope_seg)
        return self.is_prepared

    def _prepare_audio(self, text: str, evento: threading.Event | None = None, t_ini: int | None = None) -> None:
        """Sintetiza a archivo sin reproducir, en todos los modos. El audio de una preparación ya
        reemplazada, o de una síntesis que falla, se borra."""
        audio_file = None
        t_ini = time.perf_counter_ns() if t_ini is None else t_ini
        with self._candado_estado:                    # el precalentado espera a que esto termine
            self._preparaciones_en_curso += 1
            self._evento_sin_preparacion.clear()
        try:
            # La ruta se pide ANTES de sintetizar: si la síntesis falla a medias, el parcial se borra.
            audio_file = self._next_audio_file(self._extension_audio())
            audio_file, fuente = self._sintetizar_con_fuente(text, audio_file)
            if not (os.path.exists(audio_file) and os.path.getsize(audio_file) > 0):
                raise RuntimeError(f"la síntesis no produjo audio en {audio_file}")
            if self._cacheable(fuente):
                self._guardar_en_cache(text, audio_file)  # antes de publicar: el audio puede sobrar y borrarse
            with self._candado_estado:
                # Una preparación ya reemplazada (o terminada tras cleanup) no se publica: su audio sobra.
                vigente = (evento is None or evento is self._evento_preparado) and not self._cerrado
                sobrante = self._prepared_file if vigente else audio_file
                if vigente:
                    self._prepared_file = audio_file
                    self.ultima_fuente = fuente
                    self.ultima_sintesis_ns = time.perf_counter_ns() - t_ini
                    self.conteo_fuentes[fuente] = self.conteo_fuentes.get(fuente, 0) + 1
            self._borrar_audio(sobrante)              # vigente: la anterior no reproducida; si no, la propia
        except Exception as exc:  # noqa: BLE001 — se informa y el prototipo lo registra como no sintetizado
            aviso = f"{type(exc).__name__}: {exc}"
            with self._candado_estado:
                # Revisión: el fallo de una preparación ya reemplazada no borra el archivo ni el error de la vigente.
                if evento is None or evento is self._evento_preparado:
                    self._prepared_file = None
                    self.error_preparacion = aviso
                cerrado = self._cerrado
            self._borrar_audio(audio_file)            # audio parcial, si lo hubo
            if not cerrado:
                print(f"  [AVISO TTS] no se pudo sintetizar a archivo ({aviso})")
        finally:
            with self._candado_estado:
                # Solo cuando NINGUNA preparación sigue viva (dos pueden solaparse) se libera el paso.
                self._preparaciones_en_curso = max(0, self._preparaciones_en_curso - 1)
                if self._preparaciones_en_curso == 0:
                    self._evento_sin_preparacion.set()
            if evento is not None:
                evento.set()

    def _sintetizar_offline_a_archivo(self, text: str, audio_file: str | None = None) -> str:
        """offline: en macOS, un NSSpeechSynthesizer vivo si se puede (mediana 14,2 ms) y si no
        `say -o archivo.aiff` (mediana 458 ms); los dos son el mismo sintetizador de Apple y con la
        misma voz y el mismo ritmo dan el MISMO archivo byte a byte (comprobado el 16-sep-2026).
        Fuera de macOS, pyttsx3 engine.save_to_file. audio_file None → ruta nueva."""
        if self.ejecutor_offline == "say":
            audio_file = audio_file or self._next_audio_file(".aiff")
            if self._sintetizar_con_persistente(text, audio_file):
                self.ultimo_sintetizador = "nsspeech"
                return audio_file
            orden = ["say", "-r", str(int(self.rate))]
            if self.voice_name:
                orden += ["-v", self.voice_name]
            orden += ["-o", audio_file]
            subprocess.run(orden, input=text.encode("utf-8"), check=True, timeout=60)
            self.ultimo_sintetizador = "say"
            return audio_file
        audio_file = audio_file or self._next_audio_file(".wav")
        proxy = getattr(self.engine, "proxy", None)
        if proxy is not None and hasattr(proxy, "setBusy"):
            proxy.setBusy(True)
        self.engine.save_to_file(text, audio_file)
        self.engine.runAndWait()
        self.ultimo_sintetizador = "pyttsx3"
        return audio_file

    def play_prepared(self) -> None:
        """Reproduce el audio sintetizado con prepare(). Si no está listo, sintetiza y reproduce."""
        archivo, texto = self._tomar_preparado()
        self._reproducir(archivo, texto)

    def play_prepared_async(self) -> None:
        """Emite la orden de reproducción del audio sintetizado y retorna de inmediato. El archivo se
        toma en el momento de la llamada: una preparación posterior no lo reemplaza."""
        archivo, texto = self._tomar_preparado()
        threading.Thread(target=self._en_hilo, args=(self._reproducir, archivo, texto), daemon=True).start()

    def _tomar_preparado(self) -> tuple[str | None, str | None]:
        """Entrega el audio preparado y deja de publicarlo: desde aquí su dueño es quien lo reproduce."""
        with self._candado_estado:
            archivo, texto = self._prepared_file, self._prepared_text
            self._prepared_file = None
        return (archivo if archivo and os.path.exists(archivo) else None), texto

    def _reproducir(self, archivo: str | None, texto: str | None) -> None:
        """Reproduce un audio ya sintetizado, esperando su turno, y lo borra al terminar (también si la
        reproducción falla). Sin archivo, sintetiza y reproduce el texto. Propaga las excepciones: en
        play_prepared_async las captura _en_hilo."""
        if archivo is None:
            if texto:
                self.speak(texto)
            return
        borrar = True
        try:
            with self._candado_reproduccion:
                if self._cerrado:
                    self._contar_descartada()
                    return
                if not os.path.exists(archivo):
                    raise FileNotFoundError(f"el audio sintetizado ya no existe: {archivo}")
                self._speaking = True
                try:
                    borrar = self._play_audio(archivo)
                finally:
                    self._speaking = False
        finally:
            if borrar:
                self._borrar_audio(archivo)

    def _en_hilo(self, funcion, *args) -> None:
        """Cuerpo de los hilos de locución: ninguna excepción sale del hilo (sin traza de threading)."""
        try:
            funcion(*args)
        except Exception as exc:  # noqa: BLE001 — se cuenta, se expone y se avisa una vez por tipo
            self._registrar_error_reproduccion(exc)

    def _registrar_error_reproduccion(self, exc: BaseException) -> None:
        with self._candado_estado:
            if self._cerrado:
                # Error provocado por el cierre (p. ej. cleanup() borró el audio que esperaba turno).
                self.locuciones_descartadas_al_cerrar += 1
                return
            tipo = type(exc).__name__
            aviso = f"{tipo}: {exc}"
            self.errores_reproduccion += 1
            self.ultimo_error_reproduccion = aviso
            self.ultima_excepcion_reproduccion = exc
            primera_de_su_tipo = tipo not in self._tipos_error_avisados
            self._tipos_error_avisados.add(tipo)
        if primera_de_su_tipo:
            print(f"  [AVISO TTS] falló una locución en segundo plano ({aviso}); esa locución no suena. "
                  f"Los siguientes errores {tipo} no se imprimen: se cuentan en errores_reproduccion.")

    def _contar_descartada(self) -> None:
        with self._candado_estado:
            self.locuciones_descartadas_al_cerrar += 1

    def estado_reproduccion(self) -> dict:
        """Errores y descartes de las locuciones en segundo plano, para que el prototipo los consulte."""
        with self._candado_estado:
            return {
                "errores_reproduccion": self.errores_reproduccion,
                "ultimo_error_reproduccion": self.ultimo_error_reproduccion,
                "locuciones_descartadas_al_cerrar": self.locuciones_descartadas_al_cerrar,
                "cerrado": self._cerrado,
            }

    @property
    def is_prepared(self) -> bool:
        """True si hay audio sintetizado listo para reproducir."""
        f = self._prepared_file
        return f is not None and os.path.exists(f)

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    # ------------------------------------------------------------------
    # Cache de audio por texto (16-sep-2026)
    # ------------------------------------------------------------------

    def _extension_audio(self) -> str:
        """Extensión del audio que produce este modo: .mp3 (edge/gtts), .aiff (`say`) o .wav (pyttsx3)."""
        if self.mode in ("edge", "gtts"):
            return ".mp3"
        if self.mode == "pregrabada":
            return ".wav"                 # XTTS escribe WAV; la voz local cambia la extensión al sintetizar
        return ".aiff" if self.ejecutor_offline == "say" else ".wav"

    def _sintetizar_en(self, text: str, audio_file: str) -> None:
        """Sintetiza el texto EN la ruta dada, sin publicarla ni reproducirla, en los tres modos."""
        if self.mode == "edge":
            import edge_tts

            async def _gen():
                await edge_tts.Communicate(text, self.voice).save(audio_file)

            asyncio.run(_gen())
            self.ultimo_sintetizador = "edge"
        elif self.mode == "gtts":
            from gtts import gTTS
            gTTS(text=text, lang="es", tld="com.co").save(audio_file)
            self.ultimo_sintetizador = "gtts"
        else:
            self._sintetizar_offline_a_archivo(text, audio_file)

    def _sintetizar_con_fuente(self, text: str, audio_file: str, permitir_local: bool = True) -> tuple[str, str]:
        """Sintetiza y devuelve (ruta_final, fuente). En pregrabada la ruta puede cambiar de extensión (la voz
        local de macOS escribe AIFF); en los demás modos es audio_file."""
        if self.mode != "pregrabada":
            self._sintetizar_en(text, audio_file)
            return audio_file, self._fuente_de_la_cache()
        ruta = self.indice.buscar(text) if self.indice is not None else None
        if ruta is not None and os.path.exists(ruta):
            shutil.copyfile(ruta, audio_file)
            return audio_file, "pregrabada"
        return self._sintetizar_respaldo(text, audio_file, permitir_local=permitir_local)

    def _sintetizar_respaldo(self, text: str, audio_file: str, permitir_local: bool = True) -> tuple[str, str]:
        """Frase NO pregrabada: XTTS si el servicio está listo; si no (o si falla), la voz local."""
        cli = self.xtts
        if cli is not None and getattr(cli, "disponible", False):
            try:
                cli.sintetizar(text, audio_file)
                return audio_file, "xtts"
            except Exception as exc:  # noqa: BLE001 — cualquier fallo del servicio cae a la voz local
                self._borrar_audio(audio_file)
                self.ultimo_error_sintetizador = f"xtts: {type(exc).__name__}: {exc}"
                if getattr(cli, "caido", False):
                    self._al_cambiar_xtts("caido", str(exc))
                else:
                    print(f"  [AVISO TTS] XTTS no pudo decir {text!r} ({exc}); se usa la voz local para esta frase.")
        elif cli is not None and getattr(cli, "cargando", False) and permitir_local:
            self._avisar_una_vez("xtts_cargando", "  [AVISO TTS] el servicio XTTS aún está cargando: las frases "
                                                  "no pregrabadas usan la voz local hasta que esté listo.")
        if not permitir_local:
            raise RuntimeError("servicio XTTS no disponible (y la voz local no se permite aquí)")
        if not self.local_disponible:
            raise RuntimeError("frase no pregrabada, sin servicio XTTS y sin voz local")
        ext = ".aiff" if self.ejecutor_offline == "say" else ".wav"
        local = os.path.splitext(audio_file)[0] + ext
        try:
            self._sintetizar_offline_a_archivo(text, local)
        except BaseException:
            self._borrar_audio(local)
            raise
        return local, "local"

    def _cacheable(self, fuente: str) -> bool:
        """En pregrabada solo se guarda lo de XTTS (lo local se rehace con XTTS en cuanto esté listo y lo
        pregrabado ya está en disco)."""
        return fuente == "xtts" if self.mode == "pregrabada" else True

    def _speak_pregrabada(self, text: str) -> None:
        """speak() en modo pregrabada: misma cadena, síntesis y reproducción en el acto."""
        ruta = self.indice.buscar(text) if self.indice is not None else None
        if ruta is not None and os.path.exists(ruta):
            self._play_audio(str(ruta))
            fuente = "pregrabada"
        else:
            audio_file = self._next_audio_file(".wav")
            final = audio_file
            try:
                final, fuente = self._sintetizar_respaldo(text, audio_file)
                borrar = self._play_audio(final)
            except BaseException:
                self._borrar_audio(audio_file)
                self._borrar_audio(final)
                raise
            if borrar:
                self._borrar_audio(final)
        with self._candado_estado:
            self.conteo_fuentes[fuente] = self.conteo_fuentes.get(fuente, 0) + 1

    def estado_voz(self) -> dict:
        """Modo, índice, servicio XTTS y cuántas locuciones salieron de cada fuente."""
        with self._candado_estado:
            conteo = dict(self.conteo_fuentes)
        return {"modo": self.mode,
                "indice": {"ruta": str(self.indice.ruta), "enunciados": len(self.indice)} if self.indice else None,
                "voz_local": self.voice_name if self.local_disponible else None,
                "xtts": self.xtts.estado() if self.xtts is not None and hasattr(self.xtts, "estado") else None,
                "fuentes": conteo, "ultima_fuente": self.ultima_fuente, "ultima_sintesis_ns": self.ultima_sintesis_ns}

    def _firma_cache(self) -> tuple:
        """Qué hace comparable a un audio guardado: modo, voz, ritmo y ejecutor. Si cambia la voz o el
        ritmo cambia la firma y la cache entera deja de servir (se vacía en el siguiente acceso)."""
        return (self.mode, self.voice, self.voice_name, int(self.rate), self.ejecutor_offline,
                self._extension_audio(), self.xtts_hablante if self.mode == "pregrabada" else None)

    def _vaciar_cache_bloqueado(self) -> list[str]:
        """Vacía la cache y devuelve los archivos a borrar. Se llama con _candado_estado tomado."""
        sobrantes = [ruta for ruta, _ in self._cache.values()]
        self._cache.clear()
        self.cache_bytes = 0
        return sobrantes

    def _desalojar_bloqueado(self) -> list[str]:
        """Desaloja por antigüedad de uso (LRU) hasta respetar los dos topes. Con _candado_estado."""
        sobrantes: list[str] = []
        while self._cache and (len(self._cache) > self.cache_max_entradas
                               or self.cache_bytes > self.cache_max_bytes):
            _, (ruta, tam) = self._cache.popitem(last=False)
            self.cache_bytes -= tam
            self.cache_desalojos += 1
            sobrantes.append(ruta)
        return sobrantes

    def _borrar_cache_archivo(self, ruta: str | None) -> None:
        """Borra un audio de la cache. Solo actúa dentro de la subcarpeta de cache del módulo."""
        if not ruta or not self._cache_dir:
            return
        if os.path.dirname(os.path.abspath(ruta)) != os.path.abspath(self._cache_dir):
            return
        try:
            os.remove(ruta)
        except OSError:
            pass

    def _next_cache_file(self, extension: str) -> str:
        with self._candado_estado:
            self._audio_counter += 1
            numero = self._audio_counter
        return os.path.join(self._cache_dir, f"cache_{numero}{extension}")

    def _copia_desde_cache(self, text: str) -> str | None:
        """Si el texto está en la cache con la misma voz y ritmo, devuelve una COPIA nueva del audio
        (el original se queda en la cache; quien reproduce es dueño de la copia y la borra). None si
        no hay acierto, si la cache está desactivada o si el archivo guardado ya no sirve."""
        if not self.cache_activa or not self._cache_dir:
            return None
        firma = self._firma_cache()
        with self._candado_estado:
            if self._cerrado:
                return None
            sobrantes = self._vaciar_cache_bloqueado() if self._cache_firma != firma else []
            self._cache_firma = firma
            entrada = self._cache.get(text)
            if entrada is not None:
                self._cache.move_to_end(text)
            else:
                self.cache_fallos += 1
        for ruta in sobrantes:
            self._borrar_cache_archivo(ruta)
        if entrada is None:
            return None
        origen = entrada[0]
        destino = None
        try:
            if not (os.path.exists(origen) and os.path.getsize(origen) > 0):
                raise FileNotFoundError(f"el audio de la cache ya no existe: {origen}")
            destino = self._next_audio_file(os.path.splitext(origen)[1] or ".aiff")
            shutil.copyfile(origen, destino)
        except OSError as exc:                 # la entrada dejó de servir: se olvida y se sintetiza
            with self._candado_estado:
                guardada = self._cache.pop(text, None)
                if guardada is not None:
                    self.cache_bytes -= guardada[1]
                self.cache_errores += 1
                self.cache_fallos += 1
                self.ultimo_error_sintetizador = f"cache: {type(exc).__name__}: {exc}"
            self._borrar_audio(destino)
            self._borrar_cache_archivo(origen)
            return None
        with self._candado_estado:
            self.cache_aciertos += 1
        return destino

    def _guardar_en_cache(self, text: str, audio_file: str) -> None:
        """Guarda una COPIA del audio recién sintetizado. Respeta los dos topes desalojando por LRU.
        Un audio más grande que el tope de bytes no entra (no se vacía la cache por su culpa)."""
        if not (self.cache_activa and text and audio_file and self._cache_dir):
            return
        if self.cache_max_entradas <= 0 or self.cache_max_bytes <= 0:
            return
        try:
            tam = os.path.getsize(audio_file)
        except OSError:
            return
        if tam <= 0 or tam > self.cache_max_bytes:
            return
        firma = self._firma_cache()
        destino = self._next_cache_file(os.path.splitext(audio_file)[1] or ".aiff")
        try:
            shutil.copyfile(audio_file, destino)
        except OSError as exc:
            with self._candado_estado:
                self.cache_errores += 1
                self.ultimo_error_sintetizador = f"cache: {type(exc).__name__}: {exc}"
            self._borrar_cache_archivo(destino)
            return
        sobrantes: list[str] = []
        with self._candado_estado:
            if self._cerrado:
                sobrantes = [destino]
            else:
                if self._cache_firma != firma:
                    sobrantes.extend(self._vaciar_cache_bloqueado())
                    self._cache_firma = firma
                anterior = self._cache.pop(text, None)
                if anterior is not None:
                    self.cache_bytes -= anterior[1]
                    sobrantes.append(anterior[0])
                self._cache[text] = (destino, tam)
                self.cache_bytes += tam
                sobrantes.extend(self._desalojar_bloqueado())
        for ruta in sobrantes:
            self._borrar_cache_archivo(ruta)

    def precalentar(self, frases, bloquear: bool = False, tope_seg: float | None = None) -> bool:
        """Llena la cache en segundo plano con las frases dadas, en orden (las más probables primero:
        en el prototipo, las 50 glosas sueltas ya pasadas por GrammarMapper). Antes de cada frase espera
        a que no haya ninguna preparación en curso, para no competir con una locución del prototipo.
        Devuelve False si la cache está desactivada, si el módulo está cerrado, si no hay frases o si ya
        hay un precalentado en marcha."""
        if not self.cache_activa or self._cerrado or not self._cache_dir:
            return False
        limpias = list(dict.fromkeys(f for f in (frases or []) if f))
        if self.mode == "pregrabada":
            # Lo pregrabado no se sintetiza; lo demás solo con XTTS (lo local no se guarda en este modo).
            limpias = [f for f in limpias if self.indice is None or f not in self.indice]
            if self.xtts is None or getattr(self.xtts, "caido", False):
                return False
        if not limpias:
            return False
        if self._hilo_precalentado is not None and self._hilo_precalentado.is_alive():
            return False
        self._evento_precalentado.clear()
        hilo = threading.Thread(target=self._precalentar_en_hilo, args=(limpias,), daemon=True)
        self._hilo_precalentado = hilo
        hilo.start()
        if bloquear:
            self.esperar_precalentado(tope_seg)
        return True

    def esperar_precalentado(self, tope_seg: float | None = None) -> bool:
        """Espera a que termine el precalentado. True si terminó (o si no había ninguno)."""
        self._evento_precalentado.wait(timeout=tope_seg)
        return self._evento_precalentado.is_set()

    def _precalentar_en_hilo(self, frases: list[str]) -> None:
        try:
            if self.mode == "pregrabada":
                # Espera a que el servicio XTTS termine de cargar (sin bloquear a nadie: es un hilo aparte).
                if not self.xtts.esperar_listo(getattr(self.xtts, "tope_arranque_seg", 180.0)):
                    return
            for frase in frases:
                if self._cerrado:
                    break
                # Nunca por delante de una preparación real: si hay una en curso, espera a que acabe.
                self._evento_sin_preparacion.wait(timeout=30.0)
                if self._cerrado:
                    break
                with self._candado_estado:
                    ya_esta = (self._cache_firma == self._firma_cache() and frase in self._cache)
                if ya_esta:
                    continue
                audio_file = None
                try:
                    audio_file = self._next_audio_file(self._extension_audio())
                    audio_file, fuente = self._sintetizar_con_fuente(
                        frase, audio_file, permitir_local=self.mode != "pregrabada")
                    if not (os.path.exists(audio_file) and os.path.getsize(audio_file) > 0):
                        raise RuntimeError(f"la síntesis no produjo audio en {audio_file}")
                    if self._cacheable(fuente):
                        self._guardar_en_cache(frase, audio_file)
                        self.precalentadas += 1
                except Exception as exc:  # noqa: BLE001 — un fallo al precalentar no rompe la sesión
                    self.precalentado_fallos += 1
                    self.ultimo_error_sintetizador = f"precalentado: {type(exc).__name__}: {exc}"
                    if self.mode == "pregrabada" and getattr(self.xtts, "caido", False):
                        break
                finally:
                    self._borrar_audio(audio_file)     # la cache ya tiene su copia
        finally:
            self._evento_precalentado.set()

    def estado_cache(self) -> dict:
        """Estado de la cache de audio, para el prototipo y el informe."""
        with self._candado_estado:
            return {
                "activa": self.cache_activa,
                "entradas": len(self._cache),
                "bytes": self.cache_bytes,
                "max_entradas": self.cache_max_entradas,
                "max_bytes": self.cache_max_bytes,
                "aciertos": self.cache_aciertos,
                "fallos": self.cache_fallos,
                "errores": self.cache_errores,
                "desalojos": self.cache_desalojos,
                "precalentadas": self.precalentadas,
                "precalentado_fallos": self.precalentado_fallos,
                "precalentando": not self._evento_precalentado.is_set(),
                "ultima_preparacion_desde_cache": self.ultima_preparacion_desde_cache,
                "sintetizador": self.ultimo_sintetizador,
                "sintetizador_persistente": self.sintetizador_persistente and not self._ns_descartado,
                "ultimo_error_sintetizador": self.ultimo_error_sintetizador,
            }

    # ------------------------------------------------------------------
    # Sintetizador persistente (macOS, NSSpeechSynthesizer vía pyobjc)
    # ------------------------------------------------------------------

    def _descartar_persistente(self, motivo: str) -> None:
        """Deja de usar el sintetizador persistente en lo que queda de sesión y avisa una sola vez."""
        primera = not self._ns_descartado
        self._ns_descartado = True
        self._ns_syn = None
        self._ns_voz = None
        self.ultimo_error_sintetizador = motivo
        if primera:
            print(f"  [AVISO TTS] el sintetizador persistente no se pudo usar ({motivo}); "
                  "se sintetiza con `say` en lo que queda de sesión.")

    def _sintetizar_con_persistente(self, text: str, audio_file: str) -> bool:
        """Sintetiza a archivo con un NSSpeechSynthesizer vivo. True si escribió el audio.

        Mismo sintetizador de Apple que `say`, con la misma voz y el mismo ritmo: el AIFF sale byte a
        byte igual (comprobado en 190 frases el 16-sep-2026) y cuesta ~14 ms en vez de ~458 ms. El
        candado es obligatorio: dos hilos usando el objeto a la vez producen audios truncados. Cualquier
        fallo devuelve False, borra el audio a medias y deja el resto de la sesión con `say`."""
        if not self.sintetizador_persistente or self._ns_descartado:
            return False
        if self.mode != "offline" or self.ejecutor_offline != "say" or not self.voice:
            return False
        try:
            NSSpeechSynthesizer, NSDate, NSRunLoop, NSURL = _importar_nsspeech()
        except Exception as exc:  # noqa: BLE001 — pyobjc ausente o roto: `say` sigue funcionando
            self._descartar_persistente(f"{type(exc).__name__}: {exc}")
            return False
        with self._candado_ns:
            try:
                if self._ns_descartado:
                    return False
                syn = self._ns_syn
                if syn is None or self._ns_voz != self.voice:
                    # El objeto vivo guarda la voz con la que nació: si la voz del módulo cambia hay
                    # que rehacerlo o seguiría sintetizando con la anterior (verificación 16-sep-2026;
                    # el ritmo, en cambio, se aplica en cada síntesis con setRate_).
                    syn = NSSpeechSynthesizer.alloc().initWithVoice_(self.voice)
                    if syn is None:
                        raise RuntimeError(f"initWithVoice_({self.voice!r}) devolvió None")
                    self._ns_syn = syn
                    self._ns_voz = self.voice
                syn.setRate_(float(self.rate))
                if not syn.startSpeakingString_toURL_(text, NSURL.fileURLWithPath_(audio_file)):
                    raise RuntimeError("startSpeakingString_toURL_ devolvió False")
                bucle = NSRunLoop.currentRunLoop()
                limite = time.monotonic() + NS_TOPE_SINTESIS_SEG
                while syn.isSpeaking():
                    bucle.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.001))
                    if time.monotonic() > limite:
                        syn.stopSpeaking()
                        raise TimeoutError(
                            f"la síntesis no terminó en {NS_TOPE_SINTESIS_SEG} s")
                if not (os.path.exists(audio_file) and os.path.getsize(audio_file) > 0):
                    raise RuntimeError(f"no escribió audio en {audio_file}")
                return True
            except Exception as exc:  # noqa: BLE001 — se cae a `say`, que rehace el archivo
                self._descartar_persistente(f"{type(exc).__name__}: {exc}")
                try:
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                except OSError:
                    pass
                return False

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _next_audio_file(self, extension: str = ".mp3") -> str:
        """Ruta única por audio (evita conflictos con archivos aún abiertos por el reproductor). Seguro entre
        hilos: una síntesis vieja y una nueva pueden pedir ruta a la vez."""
        with self._candado_estado:
            self._audio_counter += 1
            numero = self._audio_counter
        return os.path.join(self._temp_dir, f"output_{numero}{extension}")

    def _borrar_audio(self, ruta: str | None) -> None:
        """Borra un audio temporal propio. Solo actúa dentro de la carpeta temporal del módulo; si ya no
        existe o no se puede borrar (en uso), lo deja para cleanup()."""
        if not ruta:
            return
        if os.path.dirname(os.path.abspath(ruta)) != os.path.abspath(self._temp_dir):
            return
        try:
            os.remove(ruta)
        except OSError:
            pass

    def _speak_edge(self, text: str) -> None:
        import edge_tts

        audio_file = self._next_audio_file()

        async def _generate():
            await edge_tts.Communicate(text, self.voice).save(audio_file)

        try:
            asyncio.run(_generate())
            borrar = self._play_audio(audio_file)
        except BaseException:
            self._borrar_audio(audio_file)
            raise
        if borrar:
            self._borrar_audio(audio_file)

    def _speak_gtts(self, text: str) -> None:
        from gtts import gTTS

        audio_file = self._next_audio_file()
        try:
            gTTS(text=text, lang="es", tld="com.co").save(audio_file)
            borrar = self._play_audio(audio_file)
        except BaseException:
            self._borrar_audio(audio_file)
            raise
        if borrar:
            self._borrar_audio(audio_file)

    def _speak_offline(self, text: str) -> None:
        if self.ejecutor_offline == "say":
            orden = ["say", "-r", str(int(self.rate))]
            if self.voice_name:
                orden += ["-v", self.voice_name]
            try:
                # El texto va por stdin: no se confunde con opciones ni hace falta escaparlo.
                subprocess.run(orden, input=text.encode("utf-8"), check=True, timeout=60)
                return
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"  [AVISO TTS] `say` falló ({exc}); se intenta con pyttsx3")
        # pyttsx3: encolar la orden ANTES de arrancar el bucle. Tras la primera locución el proxy
        # queda con busy=False y say() se ejecutaría fuera del bucle; al arrancar, onPumpFirst
        # pone busy=False, ejecuta el endLoop encolado y el bucle termina sin esperar la locución
        # (defecto (a) del docstring). Con setBusy(True) la orden se procesa dentro del bucle,
        # igual que en la primera llamada. No corrige (b): en macOS se usa `say`.
        proxy = getattr(self.engine, "proxy", None)
        if proxy is not None and hasattr(proxy, "setBusy"):
            proxy.setBusy(True)
        self.engine.say(text)
        self.engine.runAndWait()

    # ------------------------------------------------------------------
    # Reproducción de archivos de audio (edge y gtts)
    # ------------------------------------------------------------------

    def _play_audio(self, filepath: str) -> bool:
        """Reproduce el archivo y bloquea hasta que termina. Devuelve True si al volver el archivo ya puede
        borrarse (reproducción dentro del proceso) y False con el reproductor externo de Windows."""
        # AIFF (say -o) y WAV (pyttsx3): afplay en macOS; pygame no garantiza AIFF.
        if filepath.endswith((".aiff", ".wav")) and shutil.which("afplay"):
            subprocess.run(["afplay", filepath], check=True, timeout=60)
            return True
        reproductor = self._reproductor or _detectar_reproductor()
        if reproductor == "pygame":
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(filepath)
            try:
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(50)
            finally:
                pygame.mixer.music.unload()          # libera el archivo antes de borrarlo
        elif reproductor == "afplay":
            subprocess.run(["afplay", filepath], check=True, timeout=30)
        elif reproductor == "ffplay":
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath],
                           check=True, timeout=30)
        elif reproductor == "start":
            os.startfile(filepath)  # Windows: reproductor asociado al .mp3
            time.sleep(2)
            return False                             # puede seguir abierto: lo borra cleanup()
        else:
            raise TTSDependenciaFaltante(
                "No hay con qué reproducir el audio: instale pygame (`pip install pygame`) "
                "o tenga afplay/ffplay en el PATH."
            )
        return True

    def cleanup(self) -> None:
        """Cierra el módulo y borra la carpeta de audios temporales. Idempotente y sin esperar a los hilos:
        las locuciones que esperan turno ya no suenan y se cuentan en locuciones_descartadas_al_cerrar
        (sin aviso ni traza); la que ya está sonando no se interrumpe desde aquí."""
        with self._candado_estado:
            self._cerrado = True
            self._cache.clear()              # los audios se van con la carpeta temporal
            self.cache_bytes = 0
            self._cache_firma = None
        self._ns_descartado = True           # el sintetizador persistente no se usa tras cerrar
        self._ns_syn = None
        self._ns_voz = None
        if self.xtts is not None and hasattr(self.xtts, "detener"):
            try:
                self.xtts.detener()          # solo el proceso que lanzó este módulo
            except Exception:  # noqa: BLE001 — cerrar nunca falla por el servicio
                pass
        self._evento_sin_preparacion.set()   # el precalentado, si lo hay, sale en la siguiente vuelta
        self._preparaciones_en_curso = 0
        self._evento_precalentado.set()
        try:
            if os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Síntesis de voz del prototipo. Modo por defecto: offline (pyttsx3, sin internet).")
    parser.add_argument("--mode", choices=MODOS, default=MODO_POR_DEFECTO,
                        help="offline (por defecto, local) | pregrabada (WAV del índice + respaldo) | edge | gtts")
    parser.add_argument("--voice", type=str, default=None,
                        help="offline: identificador de voz del sistema; edge: nombre de voz Edge")
    parser.add_argument("--text", type=str, default=None, help="frase a decir (si no, cinco de prueba)")
    parser.add_argument("--rate", type=int, default=150, help="palabras por minuto en modo offline")
    parser.add_argument("--listar-voces", action="store_true",
                        help="lista las voces del sistema (pyttsx3) y cuál se elegiría; no reproduce audio")
    args = parser.parse_args()

    if args.listar_voces:
        try:
            imprimir_voces(voice=args.voice)
        except TTSDependenciaFaltante as exc:
            print(f"ERROR: {exc}")
            return 1
        return 0

    print(f"Modo: {args.mode}")
    try:
        tts = TTSModule(mode=args.mode, voice=args.voice, rate=args.rate)
        if args.mode == "pregrabada" and tts.xtts is not None:
            print("  esperando al servicio XTTS (hasta 60 s)...")
            tts.esperar_servicio_voz(60.0)
    except (TTSDependenciaFaltante, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    frases = [args.text] if args.text else [
        "Hola, buenos días",
        "Yo como mucho",
        "El hermano trabaja",
        "¿Dónde vives?",
        "Gracias, con gusto",
    ]
    for frase in frases:
        print(f"  Diciendo: {frase}")
        tts.speak(frase)
    if args.mode == "pregrabada":
        print(f"  Fuentes: {tts.estado_voz()['fuentes']}")

    tts.cleanup()
    print("Listo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
