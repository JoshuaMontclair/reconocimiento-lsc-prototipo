"""
Voz pregrabada del prototipo (decisión del autor del 23-nov-2026; ver docs/VOZ.md).

QUÉ ES. En modo `--conversor inventario` el prototipo solo puede decir un conjunto cerrado de enunciados: las
50 glosas solas y las 92 secuencias previstas del inventario (src/conversor_glosas.py, modo estricto). Con los
géneros por defecto son 142 textos distintos. El autor los generó con ElevenLabs (voz «Robert - Colombian,
Paisa» de la biblioteca de ElevenLabs) y se cortaron en 142 WAV (44,1 kHz, mono, PCM de 16 bits) que viven en
models/voz_pregrabada/ junto con su índice (indice.json). src/tts.py, en modo «pregrabada», busca el enunciado
EXACTO en el índice y reproduce su WAV; si no está, cae a XTTS-v2 (src/voz_xtts_servicio.py) y, si tampoco hay,
a la voz local de macOS (Mónica).

QUÉ HAY AQUÍ (solo biblioteca estándar: el proyecto no gana dependencias).
  · enunciados_inventario(): los textos que el conversor en modo inventario puede producir, enumerados DESDE EL
    CÓDIGO (léxico y tabla INVENTARIO), no desde una lista aparte. variantes_de_genero() da los que cambian con
    --genero-primera f / --genero-segunda f (no están pregrabados: van al respaldo).
  · IndiceVozPregrabada: carga y valida indice.json (textos, sha256 de cada texto y de cada WAV, duración,
    números consecutivos, archivos presentes) y busca por texto exacto (normalizado a NFC, sin más: «Gracias»
    no encuentra «Gracias.»).
  · cobertura(): qué parte de los enunciados posibles tiene audio.
  · construir_indice(): arma indice.json a partir de la carpeta de WAV y de los archivos de corte y
    verificación (lo usa scripts/voz_pregrabada_cortar.py al regenerar los audios).

CLI:
    python src/voz_pregrabada.py                 # valida el índice y comprueba la cobertura (sale 1 si falla)
    python src/voz_pregrabada.py --json          # lo mismo, en JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
import wave
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CARPETA_POR_DEFECTO = ROOT_DIR / "models" / "voz_pregrabada"
RUTA_INDICE_POR_DEFECTO = CARPETA_POR_DEFECTO / "indice.json"
VERSION_INDICE = 1
TOLERANCIA_DURACION_S = 0.01

# Procedencia de los audios (se copia al índice; docs/VOZ.md lo explica).
FUENTE_ELEVENLABS = {
    "proveedor": "ElevenLabs",
    "voz": "Robert - Colombian, Paisa",
    "biblioteca": "biblioteca de voces de ElevenLabs",
    "generado_por": "el autor de la tesis",
    "fecha": "2026-11-23",
    "licencia": "plan de pago de ElevenLabs que permite el uso y la distribución de los audios generados",
    "atribucion": "Voz «Robert - Colombian, Paisa» de la biblioteca de ElevenLabs",
    "archivos_origen": [
        "ElevenLabs_2026-11-23T05_29_16_Robert - Colombian, Paisa_pvc_s50_sim75_v4.mp3",
        "ElevenLabs_2026-11-23T05_29_57_Robert - Colombian, Paisa_pvc_s50_sim75_v4.mp3",
    ],
    "formato_origen": "MP3 128 kb/s, 44 100 Hz, mono; dos tomas (frases 1-71 y 72-142) con pausas de 1,5 s",
}
METODO_CORTE = {
    "herramienta": "faster-whisper, modelo «base» (CPU, int8), idioma es, marcas de tiempo por palabra",
    "pasos": [
        "MP3 → WAV PCM 16 bits a 44,1 kHz (corte) y a 16 kHz (alineación), con afconvert de macOS",
        "transcripción de cada toma y alineación de sus palabras con el texto esperado (difflib.SequenceMatcher "
        "sobre palabras normalizadas sin tildes ni puntuación)",
        "corte en el punto medio entre el fin de la última palabra de una frase y el inicio de la primera de la "
        "siguiente; si falta una marca, interpolación proporcional a la longitud del texto",
        "recorte del silencio de los bordes: se conserva desde 60 ms antes de la primera ventana de 10 ms que "
        "supera (máximo − 40 dB) hasta 70 ms después de la última",
        "verificación: re-transcripción de cada corte con el mismo modelo y similitud de caracteres con el texto "
        "esperado; dudosos = similitud < 0,8 o duración por carácter fuera de [0,45 ; 2,2] × la mediana",
        "cortes corregidos a mano (026 y 088): ventana exacta [ini_s, fin_s] con fundido lineal de 10 ms en cada "
        "borde y sin recorte de silencio; tiempos comprobados por correlación con la toma",
    ],
    "dudosos": 16,
    "revision_a_oido": "el autor escuchó los 16 dudosos: 14 correctos (la transcripción automática falla en "
                       "palabras sueltas) y 2 cortados (026 «Mal.» y 088 «Tú comes.»), cuyos tiempos se "
                       "corrigieron a mano; el autor aprobó los dos cortes rehechos",
}


# ---------------------------------------------------------------------------
# Enunciados que el conversor en modo inventario puede producir (desde el código)
# ---------------------------------------------------------------------------

def normalizar(texto: str) -> str:
    """Forma canónica del texto para buscarlo: NFC y nada más (ni minúsculas, ni espacios, ni puntuación)."""
    return unicodedata.normalize("NFC", texto) if isinstance(texto, str) else texto


def sha256_texto(texto: str) -> str:
    return hashlib.sha256(normalizar(texto).encode("utf-8")).hexdigest()


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 16), b""):
            h.update(bloque)
    return h.hexdigest()


def enunciados_inventario(genero_primera: str | None = None, genero_segunda: str | None = None) -> list[dict]:
    """Enunciados (sin respaldo) que da ConversorGlosas(estricto=True) con esos géneros, en el orden de las
    frases del índice: primero las 50 glosas solas por id (SIGN_VOCABULARY), luego las 92 secuencias de
    INVENTARIO. Cada uno: {"texto", "origen", "glosas"}. Sin repetidos (se conserva la primera aparición)."""
    from src.conversor_glosas import INVENTARIO, ConversorGlosas
    from src.grammar_mapper import SIGN_VOCABULARY

    conv = ConversorGlosas(genero_primera=genero_primera, genero_segunda=genero_segunda, estricto=True)
    salida: list[dict] = []
    vistos: set[str] = set()

    def anadir(glosas: list[str], origen: str) -> None:
        e = conv.explicar(glosas)
        if e["respaldo"] or not e["frase"]:
            raise RuntimeError(f"el inventario da respaldo para {glosas}: {e['motivo']}")
        texto = normalizar(e["frase"])
        if texto not in vistos:
            vistos.add(texto)
            salida.append({"texto": texto, "origen": origen, "glosas": list(glosas)})

    for i in sorted(SIGN_VOCABULARY):
        anadir([SIGN_VOCABULARY[i]], f"seña sola: {SIGN_VOCABULARY[i]}")
    for s in INVENTARIO:
        anadir(list(s["glosas"]), f"{s['id']}: {' + '.join(s['glosas'])}")
    return salida


def textos_inventario(genero_primera: str | None = None, genero_segunda: str | None = None) -> list[str]:
    return [e["texto"] for e in enunciados_inventario(genero_primera, genero_segunda)]


def variantes_de_genero() -> dict[str, list[str]]:
    """Textos que aparecen con algún género femenino y no con los de por defecto (m, m), por combinación."""
    base = set(textos_inventario())
    variantes = {}
    for g1 in ("m", "f"):
        for g2 in ("m", "f"):
            extra = [t for t in textos_inventario(g1, g2) if t not in base]
            if extra:
                variantes[f"yo={g1}, tú={g2}"] = extra
    return variantes


# ---------------------------------------------------------------------------
# Audio WAV (biblioteca estándar)
# ---------------------------------------------------------------------------

def info_wav(ruta: Path) -> dict:
    """Canales, frecuencia, bits y duración de un WAV PCM (wave de la biblioteca estándar)."""
    with wave.open(str(ruta), "rb") as w:
        n, sr = w.getnframes(), w.getframerate()
        return {"canales": w.getnchannels(), "frecuencia_hz": sr, "bits": 8 * w.getsampwidth(),
                "muestras": n, "duracion_s": round(n / sr, 3) if sr else 0.0}


# ---------------------------------------------------------------------------
# Índice
# ---------------------------------------------------------------------------

class IndiceInvalido(ValueError):
    """indice.json no existe, no se puede leer o no pasa la validación."""


def validar_indice(datos: dict, carpeta: Path, comprobar_audio: bool = True) -> list[str]:
    """Errores del índice (lista vacía si es válido). Función pura salvo la lectura de los WAV."""
    errores: list[str] = []
    if not isinstance(datos, dict):
        return ["el índice no es un objeto JSON"]
    if datos.get("version") != VERSION_INDICE:
        errores.append(f"versión {datos.get('version')!r} (se esperaba {VERSION_INDICE})")
    frases = datos.get("frases")
    if not isinstance(frases, list) or not frases:
        return errores + ["sin lista «frases»"]
    textos, archivos = set(), set()
    for i, f in enumerate(frases, start=1):
        if not isinstance(f, dict):
            errores.append(f"frase {i}: no es un objeto")
            continue
        texto, archivo = f.get("texto"), f.get("archivo")
        if f.get("numero") != i:
            errores.append(f"frase {i}: número {f.get('numero')!r} (deben ser consecutivos desde 1)")
        if not isinstance(texto, str) or not texto.strip():
            errores.append(f"frase {i}: texto vacío")
            continue
        if normalizar(texto) != texto:
            errores.append(f"frase {i}: el texto no está en NFC")
        if f.get("sha256_texto") != sha256_texto(texto):
            errores.append(f"frase {i}: sha256_texto no corresponde a {texto!r}")
        if texto in textos:
            errores.append(f"frase {i}: texto repetido {texto!r}")
        textos.add(texto)
        if not isinstance(archivo, str) or Path(archivo).name != archivo or not archivo.endswith(".wav"):
            errores.append(f"frase {i}: archivo {archivo!r} inválido (nombre .wav sin carpeta)")
            continue
        if archivo in archivos:
            errores.append(f"frase {i}: archivo repetido {archivo}")
        archivos.add(archivo)
        ruta = carpeta / archivo
        if not ruta.is_file():
            errores.append(f"frase {i}: falta {archivo}")
            continue
        if comprobar_audio:
            try:
                info = info_wav(ruta)
            except (wave.Error, EOFError, OSError) as exc:
                errores.append(f"frase {i}: {archivo} no es un WAV legible ({exc})")
                continue
            if info["canales"] != 1 or info["muestras"] <= 0:
                errores.append(f"frase {i}: {archivo} debe ser mono y no vacío")
            dur = f.get("duracion_s")
            if not isinstance(dur, (int, float)) or abs(dur - info["duracion_s"]) > TOLERANCIA_DURACION_S:
                errores.append(f"frase {i}: duracion_s {dur!r} y el archivo dura {info['duracion_s']}")
            if f.get("sha256_audio") is not None and f.get("sha256_audio") != sha256_archivo(ruta):
                errores.append(f"frase {i}: sha256_audio no corresponde a {archivo}")
    return errores


class IndiceVozPregrabada:
    """Índice de enunciados pregrabados: texto exacto → WAV."""

    def __init__(self, datos: dict, carpeta: Path):
        self.datos = datos
        self.carpeta = Path(carpeta)
        self.ruta = self.carpeta / "indice.json"
        self._por_texto: dict[str, Path] = {normalizar(f["texto"]): self.carpeta / f["archivo"]
                                            for f in datos["frases"]}

    @classmethod
    def cargar(cls, ruta: str | Path | None = None, comprobar_audio: bool = True) -> "IndiceVozPregrabada":
        """Lee y valida indice.json. IndiceInvalido si falta o no pasa la validación."""
        ruta = Path(ruta) if ruta is not None else RUTA_INDICE_POR_DEFECTO
        if ruta.is_dir():
            ruta = ruta / "indice.json"
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise IndiceInvalido(f"no existe el índice de voz pregrabada {ruta}") from exc
        except (OSError, ValueError) as exc:
            raise IndiceInvalido(f"no se pudo leer {ruta}: {exc}") from exc
        errores = validar_indice(datos, ruta.parent, comprobar_audio=comprobar_audio)
        if errores:
            raise IndiceInvalido(f"{ruta} no es válido ({len(errores)} errores): " + "; ".join(errores[:5]))
        indice = cls(datos, ruta.parent)
        indice.ruta = ruta
        return indice

    def buscar(self, texto) -> Path | None:
        """WAV del enunciado EXACTO (comparado en NFC) o None."""
        if not isinstance(texto, str) or not texto:
            return None
        return self._por_texto.get(normalizar(texto))

    @property
    def textos(self) -> list[str]:
        return list(self._por_texto)

    def __len__(self) -> int:
        return len(self._por_texto)

    def __contains__(self, texto) -> bool:
        return self.buscar(texto) is not None

    def describir(self) -> str:
        f = self.datos.get("fuente", {})
        return f"{len(self)} enunciados, {f.get('proveedor', '?')} «{f.get('voz', '?')}»"


def existe_indice(ruta: str | Path | None = None) -> bool:
    ruta = Path(ruta) if ruta is not None else RUTA_INDICE_POR_DEFECTO
    return (ruta / "indice.json").is_file() if ruta.is_dir() else ruta.is_file()


def cobertura(indice: IndiceVozPregrabada, textos: list[str]) -> dict:
    """Qué parte de los textos tiene audio, y qué audios no corresponden a ningún texto."""
    faltan = [t for t in textos if t not in indice]
    conjunto = {normalizar(t) for t in textos}
    sobran = [t for t in indice.textos if t not in conjunto]
    n = len(textos)
    return {"n": n, "cubiertos": n - len(faltan), "porcentaje": round(100.0 * (n - len(faltan)) / n, 2) if n else None,
            "faltan": faltan, "sobran": sobran}


# ---------------------------------------------------------------------------
# Construcción del índice (al regenerar los audios)
# ---------------------------------------------------------------------------

def construir_indice(carpeta: Path, verificacion: list[dict] | None = None, cortes: dict | None = None,
                     revisados_a_oido: set[int] | None = None, rehechos: dict[int, dict] | None = None,
                     fuente: dict | None = None, metodo: dict | None = None) -> dict:
    """indice.json para los WAV NNN.wav de `carpeta`, con los textos enumerados desde el código.

    verificacion: lista [{n, frase, oido, similitud, ...}] de la re-transcripción; cortes: {"k": {parte, ini,
    fin, corregido_a_mano?}} con k = n − 1; revisados_a_oido: números escuchados por el autor; rehechos:
    {n: {"oido": ..., "nota": ...}} para los cortes corregidos a mano (su verificación es la del corte nuevo)."""
    enunciados = enunciados_inventario()
    verif = {int(v["n"]): v for v in (verificacion or [])}
    revisados_a_oido = set(revisados_a_oido or ())
    rehechos = dict(rehechos or {})
    frases = []
    total = 0
    for n, e in enumerate(enunciados, start=1):
        archivo = f"{n:03d}.wav"
        ruta = carpeta / archivo
        info = info_wav(ruta)
        total += ruta.stat().st_size
        entrada = {"numero": n, "texto": e["texto"], "sha256_texto": sha256_texto(e["texto"]), "archivo": archivo,
                   "duracion_s": info["duracion_s"], "bytes": ruta.stat().st_size, "sha256_audio": sha256_archivo(ruta),
                   "origen": e["origen"]}
        c = (cortes or {}).get(str(n - 1))
        if c is not None:
            entrada["corte"] = {"toma": c.get("parte"), "ini_s": c.get("ini"), "fin_s": c.get("fin"),
                                "corregido_a_mano": bool(c.get("corregido_a_mano", False))}
            if c.get("corregido_a_mano"):
                entrada["corte"]["metodo"] = ("ventana exacta corregida a mano, fundido lineal de 10 ms en cada "
                                              "borde, sin recorte de silencio")
        v = verif.get(n)
        if v is not None:
            if v.get("frase") is not None and normalizar(v["frase"]) != e["texto"]:
                raise ValueError(f"la verificación de {n} es de {v['frase']!r}, no de {e['texto']!r}")
            entrada["verificacion_whisper"] = {"transcripcion": v.get("oido"), "similitud": v.get("similitud")}
        if n in rehechos:
            entrada["verificacion_whisper"] = {"transcripcion": rehechos[n].get("oido"), "similitud": None,
                                               "nota": rehechos[n].get("nota")}
        entrada["revisado_a_oido"] = n in revisados_a_oido
        entrada["corte_rehecho"] = n in rehechos
        frases.append(entrada)
    infos = {(i["canales"], i["frecuencia_hz"], i["bits"]) for i in (info_wav(carpeta / f["archivo"]) for f in frases)}
    return {
        "version": VERSION_INDICE,
        "descripcion": "Enunciados pregrabados del prototipo: los que el conversor de glosas en modo inventario "
                       "(src/conversor_glosas.py, estricto, géneros por defecto) puede producir. Ver docs/VOZ.md.",
        "conversor": {"modulo": "src/conversor_glosas.py", "modo": "inventario (estricto=True)",
                      "generos": {"yo": "m", "tú": "m"}, "n_glosas_solas": 50, "n_secuencias": 92,
                      "n_enunciados": len(frases)},
        "audio": {"formato": "WAV PCM", "combinaciones_canales_hz_bits": sorted(list(x) for x in infos),
                  "bytes_total": total},
        "fuente": dict(fuente or FUENTE_ELEVENLABS),
        "corte": dict(metodo or METODO_CORTE),
        "n_revisados_a_oido": len(revisados_a_oido),
        "frases": frases,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def comprobar(ruta: str | Path | None = None) -> dict:
    """Valida el índice y mide la cobertura del modo inventario (géneros por defecto y variantes)."""
    indice = IndiceVozPregrabada.cargar(ruta)
    base = cobertura(indice, textos_inventario())
    variantes = variantes_de_genero()
    total = sum(f.get("bytes") or (indice.carpeta / f["archivo"]).stat().st_size for f in indice.datos["frases"])
    return {"indice": str(indice.ruta), "enunciados_en_indice": len(indice), "cobertura_generos_por_defecto": base,
            "variantes_de_genero_sin_audio": {k: [t for t in v if t not in indice] for k, v in variantes.items()},
            "bytes_wav_total": total,
            "duracion_total_s": round(sum(f["duracion_s"] for f in indice.datos["frases"]), 2)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Valida el índice de voz pregrabada y su cobertura del inventario")
    p.add_argument("--indice", default=None, help=f"default {RUTA_INDICE_POR_DEFECTO.relative_to(ROOT_DIR)}")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    try:
        r = comprobar(args.indice)
    except IndiceInvalido as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        c = r["cobertura_generos_por_defecto"]
        print(f"Índice: {r['indice']} ({r['enunciados_en_indice']} enunciados, {r['bytes_wav_total']} bytes de WAV, "
              f"{r['duracion_total_s']} s de audio)")
        print(f"Cobertura del modo inventario (géneros por defecto): {c['cubiertos']}/{c['n']} ({c['porcentaje']} %)")
        for t in c["faltan"]:
            print(f"  falta: {t}")
        for t in c["sobran"]:
            print(f"  sobra: {t}")
        for k, v in r["variantes_de_genero_sin_audio"].items():
            print(f"  sin audio con {k} (van al respaldo): {', '.join(v)}")
    c = r["cobertura_generos_por_defecto"]
    return 0 if not c["faltan"] and not c["sobran"] else 1


if __name__ == "__main__":
    sys.exit(main())
