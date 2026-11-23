"""
Regenera los WAV de la voz pregrabada a partir de las tres tomas de ElevenLabs (docs/VOZ.md, «Cómo regenerarlos»).

Se ejecuta con un intérprete que tenga faster-whisper, soundfile y numpy (el entorno de voz; ver la sección opcional
de requirements-voz.txt) y en macOS (usa afconvert para decodificar el MP3 y remuestrear). Los textos NO se leen de
una lista aparte: salen de src/voz_pregrabada.py (enunciados_pregrabados), es decir, del conversor en modo inventario:
las frases 1-142 son las de los géneros por defecto (tomas 1 y 2) y las 143-146 las variantes en femenino (toma 3).

Pasos (los mismos con que se cortaron los audios el 23-nov-2026):
  1. MP3 → WAV PCM 16 bits a 44,1 kHz (para cortar) y a 16 kHz (para Whisper), con afconvert.
  2. Whisper «base» (CPU, int8, es, beam 5, marcas por palabra, initial_prompt = las 8 primeras frases de la toma)
     transcribe cada toma; las palabras se alinean con el texto esperado (difflib sobre palabras sin tildes ni
     puntuación).
  3. Corte en el punto medio entre el fin de una frase y el inicio de la siguiente (interpolando por longitud de
     texto si falta una marca). Las correcciones a mano (las frases del índice de --correcciones con
     corregido_a_mano) reemplazan a los tiempos automáticos.
  4. Recorte del silencio de los bordes: desde 60 ms antes de la primera ventana de 10 ms por encima de
     (máximo − 40 dB) hasta 70 ms después de la última. Los cortes corregidos a mano NO se recortan: son la ventana
     exacta con un fundido lineal de 10 ms en cada borde (así se rehicieron 026 y 088 el 23-nov-2026).
  4b. Toma 3 (variantes en femenino, 143-146). El modelo leyó también el número de cada frase («ciento cuarenta y
     tres, hola, bienvenida, …»), así que no se alinea: cada frase sale de una ventana fija (VENTANAS_TOMA_3, elegida
     con las marcas de Whisper para dejar fuera los números) con el audio en float64, el inicio y el fin truncados a
     la muestra, recorte del silencio de los bordes con umbral (máximo − 48 dB) desde 80 ms antes hasta 90 ms
     después y fundido lineal de 10 ms en cada borde. Así se cortaron el 23-nov-2026 y el autor los aprobó a oído.
  5. Verificación: cada corte se re-transcribe; dudosos = similitud < 0,8 o duración por carácter fuera de
     [0,45 ; 2,2] × la mediana (la de las tomas 1 y 2). Se listan para escucharlos.
  6. indice.json con construir_indice (sha256 de texto y audio, duración, procedencia, revisión a oído).

Uso:
    <python del entorno de voz> scripts/voz_pregrabada_cortar.py --mp3 toma1.mp3 toma2.mp3 toma3.mp3 \\
        --salida <carpeta> --correcciones models/voz_pregrabada/indice.json
Después: escuchar los dudosos, corregir a mano los cortes que hagan falta (editando «ini_s»/«fin_s» y
«corregido_a_mano» en el indice.json generado, y volviendo a correr con --correcciones apuntando a él), copiar
la carpeta a models/voz_pregrabada/ y comprobar con `python src/voz_pregrabada.py`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import voz_pregrabada as vp  # noqa: E402

FRASES_TOMA_1 = 71          # la toma 1 tiene las frases 1-71 y la toma 2 las 72-142
# Toma 3: variantes en femenino (143-146). Ventana [ini, fin] en segundos sobre el WAV de 44,1 kHz de la toma
# (fin None = hasta el final). Elegidas con las marcas de Whisper para dejar fuera los números que leyó el modelo.
VENTANAS_TOMA_3 = {
    "Hola, bienvenida.": (1.30, 3.05),
    "Buenos días, bienvenida.": (4.45, 6.60),
    "Tú eres bienvenida.": (8.20, 9.70),
    "Yo me siento bienvenida.": (11.30, None),
}
UMBRAL_DB_TOMA_3, ANTES_MS_TOMA_3, DESPUES_MS_TOMA_3 = 48, 80, 90
METODO_TOMA_3 = ("ventana fija de la toma 3 elegida con las marcas de Whisper (sin los números leídos), recorte del "
                 "silencio de los bordes (máximo − 48 dB, 80 ms antes y 90 ms después) y fundido lineal de 10 ms en "
                 "cada borde")


def _sin_tildes(t: str) -> str:
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def norm_palabra(w: str) -> str:
    return re.sub(r"[^a-zñ0-9]", "", _sin_tildes(w))


def norm_frase(t: str) -> str:
    return " ".join(re.sub(r"[^a-zñ0-9 ]", " ", _sin_tildes(t)).split())


def afconvert(origen: Path, destino: Path, hz: int | None = None) -> None:
    formato = "LEI16" + (f"@{hz}" if hz else "")
    subprocess.run(["afconvert", "-f", "WAVE", "-d", formato, "-c", "1", str(origen), str(destino)], check=True)


def alinear(modelo, wav16: Path, frases: list[str], primera: int, dur_s: float) -> dict[int, dict]:
    """Cortes {k: {ini, fin, palabras_alineadas, palabras}} para las frases de una toma (k = índice global)."""
    import soundfile as sf
    a16, _ = sf.read(str(wav16), dtype="float32")
    segs, _ = modelo.transcribe(a16, language="es", word_timestamps=True, beam_size=5,
                                initial_prompt=" ".join(frases[:8]))
    palabras = [(norm_palabra(w.word), w.start, w.end) for s in segs for w in s.words if norm_palabra(w.word)]
    esperado = [(norm_palabra(t), primera + i) for i, f in enumerate(frases) for t in f.split() if norm_palabra(t)]
    sm = difflib.SequenceMatcher(a=[t for t, _ in esperado], b=[p for p, _, _ in palabras], autojunk=False)
    t_ini, t_fin, aciertos = {}, {}, {}
    for blk in sm.get_matching_blocks():
        for d in range(blk.size):
            k = esperado[blk.a + d][1]
            _, s0, s1 = palabras[blk.b + d]
            t_ini[k] = min(t_ini.get(k, 1e9), s0)
            t_fin[k] = max(t_fin.get(k, 0.0), s1)
            aciertos[k] = aciertos.get(k, 0) + 1
    ks = list(range(primera, primera + len(frases)))
    cortes: list[float | None] = [0.0]
    for k in ks[:-1]:
        cortes.append((t_fin[k] + t_ini[k + 1]) / 2 if k in t_fin and (k + 1) in t_ini else None)
    cortes.append(dur_s)
    for i in range(1, len(cortes) - 1):               # interpolación proporcional a la longitud del texto
        if cortes[i] is None:
            j = i
            while cortes[j] is None:
                j += 1
            lo, hi = cortes[i - 1], cortes[j]
            largos = [len(frases[x - 1]) for x in range(i, j + 1)]
            acumulado = lo
            for x in range(i, j):
                acumulado += (hi - lo) * largos[x - i] / sum(largos)
                cortes[x] = acumulado
    return {k: {"ini": round(cortes[i], 3), "fin": round(cortes[i + 1], 3), "palabras_alineadas": aciertos.get(k, 0),
                "palabras": len([t for t in frases[i].split() if norm_palabra(t)])}
            for i, k in enumerate(ks)}


def recortar_silencio(seg, sr: int, umbral_db: float = 40, antes_ms: int = 60, despues_ms: int = 70):
    """Conserva desde `antes_ms` antes de la primera ventana de 10 ms por encima de (máximo − umbral_db) hasta
    `despues_ms` después de la última (en ventanas de 10 ms)."""
    import numpy as np
    hop = int(0.01 * sr)
    n = len(seg) // hop
    db = 20 * np.log10(np.sqrt(np.mean(seg[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    v = np.where(db > db.max() - umbral_db)[0]
    return seg[max(0, (v[0] - antes_ms // 10) * hop):min(len(seg), (v[-1] + despues_ms // 10) * hop)]


def fundido(seg, sr: int, ms: float = 10.0):
    """Fundido lineal de entrada y de salida (cortes corregidos a mano y toma 3)."""
    import numpy as np
    seg = seg.copy()
    n = min(int(ms / 1000 * sr), len(seg) // 2)
    seg[:n] *= np.linspace(0.0, 1.0, n, dtype=seg.dtype)
    seg[-n:] *= np.linspace(1.0, 0.0, n, dtype=seg.dtype)
    return seg


def correcciones_desde(ruta: Path | None) -> dict[int, dict]:
    """{n: {"toma", "ini", "fin"}} de las frases con corte corregido a mano en un indice.json."""
    if ruta is None:
        return {}
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return {f["numero"]: {"toma": f["corte"]["toma"], "ini": f["corte"]["ini_s"], "fin": f["corte"]["fin_s"]}
            for f in datos.get("frases", []) if f.get("corte", {}).get("corregido_a_mano")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Corta las tomas de ElevenLabs en los WAV de la voz pregrabada")
    p.add_argument("--mp3", nargs=3, required=True, type=Path,
                   help="toma 1 (frases 1-71), toma 2 (72-142) y toma 3 (143-146, variantes en femenino)")
    p.add_argument("--salida", required=True, type=Path)
    p.add_argument("--correcciones", type=Path, default=None,
                   help="indice.json del que tomar los cortes corregidos a mano (y la lista de revisados a oído)")
    p.add_argument("--modelo-whisper", default="base")
    args = p.parse_args(argv)

    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel

    por_defecto = vp.textos_inventario()
    frases = vp.textos_posibles()                       # 1-142 (géneros por defecto) + 143-146 (variantes)
    if set(frases[len(por_defecto):]) != set(VENTANAS_TOMA_3):
        raise SystemExit("las variantes de género del conversor no son las de VENTANAS_TOMA_3: hay que generarlas")
    tomas = [(1, por_defecto[:FRASES_TOMA_1], 0), (2, por_defecto[FRASES_TOMA_1:], FRASES_TOMA_1)]
    args.salida.mkdir(parents=True, exist_ok=True)
    trabajo = Path(tempfile.mkdtemp(prefix="voz_cortes_"))
    modelo = WhisperModel(args.modelo_whisper, device="cpu", compute_type="int8")
    cortes: dict[int, dict] = {}
    audio = {}
    for toma, textos, primera in tomas:
        wav, wav16 = trabajo / f"toma{toma}.wav", trabajo / f"toma{toma}_16k.wav"
        afconvert(args.mp3[toma - 1], wav)
        afconvert(args.mp3[toma - 1], wav16, 16000)
        a, sr = sf.read(str(wav), dtype="float32")
        audio[toma] = (a, sr)
        for k, c in alinear(modelo, wav16, textos, primera, len(a) / sr).items():
            cortes[k] = {"parte": toma, **c}
    wav3 = trabajo / "toma3.wav"
    afconvert(args.mp3[2], wav3)
    a3, sr3 = sf.read(str(wav3), dtype="float64")       # float64: como se cortaron el 23-nov-2026
    audio[3] = (a3, sr3)
    for k in range(len(por_defecto), len(frases)):
        ini, fin = VENTANAS_TOMA_3[frases[k]]
        cortes[k] = {"parte": 3, "ini": ini, "fin": round(len(a3) / sr3, 3) if fin is None else fin,
                     "hasta_el_final": fin is None, "metodo": METODO_TOMA_3}
    for n, c in correcciones_desde(args.correcciones).items():
        cortes[n - 1].update({"parte": c["toma"], "ini": c["ini"], "fin": c["fin"], "corregido_a_mano": True})

    verificacion = []
    for k in range(len(frases)):
        c = cortes[k]
        a, sr = audio[c["parte"]]
        if c.get("corregido_a_mano"):            # tiempos a mano: al número de muestra más cercano
            ventana = a[int(round(c["ini"] * sr)):int(round(c["fin"] * sr))]
        elif c.get("hasta_el_final"):
            ventana = a[int(c["ini"] * sr):]
        else:                                     # automáticos y toma 3: truncados, como en el corte original
            ventana = a[int(c["ini"] * sr):int(c["fin"] * sr)]
        if c.get("corregido_a_mano"):
            seg = fundido(ventana, sr)
        elif c["parte"] == 3:
            seg = fundido(recortar_silencio(ventana, sr, UMBRAL_DB_TOMA_3, ANTES_MS_TOMA_3, DESPUES_MS_TOMA_3), sr)
        else:
            seg = recortar_silencio(ventana, sr)
        destino = args.salida / f"{k + 1:03d}.wav"
        sf.write(str(destino), seg, sr, subtype="PCM_16")
        afconvert(destino, trabajo / "v16.wav", 16000)
        x, _ = sf.read(str(trabajo / "v16.wav"), dtype="float32")
        oido = " ".join(s.text for s in modelo.transcribe(x, language="es", beam_size=5)[0]).strip()
        verificacion.append({"n": k + 1, "frase": frases[k], "oido": oido,
                             "similitud": round(difflib.SequenceMatcher(a=norm_frase(frases[k]), b=norm_frase(oido)).ratio(), 3),
                             "dur_s": round(len(seg) / sr, 2),
                             "seg_por_caracter": round(len(seg) / sr / max(1, len(frases[k])), 3)})
    med = statistics.median(v["seg_por_caracter"] for v in verificacion[:len(por_defecto)])   # tomas 1 y 2
    dudosos = [v for v in verificacion if v["similitud"] < 0.8 or v["seg_por_caracter"] > 2.2 * med
               or v["seg_por_caracter"] < 0.45 * med]
    (args.salida / "cortes.json").write_text(json.dumps({str(k): v for k, v in cortes.items()}, ensure_ascii=False,
                                                        indent=1), encoding="utf-8")
    (args.salida / "verificacion.json").write_text(json.dumps(verificacion, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    # «revisado_a_oido» solo para lo que alguien escuchó de verdad: los marcados en el índice anterior. Los dudosos
    # nuevos se listan abajo para escucharlos; se marcan después, editando el índice.
    revisados: set[int] = set()
    if args.correcciones is not None:
        anterior = json.loads(Path(args.correcciones).read_text(encoding="utf-8"))
        revisados = {f["numero"] for f in anterior.get("frases", []) if f.get("revisado_a_oido")}
    rehechos = {n: {"oido": verificacion[n - 1]["oido"],
                    "nota": "corte rehecho a mano tras la escucha; transcripción del corte nuevo"}
                for n in correcciones_desde(args.correcciones)}
    indice = vp.construir_indice(args.salida, verificacion=verificacion, cortes={str(k): v for k, v in cortes.items()},
                                 revisados_a_oido=revisados, rehechos=rehechos)
    (args.salida / "indice.json").write_text(json.dumps(indice, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for f in trabajo.iterdir():
        f.unlink()
    trabajo.rmdir()
    print(f"{len(frases)} cortes en {args.salida}; mediana {med:.3f} s/carácter; dudosos: {len(dudosos)}")
    for v in dudosos:
        print(f"  {v['n']:03d} sim {v['similitud']} dur {v['dur_s']} s | esperado: {v['frase']} | oído: {v['oido']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
