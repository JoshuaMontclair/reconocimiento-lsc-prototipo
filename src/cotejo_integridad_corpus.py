#!/usr/bin/env python3
"""Cotejo de integridad del corpus LSC50 archivo por archivo contra el depósito publicado (actividad 1.a).

QUÉ RESUELVE. N-06 dejó pendiente el MD5 de los ZIP publicados en figshare porque exige descargarlos
enteros (9,0 GB los dos que usa la tesis; 10,6 GB los tres del depósito) y el proyecto no los
conserva: solo guarda el material ya extraído en data/raw/LSC50. Este guion cierra el mismo punto
por un camino mejor y casi sin red: el índice central de un ZIP guarda, por CADA entrada, su
tamaño sin comprimir y su CRC-32. Ese índice pesa unos 2 MB por ZIP y se lee por rangos HTTP (la
misma técnica de N-16). Comparando esos valores con el tamaño y el CRC-32 de cada archivo local se
verifican los 8.000 archivos UNO POR UNO, en vez de verificar el envoltorio. Si algo falla, además,
se sabe exactamente qué archivo falla.

QUÉ NO HACE. No descarga los ZIP ni comprueba su MD5 publicado: eso valida el archivo comprimido
como objeto, que aquí no existe en disco. El CRC-32 es una suma de comprobación de 32 bits pensada
para detectar corrupción accidental (truncamientos, bytes cambiados, sincronizaciones a medias), no
manipulación deliberada; para eso haría falta el MD5 del ZIP o un hash por archivo publicado por los
autores, que el depósito no ofrece. Esta limitación se escribe en el informe.

USO
    python src/cotejo_integridad_corpus.py                     # los dos ZIP del depósito
    python src/cotejo_integridad_corpus.py --zip LANDMARKS.zip # solo uno
    python src/cotejo_integridad_corpus.py --limite 50         # prueba rápida: 50 entradas por ZIP
    python src/cotejo_integridad_corpus.py --indice-json X.json # índice ya descargado (sin red)

SALIDAS: <salida>/integridad.json (todo el detalle) e integridad.md (informe). No toca data/.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import platform
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARTICULO = 27383016                      # figshare.com/articles/dataset/LSC50/27383016
ZIPS = ("LANDMARKS.zip", "VIDEOS.zip")   # IMU.zip no se usa en la tesis (D-01)
CORPUS_POR_DEFECTO = RAIZ / "data" / "raw" / "LSC50"
SALIDA_POR_DEFECTO = RAIZ / "models" / "integridad_corpus_2026-09-16"
BLOQUE = 4 << 20                         # 4 MiB por petición HTTP
# Archivos que el sistema operativo crea y que no salen del depósito: se listan aparte, no son sobras.
IGNORADOS = {".DS_Store", "Thumbs.db", ".localized"}


# ---------------------------------------------------------------------------
# Lectura remota del índice del ZIP (rangos HTTP; nada se descarga entero)
# ---------------------------------------------------------------------------

class ArchivoHTTP(io.RawIOBase):
    """Archivo de solo lectura sobre HTTP con Range y cache por bloques. Cada petición vuelve a pasar
    por la redirección de figshare porque la URL firmada caduca en segundos. Es la misma clase que usó
    scripts/cotejo_zip_remoto_figshare.py en N-16."""

    def __init__(self, url: str, tam: int, bloque: int = BLOQUE, abrir=None):
        self.url, self.tam, self.bloque = url, tam, bloque
        self.pos, self.cache, self.bytes = 0, {}, 0
        self.bloques = 0        # bloques distintos traídos
        self.peticiones = 0     # peticiones HTTP reales, reintentos incluidos
        self._abrir = abrir or urllib.request.urlopen

    def seekable(self): return True

    def readable(self): return True

    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.tam + off}[whence]
        return self.pos

    def _leer_bloque(self, k: int) -> bytes:
        if k not in self.cache:
            a = k * self.bloque
            b = min(self.tam, a + self.bloque) - 1
            for intento in range(4):
                try:
                    self.peticiones += 1
                    req = urllib.request.Request(
                        self.url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "tesis-lsc50-cotejo"})
                    with self._abrir(req, timeout=60) as r:
                        if getattr(r, "status", 206) != 206:
                            raise OSError(f"el servidor no devolvió un rango parcial: {r.status}")
                        self.cache[k] = r.read()
                    break
                except Exception:
                    if intento == 3:
                        raise
                    time.sleep(2)
            self.bloques += 1
            self.bytes += len(self.cache[k])
        return self.cache[k]

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.tam:
            n = self.tam - self.pos
        out = bytearray()
        while n > 0:
            k, o = divmod(self.pos, self.bloque)
            trozo = self._leer_bloque(k)[o:o + n]
            if not trozo:
                break
            out += trozo
            self.pos += len(trozo)
            n -= len(trozo)
        return bytes(out)


def metadatos_deposito(articulo: int = ARTICULO, abrir=None) -> dict:
    """Metadatos del artículo de figshare: nombre, tamaño, md5 publicado y URL de cada ZIP."""
    abrir = abrir or urllib.request.urlopen
    with abrir(f"https://api.figshare.com/v2/articles/{articulo}", timeout=60) as r:
        art = json.load(r)
    return {f["name"]: {"nombre": f["name"], "tamano": f["size"], "md5_publicado": f.get("computed_md5"),
                        "url": f["download_url"]} for f in art.get("files", [])}


def indice_remoto(meta: dict, abrir=None) -> dict:
    """Entradas del índice central de un ZIP del depósito, sin descargarlo: nombre, tamaño sin
    comprimir y CRC-32 publicado por entrada."""
    ah = ArchivoHTTP(meta["url"], meta["tamano"], abrir=abrir)
    z = zipfile.ZipFile(ah)
    entradas = [{"nombre": e.filename, "tamano": e.file_size, "crc32": f"{e.CRC:08x}"}
                for e in z.infolist() if not e.is_dir()]
    return {"zip": meta["nombre"], "md5_publicado": meta.get("md5_publicado"),
            "tamano_zip": meta["tamano"], "entradas": entradas,
            "peticiones_http": ah.peticiones, "bloques_http": ah.bloques, "bytes_http": ah.bytes}


# ---------------------------------------------------------------------------
# Cotejo con la copia local
# ---------------------------------------------------------------------------

def crc32_archivo(ruta, bloque: int = 1 << 20) -> tuple[str, int]:
    """CRC-32 (el mismo que guarda el ZIP) y tamaño de un archivo local, leído por trozos."""
    crc, n = 0, 0
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(bloque), b""):
            crc = zlib.crc32(b, crc)
            n += len(b)
    return f"{crc & 0xFFFFFFFF:08x}", n


def ruta_local(nombre_entrada: str, corpus_dir) -> Path:
    """LANDMARKS/BODY_LANDMARKS/x.csv -> <corpus>/LANDMARKS/BODY_LANDMARKS/x.csv. El primer segmento
    del ZIP es la carpeta del corpus, así que la ruta es directa."""
    return Path(corpus_dir) / nombre_entrada


def cotejar_entradas(indice: dict, corpus_dir, limite: int | None = None, avisar=None) -> dict:
    """Compara cada entrada del índice con su archivo local. Devuelve el detalle por categoría:
    coinciden, difieren (tamaño o CRC), y ausentes en local."""
    entradas = indice["entradas"]
    if limite:
        entradas = entradas[:limite]
    coinciden, difieren, ausentes = [], [], []
    bytes_leidos = 0
    for i, e in enumerate(entradas):
        p = ruta_local(e["nombre"], corpus_dir)
        if not p.exists():
            ausentes.append({"entrada": e["nombre"], "motivo": "no está en la copia local"})
            continue
        crc, tam = crc32_archivo(p)
        bytes_leidos += tam
        if crc == e["crc32"] and tam == e["tamano"]:
            coinciden.append(e["nombre"])
        else:
            difieren.append({"entrada": e["nombre"], "crc32_publicado": e["crc32"], "crc32_local": crc,
                             "tamano_publicado": e["tamano"], "tamano_local": tam,
                             "difiere_en": ("tamaño y CRC-32" if tam != e["tamano"] and crc != e["crc32"]
                                            else "tamaño" if tam != e["tamano"] else "CRC-32")})
        if avisar and (i + 1) % 250 == 0:
            avisar(f"    {indice['zip']}: {i + 1}/{len(entradas)} entradas cotejadas "
                   f"({bytes_leidos / 1e9:.2f} GB leídos)")
    return {"zip": indice["zip"], "n_entradas": len(entradas), "coinciden": coinciden,
            "difieren": difieren, "ausentes": ausentes, "bytes_leidos": bytes_leidos}


def sobras_locales(indice: dict, corpus_dir) -> dict:
    """Archivos que están en la copia local, bajo las carpetas que cubre este ZIP, y NO están en el
    depósito. Se separan los que crea el sistema operativo (.DS_Store y compañía)."""
    raices = {e["nombre"].split("/")[0] for e in indice["entradas"]}
    esperados = {e["nombre"] for e in indice["entradas"]}
    sobran, del_sistema = [], []
    for raiz in sorted(raices):
        base = Path(corpus_dir) / raiz
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = f"{raiz}/{p.relative_to(base).as_posix()}"
            if rel in esperados:
                continue
            (del_sistema if p.name in IGNORADOS else sobran).append(rel)
    return {"sobran": sobran, "del_sistema_operativo": del_sistema}


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def _n(x) -> str:
    return f"{x:,}".replace(",", ".")


def resumen(cotejos: list[dict]) -> dict:
    tot = {"entradas": 0, "coinciden": 0, "difieren": 0, "ausentes": 0, "sobran": 0, "bytes_leidos": 0}
    for c in cotejos:
        tot["entradas"] += c["n_entradas"]
        tot["coinciden"] += len(c["coinciden"])
        tot["difieren"] += len(c["difieren"])
        tot["ausentes"] += len(c["ausentes"])
        tot["sobran"] += len(c.get("sobras", {}).get("sobran", []))
        tot["bytes_leidos"] += c["bytes_leidos"]
    tot["integro"] = tot["difieren"] == 0 and tot["ausentes"] == 0
    return tot


def escribir_informe(datos: dict, destino: Path) -> Path:
    t = datos["total"]
    L = ["# Integridad del corpus LSC50 archivo por archivo (actividad 1.a)", ""]
    L.append(f"Generado por `src/cotejo_integridad_corpus.py` el {datos['fecha']}. "
             f"Depósito: figshare, artículo {datos['articulo']}.")
    L.append("")
    L.append("## Qué se comparó")
    L.append("")
    L.append("El índice central de cada ZIP publicado guarda, por cada archivo, su tamaño sin comprimir "
             "y su CRC-32. Ese índice se leyó por rangos HTTP, sin descargar los ZIP "
             f"({_n(datos['bytes_http'])} bytes transferidos en {datos['peticiones_http']} peticiones). "
             "Para cada entrada se calculó el CRC-32 y el tamaño del archivo local y se compararon.")
    L.append("")
    L.append("| ZIP | entradas | coinciden | difieren | ausentes en local | sobran en local |")
    L.append("|---|---|---|---|---|---|")
    for c in datos["cotejos"]:
        L.append(f"| {c['zip']} | {_n(c['n_entradas'])} | {_n(len(c['coinciden']))} | "
                 f"{_n(len(c['difieren']))} | {_n(len(c['ausentes']))} | "
                 f"{_n(len(c.get('sobras', {}).get('sobran', [])))} |")
    L.append(f"| **total** | **{_n(t['entradas'])}** | **{_n(t['coinciden'])}** | **{_n(t['difieren'])}** "
             f"| **{_n(t['ausentes'])}** | **{_n(t['sobran'])}** |")
    L.append("")
    L.append("## Resultado")
    L.append("")
    if t["integro"]:
        L.append(f"**La copia local es fiel al depósito.** Los {_n(t['coinciden'])} archivos cotejados "
                 "tienen el mismo tamaño y el mismo CRC-32 que las entradas publicadas. Ninguno falta y "
                 "ninguno difiere.")
    else:
        dif = f"{_n(t['difieren'])} archivo{'s' if t['difieren'] != 1 else ''}"
        aus = f"{_n(t['ausentes'])} {'faltan' if t['ausentes'] != 1 else 'falta'}"
        L.append(f"**ATENCIÓN: el cotejo NO salió limpio.** {dif} "
                 f"{'difieren' if t['difieren'] != 1 else 'difiere'} y {aus}. "
                 f"El detalle está abajo y en el JSON.")
    for c in datos["cotejos"]:
        if c["difieren"]:
            L.append("")
            L.append(f"### Difieren en {c['zip']}")
            L.append("")
            L.append("| archivo | difiere en | publicado | local |")
            L.append("|---|---|---|---|")
            for d in c["difieren"][:50]:
                L.append(f"| `{d['entrada']}` | {d['difiere_en']} | {d['crc32_publicado']} / "
                         f"{_n(d['tamano_publicado'])} B | {d['crc32_local']} / {_n(d['tamano_local'])} B |")
        if c["ausentes"]:
            L.append("")
            L.append(f"### Ausentes en la copia local ({c['zip']})")
            L.append("")
            for a in c["ausentes"][:50]:
                L.append(f"- `{a['entrada']}`")
    sistema = [f for c in datos["cotejos"] for f in c.get("sobras", {}).get("del_sistema_operativo", [])]
    sobran = [f for c in datos["cotejos"] for f in c.get("sobras", {}).get("sobran", [])]
    if sistema or sobran:
        L.append("")
        L.append("## Archivos locales que no vienen del depósito")
        L.append("")
        if sistema:
            L.append(f"- Creados por el sistema operativo ({len(sistema)}): "
                     + ", ".join(f"`{f}`" for f in sistema[:10]) + ". No son parte del corpus.")
        if sobran:
            L.append(f"- Sin explicación ({len(sobran)}): " + ", ".join(f"`{f}`" for f in sobran[:20]))
    L.append("")
    L.append("## Límites de esta comprobación")
    L.append("")
    L.append("- El CRC-32 son 32 bits: detecta corrupción accidental (truncamientos, bytes cambiados, "
             "copias a medias), que es justo el riesgo que preocupaba. No es una garantía frente a una "
             "manipulación deliberada.")
    L.append("- No se comprobó el MD5 de los ZIP publicados, que exige descargarlos enteros (9,0 GB los "
             "dos cotejados; 10,6 GB los tres que publica el depósito) y que validaría el archivo "
             "comprimido como objeto, no cada archivo del corpus. El MD5 publicado queda anotado en "
             "el JSON para quien quiera hacerlo algún día.")
    L.append("- El depósito publica también IMU.zip, que la tesis no usa (D-01): no se cotejó.")
    L.append("- Se compara contra la versión del depósito que la API devuelve hoy. Si los autores "
             "publicaran una versión nueva, habría que repetir el cotejo.")
    destino.write_text("\n".join(L) + "\n", encoding="utf-8")
    return destino


def main(argv=None):
    p = argparse.ArgumentParser(description="Cotejo de integridad del corpus LSC50 archivo por archivo (1.a).")
    p.add_argument("--corpus", default=str(CORPUS_POR_DEFECTO), help="copia local del corpus")
    p.add_argument("--salida", default=str(SALIDA_POR_DEFECTO), help="directorio de salida")
    p.add_argument("--zip", dest="zips", action="append", choices=list(ZIPS),
                   help="cotejar solo este ZIP (se puede repetir); por defecto, los dos")
    p.add_argument("--limite", type=int, default=None, help="cotejar solo las primeras N entradas de cada ZIP")
    p.add_argument("--indice-json", default=None, help="índice ya guardado (evita la red)")
    p.add_argument("--solo-indice", action="store_true", help="descarga el índice y lo guarda, sin cotejar")
    args = p.parse_args(argv)

    corpus = Path(args.corpus).expanduser().resolve()
    salida = Path(args.salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)
    zips = args.zips or list(ZIPS)
    t0 = time.perf_counter()

    print("=" * 78)
    print("  LSC50 - Integridad del corpus archivo por archivo (actividad 1.a)")
    print("=" * 78)
    print(f"\n  Copia local: {corpus}")

    if args.indice_json:
        indices = json.loads(Path(args.indice_json).read_text(encoding="utf-8"))
        print(f"  Índices leídos de {args.indice_json} (sin red)")
    else:
        meta = metadatos_deposito()
        indices = []
        for nombre in zips:
            if nombre not in meta:
                raise SystemExit(f"el depósito no tiene {nombre}")
            ind = indice_remoto(meta[nombre])
            print(f"  {nombre}: {_n(len(ind['entradas']))} entradas | índice leído con "
                  f"{ind['peticiones_http']} petición(es), {ind['bytes_http'] / 1e6:.1f} MB | "
                  f"md5 publicado {ind['md5_publicado']}")
            indices.append(ind)
        (salida / "indice_deposito.json").write_text(json.dumps(indices, indent=1), encoding="utf-8")
    if args.solo_indice:
        print(f"\n  Índice guardado en {salida / 'indice_deposito.json'}")
        return 0

    cotejos = []
    for ind in indices:
        if ind["zip"] not in zips:
            continue
        print(f"\n  Cotejando {ind['zip']} ...", flush=True)
        c = cotejar_entradas(ind, corpus, limite=args.limite, avisar=lambda m: print(m, flush=True))
        c["sobras"] = sobras_locales(ind, corpus) if not args.limite else {"sobran": [], "del_sistema_operativo": []}
        print(f"    coinciden {_n(len(c['coinciden']))} | difieren {_n(len(c['difieren']))} | "
              f"ausentes {_n(len(c['ausentes']))} | sobran {_n(len(c['sobras']['sobran']))} | "
              f"{c['bytes_leidos'] / 1e9:.2f} GB leídos")
        cotejos.append(c)

    datos = {
        "descripcion": "Cotejo archivo por archivo del corpus local contra el índice del depósito (CRC-32 y tamaño)",
        "actividad": "1.a (integridad del corpus)",
        "articulo": ARTICULO,
        "fecha": _dt.datetime.now().isoformat(timespec="seconds"),
        "comando": " ".join(sys.argv),
        "corpus_dir": str(corpus),
        "entorno": {"python": platform.python_version(), "sistema": platform.platform()},
        "md5_publicado_por_zip": {i["zip"]: i.get("md5_publicado") for i in indices},
        "peticiones_http": sum(i.get("peticiones_http", 0) for i in indices),
        "bytes_http": sum(i.get("bytes_http", 0) for i in indices),
        "limite_entradas": args.limite,
        "cotejos": cotejos,
    }
    datos["total"] = resumen(cotejos)
    (salida / "integridad.json").write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")
    escribir_informe(datos, salida / "integridad.md")
    t = datos["total"]
    print(f"\n  TOTAL: {_n(t['entradas'])} entradas | coinciden {_n(t['coinciden'])} | "
          f"difieren {_n(t['difieren'])} | ausentes {_n(t['ausentes'])} | sobran {_n(t['sobran'])}")
    print(f"  {'COPIA LOCAL FIEL AL DEPOSITO' if t['integro'] else 'ATENCION: el cotejo no salio limpio'}")
    print(f"  Salidas: {salida / 'integridad.json'} y {salida / 'integridad.md'}")
    print(f"  Tiempo total: {time.perf_counter() - t0:.0f} s ({t['bytes_leidos'] / 1e9:.2f} GB leídos)")
    return 0 if t["integro"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
