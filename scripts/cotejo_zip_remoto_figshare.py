#!/usr/bin/env python3
"""Coteja archivos concretos del depósito figshare de LSC50 SIN descargar los ZIP completos:
lee el índice central del ZIP por rangos HTTP, compara CRC-32/tamaño de las entradas y extrae solo
los archivos pedidos para compararlos byte a byte (SHA-1) con la copia local en data/raw/LSC50."""
import sys, json, hashlib, io, os, urllib.request, zipfile, time
# Uso: python scripts/cotejo_zip_remoto_figshare.py data/raw/LSC50 [muestra ...]   (por defecto, los 3 pares duplicados)
# Depósito: https://figshare.com/articles/dataset/LSC50/27383016 (artículo 27383016 de la API v2 de figshare).
with urllib.request.urlopen("https://api.figshare.com/v2/articles/27383016", timeout=60) as r: ART = json.load(r)
LOCAL = sys.argv[1]
MUESTRAS = sys.argv[2:] or ["0002_0001_0000","0002_0001_0001","0008_0002_0000","0008_0002_0001","0016_0002_0000","0016_0002_0001"]
BLOQUE = 4 << 20   # 4 MiB por petición

class ArchivoHTTP(io.RawIOBase):
    """Archivo de solo lectura sobre HTTP con Range; cache por bloques. Cada petición pasa por la
    redirección de ndownloader (la URL firmada de S3 caduca en 10 s)."""
    def __init__(self, url, tam): self.url, self.tam, self.pos, self.cache, self.peticiones, self.bytes = url, tam, 0, {}, 0, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.tam + off}[whence]; return self.pos
    def _bloque(self, k):
        if k not in self.cache:
            a = k * BLOQUE; b = min(self.tam, a + BLOQUE) - 1
            for intento in range(4):
                try:
                    req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "tesis-lsc50-cotejo"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        assert r.status == 206, r.status
                        self.cache[k] = r.read(); break
                except Exception as e:
                    if intento == 3: raise
                    time.sleep(2)
            self.peticiones += 1; self.bytes += len(self.cache[k])
        return self.cache[k]
    def read(self, n=-1):
        if n < 0 or self.pos + n > self.tam: n = self.tam - self.pos
        out = bytearray()
        while n > 0:
            k, o = divmod(self.pos, BLOQUE); b = self._bloque(k)[o:o + n]
            if not b: break
            out += b; self.pos += len(b); n -= len(b)
        return bytes(out)

def sha1(datos): return hashlib.sha1(datos).hexdigest()
def sha1_archivo(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

for f in ART["files"]:
    if f["name"] not in ("VIDEOS.zip", "LANDMARKS.zip"): continue
    print(f"\n===== {f['name']} ({f['size']:,} bytes, md5 publicado {f['computed_md5']}) =====", flush=True)
    ah = ArchivoHTTP(f["download_url"], f["size"])
    z = zipfile.ZipFile(ah)
    entradas = z.infolist(); print(f"  entradas en el ZIP: {len(entradas):,} | índice leído con {ah.peticiones} peticiones ({ah.bytes/1e6:.1f} MB)", flush=True)
    objetivo = [e for e in entradas if any(m in os.path.basename(e.filename) for m in MUESTRAS) and (
        "COLOR_BODY" in e.filename or "BODY_LANDMARKS" in e.filename or "HAND_LANDMARKS" in e.filename)]
    objetivo.sort(key=lambda e: e.filename)
    print(f"  entradas objetivo: {len(objetivo)}")
    # 1) comparación por CRC-32 y tamaño dentro del propio depósito (sin descargar contenido)
    por_carpeta = {}
    for e in objetivo: por_carpeta.setdefault(os.path.dirname(e.filename), []).append(e)
    for carpeta, lst in por_carpeta.items():
        d = {os.path.basename(e.filename): e for e in lst}
        for a, b in [(MUESTRAS[i], MUESTRAS[i+1]) for i in range(0, 6, 2)]:
            ea = next((e for n, e in d.items() if n.startswith(a)), None); eb = next((e for n, e in d.items() if n.startswith(b)), None)
            if not ea or not eb: print(f"    {carpeta}: {a}/{b} no encontrados"); continue
            igual = (ea.CRC == eb.CRC and ea.file_size == eb.file_size)
            print(f"    {carpeta:55s} {a} vs {b}: CRC {ea.CRC:08x}/{eb.CRC:08x} · {ea.file_size:,}/{eb.file_size:,} bytes → {'IDÉNTICOS EN EL DEPÓSITO' if igual else 'distintos en el depósito'}")
    # 2) extracción de cada entrada y comparación con la copia local
    print("  --- extracción parcial y cotejo con data/raw/LSC50 ---", flush=True)
    for e in objetivo:
        datos = z.read(e)
        rel = e.filename.split("LSC50/")[-1] if "LSC50/" in e.filename else e.filename
        cand = [os.path.join(LOCAL, rel)] + [os.path.join(dp, os.path.basename(e.filename)) for dp, dn, fn in os.walk(LOCAL) if os.path.basename(e.filename) in fn and os.path.basename(os.path.dirname(e.filename)) == os.path.basename(dp)]
        loc = next((c for c in cand if os.path.exists(c)), None)
        hr = sha1(datos); hl = sha1_archivo(loc) if loc else None
        print(f"    {e.filename:70s} {len(datos):>10,} B  sha1 {hr[:12]}  local: {'IGUAL' if hl == hr else ('DIFERENTE ' + (hl or '')[:12]) if loc else 'no encontrado'}", flush=True)
    print(f"  peticiones HTTP totales: {ah.peticiones} | descargado {ah.bytes/1e6:.1f} MB")
