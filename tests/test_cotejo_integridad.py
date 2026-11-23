"""Pruebas del cotejo de integridad del corpus (actividad 1.a): src/cotejo_integridad_corpus.py.

Sin red y sin tocar data/: se construye un ZIP pequeño en un directorio temporal y se sirve por un
doble que responde peticiones de rango sobre sus bytes, igual que haría figshare.
    python -m unittest tests.test_cotejo_integridad -v
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import cotejo_integridad_corpus as ci  # noqa: E402


class RespuestaFalsa(io.BytesIO):
    """Respuesta HTTP mínima: los bytes del rango y el estado 206."""

    def __init__(self, datos, status=206):
        super().__init__(datos)
        self.status = status

    def __enter__(self): return self

    def __exit__(self, *a): self.close()


class ServidorFalso:
    """Sirve unos bytes en memoria respondiendo a la cabecera Range. Cuenta las peticiones."""

    def __init__(self, contenido: bytes, status=206):
        self.contenido, self.status, self.peticiones, self.rangos = contenido, status, 0, []

    def __call__(self, req, timeout=None):
        self.peticiones += 1
        rango = req.headers.get("Range") if hasattr(req, "headers") else None
        if rango is None:                      # petición de la API, no del ZIP
            return RespuestaFalsa(self.contenido, status=200)
        a, b = rango.replace("bytes=", "").split("-")
        a, b = int(a), int(b)
        self.rangos.append((a, b))
        return RespuestaFalsa(self.contenido[a:b + 1], status=self.status)


def construir_zip(archivos: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for nombre, datos in archivos.items():
            z.writestr(nombre, datos)
    return buf.getvalue()


ARCHIVOS = {
    "LANDMARKS/BODY_LANDMARKS/0000_0000_0000.csv": b"indice,x,y\n0,1,2\n",
    "LANDMARKS/BODY_LANDMARKS/0000_0000_0001.csv": b"indice,x,y\n0,3,4\n",
    "LANDMARKS/HANDS_LANDMARKS/0000_0000_0000.csv": b"indice,x\n0,9\n",
}


class TestLecturaDelIndice(unittest.TestCase):

    def setUp(self):
        self.bytes_zip = construir_zip(ARCHIVOS)
        self.servidor = ServidorFalso(self.bytes_zip)
        self.meta = {"nombre": "LANDMARKS.zip", "tamano": len(self.bytes_zip),
                     "md5_publicado": "abc123", "url": "https://ejemplo/zip"}

    def test_el_indice_trae_crc_y_tamano_de_cada_entrada_y_no_baja_el_zip_entero(self):
        ind = ci.indice_remoto(self.meta, abrir=self.servidor)
        self.assertEqual({e["nombre"] for e in ind["entradas"]}, set(ARCHIVOS))
        for e in ind["entradas"]:
            datos = ARCHIVOS[e["nombre"]]
            self.assertEqual(e["tamano"], len(datos))
            self.assertEqual(e["crc32"], f"{zlib.crc32(datos) & 0xFFFFFFFF:08x}")
        self.assertEqual(ind["md5_publicado"], "abc123")
        # se leyeron solo los bloques del final (el índice), no todo el archivo
        self.assertLessEqual(ind["bytes_http"], len(self.bytes_zip))

    def test_las_carpetas_no_cuentan_como_entradas(self):
        conc = dict(ARCHIVOS)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("LANDMARKS/BODY_LANDMARKS/", b"")      # entrada de directorio
            for n, d in conc.items():
                z.writestr(n, d)
        servidor = ServidorFalso(buf.getvalue())
        ind = ci.indice_remoto({**self.meta, "tamano": len(buf.getvalue())}, abrir=servidor)
        self.assertEqual(len(ind["entradas"]), len(conc))

    def test_solo_se_baja_el_final_del_zip_cuando_es_mayor_que_un_bloque(self):
        # El ZIP de las demás pruebas cabe entero en un bloque, así que ahí «no bajar el ZIP
        # entero» no se distingue de bajarlo. Con uno mayor que BLOQUE sí: solo debe pedirse el
        # tramo final, que es donde está el índice. Es la afirmación central del método.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for i in range(30):
                z.writestr(f"LANDMARKS/BODY_LANDMARKS/{i:04d}_0000_0000.csv", os.urandom(200_000))
        grande = buf.getvalue()
        self.assertGreater(len(grande), ci.BLOQUE)
        servidor = ServidorFalso(grande)
        ind = ci.indice_remoto({**self.meta, "tamano": len(grande)}, abrir=servidor)
        self.assertEqual(len(ind["entradas"]), 30)
        self.assertLess(ind["bytes_http"], len(grande))            # no se bajó entero
        self.assertEqual(servidor.rangos[-1][1], len(grande) - 1)  # se pidió el tramo final

    def test_un_servidor_que_ignora_el_rango_falla_en_vez_de_dar_datos_malos(self):
        # Si el servidor devolviera el archivo entero ante una petición de rango, los desplazamientos
        # del índice quedarían corridos y las entradas saldrían mal. Debe fallar, no inventar datos.
        servidor = ServidorFalso(self.bytes_zip, status=200)
        dormir = ci.time.sleep
        ci.time.sleep = lambda s: None          # los reintentos no tienen por qué tardar en la prueba
        try:
            with self.assertRaises((zipfile.BadZipFile, OSError)):
                ci.indice_remoto(self.meta, abrir=servidor)
        finally:
            ci.time.sleep = dormir
        self.assertGreaterEqual(servidor.peticiones, 4)   # reintenta antes de rendirse


class TestCrcYCotejo(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = Path(self.tmp.name)
        for nombre, datos in ARCHIVOS.items():
            p = self.corpus / nombre
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(datos)
        self.indice = {"zip": "LANDMARKS.zip",
                       "entradas": [{"nombre": n, "tamano": len(d),
                                     "crc32": f"{zlib.crc32(d) & 0xFFFFFFFF:08x}"}
                                    for n, d in ARCHIVOS.items()]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_crc32_coincide_con_zlib_y_con_el_del_zip(self):
        p = self.corpus / "LANDMARKS/BODY_LANDMARKS/0000_0000_0000.csv"
        datos = ARCHIVOS["LANDMARKS/BODY_LANDMARKS/0000_0000_0000.csv"]
        self.assertEqual(ci.crc32_archivo(p), (f"{zlib.crc32(datos) & 0xFFFFFFFF:08x}", len(datos)))
        # el cálculo por trozos da lo mismo que de una vez
        self.assertEqual(ci.crc32_archivo(p, bloque=3), ci.crc32_archivo(p, bloque=1 << 20))

    def test_copia_fiel(self):
        c = ci.cotejar_entradas(self.indice, self.corpus)
        self.assertEqual((len(c["coinciden"]), c["difieren"], c["ausentes"]), (3, [], []))
        self.assertEqual(c["bytes_leidos"], sum(len(d) for d in ARCHIVOS.values()))

    def test_un_byte_cambiado_se_detecta_aunque_el_tamano_no_cambie(self):
        p = self.corpus / "LANDMARKS/BODY_LANDMARKS/0000_0000_0001.csv"
        datos = bytearray(p.read_bytes())
        datos[-2] = datos[-2] ^ 0x01
        p.write_bytes(bytes(datos))
        c = ci.cotejar_entradas(self.indice, self.corpus)
        self.assertEqual(len(c["difieren"]), 1)
        d = c["difieren"][0]
        self.assertEqual(d["entrada"], "LANDMARKS/BODY_LANDMARKS/0000_0000_0001.csv")
        self.assertEqual(d["difiere_en"], "CRC-32")
        self.assertEqual(d["tamano_local"], d["tamano_publicado"])

    def test_archivo_truncado_se_detecta_como_diferencia_de_tamano(self):
        p = self.corpus / "LANDMARKS/HANDS_LANDMARKS/0000_0000_0000.csv"
        p.write_bytes(p.read_bytes()[:-3])
        c = ci.cotejar_entradas(self.indice, self.corpus)
        self.assertEqual(len(c["difieren"]), 1)
        self.assertIn("tamaño", c["difieren"][0]["difiere_en"])

    def test_archivo_que_falta(self):
        (self.corpus / "LANDMARKS/BODY_LANDMARKS/0000_0000_0000.csv").unlink()
        c = ci.cotejar_entradas(self.indice, self.corpus)
        self.assertEqual(len(c["ausentes"]), 1)
        self.assertEqual(len(c["coinciden"]), 2)

    def test_limite(self):
        c = ci.cotejar_entradas(self.indice, self.corpus, limite=2)
        self.assertEqual(c["n_entradas"], 2)

    def test_sobras_locales_separan_las_del_sistema_operativo(self):
        (self.corpus / "LANDMARKS/BODY_LANDMARKS/.DS_Store").write_bytes(b"x")
        (self.corpus / "LANDMARKS/BODY_LANDMARKS/copia_de_seguridad.csv").write_bytes(b"y")
        s = ci.sobras_locales(self.indice, self.corpus)
        self.assertEqual(s["del_sistema_operativo"], ["LANDMARKS/BODY_LANDMARKS/.DS_Store"])
        self.assertEqual(s["sobran"], ["LANDMARKS/BODY_LANDMARKS/copia_de_seguridad.csv"])


class TestResumenEInforme(unittest.TestCase):

    def _cotejo(self, coinciden=3, difieren=0, ausentes=0, sobran=0):
        return {"zip": "LANDMARKS.zip", "n_entradas": coinciden + difieren + ausentes,
                "coinciden": [f"a{i}" for i in range(coinciden)],
                "difieren": [{"entrada": f"d{i}", "crc32_publicado": "1", "crc32_local": "2",
                              "tamano_publicado": 10, "tamano_local": 10, "difiere_en": "CRC-32"}
                             for i in range(difieren)],
                "ausentes": [{"entrada": f"f{i}", "motivo": "no está"} for i in range(ausentes)],
                "bytes_leidos": 100,
                "sobras": {"sobran": [f"s{i}" for i in range(sobran)], "del_sistema_operativo": []}}

    def test_integro_solo_si_no_hay_diferencias_ni_ausencias(self):
        self.assertTrue(ci.resumen([self._cotejo()])["integro"])
        self.assertFalse(ci.resumen([self._cotejo(difieren=1)])["integro"])
        self.assertFalse(ci.resumen([self._cotejo(ausentes=1)])["integro"])
        # un archivo de más en local no rompe la integridad de lo publicado, pero se cuenta
        r = ci.resumen([self._cotejo(sobran=2)])
        self.assertTrue(r["integro"])
        self.assertEqual(r["sobran"], 2)

    def test_el_informe_dice_lo_que_toca_en_cada_caso(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {"fecha": "2026-09-16T10:00:00", "articulo": 27383016,
                    "peticiones_http": 2, "bytes_http": 3_800_000}
            bien = {**base, "cotejos": [self._cotejo()], "total": ci.resumen([self._cotejo()])}
            texto = ci.escribir_informe(bien, Path(tmp) / "ok.md").read_text(encoding="utf-8")
            self.assertIn("La copia local es fiel al depósito", texto)
            self.assertIn("CRC-32 son 32 bits", texto)          # el límite se declara siempre
            mal_c = self._cotejo(difieren=1)
            mal = {**base, "cotejos": [mal_c], "total": ci.resumen([mal_c])}
            texto = ci.escribir_informe(mal, Path(tmp) / "mal.md").read_text(encoding="utf-8")
            self.assertIn("el cotejo NO salió limpio", texto)
            self.assertIn("`d0`", texto)

    def test_ruta_local_es_directa(self):
        self.assertEqual(ci.ruta_local("VIDEOS/COLOR_BODY/x.avi", "/corpus"),
                         Path("/corpus/VIDEOS/COLOR_BODY/x.avi"))


class TestCorridaReal(unittest.TestCase):
    """Sobre la salida ya guardada, si existe: no vuelve a medir ni toca la red."""

    def test_la_corrida_guardada_declara_copia_fiel(self):
        p = BASE_DIR / "models" / "integridad_corpus_2026-09-16" / "integridad.json"
        if not p.exists():
            self.skipTest("todavía no se ha corrido el cotejo completo")
        d = json.loads(p.read_text(encoding="utf-8"))
        t = d["total"]
        self.assertEqual(t["entradas"], t["coinciden"] + t["difieren"] + t["ausentes"])
        self.assertEqual(t["entradas"], 8000)
        self.assertTrue(t["integro"], f"difieren {t['difieren']}, ausentes {t['ausentes']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
