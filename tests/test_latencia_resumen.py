"""
Pruebas de src/latencia_resumen.py con registros JSONL sintéticos (pendientes de la revisión de N-34).

Cubren:
  (a) «síntesis», L_total y L_resp separadas entre ensayos con tts_sintesis_antes_de_t5 = true y el
      resto (false, ausente o null), para no promediar ≈0 ms con ≈450 ms;
  (b) W y «captura» promediadas solo sobre las ventanas que llegaron a inferirse (pendiente de N-14);
  (c) conteo de tts_ocupado_en_t5 y de los umbrales de confianza (parametros.umbral / umbral_origen);
  (d) errores de reproducción de la voz (N-37): tts_errores_reproduccion y
      tts_ultimo_error_reproduccion por sesión y en total, con un caso en el que el error aparece a
      mitad de la sesión;
  y la compatibilidad con registros escritos antes de esos campos (formato de
  registros/latencia_2026-09-14.jsonl), incluidos los dos registros reales, que solo se leen.
Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_latencia_resumen -v
Los dos registros reales se leen de registros/ o, si se define LSC_REGISTROS_DIR, de esa carpeta: con
una sesión del prototipo en curso no se debe tocar registros/ (el prototipo escribe ahí).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src import latencia_resumen as lr  # noqa: E402

MS = 1_000_000   # ns por ms
NUEVO, ANTIGUO = "nuevo", "antiguo"


def ensayo(estado="aceptada", modo="aislado", W_ms=2500.0, sintesis_ms=None, tts=None, ocupado=None,
           umbral=0.5, origen="models/m.umbral.json", formato=NUEVO, N=75, sin_parametros=False,
           errores=None, mensaje=None, sesion="s1", id_ensayo=1):
    """Registro de ensayo con la forma que escribe src/translator.py.

    estado decide qué marcas existen, como en el prototipo: rechazada_manos / rechazada_incompleta no
    infieren; rechazada_confianza infiere; aceptada en modo aislado llega a t5. formato=ANTIGUO omite los
    campos que aún no existían el 14-sep-2026 (tts_ocupado_en_t5, parametros.umbral_origen,
    tts_errores_reproduccion, tts_ultimo_error_reproduccion); en formato NUEVO los dos campos de voz
    van siempre presentes, con null cuando el motor no los informa (que no es «cero errores»).
    """
    infiere = estado in ("aceptada", "rechazada_confianza")
    completo = estado == "aceptada" and modo == "aislado"
    W = int(W_ms * MS) if N else None
    etapas = {k: None for k in lr.ETAPAS}
    if N:
        etapas["captura"] = W
        etapas["extraccion"] = 12 * MS
    L_resp = None
    if infiere:
        etapas["preprocesamiento"] = MS // 2
        etapas["inferencia"] = 20 * MS
    if completo:
        etapas["conversion"] = 10_000
        etapas["sintesis"] = int((sintesis_ms if sintesis_ms is not None else 0.1) * MS)
        L_resp = etapas["extraccion"] + etapas["preprocesamiento"] + etapas["inferencia"] \
            + etapas["conversion"] + etapas["sintesis"]
    t0 = 1_000 * MS
    c = [t0 + i * 33 * MS for i in range(N)]
    r = {
        "tipo": "ensayo", "sesion": sesion, "id_ensayo": id_ensayo, "modo": modo, "modelo": "models/m.keras",
        "tts_modo": "offline",
        "tts_sintesis_antes_de_t5": tts if completo else None,
        "parametros": {"umbral": umbral, "umbral_origen": origen, "ventana_seg": 2.5},
        "t0": t0, "c": c, "e": [ci + 12 * MS for ci in c],
        "t1": (t0 + W) if N else None, "t3": (t0 + W + 32 * MS) if infiere else None,
        "N": N, "cierre": "frames", "etapas_ns": etapas,
        "W_ns": W, "L_total_ns": (W + L_resp) if completo else None, "L_resp_ns": L_resp,
        "tasa_efectiva_fps": 30.3 if N >= 2 else None, "estado": estado,
    }
    if completo and formato == NUEVO:
        r["tts_ocupado_en_t5"] = ocupado if tts is not None else None
    elif formato == NUEVO:
        r["tts_ocupado_en_t5"] = None
    if formato == NUEVO:
        r["tts_errores_reproduccion"] = errores
        r["tts_ultimo_error_reproduccion"] = mensaje
    if formato == ANTIGUO:
        del r["parametros"]["umbral_origen"]
    if sin_parametros:
        del r["parametros"]
    return r


def turno(sintesis_ms, antes, errores=None, mensaje=None):
    return {"tipo": "turno", "sesion": "s1", "n_glosas": 2, "conversion_ns": 5_000,
            "sintesis_ns": int(sintesis_ms * MS), "tts_modo": "offline", "tts_sintesis_antes_de_orden": antes,
            "tts_errores_reproduccion": errores, "tts_ultimo_error_reproduccion": mensaje}


def escribir_jsonl(carpeta, nombre, registros):
    ruta = Path(carpeta) / nombre
    with open(ruta, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return ruta


class _ConCarpeta(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="tesis_latencia_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def resumir_archivo(self, registros, nombre="latencia_prueba.jsonl"):
        escribir_jsonl(self.tmp, nombre, registros)
        return lr.resumir(lr.leer_registros([self.tmp]))


class TestSeparacionPorSintesis(_ConCarpeta):
    """(a)"""

    def setUp(self):
        super().setUp()
        self.registros = [
            ensayo(sintesis_ms=450.0, tts=True, ocupado=False),
            ensayo(sintesis_ms=460.0, tts=True, ocupado=True),
            ensayo(sintesis_ms=1.0, tts=False, formato=ANTIGUO),        # anterior a N-34
            ensayo(sintesis_ms=3.0, tts=False, ocupado=False),          # síntesis fallida tras N-34
            ensayo(sintesis_ms=0.2, tts=None),                          # sin audio
            ensayo(estado="rechazada_confianza"),                       # no llega a t5
        ]
        self.r = self.resumir_archivo(self.registros)

    def test_grupos_con_sus_propias_estadisticas(self):
        ps = self.r["por_sintesis_antes_de_t5"]
        inc, no = ps["sintesis_incluida"], ps["sintesis_no_incluida"]
        self.assertEqual(inc["n_ensayos_hasta_t5"], 2)
        self.assertEqual(inc["sintesis"]["n"], 2)
        self.assertAlmostEqual(inc["sintesis"]["media_ms"], 455.0, places=6)
        self.assertEqual(no["n_ensayos_hasta_t5"], 3)
        self.assertEqual((no["tts_sintesis_antes_de_t5_false"], no["sin_dato_o_sin_audio"]), (2, 1))
        self.assertAlmostEqual(no["sintesis"]["media_ms"], (1.0 + 3.0 + 0.2) / 3, places=6)
        for g in lr.GRUPOS_SINTESIS:
            for m in ("L_total", "L_resp"):
                self.assertEqual(ps[g][m]["n"], ps[g]["n_ensayos_hasta_t5"], (g, m))
        # L_resp = extracción 12 + preproc. 0,5 + inferencia 20 + conversión 0,01 + síntesis
        self.assertAlmostEqual(inc["L_resp"]["media_ms"], 32.51 + 455.0, places=4)
        self.assertAlmostEqual(inc["L_total"]["media_ms"], 2500.0 + 32.51 + 455.0, places=4)

    def test_compatibilidad_magnitudes_ms_y_advertencia(self):
        m = self.r["magnitudes_ms"]
        self.assertEqual(set(m), set(lr.MAGNITUDES))
        self.assertEqual(m["sintesis"]["n"], 5)          # cálculo anterior: mezcla, se conserva
        self.assertEqual(m["L_total"]["n"], 5)
        self.assertEqual(self.r["sintesis_antes_de_t5"], {"con_audio": 4, "si": 2, "no": 2})
        self.assertTrue(any("mezclan 2 ensayos con la síntesis antes de t5 y 3 sin ella" in a
                            for a in self.r["advertencias"]), self.r["advertencias"])

    def test_tabla_muestra_filas_separadas_y_no_la_mezcla(self):
        texto = lr.tabla(self.r)
        filas = {l[:lr._ANCHO].strip(): l for l in texto.splitlines()}
        for nombre in ("sintesis", "L_total", "L_resp"):
            self.assertIn(f"{nombre} [sint. incluida]", filas)
            self.assertIn(f"{nombre} [sint. no incl.]", filas)
            self.assertNotIn(nombre, filas, f"la fila mezclada de {nombre} no debe reportarse")
        self.assertEqual(filas["sintesis [sint. incluida]"].split()[3:5], ["2", "455.0"])

    def test_solo_un_grupo_no_advierte_mezcla(self):
        r = lr.resumir([ensayo(sintesis_ms=450.0, tts=True, ocupado=False)])
        self.assertFalse(any("mezclan" in a for a in r["advertencias"]))
        self.assertIsNone(r["por_sintesis_antes_de_t5"]["sintesis_no_incluida"]["sintesis"])
        self.assertIn("sintesis [sint. no incl.]", lr.tabla(r))     # fila vacía, con n = 0

    def test_turnos_separados(self):
        r = lr.resumir([turno(450.0, True), turno(1.0, False), turno(2.0, None)])
        t = r["turnos"]
        self.assertEqual(t["sintesis"]["n"], 3)
        self.assertEqual(t["sintesis_incluida"]["n"], 1)
        self.assertAlmostEqual(t["sintesis_incluida"]["media_ms"], 450.0)
        self.assertEqual(t["sintesis_no_incluida"]["n"], 2)
        self.assertIn("sintesis [sint. incluida]", lr.tabla(r))


class TestVentanasInferidas(_ConCarpeta):
    """(b)"""

    def setUp(self):
        super().setUp()
        self.r = self.resumir_archivo([
            ensayo(W_ms=2500.0, sintesis_ms=450.0, tts=True, ocupado=False),
            ensayo(estado="rechazada_confianza", W_ms=2400.0),
            ensayo(estado="aceptada", modo="oracion", W_ms=2450.0),     # infiere, no llega a t5
            ensayo(estado="rechazada_manos", W_ms=1000.0),
            ensayo(estado="rechazada_incompleta", W_ms=300.0, N=9),     # abortada
            ensayo(estado="rechazada_incompleta", N=0),                 # sin fotogramas: sin W
        ])

    def test_W_y_captura_solo_sobre_inferidas(self):
        vi = self.r["ventanas_inferidas"]
        self.assertEqual(vi["n"], 3)
        for m in ("W", "captura"):
            self.assertEqual(vi[m]["n"], 3)
            self.assertAlmostEqual(vi[m]["media_ms"], (2500.0 + 2400.0 + 2450.0) / 3, places=6)
            self.assertAlmostEqual(vi[m]["min_ms"], 2400.0, places=6)
        self.assertEqual(self.r["magnitudes_ms"]["inferencia"]["n"], vi["n"])

    def test_compatibilidad_y_advertencia(self):
        m = self.r["magnitudes_ms"]
        self.assertEqual(m["W"]["n"], 5)                # cálculo anterior: toda ventana con fotogramas
        self.assertAlmostEqual(m["W"]["min_ms"], 300.0, places=6)
        self.assertIn("magnitudes_ms.captura/W incluyen 2 ventanas que no llegaron a inferirse: use ventanas_inferidas",
                      self.r["advertencias"])

    def test_tabla(self):
        texto = lr.tabla(self.r)
        filas = {l[:lr._ANCHO].strip(): l.split() for l in texto.splitlines()}
        self.assertEqual(filas["W [inferidas]"][2:4], ["3", "2450.0"])
        self.assertEqual(filas["captura [inferidas]"][2:4], ["3", "2450.0"])
        self.assertEqual(filas["W [todas]"][2], "5")
        self.assertNotIn("W", filas)
        self.assertNotIn("captura", filas)

    def test_inferida_por_t3_si_falta_la_etapa(self):
        r = ensayo(estado="rechazada_confianza")
        r["etapas_ns"]["inferencia"] = None
        self.assertTrue(lr.llego_a_inferirse(r))
        r["t3"] = None
        self.assertFalse(lr.llego_a_inferirse(r))
        del r["etapas_ns"]
        self.assertFalse(lr.llego_a_inferirse(r))


class TestOcupadoYUmbrales(_ConCarpeta):
    """(c)"""

    def test_tts_ocupado_en_t5(self):
        r = self.resumir_archivo([
            ensayo(sintesis_ms=450.0, tts=True, ocupado=True),
            ensayo(sintesis_ms=450.0, tts=True, ocupado=False),
            ensayo(sintesis_ms=450.0, tts=True, ocupado=False),
            ensayo(sintesis_ms=1.0, tts=False, formato=ANTIGUO),        # campo ausente
            ensayo(sintesis_ms=0.1, tts=None),                          # sin audio: no cuenta
            ensayo(estado="rechazada_confianza"),                       # sin audio: no cuenta
        ])
        self.assertEqual(r["tts_ocupado_en_t5"], {"con_audio": 4, "si": 1, "no": 2, "sin_dato": 1})
        self.assertIn("TTS ocupado al emitir t5 (otra locucion sonando; ensayos con audio 4): si 1 | no 2 | sin dato 1",
                      lr.tabla(r))

    def test_umbrales_con_sus_estados(self):
        canon = "models/canonico_holistic_mascara_manos/modelo_canonico.umbral.json"
        r = self.resumir_archivo([
            ensayo(umbral=0.5, formato=ANTIGUO, sintesis_ms=1.0, tts=False),
            ensayo(umbral=0.5, formato=ANTIGUO, estado="rechazada_confianza"),
            ensayo(umbral=0.5, formato=ANTIGUO, estado="rechazada_confianza"),
            ensayo(umbral=0.2, origen=canon, sintesis_ms=450.0, tts=True, ocupado=False),
            ensayo(umbral=0.2, origen=canon, estado="rechazada_manos"),
            ensayo(umbral=0.35, origen="parametro", estado="rechazada_confianza"),
            ensayo(sin_parametros=True, estado="rechazada_confianza"),
        ])
        u = r["umbrales"]
        self.assertEqual([(x["umbral"], x["umbral_origen"], x["n_ensayos"]) for x in u],
                         [(0.2, canon, 2), (0.35, "parametro", 1), (0.5, None, 3), (None, None, 1)])
        self.assertEqual(u[0]["contadores"], {"aceptada": 1, "rechazada_confianza": 0,
                                              "rechazada_manos": 1, "rechazada_incompleta": 0})
        self.assertAlmostEqual(u[0]["tasa_rechazo"], 0.5)
        self.assertAlmostEqual(u[2]["tasa_rechazo"], 2 / 3)
        self.assertEqual(sum(x["n_ensayos"] for x in u), r["n_ensayos"])
        self.assertIn("tasa_rechazo mezcla 4 umbrales de confianza: use umbrales", r["advertencias"])
        texto = lr.tabla(r)
        self.assertIn("MEZCLA umbrales", texto)
        self.assertIn(f"umbral 0.20 (origen: {canon}): 2 ensayos, aceptada 1, rechazo 50.0%", texto)
        self.assertIn("umbral 0.50 (origen: sin dato): 3 ensayos, aceptada 1, rechazo 66.7%", texto)
        self.assertIn("umbral sin dato (origen: sin dato): 1 ensayos", texto)

    def test_un_solo_umbral_no_advierte(self):
        r = lr.resumir([ensayo(umbral=0.2), ensayo(umbral=0.2, estado="rechazada_confianza")])
        self.assertEqual(len(r["umbrales"]), 1)
        self.assertFalse(any("umbrales" in a for a in r["advertencias"]))
        self.assertNotIn("MEZCLA", lr.tabla(r))

    def test_umbral_no_numerico_cuenta_como_sin_dato(self):
        r = ensayo()
        r["parametros"]["umbral"] = True
        self.assertEqual(lr._umbral(r), (None, "models/m.umbral.json"))
        r["parametros"]["umbral"] = 1
        self.assertEqual(lr._umbral(r), (1.0, "models/m.umbral.json"))


AFPLAY = "CalledProcessError: Command '['afplay', 'x.aiff']' returned non-zero exit status 1."
SIN_ARCHIVO = "FileNotFoundError: el audio sintetizado ya no existe"


class TestErroresDeVoz(_ConCarpeta):
    """(d) tts_errores_reproduccion (contador ACUMULADO en la sesión, leído tras t5) y
    tts_ultimo_error_reproduccion, por sesión y en total."""

    def sesion_con_error_a_mitad(self, sesion="s1"):
        """5 ensayos: la locución del 2.º no suena (el contador sube en el registro del 3.º, porque el
        audio va en segundo plano) y otra falla al final."""
        return [
            ensayo(id_ensayo=1, sesion=sesion, errores=0),
            ensayo(id_ensayo=2, sesion=sesion, errores=0),
            ensayo(id_ensayo=3, sesion=sesion, errores=1, mensaje=AFPLAY),
            ensayo(id_ensayo=4, sesion=sesion, errores=1, mensaje=AFPLAY, estado="rechazada_confianza"),
            ensayo(id_ensayo=5, sesion=sesion, errores=2, mensaje=SIN_ARCHIVO),
        ]

    def test_error_a_mitad_de_la_sesion(self):
        r = self.resumir_archivo(self.sesion_con_error_a_mitad())
        t = r["errores_voz"]["total"]
        self.assertEqual((t["n_ensayos"], t["informados"], t["sin_dato"]), (5, 5, 0))
        self.assertEqual(t["ensayos_con_errores"], 3)          # los registros 3, 4 y 5 llegan con k > 0
        self.assertEqual(t["maximo"], 2)
        self.assertEqual(t["ensayos_desde_el_primer_error"], 3)   # del 3.º al 5.º, inclusive
        self.assertEqual(t["audio_dudoso"], 4)                    # y el 2.º, que es el que no sonó (D-13)
        self.assertEqual(t["ultimo_mensaje"], SIN_ARCHIVO)
        self.assertEqual((t["n_sesiones"], t["sesiones_con_errores"]), (1, 1))
        b = r["errores_voz"]["por_sesion"][0]
        self.assertEqual(b["sesion"], "s1")
        self.assertEqual(b["primer_error_en_ensayo"], 3)
        self.assertEqual({k: b[k] for k in t if k in b}, {k: t[k] for k in t if k in b})
        self.assertIn("4 ensayos tienen audio dudoso desde el primer error de reproduccion de voz "
                      "(1 sesion/es): su audio es dudoso, vea errores_voz", r["advertencias"])
        json.dumps(r)

    def test_tabla_con_errores(self):
        texto = lr.tabla(self.resumir_archivo(self.sesion_con_error_a_mitad()))
        self.assertIn("Voz, errores de reproduccion (contador acumulado por sesion): 3 de 5 ensayos llegan "
                      "con algun error, 0 sin dato; maximo 2; 4 ensayos con audio dudoso "
                      "(D-13: desde el anterior al primer error)",
                      texto)
        self.assertIn("  sesion s1: 3 de 5 ensayos con error, maximo 2, primer error en el ensayo 3, "
                      "3 ensayos desde ese", texto)
        self.assertIn(f"  ultimo mensaje: {SIN_ARCHIVO}", texto)

    def test_por_sesion_el_contador_se_reinicia(self):
        """Dos sesiones en el mismo archivo: los tramos se cuentan por separado y el total los suma;
        «maximo» es el mayor, no la suma."""
        registros = (self.sesion_con_error_a_mitad("sA")
                     + [ensayo(id_ensayo=1, sesion="sB", errores=0),
                        ensayo(id_ensayo=2, sesion="sB", errores=1, mensaje=AFPLAY)])
        r = self.resumir_archivo(registros)
        t, sesiones = r["errores_voz"]["total"], r["errores_voz"]["por_sesion"]
        self.assertEqual([b["sesion"] for b in sesiones], ["sA", "sB"])
        self.assertEqual([b["n_ensayos"] for b in sesiones], [5, 2])
        self.assertEqual([b["maximo"] for b in sesiones], [2, 1])
        self.assertEqual([b["primer_error_en_ensayo"] for b in sesiones], [3, 2])
        self.assertEqual([b["ensayos_desde_el_primer_error"] for b in sesiones], [3, 1])
        self.assertEqual(t["n_ensayos"], 7)
        self.assertEqual(t["ensayos_con_errores"], 4)
        self.assertEqual(t["maximo"], 2)
        self.assertEqual(t["ensayos_desde_el_primer_error"], 4)
        self.assertEqual((t["n_sesiones"], t["sesiones_con_errores"]), (2, 2))
        self.assertEqual(t["ultimo_mensaje"], AFPLAY)
        texto = lr.tabla(r)
        self.assertIn("sesion sA: 3 de 5", texto)
        self.assertIn("sesion sB: 1 de 2", texto)

    def test_el_maximo_total_es_el_mayor_de_las_sesiones(self):
        """Ni el de la primera sesión ni la suma: el contador es de cada sesión."""
        r = self.resumir_archivo([ensayo(sesion="sA", id_ensayo=1, errores=1, mensaje=AFPLAY),
                                  ensayo(sesion="sB", id_ensayo=1, errores=3, mensaje=SIN_ARCHIVO)])
        t = r["errores_voz"]["total"]
        self.assertEqual([b["maximo"] for b in r["errores_voz"]["por_sesion"]], [1, 3])
        self.assertEqual(t["maximo"], 3)
        self.assertEqual((t["ensayos_con_errores"], t["sesiones_con_errores"]), (2, 2))
        self.assertIn("maximo 3;", lr.tabla(r))

    def test_sesion_sin_ningun_error(self):
        r = self.resumir_archivo([ensayo(id_ensayo=i, errores=0) for i in (1, 2, 3)])
        t = r["errores_voz"]["total"]
        self.assertEqual((t["informados"], t["sin_dato"], t["ensayos_con_errores"]), (3, 0, 0))
        self.assertEqual((t["maximo"], t["ultimo_mensaje"]), (0, None))
        self.assertEqual(t["ensayos_desde_el_primer_error"], 0)
        self.assertIsNone(r["errores_voz"]["por_sesion"][0]["primer_error_en_ensayo"])
        self.assertFalse([a for a in r["advertencias"] if "voz" in a])
        texto = lr.tabla(r)
        self.assertIn("0 de 3 ensayos llegan con algun error, 0 sin dato; maximo 0", texto)
        self.assertNotIn("ultimo mensaje", texto)
        self.assertNotIn("sesion s1:", texto)      # solo se listan las sesiones con errores

    def test_registros_sin_los_campos_nuevos(self):
        """Formato del 14-sep-2026: sin los dos campos. Ausente = sin dato, que NO es cero errores."""
        registros = [ensayo(id_ensayo=i, formato=ANTIGUO) for i in (1, 2)]
        for r in registros:
            self.assertNotIn("tts_errores_reproduccion", r)
        r = self.resumir_archivo(registros)
        t = r["errores_voz"]["total"]
        self.assertEqual((t["n_ensayos"], t["informados"], t["sin_dato"]), (2, 0, 2))
        self.assertEqual((t["ensayos_con_errores"], t["maximo"], t["ultimo_mensaje"]), (0, None, None))
        self.assertEqual(t["ensayos_desde_el_primer_error"], 0)
        self.assertFalse([a for a in r["advertencias"] if "voz" in a])
        self.assertIn("0 de 2 ensayos llegan con algun error, 2 sin dato; maximo sin dato", lr.tabla(r))
        json.dumps(r)

    def test_mezcla_de_registros_con_y_sin_los_campos(self):
        """Una sesión anterior al cambio y otra posterior en el mismo resumen."""
        r = self.resumir_archivo([ensayo(id_ensayo=1, sesion="vieja", formato=ANTIGUO),
                                  ensayo(id_ensayo=2, sesion="vieja", formato=ANTIGUO),
                                  ensayo(id_ensayo=1, sesion="nueva", errores=0),
                                  ensayo(id_ensayo=2, sesion="nueva", errores=1, mensaje=AFPLAY)])
        vieja, nueva = r["errores_voz"]["por_sesion"]
        self.assertEqual((vieja["informados"], vieja["sin_dato"], vieja["maximo"]), (0, 2, None))
        self.assertEqual((nueva["informados"], nueva["sin_dato"], nueva["maximo"]), (2, 0, 1))
        t = r["errores_voz"]["total"]
        self.assertEqual((t["informados"], t["sin_dato"], t["ensayos_con_errores"]), (2, 2, 1))
        self.assertEqual(t["ensayos_desde_el_primer_error"], 1)

    def test_null_a_mitad_no_mueve_nada_y_los_no_enteros_son_sin_dato(self):
        registros = [ensayo(id_ensayo=1, errores=1, mensaje=AFPLAY),
                     ensayo(id_ensayo=2, errores=None),            # el motor deja de informar
                     ensayo(id_ensayo=3, errores=1, mensaje=AFPLAY)]
        registros[1]["tts_ultimo_error_reproduccion"] = None
        t = self.resumir_archivo(registros)["errores_voz"]["total"]
        self.assertEqual((t["informados"], t["sin_dato"], t["ensayos_con_errores"]), (2, 1, 2))
        self.assertEqual(t["ensayos_desde_el_primer_error"], 3)    # el primero ya trae error
        self.assertEqual(t["ultimo_mensaje"], AFPLAY)
        # Valores que no son enteros (bool, texto, float) no se cuentan como contador.
        for valor in (True, "2", 2.5, [2], {"n": 2}):
            r = lr.resumir([ensayo(errores=valor)])
            self.assertEqual(r["errores_voz"]["total"]["informados"], 0, valor)
            self.assertIsNone(r["errores_voz"]["total"]["maximo"], valor)
            json.dumps(r)

    def test_los_turnos_no_se_cuentan_como_ensayos(self):
        r = self.resumir_archivo([ensayo(id_ensayo=1, errores=0),
                                  turno(1.0, False, errores=1, mensaje=AFPLAY)])
        t = r["errores_voz"]["total"]
        self.assertEqual((t["n_ensayos"], t["ensayos_con_errores"]), (1, 0))
        self.assertEqual(t["ultimo_mensaje"], None)
        self.assertEqual(r["n_turnos"], 1)

    def test_sin_ensayos_y_resumen_anterior_al_cambio(self):
        r = lr.resumir([])
        self.assertEqual(r["errores_voz"]["por_sesion"], [])
        self.assertEqual(r["errores_voz"]["total"]["n_ensayos"], 0)
        self.assertEqual(r["advertencias"], [])
        self.assertEqual(lr._lineas_errores_voz(None), [])
        # Un JSON guardado antes de este cambio no tiene la clave: la tabla lo acepta.
        r2 = self.resumir_archivo([ensayo(errores=0)])
        del r2["errores_voz"]
        self.assertNotIn("Voz, errores", lr.tabla(r2))

    def test_mensaje_muy_largo_se_recorta_solo_en_la_tabla(self):
        largo = "RuntimeError: " + "x" * 400
        r = self.resumir_archivo([ensayo(errores=1, mensaje=largo)])
        self.assertEqual(r["errores_voz"]["total"]["ultimo_mensaje"], largo)
        linea = [l for l in lr.tabla(r).splitlines() if "ultimo mensaje" in l][0]
        self.assertEqual(len(linea.split("ultimo mensaje: ")[1]), 120)
        self.assertTrue(linea.endswith("…"))


class TestAudioDudosoD13(_ConCarpeta):
    """D-13: el ensayo cuya locucion fallo suele ser el ANTERIOR al registro donde sube el contador,
    porque el audio suena en segundo plano. audio_dudoso lo incluye; el conteo estrecho se conserva."""

    def _sesion(self, errores):
        return [ensayo(tts="offline", errores=e, sesion="s1", id_ensayo=i + 1)
                for i, e in enumerate(errores)]

    def test_incluye_el_ensayo_anterior_al_primer_error(self):
        voz = self.resumir_archivo(self._sesion([0, 0, 0, 1, 1]))["errores_voz"]
        r = voz["total"]
        self.assertEqual(voz["por_sesion"][0]["primer_error_en_ensayo"], 4)
        self.assertEqual(r["ensayos_desde_el_primer_error"], 2)   # el 4 y el 5
        self.assertEqual(r["audio_dudoso"], 3)                    # y ademas el 3, que es el que fallo

    def test_si_el_error_aparece_en_el_primer_ensayo_no_hay_anterior(self):
        r = self.resumir_archivo(self._sesion([1, 1, 1]))["errores_voz"]["total"]
        self.assertEqual((r["ensayos_desde_el_primer_error"], r["audio_dudoso"]), (3, 3))

    def test_sin_errores_no_hay_audio_dudoso(self):
        r = self.resumir_archivo(self._sesion([0, 0, 0]))["errores_voz"]["total"]
        self.assertEqual((r["ensayos_desde_el_primer_error"], r["audio_dudoso"]), (0, 0))

    def test_se_suma_por_sesion_y_sale_en_la_tabla_y_en_el_aviso(self):
        registros = (self._sesion([0, 0, 1])
                     + [ensayo(tts="offline", errores=e, sesion="s2", id_ensayo=i + 1)
                        for i, e in enumerate([0, 1])])
        r = self.resumir_archivo(registros)
        self.assertEqual(r["errores_voz"]["total"]["audio_dudoso"], 2 + 2)
        self.assertIn("audio dudoso", lr.tabla(r))
        self.assertTrue([a for a in r["advertencias"] if "audio dudoso" in a], r["advertencias"])


class TestSeparacionPorCacheDeAudio(_ConCarpeta):
    """(e) Desde N-45 la sintesis puede venir de la cache (~0,1 ms) o hacerse en el momento (~15 ms).
    Promediarlas juntas no significa nada, asi que el resumen las separa y avisa."""

    def _con_cache(self, desde_cache, sintesis_ms, **kw):
        r = ensayo(sintesis_ms=sintesis_ms, tts="offline", **kw)
        if desde_cache is not None:
            r["tts_sintesis_desde_cache"] = desde_cache
        return r

    def test_separa_los_tres_grupos_y_avisa_al_mezclar(self):
        registros = ([self._con_cache(True, 0.1) for _ in range(3)]
                     + [self._con_cache(False, 15.0) for _ in range(2)]
                     + [self._con_cache(None, 460.0)])
        r = self.resumir_archivo(registros)
        pc = r["por_sintesis_desde_cache"]
        self.assertEqual(pc["desde_cache"]["n_ensayos_hasta_t5"], 3)
        self.assertEqual(pc["sintetizada"]["n_ensayos_hasta_t5"], 2)
        self.assertEqual(pc["sin_dato"]["n_ensayos_hasta_t5"], 1)
        self.assertAlmostEqual(pc["desde_cache"]["sintesis"]["mediana_ms"], 0.1, places=3)
        self.assertAlmostEqual(pc["sintetizada"]["sintesis"]["mediana_ms"], 15.0, places=3)
        self.assertAlmostEqual(pc["sin_dato"]["sintesis"]["mediana_ms"], 460.0, places=3)
        self.assertTrue(any("por_sintesis_desde_cache" in a for a in r["advertencias"]),
                        r["advertencias"])

    def test_sin_el_campo_no_hay_aviso_y_todo_queda_en_sin_dato(self):
        r = self.resumir_archivo([ensayo(sintesis_ms=460.0, tts="offline") for _ in range(3)])
        pc = r["por_sintesis_desde_cache"]
        self.assertEqual((pc["sin_dato"]["n_ensayos_hasta_t5"], pc["desde_cache"]["n_ensayos_hasta_t5"],
                          pc["sintetizada"]["n_ensayos_hasta_t5"]), (3, 0, 0))
        self.assertFalse([a for a in r["advertencias"] if "cache" in a])

    def test_la_tabla_muestra_solo_los_grupos_con_datos(self):
        texto = lr.tabla(self.resumir_archivo([self._con_cache(True, 0.1), self._con_cache(True, 0.12)]))
        self.assertIn("sintesis [desde cache]", texto)
        self.assertNotIn("[sintetizada]", texto)


class TestCompatibilidad(_ConCarpeta):
    """Registros sin los campos nuevos, casos vacíos, CLI y los dos registros reales (solo lectura)."""

    def test_registros_antiguos_sin_campos_nuevos(self):
        antiguos = [ensayo(formato=ANTIGUO, sintesis_ms=1.0, tts=False),
                    ensayo(formato=ANTIGUO, estado="rechazada_manos"),
                    ensayo(formato=ANTIGUO, modo="oracion"),
                    turno(0.1, False)]
        for r in antiguos:
            self.assertNotIn("tts_ocupado_en_t5", r)
        r = self.resumir_archivo(antiguos)
        self.assertEqual(r["sintesis_antes_de_t5"], {"con_audio": 1, "si": 0, "no": 1})
        self.assertEqual(r["tts_ocupado_en_t5"], {"con_audio": 1, "si": 0, "no": 0, "sin_dato": 1})
        self.assertEqual([(u["umbral"], u["umbral_origen"]) for u in r["umbrales"]], [(0.5, None)])
        self.assertEqual(r["por_sintesis_antes_de_t5"]["sintesis_incluida"]["n_ensayos_hasta_t5"], 0)
        texto = lr.tabla(r)
        for etiqueta in lr.ETAPAS + ("W", "L_total", "L_resp", "extractor/fotog.", "tasa efectiva", "Turnos"):
            self.assertIn(etiqueta, texto)
        json.dumps(r)

    def test_resumen_guardado_antes_de_este_cambio_se_sigue_tabulando(self):
        # Un JSON escrito con la versión anterior no tiene las claves nuevas: tabla() lo acepta.
        r = self.resumir_archivo([ensayo(sintesis_ms=1.0, tts=False), turno(0.1, False)])
        for clave in ("ventanas_inferidas", "por_sintesis_antes_de_t5", "tts_ocupado_en_t5", "umbrales",
                      "advertencias"):
            del r[clave]
        for clave in ("sintesis_incluida", "sintesis_no_incluida"):
            del r["turnos"][clave]
        texto = lr.tabla(r)
        filas = {l[:lr._ANCHO].strip() for l in texto.splitlines()}
        self.assertTrue({"captura", "sintesis", "W", "L_total", "L_resp"} <= filas, filas)

    def test_sin_ensayos(self):
        for registros in ([], [turno(0.1, False)]):
            r = lr.resumir(registros)
            self.assertEqual(r["n_ensayos"], 0)
            self.assertEqual(r["ventanas_inferidas"]["n"], 0)
            self.assertEqual(r["umbrales"], [])
            self.assertEqual(r["advertencias"], [])
            lr.tabla(r)
            json.dumps(r)

    def test_cli_tabla_y_json(self):
        escribir_jsonl(self.tmp, "latencia_a.jsonl", [ensayo(sintesis_ms=450.0, tts=True, ocupado=False)])
        escribir_jsonl(self.tmp, "latencia_b.jsonl", [ensayo(formato=ANTIGUO, sintesis_ms=1.0, tts=False)])
        destino = os.path.join(self.tmp, "salida", "resumen.json")
        with mock.patch.object(sys, "argv", ["latencia_resumen.py", self.tmp, "--json", destino]), \
                contextlib.redirect_stdout(io.StringIO()) as salida:
            lr.main()
        self.assertIn("L_resp [sint. incluida]", salida.getvalue())
        with open(destino, encoding="utf-8") as f:
            guardado = json.load(f)
        self.assertEqual(guardado["n_ensayos"], 2)
        with mock.patch.object(sys, "argv", ["latencia_resumen.py", self.tmp, "--json", "-"]), \
                contextlib.redirect_stdout(io.StringIO()) as salida:
            lr.main()
        self.assertEqual(json.loads(salida.getvalue())["por_sintesis_antes_de_t5"]["sintesis_incluida"]
                         ["n_ensayos_hasta_t5"], 1)

    def test_registros_reales_se_siguen_resumiendo(self):
        """Solo lectura. La carpeta es registros/ o la que indique LSC_REGISTROS_DIR: mientras haya una
        sesión del prototipo en curso hay que apuntar a una copia, porque el prototipo escribe ahí."""
        carpeta = Path(os.environ.get("LSC_REGISTROS_DIR") or (ROOT_DIR / "registros"))
        rutas = [carpeta / f"latencia_2026-09-{d}.jsonl" for d in ("14", "15")]
        faltan = [str(p) for p in rutas if not p.is_file()]
        if faltan:
            self.skipTest(f"no están los registros reales: {faltan}")
        crudos = []
        for p in rutas:
            with open(p, encoding="utf-8") as f:
                crudos.extend(json.loads(l) for l in f if l.strip())
        ensayos = [x for x in crudos if x.get("tipo") == "ensayo"]
        registros = lr.leer_registros([str(p) for p in rutas])
        r = lr.resumir(registros)
        texto = lr.tabla(r)
        json.dumps(r)
        self.assertEqual(r["n_ensayos"], len(ensayos))
        # (b) las inferidas son exactamente las que tienen inferencia; las rechazadas por manos quedan fuera.
        inferidas = [x for x in ensayos if x["etapas_ns"]["inferencia"] is not None]
        self.assertEqual(r["ventanas_inferidas"]["n"], len(inferidas))
        self.assertEqual(r["ventanas_inferidas"]["W"]["n"], len(inferidas))
        self.assertFalse(any(x["estado"] == "rechazada_manos" for x in inferidas))
        # (a) cada ensayo completo cae en un grupo y solo en uno.
        completos = [x for x in ensayos if x.get("L_total_ns") is not None]
        ps = r["por_sintesis_antes_de_t5"]
        self.assertEqual(sum(ps[g]["n_ensayos_hasta_t5"] for g in lr.GRUPOS_SINTESIS), len(completos))
        self.assertEqual(ps["sintesis_incluida"]["n_ensayos_hasta_t5"],
                         sum(1 for x in completos if x.get("tts_sintesis_antes_de_t5") is True))
        # (c) todos los ensayos quedan contados en algún umbral.
        self.assertEqual(sum(u["n_ensayos"] for u in r["umbrales"]), len(ensayos))
        self.assertIn("W [inferidas]", texto)
        # (d) son anteriores a N-37: ningún ensayo informa el contador de errores de voz.
        voz = r["errores_voz"]
        self.assertEqual(sum(b["n_ensayos"] for b in voz["por_sesion"]), len(ensayos))
        self.assertEqual([b["sesion"] for b in voz["por_sesion"]],
                         sorted({x["sesion"] for x in ensayos}, key=[x["sesion"] for x in ensayos].index))
        informados = [x for x in ensayos if isinstance(x.get("tts_errores_reproduccion"), int)]
        self.assertEqual(voz["total"]["informados"], len(informados))
        self.assertEqual(voz["total"]["sin_dato"], len(ensayos) - len(informados))
        self.assertIn("Voz, errores de reproduccion", texto)


if __name__ == "__main__":
    unittest.main()
