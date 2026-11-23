"""
Pruebas de src/exactitud_bloque1.py con JSONL y hojas CSV sintéticos.

No abren cámara ni audio: todo se construye en un directorio temporal y nunca se toca registros/ (el
prototipo escribe ahí y el usuario puede estar grabando).

Cubren lo que pide el frente:
  * el cruce correcto por (sesion, id_ensayo);
  * el ensayo del JSONL que no tiene fila en la hoja y la fila de la hoja que no tiene ensayo, ambos
    reportados y no descartados en silencio;
  * las ejecuciones marcadas como no válidas (y las que se quedan sin el campo «valida»);
  * top-1 y top-3 calculados a mano, con los tres denominadores abiertos de PD-15 y las tres políticas
    de repetición de PD-14;
  * la tasa de rechazo por confianza, por manos y por secuencia incompleta;
  * el caso de audio dudoso (tts_errores_reproduccion, N-37/N-38) con sus dos convenciones;
  * la matriz de confusión, el desglose por seña y por repetición, la plantilla CSV y la salida
    (JSON serializable e informe .md).

Ejecutar desde la raíz del repositorio:
    python -m unittest tests.test_exactitud_bloque1 -v
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src import exactitud_bloque1 as eb  # noqa: E402
from src.analisis_errores import top_k  # noqa: E402
from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402

MS = 1_000_000


def ensayo(id_ensayo: int, estado: str = "aceptada", id_clase: int | None = None,
           top3: list[int] | None = None, sesion: str = "s1", confianza: float = 0.9,
           errores_voz=None, mensaje=None, modo: str = "aislado", umbral: float = 0.2,
           representacion: str = "manos", mano_ausente: str = "mascara"):
    """Registro «ensayo» con la forma que escribe src/translator.py, _registro_ensayo.

    Los estados que no llegan al modelo (rechazada_manos, rechazada_incompleta) no traen id_clase ni
    top3, igual que en el prototipo; rechazada_confianza y aceptada sí.
    """
    infiere = estado in ("aceptada", "rechazada_confianza")
    if infiere and id_clase is None:
        raise AssertionError("un ensayo que infiere necesita id_clase en la prueba")
    if not infiere:
        id_clase, top3 = None, None
    if infiere and top3 is None:
        top3 = [id_clase]
    t0 = 1_000 * MS
    return {
        "tipo": "ensayo",
        "sesion": sesion,
        "id_ensayo": id_ensayo,
        "fecha_hora": f"2026-09-20T10:00:{id_ensayo:02d}.000",
        "modo": modo,
        "modelo": "models/canonico_holistic_mascara_manos/modelo_canonico.keras",
        "representacion": representacion,
        "mano_ausente": mano_ausente,
        "extractor": "HolisticLandmarker mediapipe 0.10.35",
        "tts_modo": "offline",
        "tts_sintesis_antes_de_t5": True if estado == "aceptada" else None,
        "tts_ocupado_en_t5": False if estado == "aceptada" else None,
        "tts_errores_reproduccion": errores_voz,
        "tts_ultimo_error_reproduccion": mensaje,
        "parametros": {"umbral": umbral, "umbral_origen": "modelo_canonico.umbral.json",
                       "ventana_seg": 2.5, "min_frames": 15, "max_sin_manos": 0.5},
        "t0": t0,
        "N": 75,
        "cierre": "frames",
        "estado": estado,
        "motivo": None,
        "id_clase": id_clase,
        "clase": SIGN_VOCABULARY[id_clase] if id_clase is not None else None,
        "confianza": confianza if infiere else None,
        "top3": ([{"id": i, "glosa": SIGN_VOCABULARY[i], "p": 0.9 - 0.1 * k}
                  for k, i in enumerate(top3)] if top3 else None),
        "enunciado": SIGN_VOCABULARY[id_clase] if estado == "aceptada" else None,
    }


CABECERA = ("sesion,presentacion,ronda,intento,id_ensayo,id_glosa_objetivo,glosa_objetivo,"
            "referencia_mostrada,variante,valida,estado_pantalla,observacion")


def fila(id_ensayo, objetivo, variante="no", valida="si", sesion="s1", presentacion=None,
         ronda="", intento="", estado_pantalla="", observacion="", usar_nombre=True):
    """Una fila de la hoja del operador. presentacion por defecto = id_ensayo (una por presentación)."""
    presentacion = id_ensayo if presentacion is None else presentacion
    id_glosa = "" if usar_nombre else str(objetivo)
    nombre = SIGN_VOCABULARY[objetivo] if usar_nombre else ""
    return (f"{sesion},{presentacion},{ronda},{intento},{id_ensayo},{id_glosa},{nombre},,"
            f"{variante},{valida},{estado_pantalla},{observacion}")


def escribir(carpeta: Path, ensayos: list[dict], filas: list[str], nombre_jsonl="bloque1.jsonl",
             nombre_csv="hoja.csv") -> tuple[Path, Path]:
    jsonl = carpeta / nombre_jsonl
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in ensayos:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    csv_path = carpeta / nombre_csv
    csv_path.write_text("\n".join([CABECERA] + filas) + "\n", encoding="utf-8")
    return jsonl, csv_path


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def analizar(self, ensayos, filas, **kw):
        jsonl, hoja = escribir(self.tmp, ensayos, filas)
        return eb.analizar([str(jsonl)], str(hoja), **kw)


# ---------------------------------------------------------------------------
# Cruce
# ---------------------------------------------------------------------------

class TestCruce(Base):

    def test_cruce_correcto_sin_sobras(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1), ensayo(3, id_clase=2)]
        filas = [fila(1, 0), fila(2, 1), fila(3, 2)]
        a = self.analizar(ensayos, filas)
        self.assertEqual(a["cruce"]["n_emparejados"], 3)
        self.assertEqual(a["cruce"]["ensayos_sin_fila"], [])
        self.assertEqual(a["cruce"]["filas_sin_ensayo"], [])
        self.assertEqual(a["cruce"]["filas_con_problema"], [])
        self.assertEqual(a["advertencias"], [])
        self.assertEqual(a["validez"]["validas"], 3)

    def test_ensayo_del_jsonl_sin_fila_en_la_hoja(self):
        """El ensayo 3 se ejecutó pero el operador no lo anotó: se avisa y no entra en las métricas."""
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1), ensayo(3, id_clase=2)]
        filas = [fila(1, 0), fila(2, 1)]
        a = self.analizar(ensayos, filas)
        self.assertEqual(a["cruce"]["n_emparejados"], 2)
        sueltos = a["cruce"]["ensayos_sin_fila"]
        self.assertEqual([x["clave"] for x in sueltos], ["s1/3"])
        self.assertEqual(sueltos[0]["estado"], "aceptada")
        self.assertTrue(any("no tienen fila en la hoja" in w for w in a["advertencias"]))
        # No se cuela en ninguna métrica.
        self.assertEqual(a["exactitud"]["totalidad"]["politicas"]["todos"]["n_ejecuciones"], 2)

    def test_fila_de_la_hoja_sin_ensayo_en_el_jsonl(self):
        """El operador anotó un id_ensayo que no existe (riesgo R14): se avisa y no entra."""
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1)]
        filas = [fila(1, 0), fila(2, 1), fila(9, 5)]
        a = self.analizar(ensayos, filas)
        self.assertEqual(a["cruce"]["n_emparejados"], 2)
        huerfanas = a["cruce"]["filas_sin_ensayo"]
        self.assertEqual([x["clave"] for x in huerfanas], ["s1/9"])
        self.assertEqual(huerfanas[0]["glosa_objetivo"], SIGN_VOCABULARY[5])
        self.assertTrue(any("no tienen ensayo en el JSONL" in w for w in a["advertencias"]))

    def test_las_sesiones_no_se_mezclan(self):
        """El mismo id_ensayo en dos sesiones distintas son dos ejecuciones distintas."""
        ensayos = [ensayo(1, id_clase=0, sesion="s1"), ensayo(1, id_clase=7, sesion="s2")]
        filas = [fila(1, 0, sesion="s1"), fila(1, 7, sesion="s2")]
        a = self.analizar(ensayos, filas)
        self.assertEqual(a["cruce"]["n_emparejados"], 2)
        self.assertEqual(a["cruce"]["ensayos_duplicados"], [])
        self.assertEqual(a["exactitud"]["totalidad"]["politicas"]["todos"]
                         ["denominadores"]["con_prediccion"]["aciertos_top1"], 2)

    def test_fila_repetida_se_avisa_y_se_usa_la_primera(self):
        ensayos = [ensayo(1, id_clase=0)]
        filas = [fila(1, 0), fila(1, 3)]
        a = self.analizar(ensayos, filas)
        self.assertEqual(a["cruce"]["n_emparejados"], 1)
        self.assertEqual(len(a["cruce"]["filas_duplicadas"]), 1)
        self.assertEqual(a["cruce"]["filas_duplicadas"][0]["clave"], "s1/1")
        self.assertEqual(a["por_sena"][0]["id"], 0)

    def test_filtro_por_sesion(self):
        ensayos = [ensayo(1, id_clase=0, sesion="s1"), ensayo(1, id_clase=7, sesion="s2")]
        filas = [fila(1, 0, sesion="s1"), fila(1, 7, sesion="s2")]
        a = self.analizar(ensayos, filas, sesiones=["s2"])
        self.assertEqual(a["cruce"]["n_emparejados"], 1)
        self.assertEqual([x["clave"] for x in a["cruce"]["filas_sin_ensayo"]], ["s1/1"])

    def test_estado_de_la_hoja_distinto_del_jsonl(self):
        ensayos = [ensayo(1, estado="rechazada_confianza", id_clase=0, confianza=0.1)]
        filas = [fila(1, 0, estado_pantalla="aceptada")]
        a = self.analizar(ensayos, filas)
        d = a["cruce"]["discrepancias_estado"]
        self.assertEqual(len(d), 1)
        self.assertEqual((d[0]["hoja"], d[0]["jsonl"]), ("aceptada", "rechazada_confianza"))

    def test_estado_de_la_hoja_igual_no_es_discrepancia(self):
        ensayos = [ensayo(1, estado="rechazada_manos")]
        filas = [fila(1, 0, estado_pantalla="Rechazada manos")]
        a = self.analizar(ensayos, filas)
        self.assertEqual(a["cruce"]["discrepancias_estado"], [])


# ---------------------------------------------------------------------------
# Lectura de la hoja
# ---------------------------------------------------------------------------

class TestHoja(Base):

    def test_glosa_por_nombre_por_id_y_sin_tildes(self):
        ensayos = [ensayo(1, id_clase=26), ensayo(2, id_clase=26), ensayo(3, id_clase=26)]
        filas = [fila(1, 26),                                   # «¿cómo estás?» tal cual
                 fila(2, 26, usar_nombre=False),                # por id_glosa_objetivo
                 f"s1,3,,,3,,como estas,,no,si,,"]              # sin tildes ni signos
        a = self.analizar(ensayos, filas)
        self.assertEqual(a["cruce"]["filas_con_problema"], [])
        self.assertEqual([f["id"] for f in a["por_sena"]], [26])
        self.assertEqual(a["por_sena"][0]["n_ejecuciones"], 3)

    def test_glosa_desconocida_o_incoherente_se_reporta(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=0), ensayo(3, id_clase=0)]
        filas = ["s1,1,,,1,,arepa,,no,si,,",            # no está en el vocabulario
                 "s1,2,,,2,7,gracias,,no,si,,",         # id 7 (comer) y nombre «gracias» (0)
                 "s1,3,,,3,,,,no,si,,"]                 # ni id ni nombre
        a = self.analizar(ensayos, filas)
        problemas = a["cruce"]["filas_con_problema"]
        self.assertEqual(len(problemas), 3)
        self.assertIn("no está en el vocabulario", problemas[0]["motivo"])
        self.assertIn("no coinciden", problemas[1]["motivo"])
        self.assertIn("ni id_glosa_objetivo ni glosa_objetivo", problemas[2]["motivo"])
        self.assertEqual(a["cruce"]["n_emparejados"], 0)
        self.assertTrue(any("no se pudieron interpretar" in w for w in a["advertencias"]))

    def test_variante_no_reconocida_se_reporta(self):
        ensayos = [ensayo(1, id_clase=0)]
        filas = ["s1,1,,,1,0,,,quiza,si,,"]
        a = self.analizar(ensayos, filas)
        self.assertEqual(len(a["cruce"]["filas_con_problema"]), 1)
        self.assertIn("variante", a["cruce"]["filas_con_problema"][0]["motivo"])

    def test_falta_una_columna_obligatoria(self):
        (self.tmp / "mala.csv").write_text("sesion,id_ensayo,glosa_objetivo\ns1,1,gracias\n",
                                           encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            eb.leer_hoja(self.tmp / "mala.csv")
        self.assertIn("variante", str(ctx.exception))

    def test_lineas_de_comentario_y_vacias_se_saltan(self):
        texto = "\n".join([CABECERA, fila(1, 0), ",,,,,,,,,,,",
                           "# esto es la leyenda de la plantilla", ""])
        (self.tmp / "h.csv").write_text(texto + "\n", encoding="utf-8")
        filas, problemas = eb.leer_hoja(self.tmp / "h.csv")
        self.assertEqual(len(filas), 1)
        self.assertEqual(problemas, [])

    def test_la_plantilla_del_repositorio_se_lee_y_esta_vacia(self):
        """docs/plantilla_hoja_registro_bloque1.csv debe ser legible y no traer datos inventados."""
        filas, problemas = eb.leer_hoja(eb.PLANTILLA)
        self.assertEqual(filas, [])
        self.assertEqual(problemas, [])

    def test_escribir_plantilla(self):
        destino = self.tmp / "sub" / "hoja.csv"
        eb.escribir_plantilla(destino)
        self.assertTrue(destino.is_file())
        self.assertEqual(destino.read_text(encoding="utf-8"),
                         eb.PLANTILLA.read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            eb.escribir_plantilla(destino)


# ---------------------------------------------------------------------------
# Validez
# ---------------------------------------------------------------------------

class TestValidez(Base):

    def test_ejecuciones_no_validas_salen_de_las_metricas_pero_se_cuentan(self):
        """La ejecución 2 se excluyó por una incidencia técnica (E3): se lista con su motivo."""
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1), ensayo(3, id_clase=2)]
        filas = [fila(1, 0),
                 fila(2, 1, valida="no", observacion="E3 la camara se congelo"),
                 fila(3, 2)]
        a = self.analizar(ensayos, filas)
        self.assertEqual(a["validez"]["validas"], 2)
        self.assertEqual(a["validez"]["no_validas"], 1)
        detalle = a["validez"]["no_validas_detalle"]
        self.assertEqual(detalle[0]["clave"], "s1/2")
        self.assertEqual(detalle[0]["observacion"], "E3 la camara se congelo")
        bloque = a["exactitud"]["totalidad"]["politicas"]["todos"]
        self.assertEqual(bloque["n_ejecuciones"], 2)
        self.assertEqual({f["id"] for f in a["por_sena"]}, {0, 2})

    def test_valida_en_blanco_no_cuenta_como_si(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1)]
        filas = [fila(1, 0), fila(2, 1, valida="")]
        a = self.analizar(ensayos, filas)
        self.assertEqual(a["validez"]["validas"], 1)
        self.assertEqual(a["validez"]["sin_dato"], 1)
        self.assertEqual(a["validez"]["sin_dato_detalle"][0]["clave"], "s1/2")
        self.assertTrue(any("sin el campo «valida»" in w for w in a["advertencias"]))


# ---------------------------------------------------------------------------
# Exactitud: top-1 y top-3 a mano
# ---------------------------------------------------------------------------

class TestExactitud(Base):
    """Diez ejecuciones con el resultado calculado a mano.

    aceptadas (6): ids 1-6.
        1 objetivo 0 predicha 0            -> acierto top-1 y top-3
        2 objetivo 1 predicha 1            -> acierto top-1 y top-3
        3 objetivo 2 predicha 2            -> acierto top-1 y top-3
        4 objetivo 3 predicha 9, top3 [9,3,4]  -> falla top-1, acierta top-3
        5 objetivo 4 predicha 9, top3 [9,8,7]  -> falla las dos
        6 objetivo 5 predicha 5            -> acierto top-1 y top-3
    rechazadas por confianza (2): ids 7-8, sí tienen predicción.
        7 objetivo 6 predicha 6, top3 [6,1,2]  -> acierto top-1 y top-3
        8 objetivo 7 predicha 2, top3 [2,1,0]  -> falla las dos
    rechazada por manos (1): id 9, objetivo 8, sin predicción.
    rechazada incompleta (1): id 10, objetivo 9, sin predicción.

    con_prediccion      : n = 8; top-1 = 5/8 = 0,625; top-3 = 6/8 = 0,750
    aceptadas           : n = 6; top-1 = 4/6 = 0,666…
    todas_rechazo_error : n = 10; top-1 = 4/10 = 0,4; top-3 = 5/10 = 0,5
    rechazos            : confianza 2/10, manos 1/10, incompleta 1/10, total 4/10
    """

    def datos(self):
        ensayos = [
            ensayo(1, id_clase=0, top3=[0, 1, 2]),
            ensayo(2, id_clase=1, top3=[1, 0, 2]),
            ensayo(3, id_clase=2, top3=[2, 0, 1]),
            ensayo(4, id_clase=9, top3=[9, 3, 4]),
            ensayo(5, id_clase=9, top3=[9, 8, 7]),
            ensayo(6, id_clase=5, top3=[5, 4, 3]),
            ensayo(7, estado="rechazada_confianza", id_clase=6, top3=[6, 1, 2], confianza=0.1),
            ensayo(8, estado="rechazada_confianza", id_clase=2, top3=[2, 1, 0], confianza=0.1),
            ensayo(9, estado="rechazada_manos"),
            ensayo(10, estado="rechazada_incompleta"),
        ]
        filas = [fila(i, objetivo) for i, objetivo in
                 zip(range(1, 11), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
        return ensayos, filas

    def test_top1_y_top3_por_denominador(self):
        a = self.analizar(*self.datos())
        d = a["exactitud"]["totalidad"]["politicas"]["todos"]["denominadores"]

        cp = d["con_prediccion"]
        self.assertEqual((cp["n_denominador"], cp["aciertos_top1"], cp["aciertos_top3"]), (8, 5, 6))
        self.assertAlmostEqual(cp["top1"], 5 / 8)
        self.assertAlmostEqual(cp["top3"], 6 / 8)

        ac = d["aceptadas"]
        self.assertEqual((ac["n_denominador"], ac["aciertos_top1"]), (6, 4))
        self.assertAlmostEqual(ac["top1"], 4 / 6)

        td = d["todas_rechazo_error"]
        self.assertEqual((td["n_denominador"], td["n_con_salida"], td["aciertos_top1"]), (10, 6, 4))
        self.assertAlmostEqual(td["top1"], 0.4)
        self.assertAlmostEqual(td["top3"], 0.5)
        # Las señas que se quedaron sin salida del sistema quedan nombradas.
        self.assertEqual(td["sin_salida_por_sena"]["8"]["estados"], {"rechazada_manos": 1})
        self.assertEqual(td["sin_salida_por_sena"]["9"]["estados"], {"rechazada_incompleta": 1})

    def test_los_tres_denominadores_estan_nombrados(self):
        a = self.analizar(*self.datos())
        d = a["exactitud"]["totalidad"]["politicas"]["todos"]["denominadores"]
        self.assertEqual(set(d), set(eb.DENOMINADORES))
        self.assertIn("PD-15 A", d["con_prediccion"]["etiqueta"])
        self.assertIn("PD-15 B", d["todas_rechazo_error"]["etiqueta"])
        self.assertTrue(any("PD-15" in p for p in a["pendientes_de_decision"]))
        self.assertTrue(any("PD-14" in p for p in a["pendientes_de_decision"]))

    def test_top3_coincide_con_la_convencion_del_analisis_offline(self):
        """El top-3 debe ser el mismo que calcula src/analisis_errores.py, top_k (N-35)."""
        ensayos, filas = self.datos()
        a = self.analizar(ensayos, filas)
        con_pred = [r for r in ensayos if r["id_clase"] is not None]
        objetivos = dict(zip(range(1, 11), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        reales = [objetivos[r["id_ensayo"]] for r in con_pred]
        top3s = [[t["id"] for t in r["top3"]] for r in con_pred]
        self.assertAlmostEqual(a["exactitud"]["totalidad"]["politicas"]["todos"]
                               ["denominadores"]["con_prediccion"]["top3"],
                               top_k(reales, top3s, 3))

    def test_tasas_de_rechazo(self):
        a = self.analizar(*self.datos())
        r = a["exactitud"]["totalidad"]["politicas"]["todos"]["rechazos"]
        self.assertEqual(r["n_ejecuciones"], 10)
        self.assertEqual(r["contadores"], {"aceptada": 6, "rechazada_confianza": 2,
                                           "rechazada_manos": 1, "rechazada_incompleta": 1, "otro": 0})
        self.assertAlmostEqual(r["tasa_rechazo_confianza"], 0.2)
        self.assertAlmostEqual(r["tasa_rechazo_manos"], 0.1)
        self.assertAlmostEqual(r["tasa_rechazo_incompleta"], 0.1)
        self.assertAlmostEqual(r["tasa_rechazo_total"], 0.4)

    def test_matriz_de_confusion(self):
        a = self.analizar(*self.datos())
        m = a["matriz_confusion_referencia"]["matriz"]
        self.assertEqual(m["0"], {"0": 1})
        self.assertEqual(m["3"], {"9": 1})     # objetivo 3 predicha 9
        self.assertEqual(m["4"], {"9": 1})
        self.assertEqual(m["7"], {"2": 1})     # el rechazo por confianza sí entra: tiene predicción
        self.assertNotIn("8", m)               # rechazada por manos: sin predicción
        self.assertNotIn("9", m)
        pares = a["matriz_confusion_referencia"]["pares"]
        errores = [p for p in pares if not p["acierto"]]
        self.assertEqual(len(errores), 3)
        self.assertEqual({(p["objetivo"], p["predicha"]) for p in errores}, {(3, 9), (4, 9), (7, 2)})

    def test_por_sena(self):
        a = self.analizar(*self.datos())
        por_id = {f["id"]: f for f in a["por_sena"]}
        self.assertEqual(por_id[0]["aciertos_top1_con_prediccion"], 1)
        self.assertEqual(por_id[3]["aciertos_top1_con_prediccion"], 0)
        self.assertEqual(por_id[3]["aciertos_top3_con_prediccion"], 1)
        self.assertEqual(por_id[8]["n_con_prediccion"], 0)
        self.assertIsNone(por_id[8]["top1_con_prediccion"])
        self.assertEqual(por_id[8]["top1_todas_rechazo_error"], 0.0)
        self.assertEqual(por_id[8]["estados"]["rechazada_manos"], 1)

    def test_macro_sobre_las_senas_presentadas_y_sobre_las_50(self):
        a = self.analizar(*self.datos())
        cp = a["exactitud"]["totalidad"]["politicas"]["todos"]["denominadores"]["con_prediccion"]
        # Se presentaron 8 señas con predicción (objetivos 0-7).
        self.assertEqual(cp["n_clases_macro"], 8)
        self.assertEqual(len(cp["por_clase"]), 8)
        self.assertAlmostEqual(cp["recall_macro"], 5 / 8)
        # El macro sobre las 50 clases es el que usa el nivel offline: mismo numerador, más divisor.
        self.assertLess(cp["f1_macro_50clases"], cp["f1_macro"])

    def test_sin_ninguna_aceptada_la_exactitud_es_cero_no_sin_dato(self):
        ensayos = [ensayo(1, estado="rechazada_manos"), ensayo(2, estado="rechazada_incompleta")]
        filas = [fila(1, 0), fila(2, 1)]
        a = self.analizar(ensayos, filas)
        d = a["exactitud"]["totalidad"]["politicas"]["todos"]["denominadores"]
        self.assertEqual(d["todas_rechazo_error"]["n_denominador"], 2)
        self.assertEqual(d["todas_rechazo_error"]["top1"], 0.0)
        self.assertEqual(d["con_prediccion"]["n_denominador"], 0)
        self.assertIsNone(d["con_prediccion"]["top1"])


# ---------------------------------------------------------------------------
# Variantes declaradas (reporte por duplicado, MET 229-233) y PD-13
# ---------------------------------------------------------------------------

class TestVariantes(Base):

    def test_reporte_por_duplicado(self):
        """Totalidad: 4 ejecuciones, 3 aciertos. Sin variante: 3 ejecuciones, 2 aciertos."""
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1),
                   ensayo(3, id_clase=9, top3=[9, 2, 3]), ensayo(4, id_clase=3)]
        filas = [fila(1, 0, variante="no"), fila(2, 1, variante="no"),
                 fila(3, 2, variante="no"), fila(4, 3, variante="si")]
        a = self.analizar(ensayos, filas)
        total = a["exactitud"]["totalidad"]["politicas"]["todos"]["denominadores"]["con_prediccion"]
        limpio = (a["exactitud"]["sin_variante_estricto"]["politicas"]["todos"]
                  ["denominadores"]["con_prediccion"])
        self.assertEqual((total["n_denominador"], total["aciertos_top1"]), (4, 3))
        self.assertEqual((limpio["n_denominador"], limpio["aciertos_top1"]), (3, 2))
        self.assertAlmostEqual(a["variantes"]["proporcion_con_variante"], 0.25)
        self.assertEqual(a["variantes"]["contadores"], {"si": 1, "no": 3, "no_segura": 0, "sin_dato": 0})

    def test_no_estoy_segura_se_reporta_de_las_dos_maneras(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1), ensayo(3, id_clase=2)]
        filas = [fila(1, 0, variante="no"), fila(2, 1, variante="no estoy segura"),
                 fila(3, 2, variante="si")]
        a = self.analizar(ensayos, filas)
        self.assertTrue(a["variantes"]["usa_no_estoy_segura"])
        estricto = a["exactitud"]["sin_variante_estricto"]["politicas"]["todos"]
        amplio = a["exactitud"]["sin_variante_amplio"]["politicas"]["todos"]
        self.assertEqual(estricto["n_ejecuciones"], 1)
        self.assertEqual(amplio["n_ejecuciones"], 2)
        self.assertTrue(a["exactitud"]["sin_variante_amplio"]["aplica"])
        self.assertTrue(any("PD-13" in p for p in a["pendientes_de_decision"]))

    def test_sin_la_tercera_categoria_los_dos_conjuntos_coinciden(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1)]
        filas = [fila(1, 0, variante="no"), fila(2, 1, variante="si")]
        a = self.analizar(ensayos, filas)
        self.assertFalse(a["exactitud"]["sin_variante_amplio"]["aplica"])
        self.assertEqual(a["exactitud"]["sin_variante_amplio"]["n_ejecuciones"],
                         a["exactitud"]["sin_variante_estricto"]["n_ejecuciones"])

    def test_variante_en_blanco_cuenta_en_la_totalidad_y_en_ningun_conjunto_limpio(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1)]
        filas = [fila(1, 0, variante="no"), fila(2, 1, variante="")]
        a = self.analizar(ensayos, filas)
        self.assertEqual(a["exactitud"]["totalidad"]["politicas"]["todos"]["n_ejecuciones"], 2)
        self.assertEqual(a["exactitud"]["sin_variante_estricto"]["politicas"]["todos"]["n_ejecuciones"], 1)
        self.assertEqual(a["variantes"]["contadores"]["sin_dato"], 1)
        self.assertTrue(any("sin declaración de variante" in w for w in a["advertencias"]))


# ---------------------------------------------------------------------------
# Repeticiones (PD-14)
# ---------------------------------------------------------------------------

class TestRepeticiones(Base):

    def datos(self):
        """Presentación 1: dos intentos (rechazo por confianza y luego acierto).
        Presentación 2: un intento, error. Presentación 3: dos intentos, el primero acierta."""
        ensayos = [
            ensayo(1, estado="rechazada_confianza", id_clase=0, top3=[0, 1, 2], confianza=0.1),
            ensayo(2, id_clase=0, top3=[0, 1, 2]),
            ensayo(3, id_clase=9, top3=[9, 8, 7]),
            ensayo(4, id_clase=2, top3=[2, 1, 0]),
            ensayo(5, id_clase=8, top3=[8, 7, 6]),
        ]
        filas = [fila(1, 0, presentacion=1, intento=1), fila(2, 0, presentacion=1, intento=2),
                 fila(3, 1, presentacion=2, intento=1),
                 fila(4, 2, presentacion=3, intento=1), fila(5, 2, presentacion=3, intento=2)]
        return ensayos, filas

    def test_tres_politicas(self):
        a = self.analizar(*self.datos())
        pol = a["exactitud"]["totalidad"]["politicas"]
        # todos: 5 ejecuciones, 4 con predicción... todas tienen predicción; aciertos top-1: 1,2,4 -> 3
        self.assertEqual(pol["todos"]["n_ejecuciones"], 5)
        self.assertEqual(pol["todos"]["denominadores"]["con_prediccion"]["aciertos_top1"], 3)
        # primer intento de cada presentación: ensayos 1, 3, 4 -> aciertos 1 y 4 = 2 de 3
        primero = pol["primer_intento"]
        self.assertEqual(primero["n_ejecuciones"], 3)
        self.assertEqual(primero["denominadores"]["con_prediccion"]["aciertos_top1"], 2)
        # último intento: ensayos 2, 3, 5 -> acierta solo el 2
        ultimo = pol["ultimo_intento"]
        self.assertEqual(ultimo["n_ejecuciones"], 3)
        self.assertEqual(ultimo["denominadores"]["con_prediccion"]["aciertos_top1"], 1)
        # El rechazo por confianza solo está en «todos» y en «primer_intento».
        self.assertEqual(pol["todos"]["rechazos"]["contadores"]["rechazada_confianza"], 1)
        self.assertEqual(ultimo["rechazos"]["contadores"]["rechazada_confianza"], 0)

    def test_filas_sin_presentacion_se_cuentan_aparte(self):
        """Si solo algunas filas traen presentacion, las demás son su propia presentación y se cuentan."""
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=0), ensayo(3, id_clase=1)]
        filas = [fila(1, 0, presentacion=1, intento=1), fila(2, 0, presentacion=1, intento=2),
                 "s1,,,,3,1,,,no,si,,"]
        a = self.analizar(ensayos, filas)
        pol = a["exactitud"]["totalidad"]["politicas"]
        self.assertEqual(pol["todos"]["n_ejecuciones"], 3)
        self.assertEqual(pol["primer_intento"]["n_ejecuciones"], 2)
        self.assertEqual(pol["primer_intento"]["n_sin_presentacion"], 1)
        self.assertEqual(pol["ultimo_intento"]["n_ejecuciones"], 2)

    def test_sin_columna_presentacion_no_se_inventa_la_agrupacion(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1)]
        filas = ["s1,,,,1,0,,,no,si,,", "s1,,,,2,1,,,no,si,,"]
        a = self.analizar(ensayos, filas)
        pol = a["exactitud"]["totalidad"]["politicas"]
        self.assertNotIn("no_calculable", pol["todos"])
        self.assertIn("no_calculable", pol["primer_intento"])
        self.assertIn("presentacion", pol["primer_intento"]["no_calculable"])
        self.assertIn("no_calculable", pol["ultimo_intento"])

    def test_una_politica_no_calculable_no_borra_la_tabla_de_las_demas(self):
        """Si la totalidad tiene presentaciones y el conjunto sin variante no, se imprime lo que hay."""
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1)]
        filas = [fila(1, 0, variante="si", presentacion=1, intento=1),
                 "s1,,,,2,1,,,no,si,,"]
        a = self.analizar(ensayos, filas)
        pol = a["exactitud"]
        self.assertNotIn("no_calculable", pol["totalidad"]["politicas"]["primer_intento"])
        self.assertIn("no_calculable", pol["sin_variante_estricto"]["politicas"]["primer_intento"])
        md = eb.informe_md(a)
        # La fila de la totalidad sigue estando y la del conjunto sin variante se explica.
        self.assertIn("| totalidad de las ejecuciones |", md)
        self.assertIn("no se pueden agrupar los intentos", md)

    def test_presentacion_reiniciada_por_ronda_se_avisa(self):
        """Si el operador reinicia la numeración en cada ronda, dos presentaciones distintas comparten
        número: «primer_intento» y «ultimo_intento» se quedarían con un solo intento de las dos. No se
        cambia la agrupación (PD-14 sigue abierto), pero se avisa (revisión independiente, 16-sep-2026)."""
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1),
                   ensayo(3, id_clase=2), ensayo(4, id_clase=3)]
        filas = [fila(1, 0, presentacion=1, ronda=1, intento=1),
                 fila(2, 1, presentacion=2, ronda=1, intento=1),
                 fila(3, 2, presentacion=1, ronda=2, intento=1),
                 fila(4, 3, presentacion=2, ronda=2, intento=1)]
        a = self.analizar(ensayos, filas)
        raras = a["cruce"]["presentaciones_incoherentes"]
        self.assertEqual(len(raras), 2)
        self.assertEqual(raras[0]["glosas_objetivo"], [SIGN_VOCABULARY[0], SIGN_VOCABULARY[2]])
        self.assertEqual(raras[0]["rondas"], ["1", "2"])
        self.assertEqual(raras[0]["claves"], ["s1/1", "s1/3"])
        self.assertTrue(any("presentación" in w and "PD-14" in w for w in a["advertencias"]))
        # El aviso no cambia ninguna cifra: «todos» sigue con las cuatro ejecuciones.
        self.assertEqual(a["exactitud"]["totalidad"]["politicas"]["todos"]["n_ejecuciones"], 4)
        self.assertIn("agrupa más de una glosa", eb.informe_md(a))

    def test_numeracion_de_corrido_no_avisa(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1)]
        filas = [fila(1, 0, presentacion=1, ronda=1, intento=1),
                 fila(2, 1, presentacion=2, ronda=2, intento=1)]
        a = self.analizar(ensayos, filas)
        self.assertEqual(a["cruce"]["presentaciones_incoherentes"], [])
        self.assertEqual(a["advertencias"], [])

    def test_desglose_por_intento_y_por_ronda(self):
        ensayos, filas = self.datos()
        filas = [f + "" for f in filas]
        a = self.analizar(ensayos, filas)
        por_intento = {f["intento"]: f for f in a["por_repeticion"]["por_intento"]}
        self.assertEqual(por_intento["1"]["n_ejecuciones"], 3)
        self.assertEqual(por_intento["2"]["n_ejecuciones"], 2)
        self.assertAlmostEqual(por_intento["2"]["top1_con_prediccion"], 0.5)
        # Sin columna «ronda» todo cae en «sin_dato» y el informe lo dice.
        self.assertEqual([f["ronda"] for f in a["por_repeticion"]["por_ronda"]], ["sin_dato"])

    def test_por_ronda(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1),
                   ensayo(3, id_clase=9, top3=[9, 1, 2]), ensayo(4, id_clase=1)]
        filas = [fila(1, 0, ronda=1), fila(2, 1, ronda=1),
                 fila(3, 0, ronda=2), fila(4, 1, ronda=2)]
        a = self.analizar(ensayos, filas)
        por_ronda = {f["ronda"]: f for f in a["por_repeticion"]["por_ronda"]}
        self.assertEqual(por_ronda["1"]["top1_con_prediccion"], 1.0)
        self.assertEqual(por_ronda["2"]["top1_con_prediccion"], 0.5)
        self.assertEqual(por_ronda["1"]["n_senas_distintas"], 2)


# ---------------------------------------------------------------------------
# Audio dudoso (N-37, N-38, E7, R20)
# ---------------------------------------------------------------------------

class TestAudioDudoso(Base):

    def test_error_a_mitad_de_sesion(self):
        """El contador sube en el ensayo 3: por la convención A son dudosos 3, 4 y 5; por la B también
        el 2, que es el que probablemente no sonó (el audio suena después de t5)."""
        ensayos = [ensayo(1, id_clase=0, errores_voz=0), ensayo(2, id_clase=1, errores_voz=0),
                   ensayo(3, id_clase=2, errores_voz=1, mensaje="afplay murio"),
                   ensayo(4, id_clase=3, errores_voz=1), ensayo(5, id_clase=4, errores_voz=1)]
        filas = [fila(i, i - 1) for i in range(1, 6)]
        a = self.analizar(ensayos, filas)
        ad = a["audio_dudoso"]
        self.assertEqual(ad["convencion_a"], ["s1/3", "s1/4", "s1/5"])
        self.assertEqual(ad["convencion_b"], ["s1/2", "s1/3", "s1/4", "s1/5"])
        self.assertEqual(ad["total"]["ensayos_con_errores"], 3)
        self.assertEqual(ad["total"]["n_convencion_a"], 3)
        self.assertEqual(ad["total"]["n_convencion_b"], 4)
        self.assertEqual(ad["validas_convencion_a"], 3)
        self.assertEqual(ad["validas_convencion_b"], 4)
        inc = ad["incrementos"]
        self.assertEqual(len(inc), 1)
        self.assertEqual((inc[0]["id_ensayo"], inc[0]["id_ensayo_anterior"], inc[0]["incremento"]),
                         (3, 2, 1))
        self.assertEqual(inc[0]["mensaje"], "afplay murio")
        self.assertTrue(any("audio dudoso" in w for w in a["advertencias"]))
        # E7: no se excluye ninguno de la exactitud.
        self.assertEqual(a["exactitud"]["totalidad"]["politicas"]["todos"]["n_ejecuciones"], 5)

    def test_sin_dato_no_es_cero_errores(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1, errores_voz=0)]
        filas = [fila(1, 0), fila(2, 1)]
        a = self.analizar(ensayos, filas)
        t = a["audio_dudoso"]["total"]
        self.assertEqual((t["informados"], t["sin_dato"], t["ensayos_con_errores"]), (1, 1, 0))
        self.assertEqual(a["audio_dudoso"]["convencion_a"], [])
        self.assertFalse(any("audio dudoso" in w for w in a["advertencias"]))

    def test_el_contador_se_reinicia_por_sesion(self):
        """El máximo es el mayor de los máximos, no la suma (N-38)."""
        ensayos = [ensayo(1, id_clase=0, sesion="s1", errores_voz=0),
                   ensayo(2, id_clase=1, sesion="s1", errores_voz=2),
                   ensayo(1, id_clase=2, sesion="s2", errores_voz=0),
                   ensayo(2, id_clase=3, sesion="s2", errores_voz=1)]
        filas = [fila(1, 0, sesion="s1"), fila(2, 1, sesion="s1"),
                 fila(1, 2, sesion="s2"), fila(2, 3, sesion="s2")]
        a = self.analizar(ensayos, filas)
        ad = a["audio_dudoso"]
        self.assertEqual(len(ad["por_sesion"]), 2)
        self.assertEqual({s["sesion"]: s["maximo"] for s in ad["por_sesion"]}, {"s1": 2, "s2": 1})
        self.assertEqual(ad["convencion_a"], ["s1/2", "s2/2"])
        self.assertEqual(ad["convencion_b"], ["s1/1", "s1/2", "s2/1", "s2/2"])
        self.assertEqual(ad["incrementos"][0]["incremento"], 2)

    def test_primer_ensayo_con_error(self):
        """Si el primer registro ya trae el contador > 0 no hay ensayo anterior al que mirar."""
        ensayos = [ensayo(1, id_clase=0, errores_voz=1), ensayo(2, id_clase=1, errores_voz=1)]
        filas = [fila(1, 0), fila(2, 1)]
        a = self.analizar(ensayos, filas)
        ad = a["audio_dudoso"]
        self.assertEqual(ad["convencion_a"], ["s1/1", "s1/2"])
        self.assertEqual(ad["convencion_b"], ["s1/1", "s1/2"])
        self.assertIsNone(ad["incrementos"][0]["id_ensayo_anterior"])


# ---------------------------------------------------------------------------
# Control de configuración (E2)
# ---------------------------------------------------------------------------

class TestConfiguracion(Base):

    def test_una_sola_configuracion_no_avisa(self):
        a = self.analizar([ensayo(1, id_clase=0), ensayo(2, id_clase=1)], [fila(1, 0), fila(2, 1)])
        self.assertEqual(a["configuracion"]["representacion"], {"manos": 2})
        self.assertEqual(a["configuracion"]["umbral"], {"0.2 (modelo_canonico.umbral.json)": 2})
        self.assertEqual(a["advertencias"], [])

    def test_dos_umbrales_avisan_sin_excluir(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1, umbral=0.5)]
        a = self.analizar(ensayos, [fila(1, 0), fila(2, 1)])
        self.assertEqual(len(a["configuracion"]["umbral"]), 2)
        self.assertTrue(any("umbral" in w and "E2" in w for w in a["advertencias"]))
        self.assertEqual(a["exactitud"]["totalidad"]["politicas"]["todos"]["n_ejecuciones"], 2)

    def test_un_rechazo_sin_audio_no_es_una_incidencia_de_sintesis(self):
        """tts_sintesis_antes_de_t5 = null en un rechazo es normal: no hubo audio. E2 mira el false."""
        ensayos = [ensayo(1, id_clase=0), ensayo(2, estado="rechazada_manos")]
        a = self.analizar(ensayos, [fila(1, 0), fila(2, 1)])
        self.assertEqual(a["configuracion"]["tts_sintesis_antes_de_t5"], {"True": 1, "None": 1})
        self.assertEqual(a["advertencias"], [])

    def test_sintesis_fuera_de_t5_avisa(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=1)]
        ensayos[1]["tts_sintesis_antes_de_t5"] = False
        a = self.analizar(ensayos, [fila(1, 0), fila(2, 1)])
        self.assertTrue(any("tts_sintesis_antes_de_t5 = false" in w for w in a["advertencias"]))
        self.assertEqual(a["exactitud"]["totalidad"]["politicas"]["todos"]["n_ejecuciones"], 2)


# ---------------------------------------------------------------------------
# Salidas: JSON, informe .md y línea de órdenes
# ---------------------------------------------------------------------------

class TestSalidas(Base):

    def datos(self):
        ensayos = [ensayo(1, id_clase=0), ensayo(2, id_clase=9, top3=[9, 1, 0]),
                   ensayo(3, estado="rechazada_confianza", id_clase=2, top3=[2, 1, 0], confianza=0.1),
                   ensayo(4, estado="rechazada_manos", errores_voz=1, mensaje="fallo"),
                   ensayo(5, id_clase=4, errores_voz=1)]
        filas = [fila(1, 0), fila(2, 1, variante="si"), fila(3, 2),
                 fila(4, 3, valida="no", observacion="E3"), fila(5, 4)]
        return ensayos, filas

    def test_json_serializable(self):
        a = self.analizar(*self.datos())
        texto = json.dumps(a, ensure_ascii=False)
        self.assertIn("pendientes_de_decision", texto)
        self.assertEqual(json.loads(texto)["cruce"]["n_emparejados"], 5)

    def test_informe_md(self):
        a = self.analizar(*self.datos())
        md = eb.informe_md(a)
        self.assertTrue(md.startswith("# Exactitud en línea del Bloque 1"))
        for titulo in ("## 1. Cruce", "## 2. Validez", "## 3. Exactitud por duplicado",
                       "## 4. Exactitud por seña", "## 5. Exactitud por repetición",
                       "## 6. Matriz de confusión", "## 7. Ensayos con audio dudoso",
                       "## 8. Control de configuración", "## 9. Advertencias",
                       "## 10. Lo que queda por decidir"):
            self.assertIn(titulo, md)
        # El informe declara explícitamente que no elige PD-14 ni PD-15.
        self.assertIn("no elige el denominador", md)
        self.assertIn("PD-15 A", md)
        self.assertIn("PD-15 B", md)
        # Y nombra la ejecución excluida con su motivo.
        self.assertIn("E3", md)

    def test_tabla_de_consola(self):
        a = self.analizar(*self.datos())
        t = eb.tabla(a)
        self.assertIn("Exactitud en linea del Bloque 1", t)
        self.assertIn("PD-14 y PD-15 siguen sin decidir", t)

    def test_main_escribe_json_y_md(self):
        ensayos, filas = self.datos()
        jsonl, hoja = escribir(self.tmp, ensayos, filas)
        salida = self.tmp / "salida"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            codigo = eb.main(["--registros", str(jsonl), "--hoja", str(hoja),
                              "--salida", str(salida)])
        self.assertEqual(codigo, 0)
        self.assertTrue((salida / "exactitud_bloque1.json").is_file())
        self.assertTrue((salida / "exactitud_bloque1.md").is_file())
        guardado = json.loads((salida / "exactitud_bloque1.json").read_text(encoding="utf-8"))
        self.assertEqual(guardado["cruce"]["n_emparejados"], 5)

    def test_main_acepta_una_carpeta_de_registros(self):
        ensayos, filas = self.datos()
        carpeta = self.tmp / "reg"
        carpeta.mkdir()
        escribir(carpeta, ensayos, [])
        _, hoja = escribir(self.tmp, [], filas)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eb.main(["--registros", str(carpeta), "--hoja", str(hoja), "--json", "-"])
        a = json.loads(buf.getvalue())
        self.assertEqual(a["cruce"]["n_emparejados"], 5)

    def test_main_con_json_a_stdout_y_md_a_archivo(self):
        """El JSON a stdout debe seguir siendo JSON: el aviso del .md va a stderr."""
        ensayos, filas = self.datos()
        jsonl, hoja = escribir(self.tmp, ensayos, filas)
        md = self.tmp / "informe.md"
        salida, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(salida), contextlib.redirect_stderr(error):
            eb.main(["--registros", str(jsonl), "--hoja", str(hoja), "--json", "-", "--md", str(md)])
        self.assertEqual(json.loads(salida.getvalue())["cruce"]["n_emparejados"], 5)
        self.assertTrue(md.is_file())
        self.assertIn("Informe:", error.getvalue())

    def test_main_exige_registros_y_hoja(self):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), self.assertRaises(SystemExit):
            eb.main(["--hoja", "x.csv"])
        self.assertIn("--registros", buf.getvalue())

    def test_main_escribe_la_plantilla(self):
        destino = self.tmp / "nueva.csv"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(eb.main(["--plantilla", str(destino)]), 0)
        self.assertTrue(destino.is_file())
        self.assertIn("sesion,presentacion", destino.read_text(encoding="utf-8").splitlines()[0])


class TestSinRutaPorDefecto(unittest.TestCase):
    """Nunca debe leerse registros/ sin que el operador lo pida (el usuario puede estar grabando)."""

    def test_los_argumentos_de_entrada_no_traen_valor_por_defecto(self):
        import argparse
        capturado = {}
        original = argparse.ArgumentParser.add_argument

        def espia(self, *args, **kw):
            if args and isinstance(args[0], str) and args[0].startswith("--"):
                capturado[args[0]] = kw.get("default", "SIN_DEFAULT")
            return original(self, *args, **kw)

        argparse.ArgumentParser.add_argument = espia
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    eb.main([])
        finally:
            argparse.ArgumentParser.add_argument = original
        self.assertIsNone(capturado["--registros"])
        self.assertIsNone(capturado["--hoja"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
