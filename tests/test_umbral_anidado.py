"""
Pruebas de la validación anidada por señante para el umbral de confianza (src/umbral_anidado.py;
experimento de N-46). Con dobles: sin TensorFlow, sin modelos ni processed reales.
    python -m unittest tests.test_umbral_anidado -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import umbral_anidado as ua  # noqa: E402
from src import umbral_confianza as uc  # noqa: E402
from src import train_loso  # noqa: E402

SENANTES = ["0000", "0001", "0002", "0003", "0004"]


def corpus_falso(n_clases=50, reps=4):
    """Nombres SSSS_VVVV_RRRR de un corpus como LSC50 (1000 muestras), con las 3 excluidas de D-06."""
    nombres = [f"{c:04d}_{s}_{r:04d}" for c in range(n_clases) for s in SENANTES for r in range(reps)]
    y = np.array([int(n.split("_")[0]) for n in nombres])
    senantes = [n.split("_")[1] for n in nombres]
    return nombres, y, senantes


def loso_falso():
    return {"senantes": SENANTES, "pliegues": [{"pliegue": k, "senante_test": s} for k, s in enumerate(SENANTES)]}


class TestParticionesSinFuga(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nombres, cls.y, cls.senantes = corpus_falso()
        cls.particion = ua.construir_particion(cls.nombres, cls.y, cls.senantes, loso_falso(), semilla=2026)

    def test_diez_pares_y_veinte_casillas(self):
        self.assertEqual(len(self.particion["pares"]), 10)
        self.assertEqual(len(self.particion["casillas"]), 20)
        self.assertEqual(self.particion["muestras_disponibles"], 997)
        self.assertEqual([p["par"] for p in self.particion["pares"]][:3], ["par_0000_0001", "par_0000_0002", "par_0000_0003"])

    def test_los_senantes_excluidos_no_estan_en_train_ni_val(self):
        for p in self.particion["pares"]:
            c = p["conjuntos"]
            fuera = {p["a"], p["b"]}
            for n in c["train"] + c["val"]:
                self.assertNotIn(n.split("_")[1], fuera, f"{p['par']}: {n} en train/val")
            for s in fuera:
                self.assertTrue(all(n.split("_")[1] == s for n in c["umbral"][s]))
            self.assertEqual(sorted(p["senantes_entrenamiento"]), sorted(set(SENANTES) - fuera))

    def test_conjuntos_disjuntos_cubren_997_y_val_2_por_sena(self):
        excl = set(train_loso.DUPLICADOS_EXCLUIDOS)
        for p in self.particion["pares"]:
            c = p["conjuntos"]
            tr, va = set(c["train"]), set(c["val"])
            um = set(c["umbral"][p["a"]]) | set(c["umbral"][p["b"]])
            self.assertFalse(tr & va or tr & um or va & um)
            self.assertEqual(tr | va | um, set(self.nombres) - excl)
            self.assertEqual(len(va), 100)
            por_clase = {}
            for n in va:
                por_clase[n.split("_")[0]] = por_clase.get(n.split("_")[0], 0) + 1
            self.assertEqual(set(por_clase.values()), {2})
            self.assertEqual(len(tr), 997 - 100 - len(um))

    def test_casilla_nunca_ve_al_senante_de_prueba_ni_al_de_umbral(self):
        por_par = {p["par"]: p for p in self.particion["pares"]}
        vistas = set()
        for cas in self.particion["casillas"]:
            self.assertNotEqual(cas["senante_test"], cas["senante_interno"])
            par = por_par[cas["par"]]
            self.assertEqual({par["a"], par["b"]}, {cas["senante_test"], cas["senante_interno"]})
            prohibidos = {cas["senante_test"], cas["senante_interno"]}
            for n in par["conjuntos"]["train"] + par["conjuntos"]["val"]:
                self.assertNotIn(n.split("_")[1], prohibidos)
            self.assertTrue(all(n.split("_")[1] == cas["senante_interno"] for n in par["conjuntos"]["umbral"][cas["senante_interno"]]))
            vistas.add((cas["pliegue"], cas["senante_interno"]))
        self.assertEqual(len(vistas), 20)
        # cada par alimenta exactamente dos casillas
        usos = {}
        for cas in self.particion["casillas"]:
            usos[cas["par"]] = usos.get(cas["par"], 0) + 1
        self.assertEqual(set(usos.values()), {2})

    def test_semillas_por_par_distintas_y_reproducibles(self):
        semillas = [p["semilla_validacion"] for p in self.particion["pares"]]
        self.assertEqual(len(set(semillas)), 10)
        self.assertEqual(semillas[0], 202601)
        self.assertNotIn(20260, semillas)  # no se reutilizan las del LOSO (semilla*10 + k)
        otra = ua.construir_particion(self.nombres, self.y, self.senantes, loso_falso(), semilla=2026)
        self.assertEqual(ua._comparable(otra), ua._comparable(self.particion))
        distinta = ua.construir_particion(self.nombres, self.y, self.senantes, loso_falso(), semilla=2027)
        self.assertNotEqual(ua._comparable(distinta), ua._comparable(self.particion))

    def test_verificacion_detecta_fuga(self):
        par = json.loads(json.dumps(self.particion["pares"][0]))
        colado = par["conjuntos"]["umbral"][par["a"]].pop()
        par["conjuntos"]["train"].append(colado)
        with self.assertRaises(SystemExit):
            ua.verificar_par(par, self.nombres)

    def test_no_se_persiste_en_data_partitions(self):
        with self.assertRaises(ValueError):
            ua.persistir_particion(self.particion, ROOT_DIR / "data" / "partitions" / "x.json")
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "particiones_anidadas.json"
            ua.persistir_particion(self.particion, ruta)
            self.assertTrue(ruta.exists())
            ua.persistir_particion(self.particion, ruta)  # idéntica: no falla
            otra = ua.construir_particion(self.nombres, self.y, self.senantes, loso_falso(), semilla=2027)
            with self.assertRaises(SystemExit):
                ua.persistir_particion(otra, ruta)


class TestSelectorYAgregacion(unittest.TestCase):

    def test_reutiliza_el_selector_de_D09(self):
        self.assertIs(ua.elegir_umbral, uc.elegir_umbral)
        self.assertIs(ua.utilidad, uc.utilidad)
        self.assertIs(ua.efecto, uc.efecto)
        self.assertIs(ua.REJILLA, uc.REJILLA)

    def test_mediana_en_la_rejilla(self):
        self.assertEqual(ua.mediana_rejilla([0.47, 0.55, 0.47, 0.35]), 0.47)
        self.assertEqual(ua.mediana_rejilla([0.47, 0.55, 0.47, 0.55]), 0.51)
        self.assertEqual(ua.mediana_rejilla([0.4, 0.5, 0.6, 0.7, 0.9]), 0.6)

    def test_resumen_coincide_con_efecto_de_D09_con_umbral_uniforme(self):
        rng = np.random.default_rng(1)
        conf = rng.uniform(size=300)
        ok = rng.uniform(size=300) < conf
        e = uc.efecto(conf, ok, 0.4)
        r = ua.resumen(ok, conf >= 0.4)
        for k in e:
            self.assertAlmostEqual(e[k], r[k], msg=k)
        self.assertEqual(r["utilidad"], uc.utilidad(conf, ok, 0.4, 1.0))

    def test_efecto_prueba_loso_por_pliegue_y_agregado(self):
        prueba = {0: (np.array([0.1, 0.6, 0.9]), np.array([True, False, True])),
                  1: (np.array([0.3, 0.7]), np.array([True, True]))}
        r = ua.efecto_prueba_loso(prueba, {0: 0.5, 1: 0.2})
        self.assertEqual(r["por_pliegue"]["0"]["aceptadas_correctas"], 1)
        self.assertEqual(r["por_pliegue"]["0"]["aceptadas_incorrectas"], 1)
        self.assertEqual(r["por_pliegue"]["1"]["aceptadas_correctas"], 2)
        a = r["agregado"]
        self.assertEqual((a["n"], a["aceptadas_correctas"], a["aceptadas_incorrectas"]), (5, 3, 1))
        self.assertAlmostEqual(a["tasa_rechazo"], 1 / 5)
        self.assertEqual(a["utilidad"], 2.0)
        self.assertEqual(a["umbrales"], {"0": 0.5, "1": 0.2})


def _escribir_par_falso(carpeta: Path, par: dict, rng, sesgo=0.0):
    """predicciones_<s>.json y predicciones.json de un par con un modelo falso: la confianza es
    mayor cuando acierta, para que el selector tenga algo que elegir."""
    carpeta.mkdir(parents=True, exist_ok=True)
    todas = []
    for s in (par["a"], par["b"]):
        filas = []
        for n in par["conjuntos"]["umbral"][s]:
            real = int(n.split("_")[0])
            acierta = rng.uniform() < 0.6
            pred = real if acierta else (real + 1) % 50
            conf = float(np.clip(rng.normal(0.7 if acierta else 0.35, 0.15) + sesgo, 0.02, 0.99))
            soft = np.full(50, (1 - conf) / 49)
            soft[pred] = conf
            filas.append({"muestra": n, "senante": s, "real": real, "predicha": pred, "confianza": conf,
                          "softmax": [round(float(v), 6) for v in soft]})
        (carpeta / f"predicciones_{s}.json").write_text(json.dumps(filas), encoding="utf-8")
        todas += [{"muestra": f["muestra"], "real": f["real"], "predicha": f["predicha"], "top3": [f["predicha"]],
                   "prob_predicha": f["confianza"]} for f in filas]
    (carpeta / "predicciones.json").write_text(json.dumps(todas), encoding="utf-8")
    (carpeta / "modelo.keras").write_bytes(b"falso")
    (carpeta / "metricas.json").write_text(json.dumps({"top1": 0.6}), encoding="utf-8")


class TestEleccionConDobles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nombres, cls.y, cls.senantes = corpus_falso()
        cls.particion = ua.construir_particion(cls.nombres, cls.y, cls.senantes, loso_falso(), semilla=2026)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        rng = np.random.default_rng(7)
        for p in cls.particion["pares"]:
            _escribir_par_falso(cls.out / p["par"], p, rng)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_elige_por_casilla_pliegue_y_global_con_el_selector_de_D09(self):
        llamadas = []
        original = uc.elegir_umbral

        def espia(conf, ok, c):
            llamadas.append((len(conf), c))
            return original(conf, ok, c)

        with mock.patch.object(ua, "elegir_umbral", espia):
            sel = ua.elegir_umbrales(self.out, self.particion)
        self.assertEqual(len(sel["casillas"]), 20)
        self.assertEqual(len(sel["pliegues"]), 5)
        # 20 casillas x 3 c + 5 pliegues x (1 + 3) + global (1 + 3) llamadas, todas al selector de D-09
        self.assertEqual(len(llamadas), 20 * 3 + 5 * 4 + 4)
        self.assertTrue(all(c in (0.5, 1.0, 2.0) for _, c in llamadas))
        for p in sel["pliegues"]:
            self.assertEqual(p["n"], sum(cs["n"] for cs in sel["casillas"] if cs["pliegue"] == p["pliegue"]))
            self.assertTrue(0.0 <= p["t"]["1.0"] < 1.0)
            self.assertLessEqual(p["t"]["0.5"], p["t"]["1.0"])
            self.assertLessEqual(p["t"]["1.0"], p["t"]["2.0"])
        g = sel["global"]
        self.assertEqual(g["n"], sum(p["n"] for p in sel["pliegues"]))
        self.assertEqual(sel["umbral_candidato"], g["t"]["1.0"])
        self.assertEqual(g["t_mediana_de_pliegues"]["1.0"], ua.mediana_rejilla([p["t"]["1.0"] for p in sel["pliegues"]]))
        self.assertEqual(g["t_mediana_de_medianas"]["1.0"],
                         ua.mediana_rejilla([p["t_mediana_casillas"]["1.0"] for p in sel["pliegues"]]))
        self.assertEqual(sel["dispersion_t_pliegues"]["valores"], [p["t"]["1.0"] for p in sel["pliegues"]])

    def test_predicciones_que_no_corresponden_a_la_particion_se_rechazan(self):
        with tempfile.TemporaryDirectory() as tmp2:
            out2 = Path(tmp2)
            rng = np.random.default_rng(3)
            for p in self.particion["pares"]:
                _escribir_par_falso(out2 / p["par"], p, rng)
            ruta = out2 / "par_0000_0001" / "predicciones_0000.json"
            filas = json.loads(ruta.read_text(encoding="utf-8"))
            filas[0], filas[1] = filas[1], filas[0]
            ruta.write_text(json.dumps(filas), encoding="utf-8")
            with self.assertRaises(SystemExit):
                ua.elegir_umbrales(out2, self.particion)

    def test_candidato_tiene_el_formato_del_umbral_vigente_y_no_pisa_un_umbral_json(self):
        sel = ua.elegir_umbrales(self.out, self.particion)
        vigente = ROOT_DIR / "models" / "canonico_holistic_mascara_manos" / "modelo_canonico.umbral.json"
        claves_vigente = set(json.loads(vigente.read_text(encoding="utf-8"))) if vigente.exists() else {
            "umbral", "decision", "c", "seleccionado_con", "modelo", "sha256_modelo", "script", "informe", "fecha"}
        with tempfile.TemporaryDirectory() as tmp2:
            modelo = Path(tmp2) / "modelo_canonico.keras"
            modelo.write_bytes(b"canonico-falso")
            with self.assertRaises(ValueError):
                ua.escribir_candidato(Path(tmp2) / "modelo_canonico.umbral.json", 0.4, modelo, Path(tmp2) / "i.md", sel, {})
            destino = Path(tmp2) / "candidato_umbral.json"
            datos = ua.escribir_candidato(destino, sel["umbral_candidato"], modelo, Path(tmp2) / "informe.md", sel,
                                          {"vigente_0.20": {"utilidad": 1}})
            leido = json.loads(destino.read_text(encoding="utf-8"))
            self.assertTrue(claves_vigente <= set(leido), claves_vigente - set(leido))
            self.assertEqual(leido["umbral"], sel["umbral_candidato"])
            self.assertEqual(leido["c"], 1.0)
            self.assertEqual(leido["estado"], "candidato_no_adoptado")
            self.assertIn("anidada", leido["seleccionado_con"])
            import hashlib
            self.assertEqual(leido["sha256_modelo"], hashlib.sha256(b"canonico-falso").hexdigest())
            self.assertEqual(len(leido["umbrales_por_casilla"]), 20)
            self.assertEqual(len(leido["umbrales_por_pliegue"]), 5)
            self.assertEqual(leido["umbral_global"], sel["global"]["t"])
            self.assertEqual(leido["umbral_global_mediana_de_pliegues"], sel["global"]["t_mediana_de_pliegues"])
            self.assertEqual(datos["efecto_prueba_loso"], {"vigente_0.20": {"utilidad": 1}})
        if vigente.exists():
            self.assertEqual(json.loads(vigente.read_text(encoding="utf-8"))["umbral"], 0.2)  # sigue intacto

    def test_control_transferencia_por_senante(self):
        prueba = {}
        rng = np.random.default_rng(11)
        for k, s in enumerate(SENANTES):
            n = 200 if s != "0001" else 199
            prueba[k] = (rng.uniform(size=n), rng.uniform(size=n) < 0.5)
        r = ua.control_transferencia(self.out, self.particion, prueba, loso_falso(), 0.4)
        for s in SENANTES:
            self.assertEqual(r[s]["interno_3_senantes"]["modelos"], 4)
            self.assertEqual(r[s]["externo_4_senantes"]["n"], len(prueba[SENANTES.index(s)][0]))
            self.assertIn("fraccion_conf_menor_0.40", r[s]["interno_3_senantes"])
        self.assertIn("diferencia_medianas", r["global"])


class TestControlN33(unittest.TestCase):

    def test_control_con_las_predicciones_reales_si_existen(self):
        loso_dir = ROOT_DIR / ua.LOSO_DIR_DEFECTO
        if not (loso_dir / "pliegue_4" / "predicciones.json").exists():
            self.skipTest("no estan las predicciones LOSO de N-30")
        loso = json.loads(ua.ARCHIVO_LOSO.read_text(encoding="utf-8"))
        prueba = ua.cargar_prueba_loso(loso_dir, loso)
        self.assertEqual(sum(len(v[0]) for v in prueba.values()), 997)
        control = ua.control_n33(prueba)
        self.assertTrue(all(v["coincide"] for v in control.values()), control)

    def test_control_falla_si_las_cifras_no_son_las_de_N33(self):
        prueba = {k: (np.full(10, 0.9), np.ones(10, dtype=bool)) for k in range(5)}
        control = ua.control_n33(prueba)
        self.assertFalse(any(v["coincide"] for v in control.values()))


class TestHuecosDetectadosPorElVerificador(unittest.TestCase):
    """Añadidas por el verificador (2026-09-22): cuatro mutaciones del código sobrevivían a las 17
    pruebas originales (aceptación con > en vez de >=, verificar_casillas sin efecto, U agregada sin c,
    t_k de la selección con otro c)."""

    @classmethod
    def setUpClass(cls):
        cls.nombres, cls.y, cls.senantes = corpus_falso()
        cls.particion = ua.construir_particion(cls.nombres, cls.y, cls.senantes, loso_falso(), semilla=2026)

    def test_verificar_casillas_detecta_fuga_del_senante_de_prueba(self):
        pares = json.loads(json.dumps(self.particion["pares"]))
        cas = self.particion["casillas"][0]
        par = next(p for p in pares if p["par"] == cas["par"])
        par["conjuntos"]["val"].append(par["conjuntos"]["umbral"][cas["senante_test"]][0])
        with self.assertRaises(SystemExit):
            ua.verificar_casillas(self.particion["casillas"], pares)

    def test_confianza_igual_al_umbral_se_acepta(self):
        prueba = {0: (np.array([0.5, 0.49]), np.array([True, True]))}
        a = ua.efecto_prueba_loso(prueba, {0: 0.5})["agregado"]
        self.assertEqual((a["aceptadas_correctas"], a["tasa_rechazo"]), (1, 0.5))

    def test_utilidad_agregada_usa_c(self):
        r = ua.resumen(np.array([True, False, False]), np.array([True, True, True]), c=2.0)
        self.assertEqual(r["utilidad"], 1 - 2 * 2)

    def test_t_k_es_el_selector_de_D09_sobre_la_union_de_las_cuatro_casillas(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rng = np.random.default_rng(11)
            for p in self.particion["pares"]:
                _escribir_par_falso(out / p["par"], p, rng)
            sel = ua.elegir_umbrales(out, self.particion)
            for p in sel["pliegues"]:
                confs, oks = [], []
                for cas in self.particion["casillas"]:
                    if cas["pliegue"] == p["pliegue"]:
                        conf, ok, _ = ua.cargar_predicciones_senante(out / cas["par"], cas["senante_interno"])
                        confs.append(conf)
                        oks.append(ok)
                t = uc.elegir_umbral(np.concatenate(confs), np.concatenate(oks), 1.0)[0]
                self.assertEqual(p["t"]["1.0"], t)
                self.assertEqual(p["efecto_en_seleccion"]["umbral"], t)
            self.assertEqual(sel["global"]["efecto_en_seleccion"]["umbral"], sel["umbral_candidato"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
