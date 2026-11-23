"""
Pruebas sin cámara del aviso de seña cortada en los bordes de la ventana (N-50, opción B;
src/bordes_ventana.py, src/translator.py, src/realtime.py) y de los predeterminados del prototipo
(modelo final + conversor «inventario» + captura 1280 × 720, 22-nov-2026).

  · La medida es la de src/analisis_senas_cortadas.py: con marcas a 24 fot./s reproduce rasgos_clip × 24
    (sintético y, si está en disco, un clip real del processed); la velocidad por segundo no depende de la
    tasa; los umbrales son los del archivo candidato; el borde exacto del umbral no avisa.
  · Ventanas sintéticas: reposo-seña-reposo no avisa; empezar a media seña → aviso_inicio; acabar a media
    seña → aviso_fin; sin postura en un borde → None.
  · En el prototipo: el aviso va en el JSONL (clave «bordes», última) con las banderas, los valores y el costo;
    NO cambia el estado, la clase, la confianza ni ninguna marca t0..t5; «off» no calcula nada; los rechazos
    también lo llevan y la ventana abortada lo deja en null; es igual con la representación de la red heredada
    («mascara») y con la del modelo final («imputacion»), y con los dos modelos reales si están en disco.
  · realtime.py: predeterminados nuevos, texto del aviso, consola y aviso de forma de la captura.

Ejecutar:
    python -m unittest tests.test_bordes_ventana -v
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import analisis_senas_cortadas as sc  # noqa: E402
from src import bordes_ventana as bv  # noqa: E402
from src.translator import (  # noqa: E402
    ESTADOS, IDX_MANO_DER, IDX_MANO_IZQ, IDX_POSE, MODELO_FINAL, MODELO_HEREDADO, MODELO_POR_DEFECTO, N_FEATURES,
    Extraccion, LSCTranslator, normalize_sequence,
)

CANDIDATO = ROOT_DIR / "models" / "senas_cortadas_2026-09-22" / "umbral_bordes_candidato.json"
PROCESSED = ROOT_DIR / "data" / "processed_holistic_mascara_rostro"
CLAVES_BORDES = ["modo", "umbrales", "aviso_inicio", "aviso_fin", "K", "fps_ventana", "v_pose_ini_por_s",
                 "v_pose_fin_por_s", "h_ini", "h_fin", "pares_validos_ini", "pares_validos_fin", "costo_ns"]

# ---------------------------------------------------------------------------
# Ventanas sintéticas en coordenadas de imagen (como las entrega el extractor)
# ---------------------------------------------------------------------------
# Hombros en (0,6; 0,4) y (0,4; 0,4): centro (0,5; 0,4), escala 0,2. En el espacio normalizado, el reposo
# tiene las muñecas en h = 2,0 (bajo los umbrales de altura 0,974 / 1,261, que avisan por DEBAJO) y quietas;
# la seña las sube a h = 0,5 y las mueve 0,3 anchos de hombro de amplitud a 2 Hz (≈ 2,4 anchos/s de media).
ESCALA, CX, CY = 0.2, 0.5, 0.4
H_REPOSO, H_SENA, AMPLITUD, FRECUENCIA = 2.0, 0.5, 0.3, 2.0
FPS = 30.0


def munecas(t_s: float, en_sena: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    """Posición en la imagen de las muñecas 15 y 16 en el instante t_s."""
    if en_sena:
        dx = AMPLITUD * np.sin(2 * np.pi * FRECUENCIA * t_s)
        h = H_SENA
    else:
        dx, h = 0.0, H_REPOSO
    y = CY + ESCALA * h
    return (CX + ESCALA * (0.6 + dx), y), (CX + ESCALA * (-0.6 + dx), y)


def fotograma(t_s: float, en_sena: bool, pose: bool = True) -> Extraccion:
    kp = np.zeros(N_FEATURES, dtype=np.float32)
    if pose:
        p = np.random.default_rng(0).uniform(0.3, 0.7, size=(33, 3)).astype(np.float32)
        p[11] = [0.6, 0.4, 0.0]
        p[12] = [0.4, 0.4, 0.0]
        (xi, yi), (xd, yd) = munecas(t_s, en_sena)
        p[15, :2] = (xi, yi)
        p[16, :2] = (xd, yd)
        kp[IDX_POSE] = p.reshape(-1)
    kp[IDX_MANO_IZQ] = np.random.default_rng(1).uniform(0.3, 0.7, size=63).astype(np.float32)
    kp[IDX_MANO_DER] = np.random.default_rng(2).uniform(0.3, 0.7, size=63).astype(np.float32)
    return Extraccion(keypoints=kp, pose_detectada=pose, n_manos=2, rostro=None, holistic_result=None)


def ventana(n: int = 60, sena=(12, 48), sin_pose=(), fps: float = FPS) -> list[Extraccion]:
    """n fotogramas; la seña ocupa [sena[0], sena[1]); el resto es reposo."""
    return [fotograma(i / fps, sena[0] <= i < sena[1], pose=i not in sin_pose) for i in range(n)]


def pose_normalizada(fotogramas: list[Extraccion]) -> np.ndarray:
    pose = np.stack([f.keypoints[IDX_POSE] for f in fotogramas]).astype(np.float32)
    pose[~np.asarray([f.pose_detectada for f in fotogramas])] = 0.0
    return normalize_sequence(pose)


def marcas(n: int, fps: float = FPS, inicio: int = 10**9) -> list[int]:
    return [inicio + int(round(i * 1e9 / fps)) for i in range(n)]


class ExtractorLista:
    """Devuelve los fotogramas de una lista en orden (vuelve a empezar con reiniciar())."""

    descripcion = "extractor_lista"

    def __init__(self, fotogramas):
        self.fotogramas = fotogramas
        self.i = 0

    def reiniciar(self):
        self.i = 0

    def extract(self, frame_rgb):
        f = self.fotogramas[self.i % len(self.fotogramas)]
        self.i += 1
        return Extraccion(keypoints=f.keypoints.copy(), pose_detectada=f.pose_detectada, n_manos=f.n_manos,
                          rostro=None, holistic_result=None)

    def close(self):
        pass


class RelojFalso:
    def __init__(self, paso_ns: int = 1_000):
        self.t = 10**9
        self.paso = paso_ns

    def __call__(self) -> int:
        self.t += self.paso
        return self.t


class ModeloFalso:
    def __init__(self, n_rasgos: int = 252, id_clase: int = 7, confianza: float = 0.9):
        self.input_shape = (None, 30, n_rasgos)
        self.id_clase, self.confianza = id_clase, confianza

    def predict(self, X, verbose=0):
        prob = np.full(50, (1.0 - self.confianza) / 49, dtype=np.float32)
        prob[self.id_clase] = self.confianza
        return np.repeat(prob[np.newaxis], X.shape[0], axis=0)


class TTSFalso:
    mode = "falso"

    def __init__(self):
        self.hablado = []
        self.is_speaking = False

    def speak_async(self, texto):
        self.hablado.append(texto)


def procedencia(mano_ausente: str) -> dict:
    return {"representacion": "manos", "n_rasgos_entrada": 252, "seq_length": 30, "velocidades": True,
            "mano_ausente": mano_ausente, "rostro_del_processed": "cejas_labios"}


def traductor(tmp, fotogramas, **kw) -> LSCTranslator:
    params = dict(modelo=ModeloFalso(), extractor=ExtractorLista(fotogramas), tts=TTSFalso(), reloj=RelojFalso(),
                  registro_dir=Path(tmp) / "registros", ventana_frames=len(fotogramas), min_frames=5,
                  umbral=0.5, calentar=False, procedencia=procedencia("imputacion"))
    params.update(kw)
    with redirect_stdout(io.StringIO()):
        return LSCTranslator(**params)


def correr(tr: LSCTranslator, n: int, fps: float = FPS) -> dict:
    """Una ventana de n fotogramas con c_i a la tasa fps (marcas posteriores a t0)."""
    if hasattr(tr.extractor, "reiniciar"):
        tr.extractor.reiniciar()
    t0 = tr.ahora()
    tr.abrir_ventana(t0)
    for i in range(n):
        res = tr.procesar_fotograma(np.zeros((4, 4, 3), np.uint8), c_i=t0 + 1_000_000 + int(round(i * 1e9 / fps)))
        if res.registro is not None:
            return res.registro
    raise AssertionError("la ventana no se cerró")


def leer_jsonl(ruta) -> list[dict]:
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# Módulo: umbrales y medida
# ---------------------------------------------------------------------------

class TestUmbrales(unittest.TestCase):

    def test_candidato_se_lee_tal_cual(self):
        u = bv.cargar_umbrales_bordes()
        datos = json.loads(CANDIDATO.read_text(encoding="utf-8"))
        self.assertEqual(Path(u["ruta"]), CANDIDATO)
        self.assertEqual(u["borde_s"], datos["borde_s"])
        self.assertEqual(u["v_pose_por_s"], datos["v_pose_por_s"])
        self.assertEqual(u["h"], datos["h"])
        self.assertAlmostEqual(u["borde_s"], 0.125)
        self.assertAlmostEqual(u["v_pose_por_s"]["ini"], 1.7614, places=4)
        self.assertAlmostEqual(u["v_pose_por_s"]["fin"], 0.2295, places=4)
        self.assertAlmostEqual(u["h"]["ini"], 0.9740, places=4)
        self.assertAlmostEqual(u["h"]["fin"], 1.2610, places=4)
        self.assertIn("CANDIDATO", u["estado"])
        # Son los del análisis: umbral por fotograma a 24 fps × 24 = umbral por segundo.
        res = json.loads((CANDIDATO.parent / "resultados.json").read_text(encoding="utf-8"))
        cand = res["umbral_candidato_cinco_senantes"]
        for b in ("ini", "fin"):
            self.assertAlmostEqual(u["v_pose_por_s"][b], cand["umbrales"]["v_pose"][b] * sc.FPS_CORPUS, places=9)
            self.assertAlmostEqual(u["h"][b], cand["umbrales"]["h"][b], places=9)
        self.assertAlmostEqual(u["borde_s"], cand["K"] / sc.FPS_CORPUS)

    def test_archivos_invalidos(self):
        tmp = Path(tempfile.mkdtemp(prefix="tesis_bordes_"))
        with self.assertRaises(FileNotFoundError):
            bv.cargar_umbrales_bordes(tmp / "no_existe.json")
        base = json.loads(CANDIDATO.read_text(encoding="utf-8"))
        malos = []
        for quitar in ("borde_s", "v_pose_por_s", "h"):
            d = dict(base); del d[quitar]; malos.append(d)
        d = json.loads(json.dumps(base)); del d["h"]["fin"]; malos.append(d)
        d = json.loads(json.dumps(base)); d["v_pose_por_s"]["ini"] = "1.7"; malos.append(d)
        d = json.loads(json.dumps(base)); d["borde_s"] = 0; malos.append(d)
        d = json.loads(json.dumps(base)); d["h"] = [1, 2]; malos.append(d)
        for i, d in enumerate(malos):
            ruta = tmp / f"malo_{i}.json"
            ruta.write_text(json.dumps(d), encoding="utf-8")
            with self.subTest(i=i), self.assertRaises(ValueError):
                bv.cargar_umbrales_bordes(ruta)


class TestMedidaDelAnalisis(unittest.TestCase):
    """La medida del prototipo es la de rasgos_clip del análisis, pasada a unidades por segundo."""

    def _comparar(self, seq: np.ndarray, borde_s: float = 0.125):
        N = seq.shape[0]
        m = bv.medir_bordes(seq, marcas(N, fps=sc.FPS_CORPUS), borde_s)
        K = max(3, round(borde_s * sc.FPS_CORPUS))
        completo = seq if seq.shape[1] >= 225 else np.pad(seq, ((0, 0), (0, 225 - seq.shape[1])))
        r = sc.rasgos_clip(completo, K)                               # rasgos_clip espera las 225 columnas
        self.assertEqual(m["K"], K)
        self.assertAlmostEqual(m["fps_ventana"], sc.FPS_CORPUS, places=6)
        for mio, suyo, factor in (("v_pose_ini_por_s", "v_pose_ini", sc.FPS_CORPUS),
                                  ("v_pose_fin_por_s", "v_pose_fin", sc.FPS_CORPUS),
                                  ("h_ini", "h_ini", 1.0), ("h_fin", "h_fin", 1.0)):
            if np.isnan(r[suyo]):
                self.assertIsNone(m[mio], mio)
            else:
                self.assertAlmostEqual(m[mio], r[suyo] * factor, delta=1e-6 * max(1.0, abs(r[suyo] * factor)),
                                       msg=mio)
        return m

    def test_sintetico_a_24_fps_reproduce_rasgos_clip(self):
        fot = ventana(n=60, sena=(0, 50), fps=sc.FPS_CORPUS)          # empieza a media seña
        seq = pose_normalizada(fot).astype(np.float64)
        m = self._comparar(seq)
        self.assertGreater(m["v_pose_ini_por_s"], 1.0)
        self.assertEqual(m["v_pose_fin_por_s"], 0.0)
        # Con huecos de postura en los bordes: mismos NaN/None que el análisis.
        fot = ventana(n=40, sena=(5, 30), sin_pose=(0, 1, 38), fps=sc.FPS_CORPUS)
        self._comparar(pose_normalizada(fot).astype(np.float64))

    @unittest.skipUnless((PROCESSED / "X_padded.npy").exists(), "processed del corpus no disponible")
    def test_clips_reales_del_processed(self):
        X = np.load(PROCESSED / "X_padded.npy", mmap_mode="r")
        L = np.load(PROCESSED / "lengths.npy")
        for i in (0, 123, 517, 996):
            with self.subTest(clip=i):
                self._comparar(np.asarray(X[i, :L[i], :225], dtype=np.float64))

    def test_velocidad_por_segundo_no_depende_de_la_tasa(self):
        """La misma trayectoria (lineal, 1,5 anchos de hombro por segundo) muestreada a 15 y a 30 fot./s."""
        def trayectoria(fps, n):
            seq = np.zeros((n, 99))
            seq[:, 33:35] = (0.5, 0.0)
            seq[:, 36:38] = (-0.5, 0.0)
            t = np.arange(n) / fps
            seq[:, 45] = 0.6 + 1.5 * t
            seq[:, 48] = -0.6 + 1.5 * t
            seq[:, 46] = seq[:, 49] = 1.5
            return seq, marcas(n, fps)
        for fps, n in ((15.0, 30), (30.0, 60)):
            seq, c = trayectoria(fps, n)
            m = bv.medir_bordes(seq, c, 0.125)
            self.assertAlmostEqual(m["v_pose_ini_por_s"], 1.5, places=6)
            self.assertAlmostEqual(m["v_pose_fin_por_s"], 1.5, places=6)
            self.assertEqual(m["K"], max(3, round(0.125 * fps)))
        # Marcas irregulares (como el bucle real): se divide entre el Δt de cada par.
        seq, _ = trayectoria(30.0, 20)
        c = np.cumsum([0] + [int(1e9 / 30)] * 19)
        c[5:] += 40_000_000                                          # un fotograma tardío
        t = (c - c[0]) / 1e9
        seq[:, 45] = 0.6 + 1.5 * t
        seq[:, 48] = -0.6 + 1.5 * t
        m = bv.medir_bordes(seq, c, 0.125)
        self.assertAlmostEqual(m["v_pose_ini_por_s"], 1.5, places=6)

    def test_ventanas_degeneradas(self):
        u = bv.cargar_umbrales_bordes()
        for seq, c in ((np.zeros((0, 99)), []), (np.zeros((1, 99)), [5]), (np.zeros((3, 99)), [7, 7, 7])):
            m = bv.medir_bordes(seq, c, 0.125)
            self.assertTrue(all(m[k] is None for k in bv.MEDIDAS + ("K", "fps_ventana")))
            self.assertEqual(bv.evaluar_bordes(m, u), {"aviso_inicio": None, "aviso_fin": None})
        with self.assertRaises(ValueError):
            bv.medir_bordes(np.zeros((4, 99)), [1, 2, 3], 0.125)
        # K nunca pasa de N
        fot = ventana(n=3, sena=(0, 0))
        self.assertEqual(bv.medir_bordes(pose_normalizada(fot), marcas(3, fps=500.0), 0.125)["K"], 3)


class TestRegla(unittest.TestCase):

    def setUp(self):
        self.u = bv.cargar_umbrales_bordes()

    def _avisos(self, fot):
        m = bv.medir_bordes(pose_normalizada(fot), marcas(len(fot)), self.u["borde_s"])
        return m, bv.evaluar_bordes(m, self.u)

    def test_reposo_sena_reposo_no_avisa(self):
        m, a = self._avisos(ventana(sena=(12, 48)))
        self.assertEqual(a, {"aviso_inicio": False, "aviso_fin": False})
        self.assertEqual((m["v_pose_ini_por_s"], m["v_pose_fin_por_s"]), (0.0, 0.0))
        self.assertAlmostEqual(m["h_ini"], H_REPOSO, places=5)
        self.assertEqual(m["K"], 4)                                   # round(0,125 × 30)

    def test_empieza_a_media_sena(self):
        _, a = self._avisos(ventana(sena=(0, 48)))
        self.assertEqual(a, {"aviso_inicio": True, "aviso_fin": False})

    def test_acaba_a_media_sena(self):
        _, a = self._avisos(ventana(sena=(12, 60)))
        self.assertEqual(a, {"aviso_inicio": False, "aviso_fin": True})

    def test_sin_postura_en_un_borde_es_none(self):
        m, a = self._avisos(ventana(sena=(12, 48), sin_pose=(0, 1, 2, 3)))
        self.assertIsNone(m["v_pose_ini_por_s"])
        self.assertIsNone(m["h_ini"])
        self.assertEqual(m["pares_validos_ini"], 0)
        self.assertEqual(a, {"aviso_inicio": None, "aviso_fin": False})

    def test_un_rasgo_indefinido_no_dispara_y_el_otro_decide(self):
        self.assertEqual(bv.evaluar_bordes({"v_pose_ini_por_s": None, "h_ini": 0.1,
                                            "v_pose_fin_por_s": 9.0, "h_fin": None}, self.u),
                         {"aviso_inicio": True, "aviso_fin": True})
        self.assertEqual(bv.evaluar_bordes({"v_pose_ini_por_s": None, "h_ini": 3.0,
                                            "v_pose_fin_por_s": 0.0, "h_fin": None}, self.u),
                         {"aviso_inicio": False, "aviso_fin": False})

    def test_borde_exacto_del_umbral(self):
        """Desigualdades estrictas, como aplicar_regla del análisis: igual al umbral no avisa."""
        v, h = self.u["v_pose_por_s"], self.u["h"]
        for borde, clave in (("ini", "aviso_inicio"), ("fin", "aviso_fin")):
            igual = {f"v_pose_{borde}_por_s": v[borde], f"h_{borde}": h[borde]}
            self.assertFalse(bv.evaluar_bordes(igual, self.u)[clave])
            arriba = {f"v_pose_{borde}_por_s": np.nextafter(v[borde], np.inf), f"h_{borde}": h[borde]}
            self.assertTrue(bv.evaluar_bordes(arriba, self.u)[clave])
            abajo = {f"v_pose_{borde}_por_s": v[borde], f"h_{borde}": np.nextafter(h[borde], -np.inf)}
            self.assertTrue(bv.evaluar_bordes(abajo, self.u)[clave])
        # La misma regla que aplicar_regla del análisis (rasgo «v_pose ∨ h») sobre valores por fotograma.
        u_fot = {"v_pose": {b: v[b] / 24 for b in ("ini", "fin")}, "h": dict(h)}
        for vi, hi in ((v["ini"] / 24, h["ini"]), (v["ini"] / 24 * 1.01, 2.0), (0.0, h["ini"] * 0.99)):
            r = sc.aplicar_regla({"v_pose_ini": np.array([vi]), "h_ini": np.array([hi]),
                                  "v_pose_fin": np.array([0.0]), "h_fin": np.array([2.0])}, u_fot, "v_pose_o_h")
            mio = bv.evaluar_bordes({"v_pose_ini_por_s": vi * 24, "h_ini": hi,
                                     "v_pose_fin_por_s": 0.0, "h_fin": 2.0}, self.u)
            self.assertEqual(bool(r["cortada_ini"][0]), mio["aviso_inicio"], (vi, hi))


# ---------------------------------------------------------------------------
# Prototipo: el aviso en el JSONL, sin rechazar
# ---------------------------------------------------------------------------

class TestAvisoEnElPrototipo(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tesis_bordes_")

    def test_jsonl_con_banderas_valores_y_costo(self):
        tr = traductor(self.tmp, ventana(sena=(0, 48)))              # empieza a media seña
        reg = correr(tr, 60)
        self.assertEqual(reg["estado"], "aceptada")                    # el aviso no la rechaza
        self.assertEqual(reg["clase"], "comer")
        self.assertIsNotNone(reg["t5"])
        self.assertEqual(tr.tts.hablado, [reg["enunciado"]])           # y se dijo en voz alta
        b = reg["bordes"]
        self.assertEqual(list(b), CLAVES_BORDES)
        self.assertEqual(list(reg)[-1], "bordes")
        self.assertIs(b["aviso_inicio"], True)
        self.assertIs(b["aviso_fin"], False)
        self.assertEqual(b["modo"], "aviso")
        self.assertEqual(b["umbrales"], "models/senas_cortadas_2026-09-22/umbral_bordes_candidato.json")
        self.assertEqual(b["K"], 4)
        self.assertAlmostEqual(b["fps_ventana"], FPS, places=3)
        self.assertGreater(b["v_pose_ini_por_s"], 1.7614)
        self.assertLess(b["h_ini"], 0.974)
        self.assertIsInstance(b["costo_ns"], int)
        self.assertGreaterEqual(b["costo_ns"], 0)
        self.assertLess(b["costo_ns"], 50_000_000)                     # cota laxa: medida real en docs
        linea = leer_jsonl(tr.registro_path)[-1]                       # serializable, mismas claves
        self.assertEqual(linea["bordes"], b)
        self.assertEqual(tr.avisos_bordes, {"inicio": 1, "fin": 0})
        self.assertEqual(set(reg["contadores"]), set(ESTADOS))         # ningún estado nuevo
        self.assertEqual(reg["contadores"]["aceptada"], 1)

    def test_el_aviso_no_cambia_estado_ni_marcas(self):
        """Mismo reloj falso y mismos fotogramas con «aviso» y con «off»: todo igual salvo «bordes»."""
        for sena, conf in (((0, 48), 0.9), ((12, 60), 0.9), ((0, 60), 0.3)):
            regs = {}
            for modo in ("aviso", "off"):
                tr = traductor(self.tmp, ventana(sena=sena), bordes=modo, modelo=ModeloFalso(confianza=conf))
                regs[modo] = correr(tr, 60)
            a, o = regs["aviso"], regs["off"]
            with self.subTest(sena=sena):
                self.assertEqual(o["bordes"], {"modo": "off"})
                self.assertTrue(a["bordes"]["aviso_inicio"] or a["bordes"]["aviso_fin"])
                distintos = {k for k in a if a[k] != o[k]} - {"sesion", "fecha_hora", "bordes"}
                self.assertEqual(distintos, set())
                self.assertEqual(list(a), list(o))
        self.assertEqual(regs["aviso"]["estado"], "rechazada_confianza")   # el último caso: el umbral decide

    def test_off_no_calcula_nada(self):
        with mock.patch("src.translator.medir_bordes") as medir, mock.patch("src.translator.evaluar_bordes") as ev:
            tr = traductor(self.tmp, ventana(sena=(0, 48)), bordes="off")
            reg = correr(tr, 60)
        medir.assert_not_called()
        ev.assert_not_called()
        self.assertIsNone(tr.umbrales_bordes)
        self.assertEqual(reg["bordes"], {"modo": "off"})
        self.assertEqual(tr.avisos_bordes, {"inicio": 0, "fin": 0})
        with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
            traductor(self.tmp, ventana(), bordes="rechazar")

    def test_reposo_correcto_no_avisa(self):
        tr = traductor(self.tmp, ventana(sena=(12, 48)))
        b = correr(tr, 60)["bordes"]
        self.assertEqual((b["aviso_inicio"], b["aviso_fin"]), (False, False))
        self.assertEqual(tr.avisos_bordes, {"inicio": 0, "fin": 0})

    def test_rechazos_tambien_lo_llevan_y_abortada_queda_en_null(self):
        # rechazada_manos: sin manos en toda la ventana; la postura sí está, así que hay medida.
        fot = [Extraccion(keypoints=np.concatenate([f.keypoints[IDX_POSE], np.zeros(126, np.float32)]),
                          pose_detectada=True, n_manos=0, rostro=None) for f in ventana(sena=(0, 60))]
        reg = correr(traductor(self.tmp, fot), 60)
        self.assertEqual(reg["estado"], "rechazada_manos")
        self.assertIs(reg["bordes"]["aviso_inicio"], True)
        # Abortada con X: medidas y banderas en null, y no cuenta como aviso.
        tr = traductor(self.tmp, ventana(sena=(0, 60)))
        tr.abrir_ventana()
        for i in range(10):
            tr.procesar_fotograma(np.zeros((4, 4, 3), np.uint8), c_i=tr.ahora() + i * 33_000_000)
        reg = tr.abortar_ventana()
        self.assertEqual(reg["estado"], "rechazada_incompleta")
        self.assertEqual(list(reg["bordes"]), CLAVES_BORDES)
        self.assertTrue(all(reg["bordes"][k] is None for k in ("aviso_inicio", "aviso_fin", "K") + bv.MEDIDAS))
        self.assertEqual(tr.avisos_bordes, {"inicio": 0, "fin": 0})
        # Menos fotogramas que el mínimo (pero ≥ 2): rechazada_incompleta, con medida.
        tr = traductor(self.tmp, ventana(n=20, sena=(0, 20)), min_frames=15, ventana_max_seg=0.2)
        reg = correr(tr, 20)                                           # se cierra por tiempo con 7 fotogramas
        self.assertEqual((reg["estado"], reg["cierre"], reg["N"]), ("rechazada_incompleta", "tiempo", 7))
        self.assertIs(reg["bordes"]["aviso_inicio"], True)

    def test_igual_con_la_red_heredada_y_con_el_modelo_final(self):
        """La medida solo usa la postura: es la misma con «mascara» (red heredada), con «imputacion» (modelo
        final) y con una representación con postura sin procedencia (ancho 450)."""
        fot = ventana(sena=(0, 48))
        regs = {}
        for nombre, kw in (("heredada", dict(procedencia=procedencia("mascara"))),
                           ("final", dict(procedencia=procedencia("imputacion"))),
                           ("manos_postura", dict(procedencia=None, modelo=ModeloFalso(450)))):
            tr = traductor(self.tmp, fot, **kw)
            regs[nombre] = correr(tr, 60)
        self.assertEqual(regs["heredada"]["mano_ausente"], "mascara")
        self.assertEqual(regs["final"]["mano_ausente"], "imputacion")
        sin_costo = {n: {k: v for k, v in r["bordes"].items() if k != "costo_ns"} for n, r in regs.items()}
        self.assertEqual(sin_costo["heredada"], sin_costo["final"])
        self.assertEqual(sin_costo["heredada"], sin_costo["manos_postura"])
        self.assertTrue(all(r["estado"] == "aceptada" for r in regs.values()))

    def test_umbral_bordes_propio(self):
        datos = json.loads(CANDIDATO.read_text(encoding="utf-8"))
        datos["h"] = {"ini": -10.0, "fin": -10.0}
        datos["v_pose_por_s"] = {"ini": 1e6, "fin": 1e6}                 # nada avisa
        ruta = Path(self.tmp) / "umbrales_propios.json"
        ruta.write_text(json.dumps(datos), encoding="utf-8")
        tr = traductor(self.tmp, ventana(sena=(0, 60)), umbral_bordes=ruta)
        b = correr(tr, 60)["bordes"]
        self.assertEqual((b["aviso_inicio"], b["aviso_fin"]), (False, False))
        self.assertEqual(Path(b["umbrales"]).name, "umbrales_propios.json")


@unittest.skipUnless(MODELO_FINAL.exists() and MODELO_HEREDADO.exists(), "modelos reales no disponibles")
class TestAvisoConLosModelosReales(unittest.TestCase):
    """Carga real (procedencia y umbral leídos junto al .keras, como en realtime.py); fotogramas sintéticos."""

    def test_las_dos_redes(self):
        tmp = tempfile.mkdtemp(prefix="tesis_bordes_")
        fot = ventana(sena=(0, 48))
        for ruta, regla in ((MODELO_FINAL, "imputacion"), (MODELO_HEREDADO, "mascara")):
            with self.subTest(modelo=ruta.parent.name):
                with redirect_stdout(io.StringIO()):
                    tr = LSCTranslator(model_path=str(ruta), extractor=ExtractorLista(fot), tts=TTSFalso(),
                                       registro_dir=Path(tmp) / "registros", ventana_frames=60, min_frames=15,
                                       calentar=False)
                self.assertEqual(tr.mano_ausente, regla)
                con = correr(tr, 60)
                tr.bordes_modo = "off"
                sin = correr(tr, 60)
                self.assertIs(con["bordes"]["aviso_inicio"], True)
                self.assertEqual(sin["bordes"], {"modo": "off"})
                for k in ("estado", "id_clase", "clase", "confianza", "top3", "enunciado"):
                    self.assertEqual(con[k], sin[k], k)
                self.assertIn(con["estado"], ("aceptada", "rechazada_confianza"))   # se infirió


# ---------------------------------------------------------------------------
# realtime.py: predeterminados, pantalla y consola
# ---------------------------------------------------------------------------

class TestRealtime(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src import realtime              # importa cv2; ninguna prueba abre la cámara
        cls.rt = realtime

    def test_predeterminados(self):
        args = self.rt.crear_parser().parse_args([])
        self.assertEqual(Path(args.model_path), MODELO_POR_DEFECTO)
        self.assertEqual(Path(args.model_path).relative_to(ROOT_DIR).as_posix(), "models/tcn_imputacion/modelo.keras")
        self.assertEqual(args.conversor, "inventario")
        self.assertEqual((args.ancho, args.alto), (1280, 720))
        self.assertEqual((args.bordes, args.umbral_bordes), ("aviso", None))
        self.assertIsNone(args.umbral)                       # el umbral sale de <modelo>.umbral.json
        self.assertIsNone(args.mano_ausente)                 # la regla sale de la procedencia
        # Lo heredado sigue disponible.
        args = self.rt.crear_parser().parse_args(["--model-path", str(MODELO_HEREDADO), "--conversor", "heredado",
                                                  "--bordes", "off", "--ancho", "640", "--alto", "480"])
        self.assertEqual((Path(args.model_path), args.conversor, args.bordes, args.ancho, args.alto),
                         (MODELO_HEREDADO, "heredado", "off", 640, 480))

    def test_texto_del_aviso(self):
        t = self.rt.texto_aviso_bordes
        self.assertIsNone(t({"bordes": {"aviso_inicio": False, "aviso_fin": False}}))
        self.assertIsNone(t({"bordes": {"aviso_inicio": None, "aviso_fin": None}}))
        self.assertIsNone(t({"bordes": {"modo": "off"}}))
        self.assertIsNone(t({}))
        self.assertIsNone(t(None))
        self.assertIn("empezar antes", t({"bordes": {"aviso_inicio": True, "aviso_fin": False}}))
        self.assertIn("no terminar", t({"bordes": {"aviso_inicio": False, "aviso_fin": True}}))
        ambos = t({"bordes": {"aviso_inicio": True, "aviso_fin": True}})
        self.assertIn("empezar antes", ambos)
        self.assertIn("no terminar", ambos)
        for s in (ambos, t({"bordes": {"aviso_inicio": True}})):
            self.assertTrue(s.isascii(), s)                 # la fuente de OpenCV no dibuja tildes ni eñes

    def test_consola_y_contadores(self):
        tmp = tempfile.mkdtemp(prefix="tesis_bordes_")
        tr = traductor(tmp, ventana(sena=(0, 48)))
        reg = correr(tr, 60)
        self.assertIn("[aviso borde: inicio]", self.rt._resumen_consola(reg))
        tr2 = traductor(tmp, ventana(sena=(12, 48)))
        self.assertNotIn("aviso borde", self.rt._resumen_consola(correr(tr2, 60)))
        salida = io.StringIO()
        with redirect_stdout(salida):
            self.rt.imprimir_contadores_sesion(tr)
        self.assertIn("  avisos de borde        inicio 1 | fin 0 (no rechazan)", salida.getvalue().split("\n"))
        tr3 = traductor(tmp, ventana(), bordes="off")
        salida = io.StringIO()
        with redirect_stdout(salida):
            self.rt.imprimir_contadores_sesion(tr3)
        self.assertNotIn("avisos de borde", salida.getvalue())

    def test_dibujar_ui_con_aviso_no_falla(self):
        tmp = tempfile.mkdtemp(prefix="tesis_bordes_")
        tr = traductor(tmp, ventana(sena=(0, 48)))
        correr(tr, 60)
        res = tr.procesar_fotograma(np.zeros((4, 4, 3), np.uint8))
        for w, h in ((1280, 720), (640, 480)):
            frame = np.zeros((h, w, 3), np.uint8)
            self.rt.dibujar_ui(frame, tr, res, "", 30.0)
            self.assertGreater(int(frame.sum()), 0)
        tr.abrir_ventana()                                   # la línea de la ventana abierta recuerda el reposo
        self.rt.dibujar_ui(np.zeros((720, 1280, 3), np.uint8), tr, res, "", 30.0)

    def test_aviso_de_forma_de_la_captura(self):
        f = self.rt.aviso_forma_captura
        for ancho, alto in ((1280, 720), (1920, 1080), (854, 480), (0, 0)):
            self.assertIsNone(f(ancho, alto), (ancho, alto))
        aviso = f(640, 480)
        self.assertIn("640x480", aviso)
        self.assertIn("N-57", aviso)
        self.assertIn("0.75", aviso)


if __name__ == "__main__":
    unittest.main()
