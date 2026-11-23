"""
Prototipo en vivo: ventana fija activada por tecla, estados de error e
instrumentación de latencia (anteproyecto §2.2.6 y §2.2.7).

Extractor: HolisticLandmarker de MediaPipe Tasks (D-05), el mismo con las mismas opciones
que la re-extracción del corpus; ver src/translator.py.

Flujo por ensayo:
  1. En reposo el bucle captura y extrae puntos en cada fotograma (para mostrar
     la pose, las manos y el óvalo del rostro y para medir la tasa real del bucle),
     pero NO infiere.
  2. ESPACIO sella t0 y abre la ventana. La ventana captura N fotogramas
     (--ventana-frames; por defecto round(--ventana-seg × tasa medida)) o se cierra
     por tope de tiempo (--ventana-max-seg).
  3. Al cerrarse: normalización → columnas de la representación del modelo (D-08) →
     remuestreo a 30 pasos (resample_sequence del entrenamiento) → velocidades si el
     modelo las espera (add_velocity del entrenamiento) → UNA inferencia → GrammarMapper → orden de reproducción (t5).
  4. Estados: aceptada · rechazada_confianza (< --umbral: «seña no reconocida»,
     sin audio, repita) · rechazada_manos (proporción sin manos o sin pose
     > --max-sin-manos: no se infiere) · rechazada_incompleta (abortada con X o
     menos de --min-frames fotogramas). Contadores en pantalla y en el JSONL.
  5. Un registro por ensayo en registros/latencia_<fecha>.jsonl. Nunca se escribe
     ningún fotograma ni video a disco (§1.5).
  6. Voz: si el contador acumulado de errores de reproducción (tts_errores_reproduccion,
     N-37) sube de un registro al siguiente, se avisa POR CONSOLA en ese momento; el
     overlay del video no cambia. Al cerrar, el bloque de contadores muestra los
     errores y las locuciones descartadas de LSCTranslator.estado_voz_al_cerrar.

Modo oración (--modo-oracion o tecla M): cada ventana aceptada añade su glosa;
ENTER es el fin de turno: convierte la lista con GrammarMapper y sintetiza.

Conversor de glosas (--conversor, actividad 3.a): «heredado» (por defecto) es GrammarMapper, como
siempre; «reglas» usa src/conversor_glosas.py y «inventario» el mismo en modo estricto (solo las glosas
solas y las secuencias previstas de docs/CONVERSOR_3a_ESPECIFICACION.md, §4.3). Ninguna de sus reglas está
revisada todavía por la persona señante.

Uso:
    python src/realtime.py                                   # modelo canónico vigente (manos, N-30), voz local
    python src/realtime.py --model-path models/canonico_holistic_mascara/modelo_canonico.keras   # manos + postura
    python src/realtime.py --tts none                        # sin voz (t5 = t4 + emisión vacía)
    python src/realtime.py --umbral 0.7 --ventana-seg 3.0 --modo-oracion
    python src/realtime.py --ventana-frames 60               # N fijo, sin medir la tasa
    python src/realtime.py --modo-oracion --conversor inventario --genero-primera f   # conversor 3.a estricto
    python src/realtime.py --model-path models/tcn_imputacion/modelo.keras --ancho 1280 --alto 720
                                     # opción EXPERIMENTAL (N-63): TCN + imputación + 252; la regla
                                     # «imputacion» se toma sola de su procedencia (docs/MODELO_TCN_IMPUTACION.md)

Teclas:
    ESPACIO  abrir la ventana (sella t0)
    X        abortar la ventana abierta (se cuenta como rechazada_incompleta)
    ENTER    fin de turno: convertir y sintetizar las glosas acumuladas (modo oración)
    M        alternar modo aislado / modo oración
    R        descartar las glosas acumuladas sin sintetizar
    Q / ESC  salir
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.translator import LSCTranslator, ESTADOS, MODELO_POR_DEFECTO, CONVERSORES, construir_conversor

# ---------------------------------------------------------------------------
# Dibujo de puntos característicos (solo en pantalla; nunca a disco)
# ---------------------------------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17),
]
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
]
# Óvalo del rostro: 36 índices de la malla de 468 puntos, en orden (contorno cerrado).
# Solo para la vista: el rostro no entra en la inferencia (D-05).
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
FACE_OVAL_CONNECTIONS = [(k, (k + 1) % len(FACE_OVAL_IDX)) for k in range(len(FACE_OVAL_IDX))]


def _dibujar_puntos(frame, landmarks, conexiones, color, radio):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for p in pts:
        cv2.circle(frame, p, radio, color, -1)
    for i, j in conexiones:
        if i < len(pts) and j < len(pts):
            cv2.line(frame, pts[i], pts[j], color, 1)


def dibujar_extraccion(frame, extraccion):
    """Dibuja pose, manos y el óvalo del rostro desde el HolisticLandmarkerResult crudo
    (listas planas de 33 / 21 / 21 / 478 puntos; vacías si no hubo detección)."""
    hr = getattr(extraccion, "holistic_result", None)
    if hr is None:
        return
    if hr.pose_landmarks:
        _dibujar_puntos(frame, hr.pose_landmarks, POSE_CONNECTIONS, (80, 200, 120), 2)
    # left_hand_landmarks es la mano izquierda DE LA PERSONA (naranja); la derecha, azul.
    if hr.left_hand_landmarks:
        _dibujar_puntos(frame, hr.left_hand_landmarks, HAND_CONNECTIONS, (245, 117, 66), 3)
    if hr.right_hand_landmarks:
        _dibujar_puntos(frame, hr.right_hand_landmarks, HAND_CONNECTIONS, (66, 117, 245), 3)
    if hr.face_landmarks and len(hr.face_landmarks) > max(FACE_OVAL_IDX):
        ovalo = [hr.face_landmarks[i] for i in FACE_OVAL_IDX]
        _dibujar_puntos(frame, ovalo, FACE_OVAL_CONNECTIONS, (200, 200, 200), 1)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

ROJO = (40, 40, 230)
VERDE = (40, 200, 40)
AMBAR = (0, 170, 255)
GRIS = (170, 170, 170)
BLANCO = (255, 255, 255)

MENSAJE_ESTADO = {
    "aceptada": "ACEPTADA",
    "rechazada_confianza": "SENA NO RECONOCIDA: repita la sena",
    "rechazada_manos": "MANOS NO DETECTADAS: no se infiere",
    "rechazada_incompleta": "SECUENCIA INCOMPLETA: descartada",
}


def _texto(frame, s, xy, escala=0.55, color=BLANCO, grosor=1):
    cv2.putText(frame, s, xy, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)


def _ms(ns):
    return f"{ns / 1e6:.0f} ms" if ns is not None else "-"


def dibujar_ui(frame, tr: LSCTranslator, res, aviso: str, fps_nominal: float):
    h, w = frame.shape[:2]
    ext = res.extraccion

    # Barra de progreso de la ventana
    cv2.rectangle(frame, (10, 8), (w - 10, 18), (50, 50, 50), -1)
    if tr.ventana_abierta or res.registro is not None:
        cv2.rectangle(frame, (10, 8), (10 + int((w - 20) * res.progreso), 18), AMBAR, -1)

    # Estado del bucle
    fps = tr.fps_medida()
    if tr.ventana_abierta:
        ens = tr.ensayo
        linea = f"VENTANA {ens.n}/{ens.ventana_frames}  (ESPACIO abrio t0; X aborta)"
        color = AMBAR
    else:
        linea = "REPOSO: ESPACIO abre la ventana"
        color = GRIS
    _texto(frame, linea, (10, 40), 0.6, color, 2)
    _texto(frame, (f"bucle {fps:.1f} fps" if fps else "bucle: midiendo...")
           + (f" (cam {fps_nominal:.0f})" if fps_nominal else ""),
           (w - 230, 40), 0.5, GRIS)

    # Deteccion de manos y pose: indicacion visual continua (§2.2.6)
    if not ext.pose_detectada:
        _texto(frame, "POSE NO DETECTADA", (10, 65), 0.6, ROJO, 2)
    elif ext.n_manos == 0:
        _texto(frame, "MANOS NO DETECTADAS / FUERA DE CUADRO", (10, 65), 0.6, ROJO, 2)
    else:
        _texto(frame, f"manos detectadas: {ext.n_manos}", (10, 65), 0.55, VERDE)

    # Modo y turno
    modo = "ORACION (ENTER = fin de turno)" if tr.modo_oracion else "AISLADO"
    _texto(frame, f"modo: {modo}", (10, 88), 0.5, GRIS)
    if tr.modo_oracion:
        glosas = " + ".join(tr.glosas_turno) if tr.glosas_turno else "(ninguna)"
        _texto(frame, f"glosas del turno: {glosas}", (10, 108), 0.5, (180, 220, 180))
        if tr.enunciado_turno:
            _texto(frame, f"ultimo turno: {tr.enunciado_turno}", (10, 128), 0.5, (200, 200, 255))

    # Contadores de estados (§2.2.6: distinguir errores de rechazos deliberados)
    c = tr.contadores
    _texto(frame, f"aceptadas {c['aceptada']}  |  rech. confianza {c['rechazada_confianza']}  |  "
                  f"rech. manos {c['rechazada_manos']}  |  rech. incompleta {c['rechazada_incompleta']}",
           (10, h - 95), 0.5, GRIS)

    # Ultimo ensayo
    u = tr.ultimo
    if u is not None:
        estado = u["estado"]
        color = VERDE if estado == "aceptada" else ROJO
        cabecera = MENSAJE_ESTADO[estado]
        if u["clase"] is not None:
            cabecera += f"   {u['clase']} ({u['confianza']:.0%})"
        cv2.rectangle(frame, (10, h - 82), (w - 10, h - 52), (30, 30, 30), -1)
        _texto(frame, cabecera, (15, h - 60), 0.65, color, 2)
        if u["top3"]:
            top = "  ".join(f"{t['glosa']} {t['p']:.0%}" for t in u["top3"])
            _texto(frame, f"top-3: {top}", (15, h - 38), 0.45, GRIS)
        lat = (f"W {_ms(u['W_ns'])}  L_total {_ms(u['L_total_ns'])}  L_resp {_ms(u['L_resp_ns'])}  "
               f"N={u['N']}  {u['tasa_efectiva_fps']:.1f} fps" if u["tasa_efectiva_fps"] else
               f"W {_ms(u['W_ns'])}  N={u['N']}")
        _texto(frame, lat, (15, h - 20), 0.45, GRIS)
        if u["enunciado"]:
            _texto(frame, f"Espanol: {u['enunciado']}", (w // 2, h - 38), 0.5, (200, 200, 255))
    else:
        _texto(frame, "Espanol: (abra una ventana con ESPACIO y ejecute la sena)", (15, h - 60), 0.5, GRIS)

    if aviso:
        _texto(frame, aviso, (10, h - 110), 0.5, AMBAR)

    _texto(frame, "ESPACIO ventana | X abortar | ENTER fin de turno | M modo | R reiniciar | Q salir",
           (10, h - 5), 0.42, GRIS)


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def _resumen_consola(reg: dict) -> str:
    base = (f"[ensayo {reg['id_ensayo']:3d}] {reg['estado']:<21s} N={reg['N']:3d} cierre={reg['cierre']:<8s} "
            f"W={_ms(reg['W_ns'])}")
    if reg["clase"] is not None:
        base += f"  {reg['clase']} ({reg['confianza']:.2f})"
    if reg["L_total_ns"] is not None:
        e = reg["etapas_ns"]
        base += (f"  L_total={_ms(reg['L_total_ns'])} L_resp={_ms(reg['L_resp_ns'])}"
                 f" [extr {_ms(e['extraccion'])} prep {_ms(e['preprocesamiento'])} inf {_ms(e['inferencia'])}"
                 f" conv {_ms(e['conversion'])} sint {_ms(e['sintesis'])}]")
    if reg["motivo"]:
        base += f"  ({reg['motivo']})"
    return base


class AvisoErroresVoz:
    """Avisa por consola cuando sube el contador acumulado de errores de reproducción de voz.

    Cada registro de ensayo y de turno lleva tts_errores_reproduccion (N-37): locuciones que NO
    sonaron, acumuladas en la sesión, leídas después de sellar t5. Empieza en 0 y no baja. Como la
    locución suena en segundo plano, el fallo del ensayo k suele aparecer en el registro k+1; sin este
    aviso el evaluador solo se entera al cerrar la sesión (la línea [AVISO TTS] de close()).

    Se avisa UNA vez por subida, por consola y nunca en el overlay del video (§2.2.6: el dibujado no
    cambia). null («el motor no informa», que no es «cero errores») no avisa ni mueve la referencia.
    No lee nada del traductor: solo el registro que este ya devolvió.
    """

    def __init__(self):
        self.ultimo = 0          # al abrir la sesión ninguna locución ha sonado todavía

    def revisar(self, registro) -> str | None:
        """Devuelve la línea de aviso si el contador subió en este registro; si no, None."""
        if not isinstance(registro, dict):
            return None
        valor = registro.get("tts_errores_reproduccion")
        if not isinstance(valor, int) or isinstance(valor, bool) or valor <= self.ultimo:
            return None
        nuevos = valor - self.ultimo
        self.ultimo = valor
        mensaje = registro.get("tts_ultimo_error_reproduccion")
        return (f"  [AVISO TTS] {nuevos} locucion(es) no sonaron por error de reproduccion "
                f"(total {valor} en la sesion)" + (f": {mensaje}" if mensaje else ""))


def _avisar_voz(avisos: AvisoErroresVoz, registro) -> None:
    linea = avisos.revisar(registro)
    if linea:
        print(linea)


def _voz_o_sin_dato(valor) -> str:
    """None = el motor no informa (no es cero errores); se dice así en pantalla."""
    return str(valor) if isinstance(valor, int) and not isinstance(valor, bool) else "sin dato"


def imprimir_contadores_sesion(tr) -> None:
    """Bloque final «Contadores de la sesion».

    Además de los contadores de estado, muestra el estado de la voz que deja
    LSCTranslator.estado_voz_al_cerrar (N-36 y N-37): errores de reproducción y locuciones descartadas
    al cerrar. Lectura defensiva con getattr: si el traductor no deja ese atributo (una versión
    anterior, o un doble de prueba) no se imprime nada nuevo y el resto del bloque queda igual.
    """
    print("\nContadores de la sesion:")
    for estado in ESTADOS:
        print(f"  {estado:<22s} {tr.contadores[estado]}")
    print(f"  ensayos {tr.n_ensayos} | turnos {tr.n_turnos}")
    voz = getattr(tr, "estado_voz_al_cerrar", None)
    if isinstance(voz, dict):
        print(f"  {'errores de voz':<22s} {_voz_o_sin_dato(voz.get('errores_reproduccion'))}")
        print(f"  {'locuciones descartadas':<22s} {_voz_o_sin_dato(voz.get('locuciones_descartadas_al_cerrar'))}")
        ultimo = voz.get("ultimo_error_reproduccion")
        if ultimo:
            print(f"  ultimo error de voz: {ultimo}")
    if tr.registro_activo:
        print(f"Registro: {tr.registro_path}")
        print(f"Resumen: python src/latencia_resumen.py \"{tr.registro_path}\"")


def crear_parser() -> argparse.ArgumentParser:
    """Opciones de la línea de órdenes (separadas de main para poder probarlas sin cámara)."""
    parser = argparse.ArgumentParser(
        description="Prototipo LSC en vivo: ventana fija por tecla, estados de error e instrumentacion")
    parser.add_argument("--camera", type=int, default=0, help="indice de la camara (default 0)")
    parser.add_argument("--model-path", type=str, default=str(MODELO_POR_DEFECTO),
                        help="ruta al .keras (defecto: el canónico vigente); la representación y la regla de "
                             "mano ausente se leen de <modelo>.procedencia.json o, sin ella, se deducen del ancho")
    parser.add_argument("--umbral", "--threshold", dest="umbral", type=float, default=None,
                        help="confianza minima para aceptar; parametro de operacion, no meta. Por defecto, el "
                             "elegido por validacion para el modelo (<modelo>.umbral.json, D-09) o 0.5")
    parser.add_argument("--ventana-frames", type=int, default=0,
                        help="N fotogramas que cierran la ventana; 0 = round(ventana-seg x tasa medida) (default 0)")
    parser.add_argument("--ventana-seg", type=float, default=2.5,
                        help="duracion objetivo de la ventana en s (default 2.5: clips de LSC50 media 2.57 s, "
                             "mediana 2.58 s, p90 3.29 s; 47%% de los clips <= 2.5 s, 79%% <= 3.0 s)")
    parser.add_argument("--ventana-max-seg", type=float, default=None,
                        help="tope de seguridad: cierra por tiempo aunque falten fotogramas (default 2 x ventana-seg)")
    parser.add_argument("--min-frames", type=int, default=15,
                        help="ventanas con menos fotogramas se descartan como incompletas (default 15)")
    parser.add_argument("--mano-ausente", choices=["ceros", "descartar", "mascara", "imputacion"], default=None,
                        help="D-07: tratamiento del fotograma sin una mano; por defecto, el del corpus del "
                             "modelo (procedencia). Indicarlo solo para experimentos. «imputacion» (N-63, "
                             "experimental) es la del modelo models/tcn_imputacion/modelo.keras")
    parser.add_argument("--max-sin-manos", type=float, default=0.5,
                        help="proporcion maxima de fotogramas sin manos (o sin pose); si se supera no se infiere (default 0.5)")
    parser.add_argument("--modo-oracion", action="store_true",
                        help="acumular glosas por ventana y convertir/sintetizar con ENTER")
    parser.add_argument("--tts", choices=["offline", "edge", "gtts", "none"], default="offline",
                        help="motor de voz: offline (default; local, pyttsx3, prioridad de §2.2.5), "
                             "edge/gtts (en linea, de contraste), none (sin audio)")
    parser.add_argument("--tts-voice", type=str, default=None, help="voz especifica del motor")
    parser.add_argument("--registro-dir", type=str, default=str(ROOT_DIR / "registros"),
                        help="carpeta del JSONL de latencia (default registros/)")
    parser.add_argument("--sin-registro", action="store_true", help="no escribir el JSONL")
    parser.add_argument("--no-draw", action="store_true", help="no dibujar puntos (mas rapido)")
    parser.add_argument("--ancho", type=int, default=640)
    parser.add_argument("--alto", type=int, default=480)
    # Actividad 3.a (docs/CONVERSOR_3a_ESPECIFICACION.md, §7.2): conversor de glosas. El valor por defecto
    # sigue siendo el heredado (GrammarMapper): sin esta opción el prototipo se comporta como antes.
    parser.add_argument("--conversor", choices=list(CONVERSORES), default="heredado",
                        help="conversor de glosas: heredado (default; src/grammar_mapper.py), reglas "
                             "(src/conversor_glosas.py) o inventario (reglas en modo estricto: solo las glosas "
                             "solas y las secuencias previstas del inventario). Con reglas/inventario el JSONL gana la "
                             "clave «conversion» con la traza de la conversion")
    parser.add_argument("--genero-primera", choices=["m", "f"], default=None,
                        help="genero de quien habla (YO) para la concordancia; solo con --conversor reglas o "
                             "inventario (default m, registrado como supuesto)")
    parser.add_argument("--genero-segunda", choices=["m", "f"], default=None,
                        help="genero de quien escucha (TU); solo con --conversor reglas o inventario (default m)")
    return parser


def conversor_desde_args(args, parser: argparse.ArgumentParser):
    """Construye el conversor pedido (None = heredado). Con --conversor heredado, dar --genero-primera o
    --genero-segunda termina con un error de argumentos, para que nadie crea que se aplicaron (§7.2)."""
    try:
        return construir_conversor(args.conversor, args.genero_primera, args.genero_segunda)
    except ValueError as exc:
        parser.error(str(exc))


def main():
    parser = crear_parser()
    args = parser.parse_args()
    conversor = conversor_desde_args(args, parser)

    tr = LSCTranslator(
        model_path=args.model_path,
        tts_mode=args.tts,
        tts_voice=args.tts_voice,
        umbral=args.umbral,
        ventana_frames=args.ventana_frames,
        ventana_seg=args.ventana_seg,
        ventana_max_seg=args.ventana_max_seg,
        min_frames=args.min_frames,
        max_sin_manos=args.max_sin_manos,
        mano_ausente=args.mano_ausente,
        modo_oracion=args.modo_oracion,
        registro_dir=args.registro_dir,
        registro=not args.sin_registro,
        grammar_mapper=conversor,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la camara {args.camera}")
        tr.close()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.alto)
    fps_nominal = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    print("\n" + "=" * 64)
    print("  LSC - prototipo en vivo (ventana fija por tecla)")
    print(f"  Modelo: {tr.model_path}  ({tr.n_rasgos} rasgos, velocidades={'si' if tr.usa_velocidades else 'no'})")
    print(f"  Representacion (D-08): {tr.representacion}  | procedencia: {tr.procedencia_ruta or '(no declarada; deducida del ancho)'}")
    print(f"  Extractor: {tr.extractor_desc}")
    print(f"  Camara {args.camera}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
          f", fps nominal {fps_nominal:.0f} (la tasa real del bucle se mide en pantalla)")
    print(f"  Ventana: {args.ventana_frames or 'auto'} fotogramas / objetivo {args.ventana_seg} s / "
          f"tope {tr.ventana_max_seg} s / minimo {args.min_frames}")
    print(f"  Umbral {tr.umbral:.2f} ({tr.umbral_origen}) | max sin manos {args.max_sin_manos} | modo {'oracion' if args.modo_oracion else 'aislado'}")
    print(f"  Mano ausente (D-07): {tr.mano_ausente}  (corpus del modelo: {tr.mano_ausente_modelo or 'no declarado'})")
    print(f"  TTS: {tr.tts_modo}")
    print(f"  Conversor de glosas: {args.conversor}"
          + ("" if conversor is None else f" (genero yo={conversor.genero_primera}, tu={conversor.genero_segunda}; "
                                          "sin revisar por la persona senante)"))
    print(f"  Registro: {tr.registro_path if tr.registro_activo else '(desactivado)'}")
    print("  Teclas: ESPACIO ventana | X abortar | ENTER fin de turno | M modo | R reiniciar | Q salir")
    print("=" * 64 + "\n")

    aviso = ""
    avisos_voz = AvisoErroresVoz()
    try:
        while True:
            ret, frame = cap.read()
            c_i = tr.ahora()                         # c_i: llegada del fotograma a la aplicacion
            if not ret:
                print("[ERROR] No se pudo leer de la camara")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = tr.procesar_fotograma(rgb, c_i=c_i)
            if res.registro is not None:
                print(_resumen_consola(res.registro))
                _avisar_voz(avisos_voz, res.registro)
                aviso = ""

            if not args.no_draw:
                dibujar_extraccion(frame, res.extraccion)
            dibujar_ui(frame, tr, res, aviso, fps_nominal)
            cv2.imshow("LSC - prototipo (ventana fija por tecla)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:                            # ESPACIO: sella t0 en el manejador
                t0 = tr.ahora()
                if tr.abrir_ventana(t0) is None:
                    aviso = "ya hay una ventana abierta"
                else:
                    aviso = ""
            elif key in (ord("x"), ord("X")):
                reg = tr.abortar_ventana()
                if reg is not None:
                    print(_resumen_consola(reg))
                    _avisar_voz(avisos_voz, reg)
                aviso = "ventana abortada" if reg is not None else "no hay ventana abierta"
            elif key in (13, 10):                    # ENTER: fin de turno
                t_tecla = tr.ahora()
                reg = tr.fin_de_turno(t_tecla)
                if reg is None:
                    aviso = "fin de turno sin glosas acumuladas"
                else:
                    aviso = ""
                    print(f"[turno {reg['id_turno']}] {reg['glosas']} -> \"{reg['enunciado']}\"  "
                          f"conv={_ms(reg['conversion_ns'])} sint={_ms(reg['sintesis_ns'])}")
                    _avisar_voz(avisos_voz, reg)
            elif key in (ord("m"), ord("M")):
                if tr.ventana_abierta:
                    aviso = "cierre la ventana antes de cambiar de modo"
                else:
                    tr.modo_oracion = not tr.modo_oracion
                    aviso = f"modo {'oracion' if tr.modo_oracion else 'aislado'}"
            elif key in (ord("r"), ord("R")):
                tr.reiniciar_turno()
                aviso = "glosas del turno descartadas"
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        if tr.ventana_abierta:
            tr.abortar_ventana()
        tr.close()
        cap.release()
        cv2.destroyAllWindows()

    imprimir_contadores_sesion(tr)


if __name__ == "__main__":
    main()
