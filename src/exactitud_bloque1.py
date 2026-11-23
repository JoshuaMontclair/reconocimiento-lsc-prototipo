"""
Exactitud en línea del Bloque 1 (actividad 4.a).

El prototipo escribe un JSONL por sesión (src/translator.py, _registro_ensayo) con lo que el sistema
reconoció, pero NO con la glosa que se pidió: el JSONL no guarda la glosa objetivo ni el bloque
(docs/PROTOCOLO_EXPERIMENTAL_4a.md, 9.1 y R14; PD-22 A). La exactitud en línea sale, por tanto, de cruzar
ese JSONL con una hoja que el operador llena a mano, ejecución a ejecución, cuya plantilla está en
docs/plantilla_hoja_registro_bloque1.csv. La clave de unión es (sesion, id_ensayo).

Lo que hace este guion:
  1. Lee una carpeta (o archivos, o comodines) de registros JSONL y la hoja CSV. La carpeta de registros
     es OBLIGATORIA y no tiene valor por defecto: nunca lee registros/ por su cuenta.
  2. Cruza por (sesion, id_ensayo) y reporta, sin descartar nada en silencio, los ensayos del JSONL sin
     fila en la hoja, las filas de la hoja sin ensayo, las filas duplicadas y las filas con datos que no
     se pueden interpretar (glosa desconocida, «valida» en blanco, «variante» no reconocida).
  3. Calcula top-1 y top-3, precisión, recall y F1 macro y la matriz de confusión sobre las ejecuciones
     válidas, más la tasa de rechazo por confianza, por manos y por secuencia incompleta, y lo hace POR
     DUPLICADO como exige el anteproyecto [MET 229-233]: sobre la totalidad de las ejecuciones y sobre las
     ejecuciones sin variante declarada. Añade el desglose por seña y por repetición (intento y ronda).
  4. Informa los ensayos con audio dudoso a partir de tts_errores_reproduccion (N-37, N-38).
  5. Escribe un JSON y un informe .md.

LO QUE ESTE GUION NO DECIDE. El protocolo deja abiertos dos parámetros que cambian la cifra:

  * PD-15, denominador de la exactitud. Se calculan las tres lecturas y se nombran:
      - «con_prediccion»      (PD-15 A): top-1 y top-3 sobre las ejecuciones que llegaron al modelo
                              (aceptadas y rechazadas por confianza), como en el nivel offline.
      - «aceptadas»           (PD-15 A, complemento tipo D-09): exactitud solo entre las aceptadas,
                              acompañada de la tasa de rechazo.
      - «todas_rechazo_error» (PD-15 B): aciertos de las aceptadas sobre TODAS las ejecuciones, es decir,
                              todo rechazo cuenta como error (cifra del sistema completo).
    PD-15 C es reportar A y B, que es lo que hace este guion; cuál es «la» cifra lo decide el director.

  * PD-14, política ante un rechazo (repetir o no, y qué intento cuenta). Se calcula cada conjunto con
    tres políticas: «todos» los intentos, solo el «primer_intento» de cada presentación y solo el
    «ultimo_intento». Las dos últimas necesitan la columna «presentacion» de la hoja.

  * PD-13, categorías de la declaración de variante. Si la hoja usa «no estoy segura», no está decidido si
    esa ejecución cuenta como variante: se reportan los dos conjuntos, «sin_variante_estricto»
    (solo variante = no) y «sin_variante_amplio» (no + no estoy segura).

  * E7 / N-38, qué ensayos quedan marcados por un fallo de voz. El contador tts_errores_reproduccion es
    acumulado y se lee después de t5, así que el fallo del ensayo k suele aparecer en el registro k+1
    (R20). Se dan las dos convenciones: la A, estrecha (desde el registro en que el contador ya subió;
    clave ensayos_desde_el_primer_error de src/latencia_resumen.py), y la B, que añade el ensayo
    inmediatamente anterior a cada aumento, que es el que probablemente no sonó. D-13 adoptó la B como
    definición de «audio dudoso» (es la clave audio_dudoso de src/latencia_resumen.py); la A se conserva
    para distinguirlas. No se excluye ninguno: E7 los conserva en la exactitud del Bloque 1 y pide
    contarlos; qué se hace con ellos (repetir, excluir o contar como incidencia) sigue abierto en PD-31.

Convenciones de cálculo, iguales a las del análisis offline (N-35, src/analisis_errores.py, del que se
reutilizan matriz_confusion y metricas_por_clase, y cuyo top_k replican las pruebas): F1 = 0 cuando la precisión es indefinida
(zero_division = 0 de sklearn); los promedios macro se toman sobre las señas presentadas (soporte > 0) y
se añade además el macro sobre las 50 clases, que es el que usa el nivel offline.

Uso:
    python src/exactitud_bloque1.py --registros registros/4b_bloque1 \\
           --hoja hojas/bloque1.csv --salida models/exactitud_bloque1
    python src/exactitud_bloque1.py --registros a.jsonl b.jsonl --hoja h.csv --json r.json --md r.md
    python src/exactitud_bloque1.py --plantilla hojas/bloque1.csv     # copia la plantilla en blanco
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analisis_errores import matriz_confusion, metricas_por_clase, N_CLASES  # noqa: E402
from src.grammar_mapper import SIGN_VOCABULARY  # noqa: E402
from src.latencia_resumen import ESTADOS, leer_registros  # noqa: E402

PLANTILLA = BASE_DIR / "docs" / "plantilla_hoja_registro_bloque1.csv"

# Columnas que la hoja debe traer sí o sí (la celda puede ir vacía; la COLUMNA no).
COLUMNAS_OBLIGATORIAS = ("sesion", "id_ensayo", "glosa_objetivo", "variante", "valida", "observacion")
COLUMNAS_OPCIONALES = ("presentacion", "ronda", "intento", "id_glosa_objetivo",
                       "referencia_mostrada", "estado_pantalla")

CONJUNTOS = ("totalidad", "sin_variante_estricto", "sin_variante_amplio")
POLITICAS = ("todos", "primer_intento", "ultimo_intento")
DENOMINADORES = ("con_prediccion", "aceptadas", "todas_rechazo_error")

ETIQUETA_CONJUNTO = {
    "totalidad": "totalidad de las ejecuciones",
    "sin_variante_estricto": "sin variante declarada (variante = no)",
    "sin_variante_amplio": "sin variante declarada (variante = no o «no estoy segura»)",
}
ETIQUETA_POLITICA = {
    "todos": "todos los intentos",
    "primer_intento": "solo el primer intento de cada presentación",
    "ultimo_intento": "solo el último intento de cada presentación",
}
ETIQUETA_DENOMINADOR = {
    "con_prediccion": "PD-15 A · ejecuciones con predicción (aceptadas + rechazadas por confianza)",
    "aceptadas": "PD-15 A, complemento D-09 · solo ejecuciones aceptadas",
    "todas_rechazo_error": "PD-15 B · todas las ejecuciones, todo rechazo cuenta como error",
}

# Campos de voz que escribe src/translator.py (N-37).
CLAVE_ERRORES_VOZ = "tts_errores_reproduccion"
CLAVE_ULTIMO_ERROR_VOZ = "tts_ultimo_error_reproduccion"

# Claves de configuración que la sección 3 del protocolo congela (criterio E2).
CLAVES_CONFIGURACION = ("modo", "modelo", "representacion", "mano_ausente", "extractor", "tts_modo")


# ---------------------------------------------------------------------------
# Normalización de textos de la hoja
# ---------------------------------------------------------------------------

def _sin_tildes(texto: str) -> str:
    """Minúsculas sin tildes, sin signos de puntuación y con los espacios colapsados."""
    base = unicodedata.normalize("NFD", str(texto)).lower()
    base = "".join(c for c in base if unicodedata.category(c) != "Mn")
    base = "".join(c if (c.isalnum() or c.isspace()) else " " for c in base)
    return " ".join(base.split())


# Vocabulario LSC50 indexado sin tildes ni signos: «como estas» encuentra «¿cómo estás?».
_GLOSA_A_ID = {_sin_tildes(nombre): i for i, nombre in SIGN_VOCABULARY.items()}

# Los tres conjuntos se comparan ya normalizados por _sin_tildes (minúsculas, sin tildes ni puntuación).
_SI = {"si", "s", "x", "1", "true", "verdadero"}
_NO = {"no", "n", "0", "false", "falso"}
_NO_SEGURA = {"no estoy segura", "no estoy seguro", "no segura", "no seguro", "ns", "no se", "no sabe"}


def _si_no(valor) -> bool | None:
    """si / no de una celda. None si la celda está vacía. Lanza ValueError si no se reconoce."""
    if valor is None:
        return None
    t = _sin_tildes(valor)
    if not t:
        return None
    if t in _SI:
        return True
    if t in _NO:
        return False
    raise ValueError(f"valor no reconocido para un campo si/no: {valor!r}")


def _variante(valor) -> str | None:
    """«si» / «no» / «no_segura» de la declaración de variante. None si la celda está vacía."""
    if valor is None:
        return None
    t = _sin_tildes(valor)
    if not t:
        return None
    if t in _NO_SEGURA:
        return "no_segura"
    if t in _SI:
        return "si"
    if t in _NO:
        return "no"
    raise ValueError(f"valor no reconocido para «variante»: {valor!r} "
                     f"(se espera si / no / no estoy segura)")


def _entero(valor) -> int | None:
    if valor is None:
        return None
    t = str(valor).strip()
    if not t:
        return None
    try:
        return int(t)
    except ValueError:
        raise ValueError(f"se esperaba un número entero y llegó {valor!r}")


def _id_ensayo(valor):
    """Normaliza el id_ensayo para que la hoja y el JSONL se emparejen aunque uno traiga «07»."""
    if valor is None:
        return None
    if isinstance(valor, bool):
        return str(valor)
    if isinstance(valor, int):
        return valor
    t = str(valor).strip()
    if not t:
        return None
    try:
        return int(t)
    except ValueError:
        return t


def _clave(sesion, id_ensayo) -> tuple:
    return ("" if sesion is None else str(sesion).strip(), _id_ensayo(id_ensayo))


def _texto_clave(clave: tuple) -> str:
    return f"{clave[0]}/{clave[1]}"


def _glosa_objetivo(fila: dict) -> int:
    """Identificador 0-49 de la seña pedida, de id_glosa_objetivo, de glosa_objetivo o de ambas."""
    por_id = _entero(fila.get("id_glosa_objetivo"))
    if por_id is not None and por_id not in SIGN_VOCABULARY:
        raise ValueError(f"id_glosa_objetivo {por_id} fuera de 0-{N_CLASES - 1}")
    nombre = (fila.get("glosa_objetivo") or "").strip()
    por_nombre = _GLOSA_A_ID.get(_sin_tildes(nombre)) if nombre else None
    if nombre and por_nombre is None:
        raise ValueError(f"glosa_objetivo {nombre!r} no está en el vocabulario LSC50 "
                         f"(src/grammar_mapper.py, SIGN_VOCABULARY)")
    if por_id is not None and por_nombre is not None and por_id != por_nombre:
        raise ValueError(f"id_glosa_objetivo {por_id} ({SIGN_VOCABULARY[por_id]}) y glosa_objetivo "
                         f"{nombre!r} ({SIGN_VOCABULARY[por_nombre]}) no coinciden")
    objetivo = por_id if por_id is not None else por_nombre
    if objetivo is None:
        raise ValueError("la fila no trae ni id_glosa_objetivo ni glosa_objetivo")
    return objetivo


# ---------------------------------------------------------------------------
# Lectura de la hoja
# ---------------------------------------------------------------------------

def leer_hoja(ruta) -> tuple[list[dict], list[dict]]:
    """Lee la hoja CSV del operador. Devuelve (filas interpretadas, filas con problema).

    Se saltan, sin ruido, las filas totalmente vacías y las que empiezan por «#» (la leyenda de la
    plantilla). Una fila con un valor que no se puede interpretar NO se descarta en silencio: va a la
    lista de problemas con su motivo, su número de línea y su contenido.
    """
    ruta = Path(ruta)
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        lector = csv.reader(f)
        try:
            cabecera = next(lector)
        except StopIteration:
            raise ValueError(f"{ruta} está vacía: no tiene ni cabecera")
        nombres = [_sin_tildes(c).replace(" ", "_") for c in cabecera]
        faltan = [c for c in COLUMNAS_OBLIGATORIAS if c not in nombres]
        if faltan:
            raise ValueError(f"{ruta} no trae la(s) columna(s) obligatoria(s) {', '.join(faltan)}. "
                             f"Cabecera esperada: {', '.join(COLUMNAS_OBLIGATORIAS + COLUMNAS_OPCIONALES)}")
        filas, problemas = [], []
        for num, crudo in enumerate(lector, start=2):
            if not crudo or all(not str(c).strip() for c in crudo):
                continue
            if str(crudo[0]).lstrip().startswith("#"):
                continue
            fila = {n: (crudo[i].strip() if i < len(crudo) else "") for i, n in enumerate(nombres)}
            try:
                registro = {
                    "linea": num,
                    "sesion": fila.get("sesion", "").strip(),
                    "id_ensayo": _id_ensayo(fila.get("id_ensayo")),
                    "objetivo": _glosa_objetivo(fila),
                    "presentacion": (fila.get("presentacion") or "").strip() or None,
                    "ronda": (fila.get("ronda") or "").strip() or None,
                    "intento": _entero(fila.get("intento")),
                    "referencia_mostrada": _si_no(fila.get("referencia_mostrada")),
                    "variante": _variante(fila.get("variante")),
                    "valida": _si_no(fila.get("valida")),
                    "estado_pantalla": (fila.get("estado_pantalla") or "").strip() or None,
                    "observacion": (fila.get("observacion") or "").strip(),
                }
            except ValueError as exc:
                problemas.append({"linea": num, "motivo": str(exc), "fila": fila})
                continue
            if not registro["sesion"] or registro["id_ensayo"] is None:
                problemas.append({"linea": num,
                                  "motivo": "la fila no trae sesion o id_ensayo: no se puede cruzar",
                                  "fila": fila})
                continue
            registro["glosa_objetivo"] = SIGN_VOCABULARY[registro["objetivo"]]
            filas.append(registro)
    return filas, problemas


def escribir_plantilla(destino) -> Path:
    """Copia docs/plantilla_hoja_registro_bloque1.csv en la ruta indicada, sin sobrescribir."""
    destino = Path(destino)
    if destino.exists():
        raise FileExistsError(f"{destino} ya existe; no se sobrescribe")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(PLANTILLA.read_text(encoding="utf-8"), encoding="utf-8")
    return destino


# ---------------------------------------------------------------------------
# Audio dudoso (N-37, N-38, E7, R20)
# ---------------------------------------------------------------------------

def audio_dudoso(ensayos: list[dict]) -> dict:
    """Ensayos cuya locución pudo no sonar, a partir de tts_errores_reproduccion.

    El contador es acumulado DENTRO de la sesión, empieza en 0, no baja, y se lee después de sellar t5;
    null significa «el motor no lo informa», que no es «cero errores» (N-37). Como el audio suena en
    segundo plano, el fallo del ensayo k suele aparecer en el registro k+1 (R20). De ahí dos convenciones;
    D-13 adoptó la B como definición de «audio dudoso»:
      · convencion_a: todos los ensayos desde aquel cuyo registro ya trae el contador > 0 (es la de
        src/latencia_resumen.py, clave errores_voz.ensayos_desde_el_primer_error, N-38);
      · convencion_b: la anterior más el ensayo inmediatamente anterior a cada aumento del contador, que
        es el que probablemente no sonó (observación del verificador independiente de N-38).
    """
    por_sesion: dict[str, list[dict]] = {}
    for r in ensayos:
        por_sesion.setdefault(str(r.get("sesion")), []).append(r)

    sesiones, dudosos_a, dudosos_b, incrementos_todos, anomalias = [], set(), set(), [], []
    for sesion, miembros in por_sesion.items():
        valores = [_entero_suave(r.get(CLAVE_ERRORES_VOZ)) for r in miembros]
        informados = [v for v in valores if v is not None]
        previo, incrementos = 0, []
        for i, v in enumerate(valores):
            if v is None:
                continue
            if v > previo:
                anterior = miembros[i - 1] if i > 0 else None
                incrementos.append({
                    "sesion": sesion,
                    "id_ensayo": miembros[i].get("id_ensayo"),
                    "id_ensayo_anterior": anterior.get("id_ensayo") if anterior else None,
                    "incremento": v - previo,
                    "contador": v,
                    "mensaje": miembros[i].get(CLAVE_ULTIMO_ERROR_VOZ),
                })
                if anterior is not None:
                    dudosos_b.add(_clave(sesion, anterior.get("id_ensayo")))
            elif v < previo:
                anomalias.append({"sesion": sesion, "id_ensayo": miembros[i].get("id_ensayo"),
                                  "contador": v, "contador_previo": previo})
            previo = max(previo, v)
        primero = next((i for i, v in enumerate(valores) if v is not None and v > 0), None)
        if primero is not None:
            for r in miembros[primero:]:
                dudosos_a.add(_clave(sesion, r.get("id_ensayo")))
        sesiones.append({
            "sesion": sesion,
            "n_ensayos": len(miembros),
            "informados": len(informados),
            "sin_dato": len(miembros) - len(informados),
            "ensayos_con_errores": sum(1 for v in informados if v > 0),
            "maximo": max(informados) if informados else None,
            "primer_error_en_ensayo": miembros[primero].get("id_ensayo") if primero is not None else None,
            "ensayos_desde_el_primer_error": (len(miembros) - primero) if primero is not None else 0,
            "incrementos": incrementos,
            "ultimo_mensaje": _ultimo_mensaje(miembros),
        })
        incrementos_todos.extend(incrementos)

    dudosos_b |= dudosos_a
    return {
        "por_sesion": sesiones,
        "incrementos": incrementos_todos,
        "contador_que_baja": anomalias,
        # Claves «sesion/id_ensayo», las mismas que lleva cada ejecución emparejada (clave_texto).
        "convencion_a": [_texto_clave(c) for c in sorted(dudosos_a, key=_orden_clave)],
        "convencion_b": [_texto_clave(c) for c in sorted(dudosos_b, key=_orden_clave)],
        "total": {
            "n_ensayos": len(ensayos),
            "informados": sum(s["informados"] for s in sesiones),
            "sin_dato": sum(s["sin_dato"] for s in sesiones),
            "ensayos_con_errores": sum(s["ensayos_con_errores"] for s in sesiones),
            "n_convencion_a": len(dudosos_a),
            "n_convencion_b": len(dudosos_b),
            "ultimo_mensaje": _ultimo_mensaje(ensayos),
        },
    }


def _entero_suave(valor) -> int | None:
    """int() tolerante: None cuando el campo no está o no es un entero (dato ausente, no cero)."""
    if valor is None or isinstance(valor, bool):
        return None
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def _ultimo_mensaje(ensayos: list[dict]) -> str | None:
    ultimo = None
    for r in ensayos:
        mensaje = r.get(CLAVE_ULTIMO_ERROR_VOZ)
        if mensaje is not None:
            ultimo = str(mensaje)
    return ultimo


def _orden_clave(clave: tuple) -> tuple:
    return (clave[0], 0, clave[1]) if isinstance(clave[1], int) else (clave[0], 1, str(clave[1]))


# ---------------------------------------------------------------------------
# Cruce hoja ↔ JSONL
# ---------------------------------------------------------------------------

def cruzar(ensayos: list[dict], filas: list[dict]) -> dict:
    """Une por (sesion, id_ensayo). Nada se descarta en silencio: lo que no empareja se lista."""
    por_clave, duplicados_jsonl = {}, []
    for r in ensayos:
        c = _clave(r.get("sesion"), r.get("id_ensayo"))
        if c in por_clave:
            duplicados_jsonl.append({"clave": _texto_clave(c)})
            continue
        por_clave[c] = r

    emparejados, filas_sin_ensayo, duplicadas, vistas = [], [], [], {}
    for fila in filas:
        c = _clave(fila["sesion"], fila["id_ensayo"])
        if c in vistas:
            duplicadas.append({"clave": _texto_clave(c), "linea": fila["linea"],
                               "linea_previa": vistas[c]})
            continue
        vistas[c] = fila["linea"]
        ensayo = por_clave.get(c)
        if ensayo is None:
            filas_sin_ensayo.append({"clave": _texto_clave(c), "linea": fila["linea"],
                                     "glosa_objetivo": fila["glosa_objetivo"],
                                     "observacion": fila["observacion"]})
            continue
        emparejados.append(_unir(fila, ensayo, c))

    ensayos_sin_fila = [
        {"clave": _texto_clave(c), "estado": r.get("estado"), "clase": r.get("clase"),
         "confianza": r.get("confianza"), "fecha_hora": r.get("fecha_hora")}
        for c, r in por_clave.items() if c not in vistas
    ]
    return {
        "emparejados": emparejados,
        "ensayos_sin_fila": ensayos_sin_fila,
        "filas_sin_ensayo": filas_sin_ensayo,
        "filas_duplicadas": duplicadas,
        "ensayos_duplicados": duplicados_jsonl,
    }


def _unir(fila: dict, ensayo: dict, clave: tuple) -> dict:
    top3 = ensayo.get("top3") or []
    id_clase = ensayo.get("id_clase")
    return {
        "clave": clave,
        "clave_texto": _texto_clave(clave),
        "linea": fila["linea"],
        "sesion": fila["sesion"],
        "id_ensayo": fila["id_ensayo"],
        "presentacion": fila["presentacion"],
        "ronda": fila["ronda"],
        "intento": fila["intento"],
        "objetivo": fila["objetivo"],
        "glosa_objetivo": fila["glosa_objetivo"],
        "variante": fila["variante"],
        "valida": fila["valida"],
        "referencia_mostrada": fila["referencia_mostrada"],
        "observacion": fila["observacion"],
        "estado_pantalla": fila["estado_pantalla"],
        "estado": ensayo.get("estado"),
        "id_clase": id_clase,
        "clase": ensayo.get("clase"),
        "confianza": ensayo.get("confianza"),
        "top3": [int(t["id"]) for t in top3 if t.get("id") is not None],
        "tiene_prediccion": id_clase is not None,
        "registro": ensayo,
    }


def presentaciones_incoherentes(ejecuciones: list[dict]) -> list[dict]:
    """Números de presentación que agrupan más de una glosa objetivo o más de una ronda.

    Las políticas «primer_intento» y «ultimo_intento» de PD-14 agrupan por (sesion, presentacion). Si el
    operador reinicia la numeración en cada ronda (PD-03 C), dos presentaciones distintas comparten
    número y esas políticas se quedarían con un solo intento de las dos, descartando el otro sin avisar.
    Aquí NO se cambia la agrupación: se avisa, con las claves implicadas, para que la hoja se corrija o
    se numeren las presentaciones de corrido dentro de cada sesión (revisión independiente, 16-sep-2026).
    """
    grupos: dict[tuple, list[dict]] = {}
    for e in ejecuciones:
        if e["presentacion"]:
            grupos.setdefault((e["sesion"], e["presentacion"]), []).append(e)
    fuera = []
    for (sesion, presentacion), miembros in grupos.items():
        objetivos = sorted({m["objetivo"] for m in miembros})
        rondas = sorted({str(m["ronda"]) for m in miembros if m["ronda"]})
        if len(objetivos) > 1 or len(rondas) > 1:
            fuera.append({"sesion": sesion, "presentacion": presentacion,
                          "n_ejecuciones": len(miembros),
                          "glosas_objetivo": [SIGN_VOCABULARY[o] for o in objetivos],
                          "rondas": rondas,
                          "claves": [m["clave_texto"] for m in miembros]})
    fuera.sort(key=lambda x: (x["sesion"], str(x["presentacion"])))
    return fuera


def discrepancias_estado(emparejados: list[dict]) -> list[dict]:
    """Control de transcripción (R14): estado anotado en la hoja frente al estado del JSONL."""
    fuera = []
    for e in emparejados:
        anotado = e["estado_pantalla"]
        if not anotado:
            continue
        if _sin_tildes(anotado).replace(" ", "_") != _sin_tildes(e["estado"] or "").replace(" ", "_"):
            fuera.append({"clave": e["clave_texto"], "linea": e["linea"],
                          "hoja": anotado, "jsonl": e["estado"]})
    return fuera


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _contar_estados(ejecuciones: list[dict]) -> dict:
    cuenta = {e: sum(1 for x in ejecuciones if x["estado"] == e) for e in ESTADOS}
    cuenta["otro"] = len(ejecuciones) - sum(cuenta.values())
    return cuenta


def _rechazos(ejecuciones: list[dict]) -> dict:
    """Tasas de rechazo. El denominador es el número de ejecuciones del conjunto (protocolo 7.8)."""
    n = len(ejecuciones)
    cuenta = _contar_estados(ejecuciones)
    def tasa(k):
        return (cuenta[k] / n) if n else None
    return {
        "n_ejecuciones": n,
        "contadores": cuenta,
        "tasa_rechazo_confianza": tasa("rechazada_confianza"),
        "tasa_rechazo_manos": tasa("rechazada_manos"),
        "tasa_rechazo_incompleta": tasa("rechazada_incompleta"),
        "tasa_rechazo_total": (1 - cuenta["aceptada"] / n) if n else None,
    }


def _matriz_dispersa(reales, predichas) -> dict:
    m = matriz_confusion(reales, predichas, N_CLASES)
    dispersa: dict[str, dict[str, int]] = {}
    for i in range(N_CLASES):
        fila = {str(j): int(m[i, j]) for j in range(N_CLASES) if m[i, j]}
        if fila:
            dispersa[str(i)] = fila
    return dispersa


def metricas(ejecuciones: list[dict], denominador: str) -> dict:
    """Métricas de un conjunto de ejecuciones bajo uno de los tres denominadores de PD-15.

    con_prediccion      : numerador y denominador son las ejecuciones que llegaron al modelo.
    aceptadas           : numerador y denominador son las ejecuciones aceptadas.
    todas_rechazo_error : numerador son los aciertos de las aceptadas; denominador, TODAS las
                          ejecuciones del conjunto, así que cualquier rechazo cuenta como error.
    """
    if denominador not in DENOMINADORES:
        raise ValueError(f"denominador desconocido: {denominador}")
    if denominador == "con_prediccion":
        usadas, n = [e for e in ejecuciones if e["tiene_prediccion"]], None
    elif denominador == "aceptadas":
        usadas, n = [e for e in ejecuciones if e["estado"] == "aceptada"], None
    else:
        usadas, n = [e for e in ejecuciones if e["estado"] == "aceptada"], len(ejecuciones)
    if n is None:
        n = len(usadas)

    resultado = {
        "denominador": denominador,
        "etiqueta": ETIQUETA_DENOMINADOR[denominador],
        "n_denominador": n,
        "n_con_salida": len(usadas),
        "aciertos_top1": None, "aciertos_top3": None,
        "top1": None, "top3": None,
        "precision_macro": None, "recall_macro": None, "f1_macro": None,
        "f1_macro_50clases": None, "n_clases_macro": 0,
        "por_clase": [], "matriz_confusion": {},
    }
    if denominador == "todas_rechazo_error":
        # Qué señas se quedaron sin salida del sistema y por qué (el rechazo cuenta como error).
        sin_salida: dict[str, dict] = {}
        for x in ejecuciones:
            if x["estado"] == "aceptada":
                continue
            d = sin_salida.setdefault(str(x["objetivo"]),
                                      {"glosa": x["glosa_objetivo"], "n": 0, "estados": {}})
            d["n"] += 1
            d["estados"][x["estado"]] = d["estados"].get(x["estado"], 0) + 1
        resultado["sin_salida_por_sena"] = sin_salida
        # Un rechazo no tiene clase predicha, así que no cabe en una matriz de confusión: aquí solo
        # top-1 y top-3 usan el denominador de TODAS las ejecuciones. Precisión, recall, F1, por_clase
        # y la matriz salen de las n_con_salida aceptadas, igual que en «aceptadas» (revisión
        # independiente, 16-sep-2026).
        resultado["nota_macro"] = (
            "top-1 y top-3 van sobre las n_denominador ejecuciones (todo rechazo cuenta como error); "
            "precisión, recall, F1, por_clase y matriz_confusion se calculan solo sobre las "
            "n_con_salida ejecuciones aceptadas, porque un rechazo no tiene clase predicha. Las señas "
            "que se quedaron sin salida están en sin_salida_por_sena.")

    if not usadas:
        # Sin ninguna salida del sistema: con denominador > 0 la exactitud es 0, no «sin dato».
        if n:
            resultado.update({"aciertos_top1": 0, "aciertos_top3": 0, "top1": 0.0, "top3": 0.0})
        return resultado

    reales = [e["objetivo"] for e in usadas]
    predichas = [e["id_clase"] for e in usadas]
    # top-3 con la convención de src/analisis_errores.py, top_k: el objetivo entre los tres primeros.
    aciertos1 = sum(1 for r, p in zip(reales, predichas) if r == p)
    aciertos3 = sum(1 for e in usadas if e["objetivo"] in e["top3"][:3])

    por_clase = metricas_por_clase(reales, predichas, N_CLASES)
    presentes = [c for c in range(N_CLASES) if por_clase["soporte"][c] > 0]
    def promedio(clave, cero_si_none=False):
        vals = []
        for c in presentes:
            v = por_clase[clave][c]
            if v is None:
                if not cero_si_none:
                    continue
                v = 0.0
            vals.append(float(v))
        return float(sum(vals) / len(vals)) if vals else None

    resultado.update({
        "aciertos_top1": aciertos1,
        "aciertos_top3": aciertos3,
        "top1": aciertos1 / n if n else None,
        "top3": aciertos3 / n if n else None,
        "precision_macro": promedio("precision", cero_si_none=True),
        "recall_macro": promedio("recall"),
        "f1_macro": promedio("f1"),
        "f1_macro_50clases": float(sum(por_clase["f1"]) / N_CLASES),
        "n_clases_macro": len(presentes),
        "por_clase": [{
            "id": c, "glosa": SIGN_VOCABULARY[c],
            "soporte": int(por_clase["soporte"][c]),
            "n_predichas": int(por_clase["n_predichas"][c]),
            "aciertos": int(por_clase["aciertos"][c]),
            "recall": None if por_clase["recall"][c] is None else float(por_clase["recall"][c]),
            "precision": None if por_clase["precision"][c] is None else float(por_clase["precision"][c]),
            "f1": float(por_clase["f1"][c]),
        } for c in presentes],
        "matriz_confusion": _matriz_dispersa(reales, predichas),
    })
    return resultado


def confusiones(matriz: dict) -> list[dict]:
    """Pares objetivo → predicha con al menos un caso, los errores primero y por frecuencia."""
    pares = []
    for i, fila in matriz.items():
        for j, n in fila.items():
            pares.append({"objetivo": int(i), "glosa_objetivo": SIGN_VOCABULARY[int(i)],
                          "predicha": int(j), "glosa_predicha": SIGN_VOCABULARY[int(j)],
                          "n": int(n), "acierto": i == j})
    pares.sort(key=lambda p: (p["acierto"], -p["n"], p["objetivo"], p["predicha"]))
    return pares


# ---------------------------------------------------------------------------
# Conjuntos (PD-13) y políticas de repetición (PD-14)
# ---------------------------------------------------------------------------

def subconjunto(validas: list[dict], conjunto: str) -> list[dict]:
    if conjunto == "totalidad":
        return list(validas)
    if conjunto == "sin_variante_estricto":
        return [e for e in validas if e["variante"] == "no"]
    if conjunto == "sin_variante_amplio":
        return [e for e in validas if e["variante"] in ("no", "no_segura")]
    raise ValueError(f"conjunto desconocido: {conjunto}")


def aplicar_politica(ejecuciones: list[dict], politica: str) -> list[dict] | None:
    """Selecciona qué intento de cada presentación entra (PD-14 B). None si no es calculable."""
    if politica == "todos":
        return list(ejecuciones)
    if politica not in ("primer_intento", "ultimo_intento"):
        raise ValueError(f"política desconocida: {politica}")
    if not any(e["presentacion"] for e in ejecuciones):
        return None
    grupos: dict[tuple, list[dict]] = {}
    for i, e in enumerate(ejecuciones):
        # Sin número de presentación, la ejecución es su propia presentación (y queda señalada aparte).
        clave = (e["sesion"], e["presentacion"]) if e["presentacion"] else ("", f"__sin__{i}")
        grupos.setdefault(clave, []).append((i, e))
    elegidas = []
    for miembros in grupos.values():
        miembros.sort(key=lambda par: (par[1]["intento"] if par[1]["intento"] is not None else 0, par[0]))
        elegidas.append(miembros[0] if politica == "primer_intento" else miembros[-1])
    elegidas.sort(key=lambda par: par[0])
    return [e for _, e in elegidas]


def bloque_conjunto(ejecuciones: list[dict]) -> dict:
    """Estados, rechazos y las tres lecturas de PD-15 para un conjunto ya filtrado."""
    bloque = {"n_ejecuciones": len(ejecuciones), "rechazos": _rechazos(ejecuciones), "denominadores": {}}
    for d in DENOMINADORES:
        bloque["denominadores"][d] = metricas(ejecuciones, d)
    return bloque


# ---------------------------------------------------------------------------
# Desgloses por seña y por repetición
# ---------------------------------------------------------------------------

def por_sena(ejecuciones: list[dict]) -> list[dict]:
    """Una fila por seña presentada, con los tres denominadores resumidos en conteos."""
    grupos: dict[int, list[dict]] = {}
    for e in ejecuciones:
        grupos.setdefault(e["objetivo"], []).append(e)
    filas = []
    for objetivo in sorted(grupos):
        miembros = grupos[objetivo]
        con_pred = [e for e in miembros if e["tiene_prediccion"]]
        aceptadas = [e for e in miembros if e["estado"] == "aceptada"]
        a1_pred = sum(1 for e in con_pred if e["id_clase"] == objetivo)
        a3_pred = sum(1 for e in con_pred if objetivo in e["top3"])
        a1_acep = sum(1 for e in aceptadas if e["id_clase"] == objetivo)
        filas.append({
            "id": objetivo, "glosa": SIGN_VOCABULARY[objetivo],
            "n_ejecuciones": len(miembros),
            "n_con_prediccion": len(con_pred),
            "n_aceptadas": len(aceptadas),
            "estados": _contar_estados(miembros),
            "aciertos_top1_con_prediccion": a1_pred,
            "aciertos_top3_con_prediccion": a3_pred,
            "top1_con_prediccion": a1_pred / len(con_pred) if con_pred else None,
            "top3_con_prediccion": a3_pred / len(con_pred) if con_pred else None,
            "aciertos_aceptadas": a1_acep,
            "top1_aceptadas": a1_acep / len(aceptadas) if aceptadas else None,
            "top1_todas_rechazo_error": a1_acep / len(miembros) if miembros else None,
            "variantes": _contar_variantes(miembros),
        })
    return filas


def _contar_variantes(ejecuciones: list[dict]) -> dict:
    cuenta = {"si": 0, "no": 0, "no_segura": 0, "sin_dato": 0}
    for e in ejecuciones:
        cuenta["sin_dato" if e["variante"] is None else e["variante"]] += 1
    return cuenta


def por_repeticion(ejecuciones: list[dict], campo: str) -> list[dict]:
    """Desglose por «intento» o por «ronda»: la misma cifra repetición a repetición."""
    grupos: dict[str, list[dict]] = {}
    for e in ejecuciones:
        valor = e[campo]
        grupos.setdefault("sin_dato" if valor is None else str(valor), []).append(e)
    filas = []
    for valor in sorted(grupos, key=_orden_valor):
        miembros = grupos[valor]
        con_pred = [e for e in miembros if e["tiene_prediccion"]]
        aceptadas = [e for e in miembros if e["estado"] == "aceptada"]
        a1_pred = sum(1 for e in con_pred if e["id_clase"] == e["objetivo"])
        a3_pred = sum(1 for e in con_pred if e["objetivo"] in e["top3"])
        a1_acep = sum(1 for e in aceptadas if e["id_clase"] == e["objetivo"])
        filas.append({
            campo: valor,
            "n_ejecuciones": len(miembros),
            "n_con_prediccion": len(con_pred),
            "n_aceptadas": len(aceptadas),
            "estados": _contar_estados(miembros),
            "top1_con_prediccion": a1_pred / len(con_pred) if con_pred else None,
            "top3_con_prediccion": a3_pred / len(con_pred) if con_pred else None,
            "top1_aceptadas": a1_acep / len(aceptadas) if aceptadas else None,
            "top1_todas_rechazo_error": a1_acep / len(miembros) if miembros else None,
            "n_senas_distintas": len({e["objetivo"] for e in miembros}),
        })
    return filas


def _orden_valor(valor: str) -> tuple:
    if valor == "sin_dato":
        return (2, 0.0, "")
    try:
        return (0, float(valor), "")
    except ValueError:
        return (1, 0.0, valor)


# ---------------------------------------------------------------------------
# Control de configuración (E2)
# ---------------------------------------------------------------------------

def configuracion(emparejados: list[dict]) -> dict:
    """Valores distintos de la configuración congelada (sección 3 del protocolo, criterio E2).

    No excluye nada: E2 dice que un ensayo con otra configuración «se cuenta aparte», y cuál se reporta
    lo decide el director. Aquí solo se listan los valores encontrados y cuántos ensayos trae cada uno.
    """
    resumen = {}
    for clave in CLAVES_CONFIGURACION:
        cuenta: dict[str, int] = {}
        for e in emparejados:
            cuenta[str(e["registro"].get(clave))] = cuenta.get(str(e["registro"].get(clave)), 0) + 1
        resumen[clave] = cuenta
    umbrales: dict[str, int] = {}
    for e in emparejados:
        p = e["registro"].get("parametros") or {}
        etiqueta = f"{p.get('umbral')} ({p.get('umbral_origen')})"
        umbrales[etiqueta] = umbrales.get(etiqueta, 0) + 1
    resumen["umbral"] = umbrales
    sintesis: dict[str, int] = {}
    for e in emparejados:
        sintesis[str(e["registro"].get("tts_sintesis_antes_de_t5"))] = \
            sintesis.get(str(e["registro"].get("tts_sintesis_antes_de_t5")), 0) + 1
    resumen["tts_sintesis_antes_de_t5"] = sintesis
    return resumen


# ---------------------------------------------------------------------------
# Análisis completo
# ---------------------------------------------------------------------------

PENDIENTES = [
    "PD-15 (denominador de la exactitud): el informe da las tres lecturas «con_prediccion», «aceptadas» "
    "y «todas_rechazo_error»; cuál se reporta como la exactitud del Bloque 1 lo decide el director.",
    "PD-14 (política ante un rechazo): el informe da «todos», «primer_intento» y «ultimo_intento»; "
    "cuál cuenta lo decide el director, junto con el número máximo de intentos.",
    "PD-13 (categorías de la variante): si la hoja usa «no estoy segura», no está decidido si esa "
    "ejecución cuenta como variante; se dan los conjuntos estricto y amplio.",
    "E7 / N-38 / D-13 (fallos de voz): D-13 fija la convención B como «audio dudoso»; qué se hace con esos "
    "ensayos sigue abierto en PD-31, así que aquí no se excluye ninguno, como dice E7.",
    "E2 (configuración): los ensayos con configuración distinta de la sección 3 se listan pero no se "
    "excluyen; el criterio dice contarlos aparte.",
]


def analizar(rutas_registros: list[str], ruta_hoja, sesiones: list[str] | None = None) -> dict:
    registros = leer_registros(list(rutas_registros))
    ensayos = [r for r in registros if r.get("tipo") == "ensayo"]
    turnos = [r for r in registros if r.get("tipo") == "turno"]
    if sesiones:
        pedidas = {str(s).strip() for s in sesiones}
        ensayos = [r for r in ensayos if str(r.get("sesion")).strip() in pedidas]

    filas, problemas_hoja = leer_hoja(ruta_hoja)
    voz = audio_dudoso(ensayos)
    cruce = cruzar(ensayos, filas)
    emparejados = cruce["emparejados"]

    dudosos_a = set(voz["convencion_a"])
    dudosos_b = set(voz["convencion_b"])
    for e in emparejados:
        e["audio_dudoso_a"] = e["clave_texto"] in dudosos_a
        e["audio_dudoso_b"] = e["clave_texto"] in dudosos_b

    validas = [e for e in emparejados if e["valida"] is True]
    no_validas = [e for e in emparejados if e["valida"] is False]
    sin_validez = [e for e in emparejados if e["valida"] is None]

    advertencias: list[str] = []
    if cruce["ensayos_sin_fila"]:
        advertencias.append(f"{len(cruce['ensayos_sin_fila'])} ensayo(s) del JSONL no tienen fila en la "
                            f"hoja: no se conoce su glosa objetivo y no entran en ninguna métrica.")
    if cruce["filas_sin_ensayo"]:
        advertencias.append(f"{len(cruce['filas_sin_ensayo'])} fila(s) de la hoja no tienen ensayo en el "
                            f"JSONL: revise la transcripción del id_ensayo (R14).")
    if cruce["filas_duplicadas"]:
        advertencias.append(f"{len(cruce['filas_duplicadas'])} fila(s) repiten una clave "
                            f"(sesion, id_ensayo); solo se usó la primera.")
    if cruce["ensayos_duplicados"]:
        advertencias.append(f"{len(cruce['ensayos_duplicados'])} ensayo(s) del JSONL repiten la clave "
                            f"(sesion, id_ensayo); solo se usó el primero.")
    if problemas_hoja:
        advertencias.append(f"{len(problemas_hoja)} fila(s) de la hoja no se pudieron interpretar y "
                            f"quedaron fuera; el motivo de cada una está en cruce.filas_con_problema.")
    if sin_validez:
        advertencias.append(f"{len(sin_validez)} ejecución(es) emparejada(s) sin el campo «valida»: no "
                            f"entran en ninguna métrica hasta que el operador lo llene.")
    sin_variante = [e for e in validas if e["variante"] is None]
    if sin_variante:
        advertencias.append(f"{len(sin_variante)} ejecución(es) válida(s) sin declaración de variante: "
                            f"cuentan en la totalidad y en ningún conjunto «sin variante».")
    presentaciones_raras = presentaciones_incoherentes(validas)
    if presentaciones_raras:
        advertencias.append(f"{len(presentaciones_raras)} número(s) de presentación agrupan más de una "
                            f"glosa objetivo o más de una ronda: las políticas «primer_intento» y "
                            f"«ultimo_intento» de PD-14 agrupan por (sesion, presentacion) y se "
                            f"quedarían con un solo intento de presentaciones distintas. Numere las "
                            f"presentaciones de corrido dentro de cada sesión; el detalle está en "
                            f"cruce.presentaciones_incoherentes.")
    discrepancias = discrepancias_estado(emparejados)
    if discrepancias:
        advertencias.append(f"{len(discrepancias)} ejecución(es) con el estado de la hoja distinto del "
                            f"estado del JSONL (control de transcripción, R14).")
    if voz["total"]["ensayos_con_errores"]:
        advertencias.append(f"{voz['total']['ensayos_con_errores']} ensayo(s) llegan con "
                            f"tts_errores_reproduccion > 0: {voz['total']['n_convencion_a']} ensayo(s) de "
                            f"audio dudoso por la convención A y {voz['total']['n_convencion_b']} por la B (la de D-13).")
    if voz["contador_que_baja"]:
        advertencias.append("El contador de errores de voz baja en algún punto, cosa que no debería "
                            "ocurrir dentro de una sesión: revise si el JSONL mezcla sesiones.")
    conf = configuracion(emparejados)
    for clave, valores in conf.items():
        if clave == "tts_sintesis_antes_de_t5":
            # None es normal: una ejecución rechazada no emite audio. Lo que E2 señala es el false.
            if valores.get("False"):
                advertencias.append(f"{valores['False']} ensayo(s) con tts_sintesis_antes_de_t5 = false: "
                                    f"la síntesis real no quedó dentro de t5 (criterio E2).")
            continue
        if len(valores) > 1:
            advertencias.append(f"Hay más de un valor de «{clave}» entre los ensayos emparejados "
                                f"({', '.join(sorted(valores))}): criterio E2.")
    hay_no_segura = any(e["variante"] == "no_segura" for e in validas)

    exactitud: dict[str, dict] = {}
    for conjunto in CONJUNTOS:
        base = subconjunto(validas, conjunto)
        entrada = {"etiqueta": ETIQUETA_CONJUNTO[conjunto], "n_ejecuciones": len(base), "politicas": {}}
        if conjunto == "sin_variante_amplio":
            entrada["aplica"] = hay_no_segura
            entrada["nota"] = ("idéntico al estricto: la hoja no usa «no estoy segura»" if not hay_no_segura
                               else "incluye las ejecuciones declaradas «no estoy segura» (PD-13)")
        for politica in POLITICAS:
            elegidas = aplicar_politica(base, politica)
            if elegidas is None:
                entrada["politicas"][politica] = {
                    "etiqueta": ETIQUETA_POLITICA[politica], "no_calculable":
                    "la hoja no trae la columna «presentacion» con valores; sin ella no se pueden "
                    "agrupar los intentos de una misma presentación (PD-14)."}
                continue
            bloque = bloque_conjunto(elegidas)
            bloque["etiqueta"] = ETIQUETA_POLITICA[politica]
            bloque["n_sin_presentacion"] = sum(1 for e in base if not e["presentacion"])
            entrada["politicas"][politica] = bloque
        exactitud[conjunto] = entrada

    referencia = exactitud["totalidad"]["politicas"]["todos"]
    matriz = referencia["denominadores"]["con_prediccion"]["matriz_confusion"]

    return {
        "guion": "src/exactitud_bloque1.py",
        "generado": _dt.datetime.now().isoformat(timespec="seconds"),
        "entradas": {
            "registros": list(rutas_registros),
            "hoja": str(ruta_hoja),
            "sesiones_pedidas": list(sesiones) if sesiones else None,
            "n_registros_leidos": len(registros),
            "n_ensayos_jsonl": len(ensayos),
            "n_turnos_jsonl": len(turnos),
            "sesiones_en_el_jsonl": sorted({str(r.get("sesion")) for r in ensayos}),
            "n_filas_hoja": len(filas),
        },
        "cruce": {
            "n_emparejados": len(emparejados),
            "ensayos_sin_fila": cruce["ensayos_sin_fila"],
            "filas_sin_ensayo": cruce["filas_sin_ensayo"],
            "filas_duplicadas": cruce["filas_duplicadas"],
            "ensayos_duplicados": cruce["ensayos_duplicados"],
            "filas_con_problema": problemas_hoja,
            "discrepancias_estado": discrepancias,
            "presentaciones_incoherentes": presentaciones_raras,
        },
        "validez": {
            "validas": len(validas),
            "no_validas": len(no_validas),
            "sin_dato": len(sin_validez),
            "no_validas_detalle": [{"clave": e["clave_texto"], "linea": e["linea"],
                                    "glosa_objetivo": e["glosa_objetivo"], "estado": e["estado"],
                                    "observacion": e["observacion"]} for e in no_validas],
            "sin_dato_detalle": [{"clave": e["clave_texto"], "linea": e["linea"]} for e in sin_validez],
        },
        "variantes": {
            "sobre": "ejecuciones válidas",
            "contadores": _contar_variantes(validas),
            "proporcion_con_variante": (sum(1 for e in validas if e["variante"] == "si") / len(validas))
            if validas else None,
            "proporcion_no_segura": (sum(1 for e in validas if e["variante"] == "no_segura") / len(validas))
            if validas else None,
            "usa_no_estoy_segura": hay_no_segura,
        },
        "configuracion": conf,
        "exactitud": exactitud,
        "por_sena": por_sena(validas),
        "por_repeticion": {"por_intento": por_repeticion(validas, "intento"),
                           "por_ronda": por_repeticion(validas, "ronda")},
        "matriz_confusion_referencia": {
            "conjunto": "totalidad", "politica": "todos", "denominador": "con_prediccion",
            "matriz": matriz, "pares": confusiones(matriz),
        },
        "audio_dudoso": {
            **voz,
            "validas_convencion_a": sum(1 for e in validas if e["audio_dudoso_a"]),
            "validas_convencion_b": sum(1 for e in validas if e["audio_dudoso_b"]),
        },
        "advertencias": advertencias,
        "pendientes_de_decision": PENDIENTES,
    }


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------

def _n(x, d: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{d}f}".replace(".", ",")


def _pct(x, d: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.{d}f} %".replace(".", ",")


def _fila_t2(nombre: str, bloque: dict) -> str:
    d = bloque["denominadores"]
    cp, ac, td = d["con_prediccion"], d["aceptadas"], d["todas_rechazo_error"]
    r = bloque["rechazos"]
    return (f"| {nombre} | {bloque['n_ejecuciones']} | {cp['n_denominador']} | {_n(cp['top1'])} | "
            f"{_n(cp['top3'])} | {_n(cp['precision_macro'])} | {_n(cp['recall_macro'])} | "
            f"{_n(cp['f1_macro'])} | {ac['n_denominador']} | {_n(ac['top1'])} | "
            f"{td['n_denominador']} | {_n(td['top1'])} | "
            f"{r['contadores']['rechazada_confianza']} ({_pct(r['tasa_rechazo_confianza'])}) |")


_CABECERA_T2 = (
    "| Conjunto | n ejec. | n con predicción | top-1 | top-3 | P macro | R macro | F1 macro | "
    "n aceptadas | top-1 entre aceptadas | n todas | top-1 todo rechazo = error | "
    "Rechazos por confianza (n, %) |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def informe_md(a: dict) -> str:
    L: list[str] = []
    e = a["entradas"]
    L.append("# Exactitud en línea del Bloque 1 (actividad 4.a)")
    L.append("")
    L.append(f"Generado el {a['generado']} por `{a['guion']}`.")
    L.append("")
    L.append(f"- Registros: {', '.join(e['registros'])} — {e['n_ensayos_jsonl']} ensayo(s) en "
             f"{len(e['sesiones_en_el_jsonl'])} sesión(es) ({', '.join(e['sesiones_en_el_jsonl']) or '—'}).")
    L.append(f"- Hoja de registro: `{e['hoja']}` — {e['n_filas_hoja']} fila(s) legible(s).")
    if e["sesiones_pedidas"]:
        L.append(f"- Filtro de sesiones: {', '.join(e['sesiones_pedidas'])}.")
    L.append("")
    L.append("**Este informe no elige el denominador de la exactitud (PD-15) ni la política de "
             "repetición (PD-14): los deja abiertos y calcula todas las variantes, para que el director "
             "decida.** Cada cifra lleva el nombre de la opción a la que corresponde.")
    L.append("")

    L.append("## 1. Cruce de la hoja con el JSONL")
    L.append("")
    c = a["cruce"]
    L.append("| Concepto | n |")
    L.append("|---|---|")
    L.append(f"| Ensayos en el JSONL | {e['n_ensayos_jsonl']} |")
    L.append(f"| Filas legibles en la hoja | {e['n_filas_hoja']} |")
    L.append(f"| Emparejados por (sesion, id_ensayo) | {c['n_emparejados']} |")
    L.append(f"| Ensayos del JSONL sin fila en la hoja | {len(c['ensayos_sin_fila'])} |")
    L.append(f"| Filas de la hoja sin ensayo en el JSONL | {len(c['filas_sin_ensayo'])} |")
    L.append(f"| Filas repetidas (misma clave) | {len(c['filas_duplicadas'])} |")
    L.append(f"| Ensayos repetidos (misma clave) | {len(c['ensayos_duplicados'])} |")
    L.append(f"| Filas que no se pudieron interpretar | {len(c['filas_con_problema'])} |")
    L.append(f"| Estado de la hoja distinto del JSONL | {len(c['discrepancias_estado'])} |")
    L.append(f"| Presentaciones con más de una glosa o ronda | "
             f"{len(c['presentaciones_incoherentes'])} |")
    L.append("")
    for titulo, clave, formato in (
        ("Ensayos del JSONL sin fila en la hoja", "ensayos_sin_fila",
         lambda x: f"`{x['clave']}` · estado {x['estado']} · clase {x['clase']}"),
        ("Filas de la hoja sin ensayo en el JSONL", "filas_sin_ensayo",
         lambda x: f"`{x['clave']}` · línea {x['linea']} · objetivo {x['glosa_objetivo']}"),
        ("Filas repetidas", "filas_duplicadas",
         lambda x: f"`{x['clave']}` · línea {x['linea']} (ya estaba en la {x['linea_previa']})"),
        ("Ensayos repetidos en el JSONL", "ensayos_duplicados", lambda x: f"`{x['clave']}`"),
        ("Filas que no se pudieron interpretar", "filas_con_problema",
         lambda x: f"línea {x['linea']} · {x['motivo']}"),
        ("Estado de la hoja distinto del JSONL", "discrepancias_estado",
         lambda x: f"`{x['clave']}` · línea {x['linea']} · hoja «{x['hoja']}» / JSONL «{x['jsonl']}»"),
        ("Presentaciones cuyo número agrupa más de una glosa objetivo o más de una ronda "
         "(afecta a «primer_intento» y «ultimo_intento» de PD-14)", "presentaciones_incoherentes",
         lambda x: f"sesión {x['sesion']}, presentación {x['presentacion']} · "
                   f"{x['n_ejecuciones']} ejecución(es) · glosas {', '.join(x['glosas_objetivo'])} · "
                   f"rondas {', '.join(x['rondas']) or '—'} · {', '.join(x['claves'])}"),
    ):
        if c[clave]:
            L.append(f"**{titulo}** ({len(c[clave])}):")
            L.append("")
            L.extend(f"- {formato(x)}" for x in c[clave])
            L.append("")
    if not any(c[k] for k in ("ensayos_sin_fila", "filas_sin_ensayo", "filas_duplicadas",
                              "ensayos_duplicados", "filas_con_problema", "discrepancias_estado",
                              "presentaciones_incoherentes")):
        L.append("El cruce es completo: cada ensayo del JSONL tiene su fila y cada fila su ensayo.")
        L.append("")

    L.append("## 2. Validez y variantes declaradas")
    L.append("")
    v, va = a["validez"], a["variantes"]
    L.append(f"Ejecuciones emparejadas: {c['n_emparejados']} — válidas {v['validas']}, "
             f"no válidas {v['no_validas']}, sin el campo «valida» {v['sin_dato']}. "
             f"Solo las válidas entran en las métricas.")
    L.append("")
    if v["no_validas_detalle"]:
        L.append("**Ejecuciones marcadas como no válidas** (motivo tal como lo escribió el operador):")
        L.append("")
        L.extend(f"- `{x['clave']}` · línea {x['linea']} · objetivo {x['glosa_objetivo']} · "
                 f"estado {x['estado']} · {x['observacion'] or 'sin observación'}"
                 for x in v["no_validas_detalle"])
        L.append("")
    cv = va["contadores"]
    L.append(f"Variante declarada sobre las {v['validas']} ejecuciones válidas: sí {cv['si']} "
             f"({_pct(va['proporcion_con_variante'])}), no {cv['no']}, "
             f"«no estoy segura» {cv['no_segura']}, sin dato {cv['sin_dato']}.")
    L.append("")

    L.append("## 3. Exactitud por duplicado (plantilla T2 del protocolo)")
    L.append("")
    L.append("El anteproyecto exige reportar la exactitud sobre la totalidad de las ejecuciones y sobre "
             "las ejecuciones sin variante declarada [MET 229-233]; reportar solo la cifra depurada sería "
             "un resultado con sesgo de selección. Las columnas dan las tres lecturas de PD-15.")
    L.append("")
    for politica in POLITICAS:
        L.append(f"### 3.{POLITICAS.index(politica) + 1} Política de repetición: "
                 f"{ETIQUETA_POLITICA[politica]} (PD-14)")
        L.append("")
        bloques = {cj: a["exactitud"][cj]["politicas"][politica] for cj in CONJUNTOS}
        visibles = [cj for cj in CONJUNTOS
                    if not (cj == "sin_variante_amplio" and not a["exactitud"][cj].get("aplica", True))]
        calculables = [cj for cj in visibles if "no_calculable" not in bloques[cj]]
        for cj in visibles:
            if "no_calculable" in bloques[cj]:
                L.append(f"- {ETIQUETA_CONJUNTO[cj]}: {bloques[cj]['no_calculable']}")
        if not calculables:
            L.append("")
            continue
        L.append("")
        L.append(_CABECERA_T2)
        for cj in calculables:
            L.append(_fila_t2(ETIQUETA_CONJUNTO[cj], bloques[cj]))
        L.append("")
        L.append(f"Denominadores: top-1, top-3, P, R y F1 sobre las ejecuciones con predicción = "
                 f"**{ETIQUETA_DENOMINADOR['con_prediccion']}**; «top-1 entre aceptadas» = "
                 f"**{ETIQUETA_DENOMINADOR['aceptadas']}**; «top-1 todo rechazo = error» = "
                 f"**{ETIQUETA_DENOMINADOR['todas_rechazo_error']}**. Dar las tres es la opción PD-15 C; "
                 f"cuál es la cifra del Bloque 1 lo decide el director.")
        L.append("")
        for cj in calculables:
            r = bloques[cj]["rechazos"]
            cc = r["contadores"]
            L.append(f"- {ETIQUETA_CONJUNTO[cj]}: aceptadas {cc['aceptada']}, rechazadas por confianza "
                     f"{cc['rechazada_confianza']} ({_pct(r['tasa_rechazo_confianza'])}), por manos "
                     f"{cc['rechazada_manos']} ({_pct(r['tasa_rechazo_manos'])}), incompletas "
                     f"{cc['rechazada_incompleta']} ({_pct(r['tasa_rechazo_incompleta'])}); "
                     f"rechazo total {_pct(r['tasa_rechazo_total'])}.")
        L.append("")
    if not a["variantes"]["usa_no_estoy_segura"]:
        L.append("La hoja no usa la categoría «no estoy segura», así que el conjunto amplio de PD-13 "
                 "coincide con el estricto y no se imprime aparte.")
        L.append("")

    L.append("## 4. Exactitud por seña")
    L.append("")
    L.append("Sobre las ejecuciones válidas, todos los intentos (conjunto «totalidad»).")
    L.append("")
    L.append("| id | Seña | n | con predicción | aceptadas | top-1 (con predicción) | "
             "top-3 (con predicción) | top-1 (entre aceptadas) | top-1 (todo rechazo = error) | "
             "rech. confianza | rech. manos | rech. incompleta | variantes sí |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in a["por_sena"]:
        st = f["estados"]
        L.append(f"| {f['id']} | {f['glosa']} | {f['n_ejecuciones']} | {f['n_con_prediccion']} | "
                 f"{f['n_aceptadas']} | {_n(f['top1_con_prediccion'])} | {_n(f['top3_con_prediccion'])} | "
                 f"{_n(f['top1_aceptadas'])} | {_n(f['top1_todas_rechazo_error'])} | "
                 f"{st['rechazada_confianza']} | {st['rechazada_manos']} | {st['rechazada_incompleta']} | "
                 f"{f['variantes']['si']} |")
    if not a["por_sena"]:
        L.append("| — | (sin ejecuciones válidas) | | | | | | | | | | | |")
    L.append("")

    L.append("## 5. Exactitud por repetición")
    L.append("")
    for campo, titulo in (("intento", "Por intento dentro de la presentación"), ("ronda", "Por ronda")):
        filas = a["por_repeticion"][f"por_{campo}"]
        L.append(f"**{titulo}.**")
        L.append("")
        if not filas or (len(filas) == 1 and filas[0][campo] == "sin_dato"):
            L.append(f"La hoja no trae la columna «{campo}» con valores.")
            L.append("")
            continue
        L.append(f"| {campo} | n | con predicción | aceptadas | señas distintas | top-1 (con predicción) | "
                 f"top-3 (con predicción) | top-1 (entre aceptadas) | top-1 (todo rechazo = error) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for f in filas:
            L.append(f"| {f[campo]} | {f['n_ejecuciones']} | {f['n_con_prediccion']} | "
                     f"{f['n_aceptadas']} | {f['n_senas_distintas']} | {_n(f['top1_con_prediccion'])} | "
                     f"{_n(f['top3_con_prediccion'])} | {_n(f['top1_aceptadas'])} | "
                     f"{_n(f['top1_todas_rechazo_error'])} |")
        L.append("")

    L.append("## 6. Matriz de confusión")
    L.append("")
    mr = a["matriz_confusion_referencia"]
    L.append(f"Conjunto «{mr['conjunto']}», política «{mr['politica']}», denominador "
             f"«{mr['denominador']}». La matriz completa (dispersa, solo celdas no nulas) está en el JSON, "
             f"clave `matriz_confusion_referencia.matriz`; cada conjunto y cada política tienen la suya en "
             f"`exactitud.<conjunto>.politicas.<politica>.denominadores.<denominador>.matriz_confusion`.")
    L.append("")
    errores = [p for p in mr["pares"] if not p["acierto"]]
    aciertos = [p for p in mr["pares"] if p["acierto"]]
    L.append(f"Celdas no nulas: {len(mr['pares'])} ({len(aciertos)} en la diagonal, {len(errores)} fuera).")
    L.append("")
    if errores:
        L.append("| Objetivo | Predicha | n |")
        L.append("|---|---|---|")
        for p in errores:
            L.append(f"| {p['objetivo']} {p['glosa_objetivo']} | {p['predicha']} {p['glosa_predicha']} | "
                     f"{p['n']} |")
        L.append("")

    L.append("## 7. Ensayos con audio dudoso (N-37, N-38, E7)")
    L.append("")
    ad = a["audio_dudoso"]
    t = ad["total"]
    L.append(f"De {t['n_ensayos']} ensayo(s) del JSONL, {t['informados']} informan "
             f"`tts_errores_reproduccion` y {t['sin_dato']} no lo traen (ausente o null es «sin dato», "
             f"que no es «cero errores»). Ensayos con el contador > 0: {t['ensayos_con_errores']}.")
    L.append("")
    L.append(f"- Convención A (estrecha, desde el registro en que el contador ya "
             f"subió): {t['n_convencion_a']} ensayo(s), de los cuales {ad['validas_convencion_a']} "
             f"son ejecuciones válidas.")
    L.append(f"- Convención B, la adoptada en D-13 (A más el ensayo inmediatamente anterior a cada "
             f"aumento, que es el que probablemente no sonó): {t['n_convencion_b']} ensayo(s), "
             f"{ad['validas_convencion_b']} de ellos válidos.")
    L.append("")
    L.append("Ninguno se excluye: E7 conserva estas ejecuciones en la exactitud del Bloque 1 y pide "
             "contarlas. D-13 define el audio dudoso con la convención B; qué se hace con esos ensayos "
             "(repetirlos, excluirlos o contarlos como incidencia) sigue abierto en PD-31.")
    L.append("")
    if ad["incrementos"]:
        L.append("| Sesión | Ensayo en que sube el contador | Ensayo anterior | Locuciones nuevas sin sonar "
                 "| Contador | Último mensaje |")
        L.append("|---|---|---|---|---|---|")
        for i in ad["incrementos"]:
            L.append(f"| {i['sesion']} | {i['id_ensayo']} | {i['id_ensayo_anterior']} | "
                     f"{i['incremento']} | {i['contador']} | {i['mensaje'] or '—'} |")
        L.append("")
    if ad["contador_que_baja"]:
        L.append(f"Atención: el contador baja en {len(ad['contador_que_baja'])} punto(s); dentro de una "
                 f"sesión no debería bajar.")
        L.append("")

    L.append("## 8. Control de configuración (criterio E2)")
    L.append("")
    L.append("Valores encontrados entre los ensayos emparejados. Más de un valor en una fila significa "
             "que la sesión no fue homogénea: E2 pide contar esos ensayos aparte; aquí no se excluye "
             "ninguno.")
    L.append("")
    L.append("| Clave | Valores encontrados (n ensayos) |")
    L.append("|---|---|")
    for clave, valores in a["configuracion"].items():
        detalle = ", ".join(f"{k} ({n})" for k, n in sorted(valores.items())) or "—"
        L.append(f"| {clave} | {detalle} |")
    L.append("")

    L.append("## 9. Advertencias")
    L.append("")
    if a["advertencias"]:
        L.extend(f"- {x}" for x in a["advertencias"])
    else:
        L.append("Ninguna.")
    L.append("")
    L.append("## 10. Lo que queda por decidir antes de reportar una cifra")
    L.append("")
    L.extend(f"- {x}" for x in a["pendientes_de_decision"])
    L.append("")
    return "\n".join(L)


def tabla(a: dict) -> str:
    """Resumen corto para la consola."""
    e, c, v = a["entradas"], a["cruce"], a["validez"]
    L = [f"Exactitud en linea del Bloque 1  ({a['generado']})",
         f"  Registros: {e['n_ensayos_jsonl']} ensayo(s); hoja: {e['n_filas_hoja']} fila(s); "
         f"emparejados: {c['n_emparejados']}",
         f"  Ensayos sin fila: {len(c['ensayos_sin_fila'])}   filas sin ensayo: "
         f"{len(c['filas_sin_ensayo'])}   filas con problema: {len(c['filas_con_problema'])}",
         f"  Validas: {v['validas']}   no validas: {v['no_validas']}   sin dato: {v['sin_dato']}",
         ""]
    for politica in POLITICAS:
        L.append(f"  Politica PD-14 «{politica}»")
        for cj in CONJUNTOS:
            if cj == "sin_variante_amplio" and not a["exactitud"][cj].get("aplica", True):
                continue
            b = a["exactitud"][cj]["politicas"][politica]
            if "no_calculable" in b:
                L.append(f"    {cj:<24s} no calculable (falta la columna «presentacion»)")
                continue
            d = b["denominadores"]
            L.append(f"    {cj:<24s} n={b['n_ejecuciones']:<4d} "
                     f"top1[con_pred]={_n(d['con_prediccion']['top1'])} "
                     f"top3[con_pred]={_n(d['con_prediccion']['top3'])} "
                     f"top1[aceptadas]={_n(d['aceptadas']['top1'])} "
                     f"top1[todas]={_n(d['todas_rechazo_error']['top1'])} "
                     f"rech.conf={_pct(b['rechazos']['tasa_rechazo_confianza'])}")
        L.append("")
    L.append(f"  Audio dudoso: convencion A {a['audio_dudoso']['total']['n_convencion_a']}, "
             f"convencion B {a['audio_dudoso']['total']['n_convencion_b']}")
    if a["advertencias"]:
        L.append("  Advertencias:")
        L.extend(f"    - {x}" for x in a["advertencias"])
    L.append("")
    L.append("  PD-14 y PD-15 siguen sin decidir: ninguna de estas cifras es todavia «la» exactitud "
             "del Bloque 1.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Línea de órdenes
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Exactitud en línea del Bloque 1 cruzando el JSONL del prototipo con la hoja manual "
                    "(actividad 4.a). No decide PD-14 ni PD-15: calcula y nombra todas las variantes.")
    parser.add_argument("--registros", nargs="+", default=None,
                        help="carpeta, archivos JSONL o comodines con los registros del bloque. "
                             "OBLIGATORIO: no hay valor por defecto, para no leer registros/ por error")
    parser.add_argument("--hoja", default=None, help="hoja de registro CSV del operador")
    parser.add_argument("--sesion", nargs="+", default=None,
                        help="limitar el análisis a estas sesiones del JSONL")
    parser.add_argument("--json", default=None, help="ruta del JSON de salida ('-' = stdout)")
    parser.add_argument("--md", default=None, help="ruta del informe .md de salida")
    parser.add_argument("--salida", default=None,
                        help="carpeta de salida: escribe exactitud_bloque1.json y exactitud_bloque1.md")
    parser.add_argument("--plantilla", default=None,
                        help="copiar la plantilla en blanco de la hoja en esta ruta y salir")
    args = parser.parse_args(argv)

    if args.plantilla:
        destino = escribir_plantilla(args.plantilla)
        print(f"Plantilla escrita en {destino}")
        return 0
    if not args.registros or not args.hoja:
        parser.error("hacen falta --registros y --hoja (o --plantilla)")

    a = analizar(args.registros, args.hoja, args.sesion)

    ruta_json = args.json
    ruta_md = args.md
    if args.salida:
        carpeta = Path(args.salida)
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_json = ruta_json or str(carpeta / "exactitud_bloque1.json")
        ruta_md = ruta_md or str(carpeta / "exactitud_bloque1.md")

    a_stdout = ruta_json == "-"
    if a_stdout:
        print(json.dumps(a, ensure_ascii=False, indent=2))
    else:
        print(tabla(a))
        if ruta_json:
            Path(ruta_json).parent.mkdir(parents=True, exist_ok=True)
            with open(ruta_json, "w", encoding="utf-8") as f:
                json.dump(a, f, ensure_ascii=False, indent=2)
            print(f"\nJSON: {ruta_json}")
    if ruta_md:
        Path(ruta_md).parent.mkdir(parents=True, exist_ok=True)
        Path(ruta_md).write_text(informe_md(a), encoding="utf-8")
        # Con el JSON en stdout el aviso va a stderr para no romper la salida.
        print(f"Informe: {ruta_md}", file=sys.stderr if a_stdout else sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
