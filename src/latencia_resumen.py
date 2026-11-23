"""
Resumen de los registros de latencia del prototipo (anteproyecto §2.2.7).

Lee uno o varios JSONL escritos por src/translator.py (registros/latencia_<fecha>.jsonl)
y reporta, para cada magnitud, media, desviación estándar (muestral, ddof=1), mediana y
percentil 90:

  * las seis etapas: captura (t1-t0), extracción (e_N-t1), preprocesamiento (t2-e_N),
    inferencia (t3-t2), conversión (t4-t3), síntesis (t5-t4);
  * W = t1-t0, L_total = t5-t0, L_resp = t5-t1;
  * costo del extractor por fotograma (e_i - c_i), agrupando todos los fotogramas de
    todas las ventanas (incluidas las rechazadas: la extracción sí ocurrió);
  * tasa efectiva del bucle N/(c_N - c_1) por ensayo (fórmula literal de §2.2.7);
  * contadores de estados (aceptada / rechazada_confianza / rechazada_manos /
    rechazada_incompleta) y, si los hay, los turnos del modo oración;
  * en cuántos ensayos con audio la síntesis estaba terminada al emitir la orden (t5):
    desde el 15-sep-2026 los tres modos de src/tts.py sintetizan antes de la orden; en registros
    anteriores del modo offline no lo estaba, y la etapa «síntesis» solo medía la emisión.

Las etapas completas (hasta t5) solo existen en los ensayos aceptados en modo aislado;
las estadísticas de cada magnitud se calculan sobre los ensayos que la tienen y se
reporta n en cada fila. Las magnitudes se expresan en milisegundos.

Separaciones que pidió la revisión de N-34 (15-sep-2026); en la tabla son las filas que se reportan:
  (a) «síntesis», L_total y L_resp se dan por separado para los ensayos con
      tts_sintesis_antes_de_t5 = true (la síntesis real, ≈0,45 s con `say`, está dentro de t5) y para
      el resto (false: registros anteriores a N-34, motor sin síntesis a archivo o síntesis fallida o
      fuera del tope; o sin dato: sin audio): mezclarlos promedia ≈0 ms con ≈450 ms.
      JSON: por_sintesis_antes_de_t5.{sintesis_incluida, sintesis_no_incluida}.
  (b) W y la etapa «captura» se promedian sobre las ventanas que llegaron a inferirse (inferencia
      medida: aceptadas y rechazadas por confianza), no sobre las rechazadas por manos ni las
      incompletas o abortadas (pendiente de N-14). JSON: ventanas_inferidas.
  (c) se cuentan tts_ocupado_en_t5 (otra locución sonaba al emitir la orden) y los umbrales de
      confianza usados (parametros.umbral y parametros.umbral_origen), con los estados y la tasa de
      rechazo de cada umbral. JSON: tts_ocupado_en_t5, umbrales.
  (d) errores de reproducción de la voz (N-37), por sesión y en total: cuántos ensayos informan el
      contador y cuántos llegan con algún error, el máximo alcanzado, cuántos ensayos se registraron
      desde el primero con error (audio dudoso: la locución suena en segundo plano, así que el fallo
      del ensayo k suele verse en el registro k+1) y el último mensaje. JSON: errores_voz.
  Por compatibilidad (tests/test_prototipo.py y resúmenes ya guardados), magnitudes_ms conserva el
  cálculo anterior: cada magnitud sobre todos los ensayos que la tienen, sin separar. Cuando eso mezcla
  grupos, la lista «advertencias» del JSON lo dice. Los registros anteriores sin los campos nuevos se
  leen igual: el dato ausente cuenta como «sin dato».

Uso:
    python src/latencia_resumen.py registros/latencia_2026-09-14.jsonl
    python src/latencia_resumen.py registros/*.jsonl --json resumen.json
    python src/latencia_resumen.py registros/ --json -          # JSON a stdout
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ETAPAS = ("captura", "extraccion", "preprocesamiento", "inferencia", "conversion", "sintesis")
ESTADOS = ("aceptada", "rechazada_confianza", "rechazada_manos", "rechazada_incompleta")
MAGNITUDES = ETAPAS + ("W", "L_total", "L_resp")
# (a) Magnitudes que dependen de si la síntesis quedó antes de t5.
MAGNITUDES_POR_SINTESIS = ("sintesis", "L_total", "L_resp")
GRUPOS_SINTESIS = ("sintesis_incluida", "sintesis_no_incluida")
# (e) Desde N-45 la síntesis puede venir de la cache de audio (≈0,1 ms) o sintetizarse en el momento
# (≈15 ms con el sintetizador persistente, ≈460 ms con «say»). Promediarlas juntas no significa nada.
GRUPOS_CACHE = ("desde_cache", "sintetizada", "sin_dato")
# (b) Magnitudes que se promedian solo sobre ventanas inferidas.
MAGNITUDES_VENTANA = ("captura", "W")
# (d) Campos de error de voz que escribe src/translator.py (N-37).
CLAVE_ERRORES_VOZ = "tts_errores_reproduccion"
CLAVE_ULTIMO_ERROR_VOZ = "tts_ultimo_error_reproduccion"


def leer_registros(rutas: list[str]) -> list[dict]:
    """Expande carpetas y comodines y devuelve todos los registros JSONL."""
    archivos: list[Path] = []
    for r in rutas:
        p = Path(r)
        if p.is_dir():
            archivos.extend(sorted(p.glob("*.jsonl")))
        else:
            expandidos = sorted(glob.glob(r))
            if expandidos:
                archivos.extend(Path(x) for x in expandidos)
            else:
                archivos.append(p)
    registros = []
    for a in archivos:
        if not a.is_file():
            # Ruta sin comodín que no existe: aviso en vez de traceback (revisión 14-sep).
            print(f"[AVISO] {a} no existe; se omite", file=sys.stderr)
            continue
        with open(a, encoding="utf-8") as f:
            for num, linea in enumerate(f, 1):
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    registros.append(json.loads(linea))
                except json.JSONDecodeError as exc:
                    print(f"[AVISO] {a}:{num} no es JSON válido ({exc}); se omite", file=sys.stderr)
    return registros


def estadisticas(valores_ns) -> dict | None:
    """media, DE (ddof=1), mediana, p90, min, max en ms; None si no hay datos."""
    a = np.asarray([v for v in valores_ns if v is not None], dtype=np.float64)
    if a.size == 0:
        return None
    a = a / 1e6
    return {
        "n": int(a.size),
        "media_ms": float(a.mean()),
        "de_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "mediana_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def _etapa(r: dict, nombre: str):
    return (r.get("etapas_ns") or {}).get(nombre)


def _magnitud(r: dict, nombre: str):
    """Valor en ns de una etapa o de W / L_total / L_resp; None si el ensayo no la tiene."""
    if nombre in ETAPAS:
        return _etapa(r, nombre)
    return r.get(f"{nombre}_ns")


def llego_a_inferirse(r: dict) -> bool:
    """La ventana pasó por el modelo (inferencia medida). En los registros del prototipo son las
    aceptadas y las rechazadas por confianza; nunca las rechazadas por manos ni las incompletas."""
    return _etapa(r, "inferencia") is not None or r.get("t3") is not None


def grupo_sintesis(r: dict) -> str:
    """(a) 'sintesis_incluida' solo si tts_sintesis_antes_de_t5 es true; si no (false, ausente o null),
    'sintesis_no_incluida'."""
    return "sintesis_incluida" if r.get("tts_sintesis_antes_de_t5") is True else "sintesis_no_incluida"


def grupo_cache(r: dict) -> str:
    """(e) 'desde_cache' si tts_sintesis_desde_cache es true, 'sintetizada' si es false, y 'sin_dato'
    si el registro no lo trae (todos los anteriores a N-45, y los ensayos sin audio)."""
    v = r.get("tts_sintesis_desde_cache")
    return "desde_cache" if v is True else "sintetizada" if v is False else "sin_dato"


def _umbral(r: dict) -> tuple:
    """(umbral, umbral_origen) desde parametros; se acepta también en el nivel superior."""
    parametros = r.get("parametros") if isinstance(r.get("parametros"), dict) else {}
    umbral = parametros.get("umbral", r.get("umbral"))
    origen = parametros.get("umbral_origen", r.get("umbral_origen"))
    return (float(umbral) if isinstance(umbral, (int, float)) and not isinstance(umbral, bool) else None,
            str(origen) if origen is not None else None)


def _entero(valor):
    """Entero tal como viene del JSONL; None si falta, es null o no es entero (bool no cuenta)."""
    return int(valor) if isinstance(valor, int) and not isinstance(valor, bool) else None


def _ultimo_mensaje_voz(ensayos: list[dict]) -> str | None:
    """Último tts_ultimo_error_reproduccion no nulo, en el orden en que se leyeron los registros."""
    ultimo = None
    for r in ensayos:
        mensaje = r.get(CLAVE_ULTIMO_ERROR_VOZ)
        if mensaje is not None:
            ultimo = str(mensaje)
    return ultimo


def errores_voz_de_sesion(ensayos: list[dict]) -> dict:
    """(d) Errores de voz de UNA sesión, con sus ensayos en el orden en que se registraron.

    tts_errores_reproduccion es el número de locuciones que NO sonaron, acumulado dentro de la sesión
    y leído después de sellar t5 (N-37): empieza en 0, no baja y null significa «el motor no lo
    informa», que no es «cero errores». Por eso se cuentan aparte los ensayos sin dato.

    ensayos_desde_el_primer_error incluye el ensayo en cuyo registro apareció el primer error: cuando
    ese registro se escribió el fallo ya había ocurrido, y la locución del propio ensayo suena después
    de t5.

    D-13 fija cuál de los dos conteos es «audio dudoso» para el protocolo: el ensayo cuya locución
    falló de verdad suele ser el ANTERIOR al registro donde sube el contador, porque el audio suena en
    segundo plano. Así que audio_dudoso añade ese ensayo previo. El conteo estrecho se sigue dando
    como ensayos_desde_el_primer_error, para poder distinguirlos.
    """
    n = len(ensayos)
    valores = [_entero(r.get(CLAVE_ERRORES_VOZ)) for r in ensayos]
    informados = [v for v in valores if v is not None]
    primero = next((i for i, v in enumerate(valores) if v is not None and v > 0), None)
    return {
        "n_ensayos": n,
        "informados": len(informados),
        "sin_dato": n - len(informados),
        "ensayos_con_errores": sum(1 for v in informados if v > 0),
        "maximo": max(informados) if informados else None,
        "primer_error_en_ensayo": ensayos[primero].get("id_ensayo") if primero is not None else None,
        "ensayos_desde_el_primer_error": (n - primero) if primero is not None else 0,
        # D-13: el ensayo anterior al primer registro con el contador en alto es el que probablemente
        # no sonó, así que entra en el audio dudoso.
        "audio_dudoso": (n - max(primero - 1, 0)) if primero is not None else 0,
        "ultimo_mensaje": _ultimo_mensaje_voz(ensayos),
    }


def errores_voz(ensayos: list[dict]) -> dict:
    """(d) Errores de voz por sesión (campo «sesion», en orden de aparición) y en total.

    El contador se reinicia con cada sesión del prototipo, así que el total no es un índice global:
    los conteos se suman por sesión, «maximo» es el mayor de los máximos y el último mensaje es el
    último no nulo de todos los registros leídos. Los turnos del modo oración llevan los mismos dos
    campos y no se cuentan aquí: el contador que traen es el mismo de su sesión.
    """
    grupos: dict[str, list[dict]] = {}
    for r in ensayos:
        grupos.setdefault(str(r.get("sesion")), []).append(r)
    por_sesion = [{"sesion": s, **errores_voz_de_sesion(miembros)} for s, miembros in grupos.items()]
    maximos = [b["maximo"] for b in por_sesion if b["maximo"] is not None]
    total = {
        "n_ensayos": len(ensayos),
        "informados": sum(b["informados"] for b in por_sesion),
        "sin_dato": sum(b["sin_dato"] for b in por_sesion),
        "ensayos_con_errores": sum(b["ensayos_con_errores"] for b in por_sesion),
        "maximo": max(maximos) if maximos else None,
        "ensayos_desde_el_primer_error": sum(b["ensayos_desde_el_primer_error"] for b in por_sesion),
        "audio_dudoso": sum(b["audio_dudoso"] for b in por_sesion),
        "ultimo_mensaje": _ultimo_mensaje_voz(ensayos),
        "n_sesiones": len(por_sesion),
        "sesiones_con_errores": sum(1 for b in por_sesion if b["ensayos_con_errores"]),
    }
    return {"total": total, "por_sesion": por_sesion}


def _contar_estados(ensayos: list[dict]) -> dict:
    return {e: sum(1 for r in ensayos if r.get("estado") == e) for e in ESTADOS}


def resumir(registros: list[dict]) -> dict:
    ensayos = [r for r in registros if r.get("tipo") == "ensayo"]
    turnos = [r for r in registros if r.get("tipo") == "turno"]

    contadores = _contar_estados(ensayos)
    por_modo = {m: sum(1 for r in ensayos if r.get("modo") == m) for m in ("aislado", "oracion")}

    # Cálculo anterior, sin separar (compatibilidad): cada magnitud sobre todos los ensayos que la tienen.
    magnitudes = {m: estadisticas(_magnitud(r, m) for r in ensayos) for m in MAGNITUDES}

    # (b) W y captura solo sobre ventanas que llegaron a inferirse.
    inferidas = [r for r in ensayos if llego_a_inferirse(r)]
    ventanas_inferidas = {"n": len(inferidas)}
    ventanas_inferidas.update({m: estadisticas(_magnitud(r, m) for r in inferidas) for m in MAGNITUDES_VENTANA})
    no_inferidas_con_w = sum(1 for r in ensayos if not llego_a_inferirse(r) and r.get("W_ns") is not None)

    # (a) síntesis, L_total y L_resp separadas según tts_sintesis_antes_de_t5.
    por_sintesis = {}
    for g in GRUPOS_SINTESIS:
        miembros = [r for r in ensayos if grupo_sintesis(r) == g]
        completos = [r for r in miembros if r.get("L_total_ns") is not None]
        bloque = {"n_ensayos_hasta_t5": len(completos)}
        if g == "sintesis_no_incluida":
            bloque["tts_sintesis_antes_de_t5_false"] = sum(
                1 for r in completos if r.get("tts_sintesis_antes_de_t5") is False)
            bloque["sin_dato_o_sin_audio"] = sum(
                1 for r in completos if r.get("tts_sintesis_antes_de_t5") is None)
        bloque.update({m: estadisticas(_magnitud(r, m) for r in miembros) for m in MAGNITUDES_POR_SINTESIS})
        por_sintesis[g] = bloque

    # (e) la etapa «síntesis» separada según venga de la cache de audio o se sintetice en el momento.
    por_cache = {}
    for g in GRUPOS_CACHE:
        miembros = [r for r in ensayos if grupo_cache(r) == g]
        completos = [r for r in miembros if r.get("L_total_ns") is not None]
        por_cache[g] = {"n_ensayos_hasta_t5": len(completos),
                        **{m: estadisticas(_magnitud(r, m) for r in miembros)
                           for m in MAGNITUDES_POR_SINTESIS}}

    # Costo del extractor por fotograma, agrupando todos los fotogramas de todas las ventanas.
    costos = []
    for r in ensayos:
        c, e = r.get("c") or [], r.get("e") or []
        costos.extend(ei - ci for ci, ei in zip(c, e))
    costo_extractor = estadisticas(costos)

    # Tasa efectiva por ensayo (fps): se reporta en fps, no en ms.
    tasas = np.asarray([r["tasa_efectiva_fps"] for r in ensayos if r.get("tasa_efectiva_fps")], dtype=np.float64)
    tasa_efectiva = None
    if tasas.size:
        tasa_efectiva = {
            "n": int(tasas.size),
            "media_fps": float(tasas.mean()),
            "de_fps": float(tasas.std(ddof=1)) if tasas.size > 1 else 0.0,
            "mediana_fps": float(np.median(tasas)),
            "p90_fps": float(np.percentile(tasas, 90)),
            "min_fps": float(tasas.min()),
            "max_fps": float(tasas.max()),
        }

    n_frames = np.asarray([r.get("N", 0) for r in ensayos], dtype=np.float64)
    resumen_turnos = None
    if turnos:
        resumen_turnos = {
            "n": len(turnos),
            "conversion": estadisticas(t.get("conversion_ns") for t in turnos),
            "sintesis": estadisticas(t.get("sintesis_ns") for t in turnos),
            # Misma separación (a) para el modo oración (campo tts_sintesis_antes_de_orden).
            "sintesis_incluida": estadisticas(
                t.get("sintesis_ns") for t in turnos if t.get("tts_sintesis_antes_de_orden") is True),
            "sintesis_no_incluida": estadisticas(
                t.get("sintesis_ns") for t in turnos if t.get("tts_sintesis_antes_de_orden") is not True),
            "glosas_por_turno_media": float(np.mean([t.get("n_glosas", 0) for t in turnos])),
        }

    # Con audio: ¿la síntesis quedó antes de t5 (§2.2.7) o solo se emitió la orden (offline antes del
    # 15-sep-2026, motor sin síntesis a archivo, o síntesis fallida o fuera del tope)?
    con_audio = [r for r in ensayos if r.get("tts_sintesis_antes_de_t5") is not None]
    sintesis_antes_de_t5 = {
        "con_audio": len(con_audio),
        "si": sum(1 for r in con_audio if r["tts_sintesis_antes_de_t5"]),
        "no": sum(1 for r in con_audio if not r["tts_sintesis_antes_de_t5"]),
    }

    # (c) ¿otra locución sonaba al emitir la orden? Solo en ensayos con audio; ausente = sin dato.
    tts_ocupado = {
        "con_audio": len(con_audio),
        "si": sum(1 for r in con_audio if r.get("tts_ocupado_en_t5") is True),
        "no": sum(1 for r in con_audio if r.get("tts_ocupado_en_t5") is False),
        "sin_dato": sum(1 for r in con_audio if r.get("tts_ocupado_en_t5") is None),
    }

    # (c) Umbrales de confianza usados, con sus estados: la tasa de rechazo depende del umbral.
    grupos_umbral: dict[tuple, list[dict]] = {}
    for r in ensayos:
        grupos_umbral.setdefault(_umbral(r), []).append(r)
    umbrales = []
    for (umbral, origen), miembros in sorted(grupos_umbral.items(),
                                             key=lambda kv: (kv[0][0] is None, kv[0][0] or 0.0, kv[0][1] or "")):
        cont = _contar_estados(miembros)
        umbrales.append({"umbral": umbral, "umbral_origen": origen, "n_ensayos": len(miembros),
                         "contadores": cont, "tasa_rechazo": 1.0 - cont["aceptada"] / len(miembros)})

    # (d) Errores de reproducción de la voz, por sesión y en total.
    voz = errores_voz(ensayos)

    advertencias = []
    if all(por_sintesis[g]["n_ensayos_hasta_t5"] for g in GRUPOS_SINTESIS):
        advertencias.append(
            "magnitudes_ms.sintesis/L_total/L_resp mezclan "
            f"{por_sintesis['sintesis_incluida']['n_ensayos_hasta_t5']} ensayos con la síntesis antes de t5 y "
            f"{por_sintesis['sintesis_no_incluida']['n_ensayos_hasta_t5']} sin ella: use por_sintesis_antes_de_t5")
    if no_inferidas_con_w:
        advertencias.append(
            f"magnitudes_ms.captura/W incluyen {no_inferidas_con_w} ventanas que no llegaron a inferirse: "
            "use ventanas_inferidas")
    if por_cache["desde_cache"]["n_ensayos_hasta_t5"] and por_cache["sintetizada"]["n_ensayos_hasta_t5"]:
        advertencias.append(
            f"la etapa «sintesis» mezcla {por_cache['desde_cache']['n_ensayos_hasta_t5']} ensayos servidos "
            f"desde la cache de audio y {por_cache['sintetizada']['n_ensayos_hasta_t5']} sintetizados en el "
            "momento: use por_sintesis_desde_cache")
    if len(umbrales) > 1:
        advertencias.append(f"tasa_rechazo mezcla {len(umbrales)} umbrales de confianza: use umbrales")
    if voz["total"]["ensayos_desde_el_primer_error"]:
        advertencias.append(
            f"{voz['total']['audio_dudoso']} ensayos tienen audio dudoso desde el primer error "
            f"de reproduccion de voz ({voz['total']['sesiones_con_errores']} sesion/es): su audio es dudoso, "
            "vea errores_voz")

    modelos = sorted({str(r.get("modelo")) for r in ensayos})
    sesiones = sorted({str(r.get("sesion")) for r in ensayos})
    return {
        "n_registros": len(registros),
        "n_ensayos": len(ensayos),
        "n_turnos": len(turnos),
        "sesiones": sesiones,
        "modelos": modelos,
        "tts_modos": sorted({str(r.get("tts_modo")) for r in ensayos}),
        "sintesis_antes_de_t5": sintesis_antes_de_t5,
        "tts_ocupado_en_t5": tts_ocupado,
        "errores_voz": voz,
        "umbrales": umbrales,
        "contadores": contadores,
        "tasa_rechazo": (1.0 - contadores["aceptada"] / len(ensayos)) if ensayos else None,
        "por_modo": por_modo,
        "N_fotogramas": {"media": float(n_frames.mean()), "min": int(n_frames.min()),
                         "max": int(n_frames.max())} if n_frames.size else None,
        "magnitudes_ms": magnitudes,
        "ventanas_inferidas": ventanas_inferidas,
        "por_sintesis_antes_de_t5": por_sintesis,
        "por_sintesis_desde_cache": por_cache,
        "costo_extractor_por_fotograma_ms": costo_extractor,
        "tasa_efectiva_fps": tasa_efectiva,
        "turnos": resumen_turnos,
        "advertencias": advertencias,
    }


_ETIQUETA_GRUPO = {"sintesis_incluida": "[sint. incluida]", "sintesis_no_incluida": "[sint. no incl.]"}
_ETIQUETA_CACHE = {"desde_cache": "[desde cache]", "sintetizada": "[sintetizada]", "sin_dato": "[sin dato]"}
_ANCHO = 28


def _fila(nombre: str, s: dict | None, unidad: str = "ms") -> str:
    if s is None:
        return f"{nombre:<{_ANCHO}s}{0:>5d}" + "".join(f"{'-':>10s}" for _ in range(6))
    k = unidad
    return (f"{nombre:<{_ANCHO}s}{s['n']:>5d}{s[f'media_{k}']:>10.1f}{s[f'de_{k}']:>10.1f}"
            f"{s[f'mediana_{k}']:>10.1f}{s[f'p90_{k}']:>10.1f}{s[f'min_{k}']:>10.1f}{s[f'max_{k}']:>10.1f}")


def _fmt_umbral(u: dict) -> str:
    valor = f"{u['umbral']:.2f}" if u["umbral"] is not None else "sin dato"
    origen = u["umbral_origen"] if u["umbral_origen"] is not None else "sin dato"
    return (f"  umbral {valor} (origen: {origen}): {u['n_ensayos']} ensayos, aceptada "
            f"{u['contadores']['aceptada']}, rechazo {u['tasa_rechazo']:.1%}")


def _recortar(texto: str, tope: int = 120) -> str:
    return texto if len(texto) <= tope else texto[:tope - 1] + "…"


def _lineas_errores_voz(voz: dict | None) -> list[str]:
    """(d) Una línea global, una por sesión con errores y el último mensaje. Vacío si el resumen es
    anterior a este cambio (la clave no existe)."""
    if not voz:
        return []
    t = voz["total"]
    maximo = t["maximo"] if t["maximo"] is not None else "sin dato"
    lineas = [f"Voz, errores de reproduccion (contador acumulado por sesion): "
              f"{t['ensayos_con_errores']} de {t['n_ensayos']} ensayos llegan con algun error, "
              f"{t['sin_dato']} sin dato; maximo {maximo}; "
              f"{t['audio_dudoso']} ensayos con audio dudoso (D-13: desde el anterior al primer error)"]
    for b in voz.get("por_sesion", []):
        if b["ensayos_con_errores"]:
            lineas.append(f"  sesion {b['sesion']}: {b['ensayos_con_errores']} de {b['n_ensayos']} ensayos con "
                          f"error, maximo {b['maximo']}, primer error en el ensayo "
                          f"{b['primer_error_en_ensayo']}, {b['ensayos_desde_el_primer_error']} ensayos desde ese")
    if t["ultimo_mensaje"]:
        lineas.append(f"  ultimo mensaje: {_recortar(t['ultimo_mensaje'])}")
    return lineas


def tabla(resumen: dict) -> str:
    lineas = []
    lineas.append(f"Registros: {resumen['n_registros']}  |  ensayos: {resumen['n_ensayos']}  |  "
                  f"turnos: {resumen['n_turnos']}  |  sesiones: {len(resumen['sesiones'])}")
    lineas.append("Modelos: " + ", ".join(resumen["modelos"]))
    s5 = resumen["sintesis_antes_de_t5"]
    lineas.append("TTS: " + ", ".join(resumen["tts_modos"])
                  + f"  |  ensayos con audio {s5['con_audio']}: sintesis terminada antes de t5 en {s5['si']}, "
                    f"solo orden emitida en {s5['no']}")
    oc = resumen.get("tts_ocupado_en_t5")
    if oc:
        lineas.append(f"TTS ocupado al emitir t5 (otra locucion sonando; ensayos con audio {oc['con_audio']}): "
                      f"si {oc['si']} | no {oc['no']} | sin dato {oc['sin_dato']}")
    lineas.extend(_lineas_errores_voz(resumen.get("errores_voz")))
    c = resumen["contadores"]
    tr = resumen["tasa_rechazo"]
    lineas.append(f"Estados: aceptada {c['aceptada']} | rechazada_confianza {c['rechazada_confianza']} | "
                  f"rechazada_manos {c['rechazada_manos']} | rechazada_incompleta {c['rechazada_incompleta']}"
                  + (f"  (tasa de rechazo {tr:.1%})" if tr is not None else ""))
    umbrales = resumen.get("umbrales") or []
    if umbrales:
        lineas.append("Umbral de confianza por ensayo" + (" (la tasa de rechazo global MEZCLA umbrales):"
                                                         if len(umbrales) > 1 else ":"))
        lineas.extend(_fmt_umbral(u) for u in umbrales)
    if resumen["N_fotogramas"]:
        nf = resumen["N_fotogramas"]
        lineas.append(f"Fotogramas por ventana: media {nf['media']:.1f} (min {nf['min']}, max {nf['max']})")
    lineas.append("")
    cab = (f"{'magnitud':<{_ANCHO}s}{'n':>5s}{'media':>10s}{'DE':>10s}{'mediana':>10s}{'p90':>10s}"
           f"{'min':>10s}{'max':>10s}   (ms)")
    lineas.append(cab)
    lineas.append("-" * len(cab))
    vi = resumen.get("ventanas_inferidas")
    ps = resumen.get("por_sintesis_antes_de_t5")
    m = resumen["magnitudes_ms"]
    for nombre in ETAPAS:
        if nombre == "captura" and vi is not None:
            lineas.append(_fila("captura [inferidas]", vi["captura"]))
        elif nombre == "sintesis" and ps is not None:
            lineas.extend(_fila(f"sintesis {_ETIQUETA_GRUPO[g]}", ps[g]["sintesis"]) for g in GRUPOS_SINTESIS)
            pc = resumen.get("por_sintesis_desde_cache") or {}
            lineas.extend(_fila(f"  sintesis {_ETIQUETA_CACHE[g]}", pc[g]["sintesis"])
                          for g in GRUPOS_CACHE
                          if pc.get(g) and pc[g]["n_ensayos_hasta_t5"])
        else:
            lineas.append(_fila(nombre, m[nombre]))
    lineas.append("-" * len(cab))
    lineas.append(_fila("W [inferidas]", vi["W"]) if vi is not None else _fila("W", m["W"]))
    for nombre in ("L_total", "L_resp"):
        if ps is not None:
            lineas.extend(_fila(f"{nombre} {_ETIQUETA_GRUPO[g]}", ps[g][nombre]) for g in GRUPOS_SINTESIS)
        else:
            lineas.append(_fila(nombre, m[nombre]))
    lineas.append("-" * len(cab))
    ce = resumen["costo_extractor_por_fotograma_ms"]
    if ce:
        lineas.append(_fila("extractor/fotog.", ce))
    te = resumen["tasa_efectiva_fps"]
    if te:
        lineas.append(_fila("tasa efectiva", te, unidad="fps") + "   (fps)")
    if vi is not None:
        lineas.append("Referencia, calculo anterior (toda ventana cerrada con fotogramas, inferida o no):")
        lineas.append(_fila("captura [todas]", m["captura"]))
        lineas.append(_fila("W [todas]", m["W"]))
    t = resumen["turnos"]
    if t:
        lineas.append("")
        lineas.append(f"Turnos (modo oracion): {t['n']}, glosas por turno media {t['glosas_por_turno_media']:.1f}")
        filas = [("conversion", t["conversion"])]
        if "sintesis_incluida" in t:
            filas += [(f"sintesis {_ETIQUETA_GRUPO[g]}", t[g]) for g in GRUPOS_SINTESIS]
        else:
            filas.append(("sintesis", t["sintesis"]))
        for nombre, s in filas:
            if s:
                lineas.append(f"  {nombre:<{_ANCHO - 2}s}{s['n']:>5d}{s['media_ms']:>10.1f}{s['de_ms']:>10.1f}"
                              f"{s['mediana_ms']:>10.1f}{s['p90_ms']:>10.1f}   (ms)")
    lineas.append("")
    lineas.append("DE muestral (ddof=1). Etapas completas solo en ensayos aceptados en modo aislado. "
                  "[inferidas] = ventanas que pasaron por el modelo (aceptadas y rechazadas por confianza); "
                  "extraccion se calcula sobre toda ventana cerrada con fotogramas. "
                  "[sint. incluida] = tts_sintesis_antes_de_t5 true: la sintesis real esta dentro de t5; "
                  "[sint. no incl.] = false o sin dato (registros anteriores al 15-sep-2026, motor sin sintesis "
                  "a archivo, sintesis fallida o fuera del tope, o sin audio): la etapa sintesis solo mide la orden. "
                  "Tasa efectiva = N/(c_N - c_1) (formula de §2.2.7). "
                  "Los errores de voz son locuciones que no sonaron: el contador acumulado de la sesion se lee "
                  "despues de t5, asi que un fallo suele aparecer en el registro del ensayo siguiente.")
    return "\n".join(lineas)


def main():
    parser = argparse.ArgumentParser(description="Resumen de latencias del prototipo (§2.2.7)")
    parser.add_argument("rutas", nargs="+", help="archivos JSONL, comodines o carpetas")
    parser.add_argument("--json", type=str, default=None,
                        help="escribir el resumen en JSON en esta ruta ('-' = stdout)")
    args = parser.parse_args()

    registros = leer_registros(args.rutas)
    if not registros:
        print("No se encontraron registros.", file=sys.stderr)
        sys.exit(1)
    resumen = resumir(registros)

    if args.json == "-":
        print(json.dumps(resumen, ensure_ascii=False, indent=2))
    else:
        print(tabla(resumen))
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(resumen, f, ensure_ascii=False, indent=2)
            print(f"\nJSON: {args.json}")


if __name__ == "__main__":
    main()
