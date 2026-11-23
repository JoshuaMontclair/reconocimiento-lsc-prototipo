"""
Tasa del extractor dentro y fuera de Google Drive: cabo pendiente de N-21 (actividad 1.c).

QUÉ PREGUNTA RESUELVE. N-21 midió la corrida completa de extracción (1.000 videos) en 15,0 ms
por fotograma en la etapa `holistic`, frente a 11,3-11,6 ms en las pruebas cortas de N-20, y dejó
como CAUSA PROBABLE (no verificada) la sincronización de Google Drive: el repositorio vive en
~/Library/CloudStorage y la corrida creó 5.000 archivos nuevos. N-21 anotó que «una repetición
fuera de Drive lo confirmaría». Esto es esa repetición.

QUÉ HACE. Sobre una MUESTRA determinista de videos del corpus mide el MISMO extractor de D-05
(mp.tasks.vision.HolisticLandmarker, modelo holistic_landmarker.task verificado por SHA256, las
mismas OPCIONES_HOLISTIC, modo VIDEO, delegado CPU, un landmarker nuevo por video) en tres
condiciones seguidas:
    1. drive_1 : los videos leídos desde data/raw/LSC50/VIDEOS/COLOR_BODY (dentro de Google Drive)
    2. local   : los MISMOS videos leídos desde una copia en disco local, fuera de Drive
    3. drive_2 : otra vez desde Drive, para ver si hubo deriva entre el principio y el final
El código de extracción y de resumen se IMPORTA de src/reextract_lsc50.py (extraer_video,
crear_holistic, ensure_modelo, resumen_ms, info_hardware, carga_sistema): no se reimplementa nada,
para que las cifras sean comparables con las de N-21.

QUÉ NO HACE. No escribe fotogramas, video, CSV de puntos ni nada dentro de data/ (§1.5 del
anteproyecto: no se almacenan imágenes ni video). Solo LEE los .avi y cronometra. La copia local
es una COPIA (shutil.copy2) en el directorio temporal que se indique; los originales no se mueven
ni se borran. Tampoco entrena ni toca modelos .keras.

CONVENCIONES DE MEDICIÓN (las mismas de N-21 / reextract_lsc50.py)
  · Reloj time.perf_counter_ns para todas las marcas; ms = ns / 1e6.
  · Etapas por fotograma: `decodificar` = cap.read + BGR->RGB + mp.Image (aquí es donde aparece el
    coste de LEER el archivo); `holistic` = detect_for_video; `total` = bucle completo.
  · Estadísticas por etapa con resumen_ms de reextract_lsc50: n, media, DE muestral (ddof=1),
    mediana, p90, mín, máx.
  · Tasa en fotogramas por segundo: 1000 / ms. Se dan la del bucle (media y mediana por fotograma)
    y la de pared (fotogramas / segundos de reloj de la condición, que incluye crear el landmarker
    por video y abrir cada archivo).

CONTEXTO QUE SE REGISTRA (sin él la comparación no se puede leer)
  · Procesos de Google Drive vivos, y el tiempo de CPU que consumieron DURANTE cada condición
    (delta de ps -o cputime): si es casi cero, la sincronización estaba quieta y la condición que
    N-21 describe —escritura masiva en Drive— NO se está reproduciendo.
  · Materialización de los archivos: st_blocks * 512 frente a st_size. Un archivo de Drive ya
    descargado en la caché local se lee como uno local; uno «solo en la nube» tendría bloques 0.
  · Promedio de carga del sistema (os.getloadavg) al empezar y al terminar cada condición, y el
    tiempo de CPU del propio proceso (resource.getrusage), que dice cuántos núcleos ocupó de veras.

MUESTRA (determinista y declarada). Los nombres son <seña>_<señante>_<repetición>. Con semilla
fija se eligen n/5 señas de las 50 y, para cada seña y cada uno de los 5 señantes, UNA repetición;
salen n videos equilibrados: n/5 por señante y 5 por seña (uno por señante). Con la misma semilla
sale siempre la misma lista.

Uso:
    python src/tasa_extractor_local.py --salida models/tasa_extractor_local_2026-09-16 \
        --copia-local /ruta/fuera/de/drive --n 60 --semilla 2026
    python src/tasa_extractor_local.py --n 10 --solo-preparar      # imprime la muestra y no mide
"""

import argparse
import datetime as dt
import json
import math
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Límite de CPU: este equipo puede estar grabando con la cámara mientras corre la medición, así
# que se topan los hilos de las bibliotecas numéricas ANTES de importar nada que las cargue.
# setdefault: si la corrida ya trae los valores en el entorno, se respetan.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src import reextract_lsc50 as rx  # noqa: E402  (extractor y convenciones de D-05 / N-21)

VIDEO_DIR = rx.VIDEO_DIR                       # data/raw/LSC50/VIDEOS/COLOR_BODY (dentro de Drive)
SALIDA_POR_DEFECTO = ROOT_DIR / "models" / "tasa_extractor_local_2026-09-16"
COPIA_LOCAL_POR_DEFECTO = Path.home() / "tasa_extractor_videos_local"

N_MUESTRA = 60
SEMILLA = 2026                                  # la misma semilla del resto del proyecto (N-37)
N_SENANTES = 5
ORDEN_CONDICIONES = ("drive_1", "local", "drive_2")
ETAPAS_MS = rx.ETAPAS_MS                        # ("decodificar", "holistic", "total")

# Procesos que se buscan para saber si la sincronización de Drive estaba viva. Se compara contra
# el EJECUTABLE (la primera palabra de la línea de órdenes), no contra la línea entera: el
# repositorio vive en una carpeta llamada «Google Drive», así que buscar ese texto en la línea
# completa marcaría como «de Drive» cualquier proceso cuyo directorio de trabajo esté ahí dentro.
PATRONES_DRIVE = ("/Applications/Google Drive.app/",)


# ----------------------------------------------------------------------------------------
# Muestra determinista
# ----------------------------------------------------------------------------------------

def partes_muestra(stem: str) -> tuple[int, int, int]:
    """(seña, señante, repetición) desde '0005_0002_0001'. Misma convención que
    preprocess_lsc50.parse_filename."""
    a, b, c = Path(stem).stem.split("_")[:3]
    return int(a), int(b), int(c)


def seleccionar_muestra(nombres, n: int = N_MUESTRA, semilla: int = SEMILLA,
                        n_senantes: int = N_SENANTES) -> list[str]:
    """Elige `n` muestras de forma determinista y equilibrada a partir de `nombres`.

    Criterio (declarado, reproducible con la misma semilla):
      1. Se agrupan los nombres por (seña, señante) y se ordenan.
      2. Se conservan solo las señas que tienen los `n_senantes` señantes completos.
      3. Se eligen n // n_senantes señas con random.Random(semilla).sample sobre la lista
         ORDENADA de señas.
      4. Para cada seña elegida y cada señante, se elige UNA repetición con el mismo
         generador (random.Random(semilla).choice sobre la lista ordenada de repeticiones).
    Devuelve la lista ordenada de nombres. Si n no es múltiplo de n_senantes, se toma el
    techo de señas y se recorta la lista ordenada a `n`.
    """
    if n <= 0:
        return []
    por_sena: dict[int, dict[int, list[str]]] = {}
    for nombre in nombres:
        stem = Path(nombre).stem
        try:
            sena, senante, _ = partes_muestra(stem)
        except (ValueError, IndexError):
            continue
        por_sena.setdefault(sena, {}).setdefault(senante, []).append(stem)

    completas = sorted(s for s, d in por_sena.items() if len(d) >= n_senantes)
    if not completas:
        return []
    n_senas = min(len(completas), max(1, math.ceil(n / n_senantes)))
    rng = random.Random(semilla)
    senas = sorted(rng.sample(completas, n_senas))

    elegidas: list[str] = []
    for sena in senas:
        for senante in sorted(por_sena[sena])[:n_senantes]:
            elegidas.append(rng.choice(sorted(por_sena[sena][senante])))
    return sorted(elegidas)[:n]


def describir_muestra(stems: list[str]) -> dict:
    """Composición de la muestra: señas, señantes y repeticiones que quedaron."""
    senas, senantes, repes = {}, {}, {}
    for s in stems:
        a, b, c = partes_muestra(s)
        senas[a] = senas.get(a, 0) + 1
        senantes[b] = senantes.get(b, 0) + 1
        repes[c] = repes.get(c, 0) + 1
    return {
        "n": len(stems),
        "senas": sorted(senas),
        "videos_por_sena": {str(k): v for k, v in sorted(senas.items())},
        "videos_por_senante": {str(k): v for k, v in sorted(senantes.items())},
        "videos_por_repeticion": {str(k): v for k, v in sorted(repes.items())},
        "muestras": list(stems),
    }


# ----------------------------------------------------------------------------------------
# Entorno: procesos de Drive, materialización, CPU del proceso
# ----------------------------------------------------------------------------------------

def _cputime_a_segundos(texto: str) -> float:
    """'125:59.67' o '1-02:03:04' -> segundos. 0.0 si no se puede leer."""
    texto = (texto or "").strip()
    if not texto:
        return 0.0
    dias = 0.0
    if "-" in texto:
        d, texto = texto.split("-", 1)
        dias = float(d)
    partes = texto.split(":")
    try:
        valores = [float(p) for p in partes]
    except ValueError:
        return 0.0
    seg = 0.0
    for v in valores:
        seg = seg * 60 + v
    return dias * 86400 + seg


def procesos_drive(patrones=PATRONES_DRIVE) -> list[dict]:
    """Procesos de Google Drive vivos con su tiempo de CPU acumulado (segundos) y su RSS.
    Lista vacía si no hay ninguno o si `ps` falla. No mata ni toca ningún proceso."""
    try:
        r = subprocess.run(["ps", "-Ao", "pid=,cputime=,rss=,command="],
                           capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    salida = []
    for linea in r.stdout.splitlines():
        partes = linea.strip().split(None, 3)
        if len(partes) < 4:
            continue
        pid, cputime, rss, comando = partes
        ejecutable = comando.split(" -")[0]
        if not any(p in ejecutable for p in patrones):
            continue
        salida.append({"pid": int(pid), "cpu_segundos_acumulados": round(_cputime_a_segundos(cputime), 2),
                       "rss_kb": int(rss), "comando": comando[:120]})
    return sorted(salida, key=lambda d: d["pid"])


def cpu_drive_total(procs: list[dict]) -> float:
    return round(sum(p["cpu_segundos_acumulados"] for p in procs), 2)


def cpu_drive_delta(antes: list[dict], despues: list[dict]) -> dict:
    """CPU que consumieron los procesos de Drive entre dos instantáneas, SUMANDO POR PID: si un
    proceso nace o muere entre las dos, sumar los totales daría un salto falso."""
    a = {p["pid"]: p["cpu_segundos_acumulados"] for p in antes}
    b = {p["pid"]: p["cpu_segundos_acumulados"] for p in despues}
    comunes = sorted(set(a) & set(b))
    return {
        "segundos": round(sum(b[pid] - a[pid] for pid in comunes), 2),
        "pids_comparados": comunes,
        "pids_que_nacieron": sorted(set(b) - set(a)),
        "pids_que_murieron": sorted(set(a) - set(b)),
    }


def materializacion(rutas) -> dict:
    """¿Están los archivos realmente en disco? st_blocks * 512 frente a st_size. En Google Drive
    (FileProvider) un archivo aún no descargado tiene bloques 0 y su lectura iría a la red."""
    rutas = list(rutas)
    bytes_logicos = bloques_bytes = 0
    n_sin_bloques = 0
    for p in rutas:
        try:
            st = os.stat(p)
        except OSError:
            continue
        bytes_logicos += st.st_size
        b = getattr(st, "st_blocks", 0) * 512
        bloques_bytes += b
        if b == 0:
            n_sin_bloques += 1
    return {
        "n_archivos_medidos": len(rutas),
        "bytes_logicos": bytes_logicos,
        "bytes_en_bloques": bloques_bytes,
        "archivos_sin_bloques_asignados": n_sin_bloques,
        "fraccion_materializada": (round(bloques_bytes / bytes_logicos, 4) if bytes_logicos else None),
        "nota": "bytes_en_bloques = st_blocks * 512. Si es ~= bytes_logicos el archivo ya está en "
                "disco local (en Drive, en la caché de DriveFS) y leerlo no va a la red; si hay "
                "archivos con 0 bloques, esos sí se descargarían al leerlos",
    }


def cpu_proceso() -> dict:
    """Tiempo de CPU consumido por este proceso y sus hijos (resource.getrusage)."""
    yo = resource.getrusage(resource.RUSAGE_SELF)
    hijos = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"usuario_s": yo.ru_utime + hijos.ru_utime, "sistema_s": yo.ru_stime + hijos.ru_stime}


# ----------------------------------------------------------------------------------------
# Copia local (fuera de Google Drive)
# ----------------------------------------------------------------------------------------

def copiar_a_local(stems: list[str], origen: Path, destino: Path) -> dict:
    """Copia los videos de la muestra a `destino` (shutil.copy2). NO mueve ni borra nada en
    `origen`. Si el destino ya tiene el archivo con el mismo tamaño, no lo vuelve a copiar."""
    destino.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter_ns()
    copiados = reutilizados = 0
    bytes_copiados = 0
    for stem in stems:
        src, dst = origen / f"{stem}.avi", destino / f"{stem}.avi"
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            reutilizados += 1
            continue
        shutil.copy2(src, dst)
        bytes_copiados += dst.stat().st_size
        copiados += 1
    return {"directorio": str(destino), "copiados": copiados, "reutilizados": reutilizados,
            "bytes_copiados": bytes_copiados,
            "segundos": round((time.perf_counter_ns() - t0) / 1e9, 2),
            "nota": "copia, no traslado: los originales de data/raw quedan intactos"}


# ----------------------------------------------------------------------------------------
# Estadísticas y comparación
# ----------------------------------------------------------------------------------------

def medias_por_bloque(por_video: list[dict], tam: int, etapa: str) -> list[float]:
    """Media de `etapa` por bloques consecutivos de `tam` videos, en el orden de la pasada.
    Es la serie con la que N-21 describió la subida de 12,4 a 16,0 ms."""
    salida = []
    for i in range(0, len(por_video), tam):
        bloque = [v["ms_medio"][etapa] for v in por_video[i:i + tam] if v["ms_medio"][etapa] is not None]
        if bloque:
            salida.append(round(float(np.mean(bloque)), 2))
    return salida


def fps(ms: float | None) -> float | None:
    """Fotogramas por segundo a partir de ms por fotograma."""
    if not ms:
        return None
    return round(1000.0 / ms, 2)


def estadisticas_condicion(tiempos: dict, n_fotogramas: int, segundos_pared: float) -> dict:
    """Estadísticas por etapa (resumen_ms de reextract_lsc50) y tasas en fotogramas/s."""
    ms = {k: rx.resumen_ms(tiempos[k]) for k in ETAPAS_MS}
    seg_bucle = sum(tiempos["total"]) / 1000.0
    return {
        "ms_por_fotograma": ms,
        "fotogramas_por_segundo": {
            "bucle_por_media": fps(ms["total"].get("media")),
            "bucle_por_mediana": fps(ms["total"].get("mediana")),
            "holistic_por_media": fps(ms["holistic"].get("media")),
            "pared": round(n_fotogramas / segundos_pared, 2) if segundos_pared else None,
        },
        "segundos_bucle": round(seg_bucle, 2),
        "segundos_pared": round(segundos_pared, 2),
    }


def prueba_signos(diferencias) -> dict:
    """Prueba de los signos exacta (binomial, dos colas) sobre diferencias pareadas por video.
    Los empates se descartan, como es la regla de esta prueba."""
    pos = sum(1 for d in diferencias if d > 0)
    neg = sum(1 for d in diferencias if d < 0)
    n = pos + neg
    if n == 0:
        return {"n_pares": 0, "p_valor": None,
                "nota": "sin diferencias distintas de cero: la prueba no aplica"}
    k = min(pos, neg)
    cola = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p = min(1.0, 2 * cola)
    # Un valor p exacto nunca es cero: por debajo de 1e-6, redondear a 6 decimales lo dejaría en
    # 0,0 y se leería como imposible, así que se guarda con dos cifras significativas.
    return {"n_pares": n, "veces_positiva": pos, "veces_negativa": neg,
            "p_valor": round(p, 6) if p >= 1e-6 else float(f"{p:.2g}"),
            "nota": "prueba de los signos exacta, dos colas; positiva = el primer término fue más lento"}


def comparar(a: dict, b: dict, nombre_a: str, nombre_b: str, etapa: str = "total") -> dict:
    """Compara dos condiciones en una etapa: diferencia de medias y de medianas (a - b), en ms y
    en %, más la comparación PAREADA por video (los dos pasaron por los mismos 60 videos)."""
    ms_a, ms_b = a["ms_por_fotograma"][etapa], b["ms_por_fotograma"][etapa]
    media_a, media_b = ms_a.get("media"), ms_b.get("media")
    med_a, med_b = ms_a.get("mediana"), ms_b.get("mediana")
    por_video_a = {v["muestra"]: v for v in a["por_video"]}
    por_video_b = {v["muestra"]: v for v in b["por_video"]}
    comunes = sorted(set(por_video_a) & set(por_video_b))
    difs = [por_video_a[m]["ms_medio"][etapa] - por_video_b[m]["ms_medio"][etapa] for m in comunes]
    arr = np.asarray(difs, dtype=np.float64)
    return {
        "etapa": etapa,
        "a": nombre_a, "b": nombre_b,
        "media_a_ms": media_a, "media_b_ms": media_b,
        "diferencia_media_ms": round(media_a - media_b, 3) if None not in (media_a, media_b) else None,
        "diferencia_media_pct": (round(100.0 * (media_a - media_b) / media_b, 2)
                                 if media_b else None),
        "mediana_a_ms": med_a, "mediana_b_ms": med_b,
        "diferencia_mediana_ms": round(med_a - med_b, 3) if None not in (med_a, med_b) else None,
        "fps_a": fps(media_a), "fps_b": fps(media_b),
        "pareada_por_video": {
            "n_videos": len(comunes),
            "diferencia_media_ms": round(float(arr.mean()), 3) if arr.size else None,
            "diferencia_mediana_ms": round(float(np.median(arr)), 3) if arr.size else None,
            "de_ms": round(float(arr.std(ddof=1)), 3) if arr.size > 1 else None,
            "videos_con_a_mas_lento": int((arr > 0).sum()),
            "signos": prueba_signos(difs),
        },
    }


# ----------------------------------------------------------------------------------------
# Medición de una condición
# ----------------------------------------------------------------------------------------

def medir_condicion(nombre: str, almacenamiento: str, directorio: Path, stems: list[str],
                    ruta_modelo: str, orden: int, verboso: bool = True) -> dict:
    """Corre el extractor de D-05 sobre `stems` leyéndolos de `directorio`. NO escribe nada:
    las filas de puntos se descartan en cuanto se cronometran (§1.5 del anteproyecto)."""
    rutas = [directorio / f"{s}.avi" for s in stems]
    mat = materializacion(rutas)
    carga_ini, cpu_ini, drive_ini = rx.carga_sistema(), cpu_proceso(), procesos_drive()
    t0 = time.perf_counter_ns()
    inicio = dt.datetime.now().isoformat(timespec="seconds")

    tiempos = {k: [] for k in ETAPAS_MS}
    por_video, fallidos = [], []
    bytes_leidos = 0

    for i, stem in enumerate(stems, 1):
        video = directorio / f"{stem}.avi"
        t_v0 = time.perf_counter_ns()
        holistic = rx.crear_holistic(ruta_modelo)
        try:
            res = rx.extraer_video(video, holistic, con_rostro=True)
        except Exception as e:
            print(f"  [ERROR] {nombre} {stem}: {e}")
            fallidos.append({"muestra": stem, "error": str(e)})
            continue
        finally:
            holistic.close()
        seg = (time.perf_counter_ns() - t_v0) / 1e9
        registros = res["registros"]
        for r in registros:
            for k in ETAPAS_MS:
                tiempos[k].append(r[f"{k}_ms"])
        try:
            bytes_leidos += video.stat().st_size
        except OSError:
            pass
        medios = {k: round(float(np.mean([r[f"{k}_ms"] for r in registros])), 3) for k in ETAPAS_MS} \
            if registros else {k: None for k in ETAPAS_MS}
        medianas = {k: round(float(np.median([r[f"{k}_ms"] for r in registros])), 3) for k in ETAPAS_MS} \
            if registros else {k: None for k in ETAPAS_MS}
        por_video.append({
            "muestra": stem,
            "fotogramas": len(registros),
            "fps_video": res["entrada"]["fps"],
            "bytes": video.stat().st_size if video.exists() else None,
            "ms_medio": medios,
            "ms_mediana": medianas,
            "segundos_video": round(seg, 3),
        })
        # res se descarta aquí: no se guarda ningún fotograma ni ninguna fila de puntos.
        del res
        if verboso and (i % 10 == 0 or i == len(stems)):
            print(f"  [{nombre}] {i}/{len(stems)}  holistic medio hasta aquí: "
                  f"{np.mean(tiempos['holistic']):.2f} ms", flush=True)

    segundos_pared = (time.perf_counter_ns() - t0) / 1e9
    n_fot = len(tiempos["total"])
    cpu_fin, drive_fin = cpu_proceso(), procesos_drive()
    datos = {
        "nombre": nombre,
        "orden": orden,
        "almacenamiento": almacenamiento,
        "directorio": str(directorio),
        "inicio": inicio,
        "fin": dt.datetime.now().isoformat(timespec="seconds"),
        "n_videos": len(por_video),
        "n_fotogramas": n_fot,
        "bytes_leidos": bytes_leidos,
        "megabytes_leidos": round(bytes_leidos / 2 ** 20, 1),
        "fallidos": fallidos,
        "materializacion": mat,
        "carga_sistema_1_5_15_min_inicio": carga_ini,
        "carga_sistema_1_5_15_min_fin": rx.carga_sistema(),
        "cpu_de_este_proceso": {
            "usuario_s": round(cpu_fin["usuario_s"] - cpu_ini["usuario_s"], 2),
            "sistema_s": round(cpu_fin["sistema_s"] - cpu_ini["sistema_s"], 2),
            "nucleos_medios_ocupados": round(
                ((cpu_fin["usuario_s"] - cpu_ini["usuario_s"]) +
                 (cpu_fin["sistema_s"] - cpu_ini["sistema_s"])) / segundos_pared, 2)
            if segundos_pared else None,
        },
        "google_drive": {
            "procesos_al_empezar": drive_ini,
            "procesos_al_terminar": drive_fin,
            "delta_cpu": cpu_drive_delta(drive_ini, drive_fin),
            "cpu_segundos_consumidos_durante_la_condicion":
                cpu_drive_delta(drive_ini, drive_fin)["segundos"],
            "nota": "delta del tiempo de CPU acumulado de los procesos de Google Drive entre el "
                    "principio y el final de esta condición. Cerca de cero = la sincronización "
                    "estaba quieta, es decir, NO se está reproduciendo la situación de N-21 "
                    "(5.000 archivos nuevos subiéndose)",
        },
        "por_video": por_video,
    }
    datos.update(estadisticas_condicion(tiempos, n_fot, segundos_pared))
    datos["media_holistic_por_bloques_de_10_videos"] = medias_por_bloque(por_video, 10, "holistic")
    datos["nota_bloques"] = ("media de `holistic` por bloques de 10 videos EN EL ORDEN DE LA PASADA. "
                             "N-21 vio esta serie subir de 12,4 a 16,0 ms a lo largo de sus 1.000 "
                             "videos (primeros 50: 11,8; últimos 50: 16,1); si la subida no aparece "
                             "aquí, es que no arranca en los primeros 60 videos")
    return datos


# ----------------------------------------------------------------------------------------
# Informe
# ----------------------------------------------------------------------------------------

def _fmt(x, dec=2):
    return "-" if x is None else f"{x:.{dec}f}".replace(".", ",")


def _fmt_p(p) -> str:
    """Un valor p nunca es exactamente cero: por debajo de 0,0001 se escribe como «< 0,0001»."""
    if p is None:
        return "-"
    if p < 0.0001:
        return "< 0,0001"
    return f"{p:.4f}".replace(".", ",")


def escribir_md(datos: dict, ruta: Path) -> str:
    """Informe en Markdown: tabla por condición y lectura prudente. Si la diferencia no aparece,
    el texto lo dice; no se fuerza la conclusión de N-21."""
    c = datos["condiciones"]
    m = datos["muestra"]
    L = []
    L.append("# Tasa del extractor dentro y fuera de Google Drive (cabo de N-21, actividad 1.c)")
    L.append("")
    L.append(f"Fecha de la corrida: {datos['fecha_hora']}. "
             f"Equipo: {datos['hardware']['cpu']}, {datos['hardware']['nucleos_logicos']} núcleos, "
             f"{datos['hardware']['ram_gb']} GB, {datos['hardware']['sistema_operativo']}. "
             f"mediapipe {datos['software']['mediapipe']}, delegado CPU.")
    L.append("")
    L.append("## Qué se midió")
    L.append("")
    L.append(datos["pregunta"])
    L.append("")
    L.append(f"Extractor: el de D-05 sin cambios (`src/reextract_lsc50.py`: `crear_holistic`, "
             f"`extraer_video`, `resumen_ms`), modelo `holistic_landmarker.task` "
             f"SHA256 `{datos['modelo']['sha256'][:16]}…` verificado, las mismas `OPCIONES_HOLISTIC`, "
             f"modo VIDEO, un landmarker nuevo por video. No se escribió ningún fotograma ni ninguna "
             f"fila de puntos: los resultados se descartan en cuanto se cronometran.")
    L.append("")
    L.append(f"Muestra: {m['n']} videos, semilla {datos['semilla']}, "
             f"{len(m['senas'])} señas × {len(m['videos_por_senante'])} señantes "
             f"(una repetición por par seña-señante); "
             f"{datos['totales']['fotogramas_por_pasada']} fotogramas y "
             f"{datos['totales']['megabytes_muestra']} MB por pasada.")
    L.append("")
    L.append("## Resultados")
    L.append("")
    L.append("| Condición | Orden | Almacenamiento | ms/fot. holistic (mediana / media ± DE) | "
             "ms/fot. decodificar (mediana) | ms/fot. total (media) | fot./s (media total) | fot./s (pared) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for nombre in datos["orden_condiciones"]:
        d = c[nombre]
        h, dec, tot = (d["ms_por_fotograma"]["holistic"], d["ms_por_fotograma"]["decodificar"],
                       d["ms_por_fotograma"]["total"])
        L.append(f"| {nombre} | {d['orden']} | {d['almacenamiento']} | "
                 f"{_fmt(h.get('mediana'))} / {_fmt(h.get('media'))} ± {_fmt(h.get('de'))} | "
                 f"{_fmt(dec.get('mediana'), 3)} | {_fmt(tot.get('media'))} | "
                 f"{_fmt(d['fotogramas_por_segundo']['bucle_por_media'], 1)} | "
                 f"{_fmt(d['fotogramas_por_segundo']['pared'], 1)} |")
    L.append("")
    L.append(f"n por condición: {datos['totales']['fotogramas_por_pasada']} fotogramas "
             f"({m['n']} videos). DE muestral (ddof=1).")
    L.append("")
    L.append("### Media de `holistic` por bloques de 10 videos, en el orden de la pasada")
    L.append("")
    L.append("N-21 vio esta serie subir de 12,4 a 16,0 ms a lo largo de sus 1.000 videos "
             "(primeros 50: 11,8; últimos 50: 16,1). Aquí, con 60 videos por pasada:")
    L.append("")
    for nombre in datos["orden_condiciones"]:
        serie = c[nombre].get("media_holistic_por_bloques_de_10_videos") or []
        L.append(f"- `{nombre}`: {', '.join(_fmt(x) for x in serie)} ms")
    L.append("")
    L.append("### Comparaciones (diferencia = primera condición − segunda)")
    L.append("")
    L.append("| Comparación | Etapa | Δ media (ms) | Δ media (%) | Δ mediana (ms) | "
             "Δ pareada por video, media (ms) | videos con la 1.ª más lenta | p (signos) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for comp in datos["comparaciones"]:
        pv = comp["pareada_por_video"]
        L.append(f"| {comp['a']} vs {comp['b']} | {comp['etapa']} | "
                 f"{_fmt(comp['diferencia_media_ms'], 3)} | {_fmt(comp['diferencia_media_pct'])} | "
                 f"{_fmt(comp['diferencia_mediana_ms'], 3)} | {_fmt(pv['diferencia_media_ms'], 3)} | "
                 f"{pv['videos_con_a_mas_lento']}/{pv['n_videos']} | "
                 f"{_fmt_p(pv['signos'].get('p_valor'))} |")
    L.append("")
    L.append("## Contexto del sistema durante la medición")
    L.append("")
    L.append("| Condición | Carga al empezar | Carga al terminar | CPU de Google Drive durante (s) | "
             "Núcleos medios de este proceso | Archivos ya en disco |")
    L.append("|---|---|---|---|---|---|")
    for nombre in datos["orden_condiciones"]:
        d = c[nombre]
        ci = d["carga_sistema_1_5_15_min_inicio"]
        cf = d["carga_sistema_1_5_15_min_fin"]
        mat = d["materializacion"]
        L.append(f"| {nombre} | {ci[0] if ci else '-'} | {cf[0] if cf else '-'} | "
                 f"{_fmt(d['google_drive']['cpu_segundos_consumidos_durante_la_condicion'])} | "
                 f"{_fmt(d['cpu_de_este_proceso']['nucleos_medios_ocupados'])} | "
                 f"{mat['n_archivos_medidos'] - mat['archivos_sin_bloques_asignados']}"
                 f"/{mat['n_archivos_medidos']} |")
    L.append("")
    L.append("## Lectura")
    L.append("")
    for p in datos["lectura"]:
        L.append(f"- {p}")
    L.append("")
    L.append("## Límites de esta medición")
    L.append("")
    for p in datos["limitaciones"]:
        L.append(f"- {p}")
    L.append("")
    L.append(f"Cifras completas, muestra y entorno: `{ruta.parent.name}/tasa.json`.")
    texto = "\n".join(L) + "\n"
    ruta.write_text(texto, encoding="utf-8")
    return texto


def redactar_limitaciones(datos: dict) -> list[str]:
    """Límites declarados de esta medición. Se redactan a partir del propio JSON para que
    --rehacer-informe los vuelva a escribir sin tener que medir otra vez."""
    return [
        "Muestra de {} videos ({} fotogramas por pasada), no los 1.000 de N-21.".format(
            datos["muestra"]["n"], datos["condiciones"]["local"]["n_fotogramas"]),
        "Se pidieron 2 hilos a OMP/OPENBLAS/MKL porque el equipo podía estar grabando. MediaPipe "
        "no obedece esas variables (usa su propio grupo de hilos), así que el tope no estranguló "
        "al extractor; aun así, los ms absolutos de esta corrida se dan con esa salvedad y lo que "
        "de verdad se compara es la diferencia entre las tres condiciones, medidas seguidas y en "
        "las mismas condiciones.",
        "No se vacía la caché de disco entre condiciones (haría falta permiso de administrador y "
        "no se toca el equipo): tras la copia, los archivos de las dos condiciones están tibios en "
        "la caché del sistema, así que esta medición no dice qué pasaría con una lectura en frío.",
        "Los archivos de Drive ya estaban descargados en la caché local de DriveFS: 'desde Drive' "
        "significa a través del proveedor de archivos, no bajando de la red.",
        "La corrida de N-21 ESCRIBÍA 5.000 archivos dentro de Drive; esta solo lee. Si el coste de "
        "N-21 venía de la subida (CPU del proceso de sincronización), esta medición no lo reproduce "
        "y por tanto no puede confirmarlo ni descartarlo.",
        "Una sola repetición por condición: la deriva drive_1 → drive_2 es la única estimación de "
        "la variabilidad entre pasadas.",
        "Las dos rutas viven en el MISMO volumen APFS: no hay un disco ni un punto de montaje "
        "distinto para la carpeta de Drive, así que lo que se compara es el camino de lectura, no "
        "el soporte físico.",
        "La copia se hace con shutil.copy2 y el sistema de archivos es APFS: puede ser una copia "
        "por escritura diferida (clon), de modo que los bloques físicos se compartan con el "
        "original. Lo que sí cambia con seguridad es el CAMINO DE LECTURA (el proveedor de "
        "archivos de DriveFS frente al sistema de archivos normal), que es justo lo que se quería "
        "comparar; pero no se puede afirmar que sean bloques distintos en el disco.",
        "La prueba de los signos (dos colas, exacta) usa una decisión por video: si su media fue "
        "mayor o menor. Con 60 videos pareados detecta diferencias diminutas, así que un p bajo "
        "no significa que la diferencia importe; por eso se fijó ANTES de medir un umbral en ms "
        "por fotograma y la conclusión se lee contra ese umbral, no contra el p.",
    ]


def redactar_lectura(datos: dict) -> list[str]:
    """Lectura prudente, escrita a partir de las cifras medidas. No fuerza la conclusión de N-21:
    si la diferencia entre Drive y local no aparece, lo dice."""
    c = datos["condiciones"]
    comp_dl = next((x for x in datos["comparaciones"]
                    if x["a"] == "drive_1" and x["b"] == "local" and x["etapa"] == "holistic"), None)
    comp_dl_tot = next((x for x in datos["comparaciones"]
                        if x["a"] == "drive_1" and x["b"] == "local" and x["etapa"] == "total"), None)
    comp_dec = next((x for x in datos["comparaciones"]
                     if x["a"] == "drive_1" and x["b"] == "local" and x["etapa"] == "decodificar"), None)
    comp_der = next((x for x in datos["comparaciones"]
                     if x["a"] == "drive_2" and x["b"] == "drive_1" and x["etapa"] == "holistic"), None)
    comp_dl2 = next((x for x in datos["comparaciones"]
                     if x["a"] == "drive_2" and x["b"] == "local" and x["etapa"] == "holistic"), None)
    L = []
    if comp_dl:
        d, pct = comp_dl["diferencia_media_ms"], comp_dl["diferencia_media_pct"]
        p = comp_dl["pareada_por_video"]["signos"].get("p_valor")
        signo = "más lento" if (d or 0) > 0 else "más rápido"
        L.append(f"Leer los videos desde Google Drive frente a leerlos desde disco local cambia la etapa "
                 f"`holistic` en {abs(d):.3f} ms por fotograma ({abs(pct):.2f} %, Drive {signo}); "
                 f"la comparación pareada por video da p = {p} en la prueba de los signos.")
    if comp_dl2:
        d2 = comp_dl2["diferencia_media_ms"]
        p2 = comp_dl2["pareada_por_video"]["signos"].get("p_valor")
        gana = comp_dl2["pareada_por_video"]["videos_con_a_mas_lento"]
        n2 = comp_dl2["pareada_por_video"]["n_videos"]
        if p2 is not None and p2 < 0.05 and d2:
            cola = (f" Sale distinta de cero en la prueba de los signos, pero mide {abs(d2):.3f} ms, "
                    f"unas {3.5 / abs(d2):.0f} veces menos que los 3,5 ms que N-21 tiene que explicar: "
                    f"detectable no es lo mismo que relevante, y parte de esa diferencia es deriva "
                    f"(punto siguiente).")
        else:
            cola = " No se distingue de cero."
        L.append(f"La segunda pasada desde Drive (drive_2) frente a local da {d2:+.3f} ms por fotograma "
                 f"({gana}/{n2} videos más lentos desde Drive, p = {p2})." + cola)
    if comp_dec:
        d = comp_dec["diferencia_media_ms"]
        L.append(f"En la etapa donde de verdad se lee el archivo (`decodificar`) la diferencia es "
                 f"{d:+.3f} ms por fotograma. Es ahí, y no en `holistic`, donde tendría que aparecer un "
                 f"efecto del almacenamiento.")
    if comp_dl_tot:
        d = comp_dl_tot["diferencia_media_ms"]
        umbral = datos["umbral_declarado_ms"]
        if abs(d) < umbral:
            L.append(f"La diferencia total ({d:+.3f} ms por fotograma) queda por debajo del umbral de "
                     f"{umbral} ms que se fijó como «relevante» antes de medir: con esta muestra y en "
                     f"estas condiciones NO se reproduce la brecha de 3,5 ms que separaba las pruebas "
                     f"cortas de N-20 (11,5 ms) de la corrida completa de N-21 (15,0 ms). La causa "
                     f"probable que N-21 dejó anotada —la sincronización de Drive— no queda confirmada "
                     f"por esta medición; tampoco queda descartada, porque aquí no se reproduce la "
                     f"escritura masiva de 5.000 archivos que la corrida de N-21 sí provocaba.")
        else:
            L.append(f"La diferencia total ({d:+.3f} ms por fotograma) supera el umbral de {umbral} ms "
                     f"fijado antes de medir: el almacenamiento sí deja huella medible en esta muestra.")
    if comp_der:
        d = comp_der["diferencia_media_ms"]
        L.append(f"Deriva: la repetición desde Drive al final (drive_2) difiere de la primera (drive_1) "
                 f"en {d:+.3f} ms por fotograma en `holistic`. Esa cifra es la escala de lo que se mueve "
                 f"solo por el paso del tiempo y el calentamiento del equipo, y hay que compararla con la "
                 f"diferencia entre Drive y local antes de atribuirle nada al almacenamiento.")
    drive_cpu = [c[n]["google_drive"]["cpu_segundos_consumidos_durante_la_condicion"]
                 for n in datos["orden_condiciones"]]
    L.append(f"Los procesos de Google Drive consumieron {drive_cpu} segundos de CPU durante cada "
             f"condición (en el orden {list(datos['orden_condiciones'])}). "
             + ("Con esos valores tan bajos, la sincronización estaba prácticamente quieta: la corrida "
                "de N-21 escribía 5.000 archivos y esta no escribe ninguno, así que la comparación de "
                "hoy aísla el efecto de LEER desde Drive, no el de sincronizar."
                if max(drive_cpu) < 30 else
                "La sincronización estuvo activa durante la medición, así que estas cifras no aíslan "
                "limpiamente el efecto de leer desde Drive."))
    mat = c[datos["orden_condiciones"][0]]["materializacion"]
    if mat.get("fraccion_materializada") is not None and mat["fraccion_materializada"] > 0.9:
        L.append(f"Los videos de la muestra ya estaban descargados en el disco "
                 f"({mat['fraccion_materializada']:.0%} de los bytes con bloques asignados; "
                 f"{mat['archivos_sin_bloques_asignados']} archivos sin bloques). Leer «desde Drive» "
                 f"aquí significa leer a través del proveedor de archivos de DriveFS, no bajar nada de "
                 f"la red: ese es el escenario real del proyecto, pero conviene decirlo.")
    bloques = c["drive_1"].get("media_holistic_por_bloques_de_10_videos") or []
    if bloques:
        L.append(f"Dentro de la primera pasada, la media de `holistic` por bloques de 10 videos fue "
                 f"{bloques} ms (de {min(bloques)} a {max(bloques)}). N-21 describió esa misma serie "
                 f"subiendo de 12,4 a 16,0 ms a lo largo de sus 1.000 videos; en 60 videos "
                 + ("esa subida no aparece." if (bloques[-1] - bloques[0]) < 1.0 else
                    f"ya se ve un ascenso de {bloques[-1] - bloques[0]:+.2f} ms."))
    p90 = {n: c[n]["ms_por_fotograma"]["holistic"].get("p90") for n in datos["orden_condiciones"]}
    mx = {n: c[n]["ms_por_fotograma"]["holistic"].get("max") for n in datos["orden_condiciones"]}
    L.append(f"El grueso de la distribución es el mismo en las tres condiciones (p90 de `holistic`: "
             f"{p90}). Los máximos sí difieren ({mx}), pero con un solo máximo por condición y una "
             f"sola repetición eso no distingue un efecto del almacenamiento de un pico cualquiera "
             f"del sistema: una revisión independiente repitió la medición y el pico mayor le salió "
             f"en la condición local. No se lee nada de esos máximos.")
    nucleos = [c[n]["cpu_de_este_proceso"]["nucleos_medios_ocupados"] for n in datos["orden_condiciones"]]
    m_local = c["local"]["ms_por_fotograma"]["holistic"].get("media")
    tot_local = c["local"]["ms_por_fotograma"]["total"].get("media")
    L.append(f"Cifra del extractor fuera de Drive en esta corrida: {m_local:.2f} ms por fotograma en "
             f"`holistic` ({fps(m_local):.1f} fot./s) y {tot_local:.2f} ms en el bucle completo "
             f"({fps(tot_local):.1f} fot./s).")
    L.append(f"Sobre el tope de CPU: se pidieron 2 hilos a OMP/OpenBLAS/MKL, pero MediaPipe usa su "
             f"propio grupo de hilos y no los obedece; el proceso ocupó {nucleos} núcleos de media en "
             f"cada condición. "
             + ("Es decir, el tope no estranguló al extractor y las cifras están en la misma escala "
                "que las de N-20 y N-21."
                if min(nucleos) >= 1.3 else
                "Con tan poca CPU ocupada conviene tratar los ms absolutos como un techo y quedarse "
                "solo con la diferencia entre condiciones."))
    return L


# ----------------------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------------------

def parsear_argumentos(argv=None):
    p = argparse.ArgumentParser(
        description="Tasa del extractor de D-05 dentro y fuera de Google Drive (cabo de N-21, actividad 1.c)")
    p.add_argument("--videos", type=Path, default=VIDEO_DIR,
                   help="directorio de los videos dentro de Google Drive (solo lectura)")
    p.add_argument("--copia-local", type=Path, default=COPIA_LOCAL_POR_DEFECTO,
                   help="directorio FUERA de Google Drive donde se copia la muestra")
    p.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO,
                   help="directorio del informe (tasa.json y tasa.md)")
    p.add_argument("--n", type=int, default=N_MUESTRA, help="tamaño de la muestra (por defecto 60)")
    p.add_argument("--semilla", type=int, default=SEMILLA, help="semilla de la muestra (por defecto 2026)")
    p.add_argument("--umbral-ms", type=float, default=1.0,
                   help="diferencia en ms por fotograma que se considera relevante; se fija ANTES de medir")
    p.add_argument("--rehacer-informe", action="store_true",
                   help="no mide: relee <salida>/tasa.json, vuelve a redactar la lectura y reescribe "
                        "tasa.md y tasa.json (para corregir la redacción sin repetir la corrida)")
    p.add_argument("--solo-preparar", action="store_true",
                   help="imprime la muestra elegida y termina, sin medir ni copiar")
    p.add_argument("--nota", default="", help="texto libre que se guarda en tasa.json")
    return p.parse_args(argv)


def main(argv=None):
    args = parsear_argumentos(argv)
    t_pared = time.perf_counter_ns()

    print("=" * 78)
    print("  Tasa del extractor dentro y fuera de Google Drive (cabo de N-21)")
    print("=" * 78, flush=True)

    if args.rehacer_informe:
        ruta_json = args.salida / "tasa.json"
        datos = json.loads(ruta_json.read_text(encoding="utf-8"))
        datos["limitaciones"] = redactar_limitaciones(datos)
        datos["lectura"] = redactar_lectura(datos)
        datos["informe_rehecho"] = dt.datetime.now().isoformat(timespec="seconds")
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)
        escribir_md(datos, args.salida / "tasa.md")
        print(f"  Informe rehecho desde {ruta_json} (no se volvió a medir nada).")
        for linea in datos["lectura"]:
            print("  ·", linea)
        return 0

    videos = sorted(p.stem for p in args.videos.glob("*.avi"))
    if not videos:
        print("[ERROR] No hay videos en", args.videos)
        return 1
    stems = seleccionar_muestra(videos, n=args.n, semilla=args.semilla)
    if len(stems) < args.n:
        print(f"[AVISO] solo se pudieron elegir {len(stems)} muestras de las {args.n} pedidas")
    comp = describir_muestra(stems)
    print(f"\n  Corpus: {len(videos)} videos. Muestra: {len(stems)} "
          f"(semilla {args.semilla}); por señante {comp['videos_por_senante']}", flush=True)
    if args.solo_preparar:
        for s in stems:
            print("   ", s)
        return 0

    ruta_modelo, sha_modelo = rx.ensure_modelo()
    copia = copiar_a_local(stems, args.videos, args.copia_local)
    print(f"  Copia local: {copia['copiados']} copiados, {copia['reutilizados']} reutilizados, "
          f"{copia['bytes_copiados'] / 2**20:.1f} MB, {copia['segundos']} s", flush=True)

    condiciones = {}
    plan = [("drive_1", "google_drive_cloudstorage", args.videos),
            ("local", "disco_local_fuera_de_drive", args.copia_local),
            ("drive_2", "google_drive_cloudstorage", args.videos)]
    for orden, (nombre, almacen, carpeta) in enumerate(plan, 1):
        print(f"\n  --- Condición {orden}/3: {nombre} ({almacen}) ---", flush=True)
        condiciones[nombre] = medir_condicion(nombre, almacen, carpeta, stems, ruta_modelo, orden)

    comparaciones = []
    for etapa in ("holistic", "decodificar", "total"):
        comparaciones.append(comparar(condiciones["drive_1"], condiciones["local"],
                                      "drive_1", "local", etapa))
        comparaciones.append(comparar(condiciones["drive_2"], condiciones["local"],
                                      "drive_2", "local", etapa))
        comparaciones.append(comparar(condiciones["drive_2"], condiciones["drive_1"],
                                      "drive_2", "drive_1", etapa))

    salida = args.salida
    salida.mkdir(parents=True, exist_ok=True)
    datos = {
        "descripcion": "Tasa del extractor de puntos característicos (MediaPipe HolisticLandmarker, "
                       "D-05) medida sobre la misma muestra leída desde Google Drive y desde disco "
                       "local: cabo pendiente de N-21, actividad 1.c / §2.2.3 del anteproyecto.",
        "pregunta": "N-21 midió 15,0 ms por fotograma en la corrida completa frente a 11,3-11,6 ms en "
                    "las pruebas cortas de N-20 y dejó como causa probable, sin verificar, la "
                    "sincronización de Google Drive (el repositorio vive en CloudStorage y la corrida "
                    "creó 5.000 archivos). Esta medición repite el extractor sobre una muestra leída "
                    "desde Drive y desde una copia fuera de Drive, para separar el efecto del "
                    "almacenamiento del efecto del extractor.",
        "estado": "completado",
        "fecha_hora": dt.datetime.now().isoformat(timespec="seconds"),
        "comando": [sys.argv[0]] + sys.argv[1:],
        "git_commit": rx.commit_git(),
        "nota": args.nota or None,
        "hardware": rx.info_hardware(),
        "software": {"python": platform.python_version(), "mediapipe": rx.mp.__version__,
                     "opencv": rx.cv2.__version__, "numpy": np.__version__},
        "reloj": "time.perf_counter_ns (monótono) para todas las marcas; ms = ns / 1e6. Las "
                 "estadísticas por etapa usan resumen_ms de src/reextract_lsc50.py (N-21).",
        "limites_de_cpu": {v: os.environ.get(v) for v in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "modelo": {
            "archivo": rx.MODELO_HOLISTIC,
            "ruta": str(Path(ruta_modelo).relative_to(ROOT_DIR)) if str(ruta_modelo).startswith(str(ROOT_DIR)) else str(ruta_modelo),
            "bytes": os.path.getsize(ruta_modelo),
            "sha256": sha_modelo,
            "sha256_esperado_d05": rx.SHA256_HOLISTIC,
            "opciones": {"tarea": "HolisticLandmarker", "running_mode": "VIDEO",
                         **rx.OPCIONES_HOLISTIC, "delegate": "CPU"},
            "procedencia": "importado sin cambios de src/reextract_lsc50.py (crear_holistic, "
                           "extraer_video, resumen_ms): el mismo extractor y las mismas convenciones "
                           "de medición de N-21",
        },
        "referencias_no_medidas_aqui": {
            "nota": "cifras copiadas del registro de decisiones para poder contrastar; NO se midieron "
                    "en esta corrida",
            "N-20_pruebas_cortas_holistic_ms": 11.5,
            "N-21_corrida_completa_holistic_ms": 15.01,
            "N-21_corrida_completa_total_ms": 15.8,
            "N-21_medias_por_bloques_de_100_videos_ms": [12.4, 14.3, 14.3, 14.8, 15.7, 15.1, 15.8,
                                                         15.5, 16.0, 16.0],
            "N-21_causa_probable": "sincronización de Google Drive durante la escritura de 5.000 "
                                   "archivos nuevos (declarada como probable, sin verificar)",
        },
        "semilla": args.semilla,
        "umbral_declarado_ms": args.umbral_ms,
        "muestra": {
            **comp,
            "criterio": f"{comp['n']} videos: se eligen n/{N_SENANTES} señas con "
                        f"random.Random({args.semilla}).sample sobre las señas ordenadas que tienen los "
                        f"{N_SENANTES} señantes, y para cada seña y cada señante UNA repetición con "
                        f"random.Random(...).choice sobre las repeticiones ordenadas. Determinista: la "
                        f"misma semilla da siempre la misma lista.",
        },
        "copia_local": copia,
        "orden_condiciones": list(ORDEN_CONDICIONES),
        "diseno": "tres pasadas seguidas sobre los MISMOS videos, en el orden drive_1 → local → "
                  "drive_2; la tercera repite la primera para medir la deriva del equipo y que el "
                  "calentamiento no se confunda con el efecto del almacenamiento.",
        "no_se_escribe_nada": "no se guardan fotogramas, video ni filas de puntos (§1.5 del "
                              "anteproyecto): los videos solo se leen y se cronometran. La copia local "
                              "es una copia de los .avi en un directorio temporal fuera del repositorio; "
                              "data/ no se modifica.",
        "totales": {
            "fotogramas_por_pasada": condiciones["local"]["n_fotogramas"],
            "megabytes_muestra": round(condiciones["local"]["bytes_leidos"] / 2 ** 20, 1),
            "bytes_muestra": condiciones["local"]["bytes_leidos"],
            "bytes_leidos_en_total": sum(condiciones[n]["bytes_leidos"] for n in ORDEN_CONDICIONES),
            "segundos_de_la_corrida": round((time.perf_counter_ns() - t_pared) / 1e9, 2),
        },
        "condiciones": condiciones,
        "comparaciones": comparaciones,
    }
    datos["limitaciones"] = redactar_limitaciones(datos)
    datos["lectura"] = redactar_lectura(datos)

    ruta_json = salida / "tasa.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    escribir_md(datos, salida / "tasa.md")

    print("\n" + "=" * 78)
    for linea in datos["lectura"]:
        print("  ·", linea)
    print("=" * 78)
    print(f"  Escritos: {ruta_json}  y  {salida / 'tasa.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
