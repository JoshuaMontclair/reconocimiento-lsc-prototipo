"""
Brazo M del experimento de arquitecturas (docs/EXPERIMENTO_ARQUITECTURAS_LITERATURA.md): MiniRocket
multivariado + clasificador ridge (Dempster et al. 2021 [9]; Pasos Ruiz et al. 2021 [8]).

Se ejecuta en un entorno APARTE con sktime (no en el del proyecto, que no lo tiene y no se modifica):
    uv venv <scratch>/venv_minirocket && uv pip install --python <scratch>/venv_minirocket/bin/python sktime numba
Los tensores de cada (semilla, pliegue) los exporta src/experimento_arquitecturas.py exportar desde el entorno
de TensorFlow: entrenamiento aumentado ×10 con velocidades (30 × 252), idéntico al que ven las redes con
esa semilla; validación y prueba sin aumentar. Este script no importa TensorFlow ni el resto de src/.

Configuración fijada antes de ver resultados (CONFIG_M):
  · MiniRocketMultivariate(num_kernels=10000, max_dilations_per_kernel=32, random_state=semilla) de sktime
    sobre (n, 252 canales, 30 pasos); se ajusta (sesgos) con el entrenamiento aumentado.
  · StandardScaler ajustado solo con el entrenamiento.
  · RidgeClassifier con alpha elegido por validación cruzada interna AGRUPADA POR SEÑANTE sobre el
    entrenamiento (GroupKFold con los 4 señantes de entrenamiento; las copias aumentadas heredan el señante
    de su original, así ninguna copia de un video queda a ambos lados). En cada partición interna el
    escalado se reajusta con su parte de entrenamiento. Rejilla: 17 valores en logspace(-3, 5). Gana la
    mayor exactitud interna media; en empate, el alpha mayor. Luego se reajusta con todo el entrenamiento.
  · La validación del pliegue (2 por seña) NO interviene en el ajuste: solo se mide (es la que decide el
    criterio). La prueba se evalúa una vez.

Subcomandos:
  correr     (semilla, pliegue) que falten; reanudable.
  latencia   ms por ventana con lote de 1 (transformación + escalado + ridge), sin entrenamientos activos.
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT = BASE_DIR / "models" / "arquitecturas_literatura_2026-10-02"
CARPETA = "M_minirocket"
CONFIG_M = {"num_kernels": 10000, "max_dilations_per_kernel": 32, "escalado": "StandardScaler (solo entrenamiento)",
            "clasificador": "RidgeClassifier (fit_intercept=True)",
            "alphas": [float(a) for a in np.logspace(-3, 5, 17)],
            "cv_interna": "GroupKFold por senante (4 senantes de entrenamiento), escalado reajustado por particion",
            "desempate": "alpha mayor"}


def _rel(p):
    p = Path(p)
    return str(p.relative_to(BASE_DIR)) if p.is_relative_to(BASE_DIR) else str(p)


def metricas(scores, y):
    from sklearn.metrics import precision_recall_fscore_support
    pred = np.argmax(scores, axis=1)
    top3 = np.argsort(-scores, axis=1)[:, :3]
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(scores.shape[1])), average="macro",
                                                 zero_division=0)
    return {"top1": float(np.mean(pred == y)), "top3": float(np.mean(np.any(top3 == y[:, None], axis=1))),
            "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f)}


def exactitud_ridge_todas_alphas(F_a, y_a, F_b, y_b, alphas, n_clases):
    """Exactitud en (F_b, y_b) de RidgeClassifier(alpha, fit_intercept=True) ajustado en (F_a, y_a) para
    cada alpha, con una sola descomposición espectral del núcleo lineal (n_a × n_a). Equivale a sklearn
    (se coteja en la prueba de humo). F_a y F_b ya escalados con las estadísticas de F_a."""
    mu = F_a.mean(axis=0)
    Xa = F_a - mu
    Xb = F_b - mu
    Y = -np.ones((len(y_a), n_clases))
    Y[np.arange(len(y_a)), y_a] = 1.0
    ym = Y.mean(axis=0)
    Yc = Y - ym
    K = Xa @ Xa.T
    lam, Q = np.linalg.eigh(K)
    lam = np.clip(lam, 0, None)
    QtY = Q.T @ Yc
    Kb = Xb @ Xa.T                       # n_b × n_a
    KbQ = Kb @ Q
    acc = []
    for a in alphas:
        dual = QtY / (lam + a)[:, None]
        scores = KbQ @ dual + ym
        acc.append(float(np.mean(np.argmax(scores, axis=1) == y_b)))
    return acc


def elegir_alpha(F, y, g, alphas, n_clases):
    from sklearn.preprocessing import StandardScaler
    grupos = sorted(set(g.tolist()))
    por_grupo = []
    for gr in grupos:
        a, b = g != gr, g == gr
        sc = StandardScaler().fit(F[a])
        por_grupo.append(exactitud_ridge_todas_alphas(sc.transform(F[a]), y[a], sc.transform(F[b]), y[b],
                                                      alphas, n_clases))
    media = np.mean(np.array(por_grupo), axis=0)
    mejor = float(np.max(media))
    idx = max(i for i, v in enumerate(media) if abs(v - mejor) < 1e-12)   # empate -> alpha mayor
    return float(alphas[idx]), {"exactitud_interna_media_por_alpha": media.tolist(),
                                "exactitud_interna_por_senante": [list(map(float, x)) for x in por_grupo],
                                "senantes_internos": [int(x) for x in grupos]}


def correr_uno(ruta_npz, out, n_jobs):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    D = np.load(ruta_npz)
    s, k = int(D["semilla"]), int(D["pliegue"])
    d = Path(out) / CARPETA / f"semilla_{s}" / f"pliegue_{k}"
    if (d / "resultado.json").exists():
        print(f"  ya existe {_rel(d)}; se salta", flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    n_clases = 50
    Xtr = np.ascontiguousarray(D["X_tr"].transpose(0, 2, 1))     # (n, 252, 30)
    Xva = np.ascontiguousarray(D["X_val"].transpose(0, 2, 1))
    Xte = np.ascontiguousarray(D["X_te"].transpose(0, 2, 1))
    ytr, yva, yte, gtr = D["y_tr"], D["y_val"], D["y_te"], D["g_tr"]
    mr = MiniRocketMultivariate(num_kernels=CONFIG_M["num_kernels"],
                                max_dilations_per_kernel=CONFIG_M["max_dilations_per_kernel"],
                                random_state=s, n_jobs=n_jobs)
    t = time.perf_counter()
    Ftr = np.asarray(mr.fit_transform(Xtr), dtype=np.float64)
    t_transform = time.perf_counter() - t
    t = time.perf_counter()
    alpha, cv = elegir_alpha(Ftr, ytr, gtr, CONFIG_M["alphas"], n_clases)
    t_cv = time.perf_counter() - t
    t = time.perf_counter()
    sc = StandardScaler().fit(Ftr)
    clf = RidgeClassifier(alpha=alpha).fit(sc.transform(Ftr), ytr)
    t_final = time.perf_counter() - t
    tiempo_ent = time.perf_counter() - t0
    s_va = clf.decision_function(sc.transform(np.asarray(mr.transform(Xva), dtype=np.float64)))
    mv = metricas(s_va, yva)
    s_te = clf.decision_function(sc.transform(np.asarray(mr.transform(Xte), dtype=np.float64)))   # una vez
    mt = metricas(s_te, yte)
    pred = np.argmax(s_te, axis=1)
    top3 = np.argsort(-s_te, axis=1)[:, :3]
    n_rasgos = int(Ftr.shape[1])
    with open(d / "modelo.pkl", "wb") as f:
        pickle.dump({"minirocket": mr, "escalado": sc, "ridge": clf, "config": CONFIG_M, "semilla": s,
                     "pliegue": k, "entrada": "(n, 252, 30) = transpuesta de la entrada canonica 30 x 252"}, f)
    nombres_te = [str(x) for x in D["nombres_te"]]
    with open(d / "predicciones.json", "w", encoding="utf-8") as f:
        json.dump([{"muestra": nombres_te[i], "real": int(yte[i]), "predicha": int(pred[i]),
                    "top3": [int(x) for x in top3[i]], "puntaje_predicha": float(s_te[i, pred[i]])}
                   for i in range(len(yte))], f)
    np.savez_compressed(d / "puntajes.npz", val=s_va.astype(np.float32), test=s_te.astype(np.float32),
                        nombres_val=D["nombres_val"], nombres_test=D["nombres_te"])
    n_param = int(clf.coef_.size + clf.intercept_.size + 2 * n_rasgos)
    r = {"brazo": "M", "semilla": s, "pliegue": k, "senante_test": str(D["senante_test"]),
         "config": CONFIG_M, "alpha": alpha, "alpha_en_borde_de_la_rejilla": alpha in (CONFIG_M["alphas"][0],
                                                                                      CONFIG_M["alphas"][-1]),
         "cv_interna": cv, "n_rasgos": n_rasgos,
         "n_parametros": n_param,
         "nota_parametros": "coeficientes de la ridge (50 x rasgos) + interceptos + media y escala del "
                            "StandardScaler; los nucleos de MiniRocket son fijos (no se aprenden) y sus "
                            "sesgos (uno por rasgo) se estiman por cuantiles",
         "val_top1": mv["top1"], "val_top3": mv["top3"], "val_f1_macro": mv["f1_macro"],
         "test_top1": mt["top1"], "test_top3": mt["top3"], "test_f1_macro": mt["f1_macro"],
         "test_precision_macro": mt["precision_macro"], "test_recall_macro": mt["recall_macro"],
         "n_train_aumentado": int(len(ytr)), "aug": int(D["aug"]), "n_val": int(len(yva)), "n_test": int(len(yte)),
         "epocas_ejecutadas": None,
         "tiempo_entrenamiento_s": round(tiempo_ent, 1), "tiempo_total_s": round(time.perf_counter() - t0, 1),
         "tiempos_s": {"transformacion_entrenamiento": round(t_transform, 1), "cv_interna_alpha": round(t_cv, 1),
                       "ajuste_final": round(t_final, 1)},
         "n_jobs_numba": n_jobs, "modelo": _rel(d / "modelo.pkl"),
         "fecha_fin": datetime.now().isoformat(timespec="seconds")}
    (d / "resultado.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> M s={s} k={k}: val {mv['top1']:.3f} | test top-1 {mt['top1']:.3f} | alpha {alpha:g} | "
          f"{n_rasgos} rasgos | {r['tiempo_total_s']:.0f} s ({r['tiempos_s']})", flush=True)


def cmd_correr(args):
    src = Path(args.datos)
    semillas = [int(x) for x in args.semillas.split(",")]
    folds = [int(x) for x in args.folds.split(",")]
    print(f"[{datetime.now().isoformat(timespec='seconds')}] MiniRocket | semillas {semillas} | pliegues {folds} | "
          f"datos {src} | salida {_rel(args.out)}", flush=True)
    for s in semillas:
        for k in folds:
            correr_uno(src / f"semilla_{s}_pliegue_{k}.npz", args.out, args.n_jobs)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] fin", flush=True)


def cmd_humo(args):
    """Coteja la ridge espectral con sklearn en datos sintéticos y corre un pliegue con pocos núcleos."""
    from sklearn.linear_model import RidgeClassifier
    rng = np.random.default_rng(0)
    Fa, Fb = rng.standard_normal((300, 500)), rng.standard_normal((80, 500))
    ya, yb = rng.integers(0, 50, 300), rng.integers(0, 50, 80)
    for a in (0.1, 10.0, 1000.0):
        mine = exactitud_ridge_todas_alphas(Fa, ya, Fb, yb, [a], 50)[0]
        sk = float(np.mean(RidgeClassifier(alpha=a).fit(Fa, ya).predict(Fb) == yb))
        assert abs(mine - sk) < 1e-12, (a, mine, sk)
    print("[ok] ridge espectral = sklearn RidgeClassifier en 3 alphas", flush=True)


def cmd_latencia(args):
    import subprocess
    r = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True)
    activos = [l for l in r.stdout.splitlines() if "ython" in l and
               ("experimento_arquitecturas.py entrenar" in l or "experimento_arquitecturas_minirocket.py correr" in l)
               and "ps -axo" not in l]
    if activos:
        raise SystemExit(f"hay entrenamientos activos: {activos}")
    filas = []
    rng = np.random.default_rng(0)
    for ruta in sorted((Path(args.out) / CARPETA).glob("semilla_2026/pliegue_*/modelo.pkl"))[:args.modelos]:
        with open(ruta, "rb") as f:
            m = pickle.load(f)
        x1 = rng.standard_normal((1, 252, 30)).astype(np.float32)

        def una():
            F = np.asarray(m["minirocket"].transform(x1), dtype=np.float64)
            return m["ridge"].decision_function(m["escalado"].transform(F))
        for _ in range(args.calentamiento):
            una()
        t = []
        for _ in range(args.repeticiones):
            t0 = time.perf_counter()
            una()
            t.append((time.perf_counter() - t0) * 1000)
        fila = {"brazo": "M", "modelo": _rel(ruta), "tamano_archivo_mb": round(ruta.stat().st_size / 2 ** 20, 2),
                "ms_lote1_mediana": float(np.median(t)), "ms_lote1_p90": float(np.percentile(t, 90))}
        filas.append(fila)
        print(f"  {fila}", flush=True)
    import platform
    import sklearn
    import sktime
    import numba
    (Path(args.out) / "inferencia_minirocket.json").write_text(json.dumps(
        {"fecha": datetime.now().isoformat(timespec="seconds"),
         "metodo": f"transform de sktime + StandardScaler + decision_function de la ridge con lote de 1; "
                   f"{args.calentamiento} de calentamiento y {args.repeticiones} medidas; mediana y p90; CPU; "
                   f"n_jobs del modelo guardado",
         "entorno": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "sktime": sktime.__version__, "numba": numba.__version__, "plataforma": platform.platform()},
         "modelos": filas}, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("correr")
    c.add_argument("--datos", required=True, help="carpeta con semilla_S_pliegue_K.npz")
    c.add_argument("--semillas", default="2026,2027,2028")
    c.add_argument("--folds", default="0,1,2,3,4")
    c.add_argument("--n-jobs", type=int, default=4)
    h = sub.add_parser("humo")
    la = sub.add_parser("latencia")
    la.add_argument("--calentamiento", type=int, default=10)
    la.add_argument("--repeticiones", type=int, default=100)
    la.add_argument("--modelos", type=int, default=3)
    for p in (c, h, la):
        p.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    {"correr": cmd_correr, "humo": cmd_humo, "latencia": cmd_latencia}[args.cmd](args)


if __name__ == "__main__":
    main()
