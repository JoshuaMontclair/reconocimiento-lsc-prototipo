#!/usr/bin/env bash
# congelar_exploracion.sh — Congela en git la exploración preliminar (marzo de 2026)
# y archiva los scripts muertos en legacy/src/ conservando su historia.
#
# Hace, en este orden y sin push:
#   1. Comprobaciones: raíz del repo, identidad git, sin merge/rebase en curso,
#      etiqueta inexistente, .gitignore con models/*.keras, ningún .keras/.npy
#      ni archivo > 25 MiB en lo que se va a commitear, ningún script canónico
#      importa de uno archivado.
#   2. Commit 1: el estado de marzo tal cual está (lista explícita de rutas: los
#      trackeados modificados, los scripts conocidos de src/, models/*.json|png,
#      .gitignore corregido y los dos documentos del diagnóstico). Archivos nuevos
#      de otras tareas quedan fuera y se listan al final.
#   3. Etiqueta anotada exploracion-marzo-2026 sobre ese commit.
#   4. git mv src/<muerto>.py legacy/src/<muerto>.py para cada script archivado.
#   5. Commit 2: los git mv + legacy/README.md + este script.
#
# Uso (desde cualquier sitio; el script se ubica solo):
#   bash scripts/congelar_exploracion.sh            # ejecuta
#   bash scripts/congelar_exploracion.sh --dry-run  # solo muestra lo que haría
#
# README.md y requirements.txt NO entran en el commit 1: se reescribieron en septiembre de 2026
#   y van en un commit posterior, junto con src/train_loso.py y data/partitions/.
#
# Idempotencia: si la etiqueta ya existe, o los scripts ya están en legacy/src/,
# aborta con mensaje claro y no toca nada. No borra ni sobrescribe archivos.

set -euo pipefail

TAG="exploracion-marzo-2026"
MSG1="Exploración preliminar (marzo 2026): estado congelado antes de ejecutar el plan del anteproyecto"
MSG2="Archiva los scripts de la exploración en legacy/; conserva los canónicos en src/"
MSG_TAG="Exploración preliminar de marzo de 2026 (sprint 21-23 mar) congelada: código de src/, metadatos y gráficos de models/, documentos del diagnóstico. README.md y requirements.txt se reescribieron en septiembre de 2026 y van en un commit posterior.
Cifras no reportables: selección de modelos mirando el conjunto de prueba y semilla de
partición cambiada (ver ESTADO_PROYECTO_2026-09-13.md, hallazgo B, y DECISIONES_TESIS.txt)."
MAX_KB=25600   # 25 MiB: por encima de esto se coló un .keras o un .npy

# Scripts que se archivan (ver legacy/README.md para el motivo de cada uno)
ARCHIVAR="train_v2 train_v3 train_v4 train_ultimate train_targeted train_targeted_final \
train_confusion_fix train_sample52_fix train_binary_11v40 train_extra_models \
eval_ensemble eval_full eval_smart_ensemble eval_mega_ensemble final_100_eval extract_landmarks"

# Scripts que se quedan en src/ (train_lstm se queda porque train_final_100 y finetune_lstm importan de él)
CANONICOS="preprocess_lsc50 prepare_dataset reextract_lsc50 train_lstm train_final_100 \
finetune_lstm grammar_mapper translator realtime tts"

# Lo que entra en el commit 1: SOLO el estado de marzo (los 9 archivos trackeados que se
# modificaron, los scripts de src/ conocidos, resultados/curvas de models/) más el .gitignore
# corregido y los dos documentos del diagnóstico de septiembre. Cualquier otro archivo nuevo
# (p. ej. los del LOSO) queda fuera y se avisa al final: va en commits posteriores.
RUTAS1=( .gitignore DECISIONES_TESIS.txt ESTADO_PROYECTO_2026-09-13.md
         data/dataset/dataset_info.json data/processed/metadata.json 'models/*.json' 'models/*.png' )
for s in $ARCHIVAR $CANONICOS; do RUTAS1+=( "src/$s.py" ); done

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then DRY_RUN=1; fi

abortar() { echo; echo "ERROR: $*" >&2; exit 1; }
paso()    { echo; echo "== $* =="; }

# ---------------------------------------------------------------- 1. Comprobaciones
paso "Comprobaciones previas"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || abortar "esto no es un repositorio git"
cd "$ROOT"
[[ -f .gitignore && -d src && -f src/train_final_100.py && -f src/grammar_mapper.py ]] \
  || abortar "no parece la raíz del repositorio de la tesis: $ROOT"
echo "Raíz del repositorio: $ROOT"
echo "Rama: $(git rev-parse --abbrev-ref HEAD)"

# Identidad (este Mac no tiene ~/.gitconfig: hay que fijarla una vez)
if [[ -z "$(git config user.name || true)" || -z "$(git config user.email || true)" ]]; then
  abortar "git no tiene identidad configurada. Fíjala una vez (solo para este repo) y vuelve a ejecutar:
    git config user.name  \"Joshua Montclair\"
    git config user.email \"tu-correo@ejemplo.com\""
fi
echo "Autor: $(git config user.name) <$(git config user.email)>"

# Sin merge / rebase / cherry-pick en curso y sin conflictos
for f in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD; do
  [[ -e "$(git rev-parse --git-path "$f")" ]] && abortar "hay un $f en curso; resuélvelo antes"
done
for d in rebase-merge rebase-apply; do
  [[ -d "$(git rev-parse --git-path "$d")" ]] && abortar "hay un rebase en curso; resuélvelo antes"
done
if git diff --name-only --diff-filter=U | grep -q .; then
  abortar "hay archivos con conflictos sin resolver:
$(git diff --name-only --diff-filter=U)"
fi

# Etiqueta
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  abortar "la etiqueta '$TAG' ya existe (apunta al commit $(git rev-parse --short "$TAG^{commit}")).
La congelación ya se hizo; no hay nada que repetir. Si de verdad hay que rehacerla, primero:
    git tag -d $TAG"
fi

# Scripts: los archivables están en src/ y no en legacy/src/; los canónicos están en src/
for s in $ARCHIVAR; do
  [[ -f "legacy/src/$s.py" ]] && abortar "legacy/src/$s.py ya existe: los scripts ya están archivados"
  [[ -f "src/$s.py" ]] || abortar "falta src/$s.py (esperado para archivar)"
done
for s in $CANONICOS; do
  [[ -f "src/$s.py" ]] || abortar "falta el script canónico src/$s.py"
done
# Los dos documentos del diagnóstico entran en el commit 1 (RUTAS1): tienen que existir
for f in DECISIONES_TESIS.txt ESTADO_PROYECTO_2026-09-13.md; do
  [[ -f "$f" ]] || abortar "falta $f (entra en el commit 1; ver RUTAS1)"
done

# Ningún canónico importa de un archivado
PATRON="$(echo $ARCHIVAR | sed 's/ /|/g')"
MALOS="$(for s in $CANONICOS; do
  grep -nE "^[[:space:]]*(from[[:space:]]+(src\.)?($PATRON)[[:space:]]+import|import[[:space:]]+(src\.)?($PATRON)\b|from[[:space:]]+src[[:space:]]+import[[:space:]]+.*\b($PATRON)\b)" "src/$s.py" \
    | sed "s#^#src/$s.py:#" || true
done)"
[[ -z "$MALOS" ]] || abortar "un script canónico importa de uno que se va a archivar:
$MALOS"
echo "Imports: ningún canónico depende de un archivado (OK)"

# .gitignore protege los modelos
grep -qxE 'models/\*\.keras' .gitignore || abortar ".gitignore no contiene la línea 'models/*.keras'"
git check-ignore -q models/best_model.keras || abortar "models/best_model.keras no está ignorado"

# Ningún .keras/.npy/.task puede estar sin ignorar (señal de .gitignore roto)
if git status --porcelain --untracked-files=all | cut -c4- | grep -E '\.(keras|npy|task)$' >/dev/null; then
  abortar "hay .keras/.npy/.task sin ignorar en el árbol de trabajo:
$(git status --porcelain --untracked-files=all | cut -c4- | grep -E '\.(keras|npy|task)$')"
fi

# Lo que entraría en el commit 1 (solo las rutas de la lista), sin archivos gigantes
PENDIENTE="$(git status --porcelain --untracked-files=all -- "${RUTAS1[@]}")"
N_PEND=0; KB_PEND=0
while IFS= read -r p; do
  [[ -n "$p" ]] || continue
  N_PEND=$((N_PEND + 1))
  if [[ -f "$p" ]]; then
    kb=$(du -k "$p" | cut -f1)
    KB_PEND=$((KB_PEND + kb))
    (( kb > MAX_KB )) && abortar "archivo demasiado grande para git ($((kb/1024)) MiB): $p"
  fi
done < <(echo "$PENDIENTE" | cut -c4-)
echo "Commit 1: $N_PEND rutas, $KB_PEND KiB en total (máx. $((MAX_KB/1024)) MiB por archivo)"
echo "$PENDIENTE" | sed 's/^/    /'

paso "Plan"
echo "  commit 1: «${MSG1}»"
echo "  tag:      $TAG"
echo "  git mv:   $(echo $ARCHIVAR | wc -w | tr -d ' ') scripts src/ -> legacy/src/"
echo "  commit 2: «${MSG2}»"
echo "  push:     NO (queda para el estudiante)"

if (( DRY_RUN )); then
  echo; echo "[--dry-run] No se ha modificado nada."; exit 0
fi

# ---------------------------------------------------------------- 2. Commit 1
paso "Commit 1: estado congelado"
if [[ -n "$PENDIENTE" ]]; then
  git add -- "${RUTAS1[@]}"
  git commit -q -m "$MSG1"
else
  echo "Nada pendiente de commitear: se etiqueta el HEAD actual tal cual."
fi
echo "HEAD: $(git log -1 --format='%h %s')"

# ---------------------------------------------------------------- 3. Etiqueta
paso "Etiqueta $TAG"
git tag -a "$TAG" -m "$MSG_TAG"
echo "$(git tag -n1 "$TAG")"

# ---------------------------------------------------------------- 4. git mv
paso "Archivo en legacy/src/ (git mv, conserva historia)"
mkdir -p legacy/src
for s in $ARCHIVAR; do
  git mv "src/$s.py" "legacy/src/$s.py"
  echo "  src/$s.py -> legacy/src/$s.py"
done
[[ -f legacy/README.md ]] || echo "AVISO: falta legacy/README.md (debería existir junto a este script)"
git add legacy scripts/congelar_exploracion.sh

# ---------------------------------------------------------------- 5. Commit 2
paso "Commit 2: archivo"
git commit -q -m "$MSG2"

# ---------------------------------------------------------------- Resultado
paso "Resultado"
git log --oneline -3
echo
git tag -n1 "$TAG"
echo "  -> $(git rev-parse "$TAG"^{commit})"
echo
echo "src/:        $(ls src/*.py | wc -l | tr -d ' ') scripts  ($(ls src/*.py | xargs -n1 basename | sed 's/\.py$//' | tr '\n' ' '))"
echo "legacy/src/: $(ls legacy/src/*.py | wc -l | tr -d ' ') scripts"
RESTO="$(git status --porcelain)"
if [[ -n "$RESTO" ]]; then
  echo; echo "AVISO: quedan cambios sin commitear (no son del estado de marzo; van en commits posteriores):"; echo "$RESTO" | sed 's/^/    /'
else
  echo "Árbol de trabajo limpio."
fi
echo
echo "No se ha hecho push. Cuando quieras publicarlo:  git push origin $(git rev-parse --abbrev-ref HEAD) --follow-tags"
