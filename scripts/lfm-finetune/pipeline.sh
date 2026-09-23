#!/usr/bin/env bash
# The LFM2.5 fine-tune pipeline for issue 39, one resumable stage at a time.
#
#   scripts/lfm-finetune/pipeline.sh --env my.env <stage> [args]
#
# Stages (run in this order; each one skips work it has already done):
#   split                 seeded train/val/test split of nvsh/tiers/corpus/dev.json
#   skills                NVIDIA's Jetson skills at pinned commits: tools + 104 test evals
#   augment-nvsh          variations of every train entry (augment.py, resumable)
#   augment-skills        skill requests written from each SKILL.md description
#   assemble              training sets: nvsh-train.jsonl, skills-train.jsonl
#   train <name> [nvsh|skills]   train, merge, stage into HF_CACHE as REPO
#   measure-val <name>    validation run with per-entry details (iterate on this)
#   measure-final <name>  stock and <name> back to back on the test side (a final run)
#   measure-skills <name> stock and <name> on the 104 skill evals (margin required)
#   status                what exists so far
#
# Nothing here uploads anything, and nvsh itself never runs any of it. The
# gateway key is read from the variable AUG_KEY_ENV names (set it with
# `grant run --inject VAR=NAME -- ...`), never from this file or the env file.
set -euo pipefail

# Everything runs inside main(), called on the last line, so bash parses the
# whole file before executing any of it: a stage that runs for hours is not
# broken by the script being edited or updated underneath it.
main() {

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)

die() { echo "pipeline: $*" >&2; exit 1; }

ENV_FILE=""
if [ "${1:-}" = "--env" ]; then ENV_FILE=${2:-}; shift 2; fi
[ -n "$ENV_FILE" ] || die "pass --env <file> (copy scripts/lfm-finetune/pipeline.env.example)"
# shellcheck disable=SC1090
source "$ENV_FILE"
STAGE=${1:-status}; shift || true

: "${WORK:?}" "${BASE:?}" "${BASE_REV:?}" "${REPO:?}" "${HF_CACHE:?}" "${SEED:?}"
mkdir -p "$WORK"/{splits,skills,aug,data,runs,measure}
py() { (cd "$REPO_ROOT" && uv run --frozen python "$@"); }

aug_env() {
  local role
  for role in GENERATOR CORRECTOR REVIEWER_A REVIEWER_B; do
    export "NVSH_AUG_${role}_URL=$AUG_URL"
    export "NVSH_AUG_${role}_KEY_ENV=$AUG_KEY_ENV"
  done
  export NVSH_AUG_GENERATOR_MODEL=$AUG_GENERATOR_MODEL NVSH_AUG_CORRECTOR_MODEL=$AUG_CORRECTOR_MODEL
  export NVSH_AUG_REVIEWER_A_MODEL=$AUG_REVIEWER_A_MODEL NVSH_AUG_REVIEWER_B_MODEL=$AUG_REVIEWER_B_MODEL
  # Rewriting a sentence needs no reasoning; reviewers keep theirs, and get a
  # budget large enough to finish thinking and still answer.
  export NVSH_AUG_GENERATOR_DISABLE_THINKING=1 NVSH_AUG_CORRECTOR_DISABLE_THINKING=1
  export NVSH_AUG_GENERATOR_MAX_TOKENS=${AUG_REWRITE_MAX_TOKENS:-1024}
  export NVSH_AUG_CORRECTOR_MAX_TOKENS=${AUG_REWRITE_MAX_TOKENS:-1024}
  export NVSH_AUG_REVIEWER_A_MAX_TOKENS=${AUG_REVIEWER_MAX_TOKENS:-8192}
  export NVSH_AUG_REVIEWER_B_MAX_TOKENS=${AUG_REVIEWER_MAX_TOKENS:-8192}
  export NVSH_AUG_REVIEWER_A_TIMEOUT=${AUG_REVIEWER_TIMEOUT:-300}
  export NVSH_AUG_REVIEWER_B_TIMEOUT=${AUG_REVIEWER_TIMEOUT:-300}
  [ -n "${!AUG_KEY_ENV:-}" ] || die "$AUG_KEY_ENV is not set (e.g. grant run --inject $AUG_KEY_ENV=<secret name> -- $0 ...)"
}

base_snapshot() { echo "$HF_CACHE/hub/models--${BASE%%/*}--${BASE##*/}/snapshots/$BASE_REV"; }

case "$STAGE" in
  split)
    py scripts/lfm-finetune/split.py --out-dir "$WORK/splits" --seed "$SEED"
    ;;
  skills)
    py scripts/lfm-finetune/jetson_skills.py build --work-dir "$WORK/skills/src" --out-dir "$WORK/skills"
    ;;
  augment-nvsh)
    aug_env
    py scripts/lfm-finetune/augment.py "$WORK/splits/train.json" --per-source "$PER_SOURCE_NVSH" \
      --workers "$WORKERS" --accepted-out "$WORK/aug/nvsh-accepted.jsonl" \
      --rejected-out "$WORK/aug/nvsh-rejected.jsonl"
    ;;
  augment-skills)
    aug_env
    py scripts/lfm-finetune/augment.py "$WORK/skills/tools.json" --side train \
      --per-source "$PER_SOURCE_SKILLS" --workers "$WORKERS" \
      --accepted-out "$WORK/aug/skills-accepted.jsonl" \
      --rejected-out "$WORK/aug/skills-rejected.jsonl"
    ;;
  assemble)
    touch "$WORK/aug/nvsh-accepted.jsonl"
    py scripts/lfm-finetune/merge_variations.py --split "$WORK/splits/train.json" \
      --accepted "$WORK/aug/nvsh-accepted.jsonl" --out "$WORK/data/train-augmented.json"
    py scripts/lfm-finetune/build_dataset.py --split "$WORK/data/train-augmented.json" \
      --out "$WORK/data/nvsh-train.jsonl"
    if [ -s "$WORK/aug/skills-accepted.jsonl" ]; then
      py scripts/lfm-finetune/skills_dataset.py --accepted "$WORK/aug/skills-accepted.jsonl" \
        --tools "$WORK/skills/tools.json" --test "$WORK/skills/test.jsonl" \
        --out "$WORK/data/skills-train.jsonl"
    fi
    ;;
  train)
    name=${1:?train <name> [nvsh|skills]}; set="${2:-nvsh}"
    data="$WORK/data/$set-train.jsonl"; [ -s "$data" ] || die "no $data; run assemble first"
    run="$WORK/runs/$name"; mkdir -p "$run"
    # shellcheck disable=SC2086
    "$TRAIN_PY" "$HERE/train.py" --train "$data" --out "$run" --base "$BASE" --revision "$BASE_REV" \
      $TRAIN_ARGS 2>&1 | tee "$run/train.log"
    py scripts/lfm-finetune/stage_cache.py --merged "$run/merged" --repo "$REPO" \
      --cache "$HF_CACHE" --base-snapshot "$(base_snapshot)" | tee "$run/stage.log"
    awk '/staged/{print $NF}' "$run/stage.log" > "$run/revision"
    ;;
  measure-val)
    name=${1:?measure-val <name>}; rev=$(cat "$WORK/runs/$name/revision")
    py scripts/lfm-finetune/measure.py --split "$WORK/splits/val.json" --model "$REPO" \
      --revision "$rev" --label "$name-val" --config "$NVSH_CONFIG" \
      --out "$WORK/measure/$name-val.md" --details "$WORK/measure/$name-val.jsonl" --force
    ;;
  measure-final)
    name=${1:?measure-final <name>}; rev=$(cat "$WORK/runs/$name/revision")
    py scripts/lfm-finetune/measure.py --split "$WORK/splits/test.json" --final \
      --model "$BASE" --revision "$BASE_REV" --model "$REPO" --revision "$rev" \
      --label "final-$name" --config "$NVSH_CONFIG"
    ;;
  measure-skills)
    name=${1:?measure-skills <name> --margin "<margin>"}; shift
    for label in stock "$name"; do
      model=$BASE; rev=$BASE_REV; extra=()
      if [ "$label" != stock ]; then model=$REPO; rev=$(cat "$WORK/runs/$name/revision"); extra=(--tuned "$@"); fi
      py scripts/lfm-finetune/measure_skills.py --tools "$WORK/skills/tools.json" \
        --test "$WORK/skills/test.jsonl" --manifest "$WORK/skills/manifest.json" \
        --model "$model" --model-revision "$rev" --label "$label" --launch --config "$NVSH_CONFIG" \
        --timeout "${SKILLS_TIMEOUT:-180}" --out "$WORK/measure/skills-$label.md" "${extra[@]}"
    done
    ;;
  status)
    for f in splits/train.json splits/val.json splits/test.json skills/tools.json skills/test.jsonl \
             aug/nvsh-accepted.jsonl aug/skills-accepted.jsonl data/nvsh-train.jsonl data/skills-train.jsonl; do
      if [ -e "$WORK/$f" ]; then printf '%-28s %s\n' "$f" "$(wc -l < "$WORK/$f") lines"; else printf '%-28s -\n' "$f"; fi
    done
    find "$WORK/runs" -mindepth 1 -maxdepth 1 -type d -printf 'run: %f\n' 2>/dev/null
    ;;
  *)
    die "unknown stage '$STAGE' (see the header of $0)"
    ;;
esac
}

main "$@"; exit $?
