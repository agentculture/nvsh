#!/usr/bin/env bash
# The LFM2.5 / Qwen3.5-0.8B fine-tune pipeline for issues 39 and 46, one
# resumable stage at a time.
#
#   scripts/lfm-finetune/pipeline.sh --env my.env <stage> [args]
#
# Stages (run in this order; each one skips work it has already done). The
# full, authoritative list is $STAGES below -- an unknown stage prints it.
#   split                 seeded train/val/test split of nvsh/tiers/corpus/dev.json
#   skills                NVIDIA's Jetson skills at pinned commits: tools + 104 test evals
#   stock-copy            a self-contained copy of BASE/BASE_REV's snapshot under WORK, symlinks
#                         resolved so it can be bind-mounted into a container, with a
#                         generation_config.json written (deviation d3: greedy decoding)
#   augment-nvsh          variations of every train entry (augment.py, resumable)
#   augment-skills        skill requests written from each SKILL.md (SKILLS_SEED:
#                         tools.json for the description, bodies.json for the body)
#   rereview              re-review stored accepted+rejected nvsh candidates with
#                         REVIEWER_B only (augment.py --rereview, decisions c38/c41);
#                         writes to separate *-rereview.jsonl files -- inspect them
#                         and copy over the accepted/rejected files by hand
#   filter-variations     count accepted variations against the train split with
#                         --filter-to-split, without touching assemble's own output
#                         (a leakage check between augment/rereview and assemble)
#   assemble              training sets: nvsh-train.jsonl, $SKILLS_SET-train.jsonl
#   train <name> [nvsh|skills]   train, merge, stage into HF_CACHE as REPO
#                         (a training stage: runs under TRAIN_MEMORY_MAX, mem.log); writes
#                         a generation_config.json into <run>/merged (deviation d3)
#   train-scorer          train the acceptance-margin scorer on split.py's own
#                         train/val sides (train_scorer.py, a training stage)
#   measure-val <name>    validation run with per-entry details (iterate on this)
#   measure-final <name>  stock and <name> back to back on the test side (a final run)
#   measure-skills <name> stock and <name> on the 104 skill evals (margin required)
#   scan <name>           scan a trained run's merged checkpoint for secrets/binaries
#                         (scan_bundle.py scan; writes scan.json next to it)
#   quantize <name>       Q4_K_M GGUF + INT4 AWQ export of a merged checkpoint
#                         (quantize.py, a training stage: needs LLAMA_CPP_CONVERT,
#                         LLAMA_CPP_QUANTIZE, LLAMA_CPP_IMATRIX, LLM_COMPRESSOR); writes
#                         a generation_config.json into the AWQ export dir (deviation d3)
#   heal <name> <base-run> [nvsh|skills]   a healing fine-tune that continues
#                         training from <base-run>'s own merged checkpoint instead
#                         of BASE (decisions c42/c43; a training stage). Whether
#                         healing is needed is quantize.py's heal_needed(), decided
#                         by a separate run (issue 46, task t18), not by this stage.
#                         Writes a generation_config.json into <run>/merged (deviation d3)
#   upload <name>         push a run's merged checkpoint to REPO on the Hub, private.
#                         Refuses without FINAL=1 set, a scan_bundle.py verify pass on
#                         the exact folder, and a gen_config.py check pass on it
#                         (deviation d3: no served model ships without greedy decoding
#                         pinned); the token comes only from the env var HF_TOKEN_ENV
#                         names, injected by the operator.
#   status                what exists so far
#
# Nothing here uploads anything except the guarded `upload` stage above, and
# nvsh itself never runs any of it. The gateway key is read from the variable
# AUG_KEY_ENV names (set it with `grant run --inject VAR=NAME -- ...`), never
# from this file or the env file; the same is true of the Hub token and
# HF_TOKEN_ENV.
set -euo pipefail

#: Every valid stage name, in the order above -- the unknown-stage message
#: below is the one place this list is printed, so it never drifts from the
#: case statement silently.
STAGES="split skills stock-copy augment-nvsh augment-skills rereview filter-variations \
assemble train train-scorer measure-val measure-final measure-skills scan \
quantize heal upload status"

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


# shellcheck source=scripts/lfm-finetune/capped.sh
source "$(dirname "${BASH_SOURCE[0]}")/capped.sh"

case "$STAGE" in
  split)
    py scripts/lfm-finetune/split.py --out-dir "$WORK/splits" --seed "$SEED"
    ;;
  skills)
    py scripts/lfm-finetune/jetson_skills.py build --work-dir "$WORK/skills/src" --out-dir "$WORK/skills"
    ;;
  stock-copy)
    out="$WORK/stock"
    if py scripts/lfm-finetune/gen_config.py check "$out" >/dev/null 2>&1; then
      echo "stock-copy: $out already has a valid generation_config.json"
    else
      py scripts/lfm-finetune/gen_config.py stock-copy "$(base_snapshot)" "$out" --force
    fi
    ;;
  augment-nvsh)
    aug_env
    py scripts/lfm-finetune/augment.py "$WORK/splits/train.json" --per-source "$PER_SOURCE_NVSH" \
      --workers "$WORKERS" --accepted-out "$WORK/aug/nvsh-accepted.jsonl" \
      --rejected-out "$WORK/aug/nvsh-rejected.jsonl"
    ;;
  augment-skills)
    aug_env
    py scripts/lfm-finetune/augment.py "${SKILLS_SEED:-$WORK/skills/tools.json}" --side train \
      --per-source "$PER_SOURCE_SKILLS" --workers "$WORKERS" \
      --accepted-out "$WORK/aug/${SKILLS_SET:-skills}-accepted.jsonl" \
      --rejected-out "$WORK/aug/${SKILLS_SET:-skills}-rejected.jsonl"
    ;;
  rereview)
    aug_env
    py scripts/lfm-finetune/augment.py "$WORK/aug/nvsh-accepted.jsonl" "$WORK/aug/nvsh-rejected.jsonl" \
      --rereview --accepted-out "$WORK/aug/nvsh-accepted-rereview.jsonl" \
      --rejected-out "$WORK/aug/nvsh-rejected-rereview.jsonl"
    ;;
  filter-variations)
    touch "$WORK/aug/nvsh-accepted.jsonl"
    py scripts/lfm-finetune/merge_variations.py --split "$WORK/splits/train.json" \
      --accepted "$WORK/aug/nvsh-accepted.jsonl" --out "$WORK/data/train-augmented.filtered.json" \
      --exclude "$WORK/splits/val.json" "$WORK/splits/test.json" --filter-to-split
    ;;
  assemble)
    touch "$WORK/aug/nvsh-accepted.jsonl"
    supplement=()
    [ -n "${SUPPLEMENT-$HERE/train-supplement.json}" ] && supplement=(--supplement "${SUPPLEMENT-$HERE/train-supplement.json}")
    py scripts/lfm-finetune/merge_variations.py --split "$WORK/splits/train.json" \
      --accepted "$WORK/aug/nvsh-accepted.jsonl" --out "$WORK/data/train-augmented.json" \
      --exclude "$WORK/splits/val.json" "$WORK/splits/test.json" "${supplement[@]}"
    py scripts/lfm-finetune/build_dataset.py --split "$WORK/data/train-augmented.json" \
      --out "$WORK/data/nvsh-train.jsonl" --base "$BASE" --revision "$BASE_REV"
    skills_set=${SKILLS_SET:-skills}
    if [ -s "$WORK/aug/$skills_set-accepted.jsonl" ]; then
      py scripts/lfm-finetune/skills_dataset.py --accepted "$WORK/aug/$skills_set-accepted.jsonl" \
        --tools "$WORK/skills/tools.json" --test "$WORK/skills/test.jsonl" \
        --out "$WORK/data/$skills_set-train.jsonl"
    fi
    ;;
  train)
    name=${1:?train <name> [nvsh|skills]}; set="${2:-nvsh}"
    data="$WORK/data/$set-train.jsonl"; [ -s "$data" ] || die "no $data; run assemble first"
    run="$WORK/runs/$name"; mkdir -p "$run"
    # shellcheck disable=SC2086
    run_capped "$run" "$TRAIN_PY" "$HERE/train.py" --train "$data" --out "$run" --base "$BASE" \
      --revision "$BASE_REV" $TRAIN_ARGS
    py scripts/lfm-finetune/gen_config.py write "$run/merged"
    py scripts/lfm-finetune/stage_cache.py --merged "$run/merged" --repo "$REPO" \
      --cache "$HF_CACHE" --base-snapshot "$(base_snapshot)" | tee "$run/stage.log"
    awk '/staged/{print $NF}' "$run/stage.log" > "$run/revision"
    ;;
  train-scorer)
    run="$WORK/runs/scorer"; mkdir -p "$run"
    # shellcheck disable=SC2086
    run_capped "$run" "$TRAIN_PY" "$HERE/train_scorer.py" --train "$WORK/splits/train.json" \
      --val "$WORK/splits/val.json" --out "$run" --base "$BASE" --revision "$BASE_REV" \
      ${TRAIN_SCORER_ARGS:-}
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
  scan)
    name=${1:?scan <name>}
    py scripts/lfm-finetune/scan_bundle.py scan "$WORK/runs/$name/merged"
    ;;
  quantize)
    name=${1:?quantize <name>}
    run="$WORK/runs/$name"; [ -d "$run/merged" ] || die "no $run/merged; run train $name first"
    work="$WORK/quant/$name"; mkdir -p "$work"
    calibration=()
    [ -n "${CALIBRATION_LIMIT:-}" ] && calibration=(--calibration-limit "$CALIBRATION_LIMIT")
    run_capped "$work" env -C "$REPO_ROOT" uv run --frozen python scripts/lfm-finetune/quantize.py \
      --model-dir "$run/merged" --train "$WORK/splits/train.json" --val "$WORK/splits/val.json" \
      --test "$WORK/splits/test.json" --work-dir "$work" "${calibration[@]}"
    py scripts/lfm-finetune/gen_config.py write "$work/awq"
    ;;
  heal)
    name=${1:?heal <name> <base-run> [nvsh|skills]}
    base_run=${2:?heal <name> <base-run> [nvsh|skills]}; set="${3:-nvsh}"
    data="$WORK/data/$set-train.jsonl"; [ -s "$data" ] || die "no $data; run assemble first"
    base_merged="$WORK/runs/$base_run/merged"
    [ -d "$base_merged" ] || die "no $base_merged; run train $base_run first"
    run="$WORK/runs/$name"; mkdir -p "$run"
    # A healing fine-tune continues from its own checkpoint, not from BASE
    # (decisions c42/c43); train.py's --revision is meaningless for a local
    # checkpoint directory and is left at its default, which transformers
    # ignores when --base is a local path.
    # shellcheck disable=SC2086
    run_capped "$run" "$TRAIN_PY" "$HERE/train.py" --train "$data" --out "$run" --base "$base_merged" \
      $TRAIN_ARGS
    py scripts/lfm-finetune/gen_config.py write "$run/merged"
    py scripts/lfm-finetune/stage_cache.py --merged "$run/merged" --repo "$REPO" \
      --cache "$HF_CACHE" --base-snapshot "$(base_snapshot)" | tee "$run/stage.log"
    awk '/staged/{print $NF}' "$run/stage.log" > "$run/revision"
    ;;
  upload)
    name=${1:?upload <name>}
    [ "${FINAL:-0}" = 1 ] || die "upload refuses without FINAL=1 set (mirrors measure.py's --final)"
    bundle="$WORK/runs/$name/merged"
    [ -d "$bundle" ] || die "no $bundle; run train $name first"
    py scripts/lfm-finetune/scan_bundle.py verify "$bundle" \
      || die "scan_bundle.py verify failed for $bundle; run 'scan $name' again on this exact folder"
    py scripts/lfm-finetune/gen_config.py check "$bundle" \
      || die "gen_config.py check failed for $bundle; write a generation_config.json" \
        "into it (e.g. 'gen_config.py write $bundle') before uploading (deviation d3)"
    : "${HF_TOKEN_ENV:?}"
    [ -n "${!HF_TOKEN_ENV:-}" ] \
      || die "$HF_TOKEN_ENV is not set (e.g. grant run --inject $HF_TOKEN_ENV=<secret name> -- $0 ...)"
    HF_TOKEN_ENV="$HF_TOKEN_ENV" REPO="$REPO" BUNDLE="$bundle" \
      py - <<'PYEOF'
import os

from huggingface_hub import HfApi

hf_token = os.environ[os.environ["HF_TOKEN_ENV"]]
api = HfApi(token=hf_token)
repo = os.environ["REPO"]
api.create_repo(repo, private=True, exist_ok=True)
api.update_repo_visibility(repo, private=True)
api.upload_folder(folder_path=os.environ["BUNDLE"], repo_id=repo, repo_type="model")
print(f"uploaded {os.environ['BUNDLE']} to {repo} (private)")
PYEOF
    ;;
  status)
    for f in splits/train.json splits/val.json splits/test.json skills/tools.json skills/test.jsonl \
             aug/nvsh-accepted.jsonl aug/skills-accepted.jsonl data/nvsh-train.jsonl data/skills-train.jsonl; do
      if [ -e "$WORK/$f" ]; then printf '%-28s %s\n' "$f" "$(wc -l < "$WORK/$f") lines"; else printf '%-28s -\n' "$f"; fi
    done
    find "$WORK/runs" -mindepth 1 -maxdepth 1 -type d -printf 'run: %f\n' 2>/dev/null
    ;;
  *)
    die "unknown stage '$STAGE' -- one of: $STAGES"
    ;;
esac
}

main "$@"; exit $?
