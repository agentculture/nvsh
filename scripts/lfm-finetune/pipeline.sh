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
#   train-scorer [name]   train the Track B scorer on assemble's frozen training set
#                         (data/train-augmented.json) with split.py's val side
#                         (train_scorer.py, a training stage; runs/scorer[-name])
#   measure-val <name> [args]    validation run with per-entry details (iterate on
#                         this)
#   measure-final <name> [args]  a final run on the test side
#   measure-skills <name> [--margin "<margin>"]   the 104 skill evals; a tuned
#                         <name> needs --margin, which stock never gets
#                         Each measure stage measures ONE model per call (deviation d7):
#                         <name> is "stock" (the stock copy, WORK/stock -- stock-copy
#                         first) or a run name (WORK/runs/<name>/merged). The stage
#                         serves it with serve_for_measure.sh (MEASURE_IMAGE, pinned by
#                         digest, on 127.0.0.1:MEASURE_PORT; the model dir must pass
#                         gen_config.py check), writes a per-run nvsh config attaching
#                         to it ([tiers.lfm] mode = "attach"), measures, and stops the
#                         server again, on failure too. The served docker argv is kept
#                         in WORK/measure/<label>.serve.json. Every call passes
#                         --enable-thinking ENABLE_THINKING (default false);
#                         measure.py also gets --ground-snapshot GROUND_SNAPSHOT
#                         (required; see `measure.py snapshot`) and --max-logprobs
#                         MEASURE_MAX_LOGPROBS, and measure-val and measure-final hand
#                         any further [args] to it (e.g. --ctx 2048).
#   scan <name>           scan a trained run's merged checkpoint for secrets/binaries
#                         (scan_bundle.py scan; writes scan.json next to it)
#   quantize <name>       Q4_K_M GGUF + INT4 AWQ export of a merged checkpoint
#                         (quantize.py, a training stage: needs LLAMA_CPP_CONVERT,
#                         LLAMA_CPP_QUANTIZE, LLAMA_CPP_IMATRIX, and AWQ_PY -- the
#                         separate llm-compressor venv's python, which runs
#                         awq_oneshot.py; LLAMA_CPP_DIR optional, for the commit
#                         record); writes a generation_config.json into the AWQ
#                         export dir (deviation d3)
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
# An exported MEASURE_CTX outranks the env file's (issue 46, lapse l3: the
# file's 2048 silently replaced an exported 4096, so a run labelled 4K was
# served at 2048). A measure-val run at another context gets its own names.
_measure_ctx_from_environment=${MEASURE_CTX:-}
# shellcheck disable=SC1090
source "$ENV_FILE"
if [ -n "$_measure_ctx_from_environment" ]; then MEASURE_CTX=$_measure_ctx_from_environment; fi
MEASURE_CTX=${MEASURE_CTX:-2048}
# The memory caps must reach the stages' child processes (train.py and
# train_scorer.py read NVSH_TRAIN_GPU_MEMORY_GB), not only this shell.
export TRAIN_MEMORY_MAX TRAIN_MEMORY_FLOOR TRAIN_WATCHDOG_SECONDS TRAIN_MEMORY_CAP \
  NVSH_TRAIN_GPU_MEMORY_GB
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

stock_dir() {
  # The stock copy every measure stage serves stock from, never BASE itself:
  # BASE's snapshot has no generation_config.json pinning greedy decoding
  # (deviation d3), so serving it would sample.
  local out="$WORK/stock"
  [ -d "$out" ] || die "no $out; run stock-copy first (stock is measured from the stock copy)"
  py scripts/lfm-finetune/gen_config.py check "$out" >&2 \
    || die "gen_config.py check failed for $out: no greedy generation_config.json;" \
      "run stock-copy again (deviation d3)"
  echo "$out"
}

ground_snapshot() {
  [ -n "${GROUND_SNAPSHOT:-}" ] \
    || die "GROUND_SNAPSHOT is not set; write one with 'measure.py snapshot' and name it in the env file"
  [ -f "$GROUND_SNAPSHOT" ] \
    || die "GROUND_SNAPSHOT=$GROUND_SNAPSHOT does not exist; write it with 'measure.py snapshot'"
  echo "$GROUND_SNAPSHOT"
}

measure_model_dir() {
  # The directory a measure stage serves for <name>: the stock copy, or a
  # trained run's merged checkpoint.
  local name=$1
  [[ $name =~ ^[A-Za-z0-9._-]+$ ]] || die "measure: '$name' is not a run name (letters, digits, . _ -)"
  if [ "$name" = stock ]; then stock_dir; return; fi
  [ -d "$WORK/runs/$name/merged" ] || die "no $WORK/runs/$name/merged; run train $name first"
  echo "$WORK/runs/$name/merged"
}

scorer_measure_args() {
  # Track B (--scorer): measure.py's scorer loads a tokenizer (transformers,
  # the training environment's) from a path, not from the served name (issue
  # 46, t23). Prints the extra measure.py args; the caller sets PYTHONPATH.
  local arg
  for arg in "$@"; do
    if [ "$arg" = --scorer ] || [[ $arg == --scorer=* ]]; then
      printf '%s\n' --tokenizer "$(measure_model_dir "$1")"
      return 0
    fi
  done
}

measure_pythonpath() {
  # The training site-packages for a --scorer run, else nothing.
  local arg
  for arg in "$@"; do
    if [ "$arg" = --scorer ] || [[ $arg == --scorer=* ]]; then
      train_site_packages
      return 0
    fi
  done
}

refuse_extra_ctx() {
  # An extra --ctx relabels the report without changing the server (lapse l3).
  local arg
  for arg in "$@"; do
    case $arg in
      --ctx | --ctx=*) die "do not pass --ctx to a measure stage; set MEASURE_CTX (it starts the server and labels the run)" ;;
    esac
  done
}

measure_revision() {
  # The revision recorded for <name> (attach mode records it as operator-supplied).
  if [ "$1" = stock ]; then echo "$BASE_REV"; return; fi
  [ -d "$WORK/runs/$1/merged" ] || die "no $WORK/runs/$1/merged; run train $1 first"
  [ -s "$WORK/runs/$1/revision" ] || die "no $WORK/runs/$1/revision; run train $1 first"
  cat "$WORK/runs/$1/revision"
}

# shellcheck disable=SC2317  # invoked by serve_for_measure's EXIT trap
stop_measure_server() {
  bash "$HERE/serve_for_measure.sh" stop "$MEASURE_PORT" || true
}

serve_for_measure() {
  # serve_for_measure NAME LABEL: serve NAME's model dir with the committed
  # helper, stop it whenever this script exits, and write the per-run attach
  # config $measure_config for it (deviation d7).
  local name=$1 label=$2 model_dir
  model_dir=$(measure_model_dir "$name")
  MEASURE_PORT=${MEASURE_PORT:-18060}
  MEASURE_CTX=${MEASURE_CTX:-2048}
  MEASURE_GPU_FRACTION=${MEASURE_GPU_FRACTION:-0.08}
  MEASURE_MAX_LOGPROBS=${MEASURE_MAX_LOGPROBS:-22}
  MEASURE_MODEL_NAME=$name
  export MEASURE_IMAGE TOOL_CALL_PARSER MEASURE_CTX MEASURE_GPU_FRACTION MEASURE_MAX_LOGPROBS \
    MEASURE_MODEL_NAME MEASURE_WAIT_SECONDS MEASURE_POLL_SECONDS MEASURE_GPU_ARGS
  trap stop_measure_server EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  bash "$HERE/serve_for_measure.sh" start "$model_dir" "$MEASURE_PORT" "$WORK/measure/$label.serve.json"
  bash "$HERE/serve_for_measure.sh" wait "$MEASURE_PORT" "$WORK/measure/$label.serve.log"
  measure_config="$WORK/measure/$label.nvsh.toml"
  cat > "$measure_config" <<EOF
# Written by pipeline.sh for one measure run: attach to serve_for_measure.sh's
# server for '$name' (deviation d7). Rewritten on every run.
[tiers]
enabled = true

[tiers.lfm]
engine = "vllm"
mode = "attach"
base_url = "http://127.0.0.1:$MEASURE_PORT/v1"
model = "$name"
ctx = $MEASURE_CTX
tool_call_parser = "$TOOL_CALL_PARSER"
image = "$MEASURE_IMAGE"
EOF
}

train_site_packages() {
  # TRAIN_PY's site-packages, for a repo-env step that also needs the training
  # stack (transformers): on PYTHONPATH it adds that stack while the repo's own
  # uv environment -- and so the nvsh package -- stays the interpreter's.
  "$TRAIN_PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))' 2>/dev/null \
    || die "TRAIN_PY=$TRAIN_PY does not run; it is needed for assemble's render check"
}


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
    # PROTECTED_EXTRA (space-separated files): sides beyond val/test that must
    # never reach training -- issue 46: the issue-39 test, the corpus held-out
    # and the sealed held-out. Exact matches are excluded while merging, then
    # leakage_check.py drops exact and near-duplicate matches of every
    # protected side and prints ids only (t19, deviation d14).
    read -r -a protected_extra <<<"${PROTECTED_EXTRA:-}"
    protected=("$WORK/splits/val.json" "$WORK/splits/test.json" "${protected_extra[@]}")
    mkdir -p "$WORK/data"
    py scripts/lfm-finetune/merge_variations.py --split "$WORK/splits/train.json" \
      --accepted "$WORK/aug/nvsh-accepted.jsonl" --out "$WORK/data/train-augmented.merged.json" \
      --filter-to-split --exclude "${protected[@]}" "${supplement[@]}"
    py scripts/lfm-finetune/leakage_check.py --train "$WORK/data/train-augmented.merged.json" \
      --out-filtered "$WORK/data/train-augmented.json" --protected "${protected[@]}" \
      | tee "$WORK/data/leakage.json"
    # build_dataset.py's render check loads the base's tokenizer (transformers),
    # which only the training environment has.
    site=$(train_site_packages)
    PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}" \
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
    # The same frozen, leakage-filtered set Track A renders from (issue 46:
    # both tracks share one dataset; the raw split still holds the issue-39
    # test entries and a duplicate of a test entry).
    data="$WORK/data/train-augmented.json"
    [ -s "$data" ] || die "no $data; run assemble first"
    run="$WORK/runs/scorer${1:+-$1}"; mkdir -p "$run"
    # shellcheck disable=SC2086
    run_capped "$run" "$TRAIN_PY" "$HERE/train_scorer.py" --train "$data" \
      --val "$WORK/splits/val.json" --out "$run" --base "$BASE" --revision "$BASE_REV" \
      ${TRAIN_SCORER_ARGS:-}
    # Merged, greedy and staged exactly like a Track A run, so measure-val
    # scorer[-name] serves and measures it the same way (t23). Staged under
    # its own repo name so it never shares a revision list with Track A.
    run_capped "$run" "$TRAIN_PY" "$HERE/train.py" --merge-only "$run/adapter" --out "$run" \
      --base "$BASE" --revision "$BASE_REV"
    py scripts/lfm-finetune/gen_config.py write "$run/merged"
    py scripts/lfm-finetune/stage_cache.py --merged "$run/merged" --repo "$REPO-scorer" \
      --cache "$HF_CACHE" --base-snapshot "$(base_snapshot)" | tee "$run/stage.log"
    awk '/staged/{print $NF}' "$run/stage.log" > "$run/revision"
    ;;
  measure-val)
    name=${1:?measure-val <name> [measure.py args]}; shift
    refuse_extra_ctx "$@"
    snapshot=$(ground_snapshot); rev=$(measure_revision "$name")
    label="$name-val"
    if [ "$MEASURE_CTX" != 2048 ]; then label="$name-val-ctx$MEASURE_CTX"; fi
    mapfile -t scorer_args < <(scorer_measure_args "$name" "$@")
    site=$(measure_pythonpath "$@")
    if [ -n "$site" ]; then pythonpath="$site${PYTHONPATH:+:$PYTHONPATH}"; else pythonpath="${PYTHONPATH:-}"; fi
    serve_for_measure "$name" "$label"
    PYTHONPATH="$pythonpath" \
      py scripts/lfm-finetune/measure.py --split "$WORK/splits/val.json" --model "$name" \
      --revision "$rev" --label "$label" --config "$measure_config" --ctx "$MEASURE_CTX" \
      --ground-snapshot "$snapshot" --enable-thinking "${ENABLE_THINKING:-false}" \
      --max-logprobs "$MEASURE_MAX_LOGPROBS" \
      --out "$WORK/measure/$label.md" --details "$WORK/measure/$label.jsonl" --force \
      "${scorer_args[@]}" "$@"
    ;;
  measure-final)
    name=${1:?measure-final <name> [measure.py args]}; shift
    refuse_extra_ctx "$@"
    snapshot=$(ground_snapshot); rev=$(measure_revision "$name")
    mapfile -t scorer_args < <(scorer_measure_args "$name" "$@")
    site=$(measure_pythonpath "$@")
    if [ -n "$site" ]; then pythonpath="$site${PYTHONPATH:+:$PYTHONPATH}"; else pythonpath="${PYTHONPATH:-}"; fi
    serve_for_measure "$name" "final-$name"
    PYTHONPATH="$pythonpath" \
      py scripts/lfm-finetune/measure.py --split "$WORK/splits/test.json" --final \
      --model "$name" --revision "$rev" --label "final-$name" --config "$measure_config" \
      --ctx "$MEASURE_CTX" \
      --ground-snapshot "$snapshot" --enable-thinking "${ENABLE_THINKING:-false}" \
      --max-logprobs "$MEASURE_MAX_LOGPROBS" --predictions "$WORK/final/$name" \
      "${scorer_args[@]}" "$@"
    ;;
  measure-skills)
    # measure_skills.py grounds nothing, so it takes no --ground-snapshot, and it
    # cannot read an attach config: it is pointed at the served URL with --url.
    name=${1:?measure-skills <name> [--margin "<margin>"]}; shift
    rev=$(measure_revision "$name"); extra=()
    [ "$name" = stock ] || extra=(--tuned "$@")
    serve_for_measure "$name" "skills-$name"
    py scripts/lfm-finetune/measure_skills.py --tools "$WORK/skills/tools.json" \
      --test "$WORK/skills/test.jsonl" --manifest "$WORK/skills/manifest.json" \
      --url "http://127.0.0.1:$MEASURE_PORT/v1" --model "$name" --model-revision "$rev" \
      --label "$name" --enable-thinking "${ENABLE_THINKING:-false}" \
      --timeout "${SKILLS_TIMEOUT:-180}" --out "$WORK/measure/skills-$name.md" "${extra[@]}"
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
    # shellcheck disable=SC2016  # expanded by the child, on purpose
    bash -c 'echo "caps (as a child sees them): max=${TRAIN_MEMORY_MAX:-unset}\
 floor=${TRAIN_MEMORY_FLOOR:-8G (default)} watchdog=${TRAIN_WATCHDOG_SECONDS:-5}s\
 gpu_gb=${NVSH_TRAIN_GPU_MEMORY_GB:-unset}"'
    ;;
  *)
    die "unknown stage '$STAGE' -- one of: $STAGES"
    ;;
esac
}

main "$@"; exit $?
