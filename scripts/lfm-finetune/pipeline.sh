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
#   assemble              training sets: nvsh-train.jsonl, $SKILLS_SET-train.jsonl;
#                         with SCORER_BUILD_ARGS (build_dataset.py's t14 flags) also
#                         the Track B file scorer-train.json (issue 53, deviation d2)
#   train <name> [nvsh|skills]   train, merge, stage into HF_CACHE as REPO
#                         (a training stage: runs under TRAIN_MEMORY_MAX, mem.log); writes
#                         a generation_config.json into <run>/merged (deviation d3)
#   train-scorer [name]   train the Track B scorer on assemble's frozen training set
#                         (data/train-augmented.json) with split.py's val side
#                         (train_scorer.py, a training stage; runs/scorer[-name]);
#                         data/scorer-train.json instead when assemble wrote it
#                         (SCORER_BUILD_ARGS set) and it is the newer of the two
#   measure-val <name> [args]    validation run with per-entry details (iterate on
#                         this)
#   measure-final <name> [--slice S] [--scorer M]   a final run on the test
#                         side, labelled final-<name>[-missing-candidate][-exact]
#   measure-heldout <name> [--slice S] [--scorer M]   the one run on the sealed
#                         held-out file (HELDOUT_SPLIT, measure.py --acceptance),
#                         labelled heldout-<name>[-missing-candidate][-exact]
#                         (-exact: --scorer in-process). Both take
#                         nothing else (d16): S is full|missing-candidate, M is
#                         served|in-process
#   measure-skills <name> [--margin "<margin>"]   the 104 skill evals; a tuned
#                         <name> needs --margin, which stock never gets
#                         Each measure stage measures ONE model per call (deviation d7):
#                         <name> is "stock" (the stock copy, WORK/stock -- stock-copy
#                         first), a run name (WORK/runs/<name>/merged), or a quantized
#                         build of a run (`quantize <run>` first; t25):
#                           <run>.awq     WORK/quant/<run>/awq, served by vLLM like any
#                                         model dir (it must pass gen_config.py check)
#                           <run>.q4_k_m  WORK/quant/<run>/model-q4_k_m.gguf, served by
#                                         the native llama-server LLAMA_SERVER names
#                                         (required for it), greedy by --temp 0 --top-k 1
#                         A build's revision is sha256:<16 hex> of its
#                         quantize-run.json; a build of a Track B scorer run still
#                         needs --scorer (its tokenizer: the AWQ dir, or the run's
#                         merged dir for a GGUF), and a GGUF build refuses
#                         --scorer in-process. Labels keep the build name
#                         (final-a3.awq, heldout-a3.q4_k_m-missing-candidate). The
#                         stage serves the model with serve_for_measure.sh (vLLM:
#                         MEASURE_IMAGE, pinned by digest; either way on
#                         127.0.0.1:MEASURE_PORT), writes a per-run nvsh config
#                         attaching to it ([tiers.lfm] mode = "attach", engine "vllm"
#                         or "llama-server"), measures, and stops the server again, on
#                         failure too. The served argv, backend and (llama-server)
#                         binary version are kept in WORK/measure/<label>.serve.json.
#                         Every call passes
#                         --enable-thinking ENABLE_THINKING (default false);
#                         measure.py also gets --ground-snapshot GROUND_SNAPSHOT
#                         (required; see `measure.py snapshot`) and --max-logprobs
#                         MEASURE_MAX_LOGPROBS, and measure-val hands any further
#                         [args] to it (e.g. --scorer served).
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
#   upload <name>         push a run's merged checkpoint to REPO on the Hub, private,
#                         through hub_upload.py (as upload-bundle does; REPO must be in
#                         its jetson-ai-lab/qwen3.5-0.8b-nvsh- namespace). Refuses
#                         without FINAL=1 set, a scan_bundle.py verify pass on the exact
#                         folder, and a gen_config.py check pass on it (deviation d3: no
#                         served model ships without greedy decoding pinned); fetches
#                         the commit back and compares every file's sha256. The token
#                         comes only from the env var HF_TOKEN_ENV names, injected by
#                         the operator.
#   bundle <kind> <build-name> <repo-suffix> <report.md>...   the upload folder
#                         for one build of the Qwen3.5 run (issue 46, t27), written
#                         to WORK/bundles/<repo-suffix>/ for the repository
#                         jetson-ai-lab/qwen3.5-0.8b-nvsh-<repo-suffix>, then scanned
#                         (scan_bundle.py scan). <kind>/<build-name> is one of
#                           bf16 <run>          WORK/runs/<run>/merged
#                           gguf <run>.q4_k_m   WORK/quant/<run>/model-q4_k_m.gguf, with
#                                               the run's tokenizer and chat template
#                           awq  <run>.awq      WORK/quant/<run>/awq (compressed-tensors)
#                         release_bundle.py writes LICENSE, NOTICE and the card
#                         (--licence-kind apache: only Apache-2.0 teachers, named from
#                         TEACHER_MODELS, BUNDLE_ACCEPTED and data/train-augmented.json;
#                         BUNDLE_DATA_SUMMARY describes the data). Every <report.md>
#                         (bf16 test, quantized test, edge) is quoted by its "Issue 46
#                         metrics" table. A Track B run gets --scorer; a gguf/awq
#                         bundle whose suffix ends in -gguf/-awq names the repo
#                         without that ending as the bf16 it was quantized from. A
#                         record of what was built goes to WORK/bundles/<suffix>.json.
#                         e.g. bundle bf16 a3-heal tool-jev, bundle gguf a3-heal.q4_k_m
#                         tool-jev-gguf, bundle awq scorer-b1.awq tool-jev-scorer-awq
#   bundle-dataset <repo-suffix>   the data set folder (dataset_bundle.py --apache-only
#                         --issue 46): splits/, the frozen data/train-augmented.json,
#                         BUNDLE_ACCEPTED, BUNDLE_REJECTED (space-separated), nvsh's
#                         LICENSE and TEACHER_MODELS; DATASET_MODEL_REPOS (space-
#                         separated repo suffixes) names the models it trained. Scanned
#                         like a model bundle; uploaded as a dataset repository.
#   upload-bundle <repo-suffix>   upload WORK/bundles/<repo-suffix> PRIVATE to
#                         jetson-ai-lab/qwen3.5-0.8b-nvsh-<repo-suffix> (hub_upload.py,
#                         with the training environment's huggingface_hub). Refuses
#                         without FINAL=1, a scan_bundle.py verify pass, and (bf16/awq)
#                         a gen_config.py check pass; creates the repo private, sets it
#                         private again, uploads, fetches that commit back and compares
#                         the sha256 of every file (any difference fails), then prints
#                         the repo's private flag. The token comes only from the env var
#                         HF_TOKEN_ENV names. Nothing here ever makes a repo public:
#                         that waits for the operator's approval, repo by repo.
#   status                what exists so far
#
# Nothing here uploads anything except the guarded `upload` and `upload-bundle`
# stages above -- ask the operator before running either -- and nvsh itself
# never runs any of it. The gateway key is read from the variable
# AUG_KEY_ENV names (set it with `grant run --inject VAR=NAME -- ...`), never
# from this file or the env file; the same is true of the Hub token and
# HF_TOKEN_ENV.
set -euo pipefail

#: Every valid stage name, in the order above -- the unknown-stage message
#: below is the one place this list is printed, so it never drifts from the
#: case statement silently.
STAGES="split skills stock-copy augment-nvsh augment-skills rereview filter-variations \
assemble train train-scorer measure-val measure-final measure-heldout measure-skills scan \
quantize heal upload bundle bundle-dataset upload-bundle status"

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

build_base() {
  # The run a measured <name> belongs to: <run> for a quantized build
  # (<run>.awq, <run>.q4_k_m -- what `quantize <run>` wrote), else <name>.
  case $1 in
    *.awq | *.q4_k_m) echo "${1%.*}" ;;
    *) echo "$1" ;;
  esac
}

is_gguf_build() { [[ $1 == *.q4_k_m ]]; }

quant_build() {
  # quant_build NAME: the served path of a quantized build (the AWQ dir or the
  # GGUF file), after checking `quantize <run>` finished for it.
  local name=$1 base kind path
  base=$(build_base "$name"); kind=${name##*.}
  if [ "$kind" = awq ]; then path="$WORK/quant/$base/awq"; else path="$WORK/quant/$base/model-q4_k_m.gguf"; fi
  if [ "$kind" = awq ]; then
    [ -d "$path" ] || die "no $path; run quantize $base first"
  else
    [ -s "$path" ] || die "no $path; run quantize $base first"
  fi
  [ -s "$WORK/quant/$base/quantize-run.json" ] \
    || die "no $WORK/quant/$base/quantize-run.json; run quantize $base first (it finishes by writing it)"
  echo "$path"
}

measure_model_dir() {
  # What a measure stage serves for <name>: the stock copy, a trained run's
  # merged checkpoint, or a quantized build of a run -- <run>.awq (the AWQ
  # dir, served by vLLM like any model dir) or <run>.q4_k_m (the GGUF file,
  # served by a native llama-server).
  local name=$1
  [[ $name =~ ^[A-Za-z0-9._-]+$ ]] || die "measure: '$name' is not a run name (letters, digits, . _ -)"
  if [ "$name" = stock ]; then stock_dir; return; fi
  if [ "$(build_base "$name")" != "$name" ]; then quant_build "$name"; return; fi
  [ -d "$WORK/runs/$name/merged" ] || die "no $WORK/runs/$name/merged; run train $name first"
  echo "$WORK/runs/$name/merged"
}

tokenizer_dir() {
  # A directory transformers can load <name>'s tokenizer from: the AWQ dir
  # for <run>.awq, the base run's merged dir for <run>.q4_k_m (a GGUF file
  # is no tokenizer dir), else the model dir itself.
  local name=$1
  if is_gguf_build "$name"; then
    quant_build "$name" >/dev/null
    measure_model_dir "$(build_base "$name")"
  else
    measure_model_dir "$name"
  fi
}

scorer_measure_args() {
  # Track B (--scorer): measure.py's scorer loads a tokenizer (transformers,
  # the training environment's) from a path, not from the served name (issue
  # 46, t23). Prints the extra measure.py args; the caller sets PYTHONPATH.
  local arg dir
  for arg in "$@"; do
    if [ "$arg" = --scorer ] || [[ $arg == --scorer=* ]]; then
      dir=$(tokenizer_dir "$1")
      printf '%s\n' --tokenizer "$dir"
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

refuse_scorer_without_mode() {
  # A Track B scorer run (train_scorer.py writes "objective" into its
  # train-log.json) measured without --scorer is scored as a generative
  # tool-caller, a meaningless 0 of 32 (issue 46, P66).
  # A quantized build is a scorer when its base run is.
  local name=$1 arg
  shift
  grep -q '"objective"' "$WORK/runs/$(build_base "$name")/train-log.json" 2>/dev/null || return 0
  for arg in "$@"; do
    if [ "$arg" = --scorer ] || [[ $arg == --scorer=* ]]; then return 0; fi
  done
  die "$name is a Track B scorer; pass --scorer served (decisions, latency) or --scorer in-process (exact calibration)"
}

refuse_gguf_in_process() {
  # The in-process scorer loads the model with transformers, which never
  # reads the GGUF: it would measure the bf16 base run under the quant's
  # name. A GGUF build is scored only by --scorer served (llama-server's
  # /v1/completions returns the top next-token log-probabilities it needs).
  local name=$1 arg previous=''
  shift
  is_gguf_build "$name" || return 0
  for arg in "$@"; do
    if [ "$arg" = --scorer=in-process ] || { [ "$previous" = --scorer ] && [ "$arg" = in-process ]; }; then
      die "$name is a GGUF build: --scorer in-process cannot load a GGUF (transformers would" \
        "measure the bf16 run instead); use --scorer served, or measure $(build_base "$name").awq in-process"
    fi
    previous=$arg
  done
}

check_final_args() {
  # A final or held-out run takes only --slice and --scorer, each at most
  # once, spelled out, with a known value: measure.py's argparse keeps the last
  # value and accepts abbreviations, so anything else could relabel the run,
  # swap the split or overwrite another run's predictions (issue 46, t24, d16).
  # Prints the label suffix: "-missing-candidate" for that slice, then
  # "-exact" for --scorer in-process, else nothing.
  local stage=$1 arg value slice='' scorer='' expect=''
  shift
  for arg in "$@"; do
    if [ -n "$expect" ]; then
      value=$arg
    else
      case $arg in
        --slice | --scorer) expect=$arg; continue ;;
        --slice=* | --scorer=*) expect=${arg%%=*}; value=${arg#*=} ;;
        *) die "$stage takes only --slice and --scorer, not '$arg'" ;;
      esac
    fi
    case $expect:$value in
      --slice:full | --slice:missing-candidate)
        [ -z "$slice" ] || die "$stage: --slice given twice"
        slice=$value ;;
      --scorer:served | --scorer:in-process)
        [ -z "$scorer" ] || die "$stage: --scorer given twice"
        scorer=$value ;;
      *) die "$stage: $expect takes full|missing-candidate (--slice) or served|in-process (--scorer), not '$value'" ;;
    esac
    expect=
  done
  [ -z "$expect" ] || die "$stage: $expect needs a value"
  local suffix=''
  if [ "$slice" = missing-candidate ]; then suffix=-missing-candidate; fi
  # The exact in-process scorer is a second run of the same set (d15).
  if [ "$scorer" = in-process ]; then suffix=$suffix-exact; fi
  echo "$suffix"
}

measure_revision() {
  # The revision recorded for <name> (attach mode records it as operator-supplied).
  # A quantized build has no Hub revision: it is named by the first 16 hex
  # digits of the sha256 of its quantize-run.json, which quantize.py rewrites
  # on every export, so a re-export is never mistaken for the one measured.
  if [ "$1" = stock ]; then echo "$BASE_REV"; return; fi
  if [ "$(build_base "$1")" != "$1" ]; then
    quant_build "$1" >/dev/null
    local digest
    digest=$(sha256sum "$WORK/quant/$(build_base "$1")/quantize-run.json")
    echo "sha256:${digest:0:16}"
    return
  fi
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
  local name=$1 label=$2 model_dir engine=vllm image
  model_dir=$(measure_model_dir "$name")
  MEASURE_PORT=${MEASURE_PORT:-18060}
  MEASURE_CTX=${MEASURE_CTX:-2048}
  MEASURE_GPU_FRACTION=${MEASURE_GPU_FRACTION:-0.08}
  # Kept equal to scorer.py's READOUT_TOP (issue 53 t3): the scorer requests
  # this many next-token log-probabilities per served request.
  MEASURE_MAX_LOGPROBS=${MEASURE_MAX_LOGPROBS:-20000}
  MEASURE_MODEL_NAME=$name
  # A GGUF build's native llama-server keeps its pid file and log here.
  MEASURE_RUN_DIR="$WORK/measure"
  export MEASURE_IMAGE TOOL_CALL_PARSER MEASURE_CTX MEASURE_GPU_FRACTION MEASURE_MAX_LOGPROBS \
    MEASURE_MODEL_NAME MEASURE_WAIT_SECONDS MEASURE_POLL_SECONDS MEASURE_GPU_ARGS MEASURE_RUN_DIR \
    LLAMA_SERVER
  trap stop_measure_server EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  bash "$HERE/serve_for_measure.sh" start "$model_dir" "$MEASURE_PORT" "$WORK/measure/$label.serve.json"
  bash "$HERE/serve_for_measure.sh" wait "$MEASURE_PORT" "$WORK/measure/$label.serve.log"
  # The run record's image field names what served the run: the vLLM image,
  # or for a GGUF build the native llama-server's path and --version output
  # (from the serve record; JSON's string escapes are valid TOML).
  image=$(printf '%s' "$MEASURE_IMAGE" | python3 -c 'import json, sys; print(json.dumps(sys.stdin.read()))')
  if is_gguf_build "$name"; then
    engine=llama-server
    # shellcheck disable=SC2016  # a literal $HOME is written into the record
    image=$(python3 -c '
import json, os, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
version = " ".join(record["version"].split())
binary = record["binary"]
home = os.environ.get("HOME", "")
if home and binary.startswith(home.rstrip("/") + "/"):
    binary = "$HOME" + binary[len(home.rstrip("/")):]  # no /home/<user>/ in results pages
print(json.dumps(f"native llama-server {binary} ({version})"))
' "$WORK/measure/$label.serve.json")
  fi
  measure_config="$WORK/measure/$label.nvsh.toml"
  cat > "$measure_config" <<EOF
# Written by pipeline.sh for one measure run: attach to serve_for_measure.sh's
# server for '$name' (deviation d7). Rewritten on every run.
[tiers]
enabled = true

[tiers.lfm]
engine = "$engine"
mode = "attach"
base_url = "http://127.0.0.1:$MEASURE_PORT/v1"
model = "$name"
ctx = $MEASURE_CTX
tool_call_parser = "$TOOL_CALL_PARSER"
image = $image
EOF
}

train_site_packages() {
  # TRAIN_PY's site-packages, for a repo-env step that also needs the training
  # stack (transformers): on PYTHONPATH it adds that stack while the repo's own
  # uv environment -- and so the nvsh package -- stays the interpreter's.
  "$TRAIN_PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))' 2>/dev/null \
    || die "TRAIN_PY=$TRAIN_PY does not run; it is needed for assemble's render check"
}


#: Every bundle stage's repository id is this plus a repo suffix (issue 46,
#: t27); hub_upload.py refuses any other.
BUNDLE_REPO_PREFIX=jetson-ai-lab/qwen3.5-0.8b-nvsh-

check_suffix() {
  # A repo suffix: lower-case letters and digits in '-'-separated words.
  [[ $1 =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] \
    || die "'$1' is not a repo suffix (lower-case letters, digits and '-', e.g. tool-jev-gguf)"
}

require_qwen_base() {
  # The bundle stages ship the Apache-2.0 Qwen3.5 run only (issue 46).
  [[ $BASE == Qwen/* ]] \
    || die "bundle stages ship the Apache-2.0 Qwen3.5 run only; BASE=$BASE (use the Qwen env file)"
}

require_teacher_models() {
  [ -n "${TEACHER_MODELS:-}" ] \
    || die "TEACHER_MODELS is not set; name the run's alias -> {name, licence} JSON in the env file"
  [ -s "$TEACHER_MODELS" ] || die "TEACHER_MODELS=$TEACHER_MODELS does not exist or is empty"
}

write_bundle_record() {
  # write_bundle_record SUFFIX KIND BUILD REPO_TYPE: WORK/bundles/SUFFIX.json,
  # next to (not inside) the bundle, so upload-bundle knows what it holds.
  python3 -c '
import json, sys
suffix, kind, build, repo_type, prefix, path = sys.argv[1:]
record = {"kind": kind, "build": build, "repo": prefix + suffix, "repo_type": repo_type}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(record, handle)
    handle.write("\n")
' "$1" "$2" "$3" "$4" "$BUNDLE_REPO_PREFIX" "$WORK/bundles/$1.json"
}

read_bundle_record() {
  # read_bundle_record SUFFIX: prints "<kind> <repo_type> <repo>".
  python3 -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
print(record["kind"], record["repo_type"], record["repo"])
' "$WORK/bundles/$1.json"
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
    # SCORER_BUILD_ARGS (optional, issue 53 deviation d2): build_dataset.py's
    # t14 flags (e.g. "--randomize-labels --perm-seed 53 --missing-candidate-rate
    # 0.3 --reasons"); when set, the same build also writes the corpus-format
    # Track B file data/scorer-train.json that train-scorer then prefers. Unset,
    # assemble is exactly issue 46's.
    read -r -a scorer_build <<<"${SCORER_BUILD_ARGS:-}"
    if [ "${#scorer_build[@]}" -gt 0 ]; then
      scorer_build+=(--scorer-out "$WORK/data/scorer-train.json")
    fi
    # build_dataset.py's render check loads the base's tokenizer (transformers),
    # which only the training environment has.
    site=$(train_site_packages)
    PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}" \
      py scripts/lfm-finetune/build_dataset.py --split "$WORK/data/train-augmented.json" \
      --out "$WORK/data/nvsh-train.jsonl" --base "$BASE" --revision "$BASE_REV" \
      "${scorer_build[@]}"
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
    # assemble with SCORER_BUILD_ARGS also writes data/scorer-train.json: the
    # same entries plus -nocand ones, each with its own label map (issue 53,
    # deviation d2). Used only when newer than the training set, so an
    # assemble re-run without SCORER_BUILD_ARGS never trains on a stale one.
    scorer_data="$WORK/data/scorer-train.json"
    if [ -s "$scorer_data" ] && [ "$scorer_data" -nt "$data" ]; then data=$scorer_data; fi
    echo "train-scorer: training on $data"
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
    refuse_scorer_without_mode "$name" "$@"
    refuse_gguf_in_process "$name" "$@"
    snapshot=$(ground_snapshot); rev=$(measure_revision "$name")
    label="$name-val"
    if [ "$MEASURE_CTX" != 2048 ]; then label="$name-val-ctx$MEASURE_CTX"; fi
    # Run once in this shell first: a die inside the process substitution
    # below would not stop the stage (a GGUF build's tokenizer is its base
    # run's merged dir, which nothing else checks).
    scorer_measure_args "$name" "$@" >/dev/null
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
    name=${1:?measure-final <name> [--slice S] [--scorer M]}; shift
    refuse_extra_ctx "$@"
    suffix=$(check_final_args measure-final "$@")
    refuse_scorer_without_mode "$name" "$@"
    refuse_gguf_in_process "$name" "$@"
    snapshot=$(ground_snapshot); rev=$(measure_revision "$name")
    scorer_measure_args "$name" "$@" >/dev/null  # see measure-val
    mapfile -t scorer_args < <(scorer_measure_args "$name" "$@")
    site=$(measure_pythonpath "$@")
    if [ -n "$site" ]; then pythonpath="$site${PYTHONPATH:+:$PYTHONPATH}"; else pythonpath="${PYTHONPATH:-}"; fi
    label="final-$name$suffix"
    serve_for_measure "$name" "$label"
    PYTHONPATH="$pythonpath" \
      py scripts/lfm-finetune/measure.py --split "$WORK/splits/test.json" --final \
      --model "$name" --revision "$rev" --label "$label" --config "$measure_config" \
      --ctx "$MEASURE_CTX" \
      --ground-snapshot "$snapshot" --enable-thinking "${ENABLE_THINKING:-false}" \
      --max-logprobs "$MEASURE_MAX_LOGPROBS" --predictions "$WORK/final/$name" \
      "${scorer_args[@]}" "$@"
    ;;
  measure-heldout)
    name=${1:?measure-heldout <name> [--slice S] [--scorer M]}; shift
    [ -n "${HELDOUT_SPLIT:-}" ] || die "measure-heldout needs HELDOUT_SPLIT (the sealed held-out file) in the env file"
    [ -s "$HELDOUT_SPLIT" ] || die "HELDOUT_SPLIT=$HELDOUT_SPLIT does not exist or is empty"
    refuse_extra_ctx "$@"
    suffix=$(check_final_args measure-heldout "$@")
    refuse_scorer_without_mode "$name" "$@"
    refuse_gguf_in_process "$name" "$@"
    snapshot=$(ground_snapshot); rev=$(measure_revision "$name")
    scorer_measure_args "$name" "$@" >/dev/null  # see measure-val
    mapfile -t scorer_args < <(scorer_measure_args "$name" "$@")
    site=$(measure_pythonpath "$@")
    if [ -n "$site" ]; then pythonpath="$site${PYTHONPATH:+:$PYTHONPATH}"; else pythonpath="${PYTHONPATH:-}"; fi
    label="heldout-$name$suffix"
    serve_for_measure "$name" "$label"
    PYTHONPATH="$pythonpath" \
      py scripts/lfm-finetune/measure.py --split "$HELDOUT_SPLIT" --acceptance \
      --model "$name" --revision "$rev" --label "$label" --config "$measure_config" \
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
    # The same guarded path as upload-bundle: hub_upload.py refuses a REPO
    # outside its namespace or a symlinked folder, sets the repo private with
    # whichever call this huggingface_hub has, and fetches the commit back.
    # huggingface_hub is the training environment's (the repo env has none).
    site=$(train_site_packages)
    FINAL=1 PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}" \
      py scripts/lfm-finetune/hub_upload.py --bundle "$bundle" --repo "$REPO" \
      --repo-type model --token-env "$HF_TOKEN_ENV"
    ;;
  bundle)
    usage="bundle <bf16|gguf|awq> <build-name> <repo-suffix> <report.md>..."
    kind=${1:?$usage}; build=${2:?$usage}; suffix=${3:?$usage}; shift 3
    require_qwen_base
    check_suffix "$suffix"
    extra=()
    case $kind in
      bf16)
        [ "$(build_base "$build")" = "$build" ] \
          || die "bundle bf16 takes a run name, not the quantized build '$build'" ;;
      gguf)
        [[ $build == *.q4_k_m ]] || die "bundle gguf takes <run>.q4_k_m (e.g. $build.q4_k_m), not '$build'"
        extra=(--gguf "$(quant_build "$build")") ;;
      awq)
        [[ $build == *.awq ]] || die "bundle awq takes <run>.awq (e.g. ${build%.*}.awq), not '$build'"
        extra=(--awq-dir "$(quant_build "$build")") ;;
      *) die "bundle: kind is bf16|gguf|awq, not '$kind'" ;;
    esac
    [ $# -gt 0 ] || die "bundle needs at least one measure report (<report.md>) after the repo suffix"
    results=()
    for report in "$@"; do
      [ -s "$report" ] || die "no measure report $report"
      results+=(--results "$report")
    done
    require_teacher_models
    [ -n "${BUNDLE_DATA_SUMMARY:-}" ] || die "BUNDLE_DATA_SUMMARY is not set; describe the training data in the env file"
    run=$(build_base "$build")
    merged="$WORK/runs/$run/merged"
    [ -d "$merged" ] || die "no $merged; run train $run first"
    if grep -q '"objective"' "$WORK/runs/$run/train-log.json" 2>/dev/null; then extra+=(--scorer); fi
    if [ "$kind" != bf16 ] && [[ $suffix == *-$kind ]]; then
      extra+=(--quantized-from "$BUNDLE_REPO_PREFIX${suffix%-"$kind"}")
    fi
    out="$WORK/bundles/$suffix"
    mkdir -p "$WORK/bundles"
    py scripts/lfm-finetune/release_bundle.py --kind "$kind" "${extra[@]}" --merged "$merged" \
      --base-snapshot "$(base_snapshot)" --repo "$BUNDLE_REPO_PREFIX$suffix" --run "$run" \
      "${results[@]}" --data-summary "$BUNDLE_DATA_SUMMARY" --licence-kind apache \
      --tool-call-parser "$TOOL_CALL_PARSER" --teacher-models "$TEACHER_MODELS" \
      --accepted "${BUNDLE_ACCEPTED:-$WORK/aug/nvsh-accepted.jsonl}" \
      --train-augmented "$WORK/data/train-augmented.json" --out "$out"
    write_bundle_record "$suffix" "$kind" "$build" model
    py scripts/lfm-finetune/scan_bundle.py scan "$out"
    ;;
  bundle-dataset)
    suffix=${1:?bundle-dataset <repo-suffix>}
    require_qwen_base
    check_suffix "$suffix"
    require_teacher_models
    train="$WORK/data/train-augmented.json"
    [ -s "$train" ] || die "no $train; run assemble first (the frozen training set)"
    read -r -a rejected <<<"${BUNDLE_REJECTED:-$WORK/aug/nvsh-rejected.jsonl}"
    read -r -a model_suffixes <<<"${DATASET_MODEL_REPOS:-}"
    model_repos=()
    for model in "${model_suffixes[@]}"; do
      check_suffix "$model"
      model_repos+=(--model-repo "$BUNDLE_REPO_PREFIX$model")
    done
    out="$WORK/bundles/$suffix"
    mkdir -p "$WORK/bundles"
    py scripts/lfm-finetune/dataset_bundle.py --splits "$WORK/splits" --train-augmented "$train" \
      --accepted "${BUNDLE_ACCEPTED:-$WORK/aug/nvsh-accepted.jsonl}" --rejected "${rejected[@]}" \
      --licence "$REPO_ROOT/LICENSE" --teacher-models "$TEACHER_MODELS" --apache-only \
      --issue 46 "${model_repos[@]}" --out "$out"
    write_bundle_record "$suffix" dataset "$suffix" dataset
    py scripts/lfm-finetune/scan_bundle.py scan "$out"
    ;;
  upload-bundle)
    suffix=${1:?upload-bundle <repo-suffix>}
    check_suffix "$suffix"
    [ "${FINAL:-0}" = 1 ] || die "upload-bundle refuses without FINAL=1 set (ask the operator first)"
    bundle="$WORK/bundles/$suffix"
    [ -d "$bundle" ] || die "no $bundle; run bundle (or bundle-dataset) for $suffix first"
    [ -s "$bundle.json" ] || die "no $bundle.json; run bundle (or bundle-dataset) for $suffix again"
    read -r kind repo_type repo < <(read_bundle_record "$suffix")
    [ "$repo" = "$BUNDLE_REPO_PREFIX$suffix" ] \
      || die "$bundle.json names $repo, not $BUNDLE_REPO_PREFIX$suffix; run bundle again"
    py scripts/lfm-finetune/scan_bundle.py verify "$bundle" \
      || die "scan_bundle.py verify failed for $bundle; run bundle again (it scans the new folder)"
    case $kind in
      bf16 | awq)
        py scripts/lfm-finetune/gen_config.py check "$bundle" \
          || die "gen_config.py check failed for $bundle: no greedy generation_config.json (deviation d3)" ;;
    esac
    : "${HF_TOKEN_ENV:?}"
    [ -n "${!HF_TOKEN_ENV:-}" ] \
      || die "$HF_TOKEN_ENV is not set (e.g. grant run --inject $HF_TOKEN_ENV=<secret name> -- $0 ...)"
    # huggingface_hub is the training environment's (the repo env has no
    # third-party packages), as for the scorer's transformers.
    site=$(train_site_packages)
    FINAL=1 PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}" \
      py scripts/lfm-finetune/hub_upload.py --bundle "$bundle" --repo "$repo" \
      --repo-type "$repo_type" --token-env "$HF_TOKEN_ENV"
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
