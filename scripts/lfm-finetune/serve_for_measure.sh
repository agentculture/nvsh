#!/usr/bin/env bash
# One pinned vLLM for every measured model (issue 46, deviation d7).
#
#   scripts/lfm-finetune/serve_for_measure.sh start MODEL_DIR PORT [RECORD_JSON]
#   scripts/lfm-finetune/serve_for_measure.sh wait PORT
#   scripts/lfm-finetune/serve_for_measure.sh stop PORT
#
# The stock copy, the Track A and Track B checkpoints and the AWQ build are
# all measured the same way: this helper serves one model directory with
# identical flags, and measure.py / measure_skills.py attach to it
# ([tiers.lfm] mode = "attach"). nvsh's own managed launcher cannot do this:
# it refuses an absolute model path for vLLM and cannot pass
# --limit-mm-per-prompt or --max-logprobs.
#
# `start` runs a detached container named q46-measure-PORT from MEASURE_IMAGE
# (by @sha256: digest; a tag is refused), with MODEL_DIR bind-mounted
# read-only at /model and the port published on 127.0.0.1 only, and passes
# vLLM exactly:
#
#   --model /model --served-model-name <MEASURE_MODEL_NAME, or MODEL_DIR's name>
#   --max-model-len MEASURE_CTX --gpu-memory-utilization MEASURE_GPU_FRACTION
#   --enable-auto-tool-choice --tool-call-parser TOOL_CALL_PARSER
#   --max-logprobs MEASURE_MAX_LOGPROBS --limit-mm-per-prompt '{"image": 0, "video": 0}'
#
# It refuses a MODEL_DIR that fails `gen_config.py check`: every measured
# model samples at temperature 0 from its own generation_config.json
# (deviation d3). With RECORD_JSON it writes the full docker argv there, next
# to the run's results. MEASURE_GPU_ARGS (default "--gpus all"; Jetson takes
# "--runtime nvidia") says how the container reaches the GPU.
#
# `wait` polls /v1/models until it answers, for up to MEASURE_WAIT_SECONDS
# (default 900) every MEASURE_POLL_SECONDS (default 2); on a timeout, or a
# container that has stopped, it prints the container's last log lines and
# exits 2. `stop` removes the container. Nothing here ever touches a
# container not named q46-measure-*.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)

#: The only container names this helper creates, inspects or removes.
NAME_PREFIX=q46-measure-
#: vLLM's port inside the container.
CONTAINER_PORT=8000
#: Where MODEL_DIR is mounted inside the container.
MODEL_MOUNT=/model
#: No image or video inputs: the model is measured on text alone.
NO_MULTIMODAL='{"image": 0, "video": 0}'

die() { echo "serve_for_measure: $*" >&2; exit 1; }
usage() { die "usage: $0 start MODEL_DIR PORT [RECORD_JSON] | wait PORT | stop PORT"; }

check_port() {
  if ! [[ ${1:-} =~ ^[0-9]+$ ]] || [ "$1" -lt 1024 ] || [ "$1" -gt 65535 ]; then
    die "PORT must be a number from 1024 to 65535 (got '${1:-}')"
  fi
}

container() { echo "$NAME_PREFIX$1"; }

check_settings() {
  [[ ${MEASURE_IMAGE:-} =~ ^[A-Za-z0-9][^@[:space:]]*@sha256:[0-9a-f]{64}$ ]] \
    || die "MEASURE_IMAGE must name the image by @sha256: digest, not a tag (got '${MEASURE_IMAGE:-}')"
  [ -n "${TOOL_CALL_PARSER:-}" ] || die "TOOL_CALL_PARSER is not set (qwen3_coder for Qwen, lfm2 for LFM)"
  [[ $MEASURE_CTX =~ ^[0-9]+$ ]] || die "MEASURE_CTX must be a whole number (got '$MEASURE_CTX')"
  [[ $MEASURE_MAX_LOGPROBS =~ ^[0-9]+$ ]] \
    || die "MEASURE_MAX_LOGPROBS must be a whole number (got '$MEASURE_MAX_LOGPROBS')"
  [[ $MEASURE_GPU_FRACTION =~ ^(0?\.[0-9]+|1(\.0*)?)$ ]] \
    || die "MEASURE_GPU_FRACTION must be a fraction in (0, 1] (got '$MEASURE_GPU_FRACTION')"
}

write_record() {
  # write_record FILE MODEL_DIR NAME PORT ARGV...: the docker argv as JSON.
  local record=$1 model_dir=$2 name=$3 port=$4
  shift 4
  mkdir -p "$(dirname "$record")"
  python3 - "$record" "$model_dir" "$name" "$port" "$MEASURE_IMAGE" "$(container "$port")" "$@" <<'PYEOF'
import datetime
import json
import sys

record, model_dir, name, port, image, container, *argv = sys.argv[1:]
payload = {
    "container": container,
    "image": image,
    "model_dir": model_dir,
    "served_model_name": name,
    "port": int(port),
    "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "argv": argv,
}
with open(record, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PYEOF
}

start() {
  local model_dir=${1:-} port=${2:-} record=${3:-}
  [ -n "$model_dir" ] || usage
  check_port "$port"
  MEASURE_CTX=${MEASURE_CTX:-2048}
  MEASURE_GPU_FRACTION=${MEASURE_GPU_FRACTION:-0.08}
  MEASURE_MAX_LOGPROBS=${MEASURE_MAX_LOGPROBS:-22}
  check_settings
  [ -d "$model_dir" ] || die "MODEL_DIR $model_dir is not a directory"
  model_dir=$(cd "$model_dir" && pwd -P)
  case $model_dir in
    *:* | *,*) die "MODEL_DIR $model_dir contains ':' or ',', which a docker -v argument cannot carry" ;;
  esac
  (cd "$REPO_ROOT" && uv run --frozen python scripts/lfm-finetune/gen_config.py check "$model_dir") >&2 \
    || die "gen_config.py check failed for $model_dir: no greedy generation_config.json" \
      "(deviation d3; write one with 'gen_config.py write')"
  local name=${MEASURE_MODEL_NAME:-$(basename "$model_dir")} own
  own=$(container "$port")
  if [ -n "$(docker ps -a --filter "name=^$own\$" --format '{{.Names}}')" ]; then
    die "a container named $own already exists; run '$0 stop $port' first"
  fi
  local gpu_args
  read -r -a gpu_args <<< "${MEASURE_GPU_ARGS-"--gpus all"}"
  local argv=(docker run -d --name "$own" -p "127.0.0.1:$port:$CONTAINER_PORT" "${gpu_args[@]}"
    -e HF_HUB_OFFLINE=1 -v "$model_dir:$MODEL_MOUNT:ro" "$MEASURE_IMAGE"
    --model "$MODEL_MOUNT" --served-model-name "$name"
    --max-model-len "$MEASURE_CTX" --gpu-memory-utilization "$MEASURE_GPU_FRACTION"
    --enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER"
    --max-logprobs "$MEASURE_MAX_LOGPROBS" --limit-mm-per-prompt "$NO_MULTIMODAL")
  [ -z "$record" ] || write_record "$record" "$model_dir" "$name" "$port" "${argv[@]}"
  "${argv[@]}" >/dev/null
  echo "serve_for_measure: started $own serving $model_dir as '$name' on 127.0.0.1:$port" >&2
}

wait_ready() {
  local port=${1:-}
  check_port "$port"
  local own url deadline
  own=$(container "$port")
  url="http://127.0.0.1:$port/v1/models"
  deadline=$(($(date +%s) + ${MEASURE_WAIT_SECONDS:-900}))
  while :; do
    if curl -fsS -o /dev/null --max-time 5 "$url" 2>/dev/null; then
      echo "serve_for_measure: $own is ready" >&2
      return 0
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$own" 2>/dev/null || true)" = false ]; then
      echo "serve_for_measure: $own stopped before it was ready; its last log lines:" >&2
      break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "serve_for_measure: $own not ready after ${MEASURE_WAIT_SECONDS:-900}s; its last log lines:" >&2
      break
    fi
    sleep "${MEASURE_POLL_SECONDS:-2}"
  done
  docker logs --tail "${MEASURE_LOG_LINES:-40}" "$own" >&2 2>&1 || true
  exit 2
}

stop() {
  local port=${1:-}
  check_port "$port"
  docker rm -f "$(container "$port")" >/dev/null 2>&1 || true
}

case "${1:-}" in
  start) shift; start "$@" ;;
  wait) shift; wait_ready "$@" ;;
  stop) shift; stop "$@" ;;
  *) usage ;;
esac
