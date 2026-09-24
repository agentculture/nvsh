#!/usr/bin/env bash
# One pinned vLLM for every measured model directory, and one native
# llama-server for a GGUF build (issue 46, deviation d7; t25).
#
#   scripts/lfm-finetune/serve_for_measure.sh start MODEL PORT [RECORD_JSON]
#   scripts/lfm-finetune/serve_for_measure.sh wait PORT [FULL_LOG]
#   scripts/lfm-finetune/serve_for_measure.sh stop PORT
#
# The stock copy, the Track A and Track B checkpoints and the AWQ build are
# all measured the same way: this helper serves one model directory with
# identical flags, and measure.py / measure_skills.py attach to it
# ([tiers.lfm] mode = "attach"). nvsh's own managed launcher cannot do this:
# it refuses an absolute model path for vLLM and cannot pass
# --limit-mm-per-prompt or --max-logprobs.
#
# vLLM (MODEL is a directory): `start` runs a detached container named
# q46-measure-PORT from MEASURE_IMAGE (by @sha256: digest; a tag is refused),
# with MODEL bind-mounted read-only at /model and the port published on
# 127.0.0.1 only, and passes vLLM exactly:
#
#   --model /model --served-model-name <MEASURE_MODEL_NAME, or MODEL's name>
#   --max-model-len MEASURE_CTX --gpu-memory-utilization MEASURE_GPU_FRACTION
#   --enable-auto-tool-choice --tool-call-parser TOOL_CALL_PARSER
#   --max-logprobs MEASURE_MAX_LOGPROBS --limit-mm-per-prompt '{"image": 0, "video": 0}'
#
# It refuses a MODEL dir that fails `gen_config.py check`: every measured
# model samples at temperature 0 from its own generation_config.json
# (deviation d3). MEASURE_GPU_ARGS (default "--gpus all"; Jetson takes
# "--runtime nvidia") says how the container reaches the GPU.
#
# llama-server (MODEL is a .gguf file, the Q4_K_M build): `start` runs the
# native binary LLAMA_SERVER names (required; no container), detached, bound
# to 127.0.0.1 only, with exactly:
#
#   --model MODEL --host 127.0.0.1 --port PORT --ctx-size MEASURE_CTX --jinja
#   --n-gpu-layers 999 --temp 0 --top-k 1 --alias <MEASURE_MODEL_NAME, or MODEL's name>
#
# A GGUF has no generation_config.json, so deviation d3's temperature-0 rule
# is applied by flags instead (--temp 0 --top-k 1: greedy), and the record
# says so. The server's pid goes to MEASURE_RUN_DIR/q46-measure-PORT.pid and
# its output to q46-measure-PORT.log there (MEASURE_RUN_DIR defaults to
# XDG_RUNTIME_DIR, else /tmp). MEASURE_IMAGE, TOOL_CALL_PARSER,
# MEASURE_GPU_FRACTION and MEASURE_MAX_LOGPROBS do not apply to it
# (llama-server has no log-probability cap to raise).
#
# With RECORD_JSON, `start` writes the backend ("vllm" or "llama-server") and
# the full argv there, next to the run's results, and for llama-server also
# the binary's path and its --version output: every quant row names its
# serving stack and version.
#
# `wait` polls /v1/models until it answers, for up to MEASURE_WAIT_SECONDS
# (default 900) every MEASURE_POLL_SECONDS (default 2); on a timeout, or a
# server that has stopped, it prints the server's last log lines (and with
# FULL_LOG keeps the whole log there) and exits 2. `stop` stops the
# llama-server this helper started on PORT -- only a process whose pid file
# it wrote and whose command line still names the recorded binary -- and
# removes the container. Nothing here ever touches a container not named
# q46-measure-*.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)

#: The only container, pid-file and log names this helper creates, inspects or removes.
NAME_PREFIX=q46-measure-
#: vLLM's port inside the container.
CONTAINER_PORT=8000
#: Where MODEL is mounted inside the container.
MODEL_MOUNT=/model
#: No image or video inputs: the model is measured on text alone.
NO_MULTIMODAL='{"image": 0, "video": 0}'

die() { echo "serve_for_measure: $*" >&2; exit 1; }
usage() { die "usage: $0 start MODEL PORT [RECORD_JSON] | wait PORT [FULL_LOG] | stop PORT"; }

check_port() {
  if ! [[ ${1:-} =~ ^[0-9]+$ ]] || [ "$1" -lt 1024 ] || [ "$1" -gt 65535 ]; then
    die "PORT must be a number from 1024 to 65535 (got '${1:-}')"
  fi
}

container() { echo "$NAME_PREFIX$1"; }

run_dir() { echo "${MEASURE_RUN_DIR:-${XDG_RUNTIME_DIR:-/tmp}}"; }
pid_file() { echo "$(run_dir)/$NAME_PREFIX$1.pid"; }
llama_log() { echo "$(run_dir)/$NAME_PREFIX$1.log"; }

own_llama_pid() {
  # own_llama_pid PORT: the pid of the llama-server this helper started on
  # PORT, while it still runs, else nothing. A pid file whose process has
  # gone, or whose command line no longer names the recorded binary (a
  # reused pid), is never trusted.
  local file pid='' binary='' cmdline
  file=$(pid_file "$1")
  [ -f "$file" ] || return 0
  { read -r pid; read -r binary; } <"$file" || true
  if ! [[ $pid =~ ^[0-9]+$ ]] || [ -z "$binary" ]; then return 0; fi
  cmdline=$(tr '\0' '\n' <"/proc/$pid/cmdline" 2>/dev/null) || return 0
  grep -qxF -- "$binary" <<<"$cmdline" || return 0
  echo "$pid"
}

check_ctx() {
  [[ $MEASURE_CTX =~ ^[0-9]+$ ]] || die "MEASURE_CTX must be a whole number (got '$MEASURE_CTX')"
}

check_settings() {
  [[ ${MEASURE_IMAGE:-} =~ ^[A-Za-z0-9][^@[:space:]]*@sha256:[0-9a-f]{64}$ ]] \
    || die "MEASURE_IMAGE must name the image by @sha256: digest, not a tag (got '${MEASURE_IMAGE:-}')"
  [ -n "${TOOL_CALL_PARSER:-}" ] || die "TOOL_CALL_PARSER is not set (qwen3_coder for Qwen, lfm2 for LFM)"
  check_ctx
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
    "backend": "vllm",
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

write_llama_record() {
  # write_llama_record FILE MODEL NAME PORT BINARY VERSION ARGV...: the native
  # llama-server's argv, binary and --version output as JSON.
  local record=$1 model=$2 name=$3 port=$4 binary=$5 version=$6
  shift 6
  mkdir -p "$(dirname "$record")"
  python3 - "$record" "$model" "$name" "$port" "$binary" "$version" \
    "$(pid_file "$port")" "$(llama_log "$port")" "$@" <<'PYEOF'
import datetime
import json
import sys

record, model, name, port, binary, version, pid_file, log, *argv = sys.argv[1:]
payload = {
    "backend": "llama-server",
    "binary": binary,
    "version": version,
    "model_file": model,
    "served_model_name": name,
    "port": int(port),
    "decoding": (
        "greedy by flags (--temp 0 --top-k 1): a GGUF has no generation_config.json, "
        "so deviation d3's temperature 0 rule is applied on the command line"
    ),
    "pid_file": pid_file,
    "log": log,
    "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "argv": argv,
}
with open(record, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PYEOF
}

start_llama() {
  # start_llama GGUF PORT [RECORD_JSON]: the native llama-server backend.
  local model=$1 port=$2 record=$3 binary version
  binary=${LLAMA_SERVER:-}
  [ -n "$binary" ] || die "a .gguf MODEL is served by a native llama-server; set LLAMA_SERVER" \
    "to its binary (llama.cpp's build/bin/llama-server)"
  if ! [ -f "$binary" ] || ! [ -x "$binary" ]; then
    die "LLAMA_SERVER=$binary is not an executable file; point it at llama.cpp's llama-server"
  fi
  binary=$(cd "$(dirname "$binary")" && pwd -P)/$(basename "$binary")
  model=$(cd "$(dirname "$model")" && pwd -P)/$(basename "$model")
  if [ -n "$(own_llama_pid "$port")" ]; then
    die "a llama-server this helper started already runs on port $port; run '$0 stop $port' first"
  fi
  version=$("$binary" --version 2>&1) || die "'$binary --version' failed: $version"
  local name=${MEASURE_MODEL_NAME:-$(basename "$model" .gguf)}
  mkdir -p "$(run_dir)"
  local argv=("$binary" --model "$model" --host 127.0.0.1 --port "$port"
    --ctx-size "$MEASURE_CTX" --jinja --n-gpu-layers 999 --temp 0 --top-k 1 --alias "$name")
  [ -z "$record" ] \
    || write_llama_record "$record" "$model" "$name" "$port" "$binary" "$version" "${argv[@]}"
  nohup "${argv[@]}" </dev/null >"$(llama_log "$port")" 2>&1 &
  printf '%s\n%s\n' "$!" "$binary" >"$(pid_file "$port")"
  echo "serve_for_measure: started llama-server (pid $!) serving $model as '$name'" \
    "on 127.0.0.1:$port" >&2
}

start() {
  local model_dir=${1:-} port=${2:-} record=${3:-}
  [ -n "$model_dir" ] || usage
  check_port "$port"
  MEASURE_CTX=${MEASURE_CTX:-2048}
  if [[ $model_dir == *.gguf ]]; then
    check_ctx
    [ -f "$model_dir" ] || die "MODEL $model_dir is not a file"
    start_llama "$model_dir" "$port" "$record"
    return
  fi
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
  local port=${1:-} full_log=${2:-}
  check_port "$port"
  local own url deadline native=''
  own=$(container "$port")
  # A pid file means `start` launched the native llama-server on this port.
  if [ -f "$(pid_file "$port")" ]; then
    native=1
    own="llama-server on port $port"
  fi
  url="http://127.0.0.1:$port/v1/models"
  deadline=$(($(date +%s) + ${MEASURE_WAIT_SECONDS:-900}))
  while :; do
    if curl -fsS -o /dev/null --max-time 5 "$url" 2>/dev/null; then
      echo "serve_for_measure: $own is ready" >&2
      return 0
    fi
    if [ -n "$native" ]; then
      if [ -z "$(own_llama_pid "$port")" ]; then
        echo "serve_for_measure: $own stopped before it was ready; its last log lines:" >&2
        break
      fi
    elif [ "$(docker inspect -f '{{.State.Running}}' "$own" 2>/dev/null || true)" = false ]; then
      echo "serve_for_measure: $own stopped before it was ready; its last log lines:" >&2
      break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "serve_for_measure: $own not ready after ${MEASURE_WAIT_SECONDS:-900}s; its last log lines:" >&2
      break
    fi
    sleep "${MEASURE_POLL_SECONDS:-2}"
  done
  if [ -n "$native" ]; then
    tail -n "${MEASURE_LOG_LINES:-40}" "$(llama_log "$port")" >&2 2>/dev/null || true
    if [ -n "$full_log" ]; then
      cp "$(llama_log "$port")" "$full_log" 2>/dev/null || true
      echo "serve_for_measure: the full server log is in $full_log" >&2
    fi
    exit 2
  fi
  docker logs --tail "${MEASURE_LOG_LINES:-40}" "$own" >&2 2>&1 || true
  if [ -n "$full_log" ]; then
    # The last lines rarely reach vLLM's root cause (issue 46): keep it all.
    docker logs "$own" >"$full_log" 2>&1 || true
    echo "serve_for_measure: the full server log is in $full_log" >&2
  fi
  exit 2
}

stop_llama() {
  # stop_llama PORT: TERM the llama-server this helper started on PORT (see
  # own_llama_pid), KILL it after MEASURE_STOP_SECONDS (default 10), and
  # remove its pid file; a stale pid file is removed without signalling
  # anything.
  local port=$1 pid tries=0
  pid=$(own_llama_pid "$port")
  if [ -n "$pid" ]; then
    kill -TERM "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null && [ "$tries" -lt $((${MEASURE_STOP_SECONDS:-10} * 10)) ]; do
      sleep 0.1
      tries=$((tries + 1))
    done
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$(pid_file "$port")"
}

stop() {
  local port=${1:-}
  check_port "$port"
  if [ -f "$(pid_file "$port")" ]; then stop_llama "$port"; fi
  docker rm -f "$(container "$port")" >/dev/null 2>&1 || true
}

case "${1:-}" in
  start) shift; start "$@" ;;
  wait) shift; wait_ready "$@" ;;
  stop) shift; stop "$@" ;;
  *) usage ;;
esac
