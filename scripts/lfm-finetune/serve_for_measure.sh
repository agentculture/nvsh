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
# says so. `start` refuses a port anything already listens on, claims
# MEASURE_RUN_DIR/q46-measure-PORT.pid atomically (noclobber: a concurrent
# start fails cleanly), records the exact argv (NUL-separated) in
# q46-measure-PORT.argv, launches, waits (up to MEASURE_START_SECONDS,
# default 10) until the child runs that argv, then writes its pid and start
# time (/proc/<pid>/stat field 22) into the pid file. If anything fails
# after the launch, the child it launched is stopped and `start` exits
# non-zero. The server's output goes to q46-measure-PORT.log there
# (MEASURE_RUN_DIR defaults to XDG_RUNTIME_DIR, else /tmp). MEASURE_IMAGE,
# TOOL_CALL_PARSER, MEASURE_GPU_FRACTION and MEASURE_MAX_LOGPROBS do not
# apply to it (llama-server has no log-probability cap to raise).
#
# A llama-server is "the one this helper started" only while its pid still
# has the recorded start time and its /proc/<pid>/cmdline is exactly the
# recorded argv (or that argv behind the interpreter a script binary runs
# under): a reused pid, or another invocation of the same binary, never
# qualifies. Every signal is sent only after that check.
#
# With RECORD_JSON, `start` writes the backend ("vllm" or "llama-server") and
# the full argv there, next to the run's results, and for llama-server also
# the binary's path and its --version output: every quant row names its
# serving stack and version.
#
# `wait` polls /v1/models until it answers, for up to MEASURE_WAIT_SECONDS
# (default 900) every MEASURE_POLL_SECONDS (default 2); for llama-server it
# is ready only while the child it started is still that child and
# /v1/models lists its --alias, so a stale server answering on the port is
# never taken for it. On a timeout, or a server that has stopped, it prints
# the server's last log lines (and with FULL_LOG keeps the whole log there)
# and exits 2; for llama-server it also stops the server it waited on
# (identity checked), so a standalone `wait` never leaves one behind.
# `stop` stops the llama-server this helper started on PORT (identity
# checked before TERM and again before KILL) and removes the container.
# Nothing here ever touches a container not named q46-measure-*.
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
argv_file() { echo "$(run_dir)/$NAME_PREFIX$1.argv"; }
llama_log() { echo "$(run_dir)/$NAME_PREFIX$1.log"; }

proc_starttime() {
  # proc_starttime PID: /proc/PID/stat field 22 (start time in clock ticks),
  # or nothing for a pid that is gone or a zombie.
  local stat rest fields
  stat=$(cat "/proc/$1/stat" 2>/dev/null) || return 0
  rest=${stat##*) }
  read -r -a fields <<<"$rest"
  if [ "${fields[0]:-}" = Z ]; then return 0; fi
  echo "${fields[19]:-}"
}

hex_of() { od -An -v -tx1 | tr -d ' \n'; }

is_our_llama() {
  # is_our_llama PID STARTTIME ARGV_FILE: whether PID is still the process
  # `start` launched -- the same start time, and a command line that is
  # exactly the recorded argv (or ends with it at an argument boundary, the
  # interpreter a script binary runs under in front).
  local pid=$1 starttime=$2 file=$3 now actual expected
  now=$(proc_starttime "$pid")
  if [ -z "$now" ] || [ "$now" != "$starttime" ]; then return 1; fi
  expected=$(hex_of <"$file") || return 1
  actual=$(hex_of <"/proc/$pid/cmdline" 2>/dev/null) || return 1
  if [ -z "$expected" ]; then return 1; fi
  [ "$actual" = "$expected" ] || [[ $actual == *"00$expected" ]]
}

own_llama_pid() {
  # own_llama_pid PORT: the pid of the llama-server this helper started on
  # PORT, while it is still that process (is_our_llama), else nothing.
  local file pid='' starttime=''
  file=$(pid_file "$1")
  [ -s "$file" ] || return 0
  { read -r pid; read -r starttime; } <"$file" || true
  if ! [[ $pid =~ ^[0-9]+$ ]] || ! [[ $starttime =~ ^[0-9]+$ ]]; then return 0; fi
  if is_our_llama "$pid" "$starttime" "$(argv_file "$1")"; then echo "$pid"; fi
}

port_busy() {
  # Whether anything accepts a connection on 127.0.0.1:PORT.
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

recorded_alias() {
  # The --alias in PORT's recorded argv.
  local args i
  mapfile -d '' -t args <"$(argv_file "$1")" 2>/dev/null || return 0
  for ((i = 0; i + 1 < ${#args[@]}; i++)); do
    if [ "${args[i]}" = --alias ]; then echo "${args[i + 1]}"; return 0; fi
  done
}

lists_model() {
  # lists_model JSON NAME: whether a /v1/models answer lists NAME.
  python3 -c '
import json, sys
try:
    data = json.loads(sys.argv[1]).get("data") or []
except (ValueError, AttributeError):
    sys.exit(1)
sys.exit(0 if any(isinstance(m, dict) and m.get("id") == sys.argv[2] for m in data) else 1)
' "$1" "$2"
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

claim_port() {
  # claim_port PORT: create PORT's pid file atomically (noclobber), after
  # removing a stale one whose process is gone or no longer ours. An empty
  # pid file is another start in progress and is never taken over.
  local port=$1 pidf
  pidf=$(pid_file "$port")
  if [ -e "$pidf" ]; then
    if [ -s "$pidf" ] && [ -z "$(own_llama_pid "$port")" ]; then
      rm -f "$pidf" "$(argv_file "$port")"
    else
      die "a llama-server this helper started runs (or is starting) on port $port;" \
        "run '$0 stop $port' first"
    fi
  fi
  (set -C; : >"$pidf") 2>/dev/null \
    || die "another start claimed port $port first; run '$0 stop $port' first"
}

# Set while `start` owns a claim or a just-launched child; see rollback_start.
CLAIMED_PORT=''
LAUNCHED_PID=''

# shellcheck disable=SC2317  # invoked by start_llama's EXIT trap
rollback_start() {
  # A start that failed after claiming or launching: stop the child it
  # launched (by the pid `$!` gave it) and release the claim.
  if [ -n "$LAUNCHED_PID" ]; then
    kill -TERM "$LAUNCHED_PID" 2>/dev/null || true
    sleep 0.5
    kill -KILL "$LAUNCHED_PID" 2>/dev/null || true
    echo "serve_for_measure: stopped the llama-server it had just launched (pid $LAUNCHED_PID)" >&2
  fi
  if [ -n "$CLAIMED_PORT" ]; then
    rm -f "$(pid_file "$CLAIMED_PORT")" "$(argv_file "$CLAIMED_PORT")"
  fi
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
  version=$("$binary" --version 2>&1) || die "'$binary --version' failed: $version"
  local name=${MEASURE_MODEL_NAME:-$(basename "$model" .gguf)} pidf
  mkdir -p "$(run_dir)"
  pidf=$(pid_file "$port")
  claim_port "$port"
  CLAIMED_PORT=$port
  trap rollback_start EXIT
  if port_busy "$port"; then
    die "something already listens on 127.0.0.1:$port; it would answer /v1/models in place of" \
      "$model -- stop it or pick another MEASURE_PORT"
  fi
  local argv=("$binary" --model "$model" --host 127.0.0.1 --port "$port"
    --ctx-size "$MEASURE_CTX" --jinja --n-gpu-layers 999 --temp 0 --top-k 1 --alias "$name")
  [ -z "$record" ] \
    || write_llama_record "$record" "$model" "$name" "$port" "$binary" "$version" "${argv[@]}"
  printf '%s\0' "${argv[@]}" >"$(argv_file "$port")" || die "cannot write $(argv_file "$port")"
  nohup "${argv[@]}" </dev/null >"$(llama_log "$port")" 2>&1 &
  LAUNCHED_PID=$!
  # The child runs the binary once it has exec'd; until then its command
  # line is still this shell's.
  local starttime='' tries=0
  while :; do
    starttime=$(proc_starttime "$LAUNCHED_PID")
    [ -n "$starttime" ] || die "llama-server exited at once; see $(llama_log "$port")"
    if is_our_llama "$LAUNCHED_PID" "$starttime" "$(argv_file "$port")"; then break; fi
    [ "$tries" -lt $((${MEASURE_START_SECONDS:-10} * 20)) ] \
      || die "the launched process never ran $binary with the recorded argv"
    sleep 0.05
    tries=$((tries + 1))
  done
  { printf '%s\n%s\n' "$LAUNCHED_PID" "$starttime" >"$pidf.tmp" && mv -f "$pidf.tmp" "$pidf"; } \
    2>/dev/null || die "cannot write $pidf"
  echo "serve_for_measure: started llama-server (pid $LAUNCHED_PID) serving $model as" \
    "'$name' on 127.0.0.1:$port" >&2
  LAUNCHED_PID=''
  CLAIMED_PORT=''
  trap - EXIT
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

wait_llama() {
  # wait_llama PORT FULL_LOG: the llama-server half of wait_ready.
  local port=$1 full_log=$2 own url deadline alias models timed_out=''
  own="llama-server on port $port"
  url="http://127.0.0.1:$port/v1/models"
  alias=$(recorded_alias "$port")
  deadline=$(($(date +%s) + ${MEASURE_WAIT_SECONDS:-900}))
  while :; do
    if [ -z "$(own_llama_pid "$port")" ]; then
      echo "serve_for_measure: $own stopped before it was ready; its last log lines:" >&2
      break
    fi
    # Ready only when the answer lists this child's own alias and the child
    # is still the one started: anything else on the port is not it.
    if models=$(curl -fsS --max-time 5 "$url" 2>/dev/null) && [ -n "$alias" ] \
      && lists_model "$models" "$alias" && [ -n "$(own_llama_pid "$port")" ]; then
      echo "serve_for_measure: $own is ready" >&2
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "serve_for_measure: $own not ready after ${MEASURE_WAIT_SECONDS:-900}s; its last log lines:" >&2
      timed_out=1
      break
    fi
    sleep "${MEASURE_POLL_SECONDS:-2}"
  done
  tail -n "${MEASURE_LOG_LINES:-40}" "$(llama_log "$port")" >&2 2>/dev/null || true
  if [ -n "$full_log" ]; then
    cp "$(llama_log "$port")" "$full_log" 2>/dev/null || true
    echo "serve_for_measure: the full server log is in $full_log" >&2
  fi
  # Never leave a server behind a failed wait (a standalone wait has no
  # caller to stop it); stop_llama signals it only if it is still ours.
  stop_llama "$port"
  if [ -n "$timed_out" ]; then echo "serve_for_measure: stopped $own" >&2; fi
  exit 2
}

wait_ready() {
  local port=${1:-} full_log=${2:-}
  check_port "$port"
  # A pid file means `start` launched the native llama-server on this port.
  if [ -f "$(pid_file "$port")" ]; then wait_llama "$port" "$full_log"; fi
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
  if [ -n "$full_log" ]; then
    # The last lines rarely reach vLLM's root cause (issue 46): keep it all.
    docker logs "$own" >"$full_log" 2>&1 || true
    echo "serve_for_measure: the full server log is in $full_log" >&2
  fi
  exit 2
}

stop_llama() {
  # stop_llama PORT: TERM the llama-server this helper started on PORT,
  # KILL it if it is still that process after MEASURE_STOP_SECONDS (default
  # 10), and remove its pid and argv files. Identity (own_llama_pid) is
  # checked right before each signal; a pid that is no longer ours is never
  # signalled, only its stale files removed.
  local port=$1 pid tries=0
  pid=$(own_llama_pid "$port")
  if [ -n "$pid" ]; then
    kill -TERM "$pid" 2>/dev/null || true
    while [ -n "$(own_llama_pid "$port")" ] && [ "$tries" -lt $((${MEASURE_STOP_SECONDS:-10} * 10)) ]; do
      sleep 0.1
      tries=$((tries + 1))
    done
    pid=$(own_llama_pid "$port")
    if [ -n "$pid" ]; then kill -KILL "$pid" 2>/dev/null || true; fi
  fi
  rm -f "$(pid_file "$port")" "$(argv_file "$port")"
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
