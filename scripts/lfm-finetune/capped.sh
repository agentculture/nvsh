# shellcheck shell=bash
# Sourced by pipeline.sh: run a heavy stage under a hard memory cap (issue 46, c49).
#
# The cap is RAM *and* swap. MemoryMax alone is not a hard cap on a box with
# swap: a trainer over the limit spills into swap and keeps running, slowing
# everything else on the machine (spark2 serves models next to training).
# Measured 2026-09-23 on spark and spark2: 600M of touched pages ran to
# completion under MemoryMax=200M, and was killed once MemorySwapMax=0 was set.

free_mem_line() {
  # One free-memory line, unified-memory-friendly (Jetson/GB10 share system
  # RAM with the GPU, so "free" here is the number that matters).
  free -h | awk -v ts="$(date -u +%FT%TZ)" '/^Mem:/{print ts, "free="$4, "avail="$7}'
}

run_capped() {
  # run_capped RUN_DIR CMD...: run CMD capped at $TRAIN_MEMORY_MAX (RAM + swap),
  # logging free memory to RUN_DIR/mem.log before the run and every 60s during
  # it, and CMD's output to RUN_DIR/train.log. Without systemd-run it refuses,
  # unless TRAIN_MEMORY_CAP=container says a container --memory cap applies.
  local run_dir=$1; shift
  : "${TRAIN_MEMORY_MAX:?TRAIN_MEMORY_MAX must be set (e.g. 24G)}"
  local -a cap
  if command -v systemd-run >/dev/null 2>&1; then
    cap=(systemd-run --user --scope --quiet -p "MemoryMax=$TRAIN_MEMORY_MAX" -p MemorySwapMax=0 --)
  elif [ "${TRAIN_MEMORY_CAP:-}" = container ]; then
    cap=()
  else
    echo "run_capped: no systemd-run to cap memory; set TRAIN_MEMORY_CAP=container only if a container --memory cap applies" >&2
    return 2
  fi
  mkdir -p "$run_dir"
  local mem_log="$run_dir/mem.log" out_log="$run_dir/train.log"
  free_mem_line >> "$mem_log"
  # The watcher never holds the caller's stdout/stderr open, and its sleep is
  # killed with it, so run_capped returns as soon as CMD does.
  ( trap 'kill "$nap" 2>/dev/null; exit 0' TERM
    while :; do sleep 60 & nap=$!; wait "$nap"; free_mem_line >> "$mem_log"; done
  ) > /dev/null 2>&1 &
  local watcher=$!
  local status
  set +e
  "${cap[@]}" "$@" 2>&1 | tee "$out_log"
  status=${PIPESTATUS[0]}
  set -e
  kill "$watcher" 2>/dev/null || true
  wait "$watcher" 2>/dev/null || true
  return "$status"
}
