# shellcheck shell=bash
# Sourced by pipeline.sh: run a heavy stage under a hard memory cap (issue 46, c49).
#
# The cap is RAM *and* swap. MemoryMax alone is not a hard cap on a box with
# swap: a trainer over the limit spills into swap and keeps running, slowing
# everything else on the machine (spark2 serves models next to training).
# Measured 2026-09-23 on spark and spark2: 600M of touched pages ran to
# completion under MemoryMax=200M, and was killed once MemorySwapMax=0 was set.
#
# The cap alone does not contain a GPU trainer on unified memory. Measured
# 2026-09-23 on spark2 (GB10): an 8 GB CUDA allocation succeeded inside a scope
# capped at MemoryMax=1G, because GPU allocations are not charged to the
# cgroup, yet on GB10/Jetson the GPU and CPU share one physical pool. So a
# watchdog also stops the run once the machine's MemAvailable falls below
# TRAIN_MEMORY_FLOOR (default 8G), before the serving stack is starved (h33).

#: run_capped's status when the watchdog stopped the command.
RUN_CAPPED_WATCHDOG_STATUS=3

free_mem_line() {
  # One free-memory line, unified-memory-friendly (Jetson/GB10 share system
  # RAM with the GPU, so "free" here is the number that matters).
  free -h | awk -v ts="$(date -u +%FT%TZ)" '/^Mem:/{print ts, "free="$4, "avail="$7}'
}

mem_available_kb() {
  awk '/^MemAvailable:/{print $2; exit}' /proc/meminfo
}

size_to_kb() {
  # size_to_kb SIZE: a positive whole size with an optional K/M/G/T suffix
  # (binary units; no suffix means bytes) in KiB, or status 1 if unreadable.
  local size=$1 n unit
  [[ $size =~ ^([0-9]+)([KMGTkmgt]?)$ ]] || return 1
  n=$((10#${BASH_REMATCH[1]}))
  unit=${BASH_REMATCH[2]^^}
  case $unit in
    "") n=$((n / 1024)) ;;
    K) ;;
    M) n=$((n * 1024)) ;;
    G) n=$((n * 1024 * 1024)) ;;
    T) n=$((n * 1024 * 1024 * 1024)) ;;
  esac
  [ "$n" -gt 0 ] || return 1
  echo "$n"
}

_watch_memory() {
  # _watch_memory MEM_LOG FLOOR_KB FLOOR INTERVAL PGID_FILE TRIP_FILE: log free
  # memory every 60s; check MemAvailable every INTERVAL seconds and, below
  # FLOOR_KB, record why in MEM_LOG and TRIP_FILE, then stop the command's
  # process group (SIGTERM, SIGKILL after 10 s) and exit.
  local mem_log=$1 floor_kb=$2 floor=$3 interval=$4 pgid_file=$5 trip_file=$6
  local nap avail pgid elapsed=0 i
  trap 'kill "$nap" 2>/dev/null; exit 0' TERM
  while :; do
    avail=$(mem_available_kb)
    if [ -n "$avail" ] && [ "$avail" -lt "$floor_kb" ]; then
      echo "$(date -u +%FT%TZ) watchdog: MemAvailable $((avail / 1024))M below floor $floor, stopping" \
        | tee -a "$mem_log" > "$trip_file"
      # run_capped waits for us once TRIP_FILE exists; finish the stop.
      trap '' TERM
      for ((i = 0; i < 100; i++)); do
        [ -s "$pgid_file" ] && break
        sleep 0.1
      done
      pgid=$(cat "$pgid_file" 2>/dev/null) || exit 0
      [ -n "$pgid" ] || exit 0
      kill -TERM -- "-$pgid" 2>/dev/null || exit 0
      for ((i = 0; i < 10; i++)); do
        sleep 1
        kill -0 -- "-$pgid" 2>/dev/null || exit 0
      done
      kill -KILL -- "-$pgid" 2>/dev/null
      exit 0
    fi
    sleep "$interval" & nap=$!; wait "$nap"
    elapsed=$((elapsed + interval))
    if [ "$elapsed" -ge 60 ]; then
      free_mem_line >> "$mem_log"
      elapsed=0
    fi
  done
}

run_capped() {
  # run_capped RUN_DIR CMD...: run CMD capped at $TRAIN_MEMORY_MAX (RAM + swap),
  # logging free memory to RUN_DIR/mem.log before the run and every 60s during
  # it, and CMD's output to RUN_DIR/train.log. Without systemd-run it refuses,
  # unless TRAIN_MEMORY_CAP=container says a container --memory cap applies.
  # A watchdog stops CMD once MemAvailable falls below $TRAIN_MEMORY_FLOOR
  # (default 8G, checked every $TRAIN_WATCHDOG_SECONDS, default 5); run_capped
  # then returns $RUN_CAPPED_WATCHDOG_STATUS.
  local run_dir=$1; shift
  : "${TRAIN_MEMORY_MAX:?TRAIN_MEMORY_MAX must be set (e.g. 24G)}"
  local floor=${TRAIN_MEMORY_FLOOR:-8G} interval=${TRAIN_WATCHDOG_SECONDS:-5} floor_kb
  if ! floor_kb=$(size_to_kb "$floor"); then
    echo "run_capped: TRAIN_MEMORY_FLOOR=$floor is not a positive size (e.g. 8G, 512M)" >&2
    return 2
  fi
  if ! [[ $interval =~ ^[1-9][0-9]*$ ]]; then
    echo "run_capped: TRAIN_WATCHDOG_SECONDS=$interval is not a positive whole number" >&2
    return 2
  fi
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
  local mem_log="$run_dir/mem.log" out_log="$run_dir/train.log" state
  state=$(mktemp -d)
  local pgid_file="$state/pgid" trip_file="$state/trip"
  free_mem_line >> "$mem_log"
  # The watcher never holds the caller's stdout/stderr open, and its sleep is
  # killed with it, so run_capped returns as soon as CMD does.
  _watch_memory "$mem_log" "$floor_kb" "$floor" "$interval" "$pgid_file" "$trip_file" \
    > /dev/null 2>&1 &
  local watcher=$!
  local status
  set +e
  # CMD runs as the leader of its own session (so its own process group), and
  # the watchdog signals that whole group. A process group rather than the
  # systemd scope, because it is the one handle both cap modes have: with
  # TRAIN_MEMORY_CAP=container there is no scope to stop. systemd-run --scope
  # execs CMD in place, so the recorded pid is CMD's; tee stays in our group,
  # so it drains CMD's last output after the stop.
  # shellcheck disable=SC2016 # $$ and $@ expand in the inner bash
  setsid -w bash -c 'echo "$$" > "$0"; exec "$@"' "$pgid_file" "${cap[@]}" "$@" 2>&1 \
    | tee "$out_log"
  status=${PIPESTATUS[0]}
  set -e
  if [ -s "$trip_file" ]; then
    cat "$trip_file" >&2
    wait "$watcher" 2>/dev/null || true
    status=$RUN_CAPPED_WATCHDOG_STATUS
  else
    kill "$watcher" 2>/dev/null || true
    wait "$watcher" 2>/dev/null || true
  fi
  rm -rf "$state"
  return "$status"
}
