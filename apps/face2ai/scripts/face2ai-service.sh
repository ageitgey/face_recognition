#!/usr/bin/env bash
# face2ai-service.sh {start|stop|status} — own the local Face2AI process from outside Python.
#
# This is the testable unit under the macOS launcher (ADR-005): the AppleScript applet is a thin
# shell over `start` / `stop`, and everything that can fail is here, where it can be exercised from
# a terminal on a spare port.
#
# Contract:
#   start   Refuses to start a second process (probes /healthz first), launches the app detached
#           with its output appended to $LOG_FILE and its pid in $PID_FILE, then blocks until
#           /healthz answers. On timeout it prints the tail of the log, stops what it started and
#           exits non-zero — start is atomic: either the service is up or nothing is left behind.
#   stop    SIGTERM to the launched process and its descendants, wait, escalate to SIGKILL after a
#           bound, remove the pid file. Idempotent: a clean no-op when nothing runs. It never kills
#           a process it did not start — if /healthz answers but our pid file records nothing live,
#           it refuses (the product's real port carries a live backend, a voice agent and a tunnel).
#   status  Prints the /healthz JSON on the first line, /readyz on the second, exits non-zero when down.
#
# Environment (all optional):
#   FACE2AI_HOST      (default 127.0.0.1)  host to bind and to probe
#   FACE2AI_PORT      (default 8765)       port to bind and to probe
#   FACE2AI_DATA_DIR  (default ~/.face2ai) also holds face2ai.log and face2ai.pid
#   FACE2AI_START_TIMEOUT_SECONDS (default 60)  how long start waits for /healthz
#   FACE2AI_STOP_TIMEOUT_SECONDS  (default 10)  how long stop waits before SIGKILL
# The first three are exported to the launched process, so the URL this script probes is by
# construction the URL the process binds.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

FACE2AI_HOST="${FACE2AI_HOST:-127.0.0.1}"
FACE2AI_PORT="${FACE2AI_PORT:-8765}"
FACE2AI_DATA_DIR="${FACE2AI_DATA_DIR:-$HOME/.face2ai}"
export FACE2AI_HOST FACE2AI_PORT FACE2AI_DATA_DIR

START_TIMEOUT="${FACE2AI_START_TIMEOUT_SECONDS:-60}"
STOP_TIMEOUT="${FACE2AI_STOP_TIMEOUT_SECONDS:-10}"

BASE_URL="http://${FACE2AI_HOST}:${FACE2AI_PORT}"
LOG_FILE="${FACE2AI_DATA_DIR}/face2ai.log"
PID_FILE="${FACE2AI_DATA_DIR}/face2ai.pid"

# Finder/launchd hand the applet a minimal PATH, so resolve the tools by absolute path too.
CURL="$(command -v curl 2>/dev/null || true)"
[ -n "$CURL" ] || CURL=/usr/bin/curl

usage() {
  echo "usage: ${0##*/} {start|stop|status}" >&2
}

is_up() {
  "$CURL" -fs -o /dev/null -m 2 "${BASE_URL}/healthz" 2>/dev/null
}

# Print the pid recorded by a previous start, but only when that process is still alive.
recorded_pid() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid="$(tr -dc '0-9' < "$PID_FILE")"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s' "$pid"
}

# `uv run` forks the interpreter as a child, so the launched pid is not always the server pid.
process_tree() {
  local pid="$1" child
  printf '%s\n' "$pid"
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    process_tree "$child"
  done
}

any_alive() {
  local pid
  for pid in "$@"; do
    if kill -0 "$pid" 2>/dev/null; then return 0; fi
  done
  return 1
}

# SIGTERM the process and its descendants; escalate to SIGKILL after STOP_TIMEOUT.
terminate() {
  local root="$1" pid waited=0
  local pids
  pids="$(process_tree "$root" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  set -- $pids
  for pid in "$@"; do kill -TERM "$pid" 2>/dev/null || true; done
  while [ "$waited" -lt "$((STOP_TIMEOUT * 4))" ]; do
    if ! any_alive "$@"; then return 0; fi
    sleep 0.25
    waited=$((waited + 1))
  done
  echo "face2ai: still alive ${STOP_TIMEOUT}s after SIGTERM — escalating to SIGKILL" >&2
  for pid in "$@"; do kill -KILL "$pid" 2>/dev/null || true; done
  waited=0
  while [ "$waited" -lt 8 ]; do
    if ! any_alive "$@"; then return 0; fi
    sleep 0.25
    waited=$((waited + 1))
  done
  echo "face2ai: pid ${root} survived SIGKILL" >&2
  return 1
}

# Prefer the project venv: one pid, signals land on the server itself. Fall back to uv, which
# bootstraps the environment but interposes a supervisor process.
SERVER_CMD=()
resolve_server_command() {
  if [ -x "${PROJECT_DIR}/.venv/bin/face2ai" ]; then
    SERVER_CMD=("${PROJECT_DIR}/.venv/bin/face2ai")
    return 0
  fi
  local candidate
  for candidate in "$(command -v uv 2>/dev/null || true)" "$HOME/.local/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      SERVER_CMD=("$candidate" run --project "$PROJECT_DIR" face2ai)
      return 0
    fi
  done
  echo "face2ai: neither ${PROJECT_DIR}/.venv/bin/face2ai nor uv found — run 'uv sync --project ${PROJECT_DIR} --extra recognition' first" >&2
  return 1
}

cmd_start() {
  mkdir -p "$FACE2AI_DATA_DIR"
  if is_up; then
    local running
    running="$(recorded_pid || true)"
    echo "face2ai already running${running:+ (pid ${running})} on ${BASE_URL} — not starting a second one"
    return 0
  fi
  local stale
  stale="$(recorded_pid || true)"
  if [ -n "$stale" ]; then
    echo "face2ai: pid ${stale} from ${PID_FILE} is alive but ${BASE_URL}/healthz does not answer — run '${0##*/} stop' first" >&2
    return 1
  fi
  rm -f "$PID_FILE"
  resolve_server_command || return 1

  printf '\n=== face2ai start %s — %s — %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${SERVER_CMD[*]}" "$BASE_URL" >> "$LOG_FILE"
  nohup "${SERVER_CMD[@]}" >> "$LOG_FILE" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"

  local waited=0
  while [ "$waited" -lt "$((START_TIMEOUT * 4))" ]; do
    if is_up; then
      echo "face2ai started (pid ${pid}) on ${BASE_URL} — log ${LOG_FILE}"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "face2ai: the process exited during startup — last lines of ${LOG_FILE}:" >&2
      tail -n 20 "$LOG_FILE" >&2
      rm -f "$PID_FILE"
      return 1
    fi
    sleep 0.25
    waited=$((waited + 1))
  done
  echo "face2ai: ${BASE_URL}/healthz did not answer within ${START_TIMEOUT}s — last lines of ${LOG_FILE}:" >&2
  tail -n 20 "$LOG_FILE" >&2
  terminate "$pid" || true
  rm -f "$PID_FILE"
  return 1
}

cmd_stop() {
  local pid
  pid="$(recorded_pid || true)"
  if [ -z "$pid" ]; then
    rm -f "$PID_FILE"
    if is_up; then
      echo "face2ai: ${BASE_URL}/healthz answers but ${PID_FILE} records no live process of ours — refusing to kill a process this script did not start" >&2
      return 1
    fi
    echo "face2ai not running"
    return 0
  fi
  terminate "$pid"
  rm -f "$PID_FILE"
  if is_up; then
    echo "face2ai: ${BASE_URL}/healthz still answers after stopping pid ${pid} — another process holds the port" >&2
    return 1
  fi
  echo "face2ai stopped (pid ${pid})"
  return 0
}

cmd_status() {
  local health ready pid
  if ! health="$("$CURL" -fsS -m 3 "${BASE_URL}/healthz" 2>/dev/null)"; then
    echo "face2ai down — ${BASE_URL}/healthz unreachable" >&2
    return 1
  fi
  echo "$health"
  # /readyz answers 503 while the recognition engine is missing, so do not use -f here.
  ready="$("$CURL" -sS -m 3 "${BASE_URL}/readyz" 2>/dev/null || true)"
  if [ -n "$ready" ]; then echo "$ready"; fi
  pid="$(recorded_pid || true)"
  echo "# url ${BASE_URL}  pid ${pid:-unknown}  log ${LOG_FILE}"
  return 0
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  *) usage; exit 2 ;;
esac
