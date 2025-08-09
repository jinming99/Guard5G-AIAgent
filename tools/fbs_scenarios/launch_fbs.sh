#!/usr/bin/env bash
set -euo pipefail

# Minimal stub for local testing/simulator stage.
# Provides start/stop/configure commands with expected output strings.
# Real implementation would control OAI gNB inside Docker or on host.

CMD=${1:-}
shift || true

log() { echo "$@" >&2; }

case "$CMD" in
  start)
    # Arg1: path to JSON config (optional for stub)
    CFG_PATH=${1:-}
    if [[ -n "${CFG_PATH}" && -f "${CFG_PATH}" ]]; then
      log "[stub] Using config: ${CFG_PATH}"
    else
      log "[stub] No config provided; starting with defaults"
    fi
    echo "FBS started"  # string used by scenario_runner to detect success
    ;;

  stop)
    echo "FBS stopped"  # acceptable success marker
    ;;

  configure)
    # Args: param value
    PARAM=${1:-}
    VALUE=${2:-}
    if [[ -z "${PARAM}" || -z "${VALUE}" ]]; then
      echo "Usage: $0 configure <param> <value>" >&2
      exit 2
    fi
    log "[stub] Config ${PARAM}=${VALUE}"
    echo "Configuration updated"  # string used by scenario_runner
    ;;

  *)
    echo "Usage: $0 {start <config.json>|stop|configure <param> <value>}" >&2
    exit 2
    ;;

esac
