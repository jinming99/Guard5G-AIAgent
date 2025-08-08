#!/usr/bin/env bash
set -euo pipefail

# This script checks Python deps and (optionally) starts docker-compose services
# and/or launches the host-side OAI Fake gNB via launch_fbs.sh.
#
# Controls:
#   USE_COMPOSE=1            -> prefer docker-compose services (default: auto if compose file exists)
#   STAGED_AUTOSTART_SERVICES=1 -> auto-start services (redis, auditor, expert, and oai-gnb if applicable)
#   COMPOSE_FILE=llm_fbs_utils/docker-compose.yml
#   LAUNCH=<path to OAI-5G/tools/fbs_scenarios/launch_fbs.sh>
#   SCEN_FILE=tests/tmp/staged_scenario.json
#
# Exit 0 with diagnostics even if some services are unavailable; subsequent stages
# will perform deeper checks and fail with detailed logs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-llm_fbs_utils/docker-compose.yml}"
SCEN_FILE="${SCEN_FILE:-tests/tmp/staged_scenario.json}"
LOG_DIR="tests/logs"
mkdir -p "${LOG_DIR}" tests/tmp

# Auto-detect compose preference if not set
if [[ -z "${USE_COMPOSE:-}" ]]; then
  if [[ -f "${REPO_ROOT}/${COMPOSE_FILE}" ]]; then
    USE_COMPOSE=1
  else
    USE_COMPOSE=0
  fi
fi

STAGED_AUTOSTART_SERVICES="${STAGED_AUTOSTART_SERVICES:-1}"

echo "[INFO] USE_COMPOSE=${USE_COMPOSE} STAGED_AUTOSTART_SERVICES=${STAGED_AUTOSTART_SERVICES}"
echo "[INFO] COMPOSE_FILE=${COMPOSE_FILE}"

echo "[STEP] Python environment check"
python3 - <<'PY'
import pkgutil, sys, platform
req = ["requests","redis","yaml","pandas","numpy","pytest"]
missing = [m for m in req if pkgutil.find_loader(m) is None]
print("Python", platform.python_version())
print("Required packages:", ", ".join(req))
print("Missing:", missing if missing else "None")
PY

if [[ "${STAGED_AUTOSTART_SERVICES}" == "1" ]]; then
  if [[ "${USE_COMPOSE}" == "1" ]]; then
    echo "[STEP] Starting compose services: redis-sdl, mobiflow-auditor, mobiexpert"
    if command -v docker compose >/dev/null 2>&1; then
      docker compose -f "${COMPOSE_FILE}" up -d redis-sdl mobiflow-auditor mobiexpert || true
      docker compose -f "${COMPOSE_FILE}" up -d oai-gnb || true
    else
      docker-compose -f "${COMPOSE_FILE}" up -d redis-sdl mobiflow-auditor mobiexpert || true
      docker-compose -f "${COMPOSE_FILE}" up -d oai-gnb || true
    fi
    echo "[INFO] Waiting briefly for services to initialize..."
    sleep 5
    echo "[STEP] mobiflow-auditor /health"
    curl -fsS http://localhost:8090/health || echo "[WARN] Auditor may not be up yet"
    echo "[STEP] mobiexpert /health"
    curl -fsS http://localhost:8091/health || echo "[WARN] Expert may not be up yet"
  fi

  # Host-based Fake gNB via launch_fbs.sh
  if [[ -n "${LAUNCH:-}" && -x "${LAUNCH}" ]]; then
    echo "[STEP] Detected LAUNCH=${LAUNCH}"
    cat > "${SCEN_FILE}" <<EOF
{
  "name": "Staged FBS Smoke",
  "mode": "fbs",
  "duration": 60,
  "config": { "pci": 222, "tx_power": 12 }
}
EOF
    echo "[STEP] Starting fake gNB via launch_fbs.sh with scenario: ${SCEN_FILE}"
    "${LAUNCH}" start "${SCEN_FILE}" || echo "[WARN] launch_fbs.sh start failed (see /tmp/fbs_logs/fbs_gnb.log if exists)"
  else
    echo "[INFO] LAUNCH not set or not executable; skipping host-based gNB"
  fi
fi

echo "[STEP] Redis ping (host)"
if command -v redis-cli >/dev/null 2>&1; then
  (redis-cli -h "${REDIS_HOST:-127.0.0.1}" -p "${REDIS_PORT:-6379}" ping || echo "[WARN] Redis ping failed") | sed 's/^/[REDIS] /'
else
  echo "[WARN] redis-cli not found; using docker service if present"
fi

# Diagnostics for gNB
if [[ -f "/tmp/fbs_gnb.pid" ]]; then
  echo "[INFO] Host fake gNB appears running. Tail last lines of log:"
  tail -n 30 /tmp/fbs_logs/fbs_gnb.log || true
else
  echo "[INFO] Host fake gNB not detected (OK if using compose)."
fi

# Diagnostics for compose gNB
if command -v docker >/dev/null 2>&1; then
  if docker ps --format '{{.Names}}' | grep -q '^fbs-gnb$'; then
    echo "[INFO] Compose gNB container 'fbs-gnb' is running."
    docker logs --tail=30 fbs-gnb || true
  else
    echo "[INFO] Compose gNB container 'fbs-gnb' not running."
  fi
fi

echo "[OK] env check done"
