#!/usr/bin/env bash
set -euo pipefail

LOG="/tmp/fbs_logs/fbs_gnb.log"
if [[ -f "$LOG" ]]; then
  echo "[STEP] Checking scenario banner in $LOG"
  if grep -q "=== RUNNING IN SCENARIO MODE ===" "$LOG"; then
    echo "[OK] Scenario mode banner found"
    exit 0
  else
    echo "[WARN] Scenario banner not found (this is OK if you're using docker 'oai-gnb')"
    echo "[INFO] Last 60 lines for context:"
    tail -n 60 "$LOG" || true
    exit 0
  fi
else
  echo "[INFO] Log $LOG not found; likely running via compose or not started. Skipping."
  exit 0
fi
