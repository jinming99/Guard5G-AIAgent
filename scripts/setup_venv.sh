#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/setup_venv.sh [--dev] [--full] [--python PYTHON] [--requirements PATH]
# Defaults:
#   --dev installs requirements-dev.txt (fast)
#   --full installs requirements.txt (full stack)
#   --python auto-detects python3
#   --requirements overrides the file to install from

MODE="dev"    # dev|full
PYBIN="${PYBIN:-}"
REQ_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) MODE="dev"; shift ;;
    --full) MODE="full"; shift ;;
    --python) PYBIN="$2"; shift 2 ;;
    --requirements) REQ_OVERRIDE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "${PYBIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYBIN="python3"; 
  elif command -v python >/dev/null 2>&1; then PYBIN="python";
  else echo "[ERR] No python found in PATH"; exit 1; fi
fi

REQ=""
if [[ -n "${REQ_OVERRIDE}" ]]; then
  REQ="${REQ_OVERRIDE}"
else
  if [[ "${MODE}" == "full" ]]; then
    if [[ -f "requirements.txt" ]]; then REQ="requirements.txt"; 
    elif [[ -f "llm_fbs_utils/requirements.txt" ]]; then REQ="llm_fbs_utils/requirements.txt";
    else echo "[ERR] requirements.txt not found"; exit 1; fi
  else
    if [[ -f "requirements-dev.txt" ]]; then REQ="requirements-dev.txt";
    else echo "[ERR] requirements-dev.txt not found. Use --full or create it."; exit 1; fi
  fi
fi

echo "[INFO] Using Python: ${PYBIN}"
echo "[INFO] Installing from: ${REQ}"

if [[ ! -d ".venv" ]]; then
  echo "[INFO] Creating virtualenv .venv"
  "${PYBIN}" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r "${REQ}"

echo "[OK] Virtualenv ready. Activate with:  source .venv/bin/activate"
