#!/usr/bin/env python3
"""
Smoke test ScenarioRunner in local mode using a stub launch_fbs.sh.
"""

import os
import json
import tempfile
import logging
from scenario_runner import ScenarioRunner, ScenarioConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_scenario_runner")

STUB = """#!/usr/bin/env bash
set -euo pipefail
CMD=${1:-}
shift || true
if [[ "$CMD" == "start" ]]; then
  echo "[STUB] FBS started"; exit 0
elif [[ "$CMD" == "stop" ]]; then
  echo "[STUB] FBS stopped"; exit 0
elif [[ "$CMD" == "status" ]]; then
  echo "[STUB] FBS status OK"; exit 0
elif [[ "$CMD" == "configure" ]]; then
  echo "[STUB] Configuration updated"; exit 0
else
  echo "[STUB] unknown command $CMD"; exit 1
fi
"""

def main():
    with tempfile.TemporaryDirectory() as d:
        tools = os.path.join(d, "tools", "fbs_scenarios")
        os.makedirs(tools, exist_ok=True)
        sh = os.path.join(tools, "launch_fbs.sh")
        with open(sh, "w") as f: f.write(STUB)
        os.chmod(sh, 0o755)

        os.environ["OAI_PATH"] = d
        runner = ScenarioRunner(use_docker=False, use_ssh=False)

        ok = runner.start_fbs({})
        assert ok, "start_fbs failed"

        scenario = ScenarioConfig(
            name="Local Test",
            mode="fbs",
            duration=2,
            config={"plmn":"00101","pci":1},
            events=[{"time":1,"action":"configure","param":"pci","value":2}]
        )
        runner.run_scenario(scenario)

        ok = runner.stop_fbs()
        assert ok, "stop_fbs failed"
        logger.info("ScenarioRunner local stub OK")

if __name__ == "__main__":
    main()
