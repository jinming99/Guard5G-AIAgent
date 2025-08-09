# mini03_rule_api_smoke.py
"""
Mini Case 03: Smoke-test LLM Control / Rule API endpoints if reachable.
"""
import os, json
from common.util import get_logger, detect_runtime

def run():
    logger = get_logger("mini03")
    rt = detect_runtime()
    info = {"auditor_up": rt["auditor_up"], "expert_up": rt["expert_up"]}
    logger.info(f"Service health: {info}")
    return {"title": "Mini 03 — API health check", "desc": "Queries /health endpoints (if reachable).", "metrics": {}, "artifacts": [], "notes": [str(info)]}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
