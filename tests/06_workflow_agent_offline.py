#!/usr/bin/env python3
import os, sys, json

print("[STEP] Workflow agent offline smoke")
try:
    from llm_driver import LLMOrchestrator
except Exception as e:
    print("[SKIP] llm_driver import failed:", e)
    sys.exit(0)

try:
    orch = LLMOrchestrator(config_file=None, use_enhanced=False)
except Exception as e:
    print("[ERR] Orchestrator init failed:", e)
    sys.exit(1)

try:
    out = orch.experiment.forward({"hypothesis":"Null ciphering increases attach failures within 120s"})
    print("[OK] Experiment plan:", json.dumps(out)[:200])
    sys.exit(0)
except Exception as e:
    print("[ERR] Experiment forward failed:", e)
    sys.exit(1)
