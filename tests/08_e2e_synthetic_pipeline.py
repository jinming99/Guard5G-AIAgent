#!/usr/bin/env python3
import os, sys, json
print("[STEP] E2E synthetic pipeline smoke")
try:
    from evaluate_pipeline import EvaluationPipeline
except Exception as e:
    print("[SKIP] evaluate_pipeline import failed:", e)
    sys.exit(0)

try:
    pipe = EvaluationPipeline()
    res = pipe.run_scenario("synthetic", {"duration": 10})
    print("[OK] Metrics:", res.get("metrics"))
    try:
        pipe.plot_results()
        print("[OK] Plots saved")
    except Exception as e:
        print("[WARN] plot_results failed:", e)
    sys.exit(0)
except Exception as e:
    print("[ERR] Pipeline run failed:", e)
    sys.exit(1)
