# Auto-adjust import paths for tests
# Ensures unit tests can import modules that live in subfolders without packaging the repo.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Paths that contain modules imported directly by tests
paths = [
    ROOT / "llm_fbs_utils",                    # for scenario_runner.py, llm_driver.py
    ROOT / "llm_fbs_utils" / "eval_scripts",  # for evaluate_pipeline.py
    ROOT / "MobieXpert-main" / "src" / "pypbest",  # for llm_rule_patch.py
]

for p in paths:
    if p.exists():
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
            # Debug: confirm at test collection time
            try:
                print(f"[conftest] Added to sys.path: {sp}")
            except Exception:
                pass
