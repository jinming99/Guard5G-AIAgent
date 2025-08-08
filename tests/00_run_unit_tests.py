#!/usr/bin/env python3
import sys, os, subprocess, glob, pathlib
import pytest
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Load environment variables from project .env if present
load_dotenv(dotenv_path=ROOT / ".env")

def run_pytest() -> int:
    # Run all unit tests under tests/unit (verbose, do not stop on fail)
    args = ["-vv", "tests/unit"]
    return pytest.main(args)

def fallback_run_scripts() -> int:
    """Run script-style tests directly when pytest finds none.
    We set PYTHONPATH so imports like evaluate_pipeline, llm_rule_patch resolve.
    """
    pybin = sys.executable
    env = os.environ.copy()
    # Build PYTHONPATH with project subdirs used by tests
    add_paths = [
        ROOT / "llm_fbs_utils",
        ROOT / "llm_fbs_utils" / "eval_scripts",
        ROOT / "MobieXpert-main" / "src" / "pypbest",
    ]
    # Prepend to PYTHONPATH
    existing = env.get("PYTHONPATH", "")
    prepend = os.pathsep.join(str(p) for p in add_paths if p.exists())
    env["PYTHONPATH"] = prepend + (os.pathsep + existing if existing else "")

    # Ensure prompt YAML path is set for prompt template test
    yml = ROOT / "llm_fbs_utils" / "prompt_templates" / "rule_generation.yaml"
    if yml.exists() and not env.get("PROMPT_YAML"):
        env["PROMPT_YAML"] = str(yml)

    test_files = sorted(glob.glob(str(ROOT / "tests" / "unit" / "test_*.py")))
    if not test_files:
        print("[unit-runner] No test files found under tests/unit")
        return 5

    failures = 0
    for tf in test_files:
        print(f"[unit-runner] Running script: {tf}")
        rc = subprocess.call([pybin, tf], env=env, cwd=str(ROOT))
        if rc != 0:
            print(f"[unit-runner] FAILED: {tf} (rc={rc})")
            failures += 1
        else:
            print(f"[unit-runner] PASSED: {tf}")

    return 1 if failures else 0

def main():
    rc = run_pytest()
    if rc == 0:
        raise SystemExit(0)
    # Pytest exit code 5 means "No tests collected"
    if rc == 5:
        print("[unit-runner] Pytest collected 0 tests; falling back to script execution...")
        raise SystemExit(fallback_run_scripts())
    # Otherwise, propagate pytest failures
    raise SystemExit(rc)

if __name__ == "__main__":
    main()
