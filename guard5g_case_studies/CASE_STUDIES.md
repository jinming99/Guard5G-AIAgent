# Guard5G — Case Studies & Mini-Demos

This package contains reproducible case studies and small demos to validate the end-to-end setup—
from telemetry synthesis and evaluation, through heuristic rule generation and (optionally) scenario-driven runs.

## Quick Start

```bash
cd guard5g_case_studies
# (optional) use a virtualenv that already satisfies your project requirements
python run_all.py --seed 123
```

Outputs are saved under `outputs/`, including figures and a consolidated `outputs/case_study_report.md`.

> **Note**: The scripts run fully offline by default. If your real stack is up (Redis/SDL, Auditor, Expert, OAI gNB),
> the scripts will detect it and attempt light-touch integrations (health probes, optional launcher).

## What each study shows

### Case Study 01 — Baseline rules on synthetic telemetry
- **Goal:** Show that a static ruleset produces meaningful, explainable metrics on UE-level summaries.
- **What it runs:** Generates synthetic MobiFlow-like records with controlled FBS fraction, aggregates to per-UE, applies rules in `resources/rule_templates/baseline_rules.yaml`.
- **Outputs:** Confusion matrix & basic metric bar plot, CSV of predictions, JSON of metrics & rule hits.
- **Connections to project:** Exercises the detection policy layer (rule evaluation) against data shapes consistent with your pipeline.

### Case Study 02 — Heuristic LLM-generated rule vs. baseline
- **Goal:** Demonstrate that “LLM” (here, a deterministic heuristic) can synthesize a plausible rule from telemetry summary and be evaluated like any other rule.
- **What it runs:** Creates a single heuristic rule using summary statistics (a stand-in for DSPy/OpenAI), compares to baseline.
- **Outputs:** Plots and a JSON comparing metric deltas; can optionally dry‑run patch to the P‑BEST engine if your module is available.

### Case Study 03 — Experiment design and (optional) live FBS scenario
- **Goal:** Show the full loop: “design a run → (optionally) launch via OAI → collect & evaluate.”
- **What it runs:** Produces a `designed_scenario.json`, attempts `launch_fbs.sh scenario ...` if present, otherwise runs offline with synthetic data shaped by the scenario.
- **Outputs:** Scenario JSON, metrics plots, and a JSON summarizing rule hits and whether a real run was attempted.

### Mini 01 — SDL ping
- Attempts a Redis PING to verify SDL reachability; logs a warning if down.

### Mini 02 — KPM/MobiFlow snapshot KPIs
- Offline synthetic playback and UE-level KPI summary to demonstrate data handling and aggregation.

### Mini 03 — API health check
- If your Auditor/Expert services are reachable, records their health; otherwise runs silently offline.

## Offline vs. Real Stack

- **Docker services & wiring:** `redis-sdl`, `mobiflow-auditor`, `mobiexpert`, and (optionally) `oai-gnb` are defined in `docker-compose.yml`. When these are up, the scripts can probe `/health` and (for the OAI path) try launching scenarios. fileciteturn0file1
- **Dependencies:** Python dependencies align with your `requirements.txt` (Flask/requests, pandas/numpy, redis, matplotlib, DSPy/OpenAI, etc.). Ensure your active environment satisfies them. fileciteturn0file2
- **One‑shot setup:** Your `setup.sh` assists with cloning auxiliary repos, copying patches (including `launch_fbs.sh`), and creating config files. The case studies will benefit if this has been run. fileciteturn0file3
- **O‑RAN / OAI integration:** The patched OAI stack exports O‑RAN config JSON and exposes a UNIX socket for runtime GET/SET, enabling light out‑of‑band control/inspection. fileciteturn0file4
- **nr‑softmodem scenario mode:** CLI support for `--scenario-file` (JSON) allows timed events such as TX power adjustments and PCI changes, which our scenarios target. fileciteturn0file6
- **FBS launcher:** `tools/fbs_scenarios/launch_fbs.sh` starts/stops/status the fake gNB, tails logs, and can apply simple config edits (PLMN, PCI, power) or run a scenario from JSON. Our CS‑03 uses this if available. fileciteturn0file7
- **Prompt templates:** If you want to swap the heuristic rule generator with a real DSPy/OpenAI flow, reuse your prompt templates (rule generation, experiment design, analysis) for consistency. fileciteturn0file8

## Reproducibility & Variance Control

- All scripts accept a global `--seed`; inside, we set RNGs deterministically.
- Synthetic generator uses a fixed RNG for base behavior; the global seed controls additional sampling variation (e.g., subset sizes).
- We provide summary metrics and plots designed to be stable across runs with the same seed.

## How to run individual studies

```bash
# Run just Case Study 01
python cs01_baseline_rules_eval.py

# Run just Case Study 02
python cs02_llm_rulegen_eval.py

# Run just Case Study 03
python cs03_llm_experiment_design.py

# Mini demos
python mini01_sdl_ping.py
python mini02_kpm_playback_kpis.py
python mini03_rule_api_smoke.py
```

Artifacts are collected under `outputs/`. Each script prints a small JSON block with a summary and the artifact paths.

## Expected behavior (baseline)

- Case Study 01: Baseline rules achieve non‑trivial precision/recall on synthetic data (expect F1 around mid‑range). A confusion matrix and bar chart are produced.
- Case Study 02: The heuristic “LLM” rule performs similarly to, sometimes better than, the baseline for certain seeds; JSON shows both metric sets and the generated rule thresholds.
- Case Study 03: If the OAI launcher exists, the script will invoke it for a short scenario (non‑blocking) and then proceed with evaluation. Otherwise, it logs a note and runs offline.

## Debugging tips

- Check `outputs/logs/*.log` for detailed traces.
- If Redis is down, **Mini 01** will log a warning; this is expected in offline runs.
- If Auditor/Expert are down, **Mini 03** will simply report `auditor_up=False`, `expert_up=False` (offline expected).
- If `launch_fbs.sh` is missing or not executable, **CS‑03** logs a warning and remains offline.
- For P‑BEST patching, the rule patch step runs in **dry‑run** mode when the engine is not importable; review the `metrics.json` / `compare_metrics.json` for the patch result echo.

## Mapping to the end goal

These studies are stepping stones toward the full loop:

**(LLM proposes replays / sim)** → **collects telemetry (MobiFlow/KPM)** → **analyzes** → **generates & hot‑loads P‑BEST rules** → **evaluates and summarizes**.

- **CS‑03** touches the “propose & (attempt to) run” part via scenario JSON + launcher.
- **CS‑01/02** validate “analyze → generate rules → evaluate” with deterministic data and robust metrics.
- Mini demos confirm interfaces are reachable when the real stack is up.

## Advanced: swapping the heuristic with real DSPy/OpenAI

- Point your orchestrator to `prompt_templates/rule_generation.yaml` and `experiment_design_prompt` to have a model propose rules/scenarios. fileciteturn0file8
- Ensure Docker services (`mobiflow-auditor`, `mobiexpert`) are running per your Compose and that `.env` is configured. fileciteturn0file1
- If using the OAI path, make sure the patched stack is built (see `setup.sh`) so `launch_fbs.sh` exists and logs to `/tmp/fbs_logs/fbs_gnb.log`. fileciteturn0file3 fileciteturn0file7

---

_This document and the scripts are designed to be copy‑paste ready into your repo under a `case_studies/` folder. Adjust paths as needed._
