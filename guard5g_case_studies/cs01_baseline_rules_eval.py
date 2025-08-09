# cs01_baseline_rules_eval.py
"""
Case Study 01
Baseline detection rules vs. synthetic telemetry. Computes metrics and builds plots.
"""

import os, json
import pandas as pd
from common.util import get_logger, set_seed, detect_runtime, ensure_dir
from common.data import SynthConfig, generate_mobiflow_synthetic, summarize
from common.rule_engine import load_rules_yaml, apply_rules, dryrun_patch_to_engine
from common.metrics import summarize_metrics_table
from common.viz import plot_confusion, plot_metric_bars

def run(output_dir="outputs/cs01", seed=123):
    logger = get_logger("cs01")
    ensure_dir(output_dir)
    set_seed(seed)
    rt = detect_runtime()
    logger.info(f"Runtime flags: {rt}")

    # 1) Data
    cfg = SynthConfig(n_ue=80, duration_sec=180, fbs_fraction=0.3)
    df = generate_mobiflow_synthetic(cfg)
    snap = summarize(df)
    snap["label"] = snap["suspected_fbs"].astype(int)

    # 2) Rules (baseline templates)
    rules = load_rules_yaml("resources/rule_templates/baseline_rules.yaml")
    logger.info(f"Loaded {len(rules)} baseline rules")

    # 3) Apply rules
    out, hits = apply_rules(snap, rules)
    logger.info(f"Rule hits: {hits}")

    # 4) Metrics
    out["pred"] = out["pred"].astype(int)
    metrics = summarize_metrics_table(out, label_col="label", pred_col="pred")
    row = metrics.iloc[0].to_dict()
    logger.info(f"Metrics: {row}")

    # 5) Plots
    fig1 = plot_confusion(tp=row["tp"], fp=row["fp"], tn=row["tn"], fn=row["fn"], out_dir="outputs/figures", name="cs01_confusion.png")
    fig2 = plot_metric_bars(row, out_dir="outputs/figures", name="cs01_metrics.png")

    # 6) (Optional) Dry-run patch into engine
    patch_res = dryrun_patch_to_engine(rules)

    # 7) Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    out.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({"baseline_metrics": row, "rule_hits": hits, "patch": patch_res}, f, indent=2)

    return {
        "title": "Case Study 01 — Baseline rules on synthetic telemetry",
        "desc": "Evaluates a static baseline ruleset against synthetic MobiFlow summaries. "
                "Outputs confusion matrix and metric bars.",
        "metrics": {"baseline": row},
        "artifacts": [fig1, fig2, os.path.join(output_dir, "predictions.csv"), os.path.join(output_dir, "metrics.json")],
        "notes": [
            "Synthetic data uses a fixed RNG seed for reproducibility.",
            "Dry-run patch attempts to validate rules with P-BEST if available; otherwise skips."
        ]
    }

if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
