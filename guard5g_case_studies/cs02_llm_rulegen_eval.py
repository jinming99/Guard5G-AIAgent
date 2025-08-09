# cs02_llm_rulegen_eval.py
"""
Case Study 02
Generate rules via a heuristic "LLM" from telemetry summary; compare vs baseline.
"""

import os, json
import pandas as pd
from common.util import get_logger, set_seed, detect_runtime, ensure_dir
from common.data import SynthConfig, generate_mobiflow_synthetic, summarize
from common.rule_engine import load_rules_yaml, apply_rules, dryrun_patch_to_engine
from common.metrics import summarize_metrics_table
from common.viz import plot_confusion, plot_metric_bars
from common.llm_shims import basic_rule_from_summary

def _mk_summary_stats(snap: pd.DataFrame):
    return {
        "mean_auth_reject_count": float(snap["auth_reject_count"].mean()),
        "mean_rsrp_delta": float(snap["rsrp_delta_mean"].mean()),
        "fbs_rate": float(snap["suspected_fbs"].mean()),
    }

def run(output_dir="outputs/cs02", seed=123):
    logger = get_logger("cs02")
    ensure_dir(output_dir)
    set_seed(seed)
    rt = detect_runtime()
    logger.info(f"Runtime flags: {rt}")

    # Data
    cfg = SynthConfig(n_ue=100, duration_sec=240, fbs_fraction=0.25)
    df = generate_mobiflow_synthetic(cfg)
    snap = summarize(df)
    snap["label"] = snap["suspected_fbs"].astype(int)

    # Baseline
    baseline_rules = load_rules_yaml("resources/rule_templates/baseline_rules.yaml")
    out_base, hits_base = apply_rules(snap, baseline_rules)
    base_metrics = summarize_metrics_table(out_base, label_col="label", pred_col="pred").iloc[0].to_dict()

    # "LLM" rule generation
    stats = _mk_summary_stats(snap)
    llm_rule = basic_rule_from_summary(stats)
    out_llm, hits_llm = apply_rules(snap, [llm_rule])
    llm_metrics = summarize_metrics_table(out_llm, label_col="label", pred_col="pred").iloc[0].to_dict()

    # Plots
    figb1 = plot_confusion(base_metrics["tp"], base_metrics["fp"], base_metrics["tn"], base_metrics["fn"], name="cs02_confusion_baseline.png")
    figb2 = plot_metric_bars(base_metrics, name="cs02_metrics_baseline.png")
    figl1 = plot_confusion(llm_metrics["tp"], llm_metrics["fp"], llm_metrics["tn"], llm_metrics["fn"], name="cs02_confusion_llm.png")
    figl2 = plot_metric_bars(llm_metrics, name="cs02_metrics_llm.png")

    # Dry-run patch attempts
    patch_base = dryrun_patch_to_engine(baseline_rules)
    patch_llm = dryrun_patch_to_engine([llm_rule])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "compare_metrics.json"), "w") as f:
        json.dump({"baseline": base_metrics, "llm": llm_metrics, "stats": stats, "patch": {"baseline": patch_base, "llm": patch_llm}}, f, indent=2)

    return {
        "title": "Case Study 02 — Heuristic LLM-generated rule vs. baseline",
        "desc": "Derives a rule from telemetry summary via a deterministic heuristic and compares it to the baseline ruleset.",
        "metrics": {"baseline": base_metrics, "llm": llm_metrics},
        "artifacts": [figb1, figb2, figl1, figl2, os.path.join(output_dir, "compare_metrics.json")],
        "notes": [
            "LLM heuristic adapts thresholds using mean statistics; this is a placeholder for DSPy/OpenAI-backed flows.",
            "Dry-run patch attempts to validate rules with P-BEST if available; otherwise skips."
        ]
    }

if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
