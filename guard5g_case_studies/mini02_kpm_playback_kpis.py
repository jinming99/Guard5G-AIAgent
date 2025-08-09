# mini02_kpm_playback_kpis.py
"""
Mini Case 02: Generate and 'inject' synthetic KPM/MobiFlow-like records (offline).
"""
import os, json
from common.util import get_logger, set_seed
from common.data import SynthConfig, generate_mobiflow_synthetic, summarize

def run(seed=99):
    set_seed(seed)
    logger = get_logger("mini02")
    df = generate_mobiflow_synthetic(SynthConfig(n_ue=20, duration_sec=30, fbs_fraction=0.2))
    snap = summarize(df)
    metrics = {
        "mean_auth_reject": float(snap["auth_reject_count"].mean()),
        "mean_attach_fail": float(snap["attach_failures"].mean()),
        "resel_p95": float(snap["cell_reselection_count"].quantile(0.95)),
    }
    logger.info(f"Snapshot KPIs: {metrics}")
    return {"title":"Mini 02 — KPM/MobiFlow snapshot KPIs", "desc": "Offline playback and summary KPIs.", "metrics":{"kpis": metrics}, "artifacts": [], "notes": []}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
