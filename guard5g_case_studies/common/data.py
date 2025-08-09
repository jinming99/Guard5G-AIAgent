# common/data.py
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import time
import math
import pandas as pd
import numpy as np

@dataclass
class SynthConfig:
    n_ue: int = 60
    duration_sec: int = 120
    fbs_fraction: float = 0.25   # fraction of UEs subject to FBS influence
    base_rate: float = 0.05      # baseline anomaly rate
    fbs_rate: float = 0.35       # anomaly rate under FBS
    start_ts: Optional[int] = None

def _gen_ids(prefix: str, n: int) -> List[str]:
    width = int(math.ceil(math.log10(max(n,1)+1)))
    return [f"{prefix}-{str(i+1).zfill(width)}" for i in range(n)]

def generate_mobiflow_synthetic(cfg: SynthConfig) -> pd.DataFrame:
    """
    Deterministic(ish) synthetic generator for UE telemetry events.
    Produces columns consistent with our pipeline.
    """
    n = cfg.n_ue
    T = cfg.duration_sec
    start_ts = cfg.start_ts or int(time.time())
    ue_ids = _gen_ids("UE", n)
    cell_ids = _gen_ids("CELL", 5)

    rng = np.random.default_rng(42)  # fixed seed for determinism at script level
    # Pick FBS-influenced UEs
    n_fbs = max(1, int(cfg.fbs_fraction * n))
    fbs_set = set(rng.choice(ue_ids, size=n_fbs, replace=False))

    rows = []
    for t in range(T):
        ts = start_ts + t
        for ue in ue_ids:
            under_fbs = ue in fbs_set and (t > T // 4)  # FBS manifests after first quarter of run
            p = cfg.fbs_rate if under_fbs else cfg.base_rate

            # Core indicators
            auth_reject = rng.binomial(1, p * 0.7)
            attach_fail = rng.binomial(1, p * 0.5)
            sig_anom = rng.binomial(1, p * 0.6)
            cipher_down = rng.binomial(1, p * 0.3)

            # Derived counters (aggregate over small windows via Poisson draws)
            auth_reject_count = rng.poisson(2 if auth_reject else 0.2)
            attach_failures = rng.poisson(1 if attach_fail else 0.1)
            cell_resel = rng.poisson(4 if (sig_anom and under_fbs) else 0.5)
            rsrp_delta = abs(rng.normal(18 if under_fbs else 6, 4))
            suspected_fbs = bool(under_fbs and ((auth_reject_count >= 2) or sig_anom or cipher_down or (rsrp_delta > 20)))

            rows.append({
                "ue_id": ue,
                "cell_id": rng.choice(cell_ids),
                "timestamp": ts,
                "auth_reject_count": int(auth_reject_count),
                "attach_failures": int(attach_failures),
                "signal_anomaly": bool(sig_anom),
                "cipher_downgrade": bool(cipher_down),
                "cell_reselection_count": int(cell_resel),
                "rsrp_delta": float(rsrp_delta),
                "suspected_fbs": bool(suspected_fbs),
            })

    df = pd.DataFrame(rows)
    return df

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-UE summary snapshot for an evaluation window."""
    agg = df.groupby("ue_id").agg(
        auth_reject_count=("auth_reject_count","sum"),
        attach_failures=("attach_failures","sum"),
        signal_anomaly=("signal_anomaly","sum"),
        cipher_downgrade=("cipher_downgrade","sum"),
        cell_reselection_count=("cell_reselection_count","sum"),
        rsrp_delta_mean=("rsrp_delta","mean"),
        suspected_fbs=("suspected_fbs","max"),
    ).reset_index()
    # Convert boolean sums to counts and keep a boolean 'any' flavor
    agg["any_signal_anomaly"] = agg["signal_anomaly"] > 0
    agg["any_cipher_downgrade"] = agg["cipher_downgrade"] > 0
    return agg
