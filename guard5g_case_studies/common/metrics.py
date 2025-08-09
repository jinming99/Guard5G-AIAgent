# common/metrics.py
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

def _safe_div(a, b):
    return (a / b) if b else 0.0

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc
    }

def bootstrap_ci(values, n_boot=1000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(123)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    boots = []
    for _ in range(n_boot):
        s = rng.choice(arr, size=arr.size, replace=True).mean()
        boots.append(s)
    lo = np.quantile(boots, alpha/2)
    hi = np.quantile(boots, 1 - alpha/2)
    return float(lo), float(hi)

def summarize_metrics_table(df: pd.DataFrame, label_col="label", pred_col="pred", group_col=None) -> pd.DataFrame:
    if group_col is None:
        m = binary_metrics(df[label_col].values, df[pred_col].values)
        return pd.DataFrame([m])
    out = []
    for key, sub in df.groupby(group_col):
        m = binary_metrics(sub[label_col].values, sub[pred_col].values)
        m[group_col] = key
        out.append(m)
    return pd.DataFrame(out)

