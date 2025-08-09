# common/viz.py
import os
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def plot_confusion(tp, fp, tn, fn, out_dir="outputs/figures", name="confusion_matrix.png"):
    _ensure_dir(out_dir)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    # Simple 2x2 text grid
    table_data = [
        [f"TP: {tp}", f"FP: {fp}"],
        [f"FN: {fn}", f"TN: {tn}"],
    ]
    ax.axis("off")
    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.scale(1, 2)
    out = os.path.join(out_dir, name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_metric_bars(metrics_dict, out_dir="outputs/figures", name="metrics_bar.png"):
    _ensure_dir(out_dir)
    keys = ["precision","recall","f1","accuracy"]
    vals = [metrics_dict.get(k, 0.0) for k in keys]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(keys, vals)
    ax.set_ylim(0,1)
    ax.set_title("Metrics")
    out = os.path.join(out_dir, name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
