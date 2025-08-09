# common/report.py
import os, json, datetime
from typing import Dict, Any, List
from pathlib import Path

def _fmt_pct(x: float) -> str:
    try:
        return f"{100.0 * x:5.1f}%"
    except Exception:
        return "n/a"

def render_markdown(report_path: str, sections: List[Dict[str, Any]]):
    lines = []
    lines.append(f"# Guard5G Case Studies Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.datetime.utcnow().isoformat()}Z_")
    lines.append("")

    for s in sections:
        lines.append(f"## {s.get('title','Untitled')}")
        if s.get("desc"):
            lines.append(s["desc"])
            lines.append("")
        if s.get("metrics"):
            lines.append("### Metrics")
            lines.append("")
            for label, mdict in s["metrics"].items():
                lines.append(f"- **{label}**: "
                             f"precision={_fmt_pct(mdict.get('precision',0))}, "
                             f"recall={_fmt_pct(mdict.get('recall',0))}, "
                             f"f1={_fmt_pct(mdict.get('f1',0))}, "
                             f"accuracy={_fmt_pct(mdict.get('accuracy',0))} "
                             f"(tp={mdict.get('tp',0)}, fp={mdict.get('fp',0)}, tn={mdict.get('tn',0)}, fn={mdict.get('fn',0)})")
            lines.append("")
        if s.get("artifacts"):
            lines.append("### Artifacts")
            for art in s["artifacts"]:
                p = Path(art).as_posix()
                lines.append(f"- {p}")
            lines.append("")
        if s.get("notes"):
            lines.append("### Notes")
            for n in s["notes"]:
                lines.append(f"- {n}")
            lines.append("")
        lines.append("---")
        lines.append("")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path
