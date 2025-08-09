# common/llm_shims.py
from typing import Dict, Any, List
import json

def basic_rule_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very simple heuristic "LLM": returns a single rule tuned to summary stats.
    Expects keys like mean_auth_reject, mean_rsrp_delta, etc.
    """
    # Heuristic thresholds
    thr_auth = max(2, int(round(summary.get("mean_auth_reject_count", 2))))
    thr_rsrp = max(18, float(summary.get("mean_rsrp_delta", 18.0)))
    rule = {
        "name": "LLM_Heuristic_Rule",
        "priority": 8,
        "condition": {
            "or": [
                {"field": "auth_reject_count", "gte": thr_auth},
                {"and": [
                    {"field": "signal_anomaly", "eq": True},
                    {"field": "rsrp_delta", "gt": thr_rsrp - 2},
                ]},
                {"field": "cipher_downgrade", "eq": True}
            ]
        },
        "action": {"type": "alert", "severity":"high", "message":"Heuristic LLM rule fired"}
    }
    return rule

def design_experiment_scenario(hypothesis: str) -> Dict[str, Any]:
    """
    Heuristic designer that returns a scenario JSON dict compatible with nr-softmodem --scenario-file.
    """
    scenario = {
        "name": f"CaseStudy: {hypothesis[:48]}",
        "mode": "fbs",
        "duration": 120,
        "config": {
            "plmn": "00101",
            "pci": 999,
            "tac": 999,
            "tx_power": 26,
            "cipher": "NULL"
        },
        "events": [
            {"time": 30, "action": "increase_power", "value": 6},
            {"time": 60, "action": "change_identity", "pci": 1000},
            {"time": 90, "action": "increase_power", "value": 4},
        ]
    }
    return scenario
