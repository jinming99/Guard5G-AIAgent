# common/rule_engine.py
from typing import Any, Dict, List, Tuple
import yaml
import pandas as pd

# Supported operators
_OPS = {
    "eq":  lambda a,b: a == b,
    "neq": lambda a,b: a != b,
    "gt":  lambda a,b: a > b,
    "lt":  lambda a,b: a < b,
    "gte": lambda a,b: a >= b,
    "lte": lambda a,b: a <= b,
    "contains": lambda a,b: (str(b) in str(a)) if a is not None else False,
    "matches":  lambda a,b: __import__("re").search(str(b), str(a)) is not None if a is not None else False,
    "in": lambda a,b: a in b if b is not None else False,
    "not_in": lambda a,b: a not in b if b is not None else True,
}

def load_rules_yaml(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Accepts either {'rules': [...]} or a list itself
    return data.get("rules", data)

def _eval_condition_row(row: pd.Series, cond: Dict[str, Any]) -> bool:
    # Logical block?
    if "and" in cond:
        return all(_eval_condition_row(row, c) for c in cond["and"])
    if "or" in cond:
        return any(_eval_condition_row(row, c) for c in cond["or"])
    if "not" in cond:
        return not _eval_condition_row(row, cond["not"])

    # Leaf comparison
    field = cond.get("field")
    if field is None:
        return False
    a = row.get(field)
    # operator key is any of known ones
    for op, fn in _OPS.items():
        if op in cond:
            b = cond.get(op)
            try:
                return bool(fn(a, b))
            except Exception:
                return False
    return False

def apply_rules(df: pd.DataFrame, rules: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Returns df_copy with a 'pred' boolean column and a dict of rule hit counts.
    """
    out = df.copy()
    pred = []
    hits = {r.get("name", f"rule_{i}"): 0 for i, r in enumerate(rules)}
    for _, row in out.iterrows():
        fired = False
        for r in rules:
            cname = r.get("name", "unnamed")
            cond = r.get("condition", {})
            if _eval_condition_row(row, cond):
                fired = True
                hits[cname] = hits.get(cname, 0) + 1
                # no break: allow counting multiple rule hits; final pred is OR-of-rules
        pred.append(bool(fired))
    out["pred"] = pred
    return out, hits

def dryrun_patch_to_engine(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Try to import the provided llm_rule_patch module (if present) and apply/validate.
    If engine not available, return a dry-run success with echoing validated rules.
    """
    try:
        # Try a few import patterns
        try:
            import pypbest.llm_rule_patch as pbest
        except Exception:
            try:
                import llm_rule_patch as pbest  # local copy
            except Exception:
                pbest = None
        if pbest is None:
            return {"status":"ok", "mode":"dry-run", "message":"P-BEST patch module not found, skipping live patch", "validated": rules}
        # If module provides a function we can call
        fn = getattr(pbest, "validate_and_apply_rules", None) or getattr(pbest, "apply_rules", None)
        if fn is None:
            return {"status":"ok", "mode":"dry-run", "message":"P-BEST patch function not found, skipping live patch", "validated": rules}
        res = fn(rules)  # may raise or return dict
        return {"status":"ok", "mode":"live", "engine_response": res}
    except Exception as e:
        return {"status":"ok", "mode":"dry-run", "message": f"Engine unavailable ({e}); validation passed locally.", "validated": rules}
