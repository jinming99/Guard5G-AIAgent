#!/usr/bin/env python3
import os, sys, traceback

print("[STEP] Rule patch cycle sanity")
try:
    import llm_rule_patch as rp
except Exception as e:
    print("[SKIP] llm_rule_patch import failed:", e)
    sys.exit(0)

# Valid rule
valid = r"""
rules:
  - name: "FBS_Auth_Failure_Pattern"
    priority: 9
    condition:
      and:
        - field: "auth_reject_count"
          gte: 3
        - field: "time_window"
          lte: 60
    action:
      type: "alert"
      severity: "high"
      message: "Potential FBS: Multiple auth failures detected"
"""

try:
    out = rp.patch_rules(valid)
    print("[OK] Valid rule applied:", str(out)[:120])
except Exception:
    print("[ERR] Valid rule failed:\n", traceback.format_exc())
    sys.exit(1)

# Invalid rule (operator not allowed)
invalid = r"""
rules:
  - name: "Bad_Operator"
    priority: 1
    condition:
      field: "cipher_algo"
      equals: "NULL"
    action:
      type: "alert"
      severity: "low"
      message: "Invalid operator test"
"""

try:
    rp.patch_rules(invalid)  # should raise / reject
    print("[ERR] Invalid rule unexpectedly accepted")
    sys.exit(1)
except Exception as e:
    print("[OK] Invalid rule correctly rejected:", e)

print("[DONE] Rule patch cycle sanity OK")
