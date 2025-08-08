#!/usr/bin/env python3
"""
Ensures llm_rule_patch accepts operators promised in the prompt template.
"""

import logging
import llm_rule_patch as rp

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_rule_patch")

YAML_OK = """
rules:
  - name: "FBS_Rapid_Reselection_In"
    priority: 7
    condition:
      and:
        - field: "cell_reselection_count"
          gte: 5
        - field: "suspicious_plmn"
          in: ["00101","99999"]
    action:
      type: "alert"
      severity: "high"
      message: "FBS suspected: reselections in suspicious PLMN"
"""

def main():
    res = rp.patch_rules(YAML_OK)
    assert res.get("status") != "error", f"Unexpected error: {res}"
    logger.info("llm_rule_patch accepted 'in' operator ✅")

if __name__ == "__main__":
    main()
