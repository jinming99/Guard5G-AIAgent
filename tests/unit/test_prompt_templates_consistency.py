#!/usr/bin/env python3
"""
Check that rule_generation.yaml validation rules match the runtime validator.
"""

import os
import yaml
import logging
import llm_rule_patch as rp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_prompt_templates")

def main():
    ypath = os.environ.get("PROMPT_YAML", "rule_generation.yaml")
    with open(ypath) as f:
        Y = yaml.safe_load(f)

    allowed = set(Y["validation_rules"]["condition_operators"])
    # reconstruct from rp by calling the private function indirectly:
    # we simulate validation on a rule using each operator.
    missing = []
    for op in allowed:
        fake = {
          "rules":[{"name":"X","priority":1,
            "condition": {op: []} if op in ("and","or","not") else {"field":"x", op: 1},
            "action":{"type":"log"}}]
        }
        out = rp.validate_rules(fake)
        if not out.get("valid", False):
            missing.append(op)

    assert not missing, f"Operators allowed in YAML but rejected by validator: {missing}"
    logger.info("Prompt templates and validator are aligned ✅")

if __name__ == "__main__":
    main()
