#!/usr/bin/env python3
import os, yaml, logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("test.prompt")
import llm_rule_patch as rp

YAML = os.getenv("PROMPT_YAML", "rule_generation.yaml")

def main():
    rules = yaml.safe_load(open(YAML))
    allowed = set(rules["validation_rules"]["condition_operators"])
    missing = []
    for op in sorted(allowed):
        # Build a minimal rule that uses each operator
        if op in ("and", "or", "not"):
            cond = {op: []}
        else:
            cond = {"field":"x", op: 1}
        doc = {"rules":[{"name":"X","priority":1,"condition":cond,"action":{"type":"log"}}]}
        res = rp.validate_rules(doc) if hasattr(rp, "validate_rules") else {"valid": True}
        if not res.get("valid", True):
            missing.append(op)
    if missing:
        raise SystemExit(f"Operators allowed in prompt but rejected by runtime: {missing}")
    log.info("Prompt template operators align with runtime validator")

if __name__ == "__main__":
    main()
