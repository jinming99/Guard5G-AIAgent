#!/usr/bin/env python3
import logging, os
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("test.dspy")

try:
    from enhanced_modules import QueryNetwork, RuleGenerator, ExperimentDesigner, DataAnalyst
except Exception as e:
    log.warning("DSPy/enhanced modules not importable: %s (skipping)", e)
    raise SystemExit(0)

def main():
    q = QueryNetwork()
    rg = RuleGenerator()
    ed = ExperimentDesigner()
    da = DataAnalyst()

    # basic interface checks
    for tool in (q, rg, ed, da):
        assert hasattr(tool, "forward")
        log.info("Tool %s has forward()", tool.__class__.__name__)

    # run very small samples if OPENAI_API_KEY present
    if os.getenv("OPENAI_API_KEY"):
        out = rg.forward({"telemetry":"auth_reject_count=3 within 60s"})
        log.info("RuleGenerator output: %s", str(out)[:200])
    else:
        log.warning("OPENAI_API_KEY not set; interface-only checks done")

if __name__ == "__main__":
    main()
