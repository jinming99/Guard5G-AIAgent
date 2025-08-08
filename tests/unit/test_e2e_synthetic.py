#!/usr/bin/env python3
"""
End-to-end synthetic scenario using evaluate_pipeline with only synthetic data.
"""

import logging
from evaluate_pipeline import EvaluationPipeline

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("e2e_synth")

def main():
    pipe = EvaluationPipeline()
    # only synthetic; no OAI needed
    res = pipe.run_scenario("synthetic", {"duration": 5})
    assert "metrics" in res, f"no metrics in result: {res.keys()}"
    logger.info("Synthetic E2E metrics: %s", res["metrics"])

if __name__ == "__main__":
    main()
