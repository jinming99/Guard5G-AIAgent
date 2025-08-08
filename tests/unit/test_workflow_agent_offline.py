#!/usr/bin/env python3
"""
Runs WorkflowAwareAgent in offline mode to design a scenario JSON.
"""

import json
import logging
import os

# The imports are resilient after patches
from llm_driver import LLMOrchestrator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("workflow_offline")

def main():
    orch = LLMOrchestrator(config_file=None, use_enhanced=False)  # basic path
    hypothesis = "Null ciphering increases attach failures within 120s"
    scenario = orch.experiment.forward({"hypothesis": hypothesis})
    logger.info("Designed scenario JSON:\n%s", scenario)
    # ensure it is valid json
    data = json.loads(scenario)
    assert "duration" in data and "events" in data
    print("Workflow offline scenario design OK")

if __name__ == "__main__":
    main()
