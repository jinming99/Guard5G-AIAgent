#!/usr/bin/env python3
"""
Validate that detection aggregation and plotting do not crash headless.
"""

import logging
from evaluate_pipeline import EvaluationPipeline

logging.basicConfig(level=logging.DEBUG)

def main():
    pipe = EvaluationPipeline()
    pipe.results = {
        "scenarios": [
            {"scenario":"s1","metrics":{"accuracy":0.9,"false_positive_rate":0.1}},
            {"scenario":"s2","metrics":{"accuracy":0.8,"false_positive_rate":0.2}},
            {"scenario":"s3","detections":[{"confidence":0.5},{"confidence":0.8}]}
        ]
    }
    # Should save or at least build plots without DISPLAY
    pipe.plot_results()
    print("Plotting OK")

if __name__ == "__main__":
    main()
