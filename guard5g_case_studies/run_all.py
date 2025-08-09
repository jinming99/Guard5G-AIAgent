# run_all.py
"""
Run all case studies and mini-cases, then assemble a Markdown report.
Usage:
  python run_all.py [--seed 123]
"""

import argparse, json, os
from common.util import get_logger, set_seed, ensure_dir
from common.report import render_markdown

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123, help="Global RNG seed")
    args = parser.parse_args()
    set_seed(args.seed)
    ensure_dir("outputs")

    logger = get_logger("run_all")
    sections = []

    # Case studies
    import cs01_baseline_rules_eval as cs01
    import cs02_llm_rulegen_eval as cs02
    import cs03_llm_experiment_design as cs03
    sections.append(cs01.run())
    sections.append(cs02.run())
    sections.append(cs03.run())

    # Minis
    import mini01_sdl_ping as m1
    import mini02_kpm_playback_kpis as m2
    import mini03_rule_api_smoke as m3
    sections.append(m1.run())
    sections.append(m2.run())
    sections.append(m3.run())

    report_path = render_markdown("outputs/case_study_report.md", sections)
    logger.info(f"Wrote report: {report_path}")
    print(json.dumps({"report": report_path, "sections": [s["title"] for s in sections]}, indent=2))

if __name__ == "__main__":
    main()
