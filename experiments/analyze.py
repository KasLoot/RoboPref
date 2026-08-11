"""CLI for regenerating predeclared summaries from immutable raw attempts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from experiments.harness.analysis import analyze_campaign, write_analysis
except ModuleNotFoundError:
    from harness.analysis import analyze_campaign, write_analysis  # type: ignore[no-redef]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze one RoboPref campaign")
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--outcome", default="contract_success")
    parser.add_argument("--reference-profile", default="T5")
    parser.add_argument("--stem", default="analysis_results")
    parser.add_argument("--stdout-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = analyze_campaign(
        arguments.campaign,
        outcome=arguments.outcome,
        reference_profile=arguments.reference_profile,
    )
    if not arguments.stdout_only:
        write_analysis(arguments.campaign, report, stem=arguments.stem)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    # Missing data are a scientific result rather than a CLI crash; audit and
    # report fields carry the not-estimable disposition.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
