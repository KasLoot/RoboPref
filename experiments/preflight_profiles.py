#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.harness.profile_preflight import write_profile_preflights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/calibration/profile_preflights.json"),
    )
    arguments = parser.parse_args()
    digest = write_profile_preflights(arguments.output)
    print(f"{arguments.output}: sha256={digest}")


if __name__ == "__main__":
    main()
