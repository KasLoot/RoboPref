#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.harness.scene_preflight import write_mechanism_preflights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--master-seed", type=int, default=20260811)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/calibration/mechanism_scene_preflights.json"),
    )
    arguments = parser.parse_args()
    digest = write_mechanism_preflights(
        arguments.output,
        sam_base_url=arguments.sam_base_url,
        master_seed=arguments.master_seed,
    )
    print(f"{arguments.output}: sha256={digest}")


if __name__ == "__main__":
    main()
