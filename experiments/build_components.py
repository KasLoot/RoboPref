#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.harness.components import write_component_catalogue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/component_catalogue.json"),
    )
    args = parser.parse_args()
    digest = write_component_catalogue(args.output)
    print(f"{args.output}: sha256={digest}")


if __name__ == "__main__":
    main()
