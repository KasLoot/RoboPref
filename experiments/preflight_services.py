"""Run the mandatory two-health-check plus smoke gate for all three services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.harness.preflight import (
    PreflightPolicy,
    PreflightRunner,
    ServiceConfig,
    ServiceKind,
    write_preflight_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gemma-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--sam-url", default="http://127.0.0.1:9000")
    parser.add_argument(
        "--gemma-model", default="/workspace/models/gemma-4-26B-A4B-it"
    )
    parser.add_argument(
        "--embedding-model", default="/workspace/models/embeddinggemma-300m"
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configs = (
        ServiceConfig(
            ServiceKind.GEMMA,
            args.gemma_url,
            "gemma-upper-and-execution",
            args.gemma_model,
        ),
        ServiceConfig(
            ServiceKind.EMBEDDING,
            args.embedding_url,
            "embedding-memory",
            args.embedding_model,
        ),
        ServiceConfig(ServiceKind.SAM, args.sam_url, "sam3.1-grounding"),
    )
    report = PreflightRunner(
        policy=PreflightPolicy(timeout_seconds=args.timeout_seconds)
    ).run(configs)
    write_preflight_report(args.output, report)
    print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
