"""Create immutable two-health-check plus smoke evidence for one infra retry."""

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
    write_recovery_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one restored loopback service and write immutable recovery "
            "evidence; this does not itself authorize the retry"
        )
    )
    parser.add_argument("campaign", type=Path)
    parser.add_argument("schedule_id")
    parser.add_argument("--kind", required=True, choices=[item.value for item in ServiceKind])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--service-id", required=True)
    parser.add_argument("--model-id")
    parser.add_argument(
        "--tunnel-epoch",
        help="redacted public epoch label for a newly restored local forward",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ServiceConfig(
        kind=ServiceKind(args.kind),
        base_url=args.base_url,
        label=args.service_id,
        model_id=args.model_id,
    )
    path = write_recovery_bundle(
        args.campaign,
        args.schedule_id,
        config,
        runner=PreflightRunner(
            policy=PreflightPolicy(timeout_seconds=args.timeout_seconds)
        ),
        tunnel_epoch=args.tunnel_epoch,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "recovery_report": path.relative_to(args.campaign.resolve()).as_posix(),
                "next_command": (
                    ".venv/bin/python -m experiments.run_campaign "
                    f"{args.campaign} authorize-retry {args.schedule_id} "
                    f"{path.relative_to(args.campaign.resolve()).as_posix()}"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
