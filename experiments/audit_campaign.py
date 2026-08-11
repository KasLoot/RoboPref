"""CLI for the independent completeness, checksum, and evidence-link audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from experiments.harness.audit import (
        audit_campaign,
        write_audit_report,
        write_campaign_checksums,
    )
except ModuleNotFoundError:
    from harness.audit import (  # type: ignore[no-redef]
        audit_campaign,
        write_audit_report,
        write_campaign_checksums,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit one RoboPref campaign")
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--check-live-source",
        action="store_true",
        dest="check_live_source",
        help="verify current source membership/content (the default)",
    )
    source.add_argument(
        "--skip-live-source-check",
        action="store_false",
        dest="check_live_source",
        help="diagnostic only; a locked final audit cannot pass with this option",
    )
    parser.set_defaults(check_live_source=True)
    parser.add_argument("--write-checksums", action="store_true")
    parser.add_argument("--stdout-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    audit_arguments = {
        "verify_video": not arguments.no_video,
        "require_complete": not arguments.allow_incomplete,
        "check_live_source": arguments.check_live_source,
    }
    if arguments.write_checksums and arguments.allow_incomplete:
        raise SystemExit("refusing to finalize checksums for an incomplete campaign")
    report = audit_campaign(arguments.campaign, **audit_arguments)
    missing_checksum = (
        "campaign checksum manifest has not been generated; run final audit with "
        "--write-checksums"
    )
    if arguments.write_checksums:
        non_checksum_errors = [
            error for error in report["errors"] if error != missing_checksum
        ]
        if not non_checksum_errors:
            write_campaign_checksums(arguments.campaign)
            report = audit_campaign(arguments.campaign, **audit_arguments)
    if (
        not arguments.stdout_only
        and report["passed"]
        and not arguments.allow_incomplete
    ):
        write_audit_report(arguments.campaign, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
