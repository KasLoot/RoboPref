"""CLI for creating a versioned pilot or locked campaign freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Sequence

try:
    from experiments.harness.campaign import (
        CampaignMode,
        ScheduleCell,
        build_blocked_schedule,
        default_protocol_schedule,
        freeze_protocol,
        read_schedule,
    )
except ModuleNotFoundError:  # Support ``python experiments/freeze_protocol.py``.
    from harness.campaign import (  # type: ignore[no-redef]
        CampaignMode,
        ScheduleCell,
        build_blocked_schedule,
        default_protocol_schedule,
        freeze_protocol,
        read_schedule,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze an immutable RoboPref experiment campaign"
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--mode", choices=[mode.value for mode in CampaignMode], required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--protocol", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--schedule", type=Path, help="already-generated schedule CSV")
    selection.add_argument("--cells", type=Path, help="JSON array of ScheduleCell objects")
    parser.add_argument(
        "--repetitions",
        type=int,
        help="matched repetitions for the full design when no schedule/cells are supplied",
    )
    parser.add_argument("--sensitivity-repetitions", type=int, default=10)
    parser.add_argument(
        "--sensitivity-backbone",
        action="append",
        dest="sensitivity_backbones",
        help="repeat three times to replace primary/alternative/small labels",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="required only when this command generates the schedule",
    )
    parser.add_argument("--freeze", action="append", type=Path, dest="frozen_paths")
    parser.add_argument("--source-root", action="append", type=Path, dest="source_roots")
    parser.add_argument("--metadata", type=Path, help="additional canonical JSON metadata")
    return parser


def _default_frozen_paths(repo: Path) -> list[Path]:
    candidates = [
        repo / "experiments" / "scenarios.json",
        repo / "experiments" / "config" / "profiles.json",
        repo / "experiments" / "PREREGISTRATION.md",
        repo / "experiments" / "DATA_DICTIONARY.md",
        repo / "experiments" / "harness" / "oracles.py",
        repo / "experiments" / "harness" / "campaign.py",
        repo / "experiments" / "harness" / "analysis.py",
        repo / "pyproject.toml",
        repo / "uv.lock",
    ]
    return [path for path in candidates if path.exists()]


def _default_source_roots(repo: Path) -> list[Path]:
    candidates = [
        repo / "src",
        repo / "simulation",
        repo / "experiments",
        repo / "tests",
        repo / "PROJECT_CONTEXT.md",
        repo / "README.md",
        repo / "EXPERIMENT_SUITE_DESIGN.md",
        repo / "pyproject.toml",
        repo / "uv.lock",
    ]
    return [path for path in candidates if path.exists()]


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repo = arguments.repo_root.resolve()
    runs_root = (arguments.runs_root or repo / "experiments" / "runs").resolve()
    campaign_root = runs_root / arguments.campaign_id
    protocol = arguments.protocol or repo / "experiments" / "PREREGISTRATION.md"
    split = "pilot" if arguments.mode == CampaignMode.PILOT.value else "locked"
    if arguments.mode == CampaignMode.LOCKED.value:
        protocol_text = protocol.read_text(encoding="utf-8")
        if re.search(r"(?im)^status:.*\bdraft\b", protocol_text) or re.search(
            r"(?i)\bfreeze blocker\b", protocol_text
        ):
            raise SystemExit(
                "locked freeze refused: finalize the draft status and every declared "
                "freeze blocker after pilot evidence"
            )

    if arguments.schedule:
        if arguments.seed is not None:
            raise SystemExit(
                "--seed is not accepted with --schedule because that file's embedded "
                "seeds cannot be derived or verified from it"
            )
        schedule = read_schedule(arguments.schedule)
    elif arguments.cells:
        if arguments.seed is None:
            raise SystemExit("--seed is required when generating a schedule from --cells")
        raw = json.loads(arguments.cells.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise SystemExit("--cells must contain a JSON array")
        cells = [ScheduleCell.from_mapping(item) for item in raw]
        schedule = build_blocked_schedule(
            cells, master_seed=arguments.seed, split=split
        )
    else:
        if arguments.seed is None:
            raise SystemExit("--seed is required when generating the full schedule")
        if arguments.repetitions is None:
            raise SystemExit(
                "provide --schedule, --cells, or the pilot-powered --repetitions value"
            )
        backbones = arguments.sensitivity_backbones or [
            "primary",
            "alternative",
            "small",
        ]
        if len(backbones) != 3:
            raise SystemExit("the frozen sensitivity matrix requires exactly three backbones")
        schedule = default_protocol_schedule(
            split=split,
            matched_repetitions=arguments.repetitions,
            master_seed=arguments.seed,
            sensitivity_repetitions=arguments.sensitivity_repetitions,
            sensitivity_backbones=backbones,
        )
    metadata = {}
    if arguments.metadata:
        metadata = json.loads(arguments.metadata.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise SystemExit("--metadata must contain a JSON object")
    if "master_seed" in metadata:
        raise SystemExit("metadata.master_seed is reserved; supply --seed when generating")
    metadata["schedule_origin"] = (
        "supplied_frozen_csv" if arguments.schedule else "generated_by_freeze_protocol"
    )
    if arguments.seed is not None:
        metadata["master_seed"] = arguments.seed
    mandatory_source_roots = _default_source_roots(repo)
    source_roots = mandatory_source_roots + list(arguments.source_roots or ())
    manifest = freeze_protocol(
        campaign_root,
        campaign_id=arguments.campaign_id,
        mode=arguments.mode,
        protocol_source=protocol,
        schedule=schedule,
        repo_root=repo,
        frozen_paths=arguments.frozen_paths or _default_frozen_paths(repo),
        source_roots=source_roots,
        metadata=metadata,
    )
    summary = {
        "campaign_root": str(campaign_root),
        "campaign_id": manifest["campaign_id"],
        "mode": manifest["mode"],
        "protocol_sha256": manifest["protocol_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "schedule_size": manifest["schedule_size"],
        "locked_execution_started": False,
        "approval_required": manifest["locked_approval_required"],
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    if manifest["locked_approval_required"]:
        print(
            "Approval must be a separate exact message:\n"
            f"APPROVE LOCKED CAMPAIGN {manifest['campaign_id']} "
            f"{manifest['protocol_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
