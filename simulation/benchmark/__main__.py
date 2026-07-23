from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import FAMILY_DEFINITIONS, build_catalog, build_control_catalog
from .generator import _json_bytes, _write_if_changed, generate_benchmark
from .protocols import build_memory_protocols, memory_protocol_fixtures
from .validator import validate_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and validate the RoboPref counterfactual benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate endpoint episode packets.")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FAMILY_DEFINITIONS),
        default=list(FAMILY_DEFINITIONS),
    )
    generate.add_argument("--seeds", nargs="+", type=int, default=[1])
    generate.add_argument(
        "--backend", choices=("synthetic", "mujoco"), default="synthetic"
    )
    generate.add_argument("--overwrite", action="store_true")
    generate.add_argument(
        "--include-controls",
        action="store_true",
        help="Add already-satisfied observation-only control packets.",
    )
    generate.add_argument(
        "--write-memory-protocols",
        action="store_true",
        help="Also write semantic multi-conversation protocol fixtures.",
    )

    validate = subparsers.add_parser("validate", help="Validate a generated benchmark.")
    validate.add_argument("root", type=Path)

    list_command = subparsers.add_parser("list", help="Print the declared scenario matrix.")
    list_command.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FAMILY_DEFINITIONS),
        default=list(FAMILY_DEFINITIONS),
    )
    list_command.add_argument("--seeds", nargs="+", type=int, default=[1])
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "generate":
        report = generate_benchmark(
            args.output,
            families=args.families,
            seeds=args.seeds,
            backend=args.backend,
            overwrite=args.overwrite,
            include_controls=args.include_controls,
        )
        if args.write_memory_protocols:
            scenarios = [
                *build_catalog(families=args.families, seeds=args.seeds),
                *(
                    build_control_catalog(
                        families=args.families,
                        seeds=args.seeds,
                    )
                    if args.include_controls
                    else ()
                ),
            ]
            protocols = build_memory_protocols(scenarios)
            path = report.output_root / "protocols.json"
            _write_if_changed(
                path,
                _json_bytes(
                    {
                        "schema_version": "robopref.memory-protocols.v1",
                        "fixtures": memory_protocol_fixtures(),
                        "protocols": protocols,
                    }
                ),
            )
        print(
            f"Generated {report.generated}, skipped {report.skipped}; "
            f"{report.scenario_count} scenarios at {report.output_root}"
        )
        return 0

    if args.command == "validate":
        report = validate_benchmark(args.root)
        for warning in report.warnings:
            print(f"WARNING: {warning}")
        for error in report.errors:
            print(f"ERROR: {error}")
        print(
            f"{'VALID' if report.valid else 'INVALID'}: "
            f"{report.scenario_count} scenarios"
        )
        return 0 if report.valid else 1

    catalog = build_catalog(families=args.families, seeds=args.seeds)
    print(
        json.dumps(
            [
                {
                    "scenario_id": item.scenario_id,
                    "family": item.family,
                    "scene_variant": item.scene_variant,
                    "target_id": item.target_id,
                    "outcome": item.outcome,
                    "seed": item.seed,
                }
                for item in catalog
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
