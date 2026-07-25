from __future__ import annotations

import argparse
import hashlib
from collections import Counter
import json
from pathlib import Path
from typing import Any

from agents.configs import PrefMemConfig

from .ablations import MEMORY_MODES
from .catalog import FAMILY_DEFINITIONS, build_catalog, build_control_catalog
from .generator import _json_bytes, _write_if_changed, generate_benchmark
from .protocols import build_memory_protocols, memory_protocol_fixtures
from .validator import validate_benchmark


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        help="Override the Ollama model for all four VLM agents.",
    )
    parser.add_argument("--ollama-host", help="Ollama service URL override.")
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override sampling temperature for all four VLM agents.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        help=(
            "Base Ollama seed. Repetition N uses base + N - 1 so repeated "
            "runs are reproducible but not identical."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="Per-model-call timeout in seconds.",
    )
    parser.add_argument(
        "--resize-images",
        action="store_true",
        help="Resize model images using PrefMem's configured vision bounds.",
    )
    parser.add_argument("--max-replans", type=int, default=1)
    parser.add_argument("--max-reobservations", type=int, default=1)
    parser.add_argument(
        "--memory-mode",
        choices=MEMORY_MODES,
        default="full",
        help="Evaluation-only memory ablation.",
    )
    parser.add_argument(
        "--display_all",
        "--display-all",
        dest="display_all",
        action="store_true",
        help="Also print structured background-agent diagnostics.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, validate, and evaluate the RoboPref counterfactual benchmark."
        )
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

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Run resumable isolated single-task PrefMem evaluation.",
    )
    evaluate.add_argument("root", type=Path, help="Generated benchmark root.")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--repetitions", type=int, default=1)
    evaluate.add_argument(
        "--condition",
        default="full",
        help="Stable experiment-condition label written to every result.",
    )
    evaluate.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FAMILY_DEFINITIONS),
    )
    evaluate.add_argument("--scene-variants", nargs="+")
    evaluate.add_argument("--target-ids", nargs="+")
    evaluate.add_argument("--outcomes", nargs="+")
    evaluate.add_argument("--seeds", nargs="+", type=int)
    evaluate.add_argument("--scenario-ids", nargs="+")
    evaluate.add_argument(
        "--exclude-controls",
        action="store_true",
        help="Exclude already-satisfied control packets.",
    )
    evaluate.add_argument("--max-scenarios", type=int)
    evaluate.add_argument("--shuffle-seed", type=int, default=0)
    evaluate.add_argument(
        "--no-resume",
        action="store_true",
        help="Refuse to reuse an output containing durable trial results.",
    )
    evaluate.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after persisting the first infrastructure/runtime error.",
    )
    evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the selected matrix without calling any model.",
    )
    _add_model_options(evaluate)

    memory = subparsers.add_parser(
        "evaluate-memory",
        help="Run isolated stateful history/preference protocol evaluation.",
    )
    memory.add_argument("root", type=Path, help="Generated benchmark root.")
    memory.add_argument("--output", type=Path, required=True)
    memory.add_argument("--repetitions", type=int, default=1)
    memory.add_argument("--protocol-ids", nargs="+")
    memory.add_argument(
        "--protocols-path",
        type=Path,
        help="Protocol bundle override (defaults to ROOT/protocols.json).",
    )
    memory.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate protocol selectors without calling any model.",
    )
    memory.add_argument(
        "--no-resume",
        action="store_true",
        help="Refuse to reuse an output containing durable protocol results.",
    )
    _add_model_options(memory)
    return parser


def _cold_config(args: argparse.Namespace) -> Any:
    from .evaluation import EvaluationConfig

    return EvaluationConfig(
        benchmark_root=args.root,
        output_dir=args.output,
        repetitions=args.repetitions,
        condition=args.condition,
        families=tuple(args.families or ()),
        scene_variants=tuple(args.scene_variants or ()),
        target_ids=tuple(args.target_ids or ()),
        outcomes=tuple(args.outcomes or ()),
        seeds=tuple(args.seeds or ()),
        scenario_ids=tuple(args.scenario_ids or ()),
        include_controls=not args.exclude_controls,
        max_scenarios=args.max_scenarios,
        shuffle_seed=args.shuffle_seed,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
        memory_mode=args.memory_mode,
        model=args.model,
        ollama_host=args.ollama_host,
        temperature=args.temperature,
        model_seed=args.model_seed,
        timeout_seconds=args.timeout_seconds,
        resize_images=args.resize_images,
        max_replans=args.max_replans,
        max_reobservations=args.max_reobservations,
    )


def _dry_run_cold(config: Any) -> int:
    # Selection is intentionally shared with the runner so a dry run and the
    # subsequent real run cannot silently disagree about the experiment matrix.
    from .evaluation import EvaluationDatasetError, _select_episodes

    validation = validate_benchmark(config.benchmark_root)
    if not validation.valid:
        raise EvaluationDatasetError(
            "Benchmark validation failed:\n" + "\n".join(validation.errors)
        )
    selected = _select_episodes(config)
    if not selected:
        raise EvaluationDatasetError("The requested filters selected zero packets.")
    counts = {
        "family": Counter(item.metadata.family for item in selected),
        "scene_variant": Counter(
            item.metadata.scene_variant for item in selected
        ),
        "target_id": Counter(item.metadata.target_id for item in selected),
        "outcome": Counter(item.metadata.outcome for item in selected),
        "seed": Counter(str(item.metadata.seed) for item in selected),
        "packet_kind": Counter(
            "control" if item.metadata.control_kind else "core"
            for item in selected
        ),
    }
    print(
        json.dumps(
            {
                "dry_run": True,
                "model_calls": 0,
                "selected_scenarios": len(selected),
                "repetitions": config.repetitions,
                "planned_trials": len(selected) * config.repetitions,
                "counts": {
                    name: dict(sorted(values.items()))
                    for name, values in counts.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _base_config(args: argparse.Namespace) -> PrefMemConfig:
    if args.max_replans < 0 or args.max_reobservations < 0:
        raise ValueError("Recovery budgets cannot be negative.")
    if args.temperature is not None and args.temperature < 0:
        raise ValueError("Temperature cannot be negative.")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        raise ValueError("Model timeout must be positive.")
    config = PrefMemConfig(
        max_replans=args.max_replans,
        max_reobservations=args.max_reobservations,
    )
    config.vision.resize_images = bool(args.resize_images)
    for model_config in (
        config.hri,
        config.memory,
        config.planner,
        config.validator,
    ):
        if args.model:
            model_config.model = args.model
        if args.ollama_host:
            model_config.host = args.ollama_host
        if args.temperature is not None:
            model_config.temperature = args.temperature
        if args.timeout_seconds is not None:
            model_config.timeout_seconds = args.timeout_seconds
    return config


def _selected_protocols(
    bundle: dict[str, Any],
    protocol_ids: list[str] | None,
) -> list[dict[str, Any]]:
    protocols = [
        item for item in bundle.get("protocols", []) if isinstance(item, dict)
    ]
    if protocol_ids is None:
        return protocols
    requested = list(protocol_ids)
    available = {str(item.get("protocol_id")): item for item in protocols}
    missing = set(requested) - set(available)
    if missing:
        raise ValueError(f"Unknown protocol IDs: {sorted(missing)}")
    return [available[protocol_id] for protocol_id in requested]


def _dry_run_memory(args: argparse.Namespace) -> int:
    from .protocol_evaluation import (
        EpisodeCatalog,
        ProtocolEvaluationError,
        _preflight_selection_plans,
        load_protocol_bundle,
    )

    if args.repetitions <= 0:
        raise ProtocolEvaluationError("Repetitions must be positive.")
    if args.model_seed is not None and args.model_seed < 0:
        raise ProtocolEvaluationError("Model seed cannot be negative.")
    _base_config(args)
    validation = validate_benchmark(args.root)
    if not validation.valid:
        raise ProtocolEvaluationError(
            "Benchmark validation failed:\n" + "\n".join(validation.errors)
        )
    protocol_path = args.protocols_path or args.root / "protocols.json"
    bundle = load_protocol_bundle(protocol_path)
    protocols = _selected_protocols(bundle, args.protocol_ids)
    catalog = EpisodeCatalog.from_benchmark(args.root)
    selection_plans = _preflight_selection_plans(
        protocols,
        catalog=catalog,
        repetitions=args.repetitions,
    )
    selection_plan_payload = [
        {
            "protocol_id": protocol_id,
            "repetition": repetition,
            "selections": selections,
        }
        for (protocol_id, repetition), selections in sorted(
            selection_plans.items()
        )
    ]
    selection_plan_sha256 = hashlib.sha256(
        json.dumps(
            selection_plan_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    selector_counts: dict[str, list[int]] = {}
    total_steps = 0
    for protocol in protocols:
        protocol_id = str(protocol["protocol_id"])
        matches: list[int] = []
        steps = protocol.get("steps", [])
        if not isinstance(steps, list):
            raise ProtocolEvaluationError(
                f"Protocol {protocol_id!r} steps must be a list."
            )
        total_steps += len(steps)
        for step in steps:
            selector = (
                step.get("scenario_selector")
                if isinstance(step, dict)
                else None
            )
            if isinstance(selector, dict):
                matches.append(len(catalog.compatible(selector)))
        selector_counts[protocol_id] = matches
    print(
        json.dumps(
            {
                "dry_run": True,
                "model_calls": 0,
                "selected_protocols": len(protocols),
                "steps_per_repetition": total_steps,
                "repetitions": args.repetitions,
                "planned_protocol_runs": len(protocols) * args.repetitions,
                "preflighted_selection_plans": len(selection_plans),
                "selection_plan_sha256": selection_plan_sha256,
                "selector_candidate_counts": selector_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


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
        if report.valid:
            print(f"VALID: {report.scenario_count} scenarios")
        else:
            print(
                f"INVALID: {len(report.errors)} errors while validating "
                f"{report.scenario_count} scenarios"
            )
        return 0 if report.valid else 1

    if args.command == "list":
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

    if args.command == "evaluate":
        from .evaluation import run_cold_memory_evaluation
        from .evaluation_runtime import evaluation_orchestrator_factory

        config = _cold_config(args)
        if args.dry_run:
            return _dry_run_cold(config)
        # The runner applies the selected ablation exactly once. The runtime
        # factory supplies telemetry and structured diagnostic sinks.
        report = run_cold_memory_evaluation(
            config,
            orchestrator_factory=evaluation_orchestrator_factory(
                memory_mode="full",
                display_all=args.display_all,
            ),
        )
        print(
            json.dumps(
                {
                    "output_dir": str(report.output_dir),
                    "selected_scenarios": report.selected_scenarios,
                    "planned_trials": report.planned_trials,
                    "executed_trials": report.executed_trials,
                    "resumed_trials": report.skipped_trials,
                    "passed_trials": report.passed_trials,
                    "failed_trials": report.failed_trials,
                    "error_trials": report.error_trials,
                },
                indent=2,
            )
        )
        return 0

    if args.dry_run:
        return _dry_run_memory(args)

    from dataset.benchmark import BenchmarkEpisode

    from .evaluation_runtime import build_evaluation_orchestrator
    from .executor import BenchmarkEpisodeExecutor
    from .protocol_evaluation import evaluate_memory_protocols

    def protocol_factory(context: Any) -> Any:
        episode = BenchmarkEpisode.from_path(
            context.initial_episode_path,
            context.config.workspace_root,
        )
        return build_evaluation_orchestrator(
            context.config,
            BenchmarkEpisodeExecutor(episode),
            context.run_directory,
            memory_mode=context.memory_mode,
            display_all=args.display_all,
        )

    result = evaluate_memory_protocols(
        args.root,
        args.output,
        repetitions=args.repetitions,
        protocol_ids=args.protocol_ids,
        protocols_path=args.protocols_path,
        base_config=_base_config(args),
        orchestrator_factory=protocol_factory,
        memory_mode=args.memory_mode,
        model_seed=args.model_seed,
        resume=not args.no_resume,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
