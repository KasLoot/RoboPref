from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.configs import normalize_model_base_url

from .ablations import MEMORY_MODES
from .catalog import FAMILY_DEFINITIONS, build_catalog
from .conversation_cases import (
    build_conversation_cases,
    case_matrix,
    load_episode_catalog,
)
from .conversation_models import (
    NEAR_MISS_POLICIES,
    SUITES,
    ConversationEvaluationConfig,
)
from .generator import generate_benchmark
from .model_defaults import DEFAULT_EVALUATION_PROVIDER
from .validator import validate_benchmark


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        help="Override the model for all four model-backed agents.",
    )
    parser.add_argument(
        "--model-provider",
        choices=("ollama", "vllm"),
        default=DEFAULT_EVALUATION_PROVIDER,
    )
    parser.add_argument(
        "--model-base-url",
        help="OpenAI-compatible vLLM API root ending in /v1.",
    )
    parser.add_argument("--ollama-host", help="Ollama service URL override.")
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--model-seed",
        type=int,
        help="Base seed; repetition N uses base + N - 1.",
    )
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--resize-images", action="store_true")
    parser.add_argument("--max-replans", type=int, default=1)
    parser.add_argument("--max-reobservations", type=int, default=1)
    parser.add_argument(
        "--memory-mode",
        choices=MEMORY_MODES,
        default="full",
        help="Evaluation-only memory ablation.",
    )
    parser.add_argument(
        "--display-all",
        action="store_true",
        help="Also print structured background-agent diagnostics.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, validate, and evaluate the RoboPref counterfactual benchmark."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate endpoint packets.")
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
    generate.add_argument("--include-controls", action="store_true")

    validate = commands.add_parser("validate", help="Validate a benchmark.")
    validate.add_argument("root", type=Path)

    list_command = commands.add_parser(
        "list", help="Print the declared core scenario matrix."
    )
    list_command.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FAMILY_DEFINITIONS),
        default=list(FAMILY_DEFINITIONS),
    )
    list_command.add_argument("--seeds", nargs="+", type=int, default=[1])

    evaluate = commands.add_parser(
        "evaluate-conversations",
        help="Run resumable isolated full-conversation PrefMem evaluation.",
    )
    evaluate.add_argument("root", type=Path, help="Generated benchmark root.")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--suites",
        nargs="+",
        choices=SUITES,
        default=list(SUITES),
    )
    evaluate.add_argument("--repetitions", type=int, default=1)
    evaluate.add_argument("--condition", default="full")
    evaluate.add_argument(
        "--families", nargs="+", choices=tuple(FAMILY_DEFINITIONS)
    )
    evaluate.add_argument("--scene-variants", nargs="+")
    evaluate.add_argument("--target-ids", nargs="+")
    evaluate.add_argument("--outcomes", nargs="+")
    evaluate.add_argument("--seeds", nargs="+", type=int)
    evaluate.add_argument("--scenario-ids", nargs="+")
    evaluate.add_argument("--exclude-controls", action="store_true")
    evaluate.add_argument("--max-cases", type=int)
    evaluate.add_argument("--shuffle-seed", type=int, default=0)
    evaluate.add_argument(
        "--near-miss-policy",
        choices=NEAR_MISS_POLICIES,
        default="perceptual",
    )
    evaluate.add_argument("--max-user-turns", type=int, default=6)
    evaluate.add_argument("--no-resume", action="store_true")
    evaluate.add_argument("--fail-fast", action="store_true")
    evaluate.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the terminal progress bar.",
    )
    evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the case matrix without model calls.",
    )
    evaluate.add_argument("--bootstrap-replicates", type=int, default=2_000)
    evaluate.add_argument("--bootstrap-seed", type=int, default=20_260_726)
    _add_model_options(evaluate)
    return parser


def _validate_model_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.model_base_url:
        if args.model_provider != "vllm":
            parser.error("--model-base-url requires --model-provider vllm.")
        try:
            normalize_model_base_url(args.model_base_url)
        except ValueError as error:
            parser.error(str(error))
    if args.ollama_host and args.model_provider == "vllm":
        parser.error("--ollama-host cannot be used with --model-provider vllm.")


def _conversation_config(args: argparse.Namespace) -> ConversationEvaluationConfig:
    return ConversationEvaluationConfig(
        benchmark_root=args.root,
        output_dir=args.output,
        suites=tuple(args.suites),
        repetitions=args.repetitions,
        condition=args.condition,
        families=tuple(args.families or ()),
        scene_variants=tuple(args.scene_variants or ()),
        target_ids=tuple(args.target_ids or ()),
        outcomes=tuple(args.outcomes or ()),
        seeds=tuple(args.seeds or ()),
        scenario_ids=tuple(args.scenario_ids or ()),
        include_controls=not args.exclude_controls,
        max_cases=args.max_cases,
        shuffle_seed=args.shuffle_seed,
        near_miss_policy=args.near_miss_policy,
        max_user_turns=args.max_user_turns,
        resume=not args.no_resume,
        fail_fast=args.fail_fast,
        show_progress=not args.no_progress,
        memory_mode=args.memory_mode,
        model=args.model,
        model_provider=args.model_provider,
        model_base_url=args.model_base_url,
        ollama_host=args.ollama_host,
        temperature=args.temperature,
        model_seed=args.model_seed,
        timeout_seconds=args.timeout_seconds,
        resize_images=args.resize_images,
        display_all=args.display_all,
        max_replans=args.max_replans,
        max_reobservations=args.max_reobservations,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )


def _dry_run(config: ConversationEvaluationConfig) -> int:
    validation = validate_benchmark(config.benchmark_root)
    if not validation.valid:
        raise ValueError(
            "Benchmark validation failed:\n" + "\n".join(validation.errors)
        )
    catalog = load_episode_catalog(config.benchmark_root)
    cases = build_conversation_cases(catalog, config)
    print(
        json.dumps(
            {
                "dry_run": True,
                "model_calls": 0,
                "repetitions": config.repetitions,
                "planned_runs": len(cases) * config.repetitions,
                "matrix": case_matrix(cases),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "generate":
        report = generate_benchmark(
            args.output,
            families=args.families,
            seeds=args.seeds,
            backend=args.backend,
            overwrite=args.overwrite,
            include_controls=args.include_controls,
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
            ("VALID" if report.valid else "INVALID")
            + f": {report.scenario_count} scenarios"
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

    _validate_model_options(parser, args)
    try:
        config = _conversation_config(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        return _dry_run(config)
    from .conversation_evaluation import evaluate_conversations

    report = evaluate_conversations(config)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
