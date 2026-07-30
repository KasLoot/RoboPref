"""Command-line assembly for the deterministic PrefMem runtime."""

from __future__ import annotations

import argparse
import json
import os
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

from prefmem.agents.config import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_EMBEDDING_BASE_URL,
)
from prefmem.agents.contracts import ReplanPolicy
from prefmem.agents.model_client import ModelEndpoint, StructuredAgentClient
from prefmem.agents.vision import DEFAULT_LIVE_FRAME_URL


DEFAULT_DATASET = Path("dataset/v3")
DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_MODEL_PROVIDER = "ollama"
DEFAULT_TRANSCRIPT = Path("transcripts/transcript.txt")
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_RECORDING_DIR = Path("recordings")
DEFAULT_AGENT_API_KEY = os.environ.get("PREFMEM_AGENT_API_KEY", "EMPTY")
DEFAULT_EMBEDDING_API_KEY = os.environ.get(
    "PREFMEM_EMBEDDING_API_KEY",
    "EMPTY",
)


def _valid_openai_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError(
            "model base URL must be an absolute HTTP(S) URL"
        )
    if parsed.username is not None or parsed.password is not None:
        raise argparse.ArgumentTypeError(
            "model base URL must not contain credentials"
        )
    if parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError(
            "model base URL must not contain a query or fragment"
        )
    if parsed.path.endswith("/"):
        raise argparse.ArgumentTypeError(
            "model base URL must not end with a slash"
        )
    return value


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prefmem",
        description=(
            "Run PrefMem with typed agent boundaries, Markdown memory, and "
            "camera-grounded planning."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        metavar="PATH",
        help="Directory containing ordered image frames. (default: %(default)s)",
    )
    parser.add_argument(
        "--frame-source",
        choices=["live", "dataset"],
        default="live",
        help="Read localhost camera snapshots or ordered dataset images.",
    )
    parser.add_argument(
        "--camera-url",
        default=DEFAULT_LIVE_FRAME_URL,
        type=_valid_openai_base_url,
        help="Live camera snapshot URL. (default: %(default)s)",
    )
    parser.add_argument(
        "--camera-timeout",
        type=_positive_float,
        default=5.0,
        help="Camera snapshot timeout in seconds. (default: %(default)s)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Use --dataset as one generated benchmark packet.",
    )

    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument(
        "--preference-store",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Canonical Markdown preference file. By default PrefMem uses "
            ".prefmem/memory/<username>/preferences.md."
        ),
    )
    memory_group.add_argument(
        "--memory-store",
        dest="preference_store",
        type=Path,
        metavar="PATH",
        help="Deprecated alias for --preference-store.",
    )
    parser.add_argument(
        "--username",
        default="default",
        help="Participant namespace for preference and episode ownership.",
    )
    parser.add_argument(
        "--embedding-base-url",
        type=_valid_openai_base_url,
        default=DEFAULT_EMBEDDING_BASE_URL,
        help="EmbeddingGemma OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Exact served embedding model ID. Omit when port 8080 serves one model."
        ),
    )
    parser.add_argument(
        "--embedding-api-key",
        default=DEFAULT_EMBEDDING_API_KEY,
        help=(
            "Embedding service API key; defaults to "
            "PREFMEM_EMBEDDING_API_KEY or EMPTY."
        ),
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        choices=[128, 256, 512, 768],
        default=768,
        help="EmbeddingGemma output dimensions. (default: %(default)s)",
    )
    parser.add_argument(
        "--memory-similarity-threshold",
        type=float,
        default=0.20,
        help="Embedding candidate threshold before semantic classification.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Agent model name or served model ID. (default: %(default)s)",
    )
    parser.add_argument(
        "--model-provider",
        choices=["ollama", "vllm"],
        default=DEFAULT_MODEL_PROVIDER,
        help="Agent model provider. (default: %(default)s)",
    )
    parser.add_argument(
        "--model-base-url",
        type=_valid_openai_base_url,
        default=None,
        help="OpenAI-compatible vLLM base URL; valid only with --model-provider vllm.",
    )
    parser.add_argument(
        "--model-api-key",
        default=DEFAULT_AGENT_API_KEY,
        help=(
            "OpenAI-compatible model API key; defaults to "
            "PREFMEM_AGENT_API_KEY or EMPTY."
        ),
    )
    parser.add_argument(
        "--model-config",
        choices=["ollama", "vllm"],
        default=None,
        help="Deprecated alias for --model-provider.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Optional provider reasoning effort for all reasoning agents.",
    )
    parser.add_argument(
        "--max-agent-tokens",
        type=int,
        default=16384,
        help="Maximum completion tokens for a typed agent response.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Enable VLA publication and monitoring. Without this flag PrefMem is "
            "truthful plan-only software."
        ),
    )
    parser.add_argument(
        "--vla-command-file",
        type=Path,
        default=None,
        help="JSONL command queue consumed by an external VLA executor.",
    )
    parser.add_argument(
        "--replan-policy",
        choices=[item.value for item in ReplanPolicy],
        default=ReplanPolicy.ON_DEVIATION.value,
    )
    parser.add_argument(
        "--monitor-interval",
        type=_non_negative_float,
        default=1.0,
        help="Seconds between live-monitor observations. (default: %(default)s)",
    )
    parser.add_argument("--max-monitor-checks", type=int, default=30)
    parser.add_argument("--max-replans", type=int, default=2)
    parser.add_argument("--max-reobservations", type=int, default=3)
    parser.add_argument("--success-confirmations", type=int, default=2)
    parser.add_argument("--max-dispatches", type=int, default=32)

    parser.add_argument(
        "--record-context",
        action="store_true",
        help=(
            "Write a Markdown transcript containing all user/system/agent context "
            "and save every frame read by PrefMem."
        ),
    )
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=DEFAULT_RECORDING_DIR,
        help="Parent directory for replayable experiment sessions.",
    )
    parser.add_argument(
        "--recording-name",
        default=None,
        help="Optional stable name for this recording session.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=DEFAULT_TRANSCRIPT,
        metavar="PATH",
        help="Legacy terminal-transcript path retained for CLI compatibility.",
    )

    parser.add_argument(
        "--display-all",
        "--display_all",
        dest="display_all",
        action="store_true",
        help="Print structured controller results in addition to user-facing text.",
    )
    parser.add_argument(
        "--resize-images",
        action="store_true",
        help="Resize model-bound images to fit within 640x480.",
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--query",
        default=None,
        help="Run one request non-interactively.",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        default=None,
        help="JSON file containing a list of requests.",
    )
    parser.add_argument(
        "--think",
        default=["HRI"],
        help="Legacy reasoning-agent selector retained for compatibility.",
    )
    parser.add_argument("--print-raw", action="store_true")
    parser.add_argument("--print-usage", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.model_config is not None:
        if (
            args.model_provider != DEFAULT_MODEL_PROVIDER
            and args.model_provider != args.model_config
        ):
            parser.error("--model-config conflicts with --model-provider")
        args.model_provider = args.model_config

    if args.model_provider == "vllm":
        args.model_base_url = args.model_base_url or DEFAULT_VLLM_BASE_URL
        if args.model == DEFAULT_MODEL:
            args.model = DEFAULT_AGENT_MODEL
    elif args.model_base_url is not None:
        parser.error("--model-base-url requires --model-provider vllm")

    if args.execute and args.vla_command_file is None:
        parser.error("--execute requires --vla-command-file")
    if args.benchmark:
        args.frame_source = "dataset"
    if args.max_monitor_checks < 1:
        parser.error("--max-monitor-checks must be positive")
    if args.max_agent_tokens < 512:
        parser.error("--max-agent-tokens must be at least 512")
    if args.success_confirmations < 1 or args.max_dispatches < 1:
        parser.error("success and dispatch budgets must be positive")
    if args.max_replans < 0 or args.max_reobservations < 0:
        parser.error("recovery budgets must be zero or greater")
    if not -1.0 <= args.memory_similarity_threshold <= 1.0:
        parser.error("--memory-similarity-threshold must be between -1 and 1")
    return args


def build_controller(args: argparse.Namespace):
    """Assemble the production controller while keeping construction testable."""
    from prefmem.agents.services import (
        HRIReasoner,
        MemoryReasoner,
        MonitorReasoner,
        PlannerReasoner,
        ValidatorReasoner,
    )
    from prefmem.controller import ControllerLimits, PrefMemController
    from prefmem.execution import IdempotentVLAAdapter, JsonlVLAAdapter
    from prefmem.memory.embeddings import EmbeddingGemmaClient
    from prefmem.memory.service import PrefMemMemoryService
    from prefmem.memory.store import MarkdownMemoryStore
    from prefmem.observations import (
        DatasetObservationSource,
        LiveCameraObservationSource,
    )
    from prefmem.recording import build_recorder

    configuration = {
        key: value
        for key, value in vars(args).items()
        if "api_key" not in key
    }
    recorder = build_recorder(
        enabled=args.record_context,
        root=args.recording_dir,
        session_name=args.recording_name,
        configuration=configuration,
    )

    endpoint = ModelEndpoint(
        model=args.model,
        base_url=args.model_base_url or "http://127.0.0.1:11434/v1",
        api_key=args.model_api_key,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_agent_tokens,
    )
    if args.model_provider == "ollama":
        from langchain_ollama import ChatOllama

        model = ChatOllama(
            model=args.model,
            temperature=0,
        )
        agent_client = StructuredAgentClient(
            endpoint,
            recorder=recorder,
            model=model,
            structured_method="json_schema",
        )
    else:
        agent_client = StructuredAgentClient(endpoint, recorder=recorder)

    hri = HRIReasoner(agent_client)
    planner = PlannerReasoner(agent_client)
    monitor = MonitorReasoner(agent_client)
    validator = ValidatorReasoner(agent_client)
    memory_reasoner = MemoryReasoner(agent_client)

    preference_path = (
        args.preference_store
        if args.preference_store is not None
        else MarkdownMemoryStore.default_path(args.username)
    )
    store = MarkdownMemoryStore(preference_path)
    embeddings = EmbeddingGemmaClient(
        base_url=args.embedding_base_url,
        model=args.embedding_model,
        api_key=args.embedding_api_key,
        dimensions=args.embedding_dimensions,
    )
    memory = PrefMemMemoryService(
        store,
        embeddings,
        semantic_reasoner=memory_reasoner,
        recorder=recorder,
        minimum_similarity=args.memory_similarity_threshold,
    )

    if args.frame_source == "dataset":
        observation_source = DatasetObservationSource(
            args.dataset,
            recorder=recorder,
            resize_images=args.resize_images,
        )
    else:
        observation_source = LiveCameraObservationSource(
            args.camera_url,
            timeout_seconds=args.camera_timeout,
            recorder=recorder,
            resize_images=args.resize_images,
        )

    vla = None
    if args.execute:
        assert args.vla_command_file is not None
        vla = IdempotentVLAAdapter(
            JsonlVLAAdapter(args.vla_command_file),
            recorder=recorder,
        )

    controller = PrefMemController(
        username=args.username,
        observation_source=observation_source,
        hri=hri,
        planner=planner,
        monitor=monitor,
        validator=validator,
        memory=memory,
        vla=vla,
        plan_only=not args.execute,
        replan_policy=ReplanPolicy(args.replan_policy),
        limits=ControllerLimits(
            max_monitor_checks=args.max_monitor_checks,
            max_replans=args.max_replans,
            max_reobservations=args.max_reobservations,
            monitor_interval_seconds=args.monitor_interval,
            success_confirmations=args.success_confirmations,
            max_dispatches=args.max_dispatches,
        ),
        recorder=recorder,
    )
    return controller, recorder


def _print_result(result: Any, *, display_all: bool) -> None:
    print(result.text)
    if display_all:
        payload = {
            "phase": result.phase,
            "task_id": result.task_id,
            "plan": result.plan,
            "validation": result.validation,
            "memory": result.memory,
            "metadata": result.metadata,
        }
        def serialize(value: Any) -> Any:
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                return model_dump(mode="json")
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, Path):
                return str(value)
            return str(value)

        print(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                default=serialize,
            )
        )


def _load_queries(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        raise ValueError("query file must contain a JSON array of strings")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"PrefMem {version('robopref')}")
    controller, recorder = build_controller(args)
    if recorder.enabled:
        print(f"Recording context to {recorder.markdown_path}")

    queries: list[str] = []
    if args.query is not None:
        queries.append(args.query)
    if args.query_file is not None:
        queries.extend(_load_queries(args.query_file))

    for query in queries:
        _print_result(
            controller.handle_user(query),
            display_all=args.display_all,
        )

    if queries and not args.interactive:
        return

    print(
        "Enter a request, `/remember …`, `/preferences`, `/forget <id>`, "
        "or `quit`."
    )
    while True:
        try:
            user_text = input("User: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.strip().lower() in {"quit", "exit"}:
            break
        _print_result(
            controller.handle_user(user_text),
            display_all=args.display_all,
        )
