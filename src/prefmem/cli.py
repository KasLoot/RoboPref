from __future__ import annotations

import argparse
import ipaddress
from importlib.metadata import version
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit


DEFAULT_DATASET = Path("dataset/v3")
DEFAULT_TRANSCRIPT = Path("transcripts/transcript.txt")
DEFAULT_MEMORY_STORE = Path("./memory_store")
DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_MODEL_PROVIDER = "ollama"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_CAMERA_BASE_URL = "http://127.0.0.1:1234"
DEFAULT_SAM_BASE_URL = "http://127.0.0.1:9000"
DEFAULT_EXECUTION_MODEL = "/workspace/models/gemma-4-26B-A4B-it"
MAX_SIMULATION_RENDER_SIZE = 1024


def _http_url(value: str, *, allow_path: bool = True) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise argparse.ArgumentTypeError("must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise argparse.ArgumentTypeError("URL credentials are not allowed")
    if parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("URL queries and fragments are not allowed")
    if not allow_path and parsed.path not in {"", "/"}:
        raise argparse.ArgumentTypeError("must not contain a path")
    return value.rstrip("/")


def _loopback_http_url(value: str) -> str:
    result = _http_url(value, allow_path=False)
    hostname = urlsplit(result).hostname
    assert hostname is not None
    if hostname.casefold() == "localhost":
        return result
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must use a loopback host") from error
    if not address.is_loopback:
        raise argparse.ArgumentTypeError("must use a loopback host")
    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def _probability(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return result


def _integer_at_least(minimum: int):
    def parse(value: str) -> int:
        try:
            result = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if result < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return result

    return parse


def _integer_between(minimum: int, maximum: int):
    def parse(value: str) -> int:
        result = _integer_at_least(minimum)(value)
        if result > maximum:
            raise argparse.ArgumentTypeError(f"must be at most {maximum}")
        return result

    return parse


def build_parser() -> argparse.ArgumentParser:
    """Build the authoritative PrefMem command-line parser."""

    parser = argparse.ArgumentParser(
        prog="prefmem",
        description="Run the PrefMem receding-horizon agentic system.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        metavar="PATH",
        help="Episode directory used by offline benchmark helpers.",
    )
    parser.add_argument("--benchmark", action="store_true")

    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument(
        "--preference-store",
        type=Path,
        default=None,
        metavar="PATH",
        help="Persistent preference-memory directory.",
    )
    memory_group.add_argument(
        "--memory-store",
        "--memory-store-path",
        dest="legacy_memory_store",
        type=Path,
        default=None,
        metavar="PATH",
        help="Alias for --preference-store.",
    )

    parser.add_argument("--username", default="default")
    parser.add_argument(
        "--transcript",
        type=Path,
        default=DEFAULT_TRANSCRIPT,
        metavar="PATH",
    )
    parser.add_argument(
        "--display-all",
        "--display_all",
        dest="display_all",
        action="store_true",
    )
    parser.add_argument("--resize-images", action="store_true")

    # `model-provider` is retained for compatibility with the earlier CLI.  The
    # running agents use model-config; its default is vLLM for the local port
    # forwarding workflow documented by this project.
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--model-provider",
        choices=("ollama", "vllm"),
        default=DEFAULT_MODEL_PROVIDER,
    )
    parser.add_argument("--model-base-url", default=None)
    parser.add_argument(
        "--model-config",
        choices=("vllm", "ollama"),
        default="vllm",
    )

    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--query-file", type=Path, default=None)
    parser.add_argument(
        "--think",
        nargs="*",
        default=["HRI"],
        metavar="AGENT",
        help=(
            "Agents allowed to use model thinking "
            "(Monitor and Validator are always disabled)."
        ),
    )
    parser.add_argument("--print-raw", action="store_true")
    parser.add_argument("--print-usage", action="store_true")
    parser.add_argument(
        "--render-agent-graph",
        action="store_true",
        help="Write the HRI graph image before starting the conversation.",
    )

    parser.add_argument(
        "--camera-base-url",
        default=DEFAULT_CAMERA_BASE_URL,
        help="Base URL for the live snapshot and task display server.",
    )
    parser.add_argument(
        "--executor",
        choices=("human", "mujoco"),
        default="human",
        help="Use the existing human task surface or the MuJoCo Panda executor.",
    )
    parser.add_argument(
        "--sam-base-url",
        default=DEFAULT_SAM_BASE_URL,
        help="Forwarded SAM 3.1 service used by the MuJoCo executor.",
    )
    parser.add_argument("--sam-threshold", type=_probability, default=0.5)
    parser.add_argument("--execution-model", default=DEFAULT_EXECUTION_MODEL)
    parser.add_argument(
        "--execution-model-base-url",
        default=DEFAULT_VLLM_BASE_URL,
        help="OpenAI-compatible Gemma 4 endpoint for instruction compilation.",
    )
    parser.add_argument("--simulation-seed", type=int, default=0)
    parser.add_argument(
        "--simulation-scene",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Optional MuJoCo XML scene. The default is the built-in block "
            "stacking scene."
        ),
    )
    parser.add_argument(
        "--simulation-viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open MuJoCo's interactive viewer (use Ctrl+right-drag on blocks).",
    )
    parser.add_argument(
        "--simulation-viewer-camera",
        choices=("task", "overview"),
        default="overview",
        help=(
            "Initial MuJoCo viewer camera. 'overview' is freely navigable and "
            "shows the robot; 'task' matches Monitor and is fixed."
        ),
    )
    parser.add_argument(
        "--simulation-render-size",
        type=_integer_between(128, MAX_SIMULATION_RENDER_SIZE),
        default=640,
        metavar="PIXELS",
        help="Square RGB-D size in pixels (128-1024).",
    )
    parser.add_argument(
        "--simulation-render-hz",
        type=_positive_float,
        default=5.0,
    )
    parser.add_argument(
        "--monitor-min-interval",
        type=_positive_float,
        default=1.0,
    )
    parser.add_argument(
        "--monitor-events",
        choices=("auto", "off", "summary", "verbose"),
        default="auto",
        help=(
            "Live Monitor telemetry. 'auto' selects summary for MuJoCo and "
            "off for the human executor."
        ),
    )
    parser.add_argument(
        "--monitor-timeout",
        type=_positive_float,
        default=30.0,
    )
    parser.add_argument(
        "--success-confirmations",
        type=_integer_at_least(2),
        default=2,
    )
    parser.add_argument(
        "--success-stability-seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--failure-confirmations",
        type=_integer_at_least(2),
        default=2,
    )
    parser.add_argument("--max-planning-cycles", type=_integer_at_least(1), default=20)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.model_base_url is not None:
        if args.model_provider != "vllm":
            parser.error("--model-base-url requires --model-provider vllm")
        if args.model_base_url.endswith("/"):
            parser.error("--model-base-url must not end with a slash")
        try:
            args.model_base_url = _http_url(args.model_base_url)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
    elif args.model_provider == "vllm":
        args.model_base_url = DEFAULT_VLLM_BASE_URL

    try:
        args.camera_base_url = _http_url(args.camera_base_url, allow_path=False)
    except argparse.ArgumentTypeError as error:
        parser.error(f"--camera-base-url {error}")
    try:
        args.sam_base_url = _loopback_http_url(args.sam_base_url)
    except argparse.ArgumentTypeError as error:
        parser.error(f"--sam-base-url {error}")
    try:
        args.execution_model_base_url = _http_url(
            args.execution_model_base_url
        )
    except argparse.ArgumentTypeError as error:
        parser.error(f"--execution-model-base-url {error}")
    if args.success_stability_seconds < 0:
        parser.error("--success-stability-seconds must be non-negative")

    args.preference_store = (
        args.legacy_memory_store
        if args.legacy_memory_store is not None
        else args.preference_store
    )
    args.memory_store_path = args.preference_store or DEFAULT_MEMORY_STORE
    if args.monitor_events == "auto":
        args.monitor_events = "summary" if args.executor == "mujoco" else "off"
    del args.legacy_memory_store
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"PrefMem {version('robopref')}")
    print(f"Parsed arguments: {args}")

    from prefmem.runtime import build_runtime

    runtime, hri_agent = build_runtime(args)
    if args.render_agent_graph:
        hri_agent.get_agent_graph()

    print(
        "\n"
        + "=" * 20
        + f" PrefMem Conversation Starts | Think: {args.think} "
        + "=" * 20
        + "\n"
    )
    try:
        hri_agent.run()
    finally:
        runtime.close()


__all__ = [
    "DEFAULT_CAMERA_BASE_URL",
    "DEFAULT_DATASET",
    "DEFAULT_EXECUTION_MODEL",
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_PROVIDER",
    "DEFAULT_SAM_BASE_URL",
    "DEFAULT_TRANSCRIPT",
    "DEFAULT_VLLM_BASE_URL",
    "build_parser",
    "main",
    "parse_args",
]
