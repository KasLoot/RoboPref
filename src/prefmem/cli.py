from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit
from langchain.messages import SystemMessage, HumanMessage



DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_MODEL_PROVIDER = "ollama"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"


def build_parser() -> argparse.ArgumentParser:
    """Build the authoritative PrefMem command-line parser."""
    parser = argparse.ArgumentParser(
        prog="prefmem",
        description="Run the PrefMem agentic system.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="dataset/v3",
        metavar="PATH",
        help=(
            "Episode directory containing numbered frames such as 1.png and "
            "2.png. Set it to one generated packet directory when using "
            "--benchmark. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Treat --dataset as one generated benchmark packet directory.",
    )

    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument(
        "--preference-store",
        type=Path,
        default=None,
        metavar="PATH",
        help="Approved preference-store path.",
    )
    memory_group.add_argument(
        "--memory-store",
        dest="preference_store",
        type=Path,
        metavar="PATH",
        help="Deprecated alias for --preference-store; do not pass both.",
    )

    parser.add_argument(
        "--username",
        default="default",
        help=(
            "Non-identifying participant namespace used for history and "
            "preference ownership. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--transcript",
        type=str,
        default="transcripts/transcript.txt",
        metavar="PATH",
        help=(
            "Terminal-session output path. Parent directories are created by "
            "the transcript writer. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--display-all",
        dest="display_all",
        action="store_true",
        help=(
            "Print structured HRI, Memory, Planner, VLA, Validator, and "
            "assurance diagnostics."
        ),
    )
    parser.add_argument(
        "--resize-images",
        action="store_true",
        help="Resize model-bound images to 640x480 bounds before inference.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="vllm",
        choices=["vllm", "ollama"],
        help=(
            "Model configuration to use (register in \"src/prefmem/agents/config.py\"). (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run PrefMem in interactive mode.",
    )

    parser.add_argument(
        "--query-file",
        type=Path,
        default=None,
        help="Path to a JSON file containing queries for the agent.",
    )

    parser.add_argument(
        "--think",
        default=["HRI"],
        help="Enable thinking model. Provide a list of agents to enable thinking. e.g. [all] | [HRI, Memory, Planner, Validator]",
    )

    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print each complete model response object as formatted JSON.",
    )

    parser.add_argument(
        "--print-usage",
        action="store_true",
        help="Print the usage of the PrefMem system.",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate PrefMem command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)

    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"PrefMem {version('robopref')}")
    # print the parsed arguments for debugging
    print(f"Parsed arguments: {args}")

    from prefmem.agents.hri import HRI_Agent

    hri_agent = HRI_Agent(model_config=args.model_config, args=args)

    hri_agent.get_agent_graph()

    print("\n\n\n" + "="*20 + f" PrefMem Conversation Starts | Think: {args.think} " + "="*20 + "\n\n\n")

    hri_agent.run()
