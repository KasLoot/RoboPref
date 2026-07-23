from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from agents.configs import PrefMemConfig
from agents.hri import HRIOrchestrator
from transcript import TerminalTranscript


WORKSPACE_ROOT = Path(__file__).resolve().parent


class PrefMem:
    def __init__(
        self,
        config: PrefMemConfig | None = None,
        *,
        executor: Any = None,
    ):
        self.hri_agent = HRIOrchestrator(
            config or PrefMemConfig(),
            executor=executor,
        )

    def start(self, *, display_all: bool = False) -> None:
        print("Starting PrefMem...")
        self.hri_agent.get_response(display_all=display_all)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the redesigned PrefMem prototype.")
    parser.add_argument("--dataset", help="Episode directory containing numbered frames.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Treat --dataset as a generated benchmark packet. This enables the "
            "manifest-aware safety-stop adapter without exposing oracle labels to models."
        ),
    )
    parser.add_argument("--history-store", help="Participant episodic-history JSON path.")
    parser.add_argument("--history-outbox", help="Durable failed-history retry queue path.")
    parser.add_argument("--preference-store", help="Approved semantic-preference JSON path.")
    parser.add_argument(
        "--memory-store",
        dest="preference_store",
        help="Deprecated alias for --preference-store.",
    )
    parser.add_argument("--user-id", default="default", help="Non-identifying participant code.")
    parser.add_argument("--transcript", help="Terminal transcript output path.")
    parser.add_argument(
        "--display_all",
        "--display-all",
        dest="display_all",
        action="store_true",
        help="Display complete structured outputs from all PrefMem agents and assurance gates.",
    )
    parser.add_argument("--resize-images", action="store_true", help="Resize model images to 640×480 bounds.")
    parser.add_argument("--max-replans", type=int, default=1)
    parser.add_argument("--max-reobservations", type=int, default=1)
    parser.add_argument("--model", help="Override the model for all four VLM agents.")
    parser.add_argument("--ollama-host", help="Ollama service URL override.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PrefMemConfig(
        user_id=args.user_id,
        max_replans=max(0, args.max_replans),
        max_reobservations=max(0, args.max_reobservations),
    )
    if args.dataset:
        config.dataset_path = args.dataset
    if args.history_store:
        config.history_store_path = args.history_store
    if args.history_outbox:
        config.history_outbox_path = args.history_outbox
    if args.preference_store:
        config.preference_store_path = args.preference_store
    config.vision.resize_images = args.resize_images
    for agent_config in (
        config.hri,
        config.memory,
        config.planner,
        config.validator,
    ):
        if args.model:
            agent_config.model = args.model
        if args.ollama_host:
            agent_config.host = args.ollama_host
    transcript_path = (
        Path(args.transcript)
        if args.transcript
        else WORKSPACE_ROOT / "experiments" / "record.txt"
    )
    executor = None
    if args.benchmark:
        if not args.dataset:
            raise SystemExit("--benchmark requires --dataset.")
        from dataset.benchmark import BenchmarkEpisode
        from simulation.benchmark.executor import BenchmarkEpisodeExecutor

        benchmark_episode = BenchmarkEpisode.from_path(
            config.dataset_path,
            config.workspace_root,
        )
        executor = BenchmarkEpisodeExecutor(benchmark_episode)
    with TerminalTranscript(transcript_path):
        PrefMem(config, executor=executor).start(display_all=args.display_all)


if __name__ == "__main__":
    main()
