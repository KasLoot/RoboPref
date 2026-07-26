"""Production construction and audit sinks for conversation evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from agents.configs import PrefMemConfig
from agents.diagnostics import AgentOutputDisplay
from agents.hri import HRIOrchestrator
from agents.memory import MemoryAgent
from agents.model import JsonModel, build_json_model
from agents.planner import PlannerAgent
from agents.validator import ValidatorAgent
from dataset.benchmark import BenchmarkEpisode
from memory.repositories import HistoryRepository, PreferenceRepository

from .ablations import apply_memory_mode, record_memory_context
from .conversation_models import ConversationEvaluationConfig, ConversationRunContext
from .executor import CounterfactualSequenceExecutor
from .model_defaults import apply_evaluation_model_defaults


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Durably append one already-sanitized evaluation event."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class EvaluationArtifactSink:
    """Non-throwing observer with an explicit completeness signal."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.events_written = 0
        self.errors: list[dict[str, str]] = []

    def __call__(self, event: Mapping[str, Any]) -> None:
        try:
            _append_jsonl(self.path, event)
        except Exception as error:
            self.errors.append(
                {"type": type(error).__name__, "message": str(error)}
            )
            return
        self.events_written += 1

    def status(self, *, require_event: bool = True) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "events_written": self.events_written,
            "error_count": len(self.errors),
            "errors": [dict(item) for item in self.errors],
            "complete": not self.errors
            and (self.events_written > 0 or not require_event),
        }


class EvaluationArtifactRecorder:
    """Own both structured audit streams for one isolated conversation."""

    def __init__(self, artifact_dir: str | Path) -> None:
        artifact_dir = Path(artifact_dir)
        self.model_calls = EvaluationArtifactSink(
            artifact_dir / "model_calls.jsonl"
        )
        self.agent_events = EvaluationArtifactSink(
            artifact_dir / "agent_events.jsonl"
        )

    def status(self) -> dict[str, Any]:
        streams = {
            "model_calls": self.model_calls.status(),
            "agent_events": self.agent_events.status(),
        }
        return {
            "complete": all(item["complete"] for item in streams.values()),
            "streams": streams,
        }


def build_prefmem_config(
    evaluation: ConversationEvaluationConfig,
    episode: BenchmarkEpisode,
    run_dir: str | Path,
    user_id: str,
    *,
    model_seed: int | None,
) -> PrefMemConfig:
    """Build one cold, run-local PrefMem configuration."""

    memory_dir = Path(run_dir) / "memory"
    config = PrefMemConfig(
        workspace_root=str(evaluation.benchmark_root),
        dataset_path=str(episode.episode.directory),
        history_store_path=str(memory_dir / "history.json"),
        history_outbox_path=str(memory_dir / "history_outbox.json"),
        preference_store_path=str(memory_dir / "preferences.json"),
        user_id=user_id,
        max_replans=evaluation.max_replans,
        max_reobservations=evaluation.max_reobservations,
    )
    if evaluation.model_provider in {None, "vllm"}:
        apply_evaluation_model_defaults(config)
    config.vision.resize_images = evaluation.resize_images
    for model_config in (
        config.hri,
        config.memory,
        config.planner,
        config.validator,
    ):
        if evaluation.model:
            model_config.model = evaluation.model
        if evaluation.model_provider:
            model_config.provider = evaluation.model_provider
        if evaluation.model_base_url:
            model_config.base_url = evaluation.model_base_url
        if evaluation.ollama_host:
            model_config.host = evaluation.ollama_host
        if evaluation.temperature is not None:
            model_config.temperature = evaluation.temperature
        model_config.seed = model_seed
        if evaluation.timeout_seconds is not None:
            model_config.timeout_seconds = evaluation.timeout_seconds
        model_config.__post_init__()
    return config


def _model(
    config: Any,
    *,
    agent_name: str,
    telemetry_observer: Callable[[dict[str, Any]], None],
) -> JsonModel:
    return build_json_model(
        config,
        agent_name=agent_name,
        telemetry_observer=telemetry_observer,
    )


def build_conversation_orchestrator(
    config: PrefMemConfig,
    executor: CounterfactualSequenceExecutor,
    artifact_dir: str | Path,
    *,
    memory_mode: str = "full",
    display_all: bool = False,
) -> HRIOrchestrator:
    """Build PrefMem with isolated repositories and recorded memory delivery."""

    recorder = EvaluationArtifactRecorder(artifact_dir)
    output_display = AgentOutputDisplay(
        enabled=display_all,
        observer=recorder.agent_events,
    )
    telemetry = recorder.model_calls
    history_repository = HistoryRepository(config.history_store_path)
    preference_repository = PreferenceRepository(config.preference_store_path)
    memory_agent: Any = MemoryAgent(
        config.memory,
        history_repository,
        preference_repository,
        model=_model(
            config.memory,
            agent_name="Memory Agent",
            telemetry_observer=telemetry,
        ),
        output_display=output_display,
        recent_history_limit=config.recent_history_limit,
        history_semantic_scan_limit=config.history_semantic_scan_limit,
        semantic_history_limit=config.semantic_history_limit,
        semantic_preference_limit=config.semantic_preference_limit,
        semantic_preference_threshold=config.semantic_preference_threshold,
        preference_proposal_min_episodes=config.preference_proposal_min_episodes,
        preference_proposal_confidence=config.preference_proposal_confidence,
        compact_after_write=config.compact_preferences_after_write,
    )
    memory_agent = record_memory_context(
        apply_memory_mode(memory_agent, memory_mode)
    )
    planner_agent = PlannerAgent(
        config.planner,
        vision=config.vision,
        model=_model(
            config.planner,
            agent_name="Planner Agent",
            telemetry_observer=telemetry,
        ),
        output_display=output_display,
    )
    validator_agent = ValidatorAgent(
        config.validator,
        vision=config.vision,
        model=_model(
            config.validator,
            agent_name="Validator Agent",
            telemetry_observer=telemetry,
        ),
        output_display=output_display,
    )
    orchestrator = HRIOrchestrator(
        config,
        hri_model=_model(
            config.hri,
            agent_name="HRI Agent",
            telemetry_observer=telemetry,
        ),
        memory_agent=memory_agent,
        planner_agent=planner_agent,
        validator_agent=validator_agent,
        executor=executor,
        output_display=output_display,
    )
    orchestrator.evaluation_artifact_status = recorder.status
    return orchestrator


def default_conversation_orchestrator_factory(
    context: ConversationRunContext,
) -> HRIOrchestrator:
    first = context.episodes[context.case.commands[0].episode_id]
    episode = BenchmarkEpisode.from_path(
        first.path,
        context.config.benchmark_root,
    )
    executor = CounterfactualSequenceExecutor(episode)
    config = build_prefmem_config(
        context.config,
        episode,
        context.run_directory,
        context.case.user_id,
        model_seed=context.model_seed,
    )
    return build_conversation_orchestrator(
        config,
        executor,
        context.run_directory / "artifacts",
        memory_mode=context.config.memory_mode,
        display_all=context.config.display_all,
    )


__all__ = [
    "EvaluationArtifactRecorder",
    "EvaluationArtifactSink",
    "build_conversation_orchestrator",
    "build_prefmem_config",
    "default_conversation_orchestrator_factory",
]

