"""Production PrefMem construction for benchmark evaluations."""

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
from memory.repositories import HistoryRepository, PreferenceRepository

from .ablations import apply_memory_mode
from .executor import BenchmarkEpisodeExecutor


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
    """Best-effort observer with an explicit post-run completeness signal.

    Agent diagnostics must never interrupt robot orchestration, so observer
    failures remain non-throwing. Unlike a bare callback, this sink retains the
    failure for the out-of-band evaluator to mark the experiment artifact
    incomplete.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.events_written = 0
        self.errors: list[dict[str, str]] = []

    def __call__(self, event: Mapping[str, Any]) -> None:
        try:
            _append_jsonl(self.path, event)
        except Exception as error:
            self.errors.append(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            return
        self.events_written += 1

    def status(self, *, require_event: bool = True) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "events_written": self.events_written,
            "error_count": len(self.errors),
            "errors": [dict(item) for item in self.errors],
            "complete": (
                not self.errors
                and (self.events_written > 0 or not require_event)
            ),
        }


class EvaluationArtifactRecorder:
    """Own the two structured audit streams for one isolated run."""

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
            "complete": all(
                stream["complete"] for stream in streams.values()
            ),
            "streams": streams,
        }


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


def build_evaluation_orchestrator(
    config: PrefMemConfig,
    executor: BenchmarkEpisodeExecutor,
    artifact_dir: str | Path,
    *,
    memory_mode: str = "full",
    display_all: bool = False,
) -> HRIOrchestrator:
    """Build PrefMem with isolated repositories and structured audit sinks."""

    artifact_dir = Path(artifact_dir)
    artifact_recorder = EvaluationArtifactRecorder(artifact_dir)
    output_display = AgentOutputDisplay(
        enabled=display_all,
        observer=artifact_recorder.agent_events,
    )
    telemetry = artifact_recorder.model_calls

    history_repository = HistoryRepository(config.history_store_path)
    preference_repository = PreferenceRepository(config.preference_store_path)
    memory_agent = MemoryAgent(
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
    memory_agent = apply_memory_mode(memory_agent, memory_mode)
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
    # Evaluation runners inspect this only after a turn has returned. It never
    # enters an agent prompt or changes the agent's control flow.
    orchestrator.evaluation_artifact_status = artifact_recorder.status
    return orchestrator


def evaluation_orchestrator_factory(
    *,
    memory_mode: str = "full",
    display_all: bool = False,
) -> Callable[..., HRIOrchestrator]:
    """Adapt :func:`build_evaluation_orchestrator` to runner factory hooks."""

    def factory(
        config: PrefMemConfig,
        _episode: Any,
        executor: BenchmarkEpisodeExecutor,
        artifact_dir: Path,
        _user_id: str,
    ) -> HRIOrchestrator:
        return build_evaluation_orchestrator(
            config,
            executor,
            artifact_dir,
            memory_mode=memory_mode,
            display_all=display_all,
        )

    return factory


__all__ = [
    "EvaluationArtifactRecorder",
    "EvaluationArtifactSink",
    "build_evaluation_orchestrator",
    "evaluation_orchestrator_factory",
]
