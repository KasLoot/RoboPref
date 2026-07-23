from __future__ import annotations

from typing import Any

from agents.contracts import ExecutionResult, PlanResult
from agents.vla import RecordedEpisodeExecutor
from dataset.benchmark import BenchmarkEpisode
from dataset.episode import DatasetEpisode


class BenchmarkEpisodeExecutor(RecordedEpisodeExecutor):
    """Recorded-episode adapter that can replay a declared safety interruption.

    Endpoint geometry alone cannot distinguish a safe success from an unsafe
    trajectory. For an ``unsafe`` benchmark case this adapter returns ``UNSAFE``
    from the first subtask, exercising HRI's execution gate and ensuring Validator
    is not called. It does not expose target predicates or expected labels to any
    model.
    """

    def __init__(self, benchmark_episode: BenchmarkEpisode):
        self.benchmark_episode = benchmark_episode
        super().__init__(benchmark_episode.episode)

    def set_benchmark_episode(self, benchmark_episode: BenchmarkEpisode) -> None:
        self.benchmark_episode = benchmark_episode
        self.episode = benchmark_episode.episode

    def set_episode(self, episode: DatasetEpisode) -> None:
        """Follow HRI dataset switching without retaining the previous oracle."""

        benchmark_episode = BenchmarkEpisode.from_path(
            episode.directory,
            episode.directory.parent,
        )
        self.benchmark_episode = benchmark_episode
        self.episode = episode

    @property
    def expected_outcome(self) -> str:
        return str(self.benchmark_episode.manifest.get("expected_outcome", ""))

    def model_scene_context(self) -> dict[str, Any]:
        return self.benchmark_episode.model_context()

    def execute_subtask(
        self, subtask: dict[str, Any], *, index: int, attempt: int = 1
    ) -> dict[str, Any]:
        result = super().execute_subtask(subtask, index=index, attempt=attempt)
        if self.expected_outcome != "unsafe":
            return result
        ground_truth = self.benchmark_episode.manifest.get("ground_truth", {})
        failure_mode = (
            str(ground_truth.get("failure_mode") or "safety_constraint_violated")
            if isinstance(ground_truth, dict)
            else "safety_constraint_violated"
        )
        return {
            **result,
            "status": "UNSAFE",
            "error": "The simulated execution crossed a configured safety limit.",
            "safety_event": failure_mode,
            "physical_execution_claimed": False,
        }

    def execute(self, plan: PlanResult, *, attempt: int = 1) -> ExecutionResult:
        if self.expected_outcome != "unsafe":
            return super().execute(plan, attempt=attempt)
        first_subtask = (
            plan.subtasks[0]
            if plan.subtasks
            else {"task_instruction": "Simulated benchmark execution."}
        )
        result = self.execute_subtask(
            first_subtask,
            index=1,
            attempt=attempt,
        )
        return ExecutionResult(
            status="UNSAFE",
            final_observation=self.observe(),
            subtask_results=[result],
            evidence={
                "attempt": attempt,
                "source": "recorded_benchmark",
                "physical_execution_claimed": False,
                "safety_event": result["safety_event"],
            },
            error=str(result["error"]),
        )
