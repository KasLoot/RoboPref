from __future__ import annotations

import copy
from typing import Any, Callable, Protocol

from agents.contracts import ExecutionResult, PlanResult
from dataset.episode import DatasetEpisode


class VLAExecutor(Protocol):
    def execute(self, plan: PlanResult, *, attempt: int = 1) -> ExecutionResult: ...


class RecordedEpisodeExecutor:
    """Adapter for this prototype's pre-recorded external robot attempt.

    It records every requested subtask but never claims that this process physically
    executed the robot. The dataset final frame is returned as observation evidence.
    """

    def __init__(self, episode: DatasetEpisode):
        self.episode = episode

    def set_episode(self, episode: DatasetEpisode) -> None:
        self.episode = episode

    def execute_subtask(
        self, subtask: dict[str, Any], *, index: int, attempt: int = 1
    ) -> dict[str, Any]:
        return {
            "index": index,
            "instruction": str(subtask.get("task_instruction", "")),
            "status": "RECORDED_EXTERNAL_ATTEMPT",
            "attempt": attempt,
        }

    def observe(self) -> str:
        return str(self.episode.final_frame)

    def execute(self, plan: PlanResult, *, attempt: int = 1) -> ExecutionResult:
        results = [
            self.execute_subtask(subtask, index=index, attempt=attempt)
            for index, subtask in enumerate(plan.subtasks, start=1)
        ]
        return ExecutionResult(
            status="OBSERVED_RECORDED_ATTEMPT",
            final_observation=str(self.episode.final_frame),
            subtask_results=results,
            evidence={
                "attempt": attempt,
                "source": "recorded_dataset",
                "physical_execution_claimed": False,
            },
        )


class CallableVLAExecutor:
    """Bridge for a real VLA policy/environment supplied by the application."""

    def __init__(
        self,
        execute_subtask: Callable[[dict[str, Any]], dict[str, Any]],
        observation: Callable[[], str],
    ):
        self.execute_subtask_callback = execute_subtask
        self.observation = observation

    def execute(self, plan: PlanResult, *, attempt: int = 1) -> ExecutionResult:
        results: list[dict[str, Any]] = []
        try:
            for index, subtask in enumerate(plan.subtasks, start=1):
                result = self.execute_subtask(
                    copy.deepcopy(subtask), index=index, attempt=attempt
                )
                results.append(result)
                status = str(result.get("status", "")).upper()
                if status in {"FAILED", "UNSAFE", "ABORTED", "CANCELLED"}:
                    return ExecutionResult(
                        status=status,
                        final_observation=self.observe(),
                        subtask_results=results,
                        evidence={"attempt": attempt, "source": "live_vla"},
                        error=str(result.get("error", "VLA subtask failed.")),
                    )
            return ExecutionResult(
                status="COMPLETED",
                final_observation=self.observe(),
                subtask_results=results,
                evidence={"attempt": attempt, "source": "live_vla"},
            )
        except Exception as error:
            return ExecutionResult(
                status="FAILED",
                final_observation=self.observe(),
                subtask_results=results,
                evidence={"attempt": attempt, "source": "live_vla"},
                error=str(error),
            )

    def execute_subtask(
        self, subtask: dict[str, Any], *, index: int, attempt: int = 1
    ) -> dict[str, Any]:
        result = copy.deepcopy(self.execute_subtask_callback(copy.deepcopy(subtask)))
        if not isinstance(result, dict):
            raise TypeError("VLA subtask callback must return an object.")
        result.setdefault("index", index)
        result.setdefault("attempt", attempt)
        return result

    def observe(self) -> str:
        return self.observation()
