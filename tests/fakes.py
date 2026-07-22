from __future__ import annotations

import copy
from collections import defaultdict, deque
from typing import Any, Callable, Sequence

from agents.contracts import ExecutionResult, PlanResult, ValidationResult
from memory.models import MemoryContext


JsonResponse = dict[str, Any]
JsonHandler = Callable[[dict[str, Any]], JsonResponse]


class ScriptedJsonModel:
    """JsonModel fake that never reaches Ollama and records every request."""

    def __init__(
        self,
        scripts: dict[str, JsonResponse | JsonHandler | list[JsonResponse | JsonHandler]],
    ) -> None:
        self._scripts: dict[str, JsonResponse | JsonHandler | deque[JsonResponse | JsonHandler]] = {}
        for purpose, script in scripts.items():
            self._scripts[purpose] = deque(script) if isinstance(script, list) else script
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        purpose: str,
        system_prompt: str,
        payload: dict[str, Any],
        images: Sequence[bytes] = (),
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "purpose": purpose,
                "system_prompt": system_prompt,
                "payload": copy.deepcopy(payload),
                "images": list(images),
            }
        )
        if purpose not in self._scripts:
            raise AssertionError(f"Unexpected model purpose: {purpose}")
        script = self._scripts[purpose]
        if isinstance(script, deque):
            if not script:
                raise AssertionError(f"No scripted responses remain for {purpose}")
            response = script.popleft()
        else:
            response = script
        if callable(response):
            response = response(copy.deepcopy(payload))
        return copy.deepcopy(response)

    def calls_for(self, purpose: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["purpose"] == purpose]


class RecordingMemoryAgent:
    """Small HRI-facing memory fake for state-machine tests."""

    def __init__(self, contexts: list[MemoryContext] | None = None) -> None:
        self._contexts = deque(copy.deepcopy(contexts or [MemoryContext()]))
        self.context_queries: list[Any] = []
        self.history_updates: list[dict[str, Any]] = []
        self.preference_updates: list[dict[str, Any]] = []
        self._history_version = 0

    def get_memory_context(self, query: Any) -> MemoryContext:
        self.context_queries.append(copy.deepcopy(query))
        if len(self._contexts) > 1:
            return self._contexts.popleft()
        return copy.deepcopy(self._contexts[0])

    def update_history_memory(self, source: dict[str, Any]) -> dict[str, Any]:
        self.history_updates.append(copy.deepcopy(source))
        self._history_version += 1
        return {
            "episode": copy.deepcopy(source),
            "history_summary": {
                "text": f"History version {self._history_version}",
                "source_episode_ids": [source["episode_id"]],
            },
            "history_version": self._history_version,
        }

    def update_preference_memory(
        self,
        request: dict[str, Any],
        *,
        consent: Any,
        transaction_id: str,
    ) -> dict[str, Any]:
        call = {
            "request": copy.deepcopy(request),
            "consent": copy.deepcopy(consent),
            "transaction_id": transaction_id,
        }
        self.preference_updates.append(call)
        return {"transaction": {"transaction_id": transaction_id}, "compaction": None}


class FixedPlanner:
    def __init__(self, result: PlanResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def plan(
        self,
        task_contract: dict[str, Any],
        image_path: str,
        *,
        recovery_context: dict[str, Any] | None = None,
    ) -> PlanResult:
        self.calls.append(
            {
                "task_contract": copy.deepcopy(task_contract),
                "image_path": image_path,
                "recovery_context": copy.deepcopy(recovery_context),
            }
        )
        return copy.deepcopy(self.result)


class FixedValidator:
    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def validate(self, validation_spec: Any, execution: ExecutionResult) -> ValidationResult:
        self.calls.append(
            {
                "validation_spec": copy.deepcopy(validation_spec),
                "execution": copy.deepcopy(execution),
            }
        )
        return copy.deepcopy(self.result)


class RecordingExecutor:
    def __init__(self, final_observation: str) -> None:
        self.final_observation = final_observation
        self.calls: list[dict[str, Any]] = []

    def execute(self, plan: PlanResult, *, attempt: int = 1) -> ExecutionResult:
        self.calls.append({"plan": copy.deepcopy(plan), "attempt": attempt})
        return ExecutionResult(
            status="COMPLETED",
            final_observation=self.final_observation,
            subtask_results=[{"status": "COMPLETED"}],
            evidence={"attempt": attempt, "source": "offline-test"},
        )

