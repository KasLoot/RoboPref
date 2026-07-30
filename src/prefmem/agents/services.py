"""Thin typed wrappers around PrefMem's independent reasoning roles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from prefmem.agents.contracts import (
    HRIDecision,
    MemoryContext,
    MemoryRetrievalRequest,
    MemoryStatus,
    MonitorRequest,
    MonitorResult,
    Observation,
    PlanResult,
    PlanningRequest,
    TaskContract,
    ValidationRequest,
    ValidationResult,
    ValidationSpec,
)
from prefmem.agents.model_client import StructuredAgentClient


_PROMPT_ROOT = Path(__file__).resolve().parent / "prompt"


def load_prompt(relative_path: str) -> str:
    path = _PROMPT_ROOT / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required agent prompt is missing: {path}") from exc


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def _observation_metadata(observation: Observation) -> dict[str, Any]:
    return observation.model_dump(
        mode="json",
        exclude={"image_block"},
    )


class HRIReasoner:
    def __init__(
        self,
        client: StructuredAgentClient,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt or load_prompt("hri/hri-prompt-v3.md")

    def decide(
        self,
        *,
        username: str,
        user_query: str,
        observation: Observation,
        memory: MemoryContext,
        conversation: Sequence[Mapping[str, str]],
        pending_memory_consent: Mapping[str, Any] | None = None,
        frame_path: Path | None = None,
        event: str = "USER_MESSAGE",
    ) -> HRIDecision:
        payload = {
            "event": event,
            "user_query": user_query,
            "current_frame": _observation_metadata(observation),
            "temporal_history_memory": {
                "username": username,
                "dialogue": list(conversation),
                "pending_question": (
                    pending_memory_consent.get("pending_question")
                    if pending_memory_consent is not None
                    else None
                ),
                "pending_memory_consent": (
                    dict(pending_memory_consent)
                    if pending_memory_consent is not None
                    else None
                ),
            },
            "persistent_memory_status": memory.status,
            "memory_context": memory.model_dump(mode="json"),
        }
        return self.client.invoke(
            agent="HRI",
            schema=HRIDecision,
            system_prompt=self.system_prompt,
            text=_json_text(payload),
            frame=observation.image_block,
            frame_paths=[frame_path] if frame_path is not None else (),
            metadata={"observation_id": observation.observation_id},
            structured_method="json_mode",
        )

class MemoryReasoner:
    def __init__(
        self,
        client: StructuredAgentClient,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt or load_prompt(
            "memory/prompt-v2.md"
        )

    def retrieve(
        self,
        *,
        request: MemoryRetrievalRequest,
        preference_candidates: Sequence[Mapping[str, Any]],
        history_candidates: Sequence[Mapping[str, Any]],
    ) -> MemoryContext:
        payload = {
            "operation": "RETRIEVE_MEMORY",
            "request": request.model_dump(mode="json"),
            "preference_candidates": list(preference_candidates),
            "history_candidates": list(history_candidates),
        }
        result = self.client.invoke(
            agent="Memory",
            schema=MemoryContext,
            system_prompt=self.system_prompt,
            text=_json_text(payload),
            metadata={"request_id": request.request_id},
        )
        if result.request_id != request.request_id:
            raise ValueError(
                "Memory Agent did not copy the host request ID exactly"
            )
        if result.status not in {MemoryStatus.AVAILABLE, MemoryStatus.EMPTY}:
            raise ValueError(
                "Memory Agent retrieval must return AVAILABLE or EMPTY"
            )

        preferences_by_id = {
            str(candidate["record_id"]): candidate
            for candidate in preference_candidates
        }
        history_by_id = {
            str(candidate["record_id"]): candidate
            for candidate in history_candidates
        }
        returned_ids: list[str] = []

        def canonical_preference(item: Mapping[str, Any]) -> dict[str, Any]:
            record_id = str(item.get("record_id", ""))
            candidate = preferences_by_id.get(record_id)
            if candidate is None:
                raise ValueError(
                    "Memory Agent returned a preference ID outside the "
                    "host-supplied preference candidates"
                )
            returned_ids.append(record_id)
            record = dict(candidate["record"])
            return {
                "record_id": record_id,
                "statement": record["statement"],
                "scope": record["scope"],
                "applicability": record.get("applicability", {}),
                "structured_value": record.get("structured_value"),
                "status": record.get("status"),
                "revision": record.get("revision"),
                **{
                    key: item[key]
                    for key in ("relation", "confidence", "reason")
                    if key in item
                },
            }

        def canonical_history(item: Mapping[str, Any]) -> dict[str, Any]:
            record_id = str(item.get("record_id", ""))
            candidate = history_by_id.get(record_id)
            if candidate is None:
                raise ValueError(
                    "Memory Agent returned a history ID outside the "
                    "host-supplied history candidates"
                )
            returned_ids.append(record_id)
            record = dict(candidate["record"])
            return {
                "record_id": record_id,
                "summary": record.get("summary") or str(record.get("result", "")),
                "outcome": record.get("result"),
                "authority": "HISTORY_ONLY",
                **{
                    key: item[key]
                    for key in ("confidence", "reason")
                    if key in item
                },
            }

        canonical = result.model_copy(
            update={
                "relevant_preferences": [
                    canonical_preference(item)
                    for item in result.relevant_preferences
                ],
                "relevant_history": [
                    canonical_history(item)
                    for item in result.relevant_history
                ],
                "conflicts": [
                    canonical_preference(item) for item in result.conflicts
                ],
            }
        )
        if len(returned_ids) != len(set(returned_ids)):
            raise ValueError(
                "Memory Agent returned the same record more than once"
            )
        if len(returned_ids) > request.limit:
            raise ValueError(
                "Memory Agent exceeded the host retrieval result limit"
            )
        return canonical


class PlannerReasoner:
    def __init__(
        self,
        client: StructuredAgentClient,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt or load_prompt(
            "planner/planner-prompt-v2.md"
        )

    def plan(
        self,
        *,
        planning_request: PlanningRequest,
        task_contract: TaskContract,
        observation: Observation,
        recovery_context: Mapping[str, Any] | None = None,
        frozen_validation_spec: ValidationSpec | None = None,
        frame_path: Path | None = None,
    ) -> PlanResult:
        bounded_recovery_context = dict(recovery_context or {})
        if frozen_validation_spec is not None:
            bounded_recovery_context["frozen_validation_spec"] = (
                frozen_validation_spec.model_dump(mode="json")
            )
        payload: dict[str, Any] = {
            "planning_request": planning_request.model_dump(mode="json"),
            "task_contract": task_contract.model_dump(mode="json"),
            "current_frame": _observation_metadata(observation),
            "recovery_context": bounded_recovery_context,
        }
        result = self.client.invoke(
            agent="Planner",
            schema=PlanResult,
            system_prompt=self.system_prompt,
            text=_json_text(payload),
            frame=observation.image_block,
            frame_paths=[frame_path] if frame_path is not None else (),
            metadata={"observation_id": observation.observation_id},
            structured_method="json_mode",
        )
        result.require_intent(task_contract)
        result.require_request(planning_request)
        if frozen_validation_spec is not None:
            if result.validation_spec is None:
                raise ValueError("recovery plan omitted the frozen validation spec")
            frozen_validation_spec.require_exact_match(result.validation_spec)
        return result


class MonitorReasoner:
    def __init__(
        self,
        client: StructuredAgentClient,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt or load_prompt(
            "monitor/prompt-v1.md"
        )

    def evaluate(
        self,
        request: MonitorRequest,
        *,
        frame_paths: Sequence[Path] = (),
    ) -> MonitorResult:
        observations = [
            request.dispatch_observation,
            *request.recent_observations,
        ]
        visual_blocks: list[dict[str, Any]] = []
        for index, observation in enumerate(observations):
            label = (
                "Dispatch-time observation"
                if index == 0
                else f"Recent observation {index}"
            )
            visual_blocks.extend(
                [
                    {"type": "text", "text": f"{label}:"},
                    observation.image_block,
                ]
            )
        payload = request.model_dump(
            mode="json",
            exclude={
                "dispatch_observation": {"image_block"},
                "recent_observations": {"__all__": {"image_block"}},
            },
        )
        result = self.client.invoke(
            agent="Live Monitor",
            schema=MonitorResult,
            system_prompt=self.system_prompt,
            text=_json_text(payload),
            visual_blocks=visual_blocks,
            frame_paths=frame_paths,
            metadata={
                "dispatch_id": request.dispatch_id,
                "subtask_id": request.subtask_id,
            },
        )
        result.require_current_dispatch(
            dispatch_id=request.dispatch_id,
            subtask_id=request.subtask_id,
        )
        expected_ids = {
            condition.condition_id
            for condition in request.expected_outcome.conditions
        }
        returned_ids = {check.condition_id for check in result.condition_checks}
        if expected_ids != returned_ids or len(returned_ids) != len(
            result.condition_checks
        ):
            raise ValueError(
                "monitor must check each expected condition exactly once"
            )
        return result


class ValidatorReasoner:
    def __init__(
        self,
        client: StructuredAgentClient,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt or load_prompt(
            "validator/prompt-v2.md"
        )

    def validate(
        self,
        request: ValidationRequest,
        *,
        frame_paths: Sequence[Path] = (),
    ) -> ValidationResult:
        visual_blocks: list[dict[str, Any]] = []
        for index, observation in enumerate(request.terminal_observations, start=1):
            visual_blocks.extend(
                [
                    {"type": "text", "text": f"Terminal observation {index}:"},
                    observation.image_block,
                ]
            )
        payload = request.model_dump(
            mode="json",
            exclude={
                "terminal_observations": {"__all__": {"image_block"}},
            },
        )
        result = self.client.invoke(
            agent="Validator",
            schema=ValidationResult,
            system_prompt=self.system_prompt,
            text=_json_text(payload),
            visual_blocks=visual_blocks,
            frame_paths=frame_paths,
            metadata={"spec_id": request.validation_spec.spec_id},
        )
        result.require_exact_spec(request.validation_spec)
        return result
