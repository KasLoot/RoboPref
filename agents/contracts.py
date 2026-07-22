from __future__ import annotations

import copy
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GoalCondition:
    id: str
    description: str
    observable: bool = True
    required: bool = True
    predicate: str | None = None
    arguments: tuple[str, ...] = ()
    evidence_modalities: tuple[str, ...] = ("final_image",)


@dataclass(frozen=True, slots=True)
class ValidationSpec:
    spec_id: str
    confirmed_intent: str
    goal_conditions: tuple[GoalCondition, ...]

    @classmethod
    def from_plan(cls, plan: dict[str, Any], confirmed_intent: str) -> ValidationSpec:
        raw = plan.get("validation_spec")
        if not isinstance(raw, dict):
            raw = {
                "spec_id": f"spec-{uuid.uuid4().hex}",
                "confirmed_intent": confirmed_intent,
                "goal_conditions": plan.get("goal_conditions", []),
            }
        spec_id = str(raw.get("spec_id") or f"spec-{uuid.uuid4().hex}")
        conditions: list[GoalCondition] = []
        for index, item in enumerate(raw.get("goal_conditions", [])):
            if isinstance(item, str):
                description = item.strip()
                if description:
                    conditions.append(
                        GoalCondition(id=f"goal-{index + 1}", description=description)
                    )
            elif isinstance(item, dict):
                description = str(item.get("description") or item.get("condition") or "").strip()
                if description:
                    raw_arguments = item.get("arguments", [])
                    raw_modalities = item.get("evidence_modalities", ["final_image"])
                    if not isinstance(raw_arguments, list) or not isinstance(
                        raw_modalities, list
                    ):
                        raise ValueError(
                            "Validation arguments and evidence_modalities must be lists."
                        )
                    conditions.append(
                        GoalCondition(
                            id=str(item.get("id") or f"goal-{index + 1}"),
                            description=description,
                            observable=bool(item.get("observable", True)),
                            required=bool(item.get("required", True)),
                            predicate=(
                                str(item["predicate"]).strip()
                                if item.get("predicate") is not None
                                else None
                            ),
                            arguments=tuple(map(str, raw_arguments)),
                            evidence_modalities=tuple(map(str, raw_modalities)),
                        )
                    )
        if not conditions:
            raise ValueError("A plan requires at least one observable validation goal.")
        ids = [condition.id for condition in conditions]
        if any(not goal_id.strip() for goal_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("Validation goal IDs must be unique and non-empty.")
        if any(not condition.observable for condition in conditions if condition.required):
            raise ValueError("Every required validation goal must be observable.")
        if not any(condition.required for condition in conditions):
            raise ValueError("A validation specification requires at least one required goal.")
        spec_intent = str(raw.get("confirmed_intent") or confirmed_intent).strip()
        if spec_intent != confirmed_intent.strip():
            raise ValueError("Validation spec intent must exactly match the confirmed task intent.")
        return cls(
            spec_id=spec_id,
            confirmed_intent=spec_intent,
            goal_conditions=tuple(conditions),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanResult:
    status: str
    subtasks: list[dict[str, Any]]
    validation_spec: ValidationSpec | None
    preconditions: list[dict[str, Any]] = field(default_factory=list)
    failure: dict[str, Any] | None = None
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "planning_status": self.status,
            "subtasks": copy.deepcopy(self.subtasks),
            "validation_spec": self.validation_spec.to_dict() if self.validation_spec else None,
            "preconditions": copy.deepcopy(self.preconditions),
            "failure": copy.deepcopy(self.failure),
            "planner_confidence": self.confidence,
        }


@dataclass(slots=True)
class ExecutionResult:
    status: str
    final_observation: str
    subtask_results: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationResult:
    outcome: str
    task_complete: bool
    goal_checks: list[dict[str, Any]]
    discrepancies: list[str]
    confidence: float
    recoverability: str
    user_message: str
    failure: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "task_complete": self.task_complete,
            "goal_checks": copy.deepcopy(self.goal_checks),
            "discrepancies": list(self.discrepancies),
            "validator_confidence": self.confidence,
            "recoverability": self.recoverability,
            "user_message": self.user_message,
            "failure": copy.deepcopy(self.failure),
        }
