from __future__ import annotations

import copy
import hashlib
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


STRUCTURED_PREDICATE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


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
        raw_conditions = raw.get("goal_conditions", [])
        if not isinstance(raw_conditions, list):
            raise ValueError("Validation goal_conditions must be a list.")
        conditions: list[GoalCondition] = []
        for index, item in enumerate(raw_conditions):
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
                    if any(
                        not isinstance(argument, str) or not argument.strip()
                        for argument in raw_arguments
                    ):
                        raise ValueError(
                            "Validation predicate arguments must be non-empty strings."
                        )
                    if any(
                        not isinstance(modality, str) or not modality.strip()
                        for modality in raw_modalities
                    ):
                        raise ValueError(
                            "Validation evidence modalities must be non-empty strings."
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
                            arguments=tuple(
                                argument.strip() for argument in raw_arguments
                            ),
                            evidence_modalities=tuple(
                                modality.strip() for modality in raw_modalities
                            ),
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
        for condition in conditions:
            if (
                condition.predicate is None
                or not STRUCTURED_PREDICATE.fullmatch(condition.predicate)
            ):
                raise ValueError(
                    "Every validation goal requires a structured predicate name."
                )
            if not condition.arguments:
                raise ValueError(
                    "Every validation goal requires at least one semantic predicate argument."
                )
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

    def to_model_dict(self) -> dict[str, Any]:
        """Return execution evidence without a model-visible filesystem path."""

        try:
            observation_bytes = Path(self.final_observation).read_bytes()
        except (OSError, ValueError):
            # Do not hash a path/URI: benchmark packet paths are correlated with
            # hidden scenario labels. The status-only fallback remains opaque.
            observation_bytes = (
                f"unavailable-observation:{self.status}".encode("utf-8")
            )
        observation_id = (
            "obs-" + hashlib.sha256(observation_bytes).hexdigest()[:24]
        )
        return {
            "status": self.status,
            "final_observation_id": observation_id,
            "subtask_results": copy.deepcopy(self.subtask_results),
            "evidence": copy.deepcopy(self.evidence),
            "error": self.error,
        }


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
