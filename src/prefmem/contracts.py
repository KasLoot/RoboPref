"""Typed contracts shared by planning, monitoring, and orchestration.

The planner-facing types deliberately contain no identifiers.  Identifiers are
added by :func:`assign_plan_ids` in trusted host code before a plan can be
published.  Likewise, ``MonitorAssessment.from_model_output`` wraps the small
model-produced JSON object in host-owned plan, revision, task, and timestamp
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any, Mapping, Sequence


class PlanStatus(str, Enum):
    """Result of a planner request."""

    READY = "READY"
    ALREADY_SATISFIED = "ALREADY_SATISFIED"
    BLOCKED = "BLOCKED"


class TaskPhase(str, Enum):
    """The kind of task currently published to the executor and monitor."""

    STEP = "STEP"
    FINAL_VALIDATION = "FINAL_VALIDATION"


class TaskStatus(str, Enum):
    """Per-inference status emitted by the monitor."""

    ONGOING = "ONGOING"
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"


class CriterionState(str, Enum):
    """Visible state of one expected-observation criterion."""

    MET = "MET"
    NOT_MET = "NOT_MET"
    UNKNOWN = "UNKNOWN"


class ValidationStatus(str, Enum):
    """Overall result derived from a frozen final-validation checklist."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"


class FailureKind(str, Enum):
    """Whether a terminal failure was anticipated by the planner."""

    KNOWN = "KNOWN"
    UNEXPECTED = "UNEXPECTED"


class PlannerDecisionType(str, Enum):
    """One stateless decision from the receding-horizon Planner."""

    ACT = "ACT"
    REQUEST_FINAL_VALIDATION = "REQUEST_FINAL_VALIDATION"
    BLOCKED = "BLOCKED"
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"


class PlannerTrigger(str, Enum):
    """Why a fresh planning cycle was requested."""

    CONFIRMED = "CONFIRMED"
    TASK_SUCCESS = "TASK_SUCCESS"
    TASK_FAIL = "TASK_FAIL"
    FINAL_VALIDATION_FAIL = "FINAL_VALIDATION_FAIL"
    USER_REPLAN = "USER_REPLAN"


class ExecutionOutcome(str, Enum):
    """Terminal outcome recorded once for one published task attempt."""

    SUCCESS = "SUCCESS"
    FAIL = "FAIL"
    FINAL_VALIDATION_FAIL = "FINAL_VALIDATION_FAIL"
    INTERRUPTED = "INTERRUPTED"


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _identifier(value: object, field_name: str) -> str:
    result = _text(value, field_name)
    if any(character.isspace() for character in result):
        raise ValueError(f"{field_name} must not contain whitespace")
    return result


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of strings")
    result = tuple(
        _text(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    folded = [item.casefold() for item in result]
    if len(folded) != len(set(folded)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _reject_unknown_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = set(payload) - allowed
    if unknown:
        keys = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"{field_name} contains unknown fields: {keys}")


def _mapping_sequence(value: object, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array")
    return tuple(
        _mapping(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _one_sentence(value: object, field_name: str) -> str:
    result = _text(value, field_name)
    if "\n" in result or "\r" in result:
        raise ValueError(f"{field_name} must be one sentence")
    if len([part for part in _SENTENCE_BREAK.split(result) if part.strip()]) > 1:
        raise ValueError(f"{field_name} must be one sentence")
    return result


@dataclass(frozen=True, slots=True)
class PlannerStep:
    """One human-executable state change proposed by the planner."""

    instruction: str
    expected_observation: tuple[str, ...]
    known_failure_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruction", _text(self.instruction, "instruction"))
        criteria = _string_tuple(
            self.expected_observation,
            "expected_observation",
        )
        if not 1 <= len(criteria) <= 3:
            raise ValueError("expected_observation must contain 1 to 3 criteria")
        object.__setattr__(self, "expected_observation", criteria)
        object.__setattr__(
            self,
            "known_failure_conditions",
            _string_tuple(
                self.known_failure_conditions,
                "known_failure_conditions",
            ),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlannerStep:
        payload = _mapping(payload, "step")
        _reject_unknown_keys(
            payload,
            {
                "instruction",
                "expected_observation",
                "known_failure_conditions",
            },
            "step",
        )
        return cls(
            instruction=payload.get("instruction"),
            expected_observation=payload.get("expected_observation", ()),
            known_failure_conditions=payload.get(
                "known_failure_conditions",
                (),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "expected_observation": list(self.expected_observation),
            "known_failure_conditions": list(self.known_failure_conditions),
        }


@dataclass(frozen=True, slots=True)
class PlannerPlan:
    """Validated, ID-free output expected from the planner model."""

    status: PlanStatus
    steps: tuple[PlannerStep, ...] = ()
    final_expected_observation: tuple[str, ...] = ()
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        try:
            status = PlanStatus(self.status)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid plan status: {self.status!r}") from error
        object.__setattr__(self, "status", status)

        if isinstance(self.steps, (str, bytes)) or not isinstance(
            self.steps,
            Sequence,
        ):
            raise ValueError("steps must be a sequence of PlannerStep objects")
        steps = tuple(self.steps)
        if not all(isinstance(step, PlannerStep) for step in steps):
            raise ValueError("steps must contain only PlannerStep objects")
        object.__setattr__(self, "steps", steps)

        final_criteria = _string_tuple(
            self.final_expected_observation,
            "final_expected_observation",
        )
        object.__setattr__(
            self,
            "final_expected_observation",
            final_criteria,
        )

        blocked_reason = self.blocked_reason
        if blocked_reason is not None:
            blocked_reason = _text(blocked_reason, "blocked_reason")
        object.__setattr__(self, "blocked_reason", blocked_reason)

        if status is PlanStatus.READY:
            if not steps:
                raise ValueError("a READY plan must contain at least one step")
            if not final_criteria:
                raise ValueError(
                    "a READY plan must define final_expected_observation"
                )
            if blocked_reason is not None:
                raise ValueError("a READY plan cannot have blocked_reason")
        elif status is PlanStatus.ALREADY_SATISFIED:
            if steps:
                raise ValueError(
                    "an ALREADY_SATISFIED plan cannot contain steps"
                )
            if not final_criteria:
                raise ValueError(
                    "an ALREADY_SATISFIED plan must describe the final state"
                )
            if blocked_reason is not None:
                raise ValueError(
                    "an ALREADY_SATISFIED plan cannot have blocked_reason"
                )
        else:
            if steps:
                raise ValueError("a BLOCKED plan cannot contain steps")
            if blocked_reason is None:
                raise ValueError("a BLOCKED plan must have blocked_reason")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlannerPlan:
        payload = _mapping(payload, "plan")
        _reject_unknown_keys(
            payload,
            {
                "status",
                "steps",
                "final_expected_observation",
                "blocked_reason",
            },
            "plan",
        )
        step_payloads = _mapping_sequence(payload.get("steps", ()), "steps")
        return cls(
            status=payload.get("status"),
            steps=tuple(PlannerStep.from_dict(step) for step in step_payloads),
            final_expected_observation=payload.get(
                "final_expected_observation",
                (),
            ),
            blocked_reason=payload.get("blocked_reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "steps": [step.to_dict() for step in self.steps],
            "final_expected_observation": list(
                self.final_expected_observation
            ),
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True, slots=True)
class GoalProposal:
    """Nominal, user-facing proposal produced before execution consent.

    The nominal task list explains the intended strategy.  It is deliberately
    non-binding: confirmation freezes the goal and constraints, not this list.
    """

    status: PlanStatus
    goal: str
    final_expected_observation: tuple[str, ...]
    constraints: tuple[str, ...] = ()
    nominal_tasks: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        try:
            status = PlanStatus(self.status)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid goal proposal status: {self.status!r}") from error
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "goal", _text(self.goal, "goal"))
        final = _string_tuple(
            self.final_expected_observation,
            "final_expected_observation",
        )
        if status is not PlanStatus.BLOCKED and not final:
            raise ValueError("a non-blocked proposal needs final expected observations")
        object.__setattr__(self, "final_expected_observation", final)
        object.__setattr__(
            self,
            "constraints",
            _string_tuple(self.constraints, "constraints"),
        )
        nominal = _string_tuple(self.nominal_tasks, "nominal_tasks")
        object.__setattr__(self, "nominal_tasks", nominal)
        reason = self.reason
        if reason is not None:
            reason = _text(reason, "reason")
        object.__setattr__(self, "reason", reason)
        if status is PlanStatus.READY and not nominal:
            raise ValueError("a READY proposal needs at least one nominal task")
        if status is not PlanStatus.READY and nominal:
            raise ValueError(
                f"a {status.value} proposal cannot contain nominal tasks"
            )
        if status is PlanStatus.BLOCKED and reason is None:
            raise ValueError("a BLOCKED proposal needs a reason")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoalProposal:
        payload = _mapping(payload, "goal proposal")
        _reject_unknown_keys(
            payload,
            {
                "status",
                "goal",
                "final_expected_observation",
                "constraints",
                "nominal_tasks",
                "reason",
            },
            "goal proposal",
        )
        return cls(
            status=payload.get("status"),
            goal=payload.get("goal"),
            final_expected_observation=payload.get(
                "final_expected_observation",
                (),
            ),
            constraints=payload.get("constraints", ()),
            nominal_tasks=payload.get("nominal_tasks", ()),
            reason=payload.get("reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "goal": self.goal,
            "final_expected_observation": list(self.final_expected_observation),
            "constraints": list(self.constraints),
            "nominal_tasks": list(self.nominal_tasks),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GoalContract:
    """Exact high-level goal and boundaries authorized by the user."""

    goal_id: str
    revision: int
    goal: str
    final_expected_observation: tuple[str, ...]
    constraints: tuple[str, ...] = ()
    nominal_tasks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise ValueError("revision must be a positive integer")
        if self.revision < 1:
            raise ValueError("revision must be a positive integer")
        object.__setattr__(self, "goal", _text(self.goal, "goal"))
        final = _string_tuple(
            self.final_expected_observation,
            "final_expected_observation",
        )
        if not final:
            raise ValueError("a goal contract needs final expected observations")
        object.__setattr__(self, "final_expected_observation", final)
        object.__setattr__(
            self,
            "constraints",
            _string_tuple(self.constraints, "constraints"),
        )
        object.__setattr__(
            self,
            "nominal_tasks",
            _string_tuple(self.nominal_tasks, "nominal_tasks"),
        )

    @classmethod
    def from_proposal(
        cls,
        proposal: GoalProposal,
        *,
        goal_id: str,
        revision: int = 1,
    ) -> GoalContract:
        if not isinstance(proposal, GoalProposal):
            raise TypeError("proposal must be a GoalProposal")
        if proposal.status is PlanStatus.BLOCKED:
            raise ValueError("a blocked proposal cannot become a goal contract")
        return cls(
            goal_id=goal_id,
            revision=revision,
            goal=proposal.goal,
            final_expected_observation=proposal.final_expected_observation,
            constraints=proposal.constraints,
            nominal_tasks=proposal.nominal_tasks,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoalContract:
        payload = _mapping(payload, "goal contract")
        _reject_unknown_keys(
            payload,
            {
                "goal_id",
                "revision",
                "goal",
                "final_expected_observation",
                "constraints",
                "nominal_tasks",
            },
            "goal contract",
        )
        return cls(
            goal_id=payload.get("goal_id"),
            revision=payload.get("revision"),
            goal=payload.get("goal"),
            final_expected_observation=payload.get(
                "final_expected_observation",
                (),
            ),
            constraints=payload.get("constraints", ()),
            nominal_tasks=payload.get("nominal_tasks", ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "revision": self.revision,
            "goal": self.goal,
            "final_expected_observation": list(self.final_expected_observation),
            "constraints": list(self.constraints),
            "nominal_tasks": list(self.nominal_tasks),
        }


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """One immutable terminal outcome in the current goal session."""

    cycle_id: int
    publication_id: str
    instruction: str
    expected_observation: tuple[str, ...]
    outcome: ExecutionOutcome
    observation: str
    failure_reason: str | None = None
    evidence_frame_sequence: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.cycle_id, bool) or not isinstance(self.cycle_id, int):
            raise ValueError("cycle_id must be a positive integer")
        if self.cycle_id < 1:
            raise ValueError("cycle_id must be a positive integer")
        object.__setattr__(
            self,
            "publication_id",
            _identifier(self.publication_id, "publication_id"),
        )
        object.__setattr__(
            self,
            "instruction",
            _text(self.instruction, "instruction"),
        )
        expected = _string_tuple(
            self.expected_observation,
            "expected_observation",
        )
        if not expected:
            raise ValueError("execution history needs expected observations")
        object.__setattr__(self, "expected_observation", expected)
        try:
            outcome = ExecutionOutcome(self.outcome)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid execution outcome: {self.outcome!r}") from error
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(
            self,
            "observation",
            _one_sentence(self.observation, "observation"),
        )
        failure_reason = self.failure_reason
        if failure_reason is not None:
            failure_reason = _one_sentence(failure_reason, "failure_reason")
        if outcome in {ExecutionOutcome.FAIL, ExecutionOutcome.FINAL_VALIDATION_FAIL}:
            if failure_reason is None:
                raise ValueError("failed execution history needs a failure reason")
        elif failure_reason is not None:
            raise ValueError("only failed execution history can include failure_reason")
        object.__setattr__(self, "failure_reason", failure_reason)
        frame_sequence = self.evidence_frame_sequence
        if frame_sequence is not None and (
            isinstance(frame_sequence, bool)
            or not isinstance(frame_sequence, int)
            or frame_sequence < 0
        ):
            raise ValueError("evidence_frame_sequence must be non-negative")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionRecord:
        payload = _mapping(payload, "execution record")
        _reject_unknown_keys(
            payload,
            {
                "cycle_id",
                "publication_id",
                "instruction",
                "expected_observation",
                "outcome",
                "observation",
                "failure_reason",
                "evidence_frame_sequence",
            },
            "execution record",
        )
        return cls(
            cycle_id=payload.get("cycle_id"),
            publication_id=payload.get("publication_id"),
            instruction=payload.get("instruction"),
            expected_observation=payload.get("expected_observation", ()),
            outcome=payload.get("outcome"),
            observation=payload.get("observation"),
            failure_reason=payload.get("failure_reason"),
            evidence_frame_sequence=payload.get("evidence_frame_sequence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "publication_id": self.publication_id,
            "instruction": self.instruction,
            "expected_observation": list(self.expected_observation),
            "outcome": self.outcome.value,
            "observation": self.observation,
            "failure_reason": self.failure_reason,
            "evidence_frame_sequence": self.evidence_frame_sequence,
        }


@dataclass(frozen=True, slots=True)
class PlannerCycleRequest:
    """Complete context supplied to the stateless Planner on every cycle."""

    goal_contract: GoalContract
    cycle_id: int
    trigger: PlannerTrigger
    execution_history: tuple[ExecutionRecord, ...] = ()
    operator_guidance: str | None = None
    frame_sequence: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.goal_contract, GoalContract):
            raise ValueError("goal_contract must be a GoalContract")
        if isinstance(self.cycle_id, bool) or not isinstance(self.cycle_id, int):
            raise ValueError("cycle_id must be a positive integer")
        if self.cycle_id < 1:
            raise ValueError("cycle_id must be a positive integer")
        try:
            trigger = PlannerTrigger(self.trigger)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid planner trigger: {self.trigger!r}") from error
        object.__setattr__(self, "trigger", trigger)
        history = tuple(self.execution_history)
        if not all(isinstance(item, ExecutionRecord) for item in history):
            raise ValueError("execution_history must contain ExecutionRecord objects")
        publication_ids = [item.publication_id for item in history]
        if len(publication_ids) != len(set(publication_ids)):
            raise ValueError("execution_history publication IDs must be unique")
        object.__setattr__(self, "execution_history", history)
        guidance = self.operator_guidance
        if guidance is not None:
            guidance = _text(guidance, "operator_guidance")
        object.__setattr__(self, "operator_guidance", guidance)
        if self.frame_sequence is not None and (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be non-negative")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlannerCycleRequest:
        payload = _mapping(payload, "planner cycle request")
        _reject_unknown_keys(
            payload,
            {
                "goal_contract",
                "cycle_id",
                "trigger",
                "execution_history",
                "operator_guidance",
                "frame_sequence",
                "request",
            },
            "planner cycle request",
        )
        records = _mapping_sequence(
            payload.get("execution_history", ()),
            "execution_history",
        )
        return cls(
            goal_contract=GoalContract.from_dict(
                _mapping(payload.get("goal_contract"), "goal_contract")
            ),
            cycle_id=payload.get("cycle_id"),
            trigger=payload.get("trigger"),
            execution_history=tuple(
                ExecutionRecord.from_dict(record) for record in records
            ),
            operator_guidance=payload.get("operator_guidance"),
            frame_sequence=payload.get("frame_sequence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_contract": self.goal_contract.to_dict(),
            "cycle_id": self.cycle_id,
            "trigger": self.trigger.value,
            "execution_history": [item.to_dict() for item in self.execution_history],
            "operator_guidance": self.operator_guidance,
            "frame_sequence": self.frame_sequence,
            "request": (
                "Produce the best current candidate task horizon. Only the first "
                "candidate task will be executed before replanning."
            ),
        }


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    """Validated, ID-free output from one receding-horizon Planner cycle."""

    decision: PlannerDecisionType
    candidate_tasks: tuple[PlannerStep, ...] = ()
    reason: str | None = None
    blocked_reason: str | None = None
    user_question: str | None = None

    def __post_init__(self) -> None:
        try:
            decision = PlannerDecisionType(self.decision)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid planner decision: {self.decision!r}") from error
        object.__setattr__(self, "decision", decision)
        candidates = tuple(self.candidate_tasks)
        if not all(isinstance(item, PlannerStep) for item in candidates):
            raise ValueError("candidate_tasks must contain PlannerStep objects")
        if len(candidates) > 3:
            raise ValueError("candidate_tasks may contain at most 3 tasks")
        object.__setattr__(self, "candidate_tasks", candidates)
        for name in ("reason", "blocked_reason", "user_question"):
            value = getattr(self, name)
            if value is not None:
                value = _text(value, name)
            object.__setattr__(self, name, value)
        if decision is PlannerDecisionType.ACT:
            if not candidates:
                raise ValueError("ACT requires at least one candidate task")
            if self.blocked_reason is not None or self.user_question is not None:
                raise ValueError("ACT cannot include blocked_reason or user_question")
        else:
            if candidates:
                raise ValueError(f"{decision.value} cannot include candidate tasks")
            if decision is PlannerDecisionType.REQUEST_FINAL_VALIDATION:
                if self.blocked_reason is not None or self.user_question is not None:
                    raise ValueError(
                        "REQUEST_FINAL_VALIDATION cannot include blocked_reason "
                        "or user_question"
                    )
            elif decision is PlannerDecisionType.BLOCKED:
                if self.blocked_reason is None:
                    raise ValueError("BLOCKED requires blocked_reason")
                if self.user_question is not None:
                    raise ValueError("BLOCKED cannot include user_question")
            else:
                if self.user_question is None:
                    raise ValueError("NEEDS_USER_INPUT requires user_question")
                if self.blocked_reason is not None:
                    raise ValueError("NEEDS_USER_INPUT cannot include blocked_reason")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlannerDecision:
        payload = _mapping(payload, "planner decision")
        _reject_unknown_keys(
            payload,
            {
                "decision",
                "candidate_tasks",
                "reason",
                "blocked_reason",
                "user_question",
            },
            "planner decision",
        )
        tasks = _mapping_sequence(
            payload.get("candidate_tasks", ()),
            "candidate_tasks",
        )
        return cls(
            decision=payload.get("decision"),
            candidate_tasks=tuple(PlannerStep.from_dict(item) for item in tasks),
            reason=payload.get("reason"),
            blocked_reason=payload.get("blocked_reason"),
            user_question=payload.get("user_question"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "candidate_tasks": [item.to_dict() for item in self.candidate_tasks],
            "reason": self.reason,
            "blocked_reason": self.blocked_reason,
            "user_question": self.user_question,
        }


@dataclass(frozen=True, slots=True)
class ValidationChecklistDraftItem:
    """Model-authored detailed visual checks for one broad goal outcome."""

    broad_index: int
    detailed_criteria: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.broad_index, bool)
            or not isinstance(self.broad_index, int)
            or self.broad_index < 0
        ):
            raise ValueError("broad_index must be a non-negative integer")
        criteria = _string_tuple(
            self.detailed_criteria,
            "detailed_criteria",
        )
        if not 1 <= len(criteria) <= 5:
            raise ValueError("detailed_criteria must contain 1 to 5 criteria")
        object.__setattr__(self, "detailed_criteria", criteria)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> ValidationChecklistDraftItem:
        payload = _mapping(payload, "validation checklist draft item")
        _reject_unknown_keys(
            payload,
            {"broad_index", "detailed_criteria"},
            "validation checklist draft item",
        )
        return cls(
            broad_index=payload.get("broad_index"),
            detailed_criteria=payload.get("detailed_criteria", ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "broad_index": self.broad_index,
            "detailed_criteria": list(self.detailed_criteria),
        }


@dataclass(frozen=True, slots=True)
class ValidationChecklistDraft:
    """Strict, ID-free detailed checklist returned by the Validator model."""

    broad_items: tuple[ValidationChecklistDraftItem, ...]

    def __post_init__(self) -> None:
        if isinstance(self.broad_items, (str, bytes)) or not isinstance(
            self.broad_items,
            Sequence,
        ):
            raise ValueError("broad_items must be a sequence")
        items = tuple(self.broad_items)
        if not items or not all(
            isinstance(item, ValidationChecklistDraftItem) for item in items
        ):
            raise ValueError(
                "broad_items must contain ValidationChecklistDraftItem objects"
            )
        indices = [item.broad_index for item in items]
        if len(indices) != len(set(indices)):
            raise ValueError("broad_items must not contain duplicate broad_index values")
        object.__setattr__(self, "broad_items", items)

    @classmethod
    def from_model_output(
        cls,
        payload: Mapping[str, Any],
    ) -> ValidationChecklistDraft:
        payload = _mapping(payload, "validation checklist draft")
        _reject_unknown_keys(
            payload,
            {"broad_items"},
            "validation checklist draft",
        )
        items = _mapping_sequence(
            payload.get("broad_items", ()),
            "broad_items",
        )
        return cls(
            broad_items=tuple(
                ValidationChecklistDraftItem.from_dict(item) for item in items
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {"broad_items": [item.to_dict() for item in self.broad_items]}


@dataclass(frozen=True, slots=True)
class DetailedValidationCriterion:
    """One frozen, host-identified visual criterion."""

    criterion_id: str
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "criterion_id",
            _identifier(self.criterion_id, "criterion_id"),
        )
        object.__setattr__(
            self,
            "description",
            _text(self.description, "description"),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> DetailedValidationCriterion:
        payload = _mapping(payload, "detailed validation criterion")
        _reject_unknown_keys(
            payload,
            {"id", "criterion"},
            "detailed validation criterion",
        )
        return cls(
            criterion_id=payload.get("id"),
            description=payload.get("criterion"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"id": self.criterion_id, "criterion": self.description}


@dataclass(frozen=True, slots=True)
class BroadValidationItem:
    """A frozen broad outcome and its internal detailed visual checks."""

    broad_id: str
    broad_index: int
    label: str
    detailed_criteria: tuple[DetailedValidationCriterion, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "broad_id",
            _identifier(self.broad_id, "broad_id"),
        )
        if (
            isinstance(self.broad_index, bool)
            or not isinstance(self.broad_index, int)
            or self.broad_index < 0
        ):
            raise ValueError("broad_index must be a non-negative integer")
        object.__setattr__(self, "label", _text(self.label, "label"))
        if isinstance(self.detailed_criteria, (str, bytes)) or not isinstance(
            self.detailed_criteria,
            Sequence,
        ):
            raise ValueError("detailed_criteria must be a sequence")
        criteria = tuple(self.detailed_criteria)
        if not 1 <= len(criteria) <= 5 or not all(
            isinstance(item, DetailedValidationCriterion) for item in criteria
        ):
            raise ValueError(
                "detailed_criteria must contain 1 to 5 "
                "DetailedValidationCriterion objects"
            )
        ids = [item.criterion_id for item in criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("detailed criterion IDs must be unique")
        object.__setattr__(self, "detailed_criteria", criteria)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BroadValidationItem:
        payload = _mapping(payload, "broad validation item")
        _reject_unknown_keys(
            payload,
            {"id", "broad_index", "label", "detailed_criteria"},
            "broad validation item",
        )
        criteria = _mapping_sequence(
            payload.get("detailed_criteria", ()),
            "detailed_criteria",
        )
        return cls(
            broad_id=payload.get("id"),
            broad_index=payload.get("broad_index"),
            label=payload.get("label"),
            detailed_criteria=tuple(
                DetailedValidationCriterion.from_dict(item) for item in criteria
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.broad_id,
            "broad_index": self.broad_index,
            "label": self.label,
            "detailed_criteria": [
                item.to_dict() for item in self.detailed_criteria
            ],
        }


@dataclass(frozen=True, slots=True)
class ValidationContract:
    """Frozen final checklist bound to the exact confirmed goal contract."""

    validation_id: str
    goal_contract: GoalContract
    broad_items: tuple[BroadValidationItem, ...]

    def __post_init__(self) -> None:
        validation_id = _identifier(self.validation_id, "validation_id")
        object.__setattr__(self, "validation_id", validation_id)
        if not isinstance(self.goal_contract, GoalContract):
            raise ValueError("goal_contract must be a GoalContract")
        if isinstance(self.broad_items, (str, bytes)) or not isinstance(
            self.broad_items,
            Sequence,
        ):
            raise ValueError("broad_items must be a sequence")
        items = tuple(self.broad_items)
        expected_labels = self.goal_contract.final_expected_observation
        if len(items) != len(expected_labels) or not all(
            isinstance(item, BroadValidationItem) for item in items
        ):
            raise ValueError(
                "broad_items must contain exactly one BroadValidationItem "
                "for every final expected observation"
            )
        all_ids: list[str] = []
        for index, (item, expected_label) in enumerate(
            zip(items, expected_labels, strict=True)
        ):
            if item.broad_index != index:
                raise ValueError("broad_items must be in exact zero-based goal order")
            if item.label != expected_label:
                raise ValueError(
                    "broad item labels must exactly match the frozen goal observations"
                )
            expected_broad_id = f"{validation_id}:b{index + 1}"
            if item.broad_id != expected_broad_id:
                raise ValueError("broad item IDs must be deterministic")
            all_ids.append(item.broad_id)
            for criterion_index, criterion in enumerate(
                item.detailed_criteria,
                start=1,
            ):
                expected_id = f"{expected_broad_id}:c{criterion_index}"
                if criterion.criterion_id != expected_id:
                    raise ValueError("detailed criterion IDs must be deterministic")
                all_ids.append(criterion.criterion_id)
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("validation contract IDs must be unique")
        object.__setattr__(self, "broad_items", items)

    @property
    def detailed_criteria(self) -> tuple[DetailedValidationCriterion, ...]:
        """Return detailed criteria in their mandatory assessment order."""

        return tuple(
            criterion
            for broad_item in self.broad_items
            for criterion in broad_item.detailed_criteria
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ValidationContract:
        payload = _mapping(payload, "validation contract")
        _reject_unknown_keys(
            payload,
            {"validation_id", "goal_contract", "broad_items"},
            "validation contract",
        )
        broad_items = _mapping_sequence(
            payload.get("broad_items", ()),
            "broad_items",
        )
        return cls(
            validation_id=payload.get("validation_id"),
            goal_contract=GoalContract.from_dict(
                _mapping(payload.get("goal_contract"), "goal_contract")
            ),
            broad_items=tuple(
                BroadValidationItem.from_dict(item) for item in broad_items
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "goal_contract": self.goal_contract.to_dict(),
            "broad_items": [item.to_dict() for item in self.broad_items],
        }


def freeze_validation_contract(
    goal_contract: GoalContract,
    draft: ValidationChecklistDraft,
    *,
    validation_id: str,
) -> ValidationContract:
    """Bind a model checklist to host-owned goal labels and deterministic IDs."""

    if not isinstance(goal_contract, GoalContract):
        raise TypeError("goal_contract must be a GoalContract")
    if not isinstance(draft, ValidationChecklistDraft):
        raise TypeError("draft must be a ValidationChecklistDraft")
    safe_validation_id = _identifier(validation_id, "validation_id")
    expected_indices = set(range(len(goal_contract.final_expected_observation)))
    items_by_index = {item.broad_index: item for item in draft.broad_items}
    if set(items_by_index) != expected_indices:
        raise ValueError(
            "checklist draft must map exactly once to every frozen broad outcome"
        )
    broad_items = tuple(
        BroadValidationItem(
            broad_id=f"{safe_validation_id}:b{index + 1}",
            broad_index=index,
            label=label,
            detailed_criteria=tuple(
                DetailedValidationCriterion(
                    criterion_id=(
                        f"{safe_validation_id}:b{index + 1}:c{criterion_index}"
                    ),
                    description=description,
                )
                for criterion_index, description in enumerate(
                    items_by_index[index].detailed_criteria,
                    start=1,
                )
            ),
        )
        for index, label in enumerate(
            goal_contract.final_expected_observation
        )
    )
    return ValidationContract(
        validation_id=safe_validation_id,
        goal_contract=goal_contract,
        broad_items=broad_items,
    )


@dataclass(frozen=True, slots=True)
class ValidationCriterionAssessment:
    """Visible evidence for one frozen detailed validation criterion."""

    criterion_id: str
    state: CriterionState
    evidence: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "criterion_id",
            _identifier(self.criterion_id, "criterion_id"),
        )
        try:
            state = CriterionState(self.state)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid criterion state: {self.state!r}") from error
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "evidence",
            _one_sentence(self.evidence, "evidence"),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> ValidationCriterionAssessment:
        payload = _mapping(payload, "validation criterion assessment")
        _reject_unknown_keys(
            payload,
            {"id", "state", "evidence"},
            "validation criterion assessment",
        )
        return cls(
            criterion_id=payload.get("id"),
            state=payload.get("state"),
            evidence=payload.get("evidence"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.criterion_id,
            "state": self.state.value,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class ValidationAssessment:
    """Validator output wrapped in trusted publication and frame metadata."""

    validation_id: str
    goal_id: str
    goal_revision: int
    publication_id: str
    observed_at: float
    frame_sequence: int | None
    criteria: tuple[ValidationCriterionAssessment, ...]
    observation: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validation_id",
            _identifier(self.validation_id, "validation_id"),
        )
        object.__setattr__(self, "goal_id", _identifier(self.goal_id, "goal_id"))
        if (
            isinstance(self.goal_revision, bool)
            or not isinstance(self.goal_revision, int)
            or self.goal_revision < 1
        ):
            raise ValueError("goal_revision must be a positive integer")
        object.__setattr__(
            self,
            "publication_id",
            _identifier(self.publication_id, "publication_id"),
        )
        if isinstance(self.observed_at, bool) or not isinstance(
            self.observed_at,
            (int, float),
        ):
            raise ValueError("observed_at must be a finite number")
        observed_at = float(self.observed_at)
        if not math.isfinite(observed_at):
            raise ValueError("observed_at must be a finite number")
        object.__setattr__(self, "observed_at", observed_at)
        if self.frame_sequence is not None and (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be non-negative")
        if isinstance(self.criteria, (str, bytes)) or not isinstance(
            self.criteria,
            Sequence,
        ):
            raise ValueError("criteria must be a sequence")
        criteria = tuple(self.criteria)
        if not criteria or not all(
            isinstance(item, ValidationCriterionAssessment) for item in criteria
        ):
            raise ValueError(
                "criteria must contain ValidationCriterionAssessment objects"
            )
        ids = [item.criterion_id for item in criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("validation assessment criterion IDs must be unique")
        object.__setattr__(self, "criteria", criteria)
        object.__setattr__(
            self,
            "observation",
            _one_sentence(self.observation, "observation"),
        )

    @classmethod
    def from_model_output(
        cls,
        contract: ValidationContract,
        payload: Mapping[str, Any],
        *,
        publication_id: str,
        observed_at: float,
        frame_sequence: int | None,
    ) -> ValidationAssessment:
        if not isinstance(contract, ValidationContract):
            raise TypeError("contract must be a ValidationContract")
        payload = _mapping(payload, "validation assessment")
        _reject_unknown_keys(
            payload,
            {"criteria", "observation"},
            "validation assessment",
        )
        criterion_payloads = _mapping_sequence(
            payload.get("criteria", ()),
            "criteria",
        )
        criteria = tuple(
            ValidationCriterionAssessment.from_dict(item)
            for item in criterion_payloads
        )
        expected_ids = tuple(
            item.criterion_id for item in contract.detailed_criteria
        )
        actual_ids = tuple(item.criterion_id for item in criteria)
        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError("validation assessment criterion IDs must be unique")
        if actual_ids != expected_ids:
            raise ValueError(
                "validation assessment criteria must contain the exact frozen "
                "IDs in contract order"
            )
        return cls(
            validation_id=contract.validation_id,
            goal_id=contract.goal_contract.goal_id,
            goal_revision=contract.goal_contract.revision,
            publication_id=publication_id,
            observed_at=observed_at,
            frame_sequence=frame_sequence,
            criteria=criteria,
            observation=payload.get("observation"),
        )

    def to_model_dict(self) -> dict[str, Any]:
        """Return only the model-authored portion of this assessment."""

        return {
            "criteria": [item.to_dict() for item in self.criteria],
            "observation": self.observation,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "goal_id": self.goal_id,
            "goal_revision": self.goal_revision,
            "publication_id": self.publication_id,
            "observed_at": self.observed_at,
            "frame_sequence": self.frame_sequence,
            **self.to_model_dict(),
        }


@dataclass(frozen=True, slots=True)
class BroadValidationResult:
    """One brief user-facing checklist result, without internal criteria."""

    label: str
    state: CriterionState
    evidence: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _text(self.label, "label"))
        try:
            state = CriterionState(self.state)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid broad validation state: {self.state!r}") from error
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "evidence",
            _one_sentence(self.evidence, "evidence"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "state": self.state.value,
            "evidence": self.evidence,
        }


def _evidence_request(label: str) -> str:
    clean_label = label.rstrip(".!?")
    return f"Provide a clear camera view to verify: {clean_label}."


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Deterministically aggregated brief final report for the HRI agent."""

    validation_id: str
    status: ValidationStatus
    checklist: tuple[BroadValidationResult, ...]
    evidence_requests: tuple[str, ...]
    observation: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validation_id",
            _identifier(self.validation_id, "validation_id"),
        )
        try:
            status = ValidationStatus(self.status)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid validation status: {self.status!r}") from error
        object.__setattr__(self, "status", status)
        if isinstance(self.checklist, (str, bytes)) or not isinstance(
            self.checklist,
            Sequence,
        ):
            raise ValueError("checklist must be a sequence")
        checklist = tuple(self.checklist)
        if not checklist or not all(
            isinstance(item, BroadValidationResult) for item in checklist
        ):
            raise ValueError("checklist must contain BroadValidationResult objects")
        labels = [item.label.casefold() for item in checklist]
        if len(labels) != len(set(labels)):
            raise ValueError("checklist labels must be unique")
        derived_status = _derive_validation_status(
            tuple(item.state for item in checklist)
        )
        if status is not derived_status:
            raise ValueError("validation status conflicts with broad checklist states")
        object.__setattr__(self, "checklist", checklist)
        requests = _string_tuple(self.evidence_requests, "evidence_requests")
        expected_requests = tuple(
            _evidence_request(item.label)
            for item in checklist
            if item.state is CriterionState.UNKNOWN
        )
        if requests != expected_requests:
            raise ValueError(
                "evidence_requests must exactly match UNKNOWN broad checklist items"
            )
        object.__setattr__(self, "evidence_requests", requests)
        object.__setattr__(
            self,
            "observation",
            _one_sentence(self.observation, "observation"),
        )

    @property
    def summary(self) -> str:
        total = len(self.checklist)
        met = sum(item.state is CriterionState.MET for item in self.checklist)
        unknown = sum(
            item.state is CriterionState.UNKNOWN for item in self.checklist
        )
        if self.status is ValidationStatus.COMPLETE:
            return f"Task complete — all {total} required outcomes were visually verified."
        if self.status is ValidationStatus.INCOMPLETE:
            return (
                f"Task incomplete — {met} of {total} required outcomes were "
                "visually verified."
            )
        return (
            f"Final validation needs more evidence — {unknown} of {total} "
            "required outcomes could not be verified."
        )

    @classmethod
    def from_assessment(
        cls,
        contract: ValidationContract,
        assessment: ValidationAssessment,
    ) -> ValidationReport:
        if not isinstance(contract, ValidationContract):
            raise TypeError("contract must be a ValidationContract")
        if not isinstance(assessment, ValidationAssessment):
            raise TypeError("assessment must be a ValidationAssessment")
        if (
            assessment.validation_id != contract.validation_id
            or assessment.goal_id != contract.goal_contract.goal_id
            or assessment.goal_revision != contract.goal_contract.revision
        ):
            raise ValueError("assessment does not belong to validation contract")
        expected_ids = tuple(
            item.criterion_id for item in contract.detailed_criteria
        )
        actual_ids = tuple(item.criterion_id for item in assessment.criteria)
        if actual_ids != expected_ids:
            raise ValueError(
                "validation assessment criteria must contain the exact frozen "
                "IDs in contract order"
            )
        assessment_by_id = {
            item.criterion_id: item for item in assessment.criteria
        }
        checklist: list[BroadValidationResult] = []
        for broad_item in contract.broad_items:
            detailed = tuple(
                assessment_by_id[item.criterion_id]
                for item in broad_item.detailed_criteria
            )
            state = _aggregate_criterion_states(
                tuple(item.state for item in detailed)
            )
            if state is CriterionState.NOT_MET:
                evidence = next(
                    item.evidence
                    for item in detailed
                    if item.state is CriterionState.NOT_MET
                )
            elif state is CriterionState.UNKNOWN:
                evidence = next(
                    item.evidence
                    for item in detailed
                    if item.state is CriterionState.UNKNOWN
                )
            elif len(detailed) == 1:
                evidence = detailed[0].evidence
            else:
                evidence = "All detailed visual checks for this outcome were met."
            checklist.append(
                BroadValidationResult(
                    label=broad_item.label,
                    state=state,
                    evidence=evidence,
                )
            )
        checklist_tuple = tuple(checklist)
        return cls(
            validation_id=contract.validation_id,
            status=_derive_validation_status(
                tuple(item.state for item in checklist_tuple)
            ),
            checklist=checklist_tuple,
            evidence_requests=tuple(
                _evidence_request(item.label)
                for item in checklist_tuple
                if item.state is CriterionState.UNKNOWN
            ),
            observation=assessment.observation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "status": self.status.value,
            "summary": self.summary,
            "checklist": [item.to_dict() for item in self.checklist],
            "evidence_requests": list(self.evidence_requests),
            "observation": self.observation,
        }


def _aggregate_criterion_states(
    states: tuple[CriterionState, ...],
) -> CriterionState:
    if CriterionState.NOT_MET in states:
        return CriterionState.NOT_MET
    if CriterionState.UNKNOWN in states:
        return CriterionState.UNKNOWN
    return CriterionState.MET


def _derive_validation_status(
    broad_states: tuple[CriterionState, ...],
) -> ValidationStatus:
    if CriterionState.NOT_MET in broad_states:
        return ValidationStatus.INCOMPLETE
    if CriterionState.UNKNOWN in broad_states:
        return ValidationStatus.NEEDS_EVIDENCE
    return ValidationStatus.COMPLETE


@dataclass(frozen=True, slots=True)
class ObservationCriterion:
    """Host-identified expected observation."""

    criterion_id: str
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "criterion_id",
            _identifier(self.criterion_id, "criterion_id"),
        )
        object.__setattr__(
            self,
            "description",
            _text(self.description, "description"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"id": self.criterion_id, "criterion": self.description}


@dataclass(frozen=True, slots=True)
class ExecutionStep:
    """A planner step after trusted host identifiers have been assigned."""

    step_id: str
    instruction: str
    expected_observation: tuple[ObservationCriterion, ...]
    known_failure_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(self, "instruction", _text(self.instruction, "instruction"))
        criteria = tuple(self.expected_observation)
        if not 1 <= len(criteria) <= 3 or not all(
            isinstance(item, ObservationCriterion) for item in criteria
        ):
            raise ValueError(
                "expected_observation must contain 1 to 3 ObservationCriterion objects"
            )
        criterion_ids = [item.criterion_id for item in criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("expected_observation criterion IDs must be unique")
        object.__setattr__(self, "expected_observation", criteria)
        object.__setattr__(
            self,
            "known_failure_conditions",
            _string_tuple(
                self.known_failure_conditions,
                "known_failure_conditions",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "instruction": self.instruction,
            "expected_observation": [
                criterion.to_dict() for criterion in self.expected_observation
            ],
            "known_failure_conditions": list(self.known_failure_conditions),
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Immutable plan envelope used by the deterministic controller."""

    plan_id: str
    revision: int
    status: PlanStatus
    steps: tuple[ExecutionStep, ...]
    final_expected_observation: tuple[ObservationCriterion, ...]
    final_validation_step_id: str
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise ValueError("revision must be a positive integer")
        if self.revision < 1:
            raise ValueError("revision must be a positive integer")
        try:
            status = PlanStatus(self.status)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid plan status: {self.status!r}") from error
        object.__setattr__(self, "status", status)

        steps = tuple(self.steps)
        if not all(isinstance(step, ExecutionStep) for step in steps):
            raise ValueError("steps must contain only ExecutionStep objects")
        object.__setattr__(self, "steps", steps)
        criteria = tuple(self.final_expected_observation)
        if not all(isinstance(item, ObservationCriterion) for item in criteria):
            raise ValueError(
                "final_expected_observation must contain ObservationCriterion objects"
            )
        ids = [item.criterion_id for item in criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("final criterion IDs must be unique")
        object.__setattr__(self, "final_expected_observation", criteria)
        object.__setattr__(
            self,
            "final_validation_step_id",
            _identifier(
                self.final_validation_step_id,
                "final_validation_step_id",
            ),
        )

        blocked_reason = self.blocked_reason
        if blocked_reason is not None:
            blocked_reason = _text(blocked_reason, "blocked_reason")
        object.__setattr__(self, "blocked_reason", blocked_reason)

        if status is PlanStatus.READY:
            if not steps or not criteria or blocked_reason is not None:
                raise ValueError(
                    "a READY execution plan needs steps and final criteria, and cannot be blocked"
                )
        elif status is PlanStatus.ALREADY_SATISFIED:
            if steps or not criteria or blocked_reason is not None:
                raise ValueError(
                    "an ALREADY_SATISFIED execution plan needs final criteria only"
                )
        elif steps or blocked_reason is None:
            raise ValueError(
                "a BLOCKED execution plan needs a reason and cannot contain steps"
            )

        all_task_ids = [step.step_id for step in steps] + [
            self.final_validation_step_id
        ]
        if len(all_task_ids) != len(set(all_task_ids)):
            raise ValueError("step IDs must be unique within a plan")
        all_criterion_ids = [
            criterion.criterion_id
            for step in steps
            for criterion in step.expected_observation
        ] + ids
        if len(all_criterion_ids) != len(set(all_criterion_ids)):
            raise ValueError("criterion IDs must be unique within a plan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "revision": self.revision,
            "status": self.status.value,
            "steps": [step.to_dict() for step in self.steps],
            "final_expected_observation": [
                criterion.to_dict()
                for criterion in self.final_expected_observation
            ],
            "final_validation_step_id": self.final_validation_step_id,
            "blocked_reason": self.blocked_reason,
        }


def assign_plan_ids(
    plan: PlannerPlan,
    *,
    plan_id: str,
    revision: int = 1,
) -> ExecutionPlan:
    """Create an immutable execution plan with deterministic host-owned IDs."""

    if not isinstance(plan, PlannerPlan):
        raise TypeError("plan must be a PlannerPlan")
    safe_plan_id = _identifier(plan_id, "plan_id")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("revision must be a positive integer")
    prefix = f"{safe_plan_id}:r{revision}"
    steps = tuple(
        ExecutionStep(
            step_id=f"{prefix}:s{step_number}",
            instruction=step.instruction,
            expected_observation=tuple(
                ObservationCriterion(
                    criterion_id=(
                        f"{prefix}:s{step_number}:c{criterion_number}"
                    ),
                    description=description,
                )
                for criterion_number, description in enumerate(
                    step.expected_observation,
                    start=1,
                )
            ),
            known_failure_conditions=step.known_failure_conditions,
        )
        for step_number, step in enumerate(plan.steps, start=1)
    )
    final_step_id = f"{prefix}:final"
    final_criteria = tuple(
        ObservationCriterion(
            criterion_id=f"{final_step_id}:c{criterion_number}",
            description=description,
        )
        for criterion_number, description in enumerate(
            plan.final_expected_observation,
            start=1,
        )
    )
    return ExecutionPlan(
        plan_id=safe_plan_id,
        revision=revision,
        status=plan.status,
        steps=steps,
        final_expected_observation=final_criteria,
        final_validation_step_id=final_step_id,
        blocked_reason=plan.blocked_reason,
    )


@dataclass(frozen=True, slots=True)
class CriterionAssessment:
    """One criterion state returned by a monitor inference."""

    criterion_id: str
    state: CriterionState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "criterion_id",
            _identifier(self.criterion_id, "criterion_id"),
        )
        try:
            state = CriterionState(self.state)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid criterion state: {self.state!r}") from error
        object.__setattr__(self, "state", state)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CriterionAssessment:
        payload = _mapping(payload, "criterion assessment")
        _reject_unknown_keys(
            payload,
            {"id", "state"},
            "criterion assessment",
        )
        return cls(
            criterion_id=payload.get("id"),
            state=payload.get("state"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"id": self.criterion_id, "state": self.state.value}


@dataclass(frozen=True, slots=True)
class FailureReport:
    """Visible evidence of a known or open-world terminal failure."""

    kind: FailureKind
    description: str

    def __post_init__(self) -> None:
        try:
            kind = FailureKind(self.kind)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid failure kind: {self.kind!r}") from error
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "description",
            _one_sentence(self.description, "failure.description"),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FailureReport:
        payload = _mapping(payload, "failure")
        _reject_unknown_keys(
            payload,
            {"kind", "description"},
            "failure",
        )
        return cls(
            kind=payload.get("kind"),
            description=payload.get("description"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "description": self.description}


@dataclass(frozen=True, slots=True)
class MonitorAssessment:
    """Monitor model output plus trusted host identity and capture time.

    ``observed_at`` is the frame-capture time and must use the same monotonic
    clock as ``PublishedTask.published_at``.  Host code, not the model, supplies
    this envelope through :meth:`from_model_output`.
    """

    plan_id: str
    revision: int
    step_id: str
    observed_at: float
    task_status: TaskStatus
    criteria: tuple[CriterionAssessment, ...]
    failure: FailureReport | None
    observation: str
    publication_id: str | None = None
    frame_sequence: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise ValueError("revision must be a positive integer")
        if self.revision < 1:
            raise ValueError("revision must be a positive integer")
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        if isinstance(self.observed_at, bool) or not isinstance(
            self.observed_at,
            (int, float),
        ):
            raise ValueError("observed_at must be a finite number")
        observed_at = float(self.observed_at)
        if not math.isfinite(observed_at):
            raise ValueError("observed_at must be a finite number")
        object.__setattr__(self, "observed_at", observed_at)
        try:
            task_status = TaskStatus(self.task_status)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid task status: {self.task_status!r}") from error
        object.__setattr__(self, "task_status", task_status)

        criteria = tuple(self.criteria)
        if not criteria or not all(
            isinstance(item, CriterionAssessment) for item in criteria
        ):
            raise ValueError(
                "criteria must contain CriterionAssessment objects"
            )
        criterion_ids = [item.criterion_id for item in criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("criterion assessment IDs must be unique")
        object.__setattr__(self, "criteria", criteria)
        object.__setattr__(
            self,
            "observation",
            _one_sentence(self.observation, "observation"),
        )
        publication_id = self.publication_id
        if publication_id is not None:
            publication_id = _identifier(publication_id, "publication_id")
        object.__setattr__(self, "publication_id", publication_id)
        frame_sequence = self.frame_sequence
        if frame_sequence is not None and (
            isinstance(frame_sequence, bool)
            or not isinstance(frame_sequence, int)
            or frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be non-negative")

        if self.failure is not None and not isinstance(
            self.failure,
            FailureReport,
        ):
            raise ValueError("failure must be a FailureReport or None")
        if (
            task_status is TaskStatus.ONGOING
            and self.failure is None
            and all(
                criterion.state is CriterionState.MET
                for criterion in criteria
            )
        ):
            # ``task_status`` is a redundant model summary.  Criterion states
            # are the auditable evidence, so an all-MET non-failure result is
            # canonical SUCCESS even if a cautious model emitted ONGOING.
            task_status = TaskStatus.SUCCESS
            object.__setattr__(self, "task_status", task_status)
        if task_status is TaskStatus.SUCCESS:
            if self.failure is not None:
                raise ValueError("SUCCESS cannot include a failure report")
            if any(
                criterion.state is not CriterionState.MET
                for criterion in criteria
            ):
                raise ValueError("SUCCESS requires every reported criterion to be MET")
        elif task_status is TaskStatus.FAIL:
            if self.failure is None:
                raise ValueError("FAIL requires a failure report")
        elif self.failure is not None:
            raise ValueError("ONGOING cannot include a failure report")

    @classmethod
    def from_model_output(
        cls,
        payload: Mapping[str, Any],
        *,
        plan_id: str,
        revision: int,
        step_id: str,
        observed_at: float,
        publication_id: str | None = None,
        frame_sequence: int | None = None,
    ) -> MonitorAssessment:
        """Validate model JSON while adding trusted host metadata."""

        payload = _mapping(payload, "monitor output")
        _reject_unknown_keys(
            payload,
            {"task_status", "criteria", "failure", "observation"},
            "monitor output",
        )
        criterion_payloads = _mapping_sequence(
            payload.get("criteria", ()),
            "criteria",
        )
        failure_payload = payload.get("failure")
        return cls(
            plan_id=plan_id,
            revision=revision,
            step_id=step_id,
            observed_at=observed_at,
            task_status=payload.get("task_status"),
            criteria=tuple(
                CriterionAssessment.from_dict(criterion)
                for criterion in criterion_payloads
            ),
            failure=(
                None
                if failure_payload is None
                else FailureReport.from_dict(
                    _mapping(failure_payload, "failure")
                )
            ),
            observation=payload.get("observation"),
            publication_id=publication_id,
            frame_sequence=frame_sequence,
        )

    def to_model_dict(self) -> dict[str, Any]:
        """Return only the compact portion generated by the monitor model."""

        return {
            "task_status": self.task_status.value,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "failure": None if self.failure is None else self.failure.to_dict(),
            "observation": self.observation,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "revision": self.revision,
            "step_id": self.step_id,
            "observed_at": self.observed_at,
            "publication_id": self.publication_id,
            "frame_sequence": self.frame_sequence,
            **self.to_model_dict(),
        }


@dataclass(frozen=True, slots=True)
class PublishedTask:
    """Exact task sent to both the human executor and monitor.

    ``published_at`` is a monotonic-clock timestamp, not wall-clock time.
    """

    plan_id: str
    revision: int
    step_id: str
    phase: TaskPhase
    instruction: str
    expected_observation: tuple[ObservationCriterion, ...]
    known_failure_conditions: tuple[str, ...]
    published_at: float
    publication_id: str | None = None
    cycle_id: int | None = None
    frame_sequence: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise ValueError("revision must be a positive integer")
        if self.revision < 1:
            raise ValueError("revision must be a positive integer")
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        try:
            phase = TaskPhase(self.phase)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid task phase: {self.phase!r}") from error
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "instruction", _text(self.instruction, "instruction"))
        criteria = tuple(self.expected_observation)
        if not criteria or not all(
            isinstance(item, ObservationCriterion) for item in criteria
        ):
            raise ValueError(
                "expected_observation must contain ObservationCriterion objects"
            )
        object.__setattr__(self, "expected_observation", criteria)
        object.__setattr__(
            self,
            "known_failure_conditions",
            _string_tuple(
                self.known_failure_conditions,
                "known_failure_conditions",
            ),
        )
        if isinstance(self.published_at, bool) or not isinstance(
            self.published_at,
            (int, float),
        ):
            raise ValueError("published_at must be a finite number")
        timestamp = float(self.published_at)
        if not math.isfinite(timestamp):
            raise ValueError("published_at must be a finite number")
        object.__setattr__(self, "published_at", timestamp)
        publication_id = self.publication_id
        if publication_id is None:
            publication_id = f"{self.step_id}:p{timestamp:g}"
        object.__setattr__(
            self,
            "publication_id",
            _identifier(publication_id, "publication_id"),
        )
        cycle_id = self.cycle_id
        if cycle_id is not None and (
            isinstance(cycle_id, bool)
            or not isinstance(cycle_id, int)
            or cycle_id < 1
        ):
            raise ValueError("cycle_id must be a positive integer")
        frame_sequence = self.frame_sequence
        if frame_sequence is not None and (
            isinstance(frame_sequence, bool)
            or not isinstance(frame_sequence, int)
            or frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "revision": self.revision,
            "step_id": self.step_id,
            "phase": self.phase.value,
            "instruction": self.instruction,
            "expected_observation": [
                criterion.to_dict() for criterion in self.expected_observation
            ],
            "known_failure_conditions": list(self.known_failure_conditions),
            "published_at": self.published_at,
            "publication_id": self.publication_id,
            "cycle_id": self.cycle_id,
            "frame_sequence": self.frame_sequence,
        }
