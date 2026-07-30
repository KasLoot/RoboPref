"""Typed contracts shared by PrefMem's agents and deterministic controller.

The language models may propose values for these contracts, but only host code
is allowed to advance controller state, publish an execution command, mutate
memory, or declare a task complete.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


_PREDICATE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_COVERAGE_TOKEN = re.compile(r"[A-Za-z0-9]+")
_COVERAGE_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "be",
        "by",
        "do",
        "does",
        "ensure",
        "for",
        "from",
        "in",
        "into",
        "is",
        "keep",
        "make",
        "move",
        "must",
        "of",
        "on",
        "or",
        "place",
        "put",
        "should",
        "the",
        "tidy",
        "to",
        "with",
    }
)


def utc_now() -> datetime:
    """Return a timezone-aware timestamp suitable for persisted records."""

    return datetime.now(timezone.utc)


def _non_empty(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _coverage_tokens(value: Any) -> set[str]:
    """Return conservative lexical units for deterministic spec coverage."""

    tokens: set[str] = set()
    for raw in _COVERAGE_TOKEN.findall(str(value).casefold()):
        if raw in _COVERAGE_STOP_WORDS:
            continue
        # A tiny plural normalization keeps entity checks deterministic while
        # allowing contracts such as "books" to match a predicate argument
        # named "book".  Avoid stemming words ending in "ss".
        token = (
            raw[:-1]
            if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss")
            else raw
        )
        tokens.add(token)
    return tokens


def _parameter_values(value: Any) -> list[Any]:
    """Flatten parameter leaves without treating schema keys as task entities."""

    if isinstance(value, Mapping):
        leaves: list[Any] = []
        for nested in value.values():
            leaves.extend(_parameter_values(nested))
        return leaves
    if isinstance(value, (list, tuple, set, frozenset)):
        leaves = []
        for nested in value:
            leaves.extend(_parameter_values(nested))
        return leaves
    if value is None or isinstance(value, bool):
        return []
    return [value]


class ContractModel(BaseModel):
    """Base class for strict agent/controller boundary objects."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )


# ---------------------------------------------------------------------------
# Persistent memory


class PreferenceStatus(str, Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"


class PreferenceRecord(ContractModel):
    """One authoritative revision of an explicitly approved preference."""

    id: str
    username: str
    statement: str
    scope: str = "contextual"
    applicability: dict[str, Any] = Field(default_factory=dict)
    structured_value: Any = None
    status: PreferenceStatus = PreferenceStatus.ACTIVE
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    evidence: list[dict[str, Any] | str] = Field(default_factory=list)
    revision: int = Field(default=1, ge=1)
    supersedes_id: str | None = None

    @field_validator("id", "username", "statement", "scope")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("created_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("memory timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_revision_time(self) -> "PreferenceRecord":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")
        if self.supersedes_id == self.id:
            raise ValueError("a preference revision cannot supersede itself")
        return self


class EpisodeRecord(ContractModel):
    """Immutable summary of one terminal or interrupted task episode."""

    id: str
    username: str
    request: str
    resolved_task: dict[str, Any] = Field(default_factory=dict)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    execution: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | str = Field(default_factory=dict)
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utc_now)

    @field_validator("id", "username", "request")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("summary")
    @classmethod
    def normalize_summary(cls, value: str) -> str:
        return value.strip()

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return list(dict.fromkeys(normalized))

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("episode timestamp must be timezone-aware")
        return value


class MemoryStatus(str, Enum):
    NOT_RETRIEVED = "NOT_RETRIEVED"
    AVAILABLE = "AVAILABLE"
    EMPTY = "EMPTY"
    UNAVAILABLE = "UNAVAILABLE"


class MemoryType(str, Enum):
    PERSISTENT_PREFERENCE_MEMORY = "PERSISTENT_PREFERENCE_MEMORY"
    PERSISTENT_HISTORY_MEMORY = "PERSISTENT_HISTORY_MEMORY"


class MemoryRetrievalRequest(ContractModel):
    """Semantic retrieval request proposed by HRI and executed by the host."""

    request_id: str | None = None
    memory_types: list[MemoryType] = Field(
        default_factory=lambda: [
            MemoryType.PERSISTENT_PREFERENCE_MEMORY,
            MemoryType.PERSISTENT_HISTORY_MEMORY,
        ]
    )
    search_text: str
    task_hint: str | None = None
    scene_entities: list[str] = Field(default_factory=list)
    reason_code: str
    limit: int = Field(default=5, ge=1, le=50)

    @field_validator("search_text", "reason_code")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @field_validator("scene_entities")
    @classmethod
    def normalize_entities(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return list(dict.fromkeys(normalized))

    @field_validator("memory_types")
    @classmethod
    def require_memory_type(cls, values: list[MemoryType]) -> list[MemoryType]:
        if not values:
            raise ValueError("memory_types must contain at least one memory type")
        return list(dict.fromkeys(values))


class MemoryContext(ContractModel):
    """Bounded, model-visible retrieval result; embeddings never appear here."""

    status: MemoryStatus = MemoryStatus.NOT_RETRIEVED
    request_id: str | None = None
    relevant_preferences: list[dict[str, Any]] = Field(default_factory=list)
    relevant_history: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status(self) -> "MemoryContext":
        has_records = bool(
            self.relevant_preferences
            or self.relevant_history
            or self.conflicts
        )
        if self.status == MemoryStatus.AVAILABLE and not has_records:
            raise ValueError("AVAILABLE memory context requires at least one record")
        if self.status != MemoryStatus.AVAILABLE and has_records:
            raise ValueError(
                f"{self.status.value} memory context cannot contain records"
            )
        return self


# ---------------------------------------------------------------------------
# HRI routing


class HRIDecisionKind(str, Enum):
    RETRIEVE_MEMORY = "RETRIEVE_MEMORY"
    ASK_USER = "ASK_USER"
    SUBMIT_TASK = "SUBMIT_TASK"
    RESPOND = "RESPOND"


class InteractionKind(str, Enum):
    TASK_CLARIFICATION = "TASK_CLARIFICATION"
    TASK_CONFIRMATION = "TASK_CONFIRMATION"
    MEMORY_CONSENT = "MEMORY_CONSENT"


class HRIInteraction(ContractModel):
    kind: InteractionKind
    unresolved_fields: list[str] = Field(default_factory=list)
    proposed_value: Any = None


class MemoryActionKind(str, Enum):
    NONE = "NONE"
    REMEMBER = "REMEMBER"
    UPDATE = "UPDATE"
    FORGET = "FORGET"


class HRIMemoryAction(ContractModel):
    action: MemoryActionKind = MemoryActionKind.NONE
    statement: str | None = None
    scope: dict[str, Any] = Field(default_factory=dict)
    structured_value: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_action(self) -> "HRIMemoryAction":
        if self.action != MemoryActionKind.NONE and (
            self.statement is None or not self.statement.strip()
        ):
            raise ValueError(f"{self.action.value} requires a preference statement")
        if self.action == MemoryActionKind.NONE and self.statement is not None:
            raise ValueError("NONE memory action cannot contain a statement")
        return self


class TaskContract(ContractModel):
    """Resolved task semantics passed from HRI to Planner."""

    task_type: str
    confirmed_intent: str
    objects: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    preference_refs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @field_validator("task_type", "confirmed_intent")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)


class HRIDecision(ContractModel):
    """A host-routed HRI decision; this is deliberately not a tool call."""

    decision: HRIDecisionKind
    reply_to_user: str | None = None
    memory_request: MemoryRetrievalRequest | None = None
    interaction: HRIInteraction | None = None
    task_contract: TaskContract | None = None
    memory_action: HRIMemoryAction = Field(default_factory=HRIMemoryAction)
    memory_refs_used: list[str] = Field(default_factory=list)
    reason_code: str

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str) -> str:
        return _non_empty(value, "reason_code")

    @model_validator(mode="after")
    def validate_payload_for_decision(self) -> "HRIDecision":
        if self.decision == HRIDecisionKind.RETRIEVE_MEMORY:
            if self.memory_request is None:
                raise ValueError("RETRIEVE_MEMORY requires memory_request")
            if any(
                value is not None
                for value in (
                    self.reply_to_user,
                    self.interaction,
                    self.task_contract,
                )
            ):
                raise ValueError(
                    "RETRIEVE_MEMORY is an internal route and cannot reply or submit a task"
                )
        elif self.decision == HRIDecisionKind.ASK_USER:
            if self.interaction is None:
                raise ValueError("ASK_USER requires interaction")
            if self.reply_to_user is None or not self.reply_to_user.strip():
                raise ValueError("ASK_USER requires reply_to_user")
            if self.memory_request is not None or self.task_contract is not None:
                raise ValueError("ASK_USER cannot retrieve memory or submit a task")
        elif self.decision == HRIDecisionKind.SUBMIT_TASK:
            if self.task_contract is None:
                raise ValueError("SUBMIT_TASK requires task_contract")
            if self.reply_to_user is None or not self.reply_to_user.strip():
                raise ValueError("SUBMIT_TASK requires reply_to_user")
            if self.memory_request is not None or self.interaction is not None:
                raise ValueError("SUBMIT_TASK cannot retrieve memory or ask a question")
        elif self.decision == HRIDecisionKind.RESPOND:
            if self.reply_to_user is None or not self.reply_to_user.strip():
                raise ValueError("RESPOND requires reply_to_user")
            if any(
                value is not None
                for value in (
                    self.memory_request,
                    self.interaction,
                    self.task_contract,
                )
            ):
                raise ValueError("RESPOND cannot contain another decision payload")
        if (
            self.memory_action.action != MemoryActionKind.NONE
            and self.decision == HRIDecisionKind.RETRIEVE_MEMORY
        ):
            raise ValueError("memory retrieval cannot also request a mutation")
        return self

    @property
    def user_text(self) -> str | None:
        return self.reply_to_user


# ---------------------------------------------------------------------------
# Planner and frozen validation specification


class PlanningStatus(str, Enum):
    READY = "READY"
    ALREADY_SATISFIED = "ALREADY_SATISFIED"
    BLOCKED = "BLOCKED"
    UNSUPPORTED = "UNSUPPORTED"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


class ReplanPolicy(str, Enum):
    EVERY_SUBTASK = "EVERY_SUBTASK"
    ON_DEVIATION = "ON_DEVIATION"


class PlanningRequest(ContractModel):
    """Controller-owned planning identity that Planner must echo unchanged."""

    plan_id: str
    plan_version: int = Field(ge=1)
    planning_mode: ReplanPolicy
    validation_spec_id: str
    horizon_length: int | None = Field(default=None, ge=1)

    @field_validator("plan_id", "validation_spec_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)


class RecoveryAction(str, Enum):
    NONE = "NONE"
    CONTINUE = "CONTINUE"
    REOBSERVE = "REOBSERVE"
    ADVANCE = "ADVANCE"
    AUTO_LOCAL = "AUTO_LOCAL"
    REPLAN = "REPLAN"
    USER_ASSIST = "USER_ASSIST"
    ABORT_SAFETY = "ABORT_SAFETY"


class PlannerRecoverability(str, Enum):
    REOBSERVE = "REOBSERVE"
    REPLAN = "REPLAN"
    USER_ASSIST = "USER_ASSIST"
    ABORT_SAFETY = "ABORT_SAFETY"


class FailureDetail(ContractModel):
    stage: str
    code: str
    expected: Any = None
    observed: Any = None
    recoverability: PlannerRecoverability = PlannerRecoverability.USER_ASSIST
    user_message: str

    @field_validator("stage", "code", "user_message")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)


class PredicateCondition(ContractModel):
    goal_id: str
    description: str
    observable: bool = True
    required: bool = True
    predicate: str
    arguments: list[str]
    evidence_modalities: list[str] = Field(default_factory=lambda: ["final_image"])

    @field_validator("goal_id", "description", "predicate")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        value = _non_empty(value, info.field_name)
        if info.field_name == "predicate" and not _PREDICATE.fullmatch(value):
            raise ValueError("predicate must be a relation name without arguments")
        return value

    @field_validator("arguments", "evidence_modalities")
    @classmethod
    def validate_non_empty_list(cls, values: list[str], info) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        if not normalized:
            raise ValueError(f"{info.field_name} must contain a non-empty value")
        return normalized

    @property
    def id(self) -> str:
        """Compatibility accessor for earlier controller code."""

        return self.goal_id


class ValidationSpec(ContractModel):
    spec_id: str
    confirmed_intent: str
    goal_conditions: list[PredicateCondition]

    @field_validator("spec_id", "confirmed_intent")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_goals(self) -> "ValidationSpec":
        if not self.goal_conditions:
            raise ValueError("validation spec requires goal_conditions")
        goal_ids = [goal.goal_id for goal in self.goal_conditions]
        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError("validation goal IDs must be unique")
        if not any(goal.required for goal in self.goal_conditions):
            raise ValueError("validation spec requires at least one required goal")
        unobservable_required = [
            goal.goal_id
            for goal in self.goal_conditions
            if goal.required and not goal.observable
        ]
        if unobservable_required:
            raise ValueError(
                "required validation goals must be observable: "
                + repr(unobservable_required)
            )
        return self

    def canonical_hash(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def require_exact_match(self, other: "ValidationSpec") -> None:
        if self.canonical_hash() != other.canonical_hash():
            raise ValueError("recovery plan changed the frozen validation spec")

    def require_task_coverage(self, task_contract: TaskContract) -> None:
        """Require every resolved task entity/detail in a completion-gating goal.

        This is deliberately lexical rather than semantic.  It cannot prove
        that a predicate is correct, but it prevents a planner from silently
        dropping an object, constraint, or concrete parameter value from the
        frozen validation boundary.
        """

        required_goal_tokens: set[str] = set()
        for goal in self.goal_conditions:
            if not goal.required:
                continue
            required_goal_tokens.update(_coverage_tokens(goal.description))
            required_goal_tokens.update(_coverage_tokens(goal.predicate))
            for argument in goal.arguments:
                required_goal_tokens.update(_coverage_tokens(argument))

        coverage_inputs: list[tuple[str, Any]] = [
            *(("object", item) for item in task_contract.objects),
            *(("constraint", item) for item in task_contract.constraints),
            *(
                ("parameter value", item)
                for item in _parameter_values(task_contract.parameters)
            ),
        ]
        missing: list[str] = []
        for kind, value in coverage_inputs:
            tokens = _coverage_tokens(value)
            if tokens and not tokens.issubset(required_goal_tokens):
                missing.append(f"{kind} {value!r}")
        if missing:
            raise ValueError(
                "validation spec required goals do not cover task contract: "
                + ", ".join(missing)
            )


class ExpectedCondition(ContractModel):
    condition_id: str
    description: str
    predicate: str
    arguments: list[str]
    required: bool = True
    observable: bool = True

    @field_validator("condition_id", "description", "predicate")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        value = _non_empty(value, info.field_name)
        if info.field_name == "predicate" and not _PREDICATE.fullmatch(value):
            raise ValueError("predicate must be a relation name without arguments")
        return value

    @field_validator("arguments")
    @classmethod
    def validate_arguments(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        if not normalized:
            raise ValueError("arguments must contain a non-empty value")
        return normalized


class ExpectedOutcome(ContractModel):
    conditions: list[ExpectedCondition]
    failure_conditions: list[str] = Field(default_factory=list)
    progress_cues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_conditions(self) -> "ExpectedOutcome":
        if not self.conditions:
            raise ValueError("expected_outcome requires at least one condition")
        condition_ids = [condition.condition_id for condition in self.conditions]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("expected-outcome condition IDs must be unique")
        if not any(condition.required for condition in self.conditions):
            raise ValueError("expected_outcome requires a required condition")
        return self


class TimeoutPolicy(ContractModel):
    timeout_seconds: float = Field(gt=0)
    stall_seconds: float = Field(gt=0)
    on_timeout: Literal["REPLAN", "USER_ASSIST"]
    on_stall: Literal["REOBSERVE", "REPLAN", "USER_ASSIST"]

    @model_validator(mode="after")
    def validate_intervals(self) -> "TimeoutPolicy":
        if self.stall_seconds > self.timeout_seconds:
            raise ValueError("stall_seconds must not exceed timeout_seconds")
        return self


class Subtask(ContractModel):
    subtask_id: str
    task_instruction: str
    expected_outcome: ExpectedOutcome
    timeout_policy: TimeoutPolicy
    # Compatibility fields used by the current Planner prompt.
    target: str | None = None
    source: str | None = None
    destination: str | None = None
    arm: Literal["left", "right"] | None = None

    @field_validator("subtask_id", "task_instruction")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)


class Precondition(ContractModel):
    condition_id: str
    description: str
    satisfied: bool
    evidence: str


class PlanResult(ContractModel):
    planning_status: PlanningStatus
    plan_id: str
    plan_version: int = Field(ge=1)
    planning_mode: ReplanPolicy
    preconditions: list[Precondition] = Field(default_factory=list)
    subtasks: list[Subtask] = Field(default_factory=list)
    validation_spec: ValidationSpec | None = None
    failure: FailureDetail | None = None
    planner_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, value: str) -> str:
        return _non_empty(value, "plan_id")

    @model_validator(mode="after")
    def validate_status_payload(self) -> "PlanResult":
        subtask_ids = [subtask.subtask_id for subtask in self.subtasks]
        if len(subtask_ids) != len(set(subtask_ids)):
            raise ValueError("Planner subtask IDs must be unique")

        if self.planning_status == PlanningStatus.READY:
            if not self.subtasks:
                raise ValueError("READY plan requires at least one subtask")
            if self.validation_spec is None:
                raise ValueError("READY plan requires validation_spec")
            if self.failure is not None:
                raise ValueError("READY plan cannot include failure")
            unsatisfied = [
                precondition.condition_id
                for precondition in self.preconditions
                if not precondition.satisfied
            ]
            if unsatisfied:
                raise ValueError(
                    "READY plan cannot contain unsatisfied preconditions: "
                    + ", ".join(unsatisfied)
                )
            if self.planner_confidence < 0.8:
                raise ValueError(
                    "READY plan requires planner_confidence >= 0.8"
                )
        elif self.planning_status == PlanningStatus.ALREADY_SATISFIED:
            if self.subtasks:
                raise ValueError("ALREADY_SATISFIED plan cannot include subtasks")
            if self.validation_spec is None:
                raise ValueError("ALREADY_SATISFIED requires validation_spec")
            if self.failure is not None:
                raise ValueError("ALREADY_SATISFIED cannot include failure")
            if self.planner_confidence < 0.8:
                raise ValueError(
                    "ALREADY_SATISFIED requires planner_confidence >= 0.8"
                )
        else:
            if self.subtasks:
                raise ValueError("non-executable plan cannot include subtasks")
            if self.failure is None:
                raise ValueError("non-executable plan requires failure")
        return self

    def require_intent(self, task_contract: TaskContract) -> None:
        if (
            self.validation_spec is not None
            and self.validation_spec.confirmed_intent
            != task_contract.confirmed_intent
        ):
            raise ValueError(
                "validation spec intent must exactly match the task contract"
            )
        if (
            self.validation_spec is not None
            and self.planning_status
            in {PlanningStatus.READY, PlanningStatus.ALREADY_SATISFIED}
        ):
            self.validation_spec.require_task_coverage(task_contract)

    def require_request(self, request: PlanningRequest) -> None:
        if (
            self.plan_id != request.plan_id
            or self.plan_version != request.plan_version
            or self.planning_mode != request.planning_mode
        ):
            raise ValueError(
                "Planner did not copy controller-owned plan identity exactly"
            )
        if (
            self.validation_spec is not None
            and self.validation_spec.spec_id != request.validation_spec_id
        ):
            raise ValueError(
                "Planner did not copy the controller-owned validation spec ID"
            )


# ---------------------------------------------------------------------------
# Observation, publication, live monitoring, and final validation


class Observation(ContractModel):
    observation_id: str
    captured_at: datetime = Field(default_factory=utc_now)
    sequence: int | None = Field(default=None, ge=0)
    image_block: dict[str, Any]
    content_hash: str | None = None

    @field_validator("observation_id")
    @classmethod
    def validate_observation_id(cls, value: str) -> str:
        return _non_empty(value, "observation_id")

    @field_validator("captured_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observation timestamp must be timezone-aware")
        return value


class ExecutionCommand(ContractModel):
    """The complete information boundary visible to an execution adapter."""

    dispatch_id: str
    subtask_id: str
    task_instruction: str
    observation_id: str
    current_frame: dict[str, Any]

    @field_validator(
        "dispatch_id",
        "subtask_id",
        "task_instruction",
        "observation_id",
    )
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)


class PublicationReceipt(ContractModel):
    dispatch_id: str
    published_at: datetime = Field(default_factory=utc_now)
    published: bool = True
    duplicate_suppressed: bool = False

    @field_validator("published_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("publication timestamp must be timezone-aware")
        return value


class TaskStatus(str, Enum):
    ON_GOING = "ON_GOING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class MonitorProgress(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    ADVANCING = "ADVANCING"
    VERIFYING = "VERIFYING"
    DEVIATED = "DEVIATED"
    STALLED = "STALLED"


class ObservationQuality(str, Enum):
    ADEQUATE = "ADEQUATE"
    BLURRED = "BLURRED"
    OCCLUDED = "OCCLUDED"
    STALE = "STALE"
    INSUFFICIENT = "INSUFFICIENT"


class SafetyStatus(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"
    NOT_EVALUATED = "NOT_EVALUATED"


class MonitorSafetyStatus(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


class FailureKind(str, Enum):
    DEVIATION = "DEVIATION"
    STALLED = "STALLED"
    BLOCKED = "BLOCKED"
    OBJECT_LOST = "OBJECT_LOST"
    SAFETY = "SAFETY"


class ConditionState(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNKNOWN = "UNKNOWN"


class MonitorAction(str, Enum):
    CONTINUE = "CONTINUE"
    REOBSERVE = "REOBSERVE"
    ADVANCE = "ADVANCE"
    REPLAN = "REPLAN"
    USER_ASSIST = "USER_ASSIST"
    ABORT_SAFETY = "ABORT_SAFETY"


class ConditionCheck(ContractModel):
    condition_id: str
    state: ConditionState
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("condition_id", "evidence")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)


class MonitorRequest(ContractModel):
    dispatch_id: str
    subtask_id: str
    task_instruction: str
    expected_outcome: ExpectedOutcome
    dispatch_observation: Observation
    recent_observations: list[Observation]
    elapsed_seconds: float = Field(ge=0)
    previous_result: dict[str, Any] | None = None

    @field_validator("dispatch_id", "subtask_id", "task_instruction")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_observation_window(self) -> "MonitorRequest":
        if not self.recent_observations:
            raise ValueError("monitor requires at least one recent observation")
        return self


class MonitorResult(ContractModel):
    dispatch_id: str
    subtask_id: str
    task_status: TaskStatus
    progress: MonitorProgress
    observation_quality: ObservationQuality
    safety_status: MonitorSafetyStatus
    failure_kind: FailureKind | None = None
    recommended_action: MonitorAction
    condition_checks: list[ConditionCheck]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_consistency(self) -> "MonitorResult":
        if self.task_status == TaskStatus.SUCCESS:
            if self.recommended_action != MonitorAction.ADVANCE:
                raise ValueError("SUCCESS monitor result must recommend ADVANCE")
            if self.failure_kind is not None:
                raise ValueError("SUCCESS monitor result cannot contain failure_kind")
            if self.observation_quality != ObservationQuality.ADEQUATE:
                raise ValueError(
                    "SUCCESS monitor result requires ADEQUATE observation quality"
                )
            if self.safety_status != MonitorSafetyStatus.SAFE:
                raise ValueError(
                    "SUCCESS monitor result requires SAFE safety status"
                )
        elif self.task_status == TaskStatus.FAILURE:
            if self.failure_kind is None:
                raise ValueError("FAILURE monitor result requires failure_kind")
            if self.recommended_action in {
                MonitorAction.CONTINUE,
                MonitorAction.ADVANCE,
            }:
                raise ValueError("FAILURE monitor result requires a recovery action")
        elif self.recommended_action == MonitorAction.ADVANCE:
            raise ValueError("only SUCCESS may recommend ADVANCE")

        if self.safety_status == MonitorSafetyStatus.UNSAFE and (
            self.task_status != TaskStatus.FAILURE
            or self.failure_kind != FailureKind.SAFETY
            or self.recommended_action != MonitorAction.ABORT_SAFETY
        ):
            raise ValueError(
                "UNSAFE must be a SAFETY failure recommending ABORT_SAFETY"
            )
        return self

    def require_current_dispatch(
        self,
        *,
        dispatch_id: str,
        subtask_id: str,
    ) -> None:
        if self.dispatch_id != dispatch_id or self.subtask_id != subtask_id:
            raise ValueError("stale live-monitor result for a different dispatch")

    def require_expected_outcome(self, expected: ExpectedOutcome) -> None:
        expected_ids = [condition.condition_id for condition in expected.conditions]
        returned_ids = [check.condition_id for check in self.condition_checks]
        if (
            len(returned_ids) != len(set(returned_ids))
            or set(returned_ids) != set(expected_ids)
        ):
            missing = sorted(set(expected_ids) - set(returned_ids))
            extra = sorted(set(returned_ids) - set(expected_ids))
            raise ValueError(
                "monitor must check every expected condition exactly once; "
                f"missing={missing}, extra={extra}"
            )
        if self.task_status != TaskStatus.SUCCESS:
            return
        by_id = {check.condition_id: check for check in self.condition_checks}
        unsatisfied = [
            condition.condition_id
            for condition in expected.conditions
            if condition.required
            and by_id[condition.condition_id].state != ConditionState.SATISFIED
        ]
        if unsatisfied:
            raise ValueError(
                "SUCCESS monitor result requires every required condition "
                "SATISFIED; unsatisfied="
                + repr(unsatisfied)
            )
        low_confidence = [
            condition.condition_id
            for condition in expected.conditions
            if condition.required
            and by_id[condition.condition_id].confidence < 0.8
        ]
        if self.confidence < 0.8 or low_confidence:
            raise ValueError(
                "SUCCESS monitor result requires confidence >= 0.8 for the "
                "result and every required condition; low_confidence="
                + repr(low_confidence)
            )


class GoalCheck(ContractModel):
    goal_id: str
    state: ConditionState
    evidence: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("goal_id", "evidence")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)


class ValidatorOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNKNOWN = "UNKNOWN"
    UNSAFE = "UNSAFE"


class ValidatorRecoverability(str, Enum):
    NONE = "NONE"
    REOBSERVE = "REOBSERVE"
    AUTO_LOCAL = "AUTO_LOCAL"
    REPLAN = "REPLAN"
    USER_ASSIST = "USER_ASSIST"
    ABORT_SAFETY = "ABORT_SAFETY"


class ValidationRequest(ContractModel):
    validation_spec: ValidationSpec
    terminal_observations: list[Observation]
    execution_evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_observations(self) -> "ValidationRequest":
        if not self.terminal_observations:
            raise ValueError("validator requires at least one terminal observation")
        return self

    def allowed_evidence_refs(self) -> set[str]:
        """Return host-supplied observation and dispatched-command identities."""

        allowed = {
            observation.observation_id
            for observation in self.terminal_observations
        }
        dispatches = self.execution_evidence.get("dispatches", [])
        if not isinstance(dispatches, list):
            return allowed
        for dispatch in dispatches:
            if isinstance(dispatch, str) and dispatch.strip():
                allowed.add(dispatch.strip())
                continue
            if not isinstance(dispatch, Mapping):
                continue
            candidates: list[Any] = [dispatch.get("dispatch_id")]
            for nested_key in ("command", "receipt"):
                nested = dispatch.get(nested_key)
                if isinstance(nested, Mapping):
                    candidates.append(nested.get("dispatch_id"))
            allowed.update(
                candidate.strip()
                for candidate in candidates
                if isinstance(candidate, str) and candidate.strip()
            )
        return allowed


class ValidationResult(ContractModel):
    spec_id: str
    outcome: ValidatorOutcome
    task_complete: bool
    goal_checks: list[GoalCheck]
    safety_status: SafetyStatus
    discrepancies: list[str] = Field(default_factory=list)
    recoverability: ValidatorRecoverability
    user_message: str
    validator_confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("spec_id", "user_message")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_outcome(self) -> "ValidationResult":
        if self.outcome == ValidatorOutcome.SUCCESS:
            if not self.task_complete:
                raise ValueError("SUCCESS requires task_complete=true")
            if self.recoverability != ValidatorRecoverability.NONE:
                raise ValueError("SUCCESS requires recoverability=NONE")
            if self.safety_status == SafetyStatus.UNSAFE:
                raise ValueError("UNSAFE result cannot be successful")
        elif self.task_complete:
            raise ValueError("only SUCCESS may set task_complete=true")

        if self.outcome == ValidatorOutcome.UNSAFE and (
            self.safety_status != SafetyStatus.UNSAFE
            or self.recoverability != ValidatorRecoverability.ABORT_SAFETY
        ):
            raise ValueError(
                "UNSAFE outcome requires unsafe safety status and ABORT_SAFETY"
            )
        if self.safety_status == SafetyStatus.UNSAFE and (
            self.outcome != ValidatorOutcome.UNSAFE
            or self.recoverability != ValidatorRecoverability.ABORT_SAFETY
        ):
            raise ValueError(
                "unsafe safety status requires UNSAFE outcome and ABORT_SAFETY"
            )
        return self

    def require_exact_spec(self, spec: ValidationSpec) -> None:
        if self.spec_id != spec.spec_id:
            raise ValueError(
                f"validator spec mismatch: expected {spec.spec_id}, got {self.spec_id}"
            )
        expected_ids = [goal.goal_id for goal in spec.goal_conditions]
        returned_ids = [check.goal_id for check in self.goal_checks]
        if (
            len(returned_ids) != len(set(returned_ids))
            or set(returned_ids) != set(expected_ids)
        ):
            missing = sorted(set(expected_ids) - set(returned_ids))
            extra = sorted(set(returned_ids) - set(expected_ids))
            raise ValueError(
                "validator must check every frozen goal exactly once; "
                f"missing={missing}, extra={extra}"
            )

        by_id = {check.goal_id: check for check in self.goal_checks}
        unsafe_goal = any(
            goal.predicate == "SAFE_EXECUTION"
            and by_id[goal.goal_id].state == ConditionState.VIOLATED
            for goal in spec.goal_conditions
        )
        required_satisfied = all(
            by_id[goal.goal_id].state == ConditionState.SATISFIED
            for goal in spec.goal_conditions
            if goal.required
        )
        host_complete = (
            required_satisfied
            and not unsafe_goal
            and self.safety_status != SafetyStatus.UNSAFE
        )
        if self.task_complete != host_complete:
            raise ValueError(
                "task_complete is inconsistent with the frozen required goals"
            )
        required_violated = any(
            by_id[goal.goal_id].state == ConditionState.VIOLATED
            for goal in spec.goal_conditions
            if goal.required
        )
        required_unknown = any(
            by_id[goal.goal_id].state == ConditionState.UNKNOWN
            for goal in spec.goal_conditions
            if goal.required
        )
        if unsafe_goal or self.safety_status == SafetyStatus.UNSAFE:
            expected_outcome = ValidatorOutcome.UNSAFE
        elif required_violated:
            expected_outcome = ValidatorOutcome.FAILURE
        elif required_unknown:
            expected_outcome = ValidatorOutcome.UNKNOWN
        else:
            expected_outcome = ValidatorOutcome.SUCCESS
        if self.outcome != expected_outcome:
            raise ValueError(
                "validator outcome is inconsistent with frozen required goals: "
                f"expected {expected_outcome.value}, got {self.outcome.value}"
            )

    def require_request(self, request: ValidationRequest) -> None:
        self.require_exact_spec(request.validation_spec)
        terminal_refs = {
            observation.observation_id
            for observation in request.terminal_observations
        }
        allowed = request.allowed_evidence_refs()
        execution_refs = allowed - terminal_refs
        invalid = sorted(
            {
                evidence_ref
                for check in self.goal_checks
                for evidence_ref in check.evidence_refs
                if evidence_ref not in allowed
            }
        )
        if invalid:
            raise ValueError(
                "validator referenced evidence not supplied by the host: "
                + repr(invalid)
            )
        checks_by_id = {check.goal_id: check for check in self.goal_checks}
        for goal in request.validation_spec.goal_conditions:
            check = checks_by_id[goal.goal_id]
            if (
                check.state != ConditionState.UNKNOWN
                and not check.evidence_refs
            ):
                raise ValueError(
                    f"validator check {goal.goal_id!r} made a definite claim "
                    "without a supplied evidence reference"
                )
            normalized_modalities = {
                modality.casefold() for modality in goal.evidence_modalities
            }
            modality_refs: set[str] = set()
            if normalized_modalities & {
                "terminal_observation",
                "final_image",
                "image",
            }:
                modality_refs.update(terminal_refs)
            if "execution_evidence" in normalized_modalities:
                modality_refs.update(execution_refs)
            if not modality_refs:
                modality_refs = allowed
            mismatched = sorted(
                set(check.evidence_refs) - modality_refs
            )
            if mismatched:
                raise ValueError(
                    f"validator check {goal.goal_id!r} used evidence refs "
                    f"outside its declared modalities: {mismatched!r}"
                )
        if self.outcome == ValidatorOutcome.SUCCESS:
            low_confidence = [
                goal.goal_id
                for goal in request.validation_spec.goal_conditions
                if goal.required and checks_by_id[goal.goal_id].confidence < 0.8
            ]
            if self.validator_confidence < 0.8 or low_confidence:
                raise ValueError(
                    "SUCCESS validation requires confidence >= 0.8 for the "
                    "result and every required goal; low_confidence="
                    + repr(low_confidence)
                )


class TaskPhase(str, Enum):
    IDLE = "IDLE"
    HRI = "HRI"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    PLANNING = "PLANNING"
    PUBLISHING = "PUBLISHING"
    MONITORING = "MONITORING"
    VALIDATING = "VALIDATING"
    REPLANNING = "REPLANNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    SAFETY_STOP = "SAFETY_STOP"


__all__ = [
    "ConditionCheck",
    "ConditionState",
    "ContractModel",
    "EpisodeRecord",
    "ExecutionCommand",
    "ExpectedCondition",
    "ExpectedOutcome",
    "FailureDetail",
    "FailureKind",
    "GoalCheck",
    "HRIDecision",
    "HRIDecisionKind",
    "HRIInteraction",
    "HRIMemoryAction",
    "InteractionKind",
    "MemoryActionKind",
    "MemoryContext",
    "MemoryRetrievalRequest",
    "MemoryStatus",
    "MemoryType",
    "MonitorAction",
    "MonitorProgress",
    "MonitorRequest",
    "MonitorResult",
    "MonitorSafetyStatus",
    "Observation",
    "ObservationQuality",
    "PlanResult",
    "PlanningRequest",
    "PlanningStatus",
    "PlannerRecoverability",
    "Precondition",
    "PredicateCondition",
    "PreferenceRecord",
    "PreferenceStatus",
    "PublicationReceipt",
    "RecoveryAction",
    "ReplanPolicy",
    "SafetyStatus",
    "Subtask",
    "TaskContract",
    "TaskPhase",
    "TaskStatus",
    "TimeoutPolicy",
    "ValidationRequest",
    "ValidationResult",
    "ValidationSpec",
    "ValidatorOutcome",
    "ValidatorRecoverability",
    "utc_now",
]
