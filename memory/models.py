from __future__ import annotations

import copy
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ConsentKind = Literal[
    "explicit_future_language",
    "memory_confirmation",
    "forget_confirmation",
    "memory_maintenance",
]
QuestionKind = Literal["TASK_CLARIFICATION", "TASK_CONFIRMATION", "MEMORY_CONSENT"]


@dataclass(frozen=True, slots=True)
class ConsentEvidence:
    kind: ConsentKind
    quote: str
    turn_id: str
    episode_id: str | None = None
    prompt_id: str | None = None
    authorized_action: str | None = None
    displayed_question: str | None = None
    proposal: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.kind not in {
            "explicit_future_language",
            "memory_confirmation",
            "forget_confirmation",
            "memory_maintenance",
        }:
            raise ValueError(f"Unsupported consent kind: {self.kind}")
        if self.kind != "memory_maintenance" and not self.quote.strip():
            raise ValueError("User-authorized memory changes require a non-empty quote.")
        if not self.turn_id.strip():
            raise ValueError("Consent evidence requires a turn_id.")
        if self.authorized_action is None:
            raise ValueError("Consent evidence must bind an authorized_action.")
        if self.kind == "memory_confirmation" and not self.prompt_id:
            raise ValueError("Memory confirmation requires the dedicated prompt_id.")
        if self.kind == "memory_confirmation" and (
            not self.displayed_question or not isinstance(self.proposal, dict)
        ):
            raise ValueError(
                "Memory confirmation must preserve the displayed question and proposal."
            )
        if (
            self.kind == "forget_confirmation"
            and self.prompt_id is not None
            and (not self.displayed_question or not isinstance(self.proposal, dict))
        ):
            raise ValueError(
                "Confirmed forgetting must preserve the displayed question and proposal."
            )
        if self.kind != "memory_maintenance" and not isinstance(self.proposal, dict):
            raise ValueError("User consent must preserve the authorized proposal.")
        if self.authorized_action is not None and self.authorized_action not in {
            "UPSERT",
            "DELETE",
            "COMPACT",
        }:
            raise ValueError("Consent evidence has an unsupported authorized_action.")


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    user_id: str
    request: str
    scene: dict[str, Any] = field(default_factory=dict)
    limit: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryContext:
    history_summary: str = ""
    relevant_history: list[dict[str, Any]] = field(default_factory=list)
    relevant_preferences: list[dict[str, Any]] = field(default_factory=list)
    history_version: int = 0
    preference_version: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))


@dataclass(slots=True)
class PendingQuestion:
    kind: QuestionKind
    payload: dict[str, Any]
    question: str
    prompt_id: str = field(default_factory=lambda: f"prompt-{uuid.uuid4().hex}")

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))


@dataclass(slots=True)
class AgentTurn:
    turn_id: str
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
