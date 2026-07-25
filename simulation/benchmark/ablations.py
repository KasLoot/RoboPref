"""Evaluation-only memory ablations.

These adapters never alter PrefMem's production memory implementation. They expose
controlled history/preference removals for paired experiments while preserving the
repository attributes used by the benchmark snapshotter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from memory.models import MemoryContext, MemoryQuery


MEMORY_MODES = ("full", "no-memory", "history-only", "preference-only")


@dataclass(frozen=True, slots=True)
class MemoryContextDelivery:
    """Evaluation evidence for the MemoryContext actually returned to HRI."""

    user_id: str
    history_ids: tuple[str, ...]
    preference_ids: tuple[str, ...]
    foreign_history_ids: tuple[str, ...]
    foreign_preference_ids: tuple[str, ...]
    history_content_mismatch_ids: tuple[str, ...]
    preference_content_mismatch_ids: tuple[str, ...]
    malformed_history_records: int
    malformed_preference_records: int
    history_summary_matches_repository: bool | None
    history_ownership_verified: bool
    preference_ownership_verified: bool

    @property
    def owned_history_ids(self) -> tuple[str, ...]:
        if not self.history_ownership_verified:
            return ()
        foreign = set(self.foreign_history_ids)
        return tuple(item for item in self.history_ids if item not in foreign)

    @property
    def owned_preference_ids(self) -> tuple[str, ...]:
        if not self.preference_ownership_verified:
            return ()
        foreign = set(self.foreign_preference_ids)
        return tuple(
            item for item in self.preference_ids if item not in foreign
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "history_ids": list(self.history_ids),
            "preference_ids": list(self.preference_ids),
            "owned_history_ids": list(self.owned_history_ids),
            "owned_preference_ids": list(self.owned_preference_ids),
            "foreign_history_ids": list(self.foreign_history_ids),
            "foreign_preference_ids": list(self.foreign_preference_ids),
            "history_content_mismatch_ids": list(
                self.history_content_mismatch_ids
            ),
            "preference_content_mismatch_ids": list(
                self.preference_content_mismatch_ids
            ),
            "malformed_history_records": self.malformed_history_records,
            "malformed_preference_records": (
                self.malformed_preference_records
            ),
            "history_summary_matches_repository": (
                self.history_summary_matches_repository
            ),
            "history_ownership_verified": self.history_ownership_verified,
            "preference_ownership_verified": (
                self.preference_ownership_verified
            ),
        }


class MemoryContextRecordingAgent:
    """Transparent evaluation observer for contexts delivered by Memory Agent.

    The wrapped agent still performs its normal VLM retrieval. This adapter only
    records the returned IDs and binds preference ownership against the wrapped
    repositories at delivery time; it never changes the returned context.
    """

    def __init__(self, agent: Any):
        self.agent = agent
        self._deliveries: list[MemoryContextDelivery] = []
        self._turn_start = 0
        self._last_delivery: MemoryContextDelivery | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

    def begin_evaluation_turn(self) -> None:
        self._turn_start = len(self._deliveries)

    def delivery_for_current_turn(
        self,
        *,
        attached_context: MemoryContext | None = None,
        user_id: str | None = None,
    ) -> MemoryContextDelivery | None:
        if len(self._deliveries) > self._turn_start:
            return copy.deepcopy(self._deliveries[-1])
        # A USER_REPLY does not invoke Memory Agent again. Capture the exact
        # context already attached to HRI, which may have been refreshed by a
        # terminal history update since the original retrieval.
        if isinstance(attached_context, MemoryContext) and user_id:
            return copy.deepcopy(
                self._record_delivery(user_id, attached_context)
            )
        return copy.deepcopy(self._last_delivery)

    @property
    def deliveries(self) -> tuple[MemoryContextDelivery, ...]:
        return tuple(copy.deepcopy(self._deliveries))

    def get_memory_context(self, query: MemoryQuery) -> MemoryContext:
        context = self.agent.get_memory_context(query)
        if not isinstance(context, MemoryContext):
            raise TypeError("Memory Agent get_memory_context must return MemoryContext.")
        self._record_delivery(query.user_id, context)
        return context

    def _record_delivery(
        self,
        user_id: str,
        context: MemoryContext,
    ) -> MemoryContextDelivery:
        malformed_history_records = sum(
            not isinstance(item, dict)
            or not str(item.get("episode_id", "")).strip()
            for item in context.relevant_history
        )
        history_records = [
            item
            for item in context.relevant_history
            if isinstance(item, dict)
            and str(item.get("episode_id", "")).strip()
        ]
        history_ids = tuple(
            sorted(
                {
                    str(item.get("episode_id", "")).strip()
                    for item in history_records
                }
            )
        )
        malformed_preference_records = sum(
            not isinstance(item, dict)
            or not str(item.get("id", "")).strip()
            for item in context.relevant_preferences
        )
        preference_records = [
            item
            for item in context.relevant_preferences
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        ]
        preference_ids = tuple(
            sorted({str(item["id"]).strip() for item in preference_records})
        )

        owned_history_ids: set[str] = set()
        owned_histories_by_id: dict[str, dict[str, Any]] = {}
        history_summary_matches_repository: bool | None = None
        history = getattr(self.agent, "history_repository", None)
        list_histories = getattr(history, "list_episodes", None)
        get_summary = getattr(history, "get_summary", None)
        if callable(list_histories) and callable(get_summary):
            try:
                owned_histories = list_histories(user_id)
                owned_histories_by_id = {
                    str(item.get("episode_id", "")).strip(): copy.deepcopy(
                        item
                    )
                    for item in owned_histories
                    if isinstance(item, dict)
                    and str(item.get("episode_id", "")).strip()
                }
                owned_history_ids = set(owned_histories_by_id)
                summary = get_summary(user_id)
                if not isinstance(summary, dict):
                    raise TypeError(
                        "History repository summary must be an object."
                    )
                history_summary_matches_repository = (
                    str(context.history_summary)
                    == str(summary.get("text", ""))
                )
            except Exception:
                owned_histories_by_id = {}
                owned_history_ids = set()
                history_summary_matches_repository = None

        owned_preference_ids: set[str] = set()
        owned_preferences_by_id: dict[str, dict[str, Any]] = {}
        preferences = getattr(self.agent, "preference_repository", None)
        list_preferences = getattr(preferences, "list_preferences", None)
        if callable(list_preferences):
            try:
                try:
                    owned = list_preferences(
                        user_id,
                        include_inactive=True,
                    )
                except TypeError:
                    owned = list_preferences(user_id)
                owned_preferences_by_id = {
                    str(item.get("id", "")).strip(): copy.deepcopy(item)
                    for item in owned
                    if isinstance(item, dict)
                    and str(item.get("id", "")).strip()
                }
                owned_preference_ids = set(owned_preferences_by_id)
            except Exception:
                owned_preferences_by_id = {}
                owned_preference_ids = set()

        def base_record(record: dict[str, Any]) -> dict[str, Any]:
            value = copy.deepcopy(record)
            value.pop("match", None)
            return value

        delivered_histories_by_id = {
            str(item.get("episode_id", "")).strip(): item
            for item in history_records
        }
        history_content_mismatch_ids = tuple(
            sorted(
                history_id
                for history_id, record in delivered_histories_by_id.items()
                if history_id not in owned_histories_by_id
                or base_record(record) != owned_histories_by_id[history_id]
            )
        )
        delivered_preferences_by_id = {
            str(item.get("id", "")).strip(): item
            for item in preference_records
        }
        preference_content_mismatch_ids = tuple(
            sorted(
                preference_id
                for preference_id, record in delivered_preferences_by_id.items()
                if preference_id not in owned_preferences_by_id
                or base_record(record)
                != owned_preferences_by_id[preference_id]
            )
        )
        history_ownership_verified = (
            callable(list_histories)
            and callable(get_summary)
            and history_summary_matches_repository is True
            and malformed_history_records == 0
            and not history_content_mismatch_ids
        )
        preference_ownership_verified = (
            callable(list_preferences)
            and malformed_preference_records == 0
            and not preference_content_mismatch_ids
        )

        foreign_history_ids = tuple(
            sorted(
                {
                    history_id
                    for history_id in history_ids
                    if (
                        history_id not in owned_history_ids
                        or history_id in history_content_mismatch_ids
                        or any(
                            str(item.get("episode_id", "")).strip()
                            == history_id
                            and str(
                                item.get("user_id", user_id)
                            ).strip()
                            != user_id
                            for item in history_records
                        )
                    )
                }
            )
        )

        foreign_preference_ids = tuple(
            sorted(
                {
                    preference_id
                    for preference_id in preference_ids
                    if (
                        preference_id not in owned_preference_ids
                        or preference_id in preference_content_mismatch_ids
                        or any(
                            str(item.get("id", "")).strip() == preference_id
                            and str(item.get("user_id", user_id)).strip()
                            != user_id
                            for item in preference_records
                        )
                    )
                }
            )
        )
        delivery = MemoryContextDelivery(
            user_id=user_id,
            history_ids=history_ids,
            preference_ids=preference_ids,
            foreign_history_ids=foreign_history_ids,
            foreign_preference_ids=foreign_preference_ids,
            history_content_mismatch_ids=history_content_mismatch_ids,
            preference_content_mismatch_ids=(
                preference_content_mismatch_ids
            ),
            malformed_history_records=malformed_history_records,
            malformed_preference_records=malformed_preference_records,
            history_summary_matches_repository=(
                history_summary_matches_repository
            ),
            history_ownership_verified=history_ownership_verified,
            preference_ownership_verified=preference_ownership_verified,
        )
        self._deliveries.append(delivery)
        self._last_delivery = delivery
        return delivery


class MemoryAblationAgent:
    """Filter a real Memory Agent according to an explicit evaluation condition."""

    def __init__(self, agent: Any, mode: str = "full"):
        normalized = str(mode).strip().lower()
        if normalized not in MEMORY_MODES:
            raise ValueError(
                f"Unsupported memory mode {mode!r}; expected one of {MEMORY_MODES}."
            )
        self.agent = agent
        self.mode = normalized

    @property
    def history_enabled(self) -> bool:
        return self.mode in {"full", "history-only"}

    @property
    def preference_enabled(self) -> bool:
        return self.mode in {"full", "preference-only"}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

    def get_memory_context(self, query: MemoryQuery) -> MemoryContext:
        if not self.history_enabled and not self.preference_enabled:
            return self._empty_context()
        warnings: list[str] = []
        history_summary = ""
        relevant_history: list[dict[str, Any]] = []
        relevant_preferences: list[dict[str, Any]] = []
        if self.history_enabled:
            history = getattr(self.agent, "history_repository", None)
            get_summary = getattr(history, "get_summary", None)
            if callable(get_summary):
                summary = get_summary(query.user_id)
                if isinstance(summary, dict):
                    history_summary = str(summary.get("text", ""))
            try:
                relevant_history = self.agent.get_relevant_history_memory(
                    query
                )
            except Exception as error:
                list_episodes = getattr(history, "list_episodes", None)
                limit = int(
                    getattr(
                        self.agent,
                        "semantic_history_limit",
                        query.limit,
                    )
                )
                relevant_history = (
                    list_episodes(query.user_id, limit=limit)
                    if callable(list_episodes)
                    else []
                )
                warnings.append(
                    f"semantic history retrieval failed: {error}"
                )
        if self.preference_enabled:
            try:
                relevant_preferences = (
                    self.agent.get_relevant_preference_memory(query)
                )
            except Exception as error:
                warnings.append(
                    f"semantic preference retrieval failed: {error}"
                )
        return MemoryContext(
            history_summary=history_summary,
            relevant_history=relevant_history,
            relevant_preferences=relevant_preferences,
            history_version=int(
                getattr(
                    getattr(self.agent, "history_repository", None),
                    "version",
                    0,
                )
            ),
            preference_version=int(
                getattr(
                    getattr(self.agent, "preference_repository", None),
                    "version",
                    0,
                )
            ),
            warnings=warnings,
        )

    def get_relevant_history_memory(self, request: Any) -> list[dict[str, Any]]:
        if not self.history_enabled:
            return []
        return self.agent.get_relevant_history_memory(request)

    def get_relevant_preference_memory(
        self, request: Any
    ) -> list[dict[str, Any]]:
        if not self.preference_enabled:
            return []
        return self.agent.get_relevant_preference_memory(request)

    def get_relative_preference_memory(
        self, request: Any
    ) -> list[dict[str, Any]]:
        return self.get_relevant_preference_memory(request)

    def update_history_memory(self, conversation: dict[str, Any]) -> dict[str, Any]:
        if self.history_enabled:
            return self.agent.update_history_memory(conversation)
        context = self._empty_context()
        return {
            "status": "DISABLED_FOR_ABLATION",
            "episode": None,
            "history_summary": {"text": ""},
            "history_version": context.history_version,
            "history_context": context.to_dict(),
        }

    def update_preference_memory(
        self,
        request: dict[str, Any],
        *,
        consent: Any,
        transaction_id: str | None = None,
    ) -> dict[str, Any]:
        if self.preference_enabled:
            return self.agent.update_preference_memory(
                request,
                consent=consent,
                transaction_id=transaction_id,
            )
        return {
            "status": "DISABLED_FOR_ABLATION",
            "committed": False,
            "transaction": None,
            "compaction": None,
        }

    def compact_preference_memory(self, user_id: str) -> dict[str, Any]:
        if self.preference_enabled:
            return self.agent.compact_preference_memory(user_id)
        return {
            "status": "DISABLED_FOR_ABLATION",
            "transaction": None,
            "review_required": [],
        }

    def propose_preference_question(
        self, request: dict[str, Any]
    ) -> dict[str, Any] | None:
        if not self.preference_enabled or not self.history_enabled:
            return None
        return self.agent.propose_preference_question(request)

    def _empty_context(self) -> MemoryContext:
        history = getattr(self.agent, "history_repository", None)
        preferences = getattr(self.agent, "preference_repository", None)
        return MemoryContext(
            history_version=int(getattr(history, "version", 0)),
            preference_version=int(getattr(preferences, "version", 0)),
        )


def apply_memory_mode(agent: Any, mode: str) -> Any:
    """Return the original agent for the full condition, otherwise an adapter."""

    normalized = str(mode).strip().lower()
    if isinstance(agent, MemoryAblationAgent):
        if agent.mode == normalized:
            return agent
        agent = agent.agent
    if normalized == "full":
        return agent
    return MemoryAblationAgent(agent, normalized)


def record_memory_context(agent: Any) -> MemoryContextRecordingAgent:
    """Return one transparent recorder, preserving an existing recorder."""

    if isinstance(agent, MemoryContextRecordingAgent):
        return agent
    return MemoryContextRecordingAgent(agent)


__all__ = [
    "MEMORY_MODES",
    "MemoryAblationAgent",
    "MemoryContextDelivery",
    "MemoryContextRecordingAgent",
    "apply_memory_mode",
    "record_memory_context",
]
