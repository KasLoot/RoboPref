"""Semantic retrieval and consent-gated updates over Markdown memory."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any, Protocol
from uuid import uuid4

from prefmem.agents.contracts import (
    EpisodeRecord,
    MemoryContext,
    MemoryRetrievalRequest,
    MemoryStatus,
    MemoryType,
    PreferenceRecord,
    PreferenceStatus,
)
from prefmem.memory.embeddings import (
    EmbeddingDocument,
    EmbeddingGemmaClient,
    EmbeddingServiceError,
    cosine_similarity,
)
from prefmem.memory.store import MarkdownMemoryStore
from prefmem.recording import ExperimentRecorder, NullRecorder


class MemoryAuthorizationError(PermissionError):
    """Raised when host code tries to mutate preference memory without consent."""


class MemorySemanticReasoner(Protocol):
    def retrieve(
        self,
        *,
        request: MemoryRetrievalRequest,
        preference_candidates: list[dict[str, Any]],
        history_candidates: list[dict[str, Any]],
    ) -> MemoryContext: ...


class PrefMemMemoryService:
    """Retrieve semantic memories and enforce explicit preference authorization.

    Exact duplicate preference writes are idempotent. Near-duplicate writes are
    suppressed only when scope and structured metadata match, lexical order and
    vocabulary are nearly identical, and document cosine similarity reaches
    ``semantic_duplicate_threshold`` (0.995 by default). Set the threshold to
    ``None`` to disable this conservative semantic check. Broad paraphrases are
    intentionally retained because an embedding score cannot prove equivalence.
    """

    _SEMANTIC_DUPLICATE_LEXICAL_THRESHOLD = 0.90

    def __init__(
        self,
        store: MarkdownMemoryStore,
        embeddings: EmbeddingGemmaClient,
        *,
        semantic_reasoner: MemorySemanticReasoner | None = None,
        recorder: ExperimentRecorder | None = None,
        minimum_similarity: float = 0.20,
        candidate_limit: int = 20,
        semantic_duplicate_threshold: float | None = 0.995,
    ) -> None:
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between -1 and 1")
        self.store = store
        self.embeddings = embeddings
        self.semantic_reasoner = semantic_reasoner
        self.recorder = recorder or NullRecorder()
        self.minimum_similarity = minimum_similarity
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        self.candidate_limit = candidate_limit
        if semantic_duplicate_threshold is not None and not (
            0.0 < semantic_duplicate_threshold <= 1.0
        ):
            raise ValueError(
                "semantic_duplicate_threshold must be between 0 (exclusive) "
                "and 1, or None"
            )
        self.semantic_duplicate_threshold = semantic_duplicate_threshold
        self._vector_cache: dict[str, list[float]] = {}

    @staticmethod
    def _preference_fields_document(
        *,
        statement: str,
        scope: str,
        applicability: dict[str, Any],
        structured_value: Any,
        title: str,
    ) -> EmbeddingDocument:
        applicability_text = (
            f" Applicability: {applicability}."
            if applicability
            else ""
        )
        structured = (
            f" Structured value: {structured_value}."
            if structured_value is not None
            else ""
        )
        return EmbeddingDocument(
            title=title,
            text=(
                f"{statement} Scope: {scope}."
                f"{applicability_text}{structured}"
            ),
        )

    @classmethod
    def _preference_document(cls, record: PreferenceRecord) -> EmbeddingDocument:
        return cls._preference_fields_document(
            statement=record.statement,
            scope=record.scope,
            applicability=record.applicability,
            structured_value=record.structured_value,
            title=f"Approved preference {record.id}",
        )

    @staticmethod
    def _normalized_text(value: str) -> str:
        return " ".join(value.casefold().split())

    @classmethod
    def _same_preference_metadata(
        cls,
        record: PreferenceRecord,
        *,
        scope: str,
        applicability: dict[str, Any],
        structured_value: Any,
    ) -> bool:
        return (
            cls._normalized_text(record.scope) == cls._normalized_text(scope)
            and record.applicability == applicability
            and record.structured_value == structured_value
        )

    @classmethod
    def _lexically_safe_semantic_duplicate(
        cls,
        left: str,
        right: str,
    ) -> bool:
        """Reject high-cosine opposites and role swaps before deduplication.

        Embeddings alone can place "left" and "right" instructions very close.
        The conservative guard therefore requires nearly the same lexical
        sequence and vocabulary. It intentionally misses broad paraphrases;
        those require a relation classifier rather than destructive merging.
        """

        left_tokens = re.findall(r"\w+", left.casefold())
        right_tokens = re.findall(r"\w+", right.casefold())
        if not left_tokens or not right_tokens:
            return False
        vocabulary = set(left_tokens) | set(right_tokens)
        overlap = len(set(left_tokens) & set(right_tokens)) / len(vocabulary)
        sequence = SequenceMatcher(None, left_tokens, right_tokens).ratio()
        threshold = cls._SEMANTIC_DUPLICATE_LEXICAL_THRESHOLD
        return overlap >= threshold and sequence >= threshold

    def _find_semantic_duplicate(
        self,
        *,
        statement: str,
        scope: str,
        applicability: dict[str, Any],
        structured_value: Any,
        active_preferences: list[PreferenceRecord],
    ) -> tuple[PreferenceRecord, float] | None:
        threshold = self.semantic_duplicate_threshold
        if threshold is None:
            return None
        comparable = [
            record
            for record in active_preferences
            if self._same_preference_metadata(
                record,
                scope=scope,
                applicability=applicability,
                structured_value=structured_value,
            )
            and self._lexically_safe_semantic_duplicate(
                record.statement,
                statement,
            )
        ]
        if not comparable:
            return None

        title = "Approved user preference"
        documents = [
            self._preference_fields_document(
                statement=record.statement,
                scope=record.scope,
                applicability=record.applicability,
                structured_value=record.structured_value,
                title=title,
            )
            for record in comparable
        ]
        proposed_document = self._preference_fields_document(
            statement=statement,
            scope=scope,
            applicability=applicability,
            structured_value=structured_value,
            title=title,
        )
        existing_vectors = self._vectors_for(
            kind="preference-equivalence",
            records=comparable,
            documents=documents,
        )
        proposed_vector = self.embeddings.embed_documents([proposed_document])[0]
        scored = [
            (cosine_similarity(proposed_vector, vector), record)
            for record, vector in zip(comparable, existing_vectors, strict=True)
        ]
        score, record = max(scored, key=lambda item: item[0])
        return (record, score) if score >= threshold else None

    @staticmethod
    def _episode_document(record: EpisodeRecord) -> EmbeddingDocument:
        summary = record.summary or str(record.result)
        return EmbeddingDocument(
            title=f"Prior task episode {record.id}",
            text=(
                f"User request: {record.request}. "
                f"Summary: {summary}. Tags: {', '.join(record.tags)}."
            ),
        )

    @staticmethod
    def _cache_key(kind: str, record: PreferenceRecord | EpisodeRecord) -> str:
        if isinstance(record, PreferenceRecord):
            revision = f"{record.revision}:{record.updated_at.isoformat()}"
        else:
            revision = record.timestamp.isoformat()
        return f"{kind}:{record.id}:{revision}"

    def _vectors_for(
        self,
        *,
        kind: str,
        records: list[PreferenceRecord] | list[EpisodeRecord],
        documents: list[EmbeddingDocument],
    ) -> list[list[float]]:
        keys = [self._cache_key(kind, record) for record in records]
        missing_indices = [
            index for index, key in enumerate(keys) if key not in self._vector_cache
        ]
        if missing_indices:
            missing_documents = [documents[index] for index in missing_indices]
            vectors = self.embeddings.embed_documents(missing_documents)
            for index, vector in zip(missing_indices, vectors, strict=True):
                self._vector_cache[keys[index]] = vector
        return [self._vector_cache[key] for key in keys]

    def retrieve(
        self,
        *,
        username: str,
        request: MemoryRetrievalRequest,
    ) -> MemoryContext:
        preferences = (
            self.store.load_preferences(username=username)
            if MemoryType.PERSISTENT_PREFERENCE_MEMORY in request.memory_types
            else []
        )
        episodes = (
            self.store.load_episodes(username=username)
            if MemoryType.PERSISTENT_HISTORY_MEMORY in request.memory_types
            else []
        )
        if not preferences and not episodes:
            context = MemoryContext(
                status=MemoryStatus.EMPTY,
                request_id=request.request_id,
            )
            self.recorder.record_event(
                "Memory retrieval",
                {
                    "username": username,
                    "request": request,
                    "result": context,
                },
            )
            return context

        query_vector = self.embeddings.embed_query(request.search_text)
        preference_hits: list[dict[str, Any]] = []
        history_hits: list[dict[str, Any]] = []

        if preferences:
            documents = [
                self._preference_document(record) for record in preferences
            ]
            vectors = self._vectors_for(
                kind="preference",
                records=preferences,
                documents=documents,
            )
            scored = sorted(
                (
                    (cosine_similarity(query_vector, vector), record)
                    for record, vector in zip(preferences, vectors, strict=True)
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            preference_hits = [
                {
                    "record_id": record.id,
                    "record": record.model_dump(mode="json"),
                    "embedding_similarity": round(score, 6),
                }
                for score, record in scored[: self.candidate_limit]
                if score >= self.minimum_similarity
            ]

        if episodes:
            documents = [self._episode_document(record) for record in episodes]
            vectors = self._vectors_for(
                kind="episode",
                records=episodes,
                documents=documents,
            )
            scored = sorted(
                (
                    (cosine_similarity(query_vector, vector), record)
                    for record, vector in zip(episodes, vectors, strict=True)
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            history_hits = [
                {
                    "record_id": record.id,
                    "record": record.model_dump(mode="json"),
                    "embedding_similarity": round(score, 6),
                }
                for score, record in scored[: self.candidate_limit]
                if score >= self.minimum_similarity
            ]

        if not preference_hits and not history_hits:
            context = MemoryContext(
                status=MemoryStatus.EMPTY,
                request_id=request.request_id,
            )
        elif self.semantic_reasoner is not None:
            context = self.semantic_reasoner.retrieve(
                request=request,
                preference_candidates=preference_hits,
                history_candidates=history_hits,
            )
        else:
            context = MemoryContext(
                status=MemoryStatus.AVAILABLE,
                request_id=request.request_id,
                relevant_preferences=preference_hits[: request.limit],
                relevant_history=history_hits[: request.limit],
                warnings=[
                    "Semantic Memory Agent was not configured; returning "
                    "EmbeddingGemma candidates without applicability classification."
                ],
            )
        self.recorder.record_event(
            "Memory retrieval",
            {
                "username": username,
                "request": request,
                "result": context,
            },
        )
        return context

    def remember_preference(
        self,
        *,
        username: str,
        statement: str,
        authorized: bool,
        scope: str = "contextual",
        applicability: dict[str, Any] | None = None,
        structured_value: Any = None,
        evidence: Iterable[dict[str, Any] | str] = (),
        preference_id: str | None = None,
    ) -> PreferenceRecord:
        """Write one explicit preference; ``authorized`` must come from the host."""
        if not authorized:
            raise MemoryAuthorizationError(
                "preference writes require an explicit user request or confirmation"
            )
        normalized_statement = self._normalized_text(statement)
        requested_applicability = dict(applicability or {})
        active_preferences = self.store.load_preferences(username=username)
        for existing in active_preferences:
            if (
                self._normalized_text(existing.statement) == normalized_statement
                and self._same_preference_metadata(
                    existing,
                    scope=scope,
                    applicability=requested_applicability,
                    structured_value=structured_value,
                )
            ):
                self.recorder.record_event(
                    "Duplicate preference suppressed",
                    {
                        "authorization": "explicit-user-request-or-confirmation",
                        "existing_record": existing,
                    },
                )
                return existing
        try:
            semantic_duplicate = self._find_semantic_duplicate(
                statement=statement,
                scope=scope,
                applicability=requested_applicability,
                structured_value=structured_value,
                active_preferences=active_preferences,
            )
        except EmbeddingServiceError as exc:
            # Embeddings are a rebuildable retrieval aid, not the canonical
            # store. An unavailable sidecar must not discard an explicitly
            # authorized preference write.
            self.recorder.record_event(
                "Semantic duplicate check unavailable",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            semantic_duplicate = None
        if semantic_duplicate is not None:
            existing, similarity = semantic_duplicate
            self.recorder.record_event(
                "Semantic duplicate preference suppressed",
                {
                    "authorization": "explicit-user-request-or-confirmation",
                    "existing_record": existing,
                    "embedding_similarity": round(similarity, 6),
                    "threshold": self.semantic_duplicate_threshold,
                    "policy": (
                        "same metadata, high lexical equivalence, and high "
                        "EmbeddingGemma cosine similarity"
                    ),
                },
            )
            return existing
        now = datetime.now(UTC)
        record = PreferenceRecord(
            id=preference_id or f"pref-{uuid4().hex}",
            username=username,
            statement=statement,
            scope=scope,
            applicability=requested_applicability,
            structured_value=structured_value,
            evidence=list(evidence),
            created_at=now,
            updated_at=now,
        )
        self.store.save_preference(record)
        self.recorder.record_event(
            "Preference committed",
            {
                "authorization": "explicit-user-request-or-confirmation",
                "record": record,
                "path": self.store.preference_path,
            },
        )
        return record

    def revoke_preference(
        self,
        *,
        username: str,
        preference_id: str,
        authorized: bool,
        evidence: Iterable[dict[str, Any] | str] = (),
    ) -> PreferenceRecord:
        if not authorized:
            raise MemoryAuthorizationError(
                "preference revocation requires explicit user authorization"
            )
        records = self.store.load_preferences(
            username=username,
            active_only=False,
        )
        matches = [record for record in records if record.id == preference_id]
        if not matches:
            raise KeyError(f"no preference {preference_id!r} for {username!r}")
        current = matches[0]
        revoked = current.model_copy(
            update={
                "status": PreferenceStatus.REVOKED,
                "updated_at": datetime.now(UTC),
                "revision": current.revision + 1,
                "evidence": [*current.evidence, *evidence],
            }
        )
        self.store.commit_preference_revision(
            expected=current,
            replacements=[revoked],
        )
        self.recorder.record_event(
            "Preference revoked",
            {
                "authorization": "explicit-user-request",
                "record": revoked,
            },
        )
        return revoked

    def update_preference(
        self,
        *,
        username: str,
        preference_id: str,
        statement: str,
        authorized: bool,
        scope: str = "contextual",
        applicability: dict[str, Any] | None = None,
        structured_value: Any = None,
        evidence: Iterable[dict[str, Any] | str] = (),
    ) -> PreferenceRecord:
        """Create a new active revision and revoke the explicitly targeted record."""
        if not authorized:
            raise MemoryAuthorizationError(
                "preference updates require explicit user authorization"
            )
        records = self.store.load_preferences(
            username=username,
            active_only=False,
        )
        matches = [record for record in records if record.id == preference_id]
        if not matches:
            raise KeyError(f"no preference {preference_id!r} for {username!r}")
        current = matches[0]
        now = datetime.now(UTC)
        revoked = current.model_copy(
            update={
                "status": PreferenceStatus.REVOKED,
                "updated_at": now,
                "revision": current.revision + 1,
                "evidence": [*current.evidence, *evidence],
            }
        )
        replacement = PreferenceRecord(
            id=f"pref-{uuid4().hex}",
            username=username,
            statement=statement,
            scope=scope,
            applicability=dict(applicability or {}),
            structured_value=structured_value,
            evidence=list(evidence),
            created_at=now,
            updated_at=now,
            revision=current.revision + 1,
            supersedes_id=current.id,
        )
        self.store.commit_preference_revision(
            expected=current,
            replacements=[revoked, replacement],
        )
        self.recorder.record_event(
            "Preference updated",
            {
                "authorization": "explicit-user-request",
                "revoked": revoked,
                "replacement": replacement,
            },
        )
        return replacement

    def save_episode(self, episode: EpisodeRecord) -> None:
        self.store.save_episode(episode)
        self.recorder.record_event(
            "Episode memory committed",
            {"record": episode, "path": self.store.history_path},
        )
