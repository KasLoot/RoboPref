from __future__ import annotations

import copy
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from agents.configs import AgentModelConfig, Memory_Agent_Config
from agents.diagnostics import AgentOutputDisplay, DisplayingJsonModel
from agents.model import JsonModel, OllamaJsonModel
from memory.models import ConsentEvidence, MemoryContext, MemoryQuery
from memory.repositories import HistoryRepository, PreferenceRepository


FORBIDDEN_HISTORY_KEYS = {
    "trace",
    "thinking",
    "reasoning",
    "chain_of_thought",
    "scene_inventory",
    "planning_status",
    "task_complete",
    "planner_confidence",
    "validator_confidence",
    "notes",
    "raw_image",
    "raw_images",
    "image_bytes",
    "action_tensor",
    "action_tensors",
}
HISTORY_FIELDS = {
    "summary",
    "user_request",
    "resolved_task",
    "actions",
    "execution",
    "validation",
    "user_choices",
    "user_visible_result",
    "terminal_outcome",
    "assurance",
    "memory_events",
    "tags",
}


class MemoryAgentError(RuntimeError):
    pass


class MemoryAgent:
    """VLM semantic layer over episodic history and approved preferences.

    The model decides relevance, equivalence, and proposed compaction. Repositories
    enforce consent, ownership, atomicity, idempotency, and reversible lineage.
    """

    def __init__(
        self,
        config: AgentModelConfig,
        history_repository: HistoryRepository,
        preference_repository: PreferenceRepository,
        *,
        model: JsonModel | None = None,
        output_display: AgentOutputDisplay | None = None,
        recent_history_limit: int = 5,
        history_semantic_scan_limit: int = 50,
        semantic_history_limit: int = 3,
        semantic_preference_limit: int = 5,
        semantic_preference_threshold: float = 0.75,
        preference_proposal_min_episodes: int = 2,
        preference_proposal_confidence: float = 0.75,
        compact_after_write: bool = True,
    ):
        self.config = config
        self.history_repository = history_repository
        self.preference_repository = preference_repository
        base_model = model or OllamaJsonModel(
            config.model,
            config.temperature,
            host=config.host,
            timeout_seconds=config.timeout_seconds,
        )
        self.model = (
            DisplayingJsonModel(base_model, output_display, "Memory Agent")
            if output_display is not None
            else base_model
        )
        self.recent_history_limit = recent_history_limit
        self.history_semantic_scan_limit = max(
            recent_history_limit, history_semantic_scan_limit
        )
        self.semantic_history_limit = semantic_history_limit
        self.semantic_preference_limit = semantic_preference_limit
        self.semantic_preference_threshold = semantic_preference_threshold
        self.preference_proposal_min_episodes = max(2, preference_proposal_min_episodes)
        self.preference_proposal_confidence = preference_proposal_confidence
        self.compact_after_write = compact_after_write
        self.system_prompt = Path(config.system_prompt_path).read_text(encoding="utf-8")

    def get_memory_context(self, query: MemoryQuery) -> MemoryContext:
        warnings: list[str] = []
        summary = self.history_repository.get_summary(query.user_id)
        try:
            relevant_history = self.get_relevant_history_memory(query)
        except Exception as error:
            relevant_history = self.history_repository.list_episodes(
                query.user_id, limit=self.semantic_history_limit
            )
            warnings.append(f"semantic history retrieval failed: {error}")
        try:
            relevant_preferences = self.get_relevant_preference_memory(query)
        except Exception as error:
            relevant_preferences = []
            warnings.append(f"semantic preference retrieval failed: {error}")
        return MemoryContext(
            history_summary=summary.get("text", ""),
            relevant_history=relevant_history,
            relevant_preferences=relevant_preferences,
            history_version=self.history_repository.version,
            preference_version=self.preference_repository.version,
            warnings=warnings,
        )

    def get_relative_preference_memory(
        self, hri_request: MemoryQuery | dict[str, Any] | str
    ) -> list[dict[str, Any]]:
        """Compatibility spelling retained for the user-proposed API."""
        return self.get_relevant_preference_memory(self._query(hri_request))

    def get_relevant_preference_memory(
        self, hri_request: MemoryQuery | dict[str, Any] | str
    ) -> list[dict[str, Any]]:
        query = self._query(hri_request)
        preferences = self.preference_repository.list_preferences(query.user_id)
        if not preferences:
            return []
        response = self.model.generate(
            purpose="retrieve_preferences",
            system_prompt=self.system_prompt,
            payload={
                "operation": "RETRIEVE_PREFERENCES",
                "query": query.to_dict(),
                "preferences": preferences,
                "rules": {
                    "semantic_reasoning_required": True,
                    "return_only_supplied_ids": True,
                    "current_explicit_request_overrides_memory": True,
                },
            },
        )
        raw_matches = response.get("matches", [])
        if not isinstance(raw_matches, list):
            raise MemoryAgentError("Preference retrieval requires a matches list.")
        by_id = {item["id"]: item for item in preferences}
        matches: list[dict[str, Any]] = []
        for match in raw_matches:
            if not isinstance(match, dict):
                continue
            preference_id = str(match.get("preference_id", ""))
            record = by_id.get(preference_id)
            if record is None:
                raise MemoryAgentError(
                    f"Preference reasoner returned an unknown or inactive ID: {preference_id}"
                )
            try:
                confidence = max(0.0, min(float(match.get("confidence", 0.0)), 1.0))
            except (TypeError, ValueError) as error:
                raise MemoryAgentError("Preference match confidence must be numeric.") from error
            enriched = copy.deepcopy(record)
            enriched["match"] = {
                "confidence": confidence,
                "relation": str(match.get("relation", "POSSIBLE")).upper(),
                "reason": str(match.get("reason", "")).strip(),
                "applicable_value": copy.deepcopy(match.get("applicable_value")),
            }
            if enriched["match"]["relation"] not in {"MATCH", "CONFLICT", "POSSIBLE"}:
                raise MemoryAgentError("Preference match relation must be MATCH, CONFLICT, or POSSIBLE.")
            enriched["match"]["usable_without_confirmation"] = (
                enriched["match"]["relation"] == "MATCH"
                and confidence >= self.semantic_preference_threshold
            )
            matches.append(enriched)
        matches.sort(key=lambda item: item["match"]["confidence"], reverse=True)
        return matches[: min(query.limit, self.semantic_preference_limit)]

    def get_relevant_history_memory(
        self, hri_request: MemoryQuery | dict[str, Any] | str
    ) -> list[dict[str, Any]]:
        query = self._query(hri_request)
        episodes = self.history_repository.list_episodes(
            query.user_id, limit=self.history_semantic_scan_limit
        )
        if not episodes:
            return []
        response = self.model.generate(
            purpose="retrieve_history",
            system_prompt=self.system_prompt,
            payload={
                "operation": "RETRIEVE_HISTORY",
                "query": query.to_dict(),
                "episodes": episodes,
                "rules": {
                    "history_is_context_not_preference_authority": True,
                    "return_only_supplied_ids": True,
                },
            },
        )
        raw_matches = response.get("matches", [])
        if not isinstance(raw_matches, list):
            raise MemoryAgentError("History retrieval requires a matches list.")
        by_id = {item["episode_id"]: item for item in episodes}
        matches: list[dict[str, Any]] = []
        for match in raw_matches:
            if not isinstance(match, dict):
                continue
            episode_id = str(match.get("episode_id", ""))
            episode = by_id.get(episode_id)
            if episode is None:
                raise MemoryAgentError(f"History reasoner returned unknown ID: {episode_id}")
            enriched = copy.deepcopy(episode)
            enriched["match"] = {
                "confidence": self._confidence(match.get("confidence")),
                "reason": str(match.get("reason", "")).strip(),
            }
            matches.append(enriched)
        matches.sort(key=lambda item: item["match"]["confidence"], reverse=True)
        return matches[: min(query.limit, self.semantic_history_limit)]

    def update_history_memory(self, newest_conversation: dict[str, Any]) -> dict[str, Any]:
        """Compact one completed task into an append-only history episode."""
        if not isinstance(newest_conversation, dict):
            raise MemoryAgentError("Newest conversation must be a structured object.")
        episode_id = str(
            newest_conversation.get("episode_id") or f"episode-{uuid.uuid4().hex}"
        )
        user_id = str(newest_conversation.get("user_id", "")).strip()
        if not user_id:
            raise MemoryAgentError("History updates require user_id.")
        clean_source = self._strip_reasoning(newest_conversation)
        source_fingerprint = hashlib.sha256(
            json.dumps(
                clean_source,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        existing = next(
            (
                item
                for item in self.history_repository.list_episodes(user_id)
                if item.get("episode_id") == episode_id
            ),
            None,
        )
        if existing is not None:
            if existing.get("source_fingerprint") != source_fingerprint:
                raise MemoryAgentError(
                    f"History idempotency conflict for {episode_id}: source changed."
                )
            summary = self.history_repository.get_summary(user_id)
            context = MemoryContext(
                history_summary=summary.get("text", ""),
                relevant_history=self.history_repository.list_episodes(
                    user_id, limit=self.recent_history_limit
                ),
                history_version=self.history_repository.version,
                preference_version=self.preference_repository.version,
            )
            return {
                "episode": existing,
                "history_summary": summary,
                "history_version": self.history_repository.version,
                "history_context": context.to_dict(),
            }
        try:
            response = self.model.generate(
                purpose="summarize_history_episode",
                system_prompt=self.system_prompt,
                payload={
                    "operation": "SUMMARIZE_EPISODE",
                    "conversation": clean_source,
                    "rules": {
                        "omit_reasoning": True,
                        "preserve_user_choices": True,
                        "preserve_agent_actions": True,
                        "attribute_uncertain_results": True,
                    },
                },
            )
            proposed = response.get("episode", response)
            if not isinstance(proposed, dict):
                raise MemoryAgentError("History summarizer did not return an episode object.")
            compact = {
                key: self._strip_reasoning(value)
                for key, value in proposed.items()
                if key in HISTORY_FIELDS
            }
            if not compact or not any(
                compact.get(key)
                for key in {"summary", "user_request", "resolved_task", "user_visible_result"}
            ):
                raise MemoryAgentError("History summarizer returned no meaningful episode facts.")
        except Exception:
            compact = self._fallback_episode(clean_source)

        episode = {
            "episode_id": episode_id,
            "user_id": user_id,
            "session_id": newest_conversation.get("session_id"),
            "started_at": newest_conversation.get("started_at"),
            "completed_at": newest_conversation.get("completed_at"),
            "source_fingerprint": source_fingerprint,
            **compact,
        }
        version_before_append = self.history_repository.version
        stored = self.history_repository.append(episode)
        if self.history_repository.version != version_before_append:
            self._refresh_history_summary(user_id)
        context = MemoryContext(
            history_summary=self.history_repository.get_summary(user_id).get("text", ""),
            relevant_history=self.history_repository.list_episodes(
                user_id, limit=self.recent_history_limit
            ),
            relevant_preferences=[],
            history_version=self.history_repository.version,
            preference_version=self.preference_repository.version,
        )
        return {
            "episode": stored,
            "history_summary": self.history_repository.get_summary(user_id),
            "history_version": self.history_repository.version,
            "history_context": context.to_dict(),
        }

    def update_preference_memory(
        self,
        hri_request: dict[str, Any],
        *,
        consent: ConsentEvidence,
        transaction_id: str | None = None,
    ) -> dict[str, Any]:
        """Use VLM reasoning to propose updates, then commit them with consent."""
        consent.validate()
        user_id = str(hri_request.get("user_id", "")).strip()
        if not user_id:
            raise MemoryAgentError("Preference updates require user_id.")
        transaction_id = transaction_id or str(
            hri_request.get("transaction_id") or f"txn-{uuid.uuid4().hex}"
        )
        snapshot_version = self.preference_repository.version
        existing = self.preference_repository.list_preferences(
            user_id, include_inactive=True
        )
        response = self.model.generate(
            purpose="update_preferences",
            system_prompt=self.system_prompt,
            payload={
                "operation": "PROPOSE_PREFERENCE_TRANSACTION",
                "request": copy.deepcopy(hri_request),
                "consent": consent.to_dict(),
                "existing_preferences": existing,
                "store_revision": snapshot_version,
                "rules": {
                    "semantic_reasoning_required": True,
                    "preserve_scope": True,
                    "cite_source_ids": True,
                    "one_transaction_may_have_multiple_operations": True,
                },
            },
        )
        operations = response.get("operations")
        if not isinstance(operations, list) or not operations:
            raise MemoryAgentError("Preference reasoner returned no operations.")
        normalized = self._operation_ids(operations, transaction_id)
        active_existing = [
            item for item in existing if item.get("status") == "active"
        ]
        for operation in normalized:
            if str(operation.get("action", "")).upper() != "MERGE":
                continue
            verification = self._verify_preference_merge(operation, active_existing)
            if verification is None:
                raise MemoryAgentError(
                    "A proposed merge did not pass independent semantic verification."
                )
            operation["semantic_verification"] = verification
        transaction = self.preference_repository.apply_transaction(
            user_id=user_id,
            transaction_id=transaction_id,
            operations=normalized,
            authorization=consent,
            requested_action=str(hri_request.get("requested_action", "UPSERT")),
            expected_version=snapshot_version,
            allowed_preference_ids={str(item["id"]) for item in existing},
        )
        compaction: dict[str, Any] | None = None
        if self.compact_after_write and len(self.preference_repository.list_preferences(user_id)) >= 2:
            try:
                compaction = self.compact_preference_memory(user_id)
            except Exception as error:
                # The primary user-authorized transaction is already durable.  A
                # maintenance failure must not be reported as a failed save.
                compaction = {"status": "FAILED", "error": str(error), "transaction": None}
        return {"transaction": transaction, "compaction": compaction}

    def compact_preference_memory(self, user_id: str) -> dict[str, Any]:
        """Let the VLM propose semantic merges; auto-apply only reversible lossless merges."""
        snapshot_version = self.preference_repository.version
        active = self.preference_repository.list_preferences(user_id)
        if len(active) < 2:
            return {"status": "NOT_NEEDED", "transaction": None, "review_required": []}
        response = self.model.generate(
            purpose="compact_preferences",
            system_prompt=self.system_prompt,
            payload={
                "operation": "COMPACT_PREFERENCES",
                "preferences": active,
                "store_revision": snapshot_version,
                "rules": {
                    "vlm_decides_semantic_equivalence": True,
                    "automatic_operations": ["MERGE", "NOOP"],
                    "automatic_merge_must_be_lossless": True,
                    "conflicts_or_generalizations_require_review": True,
                    "preserve_provenance": True,
                },
            },
        )
        operations = response.get("operations", [])
        if not isinstance(operations, list):
            raise MemoryAgentError("Preference compaction requires an operations list.")
        automatic: list[dict[str, Any]] = []
        review: list[dict[str, Any]] = []
        transaction_id = f"compact-{uuid.uuid4().hex}"
        for operation in self._operation_ids(operations, transaction_id):
            if str(operation.get("action", "")).upper() == "MERGE":
                verification = self._verify_preference_merge(operation, active)
                if verification is None:
                    review.append(
                        {
                            **operation,
                            "review_reason": (
                                "Independent semantic verification did not establish a "
                                "lossless, scope-preserving equivalence."
                            ),
                        }
                    )
                else:
                    operation["semantic_verification"] = verification
                    automatic.append(operation)
            elif str(operation.get("action", "")).upper() == "NOOP":
                continue
            else:
                review.append(operation)
        transaction = None
        if automatic:
            transaction = self.preference_repository.apply_transaction(
                user_id=user_id,
                transaction_id=transaction_id,
                operations=automatic,
                authorization=ConsentEvidence(
                    kind="memory_maintenance",
                    quote="VLM-proposed reversible semantic compaction.",
                    turn_id=transaction_id,
                    authorized_action="COMPACT",
                ),
                requested_action="COMPACT",
                expected_version=snapshot_version,
                allowed_preference_ids={str(item["id"]) for item in active},
            )
        return {
            "status": "COMPACTED" if transaction else "REVIEW_REQUIRED" if review else "NOOP",
            "transaction": transaction,
            "review_required": review,
        }

    def _verify_preference_merge(
        self,
        operation: dict[str, Any],
        active: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        source_ids = list(dict.fromkeys(map(str, operation.get("source_preference_ids", []))))
        by_id = {str(item.get("id")): item for item in active}
        if len(source_ids) < 2 or any(source_id not in by_id for source_id in source_ids):
            return None
        try:
            verdict = self.model.generate(
                purpose="verify_preference_merge",
                system_prompt=self.system_prompt,
                payload={
                    "operation": "VERIFY_PREFERENCE_MERGE",
                    "source_preferences": [by_id[source_id] for source_id in source_ids],
                    "proposed_preference": copy.deepcopy(operation.get("preference")),
                    "rules": {
                        "semantic_equivalence_required": True,
                        "scope_and_applicability_must_be_preserved": True,
                        "no_exception_may_be_discarded": True,
                        "independent_check_of_compaction_proposal": True,
                    },
                },
            )
        except Exception:
            return None
        returned_ids = list(
            dict.fromkeys(map(str, verdict.get("source_preference_ids", [])))
        )
        confidence = self._confidence(verdict.get("confidence"))
        if (
            set(returned_ids) != set(source_ids)
            or verdict.get("equivalent") is not True
            or verdict.get("scope_preserved") is not True
            or verdict.get("applicability_preserved") is not True
            or confidence < 0.85
        ):
            return None
        return {
            "source_preference_ids": source_ids,
            "equivalent": True,
            "scope_preserved": True,
            "applicability_preserved": True,
            "confidence": confidence,
            "reason": str(verdict.get("reason", "")).strip(),
        }

    def propose_preference_question(
        self, request: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Propose, but never persist, a dedicated future-preference question.

        The current terminal episode is considered together with persisted prior
        episodes.  The deterministic checks below enforce the minimum independent
        evidence count and reject hallucinated episode IDs; the VLM owns the semantic
        judgment that the choices actually express the same stable intention.
        """
        if not isinstance(request, dict):
            raise MemoryAgentError("Preference proposal input must be an object.")
        user_id = str(request.get("user_id", "")).strip()
        current = request.get("current_episode", request)
        if not user_id or not isinstance(current, dict):
            raise MemoryAgentError("Preference proposal requires user_id and current_episode.")
        current_id = str(current.get("episode_id", "")).strip()
        if not current_id:
            raise MemoryAgentError("Current proposal episode requires episode_id.")
        prior = self.history_repository.list_episodes(
            user_id, limit=self.recent_history_limit
        )
        if len(prior) + 1 < self.preference_proposal_min_episodes:
            return None
        episodes = prior + [self._strip_reasoning(current)]
        active = self.preference_repository.list_preferences(user_id)
        response = self.model.generate(
            purpose="propose_preference_question",
            system_prompt=self.system_prompt,
            payload={
                "operation": "PROPOSE_PREFERENCE_QUESTION",
                "current_episode_id": current_id,
                "episodes": episodes,
                "active_preferences": active,
                "rules": {
                    "minimum_independent_episodes": self.preference_proposal_min_episodes,
                    "history_is_not_consent": True,
                    "do_not_propose_if_already_saved": True,
                    "semantic_equivalence_required": True,
                },
            },
        )
        proposal = response.get("proposal")
        if proposal is None or response.get("should_ask") is False:
            return None
        if not isinstance(proposal, dict):
            raise MemoryAgentError("Preference proposal must be an object or null.")
        source_ids = list(dict.fromkeys(map(str, proposal.get("source_episode_ids", []))))
        allowed = {str(item.get("episode_id")) for item in episodes}
        if current_id not in source_ids or len(source_ids) < self.preference_proposal_min_episodes:
            raise MemoryAgentError("Preference proposal lacks enough independent episode evidence.")
        if set(source_ids) - allowed:
            raise MemoryAgentError("Preference proposal referenced an unknown episode.")
        confidence = self._confidence(proposal.get("confidence"))
        if confidence < self.preference_proposal_confidence:
            return None
        preference_request = proposal.get("preference_request")
        if not isinstance(preference_request, dict) or not isinstance(
            preference_request.get("preference"), dict
        ):
            raise MemoryAgentError("Preference proposal requires a structured preference_request.")
        statement = str(preference_request["preference"].get("statement", "")).strip()
        question = str(proposal.get("question", "")).strip()
        if not statement or not question:
            raise MemoryAgentError("Preference proposal requires a statement and user question.")
        return {
            "question": question,
            "preference_request": copy.deepcopy(preference_request),
            "source_episode_ids": source_ids,
            "confidence": confidence,
            "reason": str(proposal.get("reason", "")).strip(),
        }

    def _refresh_history_summary(self, user_id: str) -> None:
        episodes = self.history_repository.list_episodes(
            user_id, limit=self.recent_history_limit
        )
        previous = self.history_repository.get_summary(user_id)
        try:
            response = self.model.generate(
                purpose="compact_history",
                system_prompt=self.system_prompt,
                payload={
                    "operation": "COMPACT_HISTORY",
                    "previous_summary": previous,
                    "episodes": episodes,
                    "rules": {
                        "bounded": True,
                        "history_is_not_preference": True,
                        "omit_reasoning": True,
                        "cite_episode_ids": True,
                    },
                },
            )
            summary = str(response.get("summary", "")).strip()
            source_ids = response.get("source_episode_ids", [])
            if not isinstance(source_ids, list):
                raise MemoryAgentError("History summary source IDs must be a list.")
            if summary:
                self.history_repository.set_summary(user_id, summary, list(map(str, source_ids)))
        except Exception:
            # Episode persistence is authoritative; a failed rolling-summary refresh
            # must never damage or roll back it.
            return

    @staticmethod
    def _fallback_episode(source: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        mapping = {
            "user_request": "user_request",
            "resolved_task": "resolved_task",
            "actions": "actions",
            "execution": "execution",
            "validation": "validation",
            "user_choices": "user_choices",
            "user_visible_result": "user_visible_result",
            "memory_events": "memory_events",
            "terminal_outcome": "terminal_outcome",
            "assurance": "assurance",
        }
        for source_key, target_key in mapping.items():
            if source_key in source:
                result[target_key] = copy.deepcopy(source[source_key])
        result.setdefault("summary", str(source.get("user_visible_result", "Task episode recorded.")))
        return result

    @classmethod
    def _strip_reasoning(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: cls._strip_reasoning(item)
                for key, item in value.items()
                if not cls._is_forbidden_history_key(key)
            }
        if isinstance(value, list):
            return [cls._strip_reasoning(item) for item in value]
        return copy.deepcopy(value)

    @staticmethod
    def _is_forbidden_history_key(key: Any) -> bool:
        normalized = str(key).strip().lower().replace("-", "_")
        return (
            normalized in FORBIDDEN_HISTORY_KEYS
            or "chain_of_thought" in normalized
            or normalized.endswith("_reasoning")
            or normalized.endswith("_thinking")
            or normalized.startswith("hidden_")
        )

    @staticmethod
    def _operation_ids(
        operations: list[dict[str, Any]], transaction_id: str
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                raise MemoryAgentError("Each preference operation must be an object.")
            item = copy.deepcopy(operation)
            item.setdefault("operation_id", f"{transaction_id}-op-{index + 1}")
            normalized.append(item)
        return normalized

    @staticmethod
    def _query(value: MemoryQuery | dict[str, Any] | str) -> MemoryQuery:
        if isinstance(value, MemoryQuery):
            return value
        if isinstance(value, str):
            return MemoryQuery(user_id="default", request=value)
        if isinstance(value, dict):
            return MemoryQuery(
                user_id=str(value.get("user_id", "default")),
                request=str(value.get("request") or value.get("query") or ""),
                scene=copy.deepcopy(value.get("scene", {})),
                limit=int(value.get("limit", 5)),
            )
        raise MemoryAgentError("Unsupported memory query type.")

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0


# Legacy-style class name retained without preserving the old candidate policy.
Memory_Agent = MemoryAgent

__all__ = ["MemoryAgent", "Memory_Agent", "Memory_Agent_Config", "MemoryAgentError"]
