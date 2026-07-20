from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
ACTIVE_STATUSES = {"candidate", "durable"}
VALID_SCOPE_MARKERS = {"unmarked", "explicit_durable", "explicit_one_off", "correction"}
VALID_EVIDENCE_ORIGINS = {
    "explicit_preference",
    "memory_confirmation",
    "open_preference_answer",
    "user_initiated_override",
    "assistant_plan_confirmation",
    "current_task",
    "system_inference",
}
PERSISTABLE_UNMARKED_ORIGINS = {"open_preference_answer", "user_initiated_override"}
NON_PREFERENCE_ORIGINS = {"assistant_plan_confirmation", "current_task", "system_inference"}
AFFIRMATIONS = {"yes", "yeah", "yep", "sure", "ok", "okay", "sounds good", "correct"}
CONFIRMATION_EVIDENCE_THRESHOLD = 2


class PreferenceStoreError(ValueError):
    pass


class PreferenceStore:
    """Persist direct user preference evidence under an explicit-consent policy."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def retrieve(self, query: str, user_id: str = "default", limit: int = 8) -> list[dict[str, Any]]:
        query_tokens = self._tokens(query)
        matches: list[tuple[float, dict[str, Any]]] = []

        for preference in self.list_preferences(user_id=user_id):
            if preference.get("status") not in ACTIVE_STATUSES:
                continue
            searchable = json.dumps(
                {
                    "task_type": preference["task_type"],
                    "key": preference["key"],
                    "value": preference["value"],
                    "context": preference["context"],
                },
                sort_keys=True,
            )
            overlap = len(query_tokens & self._tokens(searchable))
            if query_tokens and overlap == 0:
                continue
            score = float(overlap)
            if preference["status"] == "durable":
                score += 0.25
            score += min(float(preference.get("confidence", 0.0)), 1.0) * 0.1
            matches.append((score, preference))

        matches.sort(key=lambda item: (item[0], item[1].get("updated_at", "")), reverse=True)
        return [self._for_prompt(preference) for _, preference in matches[:limit]]

    def list_preferences(self, user_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            preferences = self._read()["preferences"]
        if user_id is not None:
            preferences = [item for item in preferences if item.get("user_id") == user_id]
        return copy.deepcopy(preferences)

    def apply_operations(
        self,
        operations: list[dict[str, Any]],
        user_id: str = "default",
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(operations, list):
            raise PreferenceStoreError("Memory operations must be a list.")

        with self._lock:
            data = self._read()
            applied: list[dict[str, Any]] = []
            changed = False
            for operation in operations:
                result = self._apply_operation(data["preferences"], operation, user_id, session_id)
                if result is not None:
                    applied.append(result)
                    changed = True
            if changed:
                self._write(data)
            return applied

    def confirm_candidate(
        self,
        preference_id: str,
        *,
        user_id: str = "default",
        session_id: str | None = None,
        quote: str = "Yes.",
    ) -> dict[str, Any]:
        """Promote one candidate after an answer to a dedicated memory question."""
        with self._lock:
            data = self._read()
            target = self._by_id(data["preferences"], preference_id, user_id)
            if target.get("status") != "candidate":
                raise PreferenceStoreError("Only an active candidate can be confirmed.")

            now = self._now()
            for preference in data["preferences"]:
                if preference is target or preference.get("status") not in ACTIVE_STATUSES:
                    continue
                if self._same_identity(preference, target):
                    preference["status"] = "retracted"
                    preference["updated_at"] = now

            target["status"] = "durable"
            target["confirmation_pending"] = False
            target["updated_at"] = now
            target.setdefault("evidence", []).append(
                {
                    "quote": str(quote).strip(),
                    "reason": "The user affirmed a dedicated future-memory question.",
                    "scope_marker": "explicit_durable",
                    "origin": "memory_confirmation",
                    "independent": False,
                    "session_id": session_id,
                    "created_at": now,
                }
            )
            self._write(data)
            return {
                "action": "CONFIRMED",
                "preference_id": target["id"],
                "status": "durable",
                "value": copy.deepcopy(target["value"]),
            }

    def decline_candidate(
        self,
        preference_id: str,
        *,
        user_id: str = "default",
        session_id: str | None = None,
        quote: str = "No.",
    ) -> dict[str, Any]:
        """Close a memory prompt without deleting the underlying candidate evidence."""
        with self._lock:
            data = self._read()
            target = self._by_id(data["preferences"], preference_id, user_id)
            if target.get("status") != "candidate":
                raise PreferenceStoreError("Only an active candidate can be declined.")
            now = self._now()
            target["confirmation_pending"] = False
            target["confirmation_declined_at"] = now
            target["confirmation_prompt_evidence_count"] = target.get(
                "independent_evidence_count", target.get("evidence_count", 1)
            )
            target["updated_at"] = now
            target.setdefault("evidence", []).append(
                {
                    "quote": str(quote).strip(),
                    "reason": "The user declined a dedicated future-memory question.",
                    "scope_marker": "explicit_one_off",
                    "origin": "memory_confirmation",
                    "independent": False,
                    "session_id": session_id,
                    "created_at": now,
                }
            )
            self._write(data)
            return {
                "action": "DECLINED",
                "preference_id": target["id"],
                "status": "candidate",
                "value": copy.deepcopy(target["value"]),
            }

    def _apply_operation(
        self,
        preferences: list[dict[str, Any]],
        operation: dict[str, Any],
        user_id: str,
        session_id: str | None,
    ) -> dict[str, Any] | None:
        if not isinstance(operation, dict):
            raise PreferenceStoreError("Each memory operation must be an object.")

        action = str(operation.get("action", "")).upper()
        if action == "NOOP":
            return None
        if action == "RETRACT":
            return self._retract(preferences, operation)
        if action != "UPSERT":
            raise PreferenceStoreError(f"Unsupported memory action: {action or '<missing>'}")

        evidence = operation.get("evidence")
        if not isinstance(evidence, dict):
            raise PreferenceStoreError("UPSERT operations require evidence.")
        scope_marker = str(evidence.get("scope_marker", "unmarked"))
        if scope_marker not in VALID_SCOPE_MARKERS:
            raise PreferenceStoreError(f"Unsupported scope marker: {scope_marker}")
        # Missing provenance is fail-closed: a model omission cannot manufacture
        # preference evidence.
        origin = str(evidence.get("origin", "system_inference"))
        if origin not in VALID_EVIDENCE_ORIGINS:
            raise PreferenceStoreError(f"Unsupported evidence origin: {origin}")
        quote = str(evidence.get("quote", "")).strip()

        if scope_marker == "explicit_one_off" or origin in NON_PREFERENCE_ORIGINS:
            return None
        if self._normalise_text(quote) in AFFIRMATIONS and origin != "memory_confirmation":
            return None
        if scope_marker in {"explicit_durable", "correction"}:
            if origin != "memory_confirmation" and not self._has_durable_language(quote):
                # The curator cannot turn a current-task correction into durable consent.
                scope_marker = "unmarked"
        if scope_marker == "unmarked" and origin not in PERSISTABLE_UNMARKED_ORIGINS:
            return None

        incoming = self._validated_preference(operation.get("preference"), user_id)
        identity_matches = [
            item
            for item in preferences
            if item.get("status") in ACTIVE_STATUSES and self._same_identity(item, incoming)
        ]
        same_value = next(
            (item for item in identity_matches if self._same_value(item.get("value"), incoming["value"])),
            None,
        )
        conflicting_durable = next(
            (item for item in identity_matches if item.get("status") == "durable"),
            None,
        )
        now = self._now()
        independent = bool(evidence.get("independent", False))
        evidence_record = {
            "quote": quote,
            "reason": str(evidence.get("reason", "")).strip(),
            "scope_marker": scope_marker,
            "origin": origin,
            "independent": independent,
            "session_id": session_id,
            "created_at": now,
        }

        if same_value is not None:
            same_value["evidence_count"] = same_value.get("evidence_count", 1) + 1
            if independent:
                same_value["independent_evidence_count"] = same_value.get(
                    "independent_evidence_count", same_value.get("evidence_count", 2) - 1
                ) + 1
            same_value.setdefault("evidence", []).append(evidence_record)
            same_value["confidence"] = max(float(same_value.get("confidence", 0.0)), incoming["confidence"])
            same_value["updated_at"] = now
            if scope_marker in {"explicit_durable", "correction"}:
                self._retract_identity_peers(preferences, same_value, now)
                same_value["status"] = "durable"
                same_value["confirmation_pending"] = False
                return self._result("UPDATED", same_value)
            if same_value.get("status") == "candidate" and self._should_request_confirmation(same_value):
                same_value["confirmation_pending"] = True
                same_value["confirmation_prompt_evidence_count"] = same_value.get(
                    "independent_evidence_count", 1
                )
                return self._result("CONFIRM_REQUIRED", same_value, reason="repeated_independent_choice")
            return self._result("REINFORCED", same_value)

        if scope_marker in {"explicit_durable", "correction"}:
            target = identity_matches[0] if identity_matches else None
            if target is None:
                target = self._new_record(incoming, evidence_record, now, status="durable")
                preferences.append(target)
                return self._result("ADDED", target)
            target["value"] = copy.deepcopy(incoming["value"])
            target["confidence"] = incoming["confidence"]
            target["status"] = "durable"
            target["confirmation_pending"] = False
            target["evidence_count"] = target.get("evidence_count", 1) + 1
            target["independent_evidence_count"] = target.get("independent_evidence_count", 1) + int(independent)
            target.setdefault("evidence", []).append(evidence_record)
            target["updated_at"] = now
            self._retract_identity_peers(preferences, target, now)
            return self._result("UPDATED", target)

        candidate = self._new_record(incoming, evidence_record, now, status="candidate")
        preferences.append(candidate)
        if conflicting_durable is not None:
            candidate["confirmation_pending"] = True
            candidate["confirmation_prompt_evidence_count"] = candidate["independent_evidence_count"]
            return self._result(
                "CONFIRM_REQUIRED",
                candidate,
                reason="conflicts_with_durable",
                replaces_preference_id=conflicting_durable["id"],
            )
        return self._result("ADDED", candidate)

    @staticmethod
    def _new_record(
        incoming: dict[str, Any],
        evidence_record: dict[str, Any],
        now: str,
        *,
        status: str,
    ) -> dict[str, Any]:
        record = copy.deepcopy(incoming)
        record.update(
            {
                "id": f"pref-{uuid.uuid4().hex}",
                "status": status,
                "evidence_count": 1,
                "independent_evidence_count": int(bool(evidence_record.get("independent"))),
                "confirmation_pending": False,
                "evidence": [evidence_record],
                "created_at": now,
                "updated_at": now,
            }
        )
        return record

    @staticmethod
    def _should_request_confirmation(preference: dict[str, Any]) -> bool:
        count = preference.get("independent_evidence_count", preference.get("evidence_count", 0))
        last_prompt_count = preference.get("confirmation_prompt_evidence_count", 0)
        return (
            count >= CONFIRMATION_EVIDENCE_THRESHOLD
            and not preference.get("confirmation_pending", False)
            and count > last_prompt_count
        )

    @staticmethod
    def _result(action: str, preference: dict[str, Any], **extra: Any) -> dict[str, Any]:
        result = {
            "action": action,
            "preference_id": preference["id"],
            "status": preference["status"],
            "task_type": preference["task_type"],
            "key": preference["key"],
            "value": copy.deepcopy(preference["value"]),
        }
        result.update(extra)
        return result

    def _retract(self, preferences: list[dict[str, Any]], operation: dict[str, Any]) -> dict[str, Any]:
        preference_id = operation.get("preference_id")
        for preference in preferences:
            if preference.get("id") == preference_id:
                preference["status"] = "retracted"
                preference["updated_at"] = self._now()
                return self._result("RETRACTED", preference)
        raise PreferenceStoreError(f"Cannot retract unknown preference: {preference_id}")

    @staticmethod
    def _retract_identity_peers(
        preferences: list[dict[str, Any]], target: dict[str, Any], now: str
    ) -> None:
        for preference in preferences:
            if preference is target or preference.get("status") not in ACTIVE_STATUSES:
                continue
            if PreferenceStore._same_identity(preference, target):
                preference["status"] = "retracted"
                preference["updated_at"] = now

    @staticmethod
    def _validated_preference(preference: Any, user_id: str) -> dict[str, Any]:
        if not isinstance(preference, dict):
            raise PreferenceStoreError("UPSERT operations require a preference object.")
        task_type = str(preference.get("task_type", "")).strip()
        key = str(preference.get("key", "")).strip()
        context = preference.get("context", {})
        scope = preference.get("scope", "contextual")
        if not task_type or not key:
            raise PreferenceStoreError("Preferences require non-empty task_type and key fields.")
        if "value" not in preference:
            raise PreferenceStoreError("Preferences require a value field.")
        if not isinstance(context, dict):
            raise PreferenceStoreError("Preference context must be an object.")
        if scope not in {"global", "contextual"}:
            raise PreferenceStoreError(f"Unsupported preference scope: {scope}")
        try:
            confidence = float(preference.get("confidence", 0.5))
        except (TypeError, ValueError) as error:
            raise PreferenceStoreError("Preference confidence must be numeric.") from error
        if not 0.0 <= confidence <= 1.0:
            raise PreferenceStoreError("Preference confidence must be between 0.0 and 1.0.")
        return {
            "user_id": user_id,
            "task_type": task_type,
            "key": key,
            "value": copy.deepcopy(preference["value"]),
            "context": copy.deepcopy(context),
            "scope": scope,
            "confidence": confidence,
        }

    @staticmethod
    def _same_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return (
            left.get("user_id") == right.get("user_id")
            and left.get("task_type") == right.get("task_type")
            and left.get("key") == right.get("key")
            and left.get("scope") == right.get("scope")
            and left.get("context") == right.get("context")
        )

    @staticmethod
    def _same_value(left: Any, right: Any) -> bool:
        return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", value.lower().replace("_", " ")))

    @staticmethod
    def _normalise_text(value: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", "", value.lower()).strip()

    @staticmethod
    def _has_durable_language(value: str) -> bool:
        normalised = PreferenceStore._normalise_text(value)
        patterns = (
            r"\bfrom now on\b",
            r"\bgoing forward\b",
            r"\bin the future\b",
            r"\balways\b",
            r"\bremember\b",
            r"\bmy default\b",
            r"\bi prefer\b",
            r"\bi usually\b",
            r"\bi normally\b",
            r"\bevery time\b",
            r"\breplace (?:my|the) (?:saved )?default\b",
        )
        return any(re.search(pattern, normalised) for pattern in patterns)

    @staticmethod
    def _for_prompt(preference: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": preference["id"],
            "task_type": preference["task_type"],
            "key": preference["key"],
            "value": copy.deepcopy(preference["value"]),
            "context": copy.deepcopy(preference["context"]),
            "scope": preference["scope"],
            "status": preference["status"],
            "confidence": preference["confidence"],
            "evidence_count": preference.get("evidence_count", 1),
            "independent_evidence_count": preference.get(
                "independent_evidence_count", preference.get("evidence_count", 1)
            ),
            "confirmation_pending": preference.get("confirmation_pending", False),
        }

    @staticmethod
    def _by_id(preferences: list[dict[str, Any]], preference_id: str, user_id: str) -> dict[str, Any]:
        for preference in preferences:
            if preference.get("id") == preference_id and preference.get("user_id") == user_id:
                return preference
        raise PreferenceStoreError(f"Unknown preference: {preference_id}")

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "preferences": []}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise PreferenceStoreError(f"Could not read preference store {self.path}: {error}") from error
        if data.get("schema_version") != SCHEMA_VERSION or not isinstance(data.get("preferences"), list):
            raise PreferenceStoreError(f"Unsupported preference store schema in {self.path}")
        # Records written by the former policy may have become durable solely from
        # repetition or an action-confirmation "yes". Treat them as candidates until
        # direct durable consent is present; the source file is not rewritten on read.
        for preference in data["preferences"]:
            if preference.get("status") == "durable" and not self._has_durable_evidence(preference):
                preference["status"] = "candidate"
                preference["confirmation_pending"] = False
        return data

    @staticmethod
    def _has_durable_evidence(preference: dict[str, Any]) -> bool:
        for evidence in preference.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            origin = evidence.get("origin")
            marker = evidence.get("scope_marker")
            quote = str(evidence.get("quote", ""))
            if origin == "memory_confirmation":
                return True
            if origin == "explicit_preference" and marker in {"explicit_durable", "correction"}:
                if PreferenceStore._has_durable_language(quote):
                    return True
            # Backward-compatible acceptance for legacy records that contain an
            # unambiguously future-oriented quote but predate the origin field.
            if origin is None and marker in {"explicit_durable", "correction"}:
                if PreferenceStore._has_durable_language(quote):
                    return True
        return False

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
                temporary_path = handle.name
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
