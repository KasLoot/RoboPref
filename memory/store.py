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


class PreferenceStoreError(ValueError):
    pass


class PreferenceStore:
    """Persist structured user preferences and apply deterministic update policy."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def retrieve(self, query: str, user_id: str = "default", limit: int = 8) -> list[dict[str, Any]]:
        query_tokens = self._tokens(query)
        matches: list[tuple[float, dict[str, Any]]] = []

        for preference in self.list_preferences(user_id=user_id):
            if preference["status"] not in ACTIVE_STATUSES:
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
            score += min(float(preference["confidence"]), 1.0) * 0.1
            matches.append((score, preference))

        matches.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
        return [self._for_prompt(preference) for _, preference in matches[:limit]]

    def list_preferences(self, user_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            preferences = self._read()["preferences"]
        if user_id is not None:
            preferences = [item for item in preferences if item["user_id"] == user_id]
        return copy.deepcopy(preferences)

    def apply_operations(
        self,
        operations: list[dict[str, Any]],
        user_id: str = "default",
        session_id: str | None = None,
    ) -> list[dict[str, str]]:
        if not isinstance(operations, list):
            raise PreferenceStoreError("Memory operations must be a list.")

        with self._lock:
            data = self._read()
            applied: list[dict[str, str]] = []
            changed = False

            for operation in operations:
                result = self._apply_operation(data["preferences"], operation, user_id, session_id)
                if result is not None:
                    applied.append(result)
                    changed = True

            if changed:
                self._write(data)
            return applied

    def _apply_operation(
        self,
        preferences: list[dict[str, Any]],
        operation: dict[str, Any],
        user_id: str,
        session_id: str | None,
    ) -> dict[str, str] | None:
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
        scope_marker = evidence.get("scope_marker", "unmarked")
        if scope_marker not in VALID_SCOPE_MARKERS:
            raise PreferenceStoreError(f"Unsupported scope marker: {scope_marker}")
        if scope_marker == "explicit_one_off":
            return None

        incoming = self._validated_preference(operation.get("preference"), user_id)
        existing = self._find_matching(preferences, incoming)
        now = self._now()
        evidence_record = {
            "quote": str(evidence.get("quote", "")).strip(),
            "reason": str(evidence.get("reason", "")).strip(),
            "scope_marker": scope_marker,
            "session_id": session_id,
            "created_at": now,
        }

        if existing is not None and self._same_value(existing["value"], incoming["value"]):
            existing["evidence_count"] += 1
            existing["evidence"].append(evidence_record)
            existing["confidence"] = max(existing["confidence"], incoming["confidence"])
            if scope_marker in {"explicit_durable", "correction"} or existing["evidence_count"] >= 2:
                existing["status"] = "durable"
            existing["updated_at"] = now
            return {"action": "REINFORCED", "preference_id": existing["id"], "status": existing["status"]}

        if existing is not None and scope_marker in {"explicit_durable", "correction"}:
            existing["value"] = incoming["value"]
            existing["confidence"] = incoming["confidence"]
            existing["status"] = "durable"
            existing["evidence_count"] += 1
            existing["evidence"].append(evidence_record)
            existing["updated_at"] = now
            return {"action": "UPDATED", "preference_id": existing["id"], "status": existing["status"]}

        incoming.update(
            {
                "id": f"pref-{uuid.uuid4().hex}",
                "status": "durable" if scope_marker == "explicit_durable" else "candidate",
                "evidence_count": 1,
                "evidence": [evidence_record],
                "created_at": now,
                "updated_at": now,
            }
        )
        preferences.append(incoming)
        return {"action": "ADDED", "preference_id": incoming["id"], "status": incoming["status"]}

    def _retract(self, preferences: list[dict[str, Any]], operation: dict[str, Any]) -> dict[str, str]:
        preference_id = operation.get("preference_id")
        for preference in preferences:
            if preference["id"] == preference_id:
                preference["status"] = "retracted"
                preference["updated_at"] = self._now()
                return {"action": "RETRACTED", "preference_id": preference_id, "status": "retracted"}
        raise PreferenceStoreError(f"Cannot retract unknown preference: {preference_id}")

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
    def _find_matching(preferences: list[dict[str, Any]], incoming: dict[str, Any]) -> dict[str, Any] | None:
        identity_match = None
        for preference in preferences:
            if preference["status"] not in ACTIVE_STATUSES:
                continue
            if (
                preference["user_id"] == incoming["user_id"]
                and preference["task_type"] == incoming["task_type"]
                and preference["key"] == incoming["key"]
                and preference["scope"] == incoming["scope"]
                and preference["context"] == incoming["context"]
            ):
                if PreferenceStore._same_value(preference["value"], incoming["value"]):
                    return preference
                if identity_match is None:
                    identity_match = preference
        return identity_match

    @staticmethod
    def _same_value(left: Any, right: Any) -> bool:
        return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", value.lower().replace("_", " ")))

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
            "evidence_count": preference["evidence_count"],
        }

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
        return data

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