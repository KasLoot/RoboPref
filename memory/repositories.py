from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from memory.models import ConsentEvidence


HISTORY_SCHEMA_VERSION = 2
PREFERENCE_SCHEMA_VERSION = 2
HISTORY_OUTBOX_SCHEMA_VERSION = 1
ACTIVE_PREFERENCE_STATUS = "active"

_LOCK_REGISTRY_GUARD = threading.Lock()
_LOCK_REGISTRY: dict[str, threading.RLock] = {}


class MemoryRepositoryError(ValueError):
    pass


class _AtomicJsonRepository:
    def __init__(self, path: str | Path, empty_factory: Callable[[], dict[str, Any]]):
        self.path = Path(path)
        self._empty_factory = empty_factory
        lock_key = str(self.path.expanduser().resolve())
        with _LOCK_REGISTRY_GUARD:
            self._lock = _LOCK_REGISTRY.setdefault(lock_key, threading.RLock())

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_factory()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            backup = self.path.with_suffix(self.path.suffix + ".bak")
            if backup.exists():
                try:
                    with backup.open("r", encoding="utf-8") as handle:
                        recovered = json.load(handle)
                    if isinstance(recovered, dict):
                        corrupt_copy = self.path.with_name(
                            f"{self.path.name}.corrupt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                        )
                        shutil.copy2(self.path, corrupt_copy)
                        shutil.copy2(backup, self.path)
                        return recovered
                except (OSError, json.JSONDecodeError):
                    pass
            raise MemoryRepositoryError(f"Could not read {self.path}: {error}") from error
        if not isinstance(data, dict):
            raise MemoryRepositoryError(f"Memory repository must contain a JSON object: {self.path}")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        backup = self.path.with_suffix(self.path.suffix + ".bak")
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as current:
                    valid_current = json.load(current)
                if isinstance(valid_current, dict):
                    shutil.copy2(self.path, backup)
            except (OSError, json.JSONDecodeError):
                # Keep the last known-good backup when the primary is corrupt.
                pass
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
                handle.write("\n")
                temporary_path = handle.name
            os.replace(temporary_path, self.path)
            if not backup.exists():
                shutil.copy2(self.path, backup)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @contextmanager
    def _exclusive_file_lock(self):
        """Serialize read-modify-write transactions across repository processes."""
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class HistoryRepository(_AtomicJsonRepository):
    """Append-only structured task episodes plus bounded per-user summaries."""

    def __init__(self, path: str | Path):
        super().__init__(
            path,
            lambda: {
                "schema_version": HISTORY_SCHEMA_VERSION,
                "version": 0,
                "episodes": [],
                "summaries": {},
            },
        )

    def append(self, episode: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(episode, dict):
            raise MemoryRepositoryError("History episode must be an object.")
        episode_id = str(episode.get("episode_id", "")).strip()
        user_id = str(episode.get("user_id", "")).strip()
        if not episode_id or not user_id:
            raise MemoryRepositoryError("History episodes require episode_id and user_id.")

        with self._lock, self._exclusive_file_lock():
            data = self._read()
            self._validate(data)
            existing = next(
                (item for item in data["episodes"] if item.get("episode_id") == episode_id),
                None,
            )
            candidate = copy.deepcopy(episode)
            candidate.setdefault("created_at", self._now())
            if existing is not None:
                comparable_existing = dict(existing)
                comparable_candidate = dict(candidate)
                comparable_candidate["created_at"] = comparable_existing.get("created_at")
                if comparable_existing != comparable_candidate:
                    raise MemoryRepositoryError(
                        f"Episode idempotency conflict for {episode_id}: payload changed."
                    )
                return copy.deepcopy(existing)
            data["episodes"].append(candidate)
            data["version"] += 1
            self._validate(data)
            self._write(data)
            return copy.deepcopy(candidate)

    def list_episodes(self, user_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock, self._exclusive_file_lock():
            data = self._read()
            self._validate(data)
        episodes = [item for item in data["episodes"] if item.get("user_id") == user_id]
        episodes.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        if limit is not None:
            episodes = episodes[: max(0, limit)]
        return copy.deepcopy(episodes)

    def set_summary(
        self,
        user_id: str,
        summary: str,
        source_episode_ids: list[str],
    ) -> dict[str, Any]:
        with self._lock, self._exclusive_file_lock():
            data = self._read()
            self._validate(data)
            owned_ids = {
                item["episode_id"]
                for item in data["episodes"]
                if item.get("user_id") == user_id
            }
            unknown = set(source_episode_ids) - owned_ids
            if unknown:
                raise MemoryRepositoryError(
                    f"History summary references unknown episodes: {sorted(unknown)}"
                )
            record = {
                "text": str(summary).strip(),
                "source_episode_ids": list(dict.fromkeys(source_episode_ids)),
                "updated_at": self._now(),
            }
            data["summaries"][user_id] = record
            data["version"] += 1
            self._validate(data)
            self._write(data)
            return copy.deepcopy(record)

    def get_summary(self, user_id: str) -> dict[str, Any]:
        with self._lock, self._exclusive_file_lock():
            data = self._read()
            self._validate(data)
        return copy.deepcopy(
            data["summaries"].get(
                user_id,
                {"text": "", "source_episode_ids": [], "updated_at": None},
            )
        )

    @property
    def version(self) -> int:
        with self._lock:
            data = self._read()
            self._validate(data)
            return int(data["version"])

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        if data.get("schema_version") != HISTORY_SCHEMA_VERSION:
            raise MemoryRepositoryError("Unsupported history schema version.")
        if not isinstance(data.get("episodes"), list) or not isinstance(data.get("summaries"), dict):
            raise MemoryRepositoryError("Malformed history repository.")
        if not isinstance(data.get("version"), int):
            raise MemoryRepositoryError("History version must be an integer.")
        episode_ids: set[str] = set()
        for episode in data["episodes"]:
            if not isinstance(episode, dict):
                raise MemoryRepositoryError("History episodes must be objects.")
            episode_id = str(episode.get("episode_id", "")).strip()
            user_id = str(episode.get("user_id", "")).strip()
            if not episode_id or not user_id or episode_id in episode_ids:
                raise MemoryRepositoryError("History episode IDs and users must be valid and unique.")
            episode_ids.add(episode_id)


class PreferenceRepository(_AtomicJsonRepository):
    """Consent-gated semantic preferences with reversible VLM-driven merges."""

    def __init__(self, path: str | Path):
        super().__init__(
            path,
            lambda: {
                "schema_version": PREFERENCE_SCHEMA_VERSION,
                "version": 0,
                "preferences": [],
                "transactions": {},
            },
        )
        self.audit_path = self.path.with_suffix(self.path.suffix + ".audit.jsonl")

    def list_preferences(
        self,
        user_id: str,
        *,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read()
            self._validate(data)
        records = [item for item in data["preferences"] if item.get("user_id") == user_id]
        if not include_inactive:
            records = [item for item in records if item.get("status") == ACTIVE_PREFERENCE_STATUS]
        return copy.deepcopy(records)

    def get(self, preference_id: str, user_id: str) -> dict[str, Any]:
        records = self.list_preferences(user_id, include_inactive=True)
        for record in records:
            if record.get("id") == preference_id:
                return record
        raise MemoryRepositoryError(f"Unknown preference {preference_id!r} for user {user_id!r}.")

    def apply_transaction(
        self,
        *,
        user_id: str,
        transaction_id: str,
        operations: list[dict[str, Any]],
        authorization: ConsentEvidence,
        requested_action: str | None = None,
        expected_version: int | None = None,
        allowed_preference_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        audit_base = {
            "timestamp": self._now(),
            "transaction_id": transaction_id,
            "user_id": user_id,
            "requested_action": requested_action,
            "expected_version": expected_version,
            "authorization": {
                "kind": authorization.kind,
                "turn_id": authorization.turn_id,
                "episode_id": authorization.episode_id,
                "prompt_id": authorization.prompt_id,
                "authorized_action": authorization.authorized_action,
            },
            "operations": [
                {
                    "operation_id": operation.get("operation_id"),
                    "action": operation.get("action"),
                    "target_preference_id": operation.get("target_preference_id"),
                    "source_preference_ids": operation.get("source_preference_ids", []),
                }
                if isinstance(operation, dict)
                else {"malformed": True}
                for operation in operations
            ]
            if isinstance(operations, list)
            else [],
        }
        try:
            result = self._apply_transaction(
                user_id=user_id,
                transaction_id=transaction_id,
                operations=operations,
                authorization=authorization,
                requested_action=requested_action,
                expected_version=expected_version,
                allowed_preference_ids=allowed_preference_ids,
            )
        except Exception as error:
            self._append_audit({**audit_base, "status": "REJECTED", "error": str(error)})
            raise
        self._append_audit(
            {
                **audit_base,
                "status": "ACCEPTED",
                "committed_version": result.get("version"),
                "results": copy.deepcopy(result.get("results", [])),
            }
        )
        return result

    def _apply_transaction(
        self,
        *,
        user_id: str,
        transaction_id: str,
        operations: list[dict[str, Any]],
        authorization: ConsentEvidence,
        requested_action: str | None = None,
        expected_version: int | None = None,
        allowed_preference_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        try:
            authorization.validate()
        except ValueError as error:
            raise MemoryRepositoryError(str(error)) from error
        if not transaction_id.strip():
            raise MemoryRepositoryError("Preference transactions require an id.")
        if not isinstance(operations, list) or not operations:
            raise MemoryRepositoryError("Preference transactions require at least one operation.")

        requested_action = str(
            requested_action
            or (
                "COMPACT"
                if authorization.kind == "memory_maintenance"
                else "DELETE"
                if authorization.kind == "forget_confirmation"
                else "UPSERT"
            )
        ).upper()
        if (
            authorization.authorized_action is not None
            and authorization.authorized_action != requested_action
        ):
            raise MemoryRepositoryError(
                "Consent evidence is not bound to the requested preference action."
            )
        allowed_actions = {
            "UPSERT": {"ADD", "UPDATE", "MERGE", "RETRACT", "NOOP"},
            "DELETE": {"DELETE", "NOOP"},
            "COMPACT": {"MERGE", "NOOP"},
        }.get(requested_action)
        if allowed_actions is None:
            raise MemoryRepositoryError(f"Unsupported requested preference action: {requested_action}")
        if requested_action == "DELETE" and authorization.kind != "forget_confirmation":
            raise MemoryRepositoryError("DELETE intent requires explicit forget authorization.")
        if requested_action != "DELETE" and authorization.kind == "forget_confirmation":
            raise MemoryRepositoryError("Forget authorization cannot authorize a preference write.")
        if requested_action == "COMPACT" and authorization.kind != "memory_maintenance":
            raise MemoryRepositoryError("Automatic compaction requires maintenance authorization.")

        for operation in operations:
            if not isinstance(operation, dict):
                raise MemoryRepositoryError("Each preference operation must be an object.")
            action = str(operation.get("action", "")).upper()
            if action not in allowed_actions:
                raise MemoryRepositoryError(
                    f"{requested_action} authorization cannot apply {action or '<missing>'}."
                )
            if allowed_preference_ids is not None:
                references = set(map(str, operation.get("source_preference_ids", [])))
                if action != "MERGE" and operation.get("target_preference_id") is not None:
                    references.add(str(operation["target_preference_id"]))
                outside = references - allowed_preference_ids
                if outside:
                    raise MemoryRepositoryError(
                        f"Operation references preferences outside its supplied snapshot: {sorted(outside)}"
                    )

        with self._lock, self._exclusive_file_lock():
            data = self._read()
            self._validate(data)
            existing_transaction = data["transactions"].get(transaction_id)
            if existing_transaction is not None:
                replay = {
                    "user_id": user_id,
                    "authorization": authorization.to_dict(),
                    "operations": operations,
                    "requested_action": requested_action,
                    "expected_version": expected_version,
                    "allowed_preference_ids": sorted(allowed_preference_ids)
                    if allowed_preference_ids is not None
                    else None,
                }
                if existing_transaction.get("request") != replay:
                    raise MemoryRepositoryError(
                        f"Transaction idempotency conflict for {transaction_id}: payload changed."
                    )
                return copy.deepcopy(existing_transaction)
            if expected_version is not None and data["version"] != expected_version:
                raise MemoryRepositoryError(
                    f"Stale preference revision: expected {expected_version}, current {data['version']}."
                )

            working = copy.deepcopy(data)
            results: list[dict[str, Any]] = []
            operation_ids: set[str] = set()
            for operation in operations:
                if not isinstance(operation, dict):
                    raise MemoryRepositoryError("Each preference operation must be an object.")
                operation_id = str(operation.get("operation_id", "")).strip()
                if not operation_id or operation_id in operation_ids:
                    raise MemoryRepositoryError("Preference operation IDs must be unique and non-empty.")
                operation_ids.add(operation_id)
                results.append(
                    self._apply_operation(
                        working["preferences"], operation, user_id, authorization
                    )
                )

            working["version"] += 1
            transaction = {
                "transaction_id": transaction_id,
                "user_id": user_id,
                "authorization": authorization.to_dict(),
                "request": {
                    "user_id": user_id,
                    "authorization": authorization.to_dict(),
                    "operations": copy.deepcopy(operations),
                    "requested_action": requested_action,
                    "expected_version": expected_version,
                    "allowed_preference_ids": sorted(allowed_preference_ids)
                    if allowed_preference_ids is not None
                    else None,
                },
                "results": results,
                "created_at": self._now(),
                "version": working["version"],
            }
            working["transactions"][transaction_id] = transaction
            self._validate(working)
            self._write(working)
            return copy.deepcopy(transaction)

    def _append_audit(self, event: dict[str, Any]) -> None:
        """Append a reasoning-free accepted/rejected mutation event."""
        try:
            with self._lock, self._exclusive_file_lock():
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self.audit_path.open("a", encoding="utf-8") as handle:
                    json.dump(event, handle, ensure_ascii=False, sort_keys=True, default=str)
                    handle.write("\n")
        except OSError:
            # The transaction document also contains the complete accepted audit.
            # An auxiliary JSONL failure must never invert an already durable commit.
            return

    def _apply_operation(
        self,
        preferences: list[dict[str, Any]],
        operation: dict[str, Any],
        user_id: str,
        authorization: ConsentEvidence,
    ) -> dict[str, Any]:
        action = str(operation.get("action", "")).upper()
        if action == "NOOP":
            return {"operation_id": operation["operation_id"], "action": "NOOP"}
        if action == "MERGE":
            if authorization.kind != "memory_maintenance":
                self._require_write_consent(authorization)
            return self._merge(preferences, operation, user_id, authorization)
        if authorization.kind == "memory_maintenance":
            raise MemoryRepositoryError("Maintenance authorization permits only MERGE or NOOP.")
        if action == "ADD":
            self._require_write_consent(authorization)
            return self._add(preferences, operation, user_id, authorization)
        if action == "UPDATE":
            self._require_write_consent(authorization)
            return self._update(preferences, operation, user_id, authorization)
        if action == "RETRACT":
            self._require_write_consent(authorization)
            return self._retract(preferences, operation, user_id, authorization)
        if action == "DELETE":
            return self._delete(preferences, operation, user_id, authorization)
        raise MemoryRepositoryError(f"Unsupported preference action: {action or '<missing>'}")

    def _add(
        self,
        preferences: list[dict[str, Any]],
        operation: dict[str, Any],
        user_id: str,
        authorization: ConsentEvidence,
    ) -> dict[str, Any]:
        value = self._validated_record(operation.get("preference"), user_id)
        requested_id = str(value.get("id", "")).strip()
        preference_id = requested_id or f"pref-{uuid.uuid4().hex}"
        if any(item.get("id") == preference_id for item in preferences):
            raise MemoryRepositoryError(f"Preference already exists: {preference_id}")
        now = self._now()
        value.update(
            {
                "id": preference_id,
                "status": ACTIVE_PREFERENCE_STATUS,
                "created_at": now,
                "updated_at": now,
                "consent": [authorization.to_dict()],
                "evidence": copy.deepcopy(operation.get("evidence", [])),
                "lineage": {"merged_from": [], "supersedes": []},
                "revision": 1,
            }
        )
        preferences.append(value)
        return {
            "operation_id": operation["operation_id"],
            "action": "ADDED",
            "preference_id": preference_id,
        }

    def _update(
        self,
        preferences: list[dict[str, Any]],
        operation: dict[str, Any],
        user_id: str,
        authorization: ConsentEvidence,
    ) -> dict[str, Any]:
        target = self._owned(preferences, operation.get("target_preference_id"), user_id)
        if target.get("status") != ACTIVE_PREFERENCE_STATUS:
            raise MemoryRepositoryError("Only active preferences can be updated.")
        replacement = self._validated_record(operation.get("preference"), user_id)
        before = self._snapshot(target)
        metadata = {
            "id",
            "created_at",
            "updated_at",
            "status",
            "consent",
            "evidence",
            "lineage",
            "revision",
            "versions",
        }
        for key in list(target):
            if key not in metadata:
                del target[key]
        target.update(copy.deepcopy(replacement))
        target.setdefault("versions", []).append(before)
        target.setdefault("consent", []).append(authorization.to_dict())
        target.setdefault("evidence", []).extend(copy.deepcopy(operation.get("evidence", [])))
        target["revision"] = int(target.get("revision", 1)) + 1
        target["updated_at"] = self._now()
        return {
            "operation_id": operation["operation_id"],
            "action": "UPDATED",
            "preference_id": target["id"],
        }

    def _merge(
        self,
        preferences: list[dict[str, Any]],
        operation: dict[str, Any],
        user_id: str,
        authorization: ConsentEvidence,
    ) -> dict[str, Any]:
        source_ids = list(dict.fromkeys(map(str, operation.get("source_preference_ids", []))))
        if len(source_ids) < 2:
            raise MemoryRepositoryError("MERGE requires at least two distinct source preferences.")
        sources = [self._owned(preferences, preference_id, user_id) for preference_id in source_ids]
        if any(item.get("status") != ACTIVE_PREFERENCE_STATUS for item in sources):
            raise MemoryRepositoryError("MERGE sources must all be active.")
        if operation.get("lossless") is not True:
            raise MemoryRepositoryError("Automatic semantic merges must be declared lossless.")
        try:
            confidence = float(operation.get("confidence", 0.0))
        except (TypeError, ValueError) as error:
            raise MemoryRepositoryError("MERGE confidence must be numeric.") from error
        if confidence < 0.85:
            raise MemoryRepositoryError("Automatic semantic merge confidence must be at least 0.85.")
        verification = operation.get("semantic_verification")
        if not isinstance(verification, dict):
            raise MemoryRepositoryError("MERGE requires an independent semantic verification.")
        verified_ids = set(map(str, verification.get("source_preference_ids", [])))
        try:
            verification_confidence = float(verification.get("confidence", 0.0))
        except (TypeError, ValueError) as error:
            raise MemoryRepositoryError("Semantic verification confidence must be numeric.") from error
        if (
            verified_ids != set(source_ids)
            or verification.get("equivalent") is not True
            or verification.get("scope_preserved") is not True
            or verification.get("applicability_preserved") is not True
            or verification_confidence < 0.85
        ):
            raise MemoryRepositoryError(
                "MERGE verification did not establish lossless scope-preserving equivalence."
            )

        result_record = self._validated_record(operation.get("preference"), user_id)
        target_id = str(operation.get("target_preference_id", "")).strip()
        target = next((item for item in sources if item.get("id") == target_id), None)
        if target is None:
            target_id = f"pref-{uuid.uuid4().hex}"
            target = {
                "id": target_id,
                "user_id": user_id,
                "created_at": self._now(),
                "revision": 0,
                "consent": [],
                "evidence": [],
                "lineage": {"merged_from": [], "supersedes": []},
            }
            preferences.append(target)

        if target in sources:
            target.setdefault("versions", []).append(self._snapshot(target))
        target.update(copy.deepcopy(result_record))
        target["id"] = target_id
        target["user_id"] = user_id
        target["status"] = ACTIVE_PREFERENCE_STATUS
        target["updated_at"] = self._now()
        target["revision"] = int(target.get("revision", 0)) + 1
        target.setdefault("consent", [])
        target.setdefault("evidence", [])
        target.setdefault("lineage", {"merged_from": [], "supersedes": []})

        for source in sources:
            for consent in source.get("consent", []):
                if consent not in target["consent"]:
                    target["consent"].append(copy.deepcopy(consent))
            for evidence in source.get("evidence", []):
                if evidence not in target["evidence"]:
                    target["evidence"].append(copy.deepcopy(evidence))
            if source["id"] != target_id:
                source["status"] = "merged"
                source["merged_into"] = target_id
                source["updated_at"] = self._now()
                if source["id"] not in target["lineage"]["merged_from"]:
                    target["lineage"]["merged_from"].append(source["id"])

        return {
            "operation_id": operation["operation_id"],
            "action": "MERGED",
            "preference_id": target_id,
            "source_preference_ids": source_ids,
        }

    def _retract(
        self,
        preferences: list[dict[str, Any]],
        operation: dict[str, Any],
        user_id: str,
        authorization: ConsentEvidence,
    ) -> dict[str, Any]:
        target = self._owned(preferences, operation.get("target_preference_id"), user_id)
        target["status"] = "retracted"
        target["updated_at"] = self._now()
        target.setdefault("consent", []).append(authorization.to_dict())
        return {
            "operation_id": operation["operation_id"],
            "action": "RETRACTED",
            "preference_id": target["id"],
        }

    def _delete(
        self,
        preferences: list[dict[str, Any]],
        operation: dict[str, Any],
        user_id: str,
        authorization: ConsentEvidence,
    ) -> dict[str, Any]:
        if authorization.kind != "forget_confirmation":
            raise MemoryRepositoryError("DELETE requires explicit forget_confirmation consent.")
        target = self._owned(preferences, operation.get("target_preference_id"), user_id)
        if target.get("status") == "deleted":
            raise MemoryRepositoryError("Preference is already deleted.")
        target.setdefault("versions", []).append(self._snapshot(target))
        target["status"] = "deleted"
        target["deleted_at"] = self._now()
        target["updated_at"] = target["deleted_at"]
        target.setdefault("consent", []).append(authorization.to_dict())
        target["revision"] = int(target.get("revision", 1)) + 1
        return {
            "operation_id": operation["operation_id"],
            "action": "DELETED",
            "preference_id": target["id"],
        }

    @staticmethod
    def _validated_record(value: Any, user_id: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise MemoryRepositoryError("Preference operation requires a preference object.")
        statement = str(value.get("statement", "")).strip()
        if not statement:
            raise MemoryRepositoryError("Preference requires a semantic statement.")
        reserved = {
            "id",
            "user_id",
            "status",
            "created_at",
            "updated_at",
            "consent",
            "evidence",
            "lineage",
            "revision",
            "versions",
            "merged_into",
            "deleted_at",
        }
        result = {
            key: copy.deepcopy(item)
            for key, item in value.items()
            if key not in reserved
        }
        result["statement"] = statement
        result["user_id"] = user_id
        result.setdefault("scope", "contextual")
        result.setdefault("applicability", {})
        result.setdefault("structured_value", None)
        if not isinstance(result["applicability"], dict):
            raise MemoryRepositoryError("Preference applicability must be an object.")
        return result

    @staticmethod
    def _require_write_consent(authorization: ConsentEvidence) -> None:
        if authorization.kind not in {
            "explicit_future_language",
            "memory_confirmation",
        }:
            raise MemoryRepositoryError(
                "Preference mutation requires explicit user memory authorization."
            )

    @staticmethod
    def _owned(
        preferences: list[dict[str, Any]], preference_id: Any, user_id: str
    ) -> dict[str, Any]:
        for record in preferences:
            if record.get("id") == preference_id and record.get("user_id") == user_id:
                return record
        raise MemoryRepositoryError(
            f"Unknown preference {preference_id!r} for user {user_id!r}."
        )

    @staticmethod
    def _snapshot(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: copy.deepcopy(value)
            for key, value in record.items()
            if key not in {"versions"}
        }

    @property
    def version(self) -> int:
        with self._lock:
            data = self._read()
            self._validate(data)
            return int(data["version"])

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        if data.get("schema_version") != PREFERENCE_SCHEMA_VERSION:
            raise MemoryRepositoryError("Unsupported preference schema version.")
        if not isinstance(data.get("preferences"), list) or not isinstance(
            data.get("transactions"), dict
        ):
            raise MemoryRepositoryError("Malformed preference repository.")
        if not isinstance(data.get("version"), int):
            raise MemoryRepositoryError("Preference version must be an integer.")
        records_by_id: dict[str, dict[str, Any]] = {}
        for record in data["preferences"]:
            if not isinstance(record, dict):
                raise MemoryRepositoryError("Preference records must be objects.")
            preference_id = str(record.get("id", "")).strip()
            user_id = str(record.get("user_id", "")).strip()
            status = str(record.get("status", "")).lower()
            if not preference_id or not user_id or preference_id in records_by_id:
                raise MemoryRepositoryError("Preference IDs and users must be valid and unique.")
            if status not in {"active", "merged", "retracted", "deleted"}:
                raise MemoryRepositoryError(f"Unsupported preference status: {status}")
            if status == "active":
                consent = record.get("consent")
                if not isinstance(consent, list) or not consent:
                    raise MemoryRepositoryError("Every active preference requires consent provenance.")
            if not isinstance(record.get("revision"), int) or int(record["revision"]) < 1:
                raise MemoryRepositoryError("Preference revision must be a positive integer.")
            records_by_id[preference_id] = record
        for record in records_by_id.values():
            if record.get("status") == "merged":
                target = records_by_id.get(str(record.get("merged_into", "")))
                if target is None or target.get("user_id") != record.get("user_id"):
                    raise MemoryRepositoryError("Merged preference redirect has no owned target.")


class HistoryOutboxRepository(_AtomicJsonRepository):
    """Durable, idempotent retry queue for failed history persistence."""

    def __init__(self, path: str | Path):
        super().__init__(
            path,
            lambda: {
                "schema_version": HISTORY_OUTBOX_SCHEMA_VERSION,
                "version": 0,
                "pending": [],
            },
        )

    def enqueue(
        self,
        source: dict[str, Any],
        error: str,
        *,
        replace_source: bool = False,
    ) -> dict[str, Any]:
        episode_id = str(source.get("episode_id", "")).strip()
        user_id = str(source.get("user_id", "")).strip()
        if not episode_id or not user_id:
            raise MemoryRepositoryError("History retry requires episode_id and user_id.")
        with self._lock, self._exclusive_file_lock():
            data = self._read()
            self._validate(data)
            existing = next(
                (item for item in data["pending"] if item.get("episode_id") == episode_id),
                None,
            )
            if existing is None:
                existing = {
                    "episode_id": episode_id,
                    "user_id": user_id,
                    "source": copy.deepcopy(source),
                    "attempts": 1,
                    "first_failed_at": self._now(),
                }
                data["pending"].append(existing)
            else:
                if existing.get("source") != source and not replace_source:
                    raise MemoryRepositoryError(
                        f"History outbox conflict for {episode_id}: retry source changed."
                    )
                if replace_source:
                    existing["source"] = copy.deepcopy(source)
                existing["attempts"] = int(existing.get("attempts", 0)) + 1
            existing["last_error"] = str(error)
            existing["last_failed_at"] = self._now()
            data["version"] += 1
            self._validate(data)
            self._write(data)
            return copy.deepcopy(existing)

    def list_pending(self, user_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read()
            self._validate(data)
        pending = data["pending"]
        if user_id is not None:
            pending = [item for item in pending if item.get("user_id") == user_id]
        return copy.deepcopy(pending)

    def remove(self, episode_id: str) -> None:
        with self._lock, self._exclusive_file_lock():
            data = self._read()
            self._validate(data)
            remaining = [
                item for item in data["pending"] if item.get("episode_id") != episode_id
            ]
            if len(remaining) == len(data["pending"]):
                return
            data["pending"] = remaining
            data["version"] += 1
            self._validate(data)
            self._write(data)

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        if data.get("schema_version") != HISTORY_OUTBOX_SCHEMA_VERSION:
            raise MemoryRepositoryError("Unsupported history outbox schema version.")
        if not isinstance(data.get("version"), int) or not isinstance(
            data.get("pending"), list
        ):
            raise MemoryRepositoryError("Malformed history outbox repository.")
        ids: set[str] = set()
        for item in data["pending"]:
            if not isinstance(item, dict):
                raise MemoryRepositoryError("History outbox items must be objects.")
            episode_id = str(item.get("episode_id", ""))
            if not episode_id or episode_id in ids or not isinstance(item.get("source"), dict):
                raise MemoryRepositoryError("History outbox IDs must be valid and unique.")
            ids.add(episode_id)
