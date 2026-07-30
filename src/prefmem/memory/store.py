"""Atomic, human-readable Markdown persistence for PrefMem memory."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from prefmem.agents.contracts import (
    EpisodeRecord,
    PreferenceRecord,
    PreferenceStatus,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]


RecordT = TypeVar("RecordT", bound=BaseModel)
_FENCE_PREFIX = "```prefmem-"
_KIND_PATTERN = re.compile(r"^[a-z-]+$")
_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


class MemoryFormatError(RuntimeError):
    """Raised when a canonical memory file contains an invalid record."""


@contextmanager
def _exclusive_path_lock(path: Path):
    """Serialize a complete read-modify-write across store instances/processes."""

    with _PATH_LOCKS_GUARD:
        process_lock = _PATH_LOCKS.setdefault(path, threading.RLock())
    with process_lock:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_path = path.with_name(f".{path.name}.lock")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _record_block(kind: str, record: BaseModel) -> str:
    body = json.dumps(
        record.model_dump(mode="json"),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    record_id = getattr(record, "id", "record")
    return f"## `{record_id}`\n\n```prefmem-{kind}\n{body}\n```"


def _parse_records(
    text: str,
    *,
    kind: str,
    schema: type[RecordT],
    source: Path,
) -> list[RecordT]:
    """Parse every PrefMem fence or reject the complete canonical file.

    A regex that only selects well-formed blocks is unsafe here: a truncated or
    malformed block would disappear from the parsed result and the next write
    would permanently erase it.  Scan the fences explicitly so any PrefMem
    marker must be complete, of the expected kind, valid JSON, and unique by ID.
    """

    records: list[RecordT] = []
    seen_ids: set[str] = set()
    lines = text.splitlines()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        stripped = line.strip()
        if _FENCE_PREFIX not in line:
            line_index += 1
            continue
        if not stripped.startswith(_FENCE_PREFIX):
            raise MemoryFormatError(
                f"Malformed PrefMem fence at {source}:{line_index + 1}"
            )

        fence_kind = stripped.removeprefix(_FENCE_PREFIX)
        if not _KIND_PATTERN.fullmatch(fence_kind):
            raise MemoryFormatError(
                f"Malformed PrefMem fence at {source}:{line_index + 1}"
            )
        if fence_kind != kind:
            raise MemoryFormatError(
                f"Unexpected prefmem-{fence_kind} record in {source}; "
                f"expected prefmem-{kind}"
            )

        body_start = line_index + 1
        line_index = body_start
        while line_index < len(lines) and lines[line_index].strip() != "```":
            if lines[line_index].strip().startswith(_FENCE_PREFIX):
                raise MemoryFormatError(
                    f"Nested or unclosed PrefMem fence at "
                    f"{source}:{line_index + 1}"
                )
            line_index += 1
        if line_index >= len(lines):
            raise MemoryFormatError(
                f"Truncated prefmem-{kind} record in {source}: "
                f"opening fence at line {body_start}"
            )

        body = "\n".join(lines[body_start:line_index])
        try:
            payload = json.loads(body)
            record = schema.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise MemoryFormatError(
                f"Invalid {kind} record in {source}: {exc}"
            ) from exc
        record_id = str(getattr(record, "id", ""))
        if record_id in seen_ids:
            raise MemoryFormatError(
                f"Duplicate {kind} record ID {record_id!r} in {source}"
            )
        seen_ids.add(record_id)
        records.append(record)
        line_index += 1
    return records


class MarkdownMemoryStore:
    """Persist preferences and episodes as canonical fenced Markdown records."""

    def __init__(
        self,
        preference_path: Path,
        *,
        history_path: Path | None = None,
    ) -> None:
        preference_path = Path(preference_path).expanduser()
        if preference_path.suffix.lower() != ".md":
            preference_path = preference_path / "preferences.md"
        self.preference_path = preference_path.resolve()
        self.history_path = (
            Path(history_path).expanduser().resolve()
            if history_path is not None
            else self.preference_path.with_name(
                f"{self.preference_path.stem}.history.md"
            )
        )
        self._lock = threading.RLock()

    @staticmethod
    def default_path(username: str, root: Path = Path(".prefmem/memory")) -> Path:
        original = username.strip()
        safe_username = re.sub(
            r"[^a-zA-Z0-9._-]+",
            "-",
            original,
        ).strip("-")
        if not safe_username or safe_username in {".", ".."}:
            raise ValueError("username must contain at least one safe character")
        if safe_username != original:
            digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
            safe_username = f"{safe_username}-{digest}"
        return Path(root) / safe_username / "preferences.md"

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _render_preferences(records: Iterable[PreferenceRecord]) -> str:
        ordered = sorted(records, key=lambda record: (record.created_at, record.id))
        blocks = "\n\n".join(
            _record_block("preference", record) for record in ordered
        )
        preamble = (
            "# PrefMem preference memory\n\n"
            "This Markdown file is the canonical preference store. Each fenced "
            "`prefmem-preference` block is a machine-validated record. Preferences "
            "are written only after an explicit user request or confirmation.\n"
        )
        return f"{preamble}\n{blocks}\n" if blocks else f"{preamble}\n"

    @staticmethod
    def _render_episodes(records: Iterable[EpisodeRecord]) -> str:
        ordered = sorted(records, key=lambda record: (record.timestamp, record.id))
        blocks = "\n\n".join(_record_block("episode", record) for record in ordered)
        preamble = (
            "# PrefMem episodic history\n\n"
            "This Markdown file contains compact, outcome-grounded episode records. "
            "History is evidence, not an automatically approved preference.\n"
        )
        return f"{preamble}\n{blocks}\n" if blocks else f"{preamble}\n"

    def load_preferences(
        self,
        *,
        username: str | None = None,
        active_only: bool = True,
    ) -> list[PreferenceRecord]:
        records = _parse_records(
            self._read(self.preference_path),
            kind="preference",
            schema=PreferenceRecord,
            source=self.preference_path,
        )
        if username is not None:
            records = [record for record in records if record.username == username]
        if active_only:
            records = [
                record
                for record in records
                if record.status == PreferenceStatus.ACTIVE
            ]
        return records

    def save_preference(self, preference: PreferenceRecord) -> None:
        with self._lock, _exclusive_path_lock(self.preference_path):
            records = self.load_preferences(active_only=False)
            by_id = {record.id: record for record in records}
            existing = by_id.get(preference.id)
            if existing is not None and existing.username != preference.username:
                raise ValueError(
                    f"preference {preference.id!r} belongs to another username"
                )
            by_id[preference.id] = preference
            self._atomic_write(
                self.preference_path,
                self._render_preferences(by_id.values()),
            )

    def commit_preference_revision(
        self,
        *,
        expected: PreferenceRecord,
        replacements: Iterable[PreferenceRecord],
    ) -> None:
        """Atomically replace one exact revision while preserving other writers."""

        values = list(replacements)
        if not values:
            raise ValueError("a preference revision requires a replacement")
        ids = [record.id for record in values]
        if len(ids) != len(set(ids)):
            raise ValueError("replacement preference IDs must be unique")
        if any(record.username != expected.username for record in values):
            raise ValueError("replacement preferences must preserve username")

        with self._lock, _exclusive_path_lock(self.preference_path):
            records = self.load_preferences(active_only=False)
            by_id = {record.id: record for record in records}
            current = by_id.get(expected.id)
            if current is None:
                raise RuntimeError(
                    f"preference {expected.id!r} changed before revision commit"
                )
            if current.model_dump(mode="json") != expected.model_dump(mode="json"):
                raise RuntimeError(
                    f"preference {expected.id!r} changed before revision commit"
                )
            del by_id[expected.id]
            for replacement in values:
                collision = by_id.get(replacement.id)
                if collision is not None:
                    raise ValueError(
                        f"replacement preference ID {replacement.id!r} "
                        "already exists"
                    )
                by_id[replacement.id] = replacement
            self._atomic_write(
                self.preference_path,
                self._render_preferences(by_id.values()),
            )

    def replace_preferences(self, preferences: Iterable[PreferenceRecord]) -> None:
        values = list(preferences)
        ids = [record.id for record in values]
        if len(ids) != len(set(ids)):
            raise ValueError("preference IDs must be unique")
        with self._lock, _exclusive_path_lock(self.preference_path):
            self._atomic_write(
                self.preference_path,
                self._render_preferences(values),
            )

    def load_episodes(
        self,
        *,
        username: str | None = None,
    ) -> list[EpisodeRecord]:
        records = _parse_records(
            self._read(self.history_path),
            kind="episode",
            schema=EpisodeRecord,
            source=self.history_path,
        )
        if username is not None:
            records = [record for record in records if record.username == username]
        return records

    def save_episode(self, episode: EpisodeRecord) -> None:
        with self._lock, _exclusive_path_lock(self.history_path):
            records = self.load_episodes()
            by_id = {record.id: record for record in records}
            existing = by_id.get(episode.id)
            if existing is not None and existing.username != episode.username:
                raise ValueError(
                    f"episode {episode.id!r} belongs to another username"
                )
            if existing is not None:
                if (
                    existing.model_dump(mode="json")
                    != episode.model_dump(mode="json")
                ):
                    raise ValueError(
                        f"episode {episode.id!r} is immutable and already exists"
                    )
                return
            by_id[episode.id] = episode
            self._atomic_write(
                self.history_path,
                self._render_episodes(by_id.values()),
            )
