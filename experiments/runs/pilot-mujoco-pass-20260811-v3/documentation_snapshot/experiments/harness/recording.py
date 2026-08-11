"""Durable, append-only provenance primitives for RoboPref experiments.

The classes in this module deliberately do not depend on the robot runtime.  A
runner can therefore open an attempt and begin recording *before* it resets a
scene or contacts a model service.  Every append is protected by a process-local
lock and fsynced before the call returns.  Attempt directories are immutable by
construction: :meth:`AttemptRecorder.create` refuses to reuse an existing path.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

import cv2
import numpy as np


class RecordingError(RuntimeError):
    """Raised when provenance cannot be persisted without ambiguity."""


class RunStatus(StrEnum):
    """Protocol-level attempt statuses (kept separate from oracle verdicts)."""

    VALID_PASS = "VALID_PASS"
    VALID_SYSTEM_FAILURE = "VALID_SYSTEM_FAILURE"
    INFRA_INTERRUPTED = "INFRA_INTERRUPTED"
    INVALID_HARNESS = "INVALID_HARNESS"
    ABORTED_SAFETY = "ABORTED_SAFETY"
    NOT_RUN = "NOT_RUN"


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EVENT_ID = re.compile(r"^E(?P<number>[0-9]{6,})$")
_SECRET_FIELD_FRAGMENTS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "password",
        "private_key",
        "secret",
        "ssh_key",
        "token",
    }
)
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?-----END [^-\n]*PRIVATE KEY-----",
    flags=re.DOTALL,
)
_BEARER_VALUE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(?P<prefix>[\"']?[A-Za-z0-9_-]*(?:api[_-]?key|authorization|cookie|"
    r"credential|password|private[_-]?key|secret|ssh[_-]?key|token)"
    r"[A-Za-z0-9_-]*[\"']?\s*[:=]\s*)"
    r"(?P<value>\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|[^\s,;}]+)"
)


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(timezone.utc)


def format_utc(value: datetime) -> str:
    """Render UTC in an unambiguous RFC 3339 form with microseconds."""

    if value.tzinfo is None:
        raise ValueError("UTC timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True, slots=True)
class ClockOrigin:
    """One shared mapping between monotonic time and UTC/video elapsed time."""

    utc: datetime
    monotonic_ns: int

    def __post_init__(self) -> None:
        if self.utc.tzinfo is None:
            raise ValueError("clock origin UTC must be timezone-aware")
        if isinstance(self.monotonic_ns, bool) or self.monotonic_ns < 0:
            raise ValueError("monotonic_ns must be a non-negative integer")

    @classmethod
    def capture(cls) -> "ClockOrigin":
        # Read monotonic first: the UTC instant is no earlier than this bound and
        # all subsequent timestamps use the same monotonic source.
        monotonic_ns = time.monotonic_ns()
        return cls(utc=utc_now(), monotonic_ns=monotonic_ns)

    @property
    def utc_iso(self) -> str:
        return format_utc(self.utc)

    def stamp(self, monotonic_ns: int | None = None) -> dict[str, Any]:
        actual_ns = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        if actual_ns < self.monotonic_ns:
            raise ValueError("timestamp predates the shared monotonic origin")
        elapsed = (actual_ns - self.monotonic_ns) / 1_000_000_000
        instant = self.utc + timedelta(seconds=elapsed)
        return {
            "utc": format_utc(instant),
            "monotonic_ns": actual_ns,
            "elapsed_seconds": elapsed,
        }

    def to_json(self) -> dict[str, Any]:
        return {"utc": self.utc_iso, "monotonic_ns": self.monotonic_ns}


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
        options["separators"] = (",", ": ")
    else:
        options["separators"] = (",", ":")
    try:
        return (json.dumps(value, **options) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RecordingError(f"value is not strict JSON: {error}") from error


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_write(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    """Write and fsync one artifact, replacing only when explicitly allowed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    destination = path
    if exclusive:
        write_path = destination
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    else:
        # Mutable snapshots (recording state/checksums) are replaced only after
        # the new file is fully fsynced.  A process death leaves the previous
        # valid snapshot plus an explicitly named partial, never a torn file.
        write_path = destination.with_name(
            f".{destination.name}.{os.getpid()}.{threading.get_ident()}."
            f"{time.monotonic_ns()}.partial"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(write_path, flags, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RecordingError(f"short write while persisting {destination}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if not exclusive:
        os.replace(write_path, destination)
    _fsync_directory(destination.parent)


def durable_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    durable_write(path, _json_bytes(value, pretty=True), exclusive=exclusive)


def _field_is_secret(name: object, configured: frozenset[str]) -> bool:
    normalized = str(name).casefold().replace("-", "_")
    return normalized in configured or any(
        fragment in normalized for fragment in _SECRET_FIELD_FRAGMENTS
    )


def redact_value(
    value: Any,
    *,
    secret_fields: Iterable[str] = (),
    secret_values: Iterable[str] = (),
    _path: str = "$",
) -> tuple[Any, list[str]]:
    """Recursively redact secret-valued fields and return their JSON paths.

    The audit contains paths only.  Neither hashes nor prefixes of the removed
    values are retained, because those can themselves disclose credentials.
    """

    configured = frozenset(item.casefold().replace("-", "_") for item in secret_fields)
    configured_values = tuple(item for item in secret_values if item)

    def walk(item: Any, path: str) -> tuple[Any, list[str]]:
        if isinstance(item, Mapping):
            clean: dict[str, Any] = {}
            changed: list[str] = []
            for key, child in item.items():
                text_key = str(key)
                child_path = f"{path}.{text_key}"
                if _field_is_secret(text_key, configured):
                    clean[text_key] = "[REDACTED]"
                    changed.append(child_path)
                else:
                    clean_child, child_changed = walk(child, child_path)
                    clean[text_key] = clean_child
                    changed.extend(child_changed)
            return clean, changed
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            clean_items: list[Any] = []
            changed = []
            for index, child in enumerate(item):
                clean_child, child_changed = walk(child, f"{path}[{index}]")
                clean_items.append(clean_child)
                changed.extend(child_changed)
            return clean_items, changed
        if isinstance(item, str):
            redacted = redact_text(item, secret_values=configured_values)
            if redacted != item:
                return redacted, [path]
        return item, []

    return walk(value, _path)


def redact_text(text: str, *, secret_values: Iterable[str] = ()) -> str:
    """Redact terminal/video text without writing secret values to an audit."""

    clean = _PRIVATE_KEY_BLOCK.sub("[REDACTED PRIVATE KEY]", str(text))
    clean = _BEARER_VALUE.sub(r"\1[REDACTED]", clean)
    clean = _SENSITIVE_ASSIGNMENT.sub(r'\g<prefix>"[REDACTED]"', clean)
    for secret in secret_values:
        if secret:
            clean = clean.replace(secret, "[REDACTED]")
    return clean


class JsonlJournal:
    """Thread-safe append-only strict-JSON journal with an fsync per record."""

    def __init__(self, path: Path, *, origin: ClockOrigin) -> None:
        self.path = Path(path)
        self.origin = origin
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.close(descriptor)
        _fsync_directory(self.path.parent)
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            actual = dict(record)
            stamp = self.origin.stamp()
            actual.setdefault("utc", stamp["utc"])
            actual.setdefault("monotonic_ns", stamp["monotonic_ns"])
            actual.setdefault("elapsed_seconds", stamp["elapsed_seconds"])
            data = _json_bytes(actual)
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
            try:
                view = memoryview(data)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise RecordingError(f"short append to {self.path}")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return actual


class EventJournal(JsonlJournal):
    """Append-only event source of truth with stable ``E000001`` identifiers."""

    def __init__(
        self,
        path: Path,
        *,
        origin: ClockOrigin,
        secret_fields: Iterable[str] = (),
        secret_values: Iterable[str] = (),
        redaction_journal: JsonlJournal | None = None,
    ) -> None:
        super().__init__(path, origin=origin)
        self._next_id = self._discover_next_id()
        self.secret_fields = tuple(secret_fields)
        self.secret_values = tuple(secret_values)
        self.redaction_journal = redaction_journal

    def _discover_next_id(self) -> int:
        next_id = 1
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RecordingError(
                        f"cannot resume malformed event journal at line {line_number}"
                    ) from error
                match = _EVENT_ID.fullmatch(str(record.get("event_id", "")))
                if match is None or int(match.group("number")) != next_id:
                    raise RecordingError(
                        f"event journal is not contiguous at line {line_number}"
                    )
                next_id += 1
        return next_id

    def append(self, event_type: str, **fields: Any) -> dict[str, Any]:  # type: ignore[override]
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be a non-empty string")
        forbidden = {
            "event_id",
            "event_type",
            "utc",
            "monotonic_ns",
            "elapsed_seconds",
            "video_time_seconds",
        }
        overlap = forbidden.intersection(fields)
        if overlap:
            raise ValueError(f"event fields override reserved keys: {sorted(overlap)}")
        with self._lock:
            event_id = f"E{self._next_id:06d}"
            self._next_id += 1
            clean_fields, redacted_paths = redact_value(
                fields,
                secret_fields=self.secret_fields,
                secret_values=self.secret_values,
            )
            stamp = self.origin.stamp()
            record = {
                "event_id": event_id,
                "event_type": event_type.strip(),
                **stamp,
                "video_time_seconds": stamp["elapsed_seconds"],
                **clean_fields,
            }
            data = _json_bytes(record)
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
            try:
                view = memoryview(data)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise RecordingError(f"short append to {self.path}")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if redacted_paths and self.redaction_journal is not None:
                self.redaction_journal.append(
                    {
                        "artifact": "events.jsonl",
                        "event_id": event_id,
                        "redacted_json_paths": redacted_paths,
                        "redaction_count": len(redacted_paths),
                    }
                )
            return record


def _safe_component(value: str, label: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{label} must match {_SAFE_COMPONENT.pattern}")
    if value in {".", ".."}:
        raise ValueError(f"{label} may not be a traversal component")
    return value


def safe_attempt_directory(
    campaign_root: Path,
    schedule_id: str,
    attempt_number: int,
) -> Path:
    """Create a new, non-symlinked attempt directory below ``episodes``."""

    schedule = _safe_component(schedule_id, "schedule_id")
    if isinstance(attempt_number, bool) or not isinstance(attempt_number, int):
        raise TypeError("attempt_number must be an integer")
    if attempt_number < 1 or attempt_number > 999_999:
        raise ValueError("attempt_number must be between 1 and 999999")
    supplied_root = Path(campaign_root)
    if supplied_root.is_symlink():
        raise RecordingError("campaign root may not be a symlink")
    root = supplied_root.resolve()
    episodes = root / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    if episodes.is_symlink():
        raise RecordingError("episodes directory may not be a symlink")
    schedule_dir = episodes / schedule
    schedule_dir.mkdir(mode=0o755, exist_ok=True)
    if schedule_dir.is_symlink():
        raise RecordingError("schedule directory may not be a symlink")
    attempt = schedule_dir / f"attempt_{attempt_number}"
    try:
        attempt.mkdir(mode=0o755, exist_ok=False)
    except FileExistsError as error:
        raise RecordingError(f"refusing to reuse attempt directory: {attempt}") from error
    if attempt.resolve().parent != schedule_dir.resolve():
        raise RecordingError("attempt path escaped its schedule directory")
    _fsync_directory(schedule_dir)
    return attempt


def _image_suffix(raw: bytes, media_type: str | None) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return ".webp"
    guessed = mimetypes.guess_extension(media_type or "")
    return ".jpg" if guessed == ".jpe" else (guessed or ".img")


def decode_image_payload(payload: bytes | str) -> tuple[bytes, bytes, str | None]:
    """Decode exactly the image payload sent to a model.

    Returns ``(raw_image_bytes, exact_encoded_payload_bytes, media_type)``.  A
    string must be a data URL or strict base64; arbitrary paths/URLs are never
    dereferenced and fabricated placeholder pixels are never accepted.
    """

    if isinstance(payload, bytes):
        raw = payload
        encoded = payload
        media_type = None
    elif isinstance(payload, str):
        encoded = payload.encode("utf-8")
        media_type = None
        body = payload
        if payload.startswith("data:"):
            header, separator, body = payload.partition(",")
            if not separator or ";base64" not in header.casefold():
                raise RecordingError("image data URL must use base64 encoding")
            media_type = header[5:].split(";", 1)[0].casefold() or None
        try:
            raw = base64.b64decode(body, validate=True)
        except (ValueError, binascii.Error) as error:
            raise RecordingError("image payload is not strict base64") from error
    else:
        raise TypeError("image payload must be bytes or a base64 string")
    if not raw:
        raise RecordingError("image payload is empty")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in {2, 3}:
        raise RecordingError("payload does not decode as an image")
    return raw, encoded, media_type


class FrameRequestRecorder:
    """Content-deduplicated exact request-frame storage and logical mapping."""

    def __init__(self, attempt_dir: Path, *, origin: ClockOrigin) -> None:
        self.attempt_dir = Path(attempt_dir)
        self.frames_dir = self.attempt_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.journal = JsonlJournal(
            self.attempt_dir / "frame_requests.jsonl", origin=origin
        )
        self.origin = origin
        self._request_ids: set[str] = set()
        self._lock = threading.Lock()

    def record(
        self,
        *,
        request_id: str,
        logical_agent: str,
        payload: bytes | str,
        source_payload: bytes | str | None = None,
        camera_view_id: str,
        frame_sequence_number: int,
        prompt_or_request_body: bytes | str | Mapping[str, Any],
        source_frame_sha256: str | None = None,
        capture_monotonic_ns: int | None = None,
        response_id: str | None = None,
        event_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        request = _safe_component(request_id, "request_id")
        if not logical_agent.strip() or not camera_view_id.strip():
            raise ValueError("logical_agent and camera_view_id must be non-empty")
        if (
            isinstance(frame_sequence_number, bool)
            or not isinstance(frame_sequence_number, int)
            or frame_sequence_number < 0
        ):
            raise ValueError("frame_sequence_number must be a non-negative integer")
        if source_frame_sha256 is not None and re.fullmatch(
            r"[0-9a-f]{64}", source_frame_sha256
        ) is None:
            raise ValueError("source_frame_sha256 must be a lowercase SHA-256 digest")
        raw, encoded, media_type = decode_image_payload(payload)
        raw_hash = sha256_bytes(raw)
        encoded_hash = sha256_bytes(encoded)
        suffix = _image_suffix(raw, media_type)
        image_path = self.frames_dir / f"{raw_hash}{suffix}"
        if not image_path.exists():
            try:
                durable_write(image_path, raw, exclusive=True)
            except FileExistsError:
                pass
        elif sha256_file(image_path) != raw_hash:
            raise RecordingError(f"deduplicated frame collision at {image_path}")

        source_image_path: Path | None = None
        source_raw_hash: str | None = None
        if source_payload is not None:
            source_raw, _source_encoded, source_media_type = decode_image_payload(
                source_payload
            )
            source_raw_hash = sha256_bytes(source_raw)
            source_suffix = _image_suffix(source_raw, source_media_type)
            source_image_path = self.frames_dir / f"{source_raw_hash}{source_suffix}"
            if not source_image_path.exists():
                try:
                    durable_write(source_image_path, source_raw, exclusive=True)
                except FileExistsError:
                    pass
            elif sha256_file(source_image_path) != source_raw_hash:
                raise RecordingError(
                    f"deduplicated source-frame collision at {source_image_path}"
                )

        if isinstance(prompt_or_request_body, bytes):
            body_bytes = prompt_or_request_body
        elif isinstance(prompt_or_request_body, str):
            body_bytes = prompt_or_request_body.encode("utf-8")
        else:
            body_bytes = _json_bytes(prompt_or_request_body).rstrip(b"\n")
        capture = self.origin.stamp(capture_monotonic_ns)
        relative_path = image_path.relative_to(self.attempt_dir).as_posix()
        source_relative_path = (
            None
            if source_image_path is None
            else source_image_path.relative_to(self.attempt_dir).as_posix()
        )
        camera_frame_id = f"{camera_view_id}:{frame_sequence_number}"
        with self._lock:
            if request in self._request_ids:
                raise RecordingError(f"duplicate frame request ID: {request}")
            self._request_ids.add(request)
            record = self.journal.append(
                {
                    "record_type": "frame_request",
                    "request_id": request,
                    "logical_agent": logical_agent,
                    "capture_utc": capture["utc"],
                    "capture_monotonic_ns": capture["monotonic_ns"],
                    "capture_elapsed_seconds": capture["elapsed_seconds"],
                    "camera_view_id": camera_view_id,
                    "frame_sequence_number": frame_sequence_number,
                    "camera_frame_id": camera_frame_id,
                    "image_path": relative_path,
                    "raw_image_sha256": raw_hash,
                    "source_image_path": source_relative_path,
                    "source_raw_image_sha256": source_raw_hash,
                    "source_frame_sha256": source_frame_sha256,
                    "encoded_request_payload_sha256": encoded_hash,
                    "prompt_request_body_sha256": sha256_bytes(body_bytes),
                    "response_id": response_id,
                    "event_ids": list(event_ids),
                }
            )
        return record

    def link_response(
        self,
        request_id: str,
        *,
        response_id: str,
        event_ids: Sequence[str],
    ) -> dict[str, Any]:
        request = _safe_component(request_id, "request_id")
        if request not in self._request_ids:
            raise RecordingError(f"cannot link unknown frame request {request}")
        if not response_id.strip() or not event_ids:
            raise ValueError("response_id and at least one event_id are required")
        return self.journal.append(
            {
                "record_type": "frame_response_link",
                "request_id": request,
                "response_id": response_id,
                "event_ids": list(event_ids),
            }
        )


@dataclass(slots=True)
class _ActiveModelCall:
    started_monotonic_ns: int
    request_artifact: str
    request_sha256: str


class ModelCallRecorder:
    """Save redacted raw requests/responses, including errors and malformed data."""

    def __init__(
        self,
        attempt_dir: Path,
        *,
        origin: ClockOrigin,
        secret_fields: Iterable[str] = (),
        secret_values: Iterable[str] = (),
    ) -> None:
        self.attempt_dir = Path(attempt_dir)
        self.calls_dir = self.attempt_dir / "model_calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.journal = JsonlJournal(self.attempt_dir / "model_calls.jsonl", origin=origin)
        self.redaction_journal = JsonlJournal(
            self.calls_dir / "redactions.jsonl", origin=origin
        )
        self.origin = origin
        self.secret_fields = tuple(secret_fields)
        self.secret_values = tuple(secret_values)
        self._active: dict[str, _ActiveModelCall] = {}
        self._lock = threading.Lock()

    def _artifact(
        self,
        call_id: str,
        kind: str,
        payload: Any,
    ) -> tuple[str, str, list[str]]:
        # Network clients often expose exact JSON bytes.  Parse those bytes only
        # to redact field values; binary/malformed evidence stays byte-exact
        # except for recognizable textual credential patterns.
        if isinstance(payload, bytes):
            try:
                decoded_text = payload.decode("utf-8")
                decoded_json = json.loads(decoded_text)
            except (UnicodeDecodeError, json.JSONDecodeError):
                try:
                    decoded_text = payload.decode("utf-8")
                except UnicodeDecodeError:
                    decoded_text = ""
                if decoded_text:
                    clean_text = redact_text(
                        decoded_text, secret_values=self.secret_values
                    )
                    paths = ["$<text-pattern>"] if clean_text != decoded_text else []
                    payload = clean_text.encode("utf-8")
                else:
                    paths = []
            else:
                payload = decoded_json
                paths = []
        else:
            paths = []
        if isinstance(payload, str):
            try:
                decoded_json = json.loads(payload)
            except json.JSONDecodeError:
                clean_text = redact_text(payload, secret_values=self.secret_values)
                if clean_text != payload:
                    paths.append("$<text-pattern>")
                payload = clean_text
            else:
                payload = decoded_json
        clean, redacted_paths = redact_value(
            payload,
            secret_fields=self.secret_fields,
            secret_values=self.secret_values,
        )
        redacted_paths = paths + redacted_paths
        if isinstance(clean, bytes):
            extension = "bin"
            data = clean
        elif isinstance(clean, str):
            extension = "txt"
            data = clean.encode("utf-8")
        else:
            extension = "json"
            data = _json_bytes(clean, pretty=True)
        path = self.calls_dir / f"{call_id}.{kind}.{extension}"
        durable_write(path, data, exclusive=True)
        relative = path.relative_to(self.attempt_dir).as_posix()
        if redacted_paths:
            self.redaction_journal.append(
                {
                    "call_id": call_id,
                    "artifact": relative,
                    "redacted_json_paths": redacted_paths,
                    "redaction_count": len(redacted_paths),
                }
            )
        return relative, sha256_bytes(data), redacted_paths

    def start(
        self,
        *,
        call_id: str,
        request_id: str,
        logical_agent: str,
        request: Any,
        model_id: str | None = None,
        service_label: str | None = None,
        frame_request_ids: Sequence[str] = (),
        event_id: str | None = None,
    ) -> dict[str, Any]:
        call = _safe_component(call_id, "call_id")
        request_name = _safe_component(request_id, "request_id")
        artifact, digest, redactions = self._artifact(call, "request", request)
        started = time.monotonic_ns()
        started_stamp = self.origin.stamp(started)
        with self._lock:
            if call in self._active:
                raise RecordingError(f"duplicate active model call: {call}")
            self._active[call] = _ActiveModelCall(started, artifact, digest)
        return self.journal.append(
            {
                "record_type": "model_call_start",
                "call_id": call,
                "request_id": request_name,
                "logical_agent": logical_agent,
                "model_id": model_id,
                "service_label": service_label,
                "frame_request_ids": list(frame_request_ids),
                "event_id": event_id,
                "request_artifact": artifact,
                "request_sha256": digest,
                "redaction_count": len(redactions),
                "call_started_utc": started_stamp["utc"],
                "call_started_monotonic_ns": started,
                "call_started_elapsed_seconds": started_stamp["elapsed_seconds"],
            }
        )

    def _finish(
        self,
        *,
        call_id: str,
        outcome: str,
        payload_kind: str,
        payload: Any,
        response_id: str | None,
        event_id: str | None,
        transport_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        call = _safe_component(call_id, "call_id")
        ended = time.monotonic_ns()
        with self._lock:
            active = self._active.get(call)
        if active is None:
            raise RecordingError(f"model call is not active: {call}")
        artifact, digest, redactions = self._artifact(call, payload_kind, payload)
        with self._lock:
            removed = self._active.pop(call, None)
        if removed is not active:
            raise RecordingError(f"model call was concurrently finalized: {call}")
        ended_stamp = self.origin.stamp(ended)
        clean_transport, transport_redactions = redact_value(
            dict(transport_metadata or {}),
            secret_fields=self.secret_fields,
            secret_values=self.secret_values,
        )
        return self.journal.append(
            {
                "record_type": f"model_call_{outcome}",
                "call_id": call,
                "response_id": response_id,
                "event_id": event_id,
                f"{payload_kind}_artifact": artifact,
                f"{payload_kind}_sha256": digest,
                "duration_seconds": (ended - active.started_monotonic_ns)
                / 1_000_000_000,
                "call_completed_utc": ended_stamp["utc"],
                "call_completed_monotonic_ns": ended,
                "call_completed_elapsed_seconds": ended_stamp["elapsed_seconds"],
                "transport": clean_transport,
                "redaction_count": len(redactions) + len(transport_redactions),
            }
        )

    def end(
        self,
        *,
        call_id: str,
        response: Any,
        response_id: str | None = None,
        event_id: str | None = None,
        transport_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._finish(
            call_id=call_id,
            outcome="end",
            payload_kind="response",
            payload=response,
            response_id=response_id,
            event_id=event_id,
            transport_metadata=transport_metadata,
        )

    def error(
        self,
        *,
        call_id: str,
        error: Any,
        response_id: str | None = None,
        event_id: str | None = None,
        transport_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = error if not isinstance(error, BaseException) else {
            "type": type(error).__name__,
            "message": str(error),
        }
        return self._finish(
            call_id=call_id,
            outcome="error",
            payload_kind="error",
            payload=payload,
            response_id=response_id,
            event_id=event_id,
            transport_metadata=transport_metadata,
        )

    @property
    def active_call_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._active))


_REQUIRED_FILES = (
    "SETUP.md",
    "setup.json",
    "experiment_record.md",
    "diagnosis.md",
    "diagnosis.json",
    "events.jsonl",
    "frame_requests.jsonl",
    "model_calls.jsonl",
    "runtime_states.jsonl",
    "inputs.json",
    "results.json",
    "results.md",
    "terminal.log",
    "initial_state.json",
    "initial.png",
    "video.mp4",
    "robot_camera.mp4",
    "video_metadata.json",
    "artifact_checks.json",
    "checksums.sha256",
)
_REQUIRED_DIRECTORIES = ("frames", "model_calls")
_EMPTY_ALLOWED_FILES = frozenset(
    {
        "events.jsonl",
        "frame_requests.jsonl",
        "model_calls.jsonl",
        "runtime_states.jsonl",
    }
)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append(f"{path.name}:{line_number}: {error.msg}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"{path.name}:{line_number}: record is not an object")
                    continue
                records.append(record)
    except (OSError, UnicodeError) as error:
        errors.append(f"{path.name}: {error}")
    return records, errors


def write_checksums(attempt_dir: Path) -> Path:
    """Write sorted GNU-compatible SHA-256 entries for every regular artifact."""

    root = Path(attempt_dir)
    entries: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise RecordingError(f"refusing to checksum symlink artifact: {path}")
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        relative = path.relative_to(root).as_posix()
        if "\n" in relative or "\r" in relative:
            raise RecordingError("artifact paths may not contain newlines")
        entries.append(f"{sha256_file(path)}  {relative}")
    output = root / "checksums.sha256"
    durable_write(output, ("\n".join(entries) + "\n").encode("utf-8"))
    return output


def verify_checksums(attempt_dir: Path) -> dict[str, Any]:
    root = Path(attempt_dir)
    checksum_path = root / "checksums.sha256"
    errors: list[str] = []
    verified = 0
    covered: set[str] = set()
    if not checksum_path.is_file():
        return {"passed": False, "verified": 0, "errors": ["missing checksums.sha256"]}
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return {"passed": False, "verified": 0, "errors": [str(error)]}
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append(f"invalid checksum syntax on line {line_number}")
            continue
        candidate = root / relative
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            errors.append(f"unsafe or missing checksum target: {relative}")
            continue
        if candidate.is_symlink() or not candidate.is_file():
            errors.append(f"checksum target is not a regular file: {relative}")
            continue
        if relative in covered:
            errors.append(f"duplicate checksum target: {relative}")
            continue
        covered.add(relative)
        actual = sha256_file(candidate)
        if actual != digest:
            errors.append(f"checksum mismatch: {relative}")
        else:
            verified += 1
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "checksums.sha256"
    }
    for missing in sorted(expected - covered):
        errors.append(f"artifact missing from checksums: {missing}")
    return {"passed": not errors, "verified": verified, "errors": errors}


def audit_attempt(
    attempt_dir: Path,
    *,
    verify_video: bool = True,
    require_final_state: bool = False,
    finalizing: bool = False,
) -> dict[str, Any]:
    """Independently validate the attempt contract and provenance linkage."""

    root = Path(attempt_dir)
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[str] = []
    json_values: dict[str, dict[str, Any]] = {}
    if not root.is_dir() or root.is_symlink():
        return {"passed": False, "errors": ["attempt directory missing or unsafe"]}
    for name in _REQUIRED_FILES:
        if finalizing and name in {"artifact_checks.json", "checksums.sha256"}:
            continue
        path = root / name
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing required artifact: {name}")
        elif path.stat().st_size == 0 and name not in _EMPTY_ALLOWED_FILES:
            errors.append(f"empty required artifact: {name}")
        else:
            checked.append(name)
    for name in _REQUIRED_DIRECTORIES:
        path = root / name
        if not path.is_dir() or path.is_symlink():
            errors.append(f"missing required artifact directory: {name}/")
        else:
            checked.append(f"{name}/")
    if require_final_state:
        for name in ("final_state.json", "final.png"):
            if not (root / name).is_file():
                errors.append(f"missing final-state artifact: {name}")

    for name in (
        "setup.json",
        "inputs.json",
        "results.json",
        "diagnosis.json",
        "initial_state.json",
        "final_state.json",
        "video_metadata.json",
        "artifact_checks.json",
        "recording_state.json",
    ):
        path = root / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                errors.append(f"{name} must contain a JSON object")
            else:
                json_values[name] = value
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            errors.append(f"invalid {name}: {error}")
    results = json_values.get("results.json")
    setup_value = json_values.get("setup.json")
    if results is not None:
        try:
            result_status = RunStatus(results.get("run_status"))
        except ValueError:
            errors.append("results.json has an invalid run_status")
        else:
            verdict = results.get("oracle_verdict")
            if result_status in {
                RunStatus.VALID_PASS,
                RunStatus.VALID_SYSTEM_FAILURE,
            } and (not isinstance(verdict, str) or not verdict.strip()):
                errors.append("valid outcomes require a separate oracle verdict")
            if result_status in {
                RunStatus.VALID_PASS,
                RunStatus.VALID_SYSTEM_FAILURE,
            } and (root / "diagnosis.md").is_file():
                try:
                    diagnosis_text = (root / "diagnosis.md").read_text(
                        encoding="utf-8"
                    )
                except (OSError, UnicodeError) as error:
                    errors.append(f"could not inspect diagnosis.md: {error}")
                else:
                    evidence_markers = (
                        "experiment_record.md#event-e",
                        "video.mp4#t=",
                        "./frames/",
                        "./model_calls/",
                    )
                    if not any(marker in diagnosis_text for marker in evidence_markers):
                        errors.append(
                            "valid outcome diagnosis has no stable evidence citation"
                        )
                    sections = re.split(
                        r"(?m)^##\s+(.+?)\s*$", diagnosis_text
                    )[1:]
                    for label, body in zip(sections[0::2], sections[1::2]):
                        if label.strip().casefold() in {
                            "run status",
                            "oracle verdict",
                        }:
                            continue
                        if not any(marker in body for marker in evidence_markers):
                            errors.append(
                                "diagnosis section lacks a stable evidence citation: "
                                + label.strip()
                            )
            if setup_value is not None and setup_value.get("run_status") != result_status.value:
                errors.append("setup.json and results.json run statuses disagree")
    for name in ("initial.png", "final.png"):
        path = root / name
        if not path.is_file():
            continue
        try:
            encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        except OSError as error:
            errors.append(f"could not read {name}: {error}")
            continue
        if image is None or image.ndim not in {2, 3}:
            errors.append(f"{name} is not a decodable image")
    partials = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".partial") or path.name == "video_partial.json")
    )
    if partials:
        errors.append("finalized attempt retains partial artifacts: " + ", ".join(partials))

    events, journal_errors = _read_jsonl(root / "events.jsonl")
    errors.extend(journal_errors)
    previous_elapsed = -math.inf
    for index, event in enumerate(events, start=1):
        expected = f"E{index:06d}"
        if event.get("event_id") != expected:
            errors.append(f"events.jsonl has non-contiguous ID at record {index}")
        elapsed = event.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed):
            errors.append(f"event {expected} has invalid elapsed_seconds")
        elif elapsed < previous_elapsed:
            errors.append(f"event {expected} moves backwards in monotonic time")
        else:
            previous_elapsed = float(elapsed)
        if not isinstance(event.get("monotonic_ns"), int):
            errors.append(f"event {expected} has invalid monotonic_ns")
        if not isinstance(event.get("utc"), str):
            errors.append(f"event {expected} has invalid UTC timestamp")
        video_time = event.get("video_time_seconds")
        if not isinstance(video_time, (int, float)) or not math.isfinite(video_time):
            errors.append(f"event {expected} has invalid video_time_seconds")

    frame_records, frame_errors = _read_jsonl(root / "frame_requests.jsonl")
    errors.extend(frame_errors)
    requests: dict[str, dict[str, Any]] = {}
    links: set[str] = set()
    for record in frame_records:
        request_id = str(record.get("request_id", ""))
        if record.get("record_type") == "frame_request":
            if request_id in requests:
                errors.append(f"duplicate frame request mapping: {request_id}")
                continue
            requests[request_id] = record
            relative = record.get("image_path")
            if not isinstance(relative, str):
                errors.append(f"frame request {request_id} lacks image_path")
                continue
            frame_path = root / relative
            if not frame_path.is_file() or frame_path.is_symlink():
                errors.append(f"frame request {request_id} image is missing")
            elif sha256_file(frame_path) != record.get("raw_image_sha256"):
                errors.append(f"frame request {request_id} raw image hash mismatch")
            source_relative = record.get("source_image_path")
            source_raw_hash = record.get("source_raw_image_sha256")
            if source_relative is not None or source_raw_hash is not None:
                if not isinstance(source_relative, str) or re.fullmatch(
                    r"[0-9a-f]{64}", str(source_raw_hash or "")
                ) is None:
                    errors.append(
                        f"frame request {request_id} has invalid source image metadata"
                    )
                else:
                    source_path = root / source_relative
                    if not source_path.is_file() or source_path.is_symlink():
                        errors.append(
                            f"frame request {request_id} source image is missing"
                        )
                    elif sha256_file(source_path) != source_raw_hash:
                        errors.append(
                            f"frame request {request_id} source image hash mismatch"
                        )
            for key in (
                "raw_image_sha256",
                "encoded_request_payload_sha256",
                "prompt_request_body_sha256",
            ):
                if re.fullmatch(r"[0-9a-f]{64}", str(record.get(key, ""))) is None:
                    errors.append(f"frame request {request_id} has invalid {key}")
            if record.get("response_id") and record.get("event_ids"):
                links.add(request_id)
        elif record.get("record_type") == "frame_response_link":
            if record.get("response_id") and record.get("event_ids"):
                links.add(request_id)
    for request_id in sorted(requests.keys() - links):
        errors.append(f"frame request has no response/event link: {request_id}")

    call_records, call_errors = _read_jsonl(root / "model_calls.jsonl")
    errors.extend(call_errors)
    starts: set[str] = set()
    finishes: set[str] = set()
    for record in call_records:
        call_id = str(record.get("call_id", ""))
        if record.get("record_type") == "model_call_start":
            if call_id in starts:
                errors.append(f"duplicate model call start: {call_id}")
            starts.add(call_id)
            artifact_key = "request_artifact"
            hash_key = "request_sha256"
        elif record.get("record_type") in {"model_call_end", "model_call_error"}:
            if call_id not in starts:
                errors.append(f"model call finished without a start: {call_id}")
            finishes.add(call_id)
            artifact_key = (
                "response_artifact"
                if record.get("record_type") == "model_call_end"
                else "error_artifact"
            )
            hash_key = artifact_key.replace("artifact", "sha256")
        else:
            continue
        relative = record.get(artifact_key)
        if isinstance(relative, str):
            artifact = root / relative
            if not artifact.is_file() or artifact.is_symlink():
                errors.append(f"model call artifact missing: {relative}")
            elif sha256_file(artifact) != record.get(hash_key):
                errors.append(f"model call artifact hash mismatch: {relative}")
    for call_id in sorted(starts - finishes):
        errors.append(f"model call did not finish before finalization: {call_id}")

    record_path = root / "experiment_record.md"
    if record_path.is_file():
        try:
            if record_path.read_text(encoding="utf-8") != _render_record(events):
                errors.append("experiment_record.md is not the exact events.jsonl rendering")
        except (OSError, UnicodeError, ValueError) as error:
            errors.append(f"could not verify experiment_record.md: {error}")
    if results is not None and (root / "results.md").is_file():
        try:
            if (root / "results.md").read_text(encoding="utf-8") != _render_result(results):
                errors.append("results.md is not the exact results.json rendering")
        except (OSError, UnicodeError) as error:
            errors.append(f"could not verify results.md: {error}")
    if setup_value is not None and (root / "SETUP.md").is_file():
        try:
            if (root / "SETUP.md").read_text(encoding="utf-8") != _render_setup(
                setup_value
            ):
                errors.append("SETUP.md is not the exact setup.json rendering")
        except (OSError, UnicodeError) as error:
            errors.append(f"could not verify SETUP.md: {error}")

    if finalizing:
        checksum = {"passed": True, "verified": 0, "errors": [], "deferred": True}
    else:
        checksum = verify_checksums(root)
        if not checksum["passed"]:
            errors.extend(checksum["errors"])

    video_report: dict[str, Any] | None = None
    if verify_video and (root / "video.mp4").is_file():
        try:
            from experiments.harness.video import audit_attempt_videos

            video_report = audit_attempt_videos(root)
            if not video_report["passed"]:
                errors.extend(f"video: {item}" for item in video_report["errors"])
        except Exception as error:  # an auditor failure invalidates the harness
            errors.append(f"video audit raised {type(error).__name__}: {error}")
    return {
        "schema_version": 1,
        "passed": not errors,
        "effective_run_status": (
            None if results is None else results.get("run_status")
        ),
        "checked_artifacts": sorted(checked),
        "event_count": len(events),
        "frame_request_count": len(requests),
        "model_call_count": len(starts),
        "checksum": checksum,
        "video": video_report,
        "warnings": warnings,
        "errors": errors,
    }


def _render_record(events: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Experiment record", "", "Generated from `events.jsonl`.", ""]
    for event in events:
        event_id = str(event.get("event_id", "UNKNOWN"))
        video_seconds = float(
            event.get("video_time_seconds", event.get("elapsed_seconds", 0.0))
        )
        hours, remainder = divmod(max(0.0, video_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        video_label = f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"
        lines.extend(
            [
                f"<a id=\"event-{event_id.casefold()}\"></a>",
                f"## Event {event_id}",
                "",
                f"- Type: `{event.get('event_type', 'unknown')}`",
                f"- UTC: `{event.get('utc', 'unknown')}`",
                f"- Elapsed: `{float(event.get('elapsed_seconds', 0.0)):.6f}` s",
                f"- Evidence: [video {video_label}](./video.mp4#t={video_seconds:.3f})",
                "",
            ]
        )
        frame_path = next(
            (
                event.get(key)
                for key in ("image_path", "frame_path", "camera_frame_path")
                if isinstance(event.get(key), str)
            ),
            None,
        )
        if frame_path is not None:
            safe_frame = str(frame_path).lstrip("./")
            lines.extend([f"- Request frame: [{safe_frame}](./{safe_frame})", ""])
        lines.extend(
            [
                "```json",
                json.dumps(event, ensure_ascii=False, sort_keys=True, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _render_result(result: Mapping[str, Any]) -> str:
    status = result.get("run_status", "UNKNOWN")
    verdict = result.get("oracle_verdict", "UNKNOWN")
    return (
        "# Attempt result\n\n"
        f"- Run status: `{status}`\n"
        f"- Oracle verdict: `{verdict}`\n\n"
        "```json\n"
        + json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n```\n"
    )


def _render_setup(setup: Mapping[str, Any]) -> str:
    return (
        "# Setup\n\n"
        "Generated from `setup.json`; the pre-execution snapshot is retained "
        "as `setup.initial.json`.\n\n"
        "```json\n"
        + json.dumps(setup, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n```\n"
    )


def _render_diagnosis(diagnosis: Mapping[str, Any]) -> str:
    lines = ["# Diagnosis", ""]
    for key, value in diagnosis.items():
        label = str(key).replace("_", " ").capitalize()
        if isinstance(value, list) and all(
            not isinstance(item, (dict, list, tuple)) for item in value
        ):
            lines.extend(
                [f"## {label}", "", *[f"- {item}" for item in value], ""]
            )
        elif isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            lines.extend([f"## {label}", "", "```json", rendered, "```", ""])
        else:
            lines.extend([f"## {label}", "", str(value), ""])
    return "\n".join(lines)


class AttemptRecorder(AbstractContextManager["AttemptRecorder"]):
    """High-level owner for one never-overwritten attempt artifact tree."""

    def __init__(
        self,
        attempt_dir: Path,
        *,
        origin: ClockOrigin,
        secret_fields: Iterable[str] = (),
        secret_values: Iterable[str] = (),
    ) -> None:
        self.attempt_dir = Path(attempt_dir)
        self.origin = origin
        self.model_calls = ModelCallRecorder(
            self.attempt_dir,
            origin=origin,
            secret_fields=secret_fields,
            secret_values=secret_values,
        )
        self.events = EventJournal(
            self.attempt_dir / "events.jsonl",
            origin=origin,
            secret_fields=secret_fields,
            secret_values=secret_values,
            redaction_journal=self.model_calls.redaction_journal,
        )
        self.runtime_states = JsonlJournal(
            self.attempt_dir / "runtime_states.jsonl", origin=origin
        )
        self.hidden_states = JsonlJournal(
            self.attempt_dir / "hidden_states.jsonl", origin=origin
        )
        self.oracle_observations = JsonlJournal(
            self.attempt_dir / "oracle_observations.jsonl", origin=origin
        )
        self.frames = FrameRequestRecorder(self.attempt_dir, origin=origin)
        self.secret_fields = tuple(secret_fields)
        self.secret_values = tuple(secret_values)
        self._finalized = False
        self._terminal_lock = threading.Lock()
        # Create append-only files and directories before any experimental work.
        for name in ("terminal.log",):
            path = self.attempt_dir / name
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            os.close(descriptor)
        durable_json(
            self.attempt_dir / "recording_state.json",
            {
                "schema_version": 1,
                "state": "OPEN",
                "clock_origin": origin.to_json(),
                "created_utc": format_utc(utc_now()),
            },
            exclusive=True,
        )

    @classmethod
    def create(
        cls,
        campaign_root: Path,
        *,
        schedule_id: str,
        attempt_number: int,
        origin: ClockOrigin | None = None,
        secret_fields: Iterable[str] = (),
        secret_values: Iterable[str] = (),
    ) -> "AttemptRecorder":
        attempt = safe_attempt_directory(campaign_root, schedule_id, attempt_number)
        return cls(
            attempt,
            origin=origin or ClockOrigin.capture(),
            secret_fields=secret_fields,
            secret_values=secret_values,
        )

    def _assert_open(self) -> None:
        if self._finalized:
            raise RecordingError("attempt has already been finalized")

    def write_json(self, name: str, value: Mapping[str, Any]) -> Path:
        self._assert_open()
        safe_name = _safe_component(name, "artifact name")
        if not safe_name.endswith(".json"):
            raise ValueError("JSON artifact name must end in .json")
        clean, redacted_paths = redact_value(
            value,
            secret_fields=self.secret_fields,
            secret_values=self.secret_values,
        )
        path = self.attempt_dir / safe_name
        durable_json(path, clean, exclusive=True)
        if redacted_paths:
            self.model_calls.redaction_journal.append(
                {
                    "artifact": safe_name,
                    "redacted_json_paths": redacted_paths,
                    "redaction_count": len(redacted_paths),
                }
            )
        return path

    def write_markdown(self, name: str, text: str) -> Path:
        self._assert_open()
        safe_name = _safe_component(name, "artifact name")
        if not safe_name.endswith(".md"):
            raise ValueError("Markdown artifact name must end in .md")
        clean = redact_text(text, secret_values=self.secret_values)
        path = self.attempt_dir / safe_name
        durable_write(path, clean.encode("utf-8"), exclusive=True)
        return path

    def write_setup(self, setup: Mapping[str, Any], markdown: str | None = None) -> None:
        payload = dict(setup)
        payload.setdefault("clock_origin", self.origin.to_json())
        self.write_json("setup.initial.json", payload)
        self.write_json("setup.json", payload)
        if markdown is None:
            clean_payload, _ = redact_value(
                payload,
                secret_fields=self.secret_fields,
                secret_values=self.secret_values,
            )
            markdown = (
                _render_setup(clean_payload)
            )
        self.write_markdown("SETUP.md", markdown)

    def write_inputs(self, inputs: Mapping[str, Any]) -> Path:
        return self.write_json("inputs.json", inputs)

    def write_initial_state(
        self, state: Mapping[str, Any], image_bytes: bytes | None = None
    ) -> None:
        self.write_json("initial_state.json", state)
        if image_bytes is not None:
            self._write_state_image("initial.png", image_bytes)

    def write_final_state(
        self, state: Mapping[str, Any], image_bytes: bytes | None = None
    ) -> None:
        self.write_json("final_state.json", state)
        if image_bytes is not None:
            self._write_state_image("final.png", image_bytes)

    def _write_state_image(self, name: str, image_bytes: bytes) -> None:
        raw, _, media_type = decode_image_payload(image_bytes)
        if _image_suffix(raw, media_type) != ".png":
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise RecordingError(f"could not encode {name}")
            raw = encoded.tobytes()
        durable_write(self.attempt_dir / name, raw, exclusive=True)

    def record_runtime_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_open()
        clean, _ = redact_value(
            state,
            secret_fields=self.secret_fields,
            secret_values=self.secret_values,
        )
        return self.runtime_states.append({"state": clean})

    def record_hidden_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        self._assert_open()
        clean, _ = redact_value(
            state,
            secret_fields=self.secret_fields,
            secret_values=self.secret_values,
        )
        return self.hidden_states.append({"state": clean})

    def record_oracle_observation(
        self, oracle_id: str, observation: Mapping[str, Any]
    ) -> dict[str, Any]:
        self._assert_open()
        clean, _ = redact_value(
            observation,
            secret_fields=self.secret_fields,
            secret_values=self.secret_values,
        )
        return self.oracle_observations.append(
            {"oracle_id": oracle_id, "observation": clean}
        )

    def write_oracle_result(self, result: Mapping[str, Any]) -> Path:
        """Persist the executable oracle result before post-run diagnosis."""

        return self.write_json("oracle_result.json", result)

    def terminal(self, text: str) -> None:
        self._assert_open()
        clean = redact_text(text, secret_values=self.secret_values)
        stamp = self.origin.stamp()
        line = f"{stamp['utc']} +{stamp['elapsed_seconds']:.6f}s {clean.rstrip()}\n"
        with self._terminal_lock:
            descriptor = os.open(self.attempt_dir / "terminal.log", os.O_WRONLY | os.O_APPEND)
            try:
                view = memoryview(line.encode("utf-8"))
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise RecordingError("short append to terminal.log")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def write_error(self, error: BaseException | Mapping[str, Any] | str) -> Path:
        if isinstance(error, BaseException):
            payload: Mapping[str, Any] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        elif isinstance(error, str):
            payload = {"type": "RecordedError", "message": error}
        else:
            payload = error
        return self.write_json("error.json", payload)

    def finalize(
        self,
        *,
        run_status: RunStatus | str,
        oracle_verdict: str | None,
        result: Mapping[str, Any],
        diagnosis: Mapping[str, Any],
        require_final_state: bool = False,
    ) -> dict[str, Any]:
        self._assert_open()
        actual_status = RunStatus(run_status)
        if self.model_calls.active_call_ids:
            self.terminal(
                "WARNING unfinished model calls: "
                + ", ".join(self.model_calls.active_call_ids)
            )
        final_result = {
            **dict(result),
            "run_status": actual_status.value,
            "oracle_verdict": oracle_verdict,
            "end": self.origin.stamp(),
        }
        setup_path = self.attempt_dir / "setup.json"
        if setup_path.is_file():
            try:
                setup_payload = json.loads(setup_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RecordingError(f"cannot finalize malformed setup.json: {error}") from error
            if not isinstance(setup_payload, dict):
                raise RecordingError("cannot finalize non-object setup.json")
            setup_payload.update(
                {
                    "run_status": actual_status.value,
                    "oracle_verdict": oracle_verdict,
                    "end": final_result["end"],
                }
            )
            clean_setup, _ = redact_value(
                setup_payload,
                secret_fields=self.secret_fields,
                secret_values=self.secret_values,
            )
            durable_json(setup_path, clean_setup)
            durable_write(
                self.attempt_dir / "SETUP.md",
                _render_setup(clean_setup).encode("utf-8"),
            )
        self.write_json("results.json", final_result)
        clean_result, _ = redact_value(
            final_result,
            secret_fields=self.secret_fields,
            secret_values=self.secret_values,
        )
        self.write_markdown("results.md", _render_result(clean_result))
        events, event_errors = _read_jsonl(self.attempt_dir / "events.jsonl")
        if event_errors:
            raise RecordingError("cannot render malformed events journal")
        self.write_markdown("experiment_record.md", _render_record(events))
        diagnosis_payload = {
            "run_status": actual_status.value,
            "oracle_verdict": oracle_verdict,
            **dict(diagnosis),
        }
        clean_diagnosis, _ = redact_value(
            diagnosis_payload,
            secret_fields=self.secret_fields,
            secret_values=self.secret_values,
        )
        self.write_json("diagnosis.json", clean_diagnosis)
        self.write_markdown("diagnosis.md", _render_diagnosis(clean_diagnosis))
        state = {
            "schema_version": 1,
            "state": "FINALIZING",
            "run_status": actual_status.value,
            "oracle_verdict": oracle_verdict,
            "clock_origin": self.origin.to_json(),
            "finalized_utc": format_utc(utc_now()),
        }
        durable_json(self.attempt_dir / "recording_state.json", state)

        # artifact_checks intentionally precedes checksums.  It reports all
        # structural checks except checksum coverage of itself.
        preliminary = audit_attempt(
            self.attempt_dir,
            verify_video=True,
            require_final_state=require_final_state,
            finalizing=True,
        )
        durable_json(
            self.attempt_dir / "artifact_checks.json", preliminary, exclusive=True
        )
        write_checksums(self.attempt_dir)
        final_audit = audit_attempt(
            self.attempt_dir,
            verify_video=True,
            require_final_state=require_final_state,
        )
        if (
            not final_audit["passed"]
            and actual_status in {RunStatus.VALID_PASS, RunStatus.VALID_SYSTEM_FAILURE}
        ):
            # A behaviorally valid outcome with corrupt/missing evidence is an
            # INVALID_HARNESS attempt, never valid scientific data.  Reclassify
            # within the same finalization transaction before sealing the
            # registry; retain the original behavioral status and oracle verdict.
            behavioral_status = actual_status
            actual_status = RunStatus.INVALID_HARNESS
            final_result.update(
                {
                    "run_status": actual_status.value,
                    "behavioral_run_status_before_artifact_audit": behavioral_status.value,
                    "artifact_invalidation_errors": list(final_audit["errors"]),
                }
            )
            reclassified_result, _ = redact_value(
                final_result,
                secret_fields=self.secret_fields,
                secret_values=self.secret_values,
            )
            durable_json(self.attempt_dir / "results.json", reclassified_result)
            durable_write(
                self.attempt_dir / "results.md",
                _render_result(reclassified_result).encode("utf-8"),
            )
            if setup_path.is_file():
                setup_payload["run_status"] = actual_status.value
                setup_payload["behavioral_run_status_before_artifact_audit"] = (
                    behavioral_status.value
                )
                durable_json(setup_path, setup_payload)
                durable_write(
                    self.attempt_dir / "SETUP.md",
                    _render_setup(setup_payload).encode("utf-8"),
                )
            clean_diagnosis.update(
                {
                    "run_status": actual_status.value,
                    "artifact_invalidation": list(final_audit["errors"]),
                }
            )
            durable_json(
                self.attempt_dir / "diagnosis.json",
                clean_diagnosis,
            )
            durable_write(
                self.attempt_dir / "diagnosis.md",
                _render_diagnosis(clean_diagnosis).encode("utf-8"),
            )
            state["run_status"] = actual_status.value
        state.update(
            {
                "state": "FINALIZED" if final_audit["passed"] else "FINALIZED_AUDIT_FAILED",
                "artifact_audit_passed": final_audit["passed"],
            }
        )
        # Persist the full (non-deferred) audit, then checksum both it and the
        # final state.  A final independent audit below verifies this terminal
        # snapshot; avoiding self-referential digest content is why the report
        # records pass/fail and counts rather than its own expected digest.
        durable_json(self.attempt_dir / "recording_state.json", state)
        durable_json(self.attempt_dir / "artifact_checks.json", final_audit)
        write_checksums(self.attempt_dir)
        final_audit = audit_attempt(
            self.attempt_dir,
            verify_video=True,
            require_final_state=require_final_state,
        )
        self._finalized = True
        return final_audit

    def preserve_partial(
        self,
        *,
        reason: str,
        run_status: RunStatus | str = RunStatus.INFRA_INTERRUPTED,
        diagnosis: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Best-effort finalization that never labels missing artifacts complete."""

        actual_status = RunStatus(run_status)
        self.write_error(reason)
        self.events.append("attempt_interrupted", reason=redact_text(reason))
        try:
            return self.finalize(
                run_status=actual_status,
                oracle_verdict=None,
                result={"complete": False, "interruption_reason": redact_text(reason)},
                diagnosis=diagnosis
                or {
                    "summary": "Attempt interrupted; partial artifacts preserved.",
                    "evidence": "See the final interruption event and error.json.",
                },
            )
        except Exception as error:
            # Even audit/render failure gets an explicit durable marker.
            durable_json(
                self.attempt_dir / "partial_failure.json",
                {
                    "artifact_state": "PARTIAL",
                    "run_status": actual_status.value,
                    "reason": redact_text(reason),
                    "finalization_error": {
                        "type": type(error).__name__,
                        "message": redact_text(str(error)),
                    },
                    "time": self.origin.stamp(),
                },
            )
            try:
                write_checksums(self.attempt_dir)
            finally:
                self._finalized = True
            return {
                "schema_version": 1,
                "passed": False,
                "errors": [f"partial finalization failed: {type(error).__name__}: {error}"],
            }

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc is not None and not self._finalized:
            try:
                self.preserve_partial(
                    reason=f"uncaught {type(exc).__name__}: {exc}",
                    run_status=RunStatus.INVALID_HARNESS,
                )
            except Exception as recorder_error:
                # Do not replace the original experimental exception.  Retain
                # a best-effort sidecar if the ordinary partial finalizer itself
                # could not run (for example after a duplicate artifact write).
                try:
                    durable_json(
                        self.attempt_dir / "partial_finalizer_error.json",
                        {
                            "artifact_state": "PARTIAL",
                            "original_error": type(exc).__name__,
                            "recorder_error": {
                                "type": type(recorder_error).__name__,
                                "message": redact_text(str(recorder_error)),
                            },
                            "time": self.origin.stamp(),
                        },
                        exclusive=True,
                    )
                except Exception:
                    pass
        return False


__all__ = [
    "AttemptRecorder",
    "ClockOrigin",
    "EventJournal",
    "FrameRequestRecorder",
    "JsonlJournal",
    "ModelCallRecorder",
    "RecordingError",
    "RunStatus",
    "audit_attempt",
    "decode_image_payload",
    "durable_json",
    "durable_write",
    "format_utc",
    "redact_text",
    "redact_value",
    "safe_attempt_directory",
    "sha256_bytes",
    "sha256_file",
    "verify_checksums",
    "write_checksums",
]
