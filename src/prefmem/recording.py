"""Replay-oriented experiment recording for PrefMem.

The recorder intentionally writes ordinary Markdown plus image files.  A run can
therefore be inspected without PrefMem, committed as an experiment artifact, or
used to reconstruct the exact text and visual context supplied to each agent.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from pydantic import BaseModel


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(value)


def _safe_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", label.strip()).strip("-")
    if cleaned in {".", ".."}:
        return "frame"
    return cleaned or "frame"


class NullRecorder:
    """No-op recorder used when experiment recording is disabled."""

    enabled = False
    session_dir: Path | None = None
    markdown_path: Path | None = None

    def record_event(
        self,
        event: str,
        payload: Any = None,
        *,
        heading_level: int = 2,
    ) -> None:
        del event, payload, heading_level

    def record_frame(
        self,
        image: bytes,
        *,
        source: str,
        media_type: str = "image/jpeg",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path | None:
        del image, source, media_type, metadata
        return None

    def record_model_exchange(
        self,
        *,
        agent: str,
        messages: Sequence[Any],
        response: Any,
        frame_paths: Sequence[Path] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        del agent, messages, response, frame_paths, metadata

    def record_user(self, text: str, *, frame_path: Path | None = None) -> None:
        del text, frame_path

    def record_assistant(self, text: str) -> None:
        del text


class MarkdownExperimentRecorder:
    """Append a complete run trace to Markdown and persist ingested frames."""

    enabled = True

    def __init__(
        self,
        root: Path,
        *,
        session_name: str | None = None,
        configuration: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S.%fZ")
        suffix = uuid4().hex[:8]
        name = _safe_label(session_name or f"{timestamp}-{suffix}")

        resolved_root = Path(root).expanduser().resolve()
        resolved_root.mkdir(parents=True, exist_ok=True)
        self.session_dir = resolved_root / name
        self.frames_dir = self.session_dir / "frames"
        self.session_dir.mkdir(mode=0o700, exist_ok=False)
        self.frames_dir.mkdir(mode=0o700)
        self.markdown_path = self.session_dir / "session.md"
        self._lock = threading.Lock()
        self._event_index = 0
        self._frame_index = 0

        header = [
            "# PrefMem experiment transcript",
            "",
            f"- Session: `{name}`",
            f"- Started (UTC): `{(now or datetime.now(UTC)).isoformat()}`",
            "- Format: append-only Markdown with locally linked input frames",
            "",
        ]
        if configuration:
            header.extend(
                [
                    "## Runtime configuration",
                    "",
                    "```json",
                    json.dumps(
                        configuration,
                        indent=2,
                        ensure_ascii=False,
                        default=_json_default,
                    ),
                    "```",
                    "",
                ]
            )
        descriptor = os.open(
            self.markdown_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("\n".join(header))

    @staticmethod
    def _extension(media_type: str) -> str:
        return {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
        }.get(media_type.lower(), ".bin")

    def _append(self, text: str) -> None:
        with self._lock:
            with self.markdown_path.open("a", encoding="utf-8") as stream:
                stream.write(text)
                if not text.endswith("\n"):
                    stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

    def record_event(
        self,
        event: str,
        payload: Any = None,
        *,
        heading_level: int = 2,
    ) -> None:
        with self._lock:
            self._event_index += 1
            index = self._event_index
            observed_at = datetime.now(UTC).isoformat()
            hashes = "#" * min(max(heading_level, 2), 6)
            lines = [
                "",
                f"{hashes} {index}. {event}",
                "",
                f"_UTC: `{observed_at}`_",
                "",
            ]
            if payload is not None:
                if isinstance(payload, str):
                    lines.extend([payload, ""])
                else:
                    lines.extend(
                        [
                            "```json",
                            json.dumps(
                                payload,
                                indent=2,
                                ensure_ascii=False,
                                default=_json_default,
                            ),
                            "```",
                            "",
                        ]
                    )
            with self.markdown_path.open("a", encoding="utf-8") as stream:
                stream.write("\n".join(lines))
                stream.flush()
                os.fsync(stream.fileno())

    def record_frame(
        self,
        image: bytes,
        *,
        source: str,
        media_type: str = "image/jpeg",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        if not image:
            raise ValueError("cannot record an empty frame")
        with self._lock:
            self._frame_index += 1
            frame_index = self._frame_index
            filename = (
                f"frame-{frame_index:05d}-{_safe_label(source)}"
                f"{self._extension(media_type)}"
            )
            path = self.frames_dir / filename
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(image)

        relative = path.relative_to(self.session_dir).as_posix()
        payload = {
            "source": source,
            "media_type": media_type,
            "bytes": len(image),
            "path": relative,
            **dict(metadata or {}),
        }
        self.record_event(
            f"Frame {frame_index:05d}: {source}",
            (
                f"![Frame {frame_index:05d} from {source}]({relative})\n\n"
                "```json\n"
                f"{json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)}\n"
                "```\n"
            ),
            heading_level=3,
        )
        return path

    @staticmethod
    def _serialize_message(message: Any) -> dict[str, Any]:
        if isinstance(message, Mapping):
            raw = dict(message)
        elif isinstance(message, BaseModel):
            raw = message.model_dump(mode="json")
        else:
            raw = {
                "type": type(message).__name__,
                "content": getattr(message, "content", str(message)),
            }

        # Base64 camera payloads are already saved separately.  Keeping them in
        # Markdown would make a transcript both unreadable and unnecessarily huge.
        content = raw.get("content")
        if isinstance(content, list):
            normalized: list[Any] = []
            for block in content:
                if (
                    isinstance(block, Mapping)
                    and block.get("type") == "image_url"
                ):
                    normalized.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": "<saved frame; see frame_paths>"},
                        }
                    )
                else:
                    normalized.append(block)
            raw["content"] = normalized
        return raw

    def record_model_exchange(
        self,
        *,
        agent: str,
        messages: Sequence[Any],
        response: Any,
        frame_paths: Sequence[Path] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        relative_frames = [
            path.relative_to(self.session_dir).as_posix()
            if path.is_relative_to(self.session_dir)
            else str(path)
            for path in frame_paths
        ]
        payload = {
            "agent": agent,
            "messages": [self._serialize_message(message) for message in messages],
            "response": response,
            "frame_paths": relative_frames,
            "metadata": dict(metadata or {}),
        }
        self.record_event(f"Model exchange: {agent}", payload)

    def record_user(self, text: str, *, frame_path: Path | None = None) -> None:
        suffix = ""
        if frame_path is not None:
            relative = (
                frame_path.relative_to(self.session_dir).as_posix()
                if frame_path.is_relative_to(self.session_dir)
                else str(frame_path)
            )
            suffix = f"\n\n_Input frame: [{relative}]({relative})_"
        self.record_event("User", f"{text}{suffix}")

    def record_assistant(self, text: str) -> None:
        self.record_event("PrefMem", text)


ExperimentRecorder = MarkdownExperimentRecorder | NullRecorder


def build_recorder(
    *,
    enabled: bool,
    root: Path,
    session_name: str | None = None,
    configuration: Mapping[str, Any] | None = None,
) -> ExperimentRecorder:
    if not enabled:
        return NullRecorder()
    return MarkdownExperimentRecorder(
        root,
        session_name=session_name,
        configuration=configuration,
    )
