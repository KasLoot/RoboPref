"""Explicit, idempotent publication and cancellation boundary for VLA execution."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from prefmem.agents.contracts import ExecutionCommand, PublicationReceipt
from prefmem.recording import ExperimentRecorder, NullRecorder


class ExecutionUnavailableError(RuntimeError):
    """Raised when execution is requested without a configured VLA adapter."""


@dataclass(frozen=True, slots=True)
class CancellationReceipt:
    """Result of asking an execution adapter to stop one dispatch.

    ``accepted`` only confirms that the adapter accepted or durably queued the
    request. It does not claim that physical motion has already stopped.
    """

    dispatch_id: str
    reason: str
    supported: bool
    accepted: bool
    requested_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    duplicate_suppressed: bool = False

    def __post_init__(self) -> None:
        if not self.dispatch_id.strip():
            raise ValueError("cancellation dispatch ID must not be empty")
        if not self.reason.strip():
            raise ValueError("cancellation reason must not be empty")
        if (
            self.requested_at.tzinfo is None
            or self.requested_at.utcoffset() is None
        ):
            raise ValueError("cancellation timestamp must be timezone-aware")
        if self.accepted and not self.supported:
            raise ValueError("an unsupported cancellation cannot be accepted")

    def as_json(self) -> dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "reason": self.reason,
            "supported": self.supported,
            "accepted": self.accepted,
            "requested_at": self.requested_at.isoformat(),
            "duplicate_suppressed": self.duplicate_suppressed,
        }


class VLAAdapter(Protocol):
    def publish(self, command: ExecutionCommand) -> PublicationReceipt: ...


class CancellableVLAAdapter(Protocol):
    """Optional adapter capability for stopping a published dispatch."""

    def cancel(self, dispatch_id: str, reason: str) -> CancellationReceipt: ...


class UnavailableVLAAdapter:
    def publish(self, command: ExecutionCommand) -> PublicationReceipt:
        raise ExecutionUnavailableError(
            "No VLA execution adapter is configured; "
            f"subtask {command.subtask_id!r} was not sent."
        )


class CallbackVLAAdapter:
    """Small adapter useful for an in-process robot integration or tests."""

    def __init__(
        self,
        callback: Callable[[ExecutionCommand], None],
        *,
        cancel_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.callback = callback
        self.cancel_callback = cancel_callback

    def publish(self, command: ExecutionCommand) -> PublicationReceipt:
        self.callback(command)
        return PublicationReceipt(dispatch_id=command.dispatch_id)

    def cancel(self, dispatch_id: str, reason: str) -> CancellationReceipt:
        if self.cancel_callback is None:
            return CancellationReceipt(
                dispatch_id=dispatch_id,
                reason=reason,
                supported=False,
                accepted=False,
            )
        self.cancel_callback(dispatch_id, reason)
        return CancellationReceipt(
            dispatch_id=dispatch_id,
            reason=reason,
            supported=True,
            accepted=True,
        )


class JsonlVLAAdapter:
    """Append commands and cancellation requests to an external VLA queue."""

    _CANCELLATION_MESSAGE_TYPE = "CANCEL"

    def __init__(self, queue_path: Path) -> None:
        self.queue_path = Path(queue_path).expanduser().resolve()
        self.queue_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.Lock()
        self._published: dict[str, ExecutionCommand] = {}
        self._cancellations: dict[str, CancellationReceipt] = {}
        self._load_existing_records()

    def _load_existing_records(self) -> None:
        if not self.queue_path.exists():
            return
        with self.queue_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("record must be a JSON object")
                    if (
                        payload.get("message_type")
                        == self._CANCELLATION_MESSAGE_TYPE
                    ):
                        receipt = CancellationReceipt(
                            dispatch_id=str(payload["dispatch_id"]),
                            reason=str(payload["reason"]),
                            supported=True,
                            accepted=True,
                            requested_at=datetime.fromisoformat(
                                str(payload["requested_at"])
                            ),
                        )
                        self._cancellations.setdefault(
                            receipt.dispatch_id,
                            receipt,
                        )
                        continue

                    command = ExecutionCommand.model_validate(payload)
                    prior = self._published.get(command.dispatch_id)
                    if prior is not None and prior != command:
                        raise ValueError(
                            "dispatch ID was reused for a different "
                            "execution command"
                        )
                    self._published.setdefault(command.dispatch_id, command)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid VLA JSONL record at "
                        f"{self.queue_path}:{line_number}: {exc}"
                    ) from exc

    def _append(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        descriptor = os.open(
            self.queue_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def publish(self, command: ExecutionCommand) -> PublicationReceipt:
        with self._lock:
            prior = self._published.get(command.dispatch_id)
            if prior is not None:
                if prior != command:
                    raise ValueError(
                        "dispatch ID was reused for a different execution command"
                    )
                return PublicationReceipt(
                    dispatch_id=command.dispatch_id,
                    duplicate_suppressed=True,
                )
            self._append(command.model_dump(mode="json"))
            self._published[command.dispatch_id] = command
        return PublicationReceipt(dispatch_id=command.dispatch_id)

    def cancel(self, dispatch_id: str, reason: str) -> CancellationReceipt:
        with self._lock:
            prior = self._cancellations.get(dispatch_id)
            if prior is not None:
                return replace(prior, duplicate_suppressed=True)

            receipt = CancellationReceipt(
                dispatch_id=dispatch_id,
                reason=reason,
                supported=True,
                accepted=True,
            )
            self._append(
                {
                    "message_type": self._CANCELLATION_MESSAGE_TYPE,
                    "dispatch_id": receipt.dispatch_id,
                    "reason": receipt.reason,
                    "requested_at": receipt.requested_at.isoformat(),
                }
            )
            self._cancellations[dispatch_id] = receipt
            return receipt


class IdempotentVLAAdapter:
    """Prevent duplicate/conflicting publication and duplicate cancellation."""

    def __init__(
        self,
        delegate: VLAAdapter,
        *,
        recorder: ExperimentRecorder | None = None,
    ) -> None:
        self.delegate = delegate
        self.recorder = recorder or NullRecorder()
        self._published: dict[str, ExecutionCommand] = {}
        self._receipts: dict[str, PublicationReceipt] = {}
        self._cancellations: dict[str, CancellationReceipt] = {}
        self._lock = threading.Lock()

    def publish(self, command: ExecutionCommand) -> PublicationReceipt:
        with self._lock:
            prior = self._published.get(command.dispatch_id)
            if prior is not None:
                if prior != command:
                    raise ValueError(
                        "dispatch ID was reused for a different execution command"
                    )
                receipt = self._receipts[command.dispatch_id].model_copy(
                    update={"duplicate_suppressed": True}
                )
                self.recorder.record_event(
                    "Duplicate VLA publication suppressed",
                    {"command": command, "receipt": receipt},
                )
                return receipt

            receipt = self.delegate.publish(command)
            if receipt.dispatch_id != command.dispatch_id:
                raise ValueError("VLA receipt dispatch ID does not match the command")
            if not receipt.published:
                raise ExecutionUnavailableError(
                    f"VLA did not publish dispatch {command.dispatch_id}"
                )
            self._published[command.dispatch_id] = command
            self._receipts[command.dispatch_id] = receipt
            self.recorder.record_event(
                "VLA command published",
                {"command": command, "receipt": receipt},
            )
            return receipt

    def cancel(self, dispatch_id: str, reason: str) -> CancellationReceipt:
        with self._lock:
            prior = self._cancellations.get(dispatch_id)
            if prior is not None:
                receipt = replace(prior, duplicate_suppressed=True)
                self.recorder.record_event(
                    "Duplicate VLA cancellation suppressed",
                    {"receipt": receipt},
                )
                return receipt

            cancel = getattr(self.delegate, "cancel", None)
            if callable(cancel):
                receipt = cancel(dispatch_id, reason)
                if not isinstance(receipt, CancellationReceipt):
                    raise TypeError(
                        "VLA cancel must return a CancellationReceipt"
                    )
                if receipt.dispatch_id != dispatch_id:
                    raise ValueError(
                        "VLA cancellation receipt dispatch ID does not match"
                    )
                if receipt.reason != reason:
                    raise ValueError(
                        "VLA cancellation receipt reason does not match"
                    )
            else:
                receipt = CancellationReceipt(
                    dispatch_id=dispatch_id,
                    reason=reason,
                    supported=False,
                    accepted=False,
                )

            self._cancellations[dispatch_id] = receipt
            self.recorder.record_event(
                "VLA cancellation processed",
                {"receipt": receipt},
            )
            return receipt
