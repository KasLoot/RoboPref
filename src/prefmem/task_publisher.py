"""Typed client for the camera page's single-slot controller display.

The execution runtime owns ``session_id`` and a monotonically increasing
``sequence``.  The camera server uses those fields to reject delayed updates;
this module deliberately does not generate either value on the caller's
behalf.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import ipaddress
import json
import math
import socket
import threading
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener


TASK_API_PATH = "/api/task"
FRAME_SEQUENCE_HEADER = "X-Frame-Sequence"
MAX_EXPECTED_OBSERVATIONS = 20


class DisplayState(str, Enum):
    """States understood by the human execution surface."""

    PLANNING = "PLANNING"
    ACTIVE = "ACTIVE"
    FINAL_VALIDATION = "FINAL_VALIDATION"
    NEEDS_ATTENTION = "NEEDS_ATTENTION"
    COMPLETE = "COMPLETE"
    EMERGENCY_STOPPED = "EMERGENCY_STOPPED"


def _text(
    value: object,
    field_name: str,
    *,
    maximum: int,
    nullable: bool = False,
) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str):
        suffix = " or null" if nullable else ""
        raise ValueError(f"{field_name} must be a string{suffix}")
    result = value.strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    if len(result) > maximum:
        raise ValueError(f"{field_name} must be at most {maximum} characters")
    return result


def _optional_text(value: object, field_name: str, *, maximum: int) -> str | None:
    return _text(value, field_name, maximum=maximum, nullable=True)


def _positive_integer(value: object, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array of strings")
    if len(value) > MAX_EXPECTED_OBSERVATIONS:
        raise ValueError(
            f"{field_name} must contain at most {MAX_EXPECTED_OBSERVATIONS} items"
        )
    result = tuple(
        _text(item, f"{field_name}[{index}]", maximum=2048)
        for index, item in enumerate(value)
    )
    folded = tuple(item.casefold() for item in result)
    if len(folded) != len(set(folded)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class ControllerDisplay:
    """One complete replacement for the camera page's current display.

    This is a single-slot state projection, not a plan or task queue.  During
    ``PLANNING`` the instruction fields must be absent so an execution agent
    cannot continue acting on a retired task.
    """

    session_id: str
    sequence: int
    state: DisplayState
    goal: str
    cycle: int | None = None
    publication_id: str | None = None
    instruction: str | None = None
    expected_observation: tuple[str, ...] = ()
    message: str | None = None
    monitor_observation: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "session_id",
            _text(self.session_id, "session_id", maximum=128),
        )
        object.__setattr__(
            self,
            "sequence",
            _positive_integer(self.sequence, "sequence"),
        )
        try:
            state = DisplayState(self.state)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid display state: {self.state!r}") from exc
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "goal", _text(self.goal, "goal", maximum=4096))

        cycle = self.cycle
        if cycle is not None:
            cycle = _positive_integer(cycle, "cycle")
        object.__setattr__(self, "cycle", cycle)
        for field_name, maximum in (
            ("publication_id", 256),
            ("instruction", 4096),
            ("message", 4096),
            ("monitor_observation", 4096),
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_text(
                    getattr(self, field_name),
                    field_name,
                    maximum=maximum,
                ),
            )
        object.__setattr__(
            self,
            "expected_observation",
            _string_tuple(self.expected_observation, "expected_observation"),
        )

        if state is DisplayState.ACTIVE:
            if cycle is None:
                raise ValueError("an ACTIVE display requires cycle")
            if self.publication_id is None:
                raise ValueError("an ACTIVE display requires publication_id")
            if self.instruction is None:
                raise ValueError("an ACTIVE display requires instruction")
            if not self.expected_observation:
                raise ValueError(
                    "an ACTIVE display requires expected_observation"
                )
        elif state is DisplayState.PLANNING:
            if self.publication_id is not None or self.instruction is not None:
                raise ValueError(
                    "a PLANNING display cannot retain a task publication"
                )
            if self.expected_observation:
                raise ValueError(
                    "a PLANNING display cannot retain expected_observation"
                )
        elif state is DisplayState.FINAL_VALIDATION:
            if self.publication_id is None:
                raise ValueError(
                    "a FINAL_VALIDATION display requires publication_id"
                )
            if self.instruction is not None:
                raise ValueError(
                    "a FINAL_VALIDATION display cannot contain an execution instruction"
                )
            if not self.expected_observation:
                raise ValueError(
                    "a FINAL_VALIDATION display requires expected_observation"
                )
        elif state in {
            DisplayState.COMPLETE,
            DisplayState.EMERGENCY_STOPPED,
        } and self.instruction is not None:
            raise ValueError(
                f"a {state.value} display cannot contain an execution instruction"
            )
        if state in {
            DisplayState.NEEDS_ATTENTION,
            DisplayState.EMERGENCY_STOPPED,
        } and self.message is None:
            raise ValueError(f"a {state.value} display requires message")

    @classmethod
    def from_payload(cls, value: object) -> ControllerDisplay:
        if not isinstance(value, Mapping):
            raise ValueError("request body must be a JSON object")
        allowed = {
            "session_id",
            "sequence",
            "state",
            "goal",
            "cycle",
            "publication_id",
            "instruction",
            "expected_observation",
            "message",
            "monitor_observation",
        }
        unknown = set(value) - allowed
        if unknown:
            fields = ", ".join(sorted(str(field) for field in unknown))
            raise ValueError(f"unknown display field(s): {fields}")
        missing = {
            field
            for field in ("session_id", "sequence", "state", "goal")
            if field not in value
        }
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"missing display field(s): {fields}")
        return cls(
            session_id=value["session_id"],
            sequence=value["sequence"],
            state=value["state"],
            goal=value["goal"],
            cycle=value.get("cycle"),
            publication_id=value.get("publication_id"),
            instruction=value.get("instruction"),
            expected_observation=value.get("expected_observation", ()),
            message=value.get("message"),
            monitor_observation=value.get("monitor_observation"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "sequence": self.sequence,
            "state": self.state.value,
            "goal": self.goal,
            "cycle": self.cycle,
            "publication_id": self.publication_id,
            "instruction": self.instruction,
            "expected_observation": list(self.expected_observation),
            "message": self.message,
            "monitor_observation": self.monitor_observation,
        }


@dataclass(frozen=True, slots=True)
class DisplayEnvelope:
    """Server acknowledgement containing its monotonic display revision."""

    revision: int
    display: ControllerDisplay | None

    @classmethod
    def from_payload(cls, value: object) -> DisplayEnvelope:
        if not isinstance(value, Mapping):
            raise ValueError("response must be a JSON object")
        if set(value) != {"revision", "display"}:
            raise ValueError("response must contain only revision and display")
        revision = value["revision"]
        if type(revision) is not int or revision < 0:
            raise ValueError("response revision must be a non-negative integer")
        raw_display = value["display"]
        display = (
            None
            if raw_display is None
            else ControllerDisplay.from_payload(raw_display)
        )
        return cls(revision=revision, display=display)


class TaskPublisherError(RuntimeError):
    """Base error raised by :class:`CameraTaskPublisher`."""


class TaskPublisherConnectionError(TaskPublisherError):
    """The camera task API could not be reached."""


class TaskPublisherResponseError(TaskPublisherError):
    """The camera task API rejected a valid HTTP request."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"camera task API returned HTTP {status}: {message}")


class TaskPublisherProtocolError(TaskPublisherError):
    """The camera task API returned an invalid acknowledgement."""


def _loopback_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base_url must be a non-empty string")
    parsed = urlsplit(value.strip())
    if parsed.scheme != "http":
        raise ValueError("base_url must use http")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not contain credentials")
    if not parsed.hostname:
        raise ValueError("base_url must include a host")
    if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("base_url must not contain a path, query, or fragment")
    hostname = parsed.hostname.casefold().rstrip(".")
    if hostname != "localhost":
        try:
            address = ipaddress.ip_address(hostname.split("%", 1)[0])
        except ValueError as exc:
            raise ValueError("base_url host must be a loopback address") from exc
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            is_loopback = address.ipv4_mapped.is_loopback
        else:
            is_loopback = address.is_loopback
        if not is_loopback:
            raise ValueError("base_url host must be a loopback address")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url contains an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("base_url contains an invalid port")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


class CameraTaskPublisher:
    """Serialize display replacements to the loopback camera task API."""

    def __init__(self, base_url: str, *, timeout: float = 2.0) -> None:
        self.base_url = _loopback_base_url(base_url)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        # Never send execution instructions through an environment-configured
        # HTTP proxy, even if its NO_PROXY settings are incomplete.
        self._opener = build_opener(ProxyHandler({}))

    def publish(self, display: ControllerDisplay) -> DisplayEnvelope:
        if not isinstance(display, ControllerDisplay):
            raise TypeError("display must be a ControllerDisplay")
        payload = json.dumps(
            display.to_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        with self._lock:
            envelope = self._request("PUT", TASK_API_PATH, payload)
        if envelope.display != display:
            raise TaskPublisherProtocolError(
                "camera task API acknowledged a different display"
            )
        return envelope

    def reset(self, *, session_id: str | None = None) -> DisplayEnvelope:
        """Clear the slot and retire its session against delayed updates.

        Passing ``session_id`` adds an optimistic ownership check: the server
        returns HTTP 409 rather than clearing a newer session.
        """

        path = TASK_API_PATH
        if session_id is not None:
            validated = _text(session_id, "session_id", maximum=128)
            path = f"{path}?{urlencode({'session_id': validated})}"
        with self._lock:
            envelope = self._request("DELETE", path, None)
        if envelope.display is not None:
            raise TaskPublisherProtocolError(
                "camera task API reset acknowledgement retained a display"
            )
        return envelope

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None,
    ) -> DisplayEnvelope:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                response_body = response.read()
        except HTTPError as exc:
            response_body = exc.read()
            message = _error_message(response_body, exc.reason)
            raise TaskPublisherResponseError(exc.code, message) from exc
        except (URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise TaskPublisherConnectionError(
                f"could not reach camera task API at {self.base_url}: {exc}"
            ) from exc
        try:
            value = json.loads(response_body.decode("utf-8"))
            return DisplayEnvelope.from_payload(value)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise TaskPublisherProtocolError(
                f"camera task API returned an invalid acknowledgement: {exc}"
            ) from exc


def _error_message(body: bytes, fallback: object) -> str:
    try:
        value: Any = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return str(fallback)
    if isinstance(value, Mapping) and isinstance(value.get("error"), str):
        return value["error"]
    return str(fallback)
