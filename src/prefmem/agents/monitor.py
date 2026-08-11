"""Live, single-flight visual monitoring for one published task at a time.

The service deliberately contains no temporal success/failure aggregation.
Each valid model result is handed to the deterministic controller, which owns
stability thresholds and MPC transitions.  Emergency-stop results are the one
exception: one valid ``emergency_stop=true`` result immediately latches the
process-wide stop coordinator and is never sent through ordinary aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import inspect
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from prefmem.agents.metrics import MetricsRecordingError
from prefmem.agents.vision import DEFAULT_LIVE_FRAME_URL, image_data_url
from prefmem.contracts import MonitorAssessment
from prefmem.emergency import EmergencyStopCoordinator


LOGGER = logging.getLogger(__name__)
DEFAULT_MONITOR_MODEL = "/workspace/models/gemma-4-26B-A4B-it"
DEFAULT_MONITOR_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MONITOR_PROMPT = (
    Path(__file__).resolve().parent
    / "prompt"
    / "monitor"
    / "monitor-prompt-v1.md"
)
THINKING_DISABLED_OPTIONS = {
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
}


class MonitorOutputError(ValueError):
    """The monitor returned something outside its strict JSON contract."""


class EmergencyStopLatchedError(RuntimeError):
    """A task cannot be admitted after the process emergency latch is set."""


class MonitorErrorKind(str, Enum):
    FRAME = "FRAME_ERROR"
    MODEL = "MODEL_ERROR"
    OUTPUT = "OUTPUT_ERROR"
    CALLBACK = "CALLBACK_ERROR"
    RECORDING = "RECORDING_ERROR"


class MonitorTelemetryKind(str, Enum):
    """Read-only lifecycle events emitted by the single-flight worker."""

    INFERENCE_STARTED = "INFERENCE_STARTED"
    INFERENCE_COMPLETED = "INFERENCE_COMPLETED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class MonitorErrorEvent:
    """A system error to be routed to controller ``NEEDS_ATTENTION`` state."""

    kind: MonitorErrorKind
    message: str
    publication_id: str | None


@dataclass(frozen=True, slots=True)
class MonitorTelemetryEvent:
    """Host-owned Monitor telemetry; never an input to controller decisions."""

    kind: MonitorTelemetryKind
    publication_id: str
    frame_sequence: int | None = None
    observed_at: float | None = None
    elapsed_seconds: float | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        try:
            kind = MonitorTelemetryKind(self.kind)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid monitor telemetry kind: {self.kind!r}") from error
        publication_id = str(self.publication_id).strip()
        if not publication_id:
            raise ValueError("publication_id must be non-empty")
        if self.frame_sequence is not None and (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be non-negative or None")
        for name, value in (
            ("observed_at", self.observed_at),
            ("elapsed_seconds", self.elapsed_seconds),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative or None")
        message = None if self.message is None else str(self.message).strip()
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "publication_id", publication_id)
        object.__setattr__(self, "message", message or None)


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """Newest camera frame plus host-owned capture metadata."""

    image_block: Mapping[str, Any]
    observed_at: float
    sequence: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_block, Mapping):
            raise ValueError("image_block must be a mapping")
        image_block = dict(self.image_block)
        if image_block.get("type") != "image_url":
            raise ValueError("image_block must be an image_url content block")
        object.__setattr__(self, "image_block", image_block)
        if isinstance(self.observed_at, bool) or not isinstance(
            self.observed_at,
            (int, float),
        ):
            raise ValueError("observed_at must be a finite number")
        observed_at = float(self.observed_at)
        if not math.isfinite(observed_at):
            raise ValueError("observed_at must be a finite number")
        object.__setattr__(self, "observed_at", observed_at)
        if self.sequence is not None and (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer or None")


class FrameSource(Protocol):
    def __call__(self) -> CapturedFrame: ...


class MonitorModel(Protocol):
    def invoke(self, messages: list[Any], **kwargs: Any) -> Any: ...


class HTTPFrameSource:
    """Fetch the current camera JPEG without retaining a frame backlog.

    ``stream_camera`` supplies ``X-Frame-Sequence`` so the service can reject a
    frame captured at or before publication.  For compatibility with an older
    streamer the header may be absent, in which case timestamp and publication
    identity fencing still apply and ``sequence`` is ``None``.
    """

    _SEQUENCE_HEADERS = (
        "X-Frame-Sequence",
        "X-Camera-Frame-Sequence",
    )

    def __init__(
        self,
        snapshot_url: str = DEFAULT_LIVE_FRAME_URL,
        *,
        timeout: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(snapshot_url, str) or not snapshot_url.strip():
            raise ValueError("snapshot_url must be a non-empty string")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.snapshot_url = snapshot_url.strip()
        self.timeout = float(timeout)
        self._clock = clock

    def __call__(self) -> CapturedFrame:
        request = Request(
            self.snapshot_url,
            headers={"Accept": "image/jpeg", "Cache-Control": "no-cache"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                image = response.read()
                sequence_text = next(
                    (
                        response.headers.get(name)
                        for name in self._SEQUENCE_HEADERS
                        if response.headers.get(name) is not None
                    ),
                    None,
                )
        except HTTPError as error:
            raise RuntimeError(
                f"Could not get a monitor frame from {self.snapshot_url}: "
                f"HTTP {error.code}."
            ) from error
        except (URLError, TimeoutError) as error:
            reason = getattr(error, "reason", error)
            raise RuntimeError(
                f"Could not get a monitor frame from {self.snapshot_url}: "
                f"{reason}."
            ) from error
        if not image:
            raise RuntimeError(
                f"The camera returned an empty monitor frame from "
                f"{self.snapshot_url}."
            )

        sequence: int | None = None
        if sequence_text is not None:
            try:
                sequence = int(sequence_text)
            except ValueError as error:
                raise RuntimeError(
                    "The camera returned an invalid frame-sequence header."
                ) from error
            if sequence < 0:
                raise RuntimeError(
                    "The camera returned a negative frame-sequence header."
                )
        return CapturedFrame(
            image_block={
                "type": "image_url",
                "image_url": {"url": image_data_url(image)},
            },
            sequence=sequence,
            observed_at=self._clock(),
        )


_OUTPUT_FIELDS = frozenset(
    {
        "emergency_stop",
        "emergency_reason",
        "task_status",
        "criteria",
        "failure",
        "observation",
    }
)


@dataclass(frozen=True, slots=True)
class MonitorModelOutput:
    """Strict outer model envelope, parsed before contract construction."""

    emergency_stop: bool
    emergency_reason: str | None
    assessment_payload: Mapping[str, Any]

    @classmethod
    def from_json(cls, raw: str) -> MonitorModelOutput:
        payload = _strict_json_object(raw)
        unknown = sorted(str(key) for key in set(payload) - _OUTPUT_FIELDS)
        missing = sorted(_OUTPUT_FIELDS - set(payload))
        if unknown:
            raise MonitorOutputError(
                f"monitor output contains unknown fields: {', '.join(unknown)}"
            )
        if missing:
            raise MonitorOutputError(
                f"monitor output is missing fields: {', '.join(missing)}"
            )
        emergency = payload["emergency_stop"]
        if type(emergency) is not bool:
            raise MonitorOutputError("emergency_stop must be a JSON boolean")
        reason = payload["emergency_reason"]
        if emergency:
            if not isinstance(reason, str) or not reason.strip():
                raise MonitorOutputError(
                    "emergency_reason must be a non-empty string when "
                    "emergency_stop is true"
                )
            reason = reason.strip()
        elif reason is not None:
            raise MonitorOutputError(
                "emergency_reason must be null when emergency_stop is false"
            )

        return cls(
            emergency_stop=emergency,
            emergency_reason=reason,
            assessment_payload={
                "task_status": payload["task_status"],
                "criteria": payload["criteria"],
                "failure": payload["failure"],
                "observation": payload["observation"],
            },
        )


def _strict_json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise MonitorOutputError("monitor output must be a non-empty JSON string")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MonitorOutputError(
                    f"monitor output contains duplicate field: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise MonitorOutputError(
            f"monitor output contains non-standard JSON constant: {value}"
        )

    try:
        value = json.loads(
            raw,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except MonitorOutputError:
        raise
    except json.JSONDecodeError as error:
        raise MonitorOutputError(
            f"monitor output is not strict JSON: {error.msg}"
        ) from error
    if not isinstance(value, dict):
        raise MonitorOutputError("monitor output must be one JSON object")
    return value


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list) and len(content) == 1:
        block = content[0]
        if (
            isinstance(block, Mapping)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            return block["text"]
    raise MonitorOutputError(
        "monitor response must contain exactly one text JSON payload"
    )


def _publication_id(task: Any) -> str:
    explicit = getattr(task, "publication_id", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    # Compatibility with the pre-MPC contract.  A generation counter still
    # fences two publications even if these fields happen to repeat.
    return (
        f"{getattr(task, 'plan_id')}:{getattr(task, 'revision')}:"
        f"{getattr(task, 'step_id')}:{getattr(task, 'published_at')}"
    )


def _task_prompt(task: Any) -> str:
    criteria = [
        {
            "id": criterion.criterion_id,
            "criterion": criterion.description,
        }
        for criterion in task.expected_observation
    ]
    phase = getattr(task.phase, "value", task.phase)
    payload = {
        "publication_id": _publication_id(task),
        "plan_id": task.plan_id,
        "plan_revision": task.revision,
        "task_id": task.step_id,
        "phase": phase,
        "instruction": task.instruction,
        "expected_observation": criteria,
        "known_failure_conditions": list(task.known_failure_conditions),
    }
    return (
        "Assess the current camera frame against this exact published task. "
        "Return only the JSON object required by the system prompt.\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build_monitor_messages(
    task: Any,
    frame: CapturedFrame,
    system_prompt: str,
) -> list[Any]:
    """Build messages in the required system -> image -> task-text order."""

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                dict(frame.image_block),
                {"type": "text", "text": _task_prompt(task)},
            ]
        ),
    ]


def _validate_task_shape(task: Any) -> None:
    required = (
        "plan_id",
        "revision",
        "step_id",
        "phase",
        "instruction",
        "expected_observation",
        "known_failure_conditions",
        "published_at",
    )
    missing = [name for name in required if not hasattr(task, name)]
    if missing:
        raise TypeError(
            f"task is missing published-task fields: {', '.join(missing)}"
        )
    if not tuple(task.expected_observation):
        raise ValueError("published task must contain expected observations")


def _make_assessment(
    task: Any,
    frame: CapturedFrame,
    payload: Mapping[str, Any],
) -> MonitorAssessment:
    kwargs: dict[str, Any] = {
        "plan_id": task.plan_id,
        "revision": task.revision,
        "step_id": task.step_id,
        "observed_at": frame.observed_at,
    }
    parameters = inspect.signature(
        MonitorAssessment.from_model_output
    ).parameters
    if "publication_id" in parameters:
        kwargs["publication_id"] = _publication_id(task)
    if "frame_sequence" in parameters:
        kwargs["frame_sequence"] = frame.sequence
    assessment = MonitorAssessment.from_model_output(payload, **kwargs)
    expected_order = tuple(
        criterion.criterion_id for criterion in task.expected_observation
    )
    actual_order = tuple(
        criterion.criterion_id for criterion in assessment.criteria
    )
    if actual_order != expected_order:
        expected_ids = set(expected_order)
        actual_ids = set(actual_order)
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        details = []
        if missing:
            details.append(f"missing criterion IDs: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected criterion IDs: {', '.join(extra)}")
        if not missing and not extra:
            details.append("criterion IDs are not in the published order")
        raise MonitorOutputError("; ".join(details))
    return assessment


class MonitorService:
    """Monitor the latest frame for the latest task with one inference in flight."""

    def __init__(
        self,
        on_assessment: Callable[[MonitorAssessment], None],
        *,
        on_error: Callable[[MonitorErrorEvent], None] | None = None,
        on_event: Callable[[MonitorTelemetryEvent], None] | None = None,
        emergency: EmergencyStopCoordinator | None = None,
        model: MonitorModel | None = None,
        frame_source: FrameSource | None = None,
        system_prompt: str | None = None,
        model_name: str = DEFAULT_MONITOR_MODEL,
        model_base_url: str = DEFAULT_MONITOR_BASE_URL,
        snapshot_url: str = DEFAULT_LIVE_FRAME_URL,
        min_interval_seconds: float = 1.0,
        request_timeout: float = 60.0,
        metrics: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(on_assessment):
            raise TypeError("on_assessment must be callable")
        if on_error is not None and not callable(on_error):
            raise TypeError("on_error must be callable or None")
        if on_event is not None and not callable(on_event):
            raise TypeError("on_event must be callable or None")
        if emergency is not None and not isinstance(
            emergency,
            EmergencyStopCoordinator,
        ):
            raise TypeError("emergency must be an EmergencyStopCoordinator")
        if (
            isinstance(min_interval_seconds, bool)
            or not isinstance(min_interval_seconds, (int, float))
            or not math.isfinite(min_interval_seconds)
            or min_interval_seconds < 0
        ):
            raise ValueError("min_interval_seconds cannot be negative")
        if (
            isinstance(request_timeout, bool)
            or not isinstance(request_timeout, (int, float))
            or not math.isfinite(request_timeout)
            or request_timeout <= 0
        ):
            raise ValueError("request_timeout must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._on_assessment = on_assessment
        self._on_error = on_error
        self._on_event = on_event
        self.emergency = emergency or EmergencyStopCoordinator(clock=clock)
        self._clock = clock
        self._min_interval = float(min_interval_seconds)
        self._metrics = metrics
        self._system_prompt = (
            DEFAULT_MONITOR_PROMPT.read_text(encoding="utf-8")
            if system_prompt is None
            else system_prompt
        )
        if not isinstance(self._system_prompt, str) or not self._system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        self._model = model or ChatOpenAI(
            model=model_name,
            api_key="EMPTY",
            base_url=model_base_url,
            max_tokens=768,
            temperature=0,
            streaming=False,
            timeout=float(request_timeout),
        )
        self._frame_source = frame_source or HTTPFrameSource(
            snapshot_url,
            clock=clock,
        )
        if not callable(self._frame_source):
            raise TypeError("frame_source must be callable")

        self._condition = threading.Condition()
        self._active_task: Any | None = None
        self._generation = 0
        self._stopping = False
        self._thread: threading.Thread | None = None

    @property
    def shutdown_event(self) -> threading.Event:
        return self.emergency.shutdown_event

    @property
    def running(self) -> bool:
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    @property
    def active_publication_id(self) -> str | None:
        with self._condition:
            if self._active_task is None:
                return None
            return _publication_id(self._active_task)

    def start(self) -> None:
        with self._condition:
            if self._stopping:
                raise RuntimeError("monitor service has been stopped")
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run,
                name="prefmem-live-monitor",
                daemon=True,
            )
            self._thread.start()

    def publish(self, task: Any) -> None:
        """Replace the single current task and wake the monitor worker."""

        _validate_task_shape(task)
        if self.emergency.latched:
            raise EmergencyStopLatchedError(
                "cannot publish after an emergency stop has been latched"
            )
        self.start()
        with self._condition:
            if self._stopping:
                raise RuntimeError("monitor service has been stopped")
            if self.emergency.latched:
                raise EmergencyStopLatchedError(
                    "cannot publish after an emergency stop has been latched"
                )
            self._generation += 1
            self._active_task = task
            self._condition.notify_all()

    __call__ = publish

    def retire(self, publication_id: str | None = None) -> bool:
        """Clear a task, optionally only if its publication identity matches."""

        with self._condition:
            if self._active_task is None:
                return False
            if (
                publication_id is not None
                and publication_id != _publication_id(self._active_task)
            ):
                return False
            self._generation += 1
            self._active_task = None
            self._condition.notify_all()
            return True

    def stop(self, *, join_timeout: float = 5.0) -> None:
        if join_timeout < 0:
            raise ValueError("join_timeout cannot be negative")
        with self._condition:
            self._stopping = True
            self._generation += 1
            self._active_task = None
            thread = self._thread
            self._condition.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=join_timeout)

    close = stop

    def _run(self) -> None:
        seen_generation: int | None = None
        last_attempt_started: float | None = None
        while True:
            with self._condition:
                while (
                    self._active_task is None
                    and not self._stopping
                    and not self.emergency.latched
                ):
                    self._condition.wait(timeout=0.25)
                if self._stopping or self.emergency.latched:
                    return
                task = self._active_task
                generation = self._generation

            if generation != seen_generation:
                seen_generation = generation
                last_attempt_started = None

            if last_attempt_started is not None:
                remaining = (
                    last_attempt_started
                    + self._min_interval
                    - self._clock()
                )
                if remaining > 0:
                    with self._condition:
                        self._condition.wait(timeout=min(remaining, 0.25))
                    continue

            last_attempt_started = self._clock()
            try:
                frame = self._frame_source()
                if not isinstance(frame, CapturedFrame):
                    raise TypeError(
                        "frame_source must return a CapturedFrame"
                    )
            except BaseException as error:
                self._emit_error_if_current(
                    task,
                    generation,
                    MonitorErrorKind.FRAME,
                    error,
                )
                continue

            if not self._is_current(task, generation):
                continue
            publication_sequence = getattr(task, "frame_sequence", None)
            if (
                publication_sequence is not None
                and frame.sequence is not None
                and frame.sequence <= publication_sequence
            ):
                # No inference is started on a pre-publication frame.
                continue

            messages = build_monitor_messages(
                task,
                frame,
                self._system_prompt,
            )
            self._emit_telemetry(
                MonitorTelemetryEvent(
                    kind=MonitorTelemetryKind.INFERENCE_STARTED,
                    publication_id=_publication_id(task),
                    frame_sequence=frame.sequence,
                    observed_at=frame.observed_at,
                )
            )
            model_started = time.perf_counter()
            try:
                response = self._model.invoke(
                    messages,
                    **THINKING_DISABLED_OPTIONS,
                )
            except BaseException as error:
                self._emit_error_if_current(
                    task,
                    generation,
                    MonitorErrorKind.MODEL,
                    error,
                )
                continue
            elapsed = time.perf_counter() - model_started
            self._emit_telemetry(
                MonitorTelemetryEvent(
                    kind=MonitorTelemetryKind.INFERENCE_COMPLETED,
                    publication_id=_publication_id(task),
                    frame_sequence=frame.sequence,
                    observed_at=frame.observed_at,
                    elapsed_seconds=elapsed,
                )
            )

            if self._metrics is not None:
                try:
                    self._metrics.record(
                        agent="Monitor Agent",
                        prompt_messages=messages,
                        response=response,
                        elapsed_seconds=elapsed,
                    )
                except MetricsRecordingError as error:
                    self._emit_error_if_current(
                        task,
                        generation,
                        MonitorErrorKind.RECORDING,
                        error,
                    )
                    continue
                except BaseException:
                    LOGGER.exception("Could not record monitor metrics")

            try:
                parsed = MonitorModelOutput.from_json(
                    _response_text(response)
                )
            except BaseException as error:
                self._emit_error_if_current(
                    task,
                    generation,
                    MonitorErrorKind.OUTPUT,
                    error,
                )
                continue

            # Emergency is scene-level and is deliberately checked before
            # ordinary task fencing and assessment-schema validation.  An
            # in-flight task replacement must not suppress visible danger.
            if parsed.emergency_stop:
                assert parsed.emergency_reason is not None
                self.emergency.trigger(
                    parsed.emergency_reason,
                    publication_id=_publication_id(task),
                    observed_at=frame.observed_at,
                )
                continue

            if not self._is_current(task, generation):
                continue
            try:
                assessment = _make_assessment(
                    task,
                    frame,
                    parsed.assessment_payload,
                )
            except BaseException as error:
                self._emit_error_if_current(
                    task,
                    generation,
                    MonitorErrorKind.OUTPUT,
                    error,
                )
                continue

            if not self._is_current(task, generation):
                continue
            try:
                self._on_assessment(assessment)
            except BaseException as error:
                self._emit_error_if_current(
                    task,
                    generation,
                    MonitorErrorKind.CALLBACK,
                    error,
                )

    def _is_current(self, task: Any, generation: int) -> bool:
        with self._condition:
            return (
                not self._stopping
                and not self.emergency.latched
                and generation == self._generation
                and self._active_task is task
                and _publication_id(self._active_task) == _publication_id(task)
            )

    def _emit_error_if_current(
        self,
        task: Any,
        generation: int,
        kind: MonitorErrorKind,
        error: BaseException,
    ) -> None:
        with self._condition:
            if (
                self._stopping
                or self.emergency.latched
                or generation != self._generation
                or self._active_task is not task
                or _publication_id(self._active_task) != _publication_id(task)
            ):
                return
            event = MonitorErrorEvent(
                kind=kind,
                message=str(error) or error.__class__.__name__,
                publication_id=_publication_id(task),
            )
            # A camera, model, parser, or integration error is a pause, not a
            # task FAIL.  Retire this publication before notifying the runtime
            # so no costly retry races the NEEDS_ATTENTION transition.
            self._generation += 1
            self._active_task = None
            self._condition.notify_all()
        self._emit_telemetry(
            MonitorTelemetryEvent(
                kind=MonitorTelemetryKind.ERROR,
                publication_id=_publication_id(task),
                message=f"{event.kind.value}: {event.message}",
            )
        )
        if self._on_error is None:
            LOGGER.error("%s: %s", event.kind.value, event.message)
            return
        try:
            self._on_error(event)
        except BaseException:
            LOGGER.exception("The monitor error callback raised an error")

    def _emit_telemetry(self, event: MonitorTelemetryEvent) -> None:
        callback = self._on_event
        if callback is None:
            return
        try:
            callback(event)
        except BaseException:
            # Telemetry is deliberately best-effort and can never interrupt
            # observation, emergency handling, or controller callbacks.
            LOGGER.exception("The monitor telemetry callback raised an error")
