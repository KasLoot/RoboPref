"""Frozen-checklist compilation and live final-state validation.

The Validator has two deliberately separate model boundaries:

* :meth:`ValidatorAgent.compile_contract` expands the frozen goal's broad
  completion labels into concrete visual criteria once, at confirmation time.
* :class:`ValidatorService` assesses that immutable contract from fresh camera
  frames when the controller publishes a final-validation task.

The service is single-flight and keeps only one publication.  Within that
publication it retains the newest decisive (``MET`` or ``NOT_MET``) evidence
for every detailed criterion, so an operator can sweep a camera across several
relevant regions.  An ``UNKNOWN`` view never erases evidence already visible
in an earlier view.  The executor must not manipulate the scene during this
validation sweep; otherwise evidence from different world states would be
combined.

Overall and broad-item statuses are never model generated.  They are derived
by the trusted contracts layer after this module returns criterion evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import logging
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Protocol, TYPE_CHECKING

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from prefmem.agents.monitor import CapturedFrame, FrameSource, HTTPFrameSource
from prefmem.agents.vision import DEFAULT_LIVE_FRAME_URL
from prefmem.emergency import EmergencyStopCoordinator

if TYPE_CHECKING:
    from prefmem.contracts import (
        GoalContract,
        PublishedTask,
        ValidationAssessment,
        ValidationContract,
    )


LOGGER = logging.getLogger(__name__)
DEFAULT_VALIDATOR_MODEL = "/workspace/models/gemma-4-26B-A4B-it"
DEFAULT_VALIDATOR_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VALIDATOR_PROMPT = (
    Path(__file__).resolve().parent
    / "prompt"
    / "validator"
    / "validator-prompt-v1.md"
)
THINKING_DISABLED_OPTIONS = {
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
}


class ValidatorOutputError(ValueError):
    """The Validator returned data outside its strict JSON contract."""


class ValidatorEmergencyStopLatchedError(RuntimeError):
    """Final validation cannot start after the emergency latch is set."""


class ValidatorEmergencyRequest(RuntimeError):
    """Private control-flow signal for a valid model emergency request."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ValidatorErrorKind(str, Enum):
    FRAME = "FRAME_ERROR"
    MODEL = "MODEL_ERROR"
    OUTPUT = "OUTPUT_ERROR"
    CALLBACK = "CALLBACK_ERROR"


@dataclass(frozen=True, slots=True)
class ValidatorErrorEvent:
    """A system error; it is not evidence that the high-level goal failed."""

    kind: ValidatorErrorKind
    message: str
    publication_id: str | None


class ValidatorModel(Protocol):
    def invoke(self, messages: list[Any], **kwargs: Any) -> Any: ...


_THINKING_BLOCK = re.compile(
    r"<(?P<tag>think|thinking|analysis|reasoning)>.*?</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_JSON_FENCE = re.compile(
    r"```(?:json)?\s*(?P<body>.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _contract_module() -> Any:
    from prefmem import contracts

    return contracts


def _strict_json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise ValidatorOutputError(
            "validator output must be a non-empty JSON string"
        )

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidatorOutputError(
                    f"validator output contains duplicate field: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValidatorOutputError(
            "validator output contains non-standard JSON constant: "
            f"{value}"
        )

    try:
        value = json.loads(
            raw,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except ValidatorOutputError:
        raise
    except json.JSONDecodeError as error:
        raise ValidatorOutputError(
            f"validator output is not strict JSON: {error.msg}"
        ) from error
    if not isinstance(value, dict):
        raise ValidatorOutputError(
            "validator output must be one JSON object"
        )
    return value


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        if content.strip():
            return content
        raise ValidatorOutputError("validator returned an empty response")
    if isinstance(content, Sequence) and not isinstance(
        content,
        (str, bytes),
    ):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, Mapping):
                continue
            block_type = block.get("type")
            if block_type in {"reasoning", "thinking", "analysis"}:
                continue
            text = block.get("text")
            if not isinstance(text, str):
                text = block.get("content")
            if isinstance(text, str):
                parts.append(text)
        result = "".join(parts)
        if result.strip():
            return result
    raise ValidatorOutputError(
        "validator response did not contain visible text"
    )


def parse_validator_json_object(response: Any) -> dict[str, Any]:
    """Remove known Gemma transport wrappers, then decode strict JSON.

    This accepts plain JSON or one whole-response JSON fence after removing
    complete thinking tags.  It deliberately rejects prose outside a fence,
    multiple fences, arbitrary prefixes, duplicate keys, non-standard numeric
    constants, trailing content, and non-object top-level values.
    """

    text = _THINKING_BLOCK.sub("", _response_text(response)).strip()
    fences = list(_JSON_FENCE.finditer(text))
    if fences:
        if len(fences) != 1:
            raise ValidatorOutputError(
                "validator response contains multiple JSON fences"
            )
        if _JSON_FENCE.sub("", text).strip():
            raise ValidatorOutputError(
                "validator response contains content outside its JSON fence"
            )
        text = fences[0].group("body").strip()
    return _strict_json_object(text)


def _model_payload(value: Any, field_name: str) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(f"{field_name} must expose to_dict()")
    payload = to_dict()
    if not isinstance(payload, Mapping):
        raise TypeError(f"{field_name}.to_dict() must return a mapping")
    return dict(payload)


def _image_message(frame: CapturedFrame, request: Mapping[str, Any]) -> HumanMessage:
    if not isinstance(frame, CapturedFrame):
        raise TypeError("frame must be a CapturedFrame")
    return HumanMessage(
        content=[
            dict(frame.image_block),
            {
                "type": "text",
                "text": json.dumps(
                    request,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ]
    )


def _goal_key(goal_contract: Any) -> tuple[str, int]:
    try:
        return (goal_contract.goal_id, goal_contract.revision)
    except AttributeError as error:
        raise TypeError("goal_contract must be a GoalContract") from error


def _default_validation_id(goal_contract: Any) -> str:
    goal_id, revision = _goal_key(goal_contract)
    return f"{goal_id}:r{revision}:validation"


def build_compilation_messages(
    goal_contract: Any,
    frame: CapturedFrame,
    system_prompt: str,
    *,
    correction: str | None = None,
) -> list[Any]:
    """Build a compilation call in system -> image -> request-text order."""

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    broad = list(goal_contract.final_expected_observation)
    required_indices = list(range(len(broad)))
    request: dict[str, Any] = {
        "mode": "COMPILE_CHECKLIST",
        "frozen_goal": {
            "goal_id": goal_contract.goal_id,
            "revision": goal_contract.revision,
            "goal": goal_contract.goal,
            "constraints": list(goal_contract.constraints),
            "dynamic_object_scope": (
                None
                if goal_contract.dynamic_object_scope is None
                else goal_contract.dynamic_object_scope.to_dict()
            ),
        },
        "broad_items": [
            {"broad_index": index, "label": label}
            for index, label in enumerate(broad)
        ],
        "required_broad_item_count": len(broad),
        "required_broad_index_sequence": required_indices,
        "request": (
            f"Return exactly {len(broad)} broad_items using the exact JSON "
            "schema from the system prompt. Copy "
            f"required_broad_index_sequence {required_indices} exactly and "
            "keep all detailed criteria for one broad outcome inside its "
            "single broad_items entry."
        ),
    }
    if correction is not None:
        request["schema_correction"] = (
            "The previous response was rejected: "
            f"{correction}. The required broad_index sequence is exactly "
            f"{required_indices}; return exactly {len(broad)} entries, one "
            "entry per index. Never create multiple broad_items entries for "
            "different criteria of the same outcome; keep those criteria in "
            "that entry's detailed_criteria array. Return one corrected "
            "strict JSON object only."
        )
    return [
        SystemMessage(content=system_prompt),
        _image_message(frame, request),
    ]


def _has_duplicate_broad_indices(payload: Mapping[str, Any] | None) -> bool:
    """Identify the one model mapping error with a safe host fallback.

    Duplicate indices make the model-authored criteria ambiguous, so those
    criteria are never reordered or reused.  This predicate only decides
    whether both batch attempts are eligible for the frozen-goal fallback.
    """

    if not isinstance(payload, Mapping):
        return False
    broad_items = payload.get("broad_items")
    if isinstance(broad_items, (str, bytes)) or not isinstance(
        broad_items,
        Sequence,
    ):
        return False
    indices: list[int] = []
    for item in broad_items:
        if not isinstance(item, Mapping):
            return False
        index = item.get("broad_index")
        if isinstance(index, bool) or not isinstance(index, int):
            return False
        indices.append(index)
    return bool(indices) and len(indices) != len(set(indices))


def _frozen_goal_fallback_contract(
    goal_contract: Any,
    *,
    validation_id: str,
) -> Any:
    """Build a conservative minimum checklist from trusted frozen outcomes.

    This is used only after the model repeats ambiguous broad-index mappings
    twice.  Each host-owned broad outcome becomes its own criterion; no
    model-authored criterion is guessed, dropped, or attached by position.
    """

    contracts = _contract_module()
    draft = contracts.ValidationChecklistDraft.from_model_output(
        {
            "broad_items": [
                {
                    "broad_index": index,
                    "detailed_criteria": [label],
                }
                for index, label in enumerate(
                    goal_contract.final_expected_observation
                )
            ]
        }
    )
    return contracts.freeze_validation_contract(
        goal_contract,
        draft,
        validation_id=validation_id,
    )


def build_assessment_messages(
    validation_contract: Any,
    frame: CapturedFrame,
    system_prompt: str,
    *,
    publication_id: str,
) -> list[Any]:
    """Build an assessment call in system -> image -> request-text order."""

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    request = {
        "mode": "ASSESS_FINAL_STATE",
        "publication_id": publication_id,
        "validation_contract": _model_payload(
            validation_contract,
            "validation_contract",
        ),
        "request": (
            "Assess every immutable detailed criterion against this current "
            "frame and return the exact JSON schema from the system prompt."
        ),
    }
    return [
        SystemMessage(content=system_prompt),
        _image_message(frame, request),
    ]


def _publication_id(task: Any) -> str:
    value = getattr(task, "publication_id", None)
    if not isinstance(value, str) or not value.strip():
        raise TypeError("task must have a non-empty publication_id")
    return value.strip()


def _validate_final_task(task: Any) -> None:
    required = (
        "plan_id",
        "revision",
        "step_id",
        "phase",
        "published_at",
        "publication_id",
        "frame_sequence",
    )
    missing = [name for name in required if not hasattr(task, name)]
    if missing:
        raise TypeError(
            "task is missing published-task fields: " + ", ".join(missing)
        )
    phase = getattr(task.phase, "value", task.phase)
    if phase != "FINAL_VALIDATION":
        raise ValueError(
            "ValidatorService only accepts FINAL_VALIDATION tasks"
        )
    _publication_id(task)


def _validate_contract_shape(validation_contract: Any) -> None:
    expected_type = getattr(_contract_module(), "ValidationContract")
    if not isinstance(validation_contract, expected_type):
        raise TypeError("validation_contract must be a ValidationContract")


@dataclass(frozen=True, slots=True)
class _AssessmentEnvelope:
    emergency_stop: bool
    emergency_reason: str | None
    payload: Mapping[str, Any]

    @classmethod
    def from_response(cls, response: Any) -> _AssessmentEnvelope:
        payload = parse_validator_json_object(response)
        missing = {
            "emergency_stop",
            "emergency_reason",
        } - set(payload)
        if missing:
            raise ValidatorOutputError(
                "validator output is missing fields: "
                + ", ".join(sorted(missing))
            )
        emergency = payload.pop("emergency_stop")
        reason = payload.pop("emergency_reason")
        if type(emergency) is not bool:
            raise ValidatorOutputError(
                "emergency_stop must be a JSON boolean"
            )
        if emergency:
            if not isinstance(reason, str) or not reason.strip():
                raise ValidatorOutputError(
                    "emergency_reason must be a non-empty string when "
                    "emergency_stop is true"
                )
            reason = reason.strip()
        elif reason is not None:
            raise ValidatorOutputError(
                "emergency_reason must be null when emergency_stop is false"
            )
        return cls(
            emergency_stop=emergency,
            emergency_reason=reason,
            payload=payload,
        )


class ValidatorAgent:
    """VLM boundary for checklist compilation and one-frame assessment."""

    def __init__(
        self,
        *,
        model: ValidatorModel | None = None,
        system_prompt: str | None = None,
        model_name: str = DEFAULT_VALIDATOR_MODEL,
        model_base_url: str = DEFAULT_VALIDATOR_BASE_URL,
        request_timeout: float = 60.0,
        metrics: Any | None = None,
    ) -> None:
        if (
            isinstance(request_timeout, bool)
            or not isinstance(request_timeout, (int, float))
            or not math.isfinite(request_timeout)
            or request_timeout <= 0
        ):
            raise ValueError("request_timeout must be positive")
        self.system_prompt = (
            DEFAULT_VALIDATOR_PROMPT.read_text(encoding="utf-8")
            if system_prompt is None
            else system_prompt
        )
        if (
            not isinstance(self.system_prompt, str)
            or not self.system_prompt.strip()
        ):
            raise ValueError("system_prompt must be a non-empty string")
        self.model = model or ChatOpenAI(
            model=model_name,
            api_key="EMPTY",
            base_url=model_base_url,
            max_tokens=2048,
            temperature=0,
            streaming=False,
            timeout=float(request_timeout),
        )
        self.metrics = metrics
        self._compile_lock = threading.Lock()
        self._compiled: dict[tuple[str, int], tuple[str, Any]] = {}

    def compile_contract(
        self,
        goal_contract: GoalContract,
        confirmation_frame: CapturedFrame,
        *,
        validation_id: str | None = None,
    ) -> ValidationContract:
        """Compile and freeze one checklist per goal revision.

        The first valid result is cached.  Later calls for the same immutable
        goal revision return that exact object without another model call.
        """

        if not isinstance(confirmation_frame, CapturedFrame):
            raise TypeError("confirmation_frame must be a CapturedFrame")
        goal_payload = _model_payload(goal_contract, "goal_contract")
        signature = json.dumps(
            goal_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        key = _goal_key(goal_contract)
        requested_id = validation_id or _default_validation_id(goal_contract)

        with self._compile_lock:
            cached = self._compiled.get(key)
            if cached is not None:
                cached_signature, contract = cached
                if cached_signature != signature:
                    raise ValueError(
                        "the same goal ID/revision was reused with different "
                        "frozen content"
                    )
                if getattr(contract, "validation_id", requested_id) != requested_id:
                    raise ValueError(
                        "the goal revision already has a different validation ID"
                    )
                return contract

            correction: str | None = None
            first_error: BaseException | None = None
            duplicate_mapping_failures: list[bool] = []
            for attempt in range(2):
                messages = build_compilation_messages(
                    goal_contract,
                    confirmation_frame,
                    self.system_prompt,
                    correction=correction,
                )
                response, elapsed = self._invoke(messages)
                self._record_metrics(messages, response, elapsed)
                payload: dict[str, Any] | None = None
                try:
                    payload = parse_validator_json_object(response)
                    draft = _contract_module().ValidationChecklistDraft.from_model_output(
                        payload
                    )
                    contract = _contract_module().freeze_validation_contract(
                        goal_contract,
                        draft,
                        validation_id=requested_id,
                    )
                except (ValueError, TypeError, ValidatorOutputError) as error:
                    duplicate_mapping_failures.append(
                        _has_duplicate_broad_indices(payload)
                    )
                    if attempt == 0:
                        first_error = error
                        correction = str(error) or error.__class__.__name__
                        continue
                    if all(duplicate_mapping_failures):
                        contract = _frozen_goal_fallback_contract(
                            goal_contract,
                            validation_id=requested_id,
                        )
                        LOGGER.warning(
                            "Validator repeated ambiguous broad-index mappings; "
                            "using one trusted frozen-goal criterion per outcome "
                            "for goal %s revision %s",
                            goal_contract.goal_id,
                            goal_contract.revision,
                        )
                        self._compiled[key] = (signature, contract)
                        return contract
                    first_message = (
                        str(first_error)
                        if first_error is not None
                        else "unknown schema error"
                    )
                    raise ValidatorOutputError(
                        "validator checklist compilation failed its JSON "
                        f"contract twice: first: {first_message}; second: {error}"
                    ) from error
                self._compiled[key] = (signature, contract)
                return contract

        raise AssertionError("unreachable checklist compilation state")

    def assess_contract(
        self,
        validation_contract: ValidationContract,
        current_frame: CapturedFrame,
        *,
        publication_id: str,
        observed_at: float | None = None,
        frame_sequence: int | None = None,
    ) -> ValidationAssessment:
        """Assess all detailed criteria once, with a trusted host envelope.

        A valid emergency envelope raises :class:`ValidatorEmergencyRequest`
        before ordinary assessment parsing, allowing the service to latch the
        shared stop coordinator immediately.
        """

        _validate_contract_shape(validation_contract)
        if not isinstance(current_frame, CapturedFrame):
            raise TypeError("current_frame must be a CapturedFrame")
        if not isinstance(publication_id, str) or not publication_id.strip():
            raise ValueError("publication_id must be a non-empty string")
        if observed_at is None:
            observed_at = current_frame.observed_at
        if frame_sequence is None:
            frame_sequence = current_frame.sequence
        messages = build_assessment_messages(
            validation_contract,
            current_frame,
            self.system_prompt,
            publication_id=publication_id.strip(),
        )
        response, elapsed = self._invoke(messages)
        self._record_metrics(messages, response, elapsed)
        envelope = _AssessmentEnvelope.from_response(response)
        if envelope.emergency_stop:
            assert envelope.emergency_reason is not None
            raise ValidatorEmergencyRequest(envelope.emergency_reason)
        try:
            return _contract_module().ValidationAssessment.from_model_output(
                validation_contract,
                envelope.payload,
                publication_id=publication_id.strip(),
                observed_at=observed_at,
                frame_sequence=frame_sequence,
            )
        except (TypeError, ValueError) as error:
            raise ValidatorOutputError(str(error)) from error

    def _invoke(self, messages: list[Any]) -> tuple[Any, float]:
        started = time.perf_counter()
        response = self.model.invoke(
            messages,
            **THINKING_DISABLED_OPTIONS,
        )
        return response, time.perf_counter() - started

    def _record_metrics(
        self,
        messages: list[Any],
        response: Any,
        elapsed: float,
    ) -> None:
        if self.metrics is None:
            return
        try:
            self.metrics.record(
                agent="Validator Agent",
                prompt_messages=messages,
                response=response,
                elapsed_seconds=elapsed,
            )
        except BaseException:
            LOGGER.exception("Could not record validator metrics")


@dataclass(frozen=True, slots=True)
class _ValidationJob:
    task: Any
    contract: Any


class ValidatorService:
    """Continuously validate the latest final-validation publication."""

    def __init__(
        self,
        on_assessment: Callable[[ValidationAssessment], None],
        *,
        on_error: Callable[[ValidatorErrorEvent], None] | None = None,
        emergency: EmergencyStopCoordinator | None = None,
        validator: ValidatorAgent | None = None,
        model: ValidatorModel | None = None,
        frame_source: FrameSource | None = None,
        system_prompt: str | None = None,
        model_name: str = DEFAULT_VALIDATOR_MODEL,
        model_base_url: str = DEFAULT_VALIDATOR_BASE_URL,
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
        if emergency is not None and not isinstance(
            emergency,
            EmergencyStopCoordinator,
        ):
            raise TypeError("emergency must be an EmergencyStopCoordinator")
        if validator is not None and (
            model is not None or system_prompt is not None or metrics is not None
        ):
            raise ValueError(
                "validator cannot be combined with model, system_prompt, or metrics"
            )
        if (
            isinstance(min_interval_seconds, bool)
            or not isinstance(min_interval_seconds, (int, float))
            or not math.isfinite(min_interval_seconds)
            or min_interval_seconds < 0
        ):
            raise ValueError("min_interval_seconds cannot be negative")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._on_assessment = on_assessment
        self._on_error = on_error
        self.emergency = emergency or EmergencyStopCoordinator(clock=clock)
        self.validator = validator or ValidatorAgent(
            model=model,
            system_prompt=system_prompt,
            model_name=model_name,
            model_base_url=model_base_url,
            request_timeout=request_timeout,
            metrics=metrics,
        )
        self._frame_source = frame_source or HTTPFrameSource(
            snapshot_url,
            clock=clock,
        )
        if not callable(self._frame_source):
            raise TypeError("frame_source must be callable")
        self._clock = clock
        self._min_interval = float(min_interval_seconds)

        self._condition = threading.Condition()
        self._active_job: _ValidationJob | None = None
        self._generation = 0
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._accumulated: dict[str, Any] = {}
        self._last_observed_at: float | None = None
        self._last_frame_sequence: int | None = None

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
            if self._active_job is None:
                return None
            return _publication_id(self._active_job.task)

    def start(self) -> None:
        with self._condition:
            if self._stopping:
                raise RuntimeError("validator service has been stopped")
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run,
                name="prefmem-final-validator",
                daemon=True,
            )
            self._thread.start()

    def compile_contract(
        self,
        goal_contract: GoalContract,
        confirmation_frame: CapturedFrame,
        *,
        validation_id: str | None = None,
    ) -> ValidationContract:
        """Compile the immutable checklist through this service's Validator."""

        return self.validator.compile_contract(
            goal_contract,
            confirmation_frame,
            validation_id=validation_id,
        )

    def publish(
        self,
        task: PublishedTask,
        validation_contract: ValidationContract,
    ) -> None:
        """Replace the one validation publication and reset accumulated views."""

        _validate_final_task(task)
        _validate_contract_shape(validation_contract)
        frozen_goal = validation_contract.goal_contract
        if (
            task.plan_id != frozen_goal.goal_id
            or task.revision != frozen_goal.revision
        ):
            raise ValueError(
                "final-validation task does not belong to the frozen "
                "validation goal revision"
            )
        if self.emergency.latched:
            raise ValidatorEmergencyStopLatchedError(
                "cannot publish after an emergency stop has been latched"
            )
        self.start()
        with self._condition:
            if self._stopping:
                raise RuntimeError("validator service has been stopped")
            if self.emergency.latched:
                raise ValidatorEmergencyStopLatchedError(
                    "cannot publish after an emergency stop has been latched"
                )
            self._generation += 1
            self._active_job = _ValidationJob(task, validation_contract)
            self._accumulated = {}
            self._last_observed_at = None
            self._last_frame_sequence = None
            self._condition.notify_all()

    def retire(self, publication_id: str | None = None) -> bool:
        with self._condition:
            if self._active_job is None:
                return False
            if (
                publication_id is not None
                and publication_id != _publication_id(self._active_job.task)
            ):
                return False
            self._generation += 1
            self._active_job = None
            self._accumulated = {}
            self._last_observed_at = None
            self._last_frame_sequence = None
            self._condition.notify_all()
            return True

    def stop(self, *, join_timeout: float = 5.0) -> None:
        if join_timeout < 0:
            raise ValueError("join_timeout cannot be negative")
        with self._condition:
            self._stopping = True
            self._generation += 1
            self._active_job = None
            self._accumulated = {}
            self._last_observed_at = None
            self._last_frame_sequence = None
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
                    self._active_job is None
                    and not self._stopping
                    and not self.emergency.latched
                ):
                    self._condition.wait(timeout=0.25)
                if self._stopping or self.emergency.latched:
                    return
                job = self._active_job
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
                    job,
                    generation,
                    ValidatorErrorKind.FRAME,
                    error,
                )
                continue

            if not self._is_current(job, generation):
                continue
            if not self._is_post_publication_frame(job.task, frame):
                continue

            try:
                assessment = self.validator.assess_contract(
                    job.contract,
                    frame,
                    publication_id=_publication_id(job.task),
                    observed_at=frame.observed_at,
                    frame_sequence=frame.sequence,
                )
            except ValidatorEmergencyRequest as emergency:
                # Like Monitor, scene-level danger bypasses stale-publication
                # and ordinary output checks after a valid emergency envelope.
                self.emergency.trigger(
                    emergency.reason,
                    publication_id=_publication_id(job.task),
                    observed_at=frame.observed_at,
                )
                continue
            except ValidatorOutputError as error:
                self._emit_error_if_current(
                    job,
                    generation,
                    ValidatorErrorKind.OUTPUT,
                    error,
                )
                continue
            except BaseException as error:
                self._emit_error_if_current(
                    job,
                    generation,
                    ValidatorErrorKind.MODEL,
                    error,
                )
                continue

            if not self._is_current(job, generation):
                continue
            try:
                accumulated = self._accumulate(
                    job,
                    generation,
                    assessment,
                )
            except BaseException as error:
                self._emit_error_if_current(
                    job,
                    generation,
                    ValidatorErrorKind.OUTPUT,
                    error,
                )
                continue
            if accumulated is None or not self._is_current(job, generation):
                continue
            try:
                self._on_assessment(accumulated)
            except BaseException as error:
                self._emit_error_if_current(
                    job,
                    generation,
                    ValidatorErrorKind.CALLBACK,
                    error,
                )

    @staticmethod
    def _is_post_publication_frame(task: Any, frame: CapturedFrame) -> bool:
        if frame.observed_at <= float(task.published_at):
            return False
        publication_sequence = getattr(task, "frame_sequence", None)
        if (
            publication_sequence is not None
            and frame.sequence is not None
            and frame.sequence <= publication_sequence
        ):
            return False
        return True

    def _is_current(self, job: _ValidationJob, generation: int) -> bool:
        with self._condition:
            return (
                not self._stopping
                and not self.emergency.latched
                and generation == self._generation
                and self._active_job is job
                and _publication_id(self._active_job.task)
                == _publication_id(job.task)
            )

    def _accumulate(
        self,
        job: _ValidationJob,
        generation: int,
        assessment: Any,
    ) -> Any | None:
        criteria = tuple(getattr(assessment, "criteria", ()))
        if not criteria:
            raise ValidatorOutputError(
                "validation assessment contains no criteria"
            )
        with self._condition:
            if (
                generation != self._generation
                or self._active_job is not job
                or self._stopping
                or self.emergency.latched
            ):
                return None
            if (
                self._last_observed_at is not None
                and assessment.observed_at <= self._last_observed_at
            ):
                return None
            if (
                self._last_frame_sequence is not None
                and assessment.frame_sequence is not None
                and assessment.frame_sequence <= self._last_frame_sequence
            ):
                return None
            for criterion in criteria:
                criterion_id = getattr(criterion, "criterion_id", None)
                state = getattr(criterion, "state", None)
                state_value = getattr(state, "value", state)
                if not isinstance(criterion_id, str):
                    raise ValidatorOutputError(
                        "assessment criterion has no criterion_id"
                    )
                existing = self._accumulated.get(criterion_id)
                existing_state = getattr(
                    getattr(existing, "state", None),
                    "value",
                    getattr(existing, "state", None),
                )
                if state_value != "UNKNOWN" or existing_state in {None, "UNKNOWN"}:
                    self._accumulated[criterion_id] = criterion

            ordered_ids = tuple(
                criterion.criterion_id
                for broad in job.contract.broad_items
                for criterion in broad.detailed_criteria
            )
            if set(self._accumulated) != set(ordered_ids):
                raise ValidatorOutputError(
                    "accumulated assessment does not cover the validation contract"
                )
            payload = {
                "criteria": [
                    self._accumulated[criterion_id].to_dict()
                    for criterion_id in ordered_ids
                ],
                "observation": assessment.observation,
            }
            # Rebuild through the contracts boundary so the callback always
            # receives one fully validated object with latest host metadata.
            accumulated = _contract_module().ValidationAssessment.from_model_output(
                job.contract,
                payload,
                publication_id=assessment.publication_id,
                observed_at=assessment.observed_at,
                frame_sequence=assessment.frame_sequence,
            )
            self._last_observed_at = assessment.observed_at
            if assessment.frame_sequence is not None:
                self._last_frame_sequence = assessment.frame_sequence
            return accumulated

    def _emit_error_if_current(
        self,
        job: _ValidationJob,
        generation: int,
        kind: ValidatorErrorKind,
        error: BaseException,
    ) -> None:
        with self._condition:
            if (
                self._stopping
                or self.emergency.latched
                or generation != self._generation
                or self._active_job is not job
                or _publication_id(self._active_job.task)
                != _publication_id(job.task)
            ):
                return
            event = ValidatorErrorEvent(
                kind=kind,
                message=str(error) or error.__class__.__name__,
                publication_id=_publication_id(job.task),
            )
            self._generation += 1
            self._active_job = None
            self._accumulated = {}
            self._last_observed_at = None
            self._last_frame_sequence = None
            self._condition.notify_all()
        if self._on_error is None:
            LOGGER.error("%s: %s", event.kind.value, event.message)
            return
        try:
            self._on_error(event)
        except BaseException:
            LOGGER.exception("The validator error callback raised an error")
