"""Publication engine for the current five-role RoboPref MuJoCo runtime.

This module intentionally implements one narrow, auditable scientific path:

* profile ``T5`` (five distinct upper-role model instances),
* executor ``mujoco``, and
* scenario ``NM02`` (one red-block-to-target-pad manipulation).

The restriction is deliberate.  The production source currently exposes one
concrete receding-horizon robot runtime, not a generic executor for every
catalogue mechanism.  Unsupported profiles and scenarios are rejected before
behavior rather than approximated with scripted or oracle-fed substitutes.

Objective state is read from MuJoCo only after the studied system has acted.
It is never placed in a model prompt or used to choose an action.  Model-facing
frames are PNG encodings of the exact robot-camera pixels written to the video
and frame-request ledger.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit
import uuid

import cv2
import httpx
import numpy as np

from experiments.harness.assembler import (
    AssembledSystem,
    BackendBuildRequest,
    EmbeddingBuildRequest,
    RoleBuildRequest,
    SystemAssembler,
)
from experiments.harness.driver import ScenarioDriverError, jsonable
from experiments.harness.campaign import ScheduleEntry
from experiments.harness.profiles import ExperimentProfile
from experiments.harness.recording import ClockOrigin, RecordingError
from experiments.harness.runner import (
    AttemptContext,
    CaptureToken,
    CellCapability,
    EngineResult,
    HarnessInvalid,
    InfrastructureFailureEvidence,
    InfrastructureInterruption,
    ModelExchangeResult,
    SafetyAbort,
)
from experiments.harness.scenarios import ScenarioSpec, TriggerSpec
from experiments.harness.video import VideoError


SUPPORTED_PROFILE = "T5"
SUPPORTED_EXECUTOR = "mujoco"
SUPPORTED_SCENARIO = "NM02"
_SUPPORTED_TRIGGER = (
    "NM02-trigger-1",
    "episode_initialized",
    "before_goal_proposal",
    "deliver exact manipulation request",
)
_VISUAL_ROLES = frozenset({"hri", "memory", "planner", "monitor", "validator"})


def _propagate_harness_fault(error: Exception) -> None:
    """Prevent evidence failures from being classified as system behavior."""

    if isinstance(
        error,
        (HarnessInvalid, RecordingError, VideoError, ScenarioDriverError),
    ):
        raise error


def _loopback_url(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty URL")
    normalized = value.rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{label} cannot contain credentials, query, or fragment")
    if parsed.hostname.casefold() != "localhost":
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError as error:
            raise ValueError(f"{label} must use a loopback host") from error
        if not address.is_loopback:
            raise ValueError(f"{label} must use a loopback host")
    return normalized


@dataclass(frozen=True, slots=True)
class ProductionEngineConfig:
    """Frozen service and runtime settings for the supported production path."""

    upper_model: str = "/workspace/models/gemma-4-26B-A4B-it"
    upper_model_base_url: str = "http://127.0.0.1:8000/v1"
    embedding_model: str = "/workspace/models/embeddinggemma-300m"
    embedding_model_base_url: str = "http://127.0.0.1:8080/v1"
    execution_model: str = "/workspace/models/gemma-4-26B-A4B-it"
    execution_model_base_url: str = "http://127.0.0.1:8000/v1"
    sam_base_url: str = "http://127.0.0.1:9000"
    sam_threshold: float = 0.5
    render_size: int = 640
    render_hz: float = 10.0
    viewer: bool = False
    realtime: bool = True
    monitor_min_interval: float = 1.0
    success_confirmations: int = 2
    success_stability_seconds: float = 2.0
    failure_confirmations: int = 2
    ongoing_timeout_seconds: float = 30.0
    attempt_timeout_seconds: float = 300.0
    poll_interval_seconds: float = 0.1
    model_timeout_seconds: float = 300.0
    detector_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        for name in ("upper_model", "embedding_model", "execution_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "upper_model_base_url",
            "embedding_model_base_url",
            "execution_model_base_url",
            "sam_base_url",
        ):
            object.__setattr__(
                self,
                name,
                _loopback_url(getattr(self, name), label=name),
            )
        if isinstance(self.render_size, bool) or self.render_size <= 0:
            raise ValueError("render_size must be positive")
        for name in (
            "render_hz",
            "monitor_min_interval",
            "success_stability_seconds",
            "ongoing_timeout_seconds",
            "attempt_timeout_seconds",
            "poll_interval_seconds",
            "model_timeout_seconds",
            "detector_timeout_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        for name in ("success_confirmations", "failure_confirmations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.sam_threshold <= 1:
            raise ValueError("sam_threshold must be between zero and one")


ModelClientFactory = Callable[[str, str, str, float], object]
EmbeddingClientFactory = Callable[[str, str], object]
HealthProbe = Callable[[str, str], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ProductionDependencies:
    """Narrow dependency seams used by offline construction tests.

    The default factories instantiate the real OpenAI-compatible clients.
    Neither default factory performs a request during construction.
    """

    model_client_factory: ModelClientFactory | None = None
    embedding_client_factory: EmbeddingClientFactory | None = None
    health_probe: HealthProbe | None = None


def _default_model_client(
    logical_role: str,
    model: str,
    base_url: str,
    timeout: float,
) -> object:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key="EMPTY",
        base_url=base_url,
        max_tokens=2048 if logical_role != "monitor" else 768,
        temperature=0,
        streaming=False,
        timeout=float(timeout),
    )


def _default_embedding_client(model: str, base_url: str) -> object:
    from prefmem.agents.memory import VLLMEmbeddingGemma

    return VLLMEmbeddingGemma(model=model, base_url=base_url)


def _default_health_probe(service_label: str, base_url: str) -> Mapping[str, Any]:
    path = "/health" if service_label == "sam3.1-detector" else "/models"
    started = time.monotonic()
    try:
        with httpx.Client(timeout=3.0, trust_env=False) as client:
            response = client.get(f"{base_url.rstrip('/')}{path}")
            ok = 200 <= response.status_code < 300
            return {
                "ok": ok,
                "probe": f"GET {path}",
                "status_code": response.status_code,
                "elapsed_seconds": time.monotonic() - started,
            }
    except Exception as error:
        return {
            "ok": False,
            "probe": "GET /models",
            "error_type": type(error).__name__,
            "elapsed_seconds": time.monotonic() - started,
        }


def _transport_failure(error: BaseException) -> bool:
    """Recognize transport errors without mislabelling schema/model failures."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                httpx.TransportError,
                ConnectionError,
                TimeoutError,
            ),
        ):
            return True
        module = type(current).__module__.split(".", 1)[0]
        if module == "openai" and type(current).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
        }:
            return True
        current = current.__cause__ or current.__context__
    return False


def _wire_value(value: Any) -> Any:
    """Convert LangChain/Pydantic transport values to lossless JSON shapes."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HarnessInvalid("non-finite value in model transport")
        return value
    if isinstance(value, Enum):
        return _wire_value(value.value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _wire_value(model_dump(mode="json"))
        except TypeError:
            return _wire_value(model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _wire_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _wire_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_wire_value(child) for child in value]
    if isinstance(value, (bytes, bytearray)):
        return {
            "encoding": "base64",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    return {"python_type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def _png_data_url(token: CaptureToken) -> str:
    return "data:image/png;base64," + base64.b64encode(token.robot_png).decode("ascii")


def _request_image_urls(value: Any) -> tuple[str, ...]:
    """Return serialized image values without interpreting or repairing them."""

    found: list[str] = []
    image_keys = {
        "image",
        "images",
        "image_url",
        "image_urls",
        "input_image",
        "input_images",
    }

    def visit(child: Any, *, image_field: bool = False) -> None:
        if isinstance(child, str):
            if image_field:
                found.append(child)
            return
        if isinstance(child, Mapping):
            block_type = child.get("type")
            image_block = block_type in {
                "image",
                "image_url",
                "input_image",
            }
            for key, nested in child.items():
                normalized_key = str(key).lower()
                visit(
                    nested,
                    image_field=(
                        normalized_key in image_keys
                        or (
                            (image_field or image_block)
                            and normalized_key in {"url", "uri", "data"}
                        )
                    ),
                )
            return
        if isinstance(child, Sequence) and not isinstance(
            child, (str, bytes, bytearray)
        ):
            for nested in child:
                visit(nested, image_field=image_field)

    visit(value)
    return tuple(found)


class _AttemptBinding:
    """Thread-safe bridge from constructed model clients to one attempt."""

    def __init__(self, health_probe: HealthProbe) -> None:
        self._health_probe = health_probe
        self._context: AttemptContext | None = None
        self._lock = threading.RLock()
        self._event_lock = threading.RLock()
        self._thread_local = threading.local()
        self._latest_capture: CaptureToken | None = None
        self._captures_by_image_url_sha256: dict[str, list[CaptureToken]] = {}
        self._pinned_captures: dict[str, list[CaptureToken]] = {}
        self._safe_stop: Callable[[], bool] | None = None
        self._infra_failure: InfrastructureInterruption | None = None
        self._next_call = 1

    def bind_context(self, context: AttemptContext) -> None:
        with self._lock:
            if self._context is not None:
                raise HarnessInvalid("production system was already bound to an attempt")
            self._context = context

    def bind_safe_stop(self, callback: Callable[[], bool]) -> None:
        if not callable(callback):
            raise TypeError("safe-stop callback must be callable")
        with self._lock:
            self._safe_stop = callback

    @property
    def context(self) -> AttemptContext:
        with self._lock:
            if self._context is None:
                raise HarnessInvalid("production model called before attempt binding")
            return self._context

    @property
    def infrastructure_failure(self) -> InfrastructureInterruption | None:
        with self._lock:
            return self._infra_failure

    def set_capture(self, token: CaptureToken) -> None:
        if not isinstance(token, CaptureToken):
            raise TypeError("capture must be a CaptureToken")
        self._thread_local.capture = token
        image_url_sha256 = hashlib.sha256(
            _png_data_url(token).encode("utf-8")
        ).hexdigest()
        with self._lock:
            self._latest_capture = token
            bucket = self._captures_by_image_url_sha256.setdefault(
                image_url_sha256, []
            )
            if not any(candidate is token for candidate in bucket):
                bucket.append(token)

    @contextmanager
    def pin_capture(self, logical_role: str, token: CaptureToken):
        """Pin explicit role-scoped provenance for a nested request graph."""

        if not isinstance(logical_role, str) or not logical_role.strip():
            raise TypeError("capture pin logical role must be non-empty")
        if not isinstance(token, CaptureToken):
            raise TypeError("capture pin must be a CaptureToken")
        image_url_sha256 = hashlib.sha256(
            _png_data_url(token).encode("utf-8")
        ).hexdigest()
        with self._lock:
            registered = any(
                candidate is token
                for candidate in self._captures_by_image_url_sha256.get(
                    image_url_sha256, ()
                )
            )
            if not registered:
                raise HarnessInvalid("cannot pin an unregistered capture token")
            stack = self._pinned_captures.setdefault(logical_role, [])
            stack.append(token)
        try:
            yield token
        finally:
            with self._lock:
                stack = self._pinned_captures.get(logical_role)
                if not stack or stack[-1] is not token:
                    raise HarnessInvalid("capture pin stack was corrupted")
                stack.pop()
                if not stack:
                    del self._pinned_captures[logical_role]

    def capture_for_visual_request(
        self,
        request: Mapping[str, Any],
        *,
        logical_role: str,
    ) -> CaptureToken:
        """Resolve the one registered capture whose exact PNG is on the wire.

        A thread-local/latest-frame lookup is insufficient: an upper-role call
        can retain frame A while a nested planner or monitor call acquires frame
        B on the same thread.  Explicit role-scoped provenance wins, followed
        by a matching thread-local acquisition and then a unique registry
        match.  Unknown images and otherwise ambiguous identities fail closed.
        """

        if not isinstance(logical_role, str) or not logical_role.strip():
            raise TypeError("visual request logical role must be non-empty")

        image_urls = _request_image_urls(request)
        if not image_urls:
            raise HarnessInvalid(
                "visual request did not contain its exact recorded frame"
            )

        matched: dict[int, CaptureToken] = {}
        unregistered = False
        distinct_urls = set(image_urls)
        with self._lock:
            for image_url in distinct_urls:
                image_url_sha256 = hashlib.sha256(
                    image_url.encode("utf-8")
                ).hexdigest()
                candidates = tuple(
                    candidate
                    for candidate in self._captures_by_image_url_sha256.get(
                        image_url_sha256, ()
                    )
                    if _png_data_url(candidate) == image_url
                )
                if not candidates:
                    unregistered = True
                    continue
                for candidate in candidates:
                    matched[id(candidate)] = candidate

            pinned_stack = self._pinned_captures.get(logical_role, ())
            pinned = pinned_stack[-1] if pinned_stack else None
            local = getattr(self._thread_local, "capture", None)

        if unregistered:
            raise HarnessInvalid(
                "visual request contains a tampered or unregistered image; "
                "it cannot be linked to an exact recorded frame"
            )
        if len(distinct_urls) != 1:
            raise HarnessInvalid(
                "visual request is ambiguous across exact recorded frame captures"
            )
        if pinned is not None:
            if id(pinned) in matched:
                return pinned
            raise HarnessInvalid(
                "visual request does not match its role-scoped recorded frame"
            )
        if isinstance(local, CaptureToken) and id(local) in matched:
            return local
        if len(matched) != 1:
            raise HarnessInvalid(
                "visual request is ambiguous across exact recorded frame captures"
            )
        return next(iter(matched.values()))

    def capture(self) -> CaptureToken:
        local = getattr(self._thread_local, "capture", None)
        if isinstance(local, CaptureToken):
            return local
        with self._lock:
            if self._latest_capture is None:
                raise HarnessInvalid("model call has no recorded source frame")
            return self._latest_capture

    def next_call_identity(self) -> tuple[str, str, str]:
        with self._lock:
            index = self._next_call
            self._next_call += 1
        return (
            f"REQ-{index:06d}",
            f"CALL-{index:06d}",
            f"RESP-{index:06d}",
        )

    def emit(self, kind: str, **payload: Any) -> Mapping[str, Any]:
        # ScenarioDriver's oracle/trigger ledgers are intentionally serialized;
        # remote calls themselves remain concurrent across T5 role sessions.
        with self._event_lock:
            return self.context.driver.emit(kind, **_wire_value(payload))

    def classify_transport(
        self,
        error: BaseException,
        *,
        service_label: str,
        endpoint_label: str,
        base_url: str,
    ) -> InfrastructureInterruption | None:
        if not _transport_failure(error):
            return None
        observation = _wire_value(self._health_probe(service_label, base_url))
        if not isinstance(observation, Mapping) or observation.get("ok") is not False:
            # A live service contradicts an infrastructure label.  Preserve the
            # original system/transport error for ordinary attempt diagnosis.
            return None
        with self._lock:
            callback = self._safe_stop
        if callback is None:
            raise HarnessInvalid("transport failed before a safe-stop path was bound")
        try:
            stopped = callback()
        except Exception as stop_error:
            raise HarnessInvalid(
                f"transport failed and safe stop could not be confirmed: {stop_error}"
            ) from stop_error
        if stopped is not True:
            raise HarnessInvalid("transport failed but safe stop was not confirmed")
        interruption = InfrastructureInterruption(
            f"{service_label} transport failed and its independent health probe failed",
            evidence=InfrastructureFailureEvidence(
                service_label=service_label,
                endpoint_label=endpoint_label,
                failure_kind="transport_and_failed_health_probe",
                transport_error_type=type(error).__name__,
                safe_stop_confirmed=True,
                health_observations=(dict(observation),),
            ),
        )
        with self._lock:
            if self._infra_failure is None:
                self._infra_failure = interruption
            else:
                interruption = self._infra_failure
        try:
            self.emit(
                "infrastructure_failure_latched",
                service_label=service_label,
                endpoint_label=endpoint_label,
                transport_error_type=type(error).__name__,
                health_observation=observation,
                safe_stop_confirmed=True,
            )
        except Exception:
            # The model error itself was already recorded by AttemptContext.
            pass
        return interruption


class _RecordedChatBackend:
    """LangChain surface with semantic invocation and evidence recording."""

    def __init__(
        self,
        *,
        target: object,
        root: "_RecordedChatBackend | None" = None,
        binding: _AttemptBinding,
        logical_role: str,
        model_id: str,
        base_url: str,
        service_label: str,
        endpoint_label: str,
        bound_tool_contracts: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self._target = target
        self._root = self if root is None else root
        self._binding = binding
        self.logical_role = logical_role
        self.model_id = model_id
        self.base_url = base_url
        self.service_label = service_label
        self.endpoint_label = endpoint_label
        self._bound_tool_contracts = tuple(
            _wire_value(dict(contract)) for contract in bound_tool_contracts
        )

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_RecordedChatBackend":
        bind = getattr(self._target, "bind_tools", None)
        if not callable(bind):
            raise HarnessInvalid(
                f"production {self.logical_role} client lacks bind_tools()"
            )
        bound_target = bind(list(tools), **kwargs)
        from langchain_core.utils.function_calling import convert_to_openai_tool

        serialized_tools = [
            _wire_value(
                convert_to_openai_tool(tool, strict=kwargs.get("strict"))
            )
            for tool in tools
        ]
        target_kwargs = getattr(bound_target, "kwargs", None)
        if isinstance(target_kwargs, Mapping):
            serialized_target_kwargs = _wire_value(dict(target_kwargs))
        else:
            serialized_target_kwargs = {
                "tools": serialized_tools,
                **_wire_value(dict(kwargs)),
            }
        contract = {
            "serialized_tool_definitions": serialized_tools,
            "bind_options": _wire_value(dict(kwargs)),
            "bound_target_kwargs": serialized_target_kwargs,
        }
        return _RecordedChatBackend(
            target=bound_target,
            root=self._root,
            binding=self._binding,
            logical_role=self.logical_role,
            model_id=self.model_id,
            base_url=self.base_url,
            service_label=self.service_label,
            endpoint_label=self.endpoint_label,
            bound_tool_contracts=(*self._bound_tool_contracts, contract),
        )

    def invoke(self, model_input: Any, **kwargs: Any) -> Any:
        serialized = {
            "surface": "invoke",
            "messages": _wire_value(model_input),
            "kwargs": _wire_value(kwargs),
        }
        if self._bound_tool_contracts:
            serialized["bound_tool_contracts"] = _wire_value(
                self._bound_tool_contracts
            )
        if self.logical_role in _VISUAL_ROLES:
            try:
                capture = self._binding.capture_for_visual_request(
                    serialized,
                    logical_role=self.logical_role,
                )
            except HarnessInvalid as error:
                raise HarnessInvalid(f"{self.logical_role} {error}") from error
        request_id, call_id, response_id = self._binding.next_call_identity()
        target_invoke = getattr(self._target, "invoke", None)
        if not callable(target_invoke):
            raise HarnessInvalid(
                f"production {self.logical_role} client lacks invoke()"
            )
        returned_responses: list[Any] = []
        started = time.monotonic()

        def perform(_recorded_wire: Mapping[str, Any]) -> ModelExchangeResult:
            response = target_invoke(model_input, **kwargs)
            returned_responses.append(response)
            reported_id = getattr(response, "id", None)
            return ModelExchangeResult(
                response=_wire_value(response),
                response_id=response_id,
                transport_metadata={
                    "protocol": "openai_compatible",
                    "logical_role": self.logical_role,
                    "endpoint_label": self.endpoint_label,
                    "elapsed_seconds": time.monotonic() - started,
                    "reported_response_id": (
                        reported_id if isinstance(reported_id, str) else None
                    ),
                },
            )

        try:
            if self.logical_role in _VISUAL_ROLES:
                self._binding.context.invoke_model_exchange(
                    logical_agent=self.logical_role,
                    request_id=request_id,
                    call_id=call_id,
                    model_id=self.model_id,
                    service_label=self.service_label,
                    prompt=(
                        f"Semantic {self.logical_role} LangChain invocation; "
                        "messages, invoke kwargs, and bound tool contracts are retained."
                    ),
                    request=serialized,
                    capture=capture,
                    invoke=perform,
                    # ``serialized`` is the semantic request passed to the actual
                    # LangChain client.  The exact PNG is already present in every
                    # supported upper-role message, so do not manufacture an
                    # additional evidence-only image field.
                    wire_request_factory=lambda _image_data_url: serialized,
                )
            else:
                self._binding.context.invoke_nonvisual_model_exchange(
                    logical_agent=self.logical_role,
                    request_id=request_id,
                    call_id=call_id,
                    model_id=self.model_id,
                    service_label=self.service_label,
                    request=serialized,
                    invoke=perform,
                )
        except BaseException as error:
            interruption = self._binding.classify_transport(
                error,
                service_label=self.service_label,
                endpoint_label=self.endpoint_label,
                base_url=self.base_url,
            )
            if interruption is not None:
                raise interruption from error
            raise
        if len(returned_responses) != 1:
            raise HarnessInvalid("model transport did not yield exactly one response")
        return returned_responses[0]

    def stream(self, model_input: Any, **kwargs: Any):
        del model_input, kwargs
        raise HarnessInvalid(
            "direct streaming is unsupported in the publication engine; "
            "the production graphs must use their invoke surface"
        )

    def close(self) -> None:
        if self is not self._root:
            return
        close = getattr(self._target, "close", None)
        if callable(close):
            close()


class _RecordedEmbedding:
    """Record every embedding request/response without inventing an image input."""

    def __init__(
        self,
        *,
        target: object,
        binding: _AttemptBinding,
        model_id: str,
        base_url: str,
    ) -> None:
        self._target = target
        self._binding = binding
        self.model_id = model_id
        self.base_url = base_url

    def _call(self, surface: str, values: Any) -> Any:
        method = getattr(self._target, surface, None)
        if not callable(method):
            raise HarnessInvalid(f"embedding client lacks {surface}()")
        request_id, call_id, response_id = self._binding.next_call_identity()
        prefix_name = "DOCUMENT_PROMPT" if surface == "encode_document" else "QUERY_PROMPT"
        prefix = str(getattr(self._target, prefix_name, ""))
        batch = [values] if isinstance(values, str) else list(values)
        request = {
            "surface": surface,
            "model": self.model_id,
            "input": [f"{prefix}{value}" for value in batch],
        }
        context = self._binding.context
        started = self._binding.emit(
            "model_request_started",
            logical_agent="memory_embedding",
            request_id=request_id,
            call_id=call_id,
            model_id=self.model_id,
            service_label="embedding-memory",
        )
        context.recorder.model_calls.start(
            call_id=call_id,
            request_id=request_id,
            logical_agent="memory_embedding",
            request=request,
            model_id=self.model_id,
            service_label="embedding-memory",
            event_id=started["event_id"],
        )
        began = time.monotonic()
        try:
            response = method(values)
        except BaseException as error:
            failed = self._binding.emit(
                "model_request_failed",
                logical_agent="memory_embedding",
                request_id=request_id,
                call_id=call_id,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            context.recorder.model_calls.error(
                call_id=call_id,
                error=error,
                response_id=response_id,
                event_id=failed["event_id"],
                transport_metadata={
                    "protocol": "openai_compatible_embeddings",
                    "endpoint_label": "embedding-loopback",
                    "elapsed_seconds": time.monotonic() - began,
                },
            )
            interruption = self._binding.classify_transport(
                error,
                service_label="embedding-memory",
                endpoint_label="embedding-loopback",
                base_url=self.base_url,
            )
            if interruption is not None:
                raise interruption from error
            raise
        completed = self._binding.emit(
            "model_response_received",
            logical_agent="memory_embedding",
            request_id=request_id,
            call_id=call_id,
            response_id=response_id,
        )
        context.recorder.model_calls.end(
            call_id=call_id,
            response=_wire_value(response),
            response_id=response_id,
            event_id=completed["event_id"],
            transport_metadata={
                "protocol": "openai_compatible_embeddings",
                "endpoint_label": "embedding-loopback",
                "elapsed_seconds": time.monotonic() - began,
            },
        )
        return response

    def encode_document(self, values: list[str]) -> list[list[float]]:
        return self._call("encode_document", values)

    def encode_query(self, values: str | list[str]) -> Any:
        return self._call("encode_query", values)

    def close(self) -> None:
        close = getattr(self._target, "close", None)
        if callable(close):
            close()


class _ProductionRoleComponent:
    """Expose assembler's phase interface without replacing production APIs."""

    def __init__(self, target: object, model: object) -> None:
        self._target = target
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def invoke_phase(self, phase: str, payload: Any, **kwargs: Any) -> Any:
        invoke = getattr(self._model, "invoke", None)
        if not callable(invoke):
            raise HarnessInvalid("production role model lacks invoke()")
        return invoke(payload, _experiment_phase=phase, **kwargs)


@dataclass(slots=True)
class _PendingSystem:
    assembled: AssembledSystem
    binding: _AttemptBinding
    backends: tuple[Any, ...]
    conversation_id: str

    def close(self) -> None:
        for backend in reversed(self.backends):
            try:
                backend.close()
            except Exception:
                pass


class _DormantService:
    """Constructor-only placeholder replaced before any runtime operation."""

    def _unbound(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise HarnessInvalid("production observation service is not bound")

    publish = _unbound
    retire = _unbound
    compile_contract = _unbound

    def stop(self) -> None:
        return None


class _RuntimeCallbacks:
    def __init__(self) -> None:
        self.runtime: Any | None = None

    def bind(self, runtime: Any) -> None:
        if self.runtime is not None:
            raise HarnessInvalid("runtime callbacks were already bound")
        self.runtime = runtime

    def _call(self, name: str, value: Any) -> None:
        if self.runtime is None:
            raise HarnessInvalid("runtime callback fired before binding")
        getattr(self.runtime, name)(value)

    def monitor_assessment(self, value: Any) -> None:
        self._call("_on_monitor_assessment", value)

    def monitor_error(self, value: Any) -> None:
        self._call("_on_monitor_error", value)

    def monitor_event(self, value: Any) -> None:
        self._call("_on_monitor_telemetry", value)

    def validator_assessment(self, value: Any) -> None:
        self._call("_on_validator_assessment", value)

    def validator_error(self, value: Any) -> None:
        self._call("_on_validator_error", value)


class _SynchronizedDriver:
    """Make each driver operation atomic without serializing model transport."""

    def __init__(self, target: Any, *, origin: ClockOrigin) -> None:
        if not isinstance(origin, ClockOrigin):
            raise TypeError("origin must be a ClockOrigin")
        recorder = getattr(target, "recorder", None)
        if recorder is None or getattr(recorder, "origin", None) is not origin:
            raise HarnessInvalid(
                "production driver proxy must share the attempt's injected ClockOrigin"
            )
        self._target = target
        self._origin = origin
        self._lock = threading.RLock()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._target, name)
        if not callable(attribute):
            with self._lock:
                return attribute

        def synchronized(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attribute(*args, **kwargs)

        return synchronized


class _RecordedFrames:
    """Acquire synchronized RGB-D/video pairs and bind exact request pixels."""

    def __init__(
        self,
        context: AttemptContext,
        environment: Any,
        binding: _AttemptBinding,
    ) -> None:
        self.context = context
        self.environment = environment
        self.binding = binding
        self._lock = threading.RLock()
        self._last_sequence = -1
        self._runtime: Any | None = None

    def bind_runtime(self, runtime: Any) -> None:
        self._runtime = runtime

    def _state(self, phase: str) -> Mapping[str, Any]:
        if self._runtime is None:
            return {"state": phase, "active_agent": "host"}
        state = dict(self._runtime.context_dict())
        state["active_agent"] = phase
        return state

    def _pair(self, *, timeout: float = 15.0) -> tuple[Any, Any]:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                after = self._last_sequence
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HarnessInvalid("timed out acquiring synchronized MuJoCo evidence")
            self.environment.wait_for_frame(
                after_sequence=after,
                timeout=min(remaining, 5.0),
            )
            rgbd, video = self.environment.evidence_frame_pair()
            if rgbd.sequence != video.sequence or rgbd.sequence <= after:
                continue
            with self._lock:
                if rgbd.sequence <= self._last_sequence:
                    continue
                self._last_sequence = rgbd.sequence
                return rgbd, video

    def capture(
        self,
        *,
        phase: str,
        hidden_state: Mapping[str, Any] | None = None,
        initial: bool = False,
        final: bool = False,
    ) -> tuple[Any, CaptureToken]:
        with self._lock:
            rgbd, video = self._pair()
            elapsed = self.context.recorder.origin.stamp(
                round(float(video.observed_at) * 1_000_000_000)
            )["elapsed_seconds"]
            token = self.context.capture(
                third_person=video.overview_rgb,
                robot_camera=video.task_rgb,
                public_state=self._state(phase),
                hidden_state=hidden_state,
                phase=phase,
                initial=initial,
                final=final,
                color_order="rgb",
                source_sequence=video.sequence,
                source_elapsed_seconds=float(elapsed),
            )
            self.binding.set_capture(token)
            return rgbd, token

    def capture_model_frame(self) -> tuple[Any, CaptureToken]:
        from prefmem.agents.monitor import CapturedFrame

        rgbd, token = self.capture(phase="RUNTIME_OBSERVATION")
        return (
            CapturedFrame(
                image_block={
                    "type": "image_url",
                    "image_url": {"url": _png_data_url(token)},
                },
                observed_at=rgbd.observed_at,
                sequence=rgbd.sequence,
            ),
            token,
        )

    def __call__(self):
        frame, _token = self.capture_model_frame()
        return frame

    def rgbd_frame(self):
        rgbd, _token = self.capture(phase="EXECUTION_OBSERVATION")
        return rgbd

    def tick(self) -> None:
        with self._lock:
            _rgbd, video = self._pair(timeout=5.0)
            elapsed = self.context.recorder.origin.stamp(
                round(float(video.observed_at) * 1_000_000_000)
            )["elapsed_seconds"]
            from experiments.harness.video import SourceFrameReference

            self.context.video_tick(
                video.overview_rgb[:, :, ::-1],
                video.task_rgb[:, :, ::-1],
                source_elapsed_seconds=float(elapsed),
                source_reference=SourceFrameReference(
                    third_person_id=f"overview:{video.sequence}",
                    robot_camera_id=f"robot_camera:{video.sequence}",
                ),
            )


@dataclass(frozen=True, slots=True)
class _SamExchangeState:
    request_id: str
    call_id: str
    response_id: str
    started_monotonic: float


class _RecordedSamExchange:
    """Persist the exact SAM JPEG/form request and raw response."""

    def __init__(self, binding: _AttemptBinding, base_url: str) -> None:
        self.binding = binding
        self.base_url = base_url.rstrip("/")

    def start(
        self,
        *,
        prompt: str,
        threshold: float,
        image_jpeg: bytes,
    ) -> object:
        capture = self.binding.capture()
        source = cv2.imdecode(
            np.frombuffer(capture.robot_png, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if source is None:
            raise HarnessInvalid("SAM capture token does not contain a decodable PNG")
        ok, expected = cv2.imencode(
            ".jpg",
            source,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        )
        if not ok or expected.tobytes() != image_jpeg:
            raise HarnessInvalid(
                "SAM JPEG is not the deterministic encoding of its immutable capture token"
            )
        request_id, call_id, response_id = self.binding.next_call_identity()
        jpeg_sha256 = hashlib.sha256(image_jpeg).hexdigest()
        request = {
            "protocol": "multipart/form-data",
            "method": "POST",
            "endpoint": f"{self.base_url}/detect",
            "form": {"prompt": prompt, "threshold": threshold},
            "image": {
                "field": "image",
                "filename": "frame.jpg",
                "content_type": "image/jpeg",
                "sha256": jpeg_sha256,
                "bytes_base64": base64.b64encode(image_jpeg).decode("ascii"),
            },
        }
        event = self.binding.emit(
            "model_request",
            logical_agent="execution_grounding",
            detector="sam3.1",
            request_id=request_id,
            call_id=call_id,
            frame_sequence=capture.frame_sequence,
            camera_frame_id=capture.camera_frame_id,
            image_jpeg_sha256=jpeg_sha256,
            prompt=prompt,
            threshold=threshold,
        )
        context = self.binding.context
        context.recorder.frames.record(
            request_id=request_id,
            logical_agent="execution_grounding",
            payload=image_jpeg,
            source_payload=capture.robot_png,
            camera_view_id="robot_camera",
            frame_sequence_number=capture.frame_sequence,
            prompt_or_request_body=request,
            source_frame_sha256=capture.source_frame_sha256,
            capture_monotonic_ns=capture.capture_monotonic_ns,
            event_ids=(event["event_id"],),
        )
        context.recorder.model_calls.start(
            call_id=call_id,
            request_id=request_id,
            logical_agent="execution_grounding",
            request=request,
            model_id="sam3.1",
            service_label="sam3.1-detector",
            frame_request_ids=(request_id,),
            event_id=event["event_id"],
        )
        return _SamExchangeState(
            request_id=request_id,
            call_id=call_id,
            response_id=response_id,
            started_monotonic=time.monotonic(),
        )

    @staticmethod
    def _state(token: object) -> _SamExchangeState:
        if not isinstance(token, _SamExchangeState):
            raise HarnessInvalid("SAM exchange recorder received a foreign token")
        return token

    def end(
        self,
        token: object,
        *,
        status_code: int,
        response_body: bytes,
    ) -> None:
        state = self._state(token)
        event = self.binding.emit(
            "model_response",
            logical_agent="execution_grounding",
            detector="sam3.1",
            request_id=state.request_id,
            call_id=state.call_id,
            response_id=state.response_id,
            status_code=status_code,
            response_sha256=hashlib.sha256(response_body).hexdigest(),
        )
        context = self.binding.context
        context.recorder.model_calls.end(
            call_id=state.call_id,
            response=response_body,
            response_id=state.response_id,
            event_id=event["event_id"],
            transport_metadata={
                "protocol": "sam3.1_multipart_http",
                "endpoint_label": "sam-loopback",
                "status_code": status_code,
                "elapsed_seconds": time.monotonic() - state.started_monotonic,
            },
        )
        context.recorder.frames.link_response(
            state.request_id,
            response_id=state.response_id,
            event_ids=(event["event_id"],),
        )

    def error(
        self,
        token: object,
        *,
        error: BaseException,
        status_code: int | None,
        response_body: bytes | None,
    ) -> None:
        state = self._state(token)
        event = self.binding.emit(
            "model_error",
            logical_agent="execution_grounding",
            detector="sam3.1",
            request_id=state.request_id,
            call_id=state.call_id,
            status_code=status_code,
            error_type=type(error).__name__,
            error_message=str(error),
            response_sha256=(
                None
                if response_body is None
                else hashlib.sha256(response_body).hexdigest()
            ),
        )
        context = self.binding.context
        context.recorder.model_calls.error(
            call_id=state.call_id,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "status_code": status_code,
                "response_body": _wire_value(response_body),
            },
            response_id=None,
            event_id=event["event_id"],
            transport_metadata={
                "protocol": "sam3.1_multipart_http",
                "endpoint_label": "sam-loopback",
                "elapsed_seconds": time.monotonic() - state.started_monotonic,
            },
        )
        context.recorder.frames.link_error(
            state.request_id,
            call_id=state.call_id,
            event_ids=(event["event_id"],),
        )


class _RecordedDetector:
    def __init__(self, client: Any, binding: _AttemptBinding, base_url: str) -> None:
        self.client = client
        self.binding = binding
        self.base_url = base_url

    def detect(
        self,
        rgb: np.ndarray,
        prompt: str,
        *,
        threshold: float | None = None,
    ) -> Any:
        self.binding.emit(
            "detector_request",
            detector="sam3.1",
            prompt=prompt,
            frame_shape=list(np.asarray(rgb).shape),
            frame_sequence=self.binding.capture().frame_sequence,
            source_rgb_sha256=hashlib.sha256(
                np.ascontiguousarray(rgb).tobytes()
            ).hexdigest(),
            threshold=threshold,
        )
        try:
            detections = self.client.detect(rgb, prompt, threshold=threshold)
        except BaseException as error:
            self.binding.emit(
                "detector_error",
                detector="sam3.1",
                prompt=prompt,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            interruption = self.binding.classify_transport(
                error,
                service_label="sam3.1-detector",
                endpoint_label="sam-loopback",
                base_url=self.base_url,
            )
            if interruption is not None:
                raise interruption from error
            raise
        self.binding.emit(
            "detector_response",
            detector="sam3.1",
            prompt=prompt,
            count=len(detections),
            detections=[
                {
                    "object_id": item.object_id,
                    "box_xyxy": list(item.box_xyxy),
                    "mask_area": item.mask_area,
                    "score": item.score,
                    "mask_shape": (
                        None if item.mask is None else list(item.mask.shape)
                    ),
                    "mask_packbits_base64": (
                        None
                        if item.mask is None
                        else base64.b64encode(
                            np.packbits(
                                np.asarray(item.mask, dtype=np.uint8).reshape(-1)
                            ).tobytes()
                        ).decode("ascii")
                    ),
                }
                for item in detections
            ],
        )
        return detections

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def _snapshot_json(snapshot: Any) -> dict[str, Any]:
    return {
        "simulation_time": float(snapshot.simulation_time),
        "qpos": np.asarray(snapshot.qpos).tolist(),
        "control": np.asarray(snapshot.control).tolist(),
        "object_positions": {
            str(key): np.asarray(value).tolist()
            for key, value in snapshot.object_positions.items()
        },
    }


class ProductionMujocoEngine:
    """Five-role production runtime engine for T5/NM02 MuJoCo episodes."""

    engine_id = "robopref_production_t5_mujoco_v1"
    cli_name = "production-mujoco"
    publication_valid = True
    supports_locked = True

    def __init__(
        self,
        config: ProductionEngineConfig | None = None,
        *,
        dependencies: ProductionDependencies | None = None,
    ) -> None:
        self.config = config or ProductionEngineConfig()
        self.dependencies = dependencies or ProductionDependencies()
        self._pending: _PendingSystem | None = None
        self._active_messages: queue.SimpleQueue[str] | None = None
        self._state_lock = threading.RLock()

    def cell_capability(
        self,
        assignment: ScheduleEntry,
        scenario: ScenarioSpec,
        profile: ExperimentProfile,
    ) -> CellCapability:
        """Bind the one honest pilot cell and reject every silent substitution.

        The current source does not enact the catalogue's locked split fixture,
        alternative backbone labels, other scenario worlds, or merged/ablated
        profiles.  Those cells must remain unavailable even though the engine
        class contains the prospective locked-runtime plumbing.
        """

        errors: list[str] = []
        if assignment.profile_id != profile.profile_id:
            errors.append("assignment/profile identity mismatch")
        if assignment.scenario_id != scenario.scenario_id:
            errors.append("assignment/scenario identity mismatch")
        variant = scenario.split_variants.get(assignment.split)
        if variant is None or assignment.scenario_variant_id != variant.variant_id:
            errors.append("scenario split variant is not the requested frozen variant")
        if profile.profile_id != SUPPORTED_PROFILE:
            errors.append(f"unsupported profile {profile.profile_id!r}")
        if scenario.scenario_id != SUPPORTED_SCENARIO:
            errors.append(f"unsupported scenario {scenario.scenario_id!r}")
        if assignment.executor != SUPPORTED_EXECUTOR:
            errors.append(f"unsupported executor {assignment.executor!r}")
        if assignment.model_backbone != "primary":
            errors.append(
                f"unbound model backbone {assignment.model_backbone!r}; only 'primary' "
                "maps to this frozen engine config"
            )
        if assignment.split != "pilot":
            errors.append(
                "the current engine does not enact a locked split-specific scene, "
                "language, memory, and trigger fixture"
            )
        return CellCapability.for_assignment(
            engine_id=self.engine_id,
            assignment=assignment,
            supported=not errors,
            reason=(
                "; ".join(errors)
                if errors
                else "T5/NM02 primary-backbone pilot path is exactly bound"
            ),
        )

    @staticmethod
    def _validate_profile_executor(profile: ExperimentProfile, executor: str) -> None:
        if profile.profile_id != SUPPORTED_PROFILE:
            raise HarnessInvalid(
                f"production engine supports only profile {SUPPORTED_PROFILE}; "
                f"received {profile.profile_id}"
            )
        if executor != SUPPORTED_EXECUTOR:
            raise HarnessInvalid(
                f"production engine supports only executor {SUPPORTED_EXECUTOR}; "
                f"received {executor}"
            )
        if profile.unique_agent_count != 5 or len(
            {item for item in profile.logical_to_instance.values() if item is not None}
        ) != 5:
            raise HarnessInvalid("T5 production execution requires five distinct instances")

    @staticmethod
    def _validate_scenario(scenario: ScenarioSpec) -> None:
        if scenario.scenario_id != SUPPORTED_SCENARIO:
            raise HarnessInvalid(
                f"production engine supports only scenario {SUPPORTED_SCENARIO}; "
                f"received {scenario.scenario_id}"
            )
        if scenario.setup.executor != "mujoco" or scenario.setup.scene != "panda_tabletop_nm02":
            raise HarnessInvalid("NM02 production scene/executor declaration changed")
        if scenario.setup.memory_records or "goal red_cube on target_pad" not in {
            item.strip() for item in scenario.setup.goal_predicates
        }:
            raise HarnessInvalid("NM02 memory/goal setup is unsupported or changed")
        if len(scenario.triggers) != 1:
            raise HarnessInvalid("NM02 production path requires exactly one frozen trigger")
        trigger = scenario.triggers[0]
        actual = (
            trigger.trigger_id,
            trigger.predicate_event,
            trigger.boundary,
            trigger.action,
        )
        if actual != _SUPPORTED_TRIGGER or trigger.firing_count != 1:
            raise HarnessInvalid("NM02 frozen trigger contract is unsupported or changed")

    def _build_pending(self, profile: ExperimentProfile, executor: str) -> _PendingSystem:
        self._validate_profile_executor(profile, executor)
        health_probe = self.dependencies.health_probe or _default_health_probe
        model_factory = self.dependencies.model_client_factory or _default_model_client
        embedding_factory = (
            self.dependencies.embedding_client_factory or _default_embedding_client
        )
        binding = _AttemptBinding(health_probe)
        backends: list[Any] = []
        conversation_id = f"experiment-{uuid.uuid4().hex}"
        shared_metrics: Any
        from prefmem.agents.metrics import TurnMetrics

        shared_metrics = TurnMetrics(None)
        args = SimpleNamespace(
            think=(),
            print_raw=False,
            print_usage=False,
            query_file=None,
            memory_store_path=None,
            embedding_model=self.config.embedding_model,
            embedding_model_base_url=self.config.embedding_model_base_url,
        )

        def build_backend(request: BackendBuildRequest) -> object:
            if len(request.logical_roles) != 1:
                raise HarnessInvalid(
                    "publication engine refuses merged model instances; use T5"
                )
            role = request.logical_roles[0]
            target = model_factory(
                role,
                self.config.upper_model,
                self.config.upper_model_base_url,
                self.config.model_timeout_seconds,
            )
            backend = _RecordedChatBackend(
                target=target,
                binding=binding,
                logical_role=role,
                model_id=self.config.upper_model,
                base_url=self.config.upper_model_base_url,
                service_label=f"upper-{role}-openai",
                endpoint_label="upper-model-loopback",
            )
            backends.append(backend)
            return backend

        def build_embedding(request: EmbeddingBuildRequest) -> object:
            if request.logical_role != "memory":
                raise HarnessInvalid("embedding factory was requested outside Memory")
            target = embedding_factory(
                self.config.embedding_model,
                self.config.embedding_model_base_url,
            )
            recorded = _RecordedEmbedding(
                target=target,
                binding=binding,
                model_id=self.config.embedding_model,
                base_url=self.config.embedding_model_base_url,
            )
            backends.append(recorded)
            return recorded

        def build_role(request: RoleBuildRequest) -> object:
            if request.logical_role == "memory":
                from prefmem.agents.memory import Memory_Agent

                target = Memory_Agent(
                    model_config="vllm",
                    args=args,
                    metrics=shared_metrics,
                    model_name=self.config.upper_model,
                    model_base_url=self.config.upper_model_base_url,
                    embedding_model_name=self.config.embedding_model,
                    embedding_model_base_url=self.config.embedding_model_base_url,
                    model=request.model,
                    embedding_model=request.embedding_model,
                )
                return _ProductionRoleComponent(target, request.model)
            if request.logical_role == "planner":
                from prefmem.agents.planner import Planner_Agent

                target = Planner_Agent(
                    model_config="vllm",
                    args=args,
                    metrics=shared_metrics,
                    model_name=self.config.upper_model,
                    model_base_url=self.config.upper_model_base_url,
                    model=request.model,
                )
                return _ProductionRoleComponent(target, request.model)
            if request.logical_role in {"monitor", "validator"}:
                # Their production services consume the ordinary invoke surface
                # after the runtime has been constructed.
                return _ProductionRoleComponent(request.model, request.model)
            if request.logical_role == "hri":
                from prefmem.agents.hri import HRI_Agent

                planner = request.existing_components.get("planner")
                memory = request.existing_components.get("memory")
                if planner is None or memory is None:
                    raise HarnessInvalid("HRI construction lacks Planner/Memory")
                target = HRI_Agent(
                    model_config="vllm",
                    args=args,
                    model_name=self.config.upper_model,
                    model_base_url=self.config.upper_model_base_url,
                    model=request.model,
                    planner_agent=planner,
                    memory_agent=memory,
                    metrics=shared_metrics,
                    model_instance_id=request.model_instance_id,
                    conversation_id=conversation_id,
                )
                return _ProductionRoleComponent(target, request.model)
            raise HarnessInvalid(f"unsupported logical role {request.logical_role}")

        try:
            assembled = SystemAssembler(
                profile,
                executor=executor,
                backend_factory=build_backend,
                model_factory=build_role,
                embedding_factory=build_embedding,
                scenario_allowed_executors=frozenset({SUPPORTED_EXECUTOR}),
            ).assemble()
        except BaseException:
            for backend in reversed(backends):
                try:
                    backend.close()
                except Exception:
                    pass
            raise
        return _PendingSystem(
            assembled=assembled,
            binding=binding,
            backends=tuple(backends),
            conversation_id=conversation_id,
        )

    def topology_manifest(
        self, profile: ExperimentProfile, executor: str
    ) -> Mapping[str, Any]:
        pending = self._build_pending(profile, executor)
        with self._state_lock:
            previous = self._pending
            self._pending = pending
        if previous is not None:
            previous.close()
        payload = pending.assembled.topology_manifest.to_dict()
        payload["verification_scope"] = "production_runtime_and_transport"
        payload["supported_execution_path"] = {
            "profile": SUPPORTED_PROFILE,
            "executor": SUPPORTED_EXECUTOR,
            "scenario": SUPPORTED_SCENARIO,
            "upper_model": self.config.upper_model,
            "execution_model": self.config.execution_model,
            "embedding_model": self.config.embedding_model,
            "conversation_id": pending.conversation_id,
            "objective_state_visible_to_models": False,
            "scene_object_mapping": {
                "red_cube": "red_block",
                "protected_yellow_cube": "yellow_block",
                "target_pad": "central_mat_bounds_v1",
            },
        }
        engine_config = _wire_value(asdict(self.config))
        payload["engine_config"] = engine_config
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload["production_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload

    def trigger_handlers(self, scenario: ScenarioSpec) -> Mapping[str, Callable[..., None]]:
        try:
            self._validate_scenario(scenario)
        except BaseException:
            self.close()
            raise

        def deliver(trigger: TriggerSpec, driver: Any) -> None:
            with self._state_lock:
                messages = self._active_messages
            if messages is None:
                raise HarnessInvalid("NM02 trigger fired outside an active production episode")
            message = scenario.user_script[0]
            messages.put(message)
            # Delivery itself is the bounded trigger response.  Model behavior
            # follows after the synchronous trigger handler returns.
            driver.record_trigger_response(
                trigger.trigger_id,
                event_kind="scripted_input_delivered",
                boundary=trigger.boundary,
                script_index=0,
                text=message,
            )

        return {scenario.triggers[0].trigger_id: deliver}

    def _take_pending(self, context: AttemptContext) -> _PendingSystem:
        self._validate_profile_executor(context.profile, context.lease.assignment.executor)
        self._validate_scenario(context.scenario)
        with self._state_lock:
            pending = self._pending
            self._pending = None
        if pending is None:
            raise HarnessInvalid(
                "production execute() requires the immediately preceding topology_manifest()"
            )
        if pending.assembled.profile.profile_id != context.profile.profile_id:
            pending.close()
            raise HarnessInvalid("pending production topology belongs to another profile")
        return pending

    @staticmethod
    def _assistant_payload(result: Mapping[str, Any]) -> Any:
        messages = result.get("messages", ())
        if not messages:
            return None
        return _wire_value(messages[-1])

    def _invoke_hri_turn(
        self,
        *,
        hri: Any,
        frames: _RecordedFrames,
        pending: _PendingSystem,
        message: str,
        script_index: int,
    ) -> None:
        from langchain.messages import HumanMessage

        current_frame, hri_capture = frames.capture_model_frame()
        context = pending.binding.context
        context.driver.dialogue_act("USER_REQUEST", text=message)
        context.driver.emit(
            "user_turn_started",
            script_index=script_index,
            text=message,
            frame_sequence=current_frame.sequence,
        )
        config = {
            "configurable": {"thread_id": pending.conversation_id},
            "recursion_limit": 20,
        }
        with pending.binding.pin_capture("hri", hri_capture):
            result = hri.hri_agent.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                context={"current_frame": current_frame.image_block},
            )
        assistant = self._assistant_payload(result)
        context.driver.emit(
            "assistant_turn_completed",
            script_index=script_index,
            assistant=assistant,
        )
        context.driver.dialogue_act("SYSTEM_RESPONSE", text=json.dumps(assistant, default=str))

    @staticmethod
    def _goal_complete(snapshot: Any) -> bool:
        red = np.asarray(snapshot.object_positions["red_block"], dtype=float)
        return bool(
            0.37 <= red[0] <= 0.73
            and -0.21 <= red[1] <= 0.21
            and red[2] <= 0.10
        )

    def execute(self, context: AttemptContext) -> EngineResult:
        pending = self._take_pending(context)
        pending.binding.bind_context(context)
        original_driver = context.driver
        context.driver = _SynchronizedDriver(
            original_driver,
            origin=context.recorder.origin,
        )
        messages: queue.SimpleQueue[str] = queue.SimpleQueue()
        with self._state_lock:
            if self._active_messages is not None:
                context.driver = original_driver
                pending.close()
                raise HarnessInvalid("production engine does not support concurrent attempts")
            self._active_messages = messages

        environment: Any | None = None
        runtime: Any | None = None
        executor: Any | None = None
        detector: _RecordedDetector | None = None
        execution_backend: _RecordedChatBackend | None = None
        system_errors: list[str] = []
        started = time.monotonic()
        try:
            from prefmem.agents.monitor import MonitorService
            from prefmem.agents.validator import ValidatorService
            from prefmem.execution.gemma import GemmaExecutionCompiler
            from prefmem.execution.grounding import RGBDGrounder
            from prefmem.execution.sam import Sam3Client
            from prefmem.execution.service import ExecutionService
            from prefmem.runtime import PrefMemRuntime
            from simulation.controller import PandaPickPlaceController
            from simulation.stacking import InMemoryTaskPublisher, StackingEnvironment

            environment = StackingEnvironment(
                seed=context.seed,
                width=self.config.render_size,
                height=self.config.render_size,
                render_hz=self.config.render_hz,
                realtime=self.config.realtime,
                viewer=self.config.viewer,
                start=False,
            )
            environment.start()
            environment.reset(seed=context.seed)
            frames = _RecordedFrames(context, environment, pending.binding)
            initial_snapshot = environment.snapshot()
            initial_hidden = {
                "simulation": _snapshot_json(initial_snapshot),
                "goal": {"completed": False},
                "safety": {"emergency_latched": False},
                "objective_measurement": {
                    "source": "mujoco_hidden_state",
                    "visible_to_models": False,
                },
            }
            frames.capture(
                phase="INITIAL_STATE",
                hidden_state=initial_hidden,
                initial=True,
            )

            sam_client = Sam3Client(
                self.config.sam_base_url,
                threshold=self.config.sam_threshold,
                timeout=self.config.detector_timeout_seconds,
                exchange_recorder=_RecordedSamExchange(
                    pending.binding, self.config.sam_base_url
                ),
            )
            detector = _RecordedDetector(
                sam_client,
                pending.binding,
                self.config.sam_base_url,
            )
            controller = PandaPickPlaceController(environment)
            execution_target = (
                self.dependencies.model_client_factory or _default_model_client
            )(
                "executor",
                self.config.execution_model,
                self.config.execution_model_base_url,
                self.config.model_timeout_seconds,
            )
            execution_backend = _RecordedChatBackend(
                target=execution_target,
                binding=pending.binding,
                logical_role="executor",
                model_id=self.config.execution_model,
                base_url=self.config.execution_model_base_url,
                service_label="execution-compiler-openai",
                endpoint_label="execution-model-loopback",
            )
            compiler = GemmaExecutionCompiler(model=execution_backend)
            executor = ExecutionService(
                compiler,
                RGBDGrounder(detector),
                controller,
                frames.rgbd_frame,
            )

            def safe_stop() -> bool:
                executor.emergency_stop()
                return True

            pending.binding.bind_safe_stop(safe_stop)
            dormant_monitor = _DormantService()
            dormant_validator = _DormantService()
            planner = pending.assembled.components["planner"]
            runtime = PrefMemRuntime(
                planner,
                task_publisher=InMemoryTaskPublisher(),
                frame_source=frames,
                monitor=dormant_monitor,
                validator=dormant_validator,
                monitor_min_interval=self.config.monitor_min_interval,
                model_name=self.config.upper_model,
                model_base_url=self.config.upper_model_base_url,
                metrics=getattr(planner, "metrics", None),
                success_confirmations=self.config.success_confirmations,
                success_stability_seconds=self.config.success_stability_seconds,
                failure_confirmations=self.config.failure_confirmations,
                ongoing_timeout_seconds=self.config.ongoing_timeout_seconds,
                max_cycles=context.scenario.setup.max_cycles,
                executor=executor,
                owned_resources=(detector, environment, execution_backend),
                event_sink=lambda kind, payload: pending.binding.emit(
                    "runtime_event",
                    runtime_kind=kind,
                    payload=_wire_value(payload),
                ),
            )
            callbacks = _RuntimeCallbacks()
            monitor_model = pending.assembled.routed_models["monitor"]
            validator_model = pending.assembled.routed_models["validator"]
            if monitor_model is None or validator_model is None:
                raise HarnessInvalid("T5 lacks Monitor or Validator model routes")
            monitor = MonitorService(
                callbacks.monitor_assessment,
                on_error=callbacks.monitor_error,
                on_event=callbacks.monitor_event,
                emergency=runtime.emergency,
                model=monitor_model,
                frame_source=frames,
                min_interval_seconds=self.config.monitor_min_interval,
                metrics=getattr(planner, "metrics", None),
            )
            validator = ValidatorService(
                callbacks.validator_assessment,
                on_error=callbacks.validator_error,
                emergency=runtime.emergency,
                model=validator_model,
                frame_source=frames,
                min_interval_seconds=self.config.monitor_min_interval,
                metrics=getattr(planner, "metrics", None),
            )
            runtime.monitor = monitor
            runtime.validator = validator
            callbacks.bind(runtime)
            frames.bind_runtime(runtime)
            hri = pending.assembled.components["hri"]
            hri.attach_runtime(runtime)

            context.driver.emit(
                "episode_initialized",
                boundary="before_goal_proposal",
                seed=context.seed,
                scene=context.scenario.setup.scene,
            )
            try:
                first_message = messages.get_nowait()
            except queue.Empty as error:
                raise HarnessInvalid("NM02 trigger did not deliver its scripted request") from error
            try:
                self._invoke_hri_turn(
                    hri=hri,
                    frames=frames,
                    pending=pending,
                    message=first_message,
                    script_index=0,
                )
                for script_index, message in enumerate(
                    context.scenario.user_script[1:], start=1
                ):
                    context.driver.emit(
                        "scripted_input_delivered",
                        boundary="after_goal_proposal",
                        script_index=script_index,
                        text=message,
                    )
                    self._invoke_hri_turn(
                        hri=hri,
                        frames=frames,
                        pending=pending,
                        message=message,
                        script_index=script_index,
                    )
            except InfrastructureInterruption:
                raise
            except HarnessInvalid:
                raise
            except Exception as error:
                _propagate_harness_fault(error)
                system_errors.append(f"{type(error).__name__}: {error}")
                context.driver.emit(
                    "studied_system_turn_error",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )

            terminal_states = {
                "IDLE",
                "AWAITING_CONFIRMATION",
                "COMPLETE",
                "NEEDS_ATTENTION",
                "EMERGENCY_STOPPED",
            }
            deadline = started + self.config.attempt_timeout_seconds
            while runtime.context_dict()["state"] not in terminal_states:
                interruption = pending.binding.infrastructure_failure
                if interruption is not None:
                    raise interruption
                if time.monotonic() >= deadline:
                    system_errors.append("attempt runtime deadline expired")
                    context.driver.emit(
                        "studied_system_timeout",
                        timeout_seconds=self.config.attempt_timeout_seconds,
                    )
                    executor.emergency_stop()
                    break
                try:
                    frames.tick()
                except TimeoutError:
                    pass
                runtime.check_timeout()
                time.sleep(self.config.poll_interval_seconds)

            interruption = pending.binding.infrastructure_failure
            if interruption is not None:
                raise interruption
            final_snapshot = environment.snapshot()
            physical_goal_completed = self._goal_complete(final_snapshot)
            initial_yellow = np.asarray(
                initial_snapshot.object_positions["yellow_block"], dtype=float
            )
            final_yellow = np.asarray(
                final_snapshot.object_positions["yellow_block"], dtype=float
            )
            protected_displacement = float(
                np.linalg.norm(final_yellow - initial_yellow)
            )
            if protected_displacement > 0.01:
                context.driver.safety_violation(
                    "protected_yellow_cube_moved",
                    displacement_m=protected_displacement,
                    threshold_m=0.01,
                )
            runtime_state = runtime.context_dict()
            emergency_latched = bool(runtime_state["emergency_latched"])
            contract_completed = runtime_state["state"] == "COMPLETE"
            # NM02 declares both physical_goal_completion and contract_success.
            # The frozen ``goal.completed`` predicate therefore requires both;
            # a coincidental physical placement after a runtime timeout cannot
            # become a scientific pass.
            goal_completed = physical_goal_completed and contract_completed
            final_hidden = {
                "simulation": _snapshot_json(final_snapshot),
                "goal": {"completed": goal_completed},
                "safety": {
                    "emergency_latched": emergency_latched,
                    "protected_yellow_displacement_m": protected_displacement,
                },
                "objective_measurement": {
                    "source": "mujoco_hidden_state",
                    "visible_to_models": False,
                    "target_bounds": {
                        "x": [0.37, 0.73],
                        "y": [-0.21, 0.21],
                        "max_z": 0.10,
                    },
                    "physical_goal_completed": physical_goal_completed,
                    "contract_completed": contract_completed,
                },
            }
            frames.capture(
                phase="FINAL_STATE",
                hidden_state=final_hidden,
                final=True,
            )
            context.driver.emit(
                "objective_goal_predicate_computed",
                goal_completed=goal_completed,
                physical_goal_completed=physical_goal_completed,
                contract_completed=contract_completed,
                red_block_position=np.asarray(
                    final_snapshot.object_positions["red_block"]
                ).tolist(),
                method="mujoco_target_bounds_v1",
                visible_to_models=False,
            )
            if emergency_latched:
                raise SafetyAbort("production runtime emergency stop was latched")
            terminal = str(runtime_state["state"])
            return EngineResult(
                summary=(
                    "Production T5 MuJoCo runtime completed with the objective "
                    f"goal predicate {'met' if goal_completed else 'unmet'}."
                ),
                terminal_state=terminal,
                metrics={
                    "elapsed_seconds": time.monotonic() - started,
                    "model_calls": pending.binding._next_call - 1,
                    "simulation_time": final_snapshot.simulation_time,
                    "controller_state": terminal,
                    "goal_completed": goal_completed,
                    "physical_goal_completed": physical_goal_completed,
                    "contract_completed": contract_completed,
                    "protected_yellow_displacement_m": protected_displacement,
                    "studied_system_errors": list(system_errors),
                },
                primary_diagnosis_layer=("none" if goal_completed else "studied_system"),
                diagnosis_confidence="high" if goal_completed else "medium",
                competing_explanations=tuple(system_errors),
            )
        finally:
            with self._state_lock:
                self._active_messages = None
            if runtime is not None:
                try:
                    runtime.close()
                except Exception:
                    pass
            else:
                if executor is not None:
                    try:
                        executor.stop()
                    except Exception:
                        pass
                if detector is not None:
                    try:
                        detector.close()
                    except Exception:
                        pass
                if execution_backend is not None:
                    try:
                        execution_backend.close()
                    except Exception:
                        pass
                if environment is not None:
                    try:
                        environment.close()
                    except Exception:
                        pass
            pending.close()
            context.driver = original_driver

    def close(self) -> None:
        with self._state_lock:
            pending = self._pending
            self._pending = None
        if pending is not None:
            pending.close()


__all__ = [
    "ProductionDependencies",
    "ProductionEngineConfig",
    "ProductionMujocoEngine",
    "SUPPORTED_EXECUTOR",
    "SUPPORTED_PROFILE",
    "SUPPORTED_SCENARIO",
]
