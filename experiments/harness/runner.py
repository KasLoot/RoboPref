"""Sequential, resumable execution of frozen RoboPref campaign entries.

The runner owns attempt lifecycle and provenance.  Engines own behavior only;
they cannot choose retry status, overwrite an attempt, evaluate an oracle while
the episode is running, or bypass profile/executor safety checks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
import base64
from datetime import datetime, timezone
import hashlib
import inspect
import json
import platform
from pathlib import Path
import re
import time
import traceback
from typing import Any, Protocol

import cv2
import numpy as np

from experiments.harness.campaign import (
    AttemptLease,
    CampaignMode,
    CampaignRegistry,
    RunStatus as RegistryRunStatus,
    ScheduleEntry,
    verify_frozen_campaign,
)
from experiments.harness.driver import (
    ScenarioDriver,
    ScenarioDriverError,
    ScenarioEvaluation,
    TriggerHandler,
    jsonable,
)
from experiments.harness.profiles import (
    ExperimentProfile,
    assert_executor_allowed,
    load_profiles,
)
from experiments.harness.recording import (
    AttemptRecorder,
    ClockOrigin,
    RecordingError,
    RunStatus,
    sha256_bytes,
    sha256_file,
)
from experiments.harness.scenarios import ScenarioSpec, load_scenarios
from experiments.harness.video import (
    OverlayState,
    SourceFrameReference,
    SynchronizedVideoRecorder,
    VideoConfig,
    VideoError,
    source_frame_sha256,
)


class AttemptExecutionError(RuntimeError):
    """Base class for typed attempt termination."""


@dataclass(frozen=True, slots=True)
class InfrastructureFailureEvidence:
    """Redacted external observations supporting an infrastructure label."""

    service_label: str
    endpoint_label: str
    failure_kind: str
    transport_error_type: str
    safe_stop_confirmed: bool
    health_observations: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        for name in (
            "service_label",
            "endpoint_label",
            "failure_kind",
            "transport_error_type",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        safe_identifier = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
        if safe_identifier.fullmatch(self.service_label) is None:
            raise ValueError("service_label must be a safe evidence identifier")
        if self.safe_stop_confirmed is not True:
            raise ValueError("infrastructure evidence requires a confirmed safe stop")
        if not self.health_observations:
            raise ValueError("infrastructure evidence requires a failed external probe")
        normalized: list[Mapping[str, Any]] = []
        for index, observation in enumerate(self.health_observations, start=1):
            if not isinstance(observation, Mapping) or observation.get("ok") is not False:
                raise ValueError(
                    "each infrastructure health observation must explicitly report ok=false"
                )
            item = dict(jsonable(observation))
            if item.get("passed", False) is not False:
                raise ValueError("failed health observation cannot report passed=true")
            observed_utc = item.get("observed_utc")
            if observed_utc is None:
                observed_utc = (
                    datetime.now(timezone.utc)
                    .isoformat(timespec="microseconds")
                    .replace("+00:00", "Z")
                )
            elif not isinstance(observed_utc, str) or not observed_utc.strip():
                raise ValueError("health observation observed_utc must be a timestamp")
            service_id = item.get("service_id", self.service_label)
            if (
                not isinstance(service_id, str)
                or safe_identifier.fullmatch(service_id) is None
                or service_id != self.service_label
            ):
                raise ValueError(
                    "health observation service_id must match the safe service_label"
                )
            check_id = item.get("check_id", f"health-probe-{index:03d}")
            if (
                not isinstance(check_id, str)
                or safe_identifier.fullmatch(check_id) is None
            ):
                raise ValueError("health observation check_id must be a safe identifier")
            item.update(
                {
                    "observed_utc": observed_utc,
                    "passed": False,
                    "service_id": service_id,
                    "check_id": check_id,
                    "endpoint_label": self.endpoint_label,
                }
            )
            normalized.append(item)
        object.__setattr__(self, "health_observations", tuple(normalized))

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


class InfrastructureInterruption(AttemptExecutionError):
    """A contemporaneously evidenced external service/transport interruption."""

    def __init__(self, message: str, *, evidence: InfrastructureFailureEvidence) -> None:
        if not isinstance(evidence, InfrastructureFailureEvidence):
            raise TypeError("InfrastructureInterruption requires typed external evidence")
        super().__init__(message)
        self.evidence = evidence


class HarnessInvalid(AttemptExecutionError):
    """The runner, recorder, oracle, or fixture invalidated the attempt."""


class SafetyAbort(AttemptExecutionError):
    """The host stopped an attempt because continuing was unsafe."""


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Behavioral output, kept separate from the later oracle verdict."""

    summary: str
    terminal_state: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    primary_diagnosis_layer: str = "none"
    diagnosis_confidence: str = "not_applicable"
    competing_explanations: tuple[str, ...] = ()
    deviations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CaptureToken:
    """Immutable identity for exact pixels acquired at one camera tick."""

    frame_sequence: int
    video_frame_index: int
    camera_frame_id: str
    robot_png: bytes = field(repr=False)
    source_frame_sha256: str
    capture_monotonic_ns: int
    capture_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ModelExchangeResult:
    """Transport-facing result returned only after a recorded call has started."""

    response: Any
    response_id: str
    transport_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CellCapability:
    """Exact engine acknowledgement of one frozen schedule assignment.

    Engines must return this typed value before the registry reserves an
    attempt.  Echoing every behavior-relevant assignment identity prevents a
    profile/scenario-only check from silently standing in for split-variant or
    model-backbone support.
    """

    engine_id: str
    schedule_id: str
    profile_id: str
    scenario_id: str
    scenario_variant_id: str
    split: str
    executor: str
    model_backbone: str
    supported: bool
    reason: str

    def __post_init__(self) -> None:
        for name in (
            "engine_id",
            "schedule_id",
            "profile_id",
            "scenario_id",
            "scenario_variant_id",
            "split",
            "executor",
            "model_backbone",
            "reason",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"cell capability {name} must be non-empty")
        if type(self.supported) is not bool:
            raise TypeError("cell capability supported must be a boolean")

    @classmethod
    def for_assignment(
        cls,
        *,
        engine_id: str,
        assignment: ScheduleEntry,
        supported: bool,
        reason: str,
    ) -> "CellCapability":
        if not isinstance(assignment, ScheduleEntry):
            raise TypeError("assignment must be a ScheduleEntry")
        return cls(
            engine_id=engine_id,
            schedule_id=assignment.schedule_id,
            profile_id=assignment.profile_id,
            scenario_id=assignment.scenario_id,
            scenario_variant_id=assignment.scenario_variant_id,
            split=assignment.split,
            executor=assignment.executor,
            model_backbone=assignment.model_backbone,
            supported=supported,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AttemptEngine(Protocol):
    """One behavior implementation injected into the immutable runner shell."""

    engine_id: str
    cli_name: str
    publication_valid: bool
    supports_locked: bool

    def cell_capability(
        self,
        assignment: ScheduleEntry,
        scenario: ScenarioSpec,
        profile: ExperimentProfile,
    ) -> CellCapability: ...

    def topology_manifest(
        self, profile: ExperimentProfile, executor: str
    ) -> Mapping[str, Any]: ...

    def trigger_handlers(
        self, scenario: ScenarioSpec
    ) -> Mapping[str, TriggerHandler]: ...

    def execute(self, context: "AttemptContext") -> EngineResult: ...


def _validated_cell_capability(
    engine: AttemptEngine,
    assignment: ScheduleEntry,
    scenario: ScenarioSpec,
    profile: ExperimentProfile,
) -> CellCapability:
    """Require an exact, positive engine binding for the next frozen cell."""

    validator = getattr(engine, "cell_capability", None)
    if not callable(validator):
        raise HarnessInvalid(
            f"engine {engine.engine_id} lacks mandatory cell_capability()"
        )
    capability = validator(assignment, scenario, profile)
    if not isinstance(capability, CellCapability):
        raise HarnessInvalid(
            f"engine {engine.engine_id} returned an untyped cell capability"
        )
    expected = {
        "engine_id": engine.engine_id,
        "schedule_id": assignment.schedule_id,
        "profile_id": assignment.profile_id,
        "scenario_id": assignment.scenario_id,
        "scenario_variant_id": assignment.scenario_variant_id,
        "split": assignment.split,
        "executor": assignment.executor,
        "model_backbone": assignment.model_backbone,
    }
    actual = capability.to_dict()
    mismatches = {
        name: {"expected": value, "actual": actual.get(name)}
        for name, value in expected.items()
        if actual.get(name) != value
    }
    if mismatches:
        raise HarnessInvalid(
            "engine cell capability does not exactly bind the frozen assignment: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if capability.supported is not True:
        raise HarnessInvalid(
            f"engine {engine.engine_id} does not support frozen cell "
            f"{assignment.schedule_id}: {capability.reason}"
        )
    return capability


def _scenario_split_key(split: str) -> str:
    """Map the locked-test fixture alias to the catalogue's locked variant."""

    return "locked" if split == "locked_test" else split


def _encode_png(rgb_or_bgr: np.ndarray, *, color_order: str) -> bytes:
    image = np.asarray(rgb_or_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise HarnessInvalid("evidence frame must be uint8 HxWx3")
    if color_order == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_order != "bgr":
        raise HarnessInvalid("frame color_order must be 'rgb' or 'bgr'")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise HarnessInvalid("could not encode exact evidence frame")
    return encoded.tobytes()


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _opening_slate(label: str) -> np.ndarray:
    """Return a nonblank BGR slate available before simulator construction."""

    slate = np.full((640, 640, 3), (32, 38, 44), dtype=np.uint8)
    for coordinate in range(0, 640, 40):
        cv2.line(slate, (coordinate, 0), (coordinate, 639), (45, 52, 60), 1)
        cv2.line(slate, (0, coordinate), (639, coordinate), (45, 52, 60), 1)
    cv2.putText(
        slate,
        "ROBOPREF RECORDING ACTIVE",
        (76, 292),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        slate,
        label[:54],
        (76, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (175, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return slate


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json(value: Any) -> str:
    """Canonical comparison form for frozen execution configuration."""

    try:
        return json.dumps(
            jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise HarnessInvalid(
            f"execution configuration is not canonical JSON: {error}"
        ) from error


def _engine_config(engine: AttemptEngine) -> dict[str, Any] | None:
    """Return an engine's public instance configuration without introspection magic."""

    value = getattr(engine, "config", None)
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        rendered = asdict(value)
    elif isinstance(value, Mapping):
        rendered = dict(value)
    else:
        raise HarnessInvalid(
            "publication engine config must be a dataclass or mapping"
        )
    if not rendered:
        raise HarnessInvalid("publication engine config cannot be empty")
    return jsonable(rendered)


def _frozen_video_config(metadata: Mapping[str, Any]) -> VideoConfig:
    value = metadata.get("video_config")
    if not isinstance(value, dict):
        raise HarnessInvalid("locked manifest lacks metadata.video_config")
    try:
        config = VideoConfig(**value)
    except (TypeError, ValueError) as error:
        raise HarnessInvalid(
            f"locked metadata.video_config is invalid: {error}"
        ) from error
    if _canonical_json(asdict(config)) != _canonical_json(value):
        raise HarnessInvalid(
            "locked metadata.video_config is incomplete or non-canonical"
        )
    return config


def _recovery_operational_provenance(
    campaign_root: Path,
    recovery_authorization: Mapping[str, Any] | None,
    initial_tunnel_epochs: Any,
) -> dict[str, Any]:
    """Resolve a consumed retry binding without changing scientific config."""

    provenance: dict[str, Any] = {
        "manifest_tunnel_epochs_role": "initial_freeze_epochs",
        "initial_freeze_epochs": jsonable(initial_tunnel_epochs),
        "active_recovery_tunnel_epoch": None,
        "recovery_report": None,
    }
    if recovery_authorization is None:
        return provenance
    relative = recovery_authorization.get("path")
    digest = recovery_authorization.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise HarnessInvalid("retry authorization lacks a recovery report path")
    requested = Path(relative)
    if requested.is_absolute() or ".." in requested.parts:
        raise HarnessInvalid("retry recovery report path is unsafe")
    candidate = campaign_root / requested
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(campaign_root.resolve())
    except (OSError, ValueError) as error:
        raise HarnessInvalid("retry recovery report is unavailable") from error
    if candidate.is_symlink() or not resolved.is_file():
        raise HarnessInvalid("retry recovery report must be a regular file")
    if not isinstance(digest, str) or sha256_file(resolved) != digest:
        raise HarnessInvalid("retry recovery report hash differs from authorization")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HarnessInvalid("retry recovery report is not readable JSON") from error
    if not isinstance(report, Mapping):
        raise HarnessInvalid("retry recovery report must contain an object")
    tunnel_epoch = report.get("tunnel_epoch")
    if tunnel_epoch is not None and (
        not isinstance(tunnel_epoch, str)
        or not tunnel_epoch.strip()
        or len(tunnel_epoch) > 128
        or any(character in tunnel_epoch for character in "\r\n")
    ):
        raise HarnessInvalid("retry recovery tunnel_epoch is not a safe public label")
    provenance.update(
        {
            "active_recovery_tunnel_epoch": (
                None if tunnel_epoch is None else tunnel_epoch.strip()
            ),
            "recovery_report": {
                "path": requested.as_posix(),
                "sha256": digest,
            },
        }
    )
    return provenance


def _reference_source_path(repository: Path, reference: str) -> Path | None:
    """Resolve a frozen dotted schema reference to its defining source file."""

    parts = reference.split(".")
    if parts and parts[0] == "src":
        parts = parts[1:]
    for length in range(len(parts), 0, -1):
        candidate = repository / "src" / Path(*parts[:length]).with_suffix(".py")
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    return None


def _profile_scenario_hashes(
    profile: ExperimentProfile, scenario: ScenarioSpec
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    prompt_hashes: dict[str, str | None] = {}
    schema_source_hashes: dict[str, str | None] = {}
    for role, reference in profile.prompt_refs.items():
        if reference is None:
            prompt_hashes[role] = None
            continue
        path = (repository / reference).resolve()
        try:
            path.relative_to(repository)
        except ValueError as error:
            raise HarnessInvalid(f"prompt reference escapes repository: {reference}") from error
        if not path.is_file() or path.is_symlink():
            raise HarnessInvalid(f"prompt reference is unavailable: {reference}")
        prompt_hashes[role] = sha256_file(path)
    for role, reference in profile.schema_refs.items():
        if reference is None:
            schema_source_hashes[role] = None
            continue
        source = _reference_source_path(repository, reference)
        if source is None:
            raise HarnessInvalid(f"schema source is unavailable: {reference}")
        schema_source_hashes[role] = sha256_file(source)
    oracle_source = repository / "experiments" / "harness" / "oracles.py"
    return {
        "profile_sha256": _canonical_sha256(asdict(profile)),
        "scenario_sha256": _canonical_sha256(asdict(scenario)),
        "oracle_spec_sha256": _canonical_sha256(
            [asdict(item) for item in scenario.oracles]
        ),
        "oracle_implementation_sha256": sha256_file(oracle_source),
        "prompt_sha256_by_role": prompt_hashes,
        "schema_source_sha256_by_role": schema_source_hashes,
        "profile_catalogue_sha256": sha256_file(
            repository / "experiments" / "config" / "profiles.json"
        ),
        "scenario_catalogue_sha256": sha256_file(
            repository / "experiments" / "scenarios.json"
        ),
    }


@dataclass(slots=True)
class AttemptContext:
    lease: AttemptLease
    scenario: ScenarioSpec
    profile: ExperimentProfile
    recorder: AttemptRecorder
    video: SynchronizedVideoRecorder
    driver: ScenarioDriver
    topology_manifest: Mapping[str, Any]
    latest_overlay: OverlayState
    _frame_sequence: int = 0
    _initial_written: bool = False
    _final_written: bool = False
    _video_finalized: bool = False
    _last_third_bgr: np.ndarray | None = None
    _last_robot_bgr: np.ndarray | None = None
    _last_source_reference: SourceFrameReference | None = None
    _captures: dict[int, CaptureToken] = field(default_factory=dict)
    _last_capture: CaptureToken | None = None

    def __post_init__(self) -> None:
        """Fail closed unless every evidence surface shares one origin object."""

        if self.recorder.origin is not self.video.origin:
            raise HarnessInvalid(
                "attempt recorder and video must share the injected ClockOrigin"
            )
        driver_recorder = getattr(self.driver, "recorder", None)
        if driver_recorder is not self.recorder:
            raise HarnessInvalid(
                "scenario driver must use the attempt's injected recorder"
            )
        if driver_recorder.origin is not self.recorder.origin:
            raise HarnessInvalid(
                "scenario driver must share the injected ClockOrigin"
            )

    def adopt_opening_source(
        self,
        third_person: np.ndarray,
        robot_camera: np.ndarray,
        reference: SourceFrameReference,
    ) -> None:
        """Adopt the already-recorded setup slate as the reusable source pair."""

        if self.video.frame_count < 1:
            raise HarnessInvalid("cannot adopt an opening source before video start")
        if self._last_third_bgr is not None or self._last_robot_bgr is not None:
            raise HarnessInvalid("attempt context already has an opening source")
        third = np.asarray(third_person)
        robot = np.asarray(robot_camera)
        for name, image in (("third_person", third), ("robot_camera", robot)):
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise HarnessInvalid(f"{name} opening source must be uint8 HxWx3")
        self._last_third_bgr = np.ascontiguousarray(third).copy()
        self._last_robot_bgr = np.ascontiguousarray(robot).copy()
        self._last_source_reference = reference

    @property
    def seed(self) -> int:
        return self.lease.assignment.seed

    @property
    def attempt_id(self) -> str:
        return self.lease.attempt_id

    @property
    def last_robot_png(self) -> bytes:
        if self._last_capture is None:
            raise HarnessInvalid("no robot-camera frame has been captured")
        return self._last_capture.robot_png

    @property
    def last_capture(self) -> CaptureToken:
        if self._last_capture is None:
            raise HarnessInvalid("no robot-camera frame has been captured")
        return self._last_capture

    @property
    def frame_sequence(self) -> int:
        return self._frame_sequence

    def overlay(self, **changes: Any) -> OverlayState:
        values = asdict(self.latest_overlay)
        values.update(changes)
        self.latest_overlay = OverlayState(**values)
        return self.latest_overlay

    def video_tick(
        self,
        third_person: np.ndarray | None = None,
        robot_camera: np.ndarray | None = None,
        *,
        source_elapsed_seconds: float | None = None,
        source_capture_elapsed_seconds: float | None = None,
        source_reference: SourceFrameReference | None = None,
    ) -> int:
        """Record a new source pair or an explicit held-source overlay tick."""

        if third_person is None and robot_camera is None:
            if (
                source_elapsed_seconds is not None
                or source_capture_elapsed_seconds is not None
                or source_reference is not None
            ):
                raise HarnessInvalid(
                    "held-source overlay ticks cannot replace source timing or identity"
                )
            return self.video.advance_overlay(overlay=self.latest_overlay)
        if third_person is None or robot_camera is None:
            raise HarnessInvalid("video tick requires both synchronized source views")

        third = np.asarray(third_person)
        robot = np.asarray(robot_camera)
        reference = source_reference or self._last_source_reference or SourceFrameReference()
        index = self.video.write(
            third,
            robot,
            overlay=self.latest_overlay,
            source_elapsed_seconds=source_elapsed_seconds,
            source_capture_elapsed_seconds=source_capture_elapsed_seconds,
            source_reference=reference,
        )
        self._last_third_bgr = np.ascontiguousarray(third).copy()
        self._last_robot_bgr = np.ascontiguousarray(robot).copy()
        self._last_source_reference = reference
        return index

    def capture(
        self,
        *,
        third_person: np.ndarray,
        robot_camera: np.ndarray,
        public_state: Mapping[str, Any],
        hidden_state: Mapping[str, Any] | None = None,
        phase: str,
        initial: bool = False,
        final: bool = False,
        color_order: str = "rgb",
        source_sequence: int | None = None,
        source_elapsed_seconds: float | None = None,
    ) -> int:
        """Persist one synchronized observation without exposing hidden state."""

        if initial and self._initial_written:
            raise HarnessInvalid("initial state may be written only once")
        if final and self._final_written:
            raise HarnessInvalid("final state may be written only once")
        capture_monotonic_ns = (
            time.monotonic_ns()
            if source_elapsed_seconds is None
            else self.recorder.origin.monotonic_ns
            + round(float(source_elapsed_seconds) * 1_000_000_000)
        )
        capture_stamp = self.recorder.origin.stamp(capture_monotonic_ns)
        robot_png = _encode_png(robot_camera, color_order=color_order)
        self._frame_sequence = (
            self._frame_sequence + 1
            if source_sequence is None
            else int(source_sequence)
        )
        if self._frame_sequence in self._captures:
            raise HarnessInvalid(
                f"duplicate robot-camera frame sequence {self._frame_sequence}"
            )
        public = jsonable(dict(public_state))
        self.recorder.record_runtime_state(public)
        if hidden_state is not None:
            hidden = jsonable(dict(hidden_state))
            self.driver.replace_hidden(hidden, evidence=f"capture:{phase}")
        else:
            hidden = None
        self.overlay(phase=phase, system_state=str(public.get("state", phase)))
        third_bgr = (
            cv2.cvtColor(third_person, cv2.COLOR_RGB2BGR)
            if color_order == "rgb"
            else np.asarray(third_person)
        )
        robot_bgr = (
            cv2.cvtColor(robot_camera, cv2.COLOR_RGB2BGR)
            if color_order == "rgb"
            else np.asarray(robot_camera)
        )
        frame_index = self.video_tick(
            third_bgr,
            robot_bgr,
            source_elapsed_seconds=source_elapsed_seconds,
            source_capture_elapsed_seconds=float(capture_stamp["elapsed_seconds"]),
            source_reference=SourceFrameReference(
                third_person_id=f"overview:{self._frame_sequence}",
                # Must equal FrameRequestRecorder.camera_frame_id for exact
                # request/video source linkage.
                robot_camera_id=f"robot_camera:{self._frame_sequence}",
            ),
        )
        token = CaptureToken(
            frame_sequence=self._frame_sequence,
            video_frame_index=frame_index,
            camera_frame_id=f"robot_camera:{self._frame_sequence}",
            robot_png=robot_png,
            source_frame_sha256=source_frame_sha256(robot_bgr),
            capture_monotonic_ns=capture_monotonic_ns,
            capture_elapsed_seconds=float(capture_stamp["elapsed_seconds"]),
        )
        self._captures[self._frame_sequence] = token
        self._last_capture = token
        event = self.driver.emit(
            "video_frame_captured",
            phase=phase,
            frame_sequence=self._frame_sequence,
            video_frame_index=frame_index,
            robot_image_sha256=sha256_bytes(robot_png),
            hidden_state_recorded=hidden is not None,
        )
        state_payload = {
            "public": public,
            "hidden": hidden,
            "frame_sequence": self._frame_sequence,
            "source_event_id": event["event_id"],
        }
        if initial:
            self.recorder.write_initial_state(state_payload, robot_png)
            self._initial_written = True
        if final:
            self.recorder.write_final_state(state_payload, robot_png)
            self._final_written = True
        return token

    def invoke_model_exchange(
        self,
        *,
        logical_agent: str,
        request_id: str,
        call_id: str,
        model_id: str,
        service_label: str,
        prompt: str,
        request: Mapping[str, Any],
        capture: CaptureToken,
        invoke: Callable[[Mapping[str, Any]], ModelExchangeResult],
        wire_request_factory: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> ModelExchangeResult:
        """Record request evidence, perform transport, then record its result."""

        if self._captures.get(capture.frame_sequence) is not capture:
            raise HarnessInvalid("model request received a foreign or forged capture token")
        if not callable(invoke):
            raise TypeError("invoke must be callable")
        png = capture.robot_png
        payload = _data_url(png)
        if wire_request_factory is None:
            wire_request = jsonable(
                {**dict(request), "exact_image": payload, "prompt": prompt}
            )
        else:
            candidate = wire_request_factory(payload)
            if not isinstance(candidate, Mapping):
                raise HarnessInvalid("wire request factory must return a mapping")
            wire_request = jsonable(dict(candidate))
        request_event = self.driver.emit(
            "model_request",
            request_id=request_id,
            call_id=call_id,
            logical_agent=logical_agent,
            model_id=model_id,
            frame_sequence=capture.frame_sequence,
            camera_frame_id=capture.camera_frame_id,
        )
        frame = self.recorder.frames.record(
            request_id=request_id,
            logical_agent=logical_agent,
            payload=payload,
            camera_view_id="robot_camera",
            frame_sequence_number=capture.frame_sequence,
            prompt_or_request_body=wire_request,
            source_frame_sha256=capture.source_frame_sha256,
            capture_monotonic_ns=capture.capture_monotonic_ns,
            event_ids=(request_event["event_id"],),
        )
        self.recorder.model_calls.start(
            call_id=call_id,
            request_id=request_id,
            logical_agent=logical_agent,
            request=wire_request,
            model_id=model_id,
            service_label=service_label,
            frame_request_ids=(request_id,),
            event_id=request_event["event_id"],
        )
        try:
            result = invoke(wire_request)
            if not isinstance(result, ModelExchangeResult):
                raise HarnessInvalid("model transport returned an invalid exchange contract")
            if not result.response_id.strip():
                raise HarnessInvalid("model transport returned an empty response ID")
        except BaseException as transport_error:
            error_event = self.driver.emit(
                "model_error",
                request_id=request_id,
                call_id=call_id,
                logical_agent=logical_agent,
                error_type=type(transport_error).__name__,
                error_message=str(transport_error),
            )
            self.recorder.model_calls.error(
                call_id=call_id,
                error=transport_error,
                event_id=error_event["event_id"],
            )
            self.recorder.frames.link_error(
                request_id,
                call_id=call_id,
                event_ids=(error_event["event_id"],),
            )
            self.driver.emit(
                "request_frame_linked",
                request_id=request_id,
                response_id=None,
                call_id=call_id,
                error_event_id=error_event["event_id"],
                image_path=frame["image_path"],
                raw_image_sha256=frame["raw_image_sha256"],
                source_frame_sha256=capture.source_frame_sha256,
            )
            raise
        response_event = self.driver.emit(
            "model_response",
            request_id=request_id,
            call_id=call_id,
            response_id=result.response_id,
            logical_agent=logical_agent,
        )
        self.recorder.model_calls.end(
            call_id=call_id,
            response=result.response,
            response_id=result.response_id,
            event_id=response_event["event_id"],
            transport_metadata=result.transport_metadata,
        )
        self.recorder.frames.link_response(
            request_id=request_id,
            response_id=result.response_id,
            event_ids=(request_event["event_id"], response_event["event_id"]),
        )
        self.driver.emit(
            "request_frame_linked",
            request_id=request_id,
            response_id=result.response_id,
            image_path=frame["image_path"],
            raw_image_sha256=frame["raw_image_sha256"],
            source_frame_sha256=capture.source_frame_sha256,
        )
        return result

    def invoke_nonvisual_model_exchange(
        self,
        *,
        logical_agent: str,
        request_id: str,
        call_id: str,
        model_id: str,
        service_label: str,
        request: Mapping[str, Any],
        invoke: Callable[[Mapping[str, Any]], ModelExchangeResult],
    ) -> ModelExchangeResult:
        """Record a text-only exchange without inventing camera provenance."""

        if not callable(invoke):
            raise TypeError("invoke must be callable")
        wire_request = jsonable(dict(request))
        request_event = self.driver.emit(
            "model_request",
            request_id=request_id,
            call_id=call_id,
            logical_agent=logical_agent,
            model_id=model_id,
        )
        self.recorder.model_calls.start(
            call_id=call_id,
            request_id=request_id,
            logical_agent=logical_agent,
            request=wire_request,
            model_id=model_id,
            service_label=service_label,
            frame_request_ids=(),
            event_id=request_event["event_id"],
        )
        try:
            result = invoke(wire_request)
            if not isinstance(result, ModelExchangeResult):
                raise HarnessInvalid("model transport returned an invalid exchange contract")
            if not result.response_id.strip():
                raise HarnessInvalid("model transport returned an empty response ID")
        except BaseException as transport_error:
            error_event = self.driver.emit(
                "model_error",
                request_id=request_id,
                call_id=call_id,
                logical_agent=logical_agent,
                error_type=type(transport_error).__name__,
                error_message=str(transport_error),
            )
            self.recorder.model_calls.error(
                call_id=call_id,
                error=transport_error,
                event_id=error_event["event_id"],
            )
            raise
        response_event = self.driver.emit(
            "model_response",
            request_id=request_id,
            call_id=call_id,
            response_id=result.response_id,
            logical_agent=logical_agent,
        )
        self.recorder.model_calls.end(
            call_id=call_id,
            response=result.response,
            response_id=result.response_id,
            event_id=response_event["event_id"],
            transport_metadata=result.transport_metadata,
        )
        return result

    def finalize_video(self) -> Mapping[str, Any]:
        if self._video_finalized:
            raise HarnessInvalid("video was already finalized")
        metadata = self.video.finalize()
        self._video_finalized = True
        return metadata


@dataclass(frozen=True, slots=True)
class AttemptRunReport:
    schedule_id: str
    attempt_number: int
    attempt_path: Path
    run_status: RunStatus
    oracle_verdict: str | None
    artifact_audit_passed: bool
    evaluation: ScenarioEvaluation | None
    errors: tuple[str, ...] = ()


def _setup_payload(
    *,
    campaign_root: Path,
    manifest: Mapping[str, Any],
    lease: AttemptLease,
    scenario: ScenarioSpec,
    profile: ExperimentProfile,
    topology: Mapping[str, Any],
    origin: ClockOrigin,
    video_config: VideoConfig,
    engine: AttemptEngine,
    cell_capability: CellCapability,
    metadata: Mapping[str, Any],
    recovery_authorization: Mapping[str, Any] | None,
    verify_source: bool,
) -> dict[str, Any]:
    provenance = manifest.get("provenance", {})
    return {
        "schema_version": 1,
        "campaign_id": lease.campaign_id,
        "schedule_id": lease.schedule_id,
        "attempt_id": lease.attempt_id,
        "attempt_number": lease.attempt_number,
        "run_status": RunStatus.NOT_RUN.value,
        "utc_start": origin.utc_iso,
        "utc_end": None,
        "monotonic_clock_origin": origin.to_json(),
        "repository": {
            "commit": provenance.get("git_commit"),
            "dirty": provenance.get("dirty"),
            "dirty_patch_sha256": provenance.get("dirty_patch_sha256"),
            "source_manifest_sha256": provenance.get("source_manifest_sha256"),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **jsonable(metadata.get("environment", {})),
        },
        "assignment": lease.assignment.to_dict(),
        "experiment_family": scenario.family,
        "profile": profile.profile_id,
        "repetition": lease.assignment.repetition,
        "scenario": scenario.scenario_id,
        "scenario_variant": lease.assignment.scenario_variant_id,
        "seed": lease.assignment.seed,
        "executor": lease.assignment.executor,
        "engine": {
            "id": engine.engine_id,
            "publication_valid": engine.publication_valid,
            "supports_locked": engine.supports_locked,
            "config": _engine_config(engine),
            "configuration_source": "CAMPAIGN_MANIFEST.json#metadata",
        },
        "cell_capability": cell_capability.to_dict(),
        "instantiated_logical_agent_graph": jsonable(topology),
        "model_service_mapping": jsonable(metadata.get("model_service_mapping", {})),
        "reported_model_ids": jsonable(metadata.get("reported_model_ids", {})),
        "hashes": {
            "protocol_sha256": manifest.get("protocol_sha256"),
            "schedule_sha256": manifest.get("schedule_sha256"),
            "frozen_inputs_sha256": provenance.get("frozen_inputs_sha256"),
            **_profile_scenario_hashes(profile, scenario),
            **jsonable(metadata.get("hashes", {})),
        },
        "timeouts_thresholds_budgets": jsonable(metadata.get("controls", {})),
        "scene": scenario.setup.scene,
        "world_controller": {
            "setup_executor": scenario.setup.executor,
            "initial_predicates": list(scenario.setup.initial_predicates),
            "max_cycles": scenario.setup.max_cycles,
        },
        "user_task": list(scenario.user_script),
        "expected_oracle_conditions": [
            {"id": item.oracle_id, "type": item.oracle_type, "params": item.params}
            for item in scenario.oracles
        ],
        "schedule_position": lease.assignment.schedule_position,
        "video": asdict(video_config),
        "local_service_endpoints": jsonable(metadata.get("service_endpoints", {})),
        "health_check_results": jsonable(metadata.get("health_checks", {})),
        "tunnel_epochs": jsonable(metadata.get("tunnel_epochs", {})),
        "tunnel_epoch_provenance": _recovery_operational_provenance(
            campaign_root,
            recovery_authorization,
            metadata.get("tunnel_epochs", {}),
        ),
        "infrastructure_retry_authorization": jsonable(recovery_authorization),
        "reproduction_command": (
            f".venv/bin/python -m experiments.run_campaign "
            f"{campaign_root.as_posix()} execute --engine {engine.cli_name} "
            f"--schedule-id {lease.schedule_id}"
            + ("" if verify_source else " --no-source-check")
        ),
        "known_deviations": list(metadata.get("known_deviations", ())),
    }


def _diagnosis(
    *,
    result: EngineResult | None,
    evaluation: ScenarioEvaluation | None,
    status: RunStatus,
    error: BaseException | None,
    evidence_event: Mapping[str, Any] | None,
) -> dict[str, Any]:
    passed = bool(evaluation and evaluation.passed)
    summary = (
        result.summary
        if result is not None
        else f"Attempt terminated with {type(error).__name__}: {error}"
    )
    if evidence_event is not None:
        event_id = str(evidence_event.get("event_id", "E000001"))
        video_seconds = float(
            evidence_event.get(
                "video_time_seconds", evidence_event.get("elapsed_seconds", 0.0)
            )
        )
    else:
        event_id = "E000001"
        video_seconds = 0.0
    evidence = (
        f"[Event {event_id}](./experiment_record.md#event-{event_id.casefold()}), "
        f"[video {video_seconds:.3f}s](./video.mp4#t={video_seconds:.3f})"
    )
    failure_layer = (
        result.primary_diagnosis_layer
        if result is not None
        else "recorder/oracle/harness"
    )
    confidence = result.diagnosis_confidence if result is not None else "high"
    disposition = (
        "infrastructure"
        if status is RunStatus.INFRA_INTERRUPTED
        else "harness"
        if status is RunStatus.INVALID_HARNESS
        else "studied_system"
    )
    competing = (
        list(result.competing_explanations) if result is not None else []
    )
    return {
        "behavioral_summary": f"{summary} ({evidence}).",
        "expected_vs_observed": (
            f"All frozen trigger and oracle conditions passed ({evidence})."
            if passed
            else f"At least one frozen trigger/oracle condition failed or was unavailable ({evidence})."
        ),
        "safety_and_liveness_consequences": (
            f"See the safety oracle and ordered event record ({evidence})."
        ),
        "first_divergence_point": (
            f"none observed ({evidence})"
            if passed
            else f"See the failing oracle evidence and terminal record ({evidence})."
        ),
        "causal_timeline": (
            "[Event E000001](./experiment_record.md#event-e000001) begins the "
            f"append-only attempt timeline; the cited disposition is {evidence}."
        ),
        "likely_failure_layer": f"{failure_layer} ({evidence}).",
        "primary_diagnosis": (
            f"No failure diagnosed ({evidence})."
            if passed
            else f"{summary} ({evidence})."
        ),
        "confidence": f"{confidence} ({evidence}).",
        "competing_explanations": (
            [f"{item} ({evidence})." for item in competing]
            if competing
            else [f"none recorded ({evidence})."]
        ),
        "disconfirming_evidence": f"Raw calls, frames, states, and video are linked from the record ({evidence}).",
        "studied_system_or_infrastructure": f"{disposition} ({evidence}).",
        "recommended_follow_up": f"Review linked evidence without changing frozen results ({evidence}).",
        "reviewer_notes": f"Diagnosis was generated only after behavior and oracle finalization ({evidence}).",
        "unresolved_ambiguity": (
            f"none recorded ({evidence})"
            if passed
            else f"Manual evidence review may refine cause ({evidence})."
        ),
    }


class CampaignRunner:
    """Execute one immutable schedule entry at a time and seal its registry row."""

    def __init__(
        self,
        campaign_root: str | Path,
        *,
        engine: AttemptEngine,
        setup_metadata: Mapping[str, Any] | None = None,
        video_config: VideoConfig | None = None,
        verify_source: bool = True,
    ) -> None:
        self.root = Path(campaign_root).resolve()
        self.engine = engine
        self.verify_source = verify_source
        self.manifest = verify_frozen_campaign(self.root, check_source=verify_source)
        self.registry = CampaignRegistry(self.root)
        manifest_metadata = self.manifest.get("metadata")
        if not isinstance(manifest_metadata, Mapping):
            raise HarnessInvalid("campaign manifest metadata is malformed")
        if self.registry.mode is CampaignMode.LOCKED:
            if not verify_source:
                raise HarnessInvalid(
                    "locked execution requires live-source verification against the freeze"
                )
            if not engine.supports_locked or not engine.publication_valid:
                raise HarnessInvalid(
                    f"engine {engine.engine_id} is not authorized for locked scientific runs"
                )
            frozen = manifest_metadata.get("execution_engine")
            if not isinstance(frozen, Mapping):
                raise HarnessInvalid("locked manifest lacks execution-engine identity")
            engine_type = type(engine)
            source = inspect.getsourcefile(engine_type)
            if source is None:
                raise HarnessInvalid("could not resolve live execution-engine source")
            source_path = Path(source).resolve()
            repository = Path(__file__).resolve().parents[2]
            try:
                relative_source = source_path.relative_to(repository).as_posix()
            except ValueError as error:
                raise HarnessInvalid(
                    "execution engine source is outside the repository"
                ) from error
            live = {
                "engine_id": engine.engine_id,
                "module": engine_type.__module__,
                "class_name": engine_type.__name__,
                "source_path": relative_source,
                "source_sha256": sha256_file(source_path),
            }
            if _canonical_json(frozen) != _canonical_json(live):
                raise HarnessInvalid(
                    "live execution-engine identity/hash differs from frozen metadata"
                )

            frozen_engine_config = manifest_metadata.get("production_engine_config")
            if not isinstance(frozen_engine_config, dict) or not frozen_engine_config:
                raise HarnessInvalid(
                    "locked manifest lacks metadata.production_engine_config"
                )
            live_engine_config = _engine_config(engine)
            if live_engine_config is None or _canonical_json(
                live_engine_config
            ) != _canonical_json(frozen_engine_config):
                raise HarnessInvalid(
                    "live publication-engine config differs from frozen metadata"
                )

            frozen_video = _frozen_video_config(manifest_metadata)
            if video_config is not None and _canonical_json(
                asdict(video_config)
            ) != _canonical_json(asdict(frozen_video)):
                raise HarnessInvalid(
                    "caller video config differs from locked frozen metadata"
                )
            if setup_metadata is not None and _canonical_json(
                setup_metadata
            ) != _canonical_json(manifest_metadata):
                raise HarnessInvalid(
                    "caller setup metadata differs from locked frozen metadata"
                )
            self.setup_metadata = dict(manifest_metadata)
            self.video_config = frozen_video
            self._locked_engine_config = _canonical_json(frozen_engine_config)
            self._locked_video_config = _canonical_json(asdict(frozen_video))
            self._locked_setup_metadata = _canonical_json(manifest_metadata)
        else:
            if setup_metadata is not None and _canonical_json(
                setup_metadata
            ) != _canonical_json(manifest_metadata):
                raise HarnessInvalid(
                    "caller setup metadata differs from frozen campaign metadata"
                )
            self.setup_metadata = dict(manifest_metadata)
            if not verify_source:
                deviations = list(self.setup_metadata.get("known_deviations", ()))
                deviation = "Pilot executed without checking live source against freeze."
                if deviation not in deviations:
                    deviations.append(deviation)
                self.setup_metadata["known_deviations"] = deviations
            manifest_video = manifest_metadata.get("video_config")
            if manifest_video is not None:
                frozen_video = _frozen_video_config(manifest_metadata)
                if video_config is not None and _canonical_json(
                    asdict(video_config)
                ) != _canonical_json(asdict(frozen_video)):
                    raise HarnessInvalid(
                        "caller video config differs from frozen campaign metadata"
                    )
                self.video_config = frozen_video
            else:
                self.video_config = video_config or VideoConfig()
            self._locked_engine_config = None
            self._locked_video_config = None
            self._locked_setup_metadata = None

    def _assert_locked_execution_binding(self) -> None:
        """Recheck immutable instance bindings immediately before reservation."""

        if self.registry.mode is not CampaignMode.LOCKED:
            return
        live_engine_config = _engine_config(self.engine)
        if live_engine_config is None or _canonical_json(
            live_engine_config
        ) != self._locked_engine_config:
            raise HarnessInvalid(
                "live publication-engine config changed after locked initialization"
            )
        if _canonical_json(asdict(self.video_config)) != self._locked_video_config:
            raise HarnessInvalid(
                "live video config changed after locked initialization"
            )
        if _canonical_json(self.setup_metadata) != self._locked_setup_metadata:
            raise HarnessInvalid(
                "live setup metadata changed after locked initialization"
            )

    def close(self) -> None:
        """Release any pending engine runtime or transport resources."""

        close = getattr(self.engine, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "CampaignRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def run_next(self) -> AttemptRunReport | None:
        try:
            self._assert_locked_execution_binding()
            entry = self.registry.next_entry()
        except BaseException:
            self.close()
            raise
        if entry is None:
            self.close()
            return None
        return self.run_schedule_id(entry.schedule_id)

    def run_schedule_id(self, schedule_id: str) -> AttemptRunReport:
        try:
            return self._run_schedule_id(schedule_id)
        finally:
            # topology_manifest() may allocate transports/simulator resources
            # before any attempt artifact exists.  Always unwind them, including
            # validation and registry failures, and keep the engine reusable.
            self.close()

    def _run_schedule_id(self, schedule_id: str) -> AttemptRunReport:
        self._assert_locked_execution_binding()
        # Resolve only the next registry-authorized assignment.  Capability
        # validation therefore cannot bind a later cell out of frozen order or
        # run before locked approval/recovery authorization.
        assignment = self.registry.next_entry()
        if assignment is None:
            raise HarnessInvalid("no runnable schedule entry remains")
        if assignment.schedule_id != schedule_id:
            raise HarnessInvalid(
                f"frozen order requires {assignment.schedule_id} next; "
                f"refusing {schedule_id}"
            )
        scenario = load_scenarios().get(assignment.scenario_id)
        profile = load_profiles().get(assignment.profile_id)
        split_key = _scenario_split_key(assignment.split)
        try:
            expected_variant = scenario.split_variants[split_key].variant_id
        except KeyError as error:
            raise HarnessInvalid(
                f"schedule split has no frozen scenario variant: {assignment.split}"
            ) from error
        if assignment.scenario_variant_id != expected_variant:
            raise HarnessInvalid("schedule variant does not match frozen scenario split")
        if profile.profile_id not in scenario.applicable_profiles:
            raise HarnessInvalid("profile is not applicable to frozen scenario")
        assert_executor_allowed(
            profile,
            assignment.executor,
            scenario_allowed=scenario.safety.allowed_executors,
            safety_screened=bool(self.setup_metadata.get("safety_screened", False)),
            ethics_approved=bool(self.setup_metadata.get("ethics_approved", False)),
        )
        capability = _validated_cell_capability(
            self.engine,
            assignment,
            scenario,
            profile,
        )
        # begin_attempt() atomically rechecks authorization, recovery state, and
        # frozen order after the read-only capability check.  No recorder path
        # exists until both checks have passed.
        lease = self.registry.begin_attempt(schedule_id)
        return self._execute(lease, scenario, profile, capability)

    def _execute(
        self,
        lease: AttemptLease,
        scenario: ScenarioSpec,
        profile: ExperimentProfile,
        cell_capability: CellCapability,
    ) -> AttemptRunReport:
        # The immutable synthetic slate exists before the attempt clock.  It
        # can therefore truthfully anchor source/video time at exactly zero,
        # including when production topology construction is slow.
        slate = _opening_slate(lease.attempt_id)
        opening_reference = SourceFrameReference(
            third_person_id="setup_slate:0",
            robot_camera_id="setup_slate:0",
            source_kind="synthetic_setup",
        )
        opening_overlay = OverlayState(
            episode=lease.attempt_id,
            profile=profile.profile_id,
            scenario=scenario.scenario_id,
            seed=lease.assignment.seed,
            latest_user_request=scenario.user_script[0],
        )
        origin = ClockOrigin.capture()
        recorder = AttemptRecorder.create(
            self.root,
            schedule_id=lease.schedule_id,
            attempt_number=lease.attempt_number,
            origin=origin,
        )
        video: SynchronizedVideoRecorder | None = None
        context: AttemptContext | None = None
        evaluation: ScenarioEvaluation | None = None
        evaluation_capability: Any | None = None
        engine_result: EngineResult | None = None
        error: BaseException | None = None
        status = RunStatus.INVALID_HARNESS
        verdict: str | None = None
        audit: Mapping[str, Any] = {"passed": False, "errors": ["not finalized"]}
        infrastructure_evidence_path: str | None = None
        diagnosis_event: Mapping[str, Any] | None = None
        try:
            video = SynchronizedVideoRecorder(
                recorder.attempt_dir, origin=origin, config=self.video_config
            )
            video.write(
                slate,
                slate,
                overlay=opening_overlay,
                source_elapsed_seconds=0.0,
                source_reference=opening_reference,
            )
            attempt_entry = self.registry.snapshot()["entries"][lease.schedule_id][
                "attempts"
            ][lease.attempt_number - 1]
            recovery_authorization = attempt_entry.get("recovery_authorization")
            topology = jsonable(
                self.engine.topology_manifest(profile, lease.assignment.executor)
            )
            setup = _setup_payload(
                campaign_root=self.root,
                manifest=self.manifest,
                lease=lease,
                scenario=scenario,
                profile=profile,
                topology=topology,
                origin=origin,
                video_config=self.video_config,
                engine=self.engine,
                cell_capability=cell_capability,
                metadata=self.setup_metadata,
                recovery_authorization=recovery_authorization,
                verify_source=self.verify_source,
            )
            recorder.write_setup(setup)
            recorder.write_inputs(
                {
                    "scenario_id": scenario.scenario_id,
                    "split_variant": asdict(
                        scenario.split_variants[
                            _scenario_split_key(lease.assignment.split)
                        ]
                    ),
                    "user_script": list(scenario.user_script),
                    "seed": lease.assignment.seed,
                    "crn_key": lease.assignment.crn_key,
                }
            )
            driver, evaluation_capability = ScenarioDriver.with_evaluation_capability(
                scenario,
                recorder,
                trigger_handlers=self.engine.trigger_handlers(scenario),
            )
            context = AttemptContext(
                lease=lease,
                scenario=scenario,
                profile=profile,
                recorder=recorder,
                video=video,
                driver=driver,
                topology_manifest=topology,
                latest_overlay=opening_overlay,
            )
            context.adopt_opening_source(slate, slate, opening_reference)
            recorder.terminal(f"attempt {lease.attempt_id} started with {self.engine.engine_id}")
            driver.emit(
                "attempt_started",
                attempt_id=lease.attempt_id,
                profile_id=profile.profile_id,
                engine_id=self.engine.engine_id,
            )
            engine_result = self.engine.execute(context)
            if not isinstance(engine_result, EngineResult):
                raise HarnessInvalid("engine returned an invalid result contract")
            if not context._initial_written or not context._final_written:
                raise HarnessInvalid("engine did not persist both initial and final state")
            evaluation = driver.evaluate(evaluation_capability)
            verdict = "PASS" if evaluation.passed else "FAIL"
            status = (
                RunStatus.VALID_PASS
                if evaluation.passed
                else RunStatus.VALID_SYSTEM_FAILURE
            )
            outcome_event = driver.emit(
                "attempt_outcome_frozen",
                run_status=status.value,
                oracle_verdict=verdict,
            )
            diagnosis_event = outcome_event
            context.overlay(
                system_state=engine_result.terminal_state,
                phase="FINALIZED",
                latest_system_output=engine_result.summary,
                terminal_lines=(
                    f"{outcome_event['event_id']} {status.value} oracle={verdict}",
                ),
            )
            # The outcome and final overlay are themselves evidence.  Advance
            # the source clock after emitting them so every event is inside the
            # audited video interval, even when inference created a long gap.
            context.video_tick()
            context.finalize_video()
        except InfrastructureInterruption as caught:
            error = caught
            status = RunStatus.INFRA_INTERRUPTED
            evidence_path = recorder.write_json(
                "infrastructure_evidence.json",
                {
                    "schema_version": 1,
                    "classification": RunStatus.INFRA_INTERRUPTED.value,
                    "message": str(caught),
                    **caught.evidence.to_dict(),
                },
            )
            infrastructure_evidence_path = evidence_path.relative_to(
                recorder.attempt_dir
            ).as_posix()
        except SafetyAbort as caught:
            error = caught
            status = RunStatus.ABORTED_SAFETY
        except (HarnessInvalid, RecordingError, VideoError, ScenarioDriverError) as caught:
            error = caught
            status = RunStatus.INVALID_HARNESS
        except BaseException as caught:  # preserve unexpected failures as harness-invalid evidence
            error = caught
            status = RunStatus.INVALID_HARNESS

        if error is not None:
            recorder.terminal(f"{status.value}: {type(error).__name__}: {error}")
            try:
                termination_event = recorder.events.append(
                    "attempt_terminated",
                    run_status=status.value,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                diagnosis_event = termination_event
                recorder.write_error(
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": "".join(
                            traceback.format_exception(type(error), error, error.__traceback__)
                        ),
                    }
                )
            except Exception:
                termination_event = None
            if context is not None and not context._video_finalized:
                try:
                    if context.video.frame_count:
                        context.overlay(
                            system_state=status.value,
                            phase="TERMINATED",
                            latest_system_output=f"{type(error).__name__}: {error}",
                            terminal_lines=(
                                (
                                    f"{termination_event['event_id']} {status.value}: "
                                    f"{type(error).__name__}"
                                )
                                if termination_event is not None
                                else f"{status.value}: {type(error).__name__}"
                            ),
                        )
                        context.video_tick()
                        context.finalize_video()
                    else:
                        context.video.abort(reason=f"{type(error).__name__}: {error}")
                except Exception:
                    pass
            elif video is not None:
                try:
                    if video.frame_count:
                        video.advance_overlay(
                            overlay=OverlayState(
                                **{
                                    **asdict(opening_overlay),
                                    "system_state": status.value,
                                    "phase": "TERMINATED",
                                    "latest_system_output": (
                                        f"{type(error).__name__}: {error}"
                                    ),
                                }
                            )
                        )
                        video.finalize()
                    else:
                        video.abort(reason=f"{type(error).__name__}: {error}")
                except Exception:
                    pass

        try:
            result_payload = {
                "attempt_id": lease.attempt_id,
                "scenario_id": scenario.scenario_id,
                "profile_id": profile.profile_id,
                "engine_id": self.engine.engine_id,
                "engine_publication_valid": self.engine.publication_valid,
                "behavior": None if engine_result is None else asdict(engine_result),
                "evaluation": None if evaluation is None else evaluation.to_dict(),
                "metrics": {} if engine_result is None else jsonable(engine_result.metrics),
                "complete": error is None,
            }
            audit = recorder.finalize(
                run_status=status,
                oracle_verdict=verdict,
                result=result_payload,
                diagnosis=_diagnosis(
                    result=engine_result,
                    evaluation=evaluation,
                    status=status,
                    error=error,
                    evidence_event=diagnosis_event,
                ),
                require_final_state=error is None,
            )
            effective_status = audit.get("effective_run_status")
            if effective_status is not None:
                status = RunStatus(str(effective_status))
            if status is RunStatus.INVALID_HARNESS and error is None:
                error = HarnessInvalid(
                    "artifact integrity audit failed: "
                    + "; ".join(map(str, audit.get("errors", ())))
                )
        except Exception as finalize_error:
            error = error or finalize_error
            status = RunStatus.INVALID_HARNESS
            verdict = None
            audit = recorder.preserve_partial(
                reason=f"attempt finalization failed: {type(finalize_error).__name__}: {finalize_error}",
                run_status=status,
            )

        self.registry.seal_attempt(
            lease.schedule_id,
            lease.attempt_number,
            RegistryRunStatus(status.value),
            oracle_verdict=verdict,
            reason=None if error is None else f"{type(error).__name__}: {error}",
            artifact_audit_passed=bool(audit.get("passed")),
            independently_evidenced_infrastructure=(
                status is RunStatus.INFRA_INTERRUPTED
            ),
            infrastructure_evidence=infrastructure_evidence_path,
        )
        self.registry.export_run_index()
        return AttemptRunReport(
            schedule_id=lease.schedule_id,
            attempt_number=lease.attempt_number,
            attempt_path=recorder.attempt_dir,
            run_status=status,
            oracle_verdict=verdict,
            artifact_audit_passed=bool(audit.get("passed")),
            evaluation=evaluation,
            errors=tuple(str(item) for item in audit.get("errors", ())),
        )


__all__ = [
    "AttemptContext",
    "AttemptEngine",
    "AttemptExecutionError",
    "AttemptRunReport",
    "CampaignRunner",
    "CaptureToken",
    "CellCapability",
    "EngineResult",
    "HarnessInvalid",
    "InfrastructureInterruption",
    "InfrastructureFailureEvidence",
    "ModelExchangeResult",
    "SafetyAbort",
]
