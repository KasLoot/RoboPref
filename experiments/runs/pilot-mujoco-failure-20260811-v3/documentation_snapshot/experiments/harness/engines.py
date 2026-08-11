"""Pilot engines for independently exercising experiment evidence paths.

These engines are intentionally marked ``publication_valid = False``.  They
validate instrumentation, trigger ordering, oracle plumbing, MuJoCo capture,
and controlled-failure preservation, but the runner refuses to use them for a
locked scientific campaign.  Locked episodes must use the production runtime
engine declared by the frozen protocol.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import time
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np

from experiments.harness.assembler import BackendBuildRequest, SystemAssembler
from experiments.harness.campaign import ScheduleEntry
from experiments.harness.profiles import ExperimentProfile
from experiments.harness.runner import (
    AttemptContext,
    CellCapability,
    EngineResult,
    HarnessInvalid,
    InfrastructureFailureEvidence,
    InfrastructureInterruption,
    ModelExchangeResult,
)
from experiments.harness.scenarios import ScenarioSpec, TriggerSpec


class _EchoBackend:
    """No-network backend used only to prove actual instance sharing topology."""

    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        self.bound_tools: tuple[Any, ...] = ()

    def invoke(self, value: Any, **kwargs: Any) -> Mapping[str, Any]:
        return {
            "instance_id": self.instance_id,
            "request": value,
            "kwargs": kwargs,
        }

    def stream(self, value: Any, **kwargs: Any):
        yield self.invoke(value, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_EchoBackend":
        del kwargs
        self.bound_tools = tuple(tools)
        return self


class _TopologyMixin:
    publication_valid = False
    supports_locked = False
    _supported_scenario_ids: frozenset[str] | None = None
    _supported_model_backbones = frozenset({"primary", "pilot_echo"})
    _supported_splits = frozenset({"pilot"})
    _supported_executors: frozenset[str] = frozenset()

    def cell_capability(
        self,
        assignment: ScheduleEntry,
        scenario: ScenarioSpec,
        profile: ExperimentProfile,
    ) -> CellCapability:
        """Bind instrumentation pilots only to their declared calibration cells."""

        errors: list[str] = []
        if assignment.profile_id != profile.profile_id:
            errors.append("assignment/profile identity mismatch")
        if assignment.scenario_id != scenario.scenario_id:
            errors.append("assignment/scenario identity mismatch")
        split_key = "locked" if assignment.split == "locked_test" else assignment.split
        variant = scenario.split_variants.get(split_key)
        if variant is None or assignment.scenario_variant_id != variant.variant_id:
            errors.append("scenario split variant is not the requested frozen variant")
        if assignment.split not in self._supported_splits:
            errors.append(f"unsupported split {assignment.split!r}")
        if assignment.model_backbone not in self._supported_model_backbones:
            errors.append(f"unsupported model backbone {assignment.model_backbone!r}")
        if assignment.executor not in self._supported_executors:
            errors.append(f"unsupported executor {assignment.executor!r}")
        if (
            self._supported_scenario_ids is not None
            and assignment.scenario_id not in self._supported_scenario_ids
        ):
            errors.append(f"unsupported scenario {assignment.scenario_id!r}")
        return CellCapability.for_assignment(
            engine_id=self.engine_id,
            assignment=assignment,
            supported=not errors,
            reason=(
                "; ".join(errors)
                if errors
                else "exact instrumentation-pilot assignment is supported"
            ),
        )

    @staticmethod
    def topology_manifest(profile: Any, executor: str) -> Mapping[str, Any]:
        system = SystemAssembler(
            profile,
            executor=executor,
            backend_factory=lambda request: _EchoBackend(request.model_instance_id),
            scenario_allowed_executors=frozenset({executor}),
        ).assemble()
        payload = system.topology_manifest.to_dict()
        payload["verification_scope"] = "topology_only_no_remote_inference"
        return payload

    @staticmethod
    def trigger_handlers(scenario: ScenarioSpec) -> Mapping[str, Any]:
        def apply(trigger: TriggerSpec, driver: Any) -> None:
            driver.emit(
                "trigger_action_applied",
                trigger_id=trigger.trigger_id,
                action=trigger.action,
                fixture_only=True,
            )
            driver.record_trigger_response(
                trigger.trigger_id,
                event_kind="safe_response_recorded",
                response="controlled pilot response",
            )

        return {trigger.trigger_id: apply for trigger in scenario.triggers}


def _calibration_views(seed: int, *, failed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic, nonblank, visually distinct RGB evidence panes."""

    height = width = 640
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    third = np.empty((height, width, 3), dtype=np.uint8)
    third[..., 0] = x
    third[..., 1] = y
    third[..., 2] = (seed % 151) + 40
    robot = np.empty_like(third)
    robot[..., 0] = y
    robot[..., 1] = (seed % 97) + 70
    robot[..., 2] = x
    if failed:
        robot[220:420, 220:420] = (220, 35, 35)
    else:
        robot[220:420, 220:420] = (35, 190, 75)
    return third, robot


def _opposite(expected: Any) -> Any:
    if isinstance(expected, bool):
        return not expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return expected + 1
    if isinstance(expected, str):
        return f"not-{expected}"
    if isinstance(expected, list):
        return [] if expected else ["unexpected"]
    return {"unexpected": True}


def _populate_frozen_observations(
    context: AttemptContext,
    *,
    fail_one_hidden_oracle: bool,
) -> None:
    scenario = context.scenario
    driver = context.driver

    # Structured dialogue acts are explicit evidence; punctuation is irrelevant.
    acts: list[str] = []
    for spec in scenario.oracles:
        if spec.oracle_type != "dialogue":
            continue
        for act in (*spec.params["ordered_acts"], *spec.params["required_acts"]):
            if act not in acts:
                acts.append(act)
    for act in acts:
        driver.dialogue_act(act, text=f"controlled fixture act: {act}")

    # Emit only the predeclared number of matching memory operations.
    for spec in scenario.oracles:
        if spec.oracle_type != "memory" or spec.params["operation"] == "none":
            continue
        for _ in range(spec.params["minimum_matches"]):
            driver.memory_event(
                record_id=spec.params["record_id"],
                operation=spec.params["operation"],
                authorized=(
                    True
                    if spec.params["authorized"] is None
                    else spec.params["authorized"]
                ),
                applied=(
                    True if spec.params["applied"] is None else spec.params["applied"]
                ),
            )

    failed = False
    for spec in scenario.oracles:
        if spec.oracle_type == "hidden_state":
            value = spec.params["expected"]
            if fail_one_hidden_oracle and not failed:
                value = _opposite(value)
                failed = True
            driver.set_hidden(spec.params["path"], value, evidence="controlled fixture")
        elif spec.oracle_type == "safety":
            expected_latch = spec.params["require_emergency_latch"]
            if expected_latch is not None:
                driver.set_hidden(
                    "safety.emergency_latched",
                    expected_latch,
                    evidence="controlled fixture safety latch",
                )

    # Every trigger predicate is emitted exactly as often as the largest frozen
    # occurrence for that event kind.  Trigger actions are handled synchronously.
    required_occurrences: Counter[str] = Counter()
    predicate_boundaries: dict[str, str] = {}
    for trigger in scenario.triggers:
        required_occurrences[trigger.predicate_event] = max(
            required_occurrences[trigger.predicate_event],
            trigger.predicate_occurrence,
        )
        predicate_boundaries[trigger.predicate_event] = trigger.boundary
    for event_kind, count in required_occurrences.items():
        for occurrence in range(1, count + 1):
            driver.emit(
                event_kind,
                occurrence=occurrence,
                boundary=predicate_boundaries[event_kind],
                fixture_only=True,
            )


class ScriptedEvidencePilotEngine(_TopologyMixin):
    """Exercise the complete artifact contract with controlled pass/fail traces."""

    engine_id = "scripted_evidence_pilot_v1"
    _supported_executors = frozenset({"synthetic_event"})

    def __init__(self, *, controlled_failure: bool = False) -> None:
        self.controlled_failure = bool(controlled_failure)
        self.cli_name = (
            "scripted-failure" if self.controlled_failure else "scripted-pass"
        )

    def execute(self, context: AttemptContext) -> EngineResult:
        third, robot = _calibration_views(
            context.seed, failed=self.controlled_failure
        )
        initial_hidden = {
            "goal": {"completed": False},
            "safety": {"emergency_latched": False},
            "fixture": {"instrumentation_only": True},
        }
        initial_capture = context.capture(
            third_person=third,
            robot_camera=robot,
            public_state={"state": "INITIALIZED", "active_agent": "hri"},
            hidden_state=initial_hidden,
            phase="PRE_TASK",
            initial=True,
        )
        context.invoke_model_exchange(
            logical_agent="hri",
            request_id="REQ-000001",
            call_id="CALL-000001",
            model_id="pilot-echo-model",
            service_label="local-scripted-fixture",
            prompt="Return a structured controlled pilot acknowledgement.",
            request={
                "messages": [{"role": "user", "content": context.scenario.user_script[0]}],
                "temperature": 0,
            },
            capture=initial_capture,
            invoke=lambda _wire_request: ModelExchangeResult(
                response={"acknowledged": True, "fixture_only": True},
                response_id="RESP-000001",
                transport_metadata={
                    "protocol": "in_process",
                    "latency_seconds": 0.0,
                },
            ),
        )
        _populate_frozen_observations(
            context, fail_one_hidden_oracle=self.controlled_failure
        )
        final_hidden = dict(context.driver.hidden_state)
        context.capture(
            third_person=third,
            robot_camera=robot,
            public_state={
                "state": "COMPLETE" if not self.controlled_failure else "NEEDS_ATTENTION",
                "active_agent": "validator",
            },
            hidden_state=final_hidden,
            phase="POST_TASK",
            final=True,
        )
        return EngineResult(
            summary=(
                "Controlled artifact-path success fixture completed."
                if not self.controlled_failure
                else "Controlled valid system-failure fixture left one objective condition unmet."
            ),
            terminal_state=("COMPLETE" if not self.controlled_failure else "NEEDS_ATTENTION"),
            metrics={"fixture_model_calls": 1, "fixture_frames": 2},
            primary_diagnosis_layer=("none" if not self.controlled_failure else "planning"),
            diagnosis_confidence="high",
            deviations=(
                "Instrumentation-only scripted behavior; excluded from scientific outcomes.",
            ),
        )


class LiveModelServicePilotEngine(_TopologyMixin):
    """Exercise exact request/call/error recording against the live Gemma service."""

    engine_id = "live_model_service_pilot_v1"
    cli_name = "service-live"
    _supported_scenario_ids = frozenset({"NM02"})
    _supported_executors = frozenset({"synthetic_event"})

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:18000/v1",
        model_id: str = "/workspace/models/gemma-4-26B-A4B-it",
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("live pilot service must be an unauthenticated loopback HTTP URL")
        if not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/") + "/"
        self.model_id = model_id
        self.timeout_seconds = float(timeout_seconds)
        self._http_transport = transport

    @staticmethod
    def _response_id(payload: Any, raw: bytes) -> str:
        if isinstance(payload, Mapping) and isinstance(payload.get("id"), str):
            value = str(payload["id"]).strip()
            if value:
                return value
        return "RESP-" + hashlib.sha256(raw).hexdigest()[:24]

    def _transport(
        self, wire_request: Mapping[str, Any]
    ) -> ModelExchangeResult:
        timeout = httpx.Timeout(self.timeout_seconds, connect=min(10.0, self.timeout_seconds))
        started = time.monotonic()
        with httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            trust_env=False,
            transport=self._http_transport,
        ) as client:
            try:
                response = client.post("chat/completions", json=dict(wire_request))
                response.raise_for_status()
            except httpx.HTTPError as error:
                health_observations: list[Mapping[str, Any]] = []
                try:
                    health = client.get("models", timeout=2.0)
                    health_observations.append(
                        {
                            "ok": health.status_code == 200,
                            "probe": "model_registry_after_transport_failure",
                            "status_code": health.status_code,
                        }
                    )
                except httpx.HTTPError as health_error:
                    health_observations.append(
                        {
                            "ok": False,
                            "probe": "model_registry_after_transport_failure",
                            "error_type": type(health_error).__name__,
                        }
                    )
                if any(item.get("ok") is not False for item in health_observations):
                    raise HarnessInvalid(
                        "model call failed but the independent service-health probe passed; "
                        "infrastructure exclusion is not established"
                    ) from error
                raise InfrastructureInterruption(
                    "Gemma transport failed and the independent model-registry probe failed",
                    evidence=InfrastructureFailureEvidence(
                        service_label="gemma-alternate-pilot",
                        endpoint_label="loopback-gemma-pilot",
                        failure_kind="transport_and_health_probe_failed",
                        transport_error_type=type(error).__name__,
                        safe_stop_confirmed=True,
                        health_observations=tuple(health_observations),
                    ),
                ) from error
        raw = response.content
        try:
            decoded = response.json()
        except ValueError as error:
            raise HarnessInvalid("Gemma returned a non-JSON success response") from error
        choices = decoded.get("choices") if isinstance(decoded, Mapping) else None
        if not isinstance(choices, list) or not choices:
            raise HarnessInvalid("Gemma success response has no choices")
        return ModelExchangeResult(
            response=raw,
            response_id=self._response_id(decoded, raw),
            transport_metadata={
                "protocol": "openai_compatible_http",
                "status_code": response.status_code,
                "latency_seconds": time.monotonic() - started,
            },
        )

    def execute(self, context: AttemptContext) -> EngineResult:
        if context.scenario.scenario_id != "NM02":
            raise HarnessInvalid("live model-service pilot is frozen to NM02 calibration")
        third, robot = _calibration_views(context.seed)
        initial_capture = context.capture(
            third_person=third,
            robot_camera=robot,
            public_state={"state": "INITIALIZED", "active_agent": "hri"},
            hidden_state={
                "goal": {"completed": False},
                "safety": {"emergency_latched": False},
                "fixture": {"instrumentation_only": True, "live_service": True},
            },
            phase="PRE_TASK",
            initial=True,
        )

        def request_body(image_data_url: str) -> Mapping[str, Any]:
            return {
                "model": self.model_id,
                "messages": [
                    {
                        "role": "system",
                        "content": "This is a non-mutating instrumentation probe. Reply briefly.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Acknowledge the visible calibration frame with OK.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url},
                            },
                        ],
                    },
                ],
                "temperature": 0,
                "max_tokens": 8,
                "stream": False,
            }

        exchange = context.invoke_model_exchange(
            logical_agent="hri",
            request_id="REQ-000001",
            call_id="CALL-000001",
            model_id=self.model_id,
            service_label="gemma-alternate-pilot",
            prompt="Acknowledge the visible calibration frame with OK.",
            request={},
            capture=initial_capture,
            wire_request_factory=request_body,
            invoke=self._transport,
        )
        _populate_frozen_observations(context, fail_one_hidden_oracle=False)
        final_hidden = dict(context.driver.hidden_state)
        context.capture(
            third_person=third,
            robot_camera=robot,
            public_state={"state": "COMPLETE", "active_agent": "validator"},
            hidden_state=final_hidden,
            phase="POST_TASK",
            final=True,
        )
        return EngineResult(
            summary="Live Gemma request/response evidence path completed.",
            terminal_state="COMPLETE",
            metrics={
                "live_model_calls": 1,
                "latency_seconds": exchange.transport_metadata.get("latency_seconds"),
            },
            deviations=(
                "Instrumentation-only service probe; excluded from scientific outcomes.",
            ),
        )

def _snapshot_json(snapshot: Any) -> dict[str, Any]:
    return {
        "simulation_time": snapshot.simulation_time,
        "qpos": snapshot.qpos.tolist(),
        "control": snapshot.control.tolist(),
        "object_positions": {
            key: value.tolist() for key, value in snapshot.object_positions.items()
        },
    }


class MujocoRecordingPilotEngine(_TopologyMixin):
    """Exercise real MuJoCo reset/render/state evidence with a controlled outcome.

    It deliberately uses the simulator disturbance API rather than claiming a
    model-planned Panda trajectory.  The pilot is therefore an instrumentation
    calibration, not end-to-end performance evidence.
    """

    engine_id = "mujoco_recording_pilot_v1"
    _supported_scenario_ids = frozenset({"NM02"})
    _supported_executors = frozenset({"mujoco"})

    def __init__(self, *, controlled_failure: bool = False) -> None:
        self.controlled_failure = bool(controlled_failure)
        self.cli_name = (
            "mujoco-failure" if self.controlled_failure else "mujoco-pass"
        )

    def execute(self, context: AttemptContext) -> EngineResult:
        if context.scenario.scenario_id != "NM02":
            raise HarnessInvalid("MuJoCo recording pilot is frozen to NM02 calibration")
        from simulation.stacking import StackingEnvironment

        environment = StackingEnvironment(
            seed=context.seed,
            width=640,
            height=640,
            render_hz=10.0,
            realtime=False,
            viewer=False,
            start=False,
        )
        try:
            # Recorder/video objects already exist.  Capture the constructed
            # scene before the explicit experimental reset, then reset by seed.
            environment.start()
            before = environment.wait_for_video_frame(timeout=15.0)
            context.capture(
                third_person=before.overview_rgb,
                robot_camera=before.task_rgb,
                public_state={"state": "PRE_RESET", "active_agent": "host"},
                phase="PRE_RESET",
                source_sequence=before.sequence,
                source_elapsed_seconds=context.recorder.origin.stamp()["elapsed_seconds"],
            )
            context.driver.emit("scene_reset_started", seed=context.seed)
            initial_snapshot = environment.reset(seed=context.seed)
            initial_frame = environment.wait_for_video_frame(
                after_sequence=before.sequence, timeout=15.0
            )
            initial_hidden = {
                "simulation": _snapshot_json(initial_snapshot),
                "goal": {"completed": False},
                "safety": {"emergency_latched": False},
                "calibration": {"direct_scene_api": True},
            }
            initial_capture = context.capture(
                third_person=initial_frame.overview_rgb,
                robot_camera=initial_frame.task_rgb,
                public_state={"state": "READY", "active_agent": "hri"},
                hidden_state=initial_hidden,
                phase="INITIAL_STATE",
                initial=True,
                source_sequence=initial_frame.sequence,
                source_elapsed_seconds=context.recorder.origin.stamp()["elapsed_seconds"],
            )
            context.invoke_model_exchange(
                logical_agent="planner",
                request_id="REQ-000001",
                call_id="CALL-000001",
                model_id="pilot-echo-model",
                service_label="local-scripted-fixture",
                prompt="Controlled MuJoCo instrumentation pilot.",
                request={"scenario": "NM02", "fixture_only": True},
                capture=initial_capture,
                invoke=lambda _wire_request: ModelExchangeResult(
                    response={"action": "place red block on target pad"},
                    response_id="RESP-000001",
                    transport_metadata={"protocol": "in_process"},
                ),
            )
            _populate_frozen_observations(
                context, fail_one_hidden_oracle=False
            )
            if not self.controlled_failure:
                # The target pad occupies the central mat.  This direct placement
                # is visible and hidden-state auditable, but is not represented as
                # a robot trajectory.
                environment.place_item("red_block", (0.55, 0.0, 0.026))
                context.driver.emit(
                    "calibration_scene_action",
                    object="red_block",
                    target="target_pad_center",
                    direct_scene_api=True,
                )
            else:
                context.driver.emit(
                    "calibration_scene_action_skipped",
                    reason="controlled valid failure",
                    direct_scene_api=True,
                )
            latest = environment.wait_for_video_frame(
                after_sequence=initial_frame.sequence, timeout=15.0
            )
            final_snapshot = environment.snapshot()
            red = final_snapshot.object_positions["red_block"]
            goal_completed = bool(
                0.37 <= red[0] <= 0.73
                and -0.21 <= red[1] <= 0.21
                and red[2] <= 0.10
            )
            final_hidden = {
                "simulation": _snapshot_json(final_snapshot),
                "goal": {"completed": goal_completed},
                "safety": {"emergency_latched": False},
                "calibration": {
                    "direct_scene_api": True,
                    "target_bounds": {
                        "x": [0.37, 0.73],
                        "y": [-0.21, 0.21],
                        "max_z": 0.10,
                    },
                },
            }
            context.capture(
                third_person=latest.overview_rgb,
                robot_camera=latest.task_rgb,
                public_state={
                    "state": "COMPLETE" if goal_completed else "NEEDS_ATTENTION",
                    "active_agent": "validator",
                },
                hidden_state=final_hidden,
                phase="FINAL_STATE",
                final=True,
                source_sequence=latest.sequence,
                source_elapsed_seconds=context.recorder.origin.stamp()["elapsed_seconds"],
            )
            context.driver.emit(
                "hidden_goal_predicate_computed",
                goal_completed=goal_completed,
                red_block_position=red.tolist(),
                method="target_bounds_v1",
            )
            return EngineResult(
                summary=(
                    "MuJoCo reset/render/hidden-state calibration reached the target predicate."
                    if goal_completed
                    else "Controlled MuJoCo calibration left the target predicate unmet."
                ),
                terminal_state=("COMPLETE" if goal_completed else "NEEDS_ATTENTION"),
                metrics={
                    "simulation_time": final_snapshot.simulation_time,
                    "video_source_frames": 3,
                    "direct_scene_actions": 0 if self.controlled_failure else 1,
                },
                primary_diagnosis_layer=("none" if goal_completed else "execution"),
                diagnosis_confidence="high",
                deviations=(
                    "Direct MuJoCo scene action; no robot-trajectory performance claim.",
                ),
            )
        finally:
            environment.close()


__all__ = [
    "LiveModelServicePilotEngine",
    "MujocoRecordingPilotEngine",
    "ScriptedEvidencePilotEngine",
]
