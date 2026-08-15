"""Single-slot execution service joining language, perception, and motion."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Protocol

import numpy as np

from prefmem.contracts import DynamicObjectScope, PublishedTask, TaskPhase
from prefmem.execution.contracts import (
    ExecutionEvent,
    ExecutionState,
    ManipulationProgram,
    ObjectReference,
    SceneChangeEvent,
)
from prefmem.execution.frames import RGBDFrame
from prefmem.execution.fallback import (
    AssistedContinuationResult,
    DownstreamExecutionResult,
    ExecutionOutcome,
    GroundingAttempt,
    OracleGrounding,
    OracleGroundingProvider,
    SAM_GROUNDING_SOURCE,
    SIMULATOR_GROUND_TRUTH_SOURCE,
    StrictSystemResult,
)
from prefmem.execution.grounding import GroundedObject, GroundingError, RGBDGrounder
from prefmem.execution.sam import SamServiceError


class InstructionCompiler(Protocol):
    def compile(self, task: PublishedTask) -> ManipulationProgram: ...


class PickPlaceController(Protocol):
    def execute_pick_place(
        self,
        program: ManipulationProgram,
        source_world: np.ndarray,
        target_world: np.ndarray,
        cancel_event: threading.Event,
    ) -> None: ...

    def safe_hold(self) -> None: ...


class RGBDFrameSource(Protocol):
    def __call__(self) -> RGBDFrame: ...


@dataclass(frozen=True, slots=True)
class _ExecutionJob:
    task: PublishedTask
    dynamic_scope: DynamicObjectScope | None


class ExecutionService:
    """Keep one authoritative task and fence all work by publication ID.

    The worker blocks on the deterministic motion controller. A lightweight
    guard may inspect newer RGB-D frames concurrently and requests a safe
    cancellation when an open-set goal gains a new matching object.
    """

    def __init__(
        self,
        compiler: InstructionCompiler,
        grounder: RGBDGrounder,
        controller: PickPlaceController,
        frame_source: RGBDFrameSource,
        *,
        on_event: Callable[[ExecutionEvent], None] | None = None,
        on_scene_change: Callable[[SceneChangeEvent], None] | None = None,
        on_trace: Callable[[dict], None] | None = None,
        on_perception: Callable[[dict], None] | None = None,
        enable_oracle_grounding_fallback: bool = False,
        oracle_grounding_provider: OracleGroundingProvider | None = None,
        scene_poll_interval: float = 0.5,
        scene_change_confirmations: int = 2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not hasattr(compiler, "compile"):
            raise TypeError("compiler must expose compile()")
        if not isinstance(grounder, RGBDGrounder):
            raise TypeError("grounder must be an RGBDGrounder")
        if not hasattr(controller, "execute_pick_place") or not hasattr(
            controller, "safe_hold"
        ):
            raise TypeError("controller must expose execute_pick_place() and safe_hold()")
        if not callable(frame_source):
            raise TypeError("frame_source must be callable")
        if scene_poll_interval <= 0:
            raise ValueError("scene_poll_interval must be positive")
        if scene_change_confirmations < 1:
            raise ValueError("scene_change_confirmations must be positive")
        if type(enable_oracle_grounding_fallback) is not bool:
            raise TypeError("enable_oracle_grounding_fallback must be bool")
        provider_configured = oracle_grounding_provider is not None
        if enable_oracle_grounding_fallback != provider_configured:
            raise ValueError(
                "oracle fallback requires both the explicit opt-in and a provider"
            )
        if provider_configured:
            if not callable(
                getattr(
                    oracle_grounding_provider,
                    "ground_from_simulator_truth",
                    None,
                )
            ):
                raise TypeError(
                    "oracle_grounding_provider must expose "
                    "ground_from_simulator_truth()"
                )
            if (
                getattr(oracle_grounding_provider, "source_id", None)
                != SIMULATOR_GROUND_TRUTH_SOURCE
            ):
                raise ValueError(
                    "oracle grounding provider must identify "
                    "SIMULATOR_GROUND_TRUTH"
                )
        self.compiler = compiler
        self.grounder = grounder
        self.controller = controller
        self.frame_source = frame_source
        self.on_event = on_event
        self.on_scene_change = on_scene_change
        self.on_trace = on_trace
        self.on_perception = on_perception
        self.enable_oracle_grounding_fallback = enable_oracle_grounding_fallback
        self.oracle_grounding_provider = oracle_grounding_provider
        self.scene_poll_interval = float(scene_poll_interval)
        self.scene_change_confirmations = int(scene_change_confirmations)
        self.clock = clock

        self._condition = threading.Condition()
        self._pending: _ExecutionJob | None = None
        self._active: _ExecutionJob | None = None
        self._cancel_event: threading.Event | None = None
        self._states: dict[str, ExecutionState] = {}
        self._outcomes: dict[str, ExecutionOutcome] = {}
        self._stopping = False
        self._worker = threading.Thread(
            target=self._run,
            name="prefmem-execution",
            daemon=True,
        )
        set_trace_callback = getattr(controller, "set_trace_callback", None)
        if callable(set_trace_callback):
            set_trace_callback(self._handle_controller_trace)
        set_compiler_trace_callback = getattr(
            compiler,
            "set_trace_callback",
            None,
        )
        if callable(set_compiler_trace_callback):
            set_compiler_trace_callback(self._handle_compiler_trace)
        self._worker.start()

    def publish(
        self,
        task: PublishedTask,
        dynamic_scope: DynamicObjectScope | None = None,
    ) -> None:
        if not isinstance(task, PublishedTask):
            raise TypeError("task must be a PublishedTask")
        if task.phase is not TaskPhase.STEP:
            raise ValueError("ExecutionService accepts only STEP tasks")
        if dynamic_scope is not None and not isinstance(
            dynamic_scope, DynamicObjectScope
        ):
            raise TypeError("dynamic_scope must be DynamicObjectScope or None")
        job = _ExecutionJob(task, dynamic_scope)
        with self._condition:
            if self._stopping:
                raise RuntimeError("execution service is stopped")
            if (
                self._active is not None
                and self._active.task.publication_id != task.publication_id
            ):
                assert self._cancel_event is not None
                self._cancel_event.set()
            if self._pending is not None:
                old_id = self._pending.task.publication_id
                self._states[old_id] = ExecutionState.CANCELLED
            self._pending = job
            self._states[task.publication_id] = ExecutionState.PREPARING
            self._deliver_event(
                ExecutionEvent(
                    publication_id=task.publication_id,
                    state=ExecutionState.PREPARING,
                    message="Compiling task",
                    observed_at=self.clock(),
                )
            )
            self._trace(
                task.publication_id,
                "EXECUTION_ACCEPTED",
                instruction=task.instruction,
                phase=task.phase.value,
            )
            self._condition.notify_all()

    def retire(self, publication_id: str) -> bool:
        if not isinstance(publication_id, str) or not publication_id.strip():
            raise ValueError("publication_id must be non-empty")
        retired = False
        with self._condition:
            if (
                self._pending is not None
                and self._pending.task.publication_id == publication_id
            ):
                self._pending = None
                self._states[publication_id] = ExecutionState.CANCELLED
                retired = True
            if (
                self._active is not None
                and self._active.task.publication_id == publication_id
            ):
                assert self._cancel_event is not None
                self._cancel_event.set()
                retired = True
            self._condition.notify_all()
        return retired

    def state(self, publication_id: str) -> ExecutionState | None:
        with self._condition:
            return self._states.get(publication_id)

    def outcome(self, publication_id: str) -> dict | None:
        """Return the terminal strict/assisted result for one publication."""

        with self._condition:
            value = self._outcomes.get(publication_id)
        return None if value is None else value.to_dict()

    def outcomes(self) -> tuple[dict, ...]:
        """Return all terminal outcomes in publication insertion order."""

        with self._condition:
            values = tuple(self._outcomes.values())
        return tuple(value.to_dict() for value in values)

    def is_settled(self, publication_id: str) -> bool:
        return self.state(publication_id) is ExecutionState.SETTLED

    def emergency_stop(self) -> None:
        with self._condition:
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._pending = None
            self._condition.notify_all()
        self.controller.safe_hold()

    def stop(self, *, timeout: float = 5.0) -> None:
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            self._pending = None
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._condition.notify_all()
        self.controller.safe_hold()
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=max(0.0, timeout))

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                job = self._pending
                self._pending = None
                assert job is not None
                self._active = job
                cancel_event = threading.Event()
                self._cancel_event = cancel_event
            self._execute(job, cancel_event)
            with self._condition:
                if self._active == job:
                    self._active = None
                    self._cancel_event = None

    def _execute(self, job: _ExecutionJob, cancel_event: threading.Event) -> None:
        task = job.task
        publication_id = task.publication_id
        guard_stop = threading.Event()
        scene_change: list[tuple[str, int | None]] = []
        grounding_attempts: list[GroundingAttempt] = []
        guard: threading.Thread | None = None
        try:
            self._trace(
                publication_id,
                "COMPILER_REQUESTED",
                instruction=task.instruction,
                expected_observation=[
                    item.description for item in task.expected_observation
                ],
            )
            program = self.compiler.compile(task)
            self._trace(
                publication_id,
                "COMPILER_OUTPUT_ACCEPTED",
                program=program.to_dict(),
            )
            self._raise_if_cancelled(cancel_event)
            frame = self.frame_source()
            if not isinstance(frame, RGBDFrame):
                raise TypeError("execution frame source must return RGBDFrame")
            calibration = frame.calibration
            self._trace(
                publication_id,
                "RGBD_FRAME_CAPTURED",
                frame_sequence=frame.sequence,
                simulation_time=frame.simulation_time,
                resolution=[calibration.width, calibration.height],
                intrinsic=calibration.intrinsic.astype(float).tolist(),
                world_from_camera=(
                    calibration.world_from_camera.astype(float).tolist()
                ),
            )
            self._perception(
                {
                    "schema_version": 1,
                    "kind": "RGBD_FRAME",
                    "publication_id": publication_id,
                    "frame_sequence": frame.sequence,
                    "rgb": frame.rgb.copy(),
                    "depth_m": frame.depth_m.copy(),
                    "calibration": calibration,
                }
            )
            source = self._ground_reference(
                publication_id,
                frame,
                program.source,
                role="source",
                attempts=grounding_attempts,
            )
            target = self._ground_reference(
                publication_id,
                frame,
                program.target,
                role="target",
                attempts=grounding_attempts,
            )
            self._raise_if_cancelled(cancel_event)

            if job.dynamic_scope is not None:
                baseline_count = self.grounder.count(
                    frame, job.dynamic_scope.selector
                )
                if baseline_count < 1:
                    raise RuntimeError(
                        "dynamic object selector has no visible baseline matches"
                    )
                guard = threading.Thread(
                    target=self._guard_scene,
                    args=(
                        publication_id,
                        job.dynamic_scope,
                        frame.sequence,
                        baseline_count,
                        cancel_event,
                        guard_stop,
                        scene_change,
                    ),
                    name=f"prefmem-scene-guard-{publication_id}",
                    daemon=True,
                )
                guard.start()

            self._emit(publication_id, ExecutionState.RUNNING, "Executing motion")
            self.controller.execute_pick_place(
                program,
                source.point_world,
                target.point_world,
                cancel_event,
            )
            guard_stop.set()
            if guard is not None:
                guard.join(timeout=max(1.0, self.scene_poll_interval * 2.0))
                if not guard.is_alive():
                    guard = None
            if cancel_event.is_set():
                raise InterruptedError("execution was cancelled")
            outcome = self._record_outcome(
                publication_id,
                ExecutionState.SETTLED,
                grounding_attempts,
                DownstreamExecutionResult.PASS,
            )
            self._emit(
                publication_id,
                ExecutionState.SETTLED,
                (
                    "Motion completed with simulator-ground-truth assistance; "
                    "strict SAM grounding failed"
                    if outcome.oracle_fallback_used
                    else "Motion completed and robot is holding position"
                ),
            )
            self._trace(
                publication_id,
                "EXECUTION_SETTLED",
                execution_outcome=outcome.to_dict(),
            )
        except InterruptedError as error:
            self.controller.safe_hold()
            outcome = self._record_outcome(
                publication_id,
                ExecutionState.CANCELLED,
                grounding_attempts,
                DownstreamExecutionResult.CANCELLED,
            )
            self._emit(
                publication_id,
                ExecutionState.CANCELLED,
                str(error) or "Execution cancelled",
            )
            self._trace(
                publication_id,
                "EXECUTION_CANCELLED",
                error_type=type(error).__name__,
                error=str(error),
                execution_outcome=outcome.to_dict(),
            )
        except Exception as error:
            self.controller.safe_hold()
            if cancel_event.is_set():
                outcome = self._record_outcome(
                    publication_id,
                    ExecutionState.CANCELLED,
                    grounding_attempts,
                    DownstreamExecutionResult.CANCELLED,
                )
                self._emit(
                    publication_id,
                    ExecutionState.CANCELLED,
                    "Execution cancelled",
                )
                self._trace(
                    publication_id,
                    "EXECUTION_CANCELLED",
                    error_type=type(error).__name__,
                    error=str(error),
                    execution_outcome=outcome.to_dict(),
                )
            else:
                outcome = self._record_outcome(
                    publication_id,
                    ExecutionState.FAULT,
                    grounding_attempts,
                    DownstreamExecutionResult.FAIL,
                )
                self._emit(
                    publication_id,
                    ExecutionState.FAULT,
                    f"Execution failed: {error}",
                )
                self._trace(
                    publication_id,
                    "EXECUTION_FAULT",
                    error_type=type(error).__name__,
                    error=str(error),
                    execution_outcome=outcome.to_dict(),
                )
        finally:
            guard_stop.set()
            if guard is not None and guard is not threading.current_thread():
                guard.join(timeout=max(1.0, self.scene_poll_interval * 2.0))
            if scene_change:
                description, sequence = scene_change[0]
                self._notify_scene_change(
                    SceneChangeEvent(
                        publication_id=publication_id,
                        description=description,
                        observed_at=self.clock(),
                        frame_sequence=sequence,
                    )
                )

    def _ground_reference(
        self,
        publication_id: str,
        frame: RGBDFrame,
        reference: ObjectReference,
        *,
        role: str,
        attempts: list[GroundingAttempt],
    ) -> GroundedObject | OracleGrounding:
        """Run SAM first and use simulator truth only after a typed failure."""

        self._trace(
            publication_id,
            "SAM_GROUNDING_REQUESTED",
            role=role,
            query=reference.query,
            anchor=reference.anchor.value,
            frame_sequence=frame.sequence,
            oracle_fallback_enabled=self.enable_oracle_grounding_fallback,
        )
        sam_started = time.perf_counter()
        try:
            grounded = self.grounder.ground(frame, reference)
        except GroundingError as error:
            sam_elapsed = time.perf_counter() - sam_started
            reason_code = error.reason_code
            self._trace(
                publication_id,
                "SAM_GROUNDING_FAILED",
                role=role,
                query=reference.query,
                anchor=reference.anchor.value,
                frame_sequence=frame.sequence,
                strict_grounding_result="FAIL",
                strict_failure_reason_code=reason_code,
                strict_failure_message=str(error),
                sam_grounding_elapsed_seconds=sam_elapsed,
                oracle_fallback_enabled=self.enable_oracle_grounding_fallback,
            )
            self._perception(
                {
                    "schema_version": 1,
                    "kind": "GROUNDING_FAILURE",
                    "publication_id": publication_id,
                    "frame_sequence": frame.sequence,
                    "role": role,
                    "query": reference.query,
                    "anchor": reference.anchor.value,
                    "strict_grounding_result": "FAIL",
                    "strict_failure_reason_code": reason_code,
                    "strict_failure_message": str(error),
                }
            )
            if not self.enable_oracle_grounding_fallback:
                attempts.append(
                    GroundingAttempt(
                        role=role,
                        query=reference.query,
                        anchor=reference.anchor.value,
                        strict_grounding_result="FAIL",
                        strict_failure_reason_code=reason_code,
                        strict_failure_message=str(error),
                        grounding_source=None,
                        oracle_fallback_used=False,
                        point_world_m=None,
                        frame_sequence=frame.sequence,
                    )
                )
                raise

            provider = self.oracle_grounding_provider
            assert provider is not None
            oracle_started = time.perf_counter()
            try:
                assisted = provider.ground_from_simulator_truth(
                    frame,
                    reference,
                    role=role,
                    strict_failure_reason_code=reason_code,
                )
                self._validate_oracle_grounding(frame, reference, assisted)
            except Exception as fallback_error:
                attempts.append(
                    GroundingAttempt(
                        role=role,
                        query=reference.query,
                        anchor=reference.anchor.value,
                        strict_grounding_result="FAIL",
                        strict_failure_reason_code=reason_code,
                        strict_failure_message=str(error),
                        grounding_source=None,
                        oracle_fallback_used=False,
                        point_world_m=None,
                        frame_sequence=frame.sequence,
                    )
                )
                self._trace(
                    publication_id,
                    "ORACLE_GROUNDING_FALLBACK_FAILED",
                    role=role,
                    query=reference.query,
                    strict_failure_reason_code=reason_code,
                    fallback_error_type=type(fallback_error).__name__,
                    fallback_error=str(fallback_error),
                    oracle_grounding_elapsed_seconds=(
                        time.perf_counter() - oracle_started
                    ),
                )
                raise

            point = tuple(float(value) for value in assisted.point_world)
            attempt = GroundingAttempt(
                role=role,
                query=reference.query,
                anchor=reference.anchor.value,
                strict_grounding_result="FAIL",
                strict_failure_reason_code=reason_code,
                strict_failure_message=str(error),
                grounding_source=assisted.source_id,
                oracle_fallback_used=True,
                point_world_m=point,
                frame_sequence=assisted.frame_sequence,
            )
            attempts.append(attempt)
            self._trace(
                publication_id,
                "ORACLE_GROUNDING_FALLBACK_USED",
                **attempt.to_dict(),
                sam_grounding_elapsed_seconds=sam_elapsed,
                oracle_grounding_elapsed_seconds=(
                    time.perf_counter() - oracle_started
                ),
                diagnostics=dict(assisted.diagnostics or {}),
            )
            self._trace(
                publication_id,
                "GROUNDING_COMPLETE",
                **attempt.to_dict(),
                uncertainty_m=0.0,
                sam_grounding_elapsed_seconds=sam_elapsed,
                diagnostics=dict(assisted.diagnostics or {}),
            )
            self._perception(
                {
                    "schema_version": 1,
                    "kind": "ORACLE_GROUNDING_FALLBACK",
                    "publication_id": publication_id,
                    **attempt.to_dict(),
                    "diagnostics": dict(assisted.diagnostics or {}),
                }
            )
            return assisted
        except SamServiceError as error:
            sam_elapsed = time.perf_counter() - sam_started
            self._trace(
                publication_id,
                "SAM_GROUNDING_SERVICE_FAILURE",
                role=role,
                query=reference.query,
                anchor=reference.anchor.value,
                frame_sequence=frame.sequence,
                error_type=type(error).__name__,
                error=str(error),
                sam_grounding_elapsed_seconds=sam_elapsed,
                oracle_fallback_eligible=False,
                oracle_fallback_used=False,
            )
            self._perception(
                {
                    "schema_version": 1,
                    "kind": "GROUNDING_SERVICE_FAILURE",
                    "publication_id": publication_id,
                    "frame_sequence": frame.sequence,
                    "role": role,
                    "query": reference.query,
                    "anchor": reference.anchor.value,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "oracle_fallback_eligible": False,
                    "oracle_fallback_used": False,
                }
            )
            raise

        sam_elapsed = time.perf_counter() - sam_started
        point = tuple(float(value) for value in grounded.point_world)
        attempt = GroundingAttempt(
            role=role,
            query=grounded.query,
            anchor=grounded.anchor.value,
            strict_grounding_result="PASS",
            strict_failure_reason_code=None,
            strict_failure_message=None,
            grounding_source=SAM_GROUNDING_SOURCE,
            oracle_fallback_used=False,
            point_world_m=point,
            frame_sequence=grounded.frame_sequence,
        )
        attempts.append(attempt)
        self._trace(
            publication_id,
            "GROUNDING_COMPLETE",
            **attempt.to_dict(),
            uncertainty_m=grounded.uncertainty_m,
            sam_grounding_elapsed_seconds=sam_elapsed,
            diagnostics=dict(grounded.diagnostics or {}),
        )
        self._perception(
            {
                "schema_version": 1,
                "kind": "GROUNDING_MASK",
                "publication_id": publication_id,
                "frame_sequence": grounded.frame_sequence,
                "role": role,
                "query": grounded.query,
                "box_xyxy": grounded.detection.box_xyxy,
                "score": grounded.detection.score,
                "mask": (
                    None
                    if grounded.detection.mask is None
                    else grounded.detection.mask.copy()
                ),
                "strict_grounding_result": "PASS",
                "grounding_source": SAM_GROUNDING_SOURCE,
                "oracle_fallback_used": False,
            }
        )
        return grounded

    @staticmethod
    def _validate_oracle_grounding(
        frame: RGBDFrame,
        reference: ObjectReference,
        grounded: OracleGrounding,
    ) -> None:
        if not isinstance(grounded, OracleGrounding):
            raise TypeError("oracle provider must return OracleGrounding")
        if grounded.query != reference.query or grounded.anchor is not reference.anchor:
            raise ValueError("oracle grounding changed the requested reference")
        if grounded.frame_sequence != frame.sequence:
            raise ValueError("oracle grounding is associated with a different frame")

    def _record_outcome(
        self,
        publication_id: str,
        terminal_state: ExecutionState,
        grounding_attempts: list[GroundingAttempt],
        downstream_result: DownstreamExecutionResult,
    ) -> ExecutionOutcome:
        attempts = tuple(grounding_attempts)
        fallback_used = any(item.oracle_fallback_used for item in attempts)
        strict_grounding_failed = any(
            item.strict_grounding_result == "FAIL" for item in attempts
        )
        if strict_grounding_failed:
            strict_result = StrictSystemResult.FAIL_GROUNDING
        elif terminal_state is ExecutionState.SETTLED:
            strict_result = StrictSystemResult.PASS
        elif terminal_state is ExecutionState.CANCELLED:
            strict_result = StrictSystemResult.CANCELLED
        else:
            strict_result = StrictSystemResult.FAIL_EXECUTION

        if not fallback_used:
            assisted_result = AssistedContinuationResult.NOT_APPLICABLE
        elif downstream_result is DownstreamExecutionResult.PASS:
            assisted_result = AssistedContinuationResult.PASS
        elif downstream_result is DownstreamExecutionResult.CANCELLED:
            assisted_result = AssistedContinuationResult.CANCELLED
        else:
            assisted_result = AssistedContinuationResult.FAIL

        outcome = ExecutionOutcome(
            publication_id=publication_id,
            terminal_state=terminal_state.value,
            strict_system_result=strict_result,
            oracle_fallback_used=fallback_used,
            oracle_fallback_source=(
                SIMULATOR_GROUND_TRUTH_SOURCE if fallback_used else None
            ),
            assisted_continuation_result=assisted_result,
            downstream_execution_result=downstream_result,
            grounding_attempts=attempts,
        )
        with self._condition:
            self._outcomes[publication_id] = outcome
        return outcome

    def _guard_scene(
        self,
        publication_id: str,
        scope: DynamicObjectScope,
        initial_sequence: int,
        baseline_count: int,
        cancel_event: threading.Event,
        stop_event: threading.Event,
        result: list[tuple[str, int | None]],
    ) -> None:
        last_sequence = initial_sequence
        confirmations = 0
        while not stop_event.wait(self.scene_poll_interval):
            if cancel_event.is_set():
                return
            try:
                frame = self.frame_source()
                if not isinstance(frame, RGBDFrame) or frame.sequence <= last_sequence:
                    continue
                last_sequence = frame.sequence
                count = self.grounder.count(frame, scope.selector)
            except Exception:
                # A transient detector/frame failure is not evidence of a scene
                # change and must not interrupt an otherwise safe trajectory.
                confirmations = 0
                continue
            if stop_event.is_set() or cancel_event.is_set():
                return
            confirmations = confirmations + 1 if count > baseline_count else 0
            if confirmations >= self.scene_change_confirmations:
                result.append(
                    (
                        f"Detected {count} {scope.selector!r} objects in the robot "
                        f"workspace; execution began with {baseline_count}",
                        frame.sequence,
                    )
                )
                cancel_event.set()
                return

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            raise InterruptedError("execution was cancelled")

    def _emit(
        self,
        publication_id: str,
        state: ExecutionState,
        message: str,
    ) -> None:
        with self._condition:
            self._states[publication_id] = state
        self._deliver_event(
            ExecutionEvent(
                publication_id=publication_id,
                state=state,
                message=message,
                observed_at=self.clock(),
            )
        )

    def _deliver_event(self, event: ExecutionEvent) -> None:
        callback = self.on_event
        if callback is not None:
            try:
                callback(event)
            except Exception:
                # Runtime callbacks must not kill the physical-control worker.
                pass

    def _notify_scene_change(self, event: SceneChangeEvent) -> None:
        callback = self.on_scene_change
        if callback is not None:
            try:
                callback(event)
            except Exception:
                pass

    def _handle_controller_trace(self, event: dict) -> None:
        with self._condition:
            active = self._active
            publication_id = (
                None if active is None else active.task.publication_id
            )
        if publication_id is None:
            return
        payload = dict(event)
        kind = str(payload.pop("kind", "CONTROLLER_EVENT"))
        self._trace(publication_id, kind, **payload)

    def _handle_compiler_trace(self, event: dict) -> None:
        with self._condition:
            active = self._active
            publication_id = (
                None if active is None else active.task.publication_id
            )
        if publication_id is None:
            return
        payload = dict(event)
        kind = str(payload.pop("kind", "EXECUTION_MODEL_CALL"))
        self._trace(publication_id, kind, **payload)

    def _trace(self, publication_id: str, kind: str, **payload) -> None:
        callback = self.on_trace
        if callback is None:
            return
        event = {
            "schema_version": 1,
            "kind": kind,
            "publication_id": publication_id,
            "observed_at": self.clock(),
            **payload,
        }
        try:
            callback(event)
        except Exception:
            # Experiment logging cannot kill the physical-control worker.
            pass

    def _perception(self, artifact: dict) -> None:
        callback = self.on_perception
        if callback is None:
            return
        try:
            callback(artifact)
        except Exception:
            # Artifact persistence failures are classified by the harness.
            pass


__all__ = ["ExecutionService", "PickPlaceController", "RGBDFrameSource"]
