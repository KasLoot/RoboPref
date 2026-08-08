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
    SceneChangeEvent,
)
from prefmem.execution.frames import RGBDFrame
from prefmem.execution.grounding import RGBDGrounder


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
        self.compiler = compiler
        self.grounder = grounder
        self.controller = controller
        self.frame_source = frame_source
        self.on_event = on_event
        self.on_scene_change = on_scene_change
        self.scene_poll_interval = float(scene_poll_interval)
        self.scene_change_confirmations = int(scene_change_confirmations)
        self.clock = clock

        self._condition = threading.Condition()
        self._pending: _ExecutionJob | None = None
        self._active: _ExecutionJob | None = None
        self._cancel_event: threading.Event | None = None
        self._states: dict[str, ExecutionState] = {}
        self._stopping = False
        self._worker = threading.Thread(
            target=self._run,
            name="prefmem-execution",
            daemon=True,
        )
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
        guard: threading.Thread | None = None
        try:
            program = self.compiler.compile(task)
            self._raise_if_cancelled(cancel_event)
            frame = self.frame_source()
            if not isinstance(frame, RGBDFrame):
                raise TypeError("execution frame source must return RGBDFrame")
            source = self.grounder.ground(frame, program.source)
            target = self.grounder.ground(frame, program.target)
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
            self._emit(
                publication_id,
                ExecutionState.SETTLED,
                "Motion completed and robot is holding position",
            )
        except InterruptedError as error:
            self.controller.safe_hold()
            self._emit(
                publication_id,
                ExecutionState.CANCELLED,
                str(error) or "Execution cancelled",
            )
        except Exception as error:
            self.controller.safe_hold()
            if cancel_event.is_set():
                self._emit(
                    publication_id,
                    ExecutionState.CANCELLED,
                    "Execution cancelled",
                )
            else:
                self._emit(
                    publication_id,
                    ExecutionState.FAULT,
                    f"Execution failed: {error}",
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


__all__ = ["ExecutionService", "PickPlaceController", "RGBDFrameSource"]
