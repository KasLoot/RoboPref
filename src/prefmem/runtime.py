"""Composition and orchestration for PrefMem's receding-horizon runtime."""

from __future__ import annotations

import json
import queue
import threading
import uuid
from typing import Any, Sequence

from prefmem.agents.monitor import (
    CapturedFrame,
    HTTPFrameSource,
    MonitorErrorEvent,
    MonitorService,
)
from prefmem.contracts import (
    GoalContract,
    GoalProposal,
    MonitorAssessment,
    PlanStatus,
    PlannerCycleRequest,
)
from prefmem.controller import (
    RecedingControllerSnapshot,
    RecedingControllerState,
    RecedingHorizonController,
    RecedingResult,
    RecedingTransition,
)
from prefmem.emergency import (
    EmergencyStopCoordinator,
    EmergencyStopEvent,
    emergency_stop,
)
from prefmem.task_publisher import (
    CameraTaskPublisher,
    ControllerDisplay,
    DisplayState,
    TaskPublisherError,
)


class RuntimeClosedError(RuntimeError):
    """Raised when work is requested after runtime shutdown."""


class PrefMemRuntime:
    """Own Planner cycles, task publication, Monitor, and shutdown.

    Planner and Monitor remain stateless model boundaries.  This runtime is the
    only component allowed to append execution history, select the first task
    from a candidate horizon, publish it, or request the next planning cycle.
    """

    def __init__(
        self,
        planner: Any,
        *,
        camera_base_url: str = "http://127.0.0.1:1234",
        task_publisher: CameraTaskPublisher | None = None,
        frame_source: Any | None = None,
        controller: RecedingHorizonController | None = None,
        monitor: MonitorService | None = None,
        monitor_min_interval: float = 1.0,
        model_name: str = "/workspace/models/gemma-4-26B-A4B-it",
        model_base_url: str = "http://localhost:8000/v1",
        metrics: Any | None = None,
        success_confirmations: int = 2,
        success_stability_seconds: float = 2.0,
        failure_confirmations: int = 2,
        ongoing_timeout_seconds: float = 30.0,
        max_cycles: int = 20,
        session_id: str | None = None,
        reset_display: bool = True,
    ) -> None:
        if not hasattr(planner, "preview") or not hasattr(planner, "plan_cycle"):
            raise TypeError("planner must expose preview() and plan_cycle()")
        self.planner = planner
        self.session_id = session_id or uuid.uuid4().hex
        self.controller = controller or RecedingHorizonController(
            success_confirmations=success_confirmations,
            success_stability_seconds=success_stability_seconds,
            failure_confirmations=failure_confirmations,
            ongoing_timeout_seconds=ongoing_timeout_seconds,
            max_cycles=max_cycles,
        )
        self.publisher = task_publisher or CameraTaskPublisher(camera_base_url)
        snapshot_url = f"{camera_base_url.rstrip('/')}/snapshot.jpg"
        self.frame_source = frame_source or HTTPFrameSource(snapshot_url)
        self.notifications: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._planning_lock = threading.Lock()
        self._display_lock = threading.Lock()
        self._display_sequence = 0
        self._closed = False
        self._pending_goal: GoalContract | None = None

        self.emergency = EmergencyStopCoordinator(
            emergency_stop,
            on_stop=self._on_emergency_stop,
        )
        self.monitor = monitor or MonitorService(
            self._on_monitor_assessment,
            on_error=self._on_monitor_error,
            emergency=self.emergency,
            snapshot_url=snapshot_url,
            min_interval_seconds=monitor_min_interval,
            model_name=model_name,
            model_base_url=model_base_url,
            metrics=metrics,
        )
        if reset_display:
            # A new process/session owns a fresh single display slot.  Failure
            # here is actionable because the web page is the human executor's
            # only task surface.
            try:
                self.publisher.reset()
            except TaskPublisherError as error:
                raise RuntimeError(
                    "Could not initialize the camera task surface. Restart "
                    "`stream_camera` so its current /api/task endpoint is "
                    f"running at {camera_base_url}, then start PrefMem again: "
                    f"{error}"
                ) from error

    @property
    def shutdown_event(self) -> threading.Event:
        return self.emergency.shutdown_event

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_goal(self) -> GoalContract | None:
        return self._pending_goal

    def request_goal_preview(
        self,
        clarified_goal: str,
        *,
        constraints: Sequence[str] = (),
        operator_guidance: str | None = None,
    ) -> dict[str, Any]:
        """Ask Planner for the nominal strategy and stage its frozen goal."""

        self._ensure_running()
        previous = self.controller.snapshot
        if previous.state in {
            RecedingControllerState.PLANNING,
            RecedingControllerState.EXECUTING,
            RecedingControllerState.FINAL_VALIDATION,
        }:
            raise RuntimeError("cannot preview a replacement goal during execution")

        # Once the operator asks for a different preview, the previous proposal
        # is no longer confirmable.  This is especially important when the new
        # preview is BLOCKED and therefore never reaches ``stage_goal``.
        self._pending_goal = None
        if previous.state is RecedingControllerState.AWAITING_CONFIRMATION:
            self.controller.cancel_staged_goal()
        frame = self._capture_frame()
        proposal: GoalProposal = self.planner.preview(
            clarified_goal,
            frame.image_block,
            constraints=constraints,
            operator_guidance=operator_guidance,
        )
        if not isinstance(proposal, GoalProposal):
            raise TypeError("planner.preview() must return a GoalProposal")
        result: dict[str, Any] = {"proposal": proposal.to_dict()}
        if proposal.status is PlanStatus.BLOCKED:
            result["confirmation_required"] = False
            return result

        contract = GoalContract.from_proposal(
            proposal,
            goal_id=f"goal-{uuid.uuid4().hex}",
        )
        if previous.current_task is not None:
            try:
                self.monitor.retire(previous.current_task.publication_id)
            except Exception as error:
                raise RuntimeError(
                    f"Could not retire the paused monitor task: {error}"
                ) from error
        self.controller.stage_goal(contract)
        if previous.goal is not None:
            # Display sessions fence delayed writes and make COMPLETE terminal
            # only for one high-level goal.  Every replacement goal therefore
            # receives a fresh session before its first PLANNING publication.
            self._rotate_display_session()
        self._pending_goal = contract
        result.update(
            {
                "goal_contract": contract.to_dict(),
                "confirmation_required": True,
                "confirmation_notice": (
                    "Confirm the high-level goal and constraints. During execution, "
                    "PrefMem will replan after every terminal task observation and "
                    "publish only one current task at a time."
                ),
            }
        )
        return result

    def confirm_goal(
        self,
        *,
        goal_id: str,
        revision: int,
        confirmed: bool,
    ) -> dict[str, Any]:
        """Start execution with a fresh Planner call after exact confirmation."""

        self._ensure_running()
        goal = self._pending_goal
        if goal is None:
            raise RuntimeError("there is no goal awaiting confirmation")
        if (
            not isinstance(goal_id, str)
            or type(revision) is not int
            or goal.goal_id != goal_id
            or goal.revision != revision
        ):
            raise ValueError("confirmation does not match the staged goal and revision")
        if type(confirmed) is not bool:
            raise TypeError("confirmed must be a boolean")
        if not confirmed:
            # A rejection is not a failed execution attempt.  Keep the exact
            # proposal staged so HRI can discuss or replace it, and perform no
            # camera capture or Planner call.
            result = self.context_dict()
            result["confirmation_status"] = "DECLINED"
            return result
        frame = self._capture_frame()
        transition = self.controller.confirm_goal(
            goal,
            confirmed=confirmed,
            frame_sequence=frame.sequence,
        )
        # Confirmation moves the controller out of AWAITING_CONFIRMATION even
        # when a later page or model operation fails, so this proposal must not
        # remain available for a second confirmation attempt.
        self._pending_goal = None
        if not self._publish_transition(transition):
            return self.context_dict()
        self._run_planning_cycle(transition.planner_request, frame=frame)
        return self.context_dict()

    def resume_current_task(self) -> dict[str, Any]:
        self._ensure_running()
        frame = self._capture_frame()
        transition = self.controller.resume_after_attention(
            frame_sequence=frame.sequence
        )
        if self._publish_transition(transition) and transition.task_to_publish:
            self.monitor.publish(transition.task_to_publish)
        return self.context_dict()

    def request_replan(
        self,
        operator_guidance: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_running()
        paused_task = self.controller.snapshot.current_task
        if paused_task is not None:
            # Retire the old single-slot monitor publication before entering a
            # new planning cycle.  It must never overlap with a replacement.
            try:
                self.monitor.retire(paused_task.publication_id)
            except Exception as error:
                attention = self.controller.record_system_error(
                    f"Monitor retirement failed: {error}"
                )
                self._publish_transition(attention, best_effort=True)
                self.notifications.put(
                    attention.snapshot.attention_reason
                    or "Monitor retirement failed"
                )
                return self.context_dict()
        frame = self._capture_frame()
        transition = self.controller.request_replan(
            operator_guidance=operator_guidance,
            frame_sequence=frame.sequence,
        )
        if self._publish_transition(transition):
            self._run_planning_cycle(transition.planner_request, frame=frame)
        return self.context_dict()

    def check_timeout(self) -> bool:
        transition = self.controller.check_timeout()
        if transition is None:
            return False
        if transition.snapshot.current_task is not None:
            self.monitor.retire(transition.snapshot.current_task.publication_id)
        self._publish_transition(transition)
        self.notifications.put(transition.snapshot.attention_reason or "Needs attention")
        return True

    def context_dict(self) -> dict[str, Any]:
        snapshot = self.controller.snapshot
        return {
            "state": snapshot.state.value,
            "goal": None if snapshot.goal is None else snapshot.goal.to_dict(),
            "cycle_id": snapshot.cycle_id,
            "current_task": (
                None
                if snapshot.current_task is None
                else snapshot.current_task.to_dict()
            ),
            "execution_history": [
                record.to_dict() for record in snapshot.execution_history
            ],
            "attention_kind": (
                None
                if snapshot.attention_kind is None
                else snapshot.attention_kind.value
            ),
            "attention_reason": snapshot.attention_reason,
            "latest_observation": snapshot.latest_observation,
            "emergency_latched": self.emergency.latched,
        }

    def context_json(self) -> str:
        return json.dumps(self.context_dict(), ensure_ascii=False, default=str)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.monitor.stop()
        except Exception:
            # Closing the task surface below is still useful if a third-party
            # monitor implementation cannot stop cleanly.
            pass
        snapshot = self.controller.snapshot
        if snapshot.state in {
            RecedingControllerState.PLANNING,
            RecedingControllerState.EXECUTING,
            RecedingControllerState.FINAL_VALIDATION,
            RecedingControllerState.NEEDS_ATTENTION,
        }:
            # Do not leave an executable instruction on the page after its
            # runtime and monitor have gone away.  Session ownership keeps
            # this shutdown from clearing a newer process's display.
            try:
                self.publisher.reset(session_id=self.session_id)
            except Exception:
                # Shutdown is already in progress; a dead camera page must
                # not prevent process exit.  COMPLETE and EMERGENCY displays
                # are intentionally retained and never reach this branch.
                pass

    def _capture_frame(self) -> CapturedFrame:
        frame = self.frame_source()
        if not isinstance(frame, CapturedFrame):
            raise TypeError("frame_source must return CapturedFrame")
        return frame

    def _run_planning_cycle(
        self,
        request: PlannerCycleRequest | None,
        *,
        frame: CapturedFrame | None = None,
    ) -> None:
        if request is None or self._closed or self.emergency.latched:
            return
        with self._planning_lock:
            if self._closed or self.emergency.latched:
                return
            # A delayed callback must never run whichever planning request is
            # current now.  It may act only on the exact request that caused
            # this call.
            if self.controller.snapshot.pending_request != request:
                return
            try:
                if frame is None:
                    frame = self._capture_frame()
                request = self.controller.bind_planning_frame(frame.sequence)
                decision = self.planner.plan_cycle(request, frame.image_block)
                if self.emergency.latched:
                    return
                transition = self.controller.apply_planner_decision(
                    request,
                    decision,
                )
                if not self._publish_transition(transition):
                    return
                if transition.task_to_publish is not None:
                    try:
                        self.monitor.publish(transition.task_to_publish)
                    except Exception as error:
                        attention = self.controller.record_system_error(
                            f"Monitor publication failed: {error}"
                        )
                        self._publish_transition(attention, best_effort=True)
                        self.notifications.put(
                            attention.snapshot.attention_reason
                            or "Monitor publication failed"
                        )
            except Exception as error:
                if self.emergency.latched:
                    return
                attention = self.controller.record_system_error(
                    f"Planner cycle failed: {error}"
                )
                self._publish_transition(attention, best_effort=True)
                self.notifications.put(
                    attention.snapshot.attention_reason or "Planner cycle failed"
                )

    def _on_monitor_assessment(self, assessment: MonitorAssessment) -> None:
        """Monitor callback; controller decides whether one report is terminal."""

        if self._closed or self.emergency.latched:
            return
        transition = self.controller.record_assessment(assessment)
        if transition.result in {
            RecedingResult.IGNORED_STALE,
            RecedingResult.IGNORED_INVALID,
        }:
            return
        if transition.result is RecedingResult.REPLAN_REQUESTED:
            self.monitor.retire(assessment.publication_id)
        self._publish_transition(transition, best_effort=True)
        if transition.result is RecedingResult.REPLAN_REQUESTED:
            self._run_planning_cycle(transition.planner_request)
        elif transition.result is RecedingResult.COMPLETE:
            self.monitor.retire(assessment.publication_id)
            self.notifications.put("Task Complete.")

    def _on_monitor_error(self, event: MonitorErrorEvent) -> None:
        if self._closed or self.emergency.latched:
            return
        task = self.controller.snapshot.current_task
        if task is None or event.publication_id != task.publication_id:
            # Monitor errors are task-scoped.  An error emitted just as a task
            # is retired must not pause its replacement publication.
            return
        attention = self.controller.record_system_error(
            f"{event.kind.value}: {event.message}"
        )
        self._publish_transition(attention, best_effort=True)
        self.notifications.put(
            attention.snapshot.attention_reason or "Monitor system error"
        )

    def _on_emergency_stop(self, event: EmergencyStopEvent) -> None:
        transition = self.controller.emergency_stop(event.reason)
        # The physical/placeholder stop hook ran before this callback.  Page I/O
        # is consequently best-effort and can never delay the stop action.
        self._publish_transition(transition, best_effort=True)
        self.notifications.put(f"EMERGENCY STOP: {event.reason}")
        # Stop admission and the monitor loop immediately as part of the
        # orderly process-exit handoff.  ``MonitorService.stop`` is explicitly
        # safe when called from its own worker thread.
        try:
            self.monitor.stop()
        except Exception:
            # The emergency latch and shutdown event remain authoritative even
            # if a third-party monitor implementation cannot be stopped cleanly.
            pass

    def _publish_transition(
        self,
        transition: RecedingTransition,
        *,
        best_effort: bool = False,
    ) -> bool:
        snapshot = transition.snapshot
        if snapshot.goal is None:
            return True
        try:
            display = self._display_from_snapshot(snapshot)
            self.publisher.publish(display)
            return True
        except TaskPublisherError as error:
            if best_effort or snapshot.state is RecedingControllerState.EMERGENCY_STOPPED:
                return False
            attention = self.controller.record_system_error(
                f"Task display publication failed: {error}"
            )
            self.notifications.put(
                attention.snapshot.attention_reason
                or "Task display publication failed"
            )
            return False

    def _display_from_snapshot(
        self,
        snapshot: RecedingControllerSnapshot,
    ) -> ControllerDisplay:
        assert snapshot.goal is not None
        with self._display_lock:
            self._display_sequence += 1
            sequence = self._display_sequence
            session_id = self.session_id
        task = snapshot.current_task
        mapping = {
            RecedingControllerState.PLANNING: DisplayState.PLANNING,
            RecedingControllerState.EXECUTING: DisplayState.ACTIVE,
            RecedingControllerState.FINAL_VALIDATION: DisplayState.FINAL_VALIDATION,
            RecedingControllerState.NEEDS_ATTENTION: DisplayState.NEEDS_ATTENTION,
            RecedingControllerState.COMPLETE: DisplayState.COMPLETE,
            RecedingControllerState.EMERGENCY_STOPPED: DisplayState.EMERGENCY_STOPPED,
        }
        state = mapping.get(snapshot.state, DisplayState.PLANNING)
        expected = (
            ()
            if task is None
            else tuple(item.description for item in task.expected_observation)
        )
        instruction = (
            task.instruction
            if task is not None
            and state in {DisplayState.ACTIVE, DisplayState.NEEDS_ATTENTION}
            else None
        )
        publication_id = None if task is None else task.publication_id
        message = snapshot.attention_reason
        if state is DisplayState.PLANNING:
            message = "Planning the next task; hold position."
        elif state is DisplayState.COMPLETE:
            message = "Task complete."
        elif state is DisplayState.EMERGENCY_STOPPED:
            message = snapshot.attention_reason or "Emergency stop requested."
        return ControllerDisplay(
            session_id=session_id,
            sequence=sequence,
            state=state,
            goal=snapshot.goal.goal,
            cycle=snapshot.cycle_id or None,
            publication_id=publication_id,
            instruction=instruction,
            expected_observation=expected,
            message=message,
            monitor_observation=snapshot.latest_observation,
        )

    def _rotate_display_session(self) -> None:
        with self._display_lock:
            self.session_id = uuid.uuid4().hex
            self._display_sequence = 0

    def _ensure_running(self) -> None:
        if self._closed:
            raise RuntimeClosedError("PrefMem runtime is closed")
        if self.emergency.latched:
            raise RuntimeClosedError("PrefMem emergency stop is latched")


def build_runtime(args):
    """Build production dependencies while preserving HRI's shared metrics."""

    from prefmem.agents.hri import HRI_Agent

    hri = HRI_Agent(model_config=args.model_config, args=args)
    runtime = PrefMemRuntime(
        hri.planner_agent,
        camera_base_url=args.camera_base_url,
        monitor_min_interval=args.monitor_min_interval,
        model_base_url="http://localhost:8000/v1",
        metrics=hri.metrics,
        success_confirmations=args.success_confirmations,
        success_stability_seconds=args.success_stability_seconds,
        failure_confirmations=args.failure_confirmations,
        ongoing_timeout_seconds=args.monitor_timeout,
        max_cycles=args.max_planning_cycles,
    )
    hri.attach_runtime(runtime)
    return runtime, hri
