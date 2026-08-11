"""Composition and orchestration for PrefMem's receding-horizon runtime."""

from __future__ import annotations

import json
import queue
import re
import threading
import uuid
from typing import Any, Callable, Sequence

from prefmem.agents.monitor import (
    CapturedFrame,
    HTTPFrameSource,
    MonitorErrorEvent,
    MonitorService,
    MonitorTelemetryEvent,
    MonitorTelemetryKind,
)
from prefmem.agents.validator import (
    ValidatorErrorEvent,
    ValidatorService,
)
from prefmem.contracts import (
    CriterionState,
    GoalContract,
    GoalProposal,
    MonitorAssessment,
    PlanStatus,
    PlannerCycleRequest,
    PublishedTask,
    TaskPhase,
    TaskStatus,
    ValidationAssessment,
    ValidationContract,
    ValidationReport,
    ValidationStatus,
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
    """Own Planner cycles, publication, execution, observation, and shutdown.

    Planner, Monitor, and Validator remain model boundaries.  This runtime is
    the only component allowed to append execution history, select the first
    task from a candidate horizon, publish it, or request the next cycle.
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
        validator: ValidatorService | None = None,
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
        executor: Any | None = None,
        owned_resources: Sequence[Any] = (),
        event_sink: Callable[[str, Any], None] | None = None,
    ) -> None:
        if not hasattr(planner, "preview") or not hasattr(planner, "plan_cycle"):
            raise TypeError("planner must expose preview() and plan_cycle()")
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self._event_sink = event_sink
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
        self.executor = executor
        self._owned_resources = tuple(owned_resources)
        if executor is not None:
            for method_name in ("publish", "retire", "stop", "is_settled"):
                if not callable(getattr(executor, method_name, None)):
                    raise TypeError(
                        f"executor must expose callable {method_name}()"
                    )
            executor.on_event = self._on_execution_event
            executor.on_scene_change = self._on_execution_scene_change
        self.notifications: queue.SimpleQueue[Any] = queue.SimpleQueue()
        # High-rate Monitor telemetry is isolated from user/action
        # notifications. The bounded, drop-oldest queue guarantees that a
        # slow console can never block observation or robot control.
        self.monitor_events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=256)
        self._monitor_event_lock = threading.Lock()
        self._monitor_elapsed: dict[tuple[str, int | None], float] = {}
        self._planning_lock = threading.Lock()
        self._display_lock = threading.Lock()
        self._display_sequence = 0
        self._closed = False
        self._pending_goal: GoalContract | None = None
        self._validation_contract: ValidationContract | None = None
        self._latest_validation_report: ValidationReport | None = None
        self._latest_validation_publication_id: str | None = None
        self._validation_notification_signature: tuple[Any, ...] | None = None

        self.emergency = EmergencyStopCoordinator(
            emergency_stop,
            on_stop=self._on_emergency_stop,
        )
        self.monitor = monitor or MonitorService(
            self._on_monitor_assessment,
            on_error=self._on_monitor_error,
            on_event=self._on_monitor_telemetry,
            emergency=self.emergency,
            snapshot_url=snapshot_url,
            frame_source=self.frame_source,
            min_interval_seconds=monitor_min_interval,
            model_name=model_name,
            model_base_url=model_base_url,
            metrics=metrics,
        )
        self.validator = validator or ValidatorService(
            self._on_validator_assessment,
            on_error=self._on_validator_error,
            emergency=self.emergency,
            snapshot_url=snapshot_url,
            frame_source=self.frame_source,
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
        self._emit_event(
            "RUNTIME_INITIALIZED",
            {
                "session_id": self.session_id,
                "executor": (
                    None if self.executor is None else type(self.executor).__name__
                ),
            },
        )

    @property
    def shutdown_event(self) -> threading.Event:
        return self.emergency.shutdown_event

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_goal(self) -> GoalContract | None:
        return self._pending_goal

    def _emit_event(self, kind: str, payload: Any) -> None:
        """Send one lossless runtime event to an optional durable sink.

        Production leaves the sink unset. Experiment construction supplies a
        fail-closed append-only journal; callback errors are intentionally not
        swallowed because an unrecorded experimental action is invalid.
        """

        if self._event_sink is not None:
            self._event_sink(kind, payload)

    def request_goal_preview(
        self,
        clarified_goal: str,
        *,
        constraints: Sequence[str] = (),
        operator_guidance: str | None = None,
    ) -> dict[str, Any]:
        """Ask Planner for the nominal strategy and stage its frozen goal."""

        self._ensure_running()
        self._emit_event(
            "GOAL_PREVIEW_REQUESTED",
            {
                "clarified_goal": clarified_goal,
                "constraints": list(constraints),
                "operator_guidance": operator_guidance,
            },
        )
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
            self._emit_event("GOAL_PREVIEW_BLOCKED", result)
            return result

        contract = GoalContract.from_proposal(
            proposal,
            goal_id=f"goal-{uuid.uuid4().hex}",
        )
        if previous.current_task is not None:
            try:
                self._retire_task(previous.current_task)
            except Exception as error:
                raise RuntimeError(
                    f"Could not retire the paused observation task: {error}"
                ) from error
        self._clear_validation_state()
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
        self._emit_event("GOAL_PREVIEW_STAGED", result)
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
        self._emit_event(
            "GOAL_CONFIRMATION_RECEIVED",
            {
                "goal_id": goal_id,
                "revision": revision,
                "confirmed": confirmed,
            },
        )
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
            self._emit_event("GOAL_CONFIRMATION_DECLINED", result)
            return result
        confirmation_frame = self._capture_frame()
        try:
            validation_contract = self.validator.compile_contract(
                goal,
                confirmation_frame,
            )
        except Exception as error:
            # Checklist freezing is part of confirmation.  If it fails, no
            # Planner cycle or executable publication is authorized and this
            # exact staged goal remains confirmable for a retry.
            raise RuntimeError(
                f"Final validation checklist compilation failed: {error}"
            ) from error
        if not isinstance(validation_contract, ValidationContract):
            raise TypeError(
                "validator.compile_contract() must return ValidationContract"
            )
        if validation_contract.goal_contract != goal:
            raise ValueError(
                "validator returned a checklist for a different frozen goal"
            )

        # Checklist compilation may be a long model call.  Bind the first MPC
        # cycle to a fresh post-compilation frame, not the older checklist frame.
        frame = self._capture_frame()
        transition = self.controller.confirm_goal(
            goal,
            confirmed=confirmed,
            frame_sequence=frame.sequence,
        )
        self._validation_contract = validation_contract
        self._emit_event("VALIDATION_CONTRACT_FROZEN", validation_contract)
        self._latest_validation_report = None
        self._latest_validation_publication_id = None
        self._validation_notification_signature = None
        # Confirmation moves the controller out of AWAITING_CONFIRMATION even
        # when a later page or model operation fails, so this proposal must not
        # remain available for a second confirmation attempt.
        self._pending_goal = None
        if not self._publish_transition(transition):
            return self.context_dict()
        self._run_planning_cycle(transition.planner_request, frame=frame)
        result = self.context_dict()
        self._emit_event("GOAL_CONFIRMATION_ACCEPTED", result)
        return result

    def resume_current_task(self) -> dict[str, Any]:
        self._ensure_running()
        frame = self._capture_frame()
        transition = self.controller.resume_after_attention(
            frame_sequence=frame.sequence
        )
        if self._publish_transition(transition) and transition.task_to_publish:
            try:
                self._publish_task(transition.task_to_publish)
            except Exception as error:
                attention = self.controller.record_system_error(
                    f"Observation service publication failed: {error}"
                )
                self._publish_transition(attention, best_effort=True)
                self.notifications.put(
                    attention.snapshot.attention_reason
                    or "Observation service publication failed"
                )
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
                self._retire_task(paused_task)
            except Exception as error:
                attention = self.controller.record_system_error(
                    f"Observation service retirement failed: {error}"
                )
                self._publish_transition(attention, best_effort=True)
                self.notifications.put(
                    attention.snapshot.attention_reason
                    or "Observation service retirement failed"
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
        active_task = self.controller.snapshot.current_task
        if (
            self.executor is not None
            and active_task is not None
            and active_task.phase is TaskPhase.STEP
            and not self.executor.is_settled(active_task.publication_id)
        ):
            # Compiler, detector, and motion-controller calls have their own
            # bounded failures. Do not apply the visual no-progress timeout to
            # an executor that is still preparing or moving.
            return False
        transition = self.controller.check_timeout()
        if transition is None:
            return False
        timed_out_task = transition.snapshot.current_task
        if timed_out_task is not None:
            self._retire_task(timed_out_task)
        self._publish_transition(transition)
        if (
            timed_out_task is not None
            and timed_out_task.phase is TaskPhase.FINAL_VALIDATION
            and self._latest_validation_report is not None
            and self._latest_validation_report.status
            is ValidationStatus.NEEDS_EVIDENCE
        ):
            self._notify_validation_report(
                self._latest_validation_report,
                publication_id=(
                    self._latest_validation_publication_id
                    or timed_out_task.publication_id
                ),
                next_action=(
                    "Final validation is paused; move only the camera to show "
                    "the requested views, keep all scene objects unchanged, "
                    "then resume."
                ),
                force=True,
            )
        else:
            self.notifications.put(
                transition.snapshot.attention_reason or "Needs attention"
            )
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
            "validation": self._public_validation_context(),
            "emergency_latched": self.emergency.latched,
        }

    def context_json(self) -> str:
        return json.dumps(self.context_dict(), ensure_ascii=False, default=str)

    def close(self) -> None:
        if self._closed:
            return
        self._emit_event("RUNTIME_CLOSING", self.context_dict())
        self._closed = True
        for service in (self.executor, self.monitor, self.validator):
            if service is None:
                continue
            try:
                service.stop()
            except Exception:
                # Closing the task surface below is still useful if a
                # third-party observation service cannot stop cleanly.
                pass
        for resource in self._owned_resources:
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
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
        self._emit_event(
            "FRAME_CAPTURED",
            {
                "sequence": frame.sequence,
                "observed_at": frame.observed_at,
                "image_block": frame.image_block,
            },
        )
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
                self._emit_event("PLANNER_CYCLE_REQUESTED", request)
                decision = self.planner.plan_cycle(request, frame.image_block)
                self._emit_event("PLANNER_CYCLE_DECISION", decision)
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
                        self._publish_task(transition.task_to_publish)
                    except Exception as error:
                        attention = self.controller.record_system_error(
                            f"Observation service publication failed: {error}"
                        )
                        self._publish_transition(attention, best_effort=True)
                        self.notifications.put(
                            attention.snapshot.attention_reason
                            or "Observation service publication failed"
                        )
            except Exception as error:
                self._emit_event(
                    "PLANNER_CYCLE_ERROR",
                    {"type": type(error).__name__, "message": str(error)},
                )
                if self.emergency.latched:
                    return
                attention = self.controller.record_system_error(
                    f"Planner cycle failed: {error}"
                )
                self._publish_transition(attention, best_effort=True)
                self.notifications.put(
                    attention.snapshot.attention_reason or "Planner cycle failed"
                )

    def _queue_monitor_event(self, payload: dict[str, Any]) -> None:
        """Publish best-effort telemetry without blocking an agent thread."""

        event = dict(payload)
        event["kind"] = "MONITOR_EVENT"
        try:
            self.monitor_events.put_nowait(event)
            return
        except queue.Full:
            pass
        try:
            self.monitor_events.get_nowait()
        except queue.Empty:
            pass
        try:
            self.monitor_events.put_nowait(event)
        except queue.Full:
            # A racing producer refilled the queue. Telemetry may be dropped;
            # controller and emergency paths remain authoritative.
            pass

    def _on_monitor_telemetry(self, event: MonitorTelemetryEvent) -> None:
        self._emit_event("MONITOR_TELEMETRY", event)
        if not isinstance(event, MonitorTelemetryEvent):
            return
        if event.kind is MonitorTelemetryKind.INFERENCE_COMPLETED:
            with self._monitor_event_lock:
                if event.elapsed_seconds is not None:
                    self._monitor_elapsed[
                        (event.publication_id, event.frame_sequence)
                    ] = event.elapsed_seconds
        self._queue_monitor_event(
            {
                "event": event.kind.value,
                "publication_id": event.publication_id,
                "frame_sequence": event.frame_sequence,
                "observed_at": event.observed_at,
                "elapsed_seconds": event.elapsed_seconds,
                "message": event.message,
            }
        )

    def _take_monitor_elapsed(self, assessment: MonitorAssessment) -> float | None:
        key = (assessment.publication_id or "", assessment.frame_sequence)
        with self._monitor_event_lock:
            return self._monitor_elapsed.pop(key, None)

    def _executor_state_for(self, publication_id: str | None) -> str | None:
        if self.executor is None or publication_id is None:
            return None
        state_reader = getattr(self.executor, "state", None)
        if callable(state_reader):
            try:
                state = state_reader(publication_id)
            except Exception:
                state = None
            if state is not None:
                return str(getattr(state, "value", state))
        try:
            return (
                "SETTLED"
                if self.executor.is_settled(publication_id)
                else "ACTIVE"
            )
        except Exception:
            return "UNKNOWN"

    def _emit_monitor_assessment_event(
        self,
        assessment: MonitorAssessment,
        *,
        disposition: str,
        task: PublishedTask | None,
        snapshot: RecedingControllerSnapshot,
    ) -> None:
        descriptions = {
            criterion.criterion_id: criterion.description
            for criterion in (() if task is None else task.expected_observation)
        }
        success_required = getattr(
            self.controller,
            "success_confirmations",
            getattr(self.controller, "_success_confirmations", None),
        )
        failure_required = getattr(
            self.controller,
            "failure_confirmations",
            getattr(self.controller, "_failure_confirmations", None),
        )
        success_count = snapshot.consecutive_successes
        failure_count = snapshot.consecutive_failures
        if assessment.task_status is TaskStatus.SUCCESS and disposition in {
            RecedingResult.REPLAN_REQUESTED.value,
            RecedingResult.COMPLETE.value,
        }:
            success_count = success_required
        if assessment.task_status is TaskStatus.FAIL and disposition in {
            RecedingResult.REPLAN_REQUESTED.value,
            RecedingResult.NEEDS_ATTENTION.value,
        }:
            failure_count = failure_required
        self._queue_monitor_event(
            {
                "event": "ASSESSMENT",
                "publication_id": assessment.publication_id,
                "cycle_id": None if task is None else task.cycle_id,
                "step_id": assessment.step_id,
                "frame_sequence": assessment.frame_sequence,
                "elapsed_seconds": self._take_monitor_elapsed(assessment),
                "task_status": assessment.task_status.value,
                "criteria": [
                    {
                        "id": criterion.criterion_id,
                        "description": descriptions.get(criterion.criterion_id),
                        "state": criterion.state.value,
                    }
                    for criterion in assessment.criteria
                ],
                "observation": assessment.observation,
                "failure": (
                    None
                    if assessment.failure is None
                    else assessment.failure.to_dict()
                ),
                "disposition": disposition,
                "executor_state": self._executor_state_for(
                    assessment.publication_id
                ),
                "success_confirmation": {
                    "count": success_count,
                    "required": success_required,
                },
                "failure_confirmation": {
                    "count": failure_count,
                    "required": failure_required,
                },
            }
        )

    def _on_monitor_assessment(self, assessment: MonitorAssessment) -> None:
        """Monitor callback; controller decides whether one report is terminal."""

        self._emit_event("MONITOR_ASSESSMENT", assessment)
        if self._closed or self.emergency.latched:
            return
        before = self.controller.snapshot
        current_task = before.current_task
        if (
            current_task is None
            or current_task.phase is not TaskPhase.STEP
            or assessment.publication_id != current_task.publication_id
        ):
            # Per-step Monitor has no authority during final validation.
            self._emit_monitor_assessment_event(
                assessment,
                disposition=RecedingResult.IGNORED_STALE.value,
                task=current_task,
                snapshot=before,
            )
            return
        if (
            self.executor is not None
            and assessment.task_status is not TaskStatus.ONGOING
            and not self.executor.is_settled(current_task.publication_id)
        ):
            # Transient views while the robot is moving cannot finish or fail
            # a task. The Monitor remains completion authority once motion has
            # settled and the camera has an unobstructed post-state.
            self._emit_monitor_assessment_event(
                assessment,
                disposition="IGNORED_EXECUTOR_ACTIVE",
                task=current_task,
                snapshot=before,
            )
            return
        transition = self.controller.record_assessment(assessment)
        self._emit_monitor_assessment_event(
            assessment,
            disposition=transition.result.value,
            task=current_task,
            snapshot=transition.snapshot,
        )
        if transition.result in {
            RecedingResult.IGNORED_STALE,
            RecedingResult.IGNORED_INVALID,
        }:
            return
        if transition.result is RecedingResult.REPLAN_REQUESTED:
            self._retire_task(current_task)
        self._publish_transition(transition, best_effort=True)
        if transition.result is RecedingResult.REPLAN_REQUESTED:
            self._run_planning_cycle(transition.planner_request)

    def _on_monitor_error(self, event: MonitorErrorEvent) -> None:
        self._emit_event("MONITOR_ERROR", event)
        if self._closed or self.emergency.latched:
            return
        task = self.controller.snapshot.current_task
        if (
            task is None
            or task.phase is not TaskPhase.STEP
            or event.publication_id != task.publication_id
        ):
            # Monitor errors are task-scoped.  An error emitted just as a task
            # is retired must not pause its replacement publication.
            return
        with self._monitor_event_lock:
            stale_keys = [
                key for key in self._monitor_elapsed if key[0] == task.publication_id
            ]
            for key in stale_keys:
                self._monitor_elapsed.pop(key, None)
        attention = self.controller.record_system_error(
            f"{event.kind.value}: {event.message}"
        )
        self._publish_transition(attention, best_effort=True)
        self.notifications.put(
            attention.snapshot.attention_reason or "Monitor system error"
        )

    def _on_validator_assessment(
        self,
        assessment: ValidationAssessment,
    ) -> None:
        """Validator callback for the frozen high-level completion contract."""

        self._emit_event("VALIDATOR_ASSESSMENT", assessment)
        if self._closed or self.emergency.latched:
            return
        if not isinstance(assessment, ValidationAssessment):
            return
        snapshot = self.controller.snapshot
        task = snapshot.current_task
        contract = self._validation_contract
        if (
            snapshot.state is not RecedingControllerState.FINAL_VALIDATION
            or task is None
            or task.phase is not TaskPhase.FINAL_VALIDATION
            or assessment.publication_id != task.publication_id
            or contract is None
            or assessment.validation_id != contract.validation_id
            or assessment.goal_id != contract.goal_contract.goal_id
            or assessment.goal_revision != contract.goal_contract.revision
        ):
            return

        try:
            report = ValidationReport.from_assessment(contract, assessment)
            expected_labels = tuple(
                criterion.description for criterion in task.expected_observation
            )
            report_labels = tuple(item.label for item in report.checklist)
            if report_labels != expected_labels:
                raise ValueError(
                    "Validator broad checklist does not match the published "
                    "final observations"
                )
            controller_assessment = self._validation_controller_assessment(
                task,
                assessment,
                report,
            )
            transition = self.controller.record_assessment(
                controller_assessment
            )
        except Exception as error:
            try:
                self.validator.retire(task.publication_id)
            except Exception:
                pass
            attention = self.controller.record_system_error(
                f"Validator assessment rejected: {error}"
            )
            self._publish_transition(attention, best_effort=True)
            self.notifications.put(
                attention.snapshot.attention_reason
                or "Validator assessment rejected"
            )
            return

        if transition.result in {
            RecedingResult.IGNORED_STALE,
            RecedingResult.IGNORED_INVALID,
        }:
            return

        # Only an assessment accepted by the authoritative controller becomes
        # the latest user-facing report.  A stale frame cannot overwrite it.
        self._latest_validation_report = report
        self._latest_validation_publication_id = task.publication_id

        if transition.result in {
            RecedingResult.REPLAN_REQUESTED,
            RecedingResult.COMPLETE,
            RecedingResult.NEEDS_ATTENTION,
        }:
            self.validator.retire(task.publication_id)
        self._publish_transition(transition, best_effort=True)

        if transition.result is RecedingResult.REPLAN_REQUESTED:
            self._notify_validation_report(
                report,
                publication_id=task.publication_id,
                next_action=(
                    "PrefMem is replanning from the unmet final outcome while "
                    "keeping the confirmed high-level goal unchanged."
                ),
                force=True,
            )
            self._run_planning_cycle(transition.planner_request)
        elif transition.result is RecedingResult.COMPLETE:
            self._notify_validation_report(
                report,
                publication_id=task.publication_id,
                force=True,
            )
        elif transition.result is RecedingResult.NEEDS_ATTENTION:
            self._notify_validation_report(
                report,
                publication_id=task.publication_id,
                next_action=(
                    transition.snapshot.attention_reason
                    or "Human review is required before continuing."
                ),
                force=True,
            )
        elif report.status is ValidationStatus.NEEDS_EVIDENCE:
            self._notify_validation_report(
                report,
                publication_id=task.publication_id,
                next_action=(
                    "Move only the camera to provide the requested views; "
                    "keep all scene objects unchanged."
                ),
            )

    def _on_validator_error(self, event: ValidatorErrorEvent) -> None:
        self._emit_event("VALIDATOR_ERROR", event)
        if self._closed or self.emergency.latched:
            return
        task = self.controller.snapshot.current_task
        if (
            task is None
            or task.phase is not TaskPhase.FINAL_VALIDATION
            or event.publication_id != task.publication_id
        ):
            return
        try:
            self.validator.retire(task.publication_id)
        except Exception:
            pass
        attention = self.controller.record_system_error(
            f"Validator {event.kind.value}: {event.message}"
        )
        self._publish_transition(attention, best_effort=True)
        self.notifications.put(
            attention.snapshot.attention_reason or "Validator system error"
        )

    def _on_emergency_stop(self, event: EmergencyStopEvent) -> None:
        self._emit_event("EMERGENCY_STOP", event)
        transition = self.controller.emergency_stop(event.reason)
        # The physical/placeholder stop hook ran before this callback.  Page I/O
        # is consequently best-effort and can never delay the stop action.
        self._publish_transition(transition, best_effort=True)
        self.notifications.put(f"EMERGENCY STOP: {event.reason}")
        if self.executor is not None:
            try:
                self.executor.emergency_stop()
            except Exception:
                pass
        # Stop both observation loops immediately as part of the orderly
        # process-exit handoff.  Their stop methods are safe from worker threads.
        for service in (self.monitor, self.validator):
            try:
                service.stop()
            except Exception:
                # The emergency latch and shutdown event remain authoritative
                # even if a third-party service cannot be stopped cleanly.
                pass

    def _publish_task(self, task: PublishedTask) -> None:
        """Route one publication to its phase-specific observation/execution path."""

        self._emit_event("TASK_PUBLICATION_ROUTED", task)
        if task.phase is TaskPhase.STEP:
            self._queue_monitor_event(
                {
                    "event": "TASK_PUBLISHED",
                    "publication_id": task.publication_id,
                    "cycle_id": task.cycle_id,
                    "step_id": task.step_id,
                    "instruction": task.instruction,
                    "criteria": [
                        {
                            "id": criterion.criterion_id,
                            "description": criterion.description,
                        }
                        for criterion in task.expected_observation
                    ],
                }
            )
            self.monitor.publish(task)
            if self.executor is not None:
                goal = self.controller.snapshot.goal
                scope = None if goal is None else goal.dynamic_object_scope
                try:
                    self.executor.publish(task, scope)
                except Exception:
                    self._retire_task(task)
                    raise
            return
        contract = self._validation_contract
        if contract is None:
            raise RuntimeError(
                "final validation has no checklist frozen at confirmation"
            )
        if (
            contract.goal_contract.goal_id != task.plan_id
            or contract.goal_contract.revision != task.revision
        ):
            raise RuntimeError(
                "final validation checklist does not match the current goal"
            )
        labels = tuple(item.label for item in contract.broad_items)
        expected = tuple(
            item.description for item in task.expected_observation
        )
        if labels != expected:
            raise RuntimeError(
                "final validation checklist does not match the published outcomes"
            )
        self._latest_validation_report = None
        self._latest_validation_publication_id = None
        self._validation_notification_signature = None
        self.validator.publish(task, contract)

    def _retire_task(self, task: PublishedTask) -> bool:
        self._emit_event("TASK_RETIREMENT_REQUESTED", task)
        if task.phase is TaskPhase.FINAL_VALIDATION:
            return bool(self.validator.retire(task.publication_id))
        retired = bool(self.monitor.retire(task.publication_id))
        if self.executor is not None:
            retired = bool(self.executor.retire(task.publication_id)) or retired
        with self._monitor_event_lock:
            stale_keys = [
                key for key in self._monitor_elapsed if key[0] == task.publication_id
            ]
            for key in stale_keys:
                self._monitor_elapsed.pop(key, None)
        if retired:
            self._emit_event(
                "TASK_RETIRED",
                {"publication_id": task.publication_id},
            )
            self._queue_monitor_event(
                {
                    "event": "TASK_RETIRED",
                    "publication_id": task.publication_id,
                    "cycle_id": task.cycle_id,
                    "step_id": task.step_id,
                }
            )
        return retired

    def _on_execution_event(self, event: Any) -> None:
        """Pause on executor faults; ordinary completion remains Monitor-owned."""

        self._emit_event("EXECUTOR_EVENT", event)
        from prefmem.execution.contracts import ExecutionEvent, ExecutionState

        if (
            self._closed
            or self.emergency.latched
            or not isinstance(event, ExecutionEvent)
        ):
            return
        snapshot = self.controller.snapshot
        task = snapshot.current_task
        if (
            task is None
            or task.phase is not TaskPhase.STEP
            or task.publication_id != event.publication_id
        ):
            return
        if event.state is ExecutionState.FAULT:
            try:
                self._retire_task(task)
            except Exception:
                pass
            attention = self.controller.record_system_error(event.message)
            self._publish_transition(attention, best_effort=True)
            self.notifications.put(event.message)
        elif event.state in {
            ExecutionState.PREPARING,
            ExecutionState.RUNNING,
            ExecutionState.SETTLED,
        }:
            self.controller.record_execution_progress(event.publication_id)

    def _on_execution_scene_change(self, event: Any) -> None:
        """Safely replan an open-set goal after the controller has stopped."""

        self._emit_event("EXECUTOR_SCENE_CHANGE", event)
        from prefmem.execution.contracts import SceneChangeEvent

        if (
            self._closed
            or self.emergency.latched
            or not isinstance(event, SceneChangeEvent)
        ):
            return
        snapshot = self.controller.snapshot
        task = snapshot.current_task
        if (
            snapshot.state is not RecedingControllerState.EXECUTING
            or task is None
            or task.publication_id != event.publication_id
        ):
            return
        try:
            self._retire_task(task)
        except Exception:
            pass
        attention = self.controller.record_system_error(event.description)
        self._publish_transition(attention, best_effort=True)
        self.notifications.put(
            f"Scene changed; safely stopped and replanning: {event.description}"
        )
        try:
            self.request_replan(
                operator_guidance=(
                    "The workspace gained a matching object during execution. "
                    "Re-evaluate every object in the dynamic goal scope and "
                    "choose the next safe task from the current frame."
                )
            )
        except Exception as error:
            self.notifications.put(f"Automatic scene-change replan failed: {error}")

    def _clear_validation_state(self) -> None:
        self._validation_contract = None
        self._latest_validation_report = None
        self._latest_validation_publication_id = None
        self._validation_notification_signature = None

    @staticmethod
    def _validation_text_fragment(value: str) -> str:
        """Turn a model sentence into a clause safe for one failure sentence."""

        compact = " ".join(value.split())
        return re.sub(r"[.!?]+", "", compact).strip(" ;,:—-")

    def _validation_controller_assessment(
        self,
        task: PublishedTask,
        assessment: ValidationAssessment,
        report: ValidationReport,
    ) -> MonitorAssessment:
        """Adapt a derived broad report to the existing temporal controller."""

        if report.status is ValidationStatus.COMPLETE:
            task_status = "SUCCESS"
            failure = None
        elif report.status is ValidationStatus.INCOMPLETE:
            task_status = "FAIL"
            unmet = [
                (
                    f"{self._validation_text_fragment(item.label)} "
                    f"({self._validation_text_fragment(item.evidence)})"
                )
                for item in report.checklist
                if item.state is CriterionState.NOT_MET
            ]
            details = "; ".join(unmet) or "a required final outcome"
            failure = {
                "kind": "UNEXPECTED",
                "description": (
                    f"Final validation found unmet outcomes: {details}."
                ),
            }
        else:
            task_status = "ONGOING"
            failure = None

        payload = {
            "task_status": task_status,
            "criteria": [
                {
                    "id": expected.criterion_id,
                    "state": result.state.value,
                }
                for expected, result in zip(
                    task.expected_observation,
                    report.checklist,
                    strict=True,
                )
            ],
            "failure": failure,
            "observation": report.summary,
        }
        return MonitorAssessment.from_model_output(
            payload,
            plan_id=task.plan_id,
            revision=task.revision,
            step_id=task.step_id,
            publication_id=assessment.publication_id,
            observed_at=assessment.observed_at,
            frame_sequence=assessment.frame_sequence,
        )

    @staticmethod
    def _public_validation_report(report: ValidationReport) -> dict[str, Any]:
        payload = report.to_dict()
        # Frame-level and detailed evidence stay internal.  HRI receives the
        # deterministic summary plus the exact broad labels and states.
        payload.pop("observation", None)
        payload["checklist"] = [
            {"label": item.label, "state": item.state.value}
            for item in report.checklist
        ]
        return payload

    def _public_validation_context(self) -> dict[str, Any] | None:
        contract = self._validation_contract
        if contract is None:
            return None
        return {
            "validation_id": contract.validation_id,
            "goal_id": contract.goal_contract.goal_id,
            "goal_revision": contract.goal_contract.revision,
            "broad_checklist": [
                item.label for item in contract.broad_items
            ],
            "latest_publication_id": self._latest_validation_publication_id,
            "latest_report": (
                None
                if self._latest_validation_report is None
                else self._public_validation_report(
                    self._latest_validation_report
                )
            ),
        }

    def _notify_validation_report(
        self,
        report: ValidationReport,
        *,
        publication_id: str,
        next_action: str | None = None,
        force: bool = False,
    ) -> None:
        signature = (
            report.validation_id,
            publication_id,
            report.status.value,
            tuple(item.state.value for item in report.checklist),
            report.evidence_requests,
        )
        if not force and signature == self._validation_notification_signature:
            return
        self._validation_notification_signature = signature
        payload = self._public_validation_report(report)
        payload.update(
            {
                "kind": "FINAL_VALIDATION_REPORT",
                "publication_id": publication_id,
            }
        )
        if next_action:
            payload["next_action"] = next_action
        self.notifications.put(payload)

    def _publish_transition(
        self,
        transition: RecedingTransition,
        *,
        best_effort: bool = False,
    ) -> bool:
        self._emit_event("CONTROLLER_TRANSITION", transition)
        snapshot = transition.snapshot
        if snapshot.goal is None:
            return True
        try:
            display = self._display_from_snapshot(snapshot)
            if display is None:
                # A newer controller transition won the display race.
                return False
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
    ) -> ControllerDisplay | None:
        assert snapshot.goal is not None
        with self._display_lock:
            # Controller callbacks and timeout polling run on different
            # threads. Check freshness while serializing wire-sequence
            # allocation so an older ACTIVE snapshot can never receive a
            # higher display sequence than a newer terminal/attention state.
            if snapshot.state_sequence < self.controller.snapshot.state_sequence:
                return None
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
            and state
            in {
                DisplayState.ACTIVE,
                DisplayState.NEEDS_ATTENTION,
            }
            else None
        )
        publication_id = None if task is None else task.publication_id
        message = snapshot.attention_reason
        if state is DisplayState.PLANNING:
            message = "Planning the next task; hold position."
        elif (
            state is DisplayState.ACTIVE
            and snapshot.consecutive_successes > 0
        ):
            count = snapshot.consecutive_successes
            noun = "frame" if count == 1 else "frames"
            message = (
                f"Expected observation detected in {count} monitor {noun}; "
                "confirming that it remains stable."
            )
        elif state is DisplayState.FINAL_VALIDATION:
            # Keep ``instruction`` execution-only for compatibility with the
            # original camera API.  Old and new pages both render ``message``.
            message = (
                None if task is None else task.instruction
            ) or "Keep the scene unchanged during final validation."
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

    provider_matches = getattr(args, "model_provider", None) == args.model_config
    upper_model = getattr(args, "model", None) if provider_matches else None
    upper_model_base_url = (
        getattr(args, "model_base_url", None) if provider_matches else None
    )
    hri = HRI_Agent(
        model_config=args.model_config,
        args=args,
        model_name=upper_model,
        model_base_url=upper_model_base_url,
    )
    executor = None
    frame_source = None
    task_publisher = None
    owned_resources: tuple[Any, ...] = ()
    if getattr(args, "executor", "human") == "mujoco":
        from prefmem.execution.gemma import GemmaExecutionCompiler
        from prefmem.execution.grounding import RGBDGrounder
        from prefmem.execution.sam import Sam3Client
        from prefmem.execution.service import ExecutionService
        from simulation.controller import PandaPickPlaceController
        from simulation.stacking import InMemoryTaskPublisher, StackingEnvironment

        environment = StackingEnvironment(
            seed=args.simulation_seed,
            width=args.simulation_render_size,
            height=args.simulation_render_size,
            render_hz=args.simulation_render_hz,
            viewer=args.simulation_viewer,
            viewer_camera=args.simulation_viewer_camera,
            scene_path=getattr(args, "simulation_scene", None),
        )
        sam = Sam3Client(
            args.sam_base_url,
            threshold=args.sam_threshold,
        )
        compiler = GemmaExecutionCompiler(
            model_name=args.execution_model,
            model_base_url=args.execution_model_base_url,
        )
        controller = PandaPickPlaceController(environment)
        executor = ExecutionService(
            compiler,
            RGBDGrounder(sam),
            controller,
            environment.rgbd_frame,
        )
        frame_source = environment.captured_frame
        task_publisher = InMemoryTaskPublisher()
        owned_resources = (sam, environment)

    try:
        runtime = PrefMemRuntime(
            hri.planner_agent,
            camera_base_url=args.camera_base_url,
            task_publisher=task_publisher,
            frame_source=frame_source,
            monitor_min_interval=args.monitor_min_interval,
            model_name=hri.config.model,
            model_base_url=hri.config.model_base_url,
            metrics=hri.metrics,
            success_confirmations=args.success_confirmations,
            success_stability_seconds=args.success_stability_seconds,
            failure_confirmations=args.failure_confirmations,
            ongoing_timeout_seconds=args.monitor_timeout,
            max_cycles=args.max_planning_cycles,
            executor=executor,
            owned_resources=owned_resources,
        )
    except Exception:
        if executor is not None:
            executor.stop()
        for resource in owned_resources:
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        raise
    hri.attach_runtime(runtime)
    return runtime, hri
