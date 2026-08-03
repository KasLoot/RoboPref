"""Deterministic, thread-safe execution controller for confirmed plans."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
import time
from typing import Callable, Protocol

from prefmem.contracts import (
    CriterionState,
    ExecutionOutcome,
    ExecutionRecord,
    ExecutionPlan,
    GoalContract,
    MonitorAssessment,
    PlanStatus,
    PlannerCycleRequest,
    PlannerDecision,
    PlannerDecisionType,
    PlannerTrigger,
    PublishedTask,
    TaskPhase,
    TaskStatus,
    ObservationCriterion,
)


class ControllerState(str, Enum):
    IDLE = "IDLE"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    ALREADY_SATISFIED = "ALREADY_SATISFIED"
    BLOCKED = "BLOCKED"
    EXECUTING = "EXECUTING"
    FINAL_VALIDATION = "FINAL_VALIDATION"
    NEEDS_ATTENTION = "NEEDS_ATTENTION"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"


class AssessmentResult(str, Enum):
    ACCEPTED = "ACCEPTED"
    IGNORED_STALE = "IGNORED_STALE"
    IGNORED_INVALID = "IGNORED_INVALID"
    STEP_ADVANCED = "STEP_ADVANCED"
    FINAL_VALIDATION_STARTED = "FINAL_VALIDATION_STARTED"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"


class ControllerError(RuntimeError):
    """Base error for invalid controller operations."""


class InvalidTransitionError(ControllerError):
    """Raised when an operation is invalid in the current state."""


class PlanMismatchError(ControllerError):
    """Raised when confirmation does not contain the staged frozen plan."""


class ConfirmationRequiredError(ControllerError):
    """Raised unless the caller supplies an explicit positive confirmation."""


class TaskPublisher(Protocol):
    """Callback that publishes a task to the human execution surface."""

    def __call__(self, task: PublishedTask) -> None: ...


class MonitorRunner(Protocol):
    """Callback that starts monitoring a newly published task."""

    def __call__(self, task: PublishedTask) -> None: ...


@dataclass(frozen=True, slots=True)
class ControllerSnapshot:
    state: ControllerState
    plan: ExecutionPlan | None
    current_task: PublishedTask | None
    current_step_index: int | None
    attention_reason: str | None
    consecutive_successes: int
    consecutive_failures: int


class PlanController:
    """Own confirmation, publication, temporal evidence, and plan advancement.

    All state transitions are protected by a re-entrant lock.  External
    callbacks run after the transition is committed and outside the lock, so a
    monitor runner may safely deliver assessments from another thread.
    """

    def __init__(
        self,
        publisher: TaskPublisher,
        monitor_runner: MonitorRunner | None = None,
        *,
        success_confirmations: int = 2,
        success_stability_seconds: float = 2.0,
        failure_confirmations: int = 2,
        ongoing_timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(publisher):
            raise TypeError("publisher must be callable")
        if monitor_runner is not None and not callable(monitor_runner):
            raise TypeError("monitor_runner must be callable or None")
        if (
            isinstance(success_confirmations, bool)
            or not isinstance(success_confirmations, int)
            or success_confirmations < 2
        ):
            raise ValueError("success_confirmations must be at least 2")
        if (
            isinstance(success_stability_seconds, bool)
            or not isinstance(success_stability_seconds, (int, float))
            or not math.isfinite(success_stability_seconds)
            or success_stability_seconds < 0
        ):
            raise ValueError("success_stability_seconds cannot be negative")
        if (
            isinstance(failure_confirmations, bool)
            or not isinstance(failure_confirmations, int)
            or failure_confirmations < 2
        ):
            raise ValueError("failure_confirmations must be at least 2")
        if (
            isinstance(ongoing_timeout_seconds, bool)
            or not isinstance(ongoing_timeout_seconds, (int, float))
            or not math.isfinite(ongoing_timeout_seconds)
            or ongoing_timeout_seconds <= 0
        ):
            raise ValueError("ongoing_timeout_seconds must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._publisher = publisher
        self._monitor_runner = monitor_runner
        self._success_confirmations = success_confirmations
        self._success_stability_seconds = float(success_stability_seconds)
        self._failure_confirmations = failure_confirmations
        self._ongoing_timeout_seconds = float(ongoing_timeout_seconds)
        self._clock = clock
        self._lock = threading.RLock()

        self._state = ControllerState.IDLE
        self._plan: ExecutionPlan | None = None
        self._current_task: PublishedTask | None = None
        self._current_step_index: int | None = None
        self._attention_reason: str | None = None
        self._success_count = 0
        self._failure_count = 0
        self._first_success_at: float | None = None
        self._last_assessment_at: float | None = None
        self._last_progress_at: float | None = None
        self._best_met_count = 0
        self._failure_signature: tuple[str, str] | None = None

    @property
    def snapshot(self) -> ControllerSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def stage_plan(self, plan: ExecutionPlan) -> ControllerSnapshot:
        """Freeze a host-identified plan while publishing nothing."""

        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be an ExecutionPlan")
        with self._lock:
            if self._state in {
                ControllerState.EXECUTING,
                ControllerState.FINAL_VALIDATION,
            }:
                raise InvalidTransitionError(
                    "cannot replace a plan while it is executing"
                )
            if (
                self._plan is not None
                and plan.plan_id == self._plan.plan_id
                and plan.revision <= self._plan.revision
            ):
                raise ValueError(
                    "a replacement with the same plan_id needs a newer revision"
                )

            self._plan = plan
            self._current_task = None
            self._current_step_index = None
            self._attention_reason = plan.blocked_reason
            self._reset_evidence_locked()
            if plan.status is PlanStatus.READY:
                self._state = ControllerState.AWAITING_CONFIRMATION
                self._attention_reason = None
            elif plan.status is PlanStatus.ALREADY_SATISFIED:
                self._state = ControllerState.ALREADY_SATISFIED
                self._attention_reason = None
            else:
                self._state = ControllerState.BLOCKED
            return self._snapshot_locked()

    def confirm_and_publish(
        self,
        plan: ExecutionPlan,
        *,
        confirmed: bool,
    ) -> PublishedTask:
        """Publish step one only when the exact staged plan is confirmed."""

        if confirmed is not True:
            raise ConfirmationRequiredError(
                "explicit user confirmation is required before publication"
            )
        with self._lock:
            if self._state not in {
                ControllerState.AWAITING_CONFIRMATION,
                ControllerState.ALREADY_SATISFIED,
            }:
                raise InvalidTransitionError(
                    "the controller is not awaiting plan confirmation"
                )
            if plan != self._plan:
                raise PlanMismatchError(
                    "confirmation must include the exact staged plan and revision"
                )
            if plan.status is PlanStatus.ALREADY_SATISFIED:
                self._state = ControllerState.FINAL_VALIDATION
                self._current_step_index = None
                task = self._build_final_validation_task_locked()
            else:
                self._state = ControllerState.EXECUTING
                self._current_step_index = 0
                task = self._build_step_task_locked(0)
            self._activate_task_locked(task)
        self._dispatch(task)
        return task

    def record_assessment(
        self,
        assessment: MonitorAssessment,
    ) -> AssessmentResult:
        """Consume one monitor result and deterministically update state."""

        if not isinstance(assessment, MonitorAssessment):
            raise TypeError("assessment must be a MonitorAssessment")
        task_to_publish: PublishedTask | None = None
        with self._lock:
            task = self._current_task
            if (
                self._state
                not in {
                    ControllerState.EXECUTING,
                    ControllerState.FINAL_VALIDATION,
                }
                or task is None
            ):
                return AssessmentResult.IGNORED_STALE
            if (
                assessment.plan_id != task.plan_id
                or assessment.revision != task.revision
                or assessment.step_id != task.step_id
                or assessment.observed_at < task.published_at
                or (
                    self._last_assessment_at is not None
                    and assessment.observed_at <= self._last_assessment_at
                )
            ):
                return AssessmentResult.IGNORED_STALE

            self._last_assessment_at = assessment.observed_at
            expected_ids = {
                criterion.criterion_id
                for criterion in task.expected_observation
            }
            actual_ids = {
                criterion.criterion_id for criterion in assessment.criteria
            }
            if actual_ids != expected_ids:
                self._reset_streaks_locked()
                return AssessmentResult.IGNORED_INVALID

            met_count = sum(
                criterion.state is CriterionState.MET
                for criterion in assessment.criteria
            )
            if met_count > self._best_met_count:
                self._best_met_count = met_count
                self._last_progress_at = self._clock()

            if assessment.task_status is TaskStatus.SUCCESS:
                self._failure_count = 0
                if self._success_count == 0:
                    self._first_success_at = assessment.observed_at
                self._success_count += 1
                stable_for = (
                    assessment.observed_at - self._first_success_at
                    if self._first_success_at is not None
                    else 0.0
                )
                if (
                    self._success_count < self._success_confirmations
                    or stable_for < self._success_stability_seconds
                ):
                    return AssessmentResult.ACCEPTED

                if self._state is ControllerState.FINAL_VALIDATION:
                    self._state = ControllerState.COMPLETE
                    self._current_task = None
                    self._current_step_index = None
                    self._reset_evidence_locked()
                    return AssessmentResult.COMPLETE

                assert self._plan is not None
                assert self._current_step_index is not None
                next_index = self._current_step_index + 1
                if next_index < len(self._plan.steps):
                    self._current_step_index = next_index
                    task_to_publish = self._build_step_task_locked(next_index)
                    self._activate_task_locked(task_to_publish)
                    result = AssessmentResult.STEP_ADVANCED
                else:
                    self._state = ControllerState.FINAL_VALIDATION
                    self._current_step_index = None
                    task_to_publish = self._build_final_validation_task_locked()
                    self._activate_task_locked(task_to_publish)
                    result = AssessmentResult.FINAL_VALIDATION_STARTED
            elif assessment.task_status is TaskStatus.FAIL:
                self._success_count = 0
                self._first_success_at = None
                assert assessment.failure is not None
                failure_signature = (
                    assessment.failure.kind.value,
                    assessment.failure.description.casefold(),
                )
                if failure_signature == self._failure_signature:
                    self._failure_count += 1
                else:
                    self._failure_signature = failure_signature
                    self._failure_count = 1
                if self._failure_count < self._failure_confirmations:
                    return AssessmentResult.ACCEPTED
                self._state = ControllerState.FAILED
                self._attention_reason = assessment.failure.description
                return AssessmentResult.FAILED
            else:
                self._reset_streaks_locked()
                return AssessmentResult.ACCEPTED

        if task_to_publish is not None:
            self._dispatch(task_to_publish)
        return result

    def check_timeout(self, *, now: float | None = None) -> bool:
        """Move prolonged no-progress ONGOING work to NEEDS_ATTENTION.

        A timeout is deliberately not a task failure.  ``True`` means this call
        changed the state; ``False`` means no timeout was due.
        """

        timestamp = self._clock() if now is None else float(now)
        with self._lock:
            if self._state not in {
                ControllerState.EXECUTING,
                ControllerState.FINAL_VALIDATION,
            }:
                return False
            if self._last_progress_at is None:
                return False
            if timestamp - self._last_progress_at < self._ongoing_timeout_seconds:
                return False
            self._state = ControllerState.NEEDS_ATTENTION
            self._attention_reason = (
                "No observable progress before the monitoring timeout"
            )
            self._reset_streaks_locked()
            return True

    def resume_after_attention(self) -> PublishedTask:
        """Republish the unchanged task after the human elects to continue."""

        with self._lock:
            if (
                self._state is not ControllerState.NEEDS_ATTENTION
                or self._current_task is None
            ):
                raise InvalidTransitionError(
                    "there is no attention-paused task to resume"
                )
            phase = self._current_task.phase
            self._state = (
                ControllerState.EXECUTING
                if phase is TaskPhase.STEP
                else ControllerState.FINAL_VALIDATION
            )
            self._attention_reason = None
            task = replace(
                self._current_task,
                published_at=float(self._clock()),
            )
            self._activate_task_locked(task)
        self._dispatch(task)
        return task

    def _build_step_task_locked(self, index: int) -> PublishedTask:
        assert self._plan is not None
        step = self._plan.steps[index]
        return PublishedTask(
            plan_id=self._plan.plan_id,
            revision=self._plan.revision,
            step_id=step.step_id,
            phase=TaskPhase.STEP,
            instruction=step.instruction,
            expected_observation=step.expected_observation,
            known_failure_conditions=step.known_failure_conditions,
            published_at=float(self._clock()),
        )

    def _build_final_validation_task_locked(self) -> PublishedTask:
        assert self._plan is not None
        return PublishedTask(
            plan_id=self._plan.plan_id,
            revision=self._plan.revision,
            step_id=self._plan.final_validation_step_id,
            phase=TaskPhase.FINAL_VALIDATION,
            instruction=(
                "Keep the scene unchanged; move only the camera as needed to "
                "show every requested outcome."
            ),
            expected_observation=self._plan.final_expected_observation,
            known_failure_conditions=(),
            published_at=float(self._clock()),
        )

    def _activate_task_locked(self, task: PublishedTask) -> None:
        self._current_task = task
        self._attention_reason = None
        self._reset_evidence_locked()
        self._last_progress_at = task.published_at

    def _reset_evidence_locked(self) -> None:
        self._reset_streaks_locked()
        self._last_assessment_at = None
        self._last_progress_at = None
        self._best_met_count = 0

    def _reset_streaks_locked(self) -> None:
        self._success_count = 0
        self._failure_count = 0
        self._first_success_at = None
        self._failure_signature = None

    def _snapshot_locked(self) -> ControllerSnapshot:
        return ControllerSnapshot(
            state=self._state,
            plan=self._plan,
            current_task=self._current_task,
            current_step_index=self._current_step_index,
            attention_reason=self._attention_reason,
            consecutive_successes=self._success_count,
            consecutive_failures=self._failure_count,
        )

    def _dispatch(self, task: PublishedTask) -> None:
        try:
            self._publisher(task)
            if self._monitor_runner is not None:
                self._monitor_runner(task)
        except Exception as error:
            with self._lock:
                if self._current_task == task and self._state in {
                    ControllerState.EXECUTING,
                    ControllerState.FINAL_VALIDATION,
                }:
                    self._state = ControllerState.NEEDS_ATTENTION
                    self._attention_reason = (
                        f"Task publication failed: {type(error).__name__}"
                    )
            raise


class RecedingControllerState(str, Enum):
    """Runtime state for one confirmed receding-horizon goal session."""

    IDLE = "IDLE"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    FINAL_VALIDATION = "FINAL_VALIDATION"
    NEEDS_ATTENTION = "NEEDS_ATTENTION"
    COMPLETE = "COMPLETE"
    EMERGENCY_STOPPED = "EMERGENCY_STOPPED"


class RecedingResult(str, Enum):
    STAGED = "STAGED"
    CANCELLED = "CANCELLED"
    PLANNING_STARTED = "PLANNING_STARTED"
    TASK_PUBLISHED = "TASK_PUBLISHED"
    FINAL_VALIDATION_STARTED = "FINAL_VALIDATION_STARTED"
    ACCEPTED = "ACCEPTED"
    IGNORED_STALE = "IGNORED_STALE"
    IGNORED_INVALID = "IGNORED_INVALID"
    REPLAN_REQUESTED = "REPLAN_REQUESTED"
    NEEDS_ATTENTION = "NEEDS_ATTENTION"
    COMPLETE = "COMPLETE"
    EMERGENCY_STOPPED = "EMERGENCY_STOPPED"


class AttentionKind(str, Enum):
    NO_PROGRESS = "NO_PROGRESS"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    PLANNER_BLOCKED = "PLANNER_BLOCKED"
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"
    LOOP_GUARD = "LOOP_GUARD"


@dataclass(frozen=True, slots=True)
class RecedingControllerSnapshot:
    state: RecedingControllerState
    state_sequence: int
    goal: GoalContract | None
    cycle_id: int
    current_task: PublishedTask | None
    execution_history: tuple[ExecutionRecord, ...]
    pending_request: PlannerCycleRequest | None
    attention_kind: AttentionKind | None
    attention_reason: str | None
    latest_observation: str | None
    consecutive_successes: int
    consecutive_failures: int


@dataclass(frozen=True, slots=True)
class RecedingTransition:
    """Deterministic output for the runtime to act on outside the lock."""

    result: RecedingResult
    snapshot: RecedingControllerSnapshot
    task_to_publish: PublishedTask | None = None
    planner_request: PlannerCycleRequest | None = None


class RecedingHorizonController:
    """Freeze a goal, execute one task, observe, and request a fresh plan.

    The controller never calls a model or performs I/O.  It produces explicit
    transitions so the runtime can fetch a fresh frame, call Planner, publish
    the first candidate task, and start Monitor without holding this lock.
    """

    def __init__(
        self,
        *,
        success_confirmations: int = 2,
        success_stability_seconds: float = 2.0,
        failure_confirmations: int = 2,
        ongoing_timeout_seconds: float = 30.0,
        max_cycles: int = 20,
        max_consecutive_failures: int = 3,
        max_identical_failed_attempts: int = 2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value, minimum in (
            ("success_confirmations", success_confirmations, 2),
            ("failure_confirmations", failure_confirmations, 2),
            ("max_cycles", max_cycles, 1),
            ("max_consecutive_failures", max_consecutive_failures, 1),
            ("max_identical_failed_attempts", max_identical_failed_attempts, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        for name, value, allow_zero in (
            ("success_stability_seconds", success_stability_seconds, True),
            ("ongoing_timeout_seconds", ongoing_timeout_seconds, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (not allow_zero and value == 0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a finite {qualifier} number")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._success_confirmations = success_confirmations
        self._success_stability_seconds = float(success_stability_seconds)
        self._failure_confirmations = failure_confirmations
        self._ongoing_timeout_seconds = float(ongoing_timeout_seconds)
        self._max_cycles = max_cycles
        self._max_consecutive_failures = max_consecutive_failures
        self._max_identical_failed_attempts = max_identical_failed_attempts
        self._clock = clock
        self._lock = threading.RLock()

        self._state = RecedingControllerState.IDLE
        self._state_sequence = 0
        self._goal: GoalContract | None = None
        self._cycle_id = 0
        self._attempt = 0
        self._current_task: PublishedTask | None = None
        self._history: list[ExecutionRecord] = []
        self._pending_request: PlannerCycleRequest | None = None
        self._attention_kind: AttentionKind | None = None
        self._attention_reason: str | None = None
        self._latest_observation: str | None = None
        self._success_count = 0
        self._failure_count = 0
        self._first_success_at: float | None = None
        self._last_assessment_at: float | None = None
        self._last_frame_sequence: int | None = None
        self._last_progress_at: float | None = None
        self._best_met_count = 0
        self._consecutive_terminal_failures = 0

    @property
    def snapshot(self) -> RecedingControllerSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def stage_goal(self, goal: GoalContract) -> RecedingTransition:
        if not isinstance(goal, GoalContract):
            raise TypeError("goal must be a GoalContract")
        with self._lock:
            self._reject_emergency_locked()
            if self._state in {
                RecedingControllerState.PLANNING,
                RecedingControllerState.EXECUTING,
                RecedingControllerState.FINAL_VALIDATION,
            }:
                raise InvalidTransitionError("cannot replace an active goal")
            self._goal = goal
            self._cycle_id = 0
            self._attempt = 0
            self._current_task = None
            self._history = []
            self._pending_request = None
            self._attention_kind = None
            self._attention_reason = None
            self._latest_observation = None
            self._consecutive_terminal_failures = 0
            self._reset_evidence_locked()
            self._set_state_locked(RecedingControllerState.AWAITING_CONFIRMATION)
            return self._transition_locked(RecedingResult.STAGED)

    def cancel_staged_goal(self) -> RecedingTransition:
        """Discard an unconfirmed goal so it cannot remain authoritative."""

        with self._lock:
            self._reject_emergency_locked()
            if self._state is not RecedingControllerState.AWAITING_CONFIRMATION:
                raise InvalidTransitionError(
                    "only a goal awaiting confirmation can be cancelled"
                )
            self._goal = None
            self._cycle_id = 0
            self._attempt = 0
            self._current_task = None
            self._history = []
            self._pending_request = None
            self._attention_kind = None
            self._attention_reason = None
            self._latest_observation = None
            self._consecutive_terminal_failures = 0
            self._reset_evidence_locked()
            self._set_state_locked(RecedingControllerState.IDLE)
            return self._transition_locked(RecedingResult.CANCELLED)

    def confirm_goal(
        self,
        goal: GoalContract,
        *,
        confirmed: bool,
        frame_sequence: int | None = None,
    ) -> RecedingTransition:
        if confirmed is not True:
            raise ConfirmationRequiredError(
                "explicit user confirmation is required before planning execution"
            )
        with self._lock:
            self._reject_emergency_locked()
            if self._state is not RecedingControllerState.AWAITING_CONFIRMATION:
                raise InvalidTransitionError("the controller is not awaiting confirmation")
            if goal != self._goal:
                raise PlanMismatchError("confirmation must reference the exact staged goal")
            request = self._begin_planning_locked(
                PlannerTrigger.CONFIRMED,
                frame_sequence=frame_sequence,
                next_cycle=1,
            )
            return self._transition_locked(
                RecedingResult.PLANNING_STARTED,
                planner_request=request,
            )

    def bind_planning_frame(
        self,
        frame_sequence: int | None,
    ) -> PlannerCycleRequest:
        """Attach trusted current-frame metadata before invoking Planner."""

        with self._lock:
            self._reject_emergency_locked()
            if (
                self._state is not RecedingControllerState.PLANNING
                or self._pending_request is None
            ):
                raise InvalidTransitionError("there is no active planning request")
            request = replace(
                self._pending_request,
                frame_sequence=frame_sequence,
            )
            self._pending_request = request
            return request

    def apply_planner_decision(
        self,
        request: PlannerCycleRequest,
        decision: PlannerDecision,
    ) -> RecedingTransition:
        if not isinstance(request, PlannerCycleRequest):
            raise TypeError("request must be a PlannerCycleRequest")
        if not isinstance(decision, PlannerDecision):
            raise TypeError("decision must be a PlannerDecision")
        with self._lock:
            self._reject_emergency_locked()
            if (
                self._state is not RecedingControllerState.PLANNING
                or self._pending_request != request
            ):
                raise InvalidTransitionError("planner result is stale or unexpected")
            assert self._goal is not None

            if decision.decision is PlannerDecisionType.ACT:
                first = decision.candidate_tasks[0]
                failed_matches = sum(
                    record.outcome
                    in {ExecutionOutcome.FAIL, ExecutionOutcome.FINAL_VALIDATION_FAIL}
                    and record.instruction.casefold() == first.instruction.casefold()
                    for record in self._history
                )
                if failed_matches >= self._max_identical_failed_attempts:
                    self._pending_request = None
                    self._attention_kind = AttentionKind.LOOP_GUARD
                    self._attention_reason = (
                        "Planner repeated a task that already failed too many times"
                    )
                    self._set_state_locked(RecedingControllerState.NEEDS_ATTENTION)
                    return self._transition_locked(RecedingResult.NEEDS_ATTENTION)

                self._attempt = 1
                task = self._build_task_locked(
                    first.instruction,
                    first.expected_observation,
                    first.known_failure_conditions,
                    phase=TaskPhase.STEP,
                )
                self._current_task = task
                self._pending_request = None
                self._attention_kind = None
                self._attention_reason = None
                self._activate_task_locked(task)
                self._set_state_locked(RecedingControllerState.EXECUTING)
                return self._transition_locked(
                    RecedingResult.TASK_PUBLISHED,
                    task_to_publish=task,
                )

            if decision.decision is PlannerDecisionType.REQUEST_FINAL_VALIDATION:
                self._attempt = 1
                task = self._build_task_locked(
                    (
                        "Keep the scene unchanged; move only the camera as "
                        "needed to show every requested outcome."
                    ),
                    self._goal.final_expected_observation,
                    (),
                    phase=TaskPhase.FINAL_VALIDATION,
                )
                self._current_task = task
                self._pending_request = None
                self._attention_kind = None
                self._attention_reason = None
                self._activate_task_locked(task)
                self._set_state_locked(RecedingControllerState.FINAL_VALIDATION)
                return self._transition_locked(
                    RecedingResult.FINAL_VALIDATION_STARTED,
                    task_to_publish=task,
                )

            self._pending_request = None
            self._attention_kind = (
                AttentionKind.PLANNER_BLOCKED
                if decision.decision is PlannerDecisionType.BLOCKED
                else AttentionKind.NEEDS_USER_INPUT
            )
            self._attention_reason = (
                decision.blocked_reason
                if decision.decision is PlannerDecisionType.BLOCKED
                else decision.user_question
            )
            self._set_state_locked(RecedingControllerState.NEEDS_ATTENTION)
            return self._transition_locked(RecedingResult.NEEDS_ATTENTION)

    def record_assessment(
        self,
        assessment: MonitorAssessment,
    ) -> RecedingTransition:
        if not isinstance(assessment, MonitorAssessment):
            raise TypeError("assessment must be a MonitorAssessment")
        with self._lock:
            if self._state is RecedingControllerState.EMERGENCY_STOPPED:
                return self._transition_locked(RecedingResult.IGNORED_STALE)
            task = self._current_task
            if self._state not in {
                RecedingControllerState.EXECUTING,
                RecedingControllerState.FINAL_VALIDATION,
            } or task is None:
                return self._transition_locked(RecedingResult.IGNORED_STALE)
            if not self._assessment_matches_locked(assessment, task):
                return self._transition_locked(RecedingResult.IGNORED_STALE)

            self._last_assessment_at = assessment.observed_at
            self._last_frame_sequence = assessment.frame_sequence
            expected_ids = {
                criterion.criterion_id for criterion in task.expected_observation
            }
            actual_ids = {
                criterion.criterion_id for criterion in assessment.criteria
            }
            if actual_ids != expected_ids:
                self._reset_streaks_locked()
                return self._transition_locked(RecedingResult.IGNORED_INVALID)

            self._latest_observation = assessment.observation
            met_count = sum(
                criterion.state is CriterionState.MET
                for criterion in assessment.criteria
            )
            if met_count > self._best_met_count:
                self._best_met_count = met_count
                self._last_progress_at = self._clock()

            if assessment.task_status is TaskStatus.ONGOING:
                # An inconclusive view is not evidence that a previously
                # visible success disappeared. Preserve the candidate across
                # MET/UNKNOWN-only frames, but reset it on visible
                # contradiction. The ordinary no-progress timeout still
                # bounds how long an UNKNOWN view can remain neutral.
                self._failure_count = 0
                if any(
                    criterion.state is CriterionState.NOT_MET
                    for criterion in assessment.criteria
                ):
                    self._success_count = 0
                    self._first_success_at = None
                self._bump_sequence_locked()
                return self._transition_locked(RecedingResult.ACCEPTED)

            if assessment.task_status is TaskStatus.SUCCESS:
                self._failure_count = 0
                if self._success_count == 0:
                    self._first_success_at = assessment.observed_at
                self._success_count += 1
                stable_for = (
                    assessment.observed_at - self._first_success_at
                    if self._first_success_at is not None
                    else 0.0
                )
                if (
                    self._success_count < self._success_confirmations
                    or stable_for < self._success_stability_seconds
                ):
                    self._bump_sequence_locked()
                    return self._transition_locked(RecedingResult.ACCEPTED)

                if self._state is RecedingControllerState.FINAL_VALIDATION:
                    self._current_task = None
                    self._pending_request = None
                    self._attention_kind = None
                    self._attention_reason = None
                    self._reset_evidence_locked()
                    self._set_state_locked(RecedingControllerState.COMPLETE)
                    return self._transition_locked(RecedingResult.COMPLETE)

                self._append_record_locked(
                    task,
                    assessment,
                    ExecutionOutcome.SUCCESS,
                )
                self._consecutive_terminal_failures = 0
                request = self._request_next_cycle_locked(
                    PlannerTrigger.TASK_SUCCESS,
                    assessment.frame_sequence,
                )
                if request is None:
                    return self._transition_locked(RecedingResult.NEEDS_ATTENTION)
                return self._transition_locked(
                    RecedingResult.REPLAN_REQUESTED,
                    planner_request=request,
                )

            self._success_count = 0
            self._first_success_at = None
            self._failure_count += 1
            if self._failure_count < self._failure_confirmations:
                self._bump_sequence_locked()
                return self._transition_locked(RecedingResult.ACCEPTED)

            assert assessment.failure is not None
            final_validation = (
                self._state is RecedingControllerState.FINAL_VALIDATION
            )
            outcome = (
                ExecutionOutcome.FINAL_VALIDATION_FAIL
                if final_validation
                else ExecutionOutcome.FAIL
            )
            self._append_record_locked(task, assessment, outcome)
            self._consecutive_terminal_failures += 1
            if self._consecutive_terminal_failures >= self._max_consecutive_failures:
                self._current_task = None
                self._pending_request = None
                self._attention_kind = AttentionKind.LOOP_GUARD
                self._attention_reason = (
                    "Too many consecutive task failures; human review is required"
                )
                self._reset_evidence_locked()
                self._set_state_locked(RecedingControllerState.NEEDS_ATTENTION)
                return self._transition_locked(RecedingResult.NEEDS_ATTENTION)
            request = self._request_next_cycle_locked(
                PlannerTrigger.FINAL_VALIDATION_FAIL
                if final_validation
                else PlannerTrigger.TASK_FAIL,
                assessment.frame_sequence,
            )
            if request is None:
                return self._transition_locked(RecedingResult.NEEDS_ATTENTION)
            return self._transition_locked(
                RecedingResult.REPLAN_REQUESTED,
                planner_request=request,
            )

    def check_timeout(self, *, now: float | None = None) -> RecedingTransition | None:
        timestamp = self._clock() if now is None else float(now)
        with self._lock:
            if self._state not in {
                RecedingControllerState.EXECUTING,
                RecedingControllerState.FINAL_VALIDATION,
            } or self._last_progress_at is None:
                return None
            if timestamp - self._last_progress_at < self._ongoing_timeout_seconds:
                return None
            self._attention_kind = AttentionKind.NO_PROGRESS
            self._attention_reason = "No observable progress before the monitoring timeout"
            self._reset_streaks_locked()
            self._set_state_locked(RecedingControllerState.NEEDS_ATTENTION)
            return self._transition_locked(RecedingResult.NEEDS_ATTENTION)

    def record_system_error(self, reason: str) -> RecedingTransition:
        with self._lock:
            self._reject_emergency_locked()
            self._attention_kind = AttentionKind.SYSTEM_ERROR
            self._attention_reason = reason.strip() or "Unknown system error"
            self._reset_streaks_locked()
            self._set_state_locked(RecedingControllerState.NEEDS_ATTENTION)
            return self._transition_locked(RecedingResult.NEEDS_ATTENTION)

    def resume_after_attention(
        self,
        *,
        frame_sequence: int | None = None,
    ) -> RecedingTransition:
        with self._lock:
            self._reject_emergency_locked()
            if (
                self._state is not RecedingControllerState.NEEDS_ATTENTION
                or self._current_task is None
            ):
                raise InvalidTransitionError("there is no paused task to resume")
            self._attempt += 1
            task = replace(
                self._current_task,
                publication_id=self._publication_id_locked(self._attempt),
                published_at=float(self._clock()),
                frame_sequence=frame_sequence,
            )
            self._current_task = task
            self._attention_kind = None
            self._attention_reason = None
            self._activate_task_locked(task)
            self._set_state_locked(
                RecedingControllerState.EXECUTING
                if task.phase is TaskPhase.STEP
                else RecedingControllerState.FINAL_VALIDATION
            )
            return self._transition_locked(
                RecedingResult.TASK_PUBLISHED,
                task_to_publish=task,
            )

    def request_replan(
        self,
        *,
        operator_guidance: str | None = None,
        frame_sequence: int | None = None,
    ) -> RecedingTransition:
        with self._lock:
            self._reject_emergency_locked()
            if self._state is not RecedingControllerState.NEEDS_ATTENTION:
                raise InvalidTransitionError("replanning requires a paused task")
            if self._current_task is not None:
                observation = self._latest_observation or (
                    self._attention_reason or "The task attempt was interrupted."
                )
                if not observation.endswith((".", "!", "?")):
                    observation += "."
                self._history.append(
                    ExecutionRecord(
                        cycle_id=self._cycle_id,
                        publication_id=self._current_task.publication_id,
                        instruction=self._current_task.instruction,
                        expected_observation=tuple(
                            item.description
                            for item in self._current_task.expected_observation
                        ),
                        outcome=ExecutionOutcome.INTERRUPTED,
                        observation=observation,
                        evidence_frame_sequence=frame_sequence,
                    )
                )
            request = self._request_next_cycle_locked(
                PlannerTrigger.USER_REPLAN,
                frame_sequence,
                operator_guidance=operator_guidance,
            )
            if request is None:
                return self._transition_locked(RecedingResult.NEEDS_ATTENTION)
            return self._transition_locked(
                RecedingResult.REPLAN_REQUESTED,
                planner_request=request,
            )

    def emergency_stop(self, reason: str) -> RecedingTransition:
        """Latch a terminal emergency state; it cannot be reset in this process."""

        with self._lock:
            if self._state is RecedingControllerState.EMERGENCY_STOPPED:
                return self._transition_locked(RecedingResult.EMERGENCY_STOPPED)
            self._attention_kind = None
            self._attention_reason = reason.strip() or "Emergency stop requested"
            self._pending_request = None
            self._reset_streaks_locked()
            self._set_state_locked(RecedingControllerState.EMERGENCY_STOPPED)
            return self._transition_locked(RecedingResult.EMERGENCY_STOPPED)

    def _request_next_cycle_locked(
        self,
        trigger: PlannerTrigger,
        frame_sequence: int | None,
        *,
        operator_guidance: str | None = None,
    ) -> PlannerCycleRequest | None:
        next_cycle = self._cycle_id + 1
        self._current_task = None
        self._reset_evidence_locked()
        if next_cycle > self._max_cycles:
            self._pending_request = None
            self._attention_kind = AttentionKind.LOOP_GUARD
            self._attention_reason = "Maximum planning cycles reached"
            self._set_state_locked(RecedingControllerState.NEEDS_ATTENTION)
            return None
        return self._begin_planning_locked(
            trigger,
            frame_sequence=frame_sequence,
            next_cycle=next_cycle,
            operator_guidance=operator_guidance,
        )

    def _begin_planning_locked(
        self,
        trigger: PlannerTrigger,
        *,
        frame_sequence: int | None,
        next_cycle: int,
        operator_guidance: str | None = None,
    ) -> PlannerCycleRequest:
        assert self._goal is not None
        self._cycle_id = next_cycle
        self._current_task = None
        self._attention_kind = None
        self._attention_reason = None
        request = PlannerCycleRequest(
            goal_contract=self._goal,
            cycle_id=self._cycle_id,
            trigger=trigger,
            execution_history=tuple(self._history),
            operator_guidance=operator_guidance,
            frame_sequence=frame_sequence,
        )
        self._pending_request = request
        self._set_state_locked(RecedingControllerState.PLANNING)
        return request

    def _build_task_locked(
        self,
        instruction: str,
        expected: tuple[str, ...],
        failures: tuple[str, ...],
        *,
        phase: TaskPhase,
    ) -> PublishedTask:
        assert self._goal is not None
        suffix = "final" if phase is TaskPhase.FINAL_VALIDATION else "task"
        step_id = (
            f"{self._goal.goal_id}:r{self._goal.revision}:"
            f"c{self._cycle_id}:{suffix}"
        )
        criteria = tuple(
            ObservationCriterion(
                criterion_id=f"{step_id}:c{index}",
                description=description,
            )
            for index, description in enumerate(expected, start=1)
        )
        return PublishedTask(
            plan_id=self._goal.goal_id,
            revision=self._goal.revision,
            step_id=step_id,
            phase=phase,
            instruction=instruction,
            expected_observation=criteria,
            known_failure_conditions=failures,
            published_at=float(self._clock()),
            publication_id=self._publication_id_locked(self._attempt),
            cycle_id=self._cycle_id,
            frame_sequence=(
                None
                if self._pending_request is None
                else self._pending_request.frame_sequence
            ),
        )

    def _publication_id_locked(self, attempt: int) -> str:
        assert self._goal is not None
        return (
            f"{self._goal.goal_id}:r{self._goal.revision}:"
            f"c{self._cycle_id}:a{attempt}"
        )

    def _assessment_matches_locked(
        self,
        assessment: MonitorAssessment,
        task: PublishedTask,
    ) -> bool:
        if (
            assessment.plan_id != task.plan_id
            or assessment.revision != task.revision
            or assessment.step_id != task.step_id
            or assessment.publication_id != task.publication_id
            or assessment.observed_at < task.published_at
        ):
            return False
        if (
            self._last_assessment_at is not None
            and assessment.observed_at <= self._last_assessment_at
        ):
            return False
        if (
            task.frame_sequence is not None
            and assessment.frame_sequence is not None
            and assessment.frame_sequence <= task.frame_sequence
        ):
            return False
        if (
            self._last_frame_sequence is not None
            and assessment.frame_sequence is not None
            and assessment.frame_sequence <= self._last_frame_sequence
        ):
            return False
        return True

    def _append_record_locked(
        self,
        task: PublishedTask,
        assessment: MonitorAssessment,
        outcome: ExecutionOutcome,
    ) -> None:
        self._history.append(
            ExecutionRecord(
                cycle_id=self._cycle_id,
                publication_id=task.publication_id,
                instruction=task.instruction,
                expected_observation=tuple(
                    item.description for item in task.expected_observation
                ),
                outcome=outcome,
                observation=assessment.observation,
                failure_reason=(
                    None
                    if assessment.failure is None
                    else assessment.failure.description
                ),
                evidence_frame_sequence=assessment.frame_sequence,
            )
        )

    def _activate_task_locked(self, task: PublishedTask) -> None:
        self._reset_evidence_locked()
        self._last_progress_at = task.published_at

    def _reset_evidence_locked(self) -> None:
        self._reset_streaks_locked()
        self._last_assessment_at = None
        self._last_frame_sequence = None
        self._last_progress_at = None
        self._best_met_count = 0

    def _reset_streaks_locked(self) -> None:
        self._success_count = 0
        self._failure_count = 0
        self._first_success_at = None

    def _reject_emergency_locked(self) -> None:
        if self._state is RecedingControllerState.EMERGENCY_STOPPED:
            raise InvalidTransitionError("the emergency stop is latched")

    def _set_state_locked(self, state: RecedingControllerState) -> None:
        self._state = state
        self._bump_sequence_locked()

    def _bump_sequence_locked(self) -> None:
        self._state_sequence += 1

    def _snapshot_locked(self) -> RecedingControllerSnapshot:
        return RecedingControllerSnapshot(
            state=self._state,
            state_sequence=self._state_sequence,
            goal=self._goal,
            cycle_id=self._cycle_id,
            current_task=self._current_task,
            execution_history=tuple(self._history),
            pending_request=self._pending_request,
            attention_kind=self._attention_kind,
            attention_reason=self._attention_reason,
            latest_observation=self._latest_observation,
            consecutive_successes=self._success_count,
            consecutive_failures=self._failure_count,
        )

    def _transition_locked(
        self,
        result: RecedingResult,
        *,
        task_to_publish: PublishedTask | None = None,
        planner_request: PlannerCycleRequest | None = None,
    ) -> RecedingTransition:
        return RecedingTransition(
            result=result,
            snapshot=self._snapshot_locked(),
            task_to_publish=task_to_publish,
            planner_request=planner_request,
        )
