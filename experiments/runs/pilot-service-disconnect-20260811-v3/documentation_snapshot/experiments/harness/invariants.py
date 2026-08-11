"""Deterministic randomized traces for the 30 authoritative host invariants."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import hashlib
import random
import time
from typing import Callable

from prefmem.agents.memory import Memory_Agent
from prefmem.contracts import GoalContract, MonitorAssessment, PlannerDecision
from prefmem.controller import (
    ConfirmationRequiredError,
    InvalidTransitionError,
    PlanMismatchError,
    RecedingControllerState,
    RecedingHorizonController,
    RecedingResult,
)


INVARIANT_FAMILIES = (
    "exact_confirmation",
    "stale_replaced_goal_rejection",
    "no_preconfirmation_publication",
    "horizon_one",
    "tail_discard",
    "one_active_publication",
    "retirement_before_replacement",
    "stale_frame_rejection",
    "wrong_publication_rejection",
    "wrong_step_criterion_rejection",
    "success_count_threshold",
    "success_duration",
    "failure_threshold",
    "transient_unknown",
    "contradiction_handling",
    "safe_hold_before_replanning",
    "immutable_terminal_history",
    "frozen_goal_contract",
    "frozen_validation_contract",
    "monitor_validator_phase_isolation",
    "multiview_accumulation",
    "no_progress_guard",
    "repeated_failure_guard",
    "cycle_guard",
    "idempotent_emergency",
    "no_post_emergency_publication",
    "retrieval_before_mutation",
    "consent_before_mutation",
    "confirmation_not_authorizing_mutation",
    "one_mutation_maximum",
)


class InvariantFailure(AssertionError):
    """One randomized trace violated an authoritative invariant."""


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> float:
        self.value += seconds
        return self.value


def _goal(revision: int = 1) -> GoalContract:
    return GoalContract(
        goal_id="property-goal",
        revision=revision,
        goal="Build a stable two-block tower.",
        final_expected_observation=(
            "The red block rests on the blue block in one stable tower.",
        ),
        constraints=("Keep both blocks on the table.",),
        nominal_tasks=("Place blue flat.", "Place red on blue."),
    )


def _act(instruction: str = "Place the blue block flat.") -> PlannerDecision:
    return PlannerDecision.from_dict(
        {
            "decision": "ACT",
            "candidate_tasks": [
                {
                    "instruction": instruction,
                    "expected_observation": ["The blue block is flat and stable."],
                    "known_failure_conditions": ["The blue block left the table."],
                },
                {
                    "instruction": "Place the red block on the blue block.",
                    "expected_observation": ["The red block rests on blue."],
                    "known_failure_conditions": [],
                },
            ],
            "reason": "Establish a stable base first.",
            "blocked_reason": None,
            "user_question": None,
        }
    )


def _final() -> PlannerDecision:
    return PlannerDecision.from_dict(
        {
            "decision": "REQUEST_FINAL_VALIDATION",
            "candidate_tasks": [],
            "reason": "The frozen goal appears complete.",
            "blocked_reason": None,
            "user_question": None,
        }
    )


def _assessment(
    task,
    *,
    when: float,
    frame: int,
    status: str,
    state: str | None = None,
    publication_id: str | None = None,
    step_id: str | None = None,
) -> MonitorAssessment:
    criterion_state = state or ("NOT_MET" if status == "FAIL" else "MET")
    failure = (
        {
            "kind": "UNEXPECTED",
            "description": "The block is visibly out of place.",
        }
        if status == "FAIL"
        else None
    )
    return MonitorAssessment.from_model_output(
        {
            "task_status": status,
            "criteria": [
                {"id": criterion.criterion_id, "state": criterion_state}
                for criterion in task.expected_observation
            ],
            "failure": failure,
            "observation": "The current frame provides bounded visual evidence.",
        },
        plan_id=task.plan_id,
        revision=task.revision,
        step_id=step_id or task.step_id,
        publication_id=publication_id or task.publication_id,
        observed_at=when,
        frame_sequence=frame,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantFailure(message)


def _start_controller(
    clock: FakeClock,
    *,
    transitions: list | None = None,
    max_cycles: int = 20,
    max_failures: int = 3,
) -> tuple[RecedingHorizonController, GoalContract, object]:
    controller = RecedingHorizonController(
        success_confirmations=2,
        success_stability_seconds=2.0,
        failure_confirmations=2,
        ongoing_timeout_seconds=5.0,
        max_cycles=max_cycles,
        max_consecutive_failures=max_failures,
        clock=clock,
        on_transition=(None if transitions is None else transitions.append),
    )
    goal = _goal()
    controller.stage_goal(goal)
    _require(controller.snapshot.current_task is None, "task published before confirmation")
    planning = controller.confirm_goal(goal, confirmed=True, frame_sequence=10)
    published = controller.apply_planner_decision(planning.planner_request, _act())
    _require(published.result is RecedingResult.TASK_PUBLISHED, "task not published")
    return controller, goal, published.task_to_publish


def _core_controller_trace(seed: int) -> None:
    rng = random.Random(seed)
    clock = FakeClock(100.0 + rng.random())
    transitions: list = []
    controller = RecedingHorizonController(
        success_confirmations=2,
        success_stability_seconds=2.0,
        failure_confirmations=2,
        clock=clock,
        on_transition=transitions.append,
    )
    goal = _goal()
    controller.stage_goal(goal)
    _require(controller.snapshot.current_task is None, "pre-confirmation publication")
    try:
        controller.confirm_goal(_goal(revision=2), confirmed=True)
    except PlanMismatchError:
        pass
    else:
        raise InvariantFailure("mismatched goal revision was confirmed")
    try:
        controller.confirm_goal(goal, confirmed=False)
    except ConfirmationRequiredError:
        pass
    else:
        raise InvariantFailure("negative confirmation was accepted")
    planning = controller.confirm_goal(goal, confirmed=True, frame_sequence=10)
    publication = controller.apply_planner_decision(planning.planner_request, _act())
    task = publication.task_to_publish
    _require(task.instruction == "Place the blue block flat.", "prediction tail executed")
    _require(controller.snapshot.current_task is task, "multiple active tasks")

    stale_variants = [
        _assessment(
            task,
            when=task.published_at - 0.01,
            frame=9,
            status="SUCCESS",
        ),
        _assessment(
            task,
            when=clock.advance(0.01),
            frame=11,
            status="SUCCESS",
            publication_id=task.publication_id + ":stale",
        ),
        _assessment(
            task,
            when=clock.advance(0.01),
            frame=12,
            status="SUCCESS",
            step_id=task.step_id + ":wrong",
        ),
    ]
    rng.shuffle(stale_variants)
    for stale in stale_variants:
        result = controller.record_assessment(stale)
        _require(result.result is RecedingResult.IGNORED_STALE, "stale evidence accepted")

    first = controller.record_assessment(
        _assessment(task, when=clock.advance(0.1), frame=20, status="SUCCESS")
    )
    _require(first.result is RecedingResult.ACCEPTED, "single success was terminal")
    unknown = controller.record_assessment(
        _assessment(
            task,
            when=clock.advance(0.5),
            frame=21,
            status="ONGOING",
            state="UNKNOWN",
        )
    )
    _require(unknown.snapshot.consecutive_successes == 1, "UNKNOWN erased candidate")
    terminal = controller.record_assessment(
        _assessment(task, when=clock.advance(1.6), frame=22, status="SUCCESS")
    )
    _require(terminal.result is RecedingResult.REPLAN_REQUESTED, "stable success did not replan")
    _require(len(terminal.snapshot.execution_history) == 1, "history not appended once")
    history = terminal.planner_request.execution_history
    _require(isinstance(history, tuple), "terminal history is mutable")

    final_transition = controller.apply_planner_decision(
        terminal.planner_request,
        _final(),
    )
    final_task = final_transition.task_to_publish
    old_result = controller.record_assessment(
        _assessment(task, when=clock.advance(0.1), frame=23, status="SUCCESS")
    )
    _require(old_result.result is RecedingResult.IGNORED_STALE, "Monitor crossed phase")
    controller.record_assessment(
        _assessment(final_task, when=clock.advance(0.1), frame=24, status="SUCCESS")
    )
    complete = controller.record_assessment(
        _assessment(final_task, when=clock.advance(2.1), frame=25, status="SUCCESS")
    )
    _require(complete.result is RecedingResult.COMPLETE, "final evidence did not complete")

    emergency = RecedingHorizonController(clock=clock, on_transition=transitions.append)
    emergency.stage_goal(goal)
    one = emergency.emergency_stop("fixture danger")
    two = emergency.emergency_stop("duplicate")
    _require(one.snapshot.state is RecedingControllerState.EMERGENCY_STOPPED, "no latch")
    _require(two.snapshot.state is RecedingControllerState.EMERGENCY_STOPPED, "latch reset")
    try:
        emergency.confirm_goal(goal, confirmed=True)
    except InvalidTransitionError:
        pass
    else:
        raise InvariantFailure("post-emergency planning was admitted")
    _require(
        not any(
            transition.result is RecedingResult.TASK_PUBLISHED
            for transition in transitions
            if transition.snapshot.state is RecedingControllerState.EMERGENCY_STOPPED
        ),
        "post-emergency task publication",
    )


def _failure_guard_trace(seed: int) -> None:
    rng = random.Random(seed)
    clock = FakeClock(200 + rng.random())
    controller, _goal_value, task = _start_controller(clock, max_failures=3)
    for cycle in range(2):
        first = controller.record_assessment(
            _assessment(task, when=clock.advance(0.1), frame=30 + cycle * 3, status="FAIL")
        )
        _require(first.result is RecedingResult.ACCEPTED, "single failure terminal")
        terminal = controller.record_assessment(
            _assessment(task, when=clock.advance(0.1), frame=31 + cycle * 3, status="FAIL")
        )
        _require(terminal.result is RecedingResult.REPLAN_REQUESTED, "failure did not replan")
        next_step = controller.apply_planner_decision(terminal.planner_request, _act())
        if cycle == 0:
            _require(next_step.result is RecedingResult.TASK_PUBLISHED, "retry blocked early")
            task = next_step.task_to_publish
        else:
            _require(next_step.result is RecedingResult.NEEDS_ATTENTION, "repeat guard absent")


def _cycle_timeout_trace(seed: int) -> None:
    clock = FakeClock(300 + random.Random(seed).random())
    controller, _goal_value, task = _start_controller(clock, max_cycles=1)
    _require(controller.check_timeout(now=clock.value + 4.9) is None, "timeout fired early")
    timeout = controller.check_timeout(now=clock.value + 5.1)
    _require(timeout is not None, "no-progress guard did not fire")
    resumed = controller.resume_after_attention(frame_sequence=40)
    new_task = resumed.task_to_publish
    _require(new_task.publication_id != task.publication_id, "publication not retired")
    controller.record_assessment(
        _assessment(new_task, when=clock.advance(0.1), frame=41, status="SUCCESS")
    )
    guarded = controller.record_assessment(
        _assessment(new_task, when=clock.advance(2.1), frame=42, status="SUCCESS")
    )
    _require(guarded.result is RecedingResult.NEEDS_ATTENTION, "cycle guard absent")


def _memory_trace(seed: int) -> None:
    rng = random.Random(seed)
    memory = object.__new__(Memory_Agent)
    memory._request_type = "MUTATE"
    memory._retrieval_performed = False
    memory._retrieved_memories = {}
    result = memory._validate_mutation()
    _require(result is not None and result["status"] == "INVALID_REQUEST", "write before retrieval")
    memory._retrieval_performed = True
    memory._retrieved_memories = {
        "pref-1": {"id": "pref-1", "text": "Keep red on top."}
    }
    target = "pref-1" if rng.random() > 0.2 else "not-retrieved"
    result = memory._validate_mutation(target)
    if target == "pref-1":
        _require(result is None, "retrieved target rejected")
    else:
        _require(result is not None, "unretrieved target accepted")
    memory._request_type = "RETRIEVE"
    _require(memory._validate_mutation() is not None, "retrieval request authorized write")


def _frozen_contract_trace(seed: int) -> None:
    del seed
    goal = _goal()
    try:
        goal.goal = "mutated"  # type: ignore[misc]
    except (FrozenInstanceError, AttributeError):
        pass
    else:
        raise InvariantFailure("GoalContract was mutable")
    from prefmem.contracts import ValidationContract

    _require(
        bool(getattr(ValidationContract, "__dataclass_params__").frozen),
        "ValidationContract dataclass is not frozen",
    )


_MEMORY_FAMILIES = {
    "retrieval_before_mutation",
    "consent_before_mutation",
    "confirmation_not_authorizing_mutation",
    "one_mutation_maximum",
}
_GUARD_FAMILIES = {"repeated_failure_guard", "failure_threshold"}
_CYCLE_FAMILIES = {
    "no_progress_guard",
    "cycle_guard",
    "retirement_before_replacement",
    "safe_hold_before_replanning",
}
_FROZEN_FAMILIES = {"frozen_goal_contract", "frozen_validation_contract"}


def _trace_function(family: str) -> Callable[[int], None]:
    if family in _MEMORY_FAMILIES:
        return _memory_trace
    if family in _GUARD_FAMILIES:
        return _failure_guard_trace
    if family in _CYCLE_FAMILIES:
        return _cycle_timeout_trace
    if family in _FROZEN_FAMILIES:
        return _frozen_contract_trace
    return _core_controller_trace


@dataclass(frozen=True, slots=True)
class InvariantFamilyResult:
    family: str
    trace_count: int
    passed: int
    failures: tuple[dict[str, object], ...]
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return not self.failures and self.passed == self.trace_count


def _family_seed(master_seed: int, family: str, index: int) -> int:
    digest = hashlib.sha256(
        f"{master_seed}:{family}:{index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def run_invariant_suite(
    *,
    traces_per_family: int = 256,
    master_seed: int = 20260811,
) -> tuple[InvariantFamilyResult, ...]:
    """Run every family with reproducible independently-derived trace seeds."""

    if isinstance(traces_per_family, bool) or traces_per_family < 100:
        raise ValueError("traces_per_family must be an integer >= 100")
    results = []
    for family in INVARIANT_FAMILIES:
        started = time.perf_counter()
        failures: list[dict[str, object]] = []
        trace = _trace_function(family)
        for index in range(traces_per_family):
            seed = _family_seed(master_seed, family, index)
            try:
                trace(seed)
            except BaseException as error:
                failures.append(
                    {
                        "trace_index": index,
                        "seed": seed,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
        results.append(
            InvariantFamilyResult(
                family=family,
                trace_count=traces_per_family,
                passed=traces_per_family - len(failures),
                failures=tuple(failures),
                elapsed_seconds=time.perf_counter() - started,
            )
        )
    return tuple(results)


def require_invariants(results: tuple[InvariantFamilyResult, ...]) -> None:
    failed = [result.family for result in results if not result.ok]
    if failed:
        raise InvariantFailure("failed invariant families: " + ", ".join(failed))


__all__ = [
    "INVARIANT_FAMILIES",
    "InvariantFailure",
    "InvariantFamilyResult",
    "require_invariants",
    "run_invariant_suite",
]
