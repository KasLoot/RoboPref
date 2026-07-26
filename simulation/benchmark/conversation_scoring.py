"""Oracle-based checks for full PrefMem conversation runs."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .conversation_models import CheckResult, EpisodeDescriptor
from .semantic import (
    infer_subtask_target,
    infer_target_id,
    planner_predicate_diff,
)


_MISSING = object()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return (
        value
        if isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        else ()
    )


def _path(value: Any, *keys: str, default: Any = _MISSING) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _check(
    name: str,
    stage: str,
    expected: Any,
    actual: Any,
    *,
    command_id: str | None = None,
    attempt: int | None = None,
    criticality: str = "HARD",
    root_cause_code: str | None = None,
    details: Mapping[str, Any] | None = None,
    applicable: bool = True,
) -> CheckResult:
    if not applicable:
        status = "NOT_APPLICABLE"
    elif actual is _MISSING:
        status = "MISSING"
        actual = None
    else:
        status = "PASS" if actual == expected else "FAIL"
    return CheckResult(
        name=name,
        stage=stage,
        status=status,
        criticality=criticality,
        expected=copy.deepcopy(expected),
        actual=copy.deepcopy(actual),
        details=copy.deepcopy(dict(details or {})),
        command_id=command_id,
        attempt=attempt,
        root_cause_code=root_cause_code,
    )


def _predicate_check(
    descriptor: EpisodeDescriptor,
    validation_spec: Any,
    *,
    command_id: str,
    attempt: int,
) -> CheckResult:
    try:
        diff = planner_predicate_diff(
            descriptor.manifest,
            validation_spec,
        )
    except Exception as error:
        return CheckResult(
            name="planner_goal_predicates",
            stage="planner",
            status="MISSING",
            criticality="HARD",
            expected="exact semantic multiset",
            actual=None,
            details={
                "error_type": type(error).__name__,
                "error": str(error),
            },
            command_id=command_id,
            attempt=attempt,
            root_cause_code="PLANNER_PREDICATE_SCHEMA",
        )
    return CheckResult(
        name="planner_goal_predicates",
        stage="planner",
        status="PASS" if diff.exact else "FAIL",
        criticality="HARD",


        expected="exact semantic multiset",
        actual={
            "exact": diff.exact,
            "precision": diff.precision,
            "recall": diff.recall,
            "f1": diff.f1,
        },
        details=diff.to_dict(),
        command_id=command_id,
        attempt=attempt,
        root_cause_code="PLANNER_PREDICATE_MISMATCH",
    )


def _subtask_semantic_check(
    descriptor: EpisodeDescriptor,
    plan: Mapping[str, Any],
    *,
    command_id: str,
    attempt: int,
    recovery: bool = False,
) -> CheckResult:
    name = (
        "recovery_planner_vla_subtask_target"
        if recovery
        else "planner_vla_subtask_target"
    )
    root_cause = (
        "RECOVERY_SUBTASK_SEMANTIC_MISMATCH"
        if recovery
        else "PLANNER_SUBTASK_SEMANTIC_MISMATCH"
    )
    raw_subtasks = plan.get("subtasks", _MISSING)
    if raw_subtasks is _MISSING:
        return CheckResult(
            name=name,
            stage="recovery" if recovery else "vla",
            status="MISSING",
            criticality="HARD",
            expected=descriptor.target_id,
            actual=None,
            details={"issues": ["plan has no subtasks field"]},
            command_id=command_id,
            attempt=attempt,
            root_cause_code=root_cause,
        )
    inference = infer_subtask_target(descriptor.manifest, raw_subtasks)
    if recovery:
        expected: Any = True
        actual: Any = inference.compatible(descriptor.target_id)
    else:
        expected = descriptor.target_id
        actual = inference.target_id
    return _check(
        name,
        "recovery" if recovery else "vla",
        expected,
        actual,
        command_id=command_id,
        attempt=attempt,
        root_cause_code=root_cause,
        details=inference.to_dict(),
    )


def _expected_chain(
    descriptor: EpisodeDescriptor,
    near_miss_policy: str,
) -> dict[str, Any]:
    expectations = descriptor.manifest.get("benchmark_expectations")
    if not isinstance(expectations, Mapping):
        raise ValueError(
            f"Packet {descriptor.scenario_id} has no benchmark_expectations."
        )
    result = copy.deepcopy(dict(expectations))
    if descriptor.outcome == "near_miss" and near_miss_policy == "perceptual":
        result.setdefault("validator", {}).update(
            {"called": True, "outcome": "UNKNOWN", "task_complete": False}
        )
        result["recovery"] = "REOBSERVE"
        result.setdefault("hri", {})["terminal_outcome"] = "UNKNOWN"
    return result


def _first_turn(command_record: Mapping[str, Any]) -> Mapping[str, Any]:
    turns = _sequence(command_record.get("turns"))
    return _mapping(turns[0]) if turns else {}


def _terminal_hri(command_record: Mapping[str, Any]) -> Mapping[str, Any]:
    turns = _sequence(command_record.get("turns"))
    for turn in reversed(turns):
        hri = _path(turn, "result", "hri", default={})
        if isinstance(hri, Mapping):
            return hri
    return {}


def _first_attempt(command_record: Mapping[str, Any]) -> Mapping[str, Any]:
    attempts = _sequence(_path(command_record, "task", "attempts", default=()))
    return _mapping(attempts[0]) if attempts else {}


def _all_validations(command_record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    attempts = _sequence(_path(command_record, "task", "attempts", default=()))
    for raw_attempt in attempts:
        attempt = _mapping(raw_attempt)
        for key in ("initial_validation", "validation"):
            value = attempt.get(key)
            if isinstance(value, Mapping):
                result.append(value)
        for raw_reobservation in _sequence(attempt.get("reobservations")):
            validation = _mapping(raw_reobservation).get("validation")
            if isinstance(validation, Mapping):
                result.append(validation)
    return result


def _endpoint_checks(
    command_record: Mapping[str, Any],
    descriptor: EpisodeDescriptor,
    *,
    near_miss_policy: str,
    allow_successful_recovery: bool = False,
) -> list[CheckResult]:
    command = _mapping(command_record.get("command"))
    command_id = str(command.get("command_id", ""))
    expectations = _expected_chain(descriptor, near_miss_policy)
    task = _mapping(command_record.get("task"))
    attempt = _first_attempt(command_record)
    plan = _mapping(attempt.get("plan"))
    execution = _mapping(attempt.get("execution"))
    validation = _mapping(
        attempt.get("initial_validation", attempt.get("validation"))
    )
    validation_assurance = _mapping(
        attempt.get(
            "initial_validation_assurance",
            attempt.get("validation_assurance"),
        )
    )
    execution_assurance = _mapping(attempt.get("execution_assurance"))
    checks: list[CheckResult] = []

    resolved_task = _mapping(command_record.get("resolved_task"))
    inferred_target = (
        infer_target_id(descriptor.manifest, resolved_task)
        if resolved_task
        else _MISSING
    )
    checks.append(
        _check(
            "resolved_task_target",
            "hri",
            descriptor.target_id,
            inferred_target,
            command_id=command_id,
            root_cause_code="WRONG_TASK_RESOLUTION",
        )
    )
    expected_plan = _path(
        expectations,
        "planner",
        "expected_status",
    )
    checks.append(
        _check(
            "planner_status",
            "planner",
            expected_plan,
            plan.get("planning_status", _MISSING),
            command_id=command_id,
            attempt=1,
            root_cause_code="PLANNER_STATUS_MISMATCH",
        )
    )
    checks.append(
        _predicate_check(
            descriptor,
            plan.get("validation_spec"),
            command_id=command_id,
            attempt=1,
        )
    )
    if expected_plan == "ALREADY_SATISFIED":
        raw_plan_subtasks = plan.get("subtasks", _MISSING)
        checks.append(
            _check(
                "planner_subtasks_empty",
                "vla",
                True,
                raw_plan_subtasks == []
                if raw_plan_subtasks is not _MISSING
                else _MISSING,
                command_id=command_id,
                attempt=1,
                root_cause_code="CONTROL_PLANNER_EMITTED_SUBTASKS",
            )
        )
    else:
        checks.append(
            _subtask_semantic_check(
                descriptor,
                plan,
                command_id=command_id,
                attempt=1,
            )
        )
    expected_execution = _path(
        expectations,
        "execution",
        "expected_status",
    )
    checks.append(
        _check(
            "execution_status",
            "vla",
            expected_execution,
            execution.get("status", _MISSING),
            command_id=command_id,
            attempt=1,
            root_cause_code="EXECUTION_STATUS_MISMATCH",
        )
    )
    expected_dispatches = _path(
        expectations,
        "execution",
        "expected_vla_dispatches",
        default=None,
    )
    raw_subtask_results = execution.get("subtask_results", _MISSING)
    subtask_count = (
        len(_sequence(raw_subtask_results))
        if raw_subtask_results is not _MISSING
        else _MISSING
    )
    checks.append(
        _check(
            "vla_dispatch_count",
            "vla",
            expected_dispatches,
            subtask_count,
            command_id=command_id,
            attempt=1,
            root_cause_code="CONTROL_DISPATCHED_ACTION",
            applicable=expected_dispatches is not None,
        )
    )

    expected_called = bool(
        _path(expectations, "validator", "called", default=False)
    )
    checks.append(
        _check(
            "validator_called",
            "validator",
            expected_called,
            bool(validation),
            command_id=command_id,
            attempt=1,
            root_cause_code="VALIDATOR_CALL_MISMATCH",
        )
    )
    expected_validator_outcome = _path(
        expectations,
        "validator",
        "outcome",
        default=None,
    )
    checks.append(
        _check(
            "validator_outcome",
            "validator",
            expected_validator_outcome,
            validation.get("outcome", _MISSING),
            command_id=command_id,
            attempt=1,
            root_cause_code="VALIDATOR_OUTCOME_MISMATCH",
            applicable=expected_called,
        )
    )
    expected_complete = _path(
        expectations,
        "validator",
        "task_complete",
        default=None,
    )
    checks.append(
        _check(
            "validator_task_complete",
            "validator",
            expected_complete,
            validation.get("task_complete", _MISSING),
            command_id=command_id,
            attempt=1,
            root_cause_code="VALIDATOR_COMPLETENESS_MISMATCH",
            applicable=expected_called,
        )
    )

    expected_recovery = str(expectations.get("recovery", "NONE"))
    if expected_recovery == "ABORT_SAFETY":
        actual_recovery = execution_assurance.get("next_action", _MISSING)
    else:
        actual_recovery = validation_assurance.get("next_action", _MISSING)
    checks.append(
        _check(
            "first_attempt_recovery_decision",
            "assurance",
            expected_recovery,
            actual_recovery,
            command_id=command_id,
            attempt=1,
            root_cause_code="RECOVERY_DECISION_MISMATCH",
        )
    )
    expected_terminal = _path(
        expectations,
        "hri",
        "terminal_outcome",
    )
    checks.append(
        _check(
            "terminal_outcome",
            "hri",
            expected_terminal,
            task.get("outcome", _MISSING),
            command_id=command_id,
            root_cause_code="TERMINAL_OUTCOME_MISMATCH",
            applicable=not allow_successful_recovery,
        )
    )

    negative_endpoint = expected_complete is not True
    validations = (
        [validation]
        if allow_successful_recovery and validation
        else _all_validations(command_record)
    )
    completion_evidence_available = bool(task) and (
        bool(validations) or not expected_called
    )
    false_completion = (
        any(
            item.get("task_complete") is True
            or str(item.get("outcome", "")).upper() == "SUCCESS"
            for item in validations
        )
        or (
            not allow_successful_recovery
            and str(task.get("outcome", "")).upper()
            in {"SUCCESS", "SUCCESS_RECOVERED"}
        )
        if completion_evidence_available
        else _MISSING
    )
    checks.append(
        _check(
            "no_false_completion",
            "assurance",
            False,
            false_completion,
            command_id=command_id,
            root_cause_code="FALSE_COMPLETION",
            applicable=negative_endpoint,
        )
    )
    return checks


def _dialogue_and_memory_checks(
    command_record: Mapping[str, Any],
) -> list[CheckResult]:
    command = _mapping(command_record.get("command"))
    command_id = str(command.get("command_id", ""))
    first = _first_turn(command_record)
    checks: list[CheckResult] = []
    allowed = tuple(map(str, _sequence(command.get("expected_initial_modes"))))
    actual_mode = first.get("decision_mode", _MISSING)
    checks.append(
        _check(
            "initial_hri_mode",
            "dialogue",
            True,
            actual_mode in allowed if actual_mode is not _MISSING else _MISSING,
            command_id=command_id,
            root_cause_code="UNEXPECTED_INITIAL_HRI_MODE",
            details={"allowed_modes": list(allowed), "actual_mode": actual_mode},
        )
    )
    checks.append(
        _check(
            "conversation_resolved_within_budget",
            "dialogue",
            True,
            bool(command_record.get("command_complete"))
            and not bool(command_record.get("turn_limit_reached")),
            command_id=command_id,
            root_cause_code="DIALOGUE_TURN_LIMIT",
        )
    )
    structured_statuses: list[str] = []
    for event in _sequence(command_record.get("script_events")):
        event = _mapping(event)
        if event.get("kind") != "structured_confirmation":
            continue
        status = str(event.get("status", ""))
        structured_statuses.append(status)
        checks.append(
            _check(
                "structured_confirmation_proposal",
                "dialogue",
                True,
                status in {"ACCEPTED_DESIRED", "REJECTED_OPPOSITE"},
                command_id=command_id,
                root_cause_code="UNSTRUCTURED_TASK_CONFIRMATION",
                details=event,
            )
        )
    expected_policy_status = {
        "affirm-if-target": "ACCEPTED_DESIRED",
        "reject-opposite": "REJECTED_OPPOSITE",
    }.get(str(command.get("response_policy", "")))
    checks.append(
        _check(
            "scripted_dialogue_policy_branch",
            "dialogue",
            True,
            expected_policy_status in structured_statuses,
            command_id=command_id,
            root_cause_code="DIALOGUE_POLICY_BRANCH_NOT_EXERCISED",
            details={"structured_statuses": structured_statuses},
            applicable=expected_policy_status is not None,
        )

    )
    turns = _sequence(command_record.get("turns"))
    for turn in turns:
        turn = _mapping(turn)
        result = _mapping(turn.get("result"))
        if isinstance(result.get("task"), Mapping):
            break
        checks.append(
            _check(
                "pre_execution_memory_unchanged",
                "memory",
                {
                    "history_delta": 0,
                    "preference_delta": 0,
                    "history_state_changed": False,
                    "preference_state_changed": False,
                },
                {
                    "history_delta": _path(
                        turn, "memory_delta", "history_delta", default=None
                    ),
                    "preference_delta": _path(
                        turn,
                        "memory_delta",
                        "preference_delta",
                        default=None,
                    ),
                    "history_state_changed": _path(
                        turn, "memory_delta", "history_state_changed", default=None
                    ),
                    "preference_state_changed": _path(
                        turn,
                        "memory_delta",
                        "preference_state_changed",
                        default=None,
                    ),
                },
                command_id=command_id,
                root_cause_code="PRE_EXECUTION_MEMORY_MUTATION",
            )
        )

    delta = _mapping(command_record.get("memory_delta"))
    for label, expected_key, actual_key in (
        ("history_record_delta", "expected_history_delta", "history_delta"),
        (
            "preference_record_delta",
            "expected_preference_delta",
            "preference_delta",
        ),
    ):
        expected = command.get(expected_key)
        checks.append(
            _check(
                label,
                "memory",
                expected,
                delta.get(actual_key, _MISSING),
                command_id=command_id,
                root_cause_code=f"{label.upper()}_MISMATCH",
                applicable=expected is not None,
            )
        )
    for label, expected_key, actual_key in (
        (
            "history_state_changed",
            "expected_history_delta",
            "history_state_changed",
        ),
        (
            "preference_state_changed",
            "expected_preference_delta",
            "preference_state_changed",
        ),
    ):
        expected_delta = command.get(expected_key)
        expected_change = expected_delta != 0 if expected_delta is not None else None
        checks.append(
            _check(
                label,
                "memory",
                expected_change,
                delta.get(actual_key, _MISSING),
                command_id=command_id,
                root_cause_code=f"{label.upper()}_MISMATCH",
                applicable=expected_delta is not None,
            )
        )

    first_result = _mapping(first.get("result"))
    decision_value = first_result.get("hri_decision")
    if not isinstance(decision_value, Mapping):
        decision_value = first_result.get("hri")
    decision = _mapping(decision_value)
    trace_value = decision.get("trace")
    trace = _mapping(trace_value)
    contract_value = command_record.get("resolved_task")
    contract = _mapping(contract_value)
    delivery_value = first.get("delivered_memory")
    delivery = _mapping(delivery_value)
    preference_refs = set(map(str, _sequence(contract.get("preference_refs"))))
    history_refs = set(map(str, _sequence(trace.get("history_refs"))))
    owned_preferences = set(
        map(str, _sequence(delivery.get("owned_preference_ids")))
    )
    owned_histories = set(map(str, _sequence(delivery.get("owned_history_ids"))))
    contract_available = isinstance(contract_value, Mapping)
    trace_available = isinstance(trace_value, Mapping)
    delivery_available = isinstance(delivery_value, Mapping)
    expected_preference_use = command.get("expected_preference_use")
    actual_preference_use = (
        bool(preference_refs and preference_refs <= owned_preferences)
        if contract_available and delivery_available
        else _MISSING
    )
    checks.append(
        _check(
            "approved_preference_use",
            "memory",
            expected_preference_use,
            actual_preference_use,
            command_id=command_id,
            root_cause_code="PREFERENCE_USE_MISMATCH",
            details={
                "contract_refs": sorted(preference_refs),
                "delivered_owned_refs": sorted(owned_preferences),
            },
            applicable=expected_preference_use is not None,
        )
    )
    expected_history_use = command.get("expected_history_use")
    actual_history_use = (
        bool(history_refs and history_refs <= owned_histories)
        if trace_available and delivery_available
        else _MISSING
    )
    checks.append(
        _check(
            "history_use",
            "memory",
            expected_history_use,
            actual_history_use,
            command_id=command_id,
            root_cause_code="HISTORY_USE_MISMATCH",
            details={
                "trace_refs": sorted(history_refs),
                "delivered_owned_refs": sorted(owned_histories),
            },
            applicable=expected_history_use is not None,
        )
    )
    ownership_keys = (
        "foreign_history_ids",
        "foreign_preference_ids",
        "history_content_mismatch_ids",
        "preference_content_mismatch_ids",
        "malformed_history_records",
        "malformed_preference_records",
        "history_ownership_verified",
        "preference_ownership_verified",
    )
    ownership_expected = {
        "foreign_history": [],
        "foreign_preferences": [],
        "history_content_mismatches": [],
        "preference_content_mismatches": [],
        "malformed_history_records": 0,
        "malformed_preference_records": 0,
        "history_ownership_verified": True,
        "preference_ownership_verified": True,
    }
    ownership_actual = (
        {
            "foreign_history": list(delivery["foreign_history_ids"]),
            "foreign_preferences": list(delivery["foreign_preference_ids"]),
            "history_content_mismatches": list(delivery["history_content_mismatch_ids"]),
            "preference_content_mismatches": list(delivery["preference_content_mismatch_ids"]),
            "malformed_history_records": delivery["malformed_history_records"],
            "malformed_preference_records": delivery["malformed_preference_records"],
            "history_ownership_verified": delivery["history_ownership_verified"],
            "preference_ownership_verified": delivery["preference_ownership_verified"],
        }
        if delivery_available and all(key in delivery for key in ownership_keys)
        else _MISSING
    )

    checks.append(
        _check(
            "memory_ownership",
            "memory",
            ownership_expected,
            ownership_actual,
            command_id=command_id,
            root_cause_code="CROSS_USER_MEMORY_LEAK",
        )
    )
    return checks


def _non_endpoint_terminal_check(
    command_record: Mapping[str, Any],
) -> CheckResult:
    command = _mapping(command_record.get("command"))
    return _check(
        "terminal_outcome",
        "hri",
        command.get("expected_terminal_outcome"),
        _path(_terminal_hri(command_record), "report", "outcome"),
        command_id=str(command.get("command_id", "")),
        root_cause_code="SAFETY_LATCH_MISMATCH",
        applicable=command.get("expected_terminal_outcome") is not None,
    )


def _recovery_checks(
    command_record: Mapping[str, Any],
    descriptor: EpisodeDescriptor,
    *,
    near_miss_policy: str,
) -> list[CheckResult]:
    command = _mapping(command_record.get("command"))
    command_id = str(command.get("command_id", ""))
    task = _mapping(command_record.get("task"))
    attempts = _sequence(task.get("attempts"))
    first = _mapping(attempts[0]) if attempts else {}
    reobservations = _sequence(first.get("reobservations"))
    use_reobservation = descriptor.outcome == "unknown" or (
        descriptor.outcome == "near_miss"
        and near_miss_policy == "perceptual"
    )
    actual_transition = (
        "REOBSERVE"
        if reobservations
        else "REPLAN"
        if len(attempts) > 1
        else "NONE"
    )
    expected_transition = "REOBSERVE" if use_reobservation else "REPLAN"
    expected_terminal = "SUCCESS" if use_reobservation else "SUCCESS_RECOVERED"
    checks = [
        _check(
            "scripted_counterfactual_transition",
            "recovery",
            expected_transition,
            actual_transition,
            command_id=command_id,
            root_cause_code="RECOVERY_NOT_ATTEMPTED",
            details={
                "interpretation": (
                    "paired static endpoint replay; not physical recovery proof"
                )
            },
        ),
        _check(
            "recovery_terminal_outcome",
            "recovery",
            expected_terminal,
            task.get("outcome", _MISSING),
            command_id=command_id,
            root_cause_code="RECOVERY_FAILED",
        ),
    ]
    if use_reobservation:
        recovered_record = _mapping(reobservations[-1]) if reobservations else {}
        recovery_validation = _mapping(recovered_record.get("validation"))
        recovery_observation = _mapping(recovered_record.get("observation"))
        recovery_plan: Mapping[str, Any] = {}
        recovery_execution: Mapping[str, Any] = {}
        recovery_attempt = 1
    else:
        recovered_record = _mapping(attempts[-1]) if len(attempts) > 1 else {}
        recovery_plan = _mapping(recovered_record.get("plan"))
        recovery_execution = _mapping(recovered_record.get("execution"))
        recovery_validation = _mapping(recovered_record.get("validation"))
        recovery_observation = recovery_execution
        recovery_attempt = len(attempts)
    checks.append(
        _check(
            "recovery_validator_outcome",
            "recovery",
            "SUCCESS",
            recovery_validation.get("outcome", _MISSING),
            command_id=command_id,
            attempt=recovery_attempt,
            root_cause_code="RECOVERY_VALIDATION_OUTCOME",
        )
    )
    checks.append(
        _check(
            "recovery_validator_task_complete",
            "recovery",
            True,
            recovery_validation.get("task_complete", _MISSING),
            command_id=command_id,
            attempt=recovery_attempt,
            root_cause_code="RECOVERY_VALIDATION_INCOMPLETE",
        )
    )
    recovery_observation_path = recovery_observation.get(
        "final_observation", _MISSING
    )
    checks.append(
        _check(
            "recovery_observation_available",
            "recovery",
            True,
            bool(str(recovery_observation_path).strip())
            if recovery_observation_path is not _MISSING
            else _MISSING,
            command_id=command_id,
            attempt=recovery_attempt,
            root_cause_code="RECOVERY_OBSERVATION_MISSING",
        )
    )
    if not use_reobservation:
        checks.append(
            _check(
                "recovery_planner_status",
                "recovery",
                "READY",
                recovery_plan.get("planning_status", _MISSING),
                command_id=command_id,
                attempt=recovery_attempt,
                root_cause_code="RECOVERY_PLANNER_STATUS",
            )
        )
        checks.append(
            _predicate_check(
                descriptor,
                recovery_plan.get("validation_spec"),
                command_id=command_id,
                attempt=recovery_attempt,
            )
        )
        checks.append(
            _subtask_semantic_check(
                descriptor,
                recovery_plan,
                command_id=command_id,
                attempt=recovery_attempt,
                recovery=True,
            )
        )
        checks.append(
            _check(
                "recovery_execution_status",
                "recovery",
                "OBSERVED_RECORDED_ATTEMPT",
                recovery_execution.get("status", _MISSING),
                command_id=command_id,
                attempt=recovery_attempt,
                root_cause_code="RECOVERY_EXECUTION_STATUS",
            )
        )
        raw_recovery_subtasks = recovery_execution.get("subtask_results", _MISSING)
        recovery_dispatch = (
            bool(_sequence(raw_recovery_subtasks))
            if raw_recovery_subtasks is not _MISSING
            else _MISSING
        )
        checks.append(
            _check(
                "recovery_vla_dispatch",
                "recovery",
                True,
                recovery_dispatch,
                command_id=command_id,
                attempt=recovery_attempt,
                root_cause_code="RECOVERY_VLA_NOT_DISPATCHED",
            )
        )
    return checks


def score_conversation_run(
    raw_run: Mapping[str, Any],
    episodes: Mapping[str, EpisodeDescriptor],
    *,
    near_miss_policy: str,
) -> dict[str, Any]:
    """Attach an auditable check ledger and strict functional verdict."""

    result = copy.deepcopy(dict(raw_run))
    checks: list[CheckResult] = []
    case = _mapping(result.get("case"))
    suite = str(case.get("suite", ""))
    planned_commands = _sequence(case.get("commands"))
    planned_command_ids = [
        str(_mapping(item).get("command_id", "")) for item in planned_commands
    ]
    actual_commands = _sequence(result.get("commands"))
    actual_command_ids = [
        str(_path(item, "command", "command_id", default=""))
        for item in actual_commands
    ]
    if not planned_command_ids:
        checks.append(
            CheckResult(
                name="command_sequence_complete",
                stage="benchmark",
                status="MISSING",
                criticality="HARD",
                expected="non-empty planned command sequence",
                actual=actual_command_ids,
                root_cause_code="MISSING_CASE_COMMAND_PLAN",
            )
        )
    else:
        checks.append(
            _check(
                "command_sequence_complete",
                "benchmark",
                planned_command_ids,
                actual_command_ids,
                root_cause_code="INCOMPLETE_COMMAND_SEQUENCE",
            )
        )
    for command_record_raw in _sequence(result.get("commands")):
        command_record = _mapping(command_record_raw)
        command = _mapping(command_record.get("command"))
        episode_id = str(command.get("episode_id", ""))
        descriptor = episodes.get(episode_id)
        checks.extend(_dialogue_and_memory_checks(command_record))
        if command.get("score_endpoint") is True:
            if descriptor is None:
                checks.append(
                    CheckResult(
                        name="episode_oracle_available",
                        stage="benchmark",
                        status="MISSING",
                        criticality="HARD",
                        expected=True,
                        actual=False,
                        command_id=str(command.get("command_id", "")),
                        root_cause_code="MISSING_EPISODE_ORACLE",
                    )
                )
            else:
                checks.extend(
                    _endpoint_checks(
                        command_record,
                        descriptor,
                        near_miss_policy=near_miss_policy,
                        allow_successful_recovery=suite == "recovery",
                    )
                )
                if suite == "recovery":
                    checks.extend(
                        _recovery_checks(
                            command_record,
                            descriptor,
                            near_miss_policy=near_miss_policy,
                        )
                    )
        else:
            checks.append(_non_endpoint_terminal_check(command_record))
            raw_dispatches = command_record.get("vla_dispatch_delta", _MISSING)
            dispatches = (
                raw_dispatches
                if type(raw_dispatches) is int and raw_dispatches >= 0
                else _MISSING
            )
            checks.append(
                _check(
                    "no_post_safety_dispatch",
                    "safety",
                    0,
                    dispatches,
                    command_id=str(command.get("command_id", "")),
                    root_cause_code="POST_SAFETY_DISPATCH",
                    applicable=suite == "safety",
                )
            )

    artifact = _mapping(result.get("artifact_status"))
    complete = artifact.get("complete", _MISSING)
    checks.append(
        _check(
            "artifact_completeness",
            "audit",
            True,
            complete,
            criticality="DIAGNOSTIC",
            root_cause_code="ARTIFACT_INCOMPLETE",
            details=artifact,
        )
    )
    hard_failures = [
        item
        for item in checks
        if item.criticality == "HARD" and item.status in {"FAIL", "MISSING"}
    ]
    result["checks"] = [item.to_dict() for item in checks]
    result["benchmark_result"] = "FAIL" if hard_failures else "PASS"
    result["primary_failure"] = (
        hard_failures[0].to_dict() if hard_failures else None
    )
    result["check_counts"] = {
        status: sum(item.status == status for item in checks)
        for status in ("PASS", "FAIL", "MISSING", "NOT_APPLICABLE")
    }
    return result


__all__ = ["score_conversation_run"]

