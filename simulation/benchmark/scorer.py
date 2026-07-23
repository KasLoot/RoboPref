from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


_MISSING = object()


@dataclass(frozen=True, slots=True)
class ScoreReport:
    scenario_id: str
    passed: bool
    checks: tuple[dict[str, Any], ...]

    @property
    def correct(self) -> int:
        return sum(
            bool(check["passed"])
            for check in self.checks
            if check.get("evaluated") is True
        )

    @property
    def total(self) -> int:
        return sum(check.get("evaluated") is True for check in self.checks)

    @property
    def skipped(self) -> int:
        return sum(check.get("evaluated") is not True for check in self.checks)


def _nested(
    value: Mapping[str, Any], *paths: tuple[str, ...], default: Any = _MISSING
) -> Any:
    for path in paths:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            return current
    return default


def score_agent_result(
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    allow_partial: bool = False,
) -> ScoreReport:
    """Score a structured PrefMem result against hidden benchmark expectations.

    This function is intentionally one-way: it consumes oracle truth only after an
    agent run and returns metrics.  Its output must not be fed back as HRI context.
    """

    scenario_id = str(manifest.get("scenario_id", ""))
    expectations = manifest.get("benchmark_expectations", {})
    if not isinstance(expectations, Mapping):
        raise ValueError("Manifest benchmark_expectations must be an object.")

    validator_expected = _nested(expectations, ("validator", "outcome"))
    validator_called_expected = _nested(expectations, ("validator", "called"))
    complete_expected = _nested(
        expectations, ("validator", "task_complete")
    )
    execution_expected = _nested(
        expectations, ("execution", "expected_status")
    )
    dispatches_expected = _nested(
        expectations, ("execution", "expected_vla_dispatches")
    )
    planner_status_expected = _nested(
        expectations, ("planner", "expected_status")
    )
    hri_mode_expected = _nested(
        expectations,
        ("hri", "accepted_response_modes"),
        ("hri", "response_mode"),
    )
    hri_outcome_expected = _nested(expectations, ("hri", "terminal_outcome"))
    recovery_expected = _nested(expectations, ("recovery",))
    history_delta_expected = _nested(
        expectations, ("history", "expected_terminal_record_delta")
    )
    preference_delta_expected = _nested(
        expectations,
        ("preference", "expected_record_delta_without_explicit_consent"),
    )

    task = result.get("task") if isinstance(result.get("task"), Mapping) else {}
    attempts = task.get("attempts") if isinstance(task, Mapping) else None
    last_attempt: Mapping[str, Any] = {}
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            if isinstance(attempt, Mapping):
                last_attempt = attempt
                break
    validation = _nested(
        result,
        ("validator",),
        ("validation",),
        default=_MISSING,
    )
    if validation is _MISSING:
        validation = last_attempt.get("validation", _MISSING)
    if validation is _MISSING and (
        "outcome" in result or "task_complete" in result
    ):
        validation = result
    validator_called_actual = isinstance(validation, Mapping)
    validator_actual = (
        validation.get("outcome", _MISSING)
        if isinstance(validation, Mapping)
        else None
    )
    complete_actual = (
        validation.get("task_complete", _MISSING)
        if isinstance(validation, Mapping)
        else None
    )
    execution = _nested(result, ("execution",), default=_MISSING)
    if execution is _MISSING:
        execution = last_attempt.get("execution", _MISSING)
    execution_actual = _nested(
        execution if isinstance(execution, Mapping) else {},
        ("execution", "status"),
        ("status",),
        default=_MISSING,
    )
    hri_mode_actual = _nested(result, ("hri", "mode"))
    hri_outcome_actual = _nested(
        result,
        ("task", "outcome"),
        ("hri", "report", "outcome"),
    )
    first_attempt: Mapping[str, Any] = {}
    if isinstance(attempts, list):
        first_attempt = next(
            (
                attempt
                for attempt in attempts
                if isinstance(attempt, Mapping)
                and (
                    "validation_assurance" in attempt
                    or "execution_assurance" in attempt
                )
            ),
            {},
        )
    planner_status_actual = _nested(
        first_attempt or last_attempt,
        ("plan", "planning_status"),
    )
    dispatches_actual: Any = _MISSING
    if isinstance(attempts, list):
        dispatches_actual = 0
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            execution_result = attempt.get("execution")
            if not isinstance(execution_result, Mapping):
                continue
            subtask_results = execution_result.get("subtask_results", [])
            if isinstance(subtask_results, list):
                dispatches_actual += len(subtask_results)
    recovery_actual = _nested(
        first_attempt or last_attempt,
        ("validation_assurance", "next_action"),
        ("execution_assurance", "next_action"),
        default=_MISSING,
    )
    if recovery_actual is _MISSING:
        recovery_actual = _nested(result, ("task", "next_action"))
    history_delta_actual = _nested(
        result,
        ("memory", "history_delta"),
        ("history_delta",),
    )
    if (
        result.get("history_checkpointed") is True
        and history_delta_expected == 1
        and (
            history_delta_actual is _MISSING
            or history_delta_actual == 0
        )
    ):
        # A MEMORY_CONFIRM response has durably checkpointed the completed
        # episode in the history outbox. The repository append happens when the
        # user answers the optional memory question, but the task record is
        # already recoverable and counts as this command's terminal record.
        history_delta_actual = 1
    preference_delta_actual = _nested(
        result,
        ("memory", "preference_delta"),
        ("preference_delta",),
    )
    if preference_delta_actual is _MISSING and "memory" in result:
        memory_result = result.get("memory")
        preference_delta_actual = (
            1
            if isinstance(memory_result, Mapping)
            and memory_result.get("committed") is True
            else 0
        )

    checks: list[dict[str, Any]] = []

    def add_check(name: str, expected: Any, actual: Any) -> None:
        if expected is _MISSING:
            return
        if actual is _MISSING:
            checks.append(
                {
                    "name": name,
                    "expected": expected,
                    "actual": None,
                    "evaluated": False,
                    "passed": None,
                }
            )
            return
        checks.append(
            {
                "name": name,
                "expected": expected,
                "actual": actual,
                "evaluated": True,
                "passed": actual == expected,
            }
        )

    if isinstance(hri_mode_expected, (list, tuple, set)):
        expected_modes = list(hri_mode_expected)
        checks.append(
            {
                "name": "hri_response_mode",
                "expected": expected_modes,
                "actual": (
                    None if hri_mode_actual is _MISSING else hri_mode_actual
                ),
                "evaluated": hri_mode_actual is not _MISSING,
                "passed": (
                    None
                    if hri_mode_actual is _MISSING
                    else hri_mode_actual in expected_modes
                ),
            }
        )
    else:
        add_check("hri_response_mode", hri_mode_expected, hri_mode_actual)
    add_check("hri_terminal_outcome", hri_outcome_expected, hri_outcome_actual)
    add_check(
        "validator_called",
        validator_called_expected,
        validator_called_actual,
    )
    add_check("validator_outcome", validator_expected, validator_actual)
    add_check("task_complete", complete_expected, complete_actual)
    add_check(
        "planner_status",
        planner_status_expected,
        planner_status_actual,
    )
    add_check("execution_status", execution_expected, execution_actual)
    add_check("vla_dispatches", dispatches_expected, dispatches_actual)
    add_check("recovery", recovery_expected, recovery_actual)
    add_check("history_delta", history_delta_expected, history_delta_actual)
    add_check(
        "preference_delta_without_consent",
        preference_delta_expected,
        preference_delta_actual,
    )

    evaluated = [check for check in checks if check.get("evaluated") is True]
    skipped = [check for check in checks if check.get("evaluated") is not True]
    return ScoreReport(
        scenario_id=scenario_id,
        passed=(
            bool(evaluated)
            and all(check["passed"] for check in evaluated)
            and (allow_partial or not skipped)
        ),
        checks=tuple(checks),
    )
