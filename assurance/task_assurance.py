from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


PLANNING_STATUSES = {
    "READY",
    "ALREADY_SATISFIED",
    "BLOCKED",
    "UNSUPPORTED",
    "UNSAFE",
    "UNKNOWN",
}
RECOVERY_ACTIONS = {
    "NONE",
    "REOBSERVE",
    "AUTO_LOCAL",
    "REPLAN",
    "USER_ASSIST",
    "ABORT_UNSUPPORTED",
    "ABORT_SAFETY",
}


@dataclass(frozen=True)
class TaskAssuranceResult:
    phase: str
    outcome: str
    proceed: bool
    next_action: str
    message: str
    failure: dict[str, Any] | None = None
    model_output: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskAssurance:
    """Apply deterministic gates to probabilistic planner/validator JSON.

    This class recommends recovery actions but never claims that a physical retry was
    executed. A robot executive must perform and log any recommended action.
    """

    def __init__(
        self,
        minimum_validator_confidence: float = 0.6,
        minimum_planner_confidence: float = 0.6,
        maximum_local_retries: int = 1,
        maximum_replans: int = 1,
        maximum_reobservations: int = 1,
    ):
        self.minimum_validator_confidence = minimum_validator_confidence
        self.minimum_planner_confidence = minimum_planner_confidence
        self.maximum_local_retries = maximum_local_retries
        self.maximum_replans = maximum_replans
        self.maximum_reobservations = maximum_reobservations

    def bounded_recovery(self, requested_action: str, attempts: dict[str, int] | None = None) -> str:
        """Escalate a recovery request when the prototype retry budget is exhausted."""
        action = self._valid_recovery(requested_action, "USER_ASSIST")
        counts = attempts or {}
        if action in {"NONE", "USER_ASSIST", "ABORT_UNSUPPORTED", "ABORT_SAFETY"}:
            return action
        if action == "REOBSERVE" and counts.get("reobserve", 0) >= self.maximum_reobservations:
            return "USER_ASSIST"
        if action == "REPLAN" and counts.get("replan", 0) >= self.maximum_replans:
            return "USER_ASSIST"
        if action == "AUTO_LOCAL" and counts.get("auto_local", 0) >= self.maximum_local_retries:
            if counts.get("replan", 0) < self.maximum_replans:
                return "REPLAN"
            return "USER_ASSIST"
        return action

    def runtime_failure(
        self,
        *,
        stage: str,
        code: str,
        message: str,
        next_action: str,
        observed: str,
    ) -> TaskAssuranceResult:
        """Convert a component exception/timeout into a non-success outcome."""
        recovery = self._valid_recovery(next_action, "REOBSERVE")
        outcome = "ABORTED_SAFETY" if recovery == "ABORT_SAFETY" else "UNKNOWN"
        return self._failure_result(
            phase=stage,
            outcome=outcome,
            code=code,
            expected=f"The {stage.lower()} component returns structured evidence.",
            observed=observed,
            recoverability=recovery,
            next_action=recovery,
            message=message,
            severity="HIGH" if recovery == "ABORT_SAFETY" else "MEDIUM",
            safe_state="No further task action was dispatched by RoboPref.",
        )

    def assess_plan(self, planner_output: Any) -> TaskAssuranceResult:
        payload = self._parse_json_object(planner_output)
        if payload is None:
            return self._failure_result(
                phase="PLANNING",
                outcome="UNKNOWN",
                code="MALFORMED_PLANNER_OUTPUT",
                expected="A planner JSON object that satisfies the planning schema.",
                observed="No parseable planner JSON object was returned.",
                recoverability="REPLAN",
                next_action="REPLAN",
                message="I could not verify a valid plan, so I will not dispatch it.",
            )

        status = str(payload.get("planning_status", "")).upper()
        if not status:
            # Backward compatibility with the original planner schema.
            if payload.get("task_complete") is True:
                status = "ALREADY_SATISFIED"
            elif isinstance(payload.get("subtasks"), list) and payload["subtasks"]:
                status = "READY"
            else:
                status = "UNKNOWN"
        if status not in PLANNING_STATUSES:
            return self._failure_result(
                phase="PLANNING",
                outcome="UNKNOWN",
                code="INVALID_PLANNING_STATUS",
                expected=f"One of {sorted(PLANNING_STATUSES)}.",
                observed=status,
                recoverability="REPLAN",
                next_action="REPLAN",
                message="The planner returned an unknown status, so execution is paused.",
                model_output=payload,
            )

        if status == "READY":
            planner_confidence = self._confidence(payload.get("planner_confidence"))
            if planner_confidence < self.minimum_planner_confidence:
                return self._failure_result(
                    phase="PLANNING",
                    outcome="UNKNOWN",
                    code="LOW_PLANNER_CONFIDENCE",
                    expected=f"Planner confidence of at least {self.minimum_planner_confidence:.2f}.",
                    observed=f"Planner confidence was {planner_confidence:.2f}.",
                    recoverability="REOBSERVE",
                    next_action="REOBSERVE",
                    message="The scene or plan is not clear enough to dispatch safely.",
                    confidence=planner_confidence,
                    model_output=payload,
                )
            preconditions = payload.get("preconditions")
            if isinstance(preconditions, list):
                unmet = [
                    item
                    for item in preconditions
                    if isinstance(item, dict) and item.get("satisfied") is not True
                ]
                if unmet:
                    return self._failure_result(
                        phase="PRECONDITION",
                        outcome="BLOCKED",
                        code="UNSATISFIED_PRECONDITION",
                        expected="Every declared task precondition must be satisfied.",
                        observed=json.dumps(unmet, sort_keys=True),
                        recoverability="USER_ASSIST",
                        next_action="USER_ASSIST",
                        message="A declared task precondition is not satisfied, so the plan was not dispatched.",
                        confidence=planner_confidence,
                        model_output=payload,
                    )
            subtasks = payload.get("subtasks")
            if not isinstance(subtasks, list) or not subtasks:
                return self._failure_result(
                    phase="PLANNING",
                    outcome="BLOCKED",
                    code="NO_FEASIBLE_PLAN",
                    expected="At least one concrete subtask for an incomplete task.",
                    observed="The task is incomplete but the plan has no subtasks.",
                    recoverability="REPLAN",
                    next_action="REPLAN",
                    message="The task is not complete, but no executable plan was found.",
                    model_output=payload,
                )
            return TaskAssuranceResult(
                phase="PLANNING",
                outcome="READY",
                proceed=True,
                next_action="NONE",
                message="The plan passed precondition and structure checks.",
                model_output=payload,
            )

        if status == "ALREADY_SATISFIED":
            return TaskAssuranceResult(
                phase="PLANNING",
                outcome="ALREADY_SATISFIED",
                proceed=False,
                next_action="NONE",
                message="The requested goal is already visibly satisfied.",
                model_output=payload,
            )

        failure = self._normalise_failure(payload.get("failure"), phase="PRECONDITION")
        code = failure.get("code") or {
            "BLOCKED": "PRECONDITION_BLOCKED",
            "UNSUPPORTED": "SKILL_UNAVAILABLE",
            "UNSAFE": "UNSAFE_PLAN",
            "UNKNOWN": "INSUFFICIENT_PLANNING_EVIDENCE",
        }[status]
        default_recovery = {
            "BLOCKED": "USER_ASSIST",
            "UNSUPPORTED": "ABORT_UNSUPPORTED",
            "UNSAFE": "ABORT_SAFETY",
            "UNKNOWN": "REOBSERVE",
        }[status]
        recoverability = self._valid_recovery(failure.get("recoverability"), default_recovery)
        outcome = (
            "ABORTED_SAFETY"
            if status == "UNSAFE"
            else "BLOCKED"
            if status in {"BLOCKED", "UNSUPPORTED"}
            else "UNKNOWN"
        )
        message = failure.get("user_message") or {
            "BLOCKED": "A required task precondition is not satisfied.",
            "UNSUPPORTED": "The requested task is outside the available robot skills.",
            "UNSAFE": "The plan was stopped because it may be unsafe.",
            "UNKNOWN": "There is not enough evidence to dispatch this plan.",
        }[status]
        return self._failure_result(
            phase=failure.get("stage") or ("SAFETY" if status == "UNSAFE" else "PRECONDITION"),
            outcome=outcome,
            code=code,
            expected=failure.get("expected", "All task preconditions must be satisfied."),
            observed=failure.get("observed", "The required state was not verified."),
            recoverability=recoverability,
            next_action=recoverability,
            message=message,
            confidence=failure.get("confidence", payload.get("planner_confidence", 0.0)),
            severity=failure.get("severity", "CRITICAL" if status == "UNSAFE" else "MEDIUM"),
            safe_state=failure.get("safe_state", "No task action was dispatched."),
            model_output=payload,
        )

    def assess_validation(
        self,
        validator_output: Any,
        *,
        recovery_attempts: int = 0,
        attempt_counts: dict[str, int] | None = None,
    ) -> TaskAssuranceResult:
        payload = self._parse_json_object(validator_output)
        if payload is None:
            next_action = self.bounded_recovery("REOBSERVE", attempt_counts)
            return self._failure_result(
                phase="VALIDATION",
                outcome="UNKNOWN",
                code="MALFORMED_VALIDATOR_OUTPUT",
                expected="A validator JSON object that satisfies the outcome schema.",
                observed="No parseable validator JSON object was returned.",
                recoverability="REOBSERVE",
                next_action=next_action,
                message="I cannot verify the result from the validator output.",
            )

        confidence = self._confidence(payload.get("validator_confidence"))
        if confidence < self.minimum_validator_confidence:
            next_action = self.bounded_recovery("REOBSERVE", attempt_counts)
            return self._failure_result(
                phase="VALIDATION",
                outcome="UNKNOWN",
                code="LOW_VALIDATION_CONFIDENCE",
                expected=f"Validation confidence of at least {self.minimum_validator_confidence:.2f}.",
                observed=f"Validation confidence was {confidence:.2f}.",
                recoverability="REOBSERVE",
                next_action=next_action,
                message="The final view is not clear enough to verify success.",
                confidence=confidence,
                model_output=payload,
            )

        outcome = str(payload.get("outcome", "")).upper()
        if not outcome:
            outcome = "SUCCESS" if payload.get("task_complete") is True else "FAILURE"
        if outcome == "SUCCESS":
            if payload.get("task_complete") is True:
                verified_outcome = "SUCCESS_RECOVERED" if recovery_attempts > 0 else "SUCCESS"
                return TaskAssuranceResult(
                    phase="VALIDATION",
                    outcome=verified_outcome,
                    proceed=False,
                    next_action="NONE",
                    message="The final frame satisfies every visible task requirement.",
                    model_output=payload,
                )
            next_action = self.bounded_recovery("REOBSERVE", attempt_counts)
            return self._failure_result(
                phase="VALIDATION",
                outcome="UNKNOWN",
                code="INCONSISTENT_VALIDATOR_OUTPUT",
                expected="SUCCESS must be accompanied by task_complete=true.",
                observed=f"task_complete={payload.get('task_complete')!r}.",
                recoverability="REOBSERVE",
                next_action=next_action,
                message="The validator result is internally inconsistent, so success was not accepted.",
                confidence=confidence,
                model_output=payload,
            )

        if outcome not in {"PARTIAL", "FAILURE", "UNKNOWN", "UNSAFE"}:
            outcome = "UNKNOWN"

        failure = self._normalise_failure(payload.get("failure"), phase="VALIDATION")
        default_recovery = {
            "PARTIAL": "REPLAN",
            "FAILURE": "REPLAN",
            "UNKNOWN": "REOBSERVE",
            "UNSAFE": "ABORT_SAFETY",
        }[outcome]
        recoverability = self._valid_recovery(
            failure.get("recoverability") or payload.get("recoverability"),
            default_recovery,
        )
        next_action = self.bounded_recovery(recoverability, attempt_counts)
        reported_outcome = {
            "PARTIAL": "PARTIAL",
            "FAILURE": "FAILED",
            "UNKNOWN": "UNKNOWN",
            "UNSAFE": "ABORTED_SAFETY",
        }[outcome]
        discrepancies = payload.get("discrepancies")
        observed = failure.get("observed") or payload.get("observed_state") or "The final goal was not verified."
        expected = failure.get("expected") or payload.get("expected_state") or "All task goal conditions."
        if isinstance(discrepancies, list) and discrepancies:
            observed = f"{observed} Discrepancies: {'; '.join(map(str, discrepancies))}"
        message = failure.get("user_message") or {
            "PARTIAL": "The task is only partially complete; a bounded corrective plan is recommended.",
            "FAILURE": "The final state does not satisfy the requested task.",
            "UNKNOWN": "The final state cannot be determined reliably.",
            "UNSAFE": "Execution was stopped because the observed state may be unsafe.",
        }[outcome]
        return self._failure_result(
            phase=failure.get("stage", "SAFETY" if outcome == "UNSAFE" else "VALIDATION"),
            outcome=reported_outcome,
            code=failure.get("code") or {
                "PARTIAL": "GOAL_PARTIALLY_SATISFIED",
                "FAILURE": "GOAL_NOT_SATISFIED",
                "UNKNOWN": "INSUFFICIENT_VISUAL_EVIDENCE",
                "UNSAFE": "UNSAFE_EXECUTION_STATE",
            }[outcome],
            expected=expected,
            observed=observed,
            recoverability=recoverability,
            next_action=next_action,
            message=message,
            confidence=confidence,
            severity=failure.get("severity", "CRITICAL" if outcome == "UNSAFE" else "MEDIUM"),
            safe_state=failure.get("safe_state", "Robot state must be checked before another action."),
            attempts=recovery_attempts,
            model_output=payload,
        )

    @staticmethod
    def _parse_json_object(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return None
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end < start:
            return None
        try:
            payload = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _normalise_failure(value: Any, *, phase: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"stage": phase}
        result = dict(value)
        result.setdefault("stage", phase)
        return result

    @staticmethod
    def _valid_recovery(value: Any, default: str) -> str:
        candidate = str(value or "").upper()
        return candidate if candidate in RECOVERY_ACTIONS else default

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _failure_result(
        cls,
        *,
        phase: str,
        outcome: str,
        code: str,
        expected: str,
        observed: str,
        recoverability: str,
        next_action: str,
        message: str,
        confidence: Any = 0.0,
        severity: str = "MEDIUM",
        safe_state: str = "Robot state has not been independently verified.",
        attempts: int = 0,
        model_output: dict[str, Any] | None = None,
    ) -> TaskAssuranceResult:
        failure = {
            "stage": phase,
            "code": code,
            "expected": str(expected),
            "observed": str(observed),
            "confidence": cls._confidence(confidence),
            "severity": str(severity).upper(),
            "recoverability": recoverability,
            "safe_state": str(safe_state),
            "attempts": int(attempts),
            "next_action": next_action,
            "user_message": message,
            "memory_effect": "NONE",
        }
        return TaskAssuranceResult(
            phase=phase,
            outcome=outcome,
            proceed=False,
            next_action=next_action,
            message=message,
            failure=failure,
            model_output=model_output,
        )
