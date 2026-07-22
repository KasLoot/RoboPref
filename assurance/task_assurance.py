from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

from agents.contracts import ExecutionResult, PlanResult, ValidationResult, ValidationSpec


@dataclass(slots=True)
class TaskAssuranceResult:
    phase: str
    outcome: str
    proceed: bool
    next_action: str
    message: str
    failure: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(asdict(self))


class TaskAssurance:
    """Deterministic gates around probabilistic planning and validation."""

    def __init__(
        self,
        minimum_planner_confidence: float = 0.5,
        minimum_validator_confidence: float = 0.5,
    ):
        self.minimum_planner_confidence = minimum_planner_confidence
        self.minimum_validator_confidence = minimum_validator_confidence

    def assess_plan(self, plan: PlanResult) -> TaskAssuranceResult:
        status = plan.status.upper()
        if status == "READY":
            if plan.confidence < self.minimum_planner_confidence:
                return self._failure(
                    "PLANNING",
                    "UNKNOWN",
                    "LOW_PLANNER_CONFIDENCE",
                    "REOBSERVE",
                    "The plan is not sufficiently grounded to execute.",
                )
            unmet = [
                condition
                for condition in plan.preconditions
                if not isinstance(condition, dict) or condition.get("satisfied") is not True
            ]
            if unmet:
                return self._failure(
                    "PRECONDITION",
                    "BLOCKED",
                    "UNSATISFIED_PRECONDITION",
                    "USER_ASSIST",
                    "A required task precondition is not satisfied.",
                    observed=unmet,
                )
            malformed_subtasks = [
                subtask
                for subtask in plan.subtasks
                if not isinstance(subtask, dict)
                or not str(subtask.get("task_instruction", "")).strip()
            ]
            if not plan.subtasks or plan.validation_spec is None or malformed_subtasks:
                return self._failure(
                    "PLANNING",
                    "BLOCKED",
                    "INCOMPLETE_PLAN",
                    "REPLAN",
                    "The planner did not return executable subtasks and a validation specification.",
                )
            return TaskAssuranceResult(
                phase="PLANNING",
                outcome="READY",
                proceed=True,
                next_action="EXECUTE",
                message="The plan passed deterministic checks.",
            )
        if status == "ALREADY_SATISFIED":
            return TaskAssuranceResult(
                phase="PLANNING",
                outcome="ALREADY_SATISFIED",
                proceed=False,
                next_action="NONE",
                message="The requested state is already satisfied.",
            )
        mapping = {
            "BLOCKED": ("BLOCKED", "USER_ASSIST"),
            "UNSUPPORTED": ("BLOCKED", "ABORT_UNSUPPORTED"),
            "UNSAFE": ("ABORTED_SAFETY", "ABORT_SAFETY"),
            "UNKNOWN": ("UNKNOWN", "REOBSERVE"),
        }
        outcome, next_action = mapping.get(status, ("UNKNOWN", "REPLAN"))
        failure = plan.failure or {}
        return self._failure(
            str(failure.get("stage", "PLANNING")),
            outcome,
            str(failure.get("code", f"PLANNER_{status or 'UNKNOWN'}")),
            str(failure.get("recoverability", next_action)),
            str(failure.get("user_message", "The task could not be planned safely.")),
            observed=failure.get("observed"),
        )

    def assess_validation(
        self,
        result: ValidationResult,
        spec: ValidationSpec,
        *,
        attempt: int,
        max_replans: int,
    ) -> TaskAssuranceResult:
        if result.outcome == "UNSAFE":
            return self._failure(
                "SAFETY",
                "ABORTED_SAFETY",
                "UNSAFE_EXECUTION_STATE",
                "ABORT_SAFETY",
                result.user_message,
                observed=result.discrepancies,
            )
        if result.confidence < self.minimum_validator_confidence:
            return self._failure(
                "VALIDATION",
                "UNKNOWN",
                "LOW_VALIDATION_CONFIDENCE",
                "REOBSERVE",
                "The final observation is not clear enough to verify the task.",
            )
        expected_ids = {condition.id for condition in spec.goal_conditions}
        returned_ids = {
            str(check.get("goal_id"))
            for check in result.goal_checks
            if isinstance(check, dict)
        }
        if returned_ids != expected_ids or len(result.goal_checks) != len(expected_ids):
            return self._failure(
                "VALIDATION",
                "UNKNOWN",
                "INCOMPLETE_GOAL_CHECKS",
                "REOBSERVE",
                "The validator did not check the complete frozen validation schema.",
            )
        if result.outcome == "SUCCESS" and result.task_complete:
            checks_by_id = {
                str(check.get("goal_id")): check for check in result.goal_checks
            }
            required_ids = {
                condition.id for condition in spec.goal_conditions if condition.required
            }
            if all(
                checks_by_id[goal_id].get("satisfied") is True
                for goal_id in required_ids
            ):
                return TaskAssuranceResult(
                    phase="VALIDATION",
                    outcome="SUCCESS" if attempt == 1 else "SUCCESS_RECOVERED",
                    proceed=False,
                    next_action="NONE",
                    message="Task Complete.",
                )
        requested = result.recoverability or "REPLAN"
        next_action = requested
        if requested in {"REPLAN", "AUTO_LOCAL"} and attempt > max_replans:
            next_action = "USER_ASSIST"
        reported = {
            "PARTIAL": "PARTIAL",
            "FAILURE": "FAILED",
            "UNKNOWN": "UNKNOWN",
        }.get(result.outcome, "UNKNOWN")
        return self._failure(
            "VALIDATION",
            reported,
            str((result.failure or {}).get("code", "GOAL_NOT_SATISFIED")),
            next_action,
            result.user_message,
            observed=result.discrepancies,
        )

    def assess_execution(
        self, execution: ExecutionResult
    ) -> TaskAssuranceResult | None:
        """Return a terminal gate for non-validatable executor states."""
        status = execution.status.upper()
        if status in {
            "COMPLETED",
            "OBSERVED_RECORDED_ATTEMPT",
            "OBSERVATION_ONLY",
        }:
            return None
        if status == "UNSAFE":
            return self._failure(
                "SAFETY",
                "ABORTED_SAFETY",
                "VLA_UNSAFE",
                "ABORT_SAFETY",
                "Execution stopped because the VLA reported an unsafe state.",
                observed=execution.to_dict(),
            )
        if status in {"ABORTED", "CANCELLED"}:
            return self._failure(
                "EXECUTION",
                "ABORTED",
                f"VLA_{status}",
                "USER_ASSIST",
                "Execution stopped before the requested state was reached.",
                observed=execution.to_dict(),
            )
        return self._failure(
            "EXECUTION",
            "FAILED",
            "VLA_EXECUTION_FAILED",
            "USER_ASSIST",
            execution.error or "The VLA did not complete the requested actions.",
            observed=execution.to_dict(),
        )

    def runtime_failure(
        self,
        *,
        stage: str,
        code: str,
        message: str,
        next_action: str,
        observed: Any = None,
    ) -> TaskAssuranceResult:
        return self._failure(
            stage,
            "UNKNOWN",
            code,
            next_action,
            message,
            observed=observed,
        )

    @staticmethod
    def _failure(
        phase: str,
        outcome: str,
        code: str,
        next_action: str,
        message: str,
        *,
        observed: Any = None,
    ) -> TaskAssuranceResult:
        return TaskAssuranceResult(
            phase=phase,
            outcome=outcome,
            proceed=False,
            next_action=next_action,
            message=message,
            failure={
                "stage": phase,
                "code": code,
                "observed": copy.deepcopy(observed),
                "next_action": next_action,
                "user_message": message,
                "memory_effect": "HISTORY_ONLY",
            },
        )
