import json
import unittest

from assurance.task_assurance import TaskAssurance


def ready_plan():
    return {
        "planning_status": "READY",
        "task_complete": False,
        "subtasks": [{"task_instruction": "Place the red block on the green block."}],
        "planner_confidence": 0.9,
    }


class TaskAssuranceTests(unittest.TestCase):
    def setUp(self):
        self.assurance = TaskAssurance(minimum_validator_confidence=0.6)

    def test_ready_plan_proceeds_to_validation(self):
        result = self.assurance.assess_plan(json.dumps(ready_plan()))
        self.assertTrue(result.proceed)
        self.assertEqual(result.outcome, "READY")

    def test_missing_blocks_blocks_dispatch_and_requests_user_help(self):
        plan = {
            "planning_status": "BLOCKED",
            "task_complete": False,
            "subtasks": [],
            "planner_confidence": 0.96,
            "failure": {
                "stage": "PRECONDITION",
                "code": "MISSING_REQUIRED_OBJECT",
                "expected": "At least two visible blocks.",
                "observed": "No blocks are visible.",
                "recoverability": "USER_ASSIST",
                "user_message": "I cannot see any blocks; please place them in view.",
            },
        }
        result = self.assurance.assess_plan(plan)
        self.assertFalse(result.proceed)
        self.assertEqual(result.outcome, "BLOCKED")
        self.assertEqual(result.next_action, "USER_ASSIST")
        self.assertEqual(result.failure["code"], "MISSING_REQUIRED_OBJECT")

    def test_unsafe_plan_aborts_without_retry(self):
        result = self.assurance.assess_plan(
            {
                "planning_status": "UNSAFE",
                "subtasks": [],
                "failure": {"code": "UNSAFE_PLAN", "recoverability": "ABORT_SAFETY"},
            }
        )
        self.assertEqual(result.outcome, "ABORTED_SAFETY")
        self.assertEqual(result.next_action, "ABORT_SAFETY")

    def test_unsupported_task_is_not_sent_to_vla(self):
        result = self.assurance.assess_plan(
            {
                "planning_status": "UNSUPPORTED",
                "subtasks": [],
                "failure": {"code": "SKILL_UNAVAILABLE"},
            }
        )
        self.assertFalse(result.proceed)
        self.assertEqual(result.outcome, "BLOCKED")
        self.assertEqual(result.next_action, "ABORT_UNSUPPORTED")

    def test_already_satisfied_requires_no_action(self):
        result = self.assurance.assess_plan(
            {"planning_status": "ALREADY_SATISFIED", "task_complete": True, "subtasks": []}
        )
        self.assertEqual(result.outcome, "ALREADY_SATISFIED")
        self.assertEqual(result.next_action, "NONE")

    def test_malformed_planner_output_is_unknown_not_executable(self):
        result = self.assurance.assess_plan("not JSON")
        self.assertFalse(result.proceed)
        self.assertEqual(result.outcome, "UNKNOWN")
        self.assertEqual(result.failure["code"], "MALFORMED_PLANNER_OUTPUT")

    def test_ready_label_cannot_bypass_unsatisfied_precondition(self):
        plan = ready_plan()
        plan["preconditions"] = [
            {"condition": "A blue block is visible.", "satisfied": False, "evidence": "No blue block."}
        ]
        result = self.assurance.assess_plan(plan)
        self.assertFalse(result.proceed)
        self.assertEqual(result.outcome, "BLOCKED")
        self.assertEqual(result.failure["code"], "UNSATISFIED_PRECONDITION")

    def test_low_confidence_ready_plan_is_not_dispatched(self):
        plan = ready_plan()
        plan["planner_confidence"] = 0.3
        result = self.assurance.assess_plan(plan)
        self.assertFalse(result.proceed)
        self.assertEqual(result.outcome, "UNKNOWN")
        self.assertEqual(result.next_action, "REOBSERVE")

    def test_verified_final_state_is_success(self):
        result = self.assurance.assess_validation(
            {
                "outcome": "SUCCESS",
                "task_complete": True,
                "validator_confidence": 0.94,
                "discrepancies": [],
            }
        )
        self.assertEqual(result.outcome, "SUCCESS")
        self.assertEqual(result.next_action, "NONE")

    def test_success_after_logged_retry_is_success_recovered(self):
        result = self.assurance.assess_validation(
            {
                "outcome": "SUCCESS",
                "task_complete": True,
                "validator_confidence": 0.9,
            },
            recovery_attempts=1,
        )
        self.assertEqual(result.outcome, "SUCCESS_RECOVERED")

    def test_success_label_without_completion_flag_is_not_accepted(self):
        result = self.assurance.assess_validation(
            {"outcome": "SUCCESS", "validator_confidence": 0.9}
        )
        self.assertEqual(result.outcome, "UNKNOWN")
        self.assertEqual(result.failure["code"], "INCONSISTENT_VALIDATOR_OUTPUT")

    def test_partial_completion_replans_only_unmet_goal(self):
        result = self.assurance.assess_validation(
            {
                "outcome": "PARTIAL",
                "task_complete": False,
                "expected_state": "RGB stack",
                "observed_state": "Red and green stacked; blue remains separate.",
                "discrepancies": ["Blue is not on top."],
                "recoverability": "REPLAN",
                "validator_confidence": 0.9,
            }
        )
        self.assertEqual(result.outcome, "PARTIAL")
        self.assertEqual(result.next_action, "REPLAN")
        self.assertEqual(result.failure["memory_effect"], "NONE")

    def test_low_confidence_final_view_triggers_reobservation(self):
        result = self.assurance.assess_validation(
            {
                "outcome": "FAILURE",
                "task_complete": False,
                "validator_confidence": 0.35,
            }
        )
        self.assertEqual(result.outcome, "UNKNOWN")
        self.assertEqual(result.next_action, "REOBSERVE")

    def test_unrecoverable_execution_failure_requests_user_assistance(self):
        result = self.assurance.assess_validation(
            {
                "outcome": "FAILURE",
                "task_complete": False,
                "validator_confidence": 0.91,
                "failure": {
                    "code": "OBJECT_UNREACHABLE",
                    "recoverability": "USER_ASSIST",
                    "observed": "The blue block is outside the reachable workspace.",
                },
            }
        )
        self.assertEqual(result.outcome, "FAILED")
        self.assertEqual(result.next_action, "USER_ASSIST")

    def test_exhausted_local_retry_escalates_to_replan_then_user_help(self):
        validator = {
            "outcome": "FAILURE",
            "task_complete": False,
            "validator_confidence": 0.9,
            "failure": {"code": "GRASP_MISSED", "recoverability": "AUTO_LOCAL"},
        }
        replan = self.assurance.assess_validation(
            validator,
            attempt_counts={"auto_local": 1, "replan": 0},
        )
        user_help = self.assurance.assess_validation(
            validator,
            attempt_counts={"auto_local": 1, "replan": 1},
        )
        self.assertEqual(replan.next_action, "REPLAN")
        self.assertEqual(user_help.next_action, "USER_ASSIST")

    def test_exhausted_reobservation_does_not_loop(self):
        result = self.assurance.assess_validation(
            {
                "outcome": "UNKNOWN",
                "task_complete": False,
                "validator_confidence": 0.3,
            },
            attempt_counts={"reobserve": 1},
        )
        self.assertEqual(result.next_action, "USER_ASSIST")


if __name__ == "__main__":
    unittest.main()
