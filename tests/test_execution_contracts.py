from __future__ import annotations

import unittest

from prefmem.contracts import (
    CriterionState,
    FailureKind,
    GoalProposal,
    MonitorAssessment,
    PlanStatus,
    PlannerDecision,
    PlannerPlan,
    PlannerStep,
    TaskStatus,
    assign_plan_ids,
)


class ExecutionContractTests(unittest.TestCase):
    def test_planner_plan_is_frozen_and_host_assigns_all_ids(self) -> None:
        expected = ["The book is inside the blue zone."]
        failures = ["The book has fallen off the table."]
        step = PlannerStep(
            instruction="Place the book in the blue zone.",
            expected_observation=expected,
            known_failure_conditions=failures,
        )
        expected.append("Caller mutation must not leak into the plan.")
        failures.clear()
        draft = PlannerPlan(
            status=PlanStatus.READY,
            steps=[step],
            final_expected_observation=[
                "The book is resting inside the blue zone."
            ],
        )

        plan = assign_plan_ids(draft, plan_id="tidy-42", revision=3)

        self.assertEqual(
            plan.steps[0].expected_observation[0].criterion_id,
            "tidy-42:r3:s1:c1",
        )
        self.assertEqual(
            plan.final_expected_observation[0].criterion_id,
            "tidy-42:r3:final:c1",
        )
        self.assertEqual(len(plan.steps[0].expected_observation), 1)
        self.assertEqual(
            plan.steps[0].known_failure_conditions,
            ("The book has fallen off the table.",),
        )

    def test_step_requires_one_to_three_success_criteria(self) -> None:
        with self.assertRaisesRegex(ValueError, "1 to 3"):
            PlannerStep("Move the book.", [])
        with self.assertRaisesRegex(ValueError, "1 to 3"):
            PlannerStep(
                "Move the book.",
                ["one", "two", "three", "four"],
            )

    def test_plan_status_invariants_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one step"):
            PlannerPlan(
                status="READY",
                final_expected_observation=["The table is tidy."],
            )
        with self.assertRaisesRegex(ValueError, "blocked_reason"):
            PlannerPlan(status="BLOCKED")

        plan = PlannerPlan.from_dict(
            {
                "status": "ALREADY_SATISFIED",
                "steps": [],
                "final_expected_observation": ["The table is tidy."],
                "blocked_reason": None,
            }
        )
        self.assertIs(plan.status, PlanStatus.ALREADY_SATISFIED)

    def test_non_ready_goal_proposal_cannot_smuggle_nominal_tasks(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot contain nominal tasks"):
            GoalProposal(
                status="BLOCKED",
                goal="Move the book.",
                final_expected_observation=(),
                nominal_tasks=("Move it anyway.",),
                reason="The book is not visible.",
            )

    def test_planner_decision_rejects_fields_from_another_branch(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot include user_question"):
            PlannerDecision(
                decision="BLOCKED",
                blocked_reason="The destination is inaccessible.",
                user_question="Should I use another destination?",
            )

    def test_model_boundaries_reject_unknown_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown fields: explanation"):
            PlannerPlan.from_dict(
                {
                    "status": "BLOCKED",
                    "steps": [],
                    "final_expected_observation": [],
                    "blocked_reason": "The target is not visible.",
                    "explanation": "This field is outside the schema.",
                }
            )
        with self.assertRaisesRegex(ValueError, "unknown fields: confidence"):
            MonitorAssessment.from_model_output(
                {
                    "task_status": "ONGOING",
                    "criteria": [{"id": "p:r1:s1:c1", "state": "UNKNOWN"}],
                    "failure": None,
                    "observation": "The target is temporarily occluded.",
                    "confidence": 0.8,
                },
                plan_id="p",
                revision=1,
                step_id="p:r1:s1",
                observed_at=2.0,
            )

    def test_monitor_accepts_an_open_world_unexpected_failure(self) -> None:
        assessment = MonitorAssessment.from_model_output(
            {
                "task_status": "FAIL",
                "criteria": [{"id": "plan:r1:s1:c1", "state": "UNKNOWN"}],
                "failure": {
                    "kind": "UNEXPECTED",
                    "description": "Water is visibly spilling onto the laptop.",
                },
                "observation": "Water is spreading across the work surface.",
            },
            plan_id="plan",
            revision=1,
            step_id="plan:r1:s1",
            observed_at=10.0,
        )

        self.assertIs(assessment.task_status, TaskStatus.FAIL)
        self.assertIs(assessment.failure.kind, FailureKind.UNEXPECTED)
        self.assertEqual(
            assessment.to_model_dict()["failure"]["kind"],
            "UNEXPECTED",
        )

    def test_monitor_success_requires_reported_criteria_to_be_met(self) -> None:
        with self.assertRaisesRegex(ValueError, "every reported criterion"):
            MonitorAssessment.from_model_output(
                {
                    "task_status": "SUCCESS",
                    "criteria": [{"id": "p:r1:s1:c1", "state": "NOT_MET"}],
                    "failure": None,
                    "observation": "The object remains outside the zone.",
                },
                plan_id="p",
                revision=1,
                step_id="p:r1:s1",
                observed_at=1.0,
            )

    def test_monitor_ongoing_supports_unknown_without_failure(self) -> None:
        assessment = MonitorAssessment.from_model_output(
            {
                "task_status": "ONGOING",
                "criteria": [{"id": "p:r1:s1:c1", "state": "UNKNOWN"}],
                "failure": None,
                "observation": "The target is temporarily occluded.",
            },
            plan_id="p",
            revision=1,
            step_id="p:r1:s1",
            observed_at=2.0,
        )

        self.assertIs(assessment.criteria[0].state, CriterionState.UNKNOWN)

    def test_monitor_observation_must_be_one_sentence(self) -> None:
        with self.assertRaisesRegex(ValueError, "one sentence"):
            MonitorAssessment.from_model_output(
                {
                    "task_status": "ONGOING",
                    "criteria": [{"id": "p:r1:s1:c1", "state": "UNKNOWN"}],
                    "failure": None,
                    "observation": "The target is hidden. The camera is moving.",
                },
                plan_id="p",
                revision=1,
                step_id="p:r1:s1",
                observed_at=2.0,
            )


if __name__ == "__main__":
    unittest.main()
