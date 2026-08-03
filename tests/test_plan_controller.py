from __future__ import annotations

import threading
import unittest

from prefmem.contracts import (
    ExecutionPlan,
    MonitorAssessment,
    PlanStatus,
    PlannerPlan,
    PlannerStep,
    PublishedTask,
    TaskPhase,
    assign_plan_ids,
)
from prefmem.controller import (
    AssessmentResult,
    ConfirmationRequiredError,
    ControllerState,
    PlanController,
    PlanMismatchError,
)


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> float:
        self.value += seconds
        return self.value


def ready_plan(*, plan_id: str = "plan", revision: int = 1) -> ExecutionPlan:
    return assign_plan_ids(
        PlannerPlan(
            status=PlanStatus.READY,
            steps=(
                PlannerStep(
                    instruction="Place the book in the left zone.",
                    expected_observation=(
                        "The book is entirely inside the left zone.",
                        "The book is resting and no longer held.",
                    ),
                    known_failure_conditions=(
                        "The book has fallen off the table.",
                    ),
                ),
                PlannerStep(
                    instruction="Place the laptop in the right zone.",
                    expected_observation=(
                        "The laptop is entirely inside the right zone.",
                    ),
                ),
            ),
            final_expected_observation=(
                "The book is in the left zone.",
                "The laptop is in the right zone.",
            ),
        ),
        plan_id=plan_id,
        revision=revision,
    )


def assessment(
    task: PublishedTask,
    *,
    observed_at: float,
    status: str,
    criterion_state: str = "MET",
    failure_kind: str = "UNEXPECTED",
) -> MonitorAssessment:
    failure = None
    if status == "FAIL":
        failure = {
            "kind": failure_kind,
            "description": "The target is visibly inaccessible.",
        }
    return MonitorAssessment.from_model_output(
        {
            "task_status": status,
            "criteria": [
                {"id": criterion.criterion_id, "state": criterion_state}
                for criterion in task.expected_observation
            ],
            "failure": failure,
            "observation": "The current scene provides visible evidence.",
        },
        plan_id=task.plan_id,
        revision=task.revision,
        step_id=task.step_id,
        observed_at=observed_at,
    )


class PlanControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.published: list[PublishedTask] = []
        self.monitored: list[PublishedTask] = []
        self.controller = PlanController(
            self.published.append,
            self.monitored.append,
            success_stability_seconds=2.0,
            ongoing_timeout_seconds=10.0,
            clock=self.clock,
        )
        self.plan = ready_plan()

    def confirm(self) -> PublishedTask:
        self.controller.stage_plan(self.plan)
        return self.controller.confirm_and_publish(
            self.plan,
            confirmed=True,
        )

    def assert_success_after_stability(self, task: PublishedTask) -> AssessmentResult:
        first = assessment(
            task,
            observed_at=self.clock.value,
            status="SUCCESS",
        )
        self.assertIs(
            self.controller.record_assessment(first),
            AssessmentResult.ACCEPTED,
        )
        self.clock.advance(1.0)
        too_soon = assessment(
            task,
            observed_at=self.clock.value,
            status="SUCCESS",
        )
        self.assertIs(
            self.controller.record_assessment(too_soon),
            AssessmentResult.ACCEPTED,
        )
        self.clock.advance(1.0)
        stable = assessment(
            task,
            observed_at=self.clock.value,
            status="SUCCESS",
        )
        return self.controller.record_assessment(stable)

    def test_confirmation_freezes_exact_plan_and_precedes_publication(self) -> None:
        snapshot = self.controller.stage_plan(self.plan)

        self.assertIs(snapshot.state, ControllerState.AWAITING_CONFIRMATION)
        self.assertEqual(self.published, [])
        with self.assertRaises(ConfirmationRequiredError):
            self.controller.confirm_and_publish(self.plan, confirmed=False)
        different_revision = ready_plan(revision=2)
        with self.assertRaises(PlanMismatchError):
            self.controller.confirm_and_publish(
                different_revision,
                confirmed=True,
            )

        task = self.controller.confirm_and_publish(self.plan, confirmed=True)

        self.assertEqual(self.published, [task])
        self.assertEqual(self.monitored, [task])
        self.assertIs(task.phase, TaskPhase.STEP)
        self.assertEqual(task.step_id, self.plan.steps[0].step_id)
        self.assertIs(self.controller.snapshot.state, ControllerState.EXECUTING)

    def test_stale_identity_and_prepublication_frame_are_ignored(self) -> None:
        task = self.confirm()
        wrong_step = MonitorAssessment.from_model_output(
            {
                "task_status": "SUCCESS",
                "criteria": [
                    {"id": criterion.criterion_id, "state": "MET"}
                    for criterion in task.expected_observation
                ],
                "failure": None,
                "observation": "The book appears to be correctly placed.",
            },
            plan_id=task.plan_id,
            revision=task.revision,
            step_id="another-step",
            observed_at=self.clock.value,
        )
        old_frame = assessment(
            task,
            observed_at=task.published_at - 0.1,
            status="SUCCESS",
        )

        self.assertIs(
            self.controller.record_assessment(wrong_step),
            AssessmentResult.IGNORED_STALE,
        )
        self.assertIs(
            self.controller.record_assessment(old_frame),
            AssessmentResult.IGNORED_STALE,
        )
        self.assertEqual(self.controller.snapshot.consecutive_successes, 0)

    def test_reusing_a_plan_identity_requires_a_new_revision(self) -> None:
        self.controller.stage_plan(self.plan)

        with self.assertRaisesRegex(ValueError, "newer revision"):
            self.controller.stage_plan(self.plan)

        replacement = ready_plan(revision=2)
        snapshot = self.controller.stage_plan(replacement)
        self.assertEqual(snapshot.plan.revision, 2)

    def test_concurrent_duplicate_assessments_count_only_once(self) -> None:
        task = self.confirm()
        duplicate = assessment(task, observed_at=100.0, status="SUCCESS")
        results: list[AssessmentResult] = []
        result_lock = threading.Lock()

        def deliver() -> None:
            result = self.controller.record_assessment(duplicate)
            with result_lock:
                results.append(result)

        threads = [threading.Thread(target=deliver) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(results.count(AssessmentResult.ACCEPTED), 1)
        self.assertEqual(results.count(AssessmentResult.IGNORED_STALE), 19)
        self.assertEqual(self.controller.snapshot.consecutive_successes, 1)

    def test_success_advances_steps_then_requires_final_validation(self) -> None:
        first_task = self.confirm()

        self.assertIs(
            self.assert_success_after_stability(first_task),
            AssessmentResult.STEP_ADVANCED,
        )
        second_task = self.published[-1]
        self.assertEqual(second_task.step_id, self.plan.steps[1].step_id)
        self.assertIs(self.controller.snapshot.state, ControllerState.EXECUTING)

        self.assertIs(
            self.assert_success_after_stability(second_task),
            AssessmentResult.FINAL_VALIDATION_STARTED,
        )
        validation_task = self.published[-1]
        self.assertIs(validation_task.phase, TaskPhase.FINAL_VALIDATION)
        self.assertEqual(
            validation_task.step_id,
            self.plan.final_validation_step_id,
        )
        self.assertIs(
            self.controller.snapshot.state,
            ControllerState.FINAL_VALIDATION,
        )

        self.assertIs(
            self.assert_success_after_stability(validation_task),
            AssessmentResult.COMPLETE,
        )
        self.assertIs(self.controller.snapshot.state, ControllerState.COMPLETE)
        self.assertEqual(len(self.published), 3)
        self.assertEqual(self.published, self.monitored)

    def test_ongoing_breaks_success_streak(self) -> None:
        task = self.confirm()
        self.assertIs(
            self.controller.record_assessment(
                assessment(task, observed_at=100.0, status="SUCCESS")
            ),
            AssessmentResult.ACCEPTED,
        )
        self.clock.advance(1.0)
        self.assertIs(
            self.controller.record_assessment(
                assessment(
                    task,
                    observed_at=101.0,
                    status="ONGOING",
                    criterion_state="NOT_MET",
                )
            ),
            AssessmentResult.ACCEPTED,
        )
        self.clock.advance(2.0)
        self.assertIs(
            self.controller.record_assessment(
                assessment(task, observed_at=103.0, status="SUCCESS")
            ),
            AssessmentResult.ACCEPTED,
        )
        self.assertEqual(self.controller.snapshot.consecutive_successes, 1)

    def test_failure_requires_repeated_fresh_assessments(self) -> None:
        task = self.confirm()
        first = assessment(task, observed_at=100.0, status="FAIL")
        self.assertIs(
            self.controller.record_assessment(first),
            AssessmentResult.ACCEPTED,
        )
        self.assertIs(self.controller.snapshot.state, ControllerState.EXECUTING)

        self.clock.advance(1.0)
        second = assessment(task, observed_at=101.0, status="FAIL")
        self.assertIs(
            self.controller.record_assessment(second),
            AssessmentResult.FAILED,
        )
        self.assertIs(self.controller.snapshot.state, ControllerState.FAILED)

    def test_different_failure_evidence_restarts_confirmation(self) -> None:
        task = self.confirm()
        first = assessment(task, observed_at=100.0, status="FAIL")
        self.assertIs(
            self.controller.record_assessment(first),
            AssessmentResult.ACCEPTED,
        )
        self.clock.advance(1.0)
        different = MonitorAssessment.from_model_output(
            {
                "task_status": "FAIL",
                "criteria": [
                    {"id": criterion.criterion_id, "state": "UNKNOWN"}
                    for criterion in task.expected_observation
                ],
                "failure": {
                    "kind": "KNOWN",
                    "description": "The book is visibly damaged.",
                },
                "observation": "The book cover is visibly torn.",
            },
            plan_id=task.plan_id,
            revision=task.revision,
            step_id=task.step_id,
            observed_at=101.0,
        )

        self.assertIs(
            self.controller.record_assessment(different),
            AssessmentResult.ACCEPTED,
        )
        self.assertIs(self.controller.snapshot.state, ControllerState.EXECUTING)
        self.assertEqual(self.controller.snapshot.consecutive_failures, 1)

    def test_missing_criterion_is_invalid_and_cannot_advance(self) -> None:
        task = self.confirm()
        incomplete = MonitorAssessment.from_model_output(
            {
                "task_status": "SUCCESS",
                "criteria": [
                    {
                        "id": task.expected_observation[0].criterion_id,
                        "state": "MET",
                    }
                ],
                "failure": None,
                "observation": "Only one expected condition was assessed.",
            },
            plan_id=task.plan_id,
            revision=task.revision,
            step_id=task.step_id,
            observed_at=self.clock.value,
        )

        self.assertIs(
            self.controller.record_assessment(incomplete),
            AssessmentResult.IGNORED_INVALID,
        )
        self.assertEqual(self.controller.snapshot.consecutive_successes, 0)

    def test_no_progress_timeout_needs_attention_and_can_resume(self) -> None:
        task = self.confirm()
        self.clock.advance(9.9)
        self.assertFalse(self.controller.check_timeout())
        self.clock.advance(0.1)

        self.assertTrue(self.controller.check_timeout())
        snapshot = self.controller.snapshot
        self.assertIs(snapshot.state, ControllerState.NEEDS_ATTENTION)
        self.assertIsNot(snapshot.state, ControllerState.FAILED)

        resumed = self.controller.resume_after_attention()
        self.assertEqual(resumed.step_id, task.step_id)
        self.assertGreater(resumed.published_at, task.published_at)
        self.assertIs(self.controller.snapshot.state, ControllerState.EXECUTING)
        self.assertEqual(self.published[-1], resumed)
        self.assertEqual(self.monitored[-1], resumed)

    def test_nonready_plans_never_publish(self) -> None:
        already = assign_plan_ids(
            PlannerPlan(
                status="ALREADY_SATISFIED",
                final_expected_observation=("The table is already tidy.",),
            ),
            plan_id="already",
        )
        blocked = assign_plan_ids(
            PlannerPlan(
                status="BLOCKED",
                blocked_reason="The target object is not visible.",
            ),
            plan_id="blocked",
        )

        self.assertIs(
            self.controller.stage_plan(already).state,
            ControllerState.ALREADY_SATISFIED,
        )
        self.assertIs(
            self.controller.stage_plan(blocked).state,
            ControllerState.BLOCKED,
        )
        self.assertEqual(self.published, [])

    def test_already_satisfied_still_needs_confirmed_final_validation(self) -> None:
        already = assign_plan_ids(
            PlannerPlan(
                status="ALREADY_SATISFIED",
                final_expected_observation=("The table is already tidy.",),
            ),
            plan_id="already",
        )
        self.controller.stage_plan(already)

        task = self.controller.confirm_and_publish(already, confirmed=True)

        self.assertIs(task.phase, TaskPhase.FINAL_VALIDATION)
        self.assertEqual(
            task.instruction,
            "Hold the camera steady on the completed result.",
        )
        self.assertIs(
            self.assert_success_after_stability(task),
            AssessmentResult.COMPLETE,
        )
        self.assertIs(self.controller.snapshot.state, ControllerState.COMPLETE)


if __name__ == "__main__":
    unittest.main()
