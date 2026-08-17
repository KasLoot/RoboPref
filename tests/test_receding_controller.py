from __future__ import annotations

import unittest
import threading

from prefmem.contracts import (
    GoalContract,
    MonitorAssessment,
    PlannerDecision,
)
from prefmem.controller import (
    AttentionKind,
    InvalidTransitionError,
    RecedingControllerState,
    RecedingHorizonController,
    RecedingResult,
    TimeoutPolicy,
)


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> float:
        self.value += seconds
        return self.value


def goal() -> GoalContract:
    return GoalContract(
        goal_id="tower",
        revision=1,
        goal="Build a stable tower using both blocks.",
        final_expected_observation=(
            "The red block rests on the blue block in one upright stable tower.",
        ),
        constraints=("Keep both blocks on the table.",),
        nominal_tasks=(
            "Place the blue block flat.",
            "Place the red block on the blue block.",
        ),
    )


def act(instruction: str = "Place the blue block flat in the marked area."):
    return PlannerDecision.from_dict(
        {
            "decision": "ACT",
            "candidate_tasks": [
                {
                    "instruction": instruction,
                    "expected_observation": [
                        "The blue block is flat and stable in the marked area."
                    ],
                    "known_failure_conditions": [
                        "The blue block has fallen off the table."
                    ],
                },
                {
                    "instruction": "Place the red block on the blue block.",
                    "expected_observation": [
                        "The red block rests on the blue block.",
                        "The two-block tower remains upright and stable.",
                    ],
                    "known_failure_conditions": [],
                },
            ],
            "reason": "A flat base is needed first.",
            "blocked_reason": None,
            "user_question": None,
        }
    )


def assessment(
    task,
    *,
    when: float,
    frame: int,
    status: str,
    state: str | None = None,
):
    failure = None
    criterion_state = "MET" if state is None else state
    if status == "FAIL":
        criterion_state = "NOT_MET"
        failure = {
            "kind": "UNEXPECTED",
            "description": "The block has fallen onto its side.",
        }
    return MonitorAssessment.from_model_output(
        {
            "task_status": status,
            "criteria": [
                {"id": criterion.criterion_id, "state": criterion_state}
                for criterion in task.expected_observation
            ],
            "failure": failure,
            "observation": (
                "The block is temporarily occluded."
                if criterion_state == "UNKNOWN"
                else (
                    "The block is visibly stable."
                    if criterion_state == "MET"
                    else "The block is visibly lying on its side."
                )
            ),
        },
        plan_id=task.plan_id,
        revision=task.revision,
        step_id=task.step_id,
        publication_id=task.publication_id,
        observed_at=when,
        frame_sequence=frame,
    )


class RecedingHorizonControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.controller = RecedingHorizonController(
            success_confirmations=2,
            success_stability_seconds=2.0,
            failure_confirmations=2,
            ongoing_timeout_seconds=10.0,
            clock=self.clock,
        )
        self.goal = goal()

    def start_task(self):
        self.controller.stage_goal(self.goal)
        transition = self.controller.confirm_goal(
            self.goal,
            confirmed=True,
            frame_sequence=10,
        )
        self.assertIs(transition.result, RecedingResult.PLANNING_STARTED)
        transition = self.controller.apply_planner_decision(
            transition.planner_request,
            act(),
        )
        self.assertIs(transition.result, RecedingResult.TASK_PUBLISHED)
        return transition.task_to_publish

    def test_confirmation_starts_fresh_planning_and_publishes_only_first(self) -> None:
        task = self.start_task()

        self.assertEqual(task.instruction, "Place the blue block flat in the marked area.")
        self.assertEqual(task.cycle_id, 1)
        self.assertEqual(task.frame_sequence, 10)
        self.assertIs(self.controller.snapshot.state, RecedingControllerState.EXECUTING)

    def test_executor_heartbeat_refreshes_timeout_without_advancing_task(self) -> None:
        task = self.start_task()
        self.clock.advance(9.0)
        self.assertTrue(
            self.controller.record_execution_progress(task.publication_id)
        )
        self.clock.advance(9.0)
        self.assertIsNone(self.controller.check_timeout())
        self.assertIs(self.controller.snapshot.current_task, task)
        self.assertFalse(self.controller.record_execution_progress("stale-publication"))

    def test_stable_success_appends_history_and_requests_new_cycle(self) -> None:
        task = self.start_task()
        self.clock.advance(0.1)
        first = self.controller.record_assessment(
            assessment(task, when=self.clock.value, frame=11, status="SUCCESS")
        )
        self.assertIs(first.result, RecedingResult.ACCEPTED)
        self.clock.advance(2.0)
        terminal = self.controller.record_assessment(
            assessment(task, when=self.clock.value, frame=12, status="SUCCESS")
        )

        self.assertIs(terminal.result, RecedingResult.REPLAN_REQUESTED)
        self.assertEqual(terminal.planner_request.trigger.value, "TASK_SUCCESS")
        self.assertEqual(terminal.planner_request.cycle_id, 2)
        self.assertEqual(len(terminal.planner_request.execution_history), 1)
        self.assertEqual(
            terminal.planner_request.execution_history[0].outcome.value,
            "SUCCESS",
        )

    def test_all_met_ongoing_reports_request_the_next_cycle(self) -> None:
        task = self.start_task()
        self.clock.advance(0.1)
        first = self.controller.record_assessment(
            assessment(task, when=self.clock.value, frame=11, status="ONGOING")
        )
        self.assertIs(first.result, RecedingResult.ACCEPTED)
        self.assertEqual(first.snapshot.consecutive_successes, 1)

        self.clock.advance(2.0)
        terminal = self.controller.record_assessment(
            assessment(task, when=self.clock.value, frame=12, status="ONGOING")
        )

        self.assertIs(terminal.result, RecedingResult.REPLAN_REQUESTED)
        self.assertEqual(terminal.planner_request.trigger.value, "TASK_SUCCESS")

    def test_unknown_view_preserves_but_does_not_increment_success(self) -> None:
        task = self.start_task()
        self.clock.advance(0.1)
        self.controller.record_assessment(
            assessment(task, when=self.clock.value, frame=11, status="SUCCESS")
        )

        self.clock.advance(1.0)
        inconclusive = self.controller.record_assessment(
            assessment(
                task,
                when=self.clock.value,
                frame=12,
                status="ONGOING",
                state="UNKNOWN",
            )
        )
        self.assertIs(inconclusive.result, RecedingResult.ACCEPTED)
        self.assertEqual(inconclusive.snapshot.consecutive_successes, 1)

        self.clock.advance(1.0)
        terminal = self.controller.record_assessment(
            assessment(task, when=self.clock.value, frame=13, status="SUCCESS")
        )
        self.assertIs(terminal.result, RecedingResult.REPLAN_REQUESTED)

    def test_visible_not_met_resets_pending_success(self) -> None:
        task = self.start_task()
        self.clock.advance(0.1)
        self.controller.record_assessment(
            assessment(task, when=self.clock.value, frame=11, status="SUCCESS")
        )

        self.clock.advance(1.0)
        contradicted = self.controller.record_assessment(
            assessment(
                task,
                when=self.clock.value,
                frame=12,
                status="ONGOING",
                state="NOT_MET",
            )
        )
        self.assertEqual(contradicted.snapshot.consecutive_successes, 0)

        self.clock.advance(1.0)
        fresh_candidate = self.controller.record_assessment(
            assessment(task, when=self.clock.value, frame=13, status="SUCCESS")
        )
        self.assertIs(fresh_candidate.result, RecedingResult.ACCEPTED)
        self.assertEqual(fresh_candidate.snapshot.consecutive_successes, 1)

    def test_stable_task_failure_replans_instead_of_ending_goal(self) -> None:
        task = self.start_task()
        self.clock.advance(0.1)
        self.controller.record_assessment(
            assessment(task, when=self.clock.value, frame=11, status="FAIL")
        )
        self.clock.advance(1.0)
        terminal = self.controller.record_assessment(
            assessment(task, when=self.clock.value, frame=12, status="FAIL")
        )

        self.assertIs(terminal.result, RecedingResult.REPLAN_REQUESTED)
        self.assertEqual(terminal.planner_request.trigger.value, "TASK_FAIL")
        self.assertEqual(
            terminal.planner_request.execution_history[-1].failure_reason,
            "The block has fallen onto its side.",
        )

    def test_final_validation_is_monitor_owned(self) -> None:
        self.controller.stage_goal(self.goal)
        planning = self.controller.confirm_goal(self.goal, confirmed=True)
        transition = self.controller.apply_planner_decision(
            planning.planner_request,
            PlannerDecision.from_dict(
                {
                    "decision": "REQUEST_FINAL_VALIDATION",
                    "candidate_tasks": [],
                    "reason": "The goal appears satisfied.",
                    "blocked_reason": None,
                    "user_question": None,
                }
            ),
        )
        task = transition.task_to_publish
        self.assertIs(
            self.controller.snapshot.state,
            RecedingControllerState.FINAL_VALIDATION,
        )
        self.clock.advance(0.1)
        self.controller.record_assessment(
            assessment(task, when=self.clock.value, frame=1, status="SUCCESS")
        )
        self.clock.advance(2.0)
        completed = self.controller.record_assessment(
            assessment(task, when=self.clock.value, frame=2, status="SUCCESS")
        )

        self.assertIs(completed.result, RecedingResult.COMPLETE)
        self.assertIs(self.controller.snapshot.state, RecedingControllerState.COMPLETE)

    def test_attention_gate_can_resume_with_a_new_publication(self) -> None:
        self.controller = RecedingHorizonController(
            ongoing_timeout_seconds=10.0,
            timeout_policy=TimeoutPolicy.ATTENTION_GATE,
            clock=self.clock,
        )
        old_task = self.start_task()
        self.clock.advance(10.0)
        timeout = self.controller.check_timeout()
        self.assertIs(timeout.result, RecedingResult.NEEDS_ATTENTION)
        new_task = self.controller.resume_after_attention(
            frame_sequence=20
        ).task_to_publish
        self.assertNotEqual(old_task.publication_id, new_task.publication_id)
        self.clock.advance(0.1)
        stale = self.controller.record_assessment(
            assessment(old_task, when=self.clock.value, frame=21, status="SUCCESS")
        )
        self.assertIs(stale.result, RecedingResult.IGNORED_STALE)

    def test_step_timeout_records_interrupted_and_requests_fresh_cycle_once(self) -> None:
        task = self.start_task()
        self.clock.advance(0.1)
        accepted = self.controller.record_assessment(
            assessment(
                task,
                when=self.clock.value,
                frame=11,
                status="ONGOING",
                state="UNKNOWN",
            )
        )
        self.assertIs(accepted.result, RecedingResult.ACCEPTED)
        self.clock.advance(9.9)

        timeout = self.controller.check_timeout(
            publication_id=task.publication_id
        )

        self.assertIs(timeout.result, RecedingResult.REPLAN_REQUESTED)
        self.assertIs(timeout.task_to_retire, task)
        self.assertEqual(timeout.planner_request.trigger.value, "MONITOR_TIMEOUT")
        self.assertEqual(timeout.planner_request.cycle_id, 2)
        self.assertEqual(len(timeout.snapshot.execution_history), 1)
        record = timeout.snapshot.execution_history[0]
        self.assertEqual(record.outcome.value, "INTERRUPTED")
        self.assertEqual(record.observation, "The block is temporarily occluded.")
        self.assertIsNone(record.failure_reason)
        self.assertEqual(record.evidence_frame_sequence, 11)
        self.assertEqual(
            record.termination_context.to_dict(),
            {
                "kind": "MONITOR_TIMEOUT",
                "timeout_seconds": 10.0,
                "elapsed_no_progress_seconds": 10.0,
                "last_progress_at": 100.0,
                "last_frame_sequence": 11,
                "last_assessment_disposition": "ACCEPTED",
                "criteria": [
                    {
                        "id": task.expected_observation[0].criterion_id,
                        "state": "UNKNOWN",
                    }
                ],
                "prior_publication_id": task.publication_id,
            },
        )
        self.assertIsNone(
            self.controller.check_timeout(publication_id=task.publication_id)
        )
        self.assertEqual(len(self.controller.snapshot.execution_history), 1)

    def test_timeout_without_assessment_uses_truthful_observation(self) -> None:
        task = self.start_task()
        self.clock.advance(10.0)

        timeout = self.controller.check_timeout()

        record = timeout.snapshot.execution_history[-1]
        self.assertEqual(record.observation, "No conclusive post-state was observed.")
        self.assertEqual(record.termination_context.criteria, ())
        self.assertIsNone(
            record.termination_context.last_assessment_disposition
        )
        self.assertIs(timeout.task_to_retire, task)

    def test_concurrent_timeout_polls_are_idempotent(self) -> None:
        task = self.start_task()
        self.clock.advance(10.0)
        barrier = threading.Barrier(3)
        results = []

        def poll() -> None:
            barrier.wait()
            results.append(
                self.controller.check_timeout(
                    now=self.clock.value,
                    publication_id=task.publication_id,
                )
            )

        threads = [threading.Thread(target=poll) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2.0)

        self.assertEqual(
            sum(item is not None for item in results),
            1,
        )
        self.assertEqual(len(self.controller.snapshot.execution_history), 1)
        self.assertEqual(
            self.controller.snapshot.pending_request.trigger.value,
            "MONITOR_TIMEOUT",
        )

    def test_late_old_publication_assessments_are_stale_after_timeout(self) -> None:
        old_task = self.start_task()
        self.clock.advance(10.0)
        timeout = self.controller.check_timeout()

        for status in ("SUCCESS", "FAIL"):
            with self.subTest(status=status):
                stale = self.controller.record_assessment(
                    assessment(
                        old_task,
                        when=self.clock.advance(0.1),
                        frame=20 if status == "SUCCESS" else 21,
                        status=status,
                    )
                )
                self.assertIs(stale.result, RecedingResult.IGNORED_STALE)

        replacement = self.controller.apply_planner_decision(
            timeout.planner_request,
            act("Place the red block beside the blue block."),
        ).task_to_publish
        stale = self.controller.record_assessment(
            assessment(
                old_task,
                when=self.clock.advance(0.1),
                frame=22,
                status="SUCCESS",
            )
        )
        self.assertIs(stale.result, RecedingResult.IGNORED_STALE)
        self.assertEqual(
            self.controller.snapshot.current_task.publication_id,
            replacement.publication_id,
        )

    def test_final_validation_timeout_remains_attention_without_history(self) -> None:
        self.controller.stage_goal(self.goal)
        planning = self.controller.confirm_goal(self.goal, confirmed=True)
        transition = self.controller.apply_planner_decision(
            planning.planner_request,
            PlannerDecision.from_dict(
                {
                    "decision": "REQUEST_FINAL_VALIDATION",
                    "candidate_tasks": [],
                    "reason": "The goal appears satisfied.",
                    "blocked_reason": None,
                    "user_question": None,
                }
            ),
        )
        task = transition.task_to_publish
        self.clock.advance(10.0)

        timeout = self.controller.check_timeout()

        self.assertIs(timeout.result, RecedingResult.NEEDS_ATTENTION)
        self.assertIs(timeout.task_to_retire, task)
        self.assertIs(timeout.snapshot.attention_kind, AttentionKind.NO_PROGRESS)
        self.assertEqual(timeout.snapshot.execution_history, ())
        self.assertIsNone(timeout.planner_request)

    def test_consecutive_timeout_guard_stops_replan_loop(self) -> None:
        task = self.start_task()
        for expected_count in (1, 2):
            self.clock.advance(10.0)
            timeout = self.controller.check_timeout()
            self.assertIs(timeout.result, RecedingResult.REPLAN_REQUESTED)
            self.assertEqual(
                timeout.snapshot.consecutive_timeout_replans,
                expected_count,
            )
            task = self.controller.apply_planner_decision(
                timeout.planner_request,
                act(),
            ).task_to_publish

        self.clock.advance(10.0)
        guarded = self.controller.check_timeout()

        self.assertIs(guarded.result, RecedingResult.NEEDS_ATTENTION)
        self.assertIs(guarded.snapshot.attention_kind, AttentionKind.LOOP_GUARD)
        self.assertIn("Consecutive", guarded.snapshot.attention_reason)
        self.assertIn(task.instruction, guarded.snapshot.attention_reason)
        self.assertEqual(len(guarded.snapshot.execution_history), 2)

    def test_stable_success_resets_consecutive_timeout_guard(self) -> None:
        self.controller = RecedingHorizonController(
            success_confirmations=2,
            success_stability_seconds=0.0,
            ongoing_timeout_seconds=10.0,
            max_consecutive_timeout_replans=1,
            max_timeout_replans_per_instruction=3,
            clock=self.clock,
        )
        self.start_task()
        self.clock.advance(10.0)
        timeout = self.controller.check_timeout()
        successful_task = self.controller.apply_planner_decision(
            timeout.planner_request,
            act("Place the red block upright."),
        ).task_to_publish
        for frame in (21, 22):
            success = self.controller.record_assessment(
                assessment(
                    successful_task,
                    when=self.clock.advance(0.1),
                    frame=frame,
                    status="SUCCESS",
                )
            )
        self.assertIs(success.result, RecedingResult.REPLAN_REQUESTED)
        self.assertEqual(success.snapshot.consecutive_timeout_replans, 0)
        next_task = self.controller.apply_planner_decision(
            success.planner_request,
            act("Place the green block upright."),
        ).task_to_publish
        self.clock.advance(10.0)

        next_timeout = self.controller.check_timeout()

        self.assertIs(next_timeout.result, RecedingResult.REPLAN_REQUESTED)
        self.assertIs(next_timeout.task_to_retire, next_task)

    def test_per_instruction_timeout_guard_survives_success_reset(self) -> None:
        repeated = "Place the blue block flat in the marked area."
        self.controller = RecedingHorizonController(
            success_confirmations=2,
            success_stability_seconds=0.0,
            ongoing_timeout_seconds=10.0,
            max_consecutive_timeout_replans=3,
            max_timeout_replans_per_instruction=1,
            clock=self.clock,
        )
        self.start_task()
        self.clock.advance(10.0)
        timeout = self.controller.check_timeout()
        successful_task = self.controller.apply_planner_decision(
            timeout.planner_request,
            act("Place the red block upright."),
        ).task_to_publish
        for frame in (31, 32):
            success = self.controller.record_assessment(
                assessment(
                    successful_task,
                    when=self.clock.advance(0.1),
                    frame=frame,
                    status="SUCCESS",
                )
            )
        repeated_task = self.controller.apply_planner_decision(
            success.planner_request,
            act(repeated),
        ).task_to_publish
        self.clock.advance(10.0)

        guarded = self.controller.check_timeout()

        self.assertIs(guarded.result, RecedingResult.NEEDS_ATTENTION)
        self.assertIs(guarded.snapshot.attention_kind, AttentionKind.LOOP_GUARD)
        self.assertIn("Per-instruction", guarded.snapshot.attention_reason)
        self.assertIs(guarded.task_to_retire, repeated_task)

    def test_emergency_is_terminal_and_cannot_be_reset(self) -> None:
        self.start_task()
        stopped = self.controller.emergency_stop("A person entered the workspace.")
        self.assertIs(stopped.result, RecedingResult.EMERGENCY_STOPPED)
        self.assertIs(
            self.controller.snapshot.state,
            RecedingControllerState.EMERGENCY_STOPPED,
        )
        with self.assertRaisesRegex(InvalidTransitionError, "latched"):
            self.controller.stage_goal(self.goal)


if __name__ == "__main__":
    unittest.main()
