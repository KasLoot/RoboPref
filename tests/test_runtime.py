from __future__ import annotations

import unittest
from unittest.mock import patch

from prefmem.agents.monitor import CapturedFrame, MonitorErrorEvent, MonitorErrorKind
from prefmem.contracts import (
    GoalProposal,
    MonitorAssessment,
    PlannerDecision,
)
from prefmem.controller import RecedingControllerState, RecedingHorizonController
from prefmem.runtime import PrefMemRuntime
from prefmem.task_publisher import DisplayState, TaskPublisherResponseError


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float = 1.0) -> float:
        self.value += seconds
        return self.value


class SequenceFrames:
    def __init__(self, *sequences: int) -> None:
        self._sequences = list(sequences)
        self.calls = 0

    def __call__(self) -> CapturedFrame:
        if self.calls >= len(self._sequences):
            raise AssertionError("runtime requested an unexpected camera frame")
        sequence = self._sequences[self.calls]
        self.calls += 1
        return CapturedFrame(
            image_block={
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/"},
            },
            observed_at=float(sequence),
            sequence=sequence,
        )


class RecordingPublisher:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.reset_sessions = []
        self.displays = []

    def reset(self, *, session_id=None) -> None:
        self.reset_calls += 1
        self.reset_sessions.append(session_id)

    def publish(self, display) -> None:
        self.displays.append(display)


class RecordingMonitor:
    def __init__(self) -> None:
        self.published = []
        self.retired = []
        self.stop_calls = 0
        self.events = []

    def publish(self, task) -> None:
        self.published.append(task)
        self.events.append(("publish", task.publication_id))

    def retire(self, publication_id=None) -> bool:
        self.retired.append(publication_id)
        self.events.append(("retire", publication_id))
        return True

    def stop(self) -> None:
        self.stop_calls += 1
        self.events.append(("stop", None))


class ScriptedPlanner:
    def __init__(self, *decisions: PlannerDecision) -> None:
        self._decisions = list(decisions)
        self.preview_calls = []
        self.plan_calls = []

    def preview(
        self,
        clarified_goal,
        current_frame,
        *,
        constraints=(),
        operator_guidance=None,
    ) -> GoalProposal:
        self.preview_calls.append(
            (clarified_goal, current_frame, tuple(constraints), operator_guidance)
        )
        return GoalProposal.from_dict(
            {
                "status": "READY",
                "goal": clarified_goal,
                "final_expected_observation": [
                    "The red block rests on the blue block in one stable tower."
                ],
                "constraints": list(constraints),
                "nominal_tasks": [
                    "Place the blue block flat.",
                    "Place the red block on the blue block.",
                ],
                "reason": "The blocks can be stacked safely.",
            }
        )

    def plan_cycle(self, request, current_frame) -> PlannerDecision:
        if not self._decisions:
            raise AssertionError("runtime requested an unexpected Planner cycle")
        self.plan_calls.append((request, current_frame))
        return self._decisions.pop(0)


def act(instruction: str, future_instruction: str) -> PlannerDecision:
    return PlannerDecision.from_dict(
        {
            "decision": "ACT",
            "candidate_tasks": [
                {
                    "instruction": instruction,
                    "expected_observation": [
                        "The manipulated block is visibly stable in the requested pose."
                    ],
                    "known_failure_conditions": [
                        "The manipulated block falls off the table."
                    ],
                },
                {
                    "instruction": future_instruction,
                    "expected_observation": [
                        "The future placement is visibly complete."
                    ],
                    "known_failure_conditions": [],
                },
            ],
            "reason": "This is the best next state change.",
            "blocked_reason": None,
            "user_question": None,
        }
    )


def final_validation() -> PlannerDecision:
    return PlannerDecision.from_dict(
        {
            "decision": "REQUEST_FINAL_VALIDATION",
            "candidate_tasks": [],
            "reason": "The high-level goal appears complete.",
            "blocked_reason": None,
            "user_question": None,
        }
    )


def assessment(task, *, status: str, observed_at: float, frame: int):
    failed = status == "FAIL"
    return MonitorAssessment.from_model_output(
        {
            "task_status": status,
            "criteria": [
                {
                    "id": criterion.criterion_id,
                    "state": "NOT_MET" if failed else "MET",
                }
                for criterion in task.expected_observation
            ],
            "failure": (
                {
                    "kind": "UNEXPECTED",
                    "description": "The block visibly fell onto its side.",
                }
                if failed
                else None
            ),
            "observation": (
                "The block is visibly lying on its side."
                if failed
                else "The requested visible state is stable."
            ),
        },
        plan_id=task.plan_id,
        revision=task.revision,
        step_id=task.step_id,
        publication_id=task.publication_id,
        observed_at=observed_at,
        frame_sequence=frame,
    )


class RuntimeOrchestrationTests(unittest.TestCase):
    def build_runtime(self, planner, *, frame_sequences=(10, 20, 30)):
        clock = FakeClock()
        publisher = RecordingPublisher()
        monitor = RecordingMonitor()
        controller = RecedingHorizonController(
            success_confirmations=2,
            success_stability_seconds=0,
            failure_confirmations=2,
            ongoing_timeout_seconds=30,
            clock=clock,
        )
        runtime = PrefMemRuntime(
            planner,
            task_publisher=publisher,
            frame_source=SequenceFrames(*frame_sequences),
            controller=controller,
            monitor=monitor,
            session_id="test-session",
        )
        self.addCleanup(runtime.close)
        return runtime, clock, publisher, monitor

    def start_goal(self, runtime: PrefMemRuntime):
        preview = runtime.request_goal_preview(
            "Build a stable two-block tower.",
            constraints=("Keep both blocks on the table.",),
        )
        contract = preview["goal_contract"]
        context = runtime.confirm_goal(
            goal_id=contract["goal_id"],
            revision=contract["revision"],
            confirmed=True,
        )
        return context

    def emit_twice(self, runtime, clock, task, *, status, frames):
        for frame in frames:
            runtime._on_monitor_assessment(
                assessment(
                    task,
                    status=status,
                    observed_at=clock.advance(),
                    frame=frame,
                )
            )

    def test_success_replans_then_final_validation_completes(self) -> None:
        planner = ScriptedPlanner(
            act(
                "Place the blue block flat in the marked area.",
                "Place the red block on the blue block.",
            ),
            final_validation(),
        )
        runtime, clock, publisher, monitor = self.build_runtime(planner)

        context = self.start_goal(runtime)
        first_task = monitor.published[0]

        self.assertEqual(context["state"], "EXECUTING")
        self.assertEqual(first_task.instruction, "Place the blue block flat in the marked area.")
        self.assertNotIn(
            "Place the red block on the blue block.",
            [task.instruction for task in monitor.published],
        )

        self.emit_twice(runtime, clock, first_task, status="SUCCESS", frames=(21, 22))

        self.assertEqual(len(planner.plan_calls), 2)
        second_request = planner.plan_calls[1][0]
        self.assertEqual(second_request.cycle_id, 2)
        self.assertEqual(second_request.trigger.value, "TASK_SUCCESS")
        self.assertEqual(second_request.frame_sequence, 30)
        self.assertEqual(second_request.execution_history[0].outcome.value, "SUCCESS")
        self.assertEqual(monitor.retired, [first_task.publication_id])

        validation_task = monitor.published[1]
        self.assertEqual(validation_task.phase.value, "FINAL_VALIDATION")
        self.assertIs(publisher.displays[-1].state, DisplayState.FINAL_VALIDATION)

        self.emit_twice(
            runtime,
            clock,
            validation_task,
            status="SUCCESS",
            frames=(31, 32),
        )

        self.assertEqual(runtime.context_dict()["state"], "COMPLETE")
        self.assertIs(publisher.displays[-1].state, DisplayState.COMPLETE)
        self.assertEqual(
            monitor.retired,
            [first_task.publication_id, validation_task.publication_id],
        )

    def test_declined_confirmation_does_not_capture_or_plan(self) -> None:
        planner = ScriptedPlanner(
            act("Place the blue block flat.", "Place the red block next.")
        )
        runtime, _clock, publisher, monitor = self.build_runtime(planner)
        preview = runtime.request_goal_preview("Build a stable two-block tower.")
        contract = preview["goal_contract"]

        context = runtime.confirm_goal(
            goal_id=contract["goal_id"],
            revision=contract["revision"],
            confirmed=False,
        )

        self.assertEqual(context["confirmation_status"], "DECLINED")
        self.assertEqual(context["state"], "AWAITING_CONFIRMATION")
        self.assertEqual(runtime.frame_source.calls, 1)
        self.assertEqual(planner.plan_calls, [])
        self.assertEqual(monitor.published, [])
        self.assertEqual(publisher.displays, [])

    def test_blocked_replacement_cannot_leave_old_goal_authoritative(self) -> None:
        class BlockingReplacementPlanner(ScriptedPlanner):
            def preview(
                self,
                clarified_goal,
                current_frame,
                *,
                constraints=(),
                operator_guidance=None,
            ):
                if not self.preview_calls:
                    return super().preview(
                        clarified_goal,
                        current_frame,
                        constraints=constraints,
                        operator_guidance=operator_guidance,
                    )
                self.preview_calls.append(
                    (
                        clarified_goal,
                        current_frame,
                        tuple(constraints),
                        operator_guidance,
                    )
                )
                return GoalProposal.from_dict(
                    {
                        "status": "BLOCKED",
                        "goal": clarified_goal,
                        "final_expected_observation": [],
                        "constraints": list(constraints),
                        "nominal_tasks": [],
                        "reason": "The requested object is not visible.",
                    }
                )

        planner = BlockingReplacementPlanner()
        runtime, _clock, _publisher, _monitor = self.build_runtime(
            planner,
            frame_sequences=(10, 20),
        )
        first = runtime.request_goal_preview("Build a stable two-block tower.")
        self.assertTrue(first["confirmation_required"])

        blocked = runtime.request_goal_preview("Put the green block in the tray.")

        self.assertFalse(blocked["confirmation_required"])
        self.assertIsNone(runtime.pending_goal)
        self.assertEqual(runtime.context_dict()["state"], "IDLE")
        self.assertIsNone(runtime.context_dict()["goal"])

    def test_second_goal_rotates_display_session_after_complete(self) -> None:
        planner = ScriptedPlanner(
            act("Place the blue block flat.", "Place the red block next."),
            final_validation(),
            act("Move the red block into the tray.", "Validate the tray."),
        )
        runtime, clock, publisher, monitor = self.build_runtime(
            planner,
            frame_sequences=(10, 20, 30, 40, 50),
        )
        self.start_goal(runtime)
        first_session = runtime.session_id
        first_task = monitor.published[0]
        self.emit_twice(runtime, clock, first_task, status="SUCCESS", frames=(21, 22))
        validation = monitor.published[1]
        self.emit_twice(
            runtime,
            clock,
            validation,
            status="SUCCESS",
            frames=(31, 32),
        )
        self.assertIs(publisher.displays[-1].state, DisplayState.COMPLETE)
        self.assertEqual(publisher.displays[-1].session_id, first_session)

        preview = runtime.request_goal_preview("Move the red block into the tray.")
        second_session = runtime.session_id

        self.assertNotEqual(second_session, first_session)
        contract = preview["goal_contract"]
        runtime.confirm_goal(
            goal_id=contract["goal_id"],
            revision=contract["revision"],
            confirmed=True,
        )
        self.assertIs(publisher.displays[-1].state, DisplayState.ACTIVE)
        self.assertEqual(publisher.displays[-1].session_id, second_session)
        self.assertEqual(publisher.displays[-1].sequence, 2)

    def test_terminal_task_failure_is_history_for_the_replan(self) -> None:
        planner = ScriptedPlanner(
            act(
                "Place the blue block flat in the marked area.",
                "Place the red block on the blue block.",
            ),
            act(
                "Return the blue block upright inside the marked area.",
                "Place the red block after the base is stable.",
            ),
        )
        runtime, clock, publisher, monitor = self.build_runtime(planner)
        self.start_goal(runtime)
        failed_task = monitor.published[0]

        self.emit_twice(runtime, clock, failed_task, status="FAIL", frames=(21, 22))

        replan_request = planner.plan_calls[1][0]
        record = replan_request.execution_history[-1]
        self.assertEqual(replan_request.trigger.value, "TASK_FAIL")
        self.assertEqual(record.outcome.value, "FAIL")
        self.assertEqual(record.failure_reason, "The block visibly fell onto its side.")
        self.assertEqual(runtime.context_dict()["state"], "EXECUTING")
        self.assertEqual(len(monitor.published), 2)
        self.assertNotEqual(
            monitor.published[0].publication_id,
            monitor.published[1].publication_id,
        )
        self.assertEqual(
            monitor.published[1].instruction,
            "Return the blue block upright inside the marked area.",
        )
        self.assertIs(publisher.displays[-1].state, DisplayState.ACTIVE)

    def test_user_replan_retires_old_monitor_task_before_replacement(self) -> None:
        planner = ScriptedPlanner(
            act("Place the blue block flat.", "Place the red block next."),
            act("Move the blue block to the left.", "Place the red block later."),
        )
        runtime, _clock, _publisher, monitor = self.build_runtime(planner)
        self.start_goal(runtime)
        old_task = monitor.published[0]
        runtime.controller.record_system_error("Operator requested a correction")

        runtime.request_replan("Use the clear area on the left.")

        self.assertEqual(
            monitor.events[:3],
            [
                ("publish", old_task.publication_id),
                ("retire", old_task.publication_id),
                ("publish", monitor.published[1].publication_id),
            ],
        )
        self.assertEqual(planner.plan_calls[1][0].trigger.value, "USER_REPLAN")

    def test_close_clears_an_active_instruction_for_its_session(self) -> None:
        planner = ScriptedPlanner(
            act("Place the blue block flat.", "Place the red block next.")
        )
        runtime, _clock, publisher, monitor = self.build_runtime(
            planner,
            frame_sequences=(10, 20),
        )
        self.start_goal(runtime)

        runtime.close()

        self.assertEqual(monitor.stop_calls, 1)
        self.assertEqual(publisher.reset_calls, 2)
        self.assertEqual(publisher.reset_sessions, [None, "test-session"])

    def test_camera_reset_failure_tells_operator_to_restart_streamer(self) -> None:
        class FailingPublisher(RecordingPublisher):
            def reset(self, *, session_id=None) -> None:
                raise TaskPublisherResponseError(501, "unsupported method")

        planner = ScriptedPlanner()
        with self.assertRaisesRegex(RuntimeError, "Restart `stream_camera`"):
            PrefMemRuntime(
                planner,
                task_publisher=FailingPublisher(),
                frame_source=SequenceFrames(10),
                monitor=RecordingMonitor(),
            )

    def test_emergency_stop_latches_stops_monitor_and_publishes_terminal_state(self) -> None:
        planner = ScriptedPlanner(
            act(
                "Place the blue block flat in the marked area.",
                "Place the red block on the blue block.",
            )
        )
        stop_reasons = []
        with patch("prefmem.runtime.emergency_stop", side_effect=stop_reasons.append):
            runtime, _clock, publisher, monitor = self.build_runtime(
                planner,
                frame_sequences=(10, 20),
            )
            task = self.start_goal(runtime)["current_task"]

            accepted = runtime.emergency.trigger(
                "A person entered the robot workspace.",
                publication_id=task["publication_id"],
                observed_at=101.0,
            )
            duplicate = runtime.emergency.trigger(
                "A second report must not run the stop hook.",
                observed_at=102.0,
            )

        self.assertTrue(accepted)
        self.assertFalse(duplicate)
        self.assertEqual(stop_reasons, ["A person entered the robot workspace."])
        self.assertTrue(runtime.shutdown_event.is_set())
        self.assertIs(
            runtime.controller.snapshot.state,
            RecedingControllerState.EMERGENCY_STOPPED,
        )
        self.assertIs(publisher.displays[-1].state, DisplayState.EMERGENCY_STOPPED)
        self.assertEqual(monitor.stop_calls, 1)

    def test_stale_monitor_error_cannot_pause_a_replacement_task(self) -> None:
        planner = ScriptedPlanner(
            act("Place the blue block flat.", "Place the red block next."),
            act("Place the red block on the base.", "Validate the tower."),
        )
        runtime, clock, _publisher, monitor = self.build_runtime(planner)
        self.start_goal(runtime)
        first_task = monitor.published[0]
        self.emit_twice(runtime, clock, first_task, status="SUCCESS", frames=(21, 22))
        current = monitor.published[1]

        runtime._on_monitor_error(
            MonitorErrorEvent(
                kind=MonitorErrorKind.MODEL,
                message="late timeout",
                publication_id=first_task.publication_id,
            )
        )

        self.assertEqual(runtime.context_dict()["state"], "EXECUTING")
        self.assertEqual(
            runtime.context_dict()["current_task"]["publication_id"],
            current.publication_id,
        )


if __name__ == "__main__":
    unittest.main()
