from __future__ import annotations

import unittest
from unittest.mock import patch

from prefmem.agents.monitor import CapturedFrame, MonitorErrorEvent, MonitorErrorKind
from prefmem.agents.validator import ValidatorErrorEvent, ValidatorErrorKind
from prefmem.contracts import (
    GoalProposal,
    MonitorAssessment,
    PlannerDecision,
    ValidationAssessment,
    ValidationChecklistDraft,
    freeze_validation_contract,
)
from prefmem.controller import (
    RecedingControllerState,
    RecedingHorizonController,
    RecedingResult,
    RecedingTransition,
)
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


class RecordingValidator:
    def __init__(self) -> None:
        self.compile_calls = []
        self.compile_error = None
        self.published = []
        self.contracts = []
        self.retired = []
        self.stop_calls = 0
        self.events = []

    def compile_contract(self, goal, confirmation_frame):
        self.compile_calls.append((goal, confirmation_frame))
        if self.compile_error is not None:
            raise self.compile_error
        draft = ValidationChecklistDraft.from_model_output(
            {
                "broad_items": [
                    {
                        "broad_index": index,
                        "detailed_criteria": [
                            f"The camera visibly verifies outcome {index + 1}."
                        ],
                    }
                    for index, _label in enumerate(
                        goal.final_expected_observation
                    )
                ]
            }
        )
        return freeze_validation_contract(
            goal,
            draft,
            validation_id=f"{goal.goal_id}:validation",
        )

    def publish(self, task, contract) -> None:
        self.published.append(task)
        self.contracts.append(contract)
        self.events.append(("publish", task.publication_id))

    def retire(self, publication_id=None) -> bool:
        self.retired.append(publication_id)
        self.events.append(("retire", publication_id))
        return True

    def stop(self) -> None:
        self.stop_calls += 1
        self.events.append(("stop", None))


class RecordingExecutor:
    def __init__(self) -> None:
        self.published = []
        self.retired = []
        self.settled = set()
        self.stop_calls = 0
        self.emergency_calls = 0
        self.on_event = None
        self.on_scene_change = None

    def publish(self, task, scope=None) -> None:
        self.published.append((task, scope))

    def retire(self, publication_id) -> bool:
        self.retired.append(publication_id)
        return True

    def is_settled(self, publication_id) -> bool:
        return publication_id in self.settled

    def emergency_stop(self) -> None:
        self.emergency_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


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


class OpenSetPlanner(ScriptedPlanner):
    def preview(self, *args, **kwargs) -> GoalProposal:
        proposal = super().preview(*args, **kwargs)
        payload = proposal.to_dict()
        payload["dynamic_object_scope"] = {
            "selector": "block",
            "region": "robot_workspace",
            "membership_rule": "PRESENT_AT_VALIDATION",
        }
        return GoalProposal.from_dict(payload)


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


def validation_assessment(
    contract,
    task,
    *,
    state: str,
    observed_at: float,
    frame: int,
):
    evidence = {
        "MET": "The requested final outcome is clearly visible.",
        "NOT_MET": "The red block is visibly beside the blue block.",
        "UNKNOWN": "The requested final outcome is occluded.",
    }[state]
    return ValidationAssessment.from_model_output(
        contract,
        {
            "criteria": [
                {
                    "id": criterion.criterion_id,
                    "state": state,
                    "evidence": evidence,
                }
                for criterion in contract.detailed_criteria
            ],
            "observation": "The final scene is visible from the current view.",
        },
        publication_id=task.publication_id,
        observed_at=observed_at,
        frame_sequence=frame,
    )


class RuntimeOrchestrationTests(unittest.TestCase):
    def build_runtime(
        self,
        planner,
        *,
        frame_sequences=(10, 20, 30, 40),
        executor=None,
    ):
        clock = FakeClock()
        publisher = RecordingPublisher()
        monitor = RecordingMonitor()
        validator = RecordingValidator()
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
            validator=validator,
            session_id="test-session",
            executor=executor,
        )
        self.addCleanup(runtime.close)
        return runtime, clock, publisher, monitor

    def test_executor_receives_open_scope_and_monitor_waits_for_settle(self):
        planner = OpenSetPlanner(
            act(
                "Place the blue block flat in the marked area.",
                "Place the red block on the blue block.",
            ),
            final_validation(),
        )
        executor = RecordingExecutor()
        runtime, clock, _publisher, monitor = self.build_runtime(
            planner,
            executor=executor,
        )
        self.start_goal(runtime)
        current = monitor.published[0]
        self.assertEqual(executor.published[0][0], current)
        self.assertEqual(executor.published[0][1].selector, "block")

        self.emit_twice(
            runtime,
            clock,
            current,
            status="SUCCESS",
            frames=(31, 32),
        )
        self.assertEqual(runtime.context_dict()["state"], "EXECUTING")
        self.assertEqual(len(planner.plan_calls), 1)
        moving_events = []
        while not runtime.monitor_events.empty():
            moving_events.append(runtime.monitor_events.get_nowait())
        moving_assessments = [
            event for event in moving_events if event.get("event") == "ASSESSMENT"
        ]
        self.assertEqual(len(moving_assessments), 2)
        self.assertTrue(
            all(
                event["disposition"] == "IGNORED_EXECUTOR_ACTIVE"
                for event in moving_assessments
            )
        )
        self.assertTrue(
            all(event["executor_state"] == "ACTIVE" for event in moving_assessments)
        )

        executor.settled.add(current.publication_id)
        self.emit_twice(
            runtime,
            clock,
            current,
            status="SUCCESS",
            frames=(33, 34),
        )
        self.assertEqual(len(planner.plan_calls), 2)
        self.assertIn(current.publication_id, executor.retired)
        settled_events = []
        while not runtime.monitor_events.empty():
            settled_events.append(runtime.monitor_events.get_nowait())
        settled_assessments = [
            event for event in settled_events if event.get("event") == "ASSESSMENT"
        ]
        self.assertEqual(
            [event["disposition"] for event in settled_assessments],
            ["ACCEPTED", "REPLAN_REQUESTED"],
        )
        self.assertEqual(
            settled_assessments[0]["success_confirmation"],
            {"count": 1, "required": 2},
        )

    def test_monitor_telemetry_is_bounded_and_drops_oldest(self):
        runtime, _clock, _publisher, _monitor = self.build_runtime(
            ScriptedPlanner(final_validation())
        )

        for index in range(300):
            runtime._queue_monitor_event(
                {"event": "INFERENCE_STARTED", "frame_sequence": index}
            )

        self.assertEqual(runtime.monitor_events.qsize(), 256)
        first = runtime.monitor_events.get_nowait()
        self.assertEqual(first["frame_sequence"], 44)

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

    def emit_validation_twice(
        self,
        runtime,
        clock,
        task,
        *,
        state,
        frames,
    ):
        contract = runtime.validator.contracts[-1]
        for frame in frames:
            runtime._on_validator_assessment(
                validation_assessment(
                    contract,
                    task,
                    state=state,
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
        validator = runtime.validator

        context = self.start_goal(runtime)
        first_task = monitor.published[0]

        self.assertEqual(context["state"], "EXECUTING")
        self.assertEqual(first_task.instruction, "Place the blue block flat in the marked area.")
        self.assertNotIn(
            "Place the red block on the blue block.",
            [task.instruction for task in monitor.published],
        )
        self.assertEqual(len(validator.compile_calls), 1)
        self.assertEqual(validator.compile_calls[0][1].sequence, 20)
        self.assertEqual(planner.plan_calls[0][0].frame_sequence, 30)

        runtime._on_monitor_assessment(
            assessment(
                first_task,
                status="SUCCESS",
                observed_at=clock.advance(),
                frame=31,
            )
        )
        confirming_display = publisher.displays[-1]
        self.assertIs(confirming_display.state, DisplayState.ACTIVE)
        self.assertIn("Expected observation detected", confirming_display.message)
        self.assertIn("confirming that it remains stable", confirming_display.message)
        runtime._on_monitor_assessment(
            assessment(
                first_task,
                status="SUCCESS",
                observed_at=clock.advance(),
                frame=32,
            )
        )

        self.assertEqual(len(planner.plan_calls), 2)
        second_request = planner.plan_calls[1][0]
        self.assertEqual(second_request.cycle_id, 2)
        self.assertEqual(second_request.trigger.value, "TASK_SUCCESS")
        self.assertEqual(second_request.frame_sequence, 40)
        self.assertEqual(second_request.execution_history[0].outcome.value, "SUCCESS")
        self.assertEqual(monitor.retired, [first_task.publication_id])

        self.assertEqual(len(monitor.published), 1)
        validation_task = validator.published[0]
        self.assertEqual(validation_task.phase.value, "FINAL_VALIDATION")
        self.assertIn("Keep the scene unchanged", validation_task.instruction)
        final_display = publisher.displays[-1]
        self.assertIs(final_display.state, DisplayState.FINAL_VALIDATION)
        self.assertIsNone(final_display.instruction)
        self.assertIn("Keep the scene unchanged", final_display.message)
        self.assertIn("move only the camera", final_display.message)
        self.assertEqual(
            final_display.expected_observation,
            tuple(
                runtime.controller.snapshot.goal.final_expected_observation
            ),
        )

        runtime._on_validator_assessment(
            validation_assessment(
                validator.contracts[0],
                validation_task,
                state="MET",
                observed_at=clock.advance(),
                frame=41,
            )
        )
        self.assertTrue(runtime.notifications.empty())
        self.emit_validation_twice(
            runtime,
            clock,
            validation_task,
            state="MET",
            frames=(42,),
        )

        self.assertEqual(runtime.context_dict()["state"], "COMPLETE")
        self.assertIs(publisher.displays[-1].state, DisplayState.COMPLETE)
        self.assertEqual(
            monitor.retired,
            [first_task.publication_id],
        )
        self.assertEqual(validator.retired, [validation_task.publication_id])
        report = runtime.notifications.get_nowait()
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["checklist"][0]["state"], "MET")
        self.assertNotIn("evidence", report["checklist"][0])
        self.assertNotIn("observation", report)
        validation_context = runtime.context_dict()["validation"]
        self.assertNotIn("detailed", str(validation_context).lower())

    def test_stale_active_display_cannot_overwrite_newer_attention(self) -> None:
        planner = ScriptedPlanner(
            act(
                "Place the blue block flat in the marked area.",
                "Place the red block on the blue block.",
            )
        )
        runtime, _clock, publisher, _monitor = self.build_runtime(planner)
        self.start_goal(runtime)
        stale_active = runtime.controller.snapshot

        attention = runtime.controller.record_system_error(
            "Monitoring timed out."
        )
        self.assertTrue(runtime._publish_transition(attention))
        published_count = len(publisher.displays)
        stale_transition = RecedingTransition(
            result=RecedingResult.ACCEPTED,
            snapshot=stale_active,
        )

        self.assertFalse(runtime._publish_transition(stale_transition))
        self.assertEqual(len(publisher.displays), published_count)
        self.assertIs(publisher.displays[-1].state, DisplayState.NEEDS_ATTENTION)

    def test_final_incomplete_is_confirmed_then_replanned_with_evidence(self) -> None:
        planner = ScriptedPlanner(
            act("Place the blue block flat.", "Place the red block next."),
            final_validation(),
            act(
                "Place the red block back on the blue block.",
                "Validate the repaired tower.",
            ),
        )
        runtime, clock, publisher, monitor = self.build_runtime(
            planner,
            frame_sequences=(10, 20, 30, 40, 50),
        )
        self.start_goal(runtime)
        step = monitor.published[0]
        self.emit_twice(runtime, clock, step, status="SUCCESS", frames=(31, 32))
        validation = runtime.validator.published[0]

        runtime._on_validator_assessment(
            validation_assessment(
                runtime.validator.contracts[0],
                validation,
                state="NOT_MET",
                observed_at=clock.advance(),
                frame=41,
            )
        )
        self.assertTrue(runtime.notifications.empty())
        runtime._on_validator_assessment(
            validation_assessment(
                runtime.validator.contracts[0],
                validation,
                state="NOT_MET",
                observed_at=clock.advance(),
                frame=42,
            )
        )

        self.assertEqual(runtime.context_dict()["state"], "EXECUTING")
        self.assertIs(publisher.displays[-1].state, DisplayState.ACTIVE)
        self.assertEqual(len(monitor.published), 2)
        self.assertEqual(
            monitor.published[-1].instruction,
            "Place the red block back on the blue block.",
        )
        self.assertEqual(
            runtime.validator.retired,
            [validation.publication_id],
        )
        replan = planner.plan_calls[2][0]
        self.assertEqual(replan.trigger.value, "FINAL_VALIDATION_FAIL")
        record = replan.execution_history[-1]
        self.assertEqual(record.outcome.value, "FINAL_VALIDATION_FAIL")
        self.assertIn("red block rests on the blue block", record.failure_reason)
        self.assertIn("visibly beside the blue block", record.failure_reason)
        report = runtime.notifications.get_nowait()
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["checklist"][0]["state"], "NOT_MET")
        self.assertIn("replanning", report["next_action"])

    def test_needs_evidence_notifies_once_then_timeout_can_resume_validator(self) -> None:
        planner = ScriptedPlanner(
            act("Place the blue block flat.", "Place the red block next."),
            final_validation(),
        )
        runtime, clock, publisher, monitor = self.build_runtime(
            planner,
            frame_sequences=(10, 20, 30, 40, 50),
        )
        self.start_goal(runtime)
        step = monitor.published[0]
        self.emit_twice(runtime, clock, step, status="SUCCESS", frames=(31, 32))
        validation = runtime.validator.published[0]
        contract = runtime.validator.contracts[0]

        for frame in (41, 42):
            runtime._on_validator_assessment(
                validation_assessment(
                    contract,
                    validation,
                    state="UNKNOWN",
                    observed_at=clock.advance(),
                    frame=frame,
                )
            )

        self.assertEqual(runtime.context_dict()["state"], "FINAL_VALIDATION")
        self.assertEqual(len(planner.plan_calls), 2)
        report = runtime.notifications.get_nowait()
        self.assertEqual(report["status"], "NEEDS_EVIDENCE")
        self.assertEqual(report["checklist"][0]["state"], "UNKNOWN")
        self.assertTrue(report["evidence_requests"])
        self.assertTrue(runtime.notifications.empty())

        clock.advance(31)
        self.assertTrue(runtime.check_timeout())
        self.assertEqual(runtime.context_dict()["state"], "NEEDS_ATTENTION")
        paused_report = runtime.notifications.get_nowait()
        self.assertIn("paused", paused_report["next_action"])
        self.assertEqual(
            runtime.validator.retired,
            [validation.publication_id],
        )

        runtime.resume_current_task()

        resumed = runtime.validator.published[-1]
        self.assertEqual(runtime.context_dict()["state"], "FINAL_VALIDATION")
        self.assertNotEqual(resumed.publication_id, validation.publication_id)
        self.assertIs(runtime.validator.contracts[-1], contract)
        self.assertIsNone(
            runtime.context_dict()["validation"]["latest_report"]
        )
        self.assertIs(publisher.displays[-1].state, DisplayState.FINAL_VALIDATION)

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
        self.assertEqual(runtime.validator.compile_calls, [])
        self.assertEqual(runtime.validator.published, [])
        self.assertEqual(publisher.displays, [])

    def test_checklist_compile_failure_leaves_exact_goal_confirmable(self) -> None:
        planner = ScriptedPlanner(
            act("Place the blue block flat.", "Place the red block next.")
        )
        runtime, _clock, publisher, monitor = self.build_runtime(
            planner,
            frame_sequences=(10, 20),
        )
        preview = runtime.request_goal_preview("Build a stable two-block tower.")
        contract = preview["goal_contract"]
        runtime.validator.compile_error = RuntimeError("model unavailable")

        with self.assertRaisesRegex(
            RuntimeError,
            "checklist compilation failed: model unavailable",
        ):
            runtime.confirm_goal(
                goal_id=contract["goal_id"],
                revision=contract["revision"],
                confirmed=True,
            )

        self.assertEqual(runtime.context_dict()["state"], "AWAITING_CONFIRMATION")
        self.assertEqual(runtime.pending_goal.goal_id, contract["goal_id"])
        self.assertEqual(planner.plan_calls, [])
        self.assertEqual(monitor.published, [])
        self.assertEqual(runtime.validator.published, [])
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
            frame_sequences=(10, 20, 30, 40, 50, 60, 70),
        )
        self.start_goal(runtime)
        first_session = runtime.session_id
        first_task = monitor.published[0]
        self.emit_twice(runtime, clock, first_task, status="SUCCESS", frames=(31, 32))
        validation = runtime.validator.published[0]
        self.emit_validation_twice(
            runtime,
            clock,
            validation,
            state="MET",
            frames=(41, 42),
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

        self.emit_twice(runtime, clock, failed_task, status="FAIL", frames=(31, 32))

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
            frame_sequences=(10, 20, 30),
        )
        self.start_goal(runtime)

        runtime.close()

        self.assertEqual(monitor.stop_calls, 1)
        self.assertEqual(runtime.validator.stop_calls, 1)
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
                validator=RecordingValidator(),
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
                frame_sequences=(10, 20, 30),
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
        self.assertEqual(runtime.validator.stop_calls, 1)

    def test_stale_monitor_error_cannot_pause_a_replacement_task(self) -> None:
        planner = ScriptedPlanner(
            act("Place the blue block flat.", "Place the red block next."),
            act("Place the red block on the base.", "Validate the tower."),
        )
        runtime, clock, _publisher, monitor = self.build_runtime(planner)
        self.start_goal(runtime)
        first_task = monitor.published[0]
        self.emit_twice(runtime, clock, first_task, status="SUCCESS", frames=(31, 32))
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

    def test_monitor_has_no_authority_over_final_validation(self) -> None:
        planner = ScriptedPlanner(
            act("Place the blue block flat.", "Place the red block next."),
            final_validation(),
        )
        runtime, clock, _publisher, monitor = self.build_runtime(planner)
        self.start_goal(runtime)
        step = monitor.published[0]
        self.emit_twice(runtime, clock, step, status="SUCCESS", frames=(31, 32))
        validation = runtime.validator.published[0]

        for frame in (41, 42):
            runtime._on_monitor_assessment(
                assessment(
                    validation,
                    status="SUCCESS",
                    observed_at=clock.advance(),
                    frame=frame,
                )
            )
        runtime._on_validator_error(
            ValidatorErrorEvent(
                kind=ValidatorErrorKind.MODEL,
                message="late error from an older publication",
                publication_id="stale-publication",
            )
        )

        context = runtime.context_dict()
        self.assertEqual(context["state"], "FINAL_VALIDATION")
        self.assertEqual(
            context["current_task"]["publication_id"],
            validation.publication_id,
        )
        self.assertIsNone(context["validation"]["latest_report"])


if __name__ == "__main__":
    unittest.main()
