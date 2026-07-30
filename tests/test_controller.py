from __future__ import annotations

import unittest
from datetime import UTC, datetime

from prefmem.agents.contracts import (
    ConditionCheck,
    ConditionState,
    ExpectedCondition,
    ExpectedOutcome,
    FailureKind,
    GoalCheck,
    HRIDecision,
    HRIDecisionKind,
    MemoryContext,
    MemoryRetrievalRequest,
    MemoryStatus,
    MonitorAction,
    MonitorProgress,
    MonitorResult,
    MonitorSafetyStatus,
    Observation,
    ObservationQuality,
    PlanResult,
    PlanningStatus,
    PredicateCondition,
    SafetyStatus,
    Subtask,
    TaskContract,
    TaskPhase,
    TaskStatus,
    TimeoutPolicy,
    ValidationResult,
    ValidationSpec,
    ValidatorOutcome,
    ValidatorRecoverability,
)
from prefmem.controller import ControllerLimits, PrefMemController
from prefmem.execution import CallbackVLAAdapter, IdempotentVLAAdapter
from prefmem.observations import ObservationEnvelope


class FakeObservationSource:
    def __init__(self) -> None:
        self.count = 0

    def capture(self, *, purpose: str) -> ObservationEnvelope:
        self.count += 1
        observation = Observation(
            observation_id=f"obs-{self.count}",
            sequence=self.count,
            captured_at=datetime.now(UTC),
            image_block={
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,frame-{self.count}"
                },
            },
        )
        return ObservationEnvelope(
            observation=observation,
            frame_path=None,
            source=purpose,
        )


class FakeMemory:
    def __init__(self) -> None:
        self.episodes = []
        self.retrieve_calls = []

    def retrieve(self, *, username, request):
        self.retrieve_calls.append((username, request))
        return MemoryContext(
            status=MemoryStatus.AVAILABLE,
            request_id=request.request_id,
            relevant_preferences=[
                {"record_id": "pref-1", "statement": "Use the mug."}
            ],
        )

    def save_episode(self, episode) -> None:
        self.episodes.append(episode)


class SequenceHRI:
    def __init__(self, decisions: list[HRIDecision]) -> None:
        self.decisions = decisions
        self.memory_statuses = []

    def decide(self, **kwargs) -> HRIDecision:
        self.memory_statuses.append(kwargs["memory"].status)
        return self.decisions.pop(0)


class FakePlanner:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.calls = []

    def plan(self, **kwargs) -> PlanResult:
        self.calls.append(kwargs)
        request = kwargs["planning_request"]
        task = kwargs["task_contract"]
        spec = ValidationSpec(
            spec_id=request.validation_spec_id,
            confirmed_intent=task.confirmed_intent,
            goal_conditions=[
                PredicateCondition(
                    goal_id="goal-1",
                    description="The banana is inside the mug.",
                    predicate="INSIDE",
                    arguments=["banana", "mug"],
                )
            ],
        )
        subtasks = (
            [
                Subtask(
                    subtask_id="subtask-1",
                    task_instruction="Place the banana inside the mug.",
                    expected_outcome=ExpectedOutcome(
                        conditions=[
                            ExpectedCondition(
                                condition_id="condition-1",
                                description="The banana is inside the mug.",
                                predicate="INSIDE",
                                arguments=["banana", "mug"],
                            )
                        ]
                    ),
                    timeout_policy=TimeoutPolicy(
                        timeout_seconds=30,
                        stall_seconds=10,
                        on_timeout="REPLAN",
                        on_stall="REOBSERVE",
                    ),
                )
            ]
            if self.ready
            else []
        )
        return PlanResult(
            planning_status=(
                PlanningStatus.READY
                if self.ready
                else PlanningStatus.ALREADY_SATISFIED
            ),
            plan_id=request.plan_id,
            plan_version=request.plan_version,
            planning_mode=request.planning_mode,
            subtasks=subtasks,
            validation_spec=spec,
            planner_confidence=1.0,
        )


class FakeMonitor:
    def __init__(self) -> None:
        self.calls = []

    def evaluate(self, request, *, frame_paths=()):
        self.calls.append(request)
        return MonitorResult(
            dispatch_id=request.dispatch_id,
            subtask_id=request.subtask_id,
            task_status=TaskStatus.SUCCESS,
            progress=MonitorProgress.VERIFYING,
            observation_quality=ObservationQuality.ADEQUATE,
            safety_status=MonitorSafetyStatus.SAFE,
            recommended_action=MonitorAction.ADVANCE,
            condition_checks=[
                ConditionCheck(
                    condition_id="condition-1",
                    state=ConditionState.SATISFIED,
                    evidence="Visible inside the mug.",
                    confidence=1.0,
                )
            ],
            confidence=1.0,
        )


class UnverifiedSuccessMonitor(FakeMonitor):
    def evaluate(self, request, *, frame_paths=()):
        self.calls.append(request)
        return MonitorResult(
            dispatch_id=request.dispatch_id,
            subtask_id=request.subtask_id,
            task_status=TaskStatus.SUCCESS,
            progress=MonitorProgress.VERIFYING,
            observation_quality=ObservationQuality.ADEQUATE,
            safety_status=MonitorSafetyStatus.SAFE,
            recommended_action=MonitorAction.ADVANCE,
            condition_checks=[
                ConditionCheck(
                    condition_id="condition-1",
                    state=ConditionState.UNKNOWN,
                    evidence="The mug rim occludes the banana.",
                    confidence=0.4,
                )
            ],
            confidence=0.7,
        )


class TerminalMonitor:
    def __init__(self, action: MonitorAction) -> None:
        self.action = action
        self.calls = []

    def evaluate(self, request, *, frame_paths=()):
        self.calls.append(request)
        unsafe = self.action == MonitorAction.ABORT_SAFETY
        return MonitorResult(
            dispatch_id=request.dispatch_id,
            subtask_id=request.subtask_id,
            task_status=TaskStatus.FAILURE,
            progress=MonitorProgress.DEVIATED,
            observation_quality=ObservationQuality.ADEQUATE,
            safety_status=(
                MonitorSafetyStatus.UNSAFE
                if unsafe
                else MonitorSafetyStatus.SAFE
            ),
            failure_kind=(
                FailureKind.SAFETY if unsafe else FailureKind.BLOCKED
            ),
            recommended_action=self.action,
            condition_checks=[
                ConditionCheck(
                    condition_id="condition-1",
                    state=ConditionState.VIOLATED,
                    evidence="The current subtask cannot continue.",
                    confidence=1.0,
                )
            ],
            confidence=1.0,
        )


class FakeValidator:
    def __init__(self, outcome: ValidatorOutcome = ValidatorOutcome.SUCCESS):
        self.outcome = outcome
        self.calls = []

    def validate(self, request, *, frame_paths=()):
        self.calls.append(request)
        success = self.outcome == ValidatorOutcome.SUCCESS
        return ValidationResult(
            spec_id=request.validation_spec.spec_id,
            outcome=self.outcome,
            task_complete=success,
            goal_checks=[
                GoalCheck(
                    goal_id="goal-1",
                    state=(
                        ConditionState.SATISFIED
                        if success
                        else ConditionState.UNKNOWN
                    ),
                    evidence=(
                        "Visible inside the mug."
                        if success
                        else "The rim occludes the banana."
                    ),
                    evidence_refs=[
                        request.terminal_observations[-1].observation_id
                    ],
                    confidence=1.0 if success else 0.4,
                )
            ],
            safety_status=SafetyStatus.SAFE,
            recoverability=(
                ValidatorRecoverability.NONE
                if success
                else ValidatorRecoverability.USER_ASSIST
            ),
            user_message=(
                "The banana is inside the mug."
                if success
                else "I cannot verify that the banana is inside the mug."
            ),
            validator_confidence=1.0 if success else 0.4,
        )


def submit_decision() -> HRIDecision:
    return HRIDecision(
        decision=HRIDecisionKind.SUBMIT_TASK,
        reply_to_user="Got it.",
        task_contract=TaskContract(
            task_type="object placement",
            confirmed_intent="Place the banana inside the mug.",
            objects=["banana", "mug"],
        ),
        reason_code="TASK_RESOLVED",
    )


class PrefMemControllerTests(unittest.TestCase):
    def make_controller(
        self,
        *,
        hri,
        plan_only: bool,
        validator=None,
        vla=None,
        monitor=None,
    ):
        source = FakeObservationSource()
        memory = FakeMemory()
        planner = FakePlanner()
        monitor = monitor or FakeMonitor()
        validator = validator or FakeValidator()
        controller = PrefMemController(
            username="alice",
            observation_source=source,
            hri=hri,
            planner=planner,
            monitor=monitor,
            validator=validator,
            memory=memory,
            vla=vla,
            plan_only=plan_only,
            limits=ControllerLimits(
                monitor_interval_seconds=0,
                success_confirmations=2,
            ),
            wait=lambda _: None,
            clock=lambda: 0.0,
        )
        return controller, source, memory, planner, monitor, validator

    def test_plan_only_never_publishes_or_claims_execution(self) -> None:
        published = []
        vla = IdempotentVLAAdapter(CallbackVLAAdapter(published.append))
        controller, _, memory, planner, monitor, validator = self.make_controller(
            hri=SequenceHRI([submit_decision()]),
            plan_only=True,
            vla=vla,
        )

        result = controller.handle_user("Put the banana in the mug.")

        self.assertEqual(result.phase, TaskPhase.COMPLETE)
        self.assertIn("Plan-only mode", result.text)
        self.assertIn("no VLA command", result.text)
        self.assertEqual(published, [])
        self.assertEqual(monitor.calls, [])
        self.assertEqual(validator.calls, [])
        self.assertEqual(len(memory.episodes), 1)
        self.assertEqual(
            memory.episodes[0].result["status"],
            "PLANNED_NOT_EXECUTED",
        )
        request = planner.calls[0]["planning_request"]
        self.assertEqual(result.plan.plan_id, request.plan_id)
        self.assertEqual(
            result.plan.validation_spec.spec_id,
            request.validation_spec_id,
        )

    def test_execution_advances_only_after_two_monitor_confirmations_and_validation(
        self,
    ) -> None:
        published = []
        vla = IdempotentVLAAdapter(CallbackVLAAdapter(published.append))
        controller, _, memory, _, monitor, validator = self.make_controller(
            hri=SequenceHRI([submit_decision()]),
            plan_only=False,
            vla=vla,
        )

        result = controller.handle_user("Put the banana in the mug.")

        self.assertEqual(result.phase, TaskPhase.COMPLETE)
        self.assertEqual(len(published), 1)
        self.assertEqual(len(monitor.calls), 2)
        self.assertEqual(len(validator.calls), 1)
        self.assertTrue(result.validation.task_complete)
        self.assertEqual(
            memory.episodes[0].execution["completed_subtasks"],
            ["subtask-1"],
        )

    def test_controller_rejects_monitor_success_with_unverified_required_goal(
        self,
    ) -> None:
        published = []
        monitor = UnverifiedSuccessMonitor()
        validator = FakeValidator()
        vla = IdempotentVLAAdapter(CallbackVLAAdapter(published.append))
        controller, _, _, _, _, _ = self.make_controller(
            hri=SequenceHRI([submit_decision()]),
            plan_only=False,
            validator=validator,
            vla=vla,
            monitor=monitor,
        )

        result = controller.handle_user("Put the banana in the mug.")

        self.assertEqual(result.phase, TaskPhase.FAILED)
        self.assertEqual(len(published), 1)
        self.assertEqual(len(monitor.calls), 1)
        self.assertEqual(validator.calls, [])
        self.assertIn("every required condition SATISFIED", result.text)

    def test_safety_abort_cancels_the_published_dispatch(self) -> None:
        published = []
        cancellations = []
        vla = IdempotentVLAAdapter(
            CallbackVLAAdapter(
                published.append,
                cancel_callback=lambda dispatch_id, reason: cancellations.append(
                    (dispatch_id, reason)
                ),
            )
        )
        controller, _, memory, _, _, validator = self.make_controller(
            hri=SequenceHRI([submit_decision()]),
            plan_only=False,
            monitor=TerminalMonitor(MonitorAction.ABORT_SAFETY),
            vla=vla,
        )

        result = controller.handle_user("Put the banana in the mug.")

        self.assertEqual(result.phase, TaskPhase.SAFETY_STOP)
        self.assertEqual(len(published), 1)
        self.assertEqual(
            cancellations,
            [(published[0].dispatch_id, "live_monitor_safety_abort")],
        )
        cancellation = memory.episodes[0].execution["dispatches"][0][
            "cancellation"
        ]
        self.assertTrue(cancellation["accepted"])
        self.assertEqual(validator.calls, [])

    def test_non_safety_terminal_stop_cancels_the_published_dispatch(self) -> None:
        published = []
        cancellations = []
        vla = IdempotentVLAAdapter(
            CallbackVLAAdapter(
                published.append,
                cancel_callback=lambda dispatch_id, reason: cancellations.append(
                    (dispatch_id, reason)
                ),
            )
        )
        controller, _, _, _, _, validator = self.make_controller(
            hri=SequenceHRI([submit_decision()]),
            plan_only=False,
            monitor=TerminalMonitor(MonitorAction.USER_ASSIST),
            vla=vla,
        )

        result = controller.handle_user("Put the banana in the mug.")

        self.assertEqual(result.phase, TaskPhase.FAILED)
        self.assertEqual(
            cancellations,
            [(published[0].dispatch_id, "live_monitor_user_assist")],
        )
        self.assertEqual(validator.calls, [])

    def test_unknown_validation_is_not_reported_as_success(self) -> None:
        published = []
        validator = FakeValidator(ValidatorOutcome.UNKNOWN)
        vla = IdempotentVLAAdapter(CallbackVLAAdapter(published.append))
        controller, _, _, _, _, _ = self.make_controller(
            hri=SequenceHRI([submit_decision()]),
            plan_only=False,
            validator=validator,
            vla=vla,
        )

        result = controller.handle_user("Put the banana in the mug.")

        self.assertEqual(result.phase, TaskPhase.FAILED)
        self.assertFalse(result.validation.task_complete)
        self.assertIn("cannot verify", result.text)

    def test_hri_memory_retrieval_is_host_routed_before_response(self) -> None:
        hri = SequenceHRI(
            [
                HRIDecision(
                    decision=HRIDecisionKind.RETRIEVE_MEMORY,
                    memory_request=MemoryRetrievalRequest(
                        request_id="memory-1",
                        search_text="usual mug",
                        reason_code="PREFERENCE_SENSITIVE",
                    ),
                    reason_code="MEMORY_NEEDED",
                ),
                HRIDecision(
                    decision=HRIDecisionKind.RESPOND,
                    reply_to_user="Your saved mug preference is available.",
                    memory_refs_used=["pref-1"],
                    reason_code="DIRECT_RESPONSE",
                ),
            ]
        )
        controller, _, memory, _, _, _ = self.make_controller(
            hri=hri,
            plan_only=True,
        )

        result = controller.handle_user("What mug do I usually use?")

        self.assertEqual(result.phase, TaskPhase.COMPLETE)
        self.assertEqual(len(memory.retrieve_calls), 1)
        self.assertEqual(
            hri.memory_statuses,
            [MemoryStatus.NOT_RETRIEVED, MemoryStatus.AVAILABLE],
        )


if __name__ == "__main__":
    unittest.main()
