from __future__ import annotations

from dataclasses import replace
import unittest

from experiments.harness.assembler import SystemAssembler
from experiments.harness.profiles import load_profiles
from experiments.harness.variants import (
    DirectPlannerInteractionAdapter,
    DisabledMemoryAdapter,
    DisabledValidatorAdapter,
    FeedbackPlanningAdapter,
    FrozenOpenLoopPlanningAdapter,
    OraclePlanningAdapter,
    PlannerSelfCompletionAdapter,
    SyntheticSettledMonitorAdapter,
    VariantExecutionError,
    VariantConfigurationError,
)


class VariantBackend:
    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        self.requests = []

    def invoke(self, model_input, **kwargs):
        self.requests.append((model_input, kwargs))
        return {
            "actions": ["pick:red", "place:target"],
            "complete": True,
            "payload": model_input,
        }

    def stream(self, model_input, **kwargs):
        yield self.invoke(model_input, **kwargs)

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self


class VariantFactory:
    def __init__(self) -> None:
        self.backends = {}

    def __call__(self, request):
        backend = VariantBackend(request.model_instance_id)
        self.backends[request.model_instance_id] = backend
        return backend


def _assemble(profile_id: str):
    return SystemAssembler(
        profile_id,
        executor="synthetic_event",
        backend_factory=VariantFactory() if profile_id != "B3" else None,
    ).assemble()


class ExperimentVariantTests(unittest.TestCase):
    def test_every_profile_resolves_to_executable_variant_components(self) -> None:
        for profile_id in load_profiles().profiles:
            with self.subTest(profile_id=profile_id):
                system = _assemble(profile_id)
                self.assertEqual(system.variant.profile_id, profile_id)
                self.assertIsNotNone(system.variant.memory)
                self.assertIsNotNone(system.variant.interaction)
                self.assertIsNotNone(system.variant.planning)
                self.assertIsNotNone(system.variant.monitor)
                self.assertIsNotNone(system.variant.validator)

    def test_unknown_mechanism_mode_fails_before_backend_construction(self) -> None:
        profile = load_profiles().get("T5")
        profile = replace(
            profile,
            mechanism_modes={**profile.mechanism_modes, "planning": "typo_mode"},
        )
        factory = VariantFactory()
        with self.assertRaisesRegex(VariantConfigurationError, "unsupported planning"):
            SystemAssembler(
                profile,
                executor="synthetic_event",
                backend_factory=factory,
            ).assemble()
        self.assertEqual(factory.backends, {})

    def test_disabled_memory_is_empty_and_cannot_mutate(self) -> None:
        for profile_id in ("A-Memory", "B4"):
            with self.subTest(profile_id=profile_id):
                memory = _assemble(profile_id).variant.memory
                self.assertIsInstance(memory, DisabledMemoryAdapter)
                self.assertEqual(memory.retrieve("usual order"), ())
                result = memory.mutate("create", {"preference": "red top"}, authorized=True)
                self.assertFalse(result.applied)
                self.assertEqual(result.reason, "memory_unavailable")

    def test_no_hri_routes_raw_request_directly_to_planner_preview(self) -> None:
        system = _assemble("A-HRI")
        self.assertIsInstance(system.variant.interaction, DirectPlannerInteractionAdapter)
        response = system.variant.interaction.preview("move red", {"objects": ["red"]})
        self.assertIn("payload", response)
        planner = system.routed_models["planner"]
        assert planner is not None
        record = planner.context_ledger[-1]
        self.assertEqual(record.logical_role, "planner")
        self.assertEqual(record.phase, "DIRECT_PREVIEW")
        self.assertEqual(record.request["input"]["raw_user_request"], "move red")
        confirmation = system.variant.confirmation
        self.assertFalse(
            confirmation.authorize(
                preview_status="READY",
                staged_goal_id="g1",
                staged_revision=2,
                confirmed_goal_id="g1",
                confirmed_revision=1,
                user_confirmed=True,
            )
        )
        self.assertTrue(
            confirmation.authorize(
                preview_status="READY",
                staged_goal_id="g1",
                staged_revision=2,
                confirmed_goal_id="g1",
                confirmed_revision=2,
                user_confirmed=True,
            )
        )

    def test_open_loop_plans_once_and_ignores_later_observations(self) -> None:
        system = _assemble("A-OpenLoop")
        planning = system.variant.planning
        self.assertIsInstance(planning, FrozenOpenLoopPlanningAdapter)
        self.assertEqual(
            planning.plan_once(goal="stack", scene="scene-0"),
            ("pick:red", "place:target"),
        )
        self.assertEqual(planning.next_action(observation="changed-scene"), "pick:red")
        self.assertEqual(planning.next_action(observation="another-change"), "place:target")
        self.assertIsNone(planning.next_action(observation="ignored"))
        with self.assertRaisesRegex(VariantExecutionError, "already frozen"):
            planning.freeze(["replacement"])
        with self.assertRaisesRegex(VariantExecutionError, "already frozen"):
            planning.plan_once(goal="replacement", scene="later-scene")
        planner = system.routed_models["planner"]
        assert planner is not None
        self.assertEqual(len(planner.context_ledger), 1)

    def test_synthetic_settled_monitor_never_treats_fault_as_evidence(self) -> None:
        monitor = _assemble("A-Monitor").variant.monitor
        self.assertIsInstance(monitor, SyntheticSettledMonitorAdapter)
        self.assertEqual(monitor.assess({"state": "SETTLED"}).status, "MET")
        self.assertEqual(monitor.assess({"state": "FAULT"}).status, "SYSTEM_ERROR")
        self.assertEqual(monitor.assess({"state": "RUNNING"}).status, "UNKNOWN")

    def test_no_validator_uses_fresh_planner_self_completion(self) -> None:
        system = _assemble("A-Validator")
        validator = system.variant.validator
        self.assertIsInstance(validator, PlannerSelfCompletionAdapter)
        frozen = validator.compile("goal-contract", "confirmation-frame")
        decision = validator.assess(frozen, "fresh-final-frame")
        self.assertTrue(decision.complete)
        planner = system.routed_models["planner"]
        assert planner is not None
        self.assertEqual(planner.context_ledger[-1].phase, "DECLARE_COMPLETE")
        self.assertEqual(
            planner.context_ledger[-1].request["input"]["fresh_final_frame"],
            "fresh-final-frame",
        )

    def test_auto_confirmation_and_single_evidence_are_exact_modes(self) -> None:
        auto = _assemble("A-Confirmation").variant.confirmation
        self.assertTrue(
            auto.authorize(
                preview_status="READY",
                staged_goal_id="g1",
                staged_revision=1,
            )
        )
        self.assertFalse(
            auto.authorize(
                preview_status="BLOCKED",
                staged_goal_id="g1",
                staged_revision=1,
            )
        )

        evidence = _assemble("A-SingleEvidence").variant.evidence
        self.assertEqual(evidence.success_confirmations, 1)
        self.assertEqual(evidence.failure_confirmations, 1)
        self.assertEqual(evidence.success_stability_seconds, 0)
        self.assertTrue(evidence.terminal("MET", consecutive_count=1, stable_seconds=0))

        repeated = _assemble("T5").variant.evidence
        self.assertFalse(repeated.terminal("MET", consecutive_count=1, stable_seconds=10))
        self.assertTrue(repeated.terminal("MET", consecutive_count=2, stable_seconds=2))

    def test_dynamic_guard_freeze_is_observably_different_from_live_guard(self) -> None:
        confirmed = frozenset({"red-1"})
        current = frozenset({"red-1", "red-2"})
        live = _assemble("T5").variant.dynamic_guard.evaluate(
            confirmed_members=confirmed,
            current_members=current,
            motion_active=True,
        )
        frozen = _assemble("A-DynamicGuard").variant.dynamic_guard.evaluate(
            confirmed_members=confirmed,
            current_members=current,
            motion_active=True,
        )
        self.assertTrue(live.cancel_active_motion)
        self.assertEqual(live.effective_members, current)
        self.assertFalse(frozen.cancel_active_motion)
        self.assertEqual(frozen.effective_members, confirmed)

    def test_host_fence_ablation_accepts_stale_identity_only_in_synthetic_mode(self) -> None:
        stale = {"publication_id": "old", "step_id": "wrong", "criterion_id": "c9"}
        kwargs = {
            "active_publication_id": "active",
            "active_step_id": "step-1",
            "active_criterion_ids": frozenset({"c1"}),
        }
        self.assertFalse(_assemble("T5").variant.host_fence.accepts(stale, **kwargs))
        self.assertTrue(
            _assemble("A-HostFence").variant.host_fence.accepts(stale, **kwargs)
        )

    def test_baseline_drivers_are_distinct_and_executable(self) -> None:
        b1 = _assemble("B1").variant
        self.assertIsInstance(b1.planning, FrozenOpenLoopPlanningAdapter)
        self.assertIsInstance(b1.memory, DisabledMemoryAdapter)
        self.assertIsInstance(b1.validator, DisabledValidatorAdapter)
        b1.planning.plan_once(goal="goal", scene="scene")
        self.assertEqual(b1.planning.next_action(), "pick:red")

        b2 = _assemble("B2").variant
        self.assertIsInstance(b2.planning, FeedbackPlanningAdapter)
        response = b2.planning.next_action(
            instruction="stack",
            observation="frame",
            previous_action="pick",
            textual_feedback="success",
            history=[],
        )
        self.assertTrue(response["complete"])
        self.assertEqual(b2.monitor.assess("success").status, "MET")

        b3 = _assemble("B3").variant
        self.assertIsInstance(b3.planning, OraclePlanningAdapter)
        self.assertEqual(
            b3.planning.next_action(hidden_state={"oracle_actions": ["oracle-pick"]}),
            "oracle-pick",
        )
        self.assertEqual(b3.monitor.assess({"step_success": True}).status, "MET")
        self.assertTrue(
            b3.validator.assess({"goal_completed": True}).complete
        )

        b4 = _assemble("B4").variant
        self.assertIsInstance(b4.memory, DisabledMemoryAdapter)
        self.assertTrue(b4.confirmation.authorize(
            preview_status="READY",
            staged_goal_id="g",
            staged_revision=1,
            confirmed_goal_id="g",
            confirmed_revision=1,
            user_confirmed=True,
        ))


if __name__ == "__main__":
    unittest.main()
