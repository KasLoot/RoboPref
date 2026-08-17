from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments_suite_v2.artifacts import (
    verify_artifact_manifest,
    verify_run_checksums,
)
from experiments_suite_v2.io import iter_jsonl, load_json
from experiments_suite_v2.registry import (
    CAMERA_APPROVAL_PATH,
    PROTOCOL_MANIFEST_PATH,
    REGISTRY_PATH,
)
from experiments_suite_v2.runners.system import (
    ExternalServiceInterruption,
    SystemTrial,
    expand_system_trials,
)
from experiments_suite_v2.runners.system_bundle import (
    AttemptApplicability,
    BundleOrchestrationError,
    ResetVerification,
    ServiceHealthReport,
    SystemBundleOrchestrator,
    UnsafeResumeError,
)
from experiments_suite_v2.schemas import (
    AnalyticalClassification,
    ArtifactProfile,
)
from experiments_suite_v2.storage import RunStore


def model_component_trial() -> SystemTrial:
    base = expand_system_trials(include_workflow_ladder=False)[0]
    trial_tuple = replace(
        base.trial_tuple,
        artifact_profile=ArtifactProfile.MODEL_COMPONENT,
    )
    case = replace(
        base.case_definition,
        artifact_profile=ArtifactProfile.MODEL_COMPONENT,
        required_artifacts=(),
    )
    return replace(base, trial_tuple=trial_tuple, case_definition=case)


def passing_output(trial: SystemTrial) -> dict:
    return {
        predicate.actual_path: predicate.expected
        for predicate in trial.oracle.predicates
    }


class HealthyGate:
    def __init__(self, order: list[str] | None = None) -> None:
        self.order = order
        self.calls = 0

    def check(self, service_ids, trial) -> ServiceHealthReport:
        self.calls += 1
        if self.order is not None:
            self.order.append("health")
        return ServiceHealthReport(
            healthy=True,
            requested_service_ids=tuple(service_ids),
            observations=tuple(
                {"service_id": service_id, "status": "PASS"}
                for service_id in service_ids
            ),
        )


class VerifiedReset:
    def __init__(self, *, valid: bool = True, order: list[str] | None = None):
        self.valid = valid
        self.order = order
        self.calls = 0

    def reset_and_verify(self, trial: SystemTrial) -> ResetVerification:
        self.calls += 1
        if self.order is not None:
            self.order.append("reset")
        return ResetVerification(
            passed=self.valid,
            scene_state_id=trial.trial_tuple.scene_state_id,
            registered_state_sha256=(
                trial.trial_tuple.reset_manifest_hash
                if self.valid
                else "wrong-registered-state-hash"
            ),
            reset_manifest_sha256="actual-hidden-reset-manifest-hash",
            authoritative_snapshot={
                "scene_state_id": trial.trial_tuple.scene_state_id,
                "six_cube_poses": "scoring-only-fixture",
            },
            reason_code="RESET_VERIFIED" if self.valid else "RESET_MISMATCH",
            explanation="deterministic test reset verification",
        )


class RecordingPhysicalHooks:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    def begin(self, attempt_dir, trial, traces) -> None:
        self.order.append("physical_begin")

    def after_verified_reset(self, attempt_dir, trial, verification, traces) -> None:
        self.order.append("physical_after_reset")

    def finish(self, attempt_dir, trial, result, traces) -> None:
        self.order.append("physical_finish")


class SystemBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.suite_root = Path(self.temporary.name)
        self.store = RunStore(self.suite_root)
        # The repository protocol is intentionally being revised in parallel;
        # bundle tests isolate storage orchestration from that separate freeze
        # audit while retaining the exact current manifest payloads.
        protocol_state = type(
            "TestProtocolState",
            (),
            {
                "registry": load_json(REGISTRY_PATH),
                "protocol_manifest": load_json(PROTOCOL_MANIFEST_PATH),
                "camera_approval": load_json(CAMERA_APPROVAL_PATH),
                "confirmatory_authorized": False,
            },
        )()
        with patch(
            "experiments_suite_v2.storage.load_protocol_state",
            return_value=protocol_state,
        ):
            self.store.create_run(
                "system-run",
                source_commit="test-commit",
                dirty_state={"dirty": False},
            )
        self.applicability = AttemptApplicability(
            learned_model_live=False,
            memory_live=False,
            physical_actions=False,
        )
        self.trial = model_component_trial()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def orchestrator(self, *, reset=None, health=None, hooks=None):
        return SystemBundleOrchestrator(
            self.store,
            health_gate=health or HealthyGate(),
            reset_verifier=reset or VerifiedReset(),
            full_physical_hooks=hooks,
            environment_provider=lambda: {
                "schema_version": "test.environment.v2",
                "python": "test",
            },
        )

    def test_durable_attempt_finalizes_before_run_checksums(self) -> None:
        order: list[str] = []
        health = HealthyGate(order)
        reset = VerifiedReset(order=order)
        orchestrator = self.orchestrator(reset=reset, health=health)

        def execute(trial, traces):
            order.append("execute")
            traces.event("TEST_EXECUTOR_CALLED", {"fixture": True})
            return passing_output(trial)

        outcome = orchestrator.run_new(
            "system-run", self.trial, self.applicability, execute
        )
        self.assertEqual(order, ["health", "reset", "execute"])
        self.assertEqual(
            outcome.result.analytical_classification,
            AnalyticalClassification.PASS,
        )
        self.assertEqual(verify_artifact_manifest(outcome.attempt_dir), [])
        manifest = load_json(outcome.attempt_dir / "attempt_manifest.json")
        self.assertEqual(manifest["status"], "FINALIZED")
        self.assertEqual(manifest["identity"]["attempt_number"], 0)
        self.assertTrue((outcome.attempt_dir / "artifact_manifest.json").is_file())

        events = list(iter_jsonl(outcome.attempt_dir / "events.jsonl"))
        event_names = [item["event_type"] for item in events]
        self.assertEqual(event_names[0], "ATTEMPT_ALLOCATED")
        self.assertLess(
            event_names.index("ATTEMPT_ALLOCATED"),
            event_names.index("SERVICE_HEALTH_GATE"),
        )
        self.assertLess(
            event_names.index("REGISTERED_RESET_GATE"),
            event_names.index("SYSTEM_EXECUTION_STARTED"),
        )

        checksum_path = orchestrator.close_run("system-run")
        self.assertEqual(checksum_path.name, "checksums.sha256")
        self.assertEqual(verify_run_checksums(checksum_path.parent), [])

    def test_invalid_run_allocates_exact_a1_and_resolves_ledger(self) -> None:
        orchestrator = self.orchestrator()

        def disconnected(trial, traces):
            raise ExternalServiceInterruption(
                "forward disconnected",
                service_id="gemma-hri-planner-monitor-validator-v1",
                reason_code="VLM_CONNECTION_LOST",
            )

        invalid = orchestrator.run_new(
            "system-run", self.trial, self.applicability, disconnected
        )
        self.assertEqual(
            invalid.result.analytical_classification,
            AnalyticalClassification.INVALID_RUN,
        )
        retry = orchestrator.retry_invalid(
            invalid.attempt_dir,
            self.trial,
            self.applicability,
            lambda trial, traces: passing_output(trial),
        )
        self.assertEqual(
            retry.result.analytical_classification,
            AnalyticalClassification.PASS,
        )
        a0 = load_json(invalid.attempt_dir / "attempt_manifest.json")
        a1 = load_json(retry.attempt_dir / "attempt_manifest.json")
        self.assertEqual(a1["identity"]["attempt_number"], 1)
        self.assertEqual(
            a1["identity"]["retry_of"], a0["identity"]["attempt_id"]
        )
        self.assertEqual(
            a1["identity"]["supersedes_attempt"],
            a0["identity"]["attempt_id"],
        )
        self.assertEqual(a1["trial_tuple"], a0["trial_tuple"])
        ledger = load_json(
            self.suite_root
            / "results/shared-campaigns/system-run/invalid_retry_ledger.json"
        )
        self.assertEqual(
            ledger["entries"][0]["resolution_status"],
            "RESOLVED_BY_FIRST_VALID_RETRY",
        )
        self.assertEqual(
            ledger["entries"][0]["first_valid_attempt_id"],
            a1["identity"]["attempt_id"],
        )

    def test_reset_mismatch_is_invalid_setup_without_executor_call(self) -> None:
        orchestrator = self.orchestrator(reset=VerifiedReset(valid=False))
        calls = 0

        def must_not_execute(trial, traces):
            nonlocal calls
            calls += 1
            return passing_output(trial)

        outcome = orchestrator.run_new(
            "system-run", self.trial, self.applicability, must_not_execute
        )
        self.assertEqual(calls, 0)
        self.assertEqual(
            outcome.result.analytical_classification,
            AnalyticalClassification.INVALID_SETUP,
        )
        self.assertEqual(
            outcome.result.invalidity.reason_code, "RESET_MANIFEST_MISMATCH"
        )

    def test_resume_fences_unknown_post_execution_state(self) -> None:
        orchestrator = self.orchestrator()

        def crash_after_delivery(trial, traces):
            raise RuntimeError("simulated process death after execution started")

        attempt = orchestrator.allocate(
            "system-run", self.trial, self.applicability
        )
        with self.assertRaisesRegex(RuntimeError, "simulated process death"):
            orchestrator.resume(
                attempt, self.trial, self.applicability, crash_after_delivery
            )
        with self.assertRaises(UnsafeResumeError):
            orchestrator.resume(
                attempt,
                self.trial,
                self.applicability,
                lambda trial, traces: passing_output(trial),
            )
        candidates = orchestrator.recovery_candidates("system-run")
        self.assertEqual(candidates[0]["action"], "RESUME_ATTEMPT")

    def test_resume_recovers_interrupted_artifact_finalization(self) -> None:
        orchestrator = self.orchestrator()
        attempt = orchestrator.allocate(
            "system-run", self.trial, self.applicability
        )
        with patch(
            "experiments_suite_v2.storage.build_artifact_manifest",
            side_effect=RuntimeError("manifest crash"),
        ):
            with self.assertRaisesRegex(RuntimeError, "manifest crash"):
                orchestrator.resume(
                    attempt,
                    self.trial,
                    self.applicability,
                    lambda trial, traces: passing_output(trial),
                )
        self.assertEqual(
            load_json(attempt / "attempt_manifest.json")["status"], "FINALIZED"
        )
        recovered = orchestrator.resume(
            attempt,
            self.trial,
            self.applicability,
            lambda trial, traces: self.fail("executor must not run"),
        )
        self.assertTrue(recovered.recovered_finalization)
        self.assertEqual(verify_artifact_manifest(attempt), [])

    def test_full_physical_hooks_and_all_incremental_trace_sinks(self) -> None:
        order: list[str] = []
        hooks = RecordingPhysicalHooks(order)
        health = HealthyGate(order)
        reset = VerifiedReset(order=order)
        orchestrator = self.orchestrator(
            reset=reset, health=health, hooks=hooks
        )
        full = expand_system_trials(include_workflow_ladder=False)[0]
        services = (
            "gemma-hri-planner-monitor-validator-v1",
            "embeddinggemma-retriever-v1",
            "sam3.1-grounding-v1",
        )
        applicability = AttemptApplicability(
            learned_model_live=True,
            memory_live=True,
            physical_actions=True,
            service_ids=services,
        )

        def execute(trial, traces):
            order.append("execute")
            traces.model({"event": "MODEL_CALL", "raw_response": "fixture"})
            traces.memory({"event": "MEMORY_RETRIEVAL", "candidates": []})
            traces.execution({"event": "CONTROLLER_STAGE", "stage": "fixture"})
            traces.runtime({"event": "RUNTIME_TRANSITION", "state": "fixture"})
            return passing_output(trial)

        # This test targets hook/trace orchestration. The real shared storage
        # finalizer's complete FULL_PHYSICAL media audit is exercised only by
        # an actual capture hook and is deliberately not faked here.
        with patch.object(
            self.store,
            "finalize_attempt",
            return_value={"schema_version": "test-artifact-manifest"},
        ):
            outcome = orchestrator.run_new(
                "system-run", full, applicability, execute
            )
        self.assertEqual(
            order,
            [
                "physical_begin",
                "health",
                "reset",
                "physical_after_reset",
                "execute",
                "physical_finish",
            ],
        )
        for name in (
            "events.jsonl",
            "model_calls.jsonl",
            "memory_trace.jsonl",
            "execution_trace.jsonl",
            "runtime_states.jsonl",
        ):
            self.assertTrue((outcome.attempt_dir / name).is_file(), name)
            self.assertGreater(len(list(iter_jsonl(outcome.attempt_dir / name))), 0)

    def test_resume_rejects_changed_applicability(self) -> None:
        orchestrator = self.orchestrator()
        attempt = orchestrator.allocate(
            "system-run", self.trial, self.applicability
        )
        changed = AttemptApplicability(
            learned_model_live=True,
            memory_live=False,
            physical_actions=False,
        )
        with self.assertRaisesRegex(
            BundleOrchestrationError, "applicability differs"
        ):
            orchestrator.resume(
                attempt,
                self.trial,
                changed,
                lambda trial, traces: passing_output(trial),
            )


if __name__ == "__main__":
    unittest.main()
