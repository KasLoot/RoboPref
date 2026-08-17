from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments_suite_v2.audit import AuditReport
from experiments_suite_v2.io import (
    atomic_write_json,
    atomic_write_text,
    iter_jsonl,
    load_json,
    utc_now,
)
from experiments_suite_v2.registry import SUITE_ROOT, load_protocol_state
from experiments_suite_v2.runners.system import (
    CallableSystemAdapter,
    ExternalServiceInterruption,
    SystemBlock,
    SystemTrial,
    expand_system_trials,
    run_system_trial,
)
from experiments_suite_v2.runners.system_bundle import (
    AttemptApplicability,
    BundleOutcome,
    SystemBundleOrchestrator,
    UnsafeResumeError,
)
from experiments_suite_v2.runners.system_campaign import (
    EMBEDDING_SERVICE_ID,
    EXPECTED_ALL_ROWS,
    EXPECTED_PRIMARY_ROWS,
    EXPECTED_SUPPLEMENTARY_ROWS,
    SAM_SERVICE_ID,
    VLM_SERVICE_ID,
    SystemCampaign,
    SystemCampaignIncomplete,
    SystemCampaignRecoveryRequired,
    system_attempt_applicability,
    system_execution_schedule,
)
from experiments_suite_v2.schemas import (
    AnalyticalClassification,
    AttemptIdentity,
    AttemptStatus,
)
from experiments_suite_v2.storage import RunStore


def passing_output(trial: SystemTrial) -> dict:
    result: dict = {}
    for predicate in trial.oracle.predicates:
        target = result
        parts = predicate.actual_path.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = predicate.expected
    return result


class FakeTraceSinks:
    def event(self, *_args, **_kwargs):
        return "fake-event"

    def model(self, *_args, **_kwargs):
        return None

    def memory(self, *_args, **_kwargs):
        return None

    def execution(self, *_args, **_kwargs):
        return None

    def runtime(self, *_args, **_kwargs):
        return None


class FakeSystemBundleOrchestrator:
    """Small campaign test double; per-attempt behavior has separate tests."""

    def __init__(self, store: RunStore, order: list[str] | None = None) -> None:
        self.store = store
        self.order = [] if order is None else order
        self.unsafe_paths: set[Path] = set()

    def allocate(
        self,
        suite_run_id: str,
        trial: SystemTrial,
        applicability: AttemptApplicability,
    ) -> Path:
        path = self.store.allocate_attempt(
            suite_run_id,
            trial.trial_tuple,
            SystemBundleOrchestrator._storage_case(trial, applicability),
            learned_model_live=applicability.learned_model_live,
            memory_live=applicability.memory_live,
            physical_actions=applicability.physical_actions,
            visual_fixture_paths=applicability.visual_fixture_paths,
            service_ids=applicability.service_ids,
        )
        self.order.append("attempt_allocated")
        return path

    def resume(self, attempt_dir, trial, applicability, executor) -> BundleOutcome:
        attempt_dir = Path(attempt_dir)
        manifest = load_json(attempt_dir / "attempt_manifest.json")
        identity = AttemptIdentity.from_dict(manifest["identity"])
        if manifest["status"] == AttemptStatus.FINALIZED.value:
            return BundleOutcome(
                attempt_dir=attempt_dir,
                result=load_result(attempt_dir),
                artifact_manifest=load_json(attempt_dir / "artifact_manifest.json"),
            )
        if attempt_dir in self.unsafe_paths:
            raise UnsafeResumeError("fake post-execution crash")
        if manifest["status"] == AttemptStatus.ALLOCATED.value:
            self.store.start_attempt(attempt_dir)
        if not (attempt_dir / "terminal.log").is_file():
            atomic_write_text(
                attempt_dir / "terminal.log", "fake physical attempt\n", overwrite=False
            )
        self.order.append("executor_entered")
        try:
            run = run_system_trial(
                trial,
                CallableSystemAdapter(
                    lambda _trial: executor(trial, FakeTraceSinks())
                ),
                attempt_id=identity.attempt_id,
                retry_lineage={
                    "retry_of": identity.retry_of,
                    "supersedes_attempt": identity.supersedes_attempt,
                },
            )
        except Exception:
            self.unsafe_paths.add(attempt_dir)
            raise
        atomic_write_text(
            attempt_dir / "results.md",
            f"# Fake result\n\n{run.result.analytical_classification.value}\n",
            overwrite=False,
        )
        self.store.write_result(attempt_dir, run.result)
        manifest = load_json(attempt_dir / "attempt_manifest.json")
        manifest["status"] = AttemptStatus.FINALIZED.value
        manifest["finalized_at"] = utc_now()
        manifest["result_summary"] = {
            "formal_verdict": run.result.formal_verdict.value,
            "analytical_classification": (
                run.result.analytical_classification.value
            ),
            "valid_trial": run.result.valid_trial,
            "enters_capability_denominator": (
                run.result.enters_capability_denominator
            ),
        }
        atomic_write_json(attempt_dir / "attempt_manifest.json", manifest)
        artifact = {
            "schema_version": "fake.system-attempt-artifacts.v2",
            "attempt_id": identity.attempt_id,
        }
        atomic_write_json(
            attempt_dir / "artifact_manifest.json", artifact, overwrite=False
        )
        return BundleOutcome(
            attempt_dir=attempt_dir,
            result=run.result,
            artifact_manifest=artifact,
        )


def load_result(attempt_dir: Path):
    from experiments_suite_v2.schemas import ResultEnvelope

    return ResultEnvelope.from_dict(load_json(attempt_dir / "results.json"))


def passing_audit(path: Path) -> AuditReport:
    return AuditReport("ATTEMPT", str(path))


class SystemCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.suite_root = Path(self.temporary.name) / "suite"
        protocol = self.suite_root / "protocol"
        protocol.mkdir(parents=True)
        for name in ("system_cases.json", "physical_cases.json"):
            shutil.copy2(SUITE_ROOT / "protocol" / name, protocol / name)
        self.store = RunStore(self.suite_root)
        state = load_protocol_state()
        with patch(
            "experiments_suite_v2.storage.load_protocol_state",
            return_value=state,
        ):
            self.store.create_run(
                "system-test",
                source_commit="test-commit",
                dirty_state={"dirty": False},
                run_kind="DEVELOPMENT",
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def canonical(self, *, include_workflow: bool = True):
        return expand_system_trials(
            load_json(self.suite_root / "protocol" / "system_cases.json"),
            software_config_commit="test-commit",
            include_workflow_ladder=include_workflow,
        )

    def campaign(
        self,
        trials,
        driver,
        *,
        order: list[str] | None = None,
        include_workflow: bool = True,
    ) -> SystemCampaign:
        bundle = FakeSystemBundleOrchestrator(self.store, order)

        def runtime_factory(trial, attempt_dir, traces):
            self.assertTrue((attempt_dir / "attempt_manifest.json").is_file())
            self.assertTrue(
                (
                    self.store.run_path("system-test")
                    / "campaigns/SYSTEM/campaign_manifest.json"
                ).is_file()
            )
            if order is not None:
                order.append("runtime_factory")
            return {"trial_id": trial.trial_id}

        def configurator(runtime, trial):
            runtime["condition_id"] = trial.condition_id
            if order is not None:
                order.append("runtime_configurator")

        def runtime_driver(runtime, trial, traces):
            if order is not None:
                order.append("runtime_driver")
            return driver(runtime, trial, traces)

        return SystemCampaign(
            store=self.store,
            suite_run_id="system-test",
            bundle_orchestrator=bundle,
            runtime_factory=runtime_factory,
            runtime_configurator=configurator,
            runtime_driver=runtime_driver,
            include_workflow_ladder=include_workflow,
            attempt_audit=passing_audit,
            _test_trials=trials,
        )

    def test_applicability_matches_live_components_and_frozen_services(self) -> None:
        rows = self.canonical()
        e2e = next(item for item in rows if item.block_id is SystemBlock.E2E_N)
        self.assertEqual(
            system_attempt_applicability(e2e),
            AttemptApplicability(
                learned_model_live=True,
                memory_live=True,
                physical_actions=True,
                service_ids=(VLM_SERVICE_ID, EMBEDDING_SERVICE_ID, SAM_SERVICE_ID),
            ),
        )
        replan = next(
            item for item in rows if item.block_id is SystemBlock.AB_RPL_T
        )
        self.assertEqual(
            system_attempt_applicability(replan).service_ids,
            (VLM_SERVICE_ID, SAM_SERVICE_ID),
        )
        memory_control = next(
            item
            for item in rows
            if item.case_definition.aim_id == "AB-COMP-MEM"
            and item.condition_id == "CONTROL"
        )
        self.assertFalse(system_attempt_applicability(memory_control).memory_live)
        workflow_l0 = next(
            item
            for item in rows
            if item.condition_id == "L0-EXE-EXACT"
        )
        self.assertEqual(
            system_attempt_applicability(workflow_l0).service_ids,
            (SAM_SERVICE_ID,),
        )
        workflow_l3 = next(item for item in rows if item.condition_id == "L3-MEM")
        self.assertTrue(system_attempt_applicability(workflow_l3).memory_live)

    def test_execution_schedule_runs_ablations_before_e2e_and_workflow_last(self) -> None:
        expansion = self.canonical()
        schedule = system_execution_schedule(expansion)
        compressed = []
        for item in schedule:
            if not compressed or compressed[-1] != item.block_id.value:
                compressed.append(item.block_id.value)
        self.assertEqual(
            compressed,
            [
                "AB-COMP",
                "AB-RPL-F",
                "AB-RPL-T",
                "E2E-N",
                "E2E-A",
                "E2E-R",
                "WORKFLOW-LADDER",
            ],
        )
        first_positions: dict[str, Counter[str]] = {}
        index_by_pair: dict[str, list[int]] = {}
        for index, item in enumerate(schedule):
            if item.pair_id is not None:
                index_by_pair.setdefault(item.pair_id, []).append(index)
        for pair_id, positions in index_by_pair.items():
            self.assertEqual(positions[1], positions[0] + 1, pair_id)
            first = schedule[positions[0]]
            first_positions.setdefault(
                first.case_definition.aim_id, Counter()
            )[first.condition_id] += 1
        for aim_id, counts in first_positions.items():
            with self.subTest(aim_id=aim_id):
                self.assertEqual(sorted(counts.values()), [20, 20])
        self.assertTrue(
            all(item.analysis_role == "PRIMARY" for item in schedule[:680])
        )
        self.assertTrue(
            all(item.analysis_role != "PRIMARY" for item in schedule[680:])
        )

    def test_full_manifest_binds_all_primary_and_supplementary_rows_without_calls(self) -> None:
        bundle = FakeSystemBundleOrchestrator(self.store)
        campaign = SystemCampaign(
            store=self.store,
            suite_run_id="system-test",
            bundle_orchestrator=bundle,
            include_workflow_ladder=True,
            attempt_audit=passing_audit,
        )
        manifest = load_json(campaign.campaign_dir / "campaign_manifest.json")
        self.assertEqual(manifest["planned_rows"], EXPECTED_ALL_ROWS)
        self.assertEqual(manifest["primary_rows"], EXPECTED_PRIMARY_ROWS)
        self.assertEqual(
            manifest["supplementary_rows"], EXPECTED_SUPPLEMENTARY_ROWS
        )
        self.assertEqual(manifest["analysis_status"], "NONCONFIRMATORY")
        self.assertFalse(manifest["confirmatory_execution_authorized"])
        self.assertNotEqual(
            manifest["registry_expansion_trial_tuple_order_sha256"],
            manifest["execution_order_trial_tuple_sha256"],
        )
        self.assertEqual(
            manifest["execution_phase_order"][-1],
            "SUPPLEMENTARY:WORKFLOW-LADDER",
        )
        self.assertEqual(bundle.order, [])
        self.assertEqual(
            sum(1 for _ in iter_jsonl(campaign.campaign_dir / "input_rows.jsonl")),
            EXPECTED_ALL_ROWS,
        )

    def test_primary_only_manifest_has_680_rows_and_no_supplementary_phase(self) -> None:
        bundle = FakeSystemBundleOrchestrator(self.store)
        campaign = SystemCampaign(
            store=self.store,
            suite_run_id="system-test",
            bundle_orchestrator=bundle,
            include_workflow_ladder=False,
            attempt_audit=passing_audit,
        )
        manifest = load_json(campaign.campaign_dir / "campaign_manifest.json")
        self.assertEqual(manifest["planned_rows"], EXPECTED_PRIMARY_ROWS)
        self.assertEqual(manifest["primary_rows"], EXPECTED_PRIMARY_ROWS)
        self.assertEqual(manifest["supplementary_rows"], 0)
        self.assertNotIn(
            "SUPPLEMENTARY:WORKFLOW-LADDER", manifest["execution_phase_order"]
        )
        self.assertEqual(
            sum(1 for _ in iter_jsonl(campaign.campaign_dir / "input_rows.jsonl")),
            EXPECTED_PRIMARY_ROWS,
        )

    def test_manifest_first_lazy_runtime_and_serial_cursor(self) -> None:
        rows = self.canonical()[:2]
        order: list[str] = []
        campaign = self.campaign(
            rows,
            lambda runtime, trial, traces: passing_output(trial),
            order=order,
        )
        self.assertEqual(order, [])
        first = campaign.run_next()
        assert first is not None
        self.assertEqual(
            order,
            [
                "attempt_allocated",
                "executor_entered",
                "runtime_factory",
                "runtime_configurator",
                "runtime_driver",
            ],
        )
        self.assertTrue(first.valid)
        status = campaign.status()
        self.assertEqual(status["valid_terminal_rows"], 1)
        self.assertEqual(status["next_row_id"], f"row-{rows[1].order_index:04d}")
        second = campaign.run_next()
        assert second is not None
        self.assertTrue(second.valid)
        self.assertTrue(campaign.status()["complete"])
        self.assertIsNone(campaign.run_next())

    def test_invalid_attempt_gets_exact_linked_a1_then_first_valid_terminal(self) -> None:
        rows = self.canonical()[:1]
        calls = 0

        def driver(runtime, trial, traces):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ExternalServiceInterruption(
                    "forward disconnected",
                    service_id=VLM_SERVICE_ID,
                    reason_code="VLM_CONNECTION_LOST",
                )
            return passing_output(trial)

        campaign = self.campaign(rows, driver)
        invalid = campaign.run_next()
        assert invalid is not None
        self.assertEqual(
            invalid.analytical_classification,
            AnalyticalClassification.INVALID_RUN.value,
        )
        valid = campaign.run_next()
        assert valid is not None
        self.assertEqual(valid.attempt_number, 1)
        self.assertEqual(valid.retry_of, invalid.attempt_id)
        self.assertEqual(valid.supersedes_attempt, invalid.attempt_id)
        a0 = load_json(invalid.attempt_dir / "attempt_manifest.json")
        a1 = load_json(valid.attempt_dir / "attempt_manifest.json")
        self.assertEqual(a0["trial_tuple"], a1["trial_tuple"])
        self.assertTrue(campaign.status()["complete"])
        completion = list(
            iter_jsonl(campaign.campaign_dir / "completion_rows.jsonl")
        )
        self.assertEqual([item["attempt_number"] for item in completion], [0, 1])
        ledger = load_json(
            self.store.run_path("system-test") / "invalid_retry_ledger.json"
        )
        self.assertEqual(
            ledger["entries"][0]["resolution_status"],
            "RESOLVED_BY_FIRST_VALID_RETRY",
        )

    def test_post_execution_crash_refuses_automatic_resampling(self) -> None:
        rows = self.canonical()[:1]
        factory_calls = 0
        bundle = FakeSystemBundleOrchestrator(self.store)

        def runtime_factory(trial, attempt_dir, traces):
            nonlocal factory_calls
            factory_calls += 1
            return object()

        campaign = SystemCampaign(
            store=self.store,
            suite_run_id="system-test",
            bundle_orchestrator=bundle,
            runtime_factory=runtime_factory,
            runtime_configurator=lambda runtime, trial: None,
            runtime_driver=lambda runtime, trial, traces: (_ for _ in ()).throw(
                RuntimeError("simulated process loss")
            ),
            attempt_audit=passing_audit,
            _test_trials=rows,
        )
        with self.assertRaises(SystemCampaignIncomplete):
            campaign.run_next()
        self.assertEqual(factory_calls, 1)
        with self.assertRaises(SystemCampaignRecoveryRequired):
            campaign.run_next()
        self.assertEqual(factory_calls, 1)
        attempts = list(
            self.store.run_path("system-test").glob("*/*/attempt_manifest.json")
        )
        self.assertEqual(len(attempts), 1)

    def test_aggregate_separates_conditions_and_preserves_paired_outcomes(self) -> None:
        all_rows = self.canonical()
        start = next(
            index
            for index, item in enumerate(all_rows)
            if item.case_definition.aim_id == "AB-COMP-HRI"
            and item.condition_id == "CONTROL"
        )
        rows = all_rows[start : start + 2]

        def driver(runtime, trial, traces):
            output = passing_output(trial)
            if trial.condition_id == "TREATMENT":
                first = trial.oracle.predicates[0].actual_path
                output[first] = "deliberate-capability-failure"
            return output

        campaign = self.campaign(rows, driver)
        outcomes = campaign.run_serial(
            maximum_attempts=2, stop_after_invalid=False
        )
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(all(item.valid for item in outcomes))
        aggregate = campaign.aggregate()
        self.assertIsNone(aggregate["suite_wide_capability_success_rate"])
        aim = next(
            item for item in aggregate["aims"] if item["aim_id"] == "AB-COMP-HRI"
        )
        self.assertEqual((aim["PASS"], aim["CAPABILITY_FAIL"]), (1, 1))
        conditions = {
            item["condition_id"]: item
            for item in aggregate["conditions"]
            if item["aim_id"] == "AB-COMP-HRI"
        }
        self.assertEqual(conditions["CONTROL"]["PASS"], 1)
        self.assertEqual(conditions["TREATMENT"]["CAPABILITY_FAIL"], 1)
        paired = next(
            item
            for item in aggregate["paired_aims"]
            if item["aim_id"] == "AB-COMP-HRI"
        )
        self.assertEqual(paired["complete_valid_pairs"], 1)
        self.assertEqual(sum(paired["paired_outcome_matrix"].values()), 1)
        self.assertTrue(campaign.campaign_audit(require_complete=True)["passed"])

    def test_bounded_serial_run_continues_after_capability_failure(self) -> None:
        rows = self.canonical(include_workflow=False)[:2]
        calls = 0

        def driver(runtime, trial, traces):
            nonlocal calls
            calls += 1
            output = passing_output(trial)
            if calls == 1:
                first = trial.oracle.predicates[0].actual_path
                output[first] = "deliberate-capability-failure"
            return output

        campaign = self.campaign(rows, driver, include_workflow=False)
        outcomes = campaign.run_serial(maximum_attempts=2)

        self.assertEqual(len(outcomes), 2)
        self.assertEqual(calls, 2)
        self.assertEqual(
            [item.analytical_classification for item in outcomes],
            [
                AnalyticalClassification.CAPABILITY_FAIL.value,
                AnalyticalClassification.PASS.value,
            ],
        )


if __name__ == "__main__":
    unittest.main()
