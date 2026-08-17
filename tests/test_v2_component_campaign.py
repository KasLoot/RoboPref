from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments_suite_v2.artifacts import verify_run_checksums
from experiments_suite_v2.audit import audit_attempt, audit_run
from experiments_suite_v2.io import atomic_write_json, iter_jsonl, load_json
from experiments_suite_v2.runners.component import (
    CallableFixtureAdapter,
    EndpointInvocationError,
    InvocationCapture,
    expand_component_trials,
)
from experiments_suite_v2.runners.component_campaign import (
    EXPECTED_AIM_COUNT,
    EXPECTED_ROWS_PER_AIM,
    EXPECTED_TRIAL_COUNT,
    ComponentCampaign,
    ComponentCampaignIncomplete,
    ComponentCampaignRecoveryRequired,
)
from experiments_suite_v2.storage import RunStore


SOURCE_COMMIT = "component-campaign-test"


class _ScriptedAdapter:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.component_name = f"TEST_{mode.upper()}"

    def invoke(self, trial):
        if self.mode == "endpoint":
            call_id = f"endpoint-{trial.order_index}"
            raise EndpointInvocationError(
                "SSH-forwarded model endpoint disconnected",
                reason_code="MODEL_ENDPOINT_CONNECTION_LOST",
                service_id="gemma-hri-planner-monitor-validator-v1",
                trace=(
                    {
                        "schema_version": "prefmem.component-trace.v2",
                        "event_type": "COMPONENT_CALL_STARTED",
                        "call_id": call_id,
                        "trial_id": trial.trial_id,
                        "payload": {"invocation": dict(trial.invocation)},
                    },
                    {
                        "schema_version": "prefmem.component-trace.v2",
                        "event_type": "COMPONENT_CALL_FAILED",
                        "call_id": call_id,
                        "trial_id": trial.trial_id,
                        "payload": {
                            "error_type": "ConnectionError",
                            "error": "SSH-forwarded model endpoint disconnected",
                        },
                    },
                ),
            )
        expected = trial.oracle.expected_values["oracle_label"]
        output = (
            {"oracle_label": expected}
            if self.mode == "pass"
            else {"oracle_label": "wrong-valid-model-response"}
        )
        base = CallableFixtureAdapter(
            lambda _trial: output,
            component_name=self.component_name,
        ).invoke(trial)
        return InvocationCapture(
            output=base.output,
            trace=base.trace,
            model_calls=(
                {
                    "schema_version": "prefmem.test-model-call.v2",
                    "trial_id": trial.trial_id,
                    "request": dict(trial.invocation),
                    "response": output,
                },
            ),
        )


class _Factory:
    def __init__(self, *modes: str) -> None:
        self.modes = deque(modes)
        self.calls: list[str] = []

    def __call__(self, trial):
        self.calls.append(trial.trial_id)
        if not self.modes:
            raise AssertionError("adapter factory was called unexpectedly")
        return _ScriptedAdapter(self.modes.popleft())


class ComponentCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = expand_component_trials(
            software_config_commit=SOURCE_COMMIT
        )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.suite_root = Path(self.temporary.name)
        self.store = RunStore(self.suite_root)
        self.store.create_run(
            "component-run",
            source_commit=SOURCE_COMMIT,
            dirty_state={"dirty": False},
            runtime_options={"campaign": "component-interface"},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _campaign(self, factory, rows=3):
        return ComponentCampaign(
            store=self.store,
            suite_run_id="component-run",
            adapter_factory=factory,
            _test_trials=self.canonical[:rows],
        )

    def _campaign_window(self, factory, start, rows=1):
        return ComponentCampaign(
            store=self.store,
            suite_run_id="component-run",
            adapter_factory=factory,
            _test_trials=self.canonical[start : start + rows],
        )

    def test_full_campaign_freezes_exact_1600_registry_order_before_calls(self):
        factory = _Factory()
        campaign = ComponentCampaign(
            store=self.store,
            suite_run_id="component-run",
            adapter_factory=factory,
        )
        rows = list(iter_jsonl(campaign.campaign_dir / "input_rows.jsonl"))
        self.assertEqual(len(rows), EXPECTED_TRIAL_COUNT)
        self.assertEqual(
            [row["order_index"] for row in rows],
            list(range(EXPECTED_TRIAL_COUNT)),
        )
        aim_counts = Counter(
            row["component_trial"]["trial_tuple"]["aim_id"] for row in rows
        )
        self.assertEqual(len(aim_counts), EXPECTED_AIM_COUNT)
        self.assertEqual(set(aim_counts.values()), {EXPECTED_ROWS_PER_AIM})
        manifest = load_json(campaign.campaign_dir / "campaign_manifest.json")
        self.assertEqual(manifest["status"], "OPEN")
        self.assertEqual(manifest["scope"], "FULL_FROZEN_1600_ROWS")
        self.assertEqual(manifest["row_order"], [row["row_id"] for row in rows])
        self.assertEqual(factory.calls, [])
        self.assertFalse(any(campaign.run_root.glob("*/*/attempt_manifest.json")))

    def test_finalized_row_resumes_at_next_row_without_resampling(self):
        first_factory = _Factory("pass")
        campaign = self._campaign(first_factory)
        first = campaign.run_next()
        self.assertEqual(first.order_index, 0)
        self.assertTrue(audit_attempt(first.attempt_dir).passed)
        self.assertEqual(
            [event["event_type"] for event in iter_jsonl(first.attempt_dir / "events.jsonl")][0],
            "ATTEMPT_ALLOCATED",
        )
        component_trace = list(
            iter_jsonl(first.attempt_dir / "component_trace.jsonl")
        )
        self.assertEqual(
            component_trace[0]["event_type"],
            "CAMPAIGN_INVOCATION_DISPATCHED",
        )
        self.assertTrue((first.attempt_dir / "model_calls.jsonl").is_file())
        self.assertEqual(
            len(list(iter_jsonl(campaign.campaign_dir / "completion_rows.jsonl"))),
            1,
        )

        resumed_factory = _Factory("pass")
        resumed = self._campaign(resumed_factory)
        second = resumed.run_next()
        self.assertEqual(second.order_index, 1)
        self.assertEqual(second.attempt_number, 0)
        self.assertEqual(resumed_factory.calls, [self.canonical[1].trial_id])
        self.assertEqual(first_factory.calls, [self.canonical[0].trial_id])

    def test_endpoint_invalidity_gets_exact_a1_retry_and_first_valid_counts(self):
        factory = _Factory("endpoint", "pass", "pass")
        campaign = self._campaign(factory, rows=2)
        invalid = campaign.run_next()
        self.assertEqual(invalid.analytical_classification, "INVALID_RUN")
        self.assertEqual(invalid.attempt_number, 0)
        self.assertTrue(audit_attempt(invalid.attempt_dir).passed)
        fallback = list(iter_jsonl(invalid.attempt_dir / "model_calls.jsonl"))
        self.assertEqual(
            fallback[0]["capture_level"],
            "EXACT_COMPONENT_BOUNDARY_NO_LOWER_LEVEL_SINK_RECORD",
        )

        # Simulate a process disappearing after the immutable attempt manifest
        # was built but before RunStore refreshed its derived index/ledger.
        ledger_path = campaign.run_root / "invalid_retry_ledger.json"
        ledger = load_json(ledger_path)
        ledger["entries"] = []
        atomic_write_json(ledger_path, ledger)
        index_path = campaign.run_root / "index.json"
        index = load_json(index_path)
        index["attempts"][0]["status"] = "RUNNING"
        index["attempts"][0]["analytical_classification"] = None
        atomic_write_json(index_path, index)
        campaign = self._campaign(factory, rows=2)
        repaired_ledger = load_json(ledger_path)
        self.assertEqual(
            repaired_ledger["entries"][0]["invalid_attempt_id"],
            invalid.attempt_id,
        )
        repaired_index = load_json(index_path)
        self.assertEqual(repaired_index["attempts"][0]["status"], "FINALIZED")

        retry = campaign.run_next()
        self.assertEqual(retry.order_index, 0)
        self.assertEqual(retry.attempt_number, 1)
        self.assertEqual(retry.retry_of, invalid.attempt_id)
        self.assertEqual(retry.supersedes_attempt, invalid.attempt_id)
        self.assertTrue(audit_attempt(retry.attempt_dir).passed)
        invalid_manifest = load_json(invalid.attempt_dir / "attempt_manifest.json")
        retry_manifest = load_json(retry.attempt_dir / "attempt_manifest.json")
        self.assertEqual(
            invalid_manifest["case_definition"],
            self.canonical[0].case_definition.to_dict(),
        )
        self.assertEqual(
            invalid_manifest["trial_tuple"], retry_manifest["trial_tuple"]
        )
        self.assertEqual(
            invalid_manifest["case_definition"], retry_manifest["case_definition"]
        )

        next_row = campaign.run_next()
        self.assertEqual(next_row.order_index, 1)
        self.assertEqual(next_row.attempt_number, 0)
        aggregate = campaign.aggregate()
        self.assertEqual(aggregate["valid_trials"], 2)
        self.assertEqual(aggregate["INVALID_RUN"], 1)
        self.assertEqual(aggregate["retried_invalid_attempts"], 1)
        ledger = load_json(campaign.run_root / "invalid_retry_ledger.json")
        self.assertEqual(
            ledger["entries"][0]["resolution_status"],
            "RESOLVED_BY_FIRST_VALID_RETRY",
        )

    def test_capability_fail_is_terminal_and_never_retried(self):
        factory = _Factory("fail", "pass")
        campaign = self._campaign(factory, rows=2)
        failed = campaign.run_next()
        self.assertEqual(failed.analytical_classification, "CAPABILITY_FAIL")
        self.assertEqual(failed.attempt_number, 0)
        second = campaign.run_next()
        self.assertEqual(second.order_index, 1)
        self.assertEqual(second.attempt_number, 0)
        ledger = load_json(campaign.run_root / "invalid_retry_ledger.json")
        self.assertEqual(ledger["entries"], [])
        aggregate = campaign.aggregate()
        self.assertEqual(aggregate["PASS"], 1)
        self.assertEqual(aggregate["CAPABILITY_FAIL"], 1)
        aim = aggregate["aims"][0]
        self.assertEqual(aim["valid_trials"], 2)
        self.assertEqual(aim["predicate_counts"]["claim_accuracy"]["FAIL"], 1)

    def test_crash_after_result_recovers_without_adapter_reinvocation(self):
        factory = _Factory("pass")
        campaign = self._campaign(factory, rows=2)
        with patch.object(
            self.store,
            "finalize_attempt",
            side_effect=RuntimeError("simulated finalization crash"),
        ):
            with self.assertRaises(ComponentCampaignIncomplete) as caught:
                campaign.run_next()
        attempt_dir = caught.exception.attempt_dir
        self.assertTrue((attempt_dir / "component_evaluation.json").is_file())
        self.assertTrue((attempt_dir / "results.json").is_file())
        self.assertFalse((attempt_dir / "artifact_manifest.json").exists())

        no_call_factory = _Factory()
        resumed = self._campaign(no_call_factory, rows=2)
        with self.assertRaises(ComponentCampaignRecoveryRequired):
            resumed.run_next()
        recovered = resumed.recover_attempt(attempt_dir)
        self.assertEqual(recovered.order_index, 0)
        self.assertTrue(audit_attempt(attempt_dir).passed)
        self.assertEqual(no_call_factory.calls, [])

    def test_memory_and_visual_profiles_freeze_their_specific_evidence(self):
        memory_index = next(
            index
            for index, trial in enumerate(self.canonical)
            if trial.case_definition.aim_id == "C-MEM-QUERY"
        )
        memory = self._campaign_window(_Factory("pass"), memory_index)
        memory_outcome = memory.run_next()
        self.assertTrue(audit_attempt(memory_outcome.attempt_dir).passed)
        memory_trace = list(
            iter_jsonl(memory_outcome.attempt_dir / "memory_trace.jsonl")
        )
        self.assertEqual(memory_trace[0]["stream"], "MEMORY_TRACE")

        # Use a separate run because each campaign directory freezes one exact
        # canonical window at construction.
        self.store.create_run(
            "visual-run",
            source_commit=SOURCE_COMMIT,
            dirty_state={"dirty": False},
        )
        visual_index = next(
            index
            for index, trial in enumerate(self.canonical)
            if trial.case_definition.aim_id == "C-MON-CRIT"
        )
        visual = ComponentCampaign(
            store=self.store,
            suite_run_id="visual-run",
            adapter_factory=_Factory("pass"),
            _test_trials=self.canonical[visual_index : visual_index + 1],
        )
        visual_outcome = visual.run_next()
        self.assertTrue(audit_attempt(visual_outcome.attempt_dir).passed)
        self.assertTrue((visual_outcome.attempt_dir / "fixtures/frame.json").is_file())
        event_trace = list(
            iter_jsonl(visual_outcome.attempt_dir / "event_trace.jsonl")
        )
        self.assertEqual(
            event_trace[0]["event_type"], "CAMPAIGN_INVOCATION_DISPATCHED"
        )

    def test_campaign_bundle_and_run_close_produce_clean_audits_and_checksums(self):
        campaign = self._campaign(_Factory("pass", "pass"), rows=2)
        campaign.run_next()
        campaign.run_next()
        # Exercise the production close path on a complete canonical prefix;
        # production construction itself always enforces the full 1,600 rows.
        campaign.full_registry = True
        with patch(
            "experiments_suite_v2.runners.component_campaign.EXPECTED_TRIAL_COUNT",
            2,
        ):
            campaign.finalize_campaign()
            checksum_path = campaign.close_suite_run()
        self.assertTrue(checksum_path.is_file())
        self.assertEqual(verify_run_checksums(campaign.run_root), [])
        self.assertTrue(audit_run(campaign.run_root, require_closed=True).passed)
        self.assertEqual(
            load_json(campaign.campaign_dir / "campaign_manifest.json")["status"],
            "FINALIZED",
        )
        self.assertEqual(
            load_json(campaign.campaign_dir / "aggregate.json")["valid_trials"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
