from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments_suite_v2.audit import audit_attempt, audit_run
from experiments_suite_v2.runners.replay import expand_replay_trials
from experiments_suite_v2.runners.replay_campaign import ReplayCampaign
from experiments_suite_v2.storage import RunStore


class ReplayCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "protocol").mkdir()
        shutil.copy2(
            "experiments_suite_v2/protocol/cases_replay.json",
            self.root / "protocol" / "cases_replay.json",
        )
        self.store = RunStore(self.root)
        self.store.create_run("replay", source_commit="abc", dirty_state={})
        self.canonical = expand_replay_trials(software_config_commit="abc")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_one_row_is_allocated_and_audited_without_model_artifact(self) -> None:
        campaign = ReplayCampaign(
            store=self.store,
            suite_run_id="replay",
            _test_trials=self.canonical[:1],
        )
        outcome = campaign.run_next()
        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertEqual(outcome.classification, "PASS")
        self.assertFalse((outcome.attempt_dir / "model_calls.jsonl").exists())
        self.assertTrue((outcome.attempt_dir / "replay_trace.jsonl").is_file())
        self.assertTrue(audit_attempt(outcome.attempt_dir).passed)
        self.assertTrue(campaign.status()["complete"])

    def test_durable_evaluation_recovers_without_replaying_host_trace(self) -> None:
        campaign = ReplayCampaign(
            store=self.store,
            suite_run_id="replay",
            _test_trials=self.canonical[:1],
        )
        original = campaign._finalize_record
        with patch.object(
            campaign,
            "_finalize_record",
            side_effect=RuntimeError("crash after evaluation commit"),
        ):
            with self.assertRaisesRegex(RuntimeError, "evaluation commit"):
                campaign.run_next()
        attempt = next((campaign.run_root / "PF-REPLAN").iterdir())
        self.assertTrue((attempt / "replay_evaluation.json").is_file())
        resumed = ReplayCampaign(
            store=self.store,
            suite_run_id="replay",
            _test_trials=self.canonical[:1],
        )
        with patch(
            "experiments_suite_v2.runners.replay_campaign.run_replay_trial",
            side_effect=AssertionError("host trace was replayed"),
        ):
            recovered = resumed.run_next()
        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertTrue(recovered.recovered)
        self.assertTrue(audit_attempt(recovered.attempt_dir).passed)
        self.assertTrue(callable(original))

    def test_full_twenty_row_campaign_finalizes_and_closes(self) -> None:
        campaign = ReplayCampaign(store=self.store, suite_run_id="replay")
        self.assertEqual(len(campaign.run_serial(maximum_rows=20)), 20)
        self.assertTrue(campaign.status()["complete"])
        campaign.close_suite_run()
        self.assertTrue(audit_run(campaign.run_root, require_closed=True).passed)


if __name__ == "__main__":
    unittest.main()
