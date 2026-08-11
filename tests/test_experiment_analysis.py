from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.harness.analysis import (
    bootstrap_power,
    failure_taxonomy,
    holm_adjust,
    holm_correction,
    intention_to_treat,
    load_campaign_rows,
    outcome_value,
    paired_binary_comparison,
    select_powered_sample_size,
    wilson_interval,
)
from experiments.harness.campaign import CampaignRegistry, ScheduleCell, build_blocked_schedule


class AnalysisTests(unittest.TestCase):
    def test_wilson_interval_known_value_and_empty_is_uninformative(self) -> None:
        lower, upper = wilson_interval(5, 10)
        self.assertAlmostEqual(lower, 0.2366, places=4)
        self.assertAlmostEqual(upper, 0.7634, places=4)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_itt_uses_all_assignments_and_refuses_unresolved_point_estimate(self) -> None:
        rows = [
            {"schedule_id": "S1", "run_status": "VALID_PASS"},
            {"schedule_id": "S2", "run_status": "VALID_SYSTEM_FAILURE"},
            {"schedule_id": "S3", "run_status": "INFRA_INTERRUPTED"},
            {"schedule_id": "S4", "run_status": "NOT_RUN"},
        ]
        result = intention_to_treat(rows)
        self.assertFalse(result["estimable"])
        self.assertIsNone(result["estimate"])
        self.assertEqual(result["assigned"], 4)
        self.assertEqual(result["resolved"], 2)
        self.assertEqual(result["missing"], 2)
        self.assertEqual(result["success_rate_bounds"], [0.25, 0.75])

    def test_latest_attempt_is_canonical_but_interrupted_attempt_is_preserved(self) -> None:
        rows = [
            {
                "schedule_id": "S1",
                "attempt_number": 1,
                "run_status": "INFRA_INTERRUPTED",
            },
            {
                "schedule_id": "S1",
                "attempt_number": 2,
                "run_status": "VALID_PASS",
            },
            {
                "schedule_id": "S2",
                "attempt_number": 1,
                "run_status": "VALID_SYSTEM_FAILURE",
            },
        ]
        result = intention_to_treat(rows)
        self.assertTrue(result["estimable"])
        self.assertEqual(result["assigned"], 2)
        self.assertEqual(result["estimate"], 0.5)

    def test_paired_comparison_uses_crn_blocks_and_reports_missing_partner(self) -> None:
        rows = [
            {"schedule_id": "A1", "crn_key": "b1", "profile_id": "T5", "run_status": "VALID_PASS"},
            {"schedule_id": "B1", "crn_key": "b1", "profile_id": "B1", "run_status": "VALID_SYSTEM_FAILURE"},
            {"schedule_id": "A2", "crn_key": "b2", "profile_id": "T5", "run_status": "VALID_PASS"},
            {"schedule_id": "B2", "crn_key": "b2", "profile_id": "B1", "run_status": "VALID_PASS"},
            {"schedule_id": "A3", "crn_key": "b3", "profile_id": "T5", "run_status": "VALID_SYSTEM_FAILURE"},
            {"schedule_id": "B3", "crn_key": "b3", "profile_id": "B1", "run_status": "VALID_SYSTEM_FAILURE"},
        ]
        result = paired_binary_comparison(
            rows, "T5", "B1", bootstrap_iterations=200, bootstrap_seed=4
        )
        self.assertTrue(result["estimable"])
        self.assertAlmostEqual(result["estimate"], 1 / 3)
        self.assertEqual(result["a_only_success"], 1)
        self.assertEqual(result["b_only_success"], 0)
        incomplete = paired_binary_comparison(
            rows[:-1], "T5", "B1", bootstrap_iterations=200
        )
        self.assertFalse(incomplete["estimable"])
        self.assertIn("incomplete", incomplete["reason"])

    def test_crn_label_cannot_pair_different_scenarios(self) -> None:
        rows = [
            {
                "schedule_id": "A",
                "block_id": "block-a",
                "crn_key": "reused",
                "scenario_id": "scenario-a",
                "profile_id": "T5",
                "run_status": "VALID_PASS",
            },
            {
                "schedule_id": "B",
                "block_id": "block-b",
                "crn_key": "reused",
                "scenario_id": "scenario-b",
                "profile_id": "B1",
                "run_status": "VALID_SYSTEM_FAILURE",
            },
        ]
        result = paired_binary_comparison(rows, "T5", "B1", bootstrap_iterations=100)
        self.assertFalse(result["estimable"])
        self.assertEqual(result["complete_pairs"], 0)

    def test_holm_is_step_down_monotone_in_original_order(self) -> None:
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        corrected = holm_correction({"h1": 0.01, "h2": 0.04, "h3": 0.03})
        self.assertTrue(corrected["h1"]["reject"])
        self.assertFalse(corrected["h2"]["reject"])

    def test_bootstrap_power_is_seeded_and_selects_target(self) -> None:
        first = bootstrap_power(
            [1.0, 1.0, 1.0],
            [5, 10],
            simulations=100,
            inner_bootstrap_iterations=100,
            seed=8,
        )
        second = bootstrap_power(
            [1.0, 1.0, 1.0],
            [5, 10],
            simulations=100,
            inner_bootstrap_iterations=100,
            seed=8,
        )
        self.assertEqual(first, second)
        self.assertEqual([item["power"] for item in first], [1.0, 1.0])
        self.assertEqual(select_powered_sample_size(first), 5)
        reversed_curve = bootstrap_power(
            [1.0, 1.0, 1.0],
            [10, 5],
            simulations=100,
            inner_bootstrap_iterations=100,
            seed=8,
        )
        self.assertEqual(
            {item["sample_size"]: item for item in first},
            {item["sample_size"]: item for item in reversed_curve},
        )
        with self.assertRaisesRegex(ValueError, "integers"):
            bootstrap_power([1.0, 0.0], [2.9], simulations=100)
        insufficient = bootstrap_power([1.0], [5], simulations=100)
        self.assertFalse(insufficient[0]["estimable"])

    def test_failure_taxonomy_keeps_status_and_diagnosis_separate(self) -> None:
        report = failure_taxonomy(
            [
                {"schedule_id": "S1", "run_status": "VALID_PASS"},
                {
                    "schedule_id": "S2",
                    "run_status": "VALID_SYSTEM_FAILURE",
                    "failure_layer": "planning",
                    "primary_diagnosis": "stale goal",
                },
                {
                    "schedule_id": "S3",
                    "run_status": "INFRA_INTERRUPTED",
                    "failure_layer": "model service/infrastructure",
                },
            ]
        )
        self.assertEqual(report["status_counts"]["VALID_SYSTEM_FAILURE"], 1)
        self.assertEqual(report["failure_layer_counts"]["planning"], 1)
        self.assertEqual(report["primary_diagnosis_counts"]["stale goal"], 1)
        self.assertEqual(report["unresolved_or_invalid"], 1)

    def test_result_artifact_cannot_override_randomized_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            schedule = build_blocked_schedule(
                [ScheduleCell("matrix", "scenario", "pilot-v", ("T5",), 1)],
                master_seed=4,
                split="pilot",
            )
            registry = CampaignRegistry.create(
                root,
                campaign_id="pilot-analysis",
                mode="pilot",
                protocol_sha256="d" * 64,
                schedule=schedule,
            )
            lease = registry.begin_attempt()
            lease.attempt_path.mkdir(parents=True)
            (lease.attempt_path / "results.json").write_text(
                json.dumps(
                    {
                        "run_status": "VALID_PASS",
                        "profile_id": "B1",
                        "seed": 999,
                        "contract_success": True,
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "experiments.harness.audit.audit_attempt_artifacts",
                return_value={"passed": True, "errors": []},
            ):
                registry.seal_attempt(
                    lease.schedule_id,
                    lease.attempt_number,
                    "VALID_PASS",
                    oracle_verdict="PASS",
                    artifact_audit_passed=True,
                )
            row = load_campaign_rows(root)[0]
            self.assertEqual(row["profile_id"], "T5")
            self.assertEqual(row["seed"], schedule[0].seed)
            self.assertEqual(set(row["result_identity_conflicts"]), {"profile_id", "seed"})
            self.assertIsNone(outcome_value(row, "contract_success"))


if __name__ == "__main__":
    unittest.main()
