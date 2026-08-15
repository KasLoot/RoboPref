from __future__ import annotations

from collections import Counter
import unittest

from experiments_suite_v2.runners import ab_replan_development as runner


class ABReplanDevelopmentTests(unittest.TestCase):
    def test_exact_registered_rows_and_adjacent_pair_balance(self) -> None:
        groups = runner.build_rows()
        self.assertEqual(tuple(groups), runner.AIM_ORDER)
        self.assertEqual(sum(map(len, groups.values())), 160)
        for aim, rows in groups.items():
            self.assertEqual(len(rows), 80)
            self.assertEqual(len({row.trial_id for row in rows}), 80)
            self.assertEqual(len({row.pair_id for row in rows}), 40)
            self.assertEqual(
                Counter(row.condition_id for row in rows),
                {runner.CONDITIONS[aim][0]: 40, runner.CONDITIONS[aim][1]: 40},
            )
            for index in range(0, 80, 2):
                pair = rows[index : index + 2]
                self.assertEqual(len({row.pair_id for row in pair}), 1)
                self.assertEqual(
                    {row.condition_id for row in pair},
                    set(runner.CONDITIONS[aim]),
                )

    def test_stable_failure_fresh_replan_recovers_and_frozen_queue_does_not(self) -> None:
        rows = runner.build_rows()["AB-RPL-F"]
        results = []
        for row in rows:
            actual, trace = runner.simulate_row(row)
            results.append(runner._result_row(row, actual, trace))
        summary = runner._summary("AB-RPL-F", results)
        self.assertEqual(summary["conditions"]["FRESH_REPLAN"]["pass_count"], 40)
        self.assertEqual(summary["conditions"]["FROZEN_QUEUE"]["pass_count"], 0)
        self.assertEqual(summary["pair_metrics"]["treatment_only"], 40)
        self.assertEqual(summary["formal_verdict"], "PASS")

    def test_timeout_policies_recover_but_auto_needs_no_intervention(self) -> None:
        rows = runner.build_rows()["AB-RPL-T"]
        results = []
        for row in rows:
            actual, trace = runner.simulate_row(row)
            results.append(runner._result_row(row, actual, trace))
        summary = runner._summary("AB-RPL-T", results)
        self.assertEqual(summary["conditions"]["AUTO_REPLAN"]["pass_count"], 40)
        self.assertEqual(summary["conditions"]["ATTENTION_GATE"]["pass_count"], 40)
        self.assertEqual(summary["conditions"]["AUTO_REPLAN"]["operator_interventions"], 0)
        self.assertEqual(summary["conditions"]["ATTENTION_GATE"]["operator_interventions"], 45)
        self.assertEqual(summary["formal_verdict"], "PASS")

    def test_two_timeout_cycle_and_stale_publication_guards(self) -> None:
        rows = runner.build_rows()["AB-RPL-T"]
        t07 = [row for row in rows if row.context_id == "T07"]
        t08 = [row for row in rows if row.context_id == "T08"]
        self.assertEqual(len(t07), 10)
        self.assertEqual(len(t08), 10)
        for row in t07:
            actual, _ = runner.simulate_row(row)
            self.assertEqual(actual["timeout_cycles"], 2)
            self.assertFalse(actual["guard_breach"])
            self.assertTrue(actual["goal_recovered_within_cycle_guard"])
        for row in t08:
            actual, _ = runner.simulate_row(row)
            self.assertFalse(actual["action_after_stale_evidence"])

    def test_pair_metrics_reject_incomplete_pairs(self) -> None:
        metrics = runner._pair_metrics(
            "AB-RPL-F",
            [
                {
                    "pair_id": "pair-0",
                    "condition_id": "FRESH_REPLAN",
                    "valid_trial": True,
                    "formal_verdict": "PASS",
                }
            ],
        )
        self.assertFalse(metrics["complete"])

    def test_bundle_input_adapter_adds_exact_row_id(self) -> None:
        for rows in runner.build_rows().values():
            adapted = [dict(row.to_dict(), row_id=row.trial_id) for row in rows]
            self.assertEqual(len({row["row_id"] for row in adapted}), 80)
            self.assertTrue(
                all(value["row_id"] == source.trial_id for source, value in zip(rows, adapted))
            )


if __name__ == "__main__":
    unittest.main()
