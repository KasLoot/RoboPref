from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import unittest

from experiments_suite_v2.cli import main


class V2CliTests(unittest.TestCase):
    def test_plan_reports_full_minimum_without_execution(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            return_code = main(["plan", "--json"])
        self.assertEqual(return_code, 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["explicit_primary_valid_trials_minimum"], 22_600)
        self.assertFalse(value["confirmatory_execution_authorized"])

    def test_preflight_can_require_confirmatory_authorization(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            return_code = main(["preflight", "--json", "--require-confirmatory"])
        self.assertEqual(return_code, 2)
        value = json.loads(output.getvalue())
        self.assertEqual(value["metadata_validation"], "PASS")
        self.assertEqual(value["camera_approval_status"], "APPROVED")
        self.assertFalse(value["confirmatory_execution_authorized"])

    def test_cal_x04_plan_is_read_only_exact_and_fail_closed(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            return_code = main(["cal-x04-plan", "--json"])
        self.assertEqual(return_code, 0)
        value = json.loads(output.getvalue())
        self.assertTrue(value["dry_run"])
        self.assertFalse(value["run_or_attempt_allocated"])
        self.assertFalse(value["simulator_started"])
        self.assertEqual(value["motion_commands"], 0)
        self.assertEqual(value["external_calls"], 0)
        self.assertEqual(value["grid"]["planned_valid_terminal_rows"], 300)
        self.assertEqual(value["blockers"], ["CAL-X-03_NOT_PASSING"])

    def test_cal_x05_plan_is_read_only_and_reports_frozen_supplement(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            return_code = main(["cal-x05-plan", "--json"])
        self.assertEqual(return_code, 0)
        value = json.loads(output.getvalue())
        self.assertTrue(value["dry_run"])
        self.assertFalse(value["run_or_attempt_allocated"])
        self.assertFalse(value["simulator_started"])
        self.assertEqual(value["motion_commands"], 0)
        self.assertEqual(value["external_calls"], 0)
        self.assertEqual(value["grid"]["planned_valid_terminal_rows"], 300)
        self.assertEqual(
            value["core_blockers"],
            [
                "CAL_X05_READINESS_PROTOCOL_PIN_MISSING",
                "REPAIRED_CANONICAL_CAL_X03_NOT_DESIGNATED",
                "CAL_X04_QUALIFICATION_NOT_DESIGNATED",
            ],
        )
        self.assertEqual(
            value["scope"]["supplement_status"],
            "DEFINED_FROZEN_NOT_ALLOCATED",
        )
        self.assertEqual(value["scope"]["supplement_rows"], 70)
        self.assertEqual(value["scope"]["supplement_physical_actions"], 90)


if __name__ == "__main__":
    unittest.main()
