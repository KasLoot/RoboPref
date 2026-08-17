from __future__ import annotations

import unittest

from experiments_suite_v2.runners.cal_x04_x05_development import (
    StrictAssistedPhysicalCampaign,
    validate_execution_authorization,
)
from experiments_suite_v2.runners.physical import (
    PlacementEvaluation,
    build_cal_x04_rows,
    build_cal_x05_rows,
)
from experiments_suite_v2.schemas import AnalyticalClassification, AttemptIdentity


class CalX04X05DevelopmentTests(unittest.TestCase):
    def test_authorization_and_exact_grids_validate_without_execution(self):
        record = validate_execution_authorization()
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(len(build_cal_x04_rows()), 300)
        self.assertEqual(len(build_cal_x05_rows()), 300)

    def _result(self, *, strict: str, assisted: bool):
        row = build_cal_x05_rows()[0]
        evaluation = PlacementEvaluation(
            row=row,
            execution_path="PRODUCTION_FULL_PATH",
            trace=(),
            telemetry={
                "execution_outcome": {
                    "strict_system_result": strict,
                    "oracle_fallback_used": assisted,
                    "oracle_fallback_source": (
                        "SIMULATOR_GROUND_TRUTH" if assisted else None
                    ),
                    "assisted_continuation_result": (
                        "PASS" if assisted else "NOT_APPLICABLE"
                    ),
                    "downstream_execution_result": "PASS",
                    "grounding_attempts": [],
                }
            },
            scoring={
                "final_placement_error_mm": [0.0, 0.0, 0.0],
                "final_radial_xy_error_mm": 0.0,
            },
        )
        identity = AttemptIdentity(
            trial_id="CAL-X-05-test",
            trial_tuple_sha256="0" * 64,
            attempt_number=0,
        )
        campaign = object.__new__(StrictAssistedPhysicalCampaign)
        return campaign._result(
            identity,
            AnalyticalClassification.PASS,
            evaluation=evaluation,
        )

    def test_strict_success_remains_pass(self):
        result = self._result(strict="PASS", assisted=False)
        self.assertEqual(result.analytical_classification.value, "PASS")
        self.assertTrue(result.must_pass["strict_sam_grounding"])

    def test_assisted_success_remains_strict_capability_failure(self):
        result = self._result(strict="FAIL_GROUNDING", assisted=True)
        self.assertEqual(
            result.analytical_classification.value,
            "CAPABILITY_FAIL",
        )
        self.assertTrue(result.valid_trial)
        self.assertTrue(result.must_pass["valid_terminal_measurement"])
        self.assertFalse(result.must_pass["strict_sam_grounding"])
        self.assertEqual(result.failed_predicates, ("strict_sam_grounding",))


if __name__ == "__main__":
    unittest.main()

