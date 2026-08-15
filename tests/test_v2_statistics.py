from __future__ import annotations

import unittest

from experiments_suite_v2.statistics import (
    exact_mcnemar_power,
    exact_mcnemar_pvalue,
    percentile,
    wilson_interval,
)


class V2StatisticsTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_proportion(self) -> None:
        lower, upper = wilson_interval(32, 40)
        self.assertLess(lower, 0.8)
        self.assertGreater(upper, 0.8)
        self.assertAlmostEqual(lower, 0.652426, places=5)
        self.assertAlmostEqual(upper, 0.895000, places=5)

    def test_exact_mcnemar_pvalue_is_symmetric(self) -> None:
        self.assertEqual(exact_mcnemar_pvalue(3, 17), exact_mcnemar_pvalue(17, 3))
        self.assertAlmostEqual(exact_mcnemar_pvalue(3, 17), 0.0025768280029296875)
        self.assertEqual(exact_mcnemar_pvalue(0, 0), 1.0)

    def test_frozen_power_calculation(self) -> None:
        power = exact_mcnemar_power(
            40,
            treatment_only_probability=0.35,
            control_only_probability=0.05,
        )
        self.assertAlmostEqual(power, 0.8629984934433707, places=14)

    def test_percentile_matches_linear_interpolation(self) -> None:
        self.assertEqual(percentile([0.0, 1.0, 2.0], 50), 1.0)
        self.assertAlmostEqual(percentile([0.0, 10.0], 95), 9.5)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)
        with self.assertRaises(ValueError):
            exact_mcnemar_power(
                40,
                treatment_only_probability=0.8,
                control_only_probability=0.3,
            )
        with self.assertRaises(ValueError):
            percentile([], 95)


if __name__ == "__main__":
    unittest.main()
