from __future__ import annotations

import unittest

from experiments.harness.invariants import (
    INVARIANT_FAMILIES,
    require_invariants,
    run_invariant_suite,
)


class ExperimentInvariantTests(unittest.TestCase):
    def test_exactly_thirty_families_run_hundreds_of_traces(self) -> None:
        results = run_invariant_suite(traces_per_family=100, master_seed=17)

        self.assertEqual(len(INVARIANT_FAMILIES), 30)
        self.assertEqual({item.family for item in results}, set(INVARIANT_FAMILIES))
        self.assertTrue(all(item.trace_count == 100 for item in results))
        require_invariants(results)

    def test_trace_results_are_deterministic(self) -> None:
        first = run_invariant_suite(traces_per_family=100, master_seed=29)
        second = run_invariant_suite(traces_per_family=100, master_seed=29)

        self.assertEqual(
            [(item.family, item.passed, item.failures) for item in first],
            [(item.family, item.passed, item.failures) for item in second],
        )

    def test_trace_count_cannot_be_weakened_below_hundreds(self) -> None:
        with self.assertRaises(ValueError):
            run_invariant_suite(traces_per_family=99)


if __name__ == "__main__":
    unittest.main()
