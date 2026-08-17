from __future__ import annotations

from collections import Counter
import unittest

from experiments_suite_v2.cases.memory_grid import all_memory_grid_needs
from experiments_suite_v2.schemas import canonical_sha256


class MemoryGridCaseTests(unittest.TestCase):
    def test_frozen_composition_and_balanced_split(self) -> None:
        needs = all_memory_grid_needs()
        self.assertEqual(len(needs), 80)
        self.assertEqual(
            Counter(need.need_type for need in needs),
            {
                "ONE_RELEVANT": 40,
                "TWO_COMPLEMENTARY": 20,
                "SUPERSEDED_OR_CONFLICTING": 10,
                "NO_RELEVANT": 10,
            },
        )
        by_type_split = Counter((need.need_type, need.split) for need in needs)
        self.assertEqual(by_type_split[("ONE_RELEVANT", "DEVELOPMENT")], 20)
        self.assertEqual(by_type_split[("ONE_RELEVANT", "CONFIRMATORY")], 20)
        self.assertEqual(by_type_split[("TWO_COMPLEMENTARY", "DEVELOPMENT")], 10)
        self.assertEqual(by_type_split[("TWO_COMPLEMENTARY", "CONFIRMATORY")], 10)
        self.assertEqual(by_type_split[("SUPERSEDED_OR_CONFLICTING", "DEVELOPMENT")], 5)
        self.assertEqual(by_type_split[("SUPERSEDED_OR_CONFLICTING", "CONFIRMATORY")], 5)
        self.assertEqual(by_type_split[("NO_RELEVANT", "DEVELOPMENT")], 5)
        self.assertEqual(by_type_split[("NO_RELEVANT", "CONFIRMATORY")], 5)

    def test_queries_and_stores_are_strict_prefix_fixtures(self) -> None:
        for need in all_memory_grid_needs():
            self.assertEqual(need.query_prefix(1), need.query_prefix(3)[:1])
            self.assertEqual(need.query_prefix(3), need.query_prefix(5)[:3])
            self.assertEqual(need.query_prefix(5), need.query_prefix(10)[:5])
            self.assertEqual(need.store(5), need.store(25)[:5])
            self.assertEqual(need.store(25), need.store(100)[:25])

    def test_relevance_contracts(self) -> None:
        for need in all_memory_grid_needs():
            if need.need_type == "ONE_RELEVANT":
                self.assertEqual(len(need.required_memory_ids), 1)
            elif need.need_type == "TWO_COMPLEMENTARY":
                self.assertEqual(len(need.required_memory_ids), 2)
            elif need.need_type == "SUPERSEDED_OR_CONFLICTING":
                self.assertEqual(len(need.required_memory_ids), 1)
                self.assertEqual(len(need.superseded_or_harmful_ids), 1)
            else:
                self.assertEqual(need.required_memory_ids, ())
            first_five_ids = {item.memory_id for item in need.store(5)}
            self.assertTrue(set(need.required_memory_ids) <= first_five_ids)
            self.assertTrue(set(need.superseded_or_harmful_ids) <= first_five_ids)

    def test_fixture_is_deterministic(self) -> None:
        first = [need.to_dict() for need in all_memory_grid_needs()]
        second = [need.to_dict() for need in all_memory_grid_needs()]
        self.assertEqual(canonical_sha256(first), canonical_sha256(second))


if __name__ == "__main__":
    unittest.main()
