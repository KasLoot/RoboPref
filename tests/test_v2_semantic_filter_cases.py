from __future__ import annotations

from collections import Counter
import unittest

from experiments_suite_v2.cases.semantic_filter import (
    FILTER_CONDITIONS,
    FILTER_SEEDS,
    POLLUTION_LOADS,
    semantic_filter_contexts,
    semantic_filter_fixture_payload,
)
from experiments_suite_v2.schemas import canonical_sha256


class SemanticFilterCaseTests(unittest.TestCase):
    def test_context_composition_resolves_documented_ambiguity(self) -> None:
        contexts = semantic_filter_contexts()
        self.assertEqual(len(contexts), 32)
        self.assertEqual(
            Counter(item.context_type for item in contexts),
            {
                "WRONG_USER": 8,
                "CURRENT_OVERRIDE": 8,
                "DIFFERENT_TASK": 8,
                "NO_RELEVANT": 8,
            },
        )
        no_relevant = [item for item in contexts if item.context_type == "NO_RELEVANT"]
        self.assertTrue(all(item.relevant_candidate is None for item in no_relevant))
        others = [item for item in contexts if item.context_type != "NO_RELEVANT"]
        self.assertTrue(all(item.relevant_candidate is not None for item in others))

    def test_all_loads_are_exact_and_provenance_labeled(self) -> None:
        for context in semantic_filter_contexts():
            for load in POLLUTION_LOADS:
                candidates = context.candidates(load)
                expected = load + (context.relevant_candidate is not None)
                self.assertEqual(len(candidates), expected)
                self.assertEqual(len({item.memory_id for item in candidates}), expected)
                self.assertEqual(
                    sum(item.relevant for item in candidates),
                    0 if context.relevant_candidate is None else 1,
                )

    def test_primary_and_stochastic_counts_are_preserved(self) -> None:
        self.assertEqual(32 * len(POLLUTION_LOADS) * len(FILTER_CONDITIONS), 480)
        self.assertEqual(10 * len(FILTER_SEEDS) * len(POLLUTION_LOADS) * len(FILTER_CONDITIONS), 450)

    def test_payload_is_deterministic(self) -> None:
        first = semantic_filter_fixture_payload()
        second = semantic_filter_fixture_payload()
        self.assertEqual(canonical_sha256(first), canonical_sha256(second))


if __name__ == "__main__":
    unittest.main()
