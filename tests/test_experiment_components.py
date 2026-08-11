from __future__ import annotations

from collections import Counter
import unittest

from experiments.harness.components import (
    COMPONENT_TARGETS,
    SPLIT_COUNTS,
    build_component_catalogue,
    catalogue_payload,
    score_prediction,
)


class ExperimentComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.items = build_component_catalogue()

    def test_exact_layer_b_targets_and_splits(self) -> None:
        self.assertEqual(
            Counter(item.component for item in self.items),
            Counter(COMPONENT_TARGETS),
        )
        for component, expected in SPLIT_COUNTS.items():
            self.assertEqual(
                Counter(
                    item.split
                    for item in self.items
                    if item.component == component
                ),
                Counter(expected),
            )

    def test_semantic_templates_do_not_cross_splits(self) -> None:
        templates = {}
        for item in self.items:
            self.assertNotIn(item.template_id, templates)
            templates[item.template_id] = item.split

    def test_each_fixture_has_nonconstant_positive_and_negative_score(self) -> None:
        for component in COMPONENT_TARGETS:
            item = next(item for item in self.items if item.component == component)
            positive = score_prediction(item, dict(item.expected))
            negative_payload = dict(item.expected)
            key = next(iter(negative_payload))
            value = negative_payload[key]
            negative_payload[key] = not value if isinstance(value, bool) else "wrong"
            negative = score_prediction(item, negative_payload)
            self.assertTrue(positive["passed"])
            self.assertFalse(negative["passed"])

    def test_generator_payload_is_byte_deterministic(self) -> None:
        self.assertEqual(catalogue_payload(), catalogue_payload())


if __name__ == "__main__":
    unittest.main()
