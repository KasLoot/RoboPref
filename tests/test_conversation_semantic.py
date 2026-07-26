from __future__ import annotations

import unittest

from simulation.benchmark.semantic import (
    CanonicalPredicate,
    canonical_goal_predicates,
    canonical_text,
    infer_target_id,
    planner_predicate_diff,
    infer_subtask_target,
)


def _manifest(
    goals: list[str],
    *,
    family: str = "block_stack",
    target_id: str = "rgb_bottom_to_top",
    instruction: str = "Stack red, green, blue from bottom to top.",
    objects: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "scenario_id": "semantic-test",
        "scene": {
            "family": family,
            "initial_objects": objects
            if objects is not None
            else [
                {
                    "object_id": "obj-red",
                    "object_type": "cube",
                    "attributes": {"colour": "red"},
                },
                {
                    "object_id": "obj-green",
                    "object_type": "cube",
                    "attributes": {"colour": "green"},
                },
                {
                    "object_id": "obj-blue",
                    "object_type": "cube",
                    "attributes": {"colour": "blue"},
                },
            ],
        },
        "target": {
            "target_id": target_id,
            "instruction": instruction,
            "goal_predicates": goals,
        },
    }


class ConversationSemanticTests(unittest.TestCase):
    def test_canonical_text_and_target_inference_accept_contract_or_text(
        self,
    ) -> None:
        manifest = _manifest([])

        self.assertEqual(canonical_text("  ＢＬＵＥ-tablet! "), "blue tablet")
        self.assertEqual(
            infer_target_id(
                manifest,
                {
                    "confirmed_intent": "Build the reverse stack.",
                    "parameters": {
                        "color_positions": {
                            "bottom": "blue",
                            "middle": "green",
                            "top": "red",
                        }
                    },
                },
            ),
            "bgr_bottom_to_top",
        )
        self.assertEqual(
            infer_target_id(manifest, "Stack RGB bottom-to-top."),
            "rgb_bottom_to_top",
        )

        category_manifest = _manifest(
            [],
            family="category_sort",
            target_id="electronics_left",
            instruction=(
                "Place electronic devices on the left and printed items on the right."
            ),
        )
        self.assertEqual(
            infer_target_id(
                category_manifest,
                "Put books on the right and electronic devices on the left.",
            ),
            "electronics_left",
        )

    def test_executable_subtasks_are_grounded_for_all_families(self) -> None:
        block = _manifest([])
        block_good = infer_subtask_target(
            block,
            [
                {
                    "task_instruction": "Stack the green block on the red block.",
                    "target": "green block",
                    "destination": "red block",
                }
            ],
        )
        block_opposite = infer_subtask_target(
            block,
            [
                {
                    "task_instruction": "Stack the green block on the blue block.",
                    "target": "green block",
                    "destination": "blue block",
                }
            ],
        )
        self.assertEqual(block_good.target_id, "rgb_bottom_to_top")
        self.assertEqual(block_opposite.target_id, "bgr_bottom_to_top")

        objects = [
            {
                "object_id": "obj-book",
                "object_type": "book",
                "attributes": {
                    "role": "book",
                    "semantic_category": "printed",
                },
            },
            {
                "object_id": "obj-fork",
                "object_type": "fork",
                "attributes": {"role": "fork"},
            },
        ]
        category = _manifest(
            [],
            family="category_sort",
            target_id="printed_left",
            objects=objects,
        )
        category_good = infer_subtask_target(
            category,
            [
                {
                    "task_instruction": "Move the book to the left zone.",
                    "target": "obj-book",
                    "destination": "left",
                }
            ],
        )
        self.assertEqual(category_good.target_id, "printed_left")

        place = _manifest(
            [],
            family="place_setting",
            target_id="right_handed",
            objects=objects,
        )
        place_good = infer_subtask_target(
            place,
            [
                {"task_instruction": "Clear the frying pan from the table."},
                {
                    "task_instruction": "Place the fork on the left.",
                    "target": "obj-fork",
                    "destination": "left",
                },
            ],
        )
        self.assertEqual(place_good.target_id, "right_handed")
        self.assertEqual(place_good.auxiliary_action_count, 1)

    def test_noop_unrelated_and_auxiliary_only_subtasks_do_not_pass(self) -> None:
        manifest = _manifest([])
        for subtasks in (
            [
                {
                    "task_instruction": (
                        "Do nothing; RGB bottom-to-top is already satisfied."
                    )
                }
            ],
            [{"task_instruction": "Wave the robot arm."}],
            [{"task_instruction": "Clear the frying pan."}],
        ):
            with self.subTest(subtasks=subtasks):
                inference = infer_subtask_target(manifest, subtasks)
                self.assertIsNone(inference.target_id)
                self.assertTrue(inference.issues)

    def test_canonical_goal_predicates_are_structured_and_preserve_truth(
        self,
    ) -> None:
        manifest = _manifest(
            [
                "SUPPORTED_BY(obj-green,obj-red)=false",
                "SUPPORTED_BY(obj-blue,obj-green)=true",
            ]
        )

        goals = canonical_goal_predicates(manifest)

        self.assertEqual(
            goals,
            (
                CanonicalPredicate(
                    "supported_by",
                    ("obj-blue", "obj-green"),
                    True,
                ),
                CanonicalPredicate(
                    "supported_by",
                    ("obj-green", "obj-red"),
                    False,
                ),
            ),
        )
        self.assertEqual(
            goals[0].to_dict()["canonical"],
            "SUPPORTED_BY(obj-blue,obj-green)=true",
        )

    def test_unique_composite_aliases_and_constants_match(self) -> None:
        objects = [
            {
                "object_id": "obj-tablet",
                "object_type": "tablet",
                "attributes": {
                    "colour": "blue",
                    "role": "display",
                    "semantic_category": "electronic",
                },
            },
            {
                "object_id": "obj-book",
                "object_type": "book",
                "attributes": {
                    "colour": "red",
                    "role": "reference",
                    "semantic_category": "printed",
                },
            },
        ]
        manifest = _manifest(
            [
                "INSIDE_SORT_ZONE(obj-tablet,zone:right)=true",
                "SUPPORTED_BY(obj-tablet,obj-book)=true",
            ],
            family="category_sort",
            target_id="printed_left",
            instruction=(
                "Place printed items on the left and electronics on the right."
            ),
            objects=objects,
        )
        validation_spec = {
            "goal_conditions": [
                {
                    "id": "sort-tablet",
                    "predicate": "inside-sort-zone",
                    "arguments": ["the blue tablet", "right zone"],
                },
                {
                    "id": "tablet-on-book",
                    "predicate": "SUPPORTED_BY",
                    "arguments": ["electronic tablet", "red book"],
                },
            ]
        }

        diff = planner_predicate_diff(manifest, validation_spec)

        self.assertTrue(diff.exact, diff.to_dict())
        self.assertEqual(diff.precision, 1.0)
        self.assertEqual(diff.recall, 1.0)
        self.assertFalse(diff.unresolved)
        self.assertFalse(diff.ambiguous)

    def test_ambiguous_alias_is_reported_with_all_candidate_ids(self) -> None:
        objects = [
            {
                "object_id": "obj-book-1",
                "object_type": "book",
                "attributes": {
                    "role": "book",
                    "semantic_category": "printed",
                },
            },
            {
                "object_id": "obj-book-2",
                "object_type": "book",
                "attributes": {
                    "role": "book",
                    "semantic_category": "printed",
                },
            },
        ]
        manifest = _manifest(
            ["ALL_ITEMS_ASSIGNED(obj-book-1,obj-book-2)=true"],
            family="category_sort",
            target_id="printed_left",
            objects=objects,
        )

        diff = planner_predicate_diff(
            manifest,
            {
                "goal_conditions": [
                    {
                        "id": "all-books",
                        "predicate": "ALL_ITEMS_ASSIGNED",
                        "arguments": ["book", "obj-book-2"],
                    }
                ]
            },
        )

        self.assertFalse(diff.exact)
        self.assertEqual(len(diff.ambiguous), 1)
        self.assertEqual(
            diff.ambiguous[0]["candidates"],
            ["obj-book-1", "obj-book-2"],
        )
        self.assertEqual(diff.ambiguous[0]["argument"], "book")
        self.assertEqual(len(diff.missing), 1)
        self.assertEqual(len(diff.extra), 1)

    def test_unresolved_argument_is_not_silently_promoted_to_a_symbol(self) -> None:
        manifest = _manifest(
            ["SUPPORTED_BY(obj-blue,obj-red)=true"],
        )

        diff = planner_predicate_diff(
            manifest,
            {
                "goal_conditions": [
                    {
                        "id": "unsupported-colour",
                        "predicate": "SUPPORTED_BY",
                        "arguments": ["purple cube", "red cube"],
                    }
                ]
            },
        )
        serialized = diff.to_dict()

        self.assertFalse(diff.exact)
        self.assertEqual(len(diff.unresolved), 1)
        self.assertEqual(diff.unresolved[0]["argument"], "purple cube")
        self.assertEqual(serialized["precision"], 0.0)
        self.assertEqual(serialized["recall"], 0.0)
        for key in (
            "expected",
            "actual",
            "matched",
            "missing",
            "extra",
            "unresolved",
            "ambiguous",
            "precision",
            "recall",
            "f1",
            "exact",
        ):
            self.assertIn(key, serialized)

    def test_counter_multiplicity_affects_recall(self) -> None:
        manifest = _manifest(
            [
                "SUPPORTED_BY(obj-blue,obj-red)=true",
                "SUPPORTED_BY(obj-blue,obj-red)=true",
            ],
        )

        diff = planner_predicate_diff(
            manifest,
            {
                "goal_conditions": [
                    {
                        "id": "only-once",
                        "predicate": "SUPPORTED_BY",
                        "arguments": ["blue", "red"],
                    }
                ]
            },
        )

        self.assertFalse(diff.exact)
        self.assertEqual(len(diff.expected), 2)
        self.assertEqual(len(diff.matched), 1)
        self.assertEqual(len(diff.missing), 1)
        self.assertEqual(diff.precision, 1.0)
        self.assertEqual(diff.recall, 0.5)
        self.assertAlmostEqual(diff.f1, 2.0 / 3.0)

    def test_truth_and_direction_are_preserved(self) -> None:
        manifest = _manifest(
            ["SUPPORTED_BY(obj-blue,obj-red)=true"],
        )

        reversed_diff = planner_predicate_diff(
            manifest,
            {
                "goal_conditions": [
                    {
                        "id": "reversed",
                        "predicate": "SUPPORTED_BY",
                        "arguments": ["red", "blue"],
                    }
                ]
            },
        )
        false_diff = planner_predicate_diff(
            manifest,
            {
                "goal_conditions": [
                    {
                        "id": "false",
                        "predicate": "SUPPORTED_BY",
                        "arguments": ["blue", "red"],
                        "truth": False,
                    }
                ]
            },
        )

        self.assertFalse(reversed_diff.exact)
        self.assertEqual(
            reversed_diff.actual[0].arguments,
            ("obj-red", "obj-blue"),
        )
        self.assertFalse(false_diff.exact)
        self.assertFalse(false_diff.actual[0].truth)

    def test_only_declared_set_predicates_are_unordered(self) -> None:
        objects = [
            {
                "object_id": "obj-1",
                "object_type": "book",
                "attributes": {"role": "first"},
            },
            {
                "object_id": "obj-2",
                "object_type": "tablet",
                "attributes": {"role": "second"},
            },
        ]
        unordered_manifest = _manifest(
            ["ALL_ITEMS_ASSIGNED(obj-1,obj-2)=true"],
            family="category_sort",
            target_id="printed_left",
            objects=objects,
        )
        ordered_manifest = _manifest(
            ["SUPPORTED_BY(obj-1,obj-2)=true"],
            objects=objects,
        )

        unordered = planner_predicate_diff(
            unordered_manifest,
            {
                "goal_conditions": [
                    {
                        "predicate": "ALL_ITEMS_ASSIGNED",
                        "arguments": ["second", "first"],
                    }
                ]
            },
        )
        ordered = planner_predicate_diff(
            ordered_manifest,
            {
                "goal_conditions": [
                    {
                        "predicate": "SUPPORTED_BY",
                        "arguments": ["second", "first"],
                    }
                ]
            },
        )

        self.assertTrue(unordered.exact, unordered.to_dict())
        self.assertFalse(ordered.exact)

    def test_missing_schema_is_an_issue_even_when_oracle_goals_are_empty(
        self,
    ) -> None:
        diff = planner_predicate_diff(_manifest([]), None)

        self.assertFalse(diff.exact)
        self.assertEqual(diff.expected, ())
        self.assertTrue(diff.issues)


if __name__ == "__main__":
    unittest.main()
