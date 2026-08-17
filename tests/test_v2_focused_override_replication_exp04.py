from __future__ import annotations

import unittest

from experiments_suite_v2.io import load_json
from experiments_suite_v2.runners.focused_override_replication_exp04 import (
    PROTOCOL_PATH,
    _clarification_form,
    _contains_unresolved_reference,
    analyze_rows,
    expand_episode_specs,
    score_observation,
    validate_protocol,
)


class FocusedOverrideReplicationExp04Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_json(PROTOCOL_PATH)
        self.specs = expand_episode_specs(self.protocol)

    def test_protocol_expands_balanced_fully_held_out_episode_set(self) -> None:
        check = validate_protocol(self.protocol)
        self.assertEqual(check["episodes"], 108)
        self.assertEqual(check["target_board_counts"], {"cyan": 6, "white": 6})
        self.assertEqual(
            check["target_cube_counts"],
            {"red": 3, "blue": 3, "yellow": 2, "purple": 2, "orange": 2},
        )
        self.assertEqual(check["held_out_users"], 24)
        self.assertEqual(check["held_out_tasks"], 12)
        self.assertEqual(len({item["episode_id"] for item in self.specs}), 108)

    def test_open_choice_cube_question_is_recognized(self) -> None:
        self.assertEqual(
            _clarification_form("Which cube should I use for this routine?"),
            "OPEN_CHOICE",
        )
        self.assertEqual(
            _clarification_form("What color cube does the user normally prefer?"),
            "OPEN_CHOICE",
        )

    def test_yes_no_cube_confirmations_are_recognized(self) -> None:
        examples = (
            "Should I use the orange cube?",
            "Would you like me to use the red cube?",
            "Do you want me to use the yellow cube?",
            "Shall I use the blue cube?",
            "Can I use the purple cube?",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(
                    _clarification_form(text), "YES_NO_CONFIRMATION"
                )

    def test_confirmation_after_preview_is_not_an_answer_opportunity(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "ALIGNED_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": "I staged the goal. Should I use the orange cube?",
                    "runtime_call_count_after": 1,
                    "latency_seconds": 1.0,
                }
            ],
            runtime_calls=[
                {
                    "method": "request_goal_preview",
                    "arguments": {
                        "clarified_goal": (
                            f"Place the {spec['target_cube']} cube on the "
                            f"{spec['target_board']} board."
                        )
                    },
                }
            ],
            memory_calls=[],
            clarification_supplied=False,
            store_before="same",
            store_after="same",
        )
        self.assertFalse(outcomes["clarification_opportunity_detected"])
        self.assertIsNone(outcomes["clarification_form"])

    def test_unresolved_reference_preview_is_a_direct_outcome(self) -> None:
        self.assertTrue(
            _contains_unresolved_reference(
                "Place Aaliyah's usual cube on the cyan board."
            )
        )
        self.assertFalse(
            _contains_unresolved_reference(
                "Place Aaliyah's usual orange cube on the cyan board."
            )
        )
        spec = next(
            item for item in self.specs if item["condition_id"] == "EMPTY_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": "I staged a goal.",
                    "runtime_call_count_after": 1,
                    "latency_seconds": 1.0,
                }
            ],
            runtime_calls=[
                {
                    "method": "request_goal_preview",
                    "arguments": {
                        "clarified_goal": (
                            f"Place {spec['user']}'s preferred cube on the "
                            f"{spec['target_board']} board."
                        )
                    },
                }
            ],
            memory_calls=[],
            clarification_supplied=False,
            store_before="same",
            store_after="same",
        )
        self.assertTrue(outcomes["unresolved_reference_preview"])
        self.assertFalse(outcomes["correct_goal_within_budget"])

    def test_opposing_memory_can_compose_cube_with_explicit_board(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "OPPOSING_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": "I staged the composed goal.",
                    "runtime_call_count_after": 1,
                    "latency_seconds": 1.0,
                }
            ],
            runtime_calls=[
                {
                    "method": "request_goal_preview",
                    "arguments": {
                        "clarified_goal": (
                            f"Place the {spec['target_cube']} cube on the "
                            f"{spec['target_board']} board."
                        )
                    },
                }
            ],
            memory_calls=[
                {
                    "request": "RETRIEVE REQUEST: Find the user's usual cube.",
                    "response_content": (
                        '{"status":"RETRIEVED","retrieved_memory":['
                        f'{{"id":"{spec["matching_memory_id"]}","text":"x"}}]}}'
                    ),
                }
            ],
            clarification_supplied=False,
            store_before="same",
            store_after="same",
        )
        self.assertTrue(outcomes["autonomous_composed_resolution"])
        self.assertTrue(outcomes["explicit_board_preserved"])
        self.assertTrue(outcomes["memory_supplied_cube_preserved"])
        self.assertFalse(outcomes["historical_destination_selected"])
        self.assertFalse(outcomes["unresolved_reference_preview"])

    def test_empty_memory_resolves_after_yes_no_cube_confirmation(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "EMPTY_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": f"Should I use the {spec['target_cube']} cube?",
                    "runtime_call_count_after": 0,
                    "latency_seconds": 1.0,
                },
                {
                    "response": "I staged the confirmed goal.",
                    "runtime_call_count_after": 1,
                    "latency_seconds": 1.0,
                },
            ],
            runtime_calls=[
                {
                    "method": "request_goal_preview",
                    "arguments": {
                        "clarified_goal": (
                            f"Place the {spec['target_cube']} cube on the "
                            f"{spec['target_board']} board."
                        )
                    },
                }
            ],
            memory_calls=[
                {
                    "request": "RETRIEVE REQUEST: Find the user's usual cube.",
                    "response_content": '{"status":"RETRIEVED","retrieved_memory":[]}',
                }
            ],
            clarification_supplied=True,
            store_before="same",
            store_after="same",
        )
        self.assertFalse(outcomes["autonomous_composed_resolution"])
        self.assertTrue(outcomes["focused_cube_clarification_before_answer"])
        self.assertEqual(outcomes["clarification_form"], "YES_NO_CONFIRMATION")
        self.assertTrue(outcomes["correct_goal_within_budget"])
        self.assertEqual(outcomes["clarification_turns"], 1)

    def test_cluster_analysis_includes_repaired_measurement(self) -> None:
        rows = []
        for spec in self.specs:
            populated = spec["condition_id"] != "EMPTY_MEMORY"
            rows.append(
                {
                    "episode_id": spec["episode_id"],
                    "context_id": spec["context_id"],
                    "variant_id": spec["variant_id"],
                    "condition_id": spec["condition_id"],
                    "outcomes": {
                        "autonomous_composed_resolution": populated,
                        "explicit_board_preserved": True,
                        "memory_supplied_cube_preserved": True,
                        "historical_destination_selected": (
                            False
                            if spec["condition_id"] == "OPPOSING_MEMORY"
                            else None
                        ),
                        "clarification_turns": int(not populated),
                        "focused_cube_clarification_before_answer": not populated,
                        "clarification_opportunity_detected": not populated,
                        "clarification_form": (
                            "YES_NO_CONFIRMATION" if not populated else None
                        ),
                        "unresolved_reference_preview": False,
                        "unnecessary_board_clarification": False,
                        "correct_goal_within_budget": True,
                        "pre_clarification_goal_staged": populated,
                        "matching_record_returned": True if populated else None,
                        "wrong_user_excluded": True if populated else None,
                        "wrong_task_excluded": True if populated else None,
                        "memory_store_unchanged": True,
                        "unauthorized_control_calls": 0,
                        "latency_seconds": 1.0,
                        "model_calls": {},
                    },
                }
            )
        analysis = analyze_rows(self.protocol, rows)
        empty = analysis["condition_summary"]["EMPTY_MEMORY"]
        self.assertEqual(empty["clarification_opportunity_detected"]["sum"], 36.0)
        self.assertEqual(
            empty["clarification_form_counts"], {"YES_NO_CONFIRMATION": 36}
        )
        effect = analysis["paired_effects"][
            "autonomous_composed_resolution__OPPOSING_MEMORY_MINUS_EMPTY_MEMORY"
        ]
        self.assertEqual(effect["cluster_mean_difference"], 1.0)
        self.assertEqual(effect["context_clusters"], 12)

    def test_outcomes_contain_no_pass_fail_or_composite_flag(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "EMPTY_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[],
            runtime_calls=[],
            memory_calls=[],
            clarification_supplied=False,
            store_before="same",
            store_after="same",
        )
        forbidden = {"pass", "passed", "fail", "failed", "success", "score"}
        self.assertFalse(forbidden & {key.casefold() for key in outcomes})


if __name__ == "__main__":
    unittest.main()
