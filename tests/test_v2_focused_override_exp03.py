from __future__ import annotations

import unittest

from experiments_suite_v2.io import load_json
from experiments_suite_v2.runners.focused_override_exp03 import (
    PROTOCOL_PATH,
    analyze_rows,
    expand_episode_specs,
    score_observation,
    validate_protocol,
)


class FocusedOverrideExp03Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_json(PROTOCOL_PATH)
        self.specs = expand_episode_specs(self.protocol)

    def test_protocol_expands_balanced_held_out_episode_set(self) -> None:
        check = validate_protocol(self.protocol)
        self.assertEqual(check["episodes"], 108)
        self.assertEqual(check["target_board_counts"], {"cyan": 6, "white": 6})
        self.assertEqual(
            check["target_cube_counts"],
            {"red": 3, "blue": 3, "yellow": 2, "purple": 2, "orange": 2},
        )
        self.assertEqual(check["held_out_users"], 24)
        self.assertEqual(len({item["episode_id"] for item in self.specs}), 108)

    def test_opposing_memory_can_compose_cube_with_explicit_board(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "OPPOSING_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": "I staged the composed goal for confirmation.",
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
        self.assertTrue(outcomes["matching_record_returned"])
        self.assertTrue(outcomes["wrong_user_excluded"])
        self.assertTrue(outcomes["wrong_task_excluded"])
        self.assertTrue(outcomes["memory_store_unchanged"])

    def test_historical_destination_substitution_is_a_direct_outcome(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "OPPOSING_MEMORY"
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
                            f"Place the {spec['target_cube']} cube on the "
                            f"{spec['opposite_board']} board."
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
        self.assertTrue(outcomes["memory_supplied_cube_preserved"])
        self.assertFalse(outcomes["explicit_board_preserved"])
        self.assertTrue(outcomes["historical_destination_selected"])
        self.assertFalse(outcomes["correct_goal_within_budget"])

    def test_empty_memory_resolves_after_focused_cube_clarification(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "EMPTY_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": "Which cube should I use for this task?",
                    "runtime_call_count_after": 0,
                    "latency_seconds": 1.0,
                },
                {
                    "response": "I staged the clarified goal.",
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
                    "response_content": (
                        '{"status":"RETRIEVED","retrieved_memory":[]}'
                    ),
                }
            ],
            clarification_supplied=True,
            store_before="same",
            store_after="same",
        )
        self.assertFalse(outcomes["autonomous_composed_resolution"])
        self.assertTrue(outcomes["focused_cube_clarification_before_answer"])
        self.assertFalse(outcomes["unnecessary_board_clarification"])
        self.assertTrue(outcomes["correct_goal_within_budget"])
        self.assertTrue(outcomes["empty_retrieval_observed"])
        self.assertEqual(outcomes["clarification_turns"], 1)

    def test_board_requestion_is_distinguished_from_cube_clarification(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "EMPTY_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": "Which board should I use, cyan or white?",
                    "runtime_call_count_after": 0,
                    "latency_seconds": 1.0,
                }
            ],
            runtime_calls=[],
            memory_calls=[],
            clarification_supplied=False,
            store_before="same",
            store_after="same",
        )
        self.assertFalse(outcomes["focused_cube_clarification_before_answer"])
        self.assertTrue(outcomes["unnecessary_board_clarification"])
        self.assertFalse(outcomes["correct_goal_within_budget"])

    def test_scope_leakage_is_reported_separately_from_final_goal(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "ALIGNED_MEMORY"
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
                        f'{{"id":"{spec["matching_memory_id"]}","text":"x"}},'
                        f'{{"id":"{spec["wrong_task_memory_id"]}","text":"y"}}]}}'
                    ),
                }
            ],
            clarification_supplied=False,
            store_before="same",
            store_after="same",
        )
        self.assertTrue(outcomes["correct_goal_within_budget"])
        self.assertTrue(outcomes["matching_record_returned"])
        self.assertFalse(outcomes["wrong_task_excluded"])
        self.assertTrue(outcomes["wrong_user_excluded"])

    def test_cluster_analysis_uses_context_as_inferential_unit(self) -> None:
        rows = []
        for spec in self.specs:
            condition = spec["condition_id"]
            populated = condition in {"OPPOSING_MEMORY", "ALIGNED_MEMORY"}
            opposing = condition == "OPPOSING_MEMORY"
            rows.append(
                {
                    "episode_id": spec["episode_id"],
                    "context_id": spec["context_id"],
                    "variant_id": spec["variant_id"],
                    "condition_id": condition,
                    "outcomes": {
                        "autonomous_composed_resolution": populated,
                        "explicit_board_preserved": True,
                        "memory_supplied_cube_preserved": True,
                        "historical_destination_selected": False if opposing else None,
                        "clarification_turns": int(not populated),
                        "focused_cube_clarification_before_answer": not populated,
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
        opposing_aligned = (
            "autonomous_composed_resolution__"
            "OPPOSING_MEMORY_MINUS_ALIGNED_MEMORY"
        )
        opposing_empty = (
            "autonomous_composed_resolution__"
            "OPPOSING_MEMORY_MINUS_EMPTY_MEMORY"
        )
        clarification = "clarification_turns__OPPOSING_MEMORY_MINUS_EMPTY_MEMORY"
        self.assertEqual(
            analysis["paired_effects"][opposing_aligned]["cluster_mean_difference"],
            0.0,
        )
        self.assertEqual(
            analysis["paired_effects"][opposing_empty]["cluster_mean_difference"],
            1.0,
        )
        self.assertEqual(
            analysis["paired_effects"][clarification]["cluster_mean_difference"],
            -1.0,
        )
        self.assertEqual(
            analysis["paired_effects"][opposing_empty]["context_clusters"], 12
        )

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
