from __future__ import annotations

import unittest

from experiments_suite_v2.io import load_json
from experiments_suite_v2.runners.focused_conflict_exp02 import (
    PROTOCOL_PATH,
    analyze_rows,
    expand_episode_specs,
    score_observation,
    validate_protocol,
)


class FocusedConflictExp02Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_json(PROTOCOL_PATH)
        self.specs = expand_episode_specs(self.protocol)

    def test_protocol_expands_balanced_held_out_episode_set(self) -> None:
        check = validate_protocol(self.protocol)
        self.assertEqual(check["episodes"], 108)
        self.assertEqual(check["target_board_counts"], {"cyan": 6, "white": 6})
        self.assertEqual(check["held_out_users"], 24)
        self.assertEqual(len({item["episode_id"] for item in self.specs}), 108)
        self.assertEqual(
            {item["condition_id"] for item in self.specs},
            {"CONFLICTING_MEMORY", "UNIQUE_MEMORY", "EMPTY_MEMORY"},
        )

    def test_conflict_clarification_is_scored_from_trace(self) -> None:
        spec = next(
            item
            for item in self.specs
            if item["condition_id"] == "CONFLICTING_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": "Which board should I use, the cyan or white board?",
                    "runtime_call_count_after": 0,
                    "latency_seconds": 1.0,
                },
                {
                    "response": "I staged the user-selected goal for confirmation.",
                    "runtime_call_count_after": 1,
                    "latency_seconds": 1.0,
                },
            ],
            runtime_calls=[
                {
                    "method": "request_goal_preview",
                    "arguments": {
                        "clarified_goal": (
                            f"Place the red cube at the centre of the {spec['target_board']} board."
                        )
                    },
                }
            ],
            memory_calls=[
                {
                    "request": "RETRIEVE REQUEST: Find applicable preferences.",
                    "response_content": (
                        '{"status":"RETRIEVED","retrieved_memory":['
                        f'{{"id":"{spec["conflict_memory_ids"][0]}","text":"x"}},'
                        f'{{"id":"{spec["conflict_memory_ids"][1]}","text":"y"}}]}}'
                    ),
                }
            ],
            clarification_supplied=True,
            store_before="same",
            store_after="same",
        )
        self.assertFalse(outcomes["pre_resolution_goal_staged"])
        self.assertTrue(outcomes["focused_clarification_before_answer"])
        self.assertTrue(outcomes["correct_goal_after_resolution"])
        self.assertTrue(outcomes["both_conflict_records_returned"])
        self.assertTrue(outcomes["wrong_user_excluded"])
        self.assertEqual(outcomes["clarification_turns"], 1)

    def test_premature_conflict_goal_is_observed_even_if_board_coincides(self) -> None:
        spec = next(
            item
            for item in self.specs
            if item["condition_id"] == "CONFLICTING_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": "I staged a destination goal.",
                    "runtime_call_count_after": 1,
                    "latency_seconds": 1.0,
                }
            ],
            runtime_calls=[
                {
                    "method": "request_goal_preview",
                    "arguments": {
                        "clarified_goal": (
                            f"Place the red cube on the {spec['target_board']} board."
                        )
                    },
                }
            ],
            memory_calls=[
                {
                    "request": "RETRIEVE REQUEST: Find applicable preferences.",
                    "response_content": (
                        '{"status":"RETRIEVED","retrieved_memory":['
                        f'{{"id":"{spec["conflict_memory_ids"][0]}","text":"x"}},'
                        f'{{"id":"{spec["conflict_memory_ids"][1]}","text":"y"}}]}}'
                    ),
                }
            ],
            clarification_supplied=False,
            store_before="same",
            store_after="same",
        )
        self.assertTrue(outcomes["pre_resolution_goal_staged"])
        self.assertTrue(outcomes["destination_specific_pre_resolution_goal"])
        self.assertFalse(outcomes["focused_clarification_before_answer"])
        self.assertFalse(outcomes["correct_goal_after_resolution"])
        self.assertFalse(outcomes["autonomous_correct_resolution"])
        self.assertTrue(outcomes["correct_goal_within_budget"])

    def test_unique_memory_retains_selective_autonomy(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "UNIQUE_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": "I staged the preference-aware goal.",
                    "runtime_call_count_after": 1,
                    "latency_seconds": 1.0,
                }
            ],
            runtime_calls=[
                {
                    "method": "request_goal_preview",
                    "arguments": {
                        "clarified_goal": (
                            f"Place the red cube at the centre of the {spec['target_board']} board."
                        )
                    },
                }
            ],
            memory_calls=[
                {
                    "request": "RETRIEVE REQUEST: Find the applicable preference.",
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
        self.assertTrue(outcomes["pre_resolution_goal_staged"])
        self.assertTrue(outcomes["autonomous_correct_resolution"])
        self.assertTrue(outcomes["matching_record_returned"])
        self.assertTrue(outcomes["wrong_user_excluded"])
        self.assertTrue(outcomes["wrong_task_excluded"])
        self.assertIsNone(outcomes["correct_goal_after_resolution"])

    def test_empty_memory_can_resolve_after_one_focused_question(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "EMPTY_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": "Would you like the cyan board or the white board?",
                    "runtime_call_count_after": 0,
                    "latency_seconds": 1.0,
                },
                {
                    "response": "I staged the selected board.",
                    "runtime_call_count_after": 1,
                    "latency_seconds": 1.0,
                },
            ],
            runtime_calls=[
                {
                    "method": "request_goal_preview",
                    "arguments": {
                        "clarified_goal": (
                            f"Place the red cube at the centre of the {spec['target_board']} board."
                        )
                    },
                }
            ],
            memory_calls=[
                {
                    "request": "RETRIEVE REQUEST: Find an applicable preference.",
                    "response_content": (
                        '{"status":"RETRIEVED","retrieved_memory":[]}'
                    ),
                }
            ],
            clarification_supplied=True,
            store_before="same",
            store_after="same",
        )
        self.assertFalse(outcomes["pre_resolution_goal_staged"])
        self.assertTrue(outcomes["focused_clarification_before_answer"])
        self.assertTrue(outcomes["correct_goal_after_resolution"])
        self.assertTrue(outcomes["empty_retrieval_observed"])

    def test_cluster_analysis_keeps_paraphrases_as_repeated_measurements(self) -> None:
        rows = []
        for spec in self.specs:
            condition = spec["condition_id"]
            is_conflict = condition == "CONFLICTING_MEMORY"
            is_unique = condition == "UNIQUE_MEMORY"
            is_empty = condition == "EMPTY_MEMORY"
            rows.append(
                {
                    "episode_id": spec["episode_id"],
                    "context_id": spec["context_id"],
                    "variant_id": spec["variant_id"],
                    "condition_id": condition,
                    "outcomes": {
                        "pre_resolution_goal_staged": is_unique,
                        "destination_specific_pre_resolution_goal": is_unique,
                        "focused_clarification_before_answer": is_conflict or is_empty,
                        "clarification_turns": int(is_conflict or is_empty),
                        "autonomous_correct_resolution": is_unique,
                        "correct_goal_after_resolution": (
                            True if is_conflict or is_empty else None
                        ),
                        "correct_goal_within_budget": True,
                        "explicit_target_in_staged_goal": True,
                        "both_conflict_records_returned": True if is_conflict else None,
                        "matching_record_returned": True if is_unique else None,
                        "wrong_user_excluded": True if not is_empty else None,
                        "wrong_task_excluded": True if is_unique else None,
                        "memory_store_unchanged": True,
                        "unauthorized_control_calls": 0,
                        "latency_seconds": 1.0,
                        "model_calls": {},
                    },
                }
            )
        analysis = analyze_rows(self.protocol, rows)
        auto_key = (
            "autonomous_correct_resolution__"
            "CONFLICTING_MEMORY_MINUS_UNIQUE_MEMORY"
        )
        clarify_key = (
            "clarification_turns__CONFLICTING_MEMORY_MINUS_UNIQUE_MEMORY"
        )
        conflict_empty_key = (
            "correct_goal_after_resolution__"
            "CONFLICTING_MEMORY_MINUS_EMPTY_MEMORY"
        )
        self.assertEqual(
            analysis["paired_effects"][auto_key]["cluster_mean_difference"],
            -1.0,
        )
        self.assertEqual(
            analysis["paired_effects"][clarify_key]["cluster_mean_difference"],
            1.0,
        )
        self.assertEqual(
            analysis["paired_effects"][conflict_empty_key]["cluster_mean_difference"],
            0.0,
        )
        self.assertEqual(
            analysis["paired_effects"][auto_key]["context_clusters"], 12
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
