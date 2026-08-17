from __future__ import annotations

import unittest

from experiments_suite_v2.io import load_json
from experiments_suite_v2.runners.focused_ambiguity_exp01 import (
    PROTOCOL_PATH,
    analyze_rows,
    expand_episode_specs,
    score_observation,
    validate_protocol,
)


class FocusedAmbiguityExp01Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_json(PROTOCOL_PATH)
        self.specs = expand_episode_specs(self.protocol)

    def test_protocol_expands_balanced_frozen_episode_set(self) -> None:
        check = validate_protocol(self.protocol)
        self.assertEqual(check["episodes"], 108)
        self.assertEqual(check["target_board_counts"], {"cyan": 6, "white": 6})
        self.assertEqual(len({item["episode_id"] for item in self.specs}), 108)

    def test_scoped_memory_scores_autonomous_correct_resolution(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "SCOPED_MEMORY"
        )
        runtime_calls = [
            {
                "method": "request_goal_preview",
                "arguments": {
                    "clarified_goal": (
                        f"Place the red cube at the centre of the {spec['target_board']} board."
                    )
                },
            }
        ]
        memory_calls = [
            {
                "request": "RETRIEVE REQUEST: Find the applicable preference.",
                "response_content": (
                    '{"status":"RETRIEVED","retrieved_memory":['
                    f'{{"id":"{spec["matching_memory_id"]}","text":"x","similarity":0.9}}]'
                    "}"
                ),
            }
        ]
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": "I staged the resolved goal for confirmation.",
                    "runtime_call_count_after": 1,
                    "latency_seconds": 1.0,
                }
            ],
            runtime_calls=runtime_calls,
            memory_calls=memory_calls,
            clarification_supplied=False,
            store_before="same",
            store_after="same",
        )
        self.assertTrue(outcomes["autonomous_correct_resolution"])
        self.assertTrue(outcomes["correct_goal_within_budget"])
        self.assertTrue(outcomes["retrieval_scope_correct"])
        self.assertEqual(outcomes["clarification_turns"], 0)

    def test_empty_memory_can_reach_same_goal_after_focused_clarification(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "EMPTY_MEMORY"
        )
        outcomes = score_observation(
            spec,
            turns=[
                {
                    "response": "Which board should I use, the white or cyan board?",
                    "runtime_call_count_after": 0,
                    "latency_seconds": 1.0,
                },
                {
                    "response": "I staged that goal for confirmation.",
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
                    "request": "RETRIEVE REQUEST: Find the applicable preference.",
                    "response_content": '{"status":"RETRIEVED","retrieved_memory":[]}',
                }
            ],
            clarification_supplied=True,
            store_before="same",
            store_after="same",
        )
        self.assertFalse(outcomes["autonomous_correct_resolution"])
        self.assertTrue(outcomes["correct_goal_within_budget"])
        self.assertTrue(outcomes["focused_clarification_before_answer"])
        self.assertTrue(outcomes["retrieval_scope_correct"])
        self.assertEqual(outcomes["clarification_turns"], 1)

    def test_wrong_scope_memory_is_visible_as_separate_outcome(self) -> None:
        spec = next(
            item for item in self.specs if item["condition_id"] == "SCOPED_MEMORY"
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
                            f"Place the red cube on the {spec['opposite_board']} board."
                        )
                    },
                }
            ],
            memory_calls=[
                {
                    "request": "RETRIEVE REQUEST: Find a preference.",
                    "response_content": (
                        '{"status":"RETRIEVED","retrieved_memory":['
                        f'{{"id":"{spec["context_id"]}-wrong-user","text":"x","similarity":0.9}}]'
                        "}"
                    ),
                }
            ],
            clarification_supplied=False,
            store_before="same",
            store_after="same",
        )
        self.assertFalse(outcomes["correct_goal_within_budget"])
        self.assertFalse(outcomes["retrieval_scope_correct"])
        self.assertNotIn("passed", outcomes)
        self.assertNotIn("failed", outcomes)

    def test_cluster_analysis_uses_context_as_unit(self) -> None:
        rows = []
        for spec in self.specs:
            condition = spec["condition_id"]
            autonomous = condition in {"SCOPED_MEMORY", "EXACT_INSTRUCTION"}
            clarification = condition == "EMPTY_MEMORY"
            rows.append(
                {
                    "episode_id": spec["episode_id"],
                    "context_id": spec["context_id"],
                    "variant_id": spec["variant_id"],
                    "condition_id": condition,
                    "outcomes": {
                        "autonomous_correct_resolution": autonomous,
                        "clarification_turns": int(clarification),
                        "correct_goal_within_budget": True,
                        "explicit_target_in_staged_goal": True,
                        "memory_store_unchanged": True,
                        "unauthorized_control_calls": 0,
                        "latency_seconds": 1.0,
                    },
                }
            )
        analysis = analyze_rows(self.protocol, rows)
        autonomy = analysis["paired_effects"]["autonomous_correct_resolution"]
        clarification = analysis["paired_effects"]["clarification_turns"]
        self.assertEqual(autonomy["context_clusters"], 12)
        self.assertEqual(autonomy["paired_rows"], 36)
        self.assertEqual(autonomy["cluster_mean_difference"], 1.0)
        self.assertEqual(clarification["cluster_mean_difference"], -1.0)


if __name__ == "__main__":
    unittest.main()
