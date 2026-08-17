from __future__ import annotations

import json
import unittest

from experiments_suite_v2.runners.c_pln_development import (
    _AIM_ORDER,
    _FRAME_STATE,
    evaluate_decision,
    planner_request,
)
from experiments_suite_v2.runners.component import expand_component_trials


class CPlannerDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = tuple(
            trial
            for trial in expand_component_trials(software_config_commit="WORKTREE")
            if trial.case_definition.aim_id in _AIM_ORDER
        )

    def test_exact_registered_window_and_contexts(self) -> None:
        self.assertEqual(len(self.rows), 240)
        self.assertEqual([row.order_index for row in self.rows], list(range(240, 480)))
        for aim in _AIM_ORDER:
            rows = [row for row in self.rows if row.case_definition.aim_id == aim]
            self.assertEqual(len(rows), 40)
            self.assertEqual(
                {row.trial_tuple.extra_frozen_factors["context_label"] for row in rows},
                set(_FRAME_STATE[aim]),
            )

    def test_every_row_builds_a_typed_oracle_free_request(self) -> None:
        for row in self.rows:
            request = planner_request(row)
            payload = request.to_dict()
            self.assertEqual(payload["cycle_id"], len(payload["execution_history"]) + 1)
            self.assertNotIn("oracle_label", json.dumps(payload, sort_keys=True))
            self.assertNotIn("expected_values", json.dumps(payload, sort_keys=True))

    def _row(self, aim: str, label: str):
        return next(
            row
            for row in self.rows
            if row.case_definition.aim_id == aim
            and row.trial_tuple.extra_frozen_factors["context_label"] == label
        )

    def test_next_and_done_routes(self) -> None:
        complete = {
            "decision": "REQUEST_FINAL_VALIDATION",
            "candidate_tasks": [],
            "reason": "The visible goal is complete.",
            "blocked_reason": None,
            "user_question": None,
        }
        self.assertTrue(
            evaluate_decision(
                self._row("C-PLN-NEXT", "already_satisfied"), complete
            )["passed"]
        )
        self.assertTrue(
            evaluate_decision(
                self._row("C-PLN-DONE", "scene_correct_history_incomplete"),
                complete,
            )["passed"]
        )
        self.assertFalse(
            evaluate_decision(
                self._row("C-PLN-DONE", "history_success_scene_wrong"), complete
            )["passed"]
        )

    def test_constraint_rubric_rejects_protected_distractor(self) -> None:
        bad = {
            "decision": "ACT",
            "candidate_tasks": [
                {
                    "instruction": "Move the yellow cube, then place red on blue.",
                    "expected_observation": ["The red cube is stable."],
                    "known_failure_conditions": [],
                }
            ],
            "reason": None,
            "blocked_reason": None,
            "user_question": None,
        }
        result = evaluate_decision(
            self._row("C-PLN-GROUND", "forbidden_action"), bad
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["protected_distractor_not_targeted"])

    def test_safe_fence_routes_are_explicit(self) -> None:
        ask = {
            "decision": "NEEDS_USER_INPUT",
            "candidate_tasks": [],
            "reason": "The referent is ambiguous.",
            "blocked_reason": None,
            "user_question": "Which object should be moved?",
        }
        self.assertTrue(
            evaluate_decision(
                self._row("C-PLN-FENCE", "ambiguous_reference"), ask
            )["passed"]
        )


if __name__ == "__main__":
    unittest.main()
