from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments_suite_v2.io import load_json, sha256_file
from experiments_suite_v2.runners.component import InvocationCapture
from experiments_suite_v2.runners import i_interactions_development as subject
from experiments_suite_v2.schemas import canonical_sha256


def _capture(output=None):
    return InvocationCapture(output=output or {}, trace=())


def _trial(aim: str, label: str):
    return next(
        row
        for row in subject._prefix_trials("WORKTREE")
        if row.case_definition.aim_id == aim
        and row.trial_tuple.extra_frozen_factors["context_label"] == label
        and row.variant_id == "01"
    )


class IInteractionsDevelopmentTests(unittest.TestCase):
    def test_exact_registered_window_and_scenario_matrix(self) -> None:
        rows = subject._prefix_trials("WORKTREE")
        self.assertEqual(len(rows), 480)
        self.assertEqual([row.order_index for row in rows], list(range(1120, 1600)))
        self.assertEqual(
            canonical_sha256([row.trial_id for row in rows]),
            "6150eb66b465282f61077aaead7278fb065c26006743de809db5c3fd3f3756f3",
        )
        self.assertEqual(tuple(subject._SCENARIOS), subject._AIM_ORDER)
        self.assertTrue(all(len(values) == 8 for values in subject._SCENARIOS.values()))

    def test_unresolved_hri_contract_blocks_planner(self) -> None:
        trial = _trial("I-HP-CONTRACT", "A01_missing_board")
        hri_value = (
            {"content": "I need clarification: which board, white or cyan?"},
            _capture(),
            {"runtime_context": {}, "tool_calls": [], "console_transcript": ""},
        )
        with patch.object(subject, "_hri_call", return_value=hri_value), patch.object(
            subject, "_planner_call"
        ) as planner:
            capture = subject._hp_contract(trial)
        planner.assert_not_called()
        self.assertEqual(
            capture.output["final"]["oracle_label"],
            "A01_missing_board",
        )
        self.assertTrue(capture.output["final"]["rubric"]["passed"])

    def test_resolved_current_instruction_is_handed_to_planner_once(self) -> None:
        trial = _trial("I-HP-CONTRACT", "A05_current_instruction_conflict")
        hri_value = (
            {"content": "The explicit current instruction selects white for this task."},
            _capture(),
            {
                "runtime_context": {},
                "tool_calls": [
                    {
                        "method": "request_goal_preview",
                        "arguments": {
                            "clarified_goal": "Place the red cube at the centre of the white board.",
                            "constraints": [],
                            "operator_guidance": None,
                        },
                    }
                ],
                "console_transcript": "",
            },
        )
        planner_value = (
            {"decision": "ACT", "candidate_tasks": [{"instruction": "Place red on white."}]},
            _capture(),
            {
                "goal_contract": {"goal": "Place the red cube at the centre of the white board."},
                "trigger": "CONFIRMED",
            },
        )
        with patch.object(subject, "_hri_call", return_value=hri_value), patch.object(
            subject, "_planner_call", return_value=planner_value
        ) as planner:
            capture = subject._hp_contract(trial)
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(
            capture.output["final"]["oracle_label"],
            "A05_current_instruction_conflict",
        )

    def test_preview_report_uses_actual_decision_without_control_tool(self) -> None:
        trial = _trial("I-HP-PREVIEW", "single_step")
        planner_value = (
            {
                "decision": "ACT",
                "candidate_tasks": [
                    {"instruction": "Place the red cube at the centre of the white board."}
                ],
            },
            _capture(),
            {"trigger": "CONFIRMED", "execution_history": []},
        )
        hri_value = (
            {"content": "The plan is to place the red cube on the white board; no action has executed."},
            _capture(),
            {"runtime_context": {}, "tool_calls": [], "console_transcript": ""},
        )
        with patch.object(subject, "_planner_call", return_value=planner_value), patch.object(
            subject, "_hri_call", return_value=hri_value
        ):
            capture = subject._hp_preview(trial)
        self.assertEqual(capture.output["final"]["oracle_label"], "single_step")
        self.assertTrue(capture.output["final"]["rubric"]["passed"])

    def test_memory_fixtures_are_row_local_and_non_scoring(self) -> None:
        clean = subject._memory_records("I-HM-CLEAN", "similar_irrelevant")
        self.assertEqual(
            [row["id"] for row in clean],
            ["board-cyan", "packing-white", "unrelated-camera"],
        )
        trial = _trial("I-HM-UPDATE", "ownership_conflict")
        request = subject._memory_request(trial)
        self.assertTrue(request.startswith("MUTATE REQUEST:"))
        self.assertNotIn("oracle_label", request)

    def test_protocol_and_freeze_hashes_are_self_consistent(self) -> None:
        freeze = load_json(subject.FREEZE_PATH)
        self.assertEqual(freeze["protocol_sha256"], sha256_file(subject.PROTOCOL_PATH))
        self.assertEqual(
            freeze["implementation"]["runner_sha256"],
            sha256_file(Path(subject.__file__)),
        )
        self.assertEqual(
            freeze["implementation"]["tests_sha256"],
            sha256_file(Path(__file__)),
        )
        self.assertEqual(load_json(subject.PROTOCOL_PATH)["status"], "FROZEN_CANDIDATE_NOT_EXECUTED")


if __name__ == "__main__":
    unittest.main()
