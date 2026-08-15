from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from experiments_suite_v2.runners import c_val_development as target


class CValidatorDevelopmentTests(unittest.TestCase):
    def test_exact_registry_window_and_aim_order(self) -> None:
        rows = target._prefix_trials("TEST")
        self.assertEqual(len(rows), 200)
        self.assertEqual([row.order_index for row in rows], list(range(920, 1120)))
        self.assertEqual(list(dict.fromkeys(row.case_definition.aim_id for row in rows)), list(target._AIM_ORDER))
        for aim in target._AIM_ORDER:
            self.assertEqual(sum(row.case_definition.aim_id == aim for row in rows), 40)

    def test_all_registered_labels_are_bound(self) -> None:
        rows = target._prefix_trials("TEST")
        labels = {aim: {target._label(row) for row in rows if row.case_definition.aim_id == aim} for aim in target._AIM_ORDER}
        self.assertEqual(labels["C-VAL-LIST"], set(target._LIST_GOALS))
        self.assertEqual(labels["C-VAL-STATE"], set(target._STATE_CASES))
        self.assertEqual(labels["C-VAL-HIST"], set(target._HISTORY_CASES))
        self.assertEqual(labels["C-VAL-EVID"], set(target._EVIDENCE_CASES))

    def test_all_referenced_frames_are_hash_valid(self) -> None:
        states = {spec[2] for spec in target._STATE_CASES.values()}
        states.update(spec[0] for spec in target._HISTORY_CASES.values())
        for frames, _expected in target._EVIDENCE_CASES.values():
            states.update(frames)
        for index, state_id in enumerate(sorted(states), start=1):
            frame = target._frame(state_id, observed_at=float(index), sequence=index)
            self.assertEqual(frame.image_block["type"], "image_url")

    def test_host_contract_and_reset_paths_are_deterministic(self) -> None:
        rows = target._prefix_trials("TEST")
        for label in ("new_goal", "revision", "aborted_goal", "final_fail_replan", "completed_goal", "stale_callback", "repeated_validation", "timeout"):
            row = next(row for row in rows if row.case_definition.aim_id == "C-VAL-RESET" and target._label(row) == label)
            rubric, captures = target._reset_case(row)
            self.assertTrue(rubric["passed"], label)
            self.assertEqual(captures, ())

    def test_validation_contract_preserves_all_broad_items(self) -> None:
        contract = target._contract(("First visible outcome.", "Second visible outcome."), goal_id="test-goal")
        self.assertEqual(len(contract.broad_items), 2)
        self.assertEqual([item.broad_index for item in contract.broad_items], [0, 1])
        self.assertEqual(len(contract.detailed_criteria), 2)

    def test_model_paths_parse_production_contracts_offline(self) -> None:
        class FakeRecordingModel:
            def __init__(self) -> None:
                self.records = []

            def invoke(self, messages, **kwargs):
                del kwargs
                request = next(
                    block["text"]
                    for block in messages[-1].content
                    if block.get("type") == "text"
                )
                import json
                payload = json.loads(request)
                if payload["mode"] == "COMPILE_CHECKLIST":
                    body = {
                        "broad_items": [
                            {
                                "broad_index": item["broad_index"],
                                "detailed_criteria": [item["label"]],
                            }
                            for item in payload["broad_items"]
                        ]
                    }
                else:
                    criteria = [
                        criterion
                        for broad in payload["validation_contract"]["broad_items"]
                        for criterion in broad["detailed_criteria"]
                    ]
                    body = {
                        "emergency_stop": False,
                        "emergency_reason": None,
                        "criteria": [
                            {
                                "id": item["id"],
                                "state": "MET",
                                "evidence": "The required arrangement is visibly present.",
                            }
                            for item in criteria
                        ],
                        "observation": "The requested arrangement is visibly established.",
                    }
                raw = json.dumps(body, separators=(",", ":"))
                self.records.append({
                    "schema_version": "test", "raw_response": raw,
                    "latency_seconds": 0.0, "recorded_at": "test",
                })
                return AIMessage(content=raw)

        rows = target._prefix_trials("TEST")
        with patch.object(target, "_RecordingModel", FakeRecordingModel):
            list_row = next(row for row in rows if row.case_definition.aim_id == "C-VAL-LIST" and target._label(row) == "board")
            rubric, captures = target._list_case(list_row)
            self.assertTrue(rubric["compiled_contract"])
            self.assertEqual(len(captures), 1)
            state_row = next(row for row in rows if row.case_definition.aim_id == "C-VAL-STATE" and target._label(row) == "V04")
            rubric, captures = target._state_case(state_row)
            self.assertTrue(rubric["passed"])
            self.assertEqual(rubric["observed"], "COMPLETE")
            self.assertEqual(len(captures), 1)


if __name__ == "__main__":
    unittest.main()
