from __future__ import annotations

import unittest

from experiments_suite_v2.runners.c_mem_development import (
    _AIM_ORDER,
    _expected_ids,
    _store_records,
    evaluate_memory,
)
from experiments_suite_v2.runners.component import expand_component_trials


class CMemoryDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = tuple(
            row
            for row in expand_component_trials(software_config_commit="WORKTREE")
            if row.case_definition.aim_id in _AIM_ORDER
        )

    def _row(self, aim: str, label: str):
        return next(
            row
            for row in self.rows
            if row.case_definition.aim_id == aim
            and row.trial_tuple.extra_frozen_factors["context_label"] == label
        )

    def test_exact_registered_window(self) -> None:
        self.assertEqual(len(self.rows), 240)
        self.assertEqual([row.order_index for row in self.rows], list(range(480, 720)))
        for aim in _AIM_ORDER:
            self.assertEqual(sum(row.case_definition.aim_id == aim for row in self.rows), 40)

    def test_all_store_ids_are_unique(self) -> None:
        for row in self.rows:
            aim = row.case_definition.aim_id
            label = str(row.trial_tuple.extra_frozen_factors["context_label"])
            records = _store_records(aim, label)
            ids = [item["id"] for item in records]
            self.assertEqual(len(ids), len(set(ids)), (aim, label))

    def test_query_rubric_accepts_expected_id_only(self) -> None:
        row = self._row("C-MEM-QUERY", "board_preference")
        before = _store_records("C-MEM-QUERY", "board_preference")
        raw = {
            "status": "FOUND",
            "retrieved_memory": [
                {
                    "id": "pref-board-cyan",
                    "memory_type": "PREFERENCE",
                    "text": before[0]["text"],
                    "similarity": 0.9,
                }
            ],
            "warnings": [],
        }
        self.assertTrue(evaluate_memory(row, raw, before=before, after=before)["passed"])

    def test_semantic_filter_rejects_wrong_scope(self) -> None:
        row = self._row("C-MEM-SEM", "wrong_user_scope")
        before = _store_records("C-MEM-SEM", "wrong_user_scope")
        raw = {
            "status": "FOUND",
            "retrieved_memory": [
                {"id": "pref-bob-cyan", "memory_type": "PREFERENCE", "text": "x", "similarity": 0.8}
            ],
            "warnings": [],
        }
        result = evaluate_memory(row, raw, before=before, after=before)
        self.assertFalse(result["passed"])
        self.assertEqual(_expected_ids("C-MEM-SEM", "wrong_user_scope"), {"pref-alice-white"})

    def test_empty_store_sizes_are_exact(self) -> None:
        for size in range(8):
            label = f"store_size_{size}"
            row = self._row("C-MEM-EMPTY", label)
            before = _store_records("C-MEM-EMPTY", label)
            raw = {
                "candidates": [
                    {"id": item["id"], "text": item["text"], "similarity": 0.5}
                    for item in before[:5]
                ]
            }
            self.assertTrue(evaluate_memory(row, raw, before=before, after=before)["passed"])


if __name__ == "__main__":
    unittest.main()
