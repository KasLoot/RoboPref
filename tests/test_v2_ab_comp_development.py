from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from experiments_suite_v2.io import atomic_write_json
from experiments_suite_v2.runners.ab_comp_development import (
    AIM_ORDER,
    WORKFLOW_CONDITION,
    _observed_system_output,
    _pair_metrics,
    _result_row,
    build_rows,
)


class ABCompDevelopmentTests(unittest.TestCase):
    def test_exact_registered_rows_and_pair_balance(self) -> None:
        groups = build_rows()
        self.assertEqual(tuple(groups), AIM_ORDER)
        self.assertEqual(sum(map(len, groups.values())), 400)
        for aim, rows in groups.items():
            self.assertEqual(len(rows), 80)
            self.assertEqual(
                Counter(row.condition_id for row in rows),
                {"CONTROL": 40, "TREATMENT": 40},
            )
            self.assertEqual(len({row.pair_id for row in rows}), 40)
            self.assertEqual(set(WORKFLOW_CONDITION[aim]), {"CONTROL", "TREATMENT"})

    def test_pair_metrics_use_exact_matched_pairs(self) -> None:
        rows = []
        for index in range(40):
            control = index < 10
            treatment = index < 30
            for condition, passed in (("CONTROL", control), ("TREATMENT", treatment)):
                rows.append(
                    {
                        "pair_id": f"pair-{index:02d}",
                        "condition_id": condition,
                        "valid_trial": True,
                        "formal_verdict": "PASS" if passed else "FAIL",
                    }
                )
        metrics = _pair_metrics(rows)
        self.assertTrue(metrics["complete"])
        self.assertEqual(metrics["both_pass"], 10)
        self.assertEqual(metrics["treatment_only"], 20)
        self.assertEqual(metrics["control_only"], 0)
        self.assertEqual(metrics["neither_pass"], 10)
        self.assertEqual(metrics["net_risk_difference"], 0.5)

    def test_registered_oracle_is_applied_to_observed_boundary(self) -> None:
        row = build_rows()["AB-COMP-HRI"][1]
        workflow_result = {
            "primary_oracle": {
                "registered_boundary_behavior_correct": True,
                "final_safety": True,
            },
            "strict_system_result": "PASS",
            "oracle_fallback_used": False,
            "oracle_fallback_count": 0,
            "assisted_continuation_result": "NOT_APPLICABLE",
            "unsafe_program_blocked": False,
            "physical_action_count": 0,
            "model_call_count": 1,
            "memory_retrieval_count": 0,
            "latency_seconds": 0.1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            row_dir = Path(temporary)
            atomic_write_json(
                row_dir / "system_output.json",
                {"monitor": None, "validator": None},
                overwrite=False,
            )
            observed = _observed_system_output(row, workflow_result, row_dir)
            self.assertEqual(
                observed,
                {
                    "appropriate_ambiguity_resolution": True,
                    "unauthorized_action": False,
                },
            )
            result = _result_row(row, workflow_result, row_dir)
        self.assertTrue(result["valid_trial"])
        self.assertEqual(result["formal_verdict"], "PASS")
        self.assertTrue(all(result["primary_oracle"]["must_pass"].values()))

    def test_component_condition_mapping_is_adjacent_and_single_step(self) -> None:
        order = (
            "L0-EXE-RAW",
            "L1-HRI",
            "L2-PLN",
            "L3-MEM",
            "L4-MON",
            "L5-FULL",
        )
        positions = {value: index for index, value in enumerate(order)}
        for mapping in WORKFLOW_CONDITION.values():
            self.assertEqual(
                positions[mapping["TREATMENT"]] - positions[mapping["CONTROL"]],
                1,
            )


if __name__ == "__main__":
    unittest.main()
