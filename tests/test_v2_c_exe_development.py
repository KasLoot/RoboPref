from __future__ import annotations

import unittest

import numpy as np

from experiments_suite_v2.runners import c_exe_development as runner
from prefmem.execution.sam import SamDetection


class CExecutionDevelopmentTests(unittest.TestCase):
    def test_registry_is_exact_160_rows_in_four_40_row_aims(self):
        rows = runner.build_rows()
        self.assertEqual(list(rows), list(runner.AIM_ORDER))
        self.assertEqual({aim: len(value) for aim, value in rows.items()}, {aim: 40 for aim in runner.AIM_ORDER})
        row_ids = [row["row_id"] for aim_rows in rows.values() for row in aim_rows]
        self.assertEqual(len(row_ids), 160)
        self.assertEqual(len(set(row_ids)), 160)

    def test_visual_rows_use_five_distinct_frozen_layouts_per_object(self):
        rows = runner.build_rows()["C-EXE-SAM"]
        for object_id in runner.OBJECTS:
            selected = [row for row in rows if row["object_id"] == object_id]
            self.assertEqual([row["position_id"] for row in selected], list(runner.VISUAL_POSITIONS))
            self.assertEqual([row["variant_index"] for row in selected], [1, 2, 3, 4, 5])
            self.assertEqual(len({row["shared_frame_id"] for row in selected}), 5)

    def test_selected_frame_evidence_builds_rgbd_and_truth(self):
        row = runner.build_rows()["C-EXE-ANCH"][0]
        frame, oracle_mask, truth, path = runner._frame(row)
        self.assertEqual(frame.rgb.shape, (768, 768, 3))
        self.assertEqual(frame.depth_m.shape, (768, 768))
        self.assertEqual(oracle_mask.shape, (768, 768))
        self.assertGreater(int(oracle_mask.sum()), 0)
        self.assertEqual(truth.shape, (3,))
        self.assertTrue(path.is_file())

    def test_compiler_contract_and_capability_fence_scoring(self):
        supported = runner.build_rows()["C-EXE-COMP"][0]
        program = runner._program("red cube", "white board")
        self.assertTrue(runner._evaluate_program(supported, program, None)["passed"])
        unsupported = next(row for row in runner.build_rows()["C-EXE-COMP"] if not row["supported"])
        self.assertTrue(runner._evaluate_program(unsupported, None, "refused")["passed"])
        self.assertFalse(runner._evaluate_program(unsupported, program, None)["passed"])

    def test_sam_selection_requires_a_clear_winner(self):
        mask = np.ones((4, 4), dtype=bool)
        one = SamDetection(object_id=0, box_xyxy=(0, 0, 4, 4), mask_area=16, mask=mask, score=0.9)
        selected, reason = runner._select_detection((one,))
        self.assertIs(selected, one)
        self.assertEqual(reason, "SAM_SELECTION_AVAILABLE")
        ambiguous = SamDetection(object_id=1, box_xyxy=(0, 0, 4, 4), mask_area=16, mask=mask, score=0.87)
        selected, reason = runner._select_detection((one, ambiguous))
        self.assertIsNone(selected)
        self.assertEqual(reason, "SAM_AMBIGUOUS_MATCHES")
        self.assertEqual(runner._select_detection(())[1], "SAM_ZERO_DETECTIONS")

    def test_move_rows_are_oracle_only_and_program_is_safe(self):
        rows = runner.build_rows()["C-EXE-MOVE"]
        self.assertTrue(all(row["grounding_source"] == "SIMULATOR_GROUND_TRUTH" for row in rows))
        self.assertEqual(len({tuple(row["source_start_world_m"]) for row in rows[:5]}), 5)
        program = runner._program("yellow cube", "cyan board")
        self.assertEqual(program.source.anchor.value, "top_center")
        self.assertEqual(program.target.anchor.value, "surface_center")
        self.assertEqual(len(program.waypoints), 6)

    def test_summary_keeps_strict_and_assisted_anchor_counts_separate(self):
        rows = []
        for index in range(40):
            success = index < 30
            rows.append({
                "formal_verdict": "PASS" if success else "FAIL",
                "reason_code": "ANCHOR_ACCURACY_PASS" if success else "SAM_ZERO_DETECTIONS",
                "sam_call_count": 1, "strict_grounding_succeeded": success,
                "anchor_errors": {"radial_xy_m": 0.001} if success else None,
                "oracle_assisted_continuation": {"available": not success},
            })
        summary = runner._summarize("C-EXE-ANCH", rows, health_pre=None, health_post=None)
        self.assertEqual(summary["strict_grounding_successes"], 30)
        self.assertEqual(summary["oracle_assisted_continuation_available"], 10)
        self.assertEqual(summary["pass_count"], 30)
        self.assertEqual(summary["capability_fail_count"], 10)

    def test_frozen_setup_has_no_side_effects(self):
        result = runner.validate_setup()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["row_counts"], {aim: 40 for aim in runner.AIM_ORDER})
        self.assertEqual(result["external_calls"], 0)
        self.assertEqual(result["physical_actions"], 0)


if __name__ == "__main__":
    unittest.main()
