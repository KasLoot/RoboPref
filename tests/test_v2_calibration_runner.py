from __future__ import annotations

from collections import Counter
import json
import unittest

import numpy as np

from experiments_suite_v2.runners.calibration import (
    CAL_X01_HEIGHTS_M,
    CAL_X01_X_COORDINATES_M,
    CAL_X01_Y_COORDINATES_M,
    CAL_X_OBJECTS,
    CAL_X_REGISTERED_POSITIONS_M,
    CameraCalibrationHarness,
    ObjectCalibrationHarness,
    aggregate_cal_x01,
    aggregate_cal_x03,
    aggregate_oracle_anchor_gate,
    build_cal_x01_rows,
    build_cal_x02_rows,
    build_cal_x03_rows,
    evaluate_cal_x02_row,
    evaluate_cal_x03_row,
)


class FixedSam:
    def __init__(self) -> None:
        self.detection = None
        self.prompts: list[str] = []

    def detect(self, rgb, prompt, *, threshold=None):
        del rgb, threshold
        self.prompts.append(prompt)
        if self.detection is None:
            raise AssertionError("test did not install a detection")
        return (self.detection,)


class CalibrationGridTests(unittest.TestCase):
    def test_cal_x01_generator_is_exactly_375_rows(self) -> None:
        rows = build_cal_x01_rows()
        self.assertEqual(len(rows), 375)
        self.assertEqual(len({row.row_id for row in rows}), 375)
        self.assertEqual(
            {
                row.point_world_m[0]
                for row in rows
            },
            set(CAL_X01_X_COORDINATES_M),
        )
        self.assertEqual(
            {row.point_world_m[1] for row in rows},
            set(CAL_X01_Y_COORDINATES_M),
        )
        self.assertEqual(
            {row.point_world_m[2] for row in rows},
            set(CAL_X01_HEIGHTS_M),
        )
        counts = Counter(
            (row.x_index, row.y_index, row.height_index) for row in rows
        )
        self.assertEqual(set(counts.values()), {5})
        self.assertEqual([row.order_index for row in rows], list(range(375)))

    def test_cal_x02_and_x03_are_exact_frame_paired_520_row_grids(self) -> None:
        oracle = build_cal_x02_rows()
        sam = build_cal_x03_rows()
        self.assertEqual(len(oracle), 520)
        self.assertEqual(len(sam), 520)
        self.assertEqual(len({row.row_id for row in oracle}), 520)
        self.assertEqual(len({row.shared_frame_id for row in oracle}), 520)
        self.assertTrue(
            all(
                left.shared_frame_id == right.shared_frame_id
                for left, right in zip(oracle, sam, strict=True)
            )
        )
        self.assertEqual({row.object_id for row in oracle}, {
            item.object_id for item in CAL_X_OBJECTS
        })
        self.assertEqual(
            {row.position_id for row in oracle},
            {position_id for position_id, _point in CAL_X_REGISTERED_POSITIONS_M},
        )
        counts = Counter((row.object_id, row.position_id) for row in oracle)
        self.assertEqual(len(counts), 8 * 13)
        self.assertEqual(set(counts.values()), {5})
        selectors = {item.object_id: item.selector for item in CAL_X_OBJECTS}
        self.assertTrue(
            all(row.selector == selectors[row.object_id] for row in oracle)
        )
        json.dumps([oracle[0].to_dict(), sam[-1].to_dict()], allow_nan=False)


class CameraRoundTripTests(unittest.TestCase):
    def test_analytic_and_rendered_metric_depth_preserve_axis_convention(self) -> None:
        rows = build_cal_x01_rows()
        selected = (rows[0], rows[len(rows) // 2], rows[-1])
        with CameraCalibrationHarness(width=256, height=256) as harness:
            evaluations = tuple(harness.evaluate(row) for row in selected)
        for evaluation in evaluations:
            with self.subTest(row=evaluation.row.row_id):
                analytic = np.asarray(
                    evaluation.scoring["analytic_roundtrip_error_mm"]
                )
                rendered = np.asarray(
                    evaluation.scoring["rendered_roundtrip_error_mm"]
                )
                self.assertLess(float(np.max(np.abs(analytic))), 1e-9)
                self.assertLess(float(np.max(np.abs(rendered[:2]))), 2.0)
                self.assertLess(abs(float(rendered[2])), 0.01)
                self.assertTrue(evaluation.scoring["axis_sign_pass"])
                json.dumps(evaluation.result_row(), allow_nan=False)
        aggregate = aggregate_cal_x01(evaluations)
        self.assertEqual(aggregate["row_count"], 3)
        self.assertEqual(aggregate["status"], "PASS")


class OracleAndSamCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.oracle_rows = build_cal_x02_rows()
        cls.sam_rows = build_cal_x03_rows()
        cls.harness = ObjectCalibrationHarness(width=256, height=256)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.harness.close()

    def _paired_rows(self, object_id: str, position_id: str):
        index = next(
            index
            for index, row in enumerate(self.oracle_rows)
            if row.object_id == object_id
            and row.position_id == position_id
            and row.repeat_index == 0
        )
        return self.oracle_rows[index], self.sam_rows[index]

    def test_oracle_masks_use_production_grounder_for_cube_and_board(self) -> None:
        evaluations = []
        for object_id in ("red_cube", "white_board"):
            oracle_row, _sam_row = self._paired_rows(object_id, "P08_X_POS_Y_POS")
            observation = self.harness.capture(oracle_row)
            # Hidden truth belongs only to the harness wrapper, never the
            # production frame submitted to RGBDGrounder/runtime consumers.
            self.assertFalse(
                hasattr(observation.frame, "hidden_true_anchor_world_m")
            )
            evaluation = evaluate_cal_x02_row(oracle_row, observation)
            evaluations.append(evaluation)
            error = np.asarray(evaluation.scoring["oracle_anchor_error_mm"])
            self.assertLess(float(np.max(np.abs(error[:2]))), 2.0)
            self.assertEqual(
                evaluation.telemetry["oracle_mask"]["area_pixels"],
                int(observation.oracle_mask.sum()),
            )
            self.assertEqual(
                evaluation.telemetry["grounding"]["mask_source"],
                "sam_mask",
            )
            json.dumps(evaluation.result_row(), allow_nan=False)
        gate = aggregate_oracle_anchor_gate(evaluations)
        self.assertEqual(gate["row_count"], 2)
        self.assertEqual(gate["status"], "PASS")

    def test_deterministic_exact_sam_has_unit_iou_and_zero_increment(self) -> None:
        detector = FixedSam()
        evaluations = []
        expected_prompts = []
        for object_id in ("purple_cube", "cyan_board"):
            oracle_row, sam_row = self._paired_rows(object_id, "P00_CENTER")
            observation = self.harness.capture(oracle_row)
            detector.detection = observation.oracle_detection
            evaluation = evaluate_cal_x03_row(sam_row, observation, detector)
            evaluations.append(evaluation)
            expected_prompts.append(sam_row.selector)
            self.assertEqual(evaluation.scoring["mask_iou"], 1.0)
            self.assertEqual(
                evaluation.scoring["centroid_error_uv_pixels"],
                [0.0, 0.0],
            )
            np.testing.assert_array_equal(
                evaluation.scoring["incremental_sam_anchor_error_mm"],
                [0.0, 0.0, 0.0],
            )
            self.assertEqual(
                evaluation.telemetry["shared_frame_id"],
                oracle_row.shared_frame_id,
            )
            json.dumps(evaluation.result_row(), allow_nan=False)
        self.assertEqual(detector.prompts, expected_prompts)
        aggregate = aggregate_cal_x03(evaluations)
        self.assertEqual(aggregate["row_count"], 2)
        self.assertEqual(aggregate["mask_iou"]["minimum"], 1.0)

    def test_oracle_gate_uses_frozen_inclusive_axis_limits(self) -> None:
        passing = [
            {"oracle_anchor_error_mm": [2.0, -2.0, 20.0]},
            {"oracle_anchor_error_mm": [-2.0, 2.0, -20.0]},
            {"oracle_anchor_error_mm": [0.0, 0.0, 0.0]},
        ]
        self.assertEqual(aggregate_oracle_anchor_gate(passing)["status"], "PASS")
        failing_median = [
            {"oracle_anchor_error_mm": [2.01, 0.0, 0.0]},
            {"oracle_anchor_error_mm": [2.01, 0.0, 0.0]},
        ]
        self.assertEqual(
            aggregate_oracle_anchor_gate(failing_median)["status"],
            "FAIL",
        )
        failing_tail = [
            {"oracle_anchor_error_mm": [0.0, 0.0, 0.0]}
            for _ in range(18)
        ] + [
            {"oracle_anchor_error_mm": [6.0, 0.0, 0.0]},
            {"oracle_anchor_error_mm": [6.0, 0.0, 0.0]},
        ]
        self.assertEqual(
            aggregate_oracle_anchor_gate(failing_tail)["status"],
            "FAIL",
        )


if __name__ == "__main__":
    unittest.main()
