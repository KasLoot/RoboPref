from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from experiments_suite_v2.runners.pf_selector import (
    CANONICAL_OBJECTS,
    CANONICAL_SIZE,
    CanonicalSelectorHarness,
    PFSelectorFixture,
    aggregate_pf_selector,
    build_pf_selector_rows,
    run_pf_selector,
    score_pf_selector_fixture,
    verify_pf_selector_bundle,
)
from prefmem.execution.contracts import ObjectReference
from prefmem.execution.frames import CameraCalibration, RGBDFrame
from prefmem.execution.grounding import RGBDGrounder
from prefmem.execution.sam import SamDetection


MASK = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=bool)
MASK[360:380, 360:380] = True


class ExactDetector:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.calls = []
        self.fail_after = fail_after

    def detect(self, rgb, prompt, *, threshold=None):
        del rgb
        self.calls.append((prompt, threshold))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise ConnectionError("simulated detector outage")
        return (
            SamDetection(
                object_id=1,
                box_xyxy=(360, 360, 380, 380),
                mask_area=int(MASK.sum()),
                mask=MASK,
                score=0.99,
            ),
        )


class FrozenDetector:
    def detect(self, *_args, **_kwargs):
        return ExactDetector().detect(None, "ignored")


def synthetic_fixture(row) -> PFSelectorFixture:
    calibration = CameraCalibration(
        width=CANONICAL_SIZE,
        height=CANONICAL_SIZE,
        intrinsic=np.array(
            [[700.0, 0.0, 383.5], [0.0, 700.0, 383.5], [0.0, 0.0, 1.0]]
        ),
        world_from_camera=np.array(
            [
                [1.0, 0.0, 0.0, 0.55],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    frame = RGBDFrame(
        rgb=np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8),
        depth_m=np.full((CANONICAL_SIZE, CANONICAL_SIZE), 0.8, dtype=np.float32),
        calibration=calibration,
        observed_at=float(row.order_index + 1),
        sequence=row.order_index + 1,
    )
    specification = next(item for item in CANONICAL_OBJECTS if item.object_id == row.object_id)
    detection = ExactDetector().detect(None, row.selector)[0]
    grounded = RGBDGrounder(FrozenDetector()).ground(
        frame, ObjectReference(row.selector, specification.anchor)
    )
    return PFSelectorFixture(
        row=row,
        frame=frame,
        oracle_mask=MASK,
        other_object_masks={
            item.object_id: np.zeros_like(MASK)
            for item in CANONICAL_OBJECTS
            if item.object_id != row.object_id
        },
        object_id_segmentation=np.zeros(
            (CANONICAL_SIZE, CANONICAL_SIZE, 2), dtype=np.int32
        ),
        expected_anchor_world_m=grounded.point_world,
        robot_hidden_pass=True,
    )


def synthetic_rows():
    rows = []
    for row in build_pf_selector_rows():
        rows.append(
            {
                **row.to_dict(),
                "unique_intended_detection": True,
                "mask_iou": 1.0,
                "cross_object_overlap_pixels": {},
                "cross_object_collision": False,
                "anchor_radial_xy_error_m": 0.0,
                "anchor_3d_error_m": 0.0,
                "robot_hidden_pass": True,
            }
        )
    return rows


class PFSelectorDesignTests(unittest.TestCase):
    def test_grid_is_exact_eight_by_five_by_four(self) -> None:
        rows = build_pf_selector_rows()
        self.assertEqual(len(rows), 160)
        self.assertEqual(len({row.row_id for row in rows}), 160)
        self.assertEqual(len({row.selector for row in rows}), 8)
        counts = Counter(row.object_id for row in rows)
        self.assertEqual(set(counts.values()), {20})
        cells = Counter(
            (row.object_id, row.camera_perturbation_index, row.seed)
            for row in rows
        )
        self.assertEqual(len(cells), 8 * 5 * 4)
        self.assertEqual(set(cells.values()), {1})
        self.assertTrue(
            all(row.to_dict()["resolution"] == [768, 768] for row in rows)
        )

    def test_oracle_detection_scores_iou_anchor_and_no_collision(self) -> None:
        row = build_pf_selector_rows()[0]
        fixture = synthetic_fixture(row)
        result, detections = score_pf_selector_fixture(fixture, ExactDetector())
        self.assertEqual(len(detections), 1)
        self.assertTrue(result["unique_intended_detection"])
        self.assertEqual(result["mask_iou"], 1.0)
        self.assertFalse(result["cross_object_collision"])
        self.assertAlmostEqual(result["anchor_3d_error_m"], 0.0)
        self.assertTrue(result["robot_hidden_pass"])

    def test_gate_requires_nineteen_of_twenty_and_zero_collisions(self) -> None:
        rows = synthetic_rows()
        summary = aggregate_pf_selector(rows)
        self.assertEqual(summary["status"], "PASS")
        failed = [dict(row) for row in rows]
        first_object = CANONICAL_OBJECTS[0].object_id
        indices = [
            index for index, row in enumerate(failed) if row["object_id"] == first_object
        ]
        failed[indices[0]]["unique_intended_detection"] = False
        self.assertEqual(aggregate_pf_selector(failed)["status"], "PASS")
        failed[indices[1]]["unique_intended_detection"] = False
        self.assertEqual(aggregate_pf_selector(failed)["status"], "FAIL")
        collision = [dict(row) for row in rows]
        collision[indices[0]]["cross_object_collision"] = True
        self.assertEqual(aggregate_pf_selector(collision)["status"], "FAIL")

    def test_one_pixel_forbidden_raster_edge_is_retained_but_not_collision(self) -> None:
        row = build_pf_selector_rows()[0]
        fixture = synthetic_fixture(row)
        intended = fixture.oracle_mask.copy()
        other = np.zeros_like(intended)
        other[360, 360:380] = True
        fixture = PFSelectorFixture(
            row=fixture.row,
            frame=fixture.frame,
            oracle_mask=fixture.oracle_mask,
            other_object_masks={
                next(
                    item.object_id
                    for item in CANONICAL_OBJECTS
                    if item.object_id != row.object_id
                ): other
            },
            object_id_segmentation=fixture.object_id_segmentation,
            expected_anchor_world_m=fixture.expected_anchor_world_m,
            robot_hidden_pass=True,
        )
        result, _ = score_pf_selector_fixture(fixture, ExactDetector())
        self.assertTrue(any(result["cross_object_overlap_pixels"].values()))
        self.assertFalse(any(result["cross_object_interior_overlap_pixels"].values()))
        self.assertFalse(result["cross_object_collision"])


class PFSelectorHarnessTests(unittest.TestCase):
    def test_real_render_uses_canonical_resolution_and_restores_camera(self) -> None:
        row = build_pf_selector_rows()[0]
        with CanonicalSelectorHarness() as harness:
            baseline = harness.model.cam_pos[harness.camera_id].copy()
            fixture = harness.capture(row)
            np.testing.assert_array_equal(
                harness.model.cam_pos[harness.camera_id], baseline
            )
        self.assertEqual(fixture.frame.rgb.shape, (768, 768, 3))
        self.assertEqual(fixture.frame.depth_m.shape, (768, 768))
        self.assertEqual(fixture.object_id_segmentation.shape, (768, 768, 2))
        self.assertTrue(fixture.robot_hidden_pass)
        self.assertEqual(len(fixture.other_object_masks), 7)


class PFSelectorDurabilityTests(unittest.TestCase):
    def test_outage_leaves_resumable_rows_without_final_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pf-selector"
            detector = ExactDetector(fail_after=2)
            with self.assertRaises(ConnectionError):
                run_pf_selector(root, detector, capture=synthetic_fixture)
            manifest = __import__("json").loads(
                (root / "attempt_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "RUNNING")
            self.assertEqual(manifest["completed_rows"], 2)
            self.assertFalse((root / "artifact_manifest.json").exists())
            retry = ExactDetector(fail_after=0)
            with self.assertRaises(ConnectionError):
                run_pf_selector(root, retry, capture=synthetic_fixture)
            self.assertEqual(retry.calls[0][0], build_pf_selector_rows()[2].selector)
            self.assertEqual(
                len((root / "result_rows.jsonl").read_text().splitlines()), 2
            )

    def test_complete_bundle_is_manifested_checksummed_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pf-selector"
            with patch(
                "experiments_suite_v2.runners.pf_selector._write_frame_artifacts",
                return_value={},
            ):
                summary = run_pf_selector(
                    root, ExactDetector(), capture=synthetic_fixture
                )
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(verify_pf_selector_bundle(root), [])
            self.assertTrue((root / "artifact_manifest.json").is_file())
            self.assertTrue((root / "checksums.sha256").is_file())
            (root / "checksums.sha256").unlink()
            resumed = run_pf_selector(root, ExactDetector(), capture=synthetic_fixture)
            self.assertEqual(resumed["status"], "PASS")
            self.assertEqual(verify_pf_selector_bundle(root), [])
            (root / "summary.json").write_text("{}\n", encoding="utf-8")
            self.assertTrue(verify_pf_selector_bundle(root))


if __name__ == "__main__":
    unittest.main()
