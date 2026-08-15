from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from experiments_suite_v2.io import load_json
from experiments_suite_v2.runners.bundles import verify_row_bundle
from experiments_suite_v2.runners.pf_foundation import (
    FOUNDATION_AIMS,
    RESET_COUNT,
    _input_rows,
    run_pf_asset,
    run_pf_camera,
    run_pf_scene,
)


@unittest.skipUnless(
    os.environ.get("MUJOCO_GL") == "egl",
    "PF foundation rendering tests require MUJOCO_GL=egl",
)
class PFFoundationTests(unittest.TestCase):
    def test_registered_input_contracts_are_exact(self) -> None:
        self.assertEqual(
            FOUNDATION_AIMS, ("PF-ASSET", "PF-SCENE", "PF-CAMERA")
        )
        asset = _input_rows("PF-ASSET")
        self.assertEqual(len(asset), 1)
        self.assertEqual(asset[0]["row_id"], "PF-ASSET-PORTABLE-CLOSURE")
        for aim_id in ("PF-SCENE", "PF-CAMERA"):
            rows = _input_rows(aim_id)
            self.assertEqual(len(rows), RESET_COUNT)
            self.assertEqual(len({row["row_id"] for row in rows}), RESET_COUNT)
            self.assertTrue(
                all(row["resolution"] == [768, 768] for row in rows)
            )

    def _assert_sealed_pass(self, output: Path, result: dict) -> None:
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(verify_row_bundle(output), [])
        manifest = load_json(output / "attempt_manifest.json")
        self.assertEqual(manifest["status"], "FINALIZED")
        self.assertEqual(manifest["gate_status"], "PASS")
        self.assertEqual(manifest["agent_invocations"], 0)
        self.assertEqual(manifest["learned_service_calls"], 0)
        self.assertEqual(manifest["motion_commands"], 0)

    def test_pf_asset_executes_and_seals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "A0"
            result = run_pf_asset(output)
            self._assert_sealed_pass(output, result)
            self.assertEqual(result["published_asset_files"], 81)
            self.assertEqual(result["runtime_asset_dependencies"], 70)
            self.assertGreaterEqual(result["archived_result_scenes_compiled"], 1)
            self.assertEqual(run_pf_asset(output), result)

    def test_pf_scene_executes_twenty_resets_and_seals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "A0"
            result = run_pf_scene(output)
            self._assert_sealed_pass(output, result)
            self.assertEqual(result["reset_count"], 20)
            self.assertEqual(result["visible_object_count_per_stream"], 8)
            self.assertEqual(result["source_object_pair_count"], 28)
            self.assertEqual(result["ik_waypoint_count"], 32)
            self.assertFalse(
                result["prefmem_frame_delivery"][
                    "localhost_1234_stream_camera_required"
                ]
            )

    def test_pf_camera_executes_twenty_pairs_and_seals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "A0"
            result = run_pf_camera(output)
            self._assert_sealed_pass(output, result)
            self.assertEqual(result["sample_count"], 20)
            self.assertEqual(result["resolution"], [768, 768])
            self.assertLessEqual(
                result["maximum_rgb_depth_roundtrip_error_px"], 1e-9
            )
            self.assertFalse(
                result["prefmem_frame_delivery"][
                    "localhost_1234_stream_camera_required"
                ]
            )


if __name__ == "__main__":
    unittest.main()
