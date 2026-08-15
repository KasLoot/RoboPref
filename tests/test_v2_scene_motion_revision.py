from __future__ import annotations

import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from experiments_suite_v2.io import sha256_file
from experiments_suite_v2.preflight import EXPECTED_BLOCKS, SCENE_PATH
from simulation.stacking import TB6C_V00_POSITIONS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = REPOSITORY_ROOT / "experiments_suite_v2"
SYSTEM_CASES_PATH = SUITE_ROOT / "protocol" / "system_cases.json"
REVISION_PATH = SUITE_ROOT / "protocol" / "scene_motion_revision.json"
CAMERA_FREEZE_PATH = SUITE_ROOT / "protocol" / "camera_runtime_freeze.json"
CAMERA_APPROVAL_PATH = SUITE_ROOT / "protocol" / "camera_approval.json"
A1_RESULT_PATH = (
    SUITE_ROOT
    / "results"
    / "PF"
    / "PF-MOTION"
    / "evidence"
    / "prior-revisions"
    / "TB6C-v2-PF-MAT-01"
    / "A1"
    / "result.json"
)


def _vector(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split())


class SceneMotionRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.system_cases = json.loads(
            SYSTEM_CASES_PATH.read_text(encoding="utf-8")
        )
        cls.revision = json.loads(REVISION_PATH.read_text(encoding="utf-8"))
        cls.scene_root = ET.parse(SCENE_PATH).getroot()

    def test_runtime_scene_preflight_and_system_sources_are_identical(self) -> None:
        expected_by_body = {
            "red_block": (0.32, -0.34, 0.026),
            "green_block": (0.32, 0.0, 0.026),
            "blue_block": (0.32, 0.34, 0.026),
            "yellow_block": (0.745, -0.34, 0.026),
            "purple_block": (0.76, 0.0, 0.026),
            "orange_block": (0.745, 0.34, 0.026),
        }
        self.assertEqual(dict(TB6C_V00_POSITIONS), expected_by_body)
        self.assertEqual(EXPECTED_BLOCKS, expected_by_body)

        scene_bodies = {
            body.get("name"): _vector(body.attrib["pos"])
            for body in self.scene_root.findall("./worldbody/body")
            if body.get("name") in expected_by_body
        }
        self.assertEqual(scene_bodies, expected_by_body)

        body_to_cube = {
            "red_block": "red_cube",
            "green_block": "green_cube",
            "blue_block": "blue_cube",
            "yellow_block": "yellow_cube",
            "purple_block": "purple_cube",
            "orange_block": "orange_cube",
        }
        source_centres = self.system_cases["canonical_geometry"][
            "source_centres_m"
        ]
        self.assertEqual(
            {
                body_name: tuple(source_centres[cube_id])
                for body_name, cube_id in body_to_cube.items()
            },
            expected_by_body,
        )

        # V12 swaps the two outer distractors across y while retaining their
        # revised x source coordinate. Purple remains symbolic SOURCE at 0.760.
        v12 = next(
            state
            for state in self.system_cases["registered_states"]
            if state["state_id"] == "V12"
        )
        self.assertEqual(
            v12["cubes"]["yellow_cube"]["center_m"],
            [0.745, 0.34, 0.026],
        )
        self.assertEqual(v12["cubes"]["purple_cube"], "SOURCE")
        self.assertEqual(
            v12["cubes"]["orange_cube"]["center_m"],
            [0.745, -0.34, 0.026],
        )

    def test_revision_is_approved_and_records_the_exact_bounded_repair(self) -> None:
        self.assertEqual(
            self.system_cases["scene_revision"],
            {
                "revision_id": "TB6C-v2-PF-MOTION-01",
                "record": "protocol/scene_motion_revision.json",
                "status": "APPROVED_FOR_PREFLIGHT",
            },
        )
        self.assertEqual(self.revision["status"], "APPROVED_FOR_PREFLIGHT")
        self.assertFalse(self.revision["confirmatory_denominator"])
        changes = self.revision["scope"]["coordinate_changes"]
        self.assertEqual(
            [(item["object_id"], item["before_m"], item["after_m"]) for item in changes],
            [
                ("yellow_cube", 0.76, 0.745),
                ("orange_cube", 0.76, 0.745),
            ],
        )
        self.assertEqual(
            self.revision["scope"]["unchanged_source_centres_m"]["purple_cube"],
            [0.76, 0.0, 0.026],
        )
        self.assertFalse(self.revision["scope"]["camera_geometry_changed"])
        self.assertFalse(self.revision["scope"]["camera_routing_changed"])
        self.assertFalse(
            self.revision["scope"]["camera_approval_files_changed"]
        )
        self.assertEqual(
            self.revision["scene_identity"]["candidate_scene_sha256"],
            sha256_file(SCENE_PATH),
        )

    def test_revision_preserves_a1_failure_and_development_probe_evidence(self) -> None:
        a1_metadata = self.revision["development_evidence"]["pf_motion_a1"]
        self.assertEqual(a1_metadata["status"], "FAIL")
        self.assertEqual(a1_metadata["result_sha256"], sha256_file(A1_RESULT_PATH))
        a1 = json.loads(A1_RESULT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(a1["status"], "FAIL")
        envelopes = {item["fixture_id"]: item for item in a1["envelopes"]}
        self.assertFalse(envelopes["yellow_cube"]["pass"])
        self.assertTrue(envelopes["purple_cube"]["pass"])
        self.assertFalse(envelopes["orange_cube"]["pass"])
        self.assertEqual(
            envelopes["yellow_cube"]["points"][1]["position_world_m"][0],
            0.76,
        )
        self.assertEqual(
            envelopes["orange_cube"]["points"][1]["position_world_m"][0],
            0.76,
        )

        probe = self.revision["development_evidence"]["ik_boundary_probe"]
        self.assertEqual(probe["observed_boundary_x_m"], 0.75)
        self.assertEqual(probe["selected_source_x_m"], 0.745)
        self.assertEqual(probe["selected_margin_inside_boundary_m"], 0.005)
        self.assertEqual(probe["purple_source_x_m"], 0.76)
        selector = self.revision["development_evidence"][
            "selector_regression"
        ]
        self.assertEqual(
            selector["execution_mode"], "IN_MEMORY_SCENE_WITH_LIVE_SAM"
        )
        self.assertEqual((selector["passed_cells"], selector["total_cells"]), (40, 40))
        self.assertEqual(selector["live_model_calls"], 40)

    def test_new_view_approval_preserves_prior_hash_and_camera_geometry(self) -> None:
        prior = self.revision["prior_camera_approval_reference"]
        self.assertEqual(
            prior["camera_approval_sha256"],
            "64f3319b919e60836677ecca0daefcee8f43a0716856027e3fa869165ef96286",
        )
        self.assertNotEqual(
            sha256_file(CAMERA_APPROVAL_PATH),
            prior["camera_approval_sha256"],
        )
        self.assertEqual(
            sha256_file(CAMERA_FREEZE_PATH),
            prior["camera_runtime_freeze_sha256"],
        )
        approval = json.loads(CAMERA_APPROVAL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            sha256_file(CAMERA_APPROVAL_PATH),
            "8ce65a150084304b6167e55f8c3b95be5586baa3d8307eeed26f50276e5deec2",
        )
        self.assertEqual(approval["approval_status"], "APPROVED")
        self.assertEqual(approval["preview_scene_state"], "V00_PF_MOTION_01")
        expected_preview_hashes = {
            "sam_top_down": "dbff8a97eca5b9744ce9058d5e51ba1aa7084e8300b23d9dcb065f191e60aeeb",
            "prefmem_third_person": "e35375d63ea8d69f1106f5c4b9210a381a97ebda8ea81f139135d6740d8bf71a",
        }
        self.assertEqual(
            approval["approved_preview_hashes"], expected_preview_hashes
        )
        revision_approval = self.revision["approval_record"]
        self.assertEqual(revision_approval["authority"], "user")
        self.assertEqual(
            revision_approval["scene_sha256"], sha256_file(SCENE_PATH)
        )
        self.assertEqual(
            revision_approval["sam_preview_sha256"],
            expected_preview_hashes["sam_top_down"],
        )
        self.assertEqual(
            revision_approval["prefmem_preview_sha256"],
            expected_preview_hashes["prefmem_third_person"],
        )
        freeze = json.loads(CAMERA_FREEZE_PATH.read_text(encoding="utf-8"))
        cameras = {
            camera.get("name"): camera
            for camera in self.scene_root.findall("./worldbody/camera")
        }
        for stream in freeze["streams"].values():
            camera = cameras[stream["camera_name"]]
            self.assertEqual(_vector(camera.attrib["pos"]), tuple(stream["position_world_m"]))
            self.assertEqual(_vector(camera.attrib["xyaxes"]), tuple(stream["xyaxes"]))
            self.assertEqual(float(camera.attrib["fovy"]), stream["fovy_degrees"])


if __name__ == "__main__":
    unittest.main()
