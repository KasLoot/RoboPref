from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from agents.contracts import ExecutionResult
from simulation.benchmark.catalog import build_catalog
from simulation.benchmark.generator import generate_benchmark
from simulation.benchmark.mujoco_render import _build_model, _render
from simulation.benchmark.validator import validate_benchmark


MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None


class BenchmarkVisualOracleRegressionTests(unittest.TestCase):
    def test_execution_evidence_id_is_pixel_derived_not_path_derived(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "success-private-label.png"
            second = root / "failure-private-label.png"
            first.write_bytes(b"same-observation-bytes")
            second.write_bytes(first.read_bytes())
            first_model = ExecutionResult(
                status="OBSERVED_RECORDED_ATTEMPT",
                final_observation=str(first),
            ).to_model_dict()
            second_model = ExecutionResult(
                status="OBSERVED_RECORDED_ATTEMPT",
                final_observation=str(second),
            ).to_model_dict()
            self.assertEqual(
                first_model["final_observation_id"],
                second_model["final_observation_id"],
            )
            serialized = json.dumps([first_model, second_model])
            self.assertNotIn("success-private-label", serialized)
            self.assertNotIn("failure-private-label", serialized)

    def test_validator_rejects_render_noise_as_visual_distinction_with_new_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_benchmark(
                root,
                families=["block_stack"],
                seeds=[17],
                backend="synthetic",
            )
            packets: dict[tuple[str, str, int, str], dict[str, Path]] = {}
            for manifest_path in root.rglob("manifest.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                key = (
                    str(manifest["scene"]["family"]),
                    str(manifest["scene"]["variant"]),
                    int(manifest["scene"]["seed"]),
                    str(manifest["target"]["target_id"]),
                )
                packets.setdefault(key, {})[
                    str(manifest["expected_outcome"])
                ] = manifest_path
            siblings = next(
                outcomes
                for outcomes in packets.values()
                if {"success", "near_miss"} <= outcomes.keys()
            )
            success_final = siblings["success"].parent / "2.png"
            near_manifest_path = siblings["near_miss"]
            near_final = near_manifest_path.parent / "2.png"
            near_final.write_bytes(success_final.read_bytes())
            with Image.open(near_final) as source:
                noisy = source.convert("RGB")
            pixels = noisy.load()
            for index in range(200):
                x = index * 37 % noisy.width
                y = index * 53 % noisy.height
                red, green, blue = pixels[x, y]
                pixels[x, y] = (
                    red + 1 if red < 255 else red - 1,
                    green,
                    blue,
                )
            noisy.save(near_final, format="PNG", optimize=False)
            near_manifest = json.loads(
                near_manifest_path.read_text(encoding="utf-8")
            )
            near_manifest["frame_sha256"]["2.png"] = hashlib.sha256(
                near_final.read_bytes()
            ).hexdigest()
            near_manifest_path.write_text(
                json.dumps(near_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = validate_benchmark(root)
            self.assertFalse(report.valid)
            self.assertTrue(
                any("pixel-identical" in error for error in report.errors),
                report.errors,
            )

    @unittest.skipUnless(MUJOCO_AVAILABLE, "optional mujoco dependency is absent")
    def test_mujoco_endpoints_and_category_semantic_geoms_differ(self) -> None:
        place_scenarios = build_catalog(families=["place_setting"], seeds=[19])
        success = next(
            scenario
            for scenario in place_scenarios
            if scenario.scene_variant == "front_scatter"
            and scenario.target_id == "right_handed"
            and scenario.outcome == "success"
        )
        near_miss = next(
            scenario
            for scenario in place_scenarios
            if scenario.scene_variant == success.scene_variant
            and scenario.target_id == success.target_id
            and scenario.outcome == "near_miss"
        )
        self.assertNotEqual(
            _render(success, success.final_objects, occluded=False).tobytes(),
            _render(near_miss, near_miss.final_objects, occluded=False).tobytes(),
        )
        category = next(
            scenario
            for scenario in build_catalog(families=["category_sort"], seeds=[19])
            if scenario.outcome == "success"
        )
        model = _build_model(category, category.final_objects)
        import mujoco

        geom_names = {
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            for geom_id in range(model.ngeom)
        }
        objects = {item.object_type: item for item in category.final_objects}
        self.assertIn(f"{objects['laptop'].object_id}_screen", geom_names)
        self.assertIn(f"{objects['magazine'].object_id}_pages", geom_names)


if __name__ == "__main__":
    unittest.main()
